"""Hook ``PLEVocabParallelEmbedding`` and read the n-gram table from the checkpoint.

- Construction: ``create_weights`` would allocate the (V, 160) fp8 table (47.7 GiB);
  allocate a one-row placeholder instead
- Loading: ``weight_loader`` only validates the shape of each shard and reads no data
- After loading: find the safetensors of the checkpoint and open the table (table.py)
- Inference: ``forward`` copies the ngram ids to the host, gathers the rows from the
  table, and moves them back to the device (eager only)
"""

import glob
import os

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.models.qwen4_exp.common.ple import PLEVocabParallelEmbedding
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLEFp8EmbeddingMethod

from vllm_ngram_pager.table import NGramTable, find_shards

# vLLM configures handlers and levels only for the "vllm" logger, so use a name below it.
logger = init_logger(f"vllm.{__name__}")


class _Fp8Method(Qwen4ExpPLEFp8EmbeddingMethod):
    """Variant that keeps no rows in VRAM.

    ``weight`` is a one-row placeholder; the scale is left as it is.
    """

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes, *args, **kwargs
    ) -> None:
        super().create_weights(layer, input_size_per_partition, [1], *args, **kwargs)

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        layer._ple_open_table()


class PagedPLEEmbedding(PLEVocabParallelEmbedding):
    def __init__(self, *args, prefix: str = "", quant_method=None, **kwargs) -> None:
        if not isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            raise TypeError(
                f"{prefix}: vllm-ngram-pager only supports FP8 PLE checkpoints, got "
                f"{type(quant_method).__name__}"
            )
        if "layers." not in prefix:
            raise RuntimeError(f"{prefix}: expected a layer name containing 'layers.'")
        super().__init__(*args, prefix=prefix, quant_method=_Fp8Method(), **kwargs)
        if self.tp_size != 1:
            raise RuntimeError(
                f"{prefix}: vllm-ngram-pager does not support tensor parallelism"
            )
        # The checkpoint uses a different prefix, so match on the part from "layers."
        # onwards (table.find_shards).
        self._ple_name = prefix[prefix.index("layers.") :]
        # There is no config context when weight_loader runs, so record it here.
        config = get_current_vllm_config()
        self._ple_model = config.model_config.model
        self._ple_revision = config.model_config.revision
        self._ple_download_dir = config.load_config.download_dir
        # Shards seen by weight_loader: checkpoint_start -> shape. Checked against the
        # files when the table is opened.
        self._ple_seen: dict[int, tuple[int, ...]] = {}
        self._ple_table: NGramTable | None = None

    # ---- loading ----

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        if param is not self.weight:
            return super().weight_loader(param, loaded_weight, checkpoint_start)
        if checkpoint_start is None:
            raise RuntimeError(f"{self._ple_name}: expected checkpoint shards")
        if loaded_weight.dtype != self.weight.dtype:
            raise RuntimeError(
                f"{self._ple_name}: shard at {checkpoint_start} is {loaded_weight.dtype}, "
                f"expected {self.weight.dtype}"
            )
        # Read no data. The table mmaps the checkpoint files directly (_ple_open_table).
        self._ple_seen[checkpoint_start] = tuple(loaded_weight.shape)

    def _ple_open_table(self) -> None:
        files = _checkpoint_files(
            self._ple_model, self._ple_revision, self._ple_download_dir
        )
        shards = find_shards(files, self._ple_name)
        table = NGramTable(shards)
        expected = {
            k * table.shard_rows: (s.rows, s.cols) for k, s in enumerate(shards)
        }
        if expected != self._ple_seen:
            raise RuntimeError(
                f"{self._ple_name}: shards in the checkpoint files do not match the "
                f"shards vLLM loaded: files {expected}, loaded {self._ple_seen}"
            )
        if (table.num_rows, table.cols) != (self.org_vocab_size, self.embedding_dim):
            raise RuntimeError(
                f"{self._ple_name}: table is ({table.num_rows}, {table.cols}), layer is "
                f"({self.org_vocab_size}, {self.embedding_dim})"
            )
        self._ple_table = table
        logger.info(
            "vllm-ngram-pager: %s: %d rows x %d B (%.1f GiB) in %d shards, read via mmap",
            self._ple_name,
            table.num_rows,
            table.cols,
            table.num_rows * table.cols / 2**30,
            len(shards),
        )

    # ---- forward ----

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "vllm-ngram-pager: the n-gram lookup reads ids on the host and cannot be "
                "captured into a CUDA graph; run with --enforce-eager"
            )
        assert self._ple_table is not None
        return self._ple_table.lookup(input_, self.weight.dtype)


def _checkpoint_files(
    model: str, revision: str | None, download_dir: str | None
) -> list[str]:
    """Resolve the checkpoint folder as vLLM's loader does and list its safetensors."""
    folder = (
        model
        if os.path.isdir(model)
        else download_weights_from_hf(model, download_dir, ["*.safetensors"], revision)
    )
    return sorted(glob.glob(os.path.join(folder, "*.safetensors")))


def register() -> None:
    PLEVocabParallelEmbedding.register_oot(PagedPLEEmbedding)
