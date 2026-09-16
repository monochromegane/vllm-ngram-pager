"""SSD tier of the n-gram embedding table.

The PLE layer of Qwen4Exp looks up 16 rows per token (2-grams and 3-grams, 8 heads
each) by hash. The table has about 20 million rows per head, 320,001,536 rows x 160 B
of fp8 in total (47.7 GiB), which fits in neither VRAM nor host RAM.

The safetensors shards of the checkpoint (128 of them, each (2,500,012, 160) fp8) serve
as the table as they are, and rows are read through ``np.memmap``. No separate paging
file is written. The RAM tier is left to the OS page cache; the plugin keeps none of
its own.
"""

import json
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Shard:
    path: str
    offset: int  # bytes from the start of the file to the tensor data
    rows: int
    cols: int


def read_header(path: str) -> tuple[dict, int]:
    """Return the safetensors header (JSON) and the offset where the data begins."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def find_shards(files: Iterable[str], name: str) -> list[Shard]:
    """Return the shards that make up the n-gram table ``name``, in shard order.

    ``name`` is the layer name from "layers." onwards (for example
    ``layers.1.ple.ple_embedding.ngram_embedding``). The prefix used in the checkpoint
    (``model.language_model.``) differs from vLLM's names, so match on the suffix.
    """
    pattern = re.compile(r"(?:^|\.)" + re.escape(name) + r"\.shard_(\d+)\.weight$")
    files = list(files)
    found: dict[int, Shard] = {}
    for path in files:
        header, base = read_header(path)
        for key, info in header.items():
            m = pattern.search(key)
            if m is None:
                continue
            k = int(m.group(1))
            if k in found:
                raise ValueError(
                    f"shard {k} of {name} appears twice: {found[k].path}, {path}"
                )
            if info["dtype"] != "F8_E4M3":
                raise ValueError(f"{key}: expected F8_E4M3, got {info['dtype']}")
            rows, cols = info["shape"]
            begin, end = info["data_offsets"]
            if end - begin != rows * cols:
                raise ValueError(f"{key}: {end - begin} B for shape ({rows}, {cols})")
            found[k] = Shard(path, base + begin, rows, cols)
    if not found:
        raise FileNotFoundError(f"no shard of {name} in {len(files)} files")
    if sorted(found) != list(range(len(found))):
        raise ValueError(f"shards of {name} are not contiguous: {sorted(found)}")
    return [found[k] for k in range(len(found))]


class NGramTable:
    """The shards laid end to end as a (num_rows, cols) table.

    Row r lives in shard ``r // shard_rows``.
    """

    def __init__(self, shards: list[Shard]) -> None:
        self.shard_rows = shards[0].rows
        self.cols = shards[0].cols
        for k, s in enumerate(shards):
            last = k == len(shards) - 1
            if s.cols != self.cols or not (
                0 < s.rows <= self.shard_rows if last else s.rows == self.shard_rows
            ):
                raise ValueError(
                    f"shard {k} is ({s.rows}, {s.cols}); shard 0 is "
                    f"({self.shard_rows}, {self.cols})"
                )
        self.num_rows = sum(s.rows for s in shards)
        self.maps = [
            np.memmap(
                s.path,
                dtype=np.uint8,
                mode="r",
                offset=s.offset,
                shape=(s.rows, s.cols),
            )
            for s in shards
        ]

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Return the rows numbered ``ids`` (any shape) as ``(*ids.shape, cols)`` uint8."""
        flat = ids.reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_rows):
            raise IndexError(
                f"row ids must be in [0, {self.num_rows}), got "
                f"[{flat.min()}, {flat.max()}]"
            )
        out = np.empty((flat.size, self.cols), dtype=np.uint8)
        shard = flat // self.shard_rows
        local = flat - shard * self.shard_rows
        for k in np.unique(shard):
            m = shard == k
            out[m] = self.maps[k][local[m]]
        return out.reshape(*ids.shape, self.cols)

    def lookup(
        self, ids: torch.Tensor, dtype: torch.dtype, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the rows of ``ids`` as ``dtype`` of shape ``(*ids.shape, cols)``.

        The result is on the device of ``ids``. When ``out`` is given, write into it
        and return it (the eager section of a CUDA graph must write into a fixed
        buffer). Copying ``ids`` to the host synchronizes with the device.
        """
        rows = torch.from_numpy(self.gather(ids.cpu().numpy()))
        if out is None:
            return rows.to(ids.device).view(dtype)
        out.view(torch.uint8).copy_(rows)
        return out
