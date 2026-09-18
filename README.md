# vllm-ngram-pager

vllm-ngram-pager is a vLLM plugin for running Qwen3.8-Flash-Next (`Qwen4Exp`),
whose n-gram embedding table does not fit in GPU memory. The plugin leaves the
table in the checkpoint files, maps them into memory, and gathers only the rows
each step needs from the OS page cache or the SSD. Together with
[vllm-expert-pager](https://github.com/monochromegane/vllm-expert-pager),
which pages the expert weights, it runs `Qwen/Qwen3.8-Flash-Next-FP8` on a
single RTX 4090 (24 GB).

## Requirements

- A vLLM that contains vllm-project/vllm#55341 (`685074cd61`, 2026-09-05).
  vLLM 0.29.0, branched on 2026-08-31, does not.
- The plugin also hooks vLLM internals (`PLEVocabParallelEmbedding` and the FP8
  PLE embedding method), so versions that move them do not work.
- `Qwen/Qwen3.8-Flash-Next-FP8`. The n-gram table must be stored as
  `F8_E4M3`.
- The checkpoint on a local SSD. The table is read from the checkpoint files
  with random 4 KiB reads, so an NVMe SSD is strongly recommended.
- A single GPU. Tensor parallelism is not supported.
- Breakable CUDA graphs in `PIECEWISE` mode, or `--enforce-eager`. The lookup
  reads the n-gram ids on the host, so it cannot be captured into a CUDA
  graph; with breakable CUDA graphs it runs as an eager break and the rest of
  the model is replayed. vLLM's default `FULL_AND_PIECEWISE` mode captures
  decode as a full graph, which cannot break, and startup fails at capture.
- Linux. WSL2 works, but needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1` there: the
  model runner requires pinned host memory, which vLLM disables on WSL2 by
  default.
- Python 3.10 or later.

## Usage

Install into the environment where vLLM is installed:

```bash
pip install git+https://github.com/monochromegane/vllm-ngram-pager
```

vLLM discovers the plugin through the `vllm.general_plugins` entry point.
Set `VLLM_PLUGINS` explicitly so that only the plugins you want load (when
the variable is unset, vLLM loads every plugin it finds). vLLM does not
enable breakable CUDA graphs for `Qwen4Exp` by default, so set
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` as well:

```bash
VLLM_PLUGINS=ngram_pager \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --max-model-len 4096 --max-num-seqs 1
```

The plugin has no settings. It uses no VRAM for the table and pins no host
memory; the RAM tier is whatever the page cache can hold.

On its own the plugin only removes the n-gram table from VRAM; the expert
weights still have to fit. To run the model on a 24 GB GPU, combine it with
vllm-expert-pager:

```bash
VLLM_PLUGINS=expert_pager,ngram_pager \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
VLLM_EXPERT_PAGER_CACHE_SLOTS=46 \
VLLM_EXPERT_PAGER_RAM_SLOTS=232 \
VLLM_EXPERT_PAGER_PITCH=0.88 \
VLLM_EXPERT_PAGER_SSD_PATH=/path/to/expert_pager.bin \
vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --language-model-only \
  --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --max-model-len 4096 --max-num-seqs 1
```

`--language-model-only` skips the vision encoder to save VRAM. The
vllm-expert-pager settings above are the ones used on an RTX 4090 with 64 GB
of host RAM (about 54 GB of it visible to WSL2); see its README for how to
choose them. They pin 45.4 GiB, which is as far as that machine goes. Host RAM
that vllm-expert-pager does not pin serves as the page cache for the n-gram
table.

`PITCH=0.88` is worth setting here: every one of this model's 24,576 experts
compresses to within that ratio, and the 0.89 vllm-expert-pager defaults to —
which a different model needs — would pin half a GiB more on a machine where
host RAM is what runs out first.

## Results

Measured 2026-09-17 on an RTX 4090 (24 GB) under WSL2 with
`0.28.1rc1.dev583+g114abd1c1` and `Qwen/Qwen3.8-Flash-Next-FP8`, started as
above. The n-gram table is 320,001,536 rows x 160 B (47.7 GiB) over 128 shards,
none of it in VRAM. The client is `vllm bench serve` with one concurrent
request, the `random` dataset at 1024 input and 256 output tokens, 4 prompts,
`--ignore-eos --temperature 0 --seed 0`; the second of two warm runs is
reported.

| vllm-expert-pager setting | Pinned RAM | SSD | TPOT | TTFT | Output tok/s |
|---|---|---|---|---|---|
| `CACHE_SLOTS=46`, `RAM_SLOTS=232` | 45.4 GiB | 99.0 GiB | 60.7 ms | 4978 ms | 12.5 |
| `CACHE_SLOTS=38`, `RAM_SLOTS=192` | 37.6 GiB | 99.0 GiB | 67.3 ms | 5551 ms | 11.3 |
| `CACHE_SLOTS=38`, `RAM_SLOTS=192`, `COMPRESS=off` | 42.8 GiB | 112.5 GiB | 67.4 ms | 6214 ms | 10.9 |

## License

MIT

## Author

[monochromegane](https://github.com/monochromegane)
