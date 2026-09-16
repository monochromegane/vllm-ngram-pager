"""Tests for table.py: write small safetensors files, then find shards and gather rows."""

import json
import mmap
import struct

import numpy as np
import pytest
import torch

from vllm_ngram_pager.table import NGramTable, find_shards, read_header

NAME = "layers.1.ple.ple_embedding.ngram_embedding"
COLS = 4


def write_safetensors(
    path, tensors: dict[str, np.ndarray], dtype: str = "F8_E4M3"
) -> None:
    header, offset = {}, 0
    for name, t in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + t.nbytes],
        }
        offset += t.nbytes
    body = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        for t in tensors.values():
            f.write(t.tobytes())


def rows(start: int, n: int) -> np.ndarray:
    return (np.arange(start, start + n)[:, None] * 10 + np.arange(COLS)).astype(
        np.uint8
    )


@pytest.fixture
def files(tmp_path):
    # Shards 0 and 2 share a file, shard 1 is in another, and the last shard is short.
    # Shards of another layer, and of a layer whose number shares a prefix (layers.11),
    # are ignored.
    a, b = tmp_path / "model-00001.safetensors", tmp_path / "model-00002.safetensors"
    write_safetensors(
        a,
        {
            f"model.language_model.{NAME}.shard_2.weight": rows(10, 3),
            "model.language_model.layers.11.ple.ple_embedding.ngram_embedding.shard_0.weight": rows(
                90, 5
            ),
            f"model.language_model.{NAME}.shard_0.weight": rows(0, 5),
        },
    )
    write_safetensors(
        b,
        {
            f"model.language_model.{NAME}.weight_scale": np.zeros(1, np.uint8),
            f"model.language_model.{NAME}.shard_1.weight": rows(5, 5),
        },
    )
    return [str(a), str(b)]


def test_find_shards_orders_by_index_and_reads_offsets(files):
    shards = find_shards(files, NAME)
    assert [s.rows for s in shards] == [5, 5, 3]
    assert all(s.cols == COLS for s in shards)
    assert [s.path for s in shards] == [files[0], files[1], files[0]]
    header, base = read_header(files[0])
    key = f"model.language_model.{NAME}.shard_0.weight"
    assert shards[0].offset == base + header[key]["data_offsets"][0]


def test_find_shards_rejects_gaps_and_wrong_dtype(tmp_path, files):
    with pytest.raises(FileNotFoundError):
        find_shards(files, "layers.7.ple.ple_embedding.ngram_embedding")
    with pytest.raises(ValueError, match="not contiguous"):
        find_shards(files[:1], NAME)
    bf16 = tmp_path / "bf16.safetensors"
    write_safetensors(bf16, {f"x.{NAME}.shard_0.weight": rows(0, 5)}, dtype="BF16")
    with pytest.raises(ValueError, match="F8_E4M3"):
        find_shards([str(bf16)], NAME)


def test_gather_reads_rows_across_shards(files):
    table = NGramTable(find_shards(files, NAME))
    assert (table.num_rows, table.cols, table.shard_rows) == (13, COLS, 5)
    assert np.array_equal(table.gather(np.arange(13)), rows(0, 13))
    ids = np.array([[12, 0, 5], [4, 9, 10]])
    out = table.gather(ids)
    assert out.shape == (2, 3, COLS)
    assert np.array_equal(out, rows(0, 13)[ids])
    assert table.gather(np.zeros((0,), np.int64)).shape == (0, COLS)
    for bad in ([13], [-1]):
        with pytest.raises(IndexError):
            table.gather(np.array(bad))


def test_ngram_table_rejects_uneven_shards(files):
    shards = find_shards(files, NAME)
    with pytest.raises(ValueError):
        NGramTable([shards[0], shards[2], shards[1]])


def test_lookup_returns_fp8_rows_on_the_ids_device(files):
    table = NGramTable(find_shards(files, NAME))
    ids = torch.tensor([[12, 0], [5, 4]], dtype=torch.int64)
    out = table.lookup(ids, torch.float8_e4m3fn)
    assert out.shape == (2, 2, COLS)
    assert out.dtype == torch.float8_e4m3fn
    assert out.device == ids.device
    assert np.array_equal(out.view(torch.uint8).numpy(), rows(0, 13)[ids.numpy()])


def test_lookup_writes_into_the_given_buffer(files):
    # The eager section of a CUDA graph writes into the caller's fixed buffer
    # (embedding._ple_lookup).
    table = NGramTable(find_shards(files, NAME))
    ids = torch.tensor([[12, 0], [5, 4]], dtype=torch.int64)
    out = torch.empty((2, 2, COLS), dtype=torch.float8_e4m3fn)
    assert table.lookup(ids, torch.float8_e4m3fn, out=out) is out
    assert np.array_equal(out.view(torch.uint8).numpy(), rows(0, 13)[ids.numpy()])


def test_gather_advises_page_aligned_ranges_covering_each_row(files):
    # Each MADV_WILLNEED range must start on a page boundary and cover every byte of
    # the row. np.memmap rounds the offset down to a page boundary, so the array starts
    # partway into the mmap.
    shards = find_shards(files, NAME)
    table = NGramTable(shards)
    calls: list[list[tuple[int, int, int]]] = [[] for _ in shards]
    for k, m in enumerate(table.maps):
        m._mmap = type(
            "Recorder", (), {"madvise": lambda self, *a, _k=k: calls[_k].append(a)}
        )()
    ids = np.array([12, 0, 5, 4, 9])
    assert np.array_equal(table.gather(ids), rows(0, 13)[ids])
    for k, s in enumerate(shards):
        mmap_start = s.offset - s.offset % mmap.ALLOCATIONGRANULARITY
        expected_rows = [
            r - k * table.shard_rows for r in ids if r // table.shard_rows == k
        ]
        assert len(calls[k]) == len(expected_rows)
        for (option, start, length), r in zip(calls[k], expected_rows):
            row_start = s.offset - mmap_start + r * COLS
            assert option == mmap.MADV_WILLNEED
            assert start % mmap.PAGESIZE == 0
            assert start <= row_start and row_start + COLS <= start + length
