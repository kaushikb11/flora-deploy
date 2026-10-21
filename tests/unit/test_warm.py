"""Parallel prefetch helpers in ``modal_app/warm.py`` (no Modal, no NVIDIA code)."""

import threading
import time

import pytest

from flora_deploy.modal_app import warm


def _files(tmp_path, sizes):
    out = []
    for i, size in enumerate(sizes):
        p = tmp_path / f"model-{i + 1:05d}-of-{len(sizes):05d}.safetensors"
        p.write_bytes(bytes(range(256)) * (size // 256) + b"x" * (size % 256))
        out.append(p)
    return out


def test_plan_chunks_covers_every_byte_in_order(tmp_path):
    files = _files(tmp_path, [1000, 2500, 0, 7])
    chunks = warm.plan_chunks(files, 1000)
    assert [(c.path.name[:11], c.offset, c.length) for c in chunks] == [
        ("model-00001", 0, 1000),
        ("model-00002", 0, 1000), ("model-00002", 1000, 1000), ("model-00002", 2000, 500),
        ("model-00004", 0, 7),
    ]
    whole = warm.plan_chunks(files, None)
    assert [c.length for c in whole] == [1000, 2500, 7]


def test_plan_chunks_rejects_bad_size(tmp_path):
    with pytest.raises(ValueError):
        warm.plan_chunks(_files(tmp_path, [10]), 0)


@pytest.mark.parametrize("threads,chunk", [(1, None), (4, 300), (16, 64)])
def test_read_files_reads_everything(tmp_path, threads, chunk):
    files = _files(tmp_path, [1000, 2500, 333])
    stats = warm.read_files(files, threads=threads, chunk_bytes=chunk, block_bytes=128)
    assert stats.bytes == 3833 and stats.files == 3 and not stats.errors
    assert stats.threads == threads and stats.seconds > 0
    assert stats.per_stream_gbps is not None and stats.first_chunk_s is not None
    js = stats.as_json()
    assert js["bytes"] == 3833 and js["chunk_mib"] == (None if chunk is None else 0)


def test_read_chunk_reads_the_exact_range(tmp_path, monkeypatch):
    (p,) = _files(tmp_path, [1000])
    seen = []
    real = warm.os.preadv

    def spy(fd, bufs, offset):
        n = real(fd, bufs, offset)
        seen.append((offset, n))
        return n

    monkeypatch.setattr(warm.os, "preadv", spy)
    buf = bytearray(64)
    assert warm._read_chunk(warm.Chunk(p, 100, 200), buf) == 200
    assert seen[0] == (100, 64) and sum(n for _, n in seen) == 200
    assert seen[-1][0] + seen[-1][1] == 300


def test_read_errors_are_collected_not_raised(tmp_path, monkeypatch):
    files = _files(tmp_path, [100, 100])
    chunks = warm.plan_chunks(files, None)
    files[1].unlink()  # vanishes between planning and reading
    monkeypatch.setattr(warm, "plan_chunks", lambda f, cb: chunks)
    stats = warm.read_files(files, threads=2)
    assert stats.bytes == 100 and len(stats.errors) == 1


def test_missing_file_fails_at_planning(tmp_path):
    with pytest.raises(FileNotFoundError):
        warm.read_files([tmp_path / "missing.safetensors"])


def test_read_short_file_is_an_error(tmp_path, monkeypatch):
    (p,) = _files(tmp_path, [100])
    monkeypatch.setattr(warm, "plan_chunks", lambda files, cb: [warm.Chunk(p, 0, 500)])
    stats = warm.read_files([p], threads=1)
    assert stats.bytes == 0 and len(stats.errors) == 1 and "short read" in stats.errors[0]


def test_thread_bounds(tmp_path):
    files = _files(tmp_path, [10])
    for bad in (0, warm.MAX_THREADS + 1):
        with pytest.raises(ValueError):
            warm.read_files(files, threads=bad)


def test_concurrency_is_bounded(tmp_path, monkeypatch):
    files = _files(tmp_path, [4096])
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow(chunk, buf):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        with lock:
            active[0] -= 1
        return chunk.length

    monkeypatch.setattr(warm, "_read_chunk", slow)
    stats = warm.read_files(files, threads=3, chunk_bytes=128)
    assert stats.bytes == 4096 and 1 < peak[0] <= 3


def test_snapshot_files_skips_cache_and_other_files(tmp_path):
    _files(tmp_path, [10, 10])
    (tmp_path / "config.json").write_text("{}")
    meta = tmp_path / ".cache" / "huggingface"
    meta.mkdir(parents=True)
    (meta / "x.safetensors").write_bytes(b"0")
    names = [p.name for p in warm.snapshot_files(tmp_path)]
    assert names == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def test_background_prefetch(tmp_path):
    _files(tmp_path, [5000, 5000])
    job = warm.start_prefetch(tmp_path, threads=2, chunk_bytes=1000, block_bytes=256)
    stats = job.wait(timeout=30)
    assert job.done and stats.bytes == 10000 and stats.files == 2


def test_background_prefetch_cancel(tmp_path, monkeypatch):
    files = _files(tmp_path, [4096])
    gate = threading.Event()

    def slow(chunk, buf):
        gate.wait(5)
        return chunk.length

    monkeypatch.setattr(warm, "_read_chunk", slow)
    job = warm.BackgroundPrefetch(files, threads=1, chunk_bytes=128)
    job.cancel()
    gate.set()
    stats = job.wait(timeout=10)
    assert stats.bytes < 4096  # chunks after the cancel were skipped
