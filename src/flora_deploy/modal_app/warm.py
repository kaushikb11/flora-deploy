"""Cold-start helpers shared by the GPU workers (W3-B, ``docs/cold-start.md``).

Two independent pieces, both Modal-agnostic and free of NVIDIA code:

1. **Parallel prefetch** (:func:`read_files`, :func:`start_prefetch`).
   Reads a snapshot's weight files into the page cache with a bounded thread pool, splitting large
   files into fixed-size chunks so N readers stream in parallel even when a snapshot has only a
   few shards. Chunks are handed out in file order, so the read frontier runs roughly
   sequentially *ahead of* a single sequential consumer (vLLM's safetensors iterator, NVIDIA's
   ``validate_manifest`` hasher). The same function, with ``threads=1`` and ``chunk_bytes=None``,
   is the single-stream sequential baseline the lab measures against.

2. **No-copy model cache for NVIDIA's ``Session.load``** (:func:`build_linked_layout`). The WS-D
   worker copies the 109 GB Captioner snapshot to local disk because ``huggingface_hub`` takes a
   ``filelock`` inside ``local_dir/.cache/huggingface`` (EROFS on the read-only Volume). This
   layout instead makes ``local_dir`` a small *writable* directory that holds:

   - a real copy of ``.cache/huggingface`` (tree listing + download metadata, a few KB; locks
     are created here), and
   - one absolute symlink per payload file, pointing at the file on the Volume.

   ``huggingface_hub`` stats the symlink targets (size and mtime match the prep metadata) and
   returns the directory without a request. NVIDIA's ``validate_manifest`` then hashes every byte
   through the symlinks, unmodified. Its path guard requires each resolved file to lie under
   ``allowed_root``, which is ``path.parent.parent`` when the snapshot's parent directory is
   named ``snapshots`` (NVIDIA's provision for the hub cache layout, where snapshot files are
   symlinks into ``../../blobs``); otherwise it is the snapshot directory itself, which a link
   farm can never satisfy.

   So the model cache is ``<root>/snapshots/<kind>`` and the Volume's *real* path must lie under
   ``<root>``. On Modal a Volume mount point is a symlink to ``/__modal/volumes/<volume id>``
   (measured), so the only usable ``<root>`` is ``/``: ``FLORA_MODEL_CACHE=/snapshots`` with the
   weights at the usual ``/weights``. NVIDIA's guard then accepts any path, so we re-impose it
   per snapshot: :func:`check_links` requires every link to resolve to a file inside the resolved
   Volume snapshot it was made from. Integrity is unchanged: NVIDIA's manifest still hashes every
   byte that will be loaded.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

MiB = 1 << 20
DEFAULT_THREADS = 16
DEFAULT_CHUNK_BYTES = 256 * MiB
DEFAULT_BLOCK_BYTES = 16 * MiB
MAX_THREADS = 64
WEIGHT_PATTERNS = ("*.safetensors",)

# ---------------------------------------------------------------------------
# Parallel prefetch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    path: Path
    offset: int
    length: int


@dataclass(frozen=True)
class ReadStats:
    files: int
    bytes: int
    seconds: float
    threads: int
    chunk_bytes: int | None
    errors: tuple[str, ...] = ()
    first_chunk_s: float | None = None  # time to the first finished chunk
    chunk_seconds: tuple[float, ...] = field(default=(), repr=False)

    @property
    def gbps(self) -> float:
        """Decimal GB/s over the whole read."""
        return self.bytes / self.seconds / 1e9 if self.seconds > 0 else float("inf")

    @property
    def per_stream_gbps(self) -> float | None:
        """Mean throughput of one reader while it was reading (bytes / summed chunk time)."""
        busy = sum(self.chunk_seconds)
        return self.bytes / busy / 1e9 if busy > 0 else None

    def as_json(self) -> dict:
        return {
            "files": self.files,
            "bytes": self.bytes,
            "gib": round(self.bytes / 2**30, 3),
            "seconds": round(self.seconds, 3),
            "gbps": round(self.gbps, 3),
            "per_stream_gbps": None if self.per_stream_gbps is None
            else round(self.per_stream_gbps, 3),
            "threads": self.threads,
            "chunk_mib": None if self.chunk_bytes is None else self.chunk_bytes // MiB,
            "first_chunk_s": None if self.first_chunk_s is None else round(self.first_chunk_s, 3),
            "errors": list(self.errors),
        }


def snapshot_files(
    snapshot_dir: Path | str, patterns: Sequence[str] = WEIGHT_PATTERNS
) -> list[Path]:
    """Weight files of a snapshot in natural (shard) order; ``.cache`` is skipped."""
    root = Path(snapshot_dir)
    out = [
        p for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts
        and any(fnmatch(p.name, pat) for pat in patterns)
    ]
    return sorted(out, key=lambda p: str(p.relative_to(root)))


def plan_chunks(files: Iterable[Path], chunk_bytes: int | None) -> list[Chunk]:
    """Split ``files`` into chunks of at most ``chunk_bytes`` (``None``: one chunk per file),
    in file order then offset order."""
    if chunk_bytes is not None and chunk_bytes < 1:
        raise ValueError("chunk_bytes must be >= 1 or None")
    chunks: list[Chunk] = []
    for path in files:
        size = Path(path).stat().st_size
        if size == 0:
            continue
        step = size if chunk_bytes is None else chunk_bytes
        chunks.extend(Chunk(Path(path), off, min(step, size - off)) for off in range(0, size, step))
    return chunks


def _read_chunk(chunk: Chunk, buf: bytearray) -> int:
    view = memoryview(buf)
    done = 0
    fd = os.open(chunk.path, os.O_RDONLY)
    try:
        while done < chunk.length:
            want = min(len(buf), chunk.length - done)
            n = os.preadv(fd, [view[:want]], chunk.offset + done)
            if n <= 0:
                raise OSError(f"short read at {chunk.path}:{chunk.offset + done}")
            done += n
    finally:
        os.close(fd)
    return done


def read_files(
    files: Sequence[Path | str],
    *,
    threads: int = DEFAULT_THREADS,
    chunk_bytes: int | None = DEFAULT_CHUNK_BYTES,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    stop: threading.Event | None = None,
) -> ReadStats:
    """Read every byte of ``files`` once with at most ``threads`` concurrent readers.

    The data is discarded; the point is the page cache (and the measurement). Errors are
    collected, not raised, so a prefetch can never fail a worker's startup.
    """
    if not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"threads must be in 1..{MAX_THREADS}")
    if block_bytes < 1:
        raise ValueError("block_bytes must be >= 1")
    paths = [Path(f) for f in files]
    chunks = plan_chunks(paths, chunk_bytes)
    local = threading.local()
    lock = threading.Lock()
    errors: list[str] = []
    seconds: list[float] = []
    first: list[float] = []
    t0 = time.perf_counter()

    def one(chunk: Chunk) -> int:
        if stop is not None and stop.is_set():
            return 0
        buf = getattr(local, "buf", None)
        if buf is None:
            buf = local.buf = bytearray(block_bytes)
        t = time.perf_counter()
        try:
            n = _read_chunk(chunk, buf)
        except OSError as e:
            with lock:
                errors.append(f"{chunk.path}@{chunk.offset}: {e}")
            return 0
        done = time.perf_counter()
        with lock:
            seconds.append(done - t)
            if not first:
                first.append(done - t0)
        return n

    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="flora-prefetch") as pool:
        total = sum(pool.map(one, chunks))
    return ReadStats(
        files=len({c.path for c in chunks}),
        bytes=total,
        seconds=time.perf_counter() - t0,
        threads=threads,
        chunk_bytes=chunk_bytes,
        errors=tuple(errors),
        first_chunk_s=first[0] if first else None,
        chunk_seconds=tuple(seconds),
    )


class BackgroundPrefetch:
    """A :func:`read_files` running in a daemon thread. ``wait()`` returns its stats."""

    def __init__(self, files: Sequence[Path | str], **kwargs) -> None:
        self.files = [Path(f) for f in files]
        self.started = time.perf_counter()
        self._stop = threading.Event()
        self._result: ReadStats | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, kwargs=kwargs, daemon=True,
                                        name="flora-prefetch-main")
        self._thread.start()

    def _run(self, **kwargs) -> None:
        try:
            self._result = read_files(self.files, stop=self._stop, **kwargs)
        except BaseException as e:  # noqa: BLE001 - reported by wait()
            self._error = e

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def cancel(self) -> None:
        """Stop handing out new chunks (chunks in flight finish)."""
        self._stop.set()

    def wait(self, timeout: float | None = None) -> ReadStats | None:
        self._thread.join(timeout)
        if self._error is not None:
            raise self._error
        return self._result


def start_prefetch(
    snapshot_dir: Path | str,
    *,
    patterns: Sequence[str] = WEIGHT_PATTERNS,
    **kwargs,
) -> BackgroundPrefetch:
    """Start prefetching a snapshot's weights in the background (call first in ``enter``)."""
    return BackgroundPrefetch(snapshot_files(snapshot_dir, patterns), **kwargs)



# ---------------------------------------------------------------------------
# No-copy model cache for NVIDIA's Session.load
# ---------------------------------------------------------------------------

LINKED_ROOT = PurePosixPath("/")  # FLORA_MODEL_CACHE = /snapshots (see module docstring)
HUB_ROOT = PurePosixPath("/tmp/flora-offline")  # HF_HOME / HF_HUB_CACHE, as in WS-D
HUB_META = PurePosixPath(".cache") / "huggingface"


def linked_offline_env(
    root: Path | PurePosixPath = LINKED_ROOT, hub_root: Path | PurePosixPath = HUB_ROOT
) -> dict[str, str]:
    """Container env for :func:`build_linked_layout` (set before any import, like
    ``stages.reference.offline_env``). Only ``FLORA_MODEL_CACHE`` differs from WS-D's."""
    from flora_deploy.stages.reference import offline_env

    return {**offline_env(PurePosixPath(hub_root)),
            "FLORA_MODEL_CACHE": str(PurePosixPath(root) / "snapshots")}


@dataclass(frozen=True)
class LinkedLayout:
    root: Path
    model_cache: Path  # FLORA_MODEL_CACHE == root/snapshots
    hub_cache: Path
    hub_root: Path
    sources: dict[str, Path] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    stats: dict[str, dict] = field(default_factory=dict)

    def env(self) -> dict[str, str]:
        return linked_offline_env(self.root, self.hub_root)


def _is_relative(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def link_snapshot(src: Path, dest: Path) -> dict:
    """Symlink every payload file of ``src`` into a real directory ``dest`` and copy
    ``src/.cache/huggingface`` (minus ``*.lock``) so ``huggingface_hub`` can lock there."""
    t0 = time.monotonic()
    if dest.is_symlink():
        raise OSError(f"{dest} is a symlink; it must be a real, writable directory")
    dest.mkdir(parents=True, exist_ok=True)
    linked = copied = linked_bytes = 0
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dest / rel
        if path.is_dir():
            if not path.is_symlink():
                target.mkdir(parents=True, exist_ok=True)
            continue
        if rel.parts[:2] == HUB_META.parts:
            if path.name.endswith(".lock"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)  # mtime kept: the metadata's timestamp check reads it
            copied += 1
            continue
        if rel.parts and rel.parts[0] == ".cache":
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            if target.is_symlink() and Path(os.readlink(target)) == path:
                linked += 1
                linked_bytes += path.stat().st_size
                continue
            raise OSError(f"{target} exists and is not a link to {path}")
        target.symlink_to(path)
        linked += 1
        linked_bytes += path.stat().st_size
    return {"linked_files": linked, "linked_bytes": linked_bytes, "copied_meta_files": copied,
            "seconds": round(time.monotonic() - t0, 3)}


def check_links(src: Path, dest: Path) -> int:
    """Our copy of NVIDIA's path guard, scoped to one snapshot: every symlink in the link farm
    ``dest`` must resolve to a regular file inside the Volume snapshot ``src``. Returns the
    number of links checked."""
    real_src = Path(src).resolve()
    n = 0
    for path in Path(dest).rglob("*"):
        if not path.is_symlink():
            continue
        target = path.resolve()
        if not _is_relative(target, real_src) or not target.is_file():
            raise OSError(f"{path} -> {target} leaves the snapshot {real_src}")
        n += 1
    return n


def build_linked_layout(
    root: Path | PurePosixPath = LINKED_ROOT,
    weights_root: Path | PurePosixPath | None = None,
    *,
    hub_root: Path | PurePosixPath = HUB_ROOT,
    kinds: tuple[str, ...] = ("tempseg", "captioner"),
    require_receipts: bool = True,
) -> LinkedLayout:
    """No-copy counterpart of ``stages.reference.build_offline_layout`` (module docstring).

    Model kinds go to ``root/snapshots/<kind>`` as link farms (checked by :func:`check_links`);
    the TempSeg base is linked into ``hub_root/hf/hub`` exactly as WS-D does. Idempotent.
    """
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import WEIGHTS_MOUNT, snapshot_path
    from flora_deploy.stages.reference import (
        BASE_KEY,
        MODEL_KINDS,
        OfflineLayoutError,
        _check_snapshot,
        _symlink,
        hub_repo_folder,
        tree_listing,
    )

    root = Path(root)
    weights_root = Path(WEIGHTS_MOUNT if weights_root is None else weights_root)
    if not _is_relative(weights_root.resolve(), root.resolve()):
        raise OfflineLayoutError(
            f"weights {weights_root} (really {weights_root.resolve()}) must be inside {root}: "
            f"NVIDIA's manifest guard only accepts files under {root} (allowed_root = "
            "snapshots/..)"
        )
    model_cache, hub_cache = root / "snapshots", Path(hub_root) / "hf" / "hub"
    sources: dict[str, Path] = {}
    paths: dict[str, Path] = {}
    stats: dict[str, dict] = {}
    for kind in kinds:
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {kind!r}")
        pin = PINS_BY_KEY[kind]
        src = Path(snapshot_path(pin, PurePosixPath(weights_root)))
        _check_snapshot(pin, src, require_receipt=require_receipts)
        dest = model_cache / kind
        stats[kind] = link_snapshot(src, dest)
        stats[kind]["checked_links"] = check_links(src, dest)
        sources[kind], paths[kind] = src, dest

    base = PINS_BY_KEY[BASE_KEY]
    src = Path(snapshot_path(base, PurePosixPath(weights_root)))
    _check_snapshot(base, src, require_receipt=require_receipts)
    storage = hub_cache / hub_repo_folder(base.repo)
    _symlink(storage / "snapshots" / base.revision, src)
    _symlink(storage / "trees", tree_listing(src, base.revision).parent)
    sources[BASE_KEY], paths[BASE_KEY] = src, storage / "snapshots" / base.revision
    return LinkedLayout(root, model_cache, hub_cache, Path(hub_root), sources, paths, stats)
