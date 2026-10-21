"""Shared state of one ``ActivePatches``: byte-bounded LRU caches, decoders, a thread pool.

Every cache here is either **content-addressed** (the key is a digest of the exact input bytes,
so a hit is the output NVIDIA's deterministic code computes for those bytes) or keyed by a
source file that was hashed and matched to the probed ``Video.sha256`` before first use. That
makes correctness independent of cache scope: ``episode()`` only bounds memory and open files.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from PIL import Image

DEFAULT_CACHE_BYTES = 4 * 1024**3


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 1)))


def image_nbytes(im: Image.Image) -> int:
    bands = len(im.getbands())
    return im.width * im.height * max(bands, 1) + 256


def image_key(im: Image.Image) -> tuple:
    """Content key of a PIL image: everything PIL's encoders read (mode, size, pixels, palette,
    info). Two images with equal keys encode to the same bytes under the same parameters."""
    h = hashlib.blake2b(im.tobytes(), digest_size=20)
    palette = im.getpalette() if im.mode in ("P", "PA") else None
    info = tuple(sorted((str(k), repr(v)) for k, v in im.info.items()))
    return (im.mode, im.size, h.digest(), repr(palette), info)


class ByteLRU:
    """Thread-safe LRU of values with a byte budget (values larger than the budget are not
    kept)."""

    def __init__(self, max_bytes: int, sizeof: Callable[[Any], int] = image_nbytes):
        self.max_bytes = int(max_bytes)
        self._sizeof = sizeof
        self._data: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            self._data.move_to_end(key)
            return hit[0]

    def put(self, key: Hashable, value: Any) -> None:
        size = self._sizeof(value)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            if size > self.max_bytes:
                return
            self._data[key] = (value, size)
            self._bytes += size
            while self._bytes > self.max_bytes:
                _, (_, s) = self._data.popitem(last=False)
                self._bytes -= s

    def demote(self, key: Hashable) -> None:
        """Make ``key`` the next eviction candidate (it is no longer known to be needed)."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key, last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)


class AccelState:
    """Caches and workers shared by the patches of one ``apply_patches`` call.

    ``cache_bytes``: total budget of the decoded-frame / transport / PNG caches (each gets this
    budget; they are cleared together at the end of an ``episode()``).
    ``workers``: thread-pool size for codec work (PIL and PyAV release the GIL in codecs);
    1 runs everything inline.
    """

    def __init__(self, *, cache_bytes: int = DEFAULT_CACHE_BYTES, workers: int | None = None,
                 seek: bool = True):
        self.cache_bytes = int(cache_bytes)
        self.workers = default_workers() if workers is None else max(1, int(workers))
        self.seek = seek
        self.frames = ByteLRU(self.cache_bytes)  # decoded / composed frames
        self.transport = ByteLRU(self.cache_bytes)  # human_transport outputs
        self.png = ByteLRU(self.cache_bytes, sizeof=_png_entry_bytes)  # sha -> (bytes?, decoded)
        self.counters: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._pool: Executor | None = None
        self._io_pool: Executor | None = None
        self._decode_pool: Executor | None = None
        self._stores: dict[int, Any] = {}
        self.decoders: dict[Hashable, Any] = {}  # source key -> framecache.SourceDecoder
        self._depth = 0

    # -- workers ---------------------------------------------------------------------------
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        if self.workers <= 1:
            fut: Future = Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - surfaced by .result()
                fut.set_exception(exc)
            return fut
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="flora-accel")
            pool = self._pool
        return pool.submit(fn, *args, **kwargs)

    def map(self, fn: Callable, items: list) -> list:
        if self.workers <= 1 or len(items) <= 1:
            return [fn(x) for x in items]
        return [f.result() for f in [self.submit(fn, x) for x in items]]

    def map_io(self, fn: Callable, items: list) -> list:
        """Like ``map`` on a separate pool, for tasks that themselves ``submit`` codec work
        (one per camera), so they never wait on a pool they occupy."""
        if self.workers <= 1 or len(items) <= 1:
            return [fn(x) for x in items]
        with self._lock:
            if self._io_pool is None:
                self._io_pool = ThreadPoolExecutor(4, thread_name_prefix="flora-accel-src")
            pool = self._io_pool
        return [f.result() for f in [pool.submit(fn, x) for x in items]]

    def submit_decode(self, fn: Callable, *args) -> Future:
        """Run one decoder (GOP-parallel decode) on its own pool: decoders ``submit`` their
        colour conversions to the codec pool and must not wait on a pool they occupy."""
        with self._lock:
            if self._decode_pool is None:
                self._decode_pool = ThreadPoolExecutor(
                    self.workers, thread_name_prefix="flora-accel-dec"
                )
            pool = self._decode_pool
        return pool.submit(fn, *args)

    def frame_store(self, nv):
        """The ``framecache.FrameStore`` of this state for family handle ``nv``."""
        from flora_deploy.accel.framecache import FrameStore

        with self._lock:
            store = self._stores.get(id(nv))
            if store is None:
                store = self._stores[id(nv)] = FrameStore(nv, self)
            return store

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    # -- lifetime --------------------------------------------------------------------------
    @contextmanager
    def episode(self):
        """Bound caches and open decoders to one episode (re-entrant; clears on outer exit)."""
        with self._lock:
            self._depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._depth -= 1
                outer = self._depth == 0
            if outer:
                self.reset()

    def reset(self) -> None:
        with self._lock:
            decoders, self.decoders = self.decoders, {}
        for d in decoders.values():
            d.close()
        for store in list(self._stores.values()):
            store.reset_keys()
        self.frames.clear()
        self.transport.clear()
        self.png.clear()

    def close(self) -> None:
        self.reset()
        with self._lock:
            pools = (self._pool, self._io_pool, self._decode_pool)
            self._pool = self._io_pool = self._decode_pool = None
        for pool in pools:
            if pool is not None:
                pool.shutdown(wait=True)

    def stats(self) -> dict:
        with self._lock:
            out = dict(self.counters)
        out.update(
            frame_cache_bytes=self.frames.nbytes,
            transport_cache_bytes=self.transport.nbytes,
            png_cache_bytes=self.png.nbytes,
        )
        return out


def _png_entry_bytes(entry) -> int:
    data, image = entry
    return (len(data) if data is not None else 0) + image_nbytes(image)
