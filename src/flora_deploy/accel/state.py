"""State of one ``ActivePatches``: per-episode caches and decoders, shared workers.

Every cache here is either **content-addressed** (the key is a digest of the exact input bytes,
so a hit is the output NVIDIA's deterministic code computes for those bytes) or keyed by a
source file that was hashed and matched to the probed ``Video.sha256`` before first use. That
makes correctness independent of cache scope; scoping only bounds memory and open files.

Scoping (W3-A runs several episodes of one batch concurrently on threads):

- ``episode()`` opens an ``EpisodeScope`` bound to the *current context* (a ``ContextVar``), so
  episodes that overlap on different threads (or asyncio tasks) each get their own frame /
  transport / PNG caches, decoders, keys and lookahead counters, and each scope is closed (its
  decoders' files closed, its caches dropped) when *its* episode exits, whatever the others do.
  Enter ``episode()`` inside the thread or task that runs the episode. Work the patches hand to
  the shared thread pools runs in a copy of the submitting context, so it sees the same scope;
  ``asyncio.to_thread`` (NVIDIA's TempSeg predictor) copies the context too.
- Calls made outside any episode use a default scope that lives until ``reset()`` / ``close()``
  (the pre-W3-A behaviour).
- Memory is bounded **across** concurrent scopes: every cache of every scope draws on one byte
  budget (``max_total_cache_bytes``, default ``3 * cache_bytes``: what one episode could hold
  before), evicting from the largest cache first. Eviction only forces a recompute.
- Open decoders are bounded across scopes (``max_open_decoders``): the least recently used idle
  decoder closes its file first; it reopens (from a verified random-access point) if needed.
"""

from __future__ import annotations

import contextvars
import hashlib
import itertools
import os
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from PIL import Image

DEFAULT_CACHE_BYTES = 4 * 1024**3
DEFAULT_MAX_OPEN_DECODERS = 64


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


class ByteBudget:
    """One byte budget shared by several ``ByteLRU``s (one lock for all of them)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self.lock = threading.RLock()
        self.total = 0
        self.peak = 0
        self.evictions = 0
        self._lrus: set[ByteLRU] = set()

    def _register(self, lru: ByteLRU) -> None:
        self._lrus.add(lru)

    def _unregister(self, lru: ByteLRU) -> None:
        self._lrus.discard(lru)

    def _enforce(self, newest: ByteLRU | None = None) -> None:
        # Caller holds self.lock. Evict the oldest entry of the largest LRU; the entry just
        # put (the newest of ``newest``) goes only when nothing else is left.
        while self.total > self.max_bytes:
            others = [lru for lru in self._lrus
                      if lru._data and not (lru is newest and len(lru._data) == 1)]
            victim = max(others, key=lambda x: x._bytes, default=newest if newest is not None
                         and newest._data else None)
            if victim is None:
                break
            victim._evict_oldest()
            self.evictions += 1
        self.peak = max(self.peak, self.total)  # resident bytes (after eviction)


class ByteLRU:
    """Thread-safe LRU of values with a byte budget (values larger than the budget are not
    kept). With ``budget``, it also draws on a byte budget shared with other LRUs."""

    def __init__(self, max_bytes: int, sizeof: Callable[[Any], int] = image_nbytes,
                 budget: ByteBudget | None = None):
        self.max_bytes = int(max_bytes)
        self._sizeof = sizeof
        self._data: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._budget = budget
        self._lock = budget.lock if budget is not None else threading.Lock()
        self.closed = False
        if budget is not None:
            with self._lock:
                budget._register(self)

    def _account(self, delta: int) -> None:
        self._bytes += delta
        if self._budget is not None:
            self._budget.total += delta

    def _evict_oldest(self) -> None:
        _, (_, s) = self._data.popitem(last=False)
        self._account(-s)

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
                self._account(-old[1])
            if self.closed or size > self.max_bytes:
                return
            self._data[key] = (value, size)
            self._account(size)
            while self._bytes > self.max_bytes:
                self._evict_oldest()
            if self._budget is not None:
                self._budget._enforce(self)

    def demote(self, key: Hashable) -> None:
        """Make ``key`` the next eviction candidate (it is no longer known to be needed)."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key, last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._account(-self._bytes)

    def close(self) -> None:
        """Clear and stop accepting values (a closed episode's late writes are dropped)."""
        with self._lock:
            self.closed = True
            self.clear()
            if self._budget is not None:
                self._budget._unregister(self)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)


class EpisodeScope:
    """Caches, decoders and frame-store state of one episode (or of the default scope)."""

    _ids = itertools.count()

    def __init__(self, state: AccelState, name: str = "episode"):
        self.id = next(self._ids)
        self.name = name
        self.state = state
        budget = state._budget
        self.frames = ByteLRU(state.cache_bytes, budget=budget)  # decoded / composed frames
        self.transport = ByteLRU(state.cache_bytes, budget=budget)  # human_transport outputs
        self.png = ByteLRU(state.cache_bytes, sizeof=_png_entry_bytes, budget=budget)
        self.decoders: dict[Hashable, Any] = {}  # source key -> framecache.SourceDecoder
        self.stores: dict[int, Any] = {}  # id(nv) -> framecache.FrameStore
        self.lock = threading.RLock()
        self.depth = 0
        self.closed = False

    def store(self, nv):
        from flora_deploy.accel.framecache import FrameStore

        with self.lock:
            store = self.stores.get(id(nv))
            if store is None:
                store = self.stores[id(nv)] = FrameStore(nv, self.state, self)
            return store

    def nbytes(self) -> int:
        return self.frames.nbytes + self.transport.nbytes + self.png.nbytes

    def clear(self) -> None:
        with self.lock:
            decoders, self.decoders = self.decoders, {}
            stores = list(self.stores.values())
        for d in decoders.values():
            d.close()
        for store in stores:
            store.reset_keys()
        self.frames.clear()
        self.transport.clear()
        self.png.clear()

    def close(self) -> None:
        with self.lock:
            self.closed = True
        self.clear()
        for lru in (self.frames, self.transport, self.png):
            lru.close()


class _ScopedFrameStore:
    """What the patches hold: routes every call to the current scope's ``FrameStore``."""

    def __init__(self, state: AccelState, nv):
        self._state = state
        self._nv = nv

    def current(self):
        return self._state.scope().store(self._nv)

    def __getattr__(self, name: str):
        return getattr(self.current(), name)


class AccelState:
    """Workers shared by the patches of one ``apply_patches`` call, plus per-episode scopes.

    ``cache_bytes``: budget of each decoded-frame / transport / PNG cache of one episode.
    ``max_total_cache_bytes``: budget of all caches of all concurrent episodes together
    (default ``3 * cache_bytes``).
    ``max_open_decoders``: open source files across all episodes (idle ones close first).
    ``workers``: thread-pool size for codec work (PIL and PyAV release the GIL in codecs);
    1 runs everything inline.
    """

    def __init__(self, *, cache_bytes: int = DEFAULT_CACHE_BYTES, workers: int | None = None,
                 seek: bool = True, max_total_cache_bytes: int | None = None,
                 max_open_decoders: int = DEFAULT_MAX_OPEN_DECODERS):
        self.cache_bytes = int(cache_bytes)
        self.max_total_cache_bytes = int(
            3 * self.cache_bytes if max_total_cache_bytes is None else max_total_cache_bytes
        )
        self.max_open_decoders = max(1, int(max_open_decoders))
        self.workers = default_workers() if workers is None else max(1, int(workers))
        self.seek = seek
        self.counters: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._pool: Executor | None = None
        self._io_pool: Executor | None = None
        self._decode_pool: Executor | None = None
        self._budget = ByteBudget(self.max_total_cache_bytes)
        self._var: contextvars.ContextVar[EpisodeScope | None] = contextvars.ContextVar(
            f"flora_accel_scope_{id(self)}", default=None
        )
        self._default = EpisodeScope(self, "default")
        self._scopes: set[EpisodeScope] = set()
        self._routers: dict[int, _ScopedFrameStore] = {}
        self._open: OrderedDict[Any, None] = OrderedDict()  # decoders with an open file, LRU
        self._peak_open = 0
        self._peak_scopes = 0

    # -- scopes ----------------------------------------------------------------------------
    def scope(self) -> EpisodeScope:
        """The current context's episode scope (the default scope outside episodes)."""
        scope = self._var.get()
        if scope is None or scope.closed:
            return self._default
        return scope

    @property
    def frames(self) -> ByteLRU:
        return self.scope().frames

    @property
    def transport(self) -> ByteLRU:
        return self.scope().transport

    @property
    def png(self) -> ByteLRU:
        return self.scope().png

    @property
    def decoders(self) -> dict:
        return self.scope().decoders

    def frame_store(self, nv):
        """The ``framecache.FrameStore`` for family handle ``nv``, routed per episode scope."""
        with self._lock:
            router = self._routers.get(id(nv))
            if router is None:
                router = self._routers[id(nv)] = _ScopedFrameStore(self, nv)
            return router

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
        return pool.submit(contextvars.copy_context().run, fn, *args, **kwargs)

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
        futs = [pool.submit(contextvars.copy_context().run, fn, x) for x in items]
        return [f.result() for f in futs]

    def submit_decode(self, fn: Callable, *args) -> Future:
        """Run one decoder (GOP-parallel decode) on its own pool: decoders ``submit`` their
        colour conversions to the codec pool and must not wait on a pool they occupy."""
        with self._lock:
            if self._decode_pool is None:
                self._decode_pool = ThreadPoolExecutor(
                    self.workers, thread_name_prefix="flora-accel-dec"
                )
            pool = self._decode_pool
        return pool.submit(contextvars.copy_context().run, fn, *args)

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    # -- open decoders ---------------------------------------------------------------------
    def note_open(self, decoder) -> None:
        """``decoder`` holds an open file now; close the least recently used idle ones while
        more than ``max_open_decoders`` are open."""
        with self._lock:
            self._open[decoder] = None
            self._open.move_to_end(decoder)
            self._peak_open = max(self._peak_open, len(self._open))
            excess = len(self._open) - self.max_open_decoders
            candidates = [d for d in self._open if d is not decoder][: max(0, excess)]
        for d in candidates:
            if d.close_if_idle():
                self.count("decoders_closed_idle")

    def note_closed(self, decoder) -> None:
        with self._lock:
            self._open.pop(decoder, None)

    # -- lifetime --------------------------------------------------------------------------
    @contextmanager
    def episode(self):
        """Scope caches and open decoders to one episode of the current context (re-entrant:
        a nested ``episode()`` in the same context reuses the scope; it closes on outer exit).
        Concurrent episodes in other threads / tasks get their own scopes."""
        current = self._var.get()
        if current is not None and not current.closed:
            with current.lock:
                current.depth += 1
            try:
                yield self
            finally:
                with current.lock:
                    current.depth -= 1
            return
        scope = EpisodeScope(self)
        scope.depth = 1
        with self._lock:
            self._scopes.add(scope)
            self._peak_scopes = max(self._peak_scopes, len(self._scopes))
            self.counters["episodes"] += 1
        token = self._var.set(scope)
        try:
            yield self
        finally:
            try:
                self._var.reset(token)
            except ValueError:  # exited from another context: still close the scope
                self._var.set(None)
            with self._lock:
                self._scopes.discard(scope)
            scope.close()

    def reset(self) -> None:
        """Clear the default scope (calls made outside any episode)."""
        self._default.clear()

    def close(self) -> None:
        with self._lock:
            scopes = list(self._scopes)
            self._scopes.clear()
        for scope in scopes:
            scope.close()
        self._default.clear()
        with self._lock:
            pools = (self._pool, self._io_pool, self._decode_pool)
            self._pool = self._io_pool = self._decode_pool = None
        for pool in pools:
            if pool is not None:
                pool.shutdown(wait=True)

    def stats(self) -> dict:
        with self._lock:
            out = dict(self.counters)
            open_now = len(self._open)
            active = len(self._scopes)
            peaks = dict(peak_open_decoders=self._peak_open, peak_active_episodes=self._peak_scopes)
        scope = self.scope()
        with self._budget.lock:
            total, peak, evictions = self._budget.total, self._budget.peak, self._budget.evictions
        out.update(
            frame_cache_bytes=scope.frames.nbytes,
            transport_cache_bytes=scope.transport.nbytes,
            png_cache_bytes=scope.png.nbytes,
            cache_bytes_total=total,
            cache_bytes_total_peak=peak,
            cache_bytes_total_limit=self.max_total_cache_bytes,
            cache_budget_evictions=evictions,
            open_decoders=open_now,
            active_episodes=active,
            **peaks,
        )
        return out


def _png_entry_bytes(entry) -> int:
    data, image = entry
    return (len(data) if data is not None else 0) + image_nbytes(image)
