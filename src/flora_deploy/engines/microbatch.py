"""Round-based micro-batcher (PLAN.md §6.3, D14).

NVIDIA's segmenter awaits ``predictor.apredict(request)`` and the caption stage awaits one call
per caption plan, each on the caller's event loop. Many episodes run as coroutines on one loop;
each submits its model calls here and awaits them. The batcher holds the calls until one of:

- **every live episode is blocked** on the batcher (it has calls pending or in flight, or is
  waiting for admission), so no more calls can arrive before something returns;
- the pending calls reach ``max_batch`` requests or ``max_batch_bytes`` of frames;
- the oldest pending call has waited ``max_wait_s`` (a safety net for slow CPU preparation or
  an episode the liveness accounting does not know about),

then runs **one** ``BatchEngine.generate_batch`` (one ``llm.generate``) in a dedicated worker
thread and resolves the futures. Only one batch runs at a time; calls that arrive meanwhile wait
for the next round, so preparation of round k+1 overlaps generation of round k.

Liveness: ``async with batcher.episode():`` around an episode's work. An episode that leaves
(finished or failed) no longer holds rounds back. Calls made outside an episode never hold a
round back either.

Bounded memory: admission waits while the pending frames exceed ``max_pending_bytes`` (a single
call larger than that is still admitted when nothing is pending).

Failure isolation: the engine reports per-request failures in ``Completion.error``. If
``generate_batch`` itself raises an ``Exception``, every call of that batch gets
``EngineBatchError`` and the batcher keeps serving. A ``BaseException`` that is not an
``Exception`` (``KeyboardInterrupt`` on SIGINT, ``SystemExit``) fails that batch's calls and is
re-raised from the pump task, which ends the event loop: the caller's preemption handling runs.

Batch composition is deterministic given the order of submissions (FIFO), but it is not
NVIDIA's (batch size 1): vLLM batch grouping changes numerics (``docs/rewarder-smoke.md``), which
is why ``balanced`` needs a certificate.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from flora_deploy.engines.base import BatchEngine, Completion, Sampling, VllmRequest

GiB = 1 << 30
_EPISODE: contextvars.ContextVar[_Episode | None] = contextvars.ContextVar(
    "flora_microbatch_episode", default=None)


class EngineBatchError(RuntimeError):
    """The engine raised for a whole batch (infrastructure failure, not NVIDIA semantics)."""


class BatcherClosed(RuntimeError):
    pass


@dataclass(eq=False)
class _Episode:
    id: int
    outstanding: int = 0  # calls pending or in flight
    admitting: int = 0  # calls waiting for admission (counts as blocked)
    done: bool = False

    @property
    def blocked(self) -> bool:
        return self.outstanding > 0 or self.admitting > 0


@dataclass(eq=False)
class _Item:
    request: VllmRequest
    sampling: Sampling
    future: asyncio.Future
    episode: _Episode | None
    enqueued: float
    nbytes: int


@dataclass
class BatcherStats:
    rounds: int = 0
    requests: int = 0
    failed_batches: int = 0
    triggers: Counter = field(default_factory=Counter)
    batch_sizes: list[int] = field(default_factory=list)  # first 10k rounds
    generate_s: float = 0.0
    wait_s: float = 0.0  # time pending calls spent waiting for their round (sum)
    peak_pending_bytes: int = 0
    peak_live: int = 0
    episodes: int = 0

    def as_json(self) -> dict:
        sizes = self.batch_sizes
        return {"rounds": self.rounds, "requests": self.requests,
                "failed_batches": self.failed_batches, "triggers": dict(self.triggers),
                "mean_batch": round(sum(sizes) / len(sizes), 2) if sizes else 0,
                "max_batch": max(sizes) if sizes else 0, "batch_sizes": sizes[:200],
                "generate_s": round(self.generate_s, 3), "wait_s": round(self.wait_s, 3),
                "peak_pending_bytes": self.peak_pending_bytes, "peak_live": self.peak_live,
                "episodes": self.episodes}


Observer = Callable[[Sequence[VllmRequest], Sequence[Completion], dict], None]


class MicroBatcher:
    """``engines.base.Engine`` over a ``BatchEngine``; see the module docstring.

    Create and use it inside one running event loop (``async with MicroBatcher(...) as mb``).
    """

    def __init__(self, engine: BatchEngine, *, max_batch: int = 128,
                 max_batch_bytes: int = 3 * GiB, max_wait_s: float = 2.0,
                 max_pending_bytes: int = 6 * GiB, observer: Observer | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if max_batch < 1 or max_wait_s <= 0:
            raise ValueError("max_batch >= 1 and max_wait_s > 0")
        self.engine = engine
        self.name = getattr(engine, "name", type(engine).__name__)
        self.max_batch, self.max_batch_bytes = int(max_batch), int(max_batch_bytes)
        self.max_wait_s, self.max_pending_bytes = float(max_wait_s), int(max_pending_bytes)
        self.observer = observer
        self.clock = clock
        self.stats = BatcherStats()
        self._pending: deque[_Item] = deque()
        self._pending_bytes = 0
        self._live: set[_Episode] = set()
        self._ids = itertools.count()
        self._wake: asyncio.Event | None = None
        self._space: asyncio.Condition | None = None
        self._pump: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        self._fatal: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------------------------
    async def start(self) -> MicroBatcher:
        if self._pump is None:
            self._wake = asyncio.Event()
            self._space = asyncio.Condition()
            self._executor = ThreadPoolExecutor(1, thread_name_prefix="flora-generate")
            self._pump = asyncio.get_running_loop().create_task(self._run(), name="microbatch")
        return self

    async def close(self) -> None:
        self._closed = True
        for item in self._pending:
            if not item.future.done():
                item.future.set_exception(BatcherClosed("micro-batcher closed"))
        self._pending.clear()
        self._pending_bytes = 0
        pump = self._pump
        if pump is not None:
            if pump.done():  # it may have ended with SIGINT: retrieve, never re-raise here
                if not pump.cancelled():
                    pump.exception()
            else:
                pump.cancel()
                try:
                    await pump
                except (asyncio.CancelledError, Exception):
                    pass
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    async def __aenter__(self) -> MicroBatcher:
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- liveness ----------------------------------------------------------------------------
    @asynccontextmanager
    async def episode(self):
        """Count the enclosed work as one live episode (re-entrant per task)."""
        current = _EPISODE.get()
        if current is not None and not current.done:
            yield current
            return
        ep = _Episode(next(self._ids))
        self._live.add(ep)
        self.stats.episodes += 1
        self.stats.peak_live = max(self.stats.peak_live, len(self._live))
        token = _EPISODE.set(ep)
        try:
            yield ep
        finally:
            _EPISODE.reset(token)
            ep.done = True
            self._live.discard(ep)
            self._poke()

    @property
    def live(self) -> int:
        return len(self._live)

    def all_blocked(self) -> bool:
        return all(ep.blocked for ep in self._live)

    # -- submission --------------------------------------------------------------------------
    async def generate(self, request: VllmRequest, sampling: Sampling) -> Completion:
        (out,) = await self.generate_many([request], [sampling])
        return out

    async def generate_many(self, requests: Sequence[VllmRequest],
                            samplings: Sequence[Sampling]) -> list[Completion]:
        """Submit calls together (in order) and await all of them."""
        if len(requests) != len(samplings):
            raise ValueError("one Sampling per request")
        if not requests:
            return []
        self._check_open()
        ep = _EPISODE.get()
        if ep is not None and ep.done:
            ep = None
        loop = asyncio.get_running_loop()
        size = sum(r.nbytes for r in requests)
        await self._admit(size, ep)
        items = [_Item(r, s, loop.create_future(), ep, self.clock(), r.nbytes)
                 for r, s in zip(requests, samplings, strict=True)]
        if ep is not None:
            ep.outstanding += len(items)
        self._pending.extend(items)
        self._pending_bytes += size
        self.stats.peak_pending_bytes = max(self.stats.peak_pending_bytes, self._pending_bytes)
        self._poke()
        try:
            return list(await asyncio.gather(*(i.future for i in items)))
        finally:
            if ep is not None:
                ep.outstanding -= len(items)
                self._poke()

    async def _admit(self, size: int, ep: _Episode | None) -> None:
        assert self._space is not None, "start() the batcher first"
        if self._pending_bytes == 0 or self._pending_bytes + size <= self.max_pending_bytes:
            return
        if ep is not None:
            ep.admitting += 1
        self._poke()
        try:
            async with self._space:
                await self._space.wait_for(
                    lambda: self._closed or self._pending_bytes == 0
                    or self._pending_bytes + size <= self.max_pending_bytes)
        finally:
            if ep is not None:
                ep.admitting -= 1
        self._check_open()

    def _check_open(self) -> None:
        if self._fatal is not None:
            raise BatcherClosed(f"micro-batcher stopped: {self._fatal!r}")
        if self._closed or self._pump is None:
            raise BatcherClosed("micro-batcher is not running")

    def _poke(self) -> None:
        if self._wake is not None:
            self._wake.set()

    # -- the pump ----------------------------------------------------------------------------
    def _trigger(self) -> str | None:
        if not self._pending:
            return None
        if len(self._pending) >= self.max_batch:
            return "size"
        if self._pending_bytes >= self.max_batch_bytes:
            return "bytes"
        if self.all_blocked():
            return "all_blocked"
        if self.clock() - self._pending[0].enqueued >= self.max_wait_s:
            return "timeout"
        return None

    def _take(self) -> list[_Item]:
        batch: list[_Item] = []
        nbytes = 0
        while self._pending and len(batch) < self.max_batch:
            item = self._pending[0]
            if batch and nbytes + item.nbytes > self.max_batch_bytes:
                break
            self._pending.popleft()
            if item.future.done():  # cancelled by its episode (whose count is already fixed)
                self._pending_bytes -= item.nbytes
                continue
            batch.append(item)
            nbytes += item.nbytes
        self._pending_bytes -= nbytes
        return batch

    async def _run(self) -> None:
        assert self._wake is not None and self._space is not None
        loop = asyncio.get_running_loop()
        while True:
            trigger = self._trigger()
            if trigger is None:
                self._wake.clear()
                timeout = None
                if self._pending:
                    timeout = max(0.0, self.max_wait_s
                                  - (self.clock() - self._pending[0].enqueued))
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout)
                except TimeoutError:
                    pass
                continue
            batch = self._take()
            async with self._space:
                self._space.notify_all()
            if not batch:
                continue
            now = self.clock()
            self.stats.wait_s += sum(now - i.enqueued for i in batch)
            self.stats.rounds += 1
            self.stats.requests += len(batch)
            self.stats.triggers[trigger] += 1
            if len(self.stats.batch_sizes) < 10_000:
                self.stats.batch_sizes.append(len(batch))
            reqs = [i.request for i in batch]
            sps = [i.sampling for i in batch]
            t0 = time.perf_counter()
            try:
                outs = await loop.run_in_executor(self._executor, self.engine.generate_batch,
                                                  reqs, sps)
                if len(outs) != len(batch):
                    raise EngineBatchError(f"engine returned {len(outs)} completions for "
                                           f"{len(batch)} requests")
            except Exception as exc:  # noqa: BLE001 - fail this batch, keep serving
                self.stats.failed_batches += 1
                self.stats.generate_s += time.perf_counter() - t0
                err = exc if isinstance(exc, EngineBatchError) else EngineBatchError(
                    f"{type(exc).__name__}: {exc}")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(err)
                continue
            except BaseException as exc:  # SIGINT / SystemExit: stop the loop
                self._fatal = exc
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(BatcherClosed(repr(exc)))
                raise
            self.stats.generate_s += time.perf_counter() - t0
            if self.observer is not None:
                try:
                    self.observer(reqs, outs, {"round": self.stats.rounds, "trigger": trigger})
                except Exception:  # noqa: BLE001 - observers never break inference
                    pass
            for item, out in zip(batch, outs, strict=True):
                if not item.future.done():
                    item.future.set_result(out)

    def audit(self) -> dict[str, Any]:
        engine_audit = getattr(self.engine, "audit", None)
        return {"batcher": {"max_batch": self.max_batch, "max_batch_bytes": self.max_batch_bytes,
                            "max_wait_s": self.max_wait_s,
                            "max_pending_bytes": self.max_pending_bytes,
                            "stats": self.stats.as_json()},
                "engine": engine_audit() if callable(engine_audit) else None}
