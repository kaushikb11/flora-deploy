"""Shared fakes for job-system tests: an eventually consistent store and a fault-injecting spawner.

``FakeStore`` implements ``flora_deploy.jobs.store.Store`` with whole-file writes and a
configurable listing lag (writes become visible to ``list`` only after ``lag`` ticks), which is
how a bucket or Volume v2 can look to a reader.

``FakeSpawner`` implements ``flora_deploy.jobs.spawner.Spawner``. Workers are plain callables
``worker(job_id, shard_id, config_hash, container_id, store) -> ShardReceipt``. Each container
start gets a fresh container id. Faults model Modal semantics:

- ``preempt``: the container dies mid-shard; Modal retries the *same call* in a new container.
- ``oom``: same as preempt (a crash); Modal retries indefinitely; we stop after ``max_restarts``
  and leave the call ``pending`` forever, exactly what a poison pill looks like from outside.
- ``fail``: the worker raises a Python exception (counts against ``modal.Retries``); after the
  retries are exhausted the call is ``failed``.
- ``expire``: the call's result is no longer retrievable.

Faults are queued per shard id: ``spawner.inject(shard_id, "preempt", times=2)``. Calls run when
``spawner.tick()`` is called (or immediately with ``eager=True``), so tests control interleaving.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from flora_deploy.jobs.spawner import CallStatus
from flora_deploy.jobs.types import ShardReceipt


class Preempted(Exception):
    """Raised inside a fake container to simulate SIGINT/SIGKILL mid-shard."""


class FakeStore:
    def __init__(self, lag: int = 0):
        self.lag = lag
        self._data: dict[str, bytes] = {}
        self._written_at: dict[str, int] = {}
        self.clock = 0
        self.writes: list[str] = []

    def tick(self) -> None:
        self.clock += 1

    def put_bytes(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes | bytearray):
            raise TypeError("whole-file bytes only")
        self._data[path] = bytes(data)
        self._written_at[path] = self.clock
        self.writes.append(path)

    def get_bytes(self, path: str) -> bytes:
        return self._data[path]

    def exists(self, path: str) -> bool:
        return path in self._data

    def list(self, prefix: str) -> Iterator[str]:
        for p in sorted(self._data):
            if p.startswith(prefix) and self.clock - self._written_at[p] >= self.lag:
                yield p


Worker = Callable[[str, str, str, str, FakeStore], ShardReceipt]


@dataclass
class _Call:
    call_id: str
    kind: str
    job_id: str
    shard_id: str
    config_hash: str
    state: str = "pending"
    receipt: ShardReceipt | None = None
    error: str | None = None
    starts: int = 0
    python_failures: int = 0
    done: bool = False
    containers: list[str] = field(default_factory=list)


class FakeSpawner:
    def __init__(self, store: FakeStore, workers: dict[str, Worker], *, eager: bool = False,
                 python_retries: int = 2, max_restarts: int = 20):
        self.store = store
        self.workers = workers
        self.eager = eager
        self.python_retries = python_retries
        self.max_restarts = max_restarts
        self.calls: dict[str, _Call] = {}
        self._faults: dict[str, deque[str]] = defaultdict(deque)
        self._ids = itertools.count(1)
        self._containers = itertools.count(1)

    # fault injection --------------------------------------------------------------------
    def inject(self, shard_id: str, fault: str, times: int = 1) -> None:
        assert fault in {"preempt", "oom", "fail", "expire"}
        self._faults[shard_id].extend([fault] * times)

    # Spawner protocol --------------------------------------------------------------------
    def spawn(self, kind: str, job_id: str, shard_id: str, config_hash: str) -> str:
        call = _Call(f"fc-{next(self._ids)}", kind, job_id, shard_id, config_hash)
        self.calls[call.call_id] = call
        if self.eager:
            self._run(call)
        return call.call_id

    def poll(self, call_id: str) -> CallStatus:
        c = self.calls[call_id]
        return CallStatus(call_id, c.state, receipt=c.receipt, error=c.error)  # type: ignore[arg-type]

    def cancel(self, call_id: str) -> None:
        c = self.calls[call_id]
        if c.state == "pending":
            c.state, c.done = "cancelled", True

    # execution ----------------------------------------------------------------------------
    def tick(self) -> None:
        """Run every pending call once (one container start each), then advance the store."""
        for c in list(self.calls.values()):
            if not c.done:
                self._run(c, once=True)
        self.store.tick()

    def _run(self, c: _Call, once: bool = False) -> None:
        while not c.done:
            if c.starts >= self.max_restarts:
                return  # looks like Modal's infinite crash-loop: pending forever
            c.starts += 1
            container = f"ta-{next(self._containers):05d}"
            c.containers.append(container)
            fault = self._faults[c.shard_id].popleft() if self._faults[c.shard_id] else None
            try:
                if fault in ("preempt", "oom"):
                    raise Preempted(fault)
                receipt = self.workers[c.kind](c.job_id, c.shard_id, c.config_hash, container,
                                               self.store)
            except Preempted:
                if once:
                    return
                continue  # Modal restarts the container for the same call
            except Exception as exc:  # user-code exception: counts against modal.Retries
                c.python_failures += 1
                if c.python_failures > self.python_retries:
                    c.state, c.error, c.done = "failed", repr(exc), True
                elif once:
                    return
                continue
            c.receipt, c.done = receipt, True
            c.state = "expired" if fault == "expire" else "success"
            if fault == "expire":
                c.receipt = None
            return
