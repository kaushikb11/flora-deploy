"""``LocalSpawner``: the ``jobs.spawner.Spawner`` protocol in-process, on a thread pool.

It reproduces the Modal semantics the orchestrator depends on (``jobs/spawner.py``,
``modal_app/orchestrator_app.py::status_from_get``), so the *real* orchestrator, stores and
worker runners can run a whole job without Modal:

* ``spawn`` returns a call id at once and the call runs on a worker thread. ``poll`` is
  non-blocking: ``pending`` while it runs (or crash-loops), then ``success`` (with the receipt the
  runner returned; ``None`` for a superseded spawn, which the orchestrator treats as "no
  receipt"), ``failed``, ``expired`` or ``cancelled``. A call id this spawner never issued (the
  process that spawned it is gone) polls ``expired``: its result is unretrievable, which is what
  a lost Modal result looks like, and the orchestrator respawns it.
* Each container start has its own container id (``lc-00001`` ...). A container that survives a
  call (success or a Python exception) is kept *warm* and reused by the next call of its kind, as
  Modal does, so the runners' same-container part numbering (``_next_k``) is exercised too.
* Preemption (``KeyboardInterrupt`` reaching the spawner: the runner already flushed and marked
  the start ``preempted``) and container crashes (:class:`~flora_deploy.local.faults.
  ContainerCrash`, no mark) restart the *same call* in a new container, indefinitely (bounded
  by ``max_restarts`` so a test can never hang; past it the call stays ``pending``, which is how a
  Modal crash loop looks from outside). Python exceptions count against ``python_retries``
  (``modal.Retries(max_retries=2)`` of the worker classes); past them the call is ``failed``.
* ``cancel`` / ``terminate``: the call polls ``cancelled`` immediately; the running container gets
  SIGINT at its next checkpoint (the runner flushes and marks it preempted, as ``@modal.exit``
  does) and is not restarted.
* Faults (``infra``, ``spawnerr`` here; the rest in the engines' checkpoints) come from a
  :class:`~flora_deploy.local.faults.FaultPlan`.

The spawner never touches the store itself: everything the orchestrator learns comes from the
runners' files and ``poll``, exactly as on Modal.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from flora_deploy.jobs.spawner import CallStatus
from flora_deploy.jobs.types import ShardReceipt
from flora_deploy.local.faults import ContainerContext, ContainerCrash, FaultPlan, set_current

PYTHON_RETRIES = 2  # modal_app.factory.WORKER_RETRIES["max_retries"]
MAX_RESTARTS = 50


class LocalSpawnError(ConnectionError):
    """The ``spawnerr`` fault: a transient client error from ``spawn``."""


class ShardRunner:  # the runner interface the spawner drives (Segcap/RewardShardRunner)
    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None: ...


# kind, container_id, call_id getter -> a runner bound to that container
RunnerFactory = Callable[[str, str, Callable[[], str]], ShardRunner]


@dataclass
class LocalCall:
    call_id: str
    kind: str
    job_id: str
    shard_id: str
    config_hash: str
    token: str = ""
    state: str = "pending"
    receipt: ShardReceipt | None = None
    error: str | None = None
    containers: list[str] = field(default_factory=list)
    preemptions: int = 0
    crashes: int = 0
    python_failures: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Container:
    container_id: str
    kind: str
    runner: ShardRunner
    calls: int = 0


class LocalSpawner:
    """See the module docstring. ``runners(kind, container_id, call_id)`` builds the worker
    runner of one container (called once per container start)."""

    def __init__(self, runners: RunnerFactory, *, faults: FaultPlan | None = None,
                 max_workers: int = 8, python_retries: int = PYTHON_RETRIES,
                 max_restarts: int = MAX_RESTARTS, restart_delay_s: float = 0.02,
                 warm: bool = True, name: str | None = None):
        self.runners = runners
        self.faults = faults or FaultPlan()
        self.python_retries, self.max_restarts = python_retries, max_restarts
        self.restart_delay_s, self.warm = restart_delay_s, warm
        self.name = name or uuid.uuid4().hex[:8]
        self.calls: dict[str, LocalCall] = {}
        self._ids = itertools.count(1)
        self._cids = itertools.count(1)
        self._idle: dict[str, list[_Container]] = defaultdict(list)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="flora-local-call")
        self._closed = False

    # Spawner protocol -----------------------------------------------------------------------
    def spawn(self, kind: str, job_id: str, shard_id: str, config_hash: str, *,
              token: str = "") -> str:
        if self._closed:
            raise RuntimeError("spawner is closed")
        if self.faults.take("spawnerr", shard_id=shard_id):
            raise LocalSpawnError(f"injected transient spawn error for {shard_id}")
        call = LocalCall(f"lfc-{self.name}-{next(self._ids):05d}", kind, job_id, shard_id,
                         config_hash, token)
        with self._lock:
            self.calls[call.call_id] = call
        if self.faults.take("infra", shard_id=shard_id):
            self._end(call, "expired", error="injected infra fault: call input lost")
        else:
            self._pool.submit(self._run_call, call)
        return call.call_id

    def poll(self, call_id: str) -> CallStatus:
        with self._lock:
            c = self.calls.get(call_id)
            if c is None:
                return CallStatus(call_id, "expired",
                                  error="unknown call (spawned by another process)")
            return CallStatus(call_id, c.state, receipt=c.receipt,  # type: ignore[arg-type]
                              error=c.error)

    def cancel(self, call_id: str) -> None:
        with self._lock:
            c = self.calls.get(call_id)
            if c is None:
                return
            c.cancel.set()
            if c.state == "pending":
                c.state, c.error = "cancelled", "cancelled"

    def terminate(self, call_id: str) -> None:
        """``FunctionCall.cancel(terminate_containers=True)``: same as :meth:`cancel` here."""
        self.cancel(call_id)

    # lifecycle ----------------------------------------------------------------------------
    def wait_idle(self, timeout_s: float = 30.0) -> bool:
        """Wait until no call is running (cancelled calls included)."""
        deadline = time.monotonic() + timeout_s
        for c in list(self.calls.values()):
            left = deadline - time.monotonic()
            if left <= 0 or not c.finished.wait(left):
                return False
        return True

    def close(self, *, cancel: bool = True, timeout_s: float = 30.0) -> None:
        """Stop: cancel every live call (``cancel=True``), wait for the threads, shut down."""
        if cancel:
            for cid in list(self.calls):
                self.cancel(cid)
        self.wait_idle(timeout_s)
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self.calls.values())
        states: dict[str, int] = defaultdict(int)
        for c in calls:
            states[c.state] += 1
        return {"calls": len(calls), "states": dict(sorted(states.items())),
                "containers": next(self._cids) - 1,
                "preemptions": sum(c.preemptions for c in calls),
                "crashes": sum(c.crashes for c in calls),
                "python_failures": sum(c.python_failures for c in calls)}

    # execution ----------------------------------------------------------------------------
    def _end(self, c: LocalCall, state: str, *, receipt: ShardReceipt | None = None,
             error: str | None = None) -> None:
        with self._lock:
            if c.state == "pending":  # a cancel that came first wins (Modal: get() raises)
                c.state, c.receipt, c.error = state, receipt, error
        c.finished.set()

    def _container(self, kind: str, call: LocalCall) -> _Container:
        with self._lock:
            if self.warm and self._idle[kind]:
                return self._idle[kind].pop()
            cid = f"lc-{self.name}-{next(self._cids):05d}"
        runner = self.runners(kind, cid, _current_call_id)
        return _Container(cid, kind, runner)

    def _release(self, box: _Container) -> None:
        if self.warm and not self._closed:
            with self._lock:
                self._idle[box.kind].append(box)

    def _run_call(self, c: LocalCall) -> None:
        try:
            self._loop(c)
        except BaseException as exc:  # noqa: BLE001 - never kill the pool thread
            self._end(c, "failed", error=f"local spawner error: {exc!r}")
        finally:
            set_current(None)
            c.finished.set()

    def _loop(self, c: LocalCall) -> None:
        starts = 0
        while True:
            if c.cancel.is_set():
                self._end(c, "cancelled", error="cancelled")
                return
            if starts >= self.max_restarts:
                c.finished.set()  # crash loop: stays ``pending`` until cancelled
                while not c.cancel.wait(0.05):
                    if self._closed:
                        return
                self._end(c, "cancelled", error="cancelled")
                return
            starts += 1
            box = self._container(c.kind, c)
            c.containers.append(box.container_id)
            set_current(ContainerContext(call_id=c.call_id, container_id=box.container_id,
                                         shard_id=c.shard_id, shard_kind=c.kind,
                                         faults=self.faults, cancelled=c.cancel))
            try:
                box.calls += 1
                receipt = box.runner.run_shard(c.job_id, c.shard_id, c.config_hash,
                                               token=c.token)
            except KeyboardInterrupt:  # SIGINT: preempted, or our own cancel
                if c.cancel.is_set():
                    self._end(c, "cancelled", error="cancelled")
                    return
                c.preemptions += 1
                time.sleep(self.restart_delay_s)
                continue  # Modal restarts the same call in a new container
            except ContainerCrash:  # OOM / SIGKILL: no mark; Modal restarts it, forever
                c.crashes += 1
                time.sleep(self.restart_delay_s)
                continue
            except Exception as exc:  # user-code exception: modal.Retries
                c.python_failures += 1
                self._release(box)  # the container survives a failed input
                if c.python_failures > self.python_retries:
                    self._end(c, "failed", error=repr(exc))
                    return
                time.sleep(self.restart_delay_s)
                continue
            finally:
                set_current(None)
            self._release(box)
            self._end(c, "success", receipt=receipt)
            return


def _current_call_id() -> str:
    from flora_deploy.local.faults import current

    ctx = current()
    return ctx.call_id if ctx is not None else ""


# --- lease KV ----------------------------------------------------------------------------------


class FileKV:
    """The ``modal.Dict`` subset ``orchestrator.DictLease`` needs, as one JSON file per key in a
    directory: ``put(skip_if_exists=True)`` is an ``O_CREAT | O_EXCL`` create (atomic), other
    writes replace the file atomically. Survives process restarts, so a re-run sees the lease of
    an orchestrator that died (and waits out its TTL) exactly like the Modal one."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: Any) -> Path:
        return self.root / (sha256(json.dumps(key).encode()).hexdigest()[:40] + ".json")

    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool:
        p = self._path(key)
        data = json.dumps({"key": key, "value": value}).encode()
        if skip_if_exists:
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                return False
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return True
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return True

    def get(self, key: Any, default: Any | None = None) -> Any:
        try:
            return json.loads(self._path(key).read_bytes())["value"]
        except (FileNotFoundError, ValueError, KeyError):
            return default

    def contains(self, key: Any) -> bool:
        return self._path(key).is_file()
