"""``flora-orchestrator`` Modal app: the job loop as a non-preemptible CPU function (PLAN.md §7/§8).

- ``run(job_id, factory)``: acquires the per-job Dict lease, runs
  :meth:`~flora_deploy.jobs.orchestrator.Orchestrator.run_until_idle` with :class:`ModalSpawner`,
  and before the 23 h deadline spawns its own successor (``run.spawn``), which re-attaches to the
  in-flight calls recorded in each shard's ``call.json``.
- ``reconcile(job_id, factory)``: stateless status; revives the orchestrator if it died.

``factory`` is a ``"module:function"`` path returning ``OrchestratorDeps`` for a job id (store,
settings, manifest/attempts/progress adapters, budget gate, worker app + class names); the CLI
(P1a-6) supplies it. Everything is defined at module scope (no ``serialized=True``).
Never run from tests.
"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import modal

from flora_deploy.jobs.orchestrator import (
    DEADLINE_S,
    DictLease,
    Orchestrator,
    OrchestratorDeps,
)
from flora_deploy.jobs.reconciler import reconcile as reconcile_status
from flora_deploy.jobs.reconciler import revive_if_needed
from flora_deploy.jobs.spawner import CallStatus
from flora_deploy.jobs.types import ShardReceipt
from flora_deploy.modal_app.images import cpu_image

APP_NAME = "flora-orchestrator"
LEASE_DICT = "flora-orchestrator-leases"
TIMEOUT_S = 24 * 3600  # Modal's per-attempt maximum; the loop stops at DEADLINE_S (23 h)
CPU_CORES = 0.25
MEMORY_MIB = 1024

app = modal.App(APP_NAME)
leases = modal.Dict.from_name(LEASE_DICT, create_if_missing=True)


# ----------------------------------------------------------------------------------------------
# FunctionCall.get(timeout=0) -> CallStatus, with the TimeoutError-subclass trap handled
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalErrors:
    """The exception classes ``FunctionCall.get`` can raise (injectable for unit tests)."""

    timeout: type[BaseException]  # still pending
    output_expired: type[BaseException]  # subclass of ``timeout``!
    function_timeout: type[BaseException]  # subclass of ``timeout``!
    cancelled: tuple[type[BaseException], ...] = ()


def modal_errors() -> ModalErrors:
    from modal import exception as mx

    return ModalErrors(timeout=mx.TimeoutError, output_expired=mx.OutputExpiredError,
                       function_timeout=mx.FunctionTimeoutError,
                       cancelled=(mx.InputCancellation,))


def status_from_get(call_id: str, get: Callable[[], Any], errors: ModalErrors) -> CallStatus:
    """Map one non-blocking ``get`` to a CallStatus.

    ``OutputExpiredError`` and ``FunctionTimeoutError`` subclass Modal's ``TimeoutError``, which is
    also what a still-pending call raises; they MUST be caught first, or expired and timed-out calls
    look pending forever.
    """
    try:
        result = get()
    except errors.output_expired:
        return CallStatus(call_id, "expired")
    except errors.function_timeout as exc:
        return CallStatus(call_id, "failed", error=f"FunctionTimeoutError: {exc}")
    except errors.timeout:
        return CallStatus(call_id, "pending")
    except errors.cancelled as exc:  # InputCancellation is a BaseException
        return CallStatus(call_id, "cancelled", error=repr(exc))
    except Exception as exc:  # the remote exception, re-raised (retries exhausted, init failure)
        return CallStatus(call_id, "failed", error=repr(exc))
    return CallStatus(call_id, "success", receipt=coerce_receipt(result))


def coerce_receipt(value: Any) -> ShardReceipt | None:
    if value is None or isinstance(value, ShardReceipt):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        d = dict(value)
        for k in ("episodes_done", "episodes_failed", "parts"):
            d[k] = tuple(d.get(k, ()))
        return ShardReceipt(**d)
    raise TypeError(f"unexpected shard result type {type(value).__name__}")


class ModalSpawner:
    """``Spawner`` over deployed worker classes: ``Cls.from_name(app, cls)().run_shard.spawn``."""

    def __init__(self, worker_app: str, worker_classes: Mapping[str, str], *,
                 method: str = "run_shard", errors: ModalErrors | None = None,
                 cls_api: Any = None, call_api: Any = None):
        self.worker_app = worker_app
        self.worker_classes = dict(worker_classes)
        self.method = method
        self.errors = errors or modal_errors()
        self._cls_api = cls_api or modal.Cls
        self._call_api = call_api or modal.FunctionCall
        self._handles: dict[str, Any] = {}

    def _handle(self, kind: str) -> Any:
        if kind not in self._handles:
            try:
                name = self.worker_classes[kind]
            except KeyError:
                raise ValueError(f"no worker class configured for shard kind {kind!r}") from None
            cls = self._cls_api.from_name(self.worker_app, name)
            self._handles[kind] = getattr(cls(), self.method)
        return self._handles[kind]

    def spawn(self, kind: str, job_id: str, shard_id: str, config_hash: str) -> str:
        fc = self._handle(kind).spawn(job_id, shard_id, config_hash)
        return fc.object_id

    def poll(self, call_id: str) -> CallStatus:
        fc = self._call_api.from_id(call_id)
        return status_from_get(call_id, lambda: fc.get(timeout=0), self.errors)

    def cancel(self, call_id: str) -> None:
        self._call_api.from_id(call_id).cancel()


# ----------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------


def load_factory(path: str) -> Callable[[str], OrchestratorDeps]:
    mod, _, fn = path.partition(":")
    if not mod or not fn:
        raise ValueError(f"factory must look like 'package.module:function', got {path!r}")
    return getattr(importlib.import_module(mod), fn)


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True, default=str), flush=True)


def build_orchestrator(job_id: str, deps: OrchestratorDeps, *, lease_kv: Any,
                       events: Callable[[dict[str, Any]], None] = _print_event,
                       deadline_s: float = DEADLINE_S) -> Orchestrator:
    return Orchestrator(
        settings=deps.settings, store=deps.store,
        spawner=ModalSpawner(deps.worker_app, deps.worker_classes),
        lease=DictLease(lease_kv, f"orchestrator/{job_id}"),
        manifests=deps.manifests, attempts=deps.attempts, progress=deps.progress,
        budget=deps.budget, events=events, deadline_s=deadline_s,
    )


@app.function(image=cpu_image(), nonpreemptible=True, cpu=CPU_CORES, memory=MEMORY_MIB,
              timeout=TIMEOUT_S)
def run(job_id: str, factory: str, generation: int = 0) -> dict[str, Any]:
    """One orchestrator generation. Spawns its successor when the deadline is reached."""
    deps = load_factory(factory)(job_id)
    orch = build_orchestrator(job_id, deps, lease_kv=leases)
    outcome = orch.run_until_idle(sleep=time.sleep)  # releases the lease before returning
    if outcome.result == "continue":
        successor = run.spawn(job_id, factory, generation + 1)
        _print_event({"event": "continuation_spawned", "job_id": job_id,
                      "generation": generation + 1, "call_id": successor.object_id})
    return {**asdict(outcome), "generation": generation}


@app.function(image=cpu_image(), cpu=CPU_CORES, memory=MEMORY_MIB, timeout=15 * 60)
def reconcile(job_id: str, factory: str, revive: bool = True) -> dict[str, Any]:
    """Rebuild status from files; restart the orchestrator if it is dead and work remains."""
    deps = load_factory(factory)(job_id)
    status = reconcile_status(
        job_id=job_id, config_hash=deps.settings.config_hash, store=deps.store,
        manifests=deps.manifests, attempts=deps.attempts,
        spawner=ModalSpawner(deps.worker_app, deps.worker_classes),
        lease=DictLease(leases, f"orchestrator/{job_id}"),
        max_starts=deps.settings.max_starts_before_quarantine,
    )
    revived = revive and revive_if_needed(status, lambda: run.spawn(job_id, factory))
    return {**asdict(status), "revived": bool(revived)}
