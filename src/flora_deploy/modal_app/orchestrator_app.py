"""``flora-orchestrator`` Modal app: the job loop as a non-preemptible CPU function (PLAN.md §7/§8).

- ``run(job_id, factory)``: acquires the per-job Dict lease, drives the
  :class:`~flora_deploy.jobs.orchestrator.Orchestrator` with :class:`ModalSpawner` through
  :class:`TickDriver` (store refresh, budget hard cap, bounded retry of transient errors), and
  before the 23 h deadline spawns its own successor (``run.spawn``), which re-attaches to the
  in-flight calls recorded in each shard's ``call.json``. ``modal.Retries`` re-runs a generation
  that still dies of an error; the retried input keeps its holder (the call id), so it re-takes
  its own lease.
- ``reconcile(job_id, factory)``: stateless status; revives the orchestrator if it died.
- ``reconcile_watched()``: opt-in scheduled reconciler (``modal.Cron``, every 10 min). Deploy
  with ``FLORA_RECONCILE_CRON=1`` (or a cron string) to schedule it; jobs opt in with
  ``run(..., watch=True)``, which records them in the ``flora-orchestrator-watch`` Dict until they
  end. Reviving never yields two orchestrators: a revived ``run`` that finds the lease held exits
  without spawning anything.

``factory`` is a ``"module:function"`` path returning ``OrchestratorDeps`` for a job id (store,
settings, manifest/attempts/progress adapters, budget gate, worker app + class names); the CLI
(P1a-6) supplies it. Everything is defined at module scope (no ``serialized=True``).
Never run from tests.
"""

from __future__ import annotations

import builtins
import importlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import modal

from flora_deploy.jobs.budget import (
    BudgetState,
    CallRecord,
    apply_environment_budget,
    budget_summary,
)
from flora_deploy.jobs.orchestrator import (
    DEADLINE_S,
    POLL_INTERVAL_S,
    DictLease,
    Orchestrator,
    OrchestratorDeps,
    RunOutcome,
    ShardLedger,
    TickResult,
    utc,
    write_control,
)
from flora_deploy.jobs.reconciler import reconcile as reconcile_status
from flora_deploy.jobs.reconciler import revive_if_needed
from flora_deploy.jobs.spawner import CallStatus
from flora_deploy.jobs.types import ControlRecord, ShardReceipt
from flora_deploy.modal_app.images import cpu_image

APP_NAME = "flora-orchestrator"
LEASE_DICT = "flora-orchestrator-leases"
WATCH_DICT = "flora-orchestrator-watch"
TIMEOUT_S = 24 * 3600  # Modal's per-attempt maximum; the loop stops at DEADLINE_S (23 h)
CPU_CORES = 0.25
MEMORY_MIB = 1024
RECONCILE_EVERY = "*/10 * * * *"
MAX_CONSECUTIVE_ERRORS = 6  # transient tick errors in a row before the generation gives up
BACKOFF_S = 2.0  # 2, 4, 8, 16, 32, 60 s: 122 s in total, well inside the 300 s lease TTL
MAX_BACKOFF_S = 60.0

app = modal.App(APP_NAME)
leases = modal.Dict.from_name(LEASE_DICT, create_if_missing=True)
watched = modal.Dict.from_name(WATCH_DICT, create_if_missing=True)


# ----------------------------------------------------------------------------------------------
# FunctionCall.get(timeout=0) -> CallStatus (verified against modal 1.6.2's source)
# ----------------------------------------------------------------------------------------------
#
# modal 1.6.2, ``modal/_functions.py`` ``_Invocation.poll_function``:
#   no outputs and no unfinished inputs -> ``raise OutputExpiredError()`` (modal.exception)
#   no outputs yet                      -> ``raise TimeoutError()``  <- the *builtin*: the module
#                                          imports only ExecutionError, InvalidError,
#                                          NotFoundError and OutputExpiredError from .exception
# ``modal/_utils/function_utils.py`` ``_process_result``: GENERIC_STATUS_TIMEOUT ->
# ``FunctionTimeoutError``, INTERNAL_FAILURE -> ``InternalFailure``, else the remote exception.
# ``modal.exception.TimeoutError`` subclasses ``modal.exception.Error(Exception)``, NOT the
# builtin, and ``OutputExpiredError``/``FunctionTimeoutError`` subclass modal's TimeoutError, so
# neither family catches the other. ``tests/unit/test_orchestrator_modal.py`` re-reads the
# installed source and fails if any of this changes.


@dataclass(frozen=True)
class ModalErrors:
    """The exception classes ``FunctionCall.get(timeout=0)`` raises (injectable for tests).

    Checked in this order: ``output_expired``, ``function_timeout``, ``pending``, ``cancelled``,
    ``transient``; anything else is the call's own (remote) failure."""

    pending: tuple[type[BaseException], ...]  # still running (modal 1.6.2: builtin TimeoutError)
    output_expired: type[BaseException]
    function_timeout: type[BaseException]
    cancelled: tuple[type[BaseException], ...] = ()
    transient: tuple[type[BaseException], ...] = ()  # client/network errors: state unknown


def modal_errors() -> ModalErrors:
    from modal import exception as mx

    return ModalErrors(
        pending=(builtins.TimeoutError,),
        output_expired=mx.OutputExpiredError,
        function_timeout=mx.FunctionTimeoutError,
        cancelled=(mx.InputCancellation,),
        transient=(mx.ConnectionError, mx.ServiceError, mx.InternalError,
                   mx.ResourceExhaustedError),
    )


def status_from_get(call_id: str, get: Callable[[], Any], errors: ModalErrors) -> CallStatus:
    """Map one non-blocking ``get`` to a CallStatus.

    A pending call raises a bare ``TimeoutError()`` (no args). A builtin ``TimeoutError`` *with*
    a message is the worker's own exception re-raised (workers must not leak one), so it is a
    failure, not "pending forever". Transient client errors (network, Modal API 5xx) say nothing
    about the call: it stays ``pending`` and the next tick polls again."""
    try:
        result = get()
    except errors.output_expired:
        return CallStatus(call_id, "expired")
    except errors.function_timeout as exc:
        return CallStatus(call_id, "failed", error=f"FunctionTimeoutError: {exc}")
    except errors.pending as exc:
        if exc.args:
            return CallStatus(call_id, "failed", error=repr(exc))
        return CallStatus(call_id, "pending")
    except errors.cancelled as exc:  # InputCancellation is a BaseException
        return CallStatus(call_id, "cancelled", error=repr(exc))
    except errors.transient as exc:
        return CallStatus(call_id, "pending", error=f"transient: {exc!r}")
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

    def spawn(self, kind: str, job_id: str, shard_id: str, config_hash: str, *,
              token: str = "") -> str:
        """``token`` reaches ``run_shard(..., token=...)`` so duplicate spawns stay inert
        (``jobs.spawner`` spawn idempotency)."""
        fc = self._handle(kind).spawn(job_id, shard_id, config_hash, token=token)
        return fc.object_id

    def poll(self, call_id: str) -> CallStatus:
        fc = self._call_api.from_id(call_id)
        return status_from_get(call_id, lambda: fc.get(timeout=0), self.errors)

    def cancel(self, call_id: str) -> None:
        self._call_api.from_id(call_id).cancel()

    def terminate(self, call_id: str) -> None:
        """Cancel and stop the call's containers now (budget hard cap: stop the billing)."""
        self._call_api.from_id(call_id).cancel(terminate_containers=True)


# ----------------------------------------------------------------------------------------------
# Transient errors
# ----------------------------------------------------------------------------------------------

_TRANSIENT_NAMES = {
    # botocore / urllib3 (bucket stores)
    "EndpointConnectionError", "ConnectionClosedError", "ReadTimeoutError",
    "ConnectTimeoutError", "ProxyConnectionError", "ResponseStreamingError",
    # grpclib / modal client
    "GRPCError", "StreamTerminatedError", "ConnectionError", "ServiceError", "InternalError",
    "ResourceExhaustedError",
}
_TRANSIENT_S3_CODES = {"SlowDown", "ServiceUnavailable", "InternalError", "RequestTimeout",
                       "Throttling", "ThrottlingException", "RequestTimeTooSkewed", "500", "502",
                       "503", "504"}
_PERMANENT_OS = (PermissionError, IsADirectoryError, NotADirectoryError)


def is_transient(exc: BaseException) -> bool:
    """Worth retrying the tick: network / service hiccups of the store or the Modal API, and
    lagging reads (a listed file not yet readable). Programming and config errors are not."""
    if isinstance(exc, _PERMANENT_OS):
        return False
    if isinstance(exc, OSError):  # incl. builtin ConnectionError, TimeoutError, FileNotFound
        return True
    names = {c.__name__ for c in type(exc).__mro__}
    if names & _TRANSIENT_NAMES:
        return True
    if "ClientError" in names:  # botocore ClientError: retry throttling / 5xx only
        resp = getattr(exc, "response", None) or {}
        code = str((resp.get("Error") or {}).get("Code", ""))
        status = str((resp.get("ResponseMetadata") or {}).get("HTTPStatusCode", ""))
        return code in _TRANSIENT_S3_CODES or status in _TRANSIENT_S3_CODES
    return False


# ----------------------------------------------------------------------------------------------
# Budget hard cap (hook) and the per-tick driver
# ----------------------------------------------------------------------------------------------


def _ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def budget_state_from_ledgers(ledgers: Iterable[ShardLedger], now: float) -> BudgetState:
    """The gate's view of every call the orchestrator spawned, from ``call.json`` histories.

    A call starts at its ``spawned`` entry and ends at its first later entry (success, failed,
    expired, cancelled, cancelled_poison). A call with no end entry is live while it is the
    ledger's current call in a live status, and otherwise ended when the ledger was last saved."""
    calls: list[CallRecord] = []
    for led in ledgers:
        spawned: dict[str, float] = {}
        ended: dict[str, tuple[float, str]] = {}
        for e in led.calls:
            cid, t = e.get("call_id"), _ts(e.get("utc"))
            if not cid or t is None:
                continue
            if e.get("outcome") == "spawned":
                spawned.setdefault(cid, t)
            else:
                ended.setdefault(cid, (t, str(e.get("outcome"))))
        for cid, t0 in spawned.items():
            if cid in ended:
                t1, outcome = ended[cid]
                calls.append(CallRecord(cid, led.shard_id, led.kind, t0, t1, outcome))
            elif cid == led.call_id and led.status in ("spawning", "active"):
                calls.append(CallRecord(cid, led.shard_id, led.kind, t0))
            else:
                calls.append(CallRecord(cid, led.shard_id, led.kind, t0,
                                        _ts(led.updated_utc) or now, led.status))
    return BudgetState(now, tuple(calls))


def orchestrator_budget_state(orch: Orchestrator) -> BudgetState:
    """HOOK (Fix-J1): prefer ``Orchestrator.budget_state()`` once the core exposes one; until then
    this reads the orchestrator's loaded ledgers (``_ledgers``, the same objects ``tick`` saves)."""
    fn = getattr(orch, "budget_state", None)
    if callable(fn):
        return fn()
    return budget_state_from_ledgers(getattr(orch, "_ledgers", {}).values(), orch.clock())


class TickDriver:
    """Runs an :class:`Orchestrator` the way the Modal app needs. Per tick:

    1. ``store.refresh()`` (Volume stores: see other containers' commits; no-op elsewhere) and
       ``budget.refresh()`` (drop the gate's cached listings): nothing read is older than the tick.
    2. **Budget hard cap** (:meth:`enforce_budget`): ``budget.overspend(state)`` names live calls
       to cancel; the driver writes ``budget_stop`` first (so they are not respawned), then
       cancels them (``terminate`` when the spawner has it).
    3. ``orch.tick()``. A transient error (:func:`is_transient`) is retried with exponential
       backoff, at most ``max_consecutive_errors`` times in a row; anything else, or one error too
       many, propagates (``modal.Retries`` then re-runs the generation; the scheduled reconciler
       revives it after that).

    Everything else matches ``Orchestrator.run_until_idle``.
    """

    def __init__(self, orch: Orchestrator, *, max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
                 backoff_s: float = BACKOFF_S, max_backoff_s: float = MAX_BACKOFF_S):
        self.orch = orch
        self.max_consecutive_errors = max_consecutive_errors
        self.backoff_s, self.max_backoff_s = backoff_s, max_backoff_s
        self.errors = 0
        self.cancelled: set[str] = set()

    def _emit(self, event: str, **kw: Any) -> None:
        self.orch.events({"event": event, "job_id": self.orch.job_id, "t": self.orch.clock(),
                          "holder": self.orch.holder, **kw})

    def refresh(self) -> None:
        for obj in (self.orch.store, self.orch.budget):
            fn = getattr(obj, "refresh", None)
            if callable(fn):
                fn()

    def enforce_budget(self) -> list[str]:
        """Budget hard cap for orchestrators that don't enforce it themselves.

        The real ``Orchestrator`` runs ``enforce_budget`` inside ``tick`` (budget cancels are
        re-queued without consuming a respawn), so this driver-level fallback is a no-op for it
        and only serves stub orchestrators. Gate API: ``budget.overspend(BudgetState)``."""
        if callable(getattr(self.orch, "enforce_budget", None)):
            return []
        gate = self.orch.budget
        overspend = getattr(gate, "overspend", None)
        if not callable(overspend):
            return []
        state = orchestrator_budget_state(self.orch)
        ids = [c for c in overspend(state) if c not in self.cancelled]
        if not ids:
            return []
        detail = {**dict(getattr(gate, "last_overspend", {}) or {}), "cancel": ids}
        reason = "budget hard cap: spend + in-flight work exceeds the budget"
        control = getattr(self.orch, "_control", None)  # also makes it visible before listing
        if callable(control):
            control("budget_stop", reason, **detail)
        else:
            write_control(self.orch.store, self.orch.paths, ControlRecord(
                "budget_stop", self.orch.job_id, reason, utc(self.orch.clock()), detail))
        stop = getattr(self.orch.spawner, "terminate", None) or self.orch.spawner.cancel
        for cid in ids:
            try:
                stop(cid)
            except Exception as exc:  # retried next tick (not added to self.cancelled)
                self._emit("cancel_error", call_id=cid, error=repr(exc))
                continue
            self.cancelled.add(cid)
        self._emit("budget_overspend_cancel", calls=ids, **{k: v for k, v in detail.items()
                                                            if k != "cancel"})
        return ids

    def tick(self) -> TickResult:
        self.refresh()
        self.enforce_budget()
        return self.orch.tick()

    def run_until_idle(self, *, sleep: Callable[[float], None] = time.sleep,
                       poll_interval_s: float = POLL_INTERVAL_S,
                       max_ticks: int | None = None) -> RunOutcome:
        orch = self.orch
        if not orch.acquire():
            return RunOutcome("lease_held", 0, 0)
        ticks = 0
        try:
            while True:
                try:
                    result = self.tick()
                except Exception as exc:
                    if not is_transient(exc) or self.errors >= self.max_consecutive_errors:
                        self._emit("tick_error_fatal", error=repr(exc), consecutive=self.errors)
                        raise
                    self.errors += 1
                    delay = min(self.backoff_s * 2 ** (self.errors - 1), self.max_backoff_s)
                    self._emit("tick_error", error=repr(exc), consecutive=self.errors,
                               retry_in_s=delay)
                    sleep(delay)
                    continue
                self.errors = 0
                ticks += 1
                if result != "running":
                    return RunOutcome(result, ticks, orch.spawned)
                if max_ticks is not None and ticks >= max_ticks:
                    return RunOutcome("max_ticks", ticks, orch.spawned)
                sleep(poll_interval_s)
        finally:
            orch.release()


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


def _holder() -> str | None:
    """The function call id: stable across ``modal.Retries`` of one generation, so a retried
    input re-takes its own (unexpired) lease instead of waiting out the TTL."""
    try:
        cid = modal.current_function_call_id()
    except Exception:
        return None
    return f"orch-{cid}" if cid else None


def build_orchestrator(job_id: str, deps: OrchestratorDeps, *, lease_kv: Any,
                       events: Callable[[dict[str, Any]], None] = _print_event,
                       deadline_s: float = DEADLINE_S, holder: str | None = None) -> Orchestrator:
    return Orchestrator(
        settings=deps.settings, store=deps.store,
        spawner=ModalSpawner(deps.worker_app, deps.worker_classes),
        lease=DictLease(lease_kv, f"orchestrator/{job_id}"),
        manifests=deps.manifests, attempts=deps.attempts, progress=deps.progress,
        budget=deps.budget, events=events, deadline_s=deadline_s, holder=holder,
    )


TERMINAL = ("done", "failed", "cancelled", "budget_stopped")


def _retry(fn: Callable[[], Any], attempts: int = 3, delay_s: float = 5.0) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay_s * 2 ** i)
    return None


@app.function(image=cpu_image(), nonpreemptible=True, cpu=CPU_CORES, memory=MEMORY_MIB,
              timeout=TIMEOUT_S,
              retries=modal.Retries(max_retries=3, initial_delay=30.0, backoff_coefficient=2.0))
def run(job_id: str, factory: str, generation: int = 0, watch: bool = False) -> dict[str, Any]:
    """One orchestrator generation. Spawns its successor when the deadline is reached.

    ``watch=True`` registers the job with the scheduled reconciler (``reconcile_watched``)."""
    deps = load_factory(factory)(job_id)
    if watch:
        watched.put(job_id, factory)
    environment = os.environ.get("MODAL_ENVIRONMENT", "main")
    backstop = apply_environment_budget(environment, deps.settings)
    if generation == 0:
        _print_event({"event": "environment_budget", "job_id": job_id, **asdict(backstop)})
    orch = build_orchestrator(job_id, deps, lease_kv=leases, holder=_holder())
    outcome = TickDriver(orch).run_until_idle(sleep=time.sleep)  # releases the lease
    if outcome.result == "continue":
        successor = _retry(lambda: run.spawn(job_id, factory, generation + 1, watch))
        _print_event({"event": "continuation_spawned", "job_id": job_id,
                      "generation": generation + 1, "call_id": successor.object_id})
    elif outcome.result in TERMINAL and watch:
        watched.pop(job_id, None)
    return {**asdict(outcome), "generation": generation,
            "budget": budget_summary(environment, deps.settings, backstop, gate=deps.budget)}


def _reconcile_job(job_id: str, factory: str, revive: bool = True) -> dict[str, Any]:
    deps = load_factory(factory)(job_id)
    refresh = getattr(deps.store, "refresh", None)
    if callable(refresh):
        refresh()
    status = reconcile_status(
        job_id=job_id, config_hash=deps.settings.config_hash, store=deps.store,
        manifests=deps.manifests, attempts=deps.attempts,
        spawner=ModalSpawner(deps.worker_app, deps.worker_classes),
        lease=DictLease(leases, f"orchestrator/{job_id}"),
        max_starts=deps.settings.max_starts_before_quarantine,
    )
    # revive spawns ``run``; a second contender loses the lease and exits (never two loops)
    revived = revive and revive_if_needed(status, lambda: run.spawn(job_id, factory, 0, True))
    return {**asdict(status), "revived": bool(revived)}


@app.function(image=cpu_image(), cpu=CPU_CORES, memory=MEMORY_MIB, timeout=15 * 60)
def reconcile(job_id: str, factory: str, revive: bool = True) -> dict[str, Any]:
    """Rebuild status from files; restart the orchestrator if it is dead and work remains."""
    return _reconcile_job(job_id, factory, revive)


def reconcile_schedule(value: str | None = None) -> modal.Cron | None:
    """``FLORA_RECONCILE_CRON`` at deploy time: unset/0/off -> no schedule; 1/on/true -> every
    10 minutes; anything else is a cron string."""
    v = (os.environ.get("FLORA_RECONCILE_CRON", "") if value is None else value).strip()
    if v.lower() in ("", "0", "off", "false", "no"):
        return None
    return modal.Cron(RECONCILE_EVERY if v.lower() in ("1", "on", "true", "yes") else v)


def reconcile_all(entries: Iterable[tuple[str, str]], reconcile_one: Callable[..., dict],
                  forget: Callable[[str], Any]) -> dict[str, Any]:
    """Reconcile (and revive) every watched job; forget the ones that ended."""
    out: dict[str, Any] = {}
    for job_id, factory in entries:
        try:
            st = reconcile_one(job_id, factory, True)
        except Exception as exc:  # one broken job never stops the others
            out[job_id] = {"error": repr(exc)}
            continue
        out[job_id] = {"state": st.get("state"), "revived": st.get("revived")}
        if st.get("state") in TERMINAL:
            forget(job_id)
    return out


@app.function(image=cpu_image(), cpu=CPU_CORES, memory=MEMORY_MIB, timeout=15 * 60,
              schedule=reconcile_schedule())
def reconcile_watched() -> dict[str, Any]:
    """Scheduled reconciler (opt-in, see the module docstring): revives dead orchestrators."""
    result = reconcile_all(list(watched.items()), _reconcile_job,
                           lambda job_id: watched.pop(job_id, None))
    _print_event({"event": "reconcile_watched", "jobs": result})
    return result
