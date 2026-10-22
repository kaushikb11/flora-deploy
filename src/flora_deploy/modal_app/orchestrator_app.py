"""``flora-orchestrator`` Modal app: the job loop as a small preemptible CPU function (PLAN.md §2).

Resources, preemptibility and the worker-pool drain / pre-warm: ``jobs.pools`` (module
docstring), applied here through :class:`ModalAutoscaler`.

- ``run(job_id, factory)``: acquires the per-job Dict lease, drives the
  :class:`~flora_deploy.jobs.orchestrator.Orchestrator` with :class:`ModalSpawner` through
  :class:`TickDriver` (store refresh, budget hard cap, bounded retry of transient errors), and
  before the 23 h deadline spawns its own successor (``run.spawn``), which re-attaches to the
  in-flight calls recorded in each shard's ``call.json``. ``modal.Retries`` re-runs a generation
  that still dies of an error; the retried input keeps its holder (the call id), so it re-takes
  its own lease.
  At job end, if the orchestrator reports a worker class that could not start
  (``Orchestrator.stop_app_reason``), the worker app is stopped (:func:`maybe_stop_worker_app`)
  so Modal stops restarting the crashing class; never while another job's pool entry is busy.
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
from typing import Any

import modal

from flora_deploy.jobs.budget import (
    apply_environment_budget,
    budget_summary,
)

# The per-tick driver lives in ``jobs.driver`` (no Modal import, so local runs use it too);
# re-exported here for the Modal app and its callers.
from flora_deploy.jobs.driver import (  # noqa: E402,F401
    BACKOFF_S,
    MAX_BACKOFF_S,
    MAX_CONSECUTIVE_ERRORS,
    TickDriver,
    budget_state_from_ledgers,
    is_transient,
    orchestrator_budget_state,
)
from flora_deploy.jobs.orchestrator import (
    DEADLINE_S,
    DictLease,
    Orchestrator,
    OrchestratorDeps,
)
from flora_deploy.jobs.pools import (
    ORCHESTRATOR_CPU,
    ORCHESTRATOR_MEMORY_LIMIT_MIB,
    ORCHESTRATOR_MEMORY_MIB,
    ORCHESTRATOR_PREEMPTIBLE,
    ORCHESTRATOR_SCALEDOWN_S,
    POOL_DICT_ENV,
    PoolScaler,
    drains_path,
)
from flora_deploy.jobs.reconciler import reconcile as reconcile_status
from flora_deploy.jobs.reconciler import revive_if_needed
from flora_deploy.jobs.spawner import CallStatus
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import ShardReceipt, safe_key
from flora_deploy.modal_app.images import cpu_image

APP_NAME = "flora-orchestrator"
LEASE_DICT = "flora-orchestrator-leases"
WATCH_DICT = "flora-orchestrator-watch"
TIMEOUT_S = 24 * 3600  # Modal's per-attempt maximum; the loop stops at DEADLINE_S (23 h)
POOL_DICT = "flora-orchestrator-pools"  # this module's app; factory apps: ``<watch>-pools``
CPU_CORES = ORCHESTRATOR_CPU
MEMORY_MIB = ORCHESTRATOR_MEMORY_MIB  # request (billed at max(request, use))
MEMORY = (ORCHESTRATOR_MEMORY_MIB, ORCHESTRATOR_MEMORY_LIMIT_MIB)  # (request, limit)
RECONCILE_EVERY = "*/10 * * * *"

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
    # A call stopped with ``cancel(terminate_containers=True)`` surfaces as a generic
    # ``RemoteError(result.exception)`` carrying this server message (seen on modal 1.6.2 by the
    # canary), not as ``InputCancellation``. Only an exact match counts as cancelled.
    remote: tuple[type[BaseException], ...] = ()
    cancelled_messages: tuple[str, ...] = ()


def modal_errors() -> ModalErrors:
    from modal import exception as mx

    return ModalErrors(
        pending=(builtins.TimeoutError,),
        output_expired=mx.OutputExpiredError,
        function_timeout=mx.FunctionTimeoutError,
        cancelled=(mx.InputCancellation,),
        transient=(mx.ConnectionError, mx.ServiceError, mx.InternalError,
                   mx.ResourceExhaustedError),
        remote=(mx.RemoteError,),
        cancelled_messages=("Function call was cancelled by user or a failure.",),
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
        if (errors.remote and isinstance(exc, errors.remote)
                and str(exc).strip() in errors.cancelled_messages):
            return CallStatus(call_id, "cancelled", error=repr(exc))
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


class ModalAutoscaler:
    """``jobs.pools.Autoscaler`` over ``Cls.from_name(app, cls)().update_autoscaler`` (modal
    1.6.2; it warns as deprecated in favour of ``Function.autoscaler.update``, which a ``Cls``
    instance does not expose yet)."""

    def __init__(self, app_name: str, *, cls_api: Any = None):
        self.app_name = app_name
        self._cls_api = cls_api or modal.Cls

    def _update(self, cls_name: str, **kw: Any) -> None:
        import warnings

        obj = self._cls_api.from_name(self.app_name, cls_name)()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj.update_autoscaler(**kw)

    def set_scaledown(self, cls_name: str, seconds: int) -> None:
        self._update(cls_name, scaledown_window=int(seconds))

    def set_min_containers(self, cls_name: str, n: int) -> None:
        self._update(cls_name, min_containers=int(n))


def build_pools(job_id: str, deps: OrchestratorDeps, *, kv: Any, autoscaler: Any,
                events: Callable[[dict[str, Any]], None]) -> PoolScaler:
    """The job's ``PoolScaler``: drains on the deployed classes, pre-warm timed by the budget
    gate's estimator (when it has one)."""
    est = getattr(deps.budget, "estimator", None)
    pools = deps.settings.pools
    startup = getattr(est, "startup_s", None)
    work = None
    if est is not None:
        def work(shard: Any) -> float:
            return float(est.container_seconds(shard, pools[shard.kind]))
    return PoolScaler(
        job_id=job_id, classes=deps.worker_classes, static_s=deps.settings.scaledown_s,
        autoscaler=autoscaler, kv=kv, events=events, store=deps.store,
        path=drains_path(JobPaths(job_id).root, deps.settings.config_hash),
        startup_s=startup, work_s=work)


# ----------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------


# The deps factories a ``factory`` argument or a watch-Dict entry may select (review 11b F4): a
# Dict value never names arbitrary code for the scheduled reconciler to call.
FACTORIES: frozenset[str] = frozenset({"flora_deploy.sdk:deployed_deps"})


def load_factory(path: str) -> Callable[[str], OrchestratorDeps]:
    mod, _, fn = path.partition(":")
    if not mod or not fn:
        raise ValueError(f"factory must look like 'package.module:function', got {path!r}")
    if path not in FACTORIES:
        raise ValueError(f"factory {path!r} is not one of the allowed {sorted(FACTORIES)}")
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
                       deadline_s: float = DEADLINE_S, holder: str | None = None,
                       pools: PoolScaler | None = None) -> Orchestrator:
    # the no-start watchdog's expected startup per kind: the budget gate's estimator
    est = getattr(deps.budget, "estimator", None)
    startup = getattr(est, "startup_s", None)
    return Orchestrator(
        pools=pools, startup_s=startup if callable(startup) else None,
        settings=deps.settings, store=deps.store,
        spawner=ModalSpawner(deps.worker_app, deps.worker_classes),
        lease=DictLease(lease_kv, f"orchestrator/{job_id}"),
        manifests=deps.manifests, attempts=deps.attempts, progress=deps.progress,
        budget=deps.budget, events=events, deadline_s=deadline_s, holder=holder,
    )


TERMINAL = ("done", "completed_with_errors", "failed", "cancelled", "budget_stopped")


def stop_modal_app(app_name: str) -> None:
    """``modal app stop <app> --yes`` through the installed client (modal 1.6.2:
    ``modal.cli.app.stop``, a blocking click command; ``callback`` is the function itself).
    The environment is the container's (``MODAL_ENVIRONMENT``). An app that is already stopped
    exits with ``SystemExit``, which is not an error here."""
    from modal.cli.app import stop

    try:
        stop.callback(app_name, yes=True, env=os.environ.get("MODAL_ENVIRONMENT") or None)
    except SystemExit:
        pass


def maybe_stop_worker_app(orch: Any, outcome: Any, *, app_name: str, pools: Any,
                          stopper: Callable[[str], None] = stop_modal_app,
                          events: Callable[[dict[str, Any]], None] = _print_event
                          ) -> dict[str, Any] | None:
    """Job end: stop the worker app when ``orch.stop_app_reason()`` names a class that cannot
    start (or a budget-cancelled call nothing ever started), so Modal stops restarting it.

    Skipped while another job's pool entry says one of the classes is busy (one app serves every
    job of its config), and without an app name. The factory app's orchestrator runs inside the
    worker app, so stopping it ends this container too: everything is already written (lease
    released, controls and ledgers saved) and the event is printed first. Resuming such a job
    needs ``flora deploy`` again."""
    if getattr(outcome, "result", None) not in TERMINAL or not app_name:
        return None
    reason_of = getattr(orch, "stop_app_reason", None)
    reason = reason_of() if callable(reason_of) else None
    if not reason:
        return None
    busy_of = getattr(pools, "busy_jobs", None)
    busy = list(busy_of()) if callable(busy_of) else []
    ev = {"event": "worker_app_stop", "job_id": orch.job_id, "app": app_name, "reason": reason,
          "result": outcome.result}
    if busy:
        events({**ev, "event": "worker_app_stop_skipped", "busy_jobs": busy})
        return {**ev, "stopped": False, "busy_jobs": busy}
    events(ev)
    try:
        stopper(app_name)
    except Exception as exc:  # noqa: BLE001 - reported; the job outcome stands
        events({**ev, "event": "worker_app_stop_error", "error": repr(exc)})
        return {**ev, "stopped": False, "error": repr(exc)}
    return {**ev, "stopped": True}


def _retry(fn: Callable[[], Any], attempts: int = 3, delay_s: float = 5.0) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay_s * 2 ** i)
    return None


@app.function(image=cpu_image(), nonpreemptible=not ORCHESTRATOR_PREEMPTIBLE, cpu=CPU_CORES,
              memory=MEMORY, timeout=TIMEOUT_S, scaledown_window=ORCHESTRATOR_SCALEDOWN_S,
              retries=modal.Retries(max_retries=3, initial_delay=30.0, backoff_coefficient=2.0))
def run(job_id: str, factory: str, generation: int = 0, watch: bool = False) -> dict[str, Any]:
    """One orchestrator generation. Spawns its successor when the deadline is reached.

    ``watch=True`` registers the job with the scheduled reconciler (``reconcile_watched``)."""
    return run_generation(job_id, factory, generation, watch, spawn_successor=run.spawn)


def run_generation(job_id: str, factory: str, generation: int = 0, watch: bool = False, *,
                   spawn_successor: Callable[..., Any], watch_kv: Any = None,
                   pool_kv: Any = None, autoscaler: Any = None,
                   app_stopper: Callable[[str], None] = stop_modal_app) -> dict[str, Any]:
    """``run``'s body. ``spawn_successor(job_id, factory, generation, watch)`` starts the next
    generation in the caller's own app; ``watch_kv`` (default ``watched``) is the reconciler's
    watch list. The app factory (``modal_app.factory``) passes its own app's handles.
    ``pool_kv`` / ``autoscaler`` (default: the app's pool Dict, ``ModalAutoscaler`` on
    ``deps.worker_app``) drive ``jobs.pools``; with no worker classes there is nothing to drain."""
    watched_kv = watched if watch_kv is None else watch_kv
    deps = load_factory(factory)(job_id)
    if watch:
        watched_kv.put(job_id, factory)
    environment = os.environ.get("MODAL_ENVIRONMENT", "main")
    backstop = apply_environment_budget(environment, deps.settings)
    if generation == 0:
        _print_event({"event": "environment_budget", "job_id": job_id, **asdict(backstop)})
    pools = None
    if deps.worker_app and deps.worker_classes:
        kv = pool_kv if pool_kv is not None else modal.Dict.from_name(
            os.environ.get(POOL_DICT_ENV) or POOL_DICT, create_if_missing=True)
        pools = build_pools(job_id, deps, kv=kv, events=_print_event,
                            autoscaler=autoscaler or ModalAutoscaler(deps.worker_app))
    orch = build_orchestrator(job_id, deps, lease_kv=leases, holder=_holder(), pools=pools)
    outcome = TickDriver(orch).run_until_idle(sleep=time.sleep)  # releases the lease
    if outcome.result == "continue":
        successor = _retry(lambda: spawn_successor(job_id, factory, generation + 1, watch))
        _print_event({"event": "continuation_spawned", "job_id": job_id,
                      "generation": generation + 1, "call_id": successor.object_id})
    elif outcome.result in TERMINAL and watch:
        watched_kv.pop(job_id, None)
    result = {**asdict(outcome), "generation": generation,
              "budget": budget_summary(environment, deps.settings, backstop, gate=deps.budget)}
    maybe_stop_worker_app(orch, outcome, app_name=deps.worker_app, pools=pools,
                          stopper=app_stopper)  # last: it may end this container
    return result


def _reconcile_job(job_id: str, factory: str, revive: bool = True, *,
                   spawn_run: Callable[..., Any] | None = None) -> dict[str, Any]:
    """``spawn_run(job_id, factory, generation, watch)`` (default ``run.spawn``) revives."""
    spawn = run.spawn if spawn_run is None else spawn_run
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
        max_error_rate=deps.settings.max_episode_error_rate,
    )
    # revive spawns ``run``; a second contender loses the lease and exits (never two loops)
    revived = revive and revive_if_needed(status, lambda: spawn(job_id, factory, 0, True))
    return {**asdict(status), "revived": bool(revived)}


@app.function(image=cpu_image(), cpu=CPU_CORES, memory=MEMORY, timeout=15 * 60,
              scaledown_window=ORCHESTRATOR_SCALEDOWN_S)
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
            safe_key(job_id)  # Dict keys are job ids; never a path or an argument otherwise
            st = reconcile_one(job_id, factory, True)
        except Exception as exc:  # one broken job never stops the others
            out[job_id] = {"error": repr(exc)}
            continue
        out[job_id] = {"state": st.get("state"), "revived": st.get("revived")}
        if st.get("state") in TERMINAL:
            forget(job_id)
    return out


@app.function(image=cpu_image(), cpu=CPU_CORES, memory=MEMORY, timeout=15 * 60,
              scaledown_window=ORCHESTRATOR_SCALEDOWN_S, schedule=reconcile_schedule())
def reconcile_watched() -> dict[str, Any]:
    """Scheduled reconciler (opt-in, see the module docstring): revives dead orchestrators."""
    result = reconcile_all(list(watched.items()), _reconcile_job,
                           lambda job_id: watched.pop(job_id, None))
    _print_event({"event": "reconcile_watched", "jobs": result})
    return result
