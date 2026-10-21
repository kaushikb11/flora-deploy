"""Per-tick orchestrator driver, shared by the Modal app and local runs (no Modal import).

:class:`TickDriver` (store/gate refresh, budget hard-cap fallback, bounded retry of transient
tick errors), :func:`is_transient`, and the budget views built from the orchestrator's ledgers
(:func:`budget_state_from_ledgers`, :func:`orchestrator_budget_state`).
``modal_app.orchestrator_app`` re-exports all of them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from flora_deploy.jobs.budget import BudgetState, CallRecord
from flora_deploy.jobs.orchestrator import (
    POLL_INTERVAL_S,
    Orchestrator,
    RunOutcome,
    ShardLedger,
    TickResult,
    utc,
    write_control,
)
from flora_deploy.jobs.types import ControlRecord

MAX_CONSECUTIVE_ERRORS = 6  # transient tick errors in a row before the generation gives up
BACKOFF_S = 2.0  # 2, 4, 8, 16, 32, 60 s: 122 s in total, well inside the 300 s lease TTL
MAX_BACKOFF_S = 60.0


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
