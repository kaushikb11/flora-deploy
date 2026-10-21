"""``orchestrator_app.TickDriver`` and friends, offline: store/gate refresh before every tick,
the budget hard-cap hook (stub orchestrator; Fix-J1 owns the real core), bounded retry of
transient errors (H5), the ledger -> BudgetState mapping and the scheduled reconciler."""

from __future__ import annotations

import builtins
import inspect

import pytest
from modal import exception as mx

from flora_deploy.jobs.budget import BudgetState, CallRecord
from flora_deploy.jobs.orchestrator import ShardLedger, utc
from flora_deploy.jobs.store import JobPaths
from flora_deploy.modal_app.orchestrator_app import (
    TickDriver,
    budget_state_from_ledgers,
    is_transient,
    reconcile_all,
    reconcile_schedule,
)
from tests.fakes.jobs import Harness

T0 = 1_800_000_000.0


class StubGate:
    def __init__(self, to_cancel=()):
        self.to_cancel = list(to_cancel)
        self.states: list[BudgetState] = []
        self.refreshes = 0
        self.last_overspend = {"spend_usd": 9.0, "cap_usd": 5.0}

    def refresh(self):
        self.refreshes += 1

    def overspend(self, state):
        self.states.append(state)
        return list(self.to_cancel)


class StubSpawner:
    def __init__(self):
        self.cancelled, self.terminated, self.fail = [], [], set()

    def cancel(self, cid):
        self.cancelled.append(cid)

    def terminate(self, cid):
        if cid in self.fail:
            raise mx.ConnectionError("blip")
        self.terminated.append(cid)


class StubStore:
    def __init__(self):
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1
        return True


class StubOrch:
    """The parts of ``Orchestrator`` the driver touches."""

    def __init__(self, gate, ticks=("running", "done"), raises=()):
        self.budget, self.store, self.spawner = gate, StubStore(), StubSpawner()
        self.job_id, self.holder, self.paths = "job1", "orch-x", JobPaths("job1")
        self.events_log: list[dict] = []
        self.controls: list[tuple] = []
        self.spawned = 0
        self._ledgers: dict[str, ShardLedger] = {}
        self._ticks, self._raises = list(ticks), list(raises)
        self.held = False
        self.order: list[str] = []

    def clock(self):
        return T0

    def events(self, e):
        self.events_log.append(e)

    def _control(self, kind, reason, **detail):
        self.controls.append((kind, reason, detail))

    def acquire(self):
        self.held = True
        return True

    def release(self):
        self.held = False

    def tick(self):
        self.order.append(f"tick(store={self.store.refreshes},gate={self.budget.refreshes})")
        if self._raises:
            exc = self._raises.pop(0)
            if exc is not None:
                raise exc
        return self._ticks.pop(0)


def _ledger(shard, status, call_id, entries, updated=T0):
    led = ShardLedger(shard, "segcap", status=status, call_id=call_id, updated_utc=utc(updated))
    led.calls = [{"call_id": c, "outcome": o, "utc": utc(t)} for c, o, t in entries]
    return led


def test_budget_state_from_ledgers():
    leds = [
        _ledger("s0", "active", "fc-2", [("fc-1", "spawned", T0 - 900),
                                         ("fc-1", "failed", T0 - 600),
                                         ("fc-2", "spawned", T0 - 500)]),
        _ledger("s1", "done", None, [("fc-3", "spawned", T0 - 400),
                                     ("fc-3", "success", T0 - 100)]),
        _ledger("s2", "cancelled", "fc-4", [("fc-4", "spawned", T0 - 300)], updated=T0 - 50),
    ]
    st = budget_state_from_ledgers(leds, T0)
    got = {c.call_id: c for c in st.calls}
    assert got["fc-1"] == CallRecord("fc-1", "s0", "segcap", T0 - 900, T0 - 600, "failed")
    assert got["fc-2"].live and got["fc-2"].spawned_at == T0 - 500
    assert got["fc-3"].outcome == "success" and got["fc-3"].ended_at == T0 - 100
    assert got["fc-4"].ended_at == T0 - 50 and got["fc-4"].outcome == "cancelled"
    assert [c.call_id for c in st.live()] == ["fc-2"]


def test_driver_refreshes_store_and_gate_before_every_tick():
    orch = StubOrch(StubGate(), ticks=("running", "running", "done"))
    out = TickDriver(orch).run_until_idle(sleep=lambda s: None)
    assert out.result == "done" and out.ticks == 3 and not orch.held
    assert orch.order == ["tick(store=1,gate=1)", "tick(store=2,gate=2)", "tick(store=3,gate=3)"]


def test_driver_hard_cap_writes_budget_stop_then_terminates_calls_once():
    gate = StubGate(to_cancel=["fc-1", "fc-2"])
    orch = StubOrch(gate, ticks=("running", "running", "budget_stopped"))
    orch._ledgers = {"s0": _ledger("s0", "active", "fc-1", [("fc-1", "spawned", T0 - 10)])}
    orch.spawner.fail = {"fc-2"}
    out = TickDriver(orch).run_until_idle(sleep=lambda s: None)
    assert out.result == "budget_stopped"
    assert gate.states[0].live()[0].call_id == "fc-1"  # the gate saw the ledgers' calls
    kind, reason, detail = orch.controls[0]
    assert kind == "budget_stop" and "hard cap" in reason and detail["cancel"] == ["fc-1", "fc-2"]
    assert detail["spend_usd"] == 9.0  # the gate's numbers ride along in the control record
    # fc-1 is terminated once; fc-2 failed to cancel and is retried on every later tick
    assert orch.spawner.terminated == ["fc-1"]
    assert sum(e["event"] == "cancel_error" for e in orch.events_log) == 3
    assert [c[2]["cancel"] for c in orch.controls[1:]] == [["fc-2"], ["fc-2"]]
    assert any(e["event"] == "budget_overspend_cancel" for e in orch.events_log)


def test_driver_without_overspend_api_is_a_noop():
    class Plain:
        pass

    orch = StubOrch(Plain(), ticks=("done",))
    orch.budget.refreshes = 0
    assert TickDriver(orch).enforce_budget() == []
    assert orch.controls == []


def test_driver_retries_transient_errors_with_bounded_backoff():
    orch = StubOrch(StubGate(), ticks=("running", "done"),
                    raises=[OSError("S3 503 SlowDown"), mx.ConnectionError("reset"), None, None])
    sleeps: list[float] = []
    out = TickDriver(orch).run_until_idle(sleep=sleeps.append, poll_interval_s=15.0)
    assert out.result == "done" and out.ticks == 2
    assert sleeps == [2.0, 4.0, 15.0]
    assert [e["event"] for e in orch.events_log].count("tick_error") == 2


def test_driver_gives_up_after_max_consecutive_errors_and_on_bugs():
    orch = StubOrch(StubGate(), raises=[OSError("down")] * 10)
    sleeps: list[float] = []
    with pytest.raises(OSError):
        TickDriver(orch, max_consecutive_errors=4).run_until_idle(sleep=sleeps.append)
    assert sleeps == [2.0, 4.0, 8.0, 16.0] and not orch.held  # lease released for a successor
    full = StubOrch(StubGate(), raises=[OSError("down")] * 10)
    waits: list[float] = []
    with pytest.raises(OSError):
        TickDriver(full).run_until_idle(sleep=waits.append)
    assert sum(waits) < 300  # all default retries fit inside the 300 s lease TTL
    orch = StubOrch(StubGate(), raises=[ValueError("no pool for kind")])
    with pytest.raises(ValueError):
        TickDriver(orch).run_until_idle(sleep=lambda s: None)
    assert any(e["event"] == "tick_error_fatal" for e in orch.events_log)


def test_is_transient_classification():
    class ClientError(Exception):
        def __init__(self, code, status=400):
            self.response = {"Error": {"Code": code},
                             "ResponseMetadata": {"HTTPStatusCode": status}}

    class EndpointConnectionError(Exception):
        pass

    assert is_transient(OSError("503")) and is_transient(builtins.TimeoutError())
    assert is_transient(FileNotFoundError("listed, not yet readable"))
    assert is_transient(ClientError("SlowDown", 503)) and is_transient(ClientError("x", 500))
    assert not is_transient(ClientError("AccessDenied", 403))
    assert is_transient(EndpointConnectionError()) and is_transient(mx.ServiceError("x"))
    assert not is_transient(PermissionError("denied")) and not is_transient(ValueError("bug"))
    assert not is_transient(KeyError("x"))


def test_real_orchestrator_survives_a_transient_store_error():
    """The review's probe: one S3 503 out of ``exists`` used to kill the orchestrator."""
    h = Harness({"s0": ["a"], "s1": ["b"]})
    real = h.store.exists
    n = {"k": 0}

    def flaky(p):
        n["k"] += 1
        if n["k"] == 6:
            raise OSError("S3 503 SlowDown")
        return real(p)

    h.store.exists = flaky
    out = TickDriver(h.orchestrator()).run_until_idle(sleep=h.sleep, poll_interval_s=10.0,
                                                      max_ticks=100)
    assert out.result == "done"
    assert "tick_error" in h.names()


def test_reconcile_schedule_is_opt_in():
    assert reconcile_schedule("") is None and reconcile_schedule("0") is None
    assert reconcile_schedule("off") is None
    assert reconcile_schedule("1") is not None and reconcile_schedule("*/5 * * * *") is not None


def test_reconcile_all_revives_and_forgets_finished_jobs():
    seen, forgot = [], []

    def one(job_id, factory, revive):
        seen.append((job_id, factory, revive))
        if job_id == "broken":
            raise OSError("store down")
        return {"state": {"a": "running", "b": "done"}[job_id], "revived": job_id == "a"}

    out = reconcile_all([("a", "m:f"), ("b", "m:f"), ("broken", "m:f")], one, forgot.append)
    assert out["a"] == {"state": "running", "revived": True}
    assert out["b"]["state"] == "done" and forgot == ["b"]
    assert "store down" in out["broken"]["error"]
    assert all(r for _, _, r in seen)


def test_app_source_wires_retries_schedule_driver_and_summary():
    from flora_deploy.modal_app import orchestrator_app as oa

    src = inspect.getsource(oa)
    assert "retries=modal.Retries(" in src
    assert "schedule=reconcile_schedule()" in src
    assert "TickDriver(orch).run_until_idle" in src
    assert "budget_summary(" in src  # the Environment-budget instruction is in the run summary
