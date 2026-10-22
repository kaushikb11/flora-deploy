"""Fail fast on worker startup crashes (2026-10-22 noise-12): a worker class that crashes in
``@modal.enter`` never writes an attempt marker, so its call sat ``pending`` until the budget cap.
Startup-failure markers, the no-start watchdog and the job-end app stop end it early."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pytest

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs.orchestrator import NO_START, STARTUP_FAILED, utc
from flora_deploy.jobs.settings import PoolSettings
from flora_deploy.jobs.store import startup_failure_path
from flora_deploy.jobs.types import Shard
from tests.fakes.jobs import CFG, JOB, FakeAttempts, Harness, WorkerBehavior, make_worker
from tests.fakes.modal import FakeStore, Preempted

APP = "flora-app1"
ONE = {"segcap": PoolSettings(gpu="H200", max_containers=1, max_inputs=1)}


class StartupAttempts(FakeAttempts):
    """FakeAttempts plus the two startup methods, over the real marker files."""

    def call_started(self, shard: Shard, token: str, since_utc: str) -> bool:
        return any(True for _ in self.store.list(self._prefix(shard)))

    def startup_failures(self, kind: str, since_utc: str) -> list[A.StartupFailure]:
        return A.read_startup_failures(self.store, APP, kind=kind, since_utc=since_utc)


def worker_env(container: str) -> dict[str, str]:
    return {"FLORA_APP_NAME": APP, "FLORA_SHARD_KINDS": "segcap", "FLORA_CONFIG_HASH": CFG,
            "MODAL_TASK_ID": container}


def harness(spec, *, crash_enter=False, silent=False, startup_s=None, **kw) -> Harness:
    h = Harness(spec, pools=ONE, **kw)
    h.attempts = StartupAttempts(h.store)
    ok = make_worker(h.manifests, h.behavior, "segcap", h.processed)

    def worker(job_id, shard_id, cfg, container, store):
        if crash_enter or silent:
            if crash_enter:  # what the proposed @modal.enter wrapper does, then Modal restarts
                A.record_startup_failure(
                    store, RuntimeError("tempseg: receipt rejected: fingerprint changed (ino)"),
                    env=worker_env(container), now=h.clock())
            raise Preempted("enter crashed")
        return ok(job_id, shard_id, cfg, container, store)

    h.spawner.workers["segcap"] = worker
    h.startup_s = startup_s
    return h


def orch(h: Harness, **kw):
    return h.orchestrator(startup_s=h.startup_s, **kw)


def test_crash_looping_startup_fails_the_shard_with_the_worker_error():
    h = harness({"segcap-0": ["a"], "segcap-1": ["b"]}, crash_enter=True)
    o = orch(h)
    out = h.run(o)
    assert out.result == "failed"  # not budget_stopped, not pending until a cap
    first = next(c for c in h.spawner.calls.values() if c.shard_id == "segcap-0")
    assert first.state == "cancelled"
    assert first.starts <= 4  # stopped after the limit, plus listing lag
    rec = h.attempts.failed()["segcap-0"]
    assert rec.reason == STARTUP_FAILED and "fingerprint changed (ino)" in rec.detail["error"]
    # the second shard of the broken class is never spawned, and fails with the same reason
    assert h.spawn_counts() == {"segcap-0": 1}
    assert h.attempts.failed()["segcap-1"].detail.get("not_spawned") is True
    assert "worker_startup_failed" in h.names()
    assert o.stop_app_reason() and "fingerprint changed (ino)" in o.stop_app_reason()
    assert not h.attempts.quarantined() and not h.attempts.poison_records  # never bisected


def test_startup_failures_before_the_spawn_do_not_count():
    h = harness({"segcap-0": ["a"]})
    old = utc(h.clock() - 3600)
    for i in range(5):
        A.write_startup_failure(h.store, A.StartupFailure(APP, f"ta-old{i}", old, "x",
                                                          ("segcap",)))
    assert h.run(orch(h)).result == "done"


def test_no_start_watchdog_fails_a_call_that_never_starts():
    h = harness({"segcap-0": ["a"]}, silent=True, startup_s=lambda kind: 300.0)
    o = orch(h)
    assert o.acquire()
    t0 = h.clock()
    result = "running"
    while result == "running":
        result = o.tick()
        if result == "running":
            h.sleep(10)
    assert result == "failed"
    ev = next(e for e in h.events if e["event"] == NO_START)
    assert ev["t"] - t0 >= 900  # 3 x the expected 300 s startup
    assert ev["t"] - t0 < 900 + 30
    assert h.attempts.failed()["segcap-0"].reason == NO_START
    assert o.stop_app_reason()


def test_watchdog_has_a_floor_and_is_off_without_an_estimate():
    h = harness({"segcap-0": ["a"]}, silent=True, startup_s=lambda kind: 10.0)
    o = orch(h)
    h.run(o, max_ticks=55)  # 540 s: under the 600 s floor
    assert NO_START not in h.names()
    h2 = harness({"segcap-0": ["a"]}, silent=True)
    assert h2.run(orch(h2), max_ticks=300).result == "max_ticks"  # pending, as before
    assert NO_START not in h2.names()


def test_watchdog_never_touches_a_started_call():
    h = harness({"segcap-0": ["a"]}, startup_s=lambda kind: 10.0,
                behavior=WorkerBehavior(crash_after_marker={"segcap-0": 100}),
                max_starts_before_quarantine=1000)
    h.spawner.max_restarts = 1000
    h.run(orch(h), max_ticks=150)  # 1500 s of a started (crash-looping after its marker) call
    assert NO_START not in h.names() and STARTUP_FAILED not in h.names()


def test_healthy_job_is_unchanged():
    h = harness({"segcap-0": ["a"], "segcap-1": ["b"]}, startup_s=lambda kind: 300.0)
    o = orch(h)
    assert h.run(o).result == "done"
    assert o.stop_app_reason() is None


@dataclass
class CapOnce:
    """Budget gate whose hard cap cancels every live call once."""

    fired: bool = False
    last_overspend: dict | None = None

    def admit(self, shard, in_flight):
        from flora_deploy.jobs.orchestrator import BudgetDecision

        return BudgetDecision(True)

    def overspend(self, state) -> list[str]:
        live = [c.call_id for c in state.calls if c.live]
        if live and not self.fired:
            self.fired = True
            return live
        return []


def test_budget_stop_of_a_never_started_call_asks_to_stop_the_app():
    h = harness({"segcap-0": ["a"]}, silent=True, budget=CapOnce())
    o = orch(h)
    assert h.run(o).result == "budget_stopped"
    entry = next(e for led in o._ledgers.values() for e in led.calls
                 if e.get("outcome") == "cancelled_budget")
    assert entry["started"] is False
    assert "no container ever started" in o.stop_app_reason()


def test_budget_stop_of_a_started_call_does_not():
    h = harness({"segcap-0": ["a"]}, budget=CapOnce(),
                behavior=WorkerBehavior(crash_after_marker={"segcap-0": 100}),
                max_starts_before_quarantine=1000)
    h.spawner.max_restarts = 1000
    o = orch(h)  # the cap fires on the tick after the spawn, once the marker is visible
    assert h.run(o).result == "budget_stopped"
    entry = next(e for led in o._ledgers.values() for e in led.calls
                 if e.get("outcome") == "cancelled_budget")
    assert entry["started"] is True
    assert o.stop_app_reason() is None


# --- attempts: the marker files ---------------------------------------------------------------

def test_record_startup_failure_roundtrip_and_filters():
    st = FakeStore()
    t = 1_800_000_000.0
    p = A.record_startup_failure(st, ValueError("boom"), env=worker_env("ta-1"), now=t)
    assert p == startup_failure_path(APP, "ta-1")
    A.record_startup_failure(st, ValueError("late"), env={**worker_env("ta-2"),
                                                          "FLORA_SHARD_KINDS": "caption"},
                             now=t + 10)
    got = A.read_startup_failures(st, APP, kind="segcap")
    assert [(f.container_id, f.error, f.kinds) for f in got] == [
        ("ta-1", "ValueError: boom", ("segcap",))]
    assert A.read_startup_failures(st, APP, since_utc=utc(t + 5))[0].container_id == "ta-2"
    assert A.read_startup_failures(st, "other-app") == []


def test_record_startup_failure_never_raises():
    class Broken:
        def put_bytes(self, path, data):
            raise OSError("store down")

    assert A.record_startup_failure(Broken(), ValueError("x"), env=worker_env("ta-1")) is None
    assert A.record_startup_failure(FakeStore(), ValueError("x"), env={}) is None  # no app


def test_store_attempt_ledger_startup_methods():
    from flora_deploy.jobs.deps import StoreAttemptLedger

    st = FakeStore()
    jf = A.JobFiles(st, JOB, CFG, stage_hashes={})
    shard = Shard("segcap-0", "segcap", ("a",), 1.0)
    led = StoreAttemptLedger(jf, limit=2, app=APP)
    t0 = utc(1_800_000_000.0)
    assert not led.call_started(shard, "1.abc", t0)
    A.write_attempt(jf, "segcap", "segcap-0", "ta-9", started_utc=t0, orchestrator_attempt=1,
                    token="1.abc")
    assert led.call_started(shard, "1.abc", t0)
    assert not led.call_started(shard, "2.def", t0)
    A.record_startup_failure(st, RuntimeError("x"), env=worker_env("ta-3"), now=1_800_000_001.0)
    assert [f.container_id for f in led.startup_failures("segcap", t0)] == ["ta-3"]
    assert StoreAttemptLedger(jf, limit=2).startup_failures("segcap", t0) == []


# --- orchestrator_app: stopping the worker app at job end ---------------------------------------

@dataclass
class _Outcome:
    result: str


class _Orch:
    job_id = JOB

    def __init__(self, reason):
        self.reason = reason

    def stop_app_reason(self):
        return self.reason


class _Pools:
    def __init__(self, busy=()):
        self.busy = list(busy)

    def busy_jobs(self):
        return self.busy


def test_maybe_stop_worker_app():
    from flora_deploy.modal_app.orchestrator_app import maybe_stop_worker_app

    stopped: list[str] = []
    events: list[dict[str, Any]] = []
    kw = dict(app_name=APP, stopper=stopped.append, events=events.append)
    assert maybe_stop_worker_app(_Orch("segcap: x"), _Outcome("failed"), pools=_Pools(),
                                 **kw)["stopped"]
    assert stopped == [APP] and events[-1]["event"] == "worker_app_stop"
    # another job is busy on the shared app: never stop it
    r = maybe_stop_worker_app(_Orch("segcap: x"), _Outcome("budget_stopped"),
                              pools=_Pools(["job2"]), **kw)
    assert not r["stopped"] and stopped == [APP]
    # nothing wrong, or not a terminal outcome, or no app: nothing to do
    assert maybe_stop_worker_app(_Orch(None), _Outcome("failed"), pools=None, **kw) is None
    assert maybe_stop_worker_app(_Orch("x"), _Outcome("continue"), pools=None, **kw) is None
    assert maybe_stop_worker_app(_Orch("x"), _Outcome("failed"), pools=None,
                                 **{**kw, "app_name": ""}) is None

    def boom(app):
        raise RuntimeError("api down")

    r = maybe_stop_worker_app(_Orch("x"), _Outcome("failed"), pools=None,
                              **{**kw, "stopper": boom})
    assert not r["stopped"] and "api down" in r["error"]


def test_stop_modal_app_uses_the_installed_cli_command():
    """``stop_modal_app`` calls ``modal.cli.app.stop.callback(app, yes=True, env=...)``."""
    from modal.cli.app import stop

    params = inspect.signature(stop.callback).parameters
    assert {"app_identifier", "yes", "env"} <= set(params)


def test_pool_scaler_busy_jobs():
    from flora_deploy.jobs.pools import PoolScaler
    from tests.fakes.jobs import FakeClock

    class KV(dict):
        def put(self, k, v):
            self[k] = v

        def items(self):  # noqa: D102
            return list(super().items())

    clock = FakeClock()
    kv = KV()
    ps = PoolScaler(job_id=JOB, classes={"segcap": "Ref"}, static_s=300, autoscaler=None,
                    kv=kv, clock=clock)
    kv.put("Ref/job/other", {"busy": True, "t": clock()})
    kv.put(f"Ref/job/{JOB}", {"busy": True, "t": clock()})
    kv.put("Ref/job/stale", {"busy": True, "t": clock() - 10_000})
    assert ps.busy_jobs() == ["other"]


@pytest.mark.parametrize("reason", [STARTUP_FAILED, NO_START])
def test_failed_ledger_reasons_survive_a_restart(reason):
    h = harness({"segcap-0": ["a"]}, crash_enter=reason == STARTUP_FAILED,
                silent=reason == NO_START, startup_s=lambda kind: 300.0)
    assert h.run(orch(h)).result == "failed"
    h.clock.advance(1000)
    fresh = orch(h)  # a successor generation rebuilds the reason from call.json
    fresh.acquire()
    assert fresh.tick() == "failed"
    assert fresh.stop_app_reason() and reason in fresh.stop_app_reason()
