"""Orchestrator lifecycle: kill + restart (re-attach), lease contention, cancel, budget stop and
resume, continuation before the deadline, and the stateless reconciler (status + revive)."""

from __future__ import annotations

from flora_deploy.jobs.reconciler import needs_orchestrator, reconcile, revive_if_needed
from tests.fakes.jobs import CFG, JOB, FakeBudget, Harness, WorkerBehavior, control


def _spec(n: int, eps: int = 2) -> dict[str, list[str]]:
    return {f"segcap-{i:03d}": [f"ds--e{i}-{j}" for j in range(eps)] for i in range(n)}


def _status(h: Harness, **kw):
    return reconcile(job_id=JOB, config_hash=CFG, store=h.store, manifests=h.manifests,
                     attempts=h.attempts, lease=h.lease(), clock=h.clock, **kw)


def test_killed_orchestrator_restarts_and_reattaches_without_duplicate_spawns():
    h = Harness(_spec(6), lag=2)
    h.spawner.inject("segcap-000", "preempt", times=1)
    a = h.orchestrator(holder="A")
    assert a.acquire()
    assert a.tick() == "running"  # spawns 2
    in_flight = set(h.spawn_counts())
    assert len(in_flight) == 2
    del a  # killed: no release, no in-memory state survives

    b = h.orchestrator(holder="B")
    assert b.run_until_idle(sleep=h.sleep, max_ticks=5).result == "lease_held"
    assert sum(h.spawn_counts().values()) == 2  # B spawned nothing while A's lease was live

    st = _status(h, spawner=h.spawner)
    assert st.state == "running" and st.active_calls == 2 and st.orchestrator_alive
    assert st.orchestrator is not None and st.orchestrator.holder == "A"

    h.clock.advance(400)  # A's lease (TTL 300 s) expires
    st = _status(h)
    assert needs_orchestrator(st)
    launched: list[str] = []
    assert revive_if_needed(st, lambda: launched.append("run"))
    assert launched == ["run"]

    out = h.run(h.orchestrator(holder="C"))
    assert out.result == "done"
    assert h.spawn_counts() == dict.fromkeys(_spec(6), 1)  # no shard spawned twice
    reattached = {e["shard_id"] for e in h.events if e["event"] == "reattached"}
    assert reattached == in_flight
    assert len(h.processed) == 12

    st = _status(h)
    assert st.state == "done" and st.counts["segcap"]["done"] == 6
    assert not revive_if_needed(st, lambda: launched.append("again"))


def test_lease_contention_second_orchestrator_refuses_and_live_one_keeps_running():
    h = Harness(_spec(3), lag=1)
    a, b = h.orchestrator(holder="A"), h.orchestrator(holder="B")
    assert a.acquire()
    assert not b.acquire()
    assert b.run_until_idle(sleep=h.sleep, max_ticks=3).result == "lease_held"
    assert h.run(a).result == "done"
    assert h.spawn_counts() == dict.fromkeys(_spec(3), 1)
    held = [e for e in h.events if e["event"] == "lease_held"]
    assert held and all(e["by"] == "A" and e["holder"] == "B" for e in held)


def test_orchestrator_that_lost_its_lease_stops_spawning():
    h = Harness(_spec(4), lag=1)
    a = h.orchestrator(holder="A")
    assert a.acquire() and a.tick() == "running"
    h.clock.advance(400)  # A stalled past its TTL...
    b = h.orchestrator(holder="B")
    assert b.acquire()  # ...and B took over (new lease generation)
    n = sum(h.spawn_counts().values())
    assert a.tick() == "lost_lease"
    assert sum(h.spawn_counts().values()) == n
    assert h.run(b).result == "done"
    assert h.spawn_counts() == dict.fromkeys(_spec(4), 1)


def test_cancel_control_cancels_live_calls_and_stops():
    h = Harness(_spec(4), lag=1, behavior=WorkerBehavior(oom={"ds--e0-0", "ds--e1-0"}))
    orch = h.orchestrator()
    assert orch.acquire() and orch.tick() == "running"
    control(h.store, "cancel", h.clock())
    h.sleep(10)  # listing lag 1: visible now
    assert orch.tick() == "cancelled"
    assert all(c.state == "cancelled" for c in h.spawner.calls.values())
    assert sum(h.spawn_counts().values()) == 2
    st = _status(h)
    assert st.state == "cancelled" and st.counts["segcap"]["cancelled"] == 2
    assert not needs_orchestrator(st)


def test_budget_stop_drains_then_resume_finishes():
    budget = FakeBudget(limit=3)
    h = Harness(_spec(6), lag=1, budget=budget)
    out = h.run()
    assert out.result == "budget_stopped"
    assert sum(h.spawn_counts().values()) == 3
    assert {s for s, _ in h.processed} == set(h.spawn_counts())  # in-flight shards drained
    assert h.names().count("budget_stop") == 1
    st = _status(h)
    assert st.state == "budget_stopped" and st.active_calls == 0
    assert not needs_orchestrator(st)

    # restarting without a resume record stays stopped and spawns nothing
    h.clock.advance(5)
    assert h.run().result == "budget_stopped"
    assert sum(h.spawn_counts().values()) == 3

    budget.limit = None  # the operator raised the budget, then `flora resume`
    h.clock.advance(5)
    control(h.store, "resume", h.clock())
    h.clock.advance(5)
    h.store.tick()
    assert _status(h).state == "running" and needs_orchestrator(_status(h))
    assert h.run().result == "done"
    assert h.spawn_counts() == dict.fromkeys(_spec(6), 1)


def test_continuation_before_deadline_hands_over_cleanly():
    h = Harness(_spec(8), lag=1)
    first = h.orchestrator(holder="gen0", deadline_s=25.0)
    out = h.run(first)
    assert out.result == "continue"
    assert h.lease().current() is None  # released for the successor
    n = sum(h.spawn_counts().values())
    assert 0 < n < 8
    out = h.run(h.orchestrator(holder="gen1"))
    assert out.result == "done"
    assert h.spawn_counts() == dict.fromkeys(_spec(8), 1)


def test_reconciler_reports_poison_quarantine_and_stuck_calls():
    h = Harness({"segcap-000": ["a", "b"], "segcap-001": ["c"]}, lag=0,
                behavior=WorkerBehavior(oom={"b", "c"}))
    orch = h.orchestrator()
    assert orch.acquire() and orch.tick() == "running"
    for _ in range(3):
        h.sleep(10)
    # markers piled up but the orchestrator has not ticked: the reconciler flags it
    st = _status(h, spawner=h.spawner)
    assert {s.shard_id for s in st.stuck_calls} == {"segcap-000", "segcap-001"}
    assert all(s.starts >= 3 for s in st.stuck_calls)
    assert h.run(orch).result == "done"
    st = _status(h)
    assert st.state == "done"
    assert st.quarantined == ("segcap-000",)
    assert st.poisoned == ("segcap-001", "segcap-ep-b")
    assert len(st.errors) == 2 and st.stuck_calls == ()


def test_crash_between_spawn_intent_and_call_id_is_respawned():
    from flora_deploy.jobs.orchestrator import ShardLedger, call_path
    from flora_deploy.jobs.store import JobPaths

    h = Harness(_spec(1), lag=1)
    led = ShardLedger("segcap-000", "segcap", status="spawning", spawn_n=1)
    h.store.put_bytes(call_path(JobPaths(JOB), CFG, "segcap", "segcap-000"), led.to_bytes())
    assert h.run().result == "done"
    assert "orphan_spawn_possible" in h.names() and h.spawn_counts() == {"segcap-000": 1}


def test_reconciler_planned_and_failed_states():
    h = Harness(_spec(2), lag=0, behavior=WorkerBehavior(init_fail={"segcap-001": 99}))
    assert _status(h).state == "planned"
    assert h.run(h.orchestrator(max_respawns=0)).result == "failed"
    st = _status(h)
    assert st.state == "failed" and st.failed == ("segcap-001",)
    assert st.counts["segcap"]["done"] == 1
