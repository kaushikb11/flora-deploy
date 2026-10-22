"""``jobs.pools``: drain a worker class when its work is over, restore it before work, pre-warm a
dependent pool, and what the budget charges for each (fakes only; no Modal)."""

from __future__ import annotations

import pytest

from flora_deploy.jobs.budget import (
    BudgetState,
    CallRecord,
    ManifestEstimator,
    ProjectionBudgetGate,
    calls_cost,
    pinned_rates,
)
from flora_deploy.jobs.pools import (
    DRAINED_SCALEDOWN_S,
    PoolScaler,
    drains_path,
    first_ready_at,
    idle_tail_s,
    kind_demand,
)
from flora_deploy.jobs.settings import PoolSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import Shard
from tests.fakes.jobs import CFG, JOB, FakeClock, Harness, make_settings

SEG_CAP = {"segment": PoolSettings(gpu="L40S", max_containers=1),
           "caption": PoolSettings(gpu="H100!", max_containers=1)}
CLASSES = {"segment": "SegmentWorker", "caption": "CaptionWorker"}


class FakeAutoscaler:
    def __init__(self, clock=None, fail: int = 0):
        self.calls: list[tuple[str, str, int, float | None]] = []
        self.window: dict[str, int] = {}
        self.min: dict[str, int] = {}
        self.clock, self.fail = clock, fail

    def _t(self):
        return self.clock() if self.clock else None

    def set_scaledown(self, cls, seconds):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("modal api down")
        self.calls.append(("scaledown", cls, seconds, self._t()))
        self.window[cls] = seconds

    def set_min_containers(self, cls, n):
        self.calls.append(("min", cls, n, self._t()))
        self.min[cls] = n


class KV(dict):
    def put(self, key, value):
        self[key] = value

    def items(self):  # noqa: D401 - modal.Dict API
        return list(super().items())


def _scaler(h, auto, kv, **kw):
    return PoolScaler(job_id=JOB, classes=CLASSES, static_s=300, autoscaler=auto, kv=kv,
                      clock=h.clock, events=h.events.append, **kw)


def _seg_cap_harness(n=2, **kw):
    spec = {**{f"segment-{i}": [f"e{i}"] for i in range(n)},
            **{f"caption-{i}": [f"e{i}"] for i in range(n)}}
    kinds = {k: k.split("-")[0] for k in spec}
    deps = {f"caption-{i}": [f"segment-{i}"] for i in range(n)}
    return Harness(spec, pools=SEG_CAP, kinds=kinds, deps=deps, lag=1, **kw)


def _ev(h, name):
    return [e for e in h.events if e["event"] == name]


# ----------------------------------------------------------------------------------------------
# job-end scale-down
# ----------------------------------------------------------------------------------------------


def test_segment_drains_after_its_last_shard_caption_only_at_job_end():
    h = _seg_cap_harness(2)
    auto, kv = FakeAutoscaler(h.clock), KV()
    assert h.run(h.orchestrator(pools=_scaler(h, auto, kv))).result == "done"
    drained = {e["worker_class"]: e["t"] for e in _ev(h, "pool_drained")}
    done = {e["shard_id"]: e["t"] for e in _ev(h, "shard_done")}
    spawned = {e["shard_id"]: e["t"] for e in _ev(h, "spawned")}
    # segment: drained the tick its last receipt is seen, while caption still runs
    assert drained["SegmentWorker"] == max(done["segment-0"], done["segment-1"])
    assert drained["SegmentWorker"] < max(done["caption-0"], done["caption-1"])
    # caption: never drained while segment produces; drained at job end
    assert drained["CaptionWorker"] == max(done.values())
    assert drained["CaptionWorker"] >= max(spawned.values())
    assert auto.window == {"SegmentWorker": DRAINED_SCALEDOWN_S,
                           "CaptionWorker": DRAINED_SCALEDOWN_S}
    assert not [k for k in kv if "/job/" in k]  # the job withdrew its entries at the end
    assert "pool_restored" not in h.names()  # nothing to restore: the window was never lowered


def test_drain_is_safe_with_preemption():
    h = _seg_cap_harness(2)
    for sid in ("segment-0", "segment-1", "caption-0"):
        h.spawner.inject(sid, "preempt", times=3)  # Modal restarts the same call
    auto, kv = FakeAutoscaler(h.clock), KV()
    assert h.run(h.orchestrator(pools=_scaler(h, auto, kv))).result == "done"
    assert h.spawn_counts() == dict.fromkeys(h.spawn_counts(), 1) and len(h.spawn_counts()) == 4
    # a pool with a live (preempted, restarting) call is never drained
    drained = {e["worker_class"]: e["t"] for e in _ev(h, "pool_drained")}
    seg_done = max(e["t"] for e in _ev(h, "shard_done") if e["shard_id"].startswith("segment"))
    assert drained["SegmentWorker"] >= seg_done


def test_respawn_after_a_drain_restores_the_window_before_spawning():
    h = _seg_cap_harness(1)
    auto, kv = FakeAutoscaler(h.clock), KV()
    # a previous job (or an earlier generation) drained both classes
    kv.put("SegmentWorker/state", {"window": DRAINED_SCALEDOWN_S, "by": "old", "t": 0})
    kv.put("CaptionWorker/state", {"window": DRAINED_SCALEDOWN_S, "by": "old", "t": 0})
    order: list[str] = []
    spawn = h.spawner.spawn

    def tapped(kind, job_id, shard_id, cfg):
        order.append(f"spawn:{kind}")
        return spawn(kind, job_id, shard_id, cfg)

    h.spawner.spawn = tapped
    real = auto.set_scaledown

    def tap_scale(cls, s):
        order.append(f"window:{cls}:{s}")
        real(cls, s)

    auto.set_scaledown = tap_scale
    assert h.run(h.orchestrator(pools=_scaler(h, auto, kv))).result == "done"
    assert order.index("window:SegmentWorker:300") < order.index("spawn:segment")
    assert order.index("window:CaptionWorker:300") < order.index("spawn:caption")


def test_two_jobs_on_one_app_only_the_last_one_drains():
    clock, auto, kv = FakeClock(), FakeAutoscaler(), KV()
    a = PoolScaler(job_id="job-a", classes=CLASSES, static_s=300, autoscaler=auto, kv=kv,
                   clock=clock)
    b = PoolScaler(job_id="job-b", classes=CLASSES, static_s=300, autoscaler=auto, kv=kv,
                   clock=clock)
    a.update({"segment": True, "caption": True})
    b.update({"segment": True, "caption": True})
    clock.advance(15)
    a.update({"segment": False, "caption": True})  # A's segment is done, B's is not
    assert auto.window == {}
    a.final()  # A ends while B still runs both pools
    assert auto.window == {}
    clock.advance(15)
    b.update({"segment": False, "caption": True})
    assert auto.window == {"SegmentWorker": DRAINED_SCALEDOWN_S}
    b.final()
    assert auto.window == {"SegmentWorker": 2, "CaptionWorker": 2}
    assert b.drains()["segment"][0][1] is None and not a.drains()  # only B drained


def test_a_drain_racing_another_jobs_restore_is_undone_next_tick():
    clock, auto, kv = FakeClock(), FakeAutoscaler(), KV()
    a = PoolScaler(job_id="job-a", classes=CLASSES, static_s=300, autoscaler=auto, kv=kv,
                   clock=clock)
    b = PoolScaler(job_id="job-b", classes=CLASSES, static_s=300, autoscaler=auto, kv=kv,
                   clock=clock)
    a.final()  # A ended before B wrote its entry (no compare-and-swap): drains
    assert auto.window["SegmentWorker"] == 2
    b.update({"segment": True, "caption": True})  # B's first tick puts it back
    assert auto.window == {"SegmentWorker": 300, "CaptionWorker": 300}


def test_a_stale_entry_of_a_dead_orchestrator_does_not_block_the_drain():
    clock, auto, kv = FakeClock(), FakeAutoscaler(), KV()
    kv.put("SegmentWorker/job/dead", {"busy": True, "t": clock() - 3600})
    kv.put("CaptionWorker/job/alive", {"busy": True, "t": clock() - 10})
    s = PoolScaler(job_id="job-a", classes=CLASSES, static_s=300, autoscaler=auto, kv=kv,
                   clock=clock)
    s.final()
    assert auto.window == {"SegmentWorker": 2}  # caption is still another live job's


def test_autoscaler_errors_never_stop_the_job_and_are_retried():
    h = _seg_cap_harness(1)
    auto, kv = FakeAutoscaler(h.clock, fail=1), KV()
    assert h.run(h.orchestrator(pools=_scaler(h, auto, kv))).result == "done"
    assert _ev(h, "pool_scale_error")
    assert auto.window == {"SegmentWorker": 2, "CaptionWorker": 2}


def test_shared_class_drains_only_when_all_its_kinds_are_done():
    clock, auto, kv = FakeClock(), FakeAutoscaler(), KV()
    s = PoolScaler(job_id=JOB, classes={"reward.terminal": "RewardWorker",
                                        "reward.dense": "RewardWorker"},
                   static_s=300, autoscaler=auto, kv=kv, clock=clock)
    s.update({"reward.terminal": False, "reward.dense": True})
    assert auto.window == {}
    s.update({"reward.terminal": False, "reward.dense": False})
    assert auto.window == {"RewardWorker": 2}
    assert set(s.drains()) == {"reward.terminal", "reward.dense"}


def test_disabled_when_the_deployed_window_is_already_minimal():
    auto = FakeAutoscaler()
    s = PoolScaler(job_id=JOB, classes=CLASSES, static_s=2, autoscaler=auto, kv=KV())
    s.final()
    assert auto.calls == []


def test_kind_demand_counts_waiting_dependents():
    shards = [Shard("segment-0", "segment", ("a",), 60.0),
              Shard("caption-0", "caption", ("a",), 60.0, depends_on=("segment-0",))]
    assert kind_demand(shards, {"segment-0"}, set()) == {"segment": False, "caption": True}
    assert kind_demand(shards, set(), {"segment-0", "caption-0"}) == {"segment": False,
                                                                     "caption": False}


# ----------------------------------------------------------------------------------------------
# budget: idle charged only until the explicit scale-down
# ----------------------------------------------------------------------------------------------


def test_idle_tail_is_cut_by_a_drain():
    assert idle_tail_s(100.0, 300.0) == 300.0
    assert idle_tail_s(100.0, 300.0, [(115.0, None)]) == pytest.approx(17.0)
    assert idle_tail_s(100.0, 300.0, [(50.0, None)]) == 2.0  # already drained
    assert idle_tail_s(100.0, 300.0, [(50.0, 80.0)]) == 300.0  # restored before
    assert idle_tail_s(100.0, 300.0, [(500.0, None)]) == 300.0  # after the window


def test_calls_cost_charges_the_drained_tail_not_the_window():
    settings = make_settings(SEG_CAP)
    rates = pinned_rates()
    calls = [CallRecord("c1", "segment-0", "segment", 0.0, 600.0, "success")]
    full = calls_cost(calls, settings, rates, 2000.0, scaledown_s=300.0)
    cut = calls_cost(calls, settings, rates, 2000.0, scaledown_s=300.0,
                     drains={"segment": ((615.0, None),)})
    per_s = rates.pool_per_second(SEG_CAP["segment"])
    assert full == pytest.approx((600 + 300) * per_s)
    assert cut == pytest.approx((600 + 17) * per_s)


def test_gate_spend_uses_the_orchestrators_drains():
    h = _seg_cap_harness(1)
    auto, kv = FakeAutoscaler(h.clock), KV()
    gate = ProjectionBudgetGate(settings=h.settings, store=h.store,
                                shards=lambda: h.manifests.load().shards,
                                estimator=ManifestEstimator(), scaledown_s=300.0,
                                startup_s={}, clock=h.clock)
    orch = h.orchestrator(pools=_scaler(h, auto, kv), budget=gate)
    assert h.run(orch).result == "done"
    state = orch.budget_state()
    assert set(state.drains) == {"segment", "caption"}
    with_drains = gate.spend_report(h.clock()).calls_usd
    gate.observe(BudgetState(state.now, state.calls))  # the same calls, no drains
    assert gate.spend_report(h.clock()).calls_usd > with_drains


def test_drains_persist_for_a_successor_generation():
    h = _seg_cap_harness(1)
    path = drains_path(JobPaths(JOB).root, CFG)
    s = _scaler(h, FakeAutoscaler(), KV(), store=h.store, path=path)
    s.update({"segment": False, "caption": True})
    again = _scaler(h, FakeAutoscaler(), KV(), store=h.store, path=path)
    assert again.drains() == s.drains() and "segment" in again.drains()


# ----------------------------------------------------------------------------------------------
# pool start timing: caption is spawned lazily; pre-warm timed from the rate table
# ----------------------------------------------------------------------------------------------


def test_caption_is_spawned_only_once_its_segment_dependency_is_met():
    from tests.fakes.jobs import WorkerBehavior

    h = _seg_cap_harness(2, behavior=WorkerBehavior())
    seen: list[tuple[float, dict]] = []
    spawn = h.spawner.spawn

    def tapped(kind, job_id, shard_id, cfg):
        seen.append((h.clock(), dict(h.spawner.in_flight())))
        if kind == "caption":
            n = shard_id.split("-")[1]
            assert h.receipt(f"segment-{n}", "segment"), "caption spawned before its segment"
        return spawn(kind, job_id, shard_id, cfg)

    h.spawner.spawn = tapped
    assert h.run().result == "done"
    first_caption = min(e["t"] for e in _ev(h, "spawned") if e["kind"] == "caption")
    first_seg_done = min(e["t"] for e in _ev(h, "shard_done")
                         if e["shard_id"].startswith("segment"))
    assert first_caption >= first_seg_done  # no caption container waits on segment output


def _estimates(seg_work=600.0, startup=None):
    startup = startup or {"segment": 300.0, "caption": 325.0}
    return {"startup_s": lambda k: startup[k], "work_s": lambda s: seg_work}


def test_first_ready_at_from_live_dependency_calls():
    shards = [Shard("segment-0", "segment", ("a",), 60.0),
              Shard("caption-0", "caption", ("a",), 60.0, depends_on=("segment-0",))]
    est = _estimates()
    assert first_ready_at("caption", shards, satisfied=set(), live_spawned={}, now=0.0,
                          **est) is None
    assert first_ready_at("caption", shards, satisfied=set(),
                          live_spawned={"segment-0": 100.0}, now=150.0, **est) == 1000.0
    assert first_ready_at("caption", shards, satisfied={"segment-0"}, live_spawned={},
                          now=150.0, **est) == 150.0


def test_caption_prewarm_starts_at_ready_minus_startup_and_is_released():
    h = _seg_cap_harness(1)
    auto, kv = FakeAutoscaler(h.clock), KV()
    # the fake spawner finishes a call in one 10 s tick, so make the estimates fit that clock:
    # segment ready = spawn + 5 + 25; caption startup 20 -> pre-warm 10 s after the spawn
    pools = _scaler(h, auto, kv, startup_s={"segment": 5.0, "caption": 20.0}.__getitem__,
                    work_s=lambda s: 25.0)
    orch = h.orchestrator(pools=pools)
    assert orch.acquire()
    assert orch.tick() == "running"  # spawns segment-0: nothing to time caption from yet
    assert not _ev(h, "pool_prewarm_plan")
    assert orch.tick() == "running"  # segment-0 is live: plan caption's pre-warm
    plan = _ev(h, "pool_prewarm_plan")
    assert [p["decision"] for p in plan] == ["prewarm"] and plan[0]["kind"] == "caption"
    assert plan[0]["saves_s"] == 20.0
    t_seg = _ev(h, "spawned")[0]["t"]
    assert plan[0]["start_at"] == t_seg + 10.0 and plan[0]["ready_at"] == t_seg + 30.0
    assert auto.min == {}  # not yet
    h.clock.advance(10)  # the segment call is still live (no spawner tick)
    assert orch.tick() == "running"
    assert auto.min == {"CaptionWorker": 1}
    assert _ev(h, "pool_prewarm_started")
    while (r := orch.tick()) == "running":
        h.sleep(10)
    assert r == "done"
    assert auto.min == {"CaptionWorker": 0}
    rel = _ev(h, "pool_prewarm_released")
    assert rel and rel[0]["t"] >= min(e["t"] for e in _ev(h, "spawned")
                                      if e["kind"] == "caption")
    # the pre-warm container is charged from its start until the first caption call
    state = orch.budget_state()
    t0, t1 = state.prewarms["caption"]
    assert t1 is not None and t1 >= t0
    with_pw = calls_cost(state.calls, h.settings, pinned_rates(), state.now,
                         prewarms=state.prewarms)
    without = calls_cost(state.calls, h.settings, pinned_rates(), state.now)
    assert with_pw - without == pytest.approx(
        (t1 - t0) * pinned_rates().pool_per_second(SEG_CAP["caption"]))


def test_prewarm_is_skipped_with_a_warm_pool_and_recorded():
    h = _seg_cap_harness(1, min_containers=1)
    auto, kv = FakeAutoscaler(h.clock), KV()
    pools = _scaler(h, auto, kv, **_estimates())
    assert h.run(h.orchestrator(pools=pools)).result == "done"
    plan = _ev(h, "pool_prewarm_plan")
    assert [p["decision"] for p in plan] == ["skip"]
    assert "min_containers" in plan[0]["reason"]
    assert all(n == 1 for k, c, n, _ in auto.calls if k == "min")  # back to the deployed 1


def test_job_end_releases_a_prewarm_even_without_a_caption_call():
    clock, auto = FakeClock(), FakeAutoscaler()
    s = PoolScaler(job_id=JOB, classes=CLASSES, static_s=300, autoscaler=auto, kv=KV(),
                   clock=clock, **_estimates(seg_work=10.0))
    shards = [Shard("segment-0", "segment", ("a",), 60.0),
              Shard("caption-0", "caption", ("a",), 60.0, depends_on=("segment-0",))]
    s.tick_prewarm(shards, satisfied=set(), live_spawned={"segment-0": clock()},
                   called_kinds={"segment"})
    assert auto.min == {"CaptionWorker": 1}  # ready in 310 s, startup 325 s: start now
    s.final()  # e.g. the job was cancelled before caption ran
    assert auto.min == {"CaptionWorker": 0}


# ----------------------------------------------------------------------------------------------
# Modal adapter (offline)
# ----------------------------------------------------------------------------------------------


def test_modal_autoscaler_updates_the_class_instance():
    from flora_deploy.modal_app.orchestrator_app import ModalAutoscaler

    got = []

    class Obj:
        def update_autoscaler(self, **kw):
            got.append(kw)

    class Cls:
        @staticmethod
        def from_name(app, name):
            got.append((app, name))
            return Obj

    a = ModalAutoscaler("flora-app", cls_api=Cls)
    a.set_scaledown("CaptionWorker", 2)
    a.set_min_containers("CaptionWorker", 1)
    assert got == [("flora-app", "CaptionWorker"), {"scaledown_window": 2},
                   ("flora-app", "CaptionWorker"), {"min_containers": 1}]


def test_installed_modal_supports_the_per_class_autoscaler_override():
    import inspect

    import modal
    from modal.cls import _Obj

    sig = inspect.signature(_Obj.update_autoscaler).parameters
    assert {"scaledown_window", "min_containers", "max_containers"} <= set(sig)
    src = inspect.getsource(modal._functions._FunctionAutoscalerManager.update)
    assert "2 <= scaledown_window <= 3600" in src
