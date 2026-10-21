"""H4: the budget gate's in-flight accounting, hard cap (``overspend``), measured rates, spend
sources that do not depend on container receipts, and the review's overspend scenarios replayed
on a time-based simulation (the real Orchestrator, driven by ``orchestrator_app.TickDriver``)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime

import pytest

from flora_deploy.jobs.budget import (
    BudgetState,
    CallRecord,
    ManifestEstimator,
    ProjectionBudgetGate,
    Rates,
    RateTableEstimator,
    apply_environment_budget,
    budget_summary,
    calls_cost,
    modal_billing_provider,
    pinned_rates,
)
from flora_deploy.jobs.orchestrator import DictLease, Orchestrator, utc
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.spawner import CallStatus
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import ContainerReceipt, Shard, ShardReceipt
from flora_deploy.modal_app.orchestrator_app import TickDriver
from flora_deploy.planner.rates import RateTable
from tests.fakes.jobs import (
    CFG,
    JOB,
    FakeAttempts,
    FakeClock,
    FakeDict,
    FakeManifests,
    FakeProgress,
    make_manifest,
)
from tests.fakes.modal import FakeStore

T0 = 1_800_000_000.0
UNIT = Rates(gpu_per_s={"H200": 1.0}, core_per_s=0.0, gib_per_s=0.0, source="unit")


def settings(budget=10.0, max_containers=2, max_inputs=1, cpu=8.0, mem=96.0):
    pool = PoolSettings(gpu="H200", max_containers=max_containers, max_inputs=max_inputs,
                        cpu_cores=cpu, memory_gib=mem)
    return RunSettings(job_id=JOB, preset="exact", config_hash=CFG, budget_usd=budget,
                       max_gpus=8, pools={"segcap": pool})


def shards(n=3, est=30.0):  # est_gpu_seconds 30 at the manifest's 3.0 placeholder = 10 video-s
    return [Shard(f"segcap-{i:05d}", "segcap", (f"u{i}",), est) for i in range(n)]


def gate(budget=10.0, *, store=None, n=3, clock=lambda: T0, **kw):
    kw.setdefault("rates", UNIT)
    return ProjectionBudgetGate(settings=settings(budget, **kw.pop("pool", {})),
                                store=store or FakeStore(), shards=lambda: shards(n),
                                clock=clock, **kw)


# ----------------------------------------------------------------------------------------------
# (b) measured rates
# ----------------------------------------------------------------------------------------------


def test_default_estimator_is_rates_json_not_placeholders():
    g = gate()
    assert isinstance(g.estimator, RateTableEstimator)
    pool1 = settings().pools["segcap"]
    pool4 = settings(max_inputs=4).pools["segcap"]
    s = Shard("segcap-00000", "segcap", ("u",), 0.0)
    est = RateTableEstimator(RateTable.load(), "exact", durations={"u": 100.0})
    # rates.json (review 5 M-C): the deployed configuration (gpu_thread on) is P1a-5s's measured
    # cu128 c=4 rate, 6.4, the only usable exact segcap row (cu130 rows and the superseded
    # projection are stale; S1's 7.0 excludes gpu_thread), so c=1 uses it too
    assert est.container_seconds(s, pool1) == pytest.approx(640.0)  # 6.4 x 100 video-s
    assert est.container_seconds(s, pool4) == pytest.approx(640.0)  # 6.4 at c=4
    assert est.startup_s("segcap") == 103
    upper = RateTableEstimator(RateTable.load(), "exact", durations={"u": 100.0},
                               quantile="upper")
    assert upper.container_seconds(s, pool1) == pytest.approx(640.0 * 1.35)
    # without durations, video seconds come back out of the manifest's placeholder estimate
    assert g.estimator.container_seconds(shards(1)[0], pool1) == pytest.approx(64.0)
    # the gate and `flora plan` agree: 2.1x the old placeholder (3.0) on exact segcap
    assert g.estimator.container_seconds(shards(1)[0], pool1) / ManifestEstimator(
    ).container_seconds(shards(1)[0], pool1) == pytest.approx(6.4 / 3.0)
    d = g.admit(shards(1)[0], {})
    assert "rates.json" in d.detail["estimate_source"]


def test_projection_charges_rates_json_work_and_startup():
    p = gate(1e6).projection({})
    assert p.queued_usd == pytest.approx(3 * 64.0)
    assert p.startup_usd == pytest.approx(2 * 103)  # max_containers 2 containers to start


# ----------------------------------------------------------------------------------------------
# (c) spend sources
# ----------------------------------------------------------------------------------------------


def test_calls_cost_sweep_containers_and_scaledown():
    s = settings(max_inputs=1)
    calls = [CallRecord("a", "x", "segcap", T0, T0 + 100), CallRecord("b", "y", "segcap", T0 + 50)]
    # a: 100 s; b: live 50 -> 200 = 150 s; one scaledown after a ends (b still live)
    assert calls_cost(calls, s, UNIT, T0 + 200, scaledown_s=10) == pytest.approx(260.0)
    s2 = settings(max_inputs=2)
    # both share one container from 50 to 100
    assert calls_cost(calls, s2, UNIT, T0 + 200, scaledown_s=0) == pytest.approx(200.0)
    assert calls_cost([], s, UNIT, T0) == 0.0


class HidingStore(FakeStore):
    """A listing that forgets files (an eventually consistent list can do that between
    replicas); get/exists by path still work."""

    def __init__(self):
        super().__init__()
        self.hide = False

    def list(self, prefix):
        return iter(()) if self.hide else super().list(prefix)


def _receipt(cid, start, hb, stopped=None):
    r = ContainerReceipt(cid, "w", "H200", 1, 0, None, 0, None, None, True, utc(start), utc(hb),
                         None if stopped is None else utc(stopped))
    return json.dumps(asdict(r)).encode()


def test_known_receipts_are_reread_by_path_and_spend_never_drops():
    st = HidingStore()
    st.put_bytes(JobPaths(JOB).container("c1"), _receipt("c1", T0 - 100, T0))
    now = [T0]
    g = gate(store=st, clock=lambda: now[0], refresh_s=0)
    assert g.spend_report().receipts_usd == pytest.approx(100.0)
    st.hide = True  # the listing lost it
    now[0] += 50
    st.put_bytes(JobPaths(JOB).container("c1"), _receipt("c1", T0 - 100, T0 + 50))
    g.refresh()
    assert g.spend_report().receipts_usd == pytest.approx(150.0)  # re-read by path
    now[0] += 1000  # heartbeat goes stale: capped at hb + stale_after_s, never below before
    assert g.spend_report().receipts_usd == pytest.approx(450.0)


def test_spend_without_any_container_receipts_uses_call_history():
    g = gate(scaledown_s=0)
    g.observe(BudgetState(T0, (CallRecord("fc-1", "segcap-00000", "segcap", T0 - 300),)))
    rep = g.spend_report()
    assert rep.receipts_usd == 0 and rep.calls_usd == pytest.approx(300.0)
    assert rep.total_usd == pytest.approx(300.0) and rep.source == "calls"


def test_billing_is_reconciled_as_a_floor_and_errors_are_tolerated():
    calls = {"n": 0}

    def billing():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("billing API down")
        return 500.0

    now = [T0]
    g = gate(billing=billing, billing_refresh_s=600, clock=lambda: now[0])
    rep = g.spend_report()
    assert rep.billing_usd == 500.0 and rep.total_usd == 500.0 and rep.source == "billing"
    now[0] += 601
    assert g.spend_report().billing_usd == 500.0 and "down" in g.billing_error
    now[0] += 100
    g.spend_report()
    assert calls["n"] == 2  # refreshed at most every billing_refresh_s


def test_modal_billing_provider_filters_job_rows():
    seen = {}

    def report(**kw):
        seen.update(kw)
        return [{"description": "flora-workers", "tags": {"flora_job": JOB}, "cost": "1.25"},
                {"description": "flora-workers", "tags": {"flora_job": "other"}, "cost": "9"},
                {"description": "flora-orchestrator", "tags": {}, "cost": "0.5"}]

    p = modal_billing_provider(start=datetime(2026, 10, 21, tzinfo=UTC),
                               tags={"flora_job": JOB}, descriptions=("flora-orchestrator",),
                               report=report)
    assert p() == pytest.approx(1.75)
    assert seen["resolution"] == "h" and seen["tag_names"] == ["*"]


# ----------------------------------------------------------------------------------------------
# (a) in-flight work and the hard cap
# ----------------------------------------------------------------------------------------------


def test_admit_charges_live_calls_their_remaining_estimate():
    g = gate(1e6, estimator=ManifestEstimator(), startup_s={}, scaledown_s=0)
    # 30 s estimate per shard; one live call 10 s in: 10 spent (calls), 20 remaining
    g.observe(BudgetState(T0, (CallRecord("fc-1", "segcap-00000", "segcap", T0 - 10),)))
    p = g.projection({"segcap": 1})
    assert p.spend_usd == pytest.approx(10.0)
    assert p.in_flight_usd == pytest.approx(20.0)
    assert p.queued_usd == pytest.approx(60.0)


def test_calibration_learned_from_finished_calls():
    g = gate(1e6, estimator=ManifestEstimator(), startup_s={"segcap": 0.0})
    g.observe(BudgetState(T0, (
        CallRecord("fc-1", "segcap-00000", "segcap", T0 - 400, T0 - 100, "success"),)))
    assert g.effective_calibration()["segcap"] == pytest.approx(10.0)  # 300 s vs 30 s estimate
    g2 = gate(1e6, estimator=ManifestEstimator(), learn_calibration=False)
    g2.observe(g._state)  # type: ignore[arg-type]
    assert g2.effective_calibration() == {}


def test_calibration_subtracts_startup_only_for_cold_calls():
    """Review 5 M-C: startup is paid once per container. Three back-to-back calls on one warm
    container, each 130 s for a 30 s estimate, with a 100 s startup: the first call is cold
    (30 s of work), the next two are warm (130 s each). Subtracting startup from every call
    would learn 1.0 and under-charge the rest of the job."""
    g = gate(1e6, estimator=ManifestEstimator(), startup_s={"segcap": 100.0}, scaledown_s=60)
    g.observe(BudgetState(T0, (
        CallRecord("fc-1", "segcap-00000", "segcap", T0 - 400, T0 - 270, "success"),
        CallRecord("fc-2", "segcap-00001", "segcap", T0 - 265, T0 - 135, "success"),
        CallRecord("fc-3", "segcap-00002", "segcap", T0 - 130, T0, "success"),
    )))
    assert g.effective_calibration()["segcap"] == pytest.approx((1.0 + 130 / 30 * 2) / 3)
    cold = gate(1e6, estimator=ManifestEstimator(), startup_s={"segcap": 100.0}, scaledown_s=0)
    cold.observe(BudgetState(T0, (  # a gap longer than scaledown: a new container each time
        CallRecord("fc-1", "segcap-00000", "segcap", T0 - 400, T0 - 270, "success"),
        CallRecord("fc-2", "segcap-00001", "segcap", T0 - 200, T0 - 70, "success"),
    )))
    assert cold.effective_calibration()["segcap"] == pytest.approx(1.0)


def test_calibration_counts_calls_landing_on_a_running_container_as_warm():
    """Review 6 B3: with ``max_inputs>1`` a call spawned while a sibling is still running
    lands on that container's free slot: no startup. Treating it as cold subtracts startup and
    learns an optimistic rate."""
    g = gate(1e6, estimator=ManifestEstimator(), startup_s={"segcap": 100.0}, scaledown_s=0,
             pool={"max_inputs": 2, "max_containers": 1})
    g.observe(BudgetState(T0, (
        CallRecord("fc-1", "segcap-00000", "segcap", T0 - 400, T0 - 100, "success"),
        CallRecord("fc-2", "segcap-00001", "segcap", T0 - 300, T0 - 100, "success"),
    )))
    est = 30.0 * 2
    assert g.effective_calibration()["segcap"] == pytest.approx(((300 - 100) / est
                                                                  + 200 / est) / 2)
    one = gate(1e6, estimator=ManifestEstimator(), startup_s={"segcap": 100.0}, scaledown_s=0)
    one.observe(g._state)  # type: ignore[arg-type]
    # max_inputs=1: a sibling's container has no free slot, so it stays cold
    assert one.effective_calibration()["segcap"] == pytest.approx(
        ((300 - 100) / 30 + (200 - 100) / 30) / 2)


def test_overspend_cancels_largest_remaining_first_then_everything():
    g = gate(100.0, estimator=ManifestEstimator(), startup_s={}, scaledown_s=0,
             tick_reserve_s=0, learn_calibration=False, n=3)
    live = (CallRecord("fc-a", "segcap-00000", "segcap", T0 - 25),  # 5 left
            CallRecord("fc-b", "segcap-00001", "segcap", T0 - 5),  # 25 left
            CallRecord("fc-c", "segcap-00002", "segcap", T0 - 15))  # 15 left
    assert g.overspend(BudgetState(T0, live)) == []  # spent 45 + 45 remaining <= 100
    g.settings = settings(70.0)
    assert g.overspend(BudgetState(T0, live)) == ["fc-b"]  # 45 + 20 <= 70 after fc-b
    g.settings = settings(56.0)
    assert g.overspend(BudgetState(T0, live)) == ["fc-b", "fc-c"]
    g.settings = settings(45.0)
    assert g.overspend(BudgetState(T0, live)) == ["fc-a", "fc-b", "fc-c"]  # spend alone hits it
    assert g.last_overspend["cap_usd"] == 45.0
    assert g.overspend(BudgetState(T0, ())) == []


def test_overspend_reserves_one_tick_of_burn():
    g = gate(100.0, estimator=ManifestEstimator(), startup_s={}, scaledown_s=0,
             tick_reserve_s=30.0, learn_calibration=False)
    live = (CallRecord("fc-a", "segcap-00000", "segcap", T0 - 69),)  # 69 spent, 0 left
    assert g.overspend(BudgetState(T0, live)) == []  # 69 + 30 <= 100
    later = (CallRecord("fc-a", "segcap-00000", "segcap", T0 - 71),)
    assert g.overspend(BudgetState(T0, later)) == ["fc-a"]  # 71 + 30 > 100


def test_budget_summary_carries_the_environment_instruction():
    s = settings(10.0)
    backstop = apply_environment_budget("flora-env", s, discover=False)
    out = budget_summary("flora-env", s, backstop, gate=gate(10.0))
    assert out["environment_budget_applied"] is False
    assert "$12.00" in out["environment_budget_instruction"]
    assert "flora-env" in out["environment_budget_instruction"]
    assert "spend_usd" in out and "gate_limit_usd" in out


# ----------------------------------------------------------------------------------------------
# The review's overspend scenarios, on a time-based simulation
# ----------------------------------------------------------------------------------------------


class TimedSpawner:
    """Calls take ``run_s`` of (simulated) wall time in one container each. Workers write a
    ContainerReceipt at start, refresh its heartbeat every step and stop it at the end (or on
    cancel), then the shard receipt. ``receipts=False``: no ContainerReceipts at all."""

    def __init__(self, store, clock, s: RunSettings, rates: Rates, run_s: float, *,
                 receipts: bool = True):
        self.store, self.clock, self.settings, self.rates = store, clock, s, rates
        self.run_s, self.receipts = run_s, receipts
        self.calls: dict[str, dict] = {}

    def _container(self, c):
        if not self.receipts:
            return
        pool = self.settings.pools["segcap"]
        hb = c["stopped"] if c["stopped"] is not None else self.clock()
        r = ContainerReceipt(c["cont"], "w", pool.gpu, pool.gpu_count, pool.cpu_cores, None,
                             pool.memory_gib, None, None, True, utc(c["start"]), utc(hb),
                             None if c["stopped"] is None else utc(c["stopped"]))
        self.store.put_bytes(JobPaths(JOB).container(c["cont"]), json.dumps(asdict(r)).encode())

    def spawn(self, kind, job_id, shard_id, config_hash):
        n = len(self.calls) + 1
        c = {"sid": shard_id, "start": self.clock(), "end": self.clock() + self.run_s,
             "state": "pending", "stopped": None, "cont": f"ta-{n:05d}"}
        self.calls[f"fc-{n}"] = c
        self._container(c)
        return f"fc-{n}"

    def advance(self):
        now = self.clock()
        for c in self.calls.values():
            if c["state"] != "pending":
                continue
            if now >= c["end"]:
                c["state"], c["stopped"] = "success", c["end"]
                rec = ShardReceipt(c["sid"], "segcap", CFG, c["cont"], ("u",), (), (), "H200",
                                   self.run_s, utc(c["end"]))
                self.store.put_bytes(JobPaths(JOB).receipt(CFG, "segcap", c["sid"]),
                                     json.dumps(asdict(rec)).encode())
            self._container(c)

    def poll(self, call_id):
        return CallStatus(call_id, self.calls[call_id]["state"])

    def cancel(self, call_id):
        c = self.calls[call_id]
        if c["state"] == "pending":
            c["state"], c["stopped"] = "cancelled", self.clock()
            self._container(c)

    terminate = cancel

    def true_spend(self):
        per_s = self.rates.pool_per_second(self.settings.pools["segcap"])
        return sum(((c["stopped"] if c["stopped"] is not None else self.clock()) - c["start"])
                   * per_s for c in self.calls.values())


def simulate(*, lag, max_containers, run_s, estimator=None, policy="projection",
             receipts=True, budget=5.0):
    s = settings(budget, max_containers=max_containers)
    store, clock = FakeStore(lag=lag), FakeClock(T0)
    rates = pinned_rates()
    manifests = FakeManifests(make_manifest({f"segcap-{i:05d}": [f"d--b{i}"] for i in range(12)}))
    spawner = TimedSpawner(store, clock, s, rates, run_s, receipts=receipts)
    g = ProjectionBudgetGate(settings=s, store=store, shards=lambda: manifests.load().shards,
                             rates=rates, clock=clock, policy=policy,
                             estimator=estimator, refresh_s=60.0)
    orch = Orchestrator(settings=s, store=store, spawner=spawner,  # type: ignore[arg-type]
                        lease=DictLease(FakeDict(), f"orchestrator/{JOB}", clock=clock),
                        manifests=manifests, attempts=FakeAttempts(store),
                        progress=FakeProgress(store), budget=g, clock=clock)

    def sleep(dt):
        clock.advance(dt)
        spawner.advance()
        store.tick()

    out = TickDriver(orch).run_until_idle(sleep=sleep, poll_interval_s=15.0, max_ticks=5000)
    one_shard = run_s * rates.pool_per_second(s.pools["segcap"])
    return out, spawner.true_spend(), one_shard


REVIEW_CASES = [(0, 1, 600), (0, 4, 600), (0, 8, 600), (3, 1, 600), (3, 4, 600), (0, 8, 1800)]


@pytest.mark.parametrize("lag,mc,run_s", REVIEW_CASES)
def test_review_scenarios_with_placeholder_estimates_stay_within_cap(lag, mc, run_s):
    """The review's table (+52% .. +355% before): the manifest's 60 GPU-s estimates are 10-30x
    low, every shard is admitted, and only the hard cap stops the spend."""
    out, spent, one = simulate(lag=lag, max_containers=mc, run_s=run_s,
                               estimator=ManifestEstimator())
    assert out.result in ("budget_stopped", "done")
    assert spent <= 5.0 + one, (spent, one)


@pytest.mark.parametrize("lag,mc,run_s", REVIEW_CASES)
def test_review_scenarios_with_rates_json_partial_policy(lag, mc, run_s):
    out, spent, one = simulate(lag=lag, max_containers=mc, run_s=run_s, policy="next_shard")
    assert out.result in ("budget_stopped", "done")
    assert spent <= 5.0 + one, (spent, one)


def test_hard_cap_holds_without_any_container_receipts():
    out, spent, one = simulate(lag=3, max_containers=8, run_s=1800,
                               estimator=ManifestEstimator(), receipts=False)
    assert out.result == "budget_stopped"
    assert spent <= 5.0 + one


def test_projection_policy_with_rates_json_refuses_a_job_it_cannot_afford():
    # $2 < the projection (12 shards x 10 video-s x 6.4 + 8 x 103 s startup on H200, ~$2.5)
    out, spent, _ = simulate(lag=0, max_containers=8, run_s=600, budget=2.0)
    assert out.result == "budget_stopped" and spent == 0.0


def test_affordable_job_still_completes():
    out, spent, _ = simulate(lag=3, max_containers=4, run_s=300, budget=50.0)
    assert out.result == "done" and spent < 50.0
