"""Projection, the orchestrator BudgetGate, and the Environment-budget backstop."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from flora_deploy.jobs.budget import (
    ProjectionBudgetGate,
    Rates,
    apply_environment_budget,
    environment_budget_usd,
    project,
)
from flora_deploy.jobs.orchestrator import utc
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import ContainerReceipt, Shard
from tests.fakes.jobs import CFG, JOB, Harness

T0 = 1_800_000_000.0
# $1 per GPU-second, nothing else: costs are easy to do by hand.
UNIT = Rates(gpu_per_s={"H200": 1.0}, core_per_s=0.0, gib_per_s=0.0, source="unit")


def settings(budget=30.0, max_containers=2, max_inputs=1, gpu_count=1, factor=1.2):
    pool = PoolSettings(gpu="H200", gpu_count=gpu_count, max_containers=max_containers,
                        max_inputs=max_inputs)
    return RunSettings(job_id=JOB, preset="exact", config_hash=CFG, budget_usd=budget,
                       max_gpus=8, pools={"segcap": pool}, environment_budget_factor=factor)


def shards(n=3, est=10.0):
    return [Shard(f"segcap-{i:05d}", "segcap", (f"u{i}",), est) for i in range(n)]


def test_projection_hand_calc():
    s = settings(max_containers=2)
    p = project(spent=5.0, pending=shards(3), in_flight={"segcap": 1}, settings=s, rates=UNIT,
                startup_s={"segcap": 4.0})
    assert p.spend_usd == 5.0
    assert p.in_flight_usd == 10.0  # one in-flight shard, full estimate
    assert p.queued_usd == 20.0
    # 2 queued shards, 1 container running, max 2 -> one more container start of 4 s
    assert p.startup_usd == 4.0
    assert p.total_usd == 39.0
    half = project(spent=5.0, pending=shards(3), in_flight={"segcap": 1}, settings=s,
                   rates=UNIT, in_flight_fraction=0.5)
    assert half.in_flight_usd == 5.0 and half.startup_usd == 0.0


def test_projection_gpu_count_and_calibration():
    s = settings(gpu_count=2, max_containers=1)
    # est_gpu_seconds 10 on a 2-GPU container = 5 container-s at $2/s = $10
    p = project(spent=0, pending=shards(1), in_flight={}, settings=s, rates=UNIT,
                calibration={"segcap": 1.5})
    assert p.queued_usd == pytest.approx(15.0)


def gate(budget, *, margin=0.0, store=None, n=3, policy="projection", **kw):
    return ProjectionBudgetGate(settings=settings(budget), store=store,
                                shards=lambda: shards(n), rates=UNIT, safety_margin=margin,
                                policy=policy, clock=lambda: T0, **kw)


def test_gate_boundary(tmp_path):
    st = LocalStore(tmp_path)
    s0 = shards(1)[0]
    assert gate(30.0, store=st).admit(s0, {}).allow  # projection == limit -> allow
    d = gate(29.99, store=st).admit(s0, {})
    assert not d.allow and "exceeds" in d.reason
    assert d.detail["projected_usd"] == 30.0 and d.detail["limit_usd"] == 29.99
    # safety margin: limit = budget x (1 - margin)
    assert gate(60.0, margin=0.5, store=st).admit(s0, {}).allow
    assert not gate(59.9, margin=0.5, store=st).admit(s0, {}).allow


def test_gate_counts_done_shards_and_spend(tmp_path):
    st = LocalStore(tmp_path)
    paths = JobPaths(JOB)
    st.put_bytes(paths.receipt(CFG, "segcap", "segcap-00000"), b"{}")
    r = ContainerReceipt("c1", "app", "H200", 1, 0, None, 0, None, None, True, utc(T0 - 7),
                         utc(T0 - 7), utc(T0))
    st.put_bytes(paths.container("c1"), json.dumps(asdict(r)).encode())
    g = gate(27.0, store=st)
    p = g.projection({})
    assert p.spend_usd == pytest.approx(7.0) and p.queued_usd == 20.0
    assert g.admit(shards(3)[1], {}).allow
    assert not gate(26.9, store=st).admit(shards(3)[1], {}).allow


def test_gate_quarantined_parents_not_counted(tmp_path):
    g = gate(20.0, store=LocalStore(tmp_path), quarantined=lambda: {"segcap-00002"})
    assert g.projection({}).queued_usd == 20.0


def test_next_shard_policy(tmp_path):
    st = LocalStore(tmp_path)
    s0 = shards(1)[0]
    # whole job ($30) does not fit, but this shard ($10) + in-flight ($10) does
    assert not gate(25.0, store=st).admit(s0, {"segcap": 1}).allow
    assert gate(25.0, store=st, policy="next_shard").admit(s0, {"segcap": 1}).allow
    assert not gate(19.0, store=st, policy="next_shard").admit(s0, {"segcap": 1}).allow


def test_gate_refresh_is_cached(tmp_path):
    st = LocalStore(tmp_path)
    now = [T0]
    g = ProjectionBudgetGate(settings=settings(100), store=st, shards=lambda: shards(3),
                             rates=UNIT, clock=lambda: now[0], refresh_s=60)
    assert g.projection({}).queued_usd == 30.0
    st.put_bytes(JobPaths(JOB).receipt(CFG, "segcap", "segcap-00000"), b"{}")
    assert g.projection({}).queued_usd == 30.0  # cached
    now[0] += 61
    assert g.projection({}).queued_usd == 20.0


def test_gate_validates_args(tmp_path):
    with pytest.raises(ValueError):
        gate(10, margin=1.0, store=LocalStore(tmp_path))
    with pytest.raises(ValueError):
        gate(10, policy="yolo", store=LocalStore(tmp_path))


def test_orchestrator_budget_stops_with_real_gate():
    h = Harness({"s0": ["a"], "s1": ["b"], "s2": ["c"]})
    # each shard is 60 GPU-s = $60 at $1/s; budget is $10 -> first admit denies
    h.budget = ProjectionBudgetGate(settings=h.settings, store=h.store,
                                    shards=lambda: h.manifests.load().shards, rates=UNIT,
                                    clock=h.clock)
    out = h.run()
    assert out.result == "budget_stopped"
    assert "budget_stop" in h.names()


def test_orchestrator_completes_under_budget():
    h = Harness({"s0": ["a"], "s1": ["b"]})
    cheap = Rates(gpu_per_s={"H200": 0.01}, core_per_s=0.0, gib_per_s=0.0)
    h.budget = ProjectionBudgetGate(settings=h.settings, store=h.store,
                                    shards=lambda: h.manifests.load().shards, rates=cheap,
                                    clock=h.clock)
    assert h.run().result == "done"


def test_environment_budget_value_and_apply():
    assert environment_budget_usd(settings(budget=10.0)) == 12.0
    assert environment_budget_usd(settings(budget=33.33)) == 40.0  # 39.996 rounded up
    assert environment_budget_usd(settings(budget=10.0, factor=1.0)) == 10.0
    res = apply_environment_budget("flora-job", settings(budget=10.0), discover=False)
    assert not res.applied and res.amount_usd == 12.0
    assert "$12.00" in res.message and "flora-job" in res.message
    calls = []
    ok = apply_environment_budget("e", settings(budget=10.0),
                                  setter=lambda env, amt: calls.append((env, amt)))
    assert ok.applied and calls == [("e", 12.0)]

    def boom(env, amt):
        raise RuntimeError("nope")

    bad = apply_environment_budget("e", settings(budget=10.0), setter=boom)
    assert not bad.applied and "failed" in bad.message and "$12.00" in bad.message
