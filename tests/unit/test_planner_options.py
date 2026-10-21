"""`flora plan` options: hand-checked cost/wall, determinism, infeasibility, write-back."""

from __future__ import annotations

import math

import pytest
import yaml

from flora_deploy.cli.init import template_text
from flora_deploy.config.schema import parse_config
from flora_deploy.jobs.budget import pinned_rates
from flora_deploy.planner.options import (
    ORCHESTRATOR_CORES,
    ORCHESTRATOR_GIB,
    PlanRequest,
    WorkloadGroup,
    WorkloadSummary,
    apply_option,
    build_option,
    plan_options,
    recommended,
)
from flora_deploy.planner.rates import RateTable

H200_PER_S = (4.54 + 8 * 0.0472 + 96 * 0.008) / 3600
ORCH_PER_S = 3 * (ORCHESTRATOR_CORES * 0.0472 + ORCHESTRATOR_GIB * 0.008) / 3600


def one_rate_table(rate=4.0, startup=100.0, u=0.0) -> RateTable:
    return RateTable.from_dict({
        "schema": 1,
        "rates": [{"preset": "exact", "kind": "segcap", "cameras": 1, "concurrency": 4,
                   "container_s_per_video_s": rate, "uncertainty": u, "status": "measured",
                   "provenance": "test", "date": "2026-10-21"}],
        "startup": [{"preset": "exact", "kind": "segcap", "startup_s": startup,
                     "uncertainty": 0.0, "status": "measured", "provenance": "test",
                     "date": "2026-10-21"}]}, source="test")


def test_hand_calc_single_kind():
    table = one_rate_table()
    wl = WorkloadSummary.from_hours(10, episodes=1000)  # 36,000 video-s -> 144,000 container-s
    req = PlanRequest(wl, presets=("exact",), max_gpus=4, deadline_s=None)
    o = build_option(req, table, pinned_rates(), "exact", 4)
    k = o.kinds[0]
    assert (k.containers, k.max_inputs, k.gpu) == (4, 4, "H200")
    assert o.wall_s == pytest.approx(100 + 144_000 / 4)
    expect = H200_PER_S * (144_000 + 4 * 100) + ORCH_PER_S * (100 + 36_000)
    assert o.cost_usd == pytest.approx(expect, abs=0.01)
    assert o.peak_gpus == 4 and o.certification == "reference" and o.status == "ok"


def test_deadline_sizing_uses_fewest_gpus():
    table = one_rate_table()
    wl = WorkloadSummary.from_hours(10, episodes=1000)
    # 144,000 container-s; 20 h deadline (72,000 s) minus 100 s startup -> 3 containers
    req = PlanRequest(wl, presets=("exact",), max_gpus=10, deadline_s=20 * 3600)
    o = build_option(req, table, pinned_rates(), "exact", 4)
    assert o.kinds[0].containers == math.ceil(144_000 / (72_000 - 100)) == 3
    assert o.status == "ok"


def test_episode_cap_limits_containers():
    wl = WorkloadSummary.from_hours(1, episodes=5)
    o = build_option(PlanRequest(wl, presets=("exact",), max_gpus=10), one_rate_table(),
                     pinned_rates(), "exact", 4)
    assert o.kinds[0].containers == 2  # ceil(5 episodes / 4 per container)


def test_100_video_hours_matches_w3a_dollars_per_hour():
    wl = WorkloadSummary.from_hours(100)
    req = PlanRequest(wl, presets=("exact",), max_gpus=10)
    o = build_option(req, RateTable.load(), pinned_rates(), "exact", 4)
    assert o.cost_usd / 100 == pytest.approx(23.2, abs=0.6)  # ~$23 per video-hour


def test_options_deterministic_and_bounded():
    req = PlanRequest(WorkloadSummary.from_hours(100), deadline_s=48 * 3600, budget_usd=3000,
                      max_gpus=10)
    a, b = plan_options(req), plan_options(req)
    assert a == b
    assert 2 <= len(a) <= 4
    assert {o.preset for o in a} == {"exact", "balanced"}
    assert [o.name for o in a] == ["exact x8", "exact x4", "balanced x1", "exact x8 fastest"]
    best = recommended(a)
    assert best is not None and best.certification == "reference"
    bal = next(o for o in a if o.preset == "balanced")
    assert bal.certification == "needs certificates"
    assert any("placeholder" in f for f in bal.flags)


def test_sequential_dominated_option_dropped():
    req = PlanRequest(WorkloadSummary.from_hours(100), presets=("exact",), max_gpus=10)
    names = [o.name for o in plan_options(req)]
    assert "exact x1" not in names and "exact x4" in names


def test_deadline_infeasible():
    req = PlanRequest(WorkloadSummary.from_hours(100), presets=("exact",), max_gpus=2,
                      deadline_s=24 * 3600)
    opts = plan_options(req)
    assert all(o.status == "infeasible" for o in opts)
    assert all(any("misses the 24.0 h deadline" in f for f in o.flags) for o in opts)
    assert recommended(opts) is None


def test_budget_infeasible_and_at_risk():
    wl = WorkloadSummary.from_hours(100)
    over = build_option(PlanRequest(wl, presets=("exact",), max_gpus=10, budget_usd=100),
                        RateTable.load(), pinned_rates(), "exact", 4)
    assert over.status == "infeasible" and any("over budget" in f for f in over.flags)
    mid = over.cost_usd
    risk = build_option(PlanRequest(wl, presets=("exact",), max_gpus=10, budget_usd=mid + 1),
                        RateTable.load(), pinned_rates(), "exact", 4)
    assert risk.status == "at risk" and any("budget at risk" in f for f in risk.flags)


def test_gpu_cap_below_minimum():
    req = PlanRequest(WorkloadSummary.from_hours(1), stages=("reward.rubric",),
                      presets=("exact",), max_gpus=1)
    o = plan_options(req)[0]
    assert o.status == "infeasible" and "needs at least 2 GPUs" in o.flags[0]


def test_mixed_cameras_and_rewards():
    wl = WorkloadSummary((WorkloadGroup("human", 1, 3600, 60), WorkloadGroup("yam", 3, 3600, 60)))
    req = PlanRequest(wl, stages=("segment", "caption", "reward.dense"), presets=("exact",),
                      max_gpus=6)
    o = build_option(req, RateTable.load(), pinned_rates(), "exact", 4)
    seg = next(k for k in o.kinds if k.kind == "segcap")
    assert seg.work_s == pytest.approx(3600 * 4.08 + 3600 * 2.39)  # 3-view falls back to x2
    assert {k.kind for k in o.kinds} == {"segcap", "reward.dense"}
    assert o.peak_gpus <= 6


def test_workload_constructors():
    assert WorkloadSummary.from_hours(1).episodes == 60
    wl = WorkloadSummary.from_dict({"groups": [{"pool": "dex", "cameras": 3, "video_s": 10,
                                                "episodes": 2}]})
    assert wl.video_s == 10 and wl.groups[0].cameras == 3
    with pytest.raises(ValueError):
        WorkloadSummary.from_dict({"groups": []})
    with pytest.raises(ValueError):
        WorkloadSummary.from_hours(0)


def test_apply_option_round_trips_config():
    data = yaml.safe_load(template_text("exact_mp4_dir"))
    req = PlanRequest(WorkloadSummary.from_hours(100), presets=("exact",), max_gpus=10)
    o = plan_options(req)[0]
    new = apply_option(data, o, budget_usd=2500)
    cfg = parse_config(new)
    pool = cfg.pools()["segcap"]
    assert pool.max_containers == o.kinds[0].containers
    assert pool.max_inputs == o.kinds[0].max_inputs
    assert cfg.modal.max_gpus >= o.peak_gpus and cfg.modal.budget_usd == 2500
    bal = plan_options(PlanRequest(WorkloadSummary.from_hours(100), presets=("balanced",),
                                   max_gpus=10))[0]
    cfg2 = parse_config(apply_option(data, bal))
    assert cfg2.preset == "balanced" and "segcap" not in cfg2.resources
