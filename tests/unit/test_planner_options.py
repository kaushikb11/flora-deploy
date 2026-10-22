"""`flora plan` options: hand-checked cost/wall, determinism, infeasibility, write-back."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import yaml

from flora_deploy.cli.init import template_text
from flora_deploy.config.schema import parse_config
from flora_deploy.jobs.budget import pinned_rates
from flora_deploy.planner.options import (
    ORCHESTRATOR_CORES,
    ORCHESTRATOR_GIB,
    POOL_END_TAIL_S,
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
ORCH_PER_S = (ORCHESTRATOR_CORES * 0.0472 + ORCHESTRATOR_GIB * 0.008) / 3600  # preemptible


def one_rate_table(rate=4.0, startup=100.0, u=0.0) -> RateTable:
    return RateTable.from_dict({
        "schema": 1,
        "rates": [{"preset": "exact", "kind": "segcap", "cameras": 1, "concurrency": 4,
                   "container_s_per_video_s": rate, "uncertainty": u, "status": "measured",
                   "provenance": "test", "date": "2026-10-21"}],
        "startup": [{"preset": "exact", "kind": "segcap", "startup_s": startup,
                     "uncertainty": 0.0, "status": "measured", "provenance": "test",
                     "date": "2026-10-21"}]}, source="test")


def shipped_with_balanced(status: str) -> RateTable:
    """The shipped rates.json with its balanced segment/caption slot rows (and per-kind balanced
    startup rows) replaced by ones built here: ``placeholder`` (the rows as they stood before
    Wave H-lite) or ``measured`` with n=1, ±50%. Keeps these tests independent of whether the
    shipped slot is measured today."""
    import json
    from importlib import resources

    from flora_deploy.planner.rates import MEASURED_SLOTS

    d = json.loads(resources.files("flora_deploy.planner").joinpath("rates.json").read_text())
    rate = {"segment": 0.5, "caption": 1.0}
    rows = []
    for preset, kind in MEASURED_SLOTS:
        row = {"preset": preset, "kind": kind, "cameras": "*", "concurrency": 1,
               "container_s_per_video_s": rate[kind], "uncertainty": 1.0,
               "status": "placeholder", "provenance": "test placeholder", "date": "2026-10-21"}
        if status == "measured":
            row.update(uncertainty=0.5, status="measured", source="test run", n=1,
                       provenance="test measurement")
        rows.append(row)
    d["rates"] = [r for r in d["rates"] if (r["preset"], r["kind"]) not in MEASURED_SLOTS] + rows
    d["startup"] = [s for s in d["startup"] if s["preset"] != "balanced" or s["kind"] == "*"]
    return RateTable.from_dict(d, source="test")


def test_hand_calc_single_kind():
    table = one_rate_table()
    wl = WorkloadSummary.from_hours(10, episodes=1000)  # 36,000 video-s -> 144,000 container-s
    req = PlanRequest(wl, presets=("exact",), max_gpus=4, deadline_s=None)
    o = build_option(req, table, pinned_rates(), "exact", 4)
    k = o.kinds[0]
    assert (k.containers, k.max_inputs, k.gpu) == (4, 4, "H200")
    assert o.wall_s == pytest.approx(100 + 144_000 / 4)
    # each container's pool-end tail: one poll + the drained 2 s window (jobs.pools)
    expect = (H200_PER_S * (144_000 + 4 * (100 + POOL_END_TAIL_S))
              + ORCH_PER_S * (100 + 36_000))
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


def test_100_video_hours_matches_s1_dollars_per_hour():
    wl = WorkloadSummary.from_hours(100)
    req = PlanRequest(wl, presets=("exact",), stages=("segment", "caption"), max_gpus=10)
    o = build_option(req, RateTable.load(), pinned_rates(), "exact", 4)
    # P1a-5s, the deployed worker (gpu_thread on): 6.4 at $5.68/h = ~$36.4 per video-h
    assert o.cost_usd / 100 == pytest.approx(36.4, abs=1.0)
    assert o.rate_status == "measured" and o.requires == ()
    assert not any(f.startswith("warning:") for f in o.flags)
    off = build_option(replace(req, features=()), RateTable.load(), pinned_rates(), "exact", 4)
    assert off.cost_usd / 100 == pytest.approx(40.0, abs=1.0)  # S1, gpu_thread off: ~$40
    g = build_option(req, RateTable.load(), pinned_rates(), "exact", 8,
                     extra_features=("long_lived_container",))
    assert g.requires == ("long_lived_container",)
    assert g.name == "exact x8 +long_lived_container"
    assert any("projection only" in f for f in g.flags)
    assert any(f.startswith("warning:") and "segcap (projected)" in f for f in g.flags)


def test_options_deterministic_and_bounded():
    req = PlanRequest(WorkloadSummary.from_hours(100), deadline_s=96 * 3600, budget_usd=6000,
                      max_gpus=10)
    a, b = plan_options(req), plan_options(req)
    assert a == b
    assert 2 <= len(a) <= 4
    assert {o.preset for o in a} == {"exact", "balanced"}
    # balanced (the primary preset) is listed first; `recommended` is a separate choice
    assert [o.name for o in a] == ["balanced x1", "exact x4", "exact x8 +long_lived_container"]
    best = recommended(a)
    assert best is not None and best.name == "exact x4" and best.certification == "reference"
    bal = next(o for o in a if o.preset == "balanced")
    assert bal.certification == "needs certificates"
    # a placeholder warning exactly when the shipped balanced rates are placeholders
    assert any("placeholder" in f for f in bal.flags) == (bal.rate_status == "placeholder")
    # placeholder-priced: flagged; measured: not, and the option list is the same
    for status, flagged in (("placeholder", True), ("measured", False)):
        t = shipped_with_balanced(status)
        opts = plan_options(req, table=t)
        assert opts == plan_options(req, table=t)
        assert [o.name for o in opts] == [o.name for o in a]
        assert recommended(opts).name == "exact x4"
        tb = next(o for o in opts if o.preset == "balanced")
        assert tb.certification == "needs certificates" and tb.rate_status == status
        assert any("placeholder" in f for f in tb.flags) is flagged, (status, tb.flags)


def test_recommended_never_picks_a_projection_and_prefers_measured():
    """Review 4 M5's case: 10 video-h used to recommend exact x8 ($164) on a gpu_thread
    projection. Review 5 M-C: the deployed worker runs gpu_thread, measured at 6.4 (P1a-5s):
    exact x4 is ~$379 (it was ~$414 on S1's gpu_thread-off 7.0)."""
    req = PlanRequest(WorkloadSummary.from_hours(10, episodes=1200), budget_usd=2000,
                      max_gpus=3, stages=("segment", "caption", "reward.terminal",
                                          "reward.dense"))
    opts = plan_options(req)
    best = recommended(opts)
    assert best.name == "exact x4" and best.requires == ()
    assert best.cost_usd == pytest.approx(379, abs=5) and best.rate_status == "measured"
    cheaper = [o for o in opts if o.cost_usd < best.cost_usd and o.preset == "exact"]
    assert cheaper and all(o.requires == ("long_lived_container",) for o in cheaper)
    # enabling a feature never lets a projection displace the measured row of the same
    # configuration (equal concurrency): x4 stays on 6.4 measured
    req_g = PlanRequest(req.workload, budget_usd=2000, max_gpus=3, stages=req.stages,
                        features=("gpu_thread", "long_lived_container"))
    opts_g = plan_options(req_g)
    assert all(not o.requires for o in opts_g)
    x4 = next(o for o in opts_g if o.name == "exact x4")
    assert x4.rate_status == "measured" and x4.cost_usd == pytest.approx(best.cost_usd)
    best_g = recommended(opts_g)
    assert best_g.name == "exact x4"  # measured before projected


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
    # the cu130 3-view rows are stale: both groups use the 1-/3-view-mix cu128 rate, the
    # 3-view one scaled by rates.json camera_scaling; rewards have no multiplier
    f = RateTable.load().camera_scaling.factor("segcap", 3)
    assert f > 1
    assert seg.work_s == pytest.approx(3600 * 6.4 + 3600 * 6.4 * f)
    dense = next(k for k in o.kinds if k.kind == "reward.dense")
    assert not dense.camera_scale
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
    # review 6 H2: the option's concurrency is episodes inside one input (segcap)
    assert (pool.max_inputs, pool.episode_concurrency) == (1, o.kinds[0].max_inputs)
    assert cfg.modal.max_gpus >= o.peak_gpus and cfg.modal.budget_usd == 2500
    bal = plan_options(PlanRequest(WorkloadSummary.from_hours(100), presets=("balanced",),
                                   max_gpus=10))[0]
    cfg2 = parse_config(apply_option(data, bal), allow_uncertified=True)  # no certificates yet
    assert cfg2.preset == "balanced" and "segcap" not in cfg2.resources


def test_placeholder_rates_are_labelled_estimate():
    from flora_deploy.planner.options import DEFAULT_PRESETS, rates_label

    assert DEFAULT_PRESETS[0] == "balanced" and "exact" in DEFAULT_PRESETS
    req = PlanRequest(WorkloadSummary.from_hours(100), max_gpus=10)
    opts = plan_options(req, table=shipped_with_balanced("placeholder"))
    bal = next(o for o in opts if o.preset == "balanced")
    assert bal.rate_status == "placeholder" and rates_label(bal) == "estimate (placeholder)"
    assert rates_label(next(o for o in opts if o.name == "exact x4")) == "measured"
    # a measured slot with n=1 shows as a measurement with its sample count and band
    opts = plan_options(req, table=shipped_with_balanced("measured"))
    bal = next(o for o in opts if o.preset == "balanced")
    assert bal.rate_status == "measured" and rates_label(bal) == "measured (n=1, ±50%)"
    assert rates_label(next(o for o in opts if o.name == "exact x4")) == "measured"


def test_never_recommends_an_unavailable_option():
    """Review 6 M2: exact over budget used to make the uncertified, placeholder-priced and
    factory-refused ``balanced`` the recommendation."""
    req = PlanRequest(WorkloadSummary.from_hours(10), budget_usd=80, max_gpus=10)
    rubric = replace(req, stages=("segment", "caption", "reward.rubric"))
    cases = [  # (request, table, expected placeholder kinds)
        (req, shipped_with_balanced("placeholder"), {"segment", "caption"}),
        (req, shipped_with_balanced("measured"), set()),
        (rubric, shipped_with_balanced("measured"), {"reward.rubric"}),  # still a placeholder
        (req, None, None), (rubric, None, None)]  # the shipped table, whatever it holds today
    for r, table, kinds in cases:
        opts = plan_options(r, table=table)
        bal = [o for o in opts if o.preset == "balanced"]
        assert bal, "balanced must still be shown"
        for o in bal:
            why = " ".join(o.unavailable)
            # uncertified, whatever its rates: never recommendable without certificates
            assert "uncertified" in why, why  # deployable since P1b-1
            assert any(f.startswith("unavailable:") for f in o.flags)
            ph = {k.kind for k in o.kinds if k.rate_status == "placeholder"}
            if kinds is not None:
                assert ph == kinds, (ph, kinds)
            # the placeholder reason exactly when (and naming exactly what) is placeholder-priced
            assert ("placeholder" in why) == bool(ph), why
            if ph:
                assert why.count("priced from placeholder rates") == 1
                assert all(k in why for k in ph), why
        best = recommended(opts)
        assert best is None or (best.preset == "exact" and not best.unavailable)


def test_exact_segcap_is_available():
    req = PlanRequest(WorkloadSummary.from_hours(10), presets=("exact",), max_gpus=10)
    assert all(not o.unavailable for o in plan_options(req) if not o.requires)


def test_certified_pools_drop_the_uncertified_reason():
    """``with_certificates``: an option whose every pool is certified (``cli.plan.
    certified_options``) loses its "uncertified" unavailable reason and flag, keeps the others,
    and can be recommended once nothing else blocks it."""
    from dataclasses import replace

    from flora_deploy.planner.options import (
        PlanRequest,
        WorkloadSummary,
        plan_options,
        recommended,
        with_certificates,
    )

    req = PlanRequest(WorkloadSummary.from_hours(100), deadline_s=96 * 3600, budget_usd=6000,
                      max_gpus=10)
    opts = plan_options(req)
    bal = next(o for o in opts if o.preset == "balanced")
    assert any(u.startswith("uncertified") for u in bal.unavailable)
    out = with_certificates(opts, {bal.name})
    cbal = next(o for o in out if o.preset == "balanced")
    assert cbal.certification == "certified"
    assert not any(u.startswith("uncertified") for u in cbal.unavailable)
    assert set(cbal.unavailable) == set(bal.unavailable) - {
        u for u in bal.unavailable if u.startswith("uncertified")}
    assert not any("uncertified until" in f for f in cbal.flags)
    assert with_certificates(opts, set()) == list(opts)  # nothing certified: unchanged
    # once its rates are measured and its workers deploy, the certified, cheaper balanced wins
    ready = [replace(o, rate_status="measured", unavailable=()) if o is cbal else o for o in out]
    assert recommended(ready, certified={bal.name}).preset == "balanced"
    assert recommended([replace(bal, rate_status="measured", unavailable=tuple(
        u for u in bal.unavailable if u.startswith("uncertified")))] + opts[1:]).preset == "exact"
