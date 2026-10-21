"""``scripts/wave_h.py``: projections against the Wave H-lite caps, stop rules, resumable state,
the offline certification metrics and the numbers-only report. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wh = _load("wave_h")
h0 = sys.modules["h0_rehearsal"]


def _manifest() -> dict:
    """The golden durations the runbook prices (golden manifest v1, numbers only)."""
    dur = {"human/egodex_add_remove_lid/episode_000014": 14.6,
           "human/egodex_braid_unbraid/episode_000001": 16.1,
           "yam/aloha_static_cups_open/episode_000000": 8.0,
           "yam/aloha_static_towel/episode_000000": 10.0,
           "single_view_robot/wgo/robointer_droid_000010": 15.2,
           "single_view_robot/wgo/galaxea_045": 12.6}
    return {"episodes": [{"id": k, "duration_s": v} for k, v in dur.items()]}


def test_same_six_episodes_as_h0_and_subsets():
    ids = [i for v in h0.EPISODES.values() for i in v]
    assert len(ids) == 6 and len(set(ids)) == 6
    pools = {i.split("/")[0] for i in ids}
    assert pools == {"human", "yam", "single_view_robot"}
    assert set(wh.REWARD_EPISODES) <= set(ids) and len(wh.REWARD_EPISODES) == 3
    assert set(wh.NOISE_EPISODES) <= set(ids) and len(wh.NOISE_EPISODES) == 2


def test_projections_fit_every_cap():
    m = _manifest()
    total = 0.0
    for step in wh.STEPS:
        p = wh.project(step, m)
        assert p.usd <= wh.CAPS_USD[step] + 1e-9, (step, p.usd)
        assert p.usd_upper >= p.usd
        total += p.usd
    assert total <= wh.GLOBAL_CAP_USD
    # exact runs on the preset's H200 pool, balanced on L40S + H100
    assert {i["gpu"] for i in wh.project("exact_e2e", m).items} == {"H200"}
    assert {i["gpu"] for i in wh.project("balanced", m).items} == {"L40S", "H100"}


def test_container_prices_are_list_prices():
    from flora_deploy.jobs.budget import PINNED_PRICES

    c = wh.Container("H200", 8.0, 96.0)
    assert c.usd_per_h() == pytest.approx(PINNED_PRICES["H200"] + 8 * 0.0472 + 96 * 0.008)
    assert wh.Container(None, 2.0, 4.0).usd_per_h() == pytest.approx(2 * 0.0472 + 4 * 0.008)
    assert wh.Container("H100!", 1, 1).usd_per_s() == wh.Container("H100", 1, 1).usd_per_s()


def test_stop_rules_before_after_and_deadline():
    st = wh.State("r", 0.0, "2026-10-21")
    p = wh.Projection("exact_e2e")
    p.add("x", wh.Container("H200", 8, 96), 900)  # ~$1.42 > $1.20
    with pytest.raises(wh.StopRule, match="over its cap"):
        wh.check_before(st, "exact_e2e", p)
    st.step("balanced").tracked_usd = 2.5
    small = wh.Projection("exact_e2e")
    small.add("x", wh.Container("H200", 8, 96), 600)  # ~$0.95 fits the step, not the global
    with pytest.raises(wh.StopRule, match="global"):
        wh.check_before(st, "exact_e2e", small)
    st.step("rewards").billed_usd = 0.7  # billed beats tracked when larger
    with pytest.raises(wh.StopRule, match="rewards"):
        wh.check_after(st, "rewards")
    st2 = wh.State("r", 0.0, "2026-10-21")
    c = wh.Container("H200", 8, 96)
    assert wh.gpu_deadline_s(st2, "noise", c) == pytest.approx(0.40 / c.usd_per_s())
    assert wh.gpu_deadline_s(st2, "noise", c, spent_in_step=0.5) == 0.0


def test_state_round_trip_and_resume(tmp_path):
    st = wh.State("abc", 1.0, "2026-10-21", {"flora-x": 1.0})
    st.step("preflight").status = "done"
    st.step("preflight").tracked_usd = 0.01
    path = tmp_path / "state.json"
    wh.save_state(st, path)
    back = wh.load_state(path)
    assert back.steps["preflight"].status == "done" and back.spent == pytest.approx(0.01)
    assert back.billing_baseline == {"flora-x": 1.0}


def test_segment_metrics():
    ref = [{"start_s": 0, "end_s": 4, "caption": "grasp the cup"},
           {"start_s": 4, "end_s": 10, "caption": "place the cup"}]
    same = wh.compare_segments(ref, ref)
    assert same == {"count_delta": 0.0, "boundary_mae_s": 0.0, "segment_iou": 1.0,
                    "caption_jaccard": 1.0}
    shifted = wh.compare_segments(ref, [{"start_s": 0, "end_s": 5, "caption": "grasp the cup"},
                                        {"start_s": 5, "end_s": 10, "caption": "lift a box"}])
    assert shifted["boundary_mae_s"] == pytest.approx(1 / 3, abs=1e-3)
    assert 0 < shifted["segment_iou"] < 1 and 0 < shifted["caption_jaccard"] < 1
    assert wh.compare_segments(ref, [])["segment_iou"] == 0.0
    agg = wh.aggregate({"a": same, "b": shifted})
    assert agg["episodes"] == 2 and agg["segment_iou"] < 1


def test_certify_needs_l0_gate_a_and_metrics():
    good = {"boundary_mae_s": 0.2, "segment_iou": 0.9, "caption_jaccard": 0.8}
    l0 = [{"ids_equal": True, "grid_equal": True}]
    gate = {"max_abs_dlogit": 0.01}
    assert wh.certify(good, l0=l0, gate_a=gate)["provisional"]
    assert not wh.certify(good, l0=[{"ids_equal": False, "grid_equal": True}],
                          gate_a=gate)["provisional"]
    assert not wh.certify(good, l0=l0, gate_a=None)["provisional"]
    # against a noise floor: exact-vs-exact IoU 0.97 allows down to 0.94 with factor 2
    v = wh.certify(good, l0=l0, gate_a=gate,
                   noise={"boundary_mae_s": 0.05, "segment_iou": 0.97, "caption_jaccard": 0.95})
    assert v["against"] == "noise floor" and not v["checks"]["segment_iou"]


def test_report_is_numbers_only():
    st = wh.State("r", 0.0, "2026-10-21")
    s = st.step("certify")
    s.status = "done"
    s.result = {"verdict": {"provisional": False}, "rows": [{"caption": "PRIVATE"}],
                "per": {"x": {"task": "PRIVATE", "segment_iou": 0.5}}}
    text = wh.render_report(st, {k: wh.project(k, _manifest()) for k in wh.STEPS})
    assert "PRIVATE" not in text and "segment_iou" in text
    assert "| e. offline provisional certification | done |" in text


def test_dry_run_spends_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(h0, "golden_manifest", _manifest)

    def boom(*a, **k):
        raise AssertionError("dry run must not run a step")

    monkeypatch.setattr(wh, "RUNNERS", {k: boom for k in wh.STEPS})
    monkeypatch.setattr(wh, "billed_by_app", boom)
    state = tmp_path / "state.json"
    assert wh.main(["--dry-run", "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert "total projected" in out and "dry run" in out and not state.exists()
    assert json.dumps(wh.CAPS_USD)  # caps are plain data
