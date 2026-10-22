"""wave_h step 0: NVIDIA's qualified GDN pass statuses count as a pass (2026-10-22: the real
check reported PASS_NATIVE_GDN_FP32_CHECKPOINT_VALUES and step 0 refused to pin)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wave_h", ROOT / "scripts" / "wave_h.py")
wh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wh)


def res(gdn):
    w = {"inputs_identical": True, "ids_identical": True, "json_equal": True,
         "teacher_forced_on_a": {"a_vs_b": {"max_abs": 0.375}}}
    return {"windows": [w, w], "b_gdn": gdn, "verify_merged": True, "total_s": 1.0}


def test_qualified_pass_statuses_pass():
    ok, _ = wh.equivalence_ok(res({"vs_base_checkpoint": "PASS_NATIVE_GDN_FP32_CHECKPOINT_VALUES",
                                   "vs_merged": "PASS"}))
    assert ok


def test_anything_else_fails():
    for bad in ("FAIL", "FAIL_PASS", "passed?", "", None):
        ok, _ = wh.equivalence_ok(res({"vs_merged": bad}))
        assert not ok, bad
    assert not wh.equivalence_ok(res({}))[0]  # no GDN verdict at all is not a pass


def test_global_cap_override_raises_never_lowers():
    proj = {s: wh.Projection(step=s, items=[]) for s in wh.STEPS}
    derived = wh.plan_global_cap("balanced-only", proj)
    assert wh.plan_global_cap("balanced-only", proj, derived + 1.0) == derived + 1.0
    assert wh.plan_global_cap("balanced-only", proj, derived - 1.0) == derived


def test_cap_override_round_trips_through_the_state():
    st = wh.State("r", 0.0, "2026-10-22T00:00:00Z", cap_override=5.6)
    assert wh.State.from_json(st.to_json()).cap_override == 5.6
    assert wh.State.from_json({**st.to_json(), "cap_override": None}).cap_override is None
