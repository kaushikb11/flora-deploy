"""wave_h step 0: NVIDIA's qualified GDN pass statuses count as a pass (2026-10-22: the real
check reported PASS_NATIVE_GDN_FP32_CHECKPOINT_VALUES and step 0 refused to pin)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wave_h", ROOT / "scripts" / "wave_h.py")
wh = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules.setdefault("wave_h", wh)  # dataclasses look their module up while it loads
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


# --- review 12 F6 / F7: --global-cap is validated, scoped to its plan, saved only on a run ----


def _main(tmp_path, *args):
    import contextlib
    import io

    buf = io.StringIO()
    state = tmp_path / "state.json"
    with contextlib.redirect_stdout(buf):
        rc = wh.main([*args, "--offline", "--state", str(state), "--work", str(tmp_path / "w")])
    saved = wh.load_state(state) if state.exists() else None
    return rc, buf.getvalue(), saved


def test_a_dry_run_never_saves_the_override(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "REPORT", tmp_path / "runbook.md")
    rc, out, saved = _main(tmp_path, "--dry-run", "--global-cap", "9")
    assert rc == 0 and "would raise the global cap for plan balanced-only to $9.00" in out
    assert "user-approved override for plan balanced-only" in out  # shown, not approved
    assert saved is None or saved.cap_override is None
    rc, out, saved = _main(tmp_path, "--report", "--global-cap", "9")
    assert rc == 0 and (saved is None or saved.cap_override is None)


@pytest.mark.parametrize("cap", ["inf", "nan", "-1", "0", "50.01", "1e9"])
def test_non_finite_or_huge_caps_are_refused(tmp_path, cap):
    rc, out, saved = _main(tmp_path, "--dry-run", "--global-cap", cap)
    assert rc == 2 and "refused" in out
    assert saved is None or saved.cap_override is None


def test_a_cap_below_the_derived_one_is_refused_not_reported_as_raised(tmp_path):
    rc, out, _ = _main(tmp_path, "--dry-run", "--global-cap", "1")
    assert rc == 2 and "below the plan's derived cap" in out and "raised" not in out


def test_check_global_cap():
    assert wh.check_global_cap(9.0, spent=1.0, derived=5.4) is None
    assert wh.check_global_cap(wh.MAX_GLOBAL_CAP_USD, spent=0.0, derived=5.4) is None
    assert "ceiling" in wh.check_global_cap(51.0, spent=0.0, derived=5.4)
    assert "finite" in wh.check_global_cap(float("inf"), spent=0.0, derived=5.4)
    assert "finite" in wh.check_global_cap(float("nan"), spent=0.0, derived=5.4)
    assert "spent" in wh.check_global_cap(6.0, spent=7.0, derived=5.4)


def test_the_override_is_scoped_to_its_plan():
    proj = {s: wh.Projection(step=s, items=[]) for s in wh.STEPS}
    st = wh.State("r", 0.0, "x", cap_override=9.0, cap_override_plan="balanced-only")
    assert st.override_for("balanced-only") == 9.0 and st.override_for("full") is None
    assert wh.plan_global_cap("full", proj, st.override_for("full")) == wh.GLOBAL_CAP_USD
    back = wh.State.from_json(st.to_json())
    assert (back.cap_override, back.cap_override_plan) == (9.0, "balanced-only")
    legacy = wh.State.from_json({**st.to_json(), "cap_override_plan": None})
    assert legacy.override_for(wh.DEFAULT_PLAN) == 9.0 and legacy.override_for("full") is None
    assert "user-approved override" in wh.global_cap_label(st, "balanced-only", proj)
    assert wh.global_cap_label(st, "full", proj) == "GLOBAL_CAP_USD"


def test_finalize_reports_the_cap_in_force():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import wave_h_finalize as fz

    st = fz.wh.State("r", 0.0, "x")
    derived = fz.global_cap(st)
    st.cap_override, st.cap_override_plan = derived + 2.0, fz.wh.DEFAULT_PLAN
    assert fz.global_cap(st) == pytest.approx(derived + 2.0)
    st.cap_override_plan = "full"  # approved for another plan: not in force here
    assert fz.global_cap(st) == pytest.approx(derived)
    st.gate_cap, st.gate_cap_plan = 4.2, fz.wh.DEFAULT_PLAN  # what a run gated on wins
    assert fz.global_cap(st) == 4.2
    back = fz.wh.State.from_json(st.to_json())
    assert (back.gate_cap, back.gate_cap_plan) == (4.2, fz.wh.DEFAULT_PLAN)
    st.gate_cap_plan = "full"
    assert fz.global_cap(st) == pytest.approx(derived)
