"""``scripts/_runbook.py``: the state, billing and stop rules the runbooks share. The fixture is
the shape of the 2026-10-22 ``reports/wave-h/state.json`` (numbers and app names only; the real
file names private episodes and is never copied into the tests)."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
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
rb = sys.modules["_runbook"]

LAB_RATE = 0.0014154444444444446  # the lab step's peak $/s (the H100 Captioner)


def _att(started, finished, *, status, tracked, budget, rate, apps, heartbeat=None,
         report_error=False):
    return {"apps": apps, "budget_usd": budget, "finished": finished,
            "heartbeat": heartbeat if heartbeat is not None else finished, "notes": [],
            "rate_usd_per_s": rate, "report_error": report_error, "started": started,
            "status": status, "tracked_usd": tracked}


# 2026-10-22, in the format the runbook wrote that day (no plan, no watermark, no app ids).
TODAY = {
    "run_id": "102207184317", "created": 1792633721.916692, "billing_since": "2026-10-21",
    "billing_baseline": {"flora-convert": 0.77250759, "flora-reference": 10.35391884},
    "billed_at": 1792651185.930021, "billed_total": 3.2821, "cap_override": 7.9,
    "steps": {
        "tempseg": {"status": "done", "billed_usd": 0.3122, "notes": [], "result": {},
                    "projected_usd": None, "projected_upper_usd": None, "attempts": [
                        _att(1792633729.27, 1792634100.68, status="failed", tracked=0.1729,
                             budget=0.75, rate=0.0007532222, apps=["flora-convert"]),
                        _att(1792634151.10, 1792634437.66, status="failed", tracked=0.1647,
                             budget=0.5771, rate=0.0007532222, apps=["flora-convert"]),
                        _att(1792634510.71, 1792634513.59, status="needs_pin", tracked=0.0,
                             budget=0.4124, rate=0.0007532222, apps=[]),
                        _att(1792634678.98, 1792634682.41, status="done", tracked=0.0,
                             budget=0.4124, rate=0.0007532222, apps=[])]},
        "preflight": {"status": "done", "billed_usd": 0.0, "notes": [], "result": {},
                      "attempts": [_att(1792634689.05, 1792634701.47, status="done",
                                        tracked=0.0003, budget=0.05, rate=3.5111e-05,
                                        apps=["flora-balanced-lab"])]},
        "balanced": {"status": "done", "billed_usd": 1.3081, "result": {},
                     "notes": ["deployed job done; its lab part moved to step lab (d2), re-run "
                               "there"],
                     "attempts": [_att(1792634719.44, 1792636837.77, status="failed",
                                       tracked=2.8581, budget=3.7, rate=0.0021686667,
                                       apps=["flora-wave-h-102207184317-balanced-human-"
                                             "04f22b64fdc6", "flora-balanced-lab"])]},
        "lab": {"status": "done", "billed_usd": None, "result": {},
                "notes": ["error: BlockingIOError(36, 'File name too long')"],
                "attempts": [
                    _att(1792646076.1553621, 1792646609.788575, heartbeat=1792646586.2775989,
                         status="failed", tracked=0.722, budget=2.35, rate=LAB_RATE,
                         apps=["flora-balanced-lab"], report_error=True),
                    _att(1792650123.10, 1792650131.79, status="failed", tracked=0.0,
                         budget=0.0, rate=LAB_RATE, apps=["flora-balanced-lab"]),
                    _att(1792650191.26, 1792650198.73, status="failed", tracked=0.0,
                         budget=0.0, rate=LAB_RATE, apps=["flora-balanced-lab"]),
                    _att(1792650292.51, 1792651184.07, status="done", tracked=1.2441,
                         budget=1.628, rate=LAB_RATE, apps=["flora-balanced-lab"])]},
    },
}


def today() -> dict:
    return copy.deepcopy(TODAY)


def test_wave_h_re_exports_the_shared_objects():
    for name in ("Attempt", "StepState", "State", "StopRule", "Heartbeat"):
        assert getattr(wh, name) is getattr(rb, name)
    assert wh.BILLING_LAG_S == rb.BILLING_LAG_S and wh.CLIENT_SLACK_S == rb.CLIENT_SLACK_S


def test_todays_state_loads_and_round_trips(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(TODAY))
    st = wh.load_state(path)
    assert [a.status for a in st.steps["lab"].attempts] == ["failed", "failed", "failed", "done"]
    assert st.override_for("balanced-only") == 7.9 and st.override_for("certified") is None
    assert st.plan is None and st.gate_cap is None
    wh.save_state(st, path)
    back = wh.load_state(path)
    assert back.to_json() == st.to_json()
    assert back.spent == pytest.approx(st.spent)


@pytest.mark.skipif(not os.environ.get("FLORA_REAL_STATE"),
                    reason="set FLORA_REAL_STATE to a state.json to check it (read-only)")
def test_a_real_state_file_loads_from_a_copy(tmp_path):
    copied = tmp_path / "state.json"
    shutil.copyfile(os.environ["FLORA_REAL_STATE"], copied)  # the original is never written
    st = wh.load_state(copied)
    wh.save_state(st, copied)
    assert wh.load_state(copied).to_json() == st.to_json()


# --- WP2: the billing watermark and per-attempt reconciliation ---------------------------------

H = 3600.0
T0 = 1792645200.0  # 2026-10-22 05:00 UTC, an hour start


def _rows(*rows):
    """``(object_id, description, hour index from T0, cost)`` -> spend.rows() rows."""
    return [{"object_id": oid, "description": d, "interval_start": T0 + h * H, "cost": c}
            for oid, d, h, c in rows]


def test_spend_rows_parse_interval_start():
    spend = sys.modules.get("spend") or _load("spend")
    assert spend.interval_start("2026-10-22T05:00:00+00:00") == T0
    assert spend.interval_start("2026-10-22T05:00:00Z") == T0
    assert spend.interval_start("2026-10-22T05:00:00") == T0  # naive: UTC
    assert spend.interval_start(None) is None and spend.interval_start("garbage") is None


def test_watermark_is_the_latest_billed_hours_end():
    b = rb.Billing.from_rows(_rows(("ap-1", "a", 0, 0.5), ("ap-2", "b", 2, 0.25),
                                   ("ap-1", "a", 1, 0.5)))
    assert b == {"a": 1.0, "b": 0.25} and b.through == T0 + 3 * H
    assert rb.Billing.from_rows([]).through is None
    st = rb.State("r", T0, "x")
    assert st.watermark is None
    st.billed_at = T0 + 5 * H  # an older state: no rows were kept, the lag rule applies
    assert st.watermark == T0 + 4 * H
    rb.refresh_billing(st, now=b, clock=lambda: T0 + 3.5 * H)
    assert st.billed_through == T0 + 3 * H and st.watermark == T0 + 3 * H
    rb.refresh_billing(st, now=rb.Billing.from_rows(_rows(("ap-1", "a", 0, 0.5))))
    assert st.billed_through == T0 + 3 * H  # a later read never moves it back


def _killed(started, last, *, apps=(), ids=(), budget=1.0, tracked=0.2):
    return rb.Attempt(started=started, status="killed", heartbeat=last, budget_usd=budget,
                      tracked_usd=tracked, apps=list(apps), app_ids=list(ids))


def test_an_attempt_is_held_until_billing_covers_its_hours_then_billed():
    st = rb.State("r", T0, "x")
    a = _killed(T0 + 0.5 * H, T0 + 1.25 * H, apps=["flora-wh-r-lab-a1"], ids=["ap-lab1"])
    st.step("lab").attempts.append(a)
    st.step("lab").status = "failed"
    # billing covers 05:00-06:00 only: the attempt ran into 06:15, so it is held at its budget
    rb.refresh_billing(st, now=rb.Billing.from_rows(_rows(
        ("ap-lab1", "flora-wh-r-lab-a1", 0, 0.30))))
    assert not a.settled(st.watermark) and st.spent == pytest.approx(1.0)
    # 06:00-07:00 lands: settled, charged its apps' billed cost across both hours
    rb.refresh_billing(st, now=rb.Billing.from_rows(_rows(
        ("ap-lab1", "flora-wh-r-lab-a1", 0, 0.30), ("ap-lab1", "flora-wh-r-lab-a1", 1, 0.15))))
    assert a.settled(st.watermark) and a.billed_usd == pytest.approx(0.45)
    assert a.charged(st.watermark) == pytest.approx(0.45) and st.spent == pytest.approx(0.45)


def test_billing_rows_outside_an_attempts_hours_are_not_its():
    st = rb.State("r", T0, "x")
    a = rb.Attempt(started=T0 + 1.1 * H, status="done", finished=T0 + 1.2 * H,
                   tracked_usd=0.10, app_ids=["ap-x"])
    st.step("tempseg").attempts.append(a)
    rb.refresh_billing(st, now=rb.Billing.from_rows(_rows(
        ("ap-x", "flora-wh-r-tempseg-a1", 0, 9.0),  # an hour before it started
        ("ap-x", "flora-wh-r-tempseg-a1", 1, 0.12), ("ap-y", "other", 2, 1.0))))
    assert a.billed_usd == pytest.approx(0.12) and a.charged(st.watermark) == pytest.approx(0.12)


def test_a_legacy_shared_name_is_not_attributed_to_one_attempt():
    st = rb.State("r", T0, "x", {"flora-balanced-lab": 0.0})
    pre = rb.Attempt(started=T0 + 0.1 * H, status="done", finished=T0 + 0.2 * H,
                     tracked_usd=0.0003, apps=["flora-balanced-lab"])
    lab = _killed(T0 + 0.3 * H, T0 + 0.5 * H, apps=["flora-balanced-lab"], tracked=0.72,
                  budget=2.35)
    st.step("preflight").attempts.append(pre)
    st.step("lab").attempts.append(lab)
    rb.refresh_billing(st, now=rb.Billing.from_rows(_rows(
        ("ap-old", "flora-balanced-lab", 0, 0.80), ("ap-z", "z", 1, 0.0))))
    assert pre.billed_usd is None and lab.billed_usd is None  # the same hour, the same name
    assert lab.settled(st.watermark) and lab.charged(st.watermark) == pytest.approx(0.72)
    assert st.step("lab").billed_usd is None  # shared with preflight: the global total only
    assert st.billed_total == pytest.approx(0.80) and st.spent == pytest.approx(0.80)
