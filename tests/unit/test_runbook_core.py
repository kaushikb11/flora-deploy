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
