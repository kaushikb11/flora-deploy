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


# --- WP3: bounded charging, hold_clears_at, stop by id ----------------------------------------


def _lab_attempt_1(*, stopped: bool) -> rb.Attempt:
    """Today's lab attempt 1: crashed at 05:23:29 UTC with $0.72 tracked (heartbeat 23 s
    earlier), its ``with app.run()`` block exited (the app stopped) as the error propagated."""
    a = wh.Attempt(**today()["steps"]["lab"]["attempts"][0])
    a.slack_s = wh.LAB_SLACK_S
    a.stopped_at = a.finished if stopped else None
    return a


def test_todays_crashed_lab_attempt_is_charged_its_time_plus_slack_not_its_budget():
    a = _lab_attempt_1(stopped=True)
    through = rb.hour_floor(a.finished)  # billing covers 05:00 UTC, not the crash's hour
    ran = (a.finished - a.started) * LAB_RATE  # 533.6 s at the Captioner's $/s
    assert a.charged(through) == pytest.approx(ran + wh.LAB_SLACK_S * LAB_RATE, abs=1e-4)
    assert a.charged(through) == pytest.approx(0.84, abs=0.005)  # was its $2.35 budget
    assert a.held(through) and rb.utc(a.hold_clears_at()) == "2026-10-22 06:15 UTC"
    # without a confirmed stop nothing bounds it: its budget, as before
    assert _lab_attempt_1(stopped=False).charged(through) == pytest.approx(2.35)
    # billing covers the hour: tracked (the shared legacy name gives it no billed cost)
    assert a.charged(rb.hour_floor(a.finished) + rb.HOUR_S) == pytest.approx(0.722)


def test_a_killed_attempt_is_bounded_by_its_stop_or_its_hard_timeout():
    a = rb.Attempt(started=T0, status="killed", heartbeat=T0 + 300, tracked_usd=0.30,
                   rate_usd_per_s=0.001, budget_usd=5.0, slack_s=60, apps=["x"])
    assert a.bounded_usd() is None and a.charged(None) == 5.0  # nothing known: its budget
    a.hard_timeout_s = 2400.0  # its containers die by then
    assert a.charged(None) == pytest.approx(0.001 * (2400 + 60))
    a.stopped_at = T0 + 900.0  # the next run stopped its apps by id
    assert a.charged(None) == pytest.approx(0.001 * (900 + 60))
    assert a.settle_point == T0 + 960.0
    a.stopped_at = T0 + 9000.0  # a late stop: the hard timeout still bounds it
    assert a.charged(None) == pytest.approx(0.001 * (2400 + 60))
    a.budget_usd = 1.0
    assert a.charged(None) == pytest.approx(1.0)  # never more than its budget


class _Modal:
    """``subprocess.run`` for ``modal app list --json`` / ``modal app stop``."""

    def __init__(self, apps, *, list_rc=0, stop_rc=0):
        self.apps, self.list_rc, self.stop_rc, self.calls = apps, list_rc, stop_rc, []

    def __call__(self, args, **kw):
        self.calls.append(args)
        out = json.dumps(self.apps) if args[:3] == ["modal", "app", "list"] else ""
        rc = self.list_rc if args[:3] == ["modal", "app", "list"] else self.stop_rc
        return type("P", (), {"returncode": rc, "stdout": out, "stderr": ""})()


def test_stop_apps_stops_ephemeral_apps_by_id():
    fake = _Modal([{"app_id": "ap-1", "description": "flora-balanced-lab", "state": "ephemeral"},
                   {"app_id": "ap-0", "description": "flora-balanced-lab", "state": "stopped"},
                   {"app_id": "ap-9", "description": "other", "state": "deployed"}])
    assert rb.stop_apps(["flora-balanced-lab"], ["ap-7"], run=fake)
    stops = [c for c in fake.calls if c[:3] == ["modal", "app", "stop"]]
    assert stops == [["modal", "app", "stop", "--yes", "ap-1"]]  # never the bare name
    assert not rb.stop_apps(["flora-balanced-lab"], run=_Modal(fake.apps, stop_rc=1))
    unlisted = _Modal([], list_rc=1)
    assert not rb.stop_apps(["dep-app"], ["ap-2"], run=unlisted)  # tried, not confirmed
    assert [c[-1] for c in unlisted.calls[1:]] == ["ap-2", "dep-app"]


def test_the_next_run_stops_a_killed_attempts_apps_and_bounds_it():
    st = rb.State("r", T0, "x")
    a = rb.Attempt(started=T0, status="killed", heartbeat=T0 + 100, tracked_usd=0.1,
                   rate_usd_per_s=0.001, budget_usd=2.0, slack_s=60, hard_timeout_s=1200.0,
                   apps=["flora-wh-r-lab-a1"],
                   app_ids=["ap-lab"])
    st.step("lab").attempts.append(a)
    wh.stop_killed(st, run=_Modal([{"app_id": "ap-lab", "description": "flora-wh-r-lab-a1",
                                    "state": "ephemeral"}]), out=lambda s: None)
    assert a.stopped_at is not None and a.charged(None) == pytest.approx(0.001 * (1200 + 60))
    lines = wh.held_lines(st)
    assert len(lines) == 1 and "lab attempt 1" in lines[0] and lines[0].endswith("UTC")


# --- WP4: one app name per step and attempt ----------------------------------------------------


def test_every_attempt_of_every_step_has_its_own_app_name():
    names = {wh.app_name("102207184317", s, n) for s in wh.STEPS for n in (1, 2, 12)}
    assert len(names) == 3 * len(wh.STEPS)
    assert wh.app_name("102207184317", "exact_e2e", 3) == "flora-wh-102207184317-exact-e2e-a3"
    assert max(len(n) + len("-equivalence") for n in names) < 64  # Modal's name limit
    st = wh.State("102207184317", 0.0, "x")
    ctx = wh.Ctx(st, {}, Path("."), step="lab")
    st.step("lab").attempts += [wh.Attempt(0.0, "failed"), wh.Attempt(1.0)]
    assert ctx.app_name() == "flora-wh-102207184317-lab-a2"
    assert ctx.app_name("merge") == "flora-wh-102207184317-lab-a2-merge"


# --- WP5: one budget rule, one decision ---------------------------------------------------------

RETRY_AT = 1792650123.10  # 06:22:03 UTC (11:52 IST): today's lab retry, attempt 2


def _at_the_retry(*, stopped: bool) -> dict:
    """The state as the 11:52 retry found it: lab attempt 1 crashed, billing read then."""
    d = today()
    lab = d["steps"]["lab"]
    lab["status"], lab["attempts"] = "failed", lab["attempts"][:1]
    if stopped:  # what the runbook records now: the app.run() block exited, the slack
        lab["attempts"][0].update(stopped_at=lab["attempts"][0]["finished"],
                                  slack_s=wh.LAB_SLACK_S)
    d["billed_at"] = RETRY_AT - 60  # read before the step: no watermark, the lag rule
    return d


def _replay(monkeypatch, tmp_path, d, argv=()):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(d))
    monkeypatch.setattr(wh, "load_manifest", lambda offline, version="golden/v1":
                        wh.offline_manifest(wh.DURATION_HEADROOM, version))
    monkeypatch.setattr(wh, "refresh_billing", lambda st, **kw: False)  # billing as it was
    monkeypatch.setattr(wh, "LAB_OUT", tmp_path / "lab")  # no saved lab results
    monkeypatch.setattr(wh, "REPORT", tmp_path / "runbook.md")
    monkeypatch.chdir(tmp_path)
    seen = {}

    def lab(ctx):
        seen["budget"] = ctx.budget_of("lab")
        seen["attempt"] = ctx.st.step("lab").current.budget_usd
        return {"ok": True, "tracked_usd": 1.2441, "apps": [ctx.app_name()]}

    monkeypatch.setitem(wh.RUNNERS, "lab", lab)
    rc = wh.main(["--yes-spend", "--only", "lab", "--state", str(state), "--work",
                  str(tmp_path / "work"), *argv])
    return rc, seen, wh.load_state(state)


def test_replay_todays_lab_retry_runs_at_the_approved_global_cap(monkeypatch, tmp_path):
    d = _at_the_retry(stopped=True)
    st = wh.State.from_json(d)
    a1 = st.steps["lab"].attempts[0]
    assert a1.held(st.watermark) and a1.charged(st.watermark) == pytest.approx(0.84, abs=0.005)
    m = wh.offline_manifest(wh.DURATION_HEADROOM)
    ps = wh.plan_projections("balanced-only", m)
    dec = wh.decide(st, "balanced-only", "lab", ps)
    # G = $7.90 (cap_override): the lab gets its whole per-attempt cap and runs. The 11:52 rule
    # (cap - earlier attempts: $2.35 - $2.35) gave it $0 and skipped it inside the step.
    assert dec.action == "run" and dec.budget.binding == "attempt cap"
    assert dec.budget.usd == pytest.approx(wh.CAPS_USD["lab"])
    assert dec.budget.usd >= ps["lab"].usd_upper
    assert dec.budget.terms["global cap"] == 7.9
    rc, seen, after = _replay(monkeypatch, tmp_path, d)
    assert rc == 0 and after.steps["lab"].status == "done"
    # the stop rule, the attempt's budget and the in-step budget are the same number
    assert seen["attempt"] == seen["budget"].usd == dec.budget.usd
    assert after.steps["lab"].attempts[-1].apps == ["flora-wh-102207184317-lab-a2"]


def test_replay_runs_even_from_the_state_as_it_was_recorded(monkeypatch, tmp_path):
    """Without a recorded stop, attempt 1 is still held at its $2.35 budget, but earlier
    attempts no longer eat into the per-attempt cap: $7.90 - $5.55 spent leaves $2.35."""
    d = _at_the_retry(stopped=False)
    st = wh.State.from_json(d)
    assert st.steps["lab"].attempts[0].charged(st.watermark) == pytest.approx(2.35)
    rc, seen, _after = _replay(monkeypatch, tmp_path, d)
    assert rc == 0 and seen["budget"].usd >= wh.project(
        "lab", wh.offline_manifest(wh.DURATION_HEADROOM)).usd_upper


def test_the_stop_message_names_the_binding_term():
    m = wh.offline_manifest(wh.DURATION_HEADROOM)
    ps = wh.plan_projections("balanced-only", m)
    st = wh.State("r", 0.0, "x")
    spent = wh.Attempt(0.0, "done", finished=1.0, tracked_usd=3.0)
    st.step("tempseg").attempts.append(spent)
    st.cap_override = 7.0
    # $4.00 left: (d) fits it alone, but the cap keeps the lab's upper projection for later
    d = wh.decide(st, "balanced-only", "balanced", ps)
    assert d.action == "stop" and d.budget.binding == "reserve for lab"
    assert "binding: reserve for lab" in d.message and "kept for lab" in d.message
    assert d.budget.terms["reserve"] == pytest.approx(ps["lab"].usd_upper)
    assert wh.decide(st, "balanced-only", "lab", ps).runs  # the last step keeps nothing back
    spent.tracked_usd = 5.0  # $2.00 left
    d = wh.decide(st, "balanced-only", "lab", ps)
    assert d.action == "stop" and "binding: global left" in d.message
    assert "--global-cap 7.35" in d.command and "approval" in d.command
    big = wh.Projection("lab")
    big.add("x", wh.LAB["captioner"], 4000.0)
    with pytest.raises(wh.StopRule, match=r"over its cap \$2.35 \(binding: attempt cap\)"):
        wh.check_before(wh.State("r", 0.0, "x"), "lab", big)


def test_a_stop_that_only_held_attempts_cause_is_a_wait_with_its_clear_time():
    m = wh.offline_manifest(wh.DURATION_HEADROOM)
    ps = wh.plan_projections("balanced-only", m)
    st = wh.State.from_json(_at_the_retry(stopped=False))
    st.cap_override = 6.5  # $6.50 - $5.55 spent: no room while attempt 1 holds $2.35
    d = wh.decide(st, "balanced-only", "lab", ps)
    assert d.action == "wait" and d.clears_at == st.steps["lab"].attempts[0].hold_clears_at()
    assert "lab attempt 1 $2.35" in d.message and d.message.endswith("UTC")
    assert d.command.startswith("after 2026-10-22 06:15 UTC: ")
    assert any("held until billing covers it" in r for r in d.reasons)


def test_dry_run_rule_and_the_real_runs_budget_use_one_formula():
    m = wh.offline_manifest(wh.DURATION_HEADROOM, "golden/v2")
    ps = wh.plan_projections("certified", m)
    st = wh.State("r", 0.0, "x", plan="certified")
    caps, g = wh.plan_caps("certified", ps), wh.plan_global_cap("certified", ps)
    for s in wh.PLANS["certified"]:
        d = wh.decide(st, "certified", s, ps)
        res = wh.reserve_for(st, wh.PLANS["certified"], s, ps)
        assert d.runs and d.budget.usd == wh.check_before(
            st, s, ps[s], caps=caps, global_cap=g, reserve=res).usd == wh.step_budget(
            st, s, caps=caps, global_cap=g, reserve=res)
