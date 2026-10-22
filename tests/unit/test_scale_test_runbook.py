"""``scripts/scale_test.py``: the dry-run plan, the spend stop rules, resume, and the pure
pieces (dataset pin check, chaos schedule, measurements). Nothing here contacts Modal or the Hub.
"""

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


sc = _load("scale_test")
wh = sys.modules["wave_h"]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No billing, no Modal: every runner is recorded instead of run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wh, "billed_by_app", lambda since: {})
    monkeypatch.setattr(wh, "refresh_billing", lambda st, **kw: False)
    calls: list[str] = []

    def fake(name):
        def run(ctx):
            calls.append(name)
            return {"ok": True, "tracked_usd": 0.0, "apps": []}
        return run

    monkeypatch.setattr(sc, "RUNNERS", {s: fake(s) for s in sc.STEPS})
    return calls


# --------------------------------------------------------------------------------------------
# dry run


def test_dry_run_is_the_default_and_starts_nothing(offline, tmp_path, capsys):
    state = tmp_path / "state.json"
    assert sc.main(["--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert offline == [] and not state.exists()
    assert "dry run: nothing started" in out
    assert sc.DATASET["revision"][:12] in out and "128 episodes" in out
    assert "flora plan options" in out  # the plan step's options, printed offline
    for t in sc.TITLES.values():
        assert t in out


def test_dry_run_numbers_fit_the_caps_honestly():
    for mode in sc.MODES:
        ps = sc.projections(mode)
        for name, p in ps.items():
            assert p.usd_upper >= p.usd >= 0
            assert p.usd_upper <= sc.caps(mode)[name] + 1e-9, (mode, name)
        assert sum(p.usd_upper for p in ps.values()) <= sc.global_cap(mode) + 1e-9
    real = sc.projections("real")
    assert sc.wh.honest_caps(real)["global"] <= sc.global_cap("real")
    run = real["run"]
    gpus = {i["gpu"] for i in run.items}
    assert {"L40S", "H100"} <= gpus  # balanced pools, priced as deployed
    assert run.peak_usd_per_s > 0
    assert all(p.usd == 0 for p in sc.projections("local").values())


def test_projection_counts_every_container_and_chaos_cost():
    run = sc.project("run", "real")
    whats = " | ".join(i["what"] for i in run.items)
    n = sc.pool_sizes()
    assert f"segment: {n['segment']} cold starts" in whats
    assert f"caption: {n['caption']} cold starts" in whats
    assert "preemption" in whats and "reconciler gap" in whats
    # upper adds the client slack per container on top of the scale-down tail
    tail = next(i for i in run.items if i["what"].startswith("caption: idle tail"))
    assert tail["upper_seconds"] >= n["caption"] * (300 + sc.CLIENT_SLACK_S)
    assert sc.run_wall_s("real", upper=True) > sc.run_wall_s("real")
    # more video costs more
    assert sc.project("run", "real", 2 * sc.VIDEO_S).usd > run.usd


def test_config_is_valid_for_every_mode(tmp_path, monkeypatch):
    from flora_deploy.config.schema import load_config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "synthetic").mkdir()
    for mode in sc.MODES:
        p = tmp_path / f"{mode}.yaml"
        p.write_text(sc.config_text(mode, project="scale-x", run_id="r1", budget_usd=5.0))
        cfg = load_config(p, allow_uncertified=True)
        assert cfg.preset == "balanced"
        pools = cfg.pools()
        assert pools["segment"].max_containers == 2 and pools["caption"].max_containers == 3
        if mode == "real":
            assert cfg.input.revision == sc.DATASET["revision"]
            assert cfg.input.repo == sc.DATASET["repo"]


# --------------------------------------------------------------------------------------------
# stop rules


def test_stop_rule_refuses_a_step_whose_upper_projection_is_over_its_cap(offline, tmp_path,
                                                                        monkeypatch):
    monkeypatch.setitem(sc.CAPS_USD, "real", {**sc.caps("real"), "run": 1.0})
    state = tmp_path / "state.json"
    assert sc.main(["--yes-spend", "--state", str(state), "--work", str(tmp_path / "w")]) == 3
    assert offline == ["verify", "index", "plan", "deploy"]  # run never started
    st = wh.load_state(state)
    assert st.step("run").status == "stopped"
    assert "over its cap" in st.step("run").notes[-1]
    assert not st.step("run").attempts


def test_stop_rule_uses_the_global_cap_minus_cumulative_spend(offline, tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    st = wh.load_state(state)
    # earlier spend (a killed index attempt charged at its budget) eats the global cap
    st.step("verify").status = "done"
    st.step("index").status = "done"
    st.step("index").attempts.append(wh.Attempt(started=1.0, status="done", tracked_usd=
                                                sc.global_cap("real") - 1.0))
    wh.save_state(st, state)
    assert sc.main(["--yes-spend", "--state", str(state), "--work", str(tmp_path / "w")]) == 3
    st = wh.load_state(state)
    # the $1.00 left cannot carry the run, so nothing is deployed for it either: the global cap
    # keeps the run's upper projection back from the steps before it
    assert st.step("deploy").status == "stopped" and not st.step("deploy").attempts
    assert "global budget left" in st.step("deploy").notes[-1]
    assert "binding: reserve for run" in st.step("deploy").notes[-1]
    assert offline == ["plan"] and st.step("run").status == "pending"


def test_spend_after_a_step_over_its_cap_stops_the_runbook(offline, tmp_path, monkeypatch):
    def overspend(ctx):
        return {"ok": True, "tracked_usd": sc.caps("real")["deploy"] + 1.0, "apps": []}

    monkeypatch.setitem(sc.RUNNERS, "deploy", overspend)
    state = tmp_path / "state.json"
    assert sc.main(["--yes-spend", "--state", str(state), "--work", str(tmp_path / "w")]) == 3
    st = wh.load_state(state)
    assert "stop rule after" in st.step("deploy").notes[-1]
    assert offline == ["verify", "index", "plan"]


def test_in_step_budget_is_the_attempt_cap_within_the_global_cap_left():
    st = wh.State("r", 0.0, "2026-10-22")
    s = st.step("run")
    s.attempts.append(wh.Attempt(started=1.0, status="failed", tracked_usd=4.0))
    c, g = sc.caps("real"), sc.global_cap("real")
    b = wh.budget_for(st, "run", caps=c, global_cap=g)
    # the cap is per attempt; the earlier attempt counts against the global cap
    assert b.usd == pytest.approx(min(c["run"], g - 4.0))
    assert wh.step_budget(st, "run", caps=c, global_cap=g) == b.usd


def test_modal_modes_need_yes_spend(offline, tmp_path):
    state = tmp_path / "state.json"
    assert sc.main(["--mode", "fake-modal", "--execute", "--state", str(state)]) == 2
    assert offline == []


# --------------------------------------------------------------------------------------------
# resume


def test_resume_skips_done_steps_and_retries_the_failed_one(offline, tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    n = {"run": 0}

    def flaky(ctx):
        n["run"] += 1
        if n["run"] == 1:
            raise RuntimeError("client died")
        return {"ok": True, "tracked_usd": 2.0, "apps": ["flora-scale-app"]}

    monkeypatch.setitem(sc.RUNNERS, "run", flaky)
    work = str(tmp_path / "w")
    assert sc.main(["--yes-spend", "--state", str(state), "--work", work]) == 1
    st = wh.load_state(state)
    assert st.step("run").status == "failed" and len(st.step("run").attempts) == 1
    # the failed attempt's spend is unknown: charged its budget until billing reconciles it
    first = st.step("run").attempts[0]
    assert first.report_error and st.step("run").spent == pytest.approx(first.budget_usd)
    # ... so a retry before billing caught up is refused: its upper projection no longer fits
    offline.clear()
    assert sc.main(["--yes-spend", "--state", str(state), "--work", work]) == 3
    st = wh.load_state(state)
    assert offline == [] and "global budget left" in st.step("run").notes[-1]
    # billing read an hour after the attempt: it is charged what it tracked, and the step runs
    st.billed_at = first.last_activity + wh.BILLING_LAG_S + 1
    wh.save_state(st, state)
    assert sc.main(["--yes-spend", "--state", str(state), "--work", work]) == 0
    # verify..plan done before; deploy again (the failed run stopped the app, M3) only now
    # that the run fits; run retried by `flaky`
    assert offline == ["deploy", "report"]
    st = wh.load_state(state)
    assert [a.status for a in st.step("run").attempts] == ["failed", "done"]
    # cumulative: the retry adds to the first attempt instead of replacing it
    assert st.step("run").spent_at(st.billed_at) == pytest.approx(first.tracked_usd + 2.0)
    assert all(st.step(s).status == "done" for s in sc.STEPS)
    assert (state.parent / "report.md").exists()


def test_killed_process_is_marked_and_its_apps_stopped(offline, tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    st = wh.load_state(state)
    for s in ("verify", "index", "plan", "deploy"):
        st.step(s).status = "done"
    st.step("run").status = "running"
    st.step("run").attempts.append(wh.Attempt(started=1.0, budget_usd=0.05, tracked_usd=0.01,
                                              apps=["flora-scale-app"]))
    wh.save_state(st, state)
    stopped: list[list[str]] = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **kw: stopped.append(args)
                        or type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    assert sc.main(["--yes-spend", "--state", str(state), "--work", str(tmp_path / "w")]) == 0
    assert ["modal", "app", "stop", "--yes", "flora-scale-app"] in stopped
    st = wh.load_state(state)
    a0 = st.step("run").attempts[0]
    assert a0.status == "killed" and a0.charged(None) == pytest.approx(0.05)
    assert offline == ["deploy", "run", "report"]  # its app was stopped: deployed again


def test_report_mode_rewrites_from_state_only(offline, tmp_path):
    state = tmp_path / "state.json"
    st = wh.load_state(state)
    st.step("report").status = "done"
    st.step("report").result = {"state": "done", "criteria": [{"criterion": "end state",
                                                            "verdict": "PASS"}],
                                "measure": {"pools": {}}}
    wh.save_state(st, state)
    assert sc.main(["--report", "--state", str(state)]) == 0
    text = (tmp_path / "report.md").read_text()
    assert "| 1 | end state | done | done | PASS |" in text and offline == []


# --------------------------------------------------------------------------------------------
# pure pieces


def test_dataset_pin_check_is_content_addressed():
    info = {"codebase_version": "v3.0", "total_episodes": 128, "total_frames": 105877, "fps": 30,
            "features": {"observation.images.camera": {"dtype": "video",
                                                       "info": {"video.codec": "av1"}}}}
    raw = json.dumps(info).encode()
    d = {**sc.DATASET, "info_blob_sha1": sc.git_blob_sha1(raw)}
    assert sc.check_info(raw, d) == []
    bad = json.dumps({**info, "total_frames": 1}).encode()
    probs = sc.check_info(bad, d)
    assert any("blob" in p for p in probs) and any("total_frames" in p for p in probs)
    v = sc.verify_dataset(fetch=lambda url: raw)
    assert sc.DATASET["revision"] in v["url"] and not v["ok"]  # blob differs from the pin
    assert abs(sc.VIDEO_S - 3529.2) < 0.1


def test_chaos_fires_each_action_once_at_its_point():
    c = sc.Chaos()

    def sample(seg, cap, *, alive=True, state="running", running=1):
        def k(done, total):
            return {"done": done, "running": running, "pending": total - done - running}
        return {"state": state, "orchestrator_alive": alive,
                "counts": {"segment": k(*seg), "caption": k(*cap)}}

    assert c.due(sample((0, 8), (0, 15))) is None
    assert c.due(sample((1, 8), (0, 15))) == "preempt_segment"
    c.fired["preempt_segment"] = {}
    assert c.due(sample((7, 8), (3, 15))) is None
    assert c.due(sample((8, 8), (4, 15))) == "preempt_caption"
    c.fired["preempt_caption"] = {}
    assert c.due(sample((8, 8), (8, 15), alive=False)) is None
    assert c.due(sample((8, 8), (8, 15))) == "kill_orchestrator"
    c.fired["kill_orchestrator"] = {}
    assert c.due(sample((8, 8), (14, 15))) is None
    assert sc.Chaos().due(sample((1, 8), (0, 15), state="done")) is None


def test_measure_utilization_cold_start_idle_and_pipelining():
    def ts(s):
        from datetime import UTC, datetime

        return datetime.fromtimestamp(1000.0 + s, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    files = {
        "attempts": [
            {"kind": "segment", "shard_id": "s1", "container_id": "ta-1", "started_utc": ts(100),
             "call_id": "c1"},
            {"kind": "segment", "shard_id": "s2", "container_id": "ta-1", "started_utc": ts(300),
             "call_id": "c2", "preempted": True, "ended_utc": ts(350)},
            {"kind": "caption", "shard_id": "k1", "container_id": "ta-2", "started_utc": ts(400),
             "call_id": "c3"}],
        "receipts": [
            {"kind": "segment", "shard_id": "s1", "container_id": "ta-1",
             "finished_utc": ts(300), "episodes_done": ["e1", "e2"]},
            {"kind": "caption", "shard_id": "k1", "container_id": "ta-2",
             "finished_utc": ts(600), "episodes_done": ["e1"]}],
        "ledgers": [{"calls": [{"call_id": "c1", "outcome": "spawned", "utc": ts(0)},
                               {"call_id": "c3", "outcome": "spawned", "utc": ts(310)}]}],
    }
    containers = {"ta-1": {"start": 1000.0 + 0, "end": 1000.0 + 650},
                  "ta-2": {"start": 1000.0 + 280, "end": 1000.0 + 630}}
    m = sc.measure(files, {"e1": 100.0, "e2": 50.0}, containers,
                   usd_per_s={"segment": 0.001, "caption": 0.002})
    seg, cap = m["pools"]["segment"], m["pools"]["caption"]
    assert seg["container_s"] == 650 and seg["busy_s"] == 250
    assert seg["utilization"] == pytest.approx(250 / 650, abs=1e-3)
    assert seg["cold_start_s"]["max"] == 100 and seg["spawn_to_start_s"]["max"] == 100
    assert seg["idle_s"] == 300 and seg["preempted_starts"] == 1
    assert seg["video_s"] == 150 and cap["video_s"] == 100
    assert cap["cold_start_s"]["max"] == 120 and cap["idle_s"] == 30
    assert m["pipelining"]["first_caption_after_first_segment_s"] == 100
    assert m["pipelining"]["overlap_s"] == 0  # captions began after the last segment end
    assert m["idle_share"] == pytest.approx((300 * 0.001 + 30 * 0.002)
                                            / (650 * 0.001 + 350 * 0.002), abs=1e-3)
    assert sc.first_spawn_after(files["ledgers"], 1000.0 + 100) == pytest.approx(1310.0)
    assert sc.first_spawn_after(files["ledgers"], 1000.0 + 400) is None


def _good_run(mode: str = "real") -> dict:
    """A synthetic ``report`` result that passes all 14 criteria."""
    n = sc.pool_sizes()
    pool = {k: {"peak_concurrent": n[k], "time_to_peak_s": 100, "utilization": 0.55,
                "utilization_working": 0.9, "preempted_starts": 1, "preempted_recovered": 1}
            for k in sc.POOLS}
    return {"state": "done", "records": {"missing": 0, "duplicates": 0}, "sizes": n,
            "measure": {"pools": pool, "idle_share": 0.3, "idle_share_excl_outage": 0.1,
                        "pipelining": {"first_caption_after_first_segment_s":
                                       330 if mode == "real" else 7,
                                       "overlap_s": 900}},
            "projection_error": {"actual_usd": 11.0, "upper_usd": 19.2, "nominal_usd": 11.5,
                                 "vs_nominal": -0.04, "billed": True},
            "chaos": {"preempt_segment": {}, "preempt_caption": {}, "kill_orchestrator": {}},
            "orchestrator": {"killed": True, "revive_s": 400, "revived_by": "reconciler"},
            "timings": {"status": {"p95": 3.0}, "report_s": 5.0, "compose_s": 4.0,
                        "compose_rc": 0}}


def _verdicts(r: dict, mode: str = "real") -> dict[int, str]:
    return {x["id"]: x["verdict"] for x in sc.evaluate(r, mode)}


def test_criteria_are_the_14_of_the_doc_and_all_automatic():
    rows = sc.evaluate(_good_run(), "real")
    assert [x["id"] for x in rows] == list(range(1, 15))
    assert all(x["auto"] for x in rows) and not sc.MANUAL
    assert {x["verdict"] for x in rows} == {"PASS"}, [x for x in rows if x["verdict"] != "PASS"]
    assert sc.criteria_summary(rows) == "14 PASS, 0 FAIL, 0 n.a. (14 of 14 automatic)"
    # the doc's table lists the same 14, in order
    doc = (Path(__file__).resolve().parents[2] / "docs" / "scale-test.md").read_text()
    table = doc.split("## Success criteria", 1)[1].split("\n## ", 1)[0]
    ids = [int(line.split("|")[1]) for line in table.splitlines()
           if line.startswith("| ") and line.split("|")[1].strip().isdigit()]
    assert ids == list(range(1, 15))


def test_criteria_targets_derive_from_the_rates():
    t = sc.criteria_targets("real")
    r = sc.kind_rates("real", warm=False)
    up = sc.simulate(sc.pool_sizes(), r, upper=True)
    assert t["time_to_peak_max_s"]["caption"] == round(
        up["pools"]["caption"].t_peak + sc.POLL_S["real"] + sc.LAUNCH_MARGIN_S)
    assert t["time_to_peak_max_s"]["caption"] > t["time_to_peak_max_s"]["segment"]
    assert t["pipelining_first_caption_max_s"] == round(
        r["caption"].startup_s * (1 + r["caption"].startup_unc) + sc.LAUNCH_MARGIN_S)


def _set(d: dict, path: str, value) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


FAILS = {  # criterion -> edits to the good run that make it FAIL
    1: [("state", "failed")],
    2: [("records.missing", 1)],
    3: [("measure.pools.caption.peak_concurrent", 2)],
    4: [("measure.pools.segment.time_to_peak_s", 10_000)],
    5: [("measure.pools.caption.utilization_working", 0.4)],
    6: [("measure.pipelining.overlap_s", 0)],
    7: [("projection_error.actual_usd", 25.0)],
    8: [("projection_error.vs_nominal", 0.8)],
    9: [("measure.idle_share_excl_outage", 0.2)],
    10: [("measure.pools.segment.preempted_recovered", 0)],
    11: [("orchestrator.revived_by", "flora resume (reconciler did not revive)")],
    12: [("timings.status.p95", 11.0)],
    13: [("timings.report_s", 31.0)],
    14: [("timings.compose_rc", 1)],
}


@pytest.mark.parametrize("cid", sorted(FAILS))
def test_each_criterion_fails_on_its_own(cid):
    assert sorted(FAILS) == list(range(1, 15))
    r = _good_run()
    for path, v in FAILS[cid]:
        _set(r, path, v)
    v = _verdicts(r)
    bad = {i for i, x in v.items() if x != "PASS"}
    # 3 -> 4: a pool below its size has no peak to time; 1 -> 11: work must continue to done
    expect = {1: {1, 11}, 3: {3, 4}}.get(cid, {cid})
    assert bad == expect and v[cid] == "FAIL", v


def test_criteria_other_fail_paths():
    r = _good_run()
    _set(r, "measure.pipelining.first_caption_after_first_segment_s", 10_000)
    assert _verdicts(r)[6] == "FAIL"
    r = _good_run()
    _set(r, "timings.compose_s", 61.0)
    assert _verdicts(r)[14] == "FAIL"
    r = _good_run()
    _set(r, "orchestrator.revive_s", sc.RECONCILE_WAIT_S + 1)
    assert _verdicts(r)[11] == "FAIL"
    r = _good_run()
    del r["chaos"]["preempt_caption"]  # the chaos action never fired: no proof
    assert _verdicts(r)[10] == "FAIL"
    r = _good_run()
    r["orchestrator"] = {}
    assert _verdicts(r)[11] == "FAIL"
    r = _good_run()
    _set(r, "measure.pools.segment.utilization", 0.1)  # whole-life utilization: informational
    assert _verdicts(r)[5] == "PASS"
    r = _good_run()
    _set(r, "measure.idle_share", 0.9)  # includes the induced outage: not judged
    assert _verdicts(r)[9] == "PASS"


def test_missing_data_is_a_fail_not_a_pass():
    rows = sc.evaluate({}, "real")
    assert {x["verdict"] for x in rows} == {"FAIL"}
    assert all(x["reason"] for x in rows)


def test_local_marks_container_and_money_criteria_na_with_a_reason():
    rows = {x["id"]: x for x in sc.evaluate(_good_run("local"), "local")}
    for i in (3, 4, 5, 7, 8, 9):
        assert rows[i]["verdict"] == "n.a." and rows[i]["reason"]
    for i in (1, 2, 6, 10, 11, 12, 13, 14):
        assert rows[i]["verdict"] == "PASS", rows[i]


def test_report_rechecks_spend_once_billing_lands():
    st = wh.State("r1", 0.0, "")
    run = st.step("run")
    run.projected_usd, run.projected_upper_usd = 11.5, 19.2
    run.attempts.append(wh.Attempt(started=0.0, tracked_usd=11.0, status="done"))
    res = _good_run()
    res["projection_error"].update(actual_usd=11.0, billed=False)
    res["criteria"] = sc.evaluate(res, "real")
    st.step("report").result = res
    rows = {x["id"]: x for x in sc.recheck(st, "real")}
    assert rows[7]["verdict"] == "PASS" and rows[7]["reason"].startswith("provisional")
    run.billed_usd = 30.0  # the bill landed: over the upper projection
    rows = {x["id"]: x for x in sc.recheck(st, "real")}
    assert rows[7]["verdict"] == "FAIL" and rows[8]["verdict"] == "FAIL"
    assert "(14 of 14 automatic)" in sc.render_report(st, "real", sc.projections("real"))


# --------------------------------------------------------------------------------------------
# Pool sizing and the warm-start variant


def test_projection_uses_the_measured_balanced_rows_and_per_class_prices():
    r = sc.kind_rates("real", warm=False)
    assert r["segment"].status == r["caption"].status == "measured"
    assert r["segment"].startup_status == r["caption"].startup_status == "measured"
    assert r["segment"].cs_per_video_s == pytest.approx(0.367)
    assert r["caption"].cs_per_video_s == pytest.approx(0.906)
    assert r["segment"].startup_s == pytest.approx(300.5)
    assert r["caption"].startup_s == pytest.approx(325.0)
    rates = sc.container_rates("real")
    run = sc.project("run", "real", warm=False)
    for i in run.items:
        kind = i["what"].split(":")[0]
        want = rates.get(kind, rates[sc.CPU_CLASS]) * 3600
        assert i["usd_per_h"] == pytest.approx(want, abs=1e-2), i


def test_warm_variant_uses_a_warm_row_or_the_documented_assumption():
    from dataclasses import replace

    from flora_deploy.planner.rates import RateTable

    t = RateTable.load()
    cold = sc.kind_rates("real", t, warm=False)
    assumed = sc.kind_rates("real", t, warm=True)
    for k in sc.POOLS:
        if sc.startup_row(t, k, "warm") is None:
            assert assumed[k].startup_status == "assumed"
            assert assumed[k].startup_s == pytest.approx(cold[k].startup_s
                                                         * sc.WARM_STARTUP_FACTOR)
    seg_cold = sc.startup_row(t, "segment", "cold")
    warm_row = replace(seg_cold, startup_s=90.0, status="placeholder", compile_cache="warm")
    t2 = replace(t, startup=(warm_row, *t.startup))
    w = sc.kind_rates("real", t2, warm=True)
    assert w["segment"].startup_s == 90.0 and w["segment"].startup_status == "placeholder"
    # a warm row never leaks into the cold projection
    assert sc.kind_rates("real", t2, warm=False)["segment"].startup_s == seg_cold.startup_s
    assert sc.project("run", "real", warm=True).usd < sc.project("run", "real", warm=False).usd


def test_caps_stay_cold_under_assume_warm(offline, tmp_path, capsys):
    cold = sc.caps("real")
    assert sc.main(["--state", str(tmp_path / "s.json"), "--assume-warm"]) == 0
    out = capsys.readouterr().out
    assert sc.ASSUME_WARM is False  # restored
    assert "warm start (" in out and "cold start (" in out
    assert sc.caps("real") == cold


def test_simulated_schedule_pipelines_and_respects_pool_sizes():
    r = sc.kind_rates("real", warm=False)
    sim = sc.simulate({"segment": 2, "caption": 3}, r)
    seg, cap = sim["pools"]["segment"], sim["pools"]["caption"]
    assert seg.peak == 2 and cap.peak == 3
    assert seg.busy_s == pytest.approx(sc.VIDEO_S * 0.367, rel=1e-6)
    assert cap.busy_s == pytest.approx(sc.VIDEO_S * 0.906, rel=1e-6)
    # the pre-warmed caption container is ready at the first segment receipt, and they overlap
    assert sim["pipelining"]["first_caption_after_first_segment_s"] == pytest.approx(0.0)
    assert sim["pipelining"]["overlap_s"] > 0
    assert sim["wall_s"] > seg.last_end
    one = sc.simulate({"segment": 1, "caption": 1}, r)
    assert one["wall_s"] > sim["wall_s"]
    # without the pre-warm, captions start one caption cold start after that receipt
    cold = sc.simulate({"segment": 2, "caption": 3}, r, prewarm=False)
    assert cold["pipelining"]["first_caption_after_first_segment_s"] == pytest.approx(325.0)
    assert cold["wall_s"] >= sim["wall_s"]


def test_simulated_tails_are_the_drained_tail():
    """Once its pool's last shard ends, a container idles at most the planner's drained tail
    (one poll + 2 s), not the deployed 300 s window (which still applies to a container that
    goes idle while its pool has work)."""
    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S
    from flora_deploy.planner.options import POOL_END_TAIL_S

    r = sc.kind_rates("real", warm=False)
    sim = sc.simulate({"segment": 2, "caption": 3}, r)
    for kind, p in sim["pools"].items():
        assert 0 < p.tail_s < p.starts * WORKER_SCALEDOWN_S, kind
        assert p.tail_s <= (p.starts - 1) * WORKER_SCALEDOWN_S + POOL_END_TAIL_S + 1e-6, kind
    seg = sim["pools"]["segment"]
    assert seg.tail_s == pytest.approx(seg.starts * POOL_END_TAIL_S)  # its shards end together


def test_prewarm_is_charged_from_its_spawn():
    r = sc.kind_rates("real", warm=False)
    warm = sc.simulate({"segment": 1, "caption": 1}, r)["pools"]["caption"]
    cold = sc.simulate({"segment": 1, "caption": 1}, r, prewarm=False)["pools"]["caption"]
    assert warm.startup_s == pytest.approx(cold.startup_s)  # the same one cold start
    assert warm.busy_s == pytest.approx(cold.busy_s)


def test_pool_sizing_recommends_two_segment_three_caption():
    rows = sc.pool_sizing()
    assert len(rows) == 16
    rec = sc.recommend(rows)
    assert (rec["segment"], rec["caption"]) == (2, 3)
    assert rec["wall_upper_s"] <= sc.SIZING_TARGET_WALL_S
    assert sc.pool_sizes() == {"segment": 2, "caption": 3}  # the configured pools
    by = {(x["segment"], x["caption"]): x for x in rows}
    assert by[(3, 2)]["usd"] > by[(2, 2)]["usd"]  # the old 3 + 2 is dominated by 2 + 2
    assert by[(3, 2)]["wall_s"] == by[(2, 2)]["wall_s"]


# --------------------------------------------------------------------------------------------
# Review 11a


def test_observed_spend_prices_each_container_at_its_own_class():
    """M2: the orchestrator and reconcilers at their CPU rate (preemptible since ``jobs.pools``:
    no 3x), L40S workers at L40S, H100 at H100 (``jobs.budget`` pinned prices); not everything
    at H100."""
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.modal_app import orchestrator_app as oa

    r = pinned_rates()
    rates = sc.container_rates("real")
    cpu = r.per_second(None, 0, oa.CPU_CORES, oa.MEMORY_MIB / 1024, preemptible=True)
    assert rates[sc.CPU_CLASS] == pytest.approx(cpu)
    assert cpu == pytest.approx(
        r.per_second(None, 0, oa.CPU_CORES, oa.MEMORY_MIB / 1024, preemptible=False) / 3)
    from flora_deploy.config.presets import get_preset

    seg = get_preset("balanced").pools["segment"]
    assert rates["segment"] == pytest.approx(
        r.per_second("L40S", 1, seg.cpu_cores, seg.memory_gib))
    mon = sc.Monitor(None, "app", rates)  # type: ignore[arg-type]
    mon.containers = {"ta-orch": {"start": 0.0, "end": 3600.0},
                      "ta-seg": {"start": 0.0, "end": 3600.0},
                      "ta-cap": {"start": 0.0, "end": 1800.0}}
    mon.kinds = {"ta-seg": "segment", "ta-cap": "caption"}
    want = 3600 * cpu + 3600 * rates["segment"] + 1800 * rates["caption"]
    assert mon.observed_usd() == pytest.approx(want)
    assert mon.observed_usd() < 3600 * 2.5 * rates["caption"]  # the old all-H100 figure


class _Harness:
    """``run_run`` in real mode with every Modal / flora call faked."""

    def __init__(self, monkeypatch, tmp_path, *, states=("done",), containers=None,
                 compose_timeout=False):
        self.calls: list[list[str]] = []
        st = wh.State("r1", 0.0, "2026-10-22")
        st.step("deploy").status = "done"
        st.step("deploy").result = {"ok": True, "app_state": "deployed"}
        self.ctx = sc.Ctx("real", st, tmp_path, tmp_path / "state.json", 0.0)
        states = list(states)

        def sh(args, *, timeout=1800, check=True):
            self.calls.append(list(args))
            if compose_timeout and "compose" in args:
                raise sc.subprocess.TimeoutExpired(args, timeout)
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        self.ctx.sh = sh  # type: ignore[method-assign]
        monkeypatch.setattr(sc, "_app_name", lambda ctx: "flora-app")
        monkeypatch.setattr(sc, "_flora", lambda: "flora")
        monkeypatch.setattr(sc, "_container_kinds", lambda ctx: {})
        monkeypatch.setattr(sc.time, "sleep", lambda s: None)

        def poll(mon):
            mon.containers.update(containers or {})
            return len(mon.containers)

        def status(mon):
            s = {"t": 0.0, "state": states.pop(0) if len(states) > 1 else states[0],
                 "counts": {}, "spend_usd": 0.0}
            mon.samples.append(s)
            return s

        monkeypatch.setattr(sc.Monitor, "poll_containers", poll)
        monkeypatch.setattr(sc.Monitor, "status", status)
        monkeypatch.setattr(sc.subprocess, "run", lambda args, **kw: type(
            "P", (), {"returncode": 0, "stdout": "{}", "stderr": ""})())

    def stopped(self) -> bool:
        return ["modal", "app", "stop", "--yes", "flora-app"] in self.calls


def test_cpu_containers_do_not_trip_the_cancel_rule(monkeypatch, tmp_path):
    """M2: an hour of orchestrator + reconciler containers is cents, not an H100-hour each."""
    cpu = {f"ta-cpu{i}": {"start": 0.0, "end": 3600.0} for i in range(4)}
    h = _Harness(monkeypatch, tmp_path, states=("running", "done"), containers=cpu)
    out = sc.run_run(h.ctx)
    assert "stopped" not in out, out.get("stopped")
    assert not any("cancel" in c for c in h.calls)
    assert out["observed_usd"] == pytest.approx(4 * 3600 * sc.container_rates("real")["cpu"],
                                                abs=1e-3)


def test_a_compose_timeout_still_stops_the_app(monkeypatch, tmp_path):
    """M4: the app (and its reconciler cron) is stopped whatever compose does."""
    h = _Harness(monkeypatch, tmp_path, compose_timeout=True)
    with pytest.raises(sc.subprocess.TimeoutExpired):
        sc.run_run(h.ctx)
    assert h.stopped()
    assert h.ctx.st.step("deploy").result["app_state"] == "stopped"  # M3: recorded


def test_a_retry_after_a_failed_run_redeploys_the_stopped_app(offline, tmp_path):
    """M3: the failed run's ``finally`` ran ``modal app stop``; the retry deploys again."""
    state = tmp_path / "state.json"
    st = wh.load_state(state)
    for s in ("verify", "index", "plan", "deploy"):
        st.step(s).status = "done"
    st.step("deploy").result = {"ok": True, "app_state": "stopped"}
    st.step("run").status = "failed"
    wh.save_state(st, state)
    assert sc.main(["--yes-spend", "--state", str(state), "--work", str(tmp_path / "w")]) == 0
    assert offline == ["deploy", "run", "report"]
    st = wh.load_state(state)
    assert st.step("deploy").result["app_state"] == "deployed"
    assert len(st.step("deploy").attempts) == 1


def test_a_run_step_that_raises_marks_the_app_stopped_and_only_run_refuses(offline, tmp_path,
                                                                           monkeypatch):
    def boom(ctx):
        raise RuntimeError("client died")

    monkeypatch.setitem(sc.RUNNERS, "run", boom)
    state = tmp_path / "state.json"
    work = str(tmp_path / "w")
    assert sc.main(["--yes-spend", "--state", str(state), "--work", work]) == 1
    st = wh.load_state(state)
    assert st.step("deploy").result["app_state"] == "stopped"
    st.billed_at = st.step("run").attempts[0].last_activity + wh.BILLING_LAG_S + 1
    wh.save_state(st, state)
    offline.clear()
    monkeypatch.setitem(sc.RUNNERS, "run", lambda ctx: {"ok": True, "tracked_usd": 0.0,
                                                        "apps": []})
    assert sc.main(["--yes-spend", "--only", "run", "--state", str(state), "--work", work]) == 1
    assert offline == []
    assert "not deployed" in wh.load_state(state).step("run").notes[-1]


def test_only_index_checks_the_dataset_pin_first(offline, tmp_path):
    """L1: ``--only`` never spends past an unverified pin."""
    state = tmp_path / "state.json"
    assert sc.main(["--yes-spend", "--only", "index", "--state", str(state),
                    "--work", str(tmp_path / "w")]) == 0
    assert offline == ["verify", "index"]

    def bad(ctx):
        offline.append("verify")
        return {"ok": False, "problems": ["blob differs"], "tracked_usd": 0.0, "apps": []}

    offline.clear()
    state2 = tmp_path / "state2.json"
    sc.RUNNERS["verify"] = bad
    assert sc.main(["--yes-spend", "--only", "index", "--state", str(state2),
                    "--work", str(tmp_path / "w")]) == 1
    assert offline == ["verify"]  # index never started


def _ts(t: float) -> str:
    return wh.time.strftime("%Y-%m-%dT%H:%M:%SZ", wh.time.gmtime(1_800_000_000.0 + t))


def test_time_to_peak_is_measured_from_the_run_start(monkeypatch):
    """L2: the first container's cold start counts; a peak below max_containers is no peak."""
    monkeypatch.setitem(sc.POOLS["segment"], "max_containers", 3)
    files: dict = {"attempts": [], "receipts": [], "ledgers": []}
    cont = {}
    for i, (cid, start) in enumerate([("a", 100), ("b", 400), ("c", 700)]):
        files["attempts"].append({"container_id": cid, "kind": "segment", "shard_id": f"s{i}",
                                  "started_utc": _ts(start + 60), "ended_utc": _ts(start + 900)})
        files["receipts"].append({"container_id": cid, "kind": "segment", "shard_id": f"s{i}",
                                  "finished_utc": _ts(start + 900), "episodes_done": []})
        cont[cid] = {"start": 1_800_000_000.0 + start, "end": 1_800_000_000.0 + start + 960}
    m = sc.measure(files, {}, cont, run_start=1_800_000_000.0)
    assert m["pools"]["segment"]["time_to_peak_s"] == 700
    assert sc.measure(files, {}, cont)["pools"]["segment"]["time_to_peak_s"] is None
    two = {k: v for k, v in cont.items() if k != "c"}
    files2 = {k: [x for x in v if x.get("container_id") != "c"] for k, v in files.items()}
    m2 = sc.measure(files2, {}, two, run_start=1_800_000_000.0)
    assert m2["pools"]["segment"]["time_to_peak_s"] is None  # never reached 3 containers
    r = {"measure": m2, "state": "done"}
    v = sc.evaluate(r, "real")[3]
    assert v["id"] == 4 and v["verdict"] == "FAIL" and v["value"]["segment"] is None


def test_preemption_counts_only_when_the_shard_finished_later():
    """L3: a preempted start whose shard never got a receipt is not a recovery."""
    files = {"attempts": [
        {"container_id": "a", "kind": "segment", "shard_id": "s1", "started_utc": _ts(10),
         "ended_utc": _ts(50), "preempted": True}],
        "receipts": [], "ledgers": []}
    m = sc.measure(files, {})
    assert m["pools"]["segment"]["preempted_starts"] == 1
    assert m["pools"]["segment"]["preempted_recovered"] == 0
    r = {"measure": m, "chaos": {"preempt_segment": {}}, "state": "failed"}
    v = sc.evaluate(r, "real")[9]
    assert v["id"] == 10 and v["verdict"] == "FAIL" and v["value"]["segment"] == "0/1"
    files["attempts"].append({"container_id": "b", "kind": "segment", "shard_id": "s1",
                              "started_utc": _ts(60)})
    files["receipts"].append({"container_id": "b", "kind": "segment", "shard_id": "s1",
                              "finished_utc": _ts(90), "episodes_done": []})
    m = sc.measure(files, {})
    assert m["pools"]["segment"]["preempted_recovered"] == 1
    v = sc.evaluate({"measure": m, "chaos": {"preempt_segment": {}}}, "real")[9]
    assert v["value"]["segment"] == "1/1"  # caption recorded nothing: still a FAIL
    assert v["verdict"] == "FAIL"


def test_measure_working_utilization_and_outage_idle():
    """Criterion 5 counts each container from its first shard start to its last shard end;
    criterion 9 leaves out tails that began while the killed orchestrator was down."""
    files: dict = {"attempts": [], "receipts": [], "ledgers": []}
    for cid, shards in (("a", [(100, 200), (300, 400)]), ("b", [(100, 500)])):
        for i, (t0, t1) in enumerate(shards):
            files["attempts"].append({"container_id": cid, "kind": "caption",
                                      "shard_id": f"{cid}{i}", "started_utc": _ts(t0)})
            files["receipts"].append({"container_id": cid, "kind": "caption",
                                      "shard_id": f"{cid}{i}", "finished_utc": _ts(t1),
                                      "episodes_done": []})
    base = 1_800_000_000.0
    cont = {"a": {"start": base, "end": base + 700}, "b": {"start": base, "end": base + 600}}
    m = sc.measure(files, {}, cont, usd_per_s={"caption": 1.0},
                   outage=(base + 350, base + 450))
    cap = m["pools"]["caption"]
    assert cap["utilization_working"] == pytest.approx(600 / 700, abs=1e-3)
    assert cap["utilization"] == pytest.approx(600 / 1300, abs=1e-3)
    assert m["idle_share"] == pytest.approx(400 / 1300, abs=1e-3)
    # a's tail began at 400, inside the outage; b's at 500, after it
    assert m["idle_share_excl_outage"] == pytest.approx(100 / 1300, abs=1e-3)
