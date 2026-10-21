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
    assert "segment: 3 cold starts" in whats and "caption: 2 cold starts" in whats
    assert "preemption" in whats and "reconciler gap" in whats
    # upper adds the client slack per container on top of the scale-down tail
    tail = next(i for i in run.items if i["what"].startswith("caption: idle tail"))
    assert tail["upper_seconds"] >= 2 * (300 + sc.CLIENT_SLACK_S)
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
        assert pools["segment"].max_containers == 3 and pools["caption"].max_containers == 2
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
    assert st.step("run").status == "stopped"
    assert "global budget left" in st.step("run").notes[-1]
    assert "run" not in offline and "report" not in offline


def test_spend_after_a_step_over_its_cap_stops_the_runbook(offline, tmp_path, monkeypatch):
    def overspend(ctx):
        return {"ok": True, "tracked_usd": sc.caps("real")["deploy"] + 1.0, "apps": []}

    monkeypatch.setitem(sc.RUNNERS, "deploy", overspend)
    state = tmp_path / "state.json"
    assert sc.main(["--yes-spend", "--state", str(state), "--work", str(tmp_path / "w")]) == 3
    st = wh.load_state(state)
    assert "stop rule after" in st.step("deploy").notes[-1]
    assert offline == ["verify", "index", "plan"]


def test_in_step_budget_is_cap_minus_earlier_attempts():
    st = wh.State("r", 0.0, "2026-10-22")
    s = st.step("run")
    s.attempts.append(wh.Attempt(started=1.0, status="failed", tracked_usd=4.0))
    left = wh.step_budget(st, "run", caps=sc.caps("real"), global_cap=sc.global_cap("real"))
    assert left == pytest.approx(sc.caps("real")["run"] - 4.0)


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
    st.step("report").result = {"criteria": [{"criterion": "end state", "value": "done",
                                              "target": "done", "verdict": "PASS"}],
                                "measure": {"pools": {}}}
    wh.save_state(st, state)
    assert sc.main(["--report", "--state", str(state)]) == 0
    text = (tmp_path / "report.md").read_text()
    assert "| end state | done | done | PASS |" in text and offline == []


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


def test_criteria_verdicts():
    r = {"state": "done", "records": {"missing": 0, "duplicates": 0},
         "measure": {"pools": {"segment": {"peak_concurrent": 3, "utilization": 0.7,
                                           "time_to_peak_s": 100, "preempted_starts": 1},
                               "caption": {"peak_concurrent": 1, "utilization": 0.5,
                                           "time_to_peak_s": 100, "preempted_starts": 0}},
                     "pipelining": {"first_caption_after_first_segment_s": 60,
                                    "overlap_s": 500},
                     "idle_share": 0.1},
         "projection_error": {"actual_usd": 9.0, "upper_usd": 18.0, "vs_nominal": 0.1},
         "chaos": {"preempt_segment": {}, "preempt_caption": {}},
         "orchestrator": {"killed": True, "revive_s": 400, "revived_by": "reconciler"},
         "timings": {"status": {"p95": 3.0}, "report_s": 5.0, "compose_s": 4.0}}
    v = {x["criterion"]: x["verdict"] for x in sc.evaluate(r, "real")}
    assert v["end state"] == "PASS" and v["lost/duplicate records"] == "PASS"
    assert v["segment: autoscaled to max_containers"] == "PASS"
    assert v["caption: autoscaled to max_containers"] == "FAIL"
    assert v["caption: utilization (busy / container-s)"] == "FAIL"
    assert v["caption: preemption recovered"] == "FAIL"
    assert v["orchestrator killed -> revived by the reconciler"] == "PASS"
    assert v["spend within the upper projection"] == "PASS"
    local = {x["criterion"]: x["verdict"] for x in sc.evaluate(r, "local")}
    assert local["segment: autoscaled to max_containers"] == "n.a."


# --------------------------------------------------------------------------------------------
# Review 11a


def test_observed_spend_prices_each_container_at_its_own_class():
    """M2: the orchestrator and reconcilers at the non-preemptible CPU rate, L40S workers at
    L40S, H100 at H100 (``jobs.budget`` pinned prices); not everything at H100."""
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.modal_app import orchestrator_app as oa

    r = pinned_rates()
    rates = sc.container_rates("real")
    cpu = r.per_second(None, 0, oa.CPU_CORES, oa.MEMORY_MIB / 1024, preemptible=False)
    assert rates[sc.CPU_CLASS] == pytest.approx(cpu)
    assert cpu == pytest.approx(3 * r.per_second(None, 0, oa.CPU_CORES, oa.MEMORY_MIB / 1024))
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


def test_time_to_peak_is_measured_from_the_run_start():
    """L2: the first container's cold start counts; a peak below max_containers is no peak."""
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
    v = {x["criterion"]: x["verdict"] for x in sc.evaluate(r, "real")}
    assert v["segment: time to peak"] == "FAIL"


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
    v = {x["criterion"]: x["verdict"] for x in sc.evaluate(r, "real")}
    assert v["segment: preemption recovered"] == "FAIL"
    files["attempts"].append({"container_id": "b", "kind": "segment", "shard_id": "s1",
                              "started_utc": _ts(60)})
    files["receipts"].append({"container_id": "b", "kind": "segment", "shard_id": "s1",
                              "finished_utc": _ts(90), "episodes_done": []})
    m = sc.measure(files, {})
    assert m["pools"]["segment"]["preempted_recovered"] == 1
    v = {x["criterion"]: x["verdict"] for x in sc.evaluate({"measure": m, "chaos": {
        "preempt_segment": {}}}, "real")}
    assert v["segment: preemption recovered"] == "PASS"
