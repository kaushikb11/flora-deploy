"""``scripts/scale_test.py --report`` / ``--finalize`` on a run step that failed (budget stop)
while its job was finished outside the script (``flora deploy`` + ``flora resume``): reported
from a fake resumed job's own records. Nothing here contacts Modal."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
RATES = Path(__file__).resolve().parents[2] / "src" / "flora_deploy" / "planner" / "rates.json"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load("scale_test")
wh = sys.modules["wave_h"]
rb = sys.modules["_runbook"]

T = 1_792_696_000.0  # the run start (on an hour boundary)
APP = "flora-scale-lite-res01-abc"
RESUME = T + 10_000.0
SCALEDOWN = 60.0


@pytest.fixture(autouse=True)
def lite_profile():
    prev = sc.use_profile("lite")
    yield
    sc.use_profile(prev)


def _utc(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# shard -> (kind, container, spawn, start, end, episodes): generation 1 ran three segment
# shards (A, B) and its caption pre-warm, then was budget-stopped; generation 2 (flora resume)
# ran the other three on two new segment containers (C, D) and every caption (E, F)
SHARDS = {
    "seg-1": ("segment", "ta-A", T + 35, T + 600, T + 660, ["e0"]),
    "seg-2": ("segment", "ta-B", T + 36, T + 602, T + 650, ["e1"]),
    "seg-3": ("segment", "ta-B", T + 651, T + 655, T + 700, ["e2"]),
    "seg-4": ("segment", "ta-C", RESUME + 10, RESUME + 150, RESUME + 190, ["e3"]),
    "seg-5": ("segment", "ta-D", RESUME + 11, RESUME + 160, RESUME + 200, ["e4"]),
    "seg-6": ("segment", "ta-C", RESUME + 191, RESUME + 195, RESUME + 230, ["e5"]),
    "cap-1": ("caption", "ta-E", RESUME + 60, RESUME + 400, RESUME + 500, ["e0", "e1"]),
    "cap-2": ("caption", "ta-F", RESUME + 61, RESUME + 410, RESUME + 520, ["e2"]),
    "cap-3": ("caption", "ta-E", RESUME + 501, RESUME + 505, RESUME + 600, ["e3", "e4"]),
    "cap-4": ("caption", "ta-F", RESUME + 521, RESUME + 525, RESUME + 610, ["e5"]),
}


def _report_rows() -> list[dict]:
    """``flora report``'s rows as ``jobs.containers.container_rows`` derives them."""
    by: dict[str, list] = {}
    for kind, cid, spawn, t0, t1, _eps in SHARDS.values():
        by.setdefault(cid, []).append((kind, spawn, t0, t1))
    rows = []
    for cid, xs in sorted(by.items()):
        xs.sort(key=lambda x: x[2])
        kind, spawn, first = xs[0][0], xs[0][1], xs[0][2]
        end = max(x[3] for x in xs) + SCALEDOWN
        gpu = "L40S" if kind == "segment" else "H100"
        rows.append({"container_id": cid, "kind": kind, "gpu": gpu, "gpu_count": 1,
                     "gpu_seen": f"NVIDIA {gpu}", "started_utc": _utc(first),
                     "startup_s": round(first - spawn, 1),
                     "busy_s": round(sum(x[3] - x[2] for x in xs), 1),
                     "seconds": round(end - spawn, 1), "shards": len(xs), "stopped": True,
                     "source": "attempt_markers", "usd": 0.1})
    return rows


def _job(*, state: str = "composing", done: bool = True) -> dict:
    files = {"attempts": [], "receipts": [], "ledgers": []}
    for sid, (kind, cid, spawn, t0, t1, eps) in SHARDS.items():
        call = f"fc-{sid}"
        files["attempts"].append({"kind": kind, "shard_id": sid, "container_id": cid,
                                  "call_id": call, "started_utc": _utc(t0), "ended_utc": None,
                                  "preempted": False})
        if done:
            files["receipts"].append({"kind": kind, "shard_id": sid, "container_id": cid,
                                      "finished_utc": _utc(t1), "episodes_done": eps,
                                      "episodes_failed": [], "gpu": "x"})
        files["ledgers"].append({"kind": kind, "shard_id": sid, "status": "done",
                                 "calls": [{"outcome": "spawned", "utc": _utc(spawn),
                                            "call_id": call},
                                           {"outcome": "success", "utc": _utc(t1 + 2),
                                            "call_id": call}]})
    audits = [{"shard_audit": True, "shard_id": sid, "stage": kind,
               "batcher": {"rounds": 4, "mean_batch": 6.0, "max_batch": 10,
                           "generate_s": 40.0, "wall_s": 50.0},
               "compile_cache": {"state": "warm" if kind == "segment" else "cold"}}
              for sid, (kind, *_x) in SHARDS.items()]
    n = {"segment": 6, "caption": 4}
    counts = {k: {"done": v if done else 0, "pending": 0 if done else v, "running": 0}
              for k, v in n.items()}
    return {
        "state": state, "counts": counts, "app": APP,
        "launches": [{"utc": _utc(T + 30), "reason": "submit", "call_id": "fc-o1"},
                     {"utc": _utc(RESUME), "reason": "resume", "call_id": "fc-o2"}],
        "files": files, "durations": {f"e{i}": 100.0 for i in range(6)},
        "shards": [{"kind": k, "shard_id": sid, "episodes": eps}
                   for sid, (k, _c, _s, _a, _b, eps) in SHARDS.items()],
        "report": {"state": state, "usd_per_video_hour": 20.0,
                   "spend": {"total_usd": 3.4, "by_kind": {"caption": {"usd": 2.0}},
                             "containers": _report_rows()}},
        "summary": None,
        "drains": {"segment": [(T + 702, RESUME), (RESUME + 232, None)],
                   "caption": [(T + 702, RESUME), (RESUME + 612, None)]},
        "prewarms": {"caption": [T + 40, T + 701]},
        "plans": {"caption": {"decision": "prewarm", "start_at": T + 40, "t": T + 40}},
        "audits": audits,
        "timings": {"status_final_s": 3.1, "status_rc": 0, "report_s": 6.2, "report_rc": 3}}


def _state() -> wh.State:
    st = wh.State("res01", T - 100, "2026-10-21")
    st.step("plan").status = "done"
    st.step("plan").result = {"warm_kinds": ["segment"], "options": []}
    run = st.step("run")
    run.status, run.projected_usd, run.projected_upper_usd = "failed", 2.8, 6.4
    run.attempts.append(wh.Attempt(started=T, finished=T + 4000, status="failed",
                                   apps=[APP], budget_usd=6.4, tracked_usd=1.4))
    run.result = {"job_id": "scale-lite-res01", "started_at": T, "status_s": [20.0, 2300.0],
                  "state": "budget_stopped", "chaos": {}, "orchestrator": {}, "app": APP}
    return st


def _billing(*, settled: bool = True) -> rb.Billing:
    rows = [("ap-1", APP, rb.hour_floor(T), 1.5), ("ap-2", APP, rb.hour_floor(RESUME), 2.1)]
    through = rb.hour_floor(RESUME + 1000) + 2 * rb.HOUR_S  # and the hour after it
    if not settled:
        rows = rows[:1]
        through = rb.hour_floor(T) + rb.HOUR_S
    return rb.Billing({APP: sum(r[3] for r in rows)}, rows, through)


def _rows(res: dict) -> dict[int, dict]:
    return {x["id"]: x for x in res["criteria"]}


def test_a_resumed_job_is_measured_per_generation_and_caveated():
    res = sc.resumed_report(_job(), _state(), "real", _billing())
    rows = _rows(res)
    m, rs = res["measure"], res["resumed"]
    assert [g["reason"] for g in rs["generations"]] == ["submit", "resume"]
    assert rs["interruption"]["gap_s"] == pytest.approx(RESUME - (T + 702), abs=1)
    # segment reached its size in generation 1 (timed from the run start), caption only in 2
    # (unsampled containers count from their first call's spawn)
    assert m["pools"]["segment"]["time_to_peak_generation"] == 1
    assert m["pools"]["segment"]["time_to_peak_s"] == pytest.approx(36, abs=1)
    assert m["pools"]["caption"]["time_to_peak_generation"] == 2
    assert m["pools"]["caption"]["time_to_peak_s"] == pytest.approx(61, abs=1)
    assert rs["late_peak_pools"] == ["caption"]
    # pipelining from the generation where both ran, not across the 3-hour gap
    assert m["pipelining"]["generation"] == 2
    assert m["pipelining"]["first_caption_after_first_segment_s"] == pytest.approx(210, abs=1)
    # utilization / throughput per pool from every generation's containers
    assert m["pools"]["segment"]["containers"] == 4 and m["pools"]["caption"]["containers"] == 2
    assert rows[15]["verdict"] == "PASS"
    assert rows[15]["value"]["segment"]["containers"] == 4
    # records from the receipts: no compose summary
    assert res["records"]["missing"] == 0 and res["records"]["duplicates"] == 0
    assert rows[2]["verdict"] == "PASS" and "receipts" in rows[2]["reason"]
    # spend: settled billing over both generations, vs the projection
    assert res["projection_error"]["actual_usd"] == pytest.approx(3.6)
    assert res["projection_error"]["billed"] is True
    assert rows[7]["verdict"] == "PASS" and "caveat (resumed)" in rows[7]["reason"]
    # what the interruption decides is n.a., with the reason; nothing fails on it
    for i in (1, 8, 12, 14, 16):
        assert rows[i]["verdict"] == "n.a.", (i, rows[i])
        assert rows[i]["reason"].startswith(sc.RESUMED_NA)
    assert "2300" in str(rows[12]["value"])  # the long poll stays visible
    assert rows[16]["value"]["billed_after_end_hour_usd"] == 0.0
    assert rows[4]["verdict"] in ("PASS", "n.a.") and "caption" in rows[4]["reason"]
    assert rows[17]["verdict"] == "n.a."  # pre-warmed in gen 1, captions ran in gen 2
    assert "FAIL" not in {x["verdict"] for x in rows.values() if x["id"] in sc.RESUMED_CAVEATS}
    assert rows[10]["verdict"] == rows[11]["verdict"] == "n.a."  # lite: no chaos


def test_the_run_steps_own_samples_and_a_first_generation_miss():
    st = _state()
    # Modal saw segment B start late: the first generation's own time to peak, which the
    # interruption does not excuse
    st.step("run").result["containers"] = {"ta-B": {"start": T + 500, "end": T + 500}}
    res = sc.resumed_report(_job(), st, "real", _billing())
    assert res["measure"]["pools"]["segment"]["time_to_peak_s"] == pytest.approx(500, abs=1)
    row = _rows(res)[4]
    assert row["verdict"] == "FAIL" and "caveat (resumed): caption" in row["reason"]


def test_the_cache_state_is_read_from_its_identity():
    audits = [{"shard_audit": True, "shard_id": "s1", "stage": "segment",
               "compile_cache": "warm:tempseg-L40S-3a99:91a28f88e4381b69"},
              {"shard_audit": True, "shard_id": "c1", "stage": "caption",
               "compile_cache": "cold:forced"}]
    c = sc.cache_by_kind(audits, [], warm_kinds=("segment",))
    assert c["segment"]["ok"] and c["segment"]["states"] == {"warm": 1}
    assert c["caption"]["states"] == {"cold": 1}


def test_spend_is_provisional_until_billing_covers_the_job_end():
    res = sc.resumed_report(_job(), _state(), "real", _billing(settled=False))
    pe = res["projection_error"]
    assert pe["billed"] is False and pe["billed_usd"] == pytest.approx(1.5)
    assert pe["actual_usd"] == pytest.approx(3.4)  # flora report's call-priced total
    assert "provisional" in _rows(res)[7]["reason"]
    res = sc.resumed_report(_job(), _state(), "real", None)  # billing unreadable
    assert res["projection_error"]["actual_usd"] == pytest.approx(3.4)


def test_a_lost_record_still_fails():
    job = _job()
    job["files"]["receipts"] = [r for r in job["files"]["receipts"] if r["shard_id"] != "cap-4"]
    rows = _rows(sc.resumed_report(job, _state(), "real", _billing()))
    assert rows[2]["verdict"] == "FAIL" and rows[2]["value"] == 1


def test_resumable_only_for_a_failed_run_step():
    st = _state()
    assert sc.resumable(st, "real") and not sc.resumable(st, "local")
    st.step("run").status = "done"
    assert not sc.resumable(st, "real")
    st = _state()
    st.step("report").result = {"criteria": [{"id": 1}]}  # measured by the script itself
    assert not sc.resumable(st, "real")
    st.step("report").result["resumed"] = {"x": 1}  # built here: re-read (billing settles)
    assert sc.resumable(st, "real")


def _no_subprocess(*a, **k):
    raise AssertionError(f"--report on a resumed job ran a command: {a}")


def test_report_rebuilds_the_report_step_from_the_job(tmp_path, monkeypatch, capsys):
    spath = tmp_path / "state.json"
    wh.save_state(_state(), spath)
    reads = []
    monkeypatch.setattr(sc, "read_resumed_job", lambda ctx: reads.append(ctx.job_id) or _job())
    monkeypatch.setattr(wh, "billed_by_app", lambda since: _billing())
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", _no_subprocess)
    assert sc.main(["--mode", "real", "--report", "--state", str(spath),
                    "--work", str(tmp_path / "work")]) == 0
    out = capsys.readouterr().out
    assert reads == ["scale-lite-res01"]
    st = wh.load_state(spath)
    assert st.step("run").status == "failed"  # the script's own record stays as it was
    rep = st.step("report")
    assert rep.status == "done" and rep.result["resumed"] and sc.RESUMED_STEP_NOTE in rep.notes
    md = (tmp_path / "report.md").read_text()
    assert sc.RESUMED_NA in md and "## Success criteria" in md and "## resumed" in md
    assert "criteria:" in out and " 0 FAIL" in out
    # again, once billing has moved: re-read, not kept
    monkeypatch.setattr(wh, "billed_by_app", lambda since: _billing(settled=False))
    assert sc.main(["--mode", "real", "--report", "--state", str(spath),
                    "--work", str(tmp_path / "work")]) == 0
    assert wh.load_state(spath).step("report").result["projection_error"]["billed"] is False


def test_report_leaves_an_unfinished_job_alone(tmp_path, monkeypatch, capsys):
    spath = tmp_path / "state.json"
    wh.save_state(_state(), spath)
    monkeypatch.setattr(sc, "read_resumed_job", lambda ctx: _job(state="budget_stopped",
                                                                  done=False))
    monkeypatch.setattr(wh, "billed_by_app", lambda since: _billing())
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    assert sc.main(["--mode", "real", "--report", "--state", str(spath)]) == 0
    assert "nothing to report from it yet" in capsys.readouterr().out
    assert wh.load_state(spath).step("report").status == "pending"


def test_finalize_folds_a_resumed_job_with_notes(tmp_path, monkeypatch):
    spath = tmp_path / "state.json"
    wh.save_state(_state(), spath)
    monkeypatch.setattr(sc, "read_resumed_job", lambda ctx: _job())
    monkeypatch.setattr(wh, "billed_by_app", lambda since: _billing())
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    rates = tmp_path / "rates.json"
    rates.write_text(RATES.read_text())
    out: list[str] = []
    args = argparse.Namespace(mode="real", state=spath, rates=rates, write=False, work=None)
    assert sc.finalize_main(args, out=out.append) == 0
    text = "\n".join(out)
    assert "REFUSED" not in text, text
    assert sc.RESUMED_NA in text and "not composed on its store" in text
    assert "segment rate" in text and "caption rate" in text
    assert json.loads(rates.read_text()) == json.loads(RATES.read_text())  # dry run
    samples, _notes, refusals = sc.finalize_samples(wh.load_state(spath))
    assert not refusals and len(samples[("rate", "segment")]) == 4
