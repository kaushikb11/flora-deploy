"""``scripts/rubric_smoke.py``: dry-run cost of the TP=2 rubric jobs, the stop rule, the $0
preconditions, the measurements, the ``flora certify`` refusal and ``--finalize`` on
``planner/rates.json``. Fakes only; nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load("rubric_smoke")
wh = sys.modules["wave_h"]

H100_H, CORE_H, GIB_H = 3.95, 0.0472, 0.0080
TP2_PER_S = (2 * H100_H + 8 * CORE_H + 96 * GIB_H) / 3600.0


def _rates():
    from importlib import resources

    return json.loads(resources.files("flora_deploy.planner").joinpath("rates.json").read_text())


def _table(data=None):
    from flora_deploy.planner.rates import RateTable

    return RateTable.from_dict(data or _rates(), source="test")


def _manifest(**over):
    eps = {j.episode: 15.0 for j in rs.JOBS}
    eps.update(over)
    return {"episodes": [{"id": k, "duration_s": v} for k, v in eps.items()]}


# ---------------------------------------------------------------------------------------------
# Dry-run cost


def test_tp2_container_prices_two_gpus():
    c = rs.rubric_container()
    assert (c.gpu, c.gpu_count, c.cpu, c.memory_gib) == ("H100!", 2, 8.0, 96.0)
    assert c.usd_per_s() == pytest.approx(TP2_PER_S)
    one = wh.Container(c.gpu, c.cpu, c.memory_gib)  # wave_h's: one GPU
    assert c.usd_per_s() - one.usd_per_s() == pytest.approx(H100_H / 3600.0)


def test_job_projection_is_cold_rewarder_startup_and_placeholder_rate():
    t = _table()
    start, start_up, basis = rs.startup_s(t)
    assert start == pytest.approx(318.0)  # 229 s measured x the cold/warm 318/229
    assert start_up == pytest.approx(318.0 * 1.5)
    assert "reward.*" in basis
    p = rs.project_job(rs.JOBS[0], 20.0, t)
    nominal_s = 318.0 + 20.0 * 0.75 + rs.IDLE_AFTER_JOB_S
    upper_s = 318.0 * 1.5 + 20.0 * 0.75 * 2 + rs.IDLE_AFTER_JOB_S + rs.CLIENT_SLACK_S
    assert p.usd == pytest.approx(nominal_s * TP2_PER_S, abs=1e-3)
    assert p.usd_upper == pytest.approx(upper_s * TP2_PER_S, abs=1e-3)
    assert p.peak_usd_per_s == pytest.approx(TP2_PER_S)


def test_dry_run_table_and_caps():
    projs = rs.projections(_manifest(), _table())
    assert set(projs) == set(rs.STEPS)
    assert projs[rs.CHECK].usd == projs[rs.CHECK].usd_upper == 0.0
    caps = rs.caps_for(projs)
    for s in rs.STEPS:
        assert caps[s] >= projs[s].usd_upper - 1e-9
        assert caps[s] - projs[s].usd_upper < 0.05 + 1e-9
    gcap = rs.global_cap(projs, rs.STEPS)
    upper = sum(p.usd_upper for p in projs.values())
    assert upper <= gcap < upper + 0.1 + 1e-9
    assert rs.global_cap(projs, rs.STEPS, override=gcap + 1) == gcap + 1
    assert rs.global_cap(projs, rs.STEPS, override=0.1) == gcap  # never lowered
    assert rs.global_cap(projs, rs.planned("lid")) < gcap / 2  # one job, one cold start


def test_dry_run_spends_nothing(monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise AssertionError("the dry run started something")

    monkeypatch.setattr(rs, "run_job", boom)
    monkeypatch.setattr(rs, "cleanup", boom)
    monkeypatch.setattr(rs.subprocess, "run", boom)
    monkeypatch.setattr(rs, "load_manifest", lambda offline: _manifest())
    monkeypatch.setattr(rs, "image_lock_problems", lambda image=None: [])
    rc = rs.main(["--state", str(tmp_path / "s.json"), "--work", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert rc == 0 and "dry run: nothing started" in out
    assert "total: nominal $" in out and "upper $" in out
    assert not (tmp_path / "s.json").exists()


# ---------------------------------------------------------------------------------------------
# Stop rule


def _state(spent: float = 0.0, step: str = "lid") -> wh.State:
    st = wh.State(run_id="r1", created=0.0, billing_since="2026-10-21")
    if spent:
        st.step(step).attempts.append(wh.Attempt(started=0.0, status="done", finished=1.0,
                                                 tracked_usd=spent))
        st.step(step).status = "done"
    return st


def test_stop_rule_on_the_upper_projection():
    projs = rs.projections(_manifest(), _table())
    caps = rs.caps_for(projs)
    gcap = rs.global_cap(projs, rs.STEPS)
    wh.check_before(_state(), "lid", projs["lid"], caps=caps, global_cap=gcap)
    with pytest.raises(wh.StopRule, match="over its cap"):
        wh.check_before(_state(), "lid", projs["lid"],
                        caps={**caps, "lid": projs["lid"].usd_upper - 0.01}, global_cap=gcap)
    # the earlier jobs overspent: the next job's upper no longer fits what is left
    st = _state(spent=gcap - projs["braid"].usd_upper + 0.02)
    with pytest.raises(wh.StopRule, match="global budget left"):
        wh.check_before(st, "braid", projs["braid"], caps=caps, global_cap=gcap)


def test_in_step_deadline_prices_two_gpus():
    projs = rs.projections(_manifest(), _table())
    caps = rs.caps_for(projs)
    c = rs.rubric_container()
    d = wh.gpu_deadline_s(_state(), "lid", c, caps=caps, global_cap=10.0)
    assert d == pytest.approx((caps["lid"] - rs.CLIENT_SLACK_S * TP2_PER_S) / TP2_PER_S, rel=1e-6)
    assert d > projs["lid"].items[0]["upper_seconds"]  # the upper cold start fits the deadline


def test_yes_spend_stops_before_a_job_that_does_not_fit(monkeypatch, tmp_path):
    projs = rs.projections(_manifest(), _table())
    gcap = rs.global_cap(projs, rs.STEPS)
    state = tmp_path / "state.json"
    wh.save_state(_state(spent=gcap - 0.5), state)  # lid done, nearly all of the cap spent
    calls: list[str] = []
    monkeypatch.setattr(rs, "load_manifest", lambda offline: _manifest())
    monkeypatch.setattr(rs, "preconditions", lambda *a, **k: {})
    monkeypatch.setattr(rs, "run_job", lambda ctx, job: calls.append(job.task))
    monkeypatch.setattr(wh, "billed_by_app", lambda since: {})
    monkeypatch.setattr(rs, "cleanup", lambda apps, run_id, keep: calls.append("cleanup") or [])
    monkeypatch.setattr(rs, "REPORT", tmp_path / "runbook.md")
    monkeypatch.setattr(rs, "RESULTS", tmp_path / "results.json")
    rc = rs.main(["--yes-spend", "--state", str(state), "--work", str(tmp_path / "w")])
    st = wh.load_state(state)
    assert rc == 3 and calls == ["cleanup"]  # nothing ran; apps and Volumes still cleaned up
    assert st.steps["braid"].status == "stopped"
    assert "global budget left" in st.steps["braid"].notes[-1]


def test_yes_spend_refuses_on_a_failed_precondition(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "load_manifest", lambda offline: _manifest())
    monkeypatch.setattr(rs, "preconditions", lambda *a, **k: {"image": ["img_vllm: stale"]})
    monkeypatch.setattr(rs, "run_job", lambda ctx, job: pytest.fail("ran"))
    rc = rs.main(["--yes-spend", "--state", str(tmp_path / "s.json"),
                  "--work", str(tmp_path / "w")])
    assert rc == 2


# ---------------------------------------------------------------------------------------------
# Preconditions


def test_preconditions_pass_on_the_shipped_checkout(tmp_path):
    assert rs.preconditions(_manifest(), rs.JOBS, tmp_path, "r1") == {}
    assert rs.image_lock_problems() == []


def test_preconditions_name_what_is_missing(tmp_path, monkeypatch):
    m = {"episodes": [{"id": rs.JOBS[0].episode, "duration_s": 15.0}]}
    pre = rs.preconditions(m, rs.JOBS, tmp_path, "r1", lock_problems=lambda: ["stale lock"])
    assert set(pre) == {"staged", "image"}
    assert rs.JOBS[1].episode in pre["staged"][0] and rs.JOBS[0].episode not in pre["staged"][0]
    offline = rs.preconditions({**m, "offline": True}, rs.JOBS, tmp_path, "r1",
                               lock_problems=lambda: [])
    assert offline == {}  # staging is not checked offline
    bad = rs.Job("lid", rs.JOBS[0].episode, "lid")
    monkeypatch.setitem(rs.RUBRICS, "lid", [{"action": "", "grading": []}])
    pre = rs.preconditions(_manifest(), [bad], tmp_path, "r1", lock_problems=lambda: [])
    assert "rubric" in pre and "lid" in pre["rubric"][0]


def test_rubrics_validate_and_are_one_per_task():
    assert [j.task for j in rs.JOBS] == ["lid", "braid", "sandwich"]
    assert len({json.dumps(j.rubric["steps"]) for j in rs.JOBS}) == len(rs.JOBS)
    for j in rs.JOBS:
        assert rs.rubric_problems(j) == []
        assert j.episode.startswith("human/egodex_")
        assert j.max_total > 0


def test_tp_pool_problems():
    from flora_deploy.jobs.settings import PoolSettings

    def cfg(gpu_count=2, max_gpus=2, containers=1):
        pool = PoolSettings(gpu="H100!", gpu_count=gpu_count, max_containers=containers,
                            cpu_cores=8.0, memory_gib=96.0)
        return SimpleNamespace(preset="exact", modal=SimpleNamespace(max_gpus=max_gpus),
                               pools=lambda: {rs.KIND: pool})

    plan = SimpleNamespace(worker_classes={rs.KIND: "RubricWorker"},
                           object=lambda name: SimpleNamespace(gpu="H100!:2"))
    assert rs.tp_pool_problems(cfg(), plan) == []
    assert any("max_gpus" in p for p in rs.tp_pool_problems(cfg(max_gpus=1)))
    assert any("max_gpus" in p for p in rs.tp_pool_problems(cfg(containers=2)))
    assert any("TP=2" in p for p in rs.tp_pool_problems(cfg(gpu_count=1)))
    wrong = SimpleNamespace(worker_classes={rs.KIND: "RubricWorker"},
                            object=lambda name: SimpleNamespace(gpu="H100!"))
    assert any("plan_app" in p for p in rs.tp_pool_problems(cfg(), wrong))


def test_config_is_rubric_only_tp2_at_max_gpus_2(tmp_path):
    from flora_deploy.config.schema import load_config

    cfg = load_config(rs.write_config(tmp_path, rs.JOBS[1], "r1", budget_usd=1.9))
    assert cfg.preset == "exact" and cfg.pipeline.stages == (rs.KIND,)
    assert cfg.modal.max_gpus == 2 and cfg.pools()[rs.KIND].gpu_count == 2
    assert cfg.pipeline.rubric.task_name == rs.JOBS[1].task_name
    twin = load_config(rs.write_config(tmp_path, rs.JOBS[1], "r1", budget_usd=1.0,
                                       preset="balanced"))
    assert twin.preset == "balanced" and twin.allow_uncertified


# ---------------------------------------------------------------------------------------------
# Measurements


def test_medoid_stats():
    s = [{"scores": [2, 2, 1], "parsed": True}] * 6 + [
        {"scores": [0, 0, 0], "parsed": True}, {"scores": None, "parsed": False},
        {"scores": [2, 1, 1], "parsed": True}]
    m = rs.medoid_stats(s, 6.0)
    assert (m["samples"], m["parsed"], m["medoid"]) == (9, 8, [2.0, 2.0, 1.0])
    assert m["loo_stability"] == 1.0 and m["agreement"] == 0.75
    assert m["spread"] == pytest.approx((5 + 1) / 8 / 6.0, abs=1e-4)
    split = rs.medoid_stats([{"scores": [0], "parsed": True}, {"scores": [2], "parsed": True}], 2)
    assert split["medoid"] == [0.0] and split["loo_stability"] == 0.5  # ties: lowest index
    none = rs.medoid_stats([{"scores": None, "parsed": False}] * 9, 6.0)
    assert none["parsed"] == 0 and none["medoid"] is None


def _record(job=None, **prov):
    from flora_deploy import LOGIC_VERSION
    from flora_deploy.records import Provenance, RubricReward, StageRecord

    job = job or rs.JOBS[0]
    p = dict(models={"rewarder": "abc"}, engine="vllm-0.24.0-offline", preset="exact",
             config_hash="a" * 64, logic_version=LOGIC_VERSION, framework="flora-deploy 0.1",
             gpu="NVIDIA H100 80GB HBM3", certificate=None,
             source={"main": {"uri": "lid/x.mp4", "range_s": [0.0, 14.6]}},
             embodiment="human", cert_key={"kind": rs.KIND, "preset": "exact"})
    p.update(prov)
    return StageRecord(uid="ds--x", dataset="ds", episode_id="x", stage=rs.KIND,
                       provenance=Provenance(**p), certified=True,
                       rubric=RubricReward(scores=(2.0, 1.0, 2.0), total=5.0, samples=9,
                                           medoid_index=3, parsed_samples=8))


def test_record_problems():
    key = {"kind": rs.KIND, "preset": "exact"}
    assert rs.record_problems(_record(), rs.JOBS[0], stage_hash="a" * 64, cert_key=key) == []
    assert rs.record_problems(_record(certificate="cert-1"), rs.JOBS[0])
    assert rs.record_problems(_record(), rs.JOBS[0], stage_hash="b" * 64)
    assert rs.record_problems(_record(gpu="NVIDIA L40S"), rs.JOBS[0])
    assert rs.record_problems(_record(cert_key=None), rs.JOBS[0])
    assert rs.record_problems(_record(), rs.JOBS[0], cert_key={**key, "x": 1})
    assert rs.record_problems(_record(source={}), rs.JOBS[0])
    sandwich = rs.JOBS[2]  # its first step's max is 1: a score of 2 is out of range
    assert any("outside" in p for p in rs.record_problems(_record(), sandwich))


def test_parse_ready_reads_the_worker_line():
    from flora_deploy.modal_app.workers_app import RUBRIC_READY, rubric_ready_facts

    facts = rubric_ready_facts({"engine_build_s": 201.5, "enter_total_s": 240.0, "x": "y"},
                               "NVIDIA H100 80GB HBM3", 2)
    log = ("2026-10-22T10:00:00Z noise\n"
           f"2026-10-22T10:04:00Z fc-1 {RUBRIC_READY} {json.dumps(facts)}\n")
    got = rs.parse_ready(log)
    assert got["engine_build_s"] == 201.5 and got["gpu_count"] == 2 and "x" not in got
    assert rs.parse_ready("nothing here") is None


def test_job_metrics_from_report_rows():
    r = {"video_s": 14.6, "containers": [
        {"kind": rs.KIND, "startup_s": 300.0, "busy_s": 29.2, "seconds": 360.0, "usd": 0.9,
         "gpu_seen": "NVIDIA H100 80GB HBM3", "gpu_count": 1}]}
    m = rs.job_metrics(r)
    assert m["busy_container_s_per_video_s"] == 2.0 and m["startup_s"] == [300.0]
    assert m["usd_tp2"] == pytest.approx(360.0 * TP2_PER_S, abs=1e-4)
    assert m["gpu_count_labels"] == [1]  # jobs.containers labels every GPU row 1


# ---------------------------------------------------------------------------------------------
# flora certify refuses rubric records


def test_certify_checks_with_a_fake_cli(tmp_path):
    seen = []

    def cli(args):
        seen.append(args)
        return 2, "reward.rubric: no certification metrics"

    out = rs.certify_checks(tmp_path / "r.jsonl", tmp_path / "e.yaml", tmp_path / "b.yaml",
                            tmp_path / "out", cli=cli)
    assert out["ok"] and len(seen) == 3
    out = rs.certify_checks(tmp_path / "r.jsonl", tmp_path / "e.yaml", tmp_path / "b.yaml",
                            tmp_path / "out", cli=lambda a: (0, "ok"))
    assert not out["ok"]
    out = rs.certify_checks(tmp_path / "r.jsonl", tmp_path / "e.yaml", tmp_path / "b.yaml",
                            tmp_path / "out", cli=lambda a: (1, "Traceback (most recent"))
    assert not out["ok"]


def test_flora_certify_refuses_rubric_records(tmp_path):
    from typer.testing import CliRunner

    from flora_deploy.cli._app import app

    recs = tmp_path / "rubric.jsonl"
    recs.write_text(_record().to_json() + "\n")
    exact = rs.write_config(tmp_path, rs.JOBS[0], "r1", budget_usd=1.0)
    twin = rs.write_config(tmp_path, rs.JOBS[0], "r1", budget_usd=1.0, preset="balanced")
    out = tmp_path / "certs"
    base = ["certify", "--exact", str(recs), "--balanced", str(recs), "--out-dir", str(out)]
    for extra, needle in ((["-c", str(exact)], "reference by construction"),
                          (["-c", str(twin), "--stage", rs.KIND], "no certification metrics"),
                          (["-c", str(twin)], "no certifiable stage")):
        r = CliRunner().invoke(app, [*base, *extra])
        assert r.exit_code == 2, (extra, r.output)
        assert needle in " ".join(r.output.split()), r.output
        assert r.exception is None or isinstance(r.exception, SystemExit)
        assert not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------------------------------------
# Finalize


def _done_state(jobs=rs.JOBS, busy=30.0, video=15.0, startup=300.0) -> wh.State:
    st = _state()
    for j in jobs:
        s = st.step(j.task)
        s.status = "done"
        s.result = {"video_s": video, "containers": [
            {"kind": rs.KIND, "startup_s": startup, "busy_s": busy, "seconds": 400.0,
             "gpu": "H100", "gpu_seen": "NVIDIA H100 80GB HBM3"}]}
    return st


def test_finalize_marks_the_rubric_row_measured_and_leaves_balanced_alone():
    data = _rates()
    new = rs.finalized_rates(data, _done_state(), date="2026-10-23", source="results.json")
    row = next(r for r in new["rates"] if r["kind"] == rs.KIND)
    assert row["status"] == "measured" and row["n"] == 3
    assert row["container_s_per_video_s"] == 2.0 and row["source"] == "results.json"
    assert row["image"].startswith("img_vllm")
    for preset_kind in (("balanced", "segment"), ("balanced", "caption")):
        old = [r for r in data["rates"] if (r["preset"], r["kind"]) == preset_kind]
        assert [r for r in new["rates"] if (r["preset"], r["kind"]) == preset_kind] == old
    assert [r for r in new["startup"] if r["preset"] == "balanced"] == \
        [r for r in data["startup"] if r["preset"] == "balanced"]
    others = [r for r in data["rates"] if r["kind"] != rs.KIND]
    assert [r for r in new["rates"] if r["kind"] != rs.KIND] == others
    kinds = [r["kind"] for r in new["startup"]]
    assert kinds.index(rs.KIND) < kinds.index("reward.*")
    t = _table(new)
    st = t.startup_for("exact", rs.KIND)
    assert (st.startup_s, st.compile_cache) == (300.0, "cold")
    assert wh.cold_startup_s(st)[0] == 300.0  # measured cold: not scaled again
    assert t.startup_for("exact", "reward.terminal").startup_s == 229  # unchanged
    assert t.rate("exact", rs.KIND, 1, 1).status == "measured"


def test_finalize_one_job_is_n1_with_a_50_percent_band():
    st = _done_state(jobs=rs.JOBS[:1], video=31.0, busy=62.0)
    row = next(r for r in rs.finalized_rates(_rates(), st, date="d", source="s")["rates"]
               if r["kind"] == rs.KIND)
    assert row["n"] == 1 and row["uncertainty"] == 0.5 and "n=1" in row["provenance"]


def test_finalize_refuses_without_a_measurement(tmp_path):
    with pytest.raises(SystemExit, match="refused"):
        rs.finalized_rates(_rates(), _state(), date="d", source="s")
    st = _done_state()
    st.steps["lid"].result["containers"][0]["gpu_seen"] = "NVIDIA L40S"
    for j in rs.JOBS[1:]:
        st.steps[j.task].status = "failed"
    with pytest.raises(SystemExit, match="refused"):
        rs.finalized_rates(_rates(), st, date="d", source="s")


def test_finalize_writes_a_loadable_file(tmp_path):
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(_rates()))
    rs.finalize(_done_state(), path, date="2026-10-23")
    _table(json.loads(path.read_text()))


def test_cleanup_stops_apps_and_deletes_the_run_volumes():
    seen = []

    def run(args, **kw):
        seen.append(args)
        return SimpleNamespace(returncode=0)

    rs.cleanup(["a2", "a1", "a1"], "r1", keep=False, run=run)
    assert seen[:2] == [["modal", "app", "stop", "--yes", "a1"],
                        ["modal", "app", "stop", "--yes", "a2"]]
    assert {tuple(a[3:4]) for a in seen[2:]} == {("flora-rubric-smoke-r1",),
                                                  ("flora-rubric-smoke-media-r1",)}
    seen.clear()
    rs.cleanup(["a1"], "r1", keep=True, run=run)
    assert len(seen) == 1
