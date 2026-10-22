"""Review 6 (e2e) regressions: the local runner and the CLI/SDK driven end to end.

Each test reproduces one finding of review 6 (H1-H4, M1-M7, L2-L5) and fails on the code the
review ran against (main 74d3a0d)."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy import sdk
from flora_deploy.cli import app, main
from flora_deploy.export.compose import compose
from flora_deploy.export.results import iter_records, load_summary, read_parquet_bytes
from flora_deploy.jobs.manifest import load_manifest
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import LocalRunError, LocalRunner, run_local
from flora_deploy.local import sample_dataset as make_sample

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
JOB = "local-demo.fake"  # local runs are fake-models jobs (review 8b M5)
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")
cli = CliRunner()


# --- helpers ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    return make_sample(tmp_path_factory.mktemp("videos") / "videos")


def scenario(d: Path, videos: Path, **changes) -> Path:
    """``d/flora.yaml`` (the demo template with ``changes``: ``section__key=value``) and a
    copy of the demo videos in ``d/videos``."""
    data = yaml.safe_load(TEMPLATE.read_text())
    for key, value in changes.items():
        if "__" in key:
            sec, k = key.split("__", 1)
            data.setdefault(sec, {})[k] = value
        else:
            data[key] = value
    d.mkdir(parents=True, exist_ok=True)
    (d / "flora.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    if not (d / "videos").exists():
        shutil.copytree(videos, d / "videos")
    return d / "flora.yaml"


def counts(res) -> dict[str, int]:
    st = LocalStore(res.store_root)
    return {stage: len(list(iter_records(st, res.job_id, stage, config_hash=res.config_hash)))
            for stage in res.summary["stages"]}


def invoke(*args: str):
    return cli.invoke(app, list(args))


# --- H1 ---------------------------------------------------------------------------------------


def _commands(group) -> list[list[str]]:
    import typer.main

    out: list[list[str]] = []

    def walk(cmd, prefix):
        subs = getattr(cmd, "commands", None)
        if subs:
            for name, sub in sorted(subs.items()):
                walk(sub, [*prefix, name])
        else:
            out.append(prefix)

    walk(typer.main.get_command(group), [])
    return out


COMMANDS = _commands(app)


def test_every_command_is_listed():
    names = {c[0] for c in COMMANDS}
    assert {"run", "status", "resume", "cancel", "report", "logs", "results", "compose",
            "index", "deploy", "plan", "init", "prepare"} <= names


@pytest.mark.parametrize("cmd", COMMANDS, ids=[" ".join(c) for c in COMMANDS])
def test_help_works_for_every_command(cmd):
    """H1: rich parsed ``[/prefix]`` in the --store help as a closing tag and crashed."""
    out = invoke(*cmd, "--help")
    assert out.exit_code == 0, (out.output, repr(out.exception))
    assert "Usage" in out.output and "--help" in out.output
    if "--store" in out.output and "modal-volume://" in out.output:
        assert "[/" in out.output  # the literal brackets are shown, not eaten


# --- H2 ---------------------------------------------------------------------------------------


def test_local_applies_the_configured_error_rate(tmp_path, videos):
    """H2: 2 of 6 episodes excluded under max_episode_error_rate 0.5 is ``done`` (orchestrator
    and Modal path agree); main composed at 10% and then called the job ``failed``."""
    cfg = scenario(tmp_path, videos, retries={"max_starts_before_quarantine": 3,
                                              "max_episode_error_rate": 0.5})
    res = run_local(cfg, store=tmp_path / "store",
                    faults="nverr:ep003@segcap,nverr:ep004@segcap")
    assert res.outcome == "done" and res.state == "done", res.as_dict()
    assert res.summary["status"] == "complete" and res.summary["max_error_rate"] == 0.5
    assert res.summary["stages"]["caption"]["excused"] == 2
    out = invoke("run", "--local", "--fake-models", "--config", str(cfg), "--store",
                 str(tmp_path / "store"))
    assert out.exit_code == 0, out.output


def test_an_exclusion_within_the_rate_is_done(tmp_path, videos):
    """H2: one excluded episode within the rate is ``done`` (exit 0), as on Modal; main said 3."""
    cfg = scenario(tmp_path, videos, retries={"max_starts_before_quarantine": 3,
                                              "max_episode_error_rate": 0.2})
    res = run_local(cfg, store=tmp_path / "store", faults="nverr:ep005@reward.dense")
    assert res.state == "done", res.as_dict()
    assert res.summary["stages"]["reward.dense"]["excused"] == 1


def test_over_the_rate_is_completed_with_errors(tmp_path, videos):
    cfg = scenario(tmp_path, videos)
    res = run_local(cfg, store=tmp_path / "store", faults="nverr:ep005@reward.dense")
    assert res.outcome == "completed_with_errors"
    assert res.state == "completed_with_errors"


# --- H3 ---------------------------------------------------------------------------------------


def test_adding_a_stage_reruns_only_that_stage(tmp_path, videos):
    """H3: local mode plans with stage hashes and per-config manifests, as ``sdk.submit``: a
    config that adds ``reward.terminal`` runs only the reward shard; main refused the job
    ("planned for config ...")."""
    seg = {"stages": ["segment", "caption"], "embodiment": "human",
           "episode_instruction": {"source": "dataset_task"}, "envelope": "warn",
           "shard_target_gpu_minutes": 0.5}
    cfg = scenario(tmp_path, videos, pipeline=seg, resources={"segcap": {"max_containers": 2}})
    store = tmp_path / "store"
    a = run_local(cfg, store=store)
    assert a.state == "done" and a.spawner["calls"] == 3
    seg2 = {**seg, "stages": ["segment", "caption", "reward.terminal"]}
    cfg = scenario(tmp_path, videos, pipeline=seg2,
                   resources={"segcap": {"max_containers": 2},
                              "reward.terminal": {"max_containers": 1}})
    b = run_local(cfg, store=store)
    assert b.state == "done", b.as_dict()
    assert b.config_hash != a.config_hash
    assert b.spawner["calls"] == 1  # the reward shard only; segcap is reused
    assert counts(b) == {"segment": 6, "caption": 6, "reward.terminal": 6}
    st = LocalStore(store)
    ma, mb = load_manifest(st, JOB, a.config_hash), load_manifest(st, JOB, b.config_hash)
    assert ma.stage_hashes["segcap"] == mb.stage_hashes["segcap"] != a.config_hash
    recs = list(iter_records(st, JOB, "caption", config_hash=b.config_hash))
    assert {r.provenance.config_hash for r in recs} == {ma.stage_hashes["segcap"]}
    # back to the first config: everything is reused, nothing spawned
    cfg = scenario(tmp_path, videos, pipeline=seg, resources={"segcap": {"max_containers": 2}})
    again = run_local(cfg, store=store)
    assert again.state == "done" and again.spawner["calls"] == 0
    assert again.config_hash == a.config_hash


# --- H4 + M1 + M4 -----------------------------------------------------------------------------


def test_local_jobs_work_with_status_report_and_the_sdk(tmp_path, videos):
    """H4: a local run writes its run spec; main answered "has no run spec" (exit 2)."""
    cfg = scenario(tmp_path, videos)
    store = tmp_path / "store"
    res = run_local(cfg, store=store)
    assert res.state == "done"
    for s in (str(store), store.as_uri()):
        st = invoke("status", JOB, "--store", s)
        assert st.exit_code == 0, st.output
        assert "done" in st.output and "app local" in st.output
    js = json.loads(invoke("status", JOB, "--store", str(store), "--json").output)
    assert js["state"] == "done" and js["composed"] and js["app_name"] == "local"
    rep = invoke("report", JOB, "--store", str(store), "--json")
    assert rep.exit_code == 0, rep.output
    r = json.loads(rep.output)
    assert r["state"] == "done" and r["spend"]["total_usd"] > 0
    assert set(r["outputs"]["caption"]) == {"jsonl", "parquet"}
    assert invoke("report", JOB, "--store", str(store)).exit_code == 0
    lg = invoke("logs", JOB, "--store", str(store))
    assert lg.exit_code == 2 and "ran locally" in lg.output
    job = sdk.open_job(JOB, store=str(store))
    assert job.is_local and job.status().state == "done"
    assert len(job.results("caption")) == 6
    # M1: cancelling a finished job is a no-op
    assert job.cancel() == []
    assert job.status().state == "done"
    assert invoke("cancel", JOB, "--store", str(store), "--yes").exit_code == 2
    assert invoke("status", JOB, "--store", str(store)).exit_code == 0


def test_a_late_cancel_record_never_overrides_a_resolved_job(tmp_path, videos):
    """M1: the reconciler ignores a cancel newer than a resolved outcome (the CLI's
    check-then-write race); main reported ``cancelled`` (exit 5)."""
    from flora_deploy.jobs.orchestrator import utc, write_control
    from flora_deploy.jobs.store import JobPaths
    from flora_deploy.jobs.types import ControlRecord

    cfg = scenario(tmp_path, videos)
    res = run_local(cfg, store=tmp_path / "store")
    st = LocalStore(res.store_root)
    write_control(st, JobPaths(JOB), ControlRecord("cancel", JOB, "late", utc(4e9),
                                                   {"config_hash": res.config_hash}))
    assert sdk.open_job(JOB, store=str(res.store_root)).status().state == "done"


def test_local_budget_gate_is_the_production_gate(tmp_path, videos):
    """M4: the demo projects ~$1.2 on Modal (startup and scale-down included); main's local gate
    (no startup, no tail) ran it to ``done`` under $1."""
    cfg = scenario(tmp_path, videos)
    res = run_local(cfg, store=tmp_path / "store", budget_usd=1.0)
    assert res.state == "budget_stopped", res.as_dict()
    # the same budget again with --resume is refused, as `flora resume` refuses it
    with pytest.raises(LocalRunError, match="pass --budget-usd"):
        run_local(cfg, store=tmp_path / "store", resume=True)
    with pytest.raises(LocalRunError, match="must be positive"):
        run_local(cfg, store=tmp_path / "store", resume=True, budget_usd=-1.0)
    # a new budget still under the projection is accepted and stops again before any spend
    low = run_local(cfg, store=tmp_path / "store", resume=True, budget_usd=0.5)
    assert low.state == "budget_stopped" and low.spawner["calls"] == 0
    done = run_local(cfg, store=tmp_path / "store", resume=True, budget_usd=20.0)
    assert done.state == "done", done.as_dict()


def test_flora_resume_and_cancel_drive_local_jobs(tmp_path, videos):
    """H4: `flora resume` / `flora cancel` on budget-stopped local jobs."""
    cfg = scenario(tmp_path, videos)
    store = str(tmp_path / "store")
    out = invoke("run", "--local", "--fake-models", "--config", str(cfg), "--store", store,
                 "--budget", "0.2")
    assert out.exit_code == 4, out.output
    assert invoke("status", JOB, "--store", store).exit_code == 4
    refused = invoke("resume", JOB, "--store", store)
    assert refused.exit_code == 2 and "--budget-usd" in refused.output
    ok = invoke("resume", JOB, "--store", store, "--budget-usd", "20")
    assert ok.exit_code == 0, ok.output
    assert "resumed local-demo.fake" in ok.output and "done" in ok.output
    assert len(sdk.open_job(JOB, store=store).results("reward.dense")) == 6
    # another job in the same store: budget-stopped, then cancelled for good
    out = invoke("run", "--local", "--fake-models", "--config", str(cfg), "--store", store,
                 "--budget", "0.2", "--job", "other")
    assert out.exit_code == 4, out.output
    c = invoke("cancel", "other.fake", "--store", store, "--yes")
    assert c.exit_code == 0, c.output
    assert invoke("status", "other.fake", "--store", store).exit_code == 5
    assert invoke("resume", "other.fake", "--store", store, "--budget-usd", "20").exit_code == 2
    again = invoke("run", "--local", "--fake-models", "--config", str(cfg), "--store", store,
                   "--job", "other", "--resume", "--budget", "20")
    assert again.exit_code == 2 and "cancelled" in again.output


# --- M2 ---------------------------------------------------------------------------------------


class DiesAfter:
    """A store whose ``n``-th output write (and every later one) raises: a killed compose."""

    def __init__(self, store, n: int):
        self.store, self.n, self.writes = store, n, 0

    def put_bytes(self, path, data):
        if "/output/" in path:
            self.writes += 1
            if self.writes >= self.n:
                raise KeyboardInterrupt("compose killed")
        self.store.put_bytes(path, data)

    def get_bytes(self, path):
        return self.store.get_bytes(path)

    def exists(self, path):
        return self.store.exists(path)

    def list(self, prefix):
        return self.store.list(prefix)


def test_an_interrupted_recompose_never_mixes_generations(tmp_path, videos):
    """M2: a budget stop composes a partial, empty set; after the resume a re-compose dies after
    two output files. Readers must still see the first set whole; main returned 6 ``segment``
    records under a summary that said 0."""
    cfg = scenario(tmp_path, videos)
    store = tmp_path / "store"
    stopped = run_local(cfg, store=store, budget_usd=0.2)
    assert stopped.state == "budget_stopped"
    first = stopped.summary
    assert all(c["records"] == 0 for c in first["stages"].values())
    runner = LocalRunner(cfg, store=store, budget_usd=20.0)
    try:
        runner.index()
        runner.plan()
        runner.resume()
        assert runner.orchestrate() == "done"
    finally:
        runner.close()
    st = LocalStore(store)
    with pytest.raises(KeyboardInterrupt):
        compose(DiesAfter(st, 3), JOB, config_hash=stopped.config_hash)
    s = load_summary(st, JOB, config_hash=stopped.config_hash)
    assert s["generation"] == first["generation"] and s["status"] == "partial"
    for stage in STAGES:
        got = list(iter_records(st, JOB, stage, config_hash=stopped.config_hash))
        assert len(got) == s["stages"][stage]["records"] == 0
    full = compose(st, JOB, config_hash=stopped.config_hash)
    assert full.summary["generation"] != first["generation"]
    assert all(len(list(iter_records(st, JOB, stage, config_hash=stopped.config_hash))) == 6
               for stage in STAGES)
    again = compose(st, JOB, config_hash=stopped.config_hash)  # idempotent: same generation
    assert again.summary["generation"] == full.summary["generation"]
    assert again.summary["outputs"] == full.summary["outputs"]


# --- M3 / M6 ----------------------------------------------------------------------------------


def test_an_episode_without_task_text_is_excluded_not_the_whole_shard(tmp_path, videos):
    """M3: an extra episode with no ``.txt`` (and no fixed instruction) is a per-episode
    exclusion at plan time; main failed every reward shard (and the job) on it."""
    cfg = scenario(tmp_path, videos)
    shutil.copy(tmp_path / "videos" / "ep000.mp4", tmp_path / "videos" / "ep006.mp4")
    res = run_local(cfg, store=tmp_path / "store")
    assert res.state == "completed_with_errors", res.as_dict()  # 1 of 7 > 10%
    s = res.summary
    (uid, reason), = s["unplanned_episodes"].items()
    assert uid.endswith("--ep006") and reason.startswith("missing_task")
    assert counts(res) == dict.fromkeys(STAGES, 6)
    assert any("left out of the plan" in p for p in s["problems"])
    # within the rate it is a done job, as on Modal
    cfg = scenario(tmp_path, videos, retries={"max_starts_before_quarantine": 3,
                                              "max_episode_error_rate": 0.2})
    assert run_local(cfg, store=tmp_path / "store2").state == "done"


@pytest.mark.parametrize("rate,state", [(0.1, "completed_with_errors"), (0.2, "done")])
def test_a_planned_episode_without_task_text_is_a_worker_exclusion(tmp_path, videos, rate,
                                                                   state):
    """A manifest that still holds an episode without task text (its receipt lost the text
    after planning): the local reward workers, the Modal path's ``RewardShardRunner``, exclude
    that episode only (``missing_task``) and finish their shards; it counts toward the rate."""
    from flora_deploy.config.schema import load_config
    from flora_deploy.jobs.store import JobPaths, fake_job_id

    cfg = scenario(tmp_path, videos, retries={"max_starts_before_quarantine": 3,
                                              "max_episode_error_rate": rate})
    runner = LocalRunner(load_config(cfg), config_path=cfg, store=tmp_path / "store",
                         job_id=fake_job_id("local-demo"))
    try:
        runner.index()
        m = runner.plan()
        target = next(e.uid for e in m.episodes if e.uid.endswith("--ep002"))
        path = JobPaths(runner.job_id).index(target)
        rec = json.loads(runner.store.get_bytes(path))
        rec.update(task=None, instruction_text="")
        runner.store.put_bytes(path, json.dumps(rec).encode())
        outcome = runner.orchestrate()
        s = runner.compose(outcome)
    finally:
        runner.close()
    assert outcome == state, s
    assert s["errors"]["by_kind"] == {"missing_task": 2}
    assert s["errors"]["by_stage"] == {"reward.dense": 1, "reward.terminal": 1}
    for stage in ("reward.terminal", "reward.dense"):
        assert s["stages"][stage]["excused"] == 1 and s["stages"][stage]["records"] == 5
    assert s["stages"]["segment"]["excused"] == 0


def test_an_episode_that_fails_indexing_counts_as_excluded(tmp_path, videos):
    """M6: a zero-filled video fails indexing; it counts in the completion check and the error
    rate (1 of 7 > 10%: completed_with_errors); main ended ``done`` with 6/6."""
    cfg = scenario(tmp_path, videos)
    (tmp_path / "videos" / "ep009.mp4").write_bytes(b"\0" * 1000)
    (tmp_path / "videos" / "ep009.txt").write_text("pick up the cup")
    res = run_local(cfg, store=tmp_path / "store")
    assert res.state == "completed_with_errors", res.as_dict()
    (uid, reason), = res.summary["unplanned_episodes"].items()
    assert uid.endswith("--ep009") and reason.split(":")[0] in ("index_failed", "source_error")
    v = sdk.open_job(JOB, store=str(tmp_path / "store")).status()
    assert v.state == "completed_with_errors" and any("excluded" in n for n in v.notes)


# --- M5 ---------------------------------------------------------------------------------------


def test_local_runs_use_the_tick_driver_and_the_modal_lease_name(tmp_path, videos, monkeypatch):
    """M5: a transient tick error is retried (TickDriver), not a traceback; the lease is
    ``orchestrator/<job>`` as on Modal."""
    from flora_deploy.jobs import driver
    from flora_deploy.jobs.orchestrator import Orchestrator
    from flora_deploy.local.spawner import FileKV


    real = Orchestrator.tick
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("store hiccup")
        return real(self)

    monkeypatch.setattr(Orchestrator, "tick", flaky)
    monkeypatch.setattr(driver.TickDriver, "__init__", _fast_driver_init(driver))
    cfg = scenario(tmp_path, videos)
    res = run_local(cfg, store=tmp_path / "store")
    assert res.state == "done" and res.counts("tick_error") == 1
    kv = FileKV(tmp_path / "store" / ".flora-local" / "lease")
    assert kv.contains(f"orchestrator/{JOB}/gen/1")


def _fast_driver_init(driver):
    orig = driver.TickDriver.__init__

    def init(self, orch, **kw):
        orig(self, orch, **{**kw, "backoff_s": 0.01})

    return init


def test_timeout_fault_fails_the_call_and_is_respawned(tmp_path, videos):
    cfg = scenario(tmp_path, videos)
    res = run_local(cfg, store=tmp_path / "store", faults="timeout:reward.terminal-*")
    assert res.state == "done", res.as_dict()
    failed = [e for e in res.events if e["event"] == "call_failed"]
    assert len(failed) == 1 and "FunctionTimeoutError" in str(failed[0])
    assert res.spawner["crashes"] == 1 and not res.unfired


# --- M7 ---------------------------------------------------------------------------------------


def test_fake_records_are_not_certified(tmp_path, videos):
    cfg = scenario(tmp_path, videos)
    res = run_local(cfg, store=tmp_path / "store")
    st = LocalStore(res.store_root)
    recs = [r for stage in STAGES for r in iter_records(st, JOB, stage)]
    assert len(recs) == 24
    assert not any(r.certified for r in recs)
    assert {r.uncertified_reason for r in recs} == {"fake_models"}
    assert all(c["certified_share"] == 0 for c in res.summary["stages"].values())


# --- L2 / L3 / L4 / L5 ------------------------------------------------------------------------


def _main_exit(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["flora", *argv])
    with pytest.raises(SystemExit) as e:
        main()
    return int(e.value.code or 0)


def test_refusals_and_internal_errors_exit_2(tmp_path, videos, monkeypatch, capsys):
    """L2: main exited 1 (= ``failed``) with a traceback for these."""
    cfg = str(scenario(tmp_path, videos))
    base = ["run", "--local", "--fake-models", "--config", cfg]
    assert _main_exit(monkeypatch, *base, "--budget", "0") == 2
    assert _main_exit(monkeypatch, *base, "--budget", "-1") == 2
    assert _main_exit(monkeypatch, *base, "--job", "../evil") == 2
    assert _main_exit(monkeypatch, *base, "--job", "a b") == 2
    import flora_deploy.local as local

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(local, "run_local", boom)
    assert _main_exit(monkeypatch, *base) == 2
    assert "internal error" in capsys.readouterr().err
    missing = invoke("results", "--job", "nope", "--stage", "caption", "--store",
                     str(tmp_path / "empty"))
    assert missing.exit_code == 2
    assert invoke("compose", "--job", "nope", "--store", str(tmp_path / "empty")).exit_code == 2


def test_file_uri_store(tmp_path, videos, monkeypatch):
    """L3: ``--store file:///abs`` made a literal ``./file:`` directory."""
    cfg = scenario(tmp_path, videos)
    monkeypatch.chdir(tmp_path)
    uri = (tmp_path / "uri-store").as_uri()
    out = invoke("run", "--local", "--fake-models", "--config", str(cfg), "--store", uri)
    assert out.exit_code == 0, out.output
    assert (tmp_path / "uri-store" / "flora").is_dir() and not (tmp_path / "file:").exists()
    res = invoke("results", "--job", JOB, "--stage", "caption", "--store", uri)
    assert res.exit_code == 0 and "6 records" in res.output


def test_output_formats_are_honoured(tmp_path, videos):
    """L4: ``formats: [jsonl]`` still wrote Parquet."""
    cfg = scenario(tmp_path, videos, output={"sink": "volume://x/y", "formats": ["jsonl"]})
    res = run_local(cfg, store=tmp_path / "store")
    files = res.summary["files"]
    assert files and all(k.endswith(".jsonl") for k in files)
    assert not list(Path(res.store_root).rglob("*.parquet"))
    st = LocalStore(res.store_root)
    assert read_parquet_bytes(st, JOB, "caption")[:4] == b"PAR1"  # rebuilt from the JSONL


def test_faults_that_never_fire_are_reported(tmp_path, videos):
    """L5: ``oom:ep001`` in a LeRobot-style dataset (or a typo) silently did nothing."""
    cfg = scenario(tmp_path, videos)
    res = run_local(cfg, store=tmp_path / "store", faults="oom:ep999@segcap,preempt:segcap-*")
    assert res.state == "done"
    assert res.unfired == ["oom:ep999@segcap"]
    out = invoke("run", "--local", "--fake-models", "--config", str(cfg), "--store",
                 str(tmp_path / "store2"), "--inject", "oom:nothing")
    assert out.exit_code == 0 and "never fired" in out.output
    assert Counter(f["fault"] for f in res.faults) == {"preempt": 1}
