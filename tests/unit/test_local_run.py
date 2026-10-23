"""``flora run --local --fake-models``: whole multi-stage jobs (segcap + reward.terminal +
reward.dense) through the real index, manifest, orchestrator, store adapters, budget gate,
worker runners and compose, with the local spawner and fake engines, under injected faults.

Every scenario checks the completion state, the composed summary, the ``flora results`` output
and that the records of unaffected episodes are byte-identical to a clean run's (fake engines
are deterministic, so preemption, restarts, bisection and resume must never change outputs)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.export.results import iter_records
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs.manifest import load_manifest
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import LocalRunError, LocalRunner, completion_state, run_local
from flora_deploy.local import sample_dataset as make_sample
from flora_deploy.local.engines import FAKE_CERTIFICATE, FAKE_ENGINE
from flora_deploy.local.runner import digest_outputs

TEMPLATE = Path(__file__).resolve().parents[2] / "tests/fixtures/local_demo_exact.yaml"
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")
JOB = "local-demo.fake"  # local runs are fake-models jobs (review 8b M5)
cli = CliRunner()


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """flora.yaml (the shipped template) + its synthetic dataset, shared by the module."""
    d = tmp_path_factory.mktemp("demo")
    shutil.copy(TEMPLATE, d / "flora.yaml")
    make_sample(d / "videos")
    return d / "flora.yaml"


@pytest.fixture(scope="module")
def clean(demo, tmp_path_factory):
    res = run_local(demo, store=tmp_path_factory.mktemp("clean") / "store")
    assert res.state == "done", res.as_dict()
    return res


def shard(res, kind: str, n: int = 0) -> str:
    """The id of the ``n``-th planned ``kind`` shard. Shard ids are content-addressed (Fix-W1:
    ``<kind>-<16 hex of sha256(kind + uids)>``), so fault targets name the planned id; the plan
    is deterministic, so a clean run's ids are every run's ids."""
    m = load_manifest(LocalStore(res.store_root), res.job_id)
    return [x.shard_id for x in m.shards if x.kind == kind and not x.bisected_from][n]


def records(res) -> dict[tuple[str, str], str]:
    st = LocalStore(res.store_root)
    return {(r.uid, r.stage): r.to_json() for stage in STAGES
            for r in iter_records(st, res.job_id, stage)}


def same_as_clean(res, clean, *, missing: set[tuple[str, str]] = frozenset()) -> None:
    got, want = records(res), records(clean)
    assert set(got) == set(want) - set(missing)
    assert {k: got[k] for k in got} == {k: want[k] for k in got}


def results_cli(store, stage, *extra):
    return cli.invoke(app, ["results", "--job", JOB, "--stage", stage, "--store", str(store),
                            *extra])


# --- clean -----------------------------------------------------------------------------------


def test_clean_run_is_done_complete_and_marked_fake(clean):
    s = clean.summary
    assert s["complete"] and not s["poison_pills"] and not s["conflicts"]
    assert s["shards"] == {"done": 5}  # 3 segcap (small shards) + terminal + dense
    for stage in STAGES:
        c = s["stages"][stage]
        assert c["records"] == c["expected"] == 6 and c["missing"] == c["excused"] == 0
    recs = [json.loads(v) for v in records(clean).values()]
    assert all(r["provenance"]["certificate"] == FAKE_CERTIFICATE for r in recs)
    assert {r["provenance"]["engine"] for r in recs if r["stage"] == "caption"} == {FAKE_ENGINE}
    assert all(set(r["provenance"]["models"].values()) == {"fake"} for r in recs)
    caps = [r for r in recs if r["stage"] == "caption"]
    assert all(seg["caption"] and seg["end_s"] > seg["start_s"]
               for r in caps for seg in r["segments"])
    term = [r for r in recs if r["stage"] == "reward.terminal"]
    assert all(1 <= r["terminal"]["score"] <= 5 for r in term)
    dense = [r for r in recs if r["stage"] == "reward.dense"]
    assert all(r["dense"]["per_step"] and set(r["dense"]["per_step"]) <= {1, 2, 3}
               for r in dense)
    assert clean.spawner["calls"] == 5 and clean.wall_s < 30


def test_results_cli_reads_the_composed_outputs(clean):
    out = results_cli(clean.store_root, "caption", "--format", "jsonl")
    assert out.exit_code == 0, out.output
    lines = [json.loads(x) for x in out.output.splitlines() if x.strip()]
    assert len(lines) == 6 and {x["stage"] for x in lines} == {"caption"}
    table = results_cli(clean.store_root, "reward.terminal")
    assert table.exit_code == 0 and "6 episodes" in table.output


def test_reruns_are_deterministic_and_resumable(demo, clean, tmp_path):
    again = run_local(demo, store=tmp_path / "store")
    assert again.state == "done" and again.config_hash == clean.config_hash
    assert digest_outputs(again.summary) == digest_outputs(clean.summary)
    # the same store again: nothing to do, same outputs, no new calls
    third = run_local(demo, store=tmp_path / "store")
    assert third.state == "done" and third.spawner["calls"] == 0
    assert digest_outputs(third.summary) == digest_outputs(clean.summary)


# --- faults ----------------------------------------------------------------------------------


def test_preempt_oom_poison_bisection_and_episode_errors(demo, clean, tmp_path, monkeypatch):
    """The orchestrator quarantines a crash-looping shard once it sees K unmarked starts, while
    the K-th may still be running (D15 presumes it crashing). Whether that start reached ep003
    and crashed, or was cancelled by the quarantine first (and marked preempted), was a race:
    the crash count and the counted starts flaked under xdist load. The K-th start of each
    looping shard is therefore held at ep003 until the quarantine cancels it, every time."""
    import threading

    from flora_deploy.local import engines as LE
    from flora_deploy.local import faults as LF

    k = 3  # local_demo.yaml retries.max_starts_before_quarantine
    uid3 = next(u for u, _ in records(clean) if u.endswith("--ep003"))
    uid5 = next(u for u, _ in records(clean) if u.endswith("--ep005"))
    checkpoint, lock = LE.checkpoint, threading.Lock()
    starts: dict[str, list[str]] = {}  # shard id -> its containers, in start order
    held: set[str] = set()

    def hold_kth(uid):
        ctx = LF.current()
        if ctx is not None and uid == uid3:
            with lock:
                cs = starts.setdefault(ctx.shard_id, [])
                if ctx.container_id not in cs:
                    cs.append(ctx.container_id)
                kth = cs.index(ctx.container_id) + 1 == k
            if kth:
                held.add(ctx.shard_id)
                assert ctx.cancelled.wait(30), "the K-th start was never cancelled"
        checkpoint(uid)  # raises the cancel (KeyboardInterrupt) for a held start

    monkeypatch.setattr(LE, "checkpoint", hold_kth)
    seg0 = shard(clean, "segcap", 0)
    res = run_local(demo, store=tmp_path / "store", faults=(
        f"preempt:{seg0},oom:ep003@segcap,nverr:ep005@reward.dense,"
        f"infra:{shard(clean, 'reward.terminal')},spawnerr:{shard(clean, 'reward.dense')}"))
    assert res.state == "completed_with_errors", res.as_dict()
    s = res.summary
    # Fix-W1 "done means done": ``complete`` is only for an error-free job; here every episode
    # is accounted for (records or terminal exclusions, nothing missing) but 1 of 6 (> 10%)
    # was excluded per stage, so the job is completed_with_errors
    assert not s["complete"] and s["status"] == "completed_with_errors"
    assert all(st["missing"] == 0 for st in s["stages"].values())
    # ep003 crashed the second segcap shard every time: quarantined, bisected, then poison-pilled
    assert s["poison_pills"] == [f"{uid3}/segcap"]
    assert s["stages"]["segment"]["excused"] == s["stages"]["caption"]["excused"] == 1
    assert s["stages"]["reward.dense"]["excused"] == 1  # the nv_failure
    assert s["stages"]["reward.terminal"]["excused"] == 0
    assert s["errors"]["by_kind"] == {"nv_failure": 1, "poison_pill": 2}
    assert res.counts("bisected") == 1 and res.counts("poisoned") == 1
    assert res.counts("call_expired") == 1 and res.counts("spawn_error") == 1
    # D15 quarantines on K starts per shard (the parent, then its ep003 child); the K-th is
    # held until the quarantine cancels it, so exactly K - 1 starts of each crash
    parent = next(e["shard_id"] for e in res.events if e["event"] == "bisected")
    child = f"segcap-ep-{uid3}"
    assert held == {parent, child}
    assert res.spawner["preemptions"] == 1 and res.spawner["crashes"] == 2 * (k - 1)
    # its batch-mate ep002 finished before the crash and was kept, not redone
    bis = next(e for e in res.events if e["event"] == "bisected")
    assert bis["children"] == [child] and len(bis["done"]) == 1
    # preempted starts never count toward D15: the first segcap shard was not quarantined
    st = LocalStore(res.store_root)
    jf = A.JobFiles(st, JOB, res.config_hash)
    marks = A.markers(jf, "segcap", seg0)
    assert any(m.preempted for m in marks)
    assert A.counted_starts(jf, "segcap", seg0) == len(marks) - 1 == 1
    # quarantined only after K starts, then poison-pilled after K more; the K-th start of each,
    # cancelled by the quarantine, is marked preempted and no longer counts
    for sid in (parent, child):
        assert A.count_starts(jf, "segcap", sid) == k
        assert A.counted_starts(jf, "segcap", sid) == k - 1
    same_as_clean(res, clean, missing={(uid3, "segment"), (uid3, "caption"),
                                       (uid5, "reward.dense")})


def test_python_failures_retry_then_succeed(demo, clean, tmp_path):
    # 4 failures: the first call fails 3 times (modal.Retries=2), the orchestrator respawns
    # it, and the second call succeeds after one more failure
    res = run_local(demo, store=tmp_path / "store",
                    faults=f"fail:{shard(clean, 'segcap', 2)}x4")
    assert res.state == "done", res.as_dict()
    assert res.counts("call_failed") == 1 and res.spawner["python_failures"] == 4
    same_as_clean(res, clean)


def test_exhausted_respawns_fail_the_job(demo, clean, tmp_path):
    res = run_local(demo, store=tmp_path / "store",
                    faults=f"fail:{shard(clean, 'reward.terminal')}x99")
    assert res.state == "failed" and res.outcome == "failed"
    assert res.counts("shard_failed") == 1
    assert res.summary["shards"]["failed"] == 1
    assert res.summary["stages"]["reward.terminal"]["missing"] == 6
    assert res.summary["stages"]["caption"]["records"] == 6  # the rest still composed


def test_orchestrator_crash_and_restart_reattaches(demo, clean, tmp_path):
    runner = LocalRunner(demo, store=tmp_path / "store", lease_ttl_s=0.5,
                         faults="preempt:segcap-*x2")
    try:
        runner.index()
        runner.plan()
        assert runner.orchestrate(crash_after_ticks=1) == "interrupted"  # lease kept
        # a successor orchestrator (same "Modal": the spawner keeps running the calls)
        assert runner.orchestrate() == "done"
        assert sum(1 for e in runner.events if e["event"] == "reattached") >= 1
        summary = runner.compose()
        res = runner.result("done", summary, 0.0)
    finally:
        runner.close()
    assert res.state == "done"
    same_as_clean(res, clean)


def test_restart_in_a_new_process_respawns_lost_calls(demo, clean, tmp_path, monkeypatch):
    """The first orchestrator dies after one tick with its calls in flight. Those calls are held
    at their first checkpoint until the "process" dies (``close`` cancels them), so none can
    finish first: without the hold, a fast fake call could write its receipt during the tick
    under load, leaving nothing lost to respawn (flaky 3/30 under load, Fix-8a)."""
    import threading

    from flora_deploy.local import engines as LE
    from flora_deploy.local import faults as LF

    checkpoint, hold = LE.checkpoint, threading.Event()
    hold.set()

    def held(uid):
        ctx = LF.current()
        if hold.is_set() and ctx is not None:
            ctx.cancelled.wait(60)  # in flight until the first process dies
        checkpoint(uid)

    monkeypatch.setattr(LE, "checkpoint", held)
    first = LocalRunner(demo, store=tmp_path / "store", lease_ttl_s=0.5)
    try:
        first.index()
        first.plan()
        assert first.orchestrate(crash_after_ticks=1) == "interrupted"
        spawned = first.spawner.stats()["calls"]
    finally:
        first.close()  # the process dies: its in-flight calls are gone with it
        hold.clear()
    assert spawned >= 1
    second = LocalRunner(demo, store=tmp_path / "store", lease_ttl_s=0.5)
    try:
        res = second.run()
    finally:
        second.close()
    assert res.state == "done", res.as_dict()
    expired = [e for e in res.events if e["event"] == "call_expired"]
    finished_first = sum(1 for e in first.events if e["event"] == "shard_done")
    assert len(expired) + finished_first >= 1
    assert finished_first == 0 and len(expired) == spawned  # every lost call was respawned
    same_as_clean(res, clean)


def test_budget_stop_then_resume(demo, clean, tmp_path):
    stopped = run_local(demo, store=tmp_path / "store", budget_usd=0.05)
    assert stopped.state == "budget_stopped", stopped.as_dict()
    assert stopped.counts("budget_stop") == 1
    # the production gate (review 6 M4): the whole job's projection, startup and scale-down
    # included, is over $0.05 before the first spawn, so nothing runs
    assert stopped.summary["shards"].get("done", 0) == 0 and stopped.spawner["calls"] == 0
    # without --resume the stop stays in force
    again = run_local(demo, store=tmp_path / "store", budget_usd=5.0)
    assert again.state == "budget_stopped" and again.spawner["calls"] == 0
    resumed = run_local(demo, store=tmp_path / "store", budget_usd=5.0, resume=True)
    assert resumed.state == "done", resumed.as_dict()
    same_as_clean(resumed, clean)


# --- CLI -------------------------------------------------------------------------------------


def test_cli_run_local_with_faults_and_sample_data(clean, tmp_path):
    shutil.copy(TEMPLATE, tmp_path / "flora.yaml")
    out = cli.invoke(app, ["run", "--local", "--fake-models", "--sample-data", "--config",
                           str(tmp_path / "flora.yaml"), "--inject",
                           f"preempt:{shard(clean, 'segcap')},oom:ep001@segcap",
                           "--inject", f"fail:{shard(clean, 'reward.dense')}"])
    assert out.exit_code == 3, out.output  # completed_with_errors
    assert "COMPLETED_WITH_ERRORS" in out.output and "episodes set aside 1" in out.output
    assert (tmp_path / "videos" / "ep000.mp4").is_file()
    store = tmp_path / "flora-local-store"
    m = load_manifest(LocalStore(store), JOB)
    assert len(m.episodes) == 6
    res = results_cli(store, "segment", "--format", "jsonl")
    assert res.exit_code == 0
    lines = [x for x in res.output.splitlines() if x.strip()]
    # Fix-W1: `flora results` warns about a completed_with_errors job; then the 5 records
    other = [x for x in lines if not x.startswith("{")]
    assert len(other) == 1 and other[0].startswith(
        "warning: job local-demo.fake is completed_with_errors"), other
    assert len([x for x in lines if x.startswith("{")]) == 5
    js = cli.invoke(app, ["run", "--local", "--fake-models", "--json", "--config",
                          str(tmp_path / "flora.yaml")])
    assert js.exit_code == 3
    d = json.loads(js.output)
    assert d["state"] == "completed_with_errors" and d["spawner"]["calls"] == 0


def test_cli_refuses_what_local_mode_cannot_do(tmp_path):
    shutil.copy(TEMPLATE, tmp_path / "flora.yaml")
    cfg = str(tmp_path / "flora.yaml")
    assert cli.invoke(app, ["run", "--config", cfg]).exit_code == 2
    assert cli.invoke(app, ["run", "--local", "--config", cfg]).exit_code == 2
    bad = cli.invoke(app, ["run", "--local", "--fake-models", "--config", cfg,
                           "--inject", "explode:everything"])
    assert bad.exit_code == 2 and "bad fault" in bad.output
    missing = cli.invoke(app, ["run", "--local", "--fake-models", "--config", cfg])
    assert missing.exit_code == 2 and "--sample-data" in missing.output


def test_unsupported_kinds_and_states(tmp_path, monkeypatch):
    import flora_deploy.local.runner as LR

    text = TEMPLATE.read_text().replace("stages: [segment, caption, reward.terminal, "
                                        "reward.dense]", "stages: [reward.terminal]")
    text = text.split("resources:")[0] + "output:" + text.split("output:")[1]
    (tmp_path / "flora.yaml").write_text(text)
    monkeypatch.setattr(LR, "SUPPORTED_KINDS", ("segcap",))  # every kind has a fake engine now
    with pytest.raises(LocalRunError, match="no engine"):
        LocalRunner(tmp_path / "flora.yaml")
    monkeypatch.undo()
    with pytest.raises(LocalRunError, match="fake-models"):
        LocalRunner(TEMPLATE, fake_models=False)
    # the orchestrator's verdict, downgraded only by compose's own rate guard (review 6 H2)
    ok = {"status": "complete", "poison_pills": [], "stages": {"caption": {"excused": 1}}}
    assert completion_state("done", ok) == "done"  # an exclusion within the rate is done
    cwe = {**ok, "status": "completed_with_errors"}
    assert completion_state("done", cwe) == "completed_with_errors"
    assert completion_state("completed_with_errors", ok) == "completed_with_errors"
    assert completion_state("done", {**ok, "status": "partial"}) == "failed"
    assert completion_state("done", None) == "failed"
    assert completion_state("failed", ok) == "failed"
    assert completion_state("budget_stopped", None) == "budget_stopped"
    assert completion_state("max_ticks", None) == "failed"


def test_changed_config_for_the_same_job_replans_under_its_own_hash(demo, tmp_path):
    """Review 6 H3: as on Modal, a changed config is planned under its own hash (its own
    manifest), not refused; ``envelope`` is hashed into every kind, so everything re-runs."""
    store = tmp_path / "store"
    first = run_local(demo, store=store)
    assert first.state == "done"
    text = demo.read_text().replace("envelope: warn", "envelope: ignore")
    other = tmp_path / "other.yaml"
    other.write_text(text)
    shutil.copytree(demo.parent / "videos", tmp_path / "videos")
    second = run_local(other, store=store)
    assert second.state == "done" and second.config_hash != first.config_hash
    assert second.spawner["calls"] == 5
    st = LocalStore(store)
    assert load_manifest(st, JOB, first.config_hash).config_hash == first.config_hash
    assert load_manifest(st, JOB).config_hash == second.config_hash  # the latest plan


def test_zero_modal_a_whole_job_runs_with_modal_unimportable(demo, tmp_path):
    """The CI guarantee: ``import modal`` raises in this process, and the job still completes."""
    code = (
        "import sys; sys.modules['modal'] = None\n"  # any `import modal` now raises ImportError
        "from flora_deploy.local import run_local\n"
        f"res = run_local({str(demo)!r}, store={str(tmp_path / 'store')!r},"
        " faults='preempt:segcap-*x1')\n"
        "assert res.state == 'done', res.as_dict()\n"
        "assert not [m for m in sys.modules if m.startswith('flora_deploy.modal_app')]\n"
        "print('OK', res.wall_s)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120, cwd=Path(__file__).resolve().parents[2])
    assert out.returncode == 0 and out.stdout.startswith("OK"), out.stderr[-2000:]
