"""Review-4 job/worker/factory seam findings, end to end where the bug lived between pieces.

* completion semantics: a job never reports ``done`` while its work silently failed (C1, C2,
  H1): the two critical 0-record cases run the real ``RewardShardRunner`` under the real
  ``Orchestrator`` + ``build_deps`` + compose;
* H2 rewards-only planning, H4 content-addressed shard ids, M1 per-stage keys, M6 per-config
  manifests, M3 compose under listing lag, M4 concurrent-input preemption.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import resources

import pytest
import yaml

from flora_deploy.config.schema import parse_config, to_run_settings
from flora_deploy.export.compose import ComposeError, compose
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.deps import StoreManifestSource, build_deps
from flora_deploy.jobs.manifest import (
    build_manifest,
    kinds_for,
    load_manifest,
    save_manifest,
    shard_id_for,
    stages_of,
)
from flora_deploy.jobs.orchestrator import AllowAll, DictLease, Orchestrator
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard, ShardReceipt, episode_uid
from flora_deploy.modal_app import factory as f
from flora_deploy.modal_app import workers_app as wa
from flora_deploy.records import Provenance, Segment, StageRecord
from flora_deploy.stages import reward as R
from flora_deploy.workers_impl.reward import RewardEngine, RewardShardRunner
from tests.fakes.jobs import FakeClock, FakeDict, TokenSpawner
from tests.fakes.modal import FakeStore
from tests.unit.test_reward_stage import FakeLLM, make_nv
from tests.unit.test_reward_stage import fake_vllm as fake_vllm  # noqa: F401  (autouse)
from tests.unit.test_reward_worker import InterruptAt

JOB, CFG, DS = "seam-job", "cfgseam1", "lerobot/seam"
PATHS = JobPaths(JOB)
KIND = "reward.terminal"
POOL = PoolSettings(gpu="H100!", max_containers=1)


def template(name: str) -> dict:
    return yaml.safe_load((resources.files("flora_deploy.config") / "templates" / name)
                          .read_text())


def receipt(eid: str, *, instruction="dataset_task", task="pick the cup", text=None) -> dict:
    """An index receipt as ``flora index`` writes it: dataset-relative video uris."""
    rec = {"schema": "flora_index_receipt_v1", "status": "indexed", "uid": episode_uid(DS, eid),
           "episode_id": eid, "dataset": {"name": DS}, "instruction": instruction, "task": task,
           "streams": {"main": {"source": {"uri": f"videos/{eid}xxxx.mp4", "start_s": None,
                                           "end_s": None}}},
           "envelope": {"ok": True, "reasons": []}}
    if text is not None:
        rec["instruction_text"] = text
    return rec


def reward_job(store, eids, *, stage_hashes=None, **receipt_kw) -> Manifest:
    refs = []
    for e in eids:
        u = episode_uid(DS, e)
        store.put_bytes(PATHS.index(u), json.dumps(receipt(e, **receipt_kw)).encode())
        refs.append(EpisodeRef(u, DS, e, "human", 4.0, True, True, PATHS.index(u)))
    uids = tuple(r.uid for r in refs)
    m = Manifest(JOB, CFG, tuple(refs), (Shard(shard_id_for(KIND, uids), KIND, uids, 10.0),),
                 "2026-10-21T00:00:00.000000Z", stage_hashes=dict(stage_hashes or {}))
    save_manifest(store, m)
    return m


def run_job(store, settings_for_container, *, tmp_path, max_ticks=300):
    """Real orchestrator + build_deps + the real RewardShardRunner behind a fake Modal."""
    clock = FakeClock()
    settings = RunSettings(job_id=JOB, preset="exact", config_hash=CFG, budget_usd=50.0,
                           max_gpus=2, pools={KIND: POOL})
    deps = build_deps(store, settings, budget=AllowAll(), clock=clock)

    def worker(job_id, shard_id, cfg, container, st, *, call_id, token):
        s = dataclasses.replace(settings_for_container(container), decoder="inline",
                                local_dir=str(tmp_path / container), container_id=container,
                                limits=R.RoundLimits(nvidia_group=1))
        runner = RewardShardRunner(st, RewardEngine(FakeLLM(), "PROC"), make_nv(), s,
                                   call_id=lambda: call_id, clock=clock)
        return runner.run_shard(job_id, shard_id, cfg, token)

    spawner = TokenSpawner(store, {KIND: worker}, pools=settings.pools, python_retries=2)
    orch = Orchestrator(settings=settings, store=store, spawner=spawner,
                        lease=DictLease(FakeDict(), f"orchestrator/{JOB}", clock=clock),
                        manifests=deps.manifests, attempts=deps.attempts,
                        progress=deps.progress, budget=deps.budget, clock=clock)

    def sleep(dt):
        clock.advance(dt)
        spawner.tick()

    return orch.run_until_idle(sleep=sleep, poll_interval_s=15.0, max_ticks=max_ticks)


def rewards_plan(**instruction):
    d = template("rewards_only.yaml")
    d["pipeline"]["stages"] = [KIND]
    d["resources"] = {KIND: {"gpu": "H100!", "max_containers": 1}}
    if instruction:
        d["pipeline"]["episode_instruction"] = instruction
    return f.plan_app(parse_config(d))


# ---------------------------------------------------------------------------------------------
# completion semantics: the two critical 0-record cases


def test_c1_no_media_root_fails_the_job_never_done_with_zero_records(tmp_path):
    """Before: every episode an ``infra`` EpisodeError in a receipt, job ``done``, 0 records,
    ``flora results`` empty with exit 0. Now: no receipt, bounded retries, ``failed``."""
    store = FakeStore()
    m = reward_job(store, ["e0", "e1", "e2"])
    env = dict(rewards_plan().object("RewardWorker").env)
    env.pop("FLORA_MEDIA_ROOT")  # the old factory: no media root in the worker env
    out = run_job(store, lambda c: wa.reward_settings_from_env(env), tmp_path=tmp_path)
    assert out.result == "failed"
    jf = A.JobFiles(store, JOB, CFG)
    (shard,) = m.shards
    assert P.read_receipt(jf, KIND, shard.shard_id) is None
    prog = P.derive(jf, m, limit=3)
    assert prog.shards[shard.shard_id].state == "failed" and not prog.complete
    assert all(st == "pending" for st in prog.episodes.values())  # nothing excused
    with pytest.raises(ComposeError, match="job failed"):
        compose(store, JOB)
    errs = [e for p in store.list(PATHS.area_prefix(CFG, "errors", KIND))
            for e in P.read_errors(jf, p)]
    assert errs and {e.kind for e in errs} == {"infra"}
    assert "no media_root" in errs[0].message
    # and the factory now sets it: the worker env carries the media root
    assert rewards_plan().object("RewardWorker").env["FLORA_MEDIA_ROOT"]


def test_c1_with_media_root_the_same_job_completes(tmp_path):
    store = FakeStore()
    m = reward_job(store, ["e0", "e1", "e2"])
    env = {**rewards_plan().object("RewardWorker").env, "FLORA_MEDIA_ROOT": str(tmp_path)}
    out = run_job(store, lambda c: wa.reward_settings_from_env(env), tmp_path=tmp_path)
    assert out.result == "done"
    s = compose(store, JOB).summary
    assert s["status"] == "complete" and s["stages"][KIND]["records"] == 3
    assert len(m.episodes) == 3


def test_c2_fixed_instruction_reaches_the_worker(tmp_path):
    """Before: ``fixed`` receipts had no task and the worker no default: every episode an
    ``nv_failure`` "no task instruction", ``done`` with 0 records. Now the factory passes the
    fixed text (``FLORA_INSTRUCTION_TEXT``) and new receipts carry ``instruction_text``."""
    store = FakeStore()
    reward_job(store, ["e0", "e1"], instruction="fixed", task=None)  # an old-style receipt
    plan = rewards_plan(source="fixed", text="Fold the towel.")
    env = {**plan.object("RewardWorker").env, "FLORA_MEDIA_ROOT": str(tmp_path)}
    assert env["FLORA_INSTRUCTION_SOURCE"] == "fixed"
    assert env["FLORA_INSTRUCTION_TEXT"] == "Fold the towel."
    assert wa.reward_settings_from_env(env).default_task == "Fold the towel."
    assert run_job(store, lambda c: wa.reward_settings_from_env(env),
                   tmp_path=tmp_path).result == "done"
    s = compose(store, JOB).summary
    assert s["status"] == "complete" and s["stages"][KIND]["records"] == 2

    # without the env and without instruction_text: a config error, the job fails (not done)
    store2 = FakeStore()
    reward_job(store2, ["e0"], instruction="fixed", task=None)
    env2 = {k: v for k, v in env.items() if k != "FLORA_INSTRUCTION_TEXT"}
    assert run_job(store2, lambda c: wa.reward_settings_from_env(env2),
                   tmp_path=tmp_path / "b").result == "failed"


def test_c2_new_receipts_carry_the_resolved_instruction(tmp_path):
    from flora_deploy.jobs.indexing import IndexJob

    cfg = template("rewards_only.yaml")
    cfg["pipeline"]["episode_instruction"] = {"source": "fixed", "text": "Fold."}
    job = IndexJob.from_config(parse_config(cfg), JOB)
    assert job.instruction_for("dataset says") == "Fold."
    store = FakeStore()
    reward_job(store, ["e0"], instruction="fixed", task="dataset says", text="Fold.")
    env = {**rewards_plan().object("RewardWorker").env, "FLORA_MEDIA_ROOT": str(tmp_path)}
    assert run_job(store, lambda c: wa.reward_settings_from_env(env),
                   tmp_path=tmp_path).result == "done"


def test_none_instruction_with_rewards_is_refused_at_config_time():
    from flora_deploy.config.schema import ConfigError

    d = template("rewards_only.yaml")
    d["pipeline"]["episode_instruction"] = {"source": "none"}
    with pytest.raises(ConfigError, match="need a task instruction"):
        parse_config(d)
    d = template("exact_mp4_dir.yaml")
    d["pipeline"]["episode_instruction"] = {"source": "none"}
    parse_config(d)  # segment/caption only: fine


def test_stage_infra_error_is_retried_not_excused(tmp_path):
    """An OSError from NVIDIA's Collage (a volume read) is ``infra``: the job fails after its
    retries instead of finishing without the episode (H1)."""
    store = FakeStore()
    reward_job(store, ["e0"])
    store.put_bytes(PATHS.index(episode_uid(DS, "e0")), json.dumps(
        {**receipt("e0"), "streams": {"main": {"source": {"uri": "videos/oserr.mp4"}}}}).encode())
    env = {**rewards_plan().object("RewardWorker").env, "FLORA_MEDIA_ROOT": str(tmp_path)}
    assert run_job(store, lambda c: wa.reward_settings_from_env(env),
                   tmp_path=tmp_path).result == "failed"


def test_error_rate_guard_completed_with_errors(tmp_path):
    """NVIDIA-semantics failures are legitimate exclusions, but a job that excludes most of a
    stage is ``completed_with_errors`` (orchestrator, reconciler and compose agree)."""
    from flora_deploy.jobs.reconciler import reconcile

    store = FakeStore()
    reward_job(store, ["e0", "bad1", "bad2"])  # "bad*" videos: NVIDIA's Collage rejects them
    for e in ("bad1", "bad2"):
        store.put_bytes(PATHS.index(episode_uid(DS, e)), json.dumps(
            {**receipt(e), "streams": {"main": {"source": {"uri": f"videos/{e}.mp4"}}}}).encode())
    env = {**rewards_plan().object("RewardWorker").env, "FLORA_MEDIA_ROOT": str(tmp_path)}
    out = run_job(store, lambda c: wa.reward_settings_from_env(env), tmp_path=tmp_path)
    assert out.result == "completed_with_errors"
    s = compose(store, JOB).summary
    assert s["status"] == "completed_with_errors" and not s["complete"]
    assert s["problems"] == [f"{KIND}: 2 of 3 episodes excluded (67% > 10%)"]
    settings = RunSettings(job_id=JOB, preset="exact", config_hash=CFG, budget_usd=50.0,
                           max_gpus=2, pools={KIND: POOL})
    d = build_deps(store, settings, budget=AllowAll())
    st = reconcile(job_id=JOB, config_hash=CFG, store=store, manifests=d.manifests,
                   attempts=d.attempts)
    assert st.state == "completed_with_errors"


def test_flora_results_refuses_zero_records_for_planned_work(tmp_path):
    from typer.testing import CliRunner

    from flora_deploy.cli import app
    from flora_deploy.jobs.stores import LocalStore

    class Ticking(LocalStore):
        def tick(self):
            pass

    store = Ticking(tmp_path / "store")
    reward_job(store, ["bad1"])
    store.put_bytes(PATHS.index(episode_uid(DS, "bad1")), json.dumps(
        {**receipt("bad1"), "streams": {"main": {"source": {"uri": "videos/bad1.mp4"}}}}).encode())
    env = {**rewards_plan().object("RewardWorker").env, "FLORA_MEDIA_ROOT": str(tmp_path)}
    assert run_job(store, lambda c: wa.reward_settings_from_env(env),
                   tmp_path=tmp_path).result == "completed_with_errors"
    uri = f"file://{tmp_path / 'store'}"
    r = CliRunner().invoke(app, ["compose", "--job", JOB, "--store", uri])
    assert r.exit_code == 3 and "completed_with_errors" in r.output
    r = CliRunner().invoke(app, ["results", "--job", JOB, "--stage", KIND, "--store", uri,
                                 "--format", "jsonl"])
    assert r.exit_code == 1 and "0 records for 1 planned" in r.output


# ---------------------------------------------------------------------------------------------
# H2: rewards-only templates plan only the requested stages


@pytest.mark.parametrize("name", ["rewards_only.yaml", "exact_mp4_dir.yaml",
                                  "exact_lerobot.yaml"])
def test_templates_build_the_full_plan(name):
    """config -> app plan -> RunSettings -> manifest -> one orchestrator tick, per template."""
    from tests.unit.test_jobs_manifest import random_indexes

    cfg = parse_config(template(name))
    plan = f.plan_app(cfg, derived_receipts={"captioner_bf16": "0" * 64})
    rs = to_run_settings(cfg, JOB, derived_receipts={"captioner_bf16": "0" * 64})
    assert rs.stage_hashes == plan.stage_hashes and rs.config_hash == plan.config_hash
    import random

    m = build_manifest(random_indexes(random.Random(0), 6), rs, created_utc="t")
    assert {s.kind for s in m.shards} == set(rs.pools) == set(plan.worker_classes)
    assert m.stage_hashes == {k: plan.stage_hashes[k] for k in rs.pools}
    for o in plan.objects:
        if o.kind == "cls":
            assert o.env["FLORA_MEDIA_ROOT"], o.name
    store = FakeStore()
    save_manifest(store, m)
    deps = build_deps(store, rs, budget=AllowAll())
    spawner = TokenSpawner(store, {}, pools=rs.pools)
    orch = Orchestrator(settings=rs, store=store, spawner=spawner,
                        lease=DictLease(FakeDict(), "o"), manifests=deps.manifests,
                        attempts=deps.attempts, progress=deps.progress, budget=deps.budget)
    assert orch.acquire() and orch.tick() == "running"
    assert set(spawner.spawns) <= {s.shard_id for s in m.shards} and spawner.spawns


def test_rewards_only_kinds_come_from_pools():
    rs = RunSettings(job_id=JOB, preset="exact", config_hash=CFG, budget_usd=1.0, max_gpus=2,
                     pools={"reward.terminal": POOL, "reward.dense": POOL})
    assert kinds_for(rs) == ("reward.terminal", "reward.dense")


# ---------------------------------------------------------------------------------------------
# H4: content-addressed shard ids


def test_replan_with_new_episodes_never_reuses_old_receipts():
    import random

    from tests.unit.test_jobs_manifest import make_settings, random_indexes

    idx = random_indexes(random.Random(3), 6, files=6)
    settings = make_settings("exact", rewards=(), target_min=1e6)  # one shard per kind
    a = build_manifest(idx[::2], settings, created_utc="t")
    b = build_manifest(idx, settings, created_utc="t")
    (sa,), (sb,) = a.shards, b.shards
    assert sa.shard_id != sb.shard_id  # different episodes, different id
    assert build_manifest(idx[::2], settings, created_utc="u").shards[0].shard_id == sa.shard_id
    assert sa.shard_id == shard_id_for(sa.kind, reversed(sa.episodes))  # order-free digest
    store = FakeStore()
    save_manifest(store, a)
    jf = A.JobFiles(store, a.job_id, a.config_hash)
    P.write_receipt(jf, ShardReceipt(sa.shard_id, sa.kind, jf.key(sa.kind), "c", sa.episodes,
                                     (), (), None, 1.0, "t"))
    assert P.derive(jf, a, limit=3).complete
    save_manifest(store, b)  # re-plan under the same config hash after a re-index
    prog = P.derive(A.JobFiles(store, b.job_id, b.config_hash), b, limit=3)
    assert prog.shards[sb.shard_id].state == "ready" and not prog.complete


# ---------------------------------------------------------------------------------------------
# M1 / M6: per-stage keys, per-config manifests


def _record(uid: str, stage: str, cfg: str) -> StageRecord:
    return StageRecord(uid=uid, dataset=DS, episode_id=uid.split("--")[-1],
                       stage=stage,  # type: ignore[arg-type]
                       provenance=Provenance(models={"m": "1"}, engine="e", preset="exact",
                                             config_hash=cfg, logic_version="v",
                                             framework="t"),
                       certified=True, segments=(Segment(f"{uid}:0", 0.0, 1.0, caption="c"),))


def test_adding_a_reward_stage_reuses_finished_segcap_shards():
    """Config A = segcap; config B = segcap + reward.terminal with the same segcap stage hash:
    B's segcap shards are already done (A's receipts), only the reward shard runs."""
    store = FakeStore()
    uids = tuple(episode_uid(DS, e) for e in ("a", "b"))
    refs = tuple(EpisodeRef(u, DS, u[-1], "human", 1.0, True, True, PATHS.index(u))
                 for u in uids)
    seg = Shard(shard_id_for("segcap", uids), "segcap", uids, 1.0)
    rew = Shard(shard_id_for(KIND, uids), KIND, uids, 1.0)
    ma = Manifest(JOB, "jobA", refs, (seg,), "t", stage_hashes={"segcap": "stSEG"})
    mb = Manifest(JOB, "jobB", refs, (seg, rew), "t",
                  stage_hashes={"segcap": "stSEG", KIND: "stREWb"})
    save_manifest(store, ma)
    save_manifest(store, mb)
    assert load_manifest(store, JOB, "jobA") == ma  # M6: B's plan did not clobber A's
    assert StoreManifestSource(A.JobFiles(store, JOB, "jobA")).base() == ma
    ja = A.JobFiles(store, JOB, "jobA")
    assert ja.key("segcap") == "stSEG"
    P.write_part(ja, "segcap", seg.shard_id, "c1", 0,
                 [_record(u, st, ja.key("segcap")) for u in uids for st in stages_of("segcap")])
    P.write_receipt(ja, ShardReceipt(seg.shard_id, "segcap", "stSEG", "c1", uids, (),
                                     (ja.paths.part("jobA", "segcap", seg.shard_id, "c1", 0),),
                                     None, 1.0, "t"))
    assert ja.paths.receipt("jobA", "segcap", seg.shard_id).startswith(
        f"flora/{JOB}/stSEG/receipts/")
    assert ja.paths.call("jobA", "segcap", seg.shard_id).startswith(f"flora/{JOB}/jobA/calls/")
    jb = A.JobFiles(store, JOB, "jobB")  # stage hashes read from B's manifest
    prog = P.derive(jb, mb, limit=3)
    assert prog.shards[seg.shard_id].state == "done"
    assert prog.shards[rew.shard_id].state == "ready"
    ci = P.compose_input(jb, prog)
    assert {k for k in ci.records} == {(u, st) for u in uids for st in ("segment", "caption")}
    assert ci.foreign == 0

    # the orchestrator of B spawns only the reward shard
    rs = RunSettings(job_id=JOB, preset="exact", config_hash="jobB", budget_usd=10.0,
                     max_gpus=4, pools={"segcap": POOL, KIND: POOL})
    deps = build_deps(store, rs, budget=AllowAll())
    spawner = TokenSpawner(store, {}, pools=rs.pools)
    orch = Orchestrator(settings=rs, store=store, spawner=spawner, lease=DictLease(FakeDict(), "o"),
                        manifests=deps.manifests, attempts=deps.attempts,
                        progress=deps.progress, budget=deps.budget)
    assert orch.acquire() and orch.tick() == "running"
    assert spawner.spawns == [rew.shard_id]


def test_poisoned_shard_under_another_config_is_not_respawned():
    store = FakeStore()
    u = episode_uid(DS, "p")
    refs = (EpisodeRef(u, DS, "p", "human", 1.0, True, True, PATHS.index(u)),)
    shard = Shard(shard_id_for("segcap", (u,)), "segcap", (u,), 1.0)
    save_manifest(store, Manifest(JOB, "jobA", refs, (shard,), "t", stage_hashes={"segcap": "S"}))
    A.write_poison(A.JobFiles(store, JOB, "jobA"), shard, message="poison (test)")
    save_manifest(store, Manifest(JOB, "jobB", refs, (shard,), "t", stage_hashes={"segcap": "S"}))
    rs = RunSettings(job_id=JOB, preset="exact", config_hash="jobB", budget_usd=10.0,
                     max_gpus=4, pools={"segcap": POOL}, max_episode_error_rate=1.0)
    deps = build_deps(store, rs, budget=AllowAll())
    spawner = TokenSpawner(store, {}, pools=rs.pools)
    orch = Orchestrator(settings=rs, store=store, spawner=spawner, lease=DictLease(FakeDict(), "o"),
                        manifests=deps.manifests, attempts=deps.attempts,
                        progress=deps.progress, budget=deps.budget)
    # nothing is spawned; every episode of the stage excluded: completed_with_errors
    assert orch.acquire() and orch.tick() == "completed_with_errors" and spawner.spawns == []


# ---------------------------------------------------------------------------------------------
# M3: compose a poison-pilled job while the quarantine record is not listed yet


def test_compose_under_listing_lag_after_poison_pill():
    store = FakeStore(lag=5)
    uids = tuple(episode_uid(DS, e) for e in ("ok", "poison"))
    refs = tuple(EpisodeRef(u, DS, u[-2:], "human", 1.0, True, True, PATHS.index(u))
                 for u in uids)
    ok = Shard(shard_id_for(KIND, uids[:1]), KIND, uids[:1], 1.0)
    bad = Shard(shard_id_for(KIND, uids[1:]), KIND, uids[1:], 1.0)
    m = Manifest(JOB, CFG, refs, (ok, bad), "t")
    save_manifest(store, m)
    jf = A.JobFiles(store, JOB, CFG)
    part = P.write_part(jf, KIND, ok.shard_id, "c", 0, [_record(uids[0], KIND, CFG)])
    P.write_receipt(jf, ShardReceipt(ok.shard_id, KIND, CFG, "c", uids[:1], (), (part,), None,
                                     1.0, "t"))
    A.quarantine(jf, bad, created_utc="2026-10-21T00:00:00.000000Z", starts=3)
    prog = P.derive(jf, m, limit=3)  # the record is not listed yet (lag)
    assert not P.quarantine_records(jf)
    assert prog.complete and prog.episodes[(uids[1], KIND)] == "poison_pill"
    s = compose(store, JOB, max_error_rate=1.0).summary
    assert s["status"] == "complete" and s["poison_pills"] == [f"{uids[1]}/{KIND}"]


# ---------------------------------------------------------------------------------------------
# M4: @modal.exit with concurrent inputs marks every in-flight shard


def test_preempt_marks_every_concurrent_input(tmp_path):
    import threading

    from tests.unit.test_reward_worker import ep, make_runner, setup, tshard

    store = FakeStore()
    s1, s2 = tshard(["a"], sid="reward.terminal-a"), tshard(["b"], sid="reward.terminal-b")
    setup(store, [ep("a"), ep("b")], [s1, s2])
    gate, release = threading.Barrier(3), threading.Event()

    class Hold(FakeLLM):
        def generate(self, requests, params, use_tqdm=True):
            gate.wait(timeout=5)
            release.wait(timeout=5)
            raise KeyboardInterrupt("SIGINT")

    runner = make_runner(store, tmp_path, llm=Hold())
    errors = []

    def go(sid):
        try:
            runner.run_shard("job-r", sid, "cfg-r")
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(s.shard_id,)) for s in (s1, s2)]
    for t in threads:
        t.start()
    gate.wait(timeout=5)  # both inputs are in flight
    assert runner.preempt()  # @modal.exit
    release.set()
    for t in threads:
        t.join(timeout=5)
    assert not errors
    jf = A.JobFiles(store, "job-r", "cfg-r")
    for s in (s1, s2):
        m = A.read_attempt(jf, s.kind, s.shard_id, "c1")
        assert m is not None and m.preempted, s.shard_id
    assert InterruptAt  # imported helper stays shared with test_reward_worker


# ---------------------------------------------------------------------------------------------
# C1 / H3 / M5 / M7: factory and worker startup


def test_media_root_from_input(monkeypatch):
    lerobot = parse_config(template("exact_lerobot.yaml"))
    plan = f.plan_app(lerobot, derived_receipts={"captioner_bf16": "0" * 64})
    root = plan.object("RewardWorker").env["FLORA_MEDIA_ROOT"]
    assert root == ("hf://datasets/griffinlabs/EgoDex-LeRobot-v3.0@"
                    "41d60b449629b2181ff5b735d31c2a2cf8b3cad8/test/make_sandwich")
    staged = template("exact_lerobot.yaml")
    staged["input"]["stage_to"] = "volume://flora-stage/hub"
    plan = f.plan_app(parse_config(staged), derived_receipts={"captioner_bf16": "0" * 64})
    w = plan.object("RewardWorker")
    assert w.env["FLORA_MEDIA_ROOT"] == (
        "/media/flora-stage/hub/griffinlabs--EgoDex-LeRobot-v3.0@41d60b449629/test/make_sandwich")
    assert w.volumes["/media/flora-stage"] == f.VolumeMount("flora-stage", read_only=True)
    same = parse_config(template("exact_mp4_dir.yaml"))  # staged on the store's own Volume
    w = f.plan_app(same, derived_receipts={"captioner_bf16": "0" * 64}).object("ReferenceWorker")
    assert w.env["FLORA_MEDIA_ROOT"] == "/vol/flora-results/inputs/my-mp4-labels"
    local = template("rewards_only.yaml")
    local["input"]["stage_to"] = None
    with pytest.raises(f.DeployError, match="local directory"):
        f.plan_app(parse_config(local))
    local["input"]["stage_to"] = "s3://bucket/x"
    with pytest.raises(f.DeployError, match="Modal Volume"):
        f.plan_app(parse_config(local))


def test_preflight_fails_before_the_engine_build(tmp_path, monkeypatch):
    from flora_deploy.workers_impl.reward import PreflightError

    built = []
    monkeypatch.setattr(wa, "build_reward_engine", lambda **kw: built.append(kw))
    env = {"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": str(tmp_path)}
    monkeypatch.setattr(wa.os, "environ", env)
    worker = wa.RewardWorker.__new__(wa.RewardWorker)
    start = wa.RewardWorker.__dict__["start"]
    raw = start._get_raw_f() if hasattr(start, "_get_raw_f") else start.raw_f
    with pytest.raises(PreflightError, match="FLORA_MEDIA_ROOT is not set"):
        raw(worker)
    env["FLORA_MEDIA_ROOT"] = str(tmp_path / "missing")
    with pytest.raises(PreflightError, match="does not exist"):
        raw(worker)
    assert built == []  # the GPU engine was never built
    env["FLORA_MEDIA_ROOT"] = "hf://datasets/o/n@" + "a" * 40
    facts = wa.preflight(env, hub_check=lambda repo, rev: None)
    assert facts["media"] == "hub o/n@" + "a" * 40
    with pytest.raises(PreflightError, match="not reachable"):
        wa.preflight(env, hub_check=lambda repo, rev: (_ for _ in ()).throw(OSError("401")))
    env["FLORA_MEDIA_ROOT"] = str(tmp_path)
    env["FLORA_INSTRUCTION_SOURCE"] = "fixed"
    with pytest.raises(PreflightError, match="FLORA_INSTRUCTION_TEXT"):
        wa.preflight(env)


class _S3:
    def __init__(self, fail=None):
        self.objects, self.fail = {}, fail

    def head_object(self, Bucket, Key):
        if self.fail:
            raise self.fail
        if Key not in self.objects:
            raise type("E", (Exception,), {"response": {"Error": {"Code": "404"}}})()
        return {}

    def list_objects_v2(self, **kw):
        if self.fail:
            raise self.fail
        return {"Contents": []}


def test_bucket_store_in_workers_and_its_preflight(tmp_path):
    from flora_deploy.jobs.stores import BucketStore
    from flora_deploy.workers_impl.reward import PreflightError

    env = {"FLORA_STORE": "s3://b/p", "FLORA_STORE_KIND": "bucket", "FLORA_STORE_BUCKET": "b",
           "FLORA_STORE_PREFIX": "p", "FLORA_MEDIA_ROOT": str(tmp_path)}
    st = wa.store_from_env(env, client=_S3())
    assert isinstance(st, BucketStore) and st.bucket == "b" and st.prefix == "p/"
    assert wa.preflight(env, store=st)["store"] == "s3://b/p"
    bad = wa.store_from_env(env, client=_S3(fail=PermissionError("AccessDenied")))
    with pytest.raises(PreflightError, match="not usable"):
        wa.preflight(env, store=bad)
    from flora_deploy.modal_app.images import CPU_PACKAGES, VLLM_PACKAGES

    assert any(p.startswith("boto3==") for p in CPU_PACKAGES)
    assert any(p.startswith("boto3==") for p in VLLM_PACKAGES)
    lock = (resources.files("flora_deploy.modal_app") / "locks" / "vllm.txt").read_text()
    assert "\nboto3==" in lock


def test_worker_timeout_from_rates():
    """M5: a full 10-minute exact segcap shard (200 video-s at the 3.0 placeholder) takes about
    1724 s + startup at the measured 8.62 container-s/video-s: the timeout must exceed that."""
    t = f.worker_timeout_s(10.0, preset="exact", kinds=("segcap",), max_inputs=1)
    assert t > 1724 + 213
    assert f.worker_timeout_s(10.0) == 1800  # legacy: no kinds
    cfg = parse_config(template("exact_mp4_dir.yaml"))
    plan = f.plan_app(cfg, derived_receipts={"captioner_bf16": "0" * 64})
    assert plan.object("ReferenceWorker").timeout == t


def test_cache_is_read_only_on_workers_and_rw_on_the_warmup():
    from flora_deploy.layout import CACHE_MOUNT

    plan = rewards_plan()
    w = plan.object("RewardWorker")
    assert w.volumes[str(CACHE_MOUNT)].read_only and w.env["FLORA_CACHE_MODE"] == "ro"
    assert w.env["VLLM_CACHE_ROOT"].startswith(f.LOCAL_CACHE_ROOT)
    warm = plan.object("warm_reward_cache")
    assert not warm.volumes[str(CACHE_MOUNT)].read_only and warm.env["FLORA_CACHE_MODE"] == "rw"
    assert warm.max_containers == 1 and warm.gpu == w.gpu
    assert warm.env["VLLM_CACHE_ROOT"].startswith(str(CACHE_MOUNT))
    fns, _classes = f.registered(f.build_app(plan))
    assert "warm_reward_cache" in fns


def test_cache_env_ro_copies_the_seed(tmp_path):
    seed = tmp_path / "mount"
    vllm_dir = wa.ra.cache_env(seed)["VLLM_CACHE_ROOT"]
    (tmp_path / "mount").mkdir()
    import pathlib

    pathlib.Path(vllm_dir).mkdir(parents=True)
    (pathlib.Path(vllm_dir) / "k.bin").write_bytes(b"x")
    env = wa.cache_env_for("ro", mount=seed, local=tmp_path / "local")
    assert pathlib.Path(env["VLLM_CACHE_ROOT"], "k.bin").read_bytes() == b"x"
    assert env["VLLM_CACHE_ROOT"].startswith(str(tmp_path / "local"))


# ---------------------------------------------------------------------------------------------
# budget wiring (M2)


def test_build_deps_uses_the_real_gate_with_deployment_facts():
    from flora_deploy.jobs.budget import (
        REGION_MULTIPLIER_BROAD,
        CallRecord,
        ProjectionBudgetGate,
        calls_cost,
        pinned_rates,
    )

    rs = RunSettings(job_id=JOB, preset="exact", config_hash=CFG, budget_usd=10.0, max_gpus=2,
                     pools={KIND: POOL}, region="us-east", scaledown_s=300, min_containers=1)
    store = FakeStore()
    reward_job(store, ["e0"])
    gate = build_deps(store, rs).budget
    assert isinstance(gate, ProjectionBudgetGate)
    assert gate.scaledown_s == 300 and gate.rates.region_multiplier == REGION_MULTIPLIER_BROAD
    assert gate.estimator.video_seconds(load_manifest(store, JOB).shards[0]) == 4.0
    r = pinned_rates()
    per_s = r.pool_per_second(POOL)
    one = (CallRecord("c", "s", KIND, 0.0, 100.0, "success"),)
    assert calls_cost(one, rs, r, 200.0, scaledown_s=300) == pytest.approx((100 + 300) * per_s)
    # a warm container keeps billing after the call ended
    assert calls_cost(one, rs, r, 1000.0, scaledown_s=300, warm=1) == pytest.approx(
        1000.0 * per_s)
    assert gate.admit(load_manifest(store, JOB).shards[0], {}).allow
