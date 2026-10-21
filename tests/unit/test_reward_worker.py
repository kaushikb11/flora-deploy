"""RewardWorker core (``workers_impl.reward``): the ``workers.ShardWorker`` contract, steps 0-6,
with a fake NVIDIA core / LLM (``test_reward_stage``) and ``FakeStore`` listing lag.

The contract is also checked by ``tests/conformance`` (``registry.py`` registers the runner).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.manifest import save_manifest
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard, episode_uid
from flora_deploy.media.types import SourceRef
from flora_deploy.stages import reward
from flora_deploy.workers_impl.reward import (
    RewardEngine,
    RewardShardRunner,
    RewardWorkerSettings,
    VideoResolver,
    collage_order,
    mode_for,
    parse_index_receipt,
)
from tests.fakes.modal import FakeStore
from tests.unit.test_reward_stage import FakeLLM, make_nv
from tests.unit.test_reward_stage import fake_vllm as fake_vllm  # noqa: F401  (autouse)

JOB, CFG, DS = "job-r", "cfg-r", "lerobot/ds"
PATHS = JobPaths(JOB)
T0 = 1_800_000_000.0


def uid(eid: str) -> str:
    return episode_uid(DS, eid)


def ep(eid, task="task", cams=None, *, write_index=True, envelope_ok=True):
    """(EpisodeRef, index receipt dict | None). ``cams``: role -> uri (or SourceRef)."""
    cams = cams or {"main": f"/v/{eid}{'m' * 4}"}
    streams = {}
    for role, src in cams.items():
        src = src if isinstance(src, SourceRef) else SourceRef(src)
        streams[role] = {"source": {"uri": src.uri, "start_s": src.start_s, "end_s": src.end_s},
                         "rate": "10", "pts_us": ["0"], "duration_us": 4_000_000}
    receipt = {"episode_id": eid, "dataset": DS, "task": task, "streams": streams,
               "envelope": {"ok": envelope_ok, "reasons": []}}
    ref = EpisodeRef(uid(eid), DS, eid, "human", 4.0, True, envelope_ok,
                     PATHS.index(uid(eid)))
    return ref, (receipt if write_index else None)


def setup(store, episodes, shards):
    for ref, receipt in episodes:
        if receipt is not None:
            store.put_bytes(ref.index_ref, json.dumps(receipt).encode())
    m = Manifest(JOB, CFG, tuple(r for r, _ in episodes), tuple(shards),
                 "2026-10-21T00:00:00.000000Z")
    save_manifest(store, m)
    return m


def tshard(eids, sid="reward.terminal-00000", kind="reward.terminal", **kw):
    return Shard(sid, kind, tuple(uid(e) for e in eids), 1.0, **kw)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        self.t += 1.0
        return self.t


def make_runner(store, tmp_path, cid="c1", llm=None, group=1, **kw):
    settings = RewardWorkerSettings(
        decoder="inline", local_dir=str(tmp_path / cid), container_id=cid, gpu="H100",
        models={"rewarder": "7d"}, limits=reward.RoundLimits(nvidia_group=group), **kw)
    return RewardShardRunner(store, RewardEngine(llm or FakeLLM(), "PROC"), make_nv(), settings,
                             call_id=lambda: f"fc-{cid}", clock=Clock())


def jf_of(store):
    return A.JobFiles(store, JOB, CFG)


def prompts(llm):
    return [r["prompt"] for requests, _p, _t in llm.calls for r in requests]


def all_errors(store, shard):
    jf = jf_of(store)
    prefix = PATHS.errors(CFG, shard.kind, shard.shard_id, "x", 0)[:-len("x-0.jsonl")]
    return [e for p in store.list(prefix) for e in P.read_errors(jf, p)]


class InterruptAt(FakeLLM):
    """Raises ``exc`` on the ``n``-th ``generate`` call (1-based), optionally calling ``hook``."""

    def __init__(self, n, exc=KeyboardInterrupt, hook=None, **kw):
        super().__init__(**kw)
        self.n, self.exc, self.hook, self.armed = n, exc, hook, True

    def generate(self, requests, params, use_tqdm=True):
        if self.armed and len(self.calls) + 1 == self.n:
            self.armed = False
            self.calls.append((requests, params, use_tqdm))
            if self.hook:
                self.hook()
            raise self.exc()
        return super().generate(requests, params, use_tqdm)


# --- helpers ---------------------------------------------------------------------------------

def test_modes_collage_order_and_index_receipt():
    assert mode_for("reward.terminal") == "terminal"
    assert mode_for("reward.terminal", "terminal_rationale") == "terminal_rationale"
    assert mode_for("reward.dense") == "dense"
    with pytest.raises(NotImplementedError):
        mode_for("reward.rubric")
    with pytest.raises(ValueError):
        mode_for("segment")
    assert collage_order({"main", "left_wrist", "right_wrist"}) == (
        "left_wrist", "main", "right_wrist")
    assert collage_order({"right_wrist", "main"}) == ("main", "right_wrist")
    assert collage_order({"main"}) == ("main",)
    _ref, receipt = ep("a", cams={"main": SourceRef("/f.mp4", 1.0, 2.5)})
    idx = parse_index_receipt(json.dumps(receipt))
    assert idx.task == "task" and idx.envelope_ok is True
    assert idx.sources == {"main": SourceRef("/f.mp4", 1.0, 2.5)}


def test_video_resolver_relative_and_range_cut(tmp_path):
    cuts = []

    def cutter(src, dest):
        cuts.append(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return dest

    r = VideoResolver(media_root="/data", cut_dir=tmp_path, cutter=cutter)
    assert r.resolve("u", "main", SourceRef("x/a.mp4")) == "/data/x/a.mp4"
    assert r.resolve("u", "main", SourceRef("/abs.mp4")) == "/abs.mp4"
    got = r.resolve("u", "main", SourceRef("x/v3.mp4", 2.0, 5.0))
    assert got == str(tmp_path / "u" / "main.mp4")
    assert r.resolve("u", "main", SourceRef("x/v3.mp4", 2.0, 5.0)) == got  # cached
    assert cuts == [SourceRef("/data/x/v3.mp4", 2.0, 5.0)]
    with pytest.raises(FileNotFoundError):
        VideoResolver().resolve("u", "main", SourceRef("rel.mp4"))


# --- steps 1, 4, 6: a full shard -------------------------------------------------------------

@pytest.mark.parametrize("lag", [0, 3])
def test_terminal_shard_records_parts_and_receipt_last(tmp_path, lag):
    store = FakeStore(lag=lag)
    eps = [ep("a"), ep("b"), ep("c")]
    shard = tshard(["a", "b", "c"])
    m = setup(store, eps, [shard])
    runner = make_runner(store, tmp_path, group=2)
    token = A.spawn_token(JOB, CFG, shard.shard_id, 1)

    rec = runner.run_shard(JOB, shard.shard_id, CFG, token=token)

    jf = jf_of(store)
    assert rec.episodes_done == shard.episodes and rec.episodes_failed == ()
    assert rec.kind == "reward.terminal" and rec.config_hash == CFG and rec.gpu == "H100"
    assert rec.container_id == "c1"
    # two NVIDIA-aligned rounds (group 2): one whole part per round, k = 0, 1
    assert rec.parts == tuple(sorted(PATHS.part(CFG, shard.kind, shard.shard_id, "c1", k)
                                     for k in (0, 1)))
    assert store.writes[-1] == PATHS.receipt(CFG, shard.kind, shard.shard_id)  # LAST
    assert P.read_receipt(jf, shard.kind, shard.shard_id) == rec
    marker = A.read_attempt(jf, shard.kind, shard.shard_id, "c1")
    assert marker.token == token and marker.call_id == "fc-c1"
    assert marker.orchestrator_attempt == 1 and not marker.preempted and not marker.error
    records = [r for p in rec.parts for r in P.read_part(jf, p)]
    assert [r.uid for r in records] == list(shard.episodes)
    assert all(r.stage == "reward.terminal" and r.provenance.config_hash == CFG
               and r.certified and r.terminal is not None for r in records)
    assert records[0].provenance.source["main"]["uri"] == "/v/ammmm"  # role-keyed
    audits = [json.loads(line) for k in (0, 1) for line in store.get_bytes(
        PATHS.audit(CFG, shard.kind, shard.shard_id, "c1", k)).decode().splitlines()]
    assert [a["uid"] for a in audits] == list(shard.episodes)
    assert all(a["job_config_hash"] == CFG and len(a["config_hash"]) == 64 for a in audits)
    for _ in range(lag):
        store.tick()
    prog = P.derive(jf, m, limit=3)
    assert prog.shards[shard.shard_id].state == "done" and prog.complete
    ci = P.compose_input(jf, prog)
    assert not ci.missing and not ci.conflicts and len(ci.records) == 3


def test_dense_shard_three_views_in_collage_order(tmp_path):
    store = FakeStore()
    cams = {"main": "/v/mmmmmm", "right_wrist": "/v/rrrrrr", "left_wrist": "/v/llllll"}
    shard = tshard(["d"], sid="reward.dense-00000", kind="reward.dense")
    setup(store, [ep("d", cams=cams)], [shard])
    llm = FakeLLM()
    rec = make_runner(store, tmp_path, llm=llm, group=2).run_shard(JOB, shard.shard_id, CFG)
    # 60 frames -> 3 clips; NVIDIA groups <= 2 clips of one episode per generate call
    assert [len(c[0]) for c in llm.calls] == [2, 1]
    (part,) = rec.parts
    (r,) = P.read_part(jf_of(store), part)
    assert r.stage == "reward.dense" and r.dense is not None and r.dense.per_step
    assert [r.provenance.source[r_]["uri"] for r_ in ("left_wrist", "main", "right_wrist")] == [
        "/v/llllll", "/v/mmmmmm", "/v/rrrrrr"]


# --- step 0: superseded spawns ---------------------------------------------------------------

def test_superseded_token_writes_nothing(tmp_path):
    store = FakeStore()
    shard = tshard(["a"])
    setup(store, [ep("a")], [shard])
    jf = jf_of(store)
    t1, t2 = (A.spawn_token(JOB, CFG, shard.shard_id, n) for n in (1, 2))
    call = PATHS.call(CFG, shard.kind, shard.shard_id)
    store.put_bytes(call, json.dumps({"token": t2, "status": "active"}).encode())
    n = len(store.writes)
    runner = make_runner(store, tmp_path)
    assert runner.run_shard(JOB, shard.shard_id, CFG, token=t1) is None
    assert len(store.writes) == n and runner.engine.llm.calls == []
    assert A.read_attempt(jf, shard.kind, shard.shard_id, "c1") is None
    store.put_bytes(call, json.dumps({"token": t2, "status": "done"}).encode())
    assert runner.run_shard(JOB, shard.shard_id, CFG, token=t2) is None
    store.put_bytes(call, json.dumps({"token": t2, "status": "active"}).encode())
    assert runner.run_shard(JOB, shard.shard_id, CFG, token=t2).episodes_done == (uid("a"),)


# --- step 2 + step 5: preemption, crash, resume across containers ---------------------------

def test_preempt_flushes_marks_and_resumes_in_new_container(tmp_path):
    store = FakeStore()
    eps = [ep("a", "task a"), ep("b", "task b"), ep("c", "task c")]
    shard = tshard(["a", "b", "c"])
    m = setup(store, eps, [shard])
    jf = jf_of(store)
    first = make_runner(store, tmp_path, cid="c1", llm=InterruptAt(2))
    with pytest.raises(KeyboardInterrupt):
        first.run_shard(JOB, shard.shard_id, CFG)
    part0 = PATHS.part(CFG, shard.kind, shard.shard_id, "c1", 0)
    assert [r.uid for r in P.read_part(jf, part0)] == [uid("a")]
    assert not store.exists(PATHS.receipt(CFG, shard.kind, shard.shard_id))
    marker = A.read_attempt(jf, shard.kind, shard.shard_id, "c1")
    assert marker.preempted and marker.ended_utc
    assert A.counted_starts(jf, shard.kind, shard.shard_id) == 0  # preemption never counts
    assert first.preempt() is False  # idle again

    llm2 = FakeLLM()
    rec = make_runner(store, tmp_path, cid="c2", llm=llm2).run_shard(JOB, shard.shard_id, CFG)
    assert not any("task a" in p for p in prompts(llm2))  # a was in c1's part
    assert rec.episodes_done == shard.episodes
    assert part0 in rec.parts and len(rec.parts) == 3  # c1's part + c2's two
    ci = P.compose_input(jf, P.derive(jf, m, limit=3))
    assert not ci.missing and not ci.conflicts and ci.duplicates == 0


def test_resume_under_listing_lag_redoes_and_compose_dedupes(tmp_path):
    store = FakeStore(lag=5)
    shard = tshard(["a", "b"])
    m = setup(store, [ep("a", "task a"), ep("b", "task b")], [shard])
    jf = jf_of(store)
    with pytest.raises(KeyboardInterrupt):
        make_runner(store, tmp_path, cid="c1", llm=InterruptAt(2)).run_shard(
            JOB, shard.shard_id, CFG)
    llm2 = FakeLLM()
    rec = make_runner(store, tmp_path, cid="c2", llm=llm2).run_shard(JOB, shard.shard_id, CFG)
    assert len(prompts(llm2)) == 2  # c1's part not listed yet: redone
    assert all("/c2-" in p for p in rec.parts)
    for _ in range(5):
        store.tick()
    ci = P.compose_input(jf, P.derive(jf, m, limit=3))
    assert not ci.missing and ci.duplicates == 1 and not ci.conflicts


class Killed(BaseException):
    """SIGKILL / OOM: nothing in the container gets to run."""


def test_hard_crash_leaves_marker_unmarked(tmp_path):
    store = FakeStore()
    shard = tshard(["a", "b"])
    setup(store, [ep("a"), ep("b")], [shard])
    jf = jf_of(store)
    with pytest.raises(Killed):
        make_runner(store, tmp_path, llm=InterruptAt(2, exc=Killed)).run_shard(
            JOB, shard.shard_id, CFG)
    marker = A.read_attempt(jf, shard.kind, shard.shard_id, "c1")
    assert not marker.preempted and marker.error is None
    assert A.counted_starts(jf, shard.kind, shard.shard_id) == 1


def test_python_exception_marks_error_and_reraises(tmp_path):
    store = FakeStore()
    shard = tshard(["a"])
    setup(store, [ep("a")], [shard])
    jf = jf_of(store)
    with pytest.raises(RuntimeError, match="reset_prefix_cache"):
        make_runner(store, tmp_path, llm=FakeLLM(reset=False)).run_shard(
            JOB, shard.shard_id, CFG)
    marker = A.read_attempt(jf, shard.kind, shard.shard_id, "c1")
    assert "reset_prefix_cache" in marker.error and not marker.preempted
    assert A.counted_starts(jf, shard.kind, shard.shard_id) == 0
    assert not store.exists(PATHS.receipt(CFG, shard.kind, shard.shard_id))


def test_rubric_shard_is_deferred(tmp_path):
    store = FakeStore()
    shard = tshard(["a"], sid="reward.rubric-00000", kind="reward.rubric")
    setup(store, [ep("a")], [shard])
    with pytest.raises(NotImplementedError):
        make_runner(store, tmp_path).run_shard(JOB, shard.shard_id, CFG)
    assert "NotImplementedError" in A.read_attempt(
        jf_of(store), shard.kind, shard.shard_id, "c1").error


def test_exit_hook_preempt_is_idempotent(tmp_path):
    store = FakeStore()
    shard = tshard(["a", "b"])
    setup(store, [ep("a"), ep("b")], [shard])
    seen = []
    runner = make_runner(store, tmp_path, llm=InterruptAt(2, hook=lambda: seen.append(
        runner.preempt())))
    with pytest.raises(KeyboardInterrupt):
        runner.run_shard(JOB, shard.shard_id, CFG)
    assert seen == [True]
    attempt = PATHS.attempt(CFG, shard.kind, shard.shard_id, "c1")
    assert store.writes.count(attempt) == 2  # the start + one preempt mark
    assert A.read_attempt(jf_of(store), shard.kind, shard.shard_id, "c1").preempted


def test_same_container_rerun_never_overwrites_its_parts(tmp_path):
    store = FakeStore()
    shard = tshard(["a", "b"])
    setup(store, [ep("a", "task a"), ep("b", "task b")], [shard])
    runner = make_runner(store, tmp_path, llm=InterruptAt(2))
    with pytest.raises(KeyboardInterrupt):
        runner.run_shard(JOB, shard.shard_id, CFG)
    part0 = PATHS.part(CFG, shard.kind, shard.shard_id, "c1", 0)
    before = store.get_bytes(part0)
    rec = runner.run_shard(JOB, shard.shard_id, CFG)  # Modal retried the input in-container
    assert store.get_bytes(part0) == before
    assert rec.parts == (part0, PATHS.part(CFG, shard.kind, shard.shard_id, "c1", 1))
    assert rec.episodes_done == shard.episodes


# --- step 3: upstream poison -----------------------------------------------------------------

def test_upstream_poisoned_episodes_are_reported_not_run(tmp_path):
    store = FakeStore()
    seg = Shard("segment-00000", "segment", (uid("a"), uid("b")), 1.0)
    shard = tshard(["a", "b"], depends_on=("segment-00000",))
    setup(store, [ep("a", "task a"), ep("b", "task b")], [seg, shard])
    jf = jf_of(store)
    child = A.bisect(seg)[0]
    assert child.episodes == (uid("a"),)
    A.write_poison(jf, child, message="poison pill (D15)")
    llm = FakeLLM()
    rec = make_runner(store, tmp_path, llm=llm).run_shard(JOB, shard.shard_id, CFG)
    assert rec.episodes_failed == (uid("a"),) and rec.episodes_done == (uid("b"),)
    assert not any("task a" in p for p in prompts(llm))
    path = PATHS.errors(CFG, shard.kind, shard.shard_id, A.UPSTREAM_CONTAINER, 0)
    (err,) = P.read_errors(jf, path)
    assert err.kind == "poison_pill" and err.uid == uid("a") and err.stage == "reward.terminal"


# --- step 4: per-episode isolation -----------------------------------------------------------

def test_nv_failures_become_episode_errors(tmp_path):
    store = FakeStore()
    eps = [ep("ok", "task ok"), ep("garbled", "task GARBLE"),
           ep("badvid", "task bv", cams={"main": "/v/bad123"})]
    shard = tshard([e[0].episode_id for e in eps])
    m = setup(store, eps, [shard])
    llm = FakeLLM(answers={"GARBLE": "I refuse"})
    rec = make_runner(store, tmp_path, llm=llm, group=8).run_shard(JOB, shard.shard_id, CFG)
    assert rec.episodes_done == (uid("ok"),)
    assert set(rec.episodes_failed) == {uid("garbled"), uid("badvid")}
    kinds = {e.uid: e.kind for e in all_errors(store, shard)}
    assert kinds == {uid("garbled"): "nv_failure", uid("badvid"): "nv_failure"}
    garbled = next(e for e in all_errors(store, shard) if e.uid == uid("garbled"))
    assert "unexpected terminal answer" in garbled.message
    assert garbled.container_id == "c1" and garbled.shard_id == shard.shard_id
    jf = jf_of(store)
    prog = P.derive(jf, m, limit=3)
    assert prog.episodes[(uid("garbled"), "reward.terminal")] == "failed"
    assert not P.compose_input(jf, prog).missing


def test_infra_input_error_is_retryable_never_a_receipt(tmp_path):
    """Review-4 H1/C1: an input that cannot be read is an infra error: the finished episodes are
    flushed, the shard raises (no receipt), and the retry runs only the missing episode."""
    from flora_deploy.workers_impl.reward import ShardInfraError

    store = FakeStore()
    eps = [ep("ok", "task ok"), ep("noindex", write_index=False)]
    shard = tshard(["ok", "noindex"])
    m = setup(store, eps, [shard])
    with pytest.raises(ShardInfraError, match="noindex"):
        make_runner(store, tmp_path, group=8).run_shard(JOB, shard.shard_id, CFG)
    jf = jf_of(store)
    assert P.read_receipt(jf, shard.kind, shard.shard_id) is None
    kinds = {e.uid: e.kind for e in all_errors(store, shard)}
    assert kinds == {uid("noindex"): "infra"}
    marker = A.read_attempt(jf, shard.kind, shard.shard_id, "c1")
    assert marker is not None and marker.error and "ShardInfraError" in marker.error
    prog = P.derive(jf, m, limit=3)
    assert prog.episodes[(uid("noindex"), "reward.terminal")] == "pending"  # not excused
    assert prog.episodes[(uid("ok"), "reward.terminal")] == "pending"
    assert not prog.complete
    # the media appears (a transient error went away): the retry finishes the shard
    _ref, receipt = ep("noindex")
    store.put_bytes(_ref.index_ref, json.dumps(receipt).encode())
    llm = FakeLLM()
    rec = make_runner(store, tmp_path, cid="c2", llm=llm, group=8).run_shard(
        JOB, shard.shard_id, CFG)
    assert set(rec.episodes_done) == {uid("ok"), uid("noindex")} and not rec.episodes_failed
    assert not any("task ok" in p for p in prompts(llm))  # not redone
    assert P.derive(jf, m, limit=3).complete


def test_missing_task_fails_the_shard(tmp_path):
    """Review-4 C2: no task instruction is a configuration error, never an excused episode."""
    from flora_deploy.workers_impl.reward import TaskInstructionMissing

    store = FakeStore()
    shard = tshard(["notask"])
    setup(store, [ep("notask", task=None)], [shard])
    with pytest.raises(TaskInstructionMissing):
        make_runner(store, tmp_path).run_shard(JOB, shard.shard_id, CFG)
    assert P.read_receipt(jf_of(store), shard.kind, shard.shard_id) is None


def test_cuda_fault_calls_fatal_after_marking(tmp_path):
    from flora_deploy.workers_impl.reward import ShardInfraError

    store = FakeStore()
    shard = tshard(["a"])
    setup(store, [ep("a")], [shard])
    seen = []
    runner = make_runner(store, tmp_path)
    runner.fatal = lambda exc: seen.append(
        A.read_attempt(jf_of(store), shard.kind, shard.shard_id, "c1"))

    def boom(ref):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    runner.episode_for = boom
    with pytest.raises(ShardInfraError):
        runner.run_shard(JOB, shard.shard_id, CFG)
    assert len(seen) == 1 and seen[0].error  # the marker was written before the container dies


def test_receipt_instruction_text_and_fixed_default(tmp_path):
    """C2: the resolved instruction in the receipt (``instruction_text``) wins; a ``fixed``
    receipt without it uses the configured fixed text (``default_task``)."""
    _ref, rec = ep("a", task="dataset task")
    fixed = {**rec, "schema": "flora_index_receipt_v1", "status": "indexed",
             "dataset": {"name": DS}, "uid": uid("a"), "instruction": "fixed",
             "instruction_text": "Fold the towel."}
    assert parse_index_receipt(json.dumps(fixed)).task == "Fold the towel."
    old = {k: v for k, v in fixed.items() if k != "instruction_text"}
    assert parse_index_receipt(json.dumps(old)).task is None
    store = FakeStore()
    shard = tshard(["a"])
    ref, _ = ep("a")
    store.put_bytes(ref.index_ref, json.dumps(old).encode())
    setup(store, [(ref, None)], [shard])
    llm = FakeLLM()
    out = make_runner(store, tmp_path, llm=llm, default_task="Fold the towel.").run_shard(
        JOB, shard.shard_id, CFG)
    assert out.episodes_done == (uid("a"),)
    assert any("Fold the towel." in p for p in prompts(llm))


def test_envelope_ignore_and_default_task(tmp_path):
    store = FakeStore()
    shard = tshard(["x"])
    setup(store, [ep("x", task=None, envelope_ok=False)], [shard])
    rec = make_runner(store, tmp_path, envelope="ignore", default_task="do it").run_shard(
        JOB, shard.shard_id, CFG)
    (r,) = P.read_part(jf_of(store), rec.parts[0])
    assert r.certified


def test_bisected_child_shard_runs(tmp_path):
    store = FakeStore()
    parent = tshard(["a", "b"])
    setup(store, [ep("a"), ep("b")], [parent])
    jf = jf_of(store)
    A.write_replan(jf, parent, A.bisect(parent), created_utc="2026-10-21T00:00:00.000000Z")
    child = A.child_shard_id(parent.kind, uid("b"))
    rec = make_runner(store, tmp_path).run_shard(JOB, child, CFG)
    assert rec.shard_id == child and rec.episodes_done == (uid("b"),)


def test_part_paths_are_whole_files_under_job_layout(tmp_path):
    store = FakeStore()
    shard = tshard(["a"])
    setup(store, [ep("a")], [shard])
    rec = make_runner(store, tmp_path).run_shard(JOB, shard.shard_id, CFG)
    assert all(p.startswith(f"flora/{JOB}/{CFG}/parts/reward.terminal/") for p in rec.parts)
    # local scratch files are gone once published
    assert not any(Path(tmp_path, "c1").rglob("*.jsonl"))


def test_worker_output_composes(tmp_path):
    from flora_deploy.export.compose import compose

    store = FakeStore()
    t = tshard(["a", "b"])
    d = tshard(["a"], sid="reward.dense-00000", kind="reward.dense")
    setup(store, [ep("a", cams={"main": "/v/aaaaaa"}), ep("b")], [t, d])
    runner = make_runner(store, tmp_path, group=2)
    for s in (t, d):
        runner.run_shard(JOB, s.shard_id, CFG)
    out = compose(store, JOB)
    assert [r.uid for r in out.records["reward.terminal"]] == [uid("a"), uid("b")]
    assert [r.uid for r in out.records["reward.dense"]] == [uid("a")]


def test_index_receipt_v1_layout():
    rec = {"schema": "flora_index_receipt_v1", "status": "indexed", "uid": uid("e"),
           "episode_id": "e", "dataset": {"name": DS}, "instruction": "dataset_task",
           "task": "pick", "envelope": {"ok": False, "reasons": ["x"]},
           "streams": {"main": {"source": {"uri": "videos/f.mp4", "start_s": 1.0,
                                           "end_s": 3.0}}}}
    idx = parse_index_receipt(json.dumps(rec))
    assert (idx.dataset, idx.task, idx.envelope_ok) == (DS, "pick", False)
    assert idx.sources == {"main": SourceRef("videos/f.mp4", 1.0, 3.0)}
    assert parse_index_receipt(json.dumps({**rec, "instruction": "fixed"})).task is None
    with pytest.raises(ValueError, match="not indexed"):
        parse_index_receipt(json.dumps({**rec, "status": "source_error", "error": "missing"}))


def test_volume_store_is_refreshed_before_every_input(tmp_path):
    class Refreshing(FakeStore):
        refreshes = 0

        def refresh(self):
            self.refreshes += 1
            return True

    store = Refreshing()
    shard = tshard(["a"])
    setup(store, [ep("a")], [shard])
    runner = make_runner(store, tmp_path)
    runner.run_shard(JOB, shard.shard_id, CFG)
    runner.run_shard(JOB, shard.shard_id, CFG)
    assert store.refreshes == 2
