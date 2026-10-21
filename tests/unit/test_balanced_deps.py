"""The balanced segment -> caption dependency, per episode, across two pools.

Whole jobs through ``flora run --local --fake-models`` with ``preset: balanced`` (the real
index, manifest, orchestrator, store adapters, budget gate, ``BalancedShardRunner`` and
micro-batcher; ``local.balanced``'s fake jobs at the model boundary), plus unit tests of the
pieces: ``orchestrator.deps_ready`` (per-episode gating across bisection),
``attempts.upstream_excluded`` and ``progress.tally_exclusions`` (an episode excluded at segment
is excluded at caption with a reason, counted once).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flora_deploy.export.results import iter_records
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.manifest import load_manifest
from flora_deploy.jobs.orchestrator import deps_ready
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.jobs.types import Shard, ShardReceipt
from flora_deploy.local import LocalRunner, run_local
from flora_deploy.local import sample_dataset as make_sample
from flora_deploy.local.runner import digest_outputs

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
JOB = "local-demo"
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")


def balanced_yaml(text: str) -> str:
    return (text.replace("preset: exact", "preset: balanced\nallow_uncertified: true")
            .replace("  segcap: {max_containers: 2}",
                     "  segment: {max_containers: 1}\n  caption: {max_containers: 1}")
            .replace("shard_target_gpu_minutes: 0.5", "shard_target_gpu_minutes: 0.1")
            .replace("budget_usd: 5 ", "budget_usd: 50 "))


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("balanced-demo")
    (d / "flora.yaml").write_text(balanced_yaml(TEMPLATE.read_text()))
    make_sample(d / "videos")
    return d / "flora.yaml"


@pytest.fixture(scope="module")
def clean(demo, tmp_path_factory):
    res = run_local(demo, store=tmp_path_factory.mktemp("clean") / "store")
    assert res.state == "done", res.as_dict()
    return res


def manifest(res):
    return load_manifest(LocalStore(res.store_root), res.job_id)


def records(res) -> dict[tuple[str, str], dict]:
    st = LocalStore(res.store_root)
    return {(r.uid, r.stage): json.loads(r.to_json()) for stage in STAGES
            for r in iter_records(st, res.job_id, stage)}


def uid_of(res, ep: str) -> str:
    return next(e.uid for e in manifest(res).episodes if e.uid.endswith(f"--{ep}"))


def first(res, kind: str) -> str:
    return next(s.shard_id for s in manifest(res).shards if s.kind == kind)


def spawn_order(res) -> list[tuple[str, str]]:
    return [(e["kind"], e["shard_id"]) for e in res.events if e["event"] == "spawned"]


def done_at(res) -> dict[str, int]:
    return {e["shard_id"]: i for i, e in enumerate(res.events) if e["event"] == "shard_done"}


# --- clean -----------------------------------------------------------------------------------


def test_clean_balanced_run_has_two_pools_and_caption_follows_segments(clean):
    m = manifest(clean)
    kinds = {s.kind for s in m.shards}
    assert kinds == {"segment", "caption", "reward.terminal", "reward.dense"}
    segs = [s for s in m.shards if s.kind == "segment"]
    caps = [s for s in m.shards if s.kind == "caption"]
    assert len(segs) >= 2 and len(caps) >= 2  # several shards per pool
    owner = {u: s.shard_id for s in segs for u in s.episodes}
    for c in caps:  # each caption shard depends exactly on the segment shards of its episodes
        assert set(c.depends_on) == {owner[u] for u in c.episodes}
    # and was spawned only after every one of them finished
    spawned = {sid: i for i, e in enumerate(clean.events) if e["event"] == "spawned"
               for sid in [e["shard_id"]]}
    done = done_at(clean)
    for c in caps:
        assert all(done[d] < spawned[c.shard_id] for d in c.depends_on)
    s = clean.summary
    for stage in STAGES:
        c = s["stages"][stage]
        assert c["records"] == c["expected"] == 6 and c["excused"] == c["missing"] == 0


def test_captions_are_made_from_the_upstream_segment_records(clean):
    recs = records(clean)
    for (uid, stage), cap in recs.items():
        if stage != "caption":
            continue
        seg = recs[(uid, "segment")]
        spans = [(x["start_s"], x["end_s"]) for x in seg["segments"]]
        assert [(x["start_s"], x["end_s"]) for x in cap["segments"]] == spans
        assert all(x["caption"] for x in cap["segments"])
    for r in recs.values():
        if r["stage"] in ("segment", "caption"):
            p = r["provenance"]
            assert p["preset"] == "balanced" and p["embodiment"] == "human" and p["gpu"]
            assert p["certificate"] == "local-fake-models" and not r["certified"]


def test_rerun_resumes_with_no_calls(demo, clean, tmp_path):
    again = run_local(demo, store=tmp_path / "store")
    assert again.state == "done" and digest_outputs(again.summary) == \
        digest_outputs(clean.summary)
    third = run_local(demo, store=tmp_path / "store")
    assert third.state == "done" and third.spawner["calls"] == 0


# --- preemption, poison, nv_failure across the two pools ---------------------------------------


def test_preempted_segment_and_caption_shards_resume_identically(demo, clean, tmp_path):
    res = run_local(demo, store=tmp_path / "store",
                    faults=f"preempt:{first(clean, 'segment')},preempt:{first(clean, 'caption')}")
    assert res.state == "done", res.as_dict()
    assert res.spawner["preemptions"] == 2
    assert records(res) == records(clean)
    jf = A.JobFiles(LocalStore(res.store_root), JOB, res.config_hash)
    for kind in ("segment", "caption"):
        marks = A.markers(jf, kind, first(clean, kind))
        assert any(m.preempted for m in marks)
        assert A.counted_starts(jf, kind, first(clean, kind)) == len(marks) - 1


def _excused(res, stage):
    return res.summary["stages"][stage]["excused"], \
        res.summary["stages"][stage]["excused_upstream"]


def test_poison_at_segment_is_excluded_at_caption_once(demo, clean, tmp_path):
    res = run_local(demo, store=tmp_path / "store", faults="oom:ep003@segment")
    assert res.state == "completed_with_errors", res.as_dict()
    u3 = uid_of(res, "ep003")
    s = res.summary
    assert s["poison_pills"] == [f"{u3}/segment"]
    assert _excused(res, "segment") == (1, 0)
    assert _excused(res, "caption") == (1, 1)  # excused, inherited from segment
    assert _excused(res, "reward.terminal") == (0, 0)
    # counted once: only the segment stage is over the error rate
    assert [p.split(":")[0] for p in s["problems"]] == ["segment"]
    # the caption shard holding ep003 ran (bisection of the segment shard resolved it)
    # and reported it with a reason, without touching the model
    st = LocalStore(res.store_root)
    jf = A.JobFiles(st, JOB, res.config_hash)
    cap = next(c for c in manifest(res).shards if c.kind == "caption" and u3 in c.episodes)
    path = jf.paths.errors(res.config_hash, "caption", cap.shard_id, A.UPSTREAM_CONTAINER, 0)
    (err,) = P.read_errors(jf, path)
    assert err.uid == u3 and err.kind == "poison_pill" and err.diagnostics["upstream"]
    keep = {k for k in records(clean) if k[0] != u3 or k[1].startswith("reward")}
    got = records(res)
    assert set(got) == keep and all(got[k] == records(clean)[k] for k in keep)


def test_poison_at_caption_keeps_the_segment_record(demo, clean, tmp_path):
    res = run_local(demo, store=tmp_path / "store", faults="oom:ep003@caption")
    assert res.state == "completed_with_errors", res.as_dict()
    u3 = uid_of(res, "ep003")
    s = res.summary
    assert s["poison_pills"] == [f"{u3}/caption"]
    assert _excused(res, "segment") == (0, 0)
    assert _excused(res, "caption") == (1, 0)
    assert [p.split(":")[0] for p in s["problems"]] == ["caption"]
    got = records(res)
    assert got[(u3, "segment")] == records(clean)[(u3, "segment")]
    assert (u3, "caption") not in got


def test_nv_failure_at_segment_is_terminal_downstream_not_retried(demo, clean, tmp_path):
    res = run_local(demo, store=tmp_path / "store", faults="nverr:ep003@segment")
    assert res.state == "completed_with_errors", res.as_dict()
    assert res.spawner["calls"] == clean.spawner["calls"]  # no respawn, no retry loop
    assert res.spawner["python_failures"] == 0 and res.counts("call_failed") == 0
    assert _excused(res, "caption") == (1, 1)
    assert res.summary["errors"]["by_kind"] == {"nv_failure": 2}
    assert [p.split(":")[0] for p in res.summary["problems"]] == ["segment"]
    st = LocalStore(res.store_root)
    jf = A.JobFiles(st, JOB, res.config_hash)
    u3 = uid_of(res, "ep003")
    cap = next(c for c in manifest(res).shards if c.kind == "caption" and u3 in c.episodes)
    r = P.read_receipt(jf, "caption", cap.shard_id)
    assert u3 in r.episodes_failed


def test_budget_projection_covers_both_pools(demo, tmp_path):
    from flora_deploy.jobs.deps import build_deps

    runner = LocalRunner(demo, store=tmp_path / "store")
    try:
        runner.plan()
        deps = build_deps(runner.store, runner.settings)
    finally:
        runner.close()
    gate = deps.budget
    shards = list(deps.manifests.load().shards)
    full = gate.projection({})
    gate.shards = lambda: [s for s in shards if s.kind != "caption"]
    no_caption = gate.projection({})
    gate.shards = lambda: [s for s in shards if s.kind != "segment"]
    no_segment = gate.projection({})
    assert full.queued_usd > no_caption.queued_usd > 0
    assert full.queued_usd > no_segment.queued_usd > 0
    assert full.startup_usd > max(no_caption.startup_usd, no_segment.startup_usd)
    pools = runner.settings.pools
    for kind in ("segment", "caption"):
        sh = next(s for s in shards if s.kind == kind)
        assert gate.estimator.container_seconds(sh, pools[kind]) > 0
        assert gate.estimator.startup_s(kind) > 0


# --- unit: per-episode gating, exclusions, tallies -----------------------------------------------


def _shards():
    seg = Shard("segment-a", "segment", ("u1", "u2", "u3"), 1.0)
    cap1 = Shard("caption-1", "caption", ("u1",), 1.0, depends_on=("segment-a",))
    cap23 = Shard("caption-23", "caption", ("u2", "u3"), 1.0, depends_on=("segment-a",))
    kids = A.bisect(seg, done={"u1"})
    return seg, cap1, cap23, kids


def test_deps_ready_is_per_episode_across_bisection():
    seg, cap1, cap23, kids = _shards()
    k2, k3 = kids
    by_id = {s.shard_id: s for s in (seg, cap1, cap23, *kids)}
    # the segment shard is running: nothing downstream may start
    assert not deps_ready(cap1, by_id, set())
    # done: everything downstream may start
    assert deps_ready(cap23, by_id, {seg.shard_id})
    # quarantined + bisected; u1 was done in the parent, u2's child is done, u3's is running
    children = {seg.shard_id: [k2.shard_id, k3.shard_id]}
    assert deps_ready(cap1, by_id, {k2.shard_id}, children)
    assert not deps_ready(cap23, by_id, {k2.shard_id}, children)
    assert deps_ready(cap23, by_id, {k2.shard_id, k3.shard_id}, children)
    # a child the manifest does not show yet: wait
    lagging = {s: by_id[s] for s in by_id if s != k3.shard_id}
    assert not deps_ready(cap1, lagging, {k2.shard_id}, children)


def test_upstream_excluded_names_poison_and_nv_failures(tmp_path):
    from flora_deploy.jobs.store import JobPaths

    seg, _cap1, cap23, (k2, k3) = _shards()
    store = LocalStore(tmp_path)
    jf = A.JobFiles(store, "job", "c" * 64)
    assert A.upstream_excluded(jf, cap23, [seg, k2, k3]) == {}
    A.write_poison(jf, k3, message="poison pill (D15)")
    rec = ShardReceipt(shard_id=k2.shard_id, kind="segment", config_hash="c" * 64,
                       container_id="x", episodes_done=(), episodes_failed=("u2",), parts=(),
                       gpu=None, wall_seconds=1.0, finished_utc="2026-10-22T00:00:00Z")
    P.write_receipt(jf, rec)
    got = A.upstream_excluded(jf, cap23, [seg, k2, k3])
    assert got == {"u2": "nv_failure", "u3": "poison_pill"}
    path = A.write_upstream_failures(jf, cap23, got)
    errs = {e.uid: e for e in P.read_errors(jf, path)}
    assert errs["u2"].kind == "nv_failure" and errs["u3"].kind == "poison_pill"
    assert all(e.diagnostics == {"upstream": True, "reason": e.kind} for e in errs.values())
    assert JobPaths("job").root in path


def test_tally_counts_an_upstream_exclusion_once():
    planned = {"segment": {"a", "b", "c", "d"}, "caption": {"a", "b", "c", "d"},
               "reward.terminal": {"a", "b", "c", "d"}}
    excluded = {"segment": {"a"}, "caption": {"a", "b"}}
    t = P.tally_exclusions(planned, excluded)
    assert t["segment"] == {"planned": 4, "done": 3, "excluded": 1, "pending": 0, "upstream": 0}
    assert t["caption"] == {"planned": 3, "done": 2, "excluded": 1, "pending": 0, "upstream": 1}
    assert t["reward.terminal"]["excluded"] == 0
    # exact segcap has no upstream kind: unchanged
    t2 = P.tally_exclusions({"segcap": {"a", "b"}}, {"segcap": {"a"}})
    assert t2["segcap"] == {"planned": 2, "done": 1, "excluded": 1, "pending": 0, "upstream": 0}


def test_local_runner_accepts_balanced(demo):
    LocalRunner(demo).close()  # used to refuse segment / caption (no fake engine)
