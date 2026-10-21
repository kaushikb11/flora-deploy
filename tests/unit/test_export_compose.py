"""compose/export: dedupe + validation, Parquet round-trip, idempotency, summary counts."""

from __future__ import annotations

import json

import pytest

from flora_deploy.export.compose import (
    ComposeError,
    compose,
    stage_schema,
    to_jsonl,
    to_parquet,
    to_table,
)
from flora_deploy.export.results import (
    ResultsNotFound,
    iter_records,
    load_parquet,
    load_summary,
    records_from_table,
    store_from_uri,
)
from flora_deploy.jobs.attempts import JobFiles, quarantine
from flora_deploy.jobs.manifest import build_manifest, save_manifest, stages_of
from flora_deploy.jobs.progress import write_errors, write_part, write_receipt
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores import LocalStore, VolumeStore
from flora_deploy.jobs.types import EpisodeError, Manifest, Shard, ShardReceipt
from flora_deploy.records import (
    DenseReward,
    Provenance,
    RubricReward,
    Segment,
    StageRecord,
    TerminalReward,
)
from tests.unit.test_jobs_manifest import make_index, make_settings

JOB, CFG = "job-1", "cfg0"


def prov(cfg: str = CFG) -> Provenance:
    return Provenance(models={"tempseg": "rev1"}, engine="hf-ref-5.8.1", preset="exact",
                      config_hash=cfg, logic_version="0.1.0", framework="flora-deploy test",
                      gpu="H200", source={"main": {"uri": "/d/f0.mp4", "range_s": [0.0, 10.0]}})


def make_rec(m: Manifest, uid: str, stage: str, *, cfg: str = CFG, note: str = "") -> StageRecord:
    ref = next(e for e in m.episodes if e.uid == uid)
    certified = not uid.endswith("1")  # every episode_...1 is out of envelope
    kw: dict = {}
    if stage in ("segment", "caption"):
        kw["segments"] = tuple(
            Segment(event_id=f"event_{i:04d}", start_s=5.0 - 2 * i, end_s=6.0 - 2 * i,
                    caption=f"pick {i}{note}" if stage == "caption" else None,
                    segmentation_provenance="tempseg", degradation={"x": i},
                    nv_metadata={"k": [1, 2.5]})
            for i in range(3))
    elif stage == "reward.terminal":
        kw["terminal"] = TerminalReward(score=4, expected_score=3.875,
                                        probabilities=(0.0, 0.05, 0.1, 0.75, 0.1),
                                        rationale="ok" + note)
    return StageRecord(uid=uid, dataset=ref.dataset, episode_id=ref.episode_id,
                       stage=stage, provenance=prov(cfg), certified=certified,  # type: ignore
                       uncertified_reason=None if certified else "outside_reference_envelope",
                       **kw)


def setup_job(store, *, n: int = 4, target_min: float = 1.0, skip: tuple[str, ...] = ()
              ) -> tuple[Manifest, JobFiles]:
    idx = [make_index("lerobot/a", f"episode_{i:06d}", "/d/f0.mp4", 10.0) for i in range(n)]
    m = build_manifest(idx, make_settings("exact", job_id=JOB, cfg=CFG, target_min=target_min),
                       created_utc="t")
    save_manifest(store, m)
    jf = JobFiles(store, JOB, CFG)
    for s in m.shards:
        if s.shard_id in skip:
            continue
        finish(jf, m, s)
    return m, jf


def finish(jf: JobFiles, m: Manifest, s: Shard, *, cid: str = "c1", t: str = "2026-01-01",
           failed: tuple[str, ...] = (), note: str = "") -> str:
    recs = [make_rec(m, u, st, note=note) for u in s.episodes if u not in failed
            for st in stages_of(s.kind)]
    p = write_part(jf, s.kind, s.shard_id, cid, 0, recs)
    write_receipt(jf, ShardReceipt(
        shard_id=s.shard_id, kind=s.kind, config_hash=CFG, container_id=cid,
        episodes_done=tuple(u for u in s.episodes if u not in failed), episodes_failed=failed,
        parts=(p,), gpu="H200", wall_seconds=1.0, finished_utc=t))
    return p


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


def test_compose_writes_outputs_and_summary(store):
    m, _jf = setup_job(store)
    res = compose(store, JOB)
    paths = JobPaths(JOB)
    for stage in ("segment", "caption", "reward.terminal"):
        for fmt in ("jsonl", "parquet"):
            assert store.exists(paths.output(CFG, stage, fmt))
    s = load_summary(store, JOB)
    assert s == res.summary and s["complete"] and s["config_hash"] == CFG
    seg = s["stages"]["segment"]
    assert seg["records"] == 4 and seg["rows"] == 12 and seg["expected"] == 4
    assert seg["certified"] == 3 and seg["certified_share"] == 0.75
    assert seg["uncertified_reasons"] == {"outside_reference_envelope": 1}
    pool = m.episodes[0].pool
    assert seg["by_pool"] == {pool: {"certified": 3, "records": 4}}
    assert s["stages"]["reward.terminal"]["rows"] == 4
    assert s["records"] == 12 and s["duplicates_resolved"] == 0 and s["conflicts"] == []
    assert s["inputs"]["parts"] == len(m.shards)
    assert set(s["outputs"]) == {paths.output(CFG, st, f) for st in res.records
                                 for f in ("jsonl", "parquet")}

    recs = list(iter_records(store, JOB, "caption"))
    assert [r.uid for r in recs] == sorted(e.uid for e in m.episodes)
    assert recs == res.records["caption"]


def test_dedupe_latest_source_wins_and_conflicts_reported(store):
    m, jf = setup_job(store)
    segcap = next(s for s in m.shards if s.kind == "segcap")
    # A later container rewrote the same shard: identical segment records, a different caption.
    finish(jf, m, segcap, cid="c2", t="2026-02-01", note="!")
    # A record from another config hash in a listed part is dropped.
    foreign = make_rec(m, segcap.episodes[0], "segment", cfg="other")
    write_part(jf, segcap.kind, segcap.shard_id, "c3", 0, [foreign])
    res = compose(store, JOB)
    s = res.summary
    n = len(segcap.episodes)
    assert s["duplicates_resolved"] == 2 * n
    assert s["conflicts"] == sorted(f"{u}/caption" for u in segcap.episodes)
    assert s["foreign_records_dropped"] == 1
    caps = {r.uid: r for r in res.records["caption"]}
    assert all(caps[u].segments[0].caption.endswith("!") for u in segcap.episodes)
    assert s["stages"]["caption"]["records"] == 4  # one per uid


def test_invalid_records_refused(store):
    m, jf = setup_job(store)
    s = m.shards[0]
    bad = make_rec(m, s.episodes[0], stages_of(s.kind)[0])
    object.__setattr__(bad, "dataset", "someone/else")
    write_part(jf, s.kind, s.shard_id, "c1", 0, [bad])  # replaces the receipt-named part
    with pytest.raises(ComposeError, match="disagree with the manifest"):
        compose(store, JOB)
    assert not store.exists(JobPaths(JOB).output(CFG, "summary", "json"))


def test_malformed_part_refused(store):
    m, jf = setup_job(store)
    s = m.shards[0]
    path = jf.paths.part(CFG, s.kind, s.shard_id, "c9", 0)
    store.put_bytes(path, b'{"schema": "flora_labels_v1", "uid": 1}\n')
    with pytest.raises(ComposeError, match="unreadable compose input"):
        compose(store, JOB)


def test_incomplete_job_needs_partial(store):
    idx_m, _ = setup_job(LocalStore(store.root / "probe"))
    skip = next(s.shard_id for s in idx_m.shards if s.kind == "reward.terminal")
    m, _jf = setup_job(store, skip=(skip,))
    with pytest.raises(ComposeError, match="not complete"):
        compose(store, JOB)
    with pytest.raises(ResultsNotFound):
        list(iter_records(store, JOB, "segment"))
    res = compose(store, JOB, partial=True)
    s = res.summary
    skipped = next(sh for sh in m.shards if sh.shard_id == skip)
    assert not s["complete"] and s["partial"]
    assert s["stages"]["reward.terminal"]["missing"] == len(skipped.episodes)
    assert s["stages"]["segment"]["missing"] == 0


def test_failed_episodes_poison_pills_and_errors_in_summary(store):
    probe, _ = setup_job(LocalStore(store.root / "probe"), target_min=0.01)
    terms = [s for s in probe.shards if s.kind == "reward.terminal"]
    assert len(terms) == 4  # one episode per shard at this target
    m, jf = setup_job(store, target_min=0.01, skip=(terms[0].shard_id, terms[1].shard_id))
    # NVIDIA failed one episode: the receipt lists it as failed, with an error file.
    uid = terms[0].episodes[0]
    finish(jf, m, terms[0], failed=(uid,))
    write_errors(jf, terms[0].kind, terms[0].shard_id, "c1", 0, [EpisodeError(
        uid=uid, stage="reward.terminal", shard_id=terms[0].shard_id, container_id="c1",
        kind="nv_failure", message="no score token")])
    # Another terminal shard crash-looped and was poison-pilled.
    quarantine(jf, terms[1], created_utc="2026-01-02", starts=3)
    s = compose(store, JOB).summary
    st = s["stages"]["reward.terminal"]
    assert s["complete"]
    assert st["records"] == 2 and st["excused"] == 2 and st["missing"] == 0
    assert st["episodes"] == {"done": 2, "failed": 1, "poison_pill": 1}
    assert s["poison_pills"] == [f"{terms[1].episodes[0]}/reward.terminal"]
    assert s["errors"]["by_kind"]["nv_failure"] == 1
    assert s["errors"]["by_stage"]["reward.terminal"] >= 1


def test_idempotent_byte_identical(store):
    setup_job(store)
    a = compose(store, JOB)
    before = {p: store.get_bytes(p) for p in [*a.summary["outputs"], a.paths["summary"]]}
    b = compose(store, JOB)
    after = {p: store.get_bytes(p) for p in before}
    assert before == after and a.summary == b.summary


def test_parquet_round_trip_from_store(store):
    m, _ = setup_job(store)
    res = compose(store, JOB)
    for stage, recs in res.records.items():
        table = load_parquet(store, JOB, stage)
        assert table.schema.equals(stage_schema(stage), check_metadata=False)
        assert records_from_table(table) == recs
    seg = load_parquet(store, JOB, "segment").to_pylist()
    # Exploded rows sorted by uid, then start time.
    keys = [(r["uid"], r["start_s"]) for r in seg]
    assert keys == sorted(keys)
    assert seg[0]["pool"] == m.episodes[0].pool and json.loads(seg[0]["provenance"])["gpu"]


def _bare(stage: str, uid: str = "d--e1", **kw) -> StageRecord:
    return StageRecord(uid=uid, dataset="d", episode_id="e1", stage=stage,  # type: ignore
                       provenance=prov(), certified=True, audit_ref="a/b.jsonl", **kw)


def test_parquet_round_trip_all_reward_shapes_and_extras():
    dense = DenseReward(segments=({"label": "ok", "span": [0, 1], "rationale": "r"},),
                        per_step=(3, 3, 2, 1))
    cases = {
        "reward.dense": [_bare("reward.dense", dense=dense),
                         _bare("reward.dense", uid="d--e0", dense=dense,
                               segments=(Segment("ev", 0.0, 1.0),))],
        "reward.rubric": [_bare("reward.rubric", rubric=RubricReward(
            scores=(1.0, 0.5), total=1.5, samples=8, medoid_index=3))],
        "reward.terminal": [_bare("reward.terminal")],  # no payload at all
        "segment": [_bare("segment"),  # zero segments: one row with null segment columns
                    _bare("segment", uid="d--e2", segments=(
                        Segment("b", 2.0, 3.0, status="unsupported_target_frames"),
                        Segment("a", 0.5, 1.0)),
                        terminal=TerminalReward(1, 1.0, (1.0, 0, 0, 0, 0)))],
    }
    for stage, recs in cases.items():
        table = to_table(stage, recs)
        back = records_from_table(table)
        assert sorted(r.to_json() for r in back) == sorted(r.to_json() for r in recs), stage
        assert to_parquet(table) == to_parquet(to_table(stage, list(reversed(recs))))
        assert to_jsonl(recs) == to_jsonl(list(reversed(recs)))
    seg = to_table("segment", cases["segment"]).to_pylist()
    assert len(seg) == 3 and seg[0]["segment_index"] is None
    assert [r["event_id"] for r in seg[1:]] == ["a", "b"]  # sorted by start time
    assert json.loads(seg[1]["extra"])["terminal"]["score"] == 1


def test_config_hash_mismatch_refused(store):
    setup_job(store)
    with pytest.raises(ComposeError, match="not the manifest"):
        compose(store, JOB, config_hash="cfgX")


def test_store_from_uri(tmp_path):
    a = store_from_uri(f"file://{tmp_path}/s")
    assert isinstance(a, LocalStore) and a.root == tmp_path / "s"
    assert isinstance(store_from_uri(str(tmp_path / "p")), LocalStore)
    (tmp_path / "vol").mkdir()
    v = store_from_uri("modal-volume://flora-state", volume_mount=str(tmp_path / "vol"))
    assert isinstance(v, VolumeStore)
    for bad in ("modal-volume://nope-not-mounted", "s3://", "gs://x", "file://host/x"):
        with pytest.raises(ValueError):
            store_from_uri(bad)
