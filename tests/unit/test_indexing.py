"""``flora index`` job (jobs/indexing.py): receipts, summary, resume, uids, envelope policy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flora_deploy.jobs import indexing as I
from flora_deploy.jobs.manifest import build_manifest
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.jobs.types import episode_uid
from flora_deploy.media import index as media_index
from flora_deploy.media import synth
from flora_deploy.media.index import index_episode
from flora_deploy.media.sources import open_source
from flora_deploy.media.synth import Spec

KEYS = ("observation.images.top", "observation.images.left", "observation.images.right")
CMAP = dict(zip(KEYS, ("main", "left_wrist", "right_wrist"), strict=True))


def _job(root: Path, fmt: str, *, cameras=None, pool="yam", envelope="warn", decode=True,
         job_id="job-1", **kw) -> tuple[I.IndexJob, I.DatasetFiles]:
    spec = I.DatasetSpec(format=fmt, local_path=str(root), cameras=cameras)
    return (I.IndexJob(job_id=job_id, dataset=spec, pool=pool, envelope=envelope,
                       decode=decode, **kw), I.open_dataset(spec))


def _three(lengths):
    return [{k: Spec(n, seed=e * 10 + c) for c, k in enumerate(KEYS)}
            for e, n in enumerate(lengths)]


def _rewrite_v3(root, edit):
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = root / "meta/episodes/chunk-000/file-000.parquet"
    cols = pq.read_table(f).to_pydict()
    edit(cols)
    pq.write_table(pa.Table.from_pydict(cols), f)


@pytest.fixture
def v21(tmp_path):
    # episode 2 has one frame: out of NVIDIA's envelope (too_few_frames)
    eps = [{"observation.images.top": Spec(n, seed=e)} for e, n in enumerate([12, 9, 1])]
    return synth.lerobot_v21_tree(tmp_path / "v21", eps, ["pick", "place"])


@pytest.fixture
def v3_wrong_file(tmp_path):
    """3-view v3 tree; episode 2's ranges point past the end of its file (golden §3.3)."""
    root = synth.lerobot_v3_tree(tmp_path / "v3", _three([10, 14, 8]), ["stack cups"])
    src = open_source(root, camera_map=CMAP)
    end = src.list_episodes()[2].cameras["main"].end_s

    def past_end(cols):
        for k in KEYS:
            cols[f"videos/{k}/from_timestamp"][2] = end
            cols[f"videos/{k}/to_timestamp"][2] = end + 8 / 30

    _rewrite_v3(root, past_end)
    return root


def _receipt(store, job_id, uid):
    return json.loads(store.get_bytes(JobPaths(job_id).index(uid)))


# ---------------------------------------------------------------------------------------------


def test_v21_receipts_and_summary(v21, tmp_path):
    store = LocalStore(tmp_path / "store")
    job, files = _job(v21, "lerobot_v21")
    s = I.run_index(job, store, files, log=lambda m: None)

    name = str(v21)
    uids = [episode_uid(name, f"episode_{i:06d}") for i in range(3)]
    for uid in uids:
        assert store.exists(JobPaths("job-1").index(uid))
    r0 = _receipt(store, "job-1", uids[0])
    assert r0["status"] == "indexed" and r0["task"] == "pick"
    assert r0["envelope"] == {"ok": True, "reasons": []} and r0["oracle_eligible"]
    assert r0["episode_ref"]["uid"] == uids[0]
    assert r0["episode_ref"]["index_ref"] == JobPaths("job-1").index(uids[0])
    assert r0["frames"] == 12 and r0["duration_s"] == pytest.approx(0.4)
    assert r0["streams"]["main"]["source"]["uri"] == (
        "videos/chunk-000/observation.images.top/episode_000000.mp4")
    assert r0["receipts"]["main"]["codec"] == "h264"
    r2 = _receipt(store, "job-1", uids[2])
    assert r2["envelope"]["ok"] is False and "main:too_few_frames" in r2["envelope"]["reasons"]

    c = s["counts"]
    assert (c["episodes"], c["indexed"], c["envelope_ok"], c["out_of_envelope"]) == (3, 3, 2, 1)
    assert c["planned"] == 3 and c["planned_out_of_envelope"] == 1 and c["resumed"] == 0
    assert s["envelope_reasons"] == {"main:too_few_frames": 1}
    assert s["codecs"] == {"h264": 3}
    assert s["units"] == 1 and s["dataset"]["revision"].startswith("local-")
    assert s["durations"]["yam/1cam"]["episodes"] == 3
    (g,) = s["groups"]
    assert (g["pool"], g["cameras"], g["episodes"]) == ("yam", 1, 3)
    assert g["video_s"] == pytest.approx((12 + 9 + 1) / 30, abs=1e-3)
    on_store = json.loads(store.get_bytes(I.summary_path("job-1")))
    assert on_store["counts"] == c

    from flora_deploy.planner.options import WorkloadSummary

    w = WorkloadSummary.from_dict(s)
    assert w.episodes == 3 and w.video_s == pytest.approx(g["video_s"])


def test_envelope_policy(v21, tmp_path):
    store = LocalStore(tmp_path / "store")
    job, files = _job(v21, "lerobot_v21", envelope="enforce")
    s = I.run_index(job, store, files, log=lambda m: None)
    assert s["counts"]["planned"] == 2 and s["counts"]["planned_out_of_envelope"] == 0
    assert s["groups"][0]["episodes"] == 2
    # the receipts record the verdict as measured, whatever the policy
    assert s["counts"]["out_of_envelope"] == 1
    assert len(I.load_indexes(store, "job-1", policy="enforce").indexes) == 2
    assert len(I.load_indexes(store, "job-1", policy="ignore").indexes) == 3

    # a policy change re-plans without re-indexing
    job_w, _ = _job(v21, "lerobot_v21", envelope="ignore")
    s2 = I.run_index(job_w, store, files, log=lambda m: None)
    assert s2["counts"]["resumed"] == 3 and s2["counts"]["planned"] == 3


def test_resume_skips_matching_receipts(v21, tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    job, files = _job(v21, "lerobot_v21")
    I.run_index(job, store, files, log=lambda m: None)
    calls = []
    real = I.index_one
    monkeypatch.setattr(I, "index_one", lambda *a, **k: calls.append(1) or real(*a, **k))

    s = I.run_index(job, store, files, log=lambda m: None)
    assert calls == [] and s["counts"]["resumed"] == 3 and s["counts"]["indexed"] == 3

    # anything a receipt depends on re-indexes: the embodiment (envelope profile) ...
    job_h, _ = _job(v21, "lerobot_v21", pool="human")
    s = I.run_index(job_h, store, files, log=lambda m: None)
    assert len(calls) == 3 and s["counts"]["resumed"] == 0
    # ... the dataset revision (a changed file) ...
    synth.encode(v21 / "videos/chunk-000/observation.images.top/episode_000001.mp4",
                 Spec(9, seed=99, size=(80, 48)))
    calls.clear()
    s = I.run_index(job_h, store, files, log=lambda m: None)
    assert len(calls) == 3
    # ... and the index code version
    calls.clear()
    monkeypatch.setattr(I, "INDEX_CODE_VERSION", "999+test")
    I.run_index(job_h, store, files, log=lambda m: None)
    assert len(calls) == 3
    # --force re-indexes everything
    calls.clear()
    I.run_index(job_h, store, files, force=True, log=lambda m: None)
    assert len(calls) == 3


def test_uids_stable_and_index_round_trips(v21, tmp_path):
    store = LocalStore(tmp_path / "store")
    job, files = _job(v21, "lerobot_v21")
    I.run_index(job, store, files, log=lambda m: None)
    # same dataset name read from another place -> same uids and receipt keys
    moved = tmp_path / "elsewhere"
    shutil.copytree(v21, moved)
    store2 = LocalStore(tmp_path / "store2")
    job2 = I.IndexJob(job_id="job-1", dataset=job.dataset, pool="yam")
    I.run_index(job2, store2, I.open_dataset(job.dataset, local_root=moved), log=lambda m: None)
    a = sorted(p.rsplit("/", 1)[1] for p in store.list("flora/job-1/index/"))
    b = sorted(p.rsplit("/", 1)[1] for p in store2.list("flora/job-1/index/"))
    assert a == b and len(a) == 4  # 3 receipts + summary.json
    uid = episode_uid(str(v21), "episode_000001")
    assert _receipt(store, "job-1", uid)["key"] == _receipt(store2, "job-1", uid)["key"]

    # the receipt holds the exact EpisodeIndex
    rec = _receipt(store, "job-1", uid)
    got = I.episode_index_from_receipt(rec, root=v21)
    ep = open_source(v21).list_episodes()[1]
    want = index_episode(ep.episode_id, str(v21), ep.cameras, task=ep.task, profile="yam",
                         decode=True)
    assert got == want


def test_v3_one_scan_per_file_and_wrong_file(v3_wrong_file, tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    job, files = _job(v3_wrong_file, "lerobot_v3", cameras=CMAP)
    scanned = []
    real = media_index.scan_stream
    monkeypatch.setattr(media_index, "scan_stream", lambda p: scanned.append(p) or real(p))
    _, units = I.plan_index(job, files, max_episodes=1)
    assert len(units) == 1 and len(units[0].files) == 3  # a concatenated file is never split

    s = I.run_index(job, store, files, max_episodes=1, log=lambda m: None)
    assert len(scanned) == 3  # one demux pass per concatenated file
    c = s["counts"]
    assert (c["indexed"], c["source_error"], c["planned"]) == (2, 1, 2)
    assert s["source_errors"] == {"wrong_file": 1}
    assert s["durations"] == {"yam/3cam": pytest.approx(s["durations"]["yam/3cam"])}
    assert s["groups"] == [{"pool": "yam", "cameras": 3, "episodes": 2,
                            "video_s": pytest.approx((10 + 14) / 30, abs=1e-3)}]
    bad = _receipt(store, "job-1", episode_uid(str(v3_wrong_file), "episode_000002"))
    assert bad["status"] == "source_error" and "wrong file" in bad["error"]
    assert bad["episode_ref"] is None and bad["verify"]["ok"] is False
    ok = _receipt(store, "job-1", episode_uid(str(v3_wrong_file), "episode_000001"))
    assert set(ok["streams"]) == {"main", "left_wrist", "right_wrist"}
    assert ok["streams"]["main"]["source"]["start_s"] > 0
    assert ok["envelope"]["ok"], ok["envelope"]

    # source errors are deterministic: resumed, not retried
    s2 = I.run_index(job, store, files, log=lambda m: None)
    assert s2["counts"]["resumed"] == 3 and s2["counts"]["source_error"] == 1


def test_mp4_dir_and_fixed_instruction(tmp_path):
    root = tmp_path / "mp4"
    synth.encode(root / "a.mp4", Spec(6))
    (root / "a.txt").write_text("wipe")
    for i, role in enumerate(("main", "left_wrist", "right_wrist")):
        synth.encode(root / "b" / f"{role}.mp4", Spec(5, seed=i))
    store = LocalStore(tmp_path / "store")
    job, files = _job(root, "mp4_dir", pool="human", instruction="fixed",
                      instruction_text="x" * 2001)
    s = I.run_index(job, store, files, log=lambda m: None)
    assert s["counts"]["indexed"] == 2
    a = _receipt(store, "job-1", episode_uid(str(root), "a"))
    b = _receipt(store, "job-1", episode_uid(str(root), "b"))
    assert a["task"] == "wipe" and a["cameras"] == 1
    # the fixed instruction is what NVIDIA sees: too long for every episode
    assert "instruction_too_long" in a["envelope"]["reasons"]
    assert "human_profile_multi_view" in b["envelope"]["reasons"]
    assert set(s["durations"]) == {"human/1cam", "human/3cam"}


def test_plan_units_packs_whole_groups(v21):
    job, files = _job(v21, "lerobot_v21")
    eps = files.list_episodes()
    units = I.plan_units(eps, max_episodes=2)
    assert [len(u.episodes) for u in units] == [2, 1]
    assert [u.unit_id for u in units] == ["unit-00000", "unit-00001"]
    assert I.WorkUnit.from_dict(json.loads(json.dumps(units[0].to_dict()))) == units[0]
    assert I.plan_units(eps, max_episodes=64)[0].files == tuple(
        sorted(f for e in eps for f in I.source_files(e)))
    with pytest.raises(ValueError):
        I.plan_units(eps, max_episodes=0)
    # limit
    _, units = I.plan_index(job, files, limit=2)
    assert sum(len(u.episodes) for u in units) == 2


def test_load_indexes_feeds_build_manifest(v3_wrong_file, tmp_path):
    store = LocalStore(tmp_path / "store")
    job, files = _job(v3_wrong_file, "lerobot_v3", cameras=CMAP)
    I.run_index(job, store, files, log=lambda m: None)
    loaded = I.load_indexes(store, "job-1")
    settings = RunSettings(job_id="job-1", preset="exact", config_hash="cfg", budget_usd=1.0,
                           max_gpus=1, pools={"segcap": PoolSettings(gpu="H200")})
    m = build_manifest(loaded.indexes, settings, created_utc="t", pool=loaded.pools)
    assert sorted(e.uid for e in m.episodes) == sorted(r.uid for r in loaded.refs)
    by_uid = {r.uid: r for r in loaded.refs}
    for e in m.episodes:
        assert e == by_uid[e.uid]  # the receipt's EpisodeRef is what the planner makes
    assert len(m.shards) == 1 and m.shards[0].kind == "segcap"


def test_job_round_trip_and_validation(v21):
    job, _ = _job(v21, "lerobot_v21", instruction="none")
    assert I.IndexJob.from_dict(json.loads(json.dumps(job.to_dict()))) == job
    assert job.profile == ("yam", False) and job.instruction_for("t") == ""
    assert I.IndexJob(job_id="j", dataset=job.dataset, pool="yam32").profile == ("yam", True)
    with pytest.raises(ValueError):
        I.IndexJob(job_id="j", dataset=job.dataset, pool="robot")
    with pytest.raises(ValueError):
        I.IndexJob(job_id="j", dataset=job.dataset, pool="yam", envelope="strict")
    with pytest.raises(ValueError):
        I.DatasetSpec(format="lerobot_v21")
    hub = I.DatasetSpec(format="lerobot_v3", repo="org/ds", revision="a" * 40, subdir="x/y")
    assert hub.name == "org/ds/x/y" and hub.key == "org--ds@aaaaaaaaaaaa"
    assert episode_uid(hub.name, "episode_000001") == "org-ds-x-y--episode_000001"
