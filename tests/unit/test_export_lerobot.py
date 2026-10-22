"""`flora export --to lerobot`: synthetic v2.1 / v3.0 datasets, alignment, exclusions, refusals,
and the Hub push against a mocked HfApi (docs/export.md)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from flora_deploy.export.hub import dataset_card, fetch_source, push_to_hub
from flora_deploy.export.lerobot import (
    DENSE_COLUMN,
    REPORT_NAME,
    SUBTASK_COLUMN,
    ExportError,
    ExportRefused,
    check_policy,
    dense_per_frame,
    export_lerobot,
    subtask_per_frame,
)
from flora_deploy.records import DenseReward, Provenance, Segment, StageRecord, TerminalReward

DATASET = "org/demo"
FPS = 10
VIDEO_KEY = "observation.images.main"

# ---------------------------------------------------------------------------------------------
# Synthetic LeRobot datasets (layout as written by LeRobot; videos are placeholder bytes: the
# export never decodes them)

BASE_FEATURES = {
    VIDEO_KEY: {"dtype": "video", "shape": [48, 64, 3], "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
HF_FEATURES = {
    "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 2,
                          "_type": "Sequence"},
    **{k: {"dtype": BASE_FEATURES[k]["dtype"], "_type": "Value"}
       for k in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
}


def _frames(e: int, n: int, start: int) -> pa.Table:
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())), ("timestamp", pa.float32()),
        ("frame_index", pa.int64()), ("episode_index", pa.int64()), ("index", pa.int64()),
        ("task_index", pa.int64())],
        metadata={b"huggingface": json.dumps({"info": {"features": HF_FEATURES}}).encode()})
    return pa.Table.from_pydict({
        "observation.state": [[float(i), float(e)] for i in range(n)],
        "timestamp": [i / FPS for i in range(n)], "frame_index": list(range(n)),
        "episode_index": [e] * n, "index": list(range(start, start + n)), "task_index": [0] * n,
    }, schema=schema)


def make_v21(root: Path, lengths: list[int]) -> Path:
    info = {"codebase_version": "v2.1", "robot_type": "demo", "total_episodes": len(lengths),
            "total_frames": sum(lengths), "total_tasks": 1, "total_videos": len(lengths),
            "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
            "splits": {"train": f"0:{len(lengths)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/"
                          "episode_{episode_index:06d}.mp4",
            "features": BASE_FEATURES}
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    eps, stats, start = [], [], 0
    for e, n in enumerate(lengths):
        p = root / f"data/chunk-000/episode_{e:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_frames(e, n, start), p)
        v = root / f"videos/chunk-000/{VIDEO_KEY}/episode_{e:06d}.mp4"
        v.parent.mkdir(parents=True, exist_ok=True)
        v.write_bytes(b"not really an mp4 %d" % e)
        eps.append(json.dumps({"episode_index": e, "tasks": ["stack cups"], "length": n}))
        stats.append(json.dumps({"episode_index": e, "stats": {"timestamp": {
            "min": [0.0], "max": [(n - 1) / FPS], "mean": [0.5], "std": [0.1], "count": [n]}}}))
        start += n
    (root / "meta/episodes.jsonl").write_text("\n".join(eps) + "\n")
    (root / "meta/episodes_stats.jsonl").write_text("\n".join(stats) + "\n")
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "stack cups"})
                                           + "\n")
    (root / "README.md").write_text("source card\n")
    return root


def make_v3(root: Path, lengths: list[int], per_file: int = 2) -> Path:
    info = {"codebase_version": "v3.0", "robot_type": None, "total_episodes": len(lengths),
            "total_frames": sum(lengths), "total_tasks": 1, "chunks_size": 1000,
            "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "fps": FPS,
            "splits": {"train": f"0:{len(lengths)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": BASE_FEATURES}
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    cols: dict[str, list] = {k: [] for k in (
        "episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index", f"videos/{VIDEO_KEY}/chunk_index",
        f"videos/{VIDEO_KEY}/file_index", f"videos/{VIDEO_KEY}/from_timestamp",
        f"videos/{VIDEO_KEY}/to_timestamp", "meta/episodes/chunk_index",
        "meta/episodes/file_index", "stats/timestamp/mean")}
    start = 0
    for f in range(math.ceil(len(lengths) / per_file)):
        group = list(range(f * per_file, min((f + 1) * per_file, len(lengths))))
        p = root / f"data/chunk-000/file-{f:03d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        tables, t = [], 0.0
        for e in group:
            n = lengths[e]
            tables.append(_frames(e, n, start))
            for k, v in (("episode_index", e), ("tasks", ["stack cups"]), ("length", n),
                         ("data/chunk_index", 0), ("data/file_index", f),
                         ("dataset_from_index", start), ("dataset_to_index", start + n),
                         (f"videos/{VIDEO_KEY}/chunk_index", 0),
                         (f"videos/{VIDEO_KEY}/file_index", f),
                         (f"videos/{VIDEO_KEY}/from_timestamp", t),
                         (f"videos/{VIDEO_KEY}/to_timestamp", t + n / FPS),
                         ("meta/episodes/chunk_index", 0), ("meta/episodes/file_index", 0),
                         ("stats/timestamp/mean", [0.5])):
                cols[k].append(v)
            t += n / FPS
            start += n
        with pq.ParquetWriter(p, tables[0].schema) as w:  # one row group per episode
            for tb in tables:
                w.write_table(tb)
        v = root / f"videos/{VIDEO_KEY}/chunk-000/file-{f:03d}.mp4"
        v.parent.mkdir(parents=True, exist_ok=True)
        v.write_bytes(b"concatenated %d" % f)
    pq.write_table(pa.Table.from_pydict(cols), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pydict({"task_index": [0], "task": ["stack cups"]}),
                   root / "meta/tasks.parquet")
    (root / "meta/stats.json").write_text(json.dumps({"timestamp": {"mean": [0.5]}}))
    return root


def tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------------------------
# Fake records


def prov(**kw) -> Provenance:
    base = dict(models={"rewarder": "rev1"}, engine="vllm-0.24.0-offline", preset="balanced",
                config_hash="stagehash", logic_version="0.1.0", framework="flora-deploy test",
                certificate="cert-1")
    return Provenance(**{**base, **kw})


def records(lengths: list[int], *, skip: frozenset[int] = frozenset(),
            uncertified: frozenset[int] = frozenset(), fake: bool = False
            ) -> dict[str, list[StageRecord]]:
    out: dict[str, list[StageRecord]] = {"caption": [], "reward.terminal": [], "reward.dense": []}
    for e, n in enumerate(lengths):
        if e in skip:
            continue
        eid = f"episode_{e:06d}"
        dur = n / FPS
        kw = dict(uid=f"org-demo--{eid}", dataset=DATASET, episode_id=eid,
                  provenance=prov(certificate="local-fake-models") if fake else prov())
        if fake:
            kw.update(certified=False, uncertified_reason="fake_models")
        elif e in uncertified:
            kw.update(certified=False, uncertified_reason="uncertified_stack")
        else:
            kw.update(certified=True)
        half = round(dur / 2, 3)
        out["caption"].append(StageRecord(stage="caption", segments=(
            Segment("event_0001", half, dur, caption="place"),
            Segment("event_0000", 0.0, half, caption="pick")), **kw))
        out["reward.terminal"].append(StageRecord(
            stage="reward.terminal", terminal=TerminalReward(4, 3.875, (0, 0.05, 0.1, 0.75, 0.1)),
            **kw))
        steps = math.ceil(dur / 0.5 - 1e-9)
        out["reward.dense"].append(StageRecord(stage="reward.dense", dense=DenseReward(
            segments=(), per_step=tuple(1 + (k % 3) for k in range(steps))), **kw))
    return out


def read_data(root: Path) -> pa.Table:
    return pa.concat_tables(pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet")))


def check_schema(root: Path) -> dict:
    """What LeRobot relies on when loading: every non-video info.json feature is a data column
    of the declared dtype, and the parquet ``huggingface`` features cover every column."""
    info = json.loads((root / "meta/info.json").read_text())
    for p in sorted((root / "data").rglob("*.parquet")):
        schema = pq.read_schema(p)
        hf = json.loads(schema.metadata[b"huggingface"])["info"]["features"]
        assert set(hf) == set(schema.names), p
        for k, f in info["features"].items():
            if f["dtype"] == "video":
                continue
            assert k in schema.names, (p, k)
            if f["shape"] == [1]:
                assert str(schema.field(k).type) == {"float32": "float", "int64": "int64"}[
                    f["dtype"]], (k, schema.field(k).type)
                assert hf[k] == {"dtype": f["dtype"], "_type": "Value"}
    return info


# ---------------------------------------------------------------------------------------------
# Dense alignment edge cases


def test_dense_alignment_rule():
    per_step = [1, 2, 3]  # steps [0, .5), [.5, 1), [1, 1.5)
    t = [0.0, 0.1, 0.4998, 0.5, 0.99, 1.0, 1.49, 1.5, 2.7, -0.1]
    got = dense_per_frame(per_step, t)
    assert got.dtype == np.float32
    assert got[:7].tolist() == [1, 1, 1, 2, 2, 3, 3]
    assert np.isnan(got[7:]).all()  # past the last step, and a negative time: NaN, not held


def test_dense_alignment_float32_boundaries():
    # LeRobot stores timestamps as float32 frame_index / fps: a frame on a step boundary can be
    # a hair below it; the 1e-4 s tolerance puts it in the step it starts.
    for fps in (10, 15, 30, 50):
        ts = (np.arange(0, 4 * fps) / fps).astype(np.float32)
        k = np.floor((ts.astype(np.float64) + 1e-4) / 0.5).astype(int)
        expect = np.floor(np.arange(0, 4 * fps) * 2 / fps + 1e-9).astype(int)
        assert (k == expect).all(), fps
    below = np.nextafter(np.float32(0.5), np.float32(0))
    assert dense_per_frame([1, 2], [below]).tolist() == [2]


def test_dense_alignment_frame_rate_not_multiple_of_step():
    # 7 fps: frames at 0, 1/7, ...; frame 4 (0.571 s) is the first in step 1.
    ts = np.arange(0, 14) / 7
    got = dense_per_frame([1, 2, 3, 1], ts)
    assert got.tolist() == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1]


def test_dense_alignment_empty_and_short():
    assert np.isnan(dense_per_frame([], [0.0, 0.1])).all()
    assert np.isnan(dense_per_frame(None, [0.0])).all()
    assert dense_per_frame([3], []).shape == (0,)
    # the dataset holds more frames than the rewarder saw: the tail is NaN
    got = dense_per_frame([3, 3], np.arange(0, 15) / 10)
    assert got[:10].tolist() == [3] * 10 and np.isnan(got[10:]).all()


def test_subtask_index_gaps_and_overlaps():
    subs = [{"start_s": 0.0, "end_s": 1.0}, {"start_s": 0.8, "end_s": 1.5},
            {"start_s": 2.0, "end_s": 3.0}]
    t = [0.0, 0.85, 1.0, 1.6, 2.0, 2.9998, 3.0]
    assert subtask_per_frame(subs, t).tolist() == [0, 0, 1, -1, 2, 2, -1]
    assert subtask_per_frame(None, [0.0]).tolist() == [-1]


# ---------------------------------------------------------------------------------------------
# Round trips


LENGTHS = [23, 10, 17]  # 2.3 s, 1.0 s, 1.7 s at 10 fps


@pytest.mark.parametrize("make", [make_v21, make_v3], ids=["v2.1", "v3.0"])
def test_round_trip(tmp_path, make):
    src = make(tmp_path / "src", LENGTHS)
    before = tree_digest(src)
    recs = records(LENGTHS, skip=frozenset({1}))  # episode 1 was excluded
    out = tmp_path / "out"
    rep = export_lerobot(src, out, recs, source_info={"repo": DATASET, "revision": "abc123"},
                         job_info={"job_id": "j", "config_hash": "cfg"})
    assert tree_digest(src) == before  # the source is never modified
    assert not list(tmp_path.glob(".out.flora-export-tmp-*"))

    info = check_schema(out)
    assert info["features"][DENSE_COLUMN] == {"dtype": "float32", "shape": [1], "names": None}
    assert info["features"][SUBTASK_COLUMN] == {"dtype": "int64", "shape": [1], "names": None}
    for k in ("total_episodes", "total_frames", "fps", "data_path", "video_path"):
        assert info[k] == json.loads((src / "meta/info.json").read_text())[k]
    fl = info["flora"]
    assert fl["stages"] == ["caption", "reward.terminal", "reward.dense"]
    assert fl["policy"] == {"allow_uncertified": False, "allow_fake": False,
                            "records": {"certified": 6, "uncertified": 0, "fake": 0}}
    assert fl["source"]["revision"] == "abc123" and fl["videos"]["included"] is False
    assert "hf download org/demo" in fl["videos"]["note"]

    # videos are referenced, not copied; everything else is
    assert not (out / "videos").exists()
    for rel in before:
        if not rel.startswith(("videos/", "data/", "meta/episodes", "meta/info.json",
                               "README.md")):
            assert (out / rel).read_bytes() == (src / rel).read_bytes(), rel

    # per-frame columns, in the source's row order, other columns untouched
    data = read_data(out)
    orig = read_data(src)
    assert data.num_rows == orig.num_rows == sum(LENGTHS)
    for c in orig.column_names:
        assert data.column(c).to_pylist() == orig.column(c).to_pylist()
    rows = data.to_pylist()
    for e, n in enumerate(LENGTHS):
        er = [r for r in rows if r["episode_index"] == e]
        assert len(er) == n
        dense = [r[DENSE_COLUMN] for r in er]
        sub = [r[SUBTASK_COLUMN] for r in er]
        if e == 1:
            assert all(math.isnan(x) for x in dense) and sub == [-1] * n
            continue
        steps = recs["reward.dense"][[0, None, 1][e]].dense.per_step
        assert dense == [float(steps[int((r["timestamp"] + 1e-4) // 0.5)]) for r in er]
        half = round(n / FPS / 2, 3)
        assert sub == [0 if r["timestamp"] + 1e-4 < half else 1 for r in er]

    # per-episode metadata
    if info["codebase_version"] == "v2.1":
        eps = [json.loads(x) for x in (out / "meta/episodes.jsonl").read_text().splitlines()]
    else:
        tb = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet")
        assert str(tb.schema.field("flora_subtasks").type) == (
            "list<element: struct<start_s: double, end_s: double, caption: string>>")
        assert str(tb.schema.field("flora_terminal_score").type) == "int64"
        assert "stats/timestamp/mean" in tb.column_names  # source columns kept
        eps = tb.to_pylist()
    by = {r["episode_index"]: r for r in eps}
    assert by[0]["flora_subtasks"] == [
        {"start_s": 0.0, "end_s": 1.15, "caption": "pick"},
        {"start_s": 1.15, "end_s": 2.3, "caption": "place"}]
    assert by[0]["flora_terminal_score"] == 4 and by[0]["flora_terminal_expected"] == 3.875
    assert by[0]["flora_certified"] is True
    assert [p["stage"] for p in by[0]["flora_provenance"]] == ["caption", "reward.terminal",
                                                                "reward.dense"]
    assert by[0]["flora_provenance"][0]["certificate"] == "cert-1"
    assert by[0]["flora_provenance"][0]["preset"] == "balanced"
    assert by[0]["flora_provenance"][0]["config_hash"] == "stagehash"
    assert by[0]["tasks"] == ["stack cups"]
    for k in ("flora_subtasks", "flora_terminal_score", "flora_terminal_expected",
              "flora_certified", "flora_provenance"):
        assert by[1][k] is None, k

    # the report lists the excluded episode
    report = json.loads((out / REPORT_NAME).read_text())
    assert report == rep.to_dict()
    assert report["episodes"] == 3 and report["labeled"] == 2
    assert report["unlabeled"] == [{
        "episode_index": 1, "episode_id": "episode_000001", "uid": "org-demo--episode_000001",
        "missing_stages": ["caption", "reward.terminal", "reward.dense"], "reason": "excluded"}]
    card = (out / "README.md").read_text()
    assert "NVIDIA OneWay Noncommercial License" in card and "nvidia/Flora-Rewarder" in card
    assert "license: other" in card and "`org/demo@abc123`" in card


@pytest.mark.parametrize("make", [make_v21, make_v3], ids=["v2.1", "v3.0"])
def test_include_videos_and_stage_subset(tmp_path, make):
    src = make(tmp_path / "src", LENGTHS)
    out = tmp_path / "out"
    export_lerobot(src, out, records(LENGTHS), stages=["reward.terminal"], include_videos=True)
    for v in (src / "videos").rglob("*.mp4"):
        assert (out / v.relative_to(src)).read_bytes() == v.read_bytes()
    info = check_schema(out)
    assert DENSE_COLUMN not in info["features"] and SUBTASK_COLUMN not in info["features"]
    assert info["flora"]["episode_fields"].keys() == {
        "flora_terminal_score", "flora_terminal_expected", "flora_certified",
        "flora_provenance"}
    assert DENSE_COLUMN not in read_data(out).column_names


def test_dense_tail_reported(tmp_path):
    src = make_v3(tmp_path / "src", [20])
    recs = records([20])
    recs["reward.dense"][0] = StageRecord(**{**recs["reward.dense"][0].__dict__,
                                             "dense": DenseReward((), (3, 3, 3))})  # 1.5 s of 2
    rep = export_lerobot(src, tmp_path / "out", recs, stages=["reward.dense"])
    assert rep.dense_uncovered == [{"episode_index": 0, "frames": 20, "nan_frames": 5}]


def test_re_export_of_an_export_is_refused(tmp_path):
    src = make_v3(tmp_path / "src", LENGTHS)
    export_lerobot(src, tmp_path / "out", records(LENGTHS))
    with pytest.raises(ExportError, match="already declares"):
        export_lerobot(tmp_path / "out", tmp_path / "out2", records(LENGTHS))


# ---------------------------------------------------------------------------------------------
# Refusals and output-directory safety


def test_uncertified_refused_unless_allowed(tmp_path):
    src = make_v21(tmp_path / "src", LENGTHS)
    recs = records(LENGTHS, uncertified=frozenset({2}))
    with pytest.raises(ExportRefused, match="3 records are uncertified.*uncertified_stack: 3"):
        export_lerobot(src, tmp_path / "out", recs)
    assert not (tmp_path / "out").exists()
    rep = export_lerobot(src, tmp_path / "out", recs, allow_uncertified=True)
    assert rep.records == {"certified": 6, "uncertified": 3, "fake": 0}
    info = json.loads((tmp_path / "out/meta/info.json").read_text())
    assert info["flora"]["policy"]["allow_uncertified"] is True
    eps = [json.loads(x) for x in (tmp_path / "out/meta/episodes.jsonl").read_text().splitlines()]
    assert [e["flora_certified"] for e in eps] == [True, True, False]
    assert "uncertified" in (tmp_path / "out/README.md").read_text()


def test_fake_refused_unless_allowed(tmp_path):
    src = make_v3(tmp_path / "src", LENGTHS)
    recs = records(LENGTHS, fake=True)
    with pytest.raises(ExportRefused, match="fake models"):
        export_lerobot(src, tmp_path / "out", recs)
    with pytest.raises(ExportRefused, match="fake models"):  # --allow-uncertified is not enough
        export_lerobot(src, tmp_path / "out", recs, allow_uncertified=True)
    rep = export_lerobot(src, tmp_path / "out", recs, allow_fake=True)
    assert rep.records["fake"] == 9 and rep.allow_fake
    assert "fake models" in (tmp_path / "out/README.md").read_text()


def test_fake_job_refused_and_policy_counts():
    with pytest.raises(ExportRefused, match="fake"):
        check_policy([], allow_uncertified=True, allow_fake=False, fake_job=True)
    assert check_policy([], allow_uncertified=False, allow_fake=True, fake_job=True) == {
        "certified": 0, "uncertified": 0, "fake": 0}


def test_output_directory_safety(tmp_path):
    src = make_v21(tmp_path / "src", LENGTHS)
    recs = records(LENGTHS)
    for bad in (src, src / "sub", tmp_path):
        with pytest.raises(ExportError, match="overlaps"):
            export_lerobot(src, bad, recs)
    (tmp_path / "full").mkdir()
    (tmp_path / "full/x").write_text("x")
    with pytest.raises(ExportError, match="not empty"):
        export_lerobot(src, tmp_path / "full", recs)
    (tmp_path / "empty").mkdir()
    export_lerobot(src, tmp_path / "empty", recs)


def test_mismatched_source_and_bad_stage(tmp_path):
    src = make_v21(tmp_path / "src", [10])
    with pytest.raises(ExportError, match="not in"):
        export_lerobot(src, tmp_path / "out", records(LENGTHS))
    with pytest.raises(ExportError, match="cannot be exported"):
        export_lerobot(src, tmp_path / "out", records([10]), stages=["reward.rubric"])
    with pytest.raises(ExportError, match="not a LeRobot dataset"):
        export_lerobot(tmp_path, tmp_path / "out", records([10]))


# ---------------------------------------------------------------------------------------------
# Hub (mocked; nothing leaves the machine)


class FakeApi:
    def __init__(self, existing_private: bool | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.existing_private = existing_private

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))
        if self.existing_private is None:
            self.existing_private = kw["private"]

    def repo_info(self, **kw):
        self.calls.append(("repo_info", kw))
        return SimpleNamespace(private=self.existing_private)

    def upload_folder(self, **kw):
        files = sorted(p.relative_to(kw["folder_path"]).as_posix()
                       for p in Path(kw["folder_path"]).rglob("*") if p.is_file())
        self.calls.append(("upload_folder", {**kw, "files": files}))
        return SimpleNamespace(oid="c0ffee")

    def delete_tag(self, **kw):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        self.calls.append(("delete_tag", kw))
        resp = httpx.Response(404, request=httpx.Request("DELETE", "https://hub.invalid"))
        raise HfHubHTTPError("404 tag not found", response=resp)

    def create_tag(self, **kw):
        self.calls.append(("create_tag", kw))


def test_push_to_hub_private_by_default(tmp_path):
    src = make_v3(tmp_path / "src", LENGTHS)
    out = tmp_path / "out"
    export_lerobot(src, out, records(LENGTHS), job_info={"job_id": "j", "config_hash": "cfg"})
    api = FakeApi()
    url = push_to_hub(out, "me/labels", api=api)
    assert url == "https://huggingface.co/datasets/me/labels"
    calls = dict(api.calls)
    assert calls["create_repo"] == {"repo_id": "me/labels", "repo_type": "dataset",
                                    "private": True, "exist_ok": True}
    up = calls["upload_folder"]
    assert up["repo_type"] == "dataset" and "README.md" in up["files"]
    assert REPORT_NAME in up["files"] and "meta/info.json" in up["files"]
    assert "job j" in up["commit_message"]
    assert calls["create_tag"] == {"repo_id": "me/labels", "repo_type": "dataset",
                                   "tag": "v3.0", "revision": "c0ffee"}


def test_push_refuses_existing_public_repo(tmp_path):
    src = make_v21(tmp_path / "src", LENGTHS)
    export_lerobot(src, tmp_path / "out", records(LENGTHS))
    api = FakeApi(existing_private=False)
    with pytest.raises(ExportError, match="public"):
        push_to_hub(tmp_path / "out", "me/labels", api=api)
    assert "upload_folder" not in dict(api.calls)
    push_to_hub(tmp_path / "out", "me/labels", api=FakeApi(existing_private=False),
                private=False)
    with pytest.raises(ExportError, match="not a flora export"):
        push_to_hub(src, "me/labels", api=FakeApi())


def test_fetch_source_patterns(tmp_path):
    seen = {}

    def download(**kw):
        seen.update(kw)
        return kw["local_dir"]

    root = fetch_source("org/big", "rev1", "test/task", tmp_path / "w", download=download)
    assert root == tmp_path / "w" / "test/task"
    assert seen["allow_patterns"] == ["test/task/meta/**", "test/task/data/**"]
    assert seen["revision"] == "rev1" and seen["repo_type"] == "dataset"
    fetch_source("org/big", "rev1", "", tmp_path / "w", include_videos=True, download=download)
    assert seen["allow_patterns"] == ["meta/**", "data/**", "videos/**"]
    with pytest.raises(ExportError, match="revision"):
        fetch_source("org/big", "", "", tmp_path / "w", download=download)


def test_card_is_valid_front_matter(tmp_path):
    src = make_v3(tmp_path / "src", LENGTHS)
    export_lerobot(src, tmp_path / "out", records(LENGTHS))
    info = json.loads((tmp_path / "out/meta/info.json").read_text())
    card = dataset_card(info, None)
    import yaml

    front = yaml.safe_load(card.split("---\n")[1])
    assert front["license"] == "other" and "LeRobot" in front["tags"]
    assert front["configs"][0]["data_files"] == "data/*/*.parquet"


# ---------------------------------------------------------------------------------------------
# From a composed job (store), through the SDK and the CLI


def _composed_job(tmp_path):
    from flora_deploy.export.compose import compose
    from flora_deploy.jobs.stores import LocalStore
    from tests.unit.test_export_compose import JOB, setup_job

    store = LocalStore(tmp_path / "store")
    setup_job(store)  # dataset lerobot/a, episodes 0..3, 10 s each; episode 1 uncertified
    compose(store, JOB)
    return store, JOB


def test_export_job_sdk(tmp_path):
    from flora_deploy import sdk

    store, job = _composed_job(tmp_path)
    src = make_v3(tmp_path / "src", [100] * 5)  # episode 4 is not in the job
    with pytest.raises(ExportRefused, match="uncertified"):
        sdk.export_job(store, job, tmp_path / "out", source=src)
    api = FakeApi()
    rep = sdk.export_job(store, job, tmp_path / "out", source=src, allow_uncertified=True,
                         push_to_hub="me/x", hub_api=api)
    assert rep.pushed == "https://huggingface.co/datasets/me/x"
    assert rep.stages == ["segment", "caption", "reward.terminal"]
    assert rep.unlabeled == [{"episode_index": 4, "episode_id": "episode_000004",
                              "uid": "lerobot-a--episode_000004",
                              "missing_stages": ["segment", "caption", "reward.terminal"],
                              "reason": "not_in_job"}]
    info = json.loads((tmp_path / "out/meta/info.json").read_text())
    assert info["flora"]["job"]["job_id"] == job and info["flora"]["subtasks_from"] == "caption"
    eps = pq.read_table(tmp_path / "out/meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [len(e["flora_subtasks"] or []) for e in eps] == [3, 3, 3, 3, 0]
    assert eps[0]["flora_subtasks"][0] == {"start_s": 1.0, "end_s": 2.0, "caption": "pick 2"}
    assert [p["stage"] for p in eps[0]["flora_provenance"]] == ["caption", "reward.terminal"]
    data = read_data(tmp_path / "out").to_pylist()
    e0 = [r[SUBTASK_COLUMN] for r in data if r["episode_index"] == 0]
    assert e0[:10] == [-1] * 10 and e0[10:20] == [0] * 10 and e0[20:30] == [-1] * 10


def test_export_cli(tmp_path):
    from flora_deploy.cli import app

    store, job = _composed_job(tmp_path)
    src = make_v21(tmp_path / "src", [100] * 4)
    uri = f"file://{tmp_path / 'store'}"
    runner = CliRunner()
    env = {"COLUMNS": "200"}
    r = runner.invoke(app, ["export", "--job", job, "--store", uri, "--out",
                            str(tmp_path / "out"), "--source", str(src)], env=env)
    assert r.exit_code == 2 and "uncertified" in r.output and not (tmp_path / "out").exists()
    r = runner.invoke(app, ["export", "--job", job, "--store", uri, "--out",
                            str(tmp_path / "out"), "--source", str(src), "--allow-uncertified",
                            "--stages", "caption,reward.terminal"], env=env)
    assert r.exit_code == 0, r.output
    out = " ".join(r.output.split())
    assert "4/4 episodes labeled" in out and "6 certified, 2 uncertified, 0 fake" in out
    info = json.loads((tmp_path / "out/meta/info.json").read_text())
    assert info["flora"]["stages"] == ["caption", "reward.terminal"]
    r = runner.invoke(app, ["export", "--job", job, "--store", uri, "--out",
                            str(tmp_path / "o2"), "--source", str(src), "--to", "rlds"], env=env)
    assert r.exit_code == 2 and "only lerobot" in r.output
    r = runner.invoke(app, ["export", "--job", job, "--store", uri, "--out",
                            str(tmp_path / "o2"), "--source", str(src), "--stages",
                            "reward.dense"], env=env)
    assert r.exit_code == 2 and "no composed" in r.output
    r = runner.invoke(app, ["export", "--job", job, "--store", uri, "--out",
                            str(tmp_path / "o3"), "--allow-uncertified"], env=env)
    assert r.exit_code == 2 and "--source" in r.output  # no index summary names the dataset


# ---------------------------------------------------------------------------------------------
# Review 12 F1 / F4: the source dataset's metadata and symlinks never reach outside it


@pytest.mark.parametrize("data_path", [
    "../../home/victim/data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "../x/episode_{episode_index:06d}.parquet",
    "/abs/episode_{episode_index:06d}.parquet",
    "data/{episode_index.__class__}.parquet",
    "data/{0}.parquet",
])
def test_hostile_data_path_is_refused_and_writes_nothing_outside(tmp_path, data_path):
    victim = make_v21(tmp_path / "home" / "victim", LENGTHS)
    before = tree_digest(victim)
    src = make_v21(tmp_path / "downloads" / "evil", LENGTHS)
    info = json.loads((src / "meta/info.json").read_text())
    info["data_path"] = data_path
    (src / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ExportError, match="data_path|path"):
        export_lerobot(src, tmp_path / "downloads" / "out", records(LENGTHS))
    assert tree_digest(victim) == before
    assert not (tmp_path / "downloads" / "out").exists()


def test_data_file_symlinked_outside_is_refused(tmp_path):
    victim = make_v21(tmp_path / "victim", LENGTHS)
    before = tree_digest(victim)
    src = make_v21(tmp_path / "ds", LENGTHS)
    f = src / "data/chunk-000/episode_000000.parquet"
    f.unlink()
    f.symlink_to(victim / "data/chunk-000/episode_000000.parquet")
    with pytest.raises(ExportError, match="outside"):
        export_lerobot(src, tmp_path / "out", records(LENGTHS))
    assert tree_digest(victim) == before


def test_v3_data_file_symlinked_outside_is_refused(tmp_path):
    other = make_v3(tmp_path / "other", LENGTHS)
    src = make_v3(tmp_path / "ds", LENGTHS)
    f = sorted((src / "data").glob("chunk-*/file-*.parquet"))[0]
    rel = f.relative_to(src)
    f.unlink()
    f.symlink_to(other / rel)
    with pytest.raises(ExportError, match="outside"):
        export_lerobot(src, tmp_path / "out", records(LENGTHS))


@pytest.mark.parametrize("make", [make_v21, make_v3])
def test_symlink_out_of_the_dataset_is_not_copied(tmp_path, make):
    secret = tmp_path / "home" / ".ssh" / "id_ed25519"
    secret.parent.mkdir(parents=True)
    secret.write_text("PRIVATE KEY")
    src = make(tmp_path / "ds", LENGTHS)
    (src / "meta" / "notes.md").symlink_to(secret)
    with pytest.raises(ExportError, match="symlink"):
        export_lerobot(src, tmp_path / "out", records(LENGTHS))
    leaked = [p for p in tmp_path.rglob("*") if p.is_file() and not p.is_symlink()
              and p != secret and "PRIVATE KEY" in p.read_text(errors="ignore")]
    assert leaked == []


def test_symlinked_directory_out_of_the_dataset_is_refused(tmp_path):
    outside = tmp_path / "home" / ".ssh"
    outside.mkdir(parents=True)
    (outside / "id_ed25519").write_text("PRIVATE KEY")
    src = make_v21(tmp_path / "ds", LENGTHS)
    (src / "extra").symlink_to(outside)
    with pytest.raises(ExportError, match="symlink"):
        export_lerobot(src, tmp_path / "out", records(LENGTHS))


def test_symlink_inside_the_dataset_is_copied_as_a_file(tmp_path):
    src = make_v21(tmp_path / "ds", LENGTHS)
    (src / "NOTES.md").write_text("notes")
    (src / "meta" / "notes.md").symlink_to(src / "NOTES.md")
    export_lerobot(src, tmp_path / "out", records(LENGTHS))
    p = tmp_path / "out" / "meta" / "notes.md"
    assert not p.is_symlink() and p.read_text() == "notes"
