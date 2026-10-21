"""Golden staging + manifest building on synthetic datasets served by a fake Hub."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flora_deploy.media import synth
from flora_deploy.media.synth import Spec
from flora_deploy.validate import golden
from flora_deploy.validate.golden import (
    GOLDEN_PREFIX,
    MANIFEST_NAME,
    build_manifest,
    cut_check,
    profile_rows,
    stage_all,
    summary_markdown,
)

SHA = "a" * 40
KEYS = ("observation.images.top", "observation.images.left", "observation.images.right")
CAMS = dict(zip(("main", "left_wrist", "right_wrist"), KEYS, strict=True))


class FakeHub:
    """Serves ``repos[repo]`` (a local directory) at any revision; records every request."""

    def __init__(self, repos: dict[str, Path]):
        self.repos = repos
        self.calls: list[tuple[str, str]] = []
        self.fail: set[str] = set()

    def download(self, repo, revision, filename, local_dir):
        self.calls.append(("download", filename))
        if filename in self.fail:
            raise ConnectionError(f"boom {filename}")
        src = self.repos[repo] / filename
        if not src.is_file():
            raise FileNotFoundError(filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    def list_files(self, repo, revision, prefix):
        self.calls.append(("list", prefix))
        base = self.repos[repo]
        return sorted(p.relative_to(base).as_posix() for p in (base / prefix).rglob("*")
                      if p.is_file())


def _three_view(lengths, seed=0):
    return [{k: Spec(n, seed=seed + e * 10 + c) for c, k in enumerate(KEYS)}
            for e, n in enumerate(lengths)]


@pytest.fixture(scope="module")
def repos(tmp_path_factory):
    base = tmp_path_factory.mktemp("repos")
    out = {}
    # LeRobot v3 (concatenated per camera), inside a sub-directory of the repo.
    v3 = base / "v3"
    synth.lerobot_v3_tree(v3 / "sub" / "ds", _three_view([10, 14, 8]), ["stack cups"],
                          scratch=base / "scratch3")
    out["o/v3"] = v3
    # LeRobot v2.1 (one file per episode and camera).
    v21 = base / "v21"
    synth.lerobot_v21_tree(v21, _three_view([9, 7], seed=5), ["pick", "place"])
    out["o/v21"] = v21
    # A v3 dataset whose episode meta uses file_000.parquet (jnogga/droid_success style).
    us = base / "us"
    synth.lerobot_v3_tree(us, _three_view([6, 6], seed=9), ["t"], scratch=base / "scratch_us")
    meta = us / "meta/episodes/chunk-000"
    (meta / "file-000.parquet").rename(meta / "file_000.parquet")
    out["o/us"] = us
    # Plain MP4 files + annotations (ManiRewardBench / RoboRewardBench style).
    files = base / "files"
    for seed, key in enumerate(("front", "left", "right")):
        synth.encode(files / "YAM/task/000001" / f"{key}_rgb.mp4", Spec(12, seed=seed))
    (files / "yam_annotations.json").write_text(json.dumps({
        "YAM/task/000001/front_rgb.mp4": {"task": "Fold towel.", "subtasks": [
            {"name": "a", "start_second": 0.0, "end_second": 0.2}], "mistakes": []}}))
    synth.encode(files / "test/droid/ep1.mp4", Spec(11, rate=10))
    (files / "test/metadata.jsonl").write_text(
        json.dumps({"file_name": "droid/ep1.mp4", "task": "Move it.", "reward": 4}) + "\n")
    out["o/files"] = files
    # WGO-Bench style: MP4 bytes embedded in a parquet, one row per row group.
    wgo = base / "wgo"
    vids = [synth.encode(base / f"w{i}.mp4", Spec(8 + i, seed=i)) for i in range(2)]
    rows = {"id": ["homer_1", "droid_1"], "instruction": ["", "push"],
            "video": [{"bytes": v.read_bytes(), "path": v.name} for v in vids],
            "segments": [[{"start_sec": 0.0, "end_sec": 0.1, "subtask": "s"}], []]}
    (wgo / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pydict(rows), wgo / "data/a.parquet", row_group_size=1)
    out["o/wgo"] = wgo
    return out


def _selection():
    ds = lambda repo, fmt, **kw: {"repo": repo, "revision": SHA, "format": fmt,  # noqa: E731
                                  "license": "Apache-2.0", "task_source": "meta", **kw}
    return {
        "version": "golden/v1",
        "datasets": {
            "v3": ds("o/v3", "lerobot_v3", subdir="sub/ds"),
            "v21": ds("o/v21", "lerobot_v21"),
            "us": ds("o/us", "lerobot_v3"),
            "mrb": ds("o/files", "mp4_files", annotations="manirewardbench",
                      annotation_file="yam_annotations.json"),
            "rrb": ds("o/files", "mp4_files", annotations="roboreward",
                      annotation_file="test/metadata.jsonl"),
            "wgo": ds("o/wgo", "wgo_parquet", file="data/a.parquet", license="CC-BY-NC-SA-4.0"),
        },
        "episodes": [
            {"pool": "yam", "dataset": "v3", "slug": "episode_000001", "episode": 1,
             "profile": "yam", "cameras": CAMS, "why": "mid-file v3 episode"},
            {"pool": "dex", "dataset": "v3", "slug": "episode_000002", "episode": 2,
             "profile": "dex", "cameras": CAMS, "why": "last v3 episode"},
            {"pool": "tianji", "dataset": "v21", "slug": "episode_000001", "episode": 1,
             "profile": "tianji", "cameras": CAMS, "why": "v2.1 per-episode files"},
            {"pool": "single_view_robot", "dataset": "us", "slug": "episode_000000",
             "episode": 0, "profile": "yam", "cameras": {"main": KEYS[0]}, "why": "file_000"},
            {"pool": "yam", "dataset": "mrb", "slug": "task_000001", "episode": "YAM/task/000001",
             "profile": "yam", "why": "files",
             "cameras": {"main": "YAM/task/000001/front_rgb.mp4",
                         "left_wrist": "YAM/task/000001/left_rgb.mp4",
                         "right_wrist": "YAM/task/000001/right_rgb.mp4"}},
            {"pool": "reward", "dataset": "rrb", "slug": "ep1", "episode": "droid/ep1.mp4",
             "views": ["test/droid/ep1.mp4"], "modes": ["terminal"], "outcome": {"reward": 4},
             "why": "rubric 4"},
            {"pool": "human", "dataset": "wgo", "slug": "homer_1", "episode": "homer_1",
             "row_group": 0, "profile": "human", "cameras": {"main": "video"}, "why": "wgo"},
        ],
    }


@pytest.fixture()
def staged(tmp_path, repos):
    hub = FakeHub(repos)
    commits = []
    m = stage_all(_selection(), hub=hub, root=tmp_path / "vol", work=tmp_path / "work",
                  commit=lambda: commits.append(1), log=lambda _m: None)
    return m, hub, tmp_path, commits


def test_stage_all_writes_cuts_records_and_manifest(staged):
    m, hub, tmp, commits = staged
    assert m["complete"] and not m["failed"], m["failed"]
    assert [e["status"] for e in m["episodes"]] == ["staged"] * 7
    assert all(e["oracle_eligible"] for e in m["episodes"]), [e["id"] for e in m["episodes"]]
    on_disk = json.loads((tmp / "vol" / GOLDEN_PREFIX / MANIFEST_NAME).read_text())
    assert on_disk["counts"] == m["counts"]
    assert m["counts"]["yam"] == {"selected": 2, "staged": 2, "oracle_eligible": 2}
    assert commits  # the Volume is committed per dataset and at the end
    v3 = m["episodes"][0]
    d = tmp / "vol" / v3["path"]
    assert sorted(p.name for p in d.iterdir()) == [
        "left_wrist.mp4", "main.mp4", "record.json", "right_wrist.mp4"]
    assert {f["kind"] for f in v3["files"].values()} == {"stream_copy_cut"}
    assert v3["cut_check"] == {r: "ok" for r in CAMS}
    assert v3["index"]["main"]["frames"] == 14 and v3["task"] == "stack cups"
    assert v3["source"]["subdir"] == "sub/ds"
    # Only what the episodes need was downloaded, and the work mirror is gone.
    assert not any(p.is_file() for p in (tmp / "work").rglob("*.mp4"))
    assert ("download", "sub/ds/meta/info.json") in hub.calls


def test_whole_file_sources_are_kept_and_annotated(staged):
    m = {e["id"]: e for e in staged[0]["episodes"]}
    v21 = m["tianji/v21/episode_000001"]
    assert {f["kind"] for f in v21["files"].values()} == {"source_file"}
    assert v21["task"] == "place" and v21["index"]["main"]["frames"] == 7
    mrb = m["yam/mrb/task_000001"]
    assert mrb["task"] == "Fold towel." and mrb["annotation_counts"] == {
        "subtasks": 1, "mistakes": 0}
    rrb = m["reward/rrb/ep1"]
    assert rrb["outcome"] == {"reward": 4} and set(rrb["files"]) == {"view_0"}
    assert rrb["envelope"]["ok"] and rrb["views_synchronized"]
    wgo = m["human/wgo/homer_1"]
    assert wgo["task"] is None  # empty instruction -> no task text
    assert wgo["annotation_counts"] == {"segments": 1} and wgo["index"]["main"]["frames"] == 8
    assert m["single_view_robot/us/episode_000000"]["oracle_eligible"]  # file_000.parquet


def test_rerun_skips_completed_episodes(staged, repos):
    m, _, tmp, _ = staged
    hub = FakeHub(repos)
    again = stage_all(_selection(), hub=hub, root=tmp / "vol", work=tmp / "work",
                      log=lambda _m: None)
    assert hub.calls == []
    assert [e["selection_digest"] for e in again["episodes"]] == [
        e["selection_digest"] for e in m["episodes"]]


def test_changed_entry_and_force_restage(staged, repos):
    _, _, tmp, _ = staged
    sel = _selection()
    sel["episodes"][2]["why"] = "edited"
    hub = FakeHub(repos)
    stage_all(sel, hub=hub, root=tmp / "vol", work=tmp / "work", log=lambda _m: None)
    assert ("download", "meta/episodes.jsonl") in hub.calls  # only the edited dataset
    assert not any(c[1].startswith("sub/ds") for c in hub.calls)
    hub2 = FakeHub(repos)
    stage_all(sel, hub=hub2, root=tmp / "vol", work=tmp / "work", only=["wgo"], force=True,
              log=lambda _m: None)
    assert {c[1] for c in hub2.calls} == {"data/a.parquet"}


def test_outcome_mismatch_is_a_recorded_source_error(tmp_path, repos):
    sel = _selection()
    sel["episodes"][5]["outcome"] = {"reward": 2}
    m = stage_all(sel, hub=FakeHub(repos), root=tmp_path / "vol", work=tmp_path / "w",
                  only=["rrb"], log=lambda _m: None)
    ep = next(e for e in m["episodes"] if e["id"] == "reward/rrb/ep1")
    assert ep["status"] == "source_error" and "OutcomeMismatch" in ep["error"]
    assert not ep["oracle_eligible"]


def test_wrong_metadata_range_is_a_source_error(tmp_path, repos):
    sel = _selection()
    sel["episodes"][6]["row_group"] = 1  # row group 1 holds droid_1, not homer_1
    m = stage_all(sel, hub=FakeHub(repos), root=tmp_path / "vol", work=tmp_path / "w",
                  only=["wgo"], log=lambda _m: None)
    ep = next(e for e in m["episodes"] if e["dataset"] == "wgo")
    assert ep["status"] == "source_error" and "does not hold id" in ep["error"]


def test_network_errors_leave_no_record_and_are_retried(tmp_path, repos):
    hub = FakeHub(repos)
    hub.fail = {"test/droid/ep1.mp4"}
    m = stage_all(_selection(), hub=hub, root=tmp_path / "vol", work=tmp_path / "w",
                  only=["rrb"], log=lambda _m: None)
    assert m["failed"] and m["failed"][0]["id"] == "reward/rrb/ep1" and not m["complete"]
    assert not (tmp_path / "vol" / GOLDEN_PREFIX / "reward/rrb/ep1/record.json").exists()
    m2 = stage_all(_selection(), hub=FakeHub(repos), root=tmp_path / "vol",
                   work=tmp_path / "w", only=["rrb"], log=lambda _m: None)
    assert not m2["failed"]
    assert any(e["id"] == "reward/rrb/ep1" and e["status"] == "staged" for e in m2["episodes"])


def test_deadline_defers_unstarted_datasets(tmp_path, repos):
    m = stage_all(_selection(), hub=FakeHub(repos), root=tmp_path / "vol",
                  work=tmp_path / "w", deadline=0.0, log=lambda _m: None)
    assert not m["episodes"] and len(m["failed"]) == 7
    assert all(f["error"].startswith("deadline") for f in m["failed"])


def test_build_manifest_counts_order_and_missing():
    sel = _selection()
    recs = [
        {"id": "tianji/v21/episode_000001", "pool": "tianji", "status": "staged",
         "oracle_eligible": True, "annotations": {}},
        {"id": "yam/v3/episode_000001", "pool": "yam", "status": "source_error",
         "oracle_eligible": False, "error": "SourceError: x"},
    ]
    m = build_manifest(sel, recs, [{"id": "reward/rrb/ep1", "error": "net"}])
    assert [e["id"] for e in m["episodes"]] == ["yam/v3/episode_000001",
                                                "tianji/v21/episode_000001"]
    assert m["counts"]["yam"] == {"selected": 2, "staged": 0, "oracle_eligible": 0}
    assert m["counts"]["tianji"] == {"selected": 1, "staged": 1, "oracle_eligible": 1}
    assert "reward/rrb/ep1" not in m["missing"] and "human/wgo/homer_1" in m["missing"]
    assert not m["complete"] and len(m["selection_sha256"]) == 64
    assert m["datasets"]["v3"]["subdir"] == "sub/ds"


def test_summary_and_rows_are_numbers_only(staged):
    m = staged[0]
    text = summary_markdown(m)
    assert "| **total** | 7 | 7 | 7 |" in text
    assert "stack cups" not in text and "Fold towel" not in text
    rows = {r["id"]: r for r in profile_rows(m)}
    assert rows["yam/v3/episode_000001"]["views"] == 3
    assert rows["yam/v3/episode_000001"]["kind"] == ["stream_copy_cut"]
    assert rows["reward/rrb/ep1"]["reasons"] == []


def test_cut_check_reports_fields():
    a = {k: 1 for k in golden.CUT_CHECK_FIELDS}
    assert cut_check(a, dict(a)) == []
    assert cut_check(a, {**a, "rate": "1/2", "frames": 3}) == ["rate", "frames"]
