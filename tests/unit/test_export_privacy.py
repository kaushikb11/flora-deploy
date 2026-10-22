"""`flora export` writes no local paths, and `--push-to-hub` refuses an export that holds one
(docs/export.md "Privacy"). The Hub is the mocked HfApi of test_export_lerobot: no network."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from flora_deploy.export import hub
from flora_deploy.export.hub import LocalPathLeak, check_portable, push_to_hub, scan_export
from flora_deploy.export.lerobot import (
    REPORT_NAME,
    ExportError,
    export_lerobot,
    is_local_path,
    portable_name,
    scrub_paths,
)
from flora_deploy.jobs.types import episode_uid
from tests.unit.test_export_lerobot import LENGTHS, FakeApi, make_v3, make_v21, records

PLANTED = "/Users/alice/private-datasets/cups"  # what a laptop job records as its dataset


def local_records(name: str = PLANTED, **kw):
    return {s: [dataclasses.replace(r, dataset=name, uid=episode_uid(name, r.episode_id))
                for r in rs] for s, rs in records(LENGTHS, **kw).items()}


def text_of(root: Path) -> str:
    """Every byte of the export that is text, parquet metadata and meta/ rows included."""
    parts = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".parquet":
            parts.append(repr(pq.read_schema(p).metadata))
            parts.append(json.dumps(pq.read_table(p).to_pylist(), default=str))
        else:
            parts.append(p.read_text(errors="ignore"))
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def _quiet_host(monkeypatch):
    """A fixed host name, so a real one cannot collide with test content."""
    monkeypatch.setattr(hub.socket, "gethostname", lambda: "test-host-xyz")


# ---------------------------------------------------------------------------------------------
# Portable names


@pytest.mark.parametrize(("name", "local", "portable"), [
    ("/Users/alice/ds", True, "ds"),
    ("~/ds/", True, "ds"),
    ("./ds", True, "ds"),
    ("../up/ds", True, "ds"),
    ("file:///tmp/x/ds", True, "ds"),
    ("C:\\Users\\alice\\ds", True, "ds"),
    ("/", True, "dataset"),
    ("lerobot/aloha", False, "lerobot/aloha"),
    ("lerobot/aloha/sub", False, "lerobot/aloha/sub"),
    ("volume://flora-results/inputs/ds", False, "volume://flora-results/inputs/ds"),
    ("hf://datasets/org/ds", False, "hf://datasets/org/ds"),
    ("s3://key:secret@bucket/p", False, "s3://bucket/p"),
])
def test_portable_name(name, local, portable):
    assert is_local_path(name) is local
    assert portable_name(name) == portable


def test_scrub_paths():
    assert scrub_paths("source_error: /Users/a/x.mp4: missing") == \
        "source_error: <local path>: missing"
    assert scrub_paths("index_failed: /vol/flora-media/x (EIO)") == \
        "index_failed: <local path> (EIO)"
    for keep in ("https://huggingface.co/home/x", "data/chunk-000/file-000.parquet",
                 "hf://datasets/org/x", "missing_task: no task text"):
        assert scrub_paths(keep) == keep


# ---------------------------------------------------------------------------------------------
# The export


@pytest.mark.parametrize("make", [make_v21, make_v3])
def test_local_job_export_has_no_local_paths(tmp_path, make):
    src = make(tmp_path / "src", LENGTHS)
    out = tmp_path / "out"
    recs = local_records(skip=frozenset({1}))
    real_uid = episode_uid(PLANTED, "episode_000001")
    rep = export_lerobot(
        src, out, recs, source_info={"name": PLANTED, "format": "lerobot_v21"},
        job_info={"job_id": "j", "config_hash": "c"},
        episode_reasons={real_uid: f"unplanned: index_failed: {PLANTED}/videos/e1.mp4: EIO"})
    text = text_of(out)
    for leak in (str(tmp_path), PLANTED, "Users-alice", "/Users/", str(Path.home())):
        assert leak not in text, leak
    info = json.loads((out / "meta/info.json").read_text())
    assert info["flora"]["source"]["path"] == "cups"
    assert info["flora"]["source"]["name"] == "cups"
    assert "(cups)" in info["flora"]["videos"]["note"]
    report = json.loads((out / REPORT_NAME).read_text())
    assert report["out"] == rep.out == "out"
    assert report["unlabeled"][0]["uid"] == "cups--episode_000001"
    assert report["unlabeled"][0]["reason"] == "unplanned: index_failed: <local path>: EIO"
    assert scan_export(out) == []
    api = FakeApi()
    push_to_hub(out, "me/labels", api=api)
    assert "upload_folder" in dict(api.calls)


def test_hub_source_is_recorded_as_its_uri_not_the_fetch_dir(tmp_path):
    fetched = make_v3(tmp_path / "work" / ".out.source" / "test/task", LENGTHS)
    export_lerobot(fetched, tmp_path / "out", records(LENGTHS), source_info={
        "name": "org/demo/test/task", "repo": "org/demo", "revision": "abc", "subdir": "test/task"})
    src = json.loads((tmp_path / "out/meta/info.json").read_text())["flora"]["source"]
    assert src["path"] == "hf://datasets/org/demo/test/task"
    assert src["name"] == "org/demo/test/task" and src["revision"] == "abc"
    assert str(tmp_path) not in text_of(tmp_path / "out")
    card = (tmp_path / "out/README.md").read_text()
    assert "# org/demo with Flora labels" in card and "`org/demo@abc`" in card


def test_volume_source_keeps_its_uri(tmp_path):
    src = make_v21(tmp_path / "src", LENGTHS)
    uri = "volume://flora-media/inputs/cups"
    export_lerobot(src, tmp_path / "out", local_records(uri), source_info={"name": uri})
    info = json.loads((tmp_path / "out/meta/info.json").read_text())
    assert info["flora"]["source"]["path"] == uri and info["flora"]["source"]["name"] == uri
    check_portable(tmp_path / "out")


def test_keep_local_paths_records_them_and_push_refuses(tmp_path):
    src = make_v3(tmp_path / "src", LENGTHS)
    out = tmp_path / "out"
    export_lerobot(src, out, local_records(skip=frozenset({0})),
                   source_info={"name": PLANTED}, keep_local_paths=True)
    info = json.loads((out / "meta/info.json").read_text())
    assert info["flora"]["source"]["path"] == str(src)
    assert info["flora"]["source"]["name"] == PLANTED
    report = json.loads((out / REPORT_NAME).read_text())
    assert report["out"] == str(out)
    assert report["unlabeled"][0]["uid"] == episode_uid(PLANTED, "episode_000000")
    api = FakeApi()
    with pytest.raises(LocalPathLeak) as e:
        push_to_hub(out, "me/labels", api=api)
    msg = str(e.value)
    assert "meta/info.json: flora.source.path holds an absolute local path" in msg
    assert "meta/info.json: flora.source.name" in msg and f"{REPORT_NAME}: out" in msg
    assert "README.md: line" in msg and "Nothing was uploaded" in msg
    assert PLANTED not in msg and str(tmp_path) not in msg  # names the field, not the value
    assert api.calls == []  # refused before any Hub call


# ---------------------------------------------------------------------------------------------
# The pre-push scan: paths planted in what the export copies from the source


def _export(tmp_path, make=make_v21):
    src = make(tmp_path / "src", LENGTHS)
    return src, tmp_path / "out"


def _refused(out: Path) -> str:
    api = FakeApi()
    with pytest.raises(LocalPathLeak) as e:
        push_to_hub(out, "me/labels", api=api)
    assert api.calls == []
    return str(e.value)


def test_scan_finds_a_path_in_a_copied_jsonl(tmp_path):
    src, out = _export(tmp_path)
    (src / "meta/tasks.jsonl").write_text(json.dumps(
        {"task_index": 0, "task": "stack cups", "recorded_at": "/home/bob/rec/0001"}) + "\n")
    export_lerobot(src, out, records(LENGTHS))
    msg = _refused(out)
    assert "meta/tasks.jsonl: line 1.recorded_at holds an absolute local path" in msg
    assert "/home/bob" not in msg


def test_scan_finds_a_path_in_copied_json_and_text(tmp_path):
    src, out = _export(tmp_path, make_v3)
    (src / "meta/stats.json").write_text(json.dumps({"x": {"source": "C:\\Users\\bob\\ds"}}))
    (src / "meta/notes.txt").write_text("ok\nrecorded on /mnt/nas/robot1\n")
    export_lerobot(src, out, records(LENGTHS))
    found = {(f, fld) for f, fld, _ in scan_export(out)}
    assert ("meta/stats.json", "x.source") in found
    assert ("meta/notes.txt", "line 2") in found
    assert len(found) == 2


def test_scan_finds_a_path_in_parquet_metadata_and_meta_rows(tmp_path):
    src, out = _export(tmp_path, make_v3)
    p = src / "data/chunk-000/file-000.parquet"
    t = pq.read_table(p)
    pq.write_table(t.replace_schema_metadata(
        {**t.schema.metadata, b"origin": b"/Volumes/ext/raw/run7"}), p)
    tasks = src / "meta/tasks.parquet"
    pq.write_table(pq.read_table(tasks).append_column(
        "note", [["/scratch/u/x"]]), tasks)
    export_lerobot(src, out, records(LENGTHS))
    found = {(f, fld) for f, fld, _ in scan_export(out)}
    assert ("data/chunk-000/file-000.parquet", "parquet metadata 'origin'") in found
    assert ("meta/tasks.parquet", "row 0.note") in found


def test_scan_finds_the_home_directory_and_host_name(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/zz/carol")))
    monkeypatch.setattr(hub.socket, "gethostname", lambda: "carols-laptop.local")
    src, out = _export(tmp_path)
    (src / "meta/a.md").write_text("made in /zz/carol/work\n")  # not a known root: by needle
    (src / "meta/b.md").write_text("uid zz-carol-work--episode_000000\n")  # slugged
    (src / "meta/c.md").write_text("host carols-laptop\n")
    export_lerobot(src, out, records(LENGTHS))
    found = {f: what for f, _fld, what in scan_export(out)}
    assert found == {"meta/a.md": "the home directory",
                     "meta/b.md": "the home directory (as a slug)",
                     "meta/c.md": "the host name"}
    assert "carol" not in _refused(out)


def test_scan_skips_what_is_not_uploaded_and_binaries(tmp_path):
    src, out = _export(tmp_path)
    export_lerobot(src, out, records(LENGTHS), include_videos=True)
    (out / ".cache").mkdir()
    (out / ".cache/x.json").write_text(json.dumps({"p": "/Users/x"}))
    (out / "videos/chunk-000/observation.images.main/episode_000000.mp4").write_bytes(
        b"\x00/Users/x\x00")
    assert scan_export(out) == []


def test_sdk_refuses_keep_local_paths_with_a_push(tmp_path):
    from flora_deploy import sdk

    with pytest.raises(ExportError, match="--keep-local-paths"):
        sdk.export_job(None, "j", tmp_path / "out", push_to_hub="me/x", keep_local_paths=True,
                       hub_api=FakeApi())
    assert not (tmp_path / "out").exists()
