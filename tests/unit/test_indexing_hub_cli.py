"""``flora index``: Hub datasets through a fake hub (the Modal path's file handling) + the CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.index import store_for_sink
from flora_deploy.jobs import indexing as I
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.jobs.types import episode_uid
from flora_deploy.media import synth
from flora_deploy.media.synth import Spec

REV = "0123456789abcdef0123456789abcdef01234567"
KEYS = ("observation.images.top", "observation.images.left", "observation.images.right")
CMAP = dict(zip(KEYS, ("main", "left_wrist", "right_wrist"), strict=True))


class FakeHub:
    """Serves a local tree as ``repo`` at ``REV``; counts downloads."""

    def __init__(self, tree: Path):
        self.tree, self.downloads = Path(tree), []

    def download(self, repo, revision, filename, local_dir):
        assert revision == REV
        src = self.tree / filename
        if not src.is_file():
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        self.downloads.append(filename)
        return dest

    def list_files(self, repo, revision, prefix):
        base = self.tree / prefix if prefix else self.tree
        return sorted(p.relative_to(self.tree).as_posix() for p in base.rglob("*")
                      if p.is_file())


def _hub_job(fmt, subdir="", cameras=None, pool="yam"):
    spec = I.DatasetSpec(format=fmt, repo="org/ds", revision=REV, subdir=subdir,
                         cameras=cameras)
    return I.IndexJob(job_id="hub-job", dataset=spec, pool=pool, decode=False, revision=REV)


def test_hub_v3_downloads_only_needed_files_and_releases(tmp_path):
    tree = tmp_path / "repo"
    eps = [{k: Spec(n, seed=e * 10 + c) for c, k in enumerate(KEYS)}
           for e, n in enumerate([8, 6])]
    synth.lerobot_v3_tree(tree / "sub", eps, ["t"])
    (tree / "sub/data").mkdir()
    (tree / "sub/data/huge.parquet").write_bytes(b"x")  # never fetched
    hub = FakeHub(tree)
    job = _hub_job("lerobot_v3", subdir="sub", cameras=CMAP)
    files = I.open_dataset(job.dataset, work=tmp_path / "work", hub=hub)
    store = LocalStore(tmp_path / "store")
    s = I.run_index(job, store, files, log=lambda m: None)
    assert s["counts"]["indexed"] == 2 and s["dataset"]["revision"] == REV
    assert not any("data/" in d for d in hub.downloads)
    videos = [d for d in hub.downloads if d.endswith(".mp4")]
    assert len(videos) == 3  # one per camera file, for both episodes
    # not staged: a unit's downloads are deleted when it is done; metadata stays
    assert not list((tmp_path / "work").rglob("*.mp4"))
    rec = json.loads(store.get_bytes(
        I.JobPaths("hub-job").index(episode_uid("org/ds/sub", "episode_000001"))))
    assert rec["streams"]["main"]["source"]["uri"] == (
        "videos/observation.images.top/chunk-000/file-000.mp4")
    assert rec["dataset"] == {"name": "org/ds/sub", "format": "lerobot_v3", "repo": "org/ds",
                              "revision": REV, "subdir": "sub"}
    # resume needs no download at all
    hub.downloads.clear()
    s = I.run_index(job, store, files, log=lambda m: None)
    assert s["counts"]["resumed"] == 2 and not [d for d in hub.downloads if d.endswith(".mp4")]


def test_hub_staged_keeps_files_and_missing_file_is_source_error(tmp_path):
    tree = tmp_path / "repo"
    synth.lerobot_v21_tree(tree, [{"cam": Spec(5)}, {"cam": Spec(6, seed=1)}], ["t"])
    (tree / "videos/chunk-000/cam/episode_000001.mp4").unlink()
    hub = FakeHub(tree)
    job = _hub_job("lerobot_v21")
    files = I.open_dataset(job.dataset, work=tmp_path / "stage", hub=hub, staged=True)
    s = I.run_index(job, LocalStore(tmp_path / "store"), files, log=lambda m: None)
    assert s["counts"]["indexed"] == 1
    assert s["source_errors"] == {"missing_file": 1} and s["counts"]["failed"] == 0
    assert len(list((tmp_path / "stage").rglob("*.mp4"))) == 1


def test_hub_mp4_dir_lists_names_without_downloading(tmp_path):
    tree = tmp_path / "repo"
    synth.encode(tree / "clips/a.mp4", Spec(5))
    (tree / "clips/a.txt").write_text("fold")
    synth.encode(tree / "clips/b.mp4", Spec(4, seed=2))
    hub = FakeHub(tree)
    job = _hub_job("mp4_dir", subdir="clips", pool="human")
    files = I.open_dataset(job.dataset, work=tmp_path / "work", hub=hub)
    eps = files.list_episodes()
    assert [e["episode_id"] for e in eps] == ["a", "b"] and eps[0]["task"] == "fold"
    assert eps[1]["cameras"] == {"main": ["b.mp4", None, None]}
    assert not any(d.endswith(".mp4") for d in hub.downloads)
    s = I.run_index(job, LocalStore(tmp_path / "store"), files, log=lambda m: None)
    assert s["counts"]["indexed"] == 2


def test_failed_episode_writes_no_receipt(tmp_path, monkeypatch):
    tree = tmp_path / "repo"
    synth.lerobot_v21_tree(tree, [{"cam": Spec(5)}], ["t"])
    hub = FakeHub(tree)
    job = _hub_job("lerobot_v21")
    files = I.open_dataset(job.dataset, work=tmp_path / "work", hub=hub)
    store = LocalStore(tmp_path / "store")

    def boom(*a, **k):
        raise ConnectionError("hub down")

    monkeypatch.setattr(files, "ensure", boom)
    s = I.run_index(job, store, files, log=lambda m: None)
    assert s["counts"]["failed"] == 1 and "hub down" in s["failed"][0]["error"]
    assert [p for p in store.list("flora/hub-job/index/") if "summary" not in p] == []


# ---------------------------------------------------------------------------------------------
# CLI

CONFIG = """\
project: idx-test
license_ack: research-only
modal: {{max_gpus: 2, budget_usd: 5}}
preset: exact
input: {{format: lerobot_v21, path: ./ds}}
pipeline:
  stages: [segment, caption]
  embodiment: yam
  envelope: {envelope}
output: {{sink: "file://{sink}"}}
"""


@pytest.mark.parametrize("envelope,planned", [("warn", 2), ("enforce", 1)])
def test_cli_local(tmp_path, envelope, planned):
    synth.lerobot_v21_tree(tmp_path / "ds", [{"cam": Spec(5)}, {"cam": Spec(1, seed=1)}],
                           ["t"])
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(CONFIG.format(envelope=envelope, sink=tmp_path / "store"))
    out = tmp_path / "summary.json"
    r = CliRunner().invoke(app, ["index", "-c", str(cfg), "--local", "--out", str(out)],
                           env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    assert "2 episodes, 2 indexed" in r.output and f"planned {planned}" in r.output
    s = json.loads(out.read_text())
    assert s["job_id"] == "idx-test" and s["counts"]["planned"] == planned
    assert s["dataset"]["name"] == "ds"  # normalized by config (Fix-CFG); uids follow the config, not the cwd
    store = LocalStore(tmp_path / "store")
    assert store.exists(I.JobPaths("idx-test").index(episode_uid("ds", "episode_000000")))
    # the summary feeds `flora plan --index`
    r = CliRunner().invoke(app, ["plan", "-c", str(cfg), "--index", str(out), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["workload"]["episodes"] == planned


def test_cli_rejects_local_dataset_on_modal(tmp_path):
    (tmp_path / "ds").mkdir()
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(CONFIG.format(envelope="warn", sink=tmp_path / "store"))
    r = CliRunner().invoke(app, ["index", "-c", str(cfg)])
    assert r.exit_code == 2 and "--local" in r.output


def test_store_for_sink():
    assert store_for_sink("volume://flora-canary/x") == "modal-volume://flora-canary/x"
    assert store_for_sink("s3://b/p") == "s3://b/p"
    with pytest.raises(ValueError):
        store_for_sink("gs://b/p")
