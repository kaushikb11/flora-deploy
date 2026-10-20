import json
from dataclasses import dataclass

from typer.testing import CliRunner

from flora_deploy import cli
from flora_deploy.config.pins import RepoPin
from flora_deploy.prep.download import plan_all, plan_snapshot


@dataclass
class Lfs:
    sha256: str


@dataclass
class File:
    path: str
    size: int
    lfs: Lfs | None = None


@dataclass
class Folder:
    path: str
    tree_id: str = "t"


class FakeApi:
    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def list_repo_tree(self, repo_id, *, revision, recursive):
        self.calls.append((repo_id, revision, recursive))
        return list(self.entries)


ENTRIES = [
    Folder("batch_utils"),
    File("batch_utils/pipeline.py", 100),
    File("batch_utils/vendor/fine_temporal.py", 200),
    File("config.json", 10),
    File("model-00001-of-00002.safetensors", 5_000, Lfs("a" * 64)),
    File("model-00002-of-00002.safetensors", 7_000, Lfs("b" * 64)),
    File("LICENSE.pdf", 50),
    File("unrelated.bin", 1),
]


def pin(patterns):
    return RepoPin(key="k", repo="org/repo", revision="0" * 40, role="weights",
                   allow_patterns=patterns)


def test_filters_like_snapshot_download_and_skips_folders():
    plan = plan_snapshot(pin(("config.json", "model-*.safetensors")), FakeApi(ENTRIES))
    assert [f.path for f in plan.files] == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert plan.total_bytes == 12_010
    assert plan.files[1].lfs_sha256 == "a" * 64


def test_directory_glob_matches_nested_files():
    plan = plan_snapshot(pin(("batch_utils/*",)), FakeApi(ENTRIES))
    assert [f.path for f in plan.files] == [
        "batch_utils/pipeline.py",
        "batch_utils/vendor/fine_temporal.py",
    ]


def test_no_patterns_means_whole_snapshot():
    plan = plan_snapshot(pin(None), FakeApi(ENTRIES))
    assert len(plan.files) == 7


def test_queries_pinned_revision_recursively():
    api = FakeApi(ENTRIES)
    plan_all(api, [pin(None)])
    assert api.calls == [("org/repo", "0" * 40, True)]


def test_cli_dry_run_json(monkeypatch):
    monkeypatch.setattr(cli, "_hf_api", lambda: FakeApi(ENTRIES))
    result = CliRunner().invoke(cli.app, ["prepare", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert {r["key"] for r in rows} >= {"tempseg", "captioner", "rewarder"}
