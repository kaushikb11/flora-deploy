import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import RECEIPT_NAME, snapshot_path
from flora_deploy.prep.run import prepare_snapshot

FILES = {"batch_utils/pipeline.py": b"print('hi')\n", "README.md": b"# readme\n",
         "LICENSE.pdf": b"%PDF"}


@dataclass
class Entry:
    path: str
    size: int
    blob_id: str
    lfs: None = None


class FakeApi:
    def list_repo_tree(self, repo_id, *, revision, recursive):
        return [
            Entry(p, len(b), hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest())
            for p, b in FILES.items()
        ]


class FakeDownload:
    def __init__(self, corrupt=False):
        self.calls = []
        self.corrupt = corrupt

    def __call__(self, repo_id, *, revision, allow_patterns, local_dir):
        self.calls.append((repo_id, revision, allow_patterns, local_dir))
        for p, b in FILES.items():
            dest = Path(local_dir) / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b + (b"!" if self.corrupt else b""))
        return local_dir


def test_prepare_downloads_verifies_and_skips_second_time(tmp_path):
    pin = PINS_BY_KEY["tempseg_code"]
    dl, commits = FakeDownload(), []
    s = prepare_snapshot("tempseg_code", root=tmp_path, api=FakeApi(), download=dl,
                         commit=lambda: commits.append(1))
    dest = Path(snapshot_path(pin, tmp_path))
    assert s["status"] == "prepared" and s["ok"], s
    assert dl.calls == [(pin.repo, pin.revision, list(pin.allow_patterns), str(dest))]
    assert len(commits) == 2
    assert json.loads((dest / RECEIPT_NAME).read_text())["pin"]["key"] == "tempseg_code"

    again = prepare_snapshot("tempseg_code", root=tmp_path, api=FakeApi(), download=dl)
    assert again["status"] == "skipped" and len(dl.calls) == 1


def test_prepare_reports_failure_and_retries_next_time(tmp_path):
    bad = FakeDownload(corrupt=True)
    s = prepare_snapshot("tempseg_code", root=tmp_path, api=FakeApi(), download=bad)
    assert s["status"] == "failed" and not s["ok"]
    assert {c["name"] for c in s["failures"]} >= {"sizes", "git_blob_sha1"}
    good = FakeDownload()
    s2 = prepare_snapshot("tempseg_code", root=tmp_path, api=FakeApi(), download=good)
    assert s2["status"] == "prepared" and s2["ok"] and len(good.calls) == 1
