import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from huggingface_hub._local_folder import read_download_metadata, write_download_metadata
from huggingface_hub.utils import filter_repo_objects

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import RECEIPT_NAME, snapshot_path
from flora_deploy.prep.download import plan_snapshot
from flora_deploy.prep.run import (
    PlanMismatchError,
    SnapshotVerificationError,
    prepare_snapshot,
)
from flora_deploy.prep.verify import RECEIPT_SCHEMA, plan_digest

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


BASE = PINS_BY_KEY["tempseg_code"]
PIN = dataclasses.replace(
    BASE, plan_digest=plan_digest(plan_snapshot(BASE, FakeApi()))
)
PINS = {PIN.key: PIN}


class HubLikeDownload:
    """Mimics ``snapshot_download(local_dir=...)``: a file whose huggingface_hub metadata records
    this commit, and that is not newer than the metadata, is returned as is (no fetch, no hash).
    Uses huggingface_hub's own metadata helpers, so the skip rule is the real one.

    ``serve`` returns the bytes the Hub sends for (path, call number)."""

    def __init__(self, serve=None):
        self.calls = []
        self.fetched = []
        self.serve = serve or (lambda path, n: FILES[path])

    def __call__(self, repo_id, *, revision, allow_patterns, local_dir):
        n = len(self.calls)
        self.calls.append((repo_id, revision, allow_patterns, local_dir))
        local = Path(local_dir)
        for path in filter_repo_objects(list(FILES), allow_patterns=allow_patterns):
            dest = local / path
            meta = read_download_metadata(local, path)
            if dest.is_file() and meta is not None and meta.commit_hash == revision:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.serve(path, n))
            write_download_metadata(local, path, commit_hash=revision, etag="e")
            self.fetched.append(path)
        return local_dir


def dest_of(tmp_path):
    return Path(snapshot_path(PIN, tmp_path))


def prep(tmp_path, dl, **kw):
    return prepare_snapshot(PIN.key, root=tmp_path, api=FakeApi(), download=dl, pins=PINS, **kw)


def corrupt_in_place(path):
    """Flip a byte without changing size, inode or mtime (bit rot / storage-level corruption)."""
    st = path.stat()
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    with path.open("r+b") as f:
        f.write(data)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_prepare_downloads_verifies_and_skips_second_time(tmp_path):
    dl, commits = HubLikeDownload(), []
    s = prep(tmp_path, dl, commit=lambda: commits.append(1))
    dest = dest_of(tmp_path)
    assert s["status"] == "prepared" and s["ok"], s
    assert dl.calls == [(PIN.repo, PIN.revision, list(PIN.allow_patterns), str(dest))]
    assert len(commits) == 2
    rec = json.loads((dest / RECEIPT_NAME).read_text())
    assert rec["pin"]["key"] == "tempseg_code" and rec["schema"] == RECEIPT_SCHEMA
    assert "download" in s["timings_s"]

    again = prep(tmp_path, dl)
    assert again["status"] == "skipped" and len(dl.calls) == 1
    # A skipped row reports this run's timing only, not the receipt's old download/hash times.
    assert set(again["timings_s"]) == {"plan"}
    assert again["receipt_created_utc"] == rec["created_utc"]


def test_fake_downloader_does_not_refetch_complete_files(tmp_path):
    # The behaviour that made failures sticky (H2): re-running the download returns the
    # corrupted bytes because the metadata still says the file is complete.
    dl = HubLikeDownload()
    prep(tmp_path, dl)
    target = dest_of(tmp_path) / "README.md"
    corrupt_in_place(target)
    dl(PIN.repo, revision=PIN.revision, allow_patterns=None, local_dir=str(dest_of(tmp_path)))
    assert target.read_bytes() != FILES["README.md"] and dl.fetched == list(FILES)


def test_reverify_repairs_storage_corruption(tmp_path):
    dl = HubLikeDownload()
    prep(tmp_path, dl)
    dest = dest_of(tmp_path)
    corrupt_in_place(dest / "README.md")
    assert prep(tmp_path, dl)["status"] == "skipped"  # the fingerprint can't see bit rot

    s = prep(tmp_path, dl, reverify=True)
    assert s["status"] == "repaired" and s["ok"], s
    assert (dest / "README.md").read_bytes() == FILES["README.md"]
    # Only the failing file was fetched again.
    assert dl.fetched == [*FILES, "README.md"]
    rec = json.loads((dest / RECEIPT_NAME).read_text())
    assert rec["repair"]["removed"] == ["README.md"] and rec["repair"]["ok"]
    assert "git_blob_sha1" in rec["repair"]["first_failures"]


def test_transient_bad_download_is_deleted_and_refetched(tmp_path):
    serve = lambda path, n: FILES[path] + (b"!" if n == 0 and path == "LICENSE.pdf" else b"")  # noqa: E731
    dl = HubLikeDownload(serve)
    s = prep(tmp_path, dl)
    assert s["status"] == "repaired" and s["ok"] and len(dl.calls) == 2
    assert dl.fetched.count("LICENSE.pdf") == 2 and dl.fetched.count("README.md") == 1


def test_extra_file_is_removed_on_repair(tmp_path):
    dl = HubLikeDownload()
    prep(tmp_path, dl)
    stray = dest_of(tmp_path) / "video_preprocessor_config.json"
    stray.write_text("{}")
    s = prep(tmp_path, dl)  # reuse rejected (extra file) -> verify fails -> repaired
    assert s["status"] == "repaired" and s["ok"] and not stray.exists()


def test_persistent_failure_raises_with_clear_message(tmp_path):
    dl = HubLikeDownload(lambda path, n: FILES[path] + b"!")
    with pytest.raises(SnapshotVerificationError) as exc:
        prep(tmp_path, dl)
    msg = str(exc.value)
    assert msg.startswith("tempseg_code: verification failed") and "git_blob_sha1" in msg
    assert len(dl.calls) == 2
    rec = json.loads((dest_of(tmp_path) / RECEIPT_NAME).read_text())
    assert rec["ok"] is False and rec["repair"]["ok"] is False
    # A later run with a healthy Hub repairs it (no manual deletion on the Volume).
    good = HubLikeDownload()
    assert prep(tmp_path, good)["ok"]


def test_plan_digest_mismatch_fails_closed_before_download(tmp_path):
    dl = HubLikeDownload()
    pins = {PIN.key: dataclasses.replace(PIN, plan_digest="0" * 64)}
    with pytest.raises(PlanMismatchError, match="does not match the pinned plan_digest"):
        prepare_snapshot(PIN.key, root=tmp_path, api=FakeApi(), download=dl, pins=pins)
    unpinned = {PIN.key: dataclasses.replace(PIN, plan_digest=None)}
    with pytest.raises(PlanMismatchError):
        prepare_snapshot(PIN.key, root=tmp_path, api=FakeApi(), download=dl, pins=unpinned)
    assert dl.calls == []


def test_old_schema_receipt_is_upgraded_by_rehash_without_refetch(tmp_path):
    dl = HubLikeDownload()
    prep(tmp_path, dl)
    path = dest_of(tmp_path) / RECEIPT_NAME
    old = json.loads(path.read_text())
    old["schema"] = 1
    path.write_text(json.dumps(old))
    s = prep(tmp_path, dl, provenance={"git_commit": "b" * 40, "git_dirty": True})
    assert s["status"] == "prepared" and s["ok"] and "schema 1" in s["note"]
    assert dl.fetched == list(FILES)  # nothing re-downloaded, only re-hashed
    rec = json.loads(path.read_text())
    assert rec["schema"] == RECEIPT_SCHEMA
    assert rec["framework"]["git_commit"] == "b" * 40 and rec["framework"]["git_dirty"] is True
    assert prep(tmp_path, dl)["status"] == "skipped"
