import hashlib
import json
import subprocess

import pytest

from flora_deploy.config.pins import RepoPin
from flora_deploy.layout import RECEIPT_NAME
from flora_deploy.prep.download import FileEntry, SnapshotPlan
from flora_deploy.prep.verify import (
    build_receipt,
    hash_file,
    load_valid_receipt,
    parse_sha256_manifest,
    verify_snapshot,
    write_receipt,
)

ADAPTER = b"\x00fake adapter tensors\x01" * 1000
SHARD = b"shard-bytes" * 5000
CONFIG = b'{"model_type": "qwen3_5"}\n'


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def make_snapshot(tmp_path, *, expected=(), extra_files=None, repo="nvidia/Flora-TempSeg"):
    """Write files to tmp_path/snap and return (plan, dir) with correct tree metadata."""
    snap = tmp_path / "snap"
    snap.mkdir()
    lfs = {"adapter_model.safetensors": ADAPTER, "model-00001-of-00001.safetensors": SHARD}
    plain = {"config.json": CONFIG, **(extra_files or {})}
    prov = json.dumps(
        {"adapter_sha256": sha256(ADAPTER), "source_model_identity_sha256": "c" * 64}
    ).encode()
    plain["MODEL_PROVENANCE.json"] = prov
    manifest = "".join(
        f"{sha256(b)}  {name}\n" for name, b in sorted({**lfs, **plain}.items())
    ).encode()
    plain["SOURCE_MANIFEST.sha256"] = manifest
    entries = []
    for name, b in lfs.items():
        (snap / name).write_bytes(b)
        entries.append(FileEntry(name, len(b), sha256(b), blob(b"pointer")))
    for name, b in plain.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_bytes(b)
        entries.append(FileEntry(name, len(b), None, blob(b)))
    pin = RepoPin(key="k", repo=repo, revision="1" * 40, role="weights",
                  allow_patterns=("*",), expected=expected)
    return SnapshotPlan(pin, tuple(sorted(entries, key=lambda e: e.path))), snap


def failed(result):
    return {c.name for c in result.failures}


def test_clean_snapshot_passes(tmp_path):
    plan, snap = make_snapshot(
        tmp_path,
        expected=(("adapter_sha256", sha256(ADAPTER)), ("source_manifest_identity", "c" * 64)),
    )
    (snap / ".cache" / "huggingface").mkdir(parents=True)
    (snap / ".cache" / "huggingface" / "x.metadata").write_text("ignored")
    r = verify_snapshot(plan, snap, workers=2)
    assert r.ok, r.failures
    names = {c.name for c in r.checks}
    assert {"lfs_sha256", "git_blob_sha1", "manifest:SOURCE_MANIFEST.sha256",
            "expected:adapter_sha256", "expected:source_manifest_identity"} <= names
    assert r.total_bytes == plan.total_bytes


def test_git_blob_sha1_matches_git(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(CONFIG)
    out = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert hash_file(p, git_blob=True, chunk=7)[1] == out
    assert hash_file(p, git_blob=False)[0] == sha256(CONFIG)


def test_lfs_sha_mismatch(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    data = bytearray(SHARD)
    data[0] ^= 1
    (snap / "model-00001-of-00001.safetensors").write_bytes(bytes(data))
    r = verify_snapshot(plan, snap)
    assert {"lfs_sha256", "manifest:SOURCE_MANIFEST.sha256"} <= failed(r)
    assert "sizes" not in failed(r)


def test_blob_sha_mismatch(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    (snap / "config.json").write_bytes(CONFIG.replace(b"5", b"6"))
    assert "git_blob_sha1" in failed(verify_snapshot(plan, snap))


def test_missing_file_and_extra_file(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    (snap / "config.json").unlink()
    (snap / "stray.bin").write_bytes(b"x")
    r = verify_snapshot(plan, snap)
    assert {"files_present", "no_extra_files", "manifest:SOURCE_MANIFEST.sha256"} <= failed(r)


def test_manifest_mismatch_only(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    m = snap / "SOURCE_MANIFEST.sha256"
    m.write_text(m.read_text().replace(sha256(CONFIG), "0" * 64))
    # Rewrite the plan's blob id so only the manifest check can catch it.
    data = m.read_bytes()
    files = tuple(
        FileEntry(f.path, len(data), None, blob(data)) if f.path == m.name else f
        for f in plan.files
    )
    r = verify_snapshot(SnapshotPlan(plan.pin, files), snap)
    assert failed(r) == {"manifest:SOURCE_MANIFEST.sha256"}
    assert "config.json" in r.failures[0].detail


def test_required_manifest_absent_from_plan(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    files = tuple(f for f in plan.files if f.path != "SOURCE_MANIFEST.sha256")
    (snap / "SOURCE_MANIFEST.sha256").unlink()
    r = verify_snapshot(SnapshotPlan(plan.pin, files), snap)
    assert failed(r) == {"manifest:SOURCE_MANIFEST.sha256"}


def test_rewarder_sha256sums(tmp_path):
    plan, snap = make_snapshot(tmp_path, repo="nvidia/Flora-Rewarder")
    # Rewarder requires SHA256SUMS, which this plan lacks.
    assert "manifest:SHA256SUMS" in failed(verify_snapshot(plan, snap))


def test_expected_digest_mismatch(tmp_path):
    plan, snap = make_snapshot(
        tmp_path,
        expected=(("adapter_sha256", "f" * 64), ("source_manifest_identity", "d" * 64),
                  ("mystery", "0" * 64)),
    )
    assert failed(verify_snapshot(plan, snap)) == {
        "expected:adapter_sha256", "expected:source_manifest_identity", "expected:mystery"}


def test_parse_manifest_rejects_escapes():
    assert parse_sha256_manifest(f"{'a' * 64} *x/y.bin\n\n") == [("a" * 64, "x/y.bin")]
    for bad in (f"{'a' * 64}  ../etc", f"{'a' * 64}  /abs", "zz  file", "onlyonefield"):
        with pytest.raises(ValueError):
            parse_sha256_manifest(bad)


def test_receipt_contents_and_reuse(tmp_path):
    plan, snap = make_snapshot(tmp_path, expected=(("adapter_sha256", sha256(ADAPTER)),))
    r = verify_snapshot(plan, snap)
    path = write_receipt(build_receipt(r, timings={"download": 1.5}), snap)
    assert path == snap / RECEIPT_NAME
    rec = json.loads(path.read_text())
    assert rec["ok"] is True
    assert rec["pin"]["revision"] == "1" * 40 and rec["pin"]["repo"] == "nvidia/Flora-TempSeg"
    assert rec["files"] == len(plan.files) and rec["bytes"] == plan.total_bytes
    assert rec["timings_s"]["download"] == 1.5 and "hash" in rec["timings_s"]
    assert "python" in rec["tools"] and "huggingface_hub" in rec["tools"]
    assert all(c["ok"] for c in rec["checks"])
    assert {d["path"]: d["sha256"] for d in rec["digests"]}["adapter_model.safetensors"] == \
        sha256(ADAPTER)
    # The receipt itself is not an "extra file" on re-verification.
    assert verify_snapshot(plan, snap).ok
    assert load_valid_receipt(plan, snap) == rec

    other_rev = SnapshotPlan(RepoPin(**{**plan.pin.__dict__, "revision": "2" * 40}), plan.files)
    assert load_valid_receipt(other_rev, snap) is None
    (snap / "config.json").write_bytes(b"truncated")
    assert load_valid_receipt(plan, snap) is None


def test_failed_receipt_is_not_reused(tmp_path):
    plan, snap = make_snapshot(tmp_path, expected=(("adapter_sha256", "f" * 64),))
    write_receipt(build_receipt(verify_snapshot(plan, snap)), snap)
    assert load_valid_receipt(plan, snap) is None
