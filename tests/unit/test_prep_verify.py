import dataclasses
import hashlib
import json
import os
import subprocess

import pytest

from flora_deploy import LOGIC_VERSION, __version__
from flora_deploy.config.pins import RepoPin
from flora_deploy.layout import RECEIPT_NAME
from flora_deploy.prep.download import FileEntry, SnapshotPlan
from flora_deploy.prep.verify import (
    RECEIPT_SCHEMA,
    build_receipt,
    check_receipt,
    hash_file,
    load_valid_receipt,
    parse_sha256_manifest,
    plan_digest,
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
    plan = SnapshotPlan(pin, tuple(sorted(entries, key=lambda e: e.path)))
    return with_pin(plan, plan_digest=plan_digest(plan)), snap


def with_pin(plan, **changes):
    return SnapshotPlan(dataclasses.replace(plan.pin, **changes), plan.files)


def repin(plan):
    """Re-pin plan_digest after a test edits the plan's files."""
    return with_pin(plan, plan_digest=plan_digest(plan))


def manifest_digest(snap):
    return sha256((snap / "SOURCE_MANIFEST.sha256").read_bytes())


def failed(result):
    return {c.name for c in result.failures}


def test_clean_snapshot_passes(tmp_path):
    plan, snap = make_snapshot(
        tmp_path,
        expected=(("adapter_sha256", sha256(ADAPTER)), ("source_manifest_identity", "c" * 64)),
    )
    (snap / ".cache" / "huggingface").mkdir(parents=True)
    (snap / ".cache" / "huggingface" / "x.metadata").write_text("ignored")
    plan = with_pin(plan, expected=(*plan.pin.expected, ("manifest_sha256", manifest_digest(snap))))
    r = verify_snapshot(plan, snap, workers=2)
    assert r.ok, r.failures
    names = {c.name for c in r.checks}
    assert {"lfs_sha256", "git_blob_sha1", "manifest:SOURCE_MANIFEST.sha256",
            "manifest_covers:SOURCE_MANIFEST.sha256", "pinned_plan_digest", "readable",
            "expected:adapter_sha256", "expected:manifest_sha256",
            "expected:source_manifest_identity"} <= names
    assert r.bad_files == () and r.extra_files == ()
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
    r = verify_snapshot(repin(SnapshotPlan(plan.pin, files)), snap)
    assert failed(r) == {"manifest:SOURCE_MANIFEST.sha256"}
    assert "config.json" in r.failures[0].detail


def test_required_manifest_absent_from_plan(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    files = tuple(f for f in plan.files if f.path != "SOURCE_MANIFEST.sha256")
    (snap / "SOURCE_MANIFEST.sha256").unlink()
    r = verify_snapshot(repin(SnapshotPlan(plan.pin, files)), snap)
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


# --- receipt reuse (H1) -------------------------------------------------------------------------


def passing_receipt(tmp_path):
    plan, snap = make_snapshot(tmp_path, expected=(("adapter_sha256", sha256(ADAPTER)),))
    rec = build_receipt(verify_snapshot(plan, snap))
    assert rec["ok"], rec["checks"]
    write_receipt(rec, snap)
    assert check_receipt(plan, snap) == (rec, "reused")
    return plan, snap


def flip_byte_keep_stat(path, offset=100):
    """Same-size in-place corruption that also restores mtime (what bit rot looks like)."""
    st = path.stat()
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    with path.open("r+b") as f:  # in place: same inode
        f.write(data)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_same_size_rewrite_rejects_reuse(tmp_path):
    plan, snap = passing_receipt(tmp_path)
    p = snap / "model-00001-of-00001.safetensors"
    st = p.stat()
    data = bytearray(p.read_bytes())
    data[100] ^= 1
    p.write_bytes(bytes(data))  # same size, same inode
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    receipt, why = check_receipt(plan, snap)
    assert receipt is None and "fingerprint changed" in why


def test_replaced_file_rejects_reuse_even_with_same_mtime(tmp_path):
    plan, snap = passing_receipt(tmp_path)
    p = snap / "config.json"
    st = p.stat()
    tmp = snap / "config.json.new"
    tmp.write_bytes(CONFIG.replace(b"5", b"6"))
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(tmp, p)  # like `volume put --force`: new inode, same size and mtime
    receipt, why = check_receipt(plan, snap)
    assert receipt is None and "fingerprint changed" in why


def test_storage_corruption_needs_full_verify(tmp_path):
    # The fingerprint cannot see bit rot (size/mtime/inode unchanged); a full verify can.
    plan, snap = passing_receipt(tmp_path)
    flip_byte_keep_stat(snap / "model-00001-of-00001.safetensors")
    assert check_receipt(plan, snap)[0] is not None
    r = verify_snapshot(plan, snap)
    assert "lfs_sha256" in failed(r)
    assert "model-00001-of-00001.safetensors" in r.bad_files


def test_extra_file_rejects_reuse(tmp_path):
    plan, snap = passing_receipt(tmp_path)
    (snap / "video_preprocessor_config.json").write_text("{}")
    receipt, why = check_receipt(plan, snap)
    assert receipt is None and "video_preprocessor_config.json" in why
    # hf_hub's local_dir metadata under .cache/ is not snapshot content.
    (snap / "video_preprocessor_config.json").unlink()
    (snap / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (snap / ".cache" / "huggingface" / "download" / "config.json.metadata").write_text("x")
    assert check_receipt(plan, snap)[0] is not None


def test_changed_pin_rejects_reuse(tmp_path):
    plan, snap = passing_receipt(tmp_path)
    stale = with_pin(plan, expected=(("adapter_sha256", "f" * 64),))
    receipt, why = check_receipt(stale, snap)
    assert receipt is None and "pin or verifier changed" in why
    assert check_receipt(with_pin(plan, allow_patterns=("*.json",)), snap)[0] is None
    assert check_receipt(with_pin(plan, plan_digest="0" * 64), snap)[0] is None
    assert check_receipt(with_pin(plan, role="base"), snap)[0] is None


def test_old_schema_receipt_is_not_reused(tmp_path):
    plan, snap = passing_receipt(tmp_path)
    rec = json.loads((snap / RECEIPT_NAME).read_text())
    rec["schema"] = 1
    write_receipt(rec, snap)
    receipt, why = check_receipt(plan, snap)
    assert receipt is None and "schema 1" in why
    assert RECEIPT_SCHEMA == 2


# --- pinned anchors (M1) -------------------------------------------------------------------------


def test_pinned_plan_digest_mismatch_fails(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    assert verify_snapshot(plan, snap).ok
    r = verify_snapshot(with_pin(plan, plan_digest="0" * 64), snap)
    assert failed(r) == {"pinned_plan_digest"}
    assert "pinned_plan_digest" in failed(verify_snapshot(with_pin(plan, plan_digest=None), snap))


def test_pinned_manifest_digest(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    good = with_pin(plan, expected=(("manifest_sha256", manifest_digest(snap)),))
    assert verify_snapshot(good, snap).ok
    r = verify_snapshot(with_pin(plan, expected=(("manifest_sha256", "0" * 64),)), snap)
    assert failed(r) == {"expected:manifest_sha256"}
    assert r.bad_files == ("SOURCE_MANIFEST.sha256",)


def _rewrite_manifest(plan, snap, data: bytes):
    m = snap / "SOURCE_MANIFEST.sha256"
    m.write_bytes(data)
    files = tuple(FileEntry(f.path, len(data), None, blob(data)) if f.path == m.name else f
                  for f in plan.files)
    return repin(SnapshotPlan(plan.pin, files))


def test_empty_manifest_fails(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    r = verify_snapshot(_rewrite_manifest(plan, snap, b"\n"), snap)
    assert failed(r) == {"manifest:SOURCE_MANIFEST.sha256"}
    assert "empty" in r.failures[0].detail


def test_manifest_must_cover_safetensors(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    lines = (snap / "SOURCE_MANIFEST.sha256").read_text().splitlines()
    data = "".join(line + "\n" for line in lines if "model-" not in line).encode()
    r = verify_snapshot(_rewrite_manifest(plan, snap, data), snap)
    assert failed(r) == {"manifest_covers:SOURCE_MANIFEST.sha256"}


# --- never raises (L4) ---------------------------------------------------------------------------


def test_bad_provenance_json_is_a_failed_check(tmp_path):
    plan, snap = make_snapshot(tmp_path, expected=(("source_manifest_identity", "c" * 64),))
    p = snap / "MODEL_PROVENANCE.json"
    p.write_bytes(b"[1]")  # valid JSON, not an object
    files = tuple(FileEntry(f.path, 3, None, blob(b"[1]")) if f.path == p.name else f
                  for f in plan.files)
    r = verify_snapshot(repin(SnapshotPlan(plan.pin, files)), snap)
    assert "expected:source_manifest_identity" in failed(r)
    assert "MODEL_PROVENANCE.json" in r.bad_files


def test_unreadable_file_is_a_failed_check(tmp_path, monkeypatch):
    from flora_deploy.prep import verify

    plan, snap = make_snapshot(tmp_path)
    real = verify.hash_file

    def flaky(path, **kw):
        if path.name == "config.json":
            raise OSError("size changed while hashing")
        return real(path, **kw)

    monkeypatch.setattr(verify, "hash_file", flaky)
    r = verify_snapshot(plan, snap)
    assert "readable" in failed(r) and "config.json" in r.bad_files


def test_vanished_file_is_a_failed_check(tmp_path, monkeypatch):
    from flora_deploy.prep import verify

    plan, snap = make_snapshot(tmp_path)
    real = verify.local_files

    def racing(root):
        out = real(root)
        (snap / "config.json").unlink()  # disappears between listing and stat
        return out

    monkeypatch.setattr(verify, "local_files", racing)
    r = verify_snapshot(plan, snap)
    assert {"sizes", "readable"} <= failed(r) and "config.json" in r.bad_files


# --- provenance (M2) -----------------------------------------------------------------------------


def test_receipt_records_provenance_and_fingerprints(tmp_path):
    plan, snap = make_snapshot(tmp_path)
    rec = build_receipt(verify_snapshot(plan, snap),
                        provenance={"git_commit": "a" * 40, "git_dirty": False})
    assert rec["framework"] == {"version": __version__, "logic_version": LOGIC_VERSION,
                                "verifier_version": 2, "git_commit": "a" * 40,
                                "git_dirty": False}
    assert "huggingface_hub" in rec["tools"] and "hf_xet" in rec["tools"]
    assert rec["pin"]["plan_digest"] == plan.pin.plan_digest
    d = next(d for d in rec["digests"] if d["path"] == "config.json")
    st = (snap / "config.json").stat()
    assert (d["size"], d["mtime_ns"], d["ino"], d["dev"]) == (
        st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev)
