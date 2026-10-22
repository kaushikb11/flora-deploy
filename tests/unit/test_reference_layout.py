"""Offline model-cache layout for NVIDIA's ``Session.load`` (no Modal, no NVIDIA code)."""

import json
import os
from pathlib import Path

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import RECEIPT_NAME, snapshot_path
from flora_deploy.stages import reference as ref


def write_receipt(d: Path, pin, *, ok=True, **override) -> dict:
    """A receipt shaped like ``prep.verify.build_receipt`` for the files in ``d``."""
    from flora_deploy.prep import verify

    digests = []
    for rel in sorted(verify.local_files(d)):
        size, mtime_ns, ino, dev = verify.fingerprint(d / rel)
        sha = verify.hash_file(d / rel, git_blob=False)[0]
        digests.append({"path": rel, "size": size, "sha256": sha, "mtime_ns": mtime_ns,
                        "ino": ino, "dev": dev})
    receipt = {"schema": verify.RECEIPT_SCHEMA, "ok": ok, "pin": verify.pin_record(pin),
               "pin_digest": verify.pin_digest(pin), "plan_digest": pin.plan_digest,
               "checks": [{"name": "x", "ok": ok, "detail": ""}], "digests": digests,
               **override}
    (d / RECEIPT_NAME).write_text(json.dumps(receipt))
    return receipt


def _fake_snapshot(weights: Path, key: str, *, ok=True, tree=True) -> Path:
    pin = PINS_BY_KEY[key]
    d = Path(snapshot_path(pin, weights))
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1000)
    meta = d / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text(f"{pin.revision}\netag\n1.0\n")
    (meta / "config.json.lock").write_text("")
    os.utime(d / "config.json", ns=(1_000_000_000, 1_000_000_000))
    if tree:
        t = ref.tree_listing(d, pin.revision)
        t.parent.mkdir(parents=True)
        t.write_text("[]")
    write_receipt(d, pin, ok=ok)
    return d


REAL_EXPECTED = {"test_receipt_check_derives_expected_digests"}


@pytest.fixture(autouse=True)
def fake_expected(request, monkeypatch):
    """The fake snapshots have no adapter / NVIDIA manifest: derive pins.py's expected digests
    as matching, except in the tests that check the real derivation."""
    if request.node.name in REAL_EXPECTED:
        return
    from flora_deploy.prep import verify

    monkeypatch.setattr(verify, "_EXPECTED", {
        k: (lambda key: lambda pin, root, sha: (dict(pin.expected)[key], "x"))(k)
        for k in verify._EXPECTED})


@pytest.fixture
def weights(tmp_path):
    w = tmp_path / "weights"
    for key in ("tempseg", "captioner", "tempseg_base"):
        _fake_snapshot(w, key)
    return w


def test_layout_copies_models_and_links_base(tmp_path, weights):
    lay = ref.build_offline_layout(tmp_path / "off", weights)
    for kind in ("tempseg", "captioner"):
        dest = lay.model_cache / kind
        src = Path(snapshot_path(PINS_BY_KEY[kind], weights))
        assert dest.is_dir() and not dest.is_symlink()  # hub takes filelocks inside local_dir
        assert lay.paths[kind] == dest and lay.sources[kind] == src
        # mtimes survive, so hub's download metadata stays "current"
        assert (dest / "config.json").stat().st_mtime_ns == 1_000_000_000
        assert (dest / ".cache/huggingface/download/config.json.metadata").is_file()
        assert ref.tree_listing(dest, PINS_BY_KEY[kind].revision).is_file()
        assert not (dest / ".cache/huggingface/download/config.json.lock").exists()
        # every file is a real file under the copy: validate_manifest's allowed_root holds
        for f in dest.rglob("*"):
            assert not f.is_symlink() and f.resolve().is_relative_to(dest.resolve())
        assert lay.copies[kind]["copied_files"] == lay.copies[kind]["files"] == 5
    base = PINS_BY_KEY["tempseg_base"]
    storage = lay.hub_cache / "models--Qwen--Qwen3.5-9B"
    assert (storage / "snapshots" / base.revision).is_symlink()
    assert (storage / "snapshots" / base.revision / "config.json").is_file()
    assert (storage / "trees" / f"{base.revision}.json").is_file()
    assert lay.paths["tempseg_base"] == storage / "snapshots" / base.revision


def test_layout_is_idempotent_and_refuses_foreign_links(tmp_path, weights):
    root = tmp_path / "off"
    ref.build_offline_layout(root, weights)
    again = ref.build_offline_layout(root, weights)
    assert again.copies["captioner"]["copied_files"] == 0  # nothing re-copied
    base_link = root / "hf/hub/models--Qwen--Qwen3.5-9B/snapshots" / PINS_BY_KEY[
        "tempseg_base"].revision
    os.unlink(base_link)
    base_link.symlink_to(tmp_path)
    with pytest.raises(ref.OfflineLayoutError, match="already points"):
        ref.build_offline_layout(root, weights)


def test_layout_kinds_subset(tmp_path, weights):
    lay = ref.build_offline_layout(tmp_path / "off", weights, kinds=("tempseg",))
    assert set(lay.copies) == {"tempseg"}
    assert not (lay.model_cache / "captioner").exists()
    with pytest.raises(ValueError):
        ref.build_offline_layout(tmp_path / "off", weights, kinds=("rewarder",))


def test_layout_refuses_symlinked_model_dir(tmp_path, weights):
    root = tmp_path / "off"
    (root / "models").mkdir(parents=True)
    (root / "models" / "tempseg").symlink_to(snapshot_path(PINS_BY_KEY["tempseg"], weights))
    with pytest.raises(ref.OfflineLayoutError, match="writable copy"):
        ref.build_offline_layout(root, weights)


def test_layout_env_matches_nvidia_names(tmp_path, weights):
    env = ref.build_offline_layout(tmp_path / "off", weights).env()
    assert env["FLORA_MODEL_CACHE"] == str(tmp_path / "off" / "models")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "off" / "hf" / "hub")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert ref.offline_env()["FLORA_MODEL_CACHE"] == "/tmp/flora-offline/models"


@pytest.mark.parametrize("problem", ["missing", "bad_receipt", "no_tree"])
def test_layout_rejects_unprepared_snapshots(tmp_path, problem):
    w = tmp_path / "weights"
    _fake_snapshot(w, "tempseg")
    _fake_snapshot(w, "tempseg_base")
    if problem == "bad_receipt":
        _fake_snapshot(w, "captioner", ok=False)
    elif problem == "no_tree":
        _fake_snapshot(w, "captioner", tree=False)
    with pytest.raises(ref.OfflineLayoutError):
        ref.build_offline_layout(tmp_path / "off", w)


def test_receipts_optional(tmp_path):
    w = tmp_path / "weights"
    for key in ("tempseg", "captioner", "tempseg_base"):
        d = _fake_snapshot(w, key, ok=False)
        (d / RECEIPT_NAME).unlink()
    ref.build_offline_layout(tmp_path / "off", w, require_receipts=False)


def _models():
    p = PINS_BY_KEY
    return {
        "tempseg": {"repo": p["tempseg"].repo, "revision": p["tempseg"].revision},
        "captioner": {"repo": p["captioner"].repo, "revision": p["captioner"].revision},
        "tempseg_base": {"repo": p["tempseg_base"].repo, "revision": p["tempseg_base"].revision},
    }


def test_check_nv_models():
    ref.check_nv_models(_models(), environ={})
    bad = _models()
    bad["captioner"]["revision"] = "0" * 40
    with pytest.raises(ref.OfflineLayoutError, match="captioner"):
        ref.check_nv_models(bad, environ={})
    with pytest.raises(ref.OfflineLayoutError, match="FLORA_TEMPSEG_REPO"):
        ref.check_nv_models(_models(), environ={"FLORA_TEMPSEG_REPO": "someone/else"})


def test_nvidia_models_json_matches_pins(nv_code):
    """The real models.json of the code pin names exactly our weight pins."""
    models = json.loads((nv_code("tempseg_code") / "batch_utils" / "models.json").read_text())
    ref.check_nv_models(models, environ={})


def test_full_receipt_check(tmp_path, weights):
    pin = PINS_BY_KEY["tempseg_base"]
    d = Path(snapshot_path(pin, weights))
    rep = ref.check_prep_receipt(pin, d)
    assert rep["files"] == 2 and rep["fingerprints"] == ["size", "mtime_ns"]
    assert rep["hashed_files"] == 2  # both are small
    # the old shallow check accepted any receipt with ok + revision; these must all fail
    good = json.loads((d / RECEIPT_NAME).read_text())
    for bad, why in [({"schema": 3}, "schema"), ({"pin_digest": "0" * 64}, "pin_digest"),
                     ({"plan_digest": "0" * 64}, "plan_digest"),
                     ({"checks": [{"name": "x", "ok": False}]}, "passing"),
                     ({"pin": {**good["pin"], "allow_patterns": ["*"]}}, "pin record")]:
        (d / RECEIPT_NAME).write_text(json.dumps({**good, **bad}))
        with pytest.raises(ref.OfflineLayoutError, match=why):
            ref.check_prep_receipt(pin, d)
    (d / RECEIPT_NAME).write_text(json.dumps(good))
    # a rewritten file (same size, new mtime) and an extra file are caught
    f = d / "model-00001-of-00002.safetensors"
    os.utime(f, ns=(5, 5))
    with pytest.raises(ref.OfflineLayoutError, match="fingerprint changed"):
        ref.check_prep_receipt(pin, d)
    assert ref.check_prep_receipt(pin, d, fields=("size",))["files"] == 2
    (d / "extra.json").write_text("{}")
    with pytest.raises(ref.OfflineLayoutError, match="file set"):
        ref.check_prep_receipt(pin, d, fields=("size",))


def test_worker_receipt_check_survives_inode_change(weights):
    """The exact worker's startup check (default fields) on a fresh Volume mount: new inode
    numbers are accepted, a changed size or mtime is not."""
    pin = PINS_BY_KEY["tempseg"]
    d = Path(snapshot_path(pin, weights))
    rec = json.loads((d / RECEIPT_NAME).read_text())
    for e in rec["digests"]:
        e["ino"] += 1_000_003
        e["dev"] += 7
    (d / RECEIPT_NAME).write_text(json.dumps(rec))
    assert ref.check_prep_receipt(pin, d)["files"] == 2
    f = d / "config.json"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    with pytest.raises(ref.OfflineLayoutError, match=r"fingerprint changed \(mtime_ns\)"):
        ref.check_prep_receipt(pin, d)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    f.write_text("{ }")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ref.OfflineLayoutError, match=r"fingerprint changed \(size\)"):
        ref.check_prep_receipt(pin, d)


def test_legacy_schema1_receipt(tmp_path, weights):
    """WS-A's schema-1 receipts (on the Volume today) carry sha256 + size only."""
    pin = PINS_BY_KEY["tempseg_base"]
    d = Path(snapshot_path(pin, weights))
    good = json.loads((d / RECEIPT_NAME).read_text())
    legacy = {**good, "schema": 1, "pin": {k: v for k, v in good["pin"].items()
                                           if k != "plan_digest"}}
    legacy.pop("pin_digest")
    legacy["digests"] = [{k: x[k] for k in ("path", "size", "sha256")} for x in good["digests"]]
    (d / RECEIPT_NAME).write_text(json.dumps(legacy))
    rep = ref.check_prep_receipt(pin, d, hash_files="all")
    assert rep["schema"] == 1 and rep["fingerprints"] == ["size"] and rep["hashed_files"] == 2
    # same size, different bytes: only a hash catches it on a legacy receipt
    f = d / "model-00001-of-00002.safetensors"
    f.write_bytes(b"y" * 1000)
    assert ref.check_prep_receipt(pin, d, hash_files="none")["files"] == 2
    with pytest.raises(ref.OfflineLayoutError, match="sha256"):
        ref.check_prep_receipt(pin, d, hash_files="all")
    (d / RECEIPT_NAME).write_text(json.dumps({**legacy, "schema": 0}))
    with pytest.raises(ref.OfflineLayoutError, match="schema"):
        ref.check_prep_receipt(pin, d)


def test_receipt_check_derives_expected_digests(tmp_path, weights):
    pin = PINS_BY_KEY["tempseg"]  # expects adapter_sha256 + manifest_sha256
    with pytest.raises(ref.OfflineLayoutError, match="expected"):
        ref.check_prep_receipt(pin, Path(snapshot_path(pin, weights)))
