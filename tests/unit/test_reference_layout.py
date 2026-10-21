"""Offline model-cache layout for NVIDIA's ``Session.load`` (no Modal, no NVIDIA code)."""

import json
import os
from pathlib import Path

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import RECEIPT_NAME, snapshot_path
from flora_deploy.stages import reference as ref


def _fake_snapshot(weights: Path, key: str, *, ok=True, tree=True) -> Path:
    pin = PINS_BY_KEY[key]
    d = Path(snapshot_path(pin, weights))
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1000)
    (d / RECEIPT_NAME).write_text(
        json.dumps({"ok": ok, "pin": {"key": key, "revision": pin.revision}})
    )
    meta = d / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text(f"{pin.revision}\netag\n1.0\n")
    (meta / "config.json.lock").write_text("")
    os.utime(d / "config.json", ns=(1_000_000_000, 1_000_000_000))
    if tree:
        t = ref.tree_listing(d, pin.revision)
        t.parent.mkdir(parents=True)
        t.write_text("[]")
    return d


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
