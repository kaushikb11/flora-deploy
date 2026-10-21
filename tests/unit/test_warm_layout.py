"""No-copy model cache (``warm.build_linked_layout``) against real ``huggingface_hub`` offline
resolution and, with the NVIDIA code pin fetched, NVIDIA's own ``snapshot``/``validate_manifest``.

The fake "Volume" is made read-only (directories 0o555), so anything that tries to write into it
(``huggingface_hub``'s filelock) fails the way the read-only Modal Volume does.
"""

import hashlib
import importlib
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import RECEIPT_NAME, snapshot_path
from flora_deploy.modal_app import warm
from flora_deploy.stages import reference as ref

# NVIDIA models.snapshot()'s allow_patterns (models.py:43): only files it asks for.
PAYLOAD = {
    "config.json": b"{}",
    "model-00001-of-00002.safetensors": b"a" * 4000,
    "model-00002-of-00002.safetensors": b"b" * 3000,
    "README.md": b"readme",
}


def _prepare_like_prep(snap: Path, pin) -> None:
    """What ``flora prepare`` leaves on the Volume: payload, NVIDIA manifest, receipt, and the
    hub's own ``local_dir`` bookkeeping (tree listing + per-file download metadata)."""
    from huggingface_hub._local_folder import write_download_metadata
    from huggingface_hub._tree_cache import (
        TreeCacheEntry,
        tree_cache_folder_for_local_dir,
        write_tree_cache,
    )

    snap.mkdir(parents=True)
    for name, data in PAYLOAD.items():
        (snap / name).write_bytes(data)
    manifest = "".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in PAYLOAD.items())
    (snap / "SOURCE_MANIFEST.sha256").write_text(manifest)
    (snap / RECEIPT_NAME).write_text(json.dumps({"ok": True, "pin": {"revision": pin.revision}}))
    files = [*PAYLOAD, "SOURCE_MANIFEST.sha256"]
    entries = {n: TreeCacheEntry(size=(snap / n).stat().st_size, blob_id="0" * 40) for n in files}
    write_tree_cache(tree_cache_folder_for_local_dir(str(snap)), pin.revision, entries)
    for n in files:
        write_download_metadata(snap, n, pin.revision, "etag")
    for lock in snap.rglob("*.lock"):
        lock.unlink()


def _read_only(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_dir() and not p.is_symlink():
            p.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    root.chmod(0o555)


@pytest.fixture
def volume(tmp_path):
    """``tmp/flora/weights`` laid out like flora-weights; made read-only, restored afterwards."""
    weights = tmp_path / "flora" / "weights"
    for key in ("tempseg", "captioner", "tempseg_base"):
        pin = PINS_BY_KEY[key]
        _prepare_like_prep(Path(snapshot_path(pin, weights)), pin)
    _read_only(weights)
    yield weights
    for p in [weights, *weights.rglob("*")]:
        if p.is_dir() and not p.is_symlink():
            p.chmod(0o755)


def test_link_farm_shape(volume):
    root = volume.parent
    lay = warm.build_linked_layout(root, volume, hub_root=root / "hub")
    assert lay.model_cache == root / "snapshots"
    for kind in ("tempseg", "captioner"):
        d = lay.paths[kind]
        assert d == root / "snapshots" / kind and d.is_dir() and not d.is_symlink()
        src = lay.sources[kind]
        for name in [*PAYLOAD, "SOURCE_MANIFEST.sha256", RECEIPT_NAME]:
            assert (d / name).is_symlink() and (d / name).resolve() == (src / name).resolve()
        meta = d / ".cache" / "huggingface"
        assert not any(p.is_symlink() for p in meta.rglob("*"))  # real, writable copies
        m_src = src / ".cache" / "huggingface" / "download" / "config.json.metadata"
        m_dst = meta / "download" / "config.json.metadata"
        assert m_dst.read_text() == m_src.read_text()
        assert m_dst.stat().st_mtime_ns == m_src.stat().st_mtime_ns
        assert not list(meta.rglob("*.lock"))
        assert lay.stats[kind]["linked_bytes"] >= 7000
    base = lay.paths["tempseg_base"]
    assert base.is_symlink() and base.resolve() == lay.sources["tempseg_base"].resolve()
    assert lay.env()["FLORA_MODEL_CACHE"] == str(root / "snapshots")
    # idempotent
    again = warm.build_linked_layout(root, volume, hub_root=root / "hub")
    assert again.paths == lay.paths


def test_weights_outside_root_rejected(volume, tmp_path):
    with pytest.raises(ref.OfflineLayoutError, match="must be inside"):
        warm.build_linked_layout(tmp_path / "elsewhere", volume, hub_root=tmp_path)


def test_linked_env_matches_offline_env_except_model_cache():
    a, b = warm.linked_offline_env(), ref.offline_env(ref.OFFLINE_ROOT)
    assert a["FLORA_MODEL_CACHE"] == "/snapshots"  # Modal: allowed_root must be "/"
    assert {k: v for k, v in a.items() if k != "FLORA_MODEL_CACHE"} == {
        k: v for k, v in b.items() if k != "FLORA_MODEL_CACHE"
    }


def test_volume_behind_a_symlinked_mount(volume, tmp_path):
    """Modal mounts a Volume as ``/weights -> /__modal/volumes/<id>``: the root must contain the
    *real* path, not the mount point."""
    mount = tmp_path / "mnt" / "weights"
    mount.parent.mkdir()
    mount.symlink_to(volume, target_is_directory=True)
    with pytest.raises(ref.OfflineLayoutError, match="really"):
        warm.build_linked_layout(tmp_path / "mnt", mount, hub_root=tmp_path / "hub")
    lay = warm.build_linked_layout(tmp_path, mount, hub_root=tmp_path / "hub")
    assert lay.stats["captioner"]["checked_links"] >= len(PAYLOAD)


def test_check_links_rejects_escapes(volume, tmp_path):
    lay = warm.build_linked_layout(volume.parent, volume, hub_root=volume.parent / "hub")
    src, dest = lay.sources["captioner"], lay.paths["captioner"]
    assert warm.check_links(src, dest) == len(PAYLOAD) + 2  # + manifest + receipt
    other = lay.sources["tempseg"] / "config.json"  # inside the Volume, but another snapshot
    (dest / "config.json").unlink()
    (dest / "config.json").symlink_to(other)
    with pytest.raises(OSError, match="leaves the snapshot"):
        warm.check_links(src, dest)
    (dest / "config.json").unlink()
    (dest / "config.json").symlink_to(src / "does-not-exist")
    with pytest.raises(OSError, match="leaves the snapshot"):
        warm.check_links(src, dest)


def _nv_snapshot_call(local_dir: Path, pin) -> str:
    """NVIDIA's ``models.snapshot`` call (models.py:39-46), verbatim arguments."""
    from huggingface_hub import snapshot_download

    patterns = ["adapter_config.json", "adapter_model.safetensors", "config.json",
                "generation_config.json", "model.safetensors.index.json", "model-*.safetensors",
                "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "README.md", "MODEL_PROVENANCE.json",
                "SOURCE_MANIFEST.sha256"]
    return snapshot_download(pin.repo, revision=pin.revision, token=None,
                             allow_patterns=patterns, local_dir=local_dir)


@pytest.fixture
def offline(monkeypatch):
    import huggingface_hub.constants as hfc

    monkeypatch.setattr(hfc, "HF_HUB_OFFLINE", True)


def test_hub_offline_resolution_through_links(volume, offline):
    lay = warm.build_linked_layout(volume.parent, volume, hub_root=volume.parent / "hub")
    pin = PINS_BY_KEY["captioner"]
    got = _nv_snapshot_call(lay.paths["captioner"], pin)
    assert Path(got) == lay.paths["captioner"].resolve()
    # the hub's locks landed in the writable link farm, never on the "Volume"
    assert list((lay.paths["captioner"] / ".cache" / "huggingface").rglob("*.lock"))
    assert not list(lay.sources["captioner"].rglob("*.lock"))


def test_read_only_volume_symlink_fails_like_modal(volume, offline, tmp_path):
    """The WS-D finding this layout works around: local_dir -> read-only Volume fails."""
    pin = PINS_BY_KEY["captioner"]
    link = tmp_path / "direct" / "captioner"
    link.parent.mkdir()
    link.symlink_to(Path(snapshot_path(pin, volume)), target_is_directory=True)
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    with pytest.raises(OSError):
        _nv_snapshot_call(link, pin)


# --- NVIDIA's own code (needs scripts/fetch_nv_code.py) -----------------------------------


@pytest.fixture
def nv_models(nv_code, volume, monkeypatch):
    """NVIDIA's ``batch_utils.models`` imported fresh (without the package ``__init__``) with
    ``FLORA_MODEL_CACHE`` pointing at the link farm."""
    monkeypatch.setenv("FLORA_MODEL_CACHE", str(volume.parent / "snapshots"))
    for var in ref.NV_REPO_ENV:
        monkeypatch.delenv(var, raising=False)
    before = set(sys.modules)
    name = "_flora_warm_test_segcap"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(nv_code("tempseg_code") / "batch_utils")]
    sys.modules[name] = pkg
    models = importlib.import_module(f"{name}.models")
    yield models
    for m in set(sys.modules) - before:
        del sys.modules[m]


@pytest.mark.nvlink
def test_nvidia_snapshot_and_manifest_pass_unmodified(volume, offline, nv_models, capsys):
    lay = warm.build_linked_layout(volume.parent, volume, hub_root=volume.parent / "hub")
    ref.check_nv_models(nv_models.MODELS)
    for kind in ("tempseg", "captioner"):
        path = nv_models.snapshot(kind)
        assert path == lay.paths[kind].resolve()
        nv_models.validate_manifest(path)  # full hash through the links
        nv_models.validate_manifest(path)  # second time: NVIDIA's per-process cache
    events = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()
              if line.startswith(ref.RUNTIME_PREFIX)]
    kinds = [(e["event"], e["kind"]) for e in events]
    assert kinds == [
        ("manifest_full_verification", "tempseg"), ("manifest_verified_reuse", "tempseg"),
        ("manifest_full_verification", "captioner"), ("manifest_verified_reuse", "captioner"),
    ]
    full = [e for e in events if e["event"] == "manifest_full_verification"]
    assert all(e["payload_bytes_read"] == sum(map(len, PAYLOAD.values())) for e in full)


@pytest.mark.nvlink
def test_nvidia_guard_still_rejects_links_escaping_root(volume, offline, nv_models, tmp_path):
    """Without the ``snapshots`` parent (allowed_root == the snapshot itself) NVIDIA's guard
    rejects the link farm, and a link that leaves ``<root>`` is rejected even with it."""
    lay = warm.build_linked_layout(volume.parent, volume, hub_root=volume.parent / "hub")
    flat = tmp_path / "flat" / "captioner"
    warm.link_snapshot(lay.sources["captioner"], flat)
    with pytest.raises(RuntimeError, match="Invalid model manifest"):
        nv_models.validate_manifest(flat)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(PAYLOAD["README.md"])
    readme = lay.paths["captioner"] / "README.md"
    readme.unlink()
    readme.symlink_to(outside)
    with pytest.raises(RuntimeError, match="Invalid model manifest"):
        nv_models.validate_manifest(lay.paths["captioner"])


@pytest.mark.nvlink
def test_nvidia_manifest_catches_tampered_bytes(volume, offline, nv_models, tmp_path):
    lay = warm.build_linked_layout(volume.parent, volume, hub_root=volume.parent / "hub")
    shard = lay.paths["captioner"] / "model-00001-of-00002.safetensors"
    evil = volume.parent / "evil.safetensors"  # inside the root, so only the hash can catch it
    evil.write_bytes(b"z" * 4000)
    shard.unlink()
    shard.symlink_to(evil)
    with pytest.raises(RuntimeError, match="integrity check"):
        nv_models.validate_manifest(lay.paths["captioner"])
