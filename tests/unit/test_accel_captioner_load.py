"""The BF16 Captioner redirection patch (``accel/captioner_load.py``)."""

from __future__ import annotations

import dis
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from flora_deploy.accel import captioner_load as cl
from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.prep import convert_captioner as cc


def _derived(tmp_path: Path, *, revision: str | None = None) -> Path:
    """A minimal derived dir with a hand-written receipt (no torch needed)."""
    pin = PINS_BY_KEY["captioner"]
    d = tmp_path / "derived"
    d.mkdir()
    files = []
    for name, data in (("config.json", b"{}"), ("model-00001-of-00001.safetensors", b"x" * 99)):
        (d / name).write_bytes(data)
        files.append({"path": name, "size": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    receipt = {
        "schema": cc.RECEIPT_SCHEMA, "ok": True, "kind": cc.DERIVED_NAME,
        "converter": {"version": cc.CONVERTER_VERSION},
        "source": {"pin": {"revision": revision or pin.revision},
                   "manifest_sha256": dict(pin.expected)["manifest_sha256"]},
        "files": files,
    }
    (d / cc.DERIVED_RECEIPT).write_text(json.dumps(receipt))
    return d


def _fake_models():
    calls = []
    m = types.ModuleType("fake_nv_models")
    m.Session = object
    m.snapshot = lambda kind: calls.append(("snapshot", kind)) or Path(f"/nv/{kind}")
    m.validate_manifest = lambda path: calls.append(("validate", str(path)))
    m.runtime_event = lambda event, **kw: calls.append(("event", event))
    return m, calls


def test_default_dir_is_layout_derived_path():
    assert str(cl.default_derived_dir()) == (
        "/weights/derived/captioner-bf16@" + PINS_BY_KEY["captioner"].revision)


def test_install_redirects_only_the_captioner(tmp_path):
    d = _derived(tmp_path)
    models, calls = _fake_models()
    orig_snapshot, orig_validate = models.snapshot, models.validate_manifest
    with cl.install(models, d) as patch:
        assert models.snapshot("captioner") == d
        assert models.snapshot("tempseg") == Path("/nv/tempseg")
        models.validate_manifest(d)  # ours
        models.validate_manifest(Path("/nv/tempseg"))  # NVIDIA's
        assert calls == [("snapshot", "tempseg"),
                         ("event", "manifest_replaced_by_derived_receipt"),
                         ("validate", "/nv/tempseg")]
        assert [v["mode"] for v in patch.verifications] == ["fast", "fast"]
        with pytest.raises(RuntimeError, match="already installed"):
            cl.install(models, d)
    assert models.snapshot is orig_snapshot and models.validate_manifest is orig_validate


def test_install_accepts_family_handle_and_full_verify_env(tmp_path, monkeypatch):
    d = _derived(tmp_path)
    models, _ = _fake_models()
    monkeypatch.setenv(cl.FULL_VERIFY_ENV, "1")
    patch = cl.install(types.SimpleNamespace(models=models), d)
    models.validate_manifest(d)
    models.validate_manifest(d)
    assert [v["mode"] for v in patch.verifications] == ["fast", "full", "fast"]
    patch.uninstall()


@pytest.mark.parametrize("damage", ["size", "bytes", "revision", "missing_receipt", "extra"])
def test_bad_derived_dir_fails_before_loading(tmp_path, damage):
    d = _derived(tmp_path, revision="0" * 40 if damage == "revision" else None)
    if damage == "size":
        (d / "config.json").write_bytes(b"{} ")
    elif damage == "bytes":
        (d / "config.json").write_bytes(b"[]")
    elif damage == "missing_receipt":
        (d / cc.DERIVED_RECEIPT).unlink()
    elif damage == "extra":
        (d / "x").write_bytes(b"")
    models, calls = _fake_models()
    with pytest.raises(cl.DerivedReceiptError):
        cl.install(models, d)
    assert not hasattr(models.snapshot, "__flora_accel_captioner_bf16__")


def test_rejects_non_nvidia_module():
    with pytest.raises(TypeError):
        cl.install(types.ModuleType("x"), Path("/nowhere"))


@pytest.mark.nvlink
def test_patch_reaches_nvidias_session_load(nv_code, tmp_path):
    """``Session.load`` looks ``snapshot`` / ``validate_manifest`` up as module globals, so
    replacing the module attributes is enough, and NVIDIA's GDN pieces are left alone."""
    code = nv_code("tempseg_code") / "batch_utils"
    before = set(sys.modules)
    name = "_flora_accel_test_pkg"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(code)]
    sys.modules[name] = pkg
    try:
        models = importlib.import_module(f"{name}.models")
        globals_used = {i.argval for i in dis.get_instructions(models.Session.load)
                        if i.opname == "LOAD_GLOBAL"}
        assert {"snapshot", "validate_manifest"} <= globals_used
        imports = {i.argval for i in dis.get_instructions(models.Session.load)
                   if i.opname in ("IMPORT_NAME", "IMPORT_FROM")}
        assert {"production_gdn_fix", "_keep_native_gdn_fp32", "verify_gdn", "verify"} <= imports
        d = _derived(tmp_path)
        nv_validate = models.validate_manifest
        with cl.install(models, d):
            assert models.snapshot("captioner") == d
            assert models.validate_manifest(d) is None
        assert models.validate_manifest is nv_validate
    finally:
        for m in set(sys.modules) - before:
            del sys.modules[m]


def test_compare_models_detects_one_bit(tmp_path):
    torch = pytest.importorskip("torch")
    a = torch.nn.Linear(3, 2).to(torch.bfloat16)
    b = torch.nn.Linear(3, 2).to(torch.bfloat16)
    b.load_state_dict(a.state_dict())
    a.register_buffer("buf", torch.zeros(2))
    b.register_buffer("buf", torch.zeros(2))
    assert cl.compare_models(a, b)["identical"]
    with torch.no_grad():
        b.buf.copy_(torch.tensor([0.0, -0.0]))  # equal as floats, different bits
    report = cl.compare_models(a, b)
    assert not report["identical"] and report["buffers"]["mismatches"] == 1
