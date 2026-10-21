"""The BF16 Captioner redirection patch (``accel/captioner_load.py``)."""

from __future__ import annotations

import dis
import hashlib
import json
import os
import types
from dataclasses import replace
from pathlib import Path

import pytest

from flora_deploy.accel import captioner_load as cl
from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.prep import convert_captioner as cc

TOOLS = {"python": "3.12.1", "torch": "2.11.0+cu130", "transformers": "5.8.1",
         "safetensors": "0.7.0"}
POLICY = {"load_dtype": "bfloat16", "dtype_plan": {"A_log": "float32"},
          "non_load_dtype_keys": 2, "non_load_dtype_key_sha256": "ab" * 32}
SHARD = "model-00001-of-00001.safetensors"


@pytest.fixture(autouse=True)
def runtime_tools(monkeypatch):
    """This process "runs" the proven toolchain (the CUDA build tag is not compared)."""
    monkeypatch.setattr(cl, "tool_versions", lambda: {**TOOLS, "torch": "2.11.0+cu128"})
    monkeypatch.delenv(cl.FULL_VERIFY_ENV, raising=False)


def _write_receipt(d: Path, receipt: dict) -> cl.DerivedPin:
    raw = json.dumps(receipt).encode()
    (d / cc.DERIVED_RECEIPT).write_bytes(raw)
    return replace(cl.pinned_derived(), content_sha256=cc.content_digest(receipt))


def _derived(tmp_path: Path, *, revision: str | None = None, shard: bytes = b"x" * 99,
             tools=TOOLS) -> tuple[Path, cl.DerivedPin]:
    """A minimal derived dir with a hand-written receipt pinned by a synthetic DerivedPin."""
    pin = PINS_BY_KEY["captioner"]
    d = tmp_path / "derived"
    d.mkdir()
    files = []
    for name, data in (("config.json", b"{}"), (SHARD, shard)):
        (d / name).write_bytes(data)
        entry = {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if name == SHARD:
            entry["header_sha256"] = "0" * 64  # not a real safetensors file: fast mode fails
        files.append(entry)
    receipt = {
        "schema": cc.RECEIPT_SCHEMA, "ok": True, "kind": cc.DERIVED_NAME,
        "converter": {"version": cc.CONVERTER_VERSION},
        "source": {"pin": {"revision": revision or pin.revision},
                   "manifest_sha256": dict(pin.expected)["manifest_sha256"]},
        "files": files, "tools": tools, "dtype_policy": POLICY, "casts": {},
    }
    return d, _write_receipt(d, receipt)


def _fake_models():
    calls = []
    m = types.ModuleType("fake_nv_models")
    m.Session = object
    m.snapshot = lambda kind: calls.append(("snapshot", kind)) or Path(f"/nv/{kind}")
    m.validate_manifest = lambda path: calls.append(("validate", str(path)))
    m.runtime_event = lambda event, **kw: calls.append(("event", event))
    return m, calls


def _install(models, d, dpin, **kw):
    kw.setdefault("policy_record", lambda: POLICY)
    return cl.install(models, d, derived_pin=dpin, **kw)


def test_default_dir_is_layout_derived_path():
    assert str(cl.default_derived_dir()) == (
        "/weights/derived/captioner-bf16@" + PINS_BY_KEY["captioner"].revision)


def test_content_digest_pin_is_documented():
    dpin = cl.pinned_derived()
    # Not derivable from the docs (they do not record the per-file sha256s): fails closed
    # until read from the Volume with `flora prepare --show-receipt` and pinned.
    assert dpin.content_sha256 is None or len(dpin.content_sha256) == 64
    assert dpin.converter_version == cc.CONVERTER_VERSION and dpin.source_key == "captioner"
    doc = (Path(__file__).parents[2] / "docs" / "captioner-bf16.md").read_text()
    assert "content_sha256" in doc and "--show-receipt" in doc
    assert dict(dpin.proven_with) == {"torch": "2.11.0", "transformers": "5.8.1",
                                      "safetensors": "0.7.0"}


def test_unpinned_or_other_receipt_fails_closed(tmp_path):
    d, dpin = _derived(tmp_path)
    models, _ = _fake_models()
    with pytest.raises(cl.DerivedReceiptError, match="no content digest is pinned"):
        _install(models, d, replace(dpin, content_sha256=None))
    with pytest.raises(cl.DerivedReceiptError, match="!= pinned"):
        _install(models, d, replace(dpin, content_sha256="0" * 64))
    with pytest.raises(cl.DerivedReceiptError, match="pinned"):
        _install(models, d, None)  # the real pin (unset or not this toy receipt)
    assert not hasattr(models.snapshot, "__flora_accel_captioner_bf16__")


def test_install_redirects_only_the_captioner(tmp_path):
    d, dpin = _derived(tmp_path)
    models, calls = _fake_models()
    orig_snapshot, orig_validate = models.snapshot, models.validate_manifest
    with _install(models, d, dpin) as patch:
        assert models.snapshot("captioner") == d
        assert models.snapshot("tempseg") == Path("/nv/tempseg")
        models.validate_manifest(d)  # ours
        models.validate_manifest(Path("/nv/tempseg"))  # NVIDIA's
        assert calls == [("snapshot", "tempseg"),
                         ("event", "manifest_replaced_by_derived_receipt"),
                         ("validate", "/nv/tempseg")]
        # full hash by default at install; the load re-checks receipt + stamps only
        assert [v["mode"] for v in patch.verifications] == ["full", "reuse"]
        assert patch.verifications[0]["bytes_hashed"] == 101
        prov = patch.provenance()
        assert prov["full_verify"] and not prov["full_verify_opt_out"]
        assert prov["content_sha256_pinned"] == dpin.content_sha256
        assert prov["toolchain"]["torch"]["runtime"] == "2.11.0+cu128"
        assert prov["dtype_policy"] == POLICY
        with pytest.raises(RuntimeError, match="already installed"):
            cl.install(models, d)
    assert models.snapshot is orig_snapshot and models.validate_manifest is orig_validate


def test_runtime_recomputes_the_content_digest(tmp_path):
    d, dpin = _derived(tmp_path)
    receipt = json.loads((d / cc.DERIVED_RECEIPT).read_text())
    models, _ = _fake_models()
    # another build of the same content (other timings, host, build tag) loads
    other = {**receipt, "timings_s": {"total": 999.0}, "created_utc": "2030-01-01T00:00:00Z",
             "tools": {**TOOLS, "platform": "elsewhere"}}
    assert _write_receipt(d, other) == dpin
    _install(models, d, dpin).uninstall()
    # a stale stored digest is refused even though the recomputed one matches the pin
    _write_receipt(d, {**receipt, "content_sha256": "f" * 64})
    with pytest.raises(cl.DerivedReceiptError, match="stored content_sha256"):
        _install(models, d, dpin)
    # a receipt edited to claim the pinned digest, with other content, is refused
    forged = {**receipt, "content_sha256": dpin.content_sha256,
              "casts": {"F32->BF16:loaded": 1}}
    (d / cc.DERIVED_RECEIPT).write_text(json.dumps(forged))
    with pytest.raises(cl.DerivedReceiptError, match="stored content_sha256"):
        _install(models, d, dpin)


def test_full_verify_opt_out_is_recorded(tmp_path, monkeypatch):
    d, dpin = _derived(tmp_path, shard=b"x" * 99)
    models, _ = _fake_models()
    monkeypatch.setenv(cl.FULL_VERIFY_ENV, "0")
    patch = _install(types.SimpleNamespace(models=models), d, dpin)
    models.validate_manifest(d)
    assert [v["mode"] for v in patch.verifications] == ["fast", "reuse"]
    assert patch.provenance()["full_verify_opt_out"] is True
    patch.uninstall()


def test_corrupted_shard_body_is_refused_by_default(tmp_path, monkeypatch):
    """Review H1 (p3): a byte flipped inside a large shard passed the old default check."""
    monkeypatch.setattr(cl, "SMALL_FILE", 10)  # the toy shard counts as "large"
    data = cc.encode_header([("w", "BF16", [64], 128)], None) + b"\x11" * 128
    d, dpin = _derived(tmp_path, shard=data)
    receipt = json.loads((d / cc.DERIVED_RECEIPT).read_text())
    receipt["files"][1]["header_sha256"] = cc.header_sha256(d / SHARD)
    dpin = _write_receipt(d, receipt)
    raw = bytearray(data)
    raw[-5] ^= 0xFF
    (d / SHARD).write_bytes(bytes(raw))
    models, _ = _fake_models()
    with pytest.raises(cl.DerivedReceiptError, match="sha256 mismatch"):
        _install(models, d, dpin)
    # the documented opt-out does not read tensor bytes (that is why it is not the default)
    assert cl.verify_derived(d, full=False, derived_pin=dpin)["mode"] == "fast"


@pytest.mark.parametrize("damage", ["size", "bytes", "revision", "missing_receipt", "extra",
                                    "symlink", "subdir", "converter", "receipt_symlink"])
def test_bad_derived_dir_fails_before_loading(tmp_path, damage):
    d, dpin = _derived(tmp_path, revision="0" * 40 if damage == "revision" else None)
    if damage == "size":
        (d / "config.json").write_bytes(b"{} ")
    elif damage == "bytes":
        (d / "config.json").write_bytes(b"[]")
    elif damage == "missing_receipt":
        (d / cc.DERIVED_RECEIPT).unlink()
    elif damage == "extra":
        (d / "x").write_bytes(b"")
    elif damage == "symlink":  # same bytes, outside the derived dir: still refused
        outside = tmp_path / "elsewhere.safetensors"
        os.replace(d / SHARD, outside)
        os.symlink(outside, d / SHARD)
    elif damage == "subdir":
        (d / "nested").mkdir()
    elif damage == "converter":
        receipt = json.loads((d / cc.DERIVED_RECEIPT).read_text())
        receipt["converter"]["version"] = cc.CONVERTER_VERSION + 1
        dpin = _write_receipt(d, receipt)
    elif damage == "receipt_symlink":
        os.replace(d / cc.DERIVED_RECEIPT, tmp_path / "r.json")
        os.symlink(tmp_path / "r.json", d / cc.DERIVED_RECEIPT)
    models, _ = _fake_models()
    for full in (True, False):
        with pytest.raises(cl.DerivedReceiptError):
            _install(models, d, dpin, full_verify=full)
    assert not hasattr(models.snapshot, "__flora_accel_captioner_bf16__")


def test_background_verification_is_joined_by_the_load(tmp_path):
    d, dpin = _derived(tmp_path)
    models, _ = _fake_models()
    patch = _install(models, d, dpin, background=True)
    assert patch.provenance()["verification_pending"] or patch.wait()
    models.validate_manifest(d)  # waits for the background hash, then re-checks stamps
    assert [v["mode"] for v in patch.verifications] == ["full", "reuse"]
    assert patch.verifications[0]["background"] is True
    patch.uninstall()


def test_background_failure_fails_every_later_load(tmp_path):
    d, dpin = _derived(tmp_path)
    (d / "config.json").write_bytes(b"[]")  # same size: only the hash sees it
    models, _ = _fake_models()
    patch = _install(models, d, dpin, background=True)  # preflight passes
    with pytest.raises(cl.DerivedReceiptError, match="sha256 mismatch"):
        models.validate_manifest(d)
    (d / "config.json").write_bytes(b"{}")  # repaired: still refused (fail closed)
    with pytest.raises(cl.DerivedReceiptError, match="failed earlier"):
        models.validate_manifest(d)
    patch.uninstall()


def test_change_after_verification_is_refused(tmp_path):
    d, dpin = _derived(tmp_path)
    models, _ = _fake_models()
    patch = _install(models, d, dpin)
    st = os.stat(d / "config.json")
    os.utime(d / "config.json", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(cl.DerivedReceiptError, match="changed since verification"):
        models.validate_manifest(d)
    patch.uninstall()


@pytest.mark.parametrize("tools, match", [
    ({**TOOLS, "transformers": "5.9.0"}, "transformers"),  # receipt from another toolchain
    ({k: v for k, v in TOOLS.items() if k != "safetensors"}, "safetensors"),
])
def test_toolchain_must_match_receipt_and_proof(tmp_path, tools, match):
    d, dpin = _derived(tmp_path, tools=tools)
    models, _ = _fake_models()
    with pytest.raises(cl.DerivedReceiptError, match=match):
        _install(models, d, dpin)
    # runtime differs from the proof even though the receipt agrees with the runtime
    with pytest.raises(cl.DerivedReceiptError, match="torch"):
        cl.check_toolchain({"tools": TOOLS}, dpin, versions={**TOOLS, "torch": "2.12.0"})
    assert cl.check_toolchain({"tools": TOOLS}, dpin, versions=TOOLS)["torch"]["proven_with"] \
        == "2.11.0"


def test_dtype_plan_must_match_receipt(tmp_path):
    d, dpin = _derived(tmp_path)
    models, _ = _fake_models()
    changed = {**POLICY, "non_load_dtype_keys": 3}  # e.g. transformers keeps norms in FP32
    with pytest.raises(cl.DerivedReceiptError, match="dtype plan differs"):
        _install(models, d, dpin, policy_record=lambda: changed)
    patch = _install(models, d, dpin, check_toolchain=False, policy_record=lambda: changed)
    prov = patch.provenance()
    assert prov["toolchain_opt_out"] and prov["toolchain"] is None
    patch.uninstall()


def test_deferred_install_checks_inside_the_load(tmp_path):
    d, dpin = _derived(tmp_path)
    models, _ = _fake_models()
    patch = _install(models, d, dpin, verify_now=False)
    assert patch.verifications == [] and patch.toolchain is None
    models.validate_manifest(d)
    assert [v["mode"] for v in patch.verifications] == ["full"] and patch.toolchain
    patch.uninstall()


def test_rejects_non_nvidia_module():
    with pytest.raises(TypeError):
        cl.install(types.ModuleType("x"), Path("/nowhere"))


@pytest.mark.nvlink
def test_patch_reaches_nvidias_session_load(nv_code, tmp_path):
    """Through the Fix-B linker (verified bytes, no bytecode written to the pin cache):
    ``Session.load`` looks ``snapshot`` / ``validate_manifest`` up as module globals, so
    replacing the module attributes is enough, and NVIDIA's GDN pieces are left alone."""
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family

    nv = load_family(Family.SEGCAP, nv_code("tempseg_code"))
    models = nv.models
    globals_used = {i.argval for i in dis.get_instructions(models.Session.load)
                    if i.opname == "LOAD_GLOBAL"}
    assert {"snapshot", "validate_manifest"} <= globals_used
    imports = {i.argval for i in dis.get_instructions(models.Session.load)
               if i.opname in ("IMPORT_NAME", "IMPORT_FROM")}
    assert {"production_gdn_fix", "_keep_native_gdn_fp32", "verify_gdn", "verify"} <= imports
    # the default dtype-plan rebuild uses NVIDIA's own keep-FP32 patch from this snapshot
    keep = cl._keep_fp32_patch(nv)
    assert keep is nv.vendor("production_gdn_fix")._keep_native_gdn_fp32
    assert cl._keep_fp32_patch(models) is keep
    d, dpin = _derived(tmp_path)
    nv_validate = models.validate_manifest
    with _install(nv, d, dpin) as patch:
        assert models.snapshot("captioner") == d
        assert models.validate_manifest(d) is None
        assert patch.models is models
    assert models.validate_manifest is nv_validate
    nv.audit()


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
