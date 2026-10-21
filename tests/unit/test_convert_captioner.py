"""FP32 -> BF16 Captioner conversion on a tiny random Qwen3.5 model (CPU).

The identity claim is checked the same way the GPU proof does it, at toy size: NVIDIA's call
(``from_pretrained(dtype=bf16)`` after NVIDIA's own ``_keep_native_gdn_fp32``) on the FP32
checkpoint vs. the same call on the converted checkpoint, compared bit for bit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch  # CI runs these in the nvlink job (docs/ci.md)
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from flora_deploy.accel.captioner_load import (  # noqa: E402
    DerivedReceiptError,
    compare_models,
    fixed_forward,
    verify_derived,
)
from flora_deploy.config.pins import PINS_BY_KEY  # noqa: E402
from flora_deploy.prep import convert_captioner as cc  # noqa: E402


def _keep_fp32_patch(nv_code_root: Path | None):
    """NVIDIA's ``_keep_native_gdn_fp32`` if the code pin is fetched, else skip."""
    if nv_code_root is None:
        pytest.skip("NVIDIA code pin not fetched")
    path = nv_code_root / "batch_utils" / "vendor" / "production_gdn_fix.py"
    spec = importlib.util.spec_from_file_location("_flora_test_production_gdn_fix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._keep_native_gdn_fp32


@pytest.fixture
def keep_patch(nv_code, monkeypatch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    # NVIDIA's patch mutates the class; undo it after the test.
    monkeypatch.setattr(
        Qwen3_5PreTrainedModel, "_keep_in_fp32_modules_strict",
        Qwen3_5PreTrainedModel._keep_in_fp32_modules_strict, raising=False,
    )
    return _keep_fp32_patch(nv_code("tempseg_code"))


def tiny_config():
    from transformers import Qwen3_5Config

    text = dict(
        vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0],
                         "mrope_interleaved": True},
    )
    vision = dict(depth=1, hidden_size=32, intermediate_size=48, num_heads=2,
                  out_hidden_size=64, patch_size=4, num_position_embeddings=16)
    return Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def fp32_snapshot(tmp_path):
    """A tiny FP32 snapshot laid out like NVIDIA's (shards, index, extra files, manifest)."""
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(tiny_config(), dtype=torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            # Values with low mantissa bits set, so any BF16 rounding of A_log/dt_bias shows.
            p.copy_(torch.randn_like(p) * 0.3 + (1.0 if name.endswith("dt_bias") else 0.0))
    src = tmp_path / "src"
    model.save_pretrained(src, max_shard_size="200KB")
    shards = sorted(p.name for p in src.glob("model-*.safetensors"))
    assert len(shards) >= 3
    # NVIDIA's checkpoint also carries TransformerEngine ``_extra_state`` tensors (unexpected keys).
    index = json.loads((src / cc.INDEX_NAME).read_text())
    first = src / shards[0]
    tensors = load_file(first)
    tensors["model.language_model.layers.0.mlp.down_proj._extra_state"] = torch.arange(
        7, dtype=torch.uint8)
    save_file(tensors, first, metadata={"format": "pt"})
    index["weight_map"]["model.language_model.layers.0.mlp.down_proj._extra_state"] = shards[0]
    (src / cc.INDEX_NAME).write_text(json.dumps(index, indent=2))
    for name in cc.COPIED_FILES:
        if name != cc.MANIFEST_NAME and not (src / name).exists():
            (src / name).write_text(f"stand-in {name}\n")
    others = (n for n in cc.COPIED_FILES if n != cc.MANIFEST_NAME)
    listed = sorted([*shards, cc.INDEX_NAME, *others])
    manifest = "".join(f"{_sha(src / n)}  {n}\n" for n in listed)
    (src / cc.MANIFEST_NAME).write_text(manifest)
    pin = replace(
        PINS_BY_KEY["captioner"],
        expected=(("manifest_sha256", hashlib.sha256(manifest.encode()).hexdigest()),),
    )
    return src, pin


def _nv_load(path: Path):
    """The Captioner call of NVIDIA's ``Session.load`` (models.py), minus device placement."""
    from transformers import AutoModelForImageTextToText

    model, info = AutoModelForImageTextToText.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation={"text_config": "sdpa", "vision_config": "sdpa"},
        output_loading_info=True,
    )
    return model.eval(), info


def test_header_roundtrip(tmp_path):
    entries = [("a", "BF16", [2, 3], 12), ("b", "F32", [], 4), ("c", "U8", [0], 0)]
    head = cc.encode_header(entries, {"format": "pt"})
    assert len(head) % 8 == 0
    p = tmp_path / "x.safetensors"
    p.write_bytes(head + bytes(range(16)))
    with safe_open(str(p), framework="pt") as f:
        assert f.metadata() == {"format": "pt"}
        assert f.get_tensor("a").dtype == torch.bfloat16
        assert f.get_tensor("b").shape == ()
        assert f.get_tensor("c").numel() == 0


def test_conversion_is_bit_identical_to_nvidia_load(fp32_snapshot, keep_patch, tmp_path):
    src, pin = fp32_snapshot
    dest = tmp_path / "derived"
    receipt = cc.convert(src, dest, keep_patch, pin=pin, workers=2)
    assert receipt["ok"] and receipt["dtype_policy"]["dtype_plan"] == {
        "A_log": "float32", "dt_bias": "float32"}

    # Stored dtypes: FP32 exactly for A_log/dt_bias, unexpected keys untouched, BF16 otherwise.
    seen = set()
    for shard in dest.glob("model-*.safetensors"):
        with safe_open(str(shard), framework="pt") as f:
            assert f.metadata() == {"format": "pt"}
            for k in f.keys():
                t = f.get_tensor(k)
                seen.add(k)
                if k.endswith((".A_log", ".dt_bias")):
                    assert t.dtype == torch.float32
                elif k.endswith("_extra_state"):
                    assert t.dtype == torch.uint8 and t.tolist() == list(range(7))
                else:
                    assert t.dtype == torch.bfloat16, k
    src_index = json.loads((src / cc.INDEX_NAME).read_text())
    dst_index = json.loads((dest / cc.INDEX_NAME).read_text())
    assert dst_index["weight_map"] == src_index["weight_map"]
    assert seen == set(src_index["weight_map"])
    for name in cc.COPIED_FILES:
        assert (dest / name).read_bytes() == (src / name).read_bytes()

    a, info_a = _nv_load(src)
    b, info_b = _nv_load(dest)
    for info in (info_a, info_b):
        assert not info["missing_keys"] and not info["mismatched_keys"]
        assert set(info["unexpected_keys"]) <= {
            k for k in src_index["weight_map"] if "_extra_state" in k}
    report = compare_models(a, b)
    assert report["identical"], report
    assert report["parameters"]["dtypes"] == {"torch.bfloat16": report["parameters"]["count"] - 6,
                                              "torch.float32": 6}  # 3 GDN layers
    la, lb = fixed_forward(a, seq_len=12), fixed_forward(b, seq_len=12)
    assert torch.equal(la.view(torch.int16), lb.view(torch.int16))

    # Without NVIDIA's keep-FP32 patch the loaded A_log would be rounded: the policy matters.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    Qwen3_5PreTrainedModel._keep_in_fp32_modules_strict = None
    plain, _ = _nv_load(src)
    assert not compare_models(a, plain)["identical"]


def _dpin(dest: Path):
    """A DerivedPin for this toy receipt (the real one pins the Volume's receipt)."""
    from flora_deploy.accel.captioner_load import pinned_derived

    receipt = json.loads((dest / cc.DERIVED_RECEIPT).read_text())
    return replace(pinned_derived(), content_sha256=cc.content_digest(receipt))


def test_receipt_checks(fp32_snapshot, keep_patch, tmp_path):
    src, pin = fp32_snapshot
    dest = tmp_path / "derived"
    receipt = cc.convert(src, dest, keep_patch, pin=pin, workers=1)
    assert cc.convert(src, dest, keep_patch, pin=pin)["reused"] is True
    assert {f["path"] for f in receipt["files"]} == {p.name for p in dest.iterdir()} - {
        cc.DERIVED_RECEIPT}
    assert all(f["sha256"] == _sha(dest / f["path"]) for f in receipt["files"])
    assert receipt["converter"]["version"] == cc.CONVERTER_VERSION
    assert receipt["converter"]["module_sha256"] == _sha(Path(cc.__file__))
    assert "torch" in receipt["tools"] and receipt["source"]["pin"]["revision"] == pin.revision
    dpin = _dpin(dest)

    report = verify_derived(dest, pin=pin, derived_pin=dpin)
    assert report["mode"] == "full" and report["bytes_hashed"] == receipt["bytes"]
    assert verify_derived(dest, pin=pin, derived_pin=dpin, full=False)["mode"] == "fast"
    with pytest.raises(DerivedReceiptError, match="manifest digest"):
        verify_derived(dest, derived_pin=dpin)  # the real pin's manifest digest differs
    with pytest.raises(DerivedReceiptError, match="pinned"):
        verify_derived(dest, pin=pin)  # the real pin (unset, or not this toy build)
    assert receipt["content_sha256"] == cc.content_digest(receipt)

    # Corrupt one data byte of the largest shard: the default (full) check sees it.
    shard = max(dest.glob("model-*.safetensors"), key=lambda p: p.stat().st_size)
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(bytes(data))
    with pytest.raises(DerivedReceiptError, match="sha256 mismatch"):
        verify_derived(dest, pin=pin, derived_pin=dpin)
    data[-1] ^= 1
    shard.write_bytes(bytes(data))
    (dest / "stray.bin").write_bytes(b"x")
    with pytest.raises(DerivedReceiptError, match="extra"):
        verify_derived(dest, pin=pin, derived_pin=dpin)


def test_dtype_plan_rebuilt_at_load_matches_receipt(fp32_snapshot, keep_patch, tmp_path):
    """M1: what ``install`` rebuilds (NVIDIA's keep-FP32 patch + transformers' dtype plan on the
    meta device, from the derived config) equals the receipt's plan; a changed rule does not."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    from flora_deploy.accel import captioner_load as cl

    src, pin = fp32_snapshot
    dest = tmp_path / "derived"
    receipt = cc.convert(src, dest, keep_patch, pin=pin, workers=1)
    now = cl.nvidia_policy_record(None, dest, keep_fp32_patch=keep_patch)
    assert cl.check_dtype_policy(receipt, now) == receipt["dtype_policy"]

    def keep_more(cls):  # e.g. a transformers upgrade that keeps another module in FP32
        keep_patch(cls)
        cls._keep_in_fp32_modules_strict = [*cls._keep_in_fp32_modules_strict, "norm"]

    changed = cl.nvidia_policy_record(None, dest, keep_fp32_patch=keep_more)
    Qwen3_5PreTrainedModel._keep_in_fp32_modules_strict = None
    with pytest.raises(DerivedReceiptError, match="dtype plan differs"):
        cl.check_dtype_policy(receipt, changed)
    # and the receipt's toolchain is this process's (convert ran here)
    assert cl.check_toolchain(receipt, replace(_dpin(dest), proven_with=()))


def test_large_file_fast_check_uses_header(fp32_snapshot, keep_patch, tmp_path, monkeypatch):
    from flora_deploy.accel import captioner_load as cl

    src, pin = fp32_snapshot
    dest = tmp_path / "derived"
    cc.convert(src, dest, keep_patch, pin=pin, workers=1)
    dpin = _dpin(dest)
    monkeypatch.setattr(cl, "SMALL_FILE", 1024)
    report = verify_derived(dest, pin=pin, derived_pin=dpin, full=False)
    shards = list(dest.glob("model-*.safetensors"))
    assert report["bytes_hashed"] < sum(p.stat().st_size for p in shards)
    # Header tamper (dtype/shape/offsets) is caught by the fast (opt-out) check.
    shard = shards[0]
    raw = bytearray(shard.read_bytes())
    i = raw.index(b'"BF16"')
    raw[i : i + 6] = b'"F16" '
    shard.write_bytes(bytes(raw))
    with pytest.raises(DerivedReceiptError, match="header"):
        verify_derived(dest, pin=pin, derived_pin=dpin, full=False)


def test_source_must_match_nvidia_manifest(fp32_snapshot, keep_patch, tmp_path):
    src, pin = fp32_snapshot
    shard = sorted(src.glob("model-*.safetensors"))[1]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(bytes(data))
    with pytest.raises(cc.ConversionError, match="NVIDIA manifest"):
        cc.convert(src, tmp_path / "d", keep_patch, pin=pin, workers=1)
    with pytest.raises(cc.ConversionError, match="pinned"):
        cc.convert(src, tmp_path / "d", keep_patch, workers=1)  # real pin: manifest differs


def test_policy_rule_mirrors_transformers():
    policy = cc.DtypePolicy.build(
        "bfloat16", {"A_log": "float32", "dt_bias": "float32"},
        {"m.l.0.linear_attn.A_log": "bfloat16", "m.l.0.linear_attn.dt_bias": "bfloat16",
         "m.l.0.mlp.w": "bfloat16", "m.idx": "int64"},
    )
    assert policy.target("m.l.0.linear_attn.A_log") == "float32"
    assert policy.target("m.l.0.linear_attn.dt_bias") == "float32"
    assert policy.target("m.l.0.mlp.w") == "bfloat16"
    assert policy.target("m.idx") == "int64"
    assert policy.target("m.l.0.mlp.w._extra_state") is None
