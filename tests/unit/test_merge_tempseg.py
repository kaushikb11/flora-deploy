"""FP32 TempSeg LoRA merge (``prep/merge_tempseg.py``) on tiny synthetic tensors (CPU)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch  # CI runs these in the nvlink job (docs/ci.md)
torch = pytest.importorskip("torch")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from flora_deploy.config.pins import PINS_BY_KEY  # noqa: E402
from flora_deploy.prep import merge_tempseg as mt  # noqa: E402
from flora_deploy.prep.convert_captioner import DtypePolicy  # noqa: E402

# --- adapter spec / key mapping --------------------------------------------------------------

BASE_CFG = {"peft_type": "LORA", "r": 32, "lora_alpha": 64, "bias": "none", "use_dora": False,
            "target_modules": ["m.q", "m.k"]}


def test_scale_is_alpha_over_r():
    spec = mt.LoraSpec.from_config(BASE_CFG)
    assert spec.scale == 2.0 and spec.targets == {"m.q", "m.k"}
    assert spec.record()["scale"] == 2.0


@pytest.mark.parametrize("bad", [
    {"use_dora": True}, {"use_rslora": True}, {"bias": "all"}, {"fan_in_fan_out": True},
    {"rank_pattern": {"m.q": 8}}, {"alpha_pattern": {"m.q": 8}}, {"modules_to_save": ["x"]},
    {"peft_type": "IA3"}, {"target_modules": "all-linear"}, {"revision": "other"},
])
def test_spec_rejects_what_the_formula_does_not_cover(bad):
    with pytest.raises(mt.MergeError):
        mt.LoraSpec.from_config({**BASE_CFG, **bad}, base_revision="rev")


def _keys(mods):
    return [f"base_model.model.{m}.lora_{ab}.weight" for m in mods for ab in "AB"]


def test_key_mapping_exact():
    spec = mt.LoraSpec.from_config(BASE_CFG)
    targets = mt.map_adapter_keys(_keys(["m.q", "m.k"]), spec)
    assert set(targets) == {"m.q.weight", "m.k.weight"}
    t = targets["m.q.weight"]
    assert (t.lora_a, t.lora_b) == ("base_model.model.m.q.lora_A.weight",
                                    "base_model.model.m.q.lora_B.weight")


@pytest.mark.parametrize("keys", [
    _keys(["m.q"]),  # a target is missing
    _keys(["m.q", "m.k", "m.v"]),  # a module that is not a target
    _keys(["m.q", "m.k"])[:-1],  # lora_B missing
    [*_keys(["m.q", "m.k"]), "base_model.model.m.q.lora_A.bias"],  # not a LoRA weight
    [*_keys(["m.q", "m.k"]), "base_model.model.m.q.lora_magnitude_vector.weight"],
])
def test_key_mapping_rejects(keys):
    with pytest.raises(mt.MergeError):
        mt.map_adapter_keys(keys, mt.LoraSpec.from_config(BASE_CFG))


def test_shapes():
    spec = mt.LoraSpec.from_config({**BASE_CFG, "r": 2, "lora_alpha": 4})
    targets = mt.map_adapter_keys(_keys(["m.q", "m.k"]), spec)
    shapes = {"base_model.model.m.q.lora_A.weight": [2, 5],
              "base_model.model.m.q.lora_B.weight": [3, 2],
              "base_model.model.m.k.lora_A.weight": [2, 5],
              "base_model.model.m.k.lora_B.weight": [3, 2]}
    mt.check_shapes(targets, {"m.q.weight": [3, 5], "m.k.weight": [3, 5]}, shapes, 2)
    with pytest.raises(mt.MergeError):
        mt.check_shapes(targets, {"m.q.weight": [3, 5], "m.k.weight": [5, 3]}, shapes, 2)
    with pytest.raises(mt.MergeError):
        mt.check_shapes(targets, {"m.q.weight": [3, 5]}, shapes, 2)


def test_merge_weight_is_fp32_sum():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(6, 5, generator=g).to(torch.bfloat16)
    a, b = torch.randn(2, 5, generator=g), torch.randn(6, 2, generator=g)
    got = mt.merge_weight(w, a, b, 2.0)
    assert got.dtype == torch.float32
    assert torch.equal(got, w.float() + (b @ a) * 2.0)


def test_key_coverage_and_dtype_plan():
    policy = DtypePolicy.build("bfloat16", {"A_log": "float32"},
                               {"m.q.weight": "bfloat16", "m.A_log": "bfloat16"})
    assert policy.target("m.A_log") == "float32" and policy.target("mtp.fc.weight") is None
    targets = {"m.q.weight": mt.Target("m.q", "m.q.weight", "a", "b")}
    wm = {"m.q.weight": "s", "m.A_log": "s", "mtp.fc.weight": "s"}
    assert mt.check_key_coverage(wm, policy, ["^mtp.*"], targets) == ["mtp.fc.weight"]
    with pytest.raises(mt.MergeError):  # not ignorable
        mt.check_key_coverage(wm, policy, [], targets)
    with pytest.raises(mt.MergeError):  # a target that would not be BF16
        mt.check_key_coverage(wm, policy, ["^mtp.*"],
                              {"m.A_log": mt.Target("m", "m.A_log", "a", "b")})
    with pytest.raises(mt.MergeError):  # model key absent from the checkpoint
        mt.check_key_coverage({"m.q.weight": "s"}, policy, [], targets)


def test_metrics_helpers():
    a = torch.tensor([[[0.0, 1.0, 0.5], [2.0, 0.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.5, 0.5], [0.0, 0.0, 3.0]]])
    d = mt.logit_diff(a, b)
    assert d["max_abs"] == 3.0 and d["top1_agree"] == 0.5 and d["next_token_equal"] is False
    assert mt.logit_diff(a, a)["next_token_equal"] is True
    inputs = {"input_ids": torch.tensor([[5, 6, 7]]), "attention_mask": torch.ones(1, 3),
              "mm_token_type_ids": torch.tensor([[1, 1, 0]]), "pixel_values": torch.zeros(4, 2)}
    ext, keep = mt.teacher_forced_inputs(inputs, [8, 9, 10])
    assert keep == 3 and ext["input_ids"].tolist() == [[5, 6, 7, 8, 9]]
    assert ext["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert ext["mm_token_type_ids"].tolist() == [[1, 1, 0, 0, 0]]
    assert ext["pixel_values"] is inputs["pixel_values"]
    assert mt.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert mt.first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert mt.first_divergence([1, 2], [1, 2, 3]) == 2


# --- end to end on a tiny Qwen3.5 + real PEFT adapter ---------------------------------------


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _keep_patch(nv_code):
    path = nv_code("tempseg_code") / "batch_utils" / "vendor" / "production_gdn_fix.py"
    spec = importlib.util.spec_from_file_location("_flora_test_gdn_fix_merge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._keep_native_gdn_fp32


@pytest.fixture
def keep_patch(nv_code, monkeypatch):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    monkeypatch.setattr(Qwen3_5PreTrainedModel, "_keep_in_fp32_modules_strict",
                        Qwen3_5PreTrainedModel._keep_in_fp32_modules_strict, raising=False)
    return _keep_patch(nv_code)


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")


@pytest.fixture
def snapshots(tmp_path):
    """Tiny BF16 base (with an ignored ``mtp.*`` key and a prep receipt) + a Flora-like
    snapshot (FP32 PEFT adapter, processor stand-ins, NVIDIA-style manifest)."""
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForImageTextToText

    from tests.unit.test_convert_captioner import tiny_config

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(tiny_config(), dtype=torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * 0.3 + (1.0 if name.endswith("dt_bias") else 0.0))
    model = model.to(torch.bfloat16)
    base = tmp_path / "base"
    model.save_pretrained(base, max_shard_size="200KB")
    shards = sorted(p.name for p in base.glob("*.safetensors"))
    assert len(shards) >= 3
    index = json.loads((base / mt.INDEX_NAME).read_text())
    first = base / shards[0]
    tensors = load_file(first)
    tensors["mtp.fc.weight"] = torch.randn(4, 4).to(torch.bfloat16)
    save_file(tensors, first, metadata={"format": "pt"})
    index["weight_map"]["mtp.fc.weight"] = shards[0]
    (base / mt.INDEX_NAME).write_text(json.dumps(index, indent=2))
    files = [*shards, mt.INDEX_NAME, "config.json"]
    receipt = {"ok": True, "plan_digest": "p" * 64,
               "pin": {"revision": PINS_BY_KEY["tempseg_base"].revision},
               "digests": [{"path": n, "sha256": _sha(base / n)} for n in files]}
    (base / mt.PREP_RECEIPT).write_text(json.dumps(receipt))
    base_pin = replace(PINS_BY_KEY["tempseg_base"], plan_digest="p" * 64)

    targets = sorted(n for n, m in model.named_modules()
                     if n.startswith("model.language_model.") and n.endswith(TARGET_SUFFIXES))
    fp32 = AutoModelForImageTextToText.from_config(tiny_config(), dtype=torch.float32)
    fp32.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=targets, lora_dropout=0.0,
                          revision=base_pin.revision)
    pm = peft.get_peft_model(fp32, cfg)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.2)
    flora = tmp_path / "flora"
    pm.save_pretrained(flora)
    for name in mt.FLORA_FILES:
        (flora / name).write_text(f"flora {name}\n")
    listed = sorted([mt.ADAPTER_FILE, mt.ADAPTER_CONFIG, *mt.FLORA_FILES])
    manifest = "".join(f"{_sha(flora / n)}  {n}\n" for n in listed)
    (flora / mt.MANIFEST_NAME).write_text(manifest)
    pin = replace(PINS_BY_KEY["tempseg"], expected=(
        ("adapter_sha256", _sha(flora / mt.ADAPTER_FILE)),
        ("manifest_sha256", hashlib.sha256(manifest.encode()).hexdigest())))
    return dict(base=base, flora=flora, pin=pin, base_pin=base_pin, model=model,
                peft_model=pm, targets=targets)


def _merge(s, dest, keep, **kw):
    return mt.merge(s["flora"], s["base"], dest, keep, pin=s["pin"], base_pin=s["base_pin"],
                    expected_counts=(len(s["targets"]), 2 * len(s["targets"])), **kw)


def test_merge_end_to_end(snapshots, keep_patch, tmp_path):
    s = snapshots
    dest = tmp_path / "merged"
    receipt = _merge(s, dest, keep_patch, workers=2, provenance={"git_commit": "abc"})
    assert receipt["ok"] and receipt["lora"]["scale"] == 2.0
    assert receipt["merge_stats"]["targets"] == len(s["targets"])
    assert receipt["merge_stats"]["fraction_changed"] > 0.5
    assert receipt["dtype_policy"]["dtype_plan"] == {"A_log": "float32", "dt_bias": "float32"}
    assert receipt["dtype_policy"]["ignored_keys_copied"] == 1
    assert receipt["merger"]["git_commit"] == "abc"
    assert receipt["merger"]["module_sha256"] == _sha(Path(mt.__file__))

    # Reference: PEFT's own merge in FP32 on the upcast BF16 base, rounded once to BF16.
    ref = s["peft_model"].merge_and_unload().state_dict()
    base_sd = s["model"].state_dict()
    targets = {f"{t}.weight" for t in s["targets"]}
    seen = set()
    for shard in dest.glob("*.safetensors"):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                seen.add(k)
                if k.endswith((".A_log", ".dt_bias")):
                    assert t.dtype == torch.float32
                    assert torch.equal(t, base_sd[k].float())
                elif k == "mtp.fc.weight":
                    assert t.dtype == torch.bfloat16
                elif k in targets:
                    assert t.dtype == torch.bfloat16
                    assert torch.equal(t.view(torch.int16),
                                       ref[k].to(torch.bfloat16).view(torch.int16)), k
                    assert not torch.equal(t, base_sd[k])
                else:
                    assert torch.equal(t.view(torch.int16), base_sd[k].view(torch.int16)), k
    assert seen == set(json.loads((s["base"] / mt.INDEX_NAME).read_text())["weight_map"])

    # Flora's processor files are copied, the base's config, never adapter_config.json.
    for name in mt.FLORA_FILES:
        assert (dest / name).read_bytes() == (s["flora"] / name).read_bytes()
    assert (dest / "config.json").read_bytes() == (s["base"] / "config.json").read_bytes()
    assert not (dest / mt.ADAPTER_CONFIG).exists()
    assert {f["path"] for f in receipt["files"]} == {
        p.name for p in dest.iterdir()} - {mt.DERIVED_RECEIPT}
    assert all(f["sha256"] == _sha(dest / f["path"]) for f in receipt["files"])
    assert mt.verify_merged(dest)["files"] == len(receipt["files"])
    assert mt.verify_merged(dest, full=True)["full"] is True
    assert _merge(s, dest, keep_patch)["reused"] is True

    # Loads as a plain model through NVIDIA's call: no missing keys, GDN params FP32.
    from transformers import AutoModelForImageTextToText

    m, info = AutoModelForImageTextToText.from_pretrained(
        dest, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        output_loading_info=True)
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys",
                                         "error_msgs")), info
    dts = {n: p.dtype for n, p in m.named_parameters()}
    assert all(d == torch.float32 for n, d in dts.items() if n.endswith((".A_log", ".dt_bias")))


def test_merge_receipt_tamper(snapshots, keep_patch, tmp_path):
    s = snapshots
    dest = tmp_path / "merged"
    _merge(s, dest, keep_patch, workers=1)
    (dest / "tokenizer.json").write_text("tampered\n")
    with pytest.raises(mt.MergeError):
        mt.verify_merged(dest)
    (dest / "stray.txt").write_text("x")
    with pytest.raises(mt.MergeError):
        mt.verify_merged(dest)


def test_merge_refuses_wrong_sources(snapshots, keep_patch, tmp_path):
    s = snapshots
    manifest_sha = dict(s["pin"].expected)["manifest_sha256"]
    bad_pin = replace(s["pin"], expected=(("adapter_sha256", "0" * 64),
                                         ("manifest_sha256", manifest_sha)))
    with pytest.raises(mt.MergeError):
        mt.merge(s["flora"], s["base"], tmp_path / "x", keep_patch, pin=bad_pin,
                 base_pin=s["base_pin"], expected_counts=None)
    with pytest.raises(mt.MergeError):  # wrong expected counts (the real adapter's 248/496)
        mt.merge(s["flora"], s["base"], tmp_path / "y", keep_patch, pin=s["pin"],
                 base_pin=s["base_pin"])
    with pytest.raises(mt.MergeError):  # base receipt not bound to the pinned plan digest
        mt.merge(s["flora"], s["base"], tmp_path / "z", keep_patch, pin=s["pin"],
                 base_pin=replace(s["base_pin"], plan_digest="q" * 64), expected_counts=None)
