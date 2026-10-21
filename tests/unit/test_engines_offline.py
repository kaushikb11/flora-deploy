"""Engine argument validation (D2, Gate C) and OfflineEngine batching/isolation with a fake LLM."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flora_deploy.engines import offline_llm as O
from flora_deploy.engines.base import Sampling, VllmRequest
from flora_deploy.nv.capture import CaptureError


def spec(kind="tempseg", gpu="L40S", **kw):
    return O.EngineSpec.for_gpu(kind, "/models/x", gpu, **kw)


def test_engine_kwargs_are_explicit_and_valid():
    kw = spec().engine_kwargs()
    assert kw["dtype"] == "bfloat16"
    assert kw["gdn_prefill_backend"] == "triton"
    assert kw["enable_prefix_caching"] is False
    assert kw["generation_config"] == "vllm"
    assert kw["mm_processor_cache_gb"] == 0
    assert kw["max_model_len"] == 32768
    assert kw["limit_mm_per_prompt"] == {"image": 0, "video": 3}
    assert kw["mm_processor_kwargs"] == {"fps": 4, "max_frames": 128}
    assert (kw["max_num_seqs"], kw["max_num_batched_tokens"]) == (32, 16384)
    cap = spec("captioner", "H100!").engine_kwargs()
    assert cap["mm_processor_kwargs"] == {"fps": 8, "max_frames": 40}
    assert (cap["max_num_seqs"], cap["max_num_batched_tokens"]) == (16, 16384)


@pytest.mark.parametrize(("key", "value"), [
    ("gdn_prefill_backend", "flashinfer"),
    ("enable_prefix_caching", True),
    ("generation_config", "auto"),
    ("dtype", "auto"),
    ("max_model_len", 16384),
    ("limit_mm_per_prompt", {"image": 1, "video": 3}),
    ("mm_processor_kwargs", {"fps": 2, "max_frames": 128}),
    ("max_num_seqs", 0),
    ("quantization", "fp8"),
    ("kv_cache_dtype", "fp8"),
])
def test_validation_rejects_uncertified_values(key, value):
    kw = spec().engine_kwargs()
    kw[key] = value
    with pytest.raises(O.EngineArgsError):
        O.validate_engine_kwargs("tempseg", kw)


@pytest.mark.parametrize("key", ["gdn_prefill_backend", "enable_prefix_caching",
                                 "generation_config", "max_num_seqs", "max_num_batched_tokens",
                                 "seed"])
def test_validation_requires_explicit_keys(key):
    kw = spec().engine_kwargs()
    del kw[key]
    with pytest.raises(O.EngineArgsError):
        O.validate_engine_kwargs("tempseg", kw)


def test_processor_cache_rule_is_gate_c():
    kw = spec().engine_kwargs()
    kw["mm_processor_cache_gb"] = 4
    with pytest.raises(CaptureError):
        O.validate_engine_kwargs("tempseg", kw)
    del kw["mm_processor_cache_gb"]
    with pytest.raises(CaptureError):
        O.validate_engine_kwargs("tempseg", kw)


def test_extra_cannot_override_pinned_values():
    with pytest.raises(O.EngineArgsError):
        spec(extra={"gdn_prefill_backend": "flashinfer"}).engine_kwargs()
    kw = spec(extra={"safetensors_load_strategy": "prefetch"}).engine_kwargs()
    assert kw["safetensors_load_strategy"] == "prefetch"


def test_gpu_names_and_limits():
    assert O.normalize_gpu("H100!") == "H100"
    assert O.normalize_gpu("NVIDIA H100 80GB HBM3") == "H100"
    assert O.normalize_gpu("NVIDIA L40S") == "L40S"
    with pytest.raises(O.EngineArgsError):
        O.normalize_gpu("T4")
    with pytest.raises(O.EngineArgsError):
        O.gpu_limits("captioner", "L40S")  # 27B BF16 does not fit; no table entry


def test_label_and_hash_view():
    s = spec()
    assert s.label == "vllm-0.24.0-offline"
    assert spec(dtbias_fp32=True).label == "vllm-0.24.0-offline+dtbias"
    h = s.hashed()
    assert "gpu" not in h and "model" not in h["engine_kwargs"]
    assert h["engine_kwargs"]["max_num_seqs"] == 32  # batching changes numerics: hashed


def test_nvidia_sampling_is_greedy_eos_only():
    sp = O.nvidia_sampling(512, 248046)
    assert (sp.max_tokens, sp.temperature, sp.stop_token_ids) == (512, 0.0, (248046,))
    with pytest.raises(O.EngineArgsError):
        O.nvidia_sampling(512, [1, 2])


class Params(SimpleNamespace):
    pass


def test_to_vllm_sampling_fields():
    p = O.to_vllm_sampling(O.nvidia_sampling(1024, 7), Params)
    assert p.temperature == 0.0 and p.max_tokens == 1024 and p.stop_token_ids == [7]
    assert p.skip_special_tokens is True and p.n == 1 and p.top_k == 0


def _ro(prompt_ids, ids, finish="stop", stop_reason=None):
    return SimpleNamespace(prompt_token_ids=prompt_ids, outputs=[SimpleNamespace(
        token_ids=ids, finish_reason=finish, stop_reason=stop_reason, logprobs=None)])


class FakeLLM:
    def __init__(self, bad=()):
        self.calls = []
        self.bad = set(bad)

    def generate(self, prompts, params, use_tqdm=True):
        self.calls.append([p["prompt"] for p in prompts])
        if any(p["prompt"] in self.bad for p in prompts):
            raise ValueError("The decoder prompt (length 40000) is longer than the maximum "
                             "model length of 32768.")
        return [_ro([1, 2, 3], [9, 7]) for _ in prompts]


def vreq(prompt):
    return VllmRequest(prompt=prompt, videos=((np.zeros((1, 2, 2, 3), np.uint8),
                                               {"fps": 4.0}),), key=prompt)


def test_offline_engine_batches_and_isolates():
    llm = FakeLLM(bad={"p1"})
    eng = O.OfflineEngine(spec(), llm, params_cls=Params)
    out = eng.generate_batch([vreq("p0"), vreq("p1"), vreq("p2")], [Sampling(8)] * 3)
    assert llm.calls == [["p0", "p1", "p2"], ["p0"], ["p1"], ["p2"]]
    assert [o.ok for o in out] == [True, False, True]
    assert out[1].too_long and "maximum model length" in out[1].error
    assert out[0].token_ids == (9, 7) and out[0].prompt_tokens == 3
    assert eng.stats["isolated_retries"] == 1
    ok = eng.generate_batch([vreq("p3")], [Sampling(8)])
    assert ok[0].ok and llm.calls[-1] == ["p3"]


def test_prompt_dict_shape():
    r = vreq("hello")
    d = r.prompt_dict()
    assert d["prompt"] == "hello" and len(d["multi_modal_data"]["video"]) == 1
    assert "mm_processor_kwargs" not in d


def test_dtbias_spec_needs_the_plugin_env(monkeypatch):
    monkeypatch.delenv("FLORA_DTBIAS_FP32", raising=False)
    from flora_deploy.engines.patches.dt_bias_fp32 import DtBiasActivationError

    with pytest.raises(DtBiasActivationError):
        O.OfflineEngine(spec(dtbias_fp32=True))  # refuses before importing vllm
    monkeypatch.setenv("FLORA_DTBIAS_FP32", "1")
    with pytest.raises(DtBiasActivationError):
        O.OfflineEngine(spec())  # plugin env on, spec off: refused too
