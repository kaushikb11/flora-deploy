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
    # review 6 M5: bisection, not one request at a time
    assert llm.calls == [["p0", "p1", "p2"], ["p0"], ["p1", "p2"], ["p1"], ["p2"]]
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



def test_batch_failure_bisects_instead_of_running_every_request_alone():
    """Review 6 M5: one over-long prompt in a 64-request round costs O(log n) generates, not
    64 serial ones."""
    llm = FakeLLM(bad={"p37"})
    eng = O.OfflineEngine(spec(), llm, params_cls=Params)
    out = eng.generate_batch([vreq(f"p{i}") for i in range(64)], [Sampling(8)] * 64)
    assert [o.ok for o in out] == [i != 37 for i in range(64)]
    assert out[37].too_long
    assert len(llm.calls) <= 1 + 2 * 6, len(llm.calls)
    assert max(len(c) for c in llm.calls[1:]) == 32  # batch-mates stay batched


class DeadLLM(FakeLLM):
    def generate(self, prompts, params, use_tqdm=True):
        self.calls.append([p["prompt"] for p in prompts])
        raise RuntimeError("EngineDeadError: EngineCore encountered an issue. See stack trace "
                           "(above) for the root cause.")


@pytest.mark.parametrize("msg_cls", [DeadLLM])
def test_dead_engine_fails_fast(msg_cls):
    llm = msg_cls()
    eng = O.OfflineEngine(spec(), llm, params_cls=Params)
    with pytest.raises(O.EngineDead):
        eng.generate_batch([vreq(f"p{i}") for i in range(8)], [Sampling(8)] * 8)
    assert len(llm.calls) == 1  # no per-request retries on a dead engine
    with pytest.raises(O.EngineDead):
        eng.generate_batch([vreq("q")], [Sampling(8)])
    assert len(llm.calls) == 1  # later rounds fail without touching the engine


def test_cuda_fault_in_a_batch_is_a_dead_engine():
    class CudaLLM(FakeLLM):
        def generate(self, prompts, params, use_tqdm=True):
            self.calls.append(1)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    eng = O.OfflineEngine(spec(), CudaLLM(), params_cls=Params)
    with pytest.raises(O.EngineDead):
        eng.generate_batch([vreq("a"), vreq("b")], [Sampling(8)] * 2)


def test_dead_engine_classifies_as_cuda_fatal():
    from flora_deploy.engines.microbatch import EngineBatchError
    from flora_deploy.workers_impl.segment import _classify

    exc = EngineBatchError("EngineDead: EngineDeadError: EngineCore encountered an issue")
    assert _classify(exc) == "cuda_fatal"



class RpcLLM(FakeLLM):
    """A fake engine whose workers report GDN parameter dtypes (``collective_rpc``)."""

    def __init__(self, dt_bias="bfloat16", a_log="float32", n=48):
        super().__init__()
        self.params = [(f"model.layers.{i}.linear_attn.dt_bias", dt_bias) for i in range(n)] + \
            [(f"model.layers.{i}.linear_attn.A_log", a_log) for i in range(n)]

    def collective_rpc(self, fn):
        params = [(name, SimpleNamespace(dtype=f"torch.{dt}")) for name, dt in self.params]
        worker = SimpleNamespace(model_runner=SimpleNamespace(
            model=SimpleNamespace(named_parameters=lambda: iter(params))))
        return [fn(worker)]


@pytest.mark.parametrize(("dtbias", "dt", "a_log", "n", "ok"), [
    (False, "bfloat16", "float32", 48, True),
    (True, "float32", "float32", 48, True),
    (True, "bfloat16", "float32", 48, False),  # plugin load failed silently in vLLM
    (False, "float32", "float32", 48, False),  # stray plugin
    (False, "bfloat16", "bfloat16", 48, False),
    (True, "float32", "float32", 0, False),  # no GDN parameter found at all
])
def test_engine_build_verifies_gdn_dtypes(monkeypatch, dtbias, dt, a_log, n, ok):
    """Review 6 M6: vLLM swallows plugin load errors, so the built engine's dtypes are
    checked against the spec."""
    from flora_deploy.engines.patches import dt_bias_fp32

    monkeypatch.setattr(dt_bias_fp32, "check_activation", lambda want: None)
    llm = RpcLLM(dt, a_log, n)
    if ok:
        eng = O.OfflineEngine(spec(dtbias_fp32=dtbias), llm, params_cls=Params, verify=True)
        assert eng.gdn_dtypes["workers"][0][f"dt_bias:{dt}"] == n
    else:
        with pytest.raises(O.EngineVerifyError):
            O.OfflineEngine(spec(dtbias_fp32=dtbias), llm, params_cls=Params, verify=True)


def test_rpc_failure_fails_verification():
    with pytest.raises(O.EngineVerifyError):
        O.verify_gdn_dtypes(spec(), {"error": "RuntimeError('no rpc')"})


def test_extra_state_under_vision_tower_fails_preflight():
    """Review 6 Q5: vLLM's vision loader does a bare ``params_dict[name]``: any
    ``model.visual.*._extra_state`` key would KeyError at load."""
    lm = {"model.language_model.layers.0.mlp.down_proj._extra_state": "a.safetensors",
          "model.language_model.layers.0.mlp.down_proj.weight": "a.safetensors"}
    assert O.check_extra_state(lm) == ["model.language_model"]
    bad = {**lm, "model.visual.blocks.0.mlp.fc1._extra_state": "b.safetensors"}
    with pytest.raises(O.EngineVerifyError, match="model.visual"):
        O.check_extra_state(bad)
