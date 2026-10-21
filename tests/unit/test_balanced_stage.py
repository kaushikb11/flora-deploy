"""NVIDIA's ``models.generate`` acceptance on engine completions, the generate tap, records,
and the worker's error classification (no NVIDIA code needed)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from flora_deploy.engines.base import Completion
from flora_deploy.engines.microbatch import EngineBatchError, MicroBatcher
from flora_deploy.nv.budget import ContextBudgetError
from flora_deploy.nv.capture import CapturedRequest, CaptureProcessor
from flora_deploy.records import Provenance, StageRecord
from flora_deploy.stages import caption as C
from flora_deploy.stages import segment as S
from flora_deploy.workers_impl.segment import classify_error

EOS = 1


class InputErrorForModel(RuntimeError):
    pass


NV = SimpleNamespace(models=SimpleNamespace(InputErrorForModel=InputErrorForModel))


class Tok:
    eos_token_id = EOS

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i == EOS))

    def encode(self, text):
        return [ord(c) for c in text]


def call(budget=512):
    cap = CapturedRequest(prompt="x\n<|video_pad|>", videos=[
        (np.zeros((2, 4, 4, 3), np.uint8), {"total_num_frames": 2, "fps": 4.0,
                                            "frames_indices": [0, 1], "duration": None,
                                            "do_sample_frames": False})],
        context_budget=budget)
    return S.TappedCall(cap, budget, "s0")


def comp(text="hello", *, eos=True, prompt_tokens=100, finish="stop", **kw):
    ids = tuple(ord(c) for c in text) + ((EOS,) if eos else ())
    return Completion(key="k", token_ids=ids, finish_reason=finish if eos else "length",
                      prompt_tokens=prompt_tokens, **kw)


def test_accept_returns_nvidia_decoding():
    assert S.accept(NV, call(), comp("  a caption \n"), Tok(), EOS) == "a caption"


def test_accept_requires_eos_and_text():
    with pytest.raises(InputErrorForModel):
        S.accept(NV, call(), comp("cut off", eos=False), Tok(), EOS)
    with pytest.raises(InputErrorForModel):
        S.accept(NV, call(), comp("   "), Tok(), EOS)


def test_eos_by_finish_reason_when_not_in_ids():
    c = Completion(key="k", token_ids=tuple(map(ord, "ok")), finish_reason="stop",
                   stop_reason=None, prompt_tokens=10)
    assert S.accept(NV, call(), c, Tok(), EOS) == "ok"
    c = Completion(key="k", token_ids=tuple(map(ord, "ok")), finish_reason="stop",
                   stop_reason="STOPSTR", prompt_tokens=10)
    with pytest.raises(InputErrorForModel):
        S.accept(NV, call(), c, Tok(), EOS)


def test_context_guard_uses_vllm_prompt_length():
    S.accept(NV, call(512), comp(prompt_tokens=32768 - 512), Tok(), EOS)
    with pytest.raises(ContextBudgetError):
        S.accept(NV, call(512), comp(prompt_tokens=32768 - 511), Tok(), EOS)
    with pytest.raises(ContextBudgetError):
        S.accept(NV, call(), Completion(key="k", error="too long", too_long=True), Tok(), EOS)


def test_engine_error_is_infra():
    with pytest.raises(S.EngineRequestError):
        S.accept(NV, call(), Completion(key="k", error="CUDA error"), Tok(), EOS)
    assert classify_error(None, S.EngineRequestError("x")) == "infra"
    assert classify_error(None, EngineBatchError("x")) == "infra"
    assert classify_error(None, ContextBudgetError("x", prompt_tokens=1, budget=1,
                                                   limit=1)) == "nv_failure"
    assert classify_error(None, ValueError("NVIDIA InputError")) == "nv_failure"
    assert classify_error(None, KeyError("x")) == "infra"


def test_classify_nvidia_exception_classes():
    nv = SimpleNamespace(alias="flora_nv_segcap_78a0e4ac", _finder=SimpleNamespace(
        owns=lambda m: m.startswith("flora_runtime")))
    nvidia = type("FineSegmentationOutputError", (RuntimeError,),
                  {"__module__": "flora_runtime.video_segmentation.fine_subtask"})
    pipeline_err = type("InputErrorForModel", (RuntimeError,),
                        {"__module__": "flora_nv_segcap_78a0e4ac.models"})
    assert classify_error(nv, nvidia("x")) == "nv_failure"
    assert classify_error(nv, pipeline_err("x")) == "nv_failure"
    assert classify_error(nv, RuntimeError("x")) == "infra"


class FakePipeline:
    """Stands in for NVIDIA's ``pipeline``: a function that builds 3 captured calls in order."""

    def __init__(self):
        self.generate = self.original_generate

    @staticmethod
    def original_generate(model, proc, inputs, budget):
        return "ORIGINAL"


class EchoEngine:
    name = "echo"

    def __init__(self, answers):
        self.answers = answers
        self.batches = []

    def generate_batch(self, requests, samplings):
        self.batches.append([r.prompt for r in requests])
        return [comp(self.answers[S.unwrap_prompt(r.prompt)]) for r in requests]


def test_tap_collects_calls_and_restores():
    nv = SimpleNamespace(pipeline=FakePipeline())
    real = SimpleNamespace(tokenizer=Tok())
    tap = S.GenerateTap.acquire(nv)
    tap2 = S.GenerateTap.acquire(nv)
    assert tap is tap2 and nv.pipeline.generate is tap.replacement
    assert nv.pipeline.generate(None, None, None, 1) == "ORIGINAL"  # outside a collector

    def nvidia_fn():
        proc = CaptureProcessor(real, budget=512)
        out = []
        for i in range(3):
            inputs = proc(text=[f"p{i}\n<|video_pad|>"],
                          videos=[np.zeros((1, 2, 2, 3), np.uint8)],
                          video_metadata=[{"total_num_frames": 1, "fps": 1.0,
                                           "frames_indices": [0]}], do_sample_frames=False)
            out.append(nv.pipeline.generate(None, proc, inputs, 512))
        return out

    eng = EchoEngine({f"p{i}\n<|video_pad|>": f"text {i}" for i in range(3)})

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb, mb.episode():
            return await S.run_tapped(NV, mb, nvidia_fn, tokenizer=Tok(), eos=EOS)

    sentinels, texts = asyncio.run(main())
    assert [texts[s] for s in sentinels] == ["text 0", "text 1", "text 2"]
    assert len(eng.batches) == 1 and len(eng.batches[0]) == 3  # one round for the 3 calls
    assert all("<|vision_start|><|video_pad|><|vision_end|>" in p for p in eng.batches[0])
    tap.release()
    assert nv.pipeline.generate is tap.replacement
    tap2.release()
    assert nv.pipeline.generate == FakePipeline.original_generate


def test_first_failure_in_call_order_raises():
    nv = SimpleNamespace(pipeline=FakePipeline(), models=NV.models)
    real = SimpleNamespace(tokenizer=Tok())
    tap = S.GenerateTap.acquire(nv)

    def nvidia_fn():
        proc = CaptureProcessor(real, budget=8)
        for i in range(2):
            inputs = proc(text=[f"q{i}\n<|video_pad|>"],
                          videos=[np.zeros((1, 2, 2, 3), np.uint8)],
                          video_metadata=[{"total_num_frames": 1, "fps": 1.0,
                                           "frames_indices": [0]}], do_sample_frames=False)
            nv.pipeline.generate(None, proc, inputs, 8)

    class Eng(EchoEngine):
        def generate_batch(self, requests, samplings):
            return [comp("x", eos=False), comp("fine")]

    async def main():
        async with MicroBatcher(Eng({}), max_wait_s=30) as mb:
            return await S.run_tapped(nv, mb, nvidia_fn, tokenizer=Tok(), eos=EOS)

    try:
        with pytest.raises(InputErrorForModel):
            asyncio.run(main())
    finally:
        tap.release()


def prov():
    return Provenance(models={"tempseg": "r"}, engine="vllm-0.24.0-offline", preset="balanced",
                      config_hash="c", logic_version="l", framework="f")


def test_segment_record_round_trip():
    rows = [dict(start_s=0.5, end_s=2.25, metadata={"fine_event_id": "f0",
                                                     "segmentation_provenance": "fine",
                                                     "u2_context": {"before": None}}),
            dict(start_s=2.25, end_s=4.0, metadata={})]
    rec = S.segment_record(rows, uid="d--e", dataset="d", episode_id="e", provenance=prov())
    assert rec.certified is False and rec.uncertified_reason == "uncertified_stack"
    again = StageRecord.from_json(rec.to_json())
    assert S.rows_from_record(again) == [dict(start_s=0.5, end_s=2.25,
                                              metadata=rows[0]["metadata"]),
                                         dict(start_s=2.25, end_s=4.0, metadata={})]
    assert again.segments[0].fine_event_id == "f0"
    assert again.segments[1].segmentation_provenance == "fine"


def test_caption_record_shape():
    rows = [dict(start_s=0.0, end_s=1.0, caption="pick", status="complete",
                 segmentation_provenance="fine"),
            dict(start_s=1.0, end_s=2.0, caption="", status="unsupported_target_frames",
                 segmentation_provenance="fine_conservative_completion")]
    rec = C.caption_record(rows, uid="d--e", dataset="d", episode_id="e", provenance=prov())
    assert [s.status for s in rec.segments] == ["complete", "unsupported_target_frames"]
    assert rec.segments[0].caption == "pick" and rec.stage == "caption"
    StageRecord.from_json(rec.to_json())


def test_config_hash_changes_with_engine_and_ignores_gpu():
    from flora_deploy.engines.offline_llm import EngineSpec

    a = EngineSpec.for_gpu("tempseg", "/m", "L40S")
    b = EngineSpec.for_gpu("tempseg", "/other", "L40S")
    c = EngineSpec.for_gpu("tempseg", "/m", "L40S", dtbias_fp32=True)
    h = [S.config_hash("tempseg", spec=s.hashed(), models={"m": "1"}, budget=1024)
         for s in (a, b, c)]
    assert h[0] == h[1] != h[2]
