"""Unit tests for the Gate C capture-layer prompt fix (``nv.capture.wrap_bare_video_tokens``)."""

import numpy as np
import pytest

from flora_deploy.nv.capture import (
    FORBIDDEN_PROMPT_TOKENS,
    PLACEHOLDER,
    REQUIRED_VLLM_ENGINE_KWARGS,
    VIDEO_TOKEN,
    VISION_END,
    VISION_START,
    CapturedRequest,
    CaptureError,
    assert_engine_kwargs,
    vllm_prompt,
    wrap_bare_video_tokens,
)

V = VIDEO_TOKEN
W = VISION_START + VIDEO_TOKEN + VISION_END


def test_wraps_every_bare_token_in_place():
    p = f"head\nA:\n{V}\n\nB:\n{V}\n\ntail"
    assert wrap_bare_video_tokens(p, 2) == f"head\nA:\n{W}\n\nB:\n{W}\n\ntail"


def test_rejects_count_mismatch_and_existing_wrappers():
    with pytest.raises(CaptureError, match="2 video tokens for 1"):
        wrap_bare_video_tokens(f"\n{V}\n{V}", 1)
    with pytest.raises(CaptureError, match="vision start/end"):
        wrap_bare_video_tokens(f"\n{W}", 1)
    with pytest.raises(CaptureError, match="vision start/end"):
        wrap_bare_video_tokens(f"\n{V}{VISION_END}", 1)


def test_rejects_token_not_after_newline():
    with pytest.raises(CaptureError, match="video token 1 is not preceded by a newline"):
        wrap_bare_video_tokens(f"\n{V} and {V}", 2)


def test_vllm_prompt_dict_keeps_frames_and_metadata():
    frames = np.zeros((2, 4, 4, 3), np.uint8)
    md = dict(total_num_frames=2, fps=4, frames_indices=[0, 1], duration=None,
              do_sample_frames=False)
    req = CapturedRequest(prompt=f"x\n{V}", videos=[(frames, md)])
    d = vllm_prompt(req)
    assert d["prompt"] == f"x\n{W}"
    (f, m), = d["multi_modal_data"]["video"]
    assert f is frames and m == md and m is not md
    assert set(d) == {"prompt", "multi_modal_data"}


@pytest.mark.parametrize("token", [PLACEHOLDER, "<|image_pad|>", "<|vision_pad|>"])
def test_rejects_other_special_tokens(token):
    # e.g. a user instruction carrying HF's internal placeholder: HF turns it into an extra
    # video_pad while vLLM keeps it as text, so the two would diverge
    assert token in FORBIDDEN_PROMPT_TOKENS
    with pytest.raises(CaptureError, match="special token"):
        wrap_bare_video_tokens(f"x\n{V}\n{{\"note\":\"{token}\"}}", 1)


def test_rejects_zero_videos_and_first_token_without_newline():
    with pytest.raises(CaptureError, match="at least one video"):
        wrap_bare_video_tokens("no video here", 0)
    with pytest.raises(CaptureError, match="video token 0 is not preceded by a newline"):
        wrap_bare_video_tokens(f"A: {V}\n{V}", 2)
    with pytest.raises(CaptureError, match="video token 0"):
        wrap_bare_video_tokens(V, 1)


def test_engine_kwargs_must_disable_processor_cache():
    assert REQUIRED_VLLM_ENGINE_KWARGS == {"mm_processor_cache_gb": 0}
    ok = {"model": "m", "mm_processor_cache_gb": 0}
    assert assert_engine_kwargs(ok) is ok
    with pytest.raises(CaptureError, match="must set mm_processor_cache_gb=0"):
        assert_engine_kwargs({"model": "m"})  # vLLM's default is a 4 GiB cache
    with pytest.raises(CaptureError, match="mm_processor_cache_gb=4"):
        assert_engine_kwargs({"mm_processor_cache_gb": 4})


@pytest.mark.parametrize("value", [0, 0.0])
def test_engine_kwargs_accept_numeric_zero(value):
    assert assert_engine_kwargs({"mm_processor_cache_gb": value})


@pytest.mark.parametrize("value", [False, True, None, "0", 0.5, float("nan"), [0]])
def test_engine_kwargs_reject_non_numbers_and_nonzero(value):
    """Review 3 C5: ``False == 0`` must not pass, nor anything that is not the number 0."""
    with pytest.raises(CaptureError, match="mm_processor_cache_gb"):
        assert_engine_kwargs({"mm_processor_cache_gb": value})


# Files allowed to construct a vLLM engine directly; each must pass its kwargs through
# nv.capture.assert_engine_kwargs. NVIDIA's Rewarder builds its own engine via load_llm (no
# Flora video through our capture layer).
ENGINE_BUILDERS = {"modal_app/gatec_app.py", "engines/offline_llm.py"}
ENGINE_CALLS = {"LLM", "AsyncLLM", "EngineArgs", "AsyncEngineArgs", "AsyncLLMEngine",
                "LLMEngine"}


def test_no_unchecked_vllm_engine_construction():
    """Review 3 C5: scan src/ for ``LLM(`` / ``EngineArgs(`` style calls; any outside the allowed
    builders fails, and every allowed builder calls ``assert_engine_kwargs``."""
    import ast

    from flora_deploy.config import logic

    root = logic.package_root()
    found: dict[str, list[int]] = {}
    for rel in sorted(logic.package_modules()):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(
                f, ast.Attribute) else None
            if name in ENGINE_CALLS:
                found.setdefault(rel, []).append(node.lineno)
    assert set(found) <= ENGINE_BUILDERS, {k: v for k, v in found.items()
                                           if k not in ENGINE_BUILDERS}
    for rel in ENGINE_BUILDERS:
        assert "assert_engine_kwargs(" in (root / rel).read_text(encoding="utf-8"), rel


def test_gatec_engine_builder_asserts_cache_off():
    pytest.importorskip("modal")
    from flora_deploy.modal_app.gatec_app import engine_kwargs

    for kind in ("tempseg", "captioner"):
        assert engine_kwargs(kind, "/m")["mm_processor_cache_gb"] == 0
