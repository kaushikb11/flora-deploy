"""Unit tests for the Gate C capture-layer prompt fix (``nv.capture.wrap_bare_video_tokens``)."""

import numpy as np
import pytest

from flora_deploy.nv.capture import (
    VIDEO_TOKEN,
    VISION_END,
    VISION_START,
    CapturedRequest,
    CaptureError,
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
