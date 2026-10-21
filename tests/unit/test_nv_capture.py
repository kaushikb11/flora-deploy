"""CaptureProcessor / _CapturedInputs logic with a fake processor (no transformers needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image

from flora_deploy.nv.capture import (
    CapturedInputsAccessError,
    CaptureError,
    CaptureProcessor,
    frames_to_uint8,
    metadata_to_dict,
)


@dataclass
class VideoMetadata:  # same fields as transformers.video_utils.VideoMetadata
    total_num_frames: int
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    video_backend: str | None = None
    frames_indices: list[int] | None = None


class FakeProc:
    video_token = "<|video_pad|>"

    def __init__(self):
        self.tokenizer = object()
        self.chat_template = "t"
        self.called = False

    def apply_chat_template(self, messages, **kw):
        return "<user>" + messages[0]["content"] + "</user>"

    def __call__(self, **kw):
        self.called = True
        return {"input_ids": [1]}


def pil_video(n=3, size=(8, 6), seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
            for _ in range(n)]


def md(n=3, fps=4.0):
    return VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)))


def test_delegates_attributes_and_records_call():
    real = FakeProc()
    cap = CaptureProcessor(real)
    assert cap.video_token == "<|video_pad|>"
    assert cap.tokenizer is real.tokenizer
    text = cap.apply_chat_template([dict(role="user", content="x " + cap.video_token)])
    video = pil_video()
    out = cap(text=[text], videos=[video], video_metadata=[md()], do_sample_frames=False,
              return_metadata=True, return_tensors="pt")
    assert not real.called
    req = cap.captured
    assert req.prompt == text
    (frames, meta), = req.videos
    assert frames.dtype == np.uint8 and frames.shape == (3, 6, 8, 3)
    assert np.array_equal(frames[1], np.asarray(video[1]))
    assert meta == dict(total_num_frames=3, fps=4.0, frames_indices=[0, 1, 2], duration=None,
                        do_sample_frames=False)
    assert req.call_kwargs == {"return_metadata": True, "return_tensors": "pt"}
    popped = out.pop("video_metadata", None)
    assert popped[0].total_num_frames == 3
    assert out.pop("video_metadata", None) is None  # popped once, like a dict


def test_setattr_goes_to_real_processor():
    real = FakeProc()
    cap = CaptureProcessor(real)
    cap.chat_template = "new"
    assert real.chat_template == "new"


def test_multiple_requests_in_order():
    cap = CaptureProcessor(FakeProc())
    for i in range(3):
        cap(text=[f"p{i}"], videos=[pil_video(seed=i)], video_metadata=[md()],
            do_sample_frames=False)
    assert [r.prompt for r in cap.requests] == ["p0", "p1", "p2"]
    assert cap.pop_request().prompt == "p0"
    assert len(cap.requests) == 2


def test_without_return_metadata_pop_returns_default():
    cap = CaptureProcessor(FakeProc())
    out = cap(text=["p"], videos=[pil_video(), pil_video(2)], video_metadata=[md(), md(2)],
              do_sample_frames=False, return_tensors="pt", padding=True)
    assert out.pop("video_metadata", "absent") == "absent"
    with pytest.raises(KeyError):
        out.pop("video_metadata")
    assert [f.shape[0] for f, _ in cap.captured.videos] == [3, 2]


def test_captured_inputs_refuse_everything_else():
    cap = CaptureProcessor(FakeProc())
    out = cap(text=["p"], videos=[pil_video()], video_metadata=[md()], do_sample_frames=False)
    for op in (
        lambda: out["input_ids"],
        lambda: dict(out),
        lambda: list(out),
        lambda: len(out),
        lambda: "input_ids" in out,
        lambda: out.input_ids,
        lambda: out.pop("input_ids", None),
        lambda: out.keys(),
    ):
        with pytest.raises(CapturedInputsAccessError, match="captured processor output"):
            op()
    with pytest.raises(CapturedInputsAccessError):
        out.x = 1
    assert not hasattr(out, "input_ids")  # AttributeError subclass


@pytest.mark.parametrize("kwargs, match", [
    (dict(do_sample_frames=True), "do_sample_frames=False"),
    ({}, "do_sample_frames=False"),
    (dict(do_sample_frames=False, images=[1]), "image inputs"),
    (dict(do_sample_frames=False, fps=2), "unsupported processor kwargs"),
])
def test_rejects_unsupported_calls(kwargs, match):
    cap = CaptureProcessor(FakeProc())
    with pytest.raises(CaptureError, match=match):
        cap(text=["p"], videos=[pil_video()], video_metadata=[md()], **kwargs)


def test_rejects_bad_shapes():
    cap = CaptureProcessor(FakeProc())
    with pytest.raises(CaptureError, match="exactly one prompt"):
        cap(text=["a", "b"], videos=[pil_video()], video_metadata=[md()], do_sample_frames=False)
    with pytest.raises(CaptureError, match="one video_metadata entry per video"):
        cap(text=["a"], videos=[pil_video()], video_metadata=[], do_sample_frames=False)
    with pytest.raises(CaptureError, match="frames_indices"):
        cap(text=["a"], videos=[pil_video(3)], video_metadata=[md(4)], do_sample_frames=False)
    with pytest.raises(CaptureError, match="positional"):
        cap("a", videos=[pil_video()], video_metadata=[md()], do_sample_frames=False)


def test_frames_to_uint8_inputs():
    video = pil_video(2)
    stacked = frames_to_uint8(video)
    assert np.array_equal(frames_to_uint8([np.asarray(v) for v in video]), stacked)
    assert np.array_equal(frames_to_uint8(stacked), stacked)
    assert frames_to_uint8(stacked).flags["C_CONTIGUOUS"]
    with pytest.raises(CaptureError, match="mode 'L'"):
        frames_to_uint8([video[0].convert("L")])
    with pytest.raises(CaptureError, match="differ in shape"):
        frames_to_uint8([video[0], video[0].resize((4, 4))])
    with pytest.raises(CaptureError, match="uint8"):
        frames_to_uint8(stacked.astype(np.float32))
    with pytest.raises(CaptureError, match="non-empty"):
        frames_to_uint8([])
    with pytest.raises(CaptureError, match="unsupported type"):
        frames_to_uint8([object()])


def test_metadata_to_dict():
    assert metadata_to_dict(dict(total_num_frames=2, fps=8.0, frames_indices=(0, 3))) == dict(
        total_num_frames=2, fps=8.0, frames_indices=[0, 3], duration=None, do_sample_frames=False
    )
    with_size = VideoMetadata(total_num_frames=1, fps=1.0, frames_indices=[0], width=4)
    assert metadata_to_dict(with_size)["width"] == 4
    with pytest.raises(CaptureError, match="unexpected"):
        metadata_to_dict(dict(total_num_frames=1, fps=1, frames_indices=[0], foo=1))
    with pytest.raises(CaptureError, match="needs"):
        metadata_to_dict(dict(total_num_frames=1))
    with pytest.raises(CaptureError, match="sampling"):
        metadata_to_dict(dict(total_num_frames=1, fps=1, frames_indices=[0],
                              do_sample_frames=True))
