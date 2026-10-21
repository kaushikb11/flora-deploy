"""Capture the processor call NVIDIA's input builders make, as a vLLM-ready request (PLAN.md §6.2).

NVIDIA's builders (``preprocessing.temporal_inputs``, ``preprocessing.three_view_caption_inputs``,
``flora_unified.unified_input.processor_inputs``) all end in one call::

    proc(text=[text], videos=[...], video_metadata=[...], do_sample_frames=False, ...)

Handing them a ``CaptureProcessor`` instead of the real processor lets NVIDIA's code do all of its
own prompt building, frame selection and resizing. We record that final call as a
``CapturedRequest``: the templated prompt, each video as a ``uint8`` ``(T, H, W, 3)`` array, and a
plain metadata dict per video. Everything else (``video_token``, ``apply_chat_template``,
``tokenizer``, ...) is delegated to the real processor.

The value returned to NVIDIA's code is a ``_CapturedInputs``. NVIDIA's builders only ever do
``inputs.pop('video_metadata', None)`` on it; any other use (indexing, ``dict()``, attribute access)
means our assumptions about NVIDIA's code no longer hold, so it raises.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Keyword arguments NVIDIA's builders pass that do not change the model inputs we capture.
# (``return_metadata`` only controls whether ``video_metadata`` is echoed back; ``padding`` is
# irrelevant for a batch of one prompt; ``return_tensors`` is the output container type.)
ALLOWED_CALL_KWARGS = frozenset(
    {"do_sample_frames", "return_metadata", "return_tensors", "padding"}
)
METADATA_FIELDS = ("total_num_frames", "fps", "frames_indices", "duration")
_OPTIONAL_METADATA_FIELDS = ("width", "height", "video_backend")


class CaptureError(RuntimeError):
    """NVIDIA's processor call did not have the shape the capture layer supports."""


class CapturedInputsAccessError(AttributeError, TypeError):
    """NVIDIA's code used the processor output for something other than
    ``.pop('video_metadata', default)``."""


@dataclass(frozen=True)
class CapturedRequest:
    prompt: str
    # One entry per video: (uint8 array (T, H, W, 3), metadata dict with total_num_frames, fps,
    # frames_indices, duration, do_sample_frames=False).
    videos: list[tuple[np.ndarray, dict]]
    # The remaining keyword arguments of the call (return_tensors, padding, return_metadata).
    call_kwargs: dict = field(default_factory=dict)

    def video_metadata_objects(self) -> list[Any]:
        """Metadata as transformers ``VideoMetadata`` objects (for HF replay)."""
        from transformers.video_utils import VideoMetadata

        return [
            VideoMetadata(**{k: v for k, v in md.items() if k != "do_sample_frames"})
            for _, md in self.videos
        ]


def frames_to_uint8(video: Any) -> np.ndarray:
    """Convert one video (list of PIL images / HxWx3 arrays, or a THWC array) to uint8 THWC.

    Conversion is lossless and deterministic: PIL ``RGB`` images go through ``np.asarray``. Any
    other PIL mode is rejected rather than converted, because the real processor would apply
    its own conversion and the two could differ.
    """
    if isinstance(video, np.ndarray):
        arr = video
        if arr.ndim != 4:
            raise CaptureError(f"video array must be (T, H, W, 3), got shape {arr.shape}")
    else:
        if not isinstance(video, Sequence) or isinstance(video, str | bytes) or not video:
            raise CaptureError(f"video must be a non-empty list of frames, got {type(video)!r}")
        frames = []
        for i, frame in enumerate(video):
            if isinstance(frame, np.ndarray):
                frames.append(frame)
                continue
            mode = getattr(frame, "mode", None)
            if mode is None or not hasattr(frame, "size"):
                raise CaptureError(f"frame {i}: unsupported type {type(frame)!r}")
            if mode != "RGB":
                raise CaptureError(f"frame {i}: PIL mode {mode!r}, expected 'RGB'")
            frames.append(np.asarray(frame))
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise CaptureError(f"frames of one video differ in shape: {sorted(shapes)}")
        arr = np.stack(frames)
    if arr.dtype != np.uint8:
        raise CaptureError(f"frames must be uint8, got {arr.dtype}")
    if arr.shape[-1] != 3:
        raise CaptureError(f"frames must have 3 channels, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def metadata_to_dict(md: Any) -> dict:
    """``VideoMetadata`` (or a mapping) -> plain dict for vLLM, with ``do_sample_frames=False``."""
    if dataclasses.is_dataclass(md) and not isinstance(md, type):
        src = {f.name: getattr(md, f.name) for f in dataclasses.fields(md)}
    elif isinstance(md, Mapping):
        src = dict(md)
    else:
        raise CaptureError(f"unsupported video metadata type {type(md)!r}")
    known = set(METADATA_FIELDS) | set(_OPTIONAL_METADATA_FIELDS) | {"do_sample_frames"}
    unknown = set(src) - known
    if unknown:
        raise CaptureError(f"unexpected video metadata fields {sorted(unknown)}")
    if src.get("do_sample_frames", False) is not False:
        raise CaptureError("video metadata asks for frame sampling")
    out: dict = {}
    for key in METADATA_FIELDS:
        out[key] = src.get(key)
    if out["total_num_frames"] is None or out["fps"] is None or out["frames_indices"] is None:
        raise CaptureError("video metadata needs total_num_frames, fps and frames_indices")
    out["total_num_frames"] = int(out["total_num_frames"])
    out["frames_indices"] = [int(i) for i in out["frames_indices"]]
    for key in _OPTIONAL_METADATA_FIELDS:
        if src.get(key) is not None:
            out[key] = src[key]
    out["do_sample_frames"] = False
    return out


class _CapturedInputs:
    """Stand-in for the processor's ``BatchFeature``: only ``.pop('video_metadata', d)`` works."""

    __slots__ = ("_metadata", "_has_metadata")

    def __init__(self, metadata: list | None, has_metadata: bool):
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_has_metadata", has_metadata)

    def pop(self, key: str, *default: Any) -> Any:
        if key != "video_metadata":
            raise CapturedInputsAccessError(
                f"captured processor output: pop({key!r}) is not supported; only "
                "pop('video_metadata', default) is (the model inputs exist only after vLLM runs)"
            )
        if len(default) > 1:
            raise TypeError("pop expected at most 2 arguments")
        if self._has_metadata:
            object.__setattr__(self, "_has_metadata", False)
            return self._metadata
        if default:
            return default[0]
        raise KeyError(key)

    def _deny(self, what: str):
        raise CapturedInputsAccessError(
            f"captured processor output does not support {what}; NVIDIA's input builders were "
            "expected to call only .pop('video_metadata', default)"
        )

    def __getattr__(self, name: str):
        self._deny(f"attribute {name!r}")

    def __setattr__(self, name: str, value: Any):
        self._deny(f"setting {name!r}")

    def __getitem__(self, key):
        self._deny(f"indexing [{key!r}]")

    def __setitem__(self, key, value):
        self._deny(f"item assignment [{key!r}]")

    def __contains__(self, key):
        self._deny("'in'")

    def __iter__(self):
        self._deny("iteration / dict()")

    def __len__(self):
        self._deny("len()")

    def __repr__(self) -> str:
        return "<_CapturedInputs (pop('video_metadata') only)>"


class CaptureProcessor:
    """Proxy for the real HF processor that records the final ``__call__``.

    ``requests`` keeps every captured call in order; ``captured`` is the latest one.
    """

    def __init__(self, real: Any):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "requests", [])

    @property
    def captured(self) -> CapturedRequest | None:
        return self.requests[-1] if self.requests else None

    @property
    def real(self) -> Any:
        return self._real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # e.g. NVIDIA's models.processor() sets ``chat_template``; keep the real one in sync.
        setattr(self._real, name, value)

    def __call__(self, *args, text=None, videos=None, video_metadata=None, **kwargs):
        if args:
            raise CaptureError("processor called with positional arguments; expected keywords")
        if "images" in kwargs:
            raise CaptureError("image inputs are not supported by the capture layer")
        unknown = set(kwargs) - ALLOWED_CALL_KWARGS
        if unknown:
            raise CaptureError(f"unsupported processor kwargs {sorted(unknown)}")
        if kwargs.get("do_sample_frames") is not False:
            raise CaptureError("NVIDIA's builders always pass do_sample_frames=False")
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, list | tuple) or len(text) != 1 or not isinstance(text[0], str):
            raise CaptureError("expected text=[prompt] with exactly one prompt")
        if not videos:
            raise CaptureError("expected at least one video")
        if video_metadata is None or len(video_metadata) != len(videos):
            raise CaptureError("expected one video_metadata entry per video")
        pairs = []
        for video, md in zip(videos, video_metadata, strict=True):
            frames = frames_to_uint8(video)
            md_dict = metadata_to_dict(md)
            if len(md_dict["frames_indices"]) != len(frames):
                raise CaptureError(
                    f"metadata has {len(md_dict['frames_indices'])} frames_indices but the video "
                    f"has {len(frames)} frames"
                )
            pairs.append((frames, md_dict))
        call_kwargs = {k: v for k, v in kwargs.items() if k != "do_sample_frames"}
        self.requests.append(CapturedRequest(prompt=text[0], videos=pairs, call_kwargs=call_kwargs))
        return _CapturedInputs(
            list(video_metadata), has_metadata=bool(kwargs.get("return_metadata", False))
        )

    def pop_request(self) -> CapturedRequest:
        """Remove and return the oldest captured request."""
        if not self.requests:
            raise CaptureError("no processor call was captured")
        return self.requests.pop(0)


def replay(proc: Any, request: CapturedRequest, **overrides: Any):
    """Run a captured request through a real HF processor with ndarray frames (as vLLM would
    receive them). Returns the processor's ``BatchFeature``."""
    kwargs = dict(request.call_kwargs)
    kwargs.pop("return_metadata", None)
    kwargs.setdefault("return_tensors", "pt")
    kwargs.update(overrides)
    return proc(
        text=[request.prompt],
        videos=[frames for frames, _ in request.videos],
        video_metadata=request.video_metadata_objects(),
        do_sample_frames=False,
        **kwargs,
    )
