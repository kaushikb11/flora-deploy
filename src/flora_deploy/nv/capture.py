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
means our assumptions about NVIDIA's code no longer hold, so it raises. Note that
``temporal_inputs`` pops and returns the inputs, but ``three_view_caption_inputs``,
``processor_inputs`` and ``caption_single``'s inline call return the object as is, and
``pipeline.caption_*`` hands it straight to ``models.generate`` (which calls ``dict(inputs)`` and
raises). Driving ``pipeline.caption_*`` with a ``CaptureProcessor`` therefore requires replacing
``pipeline.generate``, as the nvlink tests do.

Process boundaries: ``CapturedRequest`` (str, uint8 arrays, plain dicts) pickles and is what
should cross them. ``CaptureProcessor`` also pickles/copies (its state is the real processor and
the captured requests), but shipping a whole HF processor between processes is rarely wanted.
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
    # ``max_new_tokens`` NVIDIA passes to ``models.generate`` for this request's model kind
    # (``nv.budget.budget_for``: 1024 TempSeg, 512 Captioner at the current pin); None if unknown.
    context_budget: int | None = None

    def check_context(self, prompt_token_ids_len: int, *, nv: Any = None) -> None:
        """NVIDIA's 32k guard for this request (``nv.budget.check_context_budget``)."""
        from flora_deploy.nv.budget import check_context_budget

        if self.context_budget is None:
            raise CaptureError("request has no context_budget; build the CaptureProcessor with "
                               "budget= (or CaptureProcessor.for_kind)")
        check_context_budget(prompt_token_ids_len, self.context_budget, nv=nv)

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
        arr = video.copy()  # never alias the caller's buffer
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

    ``requests`` keeps every captured call in order; ``captured`` is the latest one (use one
    proxy per call site, or ``pop_request``, when several episodes share a processor).
    ``budget`` is recorded as each request's ``context_budget``.
    """

    _OWN = ("_real", "requests", "_budget")

    def __init__(self, real: Any, *, budget: int | None = None):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "requests", [])
        object.__setattr__(self, "_budget", budget)

    @classmethod
    def for_kind(cls, nv: Any, real: Any, kind: str) -> CaptureProcessor:
        """Proxy whose requests carry NVIDIA's output budget for ``kind`` (from NVIDIA's
        ``pipeline.py`` call sites)."""
        from flora_deploy.nv.budget import budget_for

        return cls(real, budget=budget_for(nv, kind))

    @property
    def budget(self) -> int | None:
        return self._budget

    @property
    def captured(self) -> CapturedRequest | None:
        return self.requests[-1] if self.requests else None

    @property
    def real(self) -> Any:
        return self._real

    def __getattr__(self, name: str) -> Any:
        # Never delegate our own state or dunders: during unpickling/copying the instance has no
        # ``_real`` yet, and delegating would recurse forever.
        if name in self._OWN or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self._real, name)

    def __getstate__(self) -> dict:
        return {"real": self._real, "requests": list(self.requests), "budget": self._budget}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "_real", state["real"])
        object.__setattr__(self, "requests", list(state["requests"]))
        object.__setattr__(self, "_budget", state.get("budget"))

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
        self.requests.append(CapturedRequest(prompt=text[0], videos=pairs, call_kwargs=call_kwargs,
                                             context_budget=self._budget))
        return _CapturedInputs(
            list(video_metadata), has_metadata=bool(kwargs.get("return_metadata", False))
        )

    def pop_request(self) -> CapturedRequest:
        """Remove and return the oldest captured request."""
        if not self.requests:
            raise CaptureError("no processor call was captured")
        return self.requests.pop(0)


VIDEO_TOKEN = "<|video_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
PLACEHOLDER = "<|placeholder|>"
# Special tokens that must not occur in a prompt handed to the wrap: vision start/end (mixed
# shapes reorder HF's replacements), HF's internal ``<|placeholder|>`` (HF turns *every*
# occurrence into ``video_pad`` after expansion while vLLM tokenizes it as text, so the two
# diverge), and the image/vision pads (not part of Flora's video contract).
FORBIDDEN_PROMPT_TOKENS = (VISION_START, VISION_END, PLACEHOLDER, "<|image_pad|>",
                           "<|vision_pad|>")

# Engine settings every vLLM engine that carries Flora video must use (Gate C, docs/gate-c.md).
# Gate C certifies vLLM 0.24.0's uncached multimodal path only: with the processor cache on,
# vLLM tokenizes the prompt text-only and applies its prompt updates with cached timestamps, a
# different code path that Gate C does not exercise.
REQUIRED_VLLM_ENGINE_KWARGS: Mapping[str, Any] = {"mm_processor_cache_gb": 0}


def assert_engine_kwargs(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise ``CaptureError`` unless ``kwargs`` (``vllm.LLM`` / ``EngineArgs`` keywords) set every
    ``REQUIRED_VLLM_ENGINE_KWARGS`` value explicitly. Every engine builder that serves Flora
    video (the Gate C harness and the production TempSeg/Captioner builders) must call this on
    its final kwargs. Returns ``kwargs`` unchanged.

    The value must be a real number equal to the required one (``0`` or ``0.0`` for
    ``mm_processor_cache_gb``); ``False`` and other non-numbers are refused even though
    ``False == 0`` in Python (review 3 C5)."""
    for key, want in REQUIRED_VLLM_ENGINE_KWARGS.items():
        if key not in kwargs:
            raise CaptureError(f"vLLM engine kwargs must set {key}={want!r} (Gate C)")
        got = kwargs[key]
        if isinstance(got, bool) or not isinstance(got, (int, float)) or got != want:
            raise CaptureError(f"vLLM engine kwargs set {key}={got!r}; Gate C requires "
                               f"{want!r} (a number)")
    return kwargs


def wrap_bare_video_tokens(prompt: str, n_videos: int) -> str:
    """Gate C fix: bare ``<|video_pad|>`` -> ``<|vision_start|><|video_pad|><|vision_end|>``.

    Why: vLLM 0.24.0's Qwen3-VL processor only expands the full triplet (its ``PromptReplacement``
    target and ``_replace_video_token_placeholders`` both match ``vision_start, video_pad,
    vision_end``). A prompt with NVIDIA's bare token finds 0 placeholders for N videos and the
    request is rejected. HF's ``Qwen3VLProcessor`` accepts both shapes.

    Why the HF model inputs do not change (``docs/gate-c.md`` has the full argument and the test
    evidence): for each video in order, transformers 5.8.1 ``Qwen3VLProcessor.__call__`` replaces
    the first ``vision_start + video_token + vision_end`` if the text contains one, otherwise the
    first bare ``video_token``, with the same per-frame string
    ``<t seconds><|vision_start|><|placeholder|>*n<|vision_end|>``. The expansion writes
    ``<|placeholder|>`` and turns it back into ``video_token`` only after the loop, so while the
    loop runs no ``vision_start`` ever directly precedes a ``video_token`` (the input has none,
    and the expansions have none in that position). The bare prompt therefore takes the bare
    branch for every video and the wrapped prompt the wrapped branch; both replace the k-th
    marker with the k-th expansion, so the expanded texts and ``input_ids`` are identical.

    Preconditions (checked; ``CaptureError`` otherwise), which every NVIDIA builder meets:
    - the prompt has exactly ``n_videos`` (>= 1) video tokens and none of
      ``FORBIDDEN_PROMPT_TOKENS``: vision start/end would mix shapes and reorder HF's
      replacements; a literal ``<|placeholder|>`` (e.g. in a user instruction) becomes an extra
      ``video_pad`` in HF but stays text in vLLM;
    - each video token is preceded by a newline. vLLM tokenizes each ``<t seconds>`` string on
      its own, while HF tokenizes it in context. After a newline the tokenizer's pre-tokenizer
      split cannot join the two (locked by
      ``tests/nvlink/test_nv_link_capture_vllm_prompt_tokenizer.py``); after arbitrary text a
      token can span the boundary.
    """
    if VISION_START in prompt or VISION_END in prompt:
        raise CaptureError("prompt already contains vision start/end tokens; only bare "
                           "video tokens are supported")
    for token in FORBIDDEN_PROMPT_TOKENS:
        if token in prompt:
            raise CaptureError(f"prompt contains the special token {token}")
    if n_videos < 1:
        raise CaptureError("expected at least one video")
    if prompt.count(VIDEO_TOKEN) != n_videos:
        raise CaptureError(f"prompt has {prompt.count(VIDEO_TOKEN)} video tokens for "
                           f"{n_videos} videos")
    parts = prompt.split(VIDEO_TOKEN)
    for i, before in enumerate(parts[:-1]):
        if not before.endswith("\n"):
            raise CaptureError(f"video token {i} is not preceded by a newline")
    return prompt.replace(VIDEO_TOKEN, VISION_START + VIDEO_TOKEN + VISION_END)


def vllm_prompt(request: CapturedRequest) -> dict:
    """The vLLM prompt dict for a captured request (Gate C: wrapped video tokens).

    ``{"prompt": ..., "multi_modal_data": {"video": [(uint8 THWC, metadata dict), ...]}}``. The
    metadata dicts are passed as captured (``duration`` may be None; vLLM 0.24.0 accepts it).
    """
    return {
        "prompt": wrap_bare_video_tokens(request.prompt, len(request.videos)),
        "multi_modal_data": {"video": [(frames, dict(md)) for frames, md in request.videos]},
    }


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
