"""Engine protocol and the plain-data types that cross it (PLAN-v3.md §6.1).

A ``VllmRequest`` is one captured NVIDIA model call, already in vLLM's prompt shape (the Gate C
wrap applied by ``nv.capture.vllm_prompt``): templated text plus one ``(uint8 THWC, metadata)``
pair per video. ``Sampling`` carries NVIDIA's generation settings for the call; ``Completion``
is what came back, with enough detail to apply NVIDIA's own acceptance rules (EOS reached,
non-empty text, context budget) outside the engine.

Two protocols:

- ``BatchEngine.generate_batch(requests, samplings)``: synchronous, one call = one engine batch
  (``llm.generate``). Never raises for a single bad request: that request's ``Completion`` has
  ``error`` set (failure isolation); it raises only when the engine itself is unusable.
- ``Engine.generate(request, sampling)``: the awaitable per-request view that NVIDIA's
  ``apredict`` hook and the caption stage use. ``microbatch.MicroBatcher`` implements it on top
  of a ``BatchEngine``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Sampling:
    max_tokens: int
    temperature: float = 0.0
    n: int = 1
    top_p: float = 1.0
    top_k: int = 0
    regex: str | None = None
    logprobs: int | None = None
    seed: int | None = None
    # TempSeg / Captioner: NVIDIA stops on the tokenizer's EOS (<|im_end|>) only.
    stop_token_ids: tuple[int, ...] | None = None

    def as_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class VllmRequest:
    prompt: str  # templated text with *wrapped* video tokens (Gate C)
    videos: tuple[tuple[np.ndarray, dict], ...]  # uint8 (T, H, W, 3), vLLM video metadata
    key: str  # unique within a batch; for logs and observers only
    mm_kwargs: dict = field(default_factory=dict)  # per-request mm_processor_kwargs (empty)

    @property
    def nbytes(self) -> int:
        return int(sum(frames.nbytes for frames, _ in self.videos))

    def prompt_dict(self) -> dict:
        """The dict ``vllm.LLM.generate`` takes."""
        out: dict[str, Any] = {
            "prompt": self.prompt,
            "multi_modal_data": {"video": [(frames, dict(md)) for frames, md in self.videos]},
        }
        if self.mm_kwargs:
            out["mm_processor_kwargs"] = dict(self.mm_kwargs)
        return out


@dataclass(frozen=True)
class Completion:
    key: str
    token_ids: tuple[int, ...] = ()
    finish_reason: str | None = None  # "stop" | "length" | None (error)
    stop_reason: int | str | None = None
    prompt_tokens: int | None = None  # len(prompt_token_ids) as the engine saw it
    prompt_token_ids: tuple[int, ...] | None = None  # kept only when asked (L0 checks)
    logprobs: tuple[dict[int, float], ...] | None = None  # per generated token, when asked
    error: str | None = None  # this request failed in the engine (isolated)
    too_long: bool = False  # the engine rejected the prompt as longer than max_model_len

    @property
    def ok(self) -> bool:
        return self.error is None


class BatchEngine(Protocol):
    name: str

    def generate_batch(self, requests: Sequence[VllmRequest],
                       samplings: Sequence[Sampling]) -> list[Completion]: ...

    def audit(self) -> dict: ...


class Engine(Protocol):
    name: str

    async def generate(self, request: VllmRequest, sampling: Sampling) -> Completion: ...

    def audit(self) -> dict: ...
