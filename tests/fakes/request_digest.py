"""A literal digest of what ``vllm.LLM.generate`` receives for a ``VllmRequest`` (review 10 L3).

``stages.segment.request_digest`` hashes the *bare* prompt, the frames and their metadata. This
one hashes ``VllmRequest.prompt_dict()`` as sent: the **wrapped** prompt, every
``multi_modal_data`` entry with its array dtype, shape and bytes plus its metadata, and
``mm_processor_kwargs``. The accel identity test asserts it, so a change in any of those is
caught.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _canon(x: Any, h: Any) -> None:
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h.update(f"nd:{a.dtype.str}:{a.shape}:".encode())
        h.update(a.tobytes())
    elif isinstance(x, dict):
        h.update(b"{")
        for k in sorted(x, key=str):
            h.update(json.dumps(str(k)).encode() + b":")
            _canon(x[k], h)
        h.update(b"}")
    elif isinstance(x, list | tuple):
        h.update(b"[")
        for v in x:
            _canon(v, h)
            h.update(b",")
        h.update(b"]")
    else:
        h.update(f"{type(x).__name__}:".encode()
                 + json.dumps(x, sort_keys=True, default=str).encode())


def prompt_dict_digest(request: Any) -> str:
    """sha256 over ``request.prompt_dict()`` (module docstring)."""
    h = hashlib.sha256(b"vllm-prompt-dict/v1")
    _canon(request.prompt_dict(), h)
    return h.hexdigest()
