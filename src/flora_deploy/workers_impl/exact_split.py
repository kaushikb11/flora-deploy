"""G3 lever 3 (``docs/exact-cost.md``): NVIDIA's ``exact`` pipeline split into a TempSeg step
and a Captioner step that can run in different processes (different GPU pools), with the
TempSeg step's result persisted between them. **Analysis/proof only: not wired into any shard
kind, manifest or app** (see the doc for why it is not recommended yet).

No NVIDIA orchestration is re-expressed. Both steps run NVIDIA's own entry point
``api.DataEnginePipeline().annotate`` (``stages.reference.annotate_nvidia``) unchanged, with a
runtime swap of three ``pipeline`` attributes (:func:`install_split`) that is active only in a
thread/context that asked for it (``ContextVar``; every other call passes straight through):

- **segment step** (:func:`segment_episode`): NVIDIA's ``pipeline.segment`` runs unchanged and
  its return value ``(segments, temporal_audit)`` is captured; when NVIDIA's ``annotate`` then
  calls ``caption_single`` / ``caption_three``, the swap records the call and stops the episode
  (no Captioner call is made). Everything before that (``_inputs``, NVIDIA's checks, its
  ``Session()``, ``runtime_event('segmentation_complete')``) is NVIDIA's code.
- **caption step** (:func:`caption_episode`): NVIDIA's ``annotate`` runs again from the start
  (its ``_inputs`` re-probes the same files: NVIDIA's ``probe`` is deterministic, and the
  persisted ``input_sha256`` must equal the new probe's), and NVIDIA's ``pipeline.segment`` is
  replaced by a function returning the persisted ``(segments, temporal_audit)``. NVIDIA's own
  code then builds every Captioner input, calls the Captioner and assembles the result.

The persisted document (:func:`encode` / :func:`decode`) is a tagged JSON encoding that keeps
NVIDIA's value types exactly (tuples stay tuples, ``fractions.Fraction`` stays a Fraction,
floats round-trip by ``repr``) and refuses any other type, so the caption step gets objects
equal to the ones ``pipeline.segment`` returned in-process. ``tests/nvlink/
test_exact_scheduling_nvlink.py`` proves, on NVIDIA's real code with fake models, that the
Captioner's NVIDIA-logged ``input_tensor_digest`` per call and the full NVIDIA result are
identical to the single-process path.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from fractions import Fraction
from typing import Any

SPLIT_SCHEMA = "flora_exact_split_v1"
_STATE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("flora_exact_split",
                                                                     default=None)
_MARK = "__flora_exact_split__"


class _CaptionReached(Exception):  # noqa: N818 - control flow, never escapes this module
    """Raised by the swapped ``caption_*`` in the segment step: TempSeg is done."""


class SplitError(RuntimeError):
    """The persisted segment result does not belong to this episode (or is malformed)."""


# --- faithful encoding ------------------------------------------------------------------------


def _enc(x: Any) -> Any:
    if x is None or isinstance(x, bool | int | str):
        return x
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise SplitError("non-finite float in NVIDIA's segment result")
        return x
    if isinstance(x, Fraction):
        return {"__fraction__": [x.numerator, x.denominator]}
    if isinstance(x, tuple):
        return {"__tuple__": [_enc(v) for v in x]}
    if isinstance(x, list):
        return [_enc(v) for v in x]
    if isinstance(x, dict):
        if not all(isinstance(k, str) for k in x):
            raise SplitError("non-string dict key in NVIDIA's segment result")
        if any(k.startswith("__") and k.endswith("__") for k in x):
            raise SplitError("reserved key in NVIDIA's segment result")
        return {k: _enc(v) for k, v in x.items()}
    raise SplitError(f"cannot persist a {type(x).__name__} from NVIDIA's segment result")


def _dec(x: Any) -> Any:
    if isinstance(x, list):
        return [_dec(v) for v in x]
    if isinstance(x, dict):
        if set(x) == {"__tuple__"}:
            return tuple(_dec(v) for v in x["__tuple__"])
        if set(x) == {"__fraction__"}:
            n, d = x["__fraction__"]
            return Fraction(n, d)
        return {k: _dec(v) for k, v in x.items()}
    return x


def encode(doc: dict) -> bytes:
    body = json.dumps(_enc(doc), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return body.encode()


def decode(data: bytes) -> dict:
    return _dec(json.loads(data))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the swap ---------------------------------------------------------------------------------


def install_split(nv, patcher) -> None:
    """Swap ``pipeline.segment`` / ``caption_single`` / ``caption_three`` for dispatchers that
    pass through unless :func:`segment_episode` / :func:`caption_episode` is running in this
    context. ``patcher.restore()`` gives NVIDIA's attributes back."""
    if getattr(nv.pipeline.segment, _MARK, False):
        raise RuntimeError("exact_split is already installed")

    def seg_factory(orig):
        def segment(videos, pool, session, instruction, progress):
            st = _STATE.get()
            if st is None:
                return orig(videos, pool, session, instruction, progress)
            if st["mode"] == "segment":
                from flora_deploy.stages.reference import capture_identity_events

                # innermost route: NVIDIA's TempSeg events (incl. its to_thread predictor)
                with capture_identity_events() as (events, _lines):
                    out = orig(videos, pool, session, instruction, progress)
                st["events"] = events
                st["segment"] = out
                st["input_sha256"] = {r: v.sha256 for r, v in videos.items()}
                return out
            want = st["doc"]["input_sha256"]
            got = {r: v.sha256 for r, v in videos.items()}
            if got != want:
                raise SplitError(f"persisted segments are for inputs {want}, not {got}")
            st["substituted"] = True
            return st["doc"]["segments"], st["doc"]["temporal_audit"]

        setattr(segment, _MARK, True)
        return segment

    def cap_factory(orig):
        def caption(*args, **kwargs):
            st = _STATE.get()
            if st is not None and st["mode"] == "segment":
                st["caption_called"] = orig.__name__
                raise _CaptionReached()
            return orig(*args, **kwargs)

        setattr(caption, _MARK, True)
        return caption

    patcher.patch(nv.pipeline, "segment", seg_factory)
    patcher.patch(nv.pipeline, "caption_single", cap_factory)
    patcher.patch(nv.pipeline, "caption_three", cap_factory)


def _run(nv, session, ep: dict, state: dict):
    from flora_deploy.stages.reference import annotate_nvidia

    token = _STATE.set(state)
    try:
        return annotate_nvidia(nv, session, ep["videos"], ep["profile"],
                               ep.get("long_windows", False), ep.get("instruction", ""))
    finally:
        _STATE.reset(token)


def segment_episode(nv, session, ep: dict) -> dict:
    """The TempSeg half of one episode (``ep`` as ``stages.reference.annotate_batch`` items).
    Returns the document to persist: NVIDIA's ``(segments, temporal_audit)`` exactly, the probe
    digests it belongs to, and the TempSeg ``model_calls`` (NVIDIA-logged input digests)."""
    from flora_deploy.stages.reference import model_calls

    state: dict[str, Any] = {"mode": "segment"}
    try:
        _run(nv, session, ep, state)
    except _CaptionReached:
        pass
    else:  # NVIDIA returned without a caption call: no persisted step to split
        raise SplitError("NVIDIA's annotate finished without reaching the caption step")
    if "segment" not in state:
        raise SplitError("NVIDIA's pipeline.segment did not run")
    segments, temporal_audit = state["segment"]
    return {"schema": SPLIT_SCHEMA, "episode": ep["episode"], "segments": segments,
            "temporal_audit": temporal_audit, "input_sha256": state["input_sha256"],
            "caption_entry": state.get("caption_called"),
            "model_calls": model_calls(state.get("events", []))}


def caption_episode(nv, session, ep: dict, doc: dict) -> dict:
    """The Captioner half: NVIDIA's whole ``annotate`` with ``pipeline.segment`` answered from
    the persisted ``doc``. Returns NVIDIA's result (plus ``flora_reference``, whose
    ``model_calls`` are the Captioner calls only)."""
    if doc.get("schema") != SPLIT_SCHEMA or doc.get("episode") != ep["episode"]:
        raise SplitError(f"segment document is not {SPLIT_SCHEMA} for {ep['episode']!r}")
    state: dict[str, Any] = {"mode": "caption", "doc": doc}
    result = _run(nv, session, ep, state)
    if not state.get("substituted"):
        raise SplitError("NVIDIA's pipeline.segment was not answered from the document")
    return result
