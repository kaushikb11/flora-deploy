"""Call-by-call input identity of two runs, from NVIDIA's own runtime log (PLAN.md D17, W3-A).

The ``exact`` promise is that NVIDIA's model receives identical inputs. NVIDIA's
``models.generate`` prints, before each model call, a ``FLORA_RUNTIME`` ``generation_start``
event carrying ``input_tensor_digest`` (sha256 over every input tensor's dtype, shape and bytes,
``batch_utils/models.py:171-177``). Whole-result digests cannot prove that promise, because
later TempSeg windows (adaptive retry) and every Captioner input depend on earlier model
*outputs*, which vary run to run on the GPU even sequentially (``docs/exact-concurrency.md``).

So two runs of an episode are compared call by call, **conditioned on upstream outputs**:

- call ``k`` of run A is paired with call ``k`` of run B while every earlier call of both runs
  produced the same output (``generate_output`` sha256, printed by the W3-A GPU lock as a
  ``FLORA_OBSERVE`` event right after NVIDIA's ``generation_end``);
- a paired call whose ``input_tensor_digest`` differs is a **violation** (the inputs diverged
  although everything upstream agreed): that is what breaks ``exact``;
- so is a differing number of calls while all outputs agreed;
- after the first output divergence (or an output that was not logged) later calls cannot be
  paired; they are counted as ``unpaired`` (and, for information, how many of their inputs
  still occur somewhere in the other run).

Inputs are the per-episode event streams: ``FLORA_RUNTIME`` / ``FLORA_OBSERVE`` lines, or the
already-parsed dicts (``stages.reference.capture_runtime_events`` yields the ``FLORA_RUNTIME``
dicts of one episode; the ``generate_output`` events must be collected from the same stream).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

RUNTIME_PREFIX = "FLORA_RUNTIME "
OBSERVE_PREFIX = "FLORA_OBSERVE "
IDENTITY_SCHEMA = "flora_call_identity_v1"


@dataclass(frozen=True)
class ModelCall:
    """One NVIDIA ``generate`` call as logged."""

    index: int
    input_tensor_digest: str
    output_sha256: str | None = None  # None: not logged (call raised, or no recorder)
    input_tokens: int | None = None
    output_limit: int | None = None
    generated_tokens: int | None = None
    complete: bool | None = None


def _event(item: str | Mapping) -> dict | None:
    if isinstance(item, Mapping):
        return dict(item)
    line = str(item).rstrip("\n")
    for prefix in (RUNTIME_PREFIX, OBSERVE_PREFIX):
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):])
            except ValueError:
                return None
            return value if isinstance(value, dict) else None
    return None


def model_calls(events: Iterable[str | Mapping]) -> list[ModelCall]:
    """The ``generate`` calls of one episode's event stream, in order.

    ``generation_start`` opens a call; the next ``generation_end`` and ``generate_output``
    (before the following ``generation_start``) complete it. A ``generation_start`` without
    ``input_tensor_digest`` is an error (the log is not NVIDIA's)."""
    calls: list[dict] = []
    for item in events:
        ev = _event(item)
        if ev is None:
            continue
        kind = ev.get("event") or ev.get("kind")
        if kind == "generation_start":
            digest = ev.get("input_tensor_digest")
            if not isinstance(digest, str) or not digest:
                raise ValueError(f"generation_start without input_tensor_digest: {ev}")
            calls.append(dict(index=len(calls), input_tensor_digest=digest,
                              input_tokens=ev.get("input_tokens"),
                              output_limit=ev.get("output_limit")))
        elif kind == "generation_end" and calls:
            calls[-1].setdefault("generated_tokens", ev.get("generated_tokens"))
            calls[-1].setdefault("complete", ev.get("complete"))
        elif kind == "generate_output" and calls and "output_sha256" not in calls[-1]:
            calls[-1]["output_sha256"] = ev.get("sha256")
    return [ModelCall(**c) for c in calls]


def _calls(run: Iterable) -> list[ModelCall]:
    run = list(run)
    if all(isinstance(c, ModelCall) for c in run):
        return run
    if run and all(isinstance(c, Mapping) and "input_tensor_digest" in c and "index" in c
                   and "event" not in c and "kind" not in c for c in run):
        return [ModelCall(**c) for c in run]  # persisted ``call_records``
    return model_calls(run)


def compare_calls(a: Iterable, b: Iterable) -> dict:
    """Compare two runs of one episode (event streams or ``ModelCall`` lists), see module doc.

    Returns ``ok`` (no violation), ``conclusive`` (every call of both runs was paired),
    ``paired`` / ``unpaired`` counts, ``violations`` and ``first_output_divergence``."""
    ca, cb = _calls(a), _calls(b)
    violations: list[dict] = []
    paired = 0
    divergence: int | None = None
    upstream_agrees = True
    k = 0
    while upstream_agrees and (k < len(ca) or k < len(cb)):
        if k >= len(ca) or k >= len(cb):
            violations.append(dict(call=k, reason="call count differs although every earlier "
                                   "output agreed", calls_a=len(ca), calls_b=len(cb)))
            upstream_agrees = False
            break
        x, y = ca[k], cb[k]
        paired += 1
        if x.input_tensor_digest != y.input_tensor_digest:
            violations.append(dict(call=k, reason="input_tensor_digest differs although every "
                                   "earlier output agreed", a=x.input_tensor_digest,
                                   b=y.input_tensor_digest))
            upstream_agrees = False
        elif x.output_sha256 is None or x.output_sha256 != y.output_sha256:
            divergence = k
            upstream_agrees = False
        k += 1
    rest_a, rest_b = ca[k:], cb[k:]
    seen_b = {c.input_tensor_digest for c in cb}
    seen_a = {c.input_tensor_digest for c in ca}
    return dict(
        schema=IDENTITY_SCHEMA,
        ok=not violations,
        conclusive=not violations and not rest_a and not rest_b,
        calls_a=len(ca),
        calls_b=len(cb),
        paired=paired,
        unpaired=len(rest_a) + len(rest_b),
        unpaired_inputs_seen_in_other_run=sum(c.input_tensor_digest in seen_b for c in rest_a)
        + sum(c.input_tensor_digest in seen_a for c in rest_b),
        first_output_divergence=divergence,
        violations=violations,
    )


def compare_runs(a: Mapping[str, Iterable], b: Mapping[str, Iterable]) -> dict:
    """``compare_calls`` per episode id; an episode present in only one run is a violation."""
    episodes = {}
    for key in sorted(set(a) | set(b), key=str):
        if key not in a or key not in b:
            episodes[key] = dict(ok=False, conclusive=False, violations=[dict(
                reason=f"episode only in run {'a' if key in a else 'b'}")])
            continue
        episodes[key] = compare_calls(a[key], b[key])
    return dict(
        schema=IDENTITY_SCHEMA,
        ok=all(e["ok"] for e in episodes.values()),
        conclusive=all(e["conclusive"] for e in episodes.values()),
        episodes=episodes,
        violations=sum(len(e["violations"]) for e in episodes.values()),
        paired=sum(e.get("paired", 0) for e in episodes.values()),
    )


def call_records(events: Iterable[str | Mapping]) -> list[dict]:
    """``model_calls`` as plain dicts (to persist per episode, e.g. in ``flora_reference``)."""
    return [asdict(c) for c in model_calls(events)]
