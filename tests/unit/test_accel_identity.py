"""Call-by-call input identity of two runs (``accel/identity.py``)."""

from __future__ import annotations

import json

import pytest

from flora_deploy.accel.identity import (
    call_records,
    compare_calls,
    compare_runs,
    model_calls,
)


def log(calls, *, noise=True):
    """FLORA_RUNTIME / FLORA_OBSERVE lines as NVIDIA's generate + the W3-A lock print them;
    ``calls`` = [(input digest, output text sha or None)]."""
    lines = []
    for digest, out in calls:
        if noise:
            lines.append("some other stdout line")
        lines.append("FLORA_RUNTIME " + json.dumps(dict(
            event="generation_start", input_tokens=10, output_limit=512,
            input_tensor_digest=digest, input_tensors={})))
        if out is not None:
            lines.append("FLORA_RUNTIME " + json.dumps(dict(
                event="generation_end", generated_tokens=3, complete=True)))
            lines.append("FLORA_OBSERVE " + json.dumps(dict(kind="generate_output", sha256=out)))
    return lines


def test_parses_nvidia_and_lock_events():
    calls = model_calls(log([("i0", "o0"), ("i1", None), ("i2", "o2")]))
    assert [(c.index, c.input_tensor_digest, c.output_sha256) for c in calls] == [
        (0, "i0", "o0"), (1, "i1", None), (2, "i2", "o2")]
    assert calls[0].generated_tokens == 3 and calls[0].output_limit == 512
    with pytest.raises(ValueError, match="input_tensor_digest"):
        model_calls([{"event": "generation_start"}])


def test_identical_runs_are_conclusive():
    run = [("i0", "o0"), ("i1", "o1"), ("i2", "o2")]
    r = compare_calls(log(run), log(run, noise=False))
    assert r["ok"] and r["conclusive"] and r["paired"] == 3 and r["unpaired"] == 0


def test_output_divergence_stops_pairing_but_is_not_a_violation():
    a = log([("i0", "o0"), ("i1", "o1"), ("i2", "o2"), ("i3", "o3")])
    b = log([("i0", "o0"), ("i1", "DIFFERENT"), ("j2", "p2"), ("i3", "o3")])
    r = compare_calls(a, b)
    assert r["ok"] and not r["conclusive"]
    assert r["first_output_divergence"] == 1 and r["paired"] == 2 and r["unpaired"] == 4
    assert r["unpaired_inputs_seen_in_other_run"] == 2  # i3 occurs in both (informational)


def test_input_difference_with_identical_upstream_is_a_violation():
    a = log([("i0", "o0"), ("i1", "o1"), ("i2", "o2")])
    b = log([("i0", "o0"), ("i1", "o1"), ("X2", "o2")])
    r = compare_calls(a, b)
    assert not r["ok"] and r["violations"][0]["call"] == 2
    # the very first call has no upstream: any difference is a violation
    assert not compare_calls(log([("i0", "o0")]), log([("X", "o0")]))["ok"]


def test_call_count_difference_with_identical_outputs_is_a_violation():
    r = compare_calls(log([("i0", "o0"), ("i1", "o1")]), log([("i0", "o0")]))
    assert not r["ok"] and "call count" in r["violations"][0]["reason"]


def test_unlogged_output_is_inconclusive_not_ok_by_default():
    r = compare_calls(log([("i0", None), ("i1", "o1")]), log([("i0", None), ("i1", "o1")]))
    assert r["ok"] and not r["conclusive"] and r["paired"] == 1


def test_compare_runs_per_episode_and_persisted_records():
    a = {"ep1": log([("i0", "o0")]), "ep2": call_records(log([("k0", "q0"), ("k1", "q1")]))}
    b = {"ep1": log([("i0", "o0")]), "ep2": log([("k0", "q0"), ("k1", "q1")])}
    r = compare_runs(a, b)
    assert r["ok"] and r["conclusive"] and r["paired"] == 3
    r = compare_runs(a, {"ep1": b["ep1"]})
    assert not r["ok"] and not r["episodes"]["ep2"]["ok"]
    r = compare_runs(b, {**b, "ep1": log([("i0-other", "o0")])})
    assert not r["ok"] and r["violations"] == 1
