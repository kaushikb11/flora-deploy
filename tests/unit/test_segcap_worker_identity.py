"""Review 4 fixes folded into P1a-5s: ``accel.identity`` fails closed on zero / unparsed calls
(M1), comparisons refuse mixing determinism modes (M2), timing-dependent calls are flagged
(M4), coverage is reported (M6 / L5)."""

from __future__ import annotations

import pytest

from flora_deploy.accel.identity import compare_calls, compare_runs
from flora_deploy.stages import reference as ref


def stage_calls(digests, outs=None):
    outs = outs or [f"o{i}" for i in range(len(digests))]
    return [{"model": "tempseg", "input_digest": d, "input_tokens": 10, "output_limit": 1024,
             "generated_tokens": 3, "complete": True, "output_sha256": o}
            for d, o in zip(digests, outs, strict=True)]


def result(calls, mode="nvidia"):
    return {"segments": [], "flora_reference": {"model_calls": calls,
                                                "determinism": {"mode": mode}}}


def test_persisted_stage_records_are_compared_not_skipped():
    # the review-4 q3 failure: stage records parsed as 0 calls and passed
    a, b = stage_calls(["i0", "i1"]), stage_calls(["i0", "DRIFT"])
    r = compare_calls(a, b)
    assert not r["ok"] and r["violations"][0]["call"] == 1
    same = compare_calls(a, stage_calls(["i0", "i1"]))
    assert same["ok"] and same["conclusive"] and same["paired"] == 2 and same["coverage"] == 1


def test_zero_calls_or_unknown_records_never_pass():
    assert not compare_calls([], [])["ok"]
    assert not compare_runs({}, {})["ok"]
    with pytest.raises(ValueError):
        compare_calls([{"foo": 1}], [{"foo": 1}])
    assert not compare_runs({"e": stage_calls(["i0"])}, {})["ok"]  # missing episode


def test_modes_are_never_mixed():
    a = {"e": result(stage_calls(["i0"]), "nvidia")}
    b = {"e": result(stage_calls(["i0"]), "deterministic")}
    with pytest.raises(ref.ModeMismatch):
        ref.identity_summary(a, b)
    with pytest.raises(ref.ModeMismatch):
        ref.compare_runs(a, b)
    ok = ref.identity_summary(a, {"e": result(stage_calls(["i0"]))})
    assert ok["gate_ok"] and ok["complete_coverage"] and ok["determinism_mode"] == "nvidia"


def test_coverage_after_output_divergence_is_reported():
    a = {"e": result(stage_calls(["i0", "i1", "i2"]))}
    b = {"e": result(stage_calls(["i0", "i1", "i2"], ["X", "o1", "o2"]))}
    s = ref.identity_summary(a, b)
    assert s["gate_ok"] and not s["complete_coverage"] and s["coverage"] == round(1 / 3, 4)


def test_timing_dependent_calls_are_flagged():
    assert ref.timing_dependent({"elapsed_s": 299.0})
    assert ref.timing_dependent({"complete": False, "generated_tokens": 700,
                                 "output_limit": 1024})
    assert not ref.timing_dependent({"complete": False, "generated_tokens": 1024,
                                     "output_limit": 1024, "elapsed_s": 60})
    events = [{"event": "generation_start", "input_tensor_digest": "d", "output_limit": 1024},
              {"event": "generation_end", "generated_tokens": 5, "complete": True,
               "elapsed_s": 200.0}]
    calls = ref.model_calls(events)
    assert calls[0]["timing_dependent"] and calls[0]["elapsed_s"] == 200.0
    s = ref.identity_summary({"e": result(calls)}, {"e": result(calls)})
    assert s["timing_dependent_calls"] == 2
