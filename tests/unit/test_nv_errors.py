"""``nv.errors.classify``: NVIDIA's code is identified by where it was loaded from (review 6 H1,
M2). The fake NVIDIA module ``tests.fakes.fake_nvidia`` is registered as an NVIDIA root, as the
linked snapshot is in production; ``tests/nvlink/test_nv_link_errors.py`` runs the same table on
NVIDIA's real classes and frames."""

from __future__ import annotations

import json
import pickle

import pytest

from flora_deploy.nv import errors as E
from flora_deploy.stages.reference import NvidiaError, transport_error
from tests.fakes import fake_nvidia as F


def raised(fn, *a):
    try:
        fn(*a)
    except BaseException as exc:  # noqa: BLE001
        return exc
    raise AssertionError("did not raise")


def ours_raise(exc):  # a frame of "our" code: this test file (origin "other")
    raise exc


def test_nvidia_classes_by_origin_not_name():
    for cls in (F.FineSegmentationOutputError, F.SubtaskAnnotationAdapterError,
                F.FineWorkerError, F.U2WorkerError, F.InputError, F.InputErrorForModel):
        # raised anywhere (even from our frames) or never raised: the class is NVIDIA's
        assert E.classify(cls("x")) == "nv_failure", cls
        assert E.classify(raised(ours_raise, cls("x"))) == "nv_failure", cls
        assert E.classify(transport_error(raised(F.nvidia_raise, cls("x")))) == "nv_failure"
    # the module NAME is irrelevant: a flora_nv_-named class nobody loaded is not NVIDIA's
    fake = type("FineSegmentationOutputError", (RuntimeError,),
                {"__module__": "flora_nv_segcap_78a0e4ac.batch_utils.models"})
    assert E.classify(raised(ours_raise, fake("x"))) == "python"


def test_builtins_from_nvidia_frames_are_nvidia_verdicts():
    cases = [
        raised(F.nvidia_assert, False, "Real temporal slots must not duplicate evidence"),
        raised(F.nvidia_lookup, {}, "u2_context"),  # KeyError
        raised(F.nvidia_next, []),  # StopIteration
        raised(F.nvidia_compare, None),  # TypeError
        raised(F.nvidia_raise, ValueError("camera roles must be main")),
        raised(F.nvidia_raise, RuntimeError("shape '[2, 3]' is invalid for input of size 5")),
        raised(F.nvidia_lookup, [], 3),  # IndexError (LookupError)
        raised(F.nvidia_raise, json.JSONDecodeError("bad", "", 0)),  # ValueError subclass
    ]
    for exc in cases:
        assert E.classify(exc) == "nv_failure", repr(exc)
        assert E.classify(transport_error(exc)) == "nv_failure", repr(exc)
        # across a process boundary: the origin travels with the picklable error
        assert E.classify(pickle.loads(pickle.dumps(transport_error(exc)))) == "nv_failure"


def test_library_frames_are_skipped():
    """json.loads called by NVIDIA is NVIDIA's verdict; called by us it is ours."""
    assert E.classify(raised(F.nvidia_json, "{")) == "nv_failure"
    exc = raised(lambda: json.loads("{"))  # innermost non-lib frame: this test (ours)
    assert E.classify(exc) == "python"


def test_our_errors_stay_ours():
    from flora_deploy.workers_impl.segcap import instruction_for

    exc = raised(instruction_for, {"instruction": "fixed"}, None, None)  # flora_deploy frame
    assert E.origin(exc) == "ours" and E.classify(exc) == "python"
    assert E.classify(raised(ours_raise, ValueError("ours"))) == "python"
    assert E.classify(raised(ours_raise, KeyError("streams"))) == "python"
    assert E.classify(ValueError("never raised, not NVIDIA's")) == "infra"


def test_gpu_and_environment():
    oom = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})
    acc = type("AcceleratorError", (RuntimeError,), {"__module__": "torch"})
    cases = {
        "oom": [raised(F.nvidia_raise, oom("CUDA out of memory. Tried to allocate 2 GiB")),
                raised(F.nvidia_raise, RuntimeError("CUDA error: out of memory"))],
        "cuda_fatal": [raised(F.nvidia_raise, RuntimeError(
            "CUDA error: an illegal memory access was encountered")),
            raised(ours_raise, RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"))],
        "infra": [raised(F.nvidia_raise, acc("device lost")),
                  raised(F.nvidia_raise, F.NotReady("GPU memory is insufficient")),
                  raised(F.nvidia_raise, FileNotFoundError("/media/x.mp4")),
                  raised(F.nvidia_raise, TimeoutError("read timed out")),
                  raised(F.nvidia_raise, RuntimeError("A model asset failed its integrity check")),
                  raised(F.nvidia_raise, RuntimeError("Production code changed: api.py")),
                  raised(F.nvidia_raise, RuntimeError("CUDA is required for inference."))],
    }
    for want, excs in cases.items():
        for exc in excs:
            assert E.classify(exc) == want, (exc, want)
            assert E.classify(transport_error(exc)) == want, (exc, want)
    assert E.replaces_container("oom") and E.replaces_container("cuda_fatal")
    assert not E.replaces_container("infra") and E.terminal("nv_failure")


def test_context_budget_is_nvidia_semantics():
    from flora_deploy.nv.budget import ContextBudgetError

    e = ContextBudgetError("too long", prompt_tokens=40000, budget=1024, limit=32768)
    assert E.classify(e) == "nv_failure"


def test_transported_without_original():
    """A client-side NvidiaError (no __cause__, no frames): carried origin, else the name."""
    assert E.classify(NvidiaError("x.y.Anything", "m", "nv")) == "nv_failure"
    assert E.classify(NvidiaError("x.y.Anything", "m", "ours")) == "python"
    assert E.classify(NvidiaError("batch_utils.media.InputError", "m")) == "nv_failure"
    assert E.classify(NvidiaError("builtins.FileNotFoundError", "m", "nv")) == "infra"
    assert E.classify(NvidiaError("builtins.RuntimeError",
                                  "The pipeline returned an invalid caption interval.")) \
        == "nv_failure"
    assert E.classify(NvidiaError("builtins.RuntimeError", "who knows")) == "infra"


def test_cuda_markers_match_reward():
    from flora_deploy.workers_impl.reward import CUDA_FATAL_MARKERS

    assert E.CUDA_FATAL_MARKERS == CUDA_FATAL_MARKERS


@pytest.mark.parametrize("kind", E.KINDS)
def test_kinds_are_the_api(kind):
    assert kind in ("nv_failure", "infra", "python", "cuda_fatal", "oom")
