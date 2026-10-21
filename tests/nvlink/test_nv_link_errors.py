"""Review 6 H1 / M2 on NVIDIA's REAL linked SEGCAP family (CPU, no models): NVIDIA's own
exception classes, and builtin exceptions raised by NVIDIA's own frames, are ``nv_failure``,
through the production path (raised inside NVIDIA's code -> ``stages.reference.
transport_error`` -> ``workers_impl.segcap.classify_nvidia_error`` / ``nv.errors.classify``).
NVIDIA's environment failures stay ``infra``; our own errors stay ``python``."""

from __future__ import annotations

import pickle

import pytest
from test_nv_link_capture import nv, nv_dirs  # noqa: F401  (module-scoped fixtures)

from flora_deploy.nv import errors as E
from flora_deploy.stages.reference import transport_error
from flora_deploy.workers_impl.segcap import classify_nvidia_error

pytestmark = pytest.mark.nvlink


def raised(fn, *a, **k):
    try:
        fn(*a, **k)
    except BaseException as exc:  # noqa: BLE001
        return exc
    raise AssertionError(f"{fn} did not raise")


def raise_(exc):
    raise exc


@pytest.fixture(scope="module")
def mods(nv):  # noqa: F811
    nv.pipeline  # noqa: B018 - bootstrap -> adapter -> flora_runtime on sys.path
    v = nv.vendor
    return {
        "models": nv.models, "media": nv.media,
        "fine_subtask": v("flora_runtime.video_segmentation.fine_subtask"),
        "fine_temporal": v("flora_runtime.video_segmentation.fine_temporal"),
        "fine_worker": v("flora_runtime.video_segmentation.fine_worker"),
        "u2_worker": v("flora_runtime.subtask_captioning.u2_worker"),
        "worker_transport": v("flora_runtime.worker_transport"),
        "scope_scheduler": v("flora_runtime.video_segmentation.scope_scheduler"),
        "u2_contract": v("u2_contract"),
        "adapter": v("adapter"),
    }


def kinds(exc):
    t = transport_error(exc)
    return (classify_nvidia_error(t)[0], E.classify(t), E.classify(exc),
            E.classify(pickle.loads(pickle.dumps(t))))


def test_real_nvidia_classes_are_nv_failure(mods):
    m = mods
    short = {"gpu_calls": 3, "max_call_s": 12.0, "last_call_s": 12.0, "timing_dependent": False}
    cases = [
        m["fine_subtask"].FineSegmentationOutputError(
            "fine segmentation could not be repaired by bounded fine retry: x", diagnostics={}),
        m["worker_transport"].SubtaskAnnotationAdapterError("response identity mismatch"),
        m["fine_worker"].FineWorkerError("nearest decoded frame is farther than half a slot"),
        m["u2_worker"].U2WorkerError("U2 frame budget 999 exceeds 512"),
        m["fine_temporal"].FineTemporalError("x"),
        m["scope_scheduler"].ScopeSchedulerError("x"),
        m["u2_contract"].U2ContractError("x"),
        m["media"].InputError("Use a file containing one video stream."),
        m["models"].InputErrorForModel("The selected video exceeds the native model context "
                                       "budget."),
    ]
    for exc in cases:
        for how in (exc, raised(raise_, exc)):  # never raised, or raised from our frame
            assert kinds(how) == ("nv_failure",) * 4, (type(exc).__qualname__, kinds(how))
        assert classify_nvidia_error(transport_error(exc), short)[0] == "nv_failure"
    # NVIDIA's NotReady is the environment (GPU too small / model not loaded)
    assert set(kinds(m["models"].NotReady("GPU memory is insufficient"))) == {"infra"}


def test_builtins_raised_by_nvidia_frames(mods):
    m = mods
    adapter, u2c = m["adapter"], m["u2_contract"]
    cases = {
        "AssertionError": raised(adapter.temporal_frame_plan, None, []),
        "KeyError": raised(adapter.window_from_payload, {}, None),
        "StopIteration": raised(adapter.caption_sample,
                                {"source_fps_num": 1, "source_fps_den": 1, "blocks": []}, {}, []),
        "TypeError": raised(u2c.uniform_bin_centers, 0, 5, None),
        "AssertionError(pts)": raised(adapter.temporal_frame_plan, None, [5]),
    }
    for name, exc in cases.items():
        assert type(exc).__module__ == "builtins", name
        assert E.origin(exc) == "nv", name
        assert kinds(exc) == ("nv_failure",) * 4, (name, kinds(exc))


def test_nvidia_environment_failures_stay_infra(mods):
    for msg in ("Production code changed: api.py", "CUDA is required for inference.",
                "A model asset failed its integrity check"):
        assert set(kinds(raised(raise_, RuntimeError(msg)))) == {"infra"}


def test_our_builtins_are_not_nvidia(mods):
    from flora_deploy.workers_impl.segcap import instruction_for

    exc = raised(instruction_for, {"instruction": "fixed"}, None, None)
    assert E.classify(exc) == "python"
    assert E.classify(transport_error(raised(raise_, KeyError("streams")))) == "python"
