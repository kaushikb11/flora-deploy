"""Errors raised in a child process for the cross-process classification test (review 8a R6):
each case raises the way production does (NVIDIA's frames are ``tests.fakes.fake_nvidia``, a
registered NVIDIA root; our copies of NVIDIA's rejections are the real ``accel`` / ``stages``
code), then crosses the boundary as ``stages.reference.transport_error`` pickles it."""

from __future__ import annotations

import pickle


def _raised(fn, *a):
    try:
        fn(*a)
    except BaseException as exc:  # noqa: BLE001
        return exc
    raise AssertionError(f"{fn} did not raise")


def _ours(exc):  # a flora-side (test) frame
    raise exc


def _geometry():
    from flora_deploy.accel.geometry import make_normalized

    class _RCG:
        PACKED_IMAGE_LIMIT_BYTES = 0
        SCALES: list = []

    class _NV:
        def vendor(self, name):
            return _RCG

    return _raised(make_normalized(_NV(), None, None), {"blocks": []}, {})


def _missing_frame():
    from flora_deploy.accel.geometry import make_normalized

    class _RCG:
        PACKED_IMAGE_LIMIT_BYTES = 0
        SCALES = [1.0]

    class _NV:
        def vendor(self, name):
            return _RCG

    return _raised(make_normalized(_NV(), None, None),
                   {"blocks": [{"role": "main", "indices": [0, 1]}]}, {})


def cases() -> dict:
    from tests.fakes import fake_nvidia as F

    return {
        "nv_class": lambda: _raised(F.nvidia_raise, F.FineSegmentationOutputError("x")),
        "nv_assert": lambda: _raised(F.nvidia_assert, False, "slots"),
        "nv_memory": lambda: _raised(F.nvidia_raise, MemoryError()),
        "nv_attribute": lambda: _raised(F.nvidia_raise, AttributeError("x")),
        "nv_name": lambda: _raised(F.nvidia_raise, NameError("x")),
        "geometry": _geometry,
        "missing_frame": _missing_frame,
        "ours_value": lambda: _raised(_ours, ValueError("bad config")),
        "ours_runtime": lambda: _raised(_ours, RuntimeError("bug")),
        "oserror": lambda: _raised(_ours, FileNotFoundError("gone")),
    }


def produce(_=None) -> dict:
    """In the child: ``{case: (in-process classify, pickled transport_error)}``."""
    from flora_deploy.nv import errors as E
    from flora_deploy.stages.reference import transport_error

    out = {}
    for name, make in cases().items():
        exc = make()
        out[name] = (E.classify(exc), pickle.dumps(transport_error(exc)))
    return out
