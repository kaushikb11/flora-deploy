"""A stand-in for NVIDIA's code in unit tests (``nv.errors.register_nv_root``).

Classes defined here, and frames executing here, count as NVIDIA's for ``nv.errors.classify``,
exactly as files under the linked snapshot do in production. Names and bases mirror the real
NVIDIA classes (``tests/nvlink/test_nv_link_errors.py`` checks the real ones)."""

from __future__ import annotations

import json

try:
    from flora_deploy.nv import errors as _errors
except ImportError:  # before the shared classifier existed: nothing to register with
    _errors = None
else:
    _errors.register_nv_root(__file__)


class InputError(ValueError):  # batch_utils/media.py
    pass


class InputErrorForModel(RuntimeError):  # batch_utils/models.py
    pass


class NotReady(RuntimeError):  # batch_utils/models.py
    pass


class SubtaskAnnotationAdapterError(RuntimeError):  # flora_runtime/worker_transport.py
    pass


class FineSegmentationOutputError(SubtaskAnnotationAdapterError):  # fine_subtask.py
    pass


class FineWorkerError(RuntimeError):  # flora_runtime/video_segmentation/fine_worker.py
    pass


class U2WorkerError(RuntimeError):  # flora_runtime/subtask_captioning/u2_worker.py
    pass


def nvidia_raise(exc: BaseException):
    """Raise ``exc`` from a frame of 'NVIDIA's code'."""
    raise exc


def nvidia_assert(ok: bool, message: str = ""):
    assert ok, message


def nvidia_lookup(d: dict, key: str):
    return d[key]


def nvidia_next(items):
    return next(x for x in items if x)


def nvidia_compare(count):
    return count <= 0  # TypeError for None, raised in this frame


def nvidia_json(text: str):
    return json.loads(text)  # a library frame raises; NVIDIA's frame called it
