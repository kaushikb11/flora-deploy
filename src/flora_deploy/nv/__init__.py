"""Linked-mode boundary: the ONLY package allowed to import NVIDIA's code (PLAN.md D1).

Contract (implemented by WS-B in ``linker.py``, ``session.py``, ``capture.py``):

- ``Family``: NVIDIA code families. TempSeg and Captioner ship byte-identical ``batch_utils``, so
  they form one family (SEGCAP); the Rewarder's different ``batch_utils`` is REWARDER. **The two
  families must never be loaded in the same process**, because both are named ``batch_utils`` and
  bootstrap mutates ``sys.path``.

- ``load_family(family, code_dir) -> NvFamily``: import the family's ``batch_utils`` from
  ``code_dir`` (a snapshot of the family's *code* pin) under a unique module name, keeping
  NVIDIA's SHA-manifest bootstrap intact. It returns a handle exposing the NVIDIA modules (e.g.
  ``.pipeline``, ``.adapter``, ``.models``, ``.preprocessing``, or ``.core`` for the Rewarder).
  It raises ``FamilyConflictError`` if another family is already loaded in this process.

- ``CachedSession``: duck-types NVIDIA ``models.Session`` for ``pipeline.segment()`` and
  ``caption_*()``. ``load(kind)`` returns an already-loaded ``(model, proc)``, built once per kind
  with NVIDIA's own ``Session.load``. ``close()`` is a no-op per call, with a real
  ``close_all()`` at exit.

- ``CaptureProcessor(real_proc)``: proxy handed to NVIDIA's input builders. It delegates attribute
  access, and records the final
  ``__call__(text=..., videos=..., video_metadata=..., do_sample_frames=False, ...)`` as a
  ``CapturedRequest`` (prompt text, uint8 frame arrays, metadata dicts). The call returns a
  ``_CapturedInputs`` whose only allowed operation is ``.pop('video_metadata', default)``.
"""

from __future__ import annotations

from enum import StrEnum


class Family(StrEnum):
    SEGCAP = "segcap"  # nvidia/Flora-TempSeg + nvidia/Flora-Captioner batch_utils
    REWARDER = "rewarder"  # nvidia/Flora-Rewarder batch_utils


# Which pins.PINS key provides each family's code snapshot.
FAMILY_CODE_PIN = {Family.SEGCAP: "tempseg_code", Family.REWARDER: "rewarder"}


class FamilyConflictError(RuntimeError):
    """Raised when a second NVIDIA code family is loaded into a process that has one."""


_EXPORTS = {
    "load_family": "linker",
    "code_dir_for": "linker",
    "NvFamily": "linker",
    "CachedSession": "session",
    "CaptureProcessor": "capture",
    "CapturedRequest": "capture",
}


def __getattr__(name: str):
    # Lazy re-exports: keeps ``import flora_deploy.nv`` free of numpy and of import cycles.
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
