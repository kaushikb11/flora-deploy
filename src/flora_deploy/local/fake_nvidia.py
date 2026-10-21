"""The fake models' stand-in for NVIDIA's code (local runs, ``--fake-models``).

Registered with ``nv.errors.register_nv_root``, so what this module defines is classified as
NVIDIA's verdict exactly like a class from the linked snapshot (``nv.errors`` identifies NVIDIA's
code by where it was loaded from)."""

from __future__ import annotations

from flora_deploy.nv import errors as _errors

_errors.register_nv_root(__file__)


class FakeNvidiaInputError(ValueError):
    """The ``nverr`` fault: NVIDIA's ``ValueError`` semantics for a bad episode."""
