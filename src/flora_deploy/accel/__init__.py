"""Accelerated-NVIDIA runtime patch layer (PLAN.md D17 form 1, OPT-1; ``docs/accel.md``).

Opt-in, individually toggleable, reversible swaps of attributes of the linked NVIDIA modules in
this process, each proven output-identical to NVIDIA's original by a parity test::

    from flora_deploy.accel import apply_patches
    active = apply_patches(nv, ["decode_once", "transport_pool", "png_inmem"])
    with active.episode():           # per-thread episode scope: caches / open decoders
        segments, audit = nv.pipeline.segment(videos, pool, session, instruction, progress)
        ...
    record["accel"] = active.provenance()
    active.restore()

NVIDIA's files are never modified; its SHA manifests keep verifying.

``accel.captioner_load`` (OPT-2) is separate: ``captioner_load.install(nv, background=True)``
before loading TempSeg swaps in the bit-identical BF16 checkpoint and hashes it in parallel with
that load (``docs/captioner-bf16.md``). ``accel.identity`` compares two runs call by call from
NVIDIA's ``input_tensor_digest`` log lines.
"""

from __future__ import annotations

from flora_deploy.accel.registry import (
    PATCH_SCHEMA,
    ActivePatches,
    Patch,
    PatchError,
    apply_patches,
    register,
    registered,
)
from flora_deploy.accel.state import AccelState

__all__ = [
    "PATCH_SCHEMA",
    "AccelState",
    "ActivePatches",
    "Patch",
    "PatchError",
    "apply_patches",
    "register",
    "registered",
    "ALL_PATCHES",
]


def __getattr__(name: str):
    if name == "ALL_PATCHES":
        from flora_deploy.accel.patches import ALL

        return ALL
    raise AttributeError(name)
