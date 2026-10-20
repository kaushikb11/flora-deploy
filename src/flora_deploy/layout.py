"""Storage names and path conventions shared by every component.

Volumes hold weights, caches and the golden set only. Job state of record lives in the user's
bucket (PLAN.md D6); its layout is defined later in ``jobs/store.py``.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from flora_deploy.config.pins import RepoPin

WEIGHTS_VOLUME = "flora-weights"
CACHE_VOLUME = "flora-cache"  # VLLM_CACHE_ROOT (torch.compile / Triton caches)
GOLDEN_VOLUME = "flora-golden"

# Mount points inside Modal containers.
WEIGHTS_MOUNT = PurePosixPath("/weights")
CACHE_MOUNT = PurePosixPath("/cache")
GOLDEN_MOUNT = PurePosixPath("/golden")

# Local cache for NVIDIA code pins used by `-m nvlink` tests (never committed).
NV_CODE_CACHE_ENV = "FLORA_NV_CODE_CACHE"
NV_CODE_CACHE_DEFAULT = "~/.cache/flora-deploy/nv"


def snapshot_dirname(pin: RepoPin) -> str:
    """``nvidia--Flora-TempSeg@<full sha>``: unique per repo+revision, filesystem safe."""
    return f"{pin.repo.replace('/', '--')}@{pin.revision}"


def snapshot_path(pin: RepoPin, root: PurePosixPath = WEIGHTS_MOUNT) -> PurePosixPath:
    return root / "snapshots" / snapshot_dirname(pin)


def derived_path(name: str, source: RepoPin, root: PurePosixPath = WEIGHTS_MOUNT) -> PurePosixPath:
    """Derived weights, e.g. ``derived_path("captioner-bf16", PINS_BY_KEY["captioner"])``."""
    return root / "derived" / f"{name}@{source.revision}"


RECEIPT_NAME = "prep_receipt.json"
