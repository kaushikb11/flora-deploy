"""Pinned Modal images (PLAN.md §7). All x86_64.

Each builder returns a lazy ``modal.Image``: nothing is built or uploaded until an app using it
runs. Access ``images.img_cpu`` / ``img_hfref`` / ``img_vllm`` or call the builders directly.

Reproducibility (review M3, PLAN.md D11). Each image installs from a committed, fully transitive
lock (``locks/<image>.txt``: exact versions and sha256 hashes for linux x86_64 / CPython 3.12),
with ``--require-hashes --no-build``, using a pinned uv binary. The ``*_PACKAGES`` tuples below
are only the lock *inputs*; ``scripts/lock_images.py`` turns them into ``locks/<image>.in`` and
the lock. Bases are pinned too: the Modal image builder version (which fixes ``debian_slim``'s
Debian release and Python micro version, and ``add_python``'s build) and the CUDA base by digest.
``image_fingerprint()`` hashes all of it for the config hash. Gate C showed why one pinned build
matters: NVIDIA picks caption frame scales by compressed PNG size, which depends on the zlib in
the Pillow wheel, so golden data and production must share the exact image stack.

Base choices:
- ``img_cpu``: ``debian_slim``, no apt packages. PyAV wheels bundle FFmpeg; Modal's debian_slim apt
  index can be stale (404s on bullseye security updates), so no apt until a step needs the CLI.
- ``img_hfref``: ``debian_slim`` plus NVIDIA's pip stack verbatim. The torch 2.11.0 PyPI wheel
  ships its CUDA 13.0 user space as ``nvidia-*`` wheels, and Modal's host driver (CUDA 13.3)
  supports it on H200 (SM90). NVIDIA's reference runs HF transformers with SDPA and compiles
  nothing, so it needs no toolkit. This is also exactly what ``pip install -r requirements.txt``
  gives on a plain host.
- ``img_vllm``: ``nvidia/cuda`` *devel* 13.0 (toolkit at or below the host's 13.3, matching torch's
  cu130). vLLM 0.24.0 may JIT kernels (FlashInfer, Triton), which needs ``nvcc`` and headers that
  a slim base lacks.

All images set ``PYTHONDONTWRITEBYTECODE=1`` so importing NVIDIA code never writes ``.pyc`` files
into snapshot directories (site-packages is still byte-compiled at build time).
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from flora_deploy.config.pins import RUNTIME
from flora_deploy.layout import CACHE_MOUNT

if TYPE_CHECKING:
    import modal

PY = RUNTIME["python_image"]
PILLOW = f"Pillow=={RUNTIME['pillow']}"
AV = f"av=={RUNTIME['av']}"
# One hub/xet version in every image: img_cpu writes the Volume's ``.cache/huggingface`` metadata
# and img_hfref reads it with NVIDIA's pinned hub (review M3).
HF_HUB = f"huggingface_hub=={RUNTIME['huggingface_hub_reference']}"
HF_XET = f"hf_xet=={RUNTIME['hf_xet']}"
NUMPY = f"numpy=={RUNTIME['numpy']}"
# Tag kept for readability; the digest (multi-arch index) is what is pulled.
CUDA_BASE = f"{RUNTIME['cuda_base_vllm']}@{RUNTIME['cuda_base_vllm_digest']}"

CPU_PACKAGES = (
    AV,
    NUMPY,  # exact: debian_slim ships numpy 1.26, which breaks pyarrow 26
    PILLOW,
    f"pyarrow=={RUNTIME['pyarrow']}",
    HF_HUB,
    HF_XET,
)

# NVIDIA TempSeg/Captioner batch_utils/requirements.txt at the code pin, with Pillow and numpy
# pinned to one exact version inside NVIDIA's ranges (">=11,<13", "<3"; decode pins, PLAN.md D11).
HFREF_PACKAGES = (
    f"torch=={RUNTIME['torch_reference']}",
    f"torchvision=={RUNTIME['torchvision_reference']}",
    f"huggingface_hub=={RUNTIME['huggingface_hub_reference']}",
    f"transformers=={RUNTIME['transformers']}",
    f"peft=={RUNTIME['peft']}",
    f"accelerate=={RUNTIME['accelerate']}",
    f"safetensors=={RUNTIME['safetensors']}",
    AV,
    NUMPY,
    PILLOW,
    HF_XET,
)

# numpy is left to vLLM: its mistral-common dependency needs numpy<2.4 on CPython 3.12.
VLLM_PACKAGES = (
    f"vllm=={RUNTIME['vllm']}",
    f"transformers=={RUNTIME['transformers']}",
    f"qwen-vl-utils=={RUNTIME['qwen_vl_utils']}",
    f"decord=={RUNTIME['decord']}",
    AV,
    PILLOW,
    HF_HUB,
    HF_XET,
)

PACKAGES = {"img_cpu": CPU_PACKAGES, "img_hfref": HFREF_PACKAGES, "img_vllm": VLLM_PACKAGES}

LOCK_DIR = Path(__file__).with_name("locks")
LOCK_FILES = {"img_cpu": "cpu.txt", "img_hfref": "hfref.txt", "img_vllm": "vllm.txt"}
# Resolution target per lock: the base's glibc (debian bookworm 2.36, ubuntu 24.04 2.39), so the
# resolver only picks versions that have a wheel the base can install (vLLM's llguidance 1.7 has
# no wheel older than manylinux_2_31).
LOCK_PLATFORMS = {
    "img_cpu": "x86_64-manylinux_2_36",
    "img_hfref": "x86_64-manylinux_2_36",
    "img_vllm": "x86_64-manylinux_2_39",
}
INSTALL_OPTIONS = "--require-hashes --no-build"

_ENV = {"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
ENV = {
    "img_cpu": _ENV,
    "img_hfref": _ENV,
    "img_vllm": {**_ENV, "VLLM_CACHE_ROOT": str(CACHE_MOUNT)},
}
# debian_slim's Python micro version and Debian release follow the builder version (2025.06:
# 3.12.10 on bookworm); add_python's standalone build follows the Modal client (3.12.1 at 1.6.2).
BASES = {
    "img_cpu": f"debian_slim(python={PY})",
    "img_hfref": f"debian_slim(python={PY})",
    "img_vllm": f"{CUDA_BASE}+add_python({PY})",
}

_BUILDER_ENV = "MODAL_IMAGE_BUILDER_VERSION"


def lock_path(name: str) -> Path:
    return LOCK_DIR / LOCK_FILES[name]


def _pin_builder_version() -> None:
    """Modal reads the image builder version from this env var (else the workspace setting) when
    an image is built, so set it before any build; refuse a conflicting override."""
    want = RUNTIME["modal_image_builder_version"]
    have = os.environ.setdefault(_BUILDER_ENV, want)
    if have != want:
        raise RuntimeError(f"{_BUILDER_ENV}={have!r}, but the images are pinned to {want!r}")


def _install(image: modal.Image, name: str) -> modal.Image:
    return (
        image.uv_pip_install(
            requirements=[str(lock_path(name))],
            extra_options=INSTALL_OPTIONS,
            uv_version=RUNTIME["uv"],
        )
        .env(ENV[name])
        .add_local_python_source("flora_deploy")
    )


@cache
def cpu_image() -> modal.Image:
    import modal

    _pin_builder_version()
    return _install(modal.Image.debian_slim(python_version=PY), "img_cpu")


@cache
def hfref_image() -> modal.Image:
    import modal

    _pin_builder_version()
    return _install(modal.Image.debian_slim(python_version=PY), "img_hfref")


@cache
def vllm_image() -> modal.Image:
    import modal

    _pin_builder_version()
    base = modal.Image.from_registry(CUDA_BASE, add_python=PY).entrypoint([])
    return _install(base, "img_vllm")


def image_fingerprint(name: str) -> str:
    """Stable sha256 of everything that defines image ``name``'s contents (PLAN.md D11): the lock
    file's bytes, the base (builder version, Python series, CUDA digest), uv version, install
    options and env. The ``flora_deploy`` source is excluded; the config hash covers it as the
    framework logic version."""
    spec = {
        "image": name,
        "base": BASES[name],
        "lock_platform": LOCK_PLATFORMS[name],
        "modal_image_builder_version": RUNTIME["modal_image_builder_version"],
        "uv": RUNTIME["uv"],
        "install_options": INSTALL_OPTIONS,
        "lock_sha256": hashlib.sha256(lock_path(name).read_bytes()).hexdigest(),
        "env": dict(sorted(ENV[name].items())),
    }
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def image_fingerprints() -> dict[str, str]:
    return {name: image_fingerprint(name) for name in _BUILDERS}


_BUILDERS = {"img_cpu": cpu_image, "img_hfref": hfref_image, "img_vllm": vllm_image}


def __getattr__(name: str):
    if name in _BUILDERS:
        return _BUILDERS[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
