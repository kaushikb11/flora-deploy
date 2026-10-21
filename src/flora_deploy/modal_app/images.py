"""Pinned Modal images (PLAN.md §7). All x86_64.

Each builder returns a lazy ``modal.Image``: nothing is built or uploaded until an app using it
runs. Access ``images.img_cpu`` / ``img_hfref`` / ``img_vllm`` or call the builders directly.

Reproducibility (review M3, PLAN.md D11). Each image installs from a committed, fully transitive
lock (``locks/<image>.txt``: exact versions and sha256 hashes for linux x86_64 at the base's
CPython), with ``--require-hashes --no-build``, using the uv binary pinned by digest. The
``*_PACKAGES`` tuples below are only the lock *inputs*; ``scripts/lock_images.py`` turns them into
``locks/<image>.in`` and the lock. Every base is pulled by digest (``BASES``, the single source the
builders read), and the Modal client (which renders each builder step and picks ``add_python``'s
standalone build) and image builder version are pinned and checked before any build.
``image_fingerprint()`` hashes all of it for the config hash. Gate C showed why one pinned build
matters: NVIDIA picks caption frame scales by compressed PNG size, which depends on the zlib in
the Pillow wheel, so golden data and production must share the exact image stack.

Base choices:
- ``img_cpu``: the official ``python:3.12.10-slim-bookworm`` by digest, no apt packages (Modal's
  ``debian_slim`` adds unpinned ``apt-get install`` and ``pip install --upgrade`` layers). PyAV
  wheels bundle FFmpeg.
- ``img_hfref``: the same base plus NVIDIA's TempSeg/Captioner install recipe
  (``batch_utils/README.md``): torch 2.11.0 / torchvision 0.26.0 from the PyTorch cu128 index
  (CUDA 12.8 user space as ``nvidia-*-cu12`` wheels), then ``requirements.txt``. Modal's host
  driver (CUDA 13.x) runs CUDA 12.8 user space. NVIDIA's reference runs HF transformers with SDPA
  and compiles nothing, so it needs no toolkit.
- ``img_vllm``: ``nvidia/cuda`` *devel* 13.0 (toolkit at or below the host's driver, matching the
  PyPI torch cu130 that vLLM 0.24.0 and NVIDIA's Rewarder ``pip install -r`` get). vLLM may JIT
  kernels (FlashInfer, Triton), which needs ``nvcc`` and headers. Python is Modal's
  ``add_python`` standalone build (3.12.1 at client 1.6.2).

All images set ``PYTHONDONTWRITEBYTECODE=1`` so importing NVIDIA code never writes ``.pyc`` files
into snapshot directories (site-packages is still byte-compiled at build time). The lock files are
mounted with the ``flora_deploy`` source, so ``image_fingerprint()`` also works in a container.
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
# One numpy in every image (review A6); vLLM's mistral-common needs numpy<2.4 on CPython 3.12.
NUMPY = f"numpy=={RUNTIME['numpy']}"
# Tags kept for readability; the digest (multi-arch index) is what is pulled.
PY_BASE = f"{RUNTIME['python_base']}@{RUNTIME['python_base_digest']}"
CUDA_BASE = f"{RUNTIME['cuda_base_vllm']}@{RUNTIME['cuda_base_vllm_digest']}"
# By digest only: Modal's builder (skopeo) rejects a COPY --from source with both tag and digest,
# which is why the install step is written out below instead of using ``uv_pip_install``.
UV_IMAGE = f"ghcr.io/astral-sh/uv@{RUNTIME['uv_image_digest']}"  # uv RUNTIME["uv"]

CPU_PACKAGES = (
    AV,
    NUMPY,  # exact: an unpinned resolution once gave numpy 1.26, which breaks pyarrow 26
    PILLOW,
    f"pyarrow=={RUNTIME['pyarrow']}",
    HF_HUB,
    HF_XET,
)

# NVIDIA TempSeg/Captioner batch_utils/requirements.txt at the code pin, with Pillow and numpy
# pinned to one exact version inside NVIDIA's ranges (">=11,<13", "<3"; decode pins, PLAN.md D11).
# torch/torchvision are overridden to the +cu128 wheels (OVERRIDES).
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

VLLM_PACKAGES = (
    f"vllm=={RUNTIME['vllm']}",
    f"transformers=={RUNTIME['transformers']}",
    f"qwen-vl-utils=={RUNTIME['qwen_vl_utils']}",
    f"decord=={RUNTIME['decord']}",
    AV,
    NUMPY,
    PILLOW,
    HF_HUB,
    HF_XET,
)

PACKAGES = {"img_cpu": CPU_PACKAGES, "img_hfref": HFREF_PACKAGES, "img_vllm": VLLM_PACKAGES}

LOCK_DIR = Path(__file__).with_name("locks")
LOCK_FILES = {"img_cpu": "cpu.txt", "img_hfref": "hfref.txt", "img_vllm": "vllm.txt"}
# Resolution target per lock: the base's glibc (debian bookworm 2.36, ubuntu 24.04 2.39), so the
# resolver only picks versions that have a wheel the base can install (vLLM's llguidance 1.7 has
# no wheel older than manylinux_2_31), and the base's exact CPython for marker evaluation.
LOCK_PLATFORMS = {
    "img_cpu": "x86_64-manylinux_2_36",
    "img_hfref": "x86_64-manylinux_2_36",
    "img_vllm": "x86_64-manylinux_2_39",
}
_PY_BASE_VERSION = RUNTIME["python_base"].split(":")[1].split("-")[0]
LOCK_PYTHON = {
    "img_cpu": _PY_BASE_VERSION,
    "img_hfref": _PY_BASE_VERSION,
    "img_vllm": RUNTIME["python_standalone"],
}


def _cu_wheel(pkg: str, version_key: str) -> str:
    """``pkg @ <PyTorch index wheel URL>#sha256=...`` for NVIDIA's cu128 torch/torchvision."""
    version = RUNTIME[version_key].replace("+", "%2B")
    sha = RUNTIME[f"{version_key}_sha256"]
    wheel = f"{pkg}-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
    return f"{pkg} @ {RUNTIME['torch_index_hfref']}/{wheel}#sha256={sha}"


# ``uv pip compile --overrides`` per image (scripts/lock_images.py writes locks/<image>.overrides
# .txt): img_hfref takes torch/torchvision from the PyPI-external cu128 index by exact URL, so the
# lock (and hence the install) needs no extra index and every other wheel stays on PyPI.
OVERRIDES: dict[str, tuple[str, ...]] = {
    "img_cpu": (),
    "img_hfref": (_cu_wheel("torch", "torch_hfref"), _cu_wheel("torchvision", "torchvision_hfref")),
    "img_vllm": (),
}
INSTALL_OPTIONS = {name: "--require-hashes --no-build" for name in OVERRIDES}

_ENV = {"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
ENV = {
    "img_cpu": _ENV,
    "img_hfref": _ENV,
    "img_vllm": {**_ENV, "VLLM_CACHE_ROOT": str(CACHE_MOUNT)},
}

# What each builder passes to ``modal.Image.from_registry`` (and ``.entrypoint``). ``_build`` reads
# this dict, so it cannot drift from the fingerprint. At builder version 2025.06,
# ``from_registry`` renders only ``FROM <ref>`` plus, with add_python, the standalone COPY/ENV/ln.
_STANDALONE = f"{RUNTIME['python_standalone_release']}/{RUNTIME['python_standalone']}"
BASES: dict[str, dict] = {
    "img_cpu": {"from": PY_BASE, "add_python": None, "python_standalone": None, "entrypoint": None},
    "img_hfref": {
        "from": PY_BASE, "add_python": None, "python_standalone": None, "entrypoint": None,
    },
    "img_vllm": {
        "from": CUDA_BASE, "add_python": PY, "python_standalone": _STANDALONE, "entrypoint": [],
    },
}

_BUILDER_ENV = "MODAL_IMAGE_BUILDER_VERSION"
# Mount the locks with the source (default is .py only) so image_fingerprint() runs in containers.
SOURCE_PATTERNS = ("**/*.py", "**/modal_app/locks/*.txt")


def lock_path(name: str) -> Path:
    return LOCK_DIR / LOCK_FILES[name]


def _pin_builder_version() -> None:
    """Modal reads the image builder version from this env var (else the workspace setting) when
    an image is built, so set it before any build; refuse a conflicting override."""
    want = RUNTIME["modal_image_builder_version"]
    have = os.environ.setdefault(_BUILDER_ENV, want)
    if have != want:
        raise RuntimeError(f"{_BUILDER_ENV}={have!r}, but the images are pinned to {want!r}")


def _check_modal_client() -> None:
    """The client renders every Dockerfile step and picks add_python's build, so another client
    could build different images under the same fingerprint. Checked where images are built
    (locally); containers run the client Modal mounts and only import the builders."""
    import modal
    from modal.mount import PYTHON_STANDALONE_VERSIONS

    if not modal.is_local():
        return
    if modal.__version__ != RUNTIME["modal_client"]:
        raise RuntimeError(
            f"modal client {modal.__version__}, but the images are pinned to "
            f"{RUNTIME['modal_client']} (RUNTIME['modal_client'])"
        )
    release, version = PYTHON_STANDALONE_VERSIONS[PY]
    if f"{release}/{version}" != _STANDALONE:
        raise RuntimeError(f"add_python({PY}) is {release}/{version}, pinned {_STANDALONE}")


def _lock_context_path(name: str) -> str:
    return f"/.lock_{LOCK_FILES[name]}"


def install_commands(name: str) -> list[str]:
    """The exact Dockerfile steps that install the lock (what ``uv_pip_install`` renders, with
    the uv image by digest)."""
    dest = f"/.uv/{LOCK_FILES[name]}"
    return [
        f"COPY --from={UV_IMAGE} /uv /.uv/uv",
        f"COPY {_lock_context_path(name)} {dest}",
        "RUN /.uv/uv pip install --python $(command -v python) --compile-bytecode "
        f"{INSTALL_OPTIONS[name]} --requirements {dest}",
    ]


def _build(name: str) -> modal.Image:
    import modal

    _pin_builder_version()
    _check_modal_client()
    base = BASES[name]
    image = modal.Image.from_registry(base["from"], add_python=base["add_python"])
    if base["entrypoint"] is not None:
        image = image.entrypoint(base["entrypoint"])
    return (
        image.dockerfile_commands(
            install_commands(name),
            context_files={_lock_context_path(name): str(lock_path(name))},
        )
        .env(ENV[name])
        .add_local_python_source("flora_deploy", ignore=~modal.FilePatternMatcher(*SOURCE_PATTERNS))
    )


@cache
def cpu_image() -> modal.Image:
    return _build("img_cpu")


@cache
def hfref_image() -> modal.Image:
    return _build("img_hfref")


@cache
def vllm_image() -> modal.Image:
    return _build("img_vllm")


def lock_digest(text: str) -> str:
    """sha256 of a lock's requirement lines with comments, blank lines and line continuations
    removed, so the header and ``# via`` annotations (which name paths and the compile command)
    never change the fingerprint; only requirements, hashes and options count."""
    lines = []
    for raw in text.splitlines():
        line = "" if raw.lstrip().startswith("#") else raw.split(" #", 1)[0]
        line = line.strip().removesuffix("\\").strip()
        if line:
            lines.append(line)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def image_fingerprint(name: str) -> str:
    """Stable sha256 of everything that defines image ``name``'s contents (PLAN.md D11): the
    normalized lock, the base spec (digest, add_python standalone build, entrypoint), the Modal
    client and image builder versions (which fix how every step renders), the install steps
    (uv image digest and options), lock target and env. Uses only embedded constants and the
    lock files, which are mounted into containers. The ``flora_deploy`` source is excluded; the
    config hash covers it as the framework logic version."""
    spec = {
        "image": name,
        "base": BASES[name],
        "lock_platform": LOCK_PLATFORMS[name],
        "lock_python": LOCK_PYTHON[name],
        "modal_client": RUNTIME["modal_client"],
        "modal_image_builder_version": RUNTIME["modal_image_builder_version"],
        "uv": RUNTIME["uv"],
        "install_commands": install_commands(name),
        "lock_sha256": lock_digest(lock_path(name).read_text()),
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
