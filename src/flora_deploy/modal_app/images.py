"""Pinned Modal images (PLAN.md §7). All x86_64.

Each builder returns a lazy ``modal.Image``: nothing is built or uploaded until an app using it
runs. Access ``images.img_cpu`` / ``img_hfref`` / ``img_vllm`` or call the builders directly.

Base choices:
- ``img_cpu``: ``debian_slim`` with apt ffmpeg. No GPU.
- ``img_hfref``: ``debian_slim`` plus NVIDIA's pip stack verbatim. The torch 2.11.0 PyPI wheel
  ships its CUDA 13.0 user space as ``nvidia-*`` wheels, and Modal's host driver (CUDA 13.3)
  supports it on H200 (SM90). NVIDIA's reference runs HF transformers with SDPA and compiles
  nothing, so it needs no toolkit. This is also exactly what ``pip install -r requirements.txt``
  gives on a plain host.
- ``img_vllm``: ``nvidia/cuda`` *devel* 13.0 (toolkit at or below the host's 13.3, matching torch's
  cu130). vLLM 0.24.0 may JIT kernels (FlashInfer, Triton), which needs ``nvcc`` and headers that
  a slim base lacks.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from flora_deploy.config.pins import RUNTIME
from flora_deploy.layout import CACHE_MOUNT

if TYPE_CHECKING:
    import modal

PY = RUNTIME["python_image"]
PILLOW = f"Pillow=={RUNTIME['pillow']}"
HF_HUB = f"huggingface_hub{RUNTIME['huggingface_hub']}"
AV = f"av=={RUNTIME['av']}"

CPU_PACKAGES = (
    AV,
    "numpy<3",
    PILLOW,
    f"pyarrow=={RUNTIME['pyarrow']}",
    HF_HUB,
    "hf_xet",
)

# NVIDIA TempSeg/Captioner batch_utils/requirements.txt at the code pin, with Pillow pinned to
# one exact version inside NVIDIA's ">=11,<13" range (decode pins, PLAN.md D11).
HFREF_PACKAGES = (
    f"torch=={RUNTIME['torch_reference']}",
    f"torchvision=={RUNTIME['torchvision_reference']}",
    f"huggingface_hub=={RUNTIME['huggingface_hub_reference']}",
    f"transformers=={RUNTIME['transformers']}",
    f"peft=={RUNTIME['peft']}",
    f"accelerate=={RUNTIME['accelerate']}",
    f"safetensors=={RUNTIME['safetensors']}",
    AV,
    "numpy<3",
    PILLOW,
)

VLLM_PACKAGES = (
    f"vllm=={RUNTIME['vllm']}",
    f"transformers=={RUNTIME['transformers']}",
    f"qwen-vl-utils=={RUNTIME['qwen_vl_utils']}",
    f"decord=={RUNTIME['decord']}",
    AV,
    PILLOW,
    HF_HUB,
)

_HF_ENV = {"HF_XET_HIGH_PERFORMANCE": "1"}


@cache
def cpu_image() -> modal.Image:
    import modal

    return (
        modal.Image.debian_slim(python_version=PY)
        .apt_install("ffmpeg")
        .uv_pip_install(*CPU_PACKAGES)
        .env(_HF_ENV)
        .add_local_python_source("flora_deploy")
    )


@cache
def hfref_image() -> modal.Image:
    import modal

    return (
        modal.Image.debian_slim(python_version=PY)
        .uv_pip_install(*HFREF_PACKAGES)
        .env(_HF_ENV)
        .add_local_python_source("flora_deploy")
    )


@cache
def vllm_image() -> modal.Image:
    import modal

    return (
        modal.Image.from_registry(RUNTIME["cuda_base_vllm"], add_python=PY)
        .entrypoint([])
        .uv_pip_install(*VLLM_PACKAGES)
        .env({**_HF_ENV, "VLLM_CACHE_ROOT": str(CACHE_MOUNT)})
        .add_local_python_source("flora_deploy")
    )


_BUILDERS = {"img_cpu": cpu_image, "img_hfref": hfref_image, "img_vllm": vllm_image}


def __getattr__(name: str):
    if name in _BUILDERS:
        return _BUILDERS[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
