"""FP32 ``dt_bias`` for vLLM 0.24.0's GDN layers, as a ``vllm.general_plugins`` plugin (D5).

vLLM creates ``QwenGatedDeltaNetAttention.dt_bias`` in the model dtype (bf16) and the loader's
``copy_`` silently rounds the checkpoint's FP32 values; ``A_log`` is already FP32 (vLLM
``layers/mamba/gdn/qwen_gdn_linear_attn.py:516-527`` at v0.24.0; the upstream fix #44270 was
closed unmerged). NVIDIA's reference keeps both FP32 (``_keep_native_gdn_fp32``). Every kernel
that reads ``dt_bias`` upcasts with ``.to(tl.float32)`` / ``a.float() + dt_bias``, so an FP32
parameter needs no kernel change.

**Off by default; Gate A decides** (PLAN.md §9). The patch acts only when *both* hold:

1. vLLM discovers the entry point. This package is not pip-installed in the Modal images, so no
   entry point exists unless ``install_entry_point(dir)`` writes a minimal ``.dist-info`` and the
   returned env puts it on ``PYTHONPATH`` (vLLM loads general plugins in the front-end, the
   engine-core and every worker process, which inherit the env);
2. ``FLORA_DTBIAS_FP32=1`` in the environment. ``register()`` is a no-op otherwise, so a stray
   entry point cannot turn it on.

``check_activation(want)`` is called by ``engines.offline_llm.OfflineEngine`` before building an
engine: it refuses an engine whose spec and environment disagree, so a run cannot silently mix
patched and unpatched engines. The patch is idempotent (plugins may load several times per
process).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "vllm.general_plugins"
ENTRY_POINT_NAME = "flora_dtbias_fp32"
ENTRY_POINT_VALUE = "flora_deploy.engines.patches.dt_bias_fp32:register"
ENV_FLAG = "FLORA_DTBIAS_FP32"
DIST_NAME = "flora_dtbias_plugin"
PATCH_MARK = "_flora_dtbias_fp32"
TARGET = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention"


class DtBiasActivationError(RuntimeError):
    """The engine spec and the process environment disagree about the dt_bias plugin."""


def enabled(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_FLAG) == "1"


def check_activation(want: bool, environ: dict[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    have = enabled(env)
    if want and not have:
        raise DtBiasActivationError(
            f"spec asks for FP32 dt_bias but {ENV_FLAG}!=1: start the process with "
            "install_entry_point(...)'s env")
    if not want and have:
        raise DtBiasActivationError(
            f"{ENV_FLAG}=1 in the environment but the spec does not ask for FP32 dt_bias")
    plugins = env.get("VLLM_PLUGINS")
    if want and plugins is not None and ENTRY_POINT_NAME not in plugins.split(","):
        raise DtBiasActivationError(f"VLLM_PLUGINS={plugins!r} excludes {ENTRY_POINT_NAME}")


def patch_class(cls: Any) -> bool:
    """Make ``cls.__init__`` re-create ``dt_bias`` as an FP32 parameter with the same weight
    loader. Returns False if ``cls`` was already patched."""
    if getattr(cls, PATCH_MARK, False):
        return False
    import torch
    from torch import nn

    original = cls.__init__

    def __init__(self, *args, **kwargs):  # noqa: N807 - replaces the class's __init__
        original(self, *args, **kwargs)
        old = self.dt_bias
        loader = getattr(old, "weight_loader", None)
        new = nn.Parameter(old.data.to(torch.float32), requires_grad=old.requires_grad)
        for attr, value in vars(old).items():  # weight_loader and any other vLLM attrs
            setattr(new, attr, value)
        if loader is not None:
            new.weight_loader = loader
        self.dt_bias = new

    __init__.__wrapped__ = original  # type: ignore[attr-defined]
    cls.__init__ = __init__
    setattr(cls, PATCH_MARK, True)
    return True


def register() -> None:
    """The entry point vLLM calls in every process. No-op unless ``FLORA_DTBIAS_FP32=1``."""
    if not enabled():
        return
    import importlib

    module, _, name = TARGET.rpartition(".")
    patch_class(getattr(importlib.import_module(module), name))


def install_entry_point(root: str | Path) -> dict[str, str]:
    """Write ``<root>/flora_dtbias_plugin-0.0.0.dist-info`` (METADATA + entry_points.txt) and
    return the env that activates the plugin for processes started afterwards. Also puts
    ``root`` on this process's ``sys.path``."""
    root = Path(root)
    dist = root / f"{DIST_NAME}-0.0.0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {DIST_NAME}\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text(
        f"[{ENTRY_POINT_GROUP}]\n{ENTRY_POINT_NAME} = {ENTRY_POINT_VALUE}\n")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = os.environ.get("PYTHONPATH", "")
    return {
        "PYTHONPATH": str(root) + (os.pathsep + path if path else ""),
        ENV_FLAG: "1",
        "VLLM_PLUGINS": ENTRY_POINT_NAME,
    }


def provenance() -> dict:
    return {"plugin": ENTRY_POINT_NAME, "target": TARGET, "enabled": enabled()}
