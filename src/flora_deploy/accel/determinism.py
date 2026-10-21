"""Opt-in run-to-run deterministic GPU mode for the ``exact`` preset (D-1, ``docs/determinism.md``).

NVIDIA's reference (``batch_utils/models.generate``: greedy, batch 1) is not bit-reproducible
run to run on GPU with torch's default settings. This module is the one switch that makes it so,
**without touching NVIDIA's files, arguments or model inputs**: it only sets process-wide torch
flags and one cuBLAS env var, and records them in provenance.

Modes:

- ``nvidia`` (default): change nothing. Exactly NVIDIA's validated behavior.
- ``deterministic``: SDPA's cuDNN backend off (``torch.backends.cuda.enable_cudnn_sdp(False)``).
  This is the measured root cause on H200: torch 2.11 prefers that backend on sm90, and its
  decode is not run-to-run reproducible. The mode also sets
  ``torch.use_deterministic_algorithms(True)``, which on its own also steers SDPA away from
  cuDNN and makes any other flagged op raise. Further settings: ``cudnn.deterministic=True``,
  ``cudnn.benchmark=False``, and ``CUBLAS_WORKSPACE_CONFIG=:4096:8``. The env var must be set
  before the first cuBLAS call, so put it in the container env via :func:`container_env`.
  TF32, autocast and NVIDIA's ``generate`` arguments are untouched. On H200 this mode changes
  logits (and sometimes text) versus the default cuDNN path. It is a different reproducible
  reference, not the same bits.

Usage (before NVIDIA's ``Session.load``)::

    env = {**offline_env(...), **determinism.container_env("deterministic")}   # Modal env=
    ...
    record["determinism"] = determinism.apply("deterministic")   # in the container, pre-load

``apply`` refuses (``DeterminismError``) when the cuBLAS env var is missing or wrong, or when it
would have to set it itself after CUDA is already initialized (too late to take effect).
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping, MutableMapping
from typing import Any

DETERMINISM_SCHEMA = "flora_determinism_v1"
NVIDIA = "nvidia"
DETERMINISTIC = "deterministic"
MODES = (NVIDIA, DETERMINISTIC)
CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# The two values torch accepts for deterministic cuBLAS (torch.use_deterministic_algorithms doc).
CUBLAS_ACCEPTED = (":4096:8", ":16:8")
OPTIONAL_KERNELS = ("fla", "causal_conv1d", "flash_attn", "triton")


class DeterminismError(RuntimeError):
    """The requested determinism mode cannot be (or was not) put in place."""


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise DeterminismError(f"unknown determinism mode {mode!r}; expected one of {MODES}")
    return mode


def container_env(mode: str) -> dict[str, str]:
    """Env vars the container must start with for ``mode`` (merge into Modal ``env=``)."""
    if check_mode(mode) == DETERMINISTIC:
        return {CUBLAS_ENV: CUBLAS_WORKSPACE_CONFIG}
    return {}


def _torch(torch: Any = None) -> Any:
    if torch is None:
        import torch as torch_module

        return torch_module
    return torch


def flags(torch: Any = None, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The process-wide settings that decide whether identical inputs give identical outputs."""
    torch = _torch(torch)
    environ = os.environ if environ is None else environ
    cuda = torch.backends.cuda
    out: dict[str, Any] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": environ.get(CUBLAS_ENV),
        "sdpa": {
            "flash": cuda.flash_sdp_enabled(),
            "mem_efficient": cuda.mem_efficient_sdp_enabled(),
            "math": cuda.math_sdp_enabled(),
            "cudnn": cuda.cudnn_sdp_enabled(),
        },
        "optional_kernels": {m: importlib.util.find_spec(m) is not None
                             for m in OPTIONAL_KERNELS},
        "torch": getattr(torch, "__version__", None),
    }
    det = getattr(getattr(torch, "utils", None), "deterministic", None)
    if det is not None and hasattr(det, "fill_uninitialized_memory"):
        out["fill_uninitialized_memory"] = bool(det.fill_uninitialized_memory)
    return out


def apply(
    mode: str,
    *,
    torch: Any = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Put ``mode`` in place for this process and return its provenance record.

    Call before NVIDIA's ``Session.load`` (before any CUDA work). ``nvidia`` changes nothing and
    only records the flags. ``deterministic`` requires ``CUBLAS_WORKSPACE_CONFIG`` to be one of the
    accepted values; if it is absent and CUDA is not initialized yet it is set here, otherwise
    this raises (torch reads the variable once, when its first cuBLAS handle is made)."""
    check_mode(mode)
    torch = _torch(torch)
    environ = os.environ if environ is None else environ
    set_env = False
    if mode == DETERMINISTIC:
        have = environ.get(CUBLAS_ENV)
        if have is None:
            if torch.cuda.is_initialized():
                raise DeterminismError(
                    f"{CUBLAS_ENV} must be set before CUDA initializes; set it in the container "
                    f"env (determinism.container_env)"
                )
            environ[CUBLAS_ENV] = CUBLAS_WORKSPACE_CONFIG
            set_env = True
        elif have not in CUBLAS_ACCEPTED:
            raise DeterminismError(f"{CUBLAS_ENV}={have!r}; deterministic cuBLAS needs one of "
                                   f"{CUBLAS_ACCEPTED}")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_cudnn_sdp(False)
    return provenance(mode, torch=torch, environ=environ, set_env=set_env)


def provenance(
    mode: str,
    *,
    torch: Any = None,
    environ: Mapping[str, str] | None = None,
    set_env: bool = False,
) -> dict[str, Any]:
    """The record stored next to every result: mode, the live flags, and whether they match."""
    check_mode(mode)
    live = flags(torch, environ)
    record = {"schema": DETERMINISM_SCHEMA, "mode": mode, "flags": live,
              "cublas_env_set_by_apply": set_env}
    if mode == DETERMINISTIC:
        record["in_effect"] = bool(
            live["deterministic_algorithms"]
            and not live["deterministic_warn_only"]
            and live["cudnn_deterministic"]
            and not live["cudnn_benchmark"]
            and live["cublas_workspace_config"] in CUBLAS_ACCEPTED
            and not live["sdpa"]["cudnn"]
        )
    else:
        record["in_effect"] = True
    return record


def verify(record: Mapping[str, Any], torch: Any = None,
           environ: Mapping[str, str] | None = None) -> None:
    """Raise if the live process no longer matches a provenance ``record`` from :func:`apply`
    (call before persisting a result)."""
    now = provenance(record["mode"], torch=torch, environ=environ)
    if not now["in_effect"]:
        raise DeterminismError(f"determinism mode {record['mode']!r} is no longer in effect")
    keys = ("deterministic_algorithms", "deterministic_warn_only", "cudnn_deterministic",
            "cudnn_benchmark", "matmul_allow_tf32", "cudnn_allow_tf32", "cublas_workspace_config",
            "float32_matmul_precision", "sdpa")
    changed = {k: (record["flags"].get(k), now["flags"].get(k)) for k in keys
               if record["flags"].get(k) != now["flags"].get(k)}
    if changed:
        raise DeterminismError(f"determinism flags changed since apply(): {changed}")
