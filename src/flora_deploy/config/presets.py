"""Requirement presets (PLAN.md §2, D9, D16): which shard kinds run which stage, on which GPU.

A preset maps each pipeline stage to a shard kind and gives every shard kind a default pool
(GPU, count, cores, memory). ``flora.yaml`` ``stages:`` overrides any of these per shard kind.

- ``exact``: NVIDIA's own pipeline. Segment and caption run together in one ``segcap`` worker on
  an ``H200`` (TempSeg and Captioner both resident; ``docs/reference-smoke.md``: 8 cores, 96 GiB,
  80.95 GiB HBM high-water mark). Rewards run NVIDIA's offline vLLM engine on ``H100!``
  (``docs/cold-start.md``: 8 cores, >= 72 GiB so the 52 GiB snapshot stays in the page cache).
- ``balanced``: vLLM 0.24.0 engines, stage-separated pools (D8). TempSeg on ``L40S`` (6 cores,
  §6.7), Captioner on ``H100!`` (8 cores; memory >= BF16 snapshot ≈ 19 GiB + 16 GiB, plus
  decode/caption-cache headroom), rewards as in ``exact``.
- ``budget`` / ``interactive``: declared (D9) but not available yet; selecting one is an error.

Certification (D9, D16): ``exact`` is the reference by construction. Every other preset needs a
Gate certificate matching (shard kind, GPU, engine spec) before it may run (``find_certificates``;
the registry is ``config/certificates/``, the match rule ``validate/certkey.py``). Without a
``certified`` one (``provisional`` does not count), ``balanced`` is refused unless the config
opts in with ``allow_uncertified: true`` (its records are then an uncertified stack).

GPU overrides are checked against ``GPU_MEMORY_GIB`` (Modal GPU types) and a per-shard-kind
minimum (``min_gpu_memory_gib``): exact ``segcap`` keeps TempSeg and Captioner resident (80.95 GiB
HBM high-water), so it needs an H200/B200-class GPU; rewards need an 80 GB-class GPU.

Memory values are the pre-P0 defaults; P0 measurements replace them (§6.7).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from flora_deploy.jobs.settings import PoolSettings, Preset
from flora_deploy.jobs.types import ShardKind
from flora_deploy.records import Stage

_REWARDS: dict[ShardKind, PoolSettings] = {
    "reward.terminal": PoolSettings(gpu="H100!", cpu_cores=8.0, memory_gib=80.0),
    "reward.dense": PoolSettings(gpu="H100!", cpu_cores=8.0, memory_gib=80.0),
    # Rubric runs NVIDIA's settings at TP=2 (PLAN.md §7).
    "reward.rubric": PoolSettings(gpu="H100!", gpu_count=2, cpu_cores=8.0, memory_gib=96.0),
}


@dataclass(frozen=True)
class EngineIdentity:
    """What produces a shard kind's outputs; hashed (D11). GPU is deliberately not here.

    ``patches`` lists the output-identical runtime wraps / engine patches applied (D17); it and
    the arg/sampling/mm_kwargs placeholders are filled in as S1 and P1b-1 land."""

    name: str
    version: dict[str, str]
    image: str  # modal_app.images name, e.g. "img_hfref"
    patches: tuple[str, ...] = ()
    engine_args: dict = field(default_factory=dict)  # placeholder until P1b-1
    sampling: dict = field(default_factory=dict)  # placeholder until P1b-1
    mm_kwargs: dict = field(default_factory=dict)  # placeholder until P1b-1


@dataclass(frozen=True)
class PresetSpec:
    name: Preset
    available: bool
    stage_kinds: dict[Stage, ShardKind]  # pipeline stage -> shard kind that produces it
    pools: dict[ShardKind, PoolSettings]
    engines: dict[ShardKind, EngineIdentity]
    # NVIDIA pin keys (config/pins.py) each shard kind depends on.
    pins: dict[ShardKind, tuple[str, ...]]
    # Derived-weight receipts (prep receipts) each shard kind needs, by name.
    derived_weights: dict[ShardKind, tuple[str, ...]]
    note: str = ""


def _engines() -> tuple[EngineIdentity, EngineIdentity, EngineIdentity]:
    from flora_deploy.config.pins import RUNTIME
    from flora_deploy.nv.capture import REQUIRED_VLLM_ENGINE_KWARGS

    # Review 3 A9: the versions img_hfref actually installs (locks/hfref.txt): torch is the
    # cu128 wheel, and qwen_vl_utils / decord are not in that image (NVIDIA's HF path does not
    # use them), so they are not part of this identity. The image fingerprint covers the rest.
    hf = EngineIdentity(
        name="nvidia-hf",
        version={"transformers": RUNTIME["transformers"], "torch": RUNTIME["torch_hfref"],
                 "peft": RUNTIME["peft"]},
        image="img_hfref",
    )
    nv_vllm = EngineIdentity(name="nvidia-vllm-offline", version={"vllm": RUNTIME["vllm"]},
                             image="img_vllm")
    # Every vLLM engine that carries Flora video must use these (Gate C), so they are hashed.
    vllm = EngineIdentity(name="vllm-offline", version={"vllm": RUNTIME["vllm"]},
                          image="img_vllm", engine_args=dict(REQUIRED_VLLM_ENGINE_KWARGS))
    return hf, nv_vllm, vllm


_REWARD_PINS: tuple[str, ...] = ("rewarder",)
_SEG_PINS: tuple[str, ...] = ("tempseg", "tempseg_code", "tempseg_base")
_CAP_PINS: tuple[str, ...] = ("captioner", "captioner_code")


def _build() -> dict[str, PresetSpec]:
    hf, nv_vllm, vllm = _engines()
    reward_kinds: tuple[ShardKind, ...] = ("reward.terminal", "reward.dense", "reward.rubric")
    reward_stage_kinds: dict[Stage, ShardKind] = {k: k for k in reward_kinds}  # type: ignore[misc]
    reward_engines = dict.fromkeys(reward_kinds, nv_vllm)
    reward_pins = dict.fromkeys(reward_kinds, _REWARD_PINS)
    exact = PresetSpec(
        name="exact",
        available=True,
        stage_kinds={"segment": "segcap", "caption": "segcap", **reward_stage_kinds},
        pools={"segcap": PoolSettings(gpu="H200", cpu_cores=8.0, memory_gib=96.0), **_REWARDS},
        engines={"segcap": hf, **reward_engines},
        pins={"segcap": _SEG_PINS + _CAP_PINS, **reward_pins},
        # OPT-2: NVIDIA's loader reads our BF16 Captioner (proven identical, captioner-bf16.md).
        derived_weights={"segcap": ("captioner_bf16",)},
        note="reference by construction: NVIDIA HF code on H200, NVIDIA's vLLM Rewarder",
    )
    balanced = PresetSpec(
        name="balanced",
        available=True,
        stage_kinds={"segment": "segment", "caption": "caption", **reward_stage_kinds},
        pools={
            "segment": PoolSettings(gpu="L40S", cpu_cores=6.0, memory_gib=48.0),
            "caption": PoolSettings(gpu="H100!", cpu_cores=8.0, memory_gib=64.0),
            **_REWARDS,
        },
        engines={"segment": vllm, "caption": vllm, **reward_engines},
        pins={"segment": _SEG_PINS, "caption": _CAP_PINS, **reward_pins},
        derived_weights={"segment": ("tempseg_merged",), "caption": ("captioner_bf16",)},
        note="vLLM 0.24.0, stage-separated pools; needs Gate certificates (config/certificates/)",
    )
    unavailable = {
        "budget": "planned for P2 (FP8 / B200 kernels; uncertified until validated)",
        "interactive": "planned for v0.4 (online API, AsyncLLM servers)",
    }
    out: dict[str, PresetSpec] = {"exact": exact, "balanced": balanced}
    for name, note in unavailable.items():
        out[name] = PresetSpec(name=name, available=False, stage_kinds={}, pools={}, engines={},
                               pins={}, derived_weights={}, note=note)  # type: ignore[arg-type]
    return out


PRESETS: dict[str, PresetSpec] = _build()
AVAILABLE_PRESETS: tuple[str, ...] = tuple(n for n, p in PRESETS.items() if p.available)


class PresetUnavailableError(ValueError):
    pass


def get_preset(name: str) -> PresetSpec:
    """The preset spec, or a clear error for unknown / not-yet-available presets."""
    spec = PRESETS.get(name)
    if spec is None:
        raise PresetUnavailableError(
            f"unknown preset {name!r}; choose one of {', '.join(AVAILABLE_PRESETS)}")
    if not spec.available:
        raise PresetUnavailableError(
            f"preset {name!r} is not available yet ({spec.note}); "
            f"use one of {', '.join(AVAILABLE_PRESETS)}")
    return spec


def shard_kinds_for(preset: PresetSpec, stages: tuple[Stage, ...] | list[Stage]) -> list[ShardKind]:
    """Shard kinds a pipeline needs, deduplicated, in first-use order."""
    out: list[ShardKind] = []
    for s in stages:
        k = preset.stage_kinds[s]
        if k not in out:
            out.append(k)
    return out


# ---------------------------------------------------------------------------------------------
# GPUs and certificates

# Modal GPU type -> usable memory per GPU in GiB (approximate, as the driver reports it).
GPU_MEMORY_GIB: dict[str, float] = {
    "T4": 15.0,
    "L4": 22.0,
    "A10": 22.0,
    "A10G": 22.0,
    "A100": 39.5,
    "A100-40GB": 39.5,
    "A100-80GB": 79.2,
    "L40S": 44.5,
    "H100": 79.6,
    "H100!": 79.6,
    "H200": 139.8,
    "B200": 178.0,
}

# Minimum GPU memory per container (GiB, summed over its GPUs), per (preset, shard kind).
# exact segcap: 80.95 GiB HBM high-water (docs/reference-smoke.md), above an H100's 79.6 GiB.
# Rewards: NVIDIA's vLLM recipe is sized for 80 GB GPUs (README, docs/cold-start.md).
_REWARD_MIN_GIB = 79.0
_MIN_GPU_GIB: dict[tuple[str, str], float] = {
    ("exact", "segcap"): 81.0,
    ("balanced", "segment"): 44.0,  # L40S class (§6.7)
    ("balanced", "caption"): 79.0,  # H100 class
}


def min_gpu_memory_gib(preset: str, kind: str) -> float:
    if kind.startswith("reward."):
        return _REWARD_MIN_GIB
    return _MIN_GPU_GIB[(preset, kind)]


def gpu_problem(preset: str, kind: str, gpu: str, gpu_count: int) -> str | None:
    """Why ``gpu_count`` x ``gpu`` cannot run ``kind`` under ``preset``, or None if it can."""
    mem = GPU_MEMORY_GIB.get(gpu)
    if mem is None:
        return f"unknown GPU {gpu!r}; use one of {', '.join(GPU_MEMORY_GIB)}"
    need = min_gpu_memory_gib(preset, kind)
    if mem * gpu_count < need:
        fits = [g for g, m in GPU_MEMORY_GIB.items() if m * gpu_count >= need]
        return (f"{kind} needs >= {need:g} GiB of GPU memory per container, but {gpu_count} x "
                f"{gpu} has {mem * gpu_count:g} GiB (use one of {', '.join(fits)})")
    return None


@dataclass(frozen=True)
class Certificate:
    """A Gate certificate (D16): shard kind ``kind`` on ``gpu`` under ``preset`` is faithful."""

    id: str
    preset: str
    kind: str
    gpu: str
    verdict: str = "certified"  # certified | provisional | fail (validate/certify.py)


def needs_certificate(preset: str) -> bool:
    """Everything but ``exact`` (the reference by construction) needs certificates."""
    return preset != "exact"


def find_certificates(preset: str, pools: Mapping[str, PoolSettings], *,
                      modes: Mapping[str, Mapping] | None = None,
                      **kwargs) -> dict[str, Certificate]:
    """The ``certified`` certificate matching each (shard kind, GPU, engine spec) of ``pools``,
    by shard kind (``validate.certkey.matching``; ``modes``: each kind's output-affecting run
    modes, ``config.hashing.kind_modes``). Provisional and failed certificates never count."""
    from flora_deploy.validate.certkey import matching

    out: dict[str, Certificate] = {}
    for kind, certs in matching(preset, pools, modes=modes, **kwargs).items():
        best = certs[0]
        if best["verdict"] == "certified":
            out[kind] = Certificate(best["id"], preset, kind, best["gpu"], best["verdict"])
    return out
