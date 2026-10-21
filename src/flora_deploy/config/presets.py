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

Memory values are the pre-P0 defaults; P0 measurements replace them (§6.7).
"""

from __future__ import annotations

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

    hf = EngineIdentity(
        name="nvidia-hf",
        version={k: RUNTIME[k] for k in ("transformers", "torch_reference", "peft",
                                         "qwen_vl_utils", "decord")},
        image="img_hfref",
    )
    nv_vllm = EngineIdentity(name="nvidia-vllm-offline", version={"vllm": RUNTIME["vllm"]},
                             image="img_vllm")
    vllm = EngineIdentity(name="vllm-offline", version={"vllm": RUNTIME["vllm"]},
                          image="img_vllm")
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
        note="certified faithful: vLLM 0.24.0, stage-separated pools",
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
