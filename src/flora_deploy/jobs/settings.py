"""``RunSettings``: the narrow, validated view of ``flora.yaml`` the job system consumes.

P1a-1 (config) maps the user's YAML onto this; the orchestrator, planner, budget and workers only
ever see ``RunSettings``. Keeping it small stops the config schema and the job system from
drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from flora_deploy.jobs.types import MAX_STARTS_BEFORE_QUARANTINE, ShardKind

Preset = Literal["exact", "balanced", "budget", "interactive"]
EnvelopePolicy = Literal["warn", "enforce", "ignore"]  # PLAN.md D13


@dataclass(frozen=True)
class PoolSettings:
    """Per shard kind: which GPU, how many containers, inputs per container."""

    gpu: str  # Modal GPU string, e.g. "H200", "H100!", "L40S"
    gpu_count: int = 1
    max_containers: int = 1
    max_inputs: int = 1  # @modal.concurrent inputs per container
    cpu_cores: float = 8.0  # physical cores (billed whether used or not)
    memory_gib: float = 64.0


@dataclass(frozen=True)
class RunSettings:
    job_id: str
    preset: Preset
    config_hash: str
    budget_usd: float
    max_gpus: int  # workspace GPU cap the whole job must stay under
    pools: dict[ShardKind, PoolSettings]
    shard_target_gpu_minutes: float = 10.0
    max_starts_before_quarantine: int = MAX_STARTS_BEFORE_QUARANTINE
    envelope: EnvelopePolicy = "warn"
    environment_budget_factor: float = 1.2  # native Modal Environment budget backstop
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if self.max_gpus < 1:
            raise ValueError("max_gpus must be >= 1")
        need = sum(p.gpu_count * p.max_containers for p in self.pools.values())
        if need > self.max_gpus:
            raise ValueError(
                f"pools can use {need} GPUs at once but max_gpus is {self.max_gpus}; "
                "lower max_containers or raise max_gpus"
            )
