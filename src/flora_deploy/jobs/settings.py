"""``RunSettings``: the narrow, validated view of ``flora.yaml`` the job system consumes.

P1a-1 (config) maps the user's YAML onto this; the orchestrator, planner, budget and workers only
ever see ``RunSettings``. Keeping it small stops the config schema and the job system from
drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from flora_deploy.jobs.types import MAX_STARTS_BEFORE_QUARANTINE, ShardKind

Preset = Literal["exact", "balanced", "budget", "interactive"]
EnvelopePolicy = Literal["warn", "enforce", "ignore"]  # PLAN.md D13

# Modal ``scaledown_window`` of every deployed worker class (``modal_app.factory.WorkerRef``): the
# budget charges this much idle time each time a worker container goes away.
WORKER_SCALEDOWN_S = 300
# Job-level error-rate guard: more excluded episodes than this (per shard kind) and a job that
# resolved every shard ends ``completed_with_errors`` instead of ``done``.
MAX_EPISODE_ERROR_RATE = 0.10


@dataclass(frozen=True)
class PoolSettings:
    """Per shard kind: which GPU, how many containers, inputs per container."""

    gpu: str  # Modal GPU string, e.g. "H200", "H100!", "L40S"
    gpu_count: int = 1
    max_containers: int = 1
    max_inputs: int = 1  # @modal.concurrent inputs per container
    cpu_cores: float = 8.0  # physical cores (billed whether used or not)
    memory_gib: float = 64.0
    # Episodes in flight inside ONE input, for kinds whose worker batches in-process
    # (:data:`IN_WORKER_CONCURRENCY`); None: that kind's measured default.
    episode_concurrency: int | None = None


# The --fake-models worker containers (``modal_app.factory.FAKE_WORKER_CPU`` /
# ``FAKE_WORKER_MEMORY_GIB``; tests keep them equal): no GPU.
FAKE_WORKER_CPU = 2.0
FAKE_WORKER_MEMORY_GIB = 12.0


def billed_pool(settings: RunSettings, kind: str) -> PoolSettings:
    """``kind``'s pool as the containers that run it are billed: the configured pool, or for a
    ``fake_models`` job a GPU-less CPU container (:data:`FAKE_WORKER_CPU`)."""
    pool = settings.pools[kind]  # type: ignore[index]
    if not getattr(settings, "fake_models", False):
        return pool
    return replace(pool, gpu="", gpu_count=0, cpu_cores=FAKE_WORKER_CPU,
                   memory_gib=FAKE_WORKER_MEMORY_GIB)


# Kinds whose worker runs one shard per container (one Modal input: ``max_inputs`` must be 1) and
# runs that shard's episodes concurrently in-process, and the measured default (review 6 H2: the
# exact segcap rate "x4" is four episodes on one H200 under one GPU lock, rates.json).
IN_WORKER_CONCURRENCY: dict[str, int] = {"segcap": 4}
# How the factory hands it to such a worker (``modal_app.reference_app.episode_concurrency``).
EPISODE_CONCURRENCY_ENV = "FLORA_EPISODE_CONCURRENCY"


def episode_concurrency(kind: str, pool: PoolSettings) -> int:
    """Episodes one container of ``kind`` runs at once: the concurrency its rates are measured at
    (``planner.rates`` ``concurrency``). In-worker kinds: ``episode_concurrency`` (default
    :data:`IN_WORKER_CONCURRENCY`); every other kind: one episode per Modal input, so
    ``max_inputs``."""
    if kind in IN_WORKER_CONCURRENCY:
        return int(pool.episode_concurrency or IN_WORKER_CONCURRENCY[kind])
    return max(int(pool.max_inputs), 1)


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
    # shard kind -> stage hash (``config.hashing``); the manifest records it and shard files are
    # keyed by it (``jobs.store.JobPaths``). Empty: everything is keyed by ``config_hash``.
    stage_hashes: dict[str, str] = field(default_factory=dict)
    region: str | None = None  # ``modal.region``: pinning a region multiplies every price
    scaledown_s: float = WORKER_SCALEDOWN_S  # deployed workers' ``scaledown_window``
    min_containers: int = 0  # warm containers per worker class (``flora deploy --min-containers``)
    max_episode_error_rate: float = MAX_EPISODE_ERROR_RATE
    # Hashed, output-affecting run modes (``config.schema.output_settings``): the exact segcap
    # worker's GPU determinism mode and the reward.terminal mode. Workers apply exactly these.
    determinism: str = "nvidia"
    terminal_mode: str = "terminal"  # or "terminal_rationale"
    # ``output.formats``: what compose writes (not hashed: outputs are rebuilt from the parts)
    output_formats: tuple[str, ...] = ("jsonl", "parquet")
    # A ``--fake-models`` job (H0): every kind runs on a CPU container of FAKE_WORKER_CPU cores
    # and FAKE_WORKER_MEMORY_GIB, so spend is priced at that (``billed_pool``), not the pool GPU.
    fake_models: bool = False

    def __post_init__(self) -> None:
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if not 0.0 <= self.max_episode_error_rate <= 1.0:
            raise ValueError("max_episode_error_rate must be in [0, 1]")
        if self.scaledown_s < 0 or self.min_containers < 0:
            raise ValueError("scaledown_s and min_containers must be >= 0")
        if self.max_gpus < 1:
            raise ValueError("max_gpus must be >= 1")
        need = sum(p.gpu_count * p.max_containers for p in self.pools.values())
        if need > self.max_gpus:
            raise ValueError(
                f"pools can use {need} GPUs at once but max_gpus is {self.max_gpus}; "
                "lower max_containers or raise max_gpus"
            )
