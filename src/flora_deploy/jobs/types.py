"""Job-system contracts (PLAN.md §8, D6, D15). Owned by P1a-2/P1a-3; changes need review.

All job state is files in a ``Store`` (the user's bucket by default). Status is always *derived*
from these files; nothing here is held only in memory or in Modal Queue/Dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from flora_deploy.records import Stage

JobState = Literal[
    "indexed", "planned", "running", "composing", "done", "failed", "budget_stopped", "cancelled"
]

MAX_STARTS_BEFORE_QUARANTINE = 3  # D15


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    dataset: str
    pool: str  # human | yam | yam32 | dex | tianji (single-view robot uses its pool's rules)
    duration_s: float
    oracle_eligible: bool
    envelope_ok: bool
    index_ref: str  # store path of the EpisodeIndex receipt


@dataclass(frozen=True)
class Shard:
    shard_id: str  # stable: "<stage>-<seq>" or "<stage>-ep-<episode_id>" after bisection
    stage: Stage
    episodes: tuple[str, ...]  # episode_ids
    est_gpu_seconds: float
    depends_on: tuple[str, ...] = ()  # e.g. caption shard depends on its segment shard
    bisected_from: str | None = None


@dataclass(frozen=True)
class Manifest:
    job_id: str
    config_hash: str
    episodes: tuple[EpisodeRef, ...]
    shards: tuple[Shard, ...]
    created_utc: str


@dataclass(frozen=True)
class AttemptMarker:
    shard_id: str
    attempt: int  # 1-based start count
    container_id: str
    started_utc: str


@dataclass(frozen=True)
class ShardReceipt:
    """Written LAST by a worker; its existence means the shard is done."""

    shard_id: str
    config_hash: str
    attempt: int
    episodes_done: tuple[str, ...]
    episodes_failed: tuple[str, ...]  # recorded in errors/, not retried within this shard
    parts: tuple[str, ...]  # store paths of the part-files holding this shard's records
    container_id: str
    gpu: str | None
    wall_seconds: float
    finished_utc: str


@dataclass(frozen=True)
class ContainerReceipt:
    """Spend accounting over the container lifetime (D15/§8 budget)."""

    container_id: str
    app: str
    gpu: str | None
    gpu_count: int
    cpu_cores: float
    memory_gib: float
    started_utc: str
    heartbeat_utc: str
    stopped_utc: str | None = None


@dataclass(frozen=True)
class EpisodeError:
    episode_id: str
    stage: Stage
    shard_id: str
    attempt: int
    kind: str  # "nv_failure" (NVIDIA semantics) | "infra" | "poison_pill"
    message: str
    diagnostics: dict = field(default_factory=dict)
