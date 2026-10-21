"""Job-system contracts (PLAN.md §8, D6, D15). Owned by P1a-2/P1a-3; changes need review.

All job state is files in a ``Store`` (the user's bucket by default). Status is always *derived*
from these files; nothing here is held only in memory or in Modal Queue/Dict.

Identity: episode ids from datasets are NOT unique across datasets (every LeRobot dataset has
``episode_000001``), so every job-level key uses ``episode_uid(dataset, episode_id)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from flora_deploy.records import Stage

JobState = Literal[
    "indexed", "planned", "running", "composing", "done", "failed", "budget_stopped", "cancelled"
]

# A shard's unit of work. ``segcap`` is the exact preset's combined segment+caption worker
# (NVIDIA's pipeline runs both stages per episode); it emits records for both stages.
ShardKind = Literal["segment", "caption", "segcap", "reward.terminal", "reward.dense",
                    "reward.rubric"]

MAX_STARTS_BEFORE_QUARANTINE = 3  # D15

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def safe_key(value: str) -> str:
    """Validate a path component (shard ids, uids, container ids). Raises ValueError."""
    if not _SAFE.match(value) or ".." in value:
        raise ValueError(f"unsafe key {value!r}: use [A-Za-z0-9._-], max 200 chars")
    return value


def episode_uid(dataset: str, episode_id: str) -> str:
    """Job-unique, path-safe episode key: ``<dataset slug>--<episode id>``.

    ``dataset`` is e.g. an HF repo id (``lerobot/aloha_static_coffee``) or a local dir name.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset).strip("-.") or "dataset"
    return safe_key(f"{slug}--{re.sub(r'[^A-Za-z0-9._-]+', '-', episode_id)}")


@dataclass(frozen=True)
class EpisodeRef:
    uid: str  # episode_uid(dataset, episode_id); the key used everywhere in a job
    dataset: str
    episode_id: str  # as the source dataset names it
    pool: str  # human | yam | yam32 | dex | tianji
    duration_s: float
    oracle_eligible: bool
    envelope_ok: bool
    index_ref: str  # store path of the EpisodeIndex receipt


@dataclass(frozen=True)
class Shard:
    shard_id: str  # path-safe: "<kind>-<seq>" or "<kind>-ep-<uid>" after bisection
    kind: ShardKind
    episodes: tuple[str, ...]  # episode uids
    est_gpu_seconds: float
    depends_on: tuple[str, ...] = ()  # e.g. a caption shard depends on its segment shard
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
    """One file per container start of a shard, named by container id, so concurrent or
    Modal-retried starts never overwrite each other. Starts = number of distinct markers."""

    shard_id: str
    container_id: str  # unique per container (MODAL_TASK_ID); part of the file name
    started_utc: str
    orchestrator_attempt: int  # the spawn number the orchestrator assigned (single writer)


@dataclass(frozen=True)
class ShardReceipt:
    """Written LAST by a worker; its existence means the shard is done."""

    shard_id: str
    kind: ShardKind
    config_hash: str
    container_id: str
    episodes_done: tuple[str, ...]  # uids
    episodes_failed: tuple[str, ...]  # uids with EpisodeError files; not retried in this shard
    parts: tuple[str, ...]  # store paths of the part-files holding this shard's records
    gpu: str | None
    wall_seconds: float
    finished_utc: str


@dataclass(frozen=True)
class ContainerReceipt:
    """Spend accounting over the container lifetime (§8 budget)."""

    container_id: str
    app: str
    gpu: str | None
    gpu_count: int
    cpu_request: float  # physical cores
    cpu_limit: float | None
    memory_request_gib: float
    memory_limit_gib: float | None
    region: str | None
    preemptible: bool
    started_utc: str
    heartbeat_utc: str
    stopped_utc: str | None = None


@dataclass(frozen=True)
class EpisodeError:
    uid: str
    stage: Stage
    shard_id: str
    container_id: str
    kind: Literal["nv_failure", "infra", "poison_pill"]  # nv_failure = NVIDIA's own semantics
    message: str
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ControlRecord:
    """Operator/orchestrator decisions that status is derived from (no in-memory state)."""

    kind: Literal["quarantine", "cancel", "budget_stop", "resume"]
    target: str  # shard_id for quarantine; job_id otherwise
    reason: str
    created_utc: str
    detail: dict = field(default_factory=dict)
