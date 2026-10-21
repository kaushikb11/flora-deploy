"""Worker contract (PLAN.md §7/§8): what every GPU shard worker exposes to the orchestrator.

A worker class (Modal Cls, defined at module scope, decorated by the app factory) implements
``ShardWorker``. One call processes one shard idempotently:

1. write ``AttemptMarker`` (``JobPaths.attempt``);
2. read previous attempts' parts for this shard and skip episodes already present;
3. for each remaining episode: run the stage, append ``StageRecord`` JSON lines to a *local*
   part file; every N episodes (and on SIGINT via ``@modal.exit``) copy the part whole to the store;
   NVIDIA-semantics failures and infra errors go to ``JobPaths.error`` (never crash the container);
4. write ``ShardReceipt`` LAST.
"""

from __future__ import annotations

from typing import Protocol

from flora_deploy.jobs.types import ShardReceipt


class ShardWorker(Protocol):
    def run_shard(self, job_id: str, shard_id: str, config_hash: str) -> ShardReceipt: ...
