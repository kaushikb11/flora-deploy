"""Worker contract (PLAN.md §7/§8): what every GPU shard worker exposes to the orchestrator.

A worker class (Modal Cls, defined at module scope, decorated by the app factory) implements
``ShardWorker``. One call processes one shard idempotently (helpers in ``jobs.attempts`` /
``jobs.progress``; the shard itself comes from ``jobs.deps.StoreManifestSource.shard``, which
sees bisection children immediately):

Before step 0, every input must see current state: on a Volume store call
``store.refresh()`` (a warm container otherwise reads a stale ledger and stale parts; found by
the canary). Bucket stores need nothing.

0. if ``attempts.superseded(jf, kind, shard_id, token)``: return ``None`` at once, writing
   nothing (a duplicate or stale spawn; see ``jobs.spawner``);
1. write ``AttemptMarker`` (``attempts.write_attempt`` with ``token`` and the Modal call id);
2. read previous attempts' parts for this shard (``progress.shard_parts``) and skip episodes
   already present;
3. skip episodes whose upstream stage was poison-pilled (``attempts.upstream_poisoned``: e.g. a
   caption shard whose segment episode was isolated by D15): report them in
   ``episodes_failed`` with an ``EpisodeError(kind="poison_pill")`` in ``errors/``, never run them;
4. for each remaining episode: run the stage, append ``StageRecord`` JSON lines to a *local*
   part file; every N episodes (and on SIGINT via ``@modal.exit``) copy the part whole to the store;
   NVIDIA-semantics failures and infra errors go to ``JobPaths.errors`` (never crash the
   container);
5. on preemption (SIGINT / ``@modal.exit``), after flushing the part: ``attempts.mark_preempted``
   so this start does not count toward D15 quarantine; on an unexpected python exception:
   ``attempts.mark_exception``, then re-raise (Modal retries / the orchestrator respawns);
6. write ``ShardReceipt`` LAST, with ``parts`` = every part file holding this shard's records:
   ``shard_parts(...).parts`` (earlier containers) plus its own.
"""

from __future__ import annotations

from typing import Protocol

from flora_deploy.jobs.types import ShardReceipt


class ShardWorker(Protocol):
    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None:
        """``None`` only when the spawn was superseded (nothing written)."""
        ...
