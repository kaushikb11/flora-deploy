"""Store contract + path layout (PLAN.md D6). Implementations: P1a-2.

Rules every implementation must honor:
- Whole-file writes only (write to a temp name, then publish atomically where the backend allows).
  No appends, no partial updates, no open handles across calls. Multi-record files (parts,
  errors, audits) are written once per (shard, container, k) and never extended.
- Hash-prefixed directories for high-cardinality paths (Modal Volume v2 caps 262,144 files/dir).
- ``list`` returns full relative paths under a prefix; it may be eventually consistent, so callers
  treat receipts as the source of truth and tolerate seeing a part before its receipt.
- Every path component that comes from data goes through ``safe_key``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from typing import Protocol

from flora_deploy.jobs.types import safe_key


class Store(Protocol):
    def put_bytes(self, path: str, data: bytes) -> None: ...
    def get_bytes(self, path: str) -> bytes: ...
    def exists(self, path: str) -> bool: ...
    def list(self, prefix: str) -> Iterator[str]: ...


def _fan(key: str) -> str:
    h = hashlib.sha256(key.encode()).hexdigest()
    return f"{h[:2]}/{h[2:4]}"


class JobPaths:
    """Relative paths of every job file under ``flora/<job_id>/``.

    Two scopes below the job root (review-4 M1/M6):

    * **per config** (``<job hash>/``): the manifest (``config_manifest``), the orchestrator's
      per-shard ledgers (``calls/``) and the composed outputs (``output/``): what one config's
      orchestrator decided and composed;
    * **per stage** (``<stage hash>/``): everything a shard *produces* (parts, receipts, attempts,
      errors, audits, replans), keyed by the stage hash of the shard's kind
      (``config.hashing.stage_hashes``), so adding ``reward.dense`` to a config reuses its
      finished ``segcap`` shards instead of redoing them.

    Callers keep passing the **job** hash to the per-shard path methods; when this object knows
    the stage hashes (``JobPaths(job_id, config_hash=..., stage_hashes=...)``, which
    ``attempts.JobFiles`` builds from the manifest) a per-stage area is re-keyed to the stage hash
    of ``kind``. Without stage hashes (old manifests, tests) everything stays under the job hash.
    Any other hash passed explicitly is used as given.
    """

    PER_CONFIG_AREAS = frozenset({"calls"})

    def __init__(self, job_id: str, *, config_hash: str | None = None,
                 stage_hashes: Mapping[str, str] | Callable[[], Mapping[str, str]] | None = None):
        self.root = f"flora/{safe_key(job_id)}"
        self.config_hash = config_hash
        self._stage_hashes = stage_hashes

    def stage_key(self, cfg_hash: str, kind: str) -> str:
        """The directory key of ``kind``'s shard files when the caller passes ``cfg_hash``."""
        if self.config_hash is None or cfg_hash != self.config_hash or not self._stage_hashes:
            return cfg_hash
        sh = self._stage_hashes() if callable(self._stage_hashes) else self._stage_hashes
        return sh.get(kind) or cfg_hash

    # job-level
    @property
    def manifest(self) -> str:
        """The job's *latest planned* manifest (a convenience pointer: ``flora compose`` /
        ``flora results`` default to its config). The authority is :meth:`config_manifest`."""
        return f"{self.root}/manifest.json"

    def config_manifest(self, cfg_hash: str) -> str:
        """The manifest of one config (M6: re-planning config B never clobbers config A's)."""
        return f"{self.root}/{safe_key(cfg_hash)}/manifest.json"

    @property
    def job_cache(self) -> str:
        return f"{self.root}/job.json"  # cache only; status is derived from receipts

    def index(self, uid: str) -> str:
        return f"{self.root}/index/{_fan(uid)}/{safe_key(uid)}.json"

    def control(self, kind: str, target: str, created_utc_compact: str) -> str:
        return (f"{self.root}/control/{safe_key(kind)}/{_fan(target)}/"
                f"{safe_key(target)}-{safe_key(created_utc_compact)}.json")

    def container(self, container_id: str) -> str:
        return f"{self.root}/containers/{_fan(container_id)}/{safe_key(container_id)}.json"

    # per config
    def _cfg(self, cfg_hash: str, area: str, kind: str, shard_id: str) -> str:
        key = cfg_hash if area in self.PER_CONFIG_AREAS else self.stage_key(cfg_hash, kind)
        return (f"{self.root}/{safe_key(key)}/{area}/{safe_key(kind)}/{_fan(shard_id)}/"
                f"{safe_key(shard_id)}")

    def area_prefix(self, cfg_hash: str, area: str, kind: str) -> str:
        """Listing prefix of one area of one shard kind (``.../<key>/<area>/<kind>/``)."""
        key = cfg_hash if area in self.PER_CONFIG_AREAS else self.stage_key(cfg_hash, kind)
        return f"{self.root}/{safe_key(key)}/{area}/{safe_key(kind)}/"

    def part(self, cfg_hash: str, kind: str, shard_id: str, container_id: str, k: int) -> str:
        return f"{self._cfg(cfg_hash, 'parts', kind, shard_id)}/{safe_key(container_id)}-{k}.jsonl"

    def receipt(self, cfg_hash: str, kind: str, shard_id: str) -> str:
        return f"{self._cfg(cfg_hash, 'receipts', kind, shard_id)}.json"

    def replan(self, cfg_hash: str, kind: str, shard_id: str) -> str:
        """Durable quarantine decision of one shard (children, done, poisoned). Written once,
        *before* the quarantine ControlRecord, and read with ``exists``/``get``, never ``list``."""
        return f"{self._cfg(cfg_hash, 'replans', kind, shard_id)}.json"

    def call(self, cfg_hash: str, kind: str, shard_id: str) -> str:
        """The orchestrator's per-shard ledger (``orchestrator.ShardLedger``); workers read it to
        detect a superseded spawn (``attempts.superseded``)."""
        return f"{self._cfg(cfg_hash, 'calls', kind, shard_id)}/call.json"

    def attempt(self, cfg_hash: str, kind: str, shard_id: str, container_id: str) -> str:
        return f"{self._cfg(cfg_hash, 'attempts', kind, shard_id)}/{safe_key(container_id)}.json"

    def errors(self, cfg_hash: str, kind: str, shard_id: str, container_id: str, k: int) -> str:
        return (f"{self._cfg(cfg_hash, 'errors', kind, shard_id)}/"
                f"{safe_key(container_id)}-{k}.jsonl")

    def audit(self, cfg_hash: str, kind: str, shard_id: str, container_id: str, k: int) -> str:
        return (f"{self._cfg(cfg_hash, 'audit', kind, shard_id)}/"
                f"{safe_key(container_id)}-{k}.jsonl")

    def output(self, cfg_hash: str, stage: str, fmt: str = "parquet") -> str:
        return f"{self.root}/{safe_key(cfg_hash)}/output/{safe_key(stage)}.{safe_key(fmt)}"
