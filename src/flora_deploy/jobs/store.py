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
from collections.abc import Iterator
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

    Everything a config change invalidates (parts, receipts, attempts, errors, audits) lives under
    ``<cfg_hash>/``, so outputs of different configs never mix and resume is per config.
    """

    def __init__(self, job_id: str):
        self.root = f"flora/{safe_key(job_id)}"

    # job-level
    @property
    def manifest(self) -> str:
        return f"{self.root}/manifest.json"

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
        return (f"{self.root}/{safe_key(cfg_hash)}/{area}/{safe_key(kind)}/{_fan(shard_id)}/"
                f"{safe_key(shard_id)}")

    def part(self, cfg_hash: str, kind: str, shard_id: str, container_id: str, k: int) -> str:
        return f"{self._cfg(cfg_hash, 'parts', kind, shard_id)}/{safe_key(container_id)}-{k}.jsonl"

    def receipt(self, cfg_hash: str, kind: str, shard_id: str) -> str:
        return f"{self._cfg(cfg_hash, 'receipts', kind, shard_id)}.json"

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
