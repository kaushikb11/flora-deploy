"""Store contract + path layout (PLAN.md D6). Implementations: P1a-2.

Rules every implementation must honor:
- Whole-file writes only (write to a temp name, then publish atomically where the backend allows).
  No appends, no partial updates, no open handles across calls.
- Hash-prefixed directories for high-cardinality paths (Modal Volume v2 caps 262,144 files/dir).
- ``list`` returns full relative paths under a prefix; it may be eventually consistent, so callers
  treat receipts as the source of truth and tolerate seeing a part before its receipt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Protocol


class Store(Protocol):
    def put_bytes(self, path: str, data: bytes) -> None: ...
    def get_bytes(self, path: str) -> bytes: ...
    def exists(self, path: str) -> bool: ...
    def list(self, prefix: str) -> Iterator[str]: ...


def _fan(key: str) -> str:
    h = hashlib.sha256(key.encode()).hexdigest()
    return f"{h[:2]}/{h[2:4]}"


class JobPaths:
    """Relative paths of every job file under ``flora/<job_id>/``."""

    def __init__(self, job_id: str):
        self.root = f"flora/{job_id}"

    @property
    def manifest(self) -> str:
        return f"{self.root}/manifest.json"

    @property
    def job_cache(self) -> str:
        return f"{self.root}/job.json"  # cache only; status is derived from receipts

    def index(self, episode_id: str) -> str:
        return f"{self.root}/index/{_fan(episode_id)}/{episode_id}.json"

    def part(self, stage: str, cfg_hash: str, shard_id: str, attempt: int, k: int) -> str:
        name = f"{shard_id}-{attempt}-{k}.jsonl"
        return f"{self.root}/parts/{stage}/{cfg_hash}/{_fan(shard_id)}/{name}"

    def receipt(self, stage: str, shard_id: str) -> str:
        return f"{self.root}/receipts/{stage}/{_fan(shard_id)}/{shard_id}.json"

    def attempt(self, shard_id: str, n: int) -> str:
        return f"{self.root}/attempts/{_fan(shard_id)}/{shard_id}/{n}.json"

    def error(self, stage: str, shard_id: str) -> str:
        return f"{self.root}/errors/{stage}/{_fan(shard_id)}/{shard_id}.jsonl"

    def container(self, container_id: str) -> str:
        return f"{self.root}/containers/{_fan(container_id)}/{container_id}.json"

    def audit(self, stage: str, shard_id: str) -> str:
        return f"{self.root}/audit/{stage}/{_fan(shard_id)}/{shard_id}.jsonl"

    def output(self, stage: str, fmt: str = "parquet") -> str:
        return f"{self.root}/output/{stage}.{fmt}"
