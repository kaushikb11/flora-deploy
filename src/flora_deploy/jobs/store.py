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
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Protocol

from flora_deploy.jobs.types import safe_key


class Store(Protocol):
    def put_bytes(self, path: str, data: bytes) -> None: ...
    def get_bytes(self, path: str) -> bytes: ...
    def exists(self, path: str) -> bool: ...
    def list(self, prefix: str) -> Iterator[str]: ...


def prefetch(store: Store, paths: Iterable[str]) -> int:
    """Let ``store`` read ``paths`` ahead, in parallel, when it can (a ``ClientVolumeStore``: one
    network round trip per file otherwise); a no-op for every other store. Returns the number
    of files read ahead."""
    fn = getattr(store, "prefetch", None)
    return int(fn(paths)) if callable(fn) else 0


def prefetch_areas(store: Store, root: str, hashes: Iterable[str], areas: Iterable[str]) -> int:
    """:func:`prefetch` every file under ``<root>/<hash>/<area>/`` (a job's per-config and
    per-stage areas: ``calls``, ``receipts``, ``attempts``, ``parts``, ...)."""
    if not callable(getattr(store, "prefetch", None)):
        return 0
    paths: list[str] = []
    for h in dict.fromkeys(hashes):
        for area in areas:
            paths += store.list(f"{root}/{safe_key(h)}/{area}/")
    return prefetch(store, paths)


def snapshot(store: Store, prefix: str) -> int:
    """Let ``store`` list everything under directory ``prefix`` in one call, when it can (a
    ``ClientVolumeStore``: every later ``exists`` / ``list`` below it is then answered from that
    listing instead of one listing round trip per area); a no-op (0) for every other store."""
    fn = getattr(store, "snapshot", None)
    return int(fn(prefix)) if callable(fn) else 0


def put_many(store: Store, files: Mapping[str, bytes]) -> None:
    """Write several whole files: one upload and one commit when the store batches
    (``put_many``: a Volume through the client or a mounted Volume), else one ``put_bytes``
    each, in order. Use separate calls where the order between files matters."""
    fn = getattr(store, "put_many", None)
    if callable(fn):
        fn(dict(files))
        return
    for path, data in files.items():
        store.put_bytes(path, data)


def _fan(key: str) -> str:
    h = hashlib.sha256(key.encode()).hexdigest()
    return f"{h[:2]}/{h[2:4]}"


# Worker startup failures (``jobs.attempts.write_startup_failure``). A container whose
# ``@modal.enter`` fails has not received an input yet, so it knows its app, config and shard
# kinds but not the job: the markers live per deployed app, outside every job root, and the
# orchestrator matches them to its never-started calls by kind and time.
STARTUP_ROOT = "flora/.startup"


def startup_failures_prefix(app_name: str) -> str:
    return f"{STARTUP_ROOT}/{safe_key(app_name)}/"


def startup_failure_path(app_name: str, container_id: str) -> str:
    return f"{startup_failures_prefix(app_name)}{safe_key(container_id)}.json"


# Fake-models jobs (``flora run --fake-models``, the H0 rehearsal, and ``flora run --local
# --fake-models``) run under their own job id, ``<job>.fake`` (review 8b M5). The suffix is part
# of the job id itself, so every path under ``flora/<job>/`` (``manifest.json``, ``run.json``,
# ``job.json``, ``index/``, ``control/``, ``containers/``) and the ``orchestrator/<job>`` lease
# are separate from the real job's, with no flag to forget anywhere that builds a ``JobPaths``.
FAKE_JOB_SUFFIX = ".fake"


def is_fake_job_id(job_id: str) -> bool:
    """Whether ``job_id`` is in the fake-models namespace."""
    return job_id.endswith(FAKE_JOB_SUFFIX)


def fake_job_id(job_id: str) -> str:
    """The fake-models job id of ``job_id`` (idempotent: ``x.fake`` stays ``x.fake``)."""
    return job_id if is_fake_job_id(job_id) else f"{job_id}{FAKE_JOB_SUFFIX}"


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
