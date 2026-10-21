"""Attempts → quarantine → bisection (PLAN.md D15, §8).

Modal retries crashes, OOMs and preemptions forever inside one FunctionCall, so the orchestrator
counts **container starts** from files instead: every container that starts a shard writes one
``AttemptMarker`` named by its container id (``JobPaths.attempt``). Concurrent or Modal-retried
starts never overwrite each other, and the count is the number of distinct markers.

**Listing lag.** Markers are found by ``Store.list``, which may be eventually consistent. The
count is therefore a *lower bound* on the true number of starts: markers written in the last
"lag" window may be missing. Consequences, which the property tests check:

* quarantine never fires early (a marker is never counted that was not written);
* it fires once ``limit`` markers are visible, i.e. at most ``limit - 1 + s`` real starts, where
  ``s`` is the number of starts made during one listing-lag window (with one container per shard
  at a time and one start per tick, ``s <= lag``);
* ``known=`` lets a caller add container ids it knows about (e.g. a worker's own id, which a
  lagging listing may not show yet) for an exact count from that caller's view.

**Quarantine** writes a ``ControlRecord(kind="quarantine", target=shard_id)`` whose ``detail``
holds everything status needs: the config hash, the episodes already done with the part files
that hold them (compose reads those, since the parent never writes a receipt), the bisected
child shards, and any poison-pilled episodes. A shard with several unfinished episodes is
bisected into single-episode shards ``<kind>-ep-<uid>`` (``bisected_from`` = parent). A
single-episode shard that reaches the limit is a poison pill: an ``EpisodeError(kind=
"poison_pill")`` per stage, written whole-file to ``errors/``. Ids and error paths are
deterministic, so quarantining twice (e.g. because the first record was not listed yet) produces
the same children and the same error file: it is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field

from flora_deploy.jobs.manifest import from_json, stages_of, to_json
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import (
    AttemptMarker,
    ControlRecord,
    EpisodeError,
    Shard,
    ShardKind,
    safe_key,
)

QUARANTINE_CONTAINER = "quarantine"  # container-id slot of the poison-pill error file


@dataclass(frozen=True)
class JobFiles:
    """A store plus the job/config it is scoped to."""

    store: Store
    job_id: str
    config_hash: str
    paths: JobPaths = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", JobPaths(self.job_id))


def compact_utc(created_utc: str) -> str:
    """``2026-10-21T12:00:00.123Z`` → ``20261021T120000.123Z`` (path-safe)."""
    return safe_key(re.sub(r"[^0-9A-Za-z.]", "", created_utc) or "0")


# --- markers -------------------------------------------------------------------------------

def write_attempt(jf: JobFiles, kind: ShardKind, shard_id: str, container_id: str, *,
                  started_utc: str, orchestrator_attempt: int) -> AttemptMarker:
    """Called by the worker as the first thing it does for a shard (before any decode)."""
    m = AttemptMarker(shard_id=shard_id, container_id=container_id, started_utc=started_utc,
                      orchestrator_attempt=orchestrator_attempt)
    jf.store.put_bytes(jf.paths.attempt(jf.config_hash, kind, shard_id, container_id), to_json(m))
    return m


def attempt_containers(jf: JobFiles, kind: ShardKind, shard_id: str) -> set[str]:
    """Container ids with a visible marker (a lower bound under listing lag)."""
    prefix = jf.paths.attempt(jf.config_hash, kind, shard_id, "x")[:-len("x.json")]
    return {p[len(prefix):-len(".json")] for p in jf.store.list(prefix) if p.endswith(".json")}


def count_starts(jf: JobFiles, kind: ShardKind, shard_id: str, *,
                 known: Iterable[str] = ()) -> int:
    return len(attempt_containers(jf, kind, shard_id) | set(known))


def has_receipt(jf: JobFiles, shard: Shard) -> bool:
    return jf.store.exists(jf.paths.receipt(jf.config_hash, shard.kind, shard.shard_id))


def should_quarantine(jf: JobFiles, shard: Shard, *, limit: int,
                      known: Iterable[str] = ()) -> bool:
    """No receipt and at least ``limit`` visible starts. The receipt check uses ``exists``
    (read-after-write), so a finished shard is never quarantined because of listing lag."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if has_receipt(jf, shard):
        return False
    return count_starts(jf, shard.kind, shard.shard_id, known=known) >= limit


# --- bisection -----------------------------------------------------------------------------

def child_shard_id(kind: ShardKind, uid: str) -> str:
    sid = f"{kind}-ep-{uid}"
    if len(sid) > 200:  # safe_key cap; keep it deterministic
        sid = f"{kind}-ep-{hashlib.sha256(uid.encode()).hexdigest()[:40]}"
    return safe_key(sid)


def bisect(shard: Shard, *, done: Iterable[str] = (),
           durations: Mapping[str, float] | None = None) -> tuple[Shard, ...]:
    """Single-episode shards for ``shard``'s episodes not in ``done``, in shard order.

    Estimates split the parent's by duration (evenly without ``durations``). Children inherit
    ``depends_on``. A single-episode shard is not bisected (returns ``()``): it is a poison pill.
    """
    if len(shard.episodes) <= 1:
        return ()
    done = set(done)
    todo = [u for u in shard.episodes if u not in done]
    if durations:
        total = sum(durations[u] for u in shard.episodes) or 1.0
        est = {u: shard.est_gpu_seconds * durations[u] / total for u in todo}
    else:
        est = {u: shard.est_gpu_seconds / len(shard.episodes) for u in todo}
    return tuple(Shard(shard_id=child_shard_id(shard.kind, u), kind=shard.kind, episodes=(u,),
                       est_gpu_seconds=round(est[u], 6), depends_on=shard.depends_on,
                       bisected_from=shard.shard_id) for u in todo)


@dataclass(frozen=True)
class Quarantine:
    control_path: str
    record: ControlRecord
    children: tuple[Shard, ...]
    poisoned: tuple[str, ...]
    error_path: str | None


def quarantine(jf: JobFiles, shard: Shard, *, created_utc: str, done: Iterable[str] = (),
               done_parts: Iterable[str] = (), durations: Mapping[str, float] | None = None,
               starts: int | None = None, reason: str = "attempt limit reached",
               diagnostics: dict | None = None) -> Quarantine:
    """Quarantine ``shard``: bisect it or, if it holds one unfinished episode, poison-pill it.

    ``done`` / ``done_parts``: the episodes the shard's visible parts already hold, and those part
    paths (``progress.shard_parts`` gives both). The caller cancels the shard's outstanding
    FunctionCalls; this function only writes files.
    """
    done_t = tuple(u for u in shard.episodes if u in set(done))
    children = bisect(shard, done=done_t, durations=durations)
    poisoned: tuple[str, ...] = ()
    error_path = None
    if len(shard.episodes) == 1 and not done_t:
        poisoned = shard.episodes
        error_path = jf.paths.errors(jf.config_hash, shard.kind, shard.shard_id,
                                     QUARANTINE_CONTAINER, 0)
        errs = [EpisodeError(uid=shard.episodes[0], stage=stage, shard_id=shard.shard_id,
                             container_id=QUARANTINE_CONTAINER, kind="poison_pill",
                             message=f"{reason}: shard reached {starts} starts without a receipt",
                             diagnostics=dict(diagnostics or {}))
                for stage in stages_of(shard.kind)]
        jf.store.put_bytes(error_path, "".join(
            json.dumps(asdict(e), sort_keys=True) + "\n" for e in errs).encode())
    rec = ControlRecord(
        kind="quarantine", target=shard.shard_id, reason=reason, created_utc=created_utc,
        detail={"config_hash": jf.config_hash, "shard_kind": shard.kind, "starts": starts,
                "done": list(done_t), "done_parts": sorted(set(done_parts)),
                "children": [asdict(c) for c in children], "poisoned": list(poisoned),
                "error_path": error_path})
    path = jf.paths.control("quarantine", shard.shard_id, compact_utc(created_utc))
    jf.store.put_bytes(path, to_json(rec))
    return Quarantine(path, rec, children, poisoned, error_path)


def read_control(store: Store, path: str) -> ControlRecord:
    return from_json(ControlRecord, store.get_bytes(path))


def children_of(rec: ControlRecord) -> tuple[Shard, ...]:
    return tuple(from_json(Shard, json.dumps(c)) for c in rec.detail.get("children", ()))
