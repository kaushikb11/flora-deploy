"""Job progress derived purely from store files (PLAN.md §8): no in-memory or Modal Dict state.

Inputs: the manifest, receipts (``exists``/``get``: read-after-write), attempt markers, parts,
quarantine ``ControlRecord``s and error files (``list``: possibly lagging).

**Effective shards** = manifest shards + the children recorded in quarantine records (bisection),
transitively.

**Per-shard state** (first match wins):

* ``done``: its receipt exists (even if it was also quarantined; a late receipt from a container
  that was still running is real output);
* ``quarantined``: a quarantine record is visible (its children carry the work, or it poisoned
  its single episode);
* ``quarantine_due``: ``limit`` attempt markers are visible and no receipt: the orchestrator
  should call ``attempts.quarantine`` (idempotent, see there);
* ``blocked``: some ``depends_on`` shard is not resolved, where resolved means every one of its
  episodes is terminal for that shard's kind (done, failed or poison-pilled), which also covers a
  dependency that was quarantined and bisected;
* ``ready`` (no visible start) / ``started``.

**Per-episode status**, per shard kind: ``done`` (in a receipt's ``episodes_done`` or a
quarantine record's ``done``), ``failed`` (in a receipt's ``episodes_failed``: NVIDIA's own
failure, terminal, not retried), ``poison_pill`` (quarantine record), else ``pending``.
``done`` takes precedence. Resume set = manifest − done − failed − poison-pilled, per kind.

**Listing lag** can hide a quarantine record for a while; the parent then shows
``quarantine_due`` again and re-quarantining is idempotent. Because ``complete`` needs every
effective shard resolved and each quarantine record lists the parts that hold its done episodes,
a ``complete`` view always has every expected record reachable (``compose_input().missing`` is
empty), which the property tests check.

**Compose input** = the parts named by receipts plus the ``done_parts`` of quarantine records,
deduplicated by ``records.record_key``. Records whose ``provenance.config_hash`` differs from the
job's are dropped (counted as ``foreign``). Among duplicates (same key, same config hash) the
*latest* source wins: sources are ordered by (receipt ``finished_utc`` or quarantine
``created_utc``, container id, part path), and a later one replaces an earlier one. Within one
config hash outputs should be identical (``exact`` is deterministic), so a duplicate whose JSON
differs is reported in ``conflicts`` instead of being hidden.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal

from flora_deploy.jobs.attempts import JobFiles, children_of, count_starts, read_control
from flora_deploy.jobs.manifest import from_json, stages_of, to_json
from flora_deploy.jobs.types import (
    ControlRecord,
    EpisodeError,
    Manifest,
    Shard,
    ShardKind,
    ShardReceipt,
)
from flora_deploy.records import StageRecord, record_key

ShardState = Literal["done", "quarantined", "quarantine_due", "blocked", "ready", "started"]
EpisodeState = Literal["done", "failed", "poison_pill", "pending"]
TERMINAL: frozenset[str] = frozenset({"done", "failed", "poison_pill"})


# --- worker-side helpers -------------------------------------------------------------------

def write_part(jf: JobFiles, kind: ShardKind, shard_id: str, container_id: str, k: int,
               records: Iterable[StageRecord]) -> str:
    """One whole JSONL part file per (shard, container, k); never extended."""
    path = jf.paths.part(jf.config_hash, kind, shard_id, container_id, k)
    jf.store.put_bytes(path, "".join(r.to_json() + "\n" for r in records).encode())
    return path


def read_part(jf: JobFiles, path: str) -> list[StageRecord]:
    return [StageRecord.from_json(line)
            for line in jf.store.get_bytes(path).decode().splitlines() if line.strip()]


def write_errors(jf: JobFiles, kind: ShardKind, shard_id: str, container_id: str, k: int,
                 errors: Iterable[EpisodeError]) -> str:
    path = jf.paths.errors(jf.config_hash, kind, shard_id, container_id, k)
    jf.store.put_bytes(path, "".join(
        json.dumps(asdict(e), sort_keys=True) + "\n" for e in errors).encode())
    return path


def read_errors(jf: JobFiles, path: str) -> list[EpisodeError]:
    return [from_json(EpisodeError, line)
            for line in jf.store.get_bytes(path).decode().splitlines() if line.strip()]


def write_receipt(jf: JobFiles, receipt: ShardReceipt) -> str:
    """Written LAST by the worker."""
    path = jf.paths.receipt(jf.config_hash, receipt.kind, receipt.shard_id)
    jf.store.put_bytes(path, to_json(receipt))
    return path


def read_receipt(jf: JobFiles, kind: ShardKind, shard_id: str) -> ShardReceipt | None:
    path = jf.paths.receipt(jf.config_hash, kind, shard_id)
    if not jf.store.exists(path):
        return None
    return from_json(ShardReceipt, jf.store.get_bytes(path))


def _dir_prefix(path_of_x: str, tail: str) -> str:
    assert path_of_x.endswith(tail)
    return path_of_x[: -len(tail)]


@dataclass(frozen=True)
class ShardParts:
    parts: tuple[str, ...]  # visible part paths, sorted
    done: tuple[str, ...]  # uids with a record for every stage of the shard's kind, shard order


def shard_parts(jf: JobFiles, shard: Shard) -> ShardParts:
    """What earlier containers of ``shard`` already wrote (a worker skips these episodes on
    restart). Lagging listings can hide recent parts: the worker then redoes those episodes and
    compose dedupes."""
    prefix = _dir_prefix(jf.paths.part(jf.config_hash, shard.kind, shard.shard_id, "x", 0),
                         "x-0.jsonl")
    parts = tuple(p for p in jf.store.list(prefix) if p.endswith(".jsonl"))
    have: dict[str, set[str]] = defaultdict(set)
    for p in parts:
        for r in read_part(jf, p):
            if r.provenance.config_hash == jf.config_hash:
                have[r.uid].add(r.stage)
    need = set(stages_of(shard.kind))
    return ShardParts(parts, tuple(u for u in shard.episodes if need <= have.get(u, set())))


# --- status --------------------------------------------------------------------------------

@dataclass
class ShardStatus:
    shard: Shard
    state: ShardState
    starts: int  # visible attempt markers (lower bound)
    receipt: ShardReceipt | None = None
    quarantines: tuple[ControlRecord, ...] = ()


@dataclass
class Progress:
    manifest: Manifest
    shards: dict[str, ShardStatus]
    episodes: dict[tuple[str, ShardKind], EpisodeState]
    quarantines: dict[str, tuple[ControlRecord, ...]] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return all(s.state in ("done", "quarantined") for s in self.shards.values())

    def runnable(self) -> list[Shard]:
        return [s.shard for s in self.shards.values() if s.state in ("ready", "started")]

    def due_for_quarantine(self) -> list[Shard]:
        return [s.shard for s in self.shards.values() if s.state == "quarantine_due"]

    def resume_set(self) -> dict[ShardKind, set[str]]:
        out: dict[ShardKind, set[str]] = defaultdict(set)
        for (uid, kind), st in self.episodes.items():
            if st not in TERMINAL:
                out[kind].add(uid)
        return dict(out)

    def poisoned(self) -> set[tuple[str, ShardKind]]:
        return {k for k, st in self.episodes.items() if st == "poison_pill"}


def quarantine_records(jf: JobFiles) -> dict[str, tuple[ControlRecord, ...]]:
    prefix = f"{jf.paths.root}/control/quarantine/"
    out: dict[str, list[ControlRecord]] = defaultdict(list)
    for p in jf.store.list(prefix):
        rec = read_control(jf.store, p)
        if rec.kind == "quarantine" and rec.detail.get("config_hash") == jf.config_hash:
            out[rec.target].append(rec)
    return {k: tuple(sorted(v, key=lambda r: r.created_utc)) for k, v in out.items()}


def effective_shards(manifest: Manifest, quarantines: dict[str, tuple[ControlRecord, ...]]
                     ) -> dict[str, Shard]:
    shards = {s.shard_id: s for s in manifest.shards}
    todo = list(shards)
    while todo:
        sid = todo.pop()
        for rec in quarantines.get(sid, ()):
            for child in children_of(rec):
                if child.shard_id not in shards:
                    shards[child.shard_id] = child
                    todo.append(child.shard_id)
    return shards


def derive(jf: JobFiles, manifest: Manifest, *, limit: int) -> Progress:
    """Rebuild the full job status from files. Stateless: safe to call from any process."""
    if manifest.config_hash != jf.config_hash:
        raise ValueError("manifest and JobFiles disagree on config_hash")
    qs = quarantine_records(jf)
    shards = effective_shards(manifest, qs)

    episodes: dict[tuple[str, ShardKind], EpisodeState] = {}
    for s in shards.values():
        for u in s.episodes:
            episodes.setdefault((u, s.kind), "pending")

    def mark(uid: str, kind: ShardKind, st: EpisodeState) -> None:
        cur = episodes.get((uid, kind), "pending")
        rank = {"pending": 0, "poison_pill": 1, "failed": 2, "done": 3}
        if rank[st] > rank[cur]:
            episodes[(uid, kind)] = st

    receipts: dict[str, ShardReceipt | None] = {}
    for sid, s in shards.items():
        r = receipts[sid] = read_receipt(jf, s.kind, sid)
        if r is not None:
            for u in r.episodes_done:
                mark(u, s.kind, "done")
            for u in r.episodes_failed:
                mark(u, s.kind, "failed")
        for rec in qs.get(sid, ()):
            for u in rec.detail.get("done", ()):
                mark(u, s.kind, "done")
            for u in rec.detail.get("poisoned", ()):
                mark(u, s.kind, "poison_pill")

    def resolved(sid: str) -> bool:
        dep = shards.get(sid)
        return dep is not None and all(episodes.get((u, dep.kind)) in TERMINAL
                                       for u in dep.episodes)

    statuses: dict[str, ShardStatus] = {}
    for sid, s in shards.items():
        starts = count_starts(jf, s.kind, sid)
        r = receipts[sid]
        q = qs.get(sid, ())
        if r is not None:
            st: ShardState = "done"
        elif q:
            st = "quarantined"
        elif starts >= limit:
            st = "quarantine_due"
        elif not all(resolved(d) for d in s.depends_on):
            st = "blocked"
        else:
            st = "ready" if starts == 0 else "started"
        statuses[sid] = ShardStatus(s, st, starts, r, q)
    return Progress(manifest, statuses, episodes, qs)


# --- compose input -------------------------------------------------------------------------

@dataclass
class ComposeInput:
    records: dict[tuple[str, str], StageRecord]
    expected: set[tuple[str, str]]
    excused: set[tuple[str, str]]  # failed or poison-pilled (uid, stage)
    duplicates: int = 0  # extra copies seen (identical or not)
    conflicts: list[tuple[str, str]] = field(default_factory=list)  # differing duplicates
    foreign: int = 0  # records with another config hash, dropped

    @property
    def missing(self) -> set[tuple[str, str]]:
        return self.expected - self.records.keys() - self.excused


def compose_input(jf: JobFiles, progress: Progress) -> ComposeInput:
    sources: list[tuple[str, str, str]] = []  # (time, container, part path)
    for st in progress.shards.values():
        if st.receipt is not None:
            sources += [(st.receipt.finished_utc, st.receipt.container_id, p)
                        for p in st.receipt.parts]
        for rec in st.quarantines:
            sources += [(rec.created_utc, "", p) for p in rec.detail.get("done_parts", ())]
    sources = sorted(set(sources))

    expected: set[tuple[str, str]] = set()
    excused: set[tuple[str, str]] = set()
    for (uid, kind), est in progress.episodes.items():
        for stage in stages_of(kind):
            expected.add((uid, stage))
            if est in ("failed", "poison_pill"):
                excused.add((uid, stage))

    out = ComposeInput({}, expected, excused)
    seen_paths: set[str] = set()
    for _t, _c, path in sources:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        for rec in read_part(jf, path):
            if rec.provenance.config_hash != jf.config_hash:
                out.foreign += 1
                continue
            key = record_key(rec)
            prev = out.records.get(key)
            if prev is not None:
                out.duplicates += 1
                if prev.to_json() != rec.to_json() and key not in out.conflicts:
                    out.conflicts.append(key)
            out.records[key] = rec  # later source wins
    out.excused -= out.records.keys()  # a record that exists is never excused
    return out
