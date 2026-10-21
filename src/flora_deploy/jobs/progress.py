"""Job progress derived purely from store files (PLAN.md §2): no in-memory or Modal Dict state.

Inputs: the manifest, receipts (``exists``/``get``: read-after-write), attempt markers, parts,
quarantine ``ControlRecord``s and error files (``list``: possibly lagging).

**Effective shards** = manifest shards + bisection children, transitively. Children come from
the durable replan files (``attempts.read_replan``: ``exists``/``get``, never hidden by lag) and
from quarantine records. A replan file whose record is not listed yet stands in for the record.

**Per-shard state** (first match wins):

* ``done``: its receipt exists (even if it was also quarantined; a late receipt from a container
  that was still running is real output);
* ``failed``: a ``shard_failed`` record (respawns exhausted): terminal, never bisected; the job
  cannot complete;
* ``quarantined``: a quarantine record or replan file is visible (its children carry the work,
  or it poisoned its single episode), or, for a single-episode shard, its deterministic poison
  error file exists (``attempts.is_poisoned``: read with ``exists``, so a lagging listing of the
  quarantine record never makes a poison-pilled job look unfinished, M3);
* ``quarantine_due``: ``limit`` *counted* starts (``attempts.counted_starts``: preempted
  starts excluded) and no receipt: the orchestrator should call ``attempts.quarantine``
  (idempotent, see there);
* ``blocked``: some ``depends_on`` shard is not resolved, where resolved means every one of its
  episodes is terminal for that shard's kind (done, failed or poison-pilled), which also covers a
  dependency that was quarantined and bisected;
* ``ready`` (no visible start) / ``started``.

**Per-episode status**, per shard kind: ``done`` (in a receipt's ``episodes_done`` or a
quarantine record's ``done``), ``failed`` (in a receipt's ``episodes_failed``: NVIDIA's own
failure, terminal, not retried; workers never put ``infra`` errors in a receipt, they raise and
the shard is retried), ``poison_pill`` (quarantine record or poison file), else ``pending``.
``done`` takes precedence. Resume set = manifest − done − failed − poison-pilled, per kind.

**Completion** (:attr:`Progress.complete`) means every effective shard resolved;
:meth:`Progress.outcome` adds the error-rate guard: ``done`` only when the excluded episodes of
every kind stay within ``max_error_rate`` (and a kind with planned episodes has at least one
done), else ``completed_with_errors``; ``failed`` if a shard failed for good.

**Keys.** Shard files are keyed by the stage hash of their kind (``JobFiles.key``) and records
are accepted when their ``provenance.config_hash`` is that stage hash or the job hash
(``JobFiles.accepts``).

**Listing lag** can hide a quarantine record for a while; the parent then shows
``quarantine_due`` again and re-quarantining is idempotent. Because ``complete`` needs every
effective shard resolved and each quarantine record lists the parts that hold its done episodes,
a ``complete`` view always has every expected record reachable (``compose_input().missing`` is
empty), which the property tests check.

**Compose input** = the parts named by receipts, the ``done_parts`` of quarantine records, and
every part file *listed* under the part prefix of a done or quarantined shard (so a receipt that
names only its own container's parts loses nothing: ``ShardReceipt.parts`` should name all of
them, and whole part files are immutable, so reading extra ones is safe), deduplicated by
``records.record_key``. Records whose ``provenance.config_hash`` differs from the job's are
dropped (counted as ``foreign``). Among duplicates (same key, same config hash) the *latest*
source wins: sources are ordered by (receipt ``finished_utc`` or quarantine ``created_utc``,
container id, part path), and a later one replaces an earlier one. Within one config hash outputs
should be identical (``exact`` is deterministic), so a duplicate whose JSON differs is reported
in ``conflicts`` instead of being hidden.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from flora_deploy.jobs.attempts import (
    JobFiles,
    Replan,
    children_of,
    counted_starts,
    is_poisoned,
    read_control,
    read_replan,
)
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

ShardState = Literal["done", "failed", "quarantined", "quarantine_due", "blocked", "ready",
                     "started"]
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
    parts = tuple(p for p in jf.store.list(part_prefix(jf, shard)) if p.endswith(".jsonl"))
    ok = jf.accepts(shard.kind)
    have: dict[str, set[str]] = defaultdict(set)
    for p in parts:
        for r in read_part(jf, p):
            if r.provenance.config_hash in ok:
                have[r.uid].add(r.stage)
    need = set(stages_of(shard.kind))
    return ShardParts(parts, tuple(u for u in shard.episodes if need <= have.get(u, set())))


# --- status --------------------------------------------------------------------------------

@dataclass
class ShardStatus:
    shard: Shard
    state: ShardState
    starts: int  # counted (non-preempted) visible attempt markers (lower bound)
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

    def upstream_excluded(self) -> set[tuple[str, ShardKind]]:
        """(uid, kind) pairs excluded because an upstream kind (:data:`UPSTREAM_KINDS`) already
        excluded the episode (a balanced ``caption`` of a failed / poison-pilled ``segment``)."""
        out: set[tuple[str, ShardKind]] = set()
        for (uid, kind), st in self.episodes.items():
            if st in ("failed", "poison_pill") and any(
                    self.episodes.get((uid, up)) in ("failed", "poison_pill")  # type: ignore[arg-type]
                    for up in UPSTREAM_KINDS.get(kind, ())):
                out.add((uid, kind))
        return out

    def exclusions(self) -> dict[ShardKind, dict[str, int]]:
        """Per kind: planned episodes, done, excluded (failed + poison-pilled), pending, and
        ``upstream``: episodes excluded at an upstream kind, which are left out of this kind's
        planned / excluded counts (an episode is counted once, where it failed)."""
        out: dict[ShardKind, dict[str, int]] = {}
        up = self.upstream_excluded()
        for (uid, kind), st in self.episodes.items():
            c = out.setdefault(kind, {"planned": 0, "done": 0, "excluded": 0, "pending": 0,
                                      "upstream": 0})
            if (uid, kind) in up:
                c["upstream"] += 1
                continue
            c["planned"] += 1
            c["done" if st == "done" else "pending" if st == "pending" else "excluded"] += 1
        return out

    def outcome(self, max_error_rate: float) -> str:
        """``failed`` | ``done`` | ``completed_with_errors`` | ``incomplete`` (module docs)."""
        if any(s.state == "failed" for s in self.shards.values()):
            return "failed"
        if not self.complete:
            return "incomplete"
        return "done" if not error_rate_problems(self.exclusions(), max_error_rate) \
            else "completed_with_errors"


# Shard kind -> the kinds upstream of it (``Manifest`` ``depends_on``): an episode those excluded
# is excluded downstream too (``attempts.upstream_excluded``), but counted once, upstream.
UPSTREAM_KINDS: Mapping[str, tuple[str, ...]] = {"caption": ("segment",)}


def tally_exclusions(planned: Mapping[str, set[str]], excluded: Mapping[str, set[str]]
                     ) -> dict[str, dict[str, int]]:
    """Per kind ``planned`` / ``done`` / ``excluded`` / ``pending`` (0) / ``upstream`` counts
    for :func:`error_rate_problems`, from planned and excluded uid sets. Episodes an upstream
    kind (:data:`UPSTREAM_KINDS`) excluded leave the downstream kind's counts (``upstream``),
    so one failing episode is one exclusion, not one per stage. Shared by the orchestrator and
    the reconciler (``Progress.exclusions`` applies the same rule)."""
    out: dict[str, dict[str, int]] = {}
    for kind, eps in planned.items():
        up = {u for k in UPSTREAM_KINDS.get(kind, ()) for u in excluded.get(k, set())
              if u in planned.get(k, set())}
        x = excluded.get(kind, set()) & eps
        inherited = x & up
        own = eps - inherited
        out[kind] = {"planned": len(own), "done": len(own - x), "excluded": len(x - inherited),
                     "pending": 0, "upstream": len(inherited)}
    return out


def with_unplanned(exclusions: dict[str, dict[str, int]], unplanned: Mapping[str, str] | None
                   ) -> dict[str, dict[str, int]]:
    """Add the episodes the plan left out (``Manifest.excluded``: index failures, source errors,
    enforced envelope exclusions, missing task text) to every kind's planned and excluded counts,
    so the error-rate guard sees them (review 6 M6). Shared by the orchestrator, the reconciler
    and compose."""
    n = len(unplanned or {})
    if not n:
        return exclusions
    return {kind: {**c, "planned": c["planned"] + n, "excluded": c["excluded"] + n}
            for kind, c in exclusions.items()}


def error_rate_problems(exclusions: dict[str, dict[str, int]], max_error_rate: float
                        ) -> list[str]:
    """Why a resolved job is not a clean ``done``: per kind, more excluded episodes than
    ``max_error_rate`` of the planned ones, or planned episodes but no record at all."""
    out = []
    for kind, c in sorted(exclusions.items()):
        if not c["planned"]:
            continue
        rate = c["excluded"] / c["planned"]
        if c["done"] == 0:
            out.append(f"{kind}: 0 of {c['planned']} planned episodes have a record "
                       f"({c['excluded']} excluded)")
        elif rate > max_error_rate:
            out.append(f"{kind}: {c['excluded']} of {c['planned']} episodes excluded "
                       f"({rate:.0%} > {max_error_rate:.0%})")
    return out


def _records_of(jf: JobFiles, kind: str) -> dict[str, tuple[ControlRecord, ...]]:
    prefix = f"{jf.paths.root}/control/{kind}/"
    out: dict[str, list[ControlRecord]] = defaultdict(list)
    for p in jf.store.list(prefix):
        try:
            rec = read_control(jf.store, p)
        except (KeyError, TypeError, ValueError):
            continue  # half-visible / malformed file; never fatal
        if rec.kind == kind and rec.detail.get("config_hash") == jf.config_hash:
            out[rec.target].append(rec)
    return {k: tuple(sorted(v, key=lambda r: r.created_utc)) for k, v in out.items()}


def quarantine_records(jf: JobFiles) -> dict[str, tuple[ControlRecord, ...]]:
    """Listed quarantine records of this config (may lag)."""
    return _records_of(jf, "quarantine")


def failed_records(jf: JobFiles) -> dict[str, tuple[ControlRecord, ...]]:
    """Listed ``shard_failed`` records of this config (may lag)."""
    return _records_of(jf, "shard_failed")


def replan_record(rp: Replan, kind: ShardKind) -> ControlRecord:
    """A quarantine record synthesized from a replan file (stand-in while the record lags)."""
    return ControlRecord(
        kind="quarantine", target=rp.parent, reason="replan", created_utc=rp.created_utc,
        detail={"config_hash": rp.config_hash, "shard_kind": kind, "done": list(rp.done),
                "done_parts": list(rp.done_parts),
                "children": [asdict(c) for c in rp.children], "poisoned": [],
                "error_path": None})


def effective_shards(manifest: Manifest, quarantines: dict[str, tuple[ControlRecord, ...]],
                     jf: JobFiles | None = None,
                     replans: dict[str, Replan] | None = None) -> dict[str, Shard]:
    """Manifest shards + bisection children, transitively. With ``jf``, replan files are read
    (``exists``/``get``) for every multi-episode shard and, if ``replans`` is given, collected
    into it."""
    shards = {s.shard_id: s for s in manifest.shards}
    todo = list(shards)
    while todo:
        sid = todo.pop()
        kids: list[Shard] = []
        if jf is not None and len(shards[sid].episodes) > 1:
            rp = read_replan(jf, shards[sid])
            if rp is not None:
                kids += rp.children
                if replans is not None:
                    replans[sid] = rp
        for rec in quarantines.get(sid, ()):
            kids += children_of(rec)
        for child in kids:
            if child.shard_id not in shards:
                shards[child.shard_id] = child
                todo.append(child.shard_id)
    return shards


def derive(jf: JobFiles, manifest: Manifest, *, limit: int) -> Progress:
    """Rebuild the full job status from files. Stateless: safe to call from any process."""
    if manifest.config_hash != jf.config_hash:
        raise ValueError("manifest and JobFiles disagree on config_hash")
    qs = dict(quarantine_records(jf))
    fails = failed_records(jf)
    replans: dict[str, Replan] = {}
    shards = effective_shards(manifest, qs, jf, replans)
    for sid, rp in replans.items():
        if sid not in qs:
            qs[sid] = (replan_record(rp, shards[sid].kind),)

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

    def resolved(sid: str, downstream: Shard) -> bool:
        """Per episode: ``downstream``'s episodes are terminal at the dependency's kind."""
        dep = shards.get(sid)
        mine = set(downstream.episodes)
        return dep is not None and all(episodes.get((u, dep.kind)) in TERMINAL
                                       for u in dep.episodes if u in mine)

    # M3: a poison-pilled single-episode shard whose quarantine record is not listed yet
    poison: set[str] = set()
    for sid, s in shards.items():
        if receipts[sid] is None and not qs.get(sid) and sid not in fails \
                and is_poisoned(jf, s):
            poison.add(sid)
            mark(s.episodes[0], s.kind, "poison_pill")

    statuses: dict[str, ShardStatus] = {}
    for sid, s in shards.items():
        starts = counted_starts(jf, s.kind, sid, limit=limit)
        r = receipts[sid]
        q = qs.get(sid, ())
        if r is not None:
            st: ShardState = "done"
        elif sid in fails:
            st = "failed"
        elif q or sid in poison:
            st = "quarantined"
        elif starts >= limit:
            st = "quarantine_due"
        elif not all(resolved(d, s) for d in s.depends_on):
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


def part_prefix(jf: JobFiles, shard: Shard) -> str:
    return _dir_prefix(jf.paths.part(jf.config_hash, shard.kind, shard.shard_id, "x", 0),
                       "x-0.jsonl")


def compose_input(jf: JobFiles, progress: Progress) -> ComposeInput:
    sources: list[tuple[str, str, str]] = []  # (time, container, part path)
    accepts: dict[str, frozenset[str]] = {}  # part path -> config hashes its records may carry
    for st in progress.shards.values():
        named: set[str] = set()
        if st.receipt is not None:
            sources += [(st.receipt.finished_utc, st.receipt.container_id, p)
                        for p in st.receipt.parts]
            named |= set(st.receipt.parts)
        for rec in st.quarantines:
            sources += [(rec.created_utc, "", p) for p in rec.detail.get("done_parts", ())]
            named |= set(rec.detail.get("done_parts", ()))
        if st.receipt is not None or st.quarantines:
            # H3: also every *listed* part of a finished shard, whatever the receipt names. Part
            # files are whole and immutable, so reading extra ones is safe; lowest priority.
            listed = [p for p in jf.store.list(part_prefix(jf, st.shard))
                      if p.endswith(".jsonl") and p not in named]
            sources += [("", "", p) for p in listed]
            named |= set(listed)
        for p in named:
            accepts[p] = accepts.get(p, frozenset()) | jf.accepts(st.shard.kind)
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
        ok = accepts.get(path)
        for rec in read_part(jf, path):
            if ok is None or rec.provenance.config_hash not in ok:
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
