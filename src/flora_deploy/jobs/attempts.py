"""Attempts → quarantine → bisection (PLAN.md D15, §8).

Modal retries crashes, OOMs and preemptions forever inside one FunctionCall, so the orchestrator
counts **container starts** from files instead: every container that starts a shard writes one
``AttemptMarker`` named by its container id (``JobPaths.attempt``). Concurrent or Modal-retried
starts never overwrite each other.

**What counts (D15, revised).** Only starts that may have *crashed* count toward the limit:

* a start whose worker rewrote its marker with ``preempted=True`` (:func:`mark_preempted`, called
  from ``@modal.exit`` / the SIGINT handler: Modal gives preempted containers a 30 s grace) never
  counts, however many times a long shard is preempted;
* a start whose worker caught a python exception and rewrote its marker with ``error``
  (:func:`mark_exception`) never counts: Modal reports it through ``modal.Retries`` and the
  orchestrator respawns, then marks the shard ``shard_failed`` (never bisected);
* except when the marker also says ``crashed`` (:func:`mark_exception` with ``crashed=True``: a
  CUDA fault or GPU OOM after which the worker replaces its container, which Modal retries
  forever): that start counts, so a deterministic GPU fault is isolated, not looped (review 6 M1);
* a shard with a receipt never counts (it is done);
* every other marker counts: OOM and SIGKILL leave no trace, so a start without a preempt mark is
  presumed to have crashed or to be crashing (the newest one may still be running; with K=3 the
  third unmarked start triggers quarantine, exactly as before for crash loops).

:func:`count_starts` is the raw number of distinct markers; :func:`counted_starts` is the D15
count. A preemption that kills the container before ``@modal.exit`` finishes is counted as a crash
(conservative: it can only bring quarantine forward, never lose work, since bisection keeps every
episode).

**Listing lag.** Markers are found by ``Store.list``, which may be eventually consistent. The
count is therefore a *lower bound* on the true number of starts: markers written in the last
"lag" window may be missing. Consequences, which the property tests check:

* quarantine never fires early (a marker is never counted that was not written; the preempt flag
  is read with ``get`` from the marker itself, so a listed marker is never counted with a stale
  flag on read-after-write stores);
* it fires once ``limit`` markers are visible, i.e. at most ``limit - 1 + s`` real starts, where
  ``s`` is the number of starts made during one listing-lag window (with one container per shard
  at a time and one start per tick, ``s <= lag``);
* ``known=`` lets a caller add container ids it knows about (e.g. a worker's own id, which a
  lagging listing may not show yet) for an exact count from that caller's view.

**Quarantine** of a multi-episode shard first writes a durable *replan* file
(``JobPaths.replan``: children, done episodes and their parts), read with ``exists``/``get`` and
never ``list``, so bisection children are visible to everyone the moment the decision exists. The
first replan written wins: a re-quarantine (e.g. after a crash, or because the record was not
listed yet) reuses its children, so the child set never changes. Then it writes a
``ControlRecord(kind="quarantine", target=shard_id)`` whose ``detail`` repeats everything status
needs: the config hash, the episodes already done with the part files that hold them (compose
reads those, since the parent never writes a receipt), the bisected child shards, and any
poison-pilled episodes. A shard with several unfinished episodes is bisected into single-episode
shards ``<kind>-ep-<uid>`` (``bisected_from`` = parent). A single-episode shard that reaches the
limit is a poison pill: an ``EpisodeError(kind="poison_pill")`` per stage, written whole-file to
``errors/``. Ids and error paths are deterministic, so quarantining twice produces the same
children and the same error file: it is idempotent.

**Failure is not quarantine.** A shard whose calls keep *failing* (python exceptions after
``modal.Retries``, expiry without a receipt) is marked :func:`mark_failed`
(``ControlRecord(kind="shard_failed")``): terminal, never bisected, and the job ends ``failed``.

**Spawn tokens.** :func:`spawn_token` is the deterministic idempotency token of the orchestrator's
``n``-th spawn of a shard. Workers record it in their marker, and :func:`superseded` tells a worker
whose spawn the orchestrator has since replaced (or never confirmed) to exit without work.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

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
UPSTREAM_CONTAINER = "upstream"  # container-id slot of "upstream poisoned" error files


@dataclass(frozen=True)
class JobFiles:
    """A store plus the job/config it is scoped to.

    ``config_hash`` is the **job** hash (what the orchestrator passes to ``run_shard``).
    ``stage_hashes`` (shard kind -> stage hash) re-key every per-stage area of ``paths`` (parts,
    receipts, attempts, errors, audits, replans; ``jobs.store.JobPaths``). ``None`` (default):
    read them from this config's manifest on first use (``JobPaths.config_manifest``, then the
    job-level manifest if it is this config's), so a worker that only knows the job hash still
    writes where the orchestrator and compose look. ``{}``: key everything by the job hash.
    """

    store: Store
    job_id: str
    config_hash: str
    stage_hashes: Mapping[str, str] | None = None
    paths: JobPaths = field(init=False)

    def __post_init__(self) -> None:
        sh = None if self.stage_hashes is None else dict(self.stage_hashes)
        object.__setattr__(self, "_sh", sh)
        object.__setattr__(self, "paths", JobPaths(self.job_id, config_hash=self.config_hash,
                                                   stage_hashes=self.stage_map))

    def stage_map(self) -> dict[str, str]:
        sh = self._sh  # type: ignore[attr-defined]
        if sh is None:
            sh = {}
            probe = JobPaths(self.job_id)
            for path in (probe.config_manifest(self.config_hash), probe.manifest):
                if self.store.exists(path):
                    d = json.loads(self.store.get_bytes(path))
                    if d.get("config_hash") == self.config_hash:
                        sh = dict(d.get("stage_hashes") or {})
                        break
            object.__setattr__(self, "_sh", sh)
        return sh

    def key(self, kind: str) -> str:
        """The hash ``kind``'s shard files are keyed by and its records are stamped with."""
        return self.paths.stage_key(self.config_hash, kind)

    def accepts(self, kind: str) -> frozenset[str]:
        """``provenance.config_hash`` values a record of ``kind`` may carry and still belong to
        this config: its stage hash, or the job hash (workers that stamp the job hash)."""
        return frozenset({self.key(kind), self.config_hash})


def compact_utc(created_utc: str) -> str:
    """``2026-10-21T12:00:00.123Z`` → ``20261021T120000.123Z`` (path-safe)."""
    return safe_key(re.sub(r"[^0-9A-Za-z.]", "", created_utc) or "0")


# --- spawn tokens --------------------------------------------------------------------------

def spawn_token(job_id: str, config_hash: str, shard_id: str, n: int) -> str:
    """Deterministic token of the orchestrator's ``n``-th spawn of a shard: ``<n>.<hash12>``."""
    h = hashlib.sha256(f"{job_id}\0{config_hash}\0{shard_id}\0{n}".encode()).hexdigest()[:12]
    return f"{n}.{h}"


def token_attempt(token: str) -> int:
    """The spawn number in a token (0 for an empty / foreign token)."""
    head = token.split(".", 1)[0]
    return int(head) if head.isdigit() else 0


def superseded(jf: JobFiles, kind: ShardKind, shard_id: str, token: str) -> bool:
    """Worker step 0: True if this spawn must not run.

    The orchestrator's ledger (``JobPaths.call``, read-after-write) names the token of the one
    spawn it currently wants. A worker proceeds only if the ledger is absent (manual run), has no
    token (older orchestrator), or names *this* token while the shard is still being spawned or
    active. Anything else (a newer token, or the shard is done / quarantined / failed /
    cancelled) means this container belongs to a duplicate or stale call.
    """
    if not token:
        return False
    p = jf.paths.call(jf.config_hash, kind, shard_id)
    if not jf.store.exists(p):
        return False
    try:
        led = json.loads(jf.store.get_bytes(p))
    except ValueError:
        return False
    want = led.get("token") or ""
    if not want:
        return False
    return want != token or led.get("status") not in ("spawning", "active")


# --- markers -------------------------------------------------------------------------------

def write_attempt(jf: JobFiles, kind: ShardKind, shard_id: str, container_id: str, *,
                  started_utc: str, orchestrator_attempt: int, token: str = "",
                  call_id: str = "") -> AttemptMarker:
    """Called by the worker as the first thing it does for a shard (before any decode)."""
    m = AttemptMarker(shard_id=shard_id, container_id=container_id, started_utc=started_utc,
                      orchestrator_attempt=orchestrator_attempt, token=token, call_id=call_id)
    jf.store.put_bytes(jf.paths.attempt(jf.config_hash, kind, shard_id, container_id), to_json(m))
    return m


def read_attempt(jf: JobFiles, kind: ShardKind, shard_id: str, container_id: str
                 ) -> AttemptMarker | None:
    p = jf.paths.attempt(jf.config_hash, kind, shard_id, container_id)
    if not jf.store.exists(p):
        return None
    try:
        return from_json(AttemptMarker, jf.store.get_bytes(p))
    except (KeyError, TypeError, ValueError):
        return None


def mark_preempted(jf: JobFiles, kind: ShardKind, shard_id: str, container_id: str, *,
                   ended_utc: str, marker: AttemptMarker | None = None) -> AttemptMarker:
    """Worker-side, from ``@modal.exit`` / the SIGINT handler (after flushing the current part):
    rewrite this container's marker with ``preempted=True`` so the start does not count toward
    D15 quarantine. Whole-file rewrite of the container's own marker (it is the only writer)."""
    m = marker or read_attempt(jf, kind, shard_id, container_id) or AttemptMarker(
        shard_id=shard_id, container_id=container_id, started_utc=ended_utc,
        orchestrator_attempt=0)
    m = replace(m, preempted=True, ended_utc=ended_utc)
    jf.store.put_bytes(jf.paths.attempt(jf.config_hash, kind, shard_id, container_id), to_json(m))
    return m


def mark_exception(jf: JobFiles, kind: ShardKind, shard_id: str, container_id: str, *,
                   ended_utc: str, error: str, marker: AttemptMarker | None = None,
                   crashed: bool = False) -> AttemptMarker:
    """Worker-side, in the ``except`` around the shard (before re-raising): this start ended in a
    python exception, which Modal counts against ``modal.Retries`` and the orchestrator handles by
    respawning (then ``shard_failed``), so it must not also count as a D15 crash.

    ``crashed=True``: the exception was a CUDA fault / GPU OOM and the worker is about to replace
    its container (``os._exit``), which Modal retries like any crash, indefinitely. That start
    **does** count toward D15, so a deterministic fault is quarantined and bisected down to a
    poison pill instead of looping (review 6 M1)."""
    m = marker or read_attempt(jf, kind, shard_id, container_id) or AttemptMarker(
        shard_id=shard_id, container_id=container_id, started_utc=ended_utc,
        orchestrator_attempt=0)
    m = replace(m, error=error[:2000], ended_utc=ended_utc, crashed=crashed)
    jf.store.put_bytes(jf.paths.attempt(jf.config_hash, kind, shard_id, container_id), to_json(m))
    return m


def attempt_containers(jf: JobFiles, kind: ShardKind, shard_id: str) -> set[str]:
    """Container ids with a visible marker (a lower bound under listing lag)."""
    prefix = jf.paths.attempt(jf.config_hash, kind, shard_id, "x")[:-len("x.json")]
    return {p[len(prefix):-len(".json")] for p in jf.store.list(prefix) if p.endswith(".json")}


def count_starts(jf: JobFiles, kind: ShardKind, shard_id: str, *,
                 known: Iterable[str] = ()) -> int:
    """Raw number of distinct visible markers (plus ``known``), preempted or not."""
    return len(attempt_containers(jf, kind, shard_id) | set(known))


def counted_starts(jf: JobFiles, kind: ShardKind, shard_id: str, *, known: Iterable[str] = (),
                   limit: int | None = None) -> int:
    """D15 count: distinct markers not flagged ``preempted`` or ``error``, plus those flagged
    ``crashed`` (module docstring).

    With ``limit``, markers are only read when the raw count reaches it (the raw count is an
    upper bound of the D15 count), so healthy shards cost one ``list`` and no ``get``.
    """
    ids = attempt_containers(jf, kind, shard_id) | set(known)
    if limit is not None and len(ids) < limit:
        return len(ids)
    n = 0
    for cid in ids:
        m = read_attempt(jf, kind, shard_id, cid)
        if m is None or m.crashed or not (m.preempted or m.error):
            n += 1
    return n


def markers(jf: JobFiles, kind: ShardKind, shard_id: str) -> list[AttemptMarker]:
    out = []
    for cid in sorted(attempt_containers(jf, kind, shard_id)):
        m = read_attempt(jf, kind, shard_id, cid)
        if m is not None:
            out.append(m)
    return out


def call_for_token(jf: JobFiles, kind: ShardKind, shard_id: str, token: str) -> str | None:
    """The call id recorded by a container that started with ``token`` (adoption after a crash
    between spawn and the ledger write), or ``None`` if no such marker is visible yet."""
    if not token:
        return None
    for m in markers(jf, kind, shard_id):
        if m.token == token and m.call_id:
            return m.call_id
    return None


def is_poisoned(jf: JobFiles, shard: Shard) -> bool:
    """A single-episode shard whose episode was isolated by D15: its deterministic poison error
    file exists (``exists``: read-after-write, never hidden by listing lag)."""
    return len(shard.episodes) == 1 and jf.store.exists(poison_error_path(jf, shard))


def has_receipt(jf: JobFiles, shard: Shard) -> bool:
    return jf.store.exists(jf.paths.receipt(jf.config_hash, shard.kind, shard.shard_id))


def should_quarantine(jf: JobFiles, shard: Shard, *, limit: int,
                      known: Iterable[str] = ()) -> bool:
    """No receipt and at least ``limit`` counted starts. The receipt check uses ``exists``
    (read-after-write), so a finished shard is never quarantined because of listing lag."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if has_receipt(jf, shard):
        return False
    return counted_starts(jf, shard.kind, shard.shard_id, known=known, limit=limit) >= limit


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


# --- durable replans -----------------------------------------------------------------------

@dataclass(frozen=True)
class Replan:
    """The durable bisection decision of one shard (``JobPaths.replan``)."""

    parent: str
    config_hash: str
    children: tuple[Shard, ...]
    done: tuple[str, ...]
    done_parts: tuple[str, ...]
    created_utc: str


def _shard_from(d: Mapping[str, Any]) -> Shard:
    return from_json(Shard, json.dumps(dict(d)))


def read_replan(jf: JobFiles, shard: Shard) -> Replan | None:
    """``exists`` + ``get`` (read-after-write): never hidden by listing lag."""
    p = jf.paths.replan(jf.config_hash, shard.kind, shard.shard_id)
    if not jf.store.exists(p):
        return None
    d = json.loads(jf.store.get_bytes(p))
    return Replan(parent=d["parent"], config_hash=d["config_hash"],
                  children=tuple(_shard_from(c) for c in d["children"]),
                  done=tuple(d.get("done", ())), done_parts=tuple(d.get("done_parts", ())),
                  created_utc=d.get("created_utc", ""))


def write_replan(jf: JobFiles, shard: Shard, children: Iterable[Shard], *, created_utc: str,
                 done: Iterable[str] = (), done_parts: Iterable[str] = ()) -> Replan:
    """Persist ``shard``'s bisection. First writer wins: if a replan exists it is returned
    unchanged, so the child set of a shard never changes once decided."""
    cur = read_replan(jf, shard)
    if cur is not None:
        return cur
    children = tuple(children)
    for c in children:
        if c.bisected_from != shard.shard_id:
            raise ValueError(f"{c.shard_id!r} is not a child of {shard.shard_id!r}")
    child_eps = {u for c in children for u in c.episodes}
    done_t = tuple(u for u in shard.episodes if u in set(done) and u not in child_eps)
    rp = Replan(shard.shard_id, jf.config_hash, children, done_t,
                tuple(sorted(set(done_parts))), created_utc)
    jf.store.put_bytes(jf.paths.replan(jf.config_hash, shard.kind, shard.shard_id), json.dumps(
        {"parent": rp.parent, "config_hash": rp.config_hash,
         "children": [asdict(c) for c in rp.children], "done": list(rp.done),
         "done_parts": list(rp.done_parts), "created_utc": rp.created_utc},
        sort_keys=True).encode())
    return rp


# --- quarantine ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Quarantine:
    control_path: str
    record: ControlRecord
    children: tuple[Shard, ...]
    poisoned: tuple[str, ...]
    error_path: str | None


def poison_error_path(jf: JobFiles, shard: Shard) -> str:
    return jf.paths.errors(jf.config_hash, shard.kind, shard.shard_id, QUARANTINE_CONTAINER, 0)


def write_poison(jf: JobFiles, shard: Shard, *, message: str,
                 diagnostics: Mapping[str, Any] | None = None) -> str:
    """``EpisodeError(kind="poison_pill")`` per stage for a single-episode shard (deterministic
    path, whole file)."""
    if len(shard.episodes) != 1:
        raise ValueError(f"{shard.shard_id!r}: only single-episode shards are poison-pilled")
    path = poison_error_path(jf, shard)
    errs = [EpisodeError(uid=shard.episodes[0], stage=stage, shard_id=shard.shard_id,
                         container_id=QUARANTINE_CONTAINER, kind="poison_pill", message=message,
                         diagnostics=dict(diagnostics or {}))
            for stage in stages_of(shard.kind)]
    jf.store.put_bytes(path, "".join(
        json.dumps(asdict(e), sort_keys=True) + "\n" for e in errs).encode())
    return path


def quarantine(jf: JobFiles, shard: Shard, *, created_utc: str, done: Iterable[str] = (),
               done_parts: Iterable[str] = (), durations: Mapping[str, float] | None = None,
               starts: int | None = None, reason: str = "attempt limit reached",
               diagnostics: dict | None = None) -> Quarantine:
    """Quarantine ``shard``: bisect it or, if it holds one unfinished episode, poison-pill it.

    ``done`` / ``done_parts``: the episodes the shard's visible parts already hold, and those part
    paths (``progress.shard_parts`` gives both). Write order: replan file (multi-episode) or
    poison error file (single episode), then the ControlRecord. The caller cancels the shard's
    outstanding FunctionCalls; this function only writes files.
    """
    done_set = set(done)
    parts = set(done_parts)
    poisoned: tuple[str, ...] = ()
    error_path = None
    if len(shard.episodes) > 1:
        rp = write_replan(jf, shard, bisect(shard, done=done_set, durations=durations),
                          created_utc=created_utc, done=done_set, done_parts=parts)
        children, done_t = rp.children, rp.done
        parts |= set(rp.done_parts)
    else:
        children = ()
        done_t = tuple(u for u in shard.episodes if u in done_set)
        if not done_t:
            poisoned = shard.episodes
            error_path = write_poison(
                jf, shard, message=f"{reason}: shard reached {starts} starts without a receipt",
                diagnostics=diagnostics)
    rec = ControlRecord(
        kind="quarantine", target=shard.shard_id, reason=reason, created_utc=created_utc,
        detail={"config_hash": jf.config_hash, "shard_kind": shard.kind, "starts": starts,
                "done": list(done_t), "done_parts": sorted(parts),
                "children": [asdict(c) for c in children], "poisoned": list(poisoned),
                "error_path": error_path})
    path = jf.paths.control("quarantine", shard.shard_id, compact_utc(created_utc))
    jf.store.put_bytes(path, to_json(rec))
    return Quarantine(path, rec, tuple(children), poisoned, error_path)


def mark_failed(jf: JobFiles, shard: Shard, *, created_utc: str, reason: str,
                detail: Mapping[str, Any] | None = None) -> str:
    """Terminal failure (respawns exhausted). Never bisects; returns the record path."""
    rec = ControlRecord(kind="shard_failed", target=shard.shard_id, reason=reason,
                        created_utc=created_utc,
                        detail={**dict(detail or {}), "config_hash": jf.config_hash,
                                "shard_kind": shard.kind})
    path = jf.paths.control("shard_failed", shard.shard_id, compact_utc(created_utc))
    jf.store.put_bytes(path, to_json(rec))
    return path


def read_control(store: Store, path: str) -> ControlRecord:
    return from_json(ControlRecord, store.get_bytes(path))


def children_of(rec: ControlRecord) -> tuple[Shard, ...]:
    return tuple(_shard_from(c) for c in rec.detail.get("children", ()))


# --- upstream poison (M4) ------------------------------------------------------------------

def upstream_poisoned(jf: JobFiles, shard: Shard, upstream: Iterable[Shard]) -> set[str]:
    """uids of ``shard`` whose upstream stage (a ``depends_on`` shard, e.g. the segment shard of
    a caption shard) was poison-pilled by D15. ``upstream``: the dependency shards (from the
    manifest). Uses ``exists`` on the deterministic poison error paths only (no ``list``)."""
    mine = set(shard.episodes)
    deps = {s.shard_id: s for s in upstream if s.shard_id in set(shard.depends_on)}
    out: set[str] = set()
    for dep in deps.values():
        for uid in dep.episodes:
            if uid not in mine or uid in out:
                continue
            holder = dep if len(dep.episodes) == 1 else Shard(
                child_shard_id(dep.kind, uid), dep.kind, (uid,), 0.0,
                bisected_from=dep.shard_id)
            if jf.store.exists(poison_error_path(jf, holder)):
                out.add(uid)
    return out


def write_upstream_failures(jf: JobFiles, shard: Shard, uids: Iterable[str], *,
                            container_id: str = UPSTREAM_CONTAINER) -> str | None:
    """Errors for episodes skipped because their upstream stage was poison-pilled."""
    uids = [u for u in shard.episodes if u in set(uids)]
    if not uids:
        return None
    path = jf.paths.errors(jf.config_hash, shard.kind, shard.shard_id, container_id, 0)
    errs = [EpisodeError(uid=u, stage=stage, shard_id=shard.shard_id, container_id=container_id,
                         kind="poison_pill", message="upstream stage poison-pilled (D15)")
            for u in uids for stage in stages_of(shard.kind)]
    jf.store.put_bytes(path, "".join(
        json.dumps(asdict(e), sort_keys=True) + "\n" for e in errs).encode())
    return path
