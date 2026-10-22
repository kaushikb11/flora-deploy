"""Per-container facts of a job, derived from what the job system already writes (review 11a H1).

No worker writes a ``ContainerReceipt`` today, so ``flora report``'s container list was always
empty. Every worker start does write an ``AttemptMarker`` (container id = ``MODAL_TASK_ID``, the
Modal call id, start / end), a ``ShardReceipt`` when the shard finishes (container id, finish
time, ``nvidia-smi`` GPU) and the orchestrator a call ledger (``call.json``: when each call was
spawned). From those, per container (:func:`container_rows`):

- ``kind``: the shard kind it ran (``None`` if it ran several, never in a deployed app);
- ``gpu``: its pool's GPU (``H100``; ``gpu_seen``: the receipts' ``nvidia-smi`` name);
- ``startup_s``: its first call's spawn to its first attempt marker (the cold start, queueing
  included);
- ``busy_s``: the sum over its shard attempts of marker start to receipt (or marker end);
- ``seconds``: its first call's spawn to ``min(stopped_at, now, last end + scale-down window)``,
  what it was billed for; ``usd``: that at its pool's list price.

Startup facts (:func:`write_startup_facts` / :func:`read_startup_facts`): a worker that wants its
``@modal.enter`` facts (batch policy, executor size, container resources) next to the job writes
them once per job and container to ``<job root>/containers/<container id>.json``; the fake
balanced workers do (``local.balanced``), so the H0 rehearsal can check them on Modal.

Orchestration bookkeeping only (``config/logic.py``: not output-affecting).
"""

from __future__ import annotations

import calendar
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from typing import Any


def parse_utc(text: str | None) -> float | None:
    """``2026-10-21T21:49:38.945966Z`` (fractions optional) -> epoch seconds."""
    if not text:
        return None
    try:
        base = float(calendar.timegm(time.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return None
    frac = text[19:].rstrip("Z")
    if frac.startswith("."):
        try:
            base += float(frac)
        except ValueError:
            pass
    return base


def container_rows(markers: Iterable[Mapping[str, Any]], receipts: Iterable[Mapping[str, Any]],
                   spawned: Mapping[str, float], *, pool_gpus: Mapping[str, str | None],
                   scaledown_s: float, stopped_at: float | None = None,
                   now: float | None = None,
                   usd_per_s: Callable[[str], float] | None = None) -> list[dict[str, Any]]:
    """Module docstring. ``markers``: ``AttemptMarker`` fields plus ``kind``; ``receipts``:
    ``ShardReceipt`` fields; ``spawned``: call id -> spawn time; ``usd_per_s``: kind -> its
    pool's $/s (for ``usd``)."""
    done = {(r["shard_id"], r["container_id"]): r for r in receipts}
    by_c: dict[str, list[Mapping[str, Any]]] = {}
    for m in markers:
        by_c.setdefault(str(m["container_id"]), []).append(m)
    out = []
    for cid, ms in sorted(by_c.items()):
        ms = sorted(ms, key=lambda m: parse_utc(m.get("started_utc")) or 0.0)
        kinds = sorted({str(m["kind"]) for m in ms})
        kind = kinds[0] if len(kinds) == 1 else None
        t_start = parse_utc(ms[0].get("started_utc"))
        t_spawn = spawned.get(str(ms[0].get("call_id") or ""))
        busy, last_end, seen, open_ = 0.0, None, None, False
        for m in ms:
            s0 = parse_utc(m.get("started_utc"))
            r = done.get((m["shard_id"], cid))
            s1 = parse_utc(r.get("finished_utc")) if r else parse_utc(m.get("ended_utc"))
            if r and r.get("gpu"):
                seen = r["gpu"]
            if s0 is not None and s1 is not None and s1 >= s0:
                busy += s1 - s0
                last_end = s1 if last_end is None else max(last_end, s1)
            elif s1 is None:
                open_ = True  # still running (or killed without a mark)
        startup = (t_start - t_spawn) if (t_start is not None and t_spawn is not None
                                          and t_start >= t_spawn) else None
        born = t_spawn if t_spawn is not None else t_start
        end = None
        if last_end is not None and not open_:
            end = last_end + scaledown_s
        for cap in (stopped_at, now):
            if cap is not None:
                end = cap if end is None else min(end, cap)
        seconds = round(end - born, 1) if end is not None and born is not None else None
        gpu = (pool_gpus.get(kind) or "").rstrip("!") or None if kind else None
        row = {"container_id": cid, "kind": kind, "kinds": kinds, "gpu": gpu,
               "gpu_count": 1 if gpu else 0, "gpu_seen": seen,
               "started_utc": ms[0].get("started_utc"),
               "startup_s": round(startup, 1) if startup is not None else None,
               "busy_s": round(busy, 1), "seconds": seconds, "shards": len(ms),
               "stopped": end is not None and (now is None or end < now),
               "source": "attempt_markers"}
        row["usd"] = (round(usd_per_s(kind) * seconds, 4)
                      if usd_per_s is not None and kind and seconds is not None else 0.0)
        out.append(row)
    return out


def job_container_rows(store: Any, job_id: str, config_hash: str, manifest: Any, *,
                       pools: Mapping[str, Any], scaledown_s: float,
                       stopped_at: float | None = None, now: float | None = None,
                       rates: Any = None) -> list[dict[str, Any]]:
    """:func:`container_rows` read from a job's store. ``pools``: kind -> ``PoolSettings``."""
    from flora_deploy.jobs import attempts as A
    from flora_deploy.jobs import progress as P
    from flora_deploy.jobs.orchestrator import ShardLedger
    from flora_deploy.jobs.store import JobPaths

    sh = dict(getattr(manifest, "stage_hashes", None) or {})
    jf = A.JobFiles(store, job_id, config_hash, stage_hashes=sh)
    shards = P.effective_shards(manifest, P.quarantine_records(jf), jf)
    paths = JobPaths(job_id, config_hash=config_hash, stage_hashes=sh)
    markers: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    spawned: dict[str, float] = {}
    for s in shards.values():
        for m in A.markers(jf, s.kind, s.shard_id):
            markers.append({**asdict(m), "kind": s.kind})
        r = P.read_receipt(jf, s.kind, s.shard_id)
        if r is not None:
            receipts.append(asdict(r))
        p = paths.call(config_hash, s.kind, s.shard_id)
        if not store.exists(p):
            continue
        try:
            led = ShardLedger.from_bytes(store.get_bytes(p))
        except (KeyError, TypeError, ValueError):
            continue
        for e in led.calls:
            t = parse_utc(e.get("utc"))
            if e.get("outcome") == "spawned" and e.get("call_id") and t is not None:
                spawned.setdefault(str(e["call_id"]), t)

    def usd_per_s(kind: str) -> float:
        pool = pools.get(kind)
        return rates.pool_per_second(pool) if (rates is not None and pool is not None) else 0.0

    return container_rows(markers, receipts, spawned,
                          pool_gpus={k: getattr(p, "gpu", None) for k, p in pools.items()},
                          scaledown_s=scaledown_s, stopped_at=stopped_at, now=now,
                          usd_per_s=usd_per_s)


STARTUP_SCHEMA = "flora_container_startup_v1"


def startup_facts_path(job_id: str, container_id: str) -> str:
    from flora_deploy.jobs.store import JobPaths
    from flora_deploy.jobs.types import safe_key

    return f"{JobPaths(job_id).root}/containers/{safe_key(container_id)}.json"


def write_startup_facts(store: Any, job_id: str, container_id: str, kind: str,
                        facts: Mapping[str, Any], *, now: float | None = None) -> str:
    """Write one container's startup facts for ``job_id`` (module docstring)."""
    import json

    from flora_deploy.jobs.orchestrator import utc

    path = startup_facts_path(job_id, container_id)
    body = {"schema": STARTUP_SCHEMA, "container_id": container_id, "kind": kind,
            "written_utc": utc(time.time() if now is None else now), "startup": dict(facts)}
    store.put_bytes(path, json.dumps(body, sort_keys=True, default=str).encode())
    return path


def read_startup_facts(store: Any, job_id: str) -> dict[str, dict[str, Any]]:
    """Container id -> the startup facts written for ``job_id`` (unreadable files skipped)."""
    import json

    from flora_deploy.jobs.store import JobPaths

    out: dict[str, dict[str, Any]] = {}
    for p in store.list(f"{JobPaths(job_id).root}/containers/"):
        if not p.endswith(".json"):
            continue
        try:
            d = json.loads(store.get_bytes(p))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("schema") == STARTUP_SCHEMA:
            out[str(d.get("container_id"))] = d
    return out
