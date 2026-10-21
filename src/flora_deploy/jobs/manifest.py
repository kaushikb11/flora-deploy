"""Manifest planning (PLAN.md §8 "Planning"): episode index receipts + ``RunSettings`` → shards.

* Every episode gets ``episode_uid(dataset, episode_id)``; duplicate uids are an error.
* Envelope policy (D13): ``enforce`` drops out-of-envelope episodes from the manifest; ``warn``
  and ``ignore`` keep them. ``EpisodeRef.envelope_ok`` always records the verdict as measured;
  turning it into ``certified:false`` (``warn``) or not (``ignore``) is the stage's job.
* Episodes are grouped by source video file (the sorted set of camera URIs), and groups are laid
  out contiguously, so each concatenated LeRobot v3 MP4 is read in one pass by one shard. A group
  is split across shards only when it alone exceeds the target; it then occupies consecutive
  shards.
* Packing is next-fit in a deterministic order (group key, then episode id): a shard is closed
  when adding the next group would push its estimate past ``shard_target_gpu_minutes``. A shard
  exceeds the target only when it holds a single episode that does.
* Estimates: ``rates[kind]`` GPU-seconds per video-second × episode duration. The defaults are
  placeholders until measured rates exist (S2 → ``rates.json``).
* Kinds: exactly the shard kinds in ``settings.pools`` (the config's stages; H2: a rewards-only
  config plans no ``segcap`` shards), in pipeline order. ``caption`` shards ``depends_on`` the
  ``segment`` shards that hold their episodes.
* Shard ids are content-addressed (H4): ``<kind>-<16 hex of sha256(kind + sorted uids)>``
  (:func:`shard_id_for`), so re-planning under the same config hash (a re-index that added
  episodes, another ``shard_target_gpu_minutes``) gives a changed shard a new id and an old
  receipt never marks it done, while an unchanged shard keeps its id (and its receipt).
  Bisection gives ``<kind>-ep-<uid>`` (``jobs.attempts``).
* Stage hashes (``settings.stage_hashes``) and the packing rates are recorded in the manifest;
  shard files are keyed by the stage hash of their kind (``jobs.store.JobPaths``).
* Storage: one manifest per config (``JobPaths.config_manifest``, M6) plus the job-level
  ``manifest.json`` pointer to the latest plan (the default config of compose / results).

JSON helpers for every job dataclass live here too (``to_json`` / ``from_json``).
"""

from __future__ import annotations

import hashlib
import json
import typing
from collections.abc import Iterable, Mapping
from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any, TypeVar, get_args

from flora_deploy.jobs.settings import RunSettings
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import (
    EpisodeRef,
    Manifest,
    Shard,
    ShardKind,
    episode_uid,
    safe_key,
)
from flora_deploy.media.types import EpisodeIndex
from flora_deploy.records import Stage

REWARD_KINDS: tuple[ShardKind, ...] = ("reward.terminal", "reward.dense", "reward.rubric")
ALL_KINDS: tuple[ShardKind, ...] = get_args(ShardKind)

# GPU-seconds per second of video. PLACEHOLDERS (order-of-magnitude guesses) until S2 measures
# them; pass ``rates=`` from rates.json once it exists.
DEFAULT_RATES: dict[ShardKind, float] = {
    "segcap": 3.0,
    "segment": 0.5,
    "caption": 1.0,
    "reward.terminal": 0.3,
    "reward.dense": 0.8,
    "reward.rubric": 1.5,
}


def stages_of(kind: ShardKind) -> tuple[Stage, ...]:
    """Record stages a shard kind emits (``segcap`` emits both segment and caption)."""
    return ("segment", "caption") if kind == "segcap" else (kind,)  # type: ignore[return-value]


def kinds_for(settings: RunSettings) -> tuple[ShardKind, ...]:
    """The shard kinds a job plans: those its pools configure (= its stages), pipeline order."""
    kinds = tuple(k for k in ALL_KINDS if k in settings.pools)
    if "caption" in kinds and "segment" not in kinds:
        raise ValueError("a caption pool needs a segment pool (captions are per segment)")
    if not kinds:
        raise ValueError("settings.pools is empty: nothing to plan")
    return kinds


def shard_id_for(kind: ShardKind, uids: Iterable[str]) -> str:
    """Content-addressed shard id: ``<kind>-<16 hex of sha256(kind, sorted uids)>``."""
    h = hashlib.sha256("\0".join((kind, *sorted(uids))).encode()).hexdigest()[:16]
    return safe_key(f"{kind}-{h}")


def source_group(idx: EpisodeIndex) -> tuple[str, ...]:
    """Grouping key: the episode's source files (one per camera)."""
    return tuple(sorted({s.source.uri for s in idx.streams.values()}))


def episode_duration_s(idx: EpisodeIndex) -> float:
    if not idx.streams:
        raise ValueError(f"episode {idx.dataset}/{idx.episode_id} has no streams")
    main = idx.streams.get("main")
    return (main or max(idx.streams.values(), key=lambda s: s.duration_us)).duration_s


def _pack(groups: list[list[EpisodeRef]], rate: float, target_s: float
          ) -> list[tuple[list[str], float]]:
    shards: list[tuple[list[str], float]] = []
    cur: list[str] = []
    cur_s = 0.0

    def close() -> None:
        nonlocal cur, cur_s
        if cur:
            shards.append((cur, cur_s))
        cur, cur_s = [], 0.0

    for group in groups:
        g_s = sum(e.duration_s * rate for e in group)
        if cur and cur_s + g_s > target_s:
            close()
        if g_s <= target_s:
            cur.extend(e.uid for e in group)
            cur_s += g_s
            continue
        # The group alone is too big: split it into consecutive shards.
        for e in group:
            e_s = e.duration_s * rate
            if cur and cur_s + e_s > target_s:
                close()
            cur.append(e.uid)
            cur_s += e_s
    close()
    return shards


def build_manifest(
    indexes: Iterable[EpisodeIndex],
    settings: RunSettings,
    *,
    created_utc: str,
    rates: Mapping[ShardKind, float] | None = None,
    pool: str | Mapping[str, str] = "human",
    index_refs: Mapping[str, str] | None = None,
) -> Manifest:
    """Plan a job. ``pool``: one pool for all episodes or uid → pool. ``index_refs``: uid → store
    path of its index receipt (default ``JobPaths.index(uid)``)."""
    rates = {**DEFAULT_RATES, **(rates or {})}
    paths = JobPaths(settings.job_id)
    target_s = settings.shard_target_gpu_minutes * 60.0
    if target_s <= 0:
        raise ValueError("shard_target_gpu_minutes must be positive")

    refs: list[tuple[tuple[str, ...], EpisodeRef]] = []
    seen: set[str] = set()
    for idx in indexes:
        uid = episode_uid(idx.dataset, idx.episode_id)
        if uid in seen:
            raise ValueError(f"duplicate episode uid {uid!r}")
        seen.add(uid)
        if settings.envelope == "enforce" and not idx.envelope.ok:
            continue
        ref = EpisodeRef(
            uid=uid, dataset=idx.dataset, episode_id=idx.episode_id,
            pool=pool if isinstance(pool, str) else pool[uid],
            duration_s=episode_duration_s(idx), oracle_eligible=idx.oracle_eligible,
            envelope_ok=idx.envelope.ok,
            index_ref=(index_refs or {}).get(uid) or paths.index(uid))
        refs.append((source_group(idx), ref))
    refs.sort(key=lambda r: (r[0], r[1].episode_id, r[1].uid))

    groups: list[list[EpisodeRef]] = []
    last: tuple[str, ...] | None = None
    for key, ref in refs:
        if key != last:
            groups.append([])
            last = key
        groups[-1].append(ref)

    shards: list[Shard] = []
    by_kind: dict[ShardKind, list[Shard]] = {}
    for kind in kinds_for(settings):
        rate = rates[kind]
        out = [Shard(shard_id=shard_id_for(kind, uids), kind=kind, episodes=tuple(uids),
                     est_gpu_seconds=round(est, 6))
               for uids, est in _pack(groups, rate, target_s)]
        if kind == "caption":
            owner = {u: s.shard_id for s in by_kind["segment"] for u in s.episodes}
            out = [replace(s, depends_on=tuple(dict.fromkeys(owner[u] for u in s.episodes)))
                   for s in out]
        by_kind[kind] = out
        shards.extend(out)

    kinds = kinds_for(settings)
    return Manifest(job_id=settings.job_id, config_hash=settings.config_hash,
                    episodes=tuple(r for _, r in refs), shards=tuple(shards),
                    created_utc=created_utc,
                    stage_hashes={k: settings.stage_hashes[k] for k in kinds
                                  if k in settings.stage_hashes},
                    rates={k: float(rates[k]) for k in kinds})


# --- JSON ----------------------------------------------------------------------------------

T = TypeVar("T")


def to_json(obj: Any) -> bytes:
    """Canonical JSON bytes of a job dataclass (sorted keys, no NaN)."""
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"not a dataclass instance: {obj!r}")
    return json.dumps(asdict(obj), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _decode(cls: type[T], d: dict[str, Any]) -> T:
    hints = typing.get_type_hints(cls)
    kw: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in d:
            continue
        v = d[f.name]
        hint = hints[f.name]
        if typing.get_origin(hint) is tuple and isinstance(v, list):
            (inner, *_rest) = get_args(hint) or (Any,)
            if is_dataclass(inner):
                v = tuple(_decode(inner, x) for x in v)  # type: ignore[arg-type]
            else:
                v = tuple(v)
        kw[f.name] = v
    return cls(**kw)


def from_json(cls: type[T], data: bytes | str) -> T:
    return _decode(cls, json.loads(data))


def manifest_to_json(m: Manifest) -> bytes:
    return to_json(m)


def manifest_from_json(data: bytes | str) -> Manifest:
    return from_json(Manifest, data)


def save_manifest(store: Store, m: Manifest) -> str:
    """Write the config's manifest (``JobPaths.config_manifest``), then the job-level pointer
    (``JobPaths.manifest``: the latest plan). Returns the per-config path."""
    paths = JobPaths(m.job_id)
    data = manifest_to_json(m)
    path = paths.config_manifest(m.config_hash)
    store.put_bytes(path, data)
    store.put_bytes(paths.manifest, data)
    return path


def load_manifest(store: Store, job_id: str, config_hash: str | None = None) -> Manifest:
    """The manifest of ``config_hash`` (default: the job's latest plan). A config planned before
    per-config manifests existed is found through the job-level file if it is that config's."""
    paths = JobPaths(job_id)
    if config_hash is None:
        return manifest_from_json(store.get_bytes(paths.manifest))
    p = paths.config_manifest(config_hash)
    if store.exists(p):
        return manifest_from_json(store.get_bytes(p))
    m = manifest_from_json(store.get_bytes(paths.manifest))
    if m.config_hash != config_hash:
        raise FileNotFoundError(f"job {job_id!r} has no manifest for config {config_hash!r} "
                                f"(its latest plan is {m.config_hash!r})")
    return m
