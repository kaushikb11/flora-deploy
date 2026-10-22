"""Plan a job from its episode index: the one planning path of ``sdk.submit`` (Modal) and
``local.runner.LocalRunner.plan`` (``flora run --local``), so both plan exactly alike (review 6
H3).

:func:`plan_job` reuses the config's stored manifest (``JobPaths.config_manifest``: one per
config hash, so adding a stage to a config re-plans under a new hash while the shards of the
unchanged kinds keep their stage hashes and are not redone), or builds one from the index
receipts (``jobs.indexing.load_indexes``) and saves it. Either way the job-level pointer
(``JobPaths.manifest``: what ``flora results`` defaults to) names this config.

Episodes left out of the plan are recorded in ``Manifest.excluded`` (uid -> reason) and count as
exclusions in the error-rate guard (``jobs.progress.with_unplanned``; review 6 M6):

* ``index_failed``: the index pass raised for the episode (no receipt);
* ``source_error``: the receipt says the source is unusable (missing file, bad range, ...);
* ``out_of_envelope``: outside NVIDIA's reference envelope under ``envelope: enforce``;
* ``missing_task``: the job has reward stages and the episode has no task text and no fixed
  instruction (``jobs.indexing.episode_task``). NVIDIA's reward prompts are built from the task,
  so it is excluded here, per episode, before any GPU time is spent (review 6 M3). The reward
  worker applies the same rule to any such episode a manifest still holds: an
  ``EpisodeError(kind="missing_task")``, never a failed shard.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from typing import Any

from flora_deploy.jobs.indexing import (
    MISSING_TASK,
    episode_task,
    load_indexes,
    load_receipts,
    missing_task_message,
    planned,
    receipt_task,
    summary_path,
)
from flora_deploy.jobs.manifest import (
    build_manifest,
    load_manifest,
    manifest_to_json,
    save_manifest,
)
from flora_deploy.jobs.settings import RunSettings
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import Manifest

REWARD_KINDS = ("reward.terminal", "reward.dense", "reward.rubric")
Log = Callable[[str], None]


class NothingToPlan(ValueError):
    """No indexed episode is left to plan."""


def needs_task(kinds: Iterable[str]) -> bool:
    return any(k in REWARD_KINDS for k in kinds)


def unplanned_episodes(store: Store, job_id: str, *, policy: str, require_task: bool,
                       default_task: str | None = None) -> dict[str, str]:
    """uid -> reason for every episode the index saw that the plan leaves out (module docs)."""
    out: dict[str, str] = {}
    receipts = load_receipts(store, job_id)
    have = {r["uid"] for r in receipts}
    p = summary_path(job_id)
    if store.exists(p):
        try:
            summary = json.loads(store.get_bytes(p))
        except ValueError:
            summary = {}
        for row in summary.get("failed", ()):
            if row.get("uid") and row["uid"] not in have:
                out[row["uid"]] = f"index_failed: {row.get('error') or 'error'}"
    for rec in receipts:
        uid = rec["uid"]
        if rec.get("status") != "indexed":
            out[uid] = f"source_error: {rec.get('error_kind') or rec.get('error') or 'error'}"
            continue
        env = rec.get("envelope") or {}
        if not planned({"status": "indexed", "envelope_ok": env.get("ok")}, policy):
            reasons = ", ".join(env.get("reasons") or ()) or "outside the envelope"
            out[uid] = f"out_of_envelope: {reasons}"
            continue
        if require_task and episode_task(receipt_task(rec), default_task) is None:
            out[uid] = f"{MISSING_TASK}: {missing_task_message(uid)}"
    return dict(sorted(out.items()))


def plan_job(store: Store, settings: RunSettings, *, envelope: str, created_utc: str,
             default_task: str | None = None, log: Log | None = None) -> Manifest:
    """The manifest of ``settings.config_hash`` (stored, or planned now). Raises
    :class:`NothingToPlan` when no indexed episode is left."""
    say = log or (lambda m: None)
    job_id = settings.job_id
    paths = JobPaths(job_id)
    if store.exists(paths.config_manifest(settings.config_hash)):
        m = load_manifest(store, job_id, settings.config_hash)
        store.put_bytes(paths.manifest, manifest_to_json(m))  # the latest plan is this one
        say(f"plan: reusing the stored manifest ({len(m.shards)} shards)")
        return m
    kinds = tuple(settings.pools)
    excluded = unplanned_episodes(store, job_id, policy=envelope,
                                  require_task=needs_task(kinds), default_task=default_task)
    loaded = load_indexes(store, job_id, policy=envelope)
    keep = [i for i, ref in zip(loaded.indexes, _uids(loaded.indexes), strict=True)
            if ref not in excluded]
    if not keep:
        why = (f" ({len(excluded)} excluded: "
               + ", ".join(f"{u}: {r}" for u, r in list(excluded.items())[:5]) + ")"
               if excluded else "")
        raise NothingToPlan(f"job {job_id!r} has no indexed episodes to plan{why}")
    m = build_manifest(keep, settings, created_utc=created_utc, pool=loaded.pools)
    m = dataclasses.replace(m, excluded=excluded)
    save_manifest(store, m)
    by_kind: dict[str, int] = {}
    for s in m.shards:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
    say(f"plan: {len(m.episodes)} episodes, {len(m.shards)} shards ("
        + ", ".join(f"{k} {n}" for k, n in by_kind.items()) + ")"
        + (f"; {len(excluded)} excluded" if excluded else ""))
    for uid, reason in list(excluded.items())[:10]:
        say(f"  excluded {uid}: {reason}")
    return m


def _uids(indexes: Iterable[Any]) -> list[str]:
    from flora_deploy.jobs.types import episode_uid

    return [episode_uid(i.dataset, i.episode_id) for i in indexes]
