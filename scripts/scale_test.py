"""Scale test of ``balanced`` (``docs/scale-test.md``): several containers per pool, through the
real ``flora index -> plan -> deploy -> run -> report`` path.

Usage::

    uv run python scripts/scale_test.py                          # dry run (the default): plan
    uv run python scripts/scale_test.py --sizing                 # + the size options
    uv run python scripts/scale_test.py --mode local --execute   # local fake run (no Modal)
    uv run python scripts/scale_test.py --mode fake-modal --yes-spend  # CPU rehearsal (~cents)
    uv run python scripts/scale_test.py --mode real --yes-spend  # the GPU run (resumable)
    uv run python scripts/scale_test.py --mode real --report     # rewrite the report
    uv run python scripts/scale_test.py --finalize [--write]     # fold its rates into rates.json

Profiles (``--profile``, :data:`PROFILES`):

* ``lite`` (the default): the first 32 episodes (0.223 video-h), ``segment`` on 2 x L40S and
  ``caption`` on 2 x H100!, no chaos. It measures each pool's throughput with two containers
  (rate samples for ``--finalize``), pipelining, the job-end drain, the caption pre-warm, the
  compile cache per shard, batch fill, and spend against the projection;
* ``full``: all 128 episodes (0.980 video-h), 2 x L40S + 3 x H100!, three chaos actions.

Modes (one state file each, ``reports/scale-test/<mode>/state.json`` for ``full`` and
``reports/scale-test/<profile>/<mode>/state.json`` otherwise):

* ``real``: the pinned EgoDex subset (:data:`DATASET`) at the profile's size, ``preset:
  balanced``;
* ``fake-modal``: the same job shape on Modal with ``flora deploy/run --fake-models`` (CPU fake
  workers, the real factory app, orchestrator, Volumes and reconciler) over :data:`FAKE_EPISODES`
  small synthetic episodes with the same total video length, so the manifest has the same number
  of shards;
* ``local``: ``flora run --local --fake-models`` over the same synthetic dataset. No Modal at
  all; it validates this script's control flow, polling, chaos actions and report.

Steps, in order (a stop rule before each and a spend check after each):

1. ``verify``: the dataset pin, by one Hub metadata call (:func:`verify_dataset`); synthetic
   modes write the synthetic dataset instead;
2. ``index``: ``flora index`` (real: on Modal CPU, staged to the media Volume; fake modes:
   ``flora index --local``, mirrored into the job store as H0 does);
3. ``plan``: ``flora plan --index`` (its options are printed and kept) and this script's own
   projection (:func:`project`);
4. ``deploy``: ``flora deploy`` with the reconciler cron on (``local``: nothing to deploy);
5. ``run``: ``flora run --detach``, then live polling (``flora status --json`` and ``modal
   container list --json`` every ``--poll`` seconds) with three chaos actions (:class:`Chaos`):
   one ``segment`` and one ``caption`` preemption mid-shard (``modal container stop``: SIGINT,
   inputs rescheduled; fake modes: the fake workers' fault file / ``--inject``), and one
   orchestrator kill at half the captions (the reconciler cron must revive it; ``local``: the
   process is killed and ``flora resume`` restarts it). Then ``flora compose`` (timed) and
   ``modal app stop``;
6. ``report``: ``flora status`` and ``flora report`` (timed), measurements from the job store
   and the container samples (:func:`measure`), the success criteria (:func:`evaluate`) and
   ``reports/scale-test/<mode>/report.md``.

**Spend** is accounted exactly as in ``scripts/wave_h.py`` (its :class:`Attempt` /
:class:`State` / heartbeat / billing machinery is reused): every attempt counts, a killed or
unreported attempt is charged its budget until billing is an hour past it, and the global figure
is ``max(sum of charged attempts, billed delta of every app seen)``. **Stop rules:** before a
step its **upper** projection must fit its cap and the global cap left; during ``run`` the job
is cancelled when the observed spend (``flora status`` or the container samples, whichever is
larger) plus the cancel slack reaches the step budget, or at the upper wall time; after a step
its spend is checked against its cap and the global cap.

Nothing is spent without ``--yes-spend`` (``local`` spends nothing; it runs with ``--execute``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _runbook as rb  # noqa: E402 - the step loop, decisions, stop by id (shared)
import h0_rehearsal as h0  # noqa: E402 - staging helpers (upload, mirror, sh)
import wave_h as wh  # noqa: E402 - spend accounting, stop rules, billing

MODES = ("real", "fake-modal", "local")
STEPS = ("verify", "index", "plan", "deploy", "run", "report")
TITLES = {"verify": "1. dataset pin / synthetic dataset", "index": "2. flora index",
          "plan": "3. flora plan + projection", "deploy": "4. flora deploy",
          "run": "5. flora run + live polling + chaos", "report": "6. status / report / measure"}

# ---------------------------------------------------------------------------------------------
# The dataset (verified 2026-10-22 against the Hub; docs/scale-test.md "Dataset")

DATASET: dict[str, Any] = {
    "repo": "griffinlabs/EgoDex-LeRobot-v3.0",
    "revision": "41d60b449629b2181ff5b735d31c2a2cf8b3cad8",  # = the templates' pin
    "path": "train/wash_put_away_dishes",
    "format": "lerobot_v3",
    "embodiment": "human",
    "license": "CC-BY-NC-ND-4.0",
    "codebase_version": "v3.0",
    "total_episodes": 128,
    "total_frames": 105877,
    "fps": 30,
    "video_keys": ["observation.images.camera"],
    "codec": "av1",
    "resolution": "1920x1080",
    # git blob sha1 of <path>/meta/info.json at the revision (the Hub's ETag / X-Linked-Etag)
    "info_blob_sha1": "b1b624b8542f8768c64b4bfc22d679b60a29a27d",
    "episode_s": {"min": 3.07, "median": 13.7, "max": 78.33},  # from meta/episodes (5.5 MB)
    "bytes": 2_985_280_594,  # 13 concatenated AV1 MP4s (2.43 GB) + parquet
}
FULL_VIDEO_S = DATASET["total_frames"] / DATASET["fps"]  # 3529.2 s = 0.980 video-h
HF = "https://huggingface.co"

# Every episode's frame count and source MP4 (``videos/<key>/chunk-000/file-<i>.mp4``), in
# episode-index order, from ``meta/episodes/chunk-000/file-000.parquet`` at the pinned revision
# (git blob sha1 below; fetched 2026-10-22). They give the subsets' exact video length and
# shard packing offline; ``index`` checks the indexed length against them.
EPISODES_BLOB_SHA1 = "d58e8fbd29082d252097ac14416e4ef3bdc95b22"
EPISODE_FRAMES: tuple[int, ...] = (
    1199, 313, 253, 1633, 211, 1209, 334, 260, 185, 1600, 349, 1134, 212, 1208, 1231, 1370,
    1514, 215, 350, 1361, 257, 266, 1481, 930, 297, 1644, 276, 217, 334, 288, 335, 1584,
    284, 1372, 266, 324, 1595, 1771, 201, 1797, 1165, 256, 273, 1127, 242, 224, 1775, 1526,
    228, 1570, 316, 1085, 320, 1496, 1218, 411, 264, 1352, 269, 241, 2350, 335, 241, 1711,
    272, 331, 1510, 1889, 251, 305, 251, 252, 221, 1018, 314, 287, 1518, 1482, 258, 347,
    889, 2140, 266, 263, 1345, 1130, 237, 1362, 243, 1147, 1708, 268, 236, 92, 1371, 233,
    199, 1390, 1909, 332, 1681, 232, 2148, 243, 1167, 1462, 251, 264, 268, 906, 1236, 249,
    218, 1685, 1327, 1163, 1180, 1485, 852, 999, 1123, 302, 1216, 1076, 1072, 217, 1708, 1226)
EPISODE_FILE: tuple[int, ...] = (
    *[0] * 13, *[1] * 9, *[2] * 11, *[3] * 10, *[4] * 10, *[5] * 10, *[6] * 13, *[7] * 9,
    *[8] * 12, *[9] * 7, *[10] * 11, *[11] * 8, *[12] * 5)


def subset_video_s(n: int) -> float:
    """Video seconds of the first ``n`` episodes (by episode index)."""
    return sum(EPISODE_FRAMES[:n]) / DATASET["fps"]


# ---------------------------------------------------------------------------------------------
# Profiles: which episodes, which pools, which chaos


@dataclass(frozen=True)
class Profile:
    """One size of the test. ``episodes``: the first N by episode index (``flora index --limit
    N``; every episode for ``full``). ``packing``: ``exact`` packs the real mode's shards from
    the pinned episode table as ``jobs.manifest`` does; ``even`` splits the video into equal
    shards (``full`` keeps it, so its documented numbers stay comparable)."""

    name: str
    episodes: int
    pools: Mapping[str, Mapping[str, Any]]
    shard_target_gpu_min: float
    chaos: tuple[str, ...]
    packing: str = "even"
    blurb: str = ""

    @property
    def video_s(self) -> float:
        return subset_video_s(self.episodes)

    @property
    def source_files(self) -> int:
        return len(set(EPISODE_FILE[:self.episodes]))

    @property
    def tag(self) -> str:
        """Prefix of job ids and project names (``full`` keeps the original ``scale-<run>``)."""
        return "" if self.name == "full" else f"{self.name}-"

    def with_size(self, episodes: int | None = None, pools: Mapping[str, int] | None = None
                  ) -> Profile:
        """This profile at another episode count / pool sizes (the options table)."""
        p = {k: {**v, "max_containers": int((pools or {}).get(k, v["max_containers"]))}
             for k, v in self.pools.items()}
        return dataclasses.replace(self, episodes=episodes or self.episodes, pools=p)


ALL_CHAOS = ("preempt_segment", "preempt_caption", "kill_orchestrator")
PROFILES: dict[str, Profile] = {
    # 32 episodes (0.223 video-h; 18 under 15 s and 14 of 30 s or more), 2 + 2 containers, no
    # chaos: about a quarter of the full test's money, enough to measure every pool with two
    # containers (docs/scale-test.md "Lite profile"). 2 GPU-minute shards: ~4 segment and ~8
    # caption shards, so each container runs at least two and captions start while segment
    # still runs.
    "lite": Profile("lite", 32, {"segment": {"gpu": "L40S", "max_containers": 2},
                                 "caption": {"gpu": "H100!", "max_containers": 2}},
                    shard_target_gpu_min=2.0, chaos=(), packing="exact",
                    blurb="first 32 episodes, 2 + 2 containers, no chaos"),
    # 2 + 3: the cheapest size with >= 2 containers per pool whose upper wall is within the hour
    # at the measured rates (docs/scale-test.md "Pool sizing"; :func:`pool_sizing`,
    # :func:`recommend`). caption (0.906 container-s per video-s, 2.5x segment) is the
    # bottleneck: a third segment container only adds a cold start and a drained idle tail.
    # 4 GPU-minute shards at the manifest's packing rates (segment 0.5, caption 1.0 GPU-s per
    # video-s): ~480 video-s per segment shard (~8 shards), ~240 per caption shard (~15).
    "full": Profile("full", DATASET["total_episodes"],
                    {"segment": {"gpu": "L40S", "max_containers": 2},
                     "caption": {"gpu": "H100!", "max_containers": 3}},
                    shard_target_gpu_min=4.0, chaos=ALL_CHAOS, packing="even",
                    blurb="all 128 episodes, 2 + 3 containers, three chaos actions"),
}
DEFAULT_PROFILE = "lite"

# The active profile's values, rebound by :func:`use_profile` (``--profile``; the tests switch
# with it too). Functions read them at call time.
PROFILE: Profile = PROFILES[DEFAULT_PROFILE]
POOLS: dict[str, dict[str, Any]] = {}
VIDEO_S = 0.0
SHARD_TARGET_GPU_MIN = 0.0
CHAOS: tuple[str, ...] = ()
FAKE_EPISODES = 0
INDEX_UNITS = 0

# Synthetic stand-in (fake modes): media.synth single-view clips, 3..6 s (mean 4.5 s), so the
# total matches VIDEO_S and the manifest packs to the same number of shards.
FAKE_MIN_S, FAKE_MAX_S = 3.0, 6.0


def use_profile(p: Profile | str) -> Profile:
    """Make ``p`` the active profile; returns the previous one."""
    global PROFILE, POOLS, VIDEO_S, SHARD_TARGET_GPU_MIN, CHAOS, FAKE_EPISODES, INDEX_UNITS
    prev = PROFILE
    PROFILE = PROFILES[p] if isinstance(p, str) else p
    POOLS = {k: dict(v) for k, v in PROFILE.pools.items()}
    VIDEO_S = PROFILE.video_s
    SHARD_TARGET_GPU_MIN = PROFILE.shard_target_gpu_min
    CHAOS = tuple(PROFILE.chaos)
    FAKE_EPISODES = round(VIDEO_S / ((FAKE_MIN_S + FAKE_MAX_S) / 2))  # full: 784, lite: 178
    INDEX_UNITS = PROFILE.source_files  # one per concatenated source MP4 (full: 13, lite: 3)
    return prev


@contextlib.contextmanager
def profile_ctx(p: Profile | str):
    prev = use_profile(p)
    try:
        yield PROFILE
    finally:
        use_profile(prev)


use_profile(DEFAULT_PROFILE)

# ---------------------------------------------------------------------------------------------
# Job shape

def max_gpus() -> int:
    """``modal.max_gpus``: every pool at its size at once (full 5, lite 4)."""
    return sum(int(v["max_containers"]) for v in POOLS.values())


STORE_VOLUME = {"real": "flora-scale", "fake-modal": "flora-scale-fake"}
MEDIA_VOLUME = {"real": "flora-scale-media", "fake-modal": "flora-scale-fake-media"}

# ---------------------------------------------------------------------------------------------
# Timing constants (seconds)

IDLE_AFTER_JOB_S = 30  # last receipt -> compose -> `modal app stop`
CLIENT_SLACK_S = wh.CLIENT_SLACK_S  # 240: deadline -> cancel -> app stopped, per container
RECONCILE_WAIT_S = 660  # the reconciler cron runs every 10 min; then `flora resume` fallback
LOCAL_LEASE_TTL_S = 35.0  # local.runner lease TTL (30 s) + margin
POLL_S = {"real": 30.0, "fake-modal": 10.0, "local": 1.0}
INDEX_UNITS = 13  # one per concatenated source MP4
INDEX_UNIT_S = (120.0, 600.0)  # nominal / upper per unit (download 200 MB + probe); PLACEHOLDER
# fake-modal CPU workers (PLACEHOLDERS: not measured at this shape)
FAKE_RATE = (0.05, 3.0)  # container-s per video-s, relative uncertainty
FAKE_STARTUP = (45.0, 1.0)  # startup s, relative uncertainty

# Caps (USD). Never hard-coded: each step's cap is its own upper projection (:func:`project`,
# at today's pinned prices and rates) rounded up to $0.10, and the global cap is their sum
# (:func:`global_cap`); the plan prints both for the user to approve before the first
# --yes-spend. A price change moves the caps with the projection instead of breaking the stop
# rules. ``CAPS_USD[mode]`` overrides them (tests; an explicitly lower cap).
CAP_STEP_USD = 0.10
CAPS_USD: dict[str, dict[str, float]] = {}


def caps(mode: str) -> dict[str, float]:
    if mode in CAPS_USD:
        return dict(CAPS_USD[mode])
    if mode == "local":
        return dict.fromkeys(STEPS, 0.0)
    return {s: round(math.ceil(p.usd_upper / CAP_STEP_USD - 1e-9) * CAP_STEP_USD, 2)
            for s, p in projections(mode, warm=False).items()}


def global_cap(mode: str) -> float:
    """The global cap: the sum of the step caps."""
    return round(sum(caps(mode).values()), 2)


# ---------------------------------------------------------------------------------------------
# Dataset pin (one Hub call; never in the dry run unless --verify)


def info_url(d: Mapping[str, Any] = DATASET) -> str:
    return f"{HF}/datasets/{d['repo']}/resolve/{d['revision']}/{d['path']}/meta/info.json"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def check_info(raw: bytes, d: Mapping[str, Any] = DATASET) -> list[str]:
    """Problems with ``meta/info.json`` at the pinned revision (empty: the pin holds)."""
    out = []
    if git_blob_sha1(raw) != d["info_blob_sha1"]:
        out.append(f"info.json blob {git_blob_sha1(raw)} != pinned {d['info_blob_sha1']}")
    i = json.loads(raw)
    for k in ("codebase_version", "total_episodes", "total_frames", "fps"):
        if i.get(k) != d[k]:
            out.append(f"{k}: {i.get(k)!r} != {d[k]!r}")
    vids = sorted(k for k, v in i.get("features", {}).items() if v.get("dtype") == "video")
    if vids != d["video_keys"]:
        out.append(f"video keys {vids} != {d['video_keys']}")
    for k in vids:
        codec = i["features"][k].get("info", {}).get("video.codec")
        if codec != d["codec"]:
            out.append(f"{k} codec {codec!r} != {d['codec']!r}")
    return out


def verify_dataset(fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """One GET of ``meta/info.json`` at the full revision SHA: a 404 means the revision or path
    is gone; a content-addressed match (git blob sha1) means it is unchanged."""
    if fetch is None:
        import urllib.request

        def fetch(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read()

    raw = fetch(info_url())
    problems = check_info(raw)
    return {"ok": not problems, "problems": problems, "url": info_url(),
            "video_s": round(FULL_VIDEO_S, 1), "profile": PROFILE.name,
            "subset": {"episodes": PROFILE.episodes, "video_s": round(VIDEO_S, 2)},
            "tracked_usd": 0.0, "apps": []}


# ---------------------------------------------------------------------------------------------
# Projection (pure)


def pools_for(mode: str) -> dict[str, tuple[wh.Container, int]]:
    """kind -> (container shape, max_containers) as each mode deploys it."""
    from flora_deploy.config.presets import get_preset

    out = {}
    for kind, p in POOLS.items():
        if mode == "real":
            pool = get_preset("balanced").pools[kind]
            c = wh.Container(p["gpu"].rstrip("!"), float(pool.cpu_cores),
                             float(pool.memory_gib))
        elif mode == "fake-modal":
            from flora_deploy.modal_app.factory import FAKE_WORKER_CPU, FAKE_WORKER_MEMORY_GIB

            c = wh.Container(None, FAKE_WORKER_CPU, FAKE_WORKER_MEMORY_GIB)
        else:
            c = wh.Container(None, 0.0, 0.0)  # local threads: no Modal money
        out[kind] = (c, int(p["max_containers"]))
    return out


CPU_CLASS = "cpu"  # the app's CPU functions: orchestrator_run, reconcile, reconcile_watched


def container_rates(mode: str) -> dict[str, float]:
    """$/s of one container per class, at ``jobs.budget``'s pinned list prices: each pool's
    kind as deployed in ``mode``, and :data:`CPU_CLASS` for the orchestrator and reconciler
    (``modal_app.orchestrator_app`` resources, priced as ``jobs.pools`` deploys them; any
    container no attempt marker names is priced as one of these)."""
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.modal_app import orchestrator_app as oa

    if mode == "local":
        return dict.fromkeys((*POOLS, CPU_CLASS), 0.0)
    r = pinned_rates()
    out = {k: r.per_second(c.gpu, 1 if c.gpu else 0, c.cpu, c.memory_gib)
           for k, (c, _) in pools_for(mode).items()}
    from flora_deploy.jobs.pools import ORCHESTRATOR_PREEMPTIBLE

    out[CPU_CLASS] = r.per_second(None, 0, float(oa.CPU_CORES), oa.MEMORY_MIB / 1024,
                                  preemptible=ORCHESTRATOR_PREEMPTIBLE)
    return out


@dataclass(frozen=True)
class KindRate:
    cs_per_video_s: float
    uncertainty: float
    startup_s: float
    startup_unc: float
    status: str
    startup_status: str = "placeholder"
    startup_cache: str | None = None  # "cold" | "warm" (compile cache) | None


# Warm start (a compile cache seeded on ``flora-cache`` before the run). Until rates.json has a
# ``compile_cache: warm`` startup row for a balanced kind, the warm variant assumes the cold row
# times the Rewarder's measured warm/cold ratio on the same vLLM image and H100! (fresh
# container, warm compile cache + prefetch 229 s vs cold cache 318 s: rates.json `reward.*`
# startup, docs/cold-start.md), at +/-50%. An assumption, not a measurement.
WARM_STARTUP_FACTOR = 229.0 / 318.0  # 0.72
WARM_STARTUP_UNC = 0.5
ASSUME_WARM = False  # --assume-warm: price (and print) the warm-start variant


def startup_row(table: Any, kind: str, cache: str) -> Any | None:
    """The balanced ``kind`` startup row timed with this compile cache (``cold`` also takes a
    row that does not say), else None; a non-placeholder beats a placeholder."""
    rows = [e for e in table.startup if e.preset == "balanced" and e.kind == kind
            and (e.compile_cache == cache or (cache == "cold" and e.compile_cache is None))]
    rows.sort(key=lambda e: e.status == "placeholder")
    return rows[0] if rows else None


def kind_rates(mode: str, table: Any = None, *, warm: bool | None = None,
               warm_kinds: tuple[str, ...] = ()) -> dict[str, KindRate]:
    """Per pool kind: the work rate and the startup the projection uses. ``warm``: the
    warm-start variant for every pool (default :data:`ASSUME_WARM`); ``warm_kinds``: pools
    priced warm because a compile cache is published for their key
    (``engines.compile_cache.published_kinds``). The others are priced cold."""
    from flora_deploy.planner.rates import RateTable

    warm = ASSUME_WARM if warm is None else warm
    if mode != "real":
        return {k: KindRate(FAKE_RATE[0], FAKE_RATE[1], FAKE_STARTUP[0], FAKE_STARTUP[1],
                            "placeholder", "placeholder") for k in POOLS}
    t = table or RateTable.load()
    out = {}
    for kind in POOLS:
        e = t.rate("balanced", kind, 1, 1)
        cold = startup_row(t, kind, "cold") or t.startup_for("balanced", kind)
        s, su, sst, cache = cold.startup_s, cold.uncertainty, cold.status, cold.compile_cache
        if warm or kind in warm_kinds:
            w = startup_row(t, kind, "warm")
            if w is not None:
                s, su, sst, cache = w.startup_s, w.uncertainty, w.status, "warm"
            else:
                s, su = cold.startup_s * WARM_STARTUP_FACTOR, WARM_STARTUP_UNC
                sst, cache = "assumed", "warm"
        out[kind] = KindRate(e.container_s_per_video_s, e.uncertainty, s, su, e.status, sst,
                             cache)
    return out


def shard_video_s(kind: str) -> float:
    """Video seconds per shard at the manifest's packing rate (``jobs.manifest``)."""
    from flora_deploy.jobs.manifest import DEFAULT_RATES

    return SHARD_TARGET_GPU_MIN * 60.0 / DEFAULT_RATES[kind]


def _even_counts(video_s: float) -> dict[str, int]:
    return {k: max(1, math.ceil(video_s / shard_video_s(k))) for k in POOLS}


ShardPlan = dict[str, list[Any]]  # segment: [video_s]; caption: [(video_s, (segment idx, ...))]


def exact_plan(episodes: int, target_gpu_min: float, *, scale: float = 1.0) -> ShardPlan:
    """The shards ``jobs.manifest.build_manifest`` packs from the first ``episodes`` episodes of
    the pinned table: grouped by source MP4, in episode order, at the manifest's packing rates;
    each caption shard depends on the segment shards that own its episodes. ``scale``: stretch
    every duration (the indexed length against the table's)."""
    from types import SimpleNamespace

    from flora_deploy.jobs.manifest import DEFAULT_RATES, _pack

    fps = DATASET["fps"]
    groups: dict[int, list[Any]] = {}
    for i in range(episodes):
        groups.setdefault(EPISODE_FILE[i], []).append(
            SimpleNamespace(uid=f"episode_{i:06d}", duration_s=EPISODE_FRAMES[i] / fps * scale))
    ordered = [groups[f] for f in sorted(groups)]
    dur = {e.uid: e.duration_s for g in ordered for e in g}
    seg = _pack(ordered, DEFAULT_RATES["segment"], target_gpu_min * 60.0)
    owner = {u: j for j, (uids, _) in enumerate(seg) for u in uids}
    cap = _pack(ordered, DEFAULT_RATES["caption"], target_gpu_min * 60.0)
    return {"segment": [sum(dur[u] for u in uids) for uids, _ in seg],
            "caption": [(sum(dur[u] for u in uids), tuple(dict.fromkeys(owner[u] for u in uids)))
                        for uids, _ in cap]}


def even_plan(video_s: float) -> ShardPlan:
    """Equal shards at the profile's shard target (caption shard ``j`` depends on the segment
    shards its video span overlaps)."""
    n = _even_counts(video_s)
    seg_v, cap_v = video_s / n["segment"], video_s / n["caption"]
    cap = []
    for j in range(n["caption"]):
        lo = int(j * cap_v // seg_v)
        hi = min(n["segment"] - 1, int(((j + 1) * cap_v - 1e-6) // seg_v))
        cap.append((cap_v, tuple(range(lo, hi + 1))))
    return {"segment": [seg_v] * n["segment"], "caption": cap}


def shard_plan(video_s: float | None = None, mode: str = "real") -> ShardPlan:
    """The active profile's shards: ``exact`` (real mode, pinned table) or ``even``."""
    v = VIDEO_S if video_s is None else video_s
    if PROFILE.packing == "exact" and mode == "real":
        return exact_plan(PROFILE.episodes, SHARD_TARGET_GPU_MIN, scale=v / PROFILE.video_s)
    return even_plan(v)


def shard_counts(video_s: float | None = None, mode: str = "real") -> dict[str, int]:
    p = shard_plan(video_s, mode)
    return {k: len(p[k]) for k in POOLS}


def pool_sizes(sizes: Mapping[str, int] | None = None) -> dict[str, int]:
    return {k: int((sizes or {}).get(k, POOLS[k]["max_containers"])) for k in POOLS}


@dataclass
class PoolSim:
    """One pool in :func:`simulate`: its containers' lives and work."""

    starts: int = 0  # containers started (cold or warm)
    startup_s: float = 0.0
    busy_s: float = 0.0
    wait_s: float = 0.0  # ready but idle between shards (waiting for upstream shards)
    tail_s: float = 0.0  # after its last shard: the 300 s window, or the drain at the pool end
    container_s: float = 0.0
    peak: int = 0
    t_peak: float | None = None  # first time the pool had its size of containers started
    first_start: float = 0.0  # first shard start
    first_end: float = 0.0  # first shard end
    last_end: float = 0.0

    @property
    def utilization(self) -> float | None:
        return round(self.busy_s / self.container_s, 3) if self.container_s else None

    @property
    def utilization_working(self) -> float | None:
        """busy / (busy + waits between shards): utilization while the pool holds work."""
        w = self.busy_s + self.wait_s
        return round(self.busy_s / w, 3) if w else None


def _schedule(inputs: Sequence[tuple[float, float]], n: int, startup: float,
              scaledown: float, *, prewarm_at: float | None = None
              ) -> tuple[list[list[Any]], list[tuple[float, float]]]:
    """FIFO inputs ``(ready, seconds)`` on a pool of at most ``n`` containers that each start
    in ``startup`` s and scale down ``scaledown`` s after their last input; a queued input
    starts a new container while the pool is below ``n`` and none is free. ``prewarm_at``: one
    container is spawned then, before any input (the caption pre-warm, ``jobs.pools``).
    Returns the containers ``[spawn, ready, free, end]`` (end None while alive) and the input
    spans, in input order."""
    conts: list[list[Any]] = []
    spans: list[tuple[float, float]] = []
    if prewarm_at is not None and n > 0:
        conts.append([prewarm_at, prewarm_at + startup, 0.0, None])
    for r, d in inputs:  # ready times are non-decreasing (FIFO)
        for c in conts:  # containers idle past the window by now are gone
            if c[3] is None and c[2] > 0 and r > c[2] + scaledown:
                c[3] = c[2] + scaledown
        alive = [c for c in conts if c[3] is None]
        free = min((max(c[1], c[2]) for c in alive), default=math.inf)
        if len(alive) < n and free > r:
            conts.append([r, r + startup, 0.0, None])
            alive.append(conts[-1])
        c = min(alive, key=lambda c: max(c[1], c[2], r))
        t0 = max(c[1], c[2], r)
        c[2] = t0 + d
        spans.append((t0, t0 + d))
    return conts, spans


# The deployed pool behaviour the schedule models (``jobs.pools``): once a pool has no work
# left, the orchestrator drains it within one poll to a 2 s window, so each of its containers
# idles at most ``planner.options.POOL_END_TAIL_S`` (15 + 2 s, what ``flora plan`` charges)
# past the pool's last shard; and the caption pool is pre-warmed so its first container is
# ready when its first shard becomes runnable (skipped when ``min_containers`` > 0).
MIN_CONTAINERS = 0  # `flora deploy --min-containers` for this run (the default)


def simulate(sizes: Mapping[str, int], rates: Mapping[str, KindRate],
             video_s: float | None = None, *, upper: bool = False,
             prewarm: bool = MIN_CONTAINERS == 0, mode: str = "real",
             plan: ShardPlan | None = None) -> dict[str, Any]:
    """Shard-level schedule of the run without chaos: every ``segment`` shard is queued at
    t=0, a ``caption`` shard is queued once the segment shards covering its episodes have
    receipts (per-shard ``depends_on``), each pool autoscales up to its size, a container
    idle between shards scales down ``WORKER_SCALEDOWN_S`` after its last shard, every
    container still up when its pool's last shard ends leaves ``POOL_END_TAIL_S`` later (the
    job-aware drain), and the app stops :data:`IDLE_AFTER_JOB_S` after the last receipt.
    ``prewarm``: one caption container is spawned a caption startup before the first caption
    shard is runnable (a perfect ready estimate; it is charged from its spawn). ``upper``:
    rates and startups at ``1 + uncertainty``. ``plan``: the shards (default
    :func:`shard_plan` of the active profile). Returns ``{"wall_s", "pools": {kind:
    PoolSim}, "pipelining"}``."""
    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S
    from flora_deploy.planner.options import POOL_END_TAIL_S

    def rate(k: str) -> float:
        return rates[k].cs_per_video_s * ((1 + rates[k].uncertainty) if upper else 1.0)

    def start(k: str) -> float:
        return rates[k].startup_s * ((1 + rates[k].startup_unc) if upper else 1.0)

    plan = plan or shard_plan(video_s, mode)
    seg_c, seg_sp = _schedule([(0.0, v * rate("segment")) for v in plan["segment"]],
                              sizes["segment"], start("segment"), WORKER_SCALEDOWN_S)
    seg_end = [e for _, e in seg_sp]
    cap_in = [(max(seg_end[i] for i in deps), v * rate("caption"))
              for v, deps in plan["caption"]]
    cap_in.sort(key=lambda x: x[0])
    pre_at = max(0.0, cap_in[0][0] - start("caption")) if prewarm and cap_in else None
    cap_c, cap_sp = _schedule(cap_in, sizes["caption"], start("caption"), WORKER_SCALEDOWN_S,
                              prewarm_at=pre_at)
    wall = max(e for _, e in seg_sp + cap_sp)
    stop = wall + IDLE_AFTER_JOB_S
    out: dict[str, Any] = {"wall_s": wall, "pools": {}}
    for kind, conts, spans in (("segment", seg_c, seg_sp), ("caption", cap_c, cap_sp)):
        p = PoolSim()
        drained = max(e for _, e in spans) + POOL_END_TAIL_S
        ends = [min(c[3] if c[3] is not None else min(c[2] + WORKER_SCALEDOWN_S, drained),
                    stop) for c in conts]
        for c, end in zip(conts, ends, strict=True):
            p.starts += 1
            p.startup_s += c[1] - c[0]
            p.container_s += end - c[0]
            p.tail_s += end - c[2]
        p.busy_s = sum(e - s for s, e in spans)
        p.wait_s = max(0.0, p.container_s - p.startup_s - p.busy_s - p.tail_s)
        cur = 0
        for t, d in sorted([(c[0], 1) for c in conts] + [(e, -1) for e in ends]):
            cur += d
            p.peak = max(p.peak, cur)
            if p.t_peak is None and cur >= sizes[kind]:
                p.t_peak = t
        p.first_start = min(s for s, _ in spans)
        p.first_end = min(e for _, e in spans)
        p.last_end = max(e for _, e in spans)
        out["pools"][kind] = p
    seg, cap = out["pools"]["segment"], out["pools"]["caption"]
    out["pipelining"] = {
        "first_caption_after_first_segment_s": cap.first_start - seg.first_end,
        "overlap_s": max(0.0, min(seg.last_end, cap.last_end)
                         - max(seg.first_start, cap.first_start))}
    return out


def run_wall_s(mode: str, video_s: float | None = None, *, upper: bool = False,
               table: Any = None, warm: bool | None = None,
               sizes: Mapping[str, int] | None = None) -> float:
    """Wall seconds of the run: the shard schedule (:func:`simulate`) plus, when the profile
    kills the orchestrator, the reconciler gap (half the cron period nominal, the whole wait
    upper)."""
    sim = simulate(pool_sizes(sizes), kind_rates(mode, table, warm=warm), video_s, upper=upper,
                   mode=mode)
    gap = 0.0
    if "kill_orchestrator" in CHAOS:
        gap = RECONCILE_WAIT_S if upper else RECONCILE_WAIT_S / 2
    return sim["wall_s"] + gap


def project(step: str, mode: str, video_s: float | None = None, table: Any = None, *,
            warm: bool | None = None, sizes: Mapping[str, int] | None = None
            ) -> wh.Projection:
    """What ``step`` costs in ``mode`` (nominal and upper; see ``docs/scale-test.md``), for
    the active profile. ``warm``: the warm-start variant; ``sizes``: pool sizes other than
    :data:`POOLS`. Chaos items are charged only for the profile's chaos actions."""
    video_s = VIDEO_S if video_s is None else video_s
    p = wh.Projection(step)
    if step in ("verify", "plan", "report") or mode == "local":
        return p
    if step == "index":
        if mode == "real":
            c = wh.Container(None, 1.0, 2.0)  # modal_app.index_app units
            p.add(f"index: {INDEX_UNITS} CPU units (placeholder)", c,
                  INDEX_UNITS * INDEX_UNIT_S[0], upper_s=INDEX_UNITS * INDEX_UNIT_S[1])
            p.peak_usd_per_s = 8 * c.usd_per_s()
        return p  # fake-modal indexes locally
    if step == "deploy":  # image builds; nothing scales up (min_containers 0)
        c = wh.Container(None, 1.0, 2.0)
        p.add("image builds (placeholder)", c, 60.0, upper_s=600.0)
        p.peak_usd_per_s = c.usd_per_s()
        return p
    if step != "run":
        raise KeyError(step)
    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S

    r = kind_rates(mode, table, warm=warm)
    n = pool_sizes(sizes)
    shapes = pools_for(mode)
    nom = simulate(n, r, video_s, mode=mode)["pools"]
    up = simulate(n, r, video_s, upper=True, mode=mode)["pools"]
    plan = shard_plan(video_s, mode)
    for kind, (c, _) in shapes.items():
        k, a, b = r[kind], nom[kind], up[kind]
        s_up = k.startup_s * (1 + k.startup_unc)
        how = k.startup_cache or "cold"
        p.add(f"{kind}: {a.starts} {how} starts ({k.startup_status} {k.startup_s:.0f} s)", c,
              a.startup_s, upper_s=b.startup_s)
        p.add(f"{kind}: work {video_s:.0f} video-s x {k.cs_per_video_s:g} ({k.status} "
              f"+/-{k.uncertainty:.0%})", c, a.busy_s, upper_s=b.busy_s)
        p.add(f"{kind}: idle between shards (pipelining waits)", c, a.wait_s,
              upper_s=max(a.wait_s, b.wait_s))
        # nominal: the drained tail (POOL_END_TAIL_S); upper: the deployed window + client
        # slack, because whether Modal applies the lowered window to an already-idle
        # container is not yet confirmed live (``jobs.pools``; criterion 9 measures it)
        p.add(f"{kind}: idle tail x{a.starts} (drained / stop)", c, a.tail_s,
              upper_s=b.starts * (WORKER_SCALEDOWN_S + CLIENT_SLACK_S))
        vs = [x if kind == "segment" else x[0] for x in plan[kind]]
        shard = (sum(vs) / len(vs) if PROFILE.packing == "exact" else shard_video_s(kind)) \
            * k.cs_per_video_s  # the mean shard (exact packing), else the shard target
        if f"preempt_{kind}" in CHAOS:
            p.add(f"{kind}: 1 preemption (replacement start + lost shard)", c,
                  k.startup_s + shard / 2, upper_s=s_up + shard * (1 + k.uncertainty))
        if kind == "caption" and "kill_orchestrator" in CHAOS:
            # the orchestrator kill: captions idle, scale down, restart
            p.add(f"caption: reconciler gap x{n[kind]} (idle {WORKER_SCALEDOWN_S} s + "
                  "restart)", c, n[kind] * (WORKER_SCALEDOWN_S + k.startup_s),
                  upper_s=n[kind] * (WORKER_SCALEDOWN_S + s_up))
    from flora_deploy.jobs import pools as jp

    mult = 1.0 if jp.ORCHESTRATOR_PREEMPTIBLE else 3.0  # non-preemptible: 3x CPU/memory
    orch = wh.Container(None, jp.ORCHESTRATOR_CORES * mult, jp.ORCHESTRATOR_MEMORY_GIB * mult)
    p.add("orchestrator (CPU) over the wall", orch,
          run_wall_s(mode, video_s, table=table, warm=warm, sizes=n),
          upper_s=run_wall_s(mode, video_s, upper=True, table=table, warm=warm, sizes=n)
          + CLIENT_SLACK_S)
    p.peak_usd_per_s = sum(c.usd_per_s() * n[k] for k, (c, _) in shapes.items()) \
        + orch.usd_per_s()
    return p


def projections(mode: str, table: Any = None, *, warm: bool | None = None
                ) -> dict[str, wh.Projection]:
    return {s: project(s, mode, table=table, warm=warm) for s in STEPS}


SIZING_TARGET_WALL_S = 3600.0  # the recommendation finishes within an hour, even upper


def pool_sizing(ns: Sequence[int] = (1, 2, 3, 4), *, warm: bool | None = None,
                table: Any = None, video_s: float | None = None) -> list[dict[str, Any]]:
    """The wall-clock / cost trade-off of every ``segment`` x ``caption`` pool size in ``ns``
    (real mode; the run step with its chaos, and the bare schedule without it)."""
    video_s = VIDEO_S if video_s is None else video_s
    rows = []
    r = kind_rates("real", table, warm=warm)
    usd_s = container_rates("real")
    for a in ns:
        for b in ns:
            sz = {"segment": a, "caption": b}
            p = project("run", "real", video_s, table, warm=warm, sizes=sz)
            sim = simulate(sz, r, video_s)
            pools = sim["pools"]
            usd = sum(pools[k].container_s * usd_s[k] for k in POOLS)
            rows.append({
                "segment": a, "caption": b, "usd": round(p.usd, 2),
                "usd_upper": round(p.usd_upper, 2),
                "wall_s": round(run_wall_s("real", video_s, table=table, warm=warm, sizes=sz)),
                "wall_upper_s": round(run_wall_s("real", video_s, upper=True, table=table,
                                                 warm=warm, sizes=sz)),
                "no_chaos_usd": round(usd, 2), "no_chaos_wall_s": round(sim["wall_s"]),
                "peak": {k: pools[k].peak for k in POOLS},
                "utilization_working": {k: pools[k].utilization_working for k in POOLS},
                "utilization": {k: pools[k].utilization for k in POOLS},
                "idle_share": round(sum(pools[k].tail_s * usd_s[k] for k in POOLS) / usd, 3),
                "first_caption_after_first_segment_s":
                    round(sim["pipelining"]["first_caption_after_first_segment_s"])})
    return rows


def recommend(rows: Sequence[Mapping[str, Any]], max_wall_s: float = SIZING_TARGET_WALL_S,
              min_per_pool: int = 2) -> Mapping[str, Any] | None:
    """The cheapest size whose **upper** wall (chaos included; both rates are n=1 at +/-50%)
    is within ``max_wall_s``, with at least ``min_per_pool`` containers per pool (the test is
    about autoscaling several containers) that both pools reach (so criterion 3 can pass)."""
    ok = [x for x in rows if x["wall_upper_s"] <= max_wall_s
          and all(x[k] >= min_per_pool and x["peak"][k] >= x[k] for k in POOLS)]
    return min(ok, key=lambda x: (x["usd"], x["wall_s"])) if ok else None


# The lite size options (``--sizing`` with ``--profile lite``): episode counts x pool sizes, at
# the lite profile's shard target and chaos. The recommendation must stay within
# LITE_UPPER_MAX_USD at the upper projection (about $10-15 of the project budget is left after
# the certified run).
LITE_EPISODES = (16, 32, 48)
LITE_POOLS = ((1, 2), (2, 2), (2, 3))
LITE_UPPER_MAX_USD = 7.0


def size_options(base: Profile | None = None, episodes: Sequence[int] = LITE_EPISODES,
                 pools: Sequence[tuple[int, int]] = LITE_POOLS, *, warm: bool | None = None,
                 table: Any = None) -> list[dict[str, Any]]:
    """Every (episodes, segment x caption) size of ``base`` (default: the lite profile):
    nominal / upper totals over all steps (the caps' basis), the run's wall, shards, and
    whether each pool reaches its size and gives every container at least two shards in the
    schedule (so each container yields a rate sample and the pools pipeline)."""
    base = base or PROFILES["lite"]
    rows = []
    for n_ep in episodes:
        for a, b in pools:
            prof = base.with_size(n_ep, {"segment": a, "caption": b})
            with profile_ctx(prof):
                ps = projections("real", table, warm=warm)
                sim = simulate(pool_sizes(), kind_rates("real", table, warm=warm))
                counts = shard_counts()
                rows.append({
                    "episodes": n_ep, "segment": a, "caption": b,
                    "video_s": round(VIDEO_S, 1), "shards": counts,
                    "usd": round(sum(p.usd for p in ps.values()), 2),
                    "usd_upper": round(sum(p.usd_upper for p in ps.values()), 2),
                    "wall_s": round(run_wall_s("real", table=table, warm=warm)),
                    "wall_upper_s": round(run_wall_s("real", upper=True, table=table,
                                                     warm=warm)),
                    "peak": {k: sim["pools"][k].peak for k in POOLS},
                    "shards_per_container": {k: round(counts[k] / POOLS[k]["max_containers"],
                                                      1) for k in POOLS},
                    "overlap_s": round(sim["pipelining"]["overlap_s"])})
    return rows


def recommend_size(rows: Sequence[Mapping[str, Any]], max_upper_usd: float = LITE_UPPER_MAX_USD,
                   min_per_pool: int = 2) -> Mapping[str, Any] | None:
    """The lite recommendation: at least ``min_per_pool`` containers per pool that the schedule
    reaches, two or more shards per container, pipelined pools (overlap > 0), and an upper
    total within ``max_upper_usd``; of those, the most video (more samples per container),
    then the cheapest."""
    ok = [x for x in rows if x["usd_upper"] <= max_upper_usd + 1e-9 and x["overlap_s"] > 0
          and all(x[k] >= min_per_pool and x["peak"][k] >= x[k]
                  and x["shards_per_container"][k] >= 2 for k in POOLS)]
    return max(ok, key=lambda x: (x["video_s"], -x["usd"])) if ok else None


def planner_options(mode: str = "real") -> list[Any]:
    """``flora plan``'s options for this workload, offline (the dry run prints them)."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.planner.options import (
        PlanRequest,
        WorkloadSummary,
        plan_options,
        with_pool_overrides,
    )

    w = WorkloadSummary.from_dict({"groups": [{"pool": "human", "cameras": 1,
                                               "video_s": VIDEO_S,
                                               "episodes": PROFILE.episodes}]})
    req = PlanRequest(workload=w, stages=("segment", "caption"),
                      presets=("balanced", "exact"), max_gpus=max_gpus(),
                      budget_usd=global_cap("real"))
    bal = get_preset("balanced").pools
    from dataclasses import replace

    pools = {k: replace(bal[k], gpu=POOLS[k]["gpu"]) for k in POOLS}
    return plan_options(with_pool_overrides(req, "balanced", pools))


# ---------------------------------------------------------------------------------------------
# Chaos (pure decision; the runners act on it)

@dataclass
class Chaos:
    """Which chaos action is due for a status sample (``flora status --json``). Each fires
    once: a ``segment`` preemption once a segment shard is done and at least two are left, a
    ``caption`` preemption at 25% of the caption shards, an orchestrator kill at 50%.
    ``enabled``: the actions to fire (default: every one, :data:`ALL_CHAOS`; ``run`` passes
    the profile's)."""

    enabled: tuple[str, ...] = ALL_CHAOS
    fired: dict[str, dict[str, Any]] = field(default_factory=dict)

    def due(self, sample: Mapping[str, Any]) -> str | None:
        if sample.get("state") not in ("running", "planned"):
            return None
        counts = sample.get("counts") or {}

        def frac(kind: str) -> tuple[int, int]:
            c = counts.get(kind) or {}
            return int(c.get("done", 0)), int(sum(c.values()))

        sd, st = frac("segment")
        cd, ct = frac("caption")
        running = {k: int((counts.get(k) or {}).get("running", 0)) for k in POOLS}
        for a in self.enabled:
            if a in self.fired:
                continue
            if a == "preempt_segment" and st and sd >= 1 and st - sd >= 2 and running["segment"]:
                return a
            if a == "preempt_caption" and ct and cd >= 0.25 * ct and ct - cd >= 2 \
                    and running["caption"]:
                return a
            if a == "kill_orchestrator" and ct and cd >= 0.5 * ct and ct - cd >= 1 \
                    and sample.get("orchestrator_alive"):
                return a
        return None


# ---------------------------------------------------------------------------------------------
# Config


def config_text(mode: str, *, project: str, run_id: str, budget_usd: float,
                derived: Mapping[str, str] | None = None) -> str:
    if mode == "real":
        d = DATASET
        inp = (f"  format: {d['format']}\n  repo: {d['repo']}\n  revision: {d['revision']}\n"
               f"  path: {d['path']}\n"
               f"  stage_to: volume://{MEDIA_VOLUME[mode]}/{run_id}/wash\n")
    elif mode == "fake-modal":
        inp = ("  format: mp4_dir\n  path: ./synthetic\n"
               f"  stage_to: volume://{MEDIA_VOLUME[mode]}-{run_id}/{run_id}/synthetic\n")
    else:
        inp = "  format: mp4_dir\n  path: ./synthetic\n  stage_to: null\n"
    sink = (f"volume://{STORE_VOLUME[mode]}/{run_id}" if mode == "real" else
            f"volume://{STORE_VOLUME['fake-modal']}-{run_id}/{run_id}")
    res = "\n".join(f"  {k}: {{gpu: {v['gpu']}, max_containers: {v['max_containers']}}}"
                    for k, v in POOLS.items())
    weights = ""
    if derived:
        weights = "weights:\n  derived:\n" + "".join(
            f"    {k}: {v}\n" for k, v in sorted(derived.items()))
    return f"""# Scale test ({PROFILE.name} profile, {mode}); generated by scripts/scale_test.py
project: {project}
license_ack: research-only
modal:
  environment: main
  max_gpus: {max_gpus()}
  budget_usd: {budget_usd:.2f}
  tags: {{flora-scale-test: "{run_id}"}}
preset: balanced
allow_uncertified: true
{weights}input:
{inp}pipeline:
  stages: [segment, caption]
  embodiment: human
  episode_instruction:
    source: dataset_task
  envelope: warn
  shard_target_gpu_minutes: {SHARD_TARGET_GPU_MIN:g}
resources:
{res}
output:
  sink: {sink}
  formats: [jsonl, parquet]
retries:
  max_starts_before_quarantine: 3
"""


def job_id_for(mode: str, run_id: str) -> str:
    from flora_deploy.jobs.store import fake_job_id

    j = f"scale-{PROFILE.tag}{run_id}"
    return j if mode == "real" else fake_job_id(j)


# ---------------------------------------------------------------------------------------------
# Measurements (from the job store + the container samples)


def _t(ts: str | None) -> float | None:
    from flora_deploy.jobs.budget import parse_utc

    return parse_utc(ts) if ts else None


def _quant(xs: Sequence[float]) -> dict[str, float] | None:
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(s), "min": round(s[0], 1), "median": round(statistics.median(s), 1),
            "p95": round(s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)], 1),
            "max": round(s[-1], 1)}


def read_shard_files(store: Any, job_id: str, config_hash: str, manifest: Any
                     ) -> dict[str, list[dict[str, Any]]]:
    """``{"attempts": [...], "receipts": [...], "ledgers": [...]}`` for the pools' kinds."""
    from flora_deploy.jobs.store import JobPaths

    paths = JobPaths(job_id, config_hash=config_hash, stage_hashes=dict(manifest.stage_hashes))
    out: dict[str, list[dict[str, Any]]] = {"attempts": [], "receipts": [], "ledgers": []}
    for kind in POOLS:
        for area, key in (("attempts", "attempts"), ("receipts", "receipts"),
                          ("calls", "ledgers")):
            for f in store.list(paths.area_prefix(config_hash, area, kind)):
                if not f.endswith(".json"):
                    continue
                with contextlib.suppress(Exception):
                    d = json.loads(store.get_bytes(f))
                    d.setdefault("kind", kind)
                    out[key].append(d)
    return out


def measure(files: Mapping[str, Sequence[Mapping[str, Any]]], durations: Mapping[str, float],
            containers: Mapping[str, Mapping[str, float]] | None = None, *,
            usd_per_s: Mapping[str, float] | None = None,
            run_start: float | None = None,
            outage: tuple[float, float] | None = None) -> dict[str, Any]:
    """Per pool: containers, container-seconds, busy seconds, utilization, cold starts, idle
    tails and their cost, video-s per container-s, peak concurrency; plus pipelining.

    ``containers``: ``{container_id: {"start": t, "end": t}}`` from ``modal container list``
    samples; without it (local) a container's life is its first start to its last end.
    ``run_start``: when ``run`` launched the job. ``time_to_peak_s`` is from it to the moment
    the pool first ran ``max_containers`` containers (None: no run start, or never reached).
    ``preempted_recovered``: preempted starts whose shard a later attempt finished.
    ``utilization_working``: busy / the containers' seconds from their first shard start to
    their last shard end (no cold start, no tail). ``outage``: ``(killed, revived)`` times of
    the orchestrator kill; ``idle_share_excl_outage`` leaves out the tails that began inside
    it (containers idled by the induced outage, not by the scale-down policy)."""
    kind_of: dict[str, str] = {}
    spans: dict[str, list[tuple[float, float, str]]] = {}
    receipts = {(r["shard_id"], r["container_id"]): r for r in files.get("receipts", [])}
    spawned: dict[str, float] = {}
    for led in files.get("ledgers", []):
        for c in led.get("calls", []):
            if c.get("outcome") == "spawned" and c.get("call_id") and c.get("utc"):
                spawned[c["call_id"]] = _t(c["utc"])  # type: ignore[assignment]
    preempted: dict[str, int] = {}
    recovered: dict[str, int] = {}
    finished_after: dict[tuple[str, str], list[float]] = {}
    for rc in files.get("receipts", []):
        ft = _t(rc.get("finished_utc"))
        if ft is not None:
            finished_after.setdefault((rc["kind"], rc["shard_id"]), []).append(ft)
    latency: dict[str, list[float]] = {}
    first_call: dict[str, tuple[float, str]] = {}  # container -> (first attempt start, call id)
    for a in files.get("attempts", []):
        cid, kind = a["container_id"], a["kind"]
        kind_of[cid] = kind
        t0 = _t(a.get("started_utc"))
        if t0 is not None and (cid not in first_call or t0 < first_call[cid][0]):
            first_call[cid] = (t0, str(a.get("call_id") or ""))
        r = receipts.get((a["shard_id"], cid))
        t1 = _t(r["finished_utc"]) if r else _t(a.get("ended_utc"))
        if a.get("preempted"):
            preempted[kind] = preempted.get(kind, 0) + 1
            done = [ft for ft in finished_after.get((kind, a["shard_id"]), ())
                    if t0 is None or ft > t0]
            if done and (a["shard_id"], cid) not in receipts:
                recovered[kind] = recovered.get(kind, 0) + 1
        if t0 is not None and t1 is not None and t1 >= t0:
            spans.setdefault(cid, []).append((t0, t1, a["shard_id"]))
        if t0 is not None and a.get("call_id") in spawned:
            latency.setdefault(kind, []).append(t0 - spawned[a["call_id"]])
    out: dict[str, Any] = {"pools": {}}
    life: dict[str, tuple[float, float]] = {}
    for cid, sp in spans.items():
        first, last = min(s[0] for s in sp), max(s[1] for s in sp)
        obs = (containers or {}).get(cid)
        life[cid] = (min(obs["start"], first), max(obs["end"], last)) if obs else (first, last)
    for kind in POOLS:
        cids = [c for c, k in kind_of.items() if k == kind and c in life]
        rs = [r for r in files.get("receipts", []) if r["kind"] == kind]
        video = sum(durations.get(u, 0.0) for r in rs for u in r.get("episodes_done", ()))
        cs = sum(life[c][1] - life[c][0] for c in cids)
        busy = sum(s[1] - s[0] for c in cids for s in spans[c])
        cold = [min(s[0] for s in spans[c]) - life[c][0] for c in cids
                if containers and c in containers]
        idle = [life[c][1] - max(s[1] for s in spans[c]) for c in cids]
        working = sum(max(s[1] for s in spans[c]) - min(s[0] for s in spans[c]) for c in cids)
        idle_x = [life[c][1] - max(s[1] for s in spans[c]) for c in cids
                  if not (outage and outage[0] <= max(s[1] for s in spans[c]) <= outage[1])]
        events = sorted([(life[c][0], 1) for c in cids] + [(life[c][1], -1) for c in cids])
        peak = cur = 0
        t_full = None  # first moment the pool ran max_containers
        for t, d in events:
            cur += d
            peak = max(peak, cur)
            if t_full is None and cur >= POOLS[kind]["max_containers"]:
                t_full = t
        starts = [s[0] for c in cids for s in spans[c]]
        ends = [s[1] for c in cids for s in spans[c]]
        rate = (usd_per_s or {}).get(kind, 0.0)
        out["pools"][kind] = {
            "containers": len(cids), "peak_concurrent": peak,
            "time_to_peak_s": (round(t_full - run_start, 1)
                               if t_full is not None and run_start is not None else None),
            "container_s": round(cs, 1), "busy_s": round(busy, 1),
            "utilization": round(busy / cs, 3) if cs else None,
            "utilization_working": round(busy / working, 3) if working else None,
            "video_s": round(video, 1),
            "video_s_per_container_s": round(video / cs, 4) if cs else None,
            "window_s": round(max(ends) - min(starts), 1) if starts else None,
            "video_s_per_wall_s": (round(video / (max(ends) - min(starts)), 3)
                                   if starts and max(ends) > min(starts) else None),
            "cold_start_s": _quant(cold), "spawn_to_start_s": _quant(latency.get(kind, [])),
            "idle_tail_s": _quant(idle), "idle_s": round(sum(idle), 1),
            "idle_usd": round(sum(idle) * rate, 4), "usd": round(cs * rate, 4),
            "idle_usd_excl_outage": round(sum(idle_x) * rate, 4),
            "preempted_starts": preempted.get(kind, 0),
            "preempted_recovered": recovered.get(kind, 0), "shards_done": len(rs)}
    seg = out["pools"].get("segment", {})
    seg_sp = [s for c, k in kind_of.items() if k == "segment" for s in spans.get(c, [])]
    cap_sp = [s for c, k in kind_of.items() if k == "caption" for s in spans.get(c, [])]
    if seg_sp and cap_sp:
        seg_first_end = min(s[1] for s in seg_sp)
        overlap = min(max(s[1] for s in seg_sp), max(s[1] for s in cap_sp)) - \
            max(min(s[0] for s in seg_sp), min(s[0] for s in cap_sp))
        out["pipelining"] = {
            "first_caption_after_first_segment_s": round(min(s[0] for s in cap_sp)
                                                         - seg_first_end, 1),
            "overlap_s": round(max(0.0, overlap), 1),
            "overlap_share_of_segment_window": (round(max(0.0, overlap) / seg["window_s"], 3)
                                                if seg.get("window_s") else None)}
    total_usd = sum(p["usd"] for p in out["pools"].values())
    out["idle_share"] = (round(sum(p["idle_usd"] for p in out["pools"].values()) / total_usd, 3)
                         if total_usd else None)
    out["idle_share_excl_outage"] = (
        round(sum(p["idle_usd_excl_outage"] for p in out["pools"].values()) / total_usd, 3)
        if total_usd else None)
    out["observed_usd"] = round(total_usd, 4)
    out["containers"] = per_container(files, durations, spans, life, kind_of, first_call,
                                      spawned, containers)
    return out


PREWARM_LEAD_S = 5.0  # a container up this long before its first call was spawned: pre-warmed


def per_container(files: Mapping[str, Sequence[Mapping[str, Any]]],
                  durations: Mapping[str, float], spans: Mapping[str, list[tuple]],
                  life: Mapping[str, tuple[float, float]], kind_of: Mapping[str, str],
                  first_call: Mapping[str, tuple[float, str]], spawned: Mapping[str, float],
                  containers: Mapping[str, Mapping[str, float]] | None) -> list[dict[str, Any]]:
    """One row per container that ran a shard: its kind, busy seconds, the video seconds of
    the episodes its receipts finished, its container-seconds per video-second (a rate sample
    for ``--finalize``), shards finished, its first call's spawn-to-start latency, and whether
    it was **pre-warmed** (Modal started it more than :data:`PREWARM_LEAD_S` before its first
    call was spawned: the caption pre-warm, whose spawn-to-start is not a startup)."""
    rows = []
    done: dict[str, list[Mapping[str, Any]]] = {}
    for r in files.get("receipts", []):
        done.setdefault(str(r.get("container_id")), []).append(r)
    for cid in sorted(life):
        sp = spans.get(cid, [])
        rs = done.get(cid, [])
        video = sum(durations.get(u, 0.0) for r in rs for u in r.get("episodes_done", ()))
        busy = sum(s[1] - s[0] for s in sp)
        t_first, call = first_call.get(cid, (None, ""))
        t_spawn = spawned.get(call)
        obs = (containers or {}).get(cid)
        prewarmed = bool(obs and t_spawn is not None
                         and obs["start"] < t_spawn - PREWARM_LEAD_S)
        rows.append({
            "container_id": cid, "kind": kind_of.get(cid), "shards": len(rs),
            "busy_s": round(busy, 1), "video_s": round(video, 1),
            "container_s_per_video_s": round(busy / video, 4) if video else None,
            "spawn_to_start_s": (round(t_first - t_spawn, 1)
                                 if t_first is not None and t_spawn is not None else None),
            "started": round(life[cid][0], 1), "ended": round(life[cid][1], 1),
            "prewarmed": prewarmed})
    return rows


def first_spawn_after(ledgers: Sequence[Mapping[str, Any]], t: float) -> float | None:
    """The first spawn / adoption any orchestrator recorded after ``t``: the proof a new
    orchestrator generation took over (a stale lease can still read "alive" for its TTL)."""
    ts = [_t(c["utc"]) for led in ledgers for c in led.get("calls", [])
          if c.get("outcome") in ("spawned", "adopted") and c.get("utc")]
    after = [x for x in ts if x is not None and x > t]
    return min(after) if after else None


# ---------------------------------------------------------------------------------------------
# Success criteria (docs/scale-test.md "Success criteria"; numbers are proposals)

CRITERIA: dict[str, Any] = {
    "end_state": "done",
    "lost_or_duplicate_records": 0,
    "utilization_min": 0.60,  # busy / container-s per pool while it holds work
    "projection_within_upper": True,
    "projection_nominal_error_max": 0.50,  # |actual / nominal - 1|
    "idle_share_max": 0.15,
    "reconciler_revive_max_s": RECONCILE_WAIT_S,
    "status_p95_max_s": 10.0,
    "report_max_s": 30.0,
    "compose_max_s": 60.0,
}
LAUNCH_MARGIN_S = 120.0  # `flora run --detach` -> orchestrator up -> first spawn, + scheduling

# Added with the lite profile (they apply to full as well): what a run must prove beyond the 14.
BATCH_FILL = {"mean_batch_min": 2.0, "busy_fraction_min": 0.60}
# The engines' request caps after Fix-13 (TempSeg on L40S, Captioner on H100): batch fill is
# reported as the mean batch against them.
ENGINE_MAX_NUM_SEQS = {"segment": 48, "caption": 16}
MIN_CONTAINERS_MEASURED = 2  # per pool: containers with a finished shard (rate samples)
DRAIN_BOUND_S = h0.DRAIN_BOUND_S  # 30 s: a drained container must be gone this long after
TAIL_WATCH_S = DRAIN_BOUND_S + 15.0  # after the job ends: keep sampling containers this long
TAIL_POLL_S = 5.0  # ... this often (the resolution of the measured drain)

# The criteria: (number, name). Every one is automatic: computed by :func:`evaluate` from
# what ``run`` and ``report`` record. MANUAL lists the ones that are not (none today); each
# would carry the reason it cannot be read from the recorded data. 1-14 are the original
# criteria; 15-19 were added with the lite profile.
CRITERION_NAMES: tuple[tuple[int, str], ...] = (
    (1, "end state"),
    (2, "lost or duplicate records"),
    (3, "each pool reaches max_containers"),
    (4, "time to peak concurrency"),
    (5, "utilization while holding work"),
    (6, "segment -> caption pipelining"),
    (7, "spend within the upper projection"),
    (8, "spend vs the nominal projection"),
    (9, "idle / scale-down share of pool cost"),
    (10, "preemption recovered"),
    (11, "orchestrator revived"),
    (12, "flora status p95"),
    (13, "flora report"),
    (14, "flora compose"),
    (15, "throughput per pool from several containers"),
    (16, "scale to zero at the end of the job"),
    (17, "caption pre-warm"),
    (18, "compile cache, per shard"),
    (19, "batch fill"),
)
MANUAL: dict[int, str] = {}
LOCAL_NA = "local: calls run on threads, with no containers, cold starts or money"
LOCAL_FAKE_NA = ("local: fake engines compile nothing and simulate their batches, so cache "
                 "and batch figures mean nothing")
NO_CHAOS_NA = ("not fired by the {profile} profile: a stopped container's lost half shard "
               "would count as busy time and skew the per-container rate samples this profile "
               "exists to take, and the two replacement starts cost ~$0.75; judged by the "
               "full profile and the local / fake-modal rehearsals")
NO_KILL_NA = ("not fired by the {profile} profile: the reconciler runs every 10 minutes, about "
              "as long as this job's whole caption phase, so a kill would double the wall and "
              "its idle cost (~$1.80) would swamp the throughput and idle measurements; judged "
              "by the full profile and the local / fake-modal rehearsals")


def batch_fill(audits: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per pool, from the shard audit rows (``shard_audit``; the batcher's stats,
    ``engines.microbatch.BatcherStats``): shards, rounds, the mean batch (weighted by rounds),
    the largest batch, the GPU-busy fraction (generate seconds over the batcher's wall
    seconds) and the fill (mean batch over the engine's request cap,
    :data:`ENGINE_MAX_NUM_SEQS`)."""
    acc: dict[str, dict[str, float]] = {}
    for a in audits:
        if not a.get("shard_audit") or a.get("stage") not in POOLS:
            continue
        b = a.get("batcher") or {}
        x = acc.setdefault(a["stage"], {"shards": 0, "rounds": 0, "batched": 0.0,
                                        "max": 0, "gen": 0.0, "wall": 0.0})
        x["shards"] += 1
        r = int(b.get("rounds") or 0)
        x["rounds"] += r
        x["batched"] += float(b.get("mean_batch") or 0.0) * r
        x["max"] = max(x["max"], int(b.get("max_batch") or 0))
        x["gen"] += float(b.get("generate_s") or 0.0)
        x["wall"] += float(b.get("wall_s") or 0.0)
    out = {}
    for k, x in acc.items():
        mean = x["batched"] / x["rounds"] if x["rounds"] else None
        out[k] = {"shards": int(x["shards"]), "rounds": int(x["rounds"]),
                  "mean_batch": round(mean, 2) if mean is not None else None,
                  "max_batch": int(x["max"]),
                  "busy_fraction": round(x["gen"] / x["wall"], 3) if x["wall"] else None,
                  "fill": (round(mean / ENGINE_MAX_NUM_SEQS[k], 3)
                           if mean is not None and k in ENGINE_MAX_NUM_SEQS else None)}
    return out


def cache_by_kind(audits: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]],
                  warm_kinds: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Per pool: the compile-cache provenance of every shard (``h0_rehearsal.
    cache_provenance_check``: every shard names a state) and the states seen. A pool whose
    cache was published before the run (``warm_kinds``, from ``plan``) must load it on every
    shard (``warm``)."""
    out = {}
    for k in POOLS:
        ok, detail = h0.cache_provenance_check(list(audits), list(receipts), kinds=(k,))
        states: dict[str, int] = {}
        seen: set[str] = set()
        for d in [a for a in audits if a.get("shard_audit") and a.get("stage") == k] + \
                [r for r in receipts if r.get("kind") == k]:
            sid = str(d.get("shard_id"))
            v = next((d[c] for c in h0.CACHE_KEYS if d.get(c) is not None), None)
            if sid in seen or v is None:
                continue
            seen.add(sid)
            s = (v.get("state") or v.get("source")) if isinstance(v, dict) else v
            states[str(s)] = states.get(str(s), 0) + 1
        warm = k in warm_kinds
        if ok and warm and set(states) != {"warm"}:
            ok, detail = False, f"{detail}; a cache was published before the run, but not " \
                                "every shard loaded it"
        out[k] = {"ok": ok, "detail": detail, "states": states, "published_before": warm}
    return out


def criteria_targets(mode: str, sizes: Mapping[str, int] | None = None,
                     table: Any = None) -> dict[str, Any]:
    """The rate-derived targets: from the **upper**, cold-start schedule (:func:`simulate`) of
    the pools as configured, so they move with ``rates.json``. Time to peak: when the upper
    schedule has each pool at its size, plus a poll and :data:`LAUNCH_MARGIN_S`. First
    caption: the upper caption cold start plus the margin."""
    n = pool_sizes(sizes)
    r = kind_rates(mode, table, warm=False)
    up = simulate(n, r, upper=True, mode=mode)
    poll = POLL_S.get(mode, 30.0)
    return {**CRITERIA, "sizes": n,
            "time_to_peak_max_s": {k: round((up["pools"][k].t_peak or 0.0) + poll
                                            + LAUNCH_MARGIN_S) for k in POOLS},
            "pipelining_first_caption_max_s": round(
                r["caption"].startup_s * (1 + r["caption"].startup_unc) + LAUNCH_MARGIN_S)}


def _all(oks: Sequence[bool | None]) -> bool | None:
    if any(o is False for o in oks):
        return False
    return None if not oks or any(o is None for o in oks) else True


def evaluate(r: Mapping[str, Any], mode: str, targets: Mapping[str, Any] | None = None
             ) -> list[dict[str, Any]]:
    """PASS / FAIL / n.a. for each criterion (:data:`CRITERION_NAMES`) from the ``run`` +
    ``report`` results. A value the run did not record is a FAIL ("not recorded"): every
    criterion is meant to be measured. ``n.a.`` is only for criteria that cannot apply in
    ``local``, or for chaos the profile does not fire (``chaos_enabled``); ``reason`` says
    why."""
    T = dict(targets or criteria_targets(mode, r.get("sizes")))
    sizes = T["sizes"]
    local = mode == "local"
    rows: dict[int, dict[str, Any]] = {}

    def add(i: int, value: Any, target: str, ok: bool | None, reason: str = "") -> None:
        if ok is None and not reason:
            ok, reason = False, "not recorded"
        rows[i] = {"id": i, "criterion": dict(CRITERION_NAMES)[i], "value": value,
                   "target": target, "auto": i not in MANUAL,
                   "verdict": "n.a." if ok is None else ("PASS" if ok else "FAIL"),
                   "reason": reason}

    m = r.get("measure") or {}
    pools = m.get("pools") or {}
    st = r.get("state")
    add(1, st, T["end_state"], None if st is None else st == T["end_state"])
    rec = r.get("records") or {}
    bad = (int(rec.get("missing", 0)) + int(rec.get("duplicates", 0))) if rec else None
    add(2, bad, "0 (compose missing + conflicts)", None if bad is None else bad == 0)

    def per_pool(get: Callable[[str, Mapping[str, Any]], tuple[Any, bool | None]]
                 ) -> tuple[dict[str, Any], bool | None]:
        vals, oks = {}, []
        for kind in POOLS:
            p = pools.get(kind)
            v, ok = get(kind, p) if p else (None, None)
            vals[kind] = v
            oks.append(ok)
        return vals, _all(oks)

    if local:
        for i, tgt in ((3, f"peak >= {sizes}"), (4, "<= upper schedule"),
                       (5, f">= {T['utilization_min']}")):
            add(i, None, tgt, None, LOCAL_NA)
    else:
        v, ok = per_pool(lambda k, p: (p.get("peak_concurrent"),
                                       None if p.get("peak_concurrent") is None
                                       else p["peak_concurrent"] >= sizes[k]))
        add(3, v, f"peak >= {sizes}", ok)

        def ttp(k: str, p: Mapping[str, Any]) -> tuple[Any, bool | None]:
            t = p.get("time_to_peak_s")
            if (p.get("peak_concurrent") or 0) < sizes[k]:
                return t, False  # never reached its size: no peak to time
            return t, None if t is None else t <= T["time_to_peak_max_s"][k]
        v, ok = per_pool(ttp)
        add(4, v, "<= " + ", ".join(f"{k} {s} s" for k, s in T["time_to_peak_max_s"].items())
            + " from the run start", ok)

        def util(k: str, p: Mapping[str, Any]) -> tuple[Any, bool | None]:
            u = p.get("utilization_working")
            return u, None if u is None else u >= T["utilization_min"]
        v, ok = per_pool(util)
        add(5, v, f">= {T['utilization_min']} per pool (busy / container-s from each "
            "container's first shard start to its last shard end)", ok)
    pl = m.get("pipelining") or {}
    fc = pl.get("first_caption_after_first_segment_s")
    add(6, {"first_caption_s": fc, "overlap_s": pl.get("overlap_s")},
        f"<= {T['pipelining_first_caption_max_s']} s and overlap > 0",
        None if fc is None else (fc <= T["pipelining_first_caption_max_s"]
                                 and (pl.get("overlap_s") or 0) > 0))
    pe = r.get("projection_error") or {}
    if local:
        add(7, None, "actual <= upper", None, "local: fake call seconds are not money")
        add(8, None, f"|err| <= {T['projection_nominal_error_max']}", None,
            "local: fake call seconds are not money")
    else:
        a, u = pe.get("actual_usd"), pe.get("upper_usd")
        add(7, a, f"<= ${u}", None if a is None or u is None else a <= u + 1e-9,
            "" if pe.get("billed") else
            ("provisional: tracked spend; --report re-checks once Modal billing lands"
             if a is not None else ""))
        e = pe.get("vs_nominal")
        add(8, e, f"|err| <= {T['projection_nominal_error_max']}",
            None if e is None else abs(e) <= T["projection_nominal_error_max"])
    s = m.get("idle_share_excl_outage", m.get("idle_share"))
    add(9, s, f"<= {T['idle_share_max']} (tails that began while the killed orchestrator was "
        "down are excluded)", None if local else (None if s is None
                                                  else s <= T["idle_share_max"]),
        LOCAL_NA if local else "")
    chaos = r.get("chaos") or {}
    enabled = tuple(r.get("chaos_enabled", ALL_CHAOS))  # a report without it ran every action
    prof = r.get("profile", "full")

    def pre(k: str, p: Mapping[str, Any]) -> tuple[Any, bool | None]:
        n_pre, n_rec = p.get("preempted_starts", 0), p.get("preempted_recovered", 0)
        if f"preempt_{k}" not in chaos:
            return "not fired", False
        return f"{n_rec}/{n_pre}", n_pre >= 1 and n_rec == n_pre
    pre_tgt = ">= 1 preempted start per pool, each shard finished by a later attempt"
    if not any(a.startswith("preempt_") for a in enabled):
        add(10, None, pre_tgt, None, NO_CHAOS_NA.format(profile=prof))
    else:
        v, ok = per_pool(pre)
        add(10, v, pre_tgt, ok)
    orch = r.get("orchestrator") or {}
    rv = orch.get("revive_s")
    by = "flora resume" if local else "the reconciler"
    if "kill_orchestrator" not in enabled:
        add(11, None, f"killed, revived by {by} <= {T['reconciler_revive_max_s']} s", None,
            NO_KILL_NA.format(profile=prof))
    elif not orch.get("killed"):
        add(11, None, f"killed, revived by {by} <= {T['reconciler_revive_max_s']} s",
            False, "the orchestrator kill did not fire")
    else:
        add(11, {"revive_s": rv, "revived_by": orch.get("revived_by")},
            f"revived by {by} <= {T['reconciler_revive_max_s']} s, then work continues "
            f"(end state {T['end_state']})",
            rv is not None and rv <= T["reconciler_revive_max_s"]
            and (local or orch.get("revived_by") == "reconciler") and st == T["end_state"])
    tm = r.get("timings") or {}
    sp = (tm.get("status") or {}).get("p95")
    add(12, sp, f"<= {T['status_p95_max_s']} s", None if sp is None
        else sp <= T["status_p95_max_s"])
    rp = tm.get("report_s")
    add(13, rp, f"<= {T['report_max_s']} s", None if rp is None else rp <= T["report_max_s"])
    cp, crc = tm.get("compose_s"), tm.get("compose_rc", 0)
    add(14, cp, f"<= {T['compose_max_s']} s and exit 0",
        None if cp is None else cp <= T["compose_max_s"] and crc == 0)
    _evaluate_proofs(r, local, add)
    return [rows[i] for i, _ in CRITERION_NAMES]


def _evaluate_proofs(r: Mapping[str, Any], local: bool,
                     add: Callable[..., None]) -> None:
    """Criteria 15-19 (added with the lite profile; see :func:`evaluate`)."""
    m = r.get("measure") or {}
    rows = [c for c in m.get("containers") or () if c.get("kind") in POOLS]
    tgt15 = (f">= {MIN_CONTAINERS_MEASURED} containers per pool with a finished shard and "
             "a measured container-s per video-s")
    if local:
        add(15, None, tgt15, None, LOCAL_NA)
    elif not rows:
        add(15, None, tgt15, None)
    else:
        val, oks = {}, []
        for k in POOLS:
            xs = [c for c in rows if c["kind"] == k and c.get("shards")
                  and c.get("container_s_per_video_s")]
            rs = [c["container_s_per_video_s"] for c in xs]
            busy = sum(c["busy_s"] for c in xs)
            video = sum(c["video_s"] for c in xs)
            val[k] = {"containers": len(xs),
                      "container_s_per_video_s": round(busy / video, 4) if video else None,
                      "range": [min(rs), max(rs)] if rs else None}
            oks.append(len(xs) >= MIN_CONTAINERS_MEASURED)
        add(15, val, tgt15, all(oks))
    tgt16 = (f"every worker container gone <= {DRAIN_BOUND_S:.0f} s after its pool drained "
             "(container lifetimes against the orchestrator's drain record)")
    dr = r.get("drain")
    if local:
        add(16, None, tgt16, None, LOCAL_NA)
    elif not dr:
        add(16, None, tgt16, None)
    else:
        workers = [x for x in dr if x.get("kind") is not None]
        ok, detail = h0.drain_check(workers, DRAIN_BOUND_S)
        gaps = {k: max((x["gap_s"] for x in workers if x["kind"] == k
                        and x.get("gap_s") is not None), default=None) for k in POOLS}
        add(16, {"max_gap_s": gaps, "detail": detail[:300]}, tgt16, ok)
    tgt17 = ("caption pre-warm planned and started, first caption container within "
             f"{h0.PREWARM_TOL_S:.0f} s of the planned start")
    pw = r.get("prewarm")
    if local:
        add(17, None, tgt17, None, LOCAL_NA)
    elif not pw:
        add(17, None, tgt17, None)
    else:
        add(17, pw.get("detail"), tgt17, bool(pw.get("ok")))
    tgt18 = ("every shard names its compile-cache state; a pool whose cache was published "
             "before the run loads it on every shard")
    cache = r.get("cache")
    if local:
        add(18, None, tgt18, None, LOCAL_FAKE_NA)
    elif not cache:
        add(18, None, tgt18, None)
    else:
        oks = [cache.get(k, {}).get("ok") for k in POOLS]
        missing = any(o is None for o in oks)  # no shard of a pool names a cache: a FAIL
        add(18, {k: {"states": cache.get(k, {}).get("states"),
                     "published_before": cache.get(k, {}).get("published_before")}
                 for k in POOLS}, tgt18, False if missing else all(oks),
            "not recorded: a pool's shards carry no compile-cache provenance" if missing else "")
    tgt19 = (f"mean batch >= {BATCH_FILL['mean_batch_min']:g} and GPU-busy fraction >= "
             f"{BATCH_FILL['busy_fraction_min']:g} per pool (shard audit rows; caps "
             + ", ".join(f"{k} {v}" for k, v in ENGINE_MAX_NUM_SEQS.items()) + ")")
    bf = r.get("batch_fill")
    if local:
        add(19, None, tgt19, None, LOCAL_FAKE_NA)
    elif not bf or any(k not in bf for k in POOLS):
        add(19, bf or None, tgt19, None)
    else:
        oks = []
        for k in POOLS:
            mb, busy = bf[k].get("mean_batch"), bf[k].get("busy_fraction")
            oks.append(None if mb is None or busy is None else
                       mb >= BATCH_FILL["mean_batch_min"]
                       and busy >= BATCH_FILL["busy_fraction_min"])
        add(19, {k: {x: bf[k].get(x) for x in ("mean_batch", "fill", "busy_fraction")}
                 for k in POOLS}, tgt19, _all(oks))


def criteria_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    c = {v: sum(1 for x in rows if x["verdict"] == v) for v in ("PASS", "FAIL", "n.a.")}
    auto = sum(1 for x in rows if x["auto"])
    return (f"{c['PASS']} PASS, {c['FAIL']} FAIL, {c['n.a.']} n.a. "
            f"({auto} of {len(rows)} automatic)")


# ---------------------------------------------------------------------------------------------
# Runners


@dataclass
class Ctx:
    mode: str
    st: wh.State
    work: Path
    state_path: Path
    poll_s: float
    log: Callable[[str], None] = print
    step: str = ""
    keep: bool = False

    def sh(self, args: list[str], *, timeout: float = 1800, check: bool = True):
        return h0.sh(args, cwd=self.work, timeout=timeout, check=check,
                     log=self.work / "scale-test.log")

    def note_app(self, name: str) -> None:
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None and name not in cur.apps:
            cur.apps.append(name)
            wh.save_state(self.st, self.state_path)

    @property
    def run(self) -> str:
        return self.st.run_id

    def app_name(self, step: str) -> str:
        """This attempt's own ephemeral app name (:func:`app_name`)."""
        return app_name(self.run, step, max(1, len(self.st.step(self.step or step).attempts)))

    @property
    def cfg(self) -> Path:
        return self.work / "flora.yaml"

    @property
    def job_id(self) -> str:
        return job_id_for(self.mode, self.run)

    @property
    def store_uri(self) -> str:
        if self.mode == "local":
            return f"file://{self.work / 'store'}"
        vol = STORE_VOLUME["real"] if self.mode == "real" else \
            f"{STORE_VOLUME['fake-modal']}-{self.run}"
        return f"modal-volume://{vol}/{self.run}"

    def store_args(self) -> list[str]:
        return ["--store", self.store_uri, "--config", str(self.cfg)]


def app_name(run_id: str, step: str, attempt: int) -> str:
    """One attempt's own app name: ``flora-st-<run>-<step>-a<n>``. Billing reports spend per
    app name; a shared name (``flora-index``) would mix in any other run's index in that hour
    (``wave_h.app_name`` names its apps ``flora-wh-...`` for the same reason)."""
    return f"flora-st-{run_id}-{step.replace('_', '-')}-a{attempt}"


def _flora() -> str:
    return h0.flora_bin()


APP_DEPLOYED, APP_STOPPED = "deployed", "stopped"


def app_state(st: wh.State) -> str | None:
    """Whether the deployed app is live: ``deploy``'s ``app_state`` (None: never recorded)."""
    s = st.steps.get("deploy")
    return (s.result or {}).get("app_state") if s else None


def mark_app(st: wh.State, state: str, path: Path | None = None) -> None:
    """Record the app's state: :data:`APP_DEPLOYED` after ``flora deploy``,
    :data:`APP_STOPPED` once anything ran ``modal app stop`` on it (``run``'s ``finally``, or
    the cleanup of a killed run). A stopped app makes the next pass deploy again."""
    s = st.step("deploy")
    s.result = {**(s.result or {}), "app_state": state}
    if path is not None:
        wh.save_state(st, path)


def run_verify(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "real":
        return verify_dataset()
    from flora_deploy.local.sample import sample_dataset

    n = int(ctx.st.step("verify").result.get("episodes") or FAKE_EPISODES)
    root = sample_dataset(ctx.work / "synthetic", episodes=n, min_s=FAKE_MIN_S,
                          max_s=FAKE_MAX_S)
    return {"ok": True, "episodes": n, "path": str(root), "tracked_usd": 0.0, "apps": []}


def _write_config(ctx: Ctx, derived: Mapping[str, str] | None = None) -> None:
    budget = max(0.01, caps(ctx.mode)["run"]) if ctx.mode != "local" else 50.0
    ctx.cfg.write_text(config_text(ctx.mode, project=f"scale-{PROFILE.tag}{ctx.run}",
                                   run_id=ctx.run,
                                   budget_usd=budget, derived=derived), encoding="utf-8")


def run_index(ctx: Ctx) -> dict[str, Any]:
    derived = None
    if ctx.mode == "real":
        derived = wh.resolve_receipts(["captioner_bf16", "tempseg_merged"], require_pin=True)
    _write_config(ctx, derived)
    fl = _flora()
    summary = ctx.work / "summary.json"
    t0 = time.time()
    if ctx.mode == "real":
        name = ctx.app_name("index")  # its own app: billing attributes it to this run only
        ctx.note_app(name)
        ctx.sh([fl, "index", "--config", str(ctx.cfg), "--job", ctx.job_id, "--store",
                ctx.store_uri, "--out", str(summary), "--app-name", name,
                *index_limit_args()], timeout=3600)
        usd = project("index", "real").items[0]["usd_per_h"] / 3600 * (time.time() - t0)
    else:
        local_store = ctx.work / "store"
        ctx.sh([fl, "index", "--local", "--config", str(ctx.cfg), "--job", ctx.job_id,
                "--store", f"file://{local_store}", "--out", str(summary)], timeout=3600)
        if ctx.mode == "fake-modal":
            media = f"{MEDIA_VOLUME['fake-modal']}-{ctx.run}"
            h0.upload_dir(media, ctx.work / "synthetic", f"{ctx.run}/synthetic")
            h0.mirror_index(local_store, ctx.job_id, ctx.run,
                            f"{STORE_VOLUME['fake-modal']}-{ctx.run}")
        usd = 0.0
    s = json.loads(summary.read_text())
    c = s.get("counts", {})
    problems = subset_problems(c, s.get("video_s")) if ctx.mode == "real" else []
    return {"ok": c.get("indexed", 0) > 0 and not s.get("failed") and not problems,
            "counts": c, "video_s": s.get("video_s"), "subset_problems": problems,
            "wall_s": round(time.time() - t0, 1), "tracked_usd": round(usd, 4), "apps": []}


SUBSET_VIDEO_TOL = 0.01  # indexed video length vs the pinned table (container vs frame count)


def index_limit_args() -> list[str]:
    """``flora index --limit N`` for a subset profile: the first N episodes by episode index
    (``media.sources.lerobot_v3`` lists them in that order). Nothing for the full dataset."""
    n = PROFILE.episodes
    return [] if n >= DATASET["total_episodes"] else ["--limit", str(n)]


def subset_problems(counts: Mapping[str, Any], video_s: Any) -> list[str]:
    """Why the indexed episodes are not the profile's subset (empty: they are): the count, and
    the video length against the pinned per-episode frame counts within
    :data:`SUBSET_VIDEO_TOL`."""
    out = []
    want = PROFILE.episodes
    if int(counts.get("episodes", want)) != want or int(counts.get("indexed", 0)) != want:
        out.append(f"indexed {counts.get('indexed')} of {counts.get('episodes')} episodes; "
                   f"the {PROFILE.name} profile is the first {want}")
    if video_s is None:
        out.append("the index summary has no video_s")
    elif abs(float(video_s) / VIDEO_S - 1) > SUBSET_VIDEO_TOL:
        out.append(f"indexed {float(video_s):.1f} video-s; the pinned table says {VIDEO_S:.1f}")
    return out


def run_plan(ctx: Ctx) -> dict[str, Any]:
    p = ctx.sh([_flora(), "plan", "--config", str(ctx.cfg), "--index",
                str(ctx.work / "summary.json"), "--json"], check=False)
    opts: Any = None
    with contextlib.suppress(ValueError):
        opts = json.loads(p.stdout)
    video = float(ctx.st.step("index").result.get("video_s") or VIDEO_S)
    proj = project("run", ctx.mode, video)
    rows = []
    for o in (opts.get("options", opts) if isinstance(opts, dict) else opts) or []:
        if isinstance(o, dict):
            rows.append({k: o.get(k) for k in ("name", "cost_usd", "cost_high_usd", "wall_s",
                                               "peak_gpus", "status", "rate_status")})
    for r in rows:
        ctx.log(f"  option {r['name']}: ${r['cost_usd']} (high ${r['cost_high_usd']}), "
                f"{(r['wall_s'] or 0) / 3600:.2f} h, {r['peak_gpus']} GPUs, {r['status']}")
    ctx.log(f"  this run ({ctx.mode}): nominal ${proj.usd:.2f}, upper ${proj.usd_upper:.2f}, "
            f"{video:.0f} video-s")
    warm_kinds: list[str] = []
    if ctx.mode == "real":  # a read of flora-cache (no container): which pools start warm
        with contextlib.suppress(Exception):
            from flora_deploy.cli.plan import published_cache_kinds
            from flora_deploy.config.schema import load_config

            warm_kinds = list(published_cache_kinds(load_config(ctx.cfg,
                                                                allow_uncertified=True)))
        ctx.log(f"  compile cache published for: {warm_kinds or 'no pool (cold starts)'}")
    return {"ok": p.returncode == 0, "options": rows, "video_s": video,
            "projection": {"usd": proj.usd, "usd_upper": proj.usd_upper},
            "warm_kinds": warm_kinds, "tracked_usd": 0.0, "apps": []}


def _app_name(ctx: Ctx) -> str:
    from flora_deploy.config.receipts import resolve
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import plan_app

    c = load_config(ctx.cfg, allow_uncertified=True)
    if ctx.mode == "fake-modal":
        return plan_app(c, determinism="nvidia", fake_models=True).app_name
    return plan_app(c, determinism="nvidia", derived_receipts=resolve(c)).app_name


def _modal_flags(ctx: Ctx) -> list[str]:
    return ["--allow-uncertified", "--determinism", "nvidia",
            *(["--fake-models"] if ctx.mode == "fake-modal" else [])]


def run_deploy(ctx: Ctx) -> dict[str, Any]:
    if ctx.mode == "local":
        return {"ok": True, "skipped": "local: nothing to deploy", "tracked_usd": 0.0,
                "apps": []}
    app = _app_name(ctx)
    ctx.note_app(app)
    t0 = time.time()
    ctx.sh([_flora(), "deploy", "--config", str(ctx.cfg), "--reconcile-cron", "on",
            *_modal_flags(ctx)], timeout=2400)
    wall = time.time() - t0
    return {"ok": True, "app": app, "wall_s": round(wall, 1),
            "tracked_usd": round(project("deploy", ctx.mode).items[0]["usd_per_h"] / 3600
                                 * wall, 4), "apps": [app]}


class Monitor:
    """Live polling: ``flora status --json`` (timed) and, on Modal, ``modal container list``."""

    def __init__(self, ctx: Ctx, app: str | None, rates: Mapping[str, float]):
        self.ctx, self.app, self.rates = ctx, app, rates
        self.samples: list[dict[str, Any]] = []
        self.status_s: list[float] = []
        self.containers: dict[str, dict[str, float]] = {}
        self.kinds: dict[str, str] = {}  # container_id -> pool kind (attempt markers)
        self.t0 = time.time()
        self.last_poll: float | None = None  # a container last seen then is still alive

    def status(self) -> dict[str, Any] | None:
        t = time.time()
        p = subprocess.run([_flora(), "status", self.ctx.job_id, *self.ctx.store_args(),
                            "--json"], cwd=self.ctx.work, capture_output=True, text=True,
                           timeout=600)
        self.status_s.append(time.time() - t)
        try:
            v = json.loads(p.stdout)
        except ValueError:
            return None
        s = {"t": round(time.time() - self.t0, 1), "state": v.get("state"),
             "counts": v.get("counts"), "active_calls": v.get("active_calls"),
             "orchestrator_alive": v.get("orchestrator_alive"),
             "spend_usd": v.get("spend_usd"), "projected_usd": v.get("projected_usd"),
             "status_s": round(self.status_s[-1], 2)}
        self.samples.append(s)
        return s

    def poll_containers(self) -> int:
        if self.app is None:
            return 0
        p = subprocess.run(["modal", "container", "list", "--json"], capture_output=True,
                           text=True, timeout=120)
        now = time.time()
        if p.returncode == 0:
            self.last_poll = now
        n = 0
        with contextlib.suppress(ValueError):
            for c in json.loads(p.stdout):
                name = c.get("App Name") or c.get("app_name") or c.get("Description") or ""
                cid = c.get("Container ID") or c.get("container_id") or ""
                if name != self.app or not cid:
                    continue
                n += 1
                start = now
                with contextlib.suppress(Exception):
                    from flora_deploy.jobs.budget import parse_utc

                    start = parse_utc(str(c.get("Start Time") or c.get("start_time")))
                rec = self.containers.setdefault(cid, {"start": start, "end": now})
                rec["end"] = now
        return n

    def unmatched(self) -> bool:
        return any(cid not in self.kinds for cid in self.containers)

    def live_workers(self) -> list[str]:
        """Worker containers seen in the last successful poll."""
        if self.last_poll is None:
            return []
        return [cid for cid, c in self.containers.items()
                if cid in self.kinds and c["end"] >= self.last_poll - 1e-6]

    def watch_tail(self, *, until_s: float = TAIL_WATCH_S, every_s: float = TAIL_POLL_S,
                   sleep: Callable[[float], None] = time.sleep) -> None:
        """After the job ends: sample containers every ``every_s`` until no worker container
        is left or ``until_s`` passed, so each one's end is known to ``every_s`` (criterion
        16) before ``modal app stop`` ends them all."""
        t_end = time.time() + until_s
        while time.time() < t_end:
            sleep(every_s)
            self.poll_containers()
            if not self.live_workers():
                return

    def seen(self) -> dict[str, dict[str, Any]]:
        """Containers as ``h0_rehearsal.drain_rows`` reads them: start, and end unless it was
        still alive at the last poll."""
        lp = self.last_poll
        return {cid: {"app": self.app, "started_at": c["start"],
                      "ended_at": None if lp is not None and c["end"] >= lp - 1e-6
                      else c["end"]}
                for cid, c in self.containers.items()}

    def observed_usd(self) -> float:
        """Container-seconds seen so far, each at its own class's $/s (:func:`container_rates`):
        a container an attempt marker names at its pool's rate, any other at the CPU
        (orchestrator / reconciler) rate. A worker still in its cold start is not named yet;
        the cancel rule's slack (every pool container for ``CLIENT_SLACK_S``) covers it."""
        cpu = self.rates.get(CPU_CLASS, 0.0)
        return sum((c["end"] - c["start"]) * self.rates.get(self.kinds.get(cid, CPU_CLASS), cpu)
                   for cid, c in self.containers.items())


def _container_kinds(ctx: Ctx) -> dict[str, str]:
    """``container_id -> kind`` from the job's attempt markers (MODAL_TASK_ID per attempt)."""
    from flora_deploy import sdk
    from flora_deploy.jobs.store import JobPaths

    job = sdk.open_job(ctx.job_id, store=ctx.store_uri)
    m = job._deps().manifests.load()
    paths = JobPaths(ctx.job_id, config_hash=job.config_hash,
                     stage_hashes=dict(m.stage_hashes))
    out: dict[str, str] = {}
    for kind in POOLS:
        for f in job.store.list(paths.area_prefix(job.config_hash, "attempts", kind)):
            with contextlib.suppress(Exception):
                cid = json.loads(job.store.get_bytes(f)).get("container_id")
                if cid:
                    out[str(cid)] = kind
    return out


def _running_container(ctx: Ctx, kind: str) -> str | None:
    """A container running a ``kind`` shard now: the newest open attempt marker."""
    from flora_deploy import sdk
    from flora_deploy.jobs.store import JobPaths

    job = sdk.open_job(ctx.job_id, store=ctx.store_uri)
    m = job._deps().manifests.load()
    paths = JobPaths(ctx.job_id, config_hash=job.config_hash,
                     stage_hashes=dict(m.stage_hashes))
    best: tuple[str, str] | None = None
    for f in job.store.list(paths.area_prefix(job.config_hash, "attempts", kind)):
        with contextlib.suppress(Exception):
            a = json.loads(job.store.get_bytes(f))
            if a.get("ended_utc") or job.store.exists(
                    paths.receipt(job.config_hash, kind, a["shard_id"])):
                continue
            if best is None or a["started_utc"] > best[0]:
                best = (a["started_utc"], a["container_id"])
    return best[1] if best else None


def _orchestrator_call(ctx: Ctx) -> str | None:
    from flora_deploy import sdk

    job = sdk.open_job(ctx.job_id, store=ctx.store_uri)
    launches = job.spec.get("launches") or []
    return launches[-1]["call_id"] if launches else None


def _chaos_modal(ctx: Ctx, action: str) -> dict[str, Any]:
    if action.startswith("preempt_"):
        kind = action.split("_", 1)[1]
        cid = _running_container(ctx, kind)
        if cid is None:
            return {"skipped": f"no running {kind} container found"}
        p = ctx.sh(["modal", "container", "stop", cid], check=False, timeout=120)
        return {"container": cid, "rc": p.returncode}
    import modal

    call = _orchestrator_call(ctx)
    if call is None:
        return {"skipped": "no orchestrator launch recorded"}
    modal.FunctionCall.from_id(call).cancel(terminate_containers=True)
    return {"call_id": call, "killed": True}


def run_run(ctx: Ctx) -> dict[str, Any]:
    """``flora run`` + live polling + chaos + compose; the in-step stop rules."""
    mode, fl = ctx.mode, _flora()
    pools = pools_for(mode)
    rates = container_rates(mode)
    budget = wh.step_budget(ctx.st, "run", caps=caps(mode), global_cap=global_cap(mode))
    if mode != "local" and budget < 0.05:
        return {"ok": False, "error": f"budget left ${budget:.2f}", "tracked_usd": 0.0,
                "apps": []}
    video = float(ctx.st.step("index").result.get("video_s") or VIDEO_S)
    wall_limit = (run_wall_s(mode, video, upper=True, warm=False) + CLIENT_SLACK_S
                  if mode != "local" else 3600.0)
    slack_usd = CLIENT_SLACK_S * sum(rates[k] * n for k, (_, n) in pools.items())
    chaos = Chaos(enabled=CHAOS, fired=dict(ctx.st.step("run").result.get("chaos") or {}))
    app = None if mode == "local" else _app_name(ctx)
    mon = Monitor(ctx, app, rates)
    out: dict[str, Any] = {"app": app, "job_id": ctx.job_id, "budget_usd": round(budget, 4),
                           "wall_limit_s": round(wall_limit)}
    proc: subprocess.Popen | None = None
    orch: dict[str, Any] = {}
    logs = {"run": ctx.work / "flora-run.log", "resume": ctx.work / "flora-resume.log"}
    t0 = out["started_at"] = time.time()  # the run's start: time to peak counts from here
    killed_at: float | None = None
    resume_proc: subprocess.Popen | None = None
    end_state = None
    try:
        pre = [k for k in POOLS if f"preempt_{k}" in CHAOS]
        if mode == "local":
            inject = ",".join(f"preempt:{k}-*x1" for k in pre)
            args = [fl, "run", "--local", "--fake-models", "--config", str(ctx.cfg), "--job",
                    ctx.job_id, "--store", str(ctx.work / "store"),
                    *(["--inject", inject] if inject else []), "--json"]
            chaos.fired.update({f"preempt_{k}": {"inject": inject} for k in pre})
            proc = _popen(args, ctx.work, logs["run"])
        else:
            ctx.note_app(app)  # type: ignore[arg-type]
            if mode == "fake-modal" and pre:
                from flora_deploy.modal_app.workers_app import fake_faults_path

                store = _fake_store(ctx)
                store.put_bytes(fake_faults_path(ctx.job_id),
                                json.dumps({"preempt": dict.fromkeys(pre, 2)}).encode())
                chaos.fired.update({f"preempt_{k}": {"fake_faults": f"{k}@2"} for k in pre})
            ctx.sh([fl, "run", "--config", str(ctx.cfg), "--detach", "--job", ctx.job_id,
                    "--budget", f"{budget:.2f}", "--poll", str(ctx.poll_s), *_modal_flags(ctx)],
                   timeout=900)
        while True:
            mon.poll_containers()
            if app is not None and mon.unmatched():
                with contextlib.suppress(Exception):  # polled again next time
                    mon.kinds.update(_container_kinds(ctx))
            s = mon.status() or {}
            ctx.log(f"  [{s.get('t', 0):7.1f}s] {s.get('state')} {json.dumps(s.get('counts'))} "
                    f"spend ${s.get('spend_usd') or 0:.2f} alive={s.get('orchestrator_alive')}")
            spent = max(float(s.get("spend_usd") or 0.0), mon.observed_usd())
            if killed_at is not None and "revive_s" not in orch:
                t_new = _spawn_after(ctx, killed_at)
                if t_new is not None:
                    orch["revive_s"] = round(t_new - killed_at, 1)
                    orch["revived_utc"] = t_new
                    orch.setdefault("revived_by", "reconciler" if mode != "local" else
                                    "flora resume")
                elif mode != "local" and time.time() - killed_at > RECONCILE_WAIT_S:
                    orch["revived_by"] = "flora resume (reconciler did not revive)"
                    ctx.sh([fl, "resume", ctx.job_id, *ctx.store_args()], check=False)
                    killed_at = time.time()  # measure the fallback from here
            due = chaos.due(s) if s else None
            if due == "kill_orchestrator" and mode == "local":
                if proc is not None and proc.poll() is None:
                    proc.send_signal(signal.SIGKILL)
                    proc.wait()
                killed_at = time.time()
                orch.update(killed=True, at_s=s.get("t"), how="SIGKILL flora run --local",
                            resumes=0, killed_utc=killed_at)
                chaos.fired[due] = {"at_s": s.get("t")}
            # local: `flora resume` is a no-op while the dead orchestrator's lease is unexpired
            # (it reads "alive" for LOCAL_LEASE_TTL_S), so resume once it reads dead
            if mode == "local" and killed_at is not None and "revive_s" not in orch and \
                    (resume_proc is None or resume_proc.poll() is not None) and \
                    orch.get("resumes", 0) < 5 and (not s.get("orchestrator_alive") or
                                                   time.time() - killed_at > LOCAL_LEASE_TTL_S):
                orch["resumes"] = orch.get("resumes", 0) + 1
                resume_proc = _popen([fl, "resume", ctx.job_id, *ctx.store_args()], ctx.work,
                                     logs["resume"])
            elif due is not None and mode != "local":
                res = _chaos_modal(ctx, due)
                chaos.fired[due] = {"at_s": s.get("t"), **res}
                if due == "kill_orchestrator" and res.get("killed"):
                    killed_at = time.time()
                    orch.update(killed=True, at_s=s.get("t"), call_id=res.get("call_id"),
                                killed_utc=killed_at)
            if s.get("state") in wh_terminal() or s.get("state") == "composing":
                if mode != "local" or all(p is None or p.poll() is not None
                                          for p in (proc, resume_proc)):
                    end_state = s.get("state")
                    break
            if mode == "local" and proc is not None and proc.poll() is not None \
                    and resume_proc is None and s.get("state") not in ("running", "planned"):
                end_state = s.get("state")
                break
            if mode != "local" and spent + slack_usd >= budget:
                out["stopped"] = f"observed spend ${spent:.2f} + slack reached ${budget:.2f}"
                ctx.sh([fl, "cancel", ctx.job_id, "--config", str(ctx.cfg)], check=False)
                break
            if time.time() - t0 > wall_limit:
                out["stopped"] = f"wall limit {wall_limit:.0f} s"
                if mode != "local":
                    ctx.sh([fl, "cancel", ctx.job_id, "--config", str(ctx.cfg)], check=False)
                break
            time.sleep(ctx.poll_s)
        if app is not None and end_state is not None:
            mon.watch_tail()  # each worker's end after the job-end drain (criterion 16)
    finally:
        try:
            for p in (proc, resume_proc):
                if p is not None:
                    with contextlib.suppress(Exception):
                        p.wait(timeout=600)
            t = time.time()
            try:
                comp = ctx.sh([fl, "compose", "--job", ctx.job_id, "--store", ctx.store_uri],
                              check=False, timeout=3600)
                out["compose_rc"] = comp.returncode
            finally:
                out["compose_s"] = round(time.time() - t, 2)
        finally:  # whatever compose did: the app and its reconciler cron go down (M4)
            if app is not None:
                try:
                    ctx.sh(["modal", "app", "stop", "--yes", app], check=False, timeout=120)
                finally:
                    mark_app(ctx.st, "stopped", ctx.state_path)
    out.update(state=end_state, chaos=chaos.fired, orchestrator=orch, samples=mon.samples,
               status_s=[round(x, 2) for x in mon.status_s], containers=mon.containers,
               container_kinds=mon.kinds, seen=mon.seen(), last_poll=mon.last_poll,
               wall_s=round(time.time() - t0, 1),
               observed_usd=round(mon.observed_usd(), 4))
    rep = subprocess.run([fl, "report", ctx.job_id, *ctx.store_args(), "--json"], cwd=ctx.work,
                         capture_output=True, text=True, timeout=900)
    try:
        r = json.loads(rep.stdout)
        out.update(spend=r.get("spend", {}).get("total_usd"), report_state=r.get("state"))
    except ValueError:
        out["report_error"] = (rep.stdout + rep.stderr)[-400:]
    if mode == "local":
        out["tracked_usd"] = 0.0
    else:
        tracked = max(float(out.get("spend") or 0.0), out["observed_usd"])
        if out.get("report_error"):
            tracked = max(tracked, budget)
        out["tracked_usd"] = round(tracked, 4)
    out["ok"] = end_state in ("done", "composing") and "stopped" not in out
    out["apps"] = [app] if app else []
    return out


def wh_terminal() -> tuple[str, ...]:
    return ("done", "completed_with_errors", "failed", "cancelled", "budget_stopped")


def _popen(args: list[str], cwd: Path, log: Path) -> subprocess.Popen:
    """A background ``flora`` process whose output goes to ``log`` (never a pipe nobody
    reads: a full pipe would block it)."""
    print(f"$ {' '.join(args)} > {log}", flush=True)
    f = log.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    finally:
        f.close()  # the child holds its own descriptor


def _spawn_after(ctx: Ctx, t: float) -> float | None:
    from flora_deploy import sdk

    try:
        job = sdk.open_job(ctx.job_id, store=ctx.store_uri)
        files = read_shard_files(job.store, ctx.job_id, job.config_hash,
                                 job._deps().manifests.load())
    except Exception:  # noqa: BLE001 - polled again next time
        return None
    return first_spawn_after(files["ledgers"], t)


def cleanup_fake_modal(ctx: Ctx) -> list[str]:
    """Delete the fake-modal run's two Volumes (named per run; never another run's)."""
    notes = []
    for vol in (f"{STORE_VOLUME['fake-modal']}-{ctx.run}",
                f"{MEDIA_VOLUME['fake-modal']}-{ctx.run}"):
        p = subprocess.run(["modal", "volume", "delete", vol, "--yes"], capture_output=True,
                           text=True)
        notes.append(f"modal volume delete {vol}: rc={p.returncode}")
    return notes


def _fake_store(ctx: Ctx) -> Any:
    import modal

    from flora_deploy.sdk import RemoteVolumeStore

    vol = modal.Volume.from_name(f"{STORE_VOLUME['fake-modal']}-{ctx.run}", version=2)
    return RemoteVolumeStore(vol, ctx.run)


def run_report(ctx: Ctx) -> dict[str, Any]:
    from flora_deploy import sdk

    fl = _flora()
    timings: dict[str, Any] = {}
    t = time.time()
    st = subprocess.run([fl, "status", ctx.job_id, *ctx.store_args(), "--json"], cwd=ctx.work,
                        capture_output=True, text=True, timeout=600)
    timings["status_final_s"] = round(time.time() - t, 2)
    t = time.time()
    rp = subprocess.run([fl, "report", ctx.job_id, *ctx.store_args(), "--json"], cwd=ctx.work,
                        capture_output=True, text=True, timeout=900)
    timings["report_s"] = round(time.time() - t, 2)
    run = ctx.st.step("run").result
    timings["status"] = _quant(run.get("status_s") or [])
    timings["compose_s"] = run.get("compose_s")
    timings["compose_rc"] = run.get("compose_rc")
    rep: dict[str, Any] = {}
    with contextlib.suppress(ValueError):
        rep = json.loads(rp.stdout)
    job = sdk.open_job(ctx.job_id, store=ctx.store_uri)
    manifest = job._deps().manifests.load()
    files = read_shard_files(job.store, ctx.job_id, job.config_hash, manifest)
    dur = {e.uid: e.duration_s for e in manifest.episodes}
    o = run.get("orchestrator") or {}
    outage = ((o["killed_utc"], o.get("revived_utc") or o["killed_utc"] + RECONCILE_WAIT_S)
              if o.get("killed_utc") else None)
    meas = measure(files, dur, run.get("containers") or None,
                   usd_per_s=container_rates(ctx.mode), run_start=run.get("started_at"),
                   outage=outage)
    summary = job.summary() or {}
    stages = summary.get("stages", {})
    records = {"missing": sum(int(c.get("missing", 0)) for c in stages.values()),
               "duplicates": len(summary.get("conflicts", [])),
               "duplicates_resolved": summary.get("duplicates_resolved"),
               "by_stage": {k: {kk: c.get(kk) for kk in ("records", "expected", "missing",
                                                         "excused")}
                            for k, c in stages.items()}}
    proj = ctx.st.step("run")
    actual = proj.spent_at(ctx.st.watermark)  # tracked, or the billed delta once it lands
    pe: dict[str, Any] = {}
    if proj.projected_usd:
        pe = {"actual_usd": round(actual, 4), "billed": proj.billed_usd is not None,
              "tracked_usd": float(run.get("tracked_usd") or 0.0),
              "nominal_usd": proj.projected_usd,
              "upper_usd": proj.projected_upper_usd,
              "vs_nominal": round(actual / proj.projected_usd - 1, 3),
              "vs_upper": round(actual / (proj.projected_upper_usd or 1) - 1, 3),
              "orchestrator_first_projection_usd": next(
                  (s.get("projected_usd") for s in run.get("samples", [])
                   if s.get("projected_usd")), None),
              "flora_plan_options": ctx.st.step("plan").result.get("options")}
    res = {"state": run.get("state"), "status_rc": st.returncode, "report_rc": rp.returncode,
           "report_spend": rep.get("spend", {}).get("total_usd"),
           "usd_per_video_hour": rep.get("usd_per_video_hour"), "measure": meas,
           "records": records, "projection_error": pe, "timings": timings,
           "orchestrator": run.get("orchestrator"), "chaos": run.get("chaos"),
           "launches": len(job.spec.get("launches") or []), "sizes": pool_sizes(),
           "startup_variant": "warm" if ASSUME_WARM else "cold",
           "profile": PROFILE.name, "chaos_enabled": list(CHAOS),
           "episodes": PROFILE.episodes}
    if ctx.mode != "local":
        res.update(proof_inputs(job, ctx, run, rep, files))
    res["criteria"] = evaluate(res, ctx.mode)
    res.update(ok=rp.returncode in (0, 3) and bool(meas["pools"]), tracked_usd=0.0, apps=[])
    return res


def proof_inputs(job: Any, ctx: Ctx, run: Mapping[str, Any], rep: Mapping[str, Any],
                 files: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """What criteria 15-19 and ``--finalize`` read, from the job store (the orchestrator's
    drain and pre-warm records, the shard audit rows) and ``flora report``'s per-container
    rows: ``drain`` (``h0_rehearsal.drain_rows``), ``prewarm``, ``cache``, ``batch_fill`` and
    ``report_containers``."""
    from flora_deploy.jobs.pools import read_drains, read_prewarm_plans
    from flora_deploy.jobs.store import JobPaths

    root = JobPaths(ctx.job_id).root
    drains, prewarms = read_drains(job.store, root, job.config_hash)
    plans = read_prewarm_plans(job.store, root, job.config_hash)
    report_rows = [dict(c) for c in (rep.get("spend") or {}).get("containers") or ()]
    seen = dict(run.get("seen") or {})
    out: dict[str, Any] = {"report_containers": report_rows,
                           "warm_kinds": list(ctx.st.step("plan").result.get("warm_kinds") or ())}
    if seen:
        bound = DRAIN_BOUND_S
        out["drain"] = h0.drain_rows(str(run.get("app")), report_rows, seen,
                                     {k: tuple(v) for k, v in drains.items()}, {},
                                     observed_until=run.get("last_poll"), bound_s=bound)
    ok, detail = h0.prewarm_check(plans, prewarms, report_rows, seen)
    out["prewarm"] = {"ok": ok, "detail": detail, "plan": plans.get("caption")}
    audits = h0.job_audits(job.store, root)
    out["cache"] = cache_by_kind(audits, list(files.get("receipts") or ()), out["warm_kinds"])
    out["batch_fill"] = batch_fill(audits)
    return out


RUNNERS: dict[str, Callable[[Ctx], dict[str, Any]]] = {
    "verify": run_verify, "index": run_index, "plan": run_plan, "deploy": run_deploy,
    "run": run_run, "report": run_report}


# ---------------------------------------------------------------------------------------------
# Report / plan printing


def recheck(st: wh.State, mode: str) -> list[dict[str, Any]]:
    """The criteria from the recorded ``report`` result, with the ``run`` step's spend as it
    stands now (the billed delta once Modal billing has landed), so ``--report`` re-checks
    criteria 7 and 8 without re-running anything."""
    rep = st.steps.get("report")
    if not rep or not rep.result.get("criteria"):
        return []
    res = dict(rep.result)
    pe = dict(res.get("projection_error") or {})
    run = st.steps.get("run")
    if pe and run is not None and mode != "local":
        actual = run.spent_at(st.watermark)
        pe.update(actual_usd=round(actual, 4), billed=run.billed_usd is not None)
        if pe.get("nominal_usd"):
            pe["vs_nominal"] = round(actual / pe["nominal_usd"] - 1, 3)
        res["projection_error"] = pe
    return evaluate(res, mode)


def render_report(st: wh.State, mode: str, projs: Mapping[str, wh.Projection]) -> str:
    c = caps(mode)
    lines = [f"# Scale test report ({PROFILE.name} profile, {mode})", "",
             "Generated by `scripts/scale_test.py` (plan: `docs/scale-test.md`). Numbers only.", "",
             f"Run `{st.run_id}`: spent **${st.spent:.2f}** of the ${global_cap(mode):.2f} "
             "global cap.", "",
             "| step | status | cap | projected (upper) | attempts | tracked |",
             "|---|---|---|---|---|---|"]
    for name in STEPS:
        s = st.steps.get(name, wh.StepState())
        p = projs[name]
        lines.append(f"| {TITLES[name]} | {s.status} | ${c[name]:.2f} | ${p.usd:.2f} "
                     f"(${p.usd_upper:.2f}) | {len(s.attempts)} | ${s.tracked_usd:.2f} |")
    rep = st.steps.get("report")
    crit = recheck(st, mode)
    if rep and crit:
        lines += ["", "## Success criteria", "", f"**{criteria_summary(crit)}**", "",
                  "| # | criterion | value | target | verdict | note |",
                  "|---|---|---|---|---|---|"]
        for x in crit:
            how = x["reason"] or ("" if x["auto"] else "manual")
            lines.append(f"| {x['id']} | {x['criterion']} | {x['value']} | {x['target']} | "
                         f"{x['verdict']} | {how} |")
        if mode == "real":
            samples, notes, refusals = finalize_samples(st)
            lines += ["", "## Rate samples (`--finalize` folds these into planner/rates.json)",
                      ""]
            for (tbl, kind), xs in sorted(samples.items()):
                vals = ", ".join(f"{x['value']:g}" for x in xs)
                lines.append(f"- {kind} {tbl}: {len(xs)} sample(s): {vals}")
            lines += [f"- note: {n}" for n in notes] + [f"- refused: {r}" for r in refusals]
        for key in ("measure", "projection_error", "timings", "orchestrator", "chaos",
                    "records", "drain", "prewarm", "cache", "batch_fill"):
            lines += ["", f"## {key}", "", "```json",
                      json.dumps(rep.result.get(key), indent=1, default=str)[:8000], "```"]
    for name in STEPS:
        s = st.steps.get(name)
        if s and s.notes:
            lines += ["", f"Notes ({name}):", *[f"- {n}" for n in s.notes]]
    return "\n".join(lines) + "\n"


def print_plan(st: wh.State, mode: str, projs: Mapping[str, wh.Projection], out=print) -> None:
    c, g = caps(mode), global_cap(mode)
    out(f"Scale test, {PROFILE.name} profile ({PROFILE.blurb}; --profile "
        f"{' | '.join(PROFILES)}), {mode}, run {st.run_id}: spent ${st.spent:.2f} of ${g:.2f}")
    if mode == "real":
        d = DATASET
        sub = PROFILE.episodes < d["total_episodes"]
        durs = [f / d["fps"] for f in EPISODE_FRAMES[:PROFILE.episodes]]
        out(f"  dataset {d['repo']}@{d['revision'][:12]} {d['path']}: "
            + (f"the first {PROFILE.episodes} of {d['total_episodes']} episodes by episode "
               "index (flora index --limit), " if sub else f"{d['total_episodes']} episodes, ")
            + f"{VIDEO_S:.0f} video-s ({VIDEO_S / 3600:.3f} h; episodes {min(durs):.1f}-"
              f"{max(durs):.1f} s, median {statistics.median(durs):.1f} s), "
              f"{d['codec']} {d['resolution']}, embodiment {d['embodiment']}")
    else:
        out(f"  synthetic: {FAKE_EPISODES} episodes x {FAKE_MIN_S:g}-{FAKE_MAX_S:g} s "
            f"(~{VIDEO_S:.0f} video-s)")
    out("  pools: " + ", ".join(f"{k} {v['max_containers']} x "
                               f"{v['gpu'] if mode == 'real' else 'CPU fake worker'}"
                               for k, v in POOLS.items())
        + f"; shards {'' if PROFILE.packing == 'exact' and mode == 'real' else '~'}"
          f"{shard_counts(mode=mode)} at {SHARD_TARGET_GPU_MIN:g} GPU-min; chaos "
          f"{', '.join(CHAOS) or 'none'}")
    for name in STEPS:
        p = projs[name]
        s = st.steps.get(name, wh.StepState())
        d = rb.decide(st, STEPS, name, projs, caps=c, global_cap=g)  # what a run decides
        rule = {"run": "ok", "done": "done"}.get(d.action) or f"{d.action.upper()}: {d.message}"
        out(f"  {TITLES[name]:42s} [{s.status}] cap ${c[name]:.2f} nominal ${p.usd:.2f} "
            f"upper ${p.usd_upper:.2f}  rule: {rule}")
        for i in p.items:
            out(f"      {i['what']:62s} {i['gpu']:>4s} {i['seconds']:6.0f} s / "
                f"{i['upper_seconds']:6.0f} s x ${i['usd_per_h']:.2f}/h = ${i['usd']:.3f} "
                f"(upper ${i['usd_upper']:.3f})")
    nominal = sum(p.usd for p in projs.values())
    upper = sum(p.usd_upper for p in projs.values())
    honest = wh.honest_caps(projs)
    out(f"  total: nominal ${nominal:.2f}, upper ${upper:.2f}; global cap ${g:.2f}; honest "
        f"caps {honest}")
    how = ("overridden" if mode in CAPS_USD else
           f"each step's upper projection, ceil ${CAP_STEP_USD:.2f}; global = their sum")
    out(f"  caps ({how}): " + ", ".join(f"{k} ${v:.2f}" for k, v in c.items())
        + f"; global ${g:.2f}")
    if mode != "local":
        out(f"  wall: nominal {run_wall_s(mode) / 60:.0f} min, upper "
            f"{run_wall_s(mode, upper=True) / 60:.0f} min")
    if upper > g + 1e-9:
        out(f"  NOT HONEST AT ${g:.2f}: the upper total needs ${honest['global']:.2f}")
    if mode == "real":
        print_variants(mode, out)


def print_variants(mode: str, out=print) -> None:
    """Cold vs warm start, side by side (caps always come from the cold projection)."""
    for warm in (False, True):
        ps = projections(mode, warm=warm)
        r = kind_rates(mode, warm=warm)
        st = ", ".join(f"{k} {v.startup_s:.0f} s {v.startup_status}" for k, v in r.items())
        out(f"  {'warm' if warm else 'cold'} start ({st}): run nominal ${ps['run'].usd:.2f} "
            f"upper ${ps['run'].usd_upper:.2f}; total nominal "
            f"${sum(p.usd for p in ps.values()):.2f} upper "
            f"${sum(p.usd_upper for p in ps.values()):.2f}; wall "
            f"{run_wall_s(mode, warm=warm) / 60:.0f} / "
            f"{run_wall_s(mode, upper=True, warm=warm) / 60:.0f} min")


def print_size_options(out=print) -> None:
    """The lite size options (``--sizing`` with a subset profile): episodes x pools."""
    rows = size_options()
    rec = recommend_size(rows)
    out(f"  size options (cold start, all steps; recommended: >= 2 containers per pool that "
        f"the schedule reaches, >= 2 shards per container, pipelined, upper <= "
        f"${LITE_UPPER_MAX_USD:.2f}; then the most video):")
    out("    episodes video-h  seg cap  shards seg/cap  nominal   upper  wall  upper  peak")
    for x in rows:
        mark = " <- recommended" if rec is x else ""
        out(f"    {x['episodes']:8d} {x['video_s'] / 3600:7.3f}  {x['segment']:3d} "
            f"{x['caption']:3d}  {x['shards']['segment']:6d}/{x['shards']['caption']:<6d}  "
            f"${x['usd']:6.2f} ${x['usd_upper']:6.2f} {x['wall_s'] / 60:4.0f}m "
            f"{x['wall_upper_s'] / 60:5.0f}m  {x['peak']['segment']}/{x['peak']['caption']}"
            f"{mark}")


def print_sizing(out=print) -> None:
    """The pool-size trade-off (``--sizing``)."""
    if PROFILE.name != "full":
        print_size_options(out)
        return
    for warm in (False, True):
        rows = pool_sizing(warm=warm)
        rec = recommend(rows)
        out(f"  pool sizing ({'warm' if warm else 'cold'} start; run step with chaos, and the "
            "bare schedule):")
        out("    seg cap  nominal   upper  wall  upper  | no-chaos $  wall | util seg/cap "
            "(life)  idle  first-cap")
        for x in sorted(rows, key=lambda x: (x["segment"], x["caption"])):
            mark = " <- recommended" if rec and (x["segment"], x["caption"]) == (
                rec["segment"], rec["caption"]) else ""
            u = x["utilization"]
            out(f"    {x['segment']:3d} {x['caption']:3d}  ${x['usd']:6.2f} "
                f"${x['usd_upper']:6.2f} {x['wall_s'] / 60:4.0f}m "
                f"{x['wall_upper_s'] / 60:5.0f}m  | ${x['no_chaos_usd']:6.2f} "
                f"{x['no_chaos_wall_s'] / 60:4.0f}m | {u['segment']:.2f}/{u['caption']:.2f}"
                f"          {x['idle_share']:.3f}  "
                f"{x['first_caption_after_first_segment_s']:4d} s{mark}")


def print_options(out=print) -> None:
    try:
        opts = planner_options()
    except Exception as e:  # noqa: BLE001 - informational only
        out(f"  flora plan options unavailable offline: {e!r}")
        return
    out("  flora plan options (offline, same rates.json; cold startup):")
    for o in opts:
        kinds = ", ".join(f"{k.kind} {k.containers}x{k.gpu}" for k in o.kinds)
        out(f"    {o.name:28s} ${o.cost_usd:7.2f} (${o.cost_low_usd:.2f}-${o.cost_high_usd:.2f})"
            f" {o.wall_s / 3600:5.2f} h  {kinds}  [{o.status}; {o.rate_status}]")


def run_projection(st: wh.State, mode: str, projs: Mapping[str, wh.Projection]
                   ) -> wh.Projection:
    """The ``run`` step's projection, at the indexed video length once ``index`` ran."""
    v = st.step("index").result.get("video_s")
    return project("run", mode, float(v)) if v else projs["run"]


# ---------------------------------------------------------------------------------------------
# Finalize: the run's per-container rates into planner/rates.json, as more samples

RATES_PATH = ROOT / "src" / "flora_deploy" / "planner" / "rates.json"
COSTS_DOC = ROOT / "docs" / "costs.md"


def _wf() -> Any:
    import wave_h_finalize as wf  # scripts/, on sys.path above: the shared fold

    return wf


def finalize_samples(st: wh.State) -> tuple[dict[tuple[str, str], list[dict[str, Any]]],
                                            list[str], list[str]]:
    """``({(table, kind): samples}, notes, refusals)`` from a finished real run's ``report``
    step. ``table`` is ``rate`` or ``startup:<cold|warm>``. One rate sample per container
    that finished a shard: ``flora report``'s busy seconds (``wave_h_finalize.RECEIPT_FIELDS``)
    over the video seconds its receipts finished; at least :data:`MIN_CONTAINERS_MEASURED`
    usable containers per pool, else that pool is refused. One startup sample per container
    that was not pre-warmed, filed under the compile-cache state every shard of its pool
    named (a pool with mixed or unnamed states folds no startup). Nothing is folded unless
    the job ended ``done`` with no lost or duplicate record."""
    wf = _wf()
    rep = st.steps.get("report")
    res = rep.result if rep and rep.status == "done" else {}
    if not res:
        return {}, [], ["no finished report step in the state"]
    refusals: list[str] = []
    if res.get("state") != "done":
        refusals.append(f"the job ended {res.get('state')!r}, not 'done'")
    rec = res.get("records") or {}
    if not rec or int(rec.get("missing", 0)) + int(rec.get("duplicates", 0)):
        refusals.append(f"records not clean: {rec or 'not recorded'}")
    if refusals:
        return {}, [], refusals
    t_done = rep.attempts[-1].finished if rep.attempts else None
    date = time.strftime("%Y-%m-%d", time.gmtime(t_done or time.time()))
    source = f"docs/scale-test.md (scale test {res.get('profile', PROFILE.name)} run {st.run_id})"
    rows = {str(c["container_id"]): c for c in res.get("report_containers") or ()}
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    notes: list[str] = []
    for kind in POOLS:
        gpu = POOLS[kind]["gpu"].rstrip("!")
        rate, starts = [], []
        for c in (res.get("measure") or {}).get("containers") or ():
            if c.get("kind") != kind:
                continue
            cid = str(c["container_id"])
            r = rows.get(cid)
            why = None
            if r is None or any(r.get(f) is None for f in wf.RECEIPT_FIELDS):
                why = f"no `flora report` row with {', '.join(wf.RECEIPT_FIELDS)}"
            elif not wf.same_gpu(wf.ran_on(r), gpu):
                why = f"ran on {wf.ran_on(r)!r}, not {gpu}"
            elif not c.get("shards") or float(c.get("video_s") or 0) < wf.MIN_VIDEO_S:
                why = f"{c.get('video_s')} video-s finished (< {wf.MIN_VIDEO_S:.0f})"
            if why:
                notes.append(f"{kind} container {cid[-8:]}: dropped: {why}")
                continue
            base = {"source": source, "date": date, "unit": "container", "n": 1,
                    "container": cid[-12:], "gpu": wf.ran_on(r)}
            rate.append({**base, "value": round(float(r["busy_s"]) / float(c["video_s"]), 4),
                         "video_s": round(float(c["video_s"]), 1)})
            if c.get("prewarmed"):
                notes.append(f"{kind} container {cid[-8:]}: pre-warmed, so its spawn-to-start "
                             "is not a startup; no startup sample")
            else:
                starts.append({**base, "value": round(float(r["startup_s"]), 1)})
        if len(rate) < MIN_CONTAINERS_MEASURED:
            refusals.append(f"{kind}: {len(rate)} usable container(s); at least "
                            f"{MIN_CONTAINERS_MEASURED}")
            continue
        out[("rate", kind)] = rate
        states = set(((res.get("cache") or {}).get(kind) or {}).get("states") or {})
        if starts and len(states) == 1 and states <= {"cold", "warm"}:
            (state,) = states
            out[(f"startup:{state}", kind)] = [{**s, "compile_cache": state} for s in starts]
        elif starts:
            notes.append(f"{kind}: compile-cache states {sorted(states) or 'not recorded'}: "
                         "the startups cannot be filed under one row; none folded")
    return out, notes, refusals


def folded_rates(data: Mapping[str, Any], samples: Mapping[tuple[str, str], list[dict]]
                 ) -> dict[str, Any]:
    """``rates.json`` with ``samples`` folded into the ``balanced`` rows
    (``wave_h_finalize.fold_samples``): the rate row of each pool, and the startup row of the
    compile-cache state the samples were timed with. The rows' earlier measurements (the
    certified run's) stay as their ``base`` sample, unchanged."""
    wf = _wf()
    new = json.loads(json.dumps(data))
    for (table, kind), xs in samples.items():
        if table == "rate":
            rows = [r for r in new["rates"] if r["preset"] == "balanced" and r["kind"] == kind]
        else:
            cache = table.split(":", 1)[1]
            rows = [r for r in new["startup"] if r["preset"] == "balanced" and r["kind"] == kind
                    and (r.get("compile_cache") == cache
                         or (cache == "cold" and r.get("compile_cache") is None))]
        if len(rows) != 1:
            raise wf.FoldRefused(f"rates.json: expected one balanced {kind} {table} row, "
                                 f"found {len(rows)}")
        folded = wf.fold_samples(rows[0], xs, kind="rate" if table == "rate" else "startup")
        rows[0].clear()
        rows[0].update(folded)
    new["updated"] = max([str(new.get("updated") or "")]
                         + [str(x["date"]) for xs in samples.values() for x in xs])
    return new


def finalize_main(args: argparse.Namespace, out=print) -> int:
    """``--finalize``: print what folding the run's samples changes; ``--write`` applies it
    (``rates.json``, and the generated tables of ``docs/costs.md`` when it is the repository's
    rates file). Exit 0 written or nothing to do, 1 refused, 2 bad arguments."""
    from flora_deploy.planner.rates import RateTable

    wf = _wf()
    if args.mode != "real":
        out("--finalize folds a real run only (fake and local runs measure no GPU rates)")
        return 2
    spath = (args.state or state_path(args.mode)).resolve()
    if not spath.exists():
        out(f"{spath}: no state")
        return 1
    st = wh.load_state(spath)
    samples, notes, refusals = finalize_samples(st)
    for n in notes:
        out(f"  note: {n}")
    for r in refusals:
        out(f"  REFUSED: {r}")
    if not samples:
        out("nothing folded")
        return 1
    rates = (args.rates or RATES_PATH).resolve()
    data = json.loads(rates.read_text(encoding="utf-8"))
    try:
        new = folded_rates(data, samples)
        table = RateTable.from_dict(new, source=str(rates))  # the loader must accept it
    except (wf.FoldRefused, ValueError) as e:
        out(f"  REFUSED: {e}")
        return 1
    for (tbl, kind) in samples:
        key = "rates" if tbl == "rate" else "startup"
        old = [r for r in data[key] if r["preset"] == "balanced" and r["kind"] == kind]
        now = [r for r in new[key] if r["preset"] == "balanced" and r["kind"] == kind]
        f = wf.VALUE_FIELD["rate" if tbl == "rate" else "startup"]
        for o, n in zip(old, now, strict=True):
            if o != n:
                out(f"  {kind} {tbl}: {o.get(f)} (n={o.get('n')}, +/-{o.get('uncertainty')}, "
                    f"{o.get('status')}) -> {n.get(f)} (n={n.get('n')}, "
                    f"+/-{n.get('uncertainty')})")
    text = wf.dump_json(new)
    outputs = {rates: text}
    if args.rates is None and COSTS_DOC.exists():
        import cost_table  # scripts/

        outputs[COSTS_DOC] = cost_table.render(COSTS_DOC.read_text(encoding="utf-8"),
                                               table=dataclasses.replace(
                                                   table, source="planner/rates.json"))
    changed = {p: t for p, t in outputs.items() if p.read_text(encoding="utf-8") != t}
    if not changed:
        out("  nothing to change (already folded)")
        return 0
    for p in changed:
        out(f"  {'update' if args.write else 'would update'} {p}")
    if args.write:
        for p, t in changed.items():
            p.write_text(t, encoding="utf-8")
    else:
        out("dry run: nothing written (pass --write to apply)")
    return 0


def state_path(mode: str) -> Path:
    """``reports/scale-test/<mode>/`` for ``full`` (where it always was), else
    ``reports/scale-test/<profile>/<mode>/``."""
    base = ROOT / "reports" / "scale-test"
    return (base if PROFILE.name == "full" else base / PROFILE.name) / mode / "state.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=MODES, default="real")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print the plan (the default)")
    g.add_argument("--yes-spend", action="store_true", help="run (Modal modes spend money)")
    g.add_argument("--execute", action="store_true", help="run --mode local (spends nothing)")
    g.add_argument("--report", action="store_true", help="rewrite the report from the state")
    ap.add_argument("--only", choices=STEPS)
    ap.add_argument("--verify", action="store_true", help="dry run: also check the dataset pin")
    ap.add_argument("--episodes", type=int, default=None, help="fake modes: synthetic episodes")
    ap.add_argument("--poll", type=float, default=None, help="seconds between status polls")
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="fake-modal: keep the run's Volumes")
    ap.add_argument("--assume-warm", action="store_true",
                    help="price the warm-start variant (seeded compile cache); caps stay cold")
    ap.add_argument("--sizing", action="store_true",
                    help="dry run: also print the size options (lite: episodes x pools; "
                         "full: 1-4 per pool)")
    ap.add_argument("--profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE,
                    help=f"test size (default {DEFAULT_PROFILE}): "
                    + "; ".join(f"{k}: {p.blurb}" for k, p in PROFILES.items()))
    g.add_argument("--finalize", action="store_true",
                   help="fold a finished real run's per-container rates into "
                        "planner/rates.json as more samples (dry run unless --write)")
    ap.add_argument("--write", action="store_true", help="--finalize: write rates.json")
    ap.add_argument("--rates", type=Path, default=None,
                    help="--finalize: the rates file (default planner/rates.json)")
    args = ap.parse_args(argv)
    global ASSUME_WARM
    prev, ASSUME_WARM = ASSUME_WARM, bool(args.assume_warm)
    try:
        with profile_ctx(args.profile):
            if args.finalize:
                return finalize_main(args)
            return _main(args)
    finally:
        ASSUME_WARM = prev


def _main(args: argparse.Namespace) -> int:
    mode = args.mode
    spath = (args.state or state_path(mode)).resolve()
    work = (args.work or spath.parent / "work").resolve()
    st = wh.load_state(spath)
    go = args.yes_spend or (args.execute and mode == "local")
    if args.execute and mode != "local":
        print("--execute runs --mode local only; Modal modes need --yes-spend")
        return 2
    killed = st.mark_killed() if go else []
    projs = projections(mode)
    print_plan(st, mode, projs)
    if mode == "real" and not go:
        print_options()
        if args.sizing:
            print_sizing()
    report_path = spath.parent / "report.md"
    if args.report:
        report_path.write_text(render_report(st, mode, projs))
        print(f"wrote {report_path}")
        return 0
    if not go:
        if args.verify and mode == "real":
            v = verify_dataset()
            print(f"  dataset pin: {'OK' if v['ok'] else 'CHANGED'} {v['problems']}")
        print("dry run: nothing started (--yes-spend after the user approves; --mode local "
              "--execute for the local rehearsal)")
        return 0
    work.mkdir(parents=True, exist_ok=True)
    if args.episodes and mode != "real":
        st.step("verify").result.setdefault("episodes", args.episodes)
    if mode != "local" and not st.billing_baseline:
        try:
            st.billing_baseline = wh.billed_by_app(st.billing_since)
        except Exception as e:  # noqa: BLE001
            print(f"billing baseline unavailable ({e!r}); tracking only")
    rb.stop_killed(st)  # by id (an ephemeral app is never found by name)
    if killed and mode != "local":
        mark_app(st, APP_STOPPED)
    wh.save_state(st, spath)
    os.chdir(work)  # relative input.path is hashed against the cwd
    ctx = Ctx(mode, st, work, spath, args.poll or POLL_S[mode])
    c, gcap = caps(mode), global_cap(mode)
    selected = set(STEPS) if not args.only else {args.only}
    if args.only and args.only != "verify" and st.step("verify").status != "done":
        selected.add("verify")  # nothing is spent past an unchecked dataset pin (L1)
        print("--only: the dataset pin (verify) is not done; checking it first")

    def projections_now() -> dict[str, wh.Projection]:
        return {**projs, "run": run_projection(st, mode, projs)}

    def decide(name: str, p: wh.Projection | None = None) -> rb.Decision:
        return rb.decide(st, STEPS, name, projections_now(), caps=c, global_cap=gcap,
                         redo=name == "deploy")  # skip() said a done deploy runs again

    def refresh(s: wh.State) -> None:
        if mode != "local":
            wh.refresh_billing(s)  # looked up per call (the tests patch it)

    def skip(name: str, s: wh.StepState) -> str | None:
        redeploy = (name == "deploy" and mode != "local" and s.status == "done"
                    and st.step("run").status != "done" and app_state(st) != APP_DEPLOYED)
        if s.status == "done" and not redeploy:
            return "done"
        if redeploy:
            refresh(st)
            if not decide("run").runs:  # no app is brought up for a run its stop rule refuses
                return "no redeploy: the run step's stop rule refuses it"
            print(f"deploy again: the app is {app_state(st) or 'not known to be live'} "
                  "(an earlier run stopped it)")
            s.notes.append(f"redeploy: app {app_state(st) or 'state unknown'}")
        return None

    def refuse(name: str, s: wh.StepState) -> str | None:
        if name == "run" and mode != "local" and app_state(st) != APP_DEPLOYED:
            return (f"the app is not deployed (app {app_state(st)}); run without --only so "
                    "deploy runs first")
        return None

    def run(name: str, ctx: Ctx) -> dict[str, Any]:
        print(f"== {TITLES[name]}", flush=True)
        return RUNNERS[name](ctx)

    def status_of(res: Mapping[str, Any]) -> str:
        if ctx.step == "deploy" and res.get("ok") and mode != "local":
            mark_app(st, APP_DEPLOYED)
        return "done" if res.get("ok") else "failed"

    def on_error(name: str, att: wh.Attempt, exc: BaseException) -> None:
        att.report_error = mode != "local"  # unknown spend: charged its bound or budget
        if name in ("deploy", "run") and mode != "local":
            mark_app(st, APP_STOPPED)  # not known to be live: the next pass deploys

    rc = rb.run_steps(
        st, [n for n in STEPS if n in selected], state_path=spath, ctx=ctx, run=run,
        projection=lambda name: projections_now()[name], decide=decide,
        check_after=lambda name: wh.check_after(st, name, caps=c, global_cap=gcap),
        refresh=refresh, skip=skip, refuse=refuse, status_of=status_of, on_error=on_error,
        attempt_kw=lambda name: {"slack_s": CLIENT_SLACK_S})
    if mode == "fake-modal" and st.step("report").status == "done" and not args.keep:
        st.step("report").notes += cleanup_fake_modal(ctx)
    wh.save_state(st, spath)
    report_path.write_text(render_report(st, mode, projs))
    crit = recheck(st, mode)
    for x in crit:
        print(f"  {x['verdict']:5s} {x['id']:2d}. {x['criterion']}: {x['value']} "
              f"(target {x['target']}){' - ' + x['reason'] if x['reason'] else ''}")
    if crit:
        print(f"  criteria: {criteria_summary(crit)}")
    print(f"wrote {report_path}; state {spath}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
