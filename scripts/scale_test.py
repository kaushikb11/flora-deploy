"""Scale test of ``balanced`` (``docs/scale-test.md``): about one video-hour, several containers
per pool, through the real ``flora index -> plan -> deploy -> run -> report`` path.

Usage::

    uv run python scripts/scale_test.py                          # dry run (the default): plan
    uv run python scripts/scale_test.py --mode local --execute   # local fake run (no Modal)
    uv run python scripts/scale_test.py --mode fake-modal --yes-spend  # CPU rehearsal (~cents)
    uv run python scripts/scale_test.py --mode real --yes-spend  # the GPU run (resumable)
    uv run python scripts/scale_test.py --mode real --report     # rewrite the report

Modes (one state file each, ``reports/scale-test/<mode>/state.json``):

* ``real``: the pinned EgoDex subset (:data:`DATASET`, 128 episodes, 0.980 video-h), ``preset:
  balanced``, ``segment`` on 2 x L40S and ``caption`` on 3 x H100!;
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
VIDEO_S = DATASET["total_frames"] / DATASET["fps"]  # 3529.2 s = 0.980 video-h
HF = "https://huggingface.co"

# Synthetic stand-in (fake modes): media.synth single-view clips, 3..6 s (mean 4.5 s), so the
# total matches VIDEO_S and the manifest packs to the same number of shards.
FAKE_MIN_S, FAKE_MAX_S = 3.0, 6.0
FAKE_EPISODES = round(VIDEO_S / ((FAKE_MIN_S + FAKE_MAX_S) / 2))  # 784

# ---------------------------------------------------------------------------------------------
# Job shape

# 2 + 3: the cheapest size with >= 2 containers per pool whose upper wall is within the hour
# at the measured rates (docs/scale-test.md "Pool sizing"; :func:`pool_sizing`,
# :func:`recommend`). caption (0.906 container-s per video-s, 2.5x segment) is the bottleneck:
# a third segment container only adds a cold start and a scale-down tail.
POOLS = {"segment": {"gpu": "L40S", "max_containers": 2},
         "caption": {"gpu": "H100!", "max_containers": 3}}
MAX_GPUS = 5
# 4 GPU-minutes per shard at the manifest's packing rates (segment 0.5, caption 1.0 GPU-s per
# video-s): ~480 video-s per segment shard (~8 shards), ~240 per caption shard (~15), so each
# segment container runs ~4 shards and captions can start after the first segment shard.
SHARD_TARGET_GPU_MIN = 4.0
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
            "video_s": round(VIDEO_S, 1), "tracked_usd": 0.0, "apps": []}


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
    (``modal_app.orchestrator_app`` resources at the non-preemptible CPU / memory rate; any
    container no attempt marker names is priced as one of these)."""
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.modal_app import orchestrator_app as oa

    if mode == "local":
        return dict.fromkeys((*POOLS, CPU_CLASS), 0.0)
    r = pinned_rates()
    out = {k: r.per_second(c.gpu, 1 if c.gpu else 0, c.cpu, c.memory_gib)
           for k, (c, _) in pools_for(mode).items()}
    out[CPU_CLASS] = r.per_second(None, 0, float(oa.CPU_CORES), oa.MEMORY_MIB / 1024,
                                  preemptible=False)
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


def kind_rates(mode: str, table: Any = None, *, warm: bool | None = None
               ) -> dict[str, KindRate]:
    """Per pool kind: the work rate and the startup the projection uses. ``warm``: the
    warm-start variant (default :data:`ASSUME_WARM`)."""
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
        if warm:
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


def shard_counts(video_s: float = VIDEO_S) -> dict[str, int]:
    return {k: max(1, math.ceil(video_s / shard_video_s(k))) for k in POOLS}


def pool_sizes(sizes: Mapping[str, int] | None = None) -> dict[str, int]:
    return {k: int((sizes or {}).get(k, POOLS[k]["max_containers"])) for k in POOLS}


@dataclass
class PoolSim:
    """One pool in :func:`simulate`: its containers' lives and work."""

    starts: int = 0  # containers started (cold or warm)
    startup_s: float = 0.0
    busy_s: float = 0.0
    wait_s: float = 0.0  # ready but idle between shards (waiting for upstream shards)
    tail_s: float = 0.0  # after its last shard: scale-down window, cut at the app stop
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
              scaledown: float) -> tuple[list[list[Any]], list[tuple[float, float]]]:
    """FIFO inputs ``(ready, seconds)`` on a pool of at most ``n`` containers that each start
    in ``startup`` s and scale down ``scaledown`` s after their last input; a queued input
    starts a new container while the pool is below ``n`` and none is free. Returns the
    containers ``[spawn, ready, free, end]`` (end None while alive) and the input spans, in
    input order."""
    conts: list[list[Any]] = []
    spans: list[tuple[float, float]] = []
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


def simulate(sizes: Mapping[str, int], rates: Mapping[str, KindRate],
             video_s: float = VIDEO_S, *, upper: bool = False) -> dict[str, Any]:
    """Shard-level schedule of the run without chaos: every ``segment`` shard is queued at
    t=0, a ``caption`` shard is queued once the segment shards covering its episodes have
    receipts (per-shard ``depends_on``), each pool autoscales up to its size, containers
    scale down ``WORKER_SCALEDOWN_S`` after their last shard, and the app stops
    :data:`IDLE_AFTER_JOB_S` after the last receipt. ``upper``: rates and startups at
    ``1 + uncertainty``. Returns ``{"wall_s", "pools": {kind: PoolSim}, "pipelining"}``."""
    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S

    def rate(k: str) -> float:
        return rates[k].cs_per_video_s * ((1 + rates[k].uncertainty) if upper else 1.0)

    def start(k: str) -> float:
        return rates[k].startup_s * ((1 + rates[k].startup_unc) if upper else 1.0)

    n_shards = shard_counts(video_s)
    seg_v = video_s / n_shards["segment"]
    cap_v = video_s / n_shards["caption"]
    seg_c, seg_sp = _schedule([(0.0, seg_v * rate("segment"))] * n_shards["segment"],
                              sizes["segment"], start("segment"), WORKER_SCALEDOWN_S)
    seg_end = [e for _, e in seg_sp]
    cap_in = []
    for j in range(n_shards["caption"]):
        lo = int(j * cap_v // seg_v)
        hi = min(n_shards["segment"] - 1, int(((j + 1) * cap_v - 1e-6) // seg_v))
        cap_in.append((max(seg_end[lo:hi + 1]), cap_v * rate("caption")))
    cap_in.sort(key=lambda x: x[0])
    cap_c, cap_sp = _schedule(cap_in, sizes["caption"], start("caption"), WORKER_SCALEDOWN_S)
    wall = max(e for _, e in seg_sp + cap_sp)
    stop = wall + IDLE_AFTER_JOB_S
    out: dict[str, Any] = {"wall_s": wall, "pools": {}}
    for kind, conts, spans in (("segment", seg_c, seg_sp), ("caption", cap_c, cap_sp)):
        p = PoolSim()
        ends = [min(c[3] if c[3] is not None else c[2] + WORKER_SCALEDOWN_S, stop)
                for c in conts]
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


def run_wall_s(mode: str, video_s: float = VIDEO_S, *, upper: bool = False,
               table: Any = None, warm: bool | None = None,
               sizes: Mapping[str, int] | None = None) -> float:
    """Wall seconds of the run: the shard schedule (:func:`simulate`) plus the reconciler gap
    of the orchestrator kill (half the cron period nominal, the whole wait upper)."""
    sim = simulate(pool_sizes(sizes), kind_rates(mode, table, warm=warm), video_s, upper=upper)
    gap = RECONCILE_WAIT_S if upper else RECONCILE_WAIT_S / 2
    return sim["wall_s"] + gap


def project(step: str, mode: str, video_s: float = VIDEO_S, table: Any = None, *,
            warm: bool | None = None, sizes: Mapping[str, int] | None = None
            ) -> wh.Projection:
    """What ``step`` costs in ``mode`` (nominal and upper; see ``docs/scale-test.md``).
    ``warm``: the warm-start variant; ``sizes``: pool sizes other than :data:`POOLS`."""
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
    nom = simulate(n, r, video_s)["pools"]
    up = simulate(n, r, video_s, upper=True)["pools"]
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
        p.add(f"{kind}: idle tail x{a.starts} (scale-down / stop)", c, a.tail_s,
              upper_s=b.starts * (WORKER_SCALEDOWN_S + CLIENT_SLACK_S))
        shard = shard_video_s(kind) * k.cs_per_video_s
        p.add(f"{kind}: 1 preemption (replacement start + lost shard)", c,
              k.startup_s + shard / 2, upper_s=s_up + shard * (1 + k.uncertainty))
        if kind == "caption":  # the orchestrator kill: captions idle, scale down, restart
            p.add(f"caption: reconciler gap x{n[kind]} (idle {WORKER_SCALEDOWN_S} s + "
                  "restart)", c, n[kind] * (WORKER_SCALEDOWN_S + k.startup_s),
                  upper_s=n[kind] * (WORKER_SCALEDOWN_S + s_up))
    orch = wh.Container(None, 0.25 * 3, 1.0 * 3)  # nonpreemptible: 3x CPU/memory
    p.add("orchestrator (non-preemptible CPU) over the wall", orch,
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
                table: Any = None, video_s: float = VIDEO_S) -> list[dict[str, Any]]:
    """The wall-clock / cost trade-off of every ``segment`` x ``caption`` pool size in ``ns``
    (real mode; the run step with its chaos, and the bare schedule without it)."""
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
                                               "episodes": DATASET["total_episodes"]}]})
    req = PlanRequest(workload=w, stages=("segment", "caption"),
                      presets=("balanced", "exact"), max_gpus=MAX_GPUS,
                      budget_usd=global_cap("real"))
    bal = get_preset("balanced").pools
    from dataclasses import replace

    pools = {k: replace(bal[k], gpu=POOLS[k]["gpu"]) for k in POOLS}
    return plan_options(with_pool_overrides(req, "balanced", pools))


# ---------------------------------------------------------------------------------------------
# Chaos (pure decision; the runners act on it)

CHAOS = ("preempt_segment", "preempt_caption", "kill_orchestrator")


@dataclass
class Chaos:
    """Which chaos action is due for a status sample (``flora status --json``). Each fires
    once: a ``segment`` preemption once a segment shard is done and at least two are left, a
    ``caption`` preemption at 25% of the caption shards, an orchestrator kill at 50%."""

    enabled: tuple[str, ...] = CHAOS
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
    return f"""# Scale test ({mode}); generated by scripts/scale_test.py
project: {project}
license_ack: research-only
modal:
  environment: main
  max_gpus: {MAX_GPUS}
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

    j = f"scale-{run_id}"
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
    for a in files.get("attempts", []):
        cid, kind = a["container_id"], a["kind"]
        kind_of[cid] = kind
        t0 = _t(a.get("started_utc"))
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
    return out


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

# The 14 criteria: (number, name). Every one is automatic: computed by :func:`evaluate` from
# what ``run`` and ``report`` record. MANUAL lists the ones that are not (none today); each
# would carry the reason it cannot be read from the recorded data.
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
)
MANUAL: dict[int, str] = {}
LOCAL_NA = "local: calls run on threads, with no containers, cold starts or money"


def criteria_targets(mode: str, sizes: Mapping[str, int] | None = None,
                     table: Any = None) -> dict[str, Any]:
    """The rate-derived targets: from the **upper**, cold-start schedule (:func:`simulate`) of
    the pools as configured, so they move with ``rates.json``. Time to peak: when the upper
    schedule has each pool at its size, plus a poll and :data:`LAUNCH_MARGIN_S`. First
    caption: the upper caption cold start plus the margin."""
    n = pool_sizes(sizes)
    r = kind_rates(mode, table, warm=False)
    up = simulate(n, r, upper=True)
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
    """PASS / FAIL / n.a. for each of the 14 criteria (:data:`CRITERION_NAMES`) from the
    ``run`` + ``report`` results. A value the run did not record is a FAIL ("not
    recorded"): every criterion is meant to be measured. ``n.a.`` is only for criteria that
    cannot apply in ``local`` (``reason`` says why)."""
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

    def pre(k: str, p: Mapping[str, Any]) -> tuple[Any, bool | None]:
        n_pre, n_rec = p.get("preempted_starts", 0), p.get("preempted_recovered", 0)
        if f"preempt_{k}" not in chaos:
            return "not fired", False
        return f"{n_rec}/{n_pre}", n_pre >= 1 and n_rec == n_pre
    v, ok = per_pool(pre)
    add(10, v, ">= 1 preempted start per pool, each shard finished by a later attempt", ok)
    orch = r.get("orchestrator") or {}
    rv = orch.get("revive_s")
    by = "flora resume" if local else "the reconciler"
    if not orch.get("killed"):
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
    return [rows[i] for i, _ in CRITERION_NAMES]


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
    ctx.cfg.write_text(config_text(ctx.mode, project=f"scale-{ctx.run}", run_id=ctx.run,
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
        ctx.note_app("flora-index")
        ctx.sh([fl, "index", "--config", str(ctx.cfg), "--job", ctx.job_id, "--store",
                ctx.store_uri, "--out", str(summary)], timeout=3600)
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
    return {"ok": c.get("indexed", 0) > 0 and not s.get("failed"), "counts": c,
            "video_s": s.get("video_s"), "wall_s": round(time.time() - t0, 1),
            "tracked_usd": round(usd, 4), "apps": []}


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
    return {"ok": p.returncode == 0, "options": rows, "video_s": video,
            "projection": {"usd": proj.usd, "usd_upper": proj.usd_upper},
            "tracked_usd": 0.0, "apps": []}


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
    chaos = Chaos(fired=dict(ctx.st.step("run").result.get("chaos") or {}))
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
        if mode == "local":
            inject = "preempt:segment-*x1,preempt:caption-*x1"
            args = [fl, "run", "--local", "--fake-models", "--config", str(ctx.cfg), "--job",
                    ctx.job_id, "--store", str(ctx.work / "store"), "--inject", inject, "--json"]
            chaos.fired.update({"preempt_segment": {"inject": inject},
                                "preempt_caption": {"inject": inject}})
            proc = _popen(args, ctx.work, logs["run"])
        else:
            ctx.note_app(app)  # type: ignore[arg-type]
            if mode == "fake-modal":
                from flora_deploy.modal_app.workers_app import fake_faults_path

                store = _fake_store(ctx)
                store.put_bytes(fake_faults_path(ctx.job_id),
                                json.dumps({"preempt": {"segment": 2, "caption": 2}}).encode())
                chaos.fired.update({"preempt_segment": {"fake_faults": "segment@2"},
                                    "preempt_caption": {"fake_faults": "caption@2"}})
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
               container_kinds=mon.kinds, wall_s=round(time.time() - t0, 1),
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
    actual = proj.spent_at(ctx.st.billed_at)  # tracked, or the billed delta once it lands
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
           "startup_variant": "warm" if ASSUME_WARM else "cold"}
    res["criteria"] = evaluate(res, ctx.mode)
    res.update(ok=rp.returncode in (0, 3) and bool(meas["pools"]), tracked_usd=0.0, apps=[])
    return res


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
        actual = run.spent_at(st.billed_at)
        pe.update(actual_usd=round(actual, 4), billed=run.billed_usd is not None)
        if pe.get("nominal_usd"):
            pe["vs_nominal"] = round(actual / pe["nominal_usd"] - 1, 3)
        res["projection_error"] = pe
    return evaluate(res, mode)


def render_report(st: wh.State, mode: str, projs: Mapping[str, wh.Projection]) -> str:
    c = caps(mode)
    lines = [f"# Scale test report ({mode})", "",
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
        for key in ("measure", "projection_error", "timings", "orchestrator", "chaos",
                    "records"):
            lines += ["", f"## {key}", "", "```json",
                      json.dumps(rep.result.get(key), indent=1, default=str)[:8000], "```"]
    for name in STEPS:
        s = st.steps.get(name)
        if s and s.notes:
            lines += ["", f"Notes ({name}):", *[f"- {n}" for n in s.notes]]
    return "\n".join(lines) + "\n"


def print_plan(st: wh.State, mode: str, projs: Mapping[str, wh.Projection], out=print) -> None:
    c, g = caps(mode), global_cap(mode)
    out(f"Scale test ({mode}) run {st.run_id}: spent ${st.spent:.2f} of ${g:.2f}")
    if mode == "real":
        d = DATASET
        out(f"  dataset {d['repo']}@{d['revision'][:12]} {d['path']}: "
            f"{d['total_episodes']} episodes, {VIDEO_S:.0f} video-s ({VIDEO_S / 3600:.3f} h), "
            f"{d['codec']} {d['resolution']}, embodiment {d['embodiment']}")
    else:
        out(f"  synthetic: {FAKE_EPISODES} episodes x {FAKE_MIN_S:g}-{FAKE_MAX_S:g} s "
            f"(~{VIDEO_S:.0f} video-s)")
    out("  pools: " + ", ".join(f"{k} {v['max_containers']} x "
                               f"{v['gpu'] if mode == 'real' else 'CPU fake worker'}"
                               for k, v in POOLS.items())
        + f"; shards ~{shard_counts()} at {SHARD_TARGET_GPU_MIN:g} GPU-min")
    for name in STEPS:
        p = projs[name]
        s = st.steps.get(name, wh.StepState())
        try:
            wh.check_before(st, name, p, caps=c, global_cap=g)
            rule = "ok"
        except wh.StopRule as e:
            rule = f"STOP: {e}"
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


def print_sizing(out=print) -> None:
    """The pool-size trade-off (``--sizing``)."""
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


def state_path(mode: str) -> Path:
    return ROOT / "reports" / "scale-test" / mode / "state.json"


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
                    help="dry run: also print the pool-size trade-off (1-4 per pool)")
    args = ap.parse_args(argv)
    global ASSUME_WARM
    prev, ASSUME_WARM = ASSUME_WARM, bool(args.assume_warm)
    try:
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
    for app in killed:
        subprocess.run(["modal", "app", "stop", "--yes", app], capture_output=True, text=True)
        print(f"stopped {app} (left by a killed run)")
    if killed and mode != "local":
        mark_app(st, APP_STOPPED)
    wh.save_state(st, spath)
    os.chdir(work)  # relative input.path is hashed against the cwd
    ctx = Ctx(mode, st, work, spath, args.poll or POLL_S[mode])
    c, gcap = caps(mode), global_cap(mode)
    rc = 0
    selected = set(STEPS) if not args.only else {args.only}
    if args.only and args.only != "verify" and st.step("verify").status != "done":
        selected.add("verify")  # nothing is spent past an unchecked dataset pin (L1)
        print("--only: the dataset pin (verify) is not done; checking it first")
    for name in STEPS:
        if name not in selected:
            continue
        s = st.step(name)
        redeploy = (name == "deploy" and mode != "local" and s.status == "done"
                    and st.step("run").status != "done" and app_state(st) != APP_DEPLOYED)
        if s.status == "done" and not redeploy:
            print(f"skip {name}: done")
            continue
        if mode != "local":
            wh.refresh_billing(st)
        if redeploy:
            try:  # no app is brought up for a run its stop rule would refuse
                wh.check_before(st, "run", run_projection(st, mode, projs), caps=c,
                                global_cap=gcap)
            except wh.StopRule:
                print("skip redeploy: the run step's stop rule refuses it")
                continue
            print(f"deploy again: the app is {app_state(st) or 'not known to be live'} "
                  "(an earlier run stopped it)")
            s.notes.append(f"redeploy: app {app_state(st) or 'state unknown'}")
        p = run_projection(st, mode, projs) if name == "run" else projs[name]
        s.projected_usd, s.projected_upper_usd = p.usd, p.usd_upper
        try:
            wh.check_before(st, name, p, caps=c, global_cap=gcap)
        except wh.StopRule as e:
            s.status, rc = "stopped", 3
            s.notes.append(f"stop rule before: {e}")
            break
        if name == "run" and mode != "local" and app_state(st) != APP_DEPLOYED:
            s.notes.append(f"refused: the app is not deployed (app {app_state(st)}); run "
                           "without --only so deploy runs first")
            print(f"refuse run: {s.notes[-1]}")
            rc = 1
            break
        att = wh.Attempt(started=time.time(), rate_usd_per_s=p.peak_usd_per_s,
                         budget_usd=round(wh.step_budget(st, name, caps=c, global_cap=gcap), 4))
        s.attempts.append(att)
        s.status = "running"
        ctx.step = name
        wh.save_state(st, spath)
        print(f"== {TITLES[name]}", flush=True)
        try:
            with wh.Heartbeat(st, att, spath):
                res = RUNNERS[name](ctx)
            s.result = res
            att.tracked_usd = round(float(res.get("tracked_usd") or 0.0), 4)
            att.report_error = bool(res.get("report_error"))
            att.status = s.status = "done" if res.get("ok") else "failed"
            if name == "deploy" and res.get("ok") and mode != "local":
                mark_app(st, APP_DEPLOYED)
        except Exception as e:  # noqa: BLE001 - recorded; the runbook stops
            att.status = s.status = "failed"
            att.report_error = mode != "local"  # unknown spend: charged its budget
            s.notes.append(f"error: {e!r}")
            if name in ("deploy", "run") and mode != "local":
                mark_app(st, APP_STOPPED)  # not known to be live: the next pass deploys
        att.finished = time.time()
        for a in s.result.get("apps", ()):
            if a not in att.apps:
                att.apps.append(a)
        if mode != "local":
            wh.refresh_billing(st)
        wh.save_state(st, spath)
        try:
            wh.check_after(st, name, caps=c, global_cap=gcap)
        except wh.StopRule as e:
            s.notes.append(f"stop rule after: {e}")
            rc = 3
            break
        if s.status != "done":
            rc = 1
            break
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
