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
  balanced``, ``segment`` on 3 x L40S and ``caption`` on 2 x H100!;
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

POOLS = {"segment": {"gpu": "L40S", "max_containers": 3},
         "caption": {"gpu": "H100!", "max_containers": 2}}
MAX_GPUS = 5
# 4 GPU-minutes per shard at the manifest's packing rates (segment 0.5, caption 1.0 GPU-s per
# video-s): ~480 video-s per segment shard (~8 shards), ~240 per caption shard (~15), so each
# segment container runs ~3 shards and captions can start after the first segment shard.
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
            for s, p in projections(mode).items()}


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


def kind_rates(mode: str, table: Any = None) -> dict[str, KindRate]:
    from flora_deploy.planner.rates import RateTable

    if mode != "real":
        return {k: KindRate(FAKE_RATE[0], FAKE_RATE[1], FAKE_STARTUP[0], FAKE_STARTUP[1],
                            "placeholder") for k in POOLS}
    t = table or RateTable.load()
    out = {}
    for kind in POOLS:
        e = t.rate("balanced", kind, 1, 1)
        st = t.startup_for("balanced", kind)
        out[kind] = KindRate(e.container_s_per_video_s, e.uncertainty, st.startup_s,
                             st.uncertainty, e.status)
    return out


def shard_video_s(kind: str) -> float:
    """Video seconds per shard at the manifest's packing rate (``jobs.manifest``)."""
    from flora_deploy.jobs.manifest import DEFAULT_RATES

    return SHARD_TARGET_GPU_MIN * 60.0 / DEFAULT_RATES[kind]


def shard_counts(video_s: float = VIDEO_S) -> dict[str, int]:
    return {k: max(1, math.ceil(video_s / shard_video_s(k))) for k in POOLS}


def run_wall_s(mode: str, video_s: float = VIDEO_S, *, upper: bool = False,
               table: Any = None) -> float:
    """Wall seconds of the run: captions start after the first segment shard; the job ends when
    both pools drained; the orchestrator kill adds the reconciler gap."""
    r = kind_rates(mode, table)
    n = {k: v[1] for k, v in pools_for(mode).items()}

    def f(k: str) -> tuple[float, float]:
        m = (1 + r[k].uncertainty) if upper else 1.0
        s = r[k].startup_s * ((1 + r[k].startup_unc) if upper else 1.0)
        return s, r[k].cs_per_video_s * m

    s_seg, rate_seg = f("segment")
    s_cap, rate_cap = f("caption")
    seg_end = s_seg + video_s * rate_seg / n["segment"]
    first_seg = s_seg + shard_video_s("segment") * rate_seg
    cap_end = max(first_seg + s_cap + video_s * rate_cap / n["caption"],
                  seg_end + shard_video_s("caption") * rate_cap)
    gap = RECONCILE_WAIT_S if upper else RECONCILE_WAIT_S / 2
    return max(seg_end, cap_end) + gap


def project(step: str, mode: str, video_s: float = VIDEO_S, table: Any = None
            ) -> wh.Projection:
    """What ``step`` costs in ``mode`` (nominal and upper; see ``docs/scale-test.md``)."""
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

    r = kind_rates(mode, table)
    pools = pools_for(mode)
    for kind, (c, n) in pools.items():
        k = r[kind]
        work = video_s * k.cs_per_video_s
        s_up = k.startup_s * (1 + k.startup_unc)
        idle = WORKER_SCALEDOWN_S if kind == "segment" else IDLE_AFTER_JOB_S
        p.add(f"{kind}: {n} cold starts ({k.status} {k.startup_s:.0f} s)", c, n * k.startup_s,
              upper_s=n * s_up)
        p.add(f"{kind}: work {video_s:.0f} video-s x {k.cs_per_video_s:g} ({k.status} "
              f"+/-{k.uncertainty:.0%})", c, work, upper_s=work * (1 + k.uncertainty))
        p.add(f"{kind}: idle tail x{n} (scale-down / stop)", c, n * idle,
              upper_s=n * (WORKER_SCALEDOWN_S + CLIENT_SLACK_S))
        shard = shard_video_s(kind) * k.cs_per_video_s
        p.add(f"{kind}: 1 preemption (replacement start + lost shard)", c,
              k.startup_s + shard / 2, upper_s=s_up + shard * (1 + k.uncertainty))
        if kind == "caption":  # the orchestrator kill: captions idle, scale down, restart
            p.add(f"caption: reconciler gap x{n} (idle {WORKER_SCALEDOWN_S} s + restart)", c,
                  n * (WORKER_SCALEDOWN_S + k.startup_s),
                  upper_s=n * (WORKER_SCALEDOWN_S + s_up))
    orch = wh.Container(None, 0.25 * 3, 1.0 * 3)  # nonpreemptible: 3x CPU/memory
    p.add("orchestrator (non-preemptible CPU) over the wall", orch,
          run_wall_s(mode, video_s, table=table),
          upper_s=run_wall_s(mode, video_s, upper=True, table=table) + CLIENT_SLACK_S)
    p.peak_usd_per_s = sum(c.usd_per_s() * n for c, n in pools.values()) + orch.usd_per_s()
    return p


def projections(mode: str, table: Any = None) -> dict[str, wh.Projection]:
    return {s: project(s, mode, table=table) for s in STEPS}


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
            run_start: float | None = None) -> dict[str, Any]:
    """Per pool: containers, container-seconds, busy seconds, utilization, cold starts, idle
    tails and their cost, video-s per container-s, peak concurrency; plus pipelining.

    ``containers``: ``{container_id: {"start": t, "end": t}}`` from ``modal container list``
    samples; without it (local) a container's life is its first start to its last end.
    ``run_start``: when ``run`` launched the job. ``time_to_peak_s`` is from it to the moment
    the pool first ran ``max_containers`` containers (None: no run start, or never reached).
    ``preempted_recovered``: preempted starts whose shard a later attempt finished."""
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
            "video_s": round(video, 1),
            "video_s_per_container_s": round(video / cs, 4) if cs else None,
            "window_s": round(max(ends) - min(starts), 1) if starts else None,
            "video_s_per_wall_s": (round(video / (max(ends) - min(starts)), 3)
                                   if starts and max(ends) > min(starts) else None),
            "cold_start_s": _quant(cold), "spawn_to_start_s": _quant(latency.get(kind, [])),
            "idle_tail_s": _quant(idle), "idle_s": round(sum(idle), 1),
            "idle_usd": round(sum(idle) * rate, 4), "usd": round(cs * rate, 4),
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
    "utilization_min": 0.60,  # busy / container-s per pool, while it holds work
    "time_to_peak_max_s": 2 * 195 + 60,  # two upper cold starts + a poll
    "pipelining_first_caption_max_s": 195 + 120,  # caption cold start (upper) + scheduling
    "projection_within_upper": True,
    "projection_nominal_error_max": 0.50,  # |actual / nominal - 1|
    "idle_share_max": 0.15,
    "reconciler_revive_max_s": RECONCILE_WAIT_S,
    "status_p95_max_s": 10.0,
    "report_max_s": 30.0,
    "compose_max_s": 60.0,
}


def evaluate(r: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    """PASS / FAIL / n.a. per criterion from the ``run`` + ``report`` results."""
    out = []

    def add(name: str, value: Any, target: str, ok: bool | None) -> None:
        out.append({"criterion": name, "value": value, "target": target,
                    "verdict": "n.a." if ok is None else ("PASS" if ok else "FAIL")})

    m = r.get("measure") or {}
    pools = m.get("pools") or {}
    add("end state", r.get("state"), CRITERIA["end_state"], r.get("state") == "done"
        if r.get("state") else None)
    rec = r.get("records") or {}
    bad = rec.get("missing", 0) + rec.get("duplicates", 0) if rec else None
    add("lost/duplicate records", bad, "0", None if bad is None else bad == 0)
    for kind, p in pools.items():
        want = POOLS[kind]["max_containers"]
        add(f"{kind}: autoscaled to max_containers", p.get("peak_concurrent"), f">= {want}",
            p.get("peak_concurrent", 0) >= want if mode != "local" else None)
        t = p.get("time_to_peak_s")
        reached = p.get("peak_concurrent", 0) >= want
        # from the run start to max_containers; a pool that never got there fails, it is not
        # measured to a lower peak
        add(f"{kind}: time to peak", t, f"<= {CRITERIA['time_to_peak_max_s']} s from the run "
            "start", None if mode == "local" else
            (False if not reached else None if t is None
             else t <= CRITERIA["time_to_peak_max_s"]))
        u = p.get("utilization")
        add(f"{kind}: utilization (busy / container-s)", u, f">= {CRITERIA['utilization_min']}",
            None if u is None or mode == "local" else u >= CRITERIA["utilization_min"])
        n_pre, n_rec = p.get("preempted_starts", 0), p.get("preempted_recovered", 0)
        add(f"{kind}: preemption recovered", f"{n_rec}/{n_pre}", ">= 1 preempted start, each "
            "shard finished by a later attempt",
            (n_pre >= 1 and n_rec == n_pre) if f"preempt_{kind}" in (r.get("chaos") or {})
            else None)
    pl = m.get("pipelining") or {}
    v = pl.get("first_caption_after_first_segment_s")
    add("pipelining: first caption after first segment shard", v,
        f"<= {CRITERIA['pipelining_first_caption_max_s']} s and overlap > 0",
        None if v is None else (v <= CRITERIA["pipelining_first_caption_max_s"]
                                and pl.get("overlap_s", 0) > 0))
    pe = r.get("projection_error") or {}
    if mode != "local" and pe:
        add("spend within the upper projection", pe.get("actual_usd"),
            f"<= ${pe.get('upper_usd')}", pe.get("actual_usd", 0) <= pe.get("upper_usd", 0))
        e = pe.get("vs_nominal")
        lim = CRITERIA["projection_nominal_error_max"]
        add("spend vs nominal projection", e, f"|err| <= {lim}",
            None if e is None else abs(e) <= CRITERIA["projection_nominal_error_max"])
    s = m.get("idle_share")
    add("idle / scale-down share of pool cost", s, f"<= {CRITERIA['idle_share_max']}",
        None if s is None or mode == "local" else s <= CRITERIA["idle_share_max"])
    orch = r.get("orchestrator") or {}
    if orch:
        rv = orch.get("revive_s")
        add("orchestrator killed -> revived" + (" by the reconciler" if mode != "local" else
                                                " by flora resume"),
            rv, f"<= {CRITERIA['reconciler_revive_max_s']} s",
            None if rv is None and not orch.get("killed") else
            (rv is not None and rv <= CRITERIA["reconciler_revive_max_s"]
             and (mode == "local" or orch.get("revived_by") == "reconciler")))
    tm = r.get("timings") or {}
    st = (tm.get("status") or {}).get("p95")
    add("flora status p95", st, f"<= {CRITERIA['status_p95_max_s']} s",
        None if st is None else st <= CRITERIA["status_p95_max_s"])
    rp = tm.get("report_s")
    add("flora report", rp, f"<= {CRITERIA['report_max_s']} s",
        None if rp is None else rp <= CRITERIA["report_max_s"])
    cp = tm.get("compose_s")
    add("flora compose", cp, f"<= {CRITERIA['compose_max_s']} s",
        None if cp is None else cp <= CRITERIA["compose_max_s"])
    return out


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
    wall_limit = run_wall_s(mode, video, upper=True) + CLIENT_SLACK_S if mode != "local" \
        else 3600.0
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
                            resumes=0)
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
                    orch.update(killed=True, at_s=s.get("t"), call_id=res.get("call_id"))
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
    rep: dict[str, Any] = {}
    with contextlib.suppress(ValueError):
        rep = json.loads(rp.stdout)
    job = sdk.open_job(ctx.job_id, store=ctx.store_uri)
    manifest = job._deps().manifests.load()
    files = read_shard_files(job.store, ctx.job_id, job.config_hash, manifest)
    dur = {e.uid: e.duration_s for e in manifest.episodes}
    meas = measure(files, dur, run.get("containers") or None,
                   usd_per_s=container_rates(ctx.mode), run_start=run.get("started_at"))
    summary = job.summary() or {}
    stages = summary.get("stages", {})
    records = {"missing": sum(int(c.get("missing", 0)) for c in stages.values()),
               "duplicates": len(summary.get("conflicts", [])),
               "duplicates_resolved": summary.get("duplicates_resolved"),
               "by_stage": {k: {kk: c.get(kk) for kk in ("records", "expected", "missing",
                                                         "excused")}
                            for k, c in stages.items()}}
    proj = ctx.st.step("run")
    actual = float(run.get("tracked_usd") or 0.0)
    pe: dict[str, Any] = {}
    if proj.projected_usd:
        pe = {"actual_usd": round(actual, 4), "nominal_usd": proj.projected_usd,
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
           "launches": len(job.spec.get("launches") or [])}
    res["criteria"] = evaluate(res, ctx.mode)
    res.update(ok=rp.returncode in (0, 3) and bool(meas["pools"]), tracked_usd=0.0, apps=[])
    return res


RUNNERS: dict[str, Callable[[Ctx], dict[str, Any]]] = {
    "verify": run_verify, "index": run_index, "plan": run_plan, "deploy": run_deploy,
    "run": run_run, "report": run_report}


# ---------------------------------------------------------------------------------------------
# Report / plan printing


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
    if rep and rep.result.get("criteria"):
        lines += ["", "## Success criteria", "", "| criterion | value | target | verdict |",
                  "|---|---|---|---|"]
        for x in rep.result["criteria"]:
            lines.append(f"| {x['criterion']} | {x['value']} | {x['target']} | {x['verdict']} |")
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


def print_options(out=print) -> None:
    try:
        opts = planner_options()
    except Exception as e:  # noqa: BLE001 - informational only
        out(f"  flora plan options unavailable offline: {e!r}")
        return
    out("  flora plan options (offline, same rates; balanced rows are placeholders):")
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
    args = ap.parse_args(argv)
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
    rep = st.steps.get("report")
    if rep and rep.result.get("criteria"):
        for x in rep.result["criteria"]:
            print(f"  {x['verdict']:5s} {x['criterion']}: {x['value']} (target {x['target']})")
    print(f"wrote {report_path}; state {spath}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
