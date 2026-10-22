"""Rubric smoke: ``reward.rubric`` on real GPUs for the first time, in the ``wave_h.py`` style.

Usage::

    uv run python scripts/rubric_smoke.py                     # dry run: plan + costs, $0
    uv run python scripts/rubric_smoke.py --offline           # ... without reading the Volume
    uv run python scripts/rubric_smoke.py --yes-spend         # run (resumes: skips done steps)
    uv run python scripts/rubric_smoke.py --yes-spend --only lid
    uv run python scripts/rubric_smoke.py --report            # rewrite reports/rubric-smoke/
    uv run python scripts/rubric_smoke.py --finalize          # rates.json: rubric row measured

**What it runs.** ``reward.rubric`` alone (``preset: exact``: NVIDIA's settings, 9 samples at
temperature 1, TP=2 on ``H100!:2``, medoid selection; ``stages.rubric``) through the real
``flora index --local -> plan -> deploy -> run`` on three short golden ``human`` episodes
(:data:`JOBS`, staged on ``flora-golden:golden/v2/``, ``docs/golden-manifest-v2.md``). A config
carries one rubric (``pipeline.rubric``, hashed into the stage hash), so each episode task has
its own rubric, written for this smoke in our own words (:data:`RUBRICS`; never NVIDIA's
examples: ``tests/nvlink/test_license_content.py`` checks this file), its own config, job and
app. The jobs are independent: up to ``--max-parallel`` (default 3) run at once, each a step
with its own stop rule, reserving its budget against the global cap while it runs
(``_runbook.run_parallel``); ``check`` waits for all of them (:func:`step_deps`).
``--max-parallel 1`` runs them one after the other, as before. ``--only <task>`` runs one of
them (one cold start).

**What it measures** (per job, then pooled in step ``check``, which spends $0):

- engine build time of the TP=2 engine and the GPU count the container saw: the
  ``RubricWorker``'s ``rubric worker ready:`` line (``modal_app.workers_app.RUBRIC_READY``),
  read with ``modal app logs`` after the job;
- the 2-GPU startup (spawn to first attempt) and busy container-s per video-s, from ``flora
  report``'s per-container receipts (what ``--finalize`` writes into ``rates.json``);
- the parsed-sample rate (NVIDIA's ``parse_rubric`` accepted, out of 9) and medoid stability
  (:func:`medoid_stats`: leave-one-out stability of the chosen score vector, agreement and
  spread around it), from the job's audits (NVIDIA's per-sample scores; no answer text kept);
- record validity (:func:`record_problems`): provenance (preset, stage hash, engine, logic
  version, framework, GPU, source ranges), the certification fields (``certificate`` is
  ``None`` for ``exact``, ``embodiment``, ``cert_key`` equal to the key this checkout computes)
  and the rubric fields; and that ``flora certify`` **refuses** the records cleanly (no
  traceback, no certificate written: there is no rubric certification; :func:`certify_checks`).

**Preconditions** (:func:`preconditions`, all at $0, before any container): the episodes are
staged in the golden v2 manifest; each rubric validates (``config.schema`` and
``stages.rubric.RubricSpec``) inside a config that loads; the rubric engine's image lock is
current (:func:`image_lock_problems`); the pool is TP=2 and fits ``modal.max_gpus``
(:func:`tp_pool_problems`, ``config.schema.TP_GPUS``) and ``factory.plan_app`` plans it on
``H100!:2``. A failed precondition starts nothing.

**Spend.** The state file, attempts, heartbeat, billing cross-check and stop rules are
``wave_h.py``'s (``wave_h.State`` / ``Attempt`` / ``Heartbeat`` / ``check_before`` /
``check_after`` / ``gpu_deadline_s`` / ``refresh_billing``), with this runbook's caps. Each job
is priced (:func:`project_job`) on one TP=2 container (2 x H100 + 8 cores + 96 GiB): startup =
the Rewarder's measured startup (``rates.json`` ``reward.*``, ``docs/cold-start.md``) scaled to
a cold compile (``wave_h.cold_startup_s``: the rubric worker has no ``flora-cache`` mount), upper
x (1 + its uncertainty); work = video s x the placeholder rubric rate, upper x 2 (its +/-100%);
plus the idle until ``modal app stop`` and, upper only, ``wave_h.CLIENT_SLACK_S``. The upper
projection must fit the step cap and the global cap left before a step starts. The global cap is
the sum of the planned steps' upper projections rounded up to $0.10, or ``--global-cap`` (a cap
the user approved; never lower). At the end the apps and this run's two Volumes are removed
(``--keep`` keeps the Volumes), also after an error.

**State and results** live in ``reports/rubric-smoke/`` (git-ignored: it names private episodes):
``state.json``, ``runbook.md``, ``results.json`` and the exported records ``records/<task>.jsonl``.
``--finalize`` sets the ``reward.rubric`` row of ``planner/rates.json`` to ``measured`` (n = the
jobs measured) like ``wave_h_finalize.py`` and adds its own cold startup row; it never touches
the ``balanced`` rows.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _runbook as rb  # noqa: E402 - the step loop, decisions, stop by id (shared)
import h0_rehearsal as h0  # noqa: E402 - the same staging and config text as H0 / Wave H
import wave_h as wh  # noqa: E402 - spend accounting and stop rules, reused

OUT = ROOT / "reports" / "rubric-smoke"
STATE = OUT / "state.json"
REPORT = OUT / "runbook.md"
RESULTS = OUT / "results.json"
RECORDS = OUT / "records"
RATES = ROOT / "src" / "flora_deploy" / "planner" / "rates.json"
GOLDEN_ROOT = "golden/v2"  # docs/golden-manifest-v2.md (12 staged human episodes)
KIND = "reward.rubric"
PRESET = "exact"  # NVIDIA's rubric settings; the rubric pool is the same in every preset
EMBODIMENT = "human"
DETERMINISM = "nvidia"
TAG = "flora-rubric-smoke"
STORE_VOLUME = "flora-rubric-smoke"  # + "-<run_id>": this run's own (removed at the end)
MEDIA_VOLUME = "flora-rubric-smoke-media"
MAX_GPUS = 2  # one TP=2 container: the tightest cap that fits it
SAMPLES = 9  # NVIDIA's --num-samples (stages.rubric reads it from NVIDIA's parser)
CHECK = "check"
DURATION_HEADROOM = wh.DURATION_HEADROOM
IDLE_AFTER_JOB_S = wh.IDLE_AFTER_JOB_S
CLIENT_SLACK_S = wh.CLIENT_SLACK_S
LOGS_TIMEOUT_S = 90

# ---------------------------------------------------------------------------------------------
# The jobs: one short EgoDex episode per task, and a rubric for that task (our own words)

ROBOT = "a pair of bare human hands, filmed from the person's own point of view"
CAMERAS = ({"role": "main",
            "description": "a camera worn on the head, looking down at both hands and the "
                           "table in front of them"},)


def _grades(*pairs: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"label": label, "points": points} for label, points in pairs]


RUBRICS: dict[str, list[dict[str, Any]]] = {
    "lid": [
        {"action": "Bring a hand to the container and take hold of its lid",
         "grading": _grades(("the lid is never touched", 0), ("fingers brush the lid only", 1),
                            ("the lid is gripped", 2))},
        {"action": "Pull the lid free of the container opening",
         "grading": _grades(("the lid stays shut", 0), ("the lid is loosened or tilted", 1),
                            ("the lid comes all the way off", 2))},
        {"action": "Put the lid back so it sits level on the opening",
         "grading": _grades(("the lid is not returned", 0),
                            ("returned but crooked or loose", 1), ("seated evenly", 2))},
    ],
    "braid": [
        {"action": "Sort the loose strands into separate bundles",
         "grading": _grades(("the strands stay in one bunch", 0),
                            ("bundles are started but mixed", 1),
                            ("distinct bundles are held apart", 2))},
        {"action": "Weave the bundles by crossing an outer one over the middle, side after side",
         "grading": _grades(("no crossing happens", 0), ("one or two crossings", 1),
                            ("a regular run of crossings", 2))},
        {"action": "Take the weave apart again until the bundles hang free",
         "grading": _grades(("the weave is left as is", 0), ("partly undone", 1),
                            ("fully undone", 2))},
    ],
    "sandwich": [
        {"action": "Lay a first slice of bread flat on the work surface",
         "grading": _grades(("no slice is put down", 0), ("a slice is put down", 1))},
        {"action": "Add the filling onto that slice",
         "grading": _grades(("nothing is added", 0), ("the filling misses or slides off", 1),
                            ("the filling rests on the slice", 2))},
        {"action": "Close it with a second slice whose edges roughly match the first",
         "grading": _grades(("no second slice", 0), ("placed but badly offset", 1),
                            ("edges roughly line up", 2))},
    ],
}


@dataclass(frozen=True)
class Job:
    """One rubric job: its task slug (step name), golden episode and rubric."""

    task: str
    episode: str
    task_name: str

    @property
    def rubric(self) -> dict[str, Any]:
        return {"robot": ROBOT, "cameras": [dict(c) for c in CAMERAS],
                "task_name": self.task_name, "steps": RUBRICS[self.task]}

    @property
    def max_total(self) -> float:
        return float(sum(max(g["points"] for g in s["grading"]) for s in RUBRICS[self.task]))


JOBS: tuple[Job, ...] = (
    Job("lid", "human/egodex_add_remove_lid/episode_000014", "lid on and off"),
    Job("braid", "human/egodex_braid_unbraid/episode_000001", "braid and unbraid"),
    Job("sandwich", "human/egodex_make_sandwich/episode_000002", "sandwich assembly"),
)
JOBS_BY_TASK = {j.task: j for j in JOBS}
STEPS: tuple[str, ...] = (*JOBS_BY_TASK, CHECK)


def titles() -> dict[str, str]:
    out = {j.task: f"{j.task}: rubric job on {j.episode.split('/')[1]} (H100!:2)" for j in JOBS}
    out[CHECK] = "check: records, medoid, flora certify refusal ($0)"
    return out


# ---------------------------------------------------------------------------------------------
# Prices and projections (pure)


@dataclass(frozen=True)
class TPContainer(wh.Container):
    """A ``wave_h.Container`` with ``gpu_count`` GPUs (the TP=2 rubric pool)."""

    gpu_count: int = 1

    def usd_per_s(self, rates: Any = None) -> float:
        from flora_deploy.jobs.budget import pinned_rates

        r = rates or pinned_rates()
        gpu = r.gpu_per_s.get((self.gpu or "").rstrip("!"), 0.0) if self.gpu else 0.0
        return gpu * self.gpu_count + self.cpu * r.core_per_s + self.memory_gib * r.gib_per_s


def rubric_container() -> TPContainer:
    from flora_deploy.config.presets import get_preset

    p = get_preset(PRESET).pools[KIND]
    return TPContainer(p.gpu, float(p.cpu_cores), float(p.memory_gib), gpu_count=p.gpu_count)


def startup_s(table: Any = None) -> tuple[float, float, str]:
    """(nominal, upper, basis) startup of one fresh rubric container: the Rewarder's measured
    startup (``reward.*``) scaled to a cold compile (``wave_h.cold_startup_s``), or a measured
    cold ``reward.rubric`` row (after ``--finalize``) as is; upper x (1 + its uncertainty)."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    st = t.startup_for(PRESET, KIND)
    sec, basis = wh.cold_startup_s(st)
    return sec, sec * (1 + st.uncertainty), f"{basis}, rates.json {st.kind} {st.startup_s:g} s"


def project_job(job: Job, video_s: float, table: Any = None) -> wh.Projection:
    """One rubric job in one TP=2 container (module docstring, "Spend")."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    e = t.rate(PRESET, KIND, 1, 1)
    start, start_up, _ = startup_s(t)
    work = video_s * e.container_s_per_video_s
    c = rubric_container()
    p = wh.Projection(job.task)
    p.add(f"cold start, TP=2 engine build ({job.task})", c, start, upper_s=start_up)
    p.add(f"rubric {video_s:.1f} video s x {e.container_s_per_video_s:g} ({e.status}, "
          f"+/-{e.uncertainty:.0%})", c, work, upper_s=work * (1 + e.uncertainty))
    p.add("idle to `modal app stop` (+ client slack, upper)", c, IDLE_AFTER_JOB_S,
          upper_s=IDLE_AFTER_JOB_S + CLIENT_SLACK_S)
    p.peak_usd_per_s = c.usd_per_s()
    return p


def durations(manifest: Mapping[str, Any]) -> dict[str, float]:
    return wh.durations(manifest)


def offline_manifest(headroom: float = DURATION_HEADROOM) -> dict[str, Any]:
    """The golden selection's expected durations (``wave_h.offline_manifest``) for a dry run
    that reads no Volume; staging is then not checked."""
    return wh.offline_manifest(headroom)


def projections(manifest: Mapping[str, Any], table: Any = None) -> dict[str, wh.Projection]:
    dur = durations(manifest)
    out = {j.task: project_job(j, dur.get(j.episode, 0.0), table) for j in JOBS}
    out[CHECK] = wh.Projection(CHECK)
    return out


def caps_for(projs: Mapping[str, wh.Projection]) -> dict[str, float]:
    """Per step, its upper projection rounded up to 5 cents (``wave_h.honest_caps``)."""
    caps = wh.honest_caps(projs)
    caps.pop("global", None)
    return caps


def global_cap(projs: Mapping[str, wh.Projection], steps: Iterable[str],
               override: float | None = None) -> float:
    """The planned steps' upper projections, rounded up to $0.10; ``override`` (a cap the user
    approved) can raise it, never lower it."""
    up = round(sum(projs[s].usd_upper for s in steps), 6)
    derived = round(math.ceil(up * 10 - 1e-9) / 10, 2)
    return max(derived, override) if override is not None else derived


def step_deps(steps: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """What each step needs done first: the jobs nothing (each its own config, job, app and
    staged episode), ``check`` every job of the run (it pools their records)."""
    return {CHECK: tuple(s for s in steps if s != CHECK)}


def planned(only: str | None) -> tuple[str, ...]:
    return (only, CHECK) if only and only != CHECK else ((CHECK,) if only else STEPS)


# ---------------------------------------------------------------------------------------------
# Configs and preconditions ($0)


def store_volume(run_id: str) -> str:
    return f"{STORE_VOLUME}-{run_id}"


def media_volume(run_id: str) -> str:
    return f"{MEDIA_VOLUME}-{run_id}"


def job_id(run_id: str, job: Job) -> str:
    return f"rubric-smoke-{run_id}-{job.task}"


def config_text(job: Job, run_id: str, *, budget_usd: float, preset: str = PRESET) -> str:
    """``h0.config_text`` for a rubric-only job, with ``pipeline.rubric``, ``max_gpus`` 2 and
    (``preset`` other than exact) ``allow_uncertified`` (the certify check's twin config)."""
    import yaml

    text = h0.config_text(project=job_id(run_id, job), data_dir=job.task,
                          embodiment=EMBODIMENT, stages=(KIND,),
                          store_volume=store_volume(run_id), media_volume=media_volume(run_id),
                          run_id=run_id, budget_usd=max(budget_usd, 0.01), tag=TAG,
                          preset=preset, header="Rubric smoke; generated by "
                                                "scripts/rubric_smoke.py")
    head = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    d = yaml.safe_load(text)
    d["modal"]["max_gpus"] = MAX_GPUS
    d["pipeline"]["rubric"] = job.rubric
    return head + yaml.safe_dump(d, sort_keys=False, allow_unicode=True)


def write_config(work: Path, job: Job, run_id: str, *, budget_usd: float,
                 preset: str = PRESET) -> Path:
    suffix = "" if preset == PRESET else f"-{preset}"
    path = work / f"flora-{job.task}{suffix}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_text(job, run_id, budget_usd=budget_usd, preset=preset),
                    encoding="utf-8")
    return path


def image_lock_problems(image: str | None = None) -> list[str]:
    """The rubric engine's image lock (``modal_app/locks``) is current: present, its ``.in`` is
    what ``scripts/lock_images.py`` writes from ``images.PACKAGES``, every requirement is
    pinned with hashes, and the ``vllm`` RUNTIME pin is in it."""
    from flora_deploy.config.pins import RUNTIME
    from flora_deploy.config.presets import get_preset
    from flora_deploy.modal_app import images

    name = image or get_preset(PRESET).engines[KIND].image
    lock = images.lock_path(name)
    if not lock.is_file():
        return [f"{name}: no lock file {lock.name} (run scripts/lock_images.py)"]
    out = []
    src = [x for x in lock.with_suffix(".in").read_text().splitlines()
           if x and not x.startswith("#")] if lock.with_suffix(".in").is_file() else None
    if src != list(images.PACKAGES[name]):
        out.append(f"{name}: {lock.with_suffix('.in').name} does not match images.PACKAGES "
                   "(re-lock: scripts/lock_images.py)")
    reqs: dict[str, int] = {}  # requirement -> its --hash lines
    cur = None
    for line in lock.read_text().splitlines():
        s = line.strip()
        if s.startswith("--hash"):
            if cur is not None:
                reqs[cur] += 1
        elif s and not s.startswith(("#", "-")) and "==" in s:
            cur = s.split()[0].lower()
            reqs[cur] = 0
    unhashed = sorted(r for r, n in reqs.items() if n == 0)
    if not reqs or unhashed:
        out.append(f"{name}: {len(unhashed)} of {len(reqs)} requirement(s) without hashes "
                   f"{unhashed[:3]}")
    want = f"vllm=={RUNTIME.get('vllm')}"
    if name == "img_vllm" and want not in reqs:
        out.append(f"{name}: {want} (RUNTIME) is not in {lock.name}")
    return out


def tp_pool_problems(cfg: Any, plan: Any = None) -> list[str]:
    """The rubric pool is tensor-parallel (``TP_GPUS``), its containers fit ``max_gpus`` and the
    app plan puts ``RubricWorker`` on that many GPUs."""
    from flora_deploy.config.schema import TP_GPUS, per_device_gpu_problem

    pool = cfg.pools().get(KIND)
    if pool is None:
        return [f"the config runs no {KIND} pool"]
    out = []
    want = TP_GPUS.get(KIND, 1)
    if want != 2 or pool.gpu_count != want:
        out.append(f"{KIND}: gpu_count {pool.gpu_count}, TP_GPUS {want}; NVIDIA's rubric runs "
                   "TP=2")
    if p := per_device_gpu_problem(cfg.preset, KIND, pool.gpu, pool.gpu_count):
        out.append(p)
    need = sum(p.gpu_count * p.max_containers for p in cfg.pools().values())
    if need > cfg.modal.max_gpus:
        out.append(f"the pools can use {need} GPUs at once; modal.max_gpus is "
                   f"{cfg.modal.max_gpus}")
    if plan is not None:
        cls = plan.worker_classes.get(KIND)
        gpu = plan.object(cls).gpu if cls else None
        if cls != "RubricWorker" or gpu != f"{pool.gpu}:{want}":
            out.append(f"plan_app puts {KIND} on {cls} with gpu {gpu!r}, not RubricWorker on "
                       f"{pool.gpu}:{want}")
    return out


def rubric_problems(job: Job) -> list[str]:
    """The rubric validates as ``pipeline.rubric`` and as ``stages.rubric.RubricSpec``."""
    from pydantic import ValidationError

    from flora_deploy.config.schema import RubricSection
    from flora_deploy.stages.rubric import RubricSpec

    try:
        sec = RubricSection.model_validate(job.rubric)
        spec = RubricSpec.from_json(sec.model_dump(mode="json", exclude_none=True))
    except (ValidationError, ValueError) as e:
        return [f"{job.task}: rubric does not validate: {e}"]
    if spec.roles != ("main",):
        return [f"{job.task}: rubric cameras {spec.roles}, a human episode has one main view"]
    return []


def preconditions(manifest: Mapping[str, Any], jobs: Sequence[Job], work: Path, run_id: str, *,
                  lock_problems: Callable[[], list[str]] = image_lock_problems
                  ) -> dict[str, list[str]]:
    """Everything that must hold before any container (module docstring); ``{}`` when it all
    does. Keys: ``staged``, ``rubric``, ``image``, ``pool``. Offline: staging is not checked."""
    from flora_deploy.config.schema import ConfigError, load_config
    from flora_deploy.modal_app.factory import DeployError, plan_app

    out: dict[str, list[str]] = {}
    if not manifest.get("offline"):
        have = durations(manifest)
        miss = [j.episode for j in jobs if j.episode not in have]
        if miss:
            out["staged"] = [f"not staged in {h0.GOLDEN_VOLUME}:{GOLDEN_ROOT}: {', '.join(miss)}"]
    rub = [p for j in jobs for p in rubric_problems(j)]
    pool: list[str] = []
    for j in jobs:
        try:
            cfg = load_config(write_config(work, j, run_id, budget_usd=1.0))
            plan = plan_app(cfg, determinism=DETERMINISM)
        except (ConfigError, DeployError, ValueError) as e:
            rub.append(f"{j.task}: the config does not load or plan: {e}")
            continue
        pool += tp_pool_problems(cfg, plan)
    if rub:
        out["rubric"] = rub
    if pool:
        out["pool"] = sorted(set(pool))
    if lk := lock_problems():
        out["image"] = lk
    return out


# ---------------------------------------------------------------------------------------------
# Measurements (pure)


def medoid(vectors: Sequence[Sequence[float]]) -> int:
    """The vector with the smallest summed L1 distance to all, ties to the lowest index (the
    usual medoid; ``stages.rubric`` uses NVIDIA's own ``core.medoid`` for the record)."""
    if not vectors:
        raise ValueError("no vectors")
    cost = [sum(sum(abs(a - b) for a, b in zip(v, w, strict=True)) for w in vectors)
            for v in vectors]
    return min(range(len(vectors)), key=lambda i: (cost[i], i))


def medoid_stats(samples: Sequence[Mapping[str, Any]], max_total: float) -> dict[str, Any]:
    """From NVIDIA's per-sample ``{scores, parsed}`` of one episode: parsed count, the medoid
    of the parsed samples, its leave-one-out stability (share of parsed samples whose removal
    leaves the chosen score vector unchanged), agreement (share equal to it) and spread (mean
    L1 distance to it over ``max_total``)."""
    parsed = [list(map(float, s["scores"])) for s in samples
              if s.get("parsed") and s.get("scores") is not None]
    out: dict[str, Any] = {"samples": len(samples), "parsed": len(parsed)}
    if not parsed:
        return {**out, "medoid": None, "loo_stability": None, "agreement": None,
                "spread": None}
    m = parsed[medoid(parsed)]
    stable = 0
    for i in range(len(parsed)):
        rest = parsed[:i] + parsed[i + 1:]
        stable += (not rest) or rest[medoid(rest)] == m
    dist = [sum(abs(a - b) for a, b in zip(v, m, strict=True)) for v in parsed]
    return {**out, "medoid": m, "loo_stability": round(stable / len(parsed), 4),
            "agreement": round(sum(v == m for v in parsed) / len(parsed), 4),
            "spread": round(sum(dist) / len(dist) / max_total, 4) if max_total else None}


_HEX64 = re.compile(r"^[0-9a-f]{64}\Z")


def record_problems(rec: Any, job: Job, *, stage_hash: str | None = None,
                    cert_key: Mapping[str, Any] | None = None) -> list[str]:
    """What is wrong with one rubric ``StageRecord`` (module docstring); ``[]``: valid."""
    from flora_deploy import LOGIC_VERSION
    from flora_deploy.stages.rubric import ENGINE_LABEL

    p, r, bad = rec.provenance, rec.rubric, []
    if rec.stage != KIND:
        bad.append(f"stage {rec.stage!r}")
    if p.preset != PRESET:
        bad.append(f"provenance.preset {p.preset!r}")
    if not _HEX64.match(str(p.config_hash or "")):
        bad.append("provenance.config_hash is not a 64-hex stage hash")
    elif stage_hash and p.config_hash != stage_hash:
        bad.append("provenance.config_hash is not the deployed reward.rubric stage hash")
    if p.engine != ENGINE_LABEL:
        bad.append(f"provenance.engine {p.engine!r}")
    if p.logic_version != LOGIC_VERSION:
        bad.append(f"provenance.logic_version {p.logic_version!r}")
    if not str(p.framework or "").startswith("flora-deploy "):
        bad.append(f"provenance.framework {p.framework!r}")
    if not p.models.get("rewarder"):
        bad.append("provenance.models has no rewarder pin")
    if "H100" not in str(p.gpu or ""):
        bad.append(f"provenance.gpu {p.gpu!r}, not an H100")
    if p.certificate is not None:  # exact: the reference by construction
        bad.append(f"provenance.certificate {p.certificate!r}; exact records carry none")
    if p.embodiment != EMBODIMENT:
        bad.append(f"provenance.embodiment {p.embodiment!r}")
    if not isinstance(p.cert_key, Mapping) or p.cert_key.get("kind") != KIND \
            or p.cert_key.get("preset") != PRESET:
        bad.append("provenance.cert_key is missing or not a reward.rubric exact key")
    elif cert_key is not None and dict(p.cert_key) != dict(cert_key):
        bad.append("provenance.cert_key differs from the key this checkout computes")
    src = p.source or {}
    if set(src) != {"main"} or not all(
            v.get("uri") and len(v.get("range_s") or ()) == 2 and v["range_s"][1] > 0
            for v in src.values()):
        bad.append("provenance.source lacks the main camera's uri and read range")
    if r is None:
        return [*bad, "no rubric reward"]
    steps = RUBRICS[job.task]
    if r.samples != SAMPLES:
        bad.append(f"rubric.samples {r.samples}, not {SAMPLES}")
    if len(r.scores) != len(steps):
        bad.append(f"{len(r.scores)} scores for {len(steps)} rubric steps")
    else:
        top = [max(g["points"] for g in s["grading"]) for s in steps]
        if any(not 0 <= s <= t for s, t in zip(r.scores, top, strict=True)):
            bad.append("a score is outside [0, max] of its step")
    if r.total != round(sum(r.scores), 4):
        bad.append("rubric.total is not round(sum(scores), 4)")
    if not 0 <= r.medoid_index < SAMPLES:
        bad.append(f"rubric.medoid_index {r.medoid_index}")
    if r.parsed_samples is None or not 1 <= r.parsed_samples <= SAMPLES:
        bad.append(f"rubric.parsed_samples {r.parsed_samples}")
    return bad


def parse_ready(text: str) -> dict[str, Any] | None:
    """The last ``rubric worker ready:`` line's JSON in ``modal app logs`` output."""
    from flora_deploy.modal_app.workers_app import RUBRIC_READY

    found = None
    for line in text.splitlines():
        i = line.find(RUBRIC_READY)
        if i >= 0:
            with contextlib.suppress(ValueError):
                found = json.loads(line[i + len(RUBRIC_READY):].strip())
    return found


def job_metrics(r: Mapping[str, Any], c: TPContainer | None = None) -> dict[str, Any]:
    """Startup, busy container-s per video-s and spend of one finished job's state entry
    (``wave_h.job_record`` fields), with the TP=2 price check of each container row."""
    c = c or rubric_container()
    rows = [x for x in r.get("containers") or () if x.get("kind") == KIND]
    video = float(r.get("video_s") or 0.0)
    busy = sum(float(x.get("busy_s") or 0.0) for x in rows)
    secs = sum(float(x.get("seconds") or 0.0) for x in rows)
    starts = [float(x["startup_s"]) for x in rows if x.get("startup_s") is not None]
    usd = sum(float(x.get("usd") or 0.0) for x in rows)
    return {"containers": len(rows), "video_s": video, "startup_s": starts,
            "busy_s": round(busy, 1), "container_s": round(secs, 1),
            "busy_container_s_per_video_s": round(busy / video, 3) if video else None,
            "container_s_per_video_s_all_in": round(secs / video, 3) if video else None,
            "usd_report": round(usd, 4), "usd_tp2": round(secs * c.usd_per_s(), 4),
            "gpu_seen": sorted({str(x.get("gpu_seen")) for x in rows if x.get("gpu_seen")}),
            "gpu_count_labels": sorted({x.get("gpu_count") for x in rows
                                        if x.get("gpu_count") is not None})}


# ---------------------------------------------------------------------------------------------
# flora certify must refuse rubric records ($0, local)


def run_cli(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=600)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def certify_checks(records: Path, exact_cfg: Path, twin_cfg: Path, out_dir: Path, *,
                   cli: Callable[[list[str]], tuple[int, str]] = run_cli) -> dict[str, Any]:
    """``flora certify`` on the rubric records: with the job's (exact) config, with a balanced
    twin and ``--stage reward.rubric``, and with the twin and no ``--stage``. Each must exit
    non-zero with a message, no traceback and no certificate written."""
    fl = h0.flora_bin()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = [fl, "certify", "--exact", str(records), "--balanced", str(records),
            "--out-dir", str(out_dir)]
    cases = {"exact config": [*base, "--config", str(exact_cfg)],
             "twin, --stage reward.rubric": [*base, "--config", str(twin_cfg), "--stage", KIND],
             "twin, every stage": [*base, "--config", str(twin_cfg)]}
    out: dict[str, Any] = {}
    for name, args in cases.items():
        before = set(out_dir.rglob("*"))
        rc, text = cli(args)
        wrote = sorted(str(p.name) for p in set(out_dir.rglob("*")) - before)
        msg = " ".join(text.split())[-300:]
        out[name] = {"rc": rc, "message": msg, "written": wrote,
                     "clean": rc != 0 and bool(msg) and "Traceback" not in text and not wrote}
    out["ok"] = all(v["clean"] for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------------------------------------
# Step runners (GPU steps only with --yes-spend)


@dataclass
class Ctx:
    st: wh.State
    manifest: dict[str, Any]
    work: Path
    caps: dict[str, float]
    global_cap: float
    state_path: Path = STATE
    step: str = ""
    log: Callable[[str], None] = print
    apps: list[str] = field(default_factory=list)
    # the run's steps and projections: what the global cap keeps for the later steps
    steps: tuple[str, ...] = ()
    projections: Mapping[str, wh.Projection] | None = None

    def sh(self, args: list[str], *, timeout: float = 1800, check: bool = True):
        return h0.sh(args, cwd=self.work, timeout=timeout, check=check,
                     log=self.work / "rubric-smoke.log")

    def reserve(self, step: str) -> dict[str, float]:
        return rb.reserve_for(self.st, self.steps, step, self.projections, self.caps)

    def note_app(self, name: str) -> None:
        """Record an app in the running attempt *before* it can spend (wave_h H3)."""
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None:
            if name not in cur.apps:
                cur.apps.append(name)
            cur.stopped_at = None
            wh.save_state(self.st, self.state_path)

    def confirm_stop(self) -> None:
        """The running attempt's (deployed) app was stopped: until billing covers it, a
        failure is charged its time plus the slack (``_runbook.Attempt.bounded_usd``)."""
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None:
            cur.stopped_at = time.time()
            wh.save_state(self.st, self.state_path)


def store_uri(run_id: str) -> str:
    return f"modal-volume://{store_volume(run_id)}/{run_id}"


def golden_manifest(root: str = GOLDEN_ROOT) -> dict[str, Any]:
    import modal

    vol = modal.Volume.from_name(h0.GOLDEN_VOLUME, version=2)
    return json.loads(b"".join(vol.read_file(f"{root}/manifest.json")))


def app_logs(app: str, *, timeout: float = LOGS_TIMEOUT_S) -> str:
    try:
        p = subprocess.run(["modal", "app", "logs", app, "--tail", "3000"], capture_output=True,
                           text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:  # streamed until the timeout: keep what came
        out = e.stdout or b""
        return out.decode(errors="replace") if isinstance(out, bytes) else str(out)
    except OSError:
        return ""


def export_rubric(job_id_: str, run_id: str, dest: Path) -> tuple[list[Any], list[dict]]:
    """The job's rubric records (written to ``dest`` as JSONL) and its rubric audit lines."""
    from flora_deploy import sdk
    from flora_deploy.jobs.store import JobPaths

    job = sdk.open_job(job_id_, store=store_uri(run_id))
    recs = list(job.results(KIND))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(r.to_json() + "\n" for r in recs), encoding="utf-8")
    audits = []
    for f in job.store.list(f"{JobPaths(job_id_).root}/"):
        if "/audit/" not in f:
            continue
        for line in job.store.get_bytes(f).decode().splitlines():
            with contextlib.suppress(ValueError):
                a = json.loads(line)
                if a.get("stage") == KIND and isinstance(a.get("nv"), dict):
                    audits.append(a)
    return recs, audits


def episode_metrics(job: Job, recs: Sequence[Any], audits: Sequence[Mapping[str, Any]], *,
                    stage_hash: str | None, cert_key: Mapping[str, Any] | None
                    ) -> dict[str, Any]:
    """Per record: validity, parsed samples and medoid stats from its audit (the audit whose
    medoid matches the record's, else the last one for that uid)."""
    out = {}
    for rec in recs:
        mine = [a for a in audits if a.get("uid") == rec.uid]
        pick = next((a for a in reversed(mine) if rec.rubric is not None and
                     a["nv"].get("medoid_sample") == rec.rubric.medoid_index), None)
        pick = pick or (mine[-1] if mine else None)
        samples = [{"scores": s.get("scores"), "parsed": s.get("parsed")}
                   for s in ((pick or {}).get("nv") or {}).get("samples") or ()]
        stats = medoid_stats(samples, job.max_total) if samples else {}
        if samples and rec.rubric is not None and stats.get("medoid") is not None:
            stats["medoid_matches_record"] = stats["medoid"] == list(rec.rubric.scores)
        out[rec.uid] = {"problems": record_problems(rec, job, stage_hash=stage_hash,
                                                    cert_key=cert_key),
                        "certified": rec.certified, "audit": pick is not None,
                        "finish_reasons": [x.get("finish_reasons") for x in
                                           (pick or {}).get("requests") or ()], **stats}
    return out


def run_job(ctx: Ctx, job: Job) -> dict[str, Any]:
    """Stage one episode, ``flora index --local -> plan -> deploy -> run`` with the in-step
    deadline, ``modal app stop``, then the report, the ready line and the records."""
    from flora_deploy.config.hashing import kind_modes
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import plan_app
    from flora_deploy.validate.certkey import key_document

    run = ctx.st.run_id
    c = rubric_container()
    fl = h0.flora_bin()
    budget = max(0.01, wh.step_budget(ctx.st, job.task, caps=ctx.caps,
                                      global_cap=ctx.global_cap, reserve=ctx.reserve(job.task)))
    h0.fetch_dataset(ctx.manifest, ctx.work, episodes={job.task: (job.episode,)},
                     poison_job=None)
    h0.upload_dir(media_volume(run), ctx.work / job.task, f"{run}/{job.task}")
    cfg_path = write_config(ctx.work, job, run, budget_usd=budget)
    jid = job_id(run, job)
    local = ctx.work / f"store-{job.task}"
    summary = ctx.work / f"summary-{job.task}.json"
    ctx.sh([fl, "index", "--local", "--config", str(cfg_path), "--job", jid, "--store",
            f"file://{local}", "--out", str(summary)])
    h0.mirror_index(local, jid, run, store_volume(run))
    ctx.sh([fl, "plan", "--config", str(cfg_path), "--index", str(summary)])
    cfg = load_config(cfg_path)
    plan = plan_app(cfg, determinism=DETERMINISM)
    deadline = wh.gpu_deadline_s(ctx.st, job.task, c, caps=ctx.caps, global_cap=ctx.global_cap,
                                 reserve=ctx.reserve(job.task))
    out: dict[str, Any] = {"app": plan.app_name, "job_id": jid, "cfg": str(cfg_path),
                           "deadline_s": round(deadline), "budget_usd": round(budget, 4),
                           "stage_hash": plan.stage_hashes[KIND], "episodes": [job.episode],
                           "records": str(RECORDS / f"{job.task}.jsonl")}
    if deadline < 120:
        return {**out, "ok": False, "state": "skipped", "tracked_usd": 0.0, "apps": [],
                "error": f"budget left allows {deadline:.0f} s"}
    ctx.note_app(plan.app_name)
    ctx.sh([fl, "deploy", "--config", str(cfg_path), "--determinism", DETERMINISM],
           timeout=1800)
    try:
        args = [fl, "run", "--config", str(cfg_path), "--determinism", DETERMINISM, "--job", jid,
                "--poll", "15", "--budget", f"{budget:.2f}"]
        try:
            out["rc"] = ctx.sh(args, timeout=deadline, check=False).returncode
        except subprocess.TimeoutExpired:
            out["rc"] = "deadline"
            ctx.sh([fl, "cancel", jid, "--yes", "--config", str(cfg_path)], check=False,
                   timeout=120)
    finally:
        p = ctx.sh(["modal", "app", "stop", "--yes", plan.app_name], check=False, timeout=120)
        out["stopped_at"] = time.time()
        if getattr(p, "returncode", 1) == 0:  # a deployed app: stopped by its name
            ctx.confirm_stop()
    rep = ctx.sh([fl, "report", jid, "--config", str(cfg_path), "--json"], check=False,
                 timeout=900)
    try:
        out.update(wh.job_record(json.loads(rep.stdout), stopped_at=out["stopped_at"]))
    except ValueError:
        out["report_error"] = (rep.stdout + rep.stderr)[-400:]
    m = job_metrics(out, c) if out.get("containers") else {}
    # tracked: the report's spend, or the TP=2 price of its container-seconds when larger
    spend = max(float(out.get("spend") or 0.0), float(m.get("usd_tp2") or 0.0))
    if out.get("spend") is None:  # unknown: charged what it was allowed (wave_h H3)
        out["report_error"] = out.get("report_error") or "no spend in flora report"
        spend = budget
    out["metrics"] = m
    out["ready"] = parse_ready(app_logs(plan.app_name))
    try:
        recs, audits = export_rubric(jid, run, Path(out["records"]))
        key = key_document(PRESET, KIND, cfg.pools()[KIND].gpu, modes=kind_modes(cfg, KIND))
        out["per_episode"] = episode_metrics(job, recs, audits, stage_hash=out["stage_hash"],
                                             cert_key=key)
    except Exception as e:  # noqa: BLE001 - recorded; the job's spend still counts
        out["records_error"] = repr(e)[:400]
    ok = out.get("state") == "done" and bool(out.get("per_episode"))
    return {**out, "ok": ok, "tracked_usd": spend, "apps": [plan.app_name],
            "report_error": bool(out.get("report_error"))}


def summarize(st: wh.State) -> dict[str, Any]:
    """The pooled measurements over the finished jobs (step ``check``; ``results.json``)."""
    per, jobs = {}, {}
    for j in JOBS:
        s = st.steps.get(j.task)
        if not s or s.status != "done":
            continue
        r = s.result
        jobs[j.task] = {"app": r.get("app"), "job_id": r.get("job_id"),
                        "state": r.get("state"), "ready": r.get("ready"),
                        "metrics": r.get("metrics"), "spend": s.spent}
        per.update({f"{j.task}:{k}": v for k, v in (r.get("per_episode") or {}).items()})
    parsed = sum(int(v.get("parsed") or 0) for v in per.values())
    samples = sum(int(v.get("samples") or 0) for v in per.values())
    loo = [v["loo_stability"] for v in per.values() if v.get("loo_stability") is not None]
    builds = [float(x["ready"]["engine_build_s"]) for x in jobs.values()
              if x.get("ready") and x["ready"].get("engine_build_s") is not None]
    starts = [s for x in jobs.values() for s in (x.get("metrics") or {}).get("startup_s") or ()]
    busy = sum(float((x.get("metrics") or {}).get("busy_s") or 0.0) for x in jobs.values())
    video = sum(float((x.get("metrics") or {}).get("video_s") or 0.0) for x in jobs.values())
    return {"jobs": jobs, "episodes": per,
            "engine_build_s": builds, "startup_s": starts,
            "gpu_count_seen": sorted({x["ready"].get("gpu_count") for x in jobs.values()
                                      if x.get("ready")} - {None}),
            "busy_container_s_per_video_s": round(busy / video, 3) if video else None,
            "parsed_rate": round(parsed / samples, 4) if samples else None,
            "parsed": parsed, "samples": samples,
            "loo_stability_mean": round(sum(loo) / len(loo), 4) if loo else None,
            "records_valid": bool(per) and all(not v["problems"] for v in per.values()),
            "problems": {k: v["problems"] for k, v in per.items() if v["problems"]}}


def run_check(ctx: Ctx, *, cli: Callable[[list[str]], tuple[int, str]] = run_cli
              ) -> dict[str, Any]:
    """Step ``check`` ($0): pool the jobs and run :func:`certify_checks` on their records."""
    done = [j for j in JOBS if (ctx.st.steps.get(j.task) or wh.StepState()).status == "done"]
    if not done:
        return {"ok": False, "error": "no rubric job finished", "tracked_usd": 0.0, "apps": []}
    out = summarize(ctx.st)
    j = done[0]
    recs = Path(ctx.st.step(j.task).result["records"])
    exact_cfg = Path(ctx.st.step(j.task).result["cfg"])
    # a balanced twin (h0.config_text sets allow_uncertified) reaches certify's stage check
    twin = write_config(ctx.work, j, ctx.st.run_id, budget_usd=1.0, preset="balanced")
    out["certify"] = certify_checks(recs, exact_cfg, twin, ctx.work / "certify-out", cli=cli)
    out["ok"] = out["records_valid"] and out["certify"]["ok"]
    return {**out, "tracked_usd": 0.0, "apps": []}


# ---------------------------------------------------------------------------------------------
# Finalize: the rubric rate row of planner/rates.json


def finalized_rates(data: Mapping[str, Any], st: wh.State, *, date: str, source: str
                    ) -> dict[str, Any]:
    """``rates.json`` with the ``reward.rubric`` row measured from the finished jobs (like
    ``wave_h_finalize.updated_rates``) and a cold ``reward.rubric`` startup row ahead of
    ``reward.*``. Nothing else changes; refused (SystemExit) without a measurement."""
    import wave_h_finalize as whf

    gpu = whf.strip_gpu(rubric_container().gpu)
    m = whf.Measurement("*", KIND, gpu)
    for j in JOBS:
        s = st.steps.get(j.task)
        if not s or s.status != "done":
            continue
        r = s.result
        rows = [c for c in r.get("containers") or () if c.get("kind") == KIND]
        lacks = [f for c in rows for f in whf.RECEIPT_FIELDS if c.get(f) is None]
        if not rows or lacks or not r.get("video_s"):
            m.problems.append(f"{j.task}: no complete {KIND} container receipt or video s")
            continue
        wrong = [c for c in rows if not whf.same_gpu(whf.ran_on(c), gpu)]
        if wrong:
            m.problems.append(f"{j.task}: ran on {whf.ran_on(wrong[0])!r}, not {gpu}")
            continue
        m.samples.append((j.task, sum(float(c["busy_s"]) for c in rows), float(r["video_s"])))
        m.startups += [float(c["startup_s"]) for c in rows]
        m.container_s += sum(float(c["seconds"]) for c in rows)
    if m.refusals:  # wave_h_finalize's rules: >= 1 job, >= 30 video s, >= 1 cold start
        raise SystemExit("refused: " + "; ".join(m.refusals))
    new = json.loads(json.dumps(data))
    rows = [r for r in new["rates"] if r["kind"] == KIND]
    if len(rows) != 1:
        raise SystemExit(f"rates.json: expected one {KIND} row, found {len(rows)}")
    band = (f"{m.rate} from one job: no range, so the error band is "
            f"±{whf.SINGLE_SAMPLE_FLOOR:.0%}" if m.n == 1
            else f"{min(m.rates):.3f}-{max(m.rates):.3f} per job, {m.rate} over all")
    rows[0].update(
        container_s_per_video_s=m.rate, uncertainty=m.uncertainty, status="measured", n=m.n,
        source=source, date=date, image=whf.engine_image(PRESET, KIND),
        provenance=(f"Rubric smoke run {st.run_id}: reward.rubric deployed with flora "
                    f"deploy/run (NVIDIA's settings: 9 samples, temperature 1, TP=2) on "
                    f"{gpu}:2, n={m.n} job(s) ({m.video_s:.1f} video s), busy container-s of "
                    f"one TP=2 container per video-s {band}; excludes @enter (startup row) "
                    f"and idle ({source})"))
    k = len(m.startups)
    srow = {"preset": "*", "kind": KIND, "startup_s": m.startup_s,
            "uncertainty": m.startup_uncertainty, "status": "measured", "compile_cache": "cold",
            "n": k, "provenance": (f"Rubric smoke run {st.run_id}: spawn to first attempt of "
                                   f"n={k} fresh {gpu}:2 container(s), no flora-cache (cold "
                                   f"compile) ({source})"),
            "date": date, "source": source}
    existing = [r for r in new["startup"] if r["kind"] == KIND]
    if existing:
        existing[0].clear()
        existing[0].update(srow)
    else:
        at = next((i for i, r in enumerate(new["startup"]) if r["kind"] == "reward.*"),
                  len(new["startup"]))
        new["startup"].insert(at, srow)
    new["updated"] = max(str(new.get("updated") or ""), date)
    return new


def finalize(st: wh.State, path: Path = RATES, *, date: str | None = None,
             source: str = "reports/rubric-smoke/results.json") -> dict[str, Any]:
    import wave_h_finalize as whf

    data = json.loads(path.read_text())
    new = finalized_rates(data, st, date=date or time.strftime("%Y-%m-%d"),
                          source=f"{source} (rubric smoke run {st.run_id})")
    from flora_deploy.planner.rates import RateTable

    RateTable.from_dict(new, source=str(path))  # the shipped loader accepts it
    path.write_text(whf.dump_json(new))
    return new


# ---------------------------------------------------------------------------------------------
# Cleanup and report


def cleanup(apps: Iterable[str], run_id: str, *, keep: bool,
            run: Callable[..., Any] = subprocess.run) -> list[str]:
    """``modal app stop`` every app this run noted, then delete its two Volumes."""
    notes = []
    for name in sorted(set(apps)):
        p = run(["modal", "app", "stop", "--yes", name], capture_output=True, text=True)
        notes.append(f"modal app stop {name}: rc={p.returncode}")
    if not keep:
        for vol in (store_volume(run_id), media_volume(run_id)):
            p = run(["modal", "volume", "delete", vol, "--yes"], capture_output=True, text=True)
            notes.append(f"modal volume delete {vol}: rc={p.returncode}")
    return notes


def render_report(st: wh.State, projs: Mapping[str, wh.Projection], caps: Mapping[str, float],
                  gcap: float) -> str:
    t = titles()
    lines = ["# Rubric smoke runbook summary", "",
             "Generated by `scripts/rubric_smoke.py`. Numbers only (no episode text).", "",
             f"Run `{st.run_id}`. Spent so far **${st.spent:.2f}** of the ${gcap:.2f} cap.", "",
             "| step | status | cap | nominal (upper) | attempts | tracked | billed |",
             "|---|---|---|---|---|---|---|"]
    for name in STEPS:
        s = st.steps.get(name, wh.StepState())
        p = projs[name]
        billed = f"${s.billed_usd:.2f}" if s.billed_usd is not None else "-"
        lines.append(f"| {t[name]} | {s.status} | ${caps[name]:.2f} | ${p.usd:.2f} "
                     f"(${p.usd_upper:.2f}) | {len(s.attempts)} | ${s.tracked_usd:.2f} | "
                     f"{billed} |")
    res = (st.steps.get(CHECK) or wh.StepState()).result
    if res:
        lines += ["", "## Measurements", "", "```json",
                  json.dumps(wh._numbers_only({k: v for k, v in res.items() if k != "apps"}),
                             indent=1, default=str)[:8000], "```"]
    return "\n".join(lines) + "\n"


def print_plan(st: wh.State, projs: Mapping[str, wh.Projection], caps: Mapping[str, float],
               gcap: float, steps: Sequence[str], manifest: Mapping[str, Any],
               out: Callable[[str], None] = print,
               max_parallel: int = rb.DEFAULT_MAX_PARALLEL) -> None:
    t = titles()
    c = rubric_container()
    start, start_up, basis = startup_s()
    src = (f"expected durations from the golden selection (offline, +{DURATION_HEADROOM:.0%})"
           if manifest.get("offline") else f"{h0.GOLDEN_VOLUME}:{GOLDEN_ROOT} manifest")
    out(f"Rubric smoke run {st.run_id}: {' -> '.join(steps)}; spent ${st.spent:.2f} of "
        f"${gcap:.2f}; durations: {src}")
    out(f"  pool: {KIND} on {c.gpu}:{c.gpu_count}, {c.cpu:g} cores, {c.memory_gib:g} GiB = "
        f"${c.usd_per_h():.2f}/h; startup {start:.0f} s (upper {start_up:.0f} s; {basis})")
    for name in steps:
        p, s = projs[name], st.steps.get(name, wh.StepState())
        d = rb.decide(st, steps, name, projs, caps=caps, global_cap=gcap)  # what a run decides
        rule = {"run": "ok", "done": "done"}.get(d.action) or f"{d.action.upper()}: {d.message}"
        out(f"  {t[name]:58s} [{s.status}] cap ${caps[name]:.2f}  nominal ${p.usd:.2f}  "
            f"upper ${p.usd_upper:.2f}  rule: {rule}")
        for i in p.items:
            out(f"      {i['what']:58s} {i['seconds']:6.0f} s / {i['upper_seconds']:6.0f} s x "
                f"${i['usd_per_h']:.2f}/h = ${i['usd']:.3f} (upper ${i['usd_upper']:.3f})")
    nominal = sum(projs[s].usd for s in steps)
    upper = sum(projs[s].usd_upper for s in steps)
    out(f"  total: nominal ${nominal:.2f}, upper ${upper:.2f}; global cap ${gcap:.2f}")
    wall = {n: (0.0, 0.0) if (st.steps.get(n) or wh.StepState()).status == "done"
            else (projs[n].wall_s(), projs[n].wall_s(upper=True)) for n in steps}
    out(rb.wall_line(steps, step_deps(steps), wall, max_parallel))
    out("  note: max_tokens 20480 x 9 samples can run longer than the placeholder rate's upper; "
        "the in-step deadline (budget left / $/s, minus the client slack) cancels the job")


def load_manifest(offline: bool) -> dict[str, Any]:
    return offline_manifest() if offline else golden_manifest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print the plan; spend nothing "
                                                          "(the default)")
    g.add_argument("--yes-spend", action="store_true", help="run (the GPU steps spend money)")
    g.add_argument("--report", action="store_true", help="rewrite reports/rubric-smoke/")
    g.add_argument("--finalize", action="store_true",
                   help="set the reward.rubric row of planner/rates.json to measured")
    ap.add_argument("--only", choices=STEPS, help="run just this step (and check)")
    ap.add_argument("--global-cap", type=float, default=None, metavar="USD",
                    help="raise the global cap to this (user-approved; saved in the state)")
    ap.add_argument("--offline", action="store_true",
                    help="dry run: price expected durations; do not read the Volume")
    ap.add_argument("--keep", action="store_true", help="keep this run's two Volumes")
    ap.add_argument("--state", type=Path, default=STATE)
    ap.add_argument("--work", type=Path, default=OUT / "work")
    ap.add_argument("--max-parallel", type=int, default=rb.DEFAULT_MAX_PARALLEL, metavar="N",
                    help=f"run up to N rubric jobs at once (default {rb.DEFAULT_MAX_PARALLEL}; "
                         "check waits for them); each reserves its budget against the global "
                         "cap while it runs. 1: one after the other, as before")
    args = ap.parse_args(argv)
    if args.max_parallel < 1:
        ap.error("--max-parallel must be at least 1")
    if args.offline and args.yes_spend:
        ap.error("--offline is for the dry run (a run stages the real files)")

    st = wh.load_state(args.state)
    if args.finalize:
        new = finalize(st)
        row = next(r for r in new["rates"] if r["kind"] == KIND)
        print(f"rates.json: {KIND} {row['container_s_per_video_s']} container-s per video-s, "
              f"measured (n={row['n']}, ±{row['uncertainty']:.0%})")
        return 0
    if args.yes_spend:
        st.mark_killed()
    manifest = load_manifest(args.offline)
    projs = projections(manifest)
    caps = caps_for(projs)
    steps = planned(args.only)
    if args.global_cap is not None:
        if args.global_cap < st.spent:
            print(f"refused: --global-cap ${args.global_cap:.2f} is below the ${st.spent:.2f} "
                  "already spent")
            return 2
        st.cap_override = round(args.global_cap, 2)
    gcap = global_cap(projs, steps, st.cap_override)
    work = args.work.resolve()
    jobs = [JOBS_BY_TASK[s] for s in steps if s in JOBS_BY_TASK]
    pre = preconditions(manifest, jobs or list(JOBS), work, st.run_id)
    print_plan(st, projs, caps, gcap, steps, manifest, max_parallel=args.max_parallel)
    for k, v in pre.items():
        print(f"  PRECONDITION {k}: " + "; ".join(v))
    if not pre:
        print("  preconditions: ok (" + ("staging not checked offline; " if manifest.get(
            "offline") else "episodes staged; ") + "rubrics, image lock, TP=2 pool)")
    if args.report:
        OUT.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(render_report(st, projs, caps, gcap))
        print(f"wrote {REPORT}")
        return 0
    if not args.yes_spend:
        print("dry run: nothing started (run with --yes-spend after the user approves)")
        return 0
    if pre:
        print("refused: a precondition failed; nothing started")
        return 2

    work.mkdir(parents=True, exist_ok=True)
    args.state = args.state.resolve()
    if not st.billing_baseline:
        try:
            st.billing_baseline = wh.billed_by_app(st.billing_since)
        except Exception as e:  # noqa: BLE001
            print(f"billing baseline unavailable ({e!r}); tracking only")
    rb.stop_killed(st)  # by id where it has one; a deployed app by its name
    wh.save_state(st, args.state)
    os.chdir(work)  # a relative input.path is hashed against the cwd: plan where flora runs
    ctx = Ctx(st, manifest, work, caps, gcap, state_path=args.state, steps=tuple(steps),
              projections=projs)
    try:
        rc = rb.run_steps(
            st, steps, state_path=args.state, ctx=ctx,
            run=lambda name, c: (run_check(c) if name == CHECK
                                 else run_job(c, JOBS_BY_TASK[name])),
            projection=projs.__getitem__,
            decide=lambda name, p: rb.decide(st, steps, name, projs, caps=caps,
                                             global_cap=gcap),
            check_after=lambda name: wh.check_after(st, name, caps=caps, global_cap=gcap),
            refresh=lambda s: wh.refresh_billing(s),  # looked up per call (tests patch it)
            keep_going=lambda name: name == CHECK, deps=step_deps(steps),
            max_parallel=args.max_parallel)
    finally:
        for n in cleanup(st.apps_seen, st.run_id, keep=args.keep):
            print(f"  cleanup: {n}")
        wh.save_state(st, args.state)
        res = (st.steps.get(CHECK) or wh.StepState()).result
        if res:
            RESULTS.write_text(json.dumps(wh._numbers_only(res), indent=1, default=str))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(render_report(st, projs, caps, gcap))
        print(f"wrote {REPORT}; state {args.state}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
