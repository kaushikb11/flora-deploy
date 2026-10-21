"""Wave H-lite runbook (``docs/NEXT_EXECUTION.md``, 2026-10-22; ``docs/wave-h-runbook.md``).

Usage::

    uv run python scripts/wave_h.py --dry-run              # plan + estimates; spends nothing
    uv run python scripts/wave_h.py --yes-spend            # run (resumes: skips done steps)
    uv run python scripts/wave_h.py --yes-spend --only certify
    uv run python scripts/wave_h.py --report               # rewrite docs/wave-h-report.md

Steps, in order (each with a projected-spend stop rule before it and a spend check after it):

a. ``preflight``: balanced preflight (CPU): derived weight dirs, receipts, EOS ids
   (``balanced_lab.preflight``);
b. ``exact_e2e``: ``exact`` end to end on the 6 H0 episodes (``h0_rehearsal.EPISODES``) through
   the real ``flora index -> plan -> deploy -> run`` path, ``--determinism deterministic``, one
   H200 container per app (two apps: one embodiment per config; run one after the other, each
   app stopped as soon as its job is composed);
c. ``rewards``: ``reward.terminal`` + ``reward.dense`` on 3 of those episodes, one H100 container
   (``max_inputs: 2`` so both kinds share it);
d. ``balanced``: the balanced lab on the same 6 episodes: TempSeg on L40S, then the Captioner on
   H100, each <= 10 min, incl. L0 input parity and Gate A (``balanced_lab.run_tempseg`` /
   ``run_captioner``);
e. ``certify``: offline provisional certification of ``balanced`` from (b) + (d) ($0);
f. ``noise``: 2 of the episodes in the default (``nvidia``) determinism mode, compared with (b):
   the noise floor; (e) is re-evaluated against it.

**Spend.** Every container's cost is tracked as container-seconds x its $/h (Modal list prices
pinned in ``jobs.budget``: GPU + cores + memory), from ``flora report`` (the job's container
receipts and calls) for (b)/(c)/(f) and from the lab's own wall time for (d). ``scripts/spend.py``
(``modal billing report``) is read at the start and after every step and the per-app delta is
recorded too (billing data lags by about an hour, so the stop rule uses the larger of the two).

**Stop rules.** Before a step: its projected cost must fit its cap and the global cap minus what
was spent so far. During a step: GPU wall time is bounded by the remaining step cap at the
container's $/s (``flora run --budget`` plus a client-side deadline that cancels the job and stops
the app; lab calls get the same deadline as their timeout). After a step: tracked/billed spend
over the step cap or the global cap stops the runbook. Caps (USD): exact e2e 1.20, rewards 0.60,
balanced 1.20, noise 0.40, preflight 0.05; global 3.20.

**State** is ``reports/wave-h/state.json`` (git-ignored: it names private episodes); the report
``docs/wave-h-report.md`` holds numbers only. A step that finished is never re-run; a stopped or
failed step is retried on the next ``--yes-spend``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h0_rehearsal as h0  # noqa: E402 - the same episodes, staging and configs as H0

STATE = ROOT / "reports" / "wave-h" / "state.json"
REPORT = ROOT / "docs" / "wave-h-report.md"
STORE_VOLUME = "flora-wave-h"
MEDIA_VOLUME = "flora-wave-h-media"
GLOBAL_CAP_USD = 3.20
CAPS_USD = {"preflight": 0.05, "exact_e2e": 1.20, "rewards": 0.60, "balanced": 1.20,
            "certify": 0.0, "noise": 0.40}
LAB_MAX_S = 600  # (d): <= 10 min per lab call
LAB_EXPECTED_S = {"tempseg": 420, "captioner": 480}  # two engine loads + 6 episodes each
IDLE_AFTER_JOB_S = 30  # last receipt -> compose -> `modal app stop`
REWARD_EPISODES = ("yam/aloha_static_cups_open/episode_000000",
                   "yam/aloha_static_towel/episode_000000",
                   "single_view_robot/wgo/robointer_droid_000010")
NOISE_EPISODES = ("yam/aloha_static_cups_open/episode_000000",
                  "yam/aloha_static_towel/episode_000000")
# Provisional thresholds (until the S2b noise floor exists): balanced vs exact must be within
# NOISE_FACTOR x the measured exact-vs-exact noise, or these absolute bounds without one.
NOISE_FACTOR = 2.0
ABS_BOUNDS = {"boundary_mae_s": 1.0, "segment_iou": 0.6, "caption_jaccard": 0.3}
GATE_A_MAX_DLOGIT = 0.5


# ---------------------------------------------------------------------------------------------
# Prices and projections (pure)


@dataclass(frozen=True)
class Container:
    """One container shape: GPU type (None for CPU), cores, memory GiB."""

    gpu: str | None
    cpu: float
    memory_gib: float

    def usd_per_s(self, rates: Any = None) -> float:
        from flora_deploy.jobs.budget import pinned_rates

        r = rates or pinned_rates()
        gpu = r.gpu_per_s.get((self.gpu or "").rstrip("!"), 0.0) if self.gpu else 0.0
        return gpu + self.cpu * r.core_per_s + self.memory_gib * r.gib_per_s

    def usd_per_h(self, rates: Any = None) -> float:
        return self.usd_per_s(rates) * 3600.0


def pool_container(kind: str) -> Container:
    from flora_deploy.config.presets import get_preset

    p = get_preset("exact").pools[kind]
    return Container(p.gpu, float(p.cpu_cores), float(p.memory_gib))


LAB = {"tempseg": Container("L40S", 8.0, 48.0), "captioner": Container("H100", 8.0, 96.0)}
PREFLIGHT = Container(None, 2.0, 4.0)


@dataclass
class Projection:
    step: str
    items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, what: str, c: Container, seconds: float, *, upper_s: float | None = None
            ) -> None:
        self.items.append({"what": what, "gpu": c.gpu or "cpu", "seconds": round(seconds, 1),
                           "usd_per_h": round(c.usd_per_h(), 3),
                           "usd": round(c.usd_per_s() * seconds, 4),
                           "usd_upper": round(c.usd_per_s() * (upper_s or seconds), 4)})

    @property
    def usd(self) -> float:
        return round(sum(i["usd"] for i in self.items), 4)

    @property
    def usd_upper(self) -> float:
        return round(sum(i["usd_upper"] for i in self.items), 4)


def durations(manifest: Mapping[str, Any]) -> dict[str, float]:
    return {e["id"]: float(e["duration_s"]) for e in manifest["episodes"]}


def exact_job_seconds(video_s: float, table: Any = None) -> tuple[float, float]:
    """Nominal and upper container-seconds of one exact segcap job in one container."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    e = t.rate("exact", "segcap", 3, 1)
    st = t.startup_for("exact", "segcap")
    nominal = st.startup_s + video_s * e.container_s_per_video_s + IDLE_AFTER_JOB_S
    upper = (st.startup_s * (1 + st.uncertainty)
             + video_s * e.container_s_per_video_s * (1 + e.uncertainty) + IDLE_AFTER_JOB_S)
    return nominal, upper


def reward_job_seconds(video_s: float, table: Any = None) -> tuple[float, float]:
    """Both reward kinds in one container (``max_inputs: 2``)."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    st = t.startup_for("exact", "reward.terminal")
    work = sum(t.rate("exact", k, 3, 1).container_s_per_video_s * video_s
               for k in ("reward.terminal", "reward.dense"))
    unc = max(t.rate("exact", k, 3, 1).uncertainty for k in ("reward.terminal", "reward.dense"))
    return (st.startup_s + work + IDLE_AFTER_JOB_S,
            st.startup_s * (1 + st.uncertainty) + work * (1 + unc) + IDLE_AFTER_JOB_S)


def project(step: str, manifest: Mapping[str, Any]) -> Projection:
    """What ``step`` is expected to cost (nominal rates; ``usd_upper``: upper quantiles)."""
    dur = durations(manifest)
    p = Projection(step)
    if step == "preflight":
        p.add("balanced_lab.preflight", PREFLIGHT, 90.0)
    elif step == "exact_e2e":
        for job, ids in h0.EPISODES.items():
            nominal, upper = exact_job_seconds(sum(dur[i] for i in ids))
            p.add(f"segcap {job} ({len(ids)} ep, {sum(dur[i] for i in ids):.1f} s)",
                  pool_container("segcap"), nominal, upper_s=upper)
    elif step == "rewards":
        v = sum(dur[i] for i in REWARD_EPISODES)
        nominal, upper = reward_job_seconds(v)
        p.add(f"reward.terminal+dense ({len(REWARD_EPISODES)} ep, {v:.1f} s)",
              pool_container("reward.terminal"), nominal, upper_s=upper)
    elif step == "balanced":
        for kind in ("tempseg", "captioner"):
            p.add(f"balanced lab {kind}", LAB[kind], LAB_EXPECTED_S[kind], upper_s=LAB_MAX_S)
    elif step == "noise":
        v = sum(dur[i] for i in NOISE_EPISODES)
        nominal, upper = exact_job_seconds(v)
        p.add(f"segcap nvidia-mode ({len(NOISE_EPISODES)} ep, {v:.1f} s)",
              pool_container("segcap"), nominal, upper_s=upper)
    elif step == "certify":
        pass
    else:
        raise KeyError(step)
    return p


# ---------------------------------------------------------------------------------------------
# State and stop rules


STEPS = ("preflight", "exact_e2e", "rewards", "balanced", "certify", "noise")
TITLES = {"preflight": "a. balanced preflight (CPU)",
          "exact_e2e": "b. exact e2e, 6 episodes, deterministic, H200",
          "rewards": "c. rewards on 3 episodes, H100",
          "balanced": "d. balanced validation (L40S TempSeg + H100 Captioner), L0 + Gate A",
          "certify": "e. offline provisional certification",
          "noise": "f. 2-episode default-mode noise run"}


@dataclass
class StepState:
    status: str = "pending"  # pending | done | failed | stopped
    projected_usd: float | None = None
    tracked_usd: float = 0.0  # container-seconds x $/h
    billed_usd: float | None = None  # spend.py delta (lags ~1 h)
    started: float | None = None
    finished: float | None = None
    notes: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def spent(self) -> float:
        return max(self.tracked_usd, self.billed_usd or 0.0)


@dataclass
class State:
    run_id: str
    created: float
    billing_since: str
    billing_baseline: dict[str, float] = field(default_factory=dict)
    steps: dict[str, StepState] = field(default_factory=dict)

    def step(self, name: str) -> StepState:
        return self.steps.setdefault(name, StepState())

    @property
    def spent(self) -> float:
        return round(sum(s.spent for s in self.steps.values()), 4)

    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "created": self.created,
                "billing_since": self.billing_since, "billing_baseline": self.billing_baseline,
                "steps": {k: asdict(v) for k, v in self.steps.items()}}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> State:
        return cls(d["run_id"], d["created"], d["billing_since"],
                   dict(d.get("billing_baseline", {})),
                   {k: StepState(**v) for k, v in d.get("steps", {}).items()})


def load_state(path: Path = STATE) -> State:
    if path.exists():
        return State.from_json(json.loads(path.read_text()))
    return State(run_id=time.strftime("%m%d%H%M") + uuid.uuid4().hex[:4], created=time.time(),
                 billing_since=time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400)))


def save_state(st: State, path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st.to_json(), indent=1, sort_keys=True, default=str))


class StopRule(RuntimeError):
    """The runbook stops here (projected or actual spend over a cap)."""


def check_before(st: State, step: str, proj: Projection, *,
                 caps: Mapping[str, float] = CAPS_USD, global_cap: float = GLOBAL_CAP_USD
                 ) -> None:
    cap = caps[step]
    if proj.usd > cap + 1e-9:
        raise StopRule(f"{step}: projected ${proj.usd:.2f} is over its cap ${cap:.2f}")
    left = global_cap - st.spent
    if proj.usd > left + 1e-9:
        raise StopRule(f"{step}: projected ${proj.usd:.2f} is over the global budget left "
                       f"(${left:.2f} of ${global_cap:.2f})")


def check_after(st: State, step: str, *, caps: Mapping[str, float] = CAPS_USD,
                global_cap: float = GLOBAL_CAP_USD) -> None:
    s = st.step(step)
    if s.spent > caps[step] + 1e-9:
        raise StopRule(f"{step}: spent ${s.spent:.2f} (tracked ${s.tracked_usd:.2f}, billed "
                       f"${s.billed_usd or 0:.2f}) is over its cap ${caps[step]:.2f}")
    if st.spent > global_cap + 1e-9:
        raise StopRule(f"spent ${st.spent:.2f} in all is over the global cap ${global_cap:.2f}")


def gpu_deadline_s(st: State, step: str, c: Container, *, spent_in_step: float = 0.0,
                   caps: Mapping[str, float] = CAPS_USD,
                   global_cap: float = GLOBAL_CAP_USD) -> float:
    """Container seconds still affordable on ``c`` in this step (the in-step stop rule)."""
    left = min(caps[step] - spent_in_step, global_cap - st.spent - spent_in_step)
    return max(0.0, left) / c.usd_per_s()


# ---------------------------------------------------------------------------------------------
# Billing (scripts/spend.py)


def billed_by_app(since: str) -> dict[str, float]:
    import spend  # scripts/spend.py

    out: dict[str, float] = {}
    for row in spend.report(since):
        name = row.get("description") or row.get("Description") or "?"
        out[name] = out.get(name, 0.0) + float(row.get("cost") or row.get("Cost") or 0)
    return out


def billed_delta(st: State, apps: Iterable[str]) -> float | None:
    try:
        now = billed_by_app(st.billing_since)
    except Exception:  # noqa: BLE001 - billing is a cross-check; tracking still applies
        return None
    return round(sum(now.get(a, 0.0) - st.billing_baseline.get(a, 0.0) for a in set(apps)), 4)


# ---------------------------------------------------------------------------------------------
# Offline certification metrics (pure; unit-tested)


def _spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float, str]]:
    out = []
    for r in rows:
        a = r.get("start_s", r.get("start"))
        b = r.get("end_s", r.get("end"))
        if a is None or b is None:
            continue
        text = r.get("caption") or r.get("text") or r.get("description") or ""
        out.append((float(a), float(b), str(text)))
    return sorted(out)


def _iou(a: tuple[float, float, str], b: tuple[float, float, str]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _jaccard(x: str, y: str) -> float:
    a, b = set(x.lower().split()), set(y.lower().split())
    return len(a & b) / len(a | b) if a | b else 1.0


def compare_segments(ref: Sequence[Mapping[str, Any]], got: Sequence[Mapping[str, Any]]
                     ) -> dict[str, float]:
    """One episode: segment count delta, boundary MAE (each reference boundary to the nearest
    candidate boundary, seconds), mean best-match IoU of reference segments, and mean caption
    token Jaccard over those best matches."""
    r, g = _spans(ref), _spans(got)
    if not r or not g:
        return {"count_delta": float(len(g) - len(r)), "boundary_mae_s": math.inf,
                "segment_iou": 0.0, "caption_jaccard": 0.0}
    rb = sorted({x for s in r for x in s[:2]})
    gb = sorted({x for s in g for x in s[:2]})
    mae = sum(min(abs(b - c) for c in gb) for b in rb) / len(rb)
    ious, jac = [], []
    for s in r:
        best = max(g, key=lambda t: _iou(s, t))
        ious.append(_iou(s, best))
        jac.append(_jaccard(s[2], best[2]))
    return {"count_delta": float(len(g) - len(r)), "boundary_mae_s": round(mae, 4),
            "segment_iou": round(sum(ious) / len(ious), 4),
            "caption_jaccard": round(sum(jac) / len(jac), 4)}


def aggregate(per_episode: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_episode:
        return {}
    keys = ("boundary_mae_s", "segment_iou", "caption_jaccard")
    out = {k: round(sum(m[k] for m in per_episode.values()) / len(per_episode), 4)
           for k in keys}
    out["abs_count_delta"] = round(sum(abs(m["count_delta"]) for m in per_episode.values())
                                   / len(per_episode), 4)
    out["episodes"] = len(per_episode)
    return out


def certify(balanced: Mapping[str, float], *, l0: Sequence[Mapping[str, Any]] = (),
            gate_a: Mapping[str, Any] | None = None,
            noise: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Provisional verdict: L0 parity on every captured request, Gate A within bounds, and the
    balanced-vs-exact metrics within ``NOISE_FACTOR`` x the exact noise floor (or
    :data:`ABS_BOUNDS` before one exists)."""
    checks: dict[str, Any] = {}
    checks["l0"] = bool(l0) and all(r.get("ids_equal") and r.get("grid_equal") for r in l0)
    if gate_a:
        mx = gate_a.get("max_abs_dlogit")
        checks["gate_a"] = mx is not None and float(mx) <= GATE_A_MAX_DLOGIT
    else:
        checks["gate_a"] = False
    for k, bound in ABS_BOUNDS.items():
        v = balanced.get(k)
        if v is None:
            checks[k] = False
            continue
        if noise and noise.get(k) is not None:
            n = float(noise[k])
            if k == "boundary_mae_s":
                bound = max(NOISE_FACTOR * n, 0.1)
            else:  # similarity: allowed drop below 1 is NOISE_FACTOR x the noise drop
                bound = 1.0 - NOISE_FACTOR * (1.0 - n)
        checks[k] = (v <= bound) if k == "boundary_mae_s" else (v >= bound)
    return {"provisional": all(checks.values()), "checks": checks,
            "against": "noise floor" if noise else "absolute bounds", "metrics": dict(balanced),
            "noise": dict(noise or {})}


# ---------------------------------------------------------------------------------------------
# Step runners (GPU steps only with --yes-spend)


@dataclass
class Ctx:
    st: State
    manifest: dict[str, Any]
    work: Path
    log: Callable[[str], None] = print

    def sh(self, args: list[str], *, timeout: float = 1800, check: bool = True):
        return h0.sh(args, cwd=self.work, timeout=timeout, check=check,
                     log=self.work / "wave-h.log")


def _receipt() -> str:
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

    r = DERIVED_PINS_BY_KEY["captioner-bf16"].receipt_sha256
    if not r:
        raise RuntimeError("no pinned captioner-bf16 receipt")
    return r


def _stage_job(ctx: Ctx, job: str, ids: Sequence[str], *, embodiment: str,
               stages: tuple[str, ...], budget: float,
               resources: dict[str, str] | None = None) -> tuple[Path, str]:
    """Fetch, upload, write the config, ``flora index --local`` + mirror, ``flora plan``."""
    run = ctx.st.run_id
    h0.fetch_dataset(ctx.manifest, ctx.work, episodes={job: tuple(ids)}, poison_job=None)
    h0.upload_dir(MEDIA_VOLUME, ctx.work / job, f"{run}/{job}")
    job_id = f"wave-h-{run}-{job}"
    cfg = ctx.work / f"flora-{job}.yaml"
    cfg.write_text(h0.config_text(project=job_id, data_dir=job, embodiment=embodiment,
                                  stages=stages, store_volume=STORE_VOLUME,
                                  media_volume=MEDIA_VOLUME, run_id=run, budget_usd=budget,
                                  resources=resources, tag="flora-wave-h",
                                  header="Wave H-lite; generated by scripts/wave_h.py"))
    fl = h0.flora_bin()
    local = ctx.work / f"store-{job}"
    ctx.sh([fl, "index", "--local", "--config", str(cfg), "--job", job_id, "--store",
            f"file://{local}", "--out", str(ctx.work / f"summary-{job}.json")])
    h0.mirror_index(local, job_id, run, STORE_VOLUME)
    ctx.sh([fl, "plan", "--config", str(cfg), "--index", str(ctx.work / f"summary-{job}.json")])
    return cfg, job_id


def _run_job(ctx: Ctx, step: str, cfg: Path, job_id: str, *, receipt: bool, determinism: str,
             container: Container, spent_in_step: float) -> dict[str, Any]:
    """``flora deploy`` + ``flora run`` with the in-step stop rule, then ``modal app stop``."""
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import plan_app

    fl = h0.flora_bin()
    rec = ["--receipt", f"captioner_bf16={_receipt()}"] if receipt else []
    plan = plan_app(load_config(cfg), determinism=determinism,
                    derived_receipts={"captioner_bf16": _receipt()} if receipt else None)
    deadline = gpu_deadline_s(ctx.st, step, container, spent_in_step=spent_in_step)
    budget = max(0.01, min(CAPS_USD[step] - spent_in_step,
                           GLOBAL_CAP_USD - ctx.st.spent - spent_in_step))
    out: dict[str, Any] = {"app": plan.app_name, "job_id": job_id, "deadline_s": round(deadline)}
    ctx.sh([fl, "deploy", "--config", str(cfg), "--determinism", determinism, *rec],
           timeout=1800)
    try:
        args = [fl, "run", "--config", str(cfg), "--determinism", determinism, *rec, "--job",
                job_id, "--poll", "15", "--budget", f"{budget:.2f}"]
        try:
            p = ctx.sh(args, timeout=deadline + 240, check=False)  # + client-side slack
            out["rc"] = p.returncode
        except subprocess.TimeoutExpired:
            out["rc"] = "deadline"
            ctx.sh([fl, "cancel", job_id, "--config", str(cfg)], check=False, timeout=300)
    finally:
        ctx.sh(["modal", "app", "stop", "--yes", plan.app_name], check=False, timeout=300)
    rep = ctx.sh([fl, "report", job_id, "--config", str(cfg), "--json"], check=False,
                 timeout=900)
    try:
        r = json.loads(rep.stdout)
        out.update(state=r.get("state"), spend=r.get("spend", {}).get("total_usd"),
                   usd_per_video_hour=r.get("usd_per_video_hour"),
                   containers=r.get("spend", {}).get("containers"),
                   video_hours=r.get("video_hours"))
    except ValueError:
        out["report_error"] = (rep.stdout + rep.stderr)[-400:]
    return out


def run_preflight(ctx: Ctx) -> dict[str, Any]:
    import modal

    from flora_deploy.modal_app import balanced_lab as lab

    t0 = time.time()
    with modal.enable_output(), lab.app.run():
        out = lab.preflight.remote()
    wall = time.time() - t0
    ok = all(out.get(k, {}).get("receipt") for k in ("tempseg", "captioner"))
    return {"ok": ok, "wall_s": round(wall, 1), "tracked_usd": PREFLIGHT.usd_per_s() * wall,
            "apps": ["flora-balanced-lab"],
            "facts": {k: {kk: v.get(kk) for kk in ("receipt", "eos_token",
                                                    "extra_state_prefixes")}
                      for k, v in out.items() if isinstance(v, dict)}}


def run_exact(ctx: Ctx, step: str = "exact_e2e", *, episodes: Mapping[str, Sequence[str]] |
              None = None, determinism: str = h0.DETERMINISM) -> dict[str, Any]:
    eps = dict(h0.EPISODES if episodes is None else episodes)
    jobs, spent = {}, 0.0
    c = pool_container("segcap")
    for job in sorted(eps, key=lambda j: -len(eps[j])):  # the larger job first
        cfg, job_id = _stage_job(ctx, f"{step}-{job}", eps[job],
                                 embodiment="human" if job == "human" else "yam",
                                 stages=("segment", "caption"), budget=CAPS_USD[step])
        r = _run_job(ctx, step, cfg, job_id, receipt=True, determinism=determinism,
                     container=c, spent_in_step=spent)
        jobs[job] = r
        spent += float(r.get("spend") or 0.0)
        if r.get("state") not in ("done", "completed_with_errors"):
            break
    ok = all(r.get("state") == "done" for r in jobs.values()) and len(jobs) == len(eps)
    return {"ok": ok, "jobs": jobs, "tracked_usd": spent,
            "apps": [r["app"] for r in jobs.values()]}


def run_rewards(ctx: Ctx) -> dict[str, Any]:
    cfg, job_id = _stage_job(ctx, "rewards", REWARD_EPISODES, embodiment="yam",
                             stages=("reward.terminal", "reward.dense"),
                             budget=CAPS_USD["rewards"],
                             resources={"reward.terminal": "{max_containers: 1, max_inputs: 2}",
                                        "reward.dense": "{max_containers: 1, max_inputs: 2}"})
    r = _run_job(ctx, "rewards", cfg, job_id, receipt=False, determinism="nvidia",
                 container=pool_container("reward.terminal"), spent_in_step=0.0)
    return {"ok": r.get("state") == "done", "job": r, "tracked_usd": float(r.get("spend") or 0),
            "apps": [r["app"]]}


def run_balanced(ctx: Ctx) -> dict[str, Any]:
    import modal

    from flora_deploy.modal_app import balanced_lab as lab

    ids = [i for ids in h0.EPISODES.values() for i in ids]
    out: dict[str, Any] = {"episodes": ids, "kinds": {}}
    spent, seg_rel = 0.0, ""
    rdir = ROOT / "reports" / "wave-h"
    rdir.mkdir(parents=True, exist_ok=True)
    with modal.enable_output(), lab.app.run():
        for kind, fn in (("tempseg", lab.run_tempseg), ("captioner", lab.run_captioner)):
            c = LAB[kind]
            limit = min(LAB_MAX_S, gpu_deadline_s(ctx.st, "balanced", c, spent_in_step=spent))
            if limit < 120 or (kind == "captioner" and not seg_rel):
                out["kinds"][kind] = {"skipped": f"budget left {limit:.0f} s"
                                      if limit < 120 else "no TempSeg segments"}
                break
            t0 = time.time()
            call = fn.spawn(f"wave-h-{ctx.st.run_id}", seg_rel, ids, True)
            try:
                res = call.get(timeout=limit)
            except (TimeoutError, modal.exception.TimeoutError):
                call.cancel()
                res = {"error": f"client deadline {limit:.0f} s"}
            wall = time.time() - t0
            spent += c.usd_per_s() * wall
            (rdir / f"balanced-{kind}.json").write_text(json.dumps(res, default=str))
            base = res.get("baseline", {}) if isinstance(res, dict) else {}
            out["kinds"][kind] = {"wall_s": round(wall, 1), "usd": round(c.usd_per_s() * wall, 4),
                                  "lab_usd_estimate": res.get("usd_estimate"),
                                  "throughput": base.get("throughput"), "l0": base.get("l0"),
                                  "gate_a": res.get("gate_a"), "error": res.get("error")
                                  or base.get("error")}
            seg_rel = res.get("segments_rel", seg_rel) if kind == "tempseg" else seg_rel
    ok = all("error" not in v or not v["error"] for v in out["kinds"].values()) and \
        len(out["kinds"]) == 2 and not any("skipped" in v for v in out["kinds"].values())
    return {"ok": ok, **out, "tracked_usd": spent, "apps": ["flora-balanced-lab"]}


def _exact_rows(ctx: Ctx, step: str) -> dict[str, dict[str, list[dict]]]:
    """``{episode_id: {"segment": rows, "caption": rows}}`` from a finished exact step."""
    from flora_deploy import sdk

    rows: dict[str, dict[str, list[dict]]] = {}
    for r in ctx.st.step(step).result.get("jobs", {}).values():
        job = sdk.open_job(r["job_id"], store=f"modal-volume://{STORE_VOLUME}/{ctx.st.run_id}")
        for stage in ("segment", "caption"):
            for rec in job.results(stage):
                rows.setdefault(rec.episode_id, {})[stage] = [
                    {"start_s": s.start_s, "end_s": s.end_s, "caption": s.caption or ""}
                    for s in rec.segments]
    return rows


def _lab_rows(kind: str) -> dict[str, list[dict]]:
    p = ROOT / "reports" / "wave-h" / f"balanced-{kind}.json"
    if not p.exists():
        return {}
    res = json.loads(p.read_text())
    return {h0.episode_id(e["id"]): e.get("rows", []) for e in
            res.get("baseline", {}).get("episodes", []) if e.get("ok")}


def run_certify(ctx: Ctx, noise: Mapping[str, float] | None = None) -> dict[str, Any]:
    exact = _exact_rows(ctx, "exact_e2e")
    seg, cap = _lab_rows("tempseg"), _lab_rows("captioner")
    per = {}
    for eid, ex in exact.items():
        got = cap.get(eid) or seg.get(eid)
        if got is None:
            continue
        m = compare_segments(ex.get("caption") or ex.get("segment", []), got)
        if eid not in cap:
            m["caption_jaccard"] = 0.0
        per[eid] = m
    bal = ctx.st.step("balanced").result.get("kinds", {})
    l0 = [r for k in ("tempseg", "captioner") for r in (bal.get(k, {}).get("l0") or [])]
    gate = max((bal.get(k, {}).get("gate_a") or {} for k in ("tempseg", "captioner")),
               key=lambda g: float(g.get("max_abs_dlogit") or 0.0), default={})
    verdict = certify(aggregate(per), l0=l0, gate_a=gate or None, noise=noise)
    cert = {"schema": "flora_provisional_certificate_v0", "preset": "balanced",
            "pools": {"segment": "L40S", "caption": "H100"}, "reference": "exact deterministic",
            "verdict": verdict, "per_episode": per, "run_id": ctx.st.run_id}
    (ROOT / "reports" / "wave-h" / "certificate-provisional.json").write_text(
        json.dumps(cert, indent=1, default=str))
    return {"ok": True, "verdict": verdict, "episodes": len(per), "tracked_usd": 0.0,
            "apps": []}


def run_noise(ctx: Ctx) -> dict[str, Any]:
    r = run_exact(ctx, "noise", episodes={"yam": NOISE_EPISODES}, determinism="nvidia")
    exact = _exact_rows(ctx, "exact_e2e")
    ctx.st.step("noise").result = r  # so _exact_rows can read this step's job
    noise_rows = _exact_rows(ctx, "noise")
    per = {eid: compare_segments(exact[eid].get("caption", []), rows.get("caption", []))
           for eid, rows in noise_rows.items() if eid in exact}
    floor = aggregate(per)
    r["noise_floor"] = floor
    if ctx.st.step("certify").status == "done":
        r["certify_with_noise"] = run_certify(ctx, noise=floor)["verdict"]
    return r


RUNNERS: dict[str, Callable[[Ctx], dict[str, Any]]] = {
    "preflight": run_preflight, "exact_e2e": run_exact, "rewards": run_rewards,
    "balanced": run_balanced, "certify": run_certify, "noise": run_noise}


# ---------------------------------------------------------------------------------------------
# Report


def render_report(st: State, projections: Mapping[str, Projection]) -> str:
    lines = ["# Wave H-lite report", "",
             "Generated by `scripts/wave_h.py` (runbook: `docs/wave-h-runbook.md`). Numbers only:",
             "episode media, captions and NVIDIA material stay on private Volumes and in the",
             "git-ignored `reports/wave-h/`.", "",
             f"Run `{st.run_id}`. Spent so far **${st.spent:.2f}** of the ${GLOBAL_CAP_USD:.2f} "
             "global cap (tracked container-seconds x list $/h, or the billed delta from",
             "`scripts/spend.py` when larger; billing lags about an hour).", "",
             "| step | status | cap | projected (upper) | tracked | billed |",
             "|---|---|---|---|---|---|"]
    for name in STEPS:
        s = st.steps.get(name, StepState())
        p = projections.get(name)
        proj = f"${p.usd:.2f} (${p.usd_upper:.2f})" if p else "-"
        billed = f"${s.billed_usd:.2f}" if s.billed_usd is not None else "-"
        lines.append(f"| {TITLES[name]} | {s.status} | ${CAPS_USD[name]:.2f} | {proj} | "
                     f"${s.tracked_usd:.2f} | {billed} |")
    for name in STEPS:
        s = st.steps.get(name)
        if not s or s.status == "pending":
            continue
        lines += ["", f"## {TITLES[name]}", "", f"Status: {s.status}."]
        lines += [f"- {n}" for n in s.notes]
        res = {k: v for k, v in s.result.items() if k not in ("apps",)}
        if res:
            lines += ["", "```json", json.dumps(_numbers_only(res), indent=1, default=str)[:6000],
                      "```"]
    return "\n".join(lines) + "\n"


_PRIVATE_KEYS = {"rows", "caption", "captions", "text", "task", "facts", "episodes"}


def _numbers_only(d: Any) -> Any:
    """Drop text that could carry private episode content (captions, tasks, rows)."""
    if isinstance(d, Mapping):
        return {k: _numbers_only(v) for k, v in d.items() if k not in _PRIVATE_KEYS}
    if isinstance(d, list):
        return [_numbers_only(x) for x in d]
    return d


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print the plan; spend nothing")
    g.add_argument("--yes-spend", action="store_true", help="run (GPU steps spend money)")
    g.add_argument("--report", action="store_true", help="rewrite the report from the state")
    ap.add_argument("--only", choices=STEPS, help="run just this step (if not done)")
    ap.add_argument("--state", type=Path, default=STATE)
    ap.add_argument("--work", type=Path, default=ROOT / "reports" / "wave-h" / "work")
    args = ap.parse_args(argv)

    st = load_state(args.state)
    manifest = h0.golden_manifest()
    projections = {s: project(s, manifest) for s in STEPS}
    total = sum(p.usd for p in projections.values())
    print(f"Wave H-lite run {st.run_id}: spent ${st.spent:.2f} of ${GLOBAL_CAP_USD:.2f}")
    for name in STEPS:
        p = projections[name]
        s = st.steps.get(name, StepState())
        rule = "ok"
        try:
            check_before(st, name, p)
        except StopRule as e:
            rule = f"STOP: {e}"
        print(f"  {TITLES[name]:70s} [{s.status}] cap ${CAPS_USD[name]:.2f}  projected "
              f"${p.usd:.2f} (upper ${p.usd_upper:.2f})  rule: {rule}")
        for i in p.items:
            print(f"      {i['what']:45s} {i['gpu']:>5s} {i['seconds']:7.0f} s x "
                  f"${i['usd_per_h']:.2f}/h = ${i['usd']:.3f} (upper ${i['usd_upper']:.3f})")
    print(f"  total projected ${total:.2f} of ${GLOBAL_CAP_USD:.2f} global cap")
    if args.report:
        REPORT.write_text(render_report(st, projections))
        print(f"wrote {REPORT}")
        return 0
    if not args.yes_spend:
        print("dry run: nothing started (run with --yes-spend after the user approves)")
        return 0

    args.work = args.work.resolve()
    args.state = args.state.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    if not st.billing_baseline:
        try:
            st.billing_baseline = billed_by_app(st.billing_since)
        except Exception as e:  # noqa: BLE001
            print(f"billing baseline unavailable ({e!r}); tracking only")
    save_state(st, args.state)
    os.chdir(args.work)  # relative input.path is hashed against the cwd: plan where flora runs
    ctx = Ctx(st, manifest, args.work)
    rc = 0
    for name in STEPS:
        if args.only and name != args.only:
            continue
        s = st.step(name)
        if s.status == "done":
            print(f"skip {name}: done")
            continue
        s.projected_usd = projections[name].usd
        try:
            check_before(st, name, projections[name])
        except StopRule as e:
            s.status, rc = "stopped", 3
            s.notes.append(f"stop rule before: {e}")
            break
        s.started, s.status = time.time(), "running"
        save_state(st, args.state)
        try:
            res = RUNNERS[name](ctx)
            s.result = res
            s.tracked_usd = round(float(res.get("tracked_usd") or 0.0), 4)
            s.status = "done" if res.get("ok") else "failed"
        except Exception as e:  # noqa: BLE001 - recorded; the runbook stops
            s.status = "failed"
            s.notes.append(f"error: {e!r}")
        s.finished = time.time()
        s.billed_usd = billed_delta(st, s.result.get("apps", ()))
        save_state(st, args.state)
        try:
            check_after(st, name)
        except StopRule as e:
            s.notes.append(f"stop rule after: {e}")
            rc = 3
            break
        if s.status != "done":
            rc = 1
            break
    save_state(st, args.state)
    REPORT.write_text(render_report(st, projections))
    print(f"wrote {REPORT}; state {args.state}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
