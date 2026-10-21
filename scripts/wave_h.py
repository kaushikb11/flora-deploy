"""Wave H-lite runbook (``docs/archive/NEXT_EXECUTION.md``, 2026-10-22; ``docs/wave-h-runbook.md``).

Usage::

    uv run python scripts/wave_h.py --dry-run              # plan + estimates; spends nothing
    uv run python scripts/wave_h.py --yes-spend            # run (resumes: skips done steps)
    uv run python scripts/wave_h.py --yes-spend --only certify
    uv run python scripts/wave_h.py --report               # rewrite docs/wave-h-report.md

Steps, in order (each with a projected-spend stop rule before it and a spend check after it):

a. ``preflight``: balanced preflight (CPU): derived weight dirs, receipts, EOS ids
   (``balanced_lab.preflight``);
b. ``exact_e2e``: the ``exact`` **oracle**, end to end on the 6 H0 episodes
   (``h0_rehearsal.EPISODES``) through the real ``flora index -> plan -> deploy -> run`` path, in
   NVIDIA's **default mode** (``determinism: nvidia``: the outputs users compare with), one H200
   container per app (two apps: one embodiment per config; run one after the other, each app
   stopped as soon as its job is composed, so two H200 cold starts);
c. ``rewards``: ``reward.terminal`` + ``reward.dense`` on 3 of those episodes, one H100 container
   (``max_inputs: 2`` so both kinds share it);
d. ``balanced``: ``preset: balanced`` (``allow_uncertified: true``) through the same real
   ``flora deploy`` / ``flora run`` on the same episodes and jobs as (b) (``SegmentWorker`` on
   L40S, ``CaptionWorker`` on H100), then the ``balanced_lab`` L0 input parity + Gate A on 6
   captured requests (``balanced_lab.run_tempseg`` / ``run_captioner``);
e. ``certify``: offline provisional certification of ``balanced`` from (b) + (d) ($0);
f. ``noise``: the oracle's ``human`` job **repeated** (same config file, staged data, stage hash,
   GPU and default mode; a new job id), compared with (b): the noise floor; (e) is re-evaluated.
   The cheapest valid repeat: certification needs the noise run to repeat the oracle (Fix-Cert),
   and the input content is hashed, so a smaller dataset would be another stage hash. With 2
   noise episodes (< 5) the certificate is provisional by design.

**Derived weights** are resolved the way ``flora deploy`` resolves them
(:func:`resolve_receipts`): each needed receipt is read from the ``flora-weights`` Volume,
``config.receipts.check_pinned`` checks its content digest against ``pins.DERIVED_PINS`` and its
file sha256 is written into the generated config's ``weights.derived``; ``flora deploy`` /
``flora run`` then read it from there (``config.receipts.resolve``). Step (d) also refuses a
derived weight with no pinned content digest (the balanced workers fail closed on it).

**Spend.** Every attempt of a step is kept (:class:`Attempt`); a step's spend is the sum over its
attempts, so a retry adds instead of overwriting. Tracked cost is container-seconds x list $/h
(``flora report`` for the jobs, the lab's own wall time), and while a step runs a heartbeat stores
``elapsed x the step's peak $/s`` so a killed process leaves its spend behind. An attempt that was
killed, or whose ``flora report`` failed, is charged the most it was allowed to spend (its
budget) until billing, read again before every step for **every** app seen so far, is at least
:data:`BILLING_LAG_S` past it. The global figure is ``max(sum of charged attempts, billed delta of
all apps)``; the global cap is checked on it.

**Stop rules.** Before a step: its **upper** projection (upper rate quantiles, both cold starts,
the scale-down tail and :data:`CLIENT_SLACK_S` per app) must fit its cap and the global cap minus
the cumulative spend. During a step: GPU wall time is bounded by ``(remaining step cap - slack x
$/s) / $/s`` (``flora run --budget`` plus a client-side deadline that cancels the job and stops
the app; lab calls get the same deadline as their timeout). After a step: tracked/billed spend
over the step cap or the global cap stops the runbook. The dry run prints nominal and upper costs
per step and, when the upper total does not fit :data:`GLOBAL_CAP_USD`, the minimum honest cap.

**State** is ``reports/wave-h/state.json`` (git-ignored: it names private episodes); the report
``docs/wave-h-report.md`` holds numbers only. A step that finished is never re-run; a stopped,
failed or killed step is retried on the next ``--yes-spend`` (a killed step's apps are stopped
first).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
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
# Per-step caps (USD). (b): $1.80, set by the user for two H200 cold starts (nominal $1.19),
# below its honest upper ($2.55 with 240 s client slack per app). The others are their upper
# projections rounded up (``honest_caps``). Their sum does NOT fit the $3.20 global cap: the dry
# run says so and names the minimum honest cap; the stop rules then refuse the first step that
# does not fit. Raising the caps is the user's decision.
CAPS_USD = {"preflight": 0.05, "exact_e2e": 1.80, "rewards": 0.90, "balanced": 4.35,
            "certify": 0.0, "noise": 1.20}
LAB_MAX_S = 600  # (d): <= 10 min per lab call
LAB_EXPECTED_S = {"tempseg": 420, "captioner": 480}  # two engine loads + 6 episodes each
LAB_SLACK_S = 60  # lab: client timeout -> cancel -> container gone
IDLE_AFTER_JOB_S = 30  # last receipt -> compose -> `modal app stop`
CLIENT_SLACK_S = 240  # per app: `flora run` timeout margin, then cancel + `modal app stop`
BILLING_LAG_S = 3600  # Modal billing covers complete hours only
HEARTBEAT_S = 30.0
ORACLE_DETERMINISM = "nvidia"  # Fix-Cert: oracle and noise run in NVIDIA's default mode
NOISE_JOB = "human"  # the oracle job (f) repeats
REWARD_EPISODES = ("yam/aloha_static_cups_open/episode_000000",
                   "yam/aloha_static_towel/episode_000000",
                   "single_view_robot/wgo/robointer_droid_000010")
NOISE_EPISODES = h0.EPISODES[NOISE_JOB]
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


def pool_container(kind: str, preset: str = "exact") -> Container:
    from flora_deploy.config.presets import get_preset

    p = get_preset(preset).pools[kind]
    return Container(p.gpu, float(p.cpu_cores), float(p.memory_gib))


LAB = {"tempseg": Container("L40S", 8.0, 48.0), "captioner": Container("H100", 8.0, 96.0)}
PREFLIGHT = Container(None, 2.0, 4.0)


@dataclass
class Projection:
    step: str
    items: list[dict[str, Any]] = field(default_factory=list)
    peak_usd_per_s: float = 0.0  # the most the step can burn at once (heartbeat estimate)

    def add(self, what: str, c: Container, seconds: float, *, upper_s: float | None = None
            ) -> None:
        self.items.append({"what": what, "gpu": c.gpu or "cpu", "seconds": round(seconds, 1),
                           "upper_seconds": round(upper_s or seconds, 1),
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
    """Nominal and upper container-seconds of one exact segcap job in one container (startup,
    work, idle until ``modal app stop``; upper: upper quantiles plus :data:`CLIENT_SLACK_S`)."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    e = t.rate("exact", "segcap", 3, 1)
    st = t.startup_for("exact", "segcap")
    nominal = st.startup_s + video_s * e.container_s_per_video_s + IDLE_AFTER_JOB_S
    upper = (st.startup_s * (1 + st.uncertainty)
             + video_s * e.container_s_per_video_s * (1 + e.uncertainty) + IDLE_AFTER_JOB_S
             + CLIENT_SLACK_S)
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
            st.startup_s * (1 + st.uncertainty) + work * (1 + unc) + IDLE_AFTER_JOB_S
            + CLIENT_SLACK_S)


def balanced_job_seconds(video_s: float, table: Any = None) -> dict[str, tuple[float, float]]:
    """One balanced job (one app): kind -> (nominal, upper) container-seconds. Each pool has
    its own cold start; the segment container then idles through the scale-down window while
    captions run (the caption shards wait for it); both stay up through the client slack."""
    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    out: dict[str, tuple[float, float]] = {}
    work: dict[str, tuple[float, float, float]] = {}
    for kind in ("segment", "caption"):
        e = t.rate("balanced", kind, 3, 1)
        st = t.startup_for("balanced", kind)
        work[kind] = (st.startup_s, video_s * e.container_s_per_video_s,
                      max(e.uncertainty, st.uncertainty))
    cap_nominal = work["caption"][0] + work["caption"][1]
    for kind, (start, w, unc) in work.items():
        idle = min(WORKER_SCALEDOWN_S, cap_nominal) if kind == "segment" else 0.0
        idle_up = WORKER_SCALEDOWN_S if kind == "segment" else 0.0
        out[kind] = (start + w + idle + IDLE_AFTER_JOB_S,
                     (start + w) * (1 + unc) + idle_up + IDLE_AFTER_JOB_S + CLIENT_SLACK_S)
    return out


def project(step: str, manifest: Mapping[str, Any]) -> Projection:
    """What ``step`` is expected to cost (nominal rates; ``usd_upper``: upper quantiles plus
    every cold start, scale-down tail and client slack the step can incur)."""
    dur = durations(manifest)
    p = Projection(step)
    if step == "preflight":
        p.add("balanced_lab.preflight", PREFLIGHT, 90.0, upper_s=90.0 + LAB_SLACK_S)
        p.peak_usd_per_s = PREFLIGHT.usd_per_s()
    elif step == "exact_e2e":
        for job, ids in h0.EPISODES.items():
            v = sum(dur[i] for i in ids)
            nominal, upper = exact_job_seconds(v)
            p.add(f"segcap {job} ({len(ids)} ep, {v:.1f} s, cold start)",
                  pool_container("segcap"), nominal, upper_s=upper)
        p.peak_usd_per_s = pool_container("segcap").usd_per_s()  # apps run one at a time
    elif step == "rewards":
        v = sum(dur[i] for i in REWARD_EPISODES)
        nominal, upper = reward_job_seconds(v)
        c = pool_container("reward.terminal")
        p.add(f"reward.terminal+dense ({len(REWARD_EPISODES)} ep, {v:.1f} s)", c, nominal,
              upper_s=upper)
        p.peak_usd_per_s = c.usd_per_s()
    elif step == "balanced":
        seg, cap = pool_container("segment", "balanced"), pool_container("caption", "balanced")
        for job, ids in h0.EPISODES.items():
            v = sum(dur[i] for i in ids)
            secs = balanced_job_seconds(v)
            p.add(f"balanced segment {job} ({len(ids)} ep, {v:.1f} s, cold start)", seg,
                  secs["segment"][0], upper_s=secs["segment"][1])
            p.add(f"balanced caption {job} ({len(ids)} ep, {v:.1f} s, cold start)", cap,
                  secs["caption"][0], upper_s=secs["caption"][1])
        for kind in ("tempseg", "captioner"):
            p.add(f"balanced lab {kind} (L0 + Gate A)", LAB[kind], LAB_EXPECTED_S[kind],
                  upper_s=LAB_MAX_S + LAB_SLACK_S)
        p.peak_usd_per_s = max(seg.usd_per_s() + cap.usd_per_s(),
                               max(c.usd_per_s() for c in LAB.values()))
    elif step == "noise":
        v = sum(dur[i] for i in NOISE_EPISODES)
        nominal, upper = exact_job_seconds(v)
        p.add(f"segcap {NOISE_JOB} repeat ({len(NOISE_EPISODES)} ep, {v:.1f} s, cold start)",
              pool_container("segcap"), nominal, upper_s=upper)
        p.peak_usd_per_s = pool_container("segcap").usd_per_s()
    elif step == "certify":
        pass
    else:
        raise KeyError(step)
    return p


def honest_caps(projections: Mapping[str, Projection]) -> dict[str, float]:
    """Per step, the smallest cap (rounded up to 5 cents) its upper projection fits, and the
    global total of those (``"global"``)."""
    out = {s: math.ceil(p.usd_upper * 20 - 1e-9) / 20 for s, p in projections.items()}
    out["global"] = round(sum(out.values()), 2)
    return out


# ---------------------------------------------------------------------------------------------
# State and stop rules


STEPS = ("preflight", "exact_e2e", "rewards", "balanced", "certify", "noise")
TITLES = {"preflight": "a. balanced preflight (CPU)",
          "exact_e2e": "b. exact oracle, 6 episodes, default mode, 2 x H200",
          "rewards": "c. rewards on 3 episodes, H100",
          "balanced": "d. balanced via flora deploy/run (L40S + H100) + lab L0 / Gate A",
          "certify": "e. offline provisional certification",
          "noise": "f. noise: oracle human job repeated, default mode, H200"}


@dataclass
class Attempt:
    """One run of a step (module docstring, "Spend")."""

    started: float
    status: str = "running"  # running | done | failed | stopped | killed
    finished: float | None = None
    heartbeat: float | None = None
    rate_usd_per_s: float = 0.0  # the step's peak burn (heartbeat estimate)
    budget_usd: float = 0.0  # the most this attempt was allowed to spend
    tracked_usd: float = 0.0  # final: report / wall time; while running: the heartbeat estimate
    apps: list[str] = field(default_factory=list)
    report_error: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def last_activity(self) -> float:
        return self.finished or self.heartbeat or self.started

    def charged(self, billed_at: float | None) -> float:
        """Tracked cost, or the attempt's budget while it cannot be known (killed, no report)
        and billing has not caught up with it."""
        if self.status == "killed" or self.report_error:
            reconciled = billed_at is not None and billed_at - self.last_activity >= BILLING_LAG_S
            if not reconciled:
                return max(self.tracked_usd, self.budget_usd)
        return self.tracked_usd


@dataclass
class StepState:
    status: str = "pending"  # pending | running | done | failed | stopped
    projected_usd: float | None = None
    projected_upper_usd: float | None = None
    billed_usd: float | None = None  # billed delta of the apps only this step used
    attempts: list[Attempt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def tracked_usd(self) -> float:
        return round(sum(a.tracked_usd for a in self.attempts), 4)

    def charged(self, billed_at: float | None = None) -> float:
        return round(sum(a.charged(billed_at) for a in self.attempts), 4)

    def spent_at(self, billed_at: float | None = None) -> float:
        return max(self.charged(billed_at), self.billed_usd or 0.0)

    @property
    def spent(self) -> float:
        return self.spent_at(None)

    @property
    def current(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts and self.attempts[-1].status == "running" \
            else None

    @property
    def apps(self) -> list[str]:
        return sorted({a for x in self.attempts for a in x.apps})


@dataclass
class State:
    run_id: str
    created: float
    billing_since: str
    billing_baseline: dict[str, float] = field(default_factory=dict)
    steps: dict[str, StepState] = field(default_factory=dict)
    billed_total: float | None = None  # billed delta over every app seen so far
    billed_at: float | None = None  # when billing was last read

    def step(self, name: str) -> StepState:
        return self.steps.setdefault(name, StepState())

    @property
    def apps_seen(self) -> list[str]:
        return sorted({a for s in self.steps.values() for a in s.apps})

    @property
    def spent(self) -> float:
        """The cumulative figure the global cap is checked on (module docstring)."""
        tracked = sum(s.spent_at(self.billed_at) for s in self.steps.values())
        return round(max(tracked, self.billed_total or 0.0), 4)

    def spent_outside(self, step: str) -> float:
        """Cumulative spend minus ``step``'s running attempt (in-step deadlines add their own)."""
        cur = self.step(step).current
        own = cur.charged(self.billed_at) if cur is not None else 0.0
        return round(max(0.0, self.spent - own), 4)

    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "created": self.created,
                "billing_since": self.billing_since, "billing_baseline": self.billing_baseline,
                "billed_total": self.billed_total, "billed_at": self.billed_at,
                "steps": {k: asdict(v) for k, v in self.steps.items()}}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> State:
        steps = {}
        for k, v in d.get("steps", {}).items():
            v = dict(v)
            atts = [Attempt(**a) for a in v.pop("attempts", [])]
            legacy = float(v.pop("tracked_usd", 0.0) or 0.0)  # pre-attempts state files
            started, finished = v.pop("started", None), v.pop("finished", None)
            if legacy and not atts:
                atts = [Attempt(started=started or 0.0, finished=finished, tracked_usd=legacy,
                                status=v.get("status", "failed"))]
            steps[k] = StepState(**{**v, "attempts": atts})
        return cls(d["run_id"], d["created"], d["billing_since"],
                   dict(d.get("billing_baseline", {})), steps, d.get("billed_total"),
                   d.get("billed_at"))

    def mark_killed(self) -> list[str]:
        """Attempts left ``running`` by a killed process: ``killed`` (charged their budget
        until billing reconciles them), their step ``failed`` (retried). Returns their apps."""
        apps: list[str] = []
        for name, s in self.steps.items():
            for a in s.attempts:
                if a.status == "running":
                    a.status = "killed"
                    a.notes.append("process killed while running; charged its budget until "
                                   "billing reconciles it")
                    apps += a.apps
            if s.status == "running":
                s.status = "failed"
                s.notes.append(f"{name}: the previous run was killed mid-step")
        return sorted(set(apps))


def load_state(path: Path = STATE) -> State:
    if path.exists():
        return State.from_json(json.loads(path.read_text()))
    return State(run_id=time.strftime("%m%d%H%M") + uuid.uuid4().hex[:4], created=time.time(),
                 billing_since=time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400)))


_SAVE_LOCK = threading.RLock()


def save_state(st: State, path: Path = STATE) -> None:
    with _SAVE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.to_json(), indent=1, sort_keys=True, default=str))
        os.replace(tmp, path)


class StopRule(RuntimeError):
    """The runbook stops here (projected or actual spend over a cap)."""


def check_before(st: State, step: str, proj: Projection, *,
                 caps: Mapping[str, float] = CAPS_USD, global_cap: float = GLOBAL_CAP_USD
                 ) -> None:
    """The **upper** projection must fit the step cap and the global cap left (H2)."""
    cap = caps[step]
    up = proj.usd_upper
    if up > cap + 1e-9:
        raise StopRule(f"{step}: upper projection ${up:.2f} (nominal ${proj.usd:.2f}) is over "
                       f"its cap ${cap:.2f}")
    left = global_cap - st.spent
    if up > left + 1e-9:
        raise StopRule(f"{step}: upper projection ${up:.2f} is over the global budget left "
                       f"(${left:.2f} of ${global_cap:.2f}, ${st.spent:.2f} spent)")


def check_after(st: State, step: str, *, caps: Mapping[str, float] = CAPS_USD,
                global_cap: float = GLOBAL_CAP_USD) -> None:
    s = st.step(step)
    spent = s.spent_at(st.billed_at)
    if spent > caps[step] + 1e-9:
        raise StopRule(f"{step}: spent ${spent:.2f} (tracked ${s.tracked_usd:.2f}, billed "
                       f"${s.billed_usd or 0:.2f}) is over its cap ${caps[step]:.2f}")
    if st.spent > global_cap + 1e-9:
        raise StopRule(f"spent ${st.spent:.2f} in all is over the global cap ${global_cap:.2f}")


def step_budget(st: State, step: str, *, spent_in_step: float = 0.0,
                caps: Mapping[str, float] = CAPS_USD, global_cap: float = GLOBAL_CAP_USD
                ) -> float:
    """What the step may still spend: its cap minus its earlier attempts, and the global cap
    left, minus what this attempt already spent."""
    s = st.step(step)
    cur = s.current
    earlier = s.spent_at(st.billed_at) - (cur.charged(st.billed_at) if cur else 0.0)
    return max(0.0, min(caps[step] - earlier, global_cap - st.spent_outside(step))
               - spent_in_step)


def gpu_deadline_s(st: State, step: str, c: Container | float, *, spent_in_step: float = 0.0,
                   slack_s: float = CLIENT_SLACK_S, caps: Mapping[str, float] = CAPS_USD,
                   global_cap: float = GLOBAL_CAP_USD) -> float:
    """Wall seconds still affordable at ``c``'s $/s (a container or a $/s figure) in this step,
    keeping ``slack_s`` of it for the cancel / stop that follows the deadline (H2)."""
    rate = c if isinstance(c, float | int) else c.usd_per_s()
    left = step_budget(st, step, spent_in_step=spent_in_step, caps=caps, global_cap=global_cap)
    return max(0.0, left - slack_s * rate) / rate


# ---------------------------------------------------------------------------------------------
# Billing (scripts/spend.py)


def billed_by_app(since: str) -> dict[str, float]:
    import spend  # scripts/spend.py

    out: dict[str, float] = {}
    for row in spend.report(since):
        name = row.get("description") or row.get("Description") or "?"
        out[name] = out.get(name, 0.0) + float(row.get("cost") or row.get("Cost") or 0)
    return out


def billed_delta(st: State, apps: Iterable[str], now: Mapping[str, float] | None = None
                 ) -> float | None:
    if now is None:
        try:
            now = billed_by_app(st.billing_since)
        except Exception:  # noqa: BLE001 - billing is a cross-check; tracking still applies
            return None
    return round(sum(now.get(a, 0.0) - st.billing_baseline.get(a, 0.0) for a in set(apps)), 4)


def refresh_billing(st: State, *, now: Mapping[str, float] | None = None,
                    clock: Callable[[], float] = time.time) -> bool:
    """Re-read billing for every app seen so far (H3): the global billed total, and per step
    the billed delta of the apps only that step used (shared apps count globally only)."""
    if now is None:
        try:
            now = billed_by_app(st.billing_since)
        except Exception:  # noqa: BLE001
            return False
    st.billed_total = billed_delta(st, st.apps_seen, now)
    owners: dict[str, set[str]] = {}
    for name, s in st.steps.items():
        for a in s.apps:
            owners.setdefault(a, set()).add(name)
    for name, s in st.steps.items():
        mine = [a for a in s.apps if owners.get(a) == {name}]
        s.billed_usd = billed_delta(st, mine, now) if mine else s.billed_usd
    st.billed_at = clock()
    return True


class Heartbeat:
    """While a step runs: every ``every`` seconds, store ``elapsed x peak $/s`` as the running
    attempt's tracked spend and save the state, so a killed process leaves it behind (H3)."""

    def __init__(self, st: State, attempt: Attempt, path: Path, *, every: float = HEARTBEAT_S,
                 clock: Callable[[], float] = time.time):
        self.st, self.attempt, self.path, self.every, self.clock = st, attempt, path, every, clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wave-h-heartbeat")

    def beat(self) -> None:
        now = self.clock()
        a = self.attempt
        a.heartbeat = now
        a.tracked_usd = round(max(a.tracked_usd, (now - a.started) * a.rate_usd_per_s), 4)
        save_state(self.st, self.path)

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            self.beat()

    def __enter__(self) -> Heartbeat:
        self.beat()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


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
    state_path: Path = STATE
    step: str = ""

    def sh(self, args: list[str], *, timeout: float = 1800, check: bool = True):
        return h0.sh(args, cwd=self.work, timeout=timeout, check=check,
                     log=self.work / "wave-h.log")

    def note_app(self, name: str) -> None:
        """Record an app in the running attempt *before* it can spend (H3)."""
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None and name not in cur.apps:
            cur.apps.append(name)
            save_state(self.st, self.state_path)


def resolve_receipts(names: Iterable[str], *, read: Callable[[str], bytes | None] | None = None,
                     require_pin: bool = False) -> dict[str, str]:
    """``weights.derived`` for ``names``, as ``flora prepare`` records it (H1): each receipt read
    from the ``flora-weights`` Volume, its content digest checked against ``pins.DERIVED_PINS``
    (``config.receipts.check_pinned``), its file sha256 returned. ``require_pin``: also refuse a
    derived weight without a pinned content digest (the balanced workers fail closed on it)."""
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
    from flora_deploy.config.receipts import check_pinned, pin_key
    from flora_deploy.prep.derived import DERIVED_BUILDS, receipt_sha256

    if read is None:
        from flora_deploy.cli.prepare import _read_weights_file as read
    out: dict[str, str] = {}
    for name in sorted(set(names)):
        path = DERIVED_BUILDS[name].receipt_path()
        raw = read(path)
        if raw is None:
            raise RuntimeError(f"{name}: no receipt at flora-weights:{path} (run `flora "
                               "prepare`)")
        check_pinned(name, json.loads(raw), where=f"flora-weights:{path}")
        pin = DERIVED_PINS_BY_KEY.get(pin_key(name))
        if require_pin and (pin is None or pin.content_sha256 is None):
            raise RuntimeError(f"{name}: no content digest is pinned in pins.DERIVED_PINS; the "
                               "balanced workers refuse it at start (pin it first)")
        out[name] = receipt_sha256(raw)
    return out


def _stage_job(ctx: Ctx, job: str, ids: Sequence[str], *, embodiment: str,
               stages: tuple[str, ...], budget: float, preset: str = "exact",
               resources: dict[str, str] | None = None,
               derived: dict[str, str] | None = None) -> tuple[Path, str]:
    """Fetch, upload, write the config, ``flora index --local`` + mirror, ``flora plan``."""
    run = ctx.st.run_id
    h0.fetch_dataset(ctx.manifest, ctx.work, episodes={job: tuple(ids)}, poison_job=None)
    h0.upload_dir(MEDIA_VOLUME, ctx.work / job, f"{run}/{job}")
    job_id = f"wave-h-{run}-{job}"
    cfg = ctx.work / f"flora-{job}.yaml"
    cfg.write_text(h0.config_text(project=job_id, data_dir=job, embodiment=embodiment,
                                  stages=stages, store_volume=STORE_VOLUME,
                                  media_volume=MEDIA_VOLUME, run_id=run, budget_usd=budget,
                                  resources=resources, tag="flora-wave-h", preset=preset,
                                  derived=derived,
                                  header="Wave H-lite; generated by scripts/wave_h.py"))
    _index(ctx, cfg, job_id, job)
    return cfg, job_id


def _index(ctx: Ctx, cfg: Path, job_id: str, tag: str) -> None:
    fl = h0.flora_bin()
    local = ctx.work / f"store-{tag}"
    ctx.sh([fl, "index", "--local", "--config", str(cfg), "--job", job_id, "--store",
            f"file://{local}", "--out", str(ctx.work / f"summary-{tag}.json")])
    h0.mirror_index(local, job_id, ctx.st.run_id, STORE_VOLUME)
    ctx.sh([fl, "plan", "--config", str(cfg), "--index", str(ctx.work / f"summary-{tag}.json")])


def _run_job(ctx: Ctx, step: str, cfg: Path, job_id: str, *, determinism: str,
             usd_per_s: float, spent_in_step: float) -> dict[str, Any]:
    """``flora deploy`` + ``flora run`` with the in-step stop rule, then ``modal app stop``.
    Receipts come from the config's ``weights.derived`` (``config.receipts.resolve``), as for
    ``flora deploy``."""
    from flora_deploy.config.receipts import resolve
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import plan_app

    fl = h0.flora_bin()
    c = load_config(cfg)
    plan = plan_app(c, determinism=determinism, derived_receipts=resolve(c))
    deadline = gpu_deadline_s(ctx.st, step, usd_per_s, spent_in_step=spent_in_step)
    budget = max(0.01, step_budget(ctx.st, step, spent_in_step=spent_in_step))
    out: dict[str, Any] = {"app": plan.app_name, "job_id": job_id, "deadline_s": round(deadline),
                           "budget_usd": round(budget, 4), "cfg": str(cfg)}
    if deadline < 60:
        out["state"] = "skipped"
        out["error"] = f"budget left allows {deadline:.0f} s"
        return out
    ctx.note_app(plan.app_name)
    ctx.sh([fl, "deploy", "--config", str(cfg), "--determinism", determinism], timeout=1800)
    try:
        args = [fl, "run", "--config", str(cfg), "--determinism", determinism, "--job", job_id,
                "--poll", "15", "--budget", f"{budget:.2f}"]
        try:
            p = ctx.sh(args, timeout=deadline, check=False)
            out["rc"] = p.returncode
        except subprocess.TimeoutExpired:
            out["rc"] = "deadline"
            ctx.sh([fl, "cancel", job_id, "--config", str(cfg)], check=False, timeout=120)
    finally:
        ctx.sh(["modal", "app", "stop", "--yes", plan.app_name], check=False, timeout=120)
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
    if out.get("spend") is None:  # unknown: charge what it was allowed (H3)
        out["report_error"] = out.get("report_error") or "no spend in flora report"
        out["spend"] = budget
    return out


def run_preflight(ctx: Ctx) -> dict[str, Any]:
    import modal

    from flora_deploy.modal_app import balanced_lab as lab

    t0 = time.time()
    ctx.note_app("flora-balanced-lab")
    with modal.enable_output(), lab.app.run():
        out = lab.preflight.remote()
    wall = time.time() - t0
    ok = all(out.get(k, {}).get("receipt") for k in ("tempseg", "captioner"))
    return {"ok": ok, "wall_s": round(wall, 1), "tracked_usd": PREFLIGHT.usd_per_s() * wall,
            "apps": ["flora-balanced-lab"],
            "facts": {k: {kk: v.get(kk) for kk in ("receipt", "eos_token",
                                                    "extra_state_prefixes")}
                      for k, v in out.items() if isinstance(v, dict)}}


def _jobs_result(jobs: Mapping[str, Mapping[str, Any]], n: int) -> dict[str, Any]:
    spent = sum(float(r.get("spend") or 0.0) for r in jobs.values())
    return {"ok": len(jobs) == n and all(r.get("state") == "done" for r in jobs.values()),
            "jobs": dict(jobs), "tracked_usd": spent,
            "report_error": any(r.get("report_error") for r in jobs.values()),
            "apps": [r["app"] for r in jobs.values() if "app" in r]}


def run_exact(ctx: Ctx, step: str = "exact_e2e") -> dict[str, Any]:
    """(b): the oracle, default mode, one app per embodiment (two H200 cold starts)."""
    eps = dict(h0.EPISODES)
    derived = resolve_receipts(["captioner_bf16"])
    jobs: dict[str, dict[str, Any]] = {}
    spent = 0.0
    rate = pool_container("segcap").usd_per_s()
    for job in sorted(eps, key=lambda j: -len(eps[j])):  # the larger job first
        cfg, job_id = _stage_job(ctx, f"{step}-{job}", eps[job],
                                 embodiment="human" if job == "human" else "yam",
                                 stages=("segment", "caption"), budget=CAPS_USD[step],
                                 derived=derived)
        r = _run_job(ctx, step, cfg, job_id, determinism=ORACLE_DETERMINISM, usd_per_s=rate,
                     spent_in_step=spent)
        r["data"] = f"{step}-{job}"
        jobs[job] = r
        spent += float(r.get("spend") or 0.0)
        if r.get("state") not in ("done", "completed_with_errors"):
            break
    return _jobs_result(jobs, len(eps))


def run_rewards(ctx: Ctx) -> dict[str, Any]:
    cfg, job_id = _stage_job(ctx, "rewards", REWARD_EPISODES, embodiment="yam",
                             stages=("reward.terminal", "reward.dense"),
                             budget=CAPS_USD["rewards"],
                             resources={"reward.terminal": "{max_containers: 1, max_inputs: 2}",
                                        "reward.dense": "{max_containers: 1, max_inputs: 2}"})
    r = _run_job(ctx, "rewards", cfg, job_id, determinism="nvidia",
                 usd_per_s=pool_container("reward.terminal").usd_per_s(), spent_in_step=0.0)
    return {**_jobs_result({"rewards": r}, 1), "job": r}


def run_balanced(ctx: Ctx) -> dict[str, Any]:
    """(d): balanced through the real ``flora deploy`` / ``flora run`` on (b)'s jobs and
    episodes, then the lab's L0 + Gate A on 6 captured requests."""
    import modal

    from flora_deploy.modal_app import balanced_lab as lab

    derived = resolve_receipts(["captioner_bf16", "tempseg_merged"], require_pin=True)
    rate = (pool_container("segment", "balanced").usd_per_s()
            + pool_container("caption", "balanced").usd_per_s())
    jobs: dict[str, dict[str, Any]] = {}
    spent = 0.0
    eps = dict(h0.EPISODES)
    for job in sorted(eps, key=lambda j: -len(eps[j])):
        cfg, job_id = _stage_job(ctx, f"balanced-{job}", eps[job],
                                 embodiment="human" if job == "human" else "yam",
                                 stages=("segment", "caption"), budget=CAPS_USD["balanced"],
                                 preset="balanced", derived=derived)
        r = _run_job(ctx, "balanced", cfg, job_id, determinism=ORACLE_DETERMINISM,
                     usd_per_s=rate, spent_in_step=spent)
        jobs[job] = r
        spent += float(r.get("spend") or 0.0)
        if r.get("state") not in ("done", "completed_with_errors"):
            break
    e2e = _jobs_result(jobs, len(eps))

    ids = [i for ids in h0.EPISODES.values() for i in ids]
    out: dict[str, Any] = {"episodes": ids, "kinds": {}}
    seg_rel = ""
    rdir = ROOT / "reports" / "wave-h"
    rdir.mkdir(parents=True, exist_ok=True)
    ctx.note_app("flora-balanced-lab")
    with modal.enable_output(), lab.app.run():
        for kind, fn in (("tempseg", lab.run_tempseg), ("captioner", lab.run_captioner)):
            c = LAB[kind]
            limit = min(LAB_MAX_S, gpu_deadline_s(ctx.st, "balanced", c, spent_in_step=spent,
                                                  slack_s=LAB_SLACK_S))
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
    lab_ok = all("error" not in v or not v["error"] for v in out["kinds"].values()) and \
        len(out["kinds"]) == 2 and not any("skipped" in v for v in out["kinds"].values())
    return {"ok": lab_ok and e2e["ok"], **out, "jobs": jobs, "tracked_usd": spent,
            "report_error": e2e["report_error"],
            "apps": [*e2e["apps"], "flora-balanced-lab"]}


def _job_rows(ctx: Ctx, step: str) -> dict[str, dict[str, list[dict]]]:
    """``{episode_id: {"segment": rows, "caption": rows}}`` from a finished job step."""
    from flora_deploy import sdk

    rows: dict[str, dict[str, list[dict]]] = {}
    for r in ctx.st.step(step).result.get("jobs", {}).values():
        if "job_id" not in r:
            continue
        job = sdk.open_job(r["job_id"], store=f"modal-volume://{STORE_VOLUME}/{ctx.st.run_id}")
        for stage in ("segment", "caption"):
            for rec in job.results(stage):
                rows.setdefault(rec.episode_id, {})[stage] = [
                    {"start_s": s.start_s, "end_s": s.end_s, "caption": s.caption or ""}
                    for s in rec.segments]
    return rows


_exact_rows = _job_rows


def _lab_rows(kind: str) -> dict[str, list[dict]]:
    p = ROOT / "reports" / "wave-h" / f"balanced-{kind}.json"
    if not p.exists():
        return {}
    res = json.loads(p.read_text())
    return {h0.episode_id(e["id"]): e.get("rows", []) for e in
            res.get("baseline", {}).get("episodes", []) if e.get("ok")}


def run_certify(ctx: Ctx, noise: Mapping[str, float] | None = None) -> dict[str, Any]:
    exact = _job_rows(ctx, "exact_e2e")
    bal_rows = _job_rows(ctx, "balanced")  # the deployed balanced workers' records
    seg, cap = _lab_rows("tempseg"), _lab_rows("captioner")
    per = {}
    for eid, ex in exact.items():
        b = bal_rows.get(eid) or {}
        got = b.get("caption") or cap.get(eid) or b.get("segment") or seg.get(eid)
        if got is None:
            continue
        m = compare_segments(ex.get("caption") or ex.get("segment", []), got)
        if not (b.get("caption") or eid in cap):
            m["caption_jaccard"] = 0.0
        per[eid] = m
    bal = ctx.st.step("balanced").result.get("kinds", {})
    l0 = [r for k in ("tempseg", "captioner") for r in (bal.get(k, {}).get("l0") or [])]
    gate = max((bal.get(k, {}).get("gate_a") or {} for k in ("tempseg", "captioner")),
               key=lambda g: float(g.get("max_abs_dlogit") or 0.0), default={})
    verdict = certify(aggregate(per), l0=l0, gate_a=gate or None, noise=noise)
    cert = {"schema": "flora_provisional_certificate_v0", "preset": "balanced",
            "pools": {"segment": "L40S", "caption": "H100"},
            "reference": f"exact ({ORACLE_DETERMINISM} mode)",
            "verdict": verdict, "per_episode": per, "run_id": ctx.st.run_id}
    (ROOT / "reports" / "wave-h" / "certificate-provisional.json").write_text(
        json.dumps(cert, indent=1, default=str))
    return {"ok": True, "verdict": verdict, "episodes": len(per), "tracked_usd": 0.0,
            "apps": []}


def run_noise(ctx: Ctx) -> dict[str, Any]:
    """(f): (b)'s ``human`` job repeated: its own config file and staged data (same stage hash,
    so the same app), a new job id, default mode."""
    oracle = ctx.st.step("exact_e2e").result.get("jobs", {}).get(NOISE_JOB) or {}
    if not oracle.get("cfg"):
        raise RuntimeError(f"noise repeats (b)'s {NOISE_JOB} job, which has not run")
    cfg = Path(oracle["cfg"])
    job_id = f"wave-h-{ctx.st.run_id}-noise-{NOISE_JOB}"
    _index(ctx, cfg, job_id, f"noise-{NOISE_JOB}")
    r = _run_job(ctx, "noise", cfg, job_id, determinism=ORACLE_DETERMINISM,
                 usd_per_s=pool_container("segcap").usd_per_s(), spent_in_step=0.0)
    res = _jobs_result({NOISE_JOB: r}, 1)
    res["same_app_as_oracle"] = r.get("app") == oracle.get("app")
    ctx.st.step("noise").result = res  # so _job_rows can read this step's job
    exact = _job_rows(ctx, "exact_e2e")
    noise_rows = _job_rows(ctx, "noise")
    per = {eid: compare_segments(exact[eid].get("caption", []), rows.get("caption", []))
           for eid, rows in noise_rows.items() if eid in exact}
    floor = aggregate(per)
    res["noise_floor"] = floor
    if ctx.st.step("certify").status == "done":
        res["certify_with_noise"] = run_certify(ctx, noise=floor)["verdict"]
    return res


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
             "global cap (every attempt's tracked container-seconds x list $/h, a killed or",
             "unreported attempt at its budget, or the billed delta of every app from",
             "`scripts/spend.py` when larger; billing lags about an hour).", "",
             "| step | status | cap | projected (upper) | attempts | tracked | billed |",
             "|---|---|---|---|---|---|---|"]
    for name in STEPS:
        s = st.steps.get(name, StepState())
        p = projections.get(name)
        proj = f"${p.usd:.2f} (${p.usd_upper:.2f})" if p else "-"
        billed = f"${s.billed_usd:.2f}" if s.billed_usd is not None else "-"
        lines.append(f"| {TITLES[name]} | {s.status} | ${CAPS_USD[name]:.2f} | {proj} | "
                     f"{len(s.attempts)} | ${s.tracked_usd:.2f} | {billed} |")
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


_PRIVATE_KEYS = {"rows", "caption", "captions", "text", "task", "facts", "episodes", "cfg"}


def _numbers_only(d: Any) -> Any:
    """Drop text that could carry private episode content (captions, tasks, rows)."""
    if isinstance(d, Mapping):
        return {k: _numbers_only(v) for k, v in d.items() if k not in _PRIVATE_KEYS}
    if isinstance(d, list):
        return [_numbers_only(x) for x in d]
    return d


# ---------------------------------------------------------------------------------------------


def print_plan(st: State, projections: Mapping[str, Projection], out=print) -> None:
    out(f"Wave H-lite run {st.run_id}: spent ${st.spent:.2f} of ${GLOBAL_CAP_USD:.2f} "
        f"(billing read {'never' if st.billed_at is None else time.ctime(st.billed_at)})")
    for name in STEPS:
        p = projections[name]
        s = st.steps.get(name, StepState())
        rule = "ok"
        try:
            check_before(st, name, p)
        except StopRule as e:
            rule = f"STOP: {e}"
        out(f"  {TITLES[name]:66s} [{s.status}] cap ${CAPS_USD[name]:.2f}  nominal "
            f"${p.usd:.2f}  upper ${p.usd_upper:.2f}  rule: {rule}")
        for i in p.items:
            out(f"      {i['what']:52s} {i['gpu']:>5s} {i['seconds']:6.0f} s / "
                f"{i['upper_seconds']:6.0f} s x ${i['usd_per_h']:.2f}/h = ${i['usd']:.3f} "
                f"(upper ${i['usd_upper']:.3f})")
    nominal = sum(p.usd for p in projections.values())
    upper = sum(p.usd_upper for p in projections.values())
    out(f"  total: nominal ${nominal:.2f}, upper ${upper:.2f} (incl. {CLIENT_SLACK_S} s client "
        f"slack per app); global cap ${GLOBAL_CAP_USD:.2f}")
    honest = honest_caps(projections)
    if upper > GLOBAL_CAP_USD + 1e-9:
        out(f"  NOT HONEST AT ${GLOBAL_CAP_USD:.2f}: the upper total needs a global cap of at "
            f"least ${honest['global']:.2f} (per step: "
            + ", ".join(f"{s} ${honest[s]:.2f}" for s in STEPS) + "); the runbook stops at the "
            "first step whose upper projection no longer fits the cap left")


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
    killed_apps = st.mark_killed() if args.yes_spend else []
    manifest = h0.golden_manifest()
    projections = {s: project(s, manifest) for s in STEPS}
    print_plan(st, projections)
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
    for app in killed_apps:  # a killed run may have left them up
        subprocess.run(["modal", "app", "stop", "--yes", app], capture_output=True, text=True)
        print(f"stopped {app} (left by a killed run)")
    save_state(st, args.state)
    os.chdir(args.work)  # relative input.path is hashed against the cwd: plan where flora runs
    ctx = Ctx(st, manifest, args.work, state_path=args.state)
    rc = 0
    for name in STEPS:
        if args.only and name != args.only:
            continue
        s = st.step(name)
        if s.status == "done":
            print(f"skip {name}: done")
            continue
        refresh_billing(st)  # every app seen so far, before the step (H3)
        p = projections[name]
        s.projected_usd, s.projected_upper_usd = p.usd, p.usd_upper
        try:
            check_before(st, name, p)
        except StopRule as e:
            s.status, rc = "stopped", 3
            s.notes.append(f"stop rule before: {e}")
            break
        att = Attempt(started=time.time(), rate_usd_per_s=p.peak_usd_per_s,
                      budget_usd=round(step_budget(st, name), 4))
        s.attempts.append(att)
        s.status = "running"
        ctx.step = name
        save_state(st, args.state)
        try:
            with Heartbeat(st, att, args.state):
                res = RUNNERS[name](ctx)
            s.result = res
            att.tracked_usd = round(float(res.get("tracked_usd") or 0.0), 4)
            att.report_error = bool(res.get("report_error"))
            att.status = s.status = "done" if res.get("ok") else "failed"
        except Exception as e:  # noqa: BLE001 - recorded; the runbook stops
            att.status = s.status = "failed"
            att.report_error = True  # what it spent is unknown: charged its budget
            s.notes.append(f"error: {e!r}")
        att.finished = time.time()
        for a in s.result.get("apps", ()):
            if a not in att.apps:
                att.apps.append(a)
        refresh_billing(st)
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
