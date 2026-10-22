"""The spend-guarded runbook core shared by ``scripts/wave_h.py``, ``scripts/scale_test.py`` and
``scripts/rubric_smoke.py`` (``docs/wave-h-runbook.md``, "Spend accounting" and "Stop rules").

It holds what the runbooks have in common and nothing about their steps: the resumable state
(:class:`Attempt`, :class:`StepState`, :class:`State`, :func:`load_state` / :func:`save_state`),
the billing reads (``scripts/spend.py``), the heartbeat that leaves a killed process's spend
behind, and the stop rules. ``wave_h.py`` re-exports these names, so ``wave_h_finalize.py`` and
older callers keep using ``wave_h.State`` and friends.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BILLING_LAG_S = 3600  # Modal billing covers complete hours only
HEARTBEAT_S = 30.0
CLIENT_SLACK_S = 240  # per app: client timeout margin, then cancel + app stop
# A state from before the cap override was scoped to a plan: the plan it was set on.
LEGACY_CAP_PLAN = "balanced-only"


# ---------------------------------------------------------------------------------------------
# State


@dataclass
class Attempt:
    """One run of a step (``docs/wave-h-runbook.md``, "Spend accounting")."""

    started: float
    status: str = "running"  # running | done | failed | stopped | killed | needs_pin
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
    status: str = "pending"  # pending | running | done | failed | stopped | needs_pin
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
    cap_override: float | None = None  # a global cap the user approved (--global-cap)
    # the plan that cap was approved for: it never carries over to another plan. ``None`` in a
    # state from before the scope was saved: the default plan, the only one it was set on
    cap_override_plan: str | None = None
    # the global cap the last --yes-spend run gated its steps on, and its plan (what
    # wave_h_finalize.py reports; review 12 F6)
    gate_cap: float | None = None
    gate_cap_plan: str | None = None
    plan: str | None = None  # the plan this state was run under (wave_h_finalize.py reads it)

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

    def override_for(self, plan: str) -> float | None:
        """The approved global cap override, if it was approved for ``plan``."""
        if self.cap_override is None or (self.cap_override_plan or LEGACY_CAP_PLAN) != plan:
            return None
        return self.cap_override

    def spent_outside(self, step: str) -> float:
        """Cumulative spend minus ``step``'s running attempt (in-step deadlines add their own)."""
        cur = self.step(step).current
        own = cur.charged(self.billed_at) if cur is not None else 0.0
        return round(max(0.0, self.spent - own), 4)

    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "created": self.created,
                "billing_since": self.billing_since, "billing_baseline": self.billing_baseline,
                "billed_total": self.billed_total, "billed_at": self.billed_at,
                "cap_override": self.cap_override, "cap_override_plan": self.cap_override_plan,
                "gate_cap": self.gate_cap, "gate_cap_plan": self.gate_cap_plan,
                "plan": self.plan,
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
                   d.get("billed_at"), d.get("cap_override"), d.get("cap_override_plan"),
                   d.get("gate_cap"), d.get("gate_cap_plan"), d.get("plan"))

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


def new_state() -> State:
    import uuid

    return State(run_id=time.strftime("%m%d%H%M") + uuid.uuid4().hex[:4], created=time.time(),
                 billing_since=time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400)))


def load_state(path: Path) -> State:
    if path.exists():
        return State.from_json(json.loads(path.read_text()))
    return new_state()


_SAVE_LOCK = threading.RLock()


def save_state(st: State, path: Path) -> None:
    with _SAVE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.to_json(), indent=1, sort_keys=True, default=str))
        os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# Stop rules


class StopRule(RuntimeError):
    """The runbook stops here (projected or actual spend over a cap)."""


def check_before(st: State, step: str, proj: Any, *, caps: Mapping[str, float],
                 global_cap: float) -> None:
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


def check_after(st: State, step: str, *, caps: Mapping[str, float], global_cap: float) -> None:
    s = st.step(step)
    spent = s.spent_at(st.billed_at)
    if spent > caps[step] + 1e-9:
        raise StopRule(f"{step}: spent ${spent:.2f} (tracked ${s.tracked_usd:.2f}, billed "
                       f"${s.billed_usd or 0:.2f}) is over its cap ${caps[step]:.2f}")
    if st.spent > global_cap + 1e-9:
        raise StopRule(f"spent ${st.spent:.2f} in all is over the global cap ${global_cap:.2f}")


def step_budget(st: State, step: str, *, spent_in_step: float = 0.0, caps: Mapping[str, float],
                global_cap: float) -> float:
    """What the step may still spend: its cap minus its earlier attempts, and the global cap
    left, minus what this attempt already spent."""
    s = st.step(step)
    cur = s.current
    earlier = s.spent_at(st.billed_at) - (cur.charged(st.billed_at) if cur else 0.0)
    return max(0.0, min(caps[step] - earlier, global_cap - st.spent_outside(step))
               - spent_in_step)


def gpu_deadline_s(st: State, step: str, c: Any, *, spent_in_step: float = 0.0,
                   slack_s: float = CLIENT_SLACK_S, caps: Mapping[str, float],
                   global_cap: float) -> float:
    """Wall seconds still affordable at ``c``'s $/s (anything with ``usd_per_s()``, or a $/s
    figure) in this step, keeping ``slack_s`` of it for the cancel / stop that follows the
    deadline (H2)."""
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


def billed_delta(st: State, apps: Iterable[str], now: Mapping[str, float]) -> float:
    return round(sum(now.get(a, 0.0) - st.billing_baseline.get(a, 0.0) for a in set(apps)), 4)


def refresh_billing(st: State, *, now: Mapping[str, float],
                    clock: Callable[[], float] = time.time) -> bool:
    """Apply a billing read (``now``: billed cost per app) to every app seen so far (H3): the
    global billed total, and per step the billed delta of the apps only that step used (shared
    apps count globally only)."""
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
