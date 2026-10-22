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
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BILLING_LAG_S = 3600  # a state without a billing watermark: billing is this far behind its read
HOUR_S = 3600.0  # Modal bills by the hour: a row's interval_start is the start of its hour
# How long after an hour ends its billing rows are expected (only for "clears at" times).
BILLING_SETTLE_S = 15 * 60
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
    app_ids: list[str] = field(default_factory=list)  # Modal app ids (``ap-...``) it started
    billed_usd: float | None = None  # billed cost of its own apps (None: not attributable)
    # when every app it started was confirmed stopped (the ``with app.run()`` block exited, or a
    # stop by id returned 0); None: not known to be stopped
    stopped_at: float | None = None
    slack_s: float | None = None  # stop -> container gone (None: CLIENT_SLACK_S)
    hard_timeout_s: float | None = None  # its containers cannot outlive started + this

    @property
    def last_activity(self) -> float:
        return self.finished or self.heartbeat or self.started

    @property
    def slack(self) -> float:
        return float(CLIENT_SLACK_S if self.slack_s is None else self.slack_s)

    def _run_end(self) -> float | None:
        """The latest its containers can have run until (before the slack); None: unknown."""
        cut = None if self.hard_timeout_s is None else self.started + self.hard_timeout_s
        if self.stopped_at is not None:
            return self.stopped_at if cut is None else min(self.stopped_at, cut)
        return cut if self.status == "killed" else None

    @property
    def settle_point(self) -> float:
        """The latest time this attempt's apps can have been billed for: its known end plus the
        slack, else its last activity."""
        end = self._run_end()
        return self.last_activity if end is None else max(self.last_activity, end + self.slack)

    def bounded_usd(self) -> float | None:
        """The most a killed or unreported attempt can have cost when its end is known (a
        confirmed stop, or its containers' hard timeout): its peak rate over the time it ran
        plus the slack, at most its budget. None: nothing is known (charged its budget)."""
        end = self._run_end()
        if end is None:
            return None
        ran = max(self.tracked_usd, self.rate_usd_per_s * max(0.0, end - self.started))
        return round(min(self.budget_usd, ran + self.slack * self.rate_usd_per_s), 4)

    def held(self, through: float | None) -> bool:
        """Charged more than tracked until billing covers it (killed or unreported)."""
        return (self.status == "killed" or self.report_error) and not self.settled(through)

    def hold_clears_at(self) -> float:
        """When billing is expected to cover this attempt: the end of the billing hour holding
        its :attr:`settle_point`, plus :data:`BILLING_SETTLE_S`."""
        return hour_floor(self.settle_point) + HOUR_S + BILLING_SETTLE_S

    def settled(self, through: float | None) -> bool:
        """Billing covers every hour this attempt could have spent in (``through``: the billing
        watermark, :attr:`State.watermark`)."""
        return through is not None and through >= self.settle_point

    def charged(self, through: float | None) -> float:
        """Settled: the larger of tracked and its own apps' billed cost. Until then: tracked;
        a killed or unreported attempt its bound (:meth:`bounded_usd`) when its end is known,
        else its budget."""
        if self.settled(through):
            return max(self.tracked_usd, self.billed_usd or 0.0)
        if self.status == "killed" or self.report_error:
            bound = self.bounded_usd()
            return max(self.tracked_usd, self.budget_usd if bound is None else bound)
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
    # the billing watermark: billing covers every hour before this (the latest billed hour's
    # end, from the rows' interval_start). None: never read with rows (an older state)
    billed_through: float | None = None
    cap_override_by: str | None = None  # who approved cap_override (``--yes-spend``), and when
    cap_override_at: float | None = None

    @property
    def watermark(self) -> float | None:
        """Up to when billing is complete: :attr:`billed_through`, or for a state read before
        it was kept, :data:`BILLING_LAG_S` before the last read."""
        if self.billed_through is not None:
            return self.billed_through
        return None if self.billed_at is None else self.billed_at - BILLING_LAG_S

    def step(self, name: str) -> StepState:
        return self.steps.setdefault(name, StepState())

    @property
    def apps_seen(self) -> list[str]:
        return sorted({a for s in self.steps.values() for a in s.apps})

    @property
    def spent(self) -> float:
        """The cumulative figure the global cap is checked on (module docstring)."""
        tracked = sum(s.spent_at(self.watermark) for s in self.steps.values())
        return round(max(tracked, self.billed_total or 0.0), 4)

    def override_for(self, plan: str) -> float | None:
        """The approved global cap override, if it was approved for ``plan``."""
        if self.cap_override is None or (self.cap_override_plan or LEGACY_CAP_PLAN) != plan:
            return None
        return self.cap_override

    def spent_outside(self, step: str) -> float:
        """Cumulative spend minus ``step``'s running attempt (in-step deadlines add their own)."""
        cur = self.step(step).current
        own = cur.charged(self.watermark) if cur is not None else 0.0
        return round(max(0.0, self.spent - own), 4)

    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "created": self.created,
                "billing_since": self.billing_since, "billing_baseline": self.billing_baseline,
                "billed_total": self.billed_total, "billed_at": self.billed_at,
                "cap_override": self.cap_override, "cap_override_plan": self.cap_override_plan,
                "gate_cap": self.gate_cap, "gate_cap_plan": self.gate_cap_plan,
                "plan": self.plan, "billed_through": self.billed_through,
                "cap_override_by": self.cap_override_by,
                "cap_override_at": self.cap_override_at,
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
                   d.get("gate_cap"), d.get("gate_cap_plan"), d.get("plan"),
                   billed_through=d.get("billed_through"),
                   cap_override_by=d.get("cap_override_by"),
                   cap_override_at=d.get("cap_override_at"))

    def held(self) -> list[tuple[str, int, Attempt]]:
        """``(step, attempt number, attempt)`` of every attempt charged over its tracked cost
        until billing covers it."""
        return [(name, i, a) for name, s in self.steps.items()
                for i, a in enumerate(s.attempts, 1) if a.held(self.watermark)
                and a.charged(self.watermark) > a.tracked_usd + 1e-9]

    def killed_unstopped(self) -> list[Attempt]:
        """Killed attempts whose apps are not known to be stopped."""
        return [a for s in self.steps.values() for a in s.attempts
                if a.status == "killed" and a.stopped_at is None and (a.apps or a.app_ids)]

    def mark_killed(self, stale_before: float | None = None) -> list[str]:
        """Attempts left ``running`` by a killed process: ``killed`` (charged their bound, or
        their budget, until billing covers them), their step ``failed`` (retried). Returns
        their apps. ``stale_before`` (a preview): only attempts silent since before then."""
        apps: list[str] = []
        for s in self.steps.values():
            hit = False
            for a in s.attempts:
                if a.status == "running" and (stale_before is None
                                              or a.last_activity < stale_before):
                    a.status, hit = "killed", True
                    a.notes.append("process killed while running; charged its bound (or its "
                                   "budget) until billing covers it")
                    apps += a.apps
            if s.status == "running" and (hit or stale_before is None):
                s.status = "failed"
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


@dataclass(frozen=True)
class Budget:
    """What one attempt of a step may spend, and which term binds it (``binding``: ``"attempt
    cap"``, ``"global left"`` or ``"reserve for <steps>"``). ``terms``: every term, in $."""

    usd: float
    binding: str
    terms: dict[str, float]
    reserved_for: tuple[str, ...] = ()


def reserve_for(st: State, steps: Sequence[str], step: str,
                projections: Mapping[str, Any] | None,
                caps: Mapping[str, float] | None = None) -> dict[str, float]:
    """The upper projections of the plan's steps after ``step`` that are not done yet (each at
    most its attempt cap: a step over its cap never runs), what the global cap keeps for
    them."""
    if not projections or step not in steps:
        return {}
    out = {}
    for s in steps[list(steps).index(step) + 1:]:
        p = projections.get(s)
        if p is None or (st.steps.get(s) or StepState()).status == "done":
            continue
        up = float(p.usd_upper) if caps is None or s not in caps else min(float(p.usd_upper),
                                                                          float(caps[s]))
        if up > 0:
            out[s] = round(up, 4)
    return out


def budget_for(st: State, step: str, *, caps: Mapping[str, float], global_cap: float,
               reserve: Mapping[str, float] | None = None, spent_in_step: float = 0.0
               ) -> Budget:
    """The one budget rule (the stop rule before a step, the attempt's budget and its in-step
    deadlines all use it): ``min(attempt cap, global cap - spent outside this attempt -
    reserve for the plan's later steps) - spent_in_step``. ``caps[step]`` is a per-attempt
    ceiling: earlier attempts count against the global cap, not against it."""
    cap = float(caps[step])
    spent = st.spent_outside(step)
    left = global_cap - spent
    kept = round(sum((reserve or {}).values()), 4)
    allowance = left - kept
    terms = {"attempt cap": cap, "global cap": global_cap, "spent": spent, "global left": left,
             "reserve": kept, "allowance": allowance}
    if cap <= allowance + 1e-9:
        binding, usd = "attempt cap", cap
    elif kept > 0 and cap <= left + 1e-9:
        binding, usd = f"reserve for {', '.join(reserve or {})}", allowance
    else:
        binding, usd = "global left", allowance
    return Budget(round(max(0.0, usd - spent_in_step), 4), binding, terms,
                  tuple(reserve or {}))


def _held_note(st: State) -> str:
    held = st.held()
    if not held:
        return ""
    return "; held until billing covers them: " + ", ".join(
        f"{n} attempt {i} ${a.charged(st.watermark):.2f} (tracked ${a.tracked_usd:.2f}, about "
        f"{utc(a.hold_clears_at())})" for n, i, a in held)


def over_budget(step: str, proj: Any, b: Budget, st: State | None = None) -> str:
    """Why ``proj``'s upper projection does not fit ``b``, naming the binding term."""
    t, up = b.terms, proj.usd_upper
    if b.binding == "attempt cap":
        why = (f"{step}: upper projection ${up:.2f} (nominal ${proj.usd:.2f}) is over its cap "
               f"${t['attempt cap']:.2f} (binding: attempt cap)")
    elif b.binding == "global left":
        why = (f"{step}: upper projection ${up:.2f} is over the global budget left "
               f"(${t['global left']:.2f} of ${t['global cap']:.2f}, ${t['spent']:.2f} spent"
               + (f", ${t['reserve']:.2f} kept for {', '.join(b.reserved_for)}"
                  if t["reserve"] > 0 else "") + "; binding: global left)")
    else:
        why = (f"{step}: upper projection ${up:.2f} is over the global budget left after the "
               f"reserve (${t['allowance']:.2f}: ${t['global left']:.2f} of "
               f"${t['global cap']:.2f} left, ${t['spent']:.2f} spent, ${t['reserve']:.2f} kept "
               f"for {', '.join(b.reserved_for)}; binding: {b.binding})")
    return why + (_held_note(st) if st is not None else "")


def check_before(st: State, step: str, proj: Any, *, caps: Mapping[str, float],
                 global_cap: float, reserve: Mapping[str, float] | None = None) -> Budget:
    """The **upper** projection must fit the step's budget (:func:`budget_for`; H2). Returns
    the budget the attempt gets."""
    b = budget_for(st, step, caps=caps, global_cap=global_cap, reserve=reserve)
    if proj.usd_upper > b.usd + 1e-9:
        raise StopRule(over_budget(step, proj, b, st))
    return b


def attempt_spent(st: State, step: str) -> float:
    """What the step's latest attempt spent: its charge, or the step's billed cost over what its
    earlier attempts were charged when that is larger."""
    s = st.step(step)
    if not s.attempts:
        return float(s.billed_usd or 0.0)
    earlier = sum(a.charged(st.watermark) for a in s.attempts[:-1])
    return max(s.attempts[-1].charged(st.watermark), (s.billed_usd or 0.0) - earlier)


def check_after(st: State, step: str, *, caps: Mapping[str, float], global_cap: float) -> None:
    """After an attempt: what it spent must fit the per-attempt cap, and the cumulative spend
    the global cap."""
    s = st.step(step)
    spent = attempt_spent(st, step)
    if spent > caps[step] + 1e-9:
        raise StopRule(f"{step}: attempt {max(1, len(s.attempts))} spent ${spent:.2f} (tracked "
                       f"${s.attempts[-1].tracked_usd if s.attempts else 0:.2f}, step billed "
                       f"${s.billed_usd or 0:.2f}) is over its cap ${caps[step]:.2f}")
    if st.spent > global_cap + 1e-9:
        raise StopRule(f"spent ${st.spent:.2f} in all is over the global cap ${global_cap:.2f}")


def step_budget(st: State, step: str, *, spent_in_step: float = 0.0, caps: Mapping[str, float],
                global_cap: float, reserve: Mapping[str, float] | None = None) -> float:
    """What the running attempt may still spend (:func:`budget_for`)."""
    return budget_for(st, step, caps=caps, global_cap=global_cap, reserve=reserve,
                      spent_in_step=spent_in_step).usd


def gpu_deadline_s(st: State, step: str, c: Any, *, spent_in_step: float = 0.0,
                   slack_s: float = CLIENT_SLACK_S, caps: Mapping[str, float],
                   global_cap: float, reserve: Mapping[str, float] | None = None) -> float:
    """Wall seconds still affordable at ``c``'s $/s (anything with ``usd_per_s()``, or a $/s
    figure) in this step, keeping ``slack_s`` of it for the cancel / stop that follows the
    deadline (H2)."""
    rate = c if isinstance(c, float | int) else c.usd_per_s()
    left = step_budget(st, step, spent_in_step=spent_in_step, caps=caps, global_cap=global_cap,
                       reserve=reserve)
    return max(0.0, left - slack_s * rate) / rate


@dataclass
class Decision:
    """What the runbook does next with one step: ``run`` (with ``budget``), ``done`` (skip),
    ``stop`` (the stop rule refuses it) or ``wait`` (refused only while held attempts are
    charged over their tracked cost; it fits once billing covers them, about ``clears_at``)."""

    step: str
    action: str
    budget: Budget | None = None
    message: str = ""
    reasons: list[str] = field(default_factory=list)
    clears_at: float | None = None
    command: str | None = None

    @property
    def runs(self) -> bool:
        return self.action == "run"


def decide(st: State, steps: Sequence[str], step: str, projections: Mapping[str, Any], *,
           caps: Mapping[str, float], global_cap: float, redo: bool = False) -> Decision:
    """The one decision the dry run, ``--status`` and a real run all take for ``step`` (of the
    plan's ``steps``) on the state as it is. ``redo``: a done step that runs again (scale
    test's redeploy)."""
    s = st.steps.get(step) or StepState()
    if s.status == "done" and not redo:
        return Decision(step, "done", message=f"{step}: done")
    proj = projections[step]
    b = budget_for(st, step, caps=caps, global_cap=global_cap,
                   reserve=reserve_for(st, steps, step, projections, caps))
    reasons = [f"{n} attempt {i} charged ${a.charged(st.watermark):.2f} (tracked "
               f"${a.tracked_usd:.2f}): {'killed' if a.status == 'killed' else 'unreported'}, "
               f"held until billing covers it, about {utc(a.hold_clears_at())}"
               for n, i, a in st.held()]
    if proj.usd_upper <= b.usd + 1e-9:
        return Decision(step, "run", b, f"{step}: runs with ${b.usd:.2f} (binding: {b.binding})",
                        reasons)
    msg = over_budget(step, proj, b, st)
    excess = sum(a.charged(st.watermark) - a.tracked_usd for _, _, a in st.held())
    if b.binding != "attempt cap" and excess > 0 and \
            proj.usd_upper <= min(b.terms["attempt cap"], b.usd + excess) + 1e-9:
        when = max(a.hold_clears_at() for _, _, a in st.held())
        return Decision(step, "wait", b, msg + f"; it fits once billing covers the held "
                        f"attempts, about {utc(when)}", reasons, when)
    return Decision(step, "stop", b, msg, reasons)


# ---------------------------------------------------------------------------------------------
# Billing (scripts/spend.py)


class Billing(dict):
    """One billing read: billed cost per app description (a plain mapping, as it always was),
    plus the hourly rows it came from (``(object_id, description, interval_start, cost)``) and
    the watermark :attr:`through` (the latest billed hour's end: every hour before it is
    complete; None without rows)."""

    def __init__(self, by_name: Mapping[str, float] = (), rows: Iterable[tuple] = (),
                 through: float | None = None):
        super().__init__(by_name)
        self.rows = list(rows)
        self.through = through

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> Billing:
        by: dict[str, float] = {}
        norm = []
        for r in rows:
            name = r.get("description") or "?"
            cost = float(r.get("cost") or 0.0)
            by[name] = by.get(name, 0.0) + cost
            norm.append((r.get("object_id") or "", name, r.get("interval_start"), cost))
        starts = [t for _, _, t, _ in norm if t is not None]
        return cls(by, norm, max(starts) + HOUR_S if starts else None)


def billed_by_app(since: str) -> Billing:
    import spend  # scripts/spend.py

    return Billing.from_rows(spend.rows(since))


def hour_floor(t: float) -> float:
    return (t // HOUR_S) * HOUR_S


def utc(t: float | None) -> str:
    return "never" if t is None else time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t))


# ---------------------------------------------------------------------------------------------
# Stopping apps by id (``modal app stop <name>`` only finds deployed apps, never ephemeral ones)


def _run(args: list[str], run: Callable[..., Any] | None) -> Any:
    try:
        return (run or subprocess.run)(args, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None


def list_apps(run: Callable[..., Any] | None = None) -> list[dict[str, Any]] | None:
    """``modal app list --json`` (``app_id``, ``description``, ``state``, ...); None when it
    cannot be read."""
    p = _run(["modal", "app", "list", "--json"], run)
    if getattr(p, "returncode", 1) != 0:
        return None
    try:
        rows = json.loads(p.stdout)
    except (TypeError, ValueError):
        return None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else None


def app_ids_named(name: str, apps: Iterable[Mapping[str, Any]]) -> list[str]:
    """The ids of the listed apps whose description is ``name``."""
    return [str(r["app_id"]) for r in apps if r.get("description") == name and r.get("app_id")]


def _live(r: Mapping[str, Any]) -> bool:
    return "stopped" not in str(r.get("state") or "").lower()


def stop_app(ident: str, run: Callable[..., Any] | None = None) -> bool:
    """``modal app stop --yes <id or deployed name>``; True when it returned 0."""
    return getattr(_run(["modal", "app", "stop", "--yes", ident], run), "returncode", 1) == 0


def stop_apps(names: Iterable[str], ids: Iterable[str] = (),
              run: Callable[..., Any] | None = None) -> bool:
    """Stop an attempt's apps by id: its recorded ids, and the live listed apps named like its
    apps (an ephemeral app is only found by id). Returns whether every one is confirmed
    stopped (listed as stopped, or stopped with exit code 0). When the app list cannot be read,
    each id and each name (a deployed app's) is stopped anyway, unconfirmed."""
    names, ids = list(dict.fromkeys(names)), list(dict.fromkeys(ids))
    listed = list_apps(run)
    if listed is None:
        for ident in [*ids, *names]:
            stop_app(ident, run)
        return False
    live = {str(r.get("app_id")): r for r in listed if _live(r)}
    targets = [i for i in ids if i in live]
    targets += [i for n in names for i in app_ids_named(n, live.values()) if i not in targets]
    return all([stop_app(i, run) for i in targets])


def _attempt_keys(a: Attempt) -> set[str]:
    return set(a.app_ids) | set(a.apps)


def _attempt_hours(a: Attempt) -> tuple[float, float]:
    """The billed hours [first, last] an attempt can have spent in (hour starts)."""
    return hour_floor(a.started), hour_floor(a.settle_point)


def attribute_rows(st: State, rows: Iterable[tuple]) -> dict[int, float]:
    """Per attempt (keyed by ``id()``) that started an app: the billed cost of its apps over the
    hours it ran, from the hourly ``rows``. Each row goes to one attempt: of those with that
    app (by id or name) active in that hour, the one that started last. An app two attempts
    share (scale_test's deploy and run, a legacy shared name) is then reconciled apart:
    the earlier attempt keeps its tracked cost, and the later one is charged the shared hour.
    That over-charges the later one, which errs on the safe side."""
    atts = [a for s in st.steps.values() for a in s.attempts if _attempt_keys(a)]
    got = {id(a): 0.0 for a in atts}
    for oid, name, t, cost in rows:
        if t is None:
            continue
        cands = [a for a in atts if (oid in a.app_ids or name in a.apps)
                 and _attempt_hours(a)[0] <= t <= _attempt_hours(a)[1]]
        if cands:
            got[id(max(cands, key=lambda a: a.started))] += cost
    return got


def billed_delta(st: State, apps: Iterable[str], now: Mapping[str, float]) -> float:
    return round(sum(now.get(a, 0.0) - st.billing_baseline.get(a, 0.0) for a in set(apps)), 4)


def refresh_billing(st: State, *, now: Mapping[str, float],
                    clock: Callable[[], float] = time.time) -> bool:
    """Apply a billing read (``now``: billed cost per app; a :class:`Billing` also carries its
    rows and watermark) to every app seen so far (H3): the global billed total, per step the
    billed delta of the apps only that step used (shared apps count globally only), and with
    rows, per attempt the billed cost of its own apps (:func:`attribute_rows`) and the state's
    watermark (:attr:`State.billed_through`)."""
    st.billed_total = billed_delta(st, st.apps_seen, now)
    owners: dict[str, set[str]] = {}
    for name, s in st.steps.items():
        for a in s.apps:
            owners.setdefault(a, set()).add(name)
    for name, s in st.steps.items():
        mine = [a for a in s.apps if owners.get(a) == {name}]
        s.billed_usd = billed_delta(st, mine, now) if mine else s.billed_usd
    if isinstance(now, Billing):
        got = attribute_rows(st, now.rows)
        for s in st.steps.values():
            for a in s.attempts:
                a.billed_usd = (round(got[id(a)], 4) if id(a) in got and a.status != "running"
                                else None)
        if now.through is not None:
            st.billed_through = max(now.through, st.billed_through or now.through)
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


def stop_killed(st: State, run: Callable[..., Any] | None = None,
                out: Callable[[str], None] = print) -> None:
    """Stop, by id, the apps a killed run may have left up (``modal app stop <name>`` finds no
    ephemeral app). A confirmed stop bounds the attempt's charge (``Attempt.bounded_usd``)."""
    for a in st.killed_unstopped():
        if stop_apps(a.apps, a.app_ids, run=run):
            a.stopped_at = time.time()
            out(f"stopped {', '.join(a.app_ids or a.apps)} (left by a killed run)")
        else:
            out(f"could not confirm {', '.join(a.app_ids or a.apps)} stopped (left by a killed "
                "run): it stays charged its budget until billing covers it")


# ---------------------------------------------------------------------------------------------
# The step loop (wave_h.py, scale_test.py and rubric_smoke.py run their steps through it)


def note_stop(s: StepState, text: str) -> None:
    """A step's stop-rule note replaces the one before it (the notes never go stale)."""
    s.notes = [n for n in s.notes if not n.startswith("stop rule")] + [text]


def charge_unknown_if_started(name: str, att: Attempt, exc: BaseException) -> None:
    """The default for a runner that raised: unknown spend (charged its bound or budget) if it
    noted an app, else nothing started and it is charged $0 (review 10 L1)."""
    if att.apps or att.app_ids:
        att.report_error = True
    else:
        att.tracked_usd = 0.0
        att.notes.append("failed before any app started: charged $0")


def _status_of(res: Mapping[str, Any]) -> str:
    return "done" if res.get("ok") else "failed"


def run_steps(st: State, steps: Iterable[str], *, state_path: Path, ctx: Any,
              run: Callable[[str, Any], Mapping[str, Any]], projection: Callable[[str], Any],
              decide: Callable[[str, Any], Decision], check_after: Callable[[str], None],
              refresh: Callable[[State], Any] | None = None,
              skip: Callable[[str, StepState], str | None] | None = None,
              refuse: Callable[[str, StepState], str | None] | None = None,
              attempt_kw: Callable[[str], Mapping[str, Any]] | None = None,
              status_of: Callable[[Mapping[str, Any]], str] = _status_of,
              on_error: Callable[[str, Attempt, BaseException], None] = (
                  charge_unknown_if_started),
              after: Callable[[str, StepState], int | None] | None = None,
              keep_going: Callable[[str], bool] | None = None,
              out: Callable[[str], None] = print) -> int:
    """Run ``steps`` in order, each under the same rules: skip a done step (``skip`` says why
    another is skipped), re-read billing (``refresh``), take :func:`decide`'s verdict (a stop
    or wait ends the run, exit 3), start an attempt with the decision's budget and a heartbeat,
    record its result (``status_of``) or its error (``on_error``), re-read billing, then the
    stop rule after it (``check_after``). ``refuse``: a reason to refuse a step after its
    decision (exit 1); ``after``: an exit code that ends the run after a step (e.g. a pin);
    ``keep_going``: steps whose failure does not end the run. Returns the exit code."""
    rc = 0
    for name in steps:
        s = st.step(name)
        why = skip(name, s) if skip else ("done" if s.status == "done" else None)
        if why:
            out(f"skip {name}: {why}")
            continue
        if refresh is not None:
            refresh(st)  # every app seen so far, before the step (H3)
        p = projection(name)
        s.projected_usd, s.projected_upper_usd = p.usd, p.usd_upper
        d = decide(name, p)
        if not d.runs:
            s.status, rc = "stopped", 3
            note_stop(s, f"stop rule before: {d.message}")
            out(f"{d.action} {name}: {d.message}" + (f"\n  next: {d.command}" if d.command
                                                     else ""))
            break
        if refuse is not None and (no := refuse(name, s)):
            s.notes.append(f"refused: {no}")
            out(f"refuse {name}: {no}")
            rc = 1
            break
        att = Attempt(started=time.time(), rate_usd_per_s=p.peak_usd_per_s,
                      budget_usd=d.budget.usd, **(attempt_kw(name) if attempt_kw else {}))
        s.attempts.append(att)
        s.status = "running"
        ctx.step = name
        save_state(st, state_path)
        try:
            with Heartbeat(st, att, state_path):
                res = run(name, ctx)
            s.result = dict(res)
            att.tracked_usd = round(float(res.get("tracked_usd") or 0.0), 4)
            att.report_error = bool(res.get("report_error"))
            att.status = s.status = status_of(res)
        except StopRule as e:  # an in-step stop rule raised before any app started
            att.status = s.status = "stopped"
            att.tracked_usd = 0.0
            att.notes.append(f"stop rule: {e}")
            note_stop(s, f"stop rule before: {e}")
            rc = 3
        except Exception as e:  # noqa: BLE001 - recorded; the runbook stops
            att.status = s.status = "failed"
            on_error(name, att, e)
            att.notes.append(f"error: {e!r}")
        att.finished = time.time()
        for a in s.result.get("apps", ()):
            if a not in att.apps:
                att.apps.append(a)
        if refresh is not None:
            refresh(st)
        save_state(st, state_path)
        try:
            check_after(name)
        except StopRule as e:
            note_stop(s, f"stop rule after: {e}")
            rc = 3
            break
        if s.status == "stopped":
            break
        if after is not None and (code := after(name, s)) is not None:
            rc = code
            break
        if s.status != "done":
            rc = 1
            if not (keep_going and keep_going(name)):
                break
    return rc
