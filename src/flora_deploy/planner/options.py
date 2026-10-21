"""`flora plan` options (PLAN.md §2 "Requirement presets"): workload + presets + deadline + budget
+ GPU cap → 2–4 options with estimated cost, wall-clock, peak GPUs and certification status.

Model, per option (one preset at one episode concurrency per container):

* work per shard kind = Σ over workload groups of video seconds × the kind's rate for that
  group's camera count (``rates.json``), in container-seconds; with uncertainty ``u`` the range
  is ``mid / (1 + u)`` .. ``mid × (1 + u)`` (rates are positive, so the bar is multiplicative);
* containers per kind are allocated greedily: start with one each, then repeatedly add one to
  the kind that finishes last, until every kind meets the deadline (or, with no deadline, until
  the GPU cap or the per-kind episode cap stops it). Kinds run in parallel (reward shards do not
  wait for segment/caption), so wall-clock = max over kinds of startup + work / containers;
* cost = Σ kinds (work + containers × startup) × the pool's $/s (GPU × count + cores + memory,
  ``jobs.budget.Rates``) + the non-preemptible orchestrator for the wall-clock.

Not modelled: queueing for GPUs beyond the cap, a single long episode's critical path, idle
``scaledown_window`` time and Volume/egress costs. ``flora estimate`` calibrates the rates.

Rate status (review 4 M5, review 5 M-C): an option whose cost rests on rates that are not
``measured`` carries a ``warning:`` flag naming them. ``PlanRequest.features`` defaults to the
deployed configuration's features (``planner.rates.DEPLOYED_FEATURES``: ``gpu_thread``, wired
and on by default since P1a-5s), the same rates ``jobs.budget`` charges. Rates that need a
feature outside it (e.g. ``long_lived_container``) only appear in *projection* options
(``requires`` set, named ``... +long_lived_container``), which ``recommended`` never picks; among
runnable options it prefers measured rates over projected over placeholder, then the cheapest.
An option that is uncertified, priced from placeholder rates or uses a shard kind the deploy
factory refuses (``modal_app.factory.PENDING_WORKERS``) is shown with ``unavailable`` reasons
and never recommended (review 6 M2).

Everything is deterministic: same inputs, same options in the same order.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from flora_deploy.config.presets import AVAILABLE_PRESETS, get_preset, shard_kinds_for
from flora_deploy.jobs.budget import Rates, pinned_rates
from flora_deploy.jobs.settings import IN_WORKER_CONCURRENCY, PoolSettings
from flora_deploy.planner.rates import DEPLOYED_FEATURES, STATUS_RANK, RateTable
from flora_deploy.records import Stage

ORCHESTRATOR_CORES = 0.25  # modal_app/orchestrator_app.py
ORCHESTRATOR_GIB = 1.0
DEFAULT_EPISODE_S = 60.0  # assumed mean episode length when only total hours are known

Certification = Literal["reference", "needs certificates"]
Status = Literal["ok", "at risk", "infeasible"]

# The user-facing preset: its options are listed first. `exact` stays selectable as the
# certification reference. (Listing order only; which option is *recommended* is `recommended`.)
PRIMARY_PRESET = "balanced"


def default_presets() -> tuple[str, ...]:
    """Every available preset, the primary one first, the rest in their declared order."""
    return tuple(sorted(AVAILABLE_PRESETS, key=lambda p: p != PRIMARY_PRESET))


DEFAULT_PRESETS: tuple[str, ...] = default_presets()


def rates_label(option: PlanOption) -> str:
    """How to present an option's numbers: ``measured`` only when every rate it rests on was
    measured; otherwise ``estimate (<worst status>)``, so placeholder or projected rates are
    never shown as measurements."""
    return "measured" if option.rate_status == "measured" else f"estimate ({option.rate_status})"


# ----------------------------------------------------------------------------------------------
# Workload
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadGroup:
    pool: str  # embodiment pool (human | yam | ...)
    cameras: int  # views per episode (1 or 3)
    video_s: float
    episodes: int


@dataclass(frozen=True)
class WorkloadSummary:
    groups: tuple[WorkloadGroup, ...]

    @property
    def video_s(self) -> float:
        return sum(g.video_s for g in self.groups)

    @property
    def episodes(self) -> int:
        return sum(g.episodes for g in self.groups)

    @classmethod
    def from_hours(cls, video_hours: float, *, cameras: int = 1, pool: str = "human",
                   episodes: int | None = None) -> WorkloadSummary:
        video_s = video_hours * 3600.0
        if video_s <= 0:
            raise ValueError("video hours must be positive")
        n = episodes if episodes is not None else max(1, math.ceil(video_s / DEFAULT_EPISODE_S))
        return cls((WorkloadGroup(pool, cameras, video_s, n),))

    @classmethod
    def from_indexes(cls, indexes: Iterable[Any], *, pool: str = "human") -> WorkloadSummary:
        """From ``media.types.EpisodeIndex`` objects (the dataset index)."""
        from flora_deploy.jobs.manifest import episode_duration_s

        acc: dict[tuple[str, int], list[float]] = {}
        for idx in indexes:
            key = (pool, len(idx.streams))
            a = acc.setdefault(key, [0.0, 0])
            a[0] += episode_duration_s(idx)
            a[1] += 1
        return cls(tuple(WorkloadGroup(p, c, v, int(n)) for (p, c), (v, n) in sorted(acc.items())))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkloadSummary:
        """``{"groups": [{"pool", "cameras", "video_s", "episodes"}, ...]}``."""
        groups = tuple(WorkloadGroup(str(g.get("pool", "human")), int(g["cameras"]),
                                     float(g["video_s"]), int(g["episodes"]))
                       for g in data["groups"])
        if not groups or any(g.video_s <= 0 or g.episodes < 1 for g in groups):
            raise ValueError("workload groups need video_s > 0 and episodes >= 1")
        return cls(groups)


# ----------------------------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KindPlan:
    kind: str
    gpu: str
    gpu_count: int
    containers: int
    max_inputs: int  # episode concurrency per container
    cpu_cores: float
    memory_gib: float
    work_s: float  # container-seconds, mid estimate
    startup_s: float
    finish_s: float  # startup + work / containers
    cost_usd: float
    rate_status: str
    rate_provenance: str


@dataclass(frozen=True)
class PlanOption:
    name: str
    preset: str
    concurrency: int
    kinds: tuple[KindPlan, ...]
    cost_usd: float
    cost_low_usd: float
    cost_high_usd: float
    wall_s: float
    wall_high_s: float
    peak_gpus: int
    certification: Certification
    status: Status
    flags: tuple[str, ...]
    rate_status: str  # worst status among the kinds' rates
    requires: tuple[str, ...] = ()  # worker features this option needs that are not deployed
    # Why this option cannot be run (and is never recommended) today: uncertified, priced
    # from placeholder rates, or a shard kind the deploy factory refuses (review 6 M2).
    unavailable: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """``asdict``, with each kind's concurrency labelled as it is deployed (review 8a R7,
        as :func:`apply_option` writes it): an in-worker kind (segcap) gets ``max_inputs: 1``
        and ``episode_concurrency: c``; any other kind ``max_inputs: c`` and
        ``episode_concurrency: null``."""
        out = asdict(self)
        for k in out["kinds"]:
            c = k["max_inputs"]
            inside = k["kind"] in IN_WORKER_CONCURRENCY
            k["max_inputs"], k["episode_concurrency"] = (1, c) if inside else (c, None)
        return out


def episodes_label(k: KindPlan) -> str:
    """``x4 ep`` (episodes in flight inside one input) or ``x4`` (Modal inputs) for the plan
    table's GPUs column."""
    return f" x{k.max_inputs} ep" if k.kind in IN_WORKER_CONCURRENCY else f" x{k.max_inputs}"


@dataclass(frozen=True)
class PlanRequest:
    workload: WorkloadSummary
    stages: tuple[Stage, ...] = ("segment", "caption")
    presets: tuple[str, ...] = DEFAULT_PRESETS
    deadline_s: float | None = None
    budget_usd: float | None = None
    max_gpus: int = 10
    # per preset: shard kind -> pool overriding the preset default (e.g. from flora.yaml)
    pools: Mapping[str, Mapping[str, PoolSettings]] = field(default_factory=dict)
    max_options: int = 4
    # Worker features of the configuration that will run (rates.json ``requires`` /
    # ``excludes``): by default the deployed ones, as ``jobs.budget`` assumes.
    features: tuple[str, ...] = DEPLOYED_FEATURES


def _pool(req: PlanRequest, preset: str, kind: str) -> PoolSettings:
    over = req.pools.get(preset, {})
    return over.get(kind) or get_preset(preset).pools[kind]  # type: ignore[index]


@dataclass
class _Kind:
    kind: str
    pool: PoolSettings
    conc: int
    work: tuple[float, float, float]  # low, mid, high container-seconds
    startup: tuple[float, float, float]
    cap: int  # max useful containers
    per_s: float
    status: str
    provenance: str
    n: int = 1
    requires: tuple[str, ...] = ()

    def finish(self, which: int = 1, n: int | None = None) -> float:
        return self.startup[which] + self.work[which] / (n or self.n)


def _kinds(req: PlanRequest, table: RateTable, rates: Rates, preset: str, conc: int,
           features: tuple[str, ...] = ()) -> list[_Kind]:
    spec = get_preset(preset)
    out = []
    for kind in shard_kinds_for(spec, req.stages):
        pool = _pool(req, preset, kind)
        lo = mid = hi = 0.0
        worst, prov, used_c = "measured", [], 1
        needs: set[str] = set()
        for g in req.workload.groups:
            e = table.rate(preset, kind, g.cameras, conc, features)
            needs.update(e.requires)
            w = g.video_s * e.container_s_per_video_s
            lo += w / (1 + e.uncertainty)
            mid += w
            hi += w * (1 + e.uncertainty)
            used_c = max(used_c, e.concurrency)
            if STATUS_RANK[e.status] > STATUS_RANK[worst]:
                worst = e.status
            if e.provenance not in prov:
                prov.append(e.provenance)
        st = table.startup_for(preset, kind)
        u = st.uncertainty
        startup = (st.startup_s / (1 + u), st.startup_s, st.startup_s * (1 + u))
        cap = max(1, math.ceil(req.workload.episodes / used_c))
        out.append(_Kind(kind, pool, used_c, (lo, mid, hi), startup, cap,
                         rates.pool_per_second(pool), worst, " | ".join(prov),
                         requires=tuple(sorted(needs))))
    return out


def _allocate(kinds: list[_Kind], max_gpus: int, deadline_s: float | None) -> None:
    """Size for the *high* estimate, so a met deadline is met with the error bar too."""
    for k in kinds:
        k.n = 1
    used = sum(k.pool.gpu_count for k in kinds)
    while True:
        order = sorted(kinds, key=lambda k: (-k.finish(2), k.kind))
        top = order[0]
        if deadline_s is not None and top.finish(2) <= deadline_s:
            return
        if top.n >= top.cap or used + top.pool.gpu_count > max_gpus:
            return
        top.n += 1
        used += top.pool.gpu_count


def _hours(s: float) -> str:
    return f"{s / 3600:.1f} h"


def factory_refusals(kinds: Iterable[str]) -> dict[str, str]:
    """Shard kind -> why ``modal_app.factory`` refuses to deploy it (no registered worker)."""
    from flora_deploy.modal_app.factory import PENDING_WORKERS, WORKERS

    return {k: PENDING_WORKERS.get(k, "no worker class is registered for it")
            for k in kinds if k not in WORKERS}


def build_option(req: PlanRequest, table: RateTable, rates: Rates, preset: str, conc: int, *,
                 fill: bool = False, name: str | None = None,
                 extra_features: tuple[str, ...] = ()) -> PlanOption:
    """One option. ``fill``: ignore the deadline and use every GPU (the fastest variant).
    ``extra_features``: also use rates that need these (not enabled) features: a projection
    option, with ``requires`` set."""
    features = tuple(sorted(set(req.features) | set(extra_features)))
    kinds = _kinds(req, table, rates, preset, conc, features)
    missing = tuple(sorted({f for k in kinds for f in k.requires} - set(req.features)))
    base_gpus = sum(k.pool.gpu_count for k in kinds)
    _allocate(kinds, req.max_gpus, None if fill else req.deadline_s)
    peak = sum(k.n * k.pool.gpu_count for k in kinds)
    wall = max(k.finish(1) for k in kinds)
    wall_hi = max(k.finish(2) for k in kinds)
    orch_ps = rates.per_second(None, 0, ORCHESTRATOR_CORES, ORCHESTRATOR_GIB, preemptible=False)

    def cost(which: int, wall_s: float) -> float:
        return sum(k.per_s * (k.work[which] + k.n * k.startup[which]) for k in kinds) + \
            orch_ps * wall_s

    c_lo, c_mid, c_hi = cost(0, max(k.finish(0) for k in kinds)), cost(1, wall), cost(2, wall_hi)
    flags: list[str] = []
    status: Status = "ok"

    def worse(s: Status) -> None:
        nonlocal status
        order = ("ok", "at risk", "infeasible")
        if order.index(s) > order.index(status):
            status = s

    if base_gpus > req.max_gpus:
        flags.append(f"needs at least {base_gpus} GPUs at once; max_gpus is {req.max_gpus}")
        worse("infeasible")
    if req.deadline_s is not None:
        if wall > req.deadline_s:
            need = _gpus_for_deadline(kinds, req.deadline_s)
            hint = f" (needs ~{need} GPUs)" if need else " (startup alone exceeds it)"
            flags.append(f"misses the {_hours(req.deadline_s)} deadline: ~{_hours(wall)}{hint}")
            worse("infeasible")
        elif wall_hi > req.deadline_s:
            flags.append(f"deadline at risk: up to {_hours(wall_hi)}")
            worse("at risk")
    if req.budget_usd is not None:
        if c_mid > req.budget_usd:
            flags.append(f"over budget: ~${c_mid:,.0f} > ${req.budget_usd:,.0f}")
            worse("infeasible")
        elif c_hi > req.budget_usd:
            flags.append(f"budget at risk: up to ${c_hi:,.0f}")
            worse("at risk")
    worst = max((k.status for k in kinds), key=lambda s: STATUS_RANK[s])
    if missing:
        flags.append(f"projection only: needs {', '.join(missing)}, which the deployed "
                     "configuration does not provide (not runnable today)")
    unmeasured = [f"{k.kind} ({k.status})" for k in kinds if k.status != "measured"]
    if unmeasured:
        what = ("placeholder rates (not measured yet)" if worst == "placeholder"
                else "includes projected (unconfirmed) rates")
        flags.append(f"warning: {what}: {', '.join(unmeasured)}")
    cert: Certification = "reference" if preset == "exact" else "needs certificates"
    if cert == "needs certificates":
        flags.append(f"{preset}: uncertified until its certificates exist "
                     "(`flora run` needs --allow-uncertified)")
    unavailable: list[str] = []
    if cert != "reference":
        unavailable.append(f"uncertified ({preset} has no certificates)")
    if worst == "placeholder":
        unavailable.append("priced from placeholder rates: "
                           + ", ".join(k.kind for k in kinds if k.status == "placeholder"))
    refused = factory_refusals(k.kind for k in kinds)
    if refused:
        unavailable.append("refused by the deploy factory: "
                           + "; ".join(f"{k} ({why})" for k, why in refused.items()))
    if unavailable:
        flags.append("unavailable: " + "; ".join(unavailable))
    plans = tuple(KindPlan(
        kind=k.kind, gpu=k.pool.gpu, gpu_count=k.pool.gpu_count, containers=k.n,
        max_inputs=k.conc, cpu_cores=k.pool.cpu_cores, memory_gib=k.pool.memory_gib,
        work_s=round(k.work[1], 3), startup_s=k.startup[1], finish_s=round(k.finish(1), 3),
        cost_usd=round(k.per_s * (k.work[1] + k.n * k.startup[1]), 4), rate_status=k.status,
        rate_provenance=k.provenance) for k in kinds)
    main_c = max(k.conc for k in kinds)
    suffix = "".join(f" +{f}" for f in missing)
    return PlanOption(
        name=name or f"{preset} x{main_c}{suffix}", preset=preset, concurrency=main_c,
        kinds=plans, cost_usd=round(c_mid, 2), cost_low_usd=round(c_lo, 2),
        cost_high_usd=round(c_hi, 2), wall_s=round(wall, 1), wall_high_s=round(wall_hi, 1),
        peak_gpus=peak, certification=cert, status=status, flags=tuple(flags),
        rate_status=worst, requires=missing, unavailable=tuple(unavailable))


def _gpus_for_deadline(kinds: list[_Kind], deadline_s: float) -> int | None:
    total = 0
    for k in kinds:
        room = deadline_s - k.startup[1]
        if room <= 0:
            return None
        total += math.ceil(k.work[1] / room) * k.pool.gpu_count
    return total


def _dominated(a: PlanOption, b: PlanOption) -> bool:
    """``b`` is at least as good as ``a`` on everything and better on something (relative
    uncertainty within 5 points counts as equal)."""
    ua = (a.cost_high_usd - a.cost_usd) / max(a.cost_usd, 1e-9)
    ub = (b.cost_high_usd - b.cost_usd) / max(b.cost_usd, 1e-9)
    rank = ("ok", "at risk", "infeasible").index
    le = (b.cost_usd <= a.cost_usd and b.wall_s <= a.wall_s and ub <= ua + 0.05
          and rank(b.status) <= rank(a.status)
          and STATUS_RANK[b.rate_status] <= STATUS_RANK[a.rate_status])
    lt = b.cost_usd < a.cost_usd or b.wall_s < a.wall_s
    return a.preset == b.preset and a.requires == b.requires and le and lt


def plan_options(req: PlanRequest, *, table: RateTable | None = None,
                 rates: Rates | None = None) -> list[PlanOption]:
    """2–4 options (fewer only if fewer presets/rates exist), deterministic order: presets in
    request order, cheapest first within a preset, then a "fastest" variant if there is room."""
    table = table or RateTable.load()
    rates = rates or pinned_rates()
    presets = tuple(dict.fromkeys(req.presets))
    for p in presets:
        get_preset(p)  # clear error for unknown / unavailable presets
    runnable: list[PlanOption] = []
    projections: list[PlanOption] = []
    extra_sets = [()] + [(f,) for f in sorted(table.features() - set(req.features))]
    for preset in presets:
        for extra in extra_sets:
            feats = tuple(sorted(set(req.features) | set(extra)))
            concs: set[int] = set()
            for kind in shard_kinds_for(get_preset(preset), req.stages):
                for g in req.workload.groups:
                    concs.update(table.concurrencies(preset, kind, g.cameras, feats))
            opts = [build_option(req, table, rates, preset, c, extra_features=extra)
                    for c in sorted(concs or {1})]
            if extra:  # only options that actually use the extra feature's rates
                opts = [o for o in opts if o.requires]
            uniq: dict[tuple, PlanOption] = {}
            for o in opts:  # different requested concurrencies can resolve to the same rates
                uniq.setdefault((tuple((k.kind, k.max_inputs) for k in o.kinds), o.requires),
                                o)
            opts = list(uniq.values())
            opts = [o for o in opts if not any(_dominated(o, b) for b in opts if b is not o)]
            (projections if extra else runnable).extend(
                sorted(opts, key=lambda o: (o.cost_usd, o.wall_s, o.concurrency)))
    # Runnable options first; projections fill the remaining slots (at least one is shown when
    # there are any, so the user sees what an unwired feature would buy).
    keep = req.max_options - (1 if projections and len(runnable) >= req.max_options else 0)
    out = runnable[:keep]
    out += projections[: max(0, req.max_options - len(out))]
    if len(out) < req.max_options and req.deadline_s is not None:
        base = recommended(out)
        if base is not None:
            fast = build_option(req, table, rates, base.preset, base.concurrency, fill=True,
                                name=f"{base.name} fastest")
            if fast.peak_gpus != base.peak_gpus:
                out.append(fast)
    return out


def recommended(options: Sequence[PlanOption],
                certified: Collection[str] = ()) -> PlanOption | None:
    """The option to suggest: never a projection that needs an unwired feature, never an
    ``unavailable`` one (uncertified, placeholder-priced or refused by the factory); ``ok`` before
    ``at risk``; then the reference preset, or an option named in ``certified`` (every pool has
    a ``certified`` Gate certificate), before uncertified ones; then measured rates before
    projected before placeholders (review 4 M5); then the cheapest. So a certified ``balanced``
    that is cheaper wins over ``exact``. ``None`` when no option is feasible and runnable."""
    live = [o for o in options
            if o.status != "infeasible" and not o.requires and not o.unavailable]
    if not live:
        return None
    return min(live, key=lambda o: (o.status != "ok",
                                    o.certification != "reference" and o.name not in certified,
                                    STATUS_RANK[o.rate_status], o.cost_usd, o.wall_s, o.name))


# ----------------------------------------------------------------------------------------------
# Writing an option back into flora.yaml
# ----------------------------------------------------------------------------------------------


def apply_option(data: Mapping[str, Any], option: PlanOption, *,
                 budget_usd: float | None = None) -> dict[str, Any]:
    """A copy of a parsed ``flora.yaml`` with the option's preset and per-kind
    ``max_containers`` / ``max_inputs`` (for kinds that batch episodes in-process, segcap:
    ``max_inputs: 1`` and the option's concurrency as ``episode_concurrency``, which is what the
    factory deploys and the worker runs). Resource overrides for kinds the new preset does not
    use are dropped; other keys (gpu, cpu, memory overrides) are kept. ``modal.max_gpus`` is
    raised to the option's peak if needed, and ``budget_usd`` (when given) replaces
    ``modal.budget_usd``."""
    out = dict(data)
    out["preset"] = option.preset
    modal = dict(data.get("modal") or {})
    modal["max_gpus"] = max(int(modal.get("max_gpus") or 0), option.peak_gpus)
    if budget_usd is not None:
        modal["budget_usd"] = budget_usd
    out["modal"] = modal
    used = {k.kind for k in option.kinds}
    res = {k: dict(v or {}) for k, v in dict(data.get("resources") or {}).items() if k in used}
    if data.get("preset", "exact") != option.preset:
        res = {k: {} for k in res}  # preset GPUs differ; overrides were for the old preset
    for k in option.kinds:
        r = res.setdefault(k.kind, {})
        r["max_containers"] = k.containers
        if k.kind in IN_WORKER_CONCURRENCY:  # review 6 H2: c episodes inside ONE input
            r["max_inputs"] = 1
            r["episode_concurrency"] = k.max_inputs
        else:
            r["max_inputs"] = k.max_inputs
            r.pop("episode_concurrency", None)
    out["resources"] = res
    return out


def with_pool_overrides(req: PlanRequest, preset: str, pools: Mapping[str, PoolSettings]
                        ) -> PlanRequest:
    """Use the config's resolved pools (GPU type, cores, memory) for ``preset``; container
    counts are what the planner decides, so they are reset to one."""
    clean = {k: replace(p, max_containers=1, max_inputs=1) for k, p in pools.items()}
    return replace(req, pools={**req.pools, preset: clean})
