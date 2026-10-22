"""Budget accounting and the orchestrator's budget gate (PLAN.md §2, §5).

* **Rates** (:class:`Rates`): $/s per GPU type, per physical core and per GiB of memory, plus a
  region multiplier and the non-preemptible CPU/memory multiplier. Live prices come from
  ``modal.Workspace.from_context().billing.rates()`` through an injectable provider
  (:func:`load_rates`); without one, or if the call fails, the pinned :data:`PINNED_PRICES` table
  is used and :attr:`Rates.source` says so.
* **Spend** (:func:`container_cost`, :func:`spend`): Σ container lifetime × (GPU × count + cores +
  memory) rates, over ``ContainerReceipt`` files. A container without ``stopped_utc`` is billed up
  to *now* while its heartbeat is fresh, and up to ``heartbeat + stale_after_s`` once the heartbeat
  is stale (presumed dead; Modal bills until the container is actually gone, so this is the
  conservative end of what we can know). Spend never depends on receipts alone:
  :func:`calls_cost` estimates it from the orchestrator's spawn-time call history,
  :func:`modal_billing_provider` reads Modal's hourly billing, and the gate takes the largest of
  the three (:class:`SpendReport`).
* **Estimates** (:class:`RateTableEstimator`): ``planner/rates.json``, the same measured rates
  ``flora plan`` uses (exact segcap 6.4 container-s per video-s at c=4 with the deployed
  ``gpu_thread``, P1a-5s), plus its per-container startup, calibrated online from finished
  calls. ``Shard.est_gpu_seconds``
  (placeholder manifest rates, :class:`ManifestEstimator`) is only a fallback.
* **Projection** (:func:`project`): spend + in-flight (remaining estimate) + queued + future
  container starts.
* **Gate** (:class:`ProjectionBudgetGate`): implements ``orchestrator.BudgetGate``. Denies a spawn
  when the projection exceeds ``budget × (1 − safety_margin)``; the orchestrator then writes
  ``budget_stop`` and drains. :meth:`ProjectionBudgetGate.overspend` is the hard cap: the live
  calls to *cancel* when spend + in-flight would pass ``budget_usd``.
* **Backstop** (:func:`environment_budget_usd`, :func:`apply_environment_budget`): the native Modal
  Environment budget at ``budget × environment_budget_factor``. Modal's Python SDK (1.6.2) has no
  call to set an Environment budget, so the default is the exact manual instruction, printed when
  the orchestrator starts and carried in its run summary (:func:`budget_summary`).
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol

from flora_deploy.jobs.orchestrator import BudgetDecision
from flora_deploy.jobs.pools import DRAINED_SCALEDOWN_S, ORCHESTRATOR_PREEMPTIBLE, idle_tail_s
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import ContainerReceipt, Shard
from flora_deploy.planner.rates import RateTable

# ----------------------------------------------------------------------------------------------
# Rates
# ----------------------------------------------------------------------------------------------

#: Modal list prices in $/hour. Source: modal.com/pricing, as recorded in
#: ``research_notes/Plan review v2/modal_verification.md`` §15 (verified 2026-10-21) and
#: ``docs/exact-concurrency.md`` (H200 $0.001261/s, core $0.0000131/s, GiB $0.00000222/s).
PINNED_PRICES_SOURCE = "Modal list prices (modal.com/pricing, pinned 2026-10-21)"
PINNED_PRICES: dict[str, float] = {
    "B300": 7.10,
    "B200": 6.25,
    "H200": 4.54,
    "H100": 3.95,
    "RTX-PRO-6000": 3.03,
    "A100-80GB": 2.50,
    "A100-40GB": 2.10,
    "L40S": 1.95,
}
PINNED_CORE_HOUR = 0.0472  # per physical core
PINNED_GIB_HOUR = 0.0080  # per GiB of memory
NONPREEMPTIBLE_CPU_MEM_MULTIPLIER = 3.0  # Modal: non-preemptible is 3x on CPU/memory
REGION_MULTIPLIER_BROAD = 1.15  # Modal: region selection 1.15x (broad) .. 1.75x (narrow)

_GPU_ALIASES = {"A100": "A100-40GB", "A100-80": "A100-80GB", "A100-40": "A100-40GB",
                "RTX-PRO6000": "RTX-PRO-6000", "RTXPRO6000": "RTX-PRO-6000",
                "RTX-6000-PRO": "RTX-PRO-6000"}


def gpu_key(gpu: str) -> str:
    """Normalize a Modal GPU string: ``"H100!"`` / ``"h100:2"`` -> ``"H100"``."""
    g = gpu.strip().upper().rstrip("!").split(":")[0].replace("_", "-").replace(" ", "-")
    return _GPU_ALIASES.get(g, g)


@dataclass(frozen=True)
class Rates:
    """Prices in $/second. ``region_multiplier`` scales everything (set it when a job pins a
    ``region``); ``nonpreemptible_multiplier`` scales CPU and memory of non-preemptible
    containers."""

    gpu_per_s: Mapping[str, float]
    core_per_s: float
    gib_per_s: float
    region_multiplier: float = 1.0
    nonpreemptible_multiplier: float = NONPREEMPTIBLE_CPU_MEM_MULTIPLIER
    source: str = PINNED_PRICES_SOURCE

    def gpu(self, gpu: str | None) -> float:
        if gpu is None:
            return 0.0
        key = gpu_key(gpu)
        if key not in self.gpu_per_s:
            raise KeyError(f"no price for GPU {gpu!r} (known: {', '.join(sorted(self.gpu_per_s))})")
        return self.gpu_per_s[key]

    def per_second(self, gpu: str | None, gpu_count: int, cores: float, memory_gib: float, *,
                   preemptible: bool = True) -> float:
        """$/s of one container with these requests."""
        cm = 1.0 if preemptible else self.nonpreemptible_multiplier
        return self.region_multiplier * (
            (self.gpu(gpu) * gpu_count if gpu and gpu_count else 0.0)
            + cm * (cores * self.core_per_s + memory_gib * self.gib_per_s))

    def pool_per_second(self, pool: PoolSettings) -> float:
        return self.per_second(pool.gpu, pool.gpu_count, pool.cpu_cores, pool.memory_gib)

    def with_region(self, multiplier: float) -> Rates:
        return replace(self, region_multiplier=multiplier)


def pool_rate(rates: Rates, settings: RunSettings, kind: str) -> float:
    """$/s of one container of ``kind``'s pool **as billed**: a ``fake_models`` job (the H0
    rehearsal app) runs every kind on a CPU container (``jobs.settings.billed_pool``), never on
    the pool's GPU."""
    from flora_deploy.jobs.settings import billed_pool

    return rates.pool_per_second(billed_pool(settings, kind))


def pinned_rates(*, region_multiplier: float = 1.0) -> Rates:
    return Rates(gpu_per_s={k: v / 3600.0 for k, v in PINNED_PRICES.items()},
                 core_per_s=PINNED_CORE_HOUR / 3600.0, gib_per_s=PINNED_GIB_HOUR / 3600.0,
                 region_multiplier=region_multiplier, source=PINNED_PRICES_SOURCE)


RatesProvider = Callable[[], Mapping[str, Any]]


def modal_rates_provider() -> Mapping[str, Any]:
    """Live ``Workspace.billing.rates()`` (Team+ plans). Never called by tests."""
    import modal

    return modal.Workspace.from_context().billing.rates()


_HOURLY_THRESHOLD = 0.05  # $/s above this for one GPU means the value is $/hour


def _to_per_s(value: float) -> float:
    return value / 3600.0 if value > _HOURLY_THRESHOLD else value


def rates_from_mapping(raw: Mapping[str, Any], *, fallback: Rates | None = None,
                       source: str = "Workspace.billing.rates()") -> Rates:
    """Build :class:`Rates` from a ``billing.rates()``-style mapping.

    The key format of ``billing.rates()`` is not documented, so parsing is lenient: a key that
    mentions ``cpu``/``core`` is the per-core price, ``mem`` the per-GiB price, and any key that
    contains a known GPU name is that GPU's price. Values are $/s; a GPU value above $0.05 is
    taken as $/hour. Anything missing comes from ``fallback`` (default: the pinned table)."""
    base = fallback or pinned_rates()
    gpus = dict(base.gpu_per_s)
    core, gib = base.core_per_s, base.gib_per_s
    names = sorted(set(gpus) | set(_GPU_ALIASES), key=len, reverse=True)
    for k, v in raw.items():
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        key = re.sub(r"[\s_.]+", "-", str(k).upper())
        if "MEM" in key:
            gib = val / 3600.0 if val > 0.001 else val
        elif "CPU" in key or "CORE" in key:
            core = val / 3600.0 if val > 0.01 else val
        else:
            for name in names:
                if re.search(rf"(^|[^A-Z0-9]){re.escape(name)}($|[^A-Z0-9])", key):
                    gpus[_GPU_ALIASES.get(name, name)] = _to_per_s(val)
                    break
    return replace(base, gpu_per_s=gpus, core_per_s=core, gib_per_s=gib, source=source)


def load_rates(provider: RatesProvider | None = None, *, region_multiplier: float = 1.0
               ) -> Rates:
    """Live rates if ``provider`` works, else the pinned table (``source`` records which)."""
    pinned = pinned_rates(region_multiplier=region_multiplier)
    if provider is None:
        return pinned
    try:
        raw = provider()
    except Exception as exc:  # no Team plan, no credentials, API change: fall back, loudly
        return replace(pinned, source=f"{PINNED_PRICES_SOURCE}; live rates failed: {exc!r}")
    return rates_from_mapping(raw, fallback=pinned)


def region_multiplier_for(region: str | None) -> float:
    """Price multiplier for a pinned region (broad 1.15x; narrow regions cost up to 1.75x, so set
    ``Rates.region_multiplier`` explicitly when pinning a single zone)."""
    return 1.0 if not region else REGION_MULTIPLIER_BROAD


# ----------------------------------------------------------------------------------------------
# Spend
# ----------------------------------------------------------------------------------------------


def parse_utc(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def billed_seconds(r: ContainerReceipt, now: float, *, stale_after_s: float = 300.0) -> float:
    """Seconds this container has been (or will at least have been) billed."""
    start = parse_utc(r.started_utc)
    if r.stopped_utc is not None:
        end = parse_utc(r.stopped_utc)
    else:
        hb = parse_utc(r.heartbeat_utc)
        end = now if now - hb <= stale_after_s else hb + stale_after_s
    return max(0.0, end - start)


def container_cost(r: ContainerReceipt, rates: Rates, now: float, *,
                   stale_after_s: float = 300.0) -> float:
    return billed_seconds(r, now, stale_after_s=stale_after_s) * rates.per_second(
        r.gpu, r.gpu_count, r.cpu_request, r.memory_request_gib, preemptible=r.preemptible)


@dataclass(frozen=True)
class Spend:
    total_usd: float
    running_usd: float  # part of total from containers without stopped_utc
    containers: int
    running: int


def spend(receipts: Iterable[ContainerReceipt], rates: Rates, now: float, *,
          stale_after_s: float = 300.0) -> Spend:
    total = running_usd = 0.0
    n = running = 0
    for r in receipts:
        c = container_cost(r, rates, now, stale_after_s=stale_after_s)
        total += c
        n += 1
        if r.stopped_utc is None:
            running_usd += c
            running += 1
    return Spend(total, running_usd, n, running)


def read_container_receipts(store: Store, job_id: str) -> list[ContainerReceipt]:
    prefix = JobPaths(job_id).root + "/containers/"
    out = []
    for path in sorted(store.list(prefix)):
        if path.endswith(".json"):
            out.append(ContainerReceipt(**json.loads(store.get_bytes(path))))
    return out


# ----------------------------------------------------------------------------------------------
# Spend from the orchestrator's own call history, and from Modal billing
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallRecord:
    """One worker call as the orchestrator recorded it (``call.json`` ``calls`` entries).

    ``spawned_at`` is when the orchestrator spawned it; ``ended_at`` when the orchestrator saw it
    end (success, failure, expiry, cancel), ``None`` while it is live."""

    call_id: str
    shard_id: str
    kind: str
    spawned_at: float
    ended_at: float | None = None
    outcome: str | None = None

    @property
    def live(self) -> bool:
        return self.ended_at is None


@dataclass(frozen=True)
class BudgetState:
    """What the orchestrator tells the gate every tick (see :meth:`ProjectionBudgetGate.observe`
    and :meth:`ProjectionBudgetGate.overspend`)."""

    now: float
    calls: tuple[CallRecord, ...] = ()
    # kind -> intervals ``(start, end | None)`` its worker class was drained (``jobs.pools``)
    drains: Mapping[str, tuple[tuple[float, float | None], ...]] = field(default_factory=dict)
    # kind -> ``(start, end | None)`` of a pre-warm container waiting for its first call
    prewarms: Mapping[str, tuple[float, float | None]] = field(default_factory=dict)

    def live(self) -> list[CallRecord]:
        return [c for c in self.calls if c.live]


def calls_cost(calls: Iterable[CallRecord], settings: RunSettings, rates: Rates, now: float, *,
               scaledown_s: float = 60.0, warm: int = 0,
               drains: Mapping[str, Iterable[tuple[float, float | None]]] | None = None,
               drained_s: float = DRAINED_SCALEDOWN_S,
               prewarms: Mapping[str, tuple[float, float | None]] | None = None) -> float:
    """Spawn-time spend estimate, independent of container receipts.

    Per kind, a sweep over call intervals ``[spawned_at, ended_at or now]``: at every instant the
    kind runs ``ceil(live calls / max_inputs)`` containers, each billed at the pool rate, plus
    ``scaledown_s`` of idle time every time a container goes away (Modal keeps it for its
    ``scaledown_window``). A container never starts before its call is spawned and the
    orchestrator sees a call end only after it ended, so this over- rather than under-counts the
    containers of the calls it knows (it cannot see orphan calls; receipts and billing can).

    ``warm``: ``min_containers`` warm containers per kind, billed from the kind's first call on
    (kinds sharing one worker class are each charged their own: an over-count).

    ``drains``: per kind, when the orchestrator drained its worker class (``jobs.pools``): a
    container going idle then is charged only until ``drain start + drained_s``
    (:func:`~flora_deploy.jobs.pools.idle_tail_s`), not the whole ``scaledown_s``.

    ``prewarms``: per kind, a pre-warm container (``jobs.pools``) billed from its start until its
    first call was spawned (``end``) or up to now, on top of the calls."""
    by_kind: dict[str, list[tuple[float, int]]] = {}
    for c in calls:
        start = min(c.spawned_at, now)
        ev = by_kind.setdefault(c.kind, [])
        ev.append((start, +1))
        if c.ended_at is not None and c.ended_at < now:  # live calls run up to now
            ev.append((max(c.ended_at, start), -1))
    total = 0.0
    for kind, ev in by_kind.items():
        pool = settings.pools[kind]  # type: ignore[index]
        mi = max(pool.max_inputs, 1)
        per_s = pool_rate(rates, settings, kind)
        ev.sort(key=lambda e: (e[0], -e[1]))  # starts before ends at the same instant
        ev.append((now, 0))
        n = 0
        last = ev[0][0]
        kind_drains = tuple((drains or {}).get(kind, ()))
        for t, d in ev:
            containers = max(math.ceil(n / mi), warm)
            total += (t - last) * containers * per_s
            n += d
            after = max(math.ceil(n / mi), warm)
            if after < containers:
                tail = (idle_tail_s(t, scaledown_s, kind_drains, drained_s) if kind_drains
                        else scaledown_s)
                total += (containers - after) * tail * per_s
            last = t
    for kind, (t0, t1) in (prewarms or {}).items():
        if kind in settings.pools:
            end = now if t1 is None else min(t1, now)
            total += max(end - t0, 0.0) * pool_rate(rates, settings, kind)
    return total


BillingProvider = Callable[[], float]


def modal_billing_provider(*, start: datetime, tags: Mapping[str, str] | None = None,
                           descriptions: Sequence[str] = (),
                           report: Callable[..., Any] | None = None) -> BillingProvider:
    """Job spend from Modal's own billing (``Workspace.billing.report``, the API behind
    ``modal billing report``), hourly resolution.

    Rows count when their tags include every ``tags`` item (workers tagged with the job id) or
    their description is one of ``descriptions`` (e.g. a per-job worker app). Modal reports only
    *complete* intervals, so this lags by up to an hour (plus Modal's own processing delay): it
    is a floor that catches what receipts and the call history miss (orphan calls, idle
    containers, max(request, usage) CPU/memory billing), never the only source. Never called by
    tests; ``report`` is injectable."""

    def provider() -> float:
        fn = report
        if fn is None:
            import modal

            fn = modal.Workspace.from_context().billing.report
        rows = fn(start=start, resolution="h", tag_names=["*"])
        want = dict(tags or {})
        total = 0.0
        for row in rows:
            row_tags = dict(row["tags"] or {})
            if (want and all(row_tags.get(k) == v for k, v in want.items())) or \
                    row["description"] in descriptions:
                total += float(row["cost"])
        return total

    return provider


@dataclass(frozen=True)
class SpendReport:
    """Spend from each source, reconciled. ``total_usd`` is the largest: each source is a
    (near-)floor of the truth for a different reason, so the max never undercounts what any of
    them can see."""

    receipts_usd: float  # ContainerReceipt files (known ones re-read by path every tick)
    calls_usd: float  # spawn-time estimate from the orchestrator's call history
    billing_usd: float | None  # Modal billing report (hourly, lagging), if configured
    containers: int  # container receipts known
    live_calls: int

    @property
    def total_usd(self) -> float:
        return max(self.receipts_usd, self.calls_usd, self.billing_usd or 0.0)

    @property
    def source(self) -> str:
        vals = {"receipts": self.receipts_usd, "calls": self.calls_usd,
                "billing": self.billing_usd or 0.0}
        return max(vals, key=lambda k: vals[k])

    def as_dict(self) -> dict[str, Any]:
        return {"spend_receipts_usd": round(self.receipts_usd, 4),
                "spend_calls_usd": round(self.calls_usd, 4),
                "spend_billing_usd": None if self.billing_usd is None
                else round(self.billing_usd, 4),
                "spend_source": self.source, "containers": self.containers,
                "live_calls": self.live_calls}


# ----------------------------------------------------------------------------------------------
# Shard estimates (planner/rates.json, the same numbers ``flora plan`` uses)
# ----------------------------------------------------------------------------------------------


class ShardEstimator(Protocol):
    @property
    def source(self) -> str: ...

    def container_seconds(self, shard: Shard, pool: PoolSettings) -> float:
        """Container-seconds of ``pool`` one shard costs (amortized over ``max_inputs``)."""
        ...

    def startup_s(self, kind: str) -> float:
        """Seconds from container start to ready, paid once per container."""
        ...


@dataclass(frozen=True)
class ManifestEstimator:
    """``Shard.est_gpu_seconds`` / ``gpu_count``: the manifest's own (placeholder-rate) estimate.
    Only for tests and for kinds ``rates.json`` has no entry for."""

    source: str = "manifest est_gpu_seconds"

    def container_seconds(self, shard: Shard, pool: PoolSettings) -> float:
        return shard.est_gpu_seconds / max(pool.gpu_count, 1)

    def startup_s(self, kind: str) -> float:
        return 0.0


@dataclass(frozen=True)
class RateTableEstimator:
    """Estimates from ``planner/rates.json`` (:class:`~flora_deploy.planner.rates.RateTable`).

    Shard cost = video seconds × ``container_s_per_video_s`` for (preset, kind, cameras,
    concurrency = ``jobs.settings.episode_concurrency``: the pool's ``max_inputs``, or for segcap
    its in-worker ``episode_concurrency``; features = the deployed configuration's,
    ``planner.rates.DEPLOYED_FEATURES``), e.g. exact segcap 6.4 container-s per video-s (c=4,
    gpu_thread; the only usable cu128 row, so c=1 uses it too). Video seconds come from
    ``durations`` (uid -> seconds, from the manifest's ``EpisodeRef.duration_s``) or, without
    them, are recovered from ``est_gpu_seconds`` by dividing out the manifest's placeholder
    rate (``jobs.manifest.DEFAULT_RATES``). ``quantile="upper"`` charges the top of each
    rate's error bar (``× (1 + uncertainty)``)."""

    table: RateTable
    preset: str
    cameras: int | Mapping[str, int] = 1
    # uid -> seconds, or a callable returning that (read lazily: e.g. from the manifest)
    durations: Mapping[str, float] | Callable[[], Mapping[str, float]] | None = None
    quantile: str = "mid"
    fallback: ManifestEstimator = field(default_factory=ManifestEstimator)
    # kind -> GPU-s per video-s the manifest packed with (``Manifest.rates``), or a callable
    packing_rates: Mapping[str, float] | Callable[[], Mapping[str, float]] | None = None

    @property
    def source(self) -> str:
        return f"{self.table.source} ({self.quantile})"

    @classmethod
    def default(cls, preset: str, **kw: Any) -> RateTableEstimator:
        return cls(RateTable.load(), preset, **kw)

    def video_seconds(self, shard: Shard) -> float:
        d = self.durations() if callable(self.durations) else self.durations
        if d is not None and all(u in d for u in shard.episodes):
            return float(sum(d[u] for u in shard.episodes))
        from flora_deploy.jobs.manifest import DEFAULT_RATES

        packed = self.packing_rates() if callable(self.packing_rates) else self.packing_rates
        rate = (packed or {}).get(shard.kind) or DEFAULT_RATES[shard.kind]
        return shard.est_gpu_seconds / rate

    def _cameras(self, shard: Shard) -> int:
        if isinstance(self.cameras, int):
            return self.cameras
        return max((self.cameras.get(u, 1) for u in shard.episodes), default=1)

    def container_seconds(self, shard: Shard, pool: PoolSettings) -> float:
        from flora_deploy.jobs.settings import episode_concurrency

        try:  # the concurrency the rate was measured at (segcap: episodes in one input, H2)
            e = self.table.rate(self.preset, shard.kind, self._cameras(shard),
                                episode_concurrency(shard.kind, pool))
        except KeyError:
            return self.fallback.container_seconds(shard, pool)
        k = 1.0 + e.uncertainty if self.quantile == "upper" else 1.0
        return self.video_seconds(shard) * e.container_s_per_video_s * k

    def startup_s(self, kind: str) -> float:
        e = self.table.startup_for(self.preset, kind)
        return e.startup_s * (1.0 + e.uncertainty if self.quantile == "upper" else 1.0)


# ----------------------------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    spend_usd: float
    in_flight_usd: float
    queued_usd: float
    startup_usd: float

    @property
    def total_usd(self) -> float:
        return self.spend_usd + self.in_flight_usd + self.queued_usd + self.startup_usd

    def as_dict(self) -> dict[str, float]:
        return {"spend_usd": round(self.spend_usd, 4),
                "in_flight_usd": round(self.in_flight_usd, 4),
                "queued_usd": round(self.queued_usd, 4),
                "startup_usd": round(self.startup_usd, 4),
                "projected_usd": round(self.total_usd, 4)}


def shard_container_seconds(shard: Shard, pool: PoolSettings, calibration: float = 1.0, *,
                            estimator: ShardEstimator | None = None) -> float:
    """Container-seconds of one shard (default: the manifest's ``est_gpu_seconds`` / GPUs)."""
    return (estimator or ManifestEstimator()).container_seconds(shard, pool) * calibration


def project(*, spent: float, pending: Sequence[Shard], in_flight: Mapping[str, int],
            settings: RunSettings, rates: Rates, startup_s: Mapping[str, float] | None = None,
            calibration: Mapping[str, float] | None = None, in_flight_fraction: float = 1.0,
            estimator: ShardEstimator | None = None,
            live_remaining_usd: Mapping[str, Sequence[float]] | None = None) -> Projection:
    """Projected job cost. ``pending``: shards without a receipt (in-flight ones included).
    ``in_flight``: active calls per kind (as the orchestrator passes them). ``startup_s``:
    per-kind container startup seconds, charged for the containers still to be started
    (``max_containers`` minus those in flight) of kinds that still have queued work.
    ``live_remaining_usd``: per kind, the remaining cost of each live call the gate knows about
    (estimate less what has already run, which is in ``spent``); in-flight calls beyond those
    are charged ``mean estimate × in_flight_fraction``."""
    by_kind: dict[str, list[float]] = {}
    for s in pending:
        pool = settings.pools[s.kind]
        cal = (calibration or {}).get(s.kind, 1.0)
        by_kind.setdefault(s.kind, []).append(
            shard_container_seconds(s, pool, cal, estimator=estimator)
            * pool_rate(rates, settings, s.kind))
    in_flight_usd = queued_usd = startup_usd = 0.0
    for kind in sorted(set(by_kind) | set(in_flight) | set(live_remaining_usd or {})):
        costs = by_kind.get(kind, [])
        known = list((live_remaining_usd or {}).get(kind, ()))
        n_live = max(in_flight.get(kind, 0), len(known))
        n_flight = min(n_live, len(costs))
        mean = sum(costs) / len(costs) if costs else 0.0
        in_flight_usd += sum(known) + max(n_flight - len(known), 0) * mean * in_flight_fraction
        queued_usd += sum(costs) - n_flight * mean
        n_queued = len(costs) - n_flight
        if n_queued > 0 and startup_s and startup_s.get(kind):
            pool = settings.pools[kind]  # type: ignore[index]
            running = math.ceil(n_live / max(pool.max_inputs, 1))
            new = min(max(pool.max_containers - running, 0),
                      math.ceil(n_queued / max(pool.max_inputs, 1)))
            startup_usd += new * startup_s[kind] * pool_rate(rates, settings, kind)
    return Projection(spent, in_flight_usd, queued_usd, startup_usd)


# ----------------------------------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------------------------------


@dataclass
class ProjectionBudgetGate:
    """``BudgetGate`` for the orchestrator, plus the hard-cap hook :meth:`overspend`.

    Per tick the orchestrator driver (``jobs.driver.TickDriver``, Modal and local) calls
    :meth:`refresh` (drop cached reads), then :meth:`overspend` with the orchestrator's call
    history (:class:`BudgetState`; it also :meth:`observe`\\ s it) and cancels what it returns;
    then ``tick()`` calls :meth:`admit` before each spawn.

    **Spend** (:meth:`spend_report`) reconciles three sources and takes the largest: container
    receipts (each one ever listed is re-read *by path* every tick, so listing lag never makes a
    known one vanish, and its cost never goes down), the orchestrator's spawn-time call history
    (:func:`calls_cost`: works with no receipts at all, and sees new calls before any listing
    does), and Modal billing (``billing``, hourly, lagging).

    **Admission** denies when the projection exceeds ``budget × (1 − safety_margin)``.
    ``policy="projection"`` (default) denies as soon as the *whole remaining job* no longer fits;
    ``policy="next_shard"`` only when spend + in-flight + this shard does not fit. Live calls are
    charged their *remaining* estimate (what already ran is in spend).

    **Hard cap** (:meth:`overspend`): when spend + live calls' remaining estimates + one tick of
    burn exceeds ``budget_usd`` (no margin), returns live calls to cancel, largest remaining cost
    first, until the rest fits; all of them once spend + one tick of burn alone reaches the cap.
    That bounds the overspend by one tick of burn, whatever the estimates said.

    **Estimates** default to ``planner/rates.json`` (:class:`RateTableEstimator`), the numbers
    ``flora plan`` uses, and are calibrated online: per kind, the mean of (observed wall time −
    startup) / (estimate × max_inputs) over finished calls, never below ``calibration``; the
    startup is subtracted only from calls that needed a new container (no earlier call of the
    kind ended within ``scaledown_s`` before the spawn)."""

    settings: RunSettings
    store: Store
    shards: Callable[[], Sequence[Shard]]  # effective shards (manifest + bisection children)
    rates: Rates = field(default_factory=pinned_rates)
    safety_margin: float = 0.05
    policy: str = "projection"
    startup_s: Mapping[str, float] | None = None  # default: the estimator's startup per kind
    calibration: Mapping[str, float] = field(default_factory=dict)
    in_flight_fraction: float = 1.0
    stale_after_s: float = 300.0
    refresh_s: float = 60.0  # fallback cache lifetime when nobody calls refresh() per tick
    quarantined: Callable[[], Iterable[str]] = lambda: ()
    clock: Callable[[], float] = time.time
    estimator: ShardEstimator | None = None  # default: RateTableEstimator for settings.preset
    billing: BillingProvider | None = None
    billing_refresh_s: float = 900.0
    scaledown_s: float = 60.0  # production (``jobs.deps``): the deployed ``scaledown_window``
    tick_reserve_s: float = 30.0  # burn charged ahead in overspend(): poll interval + cancel lag
    # kind -> stage hash (or a callable): where each kind's shard receipts live (``JobPaths``)
    stage_hashes: Mapping[str, str] | Callable[[], Mapping[str, str]] | None = None
    # the orchestrator's own container (``jobs.pools``: preemptible CPU since the lean-ops
    # change), billed from the first call on
    orchestrator_cores: float = 0.0
    orchestrator_memory_gib: float = 0.0
    orchestrator_preemptible: bool = ORCHESTRATOR_PREEMPTIBLE
    learn_calibration: bool = True
    billing_error: str | None = field(default=None, init=False)
    last_overspend: dict[str, Any] = field(default_factory=dict, init=False)
    _cache_at: float = field(default=-math.inf, init=False, repr=False)
    _known: set[str] = field(default_factory=set, init=False, repr=False)
    _receipts: dict[str, ContainerReceipt] = field(default_factory=dict, init=False, repr=False)
    _floor: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _done: set[str] = field(default_factory=set, init=False, repr=False)
    _state: BudgetState | None = field(default=None, init=False, repr=False)
    _billing_at: float = field(default=-math.inf, init=False, repr=False)
    _billing_usd: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.safety_margin < 1.0:
            raise ValueError("safety_margin must be in [0, 1)")
        if self.policy not in ("projection", "next_shard"):
            raise ValueError("policy must be 'projection' or 'next_shard'")
        if self.estimator is None:
            self.estimator = RateTableEstimator.default(self.settings.preset)

    @property
    def limit_usd(self) -> float:
        return self.settings.budget_usd * (1.0 - self.safety_margin)

    # reads ----------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Start of a tick: drop cached listings so this tick re-reads the store (call it after
        the store's own ``refresh()``). Receipts already known stay known."""
        self._cache_at = -math.inf

    def _refresh(self, now: float) -> None:
        if now - self._cache_at < self.refresh_s:
            return
        paths = JobPaths(self.settings.job_id)
        self._known |= {p for p in self.store.list(paths.root + "/containers/")
                        if p.endswith(".json")}
        for p in sorted(self._known):
            old = self._receipts.get(p)
            if old is not None and old.stopped_utc is not None:
                continue  # final: never re-read
            try:  # by path, not by listing: a known receipt never disappears under lag
                self._receipts[p] = ContainerReceipt(**json.loads(self.store.get_bytes(p)))
            except Exception:  # half-visible or a transient read error: keep what we had
                continue
        sh = self.stage_hashes() if callable(self.stage_hashes) else self.stage_hashes
        keys = {self.settings.config_hash, *(sh or {}).values()}
        for key in sorted(keys):
            prefix = f"{paths.root}/{key}/receipts/"
            self._done |= {p.rsplit("/", 1)[-1].removesuffix(".json")
                           for p in self.store.list(prefix) if p.endswith(".json")}
        self._cache_at = now

    def _billing_now(self, now: float) -> float | None:
        if self.billing is None:
            return None
        if now - self._billing_at >= self.billing_refresh_s:
            try:
                self._billing_usd = max(float(self.billing()), self._billing_usd or 0.0)
                self.billing_error = None
            except Exception as exc:  # billing API down / plan without billing: other sources
                self.billing_error = repr(exc)
            self._billing_at = now
        return self._billing_usd

    # orchestrator state ---------------------------------------------------------------------
    def observe(self, state: BudgetState) -> None:
        """The orchestrator's call history this tick (live and ended calls)."""
        self._state = state
        for c in state.calls:
            if c.outcome == "success":
                self._done.add(c.shard_id)

    def _shard_map(self) -> dict[str, Shard]:
        return {s.shard_id: s for s in self.shards()}

    def _cs(self, shard: Shard) -> float:
        assert self.estimator is not None
        return self.estimator.container_seconds(shard, self.settings.pools[shard.kind])

    def _startup(self, kind: str) -> float:
        if self.startup_s is not None:
            return float(self.startup_s.get(kind, 0.0))
        assert self.estimator is not None
        return self.estimator.startup_s(kind)

    def effective_calibration(self) -> dict[str, float]:
        cal = dict(self.calibration)
        if not self.learn_calibration or self._state is None:
            return cal
        by_id = self._shard_map()
        ratios: dict[str, list[float]] = {}
        # Review 5 M-C / review 4 M5: startup is paid once per container, not per call. A call
        # spawned when an earlier call of its kind had ended within ``scaledown_s`` ran on that
        # warm container: its wall time is all work. Only the other ("cold") calls have the
        # startup subtracted (subtracting it from every call under-calibrates long jobs).
        ended: dict[str, list[float]] = {}
        for c in self._state.calls:
            if c.ended_at is not None:
                ended.setdefault(c.kind, []).append(c.ended_at)
        for c in self._state.calls:
            s = by_id.get(c.shard_id)
            if c.outcome != "success" or c.ended_at is None or s is None:
                continue
            est = self._cs(s) * max(self.settings.pools[s.kind].max_inputs, 1)
            if est <= 0:
                continue
            warm = any(c.spawned_at - self.scaledown_s <= t <= c.spawned_at
                       for t in ended.get(c.kind, ()))
            if not warm and self.settings.pools[s.kind].max_inputs > 1:
                # review 6 B3: a sibling still running at spawn time has a container with
                # free input slots; assume the call landed there (errs high, never optimistic)
                warm = any(o is not c and o.kind == c.kind and o.spawned_at < c.spawned_at
                           and (o.ended_at is None or o.ended_at >= c.spawned_at)
                           for o in self._state.calls)
            startup = 0.0 if warm else self._startup(s.kind)
            work = max(c.ended_at - c.spawned_at - startup, 0.0)
            ratios.setdefault(s.kind, []).append(work / est)
        for kind, rs in ratios.items():
            cal[kind] = max(cal.get(kind, 1.0), sum(rs) / len(rs))
        return cal

    def _remaining(self, now: float, cal: Mapping[str, float]) -> dict[str, float]:
        """call_id -> remaining $ of each live call (estimate less elapsed, never negative)."""
        if self._state is None:
            return {}
        by_id = self._shard_map()
        out: dict[str, float] = {}
        for c in self._state.live():
            pool = self.settings.pools[c.kind]  # type: ignore[index]
            mi = max(pool.max_inputs, 1)
            s = by_id.get(c.shard_id)
            est_cs = self._cs(s) * cal.get(c.kind, 1.0) if s else 0.0
            wall = self._startup(c.kind) + est_cs * mi
            left = max(wall - (now - c.spawned_at), 0.0)
            out[c.call_id] = left / mi * pool_rate(self.rates, self.settings, c.kind)
        return out

    def burn_per_s(self) -> float:
        """$/s the live calls are burning now (``ceil(calls / max_inputs)`` containers/kind)."""
        if self._state is None:
            return 0.0
        n: dict[str, int] = {}
        for c in self._state.live():
            n[c.kind] = n.get(c.kind, 0) + 1
        out = 0.0
        for kind, k in n.items():
            pool = self.settings.pools[kind]  # type: ignore[index]
            out += math.ceil(k / max(pool.max_inputs, 1)) * pool_rate(self.rates, self.settings,
                                                                      kind)
        return out

    # spend / projection ---------------------------------------------------------------------
    def spend_report(self, now: float | None = None) -> SpendReport:
        now = self.clock() if now is None else now
        self._refresh(now)
        receipts = 0.0
        for p, r in self._receipts.items():
            c = max(container_cost(r, self.rates, now, stale_after_s=self.stale_after_s),
                    self._floor.get(p, 0.0))
            self._floor[p] = c  # spend never goes down (stale heartbeat, clock skew)
            receipts += c
        calls = 0.0
        live = 0
        if self._state is not None:
            calls = calls_cost(self._state.calls, self.settings, self.rates, now,
                               scaledown_s=self.scaledown_s, warm=self.settings.min_containers,
                               drains=self._state.drains, prewarms=self._state.prewarms)
            live = len(self._state.live())
            if self._state.calls and (self.orchestrator_cores or self.orchestrator_memory_gib):
                t0 = min(c.spawned_at for c in self._state.calls)
                calls += max(now - t0, 0.0) * self.rates.per_second(
                    None, 0, self.orchestrator_cores, self.orchestrator_memory_gib,
                    preemptible=self.orchestrator_preemptible)
        return SpendReport(receipts, calls, self._billing_now(now), len(self._known), live)

    def projection(self, in_flight: Mapping[str, int]) -> Projection:
        now = self.clock()
        spent = self.spend_report(now).total_usd
        skip = self._done | set(self.quarantined())
        pending = [s for s in self.shards() if s.shard_id not in skip]
        cal = self.effective_calibration()
        remaining: dict[str, list[float]] = {}
        if self._state is not None:
            rem = self._remaining(now, cal)
            for c in self._state.live():
                remaining.setdefault(c.kind, []).append(rem[c.call_id])
        startup = self.startup_s if self.startup_s is not None else {
            k: self._startup(k) for k in {s.kind for s in pending}}
        return project(spent=spent, pending=pending, in_flight=in_flight, settings=self.settings,
                       rates=self.rates, startup_s=startup, calibration=cal,
                       in_flight_fraction=self.in_flight_fraction, estimator=self.estimator,
                       live_remaining_usd=remaining)

    def admit(self, shard: Shard, in_flight: Mapping[str, int]) -> BudgetDecision:
        p = self.projection(in_flight)
        cal = self.effective_calibration()
        this = self._cs(shard) * cal.get(shard.kind, 1.0) * pool_rate(self.rates, self.settings,
                                                                      shard.kind)
        if self.policy == "projection":
            need = p.total_usd
        else:
            need = p.spend_usd + p.in_flight_usd + this
        detail: dict[str, Any] = {**p.as_dict(), "limit_usd": round(self.limit_usd, 4),
                                  "budget_usd": self.settings.budget_usd,
                                  "safety_margin": self.safety_margin, "policy": self.policy,
                                  "needed_usd": round(need, 4), "rates_source": self.rates.source,
                                  "estimate_source": getattr(self.estimator, "source", "?"),
                                  "calibration": {k: round(v, 4) for k, v in cal.items()}}
        if need > self.limit_usd:
            return BudgetDecision(
                False,
                f"projected ${need:,.2f} exceeds ${self.limit_usd:,.2f} "
                f"(budget ${self.settings.budget_usd:,.2f} less {self.safety_margin:.0%} margin)",
                detail)
        return BudgetDecision(True, "", detail)

    # hard cap -------------------------------------------------------------------------------
    def overspend(self, state: BudgetState) -> list[str]:
        """Call ids the orchestrator must cancel now to stay within ``budget_usd``.

        Observes ``state`` first. Empty when spend + remaining in-flight estimates + one tick of
        burn (``tick_reserve_s``) fits under the hard cap. Otherwise the live calls with the
        largest remaining cost go first; once spend + one tick of burn alone reaches the cap,
        every live call goes. The caller writes ``budget_stop`` (so nothing respawns) and then
        cancels the calls; :attr:`last_overspend` holds the numbers for its event."""
        self.observe(state)
        now = state.now
        spent = self.spend_report(now).total_usd
        rem = self._remaining(now, self.effective_calibration())
        reserve = self.burn_per_s() * self.tick_reserve_s
        cap = self.settings.budget_usd
        self.last_overspend = {"spend_usd": round(spent, 4), "reserve_usd": round(reserve, 4),
                               "in_flight_remaining_usd": round(sum(rem.values()), 4),
                               "cap_usd": cap}
        if not rem or spent + reserve + sum(rem.values()) <= cap:
            return []
        if spent + reserve >= cap:
            return sorted(rem)
        left = sum(rem.values())
        out: list[str] = []
        for cid in sorted(rem, key=lambda cid: (-rem[cid], cid)):
            if spent + reserve + left <= cap:
                break
            out.append(cid)
            left -= rem[cid]
        return out


# ----------------------------------------------------------------------------------------------
# Native Modal Environment budget (backstop)
# ----------------------------------------------------------------------------------------------


def environment_budget_usd(settings: RunSettings) -> float:
    """The backstop value: ``budget × environment_budget_factor``, rounded up to the cent."""
    return math.ceil(settings.budget_usd * settings.environment_budget_factor * 100 - 1e-9) / 100


def environment_budget_instruction(environment: str, amount_usd: float) -> str:
    return (f"Set a native Modal Environment budget of ${amount_usd:,.2f} on environment "
            f"{environment!r}: modal.com -> Settings -> Environments -> {environment} -> Budget "
            "(Team plan or higher; covers compute only). The Modal Python SDK has no call for "
            "this yet, so flora cannot set it for you.")


@dataclass(frozen=True)
class BackstopResult:
    applied: bool
    amount_usd: float
    message: str


EnvironmentBudgetSetter = Callable[[str, float], None]


def find_environment_budget_setter() -> EnvironmentBudgetSetter | None:
    """A setter if the installed Modal client exposes one; ``None`` with Modal 1.6.2."""
    try:
        import modal
    except ImportError:
        return None
    env_cls = getattr(modal, "Environment", None)
    for name in ("set_budget", "update_budget"):
        fn = getattr(env_cls, name, None)
        if callable(fn):
            def setter(env: str, amount: float, _fn: Any = fn) -> None:
                _fn(env, amount)
            return setter
    return None


def apply_environment_budget(environment: str, settings: RunSettings, *,
                             setter: EnvironmentBudgetSetter | None = None,
                             discover: bool = True) -> BackstopResult:
    amount = environment_budget_usd(settings)
    setter = setter or (find_environment_budget_setter() if discover else None)
    if setter is None:
        return BackstopResult(False, amount, environment_budget_instruction(environment, amount))
    try:
        setter(environment, amount)
    except Exception as exc:
        return BackstopResult(False, amount, f"setting the Environment budget failed ({exc!r}). "
                              + environment_budget_instruction(environment, amount))
    return BackstopResult(True, amount,
                          f"Modal Environment {environment!r} budget set to ${amount:,.2f}")


def budget_summary(environment: str, settings: RunSettings, backstop: BackstopResult, *,
                   gate: Any = None) -> dict[str, Any]:
    """The budget section of a run summary: the gate's numbers and the Environment backstop,
    with the manual instruction whenever flora could not set it (Modal 1.6.2: always)."""
    out: dict[str, Any] = {
        "budget_usd": settings.budget_usd,
        "environment": environment,
        "environment_budget_usd": backstop.amount_usd,
        "environment_budget_applied": backstop.applied,
        "environment_budget_instruction": None if backstop.applied else backstop.message,
    }
    if isinstance(gate, ProjectionBudgetGate):
        rep = gate.spend_report()
        out.update({"gate_limit_usd": round(gate.limit_usd, 4), **rep.as_dict(),
                    "spend_usd": round(rep.total_usd, 4), "billing_error": gate.billing_error})
    return out
