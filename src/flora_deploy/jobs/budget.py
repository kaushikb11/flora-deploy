"""Budget accounting and the orchestrator's budget gate (PLAN.md §8 "Budget").

* **Rates** (:class:`Rates`): $/s per GPU type, per physical core and per GiB of memory, plus a
  region multiplier and the non-preemptible CPU/memory multiplier. Live prices come from
  ``modal.Workspace.from_context().billing.rates()`` through an injectable provider
  (:func:`load_rates`); without one, or if the call fails, the pinned :data:`PINNED_PRICES` table
  is used and :attr:`Rates.source` says so.
* **Spend** (:func:`container_cost`, :func:`spend`): Σ container lifetime × (GPU × count + cores +
  memory) rates, over ``ContainerReceipt`` files. A container without ``stopped_utc`` is billed up
  to *now* while its heartbeat is fresh, and up to ``heartbeat + stale_after_s`` once the heartbeat
  is stale (presumed dead; Modal bills until the container is actually gone, so this is the
  conservative end of what we can know).
* **Projection** (:func:`project`): spend + in-flight + queued + future container starts.
  Per-shard container time comes from ``Shard.est_gpu_seconds`` (GPU-seconds of the shard's pool,
  so container-seconds = GPU-seconds / ``gpu_count``) times a calibration factor; in-flight shards
  are charged ``in_flight_fraction`` (default 1.0: their full estimate, conservative because the
  part already run is also in spend).
* **Gate** (:class:`ProjectionBudgetGate`): implements ``orchestrator.BudgetGate``. Denies a spawn
  when the projection exceeds ``budget × (1 − safety_margin)``; the orchestrator then writes
  ``budget_stop`` and drains.
* **Backstop** (:func:`environment_budget_usd`, :func:`apply_environment_budget`): the native Modal
  Environment budget at ``budget × environment_budget_factor``. Modal's Python SDK (1.6.2) has no
  call to set an Environment budget, so the default is the exact manual instruction.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from flora_deploy.jobs.orchestrator import BudgetDecision
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import ContainerReceipt, Shard

# ----------------------------------------------------------------------------------------------
# Rates
# ----------------------------------------------------------------------------------------------

#: Modal list prices in $/hour. Source: modal.com/pricing, as recorded in
#: ``research_notes/Plan review v2/modal_verification.md`` §15 (verified 2026-10-21) and
#: ``docs/exact-concurrency.md`` (H200 $0.001261/s, core $0.0000131/s, GiB $0.00000222/s).
PINNED_PRICES_SOURCE = "modal.com/pricing (pinned 2026-10-21, modal_verification.md §15)"
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
            self.gpu(gpu) * (gpu_count if gpu else 0)
            + cm * (cores * self.core_per_s + memory_gib * self.gib_per_s))

    def pool_per_second(self, pool: PoolSettings) -> float:
        return self.per_second(pool.gpu, pool.gpu_count, pool.cpu_cores, pool.memory_gib)

    def with_region(self, multiplier: float) -> Rates:
        return replace(self, region_multiplier=multiplier)


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


def shard_container_seconds(shard: Shard, pool: PoolSettings, calibration: float = 1.0) -> float:
    """``est_gpu_seconds`` is GPU-seconds of the shard's pool; containers hold ``gpu_count``."""
    return shard.est_gpu_seconds / max(pool.gpu_count, 1) * calibration


def project(*, spent: float, pending: Sequence[Shard], in_flight: Mapping[str, int],
            settings: RunSettings, rates: Rates, startup_s: Mapping[str, float] | None = None,
            calibration: Mapping[str, float] | None = None, in_flight_fraction: float = 1.0
            ) -> Projection:
    """Projected job cost. ``pending``: shards without a receipt (in-flight ones included).
    ``in_flight``: active calls per kind (as the orchestrator passes them). ``startup_s``:
    per-kind container startup seconds, charged for the containers still to be started
    (``max_containers`` minus those in flight) of kinds that still have queued work."""
    by_kind: dict[str, list[float]] = {}
    for s in pending:
        pool = settings.pools[s.kind]
        cal = (calibration or {}).get(s.kind, 1.0)
        by_kind.setdefault(s.kind, []).append(
            shard_container_seconds(s, pool, cal) * rates.pool_per_second(pool))
    in_flight_usd = queued_usd = startup_usd = 0.0
    for kind in sorted(set(by_kind) | set(in_flight)):
        costs = by_kind.get(kind, [])
        n_flight = min(in_flight.get(kind, 0), len(costs)) if costs else 0
        mean = sum(costs) / len(costs) if costs else 0.0
        in_flight_usd += n_flight * mean * in_flight_fraction
        queued_usd += sum(costs) - n_flight * mean
        n_queued = len(costs) - n_flight
        if n_queued > 0 and startup_s and kind in startup_s:
            pool = settings.pools[kind]  # type: ignore[index]
            running = math.ceil(in_flight.get(kind, 0) / max(pool.max_inputs, 1))
            new = min(max(pool.max_containers - running, 0),
                      math.ceil(n_queued / max(pool.max_inputs, 1)))
            startup_usd += new * startup_s[kind] * rates.pool_per_second(pool)
    return Projection(spent, in_flight_usd, queued_usd, startup_usd)


# ----------------------------------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------------------------------


@dataclass
class ProjectionBudgetGate:
    """``BudgetGate`` for the orchestrator.

    Reads container receipts and shard receipts from the store (refreshed at most every
    ``refresh_s``), projects the job cost and denies when it exceeds the limit
    ``budget_usd × (1 − safety_margin)``. ``policy="projection"`` (default) denies as soon as the
    *whole remaining job* no longer fits, so a job never starts work it cannot finish;
    ``policy="next_shard"`` only denies when spend + in-flight + this shard does not fit (use it to
    get partial results from an under-funded job)."""

    settings: RunSettings
    store: Store
    shards: Callable[[], Sequence[Shard]]  # effective shards (manifest + bisection children)
    rates: Rates = field(default_factory=pinned_rates)
    safety_margin: float = 0.05
    policy: str = "projection"
    startup_s: Mapping[str, float] = field(default_factory=dict)
    calibration: Mapping[str, float] = field(default_factory=dict)
    in_flight_fraction: float = 1.0
    stale_after_s: float = 300.0
    refresh_s: float = 60.0
    quarantined: Callable[[], Iterable[str]] = lambda: ()
    clock: Callable[[], float] = time.time
    _cache_at: float = field(default=-math.inf, init=False, repr=False)
    _receipts: list[ContainerReceipt] = field(default_factory=list, init=False, repr=False)
    _done: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.safety_margin < 1.0:
            raise ValueError("safety_margin must be in [0, 1)")
        if self.policy not in ("projection", "next_shard"):
            raise ValueError("policy must be 'projection' or 'next_shard'")

    @property
    def limit_usd(self) -> float:
        return self.settings.budget_usd * (1.0 - self.safety_margin)

    def _refresh(self, now: float) -> None:
        if now - self._cache_at < self.refresh_s:
            return
        self._receipts = read_container_receipts(self.store, self.settings.job_id)
        paths = JobPaths(self.settings.job_id)
        prefix = f"{paths.root}/{self.settings.config_hash}/receipts/"
        self._done = {p.rsplit("/", 1)[-1].removesuffix(".json")
                      for p in self.store.list(prefix) if p.endswith(".json")}
        self._cache_at = now

    def note_done(self, shard_id: str) -> None:
        """Optional hint (e.g. from the orchestrator's events) so a lagging list is not needed."""
        self._done.add(shard_id)

    def projection(self, in_flight: Mapping[str, int]) -> Projection:
        now = self.clock()
        self._refresh(now)
        spent = spend(self._receipts, self.rates, now, stale_after_s=self.stale_after_s).total_usd
        skip = self._done | set(self.quarantined())
        pending = [s for s in self.shards() if s.shard_id not in skip]
        return project(spent=spent, pending=pending, in_flight=in_flight, settings=self.settings,
                       rates=self.rates, startup_s=self.startup_s, calibration=self.calibration,
                       in_flight_fraction=self.in_flight_fraction)

    def admit(self, shard: Shard, in_flight: Mapping[str, int]) -> BudgetDecision:
        p = self.projection(in_flight)
        if self.policy == "projection":
            need = p.total_usd
        else:
            pool = self.settings.pools[shard.kind]
            this = shard_container_seconds(shard, pool, self.calibration.get(shard.kind, 1.0)
                                           ) * self.rates.pool_per_second(pool)
            need = p.spend_usd + p.in_flight_usd + this
        detail: dict[str, Any] = {**p.as_dict(), "limit_usd": round(self.limit_usd, 4),
                                  "budget_usd": self.settings.budget_usd,
                                  "safety_margin": self.safety_margin, "policy": self.policy,
                                  "needed_usd": round(need, 4), "rates_source": self.rates.source}
        if need > self.limit_usd:
            return BudgetDecision(
                False,
                f"projected ${need:,.2f} exceeds ${self.limit_usd:,.2f} "
                f"(budget ${self.settings.budget_usd:,.2f} less {self.safety_margin:.0%} margin)",
                detail)
        return BudgetDecision(True, "", detail)


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
