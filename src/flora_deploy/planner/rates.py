"""Measured per-kind throughput rates (``rates.json``) with provenance and error bars.

A rate is *container-seconds per second of video* for one shard kind of one preset, at a given
episode concurrency per container and camera count, plus the per-container startup time. Every
entry carries ``status`` (measured | projected | placeholder | stale), ``provenance``, ``date``
and a relative ``uncertainty``, plus optionally the ``image`` it was measured on and the worker
features it ``requires`` (e.g. ``gpu_thread``). ``flora estimate`` (later) writes a file of the
same shape.

Lookup is most-specific-first: an exact ``preset`` beats ``"*"``, exact ``cameras`` beats
``"*"``, and among concurrencies the largest one not above the request is used (so a kind
without concurrency entries, like the batched rewarder, just uses its ``concurrency: 1`` rate).
``stale`` entries (measured on another image, or superseded) are never used. An entry that
``requires`` a feature is used only when it is enabled, and one that ``excludes`` a feature only
when it is not. ``features=None`` (the default, used by ``jobs.budget``) means the **deployed**
configuration's features, :data:`DEPLOYED_FEATURES`; pass ``()`` for none. At equal concurrency
a measured entry beats a projected one, and only then does the entry that uses more enabled
features win (review 5 M-C: a projection must never displace a measurement of the configuration
that actually runs).

Camera scaling (``camera_scaling`` in ``rates.json``, :class:`CameraScaling`): per camera layout
(views per episode) and shard kind, a multiplier on a rate that was not measured on that layout
(its ``cameras`` is ``"*"`` or another count). It prices the longer prompts of multi-view
episodes until per-layout rates are measured. The table carries its own ``status``
(``estimate`` | ``measured``), ``source`` and ``date``; a layout or kind it does not list is
priced at 1x.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

STATUS_RANK = {"measured": 0, "projected": 1, "placeholder": 2, "stale": 3}
Features = frozenset[str] | set[str] | tuple[str, ...]

# Worker features the deployed configuration runs with. ``gpu_thread``: the exact worker's
# persistent generate thread, ``stages.reference.ExactProfile.gpu_thread`` (on by default since
# P1a-5s; the factory never disables it). ``long_lived_container`` (warm Captioner graph cache,
# overlapping shards) is *not* deployed: rates that need it are projections.
DEPLOYED_FEATURES: tuple[str, ...] = ("gpu_thread",)


def resolve_features(features: Features | None) -> frozenset[str]:
    return frozenset(DEPLOYED_FEATURES if features is None else features)


@dataclass(frozen=True)
class RateEntry:
    preset: str  # preset name or "*"
    kind: str
    cameras: int | str  # 1, 3 or "*"
    concurrency: int
    container_s_per_video_s: float
    uncertainty: float
    status: str
    provenance: str
    date: str
    image: str | None = None  # the image it was measured / projected on
    requires: tuple[str, ...] = ()  # worker features the rate depends on
    excludes: tuple[str, ...] = ()  # features under which the rate is invalid (measured without)
    source: str | None = None  # where the measurement lives (doc / run id); see MEASURED_SLOTS
    n: int | None = None  # samples (jobs) behind a measured rate; shown with it (n=1: ±50%+)

    def usable(self, features: Features | None = None) -> bool:
        f = resolve_features(features)
        return (self.status != "stale" and set(self.requires) <= f
                and not set(self.excludes) & f)


@dataclass(frozen=True)
class StartupEntry:
    preset: str
    kind: str  # exact kind, "reward.*" style prefix, or "*"
    startup_s: float
    uncertainty: float
    status: str
    provenance: str
    date: str
    source: str | None = None
    n: int | None = None  # containers behind a measured cold start
    # The compile cache (torch.compile / Triton / CUDA graphs) a measured startup was timed
    # with: "cold" (compiled from scratch) or "warm" (loaded a seeded cache); None when not
    # recorded (projections, placeholders).
    compile_cache: str | None = None


# Rows a later measurement wave flips from ``placeholder`` to ``measured`` by editing rates.json
# only (rate, uncertainty, status, source, date), by (preset, kind). Their status must be
# ``placeholder`` or ``measured``, and a ``measured`` one must name its ``source`` and ``date``.
MEASURED_SLOTS: tuple[tuple[str, str], ...] = (("balanced", "segment"), ("balanced", "caption"))
SLOT_STATUSES: tuple[str, ...] = ("placeholder", "measured")
COMPILE_CACHE_STATES: tuple[str | None, ...] = ("cold", "warm", None)


class RatesFileError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RatesFileError(msg)


CAMERA_SCALING_STATUSES: tuple[str, ...] = ("estimate", "measured")


@dataclass(frozen=True)
class CameraScaling:
    """``rates.json`` ``camera_scaling``: container-s per video-s multipliers per camera layout
    and shard kind, for rates not measured on that layout. ``layouts``: ``((cameras,
    ((kind, multiplier), ...)), ...)``, sorted."""

    status: str  # estimate | measured
    source: str
    date: str
    layouts: tuple[tuple[int, tuple[tuple[str, float], ...]], ...]
    derivation: str = ""

    def factors(self, cameras: int) -> dict[str, float]:
        """Kind -> multiplier for ``cameras``-view episodes (empty: the layout is not listed)."""
        return dict(dict(self.layouts).get(cameras, ()))

    def factor(self, kind: str, cameras: int) -> float:
        return self.factors(cameras).get(kind, 1.0)

    @classmethod
    def from_dict(cls, d: Any, *, source: str) -> CameraScaling:
        where = f"{source}: camera_scaling"
        _check(isinstance(d, dict), f"{where} must be an object")
        _check(d.get("status") in CAMERA_SCALING_STATUSES,
               f"{where}.status {d.get('status')!r}; use {' | '.join(CAMERA_SCALING_STATUSES)}")
        _check(bool(d.get("source") and d.get("date")), f"{where} needs `source` and `date`")
        raw = d.get("layouts")
        _check(isinstance(raw, dict) and bool(raw), f"{where}.layouts must be a non-empty object")
        layouts = []
        for cams, kinds in raw.items():
            _check(str(cams).isdigit() and int(cams) >= 2,
                   f"{where}.layouts: {cams!r} is not a camera count >= 2")
            _check(isinstance(kinds, dict) and all(
                isinstance(v, int | float) and v > 0 for v in kinds.values()),
                f"{where}.layouts.{cams}: multipliers must be positive numbers")
            layouts.append((int(cams), tuple(sorted((k, float(v)) for k, v in kinds.items()))))
        return cls(str(d["status"]), str(d["source"]), str(d["date"]), tuple(sorted(layouts)),
                   derivation=str(d.get("derivation", "")))


def _check_slots(rates: list[RateEntry], source: str) -> None:
    for preset, kind in MEASURED_SLOTS:
        for e in (r for r in rates if r.preset == preset and r.kind == kind):
            _check(e.status in SLOT_STATUSES,
                   f"{source}: {preset} {kind} rate status {e.status!r}; use "
                   f"{' | '.join(SLOT_STATUSES)}")
            if e.status == "measured":
                _check(bool(e.source and e.date),
                       f"{source}: a measured {preset} {kind} rate needs `source` and `date`")


@dataclass(frozen=True)
class RateTable:
    rates: tuple[RateEntry, ...]
    startup: tuple[StartupEntry, ...]
    source: str = "planner/rates.json"
    updated: str = ""
    camera_scaling: CameraScaling | None = None

    # loading -----------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str) -> RateTable:
        _check(isinstance(data, dict) and data.get("schema") == 1,
               f"{source}: expected a rates file with schema 1")
        rates, startup = [], []
        for i, r in enumerate(data.get("rates", [])):
            try:
                e = RateEntry(**{**r, "requires": tuple(r.get("requires") or ()),
                                 "excludes": tuple(r.get("excludes") or ())})
            except TypeError as exc:
                raise RatesFileError(f"{source}: rates[{i}]: {exc}") from None
            _check(e.container_s_per_video_s > 0, f"{source}: rates[{i}] must be positive")
            _check(e.concurrency >= 1, f"{source}: rates[{i}].concurrency must be >= 1")
            _check(e.uncertainty >= 0, f"{source}: rates[{i}].uncertainty must be >= 0")
            _check(e.status in STATUS_RANK, f"{source}: rates[{i}].status {e.status!r}")
            rates.append(e)
        for i, s in enumerate(data.get("startup", [])):
            try:
                e2 = StartupEntry(**s)
            except TypeError as exc:
                raise RatesFileError(f"{source}: startup[{i}]: {exc}") from None
            _check(e2.startup_s >= 0, f"{source}: startup[{i}] must be >= 0")
            _check(e2.compile_cache in COMPILE_CACHE_STATES,
                   f"{source}: startup[{i}].compile_cache {e2.compile_cache!r}; use cold | "
                   "warm or leave it out")
            startup.append(e2)
        _check_slots(rates, source)
        cs = data.get("camera_scaling")
        return cls(tuple(rates), tuple(startup), source=source, updated=data.get("updated", ""),
                   camera_scaling=None if cs is None else CameraScaling.from_dict(cs,
                                                                                source=source))

    @classmethod
    def load(cls, path: str | Path | None = None) -> RateTable:
        if path is None:
            text = resources.files("flora_deploy.planner").joinpath("rates.json").read_text(
                encoding="utf-8")
            return cls.from_dict(json.loads(text), source="planner/rates.json")
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RatesFileError(f"{p}: not valid JSON: {exc}") from None
        return cls.from_dict(data, source=str(p))

    # lookup ------------------------------------------------------------------------------
    def features(self) -> set[str]:
        """Every feature some entry requires."""
        return {f for e in self.rates for f in e.requires}

    def _candidates(self, preset: str, kind: str, cameras: int,
                    features: Features | None = None) -> list[RateEntry]:
        out = [e for e in self.rates if e.kind == kind and e.preset in (preset, "*")
               and e.usable(features)]
        if not out:
            return []
        best_p = min(0 if e.preset == preset else 1 for e in out)
        out = [e for e in out if (0 if e.preset == preset else 1) == best_p]
        exact = [e for e in out if e.cameras == cameras]
        if exact:
            return exact
        wild = [e for e in out if e.cameras == "*"]
        if wild:
            return wild
        # nearest measured camera count (ties: the larger, which is the costlier one)
        near = min(out, key=lambda e: (abs(int(e.cameras) - cameras), -int(e.cameras)))
        return [e for e in out if e.cameras == near.cameras]

    def concurrencies(self, preset: str, kind: str, cameras: int,
                      features: Features | None = None) -> list[int]:
        return sorted({e.concurrency for e in self._candidates(preset, kind, cameras,
                                                                features)})

    def rate(self, preset: str, kind: str, cameras: int, concurrency: int = 1,
             features: Features | None = None) -> RateEntry:
        """The entry for ``concurrency`` (largest not above it, else the smallest), measured
        before projected before placeholder at that concurrency, then the most features used.
        ``features=None``: :data:`DEPLOYED_FEATURES`."""
        cands = self._candidates(preset, kind, cameras, features)
        if not cands:
            raise KeyError(f"no rate for preset {preset!r} kind {kind!r} in {self.source}")
        below = [e for e in cands if e.concurrency <= concurrency]
        if below:
            return max(below, key=lambda e: (e.concurrency, -STATUS_RANK[e.status],
                                             len(e.requires)))
        return min(cands, key=lambda e: (e.concurrency, STATUS_RANK[e.status],
                                         -len(e.requires)))

    def camera_factor(self, kind: str, cameras: int, entry: RateEntry) -> float:
        """The :class:`CameraScaling` multiplier for ``entry`` priced on ``cameras``-view
        episodes: 1.0 when the entry was measured on exactly that layout, or the table has no
        multiplier for the layout and kind."""
        if self.camera_scaling is None or entry.cameras == cameras:
            return 1.0
        return self.camera_scaling.factor(kind, cameras)

    def startup_for(self, preset: str, kind: str, *, warm: bool = False) -> StartupEntry:
        """The startup row of ``(preset, kind)``: exact kind before ``reward.*`` style prefixes
        before ``"*"``, then exact preset before ``"*"``.

        ``warm=False`` (the default, what ``jobs.budget`` charges) never returns a
        ``compile_cache: warm`` row: a container is priced as if it compiles. ``warm=True``
        (a published compile cache exists for the pool's key, or a projection assumes one:
        :func:`startup_cache`) prefers the best ``warm`` row and falls back to the default
        lookup when the kind has none."""
        if warm:
            hit = self._startup_lookup(preset, kind, lambda e: e.compile_cache == "warm")
            if hit is not None:
                return hit
        hit = self._startup_lookup(preset, kind, lambda e: e.compile_cache != "warm")
        if hit is None:
            return StartupEntry(preset, kind, 0.0, 0.0, "placeholder",
                                "no startup entry; startup not charged", "")
        return hit

    def _startup_lookup(self, preset: str, kind: str,
                        keep: Callable[[StartupEntry], bool]) -> StartupEntry | None:
        def score(e: StartupEntry) -> tuple[int, int] | None:
            p = 0 if e.preset == preset else 1 if e.preset == "*" else None
            if p is None:
                return None
            if e.kind == kind:
                k = 0
            elif e.kind.endswith(".*") and kind.startswith(e.kind[:-1]):
                k = 1
            elif e.kind == "*":
                k = 2
            else:
                return None
            return (k, p)

        scored = [(s, e) for e in self.startup if keep(e) and (s := score(e)) is not None]
        if not scored:
            return None
        return min(scored, key=lambda t: t[0])[1]


def startup_cache(kind: str, *, warm_kinds: Collection[str] = (),
                  assume_warm: bool = False) -> bool:
    """Whether ``kind``'s containers are priced warm: a compile cache is published for its key
    (``warm_kinds``, e.g. from ``engines.compile_cache.published``) or the caller projects one
    (``assume_warm``). Never inferred: no evidence, cold."""
    return bool(assume_warm or kind in warm_kinds)
