"""Measured per-kind throughput rates (``rates.json``) with provenance and error bars.

A rate is *container-seconds per second of video* for one shard kind of one preset, at a given
episode concurrency per container and camera count, plus the per-container startup time. Every
entry carries ``status`` (measured | projected | placeholder), ``provenance``, ``date`` and a
relative ``uncertainty``. ``flora estimate`` (later) writes a file of the same shape.

Lookup is most-specific-first: an exact ``preset`` beats ``"*"``, exact ``cameras`` beats
``"*"``, and among concurrencies the largest one not above the request is used (so a kind
without concurrency entries, like the batched rewarder, just uses its ``concurrency: 1`` rate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

STATUS_RANK = {"measured": 0, "projected": 1, "placeholder": 2}


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


@dataclass(frozen=True)
class StartupEntry:
    preset: str
    kind: str  # exact kind, "reward.*" style prefix, or "*"
    startup_s: float
    uncertainty: float
    status: str
    provenance: str
    date: str


class RatesFileError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RatesFileError(msg)


@dataclass(frozen=True)
class RateTable:
    rates: tuple[RateEntry, ...]
    startup: tuple[StartupEntry, ...]
    source: str = "planner/rates.json"
    updated: str = ""

    # loading -----------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str) -> RateTable:
        _check(isinstance(data, dict) and data.get("schema") == 1,
               f"{source}: expected a rates file with schema 1")
        rates, startup = [], []
        for i, r in enumerate(data.get("rates", [])):
            try:
                e = RateEntry(**r)
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
            startup.append(e2)
        return cls(tuple(rates), tuple(startup), source=source, updated=data.get("updated", ""))

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
    def _candidates(self, preset: str, kind: str, cameras: int) -> list[RateEntry]:
        out = [e for e in self.rates if e.kind == kind and e.preset in (preset, "*")]
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

    def concurrencies(self, preset: str, kind: str, cameras: int) -> list[int]:
        return sorted({e.concurrency for e in self._candidates(preset, kind, cameras)})

    def rate(self, preset: str, kind: str, cameras: int, concurrency: int = 1) -> RateEntry:
        cands = self._candidates(preset, kind, cameras)
        if not cands:
            raise KeyError(f"no rate for preset {preset!r} kind {kind!r} in {self.source}")
        below = [e for e in cands if e.concurrency <= concurrency]
        if below:
            return max(below, key=lambda e: e.concurrency)
        return min(cands, key=lambda e: e.concurrency)

    def startup_for(self, preset: str, kind: str) -> StartupEntry:
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

        scored = [(s, e) for e in self.startup if (s := score(e)) is not None]
        if not scored:
            return StartupEntry(preset, kind, 0.0, 0.0, "placeholder",
                                "no startup entry; startup not charged", "")
        return min(scored, key=lambda t: t[0])[1]
