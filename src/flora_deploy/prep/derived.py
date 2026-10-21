"""The derived weights ``flora prepare`` builds (``modal_app/convert_app.py``) and its plan.

No Modal here: :func:`plan_derived` takes a ``read`` callable that returns the bytes of a file
in the ``flora-weights`` Volume (``None`` when absent), so a receipt is checked without starting
a container. A build is skipped when its receipt passes the same reuse test the converter
applies (``prep/convert_captioner.convert``, ``prep/merge_tempseg.merge``); the converter still
re-checks on Modal, so the local test only saves a container start.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import derived_path
from flora_deploy.prep import convert_captioner as cc
from flora_deploy.prep import merge_tempseg as mt


@dataclass(frozen=True)
class DerivedBuild:
    name: str  # receipt name (presets' ``derived_weights``, flora.yaml ``weights.derived``)
    dirname: str  # ``layout.derived_path`` name and ``pins.DERIVED_PINS`` key
    source_key: str  # PINS_BY_KEY key the directory is versioned by
    function: str  # ``convert_app`` function that builds it
    command: str  # the manual equivalent
    # Container requests (``convert_app``; tests/unit/test_prep_derived.py keeps them equal) and
    # the container time measured for one build (docs/captioner-bf16.md, docs/tempseg-merge.md),
    # rounded up for image start: the dry-run estimate.
    cpu: float
    memory_mib: int
    est_seconds: float
    # (receipt field path, expected value): the converter's reuse test.
    reuse_checks: tuple[tuple[tuple[str, ...], Any], ...]

    def receipt_path(self) -> str:
        """Path of ``derived_receipt.json`` inside the ``flora-weights`` Volume."""
        root = derived_path(self.dirname, PINS_BY_KEY[self.source_key], PurePosixPath("/"))
        return str(root / cc.DERIVED_RECEIPT)

    def estimate_usd(self) -> float:
        from flora_deploy.jobs.budget import pinned_rates

        per_s = pinned_rates().per_second(None, 0, self.cpu, self.memory_mib / 1024)
        return per_s * self.est_seconds

    def stale_reason(self, receipt: Any) -> str | None:
        """Why ``receipt`` (parsed JSON) cannot be reused; ``None`` if it can."""
        if not isinstance(receipt, dict) or receipt.get("ok") is not True:
            return "receipt is not a passing one"
        for path, want in self.reuse_checks:
            got: Any = receipt
            for k in path:
                got = got.get(k) if isinstance(got, dict) else None
            if got != want:
                return f"receipt {'.'.join(path)} is {got!r}, need {want!r}"
        return None


def _rev(key: str) -> str:
    return PINS_BY_KEY[key].revision


DERIVED_BUILDS: dict[str, DerivedBuild] = {
    "tempseg_merged": DerivedBuild(
        name="tempseg_merged", dirname=mt.DERIVED_NAME, source_key=mt.SOURCE_KEY,
        function="merge_tempseg",
        command="uv run modal run src/flora_deploy/modal_app/convert_app.py::tempseg --step merge",
        cpu=8.0, memory_mib=32 * 1024, est_seconds=180.0,
        reuse_checks=((("merger", "version"), mt.MERGER_VERSION),
                      (("source", "pin", "revision"), _rev(mt.SOURCE_KEY)),
                      (("base", "pin", "revision"), _rev(mt.BASE_KEY)))),
    "captioner_bf16": DerivedBuild(
        name="captioner_bf16", dirname=cc.DERIVED_NAME, source_key=cc.SOURCE_KEY,
        function="convert",
        command="uv run modal run src/flora_deploy/modal_app/convert_app.py --step convert",
        cpu=8.0, memory_mib=56 * 1024, est_seconds=360.0,
        reuse_checks=((("converter", "version"), cc.CONVERTER_VERSION),
                      (("source", "pin", "revision"), _rev(cc.SOURCE_KEY)))),
}

Action = Literal["reuse", "build", "rebuild"]


@dataclass(frozen=True)
class DerivedStep:
    build: DerivedBuild
    action: Action
    reason: str
    receipt_sha256: str | None  # of the receipt on the Volume, when it is reused
    est_usd: float  # 0 when reused

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.build.name, "kind": "derived", "action": self.action,
                "reason": self.reason, "receipt_sha256": self.receipt_sha256,
                "est_usd": round(self.est_usd, 4), "path": self.build.receipt_path()}


def receipt_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def plan_derived(names: list[str], read: Callable[[str], bytes | None], *,
                 force: bool = False) -> list[DerivedStep]:
    """One step per name: reuse a passing receipt on the Volume, else build (``force``:
    rebuild). A ``read`` that raises (no Modal credentials, no Volume yet) plans a build."""
    steps = []
    for name in names:
        b = DERIVED_BUILDS[name]
        try:
            raw = read(b.receipt_path())
        except Exception as e:  # noqa: BLE001 - reported as the reason
            raw, why = None, f"could not read the receipt ({type(e).__name__}: {e})"
        else:
            why = "no receipt on the Volume"
        if raw is not None:
            try:
                why = b.stale_reason(json.loads(raw)) or ""
            except ValueError:
                why = "receipt is not valid JSON"
            if not why and not force:
                steps.append(DerivedStep(b, "reuse", "passing receipt on the Volume",
                                         receipt_sha256(raw), 0.0))
                continue
        action: Action = "rebuild" if force and raw is not None else "build"
        steps.append(DerivedStep(b, action, why or "--force-derived", None, b.estimate_usd()))
    return steps


def summarize(steps: list[DerivedStep]) -> Mapping[str, Any]:
    todo = [s for s in steps if s.action != "reuse"]
    return {"build": [s.build.name for s in todo],
            "reuse": [s.build.name for s in steps if s.action == "reuse"],
            "est_usd": round(sum(s.est_usd for s in todo), 4)}
