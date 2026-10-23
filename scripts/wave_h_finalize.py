"""Finalize Wave H-lite: turn the runbook's state and receipts into rates, certificates and docs.

Usage::

    uv run python scripts/wave_h_finalize.py            # dry run: print what would change
    uv run python scripts/wave_h_finalize.py --fetch    # first export the records (Volume read)
    uv run python scripts/wave_h_finalize.py --write    # apply

It reads ``reports/wave-h/state.json`` (``scripts/wave_h.py``), the records of the oracle, noise
and balanced jobs (``reports/wave-h/records/{exact,noise,balanced}/<job>/<stage>.jsonl``, which
``--fetch`` exports with ``flora results``) and the lab JSON (``reports/wave-h/balanced-<kind>
.json``). It updates, consistently:

1. **Rates** (``planner/rates.json``). Per (preset, kind, GPU), from the container receipts in
   each job's ``flora report``: container-seconds per video-second (busy seconds over video
   seconds; one sample per job), the cold start (one sample per container), and the error band
   ``max(25%, (max - min) / mean)`` with ``n`` and the range in the provenance. The ``balanced``
   ``segment`` / ``caption`` rows (``planner.rates.MEASURED_SLOTS``) flip from ``placeholder`` to
   ``measured`` with source and date, and their per-kind startup rows are written. It refuses
   (all or nothing for the slot) when a sample is too small to state a range: fewer than
   :data:`MIN_RATE_SAMPLES` jobs or containers, under :data:`MIN_VIDEO_S` video seconds, a
   report without the per-container receipt fields (:data:`RECEIPT_FIELDS`, from
   ``sdk.Job.report()``), or a worker on a GPU other than the preset's pool (that job's
   samples are dropped and named).
2. **Certificates** (``config/certificates/``). ``flora certify`` per balanced job (one
   embodiment per config) and stage, with the oracle, the noise run (where it repeated that job),
   the L0 JSON and the derived-weight content digests the run used (else the pinned ones),
   stamped with the run's own time so a re-run reproduces the same certificate ids. Metrics
   only; a ``fail`` is never written here. A certificate written earlier is replaced only by
   a new one with the same key; one whose inputs are missing now, or that certify refused,
   is kept.
3. **Report** (``reports/wave-h/report.md``, git-ignored): spend per step (tracked vs
   billed, with the billing lag), measured $/video-h for ``exact`` and ``balanced`` at list
   prices (formula shown), L0, Gate A with the ``dt_bias`` plugin recommendation (the rule
   in ``docs/balanced-engines.md``) and the certificate summary. Numbers only: no NVIDIA
   text, captions, prompts or episode ids.
4. **Docs**: the ``Results`` table of ``docs/balanced-engines.md`` and the ``Cost`` block of
   ``README.md`` (2-3 plain lines), each between ``<!-- wave-h:<name> -->`` /
   ``<!-- /wave-h:<name> -->`` markers. The script fills exactly those blocks and refuses when
   the markers are missing; it never inserts a section. Until the balanced rates are measured
   the README block is left as it is. When the rates change, the generated tables of
   ``docs/costs.md`` are regenerated from them (``scripts/cost_table.py``).

Idempotent: a second run with the same inputs changes nothing. Nothing here starts a container.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wave_h as wh  # noqa: E402 - the runbook's state format, caps and constants

from flora_deploy.validate import certfreeze as F  # noqa: E402

MIN_RATE_SAMPLES = 1  # jobs (rate) and containers (cold start) per (preset, kind, GPU)
MIN_VIDEO_S = 30.0  # video seconds behind a rate
UNCERTAINTY_FLOOR = 0.25  # the error band never claims better than +/-25% from this few samples
SINGLE_SAMPLE_FLOOR = 0.50  # one sample states no range: its band is +/-50%, stated with n=1
BILLING_MISMATCH = 0.25  # reconciled billed vs tracked: relative gap flagged as a mismatch
BILLING_MISMATCH_USD = 0.05  # ... and absolute gap
STEP_PRESET = {"exact_e2e": "exact", "noise": "exact", "balanced": "balanced"}
PRESET_KINDS = {"exact": ("segcap",), "balanced": ("segment", "caption")}
CERT_STAGES = ("segment", "caption")
DEFERRED = "deferred to the certified run"
LAB_KINDS = {"segment": "tempseg", "caption": "captioner"}
SETS = {"exact": "exact_e2e", "noise": "noise", "balanced": "balanced",
        # the stage-isolated caption job (J2) and an older noise run (reported only), as
        # ``wave_h.certify_view`` names their steps
        "isolated": "caption_iso", "repeat": "repeat"}
# (d)'s cold witness (review 13 W1): its jobs are under the balanced step's "witness_jobs"
WITNESS = ("witness", "balanced", "witness_jobs")
ISO_WITNESS = ("isolated_witness", "caption_iso", "witness_jobs")  # J2w
# Captioner calls in NVIDIA's ``model_calls`` (``stages.reference``): its token budget
CAPTIONER_OUTPUT_LIMIT = 512
DIGESTS = "digests.json"  # per side and job, beside its records: {uid: [hex64 | null per row]}
# ``flora certify --digests`` side names of the record sides
DIGEST_SIDES = {"exact": "exact", "noise": "noise", "isolated": "balanced",
                "isolated_witness": "witness", "repeat": "repeat"}
SEGCAP_DIGEST_SIDES = ("exact", "noise", "repeat")  # from the segcap audit lines
CAPTION_DIGEST_SIDES = ("isolated", "isolated_witness")  # from the caption audit rows
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FAMILY = re.compile(r"[a-z0-9_.-]{1,32}")
BLOCK_RESULTS = "results"
BLOCK_COST = "cost"
# Steps of a run that Modal billed on another workspace than the one the billing read covers:
# their billed delta there is $0 (or unattributed), which is neither reconciled nor a mismatch.
# Run 10221548f1b8: (b) and (f) ran on the first workspace; (d)'s first attempt was stopped
# there (credit low, its attempt note) and the run continued on a second one, where billing
# was read.
BILLED_ELSEWHERE: dict[str, frozenset[str]] = {
    "10221548f1b8": frozenset({"exact_e2e", "noise"})}
MISSING_HELP = {
    "too_few_samples": "at least {min_samples} candidate episodes of the pool (has {n})",
    "no_noise_run": "a noise run: the oracle repeated on at least {min_noise} of these "
                    "episodes (none for this pool)",
    "too_few_noise_samples": "at least {min_noise} noise-run episodes (has {noise_n})",
    "incomplete_coverage": "at most {max_missing} oracle episodes without a candidate "
                           "record (missing {missing})",
    "no_l0": "the L0 JSON of {lab} (input parity on the captured requests)",
    "metric_not_evaluable": "every metric evaluable ({metrics} had no value)",
    "divergence_above_noise_floor": "every check within its threshold (failing: {failing})",
    "l0_failed": "L0 passing on every captured request",
    "stale_l0": "the L0 JSON of {lab} measured under the current engine settings (re-run the "
                "lab: its engine hash is not the candidate's)",
    "engine_preempted": "no candidate record whose engine preempted while it ran "
                        "({preempted} did; lower max_num_seqs)",
    "preemptions_unknown": "the engine's preemption count on every candidate record "
                           "({unknown} without)",
    "too_few_witness_samples": "at least {min_noise} cold-witness episodes (has {witness_n})",
    "witness_not_forced_cold": "a cold witness whose records are all compile_cache cold:forced",
    "no_witness": "a cold witness: records from a container forced cold (compile_cache "
                  "cold:forced)",
}


@dataclass
class Paths:
    root: Path = ROOT
    state: Path | None = None
    records: Path | None = None
    lab: Path | None = None
    rates: Path | None = None
    certs: Path | None = None
    report: Path | None = None
    engines_doc: Path | None = None
    readme: Path | None = None
    manifest: Path | None = None
    costs_doc: Path | None = None
    limitations_doc: Path | None = None

    def __post_init__(self) -> None:
        r = self.root
        w = r / "reports" / "wave-h"
        self.state = self.state or w / "state.json"
        self.records = self.records or w / "records"
        self.lab = self.lab or w
        self.rates = self.rates or r / "src" / "flora_deploy" / "planner" / "rates.json"
        self.certs = self.certs or r / "src" / "flora_deploy" / "config" / "certificates"
        self.report = self.report or w / "report.md"  # git-ignored, never docs/
        self.engines_doc = self.engines_doc or r / "docs" / "balanced-engines.md"
        self.readme = self.readme or r / "README.md"
        self.manifest = self.manifest or w / "finalize.json"
        self.costs_doc = self.costs_doc or r / "docs" / "costs.md"
        self.limitations_doc = self.limitations_doc or r / "docs" / "limitations.md"

    def rel(self, p: str | Path) -> str:
        """``p`` as the committed files may name it: relative to the root, never a local
        absolute path (outside the root: its name only)."""
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return f"<outside the repository>/{p.name}"


# ---------------------------------------------------------------------------------------------
# Small helpers


def utc(ts: float | None, fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    return time.strftime(fmt, time.gmtime(ts)) if ts is not None else "-"


def usd(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"${x:,.{nd}f}"


def strip_gpu(g: str | None) -> str:
    return (g or "").rstrip("!")


def step_finished(st: wh.State, step: str) -> float | None:
    s = st.steps.get(step)
    if not s:
        return None
    done = [a.finished for a in s.attempts if a.finished is not None]
    return max(done) if done else None


def run_stamp(st: wh.State) -> float:
    """The run's own time: the last finished attempt of the record-producing steps."""
    stamps = [t for t in (step_finished(st, s) for s in SETS.values()) if t is not None]
    return max(stamps) if stamps else float(st.created)


def jobs_of(st: wh.State, step: str, key: str = "jobs") -> dict[str, dict[str, Any]]:
    s = st.steps.get(step)
    return dict((s.result.get(key) if s else None) or {})


# ---------------------------------------------------------------------------------------------
# 1. Rates


@dataclass
class Measurement:
    preset: str
    kind: str
    gpu: str
    samples: list[tuple[str, float, float]] = field(default_factory=list)  # (job, busy, video)
    startups: list[float] = field(default_factory=list)
    container_s: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def video_s(self) -> float:
        return round(sum(v for _, _, v in self.samples), 2)

    @property
    def rates(self) -> list[float]:
        return [b / v for _, b, v in self.samples if v > 0]

    @property
    def rate(self) -> float | None:
        """Busy container-s per video-s, rounded to 3 places (``planner/rates.json``)."""
        r = self.rate_exact
        return None if r is None else round(r, 3)

    @property
    def rate_exact(self) -> float | None:
        """:attr:`rate` unrounded: what costs are computed from (review 12 F9)."""
        v = sum(v for _, _, v in self.samples)
        return sum(b for _, b, _ in self.samples) / v if v > 0 else None

    @staticmethod
    def band(xs: Sequence[float]) -> float | None:
        if not xs:
            return None
        mean = sum(xs) / len(xs)
        if len(xs) == 1:
            return SINGLE_SAMPLE_FLOOR
        return round(max(UNCERTAINTY_FLOOR, (max(xs) - min(xs)) / mean if mean else 1.0), 2)

    @property
    def uncertainty(self) -> float | None:
        return self.band(self.rates)

    @property
    def startup_s(self) -> float | None:
        return round(sum(self.startups) / len(self.startups), 1) if self.startups else None

    @property
    def startup_uncertainty(self) -> float | None:
        return self.band(self.startups)

    @property
    def refusals(self) -> list[str]:
        out = list(self.problems)
        what = f"{self.preset} {self.kind} on {self.gpu}"
        if self.n < MIN_RATE_SAMPLES:
            out.append(f"{what}: {self.n} job sample(s); at least {MIN_RATE_SAMPLES}")
        if self.video_s < MIN_VIDEO_S:
            out.append(f"{what}: {self.video_s:.1f} video s behind the rate; at least "
                       f"{MIN_VIDEO_S:.0f} s")
        if len(self.startups) < MIN_RATE_SAMPLES:
            out.append(f"{what}: {len(self.startups)} cold-start sample(s); at least "
                       f"{MIN_RATE_SAMPLES} containers")
        return out

    def summary(self) -> dict[str, Any]:
        rs = self.rates
        return {"preset": self.preset, "kind": self.kind, "gpu": self.gpu, "n": self.n,
                "video_s": self.video_s, "container_s": round(self.container_s, 1),
                "container_s_per_video_s": self.rate,
                "range": [round(min(rs), 3), round(max(rs), 3)] if rs else None,
                "uncertainty": self.uncertainty, "startup_n": len(self.startups),
                "startup_s": self.startup_s,
                "startup_range": ([round(min(self.startups), 1), round(max(self.startups), 1)]
                                  if self.startups else None),
                "startup_uncertainty": self.startup_uncertainty,
                "refused": self.refusals}


def pool_gpu(preset: str, kind: str) -> str:
    from flora_deploy.config.presets import get_preset

    return get_preset(preset).pools[kind].gpu or ""


# The per-container receipt contract: ``flora report``'s ``spend.containers`` entries
# (``sdk.Job.report()``), which ``scripts/wave_h.py`` keeps as ``jobs.<job>.containers``. Each
# entry must carry these; anything less is refused by name, never measured as zero.
RECEIPT_FIELDS = ("kind", "startup_s", "busy_s", "seconds")
CONTRACT = ("Job.report() per-container receipts (spend.containers: "
            + ", ".join(RECEIPT_FIELDS) + ")")


def ran_on(c: Mapping[str, Any]) -> str | None:
    """The GPU a receipt's container ran on: ``gpu_seen`` (its shard receipts'
    ``nvidia-smi`` name, ``jobs.containers``) when known, else ``gpu`` (its pool's)."""
    return c.get("gpu_seen") or c.get("gpu")


def same_gpu(ran: str | None, pool: str) -> bool:
    """``ran`` (a receipt's GPU, possibly ``nvidia-smi``'s name) is the pool's GPU."""
    from flora_deploy.engines.offline_llm import EngineArgsError, normalize_gpu

    if not ran:
        return False
    try:
        return normalize_gpu(str(ran)) == normalize_gpu(pool)
    except EngineArgsError:
        return strip_gpu(ran) == strip_gpu(pool)


def job_video_s(r: Mapping[str, Any]) -> float | None:
    if r.get("video_s") is not None:
        return float(r["video_s"])
    vh = r.get("video_hours")
    if isinstance(vh, Mapping) and vh.get("done") is not None:
        return float(vh["done"]) * 3600.0
    return None


def measure(st: wh.State) -> dict[tuple[str, str], Measurement]:
    """(preset, kind) -> its measurement from the finished jobs' container receipts."""
    out: dict[tuple[str, str], Measurement] = {}
    for preset in PRESET_KINDS:
        for kind in PRESET_KINDS[preset]:
            out[(preset, kind)] = Measurement(preset, kind, strip_gpu(pool_gpu(preset, kind)))
    for step, preset in STEP_PRESET.items():
        for job, r in sorted(jobs_of(st, step).items()):
            if r.get("state") not in ("done", "completed_with_errors"):
                continue
            where = f"{step}.jobs.{job}"
            kinds = PRESET_KINDS[preset]
            video = job_video_s(r)
            containers = list(r.get("containers") or ())
            if not containers:
                for kind in kinds:
                    out[(preset, kind)].problems.append(
                        f"{where}: `flora report` lists no per-container receipts; the rates "
                        f"need {CONTRACT}")
                continue
            per: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
            broken = False
            for i, c in enumerate(containers):
                lacks = [f for f in RECEIPT_FIELDS if c.get(f) is None]
                if lacks:
                    broken = True
                    for kind in kinds:
                        out[(preset, kind)].problems.append(
                            f"{where}.containers[{i}]: receipt lacks "
                            + ", ".join(f"`{f}`" for f in lacks) + f" (the {CONTRACT})")
                    continue
                per.setdefault(str(c["kind"]), []).append((i, c))
            if broken:
                continue
            for kind in kinds:
                m = out[(preset, kind)]
                cs = per.get(kind, [])
                if not cs:
                    m.problems.append(f"{where}: no {kind} container receipt")
                    continue
                if video is None:
                    m.problems.append(f"{where}: no video seconds (`video_s` / "
                                      "`video_hours.done`)")
                    continue
                wrong = [(i, c) for i, c in cs if not same_gpu(ran_on(c), m.gpu)]
                if wrong:  # a worker on another GPU: this job's rates are not this pool's
                    m.problems += [f"{where}: {kind} rates refused: container "
                                   f"{c.get('container_id', i)} ran on {ran_on(c)!r}, not "
                                   f"the {preset} {kind} pool's {m.gpu}" for i, c in wrong]
                    continue
                m.samples.append((job, sum(float(c["busy_s"]) for _, c in cs), video))
                m.startups += [float(c["startup_s"]) for _, c in cs]
                m.container_s += sum(float(c["seconds"]) for _, c in cs)
    return out


def engine_image(preset: str, kind: str) -> str:
    from flora_deploy.config.presets import get_preset

    e = get_preset(preset).engines[kind]
    ver = ", ".join(f"{k} {v}" for k, v in sorted(e.version.items()))
    return f"{e.image} ({ver})" if ver else e.image


def updated_rates(data: Mapping[str, Any], ms: Mapping[tuple[str, str], Measurement], *,
                  run_id: str, date: str, source: str) -> dict[str, Any]:
    """``rates.json`` with the ``balanced`` slot rows measured (and their startup rows)."""
    from flora_deploy.planner.rates import MEASURED_SLOTS

    new = json.loads(json.dumps(data))
    for preset, kind in MEASURED_SLOTS:
        m = ms[(preset, kind)]
        lo, hi = min(m.rates), max(m.rates)
        band = (f"{m.rate} from one job: no range, so the error band is "
                f"±{SINGLE_SAMPLE_FLOOR:.0%}" if m.n == 1
                else f"{lo:.3f}-{hi:.3f} per job, {m.rate} over all; error band "
                     f"max({UNCERTAINTY_FLOOR:.0%}, range/mean)")
        prov = (f"Wave H-lite run {run_id}, `preset: {preset}` deployed with flora deploy/run: "
                f"n={m.n} job(s) ({m.video_s:.1f} video s) on {m.gpu}, busy container-s per "
                f"video-s {band}; excludes @enter (startup row) and idle ({source})")
        rows = [r for r in new["rates"] if r["preset"] == preset and r["kind"] == kind]
        if len(rows) != 1:
            raise SystemExit(f"rates.json: expected one {preset} {kind} row, found {len(rows)}")
        foreign = folded_elsewhere(rows[0])  # other runs' samples (a scale test): kept
        rows[0].pop("samples", None)
        rows[0].update(container_s_per_video_s=m.rate, uncertainty=m.uncertainty,
                       status="measured", source=source, date=date, n=m.n,
                       image=engine_image(preset, kind), provenance=prov)
        if foreign:
            rows[0].update(fold_samples(rows[0], foreign, kind="rate"))
        k = len(m.startups)
        spread = (f"{m.startups[0]:.0f} s, one container: no range, so the error band is "
                  f"±{SINGLE_SAMPLE_FLOOR:.0%}" if k == 1
                  else f"{min(m.startups):.0f}-{max(m.startups):.0f} s")
        srow = {"preset": preset, "kind": kind, "startup_s": m.startup_s,
                "uncertainty": m.startup_uncertainty, "status": "measured", "n": k,
                "provenance": (f"Wave H-lite run {run_id}: spawn to first attempt of n={k} "
                               f"fresh {m.gpu} container(s), {spread} ({source})"),
                "date": date, "source": source}
        existing = [r for r in new["startup"] if r["preset"] == preset and r["kind"] == kind
                    and r.get("compile_cache") != "warm"]  # the warm row is its own slot
        if existing:
            # the compile-cache state the startup was measured with is not in the state; keep
            # the one recorded beside the row (``wave_h.startup_basis`` reads it), so --write
            # stays idempotent
            if "compile_cache" in existing[0]:
                srow = {k: v for k, v in srow.items() if k != "n"} | {
                    "compile_cache": existing[0]["compile_cache"], "n": k}
                srow = {**{x: srow[x] for x in ("preset", "kind", "startup_s", "uncertainty",
                                                "status", "compile_cache", "n")}, **srow}
            foreign = folded_elsewhere(existing[0])
            existing[0].clear()
            existing[0].update(srow)
            if foreign:
                existing[0].update(fold_samples(existing[0], foreign, kind="startup"))
        else:
            star = next((i for i, r in enumerate(new["startup"])
                         if r["preset"] == preset and r["kind"] == "*"), len(new["startup"]))
            new["startup"].insert(star, srow)
    new["updated"] = max(str(new.get("updated") or ""), date)
    return new


# Samples folded in from later runs (``scripts/scale_test.py --finalize``). A measured row is one
# run's pooled figure. A later run adds its own samples beside it instead of replacing it. The
# row's first measurement becomes the ``base`` sample, kept verbatim: its value, ``n``, video
# seconds and per-job range, read back from the provenance this script writes. The pooled value,
# ``n`` and the band are recomputed over every sample. A re-fold from the same source replaces
# that source's samples only, so folding is idempotent. A later ``--write`` of this script
# re-pools the other sources' samples over its own new base.
VALUE_FIELD = {"rate": "container_s_per_video_s", "startup": "startup_s"}
_VIDEO_RE = re.compile(r"\(([\d.]+) video s\)")
_JOB_RANGE_RE = re.compile(r"(\d+\.\d+)-(\d+\.\d+) per job")
_STARTUP_RANGE_RE = re.compile(r"(\d+)-(\d+) s\b")


class FoldRefused(ValueError):
    pass


def folded_elsewhere(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The samples a row carries from other runs (everything but its ``base``)."""
    return [dict(x) for x in row.get("samples") or () if x.get("origin") != "base"]


def base_sample(row: Mapping[str, Any], *, kind: str) -> dict[str, Any] | None:
    """The row's own measurement as one sample (None: not measured, e.g. a placeholder)."""
    if row.get("status") != "measured":
        return None
    prov = str(row.get("provenance") or "")
    out: dict[str, Any] = {"origin": "base", "source": row.get("source"),
                           "date": row.get("date"), "n": int(row.get("n") or 1),
                           "value": float(row[VALUE_FIELD[kind]]),
                           "uncertainty": row.get("uncertainty"), "provenance": prov}
    if kind == "rate":
        v = _VIDEO_RE.search(prov)
        r = _JOB_RANGE_RE.search(prov)
        out.update(unit="job", video_s=float(v.group(1)) if v else None,
                   range=[float(r.group(1)), float(r.group(2))] if r else None)
    else:
        r = _STARTUP_RANGE_RE.search(prov)
        out.update(unit="container", compile_cache=row.get("compile_cache"),
                   range=[float(r.group(1)), float(r.group(2))] if r else None)
    return out


def fold_samples(row: Mapping[str, Any], add: Sequence[Mapping[str, Any]], *, kind: str
                 ) -> dict[str, Any]:
    """``row`` with ``add`` folded in as more samples (see the comment above). Rates are pooled
    weighted by video seconds (busy seconds over video seconds, as :class:`Measurement` does);
    startups are weighted by container. The band is ``max(25%, (max - min) / mean)`` over every
    sample value (a base sample with a per-job range contributes both ends), and ``n`` is the
    samples' total. Raises :class:`FoldRefused` when a rate sample has no video seconds."""
    field_ = VALUE_FIELD[kind]
    have = [dict(x) for x in row.get("samples") or ()]
    if not have:
        b = base_sample(row, kind=kind)
        have = [b] if b else []
    sources = {x.get("source") for x in add}
    keep = [x for x in have if x.get("origin") == "base" or x.get("source") not in sources]
    samples = keep + [{**dict(x), "origin": "fold"} for x in add]
    if not samples:
        raise FoldRefused(f"no samples for {row.get('preset')} {row.get('kind')}")
    if kind == "rate":
        lacks = [x.get("source") for x in samples if not x.get("video_s")]
        if lacks:
            raise FoldRefused(f"{row.get('preset')} {row.get('kind')}: rate sample(s) without "
                              f"video seconds ({lacks}); cannot pool them")
        w = [float(x["video_s"]) for x in samples]
    else:
        w = [float(x.get("n") or 1) for x in samples]
    pooled = sum(float(x["value"]) * wi for x, wi in zip(samples, w, strict=True)) / sum(w)
    values = [float(v) for x in samples for v in (x.get("range") or [x["value"]])]
    n = sum(int(x.get("n") or 1) for x in samples)
    mean = sum(values) / len(values)
    band = (SINGLE_SAMPLE_FLOOR if n == 1 else
            round(max(UNCERTAINTY_FLOOR, (max(values) - min(values)) / mean if mean else 1.0),
                  2))
    base = next((x for x in samples if x.get("origin") == "base"), None)
    folds = [x for x in samples if x.get("origin") != "base"]
    unit = "weighted by video s" if kind == "rate" else "weighted by container"
    fmt = "{:.3f}" if kind == "rate" else "{:.0f} s"
    note = (f"pooled over n={n} sample(s), {unit}: "
            + (f"base {base['source']} n={base['n']}; " if base else "")
            + f"{len(folds)} folded from "
            + ", ".join(sorted({str(x.get('source')) for x in folds}))
            + f"; values {fmt.format(min(values))}-{fmt.format(max(values))}, error band "
              f"max({UNCERTAINTY_FLOOR:.0%}, range/mean)")
    return {**dict(row), field_: round(pooled, 3 if kind == "rate" else 1),
            "uncertainty": band, "status": "measured", "n": n,
            "date": max(str(x.get("date") or "") for x in samples) or row.get("date"),
            "source": (base or folds[0]).get("source"),
            "provenance": f"{base['provenance']} | {note}" if base else note,
            "samples": samples}


def dump_json(d: Any) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------------------------
# 2. Certificates


def derived_content_default() -> dict[str, str]:
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
    from flora_deploy.config.receipts import pin_key

    out = {}
    for name in ("tempseg_merged", "captioner_bf16"):
        pin = DERIVED_PINS_BY_KEY.get(pin_key(name))
        if pin is not None and pin.content_sha256:
            out[name] = pin.content_sha256
    return out


def derived_content_used(st: wh.State) -> dict[str, str]:
    """The derived content digests the balanced step's workers verified (``wave_h.py`` records
    them as ``derived_content``), else the pinned ones."""
    s = st.steps.get("balanced")
    used = (s.result.get("derived_content") if s else None) or {}
    return dict(used) if used else derived_content_default()


def lab_result(paths: Paths, st: wh.State, kind: str) -> dict[str, Any] | None:
    """The lab's result for ``kind``: the JSON wave_h saved, else the state's summary."""
    p = paths.lab / f"balanced-{kind}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    k = next((st.steps[n].result.get("kinds", {}).get(kind) for n in ("lab", "balanced")
              if n in st.steps and st.steps[n].result.get("kinds", {}).get(kind)), None)
    if not k or not (k.get("l0") or k.get("gate_a")):
        return None
    return {"kind": kind, "baseline": {"l0": k.get("l0") or []}, "gate_a": k.get("gate_a"),
            "engine_hashes": k.get("engine_hashes")}


def l0_for_certify(lab: Mapping[str, Any] | None, kind: str, gpu: str) -> dict[str, Any] | None:
    """What ``flora certify --l0`` reads (kind, the pool GPU, L0 rows, Gate A, the engine
    settings hashes the lab recorded per engine it built), without the lab's episode rows. The
    lab records ``nvidia-smi``'s name; it is checked against the pool GPU here and replaced by
    it, since ``certify.l0_summary`` compares the strings. ``certify.l0_summary`` calls a lab
    whose baseline hash is not the candidate's engine ``stale`` (review 13 S3-1)."""
    if lab is None:
        return None
    from flora_deploy.engines.offline_llm import EngineArgsError, normalize_gpu

    base = lab.get("baseline") or {}
    lab_gpu = base.get("gpu") or lab.get("gpu")
    if lab_gpu:
        try:
            same = normalize_gpu(str(lab_gpu)) == normalize_gpu(gpu)
        except EngineArgsError:
            same = str(lab_gpu) == gpu
        if not same:
            return {"kind": kind, "baseline": {"gpu": str(lab_gpu), "l0": base.get("l0") or []},
                    "gate_a": lab.get("gate_a"), "engine_hashes": lab.get("engine_hashes")}
    return {"kind": lab.get("kind") or kind, "baseline": {"gpu": gpu, "l0": base.get("l0") or []},
            "gate_a": lab.get("gate_a"), "engine_hashes": lab.get("engine_hashes")}


def record_path(paths: Paths, side: str, job: str, stage: str) -> Path:
    return paths.records / side / job / f"{stage}.jsonl"


def run_cli(args: list[str]) -> tuple[int, str]:
    """``flora <args>`` in process (the same command a user runs)."""
    from typer.testing import CliRunner

    from flora_deploy.cli import app

    r = CliRunner().invoke(app, args)
    return r.exit_code, r.output


def certify_all(paths: Paths, st: wh.State, out_dir: Path, *, derived: Mapping[str, str],
                cli: Callable[[list[str]], tuple[int, str]] = run_cli) -> list[dict[str, Any]]:
    """``flora certify`` per balanced job and stage; one outcome dict each."""
    stamp = utc(run_stamp(st))
    outcomes = []
    bal_jobs = jobs_of(st, "balanced")
    noise_jobs = jobs_of(st, "noise")
    witness_jobs = jobs_of(st, WITNESS[1], WITNESS[2])
    tmp = out_dir / "l0"
    tmp.mkdir(parents=True, exist_ok=True)
    for job in sorted(bal_jobs):
        r = bal_jobs[job]
        for stage in CERT_STAGES:
            o: dict[str, Any] = {"job": job, "stage": stage}
            outcomes.append(o)
            cfg = r.get("cfg")
            ex = record_path(paths, "exact", job, stage)
            bal = record_path(paths, "balanced", job, stage)
            if not ex.exists():  # a balanced-only session: certification comes later
                o["deferred"] = (f"{DEFERRED} (no oracle records: {paths.rel(ex)}; the exact "
                                 "run that certifies it has not run)")
                continue
            missing = [paths.rel(p) for p in (ex, bal) if not p.exists()]
            if not cfg or not Path(cfg).exists():
                missing.append(f"balanced.jobs.{job}.cfg ({paths.rel(cfg) if cfg else None})")
            if missing:
                o["refused"] = "missing input: " + ", ".join(missing) + " (run with --fetch)"
                continue
            args = ["certify", "--exact", str(ex), "--balanced", str(bal), "-c", str(cfg),
                    "--stage", stage, "--out-dir", str(out_dir), "--json",
                    "--generated-at", stamp]
            nz = record_path(paths, "noise", job, stage)
            if job in noise_jobs and nz.exists():
                args += ["--noise", str(nz)]
            iso = isolated_inputs(paths, st, job, tmp) if stage == "caption" else None
            if iso is not None and iso.get("missing"):
                o["refused"] = "missing input: " + ", ".join(iso["missing"]) + \
                    " (run with --fetch)"
                continue
            if iso is not None:  # v4: the isolated J2 is the candidate, J2w its witness
                i = args.index("--balanced")
                args[i + 1] = iso["balanced"]
                args[args.index("-c") + 1] = iso["cfg"]
                args += iso["args"]
            else:
                wit = record_path(paths, WITNESS[0], job, stage)
                if job in witness_jobs and wit.exists():
                    args += ["--witness", str(wit)]
            kind = LAB_KINDS[stage]
            lab = l0_for_certify(lab_result(paths, st, kind), kind, pool_gpu("balanced", stage))
            if lab is not None:
                lp = tmp / f"{kind}.json"
                lp.write_text(json.dumps(lab, sort_keys=True), encoding="utf-8")
                args += ["--l0", str(lp)]
            for name, digest in sorted(derived.items()):
                args += ["--derived-content", f"{name}={digest}"]
            code, text = cli(args)
            o["exit"] = code
            o["output"] = text  # console only (print_findings); never in a written file
            cert = parse_certificate(text, stage)
            verdict = cert.get("verdict") if cert else None
            if cert is None:
                o["refused"] = f"flora certify exited {code} without a certificate"
            elif (code, verdict) not in ((0, "certified"), (0, "provisional"), (1, "fail")):
                o["refused"] = f"flora certify exited {code} with verdict {verdict}"
            else:
                o["cert"] = cert
    return outcomes


def family_of(uid: str) -> str | None:
    """A reported-only family label of a certified-plan uid (``<data dir>--<local id>``):
    the witness / rest prefix (``n-`` / ``x-``) and the source dataset's first word, e.g.
    ``n-egodex``, ``x-wgo``; None when the uid does not carry them. Label charset
    ``[a-z0-9_.-]{1,32}`` (``flora certify --families``)."""
    local = uid.rsplit("--", 1)[-1]
    for prefix in (wh.NOISE_PREFIX, wh.REST_PREFIX):
        if local.startswith(prefix):
            src = re.sub(r"[^a-z0-9.]", "", local[len(prefix):].split("_", 1)[0]
                         .split("-", 1)[0].lower())
            label = f"{prefix}{src}"[:32]
            return label if src and _FAMILY.fullmatch(label) else None
    return None


def families(uids: Sequence[str]) -> dict[str, str]:
    return {u: f for u in sorted(set(uids)) if (f := family_of(u)) is not None}


def _uids(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line).get("uid"))
    return [u for u in out if isinstance(u, str)]


def isolated_inputs(paths: Paths, st: wh.State, job: str, tmp: Path) -> dict[str, Any] | None:
    """The v4 (stage-isolated) caption certification inputs of ``job``, when its state has
    the isolated caption job (step ``caption_iso``; None otherwise: the e2e caption inputs as
    before). The candidate is J2 (records, its config), the witness J2w; ``--digests`` for
    the oracle, the noise run, J2 and J2w (:data:`DIGEST_SIDES`), the staged oracle segment
    file (``--oracle-segments``, checked against the sha256 J2's config pinned), J1's caption
    records (``--reported-e2e``), the episode families (``--families``) and an older noise run
    (``--reported-repeat``, ``--digests repeat=``) when there is one. ``missing``: what is
    not there."""
    iso_jobs = jobs_of(st, SETS["isolated"])
    if job not in iso_jobs:
        return None
    r = iso_jobs[job]
    res = (st.steps.get(SETS["isolated"]) or wh.StepState()).result
    segs = r.get("oracle_segments") or res.get("oracle_segments") or {}
    missing: list[str] = []
    cand = record_path(paths, "isolated", job, "caption")
    for p in (cand, digest_path(paths, "exact", job), digest_path(paths, "isolated", job)):
        if not p.exists():
            missing.append(paths.rel(p))
    cfg = r.get("cfg")
    if not cfg or not Path(cfg).exists():
        missing.append(f"caption_iso.jobs.{job}.cfg ({paths.rel(cfg) if cfg else None})")
    seg_file = Path(str(segs.get("path") or ""))
    if not segs.get("path") or not seg_file.is_file():
        missing.append(f"the staged oracle segments ({segs.get('path')})")
    else:
        import hashlib

        if hashlib.sha256(seg_file.read_bytes()).hexdigest() != segs.get("sha256"):
            missing.append(f"the staged oracle segments as pinned ({paths.rel(seg_file)}: "
                           "sha256 differs)")
    if missing:
        return {"missing": missing}
    args: list[str] = []
    wit = record_path(paths, ISO_WITNESS[0], job, "caption")
    if job in jobs_of(st, ISO_WITNESS[1], ISO_WITNESS[2]) and wit.exists():
        args += ["--witness", str(wit)]
    for side in ("exact", "noise", "isolated", "isolated_witness"):
        d = digest_path(paths, side, job)
        if d.exists():
            args += ["--digests", f"{DIGEST_SIDES[side]}={d}"]
    args += ["--oracle-segments", str(seg_file)]
    e2e = record_path(paths, "balanced", job, "caption")
    if e2e.exists():
        args += ["--reported-e2e", str(e2e)]
    rep, rep_d = record_path(paths, "repeat", job, "caption"), digest_path(paths, "repeat", job)
    if job in jobs_of(st, SETS["repeat"]) and rep.exists() and rep_d.exists():
        args += ["--reported-repeat", str(rep), "--digests", f"repeat={rep_d}"]
    fam = families(_uids(cand))
    if fam:
        fp = tmp / f"families-{job}.json"
        fp.write_text(json.dumps(fam, sort_keys=True), encoding="utf-8")
        args += ["--families", str(fp)]
    return {"balanced": str(cand), "cfg": str(cfg), "args": args}


def parse_certificate(text: str, stage: str) -> dict[str, Any] | None:
    """The one certificate ``flora certify --json`` printed for ``stage``, else None."""
    start = text.find("[")
    if start < 0:
        return None
    try:
        certs = json.loads(text[start:])
    except ValueError:
        return None
    if (not isinstance(certs, list) or len(certs) != 1 or not isinstance(certs[0], dict)
            or certs[0].get("stage") != stage):
        return None
    return certs[0]


def missing_for_certified(cert: Mapping[str, Any]) -> list[str]:
    th = cert.get("thresholds") or {}
    cand = (cert.get("metrics") or {}).get("candidate") or {}
    n = sum(b.get("n", 0) for b in cand.values())
    noise = (cert.get("metrics") or {}).get("noise") or {}
    metrics = cert.get("metrics") or {}
    pre = cert.get("preemptions") or {}
    ctx = {"min_samples": th.get("min_samples"), "min_noise": th.get("min_noise_samples"),
           "n": n, "noise_n": noise.get("n", 0),
           "coverage": metrics.get("coverage"), "missing": metrics.get("missing"),
           "max_missing": th.get("max_missing"),
           "witness_n": (metrics.get("witness") or {}).get("n", 0),
           "preempted": pre.get("preempted"), "unknown": pre.get("unknown"),
           "lab": LAB_KINDS.get(cert.get("stage"), "-"),
           "metrics": ", ".join(c["metric"] for c in cert.get("checks", [])
                                if c.get("pass") is None) or "-",
           "failing": ", ".join(c["metric"] for c in cert.get("checks", [])
                                if c.get("pass") is False) or "-"}
    return [MISSING_HELP.get(r, r).format(**ctx) for r in cert.get("reasons", [])]


# ---------------------------------------------------------------------------------------------
# 3. Report


def step_status(st: wh.State, name: str) -> str:
    """The step's status as the report shows it. A state from before the lab became step (d2)
    ran the lab inline in (d) and marked (d) failed when only the lab failed: (d) is ``done``
    when every deployed job it ran ended ``done``, and the lab's outcome is named apart."""
    s = st.steps[name]
    if name != "balanced" or not inline_lab(st) or s.status != "failed":
        return s.status
    jobs = jobs_of(st, name)
    if jobs and all(r.get("state") == "done" for r in jobs.values()):
        return "done (its inline lab failed; the lab is step d2 now)"
    return s.status


def billing_final_after(st: wh.State, s: wh.StepState) -> float:
    """When billing will cover every attempt of ``s``: with the rows' watermark, the end of
    the billing hour of its latest attempt (``Attempt.hold_clears_at``); in a state without
    one, :data:`wave_h.BILLING_LAG_S` after it."""
    a = max(s.attempts, key=lambda x: x.settle_point)
    if st.billed_through is None:
        return a.settle_point + wh.BILLING_LAG_S
    return a.hold_clears_at()


def billed_elsewhere(st: wh.State) -> frozenset[str]:
    """The steps of ``st``'s run billed on another workspace (:data:`BILLED_ELSEWHERE`)."""
    return BILLED_ELSEWHERE.get(st.run_id, frozenset())


def step_letter(name: str) -> str:
    """``b`` for step ``exact_e2e``: the letter its title starts with."""
    return wh.TITLES[name].split(".", 1)[0]


def billing_rows(st: wh.State, *, certify_status: str | None = None
                 ) -> tuple[list[list[str]], list[str]]:
    """The "Spend per step" rows and their warnings. ``certify_status``: what step (e) shows
    once finalize has run ``flora certify`` (its per-stage verdicts), whatever the runbook
    recorded for it."""
    rows, warnings = [], []
    wm = st.watermark  # billing is complete up to here (the rows' interval_start)
    elsewhere = billed_elsewhere(st)
    for name in wh.ALL_STEPS:
        s = st.steps.get(name)
        if s is None:
            continue
        last = max((a.settle_point for a in s.attempts), default=None)
        if name in elsewhere:
            billing = "other workspace: not in this billing read"
        elif s.billed_usd is None:
            billing = "shared app: global total only" if s.apps else "no app"
        elif st.billed_at is None or last is None or wm is None:
            billing = "not read"
        elif wm < last:
            final = utc(billing_final_after(st, s))
            billing = (f"partial: read {st.billed_at - last:.0f} s after the step, billed "
                       f"through {utc(wm)}; final after {final}")
            warnings.append(f"{name}: billing covers up to {utc(wm)}, before the step's last "
                            f"activity {utc(last)}: billed {usd(s.billed_usd)} vs tracked "
                            f"{usd(s.tracked_usd)} is partial; re-run the runbook's billing "
                            f"read after {final}, then this script")
        else:
            gap = (s.billed_usd or 0.0) - s.tracked_usd
            if (abs(gap) > BILLING_MISMATCH_USD
                    and abs(gap) > BILLING_MISMATCH * max(s.tracked_usd, 0.01)):
                billing = f"MISMATCH: billed {gap:+.2f} vs tracked"
                warnings.append(f"{name}: billed {usd(s.billed_usd)} vs tracked "
                                f"{usd(s.tracked_usd)} after the billing lag "
                                f"({gap:+.2f}); the list prices or the receipts are off")
            else:
                billing = "reconciled"
        status = step_status(st, name)
        if name == "certify" and certify_status:
            status = certify_status
        rows.append([wh.TITLES[name], status, str(len(s.attempts)),
                     usd(wh.CAPS_USD[name]),
                     usd(s.tracked_usd), usd(s.charged(st.watermark)), usd(s.billed_usd),
                     billing])
    return rows, warnings


def table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return (["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def usd_per_video_h(ms: Mapping[tuple[str, str], Measurement], preset: str
                    ) -> dict[str, Any] | None:
    """3600 x sum over the preset's kinds of (container-s per video-s x pool $/s), from the
    unrounded rates and prices, rounded once (review 12 F9: summing per-term cents gave $5.62
    for $5.613)."""
    terms, unc_num, total = [], 0.0, 0.0
    for kind in PRESET_KINDS[preset]:
        m = ms[(preset, kind)]
        if m.rate is None or m.refusals:
            return None
        c = wh.pool_container(kind, preset)
        t = 3600.0 * m.rate_exact * c.usd_per_s()
        terms.append({"kind": kind, "gpu": m.gpu, "rate": m.rate,
                      "usd_per_h": round(c.usd_per_h(), 3), "usd_per_video_h": round(t, 2)})
        unc_num += t * (m.uncertainty or 0.0)
        total += t
    return {"terms": terms, "usd_per_video_h": round(total, 2),
            "uncertainty": round(unc_num / total, 2) if total else None}


def planned_usd_per_video_h(table: Any, preset: str) -> dict[str, Any] | None:
    """What ``flora plan`` prices ``preset``'s segment + caption work at, busy seconds only:
    the rate ``planner/rates.json`` gives at the pool's concurrency (``RateTable.rate``, 1-view,
    deployed features, as ``jobs.budget`` looks it up) x the pool's list price. This is the
    figure the docs quote. A rate measured here but not written to ``rates.json`` (every
    ``exact`` one: only the ``balanced`` slots are) is not what jobs are priced at."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.jobs.settings import episode_concurrency

    terms, unc_num, total = [], 0.0, 0.0
    for kind in PRESET_KINDS[preset]:
        pool = get_preset(preset).pools[kind]
        try:
            e = table.rate(preset, kind, 1, episode_concurrency(kind, pool))
        except KeyError:
            return None
        c = wh.pool_container(kind, preset)
        t = 3600.0 * e.container_s_per_video_s * c.usd_per_s()
        terms.append({"kind": kind, "gpu": strip_gpu(c.gpu), "rate": e.container_s_per_video_s,
                      "status": e.status, "usd_per_h": round(c.usd_per_h(), 3),
                      "usd_per_video_h": round(t, 2)})
        unc_num += t * e.uncertainty
        total += t
    return {"terms": terms, "usd_per_video_h": round(total, 2),
            "uncertainty": round(unc_num / total, 2) if total else None}


def run_usd_per_video_h(st: wh.State, step: str) -> float | None:
    """All in: the step's jobs' whole spend (cold starts, idle until scale-down, slack) per
    video-hour."""
    jobs = [r for r in jobs_of(st, step).values() if r.get("spend") is not None]
    video = sum(job_video_s(r) or 0.0 for r in jobs)
    return round(sum(float(r["spend"]) for r in jobs) / video * 3600, 2) if video else None


def run_usd_per_video_h_after_startup(st: wh.State, step: str) -> float | None:
    """The step's containers' cost without each one's cold start (spawn to first attempt):
    each container's $ x (seconds - startup) / seconds, per video-hour. What a warm job of the
    same size would pay, idle time until scale-down included."""
    jobs = list(jobs_of(st, step).values())
    video = sum(job_video_s(r) or 0.0 for r in jobs)
    usd_ = 0.0
    for r in jobs:
        for c in r.get("containers") or []:
            sec, up = float(c.get("seconds") or 0.0), float(c.get("startup_s") or 0.0)
            if sec <= 0 or c.get("usd") is None:
                return None
            usd_ += float(c["usd"]) * max(0.0, sec - up) / sec
    return round(usd_ / video * 3600, 2) if video and usd_ else None


def l0_rows(labs: Mapping[str, Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    out = []
    for kind in ("tempseg", "captioner"):
        lab = labs.get(kind)
        rows = list(((lab or {}).get("baseline") or {}).get("l0") or [])
        if not rows:
            out.append({"kind": kind, "status": "missing"})
            continue

        def count(key: str, rs=rows) -> int:
            return sum(1 for r in rs if r.get(key) is True)

        ok = all(r.get("ids_equal") is True and r.get("grid_equal") is True
                 and r.get("render_ids_equal", True) is True for r in rows)
        out.append({"kind": kind, "status": "pass" if ok else "fail", "requests": len(rows),
                    "ids_equal": count("ids_equal"), "grid_equal": count("grid_equal"),
                    "render_ids_equal": count("render_ids_equal")})
    return out


def gate_a(labs: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """Gate A per kind and the plugin recommendation, by docs/balanced-engines.md's rule: turn
    the FP32 ``dt_bias`` plugin on only if its drift exceeds the oracle's noise. H-lite does not
    measure the oracle's logit noise, so the reference is the runbook's Gate A bound."""
    from flora_deploy.validate.certify import gate_a_summary

    kinds = {k: gate_a_summary(labs.get(k)) for k in ("tempseg", "captioner")}
    # a Gate A with no request is not a measurement ("0/0", review 12 F8): not measured
    kinds = {k: g if g and g.get("requests") else None for k, g in kinds.items()}
    ref = wh.GATE_A_MAX_DLOGIT
    measured = {k: g for k, g in kinds.items() if g and g.get("max_abs_dlogit") is not None}
    over = sorted(k for k, g in measured.items() if float(g["max_abs_dlogit"]) > ref)
    if not measured:
        rec, why = "off", "Gate A was not measured; keep the default (off)"
    elif over:
        rec, why = "on", (f"max |Δlogit| of {', '.join(over)} exceeds the reference {ref}: "
                          "certify the plugin-on stack instead")
    else:
        worst = max(float(g["max_abs_dlogit"]) for g in measured.values())
        rec, why = "off", (f"max |Δlogit| {worst:g} is within the reference {ref} for "
                           f"{', '.join(sorted(measured))}")
        if len(measured) < 2:
            why += f" ({', '.join(sorted(set(kinds) - set(measured)))} not measured)"
    return {"kinds": kinds, "reference": ref, "recommendation": rec, "why": why}


def cert_rows(outcomes: Sequence[Mapping[str, Any]]) -> tuple[list[list[str]], list[str]]:
    rows, missing = [], []
    for o in outcomes:
        c = o.get("cert")
        if o.get("deferred"):
            rows.append([o["stage"], o["job"], "-", "deferred", "-", "-", "-"])
            missing.append(f"{o['stage']} ({o['job']}): {o['deferred']}")
            continue
        if c is None:
            rows.append([o["stage"], o["job"], "-", "refused", "-", "-", "-"])
            missing.append(f"{o['stage']} ({o['job']}): refused: {o.get('refused')}")
            continue
        cand = (c.get("metrics") or {}).get("candidate") or {}
        noise = (c.get("metrics") or {}).get("noise") or {}
        rows.append([c["stage"], ", ".join(c["covered_pools"]), c["gpu"], c["verdict"],
                     str(sum(b.get("n", 0) for b in cand.values())), str(noise.get("n", 0)),
                     f"`{c['id']}`"])
        if c["verdict"] != "certified":
            missing += [f"{c['stage']} ({', '.join(c['covered_pools'])} on {c['gpu']}): {m}"
                        for m in missing_for_certified(c)]
    return rows, missing


def outcome_verdict(o: Mapping[str, Any]) -> str:
    if o.get("deferred"):
        return "deferred"
    return o["cert"]["verdict"] if o.get("cert") else "refused"


def stage_verdicts(outcomes: Sequence[Mapping[str, Any]], *, bold: bool = False) -> str:
    """One verdict per (stage, job): ``segment (human) certified, caption (human) fail``. A
    stage's certificate unlocks that stage on its own, so this is what the docs state;
    :func:`overall_verdict` (the worst of them) is for the console."""
    b = "**" if bold else ""
    return ", ".join(f"{o['stage']} ({o['job']}) {b}{outcome_verdict(o)}{b}"
                     for o in outcomes) or "none"


def overall_verdict(outcomes: Sequence[Mapping[str, Any]]) -> str:
    vs = [o["cert"]["verdict"] if o.get("cert") else "refused" for o in outcomes
          if not o.get("deferred")]
    for v in ("refused", "fail", "provisional"):
        if v in vs:
            return v
    if not vs:
        return "deferred" if outcomes else "none"
    return "certified"


def run_plan(st: wh.State) -> str:
    """The plan the state was run under: the one it stores (``certified`` does), else ``full``
    when a step only that plan runs was started, else the default ``balanced-only``."""
    if getattr(st, "plan", None) in wh.PLANS:
        return st.plan
    for name in wh.deferred_steps(wh.DEFAULT_PLAN):
        s = st.steps.get(name)
        if s is not None and (s.attempts or s.status != "pending"):
            return "full"
    return wh.DEFAULT_PLAN


def inline_lab(st: wh.State) -> bool:
    """A state from before the lab became step ``lab`` (d2): (d) ran it inline (its result
    has ``kinds``) and no ``lab`` step exists."""
    bal = st.steps.get("balanced")
    return "lab" not in st.steps and bal is not None and "kinds" in bal.result


def global_cap(st: wh.State) -> float:
    """The global cap in force (review 12 F6): the one the last ``--yes-spend`` run gated on
    (``State.gate_cap``) when the state has it for the state's plan. Otherwise it is
    reconstructed: ``wave_h.plan_global_cap`` for the state's plan, from the upper projection
    the runbook stored on each step before it ran it (a step it never reached is priced as the
    runbook prices it: offline manifest, today's rates). With a user-approved override for that
    plan (``--global-cap``), the runbook gates on the larger of the override and the cap derived
    from today's projections, so that is reported."""
    plan = run_plan(st)
    if st.gate_cap is not None and st.gate_cap_plan == plan:
        return st.gate_cap
    spec = wh.SPECS[plan]
    override = st.override_for(plan)
    if override is not None:  # priced as wave_h.main prices it (plan_projections)
        man = wh.offline_manifest(wh.DURATION_HEADROOM, spec.golden)
        lab_done = plan == wh.CERTIFIED and wh.lab_already_done()[0]
        reuse = () if lab_done else wh.lab_reuse(st, spec)
        today = wh.plan_projections(plan, man, lab_done=lab_done, reuse=reuse)
        return wh.plan_global_cap(plan, today, override)
    projections: dict[str, Any] = {}
    for name in wh.PLANS[plan]:
        if name == "lab" and inline_lab(st):  # gated before (d2) existed: (d)'s stored
            projections[name] = SimpleNamespace(usd_upper=0.0)  # projection covers its lab
            continue
        s = st.steps.get(name)
        if s is not None and s.projected_upper_usd is not None:
            projections[name] = SimpleNamespace(usd_upper=s.projected_upper_usd)
        else:
            projections[name] = wh.project(
                name, wh.offline_manifest(wh.DURATION_HEADROOM, spec.golden), spec)
    return wh.plan_global_cap(plan, projections)


@dataclass
class Findings:
    st: wh.State
    ms: dict[tuple[str, str], Measurement]
    rates_ok: bool
    refusals: list[str]
    outcomes: list[dict[str, Any]]
    l0: list[dict[str, Any]]
    gate: dict[str, Any]
    cost: dict[str, dict[str, Any] | None]
    billing: list[list[str]]
    warnings: list[str]
    date: str
    # what `flora plan` prices each preset at (planned_usd_per_video_h); the docs quote these
    planned: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    # the `balanced` stages a certificate unlocks (certified_kinds), for the README cost block
    certified: frozenset[str] = frozenset()


def render_report(f: Findings) -> str:
    st = f.st
    L = ["# Wave H-lite report", "",
         "Generated by `scripts/wave_h_finalize.py` from the runbook state (`scripts/wave_h.py`, "
         "`docs/wave-h-runbook.md`).",
         "Numbers only: no NVIDIA text, captions, prompts or episode ids; those stay in the",
         "git-ignored `reports/wave-h/`.", "",
         f"Run `{st.run_id}`, finished {utc(run_stamp(st))}.", "", "## Spend per step", "",
         "Tracked: container-seconds x list $/h from `flora report` (the lab: its wall time). "
         "Charged: tracked, or a killed / unreported attempt at its bound (its time plus the "
         "stop slack, once its stop was confirmed) or else its budget, until billing covers it "
         "(the billing rows' `interval_start` watermark). Billed: the billed delta of the apps "
         "only that step used (`scripts/spend.py`).", ""]
    L += table(["step", "status", "attempts", "cap", "tracked", "charged", "billed", "billing"],
               f.billing)
    tracked = sum(s.tracked_usd for s in st.steps.values())
    charged = sum(s.charged(st.watermark) for s in st.steps.values())
    elsewhere = billed_elsewhere(st) & set(st.steps)
    L += ["", f"Total: **{usd(st.spent)}** of the {usd(global_cap(st))} global cap: the "
          "larger of charged and billed, the figure the runbook gates on.", "",
          f"- tracked {usd(tracked)}: container-seconds x list $/h of every attempt;",
          f"- charged {usd(charged)}: tracked, with a killed or unreported attempt at its bound "
          "or budget until billing covers it"
          + (f" ({usd(charged - tracked)} over tracked)" if charged - tracked > 0.005 else "")
          + ";",
          f"- billed {usd(st.billed_total)}: Modal's billed delta of every app (read "
          f"{utc(st.billed_at)})"
          + (f"; steps {and_list([step_letter(n) for n in wh.STEPS if n in elsewhere])} ran "
             "on another Modal workspace, so their billing is not in this read and their "
             "tracked figure stands" if elsewhere else "")
          + "."]
    if f.warnings:
        L += ["", "Warnings:", *[f"- {w}" for w in f.warnings]]

    L += ["", "## Measured rates", "",
          "Container-seconds per video-second: busy seconds over video seconds, one sample per "
          "job. Cold start: spawn to first attempt, one sample per container. Error band: "
          f"max({UNCERTAINTY_FLOOR:.0%}, range / mean); a single sample (n=1) has no range, so "
          f"its band is ±{SINGLE_SAMPLE_FLOOR:.0%}.", ""]
    rows = []
    for (preset, kind), m in sorted(f.ms.items()):
        s = m.summary()
        rng = "-" if not s["range"] else f"{s['range'][0]:.3f}-{s['range'][1]:.3f}"
        srng = ("-" if not s["startup_range"]
                else f"{s['startup_range'][0]:.0f}-{s['startup_range'][1]:.0f}")
        unc = "-" if s["uncertainty"] is None else f"±{s['uncertainty']:.0%}"
        sunc = "-" if s["startup_uncertainty"] is None else f"±{s['startup_uncertainty']:.0%}"
        rows.append([preset, kind, m.gpu, str(m.n), f"{m.video_s:.1f}",
                     "-" if m.rate is None else f"{m.rate:.3f}", rng, unc,
                     "-" if m.startup_s is None else f"{m.startup_s:.0f}", str(len(m.startups)),
                     srng, sunc, "refused" if s["refused"] else "ok"])
    L += table(["preset", "kind", "GPU", "jobs", "video s", "container-s / video-s", "range",
                "band", "cold start s", "containers", "range", "band", "status"], rows)
    if f.rates_ok:
        L += ["", "`planner/rates.json`: the `balanced` segment / caption rows are `measured` "
              f"(source this report, {f.date}), with their own startup rows."]
    else:
        L += ["", "`planner/rates.json` **not updated** (the `balanced` rows stay "
              "`placeholder`):", *[f"- {r}" for r in f.refusals]]

    L += ["", "## $/video-hour at list prices", "",
          "```",
          "$/video-h = 3600 x sum over the preset's kinds of (container-s per video-s x pool $/s)",
          "pool $/s  = (GPU $/h + cores x core $/h + memory GiB x GiB $/h) / 3600   "
          "(jobs.budget pinned list prices)",
          "```", ""]
    rows = []
    for preset in ("exact", "balanced"):
        c = f.cost.get(preset)
        run = run_usd_per_video_h(st, "exact_e2e" if preset == "exact" else "balanced")
        if c is None:
            rows.append([preset, "not measured", "-", usd(run)])
            continue
        terms = " + ".join(f"3600 x {t['rate']:.3f} x ${t['usd_per_h']:.3f}/h / 3600 "
                           f"({t['kind']}, {t['gpu']})" for t in c["terms"])
        rows.append([preset, terms, f"**{usd(c['usd_per_video_h'])}** ±{c['uncertainty']:.0%}",
                     usd(run)])
    L += table(["preset", "terms", "$/video-h (busy, this run's rate)",
                "this run (cold starts, idle, slack)"], rows)
    warm = run_usd_per_video_h_after_startup(st, "balanced")
    L += ["", "$/video-h (busy) counts busy container-seconds only, without cold starts or idle "
          "time until scale-down: what a large job approaches if the per-video-second rate "
          "holds at scale, a projection until a scale run measures it. "
          "This run's figure is everything "
          "its jobs spent"
          + (f"; without the cold starts, `balanced` was {usd(warm)}/video-h (idle kept)"
             if warm is not None else "")
          + ". One small job (n=1): the band is a nominal convention, not a statistical "
          "interval."]
    rows = []
    for preset in ("exact", "balanced"):
        c = f.planned.get(preset)
        if c is None:
            rows.append([preset, "no rate", "-"])
            continue
        terms = " + ".join(f"{t['rate']:g} ({t['status']}) x ${t['usd_per_h']:.3f}/h "
                           f"({t['kind']}, {t['gpu']})" for t in c["terms"])
        rows.append([preset, terms, f"**{usd(c['usd_per_video_h'])}** ±{c['uncertainty']:.0%}"])
    L += ["", "What `flora plan` prices (`planner/rates.json`, busy seconds only). These are the "
          "figures the README and `docs/costs.md` quote. Only the `balanced` slots are written "
          "from this run; an `exact` rate measured above is reported, not folded in, so "
          "`exact` stays at its committed row.", ""]
    L += table(["preset", "rate x pool $/h", "$/video-h (busy)"], rows)
    run_ex, ex = f.cost.get("exact"), f.planned.get("exact")
    if run_ex and ex and abs(run_ex["usd_per_video_h"] / ex["usd_per_video_h"] - 1) > 0.10:
        L += ["", f"`exact` measured here ({usd(run_ex['usd_per_video_h'])}) and as priced "
              f"({usd(ex['usd_per_video_h'])}) differ. This run's `exact` jobs are the oracle "
              "and its noise repeat, on one app; the committed row is its own measurement "
              "(see its `provenance` in `planner/rates.json`). Folding this run's `exact` rate "
              "into `rates.json` is a maintainer decision; until then the docs quote the "
              "priced figure."]
    ex, bal = f.planned.get("exact"), f.planned.get("balanced")
    if ex and bal and bal["usd_per_video_h"]:
        L += ["", f"`exact` / `balanced` as priced: "
              f"{ex['usd_per_video_h'] / bal['usd_per_video_h']:.1f}x (segment + caption only; "
              "rewards are the same on both presets)."]

    L += ["", "## L0 (input parity: vLLM vs NVIDIA's HF processor)", ""]
    L += table(["kind", "status", "requests", "ids equal", "render ids equal", "grid equal"],
               [[r["kind"], r["status"], str(r.get("requests", "-")),
                 str(r.get("ids_equal", "-")), str(r.get("render_ids_equal", "-")),
                 str(r.get("grid_equal", "-"))] for r in f.l0])

    L += ["", "## Gate A (FP32 `dt_bias` plugin vs BF16)", ""]
    rows = []
    for kind, g in f.gate["kinds"].items():
        if not g:
            rows.append([kind, "not measured", "-", "-", "-", "-"])
            continue
        rows.append([kind, str(g.get("requests", "-")), str(g.get("greedy_identical", "-")),
                     str(g.get("positions", "-")), f"{g.get('max_abs_dlogit')}",
                     f"{g.get('mean_abs_dlogit')}"])
    L += table(["kind", "requests", "greedy identical", "positions", "max |Δlogit|",
                "mean |Δlogit|"], rows)
    L += ["", "Rule (`docs/balanced-engines.md`): turn the plugin on only if its drift exceeds "
          "the oracle's noise. H-lite does not measure the oracle's logit noise, so the "
          f"reference is the runbook's Gate A bound, max |Δlogit| <= {f.gate['reference']}.", "",
          f"**Recommendation: plugin {f.gate['recommendation']}** ({f.gate['why']})."]

    rows, missing = cert_rows(f.outcomes)
    L += ["", "## Certificate", "",
          f"Per stage: {stage_verdicts(f.outcomes, bold=True)}. Each stage's certificate "
          "unlocks that stage on its own. Reference: `exact` in NVIDIA's default mode; "
          "certificates in `src/flora_deploy/config/certificates/` (metrics only; a `fail` is "
          "never committed).", ""]
    L += table(["stage", "pool", "GPU", "verdict", "episodes", "noise episodes", "id"], rows)
    if missing:
        L += ["", "Missing for `certified`:", *[f"- {m}" for m in missing]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------
# 4. Docs blocks


def close_marker(name: str) -> str:
    return f"<!-- /wave-h:{name} -->"


def replace_block(text: str, name: str, body: str) -> str:
    """``body`` between the ``<!-- wave-h:name -->`` and ``<!-- /wave-h:name -->`` markers
    (exactly that block; everything else is kept). The markers are whole lines (the open one
    may carry a note: ``<!-- wave-h:name (note) -->``), exactly one of each, open before close.
    Anything else is refused and nothing is inserted."""
    n = re.escape(name)
    opens = list(re.finditer(rf"^<!-- wave-h:{n}(?: [^>]*)? -->[ \t]*$", text, re.M))
    closes = list(re.finditer(rf"^{re.escape(close_marker(name))}[ \t]*$", text, re.M))
    if len(opens) != 1 or len(closes) != 1 or closes[0].start() < opens[0].end():
        raise SystemExit(f"need exactly one `<!-- wave-h:{name} -->` line followed by one "
                         f"`{close_marker(name)}` line to fill (found {len(opens)} open, "
                         f"{len(closes)} close)")
    start = opens[0].end() + 1
    return text[:start] + body.rstrip("\n") + "\n" + text[closes[0].start():]


def results_block(f: Findings) -> str:
    l0 = "; ".join(f"{r['kind']}: {r['status']}" + (
        f" ({r['ids_equal']}/{r['requests']} ids, {r['grid_equal']}/{r['requests']} grid)"
        if r["status"] != "missing" else "") for r in f.l0)
    bal_jobs = [r for r in jobs_of(f.st, "balanced").values()
                if r.get("state") in ("done", "completed_with_errors")]
    e2e = (f"{len(bal_jobs)} `balanced` job(s) through `flora deploy` / `flora run` "
           f"(run `{f.st.run_id}`); certificate per stage: {stage_verdicts(f.outcomes)}")
    bal, ex = f.cost.get("balanced"), f.planned.get("exact")
    if bal and ex:
        ratio = ex["usd_per_video_h"] / bal["usd_per_video_h"]
        cost = (f"**${bal['usd_per_video_h']:.2f}/video-h** ±{bal['uncertainty']:.0%} vs `exact` "
                f"${ex['usd_per_video_h']:.2f} as `flora plan` prices it ({ratio:.1f}x cheaper), "
                f"list prices, busy seconds; {projection_basis(f)}")
    elif bal:
        cost = (f"**${bal['usd_per_video_h']:.2f}/video-h** ±{bal['uncertainty']:.0%}, list "
                f"prices, busy seconds; {projection_basis(f)}")
    else:
        cost = "**not measured** (" + "; ".join(f.refusals[:2]) + ")"
    g = f.gate
    ga = "; ".join(f"{k}: max {v['max_abs_dlogit']}, mean {v['mean_abs_dlogit']}, greedy "
                   f"{v['greedy_identical']}/{v['requests']}" for k, v in g["kinds"].items() if v)
    ga = (ga or "**not measured**") + f"; recommendation: plugin **{g['recommendation']}**"
    rows = [["L0 in-engine (ids, grid vs HF)", l0],
            ["Segments / captions end to end on golden episodes", e2e],
            ["Throughput, $/video-h vs `exact`", cost],
            ["Gate A (dt_bias FP32 vs bf16)", ga]]
    return "\n".join([*table(["Item", f"Result (Wave H-lite, {f.date})"],
                             rows)])


def cost_block(f: Findings) -> str | None:
    bal, ex = f.cost.get("balanced"), (getattr(f, "planned", None) or {}).get("exact")
    if not bal:
        return None
    # README wording (no internal run names): the busy-only figure, said plainly to be
    # busy-only and a projection; `exact` as `flora plan` prices it, so the two agree, on its
    # own line so the balanced band and job are not read as its own. That figure is the
    # planner's committed row, not this run's: when this run measured `exact` differently, say
    # both. Lines of at most ~125 characters (the README's changed lines stay short).
    basis = f"at Modal list prices; a projection from rates {projection_basis(f)}."
    tail = ("Startup, idle time and rewards are extra and dominate small jobs: `flora plan` "
            "shows the all-in estimate for your dataset.")
    split = partial_cost(f, bal)
    if split is not None:  # some stages certified, not all: never price them as one certified
        (cert, cert_usd, cert_unc), (unc, unc_usd) = split
        vs = (f",\nor `exact` runs certified segmentation + captioning for "
              f"${ex['usd_per_video_h']:.2f}" if ex else "")
        lines = [f"`balanced` costs about **${cert_usd:.2f} per video-hour** of GPU work "
                 f"(±{cert_unc:.0%}) for {and_list(cert)}, which is certified;\n"
                 f"{and_list(unc)} adds about ${unc_usd:.2f} but is not certified (its records "
                 f"say `certified: false`){vs},", basis]
        run = exact_run_line(f, ex) if ex else None
        if run:  # the planner's `exact` price above is not this run's measurement: say both
            lines.append(run[0].upper() + run[1:] + ".")
        return "\n".join([*lines, tail])
    lines = [f"`balanced` costs about **${bal['usd_per_video_h']:.2f} per video-hour** of GPU "
             f"work (±{bal['uncertainty']:.0%}) for segmentation + captioning,", basis]
    if ex:
        lines.append(exact_cost_line(f, ex))
    return "\n".join([*lines, tail])


def exact_cost_line(f: Any, ex: Mapping[str, Any]) -> str:
    """The README's `exact` line: the planner's rate (what `flora plan` prices; its committed
    row, an older measurement) and, when this run measured `exact` more than 10% off it, this
    run's figure with its band and n, so neither is attributed to the other's job."""
    rate = " + ".join(f"{t['rate']:g}" for t in ex.get("terms", ()))
    band = f" (±{ex['uncertainty']:.0%})" if ex.get("uncertainty") is not None else ""
    line = (f"`exact` is priced at ${ex['usd_per_video_h']:.2f}{band}: the planner's rate "
            f"({rate + ' GPU-s per video-s, ' if rate else ''}an older measurement)")
    run = exact_run_line(f, ex)
    return line + (f";\n{run}" if run else "") + "."


def exact_run_line(f: Any, ex: Mapping[str, Any]) -> str | None:
    """This run's `exact` figure (band, n) when it is more than 10% off the planner's rate
    ``ex``, worded as not yet folded into the planner; ``None`` otherwise."""
    run = (getattr(f, "cost", None) or {}).get("exact")
    if not (run and ex["usd_per_video_h"]
            and abs(run["usd_per_video_h"] / ex["usd_per_video_h"] - 1) > 0.10):
        return None
    m = (getattr(f, "ms", None) or {}).get(("exact", "segcap"))
    n = f", n={m.n} jobs" if m is not None and m.n else ""
    when = getattr(f, "date", None)
    return (f"the {when + ' ' if when else ''}validation run measured `exact` at "
            f"${run['usd_per_video_h']:.2f} (±{run['uncertainty']:.0%}{n}), not yet folded "
            "into the planner")


def partial_cost(f: Any, bal: Mapping[str, Any]
                 ) -> tuple[tuple[list[str], float, float], tuple[list[str], float]] | None:
    """When a certificate unlocks some of ``balanced``'s priced stages but not all
    (``f.certified``): ((certified stage words, their $/video-h, its band), (uncertified stage
    words, their $/video-h)), from the per-stage terms of ``usd_per_video_h``. ``None`` when
    every stage or none is certified, or the terms are missing (the one-price wording)."""
    kinds = PRESET_KINDS["balanced"]
    certified = set(getattr(f, "certified", None) or ()) & set(kinds)
    terms = {t["kind"]: t for t in bal.get("terms") or ()}
    if not certified or certified == set(kinds) or not set(kinds) <= set(terms):
        return None
    ms = getattr(f, "ms", None) or {}

    def unc(k: str) -> float:
        u = getattr(ms.get(("balanced", k)), "uncertainty", None)
        return bal["uncertainty"] if u is None else u

    cert = [k for k in kinds if k in certified]
    rest = [k for k in kinds if k not in certified]
    usd_c = sum(terms[k]["usd_per_video_h"] for k in cert)
    band_c = (sum(terms[k]["usd_per_video_h"] * unc(k) for k in cert) / usd_c if usd_c
              else bal["uncertainty"])
    return (([STAGE_WORDS[k] for k in cert], round(usd_c, 2), band_c),
            ([STAGE_WORDS[k] for k in rest],
             round(sum(terms[k]["usd_per_video_h"] for k in rest), 2)))


def projection_basis(f: Any) -> str:
    """Where the ``balanced`` rate comes from, in plain words: ``measured on one 3.9-min job
    (2026-10-22, n=1), not yet confirmed at scale``."""
    m = (getattr(f, "ms", None) or {}).get(("balanced", "segment"))
    if m is None or not m.n:
        return f"measured {f.date}, not yet confirmed at scale"
    what = (f"one {m.video_s / 60:.1f}-min job" if m.n == 1
            else f"{m.n} jobs ({m.video_s / 60:.1f} min of video)")
    return f"measured on {what} ({f.date}, n={m.n}), not yet confirmed at scale"


def and_list(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


STAGE_WORDS = {"segment": "segmentation", "caption": "captioning",
               "reward.terminal": "terminal rewards", "reward.dense": "dense rewards",
               "reward.rubric": "rubric rewards"}


def certified_kinds(cert_dirs: Sequence[Path]) -> dict[str, tuple[set[str], set[str]]]:
    """Stage kind -> (embodiments, GPUs) the ``balanced`` certificates in ``cert_dirs`` unlock
    (the newest for its key decides, as in ``certkey.matching``; verdict ``certified``, within
    policy)."""
    from flora_deploy.validate import certkey

    newest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for c in sorted((c for d in cert_dirs for c in certkey.load_certificates(d)
                     if c.get("preset") == "balanced"),
                    key=lambda c: (c["generated_at"], certkey.VERDICTS.index(c["verdict"]))):
        newest[cert_key(c)] = c  # the last one wins: newest, then fail over certified
    kinds: dict[str, tuple[set[str], set[str]]] = {}  # kind -> (embodiments, GPUs)
    for c in newest.values():
        if c["verdict"] != "certified" or certkey.policy_problems(c):
            continue
        embs, gpus = kinds.setdefault(c["kind"], (set(), set()))
        embs.update(c.get("covered_pools") or ())
        gpus.add(c["gpu"])
    return kinds


def exact_price(table: Any = None) -> str | None:
    """``$36.39``: what ``flora plan`` prices ``exact`` segmentation + captioning at per
    video-hour of GPU work (``planned_usd_per_video_h``, the figure the README cost block
    quotes), from ``table`` (default: the shipped ``planner/rates.json``; finalize never
    rewrites its ``exact`` rates). ``None`` when it cannot be priced."""
    try:
        if table is None:
            from flora_deploy.planner.rates import RateTable

            table = RateTable.load()
        p = planned_usd_per_video_h(table, "exact")
    except Exception:  # noqa: BLE001 - the wording then leaves the number out
        return None
    return f"${p['usd_per_video_h']:.2f}" if p else None


def certification_blocks(readme: str, cert_dirs: Sequence[Path], *, table: Any = None) -> str:
    """``readme`` with its ``wave-h:certified`` (one status line under "Label your own data")
    and ``wave-h:allow-uncertified`` (under the Configuration example) blocks written from the
    ``balanced`` certificates in ``cert_dirs`` that unlock (``certified_kinds``): which stages,
    for which embodiments, on which GPUs. Plain words, no ids. When segmentation is certified
    and captioning is not, both blocks say so and give the two honest ways to get captions
    (``exact``'s price from ``table``, see ``exact_price``)."""
    words = STAGE_WORDS
    kinds = certified_kinds(cert_dirs)
    groups: dict[tuple[str, ...], list[str]] = {}  # embodiments -> kinds, in stage order
    for k in sorted(kinds, key=lambda k: list(words).index(k) if k in words else 99):
        groups.setdefault(tuple(sorted(kinds[k][0])), []).append(k)
    parts = []
    for embs, ks in groups.items():
        gpus = list(dict.fromkeys(g for k in ks for g in sorted(kinds[k][1])))  # stage order
        parts.append(f"{and_list([words.get(k, k) for k in ks])} for "
                     f"{and_list([f'`{e}`' for e in embs])} data "
                     f"({' / '.join(f'`{g}`' for g in gpus)})")
    if not parts:
        status = ("Accuracy certificates: none yet, so `balanced` segmentation and captioning "
                  "records say `certified: false`.\nRewards run NVIDIA's own engine, as "
                  "`exact` does, and need none.")
        config = ("`allow_uncertified: true` lets `balanced` run segmentation and captioning "
                  "without accuracy certificates.\nNone exist yet, so it is required for them.")
    elif "segment" in kinds and "caption" not in kinds:  # segmentation only: say how to caption
        price = exact_price(table)
        at = f"the `exact` price (about {price} per video-hour)" if price else "the `exact` price"
        status = (f"Accuracy certificates: {'; '.join(parts)}. Captioning is not certified:\n"
                  "run it with `allow_uncertified: true` (its records say `certified: false`), "
                  "or as a separate `exact` job,\nwhich captions its own segments at "
                  f"{at}.\nRewards run NVIDIA's own engine, as `exact` does, and need none.")
        config = ("`allow_uncertified: true` runs the stages that have no accuracy certificate "
                  "for your embodiment and GPU;\ntheir records say `certified: false`. Certified "
                  f"today: {'; '.join(parts)}.\nCaptioning is not certified. For captions, "
                  "either (i) run one `balanced` job with `allow_uncertified: true`:\nits caption "
                  "records say `certified: false`; or (ii) run captions as a separate job with\n"
                  "`preset: exact, stages: [segment, caption]`: it captions `exact`'s own "
                  f"segments,\nat {at}. One config has one preset, so the two cannot be mixed.\n"
                  "A config whose every stage is certified does not need `allow_uncertified`; "
                  "`flora doctor` tells you.")
    else:
        status = (f"Accuracy certificates: {'; '.join(parts)}.\nRewards run NVIDIA's own "
                  "engine, as `exact` does, and need none. Everything else runs uncertified.")
        config = ("`allow_uncertified: true` runs the stages that have no accuracy certificate "
                  "for your embodiment and GPU;\ntheir records say `certified: false`. Certified "
                  f"today: {'; '.join(parts)}.\nA config whose every stage is certified does "
                  "not need it; `flora doctor` tells you.")
    readme = replace_block(readme, "certified", status)
    return replace_block(readme, "allow-uncertified", config)


# ---------------------------------------------------------------------------------------------
# Driver


def record_sets() -> list[tuple[str, str, str]]:
    """(side, step, result key) of every job whose records ``--fetch`` exports."""
    return [(side, step, "jobs") for side, step in SETS.items()] + [WITNESS, ISO_WITNESS]


def job_audits(job_id: str, store: str) -> list[dict[str, Any]]:
    """Every audit line of a job (its store's ``.../audit/...`` files; a Volume read)."""
    import contextlib

    from flora_deploy import sdk
    from flora_deploy.jobs.store import JobPaths

    job = sdk.open_job(job_id, store=store)
    out: list[dict[str, Any]] = []
    for f in job.store.list(f"{JobPaths(job_id).root}/"):
        if "/audit/" not in f:
            continue
        for line in job.store.get_bytes(f).decode().splitlines():
            with contextlib.suppress(ValueError):
                a = json.loads(line)
                if isinstance(a, dict):
                    out.append(a)
    return out


def _hex_or_none(x: Any) -> str | None:
    return x if isinstance(x, str) and _HEX64.fullmatch(x) else None


def _caption_rows(rec: Any) -> list[Any]:
    return list(getattr(rec, "segments", ()) or ())


def segcap_digests(records: Sequence[Any], audits: Sequence[Mapping[str, Any]]
                   ) -> dict[str, list[str | None]]:
    """``{uid: [digest | None per caption row]}`` of an exact (segcap) job: from each episode's
    audit line, ``result.flora_reference.model_calls`` with ``output_limit`` 512 (the
    Captioner's), in call order; the k-th such call pairs with the k-th caption row whose
    ``status`` is ``complete`` (``stages.reference``'s pairing), every other row is None, and
    so is a complete row with no call left. An episode with several audit lines (a retried
    shard) takes the one whose captions are the record's; if that leaves differing digest
    lists, its rows are all None (certify then refuses: ``oracle_digests_missing``). Only hex
    digests are kept: no text."""
    by_uid: dict[str, list[Mapping[str, Any]]] = {}
    for a in audits:
        if a.get("stage") == "segcap" and isinstance(a.get("result"), dict) and a.get("uid"):
            by_uid.setdefault(str(a["uid"]), []).append(a)
    out: dict[str, list[str | None]] = {}
    for rec in records:
        if getattr(rec, "stage", None) != "caption":
            continue
        rows = _caption_rows(rec)
        want = [s.caption for s in rows if s.status == "complete"]

        def calls(a: Mapping[str, Any]) -> list[str | None]:
            mc = ((a.get("result") or {}).get("flora_reference") or {}).get("model_calls") or []
            return [_hex_or_none(c.get("input_digest")) for c in mc
                    if isinstance(c, dict) and c.get("output_limit") == CAPTIONER_OUTPUT_LIMIT]

        cands = by_uid.get(rec.uid, [])
        if len(cands) > 1:  # in memory only: which line wrote this record's captions
            same = [a for a in cands if [s.get("caption") for s in
                                         (a["result"].get("segments") or ())
                                         if isinstance(s, dict)
                                         and s.get("status") == "complete"] == want]
            cands = same or cands
        lists = {tuple(calls(a)) for a in cands}
        digests = list(next(iter(lists))) if len(lists) == 1 else []
        it = iter(digests)
        out[rec.uid] = [next(it, None) if s.status == "complete" else None for s in rows]
    return out


def caption_digests(records: Sequence[Any], audits: Sequence[Mapping[str, Any]]
                    ) -> dict[str, list[str | None]]:
    """``{uid: [digest | None per caption row]}`` of a balanced caption job: each episode's
    caption audit row (``stage`` caption, not a ``shard_audit``) carries ``input_digests``,
    one per caption row in record order (None: no Captioner call). A row list whose length is
    not the record's, or an episode without one, gives all None (certify refuses:
    ``inputs_not_isolated``). Only hex digests are kept."""
    by_uid: dict[str, list[Any]] = {}
    for a in audits:
        if (a.get("stage") == "caption" and not a.get("shard_audit") and a.get("uid")
                and isinstance(a.get("input_digests"), list)):
            by_uid.setdefault(str(a["uid"]), []).append(a["input_digests"])
    out: dict[str, list[str | None]] = {}
    for rec in records:
        if getattr(rec, "stage", None) != "caption":
            continue
        n = len(_caption_rows(rec))
        lists = {tuple(_hex_or_none(x) for x in d) for d in by_uid.get(rec.uid, [])}
        got = list(next(iter(lists))) if len(lists) == 1 else []
        out[rec.uid] = got if len(got) == n else [None] * n
    return out


def write_digests(paths: Paths, side: str, job: str, records: Sequence[Any],
                  audits: Sequence[Mapping[str, Any]]) -> Path:
    """The side's digest JSON (:data:`DIGESTS`) beside its records."""
    fn = segcap_digests if side in SEGCAP_DIGEST_SIDES else caption_digests
    d = fn(records, audits)
    out = record_path(paths, side, job, "caption").with_name(DIGESTS)
    out.write_text(json.dumps(d, sort_keys=True) + "\n", encoding="utf-8")
    return out


def digest_path(paths: Paths, side: str, job: str) -> Path:
    return record_path(paths, side, job, "caption").with_name(DIGESTS)


def fetch_records(paths: Paths, st: wh.State, run: Callable[..., Any] = subprocess.run, *,
                  audits: Callable[[str, str], list[dict[str, Any]]] | None = None) -> None:
    """Export each job's segment / caption records from its store (the run's, or the one its
    state record names: recertify's oracle and noise jobs are in (b)'s; a Volume read), and
    for the sides the v4 caption certificate pairs (:data:`DIGEST_SIDES`) the digest JSON
    (:func:`write_digests`) from the job's audit lines (``audits``: injectable), when the state
    has the stage-isolated caption job they pair with."""
    import h0_rehearsal as h0

    from flora_deploy.records import StageRecord

    default = f"modal-volume://{wh.STORE_VOLUME}/{st.run_id}"
    # a substituted ``run`` (a test, a dry export) exports nothing real: the audits are then
    # read only through a substituted ``audits`` too, never from the real store
    read_audits = audits or (job_audits if run is subprocess.run else None)
    for side, step, key in record_sets():
        for job, r in sorted(jobs_of(st, step, key).items()):
            if "job_id" not in r:
                continue
            store = r.get("store") or default
            for stage in r.get("stages") or CERT_STAGES:
                out = record_path(paths, side, job, stage)
                out.parent.mkdir(parents=True, exist_ok=True)
                run([h0.flora_bin(), "results", "--job", r["job_id"], "--stage", stage,
                     "--store", store, "--format", "jsonl", "--out", str(out)], check=True)
                # which job these records are: the lab reads (d)'s only when this matches
                wh.records_source_path(out).write_text(json.dumps(
                    {"job_id": r["job_id"], "store": store, "stage": stage}) + "\n")
            cap = record_path(paths, side, job, "caption")
            # digests only pair with a stage-isolated caption job (J2) of the same job key
            if (read_audits is not None and side in DIGEST_SIDES and cap.exists()
                    and job in jobs_of(st, SETS["isolated"])):
                recs = [StageRecord.from_json(x) for x in cap.read_text().splitlines()
                        if x.strip()]
                write_digests(paths, side, job, recs, read_audits(r["job_id"], store))


def analyze(paths: Paths, *, derived: Mapping[str, str] | None = None, cert_tmp: Path,
            cli: Callable[[list[str]], tuple[int, str]] = run_cli
            ) -> tuple[Findings, dict[Path, str], list[Path]]:
    """Everything this script would write: (findings, {path: new text}, certificate files)."""
    from flora_deploy.planner.rates import RateTable

    if not paths.state.exists():
        raise SystemExit(f"{paths.state}: no runbook state (run scripts/wave_h.py first)")
    st = wh.State.from_json(json.loads(paths.state.read_text(encoding="utf-8")))
    date = utc(step_finished(st, "balanced") or run_stamp(st), "%Y-%m-%d")
    ms = measure(st)
    refusals = [r for k in (("balanced", "segment"), ("balanced", "caption"))
                for r in ms[k].refusals]
    outputs: dict[Path, str] = {}
    source = f"reports/wave-h/report.md (Wave H-lite run {st.run_id})"
    if not refusals:
        data = json.loads(paths.rates.read_text(encoding="utf-8"))
        new = updated_rates(data, ms, run_id=st.run_id, date=date, source=source)
        table = RateTable.from_dict(new, source=str(paths.rates))  # the loader must accept it
        outputs[paths.rates] = dump_json(new)
        if paths.costs_doc.exists():  # its generated tables follow the new rates
            import cost_table  # scripts/, on sys.path above

            outputs[paths.costs_doc] = cost_table.render(
                paths.costs_doc.read_text(encoding="utf-8"),
                table=replace(table, source="planner/rates.json"))
    outcomes = certify_all(paths, wh.certify_view(st), cert_tmp,
                           derived=derived_content_used(st) if derived is None else derived,
                           cli=cli)
    labs = {k: lab_result(paths, st, k) for k in ("tempseg", "captioner")}
    certified_by_finalize = any(o.get("cert") for o in outcomes)
    billing, warnings = billing_rows(
        st, certify_status=("finalize: " + stage_verdicts(outcomes)
                            if certified_by_finalize else None))
    priced = RateTable.from_dict(json.loads(outputs.get(paths.rates)
                                            or paths.rates.read_text(encoding="utf-8")),
                                 source=str(paths.rates))
    f = Findings(st=st, ms=ms, rates_ok=not refusals, refusals=refusals, outcomes=outcomes,
                 l0=l0_rows(labs), gate=gate_a(labs),
                 cost={p: usd_per_video_h(ms, p) for p in PRESET_KINDS}, billing=billing,
                 warnings=warnings, date=date,
                 planned={p: planned_usd_per_video_h(priced, p) for p in PRESET_KINDS},
                 certified=frozenset(certified_kinds([paths.certs, cert_tmp])))
    outputs[paths.report] = render_report(f)
    doc = paths.engines_doc.read_text(encoding="utf-8")
    outputs[paths.engines_doc] = replace_block(doc, BLOCK_RESULTS, results_block(f))
    body = cost_block(f)
    if body is not None:
        readme = paths.readme.read_text(encoding="utf-8")
        outputs[paths.readme] = replace_block(readme, BLOCK_COST, body)
    certs = [cert_tmp / f"{o['cert']['id']}.json" for o in outcomes
             if o.get("cert") and o["cert"]["verdict"] != "fail"]
    readme = outputs.get(paths.readme) or paths.readme.read_text(encoding="utf-8")
    outputs[paths.readme] = certification_blocks(readme, [paths.certs, cert_tmp], table=priced)
    if paths.limitations_doc.exists():  # its status tables follow the certificates and rates
        import limitations_status  # scripts/, on sys.path above

        outputs[paths.limitations_doc] = limitations_status.render(
            paths.limitations_doc.read_text(encoding="utf-8"), cert_dirs=[paths.certs, cert_tmp],
            table=RateTable.from_dict(json.loads(outputs.get(paths.rates)
                                                 or paths.rates.read_text(encoding="utf-8")),
                                      source=str(paths.rates)))
    return f, outputs, certs


def engine_facts_line(cert: Mapping[str, Any]) -> str | None:
    """What the engines did under a certificate's evidence (review 13 B1 / W1): records whose
    engine preempted or could not say, and the compile caches the records ran. Numbers and
    cache identities only."""
    pre, cc = cert.get("preemptions"), cert.get("compile_cache")
    if pre is None and cc is None:
        return None
    pre = pre or {}
    return (f"{cert.get('stage')}: engine preemptions: {pre.get('preempted', '?')} preempted / "
            f"{pre.get('unknown', '?')} unknown record(s); compile caches: "
            + ", ".join((cc or {}).get("covers") or ["unknown"]))


def print_findings(f: Findings, out=print) -> None:
    out(f"Wave H-lite run {f.st.run_id} (plan {run_plan(f.st)}): spent {usd(f.st.spent)} of "
        f"{usd(global_cap(f.st))}")
    for w in f.warnings:
        out(f"  WARNING {w}")
    if f.rates_ok:
        for k in (("balanced", "segment"), ("balanced", "caption")):
            m = f.ms[k]
            out(f"  rate {k[0]} {k[1]} on {m.gpu}: {m.rate} container-s/video-s "
                f"±{m.uncertainty:.0%} (n={m.n}), cold start {m.startup_s} s")
    else:
        out("  rates: REFUSED, planner/rates.json unchanged:")
        for r in f.refusals:
            out(f"    - {r}")
    for p, c in f.cost.items():
        if c:
            out(f"  {p}: ${c['usd_per_video_h']:.2f}/video-h ±{c['uncertainty']:.0%} at list "
                "prices")
    out(f"  Gate A: plugin {f.gate['recommendation']} ({f.gate['why']})")
    rows, missing = cert_rows(f.outcomes)
    out(f"  certificate verdict: {overall_verdict(f.outcomes)}")
    for r in rows:
        out(f"    {r[0]} {r[1]} {r[2]}: {r[3]} (episodes {r[4]}, noise {r[5]}) {r[6]}")
    for o in f.outcomes:
        if o.get("cert") and (line := engine_facts_line(o["cert"])):
            out(f"    {line}")
    if missing:
        out("  missing for certified:")
        for m in missing:
            out(f"    - {m}")
    for o in f.outcomes:  # the console only: certify's own words never reach a written file
        if o.get("refused") and o.get("output"):
            out(f"  flora certify output, {o['stage']} ({o['job']}): "
                + " ".join(str(o["output"]).split())[-400:])


def cert_key(c: Mapping[str, Any]) -> tuple[Any, ...]:
    """What a certificate is for: a newer one with the same key replaces it."""
    return (c.get("preset"), c.get("stage"), c.get("kind"), c.get("gpu"),
            tuple(c.get("covered_pools") or ()), c.get("engine_spec_hash"))


def freeze_locks(certs: Sequence[Path], cert_dir: Path, new: Sequence[Path]
                 ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[str]]:
    """The freeze locks (``validate/certfreeze.py``) ``--write`` writes beside the certificates
    of this run, the equivalent of ``scripts/cert_freeze.py --lock`` for them, so the drift
    gate can name the modules that change later: ``(certificate, key document)`` for each
    certificate in ``new`` (rewritten) or whose committed lock is missing or inconsistent. The
    document is the one ``flora certify`` locked it with, else the key this tree computes.
    Second: why a certificate cannot be locked (its key no longer matches this tree)."""
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    errors: list[str] = []
    fresh = {p.name for p in new}
    for p in certs:
        cert = json.loads(p.read_text(encoding="utf-8"))
        if p.name not in fresh and F.read_freeze(cert_dir, cert) is not None:
            continue
        try:
            out.append((cert, F.lock_document(cert, p.parent)))
        except Exception as e:  # noqa: BLE001 - reported; --write refuses
            errors.append(f"{cert.get('id')}: {e}")
    return out, errors


def superseded(cert_dir: Path, prev: Sequence[str], new: Sequence[Path]) -> list[Path]:
    """The certificates this script wrote before that a certificate of this run replaces:
    same :func:`cert_key`, another id. A certificate nothing new replaced (its inputs are
    missing, certify refused) is kept: nothing is ever removed implicitly."""
    keys = {cert_key(json.loads(p.read_text(encoding="utf-8"))) for p in new}
    ids = {p.stem for p in new}
    out = []
    for i in prev:
        p = cert_dir / f"{i}.json"
        if i in ids or not p.exists():
            continue
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue  # not ours to judge: left in place
        if cert_key(old) in keys:
            out.append(p)
    return out


def main(argv: list[str] | None = None, *, cli: Callable[[list[str]], tuple[int, str]] = run_cli
         ) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    ap.add_argument("--fetch", action="store_true",
                    help="export the jobs' records with `flora results` first (reads the Volume)")
    ap.add_argument("--root", type=Path, default=ROOT, help="repository root")
    ap.add_argument("--state", type=Path)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--lab", type=Path, help="the lab JSON directory (default reports/wave-h)")
    ap.add_argument("--cert-dir", type=Path)
    ap.add_argument("--derived-content", action="append", metavar="NAME=SHA256",
                    help="override the pinned derived-weight content digests")
    args = ap.parse_args(argv)
    paths = Paths(root=args.root, state=args.state, records=args.records, certs=args.cert_dir,
                  lab=args.lab)
    derived = None
    if args.derived_content:
        derived = dict(x.split("=", 1) for x in args.derived_content)
    if args.fetch:
        st = wh.State.from_json(json.loads(paths.state.read_text(encoding="utf-8")))
        fetch_records(paths, wh.certify_view(st))
    with tempfile.TemporaryDirectory(prefix="wave-h-finalize-") as tmp:
        f, outputs, certs = analyze(paths, derived=derived, cert_tmp=Path(tmp), cli=cli)
        print_findings(f)
        prev = []
        if paths.manifest.exists():
            prev = json.loads(paths.manifest.read_text(encoding="utf-8")).get("certificates", [])
        ids = sorted(p.stem for p in certs)
        stale = superseded(paths.certs, prev, certs)
        owned_ids = sorted((set(prev) - {s.stem for s in stale}) | set(ids))
        changed = {p: t for p, t in outputs.items()
                   if not p.exists() or p.read_text(encoding="utf-8") != t}
        new_certs = [c for c in certs if not (paths.certs / c.name).exists()
                     or (paths.certs / c.name).read_bytes() != c.read_bytes()]
        for p, t in changed.items():
            old = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
            diff = list(difflib.unified_diff(old, t.splitlines(), lineterm="", n=0))
            add = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
            rem = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
            print(f"  {'update' if args.write else 'would update'} {p} (+{add} -{rem})")
        for c in new_certs:
            print(f"  {'write' if args.write else 'would write'} {paths.certs / c.name}")
        locks, lock_errors = freeze_locks(certs, paths.certs, new_certs)
        for cert, _doc in locks:
            print(f"  {'lock' if args.write else 'would lock'} "
                  f"{F.freeze_path(paths.certs, cert['id'])}")
        for s in stale:
            print(f"  {'remove' if args.write else 'would remove'} superseded {s}")
        if not changed and not new_certs and not stale and not locks:
            print("  nothing to change")
        if lock_errors:
            for e in lock_errors:
                print(f"  REFUSED freeze lock: {e}")
            if args.write:  # fail closed: a certificate without its lock is not committed
                print("refused: nothing written (a certificate this tree cannot lock would "
                      "fail the drift gate; docs/certificate-freeze.md)")
                return 1
        if args.write:
            for p, t in changed.items():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(t, encoding="utf-8")
            paths.certs.mkdir(parents=True, exist_ok=True)
            for c in new_certs:
                shutil.copyfile(c, paths.certs / c.name)
            for cert, doc in locks:  # scripts/cert_freeze.py --lock for what it wrote
                F.write_freeze(cert, doc, paths.certs)
            for s in stale:
                s.unlink()
                F.freeze_path(paths.certs, s.stem).unlink(missing_ok=True)  # its orphan lock
            man = dump_json({"run_id": f.st.run_id, "certificates": owned_ids})
            if not paths.manifest.exists() or paths.manifest.read_text(encoding="utf-8") != man:
                paths.manifest.parent.mkdir(parents=True, exist_ok=True)
                paths.manifest.write_text(man, encoding="utf-8")
        elif changed or new_certs or stale:
            print("dry run: nothing written (pass --write to apply)")
    return 0 if f.rates_ok else 3


if __name__ == "__main__":
    sys.exit(main())
