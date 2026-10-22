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
3. **Report** (``docs/wave-h-report.md``): spend per step (tracked vs billed, with the billing
   lag), measured $/video-h for ``exact`` and ``balanced`` at list prices (formula shown), L0,
   Gate A with the ``dt_bias`` plugin recommendation (the rule in ``docs/balanced-engines.md``)
   and the certificate summary. Numbers only: no NVIDIA text, captions, prompts or episode ids.
4. **Docs**: the ``Results`` table of ``docs/balanced-engines.md`` and the ``Cost`` block of
   ``README.md`` (2-3 plain lines), each between ``<!-- wave-h:<name> -->`` /
   ``<!-- /wave-h:<name> -->`` markers. The script fills exactly those blocks and refuses when
   the markers are missing; it never inserts a section. Until the balanced rates are measured
   the README block is left as it is.

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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wave_h as wh  # noqa: E402 - the runbook's state format, caps and constants

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
SETS = {"exact": "exact_e2e", "noise": "noise", "balanced": "balanced"}
BLOCK_RESULTS = "results"
BLOCK_COST = "cost"
MISSING_HELP = {
    "too_few_samples": "at least {min_samples} candidate episodes of the pool (has {n})",
    "no_noise_run": "a noise run: the oracle repeated on at least {min_noise} of these "
                    "episodes (none for this pool)",
    "too_few_noise_samples": "at least {min_noise} noise-run episodes (has {noise_n})",
    "incomplete_coverage": "every oracle episode labelled (coverage {coverage})",
    "no_l0": "the L0 JSON of {lab} (input parity on the captured requests)",
    "metric_not_evaluable": "every metric evaluable ({metrics} had no value)",
    "divergence_above_noise_floor": "every check within its threshold (failing: {failing})",
    "l0_failed": "L0 passing on every captured request",
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

    def __post_init__(self) -> None:
        r = self.root
        w = r / "reports" / "wave-h"
        self.state = self.state or w / "state.json"
        self.records = self.records or w / "records"
        self.lab = self.lab or w
        self.rates = self.rates or r / "src" / "flora_deploy" / "planner" / "rates.json"
        self.certs = self.certs or r / "src" / "flora_deploy" / "config" / "certificates"
        self.report = self.report or r / "docs" / "wave-h-report.md"
        self.engines_doc = self.engines_doc or r / "docs" / "balanced-engines.md"
        self.readme = self.readme or r / "README.md"
        self.manifest = self.manifest or w / "finalize.json"

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


def jobs_of(st: wh.State, step: str) -> dict[str, dict[str, Any]]:
    s = st.steps.get(step)
    return dict((s.result.get("jobs") if s else None) or {})


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
        rows[0].update(container_s_per_video_s=m.rate, uncertainty=m.uncertainty,
                       status="measured", source=source, date=date, n=m.n,
                       image=engine_image(preset, kind), provenance=prov)
        k = len(m.startups)
        spread = (f"{m.startups[0]:.0f} s, one container: no range, so the error band is "
                  f"±{SINGLE_SAMPLE_FLOOR:.0%}" if k == 1
                  else f"{min(m.startups):.0f}-{max(m.startups):.0f} s")
        srow = {"preset": preset, "kind": kind, "startup_s": m.startup_s,
                "uncertainty": m.startup_uncertainty, "status": "measured", "n": k,
                "provenance": (f"Wave H-lite run {run_id}: spawn to first attempt of n={k} "
                               f"fresh {m.gpu} container(s), {spread} ({source})"),
                "date": date, "source": source}
        existing = [r for r in new["startup"] if r["preset"] == preset and r["kind"] == kind]
        if existing:
            # the compile-cache state the startup was measured with is not in the state; keep
            # the one recorded beside the row (``wave_h.startup_basis`` reads it), so --write
            # stays idempotent
            if "compile_cache" in existing[0]:
                srow = {k: v for k, v in srow.items() if k != "n"} | {
                    "compile_cache": existing[0]["compile_cache"], "n": k}
                srow = {**{x: srow[x] for x in ("preset", "kind", "startup_s", "uncertainty",
                                                "status", "compile_cache", "n")}, **srow}
            existing[0].clear()
            existing[0].update(srow)
        else:
            star = next((i for i, r in enumerate(new["startup"])
                         if r["preset"] == preset and r["kind"] == "*"), len(new["startup"]))
            new["startup"].insert(star, srow)
    new["updated"] = max(str(new.get("updated") or ""), date)
    return new


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
    return {"kind": kind, "baseline": {"l0": k.get("l0") or []}, "gate_a": k.get("gate_a")}


def l0_for_certify(lab: Mapping[str, Any] | None, kind: str, gpu: str) -> dict[str, Any] | None:
    """What ``flora certify --l0`` reads (kind, the pool GPU, L0 rows, Gate A), without the
    lab's episode rows. The lab records ``nvidia-smi``'s name; it is checked against the pool
    GPU here and replaced by it, since ``certify.l0_summary`` compares the strings."""
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
                    "gate_a": lab.get("gate_a")}
    return {"kind": lab.get("kind") or kind, "baseline": {"gpu": gpu, "l0": base.get("l0") or []},
            "gate_a": lab.get("gate_a")}


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
    ctx = {"min_samples": th.get("min_samples"), "min_noise": th.get("min_noise_samples"),
           "n": n, "noise_n": noise.get("n", 0),
           "coverage": (cert.get("metrics") or {}).get("coverage"),
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


def billing_rows(st: wh.State) -> tuple[list[list[str]], list[str]]:
    rows, warnings = [], []
    for name in wh.STEPS:
        s = st.steps.get(name)
        if s is None:
            continue
        last = max((a.last_activity for a in s.attempts), default=None)
        if s.billed_usd is None:
            billing = "shared app: global total only" if s.apps else "no app"
        elif st.billed_at is None or last is None:
            billing = "not read"
        elif st.billed_at - last < wh.BILLING_LAG_S:
            billing = (f"partial: read {st.billed_at - last:.0f} s after the step; final "
                       f"after {utc(last + wh.BILLING_LAG_S)}")
            warnings.append(f"{name}: billing was read {st.billed_at - last:.0f} s after the "
                            f"step (< {wh.BILLING_LAG_S} s lag): billed "
                            f"{usd(s.billed_usd)} vs tracked {usd(s.tracked_usd)} is partial; "
                            f"re-run the runbook's billing read after "
                            f"{utc(last + wh.BILLING_LAG_S)}, then this script")
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
        rows.append([wh.TITLES[name], step_status(st, name), str(len(s.attempts)),
                     usd(wh.CAPS_USD[name]),
                     usd(s.tracked_usd), usd(s.charged(st.billed_at)), usd(s.billed_usd),
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


def render_report(f: Findings) -> str:
    st = f.st
    L = ["# Wave H-lite report", "",
         "Generated by `scripts/wave_h_finalize.py` from the runbook state (`scripts/wave_h.py`, "
         "`docs/wave-h-runbook.md`).",
         "Numbers only: no NVIDIA text, captions, prompts or episode ids; those stay in the",
         "git-ignored `reports/wave-h/`.", "",
         f"Run `{st.run_id}`, finished {utc(run_stamp(st))}.", "", "## Spend per step", "",
         "Tracked: container-seconds x list $/h from `flora report` (the lab: its wall time). "
         "Charged: tracked, or a killed / unreported attempt at its budget until billing is "
         f"read {wh.BILLING_LAG_S} s after it. Billed: the billed delta of the apps only that "
         "step used (`scripts/spend.py`).", ""]
    L += table(["step", "status", "attempts", "cap", "tracked", "charged", "billed", "billing"],
               f.billing)
    tracked = sum(s.tracked_usd for s in st.steps.values())
    charged = sum(s.charged(st.billed_at) for s in st.steps.values())
    L += ["", f"Total: **{usd(st.spent)}** of the {usd(global_cap(st))} global cap: the "
          "larger of charged and billed, the figure the runbook gates on.", "",
          f"- tracked {usd(tracked)}: container-seconds x list $/h of every attempt;",
          f"- charged {usd(charged)}: tracked, with a killed or unreported attempt at its budget "
          f"until billing is read {wh.BILLING_LAG_S} s after it"
          + (f" ({usd(charged - tracked)} over tracked)" if charged - tracked > 0.005 else "")
          + ";",
          f"- billed {usd(st.billed_total)}: Modal's billed delta of every app (read "
          f"{utc(st.billed_at)})."]
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
    L += table(["preset", "terms", "$/video-h (busy)", "this run (cold starts, idle, slack)"],
               rows)
    warm = run_usd_per_video_h_after_startup(st, "balanced")
    L += ["", "$/video-h (busy) counts busy container-seconds only: steady state at scale, "
          "without cold starts or idle time until scale-down. This run's figure is everything "
          "its jobs spent"
          + (f"; without the cold starts, `balanced` was {usd(warm)}/video-h (idle kept)"
             if warm is not None else "")
          + ". One small job (n=1): the band is a nominal convention, not a statistical "
          "interval."]
    ex, bal = f.cost.get("exact"), f.cost.get("balanced")
    if ex and bal and bal["usd_per_video_h"]:
        L += ["", f"`exact` / `balanced`: {ex['usd_per_video_h'] / bal['usd_per_video_h']:.1f}x "
              "(segment + caption only; rewards are the same on both presets)."]

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
          f"Overall: **{overall_verdict(f.outcomes)}**. Reference: `exact` in NVIDIA's default "
          "mode; certificates in `src/flora_deploy/config/certificates/` (metrics only).", ""]
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
           f"(run `{f.st.run_id}`); certificate **{overall_verdict(f.outcomes)}**")
    bal, ex = f.cost.get("balanced"), f.cost.get("exact")
    if bal and ex:
        ratio = ex["usd_per_video_h"] / bal["usd_per_video_h"]
        cost = (f"**${bal['usd_per_video_h']:.2f}/video-h** ±{bal['uncertainty']:.0%} vs `exact` "
                f"${ex['usd_per_video_h']:.2f} ({ratio:.1f}x cheaper), list prices, measured")
    elif bal:
        cost = f"**${bal['usd_per_video_h']:.2f}/video-h** ±{bal['uncertainty']:.0%}, list prices"
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
    return "\n".join([*table(["Item", f"Result (Wave H-lite, {f.date}; `docs/wave-h-report.md`)"],
                             rows)])


def cost_block(f: Findings) -> str | None:
    bal, ex = f.cost.get("balanced"), f.cost.get("exact")
    if not bal:
        return None
    # README wording (no internal run names): the busy-only figure, said plainly to be busy-only
    vs = f" (`exact`: ${ex['usd_per_video_h']:.2f})" if ex else ""
    return (f"`balanced` costs about **${bal['usd_per_video_h']:.2f} per video-hour** of GPU "
            f"work (±{bal['uncertainty']:.0%}) for segmentation + captioning{vs},\n"
            f"measured {f.date} at Modal list prices. Startup (a few minutes per container), "
            "idle time and\n"
            "rewards are extra and dominate small jobs: `flora plan` shows the all-in estimate "
            "for your dataset.")


# ---------------------------------------------------------------------------------------------
# Driver


def fetch_records(paths: Paths, st: wh.State, run: Callable[..., Any] = subprocess.run) -> None:
    """Export each job's segment / caption records from the run's store (a Volume read)."""
    import h0_rehearsal as h0

    store = f"modal-volume://{wh.STORE_VOLUME}/{st.run_id}"
    for side, step in SETS.items():
        for job, r in sorted(jobs_of(st, step).items()):
            if "job_id" not in r:
                continue
            for stage in CERT_STAGES:
                out = record_path(paths, side, job, stage)
                out.parent.mkdir(parents=True, exist_ok=True)
                run([h0.flora_bin(), "results", "--job", r["job_id"], "--stage", stage,
                     "--store", store, "--format", "jsonl", "--out", str(out)], check=True)
                # which job these records are: the lab reads (d)'s only when this matches
                wh.records_source_path(out).write_text(json.dumps(
                    {"job_id": r["job_id"], "store": store, "stage": stage}) + "\n")


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
    source = f"docs/wave-h-report.md (Wave H-lite run {st.run_id})"
    if not refusals:
        data = json.loads(paths.rates.read_text(encoding="utf-8"))
        new = updated_rates(data, ms, run_id=st.run_id, date=date, source=source)
        RateTable.from_dict(new, source=str(paths.rates))  # the loader must accept it
        outputs[paths.rates] = dump_json(new)
    outcomes = certify_all(paths, st, cert_tmp,
                           derived=derived_content_used(st) if derived is None else derived,
                           cli=cli)
    labs = {k: lab_result(paths, st, k) for k in ("tempseg", "captioner")}
    billing, warnings = billing_rows(st)
    f = Findings(st=st, ms=ms, rates_ok=not refusals, refusals=refusals, outcomes=outcomes,
                 l0=l0_rows(labs), gate=gate_a(labs),
                 cost={p: usd_per_video_h(ms, p) for p in PRESET_KINDS}, billing=billing,
                 warnings=warnings, date=date)
    outputs[paths.report] = render_report(f)
    doc = paths.engines_doc.read_text(encoding="utf-8")
    outputs[paths.engines_doc] = replace_block(doc, BLOCK_RESULTS, results_block(f))
    body = cost_block(f)
    if body is not None:
        readme = paths.readme.read_text(encoding="utf-8")
        outputs[paths.readme] = replace_block(readme, BLOCK_COST, body)
    certs = [cert_tmp / f"{o['cert']['id']}.json" for o in outcomes
             if o.get("cert") and o["cert"]["verdict"] != "fail"]
    return f, outputs, certs


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
    ap.add_argument("--cert-dir", type=Path)
    ap.add_argument("--derived-content", action="append", metavar="NAME=SHA256",
                    help="override the pinned derived-weight content digests")
    args = ap.parse_args(argv)
    paths = Paths(root=args.root, state=args.state, records=args.records, certs=args.cert_dir)
    derived = None
    if args.derived_content:
        derived = dict(x.split("=", 1) for x in args.derived_content)
    if args.fetch:
        st = wh.State.from_json(json.loads(paths.state.read_text(encoding="utf-8")))
        fetch_records(paths, st)
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
        for s in stale:
            print(f"  {'remove' if args.write else 'would remove'} superseded {s}")
        if not changed and not new_certs and not stale:
            print("  nothing to change")
        if args.write:
            for p, t in changed.items():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(t, encoding="utf-8")
            paths.certs.mkdir(parents=True, exist_ok=True)
            for c in new_certs:
                shutil.copyfile(c, paths.certs / c.name)
            for s in stale:
                s.unlink()
            man = dump_json({"run_id": f.st.run_id, "certificates": owned_ids})
            if not paths.manifest.exists() or paths.manifest.read_text(encoding="utf-8") != man:
                paths.manifest.parent.mkdir(parents=True, exist_ok=True)
                paths.manifest.write_text(man, encoding="utf-8")
        elif changed or new_certs or stale:
            print("dry run: nothing written (pass --write to apply)")
    return 0 if f.rates_ok else 3


if __name__ == "__main__":
    sys.exit(main())
