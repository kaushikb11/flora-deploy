"""Independent check of a written certificate before it is published.

The reviewer brief is ``docs/certificate-check.md``.

Usage::

    uv run python scripts/cert_check.py            # the certified run's defaults
    uv run python scripts/cert_check.py --json     # the same rows as JSON
    uv run python scripts/cert_check.py --run-root reports/wave-h/certified \\
        --cert-dir src/flora_deploy/config/certificates --report docs/wave-h-report.md

Run it after ``scripts/wave_h.py --plan certified`` and ``scripts/wave_h_finalize.py --write``,
before the README flip and the push. It reads the records (``<run-root>/records/{exact,noise,
balanced,witness}/<job>/<stage>.jsonl``), the run state (``<run-root>/state.json``), the lab
JSON (``reports/wave-h/balanced-<kind>.json``), the certificates and their freeze locks, and
the report. It writes nothing and starts nothing. Exit 0 only when every row is PASS.

**Independence.** Nothing here imports ``validate/certify.py``. The metric math (interval IoU,
the monotone alignment, Levenshtein, Spearman, the means and their episode basis), the pass rule
(floors, the absolute ``count_mismatch`` floor that lets about one episode in ten differ in
subtask count, tolerance x noise, caps), the policy bounds and the verdict are reimplemented
from the definitions in ``docs/certification.md``, with their own constants. A bug in the
generator therefore shows up as a mismatch here instead of being reproduced. What *is* taken
from the code is what the certificate must agree with: the current key
(``validate.certkey.key_document``), the current engine settings
(``engines.offline_llm.EngineSpec``), the pinned derived digests, the freeze-lock module hashes,
the drift guard (``validate.certfreeze.check`` and ``tests/unit/test_certificate_drift.py``),
the license shingle detector (``scripts/license_shingles.py``) and the runbook's spend model
(``scripts/_runbook.py``, read only).

Checks (one or more rows each):

1. **metrics**: every certificate metric recomputed from the raw records (oracle vs balanced,
   oracle vs noise, oracle vs the cold witness), with ``n``, ``count_matched``, ``missing``,
   ``unpaired`` and ``coverage``, within :data:`TOL`;
2. **pass rule**: knobs within policy, recorded floors equal to the policy's, every threshold
   and check recomputed, the count-flip tolerance, the provisional and fail reasons, the
   verdict, and whether it may be published (``certified``, or ``--allow-provisional``);
3. **coverage**: the hashed episode ids equal the matched episodes; noise and witness episodes
   are the oracle's; one stage hash and one key on the balanced records, equal to the current
   code's key; the oracle binding; the embodiment and GPU;
4. **evidence**: a forced-cold container contributed records and every engine record states
   its compile cache; no preemptions (or the certificate is provisional); L0 and Gate A
   present, measured under the current engine settings;
5. **freeze lock**: present, consistent, equal to the current key and module hashes; the drift
   guard passes;
6. **leakage**: no NVIDIA text (shingles against the fetched NVIDIA code), and no record string
   (captions, ids, dataset names, NVIDIA metadata) in the certificate, its lock or the report;
   every string field of the certificate is one of ours;
7. **spend**: the report's tracked, charged, billed and total figures, per step and overall,
   equal what the state gives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Policy, restated from docs/certification.md (not imported from validate/) ---------------
SUPPORTED_VERSIONS = ("2", "3")  # generator versions whose rules this checker knows
CURRENT_RULE = "3"  # the rule with the count-flip tolerance (the user's decision, review 13)
TOLERANCE_BOUNDS = (1.0, 2.0)
MIN_SAMPLES = 10
MIN_NOISE_SAMPLES = 5
MAX_MISSING = 2  # v3: oracle episodes without a candidate record (the spares)
MIN_COVERAGE = 1.0  # v2
FLOOR: dict[str, float] = {
    "iou_divergence": 0.02, "boundary_error_s": 0.1, "count_mismatch": 0.05,
    "label_disagreement": 0.02, "caption_mismatch": 0.05, "caption_edit_distance": 0.02,
    "caption_similarity_divergence": 0.02, "expected_score_abs_delta": 0.02,
    "score_mismatch": 0.05, "per_step_abs_delta": 0.02, "rank_divergence": 0.05,
    "length_mismatch": 0.0,
}
# v3: up to about one episode in ten may differ in subtask count (1 in 12 passes, 2 fail).
ABS_FLOOR_V3: dict[str, float] = {"count_mismatch": 0.10}
COUNT_FLIP_SHARE = 0.10  # one flip in ten episodes must pass
CAP: dict[str, float] = {"expected_score_abs_delta": 0.05}
METRICS_V2 = {
    "segment": ("iou_divergence", "boundary_error_s", "count_mismatch", "label_disagreement"),
    "caption": ("caption_mismatch", "caption_edit_distance"),
    "reward.terminal": ("expected_score_abs_delta", "score_mismatch"),
    "reward.dense": ("per_step_abs_delta", "rank_divergence", "length_mismatch"),
}
METRICS_V3 = {**METRICS_V2,
              "caption": ("caption_mismatch", "caption_edit_distance", "count_mismatch")}
COUNT_BASIS_STAGES = ("segment", "caption")  # v3: other metrics over count-matched episodes
LAB_KIND = {"segment": "tempseg", "caption": "captioner"}
FORCED_COLD = "cold:forced"
WITNESS_CLASS = "cold-witness"
RETURN_ONLY_KWARGS = ("logprobs_mode", "max_logprobs")  # not engine settings (Gate A only)
CACHE_ID = re.compile(r"^(cold|cold:forced|unknown|warm:[0-9a-f]{6,64}:[0-9a-f]{6,64}"
                      r"|legacy:[0-9a-f]{6,64})\Z")
TOL = 2e-6  # certificates round means and thresholds to 6 places
SIDES = ("exact", "noise", "balanced", "witness")
DEFAULT_RUN_ROOT = ROOT / "reports" / "wave-h" / "certified"
DEFAULT_LAB_DIR = ROOT / "reports" / "wave-h"
DEFAULT_REPORT = ROOT / "docs" / "wave-h-report.md"
DRIFT_TEST = "tests/unit/test_certificate_drift.py"
NV_PINS = ("tempseg_code", "captioner_code", "rewarder")
NV_SKIP = {"tokenizer.json", "LICENSE.pdf"}

PASS, FAIL = "PASS", "FAIL"


@dataclass
class Row:
    check: str  # "1 metrics" ... "7 spend"
    item: str
    status: str  # PASS | FAIL
    detail: str = ""


@dataclass
class Inputs:
    run_root: Path = DEFAULT_RUN_ROOT
    cert_dir: Path | None = None  # default: validate.certkey.CERT_DIR
    lab_dir: Path = DEFAULT_LAB_DIR
    report: Path = DEFAULT_REPORT
    records: Path | None = None  # default: <run_root>/records
    state: Path | None = None  # default: <run_root>/state.json
    job: str = "human"
    embodiment: str = "human"
    preset: str = "balanced"
    stages: tuple[str, ...] = ("segment", "caption")
    nv_root: Path | None = None  # default: the fetched NVIDIA code cache (layout.py)
    allow_provisional: bool = False
    # the drift test file run as a subprocess when cert_dir is the committed directory
    drift_runner: Callable[[Path], tuple[int, str]] | None = None

    def __post_init__(self) -> None:
        from flora_deploy.validate.certkey import CERT_DIR

        self.run_root = Path(self.run_root)
        self.cert_dir = Path(self.cert_dir) if self.cert_dir is not None else CERT_DIR
        self.records = Path(self.records) if self.records else self.run_root / "records"
        self.state = Path(self.state) if self.state else self.run_root / "state.json"
        self.lab_dir, self.report = Path(self.lab_dir), Path(self.report)


# ---------------------------------------------------------------------------------------------
# Small helpers


def canonical(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def sha(doc: Any) -> str:
    return hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()


def uid_hash(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def r6(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def close(a: Any, b: Any, tol: float = TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def norm_gpu(name: Any) -> str | None:
    from flora_deploy.engines.offline_llm import EngineArgsError, normalize_gpu

    try:
        return normalize_gpu(str(name))
    except EngineArgsError:
        return None


def engine_settings_hash(hashed: Mapping[str, Any] | None) -> str | None:
    """sha256 of an ``EngineSpec.hashed()`` document without the return-only kwargs (what the
    lab records per engine it built; ``docs/certification.md``, ``--l0``)."""
    if not isinstance(hashed, Mapping):
        return None
    doc = dict(hashed)
    doc["engine_kwargs"] = {k: v for k, v in dict(doc.get("engine_kwargs") or {}).items()
                            if k not in RETURN_ONLY_KWARGS}
    return sha(doc)


def usd(x: float | None) -> str:
    return "-" if x is None else f"${x:,.2f}"


# ---------------------------------------------------------------------------------------------
# Metric math (docs/certification.md, "Metrics"), reimplemented


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s1, e1 = a
    s2, e2 = b
    overlap = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - overlap
    if union <= 0:
        return 1.0 if (s1, e1) == (s2, e2) else 0.0
    return overlap / union


def alignment(A: Sequence[tuple[float, float]], B: Sequence[tuple[float, float]]
              ) -> list[tuple[int, int, float]]:
    """Monotone matching of two ordered interval lists maximizing total IoU; only IoU > 0 pairs
    match. Ties prefer a match, then skipping an ``A`` interval (resolved from the ends)."""
    rows, cols = len(A), len(B)
    w = [[iou(a, b) for b in B] for a in A]
    best = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows):
        for j in range(cols):
            cand = max(best[i][j + 1], best[i + 1][j])
            if w[i][j] > 0:
                cand = max(cand, best[i][j] + w[i][j])
            best[i + 1][j + 1] = cand
    out: list[tuple[int, int, float]] = []
    i, j = rows, cols
    while i and j:
        v = w[i - 1][j - 1]
        if v > 0 and best[i][j] == best[i - 1][j - 1] + v:
            out.append((i - 1, j - 1, v))
            i -= 1
            j -= 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out


def lev(a: str, b: str) -> int:
    """Levenshtein distance in characters."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        diag, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            up = row[j]
            row[j] = min(up + 1, row[j - 1] + 1, diag + (a[i - 1] != b[j - 1]))
            diag = up
    return row[-1]


def _intervals(rec: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(s["start_s"]), float(s["end_s"])) for s in rec.get("segments") or ()]


def seg_episode(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float | None]:
    A, B = _intervals(a), _intervals(b)
    if not A and not B:
        return dict.fromkeys(METRICS_V2["segment"], 0.0)
    P = alignment(A, B)
    slots = max(len(A), len(B))
    sa, sb = a["segments"], b["segments"]
    bounds = [(abs(A[i][0] - B[j][0]) + abs(A[i][1] - B[j][1])) / 2 for i, j, _ in P]
    labels = [(sa[i].get("status"), sa[i].get("segmentation_provenance"))
              != (sb[j].get("status"), sb[j].get("segmentation_provenance")) for i, j, _ in P]
    return {"iou_divergence": 1.0 - sum(v for *_, v in P) / slots,
            "boundary_error_s": (sum(bounds) / len(bounds)) if bounds else None,
            "count_mismatch": 1.0 if len(A) != len(B) else 0.0,
            "label_disagreement": (sum(labels) / len(labels)) if labels else None}


def cap_episode(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float | None]:
    A, B = _intervals(a), _intervals(b)
    slots = max(len(A), len(B))
    if slots == 0:
        return {"caption_mismatch": 0.0, "caption_edit_distance": 0.0, "count_mismatch": 0.0}
    P = alignment(A, B)
    sa, sb = a["segments"], b["segments"]
    same, dist = 0, 0.0
    for i, j, _ in P:
        x = (sa[i].get("caption") or "").strip()
        y = (sb[j].get("caption") or "").strip()
        same += x == y
        longest = max(len(x), len(y))
        dist += lev(x, y) / longest if longest else 0.0
    return {"caption_mismatch": 1.0 - same / slots,
            "caption_edit_distance": (dist + (slots - len(P))) / slots,
            "count_mismatch": 1.0 if len(A) != len(B) else 0.0}


def _avg_ranks(xs: Sequence[float]) -> list[float]:
    ordered = sorted(xs)
    first: dict[float, int] = {}
    last: dict[float, int] = {}
    for k, x in enumerate(ordered, 1):
        first.setdefault(x, k)
        last[x] = k
    return [(first[x] + last[x]) / 2 for x in xs]


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float | None:
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    sxx = sum((x - ma) ** 2 for x in ra)
    syy = sum((y - mb) ** 2 for y in rb)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True)) / math.sqrt(sxx * syy)


def terminal_episode(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float | None] | None:
    ta, tb = a.get("terminal"), b.get("terminal")
    if not ta or not tb:
        return None
    return {"expected_score_abs_delta": abs(float(ta["expected_score"])
                                            - float(tb["expected_score"])),
            "score_mismatch": 1.0 if ta["score"] != tb["score"] else 0.0}


def dense_episode(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float | None] | None:
    da, db = a.get("dense"), b.get("dense")
    if not da or not db:
        return None
    xa, xb = list(da.get("per_step") or ()), list(db.get("per_step") or ())
    n = min(len(xa), len(xb))
    length = 1.0 if len(xa) != len(xb) else 0.0
    if n == 0:
        d = 0.0 if not length else None
        return {"per_step_abs_delta": d, "rank_divergence": d, "length_mismatch": length}
    xa, xb = xa[:n], xb[:n]
    d = sum(abs(x - y) for x, y in zip(xa, xb, strict=True)) / n
    rho = spearman_rho(xa, xb) if n > 1 else None
    return {"per_step_abs_delta": d, "rank_divergence": d if rho is None else 1.0 - rho,
            "length_mismatch": length}


EPISODE = {"segment": seg_episode, "caption": cap_episode, "reward.terminal": terminal_episode,
           "reward.dense": dense_episode}


def metric_names(stage: str, version: str) -> tuple[str, ...]:
    return (METRICS_V3 if version == "3" else METRICS_V2)[stage]


def basis(stage: str, metric: str, version: str) -> str:
    if version == "3" and stage in COUNT_BASIS_STAGES and metric != "count_mismatch":
        return "count_matched"
    return "all"


def per_episode(stage: str, ref: Mapping[str, Mapping[str, Any]],
                other: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    out = {}
    for uid in sorted(set(ref) & set(other)):
        m = EPISODE[stage](ref[uid], other[uid])
        if m is not None:
            out[uid] = m
    return out


def means(stage: str, version: str, eps: Mapping[str, Mapping[str, float | None]]
          ) -> dict[str, float | None]:
    out = {}
    agreeing = [u for u, m in eps.items() if not m.get("count_mismatch")]
    for name in metric_names(stage, version):
        over = agreeing if basis(stage, name, version) == "count_matched" else list(eps)
        vals = [eps[u][name] for u in over if eps[u].get(name) is not None]
        out[name] = r6(sum(vals) / len(vals)) if vals else None
    return out


def threshold(metric: str, noise: float, tolerance: float, slack: float, version: str) -> float:
    t = tolerance * max(float(noise), FLOOR.get(metric, 0.0)) + float(slack)
    if version == "3":
        t = max(t, ABS_FLOOR_V3.get(metric, 0.0))
    if metric in CAP:
        t = min(t, CAP[metric])
    return round(t, 6)


def episode_block(uids: Iterable[str]) -> dict[str, Any]:
    hashes = sorted(uid_hash(u) for u in uids)
    return {"count": len(hashes), "digest": sha(hashes), "uid_sha256": hashes}


# ---------------------------------------------------------------------------------------------
# Loading


@dataclass
class Run:
    inp: Inputs
    records: dict[str, dict[str, dict[str, dict[str, Any]]]] = field(default_factory=dict)
    load_errors: list[str] = field(default_factory=list)
    state: dict[str, Any] | None = None
    certs: dict[str, tuple[Path, dict[str, Any]]] = field(default_factory=dict)
    labs: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    report: str | None = None
    keys: dict[str, dict[str, Any] | None] = field(default_factory=dict)  # current_key cache

    def side(self, side: str, stage: str) -> dict[str, dict[str, Any]]:
        return self.records.get(side, {}).get(stage, {})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _decide_key(c: Mapping[str, Any]) -> tuple[str, int]:
    order = {"certified": 0, "provisional": 1, "fail": 2}
    return (str(c.get("generated_at") or ""), order.get(str(c.get("verdict")), 3))


def load_run(inp: Inputs) -> Run:
    run = Run(inp)
    for side in SIDES:
        for stage in inp.stages:
            p = inp.records / side / inp.job / f"{stage}.jsonl"
            if not p.exists():
                continue
            try:
                rows = load_jsonl(p)
            except (OSError, ValueError) as e:
                run.load_errors.append(f"{side}/{stage}: unreadable ({type(e).__name__})")
                continue
            by: dict[str, dict[str, Any]] = {}
            for r in rows:
                if r.get("stage") != stage:
                    continue
                if r.get("uid") in by:
                    run.load_errors.append(f"{side}/{stage}: two records for one episode")
                by[str(r.get("uid"))] = r
            run.records.setdefault(side, {})[stage] = by
    try:
        run.state = json.loads(inp.state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run.state = None
    cands: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for p in sorted(inp.cert_dir.glob("*.json")) if inp.cert_dir.is_dir() else ():
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(body, dict) and body.get("preset") == inp.preset
                and body.get("stage") in inp.stages
                and inp.embodiment in (body.get("covered_pools") or ())):
            cands.setdefault(body["stage"], []).append((p, body))
    for stage, lst in cands.items():
        run.certs[stage] = max(lst, key=lambda pb: _decide_key(pb[1]))
    for stage in inp.stages:
        kind = LAB_KIND.get(stage)
        if kind is None:
            continue
        p = inp.lab_dir / f"balanced-{kind}.json"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            run.labs[kind] = d if isinstance(d, dict) else None
        except (OSError, ValueError):
            run.labs[kind] = None
    try:
        run.report = inp.report.read_text(encoding="utf-8")
    except OSError:
        run.report = None
    return run


def version_of(cert: Mapping[str, Any]) -> str:
    return str((cert.get("generator") or {}).get("version"))


def pool_gpu(preset: str, stage: str) -> str:
    from flora_deploy.config.presets import get_preset

    spec = get_preset(preset)
    return spec.pools[spec.stage_kinds[stage]].gpu


def current_key(run: Run, stage: str, cert: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``certkey.key_document`` of the candidate stack as the code stands now: the preset's
    pool GPU, the config's modes (else the certificate's) and the pinned derived digests."""
    if stage not in run.keys:
        run.keys[stage] = _current_key(run, stage, cert)
    return run.keys[stage]


def _current_key(run: Run, stage: str, cert: Mapping[str, Any] | None) -> dict[str, Any] | None:
    from flora_deploy.config.presets import get_preset
    from flora_deploy.validate.certkey import CertKeyError, key_document
    from flora_deploy.workers_impl.certstamp import pinned_content

    preset = run.inp.preset
    kind = get_preset(preset).stage_kinds[stage]
    modes = dict((cert or {}).get("modes") or {})
    cfg = balanced_job(run).get("cfg")
    if cfg and Path(cfg).exists():
        try:
            from flora_deploy.config.hashing import kind_modes
            from flora_deploy.config.schema import load_config

            modes = kind_modes(load_config(Path(cfg)), kind)
        except Exception:  # noqa: BLE001 - fall back to the certificate's modes
            pass
    vllm = ((cert or {}).get("engine") or {}).get("vllm") or {}
    try:
        return key_document(preset, kind, pool_gpu(preset, stage), modes=modes,
                            derived_content=pinned_content(preset, kind),
                            dtbias_fp32=bool(vllm.get("dtbias_fp32")))
    except (CertKeyError, KeyError):
        return None


def step_result(run: Run, step: str) -> dict[str, Any]:
    return dict((((run.state or {}).get("steps") or {}).get(step) or {}).get("result") or {})


def balanced_job(run: Run) -> dict[str, Any]:
    return dict((step_result(run, "balanced").get("jobs") or {}).get(run.inp.job) or {})


def witness_job(run: Run) -> dict[str, Any]:
    return dict((step_result(run, "balanced").get("witness_jobs") or {}).get(run.inp.job) or {})


# ---------------------------------------------------------------------------------------------
# 1. Metrics


@dataclass
class Recomputed:
    version: str
    names: tuple[str, ...]
    cand: dict[str, dict[str, float | None]]
    noise: dict[str, dict[str, float | None]] | None
    witness: dict[str, dict[str, float | None]] | None
    cand_means: dict[str, float | None]
    noise_means: dict[str, float | None] | None
    witness_means: dict[str, float | None] | None
    missing: int
    noise_unpaired: int
    witness_unpaired: int
    oracle_n: int


def recompute(run: Run, stage: str, version: str) -> Recomputed:
    ref = run.side("exact", stage)
    bal = run.side("balanced", stage)
    nz = run.records.get("noise", {}).get(stage)
    wit = run.records.get("witness", {}).get(stage)
    cand = per_episode(stage, ref, bal)
    noise = per_episode(stage, ref, nz) if nz is not None else None
    witness = per_episode(stage, ref, wit) if wit is not None else None
    return Recomputed(
        version, metric_names(stage, version), cand, noise, witness,
        means(stage, version, cand), means(stage, version, noise) if noise is not None else None,
        means(stage, version, witness) if witness is not None else None,
        len(set(ref) - set(bal)), len(set(nz or {}) - set(ref)), len(set(wit or {}) - set(ref)),
        len(ref))


def _compare_block(check: str, item: str, got: Mapping[str, Any] | None,
                   want_n: int, want_means: Mapping[str, float | None],
                   extra: Mapping[str, Any]) -> list[Row]:
    if got is None:
        return [Row(check, item, FAIL, "not in the certificate")]
    bad = []
    if got.get("n") != want_n:
        bad.append(f"n {got.get('n')} != {want_n}")
    for k, v in extra.items():
        if got.get(k) != v:
            bad.append(f"{k} {got.get(k)} != {v}")
    gm = got.get("means") or {}
    for m, v in want_means.items():
        if not close(gm.get(m), v):
            bad.append(f"{m} {gm.get(m)} != {v}")
    if set(gm) - set(want_means):
        bad.append(f"unexpected metrics {sorted(set(gm) - set(want_means))}")
    return [Row(check, item, FAIL if bad else PASS,
                "; ".join(bad) if bad else f"n={want_n}, {len(want_means)} means agree")]


def check_metrics(run: Run, stage: str, cert: Mapping[str, Any]) -> list[Row]:
    C = "1 metrics"
    v = version_of(cert)
    if v not in SUPPORTED_VERSIONS:
        return [Row(C, stage, FAIL, f"generator version {v!r}: no rule known here")]
    if not run.side("exact", stage) or not run.side("balanced", stage):
        return [Row(C, stage, FAIL, "oracle or balanced records missing")]
    rc = recompute(run, stage, v)
    m = cert.get("metrics") or {}
    rows: list[Row] = []
    if m.get("similarity_hook"):
        rows.append(Row(C, f"{stage} similarity", FAIL,
                        "a caption-similarity hook was used: its metric cannot be recomputed"))
    pools = sorted((m.get("candidate") or {}).keys())
    if pools != [run.inp.embodiment]:
        rows.append(Row(C, f"{stage} candidate pools", FAIL, f"{pools}"))
    matched = sum(1 for x in rc.cand.values() if not x.get("count_mismatch"))
    extra: dict[str, Any] = {"count_matched": matched} if v == "3" else {}
    rows += _compare_block(C, f"{stage} oracle vs balanced",
                           (m.get("candidate") or {}).get(run.inp.embodiment),
                           len(rc.cand), rc.cand_means, extra)
    if rc.noise is None:
        if m.get("noise") is not None:
            rows.append(Row(C, f"{stage} oracle vs noise", FAIL,
                            "certificate has a noise block but no noise records"))
        else:
            rows.append(Row(C, f"{stage} oracle vs noise", FAIL, "no noise run"))
    else:
        nx: dict[str, Any] = {"unpaired": rc.noise_unpaired} if v == "3" else {}
        rows += _compare_block(C, f"{stage} oracle vs noise", m.get("noise"), len(rc.noise),
                               rc.noise_means or {}, nx)
        eps = episode_block(rc.noise)
        got = (m.get("noise") or {}).get("episodes") or {}
        if got.get("count") != eps["count"] or got.get("digest") != eps["digest"]:
            rows.append(Row(C, f"{stage} noise episodes", FAIL, "count/digest differ"))
    if v == "3":
        if rc.witness is None:
            rows.append(Row(C, f"{stage} oracle vs cold witness", FAIL, "no witness records"))
        else:
            rows += _compare_block(C, f"{stage} oracle vs cold witness", m.get("witness"),
                                   len(rc.witness), rc.witness_means or {},
                                   {"unpaired": rc.witness_unpaired})
        if m.get("missing") != rc.missing:
            rows.append(Row(C, f"{stage} missing", FAIL, f"{m.get('missing')} != {rc.missing}"))
    cov = r6(len(rc.cand) / rc.oracle_n) if rc.oracle_n else None
    rows.append(Row(C, f"{stage} coverage", PASS if close(m.get("coverage"), cov) else FAIL,
                    f"{m.get('coverage')} (recomputed {cov})"))
    return rows


# ---------------------------------------------------------------------------------------------
# 2. Pass rule


def l0_status(run: Run, stage: str, cert: Mapping[str, Any]) -> dict[str, Any]:
    """L0 from the lab JSON, as the pass rule reads it (docs/certification.md, ``--l0``)."""
    kind = LAB_KIND.get(stage)
    if kind is None:
        return {"status": "not_applicable"}
    lab = run.labs.get(kind)
    if not lab:
        return {"status": "missing"}
    rows = list((lab.get("baseline") or {}).get("l0") or [])
    if not rows:
        return {"status": "missing"}
    ok = all(r.get("ids_equal") is True and r.get("grid_equal") is True
             and r.get("render_ids_equal", True) is True for r in rows)
    got = (lab.get("engine_hashes") or {}).get("baseline")
    lab_hash = got if isinstance(got, str) and re.fullmatch(r"[0-9a-f]{64}", got) else None
    status = "pass" if ok else "fail"
    want = engine_settings_hash((cert.get("engine") or {}).get("vllm"))
    if ok and version_of(cert) == "3" and want is not None and lab_hash != want:
        status = "stale"
    return {"status": status, "requests": len(rows),
            "ids_equal": sum(r.get("ids_equal") is True for r in rows),
            "render_ids_equal": sum(r.get("render_ids_equal") is True for r in rows),
            "grid_equal": sum(r.get("grid_equal") is True for r in rows),
            "engine_hash": lab_hash}


def engine_records(run: Run, stage: str) -> list[dict[str, Any]]:
    if stage not in LAB_KIND:
        return []
    return [*run.side("balanced", stage).values(), *run.side("witness", stage).values()]


def preemption_counts(run: Run, stage: str) -> dict[str, int]:
    recs = engine_records(run, stage)
    pre = [(r.get("provenance") or {}).get("engine_preemptions") for r in recs]
    return {"preempted": sum(1 for p in pre if p is not None and p > 0),
            "unknown": sum(1 for p in pre if p is None)}


def check_pass_rule(run: Run, stage: str, cert: Mapping[str, Any]) -> list[Row]:
    C = "2 pass rule"
    v = version_of(cert)
    if v not in SUPPORTED_VERSIONS:
        return [Row(C, stage, FAIL, f"generator version {v!r}: no rule known here")]
    rows: list[Row] = []
    th = cert.get("thresholds") or {}
    bad = []
    try:
        tol = float(th["tolerance"])
        min_s, min_n = int(th["min_samples"]), int(th["min_noise_samples"])
        slack = {k: float(x) for k, x in (th.get("slack") or {}).items()}
        if not TOLERANCE_BOUNDS[0] <= tol <= TOLERANCE_BOUNDS[1]:
            bad.append(f"tolerance {tol}")
        if min_s < MIN_SAMPLES:
            bad.append(f"min_samples {min_s}")
        if min_n < MIN_NOISE_SAMPLES:
            bad.append(f"min_noise_samples {min_n}")
        if v == "3":
            max_missing = int(th["max_missing"])
            if not 0 <= max_missing <= MAX_MISSING:
                bad.append(f"max_missing {max_missing}")
        else:
            min_cov = float(th["min_coverage"])
            if min_cov < MIN_COVERAGE:
                bad.append(f"min_coverage {min_cov}")
        for k, x in slack.items():
            if k not in FLOOR or not 0.0 <= x <= FLOOR[k]:
                bad.append(f"slack {k}={x}")
    except (KeyError, TypeError, ValueError):
        return [Row(C, f"{stage} knobs", FAIL, "thresholds missing or malformed")]
    rows.append(Row(C, f"{stage} knobs within policy", FAIL if bad else PASS,
                    "; ".join(bad) or f"tolerance {tol}, min {min_s}/{min_n}"))

    rc = recompute(run, stage, v)
    pm = th.get("per_metric") or {}
    fl = []
    if v == "3":
        for m in rc.names:
            if not close((th.get("noise_floor") or {}).get(m), FLOOR.get(m, 0.0), 1e-12):
                fl.append(f"noise_floor {m}")
            if not close((pm.get(m) or {}).get("absolute_floor"), ABS_FLOOR_V3.get(m, 0.0),
                         1e-12):
                fl.append(f"absolute_floor {m}")
            if (pm.get(m) or {}).get("over") != basis(stage, m, v):
                fl.append(f"over {m}")
        if dict(th.get("absolute_floor") or {}) != {m: ABS_FLOOR_V3[m] for m in rc.names
                                                    if m in ABS_FLOOR_V3}:
            fl.append("absolute_floor table")
    for m in rc.names:
        if not close((pm.get(m) or {}).get("floor"), FLOOR.get(m, 0.0), 1e-12):
            fl.append(f"floor {m}")
        if (pm.get(m) or {}).get("cap") != CAP.get(m):
            fl.append(f"cap {m}")
    rows.append(Row(C, f"{stage} floors and caps", FAIL if fl else PASS,
                    "; ".join(fl) or "equal to docs/certification.md"))

    # thresholds and checks
    want_t = {m: (threshold(m, rc.noise_means[m], tol, slack.get(m, 0.0), v)
                  if rc.noise_means is not None and rc.noise_means.get(m) is not None else None)
              for m in rc.names}
    tb = [f"{m} {(pm.get(m) or {}).get('threshold')} != {t}" for m, t in want_t.items()
          if not close((pm.get(m) or {}).get("threshold"), t)]
    rows.append(Row(C, f"{stage} thresholds", FAIL if tb else PASS,
                    "; ".join(tb) or f"{len(want_t)} recomputed"))
    blocks: dict[str, dict[str, float | None]] = {}
    if rc.cand:
        blocks[run.inp.embodiment] = rc.cand_means
    if v == "3" and rc.witness is not None:
        blocks[WITNESS_CLASS] = rc.witness_means or {}
    want_checks = {}
    for cls, mm in blocks.items():
        for m in rc.names:
            val, t = mm.get(m), want_t[m]
            want_checks[(cls, m)] = (val, t, None if val is None or t is None
                                     else bool(val <= t + 1e-12))
    got_checks = {(c.get("class"), c.get("metric")): c for c in cert.get("checks") or ()}
    cb = []
    if set(got_checks) != set(want_checks):
        cb.append(f"check set differs: {sorted(set(got_checks) ^ set(want_checks))}")
    for k, (val, t, ok) in want_checks.items():
        g = got_checks.get(k)
        if g is None:
            continue
        if not close(g.get("value"), val) or not close(g.get("threshold"), t) \
                or g.get("pass") is not ok:
            cb.append(f"{k[0]} {k[1]}: {g.get('value')}/{g.get('threshold')}/{g.get('pass')} "
                      f"!= {val}/{t}/{ok}")
    failing = [f"{c} {m}" for (c, m), (_, _, ok) in want_checks.items() if ok is False]
    rows.append(Row(C, f"{stage} checks", FAIL if cb else PASS,
                    "; ".join(cb) or (f"{len(want_checks)} agree"
                                      + (f"; failing: {', '.join(failing)}" if failing else ""))))

    # the count-flip tolerance (user decision): one episode in ten may differ in count
    if stage in COUNT_BASIS_STAGES and "count_mismatch" not in rc.names:
        rows.append(Row(C, f"{stage} count-flip tolerance", FAIL,
                        f"generator version {v} does not check count_mismatch for {stage}"))
    elif "count_mismatch" in rc.names:
        t = want_t["count_mismatch"]
        t_rec = (pm.get("count_mismatch") or {}).get("threshold")
        flips = sum(1 for x in rc.cand.values() if x.get("count_mismatch"))
        n = len(rc.cand)
        ok = (v == CURRENT_RULE and t is not None and t >= COUNT_FLIP_SHARE - 1e-9
              and isinstance(t_rec, int | float) and t_rec >= COUNT_FLIP_SHARE - 1e-9)
        allowed = math.floor(t * n + 1e-9) if t is not None else None
        rows.append(Row(C, f"{stage} count-flip tolerance", PASS if ok else FAIL,
                        f"{flips} of {n} episodes differ in count; threshold {t} allows "
                        f"{allowed}" + ("" if ok else f" (rule needs >= {COUNT_FLIP_SHARE}: "
                                                      "1 flip in 10 must pass; generator "
                                                      f"version {v})")))

    # reasons and verdict
    l0 = l0_status(run, stage, cert)
    failures, prov = set(), set()
    if any(ok is False for *_, ok in want_checks.values()):
        failures.add("divergence_above_noise_floor")
    if l0["status"] == "fail":
        failures.add("l0_failed")
    nz_n = len(rc.noise) if rc.noise is not None else None
    if nz_n is None:
        prov.add("no_noise_run")
    elif nz_n < min_n:
        prov.add("too_few_noise_samples")
    if not rc.cand or len(rc.cand) < min_s:
        prov.add("too_few_samples")
    if v == "3":
        if rc.missing > int(th["max_missing"]):
            prov.add("incomplete_coverage")
    elif rc.oracle_n and len(rc.cand) / rc.oracle_n < float(th["min_coverage"]):
        prov.add("incomplete_coverage")
    if l0["status"] == "missing":
        prov.add("no_l0")
    if l0["status"] == "stale":
        prov.add("stale_l0")
    if nz_n is not None and any(ok is None for *_, ok in want_checks.values()):
        prov.add("metric_not_evaluable")
    if v == "3":
        pre = preemption_counts(run, stage)
        if pre["preempted"]:
            prov.add("engine_preempted")
        if pre["unknown"]:
            prov.add("preemptions_unknown")
        if rc.witness is not None:
            if len(rc.witness) < min_n:
                prov.add("too_few_witness_samples")
            if any((r.get("provenance") or {}).get("compile_cache") != FORCED_COLD
                   for r in run.side("witness", stage).values() if
                   r.get("uid") in run.side("exact", stage)):
                prov.add("witness_not_forced_cold")
    verdict = "fail" if failures else ("provisional" if prov else "certified")
    reasons = failures | prov
    got_reasons = set(cert.get("reasons") or ())
    rb = []
    if got_reasons != reasons:
        rb.append(f"reasons {sorted(got_reasons)} != {sorted(reasons)}")
    if cert.get("verdict") != verdict:
        rb.append(f"verdict {cert.get('verdict')} != {verdict}")
    if cert.get("unlocks") is not (verdict == "certified"):
        rb.append(f"unlocks {cert.get('unlocks')}")
    rows.append(Row(C, f"{stage} verdict", FAIL if rb else PASS,
                    "; ".join(rb) or f"{verdict}" + (f" ({', '.join(sorted(reasons))})"
                                                     if reasons else "")))
    ok_pub = verdict == "certified" or (verdict == "provisional" and run.inp.allow_provisional)
    rows.append(Row(C, f"{stage} publishable", PASS if ok_pub and not rb else FAIL,
                    f"{verdict}" + ("" if ok_pub else " (publishing needs certified; "
                                    "--allow-provisional accepts provisional)")))
    return rows


# ---------------------------------------------------------------------------------------------
# 3. Coverage


def _prov(recs: Iterable[Mapping[str, Any]], key: str) -> set[Any]:
    out = set()
    for r in recs:
        v = (r.get("provenance") or {}).get(key)
        out.add(canonical(v) if isinstance(v, dict | list) else v)
    return out


def _key_hashes(recs: Iterable[Mapping[str, Any]]) -> set[str | None]:
    return {sha(p["cert_key"]) if isinstance(p.get("cert_key"), dict) else None
            for p in ((r.get("provenance") or {}) for r in recs)}


def check_coverage(run: Run, stage: str, cert: Mapping[str, Any]) -> list[Row]:
    C = "3 coverage"
    from flora_deploy.config.presets import get_preset
    from flora_deploy.validate.certkey import GENERATOR_VERSIONS

    rows: list[Row] = []
    ref, bal = run.side("exact", stage), run.side("balanced", stage)
    nz, wit = run.side("noise", stage), run.side("witness", stage)
    v = version_of(cert)
    rc = recompute(run, stage, v if v in SUPPORTED_VERSIONS else CURRENT_RULE)
    want = episode_block(rc.cand)
    got = cert.get("episodes") or {}
    ok = (got.get("uid_sha256") == want["uid_sha256"] and got.get("count") == want["count"]
          and got.get("digest") == want["digest"])
    rows.append(Row(C, f"{stage} episode ids", PASS if ok else FAIL,
                    f"{want['count']} matched episodes" if ok else
                    f"certificate {got.get('count')} hashed ids, matched {want['count']} "
                    f"(missing {len(set(want['uid_sha256']) - set(got.get('uid_sha256') or []))},"
                    f" extra {len(set(got.get('uid_sha256') or []) - set(want['uid_sha256']))})"))
    for side, recs, block in (("noise", nz, "noise"), ("witness", wit, "witness")):
        extra = sorted(set(recs) - set(ref))
        unpaired = ((cert.get("metrics") or {}).get(block) or {}).get("unpaired")
        ok = not extra or (v == "3" and unpaired == len(extra))
        rows.append(Row(C, f"{stage} {side} episodes within the oracle's",
                        PASS if ok and recs else FAIL,
                        f"{len(recs)} records, {len(extra)} not the oracle's"
                        + (f" (certificate unpaired {unpaired})" if extra else "")
                        if recs else "no records"))
    # one stage hash, one key, equal to the current code's
    hashes = _prov(bal.values(), "config_hash") | _prov(wit.values(), "config_hash")
    st_hash = (balanced_job(run).get("stage_hashes") or {}).get(stage)
    ok = len(hashes) == 1 and cert.get("stage_hash") in hashes and (st_hash in (None, *hashes))
    rows.append(Row(C, f"{stage} one balanced stage hash", PASS if ok else FAIL,
                    f"{len(hashes)} distinct" + ("" if ok else
                                                 f"; certificate {str(cert.get('stage_hash'))[:12]}"
                                                 f"; state {str(st_hash)[:12]}")))
    keys = _key_hashes(bal.values()) | _key_hashes(wit.values())
    cur = current_key(run, stage, cert)
    cur_hash = sha(cur) if cur is not None else None
    ok = len(keys) == 1 and cert.get("engine_spec_hash") in keys and cur_hash in keys
    detail = f"key {str(cur_hash)[:12]}"
    if not ok:
        doc = next((r["provenance"].get("cert_key") for r in bal.values()), None) or {}
        diff = sorted(k for k in set(doc) | set(cur or {}) if doc.get(k) != (cur or {}).get(k))
        detail = (f"{len(keys)} record key(s); certificate {str(cert.get('engine_spec_hash'))[:12]}"
                  f"; current {str(cur_hash)[:12]}" + (f"; differs in {diff}" if diff else ""))
    rows.append(Row(C, f"{stage} cert_key is the current code's", PASS if ok else FAIL, detail))
    gen_ok = v in GENERATOR_VERSIONS and v in SUPPORTED_VERSIONS
    rows.append(Row(C, f"{stage} generator version", PASS if gen_ok else FAIL,
                    f"{v} (current code accepts {list(GENERATOR_VERSIONS)})"))
    # oracle binding
    o = cert.get("oracle") or {}
    ob = []
    o_hash, o_keys = _prov(ref.values(), "config_hash"), _key_hashes(ref.values())
    o_gpus = {norm_gpu(g) for g in _prov(ref.values(), "gpu")}
    if len(o_hash) != 1 or o.get("stage_hash") not in o_hash:
        ob.append("stage hash")
    if len(o_keys) != 1 or o.get("engine_spec_hash") not in o_keys:
        ob.append("key")
    exact_gpu = norm_gpu(pool_gpu("exact", stage))
    if o_gpus != {exact_gpu} or o.get("gpu") != exact_gpu:
        ob.append(f"gpu {sorted(map(str, o_gpus))} / {o.get('gpu')} (want {exact_gpu})")
    dets = {((json.loads(k) or {}).get("modes") or {}).get("determinism")
            if isinstance(k, str) else None for k in _prov(ref.values(), "cert_key")}
    if dets != {o.get("determinism")}:
        ob.append("determinism")
    if nz and (_prov(nz.values(), "config_hash") != o_hash or _key_hashes(nz.values()) != o_keys
               or {norm_gpu(g) for g in _prov(nz.values(), "gpu")} != o_gpus):
        ob.append("noise run does not repeat the oracle (stage hash, key or GPU)")
    rows.append(Row(C, f"{stage} oracle binding", FAIL if ob else PASS,
                    "; ".join(ob) or f"{exact_gpu}, determinism {o.get('determinism')}"))
    # embodiment and GPU
    embs = set()
    for side in SIDES:
        embs |= _prov(run.side(side, stage).values(), "embodiment")
    ok = embs == {run.inp.embodiment} and cert.get("covered_pools") == [run.inp.embodiment]
    rows.append(Row(C, f"{stage} embodiment", PASS if ok else FAIL,
                    f"records {sorted(map(str, embs))}, certificate {cert.get('covered_pools')}"))
    want_gpu = pool_gpu(run.inp.preset, stage)
    rec_gpus = {norm_gpu(g) for g in _prov([*bal.values(), *wit.values()], "gpu")}
    kind = get_preset(run.inp.preset).stage_kinds[stage]
    ok = (cert.get("gpu") == want_gpu and rec_gpus == {norm_gpu(want_gpu)}
          and cert.get("preset") == run.inp.preset and cert.get("kind") == kind)
    rows.append(Row(C, f"{stage} GPU", PASS if ok else FAIL,
                    f"certificate {cert.get('gpu')}, records {sorted(map(str, rec_gpus))}, "
                    f"pool {want_gpu}"))
    return rows


# ---------------------------------------------------------------------------------------------
# 4. Evidence


def _counts(recs: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        k = (r.get("provenance") or {}).get("compile_cache") or "unknown"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def check_evidence(run: Run, stage: str, cert: Mapping[str, Any]) -> list[Row]:
    C = "4 evidence"
    rows: list[Row] = []
    v = version_of(cert)
    bal, wit = run.side("balanced", stage), run.side("witness", stage)
    if stage in LAB_KIND:
        # warm vs cold
        missing = [r for r in [*bal.values(), *wit.values()]
                   if not CACHE_ID.match(str((r.get("provenance") or {}).get("compile_cache")))
                   or (r.get("provenance") or {}).get("compile_cache") == "unknown"]
        rows.append(Row(C, f"{stage} cache provenance per record", FAIL if missing else PASS,
                        f"{len(missing)} of {len(bal) + len(wit)} records without a compile "
                        "cache identity" if missing else f"{len(bal) + len(wit)} records"))
        forced = [r for r in wit.values()
                  if (r.get("provenance") or {}).get("compile_cache") == FORCED_COLD]
        wj = witness_job(run)
        kinds = {c.get("kind") for c in wj.get("containers") or ()}
        ok = bool(forced) and len(forced) == len(wit) and wj.get("state") == "done" \
            and stage in kinds
        rows.append(Row(C, f"{stage} forced-cold container", PASS if ok else FAIL,
                        f"{len(forced)} of {len(wit)} witness records cold:forced; witness job "
                        f"{wj.get('state')!r}, containers for {sorted(map(str, kinds))}"))
        if v == "3":
            cc = cert.get("compile_cache") or {}
            want = {"candidate": _counts(bal.values()), "witness": _counts(wit.values())}
            covers = sorted({k for d in want.values() for k in d})
            ok = (cc.get("candidate") == want["candidate"] and cc.get("witness") == want["witness"]
                  and cc.get("covers") == covers and FORCED_COLD in covers)
            rows.append(Row(C, f"{stage} certificate states the caches", PASS if ok else FAIL,
                            f"covers {cc.get('covers')}" if ok else
                            f"certificate {cc} != records {want}"))
        else:
            rows.append(Row(C, f"{stage} certificate states the caches", FAIL,
                            f"generator version {v} records no compile_cache block"))
        # preemptions
        pre = preemption_counts(run, stage)
        clean = not pre["preempted"] and not pre["unknown"]
        ok = clean or cert.get("verdict") == "provisional"
        if v == "3" and cert.get("preemptions") != pre:
            ok = False
        rows.append(Row(C, f"{stage} preemptions", PASS if ok else FAIL,
                        f"{pre['preempted']} preempted, {pre['unknown']} unknown"
                        + ("" if clean else f"; certificate {cert.get('verdict')}")
                        + ("" if v != "3" or cert.get("preemptions") == pre
                           else f"; certificate states {cert.get('preemptions')}")))
        # lab: L0 and Gate A under the current engine settings
        rows += check_lab(run, stage, cert)
    return rows


def check_lab(run: Run, stage: str, cert: Mapping[str, Any]) -> list[Row]:
    C = "4 evidence"
    from flora_deploy.engines.offline_llm import EngineSpec

    kind = LAB_KIND[stage]
    lab = run.labs.get(kind)
    if not lab:
        return [Row(C, f"{stage} lab {kind}", FAIL, "no lab JSON")]
    rows = []
    base = lab.get("baseline") or {}
    l0 = l0_status(run, stage, cert)
    gate = lab.get("gate_a") or {}
    ok = l0["status"] in ("pass", "stale") and int(gate.get("requests") or 0) > 0
    rows.append(Row(C, f"{stage} L0 and Gate A present", PASS if ok else FAIL,
                    f"L0 {l0['status']} on {l0.get('requests', 0)} requests, Gate A on "
                    f"{gate.get('requests')} requests"))
    gpu = pool_gpu(run.inp.preset, stage)
    current = engine_settings_hash(EngineSpec.for_gpu(kind, "", gpu).hashed())
    recorded = (lab.get("engine_hashes") or {}).get("baseline")
    cert_eng = engine_settings_hash((cert.get("engine") or {}).get("vllm"))
    lab_gpu = norm_gpu(base.get("gpu") or lab.get("gpu"))
    ok = recorded == current == cert_eng and lab_gpu == norm_gpu(gpu)
    rows.append(Row(C, f"{stage} lab engine-settings hash", PASS if ok else FAIL,
                    f"lab {str(recorded)[:12]}, current {str(current)[:12]}, certificate "
                    f"{str(cert_eng)[:12]}, lab GPU {lab_gpu}"))
    got = cert.get("l0") or {}
    keys = ("status", "requests", "ids_equal", "render_ids_equal", "grid_equal") + (
        ("engine_hash",) if version_of(cert) == "3" else ())
    bad = [k for k in keys if got.get(k) != l0.get(k)]
    ga = cert.get("gate_a") or {}
    for k in ("requests", "greedy_identical", "positions", "max_abs_dlogit", "mean_abs_dlogit"):
        if k in gate and not (ga.get(k) == gate.get(k) or close(ga.get(k), gate.get(k))):
            bad.append(f"gate_a.{k}")
    rows.append(Row(C, f"{stage} certificate L0 / Gate A", FAIL if bad else PASS,
                    f"differs in {bad}" if bad else f"L0 {got.get('status')}"))
    return rows


# ---------------------------------------------------------------------------------------------
# 5. Freeze lock


def run_drift_test(root: Path) -> tuple[int, str]:
    """``pytest tests/unit/test_certificate_drift.py`` on the committed certificates."""
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        DRIFT_TEST], cwd=root, capture_output=True, text=True, timeout=900)
    lines = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode, lines[-1] if lines else ""


def check_freeze(run: Run, stage: str, path: Path, cert: Mapping[str, Any]) -> list[Row]:
    C = "5 freeze lock"
    from flora_deploy.validate import certfreeze as F

    rows: list[Row] = []
    lock_path = run.inp.cert_dir / "freeze" / f"{cert.get('id')}.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [Row(C, f"{stage} lock", FAIL, f"no readable freeze/{lock_path.name}")]
    doc = lock.get("key_document") if isinstance(lock, dict) else None
    bad = []
    if lock.get("schema") != F.FREEZE_SCHEMA:
        bad.append("schema")
    if lock.get("certificate") != cert.get("id") or path.stem != cert.get("id"):
        bad.append("certificate id")
    if not isinstance(doc, dict) or sha(doc) != cert.get("engine_spec_hash") \
            or lock.get("engine_spec_hash") != cert.get("engine_spec_hash"):
        bad.append("key document does not hash to the certificate's key")
    rows.append(Row(C, f"{stage} lock matches the certificate", FAIL if bad else PASS,
                    "; ".join(bad) or lock_path.name))
    cur = current_key(run, stage, cert)
    diff = sorted(k for k in set(doc or {}) | set(cur or {})
                  if (doc or {}).get(k) != (cur or {}).get(k)) if cur is not None else ["?"]
    mods = F.module_hashes(cert.get("kind") or stage)
    mdiff = sorted(k for k in set(mods) | set(lock.get("modules") or {})
                   if mods.get(k) != (lock.get("modules") or {}).get(k))
    ok = not diff and not mdiff
    rows.append(Row(C, f"{stage} lock is the current code", PASS if ok else FAIL,
                    f"{len(mods)} modules" if ok else
                    f"key differs in {diff}; modules differ: {mdiff}"))
    return rows


def check_drift_guard(run: Run) -> list[Row]:
    C = "5 freeze lock"
    from flora_deploy.validate import certfreeze as F
    from flora_deploy.validate.certkey import CERT_DIR

    rep = F.check(run.inp.cert_dir, git_history=False)
    ids = {c.get("id") for _, c in run.certs.values()}
    mine = [f for f in rep.findings if f.cert in ids or f.cert in {f"{i}.json" for i in ids}]
    bad = [f"{f.severity} {f.status}: {f.message}" for f in mine
           if f.severity in ("error", "warning")]
    if not rep.ok:
        bad += [f"{f.cert} {f.status}" for f in rep.by("error") if f not in mine]
    rows = [Row(C, "drift guard (certfreeze.check)", FAIL if bad else PASS,
                "; ".join(bad)[:400] or f"{rep.certificates} certificate(s), no drift")]
    if run.inp.cert_dir.resolve() == CERT_DIR.resolve() or run.inp.drift_runner is not None:
        code, tail = (run.inp.drift_runner or run_drift_test)(ROOT)
        rows.append(Row(C, DRIFT_TEST, PASS if code == 0 else FAIL, tail[:200]))
    return rows


# ---------------------------------------------------------------------------------------------
# 6. Leakage


def nv_reference(nv_root: Path | None):
    """``license_shingles.Reference`` over the fetched NVIDIA code (None: nothing fetched)."""
    import license_shingles as L

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, snapshot_dirname

    root = Path(nv_root or NV_CODE_CACHE_DEFAULT).expanduser()
    ref = L.Reference(L.shingles(L.tokens((ROOT / "LICENSE").read_text(encoding="utf-8"))))
    files = 0
    for key in NV_PINS:
        d = root / snapshot_dirname(PINS_BY_KEY[key])
        for p in sorted(d.rglob("*")) if d.is_dir() else ():
            if not p.is_file() or ".cache" in p.relative_to(d).parts or p.name in NV_SKIP \
                    or (p.name.startswith("requirements") and p.suffix == ".txt"):
                continue
            try:
                ref.add(p.name, p.read_text(encoding="utf-8"))
                files += 1
            except (UnicodeDecodeError, OSError):
                continue
    return ref if files and len(ref) else None


def record_strings(run: Run) -> set[str]:
    """Strings from the records that must never appear in what is published: captions, ids,
    dataset names, source URIs, audit refs and NVIDIA's metadata and rationales. Short NVIDIA
    metadata values ("fine", "none") are English words, so only those of 10+ characters count;
    ids and captions count from 4 and 6 characters."""
    ids: set[str] = set()
    meta: set[str] = set()

    def walk(o: Any) -> None:
        if isinstance(o, str):
            meta.add(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for side in run.records.values():
        for recs in side.values():
            for r in recs.values():
                for k in ("uid", "episode_id", "dataset", "audit_ref"):
                    if isinstance(r.get(k), str) and len(r[k]) >= 4:
                        ids.add(r[k])
                for src in ((r.get("provenance") or {}).get("source") or {}).values():
                    if isinstance(src, dict) and isinstance(src.get("uri"), str):
                        ids.add(src["uri"])
                for seg in r.get("segments") or ():
                    cap = (seg.get("caption") or "").strip()
                    if len(cap) >= 6:
                        ids.add(cap)
                    walk(seg.get("fine_event_id"))
                    walk(seg.get("nv_metadata"))
                walk((r.get("terminal") or {}).get("rationale"))
                walk((r.get("dense") or {}).get("segments"))
    return ids | {m for m in meta if len(m) >= 10}


def our_strings(run: Run, stage: str, cert: Mapping[str, Any]) -> set[str]:
    """String values a certificate may carry: our enums, and what the current key holds."""
    allowed = {"certified", "provisional", "fail", "pass", "stale", "missing",
               "not_applicable", "all", "count_matched", "nvidia", "deterministic",
               "human", "yam", "yam32", "dex", "tianji", "exact", "balanced", "budget",
               "interactive", WITNESS_CLASS, "flora-certify", "flora-certificate/v2",
               *SUPPORTED_VERSIONS}
    allowed |= {m for ms in METRICS_V3.values() for m in ms} | {"caption_similarity_divergence"}
    allowed |= set(METRICS_V3) | set(LAB_KIND.values())
    allowed |= {"divergence_above_noise_floor", "l0_failed", "no_noise_run",
                "too_few_noise_samples", "too_few_samples", "incomplete_coverage", "no_l0",
                "stale_l0", "metric_not_evaluable", "engine_preempted", "preemptions_unknown",
                "too_few_witness_samples", "witness_not_forced_cold"}

    def walk(o: Any) -> None:
        if isinstance(o, str):
            allowed.add(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                allowed.add(str(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(current_key(run, stage, cert))
    from flora_deploy.config.presets import get_preset

    for name in ("exact", run.inp.preset):
        walk({k: p.gpu for k, p in get_preset(name).pools.items()})
    return allowed


_SAFE = re.compile(r"^(cert_[0-9a-f]{32}|[0-9a-f]{12,64}|\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
                   r"|flora-deploy [0-9][0-9A-Za-z.+\-]*|vllm-[0-9.]+-offline|[0-9.]+)\Z")


def free_text(cert: Any, allowed: set[str]) -> list[str]:
    """Paths of string fields (values, and keys outside our identifier shape) that are not ours."""
    out: list[str] = []

    def ok(s: str) -> bool:
        return s in allowed or bool(_SAFE.match(s)) or bool(CACHE_ID.match(s))

    def walk(o: Any, path: str) -> None:
        if isinstance(o, str):
            if not ok(o):
                out.append(path)
        elif isinstance(o, dict):
            for k, v in o.items():
                if not (ok(str(k)) or re.fullmatch(r"[a-z][a-z0-9_]*", str(k))):
                    out.append(f"{path}.<key>")
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(cert, "$")
    return out


def _leaks(text: str, forbidden: set[str], allowed: set[str]) -> int:
    n = 0
    low = text.lower()
    for s in forbidden:
        if s in allowed or _SAFE.match(s):
            continue
        if len(s) >= 8:
            n += s.lower() in low
        else:
            n += bool(re.search(r"(?<![a-z0-9_])" + re.escape(s.lower()) + r"(?![a-z0-9_])",
                                low))
    return n


def check_leakage(run: Run) -> list[Row]:
    C = "6 leakage"
    rows: list[Row] = []
    texts: dict[str, str] = {}
    for stage, (path, _cert) in sorted(run.certs.items()):
        texts[f"{stage} certificate"] = path.read_text(encoding="utf-8")
        lock = run.inp.cert_dir / "freeze" / path.name
        if lock.exists():
            texts[f"{stage} freeze lock"] = lock.read_text(encoding="utf-8")
    if run.report is not None:
        texts["report"] = run.report
    else:
        rows.append(Row(C, "report", FAIL, f"no report at {run.inp.report}"))
    ref = nv_reference(run.inp.nv_root)
    if ref is None:
        rows.append(Row(C, "NVIDIA text (shingles)", FAIL,
                        "NVIDIA code not fetched (scripts/fetch_nv_code.py): cannot check"))
    else:
        hits = [f"{name}: {hit.ratio:.2f}" for name, t in texts.items()
                if (hit := ref.check(name + ".json", t)) is not None]
        rows.append(Row(C, "NVIDIA text (shingles)", FAIL if hits else PASS,
                        "; ".join(hits) or f"{len(texts)} files, {len(ref)} reference docs"))
    forbidden = record_strings(run)
    for stage, (_, cert) in sorted(run.certs.items()):
        allowed = our_strings(run, stage, cert)
        free = free_text(cert, allowed)
        rows.append(Row(C, f"{stage} certificate string fields", FAIL if free else PASS,
                        f"not ours: {free[:8]}" if free else "every string is ours"))
    base_allowed = set().union(*(our_strings(run, s, c) for s, (_, c) in run.certs.items())) \
        if run.certs else set()
    for name, text in texts.items():
        n = _leaks(text, forbidden, base_allowed)
        rows.append(Row(C, f"{name}: record strings", FAIL if n else PASS,
                        f"{n} record string(s) found (captions, ids, metadata)" if n
                        else f"none of {len(forbidden)}"))
    return rows


# ---------------------------------------------------------------------------------------------
# 7. Spend


_TOTAL = re.compile(r"^Total: \*\*(\$[0-9,.]+)\*\*", re.M)
_LINE = {k: re.compile(rf"^- {k} (\$[0-9,.]+|-)", re.M) for k in ("tracked", "charged", "billed")}


def report_spend_rows(text: str) -> list[tuple[str, str, str, str]] | None:
    """``(attempts, tracked, charged, billed)`` of the report's "Spend per step" table."""
    part = text.split("## Spend per step", 1)
    if len(part) < 2:
        return None
    out = []
    lines = part[1].split("\n## ", 1)[0].splitlines()
    head = next((i for i, x in enumerate(lines) if x.startswith("| step |")), None)
    if head is None:
        return None
    for x in lines[head + 2:]:
        if not x.startswith("|"):
            break
        cells = [c.strip() for c in x.strip().strip("|").split("|")]
        if len(cells) >= 7:
            out.append((cells[2], cells[4], cells[5], cells[6]))
    return out


def check_spend(run: Run) -> list[Row]:
    C = "7 spend"
    if run.state is None:
        return [Row(C, "state", FAIL, f"no readable state at {run.inp.state}")]
    if run.report is None:
        return [Row(C, "report", FAIL, f"no report at {run.inp.report}")]
    import _runbook as rb

    st = rb.State.from_json(run.state)
    wm = st.watermark
    raw_tracked = round(sum(float(a.get("tracked_usd") or 0.0)
                            for s in (run.state.get("steps") or {}).values()
                            for a in s.get("attempts") or ()), 4)
    charged = sum(s.charged(wm) for s in st.steps.values())
    per_step = sum(max(s.charged(wm), s.billed_usd or 0.0) for s in st.steps.values())
    total = round(max(per_step, st.billed_total or 0.0), 4)
    rows = []
    inv = []
    if abs(total - st.spent) > 1e-6:
        inv.append(f"total {total} != runbook {st.spent}")
    if charged + 1e-9 < raw_tracked:
        inv.append("charged below tracked")
    if total + 1e-9 < (st.billed_total or 0.0) or total + 1e-9 < charged:
        inv.append("total below charged or billed")
    if st.gate_cap is not None and total > st.gate_cap + 1e-9:
        inv.append(f"total {usd(total)} over the gated cap {usd(st.gate_cap)}")
    rows.append(Row(C, "state is consistent", FAIL if inv else PASS,
                    "; ".join(inv) or f"total {usd(total)} = max(charged/billed per step, "
                    "billed)"))
    text = run.report
    m = _TOTAL.search(text)
    want = {"tracked": usd(raw_tracked), "charged": usd(charged), "billed": usd(st.billed_total)}
    bad = []
    if m is None or m.group(1) != usd(total):
        bad.append(f"Total {m.group(1) if m else None} != {usd(total)}")
    for k, rx in _LINE.items():
        g = rx.search(text)
        if g is None or g.group(1) != want[k]:
            bad.append(f"{k} {g.group(1) if g else None} != {want[k]}")
    rows.append(Row(C, "report totals", FAIL if bad else PASS,
                    "; ".join(bad) or f"tracked {want['tracked']}, charged {want['charged']}, "
                    f"billed {want['billed']}, total {usd(total)}"))
    got = report_spend_rows(text)
    want_rows = sorted((str(len(s.attempts)), usd(s.tracked_usd), usd(s.charged(wm)),
                        usd(s.billed_usd)) for s in st.steps.values())
    ok = got is not None and sorted(got) == want_rows
    rows.append(Row(C, "report per-step rows", PASS if ok else FAIL,
                    f"{len(want_rows)} steps" if ok else
                    f"report {sorted(got) if got is not None else None} != state {want_rows}"))
    return rows


# ---------------------------------------------------------------------------------------------
# Driver


def check_all(inp: Inputs) -> list[Row]:
    run = load_run(inp)
    rows: list[Row] = []
    for e in run.load_errors:
        rows.append(Row("0 inputs", "records", FAIL, e))
    for stage in inp.stages:
        for side in ("exact", "balanced"):
            if not run.side(side, stage):
                rows.append(Row("0 inputs", f"{side} {stage} records", FAIL,
                                f"none under {inp.records / side / inp.job}"))
        if stage not in run.certs:
            rows.append(Row("0 inputs", f"{stage} certificate", FAIL,
                            f"no {inp.preset} {stage} certificate for {inp.embodiment} in "
                            f"{inp.cert_dir}"))
    if run.state is None:
        rows.append(Row("0 inputs", "state", FAIL, f"no readable state at {inp.state}"))
    for stage in inp.stages:
        if stage not in run.certs:
            continue
        path, cert = run.certs[stage]
        for label, fn in (("1 metrics", check_metrics), ("2 pass rule", check_pass_rule),
                          ("3 coverage", check_coverage), ("4 evidence", check_evidence)):
            try:
                rows += fn(run, stage, cert)
            except Exception as e:  # noqa: BLE001 - a crash is a FAIL row, not a pass
                rows.append(Row(label, stage, FAIL, f"error: {type(e).__name__}: {e}"))
        try:
            rows += check_freeze(run, stage, path, cert)
        except Exception as e:  # noqa: BLE001
            rows.append(Row("5 freeze lock", stage, FAIL, f"error: {type(e).__name__}: {e}"))
    for label, fn in (("5 freeze lock", check_drift_guard), ("6 leakage", check_leakage),
                      ("7 spend", check_spend)):
        if fn is check_drift_guard and not run.certs:
            continue
        try:
            rows += fn(run)
        except Exception as e:  # noqa: BLE001
            rows.append(Row(label, "-", FAIL, f"error: {type(e).__name__}: {e}"))
    return sorted(rows, key=lambda r: r.check)


def render(rows: Sequence[Row]) -> str:
    ok = all(r.status == PASS for r in rows)
    out = ["| check | item | result | detail |", "|---|---|---|---|"]
    out += [f"| {r.check} | {r.item} | {r.status} | {r.detail.replace('|', '/')} |"
            for r in rows]
    nfail = sum(r.status == FAIL for r in rows)
    out += ["", f"Overall: {'PASS' if ok else 'FAIL'} ({len(rows) - nfail} pass, {nfail} fail)"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT,
                    help="the certified run's directory (records/, state.json)")
    ap.add_argument("--records", type=Path, help="records directory (default <run-root>/records)")
    ap.add_argument("--state", type=Path, help="run state (default <run-root>/state.json)")
    ap.add_argument("--cert-dir", type=Path, help="certificates (default the committed dir)")
    ap.add_argument("--lab-dir", type=Path, default=DEFAULT_LAB_DIR,
                    help="where balanced-<kind>.json are")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--job", default="human")
    ap.add_argument("--embodiment", default="human")
    ap.add_argument("--stage", action="append", help="stage(s) (default segment, caption)")
    ap.add_argument("--nv-root", type=Path, help="fetched NVIDIA code (default the cache)")
    ap.add_argument("--allow-provisional", action="store_true",
                    help="a provisional verdict may pass (it unlocks nothing)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    inp = Inputs(run_root=a.run_root, records=a.records, state=a.state, cert_dir=a.cert_dir,
                 lab_dir=a.lab_dir, report=a.report, job=a.job, embodiment=a.embodiment,
                 stages=tuple(a.stage or ("segment", "caption")), nv_root=a.nv_root,
                 allow_provisional=a.allow_provisional)
    rows = check_all(inp)
    ok = all(r.status == PASS for r in rows)
    if a.json:
        print(json.dumps({"ok": ok, "rows": [asdict(r) for r in rows]}, indent=1))
    else:
        print(render(rows))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
