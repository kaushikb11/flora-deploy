"""Gate certificate generator (PLAN.md §9, D9, D16): ``balanced`` vs ``exact``, within noise.

Inputs, all ``flora_labels_v1`` records (JSONL):

- ``exact``: the oracle run (NVIDIA's own HF code; model inputs proven identical);
- ``noise`` (optional): a second ``exact`` run of a subset in NVIDIA's default mode. ``exact``
  vs ``noise`` is NVIDIA's own run-to-run divergence (``docs/determinism.md``: the H200 default
  path is not deterministic), the **noise floor**;
- ``balanced``: the candidate stack on the same episodes;
- optional L0 / Gate A JSON from ``modal_app/balanced_lab.py``.

Per stage, every metric below is a *divergence* (0 = identical; definitions in
``docs/certification.md``), computed per episode and averaged:

- ``segment``: ``iou_divergence``, ``boundary_error_s``, ``count_mismatch``,
  ``label_disagreement`` over a monotone IoU alignment of the subtask intervals;
- ``caption``: ``caption_mismatch`` (1 - exact-match rate), ``caption_edit_distance``
  (normalized Levenshtein), and ``caption_similarity_divergence`` when a text-similarity hook is
  given (default off; nothing here touches the network);
- ``reward.terminal``: ``expected_score_abs_delta``, ``score_mismatch``;
- ``reward.dense``: ``per_step_abs_delta``, ``rank_divergence`` (1 - Spearman rho),
  ``length_mismatch``.

Pass rule, per (stage, embodiment class) and metric: ``mean(balanced vs exact) <= tolerance x
mean(noise vs exact) + slack`` (defaults 1.5 and 0). Verdict: ``fail`` if any check fails or a
present L0 fails; else ``provisional`` if there is no noise run, too few samples (per class, or
in the noise run), incomplete coverage, or (video stages) no L0; else ``certified``.

The certificate holds metrics only: no record string ever goes in (no captions, prompts,
rationales, ids, dataset names). Episode ids are sha256-hashed. The one record value carried
is the certified stage hash (``provenance.config_hash``), a hex digest of our own config.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flora_deploy.records import Segment, StageRecord
from flora_deploy.validate.certkey import (
    CERT_DIR,
    CERT_SCHEMA,
    canonical,
    certificate_id,
    sha256_hex,
)

GENERATOR = "flora-certify"
CERTIFY_VERSION = "1"

Similarity = Callable[[str, str], float]

STAGE_METRICS: dict[str, tuple[str, ...]] = {
    "segment": ("iou_divergence", "boundary_error_s", "count_mismatch", "label_disagreement"),
    "caption": ("caption_mismatch", "caption_edit_distance"),
    "reward.terminal": ("expected_score_abs_delta", "score_mismatch"),
    "reward.dense": ("per_step_abs_delta", "rank_divergence", "length_mismatch"),
}
SIMILARITY_METRIC = "caption_similarity_divergence"
# Stages whose balanced engine differs from exact's, so L0 input parity must be shown.
L0_STAGES = {"segment": "tempseg", "caption": "captioner"}
DEFAULT_CLASS = "all"


class CertifyError(ValueError):
    pass


@dataclass(frozen=True)
class Thresholds:
    tolerance: float = 1.5  # PLAN.md §9: <= 1.5x oracle noise
    min_samples: int = 5  # balanced-vs-exact episodes per (stage, embodiment class)
    min_noise_samples: int = 2  # exact-vs-exact episodes per stage
    min_coverage: float = 1.0  # share of exact episodes the balanced run labelled
    slack: Mapping[str, float] = field(default_factory=dict)  # metric -> absolute slack

    def as_dict(self) -> dict[str, Any]:
        return {"tolerance": self.tolerance, "min_samples": self.min_samples,
                "min_noise_samples": self.min_noise_samples, "min_coverage": self.min_coverage,
                "slack": {k: float(v) for k, v in sorted(self.slack.items())}}


# ---------------------------------------------------------------------------------------------
# Records


def read_records(path: str | Path) -> list[StageRecord]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(StageRecord.from_json(line))
    return out


def _index(records: Iterable[StageRecord], stage: str, side: str,
           preset: str | None) -> dict[str, StageRecord]:
    out: dict[str, StageRecord] = {}
    for r in records:
        if r.stage != stage:
            continue
        if preset is not None and r.provenance.preset != preset:
            raise CertifyError(f"{side} records must be preset {preset!r}, got "
                               f"{r.provenance.preset!r}")
        if r.uid in out:
            raise CertifyError(f"{side} has two {stage} records for one episode")
        out[r.uid] = r
    return out


# ---------------------------------------------------------------------------------------------
# Metrics


def interval_iou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = (a.end_s - a.start_s) + (b.end_s - b.start_s) - inter
    if union <= 0:
        return 1.0 if (a.start_s, a.end_s) == (b.start_s, b.end_s) else 0.0
    return inter / union


def align(ref: Sequence[Segment], cand: Sequence[Segment]) -> list[tuple[int, int, float]]:
    """Monotone alignment of two ordered interval lists maximizing total IoU; only pairs with
    IoU > 0 are matched. Returns ``(i, j, iou)`` in order. Ties prefer matching, then skipping
    a reference interval, so the result is deterministic."""
    n, m = len(ref), len(cand)
    iou = [[interval_iou(a, b) for b in cand] for a in ref]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(dp[i - 1][j], dp[i][j - 1])
            if iou[i - 1][j - 1] > 0:
                best = max(best, dp[i - 1][j - 1] + iou[i - 1][j - 1])
            dp[i][j] = best
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        v = iou[i - 1][j - 1]
        if v > 0 and dp[i][j] == dp[i - 1][j - 1] + v:
            pairs.append((i - 1, j - 1, v))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def segment_metrics(ref: StageRecord, cand: StageRecord) -> dict[str, float | None]:
    a, b = ref.segments, cand.segments
    if not a and not b:
        return {"iou_divergence": 0.0, "boundary_error_s": 0.0, "count_mismatch": 0.0,
                "label_disagreement": 0.0}
    pairs = align(a, b)
    slots = max(len(a), len(b))
    bound = [(abs(a[i].start_s - b[j].start_s) + abs(a[i].end_s - b[j].end_s)) / 2
             for i, j, _ in pairs]
    labels = [(a[i].status, a[i].segmentation_provenance) != (b[j].status,
                                                              b[j].segmentation_provenance)
              for i, j, _ in pairs]
    return {
        "iou_divergence": 1.0 - sum(v for _, _, v in pairs) / slots,
        "boundary_error_s": sum(bound) / len(bound) if bound else None,
        "count_mismatch": float(len(a) != len(b)),
        "label_disagreement": sum(labels) / len(labels) if labels else None,
    }


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    longest = max(len(a), len(b))
    return levenshtein(a, b) / longest if longest else 0.0


def caption_metrics(ref: StageRecord, cand: StageRecord,
                    similarity: Similarity | None = None) -> dict[str, float | None]:
    a, b = ref.segments, cand.segments
    slots = max(len(a), len(b))
    if not slots:
        out: dict[str, float | None] = {"caption_mismatch": 0.0, "caption_edit_distance": 0.0}
        if similarity is not None:
            out[SIMILARITY_METRIC] = 0.0
        return out
    pairs = align(a, b)
    unmatched = slots - len(pairs)
    texts = [((a[i].caption or "").strip(), (b[j].caption or "").strip()) for i, j, _ in pairs]
    out = {
        "caption_mismatch": 1.0 - sum(x == y for x, y in texts) / slots,
        "caption_edit_distance": (sum(normalized_edit_distance(x, y) for x, y in texts)
                                  + unmatched) / slots,
    }
    if similarity is not None:
        sims = [min(1.0, max(0.0, float(similarity(x, y)))) for x, y in texts]
        out[SIMILARITY_METRIC] = (sum(1.0 - s for s in sims) + unmatched) / slots
    return out


def terminal_metrics(ref: StageRecord, cand: StageRecord) -> dict[str, float | None] | None:
    if ref.terminal is None or cand.terminal is None:
        return None
    return {"expected_score_abs_delta": abs(ref.terminal.expected_score
                                            - cand.terminal.expected_score),
            "score_mismatch": float(ref.terminal.score != cand.terminal.score)}


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rho with average ranks for ties; None when either side is constant."""
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True)) / math.sqrt(va * vb)


def dense_metrics(ref: StageRecord, cand: StageRecord) -> dict[str, float | None] | None:
    if ref.dense is None or cand.dense is None:
        return None
    a, b = list(ref.dense.per_step), list(cand.dense.per_step)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n == 0:
        delta: float | None = 0.0 if len(ref.dense.per_step) == len(cand.dense.per_step) else None
        rank: float | None = delta
    else:
        delta = sum(abs(x - y) for x, y in zip(a, b, strict=True)) / n
        rho = spearman(a, b) if n >= 2 else None
        rank = (1.0 - rho) if rho is not None else (0.0 if a == b else 1.0)
    return {"per_step_abs_delta": delta, "rank_divergence": rank,
            "length_mismatch": float(len(ref.dense.per_step) != len(cand.dense.per_step))}


def episode_metrics(stage: str, ref: StageRecord, cand: StageRecord,
                    similarity: Similarity | None = None) -> dict[str, float | None] | None:
    if stage == "segment":
        return segment_metrics(ref, cand)
    if stage == "caption":
        return caption_metrics(ref, cand, similarity)
    if stage == "reward.terminal":
        return terminal_metrics(ref, cand)
    if stage == "reward.dense":
        return dense_metrics(ref, cand)
    raise CertifyError(f"no certification metrics for stage {stage!r}")


def metric_names(stage: str, similarity: Similarity | None) -> tuple[str, ...]:
    names = STAGE_METRICS[stage]
    return names + (SIMILARITY_METRIC,) if stage == "caption" and similarity else names


def _mean(values: Iterable[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def compare(stage: str, ref: Mapping[str, StageRecord], cand: Mapping[str, StageRecord],
            similarity: Similarity | None = None) -> dict[str, dict[str, float | None]]:
    """uid -> per-episode metrics, for every episode both sides labelled."""
    out = {}
    for uid in sorted(set(ref) & set(cand)):
        m = episode_metrics(stage, ref[uid], cand[uid], similarity)
        if m is not None:
            out[uid] = m
    return out


# ---------------------------------------------------------------------------------------------
# L0 / Gate A (balanced_lab output)


def l0_summary(stage: str, lab: Mapping[str, Any] | None, gpu: str) -> dict[str, Any]:
    """``{"status": pass|fail|missing|not_applicable, ...counts}`` from a balanced_lab result."""
    if stage not in L0_STAGES:
        return {"status": "not_applicable"}  # rewards: the same NVIDIA vLLM engine as exact
    if lab is None:
        return {"status": "missing"}
    if lab.get("kind") != L0_STAGES[stage]:
        raise CertifyError(f"L0 JSON is for kind {lab.get('kind')!r}, not {L0_STAGES[stage]!r}")
    lab_gpu = (lab.get("baseline") or {}).get("gpu") or lab.get("gpu")
    if lab_gpu and str(lab_gpu) != gpu:
        raise CertifyError(f"L0 JSON was measured on {lab_gpu!r}, not {gpu!r}")
    rows = list((lab.get("baseline") or {}).get("l0") or [])
    if not rows:
        return {"status": "missing"}

    def count(key: str) -> int:
        return sum(1 for r in rows if r.get(key) is True)

    ok = all(r.get("ids_equal") is True and r.get("grid_equal") is True
             and r.get("render_ids_equal", True) is True for r in rows)
    return {"status": "pass" if ok else "fail", "requests": len(rows),
            "ids_equal": count("ids_equal"), "render_ids_equal": count("render_ids_equal"),
            "grid_equal": count("grid_equal")}


def gate_a_summary(lab: Mapping[str, Any] | None) -> dict[str, Any] | None:
    g = (lab or {}).get("gate_a")
    if not isinstance(g, Mapping):
        return None
    keep = ("requests", "greedy_identical", "positions", "max_abs_dlogit", "mean_abs_dlogit")
    return {k: (_r(g[k]) if isinstance(g.get(k), float) else g.get(k)) for k in keep
            if isinstance(g.get(k), int | float) or g.get(k) is None}


# ---------------------------------------------------------------------------------------------
# Certificate


def _uid_hash(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def episode_set(uids: Iterable[str]) -> dict[str, Any]:
    hashes = sorted(_uid_hash(u) for u in uids)
    return {"count": len(hashes), "digest": sha256_hex(hashes), "uid_sha256": hashes}


def _gpu_matches(pool_gpu: str, reported: str | None) -> bool:
    if not reported:
        return True
    base = pool_gpu.rstrip("!").split("-")[0].lower()
    return base in reported.lower()


def certify_stage(
    stage: str,
    *,
    exact: Iterable[StageRecord],
    balanced: Iterable[StageRecord],
    noise: Iterable[StageRecord] | None = None,
    engine_spec: Mapping[str, Any],
    gpu: str,
    preset: str = "balanced",
    kind: str | None = None,
    classes: Mapping[str, str] | None = None,
    l0: Mapping[str, Any] | None = None,
    thresholds: Thresholds | None = None,
    similarity: Similarity | None = None,
    similarity_name: str | None = None,
    derived_weights: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The certificate body (with its content-addressed ``id``) for one stage.

    ``engine_spec``: ``validate.certkey.engine_spec_document`` of the stack the balanced records
    came from; its hash is the certificate's match key. ``classes``: uid -> embodiment class
    (pool); episodes without one are class ``all``."""
    from flora_deploy import __version__

    if stage not in STAGE_METRICS:
        raise CertifyError(f"no certification metrics for stage {stage!r}")
    th = thresholds or Thresholds()
    kind = kind or stage
    ref = _index(exact, stage, "exact", "exact")
    cand = _index(balanced, stage, "balanced", preset)
    nz = _index(noise, stage, "noise", "exact") if noise is not None else None
    if not ref:
        raise CertifyError(f"no exact {stage} records")
    if not cand:
        raise CertifyError(f"no {preset} {stage} records")
    hashes = {r.provenance.config_hash for r in cand.values()}
    if len(hashes) != 1 or not next(iter(hashes)):
        raise CertifyError(f"{preset} {stage} records must share one stage hash "
                           f"(found {len(hashes)})")
    if not all(_gpu_matches(gpu, r.provenance.gpu) for r in cand.values()):
        raise CertifyError(f"{preset} {stage} records were not produced on {gpu}")
    names = metric_names(stage, similarity)

    per_ep = compare(stage, ref, cand, similarity)
    by_class: dict[str, list[str]] = {}
    for uid in per_ep:
        by_class.setdefault((classes or {}).get(uid, DEFAULT_CLASS), []).append(uid)
    candidate = {c: {"n": len(uids), "means": {m: _r(_mean(per_ep[u][m] for u in uids))
                                               for m in names}}
                 for c, uids in sorted(by_class.items())}
    coverage = len(per_ep) / len(ref)

    noise_block = None
    noise_means: dict[str, float | None] = {}
    if nz is not None:
        nz_ep = compare(stage, ref, nz, similarity)
        noise_means = {m: _mean(nz_ep[u][m] for u in nz_ep) for m in names}
        noise_block = {"n": len(nz_ep), "means": {m: _r(v) for m, v in noise_means.items()},
                       "episodes": {k: v for k, v in episode_set(nz_ep).items()
                                    if k != "uid_sha256"}}

    per_metric = {}
    for m in names:
        slack = float(th.slack.get(m, 0.0))
        base = noise_means.get(m) if nz is not None else None
        per_metric[m] = {"slack": slack,
                         "threshold": _r(th.tolerance * base + slack) if base is not None
                         else None}
    checks = []
    for c, block in candidate.items():
        for m in names:
            v, t = block["means"][m], per_metric[m]["threshold"]
            ok = None if v is None or t is None else bool(v <= t + 1e-12)
            checks.append({"class": c, "metric": m, "value": v, "threshold": t, "pass": ok})

    l0s = l0_summary(stage, l0, gpu)
    failures, provisional = [], []
    if any(ch["pass"] is False for ch in checks):
        failures.append("divergence_above_noise_floor")
    if l0s["status"] == "fail":
        failures.append("l0_failed")
    if nz is None:
        provisional.append("no_noise_run")
    elif noise_block["n"] < th.min_noise_samples:
        provisional.append("too_few_noise_samples")
    if any(b["n"] < th.min_samples for b in candidate.values()) or not candidate:
        provisional.append("too_few_samples")
    if coverage < th.min_coverage:
        provisional.append("incomplete_coverage")
    if l0s["status"] == "missing":
        provisional.append("no_l0")
    if nz is not None and any(ch["pass"] is None for ch in checks):
        provisional.append("metric_not_evaluable")
    verdict = "fail" if failures else ("provisional" if provisional else "certified")

    eng = engine_spec.get("engine", {})
    body: dict[str, Any] = {
        "schema": CERT_SCHEMA,
        "generator": {"name": GENERATOR, "version": CERTIFY_VERSION,
                      "framework": f"flora-deploy {__version__}"},
        "verdict": verdict,
        "reasons": sorted(failures) + sorted(set(provisional)),
        "unlocks": verdict == "certified",
        "preset": preset,
        "stage": stage,
        "kind": kind,
        "gpu": gpu,
        "stage_hash": next(iter(hashes)),
        "engine_spec_hash": sha256_hex(dict(engine_spec)),
        "engine": {k: eng.get(k) for k in ("name", "version", "image", "image_fingerprint",
                                           "patches")},
        "modes": dict(engine_spec.get("modes") or {}),
        "logic": {"version": engine_spec.get("logic_version"),
                  "digest": engine_spec.get("logic_digest")},
        "pins": {k: v.get("revision") for k, v in sorted(
            (engine_spec.get("nvidia_pins") or {}).items())},
        "derived_weights": dict(sorted((derived_weights or {}).items())),
        "episodes": episode_set(per_ep),
        "metrics": {"candidate": candidate, "noise": noise_block, "coverage": _r(coverage),
                    "similarity_hook": similarity_name if similarity else None},
        "thresholds": {**th.as_dict(), "per_metric": per_metric},
        "checks": checks,
        "l0": l0s,
        "gate_a": gate_a_summary(l0),
    }
    body["id"] = certificate_id(body)
    return body


def write_certificate(cert: Mapping[str, Any], cert_dir: str | Path | None = None) -> Path:
    """Write ``<id>.json`` (content-addressed; rewriting the same body is a no-op)."""
    if cert.get("id") != certificate_id(cert):
        raise CertifyError("certificate id is not its content address")
    d = Path(cert_dir) if cert_dir is not None else CERT_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{cert['id']}.json"
    p.write_text(json.dumps(json.loads(canonical(cert)), indent=1, sort_keys=True) + "\n",
                 encoding="utf-8")
    return p
