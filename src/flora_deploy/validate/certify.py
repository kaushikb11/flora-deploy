"""Gate certificate generator (PLAN.md §4, D9, D16): ``balanced`` vs ``exact``, within noise.

Inputs, all ``flora_labels_v1`` records (JSONL):

- ``exact``: the oracle run (NVIDIA's own HF code; model inputs proven identical);
- ``noise`` (optional): the oracle run repeated on a subset of its episodes (same stack, stage
  hash, GPU and determinism mode). ``exact`` vs ``noise`` is NVIDIA's own run-to-run divergence
  (``docs/determinism.md``: the H200 default path is not deterministic), the **noise floor**;
- ``balanced``: the candidate stack on the same episodes;
- optional L0 / Gate A JSON from ``modal_app/balanced_lab.py``.

**Bound to the records** (fail closed): every record must carry its stage hash, GPU, embodiment
and ``provenance.cert_key`` (the ``validate.certkey.key_document`` of the stack that made it).
Each set must be uniform in them; the candidate's key must be the stack this checkout would run;
the noise run must repeat the oracle's. Fake-model records are refused outright
(:class:`CertifyRefused`, ``fake_models``).

Per stage, every metric below is a *divergence* (0 = identical; definitions in
``docs/certification.md``), computed per episode and averaged:

- ``segment``: ``iou_divergence``, ``boundary_error_s``, ``count_mismatch``,
  ``label_disagreement`` over a monotone IoU alignment of the subtask intervals;
- ``caption``: ``caption_mismatch`` (1 - exact-match rate), ``caption_edit_distance``
  (normalized Levenshtein), and ``caption_similarity_divergence`` when a text-similarity hook is
  given (default off; nothing here touches the network);
- ``reward.terminal``: ``expected_score_abs_delta``, ``score_mismatch``;
- ``reward.dense``: ``per_step_abs_delta``, ``rank_divergence`` (1 - Spearman rho; mean
  ``|delta|`` when a side is constant), ``length_mismatch``.

Pass rule, per (stage, embodiment pool) and metric: ``mean(balanced vs exact) <= min(cap,
max(absolute_floor, tolerance x max(noise, floor) + slack))`` (``certkey.metric_threshold``;
knobs and floors bounded by ``certkey.POLICY``). The other segment / caption metrics are
averaged over the episodes whose counts agree (:data:`COUNT_CONDITIONED`), per group (the pool,
and the cold witness on its own). ``count_mismatch`` (absolute floor 10%: about one episode in
ten may differ in subtask count) is judged once, over the **union** of the candidate's and the
witness's episodes (each witness episode is a sample of its own: same episode, another
container): ``flips <= threshold x n``, rounded half up (``certkey.count_flips_allowed``: 2 of
18 pass). ``metrics.count_flips`` records the per-group and the union figures.

**Which episodes** (review 13 S3-3): every episode both sides labelled, never a chosen subset.
The candidate is paired with the oracle (at most ``max_missing`` oracle episodes may lack a
candidate record: the spares), the noise run with the oracle (noise episodes the oracle did not
label are dropped, so a spare noise episode fills in), and so is the optional **cold witness**
(``witness``: balanced records of one container forced cold, ``FLORA_COMPILE_CACHE=off``,
held to the same mean thresholds on its own, its count flips counted in the union).

Verdict: ``fail`` if any check fails or a present L0 fails; else ``provisional`` if there is no
noise run, too few samples (per pool, in the noise run or in the witness), too many missing
episodes, (video stages) no L0 or an L0 measured under other engine settings (``stale_l0``), a
candidate or witness record whose engine preempted while it ran (``engine_preempted``) or that
does not say (``preemptions_unknown``), (video stages) no witness record forced cold
(``no_witness``), or a witness with some record not forced cold; else ``certified``. The
certificate states the compile caches its evidence ran (``compile_cache``).

The certificate holds metrics only: no record string ever goes in (no captions, prompts,
rationales, ids, dataset names). Episode ids are sha256-hashed. The record values carried are
our own: stage hashes (validated hex), the key's contents (equal to this checkout's), the
normalized GPU, the embodiment pool and the determinism mode (validated enums).
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
    _HEX64,
    _STAMP,
    ABSOLUTE_CAPS,
    ABSOLUTE_FLOOR,
    CERT_DIR,
    CERT_SCHEMA,
    COUNT_METRIC,
    FAKE_CERTIFICATES,
    GENERATOR,
    MAX_MISSING,
    MIN_NOISE_SAMPLES_FLOOR,
    MIN_SAMPLES_FLOOR,
    NOISE_FLOOR,
    UNION_CLASS,
    canonical,
    certificate_id,
    count_flips_allowed,
    document_problems,
    engine_settings_hash,
    metric_threshold,
    sha256_hex,
    threshold_problems,
)

CERTIFY_VERSION = "3"
# Provenance every record must carry to be certified against (docs/certification.md).
REQUIRED_PROVENANCE = ("config_hash", "gpu", "embodiment", "cert_key")
EMBODIMENTS = ("human", "yam", "yam32", "dex", "tianji")  # config.schema.Embodiment
DETERMINISM_MODES = ("nvidia", "deterministic")  # accel.determinism.MODES

Similarity = Callable[[str, str], float]

STAGE_METRICS: dict[str, tuple[str, ...]] = {
    "segment": ("iou_divergence", "boundary_error_s", "count_mismatch", "label_disagreement"),
    "caption": ("caption_mismatch", "caption_edit_distance", "count_mismatch"),
    "reward.terminal": ("expected_score_abs_delta", "score_mismatch"),
    "reward.dense": ("per_step_abs_delta", "rank_divergence", "length_mismatch"),
}
SIMILARITY_METRIC = "caption_similarity_divergence"
# Stages whose balanced engine differs from exact's, so L0 input parity must be shown. Their
# records also carry the compile cache and the engine's preemptions (records.Provenance).
L0_STAGES = {"segment": "tempseg", "caption": "captioner"}
# Stages whose other metrics are averaged over the episodes with equal subtask counts: an
# episode whose count differs is bounded by ``count_mismatch`` (absolute floor 10%) alone, and
# its interval / caption pairing is not meaningful.
COUNT_CONDITIONED = ("segment", "caption")
WITNESS_CLASS = "cold-witness"
FORCED_COLD = "cold:forced"  # engines.compile_cache.CacheStatus.identity of a forced container


class CertifyError(ValueError):
    pass


class CertifyRefused(CertifyError):
    """The records cannot be certified at all (``certified: false`` with ``reason``)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Thresholds:
    """The pass-rule knobs, bounded by ``certkey.POLICY`` (out of bounds: ``CertifyError``)."""

    tolerance: float = 1.5  # PLAN.md §4: <= 1.5x oracle noise
    min_samples: int = MIN_SAMPLES_FLOOR  # balanced-vs-exact episodes per (stage, pool)
    min_noise_samples: int = MIN_NOISE_SAMPLES_FLOOR  # exact-vs-exact episodes per stage
    max_missing: int = MAX_MISSING  # oracle episodes without a candidate record (the spares)
    slack: Mapping[str, float] = field(default_factory=dict)  # metric -> absolute slack

    def __post_init__(self) -> None:
        bad = threshold_problems(self.tolerance, self.min_samples, self.min_noise_samples,
                                 self.max_missing, self.slack)
        if bad:
            raise CertifyError("thresholds outside policy (certkey.POLICY): " + "; ".join(bad))

    def as_dict(self) -> dict[str, Any]:
        return {"tolerance": self.tolerance, "min_samples": self.min_samples,
                "min_noise_samples": self.min_noise_samples, "max_missing": self.max_missing,
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
        out: dict[str, float | None] = {"caption_mismatch": 0.0, "caption_edit_distance": 0.0,
                                        "count_mismatch": 0.0}
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
        "count_mismatch": float(len(a) != len(b)),
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
    """``rank_divergence`` is ``1 - rho``; when rho is undefined (a side is constant, or fewer
    than 2 steps) it is the mean ``|delta|`` instead, so equal constant series score 0 and one
    differing step in a constant series scores ``1/n``, not the maximum (M6)."""
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
        rank = (1.0 - rho) if rho is not None else delta
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


def metric_basis(stage: str, metric: str) -> str:
    """The episodes a metric's mean is over: ``all`` or ``count_matched``."""
    return ("count_matched" if stage in COUNT_CONDITIONED and metric != "count_mismatch"
            else "all")


def stage_means(stage: str, per_ep: Mapping[str, Mapping[str, float | None]],
                names: Sequence[str]) -> dict[str, float | None]:
    """Per metric, the mean over :func:`metric_basis`'s episodes."""
    matched = [u for u in per_ep if not per_ep[u].get("count_mismatch")]
    return {m: _r(_mean(per_ep[u][m] for u in (matched if metric_basis(stage, m)
                                                  == "count_matched" else per_ep)))
            for m in names}


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


def l0_summary(stage: str, lab: Mapping[str, Any] | None, gpu: str,
               engine_hash: str | None = None) -> dict[str, Any]:
    """``{"status": pass|fail|stale|missing|not_applicable, ...counts}`` from a balanced_lab
    result. ``engine_hash``: ``certkey.engine_settings_hash`` of the candidate's vLLM engine;
    a lab that did not record that hash for its baseline engine (another ``max_num_seqs``, say,
    or a lab from before the hash) is ``stale`` (review 13 S3-1)."""
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
    got = (lab.get("engine_hashes") or {}).get("baseline")
    lab_hash = got if isinstance(got, str) and _HEX64.match(got) else None
    status = "pass" if ok else "fail"
    if ok and engine_hash is not None and lab_hash != engine_hash:
        status = "stale"
    return {"status": status, "requests": len(rows),
            "ids_equal": count("ids_equal"), "render_ids_equal": count("render_ids_equal"),
            "grid_equal": count("grid_equal"), "engine_hash": lab_hash}


def gate_a_summary(lab: Mapping[str, Any] | None) -> dict[str, Any] | None:
    g = (lab or {}).get("gate_a")
    if not isinstance(g, Mapping):
        return None
    keep = ("requests", "greedy_identical", "positions", "max_abs_dlogit", "mean_abs_dlogit")
    return {k: (_r(g[k]) if isinstance(g.get(k), float) else g.get(k)) for k in keep
            if isinstance(g.get(k), int | float) or g.get(k) is None}


# ---------------------------------------------------------------------------------------------
# Binding the certificate to the records (H1, H2, H5)


def _uid_hash(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def episode_set(uids: Iterable[str]) -> dict[str, Any]:
    hashes = sorted(_uid_hash(u) for u in uids)
    return {"count": len(hashes), "digest": sha256_hex(hashes), "uid_sha256": hashes}


def normalized_gpu(name: str | None, side: str = "records") -> str:
    """``engines.offline_llm.normalize_gpu``; an unknown GPU is refused (exact names only)."""
    from flora_deploy.engines.offline_llm import EngineArgsError, normalize_gpu

    try:
        return normalize_gpu(str(name))
    except EngineArgsError:
        raise CertifyError(f"{side}: unknown GPU {name!r}") from None


def is_fake(p) -> bool:
    """Records made by fake engines (``flora run --local --fake-models``)."""
    return (p.certificate in FAKE_CERTIFICATES
            or str(p.engine).startswith(("flora-local-fake", "fake"))
            or any(str(v).startswith(("fake", "flora-local-fake")) for v in p.models.values()))


def missing_provenance(p) -> list[str]:
    """The binding fields (docs/certification.md) a record's provenance lacks."""
    out = [f for f in REQUIRED_PROVENANCE if not getattr(p, f, None)]
    if p.config_hash and not _HEX64.match(p.config_hash):
        out.append("config_hash (a 64-hex stage hash)")
    if p.cert_key is not None:
        out += [f"cert_key: {x}" for x in document_problems(p.cert_key)]
    return out


def bind(records: Mapping[str, StageRecord], stage: str, side: str) -> dict[str, Any]:
    """The identity a set of records was produced under, derived from their provenance:
    ``{key, doc, stage_hash, gpu, embodiment, determinism}``. Fake-model records are refused
    outright (:class:`CertifyRefused` ``fake_models``); a record missing a binding field
    (``missing_provenance``) or a set that is not uniform in one is refused (fail closed)."""
    from flora_deploy.config.presets import get_preset

    provs = [r.provenance for r in records.values()]
    if any(is_fake(p) for p in provs):
        raise CertifyRefused("fake_models", f"{side} {stage} records come from fake models "
                             "(provenance.certificate / engine / models); they certify nothing")
    missing = sorted({m for p in provs for m in missing_provenance(p)})
    if missing:
        raise CertifyRefused("missing_provenance", f"{side} {stage} records lack provenance "
                             f"{missing}: workers must stamp them before their records can be "
                             "certified (docs/certification.md)")

    def one(what: str, values: set) -> Any:
        if len(values) != 1:
            raise CertifyError(f"{side} {stage} records must share one {what} "
                               f"(found {len(values)})")
        return next(iter(values))

    key = one("certificate key (provenance.cert_key)", {sha256_hex(p.cert_key) for p in provs})
    doc = provs[0].cert_key
    stage_hash = one("stage hash", {p.config_hash for p in provs})
    gpu = one("GPU", {normalized_gpu(p.gpu, f"{side} {stage} records") for p in provs})
    emb = one("embodiment", {p.embodiment for p in provs})
    if emb not in EMBODIMENTS:
        raise CertifyError(f"{side} {stage} records: unknown embodiment {emb!r}")
    preset = one("preset", {p.preset for p in provs})
    if doc["preset"] != preset:
        raise CertifyError(f"{side} {stage} records: cert_key is for preset {doc['preset']!r}, "
                           f"provenance says {preset!r}")
    want_kind = get_preset(preset).stage_kinds.get(stage)
    if doc["kind"] != want_kind:
        raise CertifyError(f"{side} {stage} records: cert_key is for kind {doc['kind']!r}, "
                           f"not {want_kind!r}")
    revs = {k: (v or {}).get("revision") for k, v in (doc.get("nvidia_pins") or {}).items()}
    if any(k in revs and revs[k] != v for p in provs for k, v in p.models.items()):
        raise CertifyError(f"{side} {stage} records: provenance.models disagree with the "
                           "cert_key's NVIDIA pins")
    vllm = doc.get("vllm_engine")
    if vllm and any(p.engine != vllm.get("label") for p in provs):
        raise CertifyError(f"{side} {stage} records: provenance.engine is not the cert_key's "
                           f"engine {vllm.get('label')!r}")
    det = (doc.get("modes") or {}).get("determinism")
    if det is not None and det not in DETERMINISM_MODES:
        raise CertifyError(f"{side} {stage} records: unknown determinism mode {det!r}")
    return {"key": key, "doc": doc, "stage_hash": stage_hash, "gpu": gpu, "embodiment": emb,
            "determinism": det}


def _key_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# ---------------------------------------------------------------------------------------------
# Certificate


def now_stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_counts(records: Iterable[StageRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        k = r.provenance.compile_cache or "unknown"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _paired(side: Mapping[str, StageRecord], ref: Mapping[str, StageRecord]
            ) -> tuple[dict[str, StageRecord], int]:
    """``side`` restricted to the oracle's episodes, and how many were dropped."""
    kept = {u: r for u, r in side.items() if u in ref}
    return kept, len(side) - len(kept)


def certify_stage(
    stage: str,
    *,
    exact: Iterable[StageRecord],
    balanced: Iterable[StageRecord],
    noise: Iterable[StageRecord] | None = None,
    witness: Iterable[StageRecord] | None = None,
    engine_spec: Mapping[str, Any],
    gpu: str,
    preset: str = "balanced",
    kind: str | None = None,
    l0: Mapping[str, Any] | None = None,
    thresholds: Thresholds | None = None,
    similarity: Similarity | None = None,
    similarity_name: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """The certificate body (with its content-addressed ``id``) for one stage.

    ``engine_spec``: ``validate.certkey.key_document`` of the stack as this checkout would run
    it. The candidate records' own key (``provenance.cert_key``) must equal it; the
    certificate's key is the records'. The embodiment pool comes from the records.
    ``witness``: the cold witness (module docstring), the same stack as the candidate.
    ``generated_at``: ``YYYY-MM-DDTHH:MM:SSZ`` (default: now); the newest certificate for a key
    decides (``certkey.matching``)."""
    from flora_deploy import __version__

    if stage not in STAGE_METRICS:
        raise CertifyError(f"no certification metrics for stage {stage!r}")
    th = thresholds or Thresholds()
    kind = kind or stage
    ref = _index(exact, stage, "exact", "exact")
    cand = _index(balanced, stage, "balanced", preset)
    nz = _index(noise, stage, "noise", "exact") if noise is not None else None
    wit = _index(witness, stage, "witness", preset) if witness is not None else None
    if not ref:
        raise CertifyError(f"no exact {stage} records")
    if not cand:
        raise CertifyError(f"no {preset} {stage} records")
    stamp = generated_at or now_stamp()
    if not _STAMP.match(stamp):
        raise CertifyError(f"generated_at {stamp!r}: use YYYY-MM-DDTHH:MM:SSZ")

    # H1: the key is the records'; it must be the stack this checkout would run.
    b = bind(cand, stage, preset)
    if b["key"] != sha256_hex(dict(engine_spec)):
        raise CertifyError(f"{preset} {stage} records were made by another stack than this "
                           f"checkout's (cert_key differs in {_key_diff(b['doc'], engine_spec)})"
                           "; certify from the commit, image and weights that ran them")
    if b["doc"]["kind"] != kind:
        raise CertifyError(f"{preset} {stage} records are for kind {b['doc']['kind']!r}")
    if b["gpu"] != normalized_gpu(gpu, "certified GPU"):
        raise CertifyError(f"{preset} {stage} records were not produced on {gpu}")
    # H2: the oracle is one exact run; the noise run repeats it (same stack, stage hash, GPU,
    # determinism mode) on some of its episodes: those the oracle labelled are paired.
    o = bind(ref, stage, "exact")
    if o["embodiment"] != b["embodiment"]:
        raise CertifyError(f"exact and {preset} {stage} records are for different embodiments")
    unpaired = 0
    if nz is not None:
        n = bind(nz, stage, "noise")
        for what in ("stage_hash", "key", "gpu", "determinism"):
            if n[what] != o[what]:
                raise CertifyError(f"noise {stage} records differ from the oracle's in {what}: "
                                   "the noise run must repeat the oracle run")
        if any(r.audit_ref and u in ref and r.audit_ref == ref[u].audit_ref
               for u, r in nz.items()):
            raise CertifyError(f"noise {stage} records are copies of the oracle's (audit_ref)")
        nz, unpaired = _paired(nz, ref)
    wit_unpaired = 0
    if wit is not None:
        w = bind(wit, stage, "witness")
        for what in ("key", "stage_hash", "gpu", "embodiment"):
            if w[what] != b[what]:
                raise CertifyError(f"witness {stage} records differ from the candidate's in "
                                   f"{what}: the witness must be the same stack, forced cold")
        wit, wit_unpaired = _paired(wit, ref)
    names = metric_names(stage, similarity)
    pool = b["embodiment"]

    per_ep = compare(stage, ref, cand, similarity)
    matched = sum(1 for u in per_ep if not per_ep[u].get("count_mismatch"))
    candidate = ({pool: {"n": len(per_ep), "count_matched": matched,
                         "means": stage_means(stage, per_ep, names)}} if per_ep else {})
    coverage = len(per_ep) / len(ref)
    missing = len(set(ref) - set(cand))

    noise_block = None
    noise_means: dict[str, float | None] = {}
    if nz is not None:
        nz_ep = compare(stage, ref, nz, similarity)
        noise_means = stage_means(stage, nz_ep, names)
        noise_block = {"n": len(nz_ep), "means": noise_means, "unpaired": unpaired,
                       "episodes": {k: v for k, v in episode_set(nz_ep).items()
                                    if k != "uid_sha256"}}
    witness_block = None
    if wit is not None:
        w_ep = compare(stage, ref, wit, similarity)
        witness_block = {"n": len(w_ep), "means": stage_means(stage, w_ep, names),
                         "unpaired": wit_unpaired,
                         "episodes": {k: v for k, v in episode_set(w_ep).items()
                                      if k != "uid_sha256"}}

    per_metric = {}
    for m in names:
        slack = float(th.slack.get(m, 0.0))
        base = noise_means.get(m) if nz is not None else None
        per_metric[m] = {"slack": slack, "floor": NOISE_FLOOR.get(m, 0.0),
                         "absolute_floor": ABSOLUTE_FLOOR.get(m, 0.0),
                         "cap": ABSOLUTE_CAPS.get(m), "over": metric_basis(stage, m),
                         "threshold": metric_threshold(m, base, th.tolerance, slack)
                         if base is not None else None}
    blocks = dict(candidate)
    if witness_block is not None:
        blocks[WITNESS_CLASS] = witness_block
    checks = []
    for c, block in blocks.items():
        for m in names:
            if m == COUNT_METRIC:  # over the union, below
                continue
            v, t = block["means"][m], per_metric[m]["threshold"]
            ok = None if v is None or t is None else bool(v <= t + 1e-12)
            checks.append({"class": c, "metric": m, "value": v, "threshold": t, "pass": ok})
    # The count-flip check: over the union of the candidate's and the witness's episodes (a
    # witness episode is a sample of its own), flips <= threshold x n rounded half up.
    count_flips = None
    if COUNT_METRIC in names:
        eps = {pool: per_ep} if per_ep else {}
        if wit is not None:
            eps[WITNESS_CLASS] = w_ep
        groups = {c: {"n": len(e), "flips": sum(1 for x in e.values() if x[COUNT_METRIC])}
                  for c, e in eps.items()}
        n = sum(g["n"] for g in groups.values())
        k = sum(g["flips"] for g in groups.values())
        t = per_metric[COUNT_METRIC]["threshold"]
        allowed = count_flips_allowed(t, n) if t is not None else None
        ok = None if allowed is None or not n else bool(k <= allowed)
        count_flips = {"groups": groups,
                       "union": {"n": n, "flips": k, "rate": _r(k / n) if n else None,
                                 "threshold": t, "allowed": allowed}}
        checks.append({"class": UNION_CLASS, "metric": COUNT_METRIC,
                       "value": _r(k / n) if n else None, "threshold": t, "flips": k, "n": n,
                       "allowed": allowed, "pass": ok})

    vllm = b["doc"].get("vllm_engine")
    l0s = l0_summary(stage, l0, gpu,
                     engine_settings_hash(vllm) if stage in L0_STAGES and vllm else None)
    engine_recs = [*cand.values(), *((wit or {}).values())] if stage in L0_STAGES else []
    preempt = {"preempted": sum(1 for r in engine_recs
                                if (r.provenance.engine_preemptions or 0) > 0),
               "unknown": sum(1 for r in engine_recs if r.provenance.engine_preemptions is None)}
    failures, provisional = [], []
    if any(ch["pass"] is False for ch in checks):
        failures.append("divergence_above_noise_floor")
    if l0s["status"] == "fail":
        failures.append("l0_failed")
    if nz is None:
        provisional.append("no_noise_run")
    elif noise_block["n"] < th.min_noise_samples:
        provisional.append("too_few_noise_samples")
    if any(blk["n"] < th.min_samples for blk in candidate.values()) or not candidate:
        provisional.append("too_few_samples")
    if missing > th.max_missing:
        provisional.append("incomplete_coverage")
    if l0s["status"] == "missing":
        provisional.append("no_l0")
    if l0s["status"] == "stale":
        provisional.append("stale_l0")
    if nz is not None and any(ch["pass"] is None for ch in checks):
        provisional.append("metric_not_evaluable")
    if preempt["preempted"]:
        provisional.append("engine_preempted")
    if preempt["unknown"]:
        provisional.append("preemptions_unknown")
    # Video stages: the evidence must cover a forced-cold container (no_witness: no witness
    # record paired, or none of them forced cold).
    forced = sum(1 for r in (wit or {}).values() if r.provenance.compile_cache == FORCED_COLD)
    if stage in L0_STAGES and not forced:
        provisional.append("no_witness")
    if witness_block is not None and wit:
        if witness_block["n"] < th.min_noise_samples:
            provisional.append("too_few_witness_samples")
        if forced and forced < len(wit):
            provisional.append("witness_not_forced_cold")
    verdict = "fail" if failures else ("provisional" if provisional else "certified")

    caches: dict[str, Any] = {"candidate": _cache_counts(cand.values())}
    if wit is not None:
        caches["witness"] = _cache_counts(wit.values())
    caches["covers"] = sorted({k for v in caches.values() for k in v})
    doc = b["doc"]
    eng = doc.get("engine") or {}
    body: dict[str, Any] = {
        "schema": CERT_SCHEMA,
        "generator": {"name": GENERATOR, "version": CERTIFY_VERSION,
                      "framework": f"flora-deploy {__version__}"},
        "generated_at": stamp,
        "verdict": verdict,
        "reasons": sorted(failures) + sorted(set(provisional)),
        "unlocks": verdict == "certified",
        "preset": preset,
        "stage": stage,
        "kind": kind,
        "gpu": gpu,
        "covered_pools": [pool],
        "stage_hash": b["stage_hash"],
        "engine_spec_hash": b["key"],
        "engine": {**{k: eng.get(k) for k in ("name", "version", "image", "image_fingerprint",
                                              "patches")},
                   "vllm": doc.get("vllm_engine")},
        "modes": dict(doc.get("modes") or {}),
        "logic": {"version": doc.get("logic_version"), "digest": doc.get("logic_digest")},
        "pins": {k: v.get("revision") for k, v in sorted(
            (doc.get("nvidia_pins") or {}).items())},
        "derived_weights": dict(sorted((doc.get("derived_weights") or {}).items())),
        "oracle": {"stage_hash": o["stage_hash"], "engine_spec_hash": o["key"],
                   "gpu": o["gpu"], "determinism": o["determinism"]},
        "episodes": episode_set(per_ep),
        "metrics": {"candidate": candidate, "noise": noise_block, "witness": witness_block,
                    "count_flips": count_flips,
                    "coverage": _r(coverage), "missing": missing,
                    "similarity_hook": similarity_name if similarity else None},
        "thresholds": {**th.as_dict(),
                       "noise_floor": {m: NOISE_FLOOR.get(m, 0.0) for m in names},
                       "absolute_floor": {m: ABSOLUTE_FLOOR[m] for m in names
                                          if m in ABSOLUTE_FLOOR},
                       "per_metric": per_metric},
        "checks": checks,
        "l0": l0s,
        "gate_a": gate_a_summary(l0),
    }
    if stage in L0_STAGES:
        body["preemptions"] = preempt
        body["compile_cache"] = caches
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
