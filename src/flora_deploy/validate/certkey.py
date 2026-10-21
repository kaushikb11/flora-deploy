"""Certificate match key, policy and registry lookup (PLAN.md D9, D16; ``docs/certification.md``).

A Gate certificate (written by ``validate.certify``) says that one shard kind of a non-``exact``
preset, on one GPU type, for the embodiment pools it measured, stays within NVIDIA's own
run-to-run noise. This module decides **which certificate applies to a pool**, so it decides a
record's ``certified`` flag and ``provenance.certificate``. The certificates themselves are
metrics only.

**The key** (:func:`key_document`, hashed by :func:`engine_spec_hash`) is everything that fixes
the stack's outputs apart from the per-job input: the preset and shard kind, the engine identity
and its image fingerprint, the worker's vLLM engine spec when the kind has one (incl. the FP32
``dt_bias`` plugin flag and the per-GPU scheduler limits), the NVIDIA pins, the decode pins, the
derived weights' **content** digests, the kind's output-affecting modes, the logic version and
the kind's certificate logic digest (:func:`cert_logic_digest`). Workers stamp the same document
into every record (``provenance.cert_key``), so a certificate is computed from the records, not
from the checkout that happens to certify them.

**Match rule.** A certificate applies to (preset, shard kind, pool, embodiment) only when all
of these hold:

- ``preset``, ``kind`` and ``gpu`` are equal (``"H100!"`` and ``"H100"`` are different GPUs);
- the config's embodiment is one of its ``covered_pools`` (D16);
- its ``engine_spec_hash`` equals the key of the stack that would run now. The caller must pass
  the derived weights' content digests: without them a kind that needs derived weights matches
  nothing (fail closed);
- a ``certified`` or ``provisional`` certificate passes :func:`policy_problems`: its knobs are
  within :data:`POLICY` and its checks recompute from its own recorded metrics;
- its ``id`` is the content address of its body (a hand-edited file never matches).

Among the matches, the **newest** (``generated_at``) decides; on a tie ``fail`` beats
``provisional`` beats ``certified``. So a newer ``fail`` for the same key revokes an older
``certified``. Only ``verdict == "certified"`` unlocks a preset.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import tokenize
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

CERT_SCHEMA = "flora-certificate/v2"
KEY_SCHEMA = "flora-cert-key/v2"
CERT_LOGIC_SCHEMA = "flora-cert-logic/v1"
CERT_DIR = Path(__file__).resolve().parent.parent / "config" / "certificates"
VERDICTS = ("certified", "provisional", "fail")
GENERATOR = "flora-certify"
GENERATOR_VERSIONS = ("2",)  # generator versions whose certificates matching() accepts

# ``provenance.certificate`` values of records made by fake engines (``local/engines.py``).
FAKE_CERTIFICATES = frozenset({"fake-models", "local-fake-models"})
# Shard kinds with a worker-side vLLM ``engines.offline_llm.EngineSpec`` (its model kind).
VLLM_SPEC_KINDS = {"segment": "tempseg", "caption": "captioner"}
KEY_FIELDS = ("schema", "preset", "kind", "engine", "vllm_engine", "nvidia_pins", "decode",
              "derived_weights", "modes", "logic_version", "logic_digest")
# Not on any kind's output path, so not in the certificate logic digest (M1): the module
# registry itself (membership is still hashed through the relative paths), the D11 hash code,
# and this key code (its semantics are versioned by ``KEY_SCHEMA``).
CERT_LOGIC_EXCLUDE = frozenset({"config/logic.py", "config/hashing.py", "validate/certkey.py"})

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

# ---------------------------------------------------------------------------------------------
# Policy (H3, M4): what a certificate must meet to match. docs/certification.md, "Policy".

TOLERANCE_BOUNDS = (1.0, 2.0)  # PLAN.md §9 uses 1.5
MIN_SAMPLES_FLOOR = 10  # candidate episodes per (stage, embodiment pool)
MIN_NOISE_SAMPLES_FLOOR = 5  # noise-run episodes per stage
MIN_COVERAGE = 1.0
# Absolute noise floor per metric: the threshold is tolerance x max(noise, floor) + slack, so a
# zero measured noise does not demand a bit-identical engine.
NOISE_FLOOR: dict[str, float] = {
    "iou_divergence": 0.02,
    "boundary_error_s": 0.1,  # < half a TempSeg frame tick (4 fps = 0.25 s)
    "count_mismatch": 0.05,
    "label_disagreement": 0.02,
    "caption_mismatch": 0.05,
    "caption_edit_distance": 0.02,
    "caption_similarity_divergence": 0.02,
    "expected_score_abs_delta": 0.02,
    "score_mismatch": 0.05,
    "per_step_abs_delta": 0.02,
    "rank_divergence": 0.05,
    "length_mismatch": 0.0,  # step count is fixed by the video length
}
SLACK_MAX: dict[str, float] = dict(NOISE_FLOOR)  # --slack may at most add one floor
# Absolute bars from PLAN.md §2 ("terminal E[s] MAE <= 0.05"), on top of the noise rule.
ABSOLUTE_CAPS: dict[str, float] = {"expected_score_abs_delta": 0.05}

POLICY: dict[str, Any] = {
    "tolerance": list(TOLERANCE_BOUNDS), "min_samples": MIN_SAMPLES_FLOOR,
    "min_noise_samples": MIN_NOISE_SAMPLES_FLOOR, "min_coverage": MIN_COVERAGE,
    "noise_floor": NOISE_FLOOR, "slack_max": SLACK_MAX, "absolute_caps": ABSOLUTE_CAPS,
}


class CertKeyError(ValueError):
    """The key cannot be computed (e.g. a missing derived-weight content digest)."""


def canonical(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def sha256_hex(doc: Any) -> str:
    return hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()


def metric_threshold(metric: str, noise: float, tolerance: float, slack: float) -> float:
    """``min(cap, tolerance x max(noise, floor) + slack)`` (rounded as certificates are)."""
    t = tolerance * max(float(noise), NOISE_FLOOR.get(metric, 0.0)) + float(slack)
    if metric in ABSOLUTE_CAPS:
        t = min(t, ABSOLUTE_CAPS[metric])
    return round(t, 6)


def threshold_problems(tolerance: float, min_samples: int, min_noise_samples: int,
                       min_coverage: float, slack: Mapping[str, float]) -> list[str]:
    """Why these knobs are outside :data:`POLICY` (empty when they are within it)."""
    out = []
    lo, hi = TOLERANCE_BOUNDS
    if not lo <= float(tolerance) <= hi:
        out.append(f"tolerance {tolerance} outside [{lo}, {hi}]")
    if int(min_samples) < MIN_SAMPLES_FLOOR:
        out.append(f"min_samples {min_samples} < {MIN_SAMPLES_FLOOR}")
    if int(min_noise_samples) < MIN_NOISE_SAMPLES_FLOOR:
        out.append(f"min_noise_samples {min_noise_samples} < {MIN_NOISE_SAMPLES_FLOOR}")
    if float(min_coverage) < MIN_COVERAGE:
        out.append(f"min_coverage {min_coverage} < {MIN_COVERAGE}")
    for m, v in slack.items():
        if m not in SLACK_MAX:
            out.append(f"slack for unknown metric {m!r}")
        elif not 0.0 <= float(v) <= SLACK_MAX[m]:
            out.append(f"slack {m}={v} outside [0, {SLACK_MAX[m]}]")
    return out


# ---------------------------------------------------------------------------------------------
# Certificate logic digest (M1)


def normalized_source(text: str) -> str:
    """Source without comments, docstrings, trailing whitespace or blank lines. Built from the
    text (tokenize finds the comments, ast the docstrings), not from ``ast.dump``, so it is the
    same under every Python version a worker image or a checkout runs."""
    lines = text.splitlines()
    drop: set[int] = set()  # 0-based line numbers
    cut: dict[int, int] = {}  # line -> column where a comment starts
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            cut[row - 1] = min(col, cut.get(row - 1, col))
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef
                          | ast.AsyncFunctionDef):
            continue
        body = node.body
        if not (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            continue
        d = body[0]
        first = lines[d.lineno - 1].encode("utf-8")[:d.col_offset]
        last = lines[d.end_lineno - 1].encode("utf-8")[d.end_col_offset:].decode("utf-8")
        if first.strip() == b"" and (not last.strip() or last.strip().startswith("#")):
            drop.update(range(d.lineno - 1, d.end_lineno))
    out = []
    for i, line in enumerate(lines):
        if i in drop:
            continue
        line = line[:cut[i]] if i in cut else line
        if line.strip():
            out.append(line.rstrip())
    return "\n".join(out)


def cert_modules(kind: str) -> tuple[str, ...]:
    """The modules the certificate logic digest of ``kind`` covers: its D11 logic modules
    (``config.logic.modules_for``) minus :data:`CERT_LOGIC_EXCLUDE`."""
    from flora_deploy.config.logic import modules_for

    return tuple(m for m in modules_for(kind) if m not in CERT_LOGIC_EXCLUDE)


def cert_logic_digest(kind: str, root: Path | None = None) -> str:
    """sha256 over ``(relpath, normalized_source)`` of :func:`cert_modules`. Comment or
    docstring edits, edits to the module registry, and modules outside the kind's groups do not
    change it; a code edit on the kind's path, or adding a module to its group, does."""
    from flora_deploy.config.logic import package_root

    base = root or package_root()
    h = hashlib.sha256(CERT_LOGIC_SCHEMA.encode())
    for rel in sorted(cert_modules(kind)):
        p = base / rel
        if not p.is_file():
            raise FileNotFoundError(f"logic module {rel} not found under {base}")
        h.update(f"\0{rel}\0".encode())
        h.update(_normalized_file(p).encode("utf-8"))
    return h.hexdigest()


_NORMALIZED: dict[tuple[str, int, int], str] = {}


def _normalized_file(p: Path) -> str:
    st = p.stat()
    k = (str(p), st.st_mtime_ns, st.st_size)
    if k not in _NORMALIZED:
        _NORMALIZED[k] = normalized_source(p.read_text(encoding="utf-8"))
    return _NORMALIZED[k]


# ---------------------------------------------------------------------------------------------
# Key


def _default_fingerprint(name: str) -> str:
    from flora_deploy.modal_app.images import image_fingerprint

    return image_fingerprint(name)


def needed_derived(preset: str, kind: str) -> tuple[str, ...]:
    from flora_deploy.config.presets import get_preset

    return tuple(sorted(get_preset(preset).derived_weights.get(kind, ())))


def key_document(preset: str, kind: str, gpu: str, *,
                 modes: Mapping[str, Any] | None = None,
                 derived_content: Mapping[str, str] | None = None,
                 dtbias_fp32: bool = False, engine=None, image_fingerprint=None,
                 logic_root: Path | None = None) -> dict[str, Any]:
    """What :func:`engine_spec_hash` hashes, and what a worker stamps into
    ``provenance.cert_key`` (see the module docstring). ``derived_content``: derived-weight
    name (``captioner_bf16``) -> content digest (W1-Prepare's ``content_digest(receipt)`` /
    ``DerivedPin.content_sha256``); every name the kind needs is required. ``gpu`` only selects
    the vLLM engine's per-GPU limits; the GPU itself is matched separately."""
    from flora_deploy import LOGIC_VERSION
    from flora_deploy.config.hashing import IMAGE_ARCH, _pin_identity
    from flora_deploy.config.pins import RUNTIME
    from flora_deploy.config.presets import get_preset

    spec = get_preset(preset)
    eng = engine if engine is not None else spec.engines[kind]
    e = asdict(eng)
    e["patches"] = list(eng.patches)
    e["image_fingerprint"] = (image_fingerprint or _default_fingerprint)(eng.image)
    vllm = None
    if kind in VLLM_SPEC_KINDS:
        from flora_deploy.engines.offline_llm import EngineArgsError, EngineSpec

        try:
            vllm = EngineSpec.for_gpu(VLLM_SPEC_KINDS[kind], "", gpu,
                                      dtbias_fp32=bool(dtbias_fp32)).hashed()
        except EngineArgsError as err:
            raise CertKeyError(str(err)) from None
    dc = dict(derived_content or {})
    needed = needed_derived(preset, kind)
    missing = [n for n in needed if not _HEX64.match(str(dc.get(n) or ""))]
    if missing:
        raise CertKeyError(f"{kind}: no content digest for derived weight(s) {missing} "
                           "(64-hex DerivedPin.content_sha256); the key cannot be computed")
    return {
        "schema": KEY_SCHEMA,
        "preset": preset,
        "kind": kind,
        "engine": e,
        "vllm_engine": vllm,
        "nvidia_pins": {p: _pin_identity(p) for p in sorted(spec.pins[kind])},
        "decode": {"av": RUNTIME["av"], "pillow": RUNTIME["pillow"], "numpy": RUNTIME["numpy"],
                   "arch": IMAGE_ARCH},
        "derived_weights": {n: dc[n] for n in needed},
        "modes": dict(modes or {}),
        "logic_version": LOGIC_VERSION,
        "logic_digest": cert_logic_digest(kind, logic_root),
    }


def engine_spec_hash(preset: str, kind: str, gpu: str, **kwargs: Any) -> str:
    """The certificate match key of ``kind`` under ``preset`` on ``gpu`` as the code stands."""
    return sha256_hex(key_document(preset, kind, gpu, **kwargs))


def document_problems(doc: Any) -> list[str]:
    """Why ``doc`` (a record's ``provenance.cert_key``) is not a well-formed key document."""
    if not isinstance(doc, Mapping):
        return ["cert_key is not an object"]
    out = []
    if doc.get("schema") != KEY_SCHEMA:
        out.append(f"cert_key schema {doc.get('schema')!r} != {KEY_SCHEMA!r}")
    missing = [f for f in KEY_FIELDS if f not in doc]
    extra = sorted(set(doc) - set(KEY_FIELDS))
    if missing:
        out.append(f"cert_key lacks {missing}")
    if extra:
        out.append(f"cert_key has unknown fields {extra}")
    return out


def lookup_context(cfg, kinds: Iterable[str], *,
                   derived_content: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The keyword arguments :func:`matching` needs from a config: embodiment, each kind's
    output-affecting modes (``config.hashing.kind_modes``) and the derived content digests.
    Used by config validation, ``plan_app`` and ``flora plan`` alike (H7)."""
    from flora_deploy.config.hashing import kind_modes

    return {"embodiment": cfg.pipeline.embodiment,
            "modes": {k: kind_modes(cfg, k) for k in kinds},
            "derived_content": dict(derived_content) if derived_content is not None else None}


# ---------------------------------------------------------------------------------------------
# Registry


def certificate_id(body: Mapping[str, Any]) -> str:
    """Content address: ``cert_`` + sha256 of the canonical body without ``id``."""
    return "cert_" + sha256_hex({k: v for k, v in body.items() if k != "id"})[:32]


def load_certificates(cert_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every well-formed certificate in ``cert_dir`` whose id is its content address."""
    d = Path(cert_dir) if cert_dir is not None else CERT_DIR
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(body, dict) and body.get("schema") == CERT_SCHEMA
                and body.get("verdict") in VERDICTS
                and _STAMP.match(str(body.get("generated_at") or ""))
                and isinstance(body.get("covered_pools"), list)
                and body.get("id") == certificate_id(body)):
            out.append(body)
    return out


def policy_problems(cert: Mapping[str, Any]) -> list[str]:
    """Why a certificate does not meet :data:`POLICY`: its generator, its knobs, and its checks
    recomputed from its own recorded noise means (so a body edited to loosen a threshold, then
    re-addressed, still fails here). Empty when it meets it."""
    out = []
    gen = cert.get("generator") or {}
    if gen.get("name") != GENERATOR or gen.get("version") not in GENERATOR_VERSIONS:
        out.append(f"unknown generator {gen.get('name')!r} {gen.get('version')!r}")
    th = cert.get("thresholds") or {}
    try:
        tol = float(th["tolerance"])
        slack = {k: float(v) for k, v in (th.get("slack") or {}).items()}
        out += threshold_problems(tol, th["min_samples"], th["min_noise_samples"],
                                  th["min_coverage"], slack)
    except (KeyError, TypeError, ValueError):
        return [*out, "thresholds missing or malformed"]
    if cert.get("verdict") != "certified":
        return out
    metrics = cert.get("metrics") or {}
    noise = metrics.get("noise") or {}
    if int(noise.get("n") or 0) < int(th["min_noise_samples"]):
        out.append("too few noise samples")
    if float(metrics.get("coverage") or 0) < float(th["min_coverage"]):
        out.append("incomplete coverage")
    cand = metrics.get("candidate") or {}
    if not cand or any(int(b.get("n") or 0) < int(th["min_samples"]) for b in cand.values()):
        out.append("too few samples")
    if set(cand) != set(cert.get("covered_pools") or ()):
        out.append("covered_pools differ from the measured classes")
    means = noise.get("means") or {}
    for pool, block in cand.items():
        for m, v in (block.get("means") or {}).items():
            if v is None or means.get(m) is None:
                out.append(f"{pool} {m}: not evaluable")
                continue
            t = metric_threshold(m, means[m], tol, slack.get(m, 0.0))
            if float(v) > t + 1e-12:
                out.append(f"{pool} {m}: {v} > {t}")
    return out


def _covers(cert: Mapping[str, Any], embodiment: str | None) -> bool:
    return embodiment is not None and embodiment in (cert.get("covered_pools") or ())


def matching(preset: str, pools: Mapping[str, Any], *,
             embodiment: str | None = None,
             modes: Mapping[str, Mapping[str, Any]] | None = None,
             derived_content: Mapping[str, str] | None = None,
             dtbias_fp32: bool = False,
             engines: Mapping[str, Any] | None = None,
             cert_dir: Path | None = None, image_fingerprint=None,
             logic_root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Shard kind -> the certificates (any verdict) that apply to its pool for ``embodiment``,
    deciding one first: newest ``generated_at``, then ``fail`` before ``provisional`` before
    ``certified``. ``embodiment=None``, or no content digest for a derived weight the kind
    needs, matches nothing (fail closed)."""
    certs = load_certificates(cert_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    for kind, pool in pools.items():
        cands = [c for c in certs if c.get("preset") == preset and c.get("kind") == kind
                 and c.get("gpu") == pool.gpu and _covers(c, embodiment)]
        if not cands:
            continue
        try:
            key = engine_spec_hash(preset, kind, pool.gpu, modes=(modes or {}).get(kind),
                                   derived_content=derived_content, dtbias_fp32=dtbias_fp32,
                                   engine=(engines or {}).get(kind),
                                   image_fingerprint=image_fingerprint, logic_root=logic_root)
        except CertKeyError:
            continue
        ok = [c for c in cands if c.get("engine_spec_hash") == key
              and (c["verdict"] == "fail" or not policy_problems(c))]
        if ok:
            out[kind] = sorted(ok, key=lambda c: (c["generated_at"], VERDICTS.index(c["verdict"]),
                                                  c["id"]), reverse=True)
    return out


def status(preset: str, pools: Mapping[str, Any], **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Per shard kind of ``pools``: ``{"gpu", "status", "certificate"}`` where status is
    ``reference`` (``exact``), ``certified``, ``provisional``, ``fail``, ``none``, or
    ``unverified_weights`` (the kind needs derived-weight content digests the caller lacks)."""
    if preset == "exact":
        return {k: {"gpu": p.gpu, "status": "reference", "certificate": None}
                for k, p in pools.items()}
    found = matching(preset, pools, **kwargs)
    dc = kwargs.get("derived_content") or {}
    out = {}
    for k, p in pools.items():
        best = (found.get(k) or [None])[0]
        st = best["verdict"] if best else "none"
        if best is None and any(n not in dc for n in needed_derived(preset, k)):
            st = "unverified_weights"
        out[k] = {"gpu": p.gpu, "status": st, "certificate": best["id"] if best else None}
    return out
