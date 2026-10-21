"""Certificate match key and registry lookup (PLAN.md D9, D16; ``docs/certification.md``).

A Gate certificate (written by ``validate.certify``) says that one shard kind of a non-``exact``
preset, on one GPU type, stays within NVIDIA's own run-to-run noise. This module decides **which
certificate applies to a pool**, so it decides a record's ``certified`` flag and
``provenance.certificate``: it is output-affecting (``config/logic.py``, ``common``), even though
the certificates themselves are metrics only.

Match rule. A certificate applies to (preset, shard kind, pool) only when all of these hold:

- ``preset``, ``kind`` and ``gpu`` are equal (``"H100!"`` and ``"H100"`` are different GPUs);
- its ``engine_spec_hash`` equals :func:`engine_spec_hash` of the stack that would run now. That
  is the stage hash of the kind (``config.hashing``) without the per-job part (input data,
  embodiment, instruction, envelope): engine identity and image fingerprint, NVIDIA pins, the
  kind's output-affecting modes, the logic version and the kind's logic digest. Any engine,
  image, pin, mode or logic change therefore invalidates the certificate;
- when the caller knows the derived-weight receipts, the certificate's receipts (if it recorded
  any) equal them;
- its ``id`` is the content address of its body (a hand-edited file never matches).

Only ``verdict == "certified"`` unlocks a preset. ``provisional`` certificates (no noise run, too
few samples, or no L0) are reported but never unlock one: such a stack still needs
``allow_uncertified``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

CERT_SCHEMA = "flora-certificate/v1"
KEY_SCHEMA = "flora-cert-key/v1"
CERT_DIR = Path(__file__).resolve().parent.parent / "config" / "certificates"
VERDICTS = ("certified", "provisional", "fail")


def canonical(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def sha256_hex(doc: Any) -> str:
    return hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()


def _default_fingerprint(name: str) -> str:
    from flora_deploy.modal_app.images import image_fingerprint

    return image_fingerprint(name)


def engine_spec_document(preset: str, kind: str, *, modes: Mapping[str, Any] | None = None,
                         engine=None, image_fingerprint=None, logic_root: Path | None = None
                         ) -> dict[str, Any]:
    """What :func:`engine_spec_hash` hashes (see the module docstring)."""
    from flora_deploy import LOGIC_VERSION
    from flora_deploy.config.hashing import _pin_identity
    from flora_deploy.config.logic import logic_digest
    from flora_deploy.config.presets import get_preset

    spec = get_preset(preset)
    eng = engine if engine is not None else spec.engines[kind]
    e = asdict(eng)
    e["patches"] = list(eng.patches)
    e["image_fingerprint"] = (image_fingerprint or _default_fingerprint)(eng.image)
    return {
        "schema": KEY_SCHEMA,
        "preset": preset,
        "kind": kind,
        "engine": e,
        "nvidia_pins": {p: _pin_identity(p) for p in sorted(spec.pins[kind])},
        "modes": dict(modes or {}),
        "logic_version": LOGIC_VERSION,
        "logic_digest": logic_digest(kind, logic_root),
    }


def engine_spec_hash(preset: str, kind: str, **kwargs: Any) -> str:
    """The certificate match key of ``kind`` under ``preset`` as the code stands now."""
    return sha256_hex(engine_spec_document(preset, kind, **kwargs))


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
                and body.get("verdict") in VERDICTS and body.get("id") == certificate_id(body)):
            out.append(body)
    return out


def matching(preset: str, pools: Mapping[str, Any], *,
             modes: Mapping[str, Mapping[str, Any]] | None = None,
             engines: Mapping[str, Any] | None = None,
             derived_receipts: Mapping[str, str] | None = None,
             cert_dir: Path | None = None, image_fingerprint=None,
             logic_root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Shard kind -> the certificates (any verdict) that apply to its pool, best first
    (certified, then provisional, then fail; ties by id)."""
    certs = load_certificates(cert_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    for kind, pool in pools.items():
        cands = [c for c in certs if c.get("preset") == preset and c.get("kind") == kind
                 and c.get("gpu") == pool.gpu]
        if not cands:
            continue
        key = engine_spec_hash(preset, kind, modes=(modes or {}).get(kind),
                               engine=(engines or {}).get(kind),
                               image_fingerprint=image_fingerprint, logic_root=logic_root)
        ok = []
        for c in cands:
            if c.get("engine_spec_hash") != key:
                continue
            want = c.get("derived_weights") or {}
            if derived_receipts is not None and any(derived_receipts.get(n) != h
                                                    for n, h in want.items()):
                continue
            ok.append(c)
        if ok:
            out[kind] = sorted(ok, key=lambda c: (VERDICTS.index(c["verdict"]), c["id"]))
    return out


def status(preset: str, pools: Mapping[str, Any], **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Per shard kind of ``pools``: ``{"gpu", "status", "certificate"}`` where status is
    ``reference`` (``exact``), ``certified``, ``provisional``, ``fail`` or ``none``."""
    if preset == "exact":
        return {k: {"gpu": p.gpu, "status": "reference", "certificate": None}
                for k, p in pools.items()}
    found = matching(preset, pools, **kwargs)
    out = {}
    for k, p in pools.items():
        best = (found.get(k) or [None])[0]
        out[k] = {"gpu": p.gpu, "status": best["verdict"] if best else "none",
                  "certificate": best["id"] if best else None}
    return out
