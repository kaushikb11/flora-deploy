"""Gate certificate status per shard kind, with the reason: the one place the CLI asks
``validate.certkey`` (``flora deploy``, ``flora plan``, ``flora doctor``).

``certkey.status`` is called here and nowhere else in the CLI, so a change to its signature is a
change to this module only. Each entry is ``{"gpu", "status", "certificate", "why"}`` where
status is ``reference`` (``exact``), ``certified``, ``provisional``, ``fail``, ``none`` or
``unverified_weights`` (the derived weights' content digests are unknown: fail closed).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

STATUSES = ("reference", "certified", "provisional", "fail", "none", "unverified_weights")
UNLOCKS = frozenset({"reference", "certified"})


def _cert(cert_id: str | None) -> dict[str, Any]:
    if not cert_id:
        return {}
    from flora_deploy.validate.certkey import load_certificates

    for c in load_certificates():
        if c.get("id") == cert_id:
            return c
    return {}


def _why_none(preset: str, kind: str, gpu: str, embodiment: str | None) -> str:
    from flora_deploy.validate.certkey import load_certificates

    near = [c for c in load_certificates()
            if c.get("preset") == preset and c.get("kind") == kind and c.get("gpu") == gpu]
    if not near:
        return f"no certificate for {preset} {kind} on {gpu}"
    if not any(embodiment in (c.get("covered_pools") or ()) for c in near):
        return (f"{len(near)} certificate(s) for {preset} {kind} on {gpu}, none covering "
                f"embodiment {embodiment!r}")
    return (f"{len(near)} certificate(s) for {preset} {kind} on {gpu}, none for this stack "
            "(engine, image, pins, run modes, logic or derived weights changed)")


def certificate_report(cfg: Any, pools: Mapping[str, str | None], *,
                       preset: str | None = None,
                       derived_content: Mapping[str, str] | None = None
                       ) -> dict[str, dict[str, Any]]:
    """Shard kind -> ``{"gpu", "status", "certificate", "why"}`` for ``pools`` (kind -> GPU)
    under ``preset`` (default ``cfg.preset``) with ``cfg``'s embodiment and run modes and the
    derived weights' content digests (``derived_content``; without them a kind that needs
    derived weights is ``unverified_weights``)."""
    from flora_deploy.validate.certkey import lookup_context, needed_derived, status

    preset = preset or cfg.preset
    ns = {k: SimpleNamespace(gpu=g) for k, g in pools.items()}
    raw = status(preset, ns, **lookup_context(cfg, ns, derived_content=derived_content))
    embodiment = getattr(getattr(cfg, "pipeline", None), "embodiment", None)
    out: dict[str, dict[str, Any]] = {}
    for kind, r in raw.items():
        st, cid, gpu = r.get("status", "none"), r.get("certificate"), r.get("gpu")
        if st == "reference":
            why = "exact is NVIDIA's reference stack: no certificate needed"
        elif st == "unverified_weights":
            names = ", ".join(needed_derived(preset, kind))
            why = (f"no content digest for derived weight(s) {names}: run `flora prepare` "
                   "(weights.derived) so the certificate key can be computed")
        elif st == "none":
            why = _why_none(preset, kind, str(gpu), embodiment)
        else:
            c = _cert(cid)
            reasons = [str(x) for x in c.get("reasons") or ()]
            why = str(cid)
            if c.get("generated_at"):
                why += f", generated {c['generated_at']}"
            if reasons:
                why += f": {', '.join(reasons)}"
            if st != "certified":
                why += " (does not unlock the preset)"
        out[kind] = {"gpu": gpu, "status": st, "certificate": cid, "why": why}
    return out
