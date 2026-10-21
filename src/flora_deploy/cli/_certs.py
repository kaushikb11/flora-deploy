"""Gate certificate status per shard kind, with the reason: the one place the CLI asks
``validate.certkey`` (``flora deploy``, ``flora plan``, ``flora doctor``).

``certkey.status`` is called here and nowhere else in the CLI, so a change to its signature is a
change to this module only. Each entry is ``{"gpu", "status", "certificate", "why"}`` where
status is ``reference`` (``exact``), ``certified``, ``provisional``, ``fail`` or ``none``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

STATUSES = ("reference", "certified", "provisional", "fail", "none")


def _reasons(cert_id: str | None) -> list[str]:
    if not cert_id:
        return []
    from flora_deploy.validate.certkey import load_certificates

    for c in load_certificates():
        if c.get("id") == cert_id:
            return [str(r) for r in c.get("reasons") or ()]
    return []


def _why_none(preset: str, kind: str, gpu: str) -> str:
    from flora_deploy.validate.certkey import load_certificates

    near = [c for c in load_certificates()
            if c.get("preset") == preset and c.get("kind") == kind and c.get("gpu") == gpu]
    if not near:
        return f"no certificate for {preset} {kind} on {gpu}"
    return (f"{len(near)} certificate(s) for {preset} {kind} on {gpu}, none for this stack "
            "(engine, image, NVIDIA pins, run modes, logic or derived weights changed)")


def certificate_report(cfg: Any, pools: Mapping[str, str | None], *,
                       derived_receipts: Mapping[str, str] | None = None
                       ) -> dict[str, dict[str, Any]]:
    """Shard kind -> ``{"gpu", "status", "certificate", "why"}`` for ``pools`` (kind -> GPU)
    under ``cfg``'s preset and run modes (``config.hashing.kind_modes``)."""
    from flora_deploy.config.hashing import kind_modes
    from flora_deploy.validate.certkey import status

    preset = cfg.preset
    ns = {k: SimpleNamespace(gpu=g) for k, g in pools.items()}
    kw: dict[str, Any] = {"modes": {k: kind_modes(cfg, k) for k in ns}}
    if derived_receipts is not None:
        kw["derived_receipts"] = dict(derived_receipts)
    raw = status(preset, ns, **kw)
    out: dict[str, dict[str, Any]] = {}
    for kind, r in raw.items():
        st, cid, gpu = r.get("status", "none"), r.get("certificate"), r.get("gpu")
        if st == "reference":
            why = "exact is NVIDIA's reference stack: no certificate needed"
        elif st == "none":
            why = _why_none(preset, kind, str(gpu))
        else:
            reasons = _reasons(cid)
            why = f"{cid}" + (f": {', '.join(reasons)}" if reasons else "")
            if st != "certified":
                why += " (does not unlock the preset)"
        out[kind] = {"gpu": gpu, "status": st, "certificate": cid, "why": why}
    return out
