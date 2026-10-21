"""Derived-weight receipts: where ``flora deploy``, ``flora run`` and ``sdk.submit`` get them,
and how ``flora prepare`` records them in ``flora.yaml``.

A receipt identity is the sha256 of the ``derived_receipt.json`` a conversion wrote in the
user's ``flora-weights`` Volume (``prep/convert_captioner.py``, ``prep/merge_tempseg.py``).
``flora prepare`` builds the derived weights the config's preset needs and records them::

    weights:
      derived:
        captioner_bf16: <sha256>
        tempseg_merged: <sha256>

:func:`resolve` merges those with explicit overrides (``--receipt NAME=HASH``,
``sdk.submit(receipts=...)``; an override wins) and refuses early when one the config needs is
missing. A receipt identity names one build; what ``pins.DERIVED_PINS`` pins is the build's
*content* (``prep.convert_captioner.content_digest``), which needs the receipt itself:
``flora prepare`` checks it (:func:`check_pinned`) before it records the identity, and the
runtime recomputes it at load (``accel/captioner_load.py``).

The receipts enter the config hash through ``hashing.hash_document(derived_receipts=...)``
(``derived_weights`` per shard kind), so the hash is the same whether a receipt comes from
``flora.yaml`` or from ``--receipt``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flora_deploy.config.hashing import MissingReceiptError
from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

__all__ = ["MissingReceiptError", "ReceiptPinMismatch", "check_pinned", "needed", "pin_key",
           "record", "resolve", "set_derived_text"]

YAML_PATH = "weights.derived"


class ReceiptPinMismatch(ValueError):
    """A recorded receipt is not the one ``pins.DERIVED_PINS`` pins."""


def pin_key(name: str) -> str:
    """Receipt name (``captioner_bf16``) -> its ``DERIVED_PINS`` / Volume directory name
    (``captioner-bf16``)."""
    return name.replace("_", "-")


def needed(cfg: Any) -> list[str]:
    """Receipt names the config's preset and stages need, sorted."""
    from flora_deploy.config.presets import get_preset, shard_kinds_for

    preset = get_preset(cfg.preset)
    kinds = shard_kinds_for(preset, cfg.pipeline.stages)
    return sorted({r for k in kinds for r in preset.derived_weights.get(k, ())})


def check_pinned(name: str, receipt: Mapping, *, where: str | None = None) -> str:
    """The content digest of ``receipt`` (recomputed; a stored ``content_sha256`` must agree).
    Raises :class:`ReceiptPinMismatch` when ``name`` has a pinned content digest and this is
    not it, or the receipt cannot be digested. Unpinned names (``content_sha256`` None, or no
    ``DerivedPin``) pass. ``where`` names the receipt in the message."""
    from flora_deploy.prep.convert_captioner import ContentDigestError, checked_content_digest

    label = where or f"the {name} receipt"
    try:
        content = checked_content_digest(receipt)
    except ContentDigestError as e:
        raise ReceiptPinMismatch(f"{label}: {e}") from None
    pin = DERIVED_PINS_BY_KEY.get(pin_key(name))
    if pin is None or pin.content_sha256 is None or content == pin.content_sha256:
        return content
    raise ReceiptPinMismatch(
        f"{label} has content digest {content}, but pins.DERIVED_PINS[{pin.key!r}] pins "
        f"{pin.content_sha256}. The workers load only the pinned {pin.key} content, so this "
        "build would be refused at load: its files, dtype policy, casts or sources differ from "
        f"the proven build. Rebuild it (`flora prepare --force-derived`); if it still differs, "
        f"the converter changed and needs a new identity proof (docs/{pin.key}.md)")


def resolve(cfg: Any, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Receipts for ``cfg``: ``weights.derived`` from the config, then ``overrides`` on top.
    Raises ``MissingReceiptError`` naming what to run when one the config needs is missing."""
    recorded = dict(cfg.weights.derived)
    over = {k: v for k, v in (overrides or {}).items() if v}
    out = {**recorded, **over}
    missing = [n for n in needed(cfg) if not out.get(n)]
    if missing:
        raise MissingReceiptError(
            f"preset {cfg.preset!r} with stages {list(cfg.pipeline.stages)} needs derived "
            f"weight(s) {', '.join(missing)}, and the config records no receipt for "
            f"{'it' if len(missing) == 1 else 'them'} under {YAML_PATH}. Run `flora prepare` "
            "(it builds them in your Modal Volume and records their receipts in flora.yaml), "
            "or pass --receipt NAME=<sha256>")
    return out


# --- recording in flora.yaml ------------------------------------------------------------------

_TOP_KEY = re.compile(r"^weights\s*:")
_HEADER = "# Derived weights built in your Modal Volume, by receipt sha256 (`flora prepare`)."


def _render(derived: Mapping[str, str]) -> list[str]:
    return [f"{_HEADER}\n", "weights:\n", "  derived:\n",
            *(f"    {k}: {v}\n" for k, v in sorted(derived.items()))]


def set_derived_text(text: str, receipts: Mapping[str, str], *,
                     source: str = "flora.yaml") -> str:
    """``text`` with ``weights.derived`` updated by ``receipts`` (merged into what is there).

    A targeted edit, so every other line (and its comments) is kept byte for byte: an existing
    top-level ``weights:`` block (the key line plus the indented or blank lines under it) is
    replaced; otherwise the block is appended. The result is re-parsed and must equal the old
    document with only ``weights.derived`` changed, else ``ConfigError`` (edit it by hand)."""
    from flora_deploy.config.schema import ConfigError, loads_yaml

    data = loads_yaml(text, source=source)
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level")
    weights = data.get("weights")
    if weights is None:
        weights = {}
    if not isinstance(weights, dict) or set(weights) - {"derived"} or not isinstance(
            weights.get("derived") or {}, dict):
        raise ConfigError(f"{source}: `weights` must hold only `derived: {{name: sha256}}`")
    merged = {**(weights.get("derived") or {}), **receipts}

    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if _TOP_KEY.match(ln)), None)
    if start is None:
        head = lines
        if head and not head[-1].endswith("\n"):
            head = [*head[:-1], head[-1] + "\n"]
        new = [*head, *(["\n"] if head and head[-1].strip() else []), *_render(merged)]
    else:
        end = start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
            end += 1
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        block = _render(merged)
        if start > 0 and lines[start - 1].rstrip("\n") == _HEADER:
            start -= 1  # our own header from a previous write
        new = [*lines[:start], *block, *lines[end:]]
    out = "".join(new)

    want = {**data, "weights": {"derived": merged}}
    try:
        got = loads_yaml(out, source=source)
    except ConfigError:
        got = None
    if got != want:
        rows = "".join(f"    {k}: {v}\n" for k, v in sorted(merged.items()))
        raise ConfigError(f"{source}: could not update `weights.derived` in place; add this "
                          f"top-level block by hand:\nweights:\n  derived:\n{rows}")
    return out


def record(path: str | os.PathLike[str], receipts: Mapping[str, str]) -> bool:
    """Record ``receipts`` under ``weights.derived`` in the YAML file at ``path`` (atomic
    replace). Returns False when the file already held exactly these values."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    out = set_derived_text(text, receipts, source=str(p))
    if out == text:
        return False
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(out, encoding="utf-8")
    try:
        os.chmod(tmp, p.stat().st_mode & 0o777)
    except OSError:
        pass
    os.replace(tmp, p)
    return True
