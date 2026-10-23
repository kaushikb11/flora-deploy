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
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from flora_deploy.config.hashing import MissingReceiptError
from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

__all__ = ["KNOWN_RECEIPTS", "DerivedContent", "MissingReceiptError", "ReceiptPinMismatch",
           "UnpinnedDerivedWeight", "check_overrides", "check_pinned", "derived_content",
           "load_certified", "needed", "needed_by_kind", "pin_key", "read_volume_receipt",
           "record", "remember_verified", "require_pinned", "resolve", "set_derived_text",
           "unpinned", "unpinned_message", "verified_content"]

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


class UnpinnedDerivedWeight(ValueError):
    """A derived weight the config needs has no pinned content digest yet (``pins.DERIVED_PINS``
    ``content_sha256`` is ``None``): the workers would refuse it at start, on a billed GPU, so
    plan / deploy / run refuse it first (review 10 M1)."""


# Where each derived weight's build, identity check and pinning are described.
DERIVED_DOCS = {"tempseg-merged": "docs/tempseg-merge.md",
                "captioner-bf16": "docs/captioner-bf16.md"}
_CHECKS = {"tempseg-merged": "its equivalence check (`modal run src/flora_deploy/modal_app/"
                             "convert_app.py::tempseg --step equivalence`)",
           "captioner-bf16": "its identity proof"}


def unpinned(names: Iterable[str]) -> list[str]:
    """Of ``names`` (receipt names), those without a pinned content digest, sorted."""
    out = []
    for n in sorted(set(names)):
        pin = DERIVED_PINS_BY_KEY.get(pin_key(n))
        if pin is None or pin.content_sha256 is None:
            out.append(n)
    return out


def unpinned_message(name: str, *, kinds: Iterable[str] = (), preset: str | None = None
                     ) -> str:
    """Why ``name`` is refused and the exact steps that fix it (one text everywhere: plan,
    deploy, doctor, the worker and the runbook)."""
    key = pin_key(name)
    pin = DERIVED_PINS_BY_KEY.get(key)
    ver = f" (converter v{pin.converter_version})" if pin is not None else ""
    who = " / ".join(f"`{k}`" for k in kinds)
    head = (f"{preset or 'this preset'} {who} needs" if who else "this config needs")
    doc = DERIVED_DOCS.get(key, "docs/")
    check = _CHECKS.get(key, "its identity check")
    return (f"{head} derived weight `{key}`{ver}, whose content digest is not pinned yet "
            f"(pins.DERIVED_PINS[{key!r}].content_sha256 is None), so its workers would refuse "
            f"it at start on a billed GPU. To fix: build it (`flora prepare`), run {check} "
            f"({doc}), then pin the content digest that `flora prepare --show-receipt` prints "
            f"in src/flora_deploy/config/pins.py and redeploy. Until then use `preset: exact`.")


def require_pinned(names: Iterable[str], *, kinds_of: Mapping[str, Iterable[str]] | None = None,
                   preset: str | None = None) -> None:
    """Raise :class:`UnpinnedDerivedWeight` naming every unpinned one of ``names``."""
    bad = unpinned(names)
    if bad:
        raise UnpinnedDerivedWeight(" ".join(
            unpinned_message(n, kinds=(kinds_of or {}).get(n, ()), preset=preset) for n in bad))


def needed_by_kind(cfg: Any) -> dict[str, list[str]]:
    """Receipt name -> the config's shard kinds that need it."""
    from flora_deploy.config.presets import get_preset, shard_kinds_for

    preset = get_preset(cfg.preset)
    out: dict[str, list[str]] = {}
    for k in shard_kinds_for(preset, cfg.pipeline.stages):
        for r in preset.derived_weights.get(k, ()):
            out.setdefault(r, []).append(k)
    return out


def resolve(cfg: Any, overrides: Mapping[str, str] | None = None, *,
            fake_models: bool = False) -> dict[str, str]:
    """Receipts for ``cfg``: ``weights.derived`` from the config, then ``overrides`` on top.
    Raises ``MissingReceiptError`` naming what to run when one the config needs is missing,
    except with ``fake_models``: fake workers load no weights, so a missing one is simply left
    out, exactly as ``factory.plan_app(fake_models=True)`` plans it (same config hash)."""
    recorded = dict(cfg.weights.derived)
    over = {k: v for k, v in (overrides or {}).items() if v}
    check_overrides(over)
    out = {**recorded, **over}
    missing = [n for n in needed(cfg) if not out.get(n)]
    if missing and fake_models:
        return out
    if missing:
        raise MissingReceiptError(
            f"preset {cfg.preset!r} with stages {list(cfg.pipeline.stages)} needs derived "
            f"weight(s) {', '.join(missing)}, and the config records no receipt for "
            f"{'it' if len(missing) == 1 else 'them'} under {YAML_PATH}. Run `flora prepare` "
            "(it builds them in your Modal Volume and records their receipts in flora.yaml), "
            "or pass --receipt NAME=<sha256>")
    return out


# Receipt identities (sha256 of the ``derived_receipt.json`` file) whose content digest was read
# from the receipt itself and equals the pin: the builds the pins were proven with
# (``pins.DERIVED_PINS`` comments, docs/captioner-bf16.md, docs/tempseg-merge.md).
KNOWN_RECEIPTS: dict[str, str] = {
    "0e91e467c7ffdb636573e779672d11b203571f7869ca94317f1005eb700d9c65":
        "d4c1bea5dbd3e49746a81279df2f27b904fdd2f4a71322df266201c72de0327e",  # captioner-bf16
    "9cb1f755778da2716f190b40d71a401567061bee697c2b6b34073052ea4ae678":
        "6356ab8c1499096512e1aa2a3aff4bfe4fcce9d67cce78a57e9a8484ef579303",  # tempseg-merged
}

# Receipts verified on this machine (``flora prepare``, or a deploy that read the receipt from
# the Volume): receipt sha256 -> content digest. A cache, never a source of truth: an entry only
# says that this receipt file was read and its content digest computed.
RECEIPT_CACHE_ENV = "FLORA_RECEIPT_CACHE"
_CACHE_DEFAULT = "~/.cache/flora-deploy/verified-receipts.json"


def _cache_path() -> Path:
    return Path(os.environ.get(RECEIPT_CACHE_ENV) or _CACHE_DEFAULT).expanduser()


def _read_cache() -> dict[str, str]:
    import json

    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)} \
        if isinstance(data, dict) else {}


def remember_verified(receipt_sha256: str, content_sha256: str) -> None:
    """Record that the receipt file ``receipt_sha256`` was read and has ``content_sha256``
    (best effort: an unwritable cache only means it is read again next time)."""
    import json

    data = _read_cache()
    if data.get(receipt_sha256) == content_sha256:
        return
    data[receipt_sha256] = content_sha256
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass


def verified_content(receipt_sha256: str, name: str | None = None) -> str | None:
    """The content digest of the receipt file ``receipt_sha256`` (of derived weight ``name``),
    if it is known (a pinned build, :data:`KNOWN_RECEIPTS`) or was verified on this machine;
    else ``None``."""
    return KNOWN_RECEIPTS.get(receipt_sha256) or _read_cache().get(receipt_sha256)


class DerivedContent(dict):
    """:func:`derived_content`'s result: name -> pinned content digest, for the receipts that
    count. ``unchecked``: name -> why a needed derived weight does not count (shown as the
    reason its certificate is "not checked")."""

    def __init__(self, *args: Any, unchecked: Mapping[str, str] | None = None, **kw: Any):
        super().__init__(*args, **kw)
        self.unchecked: dict[str, str] = dict(unchecked or {})


def derived_content(cfg: Any, overrides: Mapping[str, str] | None = None, *,
                    read_receipt: Any = None) -> DerivedContent:
    """Content digests of the derived weights ``cfg`` needs, for certificate matching
    (``validate.certkey``): for each needed name whose receipt (``weights.derived`` plus
    ``overrides``) is known to have the content its ``pins.DERIVED_PINS`` entry pins.

    A receipt identity alone proves nothing (any 64 hex digits parse), so a receipt counts only
    when its content digest is known: a pinned build (:data:`KNOWN_RECEIPTS`), one ``flora
    prepare`` verified on this machine (:func:`remember_verified`), or, with ``read_receipt``
    (name -> the receipt file's bytes, or ``None``; e.g. from the ``flora-weights`` Volume),
    one read now whose sha256 is the recorded identity. Fail closed: a name left out matches no
    certificate, and ``.unchecked`` says why (the workers check the content again at load,
    ``accel/captioner_load.py``)."""
    import hashlib
    import json

    recorded = {**dict(cfg.weights.derived),
                **{k: v for k, v in (overrides or {}).items() if v}}
    out: dict[str, str] = {}
    why: dict[str, str] = {}
    for name in needed(cfg):
        pin = DERIVED_PINS_BY_KEY.get(pin_key(name))
        sha = recorded.get(name)
        if not sha:
            why[name] = f"no {name} receipt recorded (run `flora prepare`)"
            continue
        if pin is None or not pin.content_sha256:
            why[name] = f"{pin_key(name)} has no pinned content digest yet"
            continue
        content = verified_content(sha, name)
        if content is None and read_receipt is not None:
            try:
                raw = read_receipt(name)
            except Exception as e:  # noqa: BLE001 - reported as the reason
                why[name] = f"could not read the {name} receipt ({type(e).__name__}: {e})"
                continue
            if raw is None:
                why[name] = f"no {name} receipt on the Volume (run `flora prepare`)"
                continue
            if hashlib.sha256(raw).hexdigest() != sha:
                why[name] = (f"the {name} receipt on the Volume is not {sha[:12]}..., the one "
                             "the config records (run `flora prepare`)")
                continue
            try:
                content = check_pinned(name, json.loads(raw))
            except (ReceiptPinMismatch, ValueError) as e:
                why[name] = str(e)
                continue
            remember_verified(sha, content)
        if content is None:
            why[name] = (f"{name} receipt {sha[:12]}... is not one `flora prepare` verified on "
                         "this machine, so its content is unchecked (run `flora prepare`)")
        elif content != pin.content_sha256:
            why[name] = (f"receipt {sha[:12]}... for {name} has content {content[:12]}..., not "
                         f"the pinned {pin.content_sha256[:12]}...")
        else:
            out[name] = content
    return DerivedContent(out, unchecked=why)


def read_volume_receipt(name: str) -> bytes | None:
    """The ``derived_receipt.json`` of derived weight ``name`` on the ``flora-weights`` Volume,
    read through the Modal client (no container); ``None`` when it does not exist. A
    ``read_receipt`` for :func:`derived_content` on paths that contact Modal anyway."""
    import modal

    from flora_deploy.layout import WEIGHTS_VOLUME
    from flora_deploy.prep.derived import DERIVED_BUILDS

    vol = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
    try:
        return b"".join(vol.read_file(DERIVED_BUILDS[name].receipt_path()))
    except FileNotFoundError:
        return None


def load_certified(path: str | os.PathLike[str], *, allow_uncertified: bool = False,
                   overrides: Mapping[str, str] | None = None,
                   read_receipt: Any = None) -> tuple[Any, DerivedContent]:
    """``(config, derived content digests)``: ``path`` validated with the content digests of its
    own derived weights (:func:`derived_content`), so the certificate check can match a kind
    that needs derived weights. What ``flora deploy`` / ``run`` / ``plan`` and ``sdk.submit``
    load a config with. ``read_receipt``: as for :func:`derived_content`."""
    from flora_deploy.config.schema import loads_yaml, parse_config

    p = Path(path)
    data = loads_yaml(p.read_text(encoding="utf-8"), source=str(p))
    base = p.resolve().parent  # a relative input.path resolves against it
    first = parse_config(data, source=str(p), allow_uncertified=True, base_dir=base)
    content = derived_content(first, overrides, read_receipt=read_receipt)
    if allow_uncertified:
        return first, content
    return parse_config(data, source=str(p), allow_uncertified=False,
                        derived_content=content, base_dir=base), content


def check_overrides(overrides: Mapping[str, str]) -> None:
    """``--receipt NAME=HASH`` / ``receipts=``: the same rules as ``weights.derived`` (a known
    receipt name, a 64-hex sha256), so a typo is refused instead of hashed into a config
    (review 8b L1). Raises ``ValueError``."""
    from flora_deploy.config.schema import _SHA256, DERIVED_WEIGHT_NAMES

    unknown = sorted(set(overrides) - set(DERIVED_WEIGHT_NAMES))
    if unknown:
        raise ValueError(f"--receipt: unknown derived weight(s) {unknown}; known: "
                         f"{', '.join(DERIVED_WEIGHT_NAMES)}")
    bad = sorted(f"{k}={v}" for k, v in overrides.items() if not _SHA256.match(v))
    if bad:
        raise ValueError(f"--receipt {', '.join(bad)}: each receipt is the 64-hex sha256 of a "
                         "derived_receipt.json (`flora prepare --show-receipt` prints them)")


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
