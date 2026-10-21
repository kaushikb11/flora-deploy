"""D11 config hash: one sha256 over everything that can change a job's outputs.

``config_hash(cfg)`` builds a canonical JSON document and hashes it. The document is returned
too, so provenance can record exactly what was hashed and a changed hash can be diffed.

Hashed (D11):
- ``LOGIC_VERSION`` (framework logic that changes outputs);
- preset, stages, embodiment pool, episode-instruction source/text, envelope policy;
- for each NVIDIA pin a used shard kind depends on: repo, revision, role, ``plan_digest``,
  ``expected`` digests (incl. NVIDIA's manifest sha256) and code digests (``config/pins.py``).
  NVIDIA's prompt pools live in ``batch_utils/`` and are bound by the code-manifest digests;
- derived-weight receipt identities (e.g. ``captioner_bf16``), passed in by the caller;
- per shard kind: engine name, versions, patches, engine args, sampling, mm_kwargs, and the
  ``image_fingerprint()`` of the image it runs on;
- decode pins (av, Pillow, numpy, arch);
- input identity: format, dataset (repo or path), revision, subdirectory, camera map.

Excluded (they don't change outputs; they go to provenance instead): GPU type and count,
container counts, max inputs, cpu/memory, budget, max_gpus, region, tags, Modal environment,
shard sizing, retries, output sink/formats, observability, ``stage_to``, project name.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from flora_deploy import LOGIC_VERSION
from flora_deploy.config.pins import PINS_BY_KEY, RUNTIME
from flora_deploy.config.presets import EngineIdentity, get_preset, shard_kinds_for

HASH_SCHEMA = "flora-config-hash/v1"
IMAGE_ARCH = "x86_64"  # every Modal image is x86_64 (PLAN.md §7)


class MissingReceiptError(ValueError):
    """A derived-weight receipt the config needs was not supplied (run ``flora prepare``)."""


@dataclass(frozen=True)
class ConfigHash:
    hash: str  # hex sha256 of ``canonical``
    canonical: str  # the exact bytes hashed (UTF-8 JSON)
    document: dict[str, Any]


def canonical_json(doc: Any) -> str:
    """Sorted keys, no whitespace, ASCII only, no NaN: the same bytes on every run."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _pin_identity(key: str) -> dict[str, Any]:
    p = PINS_BY_KEY[key]
    return {
        "repo": p.repo,
        "revision": p.revision,
        "role": p.role,
        "plan_digest": p.plan_digest,
        "expected": dict(p.expected),
        "code_digests": dict(p.code_digests),
    }


def _default_fingerprint(name: str) -> str:
    from flora_deploy.modal_app.images import image_fingerprint

    return image_fingerprint(name)


def hash_document(
    cfg,
    *,
    derived_receipts: Mapping[str, str] | None = None,
    engines: Mapping[str, EngineIdentity] | None = None,
    image_fingerprint: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """The document ``config_hash`` hashes. See the module docstring for what is in it.

    ``derived_receipts``: receipt name -> receipt identity (e.g. the sha256 of the
    ``derived_receipt.json`` written by ``prep/convert_captioner.py``). Every receipt a used
    shard kind needs must be present, or ``MissingReceiptError`` is raised.
    ``engines``: per-shard-kind engine identity overriding the preset's (for engines whose args
    are only known at deploy time). ``image_fingerprint``: injectable for tests."""
    preset = get_preset(cfg.preset)
    fingerprint = image_fingerprint or _default_fingerprint
    receipts = dict(derived_receipts or {})
    kinds = shard_kinds_for(preset, cfg.pipeline.stages)

    needed = sorted({r for k in kinds for r in preset.derived_weights.get(k, ())})
    missing = [r for r in needed if not receipts.get(r)]
    if missing:
        raise MissingReceiptError(
            f"preset {cfg.preset!r} with stages {list(cfg.pipeline.stages)} needs derived-weight "
            f"receipt(s) {missing}; run `flora prepare` and pass their receipt hashes")

    shard_kinds: dict[str, Any] = {}
    for kind in kinds:
        eng = (engines or {}).get(kind, preset.engines[kind])
        e = asdict(eng)
        e["patches"] = list(eng.patches)
        e["image_fingerprint"] = fingerprint(eng.image)
        shard_kinds[kind] = {
            "engine": e,
            "pins": sorted(preset.pins[kind]),
            "derived_weights": sorted(preset.derived_weights.get(kind, ())),
        }

    pin_keys = sorted({p for k in kinds for p in preset.pins[k]})
    inp = cfg.input
    instr = cfg.pipeline.episode_instruction
    return {
        "schema": HASH_SCHEMA,
        "logic_version": LOGIC_VERSION,
        "preset": cfg.preset,
        "stages": list(cfg.pipeline.stages),
        "embodiment": cfg.pipeline.embodiment,
        "episode_instruction": {"source": instr.source, "text": instr.text},
        "envelope": cfg.pipeline.envelope,
        "nvidia_pins": {k: _pin_identity(k) for k in pin_keys},
        "derived_weights": {r: receipts[r] for r in needed},
        "shard_kinds": shard_kinds,
        "decode": {"av": RUNTIME["av"], "pillow": RUNTIME["pillow"], "numpy": RUNTIME["numpy"],
                   "arch": IMAGE_ARCH},
        "input": {
            "format": inp.format,
            "dataset": inp.dataset,
            "revision": inp.revision,
            "path": inp.path if inp.repo is not None else None,
            "cameras": dict(sorted(inp.cameras.items())) if inp.cameras is not None else None,
        },
    }


def config_hash(cfg, **kwargs: Any) -> ConfigHash:
    """sha256 over ``canonical_json(hash_document(cfg, ...))``; see ``hash_document``."""
    doc = hash_document(cfg, **kwargs)
    canonical = canonical_json(doc)
    return ConfigHash(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical, doc)
