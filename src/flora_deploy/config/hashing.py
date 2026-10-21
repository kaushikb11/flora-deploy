"""D11 config hash: sha256 over everything that can change a job's outputs, per shard kind.

``config_hash(cfg)`` builds a canonical JSON document and hashes it. It returns:
- ``hash``: the **job-level** hash over the whole document (display, manifest, ``RunSettings``);
- ``stage_hashes``: one hash **per shard kind**, over the shared inputs plus that kind's own
  inputs (and the hashes of the kinds it consumes). They are meant to key finished shards, so
  that adding ``reward.dense`` to ``stages`` or bumping the vLLM lock (rewards only) does not
  invalidate finished ``segcap`` shards (review A5). The job store keys each kind's shard files
  by its stage hash (``jobs.store.JobPaths``, Fix-W1); the job hash keys the manifest;
- the document and its exact bytes, so provenance records what was hashed and a changed hash can
  be diffed.

Shared (every stage hash):
- ``LOGIC_VERSION`` (a human label) and the ``common`` logic digest (``config/logic.py``: sha256
  of the framework source on every kind's output path, so a fix there changes the hash even if
  nobody bumps the label);
- preset, embodiment pool, episode-instruction source/text, envelope policy;
- decode pins (av, Pillow, numpy, arch);
- input identity: format, dataset (repo or normalized path), revision, subdirectory, camera map,
  and the input ``content`` digest: the caller's ``input_digest`` when it has one, else for a
  **local** input (no ``repo``) :func:`local_input_digest` of the directory (relative path,
  size and sha256 of every file; review 3 A8: the path string alone does not bind the
  contents). A Hugging Face input is bound by its 40-hex ``revision``. The
  camera map is hashed as written: ``cameras: None`` and an explicit map that resolves to the
  same views hash differently, because which map ``None`` resolves to depends on the dataset's
  camera keys. That costs a re-run, never a mix.

Per shard kind:
- the stages it produces in this job; its logic-group digest (``config/logic.py``);
- ``modes``: the output-affecting run modes of that kind (review 4): ``determinism`` for the
  NVIDIA HF worker (``segcap``; D-1 "deterministic" is a different reference) and
  ``terminal_mode`` for ``reward.terminal`` (score only, NVIDIA's default, or with rationale);
- engine name, versions, patches, engine args (for vLLM video engines incl. the Gate C-required
  ``mm_processor_cache_gb=0``), sampling, mm_kwargs, and the ``image_fingerprint()`` of its image;
- for each NVIDIA pin it depends on: repo, revision, role, ``plan_digest``, ``expected`` digests
  (incl. NVIDIA's manifest sha256) and code digests (``config/pins.py``). NVIDIA's prompt pools
  live in ``batch_utils/`` and are bound by the code-manifest digests;
- its derived-weight receipt identities (e.g. ``captioner_bf16``), passed in by the caller;
- ``depends_on``: the stage hashes of the kinds whose outputs it consumes (``balanced`` caption
  consumes segment).

Excluded (they don't change outputs; they go to provenance instead): GPU type and count,
container counts, max inputs, cpu/memory, budget, max_gpus, region, tags, Modal environment,
shard sizing, retries, output sink/formats, observability, ``stage_to``, project name,
``allow_uncertified``. Shard sizing really is output-neutral for rewards too (review 5 M-B):
NVIDIA's terminal batches are taken from the job's planned episode order
(``stages.reward.run(planned=...)``) and ``jobs.manifest`` cuts terminal shards on those group
boundaries, so ``shard_target_gpu_minutes`` decides which groups share a shard, never which
episodes share a batch.

GPU type (review 4 M7). It is excluded on purpose: the model inputs are identical on every GPU
(the input-identity gates run on CPU-built tensors), so a GPU override is a cost/speed choice.
But **outputs are not identical across GPU types**: an H200 (sm90) and a B200 (sm100) run
different kernels with different numerics, just as two runs on one GPU type can differ (D-1).
Records carry the GPU in ``provenance.gpu``; compare or resume across GPU types only knowing
that results are input-identical, not output-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flora_deploy import LOGIC_VERSION
from flora_deploy.config.logic import KIND_GROUPS, logic_digest
from flora_deploy.config.pins import PINS_BY_KEY, RUNTIME
from flora_deploy.config.presets import EngineIdentity, get_preset, shard_kinds_for

HASH_SCHEMA = "flora-config-hash/v3"  # v3: per-kind ``modes``, input ``content``
IMAGE_ARCH = "x86_64"  # every Modal image is x86_64 (PLAN.md §7)

# Shard kind -> the shard kinds whose outputs it consumes (``jobs.manifest``).
KIND_DEPENDS_ON: dict[str, tuple[str, ...]] = {"caption": ("segment",)}


class MissingReceiptError(ValueError):
    """A derived-weight receipt the config needs was not supplied (run ``flora prepare``)."""


@dataclass(frozen=True)
class ConfigHash:
    hash: str  # job-level: hex sha256 of ``canonical``
    canonical: str  # the exact bytes hashed (UTF-8 JSON)
    document: dict[str, Any]
    stage_hashes: dict[str, str]  # shard kind -> its stage hash


def canonical_json(doc: Any) -> str:
    """Sorted keys, no whitespace, ASCII only, no NaN: the same bytes on every run."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _sha(doc: Any) -> str:
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


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


_CHUNK = 1 << 20


def local_input_digest(path: str | os.PathLike[str] | None) -> str | None:
    """sha256 over ``(relative path, size, sha256)`` of every regular file under a local input
    directory (review 3 A8), sorted by path; hidden files and directories (``.DS_Store``,
    ``.cache``) are skipped. ``None`` when ``path`` is unset or not a local directory here (a
    path that only exists in a container): then only the path string is hashed, as before.
    Symlinked files are read through; symlinked directories are followed."""
    if not path:
        return None
    root = Path(path)
    if not root.is_dir():
        return None
    entries = []
    seen: set[str] = set()  # real directories walked (a symlink loop is walked once)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            f = Path(dirpath) / name
            if not f.is_file():
                continue
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda fh=fh: fh.read(_CHUNK), b""):
                    h.update(chunk)
            entries.append([f.relative_to(root).as_posix(), f.stat().st_size, h.hexdigest()])
    entries.sort()
    return _sha({"schema": "flora-local-input/v1", "files": entries})


def kind_modes(cfg, kind: str) -> dict[str, Any]:
    """The output-affecting run modes of ``kind`` (hashed in its entry, read by its worker)."""
    modes: dict[str, Any] = {}
    if KIND_GROUPS.get(kind) == "nvidia-hf":
        modes["determinism"] = cfg.pipeline.determinism
    if kind == "reward.terminal":
        modes["terminal_mode"] = ("terminal_rationale" if cfg.pipeline.terminal_rationale
                                  else "terminal")
    return modes


def _default_fingerprint(name: str) -> str:
    from flora_deploy.modal_app.images import image_fingerprint

    return image_fingerprint(name)


def hash_document(
    cfg,
    *,
    derived_receipts: Mapping[str, str] | None = None,
    engines: Mapping[str, EngineIdentity] | None = None,
    image_fingerprint: Callable[[str], str] | None = None,
    logic_root: Path | None = None,
    input_digest: str | None = None,
) -> dict[str, Any]:
    """The document ``config_hash`` hashes. See the module docstring for what is in it.

    ``derived_receipts``: receipt name -> receipt identity (e.g. the sha256 of the
    ``derived_receipt.json`` written by ``prep/convert_captioner.py``). Every receipt a used
    shard kind needs must be present, or ``MissingReceiptError`` is raised.
    ``engines``: per-shard-kind engine identity overriding the preset's (for engines whose args
    are only known at deploy time). ``image_fingerprint`` and ``logic_root`` (the package
    directory whose source is digested): injectable for tests. ``input_digest``: a digest of
    the input's contents (e.g. the index summary), when the caller has one; otherwise a local
    input directory is digested here (:func:`local_input_digest`)."""
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
        pins = sorted(preset.pins[kind])
        weights = sorted(preset.derived_weights.get(kind, ()))
        shard_kinds[kind] = {
            "stages": [s for s in cfg.pipeline.stages if preset.stage_kinds[s] == kind],
            "logic_digest": logic_digest(kind, logic_root),
            "modes": kind_modes(cfg, kind),
            "engine": e,
            "nvidia_pins": {p: _pin_identity(p) for p in pins},
            "derived_weights": {r: receipts[r] for r in weights},
            "depends_on": [d for d in KIND_DEPENDS_ON.get(kind, ()) if d in kinds],
        }

    inp = cfg.input
    if input_digest is None and inp.repo is None:
        input_digest = local_input_digest(inp.path)
    instr = cfg.pipeline.episode_instruction
    shared = {
        "logic_version": LOGIC_VERSION,
        "logic_common": logic_digest(None, logic_root),
        "preset": cfg.preset,
        "embodiment": cfg.pipeline.embodiment,
        "episode_instruction": {"source": instr.source, "text": instr.text},
        "envelope": cfg.pipeline.envelope,
        "decode": {"av": RUNTIME["av"], "pillow": RUNTIME["pillow"], "numpy": RUNTIME["numpy"],
                   "arch": IMAGE_ARCH},
        "input": {
            "format": inp.format,
            "dataset": inp.dataset,
            "revision": inp.revision,
            "path": inp.subdir,
            "cameras": dict(sorted(inp.cameras.items())) if inp.cameras is not None else None,
            "content": input_digest,
        },
    }
    return {
        "schema": HASH_SCHEMA,
        "shared": shared,
        "stages": list(cfg.pipeline.stages),
        "shard_kinds": shard_kinds,
    }


def _stage_hashes(doc: Mapping[str, Any]) -> dict[str, str]:
    """Per shard kind: sha256 over the shared inputs, its own entry and its dependencies'
    stage hashes. Dependencies come first in ``shard_kinds`` (stage order)."""
    out: dict[str, str] = {}
    for kind, entry in doc["shard_kinds"].items():
        out[kind] = _sha({
            "schema": doc["schema"],
            "kind": kind,
            "shared": doc["shared"],
            "shard": entry,
            "depends_on": {d: out[d] for d in entry["depends_on"]},
        })
    return out


def config_hash(cfg, **kwargs: Any) -> ConfigHash:
    """Job hash = sha256 over ``canonical_json(hash_document(cfg, ...))``, plus the per-kind
    ``stage_hashes``; see ``hash_document``."""
    doc = hash_document(cfg, **kwargs)
    canonical = canonical_json(doc)
    return ConfigHash(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical, doc,
                      _stage_hashes(doc))


def stage_hashes(cfg, **kwargs: Any) -> dict[str, str]:
    """Shard kind -> stage hash (the key finished shards of that kind are stored under)."""
    return config_hash(cfg, **kwargs).stage_hashes
