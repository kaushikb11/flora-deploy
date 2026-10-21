"""Logic digest (PLAN.md D11): the framework source that can change a job's outputs, hashed.

``LOGIC_VERSION`` (``flora_deploy/__init__.py``) is a human label. It is still hashed, but a fix in
our own output-path code no longer depends on someone remembering to bump it: the config hash
also covers a sha256 over the source of every module listed here, so editing any of them changes
the hash and ``flora resume`` refuses to mix pre-fix and post-fix shards (review A1).

Every module under the classified directories (``CLASSIFIED_DIRS``) must appear in exactly one of
``OUTPUT_MODULES`` (by group) or ``NON_OUTPUT_MODULES``; ``tests/unit/test_config_logic.py``
fails on a new, unclassified module. Groups keep the digest per shard kind minimal (review A5):
a change to the Rewarder stage does not invalidate finished ``segcap`` shards, and vice versa.

When in doubt, classify a module as output-affecting: a spurious hash change costs a re-run, a
missing one silently mixes results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import flora_deploy

LOGIC_SCHEMA = "flora-logic-digest/v1"

# Paths are relative to the ``flora_deploy`` package directory.
OUTPUT_MODULES: dict[str, tuple[str, ...]] = {
    # Every shard kind: record schema and ids, the index/decode/cut pass and its envelope
    # decisions, dataset readers (task text and camera mapping), NVIDIA code loading, and the
    # hash itself.
    "common": (
        "records.py",
        "jobs/types.py",
        "media/__init__.py",
        "media/types.py",
        "media/index.py",
        "media/rangedecode.py",
        "media/cuts.py",
        "media/sources/__init__.py",
        "media/sources/base.py",
        "media/sources/lerobot_v21.py",
        "media/sources/lerobot_v3.py",
        "media/sources/mp4_dir.py",
        "nv/__init__.py",
        "nv/linker.py",
        "config/hashing.py",
        "config/logic.py",
    ),
    # NVIDIA's HF pipeline (exact segcap) and its output-identical runtime patches (D17).
    "nvidia-hf": (
        "nv/session.py",
        "nv/segment_glue.py",
        "nv/budget.py",
        "nv/capture.py",
        "accel/__init__.py",
        "accel/captioner_load.py",
        "accel/framecache.py",
        "accel/geometry.py",
        "accel/patches.py",
        "accel/pngmem.py",
        "accel/registry.py",
        "accel/state.py",
        "accel/transport.py",
        "stages/__init__.py",
        "stages/reference.py",
        "modal_app/reference_app.py",
    ),
    # vLLM TempSeg / Captioner (balanced): NVIDIA's input builders, captured and wrapped for vLLM
    # (Gate C), on the same patch layer. No worker exists yet, so this is deliberately broad.
    "vllm-video": (
        "nv/session.py",
        "nv/segment_glue.py",
        "nv/budget.py",
        "nv/capture.py",
        "accel/__init__.py",
        "accel/captioner_load.py",
        "accel/framecache.py",
        "accel/geometry.py",
        "accel/patches.py",
        "accel/pngmem.py",
        "accel/registry.py",
        "accel/state.py",
        "accel/transport.py",
        "stages/__init__.py",
        "stages/reference.py",
    ),
    # NVIDIA's offline vLLM Rewarder recipe.
    "reward": (
        "stages/__init__.py",
        "stages/reward.py",
        "modal_app/rewarder_app.py",
    ),
}

# Logic group of each shard kind (on top of ``common``).
KIND_GROUPS: dict[str, str] = {
    "segcap": "nvidia-hf",
    "segment": "vllm-video",
    "caption": "vllm-video",
    "reward.terminal": "reward",
    "reward.dense": "reward",
    "reward.rubric": "reward",
}

# Reviewed and judged not to change outputs (or hashed by value elsewhere in the document).
NON_OUTPUT_MODULES: tuple[str, ...] = (
    "__init__.py",  # version strings; LOGIC_VERSION is hashed as its own field
    "layout.py",  # volume names and mount paths
    "workers.py",  # orchestrator <-> worker contract (types only)
    "media/synth.py",  # synthetic test media
    "config/__init__.py",
    "config/pins.py",  # hashed by value (nvidia_pins, decode)
    "config/presets.py",  # hashed by value (engine identity); pools are deliberately unhashed
    "config/schema.py",  # hashed by value (the validated fields)
)

# Directories whose every module must be classified above (the output path lives here).
CLASSIFIED_DIRS: tuple[str, ...] = ("", "nv", "accel", "stages", "media", "media/sources",
                                    "config")


def package_root() -> Path:
    return Path(flora_deploy.__file__).resolve().parent


def modules_for(kind: str | None) -> tuple[str, ...]:
    """Sorted module paths whose source is hashed for ``kind`` (``None``: ``common`` only)."""
    mods = set(OUTPUT_MODULES["common"])
    if kind is not None:
        if kind not in KIND_GROUPS:
            raise KeyError(f"shard kind {kind!r} has no logic group (config/logic.py)")
        mods |= set(OUTPUT_MODULES[KIND_GROUPS[kind]])
    return tuple(sorted(mods))


def source_digest(paths: tuple[str, ...], root: Path | None = None) -> str:
    """sha256 over ``(relpath, len, bytes)`` of every path, in sorted order. A listed module that
    is missing is an error (the list is stale)."""
    base = root or package_root()
    h = hashlib.sha256(LOGIC_SCHEMA.encode())
    for rel in sorted(paths):
        p = base / rel
        if not p.is_file():
            raise FileNotFoundError(f"logic module {rel} not found under {base} "
                                    "(update config/logic.py)")
        data = p.read_bytes()
        h.update(f"\0{rel}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def logic_digest(kind: str | None = None, root: Path | None = None) -> str:
    """Digest of the ``common`` modules, plus ``kind``'s group when given."""
    return source_digest(modules_for(kind), root)
