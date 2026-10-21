"""Logic digest (PLAN.md D11): the framework source that can change a job's outputs, hashed.

``LOGIC_VERSION`` (``flora_deploy/__init__.py``) is a human label. It is still hashed, but a fix in
our own output-path code no longer depends on someone remembering to bump it: the config hash
also covers a sha256 over the source of every module listed here, so editing any of them changes
the hash and ``flora resume`` refuses to mix pre-fix and post-fix shards (review A1).

**Every** module of the package, in every directory, must appear in exactly one of
``OUTPUT_MODULES`` (by group) or ``NON_OUTPUT_MODULES``; ``tests/unit/test_config_logic.py``
walks the whole package and fails on a new, unclassified module (review 4 M2: a fixed list of
directories let ``workers_impl/`` and ``jobs/indexing.py`` escape). Groups keep the digest per
shard kind minimal (review A5): a change to the Rewarder stage does not invalidate finished
``segcap`` shards, and vice versa.

When in doubt, classify a module as output-affecting: a spurious hash change costs a re-run, a
missing one silently mixes results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import flora_deploy

LOGIC_SCHEMA = "flora-logic-digest/v1"

# The NVIDIA HF / vLLM video pipelines share NVIDIA's input builders and the patch layer.
_VIDEO_PATH: tuple[str, ...] = (
    "nv/session.py",
    "nv/segment_glue.py",
    "nv/budget.py",
    "nv/capture.py",
    "accel/__init__.py",
    "accel/captioner_load.py",
    "accel/determinism.py",  # opt-in kernel selection (D-1) changes outputs
    "accel/framecache.py",
    "accel/geometry.py",
    "accel/gpu_thread.py",  # persistent generate thread (Perf-1); conservative
    "accel/patches.py",
    "accel/pngmem.py",
    "accel/registry.py",
    "accel/state.py",
    "accel/transport.py",
    "prep/convert_captioner.py",  # nvidia_dtype_policy: read at Captioner load
    "observe/events.py",  # NVIDIA's event lines are parsed into the stage's results
    "observe/stdout.py",  # routes NVIDIA's printed lines per episode
    "observe/timers.py",  # wraps NVIDIA's generate (the GPU lock is installed through it)
    "observe/locks.py",
    "stages/__init__.py",
    "stages/reference.py",
)

# Paths are relative to the ``flora_deploy`` package directory.
OUTPUT_MODULES: dict[str, tuple[str, ...]] = {
    # Every shard kind: record schema and ids, the index/decode/cut pass and its envelope
    # decisions (jobs/indexing.py: episode ranges, task text, instruction source, envelope
    # verdict, which the workers read back from the index receipts), dataset readers (task text
    # and camera mapping), NVIDIA code loading, and the hash itself.
    "common": (
        "records.py",
        "jobs/types.py",
        "jobs/indexing.py",
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
        *_VIDEO_PATH,
        "modal_app/reference_app.py",
        "modal_app/warm.py",  # build_linked_layout: which weight files NVIDIA's loader sees
        "workers_impl/segcap.py",  # segcap records, certification, provenance (P1a-5s)
    ),
    # vLLM TempSeg / Captioner (balanced): NVIDIA's input builders, captured and wrapped for vLLM
    # (Gate C), on the same patch layer. No worker exists yet, so this is deliberately broad.
    "vllm-video": _VIDEO_PATH,
    # NVIDIA's offline vLLM Rewarder recipe, and the worker that feeds it: camera order of the
    # collage, which file/cut is read, the task fallback, ``certified``, the settings it applies
    # (``workers_app.reward_settings_from_env``), and the job's episode order, which fixes
    # NVIDIA's terminal batches (``stages.reward`` groups by planned position, review 4 M3).
    "reward": (
        "stages/__init__.py",
        "stages/reward.py",
        "workers_impl/__init__.py",
        "workers_impl/reward.py",
        "jobs/manifest.py",
        "modal_app/rewarder_app.py",
        "modal_app/workers_app.py",
        "modal_app/warm.py",  # build_linked_layout: which weight files the engine loads
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
    "accel/identity.py",  # compares two runs' logged inputs; never on the inference path
    # cli: argument parsing; every output-affecting choice goes through the (hashed) config
    "cli/__init__.py",
    "cli/_app.py",
    "cli/deploy.py",
    "cli/index.py",
    "cli/init.py",
    "cli/plan.py",
    "cli/prepare.py",
    "cli/results.py",
    "config/__init__.py",
    "config/pins.py",  # hashed by value (nvidia_pins, decode)
    "config/presets.py",  # hashed by value (engine identity); pools are deliberately unhashed
    "config/schema.py",  # hashed by value (the validated fields)
    # export: rebuilt from the parts on every export, never mixed into stored results
    "export/__init__.py",
    "export/compose.py",
    "export/results.py",
    # jobs: orchestration (which shard runs where, when, how often), not what a shard computes
    "jobs/__init__.py",
    "jobs/attempts.py",
    "jobs/budget.py",
    "jobs/deps.py",
    "jobs/orchestrator.py",
    "jobs/progress.py",
    "jobs/reconciler.py",
    "jobs/settings.py",
    "jobs/spawner.py",
    "jobs/store.py",
    "jobs/stores/__init__.py",
    "jobs/stores/bucket.py",
    "jobs/stores/local.py",
    "jobs/stores/volume.py",
    "media/synth.py",  # synthetic test media
    # modal_app: deployment and labs. Images are hashed by value (image_fingerprint); the modes
    # the factory forwards are config fields, hashed by value (config/hashing.kind_modes).
    "modal_app/__init__.py",
    "modal_app/canary_app.py",
    "modal_app/coldstart_lab.py",
    "modal_app/convert_app.py",  # derived weights: hashed by receipt
    "modal_app/determinism_lab.py",
    "modal_app/emit.py",
    "modal_app/factory.py",
    "modal_app/gatec_app.py",
    "modal_app/golden_app.py",
    "modal_app/images.py",
    "modal_app/index_app.py",  # runs jobs/indexing.py (hashed)
    "modal_app/orchestrator_app.py",
    "modal_app/perf_lab.py",
    "modal_app/prep_app.py",
    "observe/__init__.py",
    "observe/aggregate.py",  # statistics over events
    "observe/sampler.py",  # resource sampling
    "planner/__init__.py",
    "planner/options.py",
    "planner/rates.py",
    "prep/__init__.py",
    "prep/download.py",  # downloads are verified against the pins
    "prep/merge_tempseg.py",  # derived weights: hashed by receipt
    "prep/run.py",
    "prep/verify.py",  # verification only
    "validate/__init__.py",
    "validate/gatec.py",
    "validate/gatec_driver.py",
    "validate/golden.py",  # golden build; stream_receipt digests are integrity data
)


def package_root() -> Path:
    return Path(flora_deploy.__file__).resolve().parent


def package_modules(root: Path | None = None) -> set[str]:
    """Every ``.py`` module of the package, in every directory (relative posix paths)."""
    base = root or package_root()
    return {p.relative_to(base).as_posix() for p in base.rglob("*.py")
            if "__pycache__" not in p.parts}


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
