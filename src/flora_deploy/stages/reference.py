"""The ``exact`` reference stage: NVIDIA's own TempSeg -> Captioner pipeline (PLAN.md §4, §6.5).

Modal-agnostic pieces used by ``modal_app/reference_app.py``:

- ``build_offline_layout``: make NVIDIA's ``models.Session.load`` find the weights on the
  ``flora-weights`` Volume with no network access and with NVIDIA's code unmodified (see below).
- ``annotate_reference``: the minimal re-expression of NVIDIA's ``api.annotate`` /
  ``pipeline.annotate`` orchestration around a ``CachedSession``.
- ``capture_runtime_events`` / ``generation_stats``: read NVIDIA's own ``FLORA_RUNTIME`` stdout
  events (token counts, generation time) without touching NVIDIA's modules.
- ``persist_result``: write a result JSON locally, then copy it whole onto a Volume.

How offline loading works (NVIDIA ``batch_utils/models.py`` at nvidia/Flora-TempSeg@78a0e4ac):

- ``snapshot(kind)`` (``models.py:39-46``) calls
  ``snapshot_download(repo, revision=<sha>, allow_patterns=..., local_dir=MODEL_CACHE / kind)``.
  With a commit-sha revision, ``huggingface_hub`` reads the tree listing the prep job cached in
  ``<snapshot>/.cache/huggingface/trees/`` and, per file, the download metadata in
  ``.cache/huggingface/download/`` and returns the existing file without a request
  (``HF_HUB_OFFLINE=1`` makes any request raise instead). Reading that metadata takes a
  ``filelock`` *inside* ``local_dir`` (``_local_folder.read_download_metadata``), so ``local_dir``
  must be writable: a symlink to the read-only Volume fails with EROFS (measured on Modal).
  We therefore **copy** each prepared Volume snapshot (mtimes preserved, so the metadata stays
  current) to ``FLORA_MODEL_CACHE/<kind>`` on local disk. NVIDIA's ``validate_manifest``
  (``models.py:59-98``) then hashes the local copy against NVIDIA's ``SOURCE_MANIFEST.sha256``
  with its own path guard (``allowed_root`` = the copy itself) unchanged, so the bytes the model
  loads are still verified by NVIDIA's check.
- The TempSeg base (``models.py:128-130``) uses ``snapshot_download`` into the HF cache. We set
  ``HF_HUB_CACHE`` to a tmp dir with ``models--Qwen--Qwen3.5-9B/snapshots/<sha>`` -> the base
  snapshot on the Volume and ``.../trees`` -> its cached tree listing. In cache mode a commit-sha
  revision whose pointer path exists returns immediately, with no lock and no write, so the base
  is read straight from the read-only Volume.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from flora_deploy.config.pins import PINS_BY_KEY, RepoPin
from flora_deploy.layout import RECEIPT_NAME, WEIGHTS_MOUNT, snapshot_path
from flora_deploy.observe.aggregate import token_stats

OFFLINE_ROOT = PurePosixPath("/tmp/flora-offline")
MODEL_KINDS = ("tempseg", "captioner")  # NVIDIA's ``MODEL_CACHE / kind`` names
BASE_KEY = "tempseg_base"
# models.json keys -> our pin keys. ``config.MODELS`` must agree with pins.py exactly.
NV_MODEL_PINS = {"tempseg": "tempseg", "captioner": "captioner", "tempseg_base": "tempseg_base"}
# NVIDIA lets these env vars replace the repo id (``models.py:42``); they must stay unset.
NV_REPO_ENV = ("FLORA_TEMPSEG_REPO", "FLORA_CAPTIONER_REPO")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
RUNTIME_PREFIX = "FLORA_RUNTIME "


class OfflineLayoutError(RuntimeError):
    """The Volume snapshots cannot serve NVIDIA's loader offline."""


def offline_env(root: PurePosixPath | Path = OFFLINE_ROOT) -> dict[str, str]:
    """Env vars for the container. Must be set before ``huggingface_hub`` or NVIDIA's ``config``
    is imported (both read them at import time), so Modal sets them at container start."""
    root = PurePosixPath(root)
    return {
        "FLORA_MODEL_CACHE": str(root / "models"),
        "HF_HOME": str(root / "hf"),
        "HF_HUB_CACHE": str(root / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def hub_repo_folder(repo: str) -> str:
    """``huggingface_hub``'s cache folder name for a model repo (``repo_folder_name``)."""
    return "models--" + repo.replace("/", "--")


def tree_listing(snapshot_dir: Path, revision: str) -> Path:
    """Tree listing a ``local_dir`` download caches; offline ``snapshot_download`` needs it."""
    return snapshot_dir / ".cache" / "huggingface" / "trees" / f"{revision}.json"


@dataclass(frozen=True)
class OfflineLayout:
    root: Path
    model_cache: Path
    hub_cache: Path
    sources: dict[str, Path] = field(default_factory=dict)  # pin key -> Volume snapshot dir
    paths: dict[str, Path] = field(default_factory=dict)  # pin key -> dir NVIDIA's loader uses
    copies: dict[str, dict] = field(default_factory=dict)  # kind -> copy stats

    def env(self) -> dict[str, str]:
        return offline_env(self.root)


def _same_file(src: os.stat_result, dest: Path) -> bool:
    try:
        st = dest.stat()
    except FileNotFoundError:
        return False
    return st.st_size == src.st_size and st.st_mtime_ns == src.st_mtime_ns


def copy_snapshot(src: Path, dest: Path, *, workers: int = 8) -> dict:
    """Copy a prepared snapshot dir (including ``.cache/huggingface``) with ``copy2`` so mtimes
    survive. Skips files already copied (same size and mtime) and hub ``*.lock`` files."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.monotonic()
    jobs = []
    for path in sorted(src.rglob("*")):
        if path.is_dir() or path.name.endswith(".lock"):
            continue
        rel = path.relative_to(src)
        st = path.stat()
        jobs.append((path, dest / rel, st))
    todo = [(s, d, st) for s, d, st in jobs if not _same_file(st, d)]
    for _, d, _ in todo:
        d.parent.mkdir(parents=True, exist_ok=True)

    def one(job):
        s, d, _ = job
        partial = d.with_name(d.name + ".copying")
        shutil.copy2(s, partial)
        os.replace(partial, d)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, sorted(todo, key=lambda j: -j[2].st_size)))
    return {
        "files": len(jobs),
        "bytes": sum(st.st_size for _, _, st in jobs),
        "copied_files": len(todo),
        "copied_bytes": sum(st.st_size for _, _, st in todo),
        "seconds": round(time.monotonic() - t0, 2),
    }


def _check_snapshot(pin: RepoPin, target: Path, *, require_receipt: bool) -> None:
    if not target.is_dir():
        raise OfflineLayoutError(f"{pin.key}: snapshot {target} missing (run `flora prepare`)")
    if not tree_listing(target, pin.revision).is_file():
        raise OfflineLayoutError(
            f"{pin.key}: {tree_listing(target, pin.revision)} missing; offline snapshot_download "
            "would need the Hub to list files"
        )
    if require_receipt:
        path = target / RECEIPT_NAME
        try:
            receipt = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise OfflineLayoutError(f"{pin.key}: unreadable receipt {path}: {e}") from e
        if receipt.get("ok") is not True or receipt.get("pin", {}).get("revision") != pin.revision:
            raise OfflineLayoutError(f"{pin.key}: receipt {path} is not a passing receipt")


def _symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        raise OfflineLayoutError(f"{link} already points to {os.readlink(link)}, not {target}")
    if link.exists():
        raise OfflineLayoutError(f"{link} exists and is not our symlink")
    link.symlink_to(target, target_is_directory=True)


def build_offline_layout(
    root: Path | PurePosixPath = OFFLINE_ROOT,
    weights_root: Path | PurePosixPath = WEIGHTS_MOUNT,
    *,
    kinds: tuple[str, ...] = MODEL_KINDS,
    require_receipts: bool = True,
    workers: int = 8,
) -> OfflineLayout:
    """Create the layout described in the module docstring (idempotent; re-runs skip copies)."""
    root = Path(root)
    weights_root = PurePosixPath(weights_root)
    model_cache, hub_cache = root / "models", root / "hf" / "hub"
    sources: dict[str, Path] = {}
    paths: dict[str, Path] = {}
    copies: dict[str, dict] = {}
    for kind in kinds:
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {kind!r}")
        pin = PINS_BY_KEY[kind]
        src = Path(snapshot_path(pin, weights_root))
        _check_snapshot(pin, src, require_receipt=require_receipts)
        dest = model_cache / kind
        if dest.is_symlink():
            raise OfflineLayoutError(f"{dest} is a symlink; NVIDIA's loader needs a writable copy")
        copies[kind] = copy_snapshot(src, dest, workers=workers)
        sources[kind], paths[kind] = src, dest

    base = PINS_BY_KEY[BASE_KEY]
    src = Path(snapshot_path(base, weights_root))
    _check_snapshot(base, src, require_receipt=require_receipts)
    storage = hub_cache / hub_repo_folder(base.repo)
    _symlink(storage / "snapshots" / base.revision, src)
    _symlink(storage / "trees", tree_listing(src, base.revision).parent)
    sources[BASE_KEY], paths[BASE_KEY] = src, storage / "snapshots" / base.revision
    return OfflineLayout(root, model_cache, hub_cache, sources, paths, copies)


def check_nv_models(models: dict, environ: dict[str, str] | None = None) -> None:
    """NVIDIA's ``config.MODELS`` (models.json) must name exactly our pinned repos/revisions,
    and NVIDIA's repo-override env vars must be unset, so the Volume snapshots are what it loads."""
    environ = os.environ if environ is None else environ
    for nv_key, pin_key in NV_MODEL_PINS.items():
        pin = PINS_BY_KEY[pin_key]
        got = models.get(nv_key, {})
        if (got.get("repo"), got.get("revision")) != (pin.repo, pin.revision):
            raise OfflineLayoutError(
                f"NVIDIA models.json {nv_key}={got.get('repo')}@{got.get('revision')} "
                f"!= pin {pin.repo}@{pin.revision}"
            )
    set_vars = [k for k in NV_REPO_ENV if environ.get(k)]
    if set_vars:
        raise OfflineLayoutError(f"unset {set_vars}: they redirect NVIDIA's loader")


# --- NVIDIA runtime events -----------------------------------------------------------------


def parse_runtime_line(line: str) -> dict | None:
    """The JSON of one ``FLORA_RUNTIME {json}`` line (printed by NVIDIA's ``models.runtime_event``),
    or ``None``."""
    if line.startswith(RUNTIME_PREFIX):
        try:
            return json.loads(line[len(RUNTIME_PREFIX) :])
        except ValueError:
            return None
    return None


@contextmanager
def capture_runtime_events() -> Iterator[list[dict]]:
    """Collect this context's NVIDIA runtime events while still forwarding stdout.

    Thread-safe: built on ``observe.route_stdout`` (a per-context router), not
    ``redirect_stdout``, so concurrent episode threads each see only their own events, including
    events printed from NVIDIA's ``asyncio.to_thread`` predictor calls."""
    from flora_deploy.observe.stdout import route_stdout

    events: list[dict] = []

    def sink(line: str) -> None:
        event = parse_runtime_line(line)
        if event is not None:
            events.append(event)

    with route_stdout(sink):
        yield events


def generation_stats(events: list[dict]) -> dict:
    """Totals over NVIDIA's ``generation_end`` events (``models.py:183``)."""
    gens = [e for e in events if e.get("event") == "generation_end"]
    out = [int(e.get("generated_tokens", 0)) for e in gens]
    return {
        "calls": len(gens),
        "input_tokens": sum(int(e.get("input_tokens", 0)) for e in gens),
        "max_input_tokens": max((int(e.get("input_tokens", 0)) for e in gens), default=0),
        "generated_tokens": sum(out),
        "max_generated_tokens": max(out, default=0),
        "incomplete": sum(1 for e in gens if not e.get("complete", False)),
        "generation_s": round(sum(float(e.get("elapsed_s", 0.0)) for e in gens), 3),
        "peak_allocated_gib": max((float(e.get("peak_allocated_gib", 0)) for e in gens), default=0),
    }


# --- orchestration --------------------------------------------------------------------------

ROLE_SETS = ({"main"}, {"main", "left_wrist", "right_wrist"})


def annotate_reference(
    nv,
    session,
    videos: dict[str, str],
    profile: str,
    long_windows: bool = False,
    instruction: str = "",
    progress: Callable[..., None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run NVIDIA's pipeline on camera files with an already-loaded ``CachedSession``.

    LICENSE REVIEW FLAG (PLAN.md §11): re-expresses the orchestration of NVIDIA's
    ``api.DataEnginePipeline.annotate`` (``batch_utils/api.py:44-58``) and ``pipeline.annotate``
    (``batch_utils/pipeline.py:141-169``) at nvidia/Flora-TempSeg@78a0e4ac. Every step calls
    NVIDIA's own function; the only differences are (a) the session is our ``CachedSession``
    instead of a fresh ``Session()`` closed after the video, (b) per-stage wall-clock timers and
    runtime-event capture, (c) the audit is always kept (``include_audit=True``).

    ``videos`` maps camera roles (``main`` or ``main/left_wrist/right_wrist``) to local paths;
    ``profile`` is an NVIDIA API profile (``human``, ``yam``, ``dex``, ``tianji``). Returns NVIDIA's
    result dict plus a ``flora_reference`` key with our measurements.
    """
    if set(videos) not in ROLE_SETS:
        raise ValueError(f"camera roles must be main or main/left_wrist/right_wrist: {videos}")
    progress = progress or (lambda *a, **k: None)
    pipeline, media, models = nv.pipeline, nv.media, nv.models
    t0 = clock()
    # api.py:50 -> api.py:11-38: argument checks, size cap, probe per role, check_synchronized.
    nv_videos, canonical = nv.api.DataEnginePipeline._inputs(
        videos["main"],
        profile,
        videos.get("left_wrist"),
        videos.get("right_wrist"),
        long_windows,
        instruction,
    )
    t_probe = clock()
    # pipeline.py:142-152
    media.check_synchronized(nv_videos)
    nv.preprocessing.instruction_suffix(instruction)
    if canonical not in nv.config.PROFILE_POOLS:
        raise media.InputError("Choose a supported action profile.")
    if len(nv_videos) == 3 and canonical in ("Human", "Human actions"):
        raise media.InputError("Choose YAM, Dex or Tianji when uploading wrist cameras.")
    if long_windows and len(nv_videos) != 3:
        raise media.InputError("The 32-second profile requires three synchronized camera views.")
    if long_windows and canonical not in ("YAM", "Robot actions"):
        raise media.InputError("The 32-second profile is available for YAM videos.")
    pool = "yam32" if long_windows else nv.config.PROFILE_POOLS[canonical]
    start = clock()
    # pipeline.py:157-158
    with capture_runtime_events() as seg_events:
        segments, temporal_audit = pipeline.segment(nv_videos, pool, session, instruction, progress)
        models.runtime_event(
            "segmentation_complete", windows=len(temporal_audit), segments=len(segments)
        )
    t_seg = clock()
    # pipeline.py:160-163
    with capture_runtime_events() as cap_events:
        if len(nv_videos) == 3:
            captions, caption_audit = pipeline.caption_three(
                nv_videos, segments, session, instruction, progress, profile=canonical
            )
        else:
            captions, caption_audit = pipeline.caption_single(
                nv_videos, segments, canonical, session, instruction, progress
            )
    t_cap = clock()
    # pipeline.py:164-167
    for row, segment_row in zip(captions, segments, strict=True):
        row["segmentation_provenance"] = segment_row["metadata"].get(
            "segmentation_provenance", "fine"
        )
    models.runtime_event(
        "annotation_complete",
        segments=len(captions),
        complete_captions=sum(r["status"] == "complete" for r in captions),
        elapsed_s=round(clock() - start, 3),
    )
    main = nv_videos["main"]
    result = dict(
        schema_version="flora_demo_annotation_v1",
        duration_s=main.duration_s,
        segments=captions,
        elapsed_s=round(clock() - start, 2),
        source_clock="seconds relative to the first native source presentation timestamp",
        camera_roles=list(nv_videos),
        profile=canonical,
        model_revisions=nv.config.MODELS,
        input_sha256={role: video.sha256 for role, video in nv_videos.items()},
        pipeline=dict(temporal_windows=temporal_audit, caption_inputs=caption_audit),
    )
    # api.py:52-55
    for seg in result["segments"]:
        ok = 0 <= seg["start_s"] < seg["end_s"] <= result["duration_s"] + 0.001
        if not ok:
            raise RuntimeError("The pipeline returned an invalid caption interval.")
    result["flora_reference"] = {
        "pool": pool,
        "inputs": {
            role: {
                "path": str(v.path),
                "rate": str(v.rate),
                "frames": len(v.pts_us),
                "size": list(v.size),
                "cfr": v.cfr,
                "duration_us": v.duration_us,
            }
            for role, v in nv_videos.items()
        },
        "timings_s": {
            "probe": round(t_probe - t0, 3),
            "segment": round(t_seg - start, 3),
            "caption": round(t_cap - t_seg, 3),
            "total": round(t_cap - t0, 3),
        },
        "counts": {
            "temporal_windows": len(temporal_audit),
            "segments": len(segments),
            "caption_inputs": len(caption_audit),
            "captions_complete": sum(r["status"] == "complete" for r in captions),
        },
        "generation": {
            "segment": generation_stats(seg_events),
            "caption": generation_stats(cap_events),
        },
        "tokens": {
            "segment": token_stats(seg_events),
            "caption": token_stats(cap_events),
        },
    }
    return result


def nvidia_result(result: dict) -> dict:
    """NVIDIA's dict without our measurements (for parity comparisons)."""
    return {k: v for k, v in result.items() if k != "flora_reference"}


# --- output-identical concurrency (docs/exact-concurrency.md) -------------------------------
#
# N episode threads share the resident models; one GPU lock serializes every NVIDIA
# ``generate()`` call, so each model call is exactly NVIDIA's batch-1 call and outputs must be
# identical to a sequential run. Thread-safety audit of what the threads share:
#
# - Models: only touched inside ``generate`` (``model.generate``, ``.to(device)``), which the
#   GPU lock serializes. ``CachedSession.load`` is lock-protected and models are preloaded.
# - HF processor/tokenizer: NOT shared. ``PreTrainedTokenizerFast`` mutates the Rust tokenizer's
#   padding/truncation state per call (``set_truncation_and_padding``: ``caption_three`` pads,
#   the other builders do not) while encoding releases the GIL, which can raise "Already
#   borrowed" or race. Each worker thread therefore gets its own ``copy.deepcopy`` of each
#   processor (``ProcessorClones``); ``generate`` receives that same clone from NVIDIA's code.
# - NVIDIA module globals: ``pipeline``/``media``/``preprocessing`` keep no mutable state on the
#   episode path (``models._VERIFIED_MANIFESTS``/``_MANIFEST_LOCK`` are load-time only, already
#   locked; ``api._REQUEST_LOCK`` is unused here). Vendor code has no caches or globals. Each
#   ``pipeline.segment`` builds its own ``Predictor`` and runs its own ``asyncio.run`` loop (fine
#   in a non-main thread; the predictor runs in that loop's ``to_thread`` executor).
# - ``runtime_event`` logging: ``print`` to the shared stdout; ``capture_runtime_events`` routes
#   lines per context (``observe.route_stdout``), so episodes never see each other's events.
#   ``generation_end.peak_allocated_gib`` is process-global CUDA state (diagnostic only).
# - Temp dirs: ``caption_single`` uses ``tempfile.TemporaryDirectory`` (unique per call).
# - Volume writes/commits: done after the batch, in the calling thread.

GPU_LOCK_LABEL = "models.generate"
GPU_LOCK_NAME = "gpu"
TIMING_FIELDS = ("elapsed_s",)  # NVIDIA result fields that are wall-clock, not output


def locked_generate(generate: Callable, lock) -> Callable:
    """NVIDIA's ``generate`` run under ``lock`` (an ``observe.TimedLock``) and timed."""
    import functools

    from flora_deploy.observe.timers import timed

    inner = timed(generate, GPU_LOCK_LABEL)

    @functools.wraps(generate)
    def generate_locked(*args, **kwargs):
        with lock.held(GPU_LOCK_LABEL):
            return inner(*args, **kwargs)

    generate_locked.__flora_gpu_lock__ = lock
    return generate_locked


def install_gpu_lock(nv, lock, patcher) -> None:
    """Route every NVIDIA model call through ``lock``.

    ``pipeline.py`` does ``from .models import generate`` at import time, so the name its
    functions call is ``pipeline.generate``; we replace that module attribute in our process
    (NVIDIA's files are untouched)."""
    if getattr(nv.pipeline.generate, "__flora_gpu_lock__", None) is not None:
        raise RuntimeError("NVIDIA generate() already runs under a GPU lock")
    patcher.patch(nv.pipeline, "generate", lambda orig: locked_generate(orig, lock))


class ProcessorClones:
    """One ``copy.deepcopy`` of each processor per worker thread (see the audit above)."""

    def __init__(self, copier: Callable[[Any], Any] | None = None):
        import copy
        import threading

        self._copier = copier or copy.deepcopy
        self._local = threading.local()
        self._lock = threading.Lock()
        self.clones = 0
        self.seconds = 0.0

    def get(self, kind: str, proc: Any) -> Any:
        mine = getattr(self._local, "procs", None)
        if mine is None:
            mine = self._local.procs = {}
        entry = mine.get(kind)
        if entry is None or entry[0] is not proc:
            t = time.perf_counter()
            entry = mine[kind] = (proc, self._copier(proc))
            with self._lock:
                self.clones += 1
                self.seconds += time.perf_counter() - t
        return entry[1]


class EpisodeSession:
    """Per-episode view of a shared ``CachedSession``: same resident models, this thread's
    processor clones. Duck-types NVIDIA ``models.Session`` like ``CachedSession`` does."""

    def __init__(self, base, clones: ProcessorClones | None = None):
        self.base, self.clones = base, clones

    def load(self, kind: str):
        model, proc = self.base.load(kind)
        if self.clones is not None:
            proc = self.clones.get(kind, proc)
        return model, proc

    def close(self) -> None:
        """No-op (NVIDIA calls it per video; the models stay resident)."""

    @property
    def audit(self) -> dict:
        return getattr(self.base, "audit", {})


def measurement_targets(nv, processors=()) -> list[tuple[Any, str, Any]]:
    """``(owner, attribute, label)`` for every hot-path function we time (opt-in).

    Owners are the namespaces NVIDIA's code looks the name up in at call time: ``pipeline.*``
    for names it imported at module level, the vendor modules for names imported inside
    functions, and the HF/PIL classes for methods."""
    import importlib

    def mod(name):
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    targets: list[tuple[Any, str, Any]] = [
        (nv.media, "probe", "media.probe"),
        (nv.media, "decode_selected", "media.decode_selected"),
        (nv.pipeline, "frames_for", "media.frames_for"),
        (nv.pipeline, "human_transport", "preprocessing.human_transport"),
        (nv.pipeline, "temporal_inputs", "preprocessing.temporal_inputs"),
        (nv.pipeline, "three_view_caption_inputs", "preprocessing.three_view_caption_inputs"),
    ]
    vendor = [
        ("canonical_rgb", "canonical_rgb", "vendor.canonical_rgb"),
        ("composers", "compose_top_dominant_mosaic", "vendor.compose_mosaic"),
        ("composers", "compose_frame", "vendor.compose_frame"),
        ("robot_caption_geometry", "normalized", "vendor.caption_geometry"),
        ("flora_unified.unified_input", "processor_inputs", "unified_input.processor_inputs"),
        ("flora_unified.unified_input", "load_frames", "unified_input.load_frames"),
    ]
    for module_name, attr, label in vendor:
        module = mod(module_name)
        if module is not None and hasattr(module, attr):
            targets.append((module, attr, label))
    seen = set()
    for proc in processors:
        cls = type(proc)
        if cls in seen:
            continue
        seen.add(cls)
        targets.append((cls, "__call__", "hf.processor"))
        targets.append((cls, "apply_chat_template", "hf.chat_template"))
    pil = mod("PIL.Image")
    if pil is not None:

        def save_label(image, fp=None, format=None, **kw):  # noqa: A002 - PIL's signature
            fmt = format or (Path(str(fp)).suffix.lstrip(".") if isinstance(fp, str | Path)
                             else "")
            return f"pil.save.{(fmt or 'unknown').upper()}"

        targets.append((pil.Image, "save", save_label))
        targets.append((pil.Image, "resize", "pil.resize"))
        targets.append((pil.Image, "convert", "pil.convert"))
    return targets


def install_measurement(nv, patcher, processors=()) -> list[str]:
    """Wrap every ``measurement_targets`` entry with an ``observe.timed`` timer."""
    for owner, attr, label in measurement_targets(nv, processors):
        patcher.time(owner, attr, label)
    return patcher.patched


def comparable_result(result: dict) -> dict:
    """NVIDIA's result without our measurements and without its wall-clock fields."""
    return {k: v for k, v in nvidia_result(result).items() if k not in TIMING_FIELDS}


def result_digest(result: dict) -> str:
    import hashlib

    return hashlib.sha256(dumps_result(comparable_result(result)).encode()).hexdigest()


def diff_paths(a: Any, b: Any, path: str = "$", limit: int = 20) -> list[str]:
    """JSON paths where ``a`` and ``b`` differ (no values, so no captions leak into reports)."""
    out: list[str] = []

    def walk(x, y, p):
        if len(out) >= limit:
            return
        if type(x) is not type(y):
            out.append(p)
        elif isinstance(x, dict):
            for k in sorted(set(x) | set(y), key=str):
                if k not in x or k not in y:
                    out.append(f"{p}.{k}")
                else:
                    walk(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, list):
            if len(x) != len(y):
                out.append(f"{p}[len]")
            for i, (u, v) in enumerate(zip(x, y, strict=False)):
                walk(u, v, f"{p}[{i}]")
        elif x != y:
            out.append(p)

    walk(a, b, path)
    return out[:limit]


def compare_runs(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Compare two ``{episode: result}`` maps on ``comparable_result``."""
    episodes = {}
    for ep in sorted(set(a) | set(b)):
        if ep not in a or ep not in b:
            episodes[ep] = {"identical": False, "missing_in": "a" if ep not in a else "b"}
            continue
        ca, cb = comparable_result(a[ep]), comparable_result(b[ep])
        same = ca == cb
        episodes[ep] = {"identical": same, "digest_a": result_digest(a[ep]),
                        "digest_b": result_digest(b[ep]),
                        "diff_paths": [] if same else diff_paths(ca, cb)}
    return {"identical": bool(episodes) and all(e["identical"] for e in episodes.values()),
            "episodes": episodes}


def episode_observe(recorder, result: dict | None, t_batch0: float,
                    lock_name: str = GPU_LOCK_NAME) -> dict:
    """Per-episode measurement summary from its ``observe.Recorder``."""
    from flora_deploy.observe.aggregate import call_table

    events = recorder.events
    table = call_table(events)
    timed_self = sum(r.get("self_s", 0.0) for k, r in table.items() if not k.startswith("lock:"))
    out = {"calls": table, "timed_self_s": round(timed_self, 3)}
    lock = table.get(f"lock:{lock_name}")
    if lock:
        out["gpu_lock"] = {"wait_s": lock["wait_s"], "hold_s": lock["hold_s"],
                           "max_wait_s": lock["max_wait_s"], "calls": lock["calls"]}
    if result is not None:
        total = result.get("flora_reference", {}).get("timings_s", {}).get("total")
        if total is not None:
            out["untimed_s"] = round(total - timed_self - (lock or {}).get("wait_s", 0.0), 3)
    out["start_s"] = round(recorder.t0 - t_batch0, 3)
    return out


def annotate_batch(
    nv,
    session,
    episodes: list[dict],
    *,
    concurrency: int = 1,
    gpu_lock=None,
    measure: bool = False,
    clone_processors: bool | None = None,
    sampler=None,
    progress: Callable[[str, float, str], None] | None = None,
) -> dict:
    """Run ``annotate_reference`` over ``episodes`` with ``concurrency`` worker threads.

    ``episodes`` items: ``{"episode", "videos", "profile", ["long_windows"], ["instruction"]}``.
    ``concurrency=1`` runs them in order in the calling thread (the sequential baseline). With
    ``concurrency>1`` NVIDIA's ``generate`` must already run under ``gpu_lock``
    (``install_gpu_lock``); processors are cloned per worker unless ``clone_processors=False``.
    Returns ``{"results": {episode: result}, "errors": {episode: str}, "batch": {...}}``; results
    keep input order. ``measure`` attaches ``flora_reference["observe"]`` per episode.
    """
    from concurrent.futures import ThreadPoolExecutor

    from flora_deploy.observe.events import Recorder, recording

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    names = [ep["episode"] for ep in episodes]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate episode names: {names}")
    if concurrency > 1:
        lock = getattr(nv.pipeline.generate, "__flora_gpu_lock__", None)
        if lock is None or (gpu_lock is not None and lock is not gpu_lock):
            raise RuntimeError("concurrency > 1 needs NVIDIA generate() under the GPU lock")
        gpu_lock = lock
    if clone_processors is None:
        clone_processors = concurrency > 1
    clones = ProcessorClones() if clone_processors else None
    if gpu_lock is not None:
        gpu_lock.reset()
    t0 = time.perf_counter()
    if sampler is not None:
        sampler.start()

    def one(ep: dict) -> tuple[dict | None, str | None]:
        rec = Recorder(episode=ep["episode"]) if measure else None
        report = None
        if progress is not None:
            def report(frac, desc="", _name=ep["episode"]):
                progress(_name, frac, desc)
        try:
            with recording(rec):
                result = annotate_reference(
                    nv, EpisodeSession(session, clones), ep["videos"], ep["profile"],
                    ep.get("long_windows", False), ep.get("instruction", ""), progress=report,
                )
        except Exception as e:  # noqa: BLE001 - one bad episode must not sink the batch
            return None, f"{type(e).__name__}: {e}"
        ref = result["flora_reference"]
        ref["batch"] = {"concurrency": concurrency, "end_s": round(time.perf_counter() - t0, 3)}
        if rec is not None:
            ref["observe"] = episode_observe(rec, result, t0,
                                             gpu_lock.name if gpu_lock else GPU_LOCK_NAME)
        return result, None

    if concurrency == 1:
        outcomes = [one(ep) for ep in episodes]
    else:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="flora-ep") as pool:
            outcomes = list(pool.map(one, episodes))
    wall = time.perf_counter() - t0
    results = {ep["episode"]: r for ep, (r, _) in zip(episodes, outcomes, strict=True) if r}
    errors = {ep["episode"]: e for ep, (_, e) in zip(episodes, outcomes, strict=True) if e}
    video_s = sum(float(r.get("duration_s", 0.0)) for r in results.values())
    batch: dict[str, Any] = {
        "concurrency": concurrency,
        "episodes": len(episodes),
        "ok": len(results),
        "wall_s": round(wall, 3),
        "video_s": round(video_s, 3),
        "video_s_per_wall_s": round(video_s / wall, 4) if wall > 0 else None,
        "processor_clones": {"clones": clones.clones, "seconds": round(clones.seconds, 3)}
        if clones else None,
    }
    if gpu_lock is not None:
        batch["gpu_lock"] = gpu_lock.stats()
        batch["gpu_busy_fraction"] = round(gpu_lock.busy_fraction(t0, t0 + wall), 4)
    if sampler is not None:
        batch["resources"] = sampler.stop()
    if measure:
        from flora_deploy.observe.aggregate import merge_tables

        batch["calls"] = merge_tables(
            r["flora_reference"]["observe"]["calls"] for r in results.values()
        )
    return {"results": results, "errors": errors, "batch": batch}


# --- persistence ----------------------------------------------------------------------------


def safe_name(name: str) -> str:
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"unsafe name {name!r}")
    return name


def result_relpath(run_id: str, episode: str) -> PurePosixPath:
    return PurePosixPath("reference_runs") / safe_name(run_id) / f"{safe_name(episode)}.json"


def dumps_result(result: dict) -> str:
    """Serialized exactly like NVIDIA's CLI writes results (``cli.py:73``)."""
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def persist_result(
    result: dict,
    *,
    golden_root: Path | PurePosixPath,
    run_id: str,
    episode: str,
    commit: Callable[[], None] = lambda: None,
    tmp_dir: str | None = None,
) -> Path:
    """Write ``result`` to a local temp file, copy it whole to
    ``<golden_root>/reference_runs/<run_id>/<episode>.json`` (via a ``.partial`` + rename), then
    ``commit()`` the Volume. Never keeps a file open on the Volume."""
    dest = Path(golden_root) / result_relpath(run_id, episode)
    payload = dumps_result(result)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        local = Path(tmp) / dest.name
        local.write_text(payload)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".partial")
        shutil.copyfile(local, partial)
        os.replace(partial, dest)
    commit()
    return dest


def summary_row(episode: str, result: dict) -> dict[str, Any]:
    """Flat numbers for the smoke table (no captions, no prompt hashes)."""
    ref = result.get("flora_reference", {})
    gen = ref.get("generation", {})
    counts = ref.get("counts", {})
    t = ref.get("timings_s", {})
    gpu = ref.get("gpu", {})
    return {
        "episode": episode,
        "views": len(result.get("camera_roles", [])),
        "profile": result.get("profile"),
        "duration_s": round(float(result.get("duration_s", 0.0)), 2),
        "windows": counts.get("temporal_windows"),
        "segments": counts.get("segments"),
        "captions": counts.get("captions_complete"),
        "probe_s": t.get("probe"),
        "segment_s": t.get("segment"),
        "caption_s": t.get("caption"),
        "seg_out_tok": gen.get("segment", {}).get("generated_tokens"),
        "cap_out_tok": gen.get("caption", {}).get("generated_tokens"),
        "peak_gib": gpu.get("max_allocated_gib"),
    }


# --- WS-D smoke samples (docs/reference-smoke.md) -------------------------------------------

WGO_REPO = "macrodata/WGO-Bench"
WGO_REVISION = "9cfb7bb98d60a58319f4b3d7b41374eedfb7980c"
WGO_FILE = "data/annotations.parquet"
WGO_LICENSE = "CC-BY-NC-SA-4.0 (WGO-Bench; HomER subset from toloka/HomER)"
ALOHA_REPO = "lerobot/aloha_static_coffee"
ALOHA_REVISION = "7da22eb685a613a28bcc58aa0bf7d1094578234a"  # tag v2.1 (codebase_version v2.1)
ALOHA_LICENSE = "MIT"
ALOHA_EPISODE = 0
# PLAN.md §9: main=cam_high; cam_low is dropped.
ALOHA_ROLES = {"main": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
HUMAN_COUNT = 3
HUMAN_MIN_S, HUMAN_MAX_S = 20.0, 90.0


def aloha_video_file(camera: str, episode: int = ALOHA_EPISODE) -> str:
    """LeRobot v2.1 ``video_path`` template from ``meta/info.json``."""
    chunk = episode // 1000
    return f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode:06d}.mp4"


def choose_human(
    candidates: list[dict],
    n: int = HUMAN_COUNT,
    min_s: float = HUMAN_MIN_S,
    max_s: float = HUMAN_MAX_S,
) -> list[dict]:
    """Pick the ``n`` shortest probe-passing clips in ``[min_s, max_s]`` (cheap but multi-window);
    fall back to the shortest passing clips if too few fall in range."""
    ok = sorted((c for c in candidates if c.get("error") is None), key=lambda c: c["duration_s"])
    picked = [c for c in ok if min_s <= c["duration_s"] <= max_s][:n]
    if len(picked) < n:
        picked += [c for c in ok if c not in picked][: n - len(picked)]
    if len(picked) < n:
        raise RuntimeError(f"only {len(picked)} usable human clips")
    return picked


def probe_summary(video) -> dict:
    """JSON view of NVIDIA's ``media.Video``."""
    return {
        "sha256": video.sha256,
        "rate": str(video.rate),
        "frames": len(video.pts_us),
        "duration_s": video.duration_s,
        "size": list(video.size),
        "cfr": video.cfr,
    }


def copy_whole(src: Path, dest: Path) -> None:
    """Copy onto a Volume as one file: ``.partial`` then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, partial)
    os.replace(partial, dest)


def write_json_whole(obj: dict, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / dest.name
        local.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        copy_whole(local, dest)


def stage_all(
    samples_dir: Path | PurePosixPath,
    *,
    code_dir: Path,
    commit: Callable[[], None] = lambda: None,
    force: bool = False,
) -> dict:
    """Download, check with NVIDIA's ``probe``/``check_synchronized``, and copy the smoke samples.

    Writes ``<samples_dir>/manifest.json`` listing each episode's camera files, NVIDIA profile,
    source repo/revision/license and probe summary. Idempotent unless ``force``.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from flora_deploy.nv import Family, load_family

    samples_dir = Path(samples_dir)
    manifest_path = samples_dir / "manifest.json"
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text())
    nv = load_family(Family.SEGCAP, code_dir)
    episodes = []
    with tempfile.TemporaryDirectory(prefix="wsd-") as tmp:
        tmp = Path(tmp)
        # Human single view: HomER rows of WGO-Bench (MP4 bytes embedded in the parquet).
        parquet = hf_hub_download(WGO_REPO, WGO_FILE, repo_type="dataset", revision=WGO_REVISION)
        candidates = []
        pf = pq.ParquetFile(parquet)
        for batch in pf.iter_batches(columns=["id", "instruction", "video"], batch_size=4):
            for row in batch.to_pylist():
                if not str(row["id"]).startswith("homer_"):
                    continue
                path = tmp / f"{row['id']}.mp4"
                video = row["video"]
                path.write_bytes(video["bytes"] if isinstance(video, dict) else video)
                entry = {"id": row["id"], "path": path, "instruction": row["instruction"]}
                try:
                    v = nv.media.probe(path)
                    entry.update(duration_s=v.duration_s, probe=probe_summary(v), error=None)
                except Exception as e:  # noqa: BLE001 - NVIDIA InputError and decode errors
                    entry.update(error=f"{type(e).__name__}: {e}")
                    path.unlink(missing_ok=True)
                candidates.append(entry)
        for c in choose_human(candidates):
            dest = samples_dir / "human" / f"{c['id']}.mp4"
            copy_whole(c["path"], dest)
            episodes.append(
                {
                    "episode": c["id"],
                    "profile": "human",
                    "videos": {"main": str(dest)},
                    "dataset_instruction": c["instruction"],
                    "source": {
                        "repo": WGO_REPO,
                        "revision": WGO_REVISION,
                        "file": WGO_FILE,
                        "row_id": c["id"],
                        "license": WGO_LICENSE,
                    },
                    "probe": {"main": c["probe"]},
                }
            )

        # Three synchronized views: one LeRobot v2.1 ALOHA episode, one MP4 per camera.
        paths = {
            role: Path(
                hf_hub_download(
                    ALOHA_REPO,
                    aloha_video_file(cam),
                    repo_type="dataset",
                    revision=ALOHA_REVISION,
                    local_dir=tmp / "aloha",
                )
            )
            for role, cam in ALOHA_ROLES.items()
        }
        videos = {role: nv.media.probe(p) for role, p in paths.items()}
        nv.media.check_synchronized(videos)  # identical PTS/rate/duration, and CFR
        name = f"{ALOHA_REPO.split('/')[1]}_ep{ALOHA_EPISODE:06d}"
        dests = {role: samples_dir / name / f"{role}.mp4" for role in paths}
        for role, p in paths.items():
            copy_whole(p, dests[role])
        episodes.append(
            {
                "episode": name,
                "profile": "yam",
                "videos": {role: str(d) for role, d in dests.items()},
                "source": {
                    "repo": ALOHA_REPO,
                    "revision": ALOHA_REVISION,
                    "files": {r: aloha_video_file(c) for r, c in ALOHA_ROLES.items()},
                    "camera_map": ALOHA_ROLES,
                    "license": ALOHA_LICENSE,
                },
                "probe": {role: probe_summary(v) for role, v in videos.items()},
            }
        )
    manifest = {
        "episodes": episodes,
        "human_candidates": [
            {"id": c["id"], "duration_s": c.get("duration_s"), "error": c["error"]}
            for c in candidates
        ],
    }
    write_json_whole(manifest, manifest_path)
    commit()
    return manifest
