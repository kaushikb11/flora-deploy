"""The ``exact`` reference stage: NVIDIA's own TempSeg -> Captioner pipeline (PLAN.md §4, §6.5).

Modal-agnostic pieces used by ``modal_app/reference_app.py`` (``docs/exact-worker.md``):

- ``annotate_nvidia``: runs NVIDIA's own entry point ``api.DataEnginePipeline().annotate``
  with our resident ``CachedSession`` injected through a runtime swap of ``pipeline.Session``
  (``install_nvidia_entry``). No NVIDIA orchestration is re-expressed here.
- ``annotate_batch``: N episode threads sharing the resident models under one GPU lock (W3-A),
  optionally with the OPT-1 ``accel`` patches active.
- ``model_calls`` / ``compare_model_inputs`` / ``identity_summary``: model-input identity from
  the ``input_tensor_digest`` NVIDIA's ``generate()`` logs.
- ``ExactProfile``: which proven speedups are on (each toggleable, recorded in provenance).
- ``build_offline_layout`` / ``check_prep_receipt``: WS-D copy layout and full prep-receipt
  validation (the no-copy layout is ``modal_app/warm.build_linked_layout``).
- ``capture_runtime_events`` / ``generation_stats``: NVIDIA's own ``FLORA_RUNTIME`` events.
- ``persist_result``: write a result JSON locally, then copy it whole onto a Volume.

How offline loading works (NVIDIA ``batch_utils/models.py`` at nvidia/Flora-TempSeg@78a0e4ac):

- ``snapshot(kind)`` (``models.py:39-46``) calls
  ``snapshot_download(repo, revision=<sha>, allow_patterns=..., local_dir=MODEL_CACHE / kind)``.
  With a commit-sha revision, ``huggingface_hub`` reads the tree listing the prep job cached in
  ``<snapshot>/.cache/huggingface/trees/`` and, per file, the download metadata in
  ``.cache/huggingface/download/`` and returns the existing file without a request
  (``HF_HUB_OFFLINE=1`` makes any request raise instead). Reading that metadata takes a
  ``filelock`` *inside* ``local_dir``, so ``local_dir`` must be writable. WS-D copied each
  snapshot to local disk (``build_offline_layout``); W3-B's link farm (``warm``) avoids the copy.
  Preserving mtimes in the copy is **not required** (review2-wsd L2, measured): with a changed
  mtime hub logs "Ignored metadata (outdated)" and still resolves offline; ``copy2`` keeps them
  only to avoid that warning. Integrity rests on NVIDIA's ``validate_manifest`` (TempSeg,
  Captioner) and on our prep receipts (``check_prep_receipt``; the only anchor for the base).
- The TempSeg base (``models.py:128-130``) uses ``snapshot_download`` into the HF cache. We set
  ``HF_HUB_CACHE`` to a tmp dir with ``models--Qwen--Qwen3.5-9B/snapshots/<sha>`` -> the base
  snapshot on the Volume and ``.../trees`` -> its cached tree listing. In cache mode a commit-sha
  revision whose pointer path exists returns immediately, with no lock and no write, so the base
  is read straight from the read-only Volume.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
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
    """Copy a prepared snapshot dir (including ``.cache/huggingface``) with ``copy2`` (mtimes are
    kept only to avoid hub's "outdated metadata" warning; see the module docstring). Skips files
    already copied (same size and mtime) and hub ``*.lock`` files."""
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


LEGACY_RECEIPT_SCHEMA = 1  # WS-A's first receipts: sha256 + size per file, no fingerprints
SMALL_FILE_BYTES = 64 * 2**20


def check_prep_receipt(pin: RepoPin, snapshot_dir: Path,
                       fields: tuple[str, ...] | None = None, *,
                       hash_files: str = "small") -> dict:
    """Full offline validation of a snapshot's ``prep_receipt.json`` (review2-wsd M5).

    Same rules as ``prep.verify.check_receipt`` minus the Hub plan (offline there is no Hub tree;
    the pinned ``plan_digest`` stands in for it): ``ok`` and every recorded check passing; the
    receipt's pin equal to this pin (so a receipt of an older pin or ``allow_patterns`` is
    refused); ``plan_digest`` equal to the pinned one; exactly the receipt's files on disk with
    the recorded sizes.

    - Current schema: also ``pin_digest`` (pin + verifier version) and each file's fingerprint
      (``fields``, default size, mtime_ns and inode) equal to what was hashed, so a rewrite,
      partial re-prep or ``modal volume put`` is caught without hashing.
    - Legacy schema 1 (no fingerprints, no ``pin_digest`` were recorded): sizes only, so
      ``hash_files`` decides what is re-hashed against the receipt's sha256.

    ``hash_files``: ``"small"`` (files under 64 MiB: configs, tokenizer, chat template - a few
    MB), ``"all"`` (every byte) or ``"none"``. Returns a small report; raises
    ``OfflineLayoutError``."""
    from concurrent.futures import ThreadPoolExecutor

    from flora_deploy.prep import verify

    fields = tuple(verify.REUSE_FINGERPRINT if fields is None else fields)
    root = Path(snapshot_dir)
    path = root / RECEIPT_NAME
    t0 = time.monotonic()
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise OfflineLayoutError(f"{pin.key}: unreadable receipt {path}: {e}") from e

    def bad(why: str) -> OfflineLayoutError:
        return OfflineLayoutError(f"{pin.key}: receipt {path} rejected: {why}")

    schema = receipt.get("schema") if isinstance(receipt, dict) else None
    if schema not in (verify.RECEIPT_SCHEMA, LEGACY_RECEIPT_SCHEMA):
        raise bad(f"schema {schema} is neither {verify.RECEIPT_SCHEMA} nor "
                  f"{LEGACY_RECEIPT_SCHEMA}")
    current = schema == verify.RECEIPT_SCHEMA
    if receipt.get("ok") is not True or not all(c.get("ok") for c in receipt.get("checks", [])):
        raise bad("not a passing receipt")
    want_pin = verify.pin_record(pin)
    got_pin = receipt.get("pin")
    if not current:
        # Schema 1 did not record plan_digest inside the pin, and pins.py gained expected
        # digests (manifest_sha256) after those receipts were written: what it recorded must
        # agree; the rest is derived from the receipt's own sha256 table below.
        drop = ("plan_digest", "expected")
        got_expected = dict((got_pin or {}).get("expected") or {})
        if any(dict(pin.expected).get(k) != v for k, v in got_expected.items()):
            raise bad("expected digests differ from pins.py")
        want_pin = {k: v for k, v in want_pin.items() if k not in drop}
        got_pin = {k: v for k, v in (got_pin or {}).items() if k not in drop}
    if got_pin != want_pin:
        raise bad("pin record differs from pins.py")
    if current and receipt.get("pin_digest") != verify.pin_digest(pin):
        raise bad("pin_digest differs (pin, verifier or schema changed)")
    if not pin.plan_digest or receipt.get("plan_digest") != pin.plan_digest.lower():
        raise bad("plan_digest differs from the pinned plan_digest")
    digests = {d.get("path"): d for d in receipt.get("digests", []) if isinstance(d, dict)}
    try:
        on_disk = verify.local_files(root)
    except OSError as e:
        raise bad(f"snapshot unreadable: {e}") from e
    if on_disk != set(digests):
        raise bad(f"file set differs: {sorted(on_disk ^ set(digests))[:5]}")
    for rel, rec in digests.items():
        size, mtime_ns, ino, _dev = verify.fingerprint(root / rel)
        now = {"size": size, "mtime_ns": mtime_ns, "ino": ino}
        changed = [k for k in (fields if current else ("size",)) if rec.get(k) != now[k]]
        if changed:
            raise bad(f"{rel}: fingerprint changed ({', '.join(changed)})")
    if hash_files not in ("small", "all", "none"):
        raise ValueError(f"hash_files must be small, all or none, not {hash_files!r}")
    todo = [rel for rel, rec in digests.items()
            if hash_files == "all" or (hash_files == "small"
                                       and int(rec.get("size", 0)) < SMALL_FILE_BYTES)]

    def sha(rel: str) -> tuple[str, bool]:
        got, _, _ = verify.hash_file(root / rel, git_blob=False)
        return rel, got == digests[rel].get("sha256")

    with ThreadPoolExecutor(max_workers=8) as pool:
        mismatched = [rel for rel, ok in pool.map(sha, sorted(todo)) if not ok]
    if mismatched:
        raise bad(f"sha256 differs from the receipt: {mismatched[:5]}")
    # pins.py's expected digests (e.g. NVIDIA's manifest sha256), from the receipt's table.
    table = {rel: rec.get("sha256") for rel, rec in digests.items()}
    for key, value in dict(pin.expected).items():
        derive = verify._EXPECTED.get(key)
        if derive is None:
            raise bad(f"unknown expected digest {key!r}")
        try:
            got, _src = derive(pin, root, table)
        except (OSError, ValueError) as e:
            raise bad(f"expected {key}: {e}") from e
        if got != value:
            raise bad(f"expected {key} differs from pins.py")
    return {"pin": pin.key, "schema": schema, "files": len(digests),
            "bytes": sum(int(d.get("size", 0)) for d in digests.values()),
            "fingerprints": list(fields) if current else ["size"],
            "hashed_files": len(todo),
            "hashed_bytes": sum(int(digests[r].get("size", 0)) for r in todo),
            "seconds": round(time.monotonic() - t0, 3)}


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
    model_cache: Path | None = None,
) -> OfflineLayout:
    """Create the WS-D copy layout described in the module docstring (idempotent; re-runs skip
    copies). Receipts get the full ``check_prep_receipt``. ``model_cache`` overrides
    ``<root>/models`` (it must equal the container's ``FLORA_MODEL_CACHE``)."""
    root = Path(root)
    weights_root = PurePosixPath(weights_root)
    model_cache = Path(model_cache) if model_cache is not None else root / "models"
    hub_cache = root / "hf" / "hub"
    sources: dict[str, Path] = {}
    paths: dict[str, Path] = {}
    copies: dict[str, dict] = {}
    for kind in kinds:
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {kind!r}")
        pin = PINS_BY_KEY[kind]
        src = Path(snapshot_path(pin, weights_root))
        _check_snapshot(pin, src, require_receipt=require_receipts)
        if require_receipts:
            check_prep_receipt(pin, src)
        dest = model_cache / kind
        if dest.is_symlink():
            raise OfflineLayoutError(f"{dest} is a symlink; NVIDIA's loader needs a writable copy")
        copies[kind] = copy_snapshot(src, dest, workers=workers)
        sources[kind], paths[kind] = src, dest

    base = PINS_BY_KEY[BASE_KEY]
    src = Path(snapshot_path(base, weights_root))
    _check_snapshot(base, src, require_receipt=require_receipts)
    if require_receipts:
        check_prep_receipt(base, src)
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


IDENTITY_PREFIXES = (RUNTIME_PREFIX, "FLORA_OBSERVE ")


@contextmanager
def capture_identity_events() -> Iterator[tuple[list[dict], list[str]]]:
    """Like :func:`capture_runtime_events`, and also keep the raw ``FLORA_RUNTIME`` /
    ``FLORA_OBSERVE`` lines ``accel.identity.model_calls`` reads (``generation_start`` /
    ``generation_end`` and the GPU lock's ``generate_output``), in order."""
    from flora_deploy.observe.stdout import route_stdout

    events: list[dict] = []
    lines: list[str] = []

    def sink(line: str) -> None:
        if line.startswith(IDENTITY_PREFIXES):
            lines.append(line)
        event = parse_runtime_line(line)
        if event is not None:
            events.append(event)

    with route_stdout(sink):
        yield events, lines


def identity_records(lines: list[str]) -> list[dict]:
    """``accel.identity.call_records`` of one episode's captured lines (``[]`` if none)."""
    from flora_deploy.accel.identity import call_records

    return call_records(lines)


def identity_calls_from_model_calls(calls: list[dict]) -> list[dict]:
    """``model_calls`` entries (persisted results, also older ones without ``identity_calls``)
    as ``accel.identity`` call records. The output hash is the one paired from NVIDIA's result
    (``model_calls``), so both runs compared this way must use this same conversion."""
    return [{"index": i, "input_tensor_digest": c.get("input_digest") or "",
             "output_sha256": c.get("output_sha256"), "input_tokens": c.get("input_tokens"),
             "output_limit": c.get("output_limit"),
             "generated_tokens": c.get("generated_tokens"), "complete": c.get("complete")}
            for i, c in enumerate(calls)]


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


# --- running NVIDIA's own entry point ------------------------------------------------------
#
# Review2-wsd M1: we no longer re-express ``pipeline.annotate``. Every episode runs NVIDIA's own
# ``api.DataEnginePipeline().annotate(..., include_audit=True)``, which runs NVIDIA's ``_inputs``
# (argument checks, 256 MiB cap, probe, ``check_synchronized``), ``pipeline.annotate`` (profile
# checks, ``segment``, ``caption_*``, result dict) and its own interval check. Two module
# attributes are swapped at runtime in our process (NVIDIA's files are untouched):
#
# - ``pipeline.Session``: ``pipeline.annotate`` does ``session = Session()`` (the name it imported
#   from ``.models``) and ``session.close()`` in ``finally``. We bind it to a factory returning the
#   episode's ``EpisodeSession`` view of the resident ``CachedSession`` (whose ``close`` is a
#   no-op). The episode's session travels in a ``ContextVar`` set by the episode's own thread, so
#   concurrent episodes each get their own processor clones.
# - ``api._REQUEST_LOCK`` (only for concurrency > 1): NVIDIA's RLock serializes whole requests
#   ("one video at a time"). With our GPU lock serializing every model call
#   (``install_gpu_lock``) it is replaced by a no-op lock so CPU work of other episodes overlaps.
#
# Unbinding (``Patcher.restore``) gives back NVIDIA's original attributes: that is the
# "unmodified NVIDIA" arm of the identity proof.

ROLE_SETS = ({"main"}, {"main", "left_wrist", "right_wrist"})
_EPISODE_SESSION: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "flora_episode_session", default=None
)
_SESSION_MARK = "__flora_session_factory__"


class _NoLock:
    """Stand-in for NVIDIA's ``api._REQUEST_LOCK`` when our GPU lock serializes model calls."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def acquire(self, *a, **k):
        return True

    def release(self):
        return None


def _bound_session_factory(original: Callable) -> Callable:
    def Session():  # noqa: N802 - NVIDIA's name
        session = _EPISODE_SESSION.get()
        if session is None:
            raise RuntimeError("pipeline.Session() called with no episode session bound "
                               "(use stages.reference.bind_session)")
        return session

    setattr(Session, _SESSION_MARK, original)
    return Session


def install_nvidia_entry(nv, patcher, *, request_lock: bool = True) -> None:
    """Make NVIDIA's ``annotate`` use the bound episode session (see the section comment).
    ``request_lock=False`` also swaps NVIDIA's whole-request lock for a no-op (concurrency)."""
    if getattr(nv.pipeline.Session, _SESSION_MARK, None) is not None:
        raise RuntimeError("NVIDIA's pipeline.Session is already bound")
    patcher.patch(nv.pipeline, "Session", _bound_session_factory)
    if not request_lock:
        patcher.patch(nv.api, "_REQUEST_LOCK", lambda orig: _NoLock())


@contextmanager
def bind_session(session) -> Iterator[None]:
    token = _EPISODE_SESSION.set(session)
    try:
        yield
    finally:
        _EPISODE_SESSION.reset(token)


def annotate_nvidia(
    nv,
    session,
    videos: dict[str, str],
    profile: str,
    long_windows: bool = False,
    instruction: str = "",
    progress: Callable[..., None] | None = None,
) -> dict:
    """Run NVIDIA's ``DataEnginePipeline().annotate(include_audit=True)`` on camera files.

    ``session``: the session NVIDIA's ``pipeline.annotate`` should use (``install_nvidia_entry``
    must be active), or ``None`` for NVIDIA's own fresh ``Session()`` (unmodified path).
    ``videos`` maps roles (``main`` or ``main/left_wrist/right_wrist``) to local paths. Returns
    NVIDIA's result dict plus a ``flora_reference`` key with our measurements, including
    ``model_calls``: one entry per NVIDIA ``generate()`` with NVIDIA's logged input-tensor digest.
    NVIDIA's exceptions propagate unchanged (``transport_error`` converts them at the Modal edge).
    """
    if set(videos) not in ROLE_SETS:
        raise ValueError(f"camera roles must be main or main/left_wrist/right_wrist: {videos}")
    bound = getattr(nv.pipeline.Session, _SESSION_MARK, None) is not None
    if session is not None and not bound:
        raise RuntimeError("a session was given but install_nvidia_entry is not active")
    if session is None and bound:
        raise RuntimeError("install_nvidia_entry is active: restore it for NVIDIA's own Session")
    user_progress = progress or (lambda *a, **k: None)
    marks: list[tuple[float, str]] = []

    def progress_cb(frac, desc="", *a, **k):
        if not marks:
            marks.append((time.time(), desc))  # first call: right after ``Session()``
        return user_progress(frac, desc)

    t0 = time.time()
    with (bind_session(session) if session is not None else nullcontext()), \
            capture_identity_events() as (events, lines):
        result = nv.api.DataEnginePipeline().annotate(
            videos["main"], profile=profile, left_wrist=videos.get("left_wrist"),
            right_wrist=videos.get("right_wrist"), yam32=long_windows, instruction=instruction,
            progress=progress_cb, include_audit=True,
        )
    t_end = time.time()
    split = next((i for i, e in enumerate(events) if e.get("event") == "segmentation_complete"),
                 len(events))
    seg_events, cap_events = events[:split], events[split:]

    def utc(name: str) -> float | None:
        e = next((x for x in events if x.get("event") == name), None)
        return _utc_seconds(e.get("utc")) if e else None

    t_probe = marks[0][0] if marks else None
    t_seg, t_cap = utc("segmentation_complete"), utc("annotation_complete")
    audit = result.get("pipeline", {})

    def span(a, b):
        return round(b - a, 3) if a is not None and b is not None else None

    result["flora_reference"] = {
        "entry": "api.DataEnginePipeline.annotate",
        "session": "nvidia_fresh" if session is None else type(session).__name__,
        "inputs": {role: str(path) for role, path in videos.items()},
        "timings_s": {
            "probe": span(t0, t_probe),  # NVIDIA's _inputs: probe + checks (+ Session())
            "segment": span(t_probe, t_seg),  # includes the TempSeg load on a fresh Session
            "caption": span(t_seg, t_cap),  # includes the Captioner load on a fresh Session
            "total": round(t_end - t0, 3),
        },
        "counts": {
            "temporal_windows": len(audit.get("temporal_windows", [])),
            "segments": len(result.get("segments", [])),
            "caption_inputs": len(audit.get("caption_inputs", [])),
            "captions_complete": sum(s.get("status") == "complete"
                                     for s in result.get("segments", [])),
        },
        "generation": {"segment": generation_stats(seg_events),
                       "caption": generation_stats(cap_events)},
        "tokens": {"segment": token_stats(seg_events), "caption": token_stats(cap_events)},
        "model_calls": model_calls(events, result),
        # accel.identity records from the raw log (output sha256 from the GPU lock's
        # ``generate_output`` line; None where no lock printed one, e.g. NVIDIA unmodified)
        "identity_calls": identity_records(lines),
    }
    return result


def _utc_seconds(value: str | None) -> float | None:
    from datetime import datetime

    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def nvidia_result(result: dict) -> dict:
    """NVIDIA's dict without our measurements (for parity comparisons)."""
    return {k: v for k, v in result.items() if k != "flora_reference"}


# --- model-input identity (docs/exact-worker.md) ---------------------------------------------
#
# NVIDIA's ``models.generate`` logs ``generation_start`` with ``input_tensor_digest`` (sha256 over
# every input tensor's dtype, shape and bytes) before each model call, and ``generation_end``
# (token counts, ``complete``) after it. W3-A showed NVIDIA's own sequential pipeline is not run-to-
# run deterministic on the H200 (identical inputs, different text), so identity of a wrap is
# proven on these input digests. Outputs are paired from NVIDIA's result: the k-th complete TempSeg
# call (budget 1024) is ``pipeline.temporal_windows[k].output`` (NVIDIA appends after each
# successful ``generate``), the k-th complete Captioner call (budget 512) is the k-th ``complete``
# caption (calls follow plan order, which is segment order).

TEMPSEG_BUDGET, CAPTION_BUDGET = 1024, 512
# NVIDIA's ``generate`` stops at ``max_time=300`` s (``models.py:179``): a call near that wall
# clock can be cut at a load-dependent token count (review 4 M4), so its output is timing-, not
# input-determined. Such calls are flagged ``timing_dependent`` in ``model_calls``.
NVIDIA_MAX_TIME_S = 300.0
TIMING_FLAG_S = 150.0


def timing_dependent(call: dict) -> bool:
    """A call that ran long enough to be near NVIDIA's ``max_time``, or stopped incomplete
    before its token budget (a time stop, not an EOS / budget stop)."""
    elapsed = call.get("elapsed_s")
    if isinstance(elapsed, int | float) and elapsed > TIMING_FLAG_S:
        return True
    gen, limit = call.get("generated_tokens"), call.get("output_limit")
    return (call.get("complete") is False and isinstance(gen, int) and isinstance(limit, int)
            and gen < limit)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def model_calls(events: list[dict], result: dict | None = None) -> list[dict]:
    """One record per NVIDIA ``generate()`` call of an episode, in call order."""
    calls: list[dict] = []
    open_call: dict | None = None
    for e in events:
        kind = e.get("event")
        if kind == "generation_start":
            open_call = {
                "model": "tempseg" if e.get("output_limit") == TEMPSEG_BUDGET else "captioner",
                "input_digest": e.get("input_tensor_digest"),
                "input_tokens": e.get("input_tokens"),
                "output_limit": e.get("output_limit"),
            }
            calls.append(open_call)
        elif kind == "generation_end" and open_call is not None:
            open_call.update(generated_tokens=e.get("generated_tokens"),
                             complete=bool(e.get("complete")), elapsed_s=e.get("elapsed_s"))
            open_call["timing_dependent"] = timing_dependent(open_call)
            open_call = None
    if result is not None:
        windows = [w.get("output") for w in result.get("pipeline", {}).get("temporal_windows", [])]
        captions = [s.get("caption") for s in result.get("segments", [])
                    if s.get("status") == "complete"]
        outs = {"tempseg": iter(windows), "captioner": iter(captions)}
        for c in calls:
            if c.get("complete"):
                text = next(outs[c["model"]], None)
                c["output_sha256"] = _sha(text) if isinstance(text, str) else None
            else:
                c["output_sha256"] = None
    return calls


def _same_output(a: dict, b: dict) -> bool:
    return (a.get("complete"), a.get("output_sha256"), a.get("generated_tokens")) == (
        b.get("complete"), b.get("output_sha256"), b.get("generated_tokens"))


def compare_model_inputs(a: list[dict], b: list[dict]) -> dict:
    """Model-input identity of two runs of one episode (``model_calls`` lists, call order).

    ``strict``: every call's input digest equal, same number of calls.
    ``gate_ok`` (the acceptance gate): no call has a different input while every earlier call
    (inputs and outputs) agreed; calls after the first *output* difference are downstream of
    NVIDIA's own GPU non-determinism and only reported.
    """
    from collections import Counter

    da, db = [c.get("input_digest") for c in a], [c.get("input_digest") for c in b]
    first_out_diff = None
    violation = None
    gated = 0
    for k in range(min(len(a), len(b))):
        if da[k] != db[k]:
            violation = k
            break
        gated += 1
        if not _same_output(a[k], b[k]):
            first_out_diff = k
            break
    if violation is None and first_out_diff is None and len(a) != len(b):
        violation = min(len(a), len(b))  # same everything, but one run made more calls
    pairs = [(x, y) for x, y in zip(a, b, strict=False)
             if x.get("input_digest") == y.get("input_digest")]
    return {
        "calls_a": len(a),
        "calls_b": len(b),
        "strict": da == db,
        "multiset": Counter(da) == Counter(db),
        "gate_ok": violation is None,
        "gate_violation_call": violation,
        "gated_calls": gated,
        "first_output_difference_call": first_out_diff,
        "same_input_pairs": len(pairs),
        "same_input_different_output": sum(not _same_output(x, y) for x, y in pairs),
    }


def determinism_mode(result: dict) -> str:
    """The D-1 mode a result was produced in (results from before D-1: ``nvidia``)."""
    det = result.get("flora_reference", {}).get("determinism") or {}
    return str(det.get("mode") or "nvidia")


class ModeMismatch(ValueError):
    """Two runs in different D-1 determinism modes are different references (never compared)."""


def check_same_mode(a: dict[str, dict], b: dict[str, dict]) -> str | None:
    """Raise :class:`ModeMismatch` unless every result of both runs has the same mode."""
    modes = {determinism_mode(r) for r in (*a.values(), *b.values())}
    if len(modes) > 1:
        raise ModeMismatch(f"runs mix determinism modes {sorted(modes)}: a nvidia-mode and a "
                           "deterministic-mode reference are never compared (docs/determinism.md)")
    return next(iter(modes), None)


def identity_summary(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Per-episode ``compare_model_inputs`` + output agreement for two ``{episode: result}``.

    Refuses runs in different determinism modes. ``complete_coverage``: every episode is in both
    runs and every call of both was gated (review 4 L5 / M6); ``timing_dependent_calls`` counts
    calls near NVIDIA's ``max_time`` in either run (M4)."""
    mode = check_same_mode(a, b)
    episodes = {}
    for ep in sorted(set(a) | set(b)):
        if ep not in a or ep not in b:
            episodes[ep] = {"missing_in": "a" if ep not in a else "b"}
            continue
        ca = a[ep].get("flora_reference", {}).get("model_calls", [])
        cb = b[ep].get("flora_reference", {}).get("model_calls", [])
        row = compare_model_inputs(ca, cb)
        row["timing_dependent_calls"] = sum(bool(c.get("timing_dependent")) for c in ca + cb)
        row["output_identical"] = comparable_result(a[ep]) == comparable_result(b[ep])
        row["digest_a"], row["digest_b"] = result_digest(a[ep])[:12], result_digest(b[ep])[:12]
        if not row["output_identical"]:
            row["diff_paths"] = diff_paths(comparable_result(a[ep]), comparable_result(b[ep]),
                                           limit=5)
        episodes[ep] = row
    both = [r for r in episodes.values() if "missing_in" not in r]
    calls = sum(max(r["calls_a"], r["calls_b"]) for r in both)
    gated = sum(r["gated_calls"] for r in both)
    return {
        "episodes": episodes,
        "determinism_mode": mode,
        "compared": len(both),
        "missing": len(episodes) - len(both),
        "coverage": round(gated / calls, 4) if calls else 0.0,
        "complete_coverage": bool(both) and len(both) == len(episodes) and gated == calls,
        "timing_dependent_calls": sum(r["timing_dependent_calls"] for r in both),
        "gate_ok": bool(both) and all(r["gate_ok"] for r in both),
        "strict_episodes": sum(r["strict"] for r in both),
        "output_identical_episodes": sum(r["output_identical"] for r in both),
        "gated_calls": sum(r["gated_calls"] for r in both),
        "calls_a": sum(r["calls_a"] for r in both),
        "same_input_pairs": sum(r["same_input_pairs"] for r in both),
        "same_input_different_output": sum(r["same_input_different_output"] for r in both),
    }


# --- transport across the Modal boundary (review2-wsd M4) ------------------------------------


class NvidiaError(RuntimeError):
    """An exception raised inside NVIDIA's (or our) code in the worker, carried to the client.

    NVIDIA's exception classes live in the aliased ``flora_nv_*`` modules that do not exist in the
    client process, so they cannot be unpickled there. This builtin-based class carries the
    original qualified type name and message and pickles by value."""

    def __init__(self, type_name: str, message: str):
        super().__init__(f"{type_name}: {message}")
        self.type_name, self.message = type_name, message

    def __reduce__(self):
        return (type(self), (self.type_name, self.message))


class NvidiaInputError(NvidiaError):
    """NVIDIA rejected the input (``InputError``, ``InputErrorForModel``) or ours did."""


_INPUT_ERRORS = ("InputError", "InputErrorForModel")


def transport_error(e: BaseException) -> NvidiaError:
    if isinstance(e, NvidiaError):
        return e
    cls = type(e)
    name = f"{cls.__module__}.{cls.__qualname__}"
    kind = NvidiaInputError if cls.__name__ in _INPUT_ERRORS or isinstance(e, ValueError) \
        else NvidiaError
    return kind(name, str(e))


def new_run_id(prefix: str = "exact") -> str:
    """Unique, path-safe run id: ``<prefix>-<UTC second>-<8 hex>`` (review2-wsd L1)."""
    import uuid

    return safe_name(f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
                     f"{uuid.uuid4().hex[:8]}")


# --- the `exact` profile: which proven speedups are on ---------------------------------------


@dataclass(frozen=True)
class ExactProfile:
    """Every proven speedup of the ``exact`` worker, individually toggleable (and recorded).

    - ``accel``: OPT-1 runtime patches (``docs/accel.md``), any subset of ``accel.ALL_PATCHES``;
    - ``captioner_bf16``: OPT-2 bit-identical BF16 Captioner (``docs/captioner-bf16.md``);
    - ``linked_layout``: W3-B no-copy model cache (``warm.build_linked_layout``); off = WS-D copy;
    - ``prefetch``: W3-B t=0 parallel read of the weights into the page cache;
    - ``overlap_hash``: W3-B NVIDIA's FP32 Captioner manifest hash while TempSeg loads (only
      meaningful with ``captioner_bf16`` off; NVIDIA's ``_VERIFIED_MANIFESTS`` then reuses it);
    - ``gpu_thread``: Perf-1 every NVIDIA ``generate`` on one persistent thread
      (``accel.gpu_thread``; fidelity-neutral, keeps torch's thread-local cuDNN graphs warm);
    - ``concurrency``: W3-A episode threads under one GPU lock (1 = NVIDIA's sequential order).
    """

    accel: tuple[str, ...] = ("decode_once", "transport_pool", "png_inmem", "geometry_pool")
    captioner_bf16: bool = True
    linked_layout: bool = True
    prefetch: bool = True
    overlap_hash: bool = True
    gpu_thread: bool = True
    concurrency: int = 4

    FEATURES = ("captioner_bf16", "linked_layout", "prefetch", "overlap_hash", "gpu_thread")

    @classmethod
    def from_spec(cls, disable: str = "", concurrency: int | None = None) -> ExactProfile:
        """``disable``: comma-separated feature or accel patch names to turn off."""
        off = {s.strip() for s in disable.split(",") if s.strip()}
        base = cls()
        unknown = off - set(cls.FEATURES) - set(base.accel)
        if unknown:
            raise ValueError(f"unknown exact-profile toggles {sorted(unknown)}")
        return cls(
            accel=tuple(p for p in base.accel if p not in off),
            **{f: f not in off for f in cls.FEATURES},
            concurrency=base.concurrency if concurrency is None else int(concurrency),
        )

    def provenance(self) -> dict:
        return {
            "schema": "flora_exact_profile_v1",
            "accel": list(self.accel),
            "captioner_bf16": self.captioner_bf16,
            "linked_layout": self.linked_layout,
            "prefetch": self.prefetch,
            "overlap_hash": self.overlap_hash and not self.captioner_bf16,
            "gpu_thread": self.gpu_thread,
            "concurrency": self.concurrency,
            "gpu_lock": True,
        }


# --- records (flora_labels_v1) ---------------------------------------------------------------


def stage_records(result: dict, *, uid: str, dataset: str, episode_id: str, provenance,
                  audit_ref: str | None = None, certified: bool = True,
                  uncertified_reason: str | None = None) -> list:
    """NVIDIA's result as ``segment`` and ``caption`` ``StageRecord``s (``records.py``).

    ``exact`` is the reference by construction, so records are certified unless the episode is
    outside NVIDIA's reference envelope (``certified=False`` with ``uncertified_reason``, D13)."""
    from flora_deploy.records import Segment, StageRecord

    rows = result.get("segments", [])

    def segs(with_caption: bool):
        return tuple(
            Segment(event_id=f"event_{i:04d}", start_s=float(s["start_s"]),
                    end_s=float(s["end_s"]),
                    status="complete" if s.get("status") == "complete"
                    else "unsupported_target_frames",
                    caption=s.get("caption") if with_caption else None,
                    segmentation_provenance=s.get("segmentation_provenance"))
            for i, s in enumerate(rows))

    return [StageRecord(uid=uid, dataset=dataset, episode_id=episode_id, stage=stage,
                        provenance=provenance, certified=certified,
                        uncertified_reason=None if certified else uncertified_reason,
                        segments=segs(cap), audit_ref=audit_ref)
            for stage, cap in (("segment", False), ("caption", True))]


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
#   borrowed" or race. Each running episode therefore leases its own ``copy.deepcopy`` of each
#   processor (``ProcessorClones`` pool); ``generate`` gets that same clone from NVIDIA's code.
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


def locked_generate(generate: Callable, lock, *, emit_output: bool = False) -> Callable:
    """NVIDIA's ``generate`` run under ``lock`` (an ``observe.TimedLock``) and timed.

    ``emit_output``: always print the ``FLORA_OBSERVE generate_output`` line (otherwise only
    while an ``observe`` recorder is active), so ``accel.identity`` can pair every call."""
    import functools
    import hashlib

    from flora_deploy.observe.events import current_recorder, emit_json
    from flora_deploy.observe.timers import timed

    inner = timed(generate, GPU_LOCK_LABEL)

    @functools.wraps(generate)
    def generate_locked(*args, **kwargs):
        with lock.held(GPU_LOCK_LABEL):
            text = inner(*args, **kwargs)
            if emit_output or current_recorder() is not None:
                # Printed right after NVIDIA's generation_end line (still under the lock), so a
                # log pairs each call's input_tensor_digest with a hash of its output text.
                emit_json({"kind": "generate_output", "sha256": hashlib.sha256(
                    str(text).encode()).hexdigest()})
            return text

    generate_locked.__flora_gpu_lock__ = lock
    return generate_locked


def install_gpu_lock(nv, lock, patcher, *, emit_output: bool = False) -> None:
    """Route every NVIDIA model call through ``lock``.

    ``pipeline.py`` does ``from .models import generate`` at import time, so the name its
    functions call is ``pipeline.generate``; we replace that module attribute in our process
    (NVIDIA's files are untouched)."""
    if getattr(nv.pipeline.generate, "__flora_gpu_lock__", None) is not None:
        raise RuntimeError("NVIDIA generate() already runs under a GPU lock")
    patcher.patch(nv.pipeline, "generate",
                  lambda orig: locked_generate(orig, lock, emit_output=emit_output))


class ProcessorSet:
    """One episode's private processors: a lazy ``copy.deepcopy`` of each shared one."""

    def __init__(self, pool: ProcessorClones):
        self._pool = pool
        self._procs: dict[str, tuple[Any, Any]] = {}

    def get(self, kind: str, proc: Any) -> Any:
        entry = self._procs.get(kind)
        if entry is None or entry[0] is not proc:
            entry = self._procs[kind] = (proc, self._pool._clone(proc))
        return entry[1]


class ProcessorClones:
    """Pool of ``ProcessorSet`` leases (see the audit above).

    Each running episode leases a set no other episode uses at the same time; sets go back to
    the pool afterwards, so a long-lived worker pays the deepcopy (~3 s per Captioner processor)
    once per concurrency slot, not per batch."""

    def __init__(self, copier: Callable[[Any], Any] | None = None):
        import copy
        import threading

        self._copier = copier or copy.deepcopy
        self._lock = threading.Lock()
        self._free: list[ProcessorSet] = []
        self.clones = 0
        self.seconds = 0.0

    def _clone(self, proc: Any) -> Any:
        t = time.perf_counter()
        clone = self._copier(proc)
        with self._lock:
            self.clones += 1
            self.seconds += time.perf_counter() - t
        return clone

    @contextmanager
    def lease(self) -> Iterator[ProcessorSet]:
        with self._lock:
            procs = self._free.pop() if self._free else ProcessorSet(self)
        try:
            yield procs
        finally:
            with self._lock:
                self._free.append(procs)

    def stats(self) -> dict:
        with self._lock:
            return {"clones": self.clones, "seconds": round(self.seconds, 3),
                    "sets": len(self._free)}


class EpisodeSession:
    """Per-episode view of a shared ``CachedSession``: same resident models, this thread's
    processor clones. Duck-types NVIDIA ``models.Session`` like ``CachedSession`` does."""

    def __init__(self, base, clones: ProcessorSet | None = None):
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
    """Compare two ``{episode: result}`` maps on ``comparable_result`` (same determinism mode
    only: :class:`ModeMismatch` otherwise)."""
    check_same_mode(a, b)
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
    clones: ProcessorClones | None = None,
    sampler=None,
    progress: Callable[[str, float, str], None] | None = None,
    accel=None,
    on_start: Callable[[dict], None] | None = None,
    on_done: Callable[[dict, dict | None, NvidiaError | None], None] | None = None,
) -> dict:
    """Run ``annotate_nvidia`` over ``episodes`` with ``concurrency`` worker threads.

    ``session``: the resident ``CachedSession`` (``install_nvidia_entry`` active), or ``None`` to
    run NVIDIA's own fresh ``Session()`` per episode (sequential only). ``accel``: an active
    ``accel.ActivePatches``; each episode runs inside its ``episode()`` scope.

    ``episodes`` items: ``{"episode", "videos", "profile", ["long_windows"], ["instruction"]}``.
    ``concurrency=1`` runs them in order in the calling thread (the sequential baseline). With
    ``on_start(ep)`` / ``on_done(ep, result, error)`` run in the episode's own thread, before
    and after it (``error``: the ``NvidiaError`` it raised); an exception they raise is not an
    episode error, it propagates (the batch fails). With
    ``concurrency>1`` NVIDIA's ``generate`` must already run under ``gpu_lock``
    (``install_gpu_lock``); each running episode leases private processor clones from ``clones``
    (a long-lived ``ProcessorClones`` pool; a fresh one if omitted) unless
    ``clone_processors=False``.
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
        if session is None:
            raise RuntimeError("concurrency > 1 needs the resident session")
        lock = getattr(nv.pipeline.generate, "__flora_gpu_lock__", None)
        if lock is None or (gpu_lock is not None and lock is not gpu_lock):
            raise RuntimeError("concurrency > 1 needs NVIDIA generate() under the GPU lock")
        if not isinstance(getattr(nv.api, "_REQUEST_LOCK", None), _NoLock):
            raise RuntimeError("concurrency > 1 needs install_nvidia_entry(request_lock=False)")
        gpu_lock = lock
    if session is None:
        clone_processors = False
    if clone_processors is None:
        clone_processors = concurrency > 1
    if clone_processors and clones is None:
        clones = ProcessorClones()
    if not clone_processors:
        clones = None
    clone_stats0 = clones.stats() if clones else None
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
        if on_start is not None:
            on_start(ep)
        try:
            # accel.episode() is entered here, inside the episode's own thread (docs/accel.md)
            with recording(rec), (clones.lease() if clones else nullcontext()) as procs, \
                    (accel.episode() if accel is not None else nullcontext()):
                result = annotate_nvidia(
                    nv, None if session is None else EpisodeSession(session, procs),
                    ep["videos"], ep["profile"], ep.get("long_windows", False),
                    ep.get("instruction", ""), progress=report,
                )
        except Exception as e:  # noqa: BLE001 - one bad episode must not sink the batch
            err = transport_error(e)
            if on_done is not None:
                on_done(ep, None, err)
            return None, f"{err.type_name}: {err.message}"
        ref = result["flora_reference"]
        ref["batch"] = {"concurrency": concurrency, "end_s": round(time.perf_counter() - t0, 3)}
        if accel is not None:
            ref["accel"] = accel.provenance()
        if rec is not None:
            ref["observe"] = episode_observe(rec, result, t0,
                                             gpu_lock.name if gpu_lock else GPU_LOCK_NAME)
        if on_done is not None:
            on_done(ep, result, None)
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
        "processor_clones": {
            "new_clones": clones.stats()["clones"] - clone_stats0["clones"],
            "clone_s": round(clones.stats()["seconds"] - clone_stats0["seconds"], 3),
            "sets": clones.stats()["sets"],
        } if clones else None,
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
    overwrite: bool = False,
) -> Path:
    """Write ``result`` to a local temp file, copy it whole to
    ``<golden_root>/reference_runs/<run_id>/<episode>.json`` (via a ``.partial`` + rename, so a
    reader never sees a partial file), then ``commit()`` the Volume. Never keeps a file open on
    the Volume. Refuses to replace an existing result unless ``overwrite`` (like NVIDIA's CLI)."""
    dest = Path(golden_root) / result_relpath(run_id, episode)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} exists (pass overwrite=True to replace it)")
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


# --- golden episodes (docs/golden-manifest-v1.md) --------------------------------------------

# S1 identity proof: short episodes spanning the human / single-view yam / 3-view yam / dex pools.
IDENTITY_EPISODES = (
    "human/egodex_add_remove_lid/episode_000014",
    "human/egodex_make_sandwich/episode_000002",
    "single_view_robot/wgo/robointer_droid_000010",
    "yam/aloha_static_cups_open/episode_000000",
    "yam/aloha_static_towel/episode_000000",
    "dex/g1_dex3_toasted_bread/episode_000034",
)


def golden_name(episode_id: str) -> str:
    """Path-safe result name of a golden manifest id (``pool/dataset/slug``)."""
    return safe_name(episode_id.replace("/", "__"))


def golden_episode_inputs(manifest: dict, ids, root: Path | PurePosixPath, *,
                          instruction: bool = True) -> list[dict]:
    """``annotate_batch`` episodes for golden manifest ids (camera files under ``root``).

    ``instruction``: pass the episode's dataset task text as NVIDIA's ``instruction`` (blank when
    the dataset has none), so both of NVIDIA's caption prompt paths are exercised."""
    by_id = {e["id"]: e for e in manifest.get("episodes", [])}
    out = []
    for i in ids:
        e = by_id.get(i)
        if e is None:
            raise KeyError(f"{i!r} not in the golden manifest")
        if not e.get("oracle_eligible") or e.get("pool") == "reward":
            raise ValueError(f"{i!r} is not an oracle-eligible TempSeg/Captioner episode")
        videos = {role: str(PurePosixPath(root) / e["path"] / f["file"])
                  for role, f in e["files"].items() if f.get("file")}
        task = (e.get("task") or "").strip() if instruction else ""
        out.append({"episode": golden_name(i), "id": i, "pool": e["pool"],
                    "profile": e["profile"], "long_windows": bool(e.get("yam32")),
                    "instruction": task if len(task) <= 2000 else "",
                    "duration_s": float(e.get("duration_s") or 0.0), "videos": videos})
    return out
