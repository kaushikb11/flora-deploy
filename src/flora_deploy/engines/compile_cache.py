"""Keyed, verified vLLM compile caches on ``flora-cache`` (docs/cold-start.md, "Warm start").

A fresh ``balanced`` worker spends 4-5 minutes compiling vLLM (torch.compile / Inductor,
Triton JIT, FlashInfer and CUDA JIT). This module lets **one** container per cache key publish
what it compiled, and every later container with the same key load it, without ever trusting a
cache it cannot verify.

**Key** (:func:`key_factors`): everything that can make a compiled artifact wrong for an engine:
the ``img_vllm`` fingerprint, the declared vLLM / torch / Triton / CUDA versions, the engine's
compile-relevant arguments (:func:`compile_kwargs`: model path, dtype, max lengths, scheduler
limits, ``gdn_prefill_backend``, ``enforce_eager``, ...), the ``dt_bias`` plugin flag and the
source of the two flora modules mixed into vLLM's workers, the derived weights' content digest
and the GPU type. vLLM keys its own cache below that root too (``VllmConfig.compute_hash``), but
its AOT path does **not** key the GPU (``compilation/caching.py:aot_compile_hash_factors`` takes
no Inductor/system factors unless ``VLLM_USE_MEGA_AOT_ARTIFACT``, which is off on torch 2.11), so
the GPU must be ours. ``logprobs_mode`` is dropped and ``max_logprobs`` defaulted to vLLM's 20:
``ModelConfig.compute_hash`` ignores the first and hashes the second by value, so the lab's
Gate A engines (``raw_logits``, 20) compile exactly what the workers compile.

**Layout** (under the Volume's ``balanced-cc/``)::

    <key name>/MANIFEST.json        the commit record: key factors, runtime facts, every file
    <key name>/<publish id>/vllm/...  one immutable tree per publish (triton/, cuda/, xdg/ too)
    <key name>/<publish id>/cache.tar the same files as one uncompressed tar (``archive``)

**Publish** (:func:`publish`): after ``@modal.enter`` and a shard's generates, a container
whose cache was not loaded from a verified manifest takes a **lease** (``Leases``: a
``modal.Dict`` with put-if-absent, generation-numbered so an expired holder can be replaced
by exactly one successor), uploads its local tree and a tar of it (:data:`ARCHIVE`, written to
a local temp file first and uploaded as one file, atomic per file on a v2 Volume) to a fresh
``<publish id>/`` (nobody else writes there), re-checks the lease, and only then writes
``MANIFEST.json`` (one file, atomic on a v2 Volume). A partial upload is invisible: without a
manifest naming it, a tree is never read. The worker's ``flora-cache`` mount stays read-only
(M7); uploads go through the Volume client API (``VolumeSink``). The per-file tree stays for
readers that predate the archive. A key whose manifest has no archive (published before it)
may be republished once, with one, under its own lease (``<key name>.archive``).

**Load** (:func:`seed`): read the manifest from the read-only mount, check schema, key digest,
factors and the container's own runtime versions, then restore every listed file to local
disk while hashing it, into a staging directory renamed into place only when every file
matched. With an ``archive`` that is one sequential read of ``cache.tar`` (whose size and
sha256 are checked too, and whose members must be exactly the manifest's files); without one,
the files are copied in parallel (:data:`COPY_WORKERS`). Any problem (missing, unparsable,
other key, other runtime, unsafe path, size or sha256 mismatch, I/O error) returns a ``cold``
status with the reason, and the engine compiles from scratch into an empty directory.
Nothing here raises into the worker.

**Restore guard** (:func:`expected_restore_s`, :func:`max_restore_s`). On 2026-10-22 a warm
L40S seed copied 1,932 files / 322 MB one at a time in ~405 s (~0.21 s per file, latency
bound) and the warm engine then took ~140 s more: 550 s, against ~290 s for a cold compile
(``FLORA_COMPILE_CACHE=off``). A restore pays only while restore + warm build < cold build, so
:func:`seed` compiles cold without restoring when the manifest's expected restore time is over
:data:`MAX_RESTORE_S` (290 − 140 = 150 s; ``FLORA_COMPILE_CACHE_MAX_RESTORE_S`` overrides it),
and gives up a restore that is still running at that limit. Every restore logs its format,
files, bytes, expected and actual seconds.

**Legacy lab caches** (:func:`seed_legacy`) are no longer copied: the unkeyed
``balanced/captioner`` copy (1,897 files, 403 MB) took 294 s, as long as compiling.

**Determinism: a warm start is not output-neutral** (review 13 W1). A compile cache holds the
artifacts the publishing container's compile produced, including choices that are **not** a
function of the key: torch 2.11 Inductor benchmarks several configs of some reduction kernels
at their first launch, even with ``max_autotune`` off (``runtime/triton_heuristics.py``
``_reduction_configs``: a ``DEFAULT`` hint, 3-D tiling, or the persistent-vs-looped RBLOCK
extra config, which changes the accumulation order), and caches the winner as a
``.best_config`` next to the kernel (``autotune_local_cache=True``). A later process reads it
back instead of benchmarking. So a **warm** container replays the publisher's timing-based
picks, while a **cold** one benchmarks afresh and may pick another config: a different
reduction order and possibly bf16-level drift in greedy outputs. (Triton's runtime
``@autotune`` in the FLA/GDN kernels is re-benchmarked in every process, warm or cold, since
vLLM 0.24 does not set ``cache_results``: noise either way. ``TORCHINDUCTOR_DETERMINISTIC``
would skip the benchmarking, but it changes kernels, so it would enter the key.)

The key therefore does **not** carry the cache: which cache a container ran is **recorded**
instead (:meth:`CacheStatus.identity`: ``warm:<key>:<manifest sha256[:16]>``,
``legacy:<key>``, ``cold`` or ``cold:forced``) in the startup facts, each shard's audit row and
every record's ``provenance.compile_cache``, and the certificate states the mix its evidence
covers (``validate.certify``, ``compile_cache``). ``FLORA_COMPILE_CACHE=off`` forces a
container cold (no seed, no publish) so a certified run can include an independent cold compile
(``docs/certification.md``, "Warm and cold compiles"). Not in the certificate logic digest
(``config/logic.py``): this module decides which artifacts load, recorded per record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

SCHEMA = "flora-compile-cache/v1"
ROOT = "balanced-cc"  # under the flora-cache Volume
MANIFEST = "MANIFEST.json"
TREE_DIRS = ("vllm", "triton", "cuda", "xdg")
LEASE_DICT = "flora-compile-cache-leases"
LEASE_TTL_S = 30 * 60
MAX_GENERATIONS = 16
VLLM_DEFAULT_MAX_LOGPROBS = 20  # vllm/config/model.py ModelConfig.max_logprobs (0.24.0)
# Engine kwargs vLLM 0.24.0 leaves out of every compile hash (ModelConfig.compute_hash
# ``ignored_factors``) and that only change what a request *returns*. Anything else stays in
# the key, even where vLLM ignores it: a spurious miss costs one compile, a false hit is unsafe.
COMPILE_IRRELEVANT = ("logprobs_mode",)
# flora code that runs inside vLLM's workers: the dt_bias plugin (changes parameter dtypes,
# hence the graph) and the read-only worker extension.
WORKER_SOURCES = ("engines/patches/dt_bias_fp32.py", "engines/worker_ext.py")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SKIP_SUFFIXES = (".lock", ".tmp", ".partial")
NO_CACHE = "no published cache for this key"
FORCED_COLD = "forced cold (FLORA_COMPILE_CACHE=off)"
ARCHIVE = "cache.tar"  # the publish's files as one tar, next to its per-file tree
ARCHIVE_LEASE_SUFFIX = ".archive"  # lease of a republish that adds an archive
# Restore guard (module docstring). Measured 2026-10-22 on L40S (scale-lite-102223538c21 and
# cold-human-a2): cold engine build ~290 s; warm engine build after a restore ~140 s; per-file
# Volume copy ~0.21 s per file. Throughput of one sequential Volume read is an assumption
# (conservative); the deadline bounds the cost of being wrong.
COLD_BUILD_S = 290.0
WARM_BUILD_S = 140.0
MAX_RESTORE_S = COLD_BUILD_S - WARM_BUILD_S
MAX_RESTORE_ENV = "FLORA_COMPILE_CACHE_MAX_RESTORE_S"
FILE_LATENCY_S = 0.21
READ_BYTES_PER_S = 25e6
COPY_WORKERS = 16


class CacheInvalid(RuntimeError):
    """A published cache that must not be used (the reason is the message)."""


# ---------------------------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------------------------


def _lock_version(package: str, lock: Path | None = None) -> str | None:
    if lock is None:
        from flora_deploy.modal_app.images import lock_path

        lock = lock_path("img_vllm")
    try:
        text = lock.read_text()
    except OSError:
        return None
    m = re.search(rf"^{re.escape(package)}==([^\s\\]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def declared_versions(lock: Path | None = None) -> dict[str, str | None]:
    """What the image is built to contain (pins and the img_vllm lock): computable on a laptop,
    so ``flora plan`` derives the same key as the worker."""
    from flora_deploy.config.pins import RUNTIME

    return {"vllm": RUNTIME["vllm"], "torch": _lock_version("torch", lock),
            "triton": _lock_version("triton", lock), "cuda_base": RUNTIME["cuda_base_vllm"]}


def runtime_versions() -> dict[str, str | None]:
    """What this container actually runs (no CUDA initialisation: vLLM must fork cleanly)."""
    from importlib import metadata

    def ver(pkg: str) -> str | None:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    cuda = None
    try:
        import torch

        cuda = torch.version.cuda
    except Exception:  # noqa: BLE001 - recorded as unknown, which then never matches
        pass
    return {"vllm": ver("vllm"), "torch": ver("torch"), "triton": ver("triton"),
            "torch_cuda": cuda}


def compile_kwargs(engine_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The engine kwargs that key a compile: :data:`COMPILE_IRRELEVANT` dropped,
    ``max_logprobs`` at vLLM's default when unset (vLLM hashes the value, not its presence)."""
    out = {k: v for k, v in engine_kwargs.items() if k not in COMPILE_IRRELEVANT}
    out.setdefault("max_logprobs", VLLM_DEFAULT_MAX_LOGPROBS)
    return out


def worker_sources_digest(root: Path | None = None) -> str:
    from flora_deploy.config.logic import package_root, source_digest

    return source_digest(WORKER_SOURCES, root or package_root())


def key_factors(*, engine_kind: str, gpu: str, engine_kwargs: Mapping[str, Any],
                dtbias_fp32: bool, model_content: str | None, image_fp: str,
                versions: Mapping[str, Any], worker_sources: str) -> dict[str, Any]:
    from flora_deploy.engines.offline_llm import normalize_gpu

    return {"schema": SCHEMA, "engine_kind": engine_kind, "gpu": normalize_gpu(gpu),
            "image": image_fp, "versions": dict(versions),
            "engine_kwargs": compile_kwargs(engine_kwargs), "dtbias_fp32": bool(dtbias_fp32),
            "model_content": model_content, "worker_sources": worker_sources}


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class CacheKey:
    factors: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canon(self.factors).encode()).hexdigest()

    @property
    def name(self) -> str:
        """``tempseg-L40S-<24 hex>``: readable, and safe as a path component and Dict key."""
        f = self.factors
        return f"{f['engine_kind']}{'-dtbias' if f['dtbias_fp32'] else ''}-{f['gpu']}-" \
            f"{self.digest[:24]}"


def spec_key(spec: Any, *, model_content: str | None, image_fp: str,
             versions: Mapping[str, Any] | None = None,
             worker_sources: str | None = None) -> CacheKey:
    """The key of an ``offline_llm.EngineSpec``."""
    return CacheKey(key_factors(
        engine_kind=spec.kind, gpu=spec.gpu, engine_kwargs=spec.engine_kwargs(),
        dtbias_fp32=spec.dtbias_fp32, model_content=model_content, image_fp=image_fp,
        versions=declared_versions() if versions is None else versions,
        worker_sources=worker_sources_digest() if worker_sources is None else worker_sources))


def worker_key(shard_kind: str, gpu: str, *, image_fp: str | None = None,
               model_content: str | None = None, **kw: Any) -> CacheKey:
    """The key a deployed ``segment`` / ``caption`` worker on ``gpu`` uses (computable on a
    laptop, for :func:`published`): its model path and the pinned derived content digest."""
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY, PINS_BY_KEY
    from flora_deploy.engines.offline_llm import spec_for
    from flora_deploy.layout import derived_path
    from flora_deploy.prep.derived import DERIVED_BUILDS

    name = {"segment": "tempseg_merged", "caption": "captioner_bf16"}[shard_kind]
    b = DERIVED_BUILDS[name]
    model = str(derived_path(b.dirname, PINS_BY_KEY[b.source_key]))
    if model_content is None:
        pin = DERIVED_PINS_BY_KEY.get(b.dirname)
        model_content = pin.content_sha256 if pin else None
    if image_fp is None:
        from flora_deploy.modal_app.images import image_fingerprint

        image_fp = image_fingerprint("img_vllm")
    return spec_key(spec_for(shard_kind, model, gpu), model_content=model_content,
                    image_fp=image_fp, **kw)


# ---------------------------------------------------------------------------------------------
# Files and manifests
# ---------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def tree_files(local: Path) -> dict[str, Path]:
    """Regular files under ``local``'s :data:`TREE_DIRS` (relative posix path -> file). Lock and
    temp files are skipped; so are symlinks (a cache needs none, and one could point anywhere)."""
    out: dict[str, Path] = {}
    for top in TREE_DIRS:
        base = local / top
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_symlink() or not p.is_file() or p.name.endswith(_SKIP_SUFFIXES):
                continue
            out[p.relative_to(local).as_posix()] = p
    return out


def safe_rel(rel: str) -> str:
    """A manifest path: relative, normalised, under one of :data:`TREE_DIRS`."""
    p = PurePosixPath(rel)
    if (not rel or p.is_absolute() or ".." in p.parts or "." in p.parts or "\\" in rel
            or "\0" in rel or p.parts[0] not in TREE_DIRS or len(p.parts) < 2):
        raise CacheInvalid(f"unsafe path in manifest: {rel!r}")
    return p.as_posix()


def write_archive(files: Mapping[str, Path], dest: Path) -> tuple[dict[str, dict], dict]:
    """Write ``files`` (relative path -> local file) as one uncompressed tar at ``dest``,
    reading each file once: ``(entries, archive)``, the manifest's per-file size / sha256 of
    exactly the bytes archived, and the archive's own ``{path, size, sha256}``."""
    import io

    entries: dict[str, dict] = {}
    with tarfile.open(dest, "w", format=tarfile.PAX_FORMAT) as tf:
        for rel, p in sorted(files.items()):
            data = Path(p).read_bytes()
            ti = tarfile.TarInfo(rel)
            ti.size, ti.mode, ti.mtime = len(data), 0o644, int(Path(p).stat().st_mtime)
            tf.addfile(ti, io.BytesIO(data))
            entries[rel] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return entries, {"path": ARCHIVE, "size": dest.stat().st_size, "sha256": _sha256(dest)}


def build_manifest(key: CacheKey, publish_id: str, files: Mapping[str, Path], *,
                   runtime: Mapping[str, Any], holder: str,
                   clock: Callable[[], float] = time.time,
                   entries: Mapping[str, dict] | None = None,
                   archive: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if entries is None:
        entries = {rel: {"size": p.stat().st_size, "sha256": _sha256(p)}
                   for rel, p in sorted(files.items())}
    out = {"schema": SCHEMA, "key": key.name, "digest": key.digest,
           "factors": dict(key.factors), "runtime": dict(runtime), "publish_id": publish_id,
           "holder": holder, "created_unix": round(clock(), 3), "files": dict(entries),
           "total_bytes": sum(e["size"] for e in entries.values())}
    if archive is not None:
        out["archive"] = dict(archive)
    return out


def check_manifest(m: Any, key: CacheKey, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Raise :class:`CacheInvalid` unless ``m`` is a complete manifest for exactly ``key`` and
    was published by a container running ``runtime``'s versions."""
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        raise CacheInvalid(f"not a {SCHEMA} manifest")
    if m.get("key") != key.name or m.get("digest") != key.digest:
        raise CacheInvalid(f"manifest is for {m.get('key')!r}, not {key.name!r}")
    if _canon(m.get("factors")) != _canon(dict(key.factors)):
        raise CacheInvalid("manifest key factors differ from this engine's")
    theirs = dict(m.get("runtime") or {})
    diff = {k: (theirs.get(k), v) for k, v in runtime.items() if theirs.get(k) != v}
    if diff:
        raise CacheInvalid(f"published on another runtime: {diff}")
    if not isinstance(m.get("publish_id"), str) or not _SAFE_NAME.match(m["publish_id"]):
        raise CacheInvalid(f"bad publish id {m.get('publish_id')!r}")
    files = m.get("files")
    if not isinstance(files, dict) or not files:
        raise CacheInvalid("manifest lists no files")
    for rel, e in files.items():
        safe_rel(rel)
        if not (isinstance(e, dict) and isinstance(e.get("size"), int) and e["size"] >= 0
                and isinstance(e.get("sha256"), str) and len(e["sha256"]) == 64):
            raise CacheInvalid(f"bad manifest entry for {rel!r}")
    a = m.get("archive")
    if a is not None and not (
            isinstance(a, dict) and isinstance(a.get("path"), str)
            and _SAFE_NAME.match(a["path"]) and isinstance(a.get("size"), int) and a["size"] > 0
            and isinstance(a.get("sha256"), str) and len(a["sha256"]) == 64):
        raise CacheInvalid(f"bad archive entry {a!r}")
    return m


def expected_restore_s(m: Mapping[str, Any]) -> float:
    """Seconds a restore of manifest ``m`` is expected to take (module docstring): one
    sequential read for an archive; per-file latency over :data:`COPY_WORKERS` plus the bytes
    otherwise."""
    files = m.get("files") or {}
    size = sum(int(e.get("size", 0)) for e in files.values())
    if m.get("archive"):
        return int(m["archive"]["size"]) / READ_BYTES_PER_S
    return len(files) * FILE_LATENCY_S / COPY_WORKERS + size / READ_BYTES_PER_S


def max_restore_s(env: Mapping[str, str] | None = None) -> float:
    """The restore limit: :data:`MAX_RESTORE_S`, or ``FLORA_COMPILE_CACHE_MAX_RESTORE_S``."""
    raw = (os.environ if env is None else env).get(MAX_RESTORE_ENV)
    try:
        return float(raw) if raw else MAX_RESTORE_S
    except ValueError:
        return MAX_RESTORE_S


# ---------------------------------------------------------------------------------------------
# Load (every container)
# ---------------------------------------------------------------------------------------------


@dataclass
class CacheStatus:
    """What :func:`seed` did. ``state``: ``warm`` (a verified manifest was loaded), ``legacy``
    (an unkeyed lab cache was copied; vLLM's own keys decide what hits) or ``cold``."""

    state: str
    key: str
    reason: str = ""
    files: int = 0
    bytes: int = 0
    seconds: float = 0.0
    publish_id: str | None = None
    published: dict[str, Any] | None = None
    manifest_sha256: str | None = None  # warm: sha256 of the MANIFEST.json bytes it loaded
    format: str | None = None  # the manifest's restore path: "archive" or "files"
    expected_s: float | None = None  # the restore guard's estimate (expected_restore_s)

    def identity(self) -> str:
        """Which compiled artifacts this container runs, for record provenance:
        ``warm:<key>:<manifest sha256[:16]>``, ``legacy:<key>``, ``cold:forced`` or ``cold``."""
        if self.state == "warm":
            return f"warm:{self.key}:{(self.manifest_sha256 or 'unknown')[:16]}"
        if self.state == "legacy":
            return f"legacy:{self.key}"
        return "cold:forced" if self.reason == FORCED_COLD else "cold"

    def as_json(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if v not in (None, "")}
        out["identity"] = self.identity()
        return out


def env_for(local: Path) -> dict[str, str]:
    """The cache env vars for a local tree, its directories created."""
    env = {"VLLM_CACHE_ROOT": str(local / "vllm"), "TRITON_CACHE_DIR": str(local / "triton"),
           "CUDA_CACHE_PATH": str(local / "cuda"), "XDG_CACHE_HOME": str(local / "xdg")}
    for p in env.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return env


def reset_local(local: Path) -> None:
    """An empty local tree (after a failed seed, or before a cold retry)."""
    shutil.rmtree(local, ignore_errors=True)
    env_for(local)


class _HashingReader:
    """A file object that hashes and counts every byte read through it."""

    def __init__(self, f: Any):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, size: int = -1) -> bytes:
        b = self.f.read(size)
        self.h.update(b)
        self.n += len(b)
        return b


def _copy_verified(fi: Any, dst: Path, rel: str, e: Mapping[str, Any]) -> int:
    """Copy ``fi`` to ``dst`` while hashing; raise unless it matches manifest entry ``e``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    got = 0
    with open(dst, "wb") as fo:
        while chunk := fi.read(1 << 20):
            h.update(chunk)
            got += len(chunk)
            fo.write(chunk)
    if got != e["size"] or h.hexdigest() != e["sha256"]:
        raise CacheInvalid(f"{rel}: {got} bytes sha256 {h.hexdigest()[:12]}, manifest "
                           f"{e['size']} bytes {e['sha256'][:12]}")
    return got


def _over(deadline: float, limit: float) -> None:
    if time.perf_counter() > deadline:
        raise CacheInvalid(f"restore still running after {limit:.0f} s (a cold compile is "
                           "faster)")


def _restore_archive(src: Path, m: Mapping[str, Any], stage: Path, deadline: float,
                     limit: float) -> tuple[int, int]:
    """One sequential read of the publish's tar into ``stage``: every member a regular file of
    the manifest, matching its entry, no member twice, none missing, and the archive's own
    size and sha256 as published."""
    files, a = m["files"], m["archive"]
    seen: set[str] = set()
    size = 0
    with open(src / a["path"], "rb", buffering=1 << 20) as raw:
        rd = _HashingReader(raw)
        with tarfile.open(fileobj=rd, mode="r|") as tf:  # type: ignore[call-overload]
            for ti in tf:
                _over(deadline, limit)
                if ti.isdir():
                    continue
                rel = safe_rel(ti.name)
                if not ti.isreg():
                    raise CacheInvalid(f"archive member {rel!r} is not a regular file")
                if rel not in files or rel in seen:
                    raise CacheInvalid(f"archive member {rel!r} is not in the manifest "
                                       "(or appears twice)")
                fi = tf.extractfile(ti)
                assert fi is not None
                size += _copy_verified(fi, stage / rel, rel, files[rel])
                seen.add(rel)
        while rd.read(1 << 20):  # the tar's end blocks: the whole file is hashed
            pass
    missing = set(files) - seen
    if missing:
        raise CacheInvalid(f"archive lacks {len(missing)} manifest file(s), e.g. "
                           f"{min(missing)!r}")
    if rd.n != a["size"] or rd.h.hexdigest() != a["sha256"]:
        raise CacheInvalid(f"{a['path']}: {rd.n} bytes sha256 {rd.h.hexdigest()[:12]}, "
                           f"manifest {a['size']} bytes {a['sha256'][:12]}")
    return len(seen), size


def _restore_files(src: Path, m: Mapping[str, Any], stage: Path, deadline: float,
                   limit: float) -> tuple[int, int]:
    """Copy a per-file publish into ``stage``, :data:`COPY_WORKERS` files at a time (a Volume
    read is latency bound per file)."""
    files = m["files"]

    def one(rel: str) -> int:
        _over(deadline, limit)
        with open(src / rel, "rb") as fi:
            return _copy_verified(fi, stage / safe_rel(rel), rel, files[rel])

    n = size = 0
    ex = ThreadPoolExecutor(max_workers=COPY_WORKERS, thread_name_prefix="cc-restore")
    try:
        for fut in as_completed([ex.submit(one, rel) for rel in files]):
            size += fut.result()
            n += 1
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return n, size


def seed(key: CacheKey, mount: Path | str, local: Path, *,
         runtime: Mapping[str, Any] | None = None,
         log: Callable[[str], None] = print,
         max_restore: float | None = None) -> CacheStatus:
    """Restore the verified cache of ``key`` from the (read-only) mount to ``local``. Never
    raises: any failure, or a restore expected (or still running) past ``max_restore``
    (:func:`max_restore_s`), leaves an empty ``local`` and returns ``cold`` with the reason
    (logged)."""
    t0 = time.perf_counter()
    base = Path(mount) / ROOT / key.name
    mpath = base / MANIFEST
    if not mpath.is_file():
        reset_local(local)
        return CacheStatus("cold", key.name, reason=NO_CACHE)
    stage = local.parent / f".{local.name}.staging-{uuid.uuid4().hex[:8]}"
    fmt: str | None = None
    expected: float | None = None
    try:
        runtime = runtime_versions() if runtime is None else runtime
        limit = max_restore_s() if max_restore is None else float(max_restore)
        raw = mpath.read_bytes()
        try:
            m = json.loads(raw)
        except ValueError as e:
            raise CacheInvalid(f"manifest is not JSON ({e})") from None
        check_manifest(m, key, runtime)
        fmt = "archive" if m.get("archive") else "files"
        expected = round(expected_restore_s(m), 1)
        if expected > limit:
            reset_local(local)
            reason = (f"restore expected {expected:.0f} s ({fmt}, {len(m['files'])} files) "
                      f"> {limit:.0f} s: a cold compile is faster")
            log(f"compile cache {key.name}: not restored, compiling cold ({reason})")
            return CacheStatus("cold", key.name, reason=reason, format=fmt, expected_s=expected,
                               seconds=round(time.perf_counter() - t0, 2))
        src = base / m["publish_id"]
        restore = _restore_archive if fmt == "archive" else _restore_files
        n, size = restore(src, m, stage, t0 + limit, limit)
        shutil.rmtree(local, ignore_errors=True)
        local.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, local)
        env_for(local)
        secs = round(time.perf_counter() - t0, 2)
        log(f"compile cache {key.name}: restored {n} files ({size / 1e6:.1f} MB, {fmt}) in "
            f"{secs:.1f} s (expected {expected:.0f} s, limit {limit:.0f} s)")
        return CacheStatus("warm", key.name, files=n, bytes=size, seconds=secs,
                           publish_id=m["publish_id"],
                           manifest_sha256=hashlib.sha256(raw).hexdigest(), format=fmt,
                           expected_s=expected)
    except Exception as e:  # noqa: BLE001 - a bad cache must never stop the worker
        shutil.rmtree(stage, ignore_errors=True)
        reset_local(local)
        reason = f"{type(e).__name__}: {e}"
        secs = round(time.perf_counter() - t0, 2)
        log(f"compile cache {key.name}: not used after {secs:.1f} s, compiling cold ({reason})")
        return CacheStatus("cold", key.name, reason=reason[:500], seconds=secs, format=fmt,
                           expected_s=expected)


LEGACY_SKIPPED = "unkeyed lab cache not copied: slower than a cold compile"


def seed_legacy(legacy: Path, local: Path, key: CacheKey, *,
                log: Callable[[str], None] = print) -> CacheStatus:
    """No longer copies anything (module docstring, "Legacy lab caches"): the unkeyed
    ``balanced/<engine kind>`` lab cache (``balanced_lab.cache_env``) is not verifiable and its
    per-file copy took as long as the compile it saves (``balanced/captioner``: 1,897 files in
    294 s on 2026-10-22). Returns ``cold`` with an empty ``local``; the container compiles and
    publishes a keyed cache. Kept so ``modal_app.balanced_app`` (stage-hashed) stays as is."""
    reset_local(local)
    if not (legacy / "vllm").is_dir():
        return CacheStatus("cold", key.name, reason=NO_CACHE)
    log(f"compile cache {key.name}: {LEGACY_SKIPPED} ({legacy}); compiling cold")
    return CacheStatus("cold", key.name, reason=f"{NO_CACHE}; {LEGACY_SKIPPED}")


# ---------------------------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------------------------


class Leases(Protocol):
    def put_if_absent(self, key: str, value: dict) -> bool: ...
    def get(self, key: str) -> dict | None: ...
    def put(self, key: str, value: dict) -> None: ...


class DictLeases:
    """Over a ``modal.Dict``: ``put(..., skip_if_exists=True)`` is an atomic put-if-absent
    across containers (a marker file on the Volume is not: each container sees its own view
    until it commits)."""

    def __init__(self, d: Any):
        self.d = d

    @classmethod
    def from_name(cls, name: str = LEASE_DICT) -> DictLeases:
        import modal

        return cls(modal.Dict.from_name(name, create_if_missing=True))

    def put_if_absent(self, key: str, value: dict) -> bool:
        return bool(self.d.put(key, value, skip_if_exists=True))

    def get(self, key: str) -> dict | None:
        return self.d.get(key)

    def put(self, key: str, value: dict) -> None:
        self.d.put(key, value)


class FileLeases:
    """``O_CREAT | O_EXCL`` files in one directory: atomic on one host (tests, local runs)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / (key.replace("/", "__") + ".json")

    def put_if_absent(self, key: str, value: dict) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self._path(key), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump(value, f)
        return True

    def get(self, key: str) -> dict | None:
        try:
            return json.loads(self._path(key).read_text())
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: dict) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        os.replace(tmp, self._path(key))


@dataclass(frozen=True)
class Lease:
    key: str  # "<cache key name>/<generation>"
    generation: int
    holder: str


def acquire(leases: Leases, name: str, holder: str, *, ttl_s: float = LEASE_TTL_S,
            clock: Callable[[], float] = time.time,
            max_generations: int = MAX_GENERATIONS) -> tuple[Lease | None, str]:
    """The publish lease of cache ``name``, or ``(None, why)``. Generation ``g`` is taken by
    put-if-absent; a live lease or a ``done`` one stops the caller; an expired one lets callers
    race for ``g + 1``, which exactly one wins."""
    for gen in range(max_generations):
        k = f"{name}/{gen}"
        now = clock()
        if leases.put_if_absent(k, {"holder": holder, "acquired": now, "expires": now + ttl_s}):
            return Lease(k, gen, holder), "acquired"
        cur = leases.get(k) or {}
        if cur.get("done"):
            return None, f"already published by {cur.get('holder')}"
        if float(cur.get("expires") or 0) > now:
            return None, f"leased by {cur.get('holder')} until {cur.get('expires')}"
    return None, f"no lease generation left of {max_generations}"


def still_held(leases: Leases, lease: Lease) -> bool:
    """No successor generation exists (an expired lease may have been taken over)."""
    cur = leases.get(lease.key) or {}
    return cur.get("holder") == lease.holder and \
        leases.get(f"{lease.key.rsplit('/', 1)[0]}/{lease.generation + 1}") is None


# ---------------------------------------------------------------------------------------------
# Publish (one container per key)
# ---------------------------------------------------------------------------------------------


class Sink(Protocol):
    def upload(self, files: Mapping[str, Path]) -> None: ...  # remote path -> local file
    def put_bytes(self, remote: str, data: bytes) -> None: ...


class VolumeSink:
    """Writes through the Volume client API (``batch_upload``, atomic per file on v2), so the
    worker's mount stays read-only."""

    def __init__(self, volume: Any):
        self.volume = volume

    @classmethod
    def from_name(cls, name: str | None = None) -> VolumeSink:
        import modal

        from flora_deploy.layout import CACHE_VOLUME

        return cls(modal.Volume.from_name(name or CACHE_VOLUME, version=2))

    def upload(self, files: Mapping[str, Path]) -> None:
        with self.volume.batch_upload(force=True) as batch:
            for remote, local in files.items():
                batch.put_file(str(local), "/" + remote)

    def put_bytes(self, remote: str, data: bytes) -> None:
        import io

        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), "/" + remote)


class DirSink:
    """A directory standing in for the Volume (tests, local runs); per-file atomic."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _write(self, remote: str, data: bytes) -> None:
        dst = self.root / remote
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dst)

    def upload(self, files: Mapping[str, Path]) -> None:
        for remote, local in files.items():
            self._write(remote, Path(local).read_bytes())

    def put_bytes(self, remote: str, data: bytes) -> None:
        self._write(remote, data)


def publish(key: CacheKey, local: Path, sink: Sink, leases: Leases, *, holder: str,
            runtime: Mapping[str, Any] | None = None, mount: Path | str | None = None,
            clock: Callable[[], float] = time.time,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """Publish ``local``'s tree as cache ``key`` (module docstring). Never raises; returns what
    happened (``published`` true/false and why)."""
    out: dict[str, Any] = {"key": key.name, "published": False}
    tar: Path | None = None
    try:
        lease_name = key.name
        mpath = None if mount is None else Path(mount) / ROOT / key.name / MANIFEST
        if mpath is not None and mpath.is_file():
            try:
                has_archive = bool(json.loads(mpath.read_bytes()).get("archive"))
            except ValueError:
                has_archive = True  # unreadable: leave it alone
            if has_archive:
                out["reason"] = "a manifest already exists at mount time"
                return out
            lease_name = key.name + ARCHIVE_LEASE_SUFFIX  # republish once, with an archive
            out["upgrade"] = True
        lease, why = acquire(leases, lease_name, holder, clock=clock)
        if lease is None:
            out["reason"] = why
            return out
        files = tree_files(local)
        if not any(r.startswith("vllm/") for r in files):
            out["reason"] = "nothing compiled under vllm/ (eager engine?)"
            leases.put(lease.key, {"holder": holder, "expires": 0, "released": True})
            return out
        runtime = runtime_versions() if runtime is None else runtime
        publish_id = f"p{int(clock())}-{uuid.uuid4().hex[:10]}"
        t0 = time.perf_counter()
        tar = local.parent / f".{local.name}.{publish_id}.tar"
        entries, archive = write_archive(files, tar)  # the manifest hashes what is archived
        manifest = build_manifest(key, publish_id, files, runtime=runtime, holder=holder,
                                  clock=clock, entries=entries, archive=archive)
        prefix = f"{ROOT}/{key.name}/{publish_id}"
        sink.upload({**{f"{prefix}/{rel}": p for rel, p in files.items()},
                     f"{prefix}/{ARCHIVE}": tar})
        if not still_held(leases, lease):
            out["reason"] = "lease lost during upload (the tree is orphaned, never read)"
            return out
        sink.put_bytes(f"{ROOT}/{key.name}/{MANIFEST}",
                       json.dumps(manifest, indent=1, sort_keys=True).encode())
        leases.put(lease.key, {"holder": holder, "done": True, "publish_id": publish_id,
                               "expires": float("inf")})
        out.update(published=True, publish_id=publish_id, files=len(files),
                   bytes=manifest["total_bytes"], archive_bytes=archive["size"],
                   generation=lease.generation)
        log(f"compile cache {key.name}: published {len(files)} files "
            f"({manifest['total_bytes'] / 1e6:.1f} MB, archive {archive['size'] / 1e6:.1f} MB)"
            f" as {publish_id} in {time.perf_counter() - t0:.1f} s")
        return out
    except Exception as e:  # noqa: BLE001 - publishing is best effort
        out["reason"] = f"{type(e).__name__}: {e}"[:500]
        log(f"compile cache {key.name}: publish failed ({out['reason']})")
        return out
    finally:
        if tar is not None:
            tar.unlink(missing_ok=True)


def published(key: CacheKey, exists: Callable[[str], bool]) -> bool:
    """Whether a manifest exists for ``key`` (``exists``: a Volume path probe, e.g. a
    ``ClientVolumeStore`` over ``flora-cache``). Existence only: the worker still verifies."""
    try:
        return bool(exists(f"{ROOT}/{key.name}/{MANIFEST}"))
    except Exception:  # noqa: BLE001 - unknown means "price it cold"
        return False


def published_kinds(pools: Mapping[str, str], exists: Callable[[str], bool],
                    **kw: Any) -> tuple[str, ...]:
    """The balanced shard kinds of ``pools`` (kind -> GPU) whose worker key has a published
    cache (for ``planner.options.PlanRequest.warm_kinds``)."""
    out = []
    for kind, gpu in pools.items():
        if kind not in ("segment", "caption"):
            continue
        try:
            key = worker_key(kind, gpu, **kw)
        except Exception:  # noqa: BLE001 - no key, no warm price
            continue
        if published(key, exists):
            out.append(kind)
    return tuple(sorted(out))


# ---------------------------------------------------------------------------------------------
# The worker's view
# ---------------------------------------------------------------------------------------------


@dataclass
class WorkerCache:
    """One container's compile cache: seeded at start, published at most once, after a
    shard has run generates on an engine that did not load a verified cache."""

    key: CacheKey
    local: Path
    status: CacheStatus
    publish_enabled: bool = True
    holder: str = ""
    sink: Callable[[], Sink] = VolumeSink.from_name
    leases: Callable[[], Leases] = DictLeases.from_name
    mount: Path | str | None = None
    log: Callable[[str], None] = print
    _done: bool = field(default=False, repr=False)

    @property
    def env(self) -> dict[str, str]:
        return env_for(self.local)

    def reset_cold(self, reason: str) -> None:
        """Before a cold retry of an engine build that failed on a loaded cache."""
        self.log(f"compile cache {self.key.name}: engine build failed on the {self.status.state}"
                 f" cache ({reason}); retrying cold")
        reset_local(self.local)
        self.status = CacheStatus("cold", self.key.name, reason=f"build failed: {reason}"[:500])

    def maybe_publish(self, engine: Any) -> dict[str, Any] | None:
        """After a shard: publish if this container compiled (``cold`` / ``legacy``), or loaded
        a verified cache published without an archive (``format="files"``: republished once
        with one, :func:`publish`), its engine ran requests and is alive, and publishing is
        enabled. At most one attempt."""
        if self._done or not self.publish_enabled:
            return None
        if self.status.state == "warm" and self.status.format != "files":
            return None
        stats = getattr(engine, "stats", {}) or {}
        if not stats.get("requests") or getattr(engine, "dead", None):
            return None
        self._done = True
        try:
            sink, leases = self.sink(), self.leases()
        except Exception as e:  # noqa: BLE001
            res = {"key": self.key.name, "published": False,
                   "reason": f"{type(e).__name__}: {e}"[:500]}
        else:
            res = publish(self.key, self.local, sink, leases, holder=self.holder,
                          mount=self.mount, log=self.log)
        self.status.published = res
        return res
