"""Make NVIDIA's ``Session.load('captioner')`` load our BF16 Captioner (OPT-2).

``prep/convert_captioner.py`` writes a BF16 copy of the FP32 Captioner whose tensors already
carry the dtype NVIDIA's loader would give them. This module redirects NVIDIA's loader to it,
by replacing two attributes of NVIDIA's ``batch_utils.models`` module at runtime (NVIDIA's files
are never touched):

- ``models.snapshot``: ``snapshot('captioner')`` returns the derived directory on the weights
  Volume (read-only is fine: nothing is written, so no local copy is needed). Every other kind
  goes to NVIDIA's original function.
- ``models.validate_manifest``: for the derived directory, NVIDIA's manifest re-hash (whose
  ``SOURCE_MANIFEST.sha256`` describes the FP32 files) is replaced by a check against our
  ``derived_receipt.json``. Any other path goes to NVIDIA's original function.

Everything else in ``Session.load`` runs unchanged: ``_keep_native_gdn_fp32``, the same
``from_pretrained(dtype=torch.bfloat16, device_map='balanced', ...)`` call, the loading-info
check, ``verify_cuda_residency`` and ``verify_gdn``.

What is trusted, and why (``docs/captioner-bf16.md``):

- **The receipt is pinned.** Its bytes must hash to ``DERIVED_PINS["captioner-bf16"]
  .receipt_sha256`` in ``pins.py`` (``None`` fails closed). Its schema, ``ok``, converter
  version (= ``CONVERTER_VERSION`` = the pin's), source revision and NVIDIA manifest digest must
  equal the pins.
- **Every byte is hashed by default** (``verify_derived(full=True)``): the files on disk must be
  exactly the receipt's files, each a regular file opened with ``O_NOFOLLOW`` (symlinks,
  directories, devices are refused), with the recorded size and sha256; a file that changes
  while it is hashed is refused. Files are hashed in parallel threads. ``install(background=
  True)`` runs the hash on its own thread so it overlaps the TempSeg load (it also warms the
  page cache for ``from_pretrained``); ``Session.load('captioner')`` waits for it.
  After one full hash, later loads re-check the receipt and each file's (dev, inode, size,
  mtime, ctime) stamp, the way NVIDIA's own ``validate_manifest`` reuses a verification.
- **Opt-out** (``full_verify=False`` or ``FLORA_CAPTIONER_FULL_VERIFY=0``): sizes, full hashes
  of non-shard / small files and safetensors-header hashes of large shards only. Recorded as
  ``full_verify_opt_out`` in ``provenance()``.
- **Toolchain** (``check_toolchain=True``, default): torch / transformers / safetensors versions
  (local build tag ignored) must equal both the receipt's and the identity proof's
  (``DerivedPin.proven_with``); and the dtype plan is rebuilt with NVIDIA's own
  ``_keep_native_gdn_fp32`` on the meta device and must equal the receipt's ``dtype_policy``.
  This runs synchronously in ``install`` (building the meta model touches torch's process-wide
  default dtype, so it must not overlap a model load). Opting out is recorded in provenance.

The identity proof is ``modal_app/convert_app.py::identity`` (``docs/captioner-bf16.md``); the
comparison helpers it uses (``compare_models``, ``fixed_forward``) live here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from flora_deploy.config.pins import DERIVED_PINS_BY_KEY, PINS_BY_KEY, DerivedPin, RepoPin
from flora_deploy.layout import WEIGHTS_MOUNT, derived_path
from flora_deploy.prep.convert_captioner import (
    CONVERTER_VERSION,
    DERIVED_NAME,
    DERIVED_RECEIPT,
    RECEIPT_SCHEMA,
    SOURCE_KEY,
    ConversionError,
    read_header,
    tool_versions,
)

KIND = "captioner"
FULL_VERIFY_ENV = "FLORA_CAPTIONER_FULL_VERIFY"  # "0": opt out of the full re-hash (recorded)
TOOLCHAIN = ("torch", "transformers", "safetensors")
SMALL_FILE = 64 * 2**20  # opt-out mode: shards below this are still fully hashed
DEFAULT_WORKERS = 8
PROVENANCE_SCHEMA = "flora_captioner_bf16_v2"
_MARK = "__flora_accel_captioner_bf16__"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CHUNK = 16 * 2**20


class DerivedReceiptError(RuntimeError):
    """The derived BF16 Captioner does not match its (pinned) receipt or toolchain."""


def default_derived_dir(root=WEIGHTS_MOUNT) -> Path:
    return Path(derived_path(DERIVED_NAME, PINS_BY_KEY[SOURCE_KEY], root))


def pinned_derived() -> DerivedPin:
    """The ``pins.py`` record of the BF16 Captioner (receipt digest, toolchain of the proof)."""
    return DERIVED_PINS_BY_KEY[DERIVED_NAME]


def full_verify_default() -> bool:
    return os.environ.get(FULL_VERIFY_ENV, "1") != "0"


# --- files ------------------------------------------------------------------------------------

Stamp = tuple[int, int, int, int, int]


def _stamp(st: os.stat_result) -> Stamp:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    """An fd of ``path`` opened without following a final symlink; refuses non-regular files."""
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW)
    except OSError as e:
        raise DerivedReceiptError(f"{path}: cannot open as a regular file ({e})") from e
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise DerivedReceiptError(f"{path}: not a regular file")
    return fd, st


def _hash_regular(path: Path, *, header_only: bool = False) -> tuple[str, Stamp, int]:
    """(sha256 of the file, or of its safetensors header; stamp; bytes hashed)."""
    fd, before = _open_regular(path)
    try:
        with open(fd, "rb", buffering=0, closefd=False) as f:
            if header_only:
                try:
                    raw = read_header(f)[2]
                except (ConversionError, ValueError) as e:
                    raise DerivedReceiptError(f"{path}: bad safetensors header ({e})") from e
                digest, n = hashlib.sha256(raw).hexdigest(), 0
            else:
                h = hashlib.sha256()
                buf = bytearray(_CHUNK)
                view = memoryview(buf)
                n = 0
                while k := f.readinto(buf):
                    h.update(view[:k])
                    n += k
                digest = h.hexdigest()
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _stamp(after) != _stamp(before):
        raise DerivedReceiptError(f"{path}: changed while it was hashed")
    return digest, _stamp(before), n


def _read_receipt(derived_dir: Path, dpin: DerivedPin) -> tuple[dict, str, Stamp]:
    """The receipt parsed from exactly the bytes whose sha256 is checked against the pin."""
    path = derived_dir / DERIVED_RECEIPT
    try:
        fd, st = _open_regular(path)
    except DerivedReceiptError as e:
        raise DerivedReceiptError(f"no readable receipt in {derived_dir}: {e}") from e
    try:
        with open(fd, "rb", buffering=0, closefd=False) as f:
            raw = f.read()
    finally:
        os.close(fd)
    digest = hashlib.sha256(raw).hexdigest()
    if dpin.receipt_sha256 is None:
        raise DerivedReceiptError(
            f"{derived_dir}: no receipt digest is pinned (pins.DERIVED_PINS[{dpin.key!r}]); "
            "refusing to trust an unpinned derived checkpoint"
        )
    if digest != dpin.receipt_sha256:
        raise DerivedReceiptError(
            f"{derived_dir}: receipt sha256 {digest} != pinned {dpin.receipt_sha256}"
        )
    try:
        receipt = json.loads(raw)
    except ValueError as e:
        raise DerivedReceiptError(f"{path}: not JSON ({e})") from e
    return receipt, digest, _stamp(st)


def _receipt_problems(receipt: dict, pin: RepoPin, dpin: DerivedPin) -> list[str]:
    problems = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        problems.append(f"schema {receipt.get('schema')} != {RECEIPT_SCHEMA}")
    if receipt.get("ok") is not True:
        problems.append("receipt is not passing")
    if receipt.get("kind") != DERIVED_NAME:
        problems.append(f"kind {receipt.get('kind')!r}")
    version = receipt.get("converter", {}).get("version")
    if not version == CONVERTER_VERSION == dpin.converter_version:
        problems.append(f"converter version {version} (this converter: {CONVERTER_VERSION}, "
                        f"pinned: {dpin.converter_version})")
    src = receipt.get("source", {})
    if src.get("pin", {}).get("revision") != pin.revision:
        problems.append(f"source revision {src.get('pin', {}).get('revision')} != {pin.revision}")
    if src.get("manifest_sha256") != dict(pin.expected).get("manifest_sha256"):
        problems.append("source manifest digest differs from pins.py")
    files = receipt.get("files")
    if not files:
        problems.append("receipt lists no files")
    else:
        for f in files:
            p = f.get("path") if isinstance(f, dict) else None
            if (not isinstance(p, str) or p in ("", ".", "..") or "/" in p or "\\" in p
                    or p == DERIVED_RECEIPT):
                problems.append(f"bad file entry {f!r}")
    return problems


def _listing(derived_dir: Path, expected: set[str]) -> None:
    """The directory holds exactly ``expected`` (+ the receipt), all regular files."""
    if derived_dir.is_symlink():
        raise DerivedReceiptError(f"{derived_dir}: the derived directory is a symlink")
    names, bad = set(), []
    try:
        with os.scandir(derived_dir) as it:
            for entry in it:
                if entry.name == DERIVED_RECEIPT:
                    continue
                names.add(entry.name)
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    bad.append(entry.name)
    except OSError as e:
        raise DerivedReceiptError(f"{derived_dir}: cannot list ({e})") from e
    if bad:
        raise DerivedReceiptError(f"{derived_dir}: not regular files (symlink/dir/...): "
                                  f"{sorted(bad)}")
    if names != expected:
        raise DerivedReceiptError(
            f"{derived_dir}: files differ from receipt: missing {sorted(expected - names)}, "
            f"extra {sorted(names - expected)}"
        )


def _preflight(derived_dir: Path, pin: RepoPin, dpin: DerivedPin) -> tuple[dict, dict, str]:
    receipt, digest, _ = _read_receipt(derived_dir, dpin)
    problems = _receipt_problems(receipt, pin, dpin)
    if problems:
        raise DerivedReceiptError(f"{derived_dir}: " + "; ".join(problems))
    files = {f["path"]: f for f in receipt["files"]}
    _listing(derived_dir, set(files))
    return receipt, files, digest


def _verify(derived_dir: Path, *, full: bool, workers: int, pin: RepoPin,
            dpin: DerivedPin) -> tuple[dict, dict[str, Stamp]]:
    t0 = time.monotonic()
    _, files, digest = _preflight(derived_dir, pin, dpin)

    def check(entry: dict) -> tuple[str, int, Stamp | None]:
        path = derived_dir / entry["path"]
        try:
            size = os.lstat(path).st_size
        except OSError as e:
            return f"{entry['path']}: {e}", 0, None
        if size != entry.get("size"):
            return f"{entry['path']}: size {size} != {entry.get('size')}", 0, None
        header_only = not full and size >= SMALL_FILE and "header_sha256" in entry
        try:
            got, st, n = _hash_regular(path, header_only=header_only)
        except DerivedReceiptError as e:
            return str(e), 0, None
        if st[2] != entry["size"]:
            return f"{entry['path']}: size {st[2]} != {entry['size']}", n, st
        if header_only:
            if got != entry["header_sha256"]:
                return f"{entry['path']}: safetensors header mismatch", n, st
        elif got != entry.get("sha256"):
            return f"{entry['path']}: sha256 mismatch", n, st
        return "", n, st

    with ThreadPoolExecutor(max_workers=max(1, workers),
                            thread_name_prefix="flora-captioner-hash") as pool:
        results = list(pool.map(check, files.values()))
    bad = [msg for msg, _, _ in results if msg]
    if bad:
        raise DerivedReceiptError(f"{derived_dir}: " + "; ".join(bad))
    stamps = {name: st for name, (_, _, st) in zip(files, results, strict=True)}
    report = {
        "mode": "full" if full else "fast",
        "files": len(files),
        "bytes": sum(f["size"] for f in files.values()),
        "bytes_hashed": sum(n for _, n, _ in results),
        "receipt_sha256": digest,
        "workers": max(1, workers),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }
    return report, stamps


def verify_derived(
    derived_dir: Path,
    *,
    full: bool = True,
    workers: int = DEFAULT_WORKERS,
    pin: RepoPin | None = None,
    derived_pin: DerivedPin | None = None,
) -> dict:
    """Check ``derived_dir`` against its pinned ``derived_receipt.json`` (module docstring).

    ``full=True`` (default) hashes every byte; ``full=False`` is the opt-out fast check.
    ``pin`` / ``derived_pin`` default to ``pins.py`` (tests pass synthetic ones)."""
    return _verify(Path(derived_dir), full=full, workers=workers,
                   pin=pin or PINS_BY_KEY[SOURCE_KEY], dpin=derived_pin or pinned_derived())[0]


def _reverify(derived_dir: Path, stamps: Mapping[str, Stamp], pin: RepoPin,
              dpin: DerivedPin) -> dict:
    """Cheap re-check after a verification: same pinned receipt, same files, same stamps."""
    t0 = time.monotonic()
    _, files, digest = _preflight(derived_dir, pin, dpin)
    if set(files) != set(stamps):
        raise DerivedReceiptError(f"{derived_dir}: receipt files changed since verification")
    changed = []
    for name, want in stamps.items():
        try:
            st = os.lstat(derived_dir / name)
        except OSError:
            changed.append(name)
            continue
        if not stat.S_ISREG(st.st_mode) or _stamp(st) != want:
            changed.append(name)
    if changed:
        raise DerivedReceiptError(f"{derived_dir}: changed since verification: {sorted(changed)}")
    return {"mode": "reuse", "files": len(files), "bytes": sum(f["size"] for f in files.values()),
            "bytes_hashed": 0, "receipt_sha256": digest,
            "elapsed_s": round(time.monotonic() - t0, 3)}


# --- toolchain (M1) ---------------------------------------------------------------------------


def _public(version: Any) -> str | None:
    return None if version is None else str(version).split("+", 1)[0]


def check_toolchain(receipt: Mapping, derived_pin: DerivedPin | None = None, *,
                    versions: Mapping[str, str] | None = None) -> dict:
    """torch / transformers / safetensors of this process vs the receipt and the identity
    proof (``DerivedPin.proven_with``). Raises ``DerivedReceiptError`` on any difference."""
    dpin = derived_pin or pinned_derived()
    versions = tool_versions() if versions is None else versions
    recorded = receipt.get("tools") or {}
    proven = dict(dpin.proven_with)
    out, problems = {}, []
    for name in TOOLCHAIN:
        have, rec, want = versions.get(name), recorded.get(name), proven.get(name)
        out[name] = {"runtime": have, "receipt": rec, "proven_with": want}
        if (_public(have) is None or _public(have) != _public(rec)
                or (want is not None and _public(have) != _public(want))):
            problems.append(f"{name} runtime={have} receipt={rec} proven_with={want}")
    if problems:
        raise DerivedReceiptError(
            "toolchain differs from the one the BF16 identity proof covers ("
            + "; ".join(problems) + "); re-run convert_app --step convert and --step identity"
        )
    return out


def check_dtype_policy(receipt: Mapping, record: Mapping) -> dict:
    """The dtype plan rebuilt now (``DtypePolicy.record()``) must equal the receipt's."""
    now = json.loads(json.dumps(record, sort_keys=True))
    if now != receipt.get("dtype_policy"):
        raise DerivedReceiptError(
            f"dtype plan differs from the receipt's: now {now}, receipt "
            f"{receipt.get('dtype_policy')}; the BF16 checkpoint no longer matches what "
            "NVIDIA's loader would build"
        )
    return now


def _keep_fp32_patch(nv_or_models: Any) -> Callable[[type], None]:
    """NVIDIA's ``production_gdn_fix._keep_native_gdn_fp32``, from the same snapshot."""
    vendor = getattr(nv_or_models, "vendor", None)
    if callable(vendor):
        return vendor("production_gdn_fix")._keep_native_gdn_fp32
    models = nv_or_models
    root = Path(getattr(models, "__file__", "") or ".").resolve().parent / "vendor"
    mod = importlib.import_module("production_gdn_fix")
    origin = getattr(mod, "__file__", None)
    if origin is None or not Path(origin).resolve().is_relative_to(root):
        raise DerivedReceiptError(f"production_gdn_fix resolved to {origin}, outside {root}")
    return mod._keep_native_gdn_fp32


def nvidia_policy_record(nv_or_models: Any, derived_dir: Path,
                         keep_fp32_patch: Callable[[type], None] | None = None) -> dict:
    """``DtypePolicy.record()`` of NVIDIA's ``from_pretrained(dtype=bf16)`` call today, built on
    the meta device from the derived ``config.json`` (a byte copy of NVIDIA's)."""
    from flora_deploy.prep.convert_captioner import nvidia_dtype_policy

    keep = keep_fp32_patch or _keep_fp32_patch(nv_or_models)
    return nvidia_dtype_policy(Path(derived_dir), keep).record()


# --- background verification ------------------------------------------------------------------


class Verification:
    """A receipt verification running on its own thread (``install(background=True)``)."""

    def __init__(self, fn: Callable[[], Any]):
        self._done = threading.Event()
        self._result: Any = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True,
                                        name="flora-captioner-verify")
        self._thread.start()

    def _run(self, fn) -> None:
        try:
            self._result = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised by result()
            self._error = exc
        finally:
            self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: float | None = None):
        if not self._done.wait(timeout):
            raise TimeoutError("BF16 Captioner verification still running")
        if self._error is not None:
            raise self._error
        return self._result


# --- the patch --------------------------------------------------------------------------------


@dataclass
class CaptionerBf16Patch:
    """Installed redirection; ``uninstall()`` restores NVIDIA's original attributes."""

    models: ModuleType
    derived_dir: Path
    full_verify: bool
    original_snapshot: Callable
    original_validate: Callable
    pin: RepoPin | None = None
    derived_pin: DerivedPin | None = None
    check_toolchain: bool = True
    workers: int = DEFAULT_WORKERS
    verifications: list[dict] = field(default_factory=list)
    toolchain: dict | None = None
    dtype_policy: dict | None = None
    _policy_record: Callable[[], Mapping] | None = field(default=None, repr=False)
    _stamps: dict | None = field(default=None, repr=False)
    _job: Verification | None = field(default=None, repr=False)
    _failed: BaseException | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def _pin(self) -> RepoPin:
        return self.pin or PINS_BY_KEY[SOURCE_KEY]

    @property
    def _dpin(self) -> DerivedPin:
        return self.derived_pin or pinned_derived()

    # -- checks ------------------------------------------------------------------------------
    def verify_toolchain(self) -> dict:
        """M1: versions and dtype plan vs the receipt (once; run before any model loads)."""
        with self._lock:
            if self.toolchain is None:
                receipt, _, _ = _read_receipt(self.derived_dir, self._dpin)
                toolchain = check_toolchain(receipt, self._dpin)
                build = self._policy_record or (
                    lambda: nvidia_policy_record(self.models, self.derived_dir))
                self.dtype_policy = check_dtype_policy(receipt, build())
                self.toolchain = toolchain
            return self.toolchain

    def _run_verify(self, full: bool) -> tuple[dict, dict]:
        return _verify(self.derived_dir, full=full, workers=self.workers, pin=self._pin,
                       dpin=self._dpin)

    def start_background(self) -> Verification | None:
        """Start the (full, unless opted out) hash on its own thread; ``wait()`` joins it."""
        with self._lock:
            if self._job is None and self._stamps is None and self._failed is None:
                _preflight(self.derived_dir, self._pin, self._dpin)  # cheap: fail early
                full = self.full_verify
                self._job = Verification(lambda: self._run_verify(full))
            return self._job

    def _collect(self, timeout: float | None = None) -> None:
        if self._failed is not None:
            raise DerivedReceiptError(
                f"BF16 Captioner verification failed earlier: {self._failed}"
            ) from self._failed
        if self._job is None:
            return
        try:
            report, stamps = self._job.result(timeout)
        except TimeoutError:
            raise
        except BaseException as exc:
            self._job, self._failed = None, exc
            raise
        self._job = None
        self.verifications.append({**report, "background": True})
        self._stamps = stamps

    def wait(self, timeout: float | None = None) -> list[dict]:
        """Join a background verification (re-raising its error); returns all reports."""
        with self._lock:
            self._collect(timeout)
            return list(self.verifications)

    def verify_now(self) -> dict:
        """Hash now (full unless opted out) if nothing has been verified, else re-check the
        stamps of the last verification. Fails closed for good after a failed verification."""
        with self._lock:
            self._collect()
            try:
                if self._stamps is None:
                    report, self._stamps = self._run_verify(self.full_verify)
                else:
                    report = _reverify(self.derived_dir, self._stamps, self._pin, self._dpin)
            except DerivedReceiptError as exc:
                self._failed = exc
                raise
            self.verifications.append(report)
            return report

    # -- NVIDIA's attributes -----------------------------------------------------------------
    def snapshot(self, kind):
        if kind == KIND:
            return self.derived_dir
        return self.original_snapshot(kind)

    def validate_manifest(self, path):
        if Path(path) != self.derived_dir:
            return self.original_validate(path)
        if self.check_toolchain:
            self.verify_toolchain()
        report = self.verify_now()
        runtime_event = getattr(self.models, "runtime_event", None)
        if runtime_event is not None:
            runtime_event("manifest_replaced_by_derived_receipt", kind=KIND,
                          directory=str(self.derived_dir), **report)
        return None

    def provenance(self) -> dict:
        """What was checked, for run records (``Provenance.engine``)."""
        with self._lock:
            return {
                "schema": PROVENANCE_SCHEMA,
                "derived_dir": str(self.derived_dir),
                "receipt_sha256_pinned": self._dpin.receipt_sha256,
                "converter_version": CONVERTER_VERSION,
                "full_verify": self.full_verify,
                "full_verify_opt_out": not self.full_verify,
                "check_toolchain": self.check_toolchain,
                "toolchain_opt_out": not self.check_toolchain,
                "toolchain": self.toolchain,
                "dtype_policy": self.dtype_policy,
                "verification_pending": self._job is not None,
                "verifications": list(self.verifications),
            }

    def uninstall(self) -> None:
        if getattr(self.models.snapshot, _MARK, None) is self:
            self.models.snapshot = self.original_snapshot
        if getattr(self.models.validate_manifest, _MARK, None) is self:
            self.models.validate_manifest = self.original_validate

    def __enter__(self) -> CaptionerBf16Patch:
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()


def install(
    nv_or_models: Any,
    derived_dir: Path | str | None = None,
    *,
    full_verify: bool | None = None,
    verify_now: bool = True,
    background: bool = False,
    check_toolchain: bool = True,
    policy_record: Callable[[], Mapping] | None = None,
    workers: int = DEFAULT_WORKERS,
    pin: RepoPin | None = None,
    derived_pin: DerivedPin | None = None,
) -> CaptionerBf16Patch:
    """Redirect NVIDIA's ``models`` module (or a linked family handle's ``.models``) to the BF16
    Captioner.

    - ``full_verify``: hash every byte (default; ``FLORA_CAPTIONER_FULL_VERIFY=0`` or ``False``
      opts out to the fast check, recorded in ``provenance()``).
    - ``verify_now``: verify before returning, so a bad copy fails before anything loads.
    - ``background``: instead, start the hash on a thread and return at once (call it before the
      TempSeg load to overlap the two; ``Session.load('captioner')`` / ``wait()`` join it).
    - ``check_toolchain``: versions + dtype plan vs the receipt (synchronous; call ``install``
      before any model load or episode thread). ``policy_record`` replaces the meta-device
      rebuild of NVIDIA's dtype plan (tests).

    With ``verify_now=False`` and ``background=False`` everything runs inside the first
    ``Session.load('captioner')``. Raises before patching anything if a synchronous check fails.
    """
    models = getattr(nv_or_models, "models", nv_or_models)
    if not hasattr(models, "Session") or not hasattr(models, "snapshot"):
        raise TypeError(f"{models!r} is not NVIDIA's batch_utils.models")
    if getattr(models.snapshot, _MARK, None) is not None:
        raise RuntimeError("the BF16 Captioner patch is already installed")
    derived = Path(derived_dir) if derived_dir is not None else default_derived_dir()
    if full_verify is None:
        full_verify = full_verify_default()
    owner = nv_or_models if callable(getattr(nv_or_models, "vendor", None)) else models
    patch = CaptionerBf16Patch(
        models, derived, bool(full_verify), models.snapshot, models.validate_manifest, pin,
        derived_pin, bool(check_toolchain), workers,
        _policy_record=policy_record or (lambda: nvidia_policy_record(owner, derived)),
    )
    if verify_now or background:
        if check_toolchain:
            patch.verify_toolchain()
        if background:
            patch.start_background()
        else:
            patch.verify_now()

    def snapshot(kind):
        return patch.snapshot(kind)

    def validate_manifest(path):
        return patch.validate_manifest(path)

    setattr(snapshot, _MARK, patch)
    setattr(validate_manifest, _MARK, patch)
    models.snapshot = snapshot
    models.validate_manifest = validate_manifest
    return patch


# --- identity proof helpers ------------------------------------------------------------------


def _bits(t):
    import torch

    t = t.detach()
    if t.dim() == 0:
        t = t.reshape(1)
    return t.contiguous().reshape(-1).view(torch.uint8)


def tensors_equal_bits(a, b) -> bool:
    import torch

    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and a.device == b.device and torch.equal(_bits(a), _bits(b)))


def compare_models(a, b) -> dict:
    """Every parameter and buffer of ``a`` vs ``b``: same names, dtype, shape, device, bytes."""
    out: dict = {}
    for what, get in (("parameters", lambda m: m.named_parameters(remove_duplicate=False)),
                      ("buffers", lambda m: m.named_buffers(remove_duplicate=False))):
        ta, tb = dict(get(a)), dict(get(b))
        diffs = []
        names_equal = set(ta) == set(tb)
        nbytes = 0
        for name in sorted(set(ta) & set(tb)):
            x, y = ta[name], tb[name]
            if not tensors_equal_bits(x, y):
                diffs.append({"name": name, "a": [str(x.dtype), list(x.shape), str(x.device)],
                              "b": [str(y.dtype), list(y.shape), str(y.device)]})
            nbytes += x.numel() * x.element_size()
        out[what] = {
            "count": len(ta),
            "names_equal": names_equal,
            "only_a": sorted(set(ta) - set(tb))[:10],
            "only_b": sorted(set(tb) - set(ta))[:10],
            "bytes_compared": nbytes,
            "mismatches": len(diffs),
            "first_mismatches": diffs[:10],
            "dtypes": dict(sorted(Counter(str(t.dtype) for t in ta.values()).items())),
        }
    out["identical"] = all(out[w]["names_equal"] and out[w]["mismatches"] == 0
                           for w in ("parameters", "buffers"))
    return out


def fixed_forward(model, *, seq_len: int = 32, seed: int = 1234, vocab: int | None = None):
    """Logits of one forward pass on a fixed, seeded text-only input."""
    import torch

    vocab = vocab or model.config.get_text_config().vocab_size
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, seq_len), generator=g)
    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model(input_ids=ids.to(device), attention_mask=torch.ones_like(ids).to(device),
                    use_cache=False)
    return out.logits.detach()


def logits_digest(logits) -> str:
    return hashlib.sha256(_bits(logits).cpu().numpy().tobytes()).hexdigest()
