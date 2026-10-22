"""Verify a downloaded snapshot against its ``SnapshotPlan`` and the digests pinned in ``pins.py``.

Checks, all from one streaming pass over each file (bounded memory, files hashed in parallel):

- every planned file is present with the planned size, and nothing else is in the snapshot;
- LFS files: sha256 of the content == the tree's ``lfs.sha256``;
- other files: git blob sha1 (``sha1(b"blob <size>\\0" + content)``) == the tree's ``blob_id``;
- the plan's digest == ``RepoPin.plan_digest`` (pinned in the repo, not taken from the Hub);
- NVIDIA's manifests (``SOURCE_MANIFEST.sha256`` for TempSeg/Captioner, ``SHA256SUMS`` for the
  Rewarder): non-empty, every listed file's sha256 matches, and every planned ``*.safetensors``
  file is listed;
- ``RepoPin.expected`` digests, derived as documented at ``_EXPECTED``.

The result is written as ``prep_receipt.json`` at the snapshot root.

What is ignored: only ``.cache/`` at the snapshot root (``snapshot_download(local_dir=...)`` keeps
its per-file resume metadata and tree cache there; it is not snapshot content but WS-D's offline
``snapshot_download`` reads it, so never delete it wholesale) and the receipt itself.

Receipt reuse (``check_receipt``) does not re-hash. It requires, in order: the current schema; a
passing result; the same ``pin_digest`` (the full pin incl. ``expected``, ``allow_patterns``,
``plan_digest``, plus the verifier version); the same ``plan_digest``; exactly the planned file set
on disk; and each file's fingerprint (size, mtime_ns) equal to what was hashed. A rewrite or
``modal volume put`` changes the fingerprint; storage-level bit rot does not, which is what
``flora prepare --reverify`` is for. ``st_ino`` and ``st_dev`` are recorded but not compared: a
Modal Volume assigns inode and device numbers per container mount, so a receipt written by the
prep container never matched them on a later mount (every exact worker rejected its receipts
with ``fingerprint changed (ino)``). The cost: a replacement that also restores the size and
mtime_ns is no longer caught without a re-hash. If a backend does not keep mtime_ns stable,
reuse simply fails and the snapshot is re-hashed (slow, never unsafe).

Upgrading receipts: a receipt with an older ``schema`` is never reused. The next ``flora
prepare`` treats the pin like ``--reverify``: ``snapshot_download`` finds the files complete and
returns without fetching them, every file is re-hashed, and a current-schema receipt replaces the
old one. So the first run after this change costs one full hash pass per pin and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import ssl
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath

from flora_deploy.config.pins import RepoPin
from flora_deploy.layout import RECEIPT_NAME
from flora_deploy.prep.download import SnapshotPlan

RECEIPT_SCHEMA = 2
# Bump when the checks change meaning; part of ``pin_digest``, so old receipts stop being reused.
VERIFIER_VERSION = 2
CHUNK = 8 * 2**20
MANIFEST_NAMES = ("SOURCE_MANIFEST.sha256", "SHA256SUMS")
# snapshot_download(local_dir=...) keeps its resume metadata here; it is not part of the snapshot.
IGNORED_DIRS = (".cache",)
# Fingerprint fields compared on reuse. st_ino / st_dev are recorded only: both change between
# Modal Volume mounts (see module docstring).
REUSE_FINGERPRINT = ("size", "mtime_ns")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class FileDigest:
    path: str
    size: int
    sha256: str
    git_sha1: str | None = None  # computed for non-LFS files only
    # Fingerprint of the file as hashed (stat after hashing, equal to the stat before it).
    mtime_ns: int = 0
    ino: int = 0
    dev: int = 0


@dataclass(frozen=True)
class VerifyResult:
    pin: RepoPin
    root: str
    plan_digest: str
    checks: tuple[Check, ...]
    digests: tuple[FileDigest, ...] = field(default_factory=tuple)
    hash_seconds: float = 0.0
    # Planned files implicated by a failed check (missing, wrong size, wrong hash, unreadable,
    # source of a failed expected digest), and files on disk that are not in the plan.
    bad_files: tuple[str, ...] = ()
    extra_files: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def total_bytes(self) -> int:
        return sum(d.size for d in self.digests)


def receipt_path(snapshot_dir: Path) -> Path:
    return Path(snapshot_dir) / RECEIPT_NAME


def plan_digest(plan: SnapshotPlan) -> str:
    """Identity of what was supposed to be downloaded (repo, revision, files and their hashes).

    This is the value pinned as ``RepoPin.plan_digest``; it does not depend on the pin's other
    fields, so the pin can carry it without a cycle."""
    body = {
        "repo": plan.pin.repo,
        "revision": plan.pin.revision,
        "allow_patterns": list(plan.pin.allow_patterns or []),
        "files": [[f.path, f.size, f.lfs_sha256, f.blob_id] for f in plan.files],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def pin_record(pin: RepoPin) -> dict:
    return {
        "key": pin.key,
        "repo": pin.repo,
        "revision": pin.revision,
        "role": pin.role,
        "allow_patterns": list(pin.allow_patterns) if pin.allow_patterns else None,
        "expected": dict(pin.expected),
        "plan_digest": pin.plan_digest,
    }


def pin_digest(pin: RepoPin) -> str:
    """Everything that decides whether a snapshot passes: the full pin, which manifest is required,
    what is ignored and the verifier version. A receipt is reused only if this is unchanged."""
    body = {
        "pin": pin_record(pin),
        "required_manifest": required_manifest(pin),
        "ignored_dirs": list(IGNORED_DIRS),
        "verifier": VERIFIER_VERSION,
        "schema": RECEIPT_SCHEMA,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def fingerprint(path: Path) -> tuple[int, int, int, int]:
    """(size, mtime_ns, ino, dev), like NVIDIA's ``_validate_manifest`` fingerprints."""
    st = path.stat()
    return st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev


def hash_file(path: Path, *, git_blob: bool, chunk: int = CHUNK) -> tuple[str, str | None, int]:
    """Stream ``path`` once; return (sha256, git blob sha1 or None, bytes read)."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(b"blob %d\0" % size) if git_blob else None
    read = 0
    buf = bytearray(chunk)
    view = memoryview(buf)
    with path.open("rb", buffering=0) as f:
        while n := f.readinto(buf):
            sha256.update(view[:n])
            if sha1 is not None:
                sha1.update(view[:n])
            read += n
    if read != size:
        raise OSError(f"{path}: size changed while hashing ({size} -> {read})")
    return sha256.hexdigest(), sha1.hexdigest() if sha1 else None, read


def parse_sha256_manifest(text: str) -> list[tuple[str, str]]:
    """Parse ``<sha256> [*]<relative path>`` lines (sha256sum format, as NVIDIA's models.py reads
    them). Raises ``ValueError`` on malformed lines or paths escaping the snapshot."""
    entries = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed manifest line: {line!r}")
        digest, name = parts[0].lower(), parts[1].lstrip("*")
        rel = PurePosixPath(name)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"bad sha256 in manifest line: {line!r}")
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"manifest path escapes snapshot: {name!r}")
        entries.append((digest, str(rel)))
    return entries


def required_manifest(pin: RepoPin) -> str | None:
    """The manifest NVIDIA publishes for this pin's files, if the pin downloads it."""
    if pin.repo == "nvidia/Flora-Rewarder":
        return "SHA256SUMS"
    if pin.repo in ("nvidia/Flora-TempSeg", "nvidia/Flora-Captioner") and pin.role == "weights":
        return "SOURCE_MANIFEST.sha256"
    return None


# --- RepoPin.expected derivations ----------------------------------------------------------------
#
# Each returns (value or None if its source file is absent, source file). It may raise; the
# caller records the error as a failed check.
#
# adapter_sha256 (TempSeg): NVIDIA's ``batch_utils/vendor/tempseg_fp32_loader.py`` ``load()``
#   asserts ``digest(<export>/adapter_model.safetensors) == spec['adapter_sha256']``, where
#   ``digest()`` is ``hashlib.file_digest(..., 'sha256')`` over the file bytes, and the spec value
#   is ``models.json['tempseg']['sha256']`` (passed in by ``batch_utils/models.py``
#   ``Session.load``). So: plain sha256 of the adapter file's bytes.
#
# manifest_sha256 (TempSeg, Captioner, Rewarder): plain sha256 of the bytes of the pin's
#   ``required_manifest``. This is the repo-pinned trust anchor for NVIDIA's per-file digests
#   (values and how they were obtained: ``config/pins.py``).
#
# source_manifest_identity (Captioner): secondary consistency check, not an anchor.
#   ``models.json['captioner']['source_manifest_identity']`` is never read by NVIDIA's code at the
#   code pins (grep of batch_utils finds it only in models.json). The value equals
#   ``MODEL_PROVENANCE.json``'s ``source_model_identity_sha256``, which identifies the
#   *pre-export* source model; it is not a digest of any published file. Its bytes are bound by
#   the manifest, which ``manifest_sha256`` pins.


def _expected_adapter_sha256(pin: RepoPin, root: Path, sha: dict[str, str]):
    return sha.get("adapter_model.safetensors"), "adapter_model.safetensors"


def _expected_manifest_sha256(pin: RepoPin, root: Path, sha: dict[str, str]):
    name = required_manifest(pin)
    if name is None:
        raise ValueError("pin has no required manifest")
    return sha.get(name), name


def _expected_source_manifest_identity(pin: RepoPin, root: Path, sha: dict[str, str]):
    src = "MODEL_PROVENANCE.json"
    if src not in sha:
        return None, src
    prov = json.loads((root / src).read_text())
    if not isinstance(prov, dict):
        raise ValueError(f"{src} is not a JSON object")
    value = prov.get("source_model_identity_sha256")
    if not isinstance(value, str):
        raise ValueError(f"{src} has no string source_model_identity_sha256")
    return value, src


_EXPECTED: dict[str, Callable[[RepoPin, Path, dict[str, str]], tuple[str | None, str]]] = {
    "adapter_sha256": _expected_adapter_sha256,
    "manifest_sha256": _expected_manifest_sha256,
    "source_manifest_identity": _expected_source_manifest_identity,
}


def local_files(root: Path) -> set[str]:
    """Files under ``root`` that count as snapshot content (see module docstring)."""
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            out.add((rel_dir / name).as_posix())
    out.discard(RECEIPT_NAME)
    out.discard(RECEIPT_NAME + ".tmp")
    return out


def _summary(name: str, bad: list[str], total: int) -> Check:
    if bad:
        shown = ", ".join(bad[:10]) + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else "")
        return Check(name, False, f"{len(bad)}/{total} failed: {shown}")
    return Check(name, True, f"{total} ok")


def _hash_one(root: Path, f) -> tuple[FileDigest | None, str | None]:
    """Hash one planned file; never raises. Rejects a file that changed while being hashed."""
    p = root / f.path
    try:
        before = fingerprint(p)
        s256, s1, n = hash_file(p, git_blob=f.lfs_sha256 is None)
        after = fingerprint(p)
    except Exception as e:  # noqa: BLE001 - every per-file error becomes a failed check
        return None, f"{f.path} ({type(e).__name__}: {e})"
    if before != after:
        return None, f"{f.path} (changed while hashing)"
    size, mtime_ns, ino, dev = after
    return FileDigest(f.path, n, s256, s1, mtime_ns, ino, dev), None


def verify_snapshot(
    plan: SnapshotPlan, root: Path, *, workers: int | None = None
) -> VerifyResult:
    """Hash every file of ``plan`` under ``root`` and run all checks. Never raises: per-file
    errors (unreadable, vanished, changed while hashing, malformed JSON) are failed checks."""
    root = Path(root)
    checks: list[Check] = []
    bad: set[str] = set()
    planned = {f.path: f for f in plan.files}
    try:
        local = local_files(root) if root.is_dir() else set()
    except OSError as e:
        local = set()
        checks.append(Check("listing", False, f"{type(e).__name__}: {e}"))

    missing = sorted(p for p in planned if p not in local)
    bad.update(missing)
    checks.append(_summary("files_present", missing, len(planned)))
    extra = sorted(local - planned.keys())
    checks.append(
        Check("no_extra_files", not extra, ", ".join(extra[:10]) if extra else "none")
    )

    present = [planned[p] for p in sorted(planned) if p in local]
    bad_size = []
    for f in present:
        try:
            if (root / f.path).stat().st_size != f.size:
                bad_size.append(f.path)
        except OSError:
            bad_size.append(f.path)
    bad.update(bad_size)
    checks.append(_summary("sizes", bad_size, len(present)))

    started = time.monotonic()
    # Largest first, so one big shard doesn't end up alone at the tail. hashlib releases the GIL.
    order = sorted(present, key=lambda f: -f.size)
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        hashed = list(pool.map(lambda f: _hash_one(root, f), order))
    digests = sorted((d for d, _ in hashed if d is not None), key=lambda d: d.path)
    errors = sorted(err for _, err in hashed if err is not None)
    bad.update(f.path for f, (d, _) in zip(order, hashed, strict=True) if d is None)
    checks.append(_summary("readable", errors, len(present)))
    hash_seconds = time.monotonic() - started
    sha = {d.path: d.sha256 for d in digests}

    lfs = [d for d in digests if planned[d.path].lfs_sha256]
    bad_lfs = [d.path for d in lfs if d.sha256 != (planned[d.path].lfs_sha256 or "").lower()]
    checks.append(_summary("lfs_sha256", bad_lfs, len(lfs)))
    blobs = [d for d in digests if not planned[d.path].lfs_sha256]
    bad_blob = [d.path for d in blobs if d.git_sha1 != (planned[d.path].blob_id or "").lower()]
    checks.append(_summary("git_blob_sha1", bad_blob, len(blobs)))
    bad.update(bad_lfs + bad_blob)

    digest = plan_digest(plan)
    if plan.pin.plan_digest is None:
        checks.append(Check("pinned_plan_digest", False, "pin has no plan_digest"))
    else:
        same = digest == plan.pin.plan_digest.lower()
        checks.append(Check("pinned_plan_digest", same,
                            "matches" if same else f"got {digest}, pinned {plan.pin.plan_digest}"))

    required = required_manifest(plan.pin)
    for name in MANIFEST_NAMES:
        if name not in planned:
            if name == required:
                checks.append(Check(f"manifest:{name}", False, "required manifest not in plan"))
            continue
        if name not in sha:
            checks.append(Check(f"manifest:{name}", False, "manifest file missing or unreadable"))
            continue
        try:
            entries = parse_sha256_manifest((root / name).read_text())
        except (OSError, ValueError, UnicodeDecodeError) as e:
            bad.add(name)
            checks.append(Check(f"manifest:{name}", False, str(e)))
            continue
        if not entries:
            bad.add(name)
            checks.append(Check(f"manifest:{name}", False, "manifest is empty"))
            continue
        mism = [p for d, p in entries if sha.get(p) != d]
        bad.update(p for p in mism if p in planned)
        checks.append(_summary(f"manifest:{name}", mism, len(entries)))
        if name == required:
            listed = {p for _, p in entries}
            uncovered = sorted(p for p in planned if p.endswith(".safetensors") and p not in listed)
            checks.append(_summary(f"manifest_covers:{name}", uncovered,
                                   sum(p.endswith(".safetensors") for p in planned)))

    for key, want in plan.pin.expected:
        derive = _EXPECTED.get(key)
        if derive is None:
            checks.append(Check(f"expected:{key}", False, "no known derivation"))
            continue
        src = None
        try:
            got, src = derive(plan.pin, root, sha)
        except Exception as e:  # noqa: BLE001 - recorded as a failed check
            got, err = None, f"{type(e).__name__}: {e}"
        else:
            err = f"source file {src} missing or unreadable" if got is None else ""
        ok = got is not None and got.lower() == want.lower()
        if not ok and src in planned:
            bad.add(src)
        checks.append(Check(f"expected:{key}", ok, err or f"got {got}"))

    return VerifyResult(
        pin=plan.pin,
        root=str(root),
        plan_digest=digest,
        checks=tuple(checks),
        digests=tuple(digests),
        hash_seconds=round(hash_seconds, 3),
        bad_files=tuple(sorted(bad)),
        extra_files=tuple(extra),
    )


def tool_versions() -> dict[str, str]:
    """Versions of the tools that downloaded and hashed (huggingface_hub/hf_xet as installed in
    the container that ran this)."""
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "openssl": ssl.OPENSSL_VERSION}
    for dist in ("huggingface_hub", "hf_xet", "modal"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def framework_provenance(provenance: dict | None = None) -> dict:
    """Framework identity for the receipt. ``__version__``/``LOGIC_VERSION`` come from the source
    that ran; the git commit is passed in by the CLI (the container has no checkout or dist-info).
    """
    from flora_deploy import LOGIC_VERSION, __version__

    provenance = provenance or {}
    return {
        "version": __version__,
        "logic_version": LOGIC_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
    }


def build_receipt(
    result: VerifyResult,
    *,
    timings: dict[str, float] | None = None,
    provenance: dict | None = None,
    repair: dict | None = None,
) -> dict:
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "ok": result.ok,
        "pin": pin_record(result.pin),
        "pin_digest": pin_digest(result.pin),
        "plan_digest": result.plan_digest,
        "ignored_dirs": list(IGNORED_DIRS),
        "files": len(result.digests),
        "bytes": result.total_bytes,
        "checks": [asdict(c) for c in result.checks],
        "digests": [asdict(d) for d in result.digests],
        "timings_s": {"hash": result.hash_seconds, **(timings or {})},
        "framework": framework_provenance(provenance),
        "tools": tool_versions(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if repair:
        receipt["repair"] = repair
    return receipt


def write_receipt(receipt: dict, snapshot_dir: Path) -> Path:
    """Write the receipt whole (tmp file + rename), so a reader never sees a partial file."""
    path = receipt_path(snapshot_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def check_receipt(plan: SnapshotPlan, snapshot_dir: Path) -> tuple[dict | None, str]:
    """Return (receipt, "reused") if the existing receipt may be reused without re-hashing (see
    module docstring), else (None, why not)."""
    root = Path(snapshot_dir)
    try:
        receipt = json.loads(receipt_path(root).read_text())
    except FileNotFoundError:
        return None, "no receipt"
    except (OSError, ValueError) as e:
        return None, f"unreadable receipt ({type(e).__name__})"
    if not isinstance(receipt, dict):
        return None, "malformed receipt"
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return None, f"receipt schema {receipt.get('schema')} != {RECEIPT_SCHEMA}; re-verifying"
    if receipt.get("ok") is not True:
        return None, "previous verification failed"
    if receipt.get("pin_digest") != pin_digest(plan.pin):
        return None, "pin or verifier changed since the receipt"
    if receipt.get("plan_digest") != plan_digest(plan):
        return None, "plan changed since the receipt"
    try:
        on_disk = local_files(root)
    except OSError:
        return None, "snapshot dir unreadable"
    planned = {f.path for f in plan.files}
    if on_disk != planned:
        diff = sorted(on_disk ^ planned)
        return None, f"file set changed: {', '.join(diff[:5])}"
    recorded = {d.get("path"): d for d in receipt.get("digests", []) if isinstance(d, dict)}
    for f in plan.files:
        rec = recorded.get(f.path)
        if rec is None:
            return None, f"{f.path}: not in receipt"
        try:
            size, mtime_ns, ino, _dev = fingerprint(root / f.path)
        except OSError:
            return None, f"{f.path}: unreadable"
        now = {"size": size, "mtime_ns": mtime_ns, "ino": ino}
        if size != f.size or any(rec.get(k) != now[k] for k in REUSE_FINGERPRINT):
            return None, f"{f.path}: fingerprint changed"
    return receipt, "reused"


def load_valid_receipt(plan: SnapshotPlan, snapshot_dir: Path) -> dict | None:
    """The existing receipt if ``check_receipt`` accepts it, else None."""
    return check_receipt(plan, snapshot_dir)[0]
