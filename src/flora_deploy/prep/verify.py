"""Verify a downloaded snapshot against its ``SnapshotPlan`` and NVIDIA's published digests.

Checks, all from one streaming pass over each file (bounded memory, files hashed in parallel):

- every planned file is present with the planned size, and nothing else is in the snapshot;
- LFS files: sha256 of the content == the tree's ``lfs.sha256``;
- other files: git blob sha1 (``sha1(b"blob <size>\\0" + content)``) == the tree's ``blob_id``;
- NVIDIA's manifests (``SOURCE_MANIFEST.sha256`` for TempSeg/Captioner, ``SHA256SUMS`` for the
  Rewarder): every listed file's sha256 matches;
- ``RepoPin.expected`` digests, derived the way NVIDIA's code derives them (see ``_EXPECTED``).

The result is written as ``prep_receipt.json`` at the snapshot root.
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

RECEIPT_SCHEMA = 1
CHUNK = 8 * 2**20
MANIFEST_NAMES = ("SOURCE_MANIFEST.sha256", "SHA256SUMS")
# snapshot_download(local_dir=...) keeps its resume metadata here; it is not part of the snapshot.
IGNORED_DIRS = (".cache",)


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


@dataclass(frozen=True)
class VerifyResult:
    pin: RepoPin
    root: str
    plan_digest: str
    checks: tuple[Check, ...]
    digests: tuple[FileDigest, ...] = field(default_factory=tuple)
    hash_seconds: float = 0.0

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
    """Identity of what was supposed to be downloaded (repo, revision, files and their hashes)."""
    body = {
        "repo": plan.pin.repo,
        "revision": plan.pin.revision,
        "allow_patterns": list(plan.pin.allow_patterns or []),
        "files": [[f.path, f.size, f.lfs_sha256, f.blob_id] for f in plan.files],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


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
# adapter_sha256 (TempSeg): NVIDIA's ``batch_utils/vendor/tempseg_fp32_loader.py`` ``load()``
#   asserts ``digest(<export>/adapter_model.safetensors) == spec['adapter_sha256']``, where
#   ``digest()`` is ``hashlib.file_digest(..., 'sha256')`` over the file bytes, and the spec value
#   is ``models.json['tempseg']['sha256']`` (passed in by ``batch_utils/models.py``
#   ``Session.load``). So: plain sha256 of the adapter file's bytes.
#
# source_manifest_identity (Captioner): ``models.json['captioner']['source_manifest_identity']`` is
#   never read by NVIDIA's code at the code pins (grep of batch_utils finds it only in
#   models.json). Integrity at load time comes from ``models.py`` ``_validate_manifest()``, which
#   checks each ``SOURCE_MANIFEST.sha256`` line. The value equals ``MODEL_PROVENANCE.json``'s
#   ``source_model_identity_sha256``, which identifies the *pre-export* source model. It is not
#   the sha256 of SOURCE_MANIFEST.sha256 (that is 75241af6...) or any simple digest of the
#   published files we tried. So we check the published provenance field, whose own bytes are
#   bound by the manifest and by the tree's git blob sha1.


def _expected_adapter_sha256(root: Path, sha: dict[str, str]) -> str | None:
    return sha.get("adapter_model.safetensors")


def _expected_source_manifest_identity(root: Path, sha: dict[str, str]) -> str | None:
    if "MODEL_PROVENANCE.json" not in sha:
        return None
    prov = json.loads((root / "MODEL_PROVENANCE.json").read_text())
    return prov.get("source_model_identity_sha256")


_EXPECTED: dict[str, Callable[[Path, dict[str, str]], str | None]] = {
    "adapter_sha256": _expected_adapter_sha256,
    "source_manifest_identity": _expected_source_manifest_identity,
}


def _local_files(root: Path) -> set[str]:
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


def verify_snapshot(
    plan: SnapshotPlan, root: Path, *, workers: int | None = None
) -> VerifyResult:
    """Hash every file of ``plan`` under ``root`` and run all checks. Never raises on a mismatch."""
    root = Path(root)
    checks: list[Check] = []
    planned = {f.path: f for f in plan.files}
    local = _local_files(root) if root.is_dir() else set()

    missing = sorted(p for p in planned if p not in local)
    checks.append(_summary("files_present", missing, len(planned)))
    extra = sorted(local - planned.keys())
    checks.append(
        Check("no_extra_files", not extra, ", ".join(extra[:10]) if extra else "none")
    )

    present = [planned[p] for p in sorted(planned) if p in local]
    bad_size = [f.path for f in present if (root / f.path).stat().st_size != f.size]
    checks.append(_summary("sizes", bad_size, len(present)))

    started = time.monotonic()
    # Largest first, so one big shard doesn't end up alone at the tail. hashlib releases the GIL.
    order = sorted(present, key=lambda f: -f.size)
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        hashed = list(
            pool.map(lambda f: hash_file(root / f.path, git_blob=f.lfs_sha256 is None), order)
        )
    digests = sorted(
        (FileDigest(f.path, n, s256, s1) for f, (s256, s1, n) in zip(order, hashed, strict=True)),
        key=lambda d: d.path,
    )
    hash_seconds = time.monotonic() - started
    sha = {d.path: d.sha256 for d in digests}

    lfs = [d for d in digests if planned[d.path].lfs_sha256]
    bad_lfs = [d.path for d in lfs if d.sha256 != (planned[d.path].lfs_sha256 or "").lower()]
    checks.append(_summary("lfs_sha256", bad_lfs, len(lfs)))
    blobs = [d for d in digests if not planned[d.path].lfs_sha256]
    bad_blob = [d.path for d in blobs if d.git_sha1 != (planned[d.path].blob_id or "").lower()]
    checks.append(_summary("git_blob_sha1", bad_blob, len(blobs)))

    required = required_manifest(plan.pin)
    for name in MANIFEST_NAMES:
        if name not in planned:
            if name == required:
                checks.append(Check(f"manifest:{name}", False, "required manifest not in plan"))
            continue
        if name not in sha:
            checks.append(Check(f"manifest:{name}", False, "manifest file missing"))
            continue
        try:
            entries = parse_sha256_manifest((root / name).read_text())
        except (ValueError, UnicodeDecodeError) as e:
            checks.append(Check(f"manifest:{name}", False, str(e)))
            continue
        bad = [p for digest, p in entries if sha.get(p) != digest]
        checks.append(_summary(f"manifest:{name}", bad, len(entries)))

    for key, want in plan.pin.expected:
        derive = _EXPECTED.get(key)
        if derive is None:
            checks.append(Check(f"expected:{key}", False, "no known derivation"))
            continue
        try:
            got = derive(root, sha)
        except (OSError, ValueError) as e:
            got, err = None, str(e)
        else:
            err = "source file missing" if got is None else ""
        ok = got is not None and got.lower() == want.lower()
        checks.append(Check(f"expected:{key}", ok, err or f"got {got}"))

    return VerifyResult(
        pin=plan.pin,
        root=str(root),
        plan_digest=plan_digest(plan),
        checks=tuple(checks),
        digests=tuple(digests),
        hash_seconds=round(hash_seconds, 3),
    )


def tool_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "openssl": ssl.OPENSSL_VERSION}
    for dist in ("flora-deploy", "huggingface_hub", "hf_xet", "modal"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def build_receipt(result: VerifyResult, *, timings: dict[str, float] | None = None) -> dict:
    pin = result.pin
    return {
        "schema": RECEIPT_SCHEMA,
        "ok": result.ok,
        "pin": {
            "key": pin.key,
            "repo": pin.repo,
            "revision": pin.revision,
            "role": pin.role,
            "allow_patterns": list(pin.allow_patterns) if pin.allow_patterns else None,
            "expected": dict(pin.expected),
        },
        "plan_digest": result.plan_digest,
        "files": len(result.digests),
        "bytes": result.total_bytes,
        "checks": [asdict(c) for c in result.checks],
        "digests": [asdict(d) for d in result.digests],
        "timings_s": {"hash": result.hash_seconds, **(timings or {})},
        "tools": tool_versions(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_receipt(receipt: dict, snapshot_dir: Path) -> Path:
    """Write the receipt whole (tmp file + rename), so a reader never sees a partial file."""
    path = receipt_path(snapshot_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def load_valid_receipt(plan: SnapshotPlan, snapshot_dir: Path) -> dict | None:
    """Return the existing receipt if it passed for exactly this plan and the files are still
    there with their sizes (cheap stat check, no re-hash); otherwise None."""
    path = receipt_path(snapshot_dir)
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    pin = receipt.get("pin", {})
    if not (
        receipt.get("schema") == RECEIPT_SCHEMA
        and receipt.get("ok") is True
        and pin.get("repo") == plan.pin.repo
        and pin.get("revision") == plan.pin.revision
        and receipt.get("plan_digest") == plan_digest(plan)
    ):
        return None
    for f in plan.files:
        p = Path(snapshot_dir) / f.path
        if not p.is_file() or p.stat().st_size != f.size:
            return None
    return receipt
