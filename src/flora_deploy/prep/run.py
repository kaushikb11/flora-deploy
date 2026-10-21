"""Download one pin into a snapshot dir, verify it and write its receipt.

Modal-agnostic: ``modal_app/prep_app.py`` passes the Volume's ``commit``; tests pass fakes.

Flow per pin:
1. Plan from the Hub tree and compare with the pinned ``RepoPin.plan_digest``. A mismatch raises
   ``PlanMismatchError`` before anything is downloaded (fail closed).
2. Reuse a receipt if ``verify.check_receipt`` accepts it, unless ``reverify``.
3. ``snapshot_download(local_dir=...)``: skips files its metadata says are complete, resumes
   partial ones.
4. Verify. On failure, repair once: delete the implicated files, files not in the plan, their
   ``.cache/huggingface/download`` metadata and any partial downloads (otherwise huggingface_hub
   would hand back the same bytes without fetching), download again, verify again.
5. Write the receipt (also when it failed, for inspection; a failed receipt is never reused).
   If verification still fails, raise ``SnapshotVerificationError`` so Modal's retry policy runs
   and the CLI reports the pin as an error.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath

from flora_deploy.config.pins import PINS_BY_KEY, RepoPin
from flora_deploy.layout import WEIGHTS_MOUNT, snapshot_path
from flora_deploy.prep.download import TreeLister, plan_snapshot
from flora_deploy.prep.verify import (
    VerifyResult,
    build_receipt,
    check_receipt,
    plan_digest,
    receipt_path,
    verify_snapshot,
    write_receipt,
)

Downloader = Callable[..., str]
HF_DOWNLOAD_META = Path(".cache") / "huggingface" / "download"


class PrepError(RuntimeError):
    """A pin could not be prepared. The message is complete and safe to show to the user.

    Takes a single string so it pickles cleanly across the Modal boundary."""


class PlanMismatchError(PrepError):
    """The Hub's file list for a pin no longer matches the digest pinned in ``pins.py``."""


class SnapshotVerificationError(PrepError):
    """Verification failed again after deleting and re-downloading the failing files."""


def _default_download(repo_id: str, **kw) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, **kw)


def summarize(receipt: dict, status: str, path: Path, *, timings: dict | None = None,
              note: str = "") -> dict:
    return {
        "key": receipt["pin"]["key"],
        "status": status,
        "ok": receipt["ok"],
        "revision": receipt["pin"]["revision"],
        "files": receipt["files"],
        "bytes": receipt["bytes"],
        "failures": [c for c in receipt["checks"] if not c["ok"]],
        # This run's timings. A skipped pin only planned; the receipt keeps the original ones.
        "timings_s": receipt["timings_s"] if timings is None else timings,
        "receipt": str(path),
        "receipt_created_utc": receipt.get("created_utc"),
        "note": note,
    }


def _failure_text(result: VerifyResult) -> str:
    return "; ".join(f"{c.name}: {c.detail}" for c in result.failures)


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def remove_for_redownload(dest: Path, paths: Iterable[str]) -> list[str]:
    """Delete ``paths`` (relative to ``dest``) and their huggingface_hub ``local_dir`` metadata
    and lock files, plus every partial download, so the next ``snapshot_download`` really
    fetches them. Returns the paths whose file existed."""
    dest = Path(dest)
    meta = dest / HF_DOWNLOAD_META
    removed = []
    for rel in paths:
        parts = PurePosixPath(rel)
        if parts.is_absolute() or ".." in parts.parts:
            continue  # defensive: never delete outside the snapshot
        if _unlink(dest / rel):  # unlinks a symlink itself, never its target
            removed.append(rel)
        md = meta / f"{rel}.metadata"
        _unlink(md)
        _unlink(md.with_suffix(".lock"))
    if meta.is_dir():
        for partial in meta.rglob("*.incomplete"):
            _unlink(partial)
    return removed


def prepare_snapshot(
    key: str,
    *,
    root: PurePosixPath | Path = WEIGHTS_MOUNT,
    api: TreeLister | None = None,
    download: Downloader = _default_download,
    commit: Callable[[], None] = lambda: None,
    reverify: bool = False,
    provenance: dict | None = None,
    pins: Mapping[str, RepoPin] = PINS_BY_KEY,
) -> dict:
    """Prepare pin ``key`` under ``root``; returns a summary of its receipt.

    ``reverify`` ignores any existing receipt and re-hashes everything. ``provenance`` carries
    ``git_commit``/``git_dirty`` from the CLI into the receipt. Raises ``PrepError`` subclasses."""
    pin = pins[key]
    dest = Path(snapshot_path(pin, PurePosixPath(root)))
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    t0 = time.monotonic()
    plan = plan_snapshot(pin, api)
    t_plan = round(time.monotonic() - t0, 3)
    got = plan_digest(plan)
    if pin.plan_digest is None or got != pin.plan_digest.lower():
        raise PlanMismatchError(
            f"{key}: the Hub file list for {pin.repo}@{pin.revision} does not match the pinned "
            f"plan_digest (got {got}, pinned {pin.plan_digest}). Nothing was downloaded."
        )

    if reverify:
        note = "re-verify requested"
    else:
        existing, note = check_receipt(plan, dest)
        if existing is not None:
            return summarize(existing, "skipped", receipt_path(dest), timings={"plan": t_plan},
                             note="receipt reused (no re-hash)")

    def fetch() -> float:
        t = time.monotonic()
        download(
            pin.repo,
            revision=pin.revision,
            allow_patterns=list(pin.allow_patterns) if pin.allow_patterns else None,
            local_dir=str(dest),
        )
        took = time.monotonic() - t
        commit()  # persist the bytes before the (long) verification pass
        return took

    dest.mkdir(parents=True, exist_ok=True)
    t_download = fetch()
    result = verify_snapshot(plan, dest)
    repair = None
    if not result.ok:
        first = _failure_text(result)
        removed = remove_for_redownload(dest, [*result.bad_files, *result.extra_files])
        t_download += fetch()
        result = verify_snapshot(plan, dest)
        repair = {"first_failures": first, "removed": removed, "ok": result.ok}

    receipt = build_receipt(
        result,
        timings={"plan": t_plan, "download": round(t_download, 3),
                 "total": round(time.monotonic() - t0, 3)},
        provenance=provenance,
        repair=repair,
    )
    path = write_receipt(receipt, dest)
    commit()
    if not result.ok:
        raise SnapshotVerificationError(
            f"{key}: verification failed after deleting and re-downloading the failing files: "
            f"{_failure_text(result)} (receipt: {path})"
        )
    status = "repaired" if repair else "prepared"
    return summarize(receipt, status, path, note=note if note != "no receipt" else "")
