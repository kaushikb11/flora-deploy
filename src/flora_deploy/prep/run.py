"""Download one pin into a snapshot dir, verify it and write its receipt.

Modal-agnostic: ``modal_app/prep_app.py`` passes the Volume's ``commit``; tests pass fakes.
Resumable: ``snapshot_download(local_dir=...)`` skips files it already completed and resumes
partial ones, and a passing receipt for the same plan short-circuits everything.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import WEIGHTS_MOUNT, snapshot_path
from flora_deploy.prep.download import TreeLister, plan_snapshot
from flora_deploy.prep.verify import (
    build_receipt,
    load_valid_receipt,
    receipt_path,
    verify_snapshot,
    write_receipt,
)

Downloader = Callable[..., str]


def _default_download(repo_id: str, **kw) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, **kw)


def summarize(receipt: dict, status: str, path: Path) -> dict:
    return {
        "key": receipt["pin"]["key"],
        "status": status,
        "ok": receipt["ok"],
        "revision": receipt["pin"]["revision"],
        "files": receipt["files"],
        "bytes": receipt["bytes"],
        "failures": [c for c in receipt["checks"] if not c["ok"]],
        "timings_s": receipt["timings_s"],
        "receipt": str(path),
    }


def prepare_snapshot(
    key: str,
    *,
    root: PurePosixPath | Path = WEIGHTS_MOUNT,
    api: TreeLister | None = None,
    download: Downloader = _default_download,
    commit: Callable[[], None] = lambda: None,
) -> dict:
    """Prepare pin ``key`` under ``root``; returns a summary of its receipt."""
    pin = PINS_BY_KEY[key]
    dest = Path(snapshot_path(pin, PurePosixPath(root)))
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    t0 = time.monotonic()
    plan = plan_snapshot(pin, api)
    t_plan = time.monotonic() - t0
    if (existing := load_valid_receipt(plan, dest)) is not None:
        return summarize(existing, "skipped", receipt_path(dest))

    dest.mkdir(parents=True, exist_ok=True)
    t1 = time.monotonic()
    download(
        pin.repo,
        revision=pin.revision,
        allow_patterns=list(pin.allow_patterns) if pin.allow_patterns else None,
        local_dir=str(dest),
    )
    t_download = time.monotonic() - t1
    commit()  # persist the bytes before the (long) verification pass

    result = verify_snapshot(plan, dest)
    receipt = build_receipt(
        result,
        timings={
            "plan": round(t_plan, 3),
            "download": round(t_download, 3),
            "total": round(time.monotonic() - t0, 3),
        },
    )
    path = write_receipt(receipt, dest)
    commit()
    return summarize(receipt, "prepared" if result.ok else "failed", path)
