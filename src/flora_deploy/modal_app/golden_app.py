"""``flora-golden-stage``: stage the golden selection into the ``flora-golden`` Volume (CPU only).

One ``img_cpu`` container runs ``validate.golden.stage_all`` over the whole selection,
sequentially (HF rate limits; downloads dominate). It writes only under ``golden/v1/`` and is
resumable: completed episodes (a ``record.json`` with the current selection digest) are skipped.

    uv run modal run src/flora_deploy/modal_app/golden_app.py [--only ID,...] [--force] \
        [--out manifest.json]

The selection JSON is read locally and passed to the function (``add_local_python_source``
ships ``.py`` files only). The ephemeral app is never deployed. A watchdog ends the container
after ``HARD_LIFETIME_S``; the deadline stops starting new datasets a few minutes earlier.

Sizing: 2 physical cores (PyAV demux/decode for the index pass is single-threaded per stream),
8 GiB (scan arrays + one parquet row group of embedded MP4s), default ephemeral disk (the largest
dataset group downloads ~1.5 GB of concatenated files, deleted after its cuts are made).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import modal

from flora_deploy.layout import GOLDEN_MOUNT, GOLDEN_VOLUME
from flora_deploy.modal_app.images import cpu_image

app = modal.App("flora-golden-stage")
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)

CPU_CORES = 2.0
MEMORY_MIB = 8 * 1024
HARD_LIFETIME_S = 50 * 60
DEADLINE_S = 42 * 60  # no new dataset group after this
WORK_DIR = Path("/tmp/golden-work")


def _watchdog(limit_s: float = HARD_LIFETIME_S) -> None:
    def kill():
        print(f"flora-golden-stage watchdog: {limit_s:.0f}s lifetime reached, exiting", flush=True)
        os._exit(17)

    t = threading.Timer(limit_s, kill)
    t.daemon = True
    t.start()


@app.function(
    image=cpu_image(),
    volumes={str(GOLDEN_MOUNT): golden},
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=HARD_LIFETIME_S + 60,
    env={"HF_HOME": "/tmp/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
)
def stage(selection: dict, only: list[str] | None = None, force: bool = False) -> dict:
    _watchdog()
    from flora_deploy.validate.golden import HfHub, stage_all

    t0 = time.monotonic()
    hub = HfHub()
    manifest = stage_all(
        selection, hub=hub, root=GOLDEN_MOUNT, work=WORK_DIR, commit=golden.commit,
        only=only or (), force=force, deadline=t0 + DEADLINE_S,
        log=lambda m: print(m, flush=True),
    )
    manifest["run"] = {"hub_requests": hub.requests, "wall_s": round(time.monotonic() - t0, 1)}
    return manifest


@app.local_entrypoint()
def main(only: str = "", force: bool = False, out: str = "") -> None:
    from flora_deploy.validate.golden import load_selection, pool_counts, validate_selection

    selection = validate_selection(load_selection())
    print("selection:", pool_counts(selection))
    manifest = stage.remote(selection, [s for s in only.split(",") if s], force)
    print(json.dumps({"counts": manifest["counts"], "complete": manifest["complete"],
                      "failed": manifest["failed"], "missing": manifest["missing"],
                      "run": manifest.get("run")}, indent=2))
    if out:
        Path(out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"manifest written to {out}")
