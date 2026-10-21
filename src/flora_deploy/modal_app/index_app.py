"""``flora-index``: the ``flora index`` job on CPU containers (``img_cpu``; never deployed).

``run_index_remote`` (one small coordinator container) reads the dataset metadata, plans work
units by source video file (``jobs.indexing.plan_index``: each concatenated LeRobot v3 MP4 is
read by one unit) and fans them out with ``index_unit_remote.starmap``; bounded concurrency
(``max_containers``) keeps Hugging Face Hub traffic polite (each container fetches one file at a
time, with 429/5xx backoff, through ``validate.golden.HfHub``). Units write one whole-file receipt
per episode to the job store and return small summary rows; the coordinator writes
``index/summary.json`` and returns it. Everything is resumable (receipts with the same key are
skipped) and a failed unit only loses its own episodes until a rerun.

Inputs: without ``stage`` each unit downloads its files to ``/tmp`` and deletes them when done.
With ``stage`` (``input.stage_to: volume://<volume>/<dir>``) files are staged once into the store
Volume and kept, so reruns and later stages read the Volume instead of the Hub.

Store: ``modal-volume://<volume>[/<prefix>]`` (a ``VolumeStore`` on the mounted Volume, committed
after every unit) or ``s3://bucket[/prefix]`` (``BucketStore``; the containers need ``boto3``, not
in ``img_cpu`` yet, and credentials from a Secret named in ``FLORA_INDEX_SECRETS``).

Module-scope configuration (read at import, here and in the containers): ``FLORA_INDEX_VOLUME``
(store Volume, default ``flora-canary``, mounted at ``/vol/<name>``), ``FLORA_INDEX_MAX_CONTAINERS``
(default 8), ``FLORA_INDEX_SECRETS`` (comma-separated Secret names, e.g. an HF token).

    FLORA_INDEX_VOLUME=flora-canary uv run modal run src/flora_deploy/modal_app/index_app.py \\
        --config flora.yaml --store modal-volume://flora-canary/index-test --limit 10
"""

from __future__ import annotations

import json
import os
import time
from pathlib import PurePosixPath
from urllib.parse import urlparse

import modal

from flora_deploy.modal_app.images import cpu_image

VOLUME_NAME = os.environ.get("FLORA_INDEX_VOLUME", "flora-canary")
MOUNT = PurePosixPath("/vol") / VOLUME_NAME
MAX_CONTAINERS = int(os.environ.get("FLORA_INDEX_MAX_CONTAINERS", "8"))
SECRET_NAMES = tuple(s for s in os.environ.get("FLORA_INDEX_SECRETS", "").split(",") if s)
WORK_DIR = "/tmp/flora-index"
UNIT_TIMEOUT_S = 60 * 60
COORDINATOR_TIMEOUT_S = 6 * 60 * 60

app = modal.App("flora-index")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
_ENV = {
    "HF_HOME": "/tmp/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "FLORA_INDEX_VOLUME": VOLUME_NAME, "FLORA_INDEX_MAX_CONTAINERS": str(MAX_CONTAINERS),
    "FLORA_INDEX_SECRETS": ",".join(SECRET_NAMES),
}
_SECRETS = [modal.Secret.from_name(n) for n in SECRET_NAMES]


def volume_target(uri: str) -> tuple[str, str] | None:
    """``modal-volume://name/prefix`` or ``volume://name/prefix`` -> (name, prefix); else None."""
    u = urlparse(uri)
    if u.scheme not in ("modal-volume", "volume"):
        return None
    if not u.netloc:
        raise ValueError(f"{uri!r}: use modal-volume://<volume>[/<prefix>]")
    return u.netloc, u.path.strip("/")


def _open_store(uri: str):
    from flora_deploy.export.results import store_from_uri
    from flora_deploy.jobs.stores import VolumeStore

    target = volume_target(uri)
    if target is None:
        return store_from_uri(uri)
    name, prefix = target
    if name != VOLUME_NAME:
        raise ValueError(f"store Volume {name!r} is not the mounted one ({VOLUME_NAME!r}); set "
                         f"FLORA_INDEX_VOLUME={name}")
    root = MOUNT / prefix if prefix else MOUNT
    # commit_every huge: index_unit / finish_index commit explicitly once per unit
    return VolumeStore(str(root), volume, commit_every=10**9)


def _open_files(job, stage: str | None):
    from flora_deploy.jobs.indexing import open_dataset

    if stage:
        name, prefix = volume_target(stage) or (None, None)
        if name != VOLUME_NAME:
            raise ValueError(f"stage_to {stage!r} must be on the mounted Volume {VOLUME_NAME!r}")
        return open_dataset(job.dataset, work=str(MOUNT / prefix), staged=True,
                            commit=volume.commit)
    return open_dataset(job.dataset, work=WORK_DIR)


@app.function(
    image=cpu_image(), volumes={str(MOUNT): volume}, cpu=1.0, memory=2048,
    timeout=UNIT_TIMEOUT_S, max_containers=MAX_CONTAINERS, env=_ENV, secrets=_SECRETS,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
def index_unit_remote(job: dict, unit: dict, store_uri: str, stage: str | None = None,
                      force: bool = False) -> list[dict]:
    from flora_deploy.jobs.indexing import IndexJob, WorkUnit, index_unit

    j = IndexJob.from_dict(job)
    return index_unit(j, WorkUnit.from_dict(unit), _open_files(j, stage), _open_store(store_uri),
                      force=force, log=lambda m: print(m, flush=True))


@app.function(
    image=cpu_image(), volumes={str(MOUNT): volume}, cpu=0.5, memory=2048,
    timeout=COORDINATOR_TIMEOUT_S, env=_ENV, secrets=_SECRETS,
)
def run_index_remote(job: dict, store_uri: str, stage: str | None = None,
                     limit: int | None = None, max_episodes: int = 64,
                     force: bool = False) -> dict:
    from flora_deploy.jobs.indexing import IndexJob, finish_index, plan_index

    t0 = time.monotonic()
    store = _open_store(store_uri)  # fail fast on a bad store before any download
    j, units = plan_index(IndexJob.from_dict(job), _open_files(IndexJob.from_dict(job), stage),
                          limit=limit, max_episodes=max_episodes)
    if stage:
        volume.commit()  # staged metadata
    print(f"{j.dataset.name}@{j.revision}: {sum(len(u.episodes) for u in units)} episodes in "
          f"{len(units)} units", flush=True)
    args = [(j.to_dict(), u.to_dict(), store_uri, stage, force) for u in units]
    rows: list[dict] = []
    failed_units = 0
    for unit, res in zip(units, index_unit_remote.starmap(args, return_exceptions=True),
                         strict=True):
        if isinstance(res, BaseException):
            failed_units += 1
            print(f"{unit.unit_id}: FAILED {type(res).__name__}: {res}", flush=True)
            from flora_deploy.jobs.types import episode_uid

            rows += [{"uid": episode_uid(j.dataset.name, e["episode_id"]),
                      "episode_id": e["episode_id"], "status": "failed", "pool": j.pool,
                      "error": f"unit {unit.unit_id}: {type(res).__name__}: {res}"}
                     for e in unit.episodes]
        else:
            rows += res
    return finish_index(j, rows, store, units=len(units), run={
        "mode": "modal", "app": app.name, "failed_units": failed_units,
        "max_containers": MAX_CONTAINERS, "wall_s": round(time.monotonic() - t0, 1)})


@app.local_entrypoint()
def main(config: str, store: str, job: str = "", limit: int = 0, stage: str = "",
         max_episodes: int = 64, force: bool = False, out: str = "") -> None:
    from pathlib import Path

    from flora_deploy.config.schema import load_config
    from flora_deploy.jobs.indexing import IndexJob

    cfg = load_config(config)
    j = IndexJob.from_config(cfg, job or cfg.project)
    summary = run_index_remote.remote(j.to_dict(), store, stage or cfg.input.stage_to or None,
                                      limit or None, max_episodes, force)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if out:
        Path(out).write_text(text + "\n")
