"""Reader API over composed outputs (``flora results``): records, Parquet tables, the summary.

Outputs are read only through their ``summary.json`` (compose writes it last; its ``files``
name the stage files of one complete generation, ``export.compose``), so a reader never sees a
half-composed or mixed set, even while a re-compose runs. ``config_hash`` defaults to the job
manifest's. A stage composed without JSONL (``output.formats: [parquet]``) is read from its
Parquet table, and the other way round.

``store_from_uri`` maps a ``--store`` URI to a ``Store``:

* ``file:///abs/dir`` (or a bare path) → ``LocalStore``;
* ``s3://bucket[/prefix]`` → ``BucketStore`` (boto3; endpoint from ``AWS_ENDPOINT_URL`` or
  ``endpoint_url=`` for R2/GCS; ``gs://`` / ``r2://`` are refused with the way to do that,
  ``config.schema.unsupported_sink_message``);
* ``modal-volume://name[/path]`` → ``VolumeStore`` over the Volume's mount inside a Modal
  container (``/vol/<name>`` unless ``volume_mount=`` / ``FLORA_VOLUME_MOUNT`` says otherwise;
  no Modal API is called, Modal commits the container's writes when it exits); where it is not
  mounted (a laptop), ``jobs.stores.ClientVolumeStore`` through the Modal client's Volume file
  API (``environment=``: the Modal environment; no container is started).
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq

from flora_deploy.export.compose import SEGMENT_STAGES, sort_records
from flora_deploy.jobs.manifest import load_manifest
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.records import StageRecord


class ResultsNotFound(FileNotFoundError):
    pass


def store_from_uri(uri: str, *, endpoint_url: str | None = None,
                   volume_mount: str | None = None, environment: str | None = None,
                   create: bool = False) -> Store:
    from flora_deploy.config.schema import unsupported_sink_message
    from flora_deploy.jobs.stores import BucketStore, ClientVolumeStore, LocalStore, VolumeStore

    why = unsupported_sink_message(uri)
    if why:
        raise ValueError(f"store {uri!r}: {why}")
    u = urlparse(uri)
    if u.scheme in ("", "file"):
        path = unquote(u.path) if u.scheme == "file" else uri
        if u.scheme == "file" and u.netloc not in ("", "localhost"):
            raise ValueError(f"store {uri!r}: file:// URIs take an absolute path (file:///dir)")
        if not path:
            raise ValueError(f"store {uri!r}: empty path")
        return LocalStore(path)
    if u.scheme == "s3":
        if not u.netloc:
            raise ValueError(f"store {uri!r}: s3://<bucket>[/prefix]")
        return BucketStore(u.netloc, prefix=u.path.strip("/"),
                           endpoint_url=endpoint_url or os.environ.get("AWS_ENDPOINT_URL"))
    if u.scheme == "modal-volume":
        if not u.netloc:
            raise ValueError(f"store {uri!r}: modal-volume://<name>")
        mount = Path(volume_mount or os.environ.get("FLORA_VOLUME_MOUNT") or f"/vol/{u.netloc}")
        sub = u.path.strip("/")
        if mount.is_dir():
            return VolumeStore(mount / sub if sub else mount)
        return ClientVolumeStore.from_uri(f"modal-volume://{u.netloc}/{sub}",
                                          environment=environment, create=create)
    raise ValueError(f"store {uri!r}: use file://, s3:// or modal-volume://")


def resolve_config_hash(store: Store, job_id: str, config_hash: str | None = None) -> str:
    return config_hash or load_manifest(store, job_id).config_hash


def _path(store: Store, job_id: str, stage: str, fmt: str, config_hash: str | None
          ) -> str | None:
    """The stage file the current summary points at (``None``: not composed in ``fmt``)."""
    cfg = resolve_config_hash(store, job_id, config_hash)
    paths = JobPaths(job_id)
    sp = paths.output(cfg, "summary", "json")
    if not store.exists(sp):
        raise ResultsNotFound(f"job {job_id!r} (config {cfg}) has not been composed yet; run "
                              "`flora compose`")
    summary = json.loads(store.get_bytes(sp))
    files = summary.get("files")
    if files is None:  # composed before generations: the fixed paths
        legacy = paths.output(cfg, stage, fmt)
        return legacy if store.exists(legacy) else None
    path = files.get(f"{stage}.{fmt}")
    if path is None and stage not in summary.get("stages", {}):
        raise ResultsNotFound(f"no {stage!r} output for job {job_id!r} (config {cfg})")
    return path


def _read(store: Store, job_id: str, stage: str, fmt: str, config_hash: str | None) -> bytes:
    path = _path(store, job_id, stage, fmt, config_hash)
    if path is None:
        other = "parquet" if fmt == "jsonl" else "jsonl"
        if _path(store, job_id, stage, other, config_hash) is None:
            raise ResultsNotFound(f"no {stage!r} output for job {job_id!r}")
        recs = (records_from_table(pq.read_table(io.BytesIO(
            _read(store, job_id, stage, "parquet", config_hash))))
            if other == "parquet" else
            [StageRecord.from_json(x) for x in _read(store, job_id, stage, "jsonl", config_hash)
             .decode().splitlines() if x.strip()])
        from flora_deploy.export.compose import to_jsonl, to_parquet, to_table

        return to_jsonl(recs) if fmt == "jsonl" else to_parquet(to_table(stage, recs))
    if not store.exists(path):
        raise ResultsNotFound(f"no {stage!r} output for job {job_id!r} ({path} is missing)")
    return store.get_bytes(path)


def load_summary(store: Store, job_id: str, *, config_hash: str | None = None) -> dict[str, Any]:
    cfg = resolve_config_hash(store, job_id, config_hash)
    path = JobPaths(job_id).output(cfg, "summary", "json")
    if not store.exists(path):
        raise ResultsNotFound(f"job {job_id!r} (config {cfg}) has not been composed yet")
    return json.loads(store.get_bytes(path))


def read_jsonl(store: Store, job_id: str, stage: str, *, config_hash: str | None = None) -> bytes:
    """The stage's canonical JSONL bytes."""
    return _read(store, job_id, stage, "jsonl", config_hash)


def read_parquet_bytes(store: Store, job_id: str, stage: str, *,
                       config_hash: str | None = None) -> bytes:
    return _read(store, job_id, stage, "parquet", config_hash)


def iter_records(store: Store, job_id: str, stage: str, *,
                 config_hash: str | None = None) -> Iterator[StageRecord]:
    """Composed records of one stage, validated, in canonical order."""
    for line in read_jsonl(store, job_id, stage, config_hash=config_hash).decode().splitlines():
        if line.strip():
            yield StageRecord.from_json(line)


def load_parquet(store: Store, job_id: str, stage: str, *,
                 config_hash: str | None = None) -> pa.Table:
    return pq.read_table(io.BytesIO(read_parquet_bytes(store, job_id, stage,
                                                       config_hash=config_hash)))


def _j(v: str | None) -> Any:
    return None if v is None else json.loads(v)


def records_from_table(table: pa.Table) -> list[StageRecord]:
    """Rebuild ``StageRecord``s from a composed Parquet table (the inverse of
    ``compose.to_table``), in canonical record order."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        groups.setdefault((row["uid"], row["stage"]), []).append(row)
    out: list[StageRecord] = []
    for (_uid, stage), rows in groups.items():
        r0 = rows[0]
        d: dict[str, Any] = {
            "uid": r0["uid"], "dataset": r0["dataset"], "episode_id": r0["episode_id"],
            "stage": stage, "certified": r0["certified"],
            "uncertified_reason": r0["uncertified_reason"], "audit_ref": r0["audit_ref"],
            "schema": r0["schema"], "provenance": _j(r0["provenance"]),
        }
        d.update(_j(r0["extra"]) or {})
        if stage in SEGMENT_STAGES:
            segs = sorted((r for r in rows if r["segment_index"] is not None),
                          key=lambda r: r["segment_index"])
            d["segments"] = [{
                "event_id": r["event_id"], "start_s": r["start_s"], "end_s": r["end_s"],
                "status": r["segment_status"], "caption": r["caption"],
                "fine_event_id": r["fine_event_id"],
                "segmentation_provenance": r["segmentation_provenance"],
                "degradation": _j(r["degradation"]), "nv_metadata": _j(r["nv_metadata"]),
            } for r in segs]
        elif stage == "reward.terminal" and r0["terminal_score"] is not None:
            d["terminal"] = {"score": r0["terminal_score"],
                             "expected_score": r0["terminal_expected_score"],
                             "probabilities": r0["terminal_probabilities"],
                             "rationale": r0["terminal_rationale"]}
        elif stage == "reward.dense" and r0["dense_per_step"] is not None:
            d["dense"] = {"segments": _j(r0["dense_segments"]),
                          "per_step": r0["dense_per_step"]}
        elif stage == "reward.rubric" and r0["rubric_scores"] is not None:
            d["rubric"] = {"scores": r0["rubric_scores"], "total": r0["rubric_total"],
                           "samples": r0["rubric_samples"],
                           "medoid_index": r0["rubric_medoid_index"],
                           "parsed_samples": r0.get("rubric_parsed_samples")}
        out.append(StageRecord.from_dict(d))
    return sort_records(out)
