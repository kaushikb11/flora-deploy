"""``compose``: a job's receipts' parts → deduped, validated ``flora_labels_v1`` outputs.

Input comes from ``jobs.progress`` only (``derive`` + ``compose_input``): it already finds every
part of done and quarantined shards, drops other config hashes (``foreign``), dedupes by
``records.record_key`` with the latest-source-wins tie-break and reports differing duplicates
(``conflicts``). Nothing here re-implements that.

Per config hash (``Manifest.config_hash`` of the job's latest plan unless given; each config has
its own manifest, ``JobPaths.config_manifest``) and per stage, compose writes, all as whole files.
Parts are read from where their shard kind keys them (its stage hash, ``jobs.store.JobPaths``),
so a config that reuses another config's finished ``segcap`` shards composes their records:

* ``<stage>.jsonl``: canonical ``StageRecord.to_json`` lines sorted by uid, then the record's
  earliest segment start;
* ``<stage>.parquet``: the flattened table (``stage_schema``); segment rows are exploded for
  ``segment`` / ``caption``, reward stages get their own columns;
* ``JobPaths.output(cfg, "summary", "json")``: counts, digests and problems, written **last**.

Only the formats in ``formats`` are written (the config's ``output.formats``; ``lerobot`` is
not implemented yet and falls back to Parquet, with a problem line).

**Generations** (review 6 M2). Stage files go to a generation directory named by the digest of
their contents, ``<cfg>/output/gen/<generation>/<stage>.<fmt>``, and ``summary.json`` (the
pointer: ``summary["files"]`` maps ``"<stage>.<fmt>"`` to those paths) is written after all of
them. A re-compose writes a complete new generation and then swaps the pointer, so a reader that
goes through the summary (``export.results``) always sees one complete set, even when a
re-compose dies half way. Old generations stay in the store (whole-file stores have no
delete); composing the same parts again reuses the same generation (byte-identical files).

**Status** (``summary["status"]``; never "complete" for planned work without records):

* ``complete``: every planned (uid, stage) has a record or a legitimate exclusion
  (NVIDIA-semantics failure, D15 poison pill) and, per stage, the excluded share is within
  ``max_error_rate`` with at least one record. Episodes the plan left out
  (``Manifest.excluded``: index failures, source errors, enforced envelope exclusions, no task
  text) count as excluded in every stage (``unplanned``);
* ``completed_with_errors``: the job resolved, but a stage excluded more than that (or has no
  record at all); outputs are written, ``summary["problems"]`` says why, ``complete`` is false;
* ``partial``: composed with ``partial=True`` before the job resolved (or after it failed).

A job with a failed shard (respawns exhausted, e.g. persistent infrastructure errors) is refused
without ``partial``.

Outputs are a pure function of the composed records, so composing twice gives byte-identical
files (idempotent). Compose refuses (``ComposeError``) on an invalid record, a malformed part, or
an incomplete job (unless ``partial=True``; missing keys are then counted in the summary).

Parquet layout (one table per stage; JSON columns are canonical JSON strings):

* common: ``uid, dataset, episode_id, stage, pool, certified, uncertified_reason, config_hash,
  audit_ref, schema, provenance`` (JSON), ``extra`` (JSON of payload fields the stage's columns
  do not cover, e.g. segments on a reward record; null when there are none);
* ``segment`` / ``caption``: one row per segment: ``segment_index, event_id, fine_event_id,
  start_s, end_s, segment_status, caption, segmentation_provenance, degradation`` (JSON),
  ``nv_metadata`` (JSON); a record without segments is one row with null segment columns;
* ``reward.terminal``: ``terminal_score, terminal_expected_score, terminal_probabilities,
  terminal_rationale``; ``reward.dense``: ``dense_segments`` (JSON), ``dense_per_step``;
  ``reward.rubric``: ``rubric_scores, rubric_total, rubric_samples, rubric_medoid_index``.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, get_args

import pyarrow as pa
import pyarrow.parquet as pq

from flora_deploy.jobs.attempts import JobFiles
from flora_deploy.jobs.manifest import load_manifest, stages_of
from flora_deploy.jobs.progress import Progress, compose_input, derive, read_errors
from flora_deploy.jobs.settings import MAX_EPISODE_ERROR_RATE
from flora_deploy.jobs.store import Store
from flora_deploy.jobs.types import MAX_STARTS_BEFORE_QUARANTINE, EpisodeRef
from flora_deploy.records import SCHEMA, Stage, StageRecord

STAGES: tuple[Stage, ...] = get_args(Stage)
SEGMENT_STAGES = frozenset({"segment", "caption"})
SUMMARY_SCHEMA = "flora_compose_summary_v1"

_JSON = pa.string()
COMMON = [
    pa.field("uid", pa.string(), nullable=False),
    pa.field("dataset", pa.string(), nullable=False),
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("stage", pa.string(), nullable=False),
    pa.field("pool", pa.string()),
    pa.field("certified", pa.bool_(), nullable=False),
    pa.field("uncertified_reason", pa.string()),
    pa.field("config_hash", pa.string(), nullable=False),
    pa.field("audit_ref", pa.string()),
    pa.field("schema", pa.string(), nullable=False),
    pa.field("provenance", _JSON, nullable=False),
    pa.field("extra", _JSON),
]
SEGMENT_COLS = [
    pa.field("segment_index", pa.int32()),
    pa.field("event_id", pa.string()),
    pa.field("fine_event_id", pa.string()),
    pa.field("start_s", pa.float64()),
    pa.field("end_s", pa.float64()),
    pa.field("segment_status", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("segmentation_provenance", pa.string()),
    pa.field("degradation", _JSON),
    pa.field("nv_metadata", _JSON),
]
REWARD_COLS: dict[str, list[pa.Field]] = {
    "reward.terminal": [
        pa.field("terminal_score", pa.int32()),
        pa.field("terminal_expected_score", pa.float64()),
        pa.field("terminal_probabilities", pa.list_(pa.float64())),
        pa.field("terminal_rationale", pa.string()),
    ],
    "reward.dense": [
        pa.field("dense_segments", _JSON),
        pa.field("dense_per_step", pa.list_(pa.int32())),
    ],
    "reward.rubric": [
        pa.field("rubric_scores", pa.list_(pa.float64())),
        pa.field("rubric_total", pa.float64()),
        pa.field("rubric_samples", pa.int32()),
        pa.field("rubric_medoid_index", pa.int32()),
    ],
}
# Which record payload field each stage's own columns cover (the rest goes to ``extra``).
_OWN = {"segment": "segments", "caption": "segments", "reward.terminal": "terminal",
        "reward.dense": "dense", "reward.rubric": "rubric"}
_PAYLOADS = ("segments", "terminal", "dense", "rubric")
FORMATS = ("jsonl", "parquet")


class ComposeError(ValueError):
    """Compose refused: invalid input or an incomplete job. ``problems`` lists every reason."""

    def __init__(self, message: str, problems: Iterable[str] = ()):
        self.problems = list(problems)
        super().__init__(message + "".join(f"\n  - {p}" for p in self.problems[:20]))


def cjson(obj: Any) -> str:
    """Canonical JSON (sorted keys, compact, NaN rejected)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage_schema(stage: str) -> pa.Schema:
    cols = SEGMENT_COLS if stage in SEGMENT_STAGES else REWARD_COLS[stage]
    return pa.schema(COMMON + cols, metadata={b"flora.schema": SCHEMA.encode(),
                                              b"flora.stage": stage.encode()})


# --- canonical ordering + flattening -------------------------------------------------------

def _first_start(rec: StageRecord) -> float:
    return min((s.start_s for s in rec.segments), default=0.0)


def sort_records(records: Iterable[StageRecord]) -> list[StageRecord]:
    """Canonical order: uid, then earliest segment start, then canonical JSON (total order)."""
    return sorted(records, key=lambda r: (r.uid, _first_start(r), r.to_json()))


def to_jsonl(records: Iterable[StageRecord]) -> bytes:
    return "".join(r.to_json() + "\n" for r in sort_records(records)).encode()


def _extra(d: dict[str, Any], stage: str) -> str | None:
    rest = {k: d[k] for k in _PAYLOADS if k != _OWN[stage] and d.get(k)}
    return cjson(rest) if rest else None


def flatten(rec: StageRecord, pool: str | None = None) -> list[dict[str, Any]]:
    """Parquet rows of one record (see the module docstring)."""
    d = asdict(rec)
    base = {
        "uid": rec.uid, "dataset": rec.dataset, "episode_id": rec.episode_id,
        "stage": rec.stage, "pool": pool, "certified": rec.certified,
        "uncertified_reason": rec.uncertified_reason,
        "config_hash": rec.provenance.config_hash, "audit_ref": rec.audit_ref,
        "schema": rec.schema, "provenance": cjson(d["provenance"]),
        "extra": _extra(d, rec.stage),
    }
    if rec.stage in SEGMENT_STAGES:
        if not rec.segments:
            return [{**base, **{f.name: None for f in SEGMENT_COLS}}]
        return [{**base, "segment_index": i, "event_id": s.event_id,
                 "fine_event_id": s.fine_event_id, "start_s": s.start_s, "end_s": s.end_s,
                 "segment_status": s.status, "caption": s.caption,
                 "segmentation_provenance": s.segmentation_provenance,
                 "degradation": cjson(s.degradation), "nv_metadata": cjson(s.nv_metadata)}
                for i, s in enumerate(rec.segments)]
    row = dict(base)
    if rec.stage == "reward.terminal":
        t = rec.terminal
        row.update(terminal_score=t and t.score, terminal_expected_score=t and t.expected_score,
                   terminal_probabilities=list(t.probabilities) if t else None,
                   terminal_rationale=t.rationale if t else None)
    elif rec.stage == "reward.dense":
        dn = d["dense"]
        row.update(dense_segments=cjson(dn["segments"]) if dn else None,
                   dense_per_step=list(dn["per_step"]) if dn else None)
    else:
        r = rec.rubric
        row.update(rubric_scores=list(r.scores) if r else None, rubric_total=r and r.total,
                   rubric_samples=r and r.samples, rubric_medoid_index=r and r.medoid_index)
    return [row]


def to_table(stage: str, records: Iterable[StageRecord],
             pools: dict[str, str] | None = None) -> pa.Table:
    pools = pools or {}
    rows = [row for r in sort_records(records) for row in flatten(r, pools.get(r.uid))]
    if stage in SEGMENT_STAGES:  # uid, then segment start time, then segment order
        rows.sort(key=lambda x: (x["uid"], x["start_s"] is not None, x["start_s"] or 0.0,
                                 x["segment_index"] or 0))
    return pa.Table.from_pylist(rows, schema=stage_schema(stage))


def to_parquet(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


# --- compose -------------------------------------------------------------------------------

class _DigestingStore:
    """Read-through wrapper that records the sha256 of every part file compose reads."""

    def __init__(self, store: Store):
        self.store = store
        self.parts: dict[str, str] = {}

    def put_bytes(self, path: str, data: bytes) -> None:
        self.store.put_bytes(path, data)

    def get_bytes(self, path: str) -> bytes:
        data = self.store.get_bytes(path)
        if "/parts/" in path:
            self.parts[path] = sha256(data)
        return data

    def exists(self, path: str) -> bool:
        return self.store.exists(path)

    def list(self, prefix: str) -> Iterator[str]:
        return self.store.list(prefix)


@dataclass
class ComposeResult:
    job_id: str
    config_hash: str
    records: dict[str, list[StageRecord]]  # stage -> canonical order
    summary: dict[str, Any]
    paths: dict[str, str] = field(default_factory=dict)  # "<stage>.<fmt>" | "summary" -> path


def _validate(rec: StageRecord, refs: dict[str, EpisodeRef],
              expected: set[tuple[str, str]]) -> list[str]:
    key = f"{rec.uid}/{rec.stage}"
    probs: list[str] = []
    try:
        if StageRecord.from_dict(json.loads(rec.to_json())) != rec:
            probs.append(f"{key}: record does not round-trip through JSON")
    except (TypeError, ValueError, KeyError) as e:
        probs.append(f"{key}: invalid record ({e})")
    if rec.stage not in STAGES:
        probs.append(f"{key}: unknown stage")
    if (rec.uid, rec.stage) not in expected:
        probs.append(f"{key}: not expected by the job's shards")
    ref = refs.get(rec.uid)
    if ref is not None and (ref.dataset, ref.episode_id) != (rec.dataset, rec.episode_id):
        probs.append(f"{key}: dataset/episode_id {rec.dataset}/{rec.episode_id} disagree with "
                     f"the manifest ({ref.dataset}/{ref.episode_id})")
    return probs


def _error_counts(jf: JobFiles, progress: Progress) -> dict[str, Any]:
    by_kind: Counter[str] = Counter()
    by_stage: Counter[str] = Counter()
    unreadable = 0
    for st in progress.shards.values():
        s = st.shard
        probe = jf.paths.errors(jf.config_hash, s.kind, s.shard_id, "x", 0)
        prefix = probe[: -len("x-0.jsonl")]
        for p in jf.store.list(prefix):
            if not p.endswith(".jsonl"):
                continue
            try:
                errs = read_errors(jf, p)
            except (KeyError, TypeError, ValueError):
                unreadable += 1
                continue
            for e in errs:
                by_kind[e.kind] += 1
                by_stage[e.stage] += 1
    return {"by_kind": dict(sorted(by_kind.items())), "by_stage": dict(sorted(by_stage.items())),
            "unreadable_files": unreadable}


def job_state(state: str, summary: dict[str, Any] | None) -> str:
    """A job's final state from the orchestrator's (or reconciler's) verdict and the composed
    summary, the same way everywhere (``sdk.Job.status``, ``flora run --local``; review 6 H2):
    compose may only downgrade a resolved ``done`` to ``completed_with_errors`` (its own rate
    guard found too many exclusions); every other state is the orchestrator's."""
    if state in ("done", "completed_with_errors") and summary is not None:
        if summary.get("status") == "completed_with_errors":
            return "completed_with_errors"
    return state


def output_path(paths: Any, cfg: str, generation: str, stage: str, fmt: str) -> str:
    """Where one generation's stage file lives (module docstring)."""
    base = paths.output(cfg, stage, fmt).rsplit("/", 1)
    return f"{base[0]}/gen/{generation}/{base[1]}"


def compose(store: Store, job_id: str, *, config_hash: str | None = None,
            partial: bool = False, limit: int = MAX_STARTS_BEFORE_QUARANTINE,
            write: bool = True, max_error_rate: float = MAX_EPISODE_ERROR_RATE,
            formats: Iterable[str] = FORMATS) -> ComposeResult:
    """Compose one job's outputs for one config hash (see the module docstring)."""
    wanted = [f for f in FORMATS if f in set(formats)]
    format_problems = ([] if wanted else [
        f"output.formats {sorted(set(formats))}: not implemented; wrote parquet instead"])
    wanted = wanted or ["parquet"]
    try:
        manifest = load_manifest(store, job_id, config_hash)
    except FileNotFoundError as e:
        if config_hash is None:
            raise
        raise ComposeError(str(e)) from None
    cfg = manifest.config_hash
    dstore = _DigestingStore(store)
    jf = JobFiles(dstore, job_id, cfg)  # type: ignore[arg-type]
    try:
        progress = derive(jf, manifest, limit=limit)
        ci = compose_input(jf, progress)
    except (KeyError, TypeError, ValueError) as e:
        raise ComposeError("unreadable compose input (malformed part file?)", [repr(e)]) from e

    refs = {e.uid: e for e in manifest.episodes}
    problems = [p for rec in ci.records.values() for p in _validate(rec, refs, ci.expected)]
    if problems:
        raise ComposeError(f"{len(problems)} invalid record(s)", problems)
    missing = sorted(ci.missing)
    failed_shards = sorted(sid for sid, s in progress.shards.items() if s.state == "failed")
    if not partial and failed_shards:
        raise ComposeError(
            f"job failed: {len(failed_shards)} shard(s) exhausted their retries (persistent "
            "infrastructure or python errors; see errors/ and the shard_failed records). Pass "
            "partial=True / --partial to compose what exists",
            [f"failed shard {s}" for s in failed_shards[:10]])
    if not partial and (missing or not progress.complete):
        unresolved = sorted(sid for sid, s in progress.shards.items()
                            if s.state not in ("done", "quarantined"))
        raise ComposeError("job is not complete (pass partial=True / --partial to compose what "
                           "exists)", [f"unresolved shard {s}" for s in unresolved[:10]]
                           + [f"missing {u}/{st}" for u, st in missing[:10]])

    pools = {e.uid: e.pool for e in manifest.episodes}
    by_stage: dict[str, list[StageRecord]] = defaultdict(list)
    for rec in ci.records.values():
        by_stage[rec.stage].append(rec)
    stages = sorted({st for _u, st in ci.expected} | set(by_stage), key=STAGES.index)

    data_by_key: dict[str, bytes] = {}
    paths: dict[str, str] = {}
    stage_summary: dict[str, Any] = {}
    episode_states: dict[str, Counter[str]] = defaultdict(Counter)
    for (_uid, kind), est in progress.episodes.items():
        for st in stages_of(kind):
            episode_states[st][est] += 1
    for stage in stages:
        recs = sort_records(by_stage.get(stage, []))
        by_stage[stage] = recs
        table = to_table(stage, recs, pools)
        if "jsonl" in wanted:
            data_by_key[f"{stage}.jsonl"] = to_jsonl(recs)
        if "parquet" in wanted:
            data_by_key[f"{stage}.parquet"] = to_parquet(table)
        cert = sum(r.certified for r in recs)
        per_pool: dict[str, Counter[str]] = defaultdict(Counter)
        for r in recs:
            c = per_pool[pools.get(r.uid) or "unknown"]
            c["records"] += 1
            c["certified"] += r.certified
        stage_summary[stage] = {
            "records": len(recs),
            "rows": table.num_rows,
            "certified": cert,
            "uncertified": len(recs) - cert,
            "certified_share": round(cert / len(recs), 6) if recs else None,
            "uncertified_reasons": dict(sorted(Counter(
                r.uncertified_reason for r in recs if not r.certified).items())),
            "by_pool": {k: dict(sorted(v.items())) for k, v in sorted(per_pool.items())},
            "expected": sum(1 for _u, st in ci.expected if st == stage),
            "missing": sum(1 for _u, st in missing if st == stage),
            "excused": sum(1 for _u, st in ci.excused if st == stage),
            "episodes": dict(sorted(episode_states[stage].items())),
        }

    generation = sha256("".join(f"{k} {sha256(v)}\n" for k, v in sorted(data_by_key.items())
                                ).encode())[:16]
    files: dict[str, bytes] = {}
    for key, data in data_by_key.items():
        stage, fmt = key.rsplit(".", 1)
        p = output_path(jf.paths, cfg, generation, stage, fmt)
        files[p] = data
        paths[key] = p

    poisoned = sorted(f"{u}/{k}" for u, k in progress.poisoned())
    resolved = progress.complete and not missing and not failed_shards
    unplanned = dict(getattr(manifest, "excluded", None) or {})
    problems = list(format_problems)
    # an episode an upstream kind excluded (balanced caption of a failed segment) is excused
    # here too, but counted once, at the upstream stage (``progress.UPSTREAM_KINDS``)
    upstream = progress.upstream_excluded()
    for stage in stages:
        c = stage_summary[stage]
        c["unplanned"] = len(unplanned)
        c["excused_upstream"] = up = sum(1 for _u, k in upstream if stage in stages_of(k))
        expected = c["expected"] - up
        if not expected:
            continue
        excluded = c["excused"] - up + len(unplanned)
        rate = excluded / (expected + len(unplanned))
        if c["records"] == 0:
            problems.append(f"{stage}: 0 records for {c['expected']} planned episodes "
                            f"({c['excused']} excluded, {c['missing']} missing)")
        elif rate > max_error_rate:
            problems.append(f"{stage}: {excluded} of {expected + len(unplanned)} episodes "
                            f"excluded ({rate:.0%} > {max_error_rate:.0%})"
                            + (f"; {len(unplanned)} left out of the plan" if unplanned else ""))
    status = ("partial" if not resolved else
              "completed_with_errors" if problems else "complete")
    part_lines = "".join(f"{p} {h}\n" for p, h in sorted(dstore.parts.items()))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "record_schema": SCHEMA,
        "job_id": job_id,
        "config_hash": cfg,
        "complete": status == "complete",
        "status": status,
        "problems": problems,
        "max_error_rate": max_error_rate,
        "stage_hashes": dict(manifest.stage_hashes),
        "partial": partial,
        "stages": stage_summary,
        "records": sum(len(v) for v in by_stage.values()),
        "duplicates_resolved": ci.duplicates,
        "conflicts": [f"{u}/{st}" for u, st in sorted(ci.conflicts)],
        "foreign_records_dropped": ci.foreign,
        "poison_pills": poisoned,
        "unplanned_episodes": unplanned,
        "generation": generation,
        "formats": wanted,
        "files": dict(sorted(paths.items())),
        "errors": _error_counts(jf, progress),
        "shards": dict(sorted(Counter(s.state for s in progress.shards.values()).items())),
        "inputs": {"parts": len(dstore.parts), "digest": sha256(part_lines.encode())},
        "outputs": {p: {"sha256": sha256(b), "bytes": len(b)} for p, b in sorted(files.items())},
    }
    summary_path = jf.paths.output(cfg, "summary", "json")
    paths["summary"] = summary_path
    if write:  # the generation's files first, the pointer last (module docstring)
        for p, data in files.items():
            store.put_bytes(p, data)
        store.put_bytes(summary_path, (json.dumps(summary, sort_keys=True, indent=2,
                                                  allow_nan=False) + "\n").encode())
    return ComposeResult(job_id, cfg, dict(by_stage), summary, paths)
