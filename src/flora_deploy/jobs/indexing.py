"""The ``flora index`` job (PLAN.md §8 "index", D7, D13): dataset -> per-episode index receipts.

Modal-agnostic. ``modal_app/index_app.py`` runs the same three steps on CPU containers and
``flora index --local`` runs them in-process:

1. **plan** (:func:`plan_index`): read the dataset metadata (``media.sources`` readers for LeRobot
   v2.1 / v3 and MP4 directories; Hub datasets through W3-C's ``validate.golden.DatasetMirror`` at
   the pinned revision), and group episodes by their **source video files** (the sorted set of
   camera files), so each concatenated LeRobot v3 MP4 is demuxed once, by one unit. Small groups
   (v2.1 / MP4 files: one episode each) are packed into units of up to ``max_episodes`` episodes;
   a group is never split.
2. **index** (:func:`index_unit`), per episode: ``verify_episode`` (the metadata is not trusted:
   wrong-file ranges, frame-count mismatches and missing files become a ``source_error``
   receipt), ``index_episode`` with one ``ScanCache`` per unit (``decode=True`` for the exact
   preset, as ``validate.golden`` stages oracle candidates), the envelope verdict for the job's
   embodiment and instruction source, and one whole-file receipt at ``JobPaths.index(uid)``.
3. **summarize** (:func:`finish_index`): ``flora/<job>/index/summary.json`` with counts,
   durations per pool / camera count, envelope and ineligibility reasons, source errors and
   codecs, plus ``groups`` in the shape ``flora plan --index`` reads (``WorkloadSummary``).

Resume: an episode is skipped when its receipt's ``key`` matches (:meth:`IndexJob.resume_key`:
dataset name + revision, the episode's camera ranges, embodiment, instruction source, decode mode
and :data:`INDEX_CODE_VERSION`). A Hub dataset's revision is its pinned commit; a local
directory's is a digest of its metadata contents and video file sizes. Unexpected errors
(network) write no receipt, so a rerun retries them.

The envelope **policy** (``warn | enforce | ignore``) never changes a receipt (it records the
verdict as measured, ``EpisodeRef.envelope_ok``); it decides which episodes are *planned*:
``enforce`` drops out-of-envelope episodes, ``warn`` / ``ignore`` keep them (D13). Episodes with a
source error are never planned. :func:`load_indexes` rebuilds ``EpisodeIndex`` objects from the
receipts for ``jobs.manifest.build_manifest``.

Receipt source URIs are **dataset-relative** (``videos/.../file-000.mp4``), so receipts do not
depend on where a container staged the files; pass ``root=`` to :func:`episode_index_from_receipt`
to make them local paths again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import EpisodeRef, episode_uid
from flora_deploy.media.types import EnvelopeVerdict, EpisodeIndex, SourceRef, StreamIndex

RECEIPT_SCHEMA = "flora_index_receipt_v1"
SUMMARY_SCHEMA = "flora_index_summary_v1"
INDEX_VERSION = "2"  # bump when the receipt layout or the indexing procedure changes
# 2: receipts carry ``instruction_text`` (the resolved instruction; review-4 C2)
MAX_EPISODES_PER_UNIT = 64
ENVELOPE_POLICIES = ("warn", "enforce", "ignore")
# The code that decides every receipt value; its digest is part of the index code version.
_CODE_FILES = ("media/index.py", "media/types.py", "media/sources/base.py",
               "media/sources/lerobot_v21.py", "media/sources/lerobot_v3.py",
               "media/sources/mp4_dir.py")
_PROFILES = {"human": ("human", False), "yam": ("yam", False), "yam32": ("yam", True),
             "dex": ("dex", False), "tianji": ("tianji", False)}

Log = Callable[[str], None]


def _code_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in _CODE_FILES:
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return h.hexdigest()[:12]


INDEX_CODE_VERSION = f"{INDEX_VERSION}+{_code_digest()}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------------------------
# Job description (JSON-able: it travels to Modal containers)


@dataclass(frozen=True)
class DatasetSpec:
    """Where the dataset is: a Hub repo at a pinned revision (``subdir`` optional) or a local
    directory (``local_path``)."""

    format: str  # lerobot_v21 | lerobot_v3 | mp4_dir
    repo: str | None = None
    revision: str | None = None
    subdir: str = ""
    local_path: str | None = None
    cameras: dict[str, str] | None = None  # dataset camera key -> role

    def __post_init__(self) -> None:
        if self.format not in ("lerobot_v21", "lerobot_v3", "mp4_dir"):
            raise ValueError(f"unknown input format {self.format!r}")
        if (self.repo is None) == (self.local_path is None):
            raise ValueError("a dataset is either a Hub repo or a local directory")

    @property
    def name(self) -> str:
        """Dataset identifier (``EpisodeIndex.dataset``; episode uids derive from it)."""
        if self.repo is None:
            return str(self.local_path)
        return f"{self.repo}/{self.subdir}" if self.subdir else self.repo

    @property
    def key(self) -> str:
        """Path-safe directory name of a local mirror of this dataset."""
        base = (self.repo or Path(str(self.local_path)).name).replace("/", "--")
        return f"{base}@{(self.revision or 'local')[:12]}"

    @classmethod
    def from_config(cls, inp: Any) -> DatasetSpec:
        """From ``config.schema.InputSection``: ``path`` is the repo subdirectory with ``repo``,
        the local directory without it."""
        cams = dict(inp.cameras) if inp.cameras else None
        if inp.repo is not None:
            return cls(format=inp.format, repo=inp.repo, revision=inp.revision,
                       subdir=(inp.path or "").strip("/"), cameras=cams)
        return cls(format=inp.format, local_path=str(inp.path), cameras=cams)


@dataclass(frozen=True)
class IndexJob:
    job_id: str
    dataset: DatasetSpec
    pool: str  # embodiment: human | yam | yam32 | dex | tianji
    envelope: str = "warn"  # D13 policy
    decode: bool = True  # decode-mode index for oracle-eligible cameras
    instruction: str = "dataset_task"  # dataset_task | fixed | none
    instruction_text: str | None = None
    revision: str = ""  # resolved dataset revision (Hub commit or local digest)

    def __post_init__(self) -> None:
        if self.pool not in _PROFILES:
            raise ValueError(f"unknown pool {self.pool!r}; use {sorted(_PROFILES)}")
        if self.envelope not in ENVELOPE_POLICIES:
            raise ValueError(f"unknown envelope policy {self.envelope!r}")

    @classmethod
    def from_config(cls, cfg: Any, job_id: str, *, decode: bool | None = None) -> IndexJob:
        ins = cfg.pipeline.episode_instruction
        return cls(job_id=job_id, dataset=DatasetSpec.from_config(cfg.input),
                   pool=cfg.pipeline.embodiment, envelope=cfg.pipeline.envelope,
                   decode=(cfg.preset == "exact") if decode is None else decode,
                   instruction=ins.source, instruction_text=ins.text,
                   revision=cfg.input.revision or "")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> IndexJob:
        return cls(**{**d, "dataset": DatasetSpec(**d["dataset"])})

    @property
    def profile(self) -> tuple[str, bool]:
        return _PROFILES[self.pool]

    def instruction_for(self, task: str | None) -> str:
        if self.instruction == "fixed":
            return self.instruction_text or ""
        return "" if self.instruction == "none" else task or ""

    def resume_key(self, ep: Mapping[str, Any]) -> str:
        """Everything a receipt's values depend on (not the envelope policy)."""
        return _sha({
            "v": INDEX_CODE_VERSION, "dataset": self.dataset.name, "revision": self.revision,
            "format": self.dataset.format, "episode": ep, "pool": self.pool,
            "decode": self.decode, "instruction": [self.instruction, self.instruction_text],
        })[:32]


# ---------------------------------------------------------------------------------------------
# Dataset access


def _reader(spec: DatasetSpec, root: Path):
    from flora_deploy.media.sources import LeRobotV3Source, LeRobotV21Source, Mp4DirSource

    cls = {"lerobot_v21": LeRobotV21Source, "lerobot_v3": LeRobotV3Source,
           "mp4_dir": Mp4DirSource}[spec.format]
    return cls(root, camera_map=spec.cameras, name=spec.name)


def episode_to_dict(ep: Any, root: Path) -> dict:
    """``media.sources.EpisodeRef`` -> JSON with dataset-relative camera paths."""
    cams = {}
    for role, ref in sorted(ep.cameras.items()):
        rel = Path(ref.uri).relative_to(root).as_posix()
        cams[role] = [rel, ref.start_s, ref.end_s]
    extra = {k: v for k, v in ep.extra.items() if k in ("episode_index", "length")}
    return {"episode_id": ep.episode_id, "task": ep.task, "cameras": cams, "extra": extra}


def episode_from_dict(d: Mapping[str, Any], root: Path):
    from flora_deploy.media.sources import EpisodeRef as SourceEpisode

    cams = {role: SourceRef(str(Path(root) / rel), s, e)
            for role, (rel, s, e) in d["cameras"].items()}
    return SourceEpisode(episode_id=d["episode_id"], cameras=cams, task=d.get("task"),
                         extra=dict(d.get("extra") or {}))


class DatasetFiles:
    """A local view of the dataset: ``root``, its episode list and the files units need."""

    root: Path

    def list_episodes(self) -> list[dict]:
        raise NotImplementedError

    def ensure(self, rel: str) -> Path:
        raise NotImplementedError

    def release(self, rels: Iterable[str]) -> None:
        """Drop downloaded copies once a unit is done (no-op for local / staged datasets)."""

    def revision(self) -> str:
        raise NotImplementedError

    def commit(self) -> None:
        """Make staged files durable (a Modal Volume commit); no-op elsewhere."""


class LocalFiles(DatasetFiles):
    def __init__(self, spec: DatasetSpec, root: str | os.PathLike[str] | None = None):
        self.spec = spec
        self.root = Path(root if root is not None else str(spec.local_path))

    def list_episodes(self) -> list[dict]:
        return [episode_to_dict(e, self.root) for e in _reader(self.spec, self.root)
                .list_episodes()]

    def ensure(self, rel: str) -> Path:
        return self.root / rel  # verify_episode reports a missing file

    def revision(self) -> str:
        """Digest of the metadata contents and every other file's (path, size)."""
        h = hashlib.sha256()
        for dirpath, dirs, files in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                p = Path(dirpath, name)
                rel = p.relative_to(self.root).as_posix()
                if rel.startswith("meta/") or p.suffix == ".txt":
                    h.update(f"{rel}\0".encode() + p.read_bytes() + b"\0")
                else:
                    h.update(f"{rel}\0{p.stat().st_size}\0".encode())
        return "local-" + h.hexdigest()[:24]


class HubFiles(DatasetFiles):
    """A partial mirror of a Hub dataset at its pinned revision (``validate.golden``'s
    ``DatasetMirror``: one request at a time with 429/5xx backoff). ``staged`` keeps the files
    (a Volume stage dir); otherwise :meth:`release` deletes a unit's downloads."""

    def __init__(self, spec: DatasetSpec, work: str | os.PathLike[str], *, hub: Any = None,
                 staged: bool = False, commit: Callable[[], None] | None = None):
        from flora_deploy.validate.golden import DatasetMirror, HfHub

        if not spec.revision:
            raise ValueError(f"{spec.repo}: a Hub dataset needs a pinned revision")
        self.spec, self.staged, self._commit = spec, staged, commit
        self.hub = hub if hub is not None else HfHub()
        ds = {"repo": spec.repo, "revision": spec.revision, "format": spec.format,
              "subdir": spec.subdir}
        self.mirror = DatasetMirror(spec.key, ds, self.hub, Path(work))
        self.root = self.mirror.root

    def _sub(self, path: str) -> str:
        sub = self.mirror.sub
        return path[len(sub) + 1:] if sub else path

    def list_episodes(self) -> list[dict]:
        if self.spec.format == "mp4_dir":
            return self._list_mp4_dir()
        self.mirror.fetch("meta/info.json")
        if self.spec.format == "lerobot_v3":
            for path in self.hub.list_files(self.spec.repo, self.spec.revision,
                                            self.mirror.repo_path("meta/episodes")):
                if path.endswith(".parquet"):
                    self.mirror.fetch(self._sub(path))
        else:
            self.mirror.fetch("meta/episodes.jsonl")
        return [episode_to_dict(e, self.root) for e in _reader(self.spec, self.root)
                .list_episodes()]

    def _list_mp4_dir(self) -> list[dict]:
        """Mp4DirSource reads names only: run it over a skeleton of empty files (task ``.txt``
        files are fetched for real), then map the paths back to the dataset root."""
        from flora_deploy.media.sources.mp4_dir import VIDEO_SUFFIXES

        skel = self.mirror.local.parent / f"{self.spec.key}.skeleton"
        shutil.rmtree(skel, ignore_errors=True)
        for path in self.hub.list_files(self.spec.repo, self.spec.revision, self.mirror.sub):
            rel = self._sub(path)
            dest = skel / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if Path(rel).suffix.lower() in VIDEO_SUFFIXES:
                dest.touch()
            elif rel.endswith(".txt"):
                shutil.copyfile(self.mirror.fetch(rel), dest)
        try:
            eps = _reader(self.spec, skel).list_episodes()
            out = [episode_to_dict(e, skel) for e in eps]
        finally:
            shutil.rmtree(skel, ignore_errors=True)
        return out

    def ensure(self, rel: str) -> Path:
        from huggingface_hub.errors import EntryNotFoundError

        from flora_deploy.media.sources import SourceError

        try:
            return self.mirror.fetch(rel)
        except EntryNotFoundError as e:
            raise SourceError(f"video file missing: {self.mirror.repo_path(rel)} not in "
                              f"{self.spec.repo}@{self.spec.revision}") from e

    def release(self, rels: Iterable[str]) -> None:
        if self.staged:
            return
        for rel in rels:
            (self.root / rel).unlink(missing_ok=True)

    def revision(self) -> str:
        return str(self.spec.revision)

    def commit(self) -> None:
        if self._commit is not None:
            self._commit()


def open_dataset(spec: DatasetSpec, *, work: str | os.PathLike[str] | None = None,
                 hub: Any = None, staged: bool = False,
                 commit: Callable[[], None] | None = None,
                 local_root: str | os.PathLike[str] | None = None) -> DatasetFiles:
    """``local_root``: where a local dataset is (default ``spec.local_path``; the CLI resolves a
    relative path against the config file's directory). ``work``: the Hub mirror directory."""
    if spec.repo is None:
        return LocalFiles(spec, local_root)
    if work is None:
        raise ValueError("a Hub dataset needs a work (or stage) directory")
    return HubFiles(spec, work, hub=hub, staged=staged, commit=commit)


# ---------------------------------------------------------------------------------------------
# Planning


@dataclass(frozen=True)
class WorkUnit:
    unit_id: str
    files: tuple[str, ...]  # dataset-relative source files the unit reads
    episodes: tuple[dict, ...]  # episode_to_dict() rows

    def to_dict(self) -> dict:
        return {"unit_id": self.unit_id, "files": list(self.files),
                "episodes": list(self.episodes)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> WorkUnit:
        return cls(d["unit_id"], tuple(d["files"]), tuple(d["episodes"]))


def source_files(ep: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({rel for rel, _s, _e in ep["cameras"].values()}))


def plan_units(episodes: Sequence[Mapping[str, Any]], *,
               max_episodes: int = MAX_EPISODES_PER_UNIT) -> list[WorkUnit]:
    """Group episodes by source files (one group = one demux pass per file), then pack whole
    groups next-fit, in group-key order, into units of at most ``max_episodes`` episodes (a
    larger group is a unit of its own)."""
    if max_episodes < 1:
        raise ValueError("max_episodes must be >= 1")
    groups: dict[tuple[str, ...], list[dict]] = {}
    for ep in episodes:
        groups.setdefault(source_files(ep), []).append(dict(ep))
    units: list[WorkUnit] = []
    cur: list[dict] = []
    cur_files: list[str] = []

    def close() -> None:
        nonlocal cur, cur_files
        if cur:
            units.append(WorkUnit(f"unit-{len(units):05d}", tuple(dict.fromkeys(cur_files)),
                                  tuple(cur)))
        cur, cur_files = [], []

    for key in sorted(groups):
        g = groups[key]
        if cur and len(cur) + len(g) > max_episodes:
            close()
        cur.extend(g)
        cur_files.extend(key)
    close()
    return units


def plan_index(job: IndexJob, files: DatasetFiles, *, limit: int | None = None,
               max_episodes: int = MAX_EPISODES_PER_UNIT) -> tuple[IndexJob, list[WorkUnit]]:
    """List the episodes (first ``limit``), resolve the revision and plan the units."""
    eps = files.list_episodes()
    if limit is not None:
        eps = eps[:limit]
    uids = [episode_uid(job.dataset.name, e["episode_id"]) for e in eps]
    dupes = sorted(u for u, n in Counter(uids).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate episode uids {dupes[:5]}")
    job = replace(job, revision=files.revision())
    return job, plan_units(eps, max_episodes=max_episodes)


# ---------------------------------------------------------------------------------------------
# Receipts


def _frac(x: Fraction | None) -> str | None:
    return None if x is None else f"{x.numerator}/{x.denominator}"


def _unfrac(s: str | None) -> Fraction | None:
    return None if s is None else Fraction(s)


def stream_to_dict(s: StreamIndex, root: Path | None = None) -> dict:
    """Every ``StreamIndex`` field, exactly; ``pts_us`` as integers over one denominator."""
    pts = [Fraction(p) for p in s.pts_us]
    den = math.lcm(*(p.denominator for p in pts)) if pts else 1
    uri = s.source.uri
    if root is not None:
        uri = Path(uri).relative_to(root).as_posix()
    return {
        "source": {"uri": uri, "start_s": s.source.start_s, "end_s": s.source.end_s},
        "rate": _frac(Fraction(s.rate)) if s.rate is not None else None,
        "pts_us_den": den, "pts_us_num": [int(p * den) for p in pts],
        "duration_us": s.duration_us, "size": list(s.size), "cfr": s.cfr, "codec": s.codec,
        "pix_fmt": s.pix_fmt, "starts_on_keyframe": s.starts_on_keyframe,
        "source_timestamps_s": list(s.source_timestamps_s), "time_base": _frac(s.time_base),
        "byte_size": s.byte_size, "video_streams": s.video_streams, "missing_pts": s.missing_pts,
        "colorspace": s.colorspace, "color_range": s.color_range,
        "source_pts": list(s.source_pts), "ineligible": list(s.ineligible),
    }


def stream_from_dict(d: Mapping[str, Any], root: str | os.PathLike[str] | None = None
                     ) -> StreamIndex:
    src = d["source"]
    uri = str(Path(root) / src["uri"]) if root is not None else src["uri"]
    den = d["pts_us_den"]
    return StreamIndex(
        source=SourceRef(uri, src["start_s"], src["end_s"]),
        rate=_unfrac(d["rate"]), pts_us=tuple(Fraction(n, den) for n in d["pts_us_num"]),
        duration_us=d["duration_us"], size=tuple(d["size"]), cfr=d["cfr"], codec=d["codec"],
        pix_fmt=d["pix_fmt"], starts_on_keyframe=d["starts_on_keyframe"],
        source_timestamps_s=tuple(d["source_timestamps_s"]), time_base=_unfrac(d["time_base"]),
        byte_size=d["byte_size"], video_streams=d["video_streams"],
        missing_pts=d["missing_pts"], colorspace=d["colorspace"], color_range=d["color_range"],
        source_pts=tuple(d["source_pts"]), ineligible=tuple(d["ineligible"]),
    )


def source_error_kind(message: str) -> str:
    for needle, kind in (("video file missing", "missing_file"), ("wrong file", "wrong_file"),
                         ("frames, the metadata says", "length_mismatch"),
                         ("no video packets", "no_video"), ("cannot read", "unreadable")):
        if needle in message:
            return kind
    return "other"


def index_one(job: IndexJob, ep: Mapping[str, Any], files: DatasetFiles, scans: Any) -> dict:
    """Verify + index one episode; return its receipt (not written)."""
    from flora_deploy.media.index import MediaError, index_episode
    from flora_deploy.media.index import envelope_checks as checks
    from flora_deploy.media.sources import SourceError, verify_episode
    from flora_deploy.validate.golden import stream_receipt

    uid = episode_uid(job.dataset.name, ep["episode_id"])
    profile, yam32 = job.profile
    rec: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "index_version": INDEX_CODE_VERSION,
        "key": job.resume_key(ep), "uid": uid, "episode_id": ep["episode_id"],
        "dataset": {"name": job.dataset.name, "format": job.dataset.format,
                    "repo": job.dataset.repo, "revision": job.revision,
                    "subdir": job.dataset.subdir},
        "pool": job.pool, "profile": profile, "yam32": yam32, "decode": job.decode,
        "instruction": job.instruction, "task": ep.get("task"),
        # the instruction stages use (C2): the fixed text, the dataset task, or "" for none
        "instruction_text": job.instruction_for(ep.get("task")), "episode": dict(ep),
        "source_files": list(source_files(ep)), "indexed_utc": utc_now(),
    }
    root = files.root
    try:
        for rel in source_files(ep):
            files.ensure(rel)
        sep = episode_from_dict(ep, root)
        verify_episode(sep, scans=scans)
        rec["verify"] = {"ok": True}
        ix = index_episode(sep.episode_id, job.dataset.name, sep.cameras, task=sep.task,
                           profile=profile, yam32=yam32, scans=scans, decode=job.decode)
        instruction = job.instruction_for(sep.task)
        if instruction != (sep.task or ""):
            ix = replace(ix, envelope=checks(ix.streams, profile=profile, yam32=yam32,
                                             instruction=instruction))
    except (SourceError, MediaError) as e:
        msg = f"{type(e).__name__}: {e}"
        rec.update(status="source_error", error=msg, error_kind=source_error_kind(str(e)),
                   verify={"ok": False, "error": str(e)}, envelope=None,
                   oracle_eligible=False, ineligibility=[], streams={}, receipts={},
                   episode_ref=None, duration_s=None, frames=None, cameras=len(ep["cameras"]))
        return rec
    from flora_deploy.jobs.manifest import episode_duration_s

    paths = JobPaths(job.job_id)
    duration = episode_duration_s(ix)
    ref = EpisodeRef(uid=uid, dataset=ix.dataset, episode_id=ix.episode_id, pool=job.pool,
                     duration_s=duration, oracle_eligible=ix.oracle_eligible,
                     envelope_ok=ix.envelope.ok, index_ref=paths.index(uid))
    main = ix.streams.get("main")
    rec.update(
        status="indexed", error=None, error_kind=None,
        envelope={"ok": ix.envelope.ok, "reasons": list(ix.envelope.reasons)},
        oracle_eligible=ix.oracle_eligible, ineligibility=list(ix.ineligibility),
        streams={r: stream_to_dict(s, root) for r, s in sorted(ix.streams.items())},
        receipts={r: {**stream_receipt(s), "uri": Path(s.source.uri).relative_to(root)
                      .as_posix()} for r, s in sorted(ix.streams.items())},
        episode_ref=asdict(ref), duration_s=duration,
        frames=len(main.pts_us) if main else None, cameras=len(ix.streams),
    )
    return rec


def receipt_bytes(rec: Mapping[str, Any]) -> bytes:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_receipt(store: Store, job_id: str, uid: str) -> dict | None:
    path = JobPaths(job_id).index(uid)
    try:
        if not store.exists(path):
            return None
        return json.loads(store.get_bytes(path))
    except (OSError, ValueError):
        return None


def episode_index_from_receipt(rec: Mapping[str, Any], *,
                               root: str | os.PathLike[str] | None = None) -> EpisodeIndex:
    if rec.get("status") != "indexed":
        raise ValueError(f"{rec.get('uid')}: not indexed ({rec.get('error')})")
    env = rec["envelope"]
    return EpisodeIndex(
        episode_id=rec["episode_id"], dataset=rec["dataset"]["name"],
        streams={r: stream_from_dict(s, root) for r, s in rec["streams"].items()},
        task=rec.get("task"), envelope=EnvelopeVerdict(env["ok"], tuple(env["reasons"])))


def episode_ref_from_receipt(rec: Mapping[str, Any]) -> EpisodeRef | None:
    return EpisodeRef(**rec["episode_ref"]) if rec.get("episode_ref") else None


def planned(row: Mapping[str, Any], policy: str) -> bool:
    """Whether an episode enters the manifest under the envelope policy (D13)."""
    if row.get("status") != "indexed":
        return False
    return policy != "enforce" or bool(row.get("envelope_ok"))


# ---------------------------------------------------------------------------------------------
# Running units


def summary_row(rec: Mapping[str, Any], *, resumed: bool = False) -> dict:
    """The small part of a receipt the summary needs (what units return to the coordinator)."""
    env = rec.get("envelope") or {}
    return {
        "uid": rec["uid"], "episode_id": rec["episode_id"], "status": rec["status"],
        "error": rec.get("error"), "error_kind": rec.get("error_kind"), "pool": rec["pool"],
        "cameras": rec.get("cameras"), "duration_s": rec.get("duration_s"),
        "envelope_ok": env.get("ok"), "envelope_reasons": list(env.get("reasons") or ()),
        "oracle_eligible": bool(rec.get("oracle_eligible")),
        "ineligibility": list(rec.get("ineligibility") or ()),
        "codecs": [s["codec"] for _r, s in sorted((rec.get("receipts") or {}).items())],
        "resumed": resumed,
    }


def index_unit(job: IndexJob, unit: WorkUnit, files: DatasetFiles, store: Store, *,
               force: bool = False, log: Log = print) -> list[dict]:
    """Index one unit's episodes (skipping done ones); returns one summary row per episode.

    Episodes that hit an unexpected error (network, disk) get a ``failed`` row and no receipt.
    """
    from flora_deploy.media.index import ScanCache

    paths = JobPaths(job.job_id)
    rows: list[dict] = []
    todo: list[dict] = []
    for ep in unit.episodes:
        uid = episode_uid(job.dataset.name, ep["episode_id"])
        rec = None if force else load_receipt(store, job.job_id, uid)
        if rec is not None and rec.get("key") == job.resume_key(ep):
            rows.append(summary_row(rec, resumed=True))
        else:
            todo.append(ep)
    if not todo:
        return rows
    scans = ScanCache(max_bytes=512 * 1024 * 1024)
    try:
        for ep in todo:
            uid = episode_uid(job.dataset.name, ep["episode_id"])
            t0 = time.monotonic()
            try:
                rec = index_one(job, ep, files, scans)
                store.put_bytes(paths.index(uid), receipt_bytes(rec))
            except Exception as e:  # noqa: BLE001 - retried by a rerun (no receipt written)
                rows.append({"uid": uid, "episode_id": ep["episode_id"], "status": "failed",
                             "error": f"{type(e).__name__}: {e}", "pool": job.pool,
                             "resumed": False})
                log(f"{uid}: FAILED {type(e).__name__}: {e}")
                continue
            rows.append(summary_row(rec))
            log(f"{uid}: {rec['status']} envelope_ok={(rec.get('envelope') or {}).get('ok')} "
                f"eligible={rec['oracle_eligible']} ({time.monotonic() - t0:.1f}s)")
    finally:
        files.commit()
        files.release(unit.files)
    commit = getattr(store, "commit", None)
    if commit is not None:
        commit()
    return rows


# ---------------------------------------------------------------------------------------------
# Summary


def summary_path(job_id: str) -> str:
    return f"{JobPaths(job_id).root}/index/summary.json"


def _dist(xs: list[float]) -> dict:
    return {"episodes": len(xs), "video_s": round(sum(xs), 6), "min_s": round(min(xs), 6),
            "max_s": round(max(xs), 6), "mean_s": round(sum(xs) / len(xs), 6)}


def build_summary(job: IndexJob, rows: Iterable[Mapping[str, Any]], *,
                  units: int | None = None, run: Mapping[str, Any] | None = None) -> dict:
    rows = sorted(rows, key=lambda r: r["uid"])
    status = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "indexed"]
    plan = [r for r in rows if planned(r, job.envelope)]
    by_group: dict[tuple[str, int], list[float]] = {}
    for r in ok:
        by_group.setdefault((r["pool"], int(r["cameras"])), []).append(float(r["duration_s"]))
    planned_groups: dict[tuple[str, int], list[float]] = {}
    for r in plan:
        planned_groups.setdefault((r["pool"], int(r["cameras"])), []).append(
            float(r["duration_s"]))
    out = {
        "schema": SUMMARY_SCHEMA, "job_id": job.job_id, "index_version": INDEX_CODE_VERSION,
        "dataset": {"name": job.dataset.name, "format": job.dataset.format,
                    "repo": job.dataset.repo, "revision": job.revision,
                    "subdir": job.dataset.subdir},
        "pool": job.pool, "envelope_policy": job.envelope, "decode": job.decode,
        "counts": {
            "episodes": len(rows), "indexed": status["indexed"],
            "source_error": status["source_error"], "failed": status["failed"],
            "envelope_ok": sum(1 for r in ok if r["envelope_ok"]),
            "out_of_envelope": sum(1 for r in ok if not r["envelope_ok"]),
            "oracle_eligible": sum(1 for r in ok if r["oracle_eligible"]),
            "planned": len(plan),
            "planned_out_of_envelope": sum(1 for r in plan if not r["envelope_ok"]),
            "resumed": sum(1 for r in rows if r.get("resumed")),
        },
        # ``flora plan --index`` (planner.options.WorkloadSummary.from_dict): planned episodes
        "groups": [{"pool": p, "cameras": c, "video_s": round(sum(v), 6), "episodes": len(v)}
                   for (p, c), v in sorted(planned_groups.items())],
        "durations": {f"{p}/{c}cam": _dist(v) for (p, c), v in sorted(by_group.items())},
        "video_s": round(sum(float(r["duration_s"]) for r in ok), 6),
        "envelope_reasons": dict(sorted(Counter(
            x for r in ok for x in r["envelope_reasons"]).items())),
        "ineligibility": dict(sorted(Counter(x for r in ok for x in r["ineligibility"]).items())),
        "source_errors": dict(sorted(Counter(
            r.get("error_kind") or "other" for r in rows if r["status"] == "source_error"
        ).items())),
        "codecs": dict(sorted(Counter(c for r in ok for c in r["codecs"]).items())),
        "failed": [{"uid": r["uid"], "error": r.get("error")} for r in rows
                   if r["status"] == "failed"],
        "created_utc": utc_now(),
    }
    if units is not None:
        out["units"] = units
    if run:
        out["run"] = dict(run)
    return out


def finish_index(job: IndexJob, rows: Iterable[Mapping[str, Any]], store: Store, *,
                 units: int | None = None, run: Mapping[str, Any] | None = None) -> dict:
    summary = build_summary(job, rows, units=units, run=run)
    store.put_bytes(summary_path(job.job_id),
                    json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n")
    commit = getattr(store, "commit", None)
    if commit is not None:
        commit()
    return summary


def run_index(job: IndexJob, store: Store, files: DatasetFiles, *, limit: int | None = None,
              max_episodes: int = MAX_EPISODES_PER_UNIT, force: bool = False,
              log: Log = print) -> dict:
    """In-process index job (``flora index --local``): plan, index every unit, summarize."""
    t0 = time.monotonic()
    job, units = plan_index(job, files, limit=limit, max_episodes=max_episodes)
    rows: list[dict] = []
    for unit in units:
        rows += index_unit(job, unit, files, store, force=force, log=log)
    return finish_index(job, rows, store, units=len(units),
                        run={"mode": "local", "wall_s": round(time.monotonic() - t0, 3)})


# ---------------------------------------------------------------------------------------------
# Reading the index back


def load_receipts(store: Store, job_id: str, uids: Iterable[str] | None = None) -> list[dict]:
    """Receipts of ``uids`` (default: every receipt the store lists under ``index/``)."""
    if uids is None:
        root = f"{JobPaths(job_id).root}/index/"
        found = [p.rsplit("/", 1)[1][:-5] for p in store.list(root)
                 if p.endswith(".json") and p != summary_path(job_id)]
        uids = sorted(found)
    out = []
    for uid in uids:
        rec = load_receipt(store, job_id, uid)
        if rec is not None:
            out.append(rec)
    return out


@dataclass
class LoadedIndex:
    indexes: list[EpisodeIndex] = field(default_factory=list)
    pools: dict[str, str] = field(default_factory=dict)  # uid -> pool
    refs: list[EpisodeRef] = field(default_factory=list)


def load_indexes(store: Store, job_id: str, *, policy: str = "warn",
                 root: str | os.PathLike[str] | None = None) -> LoadedIndex:
    """Planned episodes as ``EpisodeIndex`` + uid -> pool, for ``manifest.build_manifest``
    (``build_manifest(out.indexes, settings, pool=out.pools)``)."""
    out = LoadedIndex()
    for rec in load_receipts(store, job_id):
        row = summary_row(rec)
        if not planned(row, policy):
            continue
        out.indexes.append(episode_index_from_receipt(rec, root=root))
        out.pools[rec["uid"]] = rec["pool"]
        ref = episode_ref_from_receipt(rec)
        if ref is not None:
            out.refs.append(ref)
    return out
