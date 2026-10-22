"""Golden-set staging (PLAN-v3.md §9, W3-C): selection data -> per-episode oracle inputs + manifest.

The selection is data (``golden_selection_v1.json``, the human-only certification set
``golden_selection_v2.json`` and the yam (3-view) one ``golden_selection_v3_yam.json``, next to
this module; :data:`SELECTIONS`). A selection's
``version`` decides its prefix: ``golden/v2`` stages under ``<root>/golden/v2/`` and never
touches v1 (``golden/v1`` below stands for the selection's own prefix). Every episode names
its dataset (repo + pinned revision + format + license), the episode, the camera key -> role
map (or, for reward episodes, the ordered views), the pool and why it was chosen.
:func:`validate_selection` checks it without any network access.

Staging (:func:`stage_all`) is Modal-agnostic; ``modal_app/golden_app.py`` runs it on ``img_cpu``
against the ``flora-golden`` Volume. Per episode it:

1. downloads only the files that episode needs from the Hub at the pinned revision (LeRobot meta,
   the camera files, annotation files), sequentially (HF rate limits);
2. resolves the episode with ``media.sources`` (LeRobot v3/v2.1 readers; plain files otherwise)
   and checks every camera range against the real sample table (``verify_episode``);
3. indexes it (``index_episode(decode=True)``: probe-equivalent receipts, envelope and oracle
   eligibility) - reward episodes get the per-file stream checks per view;
4. keeps the oracle inputs: **stream-copy per-episode cuts** (``media.cuts``) for LeRobot v3
   concatenated files, the source files themselves for per-episode files (v2.1, plain MP4s,
   WGO-Bench's embedded MP4s); re-indexes every kept file and checks it against the source-range
   index (``cut_check``);
5. writes ``<root>/golden/v1/<pool>/<dataset>/<slug>/{<role>.mp4|view_<i>.mp4, record.json}``.

``record.json`` is written last, so an episode with a record is complete; reruns skip it unless
the selection entry changed (``selection_digest``) or ``force``. Finally
``<root>/golden/v1/manifest.json`` is rebuilt from every record (:func:`build_manifest`).
Videos and NC-licensed derivatives stay on the private Volume; only the numbers-only summary
(:func:`summary_markdown`) is committed (``docs/golden-manifest-v1.md``,
``docs/golden-manifest-v2.md``, ``docs/golden-manifest-v3-yam.md``; v2 and v3-yam are staged
and verified by ``scripts/golden_stage.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

SELECTION_PATH = Path(__file__).with_name("golden_selection_v1.json")
GOLDEN_VERSION = "golden/v1"
GOLDEN_PREFIX = PurePosixPath("golden") / "v1"
# Every version a selection may name. Each stages under its own ``golden/<v>/`` prefix
# (:func:`prefix_for`), so v2 never rewrites v1's files or manifest.
SELECTIONS = {"golden/v1": SELECTION_PATH,
              "golden/v2": Path(__file__).with_name("golden_selection_v2.json"),
              "golden/v3-yam": Path(__file__).with_name("golden_selection_v3_yam.json")}
GOLDEN_VERSIONS = tuple(SELECTIONS)
MANIFEST_NAME = "manifest.json"
RECORD_NAME = "record.json"

TEMPSEG_POOLS = ("human", "single_view_robot", "yam", "yam32", "dex", "tianji")
POOLS = (*TEMPSEG_POOLS, "reward")
# The P0 budget (W3-C): about 30 TempSeg/Captioner episodes and about 12 reward episodes.
POOL_TARGETS = {"human": 7, "single_view_robot": 6, "yam": 8, "dex": 4, "tianji": 2,
                "yam32": 2, "reward": 12}
FORMATS = ("lerobot_v3", "lerobot_v21", "mp4_files", "wgo_parquet")
ANNOTATION_KINDS = ("manirewardbench", "roboarena", "roboreward")
PROFILES = ("human", "yam", "dex", "tianji")
ROLES = ("main", "left_wrist", "right_wrist")
REWARD_MODES = ("terminal", "rubric", "dense")
MAX_EPISODE_S = 120.0
_SHA = re.compile(r"^[0-9a-f]{40}\Z")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
CUT_CHECK_FIELDS = ("rate", "frames", "duration_us", "size", "cfr", "codec", "pix_fmt",
                    "pts_us_sha256")


class SelectionError(ValueError):
    """The selection file is malformed or breaks a golden-set rule."""


class OutcomeMismatch(ValueError):
    """A label in the source differs from the one the selection recorded."""


# ---------------------------------------------------------------------------------------------
# Selection


def load_selection(path: str | Path = SELECTION_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prefix_for(version: str) -> PurePosixPath:
    """``golden/v2`` -> ``PurePosixPath("golden/v2")`` (a known version only)."""
    if version not in GOLDEN_VERSIONS:
        raise SelectionError(f"unknown golden version {version!r}; one of {GOLDEN_VERSIONS}")
    return PurePosixPath(version)


def entry_id(entry: Mapping[str, Any]) -> str:
    return f"{entry['pool']}/{entry['dataset']}/{entry['slug']}"


def episode_relpath(entry: Mapping[str, Any], prefix: PurePosixPath = GOLDEN_PREFIX
                    ) -> PurePosixPath:
    """Volume-relative directory of one episode: ``golden/v1/<pool>/<dataset>/<slug>``
    (``prefix``: the selection's version, :func:`prefix_for`)."""
    return prefix / entry["pool"] / entry["dataset"] / entry["slug"]


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def entry_digest(entry: Mapping[str, Any], dataset: Mapping[str, Any]) -> str:
    """Changes whenever anything that decides the staged bytes or labels changes."""
    return hashlib.sha256(_canonical({"entry": entry, "dataset": dataset})).hexdigest()


def selection_sha256(selection: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(selection)).hexdigest()


def _dataset_problems(key: str, ds: Mapping[str, Any]) -> list[str]:
    p = []
    where = f"dataset {key!r}"
    if not _SLUG.match(key):
        p.append(f"{where}: key must match {_SLUG.pattern}")
    if not isinstance(ds.get("repo"), str) or ds["repo"].count("/") != 1:
        p.append(f"{where}: repo must be 'owner/name'")
    if not _SHA.match(str(ds.get("revision", ""))):
        p.append(f"{where}: revision must be a full 40-hex commit sha")
    if ds.get("format") not in FORMATS:
        p.append(f"{where}: format must be one of {FORMATS}")
    if not str(ds.get("license", "")).strip():
        p.append(f"{where}: license is required")
    if not str(ds.get("task_source", "")).strip():
        p.append(f"{where}: task_source is required")
    if ds.get("format") == "wgo_parquet" and not ds.get("file"):
        p.append(f"{where}: wgo_parquet needs 'file'")
    if ds.get("format") == "mp4_files" and ds.get("annotations") not in ANNOTATION_KINDS:
        p.append(f"{where}: mp4_files needs annotations in {ANNOTATION_KINDS}")
    if ds.get("annotations") in ("manirewardbench", "roboreward") and not ds.get("annotation_file"):
        p.append(f"{where}: annotations {ds['annotations']!r} needs annotation_file")
    sub = ds.get("subdir")
    if sub is not None and (not isinstance(sub, str) or sub.startswith("/") or ".." in sub):
        p.append(f"{where}: subdir must be a relative path inside the repo")
    return p


def _entry_problems(e: Mapping[str, Any], datasets: Mapping[str, Any]) -> list[str]:
    p = []
    where = f"episode {e.get('pool')}/{e.get('dataset')}/{e.get('slug')}"
    pool = e.get("pool")
    if pool not in POOLS:
        return [f"{where}: pool must be one of {POOLS}"]
    ds = datasets.get(e.get("dataset"))
    if ds is None:
        return [f"{where}: unknown dataset {e.get('dataset')!r}"]
    if not _SLUG.match(str(e.get("slug", ""))):
        p.append(f"{where}: slug must match {_SLUG.pattern}")
    if not str(e.get("why", "")).strip():
        p.append(f"{where}: 'why' is required")
    if "episode" not in e:
        p.append(f"{where}: 'episode' is required")
    fmt = ds.get("format")
    if fmt in ("lerobot_v3", "lerobot_v21") and not isinstance(e.get("episode"), int):
        p.append(f"{where}: LeRobot episodes are integer episode indices")
    if fmt == "wgo_parquet" and not isinstance(e.get("row_group"), int):
        p.append(f"{where}: WGO-Bench episodes need an integer row_group hint")
    exp = e.get("expected_s")
    if exp is not None and not (0 < float(exp) <= MAX_EPISODE_S):
        p.append(f"{where}: expected_s must be in (0, {MAX_EPISODE_S}]")
    for h in e.get("hard_case", []):
        if not isinstance(h, str) or not h:
            p.append(f"{where}: hard_case entries are non-empty strings")
    if pool == "reward":
        views = e.get("views")
        if not isinstance(views, list) or not 1 <= len(views) <= 3:
            p.append(f"{where}: reward episodes need 1-3 views (collage order L/C/R)")
        elif len(set(views)) != len(views):
            p.append(f"{where}: views repeat")
        if "cameras" in e:
            p.append(f"{where}: reward episodes use 'views', not 'cameras'")
        modes = e.get("modes") or []
        if not modes or any(m not in REWARD_MODES for m in modes):
            p.append(f"{where}: modes must be a non-empty subset of {REWARD_MODES}")
        if not isinstance(e.get("outcome"), dict) or not e["outcome"]:
            p.append(f"{where}: reward episodes need a known outcome")
        if fmt != "mp4_files":
            p.append(f"{where}: reward episodes come from per-episode files (mp4_files)")
        return p
    cams = e.get("cameras")
    if not isinstance(cams, dict) or set(cams) not in ({"main"}, set(ROLES)):
        p.append(f"{where}: cameras must map exactly 'main' or main/left_wrist/right_wrist")
    elif len(set(cams.values())) != len(cams):
        p.append(f"{where}: one camera key mapped to two roles")
    elif set(cams.values()) & set(e.get("dropped_cameras", [])):
        p.append(f"{where}: a camera is both used and dropped")
    profile = e.get("profile")
    if profile not in PROFILES:
        p.append(f"{where}: profile must be one of {PROFILES}")
    n = len(cams) if isinstance(cams, dict) else 0
    if pool == "human" and (profile != "human" or n != 1):
        p.append(f"{where}: human episodes are single view with profile 'human'")
    if pool == "single_view_robot" and (profile in (None, "human") or n != 1):
        p.append(f"{where}: single-view robot episodes are one view with a robot profile")
    if pool in ("yam", "yam32", "dex", "tianji") and n != 3:
        p.append(f"{where}: pool {pool!r} needs three views")
    if pool in ("dex", "tianji") and profile != pool:
        p.append(f"{where}: pool {pool!r} uses profile {pool!r}")
    if pool in ("yam", "yam32") and profile != "yam":
        p.append(f"{where}: pool {pool!r} uses profile 'yam'")
    if bool(e.get("yam32")) != (pool == "yam32"):
        p.append(f"{where}: yam32=true exactly for the yam32 pool")
    return p


def validate_selection(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise ``SelectionError`` listing every problem; return the selection when it is valid."""
    problems: list[str] = []
    if selection.get("version") not in GOLDEN_VERSIONS:
        problems.append(f"version must be one of {GOLDEN_VERSIONS}")
    datasets = selection.get("datasets") or {}
    episodes = selection.get("episodes") or []
    if not datasets or not episodes:
        problems.append("datasets and episodes must be non-empty")
    for key, ds in datasets.items():
        problems += _dataset_problems(key, ds)
    seen: Counter[str] = Counter()
    for e in episodes:
        problems += _entry_problems(e, datasets)
        if {"pool", "dataset", "slug"} <= set(e):
            seen[entry_id(e)] += 1
    problems += [f"duplicate episode id {k}" for k, n in seen.items() if n > 1]
    # The same source episode selected twice (under different slugs or pools).
    src: Counter[tuple] = Counter(
        (e.get("dataset"), json.dumps(e.get("episode"))) for e in episodes
    )
    problems += [f"source episode {k[0]}:{k[1]} selected more than once"
                 for k, n in src.items() if n > 1]
    ids = {entry_id(e) for e in episodes if {"pool", "dataset", "slug"} <= set(e)}
    noise = selection.get("noise_subset")
    if noise is not None:  # the episodes a certified run's noise job repeats (golden/v2)
        if not isinstance(noise, list) or len(set(noise)) != len(noise):
            problems.append("noise_subset must be a list of distinct episode ids")
        else:
            problems += [f"noise_subset: {i} is not a selected episode" for i in noise
                         if i not in ids]
    used = {e.get("dataset") for e in episodes}
    problems += [f"dataset {k!r} is not used by any episode" for k in datasets if k not in used]
    if problems:
        raise SelectionError("invalid golden selection:\n  " + "\n  ".join(problems))
    return selection


def pool_counts(selection: Mapping[str, Any]) -> dict[str, int]:
    c = Counter(e["pool"] for e in selection["episodes"])
    return {p: c.get(p, 0) for p in POOLS}


# ---------------------------------------------------------------------------------------------
# Hub access (sequential; a fake in tests)


class Hub(Protocol):
    def download(self, repo: str, revision: str, filename: str, local_dir: Path) -> Path: ...

    def list_files(self, repo: str, revision: str, prefix: str) -> list[str]: ...


class HfHub:
    """``huggingface_hub`` at a pinned revision, one request at a time, with backoff on 429/5xx."""

    def __init__(self, *, retries: int = 5, backoff_s: float = 5.0, token: str | None = None):
        self.retries, self.backoff_s, self.token = retries, backoff_s, token
        self.requests = 0

    def _call(self, fn: Callable[[], Any]) -> Any:
        from huggingface_hub.errors import HfHubHTTPError

        for attempt in range(self.retries + 1):
            self.requests += 1
            try:
                return fn()
            except HfHubHTTPError as e:
                status = getattr(e.response, "status_code", None)
                if attempt == self.retries or status not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(self.backoff_s * 2**attempt)
        raise AssertionError("unreachable")

    def download(self, repo: str, revision: str, filename: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(self._call(lambda: hf_hub_download(
            repo, filename, repo_type="dataset", revision=revision, local_dir=local_dir,
            token=self.token,
        )))

    def list_files(self, repo: str, revision: str, prefix: str) -> list[str]:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile

        api = HfApi(token=self.token)
        entries = self._call(lambda: list(api.list_repo_tree(
            repo, path_in_repo=prefix or None, recursive=True, repo_type="dataset",
            revision=revision,
        )))
        return sorted(e.path for e in entries if isinstance(e, RepoFile))


# ---------------------------------------------------------------------------------------------
# Receipts


def _frac(x: Fraction | None) -> str | None:
    return None if x is None else f"{x.numerator}/{x.denominator}"


def pts_digest(pts_us: Iterable[Fraction]) -> str:
    return hashlib.sha256(";".join(_frac(Fraction(p)) or "" for p in pts_us).encode()).hexdigest()


def stream_receipt(s) -> dict:
    """Probe-equivalent summary of one ``StreamIndex`` (full PTS list replaced by its digest)."""
    return {
        "rate": _frac(s.rate) if s.rate else None,
        "fps": float(s.rate) if s.rate else None,
        "frames": len(s.pts_us),
        "duration_us": s.duration_us,
        "size": list(s.size),
        "cfr": s.cfr,
        "codec": s.codec,
        "pix_fmt": s.pix_fmt,
        "starts_on_keyframe": s.starts_on_keyframe,
        "ineligible": list(s.ineligible),
        "time_base": _frac(s.time_base),
        "byte_size": s.byte_size,
        "colorspace": s.colorspace,
        "color_range": s.color_range,
        "pts_us_sha256": pts_digest(s.pts_us),
        "source_range_s": [s.source.start_s, s.source.end_s],
        "first_source_pts": s.source_pts[0] if s.source_pts else None,
    }


def cut_check(expected: Mapping[str, Any], staged: Mapping[str, Any]) -> list[str]:
    """Fields where the staged file's index differs from the source-range index ([] = exact)."""
    return [k for k in CUT_CHECK_FIELDS if expected.get(k) != staged.get(k)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _same_file(a: Path, b: Path) -> bool:
    """``a`` and ``b`` both exist and are one file (symlinked paths, a resolved mount)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def copy_atomic(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.partial")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def write_json_atomic(obj: Any, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------------------------
# Resolving one selected episode to local files


@dataclass
class Resolved:
    """A selected episode with its files downloaded to the work directory."""

    cameras: dict[str, Any]  # role or "view_<i>" -> media SourceRef (local uri)
    keys: dict[str, str]  # same keys -> dataset camera key / repo path
    task: str | None
    whole_files: bool  # every camera file is the whole episode (no cut needed)
    source_files: list[str]  # repo paths downloaded for this episode
    length: int | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    episode_ref: Any = None  # media EpisodeRef (for verify_episode)


class DatasetMirror:
    """Partial local mirror of one dataset at its pinned revision (only what episodes need)."""

    def __init__(self, key: str, ds: Mapping[str, Any], hub: Hub, work: Path):
        self.key, self.ds, self.hub = key, ds, hub
        self.local = Path(work) / key
        self.sub = (ds.get("subdir") or "").strip("/")
        self.root = self.local / self.sub if self.sub else self.local
        self.fetched: list[str] = []
        self._cache: dict[str, Any] = {}

    def repo_path(self, rel: str) -> str:
        return f"{self.sub}/{rel}" if self.sub else rel

    def fetch(self, rel: str) -> Path:
        """Download ``rel`` (relative to the dataset root) unless it is already local. ``rel``
        comes from dataset metadata: it must stay under the mirror (``media.safepath``)."""
        from flora_deploy.media.safepath import contain

        dest = contain(self.root, rel)
        if not dest.is_file():
            got = self.hub.download(self.ds["repo"], self.ds["revision"], self.repo_path(rel),
                                    self.local)
            # hf_hub_download resolves local_dir, so behind a symlinked mount (Modal's
            # /vol/<name> -> /__modal/volumes/vo-...) it returns ``dest`` by its real path:
            # the same file, not a copy to make.
            if Path(got) != dest and not _same_file(Path(got), dest):
                # a hub that ignores local_dir layout
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(got, dest)
        self.fetched.append(self.repo_path(rel))
        return dest

    # LeRobot ------------------------------------------------------------------------------

    def lerobot_source(self, camera_map: Mapping[str, str]):
        from flora_deploy.media.sources import LeRobotV3Source, LeRobotV21Source

        cache_key = ("source", tuple(sorted(camera_map.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]
        self.fetch("meta/info.json")
        if self.ds["format"] == "lerobot_v3":
            prefix = self.repo_path("meta/episodes")
            for path in self.hub.list_files(self.ds["repo"], self.ds["revision"], prefix):
                if path.endswith(".parquet"):
                    local = self.fetch(path[len(self.sub) + 1:] if self.sub else path)
                    _normalize_episode_meta_name(local)
            src = LeRobotV3Source(self.root, camera_map=camera_map, name=self.key)
        else:
            self.fetch("meta/episodes.jsonl")
            src = LeRobotV21Source(self.root, camera_map=camera_map, name=self.key)
        episodes = {int(ep.extra["episode_index"]): ep for ep in src.list_episodes()}
        self._cache[cache_key] = (src, episodes)
        return self._cache[cache_key]

    def resolve_lerobot(self, entry: Mapping[str, Any]) -> Resolved:
        from flora_deploy.media.sources import SourceError

        camera_map = {key: role for role, key in entry["cameras"].items()}
        _, episodes = self.lerobot_source(camera_map)
        ep = episodes.get(int(entry["episode"]))
        if ep is None:
            raise SourceError(f"{self.key}: episode {entry['episode']} not in meta/episodes")
        files = []
        for ref in ep.cameras.values():
            rel = Path(ref.uri).relative_to(self.root).as_posix()
            self.fetch(rel)
            files.append(self.repo_path(rel))
        return Resolved(
            cameras=dict(ep.cameras), keys=dict(entry["cameras"]), task=ep.task,
            whole_files=self.ds["format"] == "lerobot_v21", source_files=sorted(set(files)),
            length=ep.extra.get("length"), episode_ref=ep,
        )

    # WGO-Bench parquet (MP4 bytes embedded per row) -------------------------------------------

    def resolve_wgo(self, entry: Mapping[str, Any]) -> Resolved:
        import pyarrow.parquet as pq

        from flora_deploy.media.sources import EpisodeRef, SourceError
        from flora_deploy.media.types import SourceRef

        parquet = self.fetch(self.ds["file"])
        pf = pq.ParquetFile(parquet)
        cols = [c for c in ("id", "instruction", "video", "segments") if c in pf.schema_arrow.names]
        rg = int(entry["row_group"])
        if not 0 <= rg < pf.num_row_groups:
            raise SourceError(f"{self.key}: row group {rg} out of range")
        rows = [r for r in pf.read_row_group(rg, columns=cols).to_pylist()
                if r["id"] == entry["episode"]]
        if len(rows) != 1:
            raise SourceError(f"{self.key}: row group {rg} does not hold id {entry['episode']!r}")
        row = rows[0]
        video = row["video"]
        data = video["bytes"] if isinstance(video, dict) else video
        path = self.local / "_rows" / f"{entry['slug']}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        task = (row.get("instruction") or "").strip() or None
        cams = {"main": SourceRef(str(path))}
        return Resolved(
            cameras=cams, keys={"main": f"{self.ds['file']}#id={entry['episode']}"}, task=task,
            whole_files=True, source_files=[self.ds["file"]],
            annotations={"segments": row.get("segments") or []},
            episode_ref=EpisodeRef(episode_id=entry["slug"], cameras=cams, task=task),
        )

    # Plain per-episode MP4 files with an annotation side file ---------------------------------

    def _json_file(self, rel: str) -> Any:
        if rel not in self._cache:
            path = self.fetch(rel)
            text = path.read_text(encoding="utf-8")
            self._cache[rel] = ([json.loads(x) for x in text.splitlines() if x.strip()]
                                if rel.endswith(".jsonl") else json.loads(text))
        return self._cache[rel]

    def resolve_files(self, entry: Mapping[str, Any]) -> Resolved:
        from flora_deploy.media.sources import EpisodeRef
        from flora_deploy.media.types import SourceRef

        reward = entry["pool"] == "reward"
        keyed = ({f"view_{i}": p for i, p in enumerate(entry["views"])} if reward
                 else dict(entry["cameras"]))
        cams = {k: SourceRef(str(self.fetch(p))) for k, p in keyed.items()}
        task, annotations, outcome = self._annotate(entry)
        return Resolved(
            cameras=cams, keys=keyed, task=task, whole_files=True,
            source_files=sorted(set(keyed.values())), annotations=annotations, outcome=outcome,
            episode_ref=EpisodeRef(episode_id=entry["slug"], cameras=cams, task=task),
        )

    def _annotate(self, entry: Mapping[str, Any]) -> tuple[str | None, dict, dict]:
        from flora_deploy.media.sources import SourceError

        kind = self.ds["annotations"]
        if kind == "manirewardbench":
            ann = self._json_file(self.ds["annotation_file"])
            rec = ann.get(entry["cameras"]["main"])
            if rec is None:
                raise SourceError(f"{self.key}: no annotation for {entry['cameras']['main']}")
            return rec.get("task"), {"subtasks": rec.get("subtasks", []),
                                     "mistakes": rec.get("mistakes", [])}, {}
        if kind == "roboreward":
            rows = self._json_file(self.ds["annotation_file"])
            hits = [r for r in rows if r.get("file_name") == entry["episode"]]
            if len(hits) != 1:
                raise SourceError(f"{self.key}: {len(hits)} metadata rows for {entry['episode']}")
            return hits[0].get("task"), {}, {"reward": hits[0].get("reward")}
        # roboarena: evaluation_sessions/<session>/metadata.yaml, policy label
        import yaml

        session, label = str(entry["episode"]).split("/")
        rel = f"evaluation_sessions/{session}/metadata.yaml"
        if rel not in self._cache:
            self._cache[rel] = yaml.safe_load(self.fetch(rel).read_text(encoding="utf-8"))
        meta = self._cache[rel]
        pol = (meta.get("policies") or {}).get(label)
        if pol is None:
            raise SourceError(f"{self.key}: session {session} has no policy {label!r}")
        outcome = {k: pol.get(k) for k in ("policy_name", "binary_success", "partial_success",
                                           "duration")}
        annotations = {"preference": meta.get("preference"),
                       "longform_feedback": meta.get("longform_feedback")}
        return meta.get("language_instruction"), annotations, outcome

    def resolve(self, entry: Mapping[str, Any]) -> Resolved:
        fmt = self.ds["format"]
        if fmt in ("lerobot_v3", "lerobot_v21"):
            return self.resolve_lerobot(entry)
        if fmt == "wgo_parquet":
            return self.resolve_wgo(entry)
        return self.resolve_files(entry)

    def cleanup(self) -> None:
        shutil.rmtree(self.local, ignore_errors=True)


def _normalize_episode_meta_name(path: Path) -> None:
    """Workaround: ``LeRobotV3Source`` globs ``meta/episodes/chunk-*/file-*.parquet``, but some
    v3 datasets (jnogga/droid_success) write ``file_000.parquet``. Rename inside the private
    work mirror only (a media/ fix is reported separately)."""
    m = re.fullmatch(r"file_(\d+)\.parquet", path.name)
    if m:
        os.replace(path, path.with_name(f"file-{m.group(1)}.parquet"))


def check_outcome(expected: Mapping[str, Any], found: Mapping[str, Any]) -> None:
    bad = {k: (v, found.get(k)) for k, v in expected.items() if found.get(k) != v}
    if bad:
        raise OutcomeMismatch(f"source labels differ from the selection: {bad}")


# ---------------------------------------------------------------------------------------------
# Staging one episode


def _reward_index(resolved: Resolved, scans) -> tuple[dict, list[str], bool]:
    """Per-view index + stream checks (the Rewarder reads whole per-episode files)."""
    from flora_deploy.media.index import get_scan, index_stream, stream_checks

    streams, reasons = {}, []
    for k, ref in resolved.cameras.items():
        scan = get_scan(scans, ref.uri)
        s = index_stream(ref, scan=scan)
        if s.starts_on_keyframe:
            s = index_stream(ref, scan=scan, decode=True)
        streams[k] = s
        reasons += [f"{k}:{r}" for r in stream_checks(s)]
    ss = list(streams.values())
    synced = all(s.rate == ss[0].rate and s.pts_us == ss[0].pts_us for s in ss)
    return streams, reasons, synced


def stage_episode(entry: Mapping[str, Any], mirror: DatasetMirror, root: Path, scans,
                  *, digest: str, prefix: PurePosixPath = GOLDEN_PREFIX) -> dict:
    """Stage one episode into ``root/<episode_relpath>``; return its record (also written)."""
    from flora_deploy.media.cuts import stream_copy_cut
    from flora_deploy.media.index import MediaError, get_scan, index_episode, index_stream
    from flora_deploy.media.sources import SourceError, verify_episode
    from flora_deploy.media.types import SourceRef

    ds = mirror.ds
    dest = Path(root) / episode_relpath(entry, prefix)
    reward = entry["pool"] == "reward"
    record: dict[str, Any] = {
        "id": entry_id(entry), "pool": entry["pool"], "dataset": entry["dataset"],
        "slug": entry["slug"], "episode": entry["episode"],
        "path": str(episode_relpath(entry, prefix)),
        "profile": entry.get("profile"), "yam32": bool(entry.get("yam32")),
        "modes": entry.get("modes"), "also_reward": entry.get("also_reward", []),
        "hard_case": entry.get("hard_case", []), "why": entry.get("why"),
        "dropped_cameras": entry.get("dropped_cameras", []),
        "expected_s": entry.get("expected_s"),
        "source": {k: ds.get(k) for k in ("repo", "revision", "format", "subdir", "license")},
        "task_source": ds.get("task_source"), "selection_digest": digest,
        "staged_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        resolved = mirror.resolve(entry)
        record.update(task=resolved.task, annotations=resolved.annotations,
                      outcome=resolved.outcome, source_files=resolved.source_files)
        if entry.get("outcome"):
            check_outcome(entry["outcome"], resolved.outcome)
        verify_episode(resolved.episode_ref, scans=scans,
                       check_length=resolved.length is not None)
        record["verify"] = {"ok": True}
        if reward:
            streams, reasons, synced = _reward_index(resolved, scans)
            envelope = {"ok": not reasons, "reasons": reasons, "checks": "stream_checks per view"}
            eligible = not reasons and all(s.starts_on_keyframe for s in streams.values())
            inel = [f"{k}:{x}" for k, s in sorted(streams.items()) for x in s.ineligible]
            record["views_synchronized"] = synced
        else:
            ix = index_episode(
                entry["slug"], entry["dataset"], resolved.cameras, task=resolved.task,
                profile=entry["profile"], yam32=bool(entry.get("yam32")), scans=scans,
                decode=True,
            )
            streams = ix.streams
            envelope = {"ok": ix.envelope.ok, "reasons": list(ix.envelope.reasons)}
            eligible, inel = ix.oracle_eligible, list(ix.ineligibility)
        receipts = {k: stream_receipt(s) for k, s in streams.items()}
        files, checks = {}, {}
        work_cuts = mirror.local / "_cuts" / entry["slug"]
        for k, s in sorted(streams.items()):
            out = dest / f"{k}.mp4"
            if resolved.whole_files:
                local, kind = Path(s.source.uri), "source_file"
            elif s.starts_on_keyframe:
                local = stream_copy_cut(s.source, work_cuts / f"{k}.mp4",
                                        scan=get_scan(scans, s.source.uri))
                kind = "stream_copy_cut"
            else:
                files[k] = {"file": None, "kind": None, "key": resolved.keys[k]}
                continue
            staged = stream_receipt(index_stream(SourceRef(str(local))))
            checks[k] = cut_check(receipts[k], staged) or "ok"
            copy_atomic(local, out)
            files[k] = {"file": out.name, "kind": kind, "key": resolved.keys[k],
                        "sha256": sha256_file(out), "bytes": out.stat().st_size}
        shutil.rmtree(work_cuts, ignore_errors=True)
        main = streams.get("main") or streams.get("view_0")
        record.update(
            status="staged", files=files, index=receipts, envelope=envelope,
            oracle_eligible=bool(eligible) and all(c == "ok" for c in checks.values()),
            ineligibility=inel, cut_check=checks,
            duration_s=main.duration_s if main else None,
            frames=len(main.pts_us) if main else None,
        )
    except (SourceError, MediaError, OutcomeMismatch) as e:
        record.update(status="source_error", error=f"{type(e).__name__}: {e}",
                      oracle_eligible=False, verify={"ok": False, "error": str(e)})
    write_json_atomic(record, dest / RECORD_NAME)
    return record


def load_record(root: Path, entry: Mapping[str, Any], prefix: PurePosixPath = GOLDEN_PREFIX
                ) -> dict | None:
    path = Path(root) / episode_relpath(entry, prefix) / RECORD_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# Whole selection


def stage_all(
    selection: Mapping[str, Any],
    *,
    hub: Hub,
    root: Path,
    work: Path,
    commit: Callable[[], None] = lambda: None,
    only: Iterable[str] = (),
    force: bool = False,
    deadline: float | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Stage every selected episode (resumable) and rebuild ``golden/<v>/manifest.json``
    (``<v>``: the selection's ``version``; v1 and v2 never share a file).

    Episodes are grouped by dataset so each dataset's files are fetched once and the work mirror
    is deleted afterwards. ``only``: episode ids or dataset keys. ``deadline``: a
    ``time.monotonic()`` value after which no new dataset is started (rerun to continue).
    Unexpected errors (network) are listed in the manifest's ``failed`` and leave no record, so a
    rerun retries them.
    """
    from flora_deploy.media.index import ScanCache

    validate_selection(selection)
    prefix = prefix_for(selection["version"])
    only = set(only)
    root, work = Path(root), Path(work)
    datasets = selection["datasets"]
    records: dict[str, dict] = {}
    failed: list[dict] = []
    by_ds: dict[str, list] = {}
    for e in selection["episodes"]:
        by_ds.setdefault(e["dataset"], []).append(e)
    for key, entries in by_ds.items():
        todo = []
        for e in entries:
            digest = entry_digest(e, datasets[key])
            rec = load_record(root, e, prefix)
            selected = not only or key in only or entry_id(e) in only
            if rec is not None and rec.get("selection_digest") == digest and not (
                force and selected
            ):
                records[entry_id(e)] = rec
            elif selected:
                todo.append((e, digest))
        if not todo:
            continue
        if deadline is not None and time.monotonic() > deadline:
            failed += [{"id": entry_id(e), "error": "deadline: not started"} for e, _ in todo]
            continue
        mirror = DatasetMirror(key, datasets[key], hub, work)
        scans = ScanCache()
        try:
            for e, digest in todo:
                t0 = time.monotonic()
                try:
                    rec = stage_episode(e, mirror, root, scans, digest=digest, prefix=prefix)
                    records[entry_id(e)] = rec
                    log(f"{entry_id(e)}: {rec['status']} eligible={rec['oracle_eligible']} "
                        f"({time.monotonic() - t0:.1f}s)")
                except Exception as ex:  # noqa: BLE001 - network etc.: retried by a rerun
                    failed.append({"id": entry_id(e), "error": f"{type(ex).__name__}: {ex}"})
                    log(f"{entry_id(e)}: FAILED {type(ex).__name__}: {ex}")
            _stage_extra_files(key, datasets[key], mirror, root, log, prefix)
        finally:
            mirror.cleanup()
            commit()
    manifest = build_manifest(selection, records.values(), failed)
    write_json_atomic(manifest, root / prefix / MANIFEST_NAME)
    commit()
    return manifest


def _stage_extra_files(key, ds, mirror: DatasetMirror, root: Path, log,
                       prefix: PurePosixPath = GOLDEN_PREFIX) -> None:
    for rel in ds.get("extra_files", []):
        dest = Path(root) / prefix / "_datasets" / key / rel
        if dest.is_file():
            continue
        try:
            copy_atomic(mirror.fetch(rel), dest)
        except Exception as ex:  # noqa: BLE001 - optional side files
            log(f"{key}: extra file {rel} not staged: {type(ex).__name__}: {ex}")


# ---------------------------------------------------------------------------------------------
# Manifest and summary


_MANIFEST_KEYS = (
    "id", "pool", "dataset", "slug", "episode", "path", "profile", "yam32", "modes",
    "also_reward", "hard_case", "why", "dropped_cameras", "source", "task", "task_source",
    "outcome", "status", "error", "verify", "files", "index", "envelope", "oracle_eligible",
    "ineligibility", "cut_check", "views_synchronized", "duration_s", "frames", "expected_s",
    "selection_digest",
)


def build_manifest(selection: Mapping[str, Any], records: Iterable[Mapping[str, Any]],
                   failed: Iterable[Mapping[str, Any]] = ()) -> dict:
    """The golden manifest: one compact record per staged episode, in selection order."""
    by_id = {r["id"]: r for r in records}
    order = [entry_id(e) for e in selection["episodes"]]
    episodes = [{k: by_id[i].get(k) for k in _MANIFEST_KEYS if k in by_id[i]}
                for i in order if i in by_id]
    for ep in episodes:  # annotation counts only; full annotations stay in record.json
        rec = by_id[ep["id"]]
        ann = rec.get("annotations") or {}
        ep["annotation_counts"] = {k: len(v) for k, v in ann.items() if isinstance(v, list)}
    counts = {}
    for pool in POOLS:
        sel = [i for i in order if i.split("/")[0] == pool]
        got = [by_id[i] for i in sel if i in by_id]
        counts[pool] = {
            "selected": len(sel),
            "staged": sum(r.get("status") == "staged" for r in got),
            "oracle_eligible": sum(bool(r.get("oracle_eligible")) for r in got),
        }
    failed = list(failed)
    missing = [i for i in order if i not in by_id and i not in {f["id"] for f in failed}]
    datasets = {k: {f: v.get(f) for f in ("repo", "revision", "format", "subdir", "license",
                                          "extra_files") if v.get(f) is not None}
                for k, v in selection["datasets"].items()}
    return {
        "version": selection.get("version", GOLDEN_VERSION),
        "selection_sha256": selection_sha256(selection),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "counts": counts,
        "complete": not failed and not missing,
        "datasets": datasets,
        "episodes": episodes,
        "failed": failed,
        "missing": missing,
    }


def _main_receipt(ep: Mapping[str, Any]) -> Mapping[str, Any] | None:
    idx = ep.get("index") or {}
    return idx.get("main") or idx.get("view_0")


def summary_markdown(manifest: Mapping[str, Any]) -> str:
    """Numbers-only public summary (no task text, captions, NVIDIA material or media)."""
    eps = list(manifest["episodes"])
    lines = [
        "| Pool | Selected | Staged | Oracle-eligible | Duration (s, total) | Duration range (s) |",
        "|---|---|---|---|---|---|",
    ]
    tot_sel = tot_staged = tot_ok = 0
    tot_dur = 0.0
    for pool in POOLS:
        c = manifest["counts"][pool]
        ds = [e["duration_s"] for e in eps if e["pool"] == pool and e.get("duration_s")]
        rng = f"{min(ds):.1f}-{max(ds):.1f}" if ds else "-"
        lines.append(f"| `{pool}` | {c['selected']} | {c['staged']} | {c['oracle_eligible']} | "
                     f"{sum(ds):.1f} | {rng} |")
        tot_sel, tot_staged, tot_ok = (tot_sel + c["selected"], tot_staged + c["staged"],
                                       tot_ok + c["oracle_eligible"])
        tot_dur += sum(ds)
    lines.append(f"| **total** | {tot_sel} | {tot_staged} | {tot_ok} | {tot_dur:.1f} | |")
    return "\n".join(lines) + "\n"


def profile_rows(manifest: Mapping[str, Any]) -> list[dict]:
    """One row per episode for the public summary: codec / size / fps / frames / eligibility."""
    rows = []
    for ep in manifest["episodes"]:
        r = _main_receipt(ep) or {}
        reasons = list((ep.get("envelope") or {}).get("reasons") or [])
        reasons += list(ep.get("ineligibility") or [])
        reasons += [f"{k}:cut_check:{','.join(v)}" for k, v in (ep.get("cut_check") or {}).items()
                    if v != "ok"]
        if ep.get("status") != "staged":
            reasons.append(str(ep.get("error", ep.get("status"))))
        rows.append({
            "id": ep["id"], "pool": ep["pool"], "views": len(ep.get("index") or {}),
            "codec": r.get("codec"), "size": "x".join(map(str, r.get("size") or [])),
            "fps": r.get("rate"), "frames": r.get("frames"), "duration_s": ep.get("duration_s"),
            "cfr": r.get("cfr"), "license": (ep.get("source") or {}).get("license"),
            "kind": sorted({f.get("kind") for f in (ep.get("files") or {}).values()
                            if f.get("kind")}),
            "oracle_eligible": ep.get("oracle_eligible"), "reasons": reasons,
        })
    return rows
