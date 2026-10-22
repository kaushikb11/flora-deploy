"""Hugging Face Hub side of ``flora export``: fetch a source dataset, write the dataset card,
push an export.

* :func:`fetch_source` downloads ``meta/`` and ``data/`` (and ``videos/`` when asked) of a Hub
  dataset at its pinned revision; nothing else.
* :func:`dataset_card` renders ``README.md`` for an export: it cites Flora, carries the NVIDIA
  noncommercial notice for the derived labels and records provenance.
* :func:`push_to_hub` uploads an export directory to a dataset repo, **private by default**. It
  refuses a repo that already exists and is public when a private push was asked, and tags the
  commit with the LeRobot ``codebase_version`` (LeRobot resolves Hub datasets by that tag).
  Before anything is sent, :func:`scan_export` reads every file to be uploaded and the push is
  refused (:class:`LocalPathLeak`) when one names an absolute local path, the home directory
  or the host name.
"""

from __future__ import annotations

import json
import re
import socket
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from flora_deploy.export.lerobot import (
    DENSE_COLUMN,
    LABEL_LICENSE,
    LOCAL_PATH_RE,
    REPORT_NAME,
    SUBTASK_COLUMN,
    ExportError,
)

UPLOAD_IGNORE = (".cache", ".git")  # top-level directories never uploaded (upload_folder)
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".tsv",
                           ".cff", ".toml", ".cfg", ".ini", ".log", ".html", ""})
SCAN_REPORT_MAX = 10  # findings listed in the error


class LocalPathLeak(ExportError):
    """The export holds a local path (or the home directory / host name); nothing was pushed."""


def _needles() -> list[tuple[str, str]]:
    """Strings of this machine that must not leave it: (needle, what it is)."""
    out: list[tuple[str, str]] = []
    homes = {str(Path.home())}
    try:
        homes.add(str(Path.home().resolve()))
    except OSError:
        pass
    for h in sorted(homes):
        parts = [p for p in Path(h).parts if p not in ("/", "\\")]
        if len(parts) >= 2 or len(h) >= 6:
            out.append((h, "the home directory"))
        if len(parts) >= 2:  # a slugged path (episode uids: jobs.types.episode_uid)
            out.append(("-".join(re.sub(r"[^A-Za-z0-9._-]+", "-", p) for p in parts),
                        "the home directory (as a slug)"))
    try:
        host = socket.gethostname()
    except OSError:
        host = ""
    for h in {host, host.split(".")[0]}:
        if len(h) >= 6 and h.lower() not in ("localhost", "localhost.localdomain"):
            out.append((h, "the host name"))
    return out


def _leak(text: str, needles: list[tuple[str, str]]) -> str | None:
    for needle, what in needles:
        if needle in text:
            return what
    return "an absolute local path" if LOCAL_PATH_RE.search(text) else None


def _strings(obj: Any, field: str) -> Iterator[tuple[str, str]]:
    """(field, string) for every key and string value in a JSON value."""
    if isinstance(obj, str):
        yield field, obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            sub = f"{field}.{k}" if field else str(k)
            yield sub, str(k)
            yield from _strings(v, sub)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{field}[{i}]")


def _file_strings(path: Path, rel: str) -> Iterator[tuple[str, str]]:
    """(field, string) to check in one export file. JSON / JSONL by field, other text by line,
    parquet by its key-value metadata (and, under ``meta/``, its rows); other binaries (videos)
    are not read."""
    yield "file name", rel
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        for k, v in (pq.read_schema(path).metadata or {}).items():
            key, val = k.decode("utf-8", "replace"), v.decode("utf-8", "replace")
            yield f"parquet metadata {key!r}", key
            try:
                yield from _strings(json.loads(val), f"parquet metadata {key!r}")
            except ValueError:
                yield f"parquet metadata {key!r}", val
        if rel.startswith("meta/"):  # episode / task tables: small, and they carry text
            for i, row in enumerate(pq.read_table(path).to_pylist()):
                yield from _strings(row, f"row {i}")
        return
    if suffix not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        try:
            yield from _strings(json.loads(text), "")
            return
        except ValueError:
            pass
    for n, line in enumerate(text.splitlines(), 1):
        if suffix == ".jsonl" and line.strip():
            try:
                yield from _strings(json.loads(line), f"line {n}")
                continue
            except ValueError:
                pass
        yield f"line {n}", line


def scan_export(out: str | Path) -> list[tuple[str, str, str]]:
    """``(file, field, what)`` for every string in the files a push of ``out`` uploads that
    names an absolute local path, the home directory or the host name. The value itself is
    never returned (it may be private)."""
    out = Path(out)
    needles = _needles()
    found = []
    for path in sorted(out.rglob("*")):
        rel = path.relative_to(out).as_posix()
        if rel.split("/")[0] in UPLOAD_IGNORE or not path.is_file():
            continue
        seen = set()
        for fld, s in _file_strings(path, rel):
            what = _leak(s, needles)
            if what and fld not in seen:
                seen.add(fld)
                found.append((rel, fld or "(top level)", what))
    return found


def check_portable(out: str | Path) -> None:
    """Raise :class:`LocalPathLeak` (naming each file and field, never the value) when
    :func:`scan_export` finds anything."""
    found = scan_export(out)
    if not found:
        return
    shown = "; ".join(f"{f}: {fld} holds {what}" for f, fld, what in found[:SCAN_REPORT_MAX])
    more = f" (and {len(found) - SCAN_REPORT_MAX} more)" if len(found) > SCAN_REPORT_MAX else ""
    raise LocalPathLeak(
        f"refusing to push: {len(found)} field(s) of the export name this machine's files: "
        f"{shown}{more}. Nothing was uploaded. Re-export without --keep-local-paths, or remove "
        "the paths from the source dataset's files")

FLORA_MODELS = ("nvidia/Flora-TempSeg", "nvidia/Flora-Captioner", "nvidia/Flora-Rewarder")
FRAMEWORK_URL = "https://github.com/kaushikb11/flora-deploy"


def fetch_source(repo: str, revision: str, subdir: str, dest: str | Path, *,
                 include_videos: bool = False, download: Callable[..., str] | None = None
                 ) -> Path:
    """``<dest>/<subdir>``: the dataset's ``meta/`` and ``data/`` at ``revision``."""
    if not revision:
        raise ExportError(f"{repo}: the job recorded no dataset revision; pass --source")
    if download is None:
        from huggingface_hub import snapshot_download as download
    prefix = f"{subdir.strip('/')}/" if subdir.strip("/") else ""
    patterns = [f"{prefix}meta/**", f"{prefix}data/**"]
    if include_videos:
        patterns.append(f"{prefix}videos/**")
    root = Path(download(repo_id=repo, repo_type="dataset", revision=revision,
                         allow_patterns=patterns, local_dir=str(dest)))
    return root / prefix if prefix else root


def _yaml_str(s: str) -> str:
    return json.dumps(s)  # a JSON string is a valid YAML scalar


def dataset_card(info: Mapping[str, Any], report: Any) -> str:
    """``README.md`` of an export (``info``: the export's ``meta/info.json``)."""
    fl = info["flora"]
    src = fl["source"]
    pol = fl["policy"]
    job = fl.get("job") or {}
    src_name = src.get("repo") or src.get("name") or src.get("path")
    src_ref = f"{src_name}@{src['revision']}" if src.get("revision") else str(src_name)
    v3 = str(info.get("codebase_version", "")).startswith("v3.")
    data_glob = "data/*/*.parquet"
    lines = [
        "---",
        "license: other",
        "license_name: nvidia-oneway-noncommercial-labels",
        "tags:",
        "- LeRobot",
        "- flora",
        "- flora-deploy",
        "task_categories:",
        "- robotics",
        "configs:",
        "- config_name: default",
        "  data_files: " + _yaml_str(data_glob),
        "---",
        "",
        f"# {src_name} with Flora labels",
        "",
        f"A copy of the LeRobot {info.get('codebase_version')} dataset `{src_ref}` with labels "
        "from NVIDIA's Flora models added by "
        f"[flora-deploy]({FRAMEWORK_URL}) (`{fl['framework']}`). Only the `flora_*` fields and "
        "this card are new; everything else is the source's.",
        "",
    ]
    if pol["allow_fake"] and pol["records"].get("fake"):
        lines += ["> **WARNING: these labels were made by fake models** (a CPU rehearsal). They "
                  "are not real labels; do not train on them.", ""]
    if pol["records"].get("uncertified"):
        lines += [f"> **{pol['records']['uncertified']} of the label records are uncertified** "
                  "(exported with `--allow-uncertified`). See `flora_certified` and "
                  "`flora_provenance` per episode.", ""]
    lines += ["## What was added", "", "Per episode (episode metadata, "
              + ("`meta/episodes/*/*.parquet`" if v3 else "`meta/episodes.jsonl`") + "):", ""]
    for name, dtype in fl["episode_fields"].items():
        lines.append(f"- `{name}`: `{dtype}`")
    if fl["frame_features"]:
        lines += ["", "Per frame (data parquet; declared in `meta/info.json` `features`):", ""]
        if DENSE_COLUMN in fl["frame_features"]:
            lines.append(f"- `{DENSE_COLUMN}` (float32): NVIDIA's dense reward, 1 failure, "
                         "2 suboptimal, 3 normal. A frame at timestamp `t` gets the 0.5 s step "
                         "`floor((t + 1e-4) / 0.5)`; NaN where no step covers the frame.")
        if SUBTASK_COLUMN in fl["frame_features"]:
            lines.append(f"- `{SUBTASK_COLUMN}` (int64): index into `flora_subtasks` of the "
                         "subtask the frame is in, -1 when none.")
    lines += [
        "",
        f"Episodes without a label have null episode fields and NaN / -1 frame values; "
        f"`{REPORT_NAME}` lists them ({fl['unlabeled_episodes']} here).",
        "",
        "## Provenance",
        "",
        f"- Source: `{src_ref}`",
        f"- Job: `{job.get('job_id')}`, config hash `{job.get('config_hash')}`, output "
        f"generation `{job.get('generation')}`, status `{job.get('status')}`",
        f"- Stages: {', '.join(f'`{s}`' for s in fl['stages'])}",
        f"- Records: {pol['records'].get('certified', 0)} certified, "
        f"{pol['records'].get('uncertified', 0)} uncertified, "
        f"{pol['records'].get('fake', 0)} fake-model",
        f"- Export flags: allow_uncertified={pol['allow_uncertified']}, "
        f"allow_fake={pol['allow_fake']}",
        "- Per-episode preset, engine, certificate and config hash: `flora_provenance`; the "
        "full export metadata: `meta/info.json` `flora`.",
        "",
    ]
    vid = fl["videos"]
    lines += ["## Videos", "", "Included." if vid["included"] else vid["note"], ""]
    lines += [
        "## License",
        "",
        LABEL_LICENSE,
        "",
        "## Citation",
        "",
        "The labels come from NVIDIA's Flora models; cite them as their model cards ask:",
        "",
        *[f"- [{m}](https://huggingface.co/{m})" for m in FLORA_MODELS],
        "",
        f"Labels were produced and exported with [flora-deploy]({FRAMEWORK_URL}). Cite the "
        "source dataset as its own card asks.",
        "",
    ]
    return "\n".join(lines)


def push_to_hub(out: str | Path, repo_id: str, *, private: bool = True, api: Any = None,
                commit_message: str | None = None) -> str:
    """Upload the export directory ``out`` to the dataset repo ``repo_id``; return its URL."""
    out = Path(out)
    info_path = out / "meta" / "info.json"
    if not info_path.is_file() or "flora" not in json.loads(info_path.read_text("utf-8")):
        raise ExportError(f"{out} is not a flora export (no meta/info.json flora block)")
    info = json.loads(info_path.read_text("utf-8"))
    check_portable(out)  # before any network call
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    existing = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if private and not getattr(existing, "private", False):
        raise ExportError(f"{repo_id} already exists and is public; refusing to push the labels "
                          "there (make it private, or pass --public)")
    job = info["flora"].get("job") or {}
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(out),
        commit_message=commit_message or (f"flora export of job {job.get('job_id')} "
                                          f"(config {job.get('config_hash')})"),
        ignore_patterns=[f"{d}/**" for d in UPLOAD_IGNORE])
    version = str(info.get("codebase_version", ""))
    if version:  # LeRobot resolves a Hub dataset's revision by this tag: point it at the push
        from huggingface_hub.errors import HfHubHTTPError

        try:
            api.delete_tag(repo_id=repo_id, repo_type="dataset", tag=version)
        except HfHubHTTPError:
            pass  # no such tag yet
        api.create_tag(repo_id=repo_id, repo_type="dataset", tag=version,
                       revision=getattr(commit, "oid", None))
    return f"https://huggingface.co/datasets/{repo_id}"
