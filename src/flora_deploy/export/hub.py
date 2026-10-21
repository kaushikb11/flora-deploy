"""Hugging Face Hub side of ``flora export``: fetch a source dataset, write the dataset card,
push an export.

* :func:`fetch_source` downloads ``meta/`` and ``data/`` (and ``videos/`` when asked) of a Hub
  dataset at its pinned revision; nothing else.
* :func:`dataset_card` renders ``README.md`` for an export: it cites Flora, carries the NVIDIA
  noncommercial notice for the derived labels and records provenance.
* :func:`push_to_hub` uploads an export directory to a dataset repo, **private by default**. It
  refuses a repo that already exists and is public when a private push was asked, and tags the
  commit with the LeRobot ``codebase_version`` (LeRobot resolves Hub datasets by that tag).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from flora_deploy.export.lerobot import (
    DENSE_COLUMN,
    LABEL_LICENSE,
    REPORT_NAME,
    SUBTASK_COLUMN,
    ExportError,
)

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
        ignore_patterns=[".cache/**", ".git/**"])
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
