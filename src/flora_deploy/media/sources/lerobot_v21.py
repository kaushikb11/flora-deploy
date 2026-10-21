"""LeRobot ``codebase_version`` v2.x datasets (one MP4 per episode and camera).

Schema, as written by LeRobot <= 2.1 and documented in the v2.1 -> v3.0 converter
(huggingface/lerobot @ afbf3f14, ``src/lerobot/scripts/convert_dataset_v21_to_v30.py`` header and
``legacy_load_episodes``/``legacy_load_tasks``; path constants ``LEGACY_EPISODES_PATH``,
``LEGACY_TASKS_PATH`` in ``src/lerobot/datasets/utils.py``):

* ``meta/info.json``: ``codebase_version`` ("v2.0"/"v2.1"), ``fps``, ``chunks_size``,
  ``video_path`` template, default
  ``videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`` with
  ``episode_chunk = episode_index // chunks_size``; ``features[key]["dtype"] == "video"`` marks
  camera keys.
* ``meta/episodes.jsonl``: one object per line, ``{"episode_index", "tasks": [str], "length"}``.
* ``meta/tasks.jsonl``: ``{"task_index", "task"}`` (not needed: episodes carry task strings).

Every episode file is the whole episode: ``SourceRef(start_s=None, end_s=None)``.
"""

from __future__ import annotations

import json
from pathlib import Path

from flora_deploy.media.safepath import UnsafePath, check_key, contain, format_template
from flora_deploy.media.sources.base import (
    CameraMap,
    EpisodeRef,
    SourceError,
    resolve_camera_map,
    task_text,
)
from flora_deploy.media.types import SourceRef

DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def load_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise SourceError(f"{root}: missing meta/info.json")
    return json.loads(path.read_text(encoding="utf-8"))


def video_keys(info: dict) -> list[str]:
    return [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]


def check_camera_keys(root: Path, roles: dict[str, str]) -> None:
    """Camera keys fill the ``video_path`` template: one safe path component each."""
    for key in roles:
        try:
            check_key(key)
        except UnsafePath as e:
            raise SourceError(f"{root}: {e}") from None


def video_ref(root: Path, template: str, **fields: str | int) -> str:
    """``root / template.format(**fields)``, refusing anything outside ``root``
    (``media.safepath``: the template and its values come from the dataset)."""
    try:
        return str(contain(root, format_template(template, **fields)))
    except UnsafePath as e:
        raise SourceError(f"{root}: {e}") from None


class LeRobotV21Source:
    def __init__(self, root: str | Path, *, camera_map: CameraMap | None = None, name: str = ""):
        self.root = Path(root)
        self.info = load_info(self.root)
        version = str(self.info.get("codebase_version", ""))
        if not version.startswith("v2."):
            raise SourceError(f"{self.root}: codebase_version {version!r} is not v2.x")
        self.template = self.info.get("video_path") or ""
        if not self.template:
            raise SourceError(f"{self.root}: dataset stores no videos (video_path is null)")
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.roles = resolve_camera_map(video_keys(self.info), camera_map)
        check_camera_keys(self.root, self.roles)
        self.name = name or self.root.name

    def list_episodes(self) -> list[EpisodeRef]:
        path = self.root / "meta" / "episodes.jsonl"
        if not path.is_file():
            raise SourceError(f"{self.root}: missing meta/episodes.jsonl")
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            e = int(row["episode_index"])
            cams = {}
            for key, role in self.roles.items():
                cams[role] = SourceRef(video_ref(
                    self.root, self.template, episode_chunk=e // self.chunks_size,
                    video_key=key, episode_index=e))
            out.append(
                EpisodeRef(
                    episode_id=f"episode_{e:06d}",
                    cameras=cams,
                    task=task_text(row.get("tasks")),
                    extra={"episode_index": e, "length": row.get("length")},
                )
            )
        return out
