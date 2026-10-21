"""Dataset readers. ``open_source`` picks one from the directory layout."""

from __future__ import annotations

import json
from pathlib import Path

from flora_deploy.media.sources.base import (
    CameraMap,
    EpisodeRef,
    Source,
    SourceError,
    resolve_camera_map,
    verify_episode,
)
from flora_deploy.media.sources.lerobot_v3 import LeRobotV3Source
from flora_deploy.media.sources.lerobot_v21 import LeRobotV21Source
from flora_deploy.media.sources.mp4_dir import Mp4DirSource

__all__ = [
    "CameraMap",
    "EpisodeRef",
    "LeRobotV21Source",
    "LeRobotV3Source",
    "Mp4DirSource",
    "Source",
    "SourceError",
    "open_source",
    "resolve_camera_map",
    "verify_episode",
]


def open_source(root: str | Path, *, camera_map: CameraMap | None = None) -> Source:
    """LeRobot v2.x / v3.x when ``meta/info.json`` exists, else an MP4 directory."""
    root = Path(root)
    info = root / "meta" / "info.json"
    if info.is_file():
        version = str(json.loads(info.read_text(encoding="utf-8")).get("codebase_version", ""))
        if version.startswith("v3."):
            return LeRobotV3Source(root, camera_map=camera_map)
        if version.startswith("v2."):
            return LeRobotV21Source(root, camera_map=camera_map)
        raise SourceError(f"{root}: unsupported LeRobot codebase_version {version!r}")
    return Mp4DirSource(root, camera_map=camera_map)
