"""LeRobot ``codebase_version`` v3.x datasets (episodes concatenated into per-camera MP4s).

Schema (huggingface/lerobot @ afbf3f14): ``src/lerobot/datasets/utils.py`` constants
``INFO_PATH``,
``DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"``,
``DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"``;
episode rows written by ``dataset_metadata.save_episode`` (``episode_index``, ``tasks`` - a list
of strings -, ``length``, plus episode metadata) and, per camera, the columns
``videos/{video_key}/chunk_index``, ``videos/{video_key}/file_index``,
``videos/{video_key}/from_timestamp`` and ``videos/{video_key}/to_timestamp`` (seconds in the
concatenated file; ``scripts/convert_dataset_v21_to_v30.convert_videos_of_camera`` and
``aggregate.py`` compute them as float sums of per-episode ``get_video_duration_in_s``).

Each camera of an episode becomes ``SourceRef(file, from_timestamp, to_timestamp)``.
``index.select_episode`` matches frames to that range with a half-frame tolerance.
"""

from __future__ import annotations

from pathlib import Path

from flora_deploy.media.sources.base import (
    CameraMap,
    EpisodeRef,
    SourceError,
    resolve_camera_map,
    task_text,
)
from flora_deploy.media.sources.lerobot_v21 import load_info, video_keys
from flora_deploy.media.types import SourceRef

DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


class LeRobotV3Source:
    def __init__(self, root: str | Path, *, camera_map: CameraMap | None = None, name: str = ""):
        self.root = Path(root)
        self.info = load_info(self.root)
        version = str(self.info.get("codebase_version", ""))
        if not version.startswith("v3."):
            raise SourceError(f"{self.root}: codebase_version {version!r} is not v3.x")
        self.template = self.info.get("video_path") or ""
        if not self.template:
            raise SourceError(f"{self.root}: dataset stores no videos (video_path is null)")
        self.roles = resolve_camera_map(video_keys(self.info), camera_map)
        self.name = name or self.root.name

    def _rows(self) -> list[dict]:
        import pyarrow.parquet as pq

        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise SourceError(f"{self.root}: no meta/episodes/chunk-*/file-*.parquet")
        rows: list[dict] = []
        for f in files:
            rows += pq.read_table(f).to_pylist()
        return sorted(rows, key=lambda r: int(r["episode_index"]))

    def list_episodes(self) -> list[EpisodeRef]:
        out = []
        for row in self._rows():
            e = int(row["episode_index"])
            cams = {}
            for key, role in self.roles.items():
                cols = [f"videos/{key}/{c}" for c in ("chunk_index", "file_index")]
                ts = [f"videos/{key}/{c}" for c in ("from_timestamp", "to_timestamp")]
                missing = [c for c in cols + ts if c not in row]
                if missing:
                    raise SourceError(f"episode {e}: missing columns {missing}")
                rel = self.template.format(
                    video_key=key, chunk_index=int(row[cols[0]]), file_index=int(row[cols[1]])
                )
                cams[role] = SourceRef(str(self.root / rel), float(row[ts[0]]), float(row[ts[1]]))
            out.append(
                EpisodeRef(
                    episode_id=f"episode_{e:06d}",
                    cameras=cams,
                    task=task_text(row.get("tasks")),
                    extra={"episode_index": e, "length": row.get("length")},
                )
            )
        return out
