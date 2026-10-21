"""A directory of MP4 files, one episode per file or per sub-directory.

Layout (both forms may be mixed in one directory; names are case-sensitive):

* **single view** - ``<root>/<episode_id>.mp4``; the file is the ``main`` camera.
  Optional task text in ``<root>/<episode_id>.txt``.
* **multi view** - ``<root>/<episode_id>/<camera_key>.mp4``, one file per camera. Camera keys
  are mapped to roles with the camera map; files named ``main.mp4``, ``left_wrist.mp4`` and
  ``right_wrist.mp4`` need no map. Optional task text in ``<root>/<episode_id>/task.txt``.

``.mp4``, ``.mov`` and ``.m4v`` are accepted. Hidden files are skipped. Each file is a whole
episode (``SourceRef.start_s/end_s`` are None). Episode ids are the stem / directory name and must
be unique.
"""

from __future__ import annotations

from pathlib import Path

from flora_deploy.media.sources.base import (
    CameraMap,
    EpisodeRef,
    SourceError,
    resolve_camera_map,
)
from flora_deploy.media.types import SourceRef

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")


def _read_task(path: Path) -> str | None:
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    return None


class Mp4DirSource:
    def __init__(self, root: str | Path, *, camera_map: CameraMap | None = None, name: str = ""):
        self.root = Path(root)
        if not self.root.is_dir():
            raise SourceError(f"{self.root}: not a directory")
        self.camera_map = camera_map
        self.name = name or self.root.name

    def list_episodes(self) -> list[EpisodeRef]:
        out: dict[str, EpisodeRef] = {}
        for entry in sorted(self.root.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_file() and entry.suffix.lower() in VIDEO_SUFFIXES:
                ep = EpisodeRef(
                    episode_id=entry.stem,
                    cameras={"main": SourceRef(str(entry))},
                    task=_read_task(entry.with_suffix(".txt")),
                )
            elif entry.is_dir():
                files = {
                    f.stem: f
                    for f in sorted(entry.iterdir())
                    if f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES and f.name[0] != "."
                }
                if not files:
                    continue
                roles = resolve_camera_map(list(files), self.camera_map)
                ep = EpisodeRef(
                    episode_id=entry.name,
                    cameras={role: SourceRef(str(files[key])) for key, role in roles.items()},
                    task=_read_task(entry / "task.txt"),
                )
            else:
                continue
            if ep.episode_id in out:
                raise SourceError(f"duplicate episode id {ep.episode_id!r} in {self.root}")
            out[ep.episode_id] = ep
        return list(out.values())
