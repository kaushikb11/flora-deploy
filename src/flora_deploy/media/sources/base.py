"""Dataset sources: enumerate episodes, their per-camera ``SourceRef``s and task text.

A source never decodes video. ``index.index_episode`` turns an ``EpisodeRef`` into an
``EpisodeIndex``.

Camera roles. NVIDIA's pipeline accepts exactly one view (``main``) or three views named
``main``, ``left_wrist`` and ``right_wrist`` (batch_utils ``api.py:16-28``, ``media.py:103-105``).
A ``CameraMap`` maps dataset camera keys (e.g. LeRobot ``observation.images.top``) to those roles;
keys not in the map are ignored (depth streams, extra views). Without a map, a dataset with a
single camera maps it to ``main``, and keys already named after roles map to themselves;
anything else is ambiguous and raises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, get_args, runtime_checkable

from flora_deploy.media.types import CameraRole, SourceRef

ROLES: tuple[str, ...] = get_args(CameraRole)
CameraMap = Mapping[str, CameraRole]


class SourceError(ValueError):
    """The dataset layout or metadata does not match the reader."""


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    cameras: dict[CameraRole, SourceRef]
    task: str | None = None
    extra: dict[str, object] = field(default_factory=dict)  # e.g. LeRobot episode_index/length


@runtime_checkable
class Source(Protocol):
    """What every dataset reader provides."""

    name: str  # dataset identifier recorded in EpisodeIndex.dataset

    def list_episodes(self) -> Sequence[EpisodeRef]: ...


def resolve_camera_map(keys: Sequence[str], camera_map: CameraMap | None) -> dict[str, CameraRole]:
    """Dataset camera key -> role, validated (each role used once, known roles only)."""
    if camera_map is None:
        if len(keys) == 1:
            return {keys[0]: "main"}
        named = {k: k for k in keys if k in ROLES}
        if named and "main" in named:
            return named  # type: ignore[return-value]
        raise SourceError(f"cameras {list(keys)}: pass a camera map (camera key -> one of {ROLES})")
    unknown_roles = sorted({r for r in camera_map.values() if r not in ROLES})
    if unknown_roles:
        raise SourceError(f"unknown camera roles {unknown_roles}; use {ROLES}")
    missing = sorted(k for k in camera_map if k not in keys)
    if missing:
        raise SourceError(f"camera keys {missing} not in dataset cameras {list(keys)}")
    roles = list(camera_map.values())
    dupes = sorted({r for r in roles if roles.count(r) > 1})
    if dupes:
        raise SourceError(f"camera roles mapped more than once: {dupes}")
    if "main" not in roles:
        raise SourceError("the camera map needs a 'main' camera")
    return dict(camera_map)


def task_text(tasks: Sequence[str] | str | None) -> str | None:
    """LeRobot episodes carry a list of task strings; one task -> itself, several -> joined
    by newlines in their stored order, none -> None."""
    if tasks is None:
        return None
    if isinstance(tasks, str):
        return tasks
    tasks = [t for t in tasks if t]
    return "\n".join(tasks) if tasks else None
