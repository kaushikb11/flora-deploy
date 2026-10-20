"""Plan (and later perform) downloads of the pinned Hugging Face snapshots.

``plan_snapshot`` resolves exactly which files ``snapshot_download`` would fetch for a pin, using
the same pattern filter, so ``flora prepare --dry-run`` shows sizes before anything is downloaded.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from huggingface_hub.utils import filter_repo_objects

from flora_deploy.config.pins import PINS, RepoPin


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int
    lfs_sha256: str | None


@dataclass(frozen=True)
class SnapshotPlan:
    pin: RepoPin
    files: tuple[FileEntry, ...] = field(default_factory=tuple)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


class TreeLister(Protocol):
    def list_repo_tree(self, repo_id: str, *, revision: str, recursive: bool) -> Iterable: ...


def plan_snapshot(pin: RepoPin, api: TreeLister) -> SnapshotPlan:
    entries = [
        e
        for e in api.list_repo_tree(pin.repo, revision=pin.revision, recursive=True)
        if not hasattr(e, "tree_id")  # skip RepoFolder entries
    ]
    selected = filter_repo_objects(
        entries,
        allow_patterns=list(pin.allow_patterns) if pin.allow_patterns else None,
        key=lambda e: e.path,
    )
    files = tuple(
        sorted(
            (
                FileEntry(
                    path=e.path,
                    size=int(e.size or 0),
                    lfs_sha256=getattr(e.lfs, "sha256", None) if e.lfs else None,
                )
                for e in selected
            ),
            key=lambda f: f.path,
        )
    )
    return SnapshotPlan(pin=pin, files=files)


def plan_all(api: TreeLister, pins: Iterable[RepoPin] = PINS) -> list[SnapshotPlan]:
    return [plan_snapshot(p, api) for p in pins]
