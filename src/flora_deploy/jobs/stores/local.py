"""``LocalStore``: a directory on the local filesystem (tests, dev, single-machine runs).

Writes are atomic: the bytes go to a hidden temp file in the destination directory, are fsynced,
then ``os.replace``d onto the final name, so a reader sees either no file or the whole file.
Listing is strongly consistent (it is the filesystem).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

TMP_MARK = ".flora-tmp-"


def check_rel_path(path: str) -> str:
    """Store paths are relative POSIX paths without ``.``/``..``/empty components."""
    parts = path.split("/")
    if (not path or path.startswith("/") or "\\" in path
            or any(p in ("", ".", "..") for p in parts) or TMP_MARK in path):
        raise ValueError(f"invalid store path {path!r}")
    return path


def list_tree(root: Path, prefix: str) -> Iterator[str]:
    """Relative paths of files under ``root`` whose path starts with ``prefix`` (a string prefix,
    not necessarily a directory boundary), sorted. Temp files are skipped. Only the deepest
    directory the prefix names is walked."""
    base = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
    start = root / base if base else root
    if not start.is_dir():
        return
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(start):
        for name in files:
            if name.startswith(TMP_MARK):
                continue
            rel = Path(dirpath, name).relative_to(root).as_posix()
            if rel.startswith(prefix):
                out.append(rel)
    yield from sorted(out)


class LocalStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs(self, path: str) -> Path:
        return self.root / check_rel_path(path)

    def put_bytes(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes | bytearray):
            raise TypeError("whole-file bytes only")
        dest = self._abs(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f"{TMP_MARK}{uuid.uuid4().hex}"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    def get_bytes(self, path: str) -> bytes:
        return self._abs(path).read_bytes()

    def exists(self, path: str) -> bool:
        return self._abs(path).is_file()

    def list(self, prefix: str) -> Iterator[str]:
        return list_tree(self.root, prefix)
