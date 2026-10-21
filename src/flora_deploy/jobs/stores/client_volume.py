"""``Store`` over a Modal Volume (v2) through the Modal **client** API, for processes that do not
mount it: your laptop (``flora results``, ``flora index --local``, ``flora status``, the SDK). No
container is started; every call is a Volume metadata / file call.

- writes: whole files through ``Volume.batch_upload(force=True)`` (atomic per file on v2);
- reads: ``Volume.read_file`` (whole file);
- listings: ``Volume.listdir(recursive=True)``.

**One snapshot per refresh (H0: ``flora status`` was slow).** Every call is a network round
trip, and the job layout fans files out by shard id (``receipts/<kind>/<xx>/<yy>/<shard>.json``),
so a status that probes thousands of shard files made one call each. This store instead lists
the directory at most :data:`LIST_DEPTH` path components deep that holds the file (e.g.
``flora/<job>/<stage_hash>/receipts/<kind>``) once, recursively, and answers ``exists`` / ``list``
for everything below it from that listing; and it keeps what it read (files up to
:data:`MAX_CACHED_FILE` bytes, :data:`MAX_CACHED_BYTES` in all), so a file read twice in one
status is fetched once. Both caches are dropped by :meth:`refresh` and by every write through
this store. Writes by *other* processes are seen after the next :meth:`refresh`; ``sdk.Job``
refreshes at the start of every ``status()`` (so every ``wait`` poll).
"""

from __future__ import annotations

import io
import posixpath
import threading
from collections.abc import Iterator
from typing import Any

from flora_deploy.jobs.stores.local import check_rel_path

_FILE = 1  # modal.volume.FileEntryType.FILE
LIST_DEPTH = 5  # flora/<job>/<hash>/<area>/<kind>
MAX_CACHED_FILE = 1 << 20
MAX_CACHED_BYTES = 64 << 20


def _not_found(e: BaseException) -> bool:
    return isinstance(e, FileNotFoundError) or type(e).__name__ == "NotFoundError"


class ClientVolumeStore:
    """See the module docstring. ``volume``: a ``modal.Volume`` (or a fake with ``read_file``,
    ``listdir`` and ``batch_upload``); ``root``: the store's path inside the Volume."""

    def __init__(self, volume: Any, root: str = ""):
        self.volume = volume
        self.root = root.strip("/")
        self._lock = threading.Lock()
        self._dirs: dict[str, frozenset[str]] = {}  # listed dir -> store-relative files under it
        self._reads: dict[str, bytes] = {}
        self._read_bytes = 0
        self.calls = {"listdir": 0, "read_file": 0, "batch_upload": 0}

    @classmethod
    def from_uri(cls, uri: str, *, environment: str | None = None) -> ClientVolumeStore:
        """``modal-volume://name[/root]`` through ``modal.Volume.from_name`` (no container)."""
        from urllib.parse import urlparse

        import modal

        u = urlparse(uri)
        if not u.netloc:
            raise ValueError(f"store {uri!r}: modal-volume://<name>[/path]")
        vol = modal.Volume.from_name(u.netloc, environment_name=environment, version=2)
        return cls(vol, u.path)

    # paths ---------------------------------------------------------------------------------
    def _abs(self, path: str) -> str:
        check_rel_path(path)
        return f"/{self.root}/{path}" if self.root else f"/{path}"

    def _abs_dir(self, d: str) -> str:
        if not d:
            return f"/{self.root}" if self.root else "/"
        return self._abs(d)

    def refresh(self) -> None:
        """Drop the cached listings and reads (see the module docstring)."""
        with self._lock:
            self._dirs.clear()
            self._reads.clear()
            self._read_bytes = 0

    # Store ---------------------------------------------------------------------------------
    def put_bytes(self, path: str, data: bytes) -> None:
        dest = self._abs(path)
        self.calls["batch_upload"] += 1
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), dest)
        self.refresh()

    def get_bytes(self, path: str) -> bytes:
        with self._lock:
            hit = self._reads.get(path)
        if hit is not None:
            return hit
        self.calls["read_file"] += 1
        try:
            data = b"".join(self.volume.read_file(self._abs(path)))
        except Exception as e:
            if _not_found(e):
                raise FileNotFoundError(path) from None
            raise
        if len(data) <= MAX_CACHED_FILE:
            with self._lock:
                if self._read_bytes + len(data) <= MAX_CACHED_BYTES:
                    self._reads[path] = data
                    self._read_bytes += len(data)
        return data

    def _files_under(self, d: str) -> frozenset[str]:
        """Store-relative paths of every file under directory ``d``."""
        files = self._listing(d)
        if not d:
            return files
        pre = d + "/"
        return frozenset(f for f in files if f.startswith(pre))

    def _listing(self, d: str) -> frozenset[str]:
        """Every file under ``d``'s ancestor at most :data:`LIST_DEPTH` components deep (a
        superset of the files under ``d``), from one cached recursive listing."""
        parts = d.split("/") if d else []
        top = "/".join(parts[:LIST_DEPTH])
        with self._lock:
            files = self._dirs.get(top)
            if files is None:  # a cached ancestor listing already holds everything below it
                for anc, fs in self._dirs.items():
                    if not anc or top.startswith(anc + "/"):
                        files = fs
                        break
        if files is None:
            self.calls["listdir"] += 1
            try:
                entries = self.volume.listdir(self._abs_dir(top), recursive=True)
            except Exception as e:
                if not _not_found(e):
                    raise
                entries = []
            cut = len(self.root) + 1 if self.root else 0
            files = frozenset(
                e.path.lstrip("/")[cut:] for e in entries
                if int(getattr(e.type, "value", e.type)) == _FILE)
            with self._lock:
                self._dirs[top] = files
        return files

    def exists(self, path: str) -> bool:
        check_rel_path(path)
        return path in self._listing(posixpath.dirname(path))

    def list(self, prefix: str) -> Iterator[str]:
        base = prefix.rstrip("/")
        d = base if prefix.endswith("/") or not base else posixpath.dirname(base)
        out = [p for p in self._files_under(d)
               if p.startswith(prefix) and ".tmp" not in p.rsplit("/", 1)[-1]]
        return iter(sorted(out))
