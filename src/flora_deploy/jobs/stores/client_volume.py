"""``Store`` over a Modal Volume (v2) through the Modal **client** API, for processes that do not
mount it: your laptop (``flora results``, ``flora index --local``, ``flora status``, the SDK). No
container is started; every call is a Volume metadata / file call.

- writes: whole files through ``Volume.batch_upload(force=True)`` (atomic per file on v2);
- reads: ``Volume.read_file`` (whole file);
- listings: ``Volume.listdir(recursive=True)`` of the deepest directory the prefix names.

**Listings are cached per directory** for the store object's lifetime (and dropped on every
write through it), so ``exists`` on many files of one directory (receipts, markers) is one
``listdir``, not one read each: ``flora status`` on a job with thousands of shard files makes
O(directories) calls instead of O(files) (H0). Writes by *other* processes after a listing are
not seen until :meth:`refresh`; a reader that polls (``Job.wait``) calls it every poll.
"""

from __future__ import annotations

import io
import posixpath
import threading
from collections.abc import Iterator
from typing import Any

from flora_deploy.jobs.stores.local import check_rel_path

_FILE = 1  # modal.volume.FileEntryType.FILE


def _not_found(e: BaseException) -> bool:
    return isinstance(e, FileNotFoundError) or type(e).__name__ == "NotFoundError"


class ClientVolumeStore:
    """See the module docstring. ``volume``: a ``modal.Volume`` (or a fake with ``read_file``,
    ``listdir`` and ``batch_upload``); ``root``: the store's path inside the Volume."""

    def __init__(self, volume: Any, root: str = ""):
        self.volume = volume
        self.root = root.strip("/")
        self._lock = threading.Lock()
        self._dirs: dict[str, frozenset[str]] = {}  # store-relative dir -> files under it
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
        """Forget cached listings (see the module docstring)."""
        with self._lock:
            self._dirs.clear()

    # Store ---------------------------------------------------------------------------------
    def put_bytes(self, path: str, data: bytes) -> None:
        dest = self._abs(path)
        self.calls["batch_upload"] += 1
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), dest)
        self.refresh()

    def get_bytes(self, path: str) -> bytes:
        self.calls["read_file"] += 1
        try:
            return b"".join(self.volume.read_file(self._abs(path)))
        except Exception as e:
            if _not_found(e):
                raise FileNotFoundError(path) from None
            raise

    def _files_under(self, d: str) -> frozenset[str]:
        """Store-relative paths of every file under directory ``d`` (cached)."""
        with self._lock:
            hit = self._dirs.get(d)
        if hit is not None:
            return hit
        # a cached ancestor listing already holds everything below it
        with self._lock:
            for anc, files in self._dirs.items():
                if not anc or d.startswith(anc + "/"):
                    pre = d + "/"
                    return frozenset(f for f in files if f.startswith(pre))
        self.calls["listdir"] += 1
        try:
            entries = self.volume.listdir(self._abs_dir(d), recursive=True)
        except Exception as e:
            if not _not_found(e):
                raise
            entries = []
        cut = len(self.root) + 1 if self.root else 0
        files = frozenset(
            e.path.lstrip("/")[cut:] for e in entries
            if int(getattr(e.type, "value", e.type)) == _FILE)
        with self._lock:
            self._dirs[d] = files
        return files

    def exists(self, path: str) -> bool:
        check_rel_path(path)
        return path in self._files_under(posixpath.dirname(path))

    def list(self, prefix: str) -> Iterator[str]:
        base = prefix.rstrip("/")
        d = base if prefix.endswith("/") or not base else posixpath.dirname(base)
        out = [p for p in self._files_under(d)
               if p.startswith(prefix) and ".tmp" not in p.rsplit("/", 1)[-1]]
        return iter(sorted(out))
