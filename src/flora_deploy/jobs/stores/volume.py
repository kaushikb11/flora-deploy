"""``VolumeStore``: job files on a mounted Modal Volume v2 (the fallback when no bucket is
configured; PLAN.md D6, §7 Storage). No ``modal`` import: the caller passes the ``modal.Volume``.

Discipline (research_notes ``modal_verification.md`` findings 2-4):

* **Never hold Volume files open.** ``put_bytes`` writes the bytes to a local temp file first
  (``tmp_dir``, default the system temp dir), then copies the whole file onto the mount under a
  temp name and renames it into place. Reads open, read everything and close at once. Nothing is
  open between calls, so a ``reload()`` in the container cannot fail on our files.
* **Commit batching.** Other containers see writes only after ``Volume.commit()``. Each write
  bumps a pending counter, and every ``commit_every`` writes (default 1: after each write) we
  commit. Call ``commit()`` explicitly after a shard's last write (the receipt) and in
  ``@modal.exit``. A SIGKILLed container loses uncommitted writes; that is safe because the
  receipt is written, and committed, last.
* **Reload only when told no files are open.** ``reload(no_open_files=True)`` is the caller's
  promise that no input in this container has a Volume file open (serialize it with a
  container-level lock when ``@modal.concurrent`` inputs share the container). While a reload
  runs the mount looks empty to the container. ``reload`` refuses without the promise.
* Listing walks the deepest directory the prefix names and shows this container's view (as of
  start or the last reload), so it lags other containers' commits: an eventually consistent
  listing, exactly what ``jobs.store`` tells callers to expect.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flora_deploy.jobs.stores.local import TMP_MARK, check_rel_path, list_tree


class VolumeStore:
    def __init__(self, mount: str | os.PathLike[str], volume: Any | None = None, *,
                 commit_every: int = 1, tmp_dir: str | os.PathLike[str] | None = None):
        if commit_every < 1:
            raise ValueError("commit_every must be >= 1")
        self.mount = Path(mount)
        self.volume = volume
        self.commit_every = commit_every
        self.tmp_dir = tmp_dir
        self._pending = 0
        self._lock = threading.Lock()

    def _abs(self, path: str) -> Path:
        return self.mount / check_rel_path(path)

    def put_bytes(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes | bytearray):
            raise TypeError("whole-file bytes only")
        dest = self._abs(path)
        fd, local = tempfile.mkstemp(prefix=TMP_MARK, dir=self.tmp_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            dest.parent.mkdir(parents=True, exist_ok=True)
            staged = dest.parent / f"{TMP_MARK}{uuid.uuid4().hex}"
            try:
                shutil.copyfile(local, staged)  # opens, copies, closes
                os.replace(staged, dest)
            finally:
                staged.unlink(missing_ok=True)
        finally:
            os.unlink(local)
        with self._lock:
            self._pending += 1
            due = self._pending >= self.commit_every
        if due:
            self.commit()

    def get_bytes(self, path: str) -> bytes:
        return self._abs(path).read_bytes()

    def exists(self, path: str) -> bool:
        return self._abs(path).is_file()

    def list(self, prefix: str) -> Iterator[str]:
        return list_tree(self.mount, prefix)

    @property
    def pending_writes(self) -> int:
        return self._pending

    def commit(self) -> None:
        """Make this container's writes visible to others (no-op without a ``volume``)."""
        with self._lock:
            self._pending = 0
        if self.volume is not None:
            self.volume.commit()

    def reload(self, *, no_open_files: bool = False) -> None:
        """See other containers' commits. ``no_open_files=True`` is required: the caller vouches
        that no Volume file is open in this container (Modal's reload fails otherwise)."""
        if not no_open_files:
            raise RuntimeError("VolumeStore.reload() needs no_open_files=True: reload fails (or "
                               "blanks the mount) while any Volume file is open")
        if self.volume is not None:
            self.volume.reload()
