"""``VolumeStore``: job files on a mounted Modal Volume v2 (the fallback when no bucket is
configured; PLAN.md D6, §2). No ``modal`` import: the caller passes the ``modal.Volume``.

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
  listing, exactly what ``jobs.store`` tells callers to expect. **``exists`` and ``get_bytes``
  lag the same way**: until this container reloads, another container's committed receipt is
  invisible, so "receipts are read-after-write" holds only for this container's own writes.
* **``refresh()`` once per orchestrator tick.** A reader that must see other containers' files
  (the orchestrator, the reconciler, the budget gate) calls :meth:`VolumeStore.refresh` at the
  start of every tick and never caches a listing across ticks. ``refresh`` reloads only when no
  store operation is in flight in this container (it tracks its own calls; nothing is held open
  between calls), commits this container's pending writes first, and otherwise skips and returns
  ``False``. The caller still vouches that *other* code in the container has no Volume file open
  (true for the single-threaded, CPU-only orchestrator, which touches the Volume only through
  this store). Bucket and local stores need no refresh; callers use
  ``getattr(store, "refresh", None)``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator, Mapping
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
        self._io = threading.Condition()
        self._busy = 0  # store calls in flight in this container (each opens and closes files)
        self._reloading = False
        self.reloads = 0
        self.skipped_refreshes = 0

    def _enter(self) -> None:
        with self._io:
            while self._reloading:  # never touch the mount while it reloads (it looks empty)
                self._io.wait()
            self._busy += 1

    def _exit(self) -> None:
        with self._io:
            self._busy -= 1
            self._io.notify_all()

    def _abs(self, path: str) -> Path:
        return self.mount / check_rel_path(path)

    def put_bytes(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes | bytearray):
            raise TypeError("whole-file bytes only")
        dest = self._abs(path)
        self._enter()
        try:
            self._put(dest, data)
        finally:
            self._exit()
        with self._lock:
            self._pending += 1
            due = self._pending >= self.commit_every
        if due:
            self.commit()

    def put_many(self, files: Mapping[str, bytes]) -> None:
        """Write every file of ``files`` (whole files, in order), then commit at most once
        (``commit_every`` counts them all), instead of one commit per file."""
        items = [(self._abs(p), d) for p, d in files.items()]
        if any(not isinstance(d, bytes | bytearray) for _p, d in items):
            raise TypeError("whole-file bytes only")
        if not items:
            return
        self._enter()
        try:
            for dest, data in items:
                self._put(dest, data)
        finally:
            self._exit()
        with self._lock:
            self._pending += len(items)
            due = self._pending >= self.commit_every
        if due:
            self.commit()

    def _put(self, dest: Path, data: bytes) -> None:
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

    def get_bytes(self, path: str) -> bytes:
        self._enter()
        try:
            return self._abs(path).read_bytes()
        finally:
            self._exit()

    def exists(self, path: str) -> bool:
        self._enter()
        try:
            return self._abs(path).is_file()
        finally:
            self._exit()

    def list(self, prefix: str) -> Iterator[str]:
        """Materialized at call time (no lazy directory walk spanning a later ``refresh``)."""
        self._enter()
        try:
            return iter(list(list_tree(self.mount, prefix)))
        finally:
            self._exit()

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
        self.reloads += 1

    def refresh(self) -> bool:
        """Safe per-tick reload: see other containers' commits. Returns ``True`` if it reloaded.

        Skips (``False``) while any call on this store is in flight in another thread, because a
        reload with an open Volume file fails or blanks the mount. Pending writes are committed
        first so a reload never races this container's own uncommitted files. Call it at the start
        of every orchestrator tick (``orchestrator_app`` does); see the module docstring for the
        caller's half of the no-open-files promise."""
        if self._pending:
            self.commit()
        with self._io:
            if self._busy or self._reloading:
                self.skipped_refreshes += 1
                return False
            self._reloading = True  # new calls on this store wait until the reload finishes
        try:
            self.reload(no_open_files=True)
        finally:
            with self._io:
                self._reloading = False
                self._io.notify_all()
        return True
