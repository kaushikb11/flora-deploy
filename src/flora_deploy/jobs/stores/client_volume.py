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

**Reads in parallel** (:meth:`prefetch`, scale-test rehearsal: ``flora run --detach`` took
5.5 min, ``flora status`` 20-29 s, ``flora report`` 69 s, ``flora compose`` 321 s). The bounded
listings left one ``read_file`` round trip per shard file, one after another: ``submit``'s plan
reads every index receipt (one per episode), ``status`` a call ledger and a receipt per shard,
``report`` also an attempt marker per shard, ``compose`` a receipt and the parts per shard.
Callers that are about to read a whole area (``jobs.store.prefetch_areas``) hand its files to
:meth:`prefetch`, which reads them :data:`PREFETCH_WORKERS` at a time into the read cache, so
the loop that follows is served from memory: the same number of reads, in about
1/:data:`PREFETCH_WORKERS` of the wall time.

**Fewer serial round trips** (perf/cli-volume-latency: ``flora status`` still took 20-32 s,
``report`` and ``compose`` ~70 s on a 15-shard job). What was left was serial: after its
``refresh`` a status listed each area it touched (``calls``, ``receipts``, ``replans``, five
``control`` kinds, ``errors``, ``containers``, ``attempts``) one call after another, about 20
listings, and compose uploaded its 9 output files as 9 separate ``batch_upload`` commits. Now
:meth:`snapshot` lists a job's whole directory in one recursive call (every later ``exists`` /
``list`` below it is answered from that listing), and :meth:`put_many` uploads several files in
one ``batch_upload`` (one commit).
"""

from __future__ import annotations

import io
import posixpath
import threading
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flora_deploy.jobs.stores.local import check_list_prefix, check_rel_path

_FILE = 1  # modal.volume.FileEntryType.FILE
LIST_DEPTH = 5  # flora/<job>/<hash>/<area>/<kind>
MAX_CACHED_FILE = 1 << 20
MAX_CACHED_BYTES = 64 << 20
PREFETCH_WORKERS = 32  # concurrent read_file calls in prefetch()


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

    def _count(self, call: str) -> None:
        with self._lock:
            self.calls[call] += 1

    @classmethod
    def from_uri(cls, uri: str, *, environment: str | None = None,
                 create: bool = False) -> ClientVolumeStore:
        """``modal-volume://name[/root]`` through ``modal.Volume.from_name`` (no container).
        ``create``: make the Volume if it does not exist yet (writers such as ``flora index
        --local``, which runs before ``flora deploy`` would create it)."""
        from urllib.parse import urlparse

        import modal

        u = urlparse(uri)
        if not u.netloc:
            raise ValueError(f"store {uri!r}: modal-volume://<name>[/path]")
        vol = modal.Volume.from_name(u.netloc, environment_name=environment, version=2,
                                     create_if_missing=create)
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
        self.put_many({path: data})

    def put_many(self, files: Mapping[str, bytes]) -> None:
        """Write every file of ``files`` (store-relative path -> bytes) in one ``batch_upload``
        (one round trip and one commit instead of one per file). Each file is still a whole-file
        write; callers that need an order between files (compose's pointer last) make separate
        calls."""
        dests = [(self._abs(p), bytes(d)) for p, d in files.items()]
        if not dests:
            return
        self._count("batch_upload")
        with self.volume.batch_upload(force=True) as batch:
            for dest, data in dests:
                batch.put_file(io.BytesIO(data), dest)
        self.refresh()

    def get_bytes(self, path: str) -> bytes:
        check_rel_path(path)  # before the cache, not only on a miss
        with self._lock:
            hit = self._reads.get(path)
        if hit is not None:
            return hit
        self._count("read_file")
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

    def prefetch(self, paths: Iterable[str], *, workers: int = PREFETCH_WORKERS) -> int:
        """Read ``paths`` (store-relative) into the read cache, ``workers`` at a time (module
        docstring). Paths already cached are skipped; a missing or unreadable file is left for
        the caller's own read to report; reading stops once the cache is full. Returns the
        number of files read."""
        with self._lock:
            todo = [p for p in dict.fromkeys(paths) if p not in self._reads]
        if not todo:
            return 0

        def one(path: str) -> int:
            with self._lock:
                full = self._read_bytes >= MAX_CACHED_BYTES
            if full:
                return 0
            try:
                self.get_bytes(path)
            except Exception:  # noqa: BLE001 - the caller's own read raises it
                return 0
            return 1

        if len(todo) == 1 or workers <= 1:
            return sum(one(p) for p in todo)
        with ThreadPoolExecutor(max_workers=min(workers, len(todo)),
                                thread_name_prefix="flora-prefetch") as pool:
            return sum(pool.map(one, todo))

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
            self._count("listdir")
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

    def snapshot(self, d: str) -> int:
        """List directory ``d`` (store-relative; e.g. a job root ``flora/<job>``) in one recursive
        call, unless a cached listing already covers it, so every ``exists`` / ``list`` below it
        is answered from memory until the next :meth:`refresh` or write. Returns the number of
        files under ``d``."""
        d = d.strip("/")
        if d:
            check_list_prefix(d + "/")
        return len(self._files_under(d))

    def exists(self, path: str) -> bool:
        check_rel_path(path)
        return path in self._listing(posixpath.dirname(path))

    def list(self, prefix: str) -> Iterator[str]:
        check_list_prefix(prefix)
        base = prefix.rstrip("/")
        d = base if prefix.endswith("/") or not base else posixpath.dirname(base)
        out = [p for p in self._files_under(d)
               if p.startswith(prefix) and ".tmp" not in p.rsplit("/", 1)[-1]]
        return iter(sorted(out))
