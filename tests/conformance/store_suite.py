"""Conformance suite for ``Store`` implementations (``flora_deploy.jobs.store`` contract).

Each implementation registers a factory ``factory(tmp_path) -> Store`` in :data:`STORES`
(Volume over a fake mount + fake ``modal.Volume``, Bucket over a fake S3 client). Checks are plain
functions raising ``AssertionError``; ``test_stores.py`` runs them all for every store.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path

from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.stores import BucketStore, LocalStore, VolumeStore, check_rel_path


class FakeVolume:
    """``modal.Volume`` subset ``VolumeStore`` calls: ``commit`` / ``reload`` counters."""

    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


class _ClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """boto3-shaped S3 client: whole-object PUT/GET/HEAD and paginated, sorted LIST."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body):  # noqa: N803 - boto3 names
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _ClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _ClientError("404")
        return {}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys, ContinuationToken=None):  # noqa: N803
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + MaxKeys]
        resp = {"Contents": [{"Key": k} for k in page], "IsTruncated": start + MaxKeys < len(keys)}
        if resp["IsTruncated"]:
            resp["NextContinuationToken"] = str(start + MaxKeys)
        return resp


STORES: dict[str, Callable[[Path], Store]] = {
    "local": lambda root: LocalStore(root / "local"),
    "volume": lambda root: VolumeStore(root / "mount", FakeVolume(), tmp_dir=root),
    "bucket": lambda root: BucketStore("bkt", prefix="pre/fix", client=FakeS3(), page_size=2),
}

P = JobPaths("job")
RECEIPT = P.receipt("cfg", "segcap", "segcap-00000")


def check_whole_file(store: Store) -> None:
    """put replaces the whole file (no append, no stale tail); bytes round-trip exactly."""
    blob = bytes(range(256)) * 4 + b"\n\x00tail"
    store.put_bytes(RECEIPT, blob)
    assert store.get_bytes(RECEIPT) == blob
    store.put_bytes(RECEIPT, b"short")
    assert store.get_bytes(RECEIPT) == b"short", "overwrite must replace, not append / keep tail"
    store.put_bytes(RECEIPT, b"")
    assert store.get_bytes(RECEIPT) == b"", "empty file is a valid whole file"
    try:
        store.put_bytes(RECEIPT, "text")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("put_bytes must reject non-bytes")


def check_read_after_write(store: Store) -> None:
    """exists / get see this process's own write immediately; missing files are reported."""
    p = P.attempt("cfg", "segcap", "segcap-00001", "ta-1")
    assert not store.exists(p)
    try:
        store.get_bytes(p)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("get_bytes of a missing file must raise FileNotFoundError")
    store.put_bytes(p, b"{}")
    assert store.exists(p) and store.get_bytes(p) == b"{}"
    assert not store.exists(p + ".other")


def check_list_prefix(store: Store) -> None:
    """list yields full relative paths under a *string* prefix, across directories and pages,
    never temp files, and nothing for an unknown prefix."""
    parts = [P.part("cfg", "segcap", "segcap-00000", f"ta-{c}", k) for c in (1, 2) for k in (0, 1)]
    other = P.part("cfg", "segcap", "segcap-00009", "ta-1", 0)
    for q in [*parts, other, RECEIPT]:
        store.put_bytes(q, b"x\n")
    shard_dir = parts[0].rsplit("/", 1)[0] + "/"
    assert sorted(store.list(shard_dir)) == sorted(parts)
    assert sorted(store.list(shard_dir + "ta-2-")) == sorted(p for p in parts if "/ta-2-" in p), \
        "a prefix need not end at a directory boundary"
    everything = set(store.list(P.root + "/"))
    assert {*parts, other, RECEIPT} <= everything
    assert all(not q.startswith("/") and q.startswith(P.root) for q in everything)
    assert all(".flora-tmp-" not in q for q in everything), "temp files leaked into list()"
    assert list(store.list(P.root + "/nope/")) == []
    assert list(store.list("flora/other-job/")) == []


def check_path_safety(store: Store) -> None:
    """Absolute, empty, dot, dot-dot, backslash and temp-marker paths are rejected."""
    bad = ["", "/abs/x", "a//b", "a/./b", "a/../b", "../x", "a/..", "a\\b", "a/.flora-tmp-x"]
    ops: list[tuple[str, Callable[[str], object]]] = [
        ("put_bytes", lambda q: store.put_bytes(q, b"x")), ("get_bytes", store.get_bytes),
        ("exists", store.exists)]
    for path in bad:
        for name, op in ops:
            try:
                op(path)
            except ValueError:
                continue
            raise AssertionError(f"{name}: unsafe path {path!r} accepted")


def check_job_paths_round_trip(store: Store) -> None:
    """Every JobPaths file kind can be written, found by listing and read back."""
    paths = [P.manifest, P.job_cache, P.index("d--e1"), P.control("cancel", "job", "20261021T1"),
             P.container("ta-1"), P.part("c", "segcap", "s", "ta-1", 0), RECEIPT,
             P.replan("c", "segcap", "s"), P.call("c", "segcap", "s"),
             P.attempt("c", "segcap", "s", "ta-1"), P.errors("c", "segcap", "s", "ta-1", 0),
             P.audit("c", "segcap", "s", "ta-1", 0), P.output("c", "segment")]
    for i, q in enumerate(paths):
        store.put_bytes(q, str(i).encode())
    listed = set(store.list(P.root + "/"))
    for i, q in enumerate(paths):
        assert q in listed, f"{q} not listed"
        assert store.get_bytes(q) == str(i).encode()


CHECKS: dict[str, Callable[[Store], None]] = {
    "whole_file": check_whole_file,
    "read_after_write": check_read_after_write,
    "list_prefix": check_list_prefix,
    "path_safety": check_path_safety,
    "job_paths": check_job_paths_round_trip,
}


# ----------------------------------------------------------------------------------------------
# Deliberately broken stores (each must fail the check it is keyed under)
# ----------------------------------------------------------------------------------------------


class DictStore:
    """A correct in-memory Store, the base of the broken variants below."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def _p(self, path: str) -> str:
        return check_rel_path(path)

    def put_bytes(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes | bytearray):
            raise TypeError("bytes only")
        self.data[self._p(path)] = bytes(data)

    def get_bytes(self, path: str) -> bytes:
        try:
            return self.data[self._p(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    def exists(self, path: str) -> bool:
        return self._p(path) in self.data

    def list(self, prefix: str) -> Iterator[str]:
        return iter(sorted(p for p in self.data if p.startswith(prefix)))


class AppendingStore(DictStore):
    def put_bytes(self, path: str, data: bytes) -> None:
        super().put_bytes(path, self.data.get(path, b"") + data)


class MissingIsEmptyStore(DictStore):
    def get_bytes(self, path: str) -> bytes:
        return self.data.get(self._p(path), b"")


class DirectoryListStore(DictStore):
    """Treats the prefix as a directory (drops the partial last component)."""

    def list(self, prefix: str) -> Iterator[str]:
        base = prefix if prefix.endswith("/") else prefix.rsplit("/", 1)[0] + "/"
        return iter(sorted(p for p in self.data if p.startswith(base)))


class UncheckedPathStore(DictStore):
    def _p(self, path: str) -> str:
        return path


class ShallowListStore(DictStore):
    """Lists one directory level only (forgets nested files)."""

    def list(self, prefix: str) -> Iterator[str]:
        return iter(sorted(p for p in self.data
                           if p.startswith(prefix) and "/" not in p[len(prefix):]))


BROKEN_STORES: dict[str, list[tuple[str, Callable[[], Store]]]] = {
    "whole_file": [("appending", AppendingStore)],
    "read_after_write": [("missing-is-empty", MissingIsEmptyStore)],
    "list_prefix": [("directory-list", DirectoryListStore)],
    "path_safety": [("unchecked-paths", UncheckedPathStore)],
    "job_paths": [("shallow-list", ShallowListStore)],
}
