"""The ``modal.Volume`` client API subset ``jobs.stores.ClientVolumeStore`` uses (modal 1.6.2:
``read_file`` yields chunks, ``listdir(recursive=True)`` returns ``FileEntry``s with ``path``
relative to the Volume root and ``type`` 1 for files, a missing path raises
``NotFoundError``; ``batch_upload(force=True)`` uploads whole files). Counts every call."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace


class NotFoundError(Exception):  # modal.exception.NotFoundError (matched by name)
    pass


class FakeClientVolume:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: Counter[str] = Counter()

    class _Batch:
        def __init__(self, vol):
            self.vol, self.staged = vol, {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, *a):
            if exc_type is None:
                self.vol.files.update(self.staged)
            return False

        def put_file(self, f, path):
            self.staged[path.lstrip("/")] = f.read()

    def batch_upload(self, force=False):
        assert force, "the store must overwrite whole files"
        self.calls["batch_upload"] += 1
        return self._Batch(self)

    def read_file(self, path):
        self.calls["read_file"] += 1
        p = path.lstrip("/")
        if p not in self.files:
            raise NotFoundError(p)
        data = self.files[p]
        yield data[: len(data) // 2]
        yield data[len(data) // 2:]

    def listdir(self, path, recursive=False):
        self.calls["listdir"] += 1
        p = path.strip("/")
        hits = sorted(k for k in self.files if not p or k.startswith(p + "/"))
        if not hits and p:
            raise NotFoundError(p)
        if not recursive:
            raise AssertionError("ClientVolumeStore lists recursively")
        dirs = {k.rsplit("/", 1)[0] for k in hits if "/" in k}
        return ([SimpleNamespace(path=k, type=1) for k in hits]
                + [SimpleNamespace(path=d, type=2) for d in sorted(dirs)])
