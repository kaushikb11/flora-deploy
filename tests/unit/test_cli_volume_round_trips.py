"""Volume round trips per first-run command over a Modal Volume through the client
(``jobs.stores.ClientVolumeStore``): ``flora status``, ``flora report`` and ``flora compose``.

On a 15-shard job these took 20-32 s, ~70 s and ~70 s, almost all of it serial round trips:
one listing per area after the status refresh (~20), and one ``batch_upload`` commit per
compose output file (9). The bounds below are per command, for the job's N shards: one listing,
one upload batch for the outputs plus one for the pointer, each file read at most once, and only
the run spec and manifest read one at a time (everything per shard goes through the parallel
read-ahead)."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter

import pytest
from typer.testing import CliRunner

from flora_deploy import sdk
from flora_deploy.export.compose import compose
from flora_deploy.jobs.store import put_many
from flora_deploy.jobs.stores import ClientVolumeStore, LocalStore, VolumeStore
from flora_deploy.jobs.stores.client_volume import PREFETCH_WORKERS
from tests.fakes.client_volume import FakeClientVolume
from tests.unit.test_client_volume_status import JOB, _Backend, _job_on_volume

STORE_URI = "modal-volume://flora-results/s"


class _CountingVolume(FakeClientVolume):
    """Every Volume call with the thread it ran on (the main thread's calls are serial)."""

    def __init__(self, files):
        super().__init__()
        self.files.update(files)
        self.log: list[tuple[str, str, str]] = []
        self._mu = threading.Lock()

    def _rec(self, op, path):
        with self._mu:
            self.log.append((op, path, threading.current_thread().name))

    def read_file(self, path):
        self._rec("read_file", path)
        yield from super().read_file(path)

    def listdir(self, path, recursive=False):
        self._rec("listdir", path)
        return super().listdir(path, recursive)

    def batch_upload(self, force=False):
        self._rec("batch_upload", "")
        return super().batch_upload(force)

    def count(self, op=None, *, serial=False):
        return sum(1 for o, _p, t in self.log
                   if (op is None or o == op) and (not serial or t == "MainThread"))


@pytest.fixture(scope="module")
def job_files(tmp_path_factory):
    return _job_on_volume(tmp_path_factory.mktemp("rt"), episodes=12).files


def _shards(files) -> int:
    st = ClientVolumeStore(FakeClientVolume(), "/s")
    st.volume.files.update(files)
    return len(sdk.open_job_in(st, JOB, store_uri=STORE_URI)._deps().manifests.load().shards)


def _cli(files, *args):
    """Run ``flora <args>`` against the job on a fake Volume; returns the Volume."""
    from flora_deploy.cli import app

    vol = _CountingVolume(files)
    sdk.set_backend(_Backend(ClientVolumeStore(vol, "/s")))
    try:
        res = CliRunner().invoke(app, [*args, JOB, "--store", STORE_URI])
    finally:
        sdk.set_backend(None)
    assert res.exit_code == 0, res.output
    return vol


def _check_reads(vol, shards, op):
    reads = [p for o, p, _t in vol.log if o == "read_file"]
    assert len(reads) == len(set(reads)), f"{op}: a file was read twice"
    # serial: the run spec, then the manifest (it names the stage hashes the rest lives under)
    assert vol.count("read_file", serial=True) <= 2, (op, vol.log)
    # parallel: the read-ahead covers the per-shard files in ceil(n / workers) waves
    waves = math.ceil((len(reads) - 2) / PREFETCH_WORKERS)
    assert waves <= math.ceil(4 * shards / PREFETCH_WORKERS), (op, len(reads), shards)


@pytest.mark.parametrize("cmd", ["status", "report"])
def test_cli_status_and_report_round_trips(job_files, cmd):
    shards = _shards(job_files)
    vol = _cli(job_files, cmd)
    # one recursive listing of the job directory answers every exists / list (it was one per
    # area: calls, receipts, replans, control kinds, errors, containers, attempts: ~20)
    assert vol.count("listdir") == 1, [x for x in vol.log if x[0] == "listdir"]
    assert vol.count("batch_upload") == 0  # read-only
    _check_reads(vol, shards, cmd)
    # serial round trips: 1 listing + 2 reads (plus the parallel read-ahead)
    assert vol.count(serial=True) <= 3, vol.log


def test_compose_round_trips(job_files):
    """``flora compose``: its 9 output files in one upload (one commit), then the pointer."""
    shards = _shards(job_files)
    vol = _CountingVolume(job_files)
    st = ClientVolumeStore(vol, "/s")
    sdk.check_results_namespace(st, JOB)  # as `flora compose` does first
    res = compose(st, JOB)
    assert len(res.summary["outputs"]) >= 4
    assert vol.count("batch_upload") == 2, vol.log
    assert vol.count("listdir") == 1, vol.log
    _check_reads(vol, shards, "compose")
    assert vol.count(serial=True) <= 5, vol.log  # listing, 2 reads, 2 uploads
    # the pointer is uploaded after the generation it names
    assert vol.files["s/" + res.paths["summary"]]
    assert all("s/" + p in vol.files for p in res.summary["files"].values())


def test_round_trips_do_not_grow_with_shards(tmp_path):
    """Doubling the job adds reads to the parallel batch, never a serial round trip."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    small = _job_on_volume(tmp_path / "a", episodes=6).files
    big = _job_on_volume(tmp_path / "b", episodes=12).files
    assert _shards(big) > _shards(small)
    for cmd in ("status", "report"):
        a, b = _cli(small, cmd), _cli(big, cmd)
        assert a.count(serial=True) == b.count(serial=True), cmd
        assert a.count("listdir") == b.count("listdir") == 1


def _status_job(files):
    vol = _CountingVolume(files)
    st = ClientVolumeStore(vol, "/s")
    return vol, sdk.open_job_in(st, JOB, store_uri=STORE_URI)


def test_later_statuses_refresh(job_files):
    """Only a status right after opening reuses the opening's listing; every later one (each
    ``wait`` poll) and one long after opening see other processes' writes."""
    vol, job = _status_job(job_files)
    job.status()
    n = vol.count("listdir")
    job.status()
    assert vol.count("listdir") == n + 1  # refreshed: one new listing, still not one per area
    vol, job = _status_job(job_files)
    job._opened_at = time.monotonic() - 2 * sdk.FRESH_VIEW_S
    before = vol.count("listdir")
    job.status()
    assert vol.count("listdir") == before + 1


def test_a_later_status_sees_another_process_write(job_files):
    uncomposed = {k: v for k, v in job_files.items() if "/output/" not in k}
    vol, job = _status_job(uncomposed)
    assert not job.status().composed
    compose(ClientVolumeStore(vol, "/s"), JOB)  # another process, same Volume
    assert job.status().composed


def test_client_put_many_is_one_upload():
    vol = FakeClientVolume()
    st = ClientVolumeStore(vol, "r")
    assert not st.exists("a/x.json")  # a cached listing that the write must drop
    st.put_many({"a/x.json": b"1", "a/y.json": b"2", "b/z.json": b"3"})
    assert vol.calls["batch_upload"] == 1
    assert {k: vol.files[k] for k in ("r/a/x.json", "r/a/y.json", "r/b/z.json")} == {
        "r/a/x.json": b"1", "r/a/y.json": b"2", "r/b/z.json": b"3"}
    assert st.exists("a/x.json") and st.get_bytes("b/z.json") == b"3"
    st.put_many({})
    assert vol.calls["batch_upload"] == 1


class _CommitCounter:
    def __init__(self):
        self.calls = Counter()

    def commit(self):
        self.calls["commit"] += 1

    def reload(self):
        self.calls["reload"] += 1


def test_mounted_put_many_commits_once(tmp_path):
    vol = _CommitCounter()
    st = VolumeStore(tmp_path, vol)
    st.put_many({"o/a.parquet": b"a", "o/b.parquet": b"b", "o/c.jsonl": b"c"})
    assert vol.calls == Counter({"commit": 1})
    assert st.get_bytes("o/b.parquet") == b"b" and st.pending_writes == 0
    with pytest.raises(TypeError):
        st.put_many({"o/d": "text"})  # type: ignore[dict-item]


def test_put_many_falls_back_to_put_bytes(tmp_path):
    st = LocalStore(tmp_path)
    put_many(st, {"a/1": b"x", "a/2": b"y"})
    assert sorted(st.list("a/")) == ["a/1", "a/2"]
