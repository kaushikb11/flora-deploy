"""``flora status`` over a Modal Volume from a laptop (``jobs.stores.ClientVolumeStore``): a
bounded number of Volume calls (H0: one listing per shard file made it slow), and the results /
SDK readers work through it."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import yaml

from flora_deploy import sdk
from flora_deploy.export.results import iter_records
from flora_deploy.jobs.stores import ClientVolumeStore
from flora_deploy.local.runner import run_local
from flora_deploy.local.sample import sample_dataset
from tests.fakes.client_volume import FakeClientVolume

TEMPLATE = Path(__file__).resolve().parents[2] / "tests/fixtures/local_demo_exact.yaml"
JOB = "local-demo.fake"  # local fake-models runs get their own job id (jobs.store.fake_job_id)


def _job_on_volume(tmp_path, episodes: int):
    d = tmp_path / "demo"
    d.mkdir()
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["modal"]["budget_usd"] = 500
    cfg["pipeline"]["shard_target_gpu_minutes"] = 0.02  # many small shards
    (d / "flora.yaml").write_text(yaml.safe_dump(cfg))
    sample_dataset(d / "videos", episodes=episodes, min_s=3.0, max_s=4.0)
    res = run_local(d / "flora.yaml", store=tmp_path / "store")
    assert res.state == "done", res.as_dict()
    vol = FakeClientVolume()
    for f in (tmp_path / "store").rglob("*"):
        if f.is_file() and ".flora-local" not in f.parts:
            vol.files["s/" + f.relative_to(tmp_path / "store").as_posix()] = f.read_bytes()
    return vol


class _Backend:
    def __init__(self, store):
        self._store = store

    def store(self, uri, environment=None):
        return self._store


def test_status_calls_are_bounded(tmp_path):
    vol = _job_on_volume(tmp_path, episodes=12)
    st = ClientVolumeStore(vol, "/s")
    job = sdk.open_job(JOB, store="modal-volume://flora-results/s", backend=_Backend(st))
    shards = len(job._deps().manifests.load().shards)
    reads: list[str] = []
    orig = vol.read_file

    def read_file(path):
        reads.append(path)
        return orig(path)

    vol.read_file = read_file
    vol.calls.clear()
    v = job.status()
    assert v.state == "done"
    # listings: per (area, kind) directory, never per shard file
    assert vol.calls["listdir"] <= 25, dict(vol.calls)
    # reads: each file at most once per status (call ledger + receipt per shard, a few more)
    assert len(reads) == len(set(reads)), "a file was read twice in one status"
    assert len(reads) <= 2 * shards + 10, (len(reads), shards)
    # the next status sees new writes (refresh at the start of every status)
    vol.calls.clear()
    job.status()
    assert vol.calls["listdir"] > 0
    # results through the same store
    recs = list(iter_records(st, JOB, "caption"))
    assert len(recs) == 12


def test_writes_invalidate_the_snapshot(tmp_path):
    st = ClientVolumeStore(FakeClientVolume(), "root")
    st.put_bytes("flora/j/h/receipts/k/aa/bb/s1.json", b"1")
    assert st.exists("flora/j/h/receipts/k/aa/bb/s1.json")
    assert not st.exists("flora/j/h/receipts/k/aa/bb/s2.json")
    st.put_bytes("flora/j/h/receipts/k/aa/bb/s2.json", b"2")
    assert st.exists("flora/j/h/receipts/k/aa/bb/s2.json")
    assert st.get_bytes("flora/j/h/receipts/k/aa/bb/s2.json") == b"2"
    st.put_bytes("flora/j/h/receipts/k/aa/bb/s2.json", b"3")
    assert st.get_bytes("flora/j/h/receipts/k/aa/bb/s2.json") == b"3"
    assert st.exists("flora/j/h/receipts/k/aa/bb/s1.json")  # a listing is cached now
    other = st.volume  # another process writes: seen after refresh
    other.files["root/flora/j/h/receipts/k/aa/bb/s3.json"] = b"x"
    assert not st.exists("flora/j/h/receipts/k/aa/bb/s3.json")
    st.refresh()
    assert st.exists("flora/j/h/receipts/k/aa/bb/s3.json")
    assert list(st.list("flora/j/h/receipts/k/")) == [
        f"flora/j/h/receipts/k/aa/bb/s{i}.json" for i in (1, 2, 3)]


class _TimedVolume(FakeClientVolume):
    """Records every ``read_file`` (path, thread) and the most reads in flight at once."""

    def __init__(self, files=None, delay_s: float = 0.0):
        super().__init__()
        self.files.update(files or {})
        self.reads: list[tuple[str, str]] = []
        self.delay_s, self.in_flight, self.max_in_flight = delay_s, 0, 0
        self._mu = threading.Lock()

    def read_file(self, path):
        with self._mu:
            self.reads.append((path, threading.current_thread().name))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay_s)
            yield from super().read_file(path)
        finally:
            with self._mu:
                self.in_flight -= 1


def test_prefetch_reads_each_file_once_in_parallel():
    vol = _TimedVolume({f"r/a/f{i:03d}.json": b"x" * 10 for i in range(40)}, delay_s=0.01)
    st = ClientVolumeStore(vol, "r")
    paths = list(st.list("a/"))
    assert st.prefetch([*paths, "a/missing.json"]) == 40
    assert vol.calls["read_file"] == 41 and vol.max_in_flight > 1
    assert {t for _p, t in vol.reads} != {"MainThread"}
    vol.calls.clear()
    assert [st.get_bytes(p) for p in paths] == [b"x" * 10] * 40
    assert vol.calls["read_file"] == 0  # served from the cache
    assert st.prefetch(paths) == 0  # nothing left to read


def test_status_report_and_compose_read_their_shard_files_in_parallel(tmp_path):
    src = _job_on_volume(tmp_path, episodes=12)
    for op in ("status", "report", "compose"):
        vol = _TimedVolume(src.files, delay_s=0.002)
        st = ClientVolumeStore(vol, "/s")
        job = sdk.open_job(JOB, store="modal-volume://flora-results/s", backend=_Backend(st))
        shards = len(job._deps().manifests.load().shards)
        vol.reads.clear()
        getattr(job, op)()
        paths = [p for p, _t in vol.reads]
        assert len(paths) == len(set(paths)), f"{op}: a file was read twice"
        shard_reads = [t for p, t in vol.reads if "/manifest" not in p and "summary" not in p]
        on_pool = sum(t.startswith("flora-prefetch") for t in shard_reads)
        # every per-shard read (ledger, receipt, marker, part) went through the parallel prefetch
        assert on_pool == len(shard_reads) >= shards, (op, on_pool, len(shard_reads), shards)
        assert vol.max_in_flight > 1, op


def test_submit_reads_the_index_receipts_once_in_parallel(tmp_path):
    """``flora run --detach`` plans from every index receipt (one per episode): read ahead in
    parallel, each once (the fake-modal rehearsal's submit took 5.5 min, one round trip at a
    time)."""
    from tests.fakes.flora_backend import RECEIPTS, DeployedApp

    app = DeployedApp(tmp_path, run=False)
    root = app.store.root
    vol = _TimedVolume({"s/" + f.relative_to(root).as_posix(): f.read_bytes()
                        for f in root.rglob("*") if f.is_file()})
    st = ClientVolumeStore(vol, "/s")
    app.backend.store = lambda uri, environment=None: st  # type: ignore[method-assign]
    sdk.submit(app.config_path, receipts=RECEIPTS, backend=app.backend)
    index = [(p, t) for p, t in vol.reads
             if "/index/" in p and not p.endswith("summary.json")]
    episodes = len([p for p in vol.files if "/index/" in p and not p.endswith("summary.json")])
    assert episodes > 1 and len(index) == episodes  # each receipt read once
    assert all(t.startswith("flora-prefetch") for _p, t in index)
    assert vol.calls["listdir"] <= 10, dict(vol.calls)
