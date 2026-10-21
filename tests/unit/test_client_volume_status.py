"""``flora status`` over a Modal Volume from a laptop (``jobs.stores.ClientVolumeStore``): a
bounded number of Volume calls (H0: one listing per shard file made it slow), and the results /
SDK readers work through it."""

from __future__ import annotations

from pathlib import Path

import yaml

from flora_deploy import sdk
from flora_deploy.export.results import iter_records
from flora_deploy.jobs.stores import ClientVolumeStore
from flora_deploy.local.runner import run_local
from flora_deploy.local.sample import sample_dataset
from tests.fakes.client_volume import FakeClientVolume

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
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
