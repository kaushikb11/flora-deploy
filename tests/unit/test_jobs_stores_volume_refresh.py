"""C3: on a Volume store the orchestrator must see other containers' files.

``MountedVolume`` models Modal Volume v2 per container: other containers' commits land in the
volume (``server``); this container's mount shows them only after ``reload()``, and this
container's writes reach the volume only on ``commit()``."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

from flora_deploy.jobs.orchestrator import DictLease, Orchestrator
from flora_deploy.jobs.settings import PoolSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores import LocalStore, VolumeStore
from flora_deploy.modal_app.orchestrator_app import TickDriver
from tests.fakes.jobs import (
    CFG,
    JOB,
    FakeAttempts,
    FakeClock,
    FakeDict,
    FakeManifests,
    FakeProgress,
    WorkerBehavior,
    make_manifest,
    make_settings,
    make_worker,
)
from tests.fakes.modal import FakeSpawner


class MountedVolume:
    def __init__(self, server: Path, mount: Path):
        self.server, self.mount = server, mount
        server.mkdir(parents=True, exist_ok=True)
        mount.mkdir(parents=True, exist_ok=True)
        self.reloads = self.commits = 0

    def commit(self):
        shutil.copytree(self.mount, self.server, dirs_exist_ok=True)
        self.commits += 1

    def reload(self):
        shutil.copytree(self.server, self.mount, dirs_exist_ok=True)
        self.reloads += 1


class OtherContainer(LocalStore):
    """A worker container whose writes are already committed to the volume."""

    def tick(self):
        pass


def _vol(tmp_path):
    vol = MountedVolume(tmp_path / "server", tmp_path / "mount")
    return vol, VolumeStore(vol.mount, vol, tmp_dir=tmp_path)


def test_new_files_hidden_until_refresh(tmp_path):
    vol, store = _vol(tmp_path)
    worker = OtherContainer(vol.server)
    receipt = JobPaths(JOB).receipt(CFG, "segcap", "s0")
    worker.put_bytes(receipt, b"{}")
    assert not store.exists(receipt)  # the trap: exists() lags across containers too
    assert list(store.list(f"flora/{JOB}/")) == []
    assert store.refresh() is True and vol.reloads == 1
    assert store.exists(receipt) and list(store.list(f"flora/{JOB}/")) == [receipt]


def test_refresh_commits_own_writes_first_and_skips_while_busy(tmp_path):
    vol, store = _vol(tmp_path)
    store.commit_every = 10
    store.put_bytes("own/a", b"1")
    assert store.pending_writes == 1
    assert store.refresh() and vol.commits == 1 and (vol.server / "own/a").exists()
    # another thread is inside a store call: refresh must not reload under it
    entered, release = threading.Event(), threading.Event()
    real_get = store._abs

    def slow_abs(path):
        entered.set()
        release.wait(5)
        return real_get(path)

    store._abs = slow_abs  # type: ignore[method-assign]
    t = threading.Thread(target=store.get_bytes, args=("own/a",))
    t.start()
    entered.wait(5)
    assert store.refresh() is False and store.skipped_refreshes == 1
    release.set()
    t.join(5)
    store._abs = real_get  # type: ignore[method-assign]
    assert store.refresh() is True


def test_listing_is_materialized_per_call(tmp_path):
    vol, store = _vol(tmp_path)
    store.put_bytes("x/1", b"1")
    it = store.list("x/")
    OtherContainer(vol.server).put_bytes("x/2", b"2")
    store.refresh()
    assert list(it) == ["x/1"]  # an old listing never mixes two views
    assert sorted(store.list("x/")) == ["x/1", "x/2"]


def _orchestrator(tmp_path, *, behavior):
    vol, store = _vol(tmp_path)
    worker_store = OtherContainer(vol.server)
    pools = {"segcap": PoolSettings(gpu="H200", max_containers=1)}
    manifests = FakeManifests(make_manifest({"s0": ["a"]}))
    spawner = FakeSpawner(worker_store,  # type: ignore[arg-type]
                          {"segcap": make_worker(manifests, behavior, "segcap")},
                          python_retries=0, max_restarts=50)
    clock = FakeClock()
    orch = Orchestrator(settings=make_settings(pools), store=store, spawner=spawner,
                        lease=DictLease(FakeDict(), f"orchestrator/{JOB}", clock=clock),
                        manifests=manifests, attempts=FakeAttempts(store),  # type: ignore
                        progress=FakeProgress(store), clock=clock)  # type: ignore[arg-type]

    def sleep(dt):
        clock.advance(dt)
        spawner.tick()

    return orch, sleep


def test_orchestrator_on_volume_sees_worker_attempt_markers_and_quarantines(tmp_path):
    """An OOM poison pill on a Volume store: without per-tick refresh the orchestrator never
    sees the workers' attempt markers and the GPU crash-loops forever; with TickDriver it
    quarantines after k starts."""
    blind, sleep = _orchestrator(tmp_path / "blind", behavior=WorkerBehavior(oom={"a"}))
    out = blind.run_until_idle(sleep=sleep, poll_interval_s=10.0, max_ticks=30)
    assert out.result == "max_ticks"  # still "running" a crash loop after 30 ticks

    orch, sleep = _orchestrator(tmp_path / "seen", behavior=WorkerBehavior(oom={"a"}))
    out = TickDriver(orch).run_until_idle(sleep=sleep, poll_interval_s=10.0, max_ticks=30)
    assert out.result == "done" and out.ticks < 10  # poisoned single episode, job finishes
    assert orch.store.reloads >= out.ticks  # type: ignore[attr-defined]
