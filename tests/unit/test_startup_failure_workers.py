"""Worker side of the startup-failure markers (``jobs.attempts.record_startup_failure``): a job
worker whose ``@modal.enter`` raises records why before re-raising, so the orchestrator's
watchdog sees it (no shard ever starts there, so no attempt marker would)."""

from __future__ import annotations

import pytest

from flora_deploy.jobs.attempts import read_startup_failures
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.modal_app import workers_app as W

APP = "flora-demo-abc"


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    env = {"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": str(tmp_path / "store"),
           "FLORA_APP_NAME": APP, "FLORA_SHARD_KINDS": "caption", "FLORA_CONFIG_HASH": "h",
           "MODAL_TASK_ID": "ta-1"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return LocalStore(env["FLORA_STORE_ROOT"])


def failures(store):
    return read_startup_failures(store, APP)


def test_record_start_failure_writes_a_marker_for_job_workers(job_env):
    assert W.record_start_failure(RuntimeError("no GPU")) is not None
    (f,) = failures(job_env)
    assert f.kinds == ("caption",) and "no GPU" in f.error


def test_not_a_job_worker_records_nothing(monkeypatch):
    for k in ("FLORA_STORE_KIND", "FLORA_STORE"):
        monkeypatch.delenv(k, raising=False)
    assert W.record_start_failure(RuntimeError("x")) is None


def test_startup_guard_records_and_reraises(job_env):
    class Worker:
        @W.startup_guard
        def start(self):
            raise ValueError("engine build failed")

    with pytest.raises(ValueError, match="engine build failed"):
        Worker().start()
    assert len(failures(job_env)) == 1


def test_reward_workers_are_guarded():
    for cls in (W.RewardWorker, W.RubricWorker, W.FakeSegcapWorker):
        raw = cls.__dict__["start"]._get_raw_f()  # the @modal.enter body
        assert raw.__wrapped__  # functools.wraps of startup_guard


def test_balanced_start_records_then_reraises(job_env, monkeypatch):
    from flora_deploy.modal_app import balanced_app as B

    def boom(*a, **k):
        raise RuntimeError("vLLM build failed")

    monkeypatch.setattr(B, "build_runner", boom)
    with pytest.raises(RuntimeError, match="vLLM build failed"):
        B._start("caption")
    assert "vLLM build failed" in failures(job_env)[0].error


def test_reference_worker_load_records_then_reraises(job_env, monkeypatch):
    from flora_deploy.modal_app import reference_app as R

    w = R.ReferenceWorker.__new__(R.ReferenceWorker)
    monkeypatch.setattr(R.ReferenceWorker, "_load",
                        lambda self: (_ for _ in ()).throw(OSError("weights missing")))
    with pytest.raises(OSError):
        w.load()
    assert "weights missing" in failures(job_env)[0].error


def test_rewarder_start_records_then_reraises(job_env, monkeypatch):
    from flora_deploy.modal_app import rewarder_app as RW

    cls = RW.Rewarder._get_user_cls()
    monkeypatch.setattr(cls, "_start", lambda self: (_ for _ in ()).throw(RuntimeError("oom")))
    with pytest.raises(RuntimeError):
        cls.__dict__["start"]._get_raw_f()(cls.__new__(cls))
    assert "oom" in failures(job_env)[0].error
