"""``workers_app.RewardWorker``: imports lazily, decorates, wires the core (no Modal run)."""

from __future__ import annotations

from types import SimpleNamespace

import modal

from flora_deploy.jobs.stores import LocalStore
from flora_deploy.modal_app import rewarder_app as ra
from flora_deploy.modal_app import workers_app as wa
from flora_deploy.workers_impl.reward import RewardShardRunner


def test_reward_worker_is_module_scope_and_decoratable():
    assert wa.RewardWorker.__module__ == "flora_deploy.modal_app.workers_app"
    assert {"start", "run_shard", "stop"} <= set(vars(wa.RewardWorker))
    cls = modal.App("flora-test").cls(**{k: v for k, v in wa.REWARD_RESOURCES.items()},
                                      volumes=wa.REWARD_VOLUMES)(wa.RewardWorker)
    assert cls is not None
    assert wa.REWARD_RESOURCES["gpu"] == ra.GPU and wa.REWARD_RESOURCES["memory"] >= 72 * 1024


def test_store_and_settings_from_env(tmp_path):
    store = wa.store_from_env({"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": str(tmp_path)})
    assert isinstance(store, LocalStore)
    s = wa.reward_settings_from_env({"FLORA_REWARD_TERMINAL_MODE": "terminal_rationale",
                                     "FLORA_MEDIA_ROOT": "/data"}, gpu="NVIDIA H100")
    assert s.terminal_mode == "terminal_rationale" and s.media_root == "/data"
    assert s.preset == "exact" and s.gpu == "NVIDIA H100" and s.models["rewarder"]
    assert s.config_extra == {"image": ra.IMAGE_FP}


def test_wrapper_delegates_to_runner_and_exit_hook():
    calls = []
    runner = SimpleNamespace(run_shard=lambda *a, **k: calls.append((a, k)) or "R",
                             preempt=lambda: calls.append("preempt"))
    worker = wa.RewardWorker.__new__(wa.RewardWorker)
    worker.runner = runner
    run_shard = wa.RewardWorker.__dict__["run_shard"]
    stop = wa.RewardWorker.__dict__["stop"]
    assert run_shard._get_raw_f()(worker, "j", "s", "c", token="t") == "R"  # modal 1.6 API
    stop._get_raw_f()(worker)
    assert calls == [(("j", "s", "c"), {"token": "t"}), "preempt"]
    assert RewardShardRunner.run_shard.__code__.co_varnames[:5] == (
        "self", "job_id", "shard_id", "config_hash", "token")
