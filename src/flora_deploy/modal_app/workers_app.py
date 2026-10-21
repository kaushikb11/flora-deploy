"""Shard worker classes for the job system (PLAN.md §7/§8): module-scope, *undecorated* by
``app.cls``. The app factory (P1a-6a, ``build_app(cfg)``) decorates them from ``RunSettings``
pools (GPU, containers, ``@modal.concurrent``, cpu/memory, timeouts, retries, Volumes, secrets);
this module only defines the lifecycle hooks and ``run_shard``. Importing it creates lazy Modal
objects only. Never run from tests.

``RewardWorker`` (P1a-5r) wraps :class:`flora_deploy.workers_impl.reward.RewardShardRunner`:

- ``@modal.enter``: :func:`build_reward_engine`, the ``flora-rewarder`` startup recipe (W3-B,
  ``docs/cold-start.md``) reusing ``rewarder_app``'s own pieces: weights prefetch first,
  compile/Triton cache env into ``/cache``, NVIDIA's Rewarder family + processor, the engine from
  ``stages.reward.build_engine`` (NVIDIA's ``load_llm`` with the terminal script's arguments,
  defined once there), then a cache commit. The factory must mount ``rewarder_app.weights`` at
  ``WEIGHTS_MOUNT`` and ``rewarder_app.cache`` at ``CACHE_MOUNT`` (:data:`REWARD_VOLUMES`) and
  should start from :data:`REWARD_RESOURCES`.
- ``run_shard(job_id, shard_id, config_hash, token="")``: the ``workers.ShardWorker`` contract.
- ``@modal.exit``: :meth:`RewardShardRunner.preempt` (flush + ``mark_preempted``; a no-op when no
  shard is in flight, e.g. a normal scale-down).

Store and worker settings come from the container env (the factory sets them):
``FLORA_STORE_KIND`` (``local`` | ``volume``), ``FLORA_STORE_ROOT`` (path / mount),
``FLORA_STORE_VOLUME`` (Volume name, for commits), ``FLORA_MEDIA_ROOT``,
``FLORA_REWARD_TERMINAL_MODE``, ``FLORA_PRESET``, ``FLORA_ENVELOPE``.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import modal

from flora_deploy.jobs.store import Store
from flora_deploy.layout import CACHE_MOUNT, WEIGHTS_MOUNT
from flora_deploy.modal_app import rewarder_app as ra
from flora_deploy.workers_impl.reward import (
    RewardEngine,
    RewardShardRunner,
    RewardWorkerSettings,
)

REWARD_VOLUMES = {str(WEIGHTS_MOUNT): ra.weights, str(CACHE_MOUNT): ra.cache}
REWARD_RESOURCES: dict[str, Any] = {
    "gpu": ra.GPU, "cpu": ra.CPU, "memory": ra.MEMORY_MIB, "env": {ra.IMAGE_FP_ENV: ra.IMAGE_FP},
}


def build_reward_engine(snapshot: Path | str = ra.SNAPSHOT, *,
                        commit: Callable[[], None] | None = None
                        ) -> tuple[RewardEngine, Any, dict[str, Any]]:
    """``rewarder_app.Rewarder.start``'s recipe, returning ``(engine, nv, startup facts)``."""
    t0 = time.perf_counter()
    from flora_deploy.modal_app import warm

    prefetch = warm.start_prefetch(str(snapshot), threads=ra.PREFETCH_THREADS)  # t=0
    env = ra.cache_env()
    os.environ.update(env)  # before vLLM is imported; the EngineCore process inherits it
    for p in env.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family
    from flora_deploy.stages import reward

    nv = load_family(Family.REWARDER, str(snapshot))
    processor = nv.engine.load_processor(str(snapshot))
    t1 = time.perf_counter()
    llm, spec = reward.build_engine(nv, str(snapshot))
    t2 = time.perf_counter()
    stats = prefetch.wait(timeout=0)
    (commit or ra.cache.commit)()
    startup = {"cache_key": ra.cache_key(), "link_and_processor_s": t1 - t0,
               "engine_build_s": t2 - t1, "enter_total_s": time.perf_counter() - t0,
               "prefetch": stats.as_json() if stats else {"done": prefetch.done},
               "engine": ra._engine_facts(llm)}
    return RewardEngine(llm, processor, spec), nv, startup


def store_from_env(env: Mapping[str, str] = os.environ) -> Store:
    kind = env.get("FLORA_STORE_KIND", "volume")
    root = env.get("FLORA_STORE_ROOT")
    if not root:
        raise RuntimeError("FLORA_STORE_ROOT is not set (the app factory sets the store)")
    if kind == "local":
        from flora_deploy.jobs.stores import LocalStore

        return LocalStore(root)
    if kind == "volume":
        from flora_deploy.jobs.stores import VolumeStore

        name = env.get("FLORA_STORE_VOLUME")
        return VolumeStore(root, modal.Volume.from_name(name, version=2) if name else None)
    raise RuntimeError(f"unsupported FLORA_STORE_KIND {kind!r} (TODO P1a-6a: bucket)")


def gpu_name() -> str | None:
    """From ``nvidia-smi`` (D11: provenance records the GPU, outside the config hash)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def reward_settings_from_env(env: Mapping[str, str] = os.environ, *,
                             gpu: str | None = None) -> RewardWorkerSettings:
    from flora_deploy.config.pins import PINS_BY_KEY

    return RewardWorkerSettings(
        terminal_mode=env.get("FLORA_REWARD_TERMINAL_MODE", "terminal"),
        preset=env.get("FLORA_PRESET", "exact"),
        models={"rewarder": PINS_BY_KEY["rewarder"].revision},
        gpu=gpu, config_extra={"image": ra.IMAGE_FP},
        envelope=env.get("FLORA_ENVELOPE", "warn"),
        media_root=env.get("FLORA_MEDIA_ROOT"),
    )


class RewardWorker:
    """``workers.ShardWorker`` for ``reward.terminal`` / ``reward.dense`` shards."""

    @modal.enter()
    def start(self) -> None:
        engine, nv, self.startup = build_reward_engine()
        self.runner = RewardShardRunner(
            store_from_env(), engine, nv, reward_settings_from_env(gpu=gpu_name()),
            call_id=modal.current_function_call_id)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        return self.runner.run_shard(job_id, shard_id, config_hash, token=token)

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()
