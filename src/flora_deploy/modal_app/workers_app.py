"""Shard worker classes for the job system (PLAN.md §7/§8): module-scope, *undecorated* by
``app.cls``. The app factory (P1a-6a, ``build_app(cfg)``) decorates them from ``RunSettings``
pools (GPU, containers, ``@modal.concurrent``, cpu/memory, timeouts, retries, Volumes, secrets);
this module only defines the lifecycle hooks and ``run_shard``. Importing it creates lazy Modal
objects only. Never run from tests.

``RewardWorker`` (P1a-5r) wraps :class:`flora_deploy.workers_impl.reward.RewardShardRunner`:

- ``@modal.enter``: first a cheap :func:`preflight` (H3/C1: the job store opens and answers a
  read, the media root exists and is readable, a fixed instruction is set when configured), so a
  misconfigured deployment fails in seconds instead of after the ~4 min GPU engine build; then
  :func:`build_reward_engine`, the ``flora-rewarder`` startup recipe (W3-B,
  ``docs/cold-start.md``) reusing ``rewarder_app``'s own pieces: weights prefetch first,
  compile/Triton cache env, NVIDIA's Rewarder family + processor, the engine from
  ``stages.reward.build_engine`` (NVIDIA's ``load_llm`` with the terminal script's arguments,
  defined once there). The factory mounts ``rewarder_app.weights`` at ``WEIGHTS_MOUNT`` and
  ``rewarder_app.cache`` at ``CACHE_MOUNT`` **read-only** on workers (M7: concurrent writers
  of one Volume are last-write-wins): ``FLORA_CACHE_MODE=ro`` copies the cache into
  container-local disk and never commits. Only :func:`warm_reward_cache` (one container, cache
  mounted read-write, ``FLORA_CACHE_MODE=rw``) writes and commits it.
- ``run_shard(job_id, shard_id, config_hash, token="")``: the ``workers.ShardWorker`` contract.
- ``@modal.exit``: :meth:`RewardShardRunner.preempt` (flush + ``mark_preempted``; a no-op when no
  shard is in flight, e.g. a normal scale-down).

The ``segcap`` worker is ``reference_app.ReferenceWorker`` (P1a-5s: the fast ``exact`` worker
around :class:`flora_deploy.workers_impl.segcap.SegcapShardRunner`); it reuses
:func:`store_from_env` and :func:`gpu_name` from here.

Store and worker settings come from the container env (the factory sets them):
``FLORA_STORE_KIND`` (``local`` | ``volume`` | ``bucket``), ``FLORA_STORE_ROOT`` (path / mount),
``FLORA_STORE_VOLUME`` (Volume name, for commits), ``FLORA_STORE_BUCKET`` /
``FLORA_STORE_PREFIX`` (+ ``AWS_*`` credentials and ``AWS_ENDPOINT_URL`` from the bucket
Secret), ``FLORA_MEDIA_ROOT`` (a mounted directory or ``hf://datasets/<repo>@<rev>[/sub]``),
``FLORA_INSTRUCTION_SOURCE`` / ``FLORA_INSTRUCTION_TEXT`` (the configured fixed instruction),
``FLORA_REWARD_TERMINAL_MODE``, ``FLORA_PRESET``, ``FLORA_ENVELOPE``, ``FLORA_CACHE_MODE``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import modal

from flora_deploy.jobs.store import Store
from flora_deploy.layout import CACHE_MOUNT, WEIGHTS_MOUNT
from flora_deploy.modal_app import rewarder_app as ra
from flora_deploy.modal_app.factory import CACHE_MODE_ENV
from flora_deploy.modal_app.factory import LOCAL_CACHE_ROOT as _LOCAL_CACHE_ROOT
from flora_deploy.workers_impl.reward import (
    PreflightError,
    RewardEngine,
    RewardShardRunner,
    RewardWorkerSettings,
    check_media_root,
)

REWARD_VOLUMES = {str(WEIGHTS_MOUNT): ra.weights, str(CACHE_MOUNT): ra.cache}
REWARD_RESOURCES: dict[str, Any] = {
    "gpu": ra.GPU, "cpu": ra.CPU, "memory": ra.MEMORY_MIB, "env": {ra.IMAGE_FP_ENV: ra.IMAGE_FP},
}


LOCAL_CACHE_ROOT = Path(_LOCAL_CACHE_ROOT)


def cache_env_for(mode: str, *, mount: Path | str = CACHE_MOUNT,
                  local: Path | str = LOCAL_CACHE_ROOT) -> dict[str, str]:
    """Compile/Triton cache env. ``rw``: straight into the mounted ``flora-cache`` (the warm-up
    function only). ``ro``: a container-local copy seeded from the read-only mount, so workers
    get the warm cache without ever writing the shared Volume (M7)."""
    if mode == "rw":
        return ra.cache_env()
    if mode != "ro":
        raise ValueError(f"{CACHE_MODE_ENV} {mode!r}: use ro or rw")
    seed = Path(ra.cache_env(mount)["VLLM_CACHE_ROOT"]).parent
    dest = Path(ra.cache_env(local)["VLLM_CACHE_ROOT"]).parent
    if seed.is_dir() and not dest.exists():
        shutil.copytree(seed, dest, symlinks=True)
    return ra.cache_env(local)


def build_reward_engine(snapshot: Path | str = ra.SNAPSHOT, *,
                        commit: Callable[[], None] | None = None, cache_mode: str = "ro"
                        ) -> tuple[RewardEngine, Any, dict[str, Any]]:
    """``rewarder_app.Rewarder.start``'s recipe, returning ``(engine, nv, startup facts)``.
    ``cache_mode`` (:func:`cache_env_for`): ``ro`` never commits the cache Volume."""
    t0 = time.perf_counter()
    from flora_deploy.modal_app import warm

    prefetch = warm.start_prefetch(str(snapshot), threads=ra.PREFETCH_THREADS)  # t=0
    env = cache_env_for(cache_mode)
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
    if cache_mode == "rw":
        (commit or ra.cache.commit)()
    startup = {"cache_key": ra.cache_key(), "link_and_processor_s": t1 - t0,
               "engine_build_s": t2 - t1, "enter_total_s": time.perf_counter() - t0,
               "prefetch": stats.as_json() if stats else {"done": prefetch.done},
               "engine": ra._engine_facts(llm), "cache_mode": cache_mode}
    return RewardEngine(llm, processor, spec), nv, startup


def store_from_env(env: Mapping[str, str] = os.environ, *, client: Any = None) -> Store:
    """The job store the factory configured. ``bucket``: ``BucketStore`` (boto3, in the image
    locks) over ``FLORA_STORE_BUCKET`` / ``FLORA_STORE_PREFIX``, credentials and
    ``AWS_ENDPOINT_URL`` (R2 / GCS) from the attached Secret."""
    kind = env.get("FLORA_STORE_KIND", "volume")
    if kind == "bucket":
        from flora_deploy.jobs.stores import BucketStore

        bucket = env.get("FLORA_STORE_BUCKET")
        if not bucket:
            raise RuntimeError("FLORA_STORE_BUCKET is not set (the app factory sets the store)")
        return BucketStore(bucket, prefix=env.get("FLORA_STORE_PREFIX", ""), client=client,
                           endpoint_url=env.get("AWS_ENDPOINT_URL") or None,
                           region_name=env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION"))
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
    raise RuntimeError(f"unsupported FLORA_STORE_KIND {kind!r} (local | volume | bucket)")


PREFLIGHT_PROBE = "flora/.preflight"


def preflight(env: Mapping[str, str] = os.environ, *, store: Store | None = None,
              hub_check: Callable[[str, str], Any] | None = None) -> dict[str, Any]:
    """Cheap startup checks, run **before** the GPU engine build (H3, C1): the store opens and
    answers a read (``exists`` + a ``list``: credentials, bucket name, Volume mount), the media
    root resolves, and a ``fixed`` instruction has its text. Raises :class:`PreflightError`
    naming what to fix; returns ``{"store": ..., "media": ...}`` facts otherwise."""
    t0 = time.perf_counter()
    try:
        st = store if store is not None else store_from_env(env)
        st.exists(PREFLIGHT_PROBE)
        next(iter(st.list("flora/.preflight/")), None)
    except PreflightError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported with the reason
        raise PreflightError(f"job store {env.get('FLORA_STORE', '?')} is not usable from this "
                             f"container: {type(exc).__name__}: {exc}") from exc
    media = check_media_root(env.get("FLORA_MEDIA_ROOT"), hub_check=hub_check)
    if env.get("FLORA_INSTRUCTION_SOURCE") == "fixed" and not env.get("FLORA_INSTRUCTION_TEXT"):
        raise PreflightError("episode_instruction.source is fixed but FLORA_INSTRUCTION_TEXT is "
                             "not set")
    return {"store": env.get("FLORA_STORE", ""), "media": media,
            "preflight_s": round(time.perf_counter() - t0, 3)}


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
    from flora_deploy.stages.reward import TERMINAL_MODE_ENV, terminal_mode

    mode = env.get("FLORA_TERMINAL_MODE")  # config.schema.output_settings, set by the factory
    if mode:
        if mode not in ("terminal", "terminal_rationale"):
            raise RuntimeError(f"FLORA_TERMINAL_MODE {mode!r}")
        mode = terminal_mode(mode == "terminal_rationale", env)  # a legacy override must agree
    else:
        mode = env.get(TERMINAL_MODE_ENV, "terminal")
    return RewardWorkerSettings(
        terminal_mode=mode,
        preset=env.get("FLORA_PRESET", "exact"),
        models={"rewarder": PINS_BY_KEY["rewarder"].revision},
        gpu=gpu, config_extra={"image": ra.IMAGE_FP},
        envelope=env.get("FLORA_ENVELOPE", "warn"),
        media_root=env.get("FLORA_MEDIA_ROOT"),
        default_task=env.get("FLORA_INSTRUCTION_TEXT") or None,
    )


def _die(exc: BaseException) -> None:
    """After a CUDA fault the engine is unusable: replace the container (the attempt marker is
    already written with the error, so this start does not count toward D15)."""
    print(f"fatal: {type(exc).__name__}: {exc}; exiting the container", flush=True)
    os._exit(70)


class RewardWorker:
    """``workers.ShardWorker`` for ``reward.terminal`` / ``reward.dense`` shards."""

    @modal.enter()
    def start(self) -> None:
        env = os.environ
        try:
            store = store_from_env(env)
        except Exception as exc:  # noqa: BLE001 - a misconfigured deployment: say so, early
            raise PreflightError(f"job store: {exc}") from exc
        facts = preflight(env, store=store)  # seconds, before the GPU engine build
        engine, nv, self.startup = build_reward_engine(
            cache_mode=os.environ.get(CACHE_MODE_ENV, "ro"))
        self.startup["preflight"] = facts
        self.runner = RewardShardRunner(
            store, engine, nv, reward_settings_from_env(gpu=gpu_name()),
            call_id=modal.current_function_call_id, fatal=_die)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        return self.runner.run_shard(job_id, shard_id, config_hash, token=token)

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()


def warm_reward_cache() -> dict[str, Any]:
    """The one writer of ``flora-cache`` for the Rewarder (M7): build the engine with the cache
    mounted read-write (``FLORA_CACHE_MODE=rw``) and commit it. The factory registers it as
    ``warm_reward_cache`` (one container); ``modal run`` it once per image / GPU before a job, so
    workers start from a warm, read-only cache."""
    _engine, _nv, startup = build_reward_engine(cache_mode="rw")
    print(f"warm_reward_cache: {startup}", flush=True)
    return {k: v for k, v in startup.items() if k != "engine"}


# ---------------------------------------------------------------------------------------------
# Fake-models workers (``flora deploy --fake-models``: the H0 CPU rehearsal on real Modal)
#
# Same contract, store, preflight, media resolution, env and lifecycle hooks as the real
# workers, but the model boundary is G2's deterministic fakes (``local.engines``) and the
# factory gives them CPU containers. No weights are read and no NVIDIA code is linked. Records
# say so (``provenance.certificate = fake-models``, ``models = fake``), and the factory hashes
# ``fake_models`` into every stage hash, so they never share store paths with real outputs.
#
# Fault injection for rehearsals is data in the job store, read only by these fake classes:
# ``flora/<job>/fake_faults.json`` = ``{"preempt": {"<target>": <episode #>}, "poison":
# ["<uid>", ...]}``. ``preempt``: once per target and job, in the first start of a shard that
# matches the target (a shard kind, e.g. ``segcap``, or an ``fnmatch`` pattern on shard ids,
# which are content-addressed) and reaches its n-th episode (1-based):
# ``modal.exception.simulate_preemption`` (SIGINT; ``@modal.exit`` flushes and marks the start
# preempted; Modal restarts the input). ``poison``: the container dies (``os._exit(137)``, no
# marker update) whenever a fake worker reaches that episode, so D15 quarantine, bisection and
# the poison pill run as they would for a real OOM.

FAKE_MODELS_ENV = "FLORA_FAKE_MODELS"
FAKE_CERTIFICATE = "fake-models"
FAKE_FAULTS_FILE = "fake_faults.json"


def fake_faults_path(job_id: str) -> str:
    from flora_deploy.jobs.store import JobPaths

    return f"{JobPaths(job_id).root}/{FAKE_FAULTS_FILE}"


def fake_fired_path(job_id: str, what: str) -> str:
    from flora_deploy.jobs.store import JobPaths
    from flora_deploy.jobs.types import safe_key

    return f"{JobPaths(job_id).root}/fake_faults/fired/{safe_key(what)}.json"


class FakeFaultHook:
    """Per-episode hook of the fake workers (see the block comment above)."""

    def __init__(self, store: Store, *, preempt_wait_s: int = 1,
                 exit_fn: Callable[[int], Any] = os._exit,
                 preempt_fn: Callable[[int], Any] | None = None,
                 sleep: Callable[[float], Any] = time.sleep):
        self.store = store
        self.preempt_wait_s = preempt_wait_s
        self.exit_fn, self.preempt_fn, self.sleep = exit_fn, preempt_fn, sleep
        import threading

        self._local = threading.local()  # per input (``@modal.concurrent`` reward pools)

    @property
    def job_id(self) -> str:
        return getattr(self._local, "job_id", "")

    @property
    def shard_id(self) -> str:
        return getattr(self._local, "shard_id", "")

    @property
    def faults(self) -> dict[str, Any]:
        return getattr(self._local, "faults", {})

    def start_shard(self, job_id: str, shard_id: str) -> None:
        import json

        loc = self._local
        loc.job_id, loc.shard_id, loc.seen = job_id, shard_id, 0
        p = fake_faults_path(job_id)
        try:
            loc.faults = json.loads(self.store.get_bytes(p)) if self.store.exists(p) else {}
        except (OSError, ValueError):
            loc.faults = {}

    def before_episode(self, uid: str) -> None:
        import json

        self._local.seen = getattr(self._local, "seen", 0) + 1
        seen = self._local.seen
        if uid in set(self.faults.get("poison", ())):
            print(f"fake-models: poison {uid} -> os._exit(137)", flush=True)
            commit = getattr(self.store, "commit", None)
            if callable(commit):
                commit()
            self.exit_fn(137)
            return
        for target, at in sorted((self.faults.get("preempt") or {}).items()):
            if seen != int(at) or not preempt_matches(target, self.shard_id):
                continue
            fired = fake_fired_path(self.job_id, f"preempt-{target}")
            if self.store.exists(fired):
                continue
            self.store.put_bytes(fired, json.dumps({"shard_id": self.shard_id, "uid": uid,
                                                    "t": time.time()}).encode())
            print(f"fake-models: simulate_preemption({self.preempt_wait_s}) on "
                  f"{self.shard_id} before {uid}", flush=True)
            fn = self.preempt_fn
            if fn is None:
                from modal.exception import simulate_preemption as fn
            fn(self.preempt_wait_s)
            self.sleep(60.0)  # the SIGINT arrives during this sleep
            return


def preempt_matches(target: str, shard_id: str) -> bool:
    """``target`` is a shard kind (``segcap`` matches the planned ``segcap-<hex>`` shards, not
    bisection children ``segcap-ep-...``) or an ``fnmatch`` pattern on the shard id."""
    import fnmatch

    if fnmatch.fnmatchcase(shard_id, target):
        return True
    head, sep, rest = shard_id.partition(f"{target}-")
    return not head and bool(sep) and "-" not in rest


def fake_determinism_record(mode: str) -> dict[str, Any]:
    from flora_deploy.accel import determinism as det

    return {"schema": det.DETERMINISM_SCHEMA, "mode": det.check_mode(mode), "in_effect": False,
            "fake_models": True}


def configured_determinism_env(env: Mapping[str, str] = os.environ) -> str:
    """``reference_app.configured_determinism`` without importing the HF worker module: the
    hashed D-1 mode (``FLORA_DETERMINISM`` must agree with ``FLORA_OUTPUT_SETTINGS``)."""
    import json

    from flora_deploy.accel import determinism as det

    mode = env.get("FLORA_DETERMINISM") or None
    outputs = env.get("FLORA_OUTPUT_SETTINGS")
    hashed = json.loads(outputs).get("determinism") if outputs else None
    if mode and hashed and mode != hashed:
        raise det.DeterminismError(
            f"FLORA_DETERMINISM {mode!r} != FLORA_OUTPUT_SETTINGS determinism {hashed!r}")
    return det.check_mode(mode or hashed or det.NVIDIA)


def fake_segcap_engine(mode: str, hook: FakeFaultHook) -> Any:
    """``local.engines.FakeSegcapEngine`` that runs ``hook`` before each episode and carries
    the configured D-1 mode (the runner refuses an engine in another mode)."""
    from flora_deploy.local.engines import FakeSegcapEngine

    class _Engine(FakeSegcapEngine):
        def __init__(self) -> None:
            super().__init__()
            self.determinism = fake_determinism_record(mode)

        def runtime_provenance(self) -> dict:
            return {**super().runtime_provenance(), "determinism": dict(self.determinism),
                    "certificate": FAKE_CERTIFICATE}

        def result_for(self, item: dict) -> dict:
            hook.before_episode(item["episode"])
            return super().result_for(item)

    return _Engine()


def fake_preflight(env: Mapping[str, str] = os.environ) -> tuple[Store, dict[str, Any]]:
    """The real workers' store + :func:`preflight`, refusing outside a fake-models app."""
    if env.get(FAKE_MODELS_ENV) != "1":
        raise PreflightError(f"{FAKE_MODELS_ENV} is not set: fake workers only run in an app "
                             "deployed with `flora deploy --fake-models`")
    try:
        store = store_from_env(env)
    except Exception as exc:  # noqa: BLE001 - a misconfigured deployment: say so, early
        raise PreflightError(f"job store: {exc}") from exc
    return store, preflight(env, store=store)


def fake_segcap_runner(store: Store, env: Mapping[str, str], hook: FakeFaultHook, *,
                       call_id: Callable[[], str | None] = lambda: "",
                       fatal: Callable[[BaseException], None] | None = None) -> Any:
    from flora_deploy.local.engines import FAKE_MODELS
    from flora_deploy.workers_impl.segcap import SegcapShardRunner, SegcapWorkerSettings

    mode = configured_determinism_env(env)
    settings = SegcapWorkerSettings(
        preset=env.get("FLORA_PRESET", "exact"), models=dict(FAKE_MODELS["segcap"]),
        gpu=gpu_name(), certificate=FAKE_CERTIFICATE, envelope=env.get("FLORA_ENVELOPE", "warn"),
        media_root=env.get("FLORA_MEDIA_ROOT"),
        instruction_text=env.get("FLORA_INSTRUCTION_TEXT") or None, determinism=mode)
    return SegcapShardRunner(store, fake_segcap_engine(mode, hook), settings, call_id=call_id,
                             fatal=fatal)


def fake_reward_runner(store: Store, env: Mapping[str, str], hook: FakeFaultHook, *,
                       call_id: Callable[[], str | None] = lambda: "",
                       fatal: Callable[[BaseException], None] | None = None) -> Any:
    """``RewardShardRunner`` with G2's fake Rewarder family and ``FakeLLM``. The caller holds
    ``local.engines.vllm_shim()`` open while it runs (``stages.reward`` imports ``vllm``)."""
    import dataclasses

    from flora_deploy.local.engines import (
        FAKE_MODELS,
        EpisodeRegistry,
        FakeLLM,
        fake_rewarder_family,
    )

    registry = EpisodeRegistry()
    settings = dataclasses.replace(
        reward_settings_from_env(env, gpu=gpu_name()), models=dict(FAKE_MODELS["reward"]),
        certificate=FAKE_CERTIFICATE, config_extra={"fake_models": FAKE_CERTIFICATE},
        decoder="inline", workers=1)
    runner = RewardShardRunner(store, RewardEngine(FakeLLM(), "fake-processor"),
                               fake_rewarder_family(registry), settings, call_id=call_id,
                               fatal=fatal)
    from_index = runner.episode_from_index

    def episode(ref):  # the runner's own reader, plus the rehearsal's fault hook
        hook.before_episode(ref.uid)
        ep = from_index(ref)
        registry.add(ep.videos, ref.uid)
        return ep

    runner.episode_for = episode
    return runner


class FakeSegcapWorker:
    """``segcap`` worker of a ``--fake-models`` app: the real ``SegcapShardRunner`` over the
    real store and media root, ``local.engines.FakeSegcapEngine`` at the model boundary."""

    @modal.enter()
    def start(self) -> None:
        self.store, self.preflight = fake_preflight()
        self.hook = FakeFaultHook(self.store)
        self.runner = fake_segcap_runner(self.store, os.environ, self.hook,
                                         call_id=modal.current_function_call_id, fatal=_die)
        print(f"fake-models segcap worker: determinism={self.runner.settings.determinism} "
              f"media={self.runner.settings.media_root} preflight={self.preflight}", flush=True)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        self.hook.start_shard(job_id, shard_id)
        return self.runner.run_shard(job_id, shard_id, config_hash, token=token)

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()


class FakeRewardWorker:
    """``reward.terminal`` / ``reward.dense`` worker of a ``--fake-models`` app: the real
    ``RewardShardRunner`` (request building, rounds, parsing, records) with G2's fakes."""

    @modal.enter()
    def start(self) -> None:
        from flora_deploy.local.engines import vllm_shim

        self.store, self.preflight = fake_preflight()
        self._shim = vllm_shim()  # held for the container's life
        self._shim.__enter__()
        self.hook = FakeFaultHook(self.store)
        self.runner = fake_reward_runner(self.store, os.environ, self.hook,
                                         call_id=modal.current_function_call_id, fatal=_die)
        print(f"fake-models reward worker: terminal_mode={self.runner.settings.terminal_mode} "
              f"media={self.runner.settings.media_root} preflight={self.preflight}", flush=True)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        self.hook.start_shard(job_id, shard_id)
        return self.runner.run_shard(job_id, shard_id, config_hash, token=token)

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()
