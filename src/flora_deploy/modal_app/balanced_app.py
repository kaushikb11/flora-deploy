"""The ``balanced`` shard workers (P1b-1): module-scope, **undecorated** classes the app factory
(``modal_app.factory``) decorates from the config's pools, as ``workers_app.RewardWorker``.

- ``SegmentWorker``: ``segment`` shards, vLLM TempSeg (``derived/tempseg-merged``) on the
  segment pool's GPU (``L40S`` by default);
- ``CaptionWorker``: ``caption`` shards, vLLM Captioner (``derived/captioner-bf16``) on the
  caption pool's GPU (``H100!``).

``@modal.enter`` (:func:`build_runner` over :func:`build_stage`, the part a future online
service reuses for the identical ``cert_key``, ``docs/interactive-design.md`` §2-3), fail closed
at every step and in this order, so a bad deployment dies in seconds, before the GPU engine
build:

0. :func:`check_pinned_kind`: a derived weight without a pinned content digest
   (``pins.DERIVED_PINS``) is refused before anything else. ``plan_app`` already refuses it at
   plan time (review 10 M1); if a container still gets here, ``@modal.enter`` does not raise:
   the container serves a :class:`RefusedRunner`, which ends every shard it is given
   ``shard_failed`` (``DEPLOYMENT_REFUSED``) at once, with no Modal retry, no orchestrator
   respawn and no D15 count;
1. ``workers_app.preflight`` (store reachable, media root readable, fixed instruction set);
2. the derived weights of the kind (``PresetSpec.derived_weights``) are **resolved from
   ``weights.derived``** (``FLORA_DERIVED_RECEIPTS``: receipt name -> sha256 of the receipt file
   ``flora prepare`` recorded) and verified (:func:`verify_derived_weights`): the receipt file
   on the weights Volume must hash to the recorded identity, its **content digest**
   (``prep.convert_captioner.content_digest``, recomputed from the bytes read) must equal the
   pin in ``pins.DERIVED_PINS`` (``None`` refuses), and the files must match the receipt
   (Captioner: ``captioner_load.verify_derived``; TempSeg: ``merge_tempseg.verify_merged``;
   every byte by default, ``FLORA_CAPTIONER_FULL_VERIFY=0`` opts out, in a background thread
   joined before the worker is ready, overlapping the engine build). The Captioner
   index is also checked for ``model.visual`` ``_extra_state`` keys (vLLM's vision loader would
   ``KeyError``);
3. the certification stamp (``workers_impl.certstamp``; its key needs those content digests);
4. NVIDIA's code (``load_family``), the processor, and the engine from
   ``engines.offline_llm.spec_for`` on the **pool GPU** (``FLORA_POOL_GPUS``), so the engine is
   exactly what keys the shard files (``config.hashing``); ``OfflineEngine`` refuses a
   ``dt_bias`` plugin flag that disagrees with the env and verifies the built engine's GDN
   parameter dtypes (review 6 M6);
5. the OPT-1 ``accel`` patches (``workers_impl.balanced.apply_accel``) and the shard runner,
   with ``fatal=workers_app._die``: after a CUDA fault / dead engine the attempt is marked and
   the container exits (it is replaced; the start does not count toward D15).

Compile caches (:func:`prepare_cache`, ``engines.compile_cache``): keyed by image, versions,
the compile-relevant engine spec, plugin flag, model content and GPU. ``flora-cache`` stays
mounted read-only (``FLORA_CACHE_MODE=ro``, M7): a verified ``balanced-cc/<key>`` manifest is
copied to container-local disk, else the lab's unkeyed ``balanced/<engine kind>`` cache (same
GPU and plugin flag only), else the engine compiles cold. A build that fails on a loaded cache
is retried once cold (:func:`build_engine`). A container that compiled publishes its cache
once, after its first shard ran generates (``WorkerCache.maybe_publish``: a ``modal.Dict``
lease, upload through the Volume client API, manifest last; ``FLORA_COMPILE_CACHE_PUBLISH=0``
turns it off). ``FLORA_COMPILE_CACHE=off`` forces the container cold (no seed, no publish). The
cache a container ran (``CacheStatus.identity``) is in its startup facts, each shard's audit row
and every record's ``provenance.compile_cache``: a warm start replays the publisher's Inductor
autotune picks, so it is not output-neutral (``engines.compile_cache``, "Determinism").

``run_shard`` is the ``workers.ShardWorker`` contract (``max_inputs`` is 1: the runner runs a
shard's episodes concurrently inside one input). ``@modal.exit`` flushes and marks a running
shard preempted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import modal

from flora_deploy.layout import CACHE_MOUNT, WEIGHTS_MOUNT
from flora_deploy.modal_app.factory import DERIVED_RECEIPTS_ENV

LOCAL_CACHE = Path("/tmp/flora-cache/balanced")
ENGINE_KINDS = {"segment": "tempseg", "caption": "captioner"}
PUBLISH_ENV = "FLORA_COMPILE_CACHE_PUBLISH"  # "0": never publish (default: publish once)
# "on" (default): seed a verified cache (else the lab's, else cold) and publish once; "off":
# force this container cold, no seed and no publish (review 13 W1: a certified run's one
# independent cold compile). Forwarded from the deploying process (factory._recipe_env).
COMPILE_CACHE_ENV = "FLORA_COMPILE_CACHE"
COMPILE_CACHE_MODES = ("on", "off")
# The GPU each unkeyed lab cache (``balanced_lab.GPUS``, ``flora-cache:/balanced/<kind>``) was
# compiled on: vLLM's AOT key has no GPU factor, so another GPU never seeds from it.
LEGACY_LAB_GPUS = {"tempseg": "L40S", "captioner": "H100"}
DERIVED_FOR_KIND = {"segment": "tempseg_merged", "caption": "captioner_bf16"}


DEPLOYMENT_REFUSED = "deployment_refused"  # shard_failed reason of a refused start (M1)


class DerivedWeightsError(RuntimeError):
    """A derived weight directory is missing, unpinned or not the recorded build."""


class UnpinnedDerivedWeights(DerivedWeightsError):
    """A derived weight the kind needs has no pinned content digest: the same answer on every
    container of this deployment (``store``: the job store, to report it per shard)."""

    store: Any = None


def check_pinned_kind(kind: str, preset: str = "balanced") -> None:
    """Step 0: refuse when a derived weight ``kind`` needs is unpinned (``pins.DERIVED_PINS``),
    with the message ``flora plan`` / ``deploy`` give (``config.receipts.unpinned_message``).
    The factory refuses this at plan time; this is the in-container backstop."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.config.receipts import unpinned, unpinned_message

    names = get_preset(preset).derived_weights.get(kind, ())  # type: ignore[call-overload]
    bad = unpinned(names)
    if bad:
        raise UnpinnedDerivedWeights(" ".join(unpinned_message(n, kinds=(kind,), preset=preset)
                                              for n in bad))


def derived_dir(name: str, root: Path | str = WEIGHTS_MOUNT) -> Path:
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.config.receipts import pin_key
    from flora_deploy.layout import derived_path
    from flora_deploy.prep.derived import DERIVED_BUILDS

    b = DERIVED_BUILDS[name]
    assert b.dirname == pin_key(name)
    return Path(derived_path(b.dirname, PINS_BY_KEY[b.source_key], root))


def recorded_receipts(env: Mapping[str, str] = os.environ) -> dict[str, str]:
    try:
        out = json.loads(env.get(DERIVED_RECEIPTS_ENV) or "{}")
    except ValueError as e:
        raise DerivedWeightsError(f"{DERIVED_RECEIPTS_ENV} is not JSON ({e})") from None
    if not isinstance(out, dict):
        raise DerivedWeightsError(f"{DERIVED_RECEIPTS_ENV} must map names to sha256")
    return {str(k): str(v) for k, v in out.items()}


def verify_derived_weights(name: str, directory: Path, recorded: str | None, *,
                           full: bool | None = None, background: bool = False,
                           verify_files: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Check one derived weight directory (module docstring, step 2). Returns ``{"name",
    "dir", "receipt_sha256", "content_sha256", "files"}``; with ``background`` the file check
    runs on a thread and ``files`` is a callable that joins it (and raises its error)."""
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
    from flora_deploy.config.receipts import pin_key
    from flora_deploy.prep.convert_captioner import ContentDigestError, checked_content_digest

    if not recorded:
        raise DerivedWeightsError(f"{name}: no receipt recorded under weights.derived (run "
                                  "`flora prepare`; the deployment passes it as "
                                  f"{DERIVED_RECEIPTS_ENV})")
    path = directory / "derived_receipt.json"
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise DerivedWeightsError(f"{name}: no receipt at {path} ({e})") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != recorded:
        raise DerivedWeightsError(f"{name}: {path} hashes to {got}, but weights.derived records "
                                  f"{recorded}: the Volume holds another build (re-run `flora "
                                  "prepare` and redeploy)")
    try:
        content = checked_content_digest(json.loads(raw))
    except (ValueError, ContentDigestError) as e:
        raise DerivedWeightsError(f"{name}: {path}: {e}") from None
    pin = DERIVED_PINS_BY_KEY.get(pin_key(name))
    if pin is None or pin.content_sha256 is None:
        from flora_deploy.config.receipts import unpinned_message

        raise UnpinnedDerivedWeights(f"{name}: {unpinned_message(name)}")
    if content != pin.content_sha256:
        raise DerivedWeightsError(f"{name}: content digest {content} != pinned "
                                  f"{pin.content_sha256}")
    if verify_files is None:
        if name == "captioner_bf16":
            from flora_deploy.accel.captioner_load import full_verify_default, verify_derived

            want_full = full_verify_default() if full is None else full

            def verify_files() -> Any:
                return verify_derived(directory, full=want_full)
        else:
            from flora_deploy.accel.captioner_load import full_verify_default
            from flora_deploy.prep.merge_tempseg import verify_merged

            want_full_merged = full_verify_default() if full is None else full

            def verify_files() -> Any:  # every byte by default, as the Captioner (review 8b L6)
                return verify_merged(directory, full=want_full_merged)
    out: dict[str, Any] = {"name": name, "dir": str(directory), "receipt_sha256": got,
                           "content_sha256": content}
    if not background:
        out["files"] = verify_files()
        return out
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["report"] = verify_files()
        except BaseException as e:  # noqa: BLE001 - re-raised by join()
            box["error"] = e

    t = threading.Thread(target=run, daemon=True, name=f"flora-verify-{name}")
    t.start()

    def join() -> Any:
        t.join()
        if "error" in box:
            raise DerivedWeightsError(f"{name}: file check failed: {box['error']}") \
                from box["error"]
        return box["report"]

    out["files"] = join
    return out


def check_extra_state(directory: Path) -> list[str]:
    from flora_deploy.engines.offline_llm import check_extra_state as check

    idx = json.loads((directory / "model.safetensors.index.json").read_text())
    return check(idx["weight_map"])


def _image_fp(env: Mapping[str, str]) -> str | None:
    from flora_deploy.modal_app.rewarder_app import IMAGE_FP_ENV

    try:
        from flora_deploy.modal_app.images import image_fingerprint

        return image_fingerprint("img_vllm")
    except OSError:
        return env.get(IMAGE_FP_ENV) or None


def prepare_cache(kind: str, spec: Any, model_content: str | None,
                  env: Mapping[str, str] = os.environ, *, mount: Path | str = CACHE_MOUNT,
                  local_root: Path = LOCAL_CACHE, image_fp: str | None = None,
                  log: Callable[[str], None] = print) -> Any:
    """The container's ``compile_cache.WorkerCache`` (module docstring). Never raises: a key
    that cannot be derived (no image fingerprint, say) means a cold, unpublished cache."""
    from flora_deploy.engines import compile_cache as cc
    from flora_deploy.workers_impl.reward import default_container_id

    engine_kind = ENGINE_KINDS[kind]
    mode = env.get(COMPILE_CACHE_ENV) or "on"
    if mode not in COMPILE_CACHE_MODES:  # a typo must not silently mean "on"
        raise ValueError(f"{COMPILE_CACHE_ENV}={mode!r}: use one of {COMPILE_CACHE_MODES}")
    try:
        image_fp = image_fp or _image_fp(env)
        if not image_fp:
            raise RuntimeError("no img_vllm fingerprint: the key would not name the image")
        key = cc.spec_key(spec, model_content=model_content, image_fp=image_fp)
    except Exception as e:  # noqa: BLE001
        log(f"compile cache: no key ({type(e).__name__}: {e}); compiling cold, not publishing")
        local = local_root / f"{engine_kind}-unkeyed"
        cc.reset_local(local)
        key = cc.CacheKey({"engine_kind": engine_kind, "dtbias_fp32": False, "gpu": "unknown"})
        # forced cold still records cold:forced (the witness's evidence; review 14 W3)
        why = cc.FORCED_COLD if mode == "off" else str(e)[:500]
        return cc.WorkerCache(key, local, cc.CacheStatus("cold", "unkeyed", reason=why),
                              publish_enabled=False, log=log)
    local = local_root / key.name
    if mode == "off":
        cc.reset_local(local)
        log(f"compile cache {key.name}: {cc.FORCED_COLD}; compiling cold, not publishing")
        return cc.WorkerCache(key, local, cc.CacheStatus("cold", key.name, reason=cc.FORCED_COLD),
                              publish_enabled=False, log=log)
    status = cc.seed(key, mount, local, log=log)
    plugin = "-dtbias" if spec.dtbias_fp32 else ""
    legacy = Path(mount) / "balanced" / f"{engine_kind}{plugin}"
    if status.state == "cold" and status.reason == cc.NO_CACHE \
            and LEGACY_LAB_GPUS.get(engine_kind) == key.factors["gpu"] \
            and (legacy / "vllm").is_dir():
        status = cc.seed_legacy(legacy, local, key, log=log)
    return cc.WorkerCache(key, local, status, publish_enabled=env.get(PUBLISH_ENV, "1") != "0",
                          holder=default_container_id(), mount=mount, log=log)


def build_engine(spec: Any, cache: Any, engine_cls: Callable[[Any], Any]) -> Any:
    """``engine_cls(spec)``; when that fails on a loaded (``warm`` / ``legacy``) cache for a
    reason other than the spec itself, empty the cache and build once more, cold."""
    from flora_deploy.engines.offline_llm import EngineArgsError, EngineVerifyError
    from flora_deploy.engines.patches.dt_bias_fp32 import DtBiasActivationError

    try:
        return engine_cls(spec)
    except (EngineArgsError, EngineVerifyError, DtBiasActivationError):
        raise  # deterministic: the same on a cold build
    except Exception as e:  # noqa: BLE001
        if cache is None or cache.status.state not in ("warm", "legacy"):
            raise
        cache.reset_cold(f"{type(e).__name__}: {e}"[:300])
        import gc

        gc.collect()
        return engine_cls(spec)


def pool_gpu(kind: str, env: Mapping[str, str] = os.environ) -> str:
    from flora_deploy.workers_impl.certstamp import POOL_GPUS_ENV

    gpu = json.loads(env.get(POOL_GPUS_ENV) or "{}").get(kind)
    if not gpu:
        raise RuntimeError(f"{POOL_GPUS_ENV} names no pool GPU for {kind!r}")
    return str(gpu)


class BalancedStage(NamedTuple):
    """What :func:`build_stage` returns: the pieces a ``segment`` / ``caption`` container runs,
    shared by the batch runner (:func:`build_runner`) and a future online service
    (``docs/interactive-design.md`` §2-3), so both carry the identical ``cert_key``.

    ``engine``: the ``OfflineEngine``. ``job``: the episode job with the accel scope applied
    (``workers_impl.balanced.AccelJob``), **not** yet stamped. ``stamp_fn(embodiment=None)``:
    the ``certstamp.CertStamp`` to stamp records with (``None``: the container's
    ``FLORA_EMBODIMENT``; a service passes each request's; the key does not depend on it).
    ``info``: the jobs' ``StageInfo``. ``store`` / ``startup``: the job store and the startup
    facts. ``cache``: the container's compile cache (``engines.compile_cache.WorkerCache``)."""

    engine: Any
    job: Any
    stamp_fn: Callable[..., Any]
    info: Any
    store: Any
    startup: dict
    cache: Any = None  # the compile_cache.WorkerCache


def build_stage(kind: str, env: Mapping[str, str] = os.environ) -> BalancedStage:
    """The stage of a real ``segment`` / ``caption`` container (module docstring, steps 0-5)."""
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.config.presets import get_preset
    from flora_deploy.engines.offline_llm import OfflineEngine, spec_for
    from flora_deploy.layout import snapshot_path
    from flora_deploy.modal_app import warm
    from flora_deploy.modal_app.workers_app import gpu_name, preflight, store_from_env
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family
    from flora_deploy.workers_impl.balanced import apply_accel, wrap_job
    from flora_deploy.workers_impl.certstamp import stamp_from_env
    from flora_deploy.workers_impl.reward import PreflightError
    from flora_deploy.workers_impl.segment import (
        InputResolver,
        SegmentEpisodeJob,
        StageInfo,
    )

    if kind not in ENGINE_KINDS:
        raise ValueError(f"no balanced worker for {kind!r}")
    t0 = time.perf_counter()
    marks: dict[str, float] = {}

    def mark(what: str) -> None:
        marks[what] = round(time.perf_counter() - t0, 2)

    try:
        store = store_from_env(env)
    except Exception as exc:  # noqa: BLE001 - a misconfigured deployment: say so, early
        raise PreflightError(f"job store: {exc}") from exc
    preset_name = env.get("FLORA_PRESET", "balanced")
    try:  # step 0: static, before any probe (review 10 M1)
        check_pinned_kind(kind, preset_name)
    except UnpinnedDerivedWeights as e:
        e.store = store
        raise
    facts = preflight(env, store=store)
    mark("preflight")

    preset = get_preset(preset_name)
    names = preset.derived_weights.get(kind, ())  # type: ignore[call-overload]
    recorded = recorded_receipts(env)
    dirs = {n: derived_dir(n) for n in names}
    model_dir = dirs[DERIVED_FOR_KIND[kind]]
    from flora_deploy.accel.captioner_load import full_verify_default

    # t=0 page-cache warm-up; the full hash below already reads every byte (review 8b L7)
    prefetch = None if full_verify_default() else warm.start_prefetch(str(model_dir),
                                                                       threads=16)
    derived = {n: verify_derived_weights(n, dirs[n], recorded.get(n),
                                         background=True) for n in names}
    if kind == "caption":
        bad = [p for p in check_extra_state(model_dir) if p.startswith("model.visual")]
        if bad:
            raise DerivedWeightsError(f"captioner_bf16: _extra_state under {bad}")
    mark("derived_receipts")
    gpu = gpu_name()
    stamp = stamp_from_env(kind, env, gpu=gpu,
                           derived_content={n: d["content_sha256"] for n, d in derived.items()})
    mark("cert_key")

    spec = spec_for(kind, str(model_dir), pool_gpu(kind, env))
    cache = prepare_cache(kind, spec, derived[DERIVED_FOR_KIND[kind]]["content_sha256"], env)
    os.environ.update(cache.env)  # before vLLM is imported
    mark("compile_cache")
    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"],
                                                       WEIGHTS_MOUNT)))
    processor = nv.models.processor(ENGINE_KINDS[kind], model_dir)
    # validated kwargs, plugin flag check, GDN dtype verify (M6); one cold retry (cache)
    engine = build_engine(spec, cache, OfflineEngine)
    mark("engine")
    files = {n: (d["files"]() if callable(d["files"]) else d["files"])
             for n, d in derived.items()}  # fail closed before ready
    mark("derived_files")

    pins = preset.pins[kind]  # type: ignore[index]
    info = StageInfo(engine_label=spec.label, engine_spec=spec.hashed(),
                     models={k: PINS_BY_KEY[k].revision for k in pins}, gpu=gpu,
                     config_extra={"derived": {n: d["content_sha256"]
                                               for n, d in derived.items()}})
    inputs = InputResolver(store, media_root=env.get("FLORA_MEDIA_ROOT"),
                           default_task=env.get("FLORA_INSTRUCTION_TEXT") or None)
    if kind == "segment":
        from flora_deploy.stages.segment import SegmentStage

        job: Any = SegmentEpisodeJob(SegmentStage(nv, processor), info, inputs)
    else:
        from flora_deploy.stages.caption import CaptionStage
        from flora_deploy.workers_impl.caption import CaptionEpisodeJob, segments_for

        # pinned oracle segments (stage-isolated certification) or this job's own
        job = CaptionEpisodeJob(CaptionStage(nv, processor), info, inputs,
                                segments=segments_for(env))
    active = apply_accel(nv)
    stats = prefetch.wait(timeout=0) if prefetch is not None else None
    startup = {"kind": kind, "preflight": facts, "marks": marks, "spec": spec.as_json(),
               "gdn_dtypes": engine.gdn_dtypes, "accel": active.provenance() if active else None,
               "derived": {n: {k: v for k, v in d.items() if k != "files"}
                           for n, d in derived.items()},
               "derived_files": {n: bool(f) for n, f in files.items()},
               "compile_cache": cache.status.as_json(),
               "prefetch": stats.as_json() if stats else {"done": getattr(prefetch, "done", None)}}

    def stamp_fn(embodiment: str | None = None) -> Any:
        if embodiment is None or embodiment == stamp.embodiment:
            return stamp
        return dataclasses.replace(stamp, embodiment=embodiment)

    return BalancedStage(engine=engine, job=wrap_job(job, active=active), stamp_fn=stamp_fn,
                         info=info, store=store, startup=startup, cache=cache)


def build_runner(kind: str, env: Mapping[str, str] = os.environ, *,
                 call_id: Callable[[], str | None] = lambda: "",
                 fatal: Callable[[BaseException], None] | None = None) -> tuple[Any, dict]:
    """The ``BalancedShardRunner`` of a real ``segment`` / ``caption`` container
    (:func:`build_stage`, the records stamped outermost). Returns ``(runner, startup facts)``."""
    from flora_deploy.workers_impl.balanced import wrap_job
    from flora_deploy.workers_impl.segment import BalancedShardRunner, BalancedWorkerSettings

    st = build_stage(kind, env)
    runner = BalancedShardRunner(st.store, st.engine, wrap_job(st.job, stamp=st.stamp_fn()),
                                 BalancedWorkerSettings(gpu=st.info.gpu, sample_resources=True),
                                 call_id=call_id, fatal=fatal)
    runner.compile_cache = st.cache
    st.startup.update(runner.startup_facts())  # batch policy, executor, peak RSS / GPU memory
    return runner, st.startup


def after_shard(runner: Any) -> None:
    """Publish the compile cache once, after a shard (``WorkerCache.maybe_publish``)."""
    cache = getattr(runner, "compile_cache", None)
    if cache is not None:
        cache.maybe_publish(getattr(runner, "engine", None))


class RefusedRunner:
    """The runner of a container whose start was refused for a deterministic deployment reason
    (:class:`UnpinnedDerivedWeights`). ``@modal.enter`` does not raise (that would be a billed
    GPU cold start per Modal retry and orchestrator respawn, review 10 M1); instead every shard
    it is given ends at once: the attempt marker carries the error (not ``crashed``, so it is
    never D15-counted into a poison pill), a ``shard_failed`` record (``DEPLOYMENT_REFUSED``,
    with the fix) makes it terminal for the orchestrator, and the call returns without a
    receipt instead of raising, so ``modal.Retries`` does not retry it."""

    def __init__(self, store: Any, error: str, *, container_id: str | None = None,
                 call_id: Callable[[], str | None] = lambda: "",
                 clock: Callable[[], float] = time.time):
        from flora_deploy.jobs.store import safe_key
        from flora_deploy.workers_impl.reward import default_container_id

        self.store, self.error, self.call_id, self.clock = store, error, call_id, clock
        self.container_id = safe_key(container_id or default_container_id())

    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        from flora_deploy.jobs import attempts as A
        from flora_deploy.jobs.deps import StoreManifestSource
        from flora_deploy.jobs.orchestrator import utc

        refresh = getattr(self.store, "refresh", None)
        if callable(refresh):
            refresh()
        jf = A.JobFiles(self.store, job_id, config_hash)
        shard = StoreManifestSource(jf).shard(shard_id)
        if A.superseded(jf, shard.kind, shard_id, token):
            return None
        now = utc(self.clock())
        marker = A.write_attempt(jf, shard.kind, shard_id, self.container_id, started_utc=now,
                                 orchestrator_attempt=A.token_attempt(token), token=token,
                                 call_id=self.call_id() or "")
        A.mark_exception(jf, shard.kind, shard_id, self.container_id, ended_utc=now,
                         error=self.error, marker=marker, crashed=False)
        A.mark_failed(jf, shard, created_utc=now, reason=DEPLOYMENT_REFUSED,
                      detail={"error": self.error, "container_id": self.container_id})
        print(f"refused {shard_id}: {self.error}", flush=True)
        return None

    def preempt(self) -> bool:
        return False


def _start(kind: str) -> tuple[Any, dict]:
    from flora_deploy.modal_app.workers_app import _die

    try:
        return build_runner(kind, call_id=modal.current_function_call_id, fatal=_die)
    except UnpinnedDerivedWeights as e:
        print(f"balanced {kind} worker refused: {e}", flush=True)
        return (RefusedRunner(e.store, str(e), call_id=modal.current_function_call_id),
                {"kind": kind, "refused": str(e)})
    except BaseException as exc:  # no shard ever starts here: tell the orchestrator why
        from flora_deploy.jobs.attempts import record_startup_failure
        from flora_deploy.modal_app.workers_app import store_from_env

        try:
            record_startup_failure(store_from_env(), exc)
        except Exception:  # noqa: BLE001 - the store itself is broken: nowhere to write it
            pass
        raise


class SegmentWorker:
    """``workers.ShardWorker`` for balanced ``segment`` shards (module docstring)."""

    @modal.enter()
    def start(self) -> None:
        self.runner, self.startup = _start("segment")
        print(f"balanced segment worker ready: {json.dumps(self.startup, default=str)}",
              flush=True)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        out = self.runner.run_shard(job_id, shard_id, config_hash, token=token)
        after_shard(self.runner)
        return out

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()


class CaptionWorker:
    """``workers.ShardWorker`` for balanced ``caption`` shards (module docstring)."""

    @modal.enter()
    def start(self) -> None:
        self.runner, self.startup = _start("caption")
        print(f"balanced caption worker ready: {json.dumps(self.startup, default=str)}",
              flush=True)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        out = self.runner.run_shard(job_id, shard_id, config_hash, token=token)
        after_shard(self.runner)
        return out

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()
