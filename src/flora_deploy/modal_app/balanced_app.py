"""The ``balanced`` shard workers (P1b-1): module-scope, **undecorated** classes the app factory
(``modal_app.factory``) decorates from the config's pools, as ``workers_app.RewardWorker``.

- ``SegmentWorker``: ``segment`` shards, vLLM TempSeg (``derived/tempseg-merged``) on the
  segment pool's GPU (``L40S`` by default);
- ``CaptionWorker``: ``caption`` shards, vLLM Captioner (``derived/captioner-bf16``) on the
  caption pool's GPU (``H100!``).

``@modal.enter`` (:func:`build_runner`), fail closed at every step and in this order, so a bad
deployment dies in seconds, before the GPU engine build:

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

Compile caches: ``flora-cache:/balanced/<engine kind>`` is mounted read-only and copied to
container-local disk (``FLORA_CACHE_MODE=ro``, M7); ``balanced_lab`` is its writer.

``run_shard`` is the ``workers.ShardWorker`` contract (``max_inputs`` is 1: the runner runs a
shard's episodes concurrently inside one input). ``@modal.exit`` flushes and marks a running
shard preempted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import modal

from flora_deploy.layout import CACHE_MOUNT, WEIGHTS_MOUNT
from flora_deploy.modal_app.factory import DERIVED_RECEIPTS_ENV

LOCAL_CACHE = Path("/tmp/flora-cache/balanced")
ENGINE_KINDS = {"segment": "tempseg", "caption": "captioner"}
DERIVED_FOR_KIND = {"segment": "tempseg_merged", "caption": "captioner_bf16"}


class DerivedWeightsError(RuntimeError):
    """A derived weight directory is missing, unpinned or not the recorded build."""


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
        raise DerivedWeightsError(f"{name}: no content digest is pinned in pins.DERIVED_PINS; "
                                  "refusing an unpinned derived checkpoint")
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


def cache_env(engine_kind: str, *, mount: Path | str = CACHE_MOUNT,
              local: Path = LOCAL_CACHE) -> dict[str, str]:
    """``balanced_lab.cache_env`` layout, copied from the read-only mount to local disk."""
    seed = Path(mount) / "balanced" / engine_kind
    dest = local / engine_kind
    if seed.is_dir() and not dest.exists():
        shutil.copytree(seed, dest, symlinks=True)
    env = {"VLLM_CACHE_ROOT": str(dest / "vllm"), "TRITON_CACHE_DIR": str(dest / "triton"),
           "CUDA_CACHE_PATH": str(dest / "cuda"), "XDG_CACHE_HOME": str(dest / "xdg")}
    for p in env.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return env


def pool_gpu(kind: str, env: Mapping[str, str] = os.environ) -> str:
    from flora_deploy.workers_impl.certstamp import POOL_GPUS_ENV

    gpu = json.loads(env.get(POOL_GPUS_ENV) or "{}").get(kind)
    if not gpu:
        raise RuntimeError(f"{POOL_GPUS_ENV} names no pool GPU for {kind!r}")
    return str(gpu)


def build_runner(kind: str, env: Mapping[str, str] = os.environ, *,
                 call_id: Callable[[], str | None] = lambda: "",
                 fatal: Callable[[BaseException], None] | None = None) -> tuple[Any, dict]:
    """The ``BalancedShardRunner`` of a real ``segment`` / ``caption`` container (module
    docstring). Returns ``(runner, startup facts)``."""
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
        BalancedShardRunner,
        BalancedWorkerSettings,
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
    facts = preflight(env, store=store)
    mark("preflight")

    preset = get_preset(env.get("FLORA_PRESET", "balanced"))
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

    os.environ.update(cache_env(ENGINE_KINDS[kind]))  # before vLLM is imported
    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"],
                                                       WEIGHTS_MOUNT)))
    processor = nv.models.processor(ENGINE_KINDS[kind], model_dir)
    spec = spec_for(kind, str(model_dir), pool_gpu(kind, env))
    engine = OfflineEngine(spec)  # validated kwargs, plugin flag check, GDN dtype verify (M6)
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
        from flora_deploy.workers_impl.caption import CaptionEpisodeJob

        job = CaptionEpisodeJob(CaptionStage(nv, processor), info, inputs)
    active = apply_accel(nv)
    runner = BalancedShardRunner(store, engine, wrap_job(job, active=active, stamp=stamp),
                                 BalancedWorkerSettings(gpu=gpu), call_id=call_id, fatal=fatal)
    stats = prefetch.wait(timeout=0) if prefetch is not None else None
    startup = {"kind": kind, "preflight": facts, "marks": marks, "spec": spec.as_json(),
               "gdn_dtypes": engine.gdn_dtypes, "accel": active.provenance() if active else None,
               "derived": {n: {k: v for k, v in d.items() if k != "files"}
                           for n, d in derived.items()},
               "derived_files": {n: bool(f) for n, f in files.items()},
               "prefetch": stats.as_json() if stats else {"done": getattr(prefetch, "done", None)}}
    return runner, startup


class SegmentWorker:
    """``workers.ShardWorker`` for balanced ``segment`` shards (module docstring)."""

    @modal.enter()
    def start(self) -> None:
        from flora_deploy.modal_app.workers_app import _die

        self.runner, self.startup = build_runner(
            "segment", call_id=modal.current_function_call_id, fatal=_die)
        print(f"balanced segment worker ready: {json.dumps(self.startup, default=str)}",
              flush=True)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        return self.runner.run_shard(job_id, shard_id, config_hash, token=token)

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()


class CaptionWorker:
    """``workers.ShardWorker`` for balanced ``caption`` shards (module docstring)."""

    @modal.enter()
    def start(self) -> None:
        from flora_deploy.modal_app.workers_app import _die

        self.runner, self.startup = build_runner(
            "caption", call_id=modal.current_function_call_id, fatal=_die)
        print(f"balanced caption worker ready: {json.dumps(self.startup, default=str)}",
              flush=True)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        return self.runner.run_shard(job_id, shard_id, config_hash, token=token)

    @modal.exit()
    def stop(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.preempt()
