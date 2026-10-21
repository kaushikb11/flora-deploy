"""``flora-reference``: the fast ``exact`` worker on one H200 (``docs/exact-worker.md``).

- ``ReferenceWorker`` (``img_hfref``, ``H200``) links NVIDIA's SEGCAP code from the weights
  Volume, loads both models once through NVIDIA's ``Session.load`` (resident in a
  ``CachedSession``), fully offline, and runs NVIDIA's own entry point
  ``api.DataEnginePipeline().annotate`` per episode with every proven speedup of the ``exact``
  profile (``stages.reference.ExactProfile``; each one can be switched off with the ``disable``
  parameter and is recorded in provenance):

  - OPT-1 ``accel`` patches (decode_once, transport_pool, png_inmem, geometry_pool);
  - OPT-2 bit-identical BF16 Captioner (``accel.captioner_load``);
  - W3-B no-copy link-farm model cache, t=0 parallel weight prefetch, and (FP32 Captioner only)
    NVIDIA's manifest hash overlapped with the TempSeg load;
  - W3-A episode concurrency (default 4) under one GPU lock around NVIDIA's ``generate``;
  - Perf-1 ``accel.gpu_thread``: every NVIDIA ``generate`` runs on one persistent thread, so
    torch's thread-local cuDNN SDPA graph cache stays warm (fidelity-neutral, default on).

  Fix-X wiring: ``captioner_load.install(nv, background=True)`` before the TempSeg load (its
  default full hash overlaps the load; ``Session.load('captioner')`` joins it), each episode's
  ``accel`` scope entered in its own thread, ``nv.audit()`` before each batch, before each
  episode and after it (``workers_impl.segcap.FastExactEngine``), the D-1 determinism mode
  (``FLORA_DETERMINISM`` from the app factory, or the ``determinism`` parameter; default
  ``nvidia`` = change nothing), and the GPU lock printing ``FLORA_OBSERVE generate_output`` after
  every call so each episode's ``flora_reference.identity_calls`` feeds ``accel.identity``.

  ``run_shard`` is the ``workers.ShardWorker`` for ``segcap`` shards
  (``workers_impl.segcap.SegcapShardRunner`` over the job store from the container env, built
  on the first call; ``fatal=workers_app._die`` replaces the container after a CUDA fault,
  the contract's only exit; there is no timer-based kill switch). Deployed as a job worker (the
  factory set ``FLORA_STORE_*``), ``load`` first runs ``workers_app.preflight`` (store reachable,
  ``FLORA_MEDIA_ROOT`` readable, a fixed instruction has its text) and checks the hashed
  determinism mode (``FLORA_DETERMINISM`` / ``FLORA_OUTPUT_SETTINGS``), all before the model
  load. ``validate_shard`` runs it on golden episodes over a container-local
  store, then NVIDIA unmodified on a subset (P1a-5s validation, ``docs/exact-worker.md``).

  ``identity_proof`` runs the S1 acceptance gate in this one container: the fast composite on
  golden episodes, then NVIDIA's unmodified path (all swaps restored, a fresh NVIDIA ``Session``
  per episode, sequential), and compares NVIDIA's logged model-input digests.
- ``offline_check`` (CPU, network blocked): full prep-receipt validation of the Volume
  snapshots, the derived BF16 receipt, NVIDIA's offline resolution through the link farm.
- ``stage_samples`` (``img_cpu``): the WS-D smoke samples (``docs/reference-smoke.md``).

Run with ``uv run python scripts/reference_smoke.py ...`` (ephemeral ``app.run()``; never
deployed). Runtime is bounded by Modal's own ``startup_timeout`` + ``timeout`` (a timed-out input
is failed, not retried as a crash); there is no in-container kill switch.
"""

import os
import threading
import time
from pathlib import Path

import modal

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import (
    GOLDEN_MOUNT,
    GOLDEN_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    snapshot_path,
)
from flora_deploy.modal_app.images import cpu_image, hfref_image
from flora_deploy.modal_app.warm import linked_offline_env

IMPORTED_AT = time.time()  # module import in the container: proxy for "container started"

app = modal.App("flora-reference")
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)

GPU = "H200"
MIN_GPU_GIB = 120  # NVIDIA sizes the Captioner as total - 12 GiB, ignoring resident TempSeg
CPU_CORES = 8.0
MEMORY_MIB = 96 * 1024  # TempSeg base is built on CPU in BF16 (~18 GiB) before .to("cuda")
STARTUP_TIMEOUT_S = 5 * 60  # @modal.enter: layout + both loads (103 s measured in S1)
TIMEOUT_S = 25 * 60  # per call; with STARTUP_TIMEOUT_S one container lasts <= 30 min
SAMPLES_DIR = GOLDEN_MOUNT / "samples" / "wsd"
RUNS_DIR = GOLDEN_MOUNT / "reference_runs"
GOLDEN_MANIFEST = Path(GOLDEN_MOUNT) / "golden" / "v1" / "manifest.json"
MODEL_CACHE = Path(linked_offline_env()["FLORA_MODEL_CACHE"])
HUB_ROOT = Path(linked_offline_env()["HF_HOME"]).parent
PREFETCH_THREADS = 16


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _determinism_flags() -> dict:
    """Process-wide torch settings that decide whether identical inputs give identical outputs."""
    import importlib.util

    import torch

    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "optional_kernels": {m: importlib.util.find_spec(m) is not None
                             for m in ("fla", "causal_conv1d", "flash_attn", "triton")},
    }


def _gpu_memory() -> dict:
    import subprocess

    import torch

    info = {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        name, used, total, driver = [s.strip() for s in out.splitlines()[0].split(",")]
        info.update(nvidia_smi_used_mib=int(used), nvidia_smi_total_mib=int(total),
                    gpu_name=name, driver=driver)
    except Exception as e:  # noqa: BLE001 - diagnostics only
        info["nvidia_smi_error"] = repr(e)
    return info


def _commit_with_retry() -> str | None:
    """``golden.commit()``, retried once (review2-wsd L4). Returns an error string or None."""
    for attempt in range(2):
        try:
            golden.commit()
            return None
        except Exception as e:  # noqa: BLE001 - reported, never loses the computed result
            err = f"{type(e).__name__}: {e}"
            if attempt == 0:
                time.sleep(2)
    return err


class _EventWatch:
    """Collect this context's NVIDIA runtime events and fire ``callback(event)`` on each one."""

    def __init__(self, callback=None):
        self.events: list[dict] = []
        self.callback = callback

    def __call__(self, line: str) -> None:
        from flora_deploy.stages.reference import parse_runtime_line

        event = parse_runtime_line(line)
        if event is not None:
            self.events.append(event)
            if self.callback is not None:
                self.callback(event)


def _load_kind(session, kind: str, callback=None) -> tuple[dict, list[dict]]:
    import torch

    from flora_deploy.observe.stdout import route_stdout

    torch.cuda.reset_peak_memory_stats()
    watch = _EventWatch(callback)
    t = time.monotonic()
    with route_stdout(watch):
        session.load(kind)
    info = {"seconds": round(time.monotonic() - t, 1), "gpu_after": _gpu_memory(),
            "events": [{k: v for k, v in e.items() if k != "utc"} for e in watch.events
                       if e.get("event", "").startswith(("model_", "manifest_"))]}
    return info, watch.events


class ReferenceWorker:
    disable: str = modal.parameter(default="")  # ExactProfile toggles to switch off
    concurrency: int = modal.parameter(default=4)
    # D-1 (docs/determinism.md): "nvidia" (NVIDIA's behavior) or "deterministic"; "" = the
    # app factory's FLORA_DETERMINISM env, else "nvidia"
    determinism: str = modal.parameter(default="")

    @modal.enter()
    def load(self) -> None:
        t0 = time.monotonic()
        self.started = time.time()
        self.state = "loading"
        marks: dict[str, float] = {}

        def mark(name: str) -> None:
            marks[name] = round(time.monotonic() - t0, 2)

        # Before any CUDA work: apply() sets CUBLAS_WORKSPACE_CONFIG itself while CUDA is not yet
        # initialized, so the class env stays NVIDIA's default for the "nvidia" mode.
        from flora_deploy.accel import determinism

        # review 4 M2: the deployed app passes the mode only through FLORA_DETERMINISM (the
        # spawner instantiates the class without parameters); a parameter that disagrees with
        # the env is refused rather than silently winning.
        env_mode = os.environ.get("FLORA_DETERMINISM")
        if self.determinism and env_mode and self.determinism != env_mode:
            raise determinism.DeterminismError(
                f"determinism parameter {self.determinism!r} != FLORA_DETERMINISM {env_mode!r}")
        self.determinism_mode = self.determinism or env_mode or determinism.NVIDIA
        self.runner, self.preflight, self.job_store = None, None, None
        self._runner_lock = threading.Lock()  # @modal.concurrent inputs share one runner
        if job_worker_env():
            # workers.py startup: cheap preflight (store reachable, media root readable, fixed
            # instruction set, hashed determinism mode agrees) BEFORE the ~2 min model load
            from flora_deploy.modal_app.workers_app import preflight, store_from_env
            from flora_deploy.workers_impl.reward import PreflightError

            try:
                self.job_store = store_from_env()
            except Exception as exc:  # noqa: BLE001 - a misconfigured deployment: say so, early
                raise PreflightError(f"job store: {exc}") from exc
            self.preflight = preflight(os.environ, store=self.job_store)
            hashed = configured_determinism()
            if hashed != self.determinism_mode:
                raise determinism.DeterminismError(
                    f"determinism mode {self.determinism_mode!r} != the job's hashed mode "
                    f"{hashed!r} (FLORA_OUTPUT_SETTINGS)")
            mark("preflight")
        self.determinism_record = determinism.apply(self.determinism_mode)

        from flora_deploy.stages.reference import (
            ExactProfile,
            build_offline_layout,
            check_nv_models,
            check_prep_receipt,
        )

        self.profile = profile = ExactProfile.from_spec(
            self.disable, episode_concurrency(default=self.concurrency))
        # 1. model-cache layout (+ full prep-receipt validation, review2-wsd M5)
        if profile.linked_layout:
            from flora_deploy.modal_app.warm import build_linked_layout

            layout = build_linked_layout(weights_root=WEIGHTS_MOUNT, hub_root=HUB_ROOT)
            layout_stats = layout.stats
        else:
            kinds = ("tempseg",) if profile.captioner_bf16 else ("tempseg", "captioner")
            layout = build_offline_layout(HUB_ROOT, WEIGHTS_MOUNT, kinds=kinds,
                                          model_cache=MODEL_CACHE)
            layout_stats = layout.copies
        # The base has no NVIDIA manifest: our receipt is its only integrity anchor. On a
        # legacy (schema-1, no fingerprints) receipt every base byte is re-hashed against it,
        # in the background while the models load; the worker is not ready until it passes.
        receipts, base_check = {}, {}
        for key, src in layout.sources.items():
            if key != "tempseg_base":
                receipts[key] = check_prep_receipt(PINS_BY_KEY[key], Path(src))

        def check_base():
            try:
                base_check["report"] = check_prep_receipt(
                    PINS_BY_KEY["tempseg_base"], Path(layout.sources["tempseg_base"]),
                    hash_files="all")
            except BaseException as e:  # noqa: BLE001 - re-raised before ready
                base_check["error"] = e

        base_thread = threading.Thread(target=check_base, daemon=True, name="flora-base-check")
        base_thread.start()
        mark("layout")

        from flora_deploy.accel.captioner_load import default_derived_dir

        # 2. t=0 prefetch of the weights this load reads (base, then the Captioner)
        from flora_deploy.modal_app.warm import BackgroundPrefetch, snapshot_files

        self.derived_dir = default_derived_dir(WEIGHTS_MOUNT)
        cap_dir = self.derived_dir if profile.captioner_bf16 else layout.sources["captioner"]
        self.prefetch = None
        if profile.prefetch:
            files = [*snapshot_files(layout.sources["tempseg_base"]), *snapshot_files(cap_dir)]
            self.prefetch = BackgroundPrefetch(files, threads=PREFETCH_THREADS)
        mark("prefetch_started")

        import huggingface_hub.constants as hfc
        import torch

        if not hfc.HF_HUB_OFFLINE:
            raise RuntimeError("HF_HUB_OFFLINE must be on before huggingface_hub loads")
        if Path(hfc.HF_HUB_CACHE) != layout.hub_cache:
            raise RuntimeError(f"HF_HUB_CACHE {hfc.HF_HUB_CACHE} != {layout.hub_cache}")
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gib < MIN_GPU_GIB:  # review2-wsd L6: two resident models need the H200
            raise RuntimeError(f"GPU has {total_gib:.0f} GiB; the resident exact worker needs "
                               f">= {MIN_GPU_GIB} GiB")

        # 3. NVIDIA's code, models.json == pins, OPT-2
        from flora_deploy.nv import CachedSession, Family, load_family

        code_dir = Path(snapshot_path(PINS_BY_KEY["tempseg_code"], WEIGHTS_MOUNT))
        self.nv = nv = load_family(Family.SEGCAP, code_dir)
        check_nv_models(nv.config.MODELS)
        if Path(nv.config.MODEL_CACHE) != layout.model_cache:
            raise RuntimeError(f"MODEL_CACHE {nv.config.MODEL_CACHE} != {layout.model_cache}")
        self.originals = self._nvidia_attributes()
        self.bf16 = None
        if profile.captioner_bf16:
            from flora_deploy.accel import captioner_load

            # Fix-X: cheap checks + toolchain now; the default full hash of every byte runs in
            # the background, overlapped with the TempSeg load; Session.load('captioner') joins
            # it and fails closed if it failed.
            self.bf16 = captioner_load.install(nv, self.derived_dir, background=True)
        mark("linked")

        # 4. TempSeg, with NVIDIA's FP32 Captioner hash overlapped (W3-B) if it is needed
        self.session = CachedSession.from_family(nv)
        self.hash_thread, hash_info = None, {}

        def start_hash(event: dict) -> None:
            if (event.get("event") == "model_assets_verified" and event.get("kind") == "tempseg"
                    and self.hash_thread is None):
                def run():
                    t = time.monotonic()
                    nv.models.validate_manifest(nv.models.snapshot("captioner"))
                    hash_info["seconds"] = round(time.monotonic() - t, 1)
                    hash_info["started_at_s"] = round(t - t0, 1)

                self.hash_thread = threading.Thread(target=run, daemon=True,
                                                    name="flora-captioner-hash")
                self.hash_thread.start()

        overlap = profile.overlap_hash and not profile.captioner_bf16
        loads = {}
        loads["tempseg"], _ = _load_kind(self.session, "tempseg", start_hash if overlap else None)
        mark("tempseg_loaded")
        t = time.monotonic()
        if self.hash_thread is not None:
            self.hash_thread.join()
        loads["captioner"], _ = _load_kind(self.session, "captioner")
        loads["captioner"]["wait_for_hash_s"] = round(time.monotonic() - t
                                                      - loads["captioner"]["seconds"], 1)
        mark("captioner_loaded")

        base_thread.join()
        if "error" in base_check:
            raise base_check["error"]
        receipts["tempseg_base"] = base_check["report"]
        mark("base_verified")
        if self.bf16 is not None:
            reports = self.bf16.wait()  # already joined by the Captioner load; re-raises
            receipts["captioner_bf16"] = [
                {k: r.get(k) for k in ("mode", "files", "bytes", "seconds", "background")}
                for r in reports]
            mark("bf16_verified")

        # 5. runtime swaps: GPU lock, NVIDIA's entry point bound to our session, OPT-1 patches
        self._install_fast()
        mark("patched")
        self.torch_threads_after_load = torch.get_num_threads()
        prefetch = None
        if self.prefetch is not None:
            stats = self.prefetch.wait(0) if self.prefetch.done else None
            prefetch = stats.as_json() if stats else {"still_running": True}
        self.state = "fast"
        self.load_report = {
            "profile": profile.provenance(),
            "determinism_mode": self.determinism_record,
            "container_import_to_enter_s": round(self.started - IMPORTED_AT, 2),
            "enter_s": round(time.monotonic() - t0, 1),
            "marks_s": marks,
            "layout": layout_stats,
            "receipts": receipts,
            "prefetch": prefetch,
            "captioner_hash": hash_info or None,
            "bf16_receipt": self.bf16.verifications[0] if self.bf16 else None,
            "captioner_bf16": self.bf16.provenance() if self.bf16 else None,  # opt-outs visible
            "gpu_thread": self.gpu_thread.provenance() if self.gpu_thread else None,
            "loads": loads,
            "loaded_kinds": list(self.session.loaded_kinds),
            "torch": torch.__version__,
            "torch_threads": self.torch_threads_after_load,
            "cpu_count": os.cpu_count(),
            "determinism": _determinism_flags(),  # live flags after load
            "cgroup_cpu_max": _read_text("/sys/fs/cgroup/cpu.max"),
            "cgroup_memory_max": _read_text("/sys/fs/cgroup/memory.max"),
            "gpu": _gpu_memory(),
        }
        print("exact worker ready:", {k: self.load_report[k] for k in ("enter_s", "marks_s")},
              flush=True)

    # -- swaps -----------------------------------------------------------------------------
    def _nvidia_attributes(self) -> dict:
        nv = self.nv
        return {"pipeline.Session": nv.pipeline.Session, "pipeline.generate": nv.pipeline.generate,
                "api._REQUEST_LOCK": nv.api._REQUEST_LOCK,
                "models.snapshot": nv.models.snapshot,
                "models.validate_manifest": nv.models.validate_manifest,
                "pipeline.frames_for": nv.pipeline.frames_for,
                "media.frames_for": nv.media.frames_for,
                "pipeline.human_transport": nv.pipeline.human_transport,
                "pipeline.caption_single": nv.pipeline.caption_single,
                "pipeline.caption_three": nv.pipeline.caption_three,
                "pipeline.Predictor.predict": nv.pipeline.Predictor.predict,
                **self._accel_targets()}

    def _accel_targets(self) -> dict:
        """Every attribute an ``accel`` patch can replace (review 4 L1: incl. the vendor
        ``robot_caption_geometry`` / ``unified_input`` targets), from the registry itself."""
        from flora_deploy.accel.registry import registered, resolve_target

        out = {}
        for patch in registered().values():
            for target in patch.targets:
                owner, attr = resolve_target(self.nv, target)
                out[target] = getattr(owner, attr)
        return out

    def _install_fast(self) -> None:
        from flora_deploy.accel import apply_patches, gpu_thread
        from flora_deploy.observe import Patcher, TimedLock
        from flora_deploy.stages.reference import (
            GPU_LOCK_NAME,
            ProcessorClones,
            install_gpu_lock,
            install_nvidia_entry,
        )
        from flora_deploy.workers_impl.segcap import FastExactEngine

        self.gpu_lock = TimedLock(GPU_LOCK_NAME)
        self.patcher = Patcher()
        # Perf-1: innermost layer (its own Patcher: one Patcher patches an attribute once), so
        # the lock is taken in the episode thread and the call itself runs on the GPU thread.
        self.gt_patcher = Patcher()
        self.gpu_thread = (gpu_thread.install(self.nv, self.gt_patcher)
                           if self.profile.gpu_thread else None)
        install_gpu_lock(self.nv, self.gpu_lock, self.patcher, emit_output=True)
        install_nvidia_entry(self.nv, self.patcher,
                             request_lock=self.profile.concurrency <= 1)
        self.accel = apply_patches(self.nv, self.profile.accel) if self.profile.accel else None
        self.clones = ProcessorClones()
        self.engine = FastExactEngine(
            self.nv, self.session, gpu_lock=self.gpu_lock, clones=self.clones,
            accel=self.accel, gpu_thread=self.gpu_thread, concurrency=self.profile.concurrency,
            profile=self.profile.provenance(), determinism=self.determinism_record,
            image_fingerprint=_image_fingerprint(), captioner_bf16=self.bf16,
            progress=lambda ep, frac, desc="": print(f"[{ep}] {frac:.0%} {desc}", flush=True))

    def _restore_nvidia(self) -> dict:
        """Undo every swap and release the resident models: NVIDIA's unmodified state."""
        import gc

        import torch

        if self.accel is not None:
            self.accel.restore()
        self.patcher.restore()
        self.gt_patcher.restore()
        if self.bf16 is not None:
            self.bf16.uninstall()
        self.session.close_all()
        gc.collect()
        torch.cuda.empty_cache()
        now = self._nvidia_attributes()
        changed = [k for k, v in now.items() if v is not self.originals[k]]
        if changed:
            raise RuntimeError(f"NVIDIA attributes not restored: {changed}")
        self.state = "unmodified"
        return {"restored": sorted(now), "gpu_after_release": _gpu_memory()}

    # -- methods ---------------------------------------------------------------------------
    @modal.method()
    def report(self) -> dict:
        return {**self.load_report, "state": self.state, "gpu_now": _gpu_memory(),
                "container_age_s": round(time.time() - self.started, 1)}

    def _batch(self, episodes, concurrency: int, *, fast: bool, measure: bool = False) -> dict:
        import torch

        from flora_deploy.observe import Sampler
        from flora_deploy.stages.reference import annotate_batch

        torch.cuda.reset_peak_memory_stats()
        out = annotate_batch(
            self.nv, self.session if fast else None, episodes,
            concurrency=concurrency if fast else 1,
            gpu_lock=self.gpu_lock if fast else None, measure=measure,
            clones=self.clones if fast else None, accel=self.accel if fast else None,
            sampler=Sampler(interval_s=1.0),
            progress=lambda ep, frac, desc="": print(f"[{ep}] {frac:.0%} {desc}", flush=True),
        )
        out["batch"]["gpu"] = _gpu_memory()
        if fast:
            out["batch"]["profile"] = self.profile.provenance()
            if self.gpu_thread is not None:
                out["batch"]["gpu_thread"] = self.gpu_thread.provenance()
        return out

    def _persist(self, out: dict, run: str) -> None:
        from flora_deploy.accel import determinism
        from flora_deploy.stages.reference import persist_result, result_digest

        try:  # D-1: the mode must still be in effect before anything is saved
            determinism.verify(self.determinism_record)
            drift = None
        except determinism.DeterminismError as e:
            if self.determinism_record["mode"] != determinism.NVIDIA:
                raise
            drift = str(e)  # NVIDIA's default mode sets nothing; report, never drop results
        out["digests"] = {}
        for name, result in out["results"].items():
            ref = result["flora_reference"]
            ref["determinism"] = {k: self.determinism_record[k] for k in ("schema", "mode")}
            ref.setdefault("gpu_thread", self.gpu_thread.provenance()
                           if self.gpu_thread is not None and self.state == "fast" else None)
            if drift:
                ref["determinism"]["flags_changed_since_enter"] = drift
            ref["run_id"] = run
            ref["digest"] = out["digests"][name] = result_digest(result)
            try:
                path = persist_result(result, golden_root=GOLDEN_MOUNT, run_id=run,
                                      episode=name, commit=lambda: None)
                ref["stored_at"] = f"{GOLDEN_VOLUME}:{path.relative_to(GOLDEN_MOUNT)}"
            except Exception as e:  # noqa: BLE001 - keep the result, report the failure
                ref["stored_at"], ref["store_error"] = None, f"{type(e).__name__}: {e}"
        out["commit_error"] = _commit_with_retry()

    @modal.method()
    def annotate_batch(self, episodes: list[dict], run_id: str, *,
                       concurrency: int | None = None, measure: bool = False,
                       tag: str = "fast", persist: bool = True) -> dict:
        """Fast ``exact`` path over ``episodes`` (``stages.reference.annotate_batch`` items)."""
        from flora_deploy.stages.reference import safe_name

        if self.state != "fast":
            raise RuntimeError(f"worker is in state {self.state!r}, not 'fast'")
        golden.reload()
        out = self._batch(episodes, concurrency or self.profile.concurrency, fast=True,
                          measure=measure)
        if persist:
            self._persist(out, safe_name(f"{run_id}-{tag}"))
        return out

    @modal.method()
    def annotate(self, videos: dict[str, str], profile: str, *, episode: str, run_id: str,
                 long_windows: bool = False, instruction: str = "") -> dict:
        """One episode on the fast path. NVIDIA's exceptions arrive as
        ``stages.reference.NvidiaError`` (picklable; original type name + message)."""
        from flora_deploy.stages.reference import NvidiaError, safe_name

        if self.state != "fast":
            raise RuntimeError(f"worker is in state {self.state!r}, not 'fast'")
        golden.reload()
        ep = {"episode": safe_name(episode), "videos": videos, "profile": profile,
              "long_windows": long_windows, "instruction": instruction}
        out = self._batch([ep], 1, fast=True)
        if out["errors"]:
            type_name, _, message = out["errors"][ep["episode"]].partition(": ")
            raise NvidiaError(type_name, message)
        self._persist(out, safe_name(run_id))
        return out["results"][ep["episode"]]

    @modal.method()
    def identity_proof(self, ids: list[str], run_id: str, *, deadline_s: float = 16 * 60,
                       instruction: bool = True) -> dict:
        """S1 gate: fast composite vs NVIDIA unmodified on golden episodes ``ids``.

        1. Fast composite (all swaps, ``profile.concurrency`` threads), longest first.
        2. Restore every swap, release the models (``_restore_nvidia``).
        3. NVIDIA unmodified: each episode through ``DataEnginePipeline().annotate`` with NVIDIA's
           own fresh ``Session()`` (loads TempSeg, then the FP32 Captioner, per episode),
           sequential, shortest first, while ``deadline_s`` (from the call start) allows.
        4. ``identity_summary``: NVIDIA-logged model-input digests per call, outputs.
        Full results stay on ``flora-golden:/reference_runs/<run_id>-{fast,nvidia}/``; the
        return value holds numbers and short digests only."""
        import json

        from flora_deploy.modal_app.warm import BackgroundPrefetch, snapshot_files
        from flora_deploy.stages.reference import (
            golden_episode_inputs,
            identity_summary,
            safe_name,
        )

        if self.state != "fast":
            raise RuntimeError(f"worker is in state {self.state!r}, not 'fast'")
        t0 = time.monotonic()
        golden.reload()
        manifest = json.loads(GOLDEN_MANIFEST.read_text())
        episodes = golden_episode_inputs(manifest, ids, GOLDEN_MOUNT, instruction=instruction)

        # NVIDIA's path will hash the FP32 Captioner once (its _VERIFIED_MANIFESTS then reuses
        # it per episode). Do that, and warm the page cache, while the fast batch runs.
        bg: dict = {}

        def warm_fp32():
            t = time.monotonic()
            src = Path(snapshot_path(PINS_BY_KEY["captioner"], WEIGHTS_MOUNT))
            stats = BackgroundPrefetch(snapshot_files(src), threads=PREFETCH_THREADS).wait()
            bg["prefetch"] = stats.as_json() if stats else None
            models = self.nv.models
            validate = self.bf16.original_validate if self.bf16 else models.validate_manifest
            nv_snapshot = self.bf16.original_snapshot if self.bf16 else models.snapshot
            validate(nv_snapshot("captioner"))
            bg["hash_done_s"] = round(time.monotonic() - t, 1)

        warm = threading.Thread(target=warm_fp32, daemon=True, name="flora-fp32-warm")
        if self.bf16 is not None:
            warm.start()
        fast_eps = sorted(episodes, key=lambda e: -e["duration_s"])
        fast = self._batch(fast_eps, self.profile.concurrency, fast=True)
        self._persist(fast, safe_name(f"{run_id}-fast"))
        t_fast = time.monotonic() - t0
        restored = self._restore_nvidia()
        if warm.is_alive():
            warm.join()

        nvidia = {"results": {}, "errors": {}, "episodes": {}, "skipped": []}
        # Shortest episode of each pool first, then the rest: the deadline cuts coverage last.
        seen: set[str] = set()
        order = []
        for ep in sorted(episodes, key=lambda e: e["duration_s"]):
            order.append((ep["pool"] in seen, ep["duration_s"], len(order), ep))
            seen.add(ep["pool"])
        for *_, ep in sorted(order, key=lambda x: x[:3]):
            elapsed = time.monotonic() - t0
            estimate = 60 + 5 * ep["duration_s"]  # fresh loads + unpatched CPU work
            if elapsed + estimate > deadline_s:
                nvidia["skipped"].append(ep["episode"])
                continue
            out = self._batch([ep], 1, fast=False)
            self._persist(out, safe_name(f"{run_id}-nvidia"))  # each one, as soon as it is done
            nvidia["results"].update(out["results"])
            nvidia["errors"].update(out["errors"])
            nvidia["episodes"][ep["episode"]] = {"wall_s": out["batch"]["wall_s"]}
            nvidia["commit_error"] = out["commit_error"]
        nvidia.setdefault("commit_error", None)
        identity = identity_summary(nvidia["results"], fast["results"])

        def rows(results):
            from flora_deploy.stages.reference import summary_row

            return {name: {**summary_row(name, r),
                           "timings_s": r["flora_reference"]["timings_s"],
                           "end_s": r["flora_reference"].get("batch", {}).get("end_s"),
                           "calls": len(r["flora_reference"]["model_calls"])}
                    for name, r in results.items()}

        return {
            "run_id": run_id,
            "episodes": [{k: e[k] for k in ("episode", "id", "pool", "profile", "duration_s")}
                         | {"instruction": bool(e["instruction"])} for e in episodes],
            "identity": identity,
            "fast": {"batch": {k: v for k, v in fast["batch"].items() if k != "calls"},
                     "errors": fast["errors"], "episodes": rows(fast["results"]),
                     "commit_error": fast["commit_error"]},
            "nvidia": {"errors": nvidia["errors"], "skipped": nvidia["skipped"],
                       "episodes": rows(nvidia["results"]), "walls": nvidia["episodes"],
                       "commit_error": nvidia["commit_error"]},
            "restored": restored,
            "fp32_warm": bg,
            "fast_wall_s": round(t_fast, 1),
            "call_s": round(time.monotonic() - t0, 1),
            "container_age_s": round(time.time() - self.started, 1),
        }

    def _shard_runner(self, store=None, settings=None):
        from flora_deploy.workers_impl.segcap import SegcapShardRunner

        if self.state != "fast":
            raise RuntimeError(f"worker is in state {self.state!r}, not 'fast'")
        from flora_deploy.modal_app.workers_app import _die, store_from_env

        if store is None:
            store = self.job_store if self.job_store is not None else store_from_env()
        if settings is None:
            settings = segcap_settings_from_env()
        # CUDA fault -> _die after the marker is written (the container is replaced)
        return SegcapShardRunner(store, self.engine, settings,
                                 call_id=modal.current_function_call_id, fatal=_die)

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        """``flora_deploy.workers.ShardWorker`` for ``segcap`` shards
        (``workers_impl.segcap.SegcapShardRunner``; the job store comes from the container env
        the app factory sets: ``FLORA_STORE_KIND`` / ``FLORA_STORE_ROOT`` / ..., checked by the
        preflight in ``load``). One runner per container, created once under a lock; it refuses
        a second concurrent shard (``ConcurrentShardRefused``: ``max_inputs`` must be 1)."""
        with self._runner_lock:
            if self.runner is None:
                self.runner = self._shard_runner()
        return self.runner.run_shard(job_id, shard_id, config_hash, token=token)

    @modal.method()
    def validate_shard(self, ids: list[str], run_id: str, *, nvidia_ids: list[str] | None = None,
                       compare_to: str | None = None, deadline_s: float = 19 * 60,
                       nvidia_s_per_video_s: float = 14.0) -> dict:
        """P1a-5s GPU validation in this one container (``docs/exact-worker.md``).

        1. ``run_shard`` of one ``segcap`` shard over golden episodes ``ids`` (longest first),
           through the production runner and engine, on a container-local job store whose index
           receipts point at the golden files: throughput and per-call generation times.
        2. Restore every swap; NVIDIA unmodified (fresh ``Session`` per episode, sequential,
           shortest first) on ``nvidia_ids`` while ``deadline_s`` allows.
        3. ``accel.identity.compare_runs`` of NVIDIA unmodified against this run's fast records
           and, with ``compare_to`` (an earlier run id), against that run's stored fast results.
        Results go to ``flora-golden:/reference_runs/<run_id>-{fast,nvidia}/``; numbers only are
        returned."""
        import json

        from flora_deploy.accel.identity import compare_runs as identity_compare
        from flora_deploy.jobs import progress as P
        from flora_deploy.jobs.attempts import JobFiles
        from flora_deploy.jobs.manifest import save_manifest
        from flora_deploy.jobs.store import JobPaths
        from flora_deploy.jobs.stores import LocalStore
        from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard, episode_uid
        from flora_deploy.stages.reference import (
            ModeMismatch,
            check_same_mode,
            golden_episode_inputs,
            golden_name,
            identity_calls_from_model_calls,
            identity_summary,
            safe_name,
        )

        if self.state != "fast":
            raise RuntimeError(f"worker is in state {self.state!r}, not 'fast'")
        t0 = time.monotonic()
        golden.reload()
        manifest = json.loads(GOLDEN_MANIFEST.read_text())
        episodes = golden_episode_inputs(manifest, ids, GOLDEN_MOUNT)
        episodes.sort(key=lambda e: -e["duration_s"])

        # 1. a one-shard job on a container-local store
        job, cfg = safe_name(f"val-{run_id}")[:60], "cfgvalidate"
        store = LocalStore(f"/tmp/flora-validate/{job}")
        paths = JobPaths(job)
        refs, names = [], {}
        for e in episodes:
            pool, ds, slug = e["id"].split("/", 2)
            dataset = f"golden/{pool}/{ds}"
            uid = episode_uid(dataset, slug)
            names[uid] = e
            refs.append(EpisodeRef(uid=uid, dataset=dataset, episode_id=slug, pool=e["pool"],
                                   duration_s=e["duration_s"], oracle_eligible=True,
                                   envelope_ok=True, index_ref=paths.index(uid)))
            store.put_bytes(paths.index(uid), json.dumps({
                "episode_id": slug, "dataset": dataset, "task": e["instruction"],
                "profile": e["profile"], "yam32": e["long_windows"],
                "streams": {role: {"source": {"uri": path, "start_s": None, "end_s": None}}
                            for role, path in e["videos"].items()},
                "envelope": {"ok": True, "reasons": []}}).encode())
        shard = Shard("segcap-00000", "segcap", tuple(r.uid for r in refs),
                      sum(r.duration_s for r in refs))
        save_manifest(store, Manifest(job, cfg, tuple(refs), (shard,),
                                      time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())))
        from flora_deploy.modal_app.workers_app import gpu_name

        runner = self._shard_runner(store, segcap_settings_from_env(
            gpu=gpu_name(), container_id="validate", local_dir=f"/tmp/flora-validate/{job}-l",
            determinism=self.determinism_mode))
        # NVIDIA's unmodified arm hashes the FP32 Captioner once (_VERIFIED_MANIFESTS reuses it):
        # do that, and warm the page cache, while the shard runs (as identity_proof does).
        bg: dict = {}

        def warm_fp32():
            from flora_deploy.modal_app.warm import BackgroundPrefetch, snapshot_files

            t = time.monotonic()
            src = Path(snapshot_path(PINS_BY_KEY["captioner"], WEIGHTS_MOUNT))
            stats = BackgroundPrefetch(snapshot_files(src), threads=PREFETCH_THREADS).wait()
            bg["prefetch"] = stats.as_json() if stats else None
            models = self.nv.models
            validate = self.bf16.original_validate if self.bf16 else models.validate_manifest
            nv_snapshot = self.bf16.original_snapshot if self.bf16 else models.snapshot
            validate(nv_snapshot("captioner"))
            bg["hash_done_s"] = round(time.monotonic() - t, 1)

        warm = threading.Thread(target=warm_fp32, daemon=True, name="flora-fp32-warm")
        if nvidia_ids and self.bf16 is not None:
            warm.start()
        gt0 = self.gpu_thread.provenance() if self.gpu_thread else None
        t_shard = time.monotonic()
        receipt = runner.run_shard(job, shard.shard_id, cfg)
        shard_wall = time.monotonic() - t_shard
        audit_prefix = paths.audit(cfg, "segcap", shard.shard_id, "x", 0)[:-len("x-0.jsonl")]
        fast: dict[str, dict] = {}
        for p in store.list(audit_prefix):
            for line in store.get_bytes(p).decode().splitlines():
                if line.strip():
                    a = json.loads(line)
                    fast[golden_name(names[a["uid"]]["id"])] = a["result"]
        jf = JobFiles(store, job, cfg)
        records = sum(len(P.read_part(jf, p)) for p in receipt.parts)
        self._persist({"results": fast}, safe_name(f"{run_id}-fast"))

        per_call = {"tempseg": [], "captioner": []}
        gen_s = 0.0
        for r in fast.values():
            g = r["flora_reference"]["generation"]
            gen_s += g["segment"]["generation_s"] + g["caption"]["generation_s"]
            for kind, key in (("tempseg", "segment"), ("captioner", "caption")):
                if g[key]["calls"]:
                    per_call[kind].append(g[key]["generation_s"] / g[key]["calls"])
        video_s = sum(float(r.get("duration_s", 0.0)) for r in fast.values())
        fast_summary = {
            "receipt": {"done": len(receipt.episodes_done), "failed": len(receipt.episodes_failed),
                        "parts": len(receipt.parts), "records": records,
                        "wall_seconds": receipt.wall_seconds},
            "shard_wall_s": round(shard_wall, 1), "video_s": round(video_s, 2),
            "generation_s": round(gen_s, 1),
            "gpu_s_per_video_s": round(gen_s / video_s, 3) if video_s else None,
            "calls": sum(len(r["flora_reference"]["model_calls"]) for r in fast.values()),
            "mean_s_per_call": {k: round(sum(v) / len(v), 2) if v else None
                                for k, v in per_call.items()},
            "batch": {k: v for k, v in (runner.last_batch or {}).items() if k != "calls"},
            "gpu_thread_before": gt0,
            "gpu_thread_after": self.gpu_thread.provenance() if self.gpu_thread else None,
            "episodes": {name: {"duration_s": r.get("duration_s"),
                                "total_s": r["flora_reference"]["timings_s"]["total"],
                                "calls": len(r["flora_reference"]["model_calls"]),
                                "outputs_logged": sum(c.get("output_sha256") is not None for c in
                                                      r["flora_reference"]["identity_calls"]),
                                "generation": r["flora_reference"]["generation"]}
                         for name, r in fast.items()},
        }
        t_fast = time.monotonic() - t0

        # 2. NVIDIA unmodified
        restored = self._restore_nvidia()
        if warm.is_alive():
            warm.join()
        nvidia = {"results": {}, "errors": {}, "walls": {}, "skipped": []}
        nv_eps = [e for e in episodes if e["id"] in set(nvidia_ids or ())]
        for ep in sorted(nv_eps, key=lambda e: e["duration_s"]):
            elapsed = time.monotonic() - t0
            if elapsed + 90 + nvidia_s_per_video_s * ep["duration_s"] > deadline_s:
                nvidia["skipped"].append(ep["episode"])
                continue
            out = self._batch([ep], 1, fast=False)
            self._persist(out, safe_name(f"{run_id}-nvidia"))
            nvidia["results"].update(out["results"])
            nvidia["errors"].update(out["errors"])
            nvidia["walls"][ep["episode"]] = {"wall_s": out["batch"]["wall_s"],
                                              "video_s": ep["duration_s"]}

        # 3. identity (accel.identity, NVIDIA's logged input digests; outputs paired from results)
        def calls(results):
            return {n: identity_calls_from_model_calls(r["flora_reference"]["model_calls"])
                    for n, r in results.items()}

        def compare(name: str, other: dict) -> None:
            # every NVIDIA-arm episode must be present (missing -> violation, M1 / L5)
            try:
                check_same_mode(nvidia["results"], other)  # M2: never across modes
            except ModeMismatch as e:
                identity[name] = {"ok": False, "paired": 0, "episodes": {}, "refused": str(e)}
                return
            identity[name] = identity_compare(calls(nvidia["results"]), calls(other))
            identity["summary_" + name] = identity_summary(nvidia["results"], other)

        identity: dict = {}
        compare("vs_this_run", {n: r for n, r in fast.items() if n in nvidia["results"]})
        if compare_to:
            d = Path(RUNS_DIR / f"{compare_to}-fast")
            compare("vs_" + compare_to, {
                n: json.loads((d / f"{n}.json").read_text()) for n in nvidia["results"]
                if (d / f"{n}.json").is_file()})
        for v in identity.values():  # numbers only: drop the per-call digests of violations
            for row in v.get("episodes", {}).values():
                for viol in row.get("violations", []):
                    viol.pop("a", None)
                    viol.pop("b", None)
        return {
            "run_id": run_id, "ids": ids, "fast": fast_summary, "fast_wall_s": round(t_fast, 1),
            "nvidia": {"errors": nvidia["errors"], "skipped": nvidia["skipped"],
                       "walls": nvidia["walls"]},
            "identity": identity, "restored": restored, "fp32_warm": bg,
            "call_s": round(time.monotonic() - t0, 1),
            "container_age_s": round(time.time() - self.started, 1),
        }

    @modal.exit()
    def close(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:  # flush + mark_preempted (no-op when no shard is in flight)
            runner.preempt()
        session = getattr(self, "session", None)
        if session is not None:
            session.close_all()


def _image_fingerprint() -> str | None:
    try:
        from flora_deploy.modal_app.images import image_fingerprint

        return image_fingerprint("img_hfref")
    except Exception:  # noqa: BLE001 - provenance only; the config hash covers the image
        return None


def segcap_settings_from_env(env=None, **overrides):
    """``SegcapWorkerSettings`` from the container env the app factory sets (``FLORA_PRESET``,
    ``FLORA_ENVELOPE``, ``FLORA_MEDIA_ROOT``, ``FLORA_INSTRUCTION_TEXT``) and the pins."""
    from flora_deploy.workers_impl.segcap import SegcapWorkerSettings, scheduling_from_env

    env = os.environ if env is None else env
    overrides = {**scheduling_from_env(env), **overrides}  # G3 longest-first (FLORA_SEGCAP_...)
    if "gpu" not in overrides:
        from flora_deploy.modal_app.workers_app import gpu_name

        overrides["gpu"] = gpu_name()
    keys = ("tempseg", "tempseg_code", "tempseg_base", "captioner")
    return SegcapWorkerSettings(**{
        "preset": env.get("FLORA_PRESET", "exact"),
        "models": {k: PINS_BY_KEY[k].revision for k in keys},
        "envelope": env.get("FLORA_ENVELOPE", "warn"),
        "media_root": env.get("FLORA_MEDIA_ROOT"),
        "instruction_text": env.get("FLORA_INSTRUCTION_TEXT") or None,
        "determinism": configured_determinism(env),
        **overrides})


def configured_determinism(env=None) -> str:
    """The D-1 mode the job was configured and hashed with (``RunSettings.determinism``):
    ``FLORA_DETERMINISM``, which must agree with ``FLORA_OUTPUT_SETTINGS["determinism"]`` when
    both are set (refused otherwise); ``nvidia`` when neither is."""
    import json

    from flora_deploy.accel import determinism

    env = os.environ if env is None else env
    mode = env.get("FLORA_DETERMINISM") or None
    outputs = env.get("FLORA_OUTPUT_SETTINGS")
    hashed = json.loads(outputs).get("determinism") if outputs else None
    if mode and hashed and mode != hashed:
        raise determinism.DeterminismError(
            f"FLORA_DETERMINISM {mode!r} != FLORA_OUTPUT_SETTINGS determinism {hashed!r}")
    return determinism.check_mode(mode or hashed or determinism.NVIDIA)


def job_worker_env(env=None) -> bool:
    """Deployed as a job worker (the app factory set a job store), not the smoke app."""
    env = os.environ if env is None else env
    return bool(env.get("FLORA_STORE_KIND") or env.get("FLORA_STORE"))


def episode_concurrency(env=None, *, default: int = 4) -> int:
    """Episodes the engine runs at once inside one shard: the planned
    ``resources.segcap.episode_concurrency`` the app factory passes as
    ``FLORA_EPISODE_CONCURRENCY`` (review 6 H2: plan, factory and worker agree), else
    ``default`` (the ``concurrency`` parameter of the smoke / lab apps)."""
    from flora_deploy.jobs.settings import EPISODE_CONCURRENCY_ENV

    env = os.environ if env is None else env
    raw = env.get(EPISODE_CONCURRENCY_ENV)
    if not raw:
        return int(default)
    value = int(raw)
    if value < 1:
        raise ValueError(f"{EPISODE_CONCURRENCY_ENV}={raw!r}: must be >= 1")
    return value


# The smoke app's binding. ``ReferenceWorker`` itself stays undecorated at module scope so the
# app factory (``modal_app.factory.build_app``, PLAN.md D10) can decorate it for a deployed job
# app too; a container finds it by name and resolves its app by the deployed app's name.
ReferenceWorkerSmoke = app.cls(
    image=hfref_image(),
    gpu=GPU,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(GOLDEN_MOUNT): golden},
    env=linked_offline_env(),
    timeout=TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    scaledown_window=60,
    max_containers=1,
    block_network=True,  # proves no weight download can happen
    # No ephemeral_disk: the link-farm layout copies nothing large (the 109 GB Captioner copy
    # was the only reason; the WS-D copy layout with the FP32 Captioner would need ~110 GB).
)(ReferenceWorker)


@app.function(
    image=hfref_image(),
    cpu=8.0,
    memory=8 * 1024,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(GOLDEN_MOUNT): golden},
    env=linked_offline_env(),
    timeout=10 * 60,
    block_network=True,
)
def offline_check() -> dict:
    """CPU-only dry run of what ``ReferenceWorker.load`` relies on, with the network blocked:
    full prep-receipt checks (``check_prep_receipt``, incl. inode fingerprints on this mount),
    the derived BF16 receipt, NVIDIA's offline ``snapshot`` + ``validate_manifest`` (TempSeg)
    through the link farm, the base resolution, and a golden Volume commit."""
    t0 = time.monotonic()
    from huggingface_hub import snapshot_download

    from flora_deploy.accel.captioner_load import default_derived_dir, verify_derived
    from flora_deploy.modal_app.warm import build_linked_layout
    from flora_deploy.nv import Family, load_family
    from flora_deploy.stages.reference import check_nv_models, check_prep_receipt

    layout = build_linked_layout(weights_root=WEIGHTS_MOUNT, hub_root=HUB_ROOT)
    out: dict = {"layout": layout.stats, "receipts": {}}
    for key, src in layout.sources.items():
        try:
            out["receipts"][key] = check_prep_receipt(PINS_BY_KEY[key], Path(src))
        except Exception as e:  # noqa: BLE001 - report every pin
            out["receipts"][key] = {"error": str(e)}
            try:  # is it only the inode (a mount-dependent field)?
                check_prep_receipt(PINS_BY_KEY[key], Path(src), fields=("size", "mtime_ns"))
                out["receipts"][key]["size_mtime_ok"] = True
            except Exception as e2:  # noqa: BLE001
                out["receipts"][key]["size_mtime_ok"] = str(e2)
    t = time.monotonic()
    out["receipts"]["tempseg_base_full_hash"] = check_prep_receipt(
        PINS_BY_KEY["tempseg_base"], Path(layout.sources["tempseg_base"]), hash_files="all")
    out["receipts"]["tempseg_base_full_hash"]["wall_s"] = round(time.monotonic() - t, 1)
    out["bf16_receipt"] = verify_derived(default_derived_dir(WEIGHTS_MOUNT))
    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"],
                                                       WEIGHTS_MOUNT)))
    check_nv_models(nv.config.MODELS)
    out["tempseg"] = str(nv.models.snapshot("tempseg"))
    out["captioner"] = str(nv.models.snapshot("captioner"))
    spec = nv.config.MODELS["tempseg_base"]
    out["tempseg_base"] = snapshot_download(  # the call at models.py:130, verbatim arguments
        spec["repo"], revision=spec["revision"], token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.safetensors", "*.json", "*.jinja"],
    )
    t = time.monotonic()
    nv.models.validate_manifest(Path(out["tempseg"]))
    out["tempseg_manifest_s"] = round(time.monotonic() - t, 2)
    golden.commit()
    out["golden_commit"] = "ok"
    out["golden_manifest"] = GOLDEN_MANIFEST.is_file()
    out["seconds"] = round(time.monotonic() - t0, 2)
    return out


@app.function(
    image=cpu_image(),
    cpu=1.0,
    memory=2 * 1024,
    volumes={str(GOLDEN_MOUNT): golden},
    timeout=5 * 60,
)
def identity_from_volume(run_id: str) -> dict:
    """Recompute ``identity_summary`` from the persisted ``<run_id>-{nvidia,fast}`` results (the
    fallback if ``identity_proof`` itself did not return)."""
    import json

    from flora_deploy.stages.reference import identity_summary

    runs = {}
    for arm in ("nvidia", "fast"):
        d = RUNS_DIR / f"{run_id}-{arm}"
        runs[arm] = {f.stem: json.loads(Path(f).read_text())
                     for f in sorted(Path(d).glob("*.json"))} if Path(d).is_dir() else {}
    return identity_summary(runs["nvidia"], runs["fast"])


@app.function(
    image=cpu_image(),
    cpu=2.0,
    memory=8 * 1024,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(GOLDEN_MOUNT): golden},
    timeout=30 * 60,
    env={"HF_HOME": "/tmp/hf", "HF_XET_HIGH_PERFORMANCE": "1"},
)
def stage_samples(force: bool = False) -> dict:
    """Copy the WS-D samples to ``flora-golden:/samples/wsd`` (see ``docs/reference-smoke.md``)."""
    from flora_deploy.stages.reference import stage_all

    return stage_all(
        SAMPLES_DIR,
        code_dir=Path(snapshot_path(PINS_BY_KEY["tempseg_code"], WEIGHTS_MOUNT)),
        commit=golden.commit,
        force=force,
    )
