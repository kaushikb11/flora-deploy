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
  - W3-A episode concurrency (default 4) under one GPU lock around NVIDIA's ``generate``.

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
STARTUP_TIMEOUT_S = 8 * 60  # @modal.enter: layout + both loads (~2-3 min measured)
TIMEOUT_S = 20 * 60  # per call; the hard runtime bound (with STARTUP_TIMEOUT_S)
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


@app.cls(
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
)
class ReferenceWorker:
    disable: str = modal.parameter(default="")  # ExactProfile toggles to switch off
    concurrency: int = modal.parameter(default=4)

    @modal.enter()
    def load(self) -> None:
        t0 = time.monotonic()
        self.started = time.time()
        self.state = "loading"
        marks: dict[str, float] = {}

        def mark(name: str) -> None:
            marks[name] = round(time.monotonic() - t0, 2)

        from flora_deploy.stages.reference import (
            ExactProfile,
            build_offline_layout,
            check_nv_models,
            check_prep_receipt,
        )

        self.profile = profile = ExactProfile.from_spec(self.disable, self.concurrency)
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

        # 2. t=0 prefetch of the weights this load reads (base, then the Captioner)
        from flora_deploy.accel.captioner_load import default_derived_dir
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

            self.bf16 = captioner_load.install(nv, self.derived_dir)
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
            "container_import_to_enter_s": round(self.started - IMPORTED_AT, 2),
            "enter_s": round(time.monotonic() - t0, 1),
            "marks_s": marks,
            "layout": layout_stats,
            "receipts": receipts,
            "prefetch": prefetch,
            "captioner_hash": hash_info or None,
            "bf16_receipt": self.bf16.verifications[0] if self.bf16 else None,
            "loads": loads,
            "loaded_kinds": list(self.session.loaded_kinds),
            "torch": torch.__version__,
            "torch_threads": self.torch_threads_after_load,
            "cpu_count": os.cpu_count(),
            "determinism": _determinism_flags(),
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
                "pipeline.Predictor.predict": nv.pipeline.Predictor.predict}

    def _install_fast(self) -> None:
        from flora_deploy.accel import apply_patches
        from flora_deploy.observe import Patcher, TimedLock
        from flora_deploy.stages.reference import (
            GPU_LOCK_NAME,
            ProcessorClones,
            install_gpu_lock,
            install_nvidia_entry,
        )

        self.gpu_lock = TimedLock(GPU_LOCK_NAME)
        self.patcher = Patcher()
        install_gpu_lock(self.nv, self.gpu_lock, self.patcher)
        install_nvidia_entry(self.nv, self.patcher,
                             request_lock=self.profile.concurrency <= 1)
        self.accel = apply_patches(self.nv, self.profile.accel) if self.profile.accel else None
        self.clones = ProcessorClones()

    def _restore_nvidia(self) -> dict:
        """Undo every swap and release the resident models: NVIDIA's unmodified state."""
        import gc

        import torch

        if self.accel is not None:
            self.accel.restore()
        self.patcher.restore()
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
        return out

    def _persist(self, out: dict, run: str) -> None:
        from flora_deploy.stages.reference import persist_result, result_digest

        out["digests"] = {}
        for name, result in out["results"].items():
            ref = result["flora_reference"]
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

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str):
        """``flora_deploy.workers.ShardWorker`` for the ``segcap`` kind.

        TODO(P1a-5): attempt marker -> skip episodes already in earlier parts -> per episode
        ``annotate_batch`` + ``stages.reference.stage_records`` (``segment`` and ``caption``
        records) into a local JSONL part -> whole-file copy to the job store -> ``ShardReceipt``
        last. Not done in S1: it needs the job store client in the container and the episode
        input resolution from ``EpisodeRef.index_ref`` (source URIs and ranges -> local camera
        files), which P1a-5 owns."""
        raise NotImplementedError("run_shard is built by P1a-5 (see the docstring)")

    @modal.exit()
    def close(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session.close_all()


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
