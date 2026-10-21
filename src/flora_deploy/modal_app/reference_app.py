"""``flora-reference``: NVIDIA's own TempSeg -> Captioner pipeline on one H200 (``exact`` preset).

- ``ReferenceWorker`` (``img_hfref``, ``H200``) links NVIDIA's SEGCAP code from the weights
  Volume, loads both models once through NVIDIA's ``Session.load`` (wrapped in ``CachedSession``)
  fully offline (``stages/reference.py`` explains how), and runs ``annotate`` per episode. Each
  result JSON is copied whole to ``flora-golden:/reference_runs/<run_id>/<episode>.json``.
- ``stage_samples`` (``img_cpu``) copies the public WS-D smoke samples into
  ``flora-golden:/samples/wsd/`` and checks them with NVIDIA's ``media.probe`` /
  ``check_synchronized``.

Run with ``modal run scripts/reference_smoke.py`` (ephemeral app; never deployed here).
Every container has a hard lifetime cap (``HARD_LIFETIME_S``) on top of Modal's timeouts.
"""

from __future__ import annotations

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
from flora_deploy.stages.reference import OFFLINE_ROOT, offline_env

app = modal.App("flora-reference")
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)

GPU = "H200"
CPU_CORES = 8.0
MEMORY_MIB = 96 * 1024  # TempSeg base is built on CPU in BF16 (~18 GiB) before .to("cuda")
HARD_LIFETIME_S = 40 * 60  # no container of this app may live longer than this
STARTUP_TIMEOUT_S = 25 * 60  # Volume hashing (~122 GiB) + both loads
TIMEOUT_S = 30 * 60  # per annotate / annotate_batch call
SAMPLES_DIR = GOLDEN_MOUNT / "samples" / "wsd"
RUNS_DIR = GOLDEN_MOUNT / "reference_runs"


def _watchdog(limit_s: float = HARD_LIFETIME_S) -> None:
    """Kill this container after ``limit_s`` no matter what it is doing (spend guard)."""

    def kill():
        print(f"flora-reference watchdog: {limit_s:.0f}s lifetime reached, exiting", flush=True)
        os._exit(17)

    t = threading.Timer(limit_s, kill)
    t.daemon = True
    t.start()


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


@app.cls(
    image=hfref_image(),
    gpu=GPU,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(GOLDEN_MOUNT): golden},
    env=offline_env(OFFLINE_ROOT),
    timeout=TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    scaledown_window=120,
    max_containers=1,
    block_network=True,  # proves no weight download can happen
)
class ReferenceWorker:
    @modal.enter()
    def load(self) -> None:
        _watchdog()
        self.started = time.time()
        t0 = time.monotonic()
        from concurrent.futures import ThreadPoolExecutor

        from flora_deploy.stages.reference import build_offline_layout

        layout = build_offline_layout(OFFLINE_ROOT, WEIGHTS_MOUNT, kinds=("tempseg",))
        # The ~102 GiB Captioner copy runs while TempSeg loads (different files, both read-only
        # on the Volume side).
        copier = ThreadPoolExecutor(max_workers=1)
        captioner_layout = copier.submit(
            build_offline_layout, OFFLINE_ROOT, WEIGHTS_MOUNT, kinds=("captioner",)
        )

        import huggingface_hub.constants as hfc

        assert hfc.HF_HUB_OFFLINE, "HF_HUB_OFFLINE must be on before huggingface_hub loads"
        assert Path(hfc.HF_HUB_CACHE) == layout.hub_cache, (hfc.HF_HUB_CACHE, layout.hub_cache)

        import torch

        from flora_deploy.nv import CachedSession, Family, load_family
        from flora_deploy.stages.reference import capture_runtime_events, check_nv_models

        code_dir = Path(snapshot_path(PINS_BY_KEY["tempseg_code"], WEIGHTS_MOUNT))
        self.nv = nv = load_family(Family.SEGCAP, code_dir)
        check_nv_models(nv.config.MODELS)
        assert Path(nv.config.MODEL_CACHE) == layout.model_cache, nv.config.MODEL_CACHE
        t_link = time.monotonic()

        self.session = CachedSession.from_family(nv)
        loads, resolved, copies = {}, {}, dict(layout.copies)
        for kind in ("tempseg", "captioner"):  # TempSeg first, then the Captioner
            t = time.monotonic()
            if kind == "captioner":
                cap = captioner_layout.result()
                copies.update(cap.copies)
                layout.paths.update(cap.paths)
                copier.shutdown()
            # Where NVIDIA's own snapshot() resolves this kind (offline; no hashing yet).
            resolved[kind] = str(nv.models.snapshot(kind))
            assert Path(resolved[kind]).resolve() == layout.paths[kind].resolve(), resolved
            torch.cuda.reset_peak_memory_stats()
            t_wait = time.monotonic() - t
            t = time.monotonic()
            with capture_runtime_events() as events:
                self.session.load(kind)
            loads[kind] = {
                "wait_for_copy_s": round(t_wait, 1),
                "seconds": round(time.monotonic() - t, 1),
                "gpu_after": _gpu_memory(),
                "events": [
                    {k: v for k, v in e.items() if k not in ("utc",)}
                    for e in events
                    if e.get("event", "").startswith(("model_", "manifest_"))
                ],
            }
            print(f"loaded {kind} in {loads[kind]['seconds']}s: {loads[kind]['gpu_after']}",
                  flush=True)
        # Every NVIDIA generate() runs under one GPU lock (docs/exact-concurrency.md); with one
        # episode at a time the lock is never contended.
        from flora_deploy.observe import Patcher, TimedLock
        from flora_deploy.stages.reference import GPU_LOCK_NAME, install_gpu_lock

        self.gpu_lock = TimedLock(GPU_LOCK_NAME)
        self.lock_patcher = Patcher()
        install_gpu_lock(nv, self.gpu_lock, self.lock_patcher)
        from flora_deploy.stages.reference import ProcessorClones

        self.clones = ProcessorClones()  # per-episode processor copies, reused across batches
        self.torch_threads_after_load = torch.get_num_threads()
        self.run_id = time.strftime("wsd-%Y%m%dT%H%M%SZ", time.gmtime())
        self.load_report = {
            "run_id": self.run_id,
            "layout_s": round(t_link - t0, 2),  # TempSeg copy + linking NVIDIA's code
            "copies": copies,
            "enter_s": round(time.monotonic() - t0, 1),
            "resolved_snapshots": resolved,
            "hub_offline": bool(hfc.HF_HUB_OFFLINE),
            "loads": loads,
            "loaded_kinds": list(self.session.loaded_kinds),
            "torch": torch.__version__,
            "torch_threads": self.torch_threads_after_load,
            "cpu_count": os.cpu_count(),
            "sched_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
            else None,
            "determinism": _determinism_flags(),
            "cgroup_cpu_max": _read_text("/sys/fs/cgroup/cpu.max"),
            "cgroup_memory_max": _read_text("/sys/fs/cgroup/memory.max"),
            "nv_audit_kinds": sorted(self.session.audit),
        }

    @modal.method()
    def report(self) -> dict:
        return {**self.load_report, "gpu_now": _gpu_memory(),
                "container_age_s": round(time.time() - self.started, 1)}

    @modal.method()
    def annotate(
        self,
        videos: dict[str, str],
        profile: str,
        long_windows: bool = False,
        instruction: str = "",
        *,
        episode: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        import torch

        from flora_deploy.stages.reference import annotate_reference, persist_result

        golden.reload()  # pick up freshly staged samples (no files are open here)
        torch.cuda.reset_peak_memory_stats()
        result = annotate_reference(
            self.nv,
            self.session,
            videos,
            profile,
            long_windows,
            instruction,
            progress=lambda frac, desc="": print(f"{frac:.0%} {desc}", flush=True),
        )
        ref = result["flora_reference"]
        ref["gpu"] = _gpu_memory()
        ref["models_resident"] = list(self.session.loaded_kinds)
        ref["run_id"] = run_id or self.run_id
        episode = episode or Path(videos["main"]).stem
        path = persist_result(
            result, golden_root=GOLDEN_MOUNT, run_id=ref["run_id"], episode=episode,
            commit=golden.commit,
        )
        ref["stored_at"] = f"{GOLDEN_VOLUME}:{path.relative_to(GOLDEN_MOUNT)}"
        return result

    @modal.method()
    def annotate_batch(
        self,
        episodes: list[dict],
        concurrency: int = 1,
        *,
        measure: bool = False,
        torch_threads: int | None = None,
        run_id: str | None = None,
        tag: str = "batch",
        persist: bool = True,
    ) -> dict:
        """Run several episodes with ``concurrency`` threads sharing the resident models
        (``stages/reference.annotate_batch``); persist each result after the batch.

        ``torch_threads`` sets torch's intra-op threads for this batch only (NVIDIA's load sets 4;
        restored afterwards). ``measure`` turns on the observe harness (per-call timers, RSS,
        ``/dev/shm``). Returns the full results plus per-episode digests of NVIDIA's output."""
        import torch

        from flora_deploy.observe import Patcher, Sampler
        from flora_deploy.stages.reference import (
            annotate_batch,
            install_measurement,
            persist_result,
            result_digest,
            safe_name,
        )

        golden.reload()
        torch.cuda.reset_peak_memory_stats()
        previous_threads = torch.get_num_threads()
        if torch_threads:
            torch.set_num_threads(int(torch_threads))
        patcher = Patcher()
        try:
            patched = []
            if measure:
                procs = [self.session.load(kind)[1] for kind in self.session.loaded_kinds]
                patched = install_measurement(self.nv, patcher, procs)
            out = annotate_batch(
                self.nv, self.session, episodes, concurrency=concurrency,
                gpu_lock=self.gpu_lock, measure=measure, clones=self.clones,
                sampler=Sampler(interval_s=1.0) if measure else None,
                progress=lambda ep, frac, desc="": print(f"[{ep}] {frac:.0%} {desc}", flush=True),
            )
        finally:
            patcher.restore()
            torch.set_num_threads(previous_threads)
        batch = out["batch"]
        batch.update(torch_threads=torch_threads or previous_threads, measured=patched,
                     gpu=_gpu_memory(), container_age_s=round(time.time() - self.started, 1))
        run = safe_name(f"{run_id or self.run_id}-{tag}")
        digests = {}
        for name, result in out["results"].items():
            digests[name] = result_digest(result)
            ref = result["flora_reference"]
            ref["run_id"], ref["digest"] = run, digests[name]
            if persist:
                path = persist_result(result, golden_root=GOLDEN_MOUNT, run_id=run,
                                      episode=name, commit=lambda: None)
                ref["stored_at"] = f"{GOLDEN_VOLUME}:{path.relative_to(GOLDEN_MOUNT)}"
        if persist:
            golden.commit()
        out["digests"] = digests
        return out

    @modal.exit()
    def close(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session.close_all()


@app.function(
    image=hfref_image(),
    cpu=2.0,
    memory=4 * 1024,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(GOLDEN_MOUNT): golden},
    env=offline_env(OFFLINE_ROOT),
    timeout=15 * 60,
    block_network=True,
)
def offline_check() -> dict:
    """CPU-only dry run of the offline resolution ``ReferenceWorker.load`` relies on.

    Resolves all three snapshots the way NVIDIA's ``Session.load`` does (``models.py:123`` and
    ``:130``) with the network blocked, runs NVIDIA's ``validate_manifest`` on the TempSeg
    snapshot, and checks the golden Volume can be committed with the network blocked.
    """
    t0 = time.monotonic()
    from flora_deploy.stages.reference import build_offline_layout, check_nv_models

    layout = build_offline_layout(OFFLINE_ROOT, WEIGHTS_MOUNT, kinds=("tempseg",))
    from huggingface_hub import snapshot_download

    from flora_deploy.nv import Family, load_family

    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"], WEIGHTS_MOUNT)))
    check_nv_models(nv.config.MODELS)
    out = {"copies": layout.copies, "tempseg": str(nv.models.snapshot("tempseg"))}
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
    out["expected"] = {k: str(v) for k, v in layout.paths.items()}
    out["seconds"] = round(time.monotonic() - t0, 2)
    return out


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
