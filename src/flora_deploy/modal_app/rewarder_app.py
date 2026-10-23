"""``flora-rewarder``: NVIDIA's Rewarder recipe (offline ``vllm.LLM``) on one ``H100!``.

WS-H smoke harness (``scripts/rewarder_smoke.py``), run ephemerally with ``modal run``:

- ``stage_samples`` (CPU): copies a few short public LeRobot v2.1 episodes (one MP4 per camera,
  in NVIDIA's collage order) into the ``flora-golden`` Volume under ``samples/wsh/`` and writes
  a jobs JSONL in NVIDIA's format.
- ``Rewarder`` (GPU): links NVIDIA's ``batch_utils`` from the weights snapshot, builds the engine
  once per container with NVIDIA's own ``engine.load_llm`` (terminal-script arguments, which the
  dense script matches, unchanged), and serves two methods on that one engine:
  - ``reference(mode)``: NVIDIA's ``terminal_reward.main`` / ``dense_reward.main`` with argv on
    the jobs JSONL (the oracle). Its ``load_llm`` call is checked against the shared engine's
    arguments and then answered with the shared engine, so no second 52 GiB load is needed.
  - ``batched(mode, ...)``: ``stages.reward.run`` in size-bounded rounds, writing part files per
    round; ``round_requests`` / ``round_mib`` override :class:`stages.reward.RoundLimits`.
  ``reference`` starts with ``reset_prefix_cache()`` (``stages.reward.run`` resets before every
  round), so no run sees another's prefix cache (NVIDIA's ``enable_prefix_caching=True`` is the
  experimental hybrid "align" mode, where hit/miss can change numerics).

Startup recipe (W3-B, ``docs/cold-start.md``):

1. ``warm.start_prefetch(<snapshot>, threads=16)`` is the first thing ``@modal.enter`` does, so
   the 52 GiB of weights stream into the page cache while Python imports and vLLM's config run.
2. ``flora-cache`` is mounted at ``/cache``; ``VLLM_CACHE_ROOT`` (torch.compile / Inductor),
   ``TRITON_CACHE_DIR``, ``CUDA_CACHE_PATH`` and ``XDG_CACHE_HOME`` (FlashInfer) all point into
   ``/cache/rewarder/<key>``, where the key hashes the image fingerprint, the model pin and the
   GPU. The Triton directory matters because vLLM sets it only on its compile path, not on a
   compile-cache hit, so without it ~95 s of JIT/autotune ran on container-local disk every time.
   The cache is committed after the engine is built.
3. ``memory`` is 80 GiB (W3-B: >= 72 GiB so the prefetched 52 GiB stays resident; 80 GiB is the
   configuration the cold-start lab measured).

Importing this module creates lazy Modal objects only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import (
    CACHE_MOUNT,
    CACHE_VOLUME,
    GOLDEN_MOUNT,
    GOLDEN_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    snapshot_path,
)
from flora_deploy.modal_app.images import cpu_image, vllm_image

app = modal.App("flora-rewarder")
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2).read_only()
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)

SNAPSHOT = snapshot_path(PINS_BY_KEY["rewarder"])
SAMPLES_DIR = GOLDEN_MOUNT / "samples" / "wsh"
JOBS_NAME = "jobs.jsonl"
SOURCES_NAME = "sources.json"

GPU = "H100!"
TIMEOUT_S = 45 * 60
MEMORY_MIB = 80 * 1024
CPU = 8.0
H100_USD_PER_H = 3.95
USD_PER_H = H100_USD_PER_H + CPU * 0.0472 + MEMORY_MIB / 1024 * 0.008
PREFETCH_THREADS = 16
IMAGE_FP_ENV = "FLORA_IMG_VLLM_FINGERPRINT"


def _image_fingerprint() -> str:
    """img_vllm's fingerprint: computed from the lock where it exists (locally), else passed
    into the container through the class env."""
    try:
        from flora_deploy.modal_app.images import image_fingerprint

        return image_fingerprint("img_vllm")
    except OSError:
        return os.environ.get(IMAGE_FP_ENV, "unknown")


IMAGE_FP = _image_fingerprint()


def cache_key(image_fp: str = IMAGE_FP, gpu: str = GPU) -> str:
    """Compile/Triton cache subdirectory key: image + model pin + GPU (vLLM keys by its own
    config hash below that, so engine arguments are covered too)."""
    pin = PINS_BY_KEY["rewarder"]
    blob = json.dumps({"image": image_fp, "model": f"{pin.repo}@{pin.revision}", "gpu": gpu},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_env(root: Path | str = CACHE_MOUNT, key: str | None = None) -> dict[str, str]:
    base = Path(root) / "rewarder" / (key or cache_key())
    return {
        "VLLM_CACHE_ROOT": str(base / "vllm"),
        "TRITON_CACHE_DIR": str(base / "triton"),
        "CUDA_CACHE_PATH": str(base / "cuda"),
        "XDG_CACHE_HOME": str(base / "xdg"),
    }

# Short public robot episodes (ungated HF datasets, LeRobot v2.1 layout: one MP4 per camera per
# episode). Cameras are listed in NVIDIA's collage order: 3 views left/center/right, 2 views
# left/right (exterior, then wrist), 1 view as is.
SAMPLE_EPISODES: tuple[dict[str, Any], ...] = (
    {
        "id": "aloha_coffee_ep000",
        "repo": "lerobot/aloha_static_coffee",
        "revision": "7da22eb685a613a28bcc58aa0bf7d1094578234a",  # tag v2.1
        "episode": 0,
        "cameras": ["observation.images.cam_left_wrist", "observation.images.cam_high",
                    "observation.images.cam_right_wrist"],
        "license": "mit",
    },
    {
        "id": "ur5_ep002",
        "repo": "lerobot/berkeley_autolab_ur5",
        "revision": "6306aa91c00b9b0de28831b7fceaa42b8096bb4f",  # tag v2.1
        "episode": 2,
        "cameras": ["observation.images.image", "observation.images.hand_image"],
        "license": "cc-by-4.0",
    },
    {
        "id": "toto_ep002",
        "repo": "lerobot/toto",
        "revision": "35ba68ad482f77e0c18eb26f1bfe480b49c84698",  # tag v2.1
        "episode": 2,
        "cameras": ["observation.images.image"],
        "license": "mit",
    },
)
MAX_EPISODE_S = 120.0


# ---------------------------------------------------------------------------
# Sample staging (CPU)
# ---------------------------------------------------------------------------


def probe_stream(path: Path) -> dict[str, Any]:
    """Frame count (packets), rate, size, codec and a constant-frame-rate check."""
    from flora_deploy.media.avsafe import open_input

    with open_input(path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        pts = sorted(p.pts for p in container.demux(stream) if p.size and p.pts is not None)
        tb = stream.time_base
        deltas = {b - a for a, b in zip(pts, pts[1:], strict=False)}
        info = {
            "frames": len(pts),
            "fps": float(rate),
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "codec": stream.codec_context.name,
            "cfr": len(deltas) == 1,
            "frame_duration_s": float(min(deltas) * tb) if deltas else None,
        }
    info["duration_s"] = info["frames"] / info["fps"]
    return info


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_episode(spec: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download one episode's camera MP4s into ``root/<id>/``; returns (job, source record)."""
    import shutil

    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(spec["repo"], name, repo_type="dataset",
                                    revision=spec["revision"]))

    info = json.loads(fetch("meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError(f"{spec['repo']}: not a LeRobot v2.1 dataset")
    task = None
    for line in fetch("meta/episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["episode_index"] == spec["episode"]:
            task = row["tasks"][0]
    if not task:
        raise ValueError(f"{spec['repo']}: episode {spec['episode']} has no task")
    chunk = spec["episode"] // info["chunks_size"]
    out = root / spec["id"]
    out.mkdir(parents=True, exist_ok=True)
    videos, cameras = [], []
    for key in spec["cameras"]:
        name = info["video_path"].format(episode_chunk=chunk, video_key=key,
                                         episode_index=spec["episode"])
        target = out / f"{key.rsplit('.', 1)[-1]}.mp4"
        shutil.copyfile(fetch(name), target)
        probe = probe_stream(target)
        cameras.append({"key": key, "file": name, "sha256": _sha256(target), **probe})
        videos.append(str(target.relative_to(root)))
    frames = {c["frames"] for c in cameras}
    fps = {c["fps"] for c in cameras}
    duration = max(c["duration_s"] for c in cameras)
    problems = []
    if not all(c["cfr"] for c in cameras):
        problems.append("not CFR")
    if len(fps) != 1 or len(frames) != 1:
        problems.append(f"cameras out of sync: fps {sorted(fps)}, frames {sorted(frames)}")
    if duration > MAX_EPISODE_S:
        problems.append(f"{duration:.1f} s > {MAX_EPISODE_S} s")
    if problems:
        raise ValueError(f"{spec['id']}: " + "; ".join(problems))
    job = {"id": spec["id"], "task": task, "videos": videos}
    source = {**spec, "task": task, "dataset_fps": info["fps"], "duration_s": duration,
              "cameras": cameras, "url": f"https://huggingface.co/datasets/{spec['repo']}"}
    return job, source


@app.function(image=cpu_image(), volumes={str(GOLDEN_MOUNT): golden}, cpu=2.0, memory=4096,
              timeout=20 * 60, env={"HF_HOME": "/tmp/hf"})
def stage_samples() -> dict[str, Any]:
    """Stage :data:`SAMPLE_EPISODES` into ``flora-golden:/samples/wsh`` (idempotent rewrite)."""
    root = Path(SAMPLES_DIR)
    root.mkdir(parents=True, exist_ok=True)
    jobs, sources = [], []
    for spec in SAMPLE_EPISODES:
        job, source = stage_episode(spec, root)
        jobs.append(job)
        sources.append(source)
    (root / JOBS_NAME).write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (root / SOURCES_NAME).write_text(json.dumps(sources, indent=2) + "\n")
    golden.commit()
    return {"jobs": jobs, "sources": sources, "dir": str(root)}


# ---------------------------------------------------------------------------
# Rewarder (GPU)
# ---------------------------------------------------------------------------


def _engine_facts(llm) -> dict[str, Any]:
    """Resolved engine settings worth recording (best effort; attribute names vary)."""
    facts: dict[str, Any] = {}
    try:
        import vllm

        facts["vllm"] = vllm.__version__
        config = llm.llm_engine.vllm_config
        facts["max_num_batched_tokens"] = config.scheduler_config.max_num_batched_tokens
        facts["max_num_seqs"] = config.scheduler_config.max_num_seqs
        facts["max_logprobs"] = config.model_config.max_logprobs
        facts["dtype"] = str(config.model_config.dtype)
        facts["enable_prefix_caching"] = config.cache_config.enable_prefix_caching
        facts["mamba_cache_mode"] = getattr(config.cache_config, "mamba_cache_mode", None)
        facts["block_size"] = config.cache_config.block_size
        # Not on a stable config path in 0.24.0; the engine log line
        # "Using Triton/FLA GDN prefill kernel (requested=triton ...)" is the evidence.
        for holder in (config.model_config, getattr(config, "kernel_config", None)):
            value = getattr(holder, "gdn_prefill_backend", None)
            if value is not None:
                facts["gdn_prefill_backend"] = str(value)
        facts["limit_mm_per_prompt"] = str(
            getattr(config.model_config.multimodal_config, "limit_per_prompt", None)
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        facts["error"] = repr(exc)
    return facts


def _vllm_defaults() -> dict[str, Any]:
    from vllm.engine.arg_utils import EngineArgs

    return {"max_logprobs": EngineArgs.max_logprobs}


def expanded_jobs(nv, jobs: str, variants: int) -> Path:
    """``jobs`` with every episode repeated under ``variants`` task phrasings (ids
    ``<id>_v<k>``; ``k=0`` is the original), as an absolute-path jobs file in ``/tmp``. Different
    tasks keep the prompts distinct well inside the first KV block, so no prefix-cache hits."""
    args = argparse.Namespace(video=None, task=None, jobs=Path(jobs))
    base = nv.engine.read_jobs(args)
    lines = []
    for k in range(variants):
        for job in base:
            task = job["task"] if k == 0 else f"{job['task']} (Variant {k}: judge the end state.)"
            lines.append({"id": f"{job['id']}_v{k}", "task": task, "videos": job["videos"]})
    out = Path("/tmp/flora-jobs") / f"jobs-x{variants}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return out


@app.cls(
    image=vllm_image(),
    gpu=GPU,
    volumes={str(WEIGHTS_MOUNT): weights, str(GOLDEN_MOUNT): golden.read_only(),
             str(CACHE_MOUNT): cache},
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    max_containers=1,
    scaledown_window=300,
    env={IMAGE_FP_ENV: IMAGE_FP},
)
class Rewarder:
    @modal.enter()
    def start(self) -> None:
        """``_start``; as a job worker, a startup failure is recorded for the orchestrator
        before it re-raises (``workers_app.record_start_failure``)."""
        try:
            self._start()
        except BaseException as exc:
            from flora_deploy.modal_app.workers_app import record_start_failure

            record_start_failure(exc)
            raise

    def _start(self) -> None:
        self.started = time.time()
        t0 = time.perf_counter()
        from flora_deploy.modal_app import warm

        prefetch = warm.start_prefetch(str(SNAPSHOT), threads=PREFETCH_THREADS)  # t=0
        env = cache_env()
        os.environ.update(env)  # before vLLM is imported; the EngineCore process inherits it
        cache_dirs = {k: Path(v) for k, v in env.items()}
        cache_state = {k: (p.is_dir() and any(p.iterdir())) for k, p in cache_dirs.items()}
        for p in cache_dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        from flora_deploy.nv import Family
        from flora_deploy.nv.linker import load_family
        from flora_deploy.stages import reward

        self.nv = load_family(Family.REWARDER, str(SNAPSHOT))
        self.processor = self.nv.engine.load_processor(str(SNAPSHOT))
        t1 = time.perf_counter()
        self.llm, self.spec = reward.build_engine(self.nv, str(SNAPSHOT))
        t2 = time.perf_counter()
        stats = prefetch.wait(timeout=0)
        self.facts = _engine_facts(self.llm)
        self.defaults = _vllm_defaults()
        cache.commit()
        t3 = time.perf_counter()
        self.engine_build_s = t2 - t1
        self.startup = {
            "cache_key": cache_key(), "cache_env": env, "cache_nonempty_at_start": cache_state,
            "link_and_processor_s": t1 - t0, "engine_build_s": t2 - t1,
            "cache_commit_s": t3 - t2, "enter_total_s": t3 - t0,
            "prefetch": stats.as_json() if stats else {"done": prefetch.done},
            "region": os.environ.get("MODAL_REGION"),
        }

    def _info(self) -> dict[str, Any]:
        uptime = time.time() - self.started
        return {
            "spec": self.spec.as_json(),
            "resolved": self.facts,
            "vllm_defaults": self.defaults,
            "engine_build_s": self.engine_build_s,
            "startup": self.startup,
            "container_uptime_s": uptime,
            "container_started": self.started,
        }

    @modal.method()
    def reference(self, mode: str, jobs: str = str(SAMPLES_DIR / JOBS_NAME),
                  variants: int = 1) -> dict[str, Any]:
        """NVIDIA's own script for ``mode`` on ``jobs`` (its argv defaults), shared engine."""
        from flora_deploy.stages import reward

        reward.check_mode(mode)
        if variants > 1:
            jobs = str(expanded_jobs(self.nv, jobs, variants))
        script = self.nv.dense_reward if mode == "dense" else self.nv.terminal_reward
        output = Path("/tmp/reference") / f"{mode}.jsonl"
        argv = reward.reference_argv(mode, jobs, output, str(SNAPSHOT))
        recording = reward.RecordingLLM(self.llm)
        if self.llm.reset_prefix_cache() is False:
            raise RuntimeError("reset_prefix_cache() returned False")
        t0 = time.perf_counter()
        with reward.shared_engine(script, recording, self.spec, self.defaults) as calls, \
                reward.tap_score_probabilities(self.nv.core) as probabilities:
            code = script.main(argv)
        wall = time.perf_counter() - t0
        lines = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        stats = recording.stats(mode, len(lines))
        stats["wall_s"] = wall
        return {"mode": mode, "argv": argv, "exit_code": code, "records": lines,
                "probabilities": probabilities if mode != "dense" else None,
                "load_llm_calls": json.loads(json.dumps(calls)), "stats": stats,
                "engine": self._info()}

    @modal.method()
    def batched(self, mode: str, episodes: list[dict[str, Any]] | None = None,
                jobs: str = str(SAMPLES_DIR / JOBS_NAME), round_requests: int | None = None,
                round_mib: int | None = None, decoder: str = "thread", workers: int = 4,
                tag: str = "default", align: str = "nvidia",
                variants: int = 1) -> dict[str, Any]:
        """``stages.reward.run`` in size-bounded rounds; parts per round under
        ``/tmp/flora-parts/<mode>-<tag>/`` (local) and ``.../published/`` (whole-file copies).
        ``episodes`` are jobs-file dicts with absolute video paths."""
        import dataclasses

        from flora_deploy.stages import reward

        if variants > 1:
            jobs = str(expanded_jobs(self.nv, jobs, variants))
        if episodes is None:
            eps = reward.read_jobs(self.nv, jobs, dataset="flora-golden/samples/wsh")
        else:
            eps = [reward.Episode.from_job(e, dataset="flora-golden/samples/wsh")
                   for e in episodes]
        limits = reward.RoundLimits(align=align)
        if round_requests is not None:
            limits = dataclasses.replace(limits, max_requests=round_requests)
        if round_mib is not None:
            limits = dataclasses.replace(limits, max_bytes=round_mib * reward.MiB)
        root = Path("/tmp/flora-parts") / f"{mode}-{tag}"
        sink = reward.LocalPartSink(root / "local", root / "published")
        pin = PINS_BY_KEY["rewarder"]
        t0 = time.perf_counter()
        result = reward.run(
            self.llm, self.nv, self.processor, eps, mode, spec=self.spec,
            models={"rewarder": pin.revision}, gpu=GPU, preset="exact",
            config_extra={"image": IMAGE_FP}, limits=limits, decoder=decoder, workers=workers,
            sink=sink, container_id=os.environ.get("MODAL_TASK_ID", "local"),
        )
        out = result.as_json()
        out["stats"]["wall_s"] = time.perf_counter() - t0
        out["part_files"] = {Path(p).name: len(Path(p).read_text().splitlines())
                             for p in result.parts}
        out["engine"] = self._info()
        return out
