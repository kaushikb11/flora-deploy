"""``flora-rewarder``: NVIDIA's Rewarder recipe (offline ``vllm.LLM``) on one ``H100!``.

WS-H smoke harness (``scripts/rewarder_smoke.py``), run ephemerally with ``modal run``:

- ``stage_samples`` (CPU): copies a few short public LeRobot v2.1 episodes (one MP4 per camera,
  in NVIDIA's collage order) into the ``flora-golden`` Volume under ``samples/wsh/`` and writes
  a jobs JSONL in NVIDIA's format.
- ``Rewarder`` (GPU): links NVIDIA's ``batch_utils`` from the weights snapshot, builds the engine
  once per container with NVIDIA's own ``engine.load_llm`` (terminal-script arguments, which the
  dense script matches), and serves two methods on that one engine:
  - ``reference(mode)``: NVIDIA's ``terminal_reward.main`` / ``dense_reward.main`` with argv on
    the jobs JSONL (the oracle). Its ``load_llm`` call is checked against the shared engine's
    arguments and then answered with the shared engine, so no second 52 GiB load is needed.
  - ``batched(mode, episodes)``: ``stages.reward.run``, one ``llm.generate`` per mode.
  Each call starts with ``reset_prefix_cache()`` so the reference and batched runs both start
  from a cold prefix cache (NVIDIA's ``enable_prefix_caching=True`` is the experimental
  hybrid "align" mode, where hit/miss can change numerics).

Importing this module creates lazy Modal objects only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import (
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

SNAPSHOT = snapshot_path(PINS_BY_KEY["rewarder"])
SAMPLES_DIR = GOLDEN_MOUNT / "samples" / "wsh"
JOBS_NAME = "jobs.jsonl"
SOURCES_NAME = "sources.json"

GPU = "H100!"
TIMEOUT_S = 45 * 60
H100_USD_PER_H = 3.95

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
    import av

    with av.open(str(path)) as container:
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
        facts["gdn_prefill_backend"] = str(
            getattr(config.model_config, "gdn_prefill_backend", None)
            or getattr(getattr(config, "kernel_config", None), "gdn_prefill_backend", None)
        )
        facts["limit_mm_per_prompt"] = str(
            getattr(config.model_config.multimodal_config, "limit_per_prompt", None)
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        facts["error"] = repr(exc)
    return facts


def _vllm_defaults() -> dict[str, Any]:
    from vllm.engine.arg_utils import EngineArgs

    return {"max_logprobs": EngineArgs.max_logprobs}


@app.cls(
    image=vllm_image(),
    gpu=GPU,
    volumes={str(WEIGHTS_MOUNT): weights, str(GOLDEN_MOUNT): golden},
    cpu=8.0,
    memory=48 * 1024,
    timeout=TIMEOUT_S,
    max_containers=1,
    scaledown_window=300,
)
class Rewarder:
    @modal.enter()
    def start(self) -> None:
        from flora_deploy.nv import Family
        from flora_deploy.nv.linker import load_family
        from flora_deploy.stages import reward

        self.started = time.time()
        self.nv = load_family(Family.REWARDER, str(SNAPSHOT))
        self.processor = self.nv.engine.load_processor(str(SNAPSHOT))
        t0 = time.perf_counter()
        self.llm, self.spec = reward.build_engine(self.nv, str(SNAPSHOT))
        self.engine_build_s = time.perf_counter() - t0
        self.facts = _engine_facts(self.llm)
        self.defaults = _vllm_defaults()

    def _info(self) -> dict[str, Any]:
        uptime = time.time() - self.started
        return {
            "spec": self.spec.as_json(),
            "resolved": self.facts,
            "vllm_defaults": self.defaults,
            "engine_build_s": self.engine_build_s,
            "container_uptime_s": uptime,
            "container_started": self.started,
        }

    @modal.method()
    def reference(self, mode: str, jobs: str = str(SAMPLES_DIR / JOBS_NAME)) -> dict[str, Any]:
        """NVIDIA's own script for ``mode`` on ``jobs`` (its argv defaults), shared engine."""
        from flora_deploy.stages import reward

        reward.check_mode(mode)
        golden.reload()
        script = self.nv.dense_reward if mode == "dense" else self.nv.terminal_reward
        output = Path("/tmp/reference") / f"{mode}.jsonl"
        argv = reward.reference_argv(mode, jobs, output, str(SNAPSHOT))
        recording = reward.RecordingLLM(self.llm)
        self.llm.reset_prefix_cache()
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
                jobs: str = str(SAMPLES_DIR / JOBS_NAME)) -> dict[str, Any]:
        """``stages.reward.run``: every request of ``mode`` in one ``llm.generate`` call."""
        from flora_deploy.stages import reward

        golden.reload()
        if episodes is None:
            eps = reward.read_jobs(self.nv, jobs)
        else:
            eps = [reward.Episode.from_job(e) for e in episodes]
        self.llm.reset_prefix_cache()
        t0 = time.perf_counter()
        result = reward.run(self.llm, self.nv, self.processor, eps, mode, provenance={
            "model": f"{PINS_BY_KEY['rewarder'].repo}@{PINS_BY_KEY['rewarder'].revision}",
            "gpu": GPU,
        })
        result.stats["wall_s"] = time.perf_counter() - t0
        return {"mode": mode, "records": result.records, "stats": result.stats,
                "engine": self._info()}
