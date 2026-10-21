"""``flora-gatec``: Gate C on Modal (PLAN.md §9). vLLM 0.24.0 model inputs vs NVIDIA's HF call.

One L40S container per processor kind (``tempseg`` for the segmenter cases, ``captioner`` for
the caption cases, because their processor configs differ). Each container:

1. loads NVIDIA's SEGCAP code from the code snapshot and builds the real processor with NVIDIA's
   own ``models.processor(kind, path)`` from the weights snapshot's processor files;
2. drives NVIDIA's pipeline functions (``validate.gatec_driver``) on synthetic frames and on a
   real video's frames, keeping NVIDIA's HF processor output (the reference) and the
   ``CapturedRequest`` for every call;
3. starts an offline ``vllm.LLM`` on a temp model dir: Qwen3.5-9B base weights (symlinks) plus
   Flora's processor/tokenizer/chat-template files (weights do not affect prompt token ids);
4. renders every request through vLLM's offline path (``LLM._preprocess_cmpl``, what
   ``LLM.generate`` calls) to read ``prompt_token_ids``, ``video_grid_thw`` and the pixel
   values, runs ``generate`` with ``max_tokens=1`` (greedy) to confirm the engine saw the same
   ids, and compares with HF (``validate.gatec``).

Variants per request: ``bare`` (NVIDIA's prompt as captured), ``wrapped`` (the capture-layer fix,
``nv.capture.vllm_prompt``), and for the first request of each case ``wrapped-no-duration``
(metadata without the ``duration`` key) and ``wrapped-naive-pixels`` (per-request
``min_pixels``/``max_pixels`` copied from NVIDIA's ``AutoProcessor`` call, to show whether that
mirror would be safe).

Run with ``uv run modal run scripts/gatec.py`` (ephemeral app; never deployed).
"""

from __future__ import annotations

import modal

from flora_deploy.layout import GOLDEN_MOUNT, GOLDEN_VOLUME, WEIGHTS_MOUNT, WEIGHTS_VOLUME
from flora_deploy.modal_app.images import vllm_image

app = modal.App("flora-gatec")
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)

GPU = "L40S"
TIMEOUT_S = 40 * 60
SAMPLE_DIR = "samples/wse"

# NVIDIA models.processor(): Qwen3VLVideoProcessor(fps=4|8, max_frames=128|40). The image-side
# AutoProcessor kwargs (min_pixels=1024, max_pixels=589824) do not apply to videos there.
NV_VIDEO_KWARGS = {"tempseg": {"fps": 4, "max_frames": 128},
                   "captioner": {"fps": 8, "max_frames": 40}}
NAIVE_PIXEL_KWARGS = {"min_pixels": 1024, "max_pixels": 589824}
VIDEO_PROCESSOR_ATTRS = ("size", "fps", "max_frames", "min_frames", "patch_size",
                         "temporal_patch_size", "merge_size", "resample", "image_mean",
                         "image_std", "rescale_factor", "do_resize", "do_rescale",
                         "do_normalize", "do_convert_rgb")
BASE_FILES = ("config.json", "model.safetensors.index.json")
FLORA_FILES = ("processor_config.json", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja")


def engine_kwargs(kind: str, model_dir: str) -> dict:
    return dict(
        model=model_dir,
        tokenizer=model_dir,
        dtype="bfloat16",
        max_model_len=32768,
        limit_mm_per_prompt={"image": 0, "video": 3},
        mm_processor_kwargs=dict(NV_VIDEO_KWARGS[kind]),
        mm_processor_cache_gb=0,  # every request is processed from scratch
        generation_config="vllm",
        gdn_prefill_backend="triton",
        enforce_eager=True,  # no graphs / compile: token ids do not depend on them
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        seed=0,
    )


def _build_model_dir(kind: str):
    from pathlib import Path

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import snapshot_path

    base = Path(snapshot_path(PINS_BY_KEY["tempseg_base"]))
    flora = Path(snapshot_path(PINS_BY_KEY[kind]))
    out = Path(f"/tmp/gatec-model-{kind}")
    out.mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:
        (out / name).symlink_to(base / name)
    for shard in sorted(base.glob("*.safetensors")):
        (out / shard.name).symlink_to(shard)
    for name in FLORA_FILES:
        (out / name).write_bytes((flora / name).read_bytes())
    return out, base, flora


def _grids(mm_items) -> list[list[int]]:
    import numpy as np

    out = []
    for item in mm_items:
        data = item["video_grid_thw"].data
        out += np.asarray(data.cpu() if hasattr(data, "cpu") else data).reshape(-1, 3).tolist()
    return out


def _pixels_equal(hf_pixels, mm_items) -> bool:
    import torch

    parts = [item["pixel_values_videos"].data for item in mm_items]
    v = torch.cat([p.reshape(-1, p.shape[-1]) for p in parts])
    h = hf_pixels.to(v.dtype)
    return bool(h.shape == v.shape and torch.equal(h.cpu(), v.cpu()))


@app.function(
    image=vllm_image(),
    gpu=GPU,
    cpu=4.0,
    memory=48 * 1024,
    timeout=TIMEOUT_S,
    volumes={str(WEIGHTS_MOUNT): weights, str(GOLDEN_MOUNT): golden},
    env={"PYTHONDONTWRITEBYTECODE": "1", "VLLM_CACHE_ROOT": "/tmp/vllm-cache",
         "TOKENIZERS_PARALLELISM": "false"},
)
def run_gate_c(kind: str, sample_name: str, max_requests: int = 2) -> dict:
    import copy
    import subprocess
    import time
    from dataclasses import asdict
    from pathlib import Path

    import vllm
    from vllm import LLM, SamplingParams

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import snapshot_path
    from flora_deploy.nv import Family
    from flora_deploy.nv.capture import vllm_prompt
    from flora_deploy.nv.linker import load_family
    from flora_deploy.validate import gatec as G
    from flora_deploy.validate import gatec_driver as D

    t0 = time.time()
    timings: dict[str, float] = {}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()

    # 1. NVIDIA processor + 2. requests
    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"])))
    proc_dir = Path(snapshot_path(PINS_BY_KEY[kind]))
    proc = nv.models.processor(kind, proc_dir)
    frames, rate = D.decode_all(Path(GOLDEN_MOUNT) / SAMPLE_DIR / sample_name)
    sources = [D.synthetic_source(), D.decoded_source(Path(sample_name).stem, frames, int(rate))]
    built = []
    for source in sources:
        for case_id in D.CASES:
            if D.processor_kind(case_id) == kind:
                built += D.build_requests(nv, {kind: proc}, case_id, source, max_requests)
    timings["build_requests_s"] = round(time.time() - t0, 1)

    # 3. engine
    model_dir, base, flora = _build_model_dir(kind)
    kwargs = engine_kwargs(kind, str(model_dir))
    t1 = time.time()
    llm = LLM(**kwargs)
    timings["engine_start_s"] = round(time.time() - t1, 1)

    tok = proc.tokenizer
    sp = G.SpecialIds(*(tok.convert_tokens_to_ids(t) for t in
                        ("<|vision_start|>", "<|vision_end|>", "<|video_pad|>")))
    hf_cfg = llm.llm_engine.model_config.hf_config
    vllm_ids = G.SpecialIds(hf_cfg.vision_start_token_id, hf_cfg.vision_end_token_id,
                            hf_cfg.video_token_id)
    mmp = llm.renderer.get_mm_processor()
    vproc = mmp.info.get_hf_processor().video_processor
    attr_diff = {a: [repr(getattr(proc.video_processor, a, None)), repr(getattr(vproc, a, None))]
                 for a in VIDEO_PROCESSOR_ATTRS
                 if repr(getattr(proc.video_processor, a, None)) != repr(getattr(vproc, a, None))}
    merge = int(proc.video_processor.merge_size)
    greedy = SamplingParams(max_tokens=1, temperature=0.0)

    # 4. compare
    rows = []
    t2 = time.time()
    for b in built:
        hf_ids = b.hf["input_ids"][0].tolist()
        hf_grids = b.hf["video_grid_thw"].tolist()
        fixed = vllm_prompt(b.request)
        bare = {"prompt": b.request.prompt, "multi_modal_data": copy.deepcopy(
            fixed["multi_modal_data"])}
        variants = [("bare", bare), ("wrapped", fixed)]
        if b.index == 0:
            no_dur = copy.deepcopy(fixed)
            no_dur["multi_modal_data"]["video"] = [
                (f, {k: v for k, v in md.items() if k != "duration"})
                for f, md in fixed["multi_modal_data"]["video"]]
            naive = {**fixed, "mm_processor_kwargs": dict(NAIVE_PIXEL_KWARGS)}
            variants += [("wrapped-no-duration", no_dur), ("wrapped-naive-pixels", naive)]
        for name, prompt in variants:
            extras: dict = {}
            try:
                (ei,) = llm._preprocess_cmpl([prompt])
                ids = list(ei["prompt_token_ids"])
                items = ei["mm_kwargs"]["video"]
                grids = _grids(items)
                extras["pixels_equal_bf16"] = _pixels_equal(b.hf["pixel_values_videos"], items)
                extras["placeholders"] = [[p.offset, p.length]
                                          for p in ei["mm_placeholders"]["video"]]
                if name == "wrapped" or b.index == 0:
                    (out,) = llm.generate([prompt], greedy, use_tqdm=False)
                    extras["generate_ids_equal_render"] = list(out.prompt_token_ids) == ids
                err = None
            except Exception as exc:  # recorded, not raised: the bare shape is expected to fail
                ids, grids, err = None, None, G.sanitize_error(exc)
            c = G.compare(case_id=b.case_id, index=b.index, source=b.source, variant=name,
                          hf_ids=hf_ids, hf_grids=hf_grids, vllm_ids=ids, vllm_grids=grids,
                          sp=sp, merge_size=merge, error=err, extras=extras)
            rows.append(c.to_dict())
    timings["compare_s"] = round(time.time() - t2, 1)
    timings["total_s"] = round(time.time() - t0, 1)

    import json
    base_cfg = json.loads((base / "config.json").read_text())
    return dict(
        kind=kind,
        gpu=gpu,
        versions=dict(vllm=vllm.__version__, transformers=__import__("transformers").__version__,
                      torch=__import__("torch").__version__),
        engine_kwargs={k: v for k, v in kwargs.items() if k not in ("model", "tokenizer")},
        special_ids=dict(processor=asdict(sp), vllm_hf_config=asdict(vllm_ids)),
        base_architectures=base_cfg.get("architectures"),
        video_processor_attr_diff=attr_diff,
        video_processor_attrs={a: repr(getattr(vproc, a, None)) for a in VIDEO_PROCESSOR_ATTRS},
        sources=[dict(name=s.name, size=list(s.size), fps=s.fps) for s in sources],
        n_requests=len(built),
        rows=rows,
        timings=timings,
    )
