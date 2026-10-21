"""``flora-gatec``: Gate C on Modal (PLAN.md §8). vLLM 0.24.0 model inputs vs NVIDIA's HF call.

Two stages, so request building (CPU, NVIDIA's single-threaded decode and PNG work) never holds
a GPU:

1. ``build_part`` (CPU containers, one per kind x part, in parallel): loads NVIDIA's SEGCAP code
   from the code snapshot, builds the real processor with NVIDIA's own
   ``models.processor(kind, path)`` from the weights snapshot's processor files, and drives
   NVIDIA's pipeline functions (``validate.gatec_driver``). Parts:
   - ``cases:<source>``: every ``gatec_driver.CASES`` entry of the kind on synthetic frames or
     on the decoded public sample video (fake decoder, ``max_requests`` calls per case);
   - ``golden:<pool>``: real staged golden episodes (Volume ``flora-golden``, ``golden/v1/``)
     with NVIDIA's real ``media.probe``/``decode_selected``, up to ``quota`` calls per pool.
   It keeps NVIDIA's HF processor output (ids, grids, per-video pixel digests) and the
   ``CapturedRequest`` of every call, and writes them as a pickle to the golden Volume
   (``gatec-tmp/<run>/``; deleted by stage 2).
2. ``run_gate_c`` (one L40S container per kind): starts an offline ``vllm.LLM`` on a temp model
   dir (Qwen3.5-9B base weights as symlinks plus Flora's processor/tokenizer/chat-template files;
   weights do not affect prompt token ids), renders every request through vLLM's offline path
   (``LLM._preprocess_cmpl``, what ``LLM.generate`` calls; a private API, recheck on any vLLM
   bump) to read ``prompt_token_ids``, ``video_grid_thw`` and the pixel values, runs
   ``generate`` with ``max_tokens=1`` to confirm the engine saw the same ids, and compares with
   HF (``validate.gatec``; PASS needs ids, grids and per-video pixels equal).

Variants per case request: ``bare`` (NVIDIA's prompt as captured), ``wrapped`` (the capture-layer
fix, ``nv.capture.vllm_prompt``), and for the first request of each case and source
``wrapped-no-duration`` (metadata without the ``duration`` key) and ``wrapped-naive-pixels``
(per-request ``min_pixels``/``max_pixels`` copied from NVIDIA's ``AutoProcessor`` call). Golden
requests run ``wrapped`` only.

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
GPU_TIMEOUT_S = 15 * 60  # also the spend bound: 2 x 15 min L40S
BUILD_TIMEOUT_S = 15 * 60
SAMPLE_DIR = "samples/wse"
TMP_DIR = "gatec-tmp"

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
ENV = {"PYTHONDONTWRITEBYTECODE": "1", "VLLM_CACHE_ROOT": "/tmp/vllm-cache",
       "TOKENIZERS_PARALLELISM": "false"}


def engine_kwargs(kind: str, model_dir: str) -> dict:
    from flora_deploy.nv.capture import assert_engine_kwargs

    kwargs = dict(
        model=model_dir,
        tokenizer=model_dir,
        dtype="bfloat16",
        max_model_len=32768,
        limit_mm_per_prompt={"image": 0, "video": 3},
        mm_processor_kwargs=dict(NV_VIDEO_KWARGS[kind]),
        mm_processor_cache_gb=0,  # required (nv.capture.REQUIRED_VLLM_ENGINE_KWARGS)
        generation_config="vllm",
        gdn_prefill_backend="triton",
        enforce_eager=True,  # no graphs / compile: token ids do not depend on them
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        seed=0,
    )
    assert_engine_kwargs(kwargs)
    return kwargs


def _build_model_dir(kind: str):
    """Fresh ``/tmp/gatec-model-<kind>`` (idempotent: a reused warm container rebuilds it)."""
    import shutil
    from pathlib import Path

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import snapshot_path

    base = Path(snapshot_path(PINS_BY_KEY["tempseg_base"]))
    flora = Path(snapshot_path(PINS_BY_KEY[kind]))
    out = Path(f"/tmp/gatec-model-{kind}")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
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


def _nv_and_proc(kind: str):
    from pathlib import Path

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import snapshot_path
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family

    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"])))
    return nv, nv.models.processor(kind, Path(snapshot_path(PINS_BY_KEY[kind])))


@app.function(image=vllm_image(), cpu=2.0, memory=12 * 1024, timeout=BUILD_TIMEOUT_S,
              volumes={str(WEIGHTS_MOUNT): weights, str(GOLDEN_MOUNT): golden}, env=ENV)
def build_part(kind: str, part: str, run_id: str, sample_name: str, max_requests: int = 2,
               quota: int = 6) -> dict:
    """Stage 1 for one ``kind`` and ``part`` (``cases:synthetic``, ``cases:sample`` or
    ``golden:<pool>``); returns the pickle's Volume path, counts and errors (class names)."""
    import pickle
    import time
    from pathlib import Path

    from flora_deploy.validate import gatec_driver as D

    t0 = time.time()
    nv, proc = _nv_and_proc(kind)
    procs = {kind: proc}
    what, arg = part.split(":", 1)
    errors: list[dict] = []
    if what == "cases":
        if arg == "synthetic":
            source = D.synthetic_source()
        else:
            frames, rate = D.decode_all(Path(GOLDEN_MOUNT) / SAMPLE_DIR / sample_name)
            source = D.decoded_source(Path(sample_name).stem, frames, rate)
        records = D.build_cases(nv, procs, kind, [source], max_requests)
    elif what == "golden":
        eps = D.golden_episodes(GOLDEN_MOUNT, D.load_golden_manifest(GOLDEN_MOUNT))
        records, errors = D.build_golden(nv, procs, kind, {arg: eps[arg]}, quota)
    else:
        raise ValueError(f"unknown part {part!r}")
    rel = f"{TMP_DIR}/{run_id}/{kind}-{part.replace(':', '-')}.pkl"
    path = Path(GOLDEN_MOUNT) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=5)
    golden.commit()
    return dict(kind=kind, part=part, path=rel, n=len(records), errors=errors,
                sources=sorted({r.source for r in records}), seconds=round(time.time() - t0, 1))


@app.function(image=vllm_image(), gpu=GPU, cpu=4.0, memory=48 * 1024, timeout=GPU_TIMEOUT_S,
              volumes={str(WEIGHTS_MOUNT): weights, str(GOLDEN_MOUNT): golden}, env=ENV)
def run_gate_c(kind: str, parts: list[str]) -> dict:
    """Stage 2: render every built request of ``kind`` through vLLM and compare with HF."""
    import copy
    import json
    import os
    import pickle
    import subprocess
    import time
    from dataclasses import asdict
    from pathlib import Path

    import vllm
    from vllm import LLM, SamplingParams

    from flora_deploy.nv.capture import vllm_prompt
    from flora_deploy.validate import gatec as G

    t0 = time.time()
    timings: dict[str, float] = {}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()

    golden.reload()
    records = []
    for rel in parts:
        with open(Path(GOLDEN_MOUNT) / rel, "rb") as f:
            records += pickle.load(f)
    _, proc = _nv_and_proc(kind)
    timings["load_s"] = round(time.time() - t0, 1)

    model_dir, base, _ = _build_model_dir(kind)
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

    rows, pixel_dtypes = [], set()
    t2 = time.time()
    for r in records:
        golden_req = r.case_id.startswith("golden-")
        try:
            fixed = vllm_prompt(r.request)
        except Exception as exc:  # the wrap's guards refused the prompt
            rows.append(G.compare(case_id=r.case_id, index=r.index, source=r.source,
                                  variant="wrapped", hf_ids=r.hf_ids, hf_grids=r.hf_grids,
                                  vllm_ids=None, vllm_grids=None, sp=sp, merge_size=merge,
                                  error=G.sanitize_error(exc)).to_dict())
            continue
        variants = [("wrapped", fixed)]
        if not golden_req:
            bare = {"prompt": r.request.prompt,
                    "multi_modal_data": copy.deepcopy(fixed["multi_modal_data"])}
            variants.append(("bare", bare))
            if r.index == 0:
                no_dur = copy.deepcopy(fixed)
                no_dur["multi_modal_data"]["video"] = [
                    (f, {k: v for k, v in md.items() if k != "duration"})
                    for f, md in fixed["multi_modal_data"]["video"]]
                naive = {**fixed, "mm_processor_kwargs": dict(NAIVE_PIXEL_KWARGS)}
                variants += [("wrapped-no-duration", no_dur), ("wrapped-naive-pixels", naive)]
        for name, prompt in variants:
            extras: dict = {}
            pix = gen = None
            try:
                (ei,) = llm._preprocess_cmpl([prompt])
                ids = list(ei["prompt_token_ids"])
                items = ei["mm_kwargs"]["video"]
                grids = _grids(items)
                parts_ = [item["pixel_values_videos"].data for item in items]
                pixel_dtypes |= {str(p.dtype) for p in parts_}
                pix = G.pixels_equal_per_video(r.hf_pixels, parts_)
                extras["placeholders"] = [[p.offset, p.length]
                                          for p in ei["mm_placeholders"]["video"]]
                if name == "wrapped" or r.index == 0:
                    (out,) = llm.generate([prompt], greedy, use_tqdm=False)
                    gen = list(out.prompt_token_ids) == ids
                err = None
            except Exception as exc:  # recorded, not raised: the bare shape is expected to fail
                ids, grids, err = None, None, G.sanitize_error(exc)
            c = G.compare(case_id=r.case_id, index=r.index, source=r.source, variant=name,
                          hf_ids=r.hf_ids, hf_grids=r.hf_grids, vllm_ids=ids, vllm_grids=grids,
                          sp=sp, merge_size=merge, pixels_equal=pix, generate_ids_equal=gen,
                          error=err, extras=extras)
            rows.append(c.to_dict())
    timings["compare_s"] = round(time.time() - t2, 1)

    for rel in parts:  # the pickles hold NC-licensed frames: do not leave them behind
        try:
            os.remove(Path(GOLDEN_MOUNT) / rel)
        except OSError:
            pass
    for d in {(Path(GOLDEN_MOUNT) / rel).parent for rel in parts}:
        try:
            d.rmdir()  # only once the other kind's container removed its pickles too
        except OSError:
            pass
    golden.commit()
    timings["total_s"] = round(time.time() - t0, 1)

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
        vllm_pixel_dtypes=sorted(pixel_dtypes),
        n_requests=len(records),
        rows=rows,
        timings=timings,
    )
