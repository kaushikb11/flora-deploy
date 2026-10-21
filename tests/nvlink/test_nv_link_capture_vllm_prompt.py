"""Gate C fix: wrapping NVIDIA's bare ``<|video_pad|>`` leaves HF's model inputs unchanged.

For every NVIDIA input-builder case (``validate.gatec_driver.CASES``), the request captured by
``CaptureProcessor`` is replayed through NVIDIA's real processor with the *wrapped* prompt that
``nv.capture.vllm_prompt`` hands to vLLM. ``input_ids``, ``video_grid_thw`` and
``pixel_values_videos`` must equal NVIDIA's own PIL-input call bit for bit. A model of vLLM
0.24.0's video prompt update (``validate.gatec.simulate_vllm_expansion``) applied to the wrapped
prompt must give exactly HF's ``input_ids``; applied to the bare prompt it finds no placeholder.
The real-vLLM check runs on Modal (``scripts/gatec.py``, ``docs/gate-c.md``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from flora_deploy.nv import Family
from flora_deploy.nv.capture import replay, vllm_prompt
from flora_deploy.nv.linker import load_family
from flora_deploy.validate import gatec as G
from flora_deploy.validate import gatec_driver as D

pytestmark = pytest.mark.nvlink

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def nv_dirs():
    import os

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    dirs = {k: root / snapshot_dirname(PINS_BY_KEY[f"{k}_code"])
            for k in ("tempseg", "captioner")}
    for key, path in dirs.items():
        if not (path / "processor_config.json").is_file():
            pytest.skip(f"{key} processor files not fetched (fetch_nv_code.py --with-processor)")
    return dirs


@pytest.fixture(scope="module")
def nv(nv_dirs):
    return load_family(Family.SEGCAP, nv_dirs["tempseg"])


@pytest.fixture(scope="module")
def procs(nv, nv_dirs):
    return {k: nv.models.processor(k, d) for k, d in nv_dirs.items()}


@pytest.mark.parametrize("case_id", sorted(D.CASES))
def test_wrapped_prompt_is_hf_equivalent(nv, procs, case_id):
    built = D.build_requests(nv, procs, case_id, D.synthetic_source(duration_s=40.0),
                             max_requests=2)
    assert built
    proc = procs[D.processor_kind(case_id)]
    tok = proc.tokenizer
    sp = G.SpecialIds(*(tok.convert_tokens_to_ids(t) for t in
                        ("<|vision_start|>", "<|vision_end|>", "<|video_pad|>")))
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    tps = proc.video_processor.temporal_patch_size
    merge = proc.video_processor.merge_size
    for b in built:
        wrapped = vllm_prompt(b.request)["prompt"]
        assert wrapped != b.request.prompt
        out = replay(proc, dataclasses.replace(b.request, prompt=wrapped))
        for key in D.HF_KEYS:
            assert torch.equal(out[key], b.hf[key]), key
        hf_ids = b.hf["input_ids"][0].tolist()
        grids = b.hf["video_grid_thw"].tolist()
        stamps = [G.vllm_timestamps(md["frames_indices"], md["fps"], tps)
                  for _, md in b.request.videos]
        sim = G.simulate_vllm_expansion(encode(wrapped), grids, stamps, encode, sp, merge)
        assert sim == hf_ids
        assert G.grids_match_runs(hf_ids, grids, sp, merge)
        with pytest.raises(RuntimeError, match="Found 0 video placeholders"):
            G.simulate_vllm_expansion(encode(b.request.prompt), grids, stamps, encode, sp,
                                      merge)
