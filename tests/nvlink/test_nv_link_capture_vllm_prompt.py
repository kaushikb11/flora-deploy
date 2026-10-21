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
        if wrapped == b.request.prompt:  # no `assert a != b`: pytest would print NVIDIA's prompt
            pytest.fail("the wrap was a no-op")
        out = replay(proc, dataclasses.replace(b.request, prompt=wrapped))
        for key in D.HF_KEYS:
            assert torch.equal(out[key], b.hf[key]), key
        hf_ids = b.hf["input_ids"][0].tolist()
        grids = b.hf["video_grid_thw"].tolist()
        stamps = [G.vllm_timestamps(md["frames_indices"], md["fps"], tps)
                  for _, md in b.request.videos]
        sim = G.simulate_vllm_expansion(encode(wrapped), grids, stamps, encode, sp, merge)
        if sim != hf_ids:  # no `assert a == b`: pytest would print ids that decode to the prompt
            pytest.fail(f"simulated vLLM ids differ from HF's at {G.first_diff(sim, hf_ids)} "
                        f"(lengths {len(sim)} vs {len(hf_ids)})")
        assert G.grids_match_runs(hf_ids, grids, sp, merge)
        with pytest.raises(RuntimeError, match="Found 0 video placeholders"):
            G.simulate_vllm_expansion(encode(b.request.prompt), grids, stamps, encode, sp,
                                      merge)


def _verdict(proc, b, out):
    """The Gate C verdict for ``out`` (a processor output standing in for vLLM's) vs ``b.hf``."""
    tok = proc.tokenizer
    sp = G.SpecialIds(*(tok.convert_tokens_to_ids(t) for t in
                        ("<|vision_start|>", "<|vision_end|>", "<|video_pad|>")))
    grids = b.hf["video_grid_thw"].tolist()
    digests = G.video_pixel_digests(b.hf["pixel_values_videos"], grids)
    vgrids = out["video_grid_thw"].tolist()
    rows = [t * h * w for t, h, w in vgrids]
    parts = list(torch.split(out["pixel_values_videos"], rows))
    ids = out["input_ids"][0].tolist()
    # Stand-ins for the checks only real vLLM can run: generate saw these ids, and each video's
    # placeholder range spans its frame runs (first vision_start .. last vision_end).
    runs, placeholders, i = G.frame_runs(ids, sp), [], 0
    for t, _, _ in vgrids:
        first, last = runs[i], runs[i + t - 1]
        placeholders.append([first.start, last.start + last.pads + 2 - first.start])
        i += t
    return G.compare(case_id="x", index=0, source="s", variant="wrapped",
                     hf_ids=b.hf["input_ids"][0].tolist(), hf_grids=grids,
                     vllm_ids=ids, vllm_grids=vgrids, sp=sp,
                     merge_size=proc.video_processor.merge_size,
                     pixels_equal=G.pixels_equal_per_video(digests, [p.bfloat16() for p in parts]),
                     generate_ids_equal=True, placeholders=placeholders)


@pytest.mark.parametrize("case_id", ["caption-three-yam", "temporal-yam-3view"])
def test_pixel_only_changes_fail_the_verdict(nv, procs, case_id):
    """Wrong frame order or a different resample filter keep ids and grids but change pixels:
    the verdict must be FAIL (M1)."""
    (b,) = D.build_requests(nv, procs, case_id, D.synthetic_source(duration_s=40.0),
                            max_requests=1)
    proc = procs[D.processor_kind(case_id)]
    same = _verdict(proc, b, replay(proc, b.request))
    assert same.passed and same.pixels_equal == [True] * len(b.request.videos)
    reversed_ = dataclasses.replace(b.request, videos=[(f[::-1].copy(), md)
                                                       for f, md in b.request.videos])
    nearest = replay(proc, b.request, resample=0)
    for out in (replay(proc, reversed_), nearest):
        v = _verdict(proc, b, out)
        assert v.ids_equal and v.grids_equal and v.runs_match_grid
        assert not all(v.pixels_equal) and not v.passed
    if len(b.request.videos) > 1:  # videos swapped across items (same shapes in this case)
        swapped = dataclasses.replace(b.request, videos=[b.request.videos[-1],
                                                         *b.request.videos[1:-1],
                                                         b.request.videos[0]])
        v = _verdict(proc, b, replay(proc, swapped))
        assert not v.passed
