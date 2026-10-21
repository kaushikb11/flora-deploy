# Gate C: vLLM 0.24.0 model inputs vs NVIDIA's HF processor

**Verdict: PASS with a capture-layer fix.** NVIDIA's prompts, as captured, are **rejected** by
vLLM 0.24.0 (0/44 requests render). With the fix (`nv.capture.vllm_prompt`, which wraps each bare
`<|video_pad|>` as `<|vision_start|><|video_pad|><|vision_end|>`), vLLM's `prompt_token_ids`,
per-video `video_grid_thw` and `pixel_values_videos` (in bf16) are **identical** to the
`input_ids` / `video_grid_thw` / `pixel_values_videos` of NVIDIA's own processor call, for all 44
requests over 11 builder cases and 2 frame sources.

Run: 2026-10-21, `uv run modal run scripts/gatec.py` (ephemeral app `flora-gatec`, two L40S
containers, driver 610.57.04), vllm 0.24.0, transformers 5.8.1, torch 2.11.0+cu130.
Results JSON (counts, positions, grids only) stays outside the repo
(`~/.cache/flora-deploy/gatec/results.json`).

## What was compared

For each case, `validate.gatec_driver` runs NVIDIA's own pipeline function (`pipeline.segment`,
`caption_three`, `caption_single`) with the processor built by NVIDIA's
`models.processor(kind, path)` from the pinned Flora processor files. The only substitutions are
the frame decoder (frames come from a source below) and `generate` (canned output). It runs twice:
once through a spy around the real processor (the **HF reference**: NVIDIA's PIL-input call), once
through `CaptureProcessor` (the `CapturedRequest`: prompt with bare `<|video_pad|>`, uint8 THWC
frames, metadata `{total_num_frames, fps, frames_indices, duration=None, do_sample_frames=False}`).
The driver checks the two runs made the same calls with the same prompt, frames and metadata.

Each captured request then goes through vLLM's offline path in the same container:
`LLM._preprocess_cmpl([prompt])` (exactly what `LLM.generate` calls) gives
`prompt_token_ids`, the per-item `video_grid_thw` and `pixel_values_videos` and the placeholder
ranges; `LLM.generate(..., max_tokens=1, temperature=0)` confirms the engine received the same
ids (true for every request it was run on).

Pass = ids identical **and** grids identical **and** the vLLM ids contain exactly the frame groups
the grids imply (`T` groups of `vision_start, video_pad*(H*W/4), vision_end` per video).

**Cases** (2 processor calls each; synthetic duration 40 s): temporal human, human + instruction
suffix, YAM 1-view, YAM 3-view, Dex 3-view, yam32 3-view (64 temporal groups, 10.7k tokens);
three-view captions YAM and Dex (+ instruction); single-view captions human, human + instruction,
robot.

**Frame sources:** `synthetic` (640x480 gradient + seeded noise) and `bbb_360_10s` (Big Buck
Bunny 10 s, 640x360 H.264 30 fps CFR, CC BY 3.0 Blender Foundation; staged at
`flora-golden:samples/wse/bbb_360_10s.mp4`, sha256 `77145c94…2016`; frames looped, wrist views
mirrored/flipped). Every video's frames are off the 32-px grid (e.g. 173x231, 367x435, 443x591,
360x640), so the processor's smart-resize is exercised in both engines.

## Engine settings

| Setting | Value | Why |
|---|---|---|
| model dir | `/tmp/gatec-model-<kind>`: Qwen3.5-9B@c2022362 `config.json`, index, 4 shards (symlinks) + Flora `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (TempSeg@114a1076 or Captioner@fef0bbfe) | base `preprocessor_config.json`/`video_preprocessor_config.json` deliberately excluded |
| `mm_processor_kwargs` (engine) | tempseg `{"fps": 4, "max_frames": 128}`, captioner `{"fps": 8, "max_frames": 40}` | NVIDIA's `Qwen3VLVideoProcessor(fps=…, max_frames=…)` overrides (equal to the files' values) |
| `limit_mm_per_prompt` | `{"image": 0, "video": 3}` | |
| other | `dtype=bfloat16`, `max_model_len=32768`, `mm_processor_cache_gb=0`, `generation_config="vllm"`, `gdn_prefill_backend="triton"`, `enforce_eager=True`, `max_num_seqs=8`, `seed=0` | |

vLLM's resolved video processor equals NVIDIA's on every attribute checked (`size`
{shortest 4096, longest 25165824}, `fps`, `max_frames`, `min_frames` 4, `patch_size` 16,
`temporal_patch_size` 2, `merge_size` 2, `resample` 3, mean/std 0.5, rescale, resize/normalize/
convert flags). Special-token ids agree between Flora's tokenizer and the base `config.json`
(`vision_start` 248053, `vision_end` 248054, `video_pad` 248057). NVIDIA's `AutoProcessor`
`min_pixels=1024, max_pixels=589824` are image-processor settings (they equal
`processor_config.json`'s image section) and are not mirrored.

## Results

Per case and variant (requests passed / run):

| case | wrapped (fix) | bare (as captured) | wrapped-naive-pixels | wrapped-no-duration |
|---|---|---|---|---|
| caption-single-human | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-single-human-instruction | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-single-robot | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-three-dex | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-three-yam | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-dex-3view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-human | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-human-instruction | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-yam-1view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-yam-3view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-yam32-3view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |

`pixel_values_videos`: vLLM == HF cast to bf16 (vLLM's model dtype), bit for bit, in every passing
row. `wrapped-naive-pixels` adds per-request `min_pixels=1024, max_pixels=589824` (a literal copy
of NVIDIA's `AutoProcessor` kwargs); it did not change any grid here, but it buys nothing and
should not be sent.

Per request, fixed prompt (`=` equal, `≠` different):

| case | # | source | variant | ids | grid | runs | HF len | vLLM len | first diff | grids (T,H,W) | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| caption-single-human | 0 | bbb_360_10s | wrapped | = | = | = | 5656 | 5656 |  | 6,22,40;12,22,40;6,22,40 |  |
| caption-single-human | 1 | bbb_360_10s | wrapped | = | = | = | 6112 | 6112 |  | 6,22,40;14,22,40;6,22,40 |  |
| caption-single-human | 0 | synthetic | wrapped | = | = | = | 6424 | 6424 |  | 6,28,36;12,28,36;6,28,36 |  |
| caption-single-human | 1 | synthetic | wrapped | = | = | = | 6944 | 6944 |  | 6,28,36;14,28,36;6,28,36 |  |
| caption-single-human-instruction | 0 | bbb_360_10s | wrapped | = | = | = | 5666 | 5666 |  | 6,22,40;12,22,40;6,22,40 |  |
| caption-single-human-instruction | 1 | bbb_360_10s | wrapped | = | = | = | 6122 | 6122 |  | 6,22,40;14,22,40;6,22,40 |  |
| caption-single-human-instruction | 0 | synthetic | wrapped | = | = | = | 6434 | 6434 |  | 6,28,36;12,28,36;6,28,36 |  |
| caption-single-human-instruction | 1 | synthetic | wrapped | = | = | = | 6954 | 6954 |  | 6,28,36;14,28,36;6,28,36 |  |
| caption-single-robot | 0 | bbb_360_10s | wrapped | = | = | = | 4283 | 4283 |  | 12,22,40;6,22,40 |  |
| caption-single-robot | 1 | bbb_360_10s | wrapped | = | = | = | 4739 | 4739 |  | 6,22,40;14,22,40 |  |
| caption-single-robot | 0 | synthetic | wrapped | = | = | = | 4859 | 4859 |  | 12,28,36;6,28,36 |  |
| caption-single-robot | 1 | synthetic | wrapped | = | = | = | 5379 | 5379 |  | 6,28,36;14,28,36 |  |
| caption-three-dex | 0 | bbb_360_10s | wrapped | = | = | = | 6465 | 6465 |  | 6,32,28;12,32,28;6,32,28 |  |
| caption-three-dex | 1 | bbb_360_10s | wrapped | = | = | = | 6990 | 6990 |  | 6,32,28;14,32,28;6,32,28 |  |
| caption-three-dex | 0 | synthetic | wrapped | = | = | = | 5121 | 5121 |  | 6,28,24;12,28,24;6,28,24 |  |
| caption-three-dex | 1 | synthetic | wrapped | = | = | = | 5534 | 5534 |  | 6,28,24;14,28,24;6,28,24 |  |
| caption-three-yam | 0 | bbb_360_10s | wrapped | = | = | = | 5111 | 5111 |  | 6,24,28;12,24,28;6,24,28 |  |
| caption-three-yam | 1 | bbb_360_10s | wrapped | = | = | = | 5524 | 5524 |  | 6,24,28;14,24,28;6,24,28 |  |
| caption-three-yam | 0 | synthetic | wrapped | = | = | = | 5111 | 5111 |  | 6,24,28;12,24,28;6,24,28 |  |
| caption-three-yam | 1 | synthetic | wrapped | = | = | = | 5524 | 5524 |  | 6,24,28;14,24,28;6,24,28 |  |
| temporal-dex-3view | 0 | bbb_360_10s | wrapped | = | = | = | 5557 | 5557 |  | 32,26,24 |  |
| temporal-dex-3view | 1 | bbb_360_10s | wrapped | = | = | = | 5556 | 5556 |  | 32,26,24 |  |
| temporal-dex-3view | 0 | synthetic | wrapped | = | = | = | 5557 | 5557 |  | 32,26,24 |  |
| temporal-dex-3view | 1 | synthetic | wrapped | = | = | = | 5556 | 5556 |  | 32,26,24 |  |
| temporal-human | 0 | bbb_360_10s | wrapped | = | = | = | 1811 | 1811 |  | 32,10,16 |  |
| temporal-human | 1 | bbb_360_10s | wrapped | = | = | = | 1810 | 1810 |  | 32,10,16 |  |
| temporal-human | 0 | synthetic | wrapped | = | = | = | 1651 | 1651 |  | 32,10,14 |  |
| temporal-human | 1 | synthetic | wrapped | = | = | = | 1650 | 1650 |  | 32,10,14 |  |
| temporal-human-instruction | 0 | bbb_360_10s | wrapped | = | = | = | 1820 | 1820 |  | 32,10,16 |  |
| temporal-human-instruction | 1 | bbb_360_10s | wrapped | = | = | = | 1819 | 1819 |  | 32,10,16 |  |
| temporal-human-instruction | 0 | synthetic | wrapped | = | = | = | 1660 | 1660 |  | 32,10,14 |  |
| temporal-human-instruction | 1 | synthetic | wrapped | = | = | = | 1659 | 1659 |  | 32,10,14 |  |
| temporal-yam-1view | 0 | bbb_360_10s | wrapped | = | = | = | 1845 | 1845 |  | 32,10,16 |  |
| temporal-yam-1view | 1 | bbb_360_10s | wrapped | = | = | = | 1844 | 1844 |  | 32,10,16 |  |
| temporal-yam-1view | 0 | synthetic | wrapped | = | = | = | 1685 | 1685 |  | 32,10,14 |  |
| temporal-yam-1view | 1 | synthetic | wrapped | = | = | = | 1684 | 1684 |  | 32,10,14 |  |
| temporal-yam-3view | 0 | bbb_360_10s | wrapped | = | = | = | 5493 | 5493 |  | 32,22,28 |  |
| temporal-yam-3view | 1 | bbb_360_10s | wrapped | = | = | = | 5492 | 5492 |  | 32,22,28 |  |
| temporal-yam-3view | 0 | synthetic | wrapped | = | = | = | 5493 | 5493 |  | 32,22,28 |  |
| temporal-yam-3view | 1 | synthetic | wrapped | = | = | = | 5492 | 5492 |  | 32,22,28 |  |
| temporal-yam32-3view | 0 | bbb_360_10s | wrapped | = | = | = | 10710 | 10710 |  | 64,22,28 |  |
| temporal-yam32-3view | 1 | bbb_360_10s | wrapped | = | = | = | 10709 | 10709 |  | 64,22,28 |  |
| temporal-yam32-3view | 0 | synthetic | wrapped | = | = | = | 10710 | 10710 |  | 64,22,28 |  |
| temporal-yam32-3view | 1 | synthetic | wrapped | = | = | = | 10709 | 10709 |  | 64,22,28 |  |

## The mismatch: a bare `<|video_pad|>` is not a vLLM placeholder

Every `bare` request fails in vLLM with
`AssertionError: Failed to apply prompt replacement for mm_items['video'][0]`; no
`prompt_token_ids` are produced. The cause, in vLLM 0.24.0 `model_executor/models/qwen3_vl.py`:

- `Qwen3VLMultiModalProcessor._get_prompt_updates` registers the video `PromptReplacement`
  with target `"<|vision_start|><|video_pad|><|vision_end|>"`, and
  `_replace_video_token_placeholders` matches the same three-token sequence in token-id space.
- NVIDIA's builders put a bare `proc.video_token` into the user message (always right after a
  newline), and the chat template emits it verbatim, so the tokenized prompt has a lone
  `video_pad` (id 248057) per video and zero `vision_start`/`vision_end` tokens. vLLM finds 0 of
  the N expected placeholders.

So the difference is not a silently different token sequence: vLLM refuses the request. There is
no position-level diff to report for `bare`; HF's sequence has, per video, `T` groups of
`<t seconds>` text + `vision_start` + `video_pad*(H*W/4)` + `vision_end`, and vLLM produces none.

HF's `Qwen3VLProcessor.__call__` (transformers 5.8.1) accepts both shapes: for each video in
order it replaces the first `vision_start+video_pad+vision_end` if the text contains one,
otherwise the first bare `video_pad` (comment in HF: "vllm may input video token directly"), with
the per-frame string `<{t:.1f} seconds><|vision_start|><|placeholder|>*n<|vision_end|>`.

## The fix and its proof

`flora_deploy.nv.capture.wrap_bare_video_tokens(prompt, n_videos)` (used by
`vllm_prompt(request)`) replaces each bare `<|video_pad|>` with
`<|vision_start|><|video_pad|><|vision_end|>`. It refuses (raises `CaptureError`) unless the
prompt has exactly `n_videos` video tokens, no vision start/end tokens, and a newline before
every video token. All NVIDIA builders satisfy this.

**HF model inputs are unchanged** (so NVIDIA's reference is still the right reference):

1. *Argument.* With no `vision_start`/`vision_end` in the input, HF takes the bare branch for every
   video on the original prompt and the wrapped branch for every video on the fixed prompt. Both
   replace the k-th video marker, in order, with the same string, because each expansion contains
   no `video_pad` (it uses `<|placeholder|>` until the end) and ends in `vision_end`, so no later
   bare token can appear wrapped. The two expanded texts are equal character for character, hence
   so are `input_ids`. Frames and metadata are untouched, so `pixel_values_videos` and
   `video_grid_thw` are too.
2. *Test.* `tests/nvlink/test_nv_link_capture_vllm_prompt.py` replays every case's captured
   request through NVIDIA's real processor with the wrapped prompt: `input_ids`,
   `video_grid_thw` and `pixel_values_videos` equal NVIDIA's original PIL-input call bit for bit
   (11 cases, 2 calls each). The existing `test_nv_link_capture.py` (captured ndarray replay ==
   NVIDIA's call) still passes.

**vLLM then matches HF** (the gate itself, on Modal, table above): 44/44 requests identical ids,
grids and pixels. Why it holds: vLLM substitutes each triplet by, per temporal group,
`tokenize("<t seconds>")` + `vision_start` + pads + `vision_end`, with `t` from
`_calculate_timestamps(frames_indices, fps, temporal_patch_size)` (the same formula as HF). HF
tokenizes the timestamp string in context, vLLM on its own; they agree because the character
before each video is a newline (and between frames a `vision_end`), so no BPE merge crosses the
boundary. That is why the fix requires a preceding newline. The nvlink test also checks a pure
model of vLLM's expansion (`validate.gatec.simulate_vllm_expansion`) against HF on the wrapped
prompt, and that the bare prompt finds no placeholder, so a change on either side shows up
without a GPU.

## `duration=None`

vLLM 0.24.0 accepts it. `Qwen3VLMultiModalProcessor._call_hf_processor` builds
`VideoMetadata(**metadata)` (minus `do_sample_frames`), and the timestamps use only
`frames_indices` and `fps` when `do_sample_frames` is False (`total_num_frames` is needed only
for sampling). Captured metadata with `duration=None` passed in every `wrapped` row; dropping the
key (`wrapped-no-duration`) gives the identical ids, grids and pixels. `do_sample_frames=False` in
the metadata dict is honoured (vLLM copies it into the per-video HF kwargs).

## Incidental finding: three-view caption geometry depends on the platform's PNG encoder

The three-view caption frame sizes differ between this macOS arm64 dev machine and the Modal
x86_64 container for the same synthetic frames (e.g. caption-three-dex synthetic: grid 22x20 per
frame locally, 28x24 on Modal). Within each machine HF and vLLM agree, so Gate C is unaffected.
The cause is NVIDIA's `robot_caption_geometry.normalized`: it picks the first scale whose packed
PNG byte size fits `PACKED_IMAGE_LIMIT_BYTES`, and compressed PNG size depends on the zlib build
in the Pillow wheel. This confirms PLAN D11 (decode and image work on x86_64 with exact Pillow):
golden data and production must share the image stack, not just the Pillow version.

## Spend and image

- Modal billing for app `flora-gatec` today: L40S $0.66 + CPU $0.06 + memory $0.13 ≈ **$0.86**
  (cap $6). One earlier attempt failed locally (sample download 403, fixed with a User-Agent)
  before any GPU container started.
- Per container: request building 124 s (tempseg) / 297 s (captioner), engine start ≈ 200 s
  (weights 13–22 s, profiling/warmup ≈ 150 s), compare ≈ 65 s.
- `img_vllm` (`vllm_image()`) built as pinned with no changes (`nvidia/cuda:13.0.2-devel-ubuntu24.04`
  + vllm==0.24.0, transformers==5.8.1, torch 2.11.0+cu130 from vLLM's own pin).

## Reproduce

```
uv run modal run scripts/gatec.py            # both kinds in parallel, prints the tables
uv run pytest -q -m nvlink tests/nvlink/test_nv_link_capture_vllm_prompt.py   # no GPU
```
