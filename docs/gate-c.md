# Gate C: vLLM 0.24.0 model inputs vs NVIDIA's HF processor

**Verdict: PASS with a capture-layer fix, under a required engine setting
(`mm_processor_cache_gb=0`).** NVIDIA's prompts, as captured, are **rejected** by vLLM 0.24.0
(0/56 builder-case requests render). With the fix (`nv.capture.vllm_prompt`, which wraps each bare
`<|video_pad|>` as `<|vision_start|><|video_pad|><|vision_end|>`), three things are **identical**
to NVIDIA's own processor call for **all 127 requests**:
- vLLM's `prompt_token_ids` and NVIDIA's `input_ids`;
- the per-video `video_grid_thw`;
- the per-video `pixel_values_videos`, compared in vLLM's dtype, bf16.

The 127 requests are 56 from 14 builder cases on 2 frame sources, plus 71 from real golden
episodes in all 6 TempSeg/Captioner pools.

What is compared: prompt token ids, grids and pixels. M-RoPE position ids are **not** compared.
Both engines derive them from exactly these ids and grids, but through different
implementations (vLLM `_get_mrope_input_positions`, HF `get_rope_index`). Checking those is left
to Gate D (logits parity).

Run: 2026-10-21, `uv run modal run scripts/gatec.py`.
- Infrastructure: ephemeral app `flora-gatec`, 16 CPU builder containers plus two L40S
  containers, driver 610.57.04.
- Versions: vllm 0.24.0, transformers 5.8.1, torch 2.11.0+cu130.
- The results JSON (counts, positions, grids, digests only, no prompt or task text) stays outside
  the repo, at `~/.cache/flora-deploy/gatec/results.json`.

## What was compared

`validate.gatec_driver` runs NVIDIA's own pipeline function (`pipeline.segment`,
`caption_three` or `caption_single`). The processor is built by NVIDIA's
`models.processor(kind, path)` from the pinned Flora processor files. Each pipeline runs twice:
- once through a spy around the real processor. This is the **HF reference**, NVIDIA's
  PIL-input call;
- once through `CaptureProcessor`. This gives the `CapturedRequest`: the prompt with bare
  `<|video_pad|>`, uint8 THWC frames, and the metadata
  `{total_num_frames, fps, frames_indices, duration=None, do_sample_frames=False}`.

The driver checks that both runs made the same calls with the same prompt, frames and metadata.
`generate` is always canned. The pipeline is stopped after the wanted number of processor calls.

**Request sources**

| Source | Decoder | Requests |
|---|---|---|
| `synthetic`: 640x480 gradient plus seeded noise, 40 s | fake `decode_selected` | 2 per case |
| `bbb_360_10s`: Big Buck Bunny 10 s, 640x360 H.264 30 fps CFR, CC BY 3.0 Blender Foundation | fake `decode_selected` | 2 per case |
| **golden v1**: Volume `flora-golden`, `golden/v1/` (`docs/golden-manifest-v1.md`) | NVIDIA's **real** `media.probe` and `decode_selected` | up to 6 per pool and kind |

Notes on the sources:
- The BBB sample is staged at `flora-golden:samples/wse/bbb_360_10s.mp4` (sha256 `77145c94…2016`).
  Its frames are looped, and the wrist views are mirrored or flipped.
- For the golden source, decoding is memoized, so the capture run reuses the spy run's frames.
- Every frame size is off the 32-px grid, so both engines exercise the processor's smart-resize.

**Builder cases (14):**
- temporal: human, human + instruction, YAM 1-view, YAM 3-view, YAM 3-view + instruction, Dex
  3-view, Tianji 3-view, and yam32 3-view (64 temporal groups, 10.7k tokens);
- three-view captions: YAM, Dex + instruction, Tianji + instruction;
- single-view captions: human, human + instruction, robot.

**Golden coverage.** Every staged, oracle-eligible episode of each pool is eligible. Requests are
spread over episodes: at most `ceil(6 / #episodes)` per episode, and at most 3. Every other
episode uses its dataset task text as the instruction. Captioner targets are three spans at 10%,
40% and 70% into the episode, each 1-4 s long. The golden requests run the `wrapped` variant
only.

| Pool | Episodes used | TempSeg requests | Captioner requests | Prompt length (tokens) | Media |
|---|---|---|---|---|---|
| `human` | 6 | 6 | 6 | 1811-7464 | HomER 1080p/720p/848x478 H.264 **VFR** (non-integer rates such as 29.99 fps and 59.94 fps); EgoDex 1080p AV1 |
| `single_view_robot` (profile YAM) | 6 | 6 | 6 | 1417-5899 | DROID 320x180 at 10 fps; Galaxea 720p AV1; droid_success 720p at 15 fps |
| `yam` | 6 | 6 | 6 | 5493-9736 | ManiRewardBench 680x480 at 10 fps; ALOHA 640x480 AV1 at 50 fps |
| `yam32` | 2 | 6 | 6 | 7923-14711 | 88.5 s and 116 s episodes, 64 temporal groups |
| `dex` (stand-in) | 3 | 6 | 6 | 5556-12136 | Unitree G1 640x480 AV1 |
| `tianji` (stand-in) | 2 | 5 | 6 | 5523-13607 | 640x480 H.264. Episode 75 has only 2 TempSeg windows |

The episodes used are the first ones in manifest order, `golden/v1/manifest.json`.

Deviation from PLAN §9's "20 golden requests per pool": the gate uses up to 6 per pool and kind,
11-12 per pool in total. The plan's 20 would mean 120 requests per kind on real media; the
coverage above spans every pool, codec, rate class (VFR, 10/15/30/50 fps) and window length at
about a third of the build cost. Raise `--quota` to 20 to run the plan's number.

**Rendering in vLLM.** Each captured request goes through vLLM's offline path in an L40S
container:
- `LLM._preprocess_cmpl([prompt])` is exactly what `LLM.generate` calls. It is a private API, so
  recheck it on any vLLM bump. It gives `prompt_token_ids`, the per-item `video_grid_thw` and
  `pixel_values_videos`, and the placeholder ranges.
- `LLM.generate(..., max_tokens=1, temperature=0)` confirms that the engine received the same
  ids. It was true for all 183 requests it ran on.

**PASS** requires all of the following (`validate.gatec.Comparison.passed`):
- no error;
- identical ids;
- identical grids;
- the vLLM ids contain exactly the frame groups the grids imply (`T` groups of
  `vision_start, video_pad*(H*W/4), vision_end` per video);
- **per-video pixels equal**: sha256 of HF's tensor cast to vLLM's dtype (bf16 here) equals that
  of vLLM's tensor. Any other dtype, or a missing comparison, fails. This is bf16 equality (the
  engine dtype), not fp32 equality;
- `generate` saw the same ids. For `wrapped` it must have run: a skipped `generate` is a FAIL.
  Diagnostic variants fail only on a negative result;
- **placeholder ranges sit on the frame runs** (`validate.gatec.placeholder_problem`): vLLM's
  `mm_placeholders['video']` has one `[offset, length]` per video. Each one must be inside the
  prompt, in order and non-overlapping. It must start at or before the video's first
  `vision_start` with no special token in between, end right after its `T`-th `vision_end`, and
  hold exactly `T*H*W/4` pads. Every pad must be inside a range. Equal ids are not enough on
  their own, because embeddings land where the placeholder ranges say. Required for `wrapped`;
- no stray bare `video_pad` in either sequence.

**Run verdict** (`validate.gatec.gate_verdict`). The gate is decided over the whole run, not
only over the rows that exist:
- every (kind, part) is built, with no build-stage error and at least one request. A pool whose
  builds all failed is a FAIL, never a silent `n/a`;
- every built request has a `wrapped` row;
- every builder case and golden pool has rows;
- every `wrapped` row passes.

`results.json` carries the verdict (problems, failed cases, a per-kind coverage table) and its
provenance: `run_id`, `git rev-parse HEAD` with a dirty flag, and `image_fingerprint("img_vllm")`
(`validate.gatec.results_provenance`).

**Status of the recorded run.** The 2026-10-21 run below predates these rules (review 3, C1/C3/
C4). It had no build errors (`errors: []` in every build), covered every pool, and ran
`generate` on every `wrapped` row. But it did not check placeholder ranges, and its
`results.json` has no git SHA or image fingerprint. Re-run `scripts/gatec.py` once on the
committed SHA to confirm under the full criteria.

Unit tests (`tests/unit/test_gatec_compare.py`) and nvlink tests with the real processor
(`tests/nvlink/test_nv_link_capture_vllm_prompt.py::test_pixel_only_changes_fail_the_verdict`)
cover three changes that keep ids and grids but change pixels. Each one must give FAIL:
- frames in the wrong order;
- a different resample filter (`resample=0` instead of 3);
- videos swapped across items.

## Engine settings

| Setting | Value | Why |
|---|---|---|
| model dir | `/tmp/gatec-model-<kind>`, rebuilt on every call: Qwen3.5-9B@c2022362 `config.json`, index and 4 shards (symlinks), plus Flora `processor_config.json`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` (TempSeg@114a1076 or Captioner@fef0bbfe) | base `preprocessor_config.json`/`video_preprocessor_config.json` deliberately excluded |
| `mm_processor_kwargs` (engine) | tempseg `{"fps": 4, "max_frames": 128}`, captioner `{"fps": 8, "max_frames": 40}` | NVIDIA's `Qwen3VLVideoProcessor(fps=…, max_frames=…)` overrides (equal to the files' values) |
| `mm_processor_cache_gb` | **0, required** | see below |
| `limit_mm_per_prompt` | `{"image": 0, "video": 3}` | |
| other | `dtype=bfloat16`, `max_model_len=32768`, `generation_config="vllm"`, `gdn_prefill_backend="triton"`, `enforce_eager=True`, `max_num_seqs=8`, `seed=0` | |

**Processor cache off is a precondition of this certificate.** With a processor cache (vLLM's
default is 4 GiB), vLLM 0.24.0 takes `_cached_apply_hf_processor`, which differs from the
uncached path in three ways:
- it tokenizes the prompt text-only;
- it processes missing items separately;
- it applies its string-target prompt updates with cached timestamps.

Gate C does not exercise that path; a cache-on variant was not run. The setting is therefore
enforced in code:
- `flora_deploy.nv.capture.REQUIRED_VLLM_ENGINE_KWARGS` (`{"mm_processor_cache_gb": 0}`) and
  `assert_engine_kwargs(kwargs)` raise unless the kwargs set it explicitly.
- The Gate C engine builder (`modal_app.gatec_app.engine_kwargs`) calls it.
- **Every production engine builder that carries Flora video (TempSeg, Captioner) must call
  `assert_engine_kwargs` on its final `LLM`/`EngineArgs` kwargs.**
- If production ever needs the cache or AsyncLLM pre-rendering (`render_cmpl_async`), Gate C
  must be extended first.

vLLM's resolved video processor equals NVIDIA's on every attribute checked:
- `size` {shortest 4096, longest 25165824}, `fps`, `max_frames`, `min_frames` 4;
- `patch_size` 16, `temporal_patch_size` 2, `merge_size` 2, `resample` 3;
- mean/std 0.5, rescale, and the resize/normalize/convert flags.

Special-token ids agree between Flora's tokenizer and the base `config.json`: `vision_start`
248053, `vision_end` 248054, `video_pad` 248057.

NVIDIA's `AutoProcessor` kwargs `min_pixels=1024, max_pixels=589824` are image-processor
settings. They equal the image section of `processor_config.json` and are not mirrored.

## Results

Per case and variant (requests passed / run), as printed by `render_report`:

| case | wrapped | bare | wrapped-naive-pixels | wrapped-no-duration |
|---|---|---|---|---|
| caption-single-human | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-single-human-instruction | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-single-robot | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-three-dex | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-three-tianji | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| caption-three-yam | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| golden-dex | PASS (12/12) | n/a | n/a | n/a |
| golden-human | PASS (12/12) | n/a | n/a | n/a |
| golden-single_view_robot | PASS (12/12) | n/a | n/a | n/a |
| golden-tianji | PASS (11/11) | n/a | n/a | n/a |
| golden-yam | PASS (12/12) | n/a | n/a | n/a |
| golden-yam32 | PASS (12/12) | n/a | n/a | n/a |
| temporal-dex-3view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-human | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-human-instruction | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-tianji-3view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-yam-1view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-yam-3view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-yam-3view-instruction | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |
| temporal-yam32-3view | PASS (4/4) | FAIL (0/4) | PASS (2/2) | PASS (2/2) |

How to read this:
- In every passing row, the pixels were equal and `generate` saw the rendered ids. In every row,
  vLLM's pixel tensors were bf16.
- `wrapped-naive-pixels` adds per-request `min_pixels=1024, max_pixels=589824`, a literal copy of
  NVIDIA's `AutoProcessor` kwargs. It did not change any grid here, but it buys nothing and should
  not be sent.
- The per-request table (239 rows: ids, grid, runs, pixels and gen marks, lengths, grids, errors)
  is in the results JSON and in the run's output. The previous revision of this page shows its
  format for the first 44 requests.

## The mismatch: a bare `<|video_pad|>` is not a vLLM placeholder

Every `bare` request fails in vLLM with
`AssertionError: Failed to apply prompt replacement for mm_items['video'][0]`, and no
`prompt_token_ids` are produced. The cause is in vLLM 0.24.0
`model_executor/models/qwen3_vl.py`:

- `Qwen3VLMultiModalProcessor._get_prompt_updates` registers the video `PromptReplacement`
  with the target `"<|vision_start|><|video_pad|><|vision_end|>"`.
  `_replace_video_token_placeholders` matches the same three-token sequence in token-id space.
- NVIDIA's builders put a bare `proc.video_token` into the user message, always right after a
  newline, and the chat template emits it verbatim. The tokenized prompt therefore has one lone
  `video_pad` (id 248057) per video and no `vision_start`/`vision_end` tokens. vLLM finds 0 of the
  N expected placeholders.

HF's `Qwen3VLProcessor.__call__` (transformers 5.8.1) accepts both shapes. HF's own comment there
says "vllm may input video token directly".

## The fix and its proof

`flora_deploy.nv.capture.wrap_bare_video_tokens(prompt, n_videos)`, used by
`vllm_prompt(request)`, replaces each bare `<|video_pad|>` with
`<|vision_start|><|video_pad|><|vision_end|>`. It refuses (`CaptureError`) unless all of these
hold:
- the prompt has exactly `n_videos` (>= 1) video tokens;
- the prompt contains none of `FORBIDDEN_PROMPT_TOKENS`: `<|vision_start|>`, `<|vision_end|>`,
  `<|placeholder|>`, `<|image_pad|>`, `<|vision_pad|>`;
- a newline precedes every video token.

All NVIDIA builders satisfy this. A user instruction containing one of the forbidden tokens is
refused (fail closed). The reason for `<|placeholder|>`: HF turns *every* `<|placeholder|>` in
the text into `video_pad` after expansion, while vLLM keeps it as text, so the two would
diverge.

**HF model inputs are unchanged**, so NVIDIA's reference is still the right reference:

1. *Argument.*
   - For each video in order, HF builds the same expansion
     `<{t:.1f} seconds><|vision_start|><|placeholder|>*n<|vision_end|>` per frame group. It
     replaces the first `vision_start+video_pad+vision_end` if the text has one, otherwise the
     first bare `video_pad`.
   - Expansions use `<|placeholder|>` and become `video_pad` only after the loop. With no
     `vision_start` in the input, no `vision_start` ever directly precedes a `video_pad` while the
     loop runs.
   - So the bare prompt takes the bare branch for every video and the wrapped prompt the wrapped
     branch, and both replace the k-th marker with the k-th expansion.
   - The texts, and hence the `input_ids`, are equal. Frames and metadata are untouched.
2. *Test.* `tests/nvlink/test_nv_link_capture_vllm_prompt.py` replays every case's captured
   request through NVIDIA's real processor with the wrapped prompt. `input_ids`,
   `video_grid_thw` and `pixel_values_videos` equal NVIDIA's original PIL-input call bit for bit
   (14 cases, 2 calls each).

**vLLM then matches HF.** This is the gate itself, run on Modal (table above).
- vLLM replaces each triplet, per temporal group, with `tokenize("<t seconds>")` +
  `vision_start` + pads + `vision_end`. The timestamp `t` comes from
  `_calculate_timestamps(frames_indices, fps, temporal_patch_size)`, the same formula HF uses.
- vLLM tokenizes the timestamp string on its own; HF tokenizes it in context. They agree because
  the character before each video is a newline, and between frames the previous token is
  `<|vision_end|>`.
- The mechanism is the pre-tokenizer's Split regex, which has no alternative that continues past
  `\n` into `<`. The vision tokens are also added specials without lstrip/rstrip.
- `tests/nvlink/test_nv_link_capture_vllm_prompt_tokenizer.py` locks this property for the
  pinned Flora `tokenizer.json`, which TempSeg and Captioner share. It checks `<t seconds>` for
  t = 0.0…5000.0 in 0.1 s steps plus large values, after 8 prefixes. It also asserts that the
  property fails after `"A:"`, which is why the fix requires a newline.
- The nvlink test also checks a pure model of vLLM's expansion
  (`validate.gatec.simulate_vllm_expansion`) against HF.

## `duration=None`

vLLM 0.24.0 accepts it:
- `Qwen3VLMultiModalProcessor._call_hf_processor` builds `VideoMetadata(**metadata)`, minus
  `do_sample_frames`.
- When `do_sample_frames` is False, the timestamps use only `frames_indices` and `fps`.
- Dropping the key (`wrapped-no-duration`) gives identical ids, grids and pixels.

## Incidental finding: three-view caption geometry depends on the platform's PNG encoder

For the same synthetic frames, three-view caption frame sizes differ between the macOS arm64 dev
machine and the Modal x86_64 container. Example, caption-three-dex synthetic: 22x20 locally,
28x24 on Modal.

The cause is NVIDIA's `robot_caption_geometry.normalized`. It picks the first scale whose packed
PNG byte size fits `PACKED_IMAGE_LIMIT_BYTES`, and compressed size depends on the zlib build in
the Pillow wheel.

Within each machine HF and vLLM agree, and stage 1 and stage 2 both run in the same image. This
confirms PLAN D11: golden data and production must share the image stack, not just the Pillow
version.

## Spend and harness

- Modal billing for app `flora-gatec`: $1.62 cumulative, which includes the first run's $0.86.
  This run cost about **$0.76** (cap $1.50). Billing lags by about an hour.
- Stage 1 (CPU, 2 vCPU and 12 GiB each), wall 556 s. The slowest part was captioner `golden:human`
  at 547 s: 1080p PNG work in `caption_single`.
- Stage 2 per L40S:

  | Kind | Load | Engine start | Compare | Total in function |
  |---|---|---|---|---|
  | tempseg | 11 s | 210 s | 115 s | 355 s |
  | captioner | 14 s | 212 s | 137 s | 382 s |

- The stage 1 pickles (which hold NC-licensed frames) go to `flora-golden:gatec-tmp/<run>/`, and
  stage 2 deletes them.
- `img_vllm` (`vllm_image()`) was used unchanged for both stages.

## Reproduce

```
uv run modal run scripts/gatec.py                 # [--quota 6] [--max-requests 2] [--no-golden]
uv run pytest -q -m nvlink tests/nvlink/test_nv_link_capture_vllm_prompt*.py   # no GPU
```
