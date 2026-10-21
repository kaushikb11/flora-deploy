# U-1: suggestions for NVIDIA's Flora Hugging Face repos (drafts)

**Summary.**
1. NVIDIA's prompts do not run on vLLM 0.24.0 as written, and wrapping each video token fixes this without changing HF inputs.
2. Most time in the reference pipeline goes to CPU work (PNG round trips, decoding from frame 0) and to loading FP32 Captioner weights. Each has a fix whose outputs we proved identical.
3. The rest are documentation items: an undocumented batching hook, GPU nondeterminism, container-level frame rates, the install notes, and places where a model card disagrees with the code.

> **Drafts only.** The flora-deploy maintainer should review these before posting any of them
> as HF discussions. Nothing here has been posted. File:line references are to the pinned code:
> TempSeg/Captioner `batch_utils/` at `78a0e4ac` (the two copies are identical) and Rewarder
> `batch_utils/` at `7de68d22`. Evidence is in the linked flora-deploy docs. Items are ordered by
> impact.

---

### 1. Wrap video tokens so prompts work on vLLM (TempSeg, Captioner)

- **Observed:** the builders insert a bare `proc.video_token` (`preprocessing.py:38`, `:50`;
  `vendor/flora_unified/unified_input.py:50`; `pipeline.py:100`). HF's Qwen3-VL processor
  accepts this. vLLM 0.24.0's Qwen3-VL processor only matches the sequence
  `<|vision_start|><|video_pad|><|vision_end|>`.
- **Impact:** vLLM rejects **0/44** requests, across 11 builder cases (`Failed to apply prompt
  replacement`). With the tokens wrapped, **44/44** match HF exactly: `input_ids`,
  `video_grid_thw`, and bf16 `pixel_values_videos`. (`docs/gate-c.md`)
- **Suggestion:** emit `<|vision_start|><|video_pad|><|vision_end|>` in the builders. HF's
  `input_ids` stay byte-identical: HF expands both forms to the same string, and we tested this
  on all 11 cases.
- **Our workaround:** yes. `nv.capture.wrap_bare_video_tokens` rewrites the captured prompt.

### 2. Skip the PNG write-and-reread in `caption_single` (TempSeg, Captioner)

- **Observed:** every caption frame is saved as a level-6 PNG and hashed
  (`pipeline.py:85-93`). `load_frames` then re-reads it, re-hashes it, and decodes it once for
  each block the frame appears in (`unified_input.py:56-83`).
- **Impact:** over 4 sequential episodes on an H200, PNG save is **43% of wall time** (573 s of
  1,322 s) and `load_frames` is another 6%. For comparison, `generate` is 22%.
  (`docs/exact-concurrency.md`)
- **Suggestion:** keep the frames in memory, keyed by a content hash. If on-disk receipts must
  stay, encode in parallel and decode each frame once. PNG is lossless, so pixels are unchanged.
- **Our workaround:** yes. The `png_inmem` patch produces the same bytes, receipts, and
  processor tensors. Caption stage is up to **3.8× faster** on CPU. (`docs/accel.md`)

### 3. Decode each frame once per pass (TempSeg, Captioner)

- **Observed:** `decode_selected` opens the file and decodes from frame 0 up to `max(wanted)` on
  every call, with `thread_count=1` (`media.py:69-95`). It is called once per TempSeg window
  (`pipeline.py:30`) and once per three-view caption plan (`pipeline.py:133`), so cost grows as
  windows × prefix.
- **Impact:** decoding is **18.6% of wall time**. On 1080p clips, generation is only 9–31% of
  each stage. (`docs/exact-concurrency.md`, `docs/reference-smoke.md`)
- **Suggestion:** both call sites already know every frame they need before decoding
  (`fine_subtask.py:254`, `pipeline.py:119-125`). Decode the union of those frames in one
  forward pass, or keep one decoder open and continue from where it stopped.
- **Our workaround:** yes. The `decode_once` patch decodes 38–59% fewer frames with
  bit-identical outputs. End to end, the CPU pipeline is **2.7–3.7× faster**. (`docs/accel.md`)

### 4. Publish a BF16 Captioner checkpoint (Captioner)

- **Observed:** the checkpoint is FP32: 109.4 GB in 11 shards, with `config.json` dtype
  `float32`. `Session.load` builds the model in BF16 anyway (`models.py:139`) and keeps only
  `A_log`/`dt_bias` in FP32 (`production_gdn_fix`).
- **Impact:** we pre-cast the weights the same way the loader does, keeping the 96 FP32 GDN
  tensors. The result is 54.7 GB, and the loaded parameters, buffers, and logits are
  **bit-identical** to NVIDIA's path (`verify_gdn` passes). Captioner start time drops from
  **≈305 s to ≈56 s**. (`docs/captioner-bf16.md`)
- **Suggestion:** publish the mixed BF16/FP32 shards, as a branch or a sibling repo, with their
  own `SOURCE_MANIFEST.sha256`.
- **Our workaround:** yes. We convert once and keep a receipt; `captioner_load.install` loads the
  converted shards.

### 5. Reduce per-process start cost: manifest hashing and read-only caches (TempSeg, Captioner)

- **Observed:**
  - `validate_manifest` re-hashes every file the first time a process uses it
    (`models.py:55-98`). The hash is single-threaded and held under `_MANIFEST_LOCK`.
  - `snapshot()` calls `snapshot_download(..., local_dir=...)` (`models.py:46`), which takes a
    filelock inside `local_dir` even when every file is already present. A read-only mount
    therefore fails with EROFS.
- **Impact:** hashing the FP32 Captioner takes **92–145 s on every container start**. Until we
  built a symlink layout, the read-only Volume also forced a 109 GB local copy (123–172 s).
  (`docs/cold-start.md`, `docs/reference-smoke.md`)
- **Suggestion:**
  - Hash the shards in parallel.
  - Allow an opt-in to trust an already-verified, read-only snapshot.
  - Return early from `snapshot()` when the pinned files are already present.
- **Our workaround:** partly. A link farm passes NVIDIA's unmodified checks with no copy, and we
  overlap the hash with the TempSeg load. Skipping the hash entirely would need a code change.

### 6. Document `apredict` and the shared-processor caveat for batching (TempSeg, Rewarder)

- **Observed:**
  - The segmenter prefers `predictor.apredict` if it exists (`fine_subtask.py:255-259`). This
    is a clean batching hook, but it is not documented.
  - The README says JSONL "is not GPU batch scheduling".
  - Rewarder dense makes one call per episode (`dense_reward.py:95-115`).
- **Impact:**
  - Run sequentially, the GPU is busy only **22–41%** of the time.
  - Four episodes run concurrently through batch-1 `generate` behind a lock are **2.11×**
    faster.
  - Batching dense Rewarder clips across episodes gave identical outputs and was **1.4×**
    faster.
  - One caveat: the HF tokenizer changes its padding state on each call, so processors must not
    be shared between threads.
  - (`docs/exact-concurrency.md`, `docs/rewarder-smoke.md`)
- **Suggestion:** document `apredict` as a supported extension point and note the
  processor-sharing caveat.
- **Our workaround:** yes. We batch through `apredict`, use per-episode processor clones, and
  use cross-episode Rewarder batches.

### 7. Note that greedy HF `generate` is not reproducible run to run (TempSeg, Captioner)

- **Observed:** batch-1 greedy `model.generate` (`models.py:179`) gave different outputs for
  byte-identical inputs, verified with NVIDIA's own `input_tensor_digest`. Two sequential runs
  on H200 disagreed on **4 of 4** episodes. Some outputs differed in length: 9 vs 18 tokens and
  83 vs 43 tokens. (`docs/exact-concurrency.md`)
- **Suggestion:** state the expected variance, and the determinism settings used to validate
  the release, so downstream users can set tolerances instead of expecting exact matches.
- **Our workaround:** n/a. Our gates use measured variance instead of zero.

### 8. Document per-episode frame rate and cutting for concatenated datasets (TempSeg, Captioner)

- **Observed:** `probe` reads the container's `stream.average_rate` (`media.py:31`). That rate
  drives the CFR test (`:55`) and the three-view sync check (`:63-67`).
  - For LeRobot v3 files that hold many episodes, the whole-file rate differs from the
    per-episode rate. Fixtures: 325/14 for the whole file vs 30/15/25 per episode.
  - A boundary stts (frame-duration entry) can make an episode cut report 43520/1449 instead of
    30, and the sync check then rejects it.
  - Edit-list trims are hidden from decoding but disappear in a naive remux.
  - (`docs/media-notes.md`)
- **Suggestion:** say in the README that inputs should be per-episode, stream-copy cuts that
  start on a keyframe and carry no edit lists.
- **Our workaround:** yes. We compute the per-episode rate with FFmpeg's formula and reject ranges
  that cannot be cut exactly.

### 9. Refresh install and access notes (TempSeg, Captioner)

- **Observed:**
  - `batch_utils/README.md:20` installs torch from the cu128 index, but `requirements.txt` pins
    a plain `torch==2.11.0`. Running only `pip install -r requirements.txt` installs PyPI's
    default build, which is cu130.
  - `batch_utils/README.md:25-26` still describes the repos as private and requiring `HF_TOKEN`.
  - `ready()` (`models.py:27-37`) also mentions "private weights", but nothing in `batch_utils`
    calls it.
- **Suggestion:**
  - Add the index URL to `requirements.txt`, or a comment there.
  - Drop the private-repo wording.
  - Remove `ready()`, or wire it into the pipeline.
- **Our workaround:** yes. We pin images explicitly.

### 10. Align the model cards with the shipped code (Rewarder, Captioner)

- **Rewarder:** the card says rubric prompts use "1 FPS, at most 82 frames per camera"
  (`README.md:75`). The code defaults to `RUBRIC_FPS = 2.0` and `RUBRIC_MAX_FRAMES = 166`
  (`core.py:665-666`, also used by `rubric_reward.py:57-58`).
- **Captioner:** the card says BEFORE/AFTER are the "nearest available non-overlapping event
  intervals" (`README.md:90`). The shipped code uses a fixed window of up to 2 s on each side:
  `vendor/pts_input_adapter.py:18` for human and `vendor/support/plans.py:17` for three-view,
  with `context_policy='common_real_up_to_2s_each_side'`.
- **Suggestion:** say which behaviour is intended and update either the card or the defaults.
- **Our workaround:** we follow the code, because it is our reference.
