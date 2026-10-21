# Accelerated NVIDIA: runtime patches (OPT-1)

Status: first patch set, 2026-10-21. Package: `src/flora_deploy/accel/`. Tests:
`tests/nvlink/test_accel_parity.py`, `tests/nvlink/test_accel_bench.py`,
`tests/unit/test_accel_registry.py`, `tests/unit/test_accel_decoder.py`.

`docs/reference-smoke.md` measured that model generation is only 9–31% of each reference stage.
The rest is CPU work inside NVIDIA's code. These patches take that CPU work out of the critical
path **without changing a single output**. This is form (1) of PLAN.md D17: runtime wraps in our
process, each proven output-identical to NVIDIA's unmodified code, which stays the oracle.

## How patches work

```python
from flora_deploy.accel import ALL_PATCHES, apply_patches
active = apply_patches(nv, ALL_PATCHES)           # or any subset, e.g. ["decode_once"]
with active.episode():                            # bounds caches + open decoders to one episode
    segments, audit = nv.pipeline.segment(videos, pool, session, instruction, progress)
    captions, cap_audit = nv.pipeline.caption_single(videos, segments, profile, session, ...)
record["accel"] = active.provenance()             # {"schema": "flora_accel_patches_v1", "patches": [...]}
active.restore()                                  # NVIDIA's attributes back, LIFO-checked
```

- A `Patch` declares four things: its family, the attribute paths it replaces, a factory, and the
  parity test that proves it. Attribute paths look like `batch_utils.pipeline.frames_for`,
  `batch_utils.pipeline.Predictor.predict` or `vendor.plans.plans_for_episode`.
  - The factory receives the *current* attribute, so patches compose in registry order.
  - `apply_patches` swaps module or class attributes of the already-imported NVIDIA modules. It
    does this in this process only.
- **NVIDIA's files are never touched.** `CODE_MANIFEST.json` and the bootstrap's vendor
  `MANIFEST.json` still verify while the patches are active. A test checks both.
- Every patch is opt-in and individually toggleable. Applying the same patch twice is refused.
  Restoring a set while a later set still owns one of its attributes is also refused, and a
  factory error rolls back everything.
- `AccelState` (`cache_bytes`, default 4 GiB per cache; `max_total_cache_bytes`, default
  `3 * cache_bytes`; `max_open_decoders`, default 64; `workers`, default `min(8, cpus)`; `seek`)
  holds:
  - per-episode **scopes** (below), each with its own byte-bounded LRUs, open decoders, source
    keys and lookahead counters;
  - three thread pools, shared: codec work, per-camera sources, and GOP decoders. They are
    separate, so no task ever waits on a pool it occupies.

  PIL and PyAV release the GIL in their codecs. Measured: 4 concurrent PyAV decodes cost the
  wall time of 1.

### Episodes and concurrency (W3-A)

`active.episode()` opens a scope bound to the **current context** (a `ContextVar`). W3-A runs
several episodes of a batch at once on threads that share one `ActivePatches`:

- Enter `active.episode()` **inside** the thread (or asyncio task) that runs the episode. Each
  overlapping episode then has its own scope, and that scope is closed when *its* episode exits
  (decoder files closed, caches dropped, keys forgotten), whatever the other episodes do. Before
  this, one shared depth counter kept everything open until the last episode ended (review M2:
  40 short episodes beside a long one held 40 decoders and their file descriptors).
- Work the patches submit to the shared pools runs in a copy of the submitting context, so it
  sees the same scope. NVIDIA's `asyncio.to_thread` predictor call copies the context too.
- A nested `episode()` in the same context reuses the scope. Calls outside any episode use a
  default scope that lives until `reset()` / `restore()`.
- Memory is bounded **across** scopes: every cache of every scope draws on one byte budget,
  `max_total_cache_bytes` (default: what one episode could hold before). Over budget, the largest
  cache gives up its oldest entry. Eviction only forces a recompute.
- Open decoders are bounded across scopes: above `max_open_decoders`, the least recently used
  *idle* decoder closes its file; it reopens from a verified random-access point if needed.
- `active.stats()` reports `active_episodes`, `open_decoders`, `cache_bytes_total` and their peaks.

`tests/unit/test_accel_decoder.py::test_overlapping_episodes_are_scoped_and_bounded` and
`tests/nvlink/test_accel_parity.py::test_overlapping_episode_threads_identical_and_bounded` (10
overlapping episode threads, the same source in two threads at once, 1-view and 3-view, with a
tight budget too) check that every scope is released at its own exit, that the bounds hold, and
that every tensor of every `generate()` call equals serial, unpatched NVIDIA's.

### Proving identity across runs: `accel.identity`

Whole-result digests cannot show that NVIDIA's model got identical inputs: later TempSeg windows
and all Captioner inputs depend on earlier model *outputs*, which vary run to run on the GPU
(`docs/exact-concurrency.md`). `accel.identity` compares two runs call by call from NVIDIA's own
log:

- `model_calls(events)` reads one episode's `FLORA_RUNTIME generation_start`
  (`input_tensor_digest`), `generation_end` and the GPU lock's `FLORA_OBSERVE generate_output`
  (output sha256), as raw lines or parsed dicts; `call_records` gives plain dicts to persist.
- `compare_calls(a, b)` pairs call *k* of both runs while every earlier output agreed. A paired
  call whose input digest differs, or a different call count with all outputs agreeing, is a
  **violation**. After the first output divergence (or an unlogged output), calls are counted
  as unpaired. It returns `ok` (no violation) and `conclusive` (everything was paired).
- `compare_runs({episode: events}, {episode: events})` does that per episode.

Gate the `exact` identity on `ok`, and report `conclusive`.

Every cache is either **content-addressed** or keyed by a source file. Content-addressed means
the key is a digest of the exact input bytes. A source-file key is only used after the file was
SHA-256-hashed once and matched to the probed `Video.sha256`. Correctness therefore never
depends on cache scope or eviction. The tests run with a 100 kB cache to prove it.

## The patches

### `decode_once`: each source frame decoded about once per pass

**Targets**

- `media.frames_for` and `pipeline.frames_for`. `pipeline.py` binds the name at import.
- `pipeline.Predictor.predict`, `pipeline.caption_three` and `plans.plans_for_episode`, which
  are wrapped for lookahead only.

**Problem.** NVIDIA's `decode_selected` (`media.py:69-95`) opens the file and decodes frames
`0..max(wanted)` on every call, once per camera. `Predictor.predict` calls `frames_for` once per
TempSeg window, and `caption_three` calls it once per caption plan. So decode work grows as
O(calls × prefix).

**Replacement.**

1. **Lookahead.** NVIDIA builds a TempSeg request with *all* the round's windows before it
   decodes a frame (`fine_subtask.py:254`). `caption_three` builds *all* its plans before its
   first `frames_for` (`pipeline.py:119-125`).
   - The `predict` wrapper computes every window's frame plan with NVIDIA's own pure
     `window_from_payload` and `temporal_frame_plan`, then calls NVIDIA's `predict` unchanged.
   - The `plans_for_episode` wrapper records the plans it returns while it runs inside
     `caption_three`.
   - The first `frames_for` of the round then decodes the **union** in one pass, within half the
     cache budget.
   - The cache tracks how many announced uses each frame has left. A frame with none left is
     moved to the front of the eviction queue, so the cache holds only frames still needed
     later, plus whatever spare room allows.
2. **Persistent cursor.** One decoder per file is opened exactly as NVIDIA opens it
   (`err_detect=explode`, `thread_count=1`, `container.decode`). A request at or after the
   cursor continues decoding instead of restarting.
3. **Random-access restarts and GOP-parallel decode.** These are used only when the file's
   demuxed packets map one-to-one onto the probed `pts_us`: no edit-list discards, the same
   count, and equal relative timestamps. That mapping is the `SeekMap`.
   - A request behind the cursor restarts at the last bitstream-verified random-access point
     (`media.index` `FLAG_RAP`: H.264 IDR, HEVC IRAP, AV1 shown key frame) at or before every
     requested frame.
   - A large cold request is split into runs. Each run is decoded by its own decoder from its own
     random-access point, concurrently.
   - Every output frame is matched to its index by its exact pts.
   - Several cases **fall back to NVIDIA's decode-from-0 path**:
     - an unknown pts;
     - a skipped requested frame;
     - a demux or decode error;
     - a seek overshoot;
     - a stream index change.

     Leading pictures are excluded because they present before the restart point.
4. **Byte-bounded LRU.** It holds converted frames for one view and composed mosaics for three
   views.

**Why it is identical.**

- Per frame, the code makes NVIDIA's timestamp check with NVIDIA's message.
- The colour branch is NVIDIA's `media.py:85` (NVIDIA's own `canonical_rgb` versus
  `to_image().convert('RGB')`), decided per decoded frame.
- Mosaics use NVIDIA's own composers.
- The dict keys and order match: ascending for one view (decode order), `wanted` iteration order
  for three views.
- Errors match NVIDIA's `InputError` messages: outside the video, unsupported layout, timestamps
  changed, could not be decoded.
- Callers get fresh copies.
- An empty request, missing roles (NVIDIA's `KeyError`) and any source whose SHA-256 differs from
  `Video.sha256` are delegated to NVIDIA's original `frames_for` untouched.

**Evidence.** `test_decode_once_frames_identical` (21 cases) runs seven sources under three
configurations.

- Sources:
  - H.264 with B-frames (1 and 3 views);
  - odd widths 72/80, which take the non-canonical branch;
  - BT.709 and full-range tags;
  - VFR;
  - SVT-AV1 (1 and 3 views).
- Configurations: default; serial with no seeking; a 300 kB cache with 4 workers.
- Requests:
  - forward, backward, overlapping and edge requests;
  - a list with duplicates;
  - a full-range request and an empty one;
  - out-of-range requests;
  - every mosaic, including an unsupported one.
- Every returned image is compared on type, mode, size, pixel SHA, `info` and `readonly`.

The test also asserts that random-access restarts happened and that none of them needed the
fallback.

Further tests:

- `test_decode_once_gop_parallel_identical` covers the parallel split.
- The untrusted-file, changed-timestamp, missing-role and monotone-windows tests show one decoder
  pass from frame 0.
- `tests/unit/test_accel_decoder.py` checks the decoder against a decode-from-0 reference on
  H.264 B-frames, VFR and x264 intra-refresh. Intra-refresh correctly gets no random-access
  restarts.

### `transport_pool`: the `human_transport` JPEG round trip

**Target.** `pipeline.human_transport`. NVIDIA's version (`preprocessing.py:14-21`) runs a
`jpeg_q98_444` encode/decode per frame per window, serially.

**Replacement.**

- It calls NVIDIA's own `encode_regularized_image` and the same decode for each frame, with the
  frames of one call running concurrently.
- Results are memoized by a content key of the input (mode, size, pixel digest, palette, `info`),
  because JPEG encode/decode is deterministic for identical input. Frames shared by overlapping
  windows are encoded once.
- Output order equals input order.
- Errors are raised in input order. A non-image goes straight to NVIDIA's code, so its error is
  NVIDIA's.

**Evidence.** `test_transport_pool_identical` (serial and threaded, duplicate content, mode `L`,
a bad input) and the end-to-end human cases.

### `png_inmem`: `caption_single`'s PNG round trip

**Targets**

- `pipeline.caption_single`, wrapped only to scope the prefetch;
- `pipeline.frames_for`, wrapped on top of `decode_once` when both are active;
- `unified_input.load_frames`.

**Problem.** NVIDIA's version (`pipeline.py:94-101`) works in two steps:

- `caption_single` saves each caption frame with `image.save(path, format='PNG')` (zlib level 6)
  and hashes the file;
- `load_frames` (`unified_input.py:56-83`) then re-reads, SHA-checks and PNG-decodes the frame
  once **per block it appears in**.

The level-6 encode was the largest CPU item of the caption stage: 18 s of 34 s on a 720p clip.

**Replacement.**

1. **Prefetch.** Frames returned *inside* `caption_single` are tagged. A worker encodes each one
   with `Image.save(buffer, format='PNG')`, the identical encoder call and parameters. When
   NVIDIA's loop calls `save(path, format='PNG')`, the prefetched bytes are written to the path.
   Any other `save` call goes to Pillow unchanged.
   - The bytes are identical by construction: the PNG writer emits its chunks through the same
     path for a file and for a buffer. The test checks this.
   - This is the WS-E concern: compressed size depends on the Pillow and zlib build. It still
     comes from the very same build and call.
2. **Decode once.** The worker also decodes those bytes (`Image.open` + `load` +
   `convert('RGB')`, as NVIDIA does) and precomputes `load_frames`' single-view downscale. Both
   are stored under the PNG's SHA-256.
3. **`load_frames`.** This is a re-expression with NVIDIA's validation unchanged:
   - `validate(sample)`;
   - the root-ownership check;
   - **reading the file and comparing its SHA-256 to the receipt**;
   - the size check;
   - the same `resize` call.

   Only the decode and resize come from the SHA-keyed, content-addressed cache.

The files are still written and read, because NVIDIA hashes what is on disk. What disappears is
the serial encode and the repeated decode.

**Evidence.**

- `test_png_prefetched_bytes_identical` checks that the bytes equal `im.save(path)` and that
  non-default parameters pass through.
- `test_png_inmem_receipts_identical` checks that the receipts NVIDIA builds (`png_sha256`, size,
  path; spied at `caption_sample`), the captions and audits, and every processor tensor are
  equal, for H.264 and VFR, with and without an instruction. The instruction branch calls
  `load_frames` directly.
- `test_png_inmem_load_frames_errors_identical` covers:
  - OK cases with 1-view and 3-view canvas, with and without `preserve_canvas`;
  - each error: `PNG changed`, `Frame dimensions changed`, `Unowned frame path`, an invalid plan.

### `geometry_pool`: `caption_three`'s caption geometry

**Targets.** `robot_caption_geometry.normalized` and `source_frames`.

**Problem.** NVIDIA's `normalized` (`robot_caption_geometry.py:14-46`) packs pairs of mosaics
(LANCZOS-resized below scale 1) and PNG-encodes them at level 1. It sums the sizes and retries
smaller scales while the total exceeds 21,866,496 B. `source_frames` decodes every PNG back.
All of it is serial, and it was 22 of 26 s of a patched 3-view caption stage before this patch.

**Replacement.**

- Re-expressions run the per-pair work in the pool.
- They call **NVIDIA's own `png()` and `sha()`** and read `SCALES` and the limit from NVIDIA's
  module, so the bytes whose sizes pick the scale come from the identical call.
- Dicts are assembled in NVIDIA's order with NVIDIA's keys. Errors (missing frame, no fit,
  `expected` assertions, crop SHA assertion) are raised in NVIDIA's order.
- A factor whose running total already exceeds the limit stops early. NVIDIA discards that
  factor's assets, so the rest are never observable.

**Evidence.** `test_geometry_pool_identical` (serial and threaded) uses four limits, which force
scale 1.0, retries down the scale ladder, and no fit. It also covers the `expected` re-check path,
a missing frame and a tampered crop SHA. Whole result dicts are compared, including the PNG
blobs.

## End-to-end evidence

`test_pipeline_end_to_end_identical` runs NVIDIA's `pipeline.segment` and then `caption_single`
or `caption_three`, with all patches on versus none. It uses the real Flora processors and a
canned `generate`. That `generate` records every tensor NVIDIA's `generate` hashes
(`mm_token_type_ids` included) with NVIDIA's `input_tensor_digest` formula. Its TempSeg answers
are valid, non-empty predictions: fixed events clipped to each window, so segments, multi-window
stitching and the caption plans that follow are exercised.

Cases (9):

- `human` on H.264;
- `human` on VFR with an instruction;
- `human` on a 30 s clip (several windows, stitched);
- `human` on HEVC open GOP (CRA + RASL);
- 1-view `yam` on AV1;
- 3-view `yam` on H.264;
- 3-view `dex` on AV1 with an instruction;
- 3-view `yam` with odd sizes and BT.709/full-range tags;
- 3-view `yam` on HEVC open GOP.

Results:

- Segments, per-window audits (frame plans, prompt hashes), captions and caption audits (blocks,
  geometry, prompt hashes) are equal.
- Every processor tensor is bit-identical.
- `..._tiny_cache` repeats two cases with a 100 kB cache.

Real staged samples (HomER, ALOHA) live only on the `flora-golden` Volume, so these runs use
generated fixtures only. Running the same suite against the samples is a Modal-side follow-up.

## Measured speedup

`tests/nvlink/test_accel_bench.py` runs one episode through NVIDIA's pipeline, unpatched versus
`ALL_PATCHES`. It uses the real processors and no model, so the time is NVIDIA's CPU work. Each
run asserts identical outputs, fewer decoded frames and a shorter total.

Setup:

- Apple M-series laptop, 8 cores. **Shared with other jobs**: load average 7–17 during the runs,
  so wall times are noisy and conservative.
- "Decoder passes" means calls to NVIDIA's `decode_selected` (each starts at frame 0). For the
  patched run, it means decoder opens: restarts and GOP runs.

| Episode | | Segment s | Caption s | Total s | Decoder passes | Frames decoded |
|---|---|---|---|---|---|---|
| 1 view 640×360, 45 s, 30 fps (`human`) | NVIDIA | 6.71 | 16.11 | 22.82 | 5 | 5,011 |
| | patched | 1.93 | 6.47 | 8.41 | 16 | 3,084 |
| | speedup | **3.5×** | **2.5×** | **2.7×** | | −38% |
| 3 views 320×240, 20 s, 30 fps (`yam`) | NVIDIA | 4.39 | 32.69 | 37.09 | 24 | 9,960 |
| | patched | 1.59 | 8.54 | 10.13 | 27 | 4,092 |
| | speedup | **2.8×** | **3.8×** | **3.7×** | | −59% |

Earlier exploratory runs, same harness, 720p:

- 1 view, 30 s:
  - segment 14.4 → 5.4 s;
  - caption 34–40 → 10.7–14 s.
- 3 views 640×480, 22 s:
  - segment 9.0 → 2.9 s;
  - caption 45 → 9.5 s.

The decode saving grows with episode length and window count, because NVIDIA's cost is
O(windows × prefix).

On 1080p HomER clips (9 windows, about 300 caption frames), the per-frame costs that dominate
are the ones removed or parallelized here. Those costs are full-resolution decode, the
`human_transport` JPEG, the level-6 PNG, and the repeated PNG decode.

## What is not (yet) identical-by-construction, and limits

- **Corrupt bitstreams before a restart point.** A random-access restart never decodes the
  packets before it. NVIDIA would raise there.
  - This cannot arise for a probed file: NVIDIA's `probe` decodes the whole file with
    `err_detect=explode`, and the patch only runs on a file whose SHA-256 equals the probe's.
  - A file modified between probe and decode is delegated to NVIDIA's code.
- **Returned objects.** They are equal-valued fresh `PIL.Image.Image` copies, not the same
  objects. Inside `caption_single`, the frames' class is the `PrefetchedPNG` subclass, whose
  only difference is `save(path, format='PNG')`. NVIDIA only reads `.size` and calls that
  `save`.
- **One redundant pass.** Frames decoded for segmentation are not the caption frames. The caption
  stage runs one more decode pass, as NVIDIA's single `frames_for` in `caption_single` does.
  Retaining raw decoded frames across stages would cost about 3 MB per 1080p frame and was left
  out.
- **Not patched.** The HF processor, `temporal_inputs`' BICUBIC resize and the prompt building
  run unchanged. They are inside NVIDIA's functions, and the processor is WS-B's
  `CaptureProcessor` path. Decoder threading (`thread_count > 1`) is also not used: FFmpeg
  documents threaded decoding as bit-exact, but it changes NVIDIA's decoder configuration.
  GOP-parallel decode from verified random-access points gives the parallelism instead.
