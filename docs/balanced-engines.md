# vLLM engines and the `balanced` segment / caption stages

Status: 2026-10-22. CPU tests, conformance and nvlink tests are green. A first GPU pass ran on
2026-10-22 (see Results): TempSeg's in-engine checks passed, the Captioner's are still to do, and
no `balanced` certificate exists yet.

Terms used below (maintainer doc):

- **oracle**: the `exact` preset's run, NVIDIA's own code on the reference GPU, which `balanced`
  is measured against.
- **Gate C**: the capture check that vLLM's model inputs match NVIDIA's HF processor
  ([`gate-c.md`](gate-c.md)).
- **L0**: the in-engine check that vLLM's prompt token ids and video grid equal NVIDIA's HF
  processor replay on captured requests.
- **Gate A**: the `dt_bias` FP32 drift check: logit differences between the stock bf16 engine and
  the FP32 `dt_bias` plugin engine (recorded only; it does not affect certification).
- **the 2026-10-22 GPU run** ("Wave H-lite" in `scripts/wave_h.py` and the Results block): the
  first measured `balanced` session ([`wave-h-runbook.md`](wave-h-runbook.md)).

`balanced` deploys and runs through `flora deploy` / `flora run` (see "Deployment" below).
Without a certificate it needs `allow_uncertified: true`. Every `segment` / `caption` record is
`certified: false` with `uncertified_reason: "uncertified_stack"` until a certificate
([`certification.md`](certification.md)) exists. `balanced`'s reward stages are not affected:
they run exactly `exact`'s engine, so they are the reference by identity and are certified
without a certificate.

## Deployment (`modal_app/factory.py`, `modal_app/balanced_app.py`)

- **Worker classes.** `segment` -> `SegmentWorker` (pool default `L40S`), `caption` ->
  `CaptionWorker` (`H100!`). Both run on `img_vllm`, with `max_inputs` 1 (episodes run
  concurrently inside the shard), `WORKER_RETRIES`, the rate-table timeouts, `scaledown_window`,
  `flora-weights` and `flora-cache` mounted read-only, and the store and media mounts. Their env
  adds `FLORA_DERIVED_RECEIPTS` (the `weights.derived` receipts of the kind),
  `FLORA_POOL_GPUS`, `FLORA_KIND_MODES` and `FLORA_EMBODIMENT`.
- **`@modal.enter`**, failing closed at each step. Step 0 is `check_pinned_kind`: a derived
  weight with no pinned content digest makes the container serve a `RefusedRunner`, which ends
  every shard `deployment_refused` at once, with no retry. The keyed compile cache is prepared
  before NVIDIA's code is loaded. Then the preflight (store, media root,
  instruction); then the derived weights. The receipt file on the Volume must hash to the
  recorded `weights.derived` identity, and its content digest (recomputed) must equal
  `pins.DERIVED_PINS` (`None` refuses; both `captioner-bf16` and `tempseg-merged` v2 are
  pinned, `config/pins.py`). Every byte is then hashed against the receipt, on a thread joined before
  ready. Next the Captioner's `model.visual` `_extra_state` check, then the certification stamp
  (`workers_impl/certstamp.py`: `embodiment`, `cert_key` from `validate.certkey.key_document`
  on the pool GPU with the verified content digests, `gpu`). Then NVIDIA's code, the processor,
  and `OfflineEngine(spec_for(kind, model, pool GPU))`, which runs the dt_bias plugin flag check
  and the GDN dtype verify. Last come the accel patches and the runner, with
  `fatal=workers_app._die` (CUDA fault or dead engine: mark, then exit the container).
- **Segment -> caption, per episode.** Caption shards `depends_on` the segment shards that hold
  their episodes (manifest). The orchestrator (`jobs.orchestrator.deps_ready`; the reconciler
  and `progress.derive` agree) spawns a caption shard once each of *its* episodes is resolved
  upstream. A segment shard is resolved when it is satisfied. A quarantined, bisected one is
  resolved when the child holding that episode is satisfied, or when the episode was done before
  the bisection. In the worker, `attempts.upstream_excluded` gives the episodes the segment
  stage excluded: ones set aside for crashing their worker every time (`poison_pill`), and ones in a segment receipt's `episodes_failed`
  (`nv_failure`). They are written as caption `EpisodeError`s with that kind and
  `diagnostics.upstream: true`, and never reach the model. The error-rate guard counts them
  once, at `segment` (`progress.UPSTREAM_KINDS`, `tally_exclusions`, compose's
  `excused_upstream`). Resume, preemption and bisection use the existing machinery for both
  pools. The budget projection prices both pools (rate table rows `balanced segment/caption`,
  with a startup per pool).
- **Scale to zero and pre-warm** (`jobs/pools.py`). When a pool has no work left (segment while
  caption still runs, and every pool at job end), the orchestrator drops that class's
  `scaledown_window` to 2 s within one 15 s poll, and restores it before new work arrives. Caption
  is never drained while segment still produces. The orchestrator also raises caption's
  `min_containers` to 1 at the estimated `segment ready - caption startup`, so an H100 boots
  while segment finishes (`pool_prewarm_*` events; the budget charges the pre-warmed container
  from its start). The pre-warm is skipped when `min_containers > 0`.
- **GPU overrides.** Each pool needs a minimum of GPU memory (segment 44 GiB, caption 79 GiB,
  `config/presets.py`) and a `GPU_LIMITS` row for its GPU (below): TempSeg runs on L40S,
  A100-80GB, H100 or H200; the Captioner on H100 or H200. `flora deploy` refuses any other.
- **Fake models.** `flora deploy/run --fake-models` registers `FakeSegmentWorker` /
  `FakeCaptionWorker` (CPU), and `flora run --local --fake-models` runs `preset: balanced` too.
  Both are the real `BalancedShardRunner` and micro-batcher, with `local/balanced.py`'s fake jobs
  at the model boundary. Fault spec `poison: {"<kind>": [uid]}` poisons an episode in one kind
  only.

## Design

```
episode coroutines (one event loop; ≤ 8 preparing, ≤ 64 segment / 32 caption inferring)
  NVIDIA FineSubtaskSegmenter ──await apredict(request)──► MicroBatchPredictor (subclass of NVIDIA's Predictor)
                                                             │ to_thread: NVIDIA's own Predictor.predict
                                                             │   (window, frame plan, frames_for, human_transport,
                                                             │    temporal_inputs → CaptureProcessor)
                                                             │   pipeline.generate is tapped → CapturedRequest + sentinel
                                                             ▼
  NVIDIA caption_three / caption_single ── same tap ──► CapturedRequests (capture wrap: vllm_prompt)
                                                             ▼
                                         MicroBatcher.generate_many  (round-based)
                                                             ▼  one llm.generate per round, worker thread
                                         OfflineEngine (vLLM 0.24.0 offline LLM, explicit args)
                                                             ▼
             accept in NVIDIA's call order: 32k guard → EOS → non-empty decode; sentinels → texts
```

- **NVIDIA does the work, unchanged.** The segment stage runs NVIDIA's own
  `pipeline.Predictor.predict`. The caption stage runs NVIDIA's own `caption_three` /
  `caption_single`. We replace only `pipeline.generate`, with a tap
  (`stages.segment.GenerateTap`, reversible and reference-counted). Inside a collector context the
  tap pops the `CapturedRequest` from the `CaptureProcessor` that NVIDIA's builders called, and
  returns a unique sentinel. Outside a collector it calls NVIDIA's original. NVIDIA's code
  therefore builds every window, plan, frame and prompt. All of an `apredict` request's windows (or
  all of an episode's caption calls) are then sent to the batcher together, and the sentinels in
  NVIDIA's response and in its audit (`Predictor.calls`, caption inputs) are replaced with the
  decoded texts.
- **The segmenter construction** comes from `nv.segment_glue`: the inline 3-view construction or
  `adapter.production_segmenter`, already proven equal to `pipeline.segment`. NVIDIA's segmenter
  then makes 1–3 `apredict` rounds (grid, recentered retry, directional retry) and runs its own
  repair and conservative completion.
- **Failure semantics.** Completions are accepted with NVIDIA's `models.generate` rules, **in
  NVIDIA's call order**, and the first failure raises:
  1. The 32k guard is applied to vLLM's `prompt_tokens`. A vLLM `max_model_len` rejection is
     mapped to the same `ContextBudgetError`, which is also NVIDIA's `InputErrorForModel`.
  2. If EOS was not reached, or the text is empty, the stage raises NVIDIA's own `InputErrorForModel`.

  The episode fails exactly as it would in `pipeline.segment` / `caption_*`. The batched path has
  already generated the request's other windows, which changes nothing. Engine failures raise
  `EngineRequestError` / `EngineBatchError`, which the worker classifies as `infra`.
- **Inputs.** NVIDIA's `api.DataEnginePipeline._inputs` does `media.probe` per camera plus the
  size, profile, wrist and sync checks. The pool comes from `pipeline.annotate`'s one-liner,
  restated in `pool_for`. The caption stage reproduces `pipeline.annotate`'s provenance copy and
  `api.annotate`'s interval check.
- **Micro-batcher** (`engines/microbatch.py`) and **batch policy** (`engines/batch_policy.py`).
  An episode prepares its inputs first (input resolve and NVIDIA's probe; at most
  `prefetch_episodes` = 8 preparing or waiting), then enters `async with mb.episode()` for
  inference only (at most `infer_episodes`: segment 64, caption 32), so a preparing episode never
  holds a round back. A round fires when one of these holds:
  - every live episode is blocked on the batcher (it has calls pending or in flight, or is waiting
    for admission);
  - `max_batch` requests or `max_batch_bytes` of frames are pending;
  - the oldest call has waited `max_wait_s` (a safety net).

  Only one `generate` runs at a time, in a dedicated thread. Calls that arrive while it runs form
  the next round, so CPU preparation overlaps generation. Batches are FIFO, so their composition is
  deterministic for a given submission order. Memory is bounded: admission waits above
  `max_pending_bytes`, except that a single oversized call is still admitted when nothing is
  pending. Failures are isolated: a batch that raises fails only its own calls, and per-request
  errors come back in `Completion.error`. On `KeyboardInterrupt` (SIGINT) the batcher fails its
  calls and the exception propagates, so the worker's preemption path runs.

  The batch policy's bounds are `max_batch` 128, `max_wait_s` 0.5, `max_batch_bytes` 3 GiB and
  `max_pending_bytes` 6 GiB. They are code, not a `flora.yaml` knob: batch grouping changes
  numerics, so `batch_policy.py` is in the `vllm-video` logic digest and a change re-keys the
  stage, its certificate and its compile cache. The effective policy is recorded in the shard
  audit row.
- **Batch bisection** (`OfflineEngine.generate_batch`). A batch that raises is split in halves
  and retried, recursively, so one bad request costs O(log n) generates and its batch-mates stay
  batched. A dead engine (`EngineDead`: vLLM's `EngineDeadError` or a CUDA fault) is never
  retried; the worker fails fast and exits.
- **Workers** (`workers_impl/segment.py`, `workers_impl/caption.py`). `BalancedShardRunner`
  follows the full `ShardWorker` contract by mirroring `RewardShardRunner`, and reuses its
  `ShardPartSink`, `parse_index_receipt` and `VideoResolver`. Episodes run concurrently on one
  event loop and their records are flushed every `flush_every` episodes. Index receipts give
  NVIDIA's arguments (profile, yam32, instruction source). Caption shards read their episodes'
  `segment` records from the upstream shard's parts, including bisection children, filtered to the
  segment stage key (`JobFiles.accepts("segment")`). The full NVIDIA segment metadata travels in `Segment.nv_metadata`, and
  `rows_from_record` turns it back into NVIDIA's rows. Both workers are registered in
  `tests/conformance/registry.py` and pass all 10 conformance checks.
- **Provenance.** The engine label is `vllm-0.24.0-offline` or `…+dtbias`. Records carry the
  model pins, the GPU and the per-camera source. The job's segment / caption **stage hashes**
  (`config/hashing.py`, which key the shard files) cover the worker's full `EngineSpec.hashed()`
  on the configured GPU (`engine.spec`: every engine kwarg incl. the per-GPU scheduler limits and
  `dtbias_fp32`; the GPU name stays out, as `docs/PLAN.md`'s hashing rule says), so a GPU override with other limits gets a new
  key instead of resuming into old shards. The stage's own hash (engine spec, NVIDIA's sampling,
  model pins, the logic version and the `vllm-video` logic digest) is kept in
  `provenance.engine_hash`; the sink stamps the stage key as `provenance.config_hash`, as for
  rewards. A `dtbias_fp32` deploy knob does not exist yet; when one lands it must reach
  `hashed_spec_for(..., dtbias_fp32=...)` in the hash. The engine, stage and batch-policy
  modules are classified in `config/logic.py` under `vllm-video`; `engines/compile_cache.py` and
  `jobs/pools.py` are classified as not output-affecting.

## Engine settings (`engines/offline_llm.py`)

`EngineSpec.engine_kwargs()` builds every argument explicitly. `validate_engine_kwargs` refuses
any missing or uncertified value. It also runs the capture check's (Gate C) `assert_engine_kwargs`, which requires the
processor cache to be off.

| Argument | TempSeg (merged, `derived/tempseg-merged@114a1076…`) | Captioner (BF16, `derived/captioner-bf16@fef0bbfe…`) |
|---|---|---|
| `dtype` | `bfloat16` | `bfloat16` (its `config.json` still says float32) |
| `gdn_prefill_backend` | `triton` | `triton` |
| `enable_prefix_caching` | `False` | `False` |
| `generation_config` | `vllm` | `vllm` (drops `generation_config.json`'s extra EOS 248044) |
| `mm_processor_cache_gb` | 0 (capture check) | 0 |
| `mm_processor_kwargs` | `{"fps": 4, "max_frames": 128}` | `{"fps": 8, "max_frames": 40}` |
| `limit_mm_per_prompt` | `{"image": 0, "video": 3}` | same |
| `max_model_len` | 32768 | 32768 |
| `max_num_seqs` / `max_num_batched_tokens` / `gpu_memory_utilization` | L40S 48 / 16384 / 0.90; A100-80GB 48; H100 48; H200 64 (on L40S a full batch of worst-case 5.1k-token requests fits in the cache: room for 55) | H100 16 / 16384 / 0.90 (worst case 10.75k tokens: room for 18); H200 32 |
| `seed` | 0 | 0 |
| sampling | greedy, 1024 tokens, `stop_token_ids=[<|im_end|>]` | greedy, 512 tokens, `stop_token_ids=[<|im_end|>]` |

Notes:
- The `mm_processor_kwargs` mirror NVIDIA's processor. An nvlink test checks them against the
  processor that NVIDIA's own `models.processor` builds.
- The token budgets are parsed from NVIDIA's `pipeline.py` (`nv.budget`). The nvlink test also
  asserts that the tokenizer's EOS is `<|im_end|>` for both models.
- Forbidden arguments: quantization, LoRA, KV/Mamba dtype overrides, `renderer_num_workers` and
  `tokenizer_mode`.
- The limits are checked at build: `check_mamba_blocks` fails the engine if `max_num_seqs`
  exceeds the Mamba cache blocks vLLM allocated (#49064). TempSeg on the L40S runs 48: 64 would
  not fit a full batch of worst-case requests (room for 55). The H100 Captioner stays at 16 (room
  for 18) until a measured run shows a higher value holds.
- `logprobs_mode` / `max_logprobs` are settable only for the `dt_bias` drift check (Gate A). They change what
  the engine *returns* for logprob requests, not greedy outputs.
- The startup recipe follows `docs/cold-start.md`: a t=0 `warm.start_prefetch`, then a keyed
  compile cache (`engines/compile_cache.py`). A verified `flora-cache:/balanced-cc/<key>/`
  manifest is copied to local disk, and `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`
  and `XDG_CACHE_HOME` point at that copy. Without one, the lab's `balanced/<kind>` cache is used
  (same GPU and plugin flag only), or the engine compiles cold and publishes its cache once
  (`FLORA_COMPILE_CACHE_PUBLISH=0` turns that off). A build that fails on a loaded cache is
  retried once cold.
- Container shapes, the same in `balanced_lab` and in the deployed pools
  (`config/presets.py`): TempSeg on L40S with 8 cores, 48 GiB (19 GB weights); the Captioner on
  H100 with 8 cores, 96 GiB (55 GB weights, hashed in full while vLLM loads). The memory limit
  is 1.25x the request (`factory.MEMORY_LIMIT_FACTOR`). The planner prices these shapes.

**`dt_bias` FP32 plugin** (`engines/patches/dt_bias_fp32.py`): it is **off by default**. It acts
only when both of these hold:
1. A `vllm.general_plugins` entry point exists. The images do not pip-install this package, so
   `install_entry_point(dir)` writes a minimal `.dist-info` and returns
   `PYTHONPATH`/`VLLM_PLUGINS` for child processes.
2. `FLORA_DTBIAS_FP32=1` is set.

`OfflineEngine` refuses a spec whose `dtbias_fp32` flag disagrees with the environment. The patch
re-creates `QwenGatedDeltaNetAttention.dt_bias` as an FP32 parameter that keeps vLLM's weight
loader, and it is idempotent. `gdn_param_dtypes(llm)` (via `collective_rpc`) reports the dtypes
seen in the engine's workers.

## Tests (CPU)

Test files (counts drift; run them for the current numbers):

- `tests/unit/test_engines_microbatch.py` covers:
  - the all-blocked round across episodes;
  - waiting for a live but busy episode;
  - the timeout bound;
  - finished or cancelled episodes releasing rounds;
  - size thresholds and FIFO order;
  - one generate at a time, with rounds overlapping preparation;
  - retry rounds only for the episodes that need them;
  - batch-failure isolation and per-request errors;
  - SIGINT propagation;
  - bounded pending memory and oversized admission;
  - the closed batcher.
- `tests/unit/test_engines_offline.py` covers:
  - every required argument and refused value;
  - the capture check's cache rule;
  - GPU tables;
  - label and hash views;
  - NVIDIA sampling;
  - `OfflineEngine` batching with per-request isolation and `too_long` detection;
  - the dt_bias spec/env mismatch.
- `tests/unit/test_engines_dtbias_plugin.py` covers:
  - off by default;
  - `register()` is a no-op without the flag (vLLM is never imported);
  - FP32 values are loaded exactly while the unpatched bf16 path rounds;
  - the loader is kept and the patch is idempotent;
  - the entry point is discoverable.
- `tests/unit/test_balanced_stage.py` covers:
  - NVIDIA's acceptance rules (EOS, empty output, the 32k guard on vLLM's length, `too_long`);
  - error classification;
  - the tap collecting and restoring, with one round per call set;
  - first failure in call order;
  - the record round trip;
  - the config hash (sensitive to the engine, not to the model path).
- `tests/unit/test_balanced_workers.py` covers index receipt → NVIDIA arguments and upstream
  segment records across bisection children and config hashes.
- `tests/unit/test_balanced_phases.py` (preparation outside the batcher),
  `tests/unit/test_compile_cache.py` (keyed compile caches) and `tests/unit/test_pools.py`
  (scale to zero and pre-warm).
- `tests/conformance`: `balanced-worker[segment]` and `balanced-worker[caption]` pass all 10
  checks. They use the real runner and the real micro-batcher. The SIGINT is raised inside
  `generate_batch` in the batcher's thread.
- `tests/nvlink/test_balanced_nvlink.py` runs NVIDIA's real code with the real Flora processors and
  synthetic frames. The reference is NVIDIA's sync `pipeline.segment` / `caption_*`. The fake
  engine answers by **content digest** from the reference's calls, so a request NVIDIA never made
  fails the test. It covers:
  - segment parity (rows, metadata, NVIDIA's temporal audit, number of rounds) for 1-view human,
    3-view YAM and 3-view yam32 (two scenarios) × {clean, recentered retry, directional retry,
    conservative completion};
  - missing-EOS and 32k-guard failures raising NVIDIA's `InputErrorForModel` on both paths;
  - two episodes sharing the grid and retry rounds;
  - caption parity for 3-view YAM, 3-view Dex with instruction, human with and without
    instruction, and 1-view robot, including an `unsupported_target_frames` segment;
  - caption failure semantics;
  - the video kwargs and EOS checks.

## GPU validation (lab commands)

```bash
uv run modal run src/flora_deploy/modal_app/balanced_lab.py::preflight     # CPU, cents
uv run modal run src/flora_deploy/modal_app/balanced_lab.py --kind tempseg --run r1    # L40S
uv run modal run src/flora_deploy/modal_app/balanced_lab.py --kind captioner --run r1  # H100!, needs r1's segments
uv run python scripts/spend.py --app flora-balanced-lab
```

**Preflight** (CPU) checks:
- both derived directories and their receipts;
- the `_extra_state` key prefixes in the Captioner index. vLLM's Qwen3.5 language-model loader
  warns and skips unknown names, but the vision tower's loader would `KeyError`, so confirm none
  are under `visual.` before paying for the H100;
- the EOS tokens;
- the episodes the GPU runs would pick: the shortest episode of `human`, `yam`,
  `single_view_robot` and `dex` that can be compared with the oracle.

**Golden paths.** Manifest `path`s are Volume-relative (`golden/v1/...`), so the lab opens
camera files under the mount itself (`balanced_lab.episode_inputs`). On 2026-10-22 the lab
joined `<mount>/golden/v1` to them. Every episode raised `FileNotFoundError` and the stage sent
zero requests. A baseline that sends zero requests now returns `error` / `no_evidence` and skips
the plugin engine. The runbook's lab step (`scripts/wave_h.py --only lab`) checks the files, the
episodes and the deployed job's segments before it starts a container.

**Each GPU call** runs one container with a 20 min hard timeout and two child processes:
1. **Baseline engine.**
   - The stage runs end to end on the picked episodes.
   - Throughput is recorded as video-s per stage-s and $/video-h at list price, against
     $1.95 + 8 cores + memory for L40S and $3.95 + … for H100.
   - **L0** compares vLLM's `prompt_token_ids` (from the completion and from `_preprocess_cmpl`)
     and `video_grid_thw` with NVIDIA's HF processor replay on 6 captured requests.
   - Greedy runs with top-20 raw logits on 6 requests.
2. **Plugin engine** (FP32 `dt_bias`, its own compile cache) runs the same 6 requests.
   - **Gate A** reports max/mean |Δlogit| over the shared top-k at every position up to the first
     greedy divergence, plus greedy identical/total.

The TempSeg run writes its segment rows to `flora-golden:balanced-lab/<run>/segments.json` for
the caption run. The full JSON lands in `reports/balanced-lab/` (git-ignored).

**Expected cost** (estimate): two cold engine builds per call, then the work. TempSeg: 270 s +
190 s builds, ≈ 615 s (deadline 769 s), ≈ $0.46 (upper $0.62). Captioner: 2 × 325 s builds,
≈ 905 s (deadline 1131 s), ≈ $1.28 (upper $1.69). In total ≈ $1.74, upper $2.31
(`wave_h.lab_seconds`). A warm `flora-cache` entry makes a build cheaper. Each result says
whether its cache was warm.

## Results

<!-- wave-h:results (owned by scripts/wave_h_finalize.py) -->
| Item | Result (Wave H-lite, 2026-10-22; `docs/wave-h-report.md`) |
|---|---|
| L0 in-engine (ids, grid vs HF) | tempseg: pass (6/6 ids, 6/6 grid); captioner: pass (6/6 ids, 6/6 grid) |
| Segments / captions end to end on golden episodes | 1 `balanced` job(s) through `flora deploy` / `flora run` (run `10221548f1b8`); certificate per stage: segment (human) certified, caption (human) fail |
| Throughput, $/video-h vs `exact` | **$5.28/video-h** ±50% vs `exact` $36.39 as `flora plan` prices it (6.9x cheaper), list prices, busy seconds; measured on one 3.9-min job (2026-10-22, n=1), not yet confirmed at scale |
| Gate A (dt_bias FP32 vs bf16) | tempseg: max 0.25, mean 0.034514, greedy 4/6; captioner: max 0.25, mean 0.019797, greedy 6/6; recommendation: plugin **off** |
<!-- /wave-h:results -->

**Recommendation for the `dt_bias` drift check (Gate A):** ship with the plugin **off**, the
default. The TempSeg merge check (`docs/tempseg-merge.md`) already showed BF16-level drift (mean |Δlogit| ≈ 0.025) with identical greedy
JSON on real windows. The `dt_bias` rounding is one more BF16 rounding of 24/48 small vectors, and
its effect should be of the same order or smaller. Run the drift check with the command above before `balanced` is
certified. If the drift exceeds the oracle's cross-GPU noise, turn the plugin on (`dtbias_fp32=True`
in the spec plus `install_entry_point` env) and certify that stack instead.

## Known limits and follow-ups

- The micro-batcher packs across episodes, so batch composition differs from NVIDIA's batch-1
  calls and changes numerics (`docs/rewarder-smoke.md`). That is allowed for `balanced`, and certification
  measures it against the oracle's noise.
- The worker uses NVIDIA's `media.probe` per camera, which fully decodes each camera, plus NVIDIA's
  decode for frames. The `accel` patches (`docs/accel.md`) (decode-once, in-memory PNG, pools) are now wired
  in. `workers_impl/balanced.py` applies every registered patch (each behind its parity test;
  `FLORA_BALANCED_ACCEL=none` turns them off) and runs each episode coroutine inside
  `active.episode()`. `tests/nvlink/test_balanced_accel.py` checks that every model request
  (prompt, frames, metadata digest), every record and every audit is byte-identical with and
  without them. It runs two overlapping episodes (1-view human, 3-view YAM) on real H.264 files
  through NVIDIA's code and the Flora processors. The probe's full decode is still NVIDIA's.
- Pre-rendering and AsyncLLM (online, `render_cmpl_async`) are out of scope. The capture check
  would need extending first.
- `_preprocess_cmpl` is a private vLLM API, used only by the lab's L0 check.
