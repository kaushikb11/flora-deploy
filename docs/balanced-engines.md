# P1b-1: vLLM engines and the `balanced` segment / caption stages

Status: 2026-10-21. Code, unit tests, conformance and nvlink tests are done and green on CPU.
**GPU validation has not been run.** The user put it on hold for budget reasons. It is ready to run
as one command per model kind (see "GPU validation" below). No Modal GPU container was started,
and P1b-1 has spent **$0**.

The `balanced` kinds are still refused by `modal_app/factory.py`. Certificates (P1b-2) unlock them.
Every record these stages write is `certified: false` with `uncertified_reason: "uncertified_stack"`.

## Design

```
episode coroutines (one event loop, ≤ max_inflight_episodes)
  NVIDIA FineSubtaskSegmenter ──await apredict(request)──► MicroBatchPredictor (subclass of NVIDIA's Predictor)
                                                             │ to_thread: NVIDIA's own Predictor.predict
                                                             │   (window, frame plan, frames_for, human_transport,
                                                             │    temporal_inputs → CaptureProcessor)
                                                             │   pipeline.generate is tapped → CapturedRequest + sentinel
                                                             ▼
  NVIDIA caption_three / caption_single ── same tap ──► CapturedRequests (Gate C wrap: vllm_prompt)
                                                             ▼
                                         MicroBatcher.generate_many  (round-based, D14)
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
- **Micro-batcher** (`engines/microbatch.py`). Each episode runs inside `async with mb.episode()`.
  A round fires when one of these holds:
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
- **Workers** (`workers_impl/segment.py`, `workers_impl/caption.py`). `BalancedShardRunner`
  follows the full `ShardWorker` contract by mirroring `RewardShardRunner`, and reuses its
  `ShardPartSink`, `parse_index_receipt` and `VideoResolver`. Episodes run concurrently on one
  event loop and their records are flushed every `flush_every` episodes. Index receipts give
  NVIDIA's arguments (profile, yam32, instruction source). Caption shards read their episodes'
  `segment` records from the upstream shard's parts, including bisection children, filtered to the
  job's config hash. The full NVIDIA segment metadata travels in `Segment.nv_metadata`, and
  `rows_from_record` turns it back into NVIDIA's rows. Both workers are registered in
  `tests/conformance/registry.py` and pass all 10 conformance checks.
- **Provenance.** The engine label is `vllm-0.24.0-offline` or `…+dtbias`. Records carry the
  model pins, the GPU and the per-camera source. The stage config hash covers the engine spec
  without the GPU (D11; the per-GPU scheduler limits stay in), NVIDIA's sampling, the logic version
  and the `vllm-video` logic digest. The job hash is stamped by the sink, as for rewards. All new
  modules are classified in `config/logic.py` under `vllm-video`. `engines/` and
  `engines/patches/` are now classified directories.

## Engine settings (`engines/offline_llm.py`)

`EngineSpec.engine_kwargs()` builds every argument explicitly. `validate_engine_kwargs` refuses
any missing or uncertified value. It also runs Gate C's `assert_engine_kwargs`, which requires the
processor cache to be off.

| Argument | TempSeg (merged, `derived/tempseg-merged@114a1076…`) | Captioner (BF16, `derived/captioner-bf16@fef0bbfe…`) |
|---|---|---|
| `dtype` | `bfloat16` | `bfloat16` (its `config.json` still says float32) |
| `gdn_prefill_backend` | `triton` | `triton` |
| `enable_prefix_caching` | `False` | `False` |
| `generation_config` | `vllm` | `vllm` (drops `generation_config.json`'s extra EOS 248044) |
| `mm_processor_cache_gb` | 0 (Gate C) | 0 |
| `mm_processor_kwargs` | `{"fps": 4, "max_frames": 128}` | `{"fps": 8, "max_frames": 40}` |
| `limit_mm_per_prompt` | `{"image": 0, "video": 3}` | same |
| `max_model_len` | 32768 | 32768 |
| `max_num_seqs` / `max_num_batched_tokens` / `gpu_memory_utilization` | L40S 32 / 16384 / 0.90; H100 48; H200 64 | H100 16 / 16384 / 0.90; H200 32 |
| `seed` | 0 | 0 |
| sampling | greedy, 1024 tokens, `stop_token_ids=[<|im_end|>]` | greedy, 512 tokens, `stop_token_ids=[<|im_end|>]` |

Notes:
- The `mm_processor_kwargs` mirror NVIDIA's processor. An nvlink test checks them against the
  processor that NVIDIA's own `models.processor` builds.
- The token budgets are parsed from NVIDIA's `pipeline.py` (`nv.budget`). The nvlink test also
  asserts that the tokenizer's EOS is `<|im_end|>` for both models.
- Forbidden arguments: quantization, LoRA, KV/Mamba dtype overrides, `renderer_num_workers` and
  `tokenizer_mode`.
- The per-GPU limits are first guesses below the #49064 Mamba-block bound. The GPU run must confirm them.
- `logprobs_mode` / `max_logprobs` are settable only for Gate A measurements. They change what
  the engine *returns* for logprob requests, not greedy outputs.
- The startup recipe follows `docs/cold-start.md`: a t=0 `warm.start_prefetch`, plus
  `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` and `XDG_CACHE_HOME` under
  `flora-cache:/balanced/<kind>[-dtbias]/`.
- Container memory is 48 GiB for TempSeg on L40S (19 GB weights) and 96 GiB for the Captioner on
  H100 (55 GB weights).

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

- `tests/unit/test_engines_microbatch.py` (15) covers:
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
- `tests/unit/test_engines_offline.py` (26) covers:
  - every required argument and refused value;
  - Gate C's cache rule;
  - GPU tables;
  - label and hash views;
  - NVIDIA sampling;
  - `OfflineEngine` batching with per-request isolation and `too_long` detection;
  - the dt_bias spec/env mismatch.
- `tests/unit/test_engines_dtbias_plugin.py` (5) covers:
  - off by default;
  - `register()` is a no-op without the flag (vLLM is never imported);
  - FP32 values are loaded exactly while the unpatched bf16 path rounds;
  - the loader is kept and the patch is idempotent;
  - the entry point is discoverable.
- `tests/unit/test_balanced_stage.py` (11) covers:
  - NVIDIA's acceptance rules (EOS, empty output, the 32k guard on vLLM's length, `too_long`);
  - error classification;
  - the tap collecting and restoring, with one round per call set;
  - first failure in call order;
  - the record round trip;
  - the config hash (sensitive to the engine, not to the model path).
- `tests/unit/test_balanced_workers.py` (3) covers index receipt → NVIDIA arguments and upstream
  segment records across bisection children and config hashes.
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

## GPU validation (ready to run, not run)

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
- the episodes the GPU runs would pick: the shortest oracle-eligible episode of `human`, `yam`,
  `single_view_robot` and `dex`.

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

**Expected cost** (estimate): about 12–15 min of container time per kind, two engine starts each.
That is ≈ $0.6 for L40S and ≈ $1.2 for H100, ≈ $1.8–2.4 in total, within the $3 cap.

## Results

| Item | Result |
|---|---|
| L0 in-engine (ids, grid vs HF) | **not run** (Gate C already passed this on 127 requests through the same `vllm_prompt`; the in-engine re-check is pending) |
| Segments / captions end to end on golden episodes | **not run** |
| Throughput, $/video-h vs `exact` (≈ $25–40 / video-h, `docs/perf-1.md`) | **not measured**; PLAN's ≈4× cheaper is still an estimate |
| Gate A (dt_bias FP32 vs bf16) | **not measured** |

**Recommendation for Gate A until it is measured:** ship with the plugin **off**, the default.
P1b-0's merge check already showed BF16-level drift (mean |Δlogit| ≈ 0.025) with identical greedy
JSON on real windows. The `dt_bias` rounding is one more BF16 rounding of 24/48 small vectors, and
its effect should be of the same order or smaller. Run Gate A with the command above before P1b-2
certifies. If the drift exceeds the oracle's cross-GPU noise, turn the plugin on (`dtbias_fp32=True`
in the spec plus `install_entry_point` env) and certify that stack instead.

## Known limits and follow-ups

- The micro-batcher packs across episodes, so batch composition differs from NVIDIA's batch-1
  calls and changes numerics (`docs/rewarder-smoke.md`). That is allowed for `balanced`, and P1b-2
  measures it against the oracle's noise.
- The worker uses NVIDIA's `media.probe` per camera, which fully decodes each camera, plus NVIDIA's
  decode for frames. The OPT-1 `accel` patches (decode-once, in-memory PNG, pools) are compatible
  (they patch the same NVIDIA names the stages call), but they are not wired into the workers yet.
  Turning them on needs `apply_patches` plus `active.episode()` per episode coroutine.
- Pre-rendering and AsyncLLM (online, `render_cmpl_async`) are out of scope. Gate C would need
  extending first.
- `_preprocess_cmpl` is a private vLLM API, used only by the lab's L0 check.
