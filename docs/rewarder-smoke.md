# Rewarder smoke (WS-H): batched vs NVIDIA reference

`stages/reward.py` scores terminal (score only, and with rationale) and dense rewards for an
episode list. NVIDIA's Rewarder `core` does frame sampling, collage, prompts, regexes, parsers,
`score_probabilities` and dense stitching. The reference is NVIDIA's own
`terminal_reward.main` / `dense_reward.main`, run with argv on the same jobs JSONL in the same
container and on the same engine. Rubric rewards are **deferred** (TP=2,
`max_model_len` 98304, one video per camera); `check_mode("rubric")` raises
`NotImplementedError`.

The driver script used for these runs (`scripts/rewarder_smoke.py`) has since been removed; the commands
below are kept as a record.

```bash
uv run modal run scripts/rewarder_smoke.py                            # stage samples + all modes
uv run modal run scripts/rewarder_smoke.py --skip-stage --variants 3  # 9 episodes (round 2 run)
```

Raw outputs are written to `reports/wsh*/`. That directory is git-ignored and never committed.

## How `stages.reward.run` works (round 2)

- **Rounds.** Requests are cut into `generate` rounds bounded by a request count (32) and
  estimated bytes (6 GiB: uint8 frames + 4x for the float32 `pixel_values_videos`, so ~304 MiB
  per 3-view request). With the default `align="nvidia"` every round is *exactly* one of NVIDIA's
  calls: the next `max_num_seqs` (8) episodes for terminal, the next ≤ 8 clips of one episode
  for dense. `align="packed"` fills rounds across episodes for throughput. It is not `exact`:
  the results below show that batch composition changes numerics on the real engine.
- **Decode overlap.** NVIDIA's `Collage` decode runs on a thread pool (default 4 workers, a
  bounded number of episodes ahead) or a spawn process pool (`nv.pool_initializer()`), so round
  k+1 decodes while round k generates. CPU fixture (8 synthetic 3-view 640x480 20 s episodes,
  dense, fake LLM sleeping 0.3 s per request, laptop): inline 16.6 s, threads 10.7 s, processes
  13.5 s (spawn + pickling). Threads are the default.
- **Isolation.** A video that fails to decode becomes an `EpisodeError` before `generate`
  (`nv_failure` for NVIDIA's `ValueError`, else `infra`). Some outputs fail NVIDIA's own checks:
  `parse_terminal` / `score_probabilities` reject them (for example a rationale cut off by
  `max_tokens=160`), or NVIDIA's stitcher rejects a dense episode. Each such output becomes an
  `nv_failure` carrying NVIDIA's message, the `finish_reason`s and the output text. If
  `generate` raises for a round, each episode is retried alone. The run fails only if every
  episode also fails alone (a dead engine).
- **Records.** Output is `records.StageRecord` with stage `reward.terminal` / `reward.dense` and
  `uid` = `episode_uid(dataset, id)`. `Provenance` is complete:
  - models, engine, preset, framework, GPU and per-camera source;
  - a config hash over engine args, sampling, trim, pins, logic version and image fingerprint.

  `certified` is True only when the episode index says the episode is inside NVIDIA's envelope
  (`envelope_ok=True`). `False` gives `outside_reference_envelope`. No verdict (ad-hoc jobs,
  these samples) gives `oracle_ineligible_source`.

  `DenseReward.per_step` holds one score per 0.5 s, the model's group length. Each is NVIDIA's
  stitched per-frame score read at the middle of the step, as NVIDIA's `per_second` does per
  second. NVIDIA's full output line goes into a separate audit entry, with per-request
  `finish_reason`s and flags (`truncated_clips`, `invalid_clips`).
- **Partial persistence.** After every round, its records, audits and errors are written as
  whole part files:
  - `LocalPartSink`: a local file, then copy + rename;
  - `StorePartSink`: a local file, then `store.put_bytes` to `JobPaths.part/audit/errors`.

  A timeout or preemption loses at most the round in flight.
- **Licensing note.** About 20 lines re-express NVIDIA's script glue against its interface:
  `SamplingParams`, the `score_probabilities` input list, the terminal output line's fields and
  rounding, and the per-step reading. No prompt, regex or parser is copied.
  `tests/nvlink/test_reward_parity.py` runs NVIDIA's real scripts as the drift guard.

## Samples (`flora-golden:/samples/wsh/`)

All three are LeRobot v2.1 datasets with one AV1 MP4 per camera, constant frame rate and 640x480
frames. Cameras are listed in NVIDIA's collage order.

| id | dataset @ revision (tag v2.1) | episode | cameras (order) | fps | frames | seconds | license |
|---|---|---|---|---|---|---|---|
| `aloha_coffee_ep000` | `lerobot/aloha_static_coffee` @ `7da22eb685a6` | 0 | cam_left_wrist, cam_high, cam_right_wrist | 50 | 1100 | 22.00 | MIT |
| `ur5_ep002` | `lerobot/berkeley_autolab_ur5` @ `6306aa91c00b` | 2 | image, hand_image | 5 | 81 | 16.20 | CC-BY-4.0 |
| `toto_ep002` | `lerobot/toto` @ `35ba68ad482f` | 2 | image | 30 | 1160 | 38.67 | MIT |

The task text is the episode's `tasks[0]` from `meta/episodes.jsonl`. `sources.json`, next to
`jobs.jsonl`, records each file's SHA-256 and its probe results.

## Engine (one per container, `H100!`)

The engine is built with NVIDIA's `engine.load_llm`, using the terminal script's arguments:

| Setting | Value |
|---|---|
| dtype | `bfloat16` |
| `gdn_prefill_backend` | `triton` (log: "Using Triton/FLA GDN prefill kernel (requested=triton)") |
| `enable_prefix_caching` | `True`, resolved to `mamba_cache_mode="align"` (experimental) |
| `block_size` | 784 |
| `max_model_len` | 16384 (both modes) |
| `max_num_seqs` | 8 |
| `max_num_batched_tokens` | 16384 (LLM_CLASS default) |
| `gpu_memory_utilization` | 0.88 |
| `seed` | 0 |
| `max_logprobs` | 20 (vLLM's default is also 20, so dense's call, which omits it, is the same engine) |
| `limit_mm_per_prompt` | `{"image": 0, "video": 1}` |
| `mm_processor_kwargs` (engine) | `{"do_sample_frames": false, "size": {"shortest_edge": 4096, "longest_edge": 29360128}}` |
| `mm_processor_kwargs` (request) | `{"do_sample_frames": false}` |
| Software | vLLM 0.24.0 |

How the reference reuses the engine:
- `shared_engine` replaces the script's `load_llm` with a function that compares every argument
  with the shared engine's (`EngineMismatchError` otherwise), then returns that engine. NVIDIA's
  calls matched in all three modes.
- `reference` calls `reset_prefix_cache()` first. `stages.reward.run` calls it before every
  round and checks that it returned True.
- The shared text before the video is 104-413 tokens, under one 784-token block, so only
  *identical* requests could ever hit the prefix cache. Duplicate episodes are rejected.
- The order of runs per mode is reference, aligned, unbounded.

## Results, round 2 (2026-10-21)

Two ephemeral `H100!` runs on `flora-golden:samples/wsh/`:
- **Run A**: the 3 staged episodes, on a fresh `flora-cache` key.
- **Run B**: 9 episodes, each sample under 3 task phrasings, so NVIDIA's terminal script makes
  calls of 8 + 1. The compile and Triton caches were warm from run A.

"Aligned" is the new default (`align="nvidia"` rounds, thread decode). "Unbounded" is one packed
`generate` per mode, the previous behaviour.

| Run | Mode | NVIDIA calls | Aligned rounds | Aligned vs NVIDIA | Unbounded vs NVIDIA |
|---|---|---|---|---|---|
| A | terminal | [3] | — | — | identical (ΔP 0) |
| A | terminal + rationale | [3] | — | — | identical (ΔP 0) |
| A | dense | [3, 2, 4] | — | — | identical |
| B | terminal | [8, 1] | [8, 1] | **identical, ΔP 0** | text equal, max ΔP 0.022 |
| B | terminal + rationale | [8, 1] | [8, 1] | text equal; ΔP 0 on 7/9, 2.5e-6 and 0.0048 on 2 | text equal, max ΔP 0.020 |
| B | dense | 9 calls, 27 clips | the same 9 groups | **identical** (all clip texts, per-frame scores) | **clip text differs** |

Run A also ran small packed rounds ([2, 1] for terminal). Greedy text and scores were equal, but
P(1..5) was up to 0.027 away from the single call.

What this shows:
- **Batch composition changes numerics on the real engine.** The previous "one `generate` for
  everything" design was not exact. On 9 episodes it moved terminal probabilities by up to 0.022
  and changed dense clip text, which confirms review M2. With a content-keyed fake LLM every
  grouping gives the same records (unit and nvlink tests), so the difference is in the numerics,
  not in the requests.
- **NVIDIA-aligned rounds reproduce NVIDIA's outputs** for terminal and dense on this set.
- **Terminal + rationale still drifts slightly** on 2 of 9 episodes (max 0.0048). Grouping,
  inputs and prompt-token counts were identical, and both runs started from a reset prefix
  cache. That points to run-to-run engine nondeterminism (async scheduling with chunked prefill
  while sequences of different lengths decode), not to our grouping. This was not proved: an
  NVIDIA-vs-NVIDIA re-run did not fit the budget.
- No errors, no flags (no truncation) and no budget splits.
- Every record is uncertified with `oracle_ineligible_source`, because the samples have no
  envelope verdict.
- Part files were written per round: terminal had 2 rounds (8 + 1 records), dense had 9.

Timing in run B, warm engine, as `generate` s / wall s per run:

| Mode | NVIDIA reference | Aligned (thread decode) | Unbounded (inline decode) |
|---|---|---|---|
| terminal | 49.1 / 63.8 (first call, includes warm-up) | 6.6 / 11.3 | 6.5 / 19.1 |
| terminal + rationale | 9.7 / 23.9 | 8.1 / 12.7 | 7.5 / 20.2 |
| dense | 35.9 / 52.4 | 34.0 / 38.0 | 26.4 / 40.5 |

Overlapped decode is the main win. Aligned rounds waited 4-5 s for decode, against 13-14 s of
serial decode before the unbounded call. Packed dense generates faster (26 s against 34 s) but is
not exact.

### Startup with the new recipe

The recipe is a t=0 prefetch, compile and Triton caches on `flora-cache`, and 80 GiB of memory.

| Phase | Run A: cold cache (us-east) | Run B: warm cache, fresh container (us-west) | Round 1 (no recipe) |
|---|---|---|---|
| Prefetch of 51.7 GiB with 16 readers | 21.7 s (2.56 GB/s) | 19.4 s (2.87 GB/s) | — |
| Link the NVIDIA family + load the processor | 29.9 s | 28.5 s | — |
| `Loading weights took` | 10.5 s | 9.4 s | ≈ 8 min |
| `init engine` (of which compilation) | 223.8 s (62.9 s) | 151.2 s (39.7 s) | 276 s (79 s) |
| `engine_build_s` | 287.2 s | 200.5 s | 929 s |
| `@modal.enter` total | **318 s** | **229 s** | ≈ 930 s |

- The warm run found the vLLM and Triton directories populated. It saved 87 s, less than W3-B's
  ≈ 130 s estimate: loading the compile cache from the Volume still took 40 s, and
  profiling/warm-up is still most of `init engine`.
- `CUDA_CACHE_PATH` and `XDG_CACHE_HOME` stayed empty, because nothing was JIT-built there.
- NVIDIA's `load_llm` arguments are unchanged.

### Spend

- Run A: 436 s of container uptime. Run B: 474 s.
- At $4.97/h (H100 + 8 CPU + 80 GiB) that is ≈ $0.60 + $0.65 = **≈ $1.25**, plus image and
  scheduling overhead.
- `scripts/spend.py --app flora-rewarder` still showed the earlier $1.88 when this was written,
  because billing lags by about an hour.

## Results, round 1 (2026-10-21, previous single-`generate` implementation)

| Mode | Requests | NVIDIA `generate` calls | Batched calls | Text | Scores / records | Max \|ΔP\| |
|---|---|---|---|---|---|---|
| terminal | 3 | 1 | 1 | 3/3 equal | identical | 0.0 |
| terminal + rationale | 3 | 1 | 1 | 3/3 equal | identical | 0.0 |
| dense | 9 clips (3+2+4) | 3 (one per episode) | 1 | 9/9 clips equal | identical, per-frame scores equal | n/a |

"Identical" means the batched records' `nv` field equals NVIDIA's output line for every key, and
the unrounded `P(1..5)` (tapped from `core.score_probabilities`) differ by exactly 0.

Dense batches the 9 clips of all episodes in one call. NVIDIA calls once per episode, so batch
composition differs. Even so, the greedy outputs and per-frame scores were identical on this
set. This is not a guarantee: prefix-cache "align" mode and batch composition can still change
numerics on other data.

Prompt tokens per request were equal for the reference and batched runs:
- terminal: 10790 / 7301 / 3851;
- with rationale: 10803 / 7314 / 3864;
- dense: 11168, 11188, 11193 / 7684, 7696 / 4222, 4242, 4247, 4247.

| Mode | Run | Prompt tok | Output tok | `generate` s | Requests/s | Prompt tok/s |
|---|---|---|---|---|---|---|
| terminal | reference | 21942 | 15 | 32.50* | 0.09 | 675 |
| terminal | batched | 21942 | 15 | 2.25 | 1.33 | 9733 |
| terminal + rationale | reference | 21981 | 98 | 5.00 | 0.60 | 4394 |
| terminal + rationale | batched | 21981 | 98 | 2.96 | 1.01 | 7424 |
| dense | reference | 65887 | 531 | 12.08 | 0.75 | 5455 |
| dense | batched | 65887 | 531 | 8.62 | 1.04 | 7642 |

\* This was the engine's first `generate` call, which includes one-time warm-up. The terminal
modes make the same single call in both runs, so their batched speed-up comes from running warm
second, not from batching. Only dense shows a real batching gain (1.4x), because one call
replaces three.

Cold start and cost:
- Engine build: 929 s. Weights loaded slowly from the Volume (9P, about 8 min this run, 99 s in
  an earlier run), followed by 276 s of init, of which 79 s was compilation.
- Container uptime: 1059 s.
- Spend for WS-H: about $1.9 in total. That covers two GPU runs (the first stopped on a model-path
  check, since fixed) and the CPU staging run.
