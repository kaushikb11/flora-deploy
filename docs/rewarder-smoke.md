# Rewarder smoke (WS-H): batched vs NVIDIA reference

`stages/reward.py` scores terminal (score only, and with rationale) and dense rewards for a whole
episode list. It makes **one `llm.generate` call per mode** and uses NVIDIA's Rewarder `core` for
frame sampling, collage, prompts, regexes, parsers, `score_probabilities` and dense stitching.
The reference is NVIDIA's own `terminal_reward.main` / `dense_reward.main`, run with argv on the
same jobs JSONL in the same container and on the same engine.

```bash
uv run modal run scripts/rewarder_smoke.py              # stage samples + all three modes
uv run modal run scripts/rewarder_smoke.py --skip-stage # reuse flora-golden:/samples/wsh
```

Raw outputs are written to `reports/wsh/`. That directory is git-ignored and never committed.

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
- Each method calls `reset_prefix_cache()` first, so the reference and batched runs both start
  with a cold prefix cache.
- The order of runs per mode is reference, then batched.

## Results (run of 2026-10-21)

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
