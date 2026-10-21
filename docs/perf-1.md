# Perf-1: why `generate` is ~3–6× slower on the cu128 `exact` image, and the fix

Status: measured 2026-10-21 with `flora-perf-lab` (ephemeral, `src/flora_deploy/modal_app/perf_lab.py`),
two runs, one H200 container per image variant per run. This page holds numbers only. No texts
or logits were written anywhere.

## Answer

- **Root cause: the cuDNN attention graph build is ~20× more expensive in torch 2.11.0+cu128
  than in +cu130, and in NVIDIA's call pattern that build happens on almost every decode
  step.**
  - On the H200, torch's SDPA picks the cuDNN backend (D-1), for prefill and for every decode
    step.
  - Torch caches the cuDNN graph it builds per exact shape key (`s_q`, `s_kv`, strides, ...).
    The cache is **thread-local** (`aten/src/ATen/native/cudnn/MHA.cpp`, v2.11.0:
    `thread_local MHAGraphCache* instance`. The comment says execution plans are not
    guaranteed thread-safe).
  - Each decode step has a new KV length, so a thread that has not yet seen that length builds
    a new graph.
  - NVIDIA's `pipeline.segment` calls `generate` from `asyncio.to_thread` workers. Those belong
    to a fresh event loop on every `segment()` call, so the threads, and their caches, die after
    each episode. The Captioner calls run on our episode-pool threads, and Captioner prompt
    lengths differ on every call. **So in the worker the cache is effectively always cold.**
  - Cost of one cold build, per decode step:

    | Image | Build cost per step |
    |---|---|
    | cu130 | ≈ 18 ms |
    | cu128 | **≈ 395 ms** |

    A warm step costs ≈ 65–70 ms on both images.
- **It is not GPU kernels, clocks, CPU, Python, or the base image.**
  - Warm cuDNN on cu128 is as fast as cu130 (table below).
  - SM clocks sat at 1,980 MHz on both variants, in the phases recorded (run 1's capture phase,
    which includes the cold w0 call).
  - Both images run the same Python 3.12.10 build (`--enable-optimizations --with-lto`) and the
    same cuDNN 9.19.0.56 (cu12 vs cu13 build).
  - The `python:3.12.10-slim-bookworm` base switch is not involved: the slowdown is entirely
    inside the cuDNN SDPA graph build. The `cu130_slim` variant is in the lab but was not run;
    the budget went to the A/B.
- **Fix in our layer, fidelity-neutral: `accel.gpu_thread`** (introduced here as an opt-in;
  wired and **on by default** in the exact worker since P1a-5s, see below).
  - It runs every NVIDIA `generate` on one persistent thread, so each shape's graph is built
    once per container instead of once per call.
  - It changes no input, no argument, no backend and no kernel selection. Only the OS thread
    that calls NVIDIA's unmodified `generate` changes.
- **The fixes that give full speed on every call change the reference:**
  - disabling cuDNN SDPA (that is D-1's `deterministic` mode);
  - going back to cu130, which is not NVIDIA's validated stack: NVIDIA's README installs torch
    from the **cu128** index.

  Each is possible only as an explicit, recorded opt-in. Details below.

## Measurements

Inputs are fixed and built by NVIDIA's own code (D-1's method). The lab replays them into
NVIDIA's unmodified `models.generate`:

- `homer_50` (WS-D sample, human profile), TempSeg windows w0 and w1;
- one Captioner call (`caption_single`, fixed span 2–6 s, BF16 Captioner via OPT-2).

The input digests were identical across variants:

| Input | Prompt tokens | Output tokens |
|---|---|---|
| w0 | 1,811 | 104 |
| w1 | 1,810 | 141 |
| Captioner | 7,464 | 20 |

Container: H200, `cpu=4`, 64 GiB. Times are NVIDIA's `generation_end.elapsed_s` and our synced
per-step timestamps (an `lm_head` hook).

**Run 2 (`r2`). Every row is the same container and the same input; the rows run in the order
shown.**

| Call | Thread / cache state | cu130 s (ms/step median) | cu128 s (ms/step median) |
|---|---|---|---|
| w0, capture (NVIDIA `segment`, `to_thread` worker) | cold | 13.5 | **47.96** |
| w0, `nvidia` | main thread, **cold** (other thread built the graphs) | 9.23 (85.6) | **48.38 (451.6)** |
| w1, `nvidia` | main thread; lengths ≤ 1,915 warm from w0 | 9.31 (53.9) | 24.61 (61.5) |
| Captioner, `nvidia` | cold (new lengths 7,464+) | 3.90 (91.4) | **11.33 (434.6)** |
| w0, 16-token profile call | warm | 2.29 | 2.07 |
| w0, `nvidia` repeat | warm | 7.30 (66.9) | — (deadline) |
| w1, `nvidia` repeat | warm | 10.19 (69.6) | **9.51 (64.8)** |
| Captioner, `nvidia` repeat | warm | 3.28 (75.4) | 4.75 (80.6) |
| w0, cuDNN SDPA off (`nocudnn`) | — | 7.19 (65.4) | 7.37 (67.5) |
| w1, `nocudnn` | — | 9.94 (68.0) | 9.53 (65.2) |
| Captioner, `nocudnn` (39 tokens) | — | 4.78 (75.7) | 4.52 (69.5) |
| w0, `flash` only | — | 7.46 (67.9) | — (deadline) |
| w0, `nvidia` + 3 PNG-encoding threads (GIL load) | warm | 8.37 (76.8) | — (deadline) |

- Prefill: cold 1.35–1.72 s on cu128 against 0.41–0.47 s on cu130 and on warm cu128. The
  Captioner prefill was 3.2 s cold against 2.0 s.
- `efficient`-only SDPA: "No available kernel" for these shapes. That backend is not a fallback.
- Run 1 (`r1`), which aborted later on a lab bug, shows the same first call:

  | | Container start to ready | w0 capture call |
  |---|---|---|
  | cu128 | 80.8 s | **62.0 s** |
  | cu130 | 60.5 s | **14.8 s** |

  Here "ready" means TempSeg loaded; the Captioner then took 38 s and 26 s.

**The numbers add up exactly under the thread-local, per-length cache model.**

| Call | Model's account | Predicted | Measured |
|---|---|---|---|
| cu128 w0, 2nd call (main thread) | as cold as the 1st call: the capture call ran on a `to_thread` worker | 48 s | 48.38 s |
| cu128 w1 | ≈ 37 new lengths (1,916–1,951 plus the new prefill shape) × 0.395 s, plus 141 warm steps × 0.067 s, plus prefill | ≈ 25 s | 24.61 s |
| cu128 Captioner | ≈ 20 × 0.395 s + 3.3 s | ≈ 11.2 s | 11.33 s |
| cu128, repeats of the same input on the same thread | warm | ≈ 65 ms/step | 61–81 ms/step (cu130's warm speed) |

The model also explains S1's two figures:

- TempSeg, 1,835 in / 142 out: ≈ 142 × 0.395 s + 142 × 0.067 s ≈ 66 s; S1 measured 54–62 s.
- Captioner, 21 tokens: S1 measured 11.5 s.

S1's 6.3–8.5 GPU-s of generation per video-s, against W3-A's 1.9, is this effect.

**GPU and host.**

- Clocks: in run 1's recorded phases, the SM clock was pinned at 1,980 MHz on both variants
  (memory 3,201 MHz, power cap 700 W). No throttle reasons were set outside idle.
  - During cold cu128 calls `nvidia-smi` utilization is low: the GPU waits while the host
    builds the graph.
  - The per-phase GPU summaries of run 2 were lost with the result upload (next section).
- Torch reports CPU capability AVX2 in one container and AVX512 in the other. That is a
  different host CPU, not an image difference. It cannot explain a 6× CPU-side gap, and the
  warm calls on the same cu128 host match cu130.

**What is not explained:** *why* the cu12 build of cuDNN 9.19 takes ~0.4 s per SDPA graph on
sm90, against ~18 ms for the cu13 build. Runtime compilation (NVRTC) in the cu12 build is the
likely cause. The lab's `torch.profiler` results, which would show the host-side calls, were
lost with run 2's result upload (see Spend). The fix does not depend on the answer.

## Fidelity of each option

| Option | Model inputs | Attention kernels | Output vs NVIDIA default (cu128, cuDNN) | Speed |
|---|---|---|---|---|
| **`accel.gpu_thread`** (proposed default for `exact`) | identical | identical: same backend, same graph key, built by the same heuristics (HeurMode A); only the cache lifetime changes | same computation. Run-to-run variance stays as in D-1 (cuDNN decode is not bit-reproducible on H200 in either case) | warm: 61–81 ms/step. Every new (shape) graph still pays 0.4 s once per container |
| cuDNN SDPA off (= D-1 `deterministic` mode, or `nocudnn`) | identical | flash / math instead of cuDNN | **different reference** (below) | 65–70 ms/step always. Reproducible (D-1) |
| torch +cu130 image | identical | cuDNN cu13 build (other engine binaries) | different stack: **not NVIDIA's validated install** (`batch_utils/README.md`: `pip install torch==2.11.0 ... --index-url .../whl/cu128`). Bit-equality with cu128 is neither expected nor tested | cold 85 ms/step, warm 67 |

Output-distribution impact of turning cuDNN off. The comparison is against the same container's
NVIDIA-default run 0, by text digest and by the first differing `lm_head` step (D-1's method):

| Input | Variant | `nocudnn` / `flash` text = default? | Default repeat text = default r0? (its first differing step) |
|---|---|---|---|
| w0 | cu130 | **no** (step 0) | yes (96) |
| w0 | cu128 | yes (step 0) | — |
| w1 | cu130 | yes (step 0) | yes (138) |
| w1 | cu128 | yes (step 0) | yes (24) |
| Captioner | cu130 | **no: 39 tokens vs 20** | yes (12) |
| Captioner | cu128 | **no: 39 tokens vs 20** | **no: 23 tokens vs 20** (step 8) |

- Turning cuDNN off changed 3 of 6 texts, including a caption that came out twice as long.
- The default path itself changed 1 of 5 texts run to run, in the same container. That is
  D-1's nondeterminism.

So a non-cuDNN backend is a *different, self-consistent* reference. It is acceptable only as the
explicit D-1 `deterministic` opt-in, recorded in provenance and never compared bit-for-bit
against default references.

## `flora_deploy.accel.gpu_thread` (on by default since P1a-5s)

- `GpuThread()` is one persistent worker thread. `run(fn, ...)` executes `fn` there and waits:
  - results and exceptions pass through unchanged;
  - `contextvars` are copied, so stdout event routing, observe recorders and the session
    binding follow the call;
  - a call from the GPU thread itself runs inline.
- **Guard:** autocast, inference mode and grad mode are per thread in torch. `run` refuses a
  caller with any non-default mode instead of dropping it.
  - NVIDIA's call sites set none of them: `models.generate` enters `inference_mode` itself, and
    TempSeg's BF16 autocast is inside its patched `model.generate`.
- `install(nv, patcher)` swaps `nv.pipeline.generate`, the name NVIDIA's `pipeline.py` calls,
  the same way `install_gpu_lock` does. It composes with the GPU lock: the lock is then always
  taken on the GPU thread. `patcher.restore()` gives back NVIDIA's attribute.
  `provenance()` → `flora_gpu_thread_v1`.
- Tests: `tests/unit/test_accel_perf_gpu_thread.py` (10). They cover:
  - one thread for every caller, including short-lived pools like `to_thread`;
  - pass-through of results and errors;
  - contextvars;
  - re-entrancy;
  - the mode guard, with fakes and with real CPU torch;
  - double-install refusal;
  - restore.
- **Wired (P1a-5s).** `stages.reference.ExactProfile.gpu_thread` is a toggle, **on by
  default**: the exact worker installs `gpu_thread` once per container, before the GPU lock, and
  records `provenance()` in the load report and the batch provenance. It is disabled only
  through the profile's `disable` list (`gpu_thread`), which the deploy factory never sets, so
  every deployed exact worker runs with it. The rate table treats it as a deployed feature
  (`planner.rates.DEPLOYED_FEATURES`).

### Expected effect (projection, not yet measured end to end)

These are per-video-second figures for 1080p human, from W3-A's call mix:

| | Calls per video-s | Size per call |
|---|---|---|
| TempSeg | 0.137 | ≈ 90 output tokens; prompt fixed at ~1,810 for human, 5,493 for 3-view |
| Captioner | 0.286 | ≈ 15 output tokens; prompts 5.4k–8.5k, a different length almost every call |

Generation GPU-seconds per video-second:

| Scenario | GPU-s per video-s |
|---|---|
| All cold (current image, no fix) | ≈ 8.8 (S1 measured 8.5 in arm (a), 6.3 in arm (b)) |
| `gpu_thread`, TempSeg warm, Captioner cold (short-lived container) | ≈ 3.9 |
| `gpu_thread`, both warm (long-lived container) | ≈ 1.8 (W3-A on cu130: 1.9) |

- TempSeg lengths stay warm after the first episode of a container.
- The Captioner's decode lengths fill in over the container's life. Covering 5.4k–8.5k is
  ≈ 3.1k lengths × 0.4 s ≈ 20 min of one-off build time per container. Each new prompt length
  also costs one prefill build (≈ 1.3 s).
- The worst case (every Captioner step cold) is still ~2.2× faster than today.

### `planner/rates.json` (as applied; review 5 M-C)

This section originally proposed 7.0 measured (no `gpu_thread`) and 4.35 projected (with it).
P1a-5s then measured the wired worker, and the table now holds:

| Entry (`exact`, `segcap`, cameras `*`) | Value | Status |
|---|---|---|
| concurrency 4, `gpu_thread` on (deployed) | **6.4** container-s/video-s (P1a-5s: 6 golden episodes, 80.97 video-s, 519 s) | `measured`, ±35%. The default for `flora plan` and the budget gate |
| concurrency 4, `gpu_thread` off | 7.0 (S1) | `measured`, `excludes: [gpu_thread]` |
| concurrency 4, `gpu_thread`, Perf-1 projection | 4.35 | `stale` (superseded by the 6.4 measurement of the same configuration) |
| concurrency 8 | 2.5 | `projected`, `requires: [gpu_thread, long_lived_container]`: a projection option only |

The 6.4 is honest for the deployed configuration but carries known biases:
- it is a 1-/3-view mix (half the episodes are 3-view), so it is bucketed cameras `*`;
- the run's container had 160 GiB of memory ($6.20/h, $39.7/video-h), while the preset's
  default pool is 96 GiB ($5.68/h, **$36.4/video-h**); plans price the configured pool;
- it includes the per-shard fill/drain idle (≈110 s fill bubble plus the drain tail, ≈21% of the
  wall; generation alone is 5.04 GPU-s per video-s), which recurs on every shard at the default
  10-GPU-minute shards. Overlapping shards would remove most of it (≈5.2, a projection);
- the Captioner stayed cold (short-lived container), and the run overlapped an FP32 Captioner
  prefetch that production does not do (pessimistic).

The 4.35 projection was ≈1.5× too optimistic for a short-lived container: TempSeg warmed as
predicted (0.41 → 0.27 s/token), the Captioner did not. Long-lived c=8 ≈ $14 stays a
projection. The W3-A $23 figure belongs to the cu130 image.

**Next step (≈ $0.6):** one cu128 run of the lab with `--threads`. Its `fresh_thread` /
`gpu_thread_1` / `gpu_thread_2` / `main_thread_again` modes on w0 directly confirm the cache
mechanism: expected ≈ 48 s / 48 s / 7.3 s / 48 s. Then an S1-style batch with the toggle on,
before S2 relies on the rate.

## Reproduce

```bash
uv run modal run src/flora_deploy/modal_app/perf_lab.py --variant cu128,cu130 \
    --deadline-min 8 --backends nocudnn,flash --reps 2 [--threads] [--profile] --out r.json
```

- Variants:
  - `cu128`: production `img_hfref`;
  - `cu130`: the pre-Fix-images `img_hfref` rebuilt from git `0812b4d`'s lock on `debian_slim`;
  - `cu130_slim`: the current base with the cu130 lock.
- The lab prints one `PERF_RESULT <variant> {json}` line per container.
- `--profile` adds `torch.profiler` (≈ 70 s per call).

## Spend

| Run | What happened | Container time | Estimate |
|---|---|---|---|
| Run 1 | aborted after the capture (lab bug: wrong profile name for the Captioner span) | ≈ 7 min total | ≈ $0.6 |
| Run 2 | | 2 × ≈ 9.5 min | ≈ $1.6 |

- Run 2's structured results were lost: with `block_network=True`, Modal could not upload the
  large return value. The lab now has the network on (HF stays offline) and also logs
  `PERF_RESULT`.
- Every number above comes from run 2's container logs: NVIDIA's `FLORA_RUNTIME`
  `generation_end` events and the lab's per-call lines.
- **Estimated total ≈ $2.2 of the $2.50 cap.** A third run did not fit.
  `scripts/spend.py --app flora-perf-lab` later read **$1.73 billed**. Billing lags ≈ 1 h, so
  the last part may not be posted yet.
