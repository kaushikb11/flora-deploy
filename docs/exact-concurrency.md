# W3-A: episode concurrency for the `exact` preset, and the measurement harness

Status: measured 2026-10-21 on Modal H200 (`flora-reference`, ephemeral). Two containers:

- `wsd-20261021T063220Z`: 4 WS-D episodes, sequential and then 4-way concurrent.
- `wsd-20261021T071056Z`: the ALOHA episode ×2, sequential and then 2-way concurrent.

This page holds numbers only. Results stay on `flora-golden:/reference_runs/<run_id>-c<N>/`.

## What was built

- **`stages/reference.annotate_batch(nv, session, episodes, concurrency=N, ...)`** runs N
  episode threads.
  - The threads share the resident models through `CachedSession`.
  - **One GPU lock serializes every NVIDIA model call.** `install_gpu_lock` swaps the
    `pipeline.generate` attribute at runtime, which is the name NVIDIA's `pipeline.py` imported
    from `.models`. Every model call is therefore exactly NVIDIA's batch-1 `generate()`. No
    NVIDIA file changes.
  - `concurrency=1` (the default) runs in the calling thread with the original processors. This
    is exactly the previous behavior.
- **`ReferenceWorker.annotate_batch(episodes, concurrency, measure, torch_threads, tag)`**
  installs the lock once at `@modal.enter`. It persists results after each batch, from the
  calling thread.
- **`scripts/reference_smoke.py`** (since removed) had new flags:
  `--concurrency N --sequential --measure [--cpu C] [--torch-threads T] [--repeat K] [--compare-to S]`.
  It runs the same episodes sequentially and then concurrently in one container, and compares
  NVIDIA's full result dicts. The comparison leaves out only `elapsed_s` and our
  `flora_reference` key.
- **`src/flora_deploy/observe/`**, the generic, opt-in measurement harness:
  - `timed`/`Patcher`: wall time, self time (nested timed calls subtracted) and thread CPU per
    call. They wrap module or class attributes at runtime.
  - `TimedLock`: wait time, hold time, busy fraction.
  - `Sampler`: RSS, cgroup memory, process CPU, `/dev/shm` and load, read from `/proc`.
  - `route_stdout`: a per-context stdout router.
  - `call_table` and `token_stats`: per-episode tables.
  - `emit_json`: one `FLORA_OBSERVE {json}` line per record.

  The NVIDIA hot-path target list is `stages/reference.measurement_targets`:
  - `media.probe`, `decode_selected` and `frames_for`;
  - `human_transport`, `temporal_inputs` and `three_view_caption_inputs`;
  - `unified_input.processor_inputs` and `load_frames`;
  - vendor `canonical_rgb`, the mosaic composer and caption geometry;
  - HF processor `__call__` and `apply_chat_template`;
  - PIL `save` (split by format), `resize` and `convert`;
  - `models.generate`, timed inside the lock.

## Thread-safety audit (what the episode threads share)

| Shared thing | Safe? | Handling |
|---|---|---|
| Models (TempSeg, Captioner) | Used only inside `generate` | GPU lock. `CachedSession.load` is already locked, and the models are preloaded |
| HF processor/tokenizer | **No.** `PreTrainedTokenizerFast.set_truncation_and_padding` mutates the Rust tokenizer per call: `caption_three` pads, the other builders don't. Encoding releases the GIL, so a concurrent padding change can race or raise "Already borrowed" | Each running episode leases private `copy.deepcopy` clones (`ProcessorClones` pool, reused across batches). NVIDIA hands that same clone to `generate`. Clone cost: ≈3 s per processor, paid once per concurrency slot |
| NVIDIA module globals | Yes: no mutable state on the episode path (`_VERIFIED_MANIFESTS` is used at load time only, under NVIDIA's lock; `api._REQUEST_LOCK` is unused). Vendor code has no caches or `global`s | none |
| `asyncio.run` in `pipeline.segment` | Yes: each call has its own loop; the predictor runs in that loop's `to_thread` executor | none |
| `runtime_event` stdout logging | `redirect_stdout` is **not** thread-safe: it swaps the global `sys.stdout` | `capture_runtime_events` now uses `observe.route_stdout`. Lines are routed by `ContextVar`, which follows `asyncio.to_thread`, and are forwarded whole under a lock |
| `generation_end.peak_allocated_gib`, `torch.cuda.*peak*` | Process-global | Diagnostic only; not part of the result |
| Temp dirs (`caption_single` PNGs) | `TemporaryDirectory`, unique per call | none |
| Volume writes and commits | Done after the batch, in the calling thread | none |

Local proof on CPU (`tests/unit/test_reference_concurrency_nvlink.py`, real NVIDIA builders and
real Flora processors, fake model):

- Concurrent episodes with cloned processors hand `generate` byte-identical tensors to a
  sequential run.
- Torch intra-op threads set to 1, 4 or ≥8 give identical tensors.
- Unit tests with a fake family check that:
  - the lock admits at most one `generate`, and the same fake overlaps without it;
  - each episode's runtime events stay with that episode;
  - clones are never shared;
  - errors are per episode.

## Identity result

**Verdict: concurrency adds no difference of its own. But a strict "bit-identical to sequential"
check cannot pass on this stack, because NVIDIA's sequential reference is itself not
reproducible run to run on the H200.**

Evidence:

1. **The inputs to every model call are byte-identical.** NVIDIA's own `generation_start` event
   carries `input_tensor_digest`, a sha256 over every input tensor's bytes, dtype and shape.
   Across the four batches there were 146 model calls on 76 distinct inputs. Every call in a
   concurrent batch matched, by digest, a call in a sequential batch, except calls that come
   after an earlier model output had already diverged. 72 inputs appeared in more than one
   batch; the 4 that appeared only once are all downstream of such a divergence. No call ever
   had a different input while everything upstream of it agreed.
2. **Run 2 is fully identical.** ALOHA 3-view, 8 calls per episode: seq r0, seq r1, concurrent
   r0 and concurrent r1 all have the same full NVIDIA result (`656a546e…`). That digest is also
   identical to the WS-D run from a different container an hour earlier.
3. **Run 1 is not identical, and the sequential batch is the outlier.** Digests per episode:

| Episode | WS-D (sequential, other container) | run 1 sequential | run 1 concurrent ×4 |
|---|---|---|---|
| aloha 3-view | `656a546e` | `ea319346` | `656a546e` (= WS-D, = run 2) |
| homer_50 | `125a5169` | `89397eed` | `125a5169` (= WS-D) |
| homer_52 | `d9219705` | `52aa9be2` | `b3258e43` |
| homer_53 | `8a96e3d7` | `0a33224f` | `9151d948` |

   - Each first difference is a model output for a byte-identical input. Three such calls in
     run 1 even differ in length: 9 vs 18, 20 vs 23 and 83 vs 43 generated tokens.
   - Everything after the first difference (segment boundaries, then caption inputs) follows
     from it.
   - Two sequential runs (WS-D and run 1 sequential) disagree on **all 4** episodes. The
     concurrent batch matches an independent sequential run on 2 of 4.

   So the variance is GPU-side, inside batch-1 HF `model.generate`. It shows up in sequential
   runs just as much.
4. Determinism flags in the container:
   - `deterministic_algorithms=False`, `cudnn_benchmark=False`, `matmul_allow_tf32=False`,
     `cudnn_allow_tf32=True`;
   - no `fla`, `causal_conv1d` or `flash_attn`, so GDN runs on the torch fallback;
   - `triton` is present.

   The root cause is not found yet. Candidates include allocator-dependent kernel selection and
   non-deterministic CUDA reductions. Turning on `torch.use_deterministic_algorithms` would
   change the reference itself, which needs a decision.

**Consequence for the golden build:** "oracle == oracle" will not hold bit for bit, even
sequentially. Gate thresholds must be set from measured run-to-run variance, not zero.
W3-A's concurrency is safe to use for the oracle runs: CPU-side inputs are identical, and model
calls are NVIDIA's batch-1 calls, one at a time.

## Speedup

Container: `cpu=8`, 96 GiB, H200, torch intra-op threads 4 (NVIDIA's setting).

| Batch | Episodes (video s) | Wall s | Video-h per GPU-h | Speedup | GPU busy (lock held) | Lock wait total / max |
|---|---|---|---|---|---|---|
| run 1 sequential | 4 (153.6) | 1,322 | **0.116** | 1.00× | 22.1% | 0 |
| run 1 concurrency 4 | 4 (153.6) | 626 | **0.245** | **2.11×** | 54.3% | 203 s / 23.9 s (27 of 73 calls waited) |
| run 2 sequential | 2× ALOHA (44.0) | 175 | 0.252 | 1.00× | 41.1% | 0 |
| run 2 concurrency 2 | 2× ALOHA (44.0) | 105 | 0.418 | 1.66× | 69.9% | 10.2 s / 6.9 s |

- Run 1 at concurrency 4 is **bounded by its longest episode**. homer_53 takes 546 s alone and
  625 s under contention. The ideal for this set is 1,322 / 546 = 2.42×; we got 2.11×.
- Process CPU averaged **2.16 busy cores** of 8, so CPU was not the limit.
- Under contention, `generate` hold time grew 16% (293 → 340 s), decode 8%, PNG writes 2%.
  `probe` wall grew 63% at the same CPU time, which is GIL contention in its per-frame `Fraction`
  loop.
- **Steady state (projection, many episodes):** GPU work is 293 s for 153.6 video s, i.e. 1.9
  GPU-s per video-s. At ≈90% lock occupancy with +16% contention, that gives ≈0.40 video-h per
  GPU-h, about 3.5× sequential. CPU needed: 8.75 core-s per video-s, i.e. ≈3.5 busy cores at
  that rate, so `cpu=8` has headroom. This needs a ≥8-episode batch to confirm (the golden
  build will).

## Where the time goes (Gate D evidence)

Run 1 sequential, 4 episodes, 1,322 s wall. Self time, so nested timed calls are not double
counted. CPU is the calling thread's.

| Component | Calls | Self s | % of wall | Thread CPU s |
|---|---|---|---|---|
| **PIL PNG save** (`caption_single` writes every caption frame as PNG) | 1,959 | **572.7** | **43.3%** | 564.0 |
| `models.generate` (GPU; lock held) | 73 | 292.7 | 22.1% | 288.1 (host thread spins on CUDA syncs) |
| `media.decode_selected` (PyAV, decodes from frame 0 on every call, `thread_count=1`) | 48 | 246.2 | 18.6% | 249.8 |
| `unified_input.load_frames` (PNG re-read + sha256 check, own part) | 44 | 78.3 | 5.9% | — |
| PIL resize | 6,138 | 48.6 | 3.7% | 47.9 |
| `media.probe` (full decode + sha256) | 6 | 36.1 | 2.7% | 34.1 |
| HF processor `__call__` (torch intra-op ×4) | 73 | 19.5 | 1.5% | 18.7 |
| PIL convert | 11,547 | 13.2 | 1.0% | 12.7 |
| vendor `canonical_rgb` (3-view decode path) | 1,164 | 4.7 | 0.4% | 4.4 |
| JPEG q98 round trip (`human_transport`) | 1,240 | 2.2 + 1.3 | 0.3% | 2.3 |
| caption geometry, mosaic, prompt building, chat template | — | ≈1.5 | 0.1% | — |
| not covered by any timer | — | ≈5 | 0.4% | — |

**Headline:** the model is 22% of the wall-clock. PNG encoding of caption frames alone is 43%,
and decode is 19%. The HF processor is only 1.5%, so torch thread tuning cannot matter.

Tokens (run 1, NVIDIA `generation_end` events):

| | Calls | Prompt tokens mean / max | Output tokens mean / max | Output limit |
|---|---|---|---|---|
| TempSeg, human 1-view | 21 | 1,810 / 1,811 | 64.9–111.5 / 141 | 1,024 |
| TempSeg, ALOHA 3-view | 2 | 5,493 / 5,493 | 74 / 84 | 1,024 |
| Captioner, human | 44 | 5,423–7,416 / 8,505 | 11.4–19.2 / 27 | 512 |
| Captioner, ALOHA 3-view | 6 | 11,145 / 13,650 | 16.7 / 26 | 512 |

Memory and IPC:

- RSS: 4.6 GiB after load, **11.8 GiB max sequentially**, **24.6 GiB max at concurrency 4**
  (cgroup 24.3 GiB). That is ≈3.2 GiB per extra concurrent episode: decoded 1080p frames plus
  processor clones.
- 96 GiB is far more than needed after load. The load itself needs ~18 GiB of CPU BF16 for the
  TempSeg base.
- GPU high-water mark: 81.0 GiB allocated, 84.7 GiB reserved, 87.9 GB in `nvidia-smi` at
  concurrency 4. Only one model call runs at a time, so this matches the sequential peak plus
  cache.
- `/dev/shm`: 0.004 MiB used. Nothing uses shared memory; temp PNGs go to `/tmp` on local disk.
- Startup (`@modal.enter`): 213 s in run 1 and 169 s in run 2 (WS-D: 251 s).

## Tuning

- **Concurrency:** use 4. It is measured at 2.1× on a straggler-bound set, with GPU busy at 54%.
- **More slots:**
  - 6–8 should approach the GPU bound on large batches: ≈3.5×, at +3.2 GiB RSS per slot.
  - Packing long episodes first (the script sorts by duration) limits straggler tails.
  - The ceiling is the GPU lock, plus a long single episode's own critical path.
- **`cpu=` cores:** a `cpu=16` run was not made. The budget only allowed a second run, and that
  run was better spent on the determinism question. The evidence says 8 is enough: 2.16 busy
  cores average at concurrency 4, and a projected ≈3.5 at GPU saturation.
  - Keep `cpu=8`. Modal reports 24 host CPUs and sets no cgroup `cpu.max`.
  - Revisit if a ≥8-slot batch shows `cores_busy` near 8.
- **Torch intra-op threads:** keep NVIDIA's 4. Locally, 1, 4 and ≥8 threads give
  byte-identical processor tensors, and torch threads do not reach the GPU path. But the HF
  processor is only 1.5% of wall time, so raising them buys nothing.
- **Memory:** 48 GiB would hold concurrency 4–6 with margin. Not changed here because of the
  load-time CPU staging.
- **Biggest remaining lever:** PNG writes and re-reads (≈49% together) and per-call decode from
  frame 0 (19%). Both are inside NVIDIA's code, so improving them needs a proven wrap or a
  certified reimplementation (PLAN D17). Concurrency only hides them.

## Cost: `exact` $/video-hour (Modal list prices)

Container rate:
- H200 at $0.001261/s = $4.540/h.
- 8 cores at $0.0000131/core-s = $0.377/h.
- 96 GiB at $0.00000222/GiB-s = $0.767/h.
- Total **$5.68/h**.

$/video-hour = $/h ÷ (video-h per container-h):

| Mode | Video-h / h | $/video-hour |
|---|---|---|
| Sequential (WS-D style) | 0.116 | 5.68 / 0.116 = **$49** |
| Concurrency 4, measured (straggler-bound) | 0.245 | 5.68 / 0.245 = **$23** |
| Concurrency 6–8, GPU-bound projection | ≈0.40 | 5.68 / 0.40 = **≈$14** |
| Same, with 48 GiB memory | ≈0.40 | 5.30 / 0.40 = **≈$13** |

- Add ≈$0.27–0.34 per container start: 169–213 s of load at $5.68/h.
- Episodes here are ≤47 s of 1080p HomER (1-view) and 22 s of 640×480 ALOHA (3-view). Cost
  scales with decoded resolution and caption frame count.

## Spend for W3-A

- Run 1: ≈2,180 s container (36.3 GPU-min) ≈ **$3.44**.
- Run 2: ≈485 s (8.1 GPU-min) ≈ **$0.77**.
- CPU staging calls: <$0.02.
- **Total ≈ $4.2** of the $5 cap (44.4 H200-minutes).
- `scripts/spend.py --app flora-reference` read $5.23 at 07:17 UTC, against $2.26 before W3-A.
  Billing covers only complete hours, so the last ≈$1.2 had not posted yet.

## Changes after the runs

- `ProcessorClones` became a lease pool reused across batches. The runs used one clone set per
  worker thread per batch, with the same semantics and 25 s of clone time per batch. Covered by
  unit and nvlink tests; not yet re-run on Modal.
- `generate_output` hash events (`FLORA_OBSERVE`) and `determinism` flags were added for run 2.
