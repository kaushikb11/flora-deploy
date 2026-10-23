# D-1: why NVIDIA's reference is not run-to-run deterministic on GPU, and the opt-in fix

Status: measured 2026-10-21 with `flora-determinism-lab` (ephemeral) on one L40S container and
one H200 container. This page holds numbers only. Texts and logits were never written anywhere.

## Answer

- **Root cause: PyTorch SDPA's cuDNN attention backend, which torch 2.11 selects on Hopper
  (sm90).** NVIDIA loads TempSeg (and the Captioner) with `attn_implementation="sdpa"`.
  - On the H200 the decode steps of that kernel are not bit-reproducible: identical inputs
    give logits that differ by one or two BF16 ulps (0.125–0.25) at a random decode step.
  - Usually the argmax survives, because top-2 margins are 14–22 logits. Occasionally it flips a
    token, and the rest of the generation and everything downstream follow. That is the `exact`
    concurrency lab's "identical input digest, different output, even sequentially".
- **The L40S (sm89) is fully deterministic with NVIDIA's settings as they are.** cuDNN SDPA is
  not used there.
- **The minimal fix is disabling only the cuDNN SDPA backend**
  (`torch.backends.cuda.enable_cudnn_sdp(False)`; flash, mem-efficient and math stay allowed).
  `torch.use_deterministic_algorithms(True)` alone has the same effect: torch then skips cuDNN
  attention. Both give bit-identical logits.
- **Cost: none measurable.** Per-call `generate` time on the H200 changed by −10% to +13%. That
  is within noise, and `generate` is ~22% of `exact` wall time.
- **Catch: this is a different reference, not the same bits.** On the H200, deterministic mode
  differs from the cuDNN path from decode step 0 (the prefill already runs through a different
  attention kernel). For one of the two inputs tested, the text also differed from **every** H200
  default run. So it can be offered only as an explicit opt-in that is recorded in provenance.
  It must never be compared bit-for-bit against default-mode references.

## Ranked root-cause candidates (with evidence)

| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| 1 | **SDPA cuDNN backend on sm90** (text full-attention layers; decode against the KV cache) | **Confirmed** | H200 baseline: 5 and 4 distinct logit sequences in 6 runs each. Disabling only cuDNN SDPA: 1 sequence in 6 runs. L40S (no cuDNN SDPA): 1 sequence in 8 runs, every mode |
| 2 | `max_time=300` in `models.generate` (wall-clock stop) | **Cannot cause it** | TempSeg calls take 8–15 s, against the 300 s budget. The timer (`MaxTimeCriteria`, which starts inside `generate`, and the GPU lock is taken before that) would end generation without EOS, and NVIDIA then raises `InputErrorForModel` ("incomplete"). The concurrency lab's divergent calls were all `complete`. Every lab call had the same token count in every run |
| 3 | Uninitialized-memory reads, or kernel choice that depends on allocator or address | No | `dirty` mode (L40S): the caching allocator was filled with up to 12 GiB of random bytes, and a random-sized tensor shifted later addresses. Logits were bit-identical to baseline over 8 runs. `det` mode NaN-fills uninitialized memory (`fill_uninitialized_memory`) and was also bit-identical to baseline on L40S |
| 4 | GDN (gated delta net) kernels | No | `fla` and `causal_conv1d` are not installed, so GDN runs on transformers' torch fallback (`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`): plain matmuls, a 64-wide inner-dim `cumsum`, elementwise ops, no atomics. Prefill hashes of all 98 hooked modules (vision tower, every decoder layer and its `linear_attn`/`self_attn`/`mlp`, `lm_head`) were identical across repeats on the H200 |
| 5 | cuBLAS workspace / split-K, TF32 | No | `matmul.allow_tf32=False` (NVIDIA's default stays). The lab container fixed `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which is the default workspace size on Hopper, and H200 baseline was still non-deterministic. Turning off cuDNN SDPA alone fixed it, with cuBLAS untouched. `cudnn.allow_tf32` only affects the vision Conv3d patch embed, which is part of the prefill, and the prefill is reproducible |
| 6 | CPU side: threads, dict ordering, tokenizer races | No | NVIDIA's `input_tensor_digest` was identical for every call (`5e17cdf5…`, `a737504b…`). The concurrency lab's CPU tests show 1, 4 and ≥8 torch threads give byte-identical tensors. Greedy decoding: `do_sample=False, num_beams=1` |

Decoding is greedy (`do_sample=False`, `num_beams=1`, `use_cache=True`) under
`torch.autocast(bfloat16)`. TempSeg has an FP32 LoRA on a BF16 base, and the GDN parameters
`A_log` and `dt_bias` stay in FP32. Nothing in NVIDIA's code is stochastic.

## Experiments

Inputs: NVIDIA's `pipeline.segment` on `homer_50` (a HomER sample, human profile), stopped after the
first 2 TempSeg windows. Their exact processor outputs were replayed into NVIDIA's unmodified
`models.generate`:

- window 0: 1,810 prompt tokens, 104 output tokens;
- window 1: 141 output tokens.

Every call records:

- NVIDIA's `generation_start/_end` events;
- the text digest;
- a sha256 of the `lm_head` output at every decode step (a forward hook);
- at the first step that differs from baseline run 0: the max |Δlogit|, whether the argmax is
  the same, and run 0's top-2 margin.

In the table, a cell with one value covers both windows. A cell with two values gives window 0,
then window 1.

| GPU | Mode | Runs per window | Distinct texts | Distinct logit sequences | First differing step vs baseline r0 | Text = baseline r0 | Mean s per call |
|---|---|---|---|---|---|---|---|
| L40S | baseline | 8 | 1 | 1 | — | yes | 7.73 |
| L40S | `dirty` | 8 | 1 | 1 | — (bit-identical) | yes | 7.76 |
| L40S | `det` (deterministic algorithms, warn_only=False) | 8 | 1 | 1 | — (bit-identical) | yes | 8.47 |
| L40S | `det_warn` | 1 | 1 | 1 | — | yes | — (no warnings raised) |
| L40S | `sdpa_math` | 8 | 1 | 1 | step 0 (Δ 0.41) | **no** (w0) | — |
| H200 | **baseline** | 6 | 1 | **5 and 4** | steps 7, 46, 46, 81, 81 (w0); 72, 72, 138 (w1); Δ 0.125–0.25, argmax kept | yes | 10.6 / 12.0 (the first call is warm-up; 10.0 / 11.4 without it) |
| H200 | prefill only, hooks on 98 modules | 4 | — | 1 (0 modules differ) | — | — | — |
| H200 | **`det`** | 6 | 1 | **1** | step 0 (Δ 0.56 / 0.50, argmax kept) | **w0 no**, w1 yes | 9.9 / 13.1 |
| H200 | prefill only, `det` | 4 | — | 1 | — | — | — |
| H200 | **`sdpa_nocudnn`** (only cuDNN SDPA off) | 6 (w0; deadline) | 1 | **1**, identical to `det` | step 0 (Δ 0.56) | w0 no | 9.1 |

Readings:

- **Prefill is reproducible on the H200. Decode is not.** The divergence starts at a random
  decode step, and different runs land on different first steps. That fits an attention kernel
  whose decode path (one query against a growing KV cache) reduces in a schedule-dependent
  order. It does not fit a stateful bug: a run that differs at step 46 can agree again with
  another run (same digest).
- `det` and `sdpa_nocudnn` produce **the same** logit sequence (`57f15357…` for window 0). So
  the only thing `use_deterministic_algorithms` changed on the H200 was steering attention off
  cuDNN. No other op was flagged; `det_warn` raised no warnings, and strict `det` raised no
  errors.
- **Deterministic mode changes NVIDIA's H200 output distribution.** Window 0's text under
  `det` is `bb4c380d…` in all runs. That equals the L40S baseline text. All 6 H200 default runs
  gave `cc035562…`, the same text the L40S produces with the math backend. Window 1's text is
  the same in every mode and on both GPUs.
- The concurrency lab's "run 2 fully identical" fits as well: most divergences change logits by one ulp and are
  absorbed by the large top-2 margins. Over 70+ calls, a few flips happen.
- Not run, because the budget did not allow it:
  - `cudnn_notf32`, `threads1` and `det_nofill` (the L40S run was cut off by a client
    disconnect after `sdpa_math`);
  - the Captioner, which uses the same `sdpa` and the same Qwen3.5 code, so it is expected to
    behave the same;
  - a strict run on H200 window 1 for `sdpa_nocudnn`.

## Recommendation

**Yes: `exact` can offer an opt-in `deterministic` mode.** It keeps NVIDIA's model inputs
byte-identical and makes outputs reproducible run to run on the H200. It is implemented as
`flora_deploy.accel.determinism`:

- `container_env("deterministic")` → `{"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}`. This goes in the
  Modal `env=`, because it must be set before CUDA initializes.
- `apply("deterministic")`, before NVIDIA's `Session.load`, sets:
  - `torch.backends.cuda.enable_cudnn_sdp(False)` (the fix itself);
  - `torch.use_deterministic_algorithms(True)` (a guard: any future nondeterministic op raises
    instead of silently varying);
  - `cudnn.deterministic=True` and `cudnn.benchmark=False`.

  It leaves TF32, float32 matmul precision, autocast and every NVIDIA argument untouched, and
  returns a provenance record (schema `flora_determinism_v1`: mode, live flags, `in_effect`).
- `verify(record)` before persisting a result raises if any flag drifted.
- `apply("nvidia")` (the default) changes nothing and only records the flags.

Not wired into `stages/reference.py` or `reference_app.py` here, because another agent owns
them. Wiring:
- merge `container_env(mode)` into `ReferenceWorker`'s `env`;
- call `apply(mode)` at the top of `@modal.enter` and store the record in `load_report` and in
  every result's `flora_reference`;
- call `verify` in `persist_result`.

Cost and risk:

- **Speed:** within noise. Per H200 call it was −10% to +13%. On the L40S it was +9.5% (where
  it is not needed). End to end that is ≤3% of `exact` wall time, because `generate` is ~22%.
- **Changes vs NVIDIA-validated behavior:**
  - The model inputs and code are identical. The attention kernel differs.
  - On the H200, outputs are a different but self-consistent fixed point. They agree with the
    default path to within BF16 rounding, and here they gave the same text as the L40S default.
  - It is not bit-equal to any default H200 run, and texts can differ (1 of 2 windows here).
- **Policy:**
  - Keep `nvidia` the default for the `exact` preset.
  - Use `deterministic` for certification-set and oracle builds where "oracle == oracle" must hold. Run
    both sides in the same mode, on the same GPU type, with the same torch and cuDNN.
  - Record the mode in provenance.
  - Gate thresholds against default-mode references must still come from the measured
    run-to-run variance of the concurrency lab.
- Cross-GPU bit-equality is not promised by either mode. The L40S and H200 logits differ even
  when the texts agree.

## How to reproduce

`uv run modal run src/flora_deploy/modal_app/determinism_lab.py --gpu H200 --cpu 4 --memory-gib 40
--reps 6 --windows 2 --modes baseline,prefill,det,prefill_det,sdpa_nocudnn --deadline-min 6.5`

Load ≈62 s, capture ≈38 s, experiments ≈400 s.

## Spend

- L40S: one aborted start (a path bug, ≈3 min) and one run of ≈17 min. 8 CPU, 64 GiB.
- H200: one run, ≈8 min container. 4 CPU, 40 GiB.
- `scripts/spend.py --app flora-determinism-lab` read **$0.67** with the H200 hour not yet
  posted. The estimated total is ≈$1.6, under the $2 cap.
