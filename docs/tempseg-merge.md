# P1b-0: the FP32-merged TempSeg checkpoint

Status: built and measured 2026-10-21 on Modal (app `flora-convert`, ephemeral). It covers the
adapter `nvidia/Flora-TempSeg@114a1076` on `Qwen/Qwen3.5-9B@c2022362`, with NVIDIA's SEGCAP code
at `78a0e4ac`, torch 2.11.0 and transformers 5.8.1. This is the D3 artifact. Gate B decides
later whether we serve it.

## What NVIDIA runs (read from `batch_utils/vendor/tempseg_fp32_loader.py`, not changed)

`Session.load('tempseg')` serves the adapter **unmerged**:

1. It applies `_keep_native_gdn_fp32`, then loads the base with
   `from_pretrained(base, dtype=bf16, sdpa)`. `A_log`/`dt_bias` stay FP32.
2. It attaches the FP32 PEFT LoRA (r=32, α=64) with FP32 holders and checks that every value
   loaded exactly.
3. It wraps `generate` in `torch.autocast('cuda', bf16)`.

vLLM's LoRA path cannot reproduce this. It has no FP32 LoRA dtype and is still affected by
#52568. So D3 serves a plain merged model instead.

## Merge (`src/flora_deploy/prep/merge_tempseg.py`)

For each of the 248 adapter targets:

Merger v2:

```
W = bf16( fp32( fp64(W_base) + (α/r) · Σ_{k=0..r-1} fp64(B[:,k]) · fp64(A[k,:]) ) ),   α/r = 2.0
```

Merger v1, which produced the result and equivalence check below, computed
`fp32(W_base) + (α/r) · (fp32(B) @ fp32(A))` with a BLAS matmul. Either way the delta is added
at full precision and the sum is rounded to BF16 once at the end. PEFT's `merge_and_unload` on a
BF16 base would round twice.

### Determinism: why v2, and why its content digest can be pinned

v1 was not reproducible across machines. A BLAS `matmul` picks its blocking, SIMD kernel and
summation order according to the CPU (AVX2, AVX-512 or AMX) and the thread count. FP32 addition
is not associative, so the same inputs could give different FP32 deltas on two hosts.
Occasionally one of those differences crosses a BF16 rounding boundary, and the shard bytes,
and with them the content digest, differ. `torch.set_num_threads(1)` does not fix this, because
the single-thread kernel still depends on the CPU.

v2 (`merge_weight`) is byte-deterministic on any IEEE-754 CPU, at any thread count, for four
reasons:
1. **Exact products.** `A` and `B` are FP32, with 24-bit significands, so every product
   `B[i,k]·A[k,j]` has at most 48 significant bits and is exact in FP64 (53 bits). A fused
   multiply-add and a separate multiply then add therefore give the same result. So does
   `addcmul_` in either its scalar or SIMD form.
2. **Fixed order.** The rank sum runs as `r` element-wise FP64 accumulations in a fixed order
   (k = 0, 1, …, r-1), over row tiles of at most 2^20 elements. Each add is a single IEEE
   correctly rounded operation per element. No reduction is split across threads or SIMD lanes,
   so neither can reorder it.
3. **Single correctly rounded steps.** `×(α/r)` is exact for α/r = 2.0 and correctly rounded in
   general. The `+ W` step and the FP64 → FP32 cast are also each correctly rounded once.
4. **Integer cast.** FP32 → BF16 is torch's integer round-to-nearest-even, identical in its
   scalar and SIMD code (see `docs/captioner-bf16.md`).

v2 is also the more accurate of the two. It is not bit-identical to v1 or to PEFT's FP32
`merge_and_unload`, and the difference is not bounded by 1 BF16 ulp. Where `W + Δ` nearly
cancels, the result is tiny, so its BF16 ulp is tiny too, while the FP32 delta's rounding error
scales with the operands. There the two can differ by several ulps. Measured on CPU (review 8b,
six random 1024 × 1024 targets, r = 32, `W ~ 0.02·N(0,1)` in BF16, `A, B ~ 0.01·N(0,1)` in FP32):

| | Result |
|---|---|
| Elements where v2 ≠ PEFT (`bf16(fp32(W) + 2·(B @ A))`) | 61 of 6,291,456 (about 1e-5) |
| Largest distance per target | 1, 1, 1, 3, 1, 10 ulps |
| Worst case (seed 15) | exact −1.35079e-9, v2 −1.35333e-9, PEFT −1.28057e-9 (10 ulps apart) |
| Elements where v2 ≠ `bf16(exact FP64 sum)` | 0 |
| Elements where PEFT ≠ `bf16(exact FP64 sum)` | 61 |

So every difference is PEFT's FP32 rounding, not v2's. The honest bound is absolute, not in ulps
of the result: |v2 − PEFT| is at most the FP32 rounding error of `Δ` plus half a BF16 ulp. That
is why the merger version is 2 and a v1 receipt is not reused.

The cost is about `r` = 32 FP64 passes over the 6.92 G merged elements in cache-sized tiles.
Locally, one 4096 × 12288 target (50 M elements) takes 0.9 s on 4 threads. That projects to
about 1 min for the whole merge with 2 workers, or a few minutes on the Modal container,
compared with v1's 45 s. Either is far inside the 30-minute limit.

Because v2 is deterministic, the content digest of `derived_receipt.json` is the same for every
user's build, and `pins.DERIVED_PINS["tempseg-merged"].content_sha256` can pin it. The content
digest is the one defined for the Captioner, plus the base pin and the LoRA spec. The merge
statistics are excluded, since their FP64 sum is a threaded reduction. The digest is still
`None` because v2 has not been built. Pin it after one v2 build (`flora prepare --show-receipt`
prints it) and a re-run of the equivalence check below, which was measured on v1. Until then,
`verify_merged` recomputes the digest and checks any stored value, but has no pin to compare
against.

- **Adapter spec.** The spec is read from `adapter_config.json` and fails closed on DoRA,
  rsLoRA, bias, `fan_in_fan_out`, rank or alpha patterns, and `modules_to_save`.
- **Key mapping.**
  - Each adapter key must be `base_model.model.<module>.lora_{A,B}.weight`.
  - Every target must have both A and B.
  - The adapter modules must equal `target_modules`, using PEFT's exact-or-suffix rule.
  - Every base weight PEFT would wrap must have an adapter.
  - Shapes must be A `[r,in]`, B `[out,r]`, W `[out,in]`.
  - The counts must be 248 targets and 496 tensors, and all must be FP32.
- **Dtypes.** Nothing is hard-coded. The policy comes from transformers' `dtype_plan` with
  NVIDIA's own `_keep_native_gdn_fp32`, plus the meta-model dtypes, as in OPT-2. The result:
  - 48 tensors are FP32: `A_log` and `dt_bias`. 24 `dt_bias` are stored as BF16 in the base and
    upcast exactly.
  - The 24 `linear_attn.norm.weight` are stored as FP32 in the base and cast to BF16, exactly as
    NVIDIA's load does.
  - The 15 `mtp.*` keys are dropped by transformers on load (`_keys_to_ignore_on_load_unexpected`)
    and are copied unchanged.
- **Inputs are verified.**
  - The adapter, `adapter_config.json` and Flora's files are checked against NVIDIA's
    `SOURCE_MANIFEST.sha256`, whose own digest is pinned in `pins.py`. The adapter sha256 is
    also pinned.
  - Base shards are hashed while they are read and checked against the base's passing
    `prep_receipt.json`, which is bound to the pinned `plan_digest`.
- **Files written.**
  - Base shard names and the key-to-shard map are kept.
  - The index is rewritten with a new `total_size`.
  - The base's `config.json` is copied.
  - **Flora's** `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`,
    `chat_template.jinja`, `README.md` and `MODEL_PROVENANCE.json` are copied. The base's
    processor files are not.
  - `adapter_config.json` is not copied, because transformers would then treat the directory as
    an adapter.
  - `derived_receipt.json` is written last.
- **Memory** is bounded at 2 workers × one base shard (≤5.4 GB), plus the adapter (0.35 GB) and
  one FP32 target at a time.

A unit test merges a tiny Qwen3.5 with a real PEFT adapter. The test checks four things:

- Each merged weight is within 1 BF16 ulp of PEFT's own FP32 `merge_and_unload`, rounded once.
  That holds for the tiny model; on full-size targets the distance can reach about 10 ulps where
  the sum cancels, with v2 the more accurate (see "Determinism" above).
- Non-target tensors are untouched.
- `A_log`/`dt_bias` are FP32 and equal the base values.
- The directory loads through NVIDIA's call with no missing or unexpected keys.

### Result (merger v1)

Output goes to `flora-weights:/derived/tempseg-merged@114a1076f70eb53f4a58ac045505db3087e79347/`.

| | |
|---|---|
| Output | 19.33 GB (19,326,398,422 B), 12 files (4 shards) |
| Casts | 248 merged; 440 BF16 kept; 24 BF16→F32; 24 F32 kept; 24 F32→BF16; 15 `mtp.*` copied |
| Merged elements | 6.92 G; 97.7 % differ from the base in BF16; max \|ΔW\| 0.096, mean 1.0e-3 |
| Time | 63 s total (45 s merge, 4 s re-read); CPU container 8 cores, 32 GiB |
| Receipt sha256 | `f4a6fd272469559aa2017745025cb649d6f1da0c248730c283ef323bceae1e72` |
| Merger | v1, module sha256 `535529b9…`, git `04f8ea6e` (clean) |

## Equivalence check (`convert_app.py::tempseg_equivalence`, one L40S, network blocked)

The check is **not** bit-identity, because merging moves the rounding. Two models were loaded one
after the other on the same GPU:

- **A**: NVIDIA's `Session.load('tempseg')` as shipped, unmerged, with FP32 LoRA and BF16
  autocast.
- **B**: the merged directory, loaded with `from_pretrained(dtype=bf16, sdpa)` after
  `_keep_native_gdn_fp32`, and run with **no** autocast, which is how it will be served.
  Variant B+ac runs the same model under autocast.

Window inputs come from NVIDIA's own `pipeline.segment` → `temporal_inputs`. Its `generate` is
wrapped, so the run stops after the first window of each episode. B builds its inputs with its
own processor, `models.processor('tempseg', merged_dir)`.

| Check | Result |
|---|---|
| Loading | B: 0 missing, 0 unexpected, 0 mismatched; 712 BF16 + 48 FP32 params. A: NVIDIA's adapter and GDN audits pass |
| `verify_gdn` on B | PASS against the merged checkpoint **and** against the base checkpoint (FP32 values unchanged) |
| Non-target parameters | 512/512 bit-identical between A (PEFT base) and B |
| Determinism floor | A vs a repeat of A: Δ = 0 |
| Processor | B's (Flora files from the merged dir) gives identical prompt ids and identical window input tensors |

Logit differences. Each cell is A vs B (A vs B+ac in brackets). Logits have mean |logit| ≈ 2.2–3.0.

| Inputs | Positions | max \|Δlogit\| | mean \|Δlogit\| | top-1 agree | greedy next token |
|---|---|---|---|---|---|
| 4 fixed chat prompts | 18–28 each | 0.31–0.74 (0.25–0.75) | 0.023–0.025 | 100 % (one prompt 96.4 % with ac) | 4/4 equal |
| 32 seeded random tokens | 32 | 0.32 (0.38) | 0.024 | 96.9 % (100 %) | equal |
| Human window, teacher-forced on A's output | 142 | 0.44 (0.41) | 0.028 | 100 % | — |
| YAM 3-view window, teacher-forced | 44 | 0.50 (0.50) | 0.033 | 100 % | — |

Greedy generation uses NVIDIA's `generate`: do_sample=False, budget 1024, stop at `<|im_end|>`.

| Window | Prompt tokens | A tokens | B tokens | Token ids | JSON text |
|---|---|---|---|---|---|
| `human/egodex_add_remove_lid/episode_000014` (1 view) | 1,811 | 142 | 142 | identical | identical |
| `yam/aloha_static_cups_open/episode_000000` (3 views) | 5,493 | 44 | 44 | identical | identical |

Reading the results:

- The merge moves logits by about 1 % of their mean magnitude: mean |Δ| ≈ 0.025, max ≤ 0.75.
  This is BF16 rounding noise.
- Running B with or without autocast makes no systematic difference.
- On real TempSeg windows, every greedy decision along A's trajectory is the same. Both windows
  produce byte-identical JSON.
- Top-1 flips happen only at near-tie positions on random or chat text.
- Two windows is a smoke check, not Gate B. Gate B needs the golden-set L3 metrics.

Load times on L40S: A takes 47.6 s and B takes 34.0 s. The whole check took 142 s, and peak GPU
memory was 19.3 GiB.

## Spend

Billing for this hour had not posted when this was written: `spend.py` still showed
`flora-convert` at $0.65, all of it from OPT-2. The figures below are estimates from wall time:

- CPU merge: about 2 min container time, ≈ $0.03.
- L40S check: about 4 min including image start, ≈ $0.15 with CPU and memory.

The estimated total is ≈ $0.2, against a cap of $2.

## Re-running

```
uv run modal run src/flora_deploy/modal_app/convert_app.py::tempseg --step merge [--force]
uv run modal run src/flora_deploy/modal_app/convert_app.py::tempseg --step equivalence
```

`merge` reuses a passing receipt of the same merger version and pins unless you pass `--force`.

## Not done here

- Wiring the merged directory into a vLLM engine or the `balanced` preset. That is Gate B
  (PLAN.md §9).
- Building merger v2, re-running the equivalence check on it, and pinning its content digest
  (`DERIVED_PINS["tempseg-merged"].content_sha256`).
