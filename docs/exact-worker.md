# S1: the fast `exact` worker, and its identity proof

Status: measured 2026-10-21 on Modal, one H200 container, ephemeral app `flora-reference`, run
`s1-20261021T104132Z-d75244fb`. The image is `img_hfref` after Fix-images (torch 2.11.0+cu128).
This page holds numbers only. Full results are on `flora-golden:/reference_runs/<run_id>-{fast,nvidia}/`.

## What changed

**NVIDIA's own entry point (review2-wsd M1).**
- Each episode runs `api.DataEnginePipeline().annotate(..., include_audit=True)` unchanged (`stages/reference.annotate_nvidia`). That covers NVIDIA's `_inputs`, `pipeline.annotate` and its interval check.
- Two module attributes are swapped at runtime, in our process only (`install_nvidia_entry`):
  - `pipeline.Session` → a factory that returns the episode's view of the resident `CachedSession`. It is bound per thread through a `ContextVar`.
  - `api._REQUEST_LOCK` → a no-op lock, only when concurrency is above 1. The GPU lock then serializes every model call instead.
- The 30 re-expressed lines of `pipeline.annotate` are gone, along with the verbatim NVIDIA error strings and literals. No re-expressed orchestration is left.
- `Patcher.restore()` gives back NVIDIA's original attributes, and the worker checks each one by identity.

**One `exact` profile (`ExactProfile`).** Every proven speedup is on by default and can be switched off with `ReferenceWorker(disable="...")`. Each is recorded in `load_report.profile` and in batch provenance.

| Toggle | What it is |
|---|---|
| `decode_once`, `transport_pool`, `png_inmem`, `geometry_pool` | OPT-1 accel patches. Each episode thread runs inside `ActivePatches.episode()` |
| `captioner_bf16` | OPT-2 bit-identical BF16 Captioner (`captioner_load.install`) |
| `linked_layout` | W3-B no-copy link farm. Off means the WS-D copy layout |
| `prefetch` | W3-B t=0 16-reader read of the base and Captioner weights |
| `overlap_hash` | W3-B: NVIDIA's FP32 Captioner manifest hash starts as soon as TempSeg's own `validate_manifest` returns. Only used with `captioner_bf16` off |
| `concurrency=4` | W3-A episode threads under one GPU lock |
| `determinism="nvidia"` or `"deterministic"` | D-1 opt-in mode. `apply()` runs first in `@enter`, before CUDA, and `verify()` runs before every save. `nvidia` changes nothing |

**Review2-wsd fixes.**
- **M5, receipts.** `check_prep_receipt` replaces the old `ok` + revision check. It requires:
  - a passing receipt with all checks ok;
  - the pin record equal to `pins.py`;
  - `plan_digest` equal to the pinned one;
  - exactly the receipt's file set on disk, with the recorded sizes;
  - `pins.py`'s expected digests (adapter, NVIDIA manifest sha256), derived from the receipt's own sha256 table;
  - the sha256 of every small file (configs, tokenizer, chat template);
  - for schema 2 receipts, `pin_digest` and size/mtime fingerprints. The inode is recorded but not compared: a Modal Volume assigns new inode numbers on every container mount, so comparing it rejected every receipt with `fingerprint changed (ino)` (2026-10-22).

  **The Volume's receipts are still schema 1**, which has no fingerprints. So the worker re-hashes all 19.3 GB of the TempSeg base against its receipt in the background (41 s, overlapped). The base is the only snapshot with no NVIDIA manifest. Re-running `flora prepare` would upgrade the receipts to schema 2. I did not do that, because it rewrites the shared weights Volume.
- **OPT-2 shard check.** `captioner_load`'s install-time check is fast and does not catch corrupted shard bodies. The worker therefore also runs `verify_derived(full=True)` over all 54.7 GB, in the background during the TempSeg load. It is not ready until that passes. *This was added after the measured run; its cost is not measured yet* (≈25 s of I/O, overlapped).
- **M3, runtime bound.** The `os._exit` watchdog is gone. Runtime is bounded by Modal's `startup_timeout` plus the per-call `timeout`: 8 + 20 = 28 min in S1, and 5 + 25 = 30 min since P1a-5s (`--timeout-min` lowers it per run). A timed-out input fails; it is not retried as a crash. The script stops submitting work against its own deadlines and never starts a fresh container just to report (L7).
- **M4, exceptions.** Errors cross to the client as `NvidiaError` / `NvidiaInputError`. These are `RuntimeError` subclasses that carry the original qualified type name and message, and they pickle by value.
- **L1, L4, L6.**
  - Client-generated `new_run_id()` (UTC time plus uuid4 hex).
  - `persist_result` refuses to overwrite an existing result, and writes are whole-file via `.partial` + rename.
  - `golden.commit()` is retried once. On failure the result is kept, with `stored_at=None` plus an error.
  - The worker refuses GPUs under 120 GiB.
- **L3.** No `ephemeral_disk`: the link farm copies only a few KB of hub metadata.
- **L2.** The docstring now says mtimes are not required. Hub logs "outdated metadata" and still resolves offline.

**ShardWorker.** S1 left `ReferenceWorker.run_shard` as a `NotImplementedError`. P1a-5s implements it (next section).

## P1a-5s: the SegcapWorker and the fully wired worker

Status: measured 2026-10-21, one H200 container (`cpu=8`, 160 GiB, $6.20/h), ephemeral `flora-reference`, run `p1a5s-20261021T155655Z-ba42121d`. Results are on `flora-golden:/reference_runs/p1a5s-…-{fast,nvidia}/`.

### Contract (`workers_impl/segcap.py`)

`SegcapShardRunner` implements the `workers.py` contract, steps 0–6. It has the same structure as `RewardShardRunner` and reuses its part sink, index-receipt parser and stream-copy `VideoResolver`:

- `store.refresh()` runs first. Then:
  - step 0: a superseded token returns `None` and writes nothing;
  - step 1: the attempt marker, with token and call id;
  - step 2: episodes that already have both a `segment` and a `caption` record are skipped;
  - step 3: upstream poison becomes `poison_pill` errors.
- Step 4, inputs from each index receipt:
  - camera roles become local files; ranges are stream-copy cut;
  - NVIDIA's `profile` and `yam32`;
  - NVIDIA's `instruction`, from the receipt's instruction source. `fixed` needs `FLORA_INSTRUCTION_TEXT`.
- Step 4, one `FastExactEngine` batch (`annotate_batch`):
  - **Each episode is flushed as soon as it finishes, in its own thread.** Its two records and one audit line go to the store as whole files. The audit line holds NVIDIA's full result and the runtime provenance.
  - NVIDIA `InputError` / `ValueError` becomes `nv_failure`; anything else becomes `infra`. There is one `EpisodeError` per stage, and the container never crashes.
- Step 5: SIGINT or `preempt()` (`@modal.exit`) flushes, then `mark_preempted`. Any other exception does `mark_exception`, then re-raises.
- Step 6: the receipt is written last, with the earlier parts plus this container's own.

Records:
- `certified` follows the envelope verdict (D13). `envelope=ignore` certifies regardless.
- `provenance.engine` is `nvidia-hf`, or `nvidia-hf+deterministic` in D-1 mode, which is a different reference.
- `provenance.source` holds the camera URIs and ranges; `models` holds the four pins.
- The rest of the provenance is in the audit line, under `runtime`:
  - exact profile, `accel` patches, `gpu_thread`, determinism record;
  - BF16 receipt provenance, image fingerprint, `nv.audit_report()`;
  - `timing_dependent_calls`.

**Conformance:**
- `tests/conformance/registry.py` registers `segcap-worker`: the real runner and the real `FastExactEngine` composite over the fake NVIDIA family of `test_reference_concurrency`, with gpu_thread, GPU lock and entry swap, 2 episode threads. **All 10 checks pass.**
- Unit tests: `tests/unit/test_segcap_worker.py` and `tests/unit/test_segcap_worker_identity.py`.

### Wiring (`modal_app/reference_app.py`, the class the factory decorates)

- **Fix-X / review 4 M5.** `captioner_load.install(nv, background=True)` runs before the TempSeg load. Its default full hash (54.7 GB) is the only hash and runs overlapped; `Session.load('captioner')` joins it. The separate `verify_derived` thread is gone, and `load_report.captioner_bf16` records `provenance()`. Measured `@enter`: **78 s**, against S1's 103 s.
- **`gpu_thread`.** It is an `ExactProfile` toggle, `gpu_thread`, on by default, installed exactly once with its own `Patcher`, as the innermost layer. The composite is therefore `locked(pinned(generate))`:
  - the GPU lock is taken in the episode thread, so lock wait/hold metrics stay meaningful;
  - the call itself runs on `flora-gpu`.
  - `docs/perf-1.md`'s "lock always taken on the GPU thread" and its single-`Patcher` recipe are wrong (review 4 M3).
  - `FastExactEngine.annotate` refuses to run if the toggle is on and `generate` is not on that thread. Provenance goes into `load_report`, batch, persisted results and audit lines.
  - Measured: 41/41 calls on the GPU thread, 0 inline.
- **Per-thread `accel.episode()`.** It is entered inside each episode thread (`annotate_batch`).
- **`nv.audit()`.** It runs before every batch, before each episode (`on_start`) and after it, before its records are written (review 4 L2). It fails closed.
- **Determinism (review 4 M2).**
  - The worker reads `FLORA_DETERMINISM` (set by the app factory); the `determinism` parameter now defaults to "". A parameter that disagrees with the env is refused. `apply()` still runs first in `@enter`.
  - `stages.reference.identity_summary` / `compare_runs` raise `ModeMismatch` on arms of different modes, and `validate_shard` refuses them.
  - Hashing the mode into the config is Fix-W2's.
- **Identity logging.**
  - The GPU lock prints `FLORA_OBSERVE generate_output` after every call (`emit_output=True`).
  - `annotate_nvidia` keeps the raw `FLORA_RUNTIME` / `FLORA_OBSERVE` lines and stores `flora_reference.identity_calls` (`accel.identity.call_records`). Measured: output logged for 41/41 calls.
- **`accel.identity` (review 4 M1).** It fails closed:
  - it reads persisted `model_calls` (`input_digest`);
  - it raises on unrecognised records;
  - `ok` requires at least one paired call, and a missing episode is a violation;
  - it reports `coverage`.
- **`identity_summary`.** It reports `coverage`, `complete_coverage` and `timing_dependent_calls`.
- **Review 4 M4.** `model_calls` records `elapsed_s` and flags `timing_dependent` calls: over 150 s, or incomplete below budget, near NVIDIA's `max_time=300`. The worker prints `FLORA_WARN` for them. **0 such calls in this run.**
- **Review 4 L1.** `_restore_nvidia` also checks every `accel` registry target, including the vendor ones.
- `run_shard(job_id, shard_id, config_hash, token="")` builds the runner over `store_from_env()` on first use. `@modal.exit` calls `runner.preempt()`.
- `validate_shard` runs the production runner over a container-local store whose index receipts point at the golden files, then NVIDIA unmodified.

### Identity: S1's gap, now closed for 3 of the 4 episodes

The NVIDIA unmodified arm ran in the same container after `_restore_nvidia`, with a fresh `Session` per episode, sequentially, in `nvidia` mode. It was compared call by call with `accel.identity.compare_runs` and with the stage gate:

| Episode | Calls | vs this run's fast worker (gpu_thread) | vs S1's fast results |
|---|---|---|---|
| `human/egodex_make_sandwich/episode_000002` | 12 | strict 12/12; all 12 paired, outputs identical | inputs strict 12/12; output differed at call 2 (same input), so 3 gated |
| `single_view_robot/wgo/robointer_droid_000010` | 4 | strict 4/4; 4 paired; result identical | strict 4/4; 4 gated; identical |
| `yam/aloha_static_towel/episode_000000` | 4 | strict 4/4; 4 paired; result identical | strict 4/4; 4 gated; identical |
| `dex/g1_dex3_toasted_bread/episode_000034` | — | **not run** (deadline) | — |

**Verdict.**
- Against the new fully wired worker: **0 violations, 20/20 calls paired, coverage 1.0, conclusive.** All three full NVIDIA results are identical.
- Against S1's composite: 0 violations, but 11/20 calls gated (coverage 0.55, not conclusive). Every one of the 20 input digests is still identical (strict).
- S1 plus this run: **5 of 6 golden episodes are now proven in hardware.** All pools are covered except dex, which still rests on the CPU twin (`tests/nvlink/test_reference_entry.py`).
- **Dex was skipped** because the conservative deadline estimate (90 s + 14 s per video-s) did not fit after the fast shard. Finishing it needs a dex-only unmodified run (≈$0.8). It did not fit the $3 cap with billing lag.
- Review 4 M6: the next identity proof should run in `--determinism deterministic`, so that output divergence does not cut coverage.

### Throughput of the fully wired worker (c=4, gpu_thread on)

The shard ran 6 golden episodes, 80.97 video-s, in a single short-lived container (cold graph cache):

| | This run | S1 (no gpu_thread) | Perf-1 projection |
|---|---|---|---|
| Batch wall | **519 s** | 566 s | — |
| Video-h per container-h | **0.156** | 0.143 | ≈ 0.23 |
| Generation GPU-s per video-s | **5.04** | 6.35 | ≈ 3.9 (TempSeg warm, Captioner cold) |
| TempSeg per call | 18.9 s (10 calls, 0.27 s/token) | 54–62 s for 142 tokens (≈0.41 s/token) | 0.067 s/token warm |
| Captioner per call | 7.1 s (31 calls, 0.45 s/output token incl. prefill) | 11.5 s for 21 tokens | cold: ≈ 0.4 s/token + ≈3 s prefill |
| $/video-h at $6.20/h (160 GiB) | **$39.7** | $43 | — |
| $/video-h at $5.68/h (96 GiB default) | **$36.4** | $40 | $25 |

**Reading.**
- `gpu_thread` works: TempSeg dropped from ≈0.41 to 0.27 s per token. Calls of later episodes reuse the shapes of earlier ones; the first call was still 37 s for 44 tokens.
- The Captioner stays cold, as Perf-1 predicted. Its prompt lengths differ on every call, so in a short container almost every decode step builds a graph.
- Two artifacts of this validation run make the number pessimistic. Production never does either:
  - ≈115 s passed from `@enter` ready to the first `generate`;
  - the first ≈3.5 min of the batch overlapped the background FP32 Captioner prefetch and hash that the unmodified arm needed (109 GB read plus `manifest_full_verification`, done at 16:02:11).
- Projection, not measured: a long-lived worker should approach Perf-1's ≈3.9 GPU-s per video-s.

### Proposed `planner/rates.json` value (not applied)

| Entry (`exact`, `segcap`, cameras 1, concurrency 4) | Now | Proposal |
|---|---|---|
| container-s per video-s | 4.08, "measured" (W3-A, cu130 image) | **6.4, status `measured`, ±35%** (519 s / 80.97 video-s, this image, gpu_thread on, short-lived container). Re-measure after a long-lived run. Perf-1's 4.35 stays a projection |
| concurrency 8 | 2.5, projected | Keep as projected. It is only plausible with gpu_thread and a long-lived container with a warm Captioner |

That gives ≈$36/video-h at the default $5.68/h. W3-A's $23 belongs to the cu130 image.

### Spend (P1a-5s)

| Item | Cost |
|---|---|
| One H200 container, ≈20 min (15:57–16:17 UTC) at $6.20/h | ≈ $2.05 |
| `scripts/spend.py --app flora-reference` | $8.33 before, $8.63 shortly after (+$0.30 so far; billing lags ≈1 h) |

**≈ $2.1 of the $3 cap.** The run's local summary was lost:
- `validate_shard` crashed in its last step, because `RUNS_DIR` is a `PurePosixPath` (fixed);
- the persisted results above were downloaded from `flora-golden`;
- they were re-scored with this branch's `accel.identity` / `identity_summary`.

```bash
uv run python scripts/reference_smoke.py --validate --memory-gib 160 --deadline-min 19 \
    --timeout-min 22 [--golden ID ...] [--nvidia ID ...] [--determinism deterministic] \
    --summary p1a5s.json
```

The script exits non-zero if any comparison has 0 paired calls, a violation, or a missing episode.

## Identity proof

Method (`ReferenceWorker.identity_proof`, one container):
1. **(b) Fast composite.** All swaps on, concurrency 4, on 6 golden episodes. The dataset task text is NVIDIA's `instruction`; it is blank where a dataset has none.
2. Restore every swap and release the models.
3. **(a) NVIDIA unmodified.** No swap at all, NVIDIA's own fresh `Session()` per episode (so TempSeg and the FP32 Captioner load per episode), sequential.

The two arms are compared call by call, in order, on NVIDIA's logged `input_tensor_digest`. The gate fails if any call's input differs while every earlier call agreed on both inputs and outputs. Calls after a first *output* difference are counted but not gated. In this run there was no such difference.

| Episode | Pool | Calls (a / b) | Inputs identical (strict) | Same input, different output | Full NVIDIA result identical |
|---|---|---|---|---|---|
| `human/egodex_add_remove_lid/episode_000014` (1080p AV1, instruction) | human | 12 / 12 | **yes, 12/12** | 0 | yes |
| `yam/aloha_static_cups_open/episode_000000` (3-view AV1) | yam | 3 / 3 | **yes, 3/3** | 0 | yes |
| `human/egodex_make_sandwich`, `single_view_robot/wgo/robointer_droid_000010`, `yam/aloha_static_towel`, `dex/g1_dex3_toasted_bread/episode_000034` | | — / 4–12 | not compared | | |

**Verdict:** the gate passes. 15 of 15 compared model calls have identical inputs, and the full results of both compared episodes are identical too. All 6 episodes ran on the fast side without errors (41 calls).

**Gap:** 4 of 6 episodes did not get the unmodified arm. The soft deadline (16 min from the call start) skipped them, because the fast batch took 580 s rather than the ≈200 s I had estimated (see the next section). Dex and single-view robot are therefore covered only by the CPU proof below.

Their fast results are stored. To finish the comparison:
- run the unmodified arm alone for those 4 episodes, at an estimated ≈$1–1.7;
- then run `identity_from_volume`.

**Output agreement given identical inputs:** 0 of 15 differed in this run. W3-A had seen output differences on byte-identical inputs. D-1 traced those to cuDNN SDPA. This run's 15 pairs happened to agree. That is not evidence of determinism; use `determinism="deterministic"` when oracle == oracle must hold.

**CPU twin (`tests/nvlink/test_reference_entry.py`).** NVIDIA's real `annotate` and real `models.generate` (its digest logging), the real Flora processors, synthetic H.264 and AV1 videos and a deterministic fake model. Cases: 1-view human with and without an instruction, 3-view YAM, 3-view Dex. Unmodified (fresh sessions, sequential) against the full composite (lock, entry swap, all accel patches, 3 threads): strict input identity and full-result identity on every case. NVIDIA's own `InputError`s pass through the composite unchanged.

## Startup (`@modal.enter`, BF16 Captioner, link farm, prefetch)

| Step | Ends at | Took |
|---|---|---|
| Link farm + receipt checks, base full re-hash started in the background | 6.9 s | 6.9 s |
| Prefetch started (15 files, 68.9 GiB, 16 readers; done after 34 s at 2.2 GB/s) | 10.2 s | 3.3 s |
| NVIDIA code linked, `models.json` == pins, OPT-2 install with its fast receipt check (includes the torch/CUDA import) | 49.8 s | 39.6 s |
| TempSeg `Session.load` (17.9 GiB on the GPU) | 90.8 s | 41.0 s |
| Captioner `Session.load` from the BF16 copy, out of the page cache (68.8 GiB allocated) | 102.8 s | 12.0 s |
| Base re-hash (19.3 GB, 41 s, overlapped) and patches installed | 102.9 s | ≈0 |
| **Total** | | **102.9 s** |

The same step took **251 s** in WS-D and **169–213 s** in W3-A. Client submit to worker ready was ≈126 s, including scheduling. At $5.68/h a cold start costs ≈$0.16. That does not include the new full BF16 hash, which is not measured yet.

## Throughput and cost

Container settings for this run: `cpu=8`, **160 GiB** (so the FP32 Captioner of arm (a) stays in the page cache), H200. That costs **$6.20/h**: $4.540 H200 + $0.377 for 8 cores + $1.279 for 160 GiB. The default 96 GiB worker costs $5.68/h.

| Arm | Episodes (video s) | Wall s | Video-h per GPU-h | $/video-h at $6.20/h | at $5.68/h |
|---|---|---|---|---|---|
| Fast composite, c=4 | 6 (81.0) | 566 | **0.143** | $43 | **$40** |
| NVIDIA unmodified, sequential, fresh `Session` per episode | 2 (22.6) | 389 | 0.058 | $107 | $98 |

Earlier baselines (W3-A, previous image): WS-D sequential $49, W3-A c=4 **$23** (0.245 video-h/GPU-h).

**Why the fast worker is not faster than W3-A here: model generation got about 3× slower on this image, and it now dominates.**

- The GPU lock was held for 520 s of the 566 s batch, which is **92% busy**. 40 of the 41 calls waited for it, 1,318 s in total.
- CPU was not the limit: 2.5 busy cores of 8, and peak RSS was 25.9 GiB.
- The same input generates in the same time in both arms, so the slowdown is not caused by the patches or the lock. Some pairs:

  | Model | Input tokens | Output tokens | Arm (a) | Arm (b) |
  |---|---|---|---|---|
  | TempSeg | 1,835 | 142 | 54.4 s | 62.0 s |
  | Captioner | 5,669 | 21 | 11.5 s | 11.6 s |

- Throughput: TempSeg ran at 2.7 output tok/s and the Captioner at 2.0 tok/s. Generation cost 6.3 GPU-s per video-s in (b) and 8.5 in (a), against **1.9** in W3-A.
- The likely cause is the Fix-images switch to torch 2.11.0+**cu128** (W3-A ran before it). The GDN layers still use the torch fallback (no `fla`). This is not diagnosed.
- **Next step:** a short timing A/B of `generate` on cu128 against the previous image, before S2 relies on these rates.

Since generation is the bound, OPT-1's CPU savings mostly show up as lock queueing. Still, per-episode CPU time is down. Example: `aloha_static_cups_open` took 105 s under contention in (b) against 111 s in (a), where (a)'s figure includes its fresh model loads.

## Spend

| Item | Cost |
|---|---|
| One H200 container, ≈1,150 s at $6.20/h | ≈$1.98 |
| 4 CPU `offline_check` runs | <$0.03 |
| `scripts/spend.py --app flora-reference` | $6.44 before, $8.32 at 11:01 UTC: **+$1.88 billed** (billing lags ≈1 h) |

Total ≈ **$2.0 of the $3 cap**.

## Re-running

```bash
uv run python scripts/reference_smoke.py --offline-check
uv run python scripts/reference_smoke.py --identity --memory-gib 160 --summary s1.json \
    [--golden pool/dataset/slug ...] [--deadline-min 16] [--disable png_inmem,...] \
    [--determinism deterministic]
```
