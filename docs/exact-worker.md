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
  - for schema 2 receipts, `pin_digest` and size/mtime/inode fingerprints.

  **The Volume's receipts are still schema 1**, which has no fingerprints. So the worker re-hashes all 19.3 GB of the TempSeg base against its receipt in the background (41 s, overlapped). The base is the only snapshot with no NVIDIA manifest. Re-running `flora prepare` would upgrade the receipts to schema 2. I did not do that, because it rewrites the shared weights Volume.
- **OPT-2 shard check.** `captioner_load`'s install-time check is fast and does not catch corrupted shard bodies. The worker therefore also runs `verify_derived(full=True)` over all 54.7 GB, in the background during the TempSeg load. It is not ready until that passes. *This was added after the measured run; its cost is not measured yet* (≈25 s of I/O, overlapped).
- **M3, runtime bound.** The `os._exit` watchdog is gone. Runtime is bounded by Modal's `startup_timeout` (8 min) plus the per-call `timeout` (20 min), so one container lasts at most 28 min. A timed-out input fails; it is not retried as a crash. The script stops submitting work against its own deadlines and never starts a fresh container just to report (L7).
- **M4, exceptions.** Errors cross to the client as `NvidiaError` / `NvidiaInputError`. These are `RuntimeError` subclasses that carry the original qualified type name and message, and they pickle by value.
- **L1, L4, L6.**
  - Client-generated `new_run_id()` (UTC time plus uuid4 hex).
  - `persist_result` refuses to overwrite an existing result, and writes are whole-file via `.partial` + rename.
  - `golden.commit()` is retried once. On failure the result is kept, with `stored_at=None` plus an error.
  - The worker refuses GPUs under 120 GiB.
- **L3.** No `ephemeral_disk`: the link farm copies only a few KB of hub metadata.
- **L2.** The docstring now says mtimes are not required. Hub logs "outdated metadata" and still resolves offline.

**ShardWorker.** `ReferenceWorker.run_shard` is a documented `NotImplementedError` (TODO P1a-5). The reason: it needs the job-store client inside the container and episode input resolution from `EpisodeRef.index_ref`, which P1a-5 owns. The record mapping it needs is in place: `stages.reference.stage_records` produces certified `segment` and `caption` `StageRecord`s from NVIDIA's result.

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
