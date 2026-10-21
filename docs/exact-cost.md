# G3: cutting the `exact` preset's cost without changing any model input

Status: analysis plus a CPU-proven implementation, 2026-10-21. **No GPU or Modal time was used.**
Every projection below comes from component times already measured and recorded in
`docs/exact-worker.md` (P1a-5s), `docs/perf-1.md`, `docs/exact-concurrency.md` (W3-A) and
`docs/cold-start.md` (W3-B). It also uses the P1a-5s per-episode results (`flora-golden:/reference_runs/p1a5s-20261021T155655Z-ba42121d-fast/`).
Prices are Modal list prices as these docs use them:
- H200 container (8 cores, 96 GiB): **$5.68/h**;
- H100: $3.95/h;
- CPU: $0.0472 per core-h;
- memory: $0.008 per GiB-h.
- L40S at **$1.95/h** is an assumption: it is Modal's list price, but none of these docs records it.

The rule for every option: **every model call gets exactly the input it gets today.** No NVIDIA
file changes, every model call is still NVIDIA's batch-1 `generate`, and the rest of the
orchestration stays NVIDIA's own code.

## Baseline (measured, P1a-5s)

One H200 container, c=4, `gpu_thread` on, cold cuDNN graph cache. One 6-episode shard of 80.97 video-s.

| Quantity | Value | Source |
|---|---|---|
| Batch wall | 519 s | exact-worker.md |
| Generation GPU-s (lock held) | 408.2 s = TempSeg 188.8 s (10 calls) + Captioner 219.4 s (31 calls) | per-episode `generation` stats |
| Container-s per video-s | 519 / 80.97 = **6.41** | rates.json default (6.4) |
| $/video-h | 6.41 × $5.68 = **$36.4** | |
| GPU floor (cold) | 408.2 / 80.97 = 5.04 container-s/video-s, which is **$28.6** | |
| `@enter` | 78 s; plus ≈23 s of scheduling (S1: 126 s from submit to ready, against 103 s for `@enter`) | exact-worker.md |

**Where the 111 s of GPU idle went.** Each result's `batch.end_s − timings_s.total` gives when it started.
- All four first episodes started **54–60 s** after the batch began. Nothing ran before them, and the cause is not attributed. The run overlapped the validation-only FP32 Captioner prefetch and hash, so this is possibly an artifact; it could also be the per-episode `nv.audit` / `accel.episode()` entry under that I/O load.
- They then spent 45–60 s in NVIDIA's `_inputs` probe on 1080p video (18 s for droid, 2 s for the 3-view YAM clips).
- The first `generate` came at about 106 s. From then on the GPU was essentially saturated: 408 GPU-s ran in the remaining 413 s.

**So the idle is a per-shard *fill*, not a drain.** It is 50–111 s per shard: ≈50 s for certain, and up to 111 s if the pre-start gap is real in production.

**The measured run was already longest-first.** `validate_shard` sorts its golden episodes longest first. Production runs in manifest plan order (see lever 2).

## Options

| # | Lever | Input-identical? | Projected container-s/video-s | **$/video-h** | Complexity | Implemented |
|---|---|---|---|---|---|---|
| 0 | Today, P1a-5s as measured (longest-first order) | — | 6.41 | **$36.4** | — | — |
| 0′ | Today in production: plan order | yes | ≈7.0 (model: +9.7%) | **≈$40** | — | — |
| 2 | Longest-first order (LPT) | yes: start order only | 6.41 (recovers 0′ → 0) | **$36.4** (−9% vs 0′) | low | **yes** (`episode_order`) |
| 1 | Shard overlap, `max_inputs` 2, shared slots (+ LPT) | yes: start time only | 5.04–5.25 | **$28.6–29.8** (≈−20%) | medium | **yes** (`shard_overlap`) |
| 4a | Larger shards (3× target) instead of overlap | yes | 5.25–5.50 | $29.8–31.2 | config only | no (dominated by 1) |
| 4b | Long-lived containers (warm cuDNN cache) + 1 + 2 | yes (Perf-1: same graph key, same heuristics) | ≈3.9–4.1 at c=4; ≈2.5 at c=8 | **≈$22–23**; ≈$14 at c=8 | config + measurement | no (needs GPU measurement) |
| 3 | Split pools: TempSeg on L40S, Captioner on H100 | yes, *given the same TempSeg output*; but the GPU class changes the reference (below) | — | **$24.9–30.5** | high | core + CPU proof only, not wired |
| 5a | Memory 96 → 48 GiB | yes: no input involved | ×5.30/5.68 | $33.9; **$27.3** with 1+2 | config only | no (startup to re-check) |
| 5b | Input prefetch / per-thread resolution / decode-ahead | yes | subsumed by 1 | — | — | via 1 |

### Lever 1: overlap shards inside one container

**What changes.** `@modal.concurrent(max_inputs=2)` already delivers a second shard to a busy
container, and the factory already scales the call timeout by `max_inputs`. The runner used to refuse that second shard.

Now:
- `SegcapShardRunner(max_shards=N)` runs up to N shards. Each has its own marker, part sink, receipt and step 0–6 path.
- The engine `FastExactEngine(overlap=True)` shares its `concurrency` episode slots between them through `stages.reference.EpisodeSlots`.
- A free slot always goes to the earliest shard's next queued episode, so a later shard only fills slots a draining shard frees. Running episodes never exceed c=4, so RSS is unchanged (W3-A: +3.2 GiB per slot).

The fill (probe, decode, input resolution) of shard N+1 therefore runs while shard N keeps the
GPU busy. It is paid once per container, not once per shard.

**Math.** Container-s/video-s = (408.2 + B/K + d·408.2) / 80.97, where:
- B is the per-shard fill: 50–111 s;
- K is shards per container lifetime: ≥4 in any multi-shard job, because the 300 s scaledown keeps containers warm between inputs;
- d is the residual idle after the fill: 0–2%, since the measured GPU was ~100% busy after the fill.

That gives **5.04–5.25**, i.e. **$28.6–29.8**. Without overlap, production pays B on every shard: (408 + 50…111)/80.97 = 5.66–6.41, i.e. $32.2–36.4.

The saving is therefore $3–7 per video-h. It depends on whether the 56 s pre-start gap is real in production (measure it).

**Why inputs are identical.** Each episode still runs NVIDIA's own `api.DataEnginePipeline().annotate` alone in its thread, with its own processor clones. Only the moment it starts changes.

This is W3-A's proven setting: episode threads sharing resident models under one GPU lock. Overlap only lets two batches feed the same thread budget.

**Risks found and handled:**
1. **Volume reload during an overlapped shard.** `store.refresh()` at a shard's start would reload the Volume while another shard's NVIDIA thread has a media file open. That happens when `stage_to` names the store Volume (`factory.media_for` then puts the media under the store mount). A reload fails, or blanks the mount, while files are open.
   - The runner now skips the refresh in exactly that case: another shard in flight, and media under the store's mount. It counts `skipped_refreshes`.
   - That shard then sees the container's last view, which the store contract already tolerates (eventually consistent).
   - **The same hazard exists for `RewardShardRunner` with `max_inputs > 1`**, which refreshes unconditionally. That belongs to the reward owner.
2. **Preempt ordering.** A cancelled shard's thread used to be able to write its `mark_exception` before `preempt()` reached it. Its start then counted toward D15.
   - `preempt()` now marks every shard in flight *first*, then cancels the queued episodes.
   - The new test caught this race in about 1 run in 3.
3. **Lock statistics** are reset only by the first batch in flight (`reset_lock`). A newcomer never clears a running batch's cancel.

### Lever 2: longest-episode-first ordering (LPT)

**Math.** I fitted a discrete-event model (c slots, one GPU server) to the measured shard. It reproduces the 463 s makespan with CPU phases of 0.74 × (non-GPU time measured under contention).
- Over all 720 start orders of these 6 episodes, the mean makespan is 512 s (range 438–594 s); LPT gives 462 s.
- Plan order is effectively a random order with respect to duration, so it costs (56+512)/(56+462) = +9.7%: **≈7.0 container-s/video-s, ≈$40**.
- LPT brings that back to the measured $36.4.
- In the warm-cache regime the same model gives 4.71 → 4.02 (−15%).
- With overlap, LPT only matters for a container's last shard, so it adds ≈0 on top of lever 1.

**Identity.** It changes start order only. Receipts keep plan order, and records are byte-identical (tests).

### Lever 3: split pools (TempSeg on L40S, Captioner on H100)

Per-model GPU time from the measured shard:
- TempSeg: 188.8/80.97 = **2.33 GPU-s/video-s**;
- Captioner: 219.4/80.97 = **2.71 GPU-s/video-s**.

Both pools are assumed to have the same fill factor, 519/408 = 1.27, with speed factors s_L (L40S vs H200) and s_H (H100 vs H200).

Pool prices:
- TempSeg pool: L40S + 8 cores + 48 GiB = 1.95 + 0.377 + 0.384 = **$2.71/h**;
- Captioner pool: H100 + 8 cores + 70 GiB = 3.95 + 0.377 + 0.56 = **$4.89/h**.

$/video-h = 2.33·1.27·s_L·2.71 + 2.71·1.27·s_H·4.89 = **8.02·s_L + 16.8·s_H**.

| Case | Calculation | $/video-h |
|---|---|---|
| s = 1.0 for both | | $24.9 |
| s_L = 1.5, s_H = 1.1 | warm decode is overhead-bound at 65–70 ms/step (Perf-1); L40S bandwidth is 0.86 TB/s against 4.8 TB/s, and its BF16 compute is ~0.37×, which hits prefill | $30.5 |
| Both pools on H100 | 2.33·1.27·1.05·4.71 + 2.71·1.27·1.05·4.89 | $32.3 |

On top of that come two container starts per job slot, a second `_inputs` probe per episode (CPU, ≈2–60 s), and the persisted segment documents.

**Memory fits, by estimate.** Allocator high-water marks are process-wide:
- Captioner alone ≈ 82.4 − 17.9 = 64.5 GiB, against an H100's 79.6 GiB;
- TempSeg alone ≈ 81.4 − 68.8 + 17.9 ≈ 30.5 GiB (3-view), against an L40S's 44.4 GiB.

**The fidelity problem (why this is not recommended yet).** `config/hashing.py` excludes the GPU type: the reference is input-identical across GPU classes, not output-identical.
- On an L40S (sm89), torch's SDPA does not take the cuDNN path it takes on sm90. Perf-1 showed that a non-cuDNN backend changed 3 of 6 texts.
- So TempSeg's *outputs* would differ from the H200 reference. With them, the segment boundaries change, and so does what the Captioner is asked.
- The Captioner on an H100 is the same sm90 family, but a different SM count can change cuDNN's engine choice.
- A split "exact" is therefore a different reference unless TempSeg stays on an sm90 GPU (the H100+H100 case, $32.3, a small win).

**What was built (proof only, not wired).** `workers_impl/exact_split.py`:
- **Segment step.** NVIDIA's `annotate` runs and stops when it reaches `caption_single`/`caption_three`. The return value of NVIDIA's `pipeline.segment` is captured.
- **Persistence.** It is persisted in a tagged JSON encoding that keeps tuples, Fractions and floats exactly and refuses any other type.
- **Caption step.** NVIDIA's whole `annotate` runs again, with `pipeline.segment` answered from the document. NVIDIA re-probes the files, and the persisted `input_sha256` must match.
- No NVIDIA orchestration is re-expressed. The swaps are context-scoped and pass through everywhere else.
- `tests/nvlink/test_exact_scheduling_nvlink.py::test_split_tempseg_captioner_identical_to_single_process` covers 1-view H.264 with and without an instruction, 3-view YAM, and 3-view Dex AV1. It checks, against the single-process run:
  - the TempSeg calls' input digests;
  - the **Captioner calls' NVIDIA-logged `input_tensor_digest`s**;
  - the full NVIDIA result;
  - **all identical.**

Wiring it would need new shard kinds, `depends_on`, factory pools and receipts for the documents. I did not do that.

### Lever 4: longer-lived containers and larger shards

**Fixed cost per container.** 78 s `@enter` + ≈23 s scheduling + 300 s scaledown idle (`WORKER_SCALEDOWN_S`, which the budget charges) ≈ 401 s ≈ **$0.63** per container.

Containers already serve successive shards while the queue is non-empty, so this is paid per container *lifetime*: +11% for a 1 container-hour life, +2.8% for 4 h.

**Larger shards** (without overlap) divide the fill by K, the same effect as lever 1 (row 4a). But they:
- multiply the call timeout;
- make three preemptions of one shard (D15 quarantine plus bisection) more likely;
- reduce parallelism for small jobs.

Overlap gets the benefit without those costs.

**Preemption loss** is independent of shard size, because every episode is flushed when it finishes. It is ≤c in-flight episodes, about half an episode's wall each (≈150–230 s of container time), plus a restart (≈101 s) and a cold cache: ≈250–330 s ≈ **$0.40–0.52 per preemption**. The preemption rate on Modal H200 is **unmeasured**.

**The big lever is the warm cuDNN cache that a long-lived container keeps.** `gpu_thread` makes the cache persistent.
- Perf-1 puts generation fully warm at 1.8 GPU-s/video-s, against 5.04 cold.
- TempSeg warms quickly: its prompt length is fixed per profile (1,832–1,859 human, 5,509–5,566 3-view).
- Captioner prompt lengths cluster (this shard: 31 calls, 22 distinct lengths; 4,369 ×3, 4,391 ×5).
- The model at warm = 0.45 (GPU 408 → 184 s per shard) gives 3.9–4.0 container-s/video-s, **≈$22–23**. That matches W3-A's c=4 measurement on cu130, where graph builds cost 18 ms (4.08).
- Once warm, c=8 (`rates.json`'s 2.5, `requires: long_lived_container`) gives ≈$14.
- **How fast the Captioner warms is the key unknown.** Measure it.

### Lever 5: other input-identical levers

- **Memory 96 → 48 GiB** cuts the hourly price from $5.68 to $5.30 (−6.7%).
  - The batch needs ≤26 GiB RSS at c=4 (P1a-5s: 25.9 GiB), plus ≈18 GiB of CPU BF16 staging at load.
  - The cost: the 68.9 GiB t=0 prefetch no longer fits in the page cache, so the Captioner GPU load reads from the Volume again (≈54.7 GB at 3–5 GB/s, ≈+15 s of startup). Measure it.
- **Decode-ahead and input prefetch.** NVIDIA's `_inputs` probe (45–60 s at 1080p) and our serial input resolution (review 5 L1) only cost GPU time during the fill. Overlap hides them. A separate prefetch would mean caching NVIDIA probe objects for no extra gain.
- **The 56 s pre-start gap.** Attribute it first. If it is the per-episode `nv.audit` or `accel.episode()` entry, it is a per-episode cost that concurrency hides, except during the fill.
- **cpu 8 → 6** saves 1.7%. That is noise against the risk of a longer fill.

## Implemented (default off)

| Piece | Where |
|---|---|
| `pipeline.exact_scheduling: {episode_order: plan \| longest_first, shard_overlap: false}`, validated: overlap needs `resources.segcap.max_inputs` 2–4, segcap only. `exact_scheduling(cfg)` gives `{episode_order, max_shards}` | `config/schema.py` |
| `SegcapWorkerSettings.episode_order` / `max_shards`; `order_episodes` (LPT, ties in plan order); `scheduling_from_env` (`FLORA_SEGCAP_SCHEDULING`); multi-shard runner; refresh guard; preempt marks before it cancels; audit lines record `runtime.scheduling` | `workers_impl/segcap.py` |
| `FastExactEngine(overlap=True)`: shared slots, a cancel that never clears under a running batch, a lock reset only by the first batch | `workers_impl/segcap.py` |
| `EpisodeSlots` (priority by `(batch seq, position)`, cancellable); `annotate_batch(slots=, slot_key=, cancelled=, reset_lock=)` | `stages/reference.py` |
| Split TempSeg/Captioner steps (proof only) | `workers_impl/exact_split.py` (classified `nvidia-hf` in `config/logic.py`) |

**Hashing.** The two flags are **not hashed**. `config/hashing.py` picks fields explicitly, and `exact_scheduling` is not one of them; a test asserts that the job hash and the stage hashes are unchanged when they are toggled.

The flags change only *when* an episode starts. They change no argument, processor state or model call, as the identity tests show. The code itself lives in `nvidia-hf` logic modules, so this change set alters the `segcap` stage hash once, like any logic edit.

**Not wired (other agents' files).** Each needs about two lines:
- `modal_app/factory.py` sets `FLORA_SEGCAP_SCHEDULING=json.dumps(schema.exact_scheduling(cfg))`;
- `modal_app/reference_app.py` passes `**segcap.scheduling_from_env()` in `segcap_settings_from_env`, and `overlap=settings.max_shards > 1` to `FastExactEngine`;
- `rates.json` would get a `requires: ["shard_overlap"]` row once it is measured.

### Evidence (CPU)

| Test | What it proves |
|---|---|
| `tests/unit/test_exact_scheduling.py` (12) | LPT order; `EpisodeSlots` cap, priority and cancel. Two shards overlapped in one runner (production `FastExactEngine` composite over the fake NVIDIA family; NVIDIA-logged input digests) against sequential plan order: **identical records, strict per-call input digests, 100% paired**, cap held at 2, the second shard started while the first ran. Refusal beyond `max_shards`; preempt marks both shards; refresh guard; config validated and not hashed |
| `tests/nvlink/test_exact_scheduling_nvlink.py` (2) | **NVIDIA's real code**, real processors, synthetic H.264/AV1 clips: two overlapped batches (shared slots, reversed order, all accel patches) against NVIDIA unmodified, giving `identity_summary` strict on 4/4 episodes and full results identical 4/4. Split TempSeg/Captioner (above) |
| `tests/unit/test_exact_scheduling_local.py` (2) | **G2 end to end.** A whole local job (segcap + rewards) with `longest_first` and a preemption fault is `done`, keeps the same config hash, and gives byte-identical records to a clean plan-order run. Every segcap shard of G2's real index and manifest run at once in one container gives identical records and input digests to a sequential run |
| `tests/conformance` | New `segcap-worker[overlap]` spec (overlap, `max_shards=2`, longest-first): all contract checks pass |

## Recommendation

1. **Turn on `episode_order: longest_first` now.** It is zero-risk, recovers ≈9% against plan order, and brings ≈$40 back to the measured $36.4.
2. **Wire and enable `shard_overlap` with `max_inputs: 2`** after one GPU session confirms it. The expectation is **$29–30/video-h**; with 48 GiB memory, **$27–28**.
3. **Prioritize measuring long-lived warm containers.** That is the largest lever (≈$22–23 at c=4, ≈$14 at c=8) and needs no code near NVIDIA.
4. **Defer split pools.** The input-identity proof is clean, but TempSeg off sm90 is a different reference. On H100+H100 the gain is small ($32) and the complexity is high.

## What to measure in the next approved GPU session (≈$3)

One H200 container, `--memory-gib 96`, overlap plus longest-first, `max_inputs` 2. Three consecutive 6-episode golden shards, with 1-view and 3-view kept separate. About 25 min × $5.68 ≈ $2.4.

Record:
1. per-shard wall, `gpu_busy_fraction` over the whole container, and `episode_slots` stats;
   - expect ≥95% busy after the first fill and **≈5.1–5.3 container-s/video-s, i.e. $29–30**;
2. the first-episode start offset, timing `on_start` (`nv.audit_report`), `accel.episode()` entry and the probe separately: is the 56 s gap real without the validation prefetch?
3. Captioner and TempSeg s/call against container age and against prompt length seen before: the warm-up curve that decides lever 4b;
4. peak RSS and cgroup memory, to decide 48 GiB;
5. one extra start at 48 GiB to time `@enter` without the full page-cache prefetch.

Re-run the identity gate in `--determinism deterministic` on the same shards. The fast arm uses overlap, the reference arm is sequential.

## Reproduce

```bash
uv run pytest -q tests/unit/test_exact_scheduling.py tests/unit/test_exact_scheduling_local.py \
    tests/conformance
uv run pytest -q -m nvlink tests/nvlink/test_exact_scheduling_nvlink.py
```

The cost model I used for levers 1–2 is a discrete-event simulation calibrated to the P1a-5s shard: c slots, one GPU server, per-episode probe plus n×(CPU, GPU) phases. Its inputs are the per-episode `timings_s` and `generation` numbers from the P1a-5s results listed above.
