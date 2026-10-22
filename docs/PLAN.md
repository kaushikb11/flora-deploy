# Build plan v4: `flora-deploy`, balanced-first

> **Status:** v4 · 2026-10-22 · github.com/kaushikb11/flora-deploy (Apache-2.0).
> Supersedes v3, which is kept as [`archive/PLAN-v3.md`](archive/PLAN-v3.md). Code comments that
> cite `PLAN.md §N` use v3's section numbers; §11 maps them. The execution history (Phase 0,
> reviews 3-6, Wave H planning) is in [`archive/`](archive/). The index of every doc is
> [`README.md`](README.md).

## 0. Summary

`flora-deploy` deploys NVIDIA's Flora models (TempSeg, Captioner, Rewarder) into the user's own
Modal workspace. It labels robot and human-demo video datasets as resumable batch jobs under a
hard budget cap.

- **`balanced` is the product.** It runs vLLM 0.24.0: TempSeg on an `L40S`, the Captioner on an
  `H100!` and the Rewarder on an `H100!`. It is what `flora init --template balanced_lerobot`
  writes and what the README documents.
- **`exact` is the reference.** It runs NVIDIA's own HF code on one `H200`, with model inputs
  proven identical to unmodified NVIDIA. It exists so `balanced` can be certified against it; it
  is not tuned for cost.
- **A `balanced` stack is trusted only through a certificate** that shows it diverges from
  `exact` no more than `exact` diverges from itself ([`certification.md`](certification.md)).
  One is committed: **`segment` is certified for `human` on the L40S** (run `10221548f1b8`: 12
  episodes, a 6-episode noise run and a 6-episode cold witness). **`caption` is not certified**:
  it failed its noise-floor check, and a re-certification under rules fixed before the run is
  planned. A stage without a certificate needs `allow_uncertified: true`, and its records say
  `certified: false` (`uncertified_stack`).

**Where we are (2026-10-22).** The job system, both presets, rewards, the certificate generator
and its consumption, local fake-model runs and the CI gates are built and tested on CPU. GPU
evidence exists for the `exact` path, the Rewarder, Gate C (per request), the BF16 Captioner and
the TempSeg merge. `balanced` has run on GPUs through the real `flora deploy` / `flora run` path
(Wave H-lite, [`wave-h-report.md`](wave-h-report.md)): its segment and caption rates are
measured (one small job, n=1) and segmentation is certified for `human`. Next: the caption
re-certification and the scale test (§9). The project budget is `FLORA_PROJECT_BUDGET_USD`
(default $50); `scripts/spend.py` reports what is left. Nothing is released; v0.x labels are
internal milestones only.

---

## 1. Product

### Users
| Persona | Job | Entry point |
|---|---|---|
| Robot-learning researcher | "Caption and reward my LeRobot episodes by tomorrow, under $X." | `flora init` → `prepare` → `index` → `plan` → `deploy` → `run` |
| Data-platform engineer | "Label continuously, resume after failures, know my $/video-hour." | config, stores, `status` / `report`, budgets |
| RL / eval engineer | "Score rollouts (terminal / dense)." | reward stages |
| Tool builder | "Call Flora from Python." | `flora_deploy.sdk` (`submit`, `Job`) |

### Presets (`config/presets.py`)
| Preset | Status | Engines | Default pools | Use |
|---|---|---|---|---|
| **`balanced`** | **the product**; `segment` certified for `human` (L40S); `caption` not yet; rewards need none | vLLM 0.24.0 offline `LLM` + micro-batcher (TempSeg merged FP32→BF16, Captioner BF16); Rewarder on NVIDIA's own vLLM recipe | `segment` L40S (8 cores, 48 GiB) · `caption` H100! (8, 96 GiB) · rewards H100! (8, 80 GiB) | all labeling |
| `exact` | the certification reference | NVIDIA's HF code via `CachedSession`, OPT-1/OPT-2 identity-proven patches | `segcap` H200 (8 cores, 96 GiB) · rewards H100! | oracle and noise runs for certification; gold labels |
| `budget` | declared, not available | FP8 / B200 kernels | — | P4 |
| `interactive` | declared, not available | AsyncLLM servers | — | §9 step 5 |

### Goals
- **G1 Fidelity.** `balanced` is certified per (preset, stage, GPU, embodiment pool) within the
  measured noise of `exact` (§4).
- **G2 One-command path** into the user's own account; workers scale to zero.
- **G3 Cost control.** An estimate before the run (`flora plan`), a hard cap during it (the
  orchestrator's budget gate), a spend report after it (`flora report`).
- **G4 Resumability.** A crash, preemption, OOM or Ctrl-C loses at most one shard; poison
  episodes are isolated and excluded; `flora resume` is idempotent.
- **G5 I/O.** In: LeRobot v2.1 / v3 and MP4 directories. Out: JSONL + Parquet + `summary.json`
  ([`output-format.md`](output-format.md)).
- **G6 Online API** (`interactive`), later.

Non-goals: training Flora; a hosted multi-tenant service (commercial use, §7); non-Modal backends
(the code stays Modal-agnostic); robot control or safety; certified results outside NVIDIA's
envelope (D13).

---

## 2. Architecture

```
user machine:  flora CLI / SDK  (init · doctor · prepare · index · plan · deploy · run ·
               status · logs · cancel · resume · report · results · certify)
                    │
   index ──► plan ──► deploy ──► run ───────────────────────────────────► compose ──► results
     │        │         │         │                                          │
     ▼        ▼         ▼         ▼                                          ▼
  index     shards    one Modal  orchestrator (CPU, preemptible,         deduped JSONL /
  receipts  + rate    app per    Dict lease, 23 h continuation)          Parquet / summary.json
  per       estimate  config     ├─ spawns shard calls per kind, chained per episode
  episode             (factory)  ├─ attempts → quarantine → bisection → errors/
                                 ├─ budget gate (spend + in-flight + remaining)
                                 └─ reconciler: status rebuilt from store files
                                          │
           workers (module-scope classes, decorated by build_app):
             balanced:  SegmentWorker (L40S) ──per episode──► CaptionWorker (H100!)
             exact:     ReferenceWorker / segcap (H200)
             both:      RewardWorker (H100!): reward.terminal, reward.dense
                                          │  whole-file writes: attempt marker first,
                                          ▼  parts, receipt last
           stores (state of record): bucket (s3/gs/r2) · Modal Volume · local (tests)
           Volumes: flora-weights (snapshots + derived) · flora-cache (compile caches)
                    flora-golden (golden set, oracle runs) · media staging Volumes
```

| Step | What it does | Code |
|---|---|---|
| **index** | Dataset → one index receipt per episode: per-episode `average_rate` (FFmpeg's formula over the episode's packets), relative PTS, duration, CFR, codec, keyframe start, camera roles, 3-view sync, envelope verdict, task text. Local (`--local`) or a CPU fan-out on Modal. | `jobs/indexing.py`, `media/`, `modal_app/index_app.py`; [`media-notes.md`](media-notes.md) |
| **plan** | Index + `RunSettings` → manifest of shards (grouped by source file, packed to a target of GPU-minutes); per-pool cost / time options from `planner/rates.json` and pinned or live prices; certificate status per pool. | `jobs/manifest.py`, `jobs/planning.py`, `planner/` |
| **deploy** | `plan_app(cfg)` → `AppPlan` (pools, GPUs, env, Volumes, secrets, timeouts); `build_app` decorates module-scope classes from it. `--dry-run` and `--emit` show the same plan. Refuses an uncertified non-`exact` preset unless `allow_uncertified`. | `modal_app/factory.py`, `modal_app/emit.py` |
| **orchestrator** | The job loop: lease, chaining by receipts (caption shards depend on the segment shards holding their episodes), GPU cap, budget gate, cancel, continuation; catches Modal's `TimeoutError` subclasses first. | `jobs/orchestrator.py`, `jobs/reconciler.py`, `modal_app/orchestrator_app.py` |
| **workers** | The `ShardWorker` contract: attempt marker → skip done episodes → per-episode records → local parts → whole-file copy → receipt last; preemption flush; NVIDIA failures become `EpisodeError` (`nv_failure`), infrastructure failures crash. Every worker passes the shared conformance suite. | `workers_impl/`, `modal_app/{balanced,reference,rewarder,workers}_app.py`, `tests/conformance/` |
| **stores** | Whole files under hash-prefixed directories, keyed by the stage hash. Volume store follows commit/reload discipline. | `jobs/store.py`, `jobs/stores/` |
| **compose** | Parts → deduped, validated `flora_labels_v1` per stage; `summary.json` with exclusions. | `export/compose.py`, `export/results.py`, `records.py` |

Inside a `balanced` worker, NVIDIA's own code builds every window, frame plan and prompt. We
replace only `pipeline.generate` with a tap that collects the captured requests, sends them to the
micro-batcher and puts the texts back. Completions are accepted with NVIDIA's rules in NVIDIA's
call order. Details: [`balanced-engines.md`](balanced-engines.md).

Rules: only `nv/` imports NVIDIA modules, and never two NVIDIA repo families in one process.
`stages/` and `engines/` never import `modal`. Every package module is classified for the logic
digest (`config/logic.py`).

---

## 3. Decisions

Status: **active**, **revised** (still holds, changed as noted), or **superseded** (by the
decision named).

| # | Decision | Why | Status |
|---|---|---|---|
| D1 | **Linked mode.** NVIDIA's modules are imported at runtime from the user's HF snapshot, under unique module names, one process per NVIDIA repo family. We replace only model invocation and media I/O. | Faithful by construction; no noncommercial code in the repo; contains the three `batch_utils` name collisions. | active |
| D2 | **vLLM 0.24.0 with explicit engine args everywhere** (`gdn_prefill_backend="triton"`, `max_num_seqs`, `max_num_batched_tokens`, `generation_config="vllm"` / stop tokens, prefix caching per stage, `mm_processor_cache_gb=0`). | Matches NVIDIA's stack; removes GPU-dependent defaults from certificates. | active (`engines/offline_llm.py` refuses missing or uncertified values) |
| D3 | **TempSeg served as an FP32-merged checkpoint** (BF16 at runtime) as a plain model; HF+PEFT is the oracle. | vLLM #52568 (NaN with LoRA on hybrid GDN); HF padding with recurrent state is unsafe. | active; merge built and checked ([`tempseg-merge.md`](tempseg-merge.md)); merger v2 content digest not pinned yet, so `SegmentWorker` refuses until it is; Gate B pending |
| D4 | **Captioner gets a one-time BF16 conversion**, `A_log`/`dt_bias` FP32. | Halves size; checkable. | active; proven bit-identical to NVIDIA's in-memory weights, also used by `exact` (OPT-2, [`captioner-bf16.md`](captioner-bf16.md)) |
| D5 | **FP32 `dt_bias` patch is gated (Gate A)**, via `vllm.general_plugins` if shipped. | Upstream fix closed unmerged; pay for it only if drift matters. | active; plugin built, **off**; Gate A not measured; plugin-on is a different certificate key |
| D6 | **State of record is the user's store.** Whole-file writes under hash-prefixed directories. | Volume v2 is beta; 262k files per directory; reload fails on open files. | revised: a Modal Volume sink (`volume://`) is a supported store with commit/reload discipline, not only a warned fallback |
| D7 | **Range decode from the source; per-episode probe equivalence; stream-copy cuts for oracle runs.** | Re-encoding changes time base and colour tags; a clip cache is ≈PB at scale. | active for index and cuts; the `balanced` workers still call NVIDIA's own `media.probe` and decode per camera (see `balanced-engines.md`, known limits) |
| D8 | **Stage-separated GPU pools**, chained per shard. | Different GPUs per stage; independent scaling. | revised: `balanced` uses separate pools chained **per episode**; `exact` keeps one fused `segcap` H200 worker (both models resident), as the reference |
| D9 | **Presets** `exact` / `balanced` / `budget` / `interactive`; uncertified combinations refused unless allowed. | Makes the trade-off explicit. | revised by D18: the opt-in is the `allow_uncertified` config key; `budget` / `interactive` are declared but refused |
| D10 | **App factory.** Module-scope worker classes decorated in `build_app(cfg)`; `--emit`; no `serialized=True`. | Modal needs global-scope definitions; version-independent deploys. | active |
| D11 | **Config hash** over logic version, NVIDIA code and manifests, prompt pool, model revisions, derived-weight receipts, preset / engine / patches, all engine args, sampling, mm kwargs, decode pins; **excludes GPU type** (provenance). | Fixes invalidate old results; GPU fallback does not break resume. | revised: also a sha256 over the source of every output-affecting module (`config/logic.py`, review A1), per-GPU engine limits, the image fingerprint and a `fake_models` component |
| D12 | **CPU work in a process pool** with budgets; vLLM preprocessing in its own workers. | GIL contention; CPU billed whether used or not. | superseded by D17 form (1): CPU work is cut by identity-proven runtime patches (decode-once, in-memory PNG, transport and geometry pools) plus episode concurrency, not a separate pool ([`accel.md`](accel.md), [`exact-concurrency.md`](exact-concurrency.md)) |
| D13 | **Envelope policy `warn`** by default: out-of-envelope episodes run and are marked `certified:false`; `enforce` skips; `ignore` runs them like `warn` (their records are still `certified:false`: no certificate covers them). | Data gets labeled, status stays honest. | active |
| D14 | **Batch = offline `vllm.LLM` with a round-based micro-batcher**; online = AsyncLLM with pre-rendering. | NVIDIA's validated defaults; deterministic batch composition; AsyncLLM preprocesses on the event loop. | active for batch (`engines/microbatch.py`); online deferred to `interactive` (Gate E) |
| D15 | **Poison pills and OOM.** Attempt markers; quarantine after K=3 starts; bisection; `errors/`; `cancel()`; `memory=(request, limit)`. | Modal retries crashes and OOMs forever. | revised: a CUDA fault or dead engine marks the shard crashed and replaces the container |
| D16 | **A small explicit certified matrix**, certificates as metrics-only JSON, community certificates by PR. | Certification cost grows combinatorially. | revised: one certificate per (preset, kind, GPU, embodiment pool), content-addressed, with a bounded policy and revocation order ([`certification.md`](certification.md)); first targets: segment L40S, caption H100!, pool `human` (rewards run NVIDIA's own recipe in both presets, so they are the reference and need no certificate) |
| D17 | **Never modify NVIDIA's files; improve around them**: (1) runtime wraps proven output-identical, (2) faster reimplementations behind `balanced`, certified, (3) upstream suggestions. | A modified copy is an NC derivative, fails NVIDIA's SHA manifests and destroys the reference. | active ([`upstream-suggestions.md`](upstream-suggestions.md) for form 3) |
| **D18** | **`balanced` is the product; `exact` is the reference.** Users get `balanced` by default (template, README, planner). `exact` is kept to produce oracle and noise runs for certification and is not cost-optimised further (shard overlap was removed for this reason, [`exact-cost.md`](exact-cost.md)). | Measured `exact` is ≈ $36/video-h on H200 (§5), too costly as a product; `balanced` is the cheaper path and can be made trustworthy by certification; maintaining two optimised paths splits effort. | **new (2026-10-22)**; supersedes v3's release train (ship `exact` first as v0.1, `balanced` as v0.2) |

---

## 4. Certification

Full rules: [`certification.md`](certification.md). In short:

- `exact` is **not** run-to-run deterministic on H200 in NVIDIA's default mode (§6). So
  "faithful" means: `balanced` diverges from `exact` no more than `exact` diverges from a repeat
  of itself.
- Inputs: the oracle's records, the candidate's records on the same episodes, a noise run (the
  oracle repeated on ≥ 5 of its episodes), the L0 lab result (vLLM `prompt_token_ids` and
  `video_grid_thw` vs NVIDIA's HF processor) for video stages, and the derived-weight content
  digests.
- Metrics are divergences per stage (segment IoU / boundary / count / label, caption mismatch and
  edit distance, terminal E[s] delta, dense per-step and rank divergence). Pass rule:
  `mean(exact, balanced) <= min(cap, max(absolute_floor, 1.5 x max(noise, floor) + slack))`
  (`validate.certkey.metric_threshold`; the absolute floor is set only for `count_mismatch`,
  0.10).
- Verdicts: `fail`, `provisional` (no noise run, < 10 candidate or < 5 noise episodes, missing
  L0, incomplete coverage) or `certified`. **Only a deciding `certified` certificate unlocks a
  preset.** Provenance on every record binds it to the stack (`cert_key`, stage hash, GPU,
  embodiment); fake-model records are refused.
- Certificates live in `src/flora_deploy/config/certificates/`, metrics only.

The first GPU session (§9 step 1) could produce only a **provisional** certificate (too few
episodes by design). The certified run after it (§9 step 2) produced the committed `segment` /
`human` / L40S certificate; `caption` failed.

---

## 5. Cost model

Spend = container lifetime × (GPU + cores + memory) at Modal list price (`jobs/budget.py`,
pinned 2026-10-21: H200 $4.54/h, H100 $3.95/h, L40S $1.95/h, $0.0472 per core-hour, $0.008 per
GiB-hour; live `billing.rates()` when available). Container time per video-second comes from
`planner/rates.json`; startup is added per container.

| Path | Pool | Container-s per video-s | Status | ≈ $/video-h |
|---|---|---|---|---|
| `exact` segcap, longest-first | H200, 8 cores, 96 GiB ($5.69/h) | 6.4 | measured (P1a-5s, one 6-episode shard) | **$36.4** |
| `exact` segcap, plan order | same | 7.0 | measured + model | ≈ $40 |
| `exact`, long-lived warm containers, c=8 | same | ≈ 2.5 | projected (needs a GPU measurement) | ≈ $14 |
| `balanced` segment | L40S, 8 cores, 48 GiB ($2.71/h) | see `rates.json` | measured (Wave H-lite, one job, n=1, ±50%) | see [`costs.md`](costs.md) |
| `balanced` caption | H100!, 8 cores, 96 GiB ($5.10/h) | see `rates.json` | measured (Wave H-lite, one job, n=1, ±50%) | see [`costs.md`](costs.md) |
| `reward.terminal` | H100!, 8 cores, 80 GiB ($4.97/h) | 0.055 | measured (WS-H, 3 samples) | ≈ $0.27 |
| `reward.dense` | same | 0.165 | measured (WS-H, 3 samples) | ≈ $0.82 |

Startup: `exact` `@enter` 103 s (measured, conservative); Rewarder 229 s with a warm compile
cache (measured); `balanced` segment and caption cold starts are measured per kind
(`planner/rates.json` startup rows; warm starts not measured yet).

The `balanced` rows come from one 3.9-minute job ([`wave-h-report.md`](wave-h-report.md)). The
numbers live in `planner/rates.json` and the generated tables of [`costs.md`](costs.md); they
are not repeated here so they cannot go stale. Large-job behaviour is unmeasured until the
[scale test](scale-test.md). Sources: [`exact-cost.md`](exact-cost.md),
[`exact-worker.md`](exact-worker.md), [`perf-1.md`](perf-1.md),
[`exact-concurrency.md`](exact-concurrency.md), [`rewarder-smoke.md`](rewarder-smoke.md),
[`cold-start.md`](cold-start.md).

---

## 6. Determinism

Evidence: [`determinism.md`](determinism.md), [`exact-concurrency.md`](exact-concurrency.md),
[`rewarder-smoke.md`](rewarder-smoke.md).

- **`exact` means identical model inputs, not identical outputs.** On H200, PyTorch SDPA's
  cuDNN attention is not bit-reproducible between runs; occasionally a greedy token flips and
  everything downstream follows.
- **Deterministic mode** (`pipeline.determinism: deterministic`, cuDNN SDPA off) is opt-in. It is
  a different reference, not the same bits. Identity proofs (fast `exact` worker vs unmodified
  NVIDIA) run in deterministic mode, because in default mode the first output difference leaves
  every later call unchecked. Oracle and noise runs use NVIDIA's default mode, the behaviour
  users would get from NVIDIA's code.
- **Batch composition changes vLLM numerics.** The Rewarder's outputs depend on how requests are
  grouped, and the micro-batcher packs across episodes. This is allowed for `balanced` and is
  exactly what certification measures against the noise floor.
- Every record states its determinism mode; certificates bind oracle and noise runs to the same
  mode.
- Byte-determinism of derived weights across machines is covered in
  [`captioner-bf16.md`](captioner-bf16.md) and [`tempseg-merge.md`](tempseg-merge.md).

---

## 7. Licensing

- **Repo:** Apache-2.0, with no NVIDIA code, prompts, weights or outputs. The license guard
  (`tests/unit/test_license_guard.py`, `tests/nvlink/test_license_content.py`) enforces this.
  NVIDIA's code is fetched at test or run time (`scripts/fetch_nv_code.py`), never committed.
- **Re-expressed NVIDIA logic** is minimised (S1 removed the largest piece: the fast `exact`
  worker calls NVIDIA's own `annotate()`). What remains, such as the 3-view segmenter
  construction in `nv/segment_glue.py`, is flagged for counsel review.
- **Derived weights** (BF16 Captioner, merged TempSeg) are built only in the user's Volume and
  never published.
- **Acknowledgement:** `license_ack: research-only`. Flora is NVIDIA OneWay Noncommercial:
  research and evaluation only; a paid hosted service is commercial use; not for robot control or
  safety; whether labels are restricted is unresolved, so get legal advice before commercial use
  ([`../NOTICE.md`](../NOTICE.md)).
- **Certificates hold metrics only**, never captions, prompts, episode ids or dataset names (a
  property test checks this).
- **Accepted risk:** NVIDIA deletes or rewrites the HF repos; we do not mirror them.
- *Not legal advice.* Counsel reviews before 1.0.

---

## 8. Gates and open technical items

| Gate | Question | Status |
|---|---|---|
| A (D5) | Does FP32 `dt_bias` drift matter? | measured in Wave H-lite: max \|Δlogit\| 0.25, within the 0.5 reference; plugin stays off ([`wave-h-report.md`](wave-h-report.md)) |
| B (D3) | Does the merged TempSeg serve correctly? | CPU / L40S equivalence check passed (BF16-level drift, identical greedy JSON); merger v2 content digest pinned (`config/pins.py`); served on vLLM in Wave H-lite (L0 pass), and its segmentation is certified for `human` |
| C | Does vLLM see the same model inputs as NVIDIA's HF processor? | every compared request passes (127/127) with the capture-layer wrap and `mm_processor_cache_gb=0`; the gate is FAIL on two run-level criteria: the `tianji` quota and the missing `img_hfref` cross-check ([`gate-c.md`](gate-c.md)) |
| D | Where does CPU time go? | answered for `exact` (generation 9-31% of a stage on 1080p; fixed by `accel`); `balanced` throughput measured on one small job, at scale pending the [scale test](scale-test.md) |
| E (D14) | Offline micro-batching vs AsyncLLM | offline is the batch default; AsyncLLM comparison moves to `interactive` |

Other open items: the per-GPU `max_num_seqs` limits are derived from the worst-case KV fit
(`engines/offline_llm.py` `GPU_LIMITS`: TempSeg/L40S 48, Captioner/H100 16), and raising them
needs a measured run with zero preemptions;
the rubric reward is deferred (TP=2, `max_model_len` 98304); the golden set has no real `dex` or
`tianji` rig ([`golden-candidates.md`](golden-candidates.md),
[`golden-manifest-v1.md`](golden-manifest-v1.md)).

---

## 9. Roadmap from here

Each step needs the one before it. GPU steps need the user's approval and run under stop rules.

### 1. GPU session (Wave H-lite) — done
- **Before:** review clean, CI green, and H0 passing: `scripts/h0_rehearsal.py`, the real
  factory app on Modal with fake engines in CPU containers, both presets, one preemption and one
  poison episode (≈ $0.10).
- **Session** ([`wave-h-runbook.md`](wave-h-runbook.md), `scripts/wave_h.py`), reordered
  balanced-first: `exact` oracle in default mode → `balanced` through the real `flora deploy` /
  `flora run` on the same episodes, plus the lab's L0 and Gate A → rewards → a noise run →
  offline provisional certificate and measured rates.
- **Scope (decision 2026-10-22):** one embodiment (`human`) plus the lab's L0 / Gate A checks.
  Rough estimate ≈ $3.2 nominal, ≈ $6.5 upper; the runbook's dry run prints the real
  projections and refuses a step whose upper projection does not fit its cap.
- **Exit (met):** measured `balanced` rates replaced the placeholders; Gates A and B decided; the
  deployed path ran on GPUs ([`wave-h-report.md`](wave-h-report.md)).

### 2. Certified — in progress
- Golden build sized for a `certified` verdict: ≥ 10 candidate episodes per pool and ≥ 5 noise
  episodes, starting with `human`, then `yam` and single-view robot. The full golden build (≈ $9)
  was deferred by the user and needs approval.
- **Done for `human`:** run `10221548f1b8` certified `segment` on the L40S (12 episodes, a
  6-episode noise run, a 6-episode cold witness). TempSeg merger v2 is pinned and the CLI / SDK
  pass the derived content digests ([`certification.md`](certification.md)).
- **Open:** `caption` failed its noise-floor check; it is re-certified under rules fixed before
  that run. Close Gate C's run-level items; `yam` and single-view robot pools.
- **Exit:** committed `certified` certificates; `balanced` runs without `allow_uncertified` on the
  covered pools; `flora plan` within ±25% of actual cost.

### 3. Scale test
- A 1,000-episode `balanced` job: forced preemptions (`simulate_preemption()`), an injected OOM
  episode, an orchestrator kill, a budget stop and resume; no lost or duplicated episode.
- Long-lived containers and per-GPU limits tuned from measurements; GPU utilisation > 60% while
  busy.

### 4. Rubric
- `reward.rubric` on `H100!:2` (TP=2), per-episode files in a codec decord reads, a rubric L0, and
  a distributional certificate.

### 5. Interactive
- `@app.server` with AsyncLLM and pre-rendered inputs (Gate E), auth, an input allowlist, size and
  duration limits, our own rate limits; Gate C extended to the online path.

### 6. Release
- Only when the user asks: docs reviewed against code, counsel review of licensing, tag and
  publish. Until then nothing is tagged, published or announced.

Deferred until the user decides: production-grade certificates over every pool, posting the
upstream suggestions, `budget` preset work (FP8, B200), LeRobot v3.1 write-back, a non-Modal
backend.

---

## 10. Operating rules

- **Fidelity invariant.** Anything labeled `exact` is proven input-identical to unmodified NVIDIA
  code (D17). `balanced` is trusted only through certificates. No exceptions.
- **CPU first.** Everything that can be proven without GPUs is proven first: unit and
  conformance tests, nvlink tests against NVIDIA's real code, `flora run --local --fake-models`,
  the real-Modal CPU canary and H0. GPU minutes are spent only on what a CPU test cannot catch.
- **Spend.** Ask before any single run over ≈ $2. Each GPU step has a cap and a stop rule; track
  actual spend with `scripts/spend.py`.
- **Process.** Work happens on branches with owned paths; every branch gets an independent review
  before merge; CI gates are listed in [`ci.md`](ci.md).

---

## 11. Where v3 sections went

Code comments and older docs cite v3 (`PLAN.md §N`, `PLAN §N`). They refer to
[`archive/PLAN-v3.md`](archive/PLAN-v3.md):

| v3 section | Now |
|---|---|
| §0 TL;DR, release train | §0, D18 |
| §1 verification findings (NVIDIA, vLLM, Modal) | v3 only (evidence); the consequences live in D1-D17 and the code |
| §2 product, presets, goals, success metrics | §1; metrics in §4 and §9 |
| §3 decisions D1-D17 | §3 |
| §4 architecture | §2 |
| §5 repository layout and import rules | §2 (rules); the tree is the source |
| §6 interfaces (engine, capture, micro-batcher, stage contract, `CachedSession`, schema, budgets) | code; schema in [`output-format.md`](output-format.md) |
| §7 Modal topology (images, storage, functions) | §2 and `modal_app/` |
| §8 job lifecycle, fault tolerance, budget | §2, D15, §5 |
| §9 validation and certification, gates | §4, §8, [`certification.md`](certification.md) |
| §10 phased roadmap (P0-P4) | §9 |
| §11 licensing | §7 |
| §12 testing strategy | [`ci.md`](ci.md), §10 |
| §13 risks | §8 and the decision rationales |
| §14 open questions | §12 |
| §15 first two weeks, §16 changelog | v3 only |

---

## 12. Open questions for the user

1. Modal plan tier (Starter 10 / Team 50 / Enterprise GPUs): sets defaults and the scale-test size.
2. Commercial intent for the labels: decides when counsel review happens.
3. Envelope default for episodes over 120 s: `warn` (today) or `enforce`.
4. When to approve the GPU session (§9 step 1) and the certification golden build (step 2).

Resolved since v3: the name and home (`flora-deploy`); the first dataset (public golden
candidates, no private data); release priority (D18); the state of record (bucket or Volume, D6).
