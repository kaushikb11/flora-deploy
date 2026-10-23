# Architecture

How flora-deploy is put together: the pipeline from dataset to labels, the design decisions
behind it, and what determinism and licensing mean here. Code comments that cite `PLAN.md §N`,
`PLAN §N` or `PLAN-v3` refer to earlier internal build plans, which are not part of the
repository; the decisions (`D1`-`D18`) below keep their numbers.

**Presets.** `balanced` serves NVIDIA's Flora models with vLLM (TempSeg on an L40S, the Captioner
and the Rewarder on H100s) and is what the templates use. `exact` runs NVIDIA's own code on one
H200 and is the reference that `balanced` is measured against
([`certification.md`](certification.md)). Rewards run NVIDIA's own engine in both.

## Pipeline

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

## Decisions

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
| D11 | **Config hash** over logic version, NVIDIA code and manifests, prompt pool, model revisions, derived-weight receipts, preset / engine / patches, all engine args, sampling, mm kwargs, decode pins; **excludes GPU type** (provenance). | Fixes invalidate old results; GPU fallback does not break resume. | revised: also a sha256 over the source of every output-affecting module (`config/logic.py`), per-GPU engine limits, the image fingerprint and a `fake_models` component |
| D12 | **CPU work in a process pool** with budgets; vLLM preprocessing in its own workers. | GIL contention; CPU billed whether used or not. | superseded by D17 form (1): CPU work is cut by identity-proven runtime patches (decode-once, in-memory PNG, transport and geometry pools) plus episode concurrency, not a separate pool ([`accel.md`](accel.md), [`exact-concurrency.md`](exact-concurrency.md)) |
| D13 | **Envelope policy `warn`** by default: out-of-envelope episodes run and are marked `certified:false`; `enforce` skips; `ignore` runs them like `warn` (their records are still `certified:false`: no certificate covers them). | Data gets labeled, status stays honest. | active |
| D14 | **Batch = offline `vllm.LLM` with a round-based micro-batcher**; online = AsyncLLM with pre-rendering. | NVIDIA's validated defaults; deterministic batch composition; AsyncLLM preprocesses on the event loop. | active for batch (`engines/microbatch.py`); online deferred to `interactive` (Gate E) |
| D15 | **Poison pills and OOM.** Attempt markers; quarantine after K=3 starts; bisection; `errors/`; `cancel()`; `memory=(request, limit)`. | Modal retries crashes and OOMs forever. | revised: a CUDA fault or dead engine marks the shard crashed and replaces the container |
| D16 | **A small explicit certified matrix**, certificates as metrics-only JSON, community certificates by PR. | Certification cost grows combinatorially. | revised: one certificate per (preset, kind, GPU, embodiment pool), content-addressed, with a bounded policy and revocation order ([`certification.md`](certification.md)); first targets: segment L40S, caption H100!, pool `human` (rewards run NVIDIA's own recipe in both presets, so they are the reference and need no certificate) |
| D17 | **Never modify NVIDIA's files; improve around them**: (1) runtime wraps proven output-identical, (2) faster reimplementations behind `balanced`, certified, (3) upstream suggestions. | A modified copy is an NC derivative, fails NVIDIA's SHA manifests and destroys the reference. | active |
| **D18** | **`balanced` is the product; `exact` is the reference.** Users get `balanced` by default (template, README, planner). `exact` is kept to produce oracle and noise runs for certification and is not cost-optimised further (shard overlap was removed for this reason, [`exact-cost.md`](exact-cost.md)). | Measured `exact` is ≈ $36/video-h on H200 ([`costs.md`](costs.md)), too costly as a product; `balanced` is the cheaper path and can be made trustworthy by certification; maintaining two optimised paths splits effort. | active (2026-10-22) |

## Determinism

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

## Licensing

- **Repo:** Apache-2.0, with no NVIDIA code, prompts, weights or outputs. The license guard
  (`tests/unit/test_license_guard.py`, `tests/nvlink/test_license_content.py`) enforces this.
  NVIDIA's code is fetched at test or run time (`scripts/fetch_nv_code.py`), never committed.
- **Re-expressed NVIDIA logic** is minimised (the fast `exact` worker calls NVIDIA's own
  `annotate()`). What remains, such as the 3-view segmenter
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
