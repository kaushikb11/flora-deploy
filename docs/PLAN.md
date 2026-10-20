# Build plan: `flora-deploy`, an open framework for running NVIDIA Flora at scale on Modal

> **Status:** draft v3 · 2026-10-21 · project `flora-deploy` (github.com/kaushikb11/flora-deploy)
>
> **Inputs:**
> - Research report: `reports/Flora deployment framework on Modal.md`
> - Research notes: `research_notes/Flora deployment framework on Modal/`
> - Verification of this plan in `research_notes/Plan review v2/`:
>   - `vllm_0240_verification.md` (vLLM source at tag v0.24.0)
>   - `modal_verification.md` (Modal docs and the modal 1.6.2 client)
>   - `nvidia_code_audit.md` (NVIDIA's TempSeg, Captioner and Rewarder `batch_utils`, file:line evidence)
>
> v3 applies those verifications and restructures the release strategy. See the changelog in §16.

---

## 0. TL;DR

We build an Apache-2.0 Python package and CLI. With one command it deploys Flora's three models into
the user's own Modal workspace and labels robot or human video datasets, either as resumable batch jobs
or behind a warm HTTP API. The framework sizes each run from what the developer asks for: a budget, a
deadline and a fidelity level. Every faster path must be certified against NVIDIA's own code before it
can be used.

### Strategy: separate "exact at scale" from "cheap"

The highest-value first release is NVIDIA's own pipeline running unchanged on Modal with horizontal
scale-out, resumable jobs and a hard budget cap.

- It is exact by construction, so it needs no certification.
- It is expensive per GPU-hour. Even so, a researcher with 10–500 video-hours can use it today.

Then we make it cheaper. Each faster engine path ships only after it passes certification against the
reference.

| Release | Profile | What it is | When |
|---|---|---|---|
| **v0.1** | `exact` | Job system plus NVIDIA's reference code for all stages. The Rewarder runs on NVIDIA's own offline-`LLM` recipe. | ≈ week 6–7 |
| **v0.2** | `balanced` | Certified vLLM paths for TempSeg and the Captioner. The Rewarder is already on vLLM. About 4× cheaper (estimate). | ≈ week 13 |
| **v0.3** | — | Throughput, breadth, QA tooling | ≈ week 16 |
| **v0.4** | — | Online API and scale testing | ≈ week 20 |

### The decisions that matter most

1. **Linked mode.** NVIDIA's modules are imported at runtime from the user's own HF snapshot, under
   unique module names and in separate processes per repo. The framework replaces only model calls and
   media I/O. Our repo contains no noncommercial code.
2. **Batch through NVIDIA's own hook.** The segmenter awaits `predictor.apredict(request)`, and all
   paths stay on one event loop. We answer those calls with a **round-based micro-batcher**: it waits
   until every in-flight episode is blocked on the model, then sends all their windows to vLLM at once.
   This is the default for batch work (D14). The streaming `AsyncLLM` engine is used for online serving.
3. **A capture processor builds the vLLM requests.** It records NVIDIA's final
   `proc(text=…, videos=…, video_metadata=…, do_sample_frames=False)` call. The only result field
   NVIDIA's code reads back is `.pop('video_metadata')`.
4. **Decode from the source, and replicate NVIDIA's probe for each episode.** We never keep a clip
   cache. An index pass computes each episode's own `average_rate`, CFR flag and duration using
   FFmpeg's formulas, because values taken from a concatenated container are wrong. Oracle runs use
   **stream-copy** per-episode cuts, never re-encodes.
5. **The state of record lives in the user's bucket.** Workers write whole files (parts and receipts)
   under hash-prefixed directories. Modal Volumes hold only weights and caches; Volume v2 is beta and
   may lose data.
6. **Explicit handling of poison pills and OOMs.** Modal retries crashes and OOM kills forever, so
   attempt markers, quarantine and bisection are mandatory.
7. **Requirements-driven planning.** `flora plan --budget 2000 --deadline 48h --fidelity balanced`
   reads the dataset index, measured rates and the workspace quota. It proposes configurations with
   cost, wall-clock time and certification status.

**Timeline:** ≈18–20 weeks for one strong engineer. **GPU spend for development and certification:**
≈$1.5–2.5k (estimate).

---

## 1. What we verified (and what it changed)

### 1.1 NVIDIA code (`nvidia_code_audit.md`; the Captioner's `batch_utils` is byte-identical to TempSeg's)

| Finding | Evidence | Consequence |
|---|---|---|
| `apredict` is preferred. Hierarchical segmentation calls `segment_with_priors_async`, and repair calls `predict_repair_candidates_async`, all on the caller's loop. An episode makes **up to 3 `apredict` requests**: the grid, then a recentered retry, then a directional retry, with a 32-window retry budget. | `fine_subtask.py:250-305, 600, 834-844` | The micro-batcher design is sound. Expect up to 3 rounds per episode. |
| A parse error, a policy-budget violation or a missing EOS **fails the whole episode**, which goes to NVIDIA's adaptive repair and then conservative completion | `fine_subtask.py` | We reproduce NVIDIA's failure semantics. Our own error path applies only to infrastructure errors. |
| **The 3-view segmenter is built inline in `pipeline.py:47-49`**, with a different profile (`robot_atomic_intent_v1`) and quality policy. `adapter.production_segmenter` covers human and single-view robot episodes (the human policy). | `pipeline.py`, `adapter.py:44-47` | `nv/compat.py` mirrors both constructions exactly. The 3-view constants are re-expressed NC glue (§11). |
| **`media.probe` reads `stream.average_rate`, a whole-file value.** The 1 µs CFR test gates the 3-view path. The rate also feeds caption fps, timestamp tokens and u2 frame counts. | `media.py:31, :55` | The index pass computes the per-episode rate with FFmpeg's `average_rate` formula over that episode's packets, plus relative PTS, `duration_us = round(last + tail)` and CFR. L0 checks it against a stream-copy per-episode file. |
| **Lossless re-encodes change the time base** (≈1 ms quantization) and possibly color tags. Color tags decide `canonical_rgb` vs swscale. | audit §4 | Oracle clips are **stream-copy cuts**, which needs episodes that start on a keyframe. Datasets that don't are oracle-ineligible (§9). |
| `sha256` is never a model input; it feeds only ids and receipts | audit §2 | We can hash source byte ranges or skip hashing. |
| Decoding from a true keyframe (H.264 IDR, AV1 key frame) is bit-identical, and `thread_count` doesn't change pixels. Single-view decode goes through swscale, which depends on the build and the CPU. | audit §4 | Pin `av==16.0.1` and the exact Pillow version, and decode on x86_64 only. L0 checks open-GOP streams per dataset. |
| Three different packages are all called `batch_utils`. Importing NVIDIA's runs `bootstrap`, which inserts vendor directories into `sys.path`. | `bootstrap.py:14-18` | The linker loads each repo under a unique module name. TempSeg/Captioner code and Rewarder code never share a process. |
| NVIDIA's prompts carry a **bare `<\|video_pad\|>`** (no vision start/end), and the chat template emits it as-is | audit §3 | vLLM's placeholder expansion for this shape is the most likely L0 mismatch. Gate C is a Phase-0 test. |
| Captioner: `max_new_tokens=512`, stop **only on `<\|im_end\|>`**. Its `generation_config` adds EOS id 248044, which vLLM would also stop on by default. TempSeg: 1,024 tokens. | audit §7 | Set `generation_config="vllm"` or explicit `stop_token_ids` for TempSeg and the Captioner. The Rewarder keeps NVIDIA's defaults. |
| `_CapturedInputs` needs only `.pop('video_metadata', default)`. NVIDIA's 32k guard reads `input_ids` length, which we can't know without the processor. | audit §3 | Enforce the guard after the fact using vLLM's `prompt_token_ids` (or a `max_model_len` rejection), mapped onto NVIDIA's error semantics. |
| Single-view caption frames are capped at **262,144 px** by `unified_input` (the 524,288 3-view cap there is unused). Human captions round-trip through PNG, which is lossless. `ready()` is never called. | `unified_input.py`, `models.py` | Write PNGs at compress level 0 to tmpfs, or skip them if L0 proves equivalence. Remove v2's `ready()` workaround. |
| Envelope checks also exist outside `media.py`: a 256 MiB file cap, profile and camera-role rules (3-view must be exactly `main/left_wrist/right_wrist`), and CFR for single-view robot captions | audit §7 | The index pass evaluates all of these and records the verdict per episode. |
| Captioning needs the segmenter's metadata (`fine_event_id`, `u2_context`). Segment status values are `complete` and `unsupported_target_frames`. `segmentation_provenance` has 5 values, plus degradation flags. | audit §7 | We persist the segmenter's full metadata. The output schema carries every value. |
| Memory: compose 3-view mosaics while streaming, so the peak is one frame triple. A 3-view caption frame union can reach ≈1.5 GB per episode, and dense reward holds ≈0.65 GB per 120 s episode. Stitching CPU is minor; the cost is in decode, `canonical_rgb`, resizes and PNG/JPEG round trips. | audit §5 | §6.7 budgets are revised, with a sliding caption cache. Gate D focuses on decode and image work. |
| The Rewarder's request building is pure functions. Terminal mode needs frames by index only. Camera order is **left, center, right** (TempSeg's is main, left wrist, right wrist). Rubric reads per-episode files through `decord`, which probably can't decode AV1. | `core.py`, `rubric_reward.py` | Terminal and dense batch across episodes. Rubric gets ephemeral per-episode files in a codec decord reads, with a rubric-specific L0 check (P2). |

### 1.2 vLLM v0.24.0 (`vllm_0240_verification.md`; released 2026-06-29)

| Finding | Consequence |
|---|---|
| `AsyncLLM` runs the model in a **separate engine-core process**. Preprocessing runs in the caller's process, in a thread pool with **1 worker by default**, sharing the caller's GIL. More workers are allowed only with the processor cache disabled (default 4 GB). A raw prompt dict passed to `generate()` is processed synchronously on the event loop, unless you first `await llm.renderer.render_cmpl_async([...])`. | Gate D/E (§9): measure preprocessing throughput. AsyncLLM paths must pre-render. Batch work defaults to the offline `LLM` with round-based micro-batching and a dedicated preprocessing pool (D14). |
| `AsyncLLM.from_engine_args` defaults to **128 sequences / 2,048 batched tokens**. The offline `LLM` uses 1,024 / 16,384 on H100. | Always set `max_num_seqs` and `max_num_batched_tokens` explicitly, or pass `usage_context=LLM_CLASS`. |
| `gdn_prefill_backend` defaults to **auto: FlashInfer on H100, Triton on L40S/A100** | Set `gdn_prefill_backend="triton"` on every stage, so certificates don't depend on the GPU. |
| Prefix caching is **off by default for hybrid models**. NVIDIA's `enable_prefix_caching=True` puts Qwen3.5 into an experimental "align" mode where hits and misses may give different numerics. #49064 (`max_num_seqs` vs available Mamba blocks) is present. | The Rewarder keeps NVIDIA's setting, and its certificate covers it. TempSeg and the Captioner leave it off. `max_num_seqs` comes from a per-GPU table. |
| `dt_bias` takes the model dtype, and FP32 checkpoint values are cast silently. The fix PR #44270 was closed unmerged. A `vllm.general_plugins` entry point loads in the engine-core and worker processes. | The D5 patch is feasible, and Gate A still decides whether it ships. |
| `mamba_ssm_dtype: float32` in the configs is picked up automatically, so the GDN state is FP32. Mixed-dtype safetensors load fine. `dtype=auto` turns the FP32-merged TempSeg into BF16. | Drop v2's manual `--mamba-ssm-cache-dtype` flag; keep an assertion. |
| Timestamps are computed from `frames_indices`/`fps`; `duration`/`total_num_frames` are accepted. #35909 and #37439 are already in 0.24.0. Per-request `mm_processor_kwargs` override engine-level ones. | Confirms the capture → request mapping. |
| `StructuredOutputsParams(regex=…)` uses xgrammar. `max_logprobs` defaults to 20. Per-request `seed` works; `n>1` children get `seed+i`. | Confirms the Rewarder interface. Rubric certification is distributional. |
| LoRA covers the GDN projections. Max rank defaults to 16, and there is no FP32 LoRA dtype. | The fallback needs `max_lora_rank=32`. D3 stands. |
| `VLLM_CACHE_ROOT` is enough, since vLLM redirects the Inductor and Triton caches. CUDA graphs are recaptured on every start. Sleep mode exists. | `flora-cache` Volume plus `VLLM_CACHE_ROOT`. Cold start always includes graph capture. |
| The OpenAI server can carry per-request `frames_indices`/`fps`, but the metadata is **per request**, so 3-video requests with different frame plans break. The server also applies the chat template itself. | That fallback (Gate D option c) is weak; we deprioritize it. |
| Two engines on one GPU work but must start one after the other. No in-tree test covers two AsyncLLMs in one process. | The fused H200 online profile is experimental, with two processes preferred. |
| transformers 5.8.1 needs `huggingface_hub>=1.5,<2.0`; vLLM has no direct pin | `huggingface_hub>=1.29,<2.0` works in every image. |

### 1.3 Modal (`modal_verification.md`; modal 1.6.2)

| Finding | Consequence |
|---|---|
| In deployed apps, **crashes, OOM kills and preemptions are retried indefinitely**. `modal.Retries` covers only exceptions raised by user code. | §8: per-shard attempt markers, quarantine after K starts, bisection to find the bad episode, and `FunctionCall.cancel()`. Use `memory=(request, limit)` so an OOM kills only that container. |
| `Volume.reload()` **fails while any file on the Volume is open**, and the mount looks empty to that container while it reloads. | Write parts in local `/tmp` and copy each one whole. Never hold files open on a Volume. Serialize reloads, and do them only when the worker is idle. |
| **Volume v2 is beta with possible data loss** | The state of record moves to the user's bucket (D6). Volumes hold weights, caches and the golden set (which can be rebuilt). |
| **v2 caps a directory at 262,144 files** | Hash-prefixed subdirectories (`ab/cd/…`). Receipts roll up into Parquet per 1k shards. |
| `cpu=` counts **physical cores (2 vCPU each)** and bills the request even when idle. `cpu=16` plus 64 GiB adds ≈$1.26/h, about +65% on an L40S. | Size `cpu` and `memory` from P0 measurements. The cost model includes CPU and memory. §6.7 uses consistent units (cores). |
| A factory-built app needs **global-scope class definitions**. Avoid `serialized=True`, which ties local and remote Python versions together. | Worker classes live undecorated at module level, and `build_app(cfg)` applies the decorators. |
| There is no `MODAL_GPU` env var. `with_options(gpu=)` takes **a single string**, with no fallback list. | Read the GPU from `nvidia-smi`. Fallback lists exist only at deploy time. |
| `OutputExpiredError` and `FunctionTimeoutError` subclass `TimeoutError`, so `.get(timeout=0)` handling can mistake them for "still pending" | Catch them first in the orchestrator. Covered by a unit test. |
| `@app.server` has `startup_timeout=30 s` by default. `max_concurrency` is a per-container cap, not a rate limit, and rejections are 429 or 503 (the docs disagree). | Raise `startup_timeout` to ≈900 s for 27B. Add our own per-token rate limiting. Clients retry 429 and 503. |
| Concurrent async inputs share one event loop and the async `@modal.enter` state | One engine per container with `@modal.concurrent`. Make `enter` async. |
| Better primitives: a **`nonpreemptible=True` CPU** function, `modal.Dict.put(skip_if_exists=True)` as a lease, **native Workspace/Environment budgets**, `Workspace.billing.rates()`, `experimental_spawn_map` (returns a FunctionCall), `modal.experimental.stop_fetching_inputs()`, `simulate_preemption()` | The orchestrator becomes a non-preemptible CPU function with a Dict lease, plus a stateless reconciler. Prices are read live. Environment budgets back up our cap. A bad container drains instead of crashing. Chaos tests use `simulate_preemption()`. |
| Also confirmed: spawn across the same app; `FunctionCall.from_id(...).get(timeout=0)`; 7-day result retention; self-respawn; `App.deploy()` with `add_local_python_source`; `with_options` variants can't hold `min_containers`; preemption gives SIGINT and 30 s before SIGKILL with `@modal.exit` running; `scaledown_window` 2 s–20 min (up to 3,600 s via `autoscaler.update`); 3 TiB ephemeral disk (512 GiB default write quota); R2/GCS mounts need `bucket_endpoint_url`; GPU snapshots alpha and single-GPU; prices, the 10/50/custom GPU caps and 1.15–1.75× region multipliers are current; billing is hourly, lags minutes, Team+ only | Unchanged parts of the design stand. |
| Still unmeasured: `/dev/shm` size under gVisor, per-container CPU/memory maximums, CloudBucketMount throughput, per-GPU-type sub-limits | P0 measurement items. |

---

## 2. Product definition

### Users and their jobs
| Persona | Job to be done | Entry point |
|---|---|---|
| Robot-learning researcher with a LeRobot dataset | "Add subtask captions and per-step rewards to my 2,000 teleop episodes by tomorrow, for under $X." | `flora plan` → `flora run` |
| Data-platform engineer at a lab | "Label 50k+ hours continuously, resume after failures, and know my $/hour." | config, sinks, observability, quotas |
| RL / eval engineer | "Score policy rollouts (outcome / dense / rubric) from my eval harness." | reward stages (v0.1), endpoint (v0.4) |
| Tool builder | "Call Flora from Python / a notebook / another pipeline." | SDK |

### Requirement presets (what "configurable to their requirements" means)

| Preset | Fidelity | Engines | Default GPUs | Use when |
|---|---|---|---|---|
| `exact` | Reference by construction | NVIDIA HF code (TempSeg/Captioner) + NVIDIA offline-LLM Rewarder | `H200` (both models resident) | Small or medium datasets, gold labels, or when you need to know nothing changed |
| `balanced` (default from v0.2) | Certified faithful | vLLM 0.24.0 paths | TempSeg `L40S`, Captioner `H100!`, Rewarder `H100!` | Most production labeling |
| `budget` | Uncertified until validated | `fast` options (FP8, B200 kernels) | `B200` / FP8 | Large exploratory runs; `flora validate` first |
| `interactive` | Certified faithful | AsyncLLM servers | Fused `H200` or per-stage | Online API (v0.4) |

`flora plan` combines the preset with the dataset index (total duration and episode mix), measured
per-stage rates (from `flora estimate`, or shipped defaults with error bars), live Modal prices
(`Workspace.billing.rates()`) and the workspace GPU cap. It prints 2–4 options with estimated cost,
wall-clock time, peak GPUs and certification status, then writes the chosen config.

### Goals (v1)
- **G1 Fidelity.** `exact` is identical to the reference. `balanced` must stay within measured noise, with certificates per (stage, pool, preset, engine version, GPU), scoped to the pools actually covered.
- **G2 One-command deploy into the user's account.** `pip install` → `flora init` → `flora prepare` → `flora plan` → `flora run`. No cluster, and it scales to zero.
- **G3 Cost control.**
  - Before the run: a calibrated estimate.
  - During the run: a hard cap from our projection, backed by a native Modal Environment budget.
  - After the run: per-stage $/video-hour, counting GPU, CPU and memory over each container's whole lifetime.
- **G4 Resumability.** Crashes, preemptions, OOMs and Ctrl-C lose at most one shard's uncommitted work. Poison pills are quarantined automatically, and `flora resume` is idempotent.
- **G5 Inputs and outputs.** In: LeRobot v2.1/v3.x and directories of MP4s. Out: JSONL/Parquet, plus LeRobot v3.1 `language_persistent` write-back to a new repo.
- **G6 Online API** (v0.4), with auth, input allowlists, rate limits and autoscaling.

### Non-goals (v1)
- Training Flora.
- A hosted multi-tenant service (that would be commercial use, §11).
- Non-Modal backends. The code stays Modal-agnostic so one can be added later.
- Robot control or safety monitoring.
- Certified results outside NVIDIA's envelope (D13).

### Success metrics
- **Fidelity.** `exact` reproduces NVIDIA's outputs on the golden set: identical for greedy stages, the same distribution for rubric. `balanced` is within 1.5× the oracle's self-noise, with boundary F1 at most 1 point below and terminal E[s] MAE ≤ 0.05.
- **Throughput.** `balanced` reaches ≥3 video-hours per GPU-hour on segment+caption on `H100!`. This target comes from the report's estimate band and will be replaced by P0 measurements. GPU utilization stays above 60% while busy.
- **Operations.**
  - A 1,000-episode job survives five forced preemptions via `simulate_preemption()`, one injected OOM episode and one orchestrator kill.
  - No episode is lost or duplicated.
  - The bad episode is quarantined within K attempts.
- **Developer experience.** A new user reaches first labels on a sample dataset in under 30 minutes, excluding the weight download. `flora plan` predicts cost within ±25% after calibration.

---

## 3. Core design decisions (ADR summary)

| # | Decision | Alternatives rejected | Why |
|---|---|---|---|
| D1 | **Linked mode.** NVIDIA modules are imported from the user's HF snapshot under **unique module names**, with **one process per NVIDIA repo family**. We replace only model invocation and media I/O. | Reimplementation; vendoring | Faithful by construction and license-clean. The name and `sys.path` collisions are contained. |
| D2 | **vLLM 0.24.0** for every vLLM path, with explicit engine args everywhere: `gdn_prefill_backend="triton"`, `max_num_seqs`, `max_num_batched_tokens`, `generation_config`/stop tokens, prefix caching per stage | Latest vLLM; SGLang; TRT-LLM | Matches NVIDIA's stack and fixes the GPU-dependent defaults. |
| D3 | **TempSeg is served as an FP32-merged checkpoint** (BF16 at runtime) as a plain model. HF+PEFT is the oracle. | Dynamic LoRA; HF batching | vLLM #52568 (NaN on hybrid GDN with LoRA) is open, and HF padding with recurrent state is unsafe. |
| D4 | **The Captioner gets a one-time BF16 conversion.** A_log/dt_bias stay FP32, and L1 checks it is bit-identical to the reference's in-memory weights. | Serving FP32 | Halves the size and is checkable. |
| D5 | **The FP32 `dt_bias` patch is gated (Gate A).** If shipped, it goes in through `vllm.general_plugins`. | Always/never | The upstream fix was closed unmerged. We pay the maintenance cost only if drift matters. |
| D6 | **The state of record is the user's bucket.** Job state is a manifest, shard part-files and receipts. They are written as whole files under hash-prefixed directories, through `CloudBucketMount` or the S3 API. Volumes hold weights, caches and the golden set. Without a bucket we fall back to Volume v2 and print a warning. | Volume v2 as the record; Queue/Dict; a database | v2 is beta and may lose data. Directories cap at 262k files. Reloads fail on open files. |
| D7 | **Range decode from the source.** An index pass replicates `probe` per episode: FFmpeg `average_rate` over the episode's packets, relative PTS, duration, CFR and every envelope check. Oracle runs and rubric use **stream-copy** per-episode cuts, made ephemerally. | A persistent lossless clip cache; re-encoded clips | Re-encoding changes the time base and color tags, and a cache is ≈PB at scale. |
| D8 | **Stage-separated GPU pools** for batch, chained per shard. Fused serving is experimental and online-only. | A fused batch worker | Different GPUs per stage; independent scaling. |
| D9 | **Presets `exact` / `balanced` / `budget` / `interactive`.** `run` refuses uncertified combinations unless given `--allow-uncertified`. | Free-form flags | Makes the trade-off explicit (§2). |
| D10 | **App factory.** Worker classes are defined at module scope and decorated inside `build_app(cfg)`. `--emit` writes the equivalent script. No `serialized=True`. | Static app; serialized closures | Modal requires global-scope definitions, and this keeps the deploy version-independent. |
| D11 | **The config hash** covers the framework logic version, NVIDIA code revisions plus the bootstrap manifest SHA, prompt-pool SHA, model revisions, derived-weight receipts, preset/engine/version/patches, all engine args, sampling and stop tokens, mm_kwargs, and decode pins (av/Pillow/arch). It **excludes GPU type**, which goes into provenance instead. | — | Fixes invalidate old results, and GPU fallback doesn't break resume. |
| D12 | **CPU work runs in a process pool** with budgets: decode, `canonical_rgb`, mosaics, resizes, JPEG/PNG round trips and plan building. vLLM preprocessing gets its own thread workers with the processor cache off, or is pre-rendered. `cpu` and `memory` are sized from measurements. | Threads in the engine process; a fixed cpu=16 | GIL contention; CPU is billed whether used or not. |
| D13 | **Long-episode policy `envelope: warn`** (the default). Out-of-envelope episodes run and are marked `certified:false`. `enforce` skips them; `ignore` removes the flag. | — | The data gets labeled and its status stays honest. |
| D14 | **Batch inference = offline `vllm.LLM` with a round-based micro-batcher.** `apredict` and caption/reward calls enqueue futures. When every live episode is blocked, or a size or timeout threshold is hit, one `llm.generate` call runs in a worker thread and resolves the futures. **Online = AsyncLLM with `render_cmpl_async` pre-rendering.** Gate E confirms the batch default. | AsyncLLM for batch | The offline LLM is NVIDIA's validated wrapper and defaults (1,024/16,384), and gives deterministic batches. AsyncLLM preprocesses on the event loop and has different defaults. |
| D15 | **Poison pills and OOM.** Shard attempt markers. After K=3 starts the shard is quarantined and bisected into single-episode shards. A repeatedly failing episode goes to `errors/`, and its stale FunctionCalls are `cancel()`ed. `memory=(request, limit)`. On a fatal engine error, `stop_fetching_inputs()` drains the container. | Relying on `modal.Retries` | Modal retries crashes and OOMs forever, and an OOM takes down the container's concurrent shards. |
| D16 | **A small, explicit certified matrix.** Initial certificates: {segment, caption, terminal, dense} × {`H100!`, plus `L40S` for segment} × covered pools. Anything else is uncertified until a certificate exists. Certificates are plain JSON metrics, and **community-contributed certificates** are accepted through PRs once the golden-set hash matches. | Certifying everything | Certification cost would grow combinatorially. |

---

## 4. Architecture

```
                         ┌──────────────────────── user machine ────────────────────────┐
                         │  flora CLI / SDK: init · prepare · index · plan · run · status │
                         │  (deps: modal, pydantic, typer, pyarrow, rich)                │
                         └───────────────┬──────────────────────────────────────────────┘
                                         │ App.deploy() / spawn / FunctionCall.from_id
┌────────────────────────────────────────▼───────────────────────── Modal workspace (user-owned) ─┐
│ [orchestrator] CPU, nonpreemptible, Dict lease (skip_if_exists), ≤23h → continuation;            │
│                stateless reconciler also callable from `flora status`                           │
│   index → plan shards → spawn per stage → chain per shard → budget/GPU-cap gate → quarantine     │
│        │                 │                    │                     │                           │
│        ▼                 ▼                    ▼                     ▼                           │
│  [index] CPU       [segment] GPU        [caption] GPU         [reward] GPU                      │
│   per-episode       L40S (balanced)      H100! (balanced)      H100! (rubric H100!:2)           │
│   probe-equivalent  offline LLM +        offline LLM +         offline LLM, NVIDIA core.py      │
│   receipts          micro-batcher ←      micro-batcher ←       (terminal/dense batched)         │
│   (rate, PTS, CFR,  NVIDIA segmenter     NVIDIA caption plans  rubric: ephemeral stream-copy    │
│   envelope, sync)   via apredict         + capture processor   per-episode files                │
│                     CPU proc pool        CPU proc pool         CPU proc pool                    │
│   exact preset: [GoldenWorker-style ReferenceWorker] H200, NVIDIA HF code via CachedSession,     │
│                 ephemeral stream-copy clips on local disk                                        │
│        │                 │                    │                     │                           │
│        └───── whole-file writes: parts, receipts, attempts, errors ─┘                           │
│                                   ▼                                                             │
│   USER BUCKET (state of record):  flora/<job>/{manifest.parquet, index/, parts/ab/cd/…,          │
│                                   receipts/ab/cd/…, attempts/…, errors/…, containers/…, report/} │
│   Volume `flora-weights` · Volume `flora-cache` (VLLM_CACHE_ROOT) · Volume `flora-golden`        │
│   Source data: CloudBucketMount (read-only, bucket_endpoint_url for R2/GCS) or staged copy       │
│                                                                                                 │
│ [compose] CPU → final Parquet/JSONL in sink; optional LeRobot v3.1 write-back to a NEW repo      │
│ [server] (v0.4) @app.server(startup_timeout≈900s) FastAPI + AsyncLLM (pre-rendered inputs)      │
│ [validate] golden builder (H200, CachedSession) · certifier (L0–L3) · certificate registry      │
│ Backstop: Modal Environment budget = job budget × 1.2                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data flow for one shard (`balanced` preset)
1. **index** (CPU, once per dataset revision), per episode and camera:
   - packets in the episode's range → FFmpeg-formula `average_rate`;
   - relative PTS, `duration_us`, CFR, size and codec, keyframe at the start (oracle-eligibility);
   - 3-view sync and camera-role mapping;
   - the envelope verdict (all checks from §1.1);
   - exact source frame timestamps for write-back.

   The output is Parquet under hash-prefixed directories.
2. **segment** (GPU):
   - NVIDIA's segmenter is built exactly as `pipeline.py:47-49` (3-view) or `adapter.production_segmenter` (other episodes) would build it, with our `MicroBatchPredictor` (§6.3).
   - The process pool range-decodes, composes mosaics while streaming, regularizes and returns uint8 frames.
   - Rounds: grid windows for all live episodes, then up to 2 retry rounds.
   - Outputs, including the full segment metadata, go to `/tmp/part-k.jsonl`, which is copied to the bucket when it reaches its size limit.
3. **caption** (GPU, after the segment receipt):
   - NVIDIA's caption plans, with frames from a sliding cache at caption sampling and inputs built through the capture processor (PNG step at compress level 0 on tmpfs).
   - Batched across episodes in rounds.
   - Stop tokens are `<|im_end|>` only, with 512 max tokens.
4. **reward** (GPU, independent):
   - Requests come from NVIDIA's `core` (collage order left/center/right) and are batched.
   - Parsing uses `core`.
   - Rubric writes ephemeral per-episode files.
5. **compose** (CPU): dedupe parts by (episode, stage), write the final schema, and optionally write back to LeRobot.

In the `exact` preset, steps 2 and 3 are replaced by a `ReferenceWorker` that runs NVIDIA's own
`pipeline.segment`/`caption_*` functions through a `CachedSession`, on ephemeral stream-copy clips.
Its orchestration, receipts and outputs are the same.

---

## 5. Repository layout

```
flora-deploy/
├── LICENSE (Apache-2.0) · NOTICE.md (Flora = NVIDIA OneWay Noncommercial; research use only)
├── pyproject.toml               # CLI extras only; GPU deps live in Modal images
├── src/flora_deploy/
│   ├── config/   schema.py · presets.py · profiles.py · pins.py · hashing.py
│   ├── nv/                      # linked-mode boundary — only package importing NVIDIA code
│   │   ├── snapshot.py          # locate/verify snapshots + manifests
│   │   ├── linker.py            # import each repo's batch_utils under a unique name; process guard
│   │   ├── capture.py           # CaptureProcessor (+ _CapturedInputs.pop)
│   │   ├── session.py           # CachedSession for exact preset / oracle
│   │   └── compat.py            # segmenter constructions (3-view inline + production_segmenter), caption
│   │                            # plan calls, failure-semantics mapping, 32k guard mapping
│   ├── media/    sources/{base,lerobot_v21,lerobot_v3,mp4_dir}.py · index.py (probe-equivalent)
│   │             rangedecode.py · workers.py (process pool) · cuts.py (stream-copy ephemeral clips)
│   ├── engines/  base.py · offline_llm.py (+ microbatch.py) · async_llm.py (online) · hf_reference.py
│   │             gdn_audit.py · patches/dt_bias_fp32.py (gated)
│   ├── stages/   segment.py · caption.py · reward.py · reference.py (exact preset) · types.py
│   ├── prep/     download.py · convert_captioner.py · merge_tempseg.py
│   ├── jobs/     manifest.py · store.py (bucket/Volume backends, whole-file, hash-prefix) ·
│   │             orchestrator.py · reconciler.py · attempts.py (quarantine/bisect) · budget.py
│   ├── planner/  rates.py (measured/default rates + error bars) · options.py (`flora plan`)
│   ├── modal_app/ workers.py (undecorated module-scope classes) · factory.py · images.py · emit.py
│   ├── serving/  (v0.4) app.py · routes.py · auth.py · allowlist.py · ratelimit.py
│   ├── validate/ golden.py · layers.py · metrics.py · certificate.py · registry.py
│   ├── export/   jsonl.py · parquet.py · lerobot_v31.py
│   ├── observe/  events.py · report.py · qa.py · profiler.py
│   ├── sdk.py · cli.py
├── certificates/                # community/official certificate JSON (metrics only, no NC content)
├── tests/{unit,nvlink,modal}/
├── examples/                    # rollouts_terminal_reward, human_mp4, lerobot_3view (flora.yaml)
└── docs/                        # quickstart · presets & planning · certification & coverage ·
                                 # envelope policy · licensing · deviations-from-card · measurements
```

Rules:
- Only `nv/` imports NVIDIA modules, and it never mixes repo families in one process.
- `stages/` and `engines/` never import `modal`.
- Pool workers exchange plain data and uint8 arrays through shared memory. If `/dev/shm` under gVisor turns out to be too small, they fall back to pickled bytes (a P0 measurement).

---

## 6. Key interfaces (sketches)

### 6.1 Engine protocol
```python
class Engine(Protocol):
    name: str                        # "vllm-0.24.0-offline[+dtbias]" | "vllm-0.24.0-async" | "hf-ref-5.8.1"
    async def generate(self, req: VllmRequest, sp: Sampling) -> Completion: ...
    def audit(self) -> dict: ...     # gdn_audit, full engine args, GPU (nvidia-smi), versions

@dataclass(frozen=True)
class VllmRequest:
    prompt: str                                  # templated text (bare <|video_pad|> per NVIDIA)
    videos: list[tuple[np.ndarray, dict]]        # uint8 (T,H,W,3); {total_num_frames, fps,
                                                 #  frames_indices, duration, do_sample_frames: False}
    mm_kwargs: dict                              # per-request override (pins.py)
    key: str

@dataclass(frozen=True)
class Sampling:
    max_tokens: int; temperature: float = 0.0; n: int = 1; top_p: float = 1.0; top_k: int = 0
    regex: str | None = None; logprobs: int | None = None; seed: int | None = None
    stop_token_ids: tuple[int, ...] | None = None    # Captioner/TempSeg: (<|im_end|>,) only
```

### 6.2 Capture processor
```python
class CaptureProcessor:
    def __init__(self, real): self._real, self.captured = real, None
    def __getattr__(self, k): return getattr(self._real, k)      # video_token, apply_chat_template, …
    def __call__(self, *, text, videos, video_metadata, **kw):
        assert kw.get("do_sample_frames") is False
        self.captured = VllmRequest(prompt=text[0], videos=[(to_uint8(v), md_to_dict(m))
                                    for v, m in zip(videos, video_metadata, strict=True)], …)
        return _CapturedInputs()       # only .pop('video_metadata', default) allowed; else raise
```
L0 replays the captured args through the real processor and compares digests with the oracle. It also
compares vLLM's `prompt_token_ids` and grid with HF's for the same request. Gate C is the bare
`<|video_pad|>` expansion check.

### 6.3 Micro-batching predictor (batch default, D14)
```python
class MicroBatcher:
    """Collects requests from many episodes' coroutines; when every live episode is blocked (or
    max_batch / max_wait hit), runs ONE llm.generate in a worker thread and resolves futures."""
    def __init__(self, llm, max_batch, max_wait_s): ...
    async def submit(self, req: VllmRequest, sp: Sampling) -> Completion: ...
    def episode_started(self): ...; def episode_finished(self): ...     # liveness accounting

class MicroBatchPredictor:                        # handed to NVIDIA's segmenter
    async def apredict(self, request):
        windows = [self.nv.window_from_payload(p, self.source) for p in request["windows"]]
        plans   = [self.nv.temporal_frame_plan(w, self.index.pts_us) for w in windows]
        frames  = await self.cpu.run(decode_compose_regularize, self.index, plans, self.layout)
        reqs    = [self.nv.build_temporal_request(w, frames, self.pool, self.instruction) for w in windows]
        outs    = await asyncio.gather(*(self.batcher.submit(r, TEMPSEG_SAMPLING) for r in reqs))
        self.nv.check_context_budget(outs)        # 32k guard from prompt_token_ids → NVIDIA error type
        return self.nv.fine_response(request, windows, outs, self.bound)
```
Rounds per shard: the grid round, then ≤2 retry rounds for the episodes that need them, then the
caption round(s) in the caption worker. While a round is generating, the CPU pool already prepares the
next shard's grid inputs (pipelining across `@modal.concurrent` inputs).

### 6.4 Stage contract (Modal-agnostic)
```python
async def segment_episode(ep: EpisodeIndex, engine: Engine, cpu: CpuPool) -> StageRecord
async def caption_episode(ep: EpisodeIndex, segments: StageRecord, engine, cpu) -> StageRecord
async def reward_episode(ep: EpisodeIndex, mode, task: str, rubric: list | None, engine, cpu) -> StageRecord
def reference_episode(ep: EpisodeIndex, session: CachedSession, clips: Path) -> StageRecord   # exact preset
```

### 6.5 `CachedSession`
`load(kind)` returns an already-loaded `(model, proc)` built once with NVIDIA's own `Session.load`,
one inner `Session` per kind. `close()` does nothing per call; the real close happens at container exit.
On H200, TempSeg (~20 GB) and the Captioner (~55 GB) stay resident. The fallback is two passes.

### 6.6 Output schema `flora_labels_v1`
One record per episode per stage. Records are appended in local part-files and copied whole.
```json
{"schema":"flora_labels_v1","episode_id":"…","stage":"caption","certified":true,
 "segments":[{"event_id":"event_0003","fine_event_id":"…","start_s":12.35,"end_s":15.80,
              "caption":"…","status":"complete|unsupported_target_frames",
              "segmentation_provenance":"<one of NVIDIA's 5 values>","degradation":{…},
              "nv_metadata":{…}}],
 "provenance":{"models":{…},"engine":"vllm-0.24.0-offline","preset":"balanced",
               "certificate":"cert_…","gpu":"H100","config_hash":"…","logic_version":"0.2.0",
               "source":{"main":{"uri":"…","range_s":[…]}},"framework":"flora-deploy 0.2.0"},
 "audit_ref":"…"}
```
Rewards add `terminal{score, expected_score, probabilities[5], rationale?}`, `dense{segments, per_step}`
and `rubric{scores, total, samples, medoid_index}`. `certified:false` comes with a reason:
`outside_reference_envelope`, `uncertified_stack`, `uncovered_pool` or `oracle_ineligible_source`.

### 6.7 Resource budgets (per GPU container; units are cores and GiB)
These are the defaults before P0, and P0 replaces them.
- **`max_concurrent_decodes = 4`.** 3-view mosaics are composed while frames stream, so the peak is about one frame triple per decode, plus the codec's buffers.
- **`max_inflight_episodes = 32`** for segmenting. TempSeg 3-view frames are ≤160k px, about 0.5 MB, so a 120 s episode holds ≈0.25 GB.
- **The caption sliding cache is capped at 2 GiB.** The 3-view frame union can reach ≈1.5 GB per episode.
- **Dense reward holds ≈0.65 GB** per 120 s episode.
- **Memory.** `memory = (request, limit)`, where request = base + decodes × peak_decode + inflight × held + cache + headroom, and limit = request × 1.25. OOM kills only this container (D15).
- **CPU.** `cpu` is in **physical cores**. It starts at 6 for L40S TempSeg and 8 for H100 caption, and is then set from measured decode-core demand. The cost model includes CPU and memory at Modal rates.

---

## 7. Modal topology

### Images (pinned, x86_64)
| Image | Contents | Used by |
|---|---|---|
| `img-cpu` | python 3.12, `av==16.0.1`, ffmpeg, exact Pillow, pyarrow, `huggingface_hub>=1.29,<2.0`, `flora_deploy` | orchestrator, index, compose, prep |
| `img-vllm` | `vllm==0.24.0`, `transformers==5.8.1`, `qwen-vl-utils==0.0.14`, `decord==0.6.0`, `av==16.0.1`, exact Pillow, `huggingface_hub>=1.29,<2.0`, `flora_deploy`, optional dt_bias plugin | segment, caption, reward, server |
| `img-hfref` | NVIDIA's TempSeg/Captioner `requirements.txt` exactly + `av==16.0.1` | `exact` preset, golden builder |

### Storage
- **User bucket (state of record).** Job files are written whole, under hash-prefixed directories.
- **`flora-weights` (v2).** Snapshots and derived weights, each with a receipt.
- **`flora-cache`.** `VLLM_CACHE_ROOT`, keyed by image digest and model hash. Mounted read-mostly. Updates are committed only from a dedicated warmup job, to avoid reload-with-open-files.
- **`flora-golden`.** Oracle intermediates, which can be rebuilt.
- **No mutable job state on Volumes when a bucket is configured.** Without a bucket, job state goes on a Volume v2 with these rules:
  - parts are written in `/tmp` and copied whole;
  - writers never `reload()` while files are open;
  - only the orchestrator reads, and it reloads between scheduling ticks.

### Functions and classes
Workers are defined undecorated at module scope. `build_app(cfg)` applies the decorators.

| Object | Kind | GPU | Notes |
|---|---|---|---|
| `orchestrator.run` | fn, **nonpreemptible** CPU | — | Dict lease; continuation before 23 h; catches `OutputExpiredError`/`FunctionTimeoutError` before `TimeoutError` |
| `reconciler.tick` | fn | — | stateless; rebuilds status from receipts; revives the orchestrator; optional cron |
| `index.run_shard` | fn | — | probe-equivalent receipts |
| `SegmentWorker` | cls | `["L40S","A100-80GB"]` | offline LLM + micro-batcher; `@modal.concurrent(max_inputs=2–3)`; async `@modal.enter`; `scaledown_window=600` |
| `CaptionWorker` | cls | `["H100!"]` | BF16 Captioner; `limit_mm_per_prompt={"image":0,"video":3}`; `max_model_len=32768`; stop `<\|im_end\|>` |
| `RewardWorker` | cls | `["H100!"]` | NVIDIA engine args + explicit `gdn_prefill_backend="triton"`, prefix caching on (as NVIDIA) |
| `RubricWorker` | cls | `"H100!:2"` | TP=2; NVIDIA settings; ephemeral per-episode files |
| `ReferenceWorker` | cls | `"H200"` | `exact` preset and golden builder; img-hfref; CachedSession; stream-copy clips |
| `compose.run` | fn | — | final outputs to sink |
| `Server` (v0.4) | `@app.server(startup_timeout≈900)` | per profile | AsyncLLM + `render_cmpl_async`; Proxy Tokens; allowlist; our own rate limit |

Every GPU container:
- writes `containers/<id>.json` at enter and periodically after: GPU from `nvidia-smi`, HBM, image digest, start and heartbeat times;
- runs `gdn_audit` after the engine loads;
- calls `stop_fetching_inputs()` and exits on an unrecoverable engine error.

---

## 8. Job lifecycle, fault tolerance, budget

**States.** `indexed → planned → running → composing → done | failed | budget_stopped | cancelled`. They are always derived from bucket files; `job.json` is a cache.

**Planning (`flora plan` → manifest).**
- Episodes are grouped by source file and packed into shards of `target_gpu_minutes` (default 10).
- Packing uses measured per-stage rates, or defaults with error bars.
- Envelope and oracle-eligibility verdicts carry through.

**Orchestrator.**
- A single instance is enforced by a Dict lease (`skip_if_exists`, with TTL renewal). It is non-preemptible.
- It continues itself before 23 h. The reconciler can rebuild all state from receipts at any time.

**Chaining.**
- In-flight work per stage stays ≤ `max_containers × max_inputs`, and the GPU total stays ≤ `max_gpus`.
- A caption shard becomes eligible once its segment receipt exists. Reward shards become eligible after indexing.

**Writes and resume.**
- A worker writes a part locally and copies it whole to `parts/<stage>/<cfg_hash>/<ab>/<cd>/<shard>-<attempt>-<k>.jsonl`.
- On restart, it reads the previous attempts' parts before opening anything and skips the episodes they contain.
- `compose` dedupes by (episode, stage).
- The receipt is written last.

**Attempts and quarantine (D15).**
- Each shard start writes `attempts/<shard>/<n>.json`.
- At K=3 starts without a receipt, the orchestrator:
  1. `cancel()`s outstanding calls;
  2. marks the shard quarantined;
  3. re-plans its unfinished episodes as single-episode shards.
- An episode whose single-episode shard also reaches K is written to `errors/` with the last diagnostics and skipped.
- This bounds Modal's infinite crash/OOM retries.

**Preemption.** At most one shard's uncommitted tail is lost (≤10 GPU-minutes by default). On SIGINT, `@modal.exit` flushes and copies the current part within the 30 s grace period.

**Budget.**
- Spend = Σ container lifetimes × (GPU + CPU + memory) rates, using `Workspace.billing.rates()`.
- Projection = spend + in-flight + remaining work × measured rate.
- At `budget_usd`:
  1. stop spawning;
  2. drain in-flight shards;
  3. scale pools to zero;
  4. mark the job `budget_stopped`.
- Backstop: `flora run` sets a native Modal Environment budget at job budget × 1.2. Hourly billing reports reconcile afterwards.

---

## 9. Validation and certification

**Golden set (P0).**
- About 100 oracle-eligible episodes: ≤120 s, starting on a keyframe so stream-copy cuts work, and passing every envelope check.
- Stratified by pool and profile. Hard cases (retries, pauses, handovers) are included on purpose.
- The Rewarder set covers terminal, dense and rubric.
- The oracle runs NVIDIA's code on `H200` via `CachedSession` on stream-copy clips and stores every intermediate.

**Coverage, stated honestly.**
- Stand-ins:
  - WGO-Bench (single-view);
  - an egocentric human set;
  - ALOHA-style bimanual sets for 3-view. ALOHA has 4 cameras: `main=cam_high`, and `cam_low` is dropped. This is recorded and revisitable.
- Dex and Tianji likely stay uncertified until users run `flora golden` on their own data.
- Datasets whose episodes don't start on keyframes can't be oracled directly. They can still run `balanced`, under a certificate from the same pool on other data, and are labeled `oracle_ineligible_source`.

**Self-noise.** The oracle runs twice and on two GPU types. Thresholds are set relative to this noise.

| Layer | Check | Pass rule |
|---|---|---|
| L0 input parity | (a) index-pass values == `probe` on the stream-copy per-episode file (rate, PTS, duration, CFR); (b) range-decoded frames == NVIDIA `decode_selected` on that file; (c) captured args → real-processor digests == oracle; (d) vLLM `prompt_token_ids` + grid == HF for the same request (**Gate C: bare `<\|video_pad\|>`**) | bit-exact |
| L1 weight parity | `A_log`/`dt_bias` dtype and bits in the engine; converted Captioner == reference in-memory BF16; merged TempSeg ≠ base; no NaN on the longest windows; engine args audit (triton GDN, stop tokens, prefix-cache mode) | bit-exact / boolean |
| L2 distribution parity | teacher-forced greedy agreement; mean/p99 \|Δlogprob\| | ≤1.5× oracle cross-GPU noise; near-ties excluded |
| L3 task metrics | TempSeg: parse success, window EM, boundary F1 @2 ticks, seams, retry counts, provenance distribution. Caption: EM → embedding → LLM-judge. Reward: KL / E[s] MAE, dense agreement, rubric distribution distance. | ≤1.5× noise; F1 ≤1 pt below |

**`exact` preset check.** Not a certification: a regression test showing that our orchestration around NVIDIA's code reproduces the oracle bit-for-bit on the golden set, for greedy stages.

**Gates.**
- **A (D5):** ship the `dt_bias` patch only if drift exceeds the noise floor.
- **B (D3):** if the merged TempSeg fails, use vLLM-LoRA only if #52568 doesn't reproduce. Otherwise HF-PEFT at batch 1.
- **C (P0):** vLLM's expansion of a bare `<|video_pad|>` matches HF. If not, fix it in the capture layer: add the tokens NVIDIA's HF processor effectively produces, prove the equivalence and document it.
- **D (P0):** where CPU time goes (decode, `canonical_rgb`, resize, JPEG/PNG, vLLM preprocessing), and the CPU cores and processes needed per GPU.
- **E (P0/P1b):** offline-LLM micro-batching vs AsyncLLM with `render_cmpl_async`, compared on throughput, GPU utilization and parity. Default is offline.

**Registry.** `certificates/*.json` in the repo, and `flora certify --submit` opens a PR. A certificate is accepted only if its golden-set hash and metrics are reproducible by maintainers.

---

## 10. Phased roadmap

Effort assumes one senior engineer. GPU spend is estimated at Modal list prices.

### P0: Reference on Modal, golden set, measurements (≈2.5 weeks, ≈$200–350)
- [ ] Repo skeleton, Apache-2.0, NOTICE, CI; `pins.py` (av 16.0.1, Pillow exact, hub `>=1.29,<2.0`).
- [ ] `prep/download.py` → `flora-weights` with manifest verification.
- [ ] `nv/linker.py` (unique module names, process separation) and `nv/session.py` (`CachedSession`).
- [ ] `media/index.py` (probe-equivalent per episode) and `media/cuts.py` (stream-copy clips, keyframe-start check), for MP4 dirs and LeRobot v2.1/v3.
- [ ] `ReferenceWorker` on H200. Rewarder via NVIDIA's own scripts' functions (offline `LLM`).
- [ ] Golden datasets chosen; ~100 episodes with every intermediate; oracle ×2 runs on 2 GPU types.
- [ ] **Gate C test**: bare `<|video_pad|>` in vLLM 0.24.0 vs HF, on 20 golden requests per pool.
- [ ] **Measurement milestone → `docs/measurements-p0.md`.** Per pool:
  - prompt and **output token lengths**;
  - CPU-seconds per episode by component (decode H.264/AV1, `canonical_rgb`, mosaic, resize, JPEG/PNG, HF processor);
  - peak RSS;
  - `/dev/shm` size;
  - CloudBucketMount read throughput;
  - vLLM cold start (load, compile, graphs) with and without `flora-cache`;
  - oracle wall-clock per video-minute;
  - L40S/H100/H200 availability.
- **Exit:** golden set and noise report; measurements doc; Gates C and D decided; first real cost model.

### P1a: Job system + the `exact` preset → **release v0.1** (≈4 weeks, ≈$150–300)
- [ ] `config/` (schema, presets, D11 hashing); `flora init` templates.
- [ ] `jobs/`:
  - bucket store with whole-file writes and hash-prefixed dirs;
  - manifest;
  - orchestrator (lease, continuation, chaining, GPU cap);
  - reconciler;
  - attempts, quarantine and bisection;
  - budget (lifetime receipts, live rates, Environment-budget backstop).
- [ ] `modal_app/` (module-scope workers, factory, `--emit`); CLI `prepare/index/plan/deploy/run/status/resume/results/report/cancel`; SDK `submit/Job`.
- [ ] `stages/reference.py`: the `exact` preset for segment+caption via `ReferenceWorker`. Terminal and dense rewards on NVIDIA's offline-LLM recipe, batched per shard. **These are NVIDIA's own code paths, so they're exact by construction.**
- [ ] `planner/` with `flora plan` (shipped default rates from P0 + `flora estimate --sample 20`).
- [ ] `export/` JSONL and Parquet.
- [ ] Chaos suite: `simulate_preemption()` ×5, injected OOM episode, orchestrator kill, budget stop/resume, lease contention.
- **Exit (v0.1):** `exact` regression-tests bit-identical vs the oracle; a 1,000-episode job passes the chaos suite; `flora plan` within ±25% of the actual cost after calibration; docs + quickstart.

### P1b: Certified `balanced` preset → **release v0.2** (≈6 weeks, ≈$300–600)
- [ ] `prep/convert_captioner.py` (L1 bit-equality) and `prep/merge_tempseg.py`; `gdn_audit`.
- [ ] `engines/offline_llm.py` + `microbatch.py` with explicit engine args (D2); per-GPU `max_num_seqs` / `max_num_batched_tokens` tables; stop tokens.
- [ ] `nv/capture.py` and `nv/compat.py` covering every builder path. NVIDIA's failure semantics and the 32k-guard mapping.
- [ ] `stages/segment.py`, `stages/caption.py` (sliding cache), `stages/reward.py` (pre-batched NVIDIA requests in the offline LLM).
- [ ] `media/rangedecode.py` + `workers.py` (process pool; streaming mosaics; budgets from P0).
- [ ] Gates A, B and E; certify the D16 matrix; publish certificates.
- **Exit (v0.2):** the certified matrix is published; GPU utilization >60% busy; measured `balanced` $/video-hour vs `exact` is in the run report and the docs.

### P2: Throughput, breadth, QA → **v0.3** (≈3 weeks, ≈$200–400)
- [ ] Tune concurrency, decode pools and engine tables from profiles; CPU/memory right-sizing.
- [ ] Optional `color: fast` decode (torchcodec/NVDEC); certification required.
- [ ] LeRobot v3.1 write-back into a new repo, validated with LeRobot's validator.
- [ ] Observability: OTel/Datadog; `flora report` showing a CPU-vs-GPU-bound indicator; stratified QA sample → overlay MP4s + LeRobot-visualizer dataset.
- [ ] Rubric reward (`H100!:2`, ephemeral per-episode files in a codec decord reads, rubric-specific L0) + distributional certificate.
- [ ] B200 certification attempt for the Captioner; certificate-registry submission flow.

### P3: Online API + scale → **v0.4** (≈4 weeks, ≈$300–500)
- [ ] `serving/`:
  - `@app.server(startup_timeout≈900)` with AsyncLLM and pre-rendered inputs;
  - endpoints `/v1/segment`, `/v1/caption`, `/v1/reward/*`, `/v1/annotate` (spawn-and-poll);
  - Proxy Tokens;
  - **input allowlist** (bucket keys / configured domains; no arbitrary fetch);
  - size, duration and frame limits;
  - **our own per-token rate limits**; clients retry 429/503.
- [ ] Per-stage servers by default. The fused H200 profile is experimental: two processes, engines started one after the other.
- [ ] SDK `Client`; p50/p95 latency per profile; opt-in GPU snapshots (single-GPU).
- [ ] Scale test at the largest size the plan tier allows; Enterprise-quota guidance doc.

### P4: Extensions (ongoing)
- `budget` preset items (FP8, SM100 kernels, MTP), each certified.
- vLLM upgrade track (LoRA once #62224 ships and #52568 is fixed).
- Baseline reward plug-ins (Robometer, TOPReward).
- RLDS/MCAP/Zarr sources; a non-Modal backend.

---

## 11. Licensing and compliance plan

- **Repo:** Apache-2.0, with no NVIDIA code, prompts or weights. Linked mode imports them from the user's HF download and binds to them by name.
- **Re-expressed NC orchestration logic, flagged for counsel review:**
  - the async predictor vs `pipeline.Predictor`;
  - the 3-view segmenter construction from `pipeline.py:47-49` (profile and policy constants);
  - the caption-loop glue.

  We call NVIDIA functions everywhere we can and keep the re-expressed parts minimal and documented.
- **Derived weights** (BF16 Captioner, merged TempSeg) are built only in the user's Volume and never published. The docs explain the redistribution obligations: OneWay license, LICENSE.pdf, notices, derivative marking.
- **Required acknowledgment:** `license_ack: research-only`. A first-run notice covers:
  - research/evaluation only;
  - a paid hosted service counts as commercial use;
  - not for robot control or safety;
  - **whether labels are restricted is unresolved; get legal advice for commercial use of labels.**
- **The certificate registry holds metrics only.** Contributors must not include NVIDIA prompts or outputs that would amount to redistributing them. Captions in golden sets stay in users' own storage.
- **Accepted risk:** NVIDIA deletes or rewrites the HF repos. We can't mirror them.
- **Name collision:** an unrelated "FLORA" reward-model paper exists (arXiv 2605.22123). The project name should be distinct (§14).
- **Provenance on every row.** *Not legal advice.* Counsel reviews before 1.0.

---

## 12. Testing strategy

| Tier | Runs | What |
|---|---|---|
| Unit (CPU, no network) | every PR | config/hash invalidation; manifest packing; whole-file store with hash-prefix; resume dedupe; attempts → quarantine → bisect; budget math; orchestrator state machine with fake Modal, **including the `TimeoutError` subclass trap**; lease; micro-batcher liveness and round semantics; capture processor; output schema; resource calculator |
| NV-link (CPU; NVIDIA code, no weights) | every PR (private cache) | unique-name imports per repo; manifests; prompt-SHA reproduction for every pool; `apredict` + micro-batcher with canned JSON == NVIDIA sync path, including retry rounds and failure semantics; 3-view inline construction parity; CachedSession duck-typing; index-pass == `probe` on fixtures (H.264, AV1, concatenated v3 fixture vs stream-copy cut) |
| Modal smoke (GPU, ~$5) | nightly / manual | 3 tiny episodes per preset; L0/L1 |
| Certification (GPU, ~$50–150) | any engine/image/pin/logic bump; before release | L0–L3 → certificate |
| Chaos & load | each release | `simulate_preemption()`, OOM injection, orchestrator kill, lease contention, budget stop/resume, memory soak, bucket write throughput |

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Container-derived probe values differ on concatenated files (`average_rate`, CFR) | Per-episode FFmpeg-formula index; L0(a) per dataset |
| Bare `<\|video_pad\|>` expands differently in vLLM | Gate C in P0; fix in the capture layer with a proven equivalence; `exact` preset unaffected |
| GPU-dependent defaults (GDN backend auto, AsyncLLM sequence and token limits, generation_config EOS) | D2 explicit args; L1 engine-args audit; certificates record full args |
| CPU/GIL bottleneck (decode, image ops, vLLM preprocessing) | D12 pool; offline micro-batching; Gate D/E; right-sized CPU |
| Modal's infinite crash/OOM retries | D15 attempts, quarantine, bisection, cancel, memory limits |
| Volume v2 data loss / reload failures / 262k-file directories | D6 bucket as state of record; whole-file writes; hash-prefixed dirs; no open files on Volumes |
| CPU/memory over-provisioning cost | Measured sizing; cost model includes CPU and memory |
| Factory/serialization breakage | Module-scope workers; no `serialized=True`; emit tests |
| Linked-mode breakage on NVIDIA revision bumps; `batch_utils` name collisions | Pins; `nv/compat.py`; unique names; per-family processes; NV-link CI |
| Datasets without keyframe-aligned episodes can't be oracled | Labeled `oracle_ineligible_source`; same-pool certificates apply |
| Rubric decode (decord/AV1) | Ephemeral per-episode files in a decord-readable codec + rubric L0 (P2) |
| Prefix-cache "align" mode nondeterminism in the Rewarder | Kept only because NVIDIA validated it; certified; documented |
| Fused online profile untested in vLLM | Per-stage servers default; fused is experimental, in two processes |
| `exact` preset cost high (estimate) | Positioned for small/medium datasets and gold labels; `flora plan` shows the cost up front |
| Golden coverage gaps | Certificates per pool; `flora golden` on user data; community registry |
| License ambiguity on labels; name collision | Notice; counsel before 1.0; distinct name |

---

## 14. Open questions for you

1. ~~**Name and home.**~~ Resolved: `flora-deploy` at github.com/kaushikb11/flora-deploy (public), Python package `flora_deploy`, CLI `flora`. The unrelated FLORA reward paper (arXiv 2605.22123) remains a search-visibility note only.
2. **Modal plan tier** (Starter 10 / Team 50 / Enterprise)? It sets the defaults and the scale-test size.
3. **First target dataset(s) and camera setup?** This becomes the first golden stratum, the quickstart, and the measurement baseline.
4. **Commercial intent for the labels?** This affects when counsel review happens.
5. **Release priority.** Ship v0.1 `exact` early (week 6–7, expensive but exact)? Or hold for v0.2 `balanced` (week 13)? The plan recommends shipping v0.1.
6. **Envelope default** for episodes over 120 s: `warn` or `enforce`?
7. **State of record.** Is requiring a user bucket (S3/R2/GCS) acceptable for v0.1, with Volume-only as a warned fallback?

---

## 15. First two weeks, concretely

1. **Day 1:** repo skeleton, CI, `pins.py`; `flora prepare --dry-run`.
2. **Day 2:** `prep/download.py` on Modal; manifest checks; time the 109 GB Captioner download.
3. **Day 3:** `nv/linker.py` (unique names, process separation) + `CachedSession`; NV-link tests (prompt SHAs, `apredict` with canned JSON, 3-view construction).
4. **Day 4:** `media/index.py` + `cuts.py`; index-pass == `probe` on a v2.1 file and on a v3 concatenated file vs its stream-copy cut; keyframe-start check.
5. **Day 5:** `ReferenceWorker` on H200 runs NVIDIA's pipeline on 3 human MP4s with models resident; first wall-clock numbers.
6. **Days 6–7:** **Gate C** (bare `<|video_pad|>`: vLLM 0.24.0 vs HF) on early golden requests. Rewarder terminal/dense via NVIDIA's offline recipe on Modal.
7. **Days 8–10:** golden selection and builds per available pool; oracle repeat runs; measurement milestone (Gate D); `docs/measurements-p0.md`.

---

## 16. Changelog

**v3 (2026-10-21)**: verification-driven revision. Evidence is in `research_notes/Plan review v2/`.
- **Strategy:**
  - release train v0.1 `exact` (NVIDIA code at scale, week 6–7) → v0.2 `balanced` (certified vLLM) → v0.3 → v0.4;
  - requirement presets and `flora plan` (budget, deadline, fidelity → options);
  - a small certified matrix and a community certificate registry (D16).
- **NVIDIA code findings:**
  - per-episode FFmpeg-formula `average_rate`, CFR and duration in the index pass;
  - oracle clips are **stream-copy** cuts, not lossless re-encodes;
  - the 3-view segmenter is built inline in `pipeline.py:47-49`;
  - `batch_utils` name collisions → unique names and per-family processes;
  - bare `<|video_pad|>` → Gate C;
  - Captioner stop on `<|im_end|>` only, overriding generation_config's extra EOS;
  - up to 3 `apredict` rounds; NVIDIA's whole-episode failure semantics;
  - 32k guard computed after the fact;
  - `_CapturedInputs.pop` only;
  - full segment metadata persisted; the 5 provenance values in the schema;
  - streaming mosaics and the sliding caption cache;
  - Rewarder camera order L/C/R;
  - rubric per-episode files and the decord/AV1 issue;
  - av/Pillow pins on x86_64;
  - corrected the pixel caps; removed the `ready()` workaround.
- **vLLM 0.24.0 findings:**
  - batch moves to the offline `LLM` with a round-based micro-batcher (D14); AsyncLLM only online, with pre-rendering;
  - explicit `gdn_prefill_backend="triton"`, `max_num_seqs`/`max_num_batched_tokens` and stop tokens (D2);
  - prefix caching only for the Rewarder (as NVIDIA);
  - the `dt_bias` patch is feasible (upstream fix closed unmerged);
  - SSM state is FP32 automatically;
  - the OpenAI-server fallback is deprioritized; the fused profile is experimental;
  - hub pin resolved.
- **Modal findings:**
  - the bucket is the state of record (Volume v2 is beta, reload-with-open-files fails, 262k-file directories);
  - D15 attempts, quarantine and bisection for infinite crash/OOM retries;
  - CPU in physical cores, billed whether used or not → measured sizing, and CPU/memory in the cost model;
  - module-scope workers in the factory;
  - non-preemptible orchestrator with a Dict lease plus a reconciler;
  - native Environment budget backstop and live `billing.rates()`;
  - `stop_fetching_inputs()`, `simulate_preemption()`;
  - the `TimeoutError` subclass trap; `@app.server` startup_timeout; our own rate limits.
- **Schedule:** ≈18–20 weeks (P0 2.5 · P1a 4 · P1b 6 · P2 3 · P3 4).

**v2 (2026-10-21):** applied the first in-depth review:
- no clip cache;
- shard part-files;
- CachedSession;
- process pool and budgets;
- the long-episode policy;
- expanded config hash;
- the Rewarder certified like the other stages;
- the terminal-reward vertical slice.

**v1 (2026-10-21):** initial plan.
