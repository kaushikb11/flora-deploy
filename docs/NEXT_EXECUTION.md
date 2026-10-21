# Execution plan: closing Phase 0, then P1a (`exact`) and P1b (`balanced`)

> 2026-10-21. These are internal milestones only; no release, tag or publish until the user asks.
> Budget: $50 total. $10.25 spent at writing, **$39.75 left**. Track with `scripts/spend.py`.

## 1. Where we are

| Done (on `main`) | Running now |
|---|---|
| Weights verified on Modal; linked mode hardened (Fix-B); media fail-closed (Fix-C); Gate C passes; Rewarder batched == NVIDIA; reference runs on Modal; BF16 Captioner bit-identical (OPT-2, 305 s → 56 s); cold-start recipes (W3-B); 42 golden episodes staged (W3-C); hash-locked images; contracts v2 | W3-A (GPU-lock concurrency + profiling), OPT-1 (decode-once, in-memory PNG), Fix-H (Rewarder robustness), Fix-E (Gate C hardening), Fix-images-2 (cu128 torch, fingerprint) |

## 2. Dependency graph

```
running:  W3-A ─┐   OPT-1 ─┐   Fix-images-2 ─┐      Fix-H ─────────────┐     Fix-E ──────────────┐
                ▼          ▼                 ▼                         │                         │
         ┌──────────── S1  Fix-D: fast exact worker (proven identical) │                         │
         │                 ▼                                           │                         │
         │           S2  golden build ─► measurements-p0 ─► PLAN v4    │                         │
         │                 │                         (rates feed P1a-4)│                         │
unblocked now:             │                                           ▼                         │
  C0 contracts+ ─► P1a-1 config ─┐                                P1a-5 workers ◄─ S1            │
                   P1a-2 store ──┼─► P1a-3 orchestrator ─► P1a-6 CLI/SDK ─► S3 e2e + chaos       │
                   P1a-4 budget ─┘        (fake Modal)                                           │
  P1b-0 TempSeg merge (CPU) ─────────────────────────────────► P1b-1 vLLM engines ◄──────────────┘
  U-1 upstream drafts (no spend)                                     ▼
                                                    P1b-2 certification vs S2 oracle
```

## 3. Workstreams

Every subagent works in its own git worktree with a fixed set of owned paths, and finishes with green `ruff` + `pytest` (+ `-m nvlink` where relevant). Commits carry **no Claude attribution**, and subagents never push. Every branch then gets an **independent re-review subagent** before merge.

### Coordinator-first

| ID | Work | Why first |
|---|---|---|
| **C0 contracts+** | Add `RunSettings` (max_gpus, budget_usd, per-kind max_containers/max_inputs, shard_target_gpu_minutes, retries, envelope policy, preset) and a `Spawner` protocol (spawn / poll / cancel FunctionCalls) in `jobs/`; a shared `tests/fakes/modal.py` (FakeSpawner with preemption/OOM/expiry injection, FakeStore) | P1a-1/2/3/4/5 code against one narrow surface and one fake, so they can't drift |

### Wave A: unblocked now (CPU / fake Modal; ≈$1)

| ID | Owns | Builds | Done when |
|---|---|---|---|
| **P1a-1 config** | `config/schema.py`, `config/presets.py`, `config/hashing.py`, `examples/*.yaml`, `cli` `init` | `flora.yaml` (pydantic) → `RunSettings` + stage configs; presets `exact`/`balanced`; D11 config hash (logic version, NVIDIA code + manifest digests, prompt-pool SHA, model revs, derived-weight receipts, engine args, sampling, mm_kwargs, decode pins, `image_fingerprint()`; **excludes GPU**) | Changing any hashed input changes the hash; GPU change doesn't; templates validate |
| **P1a-2 store + planning** | `jobs/stores/{volume,bucket,local}.py`, `jobs/manifest.py`, `jobs/attempts.py` | Whole-file stores (Volume v2 with commit/reload discipline; S3/R2/GCS bucket; local for tests); manifest from the index (group by source file, pack to target GPU-minutes, `segcap` shards for `exact`); attempts → quarantine → bisection (D15); dedupe-by-uid compose input | Property tests: no lost/duplicate episodes under random crash/retry interleavings in FakeStore with eventual consistency |
| **P1a-3 orchestrator** | `jobs/orchestrator.py`, `jobs/reconciler.py`, `modal_app/orchestrator_app.py` | Nonpreemptible CPU orchestrator, Dict lease (`skip_if_exists` + TTL), continuation before 23 h, chaining by receipts, GPU cap, cancel/budget-stop via `ControlRecord`s; stateless reconciler that rebuilds status from files; `TimeoutError`-subclass trap handled | Fake-Modal simulations: preemption ×N, OOM poison pill → quarantine → bisect → `errors/`, lease contention, orchestrator kill + reconcile, budget stop + resume |
| **P1a-4 budget + planner** | `jobs/budget.py`, `planner/**`, `cli` `plan`/`estimate` | Container-lifetime spend (GPU+CPU+mem via `Workspace.billing.rates()`), projection, Environment-budget backstop, `flora plan --budget --deadline --preset` options table; rates from a `rates.json` (placeholder now, real after S2) | Given rates + index + quota → deterministic options; spend math unit-tested against hand calculations |
| **P1b-0 TempSeg merge** | `prep/merge_tempseg.py`, `modal_app/convert_app.py` (merge fn) | FP32 merge `W += 2·B@A` on CPU, cast linears to BF16, keep `A_log`/`dt_bias` FP32, Flora processor config + template; receipt | Merged weights load; the merged forward equals unmerged PEFT forward within BF16 tolerance on CPU (small inputs); ≤$1 |
| **U-1 upstream drafts** | `docs/upstream-suggestions.md` | Concise, evidence-backed suggestions for NVIDIA's HF repos (BF16 Captioner, wrapped video tokens, decode reuse, `apredict` docs, cu128 note, manifest re-hash cost) | Draft only; you decide what to post |

### Wave B: after the running agents land

| ID | Depends on | Builds | Cap |
|---|---|---|---|
| **S1 Fix-D: fast `exact` worker** | W3-A, OPT-1, Fix-images-2 | NVIDIA's own `annotate()` with `CachedSession` injected (drop re-expressed glue); OPT-1 + OPT-2 patches; W3-B no-copy layout + prefetch + parallel Captioner hash; GPU-lock concurrency; review2_wsd fixes (receipt check, watchdog semantics, exception transport, run_id). **Identity proof:** composite vs unmodified NVIDIA on ~6 golden episodes, full result dicts identical. | $3 |
| **P1a-5 workers + compose** | C0, P1a-2, S1, Fix-H | `SegcapWorker` (exact) and `RewardWorker` implementing `ShardWorker` (attempt marker → skip done → per-episode records → local parts → whole-file copy → receipt last; `@modal.exit` flush; `stop_fetching_inputs` on fatal); `compose` → Parquet/JSONL | $1 |
| **P1b-1 vLLM engines** | P1b-0, Fix-E | Offline `LLM` + micro-batcher (D14) for TempSeg (merged) and Captioner (BF16); `vllm_prompt`, processor cache off, 32k budget, explicit engine args, `gdn_prefill_backend="triton"`, stop tokens; Gate A (`dt_bias` FP32 plugin) measured | $3 |

### Wave C: integration

| ID | Work | Cap |
|---|---|---|
| **S2 golden build** (coordinator + 1 subagent for tooling) | Oracle ×2 over 42 golden episodes with the S1 worker (self-noise); Rewarder reference on 12; write intermediates to `flora-golden`; `docs/measurements-p0.md`; Gate D; `rates.json`; PLAN v4 cost model. **Stop rule:** after the first 10 episodes, re-estimate; halt if projected > $8. | $8 |
| **P1a-6 CLI/SDK** | `init/prepare/index/plan/deploy/run/status/resume/results/report/cancel`, SDK `submit/Job`, quickstart in README | $0 |
| **S3 e2e + chaos** | v0.1 internal milestone: real job on ~10 golden episodes end to end, then chaos (`simulate_preemption()` ×5, injected OOM episode, orchestrator kill, budget stop/resume); compare outputs to S2 oracle (must be identical) | $5 |
| **P1b-2 certification** | `balanced` vs S2 oracle, L0–L3, Gates A/B/E, certificates for the D16 matrix (segment L40S + H100!, caption H100!, dense H100!) | $6 |

## 4. Budget plan (remaining $39.75)

| Bucket | Cap |
|---|---|
| Running agents (W3-A, OPT-1, Fix-H, Fix-E, Fix-images-2), remainder | ~$6 |
| Wave A (P1b-0 merge on CPU) | $1 |
| S1 fast exact worker | $3 |
| P1a-5 workers | $1 |
| P1b-1 engines | $3 |
| S2 golden build | $8 |
| S3 e2e + chaos | $5 |
| P1b-2 certification | $6 |
| Reserve | ~$6 |

Rules:
- Each subagent gets a hard cap in its prompt and reports actual spend. I reconcile with `spend.py` after each wave.
- Any GPU job longer than 10 minutes has a stop rule.

## 5. Operating rules

- **Concurrency:** at most ~6 concurrent subagents, to avoid the session rate limit that paused agents twice. Wave A launches as the current agents finish.
- **Ownership:** disjoint paths per agent. Shared files (`pyproject.toml`, `cli.py`, `pins.py`) are coordinator-owned or assigned to one agent per wave. `cli.py` is split into per-command modules (`cli/*.py`) during C0 to end conflicts.
- **Merges:** independent re-review → merge → full suite (`pytest` incl. nvlink) → push. Conflicts are resolved by the coordinator.
- **Fidelity invariant:** anything labeled `exact` must be proven bit-identical to unmodified NVIDIA code (D17). `balanced` needs certificates. No exceptions.
- **License:** no NVIDIA code, prompts or outputs in the repo; re-expressed glue is minimized (S1 removes the largest piece).

## 6. Milestone exit criteria (internal)

- **Phase 0 done:**
  - golden oracle ×2 exists with noise measured;
  - `measurements-p0.md` replaces every estimate;
  - Gates C and D decided;
  - PLAN v4.
- **v0.1 `exact` (internal):**
  - `flora plan` → `flora run` labels golden episodes end to end, identical to the oracle;
  - the chaos suite passes;
  - the spend report is within ±25% of the plan's estimate.
- **v0.2 `balanced` (internal):**
  - certificates for the D16 matrix;
  - measured $/video-hour vs `exact`;
  - GPU utilization >60% while busy.
