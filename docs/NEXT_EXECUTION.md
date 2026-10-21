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

---

## Update after review 3 (2026-10-21): what's left and how it's split

**Budget:** $14.23 spent, **$35.77 left**.

### Lessons that change the plan

1. **Integration seams hid the critical bugs.** Each piece passed its own tests, but C1 (Modal raises the *builtin* `TimeoutError`), C2 (quarantine resolved under listing lag) and C3 (Volume never reloaded) only appeared when real pieces were wired together or checked against real Modal behavior. So:
   - every worker must pass a shared **conformance suite**;
   - every Modal-facing piece must pass a cheap **real-Modal CPU canary** before any GPU run.
2. **"exact" means identical model inputs, not identical outputs.**
   - D-1 traced output variance to cuDNN attention on H200.
   - Fix-H showed vLLM batch grouping changes Rewarder numbers.
   - Certification compares against the oracle's *measured* run-to-run noise.
   - Every record states whether deterministic mode was on.
3. **Missing components** that the earlier plan assumed:
   - an **index job** (`flora index`: dataset → episode index receipts in the store);
   - the **app factory** (one deployed app per config with module-scope workers, D10);
   - **compose/export**.

### Coordinator first (after Fix-J1/J2 land)

| ID | Work |
|---|---|
| **C1-conformance** | `tests/conformance/`: one suite every `ShardWorker` and `Store` implementation must pass (attempt marker first, skip-done, parts listing per the new `ShardReceipt.parts` rule, receipt last, preempt mark on SIGINT, NVIDIA failures → `EpisodeError` not crash, records validate as `flora_labels_v1`, idempotent re-run). Workers get it for free by parametrizing over their factory. |
| **C2-canary harness** | `tests/modal/` + `scripts/canary.py`: a real-Modal **CPU-only** run (cents) of orchestrator + VolumeStore/BucketStore + a trivial CPU worker: real `FunctionCall` polling, real Volume reload, real Dict lease, real preemption via `simulate_preemption()`. Gate for every Modal-facing merge. |

### Wave D (parallel; ≤4–5 agents at a time)

| ID | Depends on | Builds | Cap |
|---|---|---|---|
| **P1a-5r RewardWorker** | Fix-H (merged), J1 | `ShardWorker` around `stages/reward.py`; passes conformance | $0.5 |
| **P1a-5c compose/export** | J1 | parts → deduped `flora_labels_v1` JSONL + Parquet per stage, per config hash; `flora results` reader | $0 |
| **P1a-6a app factory + deploy** | J2, Fix-CFG | `build_app(cfg)`: module-scope worker classes decorated from `RunSettings` pools (GPU, containers, `@modal.concurrent`, cpu/memory, timeouts, retries, Volumes, secrets), orchestrator + reconciler cron, `--emit` script; `flora deploy` | $0.5 (canary) |
| **P1a-6b index job** | Fix-C (merged), J1 | `flora index`: CPU Modal fan-out over a dataset (LeRobot v2.1/v3, mp4 dir) → index receipts + `EpisodeRef`s in the store; reuses W3-C staging logic; resumable | $0.5 |
| **P1a-5s SegcapWorker** | **S1** | `ShardWorker` around the fast exact worker (identity-proven); passes conformance | $1 |
| **R-int canary run** | C2, 6a | Run the canary on Modal; must pass before S3 | $0.5 |

### Wave E (integration)

| ID | Depends on | Builds | Cap |
|---|---|---|---|
| **S2 golden build** (coordinator + 1 tooling agent) | S1 | Oracle ×2 (NVIDIA defaults) on the 42 golden episodes, plus ×1 in deterministic mode if budget allows; Rewarder reference on 12; per-call input digests stored. Writes `docs/measurements-p0.md`, decides Gate D, PLAN v4. **Stop rule:** halt if projection exceeds cap after the first 10 episodes. | $8 |
| **P1a-6c CLI/SDK** | 6a, 6b, 5c | `flora run/status/resume/results/report/cancel` wired to orchestrator, reconciler, planner and compose; SDK `submit/Job`; README quickstart | $0 |
| **S3 e2e + chaos** | Wave D, 6c, R-int | Internal v0.1: real `flora index → plan → run` on ~8 golden episodes, outputs vs S2 oracle (inputs identical; outputs within noise); chaos: `simulate_preemption()`, injected OOM episode, orchestrator kill, budget stop/resume | $4 |

### Wave F (the cheap path)

| ID | Depends on | Builds | Cap |
|---|---|---|---|
| **P1b-1 vLLM engines** | Fix-X, P1b-0 (merged), Fix-E (merged) | Offline `LLM` + micro-batcher for merged TempSeg and BF16 Captioner; `vllm_prompt` + `assert_engine_kwargs` + 32k budget; Gate A (`dt_bias` FP32) measured | $3 |
| **P1b-2 certification** | S2, P1b-1 | `balanced` vs oracle: L0 (input parity) and L3 (task metrics within oracle noise); certificates for the D16 matrix | $6 |

**Reserve ≈ $8.** Every branch gets an independent re-review before merge. Thematic reviewers (like review 3) run at each wave boundary to catch integration seams.

---

## Update after review 4 (2026-10-21): the path to the internal v0.1 / v0.2 milestones

**Budget:** ≈$19 spent, **≈$31 left**.

### What review 4 changed

1. **"Done" must mean done.** Any job outcome must be derivable from records and *legitimate* exclusions only (Fix-W1). Every later milestone is judged with this rule.
2. **The golden build must be re-costed at the real rate.**
   - Measured `exact` on cu128 is ≈$40/video-h before the persistent-generate-thread fix and ≈$25 projected with it, not the $14 assumed earlier.
   - The 30 golden episodes are 928 s of video (≈0.26 h) plus 12 reward episodes.
   - Two full default-mode oracle runs would be ≈$13–20, which doesn't fit.
3. **Identity proofs belong in deterministic mode.** In default mode, the first output difference (routine on H200) leaves every later call unchecked. Input-identity proofs therefore run with `determinism: deterministic`. NVIDIA-as-shipped behaviour (default mode) is still what the oracle records.

### Revised Phase-0 close: S2 golden build ($9 cap)

| Run | What | Why | Est. |
|---|---|---|---|
| **S2a** | Default-mode oracle over all 30 labeling episodes, fast exact worker (inputs proven identical) | The reference outputs "NVIDIA as shipped" produces | ≈$6.5 |
| **S2b** | Second default-mode run on a ~10-episode stratified subset | Run-to-run noise estimate for certification thresholds | ≈$2 |
| **S2c** | Rewarder reference on the 12 reward episodes (NVIDIA grouping) | Reward oracle | ≈$0.5 |

**Stop rule:** S2a halts if the measured rate after 8 episodes projects above $7. Identity proofs (fast vs unmodified, deterministic mode) are P1a-5s's job, not S2's.

**Outputs:**
- `docs/measurements-p0.md` (rates, startup, CPU breakdown, noise, per-pool timings);
- `rates.json` marked *measured*;
- Gate D closed;
- PLAN v4: rewrite §0/§1/§3 for the realities found (identical-inputs semantics, cuDNN nondeterminism, cu128 thread cache, Rewarder batch-grouping sensitivity, measured costs).

### Remaining work, with dependencies

| ID | Depends on | Builds | Cap |
|---|---|---|---|
| **P1a-5s** (running) | — | SegcapWorker + full wiring; identity on the 4 remaining pools; measured rate with the persistent generate thread | $3 |
| **Fix-W1 / Fix-W2** (running) | — | Review-4 fixes | ≈$0.2 |
| **P1b-1 vLLM engines** | Fix-X, P1b-0, Fix-E (all merged) — **can start now** | `engines/`: offline `LLM` + micro-batcher (D14) for merged TempSeg and BF16 Captioner; `vllm_prompt` + `assert_engine_kwargs` + 32k budget; balanced `segment`/`caption` stages and workers (conformance); Gate A (`dt_bias` FP32 plugin) measured | $3 |
| **S2 golden** | P1a-5s, W2 (rates/determinism) | As above | $9 |
| **P1a-6c CLI/SDK** | W1 | `flora run/status/resume/cancel/report`, SDK `submit/Job`, README quickstart. **Exit criterion for v0.1 usability:** a new user goes from `flora init` to results on a sample dataset by following README alone. | $0 |
| **S3 e2e + chaos (internal v0.1)** | W1, P1a-5s, 6c | Real deployed app via factory: `index → plan → deploy → run` on ~6 golden episodes (segcap + rewards). **Model inputs identical to S2's per call; outputs within noise; job `done` only by the new rule.** GPU chaos: `simulate_preemption()` on a GPU worker mid-shard, one injected OOM episode. | $3 |
| **P1b-2 certification (internal v0.2)** | S2, P1b-1 | `balanced` vs oracle: L0 input parity, L2 distribution parity, L3 task metrics within S2b noise. Certificates for the D16 matrix → `balanced` unlocked by certificate, not by flag. | $6 |
| **Review 5** | S3, P1b-2 | Final thematic review before calling the milestones done | $0 |

**Reserve ≈ $6.**

### Parallel schedule (≤4 agents at a time)

1. **Now:** P1a-5s, W1, W2, plus **P1b-1** (independent new modules).
2. **After W1:** P1a-6c.
3. **After P1a-5s + W2:** S2 (coordinator-run, one tooling agent if needed).
4. **After S2 + 6c:** S3 ∥ P1b-2.
5. **Last:** review 5 → fixes → PLAN v4.

---

## Update (2026-10-21, late): budget-first plan

The user deferred the $9 golden build ("a lot of dollars for now"). **Rule: ask before any single run over ~$2.** Spent ≈$19; ≈$31 left, but treat GPU spend as scarce.

### Principle

Do every remaining piece that can be proven **without GPUs** first (fake models, CPU fixtures, real-Modal CPU canary). Then compress all GPU validation into **one lean, user-approved session** where each GPU minute serves several purposes at once.

### Wave G: zero GPU spend (subagents, ≤4 at a time)

| ID | Depends on | Builds | Done when |
|---|---|---|---|
| **G1 CLI/SDK** | Fix-W1 | `flora run/status/resume/cancel/report/logs`, SDK `submit()/Job` (wait, stream results, cost so far), README quickstart | A new user follows README from `flora init` to `flora results` (using G2's local mode) without help |
| **G2 local runner** | Fix-W1 | `flora run --local --fake-models`: the *real* orchestrator, stores, workers, compose and CLI, with deterministic fake engines on `media.synth` / staged-free fixtures, in-process (no Modal). A CI job that runs a full multi-stage job end to end with injected faults on every PR. | CI runs a complete job (index → plan → run → compose → results) in <2 min with zero Modal calls |
| **G3 exact-split design** | P1a-5s | Cost study plus implementation if identity-testable on CPU. NVIDIA's `pipeline.segment()` and `caption_*()` are separately callable, so `exact` could run **TempSeg on L40S** (9B) and **Captioner on H100** (27B) as two pools, with persisted segments, without re-expressing NVIDIA glue. Also: output-identical Captioner speedups (bucketing caption calls by prompt length to reuse cuDNN graphs within a container; pre-warming common shapes). | Projected $/video-h per option with measured inputs; CPU identity tests for any implemented option |
| **G4 CI hardening** | — | CI jobs for conformance, test-order independence, logic-digest coverage, license guard, nvlink; cached NVIDIA code; fail on any unclassified module | CI green with all gates |
| **G5 docs + PLAN v4** | review 5 | Architecture doc, user guide (presets, costs, determinism, license), contributor guide (D17, conformance, canary), PLAN v4 rewritten to reality | Docs reviewed by a subagent for accuracy against code |
| **Review 5 + fixes** | running | — | — |

### Wave H: one lean GPU session (needs approval; ≈$6 total)

One combined run where each step reuses the previous one's containers and data:

1. **Deploy the real app** via the factory (CPU, cents).
2. **End-to-end exact (S3), the internal v0.1 milestone.** `flora index → plan → run` on **8 short golden episodes** (≈90 s of video) in **deterministic mode** (reproducible and likely faster), plus rewards on 4 reward episodes. This run **is** the mini oracle (S2-lite), so there's no separate golden build.
   - Pass criteria: job `done` by the new rule; per-call model inputs identical to the unmodified NVIDIA path on 2 of the episodes (CPU-proven elsewhere).
   - Est. ≈$3.
3. **Balanced validation + certification on the same 8 episodes.** Run P1b-1's engines on L40S/H100 for L0 parity, Gate A, throughput, and L3 metrics against the S3 outputs, with the noise floor taken from one default-mode re-run of 3 episodes. Result: a **provisional certificate** scoped to the covered pools. Est. ≈$3.

The full golden build (30+12 episodes) stays deferred until the user wants production-grade certificates.

### Order

1. **Now:** review 5 (running), Fix-W1 and P1b-1 code (running), G4.
2. **After Fix-W1:** G1 and G2 in parallel; G3 after P1a-5s review.
3. **After G1/G2 + review fixes:** ask the user for Wave H; then G5 and PLAN v4.

---

## Update (2026-10-21, night): critical path to the internal milestones

**State:**
- **Local `main`:** P1b-1 merged (vLLM engines and `balanced` workers, CPU-proven); unpushed until Fix-S fixes the two segcap tests.
- **Running:** Fix-S, Fix-R5, G2 (local runner), G1 (CLI/SDK/README).
- **Spend:** ≈$19.

### Critical path

```
Fix-S ─┐
Fix-R5 ┼─► review 6 (thematic, CPU) ─► fixes ─► G4 CI gates ─► Wave H (ask user) ─► G5 docs + PLAN v4
G2 ────┤                                              ▲
G1 ────┘                                   G3 cost study (CPU) ┘
```

### Remaining zero-spend items

| ID | What | Why it matters |
|---|---|---|
| **Review 6** | One integration reviewer that runs the **G2 local runner** end to end on every preset and fault, plus a docs-vs-code reviewer for G1's README | G2 makes the seams testable without Modal; README claims must match code |
| **G3 cost study** | (a) **Overlap shards inside a container** (R5a: ~21% fill/drain idle per 10-min shard); (b) **longest-episode-first** ordering; (c) **exact-split**: TempSeg pool on L40S plus Captioner pool on H100, using NVIDIA's separately callable `pipeline.segment()` and `caption_*()`; (d) longer-lived containers. Each option gets a CPU identity test (model inputs unchanged) and a projected $/video-h from measured component times. | Measured `exact` is ≈$36/video-h; these are the input-identical levers left |
| **G4 CI gates** | Conformance, order independence, logic coverage, license guard, nvlink, and the G2 local e2e on every PR | Locks in what reviews keep finding |
| **G5 docs + PLAN v4** | Architecture, user guide, contributor guide (D17, conformance, canary, classification rule); PLAN v4 rewritten to reality | Done last, so it describes what exists |
| **Hygiene** | Remove merged worktrees and branches; prune scratch `.pyc` in the NVIDIA code cache | Keeps agents from branching off stale state |

### Wave H: one GPU session, ≈$7–8 total, user approval required

All on the same staged golden episodes, reusing containers where possible:

1. **Balanced preflight:** CPU, cents. Can vLLM load NVIDIA's Captioner weights?
2. **Exact e2e (internal v0.1).** Through the real deployed app, on 8 short episodes in deterministic mode, plus 4 reward episodes. This is the mini oracle, ≈$3.
3. **Balanced validation:** L0, Gate A, throughput, ≈$2.
4. **Balanced provisional certification** against step 2, plus a 3-episode noise run, ≈$2.

**Gate to ask:** only after review 6 is clean and CI is green, so no GPU dollars are spent on bugs a CPU test could have caught.

---

## Update (2026-10-22): de-risking the one GPU session

The biggest remaining risk is spending Wave H's GPU dollars on **deploy and plumbing bugs**:
- media root and mounts;
- secrets and stores;
- worker-class names;
- env forwarding;
- orchestrator spawning into the deployed app;
- `--receipt` and determinism flow.

Local mode can't see these, and the canary only covers part of them. So, before any GPU:

| Step | What | Cost | Subagents |
|---|---|---|---|
| **R6 fixes** | Fix whatever review 6 finds (e2e/README, balanced/rewards, segcap) | $0 | 2–3, by theme |
| **H0 CPU rehearsal on real Modal** | `flora deploy --fake-models`: the **real factory app** with the real orchestrator, stores, Volumes, media staging, env and worker classes, but with the fake engines from G2 running in **CPU** containers. Then `flora index → plan → run → status → results` on 6 staged golden episodes (segcap + rewards), plus one `simulate_preemption()` and one injected poison episode. This rehearses exactly what Wave H will run, minus the models. | ≈$0.10 | 1 (builds `--fake-models` deploy mode: fake engines selectable per worker class in the factory; hashed so fake outputs never mix with real) |
| **H runbook** | `scripts/wave_h.py`: the exact Wave H sequence (balanced preflight → exact e2e → balanced validation → provisional certification) with a spend check (`spend.py` + container-minute tracking) and a **stop rule between steps**, resumable, writing a single `docs/wave-h-report.md` | $0 to build | 1 |
| **Ask user → Wave H** | Run the runbook | ≈$7–8 | coordinator |
| **Post-H** | Fixes from H; certificates committed (metrics only); rates marked measured; PLAN v4 + docs (G5) with a docs-vs-code reviewer; worktree/branch cleanup | $0 | 2 |

**Deferred (user decision later):**
- full golden build (≈$9);
- production-grade certificates over all pools;
- a scale test;
- posting upstream suggestions.

**Decision (2026-10-22): Wave H-lite, ≈$3 total**, replaces the ≈$7–8 Wave H:
1. H0 CPU rehearsal (≈$0.10);
2. `exact` e2e on 6 short episodes, deterministic mode, one H200 container (≈$1);
3. rewards on 3 episodes (≈$0.50);
4. `balanced` preflight + validation on the same 6 episodes, L40S + H100, ≤10 min each (≈$1);
5. provisional certification computed offline ($0);
6. 2-episode default-mode noise run (≈$0.30).

Each step has a projected-spend stop rule. The runbook runs only after review 6 is clean, CI is green and H0 passes.
