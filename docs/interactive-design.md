# Interactive / online API design for flora-deploy

> **Status: proposal (2026-10-22), not implemented.** Written by a read-only design pass over the code at that date; file:line references may drift.

The main recommendation is to **ship the online API without AsyncLLM at first.** Run the same offline `LLM` engine and `MicroBatcher` that batch uses, inside long-lived Modal services. That keeps the model inputs and the `cert_key` identical to the batch path, so batch certificates can cover online records. AsyncLLM becomes an optional, separately certified `interactive` preset later.

Three facts shape everything below:
- **Nothing in `balanced` is certified yet.** The `config/certificates/` directory holds only a README. P1b-1 GPU validation has not been run (`docs/balanced-engines.md:4`). The TempSeg merge is not pinned, so `SegmentWorker` refuses to start (`balanced-engines.md:25`). Gate C is recorded as FAIL on the quota and C2 run-level criteria (`docs/gate-c.md`).
- **The Rewarder is the only certified-today path.** It runs NVIDIA's own engine, and `certified` depends only on the envelope verdict (`stages/reward.py:648`).
- **Per-request cost is tiny compared with idle cost.** Cost control is mostly about keeping GPUs cold.

## 1. Use cases and API shape

**Who calls it:**

| Caller | Stages | Latency need | Shape |
|---|---|---|---|
| Eval harness / RL (PLAN §2 persona) | `reward.terminal`, `reward.dense` | Seconds to minutes | Many requests, often in groups |
| Active-learning loop | Rewards first, then segment/caption on the episodes it picks | Minutes | Async, in batches |
| Teleop labeling UI | segment → caption (full pipeline) | Under 60 s once warm, with a visible "warming" state | One episode at a time, a human waiting |

All three are job-shaped, not chat-shaped. NVIDIA's acceptance rules (EOS reached, non-empty text, the 32k guard; `stages/segment.py:218`) and its parsers need the whole completion, and TempSeg can run up to two more `apredict` retry rounds per episode. Token streaming would therefore have nothing useful to send.

**Endpoints** (CPU gateway; every route takes the same `JobRequest`):
- `POST /v1/jobs` with `stages: [...]` → `202 {job_id, estimate_usd, warm: {kind: bool}}`.
- `GET /v1/jobs/{id}` → `{status: queued|warming|running|done|failed|refused, records: [StageRecord], errors: [EpisodeError], cost, timings}`.
- `?wait=<=120` long-polls inside Modal's web-request timeout. This is the "sync" mode.
- Shortcuts:
  - `/v1/segment` and `/v1/caption`. Caption accepts a `segment` StageRecord. If that record was not produced by this deployment (checked against its `provenance.cert_key` and `engine_hash`), the caption record is uncertified.
  - `/v1/reward/terminal` and `/v1/reward/dense`.
  - `/v1/reward/terminal:batch` takes up to `max_num_seqs` episodes as **one NVIDIA group**. NVIDIA's terminal score depends on its batch-mates (`stages/reward.py:16-24`), so this is how an eval harness reproduces batch numbers.
  - `/v1/annotate` runs the full pipeline.
  - `/v1/reward/rubric` returns 501, because rubric is deferred (`check_mode("rubric")` raises).
- `POST /v1/uploads` → presigned PUT URLs.

**Request schema** (`serving/schemas.py`):
```
JobRequest{stages, episode: {id, videos: {main, left_wrist?, right_wrist?} -> allowlisted URI,
           range_s?, task, embodiment, profile?}, options: {terminal_rationale, envelope},
           segments?: StageRecord, idempotency_key?}
```
- Responses carry `StageRecord.to_json()` unchanged (`records.py:85-107`), so `certified` / `uncertified_reason` travel exactly as in batch.
- `provenance.source` is `{role: {uri: "online://<sha256>", range_s}}`.

**Auth.** Use Modal proxy auth (`requires_proxy_auth=True`, with `Modal-Key`/`Modal-Secret` headers) on the gateway, and never ship a public mode. On top of that, keep a token → quota table in a `modal.Dict` for our own rate limits (PLAN §1.3: Modal's own concurrency cap is not a rate limit).

**Idempotency.** `request_key = sha256(stages, per-file content sha256, range, task, embodiment, profile, options, stage hash / cert_key of the serving stack)`.
- Finished results are cached at `flora/online/<deployment>/results/<key>.json` in the user's bucket. `Idempotency-Key` maps to a `job_id` in the Dict.
- Because of D-1 nondeterminism, a repeated request returns the first result rather than a new sample. That is the right behavior for a UI.

## 2. Serving architecture on Modal

**Topology** (`modal_app/serve_app.py`, planned by a new `factory.plan_serve(cfg)` that reuses the `AppPlan`/`modal_kwargs` machinery):
1. **`gateway`**: `@modal.asgi_app` FastAPI on `img_cpu`, `min_containers=1` (cents per hour). It does auth, the allowlist, rate limits, the budget gate and the idempotency lookup, then `.spawn()`s `pipeline_run`.
2. **`pipeline_run(request)`**: a CPU function.
   - It indexes the episode with the batch indexer (`jobs/indexing.py:497` `index_one`), which gives the envelope and oracle-eligibility verdict and the index receipt.
   - It writes that receipt to `flora/online/<deployment>/<req>/`.
   - It then calls `segment → caption` and `reward.*` **in parallel**, since rewards don't need segments. Wall time ≈ max(seg + cap, reward).
3. **GPU services** (`@modal.cls`, async methods, async `@modal.enter`, one engine per container):
   - **SegmentService** on L40S. `@modal.concurrent(max_inputs=8)`, under `max_num_seqs=32` (`engines/offline_llm.py:96`).
   - **CaptionService** on H100!. `max_inputs=4` (`max_num_seqs=16`, `offline_llm.py:100`).
   - **RewardService** on H100!. `max_inputs=4`; generation is serialized under a lock, and only the Collage decode overlaps.

   These are new classes. The batch factory's refusal of `max_inputs>1` (`factory.py:646`) does not apply to them.

**Engine choice: offline engine plus a long-lived MicroBatcher.** Each request runs `async with mb.episode():`. The batcher's "every live episode is blocked" trigger already fires a round as soon as all in-flight requests are waiting. Set `max_wait_s` to about 0.25 s instead of the 2.0 s default (`microbatch.py:141`); its batching parameters are not hashed.
- The trade-off is that only one `generate` runs at a time. A request that arrives mid-round waits up to one round, roughly 5–30 s for Captioner rounds.
- AsyncLLM continuous batching fixes that, but only if measured p95 calls for it (Phase 3).

**Same code path as batch.** Split `balanced_app.build_runner` (`balanced_app.py:197-295`) into `build_stage(kind) -> (engine, job, stamp_fn, info)`, used by both `BalancedShardRunner` and the services. The services feed `EpisodeRef`s that point at real index receipts through the unmodified `InputResolver` (`workers_impl/segment.py:99-140`) and `SegmentEpisodeJob` / `CaptionEpisodeJob`.
- The caption job's `segments=` hook (`workers_impl/caption.py:75-77`) takes the upstream segment record directly, with no part files.
- Rewards call `stages.reward.run([episode])` per request. That is a one-line NVIDIA jobs file, so it is reference by construction. The `reset_prefix_cache` before every round is kept.
- Check that `SegmentEpisodeJob.run` never reads `ctx`, or pass a stub `ShardContext`.

**Segment → caption chaining** costs one Modal call hop (under 1 s) plus re-probing the video in the caption container (`prepare_videos`, a few seconds).
- Media sits on a `flora-online` Volume or the user's bucket mount, so nothing is copied.
- Fusing both engines on one H200 removes the hop but is a different stack. See Phase 3.

**Video upload:**
- Never send video bodies through the web endpoint. `POST /v1/uploads` returns presigned PUTs to `<bucket>/flora/online/uploads/<sha256>.mp4` (the user's bucket is the state of record, D6). The no-bucket fallback is `modal volume put` to `flora-online`.
- Requests may only name keys under the configured prefixes. There is no arbitrary fetch (PLAN P3).
- Limits: 256 MiB per file (NVIDIA's envelope cap, PLAN:91), a configurable duration cap (default 300 s), and three cameras at most.

**Cold starts:**
- Measured: Rewarder 229 s with a warm compile cache and prefetch, 318 s cold. Balanced is projected at about 130 s (`planner/rates.json` startup).
- Keep the `flora-cache` read-only copy plus `warm.start_prefetch`, and close the roughly 95 s container-local cache gap (`docs/cold-start.md:81-90`).
- GPU memory snapshots (`enable_memory_snapshot=True`, `experimental_options={"enable_gpu_snapshot": True}`; alpha, single GPU) are a Phase 3 experiment. vLLM's separate engine-core process may not snapshot cleanly; measure first.
- After a restore, re-run `verify_gdn_dtypes` and the receipt-hash check. The full byte verification runs before the snapshot.

## 3. Fidelity

**MVP: reuse the batch `cert_key`. No separate interactive certificate.** `key_document` (`validate/certkey.py:236-284`) hashes:
- the preset name;
- the engine identity and image;
- `EngineSpec.hashed()`, including the per-GPU limits;
- the pins and derived-weight digests;
- the modes;
- the logic digest of the `vllm-video` group (`config/logic.py:103-120`, which includes `balanced_app.py` and `workers_impl/*`).

If the services keep `preset="balanced"`, use `build_stage`, and keep transport code (`serving/*`) out of every logic group, they produce the **identical key**, so the batch certificate matches them.

Conditions:
- Embodiment comes per request. Call `cert_stamp(preset, kind, gpu, embodiment=...)` (`workers_impl/certstamp.py:52`) per request instead of reading `FLORA_EMBODIMENT`. An embodiment outside the certificate's covered pools gets `uncovered_pool`.
- **Land the `build_stage` refactor before P1b-2 certifies.** `balanced_app.py` is hashed, so changing it after certification would invalidate the certificate. Since none exist yet, doing it now costs nothing.
- Online batches are smaller than batch rounds. Ask P1b-2 to add a c=1 arm, so the certificate explicitly covers singleton composition.
- Rewards are served under NVIDIA's engine identity. A single-episode request is NVIDIA's own one-line run, but its logprobs can differ from a batch-job group by up to about 0.03 (the H100 measurement in `stages/reward.py:13`). Document this and point eval harnesses to `:batch`.

**AsyncLLM (`interactive` preset) needs its own certificate:**
1. **Gate C-async.** Render the same 127 requests through `await llm.renderer.render_cmpl_async([vllm_prompt(r)])` (`nv/capture.py:387-396`) and require ids, grids, bf16 pixel digests and placeholder ranges equal to `LLM._preprocess_cmpl`, under `mm_processor_cache_gb=0` (`capture.py:317`).
   - Keep `renderer_num_workers` at its default; it is in `FORBIDDEN` (`offline_llm.py:138-139`).
   - Set `max_num_seqs` / `max_num_batched_tokens` explicitly. AsyncLLM's defaults are 128 / 2048.
   - This is a new `validate/gatec.py` variant, `async_rendered`.
2. Add a new `EngineIdentity("vllm-async")`. Continuous batching changes composition, so the cert key changes.
3. Certify with P1b-2's golden set and **reuse its oracle noise runs**. Only candidate runs are new spend.

## 4. Cost controls

**Idle burn at pinned prices** (`jobs/budget.py:58-69`, including cores and memory):

| Service | Pool | $/h idle |
|---|---|---|
| Segment | L40S, 6 cores, 48 GiB | ≈ $2.61 |
| Caption | H100!, 8 cores, 64 GiB | ≈ $4.84 |
| Reward | H100!, 8 cores, 80 GiB | ≈ $4.97 |
| **All three warm** | | **≈ $12.4/h (≈ $300/day)** |

**Marginal cost per 60 s episode** (placeholder or measured rates): segment ≈ $0.02, caption ≈ $0.08, terminal ≈ $0.005, dense ≈ $0.014.

**Recommendation:**
- **Scale to zero by default**, with `scaledown_window` of 10 min.
- Add `flora serve warm --kinds segment,caption --for 2h`. It calls `update_autoscaler(min_containers=1)` and records an expiry; a CPU cron reverts it. A forgotten warm pool therefore costs hours of idle GPU, not weeks.
- `buffer_containers=0`.

**Spend cap** (`serving/budget.py`, reusing `jobs.budget`):
- **Ledger.** Every GPU container already writes `containers/<id>.json` heartbeats. `spend()` over the deployment's namespace, maxed with `modal_billing_provider`, gives spend per period.
- **Admission.** The gateway refuses when `spend + warm containers × remaining scaledown + request estimate > cap × (1 − margin)`. It returns 402 `budget_exhausted` and never spawns.
- **Hard cap.** A one-minute CPU watchdog sets `min_containers=0` and then `modal app stop`s the GPU services once `spend ≥ cap`.
- **Backstop.** A Modal Environment budget at 1.2× the cap. It is set by hand, because SDK 1.6.2 has no setter (`budget.py` docstring).
- **Per-request accounting.** Estimate from `rates.json`, then record actuals from the audit timings. Container time is split across in-flight requests by video-seconds, and the remainder goes to an explicit `idle` line in `flora serve report`.

## 5. Phased plan

**Phase 0: fakes ($0–1, CPU only).** All packages are tested with `local/engines.py`, `local/balanced.py`, `tests/fakes/flora_backend.py` and the FastAPI TestClient.

| WP | Files owned | Test |
|---|---|---|
| 0.1 Schemas and keys | `serving/schemas.py`, `serving/keys.py` | Round-trip StageRecord; key stability |
| 0.2 Gateway | `serving/gateway.py`, `serving/auth.py`, `serving/allowlist.py`, `serving/ratelimit.py` | 401/403/413/429/402 paths |
| 0.3 Runtime | `serving/runtime.py` (long-lived batcher, request → receipt → `EpisodeRef`) | Concurrent fake episodes; cancellation |
| 0.4 Build refactor | `modal_app/balanced_app.py` (`build_stage`) | Existing tests; `key_document` equal from runner and service |
| 0.5 Budget | `serving/budget.py`, watchdog | Fake receipts and clock |
| 0.6 App and CLI | `modal_app/serve_app.py`, `factory.plan_serve`, `cli/serve.py`; `presets.py` gets a `serve:` section, not a preset | `--fake-models` deploy on `img_cpu` (under $1) |

**Phase 1 MVP: rewards online ($3–5).**
- Scope: `/v1/jobs`, terminal/dense/`:batch`, uploads, proxy auth, budget gate, scale to zero.
- This serves the eval harness with records certified today.
- Validation:
  - one cold start (about 4 min of H100);
  - 20 golden episodes, with parity against batch `stages.reward.run` using the same singleton grouping (equal records);
  - scaledown.

**Phase 2: segment, caption, annotate ($8–12).**
- Gated on the P1b-1 GPU validation (≈ $2.4, already planned) and the TempSeg pin.
- Validation:
  - online vs batch records at c=1;
  - a 30-minute p50/p95 load test on L40S + H100.
- Records are `uncertified_stack` until P1b-2 certificates exist.

**Phase 3: hardening and options ($20–60).**
- Warm schedules and SSE progress events.
- GPU snapshots: measure p50 startup.
- AsyncLLM: `engines/async_llm.py` implementing `Engine.generate` (`engines/base.py:93-96`), Gate C-async (about $2), and the interactive certificate (candidate runs only).
- Fused H200 profile: TempSeg + Captioner at about $4.54/h idle, against $7.45/h for L40S + H100. It needs split `gpu_memory_utilization`, a different key, and its own certificate.

## 6. Risks and open questions

**Risks:**
- No certificates exist, the TempSeg pin is missing, and Gate C is FAIL. Segment and caption online labels are uncertified until P1b-2.
- Cold starts of 2–4 min clash with UI expectations. Warm windows cost about $7–12/h.
- GPU snapshots are alpha; vLLM's multi-process engine may not restore.
- The cap lags. Billing trails by minutes and heartbeats are stale for up to 300 s, so the cap can overshoot by roughly one scaledown window.
- License. An endpoint reachable by third parties could count as "hosting" the noncommercial models. Proxy auth only, and say so in the docs.
- Online and batch numerics differ (singleton batches), and repeat runs differ (D-1).

**Questions for you:**
1. What p95 does the teleop UI need, and is a "warming" state acceptable?
2. Are callers your own team only?
3. What is the serving budget per day or month, and how much idle spend is acceptable?
4. Is an MVP with uncertified segment/caption acceptable, or should it be rewards-only until P1b-2?
5. Is a user bucket available for presigned uploads?
6. Is embodiment set per request or per deployment?
7. Should online results be composable into batch outputs or LeRobot write-back?

### Critical files for implementation
- /Users/kaushikbokka/apps/robotics/flora-deploy/src/flora_deploy/modal_app/balanced_app.py
- /Users/kaushikbokka/apps/robotics/flora-deploy/src/flora_deploy/engines/microbatch.py
- /Users/kaushikbokka/apps/robotics/flora-deploy/src/flora_deploy/validate/certkey.py
- /Users/kaushikbokka/apps/robotics/flora-deploy/src/flora_deploy/stages/reward.py
- /Users/kaushikbokka/apps/robotics/flora-deploy/src/flora_deploy/jobs/budget.py