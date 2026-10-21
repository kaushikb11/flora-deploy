# Phase 0 execution plan: workstreams, subagents, gates

> Companion to [`PLAN.md`](PLAN.md) §10 (P0) and §15. Status: proposed, 2026-10-21.

## Goal of Phase 0

Phase 0 ends when four things are true:

1. NVIDIA's own pipeline runs on Modal with its models kept resident.
2. The ~100-episode golden set exists, with every intermediate stored.
3. The two riskiest assumptions have been tested:
   - **Gate C:** vLLM's expansion of a bare `<|video_pad|>` token matches HF.
   - **Gate D:** where CPU time goes.
4. `docs/measurements-p0.md` has replaced the plan's estimates with measured numbers.

## Dependency graph

```
WS0 contracts (coordinator) ──┬─► WS-A weights prep + Modal scaffolding ─┐
                              ├─► WS-B linked mode (nv/)  ───────────────┼─► WS-D reference worker (H200) ─┐
                              ├─► WS-C media index / cuts / sources ─────┤                                 ├─► WS-G golden build
                              └─► WS-F golden-dataset research ──────────┼─► WS-E Gate C harness (L40S) ───┤   + measurements
                                                                         └─► WS-H Rewarder recipe (H100) ──┘   (coordinator)
           wave 1: no Modal spend, runs in parallel     wave 2: GPU, spend-gated          wave 3: spend-gated
```

## How the work is split

- **Coordinator (me).**
  - Writes the shared contracts first (WS0), so parallel work can't drift.
  - Launches each wave and reviews every diff.
  - Runs the full test suite, merges, and pushes.
  - **Starts any Modal run that costs money, and only after you approve it.** Subagents write and unit-test code but never spend on your account.
- **Subagents.** One per workstream, each in its own git worktree (`isolation: "worktree"`, branch `ws/<name>`).
  - Each owns a fixed set of paths. Shared files (`pyproject.toml`, CI, `cli.py`) belong to WS0 or to one named workstream.
  - Each must finish with green `ruff` and `pytest`, commit on its own branch, and not push.
  - **Commit rules:** no Claude attribution. Never commit NVIDIA code, prompts or weights (the license-guard test enforces this).
  - Return a summary of what changed, how it was tested, and any open issues.
- **Review.** After each wave, the coordinator reads every diff. A review subagent checks correctness and license hygiene before the merge.

---

## WS0: contracts (coordinator, before wave 1)

Small, but it unblocks parallel work.

| Item | Purpose |
|---|---|
| `.python-version` → 3.12; add `av==16.0.1`, `numpy`, `pillow`, `pyarrow`, `modal` to deps; pytest markers `nvlink` (needs NVIDIA code) and `modal` (needs the Modal account) | No workstream needs to edit `pyproject.toml` |
| `scripts/fetch_nv_code.py`: downloads **only the code pins** (no weights) to `~/.cache/flora-deploy/nv/<repo>@<rev>/`; `tests/conftest.py` fixture `nv_code(family)` that skips when absent | WS-B and WS-C nvlink tests run against the real NVIDIA code without committing it |
| `src/flora_deploy/layout.py`: Volume names (`flora-weights`, `flora-cache`, `flora-golden`), snapshot paths `snapshots/<repo>@<rev>/`, derived paths | One naming scheme for WS-A, WS-D, WS-E and WS-H |
| `src/flora_deploy/media/types.py`: `SourceRef`, `StreamIndex` (fields mirror NVIDIA's `Video`: rate as a `Fraction`, `pts_us`, `duration_us`, size, CFR, codec, keyframe-start flag), `EpisodeIndex`, `EnvelopeVerdict` | WS-C produces these, WS-D and WS-E consume them |
| `src/flora_deploy/nv/__init__.py`: interface contract (`Family` enum `SEGCAP`/`REWARDER`, `load_family()`, `CachedSession` protocol, `CaptureProcessor` signature) as docstrings and stubs | WS-B implements it, WS-D and WS-E call it |
| `tests/unit/test_license_guard.py`: fails if any tracked file is under `batch_utils/`, is named `TEMPSEG_PROMPTS.json`, `*.safetensors` or `LICENSE.pdf`, or is >1 MB | Mechanical protection for the public repo |
| CI: an `nvlink` job that runs `fetch_nv_code.py`, then `pytest -m nvlink` | NVIDIA code is downloaded to run tests, never stored in the repo |

## Wave 1 (parallel, no spend)

### WS-A: weights prep + Modal scaffolding
- **Owns:** `src/flora_deploy/modal_app/**`, `src/flora_deploy/prep/**`, the `prepare` command in `cli.py`.
- **Builds:**
  - `images.py` with `img-cpu`, `img-hfref` and `img-vllm`, pinned per PLAN §7.
  - A `prep_app.py` Modal function that runs `snapshot_download` for each pin into `flora-weights` (`HF_XET_HIGH_PERFORMANCE=1`). It is resumable and skips complete snapshots.
  - Verification:
    - every LFS file's sha256 matches the tree metadata;
    - NVIDIA's `SOURCE_MANIFEST.sha256`;
    - the adapter sha256;
    - the Captioner manifest identity, computed the way NVIDIA's code computes it.
  - Writes `prep_receipt.json`.
  - `flora prepare [--only KEY]` runs it via `app.run()`.
- **Done when:** unit tests with fake HF/Modal pass, `modal` imports cleanly, and `--emit`-style printing shows exactly what would run. **No Modal run.**

### WS-B: linked mode (`nv/`)
- **Owns:** `src/flora_deploy/nv/**`, `tests/nvlink/test_nv_*.py`, `tests/unit/test_nv_*.py`.
- **Builds:**
  - `linker.py`:
    - loads each family's `batch_utils` under a unique module name;
    - contains bootstrap's `sys.path` inserts;
    - raises if two families load in one process;
    - verifies NVIDIA's vendor SHA manifest.
  - `session.py`: `CachedSession`.
  - `capture.py`: `CaptureProcessor` and `_CapturedInputs` (`.pop` only).
- **nvlink tests:**
  1. Bootstrap passes.
  2. `temporal_prompt` passes NVIDIA's own SHA assertion for every pool.
  3. Both segmenter constructions (3-view inline, as in `pipeline.py:47-49`, and `production_segmenter`) given canned JSON produce identical segments via `apredict` and via sync `predict`. This includes a canned case that triggers retry rounds and NVIDIA's failure semantics.
  4. Through the real Flora processor files (tokenizer, processor config, chat template; downloaded at test time), `CaptureProcessor` captures exactly what `temporal_inputs`, `three_view_caption_inputs` and `processor_inputs` would pass. Replaying the captured args gives identical `input_ids` and `video_grid_thw`.
- **Done when:** unit and nvlink tests are green on CPU.

### WS-C: media
- **Owns:** `src/flora_deploy/media/**` (except `types.py`), `tests/*/test_media_*`.
- **Builds:**
  - `index.py`: per-episode probe equivalent from a source file plus an optional time range. It produces relative PTS, `duration_us`, `average_rate`, CFR, size and codec, the keyframe-at-start flag, and every envelope check from PLAN §1.1.
  - `cuts.py`: stream-copy per-episode cuts.
  - `sources/`: `mp4_dir`, `lerobot_v21` and `lerobot_v3` readers (meta parsing, camera-key mapping).
  - Fixtures generated at test time: H.264 and AV1, CFR and VFR, a concatenated v3-style file, a non-keyframe-aligned episode.
- **Ground truth:** **NVIDIA's `media.probe` run on the stream-copy cut**.
  - For MP4, `average_rate` comes from the track's `stts`. WS-C must determine empirically what a per-episode cut reports, then reproduce it from the concatenated file.
- **nvlink tests:** index == `probe(cut)` for every fixture. Decoded frames via `decode_selected` on the cut == our range decode from the concatenated file.
- **Done when:** unit and nvlink tests are green, and there is a short `docs/media-notes.md` covering the rules found.

### WS-F: golden-dataset research (research only, no code)
- **Owns:** `docs/golden-candidates.md`.
- **Finds** public datasets per pool:
  - human egocentric single-view;
  - 3-view bimanual (ALOHA-family on the LeRobot Hub: camera keys, how they map to `main/left_wrist/right_wrist`);
  - single-view robot (DROID/Galaxea/WGO-Bench);
  - Dex/Tianji if anything exists.
- **Records for each:** license, LeRobot format version, codec, fps, resolution, episode-duration distribution (share ≤120 s), size.
- **Spot checks** (small sampled downloads to scratch): codec, CFR, whether episodes start on keyframes.
- **Proposes** a stratified ~100-episode selection and states which pools stay uncovered.

## Wave 2 (needs Modal; coordinator runs each after approval)

| WS | Builds | GPU / est. spend | Exit |
|---|---|---|---|
| **A-run** | Execute `flora prepare` → 172 GiB into `flora-weights` + receipt | CPU, ≈$2–5 | Receipt with all checks passing; timing recorded |
| **WS-D reference worker** | `ReferenceWorker` (img-hfref) on `H200`: `CachedSession`, NVIDIA `pipeline.segment` / `caption_single` / `caption_three` on stream-copy clips; stores outputs + NVIDIA audit intermediates to `flora-golden` | H200, ≈$5–10 smoke | 3 human MP4s + 1 three-view episode labeled with models resident; wall-clock logged |
| **WS-E Gate C harness** | L40S function: vLLM 0.24.0 + Qwen3.5-9B base weights + Flora processor/template; requests built by `CaptureProcessor` from NVIDIA's builders on fixture frames; compares vLLM `prompt_token_ids` / grid with HF | L40S, ≈$1–3 | Pass/fail report per pool and builder |
| **WS-H Rewarder recipe** | img-vllm H100 class running NVIDIA's `core` (terminal + dense) on the offline `LLM` over an episode list; outputs + probabilities | H100, ≈$3–6 | Terminal/dense outputs on 3 episodes, matching NVIDIA's CLI scripts run in the same container |

Wave 2 total: **≈$15–30** (estimate). WS-D, WS-E and WS-H are written by subagents in parallel. I run each one in turn on Modal and feed failures back to the same subagent.

## Wave 3: golden build + measurements (coordinator + one subagent)

- **WS-G (subagent):**
  - A measurement harness: per-component CPU timers; prompt/output token stats; RSS sampling; cold-start breakdown with and without `flora-cache`; `/dev/shm`; CloudBucketMount throughput.
  - A golden-set builder over the WS-F selection, with 2 oracle runs on 2 GPU types.
- **Coordinator:** runs the build (≈$150–300, estimate; separate approval) and writes `docs/measurements-p0.md`. Decides Gates C and D, then updates PLAN.md's cost model.

## Decisions needed before starting

1. **Approve WS0 + wave 1** (no spend).
2. **A Modal budget for wave 2** (proposed cap $40). I'll still announce each run before starting it.
3. **Golden-dataset preference.** If you have your own data or camera setup, WS-F prioritizes it.
4. **Standing push permission** for merges to `main` at review checkpoints, or ask each time.

---

## Status after waves 1–2 (2026-10-21)

All merged to `main` (267 tests): WS0, WS-A/B/C/F, WS-D/E/H, Fix-A, Fix-C. Fix-B and Fix-images
are running. Modal spend so far ≈ $8 (weights download, Gate C $0.86, reference $2.4, rewarder $1.9).

| Result | Evidence | Consequence |
|---|---|---|
| 172 GiB of pinned weights on `flora-weights`, every byte verified; trust anchors pinned | WS-A, Fix-A | Done |
| **Gate C passes** after wrapping NVIDIA's bare `<\|video_pad\|>` (44/44 requests: token ids, grids, bf16 pixels identical). Without the wrap, vLLM rejects every prompt. | `docs/gate-c.md` | The `balanced` path is viable |
| **Rewarder: batched output == NVIDIA's scripts** (terminal, terminal+rationale, dense) | `docs/rewarder-smoke.md` | Reward stage is ready for v0.1 |
| **Reference pipeline runs on Modal** (4 episodes, both models resident on H200) | `docs/reference-smoke.md` | The `exact` preset is viable |
| **CPU dominates the reference run:** model generation is only 9–31% of each stage on 1080p clips | WS-D | Gate D effectively answered. The GPU idles most of the time in `exact` |
| **Cold starts are large and variable:** reference startup 251 s (Captioner copy 123 s + NVIDIA manifest re-hash ~100 s); Rewarder engine startup 99 s → 929 s across runs (Volume read variance + compile) | WS-D, WS-H | Long-lived containers, compile cache and Volume throughput are now first-order cost items |
| PNG compression size depends on the Pillow/zlib build and changes caption frame scale | WS-E | All processing must run in one pinned image (Fix-images) |

## Next: wave 3 (finishes Phase 0) and Phase 1a

### Wave 3 (parallel subagents)

| WS | Builds | Spend |
|---|---|---|
| **W3-A: exact-preset concurrency** | N episode threads run NVIDIA's pipeline concurrently, while one **GPU lock** serializes every `generate()`. Each model call stays NVIDIA's batch-1 call, so outputs stay identical, and CPU work (decode, PNG/JPEG, resize) overlaps across episodes. Prove bit-identical output vs sequential runs and measure the speedup. Also tune `cpu=` cores and torch threads. | ≈$5 |
| **W3-B: cold-start lab** | Break down and fix startup: Volume read throughput and variance (sequential vs parallel shard reads), `VLLM_CACHE_ROOT` on `flora-cache`, CUDA-graph capture cost, local-disk copy vs direct read, and whether NVIDIA's Captioner re-hash can run once per container lifetime instead of per load. Output: recommended startup recipe per worker + numbers. | ≈$8 |
| **W3-C: golden staging** | From `docs/golden-candidates.md`: stage the ~100-episode selection + ~40 reward episodes into `flora-golden` (subsets only), run index + `verify_episode` + eligibility, make stream-copy cuts, write the golden manifest. CPU only. | ≈$2 |
| **W3-D: measurement harness** | Per-component CPU timers around NVIDIA calls (sampling profiler for NVIDIA internals), token stats from NVIDIA audits, RSS, `/dev/shm`, CloudBucketMount throughput, all as structured events. Built into ReferenceWorker/RewardWorker as opt-in. | ≈$2 |
| **Golden build (coordinator, after W3-A/C/D)** | Oracle ×2 on H200, ×1 on B200 (cross-GPU noise) + reward reference runs, using W3-A concurrency | **≈$100–150, separate approval** |
| **Coordinator** | `docs/measurements-p0.md`; Gate D write-up; PLAN v4 (cost model from measured rates); P1a contracts | — |

### Phase 1a → v0.1 `exact` (parallel subagents after coordinator contracts)

Contracts first (coordinator): `flora_labels_v1` record schema, `Manifest`/`Shard`/`Receipt` types,
`Store` interface (bucket + Volume backends), worker interface `run_shard(shard_id, cfg_hash)`,
event schema.

| WS | Owns |
|---|---|
| **P1a-1 config** | `flora.yaml` schema, presets, D11 config hash (incl. image fingerprint from Fix-images), `flora init` templates |
| **P1a-2 store + planning** | bucket/Volume stores (whole-file, hash-prefix dirs), manifest + shard packing from index, attempts/quarantine/bisection (D15) |
| **P1a-3 orchestrator** | non-preemptible orchestrator with Dict lease + continuation, stateless reconciler, chaining, GPU cap; fully testable with a fake Modal |
| **P1a-4 budget + planner** | container-lifetime receipts, live `billing.rates()`, Environment budget backstop, `flora plan` / `flora estimate` from measured rates |
| **P1a-5 workers** | exact segment+caption shard worker (W3-A concurrency, W3-B startup recipe), batched reward shard worker, compose/export (JSONL/Parquet) |
| **P1a-6 CLI/SDK/docs** | `init/prepare/index/plan/deploy/run/status/resume/results/report/cancel`, SDK `submit/Job`, quickstart |
| **Chaos (coordinator + 1 subagent)** | `simulate_preemption()` ×5, OOM injection, orchestrator kill, budget stop/resume on a 1,000-episode job (≈$20–40, approval) |

Review: every branch gets an independent re-review subagent before merge (as with Fix-A/C).

---

## After the current fixes land (plan, 2026-10-21)

Budget: $50 total, $10.25 spent, **$39.75 left**. Sequence and caps:

| Step | What | Parallel subagents | Cap |
|---|---|---|---|
| **1. Integrate the fast `exact` worker** (Fix-D) | ReferenceWorker = NVIDIA's own `annotate()` with `CachedSession` injected (no re-expressed glue) + OPT-1 patches (decode-once, in-memory PNG) + OPT-2 BF16 Captioner + W3-B no-copy layout and prefetch + W3-A GPU-lock concurrency, on the cu128 image. **Prove the composite is bit-identical to unmodified NVIDIA** on ~6 golden episodes, then measure startup and throughput. | 1 | $3 |
| **2. Golden build → Phase 0 done** | Oracle on the 42 staged golden episodes, 2 runs (self-noise) with the proven-identical fast worker; Rewarder reference on the 12 reward episodes. Write `docs/measurements-p0.md`, decide Gate D, PLAN v4 cost model from measured rates. | 1 + coordinator | $8 |
| **3. P1a → v0.1 `exact`** (starts in parallel with step 2; mostly CPU and fake Modal) | On contracts v2: **P1a-1** config/presets/D11 hash · **P1a-2** stores + manifest/shard packing + quarantine/bisection · **P1a-3** orchestrator (nonpreemptible, Dict lease, continuation) + reconciler · **P1a-4** budget + `flora plan` / `flora estimate` · **P1a-5** shard workers (`segcap` exact, reward) + compose/export · **P1a-6** CLI/SDK/quickstart. Every branch re-reviewed. Then end-to-end on a golden subset and a chaos run (`simulate_preemption` ×5, OOM episode, orchestrator kill, budget stop/resume). | 6 + reviewers | $7 |
| **4. P1b → v0.2 `balanced`** (prep starts in parallel) | TempSeg FP32 merge → BF16 on CPU; vLLM engines with micro-batching; `vllm_prompt` + processor cache off + 32k budget; certify segment/caption/dense against the step-2 oracle (L0–L3, Gates A/B/E) on L40S/H100. | 3–4 | $10 |
| **5. Upstream to NVIDIA** | Draft suggestions (BF16 Captioner, wrapped video tokens, decode reuse, apredict docs, CUDA wheel note); you post them. | 1 | $0 |
| Reserve | Re-runs and surprises | — | ~$8 |
