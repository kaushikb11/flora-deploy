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
