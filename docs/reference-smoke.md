# WS-D reference smoke: NVIDIA's pipeline on one Modal H200

> **Superseded in part by S1 (`docs/exact-worker.md`).** The worker now runs NVIDIA's own
> `api.DataEnginePipeline().annotate` (the `annotate_reference` re-expression below is gone).
> It uses the W3-B link farm instead of the 109 GB copy and the BF16 Captioner, and it has no
> watchdog: runtime is bounded by Modal's timeouts. Measurements on this page are from the WS-D run.

Status: first run 2026-10-21, run id `wsd-20261021T052758Z`. Numbers only. The full results
(segments, captions, NVIDIA's audit with prompt hashes) stay on the `flora-golden` Volume under
`reference_runs/<run_id>/<episode>.json` and are never committed.

## What runs

- `src/flora_deploy/modal_app/reference_app.py`: ephemeral app `flora-reference`.
  - `ReferenceWorker` runs on `img_hfref` (NVIDIA's `requirements.txt` exactly) on an `H200`, with
    `cpu=8`, 96 GiB, `block_network=True`, the weights Volume read-only and `flora-golden`
    read-write. A watchdog kills the container after 44 min. Timeouts are 25 min for startup and
    20 min per call.
  - `@modal.enter` links NVIDIA's SEGCAP code from the Volume (`load_family`), checks that
    `models.json` names exactly our pins, then loads TempSeg and then the Captioner through
    NVIDIA's own `Session.load`, wrapped in `CachedSession`. Both models stay resident.
  - `annotate(videos, profile, long_windows=False, instruction="")` calls
    `stages/reference.annotate_reference`. That is the minimal re-expression of NVIDIA's
    `api.annotate` (`api.py:44-58`) and `pipeline.annotate` (`pipeline.py:141-169`), flagged for
    license review. It calls NVIDIA's `_inputs` (probe, size cap, `check_synchronized`),
    `pipeline.segment`, and `caption_single` or `caption_three`. It returns NVIDIA's result dict
    with the audit always kept, plus a `flora_reference` key with:
    - per-stage timings;
    - token counts parsed from NVIDIA's own `FLORA_RUNTIME` stdout events;
    - GPU memory.
  - `stage_samples` (CPU) stages the samples and checks them with NVIDIA's `probe` and
    `check_synchronized`. `offline_check` (CPU, network blocked) dry-runs the offline
    resolution.
- `scripts/reference_smoke.py` (since removed): `uv run python scripts/reference_smoke.py [--summary PATH]`.

## How offline weight loading works

NVIDIA's `snapshot(kind)` calls `snapshot_download(..., local_dir=MODEL_CACHE/kind)` with a
commit-sha revision. Offline, `huggingface_hub` returns the existing files using the tree
listing and download metadata that prep cached in `<snapshot>/.cache/huggingface/`. It does take
a `filelock` *inside* `local_dir`, though. A symlink to the read-only Volume therefore fails with
EROFS; this was measured.

So the worker:

- copies the TempSeg and Captioner snapshots to local disk at `FLORA_MODEL_CACHE=/tmp/flora-offline/models/<kind>`
  with `copy2`, which keeps mtimes so the metadata stays valid. The Captioner copy overlaps the
  TempSeg load.
- points `HF_HUB_CACHE` at a tmp cache whose `models--Qwen--Qwen3.5-9B/snapshots/<sha>` and
  `trees` are symlinks to the base snapshot on the Volume. In cache mode an existing pointer
  returns without a lock.

`HF_HUB_OFFLINE=1` and `block_network=True` mean nothing can be downloaded. NVIDIA's
`validate_manifest` (it fully hashes the local copy against `SOURCE_MANIFEST.sha256`) and its
GDN, adapter and load checks all run unmodified and pass (`gdn_values_exact`, `adapter_values_exact`,
`gdn_fp32` all true).

## Samples (`flora-golden:/samples/wsd/`, manifest at `samples/wsd/manifest.json`)

| Episode | Source | License | Views | Probe (NVIDIA) |
|---|---|---|---|---|
| `homer_52`, `homer_50`, `homer_53` | `macrodata/WGO-Bench@9cfb7bb9` `data/annotations.parquet` (HomER rows, from `toloka/HomER`) | CC-BY-NC-SA-4.0 | main | 38.0 s / 46.5 s / 47.1 s; 30, 30 and 59.94 fps; 1080p, 848×478 and 1080p; the first two are VFR |
| `aloha_static_coffee_ep000000` | `lerobot/aloha_static_coffee@7da22eb6` (tag `v2.1`), episode 0: `cam_high→main`, `cam_left_wrist→left_wrist`, `cam_right_wrist→right_wrist`; `cam_low` dropped | MIT | 3 | 22.0 s, 50 fps, 640×480, CFR, identical PTS (passes `check_synchronized`) |

Selection: the three shortest HomER clips in 20–90 s that pass NVIDIA's `probe`. 11 of the 25
HomER clips exceed 120 s or 7,200 frames. The instruction is blank, which is NVIDIA's CLI
default.

## Measurements

Load, once per container:

| Step | Time |
|---|---|
| TempSeg copy (0.37 GB) + NVIDIA code link | 11 s |
| TempSeg `Session.load` (manifest 0.3 s; base BF16 on CPU + FP32 LoRA + `.to(cuda)`) | 53.7 s |
| Captioner copy, Volume → local disk (109.4 GB, ≈0.9 GB/s; overlapped with the TempSeg load) | 122.6 s |
| Captioner `Session.load`: NVIDIA manifest hash of 109.4 GB | 101.5 s |
| Captioner `Session.load`: weights to GPU, BF16 | 22 s |
| Captioner `Session.load`: total | 124.9 s |
| `@modal.enter` total | 250.6 s |

GPU memory, `torch` allocated:

| Point | Allocated | Notes |
|---|---|---|
| After TempSeg | 17.9 GiB | |
| After the Captioner | 68.8 GiB | `nvidia-smi` 76.8 GB |
| High-water mark over all episodes | 80.95 GiB | the 3-view caption |

Per episode:

| Episode | Dur. s | TempSeg windows | Segments / captions | Probe s | Segment s (gen s) | Caption s (gen s) | Out tokens seg / cap | Max input tokens seg / cap |
|---|---|---|---|---|---|---|---|---|
| homer_52 | 38.0 | 9 | 11 / 11 | 13.6 | 142.9 (43.2) | 268.6 (32.8) | 544 / 165 | 1,811 / 8,505 |
| homer_50 | 46.5 | 4 | 20 / 20 | 2.3 | 37.5 (30.3) | 133.0 (53.2) | 446 / 387 | 1,811 / 8,504 |
| homer_53 | 47.1 | 8 | 13 / 13 | 15.7 | 124.2 (38.2) | 322.5 (30.4) | 601 / 148 | 1,811 / 8,504 |
| aloha 3-view (YAM) | 22.0 | 2 | 6 / 6 | 4.9 | 20.9 (11.7) | 67.1 (24.1) | 148 / 88 | 5,493 / 12,131 |

- Every caption is `complete`, and no generation was incomplete.
- Retry windows fired on homer_52 and homer_53: 5 and 4 segments are `fine_repaired_by_fine_retry`.
- The container ran 1,411 s in total, about 4 min of it loading.

**Model time is a minority of wall-clock.** On the 1080p clips, generation is 9–31% of each
stage. The rest is CPU work inside NVIDIA's code: full-resolution decode per call,
`human_transport` JPEG, PNG writes of every caption frame, and the HF processor. This is direct
evidence for Gate D.

## Spend

| Item | Cost |
|---|---|
| H200 container, ≈24 min at $4.54/h | ≈$1.8 |
| 8 cores + 96 GiB for the same time | ≈$0.5 |
| CPU functions (image build, staging, offline checks) | <$0.1 |
| **Total** | **≈$2.4**, inside the $10 cap |

## Problems found

- `huggingface_hub` takes a filelock inside `local_dir` even for cached files, so NVIDIA's
  `snapshot()` can't read a read-only Volume directly. We work around it with a local copy:
  ≈2 min, overlapped, plus 109 GB of local disk.
- NVIDIA's Captioner `validate_manifest` re-hashes 109 GB on every container start (≈100 s on
  H200 time). There is no way to skip it without changing NVIDIA's code.
- (Fixed since.) The numpy/pyarrow mismatch in `img_cpu`: `images.CPU_PACKAGES` now pins
  numpy exactly.
- PEFT warns that `base_model_name_or_path` was renamed to the local base path. This is
  harmless: NVIDIA's loader asserts that the adapter values are exact.
