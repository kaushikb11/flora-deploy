# Golden set v2: the `human` certification set

Staged 2026-10-22 into the private Volume `flora-golden` under `golden/v2/` by
`uv run python scripts/golden_stage.py --version golden/v2`. That script calls
`modal_app/golden_app.py`'s `stage` function with the v2 selection: CPU only, ephemeral app
`flora-golden-stage`, one container (2 cores, 8 GiB), 161.5 s in the container (182 s app
wall), 37 Hub requests.

The selection is data: `src/flora_deploy/validate/golden_selection_v2.json`. It exists for one
purpose: the **certified run** (`scripts/wave_h.py --plan certified`, `docs/wave-h-runbook.md`),
which certifies `balanced` for the `human` embodiment. v1 (`docs/golden-manifest-v1.md`) is
unchanged, and v2 never writes under `golden/v1/`.

This page has numbers only. Videos, task text and every NC-licensed derivative stay on the
Volume (`golden/v2/<pool>/<dataset>/<slug>/record.json`, `golden/v2/manifest.json`). There is
no NVIDIA material here.

## Why 12 episodes

`flora certify` needs at least 10 candidate episodes per pool and every oracle episode labelled
(`docs/certification.md`). v1 has only 7 `human` episodes, and Wave H-lite step (d) ran 6 of
them. v2 has **12**: 10 for the certificate plus 2 spares, so a failure or exclusion at run time
still leaves 10. The certified run sends all 12 to the oracle and to `balanced`.

- **Kept from v1: the 6 episodes step (d) ran.** Their selection entries and datasets are
  byte-for-byte the v1 ones, so they have the same selection digest. Staging re-cut them under
  `golden/v2/`, and the verify pass found every one of their files to be **byte-identical** to
  its v1 copy (sha256).
- **New: 6 short EgoDex v3.0 test episodes**, from the same repo at the same pinned revision
  (`griffinlabs/EgoDex-LeRobot-v3.0@41d60b44…`). Each comes from a different task, and each task
  has a single small concatenated file (14-121 MB to download). They are 7-13 s long, which
  keeps the oracle cheap (about $0.01 per video second on H200), while the v1 HomER rows keep
  the 38-47 s tail. That gives a realistic mix of 6 short clips, 3 medium ones and 3 long ones.
- **Left out:** `homer_40` (59.6 s), for cost, as in Wave H-lite.

## Counts and eligibility

| Pool | Selected | Staged | Oracle-eligible | Duration (s, total) | Duration range (s) |
|---|---|---|---|---|---|
| `human` | 12 | 12 | 12 | 235.9 | 7.1-47.1 |

| Episode | Source | Codec | Size | fps | Frames | Duration (s) | CFR | Kind | v1 |
|---|---|---|---|---|---|---|---|---|---|
| `egodex_add_remove_lid/episode_000014` | EgoDex | AV1 | 1920x1080 | 30 | 439 | 14.63 | yes | stream_copy_cut | kept |
| `egodex_braid_unbraid/episode_000001` | EgoDex | AV1 | 1920x1080 | 30 | 482 | 16.07 | yes | stream_copy_cut | kept |
| `egodex_make_sandwich/episode_000002` | EgoDex | AV1 | 1920x1080 | 30 | 479 | 15.97 | yes | stream_copy_cut | kept |
| `wgo/homer_52` | WGO HomER | H.264 | 1920x1080 | VFR | 1140 | 37.99 | no | source_file | kept |
| `wgo/homer_53` | WGO HomER | H.264 | 1920x1080 | 59.94 | 2824 | 47.11 | yes | source_file | kept |
| `wgo/homer_50` | WGO HomER | H.264 | 848x478 | VFR | 1394 | 46.48 | no | source_file | kept |
| `egodex_pour/episode_000026` | EgoDex | AV1 | 1920x1080 | 30 | 214 | 7.13 | yes | stream_copy_cut | new |
| `egodex_zip_unzip_bag/episode_000012` | EgoDex | AV1 | 1920x1080 | 30 | 255 | 8.50 | yes | stream_copy_cut | new |
| `egodex_screw_unscrew_bottle_cap/episode_000019` | EgoDex | AV1 | 1920x1080 | 30 | 258 | 8.60 | yes | stream_copy_cut | new |
| `egodex_stack_unstack_cups/episode_000007` | EgoDex | AV1 | 1920x1080 | 30 | 270 | 9.00 | yes | stream_copy_cut | new |
| `egodex_insert_remove_usb/episode_000015` | EgoDex | AV1 | 1920x1080 | 30 | 347 | 11.57 | yes | stream_copy_cut | new |
| `egodex_fold_unfold_paper_basic/episode_000011` | EgoDex | AV1 | 1920x1080 | 30 | 386 | 12.87 | yes | stream_copy_cut | new |

Duration mix: 6 episodes of 7-13 s, 3 of 15-16 s and 3 of 38-47 s. `homer_52` is 38.0 s staged,
against the 31.9 s the selection expected, so the dry run reads the manifest, not the
selection.

All 12 files total 191 MiB. Every episode is single view (`main`) with profile `human`. Human
single view does not require CFR, so the two VFR HomER rows stay eligible.

## Noise subset

The selection's `noise_subset` names the 5 episodes the noise run repeats: `wgo/homer_52`,
`egodex_braid_unbraid/000001`, `egodex_zip_unzip_bag/000012`,
`egodex_stack_unstack_cups/000007` and `egodex_fold_unfold_paper_basic/000011`, 84.4 s in all.
The set mixes one VFR HomER row with four EgoDex clips of three lengths, so the noise floor is
not measured on one source only. The other HomER rows are left out to keep the noise run cheap.

## Verification

`scripts/golden_stage.py` verifies after staging (Volume reads from the client, no container),
and `--verify-only` repeats the check. It found:
- 12/12 episodes `staged`, `oracle_eligible`, `verify_episode` ok and `cut_check` ok;
- **every file read back from the Volume** with its size and sha256 equal to the manifest's;
- the manifest on the Volume equal to the one the staging call returned;
- the 6 kept episodes byte-identical to their `golden/v1` files.

There were no source errors, no failed downloads and no substitutions.

## Cost

- **Tracked:** 182 s of app wall × the pinned list price of 2 cores + 8 GiB = **$0.008**.
- **Billed:** `scripts/spend.py --app flora-golden-stage` still showed the earlier v1 staging's
  $0.02 at the time of writing, because Modal billing lags about an hour. Total workspace spend
  read $24.47 before and after.

## Licenses

| License | Episodes | Source |
|---|---|---|
| CC-BY-NC-ND-4.0 | 9 | EgoDex (`griffinlabs/EgoDex-LeRobot-v3.0`) |
| CC-BY-NC-SA-4.0 | 3 | WGO-Bench HomER rows (upstream toloka/HomER has no license field) |

The cuts stay on the private Volume. Certificates publish metrics and hashes only: no episode
id, caption or task text.
