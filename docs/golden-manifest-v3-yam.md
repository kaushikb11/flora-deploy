# Golden set v3-yam: the `yam` (3-view) certification set

Staged 2026-10-22 into the private Volume `flora-golden` under `golden/v3-yam/` by
`uv run python scripts/golden_stage.py --version golden/v3-yam`. That script calls
`modal_app/golden_app.py`'s `stage` function with the v3-yam selection: CPU only, ephemeral app
`flora-golden-stage`, one container (2 cores, 8 GiB), 99.0 s in the container (113 s app wall),
50 Hub requests.

The selection is data: `src/flora_deploy/validate/golden_selection_v3_yam.json`. It exists for
one purpose: the **yam certified run** (`scripts/wave_h.py --plan certified --embodiment yam`,
`docs/wave-h-runbook.md`), which certifies `balanced` for the `yam` embodiment. v1 and v2 are
unchanged; v3-yam never writes under `golden/v1/` or `golden/v2/`.

This page has numbers only. Videos, task text, annotations and every NC-licensed derivative
stay on the Volume (`golden/v3-yam/<pool>/<dataset>/<slug>/record.json`,
`golden/v3-yam/manifest.json`). There is no NVIDIA material here.

## Why these 12 episodes

`flora certify` needs at least 10 candidate episodes per pool, every oracle episode labelled,
and a noise run of at least 5 of them (`docs/certification.md`). Certificates are per (preset,
kind, GPU, embodiment), so the human certificate from golden/v2 does not cover `yam`. v1 has 9
`yam` episodes. v3-yam has **12**: 10 for the certificate plus 2 spares.

- **Only datasets v1 already pins**, at the same revisions. No new dataset was added.
- **Kept from v1: 8 of its 9 `yam` episodes.** Their selection entries and datasets are
  byte-for-byte the v1 ones, so they have the same selection digest. The verify pass found
  every one of their 24 files **byte-identical** to its v1 copy (sha256).
- **Left out:** `robocoin_organize_desk_fail/episode_000078` (53.3 s, the failed-demo case),
  for cost. A 3-view second costs about 1.3x (exact) to 2.6x (balanced TempSeg) a 1-view one.
- **New: 4 short ManiRewardBench YAM episodes** (`ajyanggg/manirewardbench_bimanual_yam@a70a270d…`),
  each the shortest or near-shortest episode of a task v1 does not have. They are 10.7-13.8 s
  long, with 3-4 timed subtasks each.
- **Not usable here:** the WGO-Bench robot rows (DROID, Galaxea) and `droid_success` are
  single view. They are the `single_view_robot` pool, not 3-view `yam`.

The mix: 3 ALOHA episodes (AV1, 50 fps), 1 RoboCOIN Galaxea R1 Lite episode (AV1, 720p, 30
fps) and 8 real-YAM ManiRewardBench episodes (H.264, 10 fps), over 12 different tasks.

## Counts and eligibility

| Pool | Selected | Staged | Oracle-eligible | Duration (s, total) | Duration range (s) |
|---|---|---|---|---|---|
| `yam` | 12 | 12 | 12 | 220.2 | 8.0-30.0 |

Every episode has three views (`main`, `left_wrist`, `right_wrist`) with profile `yam`. In every
episode, the three views have the same size, frame count and PTS digest.

| Episode | Source | Codec | Size (each view) | fps | Frames | Duration (s) | Kind | v1 |
|---|---|---|---|---|---|---|---|---|
| `aloha_static_cups_open/episode_000000` | ALOHA static | AV1 | 640x480 | 50 | 400 | 8.00 | stream_copy_cut | kept |
| `aloha_static_towel/episode_000000` | ALOHA static | AV1 | 640x480 | 50 | 500 | 10.00 | stream_copy_cut | kept |
| `aloha_mobile_cabinet/episode_000081` | ALOHA mobile | AV1 | 640x480 | 50 | 1500 | 30.00 | stream_copy_cut | kept |
| `robocoin_classify_object_four/episode_000164` | RoboCOIN Galaxea | AV1 | 1280x720 | 30 | 577 | 19.23 | source_file | kept |
| `manirewardbench_yam/stack_plates_000003` | ManiRewardBench | H.264 | 680x480 | 10 | 220 | 22.00 | source_file | kept |
| `manirewardbench_yam/build_a_pyramid_000004` | ManiRewardBench | H.264 | 680x480 | 10 | 264 | 26.40 | source_file | kept |
| `manirewardbench_yam/set_table_000001` | ManiRewardBench | H.264 | 680x480 | 10 | 269 | 26.90 | source_file | kept |
| `manirewardbench_yam/cap_the_glue_stick_000002` | ManiRewardBench | H.264 | 680x480 | 10 | 274 | 27.40 | source_file | kept |
| `manirewardbench_yam/stack_the_plates_000002` | ManiRewardBench | H.264 | 680x480 | 10 | 107 | 10.70 | source_file | new |
| `manirewardbench_yam/take_out_the_chewing_gum_000002` | ManiRewardBench | H.264 | 680x480 | 10 | 123 | 12.30 | source_file | new |
| `manirewardbench_yam/remove_the_cap_000005` | ManiRewardBench | H.264 | 680x480 | 10 | 138 | 13.80 | source_file | new |
| `manirewardbench_yam/put_the_pocky_in_the_bin_000005` | ManiRewardBench | H.264 | 680x480 | 10 | 135 | 13.50 | source_file | new |

Duration mix: 6 episodes of 8-14 s, 2 of 19-22 s and 4 of 26-30 s. The staged ManiRewardBench
files run 0.1-0.7 s longer than the last annotated subtask (the selection's `expected_s`), so
the dry run reads the manifest.

All 36 files total 79.5 MiB. Hard cases carried over from v1: a camera file rollover
(`aloha_mobile_cabinet` ep 81), many short actions (`stack_plates`), bimanual coordination
(`cap_the_glue_stick`, and now its inverse `remove_the_cap`).

## Noise subset

The selection's `noise_subset` names the 5 episodes the noise run repeats:
`aloha_static_cups_open/000000`, `robocoin_classify_object_four/000164`,
`manirewardbench_yam/stack_plates_000003`, `stack_the_plates_000002` and
`take_out_the_chewing_gum_000002`, 72.2 s in all. It mixes all three sources (ALOHA AV1 at 50
fps, RoboCOIN 720p AV1, ManiRewardBench H.264 at 10 fps), so the noise floor is not measured on
one source only.

## Verification

`scripts/golden_stage.py` verifies after staging (Volume reads from the client, no container),
and `--verify-only` repeats the check. It found:
- 12/12 episodes `staged`, `oracle_eligible`, `verify_episode` ok and `cut_check` ok;
- **every file (36) read back from the Volume** with its size and sha256 equal to the
  manifest's;
- the manifest on the Volume equal to the one the staging call returned;
- the 8 kept episodes byte-identical to their `golden/v1` files (`--compare golden/v1`, the
  default for every version after v1).

There were no source errors, no failed downloads and no substitutions.

## Cost

- **Tracked:** 113 s of app wall × the pinned list price of 2 cores + 8 GiB = **$0.005**.
- **Billed:** `scripts/spend.py --app flora-golden-stage` still showed the earlier stagings'
  $0.03 right after the run, because Modal billing lags about an hour. Total workspace spend
  read $26.10 before.

## Licenses

| License | Episodes | Source |
|---|---|---|
| Apache-2.0 | 11 | ManiRewardBench (8), `aloha_static_towel`, `aloha_mobile_cabinet`, RoboCOIN |
| MIT | 1 | `aloha_static_cups_open` |

None of them is NC. The cuts stay on the private Volume anyway, and certificates publish
metrics and hashes only: no episode id, caption or task text.

## The other robot embodiments: what pinned data could cover them

These are **not staged**. "Oracle-eligible" counts only episodes the staging pipeline verified;
the larger numbers are from `docs/golden-candidates.md` (metadata and moov spot checks), not
from the index pass.

| Pool | Pinned datasets (v1) | Verified eligible today | Candidates (spot-checked) | What is missing |
|---|---|---|---|---|
| `dex` | Unitree G1 Dex3 ToastedBread (418 eps, 11-75 s), G1 Inspire Pick_Up_Drinks (300 eps, 22-91 s) | 4 (v1) | 148 + 70 episodes with equal cross-camera timing | 6+ more staged episodes. Both are **stand-ins** (Unitree hands, not Flora's dex rig), AV1 only, and 2 tasks in all |
| `tianji` | DingYFa `tianji_pick_and_place_cubes` (110 eps, 24-60 s) | 2 (v1) | 60 episodes with equal cross-camera timing | 8+ more staged episodes, from **one task** only. Rig family not verified as Flora's `tianji`. Gate C's tianji quota is also short (5 of 6 TempSeg requests) |
| `yam32` | allenai yam-01 (50 eps, 44-88 s, one task), RoboCOIN heat_burger (819 eps, 79-116 s, 141 GB), long ManiRewardBench episodes (up to 58.5 s) | 2 (v1) | yam-01's 50 (9/9 moov checks); heat_burger's 3 checked | 8+ more staged episodes. No dedicated yam32 data: the same YAM/ALOHA-style data restricted to long episodes. Long episodes make it the most expensive pool |

Rough certified-run prices, on the same rates and three-view scaling as yam (12 episodes, 5 of
them for noise, lab already done, hypothetical durations at the short end of each dataset,
+10% headroom). Nominal / upper:

| Pool | Video s | (b) oracle | (f) noise | (d) balanced | Total |
|---|---|---|---|---|---|
| `dex` | 242 | $3.40 / $5.05 | $1.29 / $2.20 | $1.59 / $2.76 | **$6.27 / $10.01** |
| `tianji` | 348 | $4.80 / $6.95 | $1.99 / $3.15 | $1.86 / $3.16 | **$8.65 / $13.26** |
| `yam32` | 751 | $10.11 / $14.11 | $3.69 / $5.45 | $2.88 / $4.69 | **$16.67 / $24.25** |

yam32's TempSeg prompts are longer again (64 temporal groups, about 10.7k tokens, Gate C), which
these numbers do not include, so its price is a floor. Until one of these pools has a golden
set and a certificate, its users need `allow_uncertified: true`.
