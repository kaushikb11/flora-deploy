# Golden set v1: staging summary (W3-C)

Staged 2026-10-21 into the private Volume `flora-golden` under `golden/v1/` by
`modal run src/flora_deploy/modal_app/golden_app.py` (CPU only, ephemeral app
`flora-golden-stage`, one container, 323 s wall, 157 Hub requests, sequential).
The selection is data: `src/flora_deploy/validate/golden_selection_v1.json`. The basis is
`docs/golden-candidates.md`, cut to the P0 budget of about 30 TempSeg/Captioner episodes plus
about 12 reward episodes.

This page has numbers only. Videos, task text, annotations and every NC-licensed derivative
stay on the Volume. Per-episode details are in `golden/v1/<pool>/<dataset>/<slug>/record.json`
and in `golden/v1/manifest.json`. There is no NVIDIA material here.

## Counts and eligibility

| Pool | Selected | Staged | Oracle-eligible | Duration (s, total) | Duration range (s) |
|---|---|---|---|---|---|
| `human` | 7 | 7 | 7 | 239.8 | 14.6-61.5 |
| `single_view_robot` | 6 | 6 | 6 | 125.7 | 12.6-36.5 |
| `yam` | 9 | 9 | 9 | 223.3 | 8.0-53.3 |
| `yam32` | 2 | 2 | 2 | 204.6 | 88.5-116.0 |
| `dex` (stand-in) | 4 | 4 | 4 | 83.6 | 17.2-23.9 |
| `tianji` (stand-in) | 2 | 2 | 2 | 51.1 | 23.7-27.3 |
| `reward` | 12 | 12 | 12 | 151.7 | 2.1-24.7 |
| **total** | 42 | 42 | 42 | 1079.6 | |

Every episode passed all of these:
- `verify_episode`: metadata ranges match the real sample tables, and frame counts equal LeRobot `length`;
- the decode-mode index (`index_episode(decode=True)`);
- NVIDIA's envelope, with the episode's profile;
- `cut_check`: every staged file was re-indexed and equals the source-range index in rate, frames, duration, size, CFR, codec, pix_fmt and the PTS digest.

There were no ineligibility reasons, no source errors and no failed downloads. The 30
TempSeg/Captioner episodes come to 928 s of video, or 15.5 min per view.

Reward episodes get NVIDIA's per-file stream checks per view, not the TempSeg role and envelope
checks. The views of each RoboArena episode have identical PTS.

## What is staged

| Kind | Episodes | Rule |
|---|---|---|
| `stream_copy_cut` (`media.cuts`) | 15 (EgoDex, droid_success, ALOHA, Unitree G1, tianji, yam-01) | LeRobot v3: concatenated file downloaded, episode cut, source file deleted |
| `source_file` | 27 (WGO-Bench embedded MP4s, ManiRewardBench, RoboCOIN v2.1, RoboArena, RoboRewardBench) | per-episode files kept as is |

That is 684 MiB of video in total. In addition, `golden/v1/_datasets/aloha_static_towel/meta/temporal_proportions_*.json`
holds dense-progress priors.

## Media profile

- **Codecs:** 27 H.264 and 15 AV1.
  - AV1: EgoDex, Galaxea (WGO and RoboCOIN), ALOHA, Unitree G1, yam-01, organize_desk_fail.
  - The rubric/reward set is all H.264.
- **Resolutions (main view):**

  | Resolution | Episodes |
  |---|---|
  | 640x480 | 14 |
  | 1280x720 | 6 |
  | 512x288 | 6 |
  | 1920x1080 | 5 |
  | 680x480 | 4 |
  | 320x192 | 3 |
  | 320x180 | 2 |
  | 848x478 | 1 |
  | 640x360 | 1 |

- **Frame rates:** 10, 15, 30 and 50 fps CFR. Three of the four HomER rows (human) are
  **VFR**, with non-integer `average_rate`. Two of those are 1080p and one is 720p; only
  `homer_50` matches the 848x478 that WS-F spot-checked. Human single view does not require CFR,
  so they stay eligible. They are useful as VFR/odd-rate cases.
- **Longest episodes:** yam32 at 88.5 s (yam-01, 2656 frames) and 116.0 s (heat_burger,
  3481 frames, just under the 120 s cap).
- **RoboRewardBench temporal clip:** the selected clip
  (`bridge_train_118_attempt_1_score_3`, 21 frames) turned out to be CFR. The VFR clipped
  files that WS-F saw are not in the set.

## Licenses (episodes)

| License | Episodes | Sources |
|---|---|---|
| Apache-2.0 | 16 | ManiRewardBench, ALOHA (towel/cabinet), RoboCOIN, Unitree G1, tianji, yam-01 |
| CC-BY-4.0 | 8 | jnogga/droid_success, RoboRewardBench |
| CC-BY-NC-SA-4.0 | 8 | WGO-Bench (HomER, DROID and Galaxea rows) |
| MIT | 7 | RoboArena, aloha_static_cups_open |
| CC-BY-NC-ND-4.0 | 3 | EgoDex |

NC/NC-ND cuts stay on the private Volume. Certificates publish metrics and hashes only.

## Coverage notes

- **Hard cases tagged in the selection:**

  | Hard case | Sources |
  |---|---|
  | many short actions | HomER 27 segments in 46 s; Galaxea 12 in 26 s; ManiRewardBench stack_plates 6 in 21 s |
  | repetition | EgoDex braid, add/remove lid; tianji |
  | failed demo / retries | organize_desk_fail; RoboArena `018316ac_C` |
  | camera file rollover | aloha_mobile_cabinet ep 81 |
  | near the 120 s cap | heat_burger |
  | moving head camera | G1 Inspire |

  Explicit handover data was not traced (the RoboCOIN transfer tasks are left for later).
  `cap_the_glue_stick` is the closest bimanual-coordination case.
- **Reward set:**
  - 6 RoboArena terminal episodes: 3 successes and 3 failures, with partial success 0.0, 0.7
    and 0.75; one has 3 views (L/C/R).
  - 6 RoboRewardBench rubric episodes: one per score 1-5, plus a temporal clip.
  - Dense targets reuse the 4 ManiRewardBench episodes (timed subtasks) and aloha_static_towel
    (shipped priors), so they are not separate episodes.
- **Profiles:** single-view robot episodes use profile `yam`. Dex and tianji remain stand-ins
  (Unitree G1, DingYFa `tianji_arm`).
- **Role semantics:** wrist/head roles were assigned from camera names, not checked visually.
- **No substitutions were needed.** Every candidate dataset was ungated and available at the
  pinned revision.

## Known media/ issue (worked around, not fixed here)

`LeRobotV3Source` globs `meta/episodes/chunk-*/file-*.parquet`. `jnogga/droid_success` writes
`file_000.parquet` (underscore; its `info.json` templates also use `file_{file_index:03d}`). The
reader then raises "no meta/episodes". `validate/golden.py` renames the file inside its private
work mirror. The reader should accept both spellings, or derive the path from `info.json`.
