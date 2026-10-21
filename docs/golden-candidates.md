# Golden-set candidates (WS-F)

Research date: 2026-10-21. Scope: public datasets that can stand in for each Flora embodiment
pool in the P0 golden set (archive/PLAN-v3.md §9), checked against the envelope rules in archive/PLAN-v3.md §1.1.
This file holds dataset facts and links only. It contains no NVIDIA prompts or code.

**How the facts were collected.**
- Repo facts (license, gating, file counts, sizes) come from the Hugging Face Hub API (`dataset_info`).
- Format facts come from each dataset's `meta/info.json` and `meta/episodes*` (v3: `meta/episodes/chunk-*/file-*.parquet`; v2.1: `meta/episodes.jsonl`).
- Container facts come from the MP4 `moov` box. It was fetched with HTTP range requests and parsed for `stts`, `ctts`, `stss` and `elst`, which gives per-sample PTS and sync-sample flags without downloading the video data. On two local files the parser matched a PyAV 16.0.1 demux exactly (packet count, keyframe count and per-episode starts).
- Small samples were downloaded in full and inspected with PyAV 16.0.1 (`codec`, `pix_fmt`, color tags, `time_base`, `average_rate`, PTS deltas, keyframes).

The scratch scripts and samples are outside the repo and are not committed.

Definitions used below:
- **Starts on key:** the sample at the episode's `from_timestamp` is a sync sample (`stss`).
- **CFR:** every PTS delta inside the episode is identical.
- **x-cam equal:** episode-relative PTS sequences are identical across the cameras used.

These are spot checks. The WS-C index pass (`flora_deploy.media`) is authoritative, including the boundary-`stts` and `average_rate` rules in `docs/media-notes.md`.

---

## 1. Summary

| Pool | Best candidates | License | Format | Codec / fps / res | Eps ≤120 s | Spot check |
|---|---|---|---|---|---|---|
| `human` | [macrodata/WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) HomER rows | CC-BY-NC-SA-4.0 (upstream HomER has no license) | Parquet with embedded MP4 | H.264 Baseline, 20 fps, 848×478 | 14/25 | CFR, key at 0 ✔ |
| `human` | [griffinlabs/EgoDex-LeRobot-v3.0](https://huggingface.co/datasets/griffinlabs/EgoDex-LeRobot-v3.0) (`test/*`) | CC-BY-NC-ND-4.0 | v3.0 | AV1, 30 fps, 1920×1080 | 100% (3–32 s) | 41/41 starts on key, CFR ✔ |
| single-view robot | WGO-Bench DROID (RoboInter) + Galaxea (RoboCOIN) rows | CC-BY-NC-SA-4.0 | embedded MP4 | DROID: H.264, 10 fps, 320×180. Galaxea: AV1, 30 fps, 1280×720 | 100% | CFR, key at 0 ✔ |
| single-view robot | [jnogga/droid_success](https://huggingface.co/datasets/jnogga/droid_success) (one camera) | CC-BY-4.0 | v3.0 | H.264, 15 fps, 1280×720 | ≈100% | 265/265 starts on key, CFR, x-cam ✔ |
| `yam` (3-view) | [ajyanggg/manirewardbench_bimanual_yam](https://huggingface.co/datasets/ajyanggg/manirewardbench_bimanual_yam) | Apache-2.0 | per-episode MP4 + JSON | H.264, 10 fps, 680×480 | 97/97 | CFR, key at 0, x-cam ✔ |
| `yam` | [allenai/24112025-yam-01](https://huggingface.co/datasets/allenai/24112025-yam-01) (MolmoAct2) | Apache-2.0 | v3.0 | AV1, 30 fps, 640×360 | 50/50 (44–88 s) | 9/9 on key, CFR, x-cam ✔ |
| `yam` stand-in | ALOHA static/mobile (`lerobot/aloha_*`) | MIT / Apache-2.0 (varies per repo) | v3.0 | AV1, 50 fps, 640×480 | 100% | 800+ episode-cams on key, CFR, x-cam ✔. **One metadata error** (§3.3) |
| `yam` stand-in | RoboCOIN Agilex Cobot Magic (`RoboCOIN/Agilex_Cobot_Magic_*`) | Apache-2.0 | v2.1 | AV1 or H.264, 30 fps, 640×480 | 99–100% | on key, CFR, x-cam ✔ |
| `yam32` | Long episodes from the `yam` rows (allenai yam-01, aloha_mobile_shrimp, RoboCOIN heat_burger) | as above | | | | |
| `dex` stand-in | [unitreerobotics/G1_Dex3_ToastedBread_Dataset](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset), [G1_WBT_Inspire_Pick_Up_Drinks_v0](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Drinks_v0) | Apache-2.0 | v3.0 | AV1, 30 fps, 640×480 | 100% | 1,104 episode-cams on key, CFR, x-cam ✔ |
| `tianji` (probable) | [DingYFa/tianji_pick_and_place_cubes_20260911](https://huggingface.co/datasets/DingYFa/tianji_pick_and_place_cubes_20260911) | Apache-2.0 | v3.0 | H.264, 30 fps, 640×480 | 110/110 | 180/180 on key, CFR, x-cam ✔ |
| Reward: terminal + rubric | [teetone/RoboReward](https://huggingface.co/datasets/teetone/RoboReward) `test` (RoboRewardBench) | CC-BY-4.0 | MP4 + `metadata.jsonl` | H.264, 10 fps, ≤640×480 | 100% | key at 0 ✔. **Some clipped files are VFR** |
| Reward: terminal + partial + feedback | [RoboArena/DataDump_07-17-2026](https://huggingface.co/datasets/RoboArena/DataDump_07-17-2026) | MIT | per-session MP4 + YAML | H.264, 10 fps, 512×288 | ≈100% (≈40 s) | CFR, key at 0 ✔ |
| Reward: dense (progress) | ManiRewardBench YAM (subtask timings); `lerobot/aloha_static_towel` (`temporal_proportions_*.json`) | Apache-2.0 | | | | |

Coverage:
- **Covered with real-format data:** `human` and single-view robot (license-limited, see §4), `yam`, and reward terminal, dense and rubric-style.
- **Stand-ins only:** `yam32` (the same yam data, long episodes only), `dex` (Unitree G1 dexterous hands, not Flora's dex rig) and `tianji` (a robot named Tianji, but camera semantics unverified).
- **Uncovered:** the real `dex` and `tianji` reference rigs and their camera conventions. Also missing is a human egocentric set under a permissive license that has task text and ≤120 s clips (§5).

---

## 2. Envelope reminders (archive/PLAN-v3.md §1.1, §9)

Oracle-eligible episodes must meet all of these:
- ≤120 s, ≥4 fps, ≤4096 px per side, ≤7200 frames, ≤256 MiB per file;
- CFR, required for 3-view and single-view robot captions;
- for 3-view, roles exactly `main/left_wrist/right_wrist` with identical frame timestamps;
- a start on a keyframe, so the stream-copy cut works.

The Rewarder's camera order is left, center, right. TempSeg's is main, left wrist, right wrist.

---

## 3. Per-dataset details

### 3.1 Human (egocentric, single view)

**macrodata/WGO-Bench, HomER rows** (Flora reports on WGO-Bench)
- One Parquet file, `data/annotations.parquet` (1.40 GB, 100 rows, one row per row group, MP4 bytes in the `video` column). The schema has `id`, `video`, `instruction`, `segments[{start_sec,end_sec,subtask}]`, `metadata` (JSON) and `perception_state`.
- Families: 25 `homer_*`, 50 `robointer_droid_*` and 25 `galaxea_*`. There are 743 gold segments.
- License: CC-BY-NC-SA-4.0, ungated. The upstream [toloka/HomER](https://huggingface.co/datasets/toloka/HomER) is only an index of 63 videos hosted on Azure blob URLs. It has **no license field**, and 35/63 videos are ≤120 s (33–178 s). Its task categories include *Object Handoff* (2 videos) and *Cooking Sequences* (9).
- HomER durations in WGO are 31.9–167.5 s, so **14/25 are ≤120 s**. Rows have 6–58 segments each, and many short actions make good hard cases.
- Spot check on `homer_15.mp4` (28.8 MB, "fold the laundry items neatly"): H.264 Baseline, yuv420p, 848×478, tb 1/90000, `average_rate` 20, 2,646 packets, every delta 4500 (CFR), key at pts 0, GOP 15 s. Color tags: range 1 (tv), primaries 5, trc 6, space 6 (BT.601). This episode is 132.3 s, so it is **not** eligible.
- Evidence: `README.md`, `summary.json`, the non-video columns of the Parquet (column projection), and the full video bytes of 4 rows.

**griffinlabs/EgoDex-LeRobot-v3.0** (Apple EgoDex, converted)
- License CC-BY-NC-ND-4.0 (inherits EgoDex's), ungated. 1.47 TB total; `test/` is 15.4 GB in 111 sub-datasets, each its own v3.0 root.
- Camera: one `observation.images.camera`, AV1 yuv420p, 1920×1080, 30 fps.
- Tasks are per-episode text like "basic fold: Fold a medium-sized tshirt on a blue tablecloth…".
- `test/add_remove_lid`: 26 episodes, 3.0–16.3 s. `test/basic_fold`: 77 episodes, 6.3–31.6 s. 100% are ≤120 s.
- Spot check on `test/basic_fold` file-000, via moov: 41/41 episode starts on sync samples, all CFR. GOP is 2 frames, with 20 single-frame GOPs at joins.
- Evidence: `meta/info.json`, `meta/episodes/*`, and the moov of one video file per sub-dataset checked.

**Rejected or deferred**
- [builddotai/Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K): Apache-2.0 but gated (form with phone number). H.265 1080p 30 fps fisheye in WebDataset tars. Median clip is 180 s (>120 s), and there are no task instructions. The [Voxel51 subset](https://huggingface.co/datasets/Voxel51/Egocentric_10K_subset) (Apache-2.0, ungated, 416 samples) has the same issues apart from gating.
- `EgoSteer/EgoSteer-Egocentric`: "other" license, aggregates Ego4D/EPIC and others with their terms.
- Ego4D: heavy gating.

### 3.2 Single-view robot

**WGO-Bench DROID rows** (source `InternRobotics/RoboInter-Data`, LeRobot v2.1 tar members)
- 50 rows, 10.4–107.3 s, all ≤120 s, with 2–8 segments each. Durations come in identical pairs, and `source_episode_id` ends in `_exterior_image_2_left`. Rows are probably two views or two annotations of 25 episodes; dedupe by `source_episode_id` before selection.
- Spot check on `robointer_droid_000001` (0.15 MB): H.264 High, 320×180, 10 fps, tb 1/10240, CFR, a single keyframe at 0, color tags BT.709 unspecified range (`0`).

**WGO-Bench Galaxea rows** (source `RoboCOIN/Galaxea_R1_Lite_*`, camera `cam_head_left_rgb`)
- 25 rows, 4.3–51.5 s, with 2–17 segments each.
- Spot check on `galaxea_002` and `galaxea_045`: AV1 (libdav1d), Main, yuv420p, 1280×720, 30 fps, tb 1/15360, CFR. Keyframe every 2 frames, key at 0. Color range 1.
- The OpenGalaxea release ([OpenGalaxea/Galaxea-Open-World-Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)) is a different source. It is gated (auto), CC-BY-NC-SA-4.0, 2.87 TB in tar.gz, and recorded at 15 fps. **Use the RoboCOIN repos instead.**

**RoboCOIN/Galaxea_R1_Lite_\*** (e.g. [classify_object_four](https://huggingface.co/datasets/RoboCOIN/Galaxea_R1_Lite_classify_object_four))
- Apache-2.0, ungated, v2.1, one MP4 per episode per camera.
- Four cameras (`cam_head_left_rgb`, `cam_head_right_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`), AV1 720p, 30 fps.
- Per-frame `subtask_annotation` and `scene_annotation` fields are present.
- classify_object_four: 191 episodes, 19.2–42.9 s, 100% ≤120 s, 9.0 GB. It is also a 3-view candidate (main = `cam_head_left_rgb`).
- moov check: 12/12 on key, CFR, x-cam equal.

**jnogga/droid_success / jnogga/droid_failure** (DROID 1.0.1 raw, re-ported at full resolution)
- CC-BY-4.0, ungated, v3.0, H.264 1280×720, 15 fps. Cameras: `left_external`, `wrist`, `right_external`.
- Success: 53,282 episodes, p50 14.7 s, 100% ≤120 s, every episode has 3 instruction paraphrases.
- Failure: 13,747 episodes, 99.9% ≤120 s, but **all `tasks` and `task` fields are empty**.
- Video files use the non-default template `file_{file_index:03d}.mp4`.
- moov check (success, file 0): 265/265 on key, CFR, x-cam equal. GOP is variable (88–250 frames), but a keyframe is forced at every episode start.

**lerobot/droid_1.0.1**
- Apache-2.0, v3.0, 95,658 episodes, AV1 **320×180**, 15 fps.
- `language_instruction` (some empty) and a per-frame `is_episode_successful` (data Parquet, not meta).
- 99.9% ≤120 s, 412 GB. Lower resolution than jnogga's; use it only if the 180p WGO variant is wanted.

**RC-Tianji ([gimai/RC-Tianji](https://huggingface.co/datasets/gimai/RC-Tianji), "RoboChrono")**
- CC-BY-4.0. Scenarios cover the `tianji`, `tianjihand` and `hand` embodiments, 50 episodes each. The manifest lists views `main/wrist_left/wrist_right` at 640×480, 30 fps.
- Full-length **episode videos exist only for `main`**. Wrists exist only as stills for VQA. Media sits inside uncompressed tars (`media/episodes.tar`). Subtask ids and `action_time` intervals are given.
- Spot check on `stack_cubes_tianjihand/file-000@main.mp4` (range-read from the tar, 6.9 MB): H.264 High, 640×480, 30 fps, CFR, 24.6 s, **all-intra** (737/737 keyframes).
- Usable as single-view robot on the Tianji embodiment, not as 3-view.

### 3.3 `yam` / `yam32` (3-view bimanual)

**ajyanggg/manirewardbench_bimanual_yam** (ManiRewardBench / TOPReward, real I2RT YAM)
- Apache-2.0, 0.56 GB. 97 episodes over 20 tasks (5 per task). Layout: `YAM/<task>/<ep>/{front_rgb,left_rgb,right_rgb}.mp4` plus a per-frame JSON.
- `yam_annotations.json` holds the task text and timed subtasks (e.g. reach_towel / fold_once / fold_twice). The `mistakes` lists are all empty.
- Episode end times: 10.6–58.5 s, all ≤120 s.
- Spot check (fold_towel/000001, align_the_cubes_horizontally/000001): each of the 3 files is H.264 High, 680×480, 10 fps, tb 1/10240, CFR, with identical packet counts (509/509/509 and 507/507/507) and identical PTS. Key at 0. GOPs differ per camera, which is irrelevant for per-episode files.
- **Role map:** `front_rgb→main`, `left_rgb→left_wrist`, `right_rgb→right_wrist`. Confirm visually that left and right are wrist cameras.

**allenai/24112025-yam-01** (one subset of MolmoAct2-BimanualYAM, real YAM)
- Apache-2.0, v3.0, 50 episodes, 3.0 GB. Cameras `top`, `left`, `right`: AV1, 640×360, 30 fps. Task "Fold the cloth", plus per-episode annotated instructions in `meta/tasks_annotated.parquet`.
- Duration 44–88 s (p50 67.9 s), so these are good `yam32` long-window cases.
- Each episode is its own video file (from_timestamp 0). moov: 9/9 on key, CFR, x-cam equal, GOP 2.
- **Role map:** `top→main`, `left→left_wrist`, `right→right_wrist`.
- Other `allenai/*-yam-*` subsets exist and were not checked.

**ALOHA family on the LeRobot Hub** (all v3.0, AV1 yuv420p 640×480, 50 fps, one task per repo)

| Repo | License | Eps | Duration | Cameras | Size |
|---|---|---|---|---|---|
| [aloha_static_cups_open](https://huggingface.co/datasets/lerobot/aloha_static_cups_open) | MIT | 50 | 8 s | high, low, L/R wrist | 0.51 GB |
| [aloha_static_coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee) | MIT | 50 | 22 s | high, low, L/R wrist | 1.57 GB |
| [aloha_static_towel](https://huggingface.co/datasets/lerobot/aloha_static_towel) | Apache-2.0 | 50 | 10 s | high, low, L/R wrist | 0.59 GB |
| [aloha_static_battery](https://huggingface.co/datasets/lerobot/aloha_static_battery) | Apache-2.0 | 49 | 12 s | high, low, L/R wrist | 0.79 GB |
| [aloha_mobile_cabinet](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet) | Apache-2.0 | 85 | 30 s | high, L/R wrist | 1.74 GB |
| [aloha_mobile_wipe_wine](https://huggingface.co/datasets/lerobot/aloha_mobile_wipe_wine) | MIT | 50 | 26 s | high, L/R wrist | 1.30 GB |
| [aloha_mobile_shrimp](https://huggingface.co/datasets/lerobot/aloha_mobile_shrimp) | Apache-2.0 | 18 | 75 s | high, L/R wrist | 1.32 GB |

- All episodes within a repo have the same length (fixed-length teleop), and all are ≤120 s.
- **Role map:** `cam_high→main`, `cam_left_wrist→left_wrist`, `cam_right_wrist→right_wrist`. Drop `cam_low` (static only) and record that it was dropped (archive/PLAN-v3.md §9).
- moov check over all episodes of cups_open, cabinet, coffee, shrimp and wipe_wine (≈850 episode-camera pairs): every start is on a sync sample, every episode is CFR (tb 1/12800, delta 256), and x-cam is equal. GOP is 2 frames (LeRobot's encoder default), so nearly any frame-aligned start is a keyframe.
- **Metadata defect found:** in `aloha_mobile_wipe_wine`, episode 43 `cam_high` says `file_index 0, from 1118.0 s`. That file holds exactly 55,900 frames (1118.0 s), so the episode actually starts at file-001 0.0 s (file-001 holds 7 × 26 s = 182 s). The wrist cameras for ep 43 are correct. **The index pass must check each episode's range against the real sample table, not trust `meta/episodes`.**
- Per-camera file rollover differs (13/18 shrimp, 19/85 cabinet and 6/50 wipe_wine episodes have cameras in different `(chunk, file)` slots). The timestamps still match after episode-relative alignment.
- `aloha_static_towel` ships `meta/temporal_proportions_{dense,sparse}.json` (SARM-style progress priors), which is useful for dense-reward sanity checks.

**RoboCOIN Agilex Cobot Magic** (ALOHA-style; `cam_head_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`)
- Apache-2.0, v2.1 (one file per episode per camera), with per-frame subtask annotations.
- [organize_desk_fail](https://huggingface.co/datasets/RoboCOIN/Agilex_Cobot_Magic_organize_desk_fail): 117 **failed** demos, 53–123 s (99.1% ≤120 s), AV1 640×480, 30 fps, 4.5 GB, with a long multi-step instruction.
- [heat_burger](https://huggingface.co/datasets/RoboCOIN/Agilex_Cobot_Magic_heat_burger): 819 episodes, 79–116 s, **H.264**, 141 GB.
- moov check on 3 episodes each: on key, CFR, x-cam equal.
- **Role map:** `cam_head_rgb→main`, then `cam_left_wrist_rgb` and `cam_right_wrist_rgb`.
- Some RoboCOIN repos are gated (auto), e.g. `RoboCOIN/Cobot_Magic_*` and `RoboCOIN/AgiBot-g1_*`.

**Checked and not recommended**
- [nuffnuff/robocoin-yam-v3-269x500](https://huggingface.co/datasets/nuffnuff/robocoin-yam-v3-269x500): Apache-2.0, v3.0, 46,173 episodes, H.264 640×480, 30 fps, `head/wrist_left/wrist_right`, with subtasks. Its `robot_type` is `YAM_ST_bimanual`, but its tasks are RoboCOIN's ("Heat the burger.", ...). The README says only "merged … from parallel conversion shards". **Provenance is unclear**, so prefer the upstream RoboCOIN repos. It does contain explicit handover tasks ("…transfer it from the left gripper to the right gripper"), which are worth tracing to their RoboCOIN source.
- [andlyu/Public-YAM-runs](https://huggingface.co/datasets/andlyu/Public-YAM-runs): real YAM policy runs with retained failures and `outcome` labels, which would be ideal for retries and rewards. But it has **no license**, its cameras are software-synced (with a `camera_skew_s` field), it runs at 10 fps, and only 61% of episodes are ≤120 s.
- [NMotion-Initiative/unitzero-…-yam-lerobot](https://huggingface.co/datasets/NMotion-Initiative/unitzero-sort-household-objects-into-labeled-bins-yam-lerobot): YAM Pro, H.264 720p, 50 fps, on key, CFR, x-cam equal. But only 3.3% of episodes are ≤120 s, and only 15.6% are ≤7200 frames (p50 158 s).

### 3.4 `dex` stand-ins (3-view dexterous hands)

**unitreerobotics/G1_Dex3_ToastedBread_Dataset**
- Apache-2.0, v3.0, 418 episodes, 11.1–75.2 s (100% ≤120 s), 14.9 GB. Cameras `cam_left_high`, `cam_right_high` (head stereo pair), `cam_left_wrist`, `cam_right_wrist`: AV1 640×480, 30 fps. Task "toasted bread".
- moov: 725 episode-cameras on key, CFR, 148 episodes x-cam equal.

**unitreerobotics/G1_WBT_Inspire_Pick_Up_Drinks_v0**
- Apache-2.0, v3.0, 300 episodes, 21.7–90.9 s, 22 GB. Cameras `head_stereo_left/right` and `wrist_left/right`, AV1 640×480, 30 fps. Inspire hands.
- moov: 379 on key, CFR, 70 x-cam equal.

**Role map** for both: head-left→`main`, then the left and right wrist cameras. Drop the right head camera.

There are many more `unitreerobotics/G1_*` repos (Dex1, Dex3, Inspire), all similar.
[Dexora/Dexora_Real-World_Dataset](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset) (MIT, 190 GB, not LeRobot-meta at the root) was not checked.

### 3.5 `tianji`

**DingYFa/tianji_pick_and_place_cubes_20260911**
- Apache-2.0, v3.0, `robot_type: tianji_arm`, 110 episodes, 23.7–59.6 s, 3.5 GB, H.264 640×480, 30 fps. Cameras `head`, `left_wrist`, `right_wrist`, plus four tactile "videos" (400×700).
- moov: 180/180 on key, CFR, 60 episodes x-cam equal (head and wrists).
- **Role map:** `head→main`, `left_wrist`, `right_wrist`. Ignore the tactile streams.
- We have not verified that this is the same rig family as Flora's `tianji` pool.

Also seen:
- [zzzqzzz/ego_2_tianji](https://huggingface.co/datasets/zzzqzzz/ego_2_tianji): no license, one `head_left` 1080p view, Chinese tasks, GOP 250, starts on key. Rejected for the missing license.
- [loopeva/tianji-box-subtask-v4](https://huggingface.co/datasets/loopeva/tianji-box-subtask-v4): CC-BY-NC-4.0. Episodes are pre-split into subtask segments plus 1-frame "hold" samples, and its README says the `fps` field is a nominal 10 while the real rate is 10 or 15 Hz. Rejected, because its timestamps don't reflect real time.

### 3.6 Reward data

**teetone/RoboReward, `test` = RoboRewardBench** (human-verified)
- CC-BY-4.0, 2,831 rows, 1.63 GB. `test/metadata.jsonl` has `file_name`, `task`, `reward` ∈ {1..5} and `gpt5_mini_check`.
- Rubric: 1 no success, 2 minimal, 3 partial, 4 near, 5 perfect. Score counts: 1: 689, 2: 641, 3: 556, 4: 418, 5: 527.
- 1,000 rows come from RoboArena, and the rest from 22 OXE subsets (100 each from droid, bridge and fractal…). 791 rows are counterfactual relabels or temporal clips (`*_attempt_*_score_N.mp4`).
- Spot check on 25 files: all H.264 yuv420p, 10 fps, key at 0. Resolutions vary (256×256 to 640×480). Durations go down to 0.3 s (3 frames).
- **3/25 clipped files are VFR**: a single doubled delta, e.g. 70×1024 + 1×2048 ticks, which is a dropped frame. Their rows can serve as rubric/terminal reward cases only if the Rewarder's envelope accepts VFR. Otherwise prefer the unclipped originals.

**RoboArena/DataDump_07-17-2026**
- MIT. 3,883 sessions, 10,783 policy episodes, 27,148 MP4s, 21.7 GB.
- Each session's `metadata.yaml` has `language_instruction`, a pairwise `preference`, `longform_feedback` (free-text judgement), and per policy `binary_success`, `partial_success` ∈ [0,1] and `duration`.
- Views: `video_left`/`video_right` (exterior, when available) and `video_wrist`.
- Spot check (session `ffeb64c8…`, 4 files): H.264 512×288, 10 fps, CFR, key at 0, 40.1 s. The left and right exterior files have identical packet counts.
- This is the best source of **organic** terminal outcomes with instructions, plus graded partial success.

**ManiRewardBench YAM** (§3.3): timed subtasks give dense-progress targets (fraction of subtasks completed). Every episode is a success.

**RoboCOIN `*_fail` repos** (e.g. organize_desk_fail): failed demos with full instructions and per-frame subtask labels. Useful as terminal negatives with 3 views.

**robometer/rbm-1m-ood and rbm-1m-id-eval** (Apache-2.0): each "episode" is a resampled 32-frame clip at 10 fps (p50 3.2 s, 240×320). They carry `quality_label` and `partial_success`. **Not suitable**, because they are pre-subsampled rather than source video. The same applies to `aliangdw/rbm-1m-ood-lerobot`.

**Rubric-graded data:** nothing public matches Flora's rubric mode exactly. The closest are RoboRewardBench's 1–5 rubric scores and RoboArena's `partial_success` plus long-form feedback.

---

## 4. Proposed golden selection (100 episodes + reward set)

Rules applied to every pick:
- pass every envelope check;
- prefer H.264 where a choice exists, so decord, rubric and NVDEC all work, but keep some AV1 on purpose;
- spread durations across short (<15 s), medium (15–60 s) and long (60–120 s);
- choose episodes deterministically, by sorting candidates on `(dataset, episode_index)` and taking a stride. Record the exact list in `golden/selection.json` (WS-G).

| Pool | Dataset | n | Duration mix | Hard cases on purpose |
|---|---|---|---|---|
| `human` | WGO-Bench HomER (≤120 s rows only) | 10 | 30–120 s | the most-segmented rows (≥20 segments), pouring, laundry folding |
| `human` | EgoDex `test/*` (≥6 different tasks) | 10 | 3–32 s | thread/unthread bead, piano, sweep: fine dexterity and repetition |
| single-view robot | WGO-Bench DROID (deduped by `source_episode_id`) | 8 | 10–107 s | the 4 rows with ≥5 segments |
| single-view robot | WGO-Bench Galaxea | 7 | 4–52 s | bimanual pouring/mixing (two simultaneous objects) |
| single-view robot | jnogga/droid_success (`left_external` only) | 6 | 5–60 s | 720p H.264 contrast with 180p |
| single-view robot | RC-Tianji `tianji`/`tianjihand` main view | 4 | 20–40 s | all-intra stream; has `action_time` GT |
| `yam` | ManiRewardBench YAM | 10 | 10–58 s | build pyramid, set table, group cubes by color (many subtasks) |
| `yam` | ALOHA static (cups_open, coffee, towel, battery) | 4 | 8–22 s | `cam_low` dropped; 50 fps |
| `yam` | ALOHA mobile (cabinet, wipe_wine) | 2 | 26–30 s | camera file rollover inside the set; **exclude wipe_wine ep 43** |
| `yam` | RoboCOIN Agilex organize_desk_fail | 3 | 53–110 s | failed demos (retries, regrasps), long instruction |
| `yam` | RoboCOIN handover task (traced from the nuffnuff "transfer … left gripper to right gripper" tasks) | 2 | — | **handover** |
| `yam` | RoboCOIN Galaxea R1 Lite classify_object_four (head-left as main) | 4 | 19–43 s | 720p AV1, two-handed sorting |
| `yam32` | allenai yam-01 | 5 | 44–88 s | long cloth folding (pauses and regrasps) |
| `yam32` | aloha_mobile_shrimp | 2 | 75 s | long cooking sequence |
| `yam32` | RoboCOIN heat_burger (H.264) | 3 | 80–116 s | near the 120 s cap |
| `dex` stand-in | G1_Dex3_ToastedBread | 5 | 11–75 s | dexterous hand, stereo head (drop right) |
| `dex` stand-in | G1_WBT_Inspire_Pick_Up_Drinks_v0 | 5 | 22–91 s | whole-body motion (moving head camera) |
| `tianji` | DingYFa tianji_pick_and_place_cubes | 6 | 24–60 s | repeated pick-place (cube count varies) |
| **Total** | | **96 + 4 spare** | | |

Notes:
- **Spares:** keep 4 extra from ManiRewardBench and EgoDex for episodes that fail at index time.
- **Codec mix of the 100:** ≈45 H.264 and ≈55 AV1. Each pool has at least one H.264 dataset, except `dex`, which is AV1 only.
- **Retries:** no source labels retries explicitly. The proxies are the RoboCOIN `_fail` demos, the RoboArena failures (in the reward set) and HomER rows with many segments. Choosing true retries means watching candidates, or reading RoboArena `longform_feedback` for words like "tried again" or "regrasp".
- **Pauses:** use the long-window cases (yam32, HomER).
- **Handovers:** RoboCOIN transfer tasks; HomER's *Object Handoff* videos are too long, or are absent from WGO.

**Reward set (≈40 episodes, separate from the 100)**

| Mode | Source | n | Selection |
|---|---|---|---|
| terminal, known outcome | RoboArena DataDump | 12 | 6 `binary_success=1` and 6 `=0`, spread over `partial_success`, wrist + one exterior |
| terminal / rubric | RoboRewardBench test | 15 | 3 per score 1–5, mostly unclipped originals from robo_arena, droid and bridge, plus 2 VFR clipped files tagged as hard cases |
| terminal, 3-view negatives | RoboCOIN organize_desk_fail | 3 | overlaps the `yam` rows |
| dense | ManiRewardBench YAM | 6 | subtask boundaries as progress milestones |
| dense | aloha_static_towel | 2 | uses the shipped temporal-proportion priors |
| rubric | RoboRewardBench (1–5 rubric) | shared with above | the only public rubric-scored set; it is a stand-in for Flora's rubric mode |

---

## 5. Uncovered or weak pools

- **Real `dex` and `tianji` rigs.** Flora's reference camera semantics for these pools aren't public. Unitree G1 (Dex3/Inspire) and DingYFa's `tianji_arm` are structurally valid 3-view stand-ins (on key, CFR, x-cam equal). Certificates for these pools stay "stand-in" until users run `flora golden` on their own data (archive/PLAN-v3.md §9).
- **`yam32`** has no dedicated data. It is the same YAM/ALOHA-style data restricted to long episodes. Real YAM long data (NMotion, Public-YAM-runs) fails the duration cap or the license check.
- **Human egocentric under a permissive license.** None found that is ungated, ≤120 s and has task text. WGO-Bench is NC-SA, EgoDex is NC-ND, HomER upstream has no license, and Egocentric-10K is gated with 180 s clips and no tasks. NC terms are probably fine for internal evaluation, but the golden **outputs** must not be redistributed in the repo.
- **Rubric mode** has no exact public counterpart (see §3.6).
- **Explicit retry labels** don't exist in any license-clean source.

## 6. Risks

- **Licenses.**
  - Several NC or NC-ND sources (WGO-Bench, EgoDex, loopeva).
  - Some have no license at all (toloka/HomER, Public-YAM-runs, ego_2_tianji, TrossenRoboticsCommunity ALOHA), and those are excluded.
  - Stream-copy cuts of NC-ND videos are arguably derivatives, so keep them private to the `flora-golden` Volume.
  - Certificates publish metrics and hashes only.
- **Gating.** Galaxea Open-World, Egocentric-10K and some RoboCOIN repos are gated (auto). Gated sources were avoided in the selection.
- **Keyframe alignment.**
  - Every LeRobot v3 file checked put an episode start on a sync sample: GOP 2 (LeRobot AV1/H.264 default) or a forced key at joins (jnogga).
  - Datasets with long GOPs that pack episodes without forcing keys would fail. None were seen, but the index pass must check every episode.
  - `stss` marks sync samples. For AV1, confirm that these are true `KEY_FRAME`s (not intra-only or S-frames). For H.264, confirm IDR rather than recovery-point I-frames, in L0 (archive/PLAN-v3.md §1.1 "open-GOP per dataset").
- **Metadata errors.** `aloha_mobile_wipe_wine` ep 43 shows that `meta/episodes` can point at the wrong file. Validate every `(file, from, to)` against the real sample table.
- **VFR inside "CFR" sets.** RoboRewardBench clipped files have dropped frames.
- **AV1 decode.** Roughly half the selection is AV1 (ALOHA, EgoDex, Unitree, Galaxea, allenai).
  - H100/A100 have no NVDEC AV1, so decode is CPU (dav1d), or uses L4/L40S/B200.
  - decord probably can't read AV1, which matters for the rubric path (archive/PLAN-v3.md §1.1). Pick rubric episodes from H.264 sources, as RoboRewardBench and RoboArena are.
- **Color tags.**
  - HomER is BT.601 tv range.
  - DROID/RoboReward files carry unspecified range (`0`).
  - Galaxea is tv range BT.709.
  - These decide `canonical_rgb` vs swscale, so keep a mix.
- **Tiny episodes.** Some RoboRewardBench and jnogga episodes are <1 s (3–15 frames). Check Flora's minimum frame or window constraints before including any.
- **Role semantics.** "left"/"right" in YAM datasets and "high" in Unitree data are named, not verified. Confirm visually that each is a wrist or head camera before freezing the selection.

## 7. Next steps

1. WS-G: run the WS-C index pass over the candidate datasets above, over the selected episodes only (moov plus range reads are enough). Emit `EnvelopeVerdict`s and pick the final 100 + 40 deterministically.
2. Look at 2–3 frames per camera to confirm role mappings (YAM `left`/`right`, Unitree `*_high`, tianji `head`).
3. Trace the RoboCOIN source repos for the handover ("transfer … gripper to … gripper") tasks.
4. Mine RoboArena `longform_feedback` for retry and regrasp language, to label hard cases.
5. Run an L0 check that AV1 `stss` samples are true key frames and that H.264 sync samples are IDR, on one file per dataset.
6. Decide whether NC/NC-ND sources are acceptable for the private golden Volume. If not, `human` drops to stand-ins only.
7. If the user has their own YAM, dex or Tianji data, it takes priority ([`archive/P0_EXECUTION.md`](archive/P0_EXECUTION.md), decision 3).
