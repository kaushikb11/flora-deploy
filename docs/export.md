# Exporting labels into a LeRobot dataset

`flora results` gives you one file per stage, keyed by `uid`. If you train on LeRobot datasets,
`flora export` writes the labels **into** a copy of the dataset instead: per-episode fields in
the episode metadata, and per-frame columns in the data parquet. The code is
`src/flora_deploy/export/lerobot.py` (the export) and `src/flora_deploy/export/hub.py` (card,
upload). `tests/unit/test_export_doc.py` checks the field tables below against the code.

```bash
flora export --job my-labels --store s3://bucket/prefix --to lerobot --out ./demo-flora \
    --allow-uncertified                                           # see Safety
flora export --job my-labels --to lerobot --out ./demo-flora --allow-uncertified \
    --stages caption,reward.dense --push-to-hub me/demo-flora      # private repo
```

**`flora export` refuses uncertified records unless you pass `--allow-uncertified`.** Today that
covers every `balanced` job: its `segment` and `caption` records are uncertified until a
certificate exists (its rewards are NVIDIA's own engine and certified). Fake-model records need
`--allow-fake` instead. A refusal exits 2 and writes nothing.

`--store` is the job's `output.sink` (the store root, not the `flora/<job>/` directory). Run in
the directory of `flora.yaml`, `--job` defaults to the config's project (falling back to
`<project>.fake` with a notice when no real run exists) and `--store` to `output.sink`, then
`<config dir>/flora-local-store`.

```python
from flora_deploy import sdk

job = sdk.open_job("my-labels", store="s3://bucket/prefix")
report = job.export("./demo-flora", allow_uncertified=True, push_to_hub="me/demo-flora")
report.unlabeled            # episodes that got no label, with the reason
```

## What gets written

`--out` must be a new (or empty) directory that does not overlap the source. The source is
**never modified**. The export is built in a temporary sibling directory and renamed into place
when it is complete, so a failed export leaves nothing behind.

* **Copied unchanged:** everything else in the source except `videos/` and tool directories
  (`.git`, `.cache`, `.scratch`, `__pycache__`): `meta/tasks.*`, `meta/stats.json` /
  `meta/episodes_stats.jsonl`, and the rest. A symlink that points outside the source dataset
  stops the export (see Safety).
* **Rewritten with labels added:** `meta/info.json`, the episode metadata
  (`meta/episodes.jsonl` in v2.1, `meta/episodes/chunk-*/file-*.parquet` in v3.0) and every data
  parquet file. Source columns, rows, row order and row groups are kept as they are.
* **New:** `README.md` (the dataset card; it replaces the source's card) and
  `flora_export_report.json`.

**Videos are not copied by default.** Paths are unchanged, so the export becomes a complete
dataset as soon as the source revision's `videos/` sits next to it.
`meta/info.json` → `flora.videos.note` gives the exact command, for example
`hf download <repo> --repo-type dataset --revision <rev> --include 'videos/*' --local-dir <out>`.
Pass `--include-videos` to copy them instead.

**The source.** By default, the source is the dataset `flora index` recorded for the job: a local
directory, or the Hub repo at its **pinned revision**. For a Hub repo, only `meta/` and `data/`
are downloaded (plus `videos/` with `--include-videos`), into `.<out>.source/` next to `--out`.
Pass `--source <dir>` to use a local copy.

**Stages.** `--stages` takes `segment`, `caption`, `reward.terminal` and `reward.dense`
(default: every one the job composed). `reward.rubric` is not exported. Subtasks come from
`caption`, which carries the segments with their captions. They come from `segment` (with null
captions) only when `caption` is not exported.

## Per-episode fields

They are added to each episode row: as JSON keys in v2.1 `meta/episodes.jsonl`, and as Parquet
columns in v3.0 `meta/episodes/*`. A field is written only when its stage is exported.
`flora_certified` and `flora_provenance` are always present (null for an episode without a
label).

<!-- fields: episode -->
| Field | Type | Meaning |
|---|---|---|
| `flora_subtasks` | list<struct<start_s: double, end_s: double, caption: string>> | the episode's segments ordered by `start_s`, in seconds from the episode start (`segment` / `caption`) |
| `flora_terminal_score` | int64 | `terminal.score`, 1 (failure) to 5 (success) (`reward.terminal`) |
| `flora_terminal_expected` | double | `terminal.expected_score` (`reward.terminal`) |
| `flora_certified` | bool | true when every record behind the episode's fields is certified (the `segment` record is skipped when `caption` supplies the subtasks) |
| `flora_provenance` | list<struct<...>> | one entry per record behind the episode's fields: `caption` (or `segment` when `caption` is not exported), `reward.terminal`, `reward.dense` (below) |

<!-- fields: episode.provenance -->
| Field | Type | Meaning |
|---|---|---|
| `stage` | string | the record's stage |
| `preset` | string | `provenance.preset` (`exact` or `balanced`) |
| `engine` | string | `provenance.engine` |
| `certified` | bool | the record's `certified` |
| `uncertified_reason` | string | the record's `uncertified_reason` (null when certified) |
| `certificate` | string | `provenance.certificate`: the certificate id, null for `exact` and uncertified stacks |
| `config_hash` | string | `provenance.config_hash` (the stage hash) |
| `framework` | string | `provenance.framework` |

## Per-frame columns

The export adds these columns to every data parquet file. It declares them in
`meta/info.json` `features` the way LeRobot declares scalar features (`shape: [1]`), and, when
the source file has a `huggingface` schema metadata entry (LeRobot writes one), in that entry as
`Value` features. LeRobot and `datasets` load them like
any other feature.

<!-- fields: frame -->
| Column | Type | Meaning |
|---|---|---|
| `flora_dense_reward` | float32 | the dense reward step covering the frame: 1 failure, 2 suboptimal, 3 normal; NaN when none (`reward.dense`) |
| `flora_subtask_index` | int64 | index into the episode's `flora_subtasks` of the subtask the frame is in; -1 when none (`segment` / `caption`) |

### Dense alignment

NVIDIA's dense reward is `dense.per_step`: one value per 0.5 s step of the episode,
`ceil(duration / 0.5)` steps. Each value is read at the middle of its step
(`stages.reward.per_step`). **Step `k` covers `[0.5·k, 0.5·(k+1))` seconds from the episode
start.**

A frame with LeRobot `timestamp` `t` gets the following value. `t` is in seconds from its
episode's start, in both v2.1 and v3.0.

```
k = floor((t + 1e-4) / 0.5)
flora_dense_reward = per_step[k]    if t >= 0 and 0 <= k < len(per_step)
                   = NaN            otherwise, and for every frame of an episode with no record
```

* The 1e-4 s tolerance absorbs float32 timestamp rounding. A frame stored a hair below a step
  boundary (`0.49999997` for 0.5 s) belongs to the step that starts there. Frames are at least
  1/fps apart, so this never moves a frame by more than its rounding.
* **Gaps are not filled.** A frame past the last step is NaN, not the last value held. That
  happens when the dataset holds more frames than the video the Rewarder saw. A frame without a
  dense record is NaN too. `flora_export_report.json` → `dense_uncovered` counts these frames for
  each episode that has a dense record.
* NaN rather than null, because LeRobot turns every frame into tensors, and a tensor cannot hold
  a null. Mask with `torch.isnan`.

### Subtask index

A frame at `t` gets the index of the first entry of `flora_subtasks` with
`start_s <= t + 1e-4 < end_s`. Entries are ordered by `start_s`, so when two segments overlap,
the earlier one wins. A frame in no segment, or in an episode without subtasks, gets -1.

## `meta/info.json`

`features` gains the frame columns. Everything else LeRobot reads (`total_*`, `fps`, `splits`,
paths, `codebase_version`) is unchanged. A `flora` block records the export:

<!-- fields: info.flora -->
| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `flora_lerobot_export_v1` |
| `labels_schema` | string | `flora_labels_v1` |
| `framework` | string | `flora-deploy <version>` that wrote the export |
| `job` | object | `job_id`, `config_hash`, `generation`, `status`, `stage_hashes` of the composed outputs |
| `stages` | list of strings | the exported stages |
| `subtasks_from` | string or null | `caption` or `segment` |
| `source` | object | `path`, `name`, `repo`, `revision`, `subdir`, `format`, `codebase_version` of the source |
| `videos` | object | `included`, and a `note` on how to fetch them when they are not |
| `policy` | object | `allow_uncertified`, `allow_fake`, and `records`: counts of `certified`, `uncertified` and `fake` exported records |
| `episode_fields` | object | episode field to Arrow type |
| `frame_features` | list of strings | the added frame columns |
| `dense_reward` | object or null | `column`, `step_s`, `time_eps_s`, `values`, `rule` |
| `subtask_index` | object or null | `column`, `rule` |
| `unlabeled_episodes` | int | episodes missing at least one exported stage |
| `license` | string | the notice for the derived labels |

LeRobot does not compute stats for the new columns. They are labels, not policy inputs to
normalize.

## Safety

* **Uncertified records are refused** unless you pass `--allow-uncertified`: `balanced` segment
  and caption records, out-of-envelope episodes, uncovered pools. The flag is not read from
  `flora.yaml`'s `allow_uncertified`.
* **Fake-model records are refused** unless you pass `--allow-fake`. These are
  `--fake-models` runs and `*.fake` job ids; they are counted as `fake`, not `uncertified`, and
  `--allow-fake` alone is enough for them. `--allow-uncertified` does not cover them.
* **Paths stay inside the dataset and `--out`.** A source whose `meta/info.json` `data_path`
  template (or episode values) would name a file outside the dataset is refused, and the export
  never writes outside `--out`. An overlapping or non-empty `--out` is refused too.
* A refusal fails the whole export (exit 2) and writes nothing. Records are never silently
  dropped. The flags used are recorded in `flora.policy`, in each episode's
  `flora_certified` / `flora_provenance`, and in a warning on the dataset card.
* An export of an export is refused: the source already has `flora_*` columns.
* The job does not have to be `complete`: a `completed_with_errors` (or `--partial`-composed)
  job exports, and its status is recorded in `flora.job.status` and on the card.

**Episodes without a label** keep their rows, with null episode fields and NaN / -1 frame
values. `flora_export_report.json` lists each one as `{episode_index, episode_id, uid,
missing_stages, reason}` under `unlabeled`; `dense_uncovered` lists `{episode_index,
nan_frames}`. The report also has `schema` (`flora_lerobot_export_report_v1`), `episodes`,
`labeled`, `records`, the `allow_*` flags, `videos_included`, `job`, `source`, `pushed` and
`stages`. `reason` is one of:

* `excluded`: planned, but the stage excluded it (`nv_failure` or a worker's `missing_task`;
  see `summary.json`), or planned with no record when the job has no manifest;
* `poison_pill`: the episode crashed its worker every time it ran, so it was set aside;
* `unplanned: <why>`: the planner left it out, for example `unplanned: missing_task: ...` for an
  episode with no task text in a job with reward stages (all stages are then missing);
* `not_in_job`: the job never included it (for example, `--limit`).

## `--push-to-hub`

`--push-to-hub user/repo` uploads the export with `huggingface_hub`:

* The local export is written first. If the push is then refused, the local directory stays
  and the command exits 2.
* A new repo is **private** by default. Pass `--public` for a public one. `--public` does not
  make an existing private repo public.
* A private push into an existing **public** repo is refused.
* `.cache/**` and `.git/**` are not uploaded.
* After the upload, the commit is tagged with the `codebase_version` (`v3.0` / `v2.1`). LeRobot
  resolves a Hub dataset by that tag.

The dataset card (`README.md`):

* cites the Flora models (`nvidia/Flora-TempSeg`, `nvidia/Flora-Captioner`,
  `nvidia/Flora-Rewarder`) and flora-deploy;
* records the provenance: source revision, job, config hash, stages, record counts and flags;
* carries the license notice for the derived labels. The models are under the **NVIDIA OneWay
  Noncommercial License**, which allows research or evaluation use only. Whether generated labels
  are themselves restricted is unresolved (see `NOTICE.md`). The rest of the dataset keeps the
  source's license.

Videos are not uploaded unless the export includes them.
