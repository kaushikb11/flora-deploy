# Output format

This page describes what a finished job writes. The sources are `src/flora_deploy/records.py`
(the record schema `flora_labels_v1`) and `src/flora_deploy/export/compose.py` (JSONL, Parquet
and `summary.json`). `tests/unit/test_output_format_doc.py` checks that every field in the tables
below exists in the code and that every field the code writes is in a table. If you change the
code, update this page in the same change.

All examples on this page are synthetic.

## Directory layout

Every path is relative to `output.sink`:

* `s3://bucket/prefix`: object keys under `prefix/` (Cloudflare R2 and Google Cloud Storage
  work as `s3://` with their S3 endpoint in `AWS_ENDPOINT_URL`; `gs://` and `r2://` sinks are
  refused);
* `volume://name/dir`: files under `/dir` in the Modal Volume `name`;
* `file:///abs/dir`: files under `/abs/dir` (local runs only).

```
flora/<job_id>/
  manifest.json                 # the job's latest planned manifest (a pointer)
  run.json                      # the latest run spec (`flora status` reads it)
  runs/<config_hash>.json       # one run spec per config
  job.json                      # a status cache; status is derived from receipts
  control/  containers/         # cancel/resume requests and container heartbeats
  index/                        # `flora index`: one receipt per episode, plus summary.json
  <config_hash>/
    manifest.json               # this config's manifest (episodes, shards, stage hashes)
    calls/                      # the orchestrator's per-shard call ledgers
    pools/drains.json           # Modal runs: when idle worker pools were scaled down
    output/
      summary.json              # the pointer: written last, names the current generation
      gen/<generation>/         # one complete compose; a re-compose writes a new one
        segment.jsonl           segment.parquet
        caption.jsonl           caption.parquet
        reward.terminal.jsonl   reward.terminal.parquet
        reward.dense.jsonl      reward.dense.parquet
        reward.rubric.jsonl     reward.rubric.parquet
  <stage_hash>/                 # what shards produce, keyed by the hash of their shard kind
    parts/  receipts/  attempts/  errors/  audit/  replans/
flora/deployments/<app>.json    # what `flora deploy` deployed
```

A local run (`flora run --local`) also keeps its own state under `<store>/.flora-local/`.

Only the stages the config lists get files, in the formats `output.formats` lists (default:
Parquet only; add `jsonl` to get the JSONL files too; `lerobot` is not an output format, use
`flora export`, see [`export.md`](export.md)). Read the stage files through `summary.json`: its `files` maps
`<stage>.<format>` to the path under `output/gen/<generation>/` (`flora results` and the SDK do
this). `<generation>` is a digest of the composed contents, so composing the same parts again
reuses the same directory, and a reader never sees a half-written set. Inside `<stage_hash>/`,
files are fanned out by shard id (`parts/<kind>/<xx>/<yy>/<shard_id>/...`).
`<config_hash>` is the job's config hash. `flora results --job <job>` uses the manifest's hash
by default; pass `--config-hash` to pick another one. A config that adds a stage reuses the
finished shards of the stages it shares with an earlier config, so their parts stay under the
same `<stage_hash>` directory.

## JSONL: one record per episode per stage

`<stage>.jsonl` has one canonical JSON object per line (sorted keys, no NaN). Lines are sorted by
`uid`, then by the record's earliest segment start. Each line is a `StageRecord`. Load it with
`StageRecord.from_json`.

<!-- fields: record -->
| Field | Type | Meaning |
|---|---|---|
| `schema` | string | always `flora_labels_v1` |
| `uid` | string | episode key, unique within a job: `<dataset slug>--<episode id>`. Join stages on it |
| `dataset` | string | source dataset (Hugging Face repo id, or the local directory) |
| `episode_id` | string | the episode's id as the dataset names it (not unique across datasets) |
| `stage` | string | `segment`, `caption`, `reward.terminal`, `reward.dense` or `reward.rubric` |
| `provenance` | object | how the record was made (see below) |
| `certified` | bool | true when the record comes from a certified path inside NVIDIA's reference envelope |
| `uncertified_reason` | string or null | why `certified` is false; null when it is true |
| `segments` | list of objects | temporal segments (`segment` and `caption`); empty otherwise |
| `terminal` | object or null | terminal reward (`reward.terminal`) |
| `dense` | object or null | dense reward (`reward.dense`) |
| `rubric` | object or null | rubric reward (`reward.rubric`) |
| `audit_ref` | string or null | where the episode's full NVIDIA audit is: `<stage_hash>/audit/<kind>/#uid=<uid>` (the audit line for this uid is in one of the JSONL files under that directory). Set by `exact` segment/caption; null for reward stages and `balanced` |

`uncertified_reason` is one of:

* `outside_reference_envelope`: the episode fails one of NVIDIA's input checks (length, size,
  frame rate, cameras; see `docs/media-notes.md`, "Envelope"), and `pipeline.envelope: warn` let
  it run anyway;
* `uncertified_stack`: no accuracy certificate covers this stage on this GPU for the job's
  embodiment, and `allow_uncertified: true` let it run anyway (`flora doctor` and `flora plan`
  show each stage's status);
* `uncovered_pool`: no certificate covers the episode's embodiment pool (reserved: not written
  today);
* `oracle_ineligible_source`: the source video cannot be cut so that it decodes to exactly the
  source frames, so the record cannot be compared with the reference;
* `fake_models`: the record was made by fake engines in a local run (`flora run --local
  --fake-models`). It is never a real label, and `flora certify` refuses it, as it refuses any
  record whose `provenance.certificate` is `fake-models` (the `--fake-models` CPU rehearsal app).

### `provenance`

<!-- fields: provenance -->
| Field | Type | Meaning |
|---|---|---|
| `models` | object | pin key to model revision (full commit) used for this record |
| `engine` | string | inference engine label, such as `vllm-0.24.0-offline` |
| `preset` | string | `exact` or `balanced` |
| `config_hash` | string | the stage hash of the record's shard kind (the `<stage_hash>` directory its parts are in), stamped by the worker; also the Parquet `config_hash` column |
| `logic_version` | string | framework logic version |
| `framework` | string | `flora-deploy <version>` |
| `gpu` | string or null | GPU the worker ran on (from `nvidia-smi`); not part of the config hash |
| `certificate` | string or null | id of the accuracy certificate that covers this record ([`certification.md`](certification.md)); null for `exact` (the reference), for stages that are the reference by identity (today `balanced`'s reward stages, which are certified without a certificate, unless a `fail` certificate exists for them) and for uncertified stacks; `fake-models` / `local-fake-models` for fake engines |
| `source` | object | camera role to `{uri, range_s}`: the video file and time range the episode came from |
| `engine_hash` | string or null | the producing stage's own hash (for `balanced`, the engine spec, sampling and pins), kept when the worker stamps `config_hash` with the job's stage hash; null when the stage set none |
| `embodiment` | string or null | the job's `pipeline.embodiment` (the embodiment pool a certificate must cover) |
| `cert_key` | object or null | the certificate key document of the stack that produced the record (`validate.certkey.key_document`: engine and image, vLLM engine spec, NVIDIA and decode pins, derived-weight content digests, run modes, logic); `flora certify` refuses records without it (`docs/certification.md`) |
| `compile_cache` | string or null | `balanced` segment / caption: the vLLM compile cache the worker ran (`warm:<key>:<manifest sha256 prefix>`, `legacy:<key>`, `cold`, or `cold:forced` under `FLORA_COMPILE_CACHE=off`). A warm start replays the publisher's Inductor autotune choices, so it is recorded (`engines/compile_cache.py`, "Determinism"); null for other stages |
| `engine_preemptions` | int or null | `balanced` segment / caption: vLLM preemptions (`vllm:num_preemptions`) while the episode's requests were in flight; > 0 means some request then was preempted and recomputed (a path no certificate measures), null that the counter could not be read. `flora certify` certifies neither |

### `segments[]` (segment and caption stages)

<!-- fields: segment -->
| Field | Type | Meaning |
|---|---|---|
| `event_id` | string | segment id, unique within the episode |
| `start_s` | float | segment start, in seconds from the episode start |
| `end_s` | float | segment end, in seconds |
| `status` | string | `complete`, or `unsupported_target_frames` when NVIDIA could not segment at the target frame count |
| `caption` | string or null | the caption (`caption` stage); null in `segment` records |
| `fine_event_id` | string or null | id of the fine-grained event, when NVIDIA reports one |
| `segmentation_provenance` | string or null | NVIDIA's value, passed through unchanged |
| `degradation` | object | input degradations applied (for example, frame resampling); empty when none |
| `nv_metadata` | object | NVIDIA's segment metadata, passed through unchanged |

### `terminal` (reward.terminal)

<!-- fields: terminal -->
| Field | Type | Meaning |
|---|---|---|
| `score` | int | decoded score, 1 (failure) to 5 (success) |
| `expected_score` | float | expected score over the renormalized top-k score-token distribution |
| `probabilities` | list of 5 floats | probability of each score 1..5 |
| `rationale` | string or null | NVIDIA's rationale, only with `pipeline.terminal_rationale: true` |

### `dense` (reward.dense)

<!-- fields: dense -->
| Field | Type | Meaning |
|---|---|---|
| `segments` | list of objects | NVIDIA's stitched segments, passed through (`start_frame`, `end_frame`, `start_s`, `end_s`, `score`, `label`, `rationale`) |
| `per_step` | list of ints | one value per 0.5 s step: 1 failure, 2 suboptimal, 3 normal |

### `rubric` (reward.rubric)

<!-- fields: rubric -->
| Field | Type | Meaning |
|---|---|---|
| `scores` | list of floats | one score per rubric step (`pipeline.rubric.steps`, in order), from NVIDIA's medoid sample |
| `total` | float | sum of `scores`, rounded to 4 places (as NVIDIA's `rubric_reward.py` writes it) |
| `samples` | int | answers sampled at temperature 1 (NVIDIA's `--num-samples`, 9) |
| `medoid_index` | int | index, among all samples, of the medoid sample the scores come from: the parsed sample with the smallest summed L1 distance to the other parsed samples (ties: the lowest index) |
| `parsed_samples` | int or null | samples whose answer NVIDIA's parser accepted (the medoid is taken over these); an episode with none is an `nv_failure` error |

### Example records (synthetic)

```json
{"audit_ref":null,"certified":false,"dataset":"org/demo","dense":null,"episode_id":"episode_000000","provenance":{"cert_key":{"schema":"flora-cert-key/v2","...":"..."},"certificate":null,"config_hash":"3f2a...","embodiment":"yam","engine":"vllm-0.24.0-offline","engine_hash":"9c1e...","framework":"flora-deploy 0.1.0","gpu":"NVIDIA H100 80GB HBM3","logic_version":"0.1.0","models":{"captioner":"fef0bbfe..."},"preset":"balanced","source":{"main":{"range_s":[0.0,8.0],"uri":"videos/cam_high/file-000.mp4"}}},"rubric":null,"schema":"flora_labels_v1","segments":[{"caption":"The left arm lifts the cup lid.","degradation":{},"end_s":3.5,"event_id":"event_0000","fine_event_id":null,"nv_metadata":{},"segmentation_provenance":null,"start_s":0.0,"status":"complete"}],"stage":"caption","terminal":null,"uid":"org-demo--episode_000000","uncertified_reason":"uncertified_stack"}
```

A `reward.terminal` record has the same envelope, with `"segments":[]` and
`"terminal":{"expected_score":4.6,"probabilities":[0.0,0.0,0.05,0.3,0.65],"rationale":null,"score":5}`.
A `reward.dense` record has
`"dense":{"per_step":[3,3,2,3],"segments":[{"end_frame":100,"end_s":2.0,"label":"normal","rationale":"...","score":3,"start_frame":0,"start_s":0.0}]}`.

## Parquet: one table per stage

`<stage>.parquet` (zstd) holds the same records flattened into columns. Columns marked JSON hold
canonical JSON strings: decode them with `json.loads`. The schema metadata carries
`flora.schema` (`flora_labels_v1`) and `flora.stage`. `export.results.records_from_table` turns a
table back into `StageRecord`s.

Every stage has these columns:

<!-- fields: parquet.common -->
| Column | Type | Meaning |
|---|---|---|
| `uid` | string | as in JSONL |
| `dataset` | string | as in JSONL |
| `episode_id` | string | as in JSONL |
| `stage` | string | as in JSONL |
| `pool` | string | embodiment pool from the manifest (`human`, `yam`, `yam32`, `dex`, `tianji`) |
| `certified` | bool | as in JSONL |
| `uncertified_reason` | string | as in JSONL |
| `config_hash` | string | `provenance.config_hash` |
| `audit_ref` | string | as in JSONL |
| `schema` | string | `flora_labels_v1` |
| `provenance` | string | JSON: the whole `provenance` object |
| `extra` | string | JSON: payload fields the stage's own columns do not cover (for example, segments on a reward record); null when there are none |

`segment` and `caption` have one row per segment, ordered by `uid`, then `start_s`. A record with
no segments is one row with null segment columns.

<!-- fields: parquet.segment -->
| Column | Type | Meaning |
|---|---|---|
| `segment_index` | int32 | position of the segment in the record's `segments` |
| `event_id` | string | `segments[].event_id` |
| `fine_event_id` | string | `segments[].fine_event_id` |
| `start_s` | double | `segments[].start_s` |
| `end_s` | double | `segments[].end_s` |
| `segment_status` | string | `segments[].status` |
| `caption` | string | `segments[].caption` (null in `segment`) |
| `segmentation_provenance` | string | `segments[].segmentation_provenance` |
| `degradation` | string | JSON: `segments[].degradation` |
| `nv_metadata` | string | JSON: `segments[].nv_metadata` |

Reward stages have one row per record:

<!-- fields: parquet.reward.terminal -->
| Column | Type | Meaning |
|---|---|---|
| `terminal_score` | int32 | `terminal.score` |
| `terminal_expected_score` | double | `terminal.expected_score` |
| `terminal_probabilities` | list<item: double> | `terminal.probabilities` |
| `terminal_rationale` | string | `terminal.rationale` |

<!-- fields: parquet.reward.dense -->
| Column | Type | Meaning |
|---|---|---|
| `dense_segments` | string | JSON: `dense.segments` |
| `dense_per_step` | list<item: int32> | `dense.per_step` |

<!-- fields: parquet.reward.rubric -->
| Column | Type | Meaning |
|---|---|---|
| `rubric_scores` | list<item: double> | `rubric.scores` |
| `rubric_total` | double | `rubric.total` |
| `rubric_samples` | int32 | `rubric.samples` |
| `rubric_medoid_index` | int32 | `rubric.medoid_index` |
| `rubric_parsed_samples` | int32 | `rubric.parsed_samples` |

## `summary.json`

<!-- fields: summary -->
| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `flora_compose_summary_v1` |
| `record_schema` | string | `flora_labels_v1` |
| `job_id` | string | the job |
| `config_hash` | string | the config these outputs belong to |
| `status` | string | `complete`, `completed_with_errors` or `partial` |
| `complete` | bool | `status == "complete"` |
| `problems` | list of strings | why the status is not `complete` |
| `max_error_rate` | float | excluded share of a stage above which the job is `completed_with_errors` |
| `stage_hashes` | object | shard kind to stage hash (the `<stage_hash>` directories) |
| `partial` | bool | composed with `--partial` |
| `stages` | object | stage to per-stage counts (below) |
| `records` | int | records over all stages |
| `duplicates_resolved` | int | duplicate records dropped (a retried shard wrote the same record twice) |
| `conflicts` | list of strings | `uid/stage` keys whose duplicates differed (the latest one is kept) |
| `foreign_records_dropped` | int | records from another config hash, ignored |
| `poison_pills` | list of strings | `uid/kind` episodes set aside because they crashed their worker every time they ran |
| `unplanned_episodes` | object | episode to reason, for indexed episodes the planner excluded (for example, no task text); they count as excluded in every stage |
| `generation` | string | digest of this compose's stage files: they live under `output/gen/<generation>/` |
| `formats` | list of strings | the output formats written (`output.formats`) |
| `files` | object | `<stage>.<format>` to the store path of that file in this generation (read files through this pointer) |
| `errors` | object | episode error counts: `by_kind`, `by_stage`, `unreadable_files` |
| `shards` | object | shard state to count |
| `inputs` | object | `parts` (part files read) and `digest` (sha256 over their paths and hashes) |
| `outputs` | object | output path to `{sha256, bytes}` |

<!-- fields: summary.stage -->
| Field | Type | Meaning |
|---|---|---|
| `records` | int | records in the stage files |
| `rows` | int | Parquet rows (segments for `segment` / `caption`) |
| `certified` | int | certified records |
| `uncertified` | int | uncertified records |
| `certified_share` | float or null | `certified / records` |
| `uncertified_reasons` | object | `uncertified_reason` to count |
| `by_pool` | object | pool to `{records, certified}` |
| `expected` | int | episodes planned for this stage |
| `missing` | int | planned episodes with neither a record nor an exclusion (0 unless the compose is `partial`) |
| `excused` | int | excluded episodes (see below) |
| `excused_upstream` | int | of `excused`, episodes an upstream stage already excluded (a `balanced` caption of a segment-excluded episode); counted toward the error rate once, at the upstream stage |
| `unplanned` | int | indexed episodes the planner excluded (`unplanned_episodes`); they count as excluded |
| `episodes` | object | episode state to count |

## Exclusions

An excluded episode has **no record** in its stage's JSONL or Parquet. A worker exclusion is
counted in `summary.json`, under `stages.<stage>.excused`, `poison_pills` and `errors.by_kind`
(an episode the planner left out is under `stages.<stage>.unplanned` instead), and it has a
line in an error file under `flora/<job>/<stage_hash>/errors/`. Episodes are excluded for three
reasons:

* `nv_failure`: NVIDIA's own code rejected the episode (for example, an unparseable model answer
  where NVIDIA has no fallback);
* `poison_pill`: the episode crashed its worker every time it ran, so it was set aside (the shard
  was split until only that episode was left);
* `missing_task`: a reward stage needs the episode's task text, and it has none (empty in the
  dataset, with `episode_instruction.source: dataset_task`). The planner leaves such an episode
  out of the whole job, segment and caption included (`unplanned_episodes`, reason
  `missing_task: ...`), and it counts as excluded in every stage, so a few of them can push the
  job past `max_episode_error_rate`. Set `episode_instruction.source: fixed` with a `text`, or
  drop the reward stages, to label it. A reward worker that still meets one excludes that
  episode only, never its shard.

The planner's reasons in `unplanned_episodes` are written as `<reason>: <detail>`: `index_failed`
(the index pass raised), `source_error` (the source is unusable: missing file, bad range),
`out_of_envelope` (only with `pipeline.envelope: enforce`) and `missing_task`.

`infra` errors (missing media, I/O, CUDA) are diagnostics: the shard is retried and they never
exclude an episode by themselves. If a stage excludes more than `retries.max_episode_error_rate`
of its episodes (default 0.1), the job ends `completed_with_errors` (exit code 3) and `problems` says which
stage.

Error files are JSONL, one object per line:

<!-- fields: error -->
| Field | Type | Meaning |
|---|---|---|
| `uid` | string | episode |
| `stage` | string | stage |
| `shard_id` | string | shard that reported it |
| `container_id` | string | container that reported it |
| `kind` | string | `nv_failure`, `poison_pill`, `missing_task` or `infra` |
| `message` | string | one-line reason |
| `diagnostics` | object | extra detail |

An uncertified record is not an exclusion: it is in the outputs, with `certified: false`. Filter
on `certified` if you want only certified labels.

## Loading the output

Copy a stage locally with `flora results` (or read from the bucket with `s3fs`):

```bash
flora results caption --job my-labels --format parquet --out caption.parquet
flora results reward.terminal --job my-labels --format parquet --out terminal.parquet
```

`--job` defaults to the `flora.yaml` project, and falls back to `<project>.fake` (a fake-model
local run) with a notice when no real run exists; an explicit `--job` never falls back.
`--store` defaults to `output.sink`, then `<config dir>/flora-local-store`.

pandas:

```python
import json
import pandas as pd

caption = pd.read_parquet("caption.parquet")            # one row per segment
terminal = pd.read_parquet("terminal.parquet")          # one row per episode
labels = caption.merge(terminal[["uid", "terminal_score", "terminal_expected_score"]],
                       on="uid", how="left")             # join stages on the episode uid
caption["provenance"] = caption["provenance"].map(json.loads)
```

Hugging Face `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("parquet", data_files={"caption": "caption.parquet",
                                         "terminal": "terminal.parquet"})
# or the nested records: load_dataset("json", data_files="caption.jsonl")
```

From Python with the store directly:

```python
from flora_deploy.export.results import iter_records, load_parquet, store_from_uri

store = store_from_uri("s3://my-bucket/prefix")   # the job's output.sink, not the job directory
records = list(iter_records(store, "my-labels", "caption"))   # StageRecord objects
table = load_parquet(store, "my-labels", "reward.dense")       # pyarrow.Table
```

`segment` and `caption` share segments: the `caption` table already has the segment columns, so
there is no need to join it with `segment`. To join on segments, use `(uid, event_id)`.
