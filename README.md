# flora-deploy

Label robot and human-demo videos with NVIDIA's [Flora](https://research.nvidia.com/labs/gear/flora/)
models on [Modal](https://modal.com): temporal segmentation, captions, and terminal / dense rewards.
Point it at a LeRobot dataset or a folder of MP4s and get JSONL + Parquet labels back.

## Try it locally (no GPU, no Modal account)

```bash
git clone https://github.com/kaushikb11/flora-deploy && cd flora-deploy
uv sync && source .venv/bin/activate        # or: pip install -e .   (Python >= 3.11)

flora init --template local_demo            # writes flora.yaml
flora run --local --fake-models --sample-data
flora results --job local-demo.fake --stage caption --store flora-local-store
```

This runs the real job system with fake models, so the labels are placeholders. Fake-models
jobs run under the job id `<project>.fake`, apart from real jobs. Use it to see the
output format and how jobs behave.

## Run on Modal

You need:
- a Modal account (`modal setup`);
- a Hugging Face token that can read `nvidia/Flora-*`;
- an output location: a bucket (`s3://`, `gs://`, `r2://`, with a Modal Secret `flora-bucket`) or
  a Modal Volume (`volume://name/dir`).

```bash
flora init --template balanced_lerobot       # writes flora.yaml (preset: balanced)
flora doctor                                 # preflight: config, Modal, HF access, store, weights (no spend)
flora prepare                                # download + verify the weights into your Modal Volume,
                                             # build the derived weights, record them in flora.yaml
flora index                                  # scan the dataset
flora plan --index summary.json              # cost / time options
flora deploy
flora run
flora results --job <job> --stage caption --store <store>   # store: s3://... or modal-volume://...
```

The `balanced` preset uses vLLM: TempSeg runs on an L40S, the Captioner on an H100 and the
Rewarder on an H100. Workers scale from zero, so you pay only while a job runs.

## Config

The keys that matter in `flora.yaml`:

```yaml
project: my-labels
license_ack: research-only
preset: balanced
allow_uncertified: true   # required for balanced until its Gate certificates exist

modal:
  budget_usd: 25          # hard cap: the job stops before it would exceed this
  max_gpus: 6             # covers every pool below at its max_containers

input:
  format: lerobot_v3      # lerobot_v21 | lerobot_v3 | mp4_dir
  repo: org/dataset
  revision: <full commit sha>
  cameras:                # optional: map dataset cameras to Flora views
    observation.images.top: main

pipeline:
  stages: [segment, caption, reward.terminal, reward.dense]
  embodiment: human       # human | yam | yam32 | dex | tianji
  episode_instruction: {source: dataset_task}

resources:                # optional per-stage overrides
  segment: {gpu: L40S, max_containers: 2}
  caption: {gpu: H100!, max_containers: 2}

output:
  sink: s3://my-bucket/flora/my-labels
```

Unknown keys are an error. The templates in `src/flora_deploy/config/templates/` document every
key.

## Managing jobs

| Command | |
|---|---|
| `flora status <job>` | progress, errors, spend vs. budget, ETA |
| `flora logs <job> [--follow]` | container logs |
| `flora cancel <job>` | stop the job |
| `flora resume <job> [--budget-usd X]` | continue after a budget stop or crash |
| `flora report <job>` | spend breakdown and output paths |

Exit codes:

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | failed |
| 2 | refused (bad config or arguments) |
| 3 | completed with some episodes excluded |
| 4 | stopped by the budget |
| 5 | cancelled |

Jobs are resumable. A preempted or crashed container loses at most the shard it was working on.
An episode that keeps crashing workers is isolated and excluded rather than failing the job.

## Output

Each stage gets a JSONL file (one nested record per episode) and a Parquet file (flattened; one
row per segment for `segment` / `caption`) under
`<sink>/flora/<job>/<config_hash>/output/gen/<generation>/`, and
`<sink>/flora/<job>/<config_hash>/output/summary.json` points at them (its `files` map; `flora
results` reads through it). Every record carries `uid`, `certified` and `provenance` (engine,
models, GPU, embodiment, certificate key). Join stages on `uid`. Excluded episodes have no record; `summary.json` counts them.
Example records (synthetic, shortened):

```json
{"stage":"segment","uid":"org-demo--episode_000000","certified":false,"uncertified_reason":"uncertified_stack","segments":[{"event_id":"event_0000","start_s":0.0,"end_s":3.5,"caption":null,"status":"complete"}],"provenance":{"preset":"balanced","engine":"vllm-0.24.0-offline"}}
{"stage":"caption","uid":"org-demo--episode_000000","certified":false,"uncertified_reason":"uncertified_stack","segments":[{"event_id":"event_0000","start_s":0.0,"end_s":3.5,"caption":"The left arm lifts the cup lid.","status":"complete"}]}
{"stage":"reward.terminal","uid":"org-demo--episode_000000","terminal":{"score":5,"expected_score":4.6,"probabilities":[0.0,0.0,0.05,0.3,0.65],"rationale":null}}
{"stage":"reward.dense","uid":"org-demo--episode_000000","dense":{"per_step":[3,3,2,3],"segments":[{"start_s":0.0,"end_s":2.0,"score":3,"label":"normal"}]}}
```

```python
import pandas as pd
caption, terminal = pd.read_parquet("caption.parquet"), pd.read_parquet("terminal.parquet")
labels = caption.merge(terminal[["uid", "terminal_score"]], on="uid")
```

Get the files with `flora results --job <job> --stage caption --store <store> --format parquet --out caption.parquet`.
Every field is described in [`docs/output-format.md`](docs/output-format.md).

## Python SDK

```python
from flora_deploy import sdk

job = sdk.submit("flora.yaml")    # derived-weight receipts come from `flora prepare`
job.wait()
records = job.results("caption")
```

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run pytest -q
```

Design notes live in [`docs/`](docs/).
