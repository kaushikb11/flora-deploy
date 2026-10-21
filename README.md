# flora-deploy

Label robot and human-demo videos with NVIDIA's [Flora](https://research.nvidia.com/labs/gear/flora/)
models on [Modal](https://modal.com): temporal segmentation, captions, and terminal / dense rewards.
Point it at a LeRobot dataset or a folder of MP4s and get JSONL + Parquet labels back.

## Try it locally (no GPU, no Modal account)

```bash
git clone https://github.com/kaushikb11/flora-deploy && cd flora-deploy
uv sync                                     # or: pip install -e .   (Python >= 3.11)

flora init --template local_demo            # writes flora.yaml
flora run --local --fake-models --sample-data
flora results --job local-demo --stage caption --store flora-local-store
```

This runs the real job system with fake models, so the labels are placeholders. Use it to see the
output format and how jobs behave.

## Run on Modal

You need:
- a Modal account (`modal setup`);
- a Hugging Face token that can read `nvidia/Flora-*`;
- an output location: a bucket (`s3://`, `gs://`, `r2://`, with a Modal Secret `flora-bucket`) or
  a Modal Volume (`volume://name/dir`).

```bash
flora init --template exact_lerobot          # then set preset: balanced (see config below)
flora prepare                                # download + verify the weights into your Modal Volume

# one-time weight conversions; each prints a receipt hash
uv run modal run src/flora_deploy/modal_app/convert_app.py::tempseg --step merge
uv run modal run src/flora_deploy/modal_app/convert_app.py --step convert

flora index                                  # scan the dataset
flora plan --index summary.json              # cost / time options
flora deploy --receipt tempseg_merged=<sha> --receipt captioner_bf16=<sha>
flora run    --receipt tempseg_merged=<sha> --receipt captioner_bf16=<sha>
flora results --job <job> --stage caption
```

The `balanced` preset uses vLLM: TempSeg runs on an L40S, the Captioner on an H100 and the
Rewarder on an H100. Workers scale from zero, so you pay only while a job runs.

## Config

The keys that matter in `flora.yaml`:

```yaml
project: my-labels
license_ack: research-only
preset: balanced

modal:
  budget_usd: 25          # hard cap: the job stops before it would exceed this
  max_gpus: 3

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

## Python SDK

```python
from flora_deploy import sdk

job = sdk.submit("flora.yaml", receipts={"tempseg_merged": "...", "captioner_bf16": "..."})
job.wait()
records = job.results("caption")
```

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run pytest -q
```

Design notes live in [`docs/`](docs/).
