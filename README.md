# flora-deploy

**Turn robot and human-demonstration videos into training labels, at dataset scale.**

flora-deploy runs NVIDIA's [Flora](https://research.nvidia.com/labs/gear/flora/) models in your own
[Modal](https://modal.com) account. Point it at a LeRobot dataset or a folder of MP4s and get back,
for every episode:

| Stage | You get | Example |
|---|---|---|
| `segment` | the episode split into subtasks, with timestamps | `0.0–3.5 s`, `3.5–7.2 s`, … |
| `caption` | a description of each subtask | "The left arm lifts the cup lid." |
| `reward.terminal` | did the episode succeed? A 1–5 score with probabilities | `4.6` |
| `reward.dense` | progress score for every step of the episode | `3, 3, 2, 3, …` |

Labels come back as Parquet and JSONL, keyed by episode, ready for pandas or 🤗 `datasets`.

## Try it in two minutes

No GPU or Modal account needed. This runs the full pipeline locally with stand-in models, so you
can see the workflow and the output format before spending anything.

```bash
uv tool install git+https://github.com/kaushikb11/flora-deploy   # or: pip install git+https://...

mkdir flora-demo && cd flora-demo
flora init --template local_demo
flora run --local --fake-models --sample-data
flora results caption
```

## Label your own data

**You need:**
- A [Modal](https://modal.com) account. Run `modal setup` once.
- An output location Modal can write to: a Modal Volume (`volume://name/dir`, nothing to set up)
  or a bucket (`s3://`, `gs://`, `r2://`) with its credentials in a Modal Secret named
  `flora-bucket`.

```bash
flora init                 # writes flora.yaml; set your dataset and output location
flora doctor               # checks your setup without spending anything
flora prepare              # one-time: downloads the models into your Modal account
flora index                # scans the dataset
flora plan                 # shows cost and time before anything runs
flora deploy && flora run  # labels the dataset, streaming progress
flora results caption      # or segment, reward.terminal, reward.dense
```

`flora run` stops before it would exceed your budget, and `flora resume` picks up where a stopped
or interrupted job left off. GPUs start when the job starts and shut down when it ends.

## Configuration

`flora init` writes a commented `flora.yaml`. These are the settings you'll change:

```yaml
project: my-labels
license_ack: research-only        # the Flora models are licensed for research use only
preset: balanced
allow_uncertified: true           # required until balanced's accuracy certificates are published

modal:
  budget_usd: 25                  # hard spending cap for the job
  max_gpus: 6

input:
  format: lerobot_v3              # lerobot_v21 | lerobot_v3 | mp4_dir
  repo: your-org/your-dataset
  revision: <full commit sha>     # pinned, so a rerun labels the same data

pipeline:
  stages: [segment, caption, reward.terminal, reward.dense]
  embodiment: human               # human | yam | yam32 | dex | tianji: closest match to your data
  episode_instruction: {source: dataset_task}

output:
  sink: volume://flora-results/my-labels
```

**Cameras.** Flora reads one camera view, or three (`main`, `left_wrist`, `right_wrist`). If your
dataset has several cameras, map them under `input.cameras`; the template shows how.

## Using the labels

```python
import pandas as pd

captions = pd.read_parquet("caption.parquet")      # flora results caption --format parquet --out caption.parquet
scores = pd.read_parquet("terminal.parquet")       # flora results reward.terminal --format parquet --out terminal.parquet
labels = captions.merge(scores[["uid", "terminal_score"]], on="uid")
```

Training on LeRobot? `flora export --to lerobot --out labeled/` writes the labels into a copy of
your dataset: subtasks and success scores per episode, dense rewards per frame
([`docs/export.md`](docs/export.md)).

Every record has a `uid` (the episode) to join stages on, plus provenance: which models, engine and
GPU produced it. Field reference: [`docs/output-format.md`](docs/output-format.md).

## Cost

`flora plan` estimates the cost of your dataset before anything runs, and `flora report` shows what
a finished job actually spent. Segmentation runs on an L40S, captioning and rewards on H100s.

<!-- wave-h:cost -->
Measured in Wave H-lite (2026-10-22): `balanced` costs about **$5.62 per video-hour** (±50%)
at Modal list prices for segmentation + captioning. Rewards and the cold
start of each container are extra; details in `docs/wave-h-report.md`.
<!-- /wave-h:cost -->

## While a job runs

```bash
flora status          # progress, errors, spend so far, ETA
flora logs --follow   # container logs
flora cancel          # stop the job
flora resume          # continue after a budget stop or an interruption
flora report          # what it cost and where the outputs are
```

Preempted or crashed GPUs lose at most a few minutes of work. An episode that repeatedly crashes is
set aside and reported, so one bad video never sinks a job. Commands and exit codes:
[`docs/cli.md`](docs/cli.md).

## Python

```python
from flora_deploy import sdk

job = sdk.submit("flora.yaml")
job.wait()
captions = job.results("caption")
```

## License

flora-deploy is Apache-2.0. The Flora models are released by NVIDIA for non-commercial research
only and are downloaded into your own account; this repository contains none of their weights,
prompts or code. See [`NOTICE.md`](NOTICE.md).

## Contributing

```bash
uv sync --group dev
uv run pytest -n auto        # ~30 s
```

How it works: [`docs/PLAN.md`](docs/PLAN.md). All docs: [`docs/`](docs/).
