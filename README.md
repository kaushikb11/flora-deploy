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
- Nothing to request on [Hugging Face](https://huggingface.co): the `nvidia/Flora-*` model
  repos are public and not gated, so there are no terms to accept and no token is required
  (`flora prepare` downloads them anonymously). A token (`hf auth login`) is optional and raises
  rate limits. `flora doctor` checks that every model repo and your dataset can be read.
- An output location Modal can write to: a Modal Volume (`volume://name/dir`, the `flora init`
  default; `flora deploy` creates the Volume if it does not exist) or a bucket (`s3://`,
  `gs://`, `r2://`) with its credentials in a Modal Secret named `flora-bucket`.

To start from a working config instead, see [`examples/`](examples/): `lerobot_small` labels a
50-episode public dataset for a few dollars.

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
`balanced` costs about **$5.61 per video-hour** of GPU work (±50%) for segmentation + captioning,
measured 2026-10-22 at Modal list prices. Startup (a few minutes per container), idle time and
rewards are extra and dominate small jobs: `flora plan` shows the all-in estimate for your dataset.
<!-- /wave-h:cost -->

Three numbers, three definitions:
- **GPU work** (above): only the seconds the GPUs spend labelling. A large job's segmentation +
  captioning approaches it.
- **`flora plan`**: the all-in estimate for your dataset: GPU work for every stage you chose,
  plus each container's startup and the orchestrator. Idle time after the work is not included.
- **`flora report`**: what the job actually spent: everything above plus idle time. On a job of
  a few minutes of video, startup and idle can make this several times the GPU-work rate.

`flora index` runs on Modal CPU containers (no GPU, about $0.06 per container-hour), so it costs a
few cents for most datasets; `flora index --local` runs it on your machine for free. `flora
doctor` and `flora plan` cost nothing. The measurement behind these rates:
[`docs/wave-h-report.md`](docs/wave-h-report.md).

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
