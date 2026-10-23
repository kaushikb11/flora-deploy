# Running on Modal

flora-deploy runs its jobs on [Modal](https://modal.com), in your own account. This page covers
what runs where, what it creates and how to clean up. For costs, see [`costs.md`](costs.md).

## What runs where

- **Your machine (or CI)** runs the `flora` CLI. It deploys, starts and watches jobs.
- **One Modal app per config** (`flora-<project>-<hash>`) holds the job:
  - a small CPU **orchestrator** that runs the job. It hands out work, enforces the budget and
    retries failures. It keeps running for as long as the job needs.
  - one **GPU pool per stage**: TempSeg on an L40S (`segment`), the Captioner on an H100
    (`caption`) and the Rewarder on an H100 (`reward.terminal`, `reward.dense`). `exact` runs
    segmentation and captioning on an H200.
- **The output location** (`output.sink`) holds the labels and the job's state. It is a Modal
  Volume (`volume://name/dir`) or an S3 bucket (`s3://bucket/prefix`).

`flora run` only watches the job. Ctrl-C ("stopped watching; the job keeps running") or a closed
laptop leaves the job running. `flora status`, `flora logs` and `flora cancel` work from any
terminal, and `flora run --detach` starts the job and returns.

Pools scale from zero. A stage's containers stop about 17 s after it runs out of work, and
nothing runs between jobs. A preempted or crashed container redoes at most one shard of work
(about 10 GPU-minutes by default). An episode that crashes its worker every time is set aside and
reported.

## How often each step runs

| Step | How often | What it does |
|---|---|---|
| `modal setup` | once per machine | logs the CLI in to Modal |
| `flora prepare` | once per Modal environment | downloads and verifies the models, builds derived weights |
| `flora deploy` | once per config | deploys the config's app; a changed config gets a new app |
| `flora index`, `flora plan`, `flora run` | every job | scan the episodes, price the job, run it |

`flora prepare` downloads about 185 GB of pinned weights (Flora TempSeg with its Qwen3.5-9B base,
the Captioner and the Rewarder) into the `flora-weights` Volume, and checks every hash. It then
builds a BF16 Captioner and a merged TempSeg on CPU containers, which takes about 10 minutes and
costs about $0.12. The Volume ends at about 260 GB, inside Modal's 1 TiB of free Volume storage.
No GPU runs. `flora prepare --dry-run` lists every file and its size. It runs as a temporary app,
so keep the terminal open until it finishes.

## Your data

- **Public Hub dataset:** set `input.repo` and `input.revision` (a full commit). Workers read the
  files they need from the Hub. For a large dataset, set `input.stage_to: volume://<vol>/<dir>`
  so `flora index` copies it to a Volume once and workers read it from there.
- **Local or private data:** upload it once, then point the config at the copy:

  ```bash
  modal volume put flora-results ./videos inputs/my-labels
  ```

  ```yaml
  input:
    format: mp4_dir
    path: ./videos
    stage_to: volume://flora-results/inputs/my-labels
  ```

  (`flora init --template balanced_mp4_dir` writes this layout.)
- **S3 output:** set `output.sink: s3://bucket/prefix`, and put the bucket's credentials in a
  Modal Secret named `flora-bucket`. For R2 or GCS, use their S3 endpoint in `AWS_ENDPOINT_URL`.

## Budget backstop

`modal.budget_usd` is enforced by the orchestrator from Modal's list prices
([`costs.md`](costs.md#set-a-budget-then-check-flora-report)). Also set a Modal Environment budget
a little above it as a backstop; a job prints how when it starts. Modal's own billing is the
final word.

## What it creates in your account

| Object | Name | What for |
|---|---|---|
| App | `flora-<project>-<hash>` | orchestrator and workers; one per config, idle between jobs |
| Volume | `flora-weights` | the models and derived weights (about 260 GB) |
| Volume | `flora-cache` | compiled GPU kernels, so later containers start faster |
| Volume | your `output.sink`, e.g. `flora-results` | labels and job state; created on first use |
| Volume | your `input.stage_to`, if set | a copy of the input videos |
| Dicts | `flora-*` | job leases and pool state |
| Secret | `flora-bucket` (you create it) | S3 credentials, only for an `s3://` sink |

`flora prepare` and `flora index` run as temporary apps and leave nothing deployed.

## Cleaning up

Idle apps and Dicts cost nothing. Volumes are free up to 1 TiB in total.

```bash
modal app list                           # then: modal app stop <app> for each flora-* app
modal volume delete flora-results        # labels and job state: export them first
modal volume delete flora-cache          # the next job starts cold
modal volume delete flora-weights        # the next prepare downloads it all again
modal dict list                          # then: modal dict delete <name> for each flora-* Dict
modal secret delete flora-bucket         # only if you made one
```

Add `--env <name>` to each command if `modal.environment` isn't `main`.
