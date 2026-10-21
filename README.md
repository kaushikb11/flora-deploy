# flora-deploy

Run NVIDIA's [Flora](https://research.nvidia.com/labs/gear/flora/) robotics labeling models at
scale on [Modal](https://modal.com), inside your own Modal workspace. The models cover temporal
segmentation, contextual captioning, and outcome (terminal), dense and rubric rewards.

You point it at a LeRobot dataset (v2.1 / v3) or a directory of MP4s. It indexes the episodes,
estimates the cost, deploys one Modal app per config that scales from zero, and runs a resumable
batch job under a hard budget. The labels come out as JSONL + Parquet (`flora_labels_v1`).

> **Status: pre-release (internal v0.1 milestone in progress).** The job system, CLI and SDK
> are tested end to end on CPU (local mode, fake models). The full `exact` path on Modal GPUs has
> not yet run end to end through `flora run`. The pieces have run on Modal separately. Nothing is
> published to PyPI. See [`docs/PLAN.md`](docs/PLAN.md) and
> [`docs/NEXT_EXECUTION.md`](docs/NEXT_EXECUTION.md).

## License: read this first

The framework code is Apache-2.0. **The Flora models are research-only.** Their weights, prompts
and inference code are under the NVIDIA OneWay Noncommercial License, which allows research or
evaluation use only. This repository contains no NVIDIA weights, prompts or noncommercially
licensed code. They are downloaded into *your* Modal workspace at runtime. A paid or hosted
labeling service built on them would be commercial use. Every config must say
`license_ack: research-only`. Read [`NOTICE.md`](NOTICE.md) before you use it.

## Quickstart: local, 5 minutes, no Modal account

Local mode runs the real job system in one process: index, plan, orchestrator, stores, worker
runners, budget gate, compose and results. NVIDIA's models are replaced by deterministic fakes.
It needs no GPU, no Modal account and no weights. The outputs only demonstrate the pipeline. They
carry the certificate `local-fake-models` and never share a config hash with real outputs.

```bash
git clone <this repo> && cd flora-deploy
pip install -e .                  # Python >= 3.11 (or: uv sync)

flora init --template local_demo  # writes flora.yaml (project: local-demo)
flora run --local --fake-models --sample-data
                                  # --sample-data writes 6 synthetic episodes to ./videos first
flora results --job local-demo --stage caption --store flora-local-store
```

`flora run --local` exits with 0 when the job is `done`, 3 when it is `completed_with_errors`
(some episodes were excluded, see the summary), 1 when it `failed` and 4 when it is
`budget_stopped`. You can inject faults to watch the job survive them, for example
`--inject preempt:<shard id>,oom:ep001@segcap`. The fault kinds are listed in
`flora_deploy/local/faults.py`. Other stages to try: `--stage segment`, `reward.terminal`,
`reward.dense`. Add `--format jsonl` to get records instead of a table.

## The real path: Modal

You need a Modal account (`pip install` brings the `modal` client; run `modal setup`), a Hugging
Face login that can read the `nvidia/Flora-*` repos, and a store the containers and your laptop
can both reach. A bucket (`s3://`, `r2://`, `gs://`, with a Modal Secret named `flora-bucket`
holding its credentials) works everywhere. A Modal Volume (`volume://name/dir`) works too, with
the limits noted under step 6.

```bash
flora init --template exact_lerobot     # edit input.repo / revision / output.sink, budget
flora prepare                           # download and verify the pinned weights in your Volume
uv run modal run src/flora_deploy/modal_app/convert_app.py --step convert
                                        # build the BF16 Captioner the exact preset needs
flora index                             # dataset -> episode index receipts (Modal CPU)
flora plan --index summary.json         # 2-4 options: cost, wall-clock, GPUs, certification
flora deploy --receipt captioner_bf16=<sha256>
flora run --receipt captioner_bf16=<sha256>
flora results --job <project> --stage caption --store <your sink>
```

1. **`flora prepare`** downloads the pinned Flora snapshots into the `flora-weights` Volume and
   verifies them. `--dry-run` lists the files and sizes without downloading anything.
2. **The BF16 Captioner** is a derived copy built inside your Volume and checked against a
   pinned receipt. Its receipt sha256 is the `captioner_bf16` value `flora deploy` and
   `flora run` need. The pinned one is in `config/pins.py`
   (`DERIVED_PINS["captioner-bf16"].receipt_sha256`). See
   [`docs/captioner-bf16.md`](docs/captioner-bf16.md).
3. **`flora index`** writes one receipt per episode to the job store (duration, cameras,
   envelope, eligibility) and a local `summary.json`. A local `input.path` needs `--local`.
4. **`flora plan`** combines the index, the measured rates in `planner/rates.json` and Modal
   prices into a table of options. `--write N` writes option N into `flora.yaml`.
5. **`flora deploy`** builds the config's app (`flora-<project>-<hash>`). The app's workers scale
   from zero. The deployment is recorded in the store. `--dry-run` prints the plan and `--emit
   app.py` writes an equivalent standalone script you can review.
6. **`flora run`** refuses to start if the config is not deployed or hashes differently from the
   deployment. Pass the same `--receipt` (and `--determinism`) you gave `flora deploy`. It plans
   the job from the index, starts the deployed orchestrator, streams status, and composes the
   outputs when the job resolves. `--detach` starts the job and returns. Composing runs on your
   machine. The current `flora results` / `flora compose` read `modal-volume://` stores only
   inside a container that mounts the Volume. With a Volume sink, read the results from Python
   (`Job.results(stage)`, below) or copy the files out with `modal volume get`.

While a job runs (`<job>` defaults to the config's `project`):

| Command | What it does |
|---|---|
| `flora status <job>` | Shows the state, shard progress per kind, stuck calls, error files, spend so far against the budget, the projected total and a rough ETA from the rate table |
| `flora logs <job> [--kind orchestrator\|segcap\|...] [--follow]` | Runs `modal app logs` for the job's calls |
| `flora cancel <job>` | Stops the job for good. The orchestrator cancels its calls on its next tick |
| `flora resume <job> [--budget-usd X]` | Continues a `budget_stopped` job (raising the budget) or restarts a dead orchestrator |
| `flora report <job>` | Shows spend by kind and container, $/video-hour, errors, certification per stage and the output paths |

These commands find the store and the Modal environment from `flora.yaml`, or from `--store` /
`FLORA_STORE` and `--env`. Their exit codes match `flora run`: 0 `done` or still running,
1 `failed`, 3 `completed_with_errors`, 4 `budget_stopped`, 5 `cancelled`, 2 a refusal.

### Job states

A job ends `done` only when every planned episode has a record or a legitimate exclusion.
Legitimate exclusions are NVIDIA's own failure semantics and episodes isolated as poison pills
after 3 crashing starts. The exclusions must also stay within `retries.max_episode_error_rate`
(default 10%) per stage. Otherwise a resolved job is `completed_with_errors`. Its outputs exist
but are not a clean run. A shard whose infrastructure or Python errors exhaust its retries makes
the job `failed`. Preemptions and crashes lose at most the uncommitted tail of one shard, and the
work resumes where it stopped.

### Python SDK

```python
from flora_deploy import sdk

options = sdk.plan("flora.yaml", video_hours=2.0)       # like `flora plan`
job = sdk.submit("flora.yaml", receipts={"captioner_bf16": "<sha256>"})   # `flora run --detach`
view = job.wait(poll_s=30)                              # polls status, composes when resolved
print(view.state, view.spend_usd, view.budget_usd)
records = job.results("caption")                        # list of StageRecord
report = job.report()                                   # `flora report`
job = sdk.open_job("my-project", config="flora.yaml")    # reattach later
job.resume(budget_usd=40)                               # `flora resume --budget-usd 40`
job.cancel()                                            # `flora cancel`
```

Every Modal call goes through an injectable `sdk.Backend`, so code built on the SDK can be
tested without Modal (`tests/fakes/flora_backend.py`).

## Presets and costs

| Preset | What runs | Fidelity | Status |
|---|---|---|---|
| `exact` | NVIDIA's own HF code for TempSeg + Captioner (one `segcap` worker on H200) and NVIDIA's offline vLLM Rewarder (H100!) | The reference: model inputs proven identical to unmodified NVIDIA code | Available |
| `balanced` | vLLM 0.24.0 engines (TempSeg L40S, Captioner H100!) | Must be certified against `exact` | Engine code done, **uncertified**. No Gate certificates exist yet and the app factory does not register the balanced workers yet, so `flora deploy` refuses it. GPU validation has not run. `flora plan` shows it with placeholder rates |

**Measured cost of `exact` today: about $36–40 per video-hour of single-camera video** for
segment + caption on one H200. The figure comes from one short validation run (6 episodes,
81 video-seconds, 4 episodes per container, default determinism mode). `planner/rates.json`
records it as 7.0 container-seconds per video-second, ±35%, at about $5.68/hour for the container
(H200, 8 cores, 96 GiB, Modal list prices pinned on 2026-10-21). Rewards add little: about 0.055
(terminal) and 0.165 (dense) container-seconds per video-second on an H100!, measured on 3
samples only. Rubric rewards have no worker yet. Container startup (several minutes for the
exact worker) and Modal's 5-minute scale-down window are billed on top. `flora plan` includes
them, so small jobs cost much more per video-hour. Treat every number as an estimate. Modal's
billing (`modal billing report`) is the authority.

**Budget.** `modal.budget_usd` is a hard cap. The orchestrator stops spawning when the projected
cost no longer fits. It cancels live calls when spend plus in-flight work would pass the cap.
Spend counts container lifetimes (GPU + CPU + memory). Modal has no API to set an Environment
budget, so the orchestrator prints the manual instruction for one at 1.2× the job budget. Set it
as a backstop.

## Determinism

On Hopper GPUs (H200/H100), NVIDIA's reference is not bit-reproducible run to run. PyTorch's
cuDNN attention kernel occasionally flips a token, and everything downstream follows. The L40S
is deterministic as is. `pipeline.determinism: deterministic` (or `flora deploy --determinism
deterministic`) turns that kernel off at no measurable cost. That is a *different* reference,
not the same bits: it is hashed into the config and recorded in every record's provenance, and
it must never be compared bit for bit with default-mode outputs. `exact` therefore means
*identical model inputs*. Outputs match NVIDIA's own run-to-run variation. Details:
[`docs/determinism.md`](docs/determinism.md).

## More

- [`docs/PLAN.md`](docs/PLAN.md): design, decisions and roadmap
- [`docs/exact-worker.md`](docs/exact-worker.md): the fast `exact` worker, identity proofs and measured rates
- [`docs/balanced-engines.md`](docs/balanced-engines.md): the vLLM engines behind `balanced`
- [`docs/cold-start.md`](docs/cold-start.md), [`docs/captioner-bf16.md`](docs/captioner-bf16.md), [`docs/gate-c.md`](docs/gate-c.md)
- [`NOTICE.md`](NOTICE.md): licensing of the NVIDIA models

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run pytest -q
uv run flora prepare --dry-run   # list the pinned Hugging Face files and their sizes (no downloads)
```
