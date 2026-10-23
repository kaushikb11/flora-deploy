# What a job costs

This page answers "what will my job cost, and why" for the `balanced` preset on Modal.

<!-- The tables between cost-table markers are generated from src/flora_deploy/planner/rates.json
     and the planner by scripts/cost_table.py; do not edit them by hand.
     Regenerate: uv run python scripts/cost_table.py --write -->

## The short answer

A small job costs more per video-hour than a big one, because every container pays a few minutes
of startup however much video it labels. These are the planner's estimates (the same model
`flora plan` uses) for a dataset of one-minute, single-camera episodes:

<!-- cost-table:short -->
| Video-hours | Episodes | Segment + caption, per video-hour | Job | Wall-clock | + rewards, per video-hour | Job |
|---|---|---|---|---|---|---|
| 0.05 | 3 | **$21.00** ($14.20-$31.20) | $1.05 | 9 min | **$35.80** ($24.20-$53.20) | $1.79 |
| 0.5 | 30 | **$6.86** ($4.60-$10.26) | $3.43 | 32 min | **$9.32** ($6.08-$14.44) | $4.66 |
| 1 | 60 | **$6.07** ($4.06-$9.09) | $6.07 | 57 min | **$7.85** ($5.08-$12.27) | $7.85 |
| 10 | 600 | **$5.37** ($3.58-$8.04) | $53.65 | 8.6 h | **$6.53** ($4.17-$10.33) | $65.25 |
| 100 | 6,000 | **$5.29** ($3.53-$7.94) | $529 | 85.5 h | **$6.39** ($4.08-$10.14) | $639 |

One container per stage, cold starts, Modal list prices. Bold: the estimate; in brackets: its range from the error bands below. Rewards are terminal + dense (`reward.rubric` is not measured yet). The 10 and 100 video-hour rows are projected from rates measured on one small job; not yet confirmed at scale. What each rate rests on:

- `segment`: GPU work 0.342 container-s per video-s, measured (n=1, ±50%); startup 295 s, measured (n=1, ±50%)
- `caption`: GPU work 0.854 container-s per video-s, measured (n=1, ±50%); startup 373 s, measured (n=1, ±50%)
- `reward.terminal`: GPU work 0.055 container-s per video-s, measured (±100%); startup 229 s, measured (±50%)
- `reward.dense`: GPU work 0.165 container-s per video-s, measured (±100%); startup 229 s, measured (±50%)
<!-- /cost-table:short -->

The segment and caption rates come from one small test job and the reward rates from three
sample videos, so treat these numbers as a guide and not a quote: the real cost can land anywhere
in the bracket. Wall-clock assumes the stages overlap; on a small job expect a few minutes more. `flora plan` gives the same kind of
estimate for your own dataset, and `flora report` shows what a finished job actually spent.

### Episodes with several cameras

Each camera view is another video in the prompt, so a three-camera (yam-style) episode costs
more per video-hour than a single-camera one:

<!-- cost-table:cameras -->
| Cameras per episode | 1 video-hour, per video-hour | 100 video-hours, per video-hour |
|---|---|---|
| 1 | **$6.07** ($4.06-$9.09) | **$5.29** ($3.53-$7.94) |
| 3 | **$9.36** ($6.25-$14.01) | **$8.58** ($5.72-$12.86) |

Segment + caption, one container per stage, cold starts. 3-view GPU work is the 1-view rate times `segment` x2.56, `caption` x1.42: an estimate, not measured on 3-view episodes (yam certification prep: scripts/wave_h.py three_view_scale, 2026-10-22). Only the share of the work spent generating text grows with the longer three-view prompt; decoding and frame preparation are kept at 1x. `flora plan` applies the same multipliers and says so.
<!-- /cost-table:cameras -->

### Why you may have seen other numbers

The README quotes the cost of GPU work alone, with no startup or idle time. A very large job
should tend towards it:

<!-- cost-table:busy -->
**$5.28 per video-hour** (segment + caption; $6.37 with rewards), projected from rates measured on one small job; not yet confirmed at scale
<!-- /cost-table:busy -->

`flora plan` gives the all-in estimate for your dataset: the rows above, worked out for your
episodes, stages and pool sizes.

The first test job (run `102207184317`, three minutes of video) cost $33.36 per video-hour all
in. Back then each container also sat idle for five minutes after its work. Pools are now told
to shut down as soon as their work is done, so the same job is priced at the 0.05 row above, and
most of what is left is startup. A later 3.9-minute job cost $27.10 per
video-hour all in, $9.90 without its cold starts.

## Where the money goes

Modal bills each container per second, for as long as it is up: the GPU, plus its CPU cores and
memory. A job pays for four things:

- **Startup.** A fresh container loads the models and builds its GPU kernels before it labels
  anything: about 5 minutes per container today. It costs the same however much video the
  container then labels.
- **GPU work.** The seconds spent labelling. This grows with the video, at the same rate for a
  small job as for a big one.
- **Idle at the end.** Once a stage has no work left, the orchestrator notices within one 15 s
  poll and drops that pool's idle window to 2 s, so its containers stay up about 17 s more and
  then scale to zero. This happens per stage, mid-job too: the segment pool drains while
  captioning still runs. Containers you keep warm with `flora deploy --min-containers` are left
  alone.
- **The orchestrator.** A small CPU container that runs the job. It costs under a cent an hour.

Segment + caption, one container per stage, per video-hour:

<!-- cost-table:breakdown -->
| Video-hours | Startup | GPU work | Idle at the end | Orchestrator | Total per video-hour |
|---|---|---|---|---|---|
| 0.05 | $15.00 (71%) | $5.28 (25%) | $0.74 (4%) | $0.00 (0%) | **$21.00** |
| 1 | $0.75 (12%) | $5.28 (87%) | $0.04 (1%) | $0.00 (0%) | **$6.07** |
| 100 | $0.01 (0%) | $5.28 (100%) | $0.00 (0%) | $0.01 (0%) | **$5.29** |

Idle at the end is an estimate (17 s per container), not yet confirmed on a live job. Everything else comes from the measured rates above; for segment and caption that is one small job, so the large rows are an extrapolation.
<!-- /cost-table:breakdown -->

On a three-minute job, startup is most of the bill. From about one video-hour up, GPU work is
almost all of it. What each container costs while it is up:

<!-- cost-table:prices -->
| Stage | GPU | CPU and memory | Per container-hour |
|---|---|---|---|
| `segment` | L40S | 8 cores, 48 GiB | $2.71 |
| `caption` | H100 | 8 cores, 96 GiB | $5.10 |
| `reward.terminal` | H100 | 8 cores, 80 GiB | $4.97 |
| `reward.dense` | H100 | 8 cores, 80 GiB | $4.97 |
| orchestrator | none (CPU) | 0.125 cores, 0.25 GiB | $0.008 |

Prices: Modal list prices (modal.com/pricing, pinned 2026-10-21). Rates: planner/rates.json, updated 2026-10-22.
<!-- /cost-table:prices -->

## How to spend less

### Label several datasets in one job

Each job pays startup again for every stage, so many small jobs cost much more than one big one.

<!-- cost-table:batching -->
10 separate jobs of 0.05 video-hours cost about **$10.50**; the same 0.5 video-hours as one job cost about **$3.43** (segment + caption, one container per stage).
<!-- /cost-table:batching -->

If you have several small datasets, gather their videos into one input (for example one
`mp4_dir` folder) and label them in one job.

### Choose pool sizes and `max_containers`

Each container you add pays its own startup. In return the job finishes sooner. On a big job the
extra startups barely show; on a small one they can double the bill. `flora plan` spreads a job
over as many GPUs as `modal.max_gpus` allows, so it picks more containers when you raise it:

<!-- cost-table:gpus -->
| Video-hours | 1 container per stage | `max_gpus: 4` | `max_gpus: 10` |
|---|---|---|---|
| 0.5 | $6.86, 32 min (1+1) | $9.06, 15 min (1+3) | $14.42, 10 min (3+7) |
| 10 | $5.37, 8.6 h (1+1) | $5.47, 3.5 h (1+3) | $5.74, 1.3 h (3+7) |
| 100 | $5.29, 85.5 h (1+1) | $5.30, 34.3 h (1+3) | $5.33, 12.3 h (3+7) |

Each cell: cost per video-hour, wall-clock and the containers `flora plan` picks (segment + caption).
<!-- /cost-table:gpus -->

- If time doesn't matter, run one container per stage (`max_containers: 1`, as in
  `flora init --template balanced_lerobot`; `--template balanced_mp4_dir` ships 2 per stage). That is the cheapest way
  to run. `flora plan` recommends a layout that fills `modal.max_gpus`, so set
  `modal.max_gpus: 2` if you want it to pick one container per stage.
- If you have a deadline, pass `flora plan --deadline 12h`. The planner adds containers only
  until the job meets it.
- `flora plan --write N` writes option N's `max_containers` into `flora.yaml`.
- Avoid `flora deploy --min-containers`. It keeps containers warm between jobs, and you pay for
  them while they wait.

### Warm starts

The first container of a stage builds its GPU kernels and then saves them (a compile cache),
once, after its first shard. Later containers of the same stage load the cache, which should take
minutes off each startup. The cache lives on the `flora-cache` Volume, so it is shared across
jobs and redeploys. It is keyed by the image, the vLLM / torch / Triton / CUDA versions, the
engine settings (model, lengths and scheduler limits such as `max_num_seqs`) and the GPU: the
first job after an image or engine change starts cold again and publishes a new cache, and
later jobs start warm.

Separately, the caption pool starts booting while segmentation is still running, so its startup
overlaps the segment work instead of following it. The orchestrator starts it about one caption
startup before the first caption shard can run, and the budget charges it from its start. If the
estimate is early the H100 waits, at about $0.09 per minute; on a small job it saves several
minutes of wall-clock. It is skipped when `flora deploy --min-containers` already keeps the
pool warm.

Batch fill (`engines/batch_policy.py`) is on by default: it feeds the GPU bigger batches, so it
does the same work in fewer seconds. It matters on big jobs, where GPU work is the whole bill.
It was on when the measured rates were taken, but that job (12 episodes) was too small to fill a
batch, so its gain is not in them. Neither warm starts nor full batches have been measured yet,
so those columns below are estimates:

<!-- cost-table:levers -->
| Video-hours | Measured rates (cold) | Warm start | Full batches | Both |
|---|---|---|---|---|
| 0.05 | **$21.00** ($14.20-$31.20) | **$13.40** ($8.80-$20.40) | no gain (too few episodes to fill a batch) | no gain (too few episodes to fill a batch) |
| 1 | **$6.07** ($4.06-$9.09) | **$5.69** ($3.79-$8.55) | **$4.14** ($2.77-$6.19) | **$3.75** ($2.50-$5.65) |
| 100 | **$5.29** ($3.53-$7.94) | **$5.29** ($3.53-$7.93) | **$3.36** ($2.24-$5.04) | **$3.35** ($2.24-$5.03) |

- Measured rates: the rates and cold startups above. Batch fill is on by default and was on when they were measured, but the measuring job was too small to fill a batch.
- Warm start: segment startup 150 s, estimate (placeholder, ±60%); caption startup 180 s, estimate (placeholder, ±60%). Not measured yet.
- Full batches: what batch fill should reach on a job big enough to fill its batches; an estimate, not measured (segment 0.2 container-s per video-s, caption 0.55 container-s per video-s, ±50%, for jobs of at least 16 episodes; the gain can be smaller when preparing frames is the bottleneck rather than the GPU).
<!-- /cost-table:levers -->

`flora plan` prices startups cold unless you ask:

- `flora plan --check-cache` prices a stage warm only when a compile cache is saved for it
  (needs Modal credentials);
- `flora plan --assume-warm` prices every stage warm, as a what-if.

### Read `flora plan`

`flora plan` costs nothing and starts nothing. It prints a few options, one per row:

| Column | What it means |
|---|---|
| cost (range) | The all-in estimate, and its range from the rates' error bands |
| wall-clock | How long the job should take; `<=` is the slow end of the range |
| GPUs | How many GPUs the option uses at once |
| fidelity | `certified`, `partly certified` (some stages have a certificate, or are NVIDIA's own engine, and others do not), `uncertified` or `reference`: what the records will say about accuracy |
| rates | `measured (n=1, ±50%)` means measured, on one job, ±50%; `estimate (...)` means part of the price is not measured; `+ camera scaling (estimate)` means multi-camera episodes were priced with the camera multipliers above |
| status | `ok`, `at risk` (the high end of the range passes your budget or deadline), `infeasible`, `unavailable` or `projection` (needs a feature that is not deployed; shown for comparison and never recommended) |

Lines under the table give each option's pools (for example `segment 1 container × 1 L40S, 1 input
each; caption 3 containers × 1 H100!, 1 input each`) and any warnings. The last line names the recommended option. Big episodes, GPU queueing and
storage are not in the estimate; see [What isn't included](#what-isnt-included).

### Set a budget, then check `flora report`

`modal.budget_usd` in `flora.yaml` is a hard cap for the job:

- before starting each piece of work, the orchestrator projects the job's total spend. If that
  projection passes 95% of the budget, it stops starting new work and the job ends as
  `budget_stopped` (exit code 4);
- if the spend so far plus the work in flight would pass the budget itself, it cancels the work
  in flight;
- `flora resume --budget-usd 50` raises the cap and carries on where the job stopped.

The budget always counts startups as cold, so it errs on the high side. When a job starts it
also prints how to set a Modal Environment budget as a backstop, a little above yours.

`flora report` shows what a job actually spent: the total, a breakdown by stage and by container,
and the cost per video-hour. `flora status` shows the spend so far while the job runs. Both work
from Modal's list prices; Modal's own billing is the final word (`modal billing report`, on
Team and Enterprise plans; otherwise the usage page of the Modal dashboard).

## What isn't included

- **Modal plans and credits.** Every number here is at Modal's list price, before any free
  credits (the Starter plan comes with $30 of credits) and before any plan fee or discount.
- **Storage and transfer.** Modal Volumes cost $0.09 per GiB a month above 1 TiB free, and
  downloads out of Modal cost $0.04 per GiB beyond the transfer your plan includes. Videos, model weights and outputs live on Volumes.
- **Preparing the data.** `flora index` runs on CPU containers and costs a few cents for most
  datasets (or nothing with `--local`); downloading a Hugging Face dataset is not priced.
- **Region pinning.** Setting `modal.region` multiplies every price by 1.15 to 1.75.
  `flora plan` applies 1.15 (a broad region), so a narrow region can cost up to 1.75x its
  estimate; the tables here apply neither.
- **Retries and preemptions.** A preempted or crashed container redoes at most one shard of
  work (about 10 GPU-minutes by default), which these tables don't count.
- **`reward.rubric`.** Its rate is not measured yet, so `flora plan` marks an option that
  uses it as unavailable.
- **Certification runs.** Running the `exact` reference to certify a stack is a separate job on
  H200s.
