# Scale test of `balanced` (prepared, not run)

`scripts/scale_test.py` runs `balanced` on about one video-hour, with several containers per pool,
through the real `flora index -> plan -> deploy -> run -> report` path. It is prepared but has
**not been run on Modal**. Only the local fake-models variant has run.

```
uv run python scripts/scale_test.py                                # dry run (default): plan + costs
uv run python scripts/scale_test.py --dry-run --sizing             # + the pool-size trade-off
uv run python scripts/scale_test.py --dry-run --assume-warm        # price the warm-start variant
uv run python scripts/scale_test.py --mode local --execute         # local fake run (no Modal, $0)
uv run python scripts/scale_test.py --mode fake-modal --yes-spend  # CPU rehearsal on Modal (~$0.10)
uv run python scripts/scale_test.py --mode real --yes-spend        # the GPU run (after approval)
uv run python scripts/scale_test.py --mode real --report           # rewrite the report from state
```

The order is: `local` (done), then `fake-modal`, then `real`. Each mode has its own resumable
state at `reports/scale-test/<mode>/state.json`. The report goes next to it as `report.md`.
Both are git-ignored, and the report holds numbers only.

## Goals

The test must validate, at a scale none of the earlier runs reached (≤ 6 episodes, one container
per pool):

1. **Autoscaling.** Do the pools reach `max_containers`, and how does throughput relate to it?
2. **Segment → caption pipelining.** Do caption shards start while segment shards are still
   running?
3. **Budget projection vs actual.** This covers the script's own projection, `flora plan`'s
   option and the orchestrator's live projection.
4. **Orchestrator continuation over a long job.** The orchestrator is killed mid-job and a new
   generation must re-attach from the ledgers.
5. **Preemption recovery at scale.** One `segment` and one `caption` container are stopped
   mid-shard.
6. **Reconciler behaviour.** The reconciler cron must revive the killed orchestrator.
7. **`flora status` / `flora report` on a big job.** How long do they take?
8. **Results compose time.**

## Dataset

The dataset is `griffinlabs/EgoDex-LeRobot-v3.0` at revision
`41d60b449629b2181ff5b735d31c2a2cf8b3cad8`, path `train/wash_put_away_dishes`, format LeRobot
v3.0.

| | |
|---|---|
| Episodes | 128 |
| Frames | 105,877 at 30 fps = **3,529 s = 0.980 video-hours** |
| Episode length | 3.1–78.3 s (median 13.7 s), every one under NVIDIA's 120 s cap |
| Video | one view (`observation.images.camera`), AV1 1920×1080 CFR, in 13 concatenated MP4s (2.4 GB; 3.0 GB with parquet) |
| Task text | 11 tasks (`meta/tasks.parquet`), so `episode_instruction: dataset_task` works |
| License | CC-BY-NC-ND-4.0. Outputs stay on the private Volume and the report holds numbers only. |

Why this one:
- **Embodiment.** It is `human`, the embodiment the GPU session certifies.
- **Same source as the golden episodes.** It is the repo, format and revision of the golden
  `human` episodes and of the `balanced_lerobot.yaml` template, so no new reader path is
  involved.
- **Size.** It is the only EgoDex subset within 3% of one hour. I checked all 240 subsets: the
  `test/*` subsets are at most 0.32 h each, and the next-closest `train/*` subset is 0.76 h.
- **One config.** A config has one embodiment and one input path, so it fits a single config.

**How the pin is checked.** I fetched `meta/info.json` at the full SHA on 2026-10-22. The Hub
returned `X-Repo-Commit` equal to the pin, and the file's git blob SHA-1 was
`b1b624b8542f8768c64b4bfc22d679b60a29a27d`. The episode lengths come from
`meta/episodes/chunk-000/file-000.parquet` (5.5 MB). Step 1 (`verify`) repeats the `info.json`
GET, about 49 kB, before anything is spent. It compares the content-addressed blob and the
counts with `DATASET`, and a 404 or a mismatch stops the runbook. `--dry-run --verify` runs the
same check without spending.

The fake modes use 784 synthetic 3–6 s clips (`local.sample`). Together they are the same 3,529
video-seconds, so the manifest packs them into the same number of shards.

## Job shape

| | |
|---|---|
| Preset | `balanced`, `allow_uncertified: true` (no certificates yet) |
| Stages | `segment`, `caption` |
| `segment` pool | **2 × L40S** (8 cores, 48 GiB; $2.71/h) |
| `caption` pool | **3 × H100!** (8 cores, 96 GiB; $5.10/h) |
| `max_gpus` | 5 |
| `shard_target_gpu_minutes` | 4: about 8 `segment` shards (~480 video-s each) and 15 `caption` shards (~240 video-s each) |

The pools were 3 + 2 when the rates were placeholders. At the measured rates they are 2 + 3 (see
[Pool sizing](#pool-sizing)). With these shards, each segment container runs about 4 shards.
Captions can start after the first segment shard. The local run confirmed 8 + 15 shards.

Two more settings matter:
- **Deploy** uses `--reconcile-cron on`, so the reconciler runs every 10 minutes.
- **Run** uses `flora run --detach`. The script polls by itself: `flora status --json` and
  `modal container list --json` every 30 s (10 s for fake-modal, 1 s for local).

**Chaos actions.** Each fires once (`Chaos`):
- **`segment` preemption:** once a segment shard is done and at least 2 are left.
- **`caption` preemption:** at 25% of the caption shards.
- **Orchestrator kill:** at 50% of the caption shards.

How each mode carries them out:

| Mode | Preemption | Orchestrator kill |
|---|---|---|
| `real` | `modal container stop <id>` (SIGINT; Modal reschedules its inputs) on the newest open attempt marker | `FunctionCall.cancel()` on the last launch; the reconciler cron must revive it within 11 minutes, otherwise the script falls back to `flora resume` and records a FAIL |
| `fake-modal` | the fake workers' fault file (`segment@2`, `caption@2`) | as in `real` |
| `local` | `--inject preempt:segment-*x1,preempt:caption-*x1` | SIGKILL of `flora run --local`, then `flora resume` once the lease reads dead |

A revival is proven by a new spawn or adoption in a shard ledger after the kill. A stale lease
still reads "alive" for its TTL, so that alone proves nothing.

After the job ends, the script times `flora compose`, runs `modal app stop`, then times
`flora status` and `flora report`.

## Cost estimate (dry run, measured rates)

The dry run prices each container class at its own pinned list price (`jobs.budget`): L40S
workers at $2.71/h, H100 workers at $5.10/h, and the orchestrator at the preemptible CPU rate
(0.125 cores, 0.25 GiB). `planner/rates.json` supplies the work rates and startups. The
`balanced` rows are **measured** (the first `balanced` GPU run, 102207184317,
`docs/wave-h-report.md`):

| Kind | GPU | Work (container-s per video-s) | Cold startup | Warm startup |
|---|---|---|---|---|
| `segment` | L40S | **0.367**, measured, n=1, ±50% | **300.5 s**, measured, n=1, ±50% | 150 s, *placeholder*, ±60% |
| `caption` | H100! | **0.906**, measured, n=1, ±50% | **325 s**, measured, n=1, ±50% | 180 s, *placeholder*, ±60% |

The work rates were measured before batch fill (`engines/batch_policy.py`: up to 64 segment and
32 caption episodes inferring per container), so they can only overstate the work. Criterion 8
compares the run against them.

**The warm variant** (`--assume-warm`) prices a run whose compile cache is already published
for each pool's key (`engines/compile_cache.py`). It uses the `compile_cache: warm` startup row
from `rates.json` for each balanced kind. The `balanced` warm rows exist as placeholders (150 s
/ 180 s, ±60%, from the lab's warm engine builds). The cold × 0.72 fallback (the Rewarder's
229 s / 318 s, `docs/cold-start.md`) applies only to a kind with no warm row. In a real run the
cache is published by the first container of each pool after its first shard, so containers
that start before that still start cold. The **caps always come from the cold projection**, so
`--assume-warm` never raises the spend allowed.

**How the projection is built.** The run's cost comes from a shard-level schedule (`simulate`),
not from a fluid estimate. It models the deployed pool behaviour (`jobs/pools.py`):
- all 8 segment shards are queued at t=0;
- a caption shard is queued once the segment shards that cover its episodes have receipts
  (per-shard `depends_on`);
- **caption pre-warm:** one caption container is started a caption startup before the first
  caption shard is runnable, so it is ready at the first segment receipt (the orchestrator
  does this when `min_containers` is 0, and the budget charges it from its start);
- each pool autoscales up to `max_containers`; a container that goes idle while its pool still
  has work scales down 300 s after its last shard (the deployed `scaledown_window`);
- **scale-to-zero at the pool's end:** once a pool has no work left, the orchestrator notices
  within one 15 s poll and drops its window to 2 s, so each container still up idles at most
  17 s more (`POOL_END_TAIL_S`, the same tail `flora plan` charges);
- the app stops 30 s after the last receipt.

This gives each pool's starts, busy time, waits between shards and tails. The chaos costs are
added on top: one replacement start plus a lost shard per preemption, and the reconciler gap
(each caption container idles 300 s while the orchestrator is down, scales down, and restarts).
The **upper** figure runs the same schedule at the upper rates and startups. It counts the full
300 s window plus 240 s of client slack per container, because whether Modal applies the
lowered window to a container that is already idle is not yet confirmed on a live job
(criterion 9 measures it), and the full 660 s reconciler wait.

| Step | Cap | Cold: nominal | Cold: upper | Warm: nominal | Warm: upper |
|---|---|---|---|---|---|
| verify (Hub metadata) | $0 | $0 | $0 | $0 | $0 |
| index (13 CPU units; **placeholder** 120 s / 600 s each) | $0.20 | $0.03 | $0.14 | $0.03 | $0.14 |
| plan | $0 | $0 | $0 | $0 | $0 |
| deploy (image builds; **placeholder**) | $0.10 | $0.00 | $0.01 | $0.00 | $0.01 |
| run | **$19.20** | **$11.46** | **$19.15** | **$9.47** | **$16.70** |
| report | $0 | $0 | $0 | $0 | $0 |
| **total** | global **$19.50** | **$11.49** | **$19.30** | **$9.50** | **$16.84** |
| wall (run) | | 36 min | 57 min | 31 min | 50 min |

The caps are derived, not chosen. Each step's cap is its cold upper projection rounded up to
$0.10, and the global cap is their sum. `scripts/scale_test.py --dry-run` prints the current
numbers, and it prints both variants on every run. The numbers move with the rate table; these
are from 2026-10-22.

The old estimate, from placeholder rates at 3 + 2, was $9.58 nominal and $19.49 upper. The new
nominal is higher for three reasons:
- the measured startups are 300–325 s, not the projected 130 s;
- the work rates are still within the old placeholder band (0.367 against 0.5, 0.906 against
  1.0);
- the third caption container adds one more cold start and one more reconciler-gap restart.

The upper figure is lower than before, because the error bars shrank from ±100% to ±50%.

How the run's nominal cost breaks down (cold, 2 + 3):

| Pool | Item | Cost |
|---|---|---|
| `segment` | 2 cold starts | $0.45 |
| `segment` | work (1,295 container-s) | $0.98 |
| `segment` | 2 drained tails (17 s each) | $0.03 |
| `segment` | preemption | $0.29 |
| `caption` | 3 cold starts (one of them the pre-warm) | $1.38 |
| `caption` | work (3,198 container-s) | $4.53 |
| `caption` | tails (one container runs out of shards about 5 minutes before the last one ends) | $0.53 |
| `caption` | preemption | $0.61 |
| `caption` | reconciler gap (3 × idle 300 s + restart) | $2.65 |
| orchestrator | | $0.01 |

About $3.60 of the $11.46 is chaos: the two preemptions and the orchestrator kill. Without
chaos, the same schedule costs $7.89 and takes 31 minutes.

- **Fake-modal:** $0.10 nominal / $0.34 upper (caps $0.50 global). That is 5 CPU worker
  containers (2 cores, 12 GiB, $0.19/h) whose fake rate and startup are **placeholders**, plus
  the orchestrator.
- **Caps:** each is the upper projection rounded up to $0.10 per step. The script also prints
  the unrounded upper numbers (`honest caps`, global $19.40). They are proposals. Raising them,
  or approving the spend, is the user's decision.
- **Retries:** an attempt that is killed or unreported is charged its whole budget until billing
  is an hour past it. A retry soon after such an attempt is therefore refused by design (see
  `docs/wave-h-runbook.md`).

For comparison, `flora plan` (offline, same rates, cold) picks the same layout:

| Option | Containers | Cost (range) | Wall |
|---|---|---|---|
| `balanced x1` | 2 × L40S + 3 × H100! | $7.44 ($4.99–$11.10) | 0.39 h |
| `exact x4` | 5 × H200 | $36.63 ($26.97–$49.93) | 1.28 h (infeasible) |
| `exact x8 +long_lived_container` | 5 × H200 | $14.89 ($9.25–$24.06) | 0.52 h (at risk; a projection, never recommended) |

`flora plan` charges each container the same 17 s drained tail as the script. It is cheaper
than the script's $11.46 because it does not model client slack or the chaos actions. Its $7.44
is close to the script's $7.89 without chaos. Measuring that gap is part of the point.

## Pool sizing

`scripts/scale_test.py --dry-run --sizing` prints this table for both variants (`pool_sizing`).
It runs the same schedule and chaos model for 1–4 containers per pool, at cold start.

| seg × cap | Nominal | Upper | Wall | Wall (upper) | No chaos | Wall (no chaos) | Utilization seg / cap | Idle share |
|---|---|---|---|---|---|---|---|---|
| 1 × 1 | $8.02 | $13.27 | 66 min | 102 min | $6.22 | 61 min | 0.80 / 0.90 | 0.006 |
| 1 × 2 | $9.61 | $15.84 | 45 min | 71 min | $6.92 | 40 min | 0.80 / 0.79 | 0.040 |
| 1 × 3 | $11.19 | $18.42 | 38 min | 60 min | $7.62 | 33 min | 0.80 / 0.71 | 0.067 |
| 1 × 4 | $12.93 | $21.49 | 36 min | 56 min | $8.48 | 30 min | 0.80 / 0.62 | 0.067 |
| 2 × 1 | $8.26 | $14.02 | 66 min | 102 min | $6.46 | 61 min | 0.67 / 0.90 | 0.008 |
| 2 × 2 | $9.79 | $16.58 | 44 min | 68 min | $7.11 | 38 min | 0.67 / 0.80 | 0.033 |
| **2 × 3** | **$11.46** | **$19.15** | **36 min** | **57 min** | **$7.89** | **31 min** | **0.67 / 0.70** | **0.071** |
| 2 × 4 | $12.90 | $21.72 | 33 min | 52 min | $8.45 | 27 min | 0.67 / 0.65 | 0.077 |
| 3 × 1 | $8.63 | $14.76 | 66 min | 102 min | $6.82 | 61 min | 0.54 / 0.90 | 0.027 |
| 3 × 2 (earlier layout) | $10.15 | $17.33 | 44 min | 68 min | $7.47 | 38 min | 0.54 / 0.80 | 0.049 |
| 3 × 3 | $11.82 | $19.90 | 36 min | 57 min | $8.25 | 31 min | 0.54 / 0.70 | 0.084 |
| 3 × 4 | $13.49 | $22.46 | 33 min | 52 min | $9.04 | 27 min | 0.54 / 0.63 | 0.113 |
| 4 × 1 | $8.74 | $15.51 | 66 min | 102 min | $6.94 | 61 min | 0.51 / 0.90 | 0.011 |
| 4 × 2 | $10.27 | $18.08 | 44 min | 68 min | $7.58 | 38 min | 0.51 / 0.80 | 0.034 |
| 4 × 3 | $11.94 | $20.64 | 36 min | 57 min | $8.37 | 31 min | 0.51 / 0.70 | 0.070 |
| 4 × 4 | $13.61 | $23.21 | 33 min | 52 min | $9.16 | 27 min | 0.51 / 0.63 | 0.099 |

"Nominal", "Upper" and "Wall" are for the run step with chaos. "No chaos" is the bare schedule.
Utilization is busy ÷ whole container life, including the cold start and the tail. Idle share
is the tail cost ÷ pool cost, without chaos. With a warm start (placeholder 150 s / 180 s),
every row is $0.84–$2.62 cheaper and 2.5–5 minutes shorter, and the recommendation does not
change (2 × 3 warm: $9.47 / $16.70, 31 / 50 min).

**What the table shows:**
- **`caption` is the bottleneck.** It needs 0.906 container-s per video-s, against 0.367 for
  segment, which is 2.5× as much on a container that costs 1.9× as much. The wall clock is set
  by the caption pool size: 66, 44, 36 and 33 minutes for 1, 2, 3 and 4 caption containers
  (a single segment container adds 1–3 minutes).
- **Segment containers beyond 1 buy almost no wall time.** Each extra segment container adds a
  300 s cold start (about $0.23) and a 17 s drained tail. Even one segment container stays
  ahead of 3 caption containers. The earlier 3 × 2 layout is dominated by 2 × 2: it has the
  same 44 minutes and costs $0.36 more.
- **Startup amortization.** Each container pays 300–325 s once. Against 1,295 s of segment work
  and 3,198 s of caption work, cold starts are about a third of segment's cost and a fifth of
  caption's at 2 × 3 without chaos. That is why whole-life utilization of segment is only 0.67,
  even though segment never waits between shards.
- **Pipelining.** With the caption pre-warm, the first caption starts at the first segment
  receipt in every row (0 s). Without it, it would start one caption cold start (325 s) later.
  Segment is ahead, so the caption pool never waits on segment work ("idle between shards" is
  0 s). The overlap is the whole segment window.
- **Tails are small.** The drain caps the tail after a pool's last shard at 17 s per container.
  The idle share that remains is caption containers that run out of shards a few minutes
  before the pool's last one ends (they keep the deployed 300 s window while the pool has work).
- **Chaos scales with the caption pool.** The reconciler gap costs 300 s of idle plus one
  restart per caption container, about $0.88 each. That is why 2 × 3 costs $1.67 more than
  2 × 2 with chaos, but only $0.78 more without it.

**Recommendation: 2 × L40S + 3 × H100!** (`recommend`). This is the cheapest size that:
- has at least 2 containers per pool, because the test is about autoscaling several
  containers;
- finishes within 60 minutes even at the upper rates (57 minutes, against 68 for 2 × 2). Both
  rates are n=1 at ±50%, so the upper wall is the honest one;
- reaches every pool's size in the schedule, so criterion 3 can pass.

It costs $1.67 (17%) more than the cheapest multi-container option (2 × 2), and cuts the
nominal wall from 44 to 36 minutes. `flora plan` independently picks the same 2 + 3. The
cheapest size within the hour, 1 × 3 at $11.19, was rejected because a one-container segment
pool cannot test segment autoscaling.

## Success criteria

The report evaluates all 14 criteria **automatically** (`evaluate`), from what `run` and
`report` record. None of them is manual (`MANUAL` is empty). The report prints one row per
criterion with its value, target, PASS / FAIL / n.a. and a reason, plus a summary line
(`criteria_summary`).

The verdict rules:
- **A value the run did not record is a FAIL ("not recorded"), not a pass.** For example, a
  chaos action that never fired fails criterion 10 or 11.
- **`n.a.` is only for `local`, and it always carries a reason.** Locally there are no
  containers, cold starts or money, so criteria 3, 4, 5, 7, 8 and 9 do not apply.
- **`--report` re-checks criteria 7 and 8** against the run step's spend as it stands, which is
  the billed delta once Modal billing lands (about an hour after the run). Until then, the
  tracked spend is used and the reason says "provisional".

Targets marked *derived* come from the cold **upper** schedule of the configured pools, so they
move with `rates.json` (`criteria_targets`). The values shown are for the current rates at
2 + 3.

| # | Criterion | Target | Recorded from | Decision it informs |
|---|---|---|---|---|
| 1 | End state | `done` | the final `flora status` state | the job system survives scale plus chaos; it gates the first public release |
| 2 | Lost or duplicate records | 0 (compose `missing` + conflicts) | `job.summary()` | correctness of retry, dedupe and compose at 8 + 15 shards |
| 3 | Each pool reaches `max_containers` | segment 2, caption 3 | peak concurrency from `modal container list` samples | whether the factory's `max_containers` is honoured, i.e. whether the planner's container counts can be trusted |
| 4 | Time to peak concurrency, from the run start | *derived*: segment ≤ 150 s, caption ≤ 844 s (upper schedule + a poll + 120 s launch margin); a pool that never reaches its size fails | container samples and the run start | whether `min_containers` / warm pools are needed for short jobs |
| 5 | Utilization while holding work, per pool | ≥ 0.60 busy ÷ container-seconds, from each container's first shard start to its last shard end | attempt markers and receipts | the release goal of ">60% GPU utilization while busy"; the pool ratio and shard size |
| 6 | Pipelining: first caption start after the first segment receipt | *derived*: ≤ 608 s (upper caption cold start + 120 s), and overlap > 0 | attempt markers and receipts | whether `depends_on` per shard (not per stage) works; the wall-time model in `flora plan` |
| 7 | Spend within the upper projection | actual ≤ upper | tracked spend, or the billed delta once it lands | whether the stop rules and caps are safe at scale |
| 8 | Spend vs the nominal projection | \|actual ÷ nominal − 1\| ≤ 0.5 | as 7 | whether the measured `balanced` rows hold at n > 1 (they were n=1, and measured before batch fill) |
| 9 | Idle / scale-down share of pool cost | ≤ 0.15, leaving out tails that began while the killed orchestrator was down | container samples, receipts, and the kill and revive times | whether the job-aware drain (`jobs/pools.py`) actually ends the tail on a live job |
| 10 | Preemption recovered | each pool has a `preempted` start, and its shard still ends with a receipt from a later attempt | attempt markers and receipts | spot instances are safe for `balanced` |
| 11 | Orchestrator revived by the reconciler | kill fired, revived by the reconciler (not the `flora resume` fallback) within 660 s, and the job ends `done` | the first spawn or adoption in a ledger after the kill | whether the reconciler cron is mandatory in `flora deploy`; continuation semantics |
| 12 | `flora status` p95 | ≤ 10 s | the run's status polls | whether status needs a cached `job.json` instead of a full reconcile |
| 13 | `flora report` | ≤ 30 s | timed in `report` | as above |
| 14 | `flora compose` | ≤ 60 s, exit 0 | timed in `run` | whether compose must move to Modal, or be incremental, for 10+ hour jobs |

**Changes from the earlier draft, and why:**
- **Criterion 4** was "≤ 450 s for both pools" (2 × the old 195 s upper startup + a poll). The
  caption pool cannot be at its size before the first segment receipt, so each pool now has
  its own target from the schedule.
- **Criterion 5** was busy ÷ whole container life. At the measured 300 s cold starts (and the
  300 s scale-down window the schedule modelled before the job-aware drain), that failed by
  construction: segment reached 0.52 even with no waiting at all (0.67 with the drain). The
  cold start is judged by criterion 4 and the tail by criterion 9, so criterion 5 now judges
  the time in between, which is what the release goal's "while busy" means. The whole-life
  figure is still reported (`utilization`).
- **Criterion 6** was ≤ 315 s, from the old 195 s startup. It is now derived from the measured
  caption cold start. With the caption pre-warm the expected gap is near 0 s, so the target is
  a loose upper bound (it still passes if the pre-warm does not fire).
- **Criterion 9** left the orchestrator kill's induced idle in the share. Those 3 × 300 s of
  caption idle ($1.27) would have pushed it over 0.15 by construction. Tails that begin inside
  the outage window are now left out (`idle_share_excl_outage`); the raw share is still
  reported.
- **Criterion 8** no longer decides the placeholder → measured flip, since that has happened.
  It now checks the n=1 measurements at n > 1.

**What each metric feeds.**

| Metric | What it feeds |
|---|---|
| Video-s per container-s, per pool | the `rates.json` value at n > 1 (it narrows the ±50%) |
| Cold-start distribution and spawn-to-start latency | the `startup` rows (cold now; warm once a compile cache is published) |
| Idle seconds and idle cost | the drained-tail model in `jobs.budget` and `flora plan` |
| `pool_drained` / `pool_prewarm_*` events | whether the drain and the caption pre-warm work on a live job |
| Projection error (script, `flora plan`, the orchestrator's first projection) | which estimator to trust |

## What a run proves

If all 14 criteria pass on `--mode real`, the run shows the following, at 2 + 3 containers,
128 episodes and about one video-hour:
- **The job system completes a multi-container job through chaos.** That covers 8 + 15 shards,
  two mid-shard preemptions and an orchestrator kill, with no lost or duplicate records
  (criteria 1, 2, 10 and 11).
- **Autoscaling and per-shard pipelining work as the planner assumes.** Both pools reach
  `max_containers` in the time the schedule predicts, and captions start while segment is still
  running (criteria 3, 4 and 6).
- **The cost model holds at n > 1.** The measured rates and per-class prices predict spend
  within ±50%, and the caps hold (criteria 7 and 8). The run also replaces the n=1 rate rows
  with many-shard measurements.
- **The idle policy and the client commands are fit for this size** (criteria 5, 9, 12, 13 and
  14).

What it does not prove:
- the 23 h continuation hand-off (see the limits below);
- warm-start startups, unless a compile cache was published for each pool's key before the run;
- behaviour at 10+ video-hours, beyond the trend in criteria 12–14;
- anything about output quality. This test checks the job system, not the captions.

## Known limits of the test

- **The 23 h continuation deadline is not reached.** A one-hour job never hits
  `orchestrator.DEADLINE_S` (23 h), and `modal_app` has no knob to shorten it. The test instead
  kills a generation mid-job. The successor's re-attach is the same code path (a fresh
  `Orchestrator` rebuilding its view from the ledgers), but the `continuation_spawned` hand-off
  itself is covered only by unit tests. A `FLORA_ORCH_DEADLINE_S` override in
  `modal_app/orchestrator_app.py` (not owned here) would let the test cover it.
- **Container lifetimes are sampled.** Workers write no `ContainerReceipt`s, so lifetimes come
  from `modal container list` samples, at the poll resolution of 30 s. Locally, a container's
  life is its first start to its last receipt. Busy time comes from attempt markers and
  receipts.
- **Local spend is not money.** The local runner prices fake call seconds at GPU list prices, so
  its "spend" exercises the budget gate only.
- **Local revival.** A local `flora resume` is a no-op until the killed orchestrator's lease
  expires (30 s TTL). The script resumes once status reads the orchestrator as dead.

## Local rehearsal result

I ran `--mode local --execute` on 2026-10-22, with 784 synthetic episodes (3,527 video-s), at
the earlier 3 + 2 pools. All six steps completed in about 2 minutes, and the job ended `done`. The full numbers are in
`reports/scale-test/local/report.md` (git-ignored, so not in a fresh checkout).

**Shards and records**
- 8 `segment` and 15 `caption` shards.
- 784 of 784 records per stage, with 0 missing, 0 excused and 0 conflicts.

**Chaos**
- **Preemptions:** 1 `segment` and 1 `caption` preempted start. Both shards still finished.
- **Orchestrator kill:** the orchestrator was SIGKILLed at caption 8/15, mid-job. `flora resume`
  revived it after 31.9 s, which is the lease TTL. The new generation re-spawned the in-flight
  shards, which had died with the process.

**Pipelining**
- The first caption shard started 6.8 s after the first segment receipt, and the two pools
  overlapped.
- Peak concurrency was 3 segment and 2 caption containers.

**Client commands**

| Command | Time |
|---|---|
| `flora status` (51 polls) | median 0.4 s, p95 0.9 s |
| `flora report` | 0.7 s |
| `flora compose` | 1.5 s |

**What the local run cannot show.** Autoscaling time, utilization and idle cost are `n.a.`
locally: there are no cold starts and no scale-down, and calls run on threads. Those criteria
can only be judged on Modal.

**Re-run at 2 + 3 with the automatic criteria (2026-10-22).** The job ended `done`, with
8 + 15 shards and 784 of 784 records per stage. Peak concurrency was 2 segment and 3 caption
containers. Both preemptions were recovered, `flora resume` revived the orchestrator after
31.8 s, and status p95 was 0.4 s. The report read **8 PASS, 0 FAIL, 6 n.a. (14 of 14
automatic)**. The six n.a. are criteria 3, 4, 5, 7, 8 and 9, which need containers or money.

**A bug the rehearsal found and fixed.** The first attempt hung because `flora resume`, issued
right after the kill, saw the unexpired lease ("orchestrator still running") and did nothing.
The script now resumes only once the orchestrator reads dead.
