# Scale test of `balanced` (prepared, not run)

`scripts/scale_test.py` runs `balanced` with several containers per pool, through the real
`flora index -> plan -> deploy -> run -> report` path. It has two sizes (profiles):

| Profile | Episodes | Video | Containers | Chaos | Cost, nominal / upper | Wall, nominal / upper |
|---|---|---|---|---|---|---|
| **`lite`** (the default) | the first 32 | 802 s = 0.22 h | 2 × L40S + 2 × H100! | none | **$2.77 / $6.32** | 15 / 23 min |
| `full` | all 128 | 3,529 s = 0.98 h | 2 × L40S + 3 × H100! | 2 preemptions, 1 orchestrator kill | $11.49 / $19.30 | 36 / 57 min |

Both are prepared but **have not been run on Modal**. Only the local fake-models variant has
run. `lite` is the default because only about $10–15 of the project budget (default $50) is
left after the certified run, and the full test's upper projection ($19.30) no longer fits.
The project budget is one setting, `FLORA_PROJECT_BUDGET_USD` (default `50`), read by
`scripts/spend.py` (where `--budget` overrides it), the `cert_preflight.py` budget check and
the `wave_h.py --global-cap` ceiling.

```
uv run python scripts/scale_test.py                                # dry run (default): lite plan + costs
uv run python scripts/scale_test.py --sizing                       # + the lite size options
uv run python scripts/scale_test.py --profile full --sizing        # the full test and its pool sizing
uv run python scripts/scale_test.py --assume-warm                  # price the warm-start variant
uv run python scripts/scale_test.py --mode local --execute         # local fake run (no Modal, $0)
uv run python scripts/scale_test.py --mode fake-modal --yes-spend  # CPU rehearsal on Modal (~cents)
uv run python scripts/scale_test.py --mode real --yes-spend        # the GPU run (after approval)
uv run python scripts/scale_test.py --mode real --report           # rewrite the report from state
uv run python scripts/scale_test.py --finalize                     # what folding its rates would change
uv run python scripts/scale_test.py --finalize --write             # fold them into planner/rates.json
```

Every command takes `--profile lite` (the default) or `--profile full`. The order is: `local`,
then `fake-modal`, then `real`. Each profile and mode has its own resumable state:
`reports/scale-test/lite/<mode>/state.json` for lite and `reports/scale-test/<mode>/state.json`
for full (where it always was). The report goes next to it as `report.md`. Both are
git-ignored, and the report holds numbers only.

## The lite profile

### What it must prove

On real GPUs, with two containers in each pool:

| What | How it is shown | Criteria |
|---|---|---|
| **Throughput per pool with several containers** | container-seconds per video-second for each pool, and for each container in it, from at least 2 containers per pool. Each container is one rate sample, which narrows the ±50% band of today's single-job rates. | 15, 3, 5 |
| **Segment → caption pipelining** | the first caption shard starts soon after the first segment receipt, and the two pools overlap | 6 |
| **Scale to zero at the end of the job** | each worker container's end, sampled every 5 s after the job ends, against the moment the orchestrator drained its pool | 16, 9 |
| **Caption pre-warm** | the pre-warm was planned and started, and the first caption container started near the planned time | 17 |
| **Compile-cache use** | every shard records which compile cache it ran with (`cold` or `warm`). A pool whose cache was published before the run must load it on every shard. | 18 |
| **Batch fill** | mean batch size and GPU-busy fraction per pool, from the shard audit rows, against the engines' request caps (TempSeg 48 on the L40S, Captioner 16 on the H100) | 19 |
| **Budget projection vs actual** | the spend against the script's nominal and upper projection | 7, 8 |

It also covers what holds at any size: the job ends `done` with no lost or duplicate record,
and `flora status`, `flora report` and `flora compose` stay fast (criteria 1, 2, 12–14).

What lite does **not** prove: preemption recovery and the orchestrator's revival by the
reconciler (criteria 10 and 11 are `n.a.`; see [Success criteria](#success-criteria)), and
anything about one video-hour or more. Those stay with the full profile and the rehearsals.

### The episodes

The dataset is the same pinned one as the full test (see [Dataset](#dataset)). Lite takes
**the first 32 episodes by episode index**. Step 2 runs `flora index --limit 32`, and the
LeRobot reader lists episodes in episode-index order, so the subset is the same on every run.

| | |
|---|---|
| Episodes | 0–31 (24,050 frames at 30 fps = **802 s = 0.223 video-hours**) |
| Episode length | 6.2–54.8 s, median 11.4 s: 18 under 15 s and 14 of 30 s or more |
| Source files | the first 3 of the 13 concatenated MP4s (the third only in part) |

Why the first 32, not a hash order:
- **Deterministic without new code.** `--limit` already exists; a hash order would need a
  selection option in `flora index`.
- **Realistic spread for batching.** The task alternates between long "wash dishes" episodes
  (31–55 s here) and short "put them away" episodes (6–12 s), so any run of consecutive episodes
  mixes both, in about the full dataset's proportion (65 of 128 under 15 s).
- **Fewer downloads.** Three MP4s are read instead of thirteen.

**How the subset is pinned.** The script carries every episode's frame count and source file
(`EPISODE_FRAMES`, `EPISODE_FILE`), read from `meta/episodes/chunk-000/file-000.parquet` at the
pinned revision (git blob SHA-1 `d58e8fbd29082d252097ac14416e4ef3bdc95b22`, fetched
2026-10-22). They sum to the dataset's 105,877 frames. From them, the dry run prices the exact
shards the manifest will pack. After indexing, step 2 checks that exactly 32 episodes were
indexed and that their length is within 1% of the 802 s the table gives; otherwise it stops
before anything else is spent.

### Job shape

| | |
|---|---|
| `segment` pool | **2 × L40S** |
| `caption` pool | **2 × H100!** |
| `max_gpus` | 4 |
| `shard_target_gpu_minutes` | 2, so **6 segment shards** (240, 56, 196, 63, 193 and 53 video-s; shards follow the source files) and **9 caption shards** (53–115 video-s) |
| Chaos | none |

With 6 and 9 shards, each container runs at least 3 shards. Each container therefore yields a
rate sample, captions start while segment shards are still running, and both pools reach 2
containers in the schedule. The full test's 4-minute shards would give only 3 segment shards
here, and the second segment container would sit idle.

**Why no chaos.** A preemption's lost half-shard is counted as busy time, which skews the
per-container rate samples that lite exists to take. The two replacement starts cost about
$0.75. An orchestrator kill costs more: the reconciler runs every 10 minutes, which is about as
long as lite's whole caption phase, so a kill would double the wall clock, and the 2 × (300 s
idle + restart) on H100s (about $1.80) would swamp the throughput and idle measurements.

### Cost (dry run, today's measured rates)

The same model as the full test (see [How the projection is built](#how-the-projection-is-built)),
on the exact 6 + 9 shards:

| Step | Cap | Cold: nominal | Cold: upper | Warm: nominal | Warm: upper |
|---|---|---|---|---|---|
| verify (Hub metadata) | $0 | $0 | $0 | $0 | $0 |
| index (3 CPU units; *placeholder* 120 s / 600 s each) | $0.10 | $0.01 | $0.03 | $0.01 | $0.03 |
| plan | $0 | $0 | $0 | $0 | $0 |
| deploy (image builds; *placeholder*) | $0.10 | $0.00 | $0.01 | $0.00 | $0.01 |
| run | **$6.30** | **$2.76** | **$6.28** | $2.12 | $5.40 |
| report | $0 | $0 | $0 | $0 | $0 |
| **total** | global **$6.50** | **$2.77** | **$6.32** | $2.13 | $5.44 |
| wall (run) | | 15 min | 23 min | 12 min | 18 min |

How the run's nominal $2.76 breaks down:

| Pool | Item | Cost |
|---|---|---|
| `segment` | 2 cold starts (300 s each) | $0.45 |
| `segment` | work (294 container-s) | $0.22 |
| `segment` | 2 drained tails | $0.05 |
| `caption` | 2 cold starts (325 s each; one is the pre-warm) | $0.92 |
| `caption` | work (726 container-s) | $1.03 |
| `caption` | 2 drained tails | $0.09 |
| orchestrator | | $0.00 |

At this size **cold starts are half the cost** ($1.37 of $2.76). That is why a warm compile
cache saves $0.64, and why criterion 18 records which cache each shard used. The upper figure
is dominated by the idle tails ($2.34 of $6.28): it charges every container the full 300 s
scale-down window plus 240 s of client slack, because whether Modal applies the lowered window
to an already-idle container is not yet confirmed on a live job. Criterion 16 measures exactly
that, so after this run the upper bound can come down.

### Size options

`--sizing` prints this table (`size_options`, cold start, all steps):

| Episodes | Video-h | seg × cap | Shards seg / cap | Nominal | Upper | Wall | Wall (upper) | Peak seg / cap |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.118 | 1 × 2 | 2 / 5 | $1.91 | $4.68 | 13 min | 19 min | 1 / 2 |
| 16 | 0.118 | 2 × 2 | 2 / 5 | $2.53 | $5.43 | 13 min | 19 min | 2 / **1** |
| 16 | 0.118 | 2 × 3 | 2 / 5 | $3.41 | $6.88 | 13 min | 19 min | 2 / **1** |
| 32 | 0.223 | 1 × 2 | 6 / 9 | $2.48 | $5.58 | 15 min | 23 min | 1 / 2 |
| **32** | **0.223** | **2 × 2** | **6 / 9** | **$2.77** | **$6.32** | **15 min** | **23 min** | **2 / 2** |
| 32 | 0.223 | 2 × 3 | 6 / 9 | $3.32 | $7.78 | 15 min | 22 min | 2 / 3 |
| 48 | 0.354 | 1 × 2 | 9 / 14 | $3.23 | $6.70 | 19 min | 28 min | 1 / 2 |
| 48 | 0.354 | 2 × 2 | 9 / 14 | $3.55 | $7.45 | 19 min | 29 min | 2 / 2 |
| 48 | 0.354 | 2 × 3 | 9 / 14 | $4.07 | $8.90 | 17 min | 25 min | 2 / 3 |

**Recommendation: 32 episodes on 2 × L40S + 2 × H100!** (`recommend_size`). It is the largest
size that:
- has at least 2 containers in each pool, and both pools reach them in the schedule (16
  episodes give only 2 segment shards, so the second caption container never starts);
- gives every container at least 2 shards, so each one is a rate sample, and the pools overlap;
- stays within **$7 at the upper projection** (48 episodes on 2 × 2 needs $7.45).

1 × 2 is $0.29 cheaper but cannot measure segment with two containers. 2 × 3 adds a third
caption cold start and $1.46 of upper for no new kind of evidence.

### Feeding the rates back (`--finalize`)

After a real lite run, `--finalize` folds its measurements into `planner/rates.json` as **more
samples**. It never replaces the certified run's record:

- **The certified run's figure stays as the `base` sample.** On the first fold, a measured row's
  own value, its `n`, its video seconds and its per-job range (read back from the provenance the
  certified run's finalize wrote) become one sample in the row's new `samples` list, unchanged.
- **Each lite container is one more sample**: `flora report`'s busy seconds for that container
  over the video seconds its receipts finished. A container on another GPU than its pool's, or
  with under 30 video-s, is dropped by name. Fewer than 2 usable containers in a pool refuses
  that pool.
- **The row is re-pooled.** Its rate is the video-second-weighted mean of all samples, `n` is
  their total, and the band is `max(25%, (max − min) / mean)` over every sample value, so it
  comes from the real spread instead of the ±50% that a single sample is given.
- **Startups** go to the startup row of the compile-cache state the pool's shards recorded:
  `cold`, or `warm` (which turns the warm placeholder into a measurement). A pre-warmed caption
  container is skipped, because its spawn-to-start is not a startup. A pool with mixed states
  folds no startup.
- **Nothing is folded** unless the job ended `done` with no lost or duplicate record.
- **Idempotent.** Folding the same run again changes nothing. If the certified finalize
  (`scripts/wave_h_finalize.py --write`) runs again later, it rewrites its own base sample and
  re-pools the lite samples on top.

`--finalize` is a dry run that prints each row's before and after. `--write` writes
`rates.json` and regenerates the generated tables of `docs/costs.md`. The pooling itself is
`fold_samples` in `scripts/wave_h_finalize.py`, shared by both scripts.

## The full profile

### Goals

The full test must validate, at a scale none of the earlier runs reached (≤ 6 episodes, one
container per pool):

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
v3.0. Both profiles use it; lite takes its first 32 episodes.

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

The fake modes use synthetic 3–6 s clips (`local.sample`) with the same total video length as
the profile: 784 clips for full (3,529 s) and 178 for lite (802 s). The fake modes split the
video into equal shards; only the real mode packs lite's exact shards.

## Job shape (full)

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

Two more settings matter (both profiles):
- **Deploy** uses `--reconcile-cron on`, so the reconciler runs every 10 minutes.
- **Run** uses `flora run --detach`. The script polls by itself: `flora status --json` and
  `modal container list --json` every 30 s (10 s for fake-modal, 1 s for local). After the job
  ends it keeps listing containers every 5 s for up to 45 s, until no worker container is left,
  so each one's end is known to 5 s (criterion 16).

**Chaos actions** (full only). Each fires once (`Chaos`):
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

## Cost estimate (full; dry run, measured rates)

The dry run prices each container class at its own pinned list price (`jobs.budget`): L40S
workers at $2.71/h, H100 workers at $5.10/h, and the orchestrator at the preemptible CPU rate
(0.125 cores, 0.25 GiB). `planner/rates.json` supplies the work rates and startups. The
`balanced` rows are **measured**, from the first `balanced` GPU run (run 102207184317,
[`wave-h-report.md`](wave-h-report.md)):

| Kind | GPU | Work (container-s per video-s) | Cold startup | Warm startup |
|---|---|---|---|---|
| `segment` | L40S | **0.367**, measured, n=1, ±50% | **300.5 s**, measured, n=1, ±50% | 150 s, *placeholder*, ±60% |
| `caption` | H100! | **0.906**, measured, n=1, ±50% | **325 s**, measured, n=1, ±50% | 180 s, *placeholder*, ±60% |

The work rates were measured before batch fill (`engines/batch_policy.py`), so they can only
overstate the work. Criterion 8 compares the run against them, and criterion 19 measures how
full the batches actually are.

**The warm variant** (`--assume-warm`) prices a run whose compile cache is already published
for each pool's key (`engines/compile_cache.py`). It uses the `compile_cache: warm` startup row
from `rates.json` for each balanced kind. The `balanced` warm rows exist as placeholders (150 s
/ 180 s, ±60%, from the lab's warm engine builds). The cold × 0.72 fallback (the Rewarder's
229 s / 318 s, `docs/cold-start.md`) applies only to a kind with no warm row. In a real run the
cache is published by the first container of each pool after its first shard, so containers
that start before that still start cold. The **caps always come from the cold projection**, so
`--assume-warm` never raises the spend allowed.

### How the projection is built

The run's cost comes from a shard-level schedule (`simulate`), not from a fluid estimate. It
models the deployed pool behaviour (`jobs/pools.py`):
- all segment shards are queued at t=0;
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

The full profile splits the video into equal shards at its shard target. The lite profile packs
the exact shards the manifest will build from the pinned episode table (grouped by source file,
in episode order).

This gives each pool's starts, busy time, waits between shards and tails. The chaos costs are
added on top for the profile's chaos actions: one replacement start plus a lost shard per
preemption, and the reconciler gap (each caption container idles 300 s while the orchestrator is
down, scales down, and restarts). The **upper** figure runs the same schedule at the upper
rates and startups. It counts the full 300 s window plus 240 s of client slack per container,
because whether Modal applies the lowered window to a container that is already idle is not yet
confirmed on a live job (criterion 16 measures it), and the full 660 s reconciler wait.

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
numbers for the chosen profile, and it prints both variants on every run. The numbers move with
the rate table; these are from 2026-10-22.

The old estimate, from placeholder rates at 3 + 2, was $9.58 nominal and $19.49 upper. The new
nominal is higher for three reasons:
- the measured startups are 300–325 s, not the projected 130 s;
- the work rates are still within the old placeholder band (0.367 against 0.5, 0.906 against
  1.0);
- the third caption container adds one more cold start and one more reconciler-gap restart.

The upper figure is lower than before, because the error bars shrank from ±100% to ±50%.

How the full run's nominal cost breaks down (cold, 2 + 3):

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

- **Fake-modal:** $0.10 nominal / $0.34 upper for full (caps $0.50 global). That is 5 CPU
  worker containers (2 cores, 12 GiB, $0.19/h) whose fake rate and startup are
  **placeholders**, plus the orchestrator. Lite's fake-modal rehearsal is $0.02 nominal /
  $0.16 upper (caps $0.30 global), on 178 synthetic clips and 4 CPU worker containers.
- **Caps:** each is the upper projection rounded up to $0.10 per step. The script also prints
  the unrounded upper numbers (`honest caps`, global $19.40 for full, $6.40 for lite). They are
  proposals. Raising them, or approving the spend, is the user's decision.
- **Retries:** an attempt that is killed or unreported is charged its whole budget until billing
  is an hour past it. A retry soon after such an attempt is therefore refused by design (see
  `docs/wave-h-runbook.md`).

For comparison, `flora plan` (offline, same rates, cold) picks the same layout for the full
test:

| Option | Containers | Cost (range) | Wall |
|---|---|---|---|
| `balanced x1` | 2 × L40S + 3 × H100! | $7.44 ($4.99–$11.10) | 0.39 h |
| `exact x4` | 5 × H200 | $36.63 ($26.97–$49.93) | 1.28 h (infeasible) |
| `exact x8 +long_lived_container` | 5 × H200 | $14.89 ($9.25–$24.06) | 0.52 h (at risk; a projection, never recommended) |

`flora plan` charges each container the same 17 s drained tail as the script. It is cheaper
than the script's $11.46 because it does not model client slack or the chaos actions. Its $7.44
is close to the script's $7.89 without chaos. Measuring that gap is part of the point.

## Pool sizing

`scripts/scale_test.py --profile full --sizing` prints this table for both variants
(`pool_sizing`). It runs the same schedule and chaos model for 1–4 containers per pool, at cold
start.

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

**Recommendation for full: 2 × L40S + 3 × H100!** (`recommend`). This is the cheapest size that:
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

The report evaluates all 19 criteria **automatically** (`evaluate`), from what `run` and
`report` record. None of them is manual (`MANUAL` is empty). Criteria 1–14 are the original
ones; 15–19 were added with the lite profile and apply to both profiles. The report prints one
row per criterion with its value, target, PASS / FAIL / n.a. and a reason, plus a summary line
(`criteria_summary`).

The verdict rules:
- **A value the run did not record is a FAIL ("not recorded"), not a pass.** For example, a
  chaos action that never fired fails criterion 10 or 11, and a pool whose shards carry no
  compile-cache state fails criterion 18.
- **`n.a.` always carries a reason**, and is used in two cases only:
  - in `local`, where there are no containers, cold starts or money, criteria 3, 4, 5, 7, 8, 9,
    15, 16 and 17 do not apply, and 18 and 19 mean nothing on fake engines;
  - in `lite`, criteria 10 and 11 are not fired (see [Why no chaos](#job-shape)).
- **`--report` re-checks criteria 7 and 8** against the run step's spend as it stands, which is
  the billed delta once Modal billing lands (about an hour after the run). Until then, the
  tracked spend is used and the reason says "provisional".

Targets marked *derived* come from the cold **upper** schedule of the configured pools, so they
move with `rates.json` and with the profile (`criteria_targets`). The values shown are for the
current rates, full at 2 + 3 and lite at 2 + 2.

| # | Criterion | Target | Recorded from | Decision it informs |
|---|---|---|---|---|
| 1 | End state | `done` | the final `flora status` state | the job system survives scale plus chaos; it gates the first public release |
| 2 | Lost or duplicate records | 0 (compose `missing` + conflicts) | `job.summary()` | correctness of retry, dedupe and compose at 8 + 15 shards (lite: 6 + 9) |
| 3 | Each pool reaches `max_containers` | full: segment 2, caption 3; lite: 2 and 2 | peak concurrency from `modal container list` samples | whether the factory's `max_containers` is honoured, i.e. whether the planner's container counts can be trusted |
| 4 | Time to peak concurrency, from the run start | *derived*: full: segment ≤ 150 s, caption ≤ 844 s; lite: segment ≤ 150 s, caption ≤ 733 s (upper schedule + a poll + 120 s launch margin); a pool that never reaches its size fails | container samples and the run start | whether `min_containers` / warm pools are needed for short jobs |
| 5 | Utilization while holding work, per pool | ≥ 0.60 busy ÷ container-seconds, from each container's first shard start to its last shard end | attempt markers and receipts | the release goal of ">60% GPU utilization while busy"; the pool ratio and shard size |
| 6 | Pipelining: first caption start after the first segment receipt | *derived*: ≤ 608 s (upper caption cold start + 120 s), and overlap > 0 | attempt markers and receipts | whether `depends_on` per shard (not per stage) works; the wall-time model in `flora plan` |
| 7 | Spend within the upper projection | actual ≤ upper | tracked spend, or the billed delta once it lands | whether the stop rules and caps are safe at scale |
| 8 | Spend vs the nominal projection | \|actual ÷ nominal − 1\| ≤ 0.5 | as 7 | whether the measured `balanced` rows hold at n > 1 (they were n=1, and measured before batch fill) |
| 9 | Idle / scale-down share of pool cost | ≤ 0.15, leaving out tails that began while the killed orchestrator was down | container samples, receipts, and the kill and revive times | whether the job-aware drain (`jobs/pools.py`) actually ends the tail on a live job |
| 10 | Preemption recovered | each pool has a `preempted` start, and its shard still ends with a receipt from a later attempt; **lite: n.a.** | attempt markers and receipts | spot instances are safe for `balanced` |
| 11 | Orchestrator revived by the reconciler | kill fired, revived by the reconciler (not the `flora resume` fallback) within 660 s, and the job ends `done`; **lite: n.a.** | the first spawn or adoption in a ledger after the kill | whether the reconciler cron is mandatory in `flora deploy`; continuation semantics |
| 12 | `flora status` p95 | ≤ 10 s | the run's status polls | whether status needs a cached `job.json` instead of a full reconcile |
| 13 | `flora report` | ≤ 30 s | timed in `report` | as above |
| 14 | `flora compose` | ≤ 60 s, exit 0 | timed in `run` | whether compose must move to Modal, or be incremental, for 10+ hour jobs |
| 15 | Throughput per pool from several containers | ≥ 2 containers per pool that finished a shard, each with a measured container-s per video-s; the pool's figure and the per-container range are reported | attempt markers, receipts and their episodes' lengths | the `rates.json` value and band at n > 1 (`--finalize`) |
| 16 | Scale to zero at the end of the job | every worker container gone ≤ 30 s after its pool was drained | the container samples (every 5 s after the job ends) and the orchestrator's drain record | whether the upper projection can stop charging the full 300 s window plus slack per container |
| 17 | Caption pre-warm | the pre-warm was planned and started, and the first caption container started within 120 s of the planned start | the orchestrator's pre-warm record and `flora report`'s container rows | whether the pre-warm saves a caption cold start of wall time, as the schedule assumes |
| 18 | Compile cache, per shard | every shard names the compile cache it ran with; a pool whose cache was published before the run loads it on every shard | the shard audit rows and receipts (cache state), and `plan`'s check of the cache Volume | whether warm starts work on a live job, and which startup row the measured startups belong to |
| 19 | Batch fill | mean batch ≥ 2 and GPU-busy fraction ≥ 0.60 per pool; also reported against the engines' caps (TempSeg 48, Captioner 16) | the shard audit rows (the micro-batcher's rounds, batch sizes, generate and wall seconds) | whether batch fill delivers the throughput the rates assume, and whether larger shards would help |

**At lite size**, criteria 12–14 are judged as usual but are easy to pass with 15 shards; they
say little about large jobs. Criterion 9 has no outage to leave out.

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
- **Criteria 15–19** are new. They turn what the lite profile must prove into automatic checks.
  The batch-fill and pre-warm targets are proposals, like the rest.

**What each metric feeds.**

| Metric | What it feeds |
|---|---|
| Video-s per container-s, per pool and per container | the `rates.json` value and band at n > 1 (`--finalize`) |
| Cold-start distribution and spawn-to-start latency | the `startup` rows: cold, or warm once a compile cache is published (`--finalize`) |
| Idle seconds and idle cost; each container's end after the drain | the drained-tail model in `jobs.budget` and `flora plan`, and the upper projection's tail |
| The pre-warm record against the first caption container | whether the pre-warm's saving is real |
| Mean batch and GPU-busy fraction | batch policy and shard size |
| Projection error (script, `flora plan`, the orchestrator's first projection) | which estimator to trust |

## What a run proves

**Lite**, if every criterion that applies passes on `--mode real`, at 2 + 2 containers, 32
episodes and 0.22 video-hours:
- **Several containers per pool work, and their throughput is measured.** Both pools reach 2
  containers and each container is one rate sample, which `--finalize` folds into
  `rates.json` (criteria 3, 4, 5 and 15).
- **Per-shard pipelining, the caption pre-warm and the job-end drain work on a live job**
  (criteria 6, 16 and 17).
- **The cost model holds at n > 1**, warm or cold, and batch fill is effective (criteria 7, 8,
  18 and 19).
- **The job completes with no lost or duplicate records** (criteria 1, 2, 12–14).

**Full**, if all 19 pass, at 2 + 3 containers, 128 episodes and about one video-hour, it also
shows:
- **The job system completes a multi-container job through chaos.** That covers 8 + 15 shards,
  two mid-shard preemptions and an orchestrator kill, with no lost or duplicate records
  (criteria 1, 2, 10 and 11).
- **The idle policy and the client commands are fit for this size** (criteria 5, 9, 12, 13 and
  14).

What neither proves:
- the 23 h continuation hand-off (see the limits below);
- warm-start startups, unless a compile cache was published for each pool's key before the run;
- behaviour at 10+ video-hours, beyond the trend in criteria 12–14;
- anything about output quality. This test checks the job system, not the captions.

## Known limits of the test

- **The 23 h continuation deadline is not reached.** A one-hour job never hits
  `orchestrator.DEADLINE_S` (23 h), and `modal_app` has no knob to shorten it. The full test
  instead kills a generation mid-job. The successor's re-attach is the same code path (a fresh
  `Orchestrator` rebuilding its view from the ledgers), but the `continuation_spawned` hand-off
  itself is covered only by unit tests. A `FLORA_ORCH_DEADLINE_S` override in
  `modal_app/orchestrator_app.py` (not owned here) would let the test cover it.
- **Container lifetimes are sampled.** Lifetimes come from `modal container list` samples, at
  the poll resolution of 30 s during the job and 5 s after it. Locally, a container's life is
  its first start to its last receipt. Busy time comes from attempt markers and receipts.
- **A container's video seconds come from its receipts.** A shard a container started but
  another finished counts toward the second only, and its partial work toward the first's busy
  time. Lite fires no preemption, so this does not arise there.
- **Local spend is not money.** The local runner prices fake call seconds at GPU list prices, so
  its "spend" exercises the budget gate only.
- **Local revival.** A local `flora resume` is a no-op until the killed orchestrator's lease
  expires (30 s TTL). The script resumes once status reads the orchestrator as dead.

## Local rehearsal result (full profile)

I ran `--mode local --execute` on 2026-10-22, with 784 synthetic episodes (3,527 video-s), at
the earlier 3 + 2 pools. All six steps completed in about 2 minutes, and the job ended `done`.
The full numbers are in `reports/scale-test/local/report.md` (git-ignored, so not in a fresh
checkout).

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
automatic)**; there were 14 criteria then. The six n.a. are criteria 3, 4, 5, 7, 8 and 9, which
need containers or money.

**A bug the rehearsal found and fixed.** The first attempt hung because `flora resume`, issued
right after the kill, saw the unexpired lease ("orchestrator still running") and did nothing.
The script now resumes only once the orchestrator reads dead.

## Local rehearsal result (lite profile)

I ran `--mode local --execute` (lite) on 2026-10-22, with 178 synthetic episodes (802 video-s)
at 2 + 2. All six steps completed, and the job ended `done` in about 9 seconds of job time.
- 4 `segment` and 7 `caption` shards (the fake modes split the video evenly), and 178 of 178
  records per stage, with 0 missing, 0 excused and 0 conflicts.
- No chaos fired. The first caption shard started at the first segment receipt, and the pools
  overlapped.
- `flora status` p95 0.6 s, `flora report` 0.3 s, `flora compose` 0.5 s.
- The report read **6 PASS, 0 FAIL, 13 n.a. (19 of 19 automatic)**. The n.a. are criteria 3–5,
  7–9 and 15–19, which need containers, money or real engines, and 10 and 11, which lite does
  not fire.
