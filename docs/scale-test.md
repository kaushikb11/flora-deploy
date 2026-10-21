# Scale test of `balanced` (prepared, not run)

`scripts/scale_test.py` runs `balanced` on about one video-hour, with several containers per pool,
through the real `flora index -> plan -> deploy -> run -> report` path. It is prepared but has
**not been run on Modal**. Only the local fake-models variant has run.

```
uv run python scripts/scale_test.py                                # dry run (default): plan + costs
uv run python scripts/scale_test.py --mode local --execute         # local fake run (no Modal, $0)
uv run python scripts/scale_test.py --mode fake-modal --yes-spend  # CPU rehearsal on Modal (~$0.13)
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
| `segment` pool | **3 × L40S** (6 cores, 48 GiB; $2.62/h) |
| `caption` pool | **2 × H100!** (8 cores, 64 GiB; $4.84/h) |
| `max_gpus` | 5 |
| `shard_target_gpu_minutes` | 4: about 8 `segment` shards (~480 video-s each) and 15 `caption` shards (~240 video-s each) |

With these shards, each segment container runs about 3 shards, and captions can start after the
first segment shard. The local run confirmed 8 + 15.

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

## Cost estimate (dry run, current rates)

The dry run prices container-seconds at pinned list prices (`jobs.budget`).
`planner/rates.json` supplies the work rates and startup times:
- `balanced` `segment`: 0.5 container-s per video-s, **placeholder ±100%**;
- `balanced` `caption`: 1.0 container-s per video-s, **placeholder ±100%**;
- startup: 130 s, **projected ±50%**.

The upper figure adds, per container:
- the upper rate and startup;
- a full 300 s scale-down window;
- 240 s of client slack.

It also adds one replacement start plus one lost shard per preemption, and the reconciler gap
(caption containers idle 300 s, then restart).

| Step | Cap | Nominal | Upper |
|---|---|---|---|
| verify (Hub metadata) | $0 | $0 | $0 |
| index (13 CPU units; **placeholder** 120 s / 600 s each) | $0.20 | $0.03 | $0.14 |
| plan | $0 | $0 | $0 |
| deploy (image builds; **placeholder**) | $0.10 | $0.00 | $0.01 |
| run | **$19.40** | **$9.55** | **$19.34** |
| report | $0 | $0 | $0 |
| **total** | global **$19.70** | **$9.58** | **$19.49** |

Caps are derived, not chosen: each step's cap is its upper projection rounded up to $0.10, and
the global cap is their sum. `scripts/scale_test.py --dry-run` prints the current numbers, which
move with the rate table (these are from 2026-10-22, with balanced rates still placeholders).

How the run's nominal cost breaks down (at the earlier $9.11 estimate):

| Pool | Item | Cost |
|---|---|---|
| `segment` | 3 cold starts | $0.28 |
| `segment` | work | $1.28 |
| `segment` | 3 scale-down tails | $0.65 |
| `segment` | preemption | $0.18 |
| `caption` | 2 cold starts | $0.35 |
| `caption` | work | $4.75 |
| `caption` | tails | $0.08 |
| `caption` | preemption | $0.34 |
| `caption` | reconciler gap | $1.16 |
| orchestrator | | $0.04 |

The upper figure is about 2× the nominal because both balanced rates are placeholders at ±100%.

- **Wall time:** 43 minutes nominal, 84 minutes upper.
- **Fake-modal:** $0.13 nominal / $0.34 upper (caps $0.40, global $0.45). That is 5 CPU
  containers (2 cores, 12 GiB, $0.19/h) whose fake rate and startup are **placeholders**.
- **Caps:** they equal the honest upper numbers (`honest_caps`). They are proposals. Raising
  them, or approving the spend, is the user's decision.
- **Retries:** an attempt that is killed or unreported is charged its whole budget until billing
  is an hour past it. A retry soon after such an attempt is therefore refused by design (see
  `docs/wave-h-runbook.md`).

For comparison, `flora plan` (offline, same rates) gives:

| Option | Containers | Cost (range) | Wall |
|---|---|---|---|
| `balanced x1` | 2 × L40S + 3 × H100! | $6.76 ($3.50–$13.17) | 0.36 h |
| `exact x4` | 5 × H200 | $36.56 | |

The planner prefers **2 segment + 3 caption** containers under the placeholder rates. The test
keeps the requested 3 + 2 on purpose: the measured ratio decides which layout is right, which is
decision (a) below.

`flora plan` is cheaper because it does not model:
- the scale-down tail;
- client slack;
- the chaos actions;
- pipelining delay.

Measuring that gap is part of the point.

## Success criteria

These are proposals; they are checked in the report (`evaluate`).

| # | Criterion | Target | Decision it informs |
|---|---|---|---|
| 1 | End state | `done` | the job system survives scale plus chaos; it gates v0.2 |
| 2 | Lost or duplicate records | 0 (compose `missing` + conflicts) | correctness of retry, dedupe and compose at 8 + 15 shards |
| 3 | Each pool reaches `max_containers` | segment 3, caption 2 | whether the factory's `max_containers` is honoured, i.e. whether the planner's container counts can be trusted |
| 4 | Time to peak concurrency | ≤ 450 s (2 upper cold starts + a poll) | (a) whether `min_containers` / warm pools are needed for short jobs |
| 5 | Utilization (busy ÷ container-seconds) per pool | ≥ 0.60 | v0.2's ">60% GPU utilization while busy"; (a) the pool ratio and shard size |
| 6 | Pipelining: first caption start after the first segment receipt | ≤ 315 s, and overlap > 0 | whether `depends_on` per shard (not per stage) works; the wall-time model in `flora plan` |
| 7 | Spend within the upper projection | actual ≤ upper | (b) whether the stop rules and caps are safe at scale |
| 8 | Spend vs the nominal projection | \|actual ÷ nominal − 1\| ≤ 0.5 | (b) **flipping the `balanced` rows of `rates.json` from placeholder to measured** |
| 9 | Idle / scale-down share of pool cost | ≤ 0.15 | (c) whether to lower `scaledown_window` (300 s) for `segment` or stop pools early |
| 10 | Preemption recovered | each stopped container leaves a `preempted` start, and its shard still ends with a receipt | spot instances are safe for `balanced` |
| 11 | Orchestrator revived by the reconciler | ≤ 660 s, and work continues | whether the reconciler cron is mandatory in `flora deploy`; continuation semantics |
| 12 | `flora status` p95 | ≤ 10 s | (d) whether status needs a cached `job.json` instead of a full reconcile |
| 13 | `flora report` | ≤ 30 s | (d) as above |
| 14 | `flora compose` | ≤ 60 s | (d) whether compose must move to Modal, or be incremental, for 10+ hour jobs |

**What each metric feeds.**

| Metric | What it feeds |
|---|---|
| Video-s per container-s, per pool | the new `rates.json` value |
| Cold-start distribution and spawn-to-start latency | the `startup` rows |
| Idle seconds and idle cost | the scale-down model in `jobs.budget` |
| Projection error (script, `flora plan`, the orchestrator's first projection) | which estimator to trust |

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

I ran `--mode local --execute` on 2026-10-22, with 784 synthetic episodes (3,527 video-s). All
six steps completed in about 2 minutes, and the job ended `done`. The full numbers are in
`reports/scale-test/local/report.md`.

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

**A bug the rehearsal found and fixed.** The first attempt hung because `flora resume`, issued
right after the kill, saw the unexpired lease ("orchestrator still running") and did nothing.
The script now resumes only once the orchestrator reads dead.
