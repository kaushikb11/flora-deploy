# Wave H-lite runbook

Wave H-lite is the one GPU session that replaces Wave H (decision of 2026-10-22 in
`docs/NEXT_EXECUTION.md`). The budget is about $3, with a hard global cap of **$3.20**. It runs
only after three things are true: review 6 is clean, CI is green and H0 passes.

```
uv run python scripts/h0_rehearsal.py          # H0: CPU rehearsal on real Modal (~$0.10)
uv run python scripts/wave_h.py --dry-run      # the plan and its estimates; spends nothing
uv run python scripts/wave_h.py --yes-spend    # after the user approves; resumable
uv run python scripts/wave_h.py --report       # rewrite docs/wave-h-report.md from the state
```

## Episodes

H0 and every Wave H step use the same 6 short golden episodes (`flora-golden:golden/v1/`),
76.5 s of video in total:

| Job | Episodes | Views |
|---|---|---|
| `human` | EgoDex `add_remove_lid/000014`, `braid_unbraid/000001` | 1 |
| `yam` | ALOHA `static_cups_open/000000`, `static_towel/000000` (also a dense target) | 3 |
| `yam` | WGO `robointer_droid_000010`, `galaxea_045` (single-view robot, profile `yam`) | 1 |

There are two jobs because a config has one embodiment. As a result, (b) has two H200 cold starts
and (d) has two cold starts per pool. Rewards (step c) run on `cups_open`, `towel` and
`robointer`. The noise run (step f) repeats (b)'s `human` job: the same config file, staged data
and stage hash, under a new job id. Certification needs the noise run to repeat the oracle, and
the input content is hashed, so a smaller dataset would get another stage hash.

Every job is staged the same way:
- the episodes are fetched into a local `mp4_dir`, with the task text next to each video;
- the dataset is uploaded to the dedicated media Volume and named as `input.stage_to`;
- the derived-weight receipts are read from `flora-weights` and checked against
  `pins.DERIVED_PINS` (`config.receipts.check_pinned`). Their file sha256 is written into the
  generated config's `weights.derived`, which is how `flora deploy` resolves them
  (`config.receipts.resolve`). Step (d) also refuses an unpinned derived weight, since the
  balanced workers fail closed on it;
- `flora index --local` writes the receipts into a local store, and they are mirrored into the
  job store on the Volume;
- then `flora plan --index`, `flora deploy` and `flora run`.

## Steps and caps

The oracle (b) and the noise run (f) use NVIDIA's **default mode** (`determinism: nvidia`).

| Step | What | GPU | Cap | Nominal | Upper |
|---|---|---|---|---|---|
| a | balanced preflight: derived dirs, receipts, EOS ids | CPU | $0.05 | $0.00 | $0.01 |
| b | exact oracle, default mode, via `flora deploy`/`flora run`; 2 apps, each stopped right after compose | 2 x H200 | **$1.80** | $1.19 | **$2.55** |
| c | `reward.terminal` + `reward.dense` on 3 episodes, one container (`max_inputs: 2`) | H100 | $0.90 | $0.37 | $0.87 |
| d | `preset: balanced` (`allow_uncertified: true`) via `flora deploy`/`flora run` on (b)'s 2 jobs, then the lab's L0 + Gate A on 6 captured requests | 2 x (L40S + H100), lab L40S + H100 | $4.35 | $2.03 | $4.32 |
| e | offline provisional certification of `balanced` from (b) + (d) | none | $0 | $0 | $0 |
| f | (b)'s `human` job repeated, default mode: the noise floor; (e) is re-evaluated | H200 | $1.20 | $0.52 | $1.17 |
| | **total** | | global **$3.20** | **$4.12** | **$8.91** |

The plan **does not fit the $3.20 global cap**. Even the nominal total is $4.12. At upper rates
the minimum honest global cap is **$9.05**, and (b)'s honest cap is $2.55 rather than $1.80. The
dry run prints this, and the stop rules refuse (b) as configured. Raising the caps, or cutting
steps, is the user's decision. The biggest items are:
- d's four cold starts plus the lab's two engine pairs;
- the 240 s client slack per app;
- the balanced rows of `planner/rates.json`, which are placeholders at ±100%.

Projections are container-seconds × list $/h (GPU + cores + memory, as pinned in `jobs.budget`).
Container-seconds come from `planner/rates.json`: startup + video-s × container-s per video-s, plus
30 s from the last receipt to `modal app stop`. "Upper" adds:
- the table's upper quantiles;
- **240 s client slack per app** (the `flora run` timeout margin, then cancel + `modal app stop`);
- every cold start;
- the segment pool's scale-down tail while captions run;
- the lab's 10-minute limit plus 60 s.

## Stop rules

- **Before a step:** its **upper** projection must fit its cap and the global cap minus the
  cumulative spend. Otherwise the runbook stops (exit 3) and records why.
- **During a GPU step:** the affordable wall time is (remaining step cap − slack × $/s) ÷ $/s,
  so the deadline plus the slack never exceeds the cap. It is enforced three ways:
  - as `flora run --budget` (the job's own budget gate);
  - as a client-side deadline that cancels the job;
  - as the timeout of each lab call.

  Every app is stopped (`modal app stop`) once its job ends.
- **After a step:** the step's spend is checked against its cap, and the cumulative figure
  against the global cap.

## Spend accounting

- **Every attempt counts.** Each run of a step is kept, and a retry adds to the earlier attempts
  instead of overwriting them.
- **Kills leave a record.** While a step runs, a heartbeat stores elapsed × the step's peak $/s
  every 30 s. A process killed mid-step therefore leaves its spend behind. The next
  `--yes-spend` marks the attempt `killed`, stops its apps and retries the step.
- **Unknown spend is charged at budget.** A killed attempt, or one whose `flora report` failed,
  is charged its whole budget until billing has been read at least an hour after it.
- **Billing is re-read before every step and after it, for every app seen so far.** A step's
  billed delta covers only the apps no other step used. (f) reuses (b)'s app, and the lab app is
  shared by (a) and (d); those count in the global billed total only.
- **The global figure** is max(Σ per step max(charged attempts, billed delta), the global billed
  delta).

## State and report

- **State:** `reports/wave-h/state.json`, which is git-ignored because it names private episodes.
  A finished step is never re-run. A stopped, failed or killed step is retried on the next
  `--yes-spend`; `--only <step>` runs one step.
- **Report:** `docs/wave-h-report.md`. It holds numbers only. Captions, tasks and segment rows
  stay in `reports/wave-h/`.
- **Provisional certificate:** `reports/wave-h/certificate-provisional.json`. Its checks:
  - L0 ids and grids are equal on every captured request;
  - Gate A max |Δlogit| ≤ 0.5;
  - boundary MAE, segment IoU and caption token Jaccard of the deployed balanced records
    against `exact` are within 2× the measured exact-vs-exact noise, or within absolute bounds
    before step f has run.

  With 2 noise episodes (fewer than 5) and 6 candidates (fewer than 10 per class), H-lite is
  **provisional by design** (`docs/certification.md`).

## What H0 proves first

H0 (`scripts/h0_rehearsal.py`) runs steps (b), (c) and (d)'s deployment on the same episodes with
`flora deploy --fake-models`. That is the real factory app: the same orchestrator, stores,
Volumes, media mounts, env, secrets, worker contract and timeouts. The differences are:
- G2's fake engines in CPU containers;
- a distinct `fake_models` hash component, so fake outputs never share an app or store path with
  real ones;
- records marked `certificate: fake-models`.

The exact `yam` job gets one `simulate_preemption()` and one poison episode. The balanced jobs
cover the same 6 episodes:
- `bal-human`: clean;
- `bal-yam-seg`: a `segment` preemption, with the poison episode poisoned at `segment`. It is
  excluded at caption too, and counted once;
- `bal-yam-cap`: a `caption` preemption, with the poison episode poisoned at `caption` only.

H0 asserts the end state, lost and duplicate records, stage-hash keyed paths, the media root,
instruction flow, the determinism setting and the upstream exclusion count. Then it stops its
apps and deletes its two Volumes. `--preset exact|balanced` runs one family only.
