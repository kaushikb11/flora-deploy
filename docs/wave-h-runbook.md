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

There are two jobs because a config has one embodiment. The cost of that is a second cold start
in the exact e2e step. Rewards (step c) run on `cups_open`, `towel` and `robointer`. The noise
run (step f) uses `cups_open` and `towel`.

Every job is staged the same way:
- the episodes are fetched into a local `mp4_dir`, with the task text next to each video;
- the dataset is uploaded to the dedicated media Volume and named as `input.stage_to`;
- `flora index --local` writes the receipts into a local store, and they are mirrored into the
  job store on the Volume;
- then `flora plan --index`, `flora deploy` and `flora run`.

## Steps and caps

| Step | What | GPU | Cap | Projected (upper) |
|---|---|---|---|---|
| a | balanced preflight: derived dirs, receipts, EOS ids | CPU | $0.05 | $0.00 |
| b | exact e2e, deterministic mode, via `flora deploy`/`flora run`; apps stopped right after compose | H200, one per app | $1.20 | $1.19 ($1.79) |
| c | `reward.terminal` + `reward.dense` on 3 episodes, one container (`max_inputs: 2`) | H100 | $0.60 | $0.37 ($0.54) |
| d | balanced lab: TempSeg, then Captioner on TempSeg's segments, L0 + Gate A, ≤ 10 min each | L40S + H100 | $1.20 | $1.00 ($1.30) |
| e | offline provisional certification of `balanced` from (b) + (d) | none | $0 | $0 |
| f | 2 episodes in the default `nvidia` mode vs (b): the noise floor; (e) is re-evaluated | H200 | $0.40 | $0.39 ($0.62) |

Projections are container-seconds × list $/h (GPU + cores + memory, as pinned in `jobs.budget`).
Container-seconds come from `planner/rates.json`: startup + video-s × container-s per video-s, plus
30 s from the last receipt to `modal app stop`. "Upper" uses the table's upper quantiles, and the
lab's 10-minute limit for (d).

## Stop rules

- **Before a step:** its projection must fit its cap and the global budget left. Otherwise the
  runbook stops (exit 3) and records why.
- **During a GPU step:** the time that is still affordable is (remaining step cap) ÷ (container
  $/s). It is enforced three ways:
  - as `flora run --budget` (the job's own budget gate);
  - as a client-side deadline that cancels the job;
  - as the timeout of each lab call.

  Every app is stopped (`modal app stop`) once its job ends.
- **After a step:** spend is the larger of the tracked cost and the billed delta. Tracked cost is
  `flora report` (container receipts and calls) or the lab's wall time × $/h. The billed delta is
  `scripts/spend.py` / `modal billing report` against the baseline taken at the start (billing
  lags about an hour). If that spend is over the step's cap or the global cap, the runbook stops.

## State and report

- **State:** `reports/wave-h/state.json`, which is git-ignored because it names private episodes.
  A finished step is never re-run. A stopped or failed step is retried on the next `--yes-spend`;
  `--only <step>` runs one step.
- **Report:** `docs/wave-h-report.md`. It holds numbers only. Captions, tasks and segment rows
  stay in `reports/wave-h/`.
- **Provisional certificate:** `reports/wave-h/certificate-provisional.json`. Its checks:
  - L0 ids and grids are equal on every captured request;
  - Gate A max |Δlogit| ≤ 0.5;
  - boundary MAE, segment IoU and caption token Jaccard against `exact` are within 2× the
    measured exact-vs-exact noise, or within absolute bounds before step f has run.

## What H0 proves first

H0 (`scripts/h0_rehearsal.py`) runs steps (b) and (c) on the same episodes with
`flora deploy --fake-models`. That is the real factory app: the same orchestrator, stores,
Volumes, media mounts, env, secrets, worker contract and timeouts. The differences are:
- G2's fake engines in CPU containers;
- a distinct `fake_models` hash component, so fake outputs never share an app or store path with
  real ones;
- records marked `certificate: fake-models`.

It also injects one `simulate_preemption()` and one poison episode, and asserts the end state,
lost and duplicate records, stage-hash keyed paths, the media root, instruction flow and the
determinism setting. Then it stops its apps and deletes its two Volumes.
