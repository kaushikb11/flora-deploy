# Wave H-lite runbook

Wave H-lite is the one GPU session that replaces Wave H (decision of 2026-10-22 in
`docs/archive/NEXT_EXECUTION.md`). The user then narrowed it (2026-10-22): **run the GPU
session balanced only, now.** The default plan, `--plan balanced-only`, runs three steps:
- step 0: build TempSeg with merger v2, check its equivalence, then pin it;
- (a): the balanced preflight;
- (d): `balanced` through the real `flora deploy` / `flora run` on the 6 human episodes, plus the
  lab's L0 input parity and Gate A.

The exact oracle (b), the provisional certification (e) and the noise run (f) are **deferred to
the certified run**, a later session that uses `--plan certified` (see "The certified run"
below; `--plan full` cannot certify). The approved scope is about $2.1
nominal and $4.2 upper. With the cold-compile startup priced in (see "Cold compile" below), the
dry run comes to **$2.25 nominal and $4.42 upper**, so its global cap is **$4.50**. The session
runs only after review 10 is clean, CI is green and H0 passes.

```
uv run python scripts/h0_rehearsal.py               # H0: CPU rehearsal on real Modal (~$0.10)
uv run python scripts/wave_h.py --dry-run           # balanced-only: the plan and its estimates
uv run python scripts/wave_h.py --dry-run --offline # ... without reading the golden manifest
uv run python scripts/wave_h.py --yes-spend         # step 0; exits 4 with the pin to make
#   pin the digest (one line in src/flora_deploy/config/pins.py), commit, then:
uv run python scripts/wave_h.py --yes-spend         # step 0 done at $0, then (a), then (d)
uv run python scripts/wave_h.py --status            # spend, holds, cap, the next decision
uv run python scripts/wave_h.py --report            # rewrite reports/wave-h/runbook.md
uv run python scripts/wave_h.py --plan full --dry-run  # every v1 step (cannot certify)
uv run python scripts/wave_h.py --plan certified --dry-run  # the certified run (golden/v2)
```

`--only <step>` runs one step of the chosen plan. Under `balanced-only`, `--only exact_e2e`,
`certify` or `noise` is refused. Both plans share `reports/wave-h/state.json`, so a later
`--plan full` skips the steps that `balanced-only` finished.

`docs/wave-h-report.md` belongs to `scripts/wave_h_finalize.py`; `wave_h.py` writes its own
summary to `reports/wave-h/runbook.md` (git-ignored) and never touches the docs report.

## Episodes

Six of the seven staged `human` golden episodes (`flora-golden:golden/v1/`,
`docs/golden-manifest-v1.md`: 7 selected, 7 staged, all oracle-eligible), single view, one job:

| Episodes | Expected s | Notes |
|---|---|---|
| EgoDex `add_remove_lid/000014`, `braid_unbraid/000001`, `make_sandwich/000002` | 14.6, 16.1, 16.0 | 1080p AV1 |
| HomER (WGO) `homer_52`, `homer_53`, `homer_50` | 31.9, 44.7, 46.5 | VFR; `homer_52` has no instruction text |

That is 169.8 s of expected video (the staged files run about 5% longer; the caps price +10%).
The seventh, `homer_40` (59.6 s), is left out to keep the cost down.

**(d) does not depend on (b).** It takes its episodes from the golden selection
(`wave_h.EPISODES`, in that order) through `balanced_episodes`, and reads nothing that (b)
writes. All six must be staged: otherwise (d) refuses before starting anything, and is charged
$0. (b), which runs only in `--plan full`, uses whatever the manifest stages and says so.

One job means **one app per stage**: one cold start per pool for (d). In the certified run there
is also one H200 cold start for (b) and one for (f). The noise run (f) repeats (b)'s job: the
same config file, staged data and stage hash under a new job id, on **all six** episodes.
Fix-Cert H2 needs the noise episodes to be a subset of the oracle's and the stage hash to match.
The input content is hashed, so a strict subset would be another stage hash, which
`flora certify` refuses.

Every job is staged the same way:
- the episodes are fetched into a local `mp4_dir`, with the task text next to each video;
- the dataset is uploaded to the dedicated media Volume and named as `input.stage_to`;
- the derived-weight receipts are read from `flora-weights` and checked against
  `pins.DERIVED_PINS` (`config.receipts.check_pinned`). Their file sha256 is written into the
  generated config's `weights.derived`, which is how `flora deploy` resolves them
  (`config.receipts.resolve`). An unpinned derived weight is refused before anything starts
  (`plan_app` refuses it too, review 10 M1);
- `flora index --local` writes the receipts into a local store, and they are mirrored into the
  job store on the Volume;
- then `flora plan --index`, `flora deploy` and `flora run`.

## Step 0: build and pin TempSeg

`pins.DERIVED_PINS["tempseg-merged"].content_sha256` is `None` until merger v2 is built and
checked, and every balanced plan refuses it until then. Step 0:

1. `modal run src/flora_deploy/modal_app/convert_app.py::tempseg --step merge --force` (CPU,
   8 cores, 32 GiB; client deadline 20 min);
2. `... --step equivalence` (one L40S, 8 cores, 48 GiB; deadline 10 min). It must show identical
   inputs and greedy ids on every window, `verify_gdn` PASS and the merged files verified;
   otherwise the runbook says "do not pin";
3. `flora prepare --show-receipt`: the receipt's content digest;
4. **stop** (exit 4) and print the exact one-line edit, in `src/flora_deploy/config/pins.py`
   inside `DerivedPin(key="tempseg-merged", ...)`:

   ```
   -        content_sha256=None,
   +        content_sha256="<the digest step 0 printed>",
   ```

   The operator makes it, records the result in `docs/tempseg-merge.md`, commits, and re-runs
   `--yes-spend` from that commit.

A re-run before the pin never rebuilds a receipt it has already reported. Once the pin matches
the receipt on the Volume, step 0 is done at $0. Steps (a) and (d) check the pin first
(`require_tempseg_pin`) and refuse, before starting anything, until
`DERIVED_PINS["tempseg-merged"].content_sha256` matches the digest step 0 printed.

## What each step proves

| Step | Proves |
|---|---|
| 0 | The merger v2 TempSeg build is equivalent to the unmerged model (identical inputs and greedy ids, `verify_gdn` PASS), and its content digest is pinned in code. |
| a | On CPU: the derived weight dirs and their receipts exist, the `_extra_state` prefixes are safe for vLLM, and the EOS ids are right. |
| d | `preset: balanced` deploys and runs through the real `flora deploy` / `flora run` path on L40S (TempSeg) and H100 (Captioner), and its records, spend and per-container facts come back. The workers verify the pinned derived content (`derived_content`). The lab then shows L0 input parity and Gate A (max \|Δlogit\| ≤ 0.5 with the `dt_bias` plugin) on 6 captured requests. |

What this session does **not** prove, because it is deferred: that the balanced outputs agree
with the `exact` oracle (b, e), and how large the exact-vs-exact noise floor is (f). No
certificate, provisional or otherwise, is written in `balanced-only`.

## Steps and caps

Prices come from the dry run (`--offline`, expected durations +10%, 186.8 s of video). This is
the `balanced-only` dry run:

| Step | What | GPU | Cap | Nominal | Upper |
|---|---|---|---|---|---|
| 0 | TempSeg merger v2 build + equivalence, then stop for the pin | CPU + L40S | $0.75 | $0.23 | $0.72 |
| a | balanced preflight: derived dirs, receipts, EOS ids | CPU | $0.05 | $0.00 | $0.01 |
| d | `preset: balanced` (`allow_uncertified: true`) via `flora deploy`/`flora run`, one app, cold compile | L40S + H100 | $2.30 | $1.27 | $2.28 |
| d2 | `lab`: only against (d)'s finished job, the lab's L0 + Gate A on 6 captured requests (two cold compiles per call) | L40S, then H100 | $2.35 | $1.74 | $2.31 |
| b | exact oracle | deferred to the certified run | | | |
| e | offline provisional certification | deferred to the certified run | | | |
| f | noise run | deferred to the certified run | | | |
| | **total** | | global **$5.40** | **$3.25** | **$5.31** |

The global cap of `balanced-only` is not a fixed number: it is the sum of its steps' upper
projections, rounded up to $0.10 (`plan_global_cap`), and the dry run prints it. Each step's cap
is its own upper projection rounded up to 5 cents, so no step is refused by its own cap.

`--plan full` prices every v1 step: $7.44 nominal and $11.91 upper, against its fixed cap of
$6.50. Its dry run says "NOT HONEST AT $6.50". It is superseded by `--plan certified` (below),
which is the only plan that can make a certificate come out `certified`.

`wave_h_finalize.py` needs at least 2 job samples and 2 cold starts per (preset, kind) to write a
measured rate. One job per stage gives one of each, so finalize **refuses the rate update** after
this session (it still writes the report).

### The lab step (d2) and `--only lab`

Until 2026-10-22 the lab ran inside (d). That run's deployed job finished (6 episodes, 72
segments, all captioned), but the lab gave no evidence:
- **TempSeg lab: zero requests.** The child opened camera files under
  `/golden/golden/v1/golden/v1/...`: the manifest's `path` is already Volume-relative
  (`golden/v1/<pool>/...`) and the lab joined it to `<mount>/golden/v1`. All 6 episodes raised
  `FileNotFoundError`, which was recorded per episode, and the call still returned without an
  error (`l0: []`, Gate A over 0 requests). The episode selection was right: (d)'s 6 ids.
- **Captioner lab: cut off.** Its 600 s client deadline could not hold two cold engine builds
  (about 325 s each on H100) plus the work. It was cancelled after about $0.85.

Now:
- The lab is its own step, `lab` (d2). `wave_h.py --yes-spend --only lab` retries it against
  (d)'s finished job without redoing the job. A state where (d) failed only because of its
  inline lab counts (d) as done.
- The lab runs (d)'s episodes. The Captioner captions **(d)'s segment records**: the exported
  JSONL, or the job store. It no longer uses the TempSeg lab's output.
- **Preconditions before any container** (`lab_preconditions`):
  - the episodes are (d)'s and staged;
  - every camera file exists on `flora-golden` at the path the lab child will open;
  - (d)'s segments cover every episode.

  If one fails, no container starts and the step is charged $0.
- **Fail loudly.** A baseline that runs no episode or sends zero requests returns `error` and
  `no_evidence`, and the plugin engine is not built. Wave H then skips the Captioner (it reads
  the same inputs). A kind with no L0 rows or no Gate A requests is reported as "produced no
  L0 / Gate A evidence". The step is charged only the wall time of the calls it started.
- **Reuse.** A kind's saved result (`reports/wave-h/balanced-<kind>.json`) is kept, not re-run,
  when it has evidence: L0 rows, Gate A requests and no error, for the same run id and (d)'s
  episodes. The dry run prices only the kinds that still run. The global cap stays derived from
  the whole step. Each kind's summary is saved to the state as soon as it returns, and a lab
  call that raises is recorded as that kind's error.
- **Child args** go to the engine child through a JSON file. Inline argv broke Linux's 128 KiB
  per-argument limit once the deployed segments were included (2026-10-22 Captioner).
- **Deadlines** cover two cold compiles: overhead + baseline build + plugin build + work, × 1.25.
  That is TempSeg 769 s and Captioner 1131 s, under the 1200 s container timeout. A call the
  budget left cannot carry to its expected length is not started.

### Cold compile

The image smoke found that `flora-cache` holds **no compile cache** for `balanced/tempseg` or
`balanced/captioner` at the current image key. The first L40S and H100 engine starts in (d)
therefore compile from scratch (torch.compile, Triton, CUDA graphs). The balanced startup row of
`planner/rates.json` are now **measured** with a cold cache (segment 300.5 s, caption 325 s,
`compile_cache: cold`). They are taken as they are. Only a startup that is not a cold
measurement (a projection, a placeholder or a warm measurement) is scaled by the Rewarder's
cold/warm `@modal.enter` ratio, 318 s / 229 s ≈ 1.39 (`cold_startup_s`; `docs/rewarder-smoke.md`,
`docs/cold-start.md`). The dry run says which applies.

**Write-back.** The deployed balanced workers mount `flora-cache` **read-only**
(`FLORA_CACHE_MODE=ro`; a local copy, never committed), so the deployed job leaves no cache
behind, and nothing can flow from a deployed run to the lab. The lab calls (d2) are the
writers: each writes `flora-cache:/balanced/<engine>` (and `<engine>-dtbias` for Gate A's plugin
side) and commits it after each engine. Each result records whether its cache was warm
(`compile_cache`). The deployed workers seed from the same `balanced/<engine>` directory. Whether
their vLLM config hash matches the lab's has not been measured: the lab's engine adds
`logprobs_mode: raw_logits` and `max_logprobs: 20`. Triton and CUDA kernel caches are
content-addressed either way. After 2026-10-22, `flora-cache` holds finished torch.compile
entries for `balanced/tempseg`, `balanced/captioner` and `balanced/tempseg-dtbias`.
`balanced/captioner-dtbias` was cut off mid-compile.

Projections are container-seconds × list $/h (GPU + cores + memory, as pinned in `jobs.budget`).
Container-seconds come from `planner/rates.json`: startup + video-s × container-s per video-s, plus
30 s from the last receipt to `modal app stop`. "Upper" adds:
- the table's upper quantiles;
- **240 s client slack per app** (the `flora run` timeout margin, then cancel + `modal app stop`);
- every cold start (for (d), a cold compile);
- the segment pool's scale-down tail while captions run;
- each lab call's two-cold-compile deadline plus 60 s, and step 0's deadlines plus 60 s.

## Stop rules

The state, the billing reads, the stop rules and the step loop live in `scripts/_runbook.py`.
`wave_h.py`, `scale_test.py` and `rubric_smoke.py` all use it.

- **One budget rule** (`budget_for`). An attempt of a step may spend
  `min(attempt cap, global cap − spent so far − reserve)`. The step's cap is a **per-attempt**
  ceiling: earlier attempts count against the global cap, not against the cap. The reserve is
  the upper projections of the plan's later steps that are not done yet, each at most its own
  cap. The global cap holds that much back for them. The stop rule before a step, the
  attempt's budget and its in-step deadlines all use this one rule. Every message names the
  term that binds: `attempt cap`, `global left` or `reserve for <steps>`.
- **One decision** (`decide`). Before a step the runbook takes one of four actions:
  - `run` (with that budget);
  - `done` (skipped);
  - `stop`: the upper projection does not fit. The runbook exits 3 and records why, and when
    the global term binds it prints the `--global-cap` that would fit. Raising the cap is the
    user's decision.
  - `wait`: the step only fails to fit because held attempts are charged more than they
    tracked. It fits once billing covers them, and the runbook prints when, in UTC.
  The dry run's `rule` column, `--status` and `--yes-spend` all take this same decision.
- **During a GPU step:** the affordable wall time is (budget − slack × $/s) ÷ $/s, so the
  deadline plus the slack never exceeds the budget. It is enforced three ways:
  - as `flora run --budget` (the job's own budget gate);
  - as a client-side deadline that cancels the job;
  - as the timeout of each lab and conversion call.

  A lab call that the budget cannot carry to its expected length is a stop rule raised before
  the lab app starts, not a skip inside a running app. Every app is stopped once its job ends:
  a deployed app by its name, an ephemeral app by its id (`modal app stop <name>` only finds
  deployed apps).
- **After a step:** the attempt's spend is checked against its cap, and the cumulative figure
  against the global cap.
- **The lab (d2) runs only against (d)'s `done` job.** If the job failed or was cut short,
  (d) ends `failed` with `lab: not run: ...`, and the lab step refuses to run before any app
  starts, so it spends nothing (review 10 L2).
- **Step 0** ends with exit 4 (`needs_pin`) until the pin is set; see above.

## Spend accounting

- **Every attempt counts.** Each run of a step is kept, and a retry adds to the earlier attempts
  instead of overwriting them.
- **Kills leave a record.** While a step runs, a heartbeat stores elapsed × the step's peak $/s
  every 30 s. A process killed mid-step therefore leaves its spend behind. The next
  `--yes-spend` marks the attempt `killed`, stops its apps **by id** (its recorded ids, or the
  live apps `modal app list --json` shows under its names) and retries the step.
- **Every attempt has its own app.** Each run is named `flora-wh-<run>-<step>-a<n>` (with
  `-merge` / `-equivalence` for step 0's two apps), through `App.run(name=...)` or `modal run
  -n`. Billing reports cost per app name, so each attempt's spend is its own, and the app's id
  is recorded when it starts. The deployed (d) app keeps the factory's name; it belongs to (d)
  alone.
- **Billing has a watermark.** Modal's hourly billing rows carry `interval_start`. The latest
  billed hour's end is the watermark (`billed_through`): billing is complete up to it. A state
  from before the watermark falls back to the old rule: billing complete an hour before its
  last read.
- **Unknown spend is bounded when the end is known.** A killed attempt, or one whose
  `flora report` failed, is held until the watermark passes its end. Until then it is charged:
  - when its stop was confirmed (the `with app.run()` block exited, or a stop by id returned
    0): its peak $/s over the time it ran, plus its stop slack (60 s for lab and conversion
    calls, 240 s for a deployed job);
  - when its containers' hard timeout is known (the lab: two 20-minute calls): the same up to
    that timeout;
  - otherwise its whole budget.

  The charge is never more than the budget. Each hold prints when it should clear: the end of
  its billing hour plus 15 minutes, in UTC. Today's crashed lab attempt (tracked $0.72) is
  charged about $0.84 this way, not its $2.35 budget. An attempt that failed before it noted
  any app (a refusal on the CPU, such as an unpinned weight) started nothing and is charged $0
  (review 10 L1).
- **Billing is re-read before every step and after it, for every app seen so far.** Once settled,
  an attempt is charged the larger of tracked and its own apps' billed cost over the hours it
  ran. Each billed hour of an app goes to the attempt with that app that started last. So an
  app two attempts share (scale test's deploy and run, a legacy shared name) is reconciled
  separately, and the later attempt carries the shared hour. A step's billed delta still covers
  only the apps no other step used.
- **The global figure** is max(Σ per step max(charged attempts, billed delta), the global billed
  delta).
- **Read-only previews.** `--dry-run` and `--status` decide on a copy of the state, and the state
  file stays byte-identical. Before deciding they do what a run does first: read billing
  (except with `--offline`), treat a running attempt whose heartbeat is older than 90 s as
  killed, and take its apps as stopped. `--status` shows:
  - the spend;
  - the held attempts and when they clear;
  - the cap in force, and who approved a `--global-cap` and when;
  - each step's latest error (notes are kept per attempt, so old errors do not pile up);
  - the next decision, with its command.

## State and report

- **State:** `reports/wave-h/state.json`, which is git-ignored because it names private episodes.
  A finished step is never re-run. A stopped, failed or killed step is retried on the next
  `--yes-spend`; `--only <step>` runs one step. Per job it holds what `wave_h_finalize.py`
  reads: `state`, `spend`, `video_s`, `cfg`, `job_id`, `records` (the JSONL paths `--fetch`
  exports to), and per container `kind`, `gpu`, `startup_s`, `busy_s` and `seconds`. Those come
  from `flora report` (`sdk.Job.report`, `jobs/containers.py`: attempt markers, shard receipts
  and call ledgers), cut at the app stop. Step (d) also records the derived content digests its
  workers verified (`derived_content`), which finalize certifies with.
- **Report:** `reports/wave-h/runbook.md` (`--report`, and after every run). Numbers only.
  `docs/wave-h-report.md` is `wave_h_finalize.py`'s. Captions, tasks and segment rows stay in
  `reports/wave-h/`.
- **Provisional certificate** (`--plan full` only; `balanced-only` writes none, and
  `--plan certified` uses the real `flora certify` instead):
  `reports/wave-h/certificate-provisional.json`. Its checks:
  - L0 ids and grids are equal on every captured request;
  - Gate A max |Δlogit| ≤ 0.5;
  - boundary MAE, segment IoU and caption token Jaccard of the deployed balanced records
    against `exact` are within 2× the measured exact-vs-exact noise, or within absolute bounds
    before step f has run.

  With 6 noise episodes (at least 5) but 6 candidates (fewer than 10 per pool), H-lite is
  **provisional by design** (`docs/certification.md`).

## Before a paid run

Run the preflight first. It only reads, starts nothing and says **GO** (exit 0) or **NO-GO**
(exit 1, usage error 2). Every check prints a one-line fix:

```
uv run python scripts/cert_preflight.py            # human (default)
uv run python scripts/cert_preflight.py --json     # the same, as JSON
```

| Check | GO when |
|---|---|
| git | the tree is clean, HEAD is `origin/main`, and every CI job for that SHA succeeded (a warning without `gh`) |
| plan | the certified plan's next step **runs** under the runbook's own decision (`--status --json`, else `--status` / `--dry-run`); it prints the caps, the upper total vs the global cap, and any held attempt |
| episodes | every golden/v2 `human` episode (`--embodiment yam`: golden/v3-yam) is on `flora-golden` with the manifest's file sizes |
| weights | `DERIVED_PINS` match the receipts on `flora-weights` (`flora prepare --show-receipt`) |
| images | the image locks are current (`lock_images.py --check`; until that mode exists, only the `.in` inputs are compared, as a warning) |
| apps | no `flora*` Modal app is running (other apps are ignored, never touched) |
| lab | the lab runs, or its saved result's engine-settings hash equals the current one (no hash yet: a warning, "will re-run") |
| freeze | `cert_freeze.py --check` and `tests/unit/test_certificate_drift.py` pass |
| budget | billed spend plus the plan's upper total leaves at least the margin (default $2) of the $50 project budget; it prints what remains |

A NO-GO costs nothing. A held attempt ("wait") clears once billing covers it; re-run the
preflight then rather than starting the run early.

## The certified run (`--plan certified`)

This run makes `balanced` **certified** for the `human` embodiment, so human data no longer
needs `allow_uncertified: true`. It is prepared and priced, **not run**: it spends GPU money and
waits for the user's go-ahead and cap.

```
uv run python scripts/golden_stage.py --version golden/v2 --verify-only  # golden/v2 is staged
uv run python scripts/wave_h.py --plan certified --dry-run                # plan, prices, caps
uv run python scripts/wave_h.py --plan certified --yes-spend              # b -> f -> d -> d2 -> e
uv run python scripts/wave_h.py --plan certified --yes-spend --only noise # one step (resumes)
```

### What it needs, and how each need is met

`flora certify` gives `certified` only when all of these hold (`docs/certification.md`, H4):

| Requirement | How the plan meets it |
|---|---|
| at least 10 candidate episodes of the pool | 12 golden/v2 `human` episodes (`docs/golden-manifest-v2.md`): 10 plus 2 spares, all run. (b) and (d) refuse before any app when fewer than 10 are staged |
| every oracle episode labelled (coverage 1.0) | (d) runs exactly (b)'s episodes |
| oracle and candidate uids pair | **every job stages one data dir**, `cert-human`. Episode uids are `<data dir>--<episode id>`, and `--plan full` staged one dir per step, so its uids never paired and every certificate it could make had coverage 0 |
| a noise run of at least 5 of the oracle's episodes, same stage hash, key, GPU and determinism | (f) re-indexes **(b)'s own config and data dir** with `flora index --limit 5`. The stage hash covers the config and the whole input dir, not the index, so it is (b)'s. The five noise episodes (`noise_subset` in the selection) are named `n-<id>` and the rest `x-<id>`, so they sort first. Before any app starts, (f) checks that the index holds exactly those five (count and video seconds). Same H200, `nvidia` mode, a new job id |
| L0 passing for `segment` and `caption` | the lab's JSON (d2). It is **skipped when already done**: `balanced-<kind>.json` with L0 rows that all pass, Gate A requests, no error, and the pool's GPU (`lab_already_done`). L0 is per model kind, not per episode. Otherwise (d2) runs against (d)'s job on the golden/v1 six (its deadlines were measured on them) |
| the candidate key equals this checkout's | (e) certifies with the derived content digests (d)'s workers verified, from the commit that ran (d) |

### Steps, in order

| Step | What | GPU | Stop rule (cap) | Nominal | Upper |
|---|---|---|---|---|---|
| b | exact oracle, 12 episodes, 235.9 s, `nvidia` mode, one app | H200 | $4.00 | $2.59 | $3.97 |
| f | noise: 5 of (b)'s episodes, 84.4 s, (b)'s config and data dir, new job id | H200 | $1.95 | $1.06 | $1.90 |
| d | `balanced` via the real `flora deploy` / `flora run`, the 12 episodes | L40S + H100 | $2.40 | $1.35 | $2.39 |
| d2 | lab L0 + Gate A, **skipped at $0 when already done** | L40S, then H100 | $2.35 (or $0) | $1.74 (or $0) | $2.31 (or $0) |
| e | `flora certify` per stage, then `scripts/wave_h_finalize.py --write` | CPU | $0 | $0 | $0 |
| | **total, lab runs** | | global **$10.60** | **$6.75** | **$10.58** |
| | **total, lab already done** | | global **$8.30** | **$5.00** | **$8.27** |

The dry run on the staged manifest (`--plan certified --dry-run`) prints these numbers.

- **Caps.** Each step's cap is its own upper projection rounded up to $0.05 (`plan_caps`), and
  the global cap is the sum of the upper projections rounded up to $0.10 (`plan_global_cap`).
  Both are derived from the measured rates and printed by the dry run. `--global-cap` can only
  raise the global cap. The in-step stop rules are the same as for the other plans: a GPU
  deadline from the step's cap left, `flora run --budget`, and `modal app stop`.
- **Resumable.** The plan has its own state, `reports/wave-h/certified/state.json`, and its own
  work dir and records. It never inherits a v1 job. A done step is skipped, and a failed or
  killed step is retried. The state records `plan: certified`, and a v1 plan refuses to run on
  it (and the other way round).
- **(e)** exports the three jobs' records (`wave_h_finalize.fetch_records`, a Volume read). It
  then runs `flora certify` per stage (`wave_h_finalize.certify_all`) with the oracle, the
  noise run, the L0 JSON and the derived content digests. The step is `done` only when
  **both** `segment` and `caption` come out `certified`. Only then does it run
  `wave_h_finalize.py --state reports/wave-h/certified/state.json --records
  reports/wave-h/certified/records --write`, which writes the rates, the certificates in
  `config/certificates/`, the report and the docs blocks; the operator commits them. A
  `provisional` or `fail` result writes nothing and names what is missing.

### Assumptions in the prices

- **Exact rate:** the measured 6.4 container-s per video-s ±35% (H200, default mode, c=4,
  P1a-5s). It was measured on a 1-/3-view mix in a 160 GiB container. Here all 12 episodes are
  single view, so it is probably pessimistic.
- **Exact cold start: measured.** It is 103 s @enter ±100% per job (S1; P1a-5s measured 78 s).
  (b) and (f) are each a cold start, because the oracle's app is stopped before (f) deploys it
  again. Image pull and scheduling are not in the measured startup; the +100% and the 240 s of
  client slack per app cover them.
- **Balanced:** the measured n=1 rates (segment 0.367 on L40S, caption 0.906 on H100, ±50%) and
  the measured cold startups (300 s and 325 s, `compile_cache: cold`). A warm compile cache
  only makes (d) cheaper.
- **Lab:** two cold compiles per call, priced as in "The lab step (d2)". It costs $0 when the
  lab is already done.

### Reusing step (d)'s balanced records: not possible

On this commit, step (d)'s records still match the checkout: their `cert_key` equals
`key_document(...)` and their stage hashes (`17bd065f…` segment, `d1e5c419…` caption) equal
what the same config hashes to now. They still cannot be the certified run's candidate:
- **Coverage.** They cover 6 episodes, and `flora certify` needs all of the oracle's ≥ 10.
- **Uniformity.** The candidate set must share one stage hash, and the stage hash covers the
  input dir's content. A second balanced job on only the 6 new episodes would therefore have
  another stage hash, and certify refuses the mixed set.
- **Uids.** They are `balanced-human--…`, while the certified oracle's are `cert-human--…`.
- **Key.** `modal_app/balanced_app.py` is on the `segment` / `caption` certificate logic digest
  (`certkey.cert_modules`), and it is being changed. Any code change there changes the key, and
  certify (H1) then refuses records made before it.

So **(d) is rerun on the 12 episodes.** It is about $1.35 nominal, and is priced above.

### Other embodiments (`--embodiment`)

Certificates are per embodiment, so the human run does not cover robot data. `--plan certified
--embodiment yam` runs the same steps on golden/v3-yam (`docs/golden-manifest-v3-yam.md`: 12
three-view episodes, 220.2 s, noise subset 5 episodes, 72.2 s). `--embodiment human` is the
default, and its dry run is byte-identical to the one before the flag existed.

```
uv run python scripts/golden_stage.py --version golden/v3-yam --verify-only
uv run python scripts/wave_h.py --plan certified --embodiment yam --dry-run
```

| | human (default) | yam |
|---|---|---|
| golden set | golden/v2 | golden/v3-yam |
| data dir (uids `<dir>--<id>`) | `cert-human` | `cert-yam` |
| state, work dir, records | `reports/wave-h/certified/` | `reports/wave-h/certified-yam/` |
| (b) oracle, nominal / upper | $2.59 / $3.97 | $3.11 / $4.67 |
| (f) noise | $1.06 / $1.90 | $1.16 / $2.04 |
| (d) balanced | $1.35 / $2.39 | $1.53 / $2.68 |
| total, lab already done (global cap) | $5.00 / $8.27 ($8.30) | $5.81 / $9.38 ($9.40) |

- **Separate runs.** Each embodiment has its own state file, so the two runs never collide and
  resume independently. `--state` pointing at the other embodiment's state is refused. Caps are
  derived from each embodiment's own projections.
- **Three-view prices.** The balanced rates were measured on 1-view human episodes. For yam,
  the share of the work that is generation (lab `generate_s` over (d)'s busy seconds: segment
  50.4 / 65.4 s, caption 70.5 / 161.5 s) is scaled by the per-request prompt ratio, and the
  rest stays at 1x (`three_view_scale`). The prompt ratio is 3-view (ALOHA, NVIDIA's own
  requests, `docs/exact-concurrency.md`) over 1-view (the balanced lab): TempSeg 5,493 / 1,818
  tokens and Captioner 11,145 / 5,663. That gives **segment x2.56 and caption x1.42**. Exact
  segcap gets x1.30: the same rule on NVIDIA's call mix, applied to the 1-view half of its
  measured 1-/3-view mix. The rule assumes yam has as many requests per video second as human
  (ALOHA had fewer). It keeps decode and frame preparation at 1x, although 11 of the 12
  episodes are three 640-680x480 views, about 0.45x the pixels of one 1080p view.
- **The 1-view lab prompts are not short.** The Captioner lab's 4.4k-7k-token requests (2-3
  videos each) were human single-view captions: NVIDIA's single-view caption prompt carries
  several clips. Only TempSeg is ~1.8k for 1-view.
- **Lab (d2).** L0 is per model kind, so the human lab evidence is reused ($0), as for human.
  It does not exercise three-view balanced requests; Gate C covered three-view rendering
  against vLLM on golden yam media. If the evidence is missing, the lab runs on yam's first six
  episodes, priced at the human lab's measured times.
- **(e)** runs `flora certify` but not `wave_h_finalize.py --write`, which writes the human
  rates, report and docs. The certificates stay in the work dir until finalize is
  embodiment-aware.
- **dex, tianji, yam32** are refused: they have no golden certification set yet
  (`docs/golden-manifest-v3-yam.md` lists the pinned data and rough prices). Their users need
  `allow_uncertified: true`.

### Tested offline

`tests/unit/test_wave_h_certified.py` runs the certified plan's (e) on fake records shaped like
the real ones:
- the real uids (`cert-human--n-…` / `x-…`);
- the H200 / L40S / H100 GPU names;
- the `nvidia` determinism in the oracle's key;
- a real-format balanced config and lab JSON.

It drives them through `wave_h_finalize.certify_all` (the `flora certify` CLI). When the numbers
agree, both stages come out **`certified`** (not provisional), with coverage 1.0 and 5 noise
episodes, and `wave_h_finalize.py --write` is called. With 4 noise episodes they come out
`provisional` (`too_few_noise_samples`), and a 0.6 s boundary shift is a `fail`; neither writes
anything. The test also checks:
- that the noise subset is exactly what `--limit 5` indexes;
- that the oracle's and balanced's uids pair;
- the caps and the "lab already done" skip.

## What H0 proves first

H0 (`scripts/h0_rehearsal.py`) runs the exact, reward and balanced deployments on its own 6
episodes (2 human, 4 yam / single-view robot; a wider rehearsal than this session's scope) with
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
