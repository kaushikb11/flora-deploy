# Wave H-lite runbook

Wave H-lite is the one GPU session that replaces Wave H (decision of 2026-10-22 in
`docs/archive/NEXT_EXECUTION.md`). The user then narrowed it (2026-10-22): **run the GPU
session balanced only, now.** The default plan, `--plan balanced-only`, runs three steps:
- step 0: build TempSeg with merger v2, check its equivalence, then pin it;
- (a): the balanced preflight;
- (d): `balanced` through the real `flora deploy` / `flora run` on the 6 human episodes, plus the
  lab's L0 input parity and Gate A.

The exact oracle (b), the provisional certification (e) and the noise run (f) are **deferred to
the certified run**, a later session that uses `--plan full`. The approved scope is about $2.1
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
uv run python scripts/wave_h.py --report            # rewrite reports/wave-h/runbook.md
uv run python scripts/wave_h.py --plan full --dry-run  # the later certified run: every step
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

The certified run (`--plan full`) prices every step: $6.45 nominal and $11.02 upper, against its
fixed cap of $6.50. Its dry run says "NOT HONEST AT $6.50" and names the minimum honest cap
($11.10). The cap for that session is the user's decision.

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

- **Before a step:** its **upper** projection must fit its cap and the plan's global cap minus
  the cumulative spend. Otherwise the runbook stops (exit 3) and records why.
- **During a GPU step:** the affordable wall time is (remaining step cap − slack × $/s) ÷ $/s,
  so the deadline plus the slack never exceeds the cap. It is enforced three ways:
  - as `flora run --budget` (the job's own budget gate);
  - as a client-side deadline that cancels the job;
  - as the timeout of each lab and conversion call.

  Every app is stopped (`modal app stop`) once its job ends.
- **After a step:** the step's spend is checked against its cap, and the cumulative figure
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
  `--yes-spend` marks the attempt `killed`, stops its apps and retries the step.
- **Unknown spend is charged at budget.** A killed attempt, or one whose `flora report` failed,
  is charged its whole budget until billing has been read at least an hour after it. An attempt
  that failed before it noted any app (a refusal on the CPU, such as an unpinned weight)
  started nothing and is charged $0 (review 10 L1).
- **Billing is re-read before every step and after it, for every app seen so far.** A step's
  billed delta covers only the apps no other step used. (f) reuses (b)'s app, and the lab app is
  shared by (a) and (d); those count in the global billed total only.
- **The global figure** is max(Σ per step max(charged attempts, billed delta), the global billed
  delta).

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
- **Provisional certificate** (certified run only; `balanced-only` writes none):
  `reports/wave-h/certificate-provisional.json`. Its checks:
  - L0 ids and grids are equal on every captured request;
  - Gate A max |Δlogit| ≤ 0.5;
  - boundary MAE, segment IoU and caption token Jaccard of the deployed balanced records
    against `exact` are within 2× the measured exact-vs-exact noise, or within absolute bounds
    before step f has run.

  With 6 noise episodes (at least 5) but 6 candidates (fewer than 10 per pool), H-lite is
  **provisional by design** (`docs/certification.md`).

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
