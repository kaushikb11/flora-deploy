# Checking a certificate before it is published

Reader: the maintainer or reviewer who signs off the first real certificate. This page says what
to run after the certified run, how to read the output, and what blocks publishing (the README
flip and the push).

`scripts/cert_check.py` verifies a certificate **independently**. It does not import
`validate/certify.py`. It recomputes every number from the raw records, using the definitions in
[`certification.md`](certification.md) and its own constants, so a bug in the generator shows up
as a mismatch instead of being repeated. It writes nothing, starts no container and calls no
network service.

## When and what to run

Run it once both of these have finished:
1. `scripts/wave_h.py --plan certified` ([`wave-h-runbook.md`](wave-h-runbook.md), "The
   certified run");
2. `scripts/wave_h_finalize.py --write`. This writes the certificates and their freeze locks
   into `src/flora_deploy/config/certificates/`, `planner/rates.json`, `docs/wave-h-report.md`
   and the README marker blocks.

Then, from the repository root, before you commit the README flip or push:

```bash
uv run python scripts/fetch_nv_code.py          # once: the NVIDIA code the leakage check needs
uv run python scripts/cert_check.py             # the PASS/FAIL table; exit 0 only if all pass
uv run python scripts/cert_check.py --json      # the same rows, for a review comment or CI
```

The defaults are the certified run's paths:

| Input | Default | Option |
|---|---|---|
| records | `reports/wave-h/certified/records/{exact,noise,balanced,witness}/human/{segment,caption}.jsonl` | `--run-root`, `--records` |
| run state | `reports/wave-h/certified/state.json` | `--state` |
| lab L0 / Gate A | `reports/wave-h/balanced-{tempseg,captioner}.json` | `--lab-dir` |
| certificates + locks | `src/flora_deploy/config/certificates/` (+ `freeze/`) | `--cert-dir` |
| report | `docs/wave-h-report.md` | `--report` |
| NVIDIA code (leakage) | the `scripts/fetch_nv_code.py` cache | `--nv-root` |

It checks the deciding certificate (the newest `generated_at`) of each stage (`segment`,
`caption`) for `balanced` on `human` (`--stage`, `--embodiment`, `--job`). With the committed
certificate directory, it also runs `tests/unit/test_certificate_drift.py` (about a minute).

## Reading the output

One row per finding: `| check | item | result | detail |`, then `Overall: PASS|FAIL (n pass, m
fail)`. Every row is PASS or FAIL; there is no "skipped". A check that cannot run (a missing
file, an unknown generator version, a crash) is a FAIL that names the reason.

| Check | What PASS means | A FAIL usually means |
|---|---|---|
| `0 inputs` | the records, state and certificate were found | wrong paths, or finalize did not write a certificate for the stage |
| `1 metrics` | every mean, `n`, `count_matched`, `missing`, `unpaired` and `coverage` in the certificate equals the value recomputed from the records (oracle vs balanced, oracle vs noise, oracle vs cold witness), within 2e-6 | the records changed after `flora certify` ran, or the generator computes a metric differently from the definition. The detail names the metric and both values |
| `2 pass rule` | the knobs are within policy; the recorded floors, caps, `over` basis and thresholds are the policy's; every check, every reason and the verdict recompute; **count-flip tolerance**: the `count_mismatch` threshold lets one episode in ten differ in subtask count (0.10; 1 in 12 passes, 2 fail) | a loosened or older rule (generator version 2 has no count-flip tolerance and no caption `count_mismatch`), or a verdict that does not follow from the numbers |
| `2 pass rule` / `publishable` | the verdict is `certified` (or `provisional` with `--allow-provisional`) | the run did not certify; the detail and the `verdict` row give the reasons |
| `3 coverage` | the certificate's hashed episode ids are exactly the oracle/balanced pairs; the noise and witness episodes are the oracle's (or are counted `unpaired`); the balanced and witness records share one stage hash and one `cert_key`, and that key is the key the current code computes; the oracle binding (stage hash, key, H200, `nvidia` determinism) holds and the noise run repeats it; the embodiment is `human` and the GPU is the pool's (`L40S` for segment, `H100!` for caption) | records from another job or another build, or code changed since the run (the detail lists the key fields that differ) |
| `4 evidence` | every engine record states its compile cache; the cold-witness job finished and all its records are `cold:forced`; the certificate's `compile_cache` block matches the records; no record preempted or lacks the count (or the certificate is `provisional` because of it); the lab's L0 and Gate A are present and its baseline engine-settings hash equals the current engine settings and the certificate's engine | no forced-cold evidence, a preempted engine, or a lab measured under other engine settings (re-run the lab) |
| `5 freeze lock` | `freeze/<id>.json` exists, names the certificate, hashes to its key, and equals the current key document and module hashes; `certfreeze.check` reports no error or warning for it; `test_certificate_drift.py` passes | the lock is missing or stale, or the code drifted after the certificate was written ([`certificate-freeze.md`](certificate-freeze.md)) |
| `6 leakage` | no near-copy of NVIDIA text (8-word shingles against the fetched NVIDIA code, as the license guard does) in the certificate, its lock or the report; none of the records' captions, ids, dataset names, source URIs or NVIDIA metadata in them; every string field of the certificate is one of ours (enums, hashes, the key's own values) | a free-text field crept into the certificate, or the report quotes a caption or an episode id. The detail names the field path or the count, never the text |
| `7 spend` | the report's per-step rows (attempts, tracked, charged, billed) and its tracked, charged, billed and total lines equal what the state gives (`scripts/_runbook.py`'s accounting); the total is the larger of charged and billed and within the cap the run was gated on | the report was written before the last billing read or a later step: re-run `wave_h_finalize.py --write` |

## What blocks publishing

Publish (flip the README, push the certificate) only when `cert_check.py` exits 0 on the
committed paths, with the NVIDIA code fetched and without `--allow-provisional`. In particular,
any of these blocks it:
- any metric, threshold, check, reason or verdict mismatch (`1`, `2`): the certificate does not
  say what the records say;
- a verdict other than `certified`. A `provisional` certificate unlocks nothing. The README must
  not call the stack certified, so its blocks must stay as finalize writes them for it;
- a key, stage hash, embodiment or GPU mismatch (`3`), or a drifted lock (`5`): the certificate
  would not match the code users run;
- missing forced-cold or lab evidence, or preemptions on a `certified` certificate (`4`);
- any leakage row (`6`): nothing that fails it may be committed;
- a spend mismatch (`7`): fix the report first, then re-run the check.

A FAIL is not fixed by editing the certificate. A hand edit breaks its content address, and
re-addressing it is exactly what `1` and `2` catch. Re-run `flora certify` / finalize from the
records, or re-run the step whose evidence is missing.

## Notes for reviewers

- **The cold witness is held to the candidate's thresholds on 6 episodes.** One subtask-count
  flip among them is 1/6 = 17%, which is above the 10% count-flip floor. So a single flip in the
  witness fails the certificate, even though the same flip among the 12 candidate episodes
  would pass. That is the generator's rule, and the check reports it as it is (`2 pass rule`,
  `cold-witness count_mismatch` among the failing checks).
- The checker knows the rules of generator versions 2 and 3. A version-2 certificate always
  fails `count-flip tolerance` and `certificate states the caches`: it predates the user's
  count-flip decision and the cache evidence.
- The tests (`tests/unit/test_cert_check.py`, fixtures in `tests/fakes/certcheck_run.py`) build
  a fake certified run whose certificates the real generator writes. The good run passes every
  row, which also cross-checks the generator against this independent reimplementation. Each
  check has a fixture that breaks what it guards.
