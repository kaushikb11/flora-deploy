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
1. `scripts/wave_h.py --plan recertify` for human ([`wave-h-runbook.md`](wave-h-runbook.md),
   "The re-certification run"), or `--plan certified --embodiment yam`;
2. `scripts/wave_h_finalize.py --write`. This writes the certificates and their freeze locks
   into `src/flora_deploy/config/certificates/`, `planner/rates.json`, `docs/wave-h-report.md`
   and the README marker blocks.

Then, from the repository root, before you commit the README flip or push:

```bash
uv run python scripts/fetch_nv_code.py          # once: the NVIDIA code the leakage check needs
uv run python scripts/cert_check.py             # the PASS/FAIL table; exit 0 only if all pass
uv run python scripts/cert_check.py --json      # the same rows, for a review comment or CI
# v4 caption evidence (defaults: where wave_h_finalize.py --fetch and wave_h.py put it):
#   --digests SIDE=PATH (exact, noise, balanced, witness, repeat), --oracle-segments PATH,
#   --reported-e2e PATH, --reported-repeat PATH, --families PATH
```

The defaults are the (v3) certified run's paths; for the re-certification run pass its run root
(`--run-root reports/wave-h/recertified`):

| Input | Default | Option |
|---|---|---|
| records | `reports/wave-h/certified/records/{exact,noise,balanced,witness}/human/{segment,caption}.jsonl` | `--run-root`, `--records` |
| v4 caption evidence | the pinned oracle segment JSONL, the per-side digest JSONs (`exact`, `noise`, `balanced`, `witness`, optionally `repeat`) and the end-to-end (J1) caption records, as `wave_h_finalize.py --fetch` exported them under the run root | `--run-root` |
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
| `2 pass rule` | the knobs are within policy; the recorded floors, caps, `over` basis and thresholds are the policy's; every check, every reason and the verdict recompute; **count-flip tolerance**: the `count_mismatch` threshold lets one episode in ten differ in subtask count (0.10), counted over the union of the candidate's and the witness's episodes and rounded half up (2 of 18 pass, 3 fail; without a witness 1 in 12 passes, 2 fail), and the certificate's per-group and union figures are the records' | a loosened or older rule (generator version 2 has no count-flip tolerance and no caption `count_mismatch`), or a verdict that does not follow from the numbers |
| `1 metrics` / isolated caption (v4) | the stage-isolated caption means are recomputed from the records and the digest JSONs by the checker's own code: the slots are the oracle rows with a Captioner digest, the pairs are row for row, `caption_mismatch` and `caption_edit_distance` are macro means over the episodes with a slot, and `n` and `missing` follow | the generator's slot or pairing rule differs from the definition, or the records changed |
| `1 metrics` / digest identity (v4) | the candidate's and the witness's records were captioned from the pinned oracle segment file (`provenance.segments_from` sha256 and oracle stage hash, same row count and bounds within 1e-6 s), every oracle row with a call has a digest, every candidate and witness digest equals the oracle's, and the certificate's `input_identity` (per `candidate` and `witness`: `pairs`, `equal`, `episodes`, `status: pass`, the stage hashes) is what the digests give | the harness rebuilt different Captioner inputs (the generator should have refused: `segments_not_oracle`, `oracle_digests_missing`, `inputs_not_isolated`), or `input_identity` was edited |
| `1 metrics` / isolated noise (v4) | the noise means are recomputed over the input-identical pairs only (oracle vs noise aligned by IoU, kept where both digests are equal), episodes with no such pair are left out, count-flip episodes are kept, and the noise `n` is the number of episodes with at least one pair | the noise basis differs from the definition (for example equal bounds instead of equal digests) |
| `2 pass rule` / reported block (v4) | `metrics.reported` is recomputed (the e2e block with the v3 definitions, from the J1 records) and it says `reported_only: true`; no check, reason or threshold names a reported metric; the verdict recomputes the same with the block removed | a reported-only metric leaked into the gate |
| `2 pass rule` / v4 policy | the generator version is `4`; a caption certificate has `scoring: {caption: stage_isolated}`, `segments_from`, `input_identity` `pass` for the candidate and the witness, and no `count_mismatch` check (no union count check for caption); the knobs, floors and thresholds are the unchanged policy's (caption mismatch 0.075 at noise <= 0.05) | an older generator version (a v3 certificate is retired and never publishable), a missing `scoring`, or a caption certificate scored end to end |
| `2 pass rule` / `publishable` | the verdict is `certified` (or `provisional` with `--allow-provisional`) | the run did not certify; the detail and the `verdict` row give the reasons |
| `3 coverage` | the certificate's hashed episode ids are exactly the oracle/balanced pairs; the noise and witness episodes are the oracle's (or are counted `unpaired`); the balanced and witness records share one stage hash and one `cert_key`, and that key is the key the current code computes; the oracle binding (stage hash, key, H200, `nvidia` determinism) holds and the noise run repeats it; the embodiment is `human` and the GPU is the pool's (`L40S` for segment, `H100!` for caption) | records from another job or another build, or code changed since the run (the detail lists the key fields that differ) |
| `4 evidence` | every engine record states its compile cache; the cold-witness job finished and all its records are `cold:forced`; the certificate's `compile_cache` block matches the records; no record preempted or lacks the count (or the certificate is `provisional` because of it); the lab's L0 and Gate A are present and its baseline engine-settings hash equals the current engine settings and the certificate's engine | no forced-cold evidence, a preempted engine, or a lab measured under other engine settings (re-run the lab) |
| `5 freeze lock` | `freeze/<id>.json` exists, names the certificate, hashes to its key, and equals the current key document and module hashes; `certfreeze.check` reports no error or warning for it; `test_certificate_drift.py` passes | the lock is missing or stale, or the code drifted after the certificate was written ([`certificate-freeze.md`](certificate-freeze.md)) |
| `6 leakage` | no near-copy of NVIDIA text (8-word shingles against the fetched NVIDIA code, as the license guard does) in the certificate, its lock or the report; none of the records' captions, ids, dataset names, source URIs or NVIDIA metadata in them; every string field of the certificate is one of ours (enums, hashes, the key's own values). The records' ids (uids, the episode id inside each `<data dir>--<episode id>`, datasets) are looked for exactly before the hash allowance, so an all-hex or all-digit episode id is still found; a hash-shaped one matches only as a whole token, never inside a longer hash | a free-text field crept into the certificate, or the report quotes a caption or an episode id. The detail names the field path or the count, never the text |
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

- **Count flips are counted over the union of the candidate's and the witness's episodes**
  (`docs/certification.md`, "Pass rule"): `flips <= 0.10 x n` rounded half up, so 2 of the
  certified run's 18 pass and 3 fail. The checker recomputes the per-group and union figures
  from the records (`count-flip tolerance` names both) and compares them with
  `metrics.count_flips` and the `union count_mismatch` check. The witness's other means are
  still checked on their own (`cold-witness <metric>` among the failing checks).
- The checker knows the rules of generator versions 2, 3 and 4. A version-2 certificate always
  fails `count-flip tolerance` and `certificate states the caches`: it predates the user's
  count-flip decision and the cache evidence. Only version 4 is publishable: version 3 is
  retired (`certkey.GENERATOR_VERSIONS = ("4",)`), so a v3 certificate fails `v4 policy`.
- **Version 4 caption certificates are stage-isolated** (`docs/certification.md`, "Caption").
  The checker reimplements the isolated metrics, the digest identity, the isolated noise, the
  reported-block exclusion and the v4 policy rows itself; it does not import `certify.py`. The
  count-flip rows apply to `segment` only. The digest JSONs hold hex digests only, no text.
- The tests (`tests/unit/test_cert_check.py`, fixtures in `tests/fakes/certcheck_run.py`) build
  a fake certified run whose certificates the real generator writes. The good run passes every
  row, which also cross-checks the generator against this independent reimplementation. Each
  check has a fixture that breaks what it guards.
