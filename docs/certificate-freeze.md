# Certificate freeze: what a certificate pins, and how to change it

For contributors. Once an accuracy certificate for a stack is committed under
`src/flora_deploy/config/certificates/`, the code it covers is **frozen**. Changing that code
does not break anything you can see. The certificate just stops matching, and users of that
stack see the stage as `not certified` (and the option as `partly certified` or `uncertified`)
in `flora plan` and `flora deploy`, with `certified: false` in its records. This page says what
is frozen, how to tell whether your change touches it, and what to do when it does. How
certificates are computed and matched, and what partial certification looks like to users, is
in [`certification.md`](certification.md).

## What is frozen

A certificate matches only while the **certificate key** of the code that would run equals the
key it was issued for (`validate/certkey.py`, `key_document` -> `engine_spec_hash`). For one
(preset, shard kind, GPU) the key covers:

| Part of the key | What changes it |
|---|---|
| `logic_digest` | a **code** edit in any module of the kind's logic group (`config/logic.py`: `common` + the kind's group, minus `config/logic.py`, `config/hashing.py`, `validate/certkey.py`). Comments, docstrings, blank lines and trailing whitespace are ignored. Adding a module to the group changes it too |
| `logic_version` | bumping `LOGIC_VERSION` (`flora_deploy/__init__.py`) |
| `engine` | the preset's `EngineIdentity` (name, versions, image, patches, engine args, sampling, mm kwargs) and the image fingerprint (`modal_app/images.py`: base, lock files, install steps, env) |
| `vllm_engine` | `segment` / `caption`: `engines/offline_llm.EngineSpec.hashed()` on that GPU: every engine kwarg, `GPU_LIMITS` for (model, GPU), the `dt_bias` plugin flag |
| `nvidia_pins` | `config/pins.py`: an NVIDIA pin's revision, plan digest, expected files or code digests |
| `decode` | the `av`, `pillow`, `numpy` runtime pins and the image arch |
| `derived_weights` | the pinned content digest of a derived weight (`pins.DERIVED_PINS`), or the preset needing another one |
| `modes` | the kind's output-affecting modes (`config.hashing.kind_modes`): a new or renamed mode |

Matching also refuses a certificate that no longer meets `certkey.POLICY` (thresholds,
`GENERATOR_VERSIONS`), one for a preset, kind, GPU or embodiment the code no longer has, and a
file whose `id` is not its content address. The guard checks all of these.

`exact` is the reference by construction and needs no certificate, so nothing of it is frozen.
Rewards (`reward.*`) under `balanced` run exactly `exact`'s engine, image, pins, modes and
logic, so they are the reference by identity (`certkey.reference_identical`) and need no
certificate. Nothing of them is frozen. A change that makes a reward key diverge from `exact`'s
turns them into an uncertified stack that needs its own certificate (see
[`certification.md`](certification.md#reference-by-identity)).

## How to tell whether a change affects a certificate

- **CI tells you.** The first gate after install in the `test` job, `Gate: certificate drift
  (committed certificates still match the code)`, recomputes the
  key of every committed certificate (`tests/unit/test_certificate_drift.py`,
  `validate/certfreeze.py`). A drifted `certified` certificate fails the PR with a diff of the
  key documents: which modules changed, which engine kwargs, pins or derived digests changed,
  and the choice below. The same report is in the step summary and as `::error` annotations.
- **Locally**, before you push:

  ```bash
  uv run python scripts/cert_freeze.py --check      # drift report; exit 1 on an error
  uv run python scripts/cert_freeze.py              # what is frozen, per (preset, kind, GPU)
  ```

- **By hand**: if your diff touches a module listed under the kind in the listing below, or
  `config/pins.py`, `engines/offline_llm.py` (`GPU_LIMITS`, `REQUIRED`), `config/presets.py`
  (engines, derived weights), a lock file under `modal_app/locks/` or `modal_app/images.py`, it
  can.

What the guard reports, per certificate:

| Verdict of the deciding certificate | On drift |
|---|---|
| `certified` | **error**: the PR fails |
| `provisional` | warning: it unlocks nothing, so it does not block |
| `fail` (a newer `fail` revokes the key) | not checked (an `info` line) |
| superseded (a newer certificate decides the same key and pool) | not checked (an `info` line) |

Other findings: a certificate committed without its freeze lock is a `no_lock` warning, a lock
without a certificate an `orphan_lock` warning, a file not named after its id a warning, and a
certificate file that is not valid JSON or not a valid certificate an error.

The deciding certificate is the one `certkey.matching` uses: per (preset, kind, GPU, key,
embodiment pool) the newest `generated_at`, and on a tie `fail` beats `provisional` beats
`certified`. A certificate whose key matches is also run through `certkey.matching` itself, end
to end. With no certificate committed, the guard passes.

### Freeze locks

The certificate body records only part of its key (pin revisions, not pin digests; no decode
pins; one logic digest for all modules). So next to every certificate, a **freeze lock**
`config/certificates/freeze/<id>.json` keeps the full key document and the sha256 of each
module's normalized source. With it, the drift report names the exact modules and fields that
changed. Without it, the report diffs what the body records and, when git history is available,
lists the kind's modules touched since the certificate was committed. Matching never reads a
lock. It is for diagnosis only.

`flora certify` writes the lock next to every certificate it writes.
`scripts/wave_h_finalize.py --write` writes the locks too, for every certificate it writes (and
any committed one whose lock is missing). If this tree can no longer compute a certificate's key
it writes nothing (exit 1). When it replaces a certificate it wrote earlier for the same preset,
stage, kind, GPU, pools and key, it removes the old one and its lock; it never removes one for an
*old* key, so the manual delete below is still needed after a re-certification.
`scripts/cert_freeze.py --lock` is only needed for certificates copied in by hand: it writes a
missing lock when this tree still computes the certificate's key, and removes locks whose
certificate is gone.

## When your change affects a certificate: revert or re-certify

The drift report ends with the choice:

1. **Revert**, if the change was not meant to change that stack's outputs: a refactor that
   changed code on the path, a pin bump that can wait. Often the change can be moved out of
   the kind's modules, or made a comment-only edit.
2. **Re-certify**, if it was meant to (a fix, a new engine setting, new weights). Re-certifying
   costs GPU time: the oracle (the `exact` reference run), the noise run and `balanced` on the
   human episodes. Maintainers decide the spend (`docs/wave-h-runbook.md`).

### Re-certification workflow

On the commit that carries the change:

```bash
# 1. the certified run: exact oracle, noise run and balanced on the golden human episodes
uv run python scripts/wave_h.py --plan certified --dry-run    # check the steps and the cap
uv run python scripts/wave_h.py --plan certified

# 2. certify (one certificate per stage; or scripts/wave_h_finalize.py --write, which runs
#    flora certify for every balanced job)
#    --witness: the cold witness's records (without it the verdict is at best provisional);
#    --generated-at: the run's own time, so a re-run reproduces the same certificate id
flora certify -c flora.yaml --exact oracle/segment.jsonl --balanced bal/segment.jsonl \
  --noise noise/segment.jsonl --witness witness/segment.jsonl \
  --l0 reports/wave-h/balanced-tempseg.json \
  --derived-content tempseg_merged=<sha256> --generated-at <run time, YYYY-MM-DDTHH:MM:SSZ>

# 3. locks, then the check must pass
uv run python scripts/cert_freeze.py --lock
uv run python scripts/cert_freeze.py --check
```

Then delete the superseded certificate (and its lock) for the old key: the guard keeps
failing on it, since it can never match again. Commit the new certificate and its lock with the
change, in one PR, so `main` never has code without a matching certificate. A certificate is
bound to the records' own provenance, so certify from the commit, image and weights that
produced them. `flora certify` refuses otherwise.

Edge cases:
- **A GPU or embodiment no longer in the presets.** A certificate for a GPU the kind can no longer
  use (unknown, too small, or no `GPU_LIMITS` entry) or for a removed embodiment pool can
  never match. Delete it, or restore what it names.
- **Several certificates for one key.** Only the deciding (newest) one is checked. Delete the
  superseded ones when convenient. A newer `fail` revokes the key, and the drift of a revoked
  key is not reported.
- **A certificate for another key in the same slot** (another determinism mode, for example) is
  live too, and is checked on its own modes.
- **A `provisional` certificate** drifts with a warning only. Re-certify it or delete it.

## Current freeze listing

Run `uv run python scripts/cert_freeze.py` for the current listing. For each (preset, kind, GPU)
it prints the key with the default template's modes and the pinned derived digests, every part
of the key with its hash, the modules of the logic digest with the sha256 of their normalized
source, and the certificate that covers it. The hashes change with every merge on these paths,
so they are not copied here.

**One certificate is committed, so one row is frozen:** `balanced` / `segment` / `L40S` for
`human` (`cert_f5d6433f….json`, lock in `certificates/freeze/`, run `10221548f1b8`). `caption`
failed on that run and has no certificate, so its row is not frozen until it is re-certified.
What the listing covers:

- `exact`: nothing (the reference by construction).
- `balanced` / `segment` / `L40S` and `balanced` / `caption` / `H100!`: the `vllm-offline`
  engine on `img_vllm`, with `max_num_seqs` 48 (TempSeg on the L40S) and 16 (Captioner on the
  H100), `max_num_batched_tokens` 16384 and `gpu_memory_utilization` 0.9; the TempSeg or
  Captioner NVIDIA pins; the `tempseg_merged` or `captioner_bf16` derived digest; and a cert
  logic digest over the 61 modules of the `common` + `vllm-video` groups (`engines/batch_policy.py`
  included). The segment row is frozen by the committed certificate; the caption row will be
  frozen by its re-certification.
- `balanced` / `reward.terminal`, `reward.dense`, `reward.rubric` / `H100!`: the
  `nvidia-vllm-offline` engine, the Rewarder pin, a logic digest over 30 modules (32 for
  rubric). These are the reference by identity, so no certificate will freeze them; the script
  still lists them as "no certificate (uncertified ...)".
