# Certificate freeze: what a certificate pins, and how to change it

For contributors. Once a Gate certificate for a stack is committed under
`src/flora_deploy/config/certificates/`, the code it covers is **frozen**. Changing that code
does not break anything you can see. The certificate just stops matching, and users of that
stack see "uncertified" (`flora plan`, `flora deploy`, records with `certified: false`). This
page says what is frozen, how to tell whether your change touches it, and what to do when it
does. How certificates are computed and matched is in [`certification.md`](certification.md).

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
Rewards (`reward.*`) under `balanced` run NVIDIA's own vLLM recipe, but they still need their
own certificate to unlock, and are frozen the same way once one exists.

## How to tell whether a change affects a certificate

- **CI tells you.** The first gate of the `test` job, `Gate: certificate drift`, recomputes the
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
| `fail` (a newer `fail` revokes the key) | not checked |
| superseded (a newer certificate decides the same key and pool) | not checked |

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
`scripts/wave_h_finalize.py --write` copies only the certificates, so run
`scripts/cert_freeze.py --lock` after it. `--lock` writes a missing lock when this tree still
computes the certificate's key, and removes locks whose certificate is gone.

## When your change affects a certificate: revert or re-certify

The drift report ends with the choice:

1. **Revert**, if the change was not meant to change that stack's outputs: a refactor that
   changed code on the path, a pin bump that can wait. Often the change can be moved out of
   the kind's modules, or made a comment-only edit.
2. **Re-certify**, if it was meant to (a fix, a new engine setting, new weights). Re-certifying
   costs GPU time: the oracle, the noise run and `balanced` on the human episodes. Maintainers
   decide the spend (`docs/wave-h-runbook.md`).

### Re-certification workflow

On the commit that carries the change:

```bash
# 1. the certified run: exact oracle, noise run and balanced on the golden human episodes
uv run python scripts/wave_h.py --plan certified --dry-run    # check the steps and the cap
uv run python scripts/wave_h.py --plan certified

# 2. certify (one certificate per stage; or scripts/wave_h_finalize.py --write, which runs
#    flora certify for every balanced job)
flora certify -c flora.yaml --exact oracle/segment.jsonl --balanced bal/segment.jsonl \
  --noise noise/segment.jsonl --l0 reports/balanced-lab/tempseg.json \
  --derived-content tempseg_merged=<sha256>

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

`uv run python scripts/cert_freeze.py` on this tree (2026-10-22, `main` + this change). For
each (preset, kind, GPU): the key with the default template's modes and the pinned derived
digests, every part of the key with its hash, the modules of the logic digest with the sha256
of their normalized source, and the certificate that covers it. **No certificate is committed
yet, so nothing is frozen.** The first `balanced` certificates from the certified run will
freeze the `segment` / `caption` rows (and the reward rows, if certified). Hashes change with
every merge on these paths until then. Re-run the script for the current values.

<details>
<summary>Full listing</summary>

```text
# Certificate freeze listing (scripts/cert_freeze.py)

`exact` is the reference by construction: nothing of it is frozen.

## balanced / segment / L40S (default pool)

- key (`engine_spec_hash`, default modes, pinned derived digests): `6a8914be2ad6d39a`
- engine: `vllm-offline` {'vllm': '0.24.0'}, image `img_vllm` fingerprint `4df6c7826a52`, patches [], engine_args `69000031fceb`, sampling `44136fa355b3`, mm_kwargs `44136fa355b3`
- vllm_engine: `vllm-0.24.0-offline` (dtbias_fp32=False), max_num_seqs=32, max_num_batched_tokens=16384, gpu_memory_utilization=0.9, engine_kwargs `cbf7a2a5023c`
- nvidia_pins: `tempseg`@114a1076f70e (`a658b05c0b79`), `tempseg_base`@c20223623576 (`03d525461ebc`), `tempseg_code`@78a0e4ac75d3 (`05df4cce31a5`)
- decode: av 16.0.1, pillow 12.3.0, numpy 2.3.5, arch x86_64
- derived_weights: `tempseg_merged` `6356ab8c1499`
- modes: {}
- logic: `0.0.1`, cert logic digest `92d64eb036f679dc` over 60 modules:
  - `accel/__init__.py` `676cbb99d188`
  - `accel/captioner_load.py` `99bda878938f`
  - `accel/determinism.py` `eb361614c047`
  - `accel/framecache.py` `0ebdc4c2209e`
  - `accel/geometry.py` `76f03e3993a1`
  - `accel/gpu_thread.py` `f6862177f058`
  - `accel/patches.py` `d412a607ead0`
  - `accel/pngmem.py` `ca433c7acf1e`
  - `accel/registry.py` `be0ffb6fb0e3`
  - `accel/state.py` `4dc66ac0a02f`
  - `accel/transport.py` `9a4a8459850d`
  - `config/schema.py` `be1e93e2d01e`
  - `engines/__init__.py` `e3b0c44298fc`
  - `engines/base.py` `de5c025e8099`
  - `engines/microbatch.py` `f6eb1697248a`
  - `engines/offline_llm.py` `2afb1db81245`
  - `engines/patches/__init__.py` `e3b0c44298fc`
  - `engines/patches/dt_bias_fp32.py` `d2f7acd405e8`
  - `engines/worker_ext.py` `ba5bbfe538cf`
  - `jobs/indexing.py` `82fc1ced0547`
  - `jobs/planning.py` `92a62110f8ac`
  - `jobs/settings.py` `caaadcd39505`
  - `jobs/types.py` `9891533d7e60`
  - `media/__init__.py` `e3b0c44298fc`
  - `media/avsafe.py` `bec4f276de2b`
  - `media/cuts.py` `c73666ca2d09`
  - `media/index.py` `d9becae9d2b3`
  - `media/rangedecode.py` `d0133ce5903c`
  - `media/safepath.py` `b8f47d754db0`
  - `media/sources/__init__.py` `30b16c9dec00`
  - `media/sources/base.py` `2a5e5f266091`
  - `media/sources/lerobot_v21.py` `20726442a33a`
  - `media/sources/lerobot_v3.py` `659c5fea74cf`
  - `media/sources/mp4_dir.py` `b4b7f9813a1a`
  - `media/types.py` `29f1608ac67f`
  - `modal_app/balanced_app.py` `dbb18ed30bba`
  - `modal_app/factory.py` `52e5826554fd`
  - `nv/__init__.py` `52bb0565cbb7`
  - `nv/budget.py` `330b1aef80c1`
  - `nv/capture.py` `ea3cffdd9cf9`
  - `nv/errors.py` `efc4f2053481`
  - `nv/linker.py` `ee6c24e1469f`
  - `nv/segment_glue.py` `518bdabee919`
  - `nv/session.py` `49fadc7f1f5d`
  - `observe/events.py` `59e290f57a71`
  - `observe/locks.py` `7ddfe2335626`
  - `observe/stdout.py` `184573a500c3`
  - `observe/timers.py` `2f4463f8f91b`
  - `prep/convert_captioner.py` `d9e0e97eacff`
  - `records.py` `1fa4bc7179ac`
  - `stages/__init__.py` `e3b0c44298fc`
  - `stages/caption.py` `276c7156d311`
  - `stages/reference.py` `5dfeedf19524`
  - `stages/segment.py` `b723e411ed8d`
  - `workers_impl/balanced.py` `73c640315191`
  - `workers_impl/caption.py` `bae68943032a`
  - `workers_impl/certstamp.py` `56ff03102855`
  - `workers_impl/reward.py` `1ed7c890d26d`
  - `workers_impl/segcap.py` `67d9cf6a4a7e`
  - `workers_impl/segment.py` `9100d4c1b579`
- covered by: **no certificate** (uncertified; nothing frozen yet)

## balanced / caption / H100! (default pool)

- key (`engine_spec_hash`, default modes, pinned derived digests): `65afdf765ae199bf`
- engine: `vllm-offline` {'vllm': '0.24.0'}, image `img_vllm` fingerprint `4df6c7826a52`, patches [], engine_args `69000031fceb`, sampling `44136fa355b3`, mm_kwargs `44136fa355b3`
- vllm_engine: `vllm-0.24.0-offline` (dtbias_fp32=False), max_num_seqs=16, max_num_batched_tokens=16384, gpu_memory_utilization=0.9, engine_kwargs `34e22299ab81`
- nvidia_pins: `captioner`@fef0bbfe1182 (`3841e5e831e0`), `captioner_code`@f736ca36925a (`9a8050017924`)
- decode: av 16.0.1, pillow 12.3.0, numpy 2.3.5, arch x86_64
- derived_weights: `captioner_bf16` `d4c1bea5dbd3`
- modes: {}
- logic: `0.0.1`, cert logic digest `92d64eb036f679dc` over 60 modules:
  - `accel/__init__.py` `676cbb99d188`
  - `accel/captioner_load.py` `99bda878938f`
  - `accel/determinism.py` `eb361614c047`
  - `accel/framecache.py` `0ebdc4c2209e`
  - `accel/geometry.py` `76f03e3993a1`
  - `accel/gpu_thread.py` `f6862177f058`
  - `accel/patches.py` `d412a607ead0`
  - `accel/pngmem.py` `ca433c7acf1e`
  - `accel/registry.py` `be0ffb6fb0e3`
  - `accel/state.py` `4dc66ac0a02f`
  - `accel/transport.py` `9a4a8459850d`
  - `config/schema.py` `be1e93e2d01e`
  - `engines/__init__.py` `e3b0c44298fc`
  - `engines/base.py` `de5c025e8099`
  - `engines/microbatch.py` `f6eb1697248a`
  - `engines/offline_llm.py` `2afb1db81245`
  - `engines/patches/__init__.py` `e3b0c44298fc`
  - `engines/patches/dt_bias_fp32.py` `d2f7acd405e8`
  - `engines/worker_ext.py` `ba5bbfe538cf`
  - `jobs/indexing.py` `82fc1ced0547`
  - `jobs/planning.py` `92a62110f8ac`
  - `jobs/settings.py` `caaadcd39505`
  - `jobs/types.py` `9891533d7e60`
  - `media/__init__.py` `e3b0c44298fc`
  - `media/avsafe.py` `bec4f276de2b`
  - `media/cuts.py` `c73666ca2d09`
  - `media/index.py` `d9becae9d2b3`
  - `media/rangedecode.py` `d0133ce5903c`
  - `media/safepath.py` `b8f47d754db0`
  - `media/sources/__init__.py` `30b16c9dec00`
  - `media/sources/base.py` `2a5e5f266091`
  - `media/sources/lerobot_v21.py` `20726442a33a`
  - `media/sources/lerobot_v3.py` `659c5fea74cf`
  - `media/sources/mp4_dir.py` `b4b7f9813a1a`
  - `media/types.py` `29f1608ac67f`
  - `modal_app/balanced_app.py` `dbb18ed30bba`
  - `modal_app/factory.py` `52e5826554fd`
  - `nv/__init__.py` `52bb0565cbb7`
  - `nv/budget.py` `330b1aef80c1`
  - `nv/capture.py` `ea3cffdd9cf9`
  - `nv/errors.py` `efc4f2053481`
  - `nv/linker.py` `ee6c24e1469f`
  - `nv/segment_glue.py` `518bdabee919`
  - `nv/session.py` `49fadc7f1f5d`
  - `observe/events.py` `59e290f57a71`
  - `observe/locks.py` `7ddfe2335626`
  - `observe/stdout.py` `184573a500c3`
  - `observe/timers.py` `2f4463f8f91b`
  - `prep/convert_captioner.py` `d9e0e97eacff`
  - `records.py` `1fa4bc7179ac`
  - `stages/__init__.py` `e3b0c44298fc`
  - `stages/caption.py` `276c7156d311`
  - `stages/reference.py` `5dfeedf19524`
  - `stages/segment.py` `b723e411ed8d`
  - `workers_impl/balanced.py` `73c640315191`
  - `workers_impl/caption.py` `bae68943032a`
  - `workers_impl/certstamp.py` `56ff03102855`
  - `workers_impl/reward.py` `1ed7c890d26d`
  - `workers_impl/segcap.py` `67d9cf6a4a7e`
  - `workers_impl/segment.py` `9100d4c1b579`
- covered by: **no certificate** (uncertified; nothing frozen yet)

## balanced / reward.terminal / H100! (default pool)

- key (`engine_spec_hash`, default modes, pinned derived digests): `24154dd56379f1e5`
- engine: `nvidia-vllm-offline` {'vllm': '0.24.0'}, image `img_vllm` fingerprint `4df6c7826a52`, patches [], engine_args `44136fa355b3`, sampling `44136fa355b3`, mm_kwargs `44136fa355b3`
- vllm_engine: none (not a balanced vLLM video kind)
- nvidia_pins: `rewarder`@7de68d22c4a9 (`0c6136d2830a`)
- decode: av 16.0.1, pillow 12.3.0, numpy 2.3.5, arch x86_64
- derived_weights: none
- modes: {"terminal_mode": "terminal"}
- logic: `0.0.1`, cert logic digest `f7f55ffb66f4f93f` over 30 modules:
  - `config/schema.py` `be1e93e2d01e`
  - `jobs/indexing.py` `82fc1ced0547`
  - `jobs/manifest.py` `1b2cd8b89ad6`
  - `jobs/planning.py` `92a62110f8ac`
  - `jobs/settings.py` `caaadcd39505`
  - `jobs/types.py` `9891533d7e60`
  - `media/__init__.py` `e3b0c44298fc`
  - `media/avsafe.py` `bec4f276de2b`
  - `media/cuts.py` `c73666ca2d09`
  - `media/index.py` `d9becae9d2b3`
  - `media/rangedecode.py` `d0133ce5903c`
  - `media/safepath.py` `b8f47d754db0`
  - `media/sources/__init__.py` `30b16c9dec00`
  - `media/sources/base.py` `2a5e5f266091`
  - `media/sources/lerobot_v21.py` `20726442a33a`
  - `media/sources/lerobot_v3.py` `659c5fea74cf`
  - `media/sources/mp4_dir.py` `b4b7f9813a1a`
  - `media/types.py` `29f1608ac67f`
  - `modal_app/factory.py` `52e5826554fd`
  - `modal_app/rewarder_app.py` `11338ffc0a49`
  - `modal_app/warm.py` `f3768d107f7e`
  - `modal_app/workers_app.py` `30b288a63562`
  - `nv/__init__.py` `52bb0565cbb7`
  - `nv/linker.py` `ee6c24e1469f`
  - `records.py` `1fa4bc7179ac`
  - `stages/__init__.py` `e3b0c44298fc`
  - `stages/reward.py` `a8ae59644323`
  - `workers_impl/__init__.py` `e3b0c44298fc`
  - `workers_impl/certstamp.py` `56ff03102855`
  - `workers_impl/reward.py` `1ed7c890d26d`
- covered by: **no certificate** (uncertified; nothing frozen yet)

## balanced / reward.dense / H100! (default pool)

- key (`engine_spec_hash`, default modes, pinned derived digests): `1edaa3b60ea88874`
- engine: `nvidia-vllm-offline` {'vllm': '0.24.0'}, image `img_vllm` fingerprint `4df6c7826a52`, patches [], engine_args `44136fa355b3`, sampling `44136fa355b3`, mm_kwargs `44136fa355b3`
- vllm_engine: none (not a balanced vLLM video kind)
- nvidia_pins: `rewarder`@7de68d22c4a9 (`0c6136d2830a`)
- decode: av 16.0.1, pillow 12.3.0, numpy 2.3.5, arch x86_64
- derived_weights: none
- modes: {}
- logic: `0.0.1`, cert logic digest `f7f55ffb66f4f93f` over 30 modules:
  - `config/schema.py` `be1e93e2d01e`
  - `jobs/indexing.py` `82fc1ced0547`
  - `jobs/manifest.py` `1b2cd8b89ad6`
  - `jobs/planning.py` `92a62110f8ac`
  - `jobs/settings.py` `caaadcd39505`
  - `jobs/types.py` `9891533d7e60`
  - `media/__init__.py` `e3b0c44298fc`
  - `media/avsafe.py` `bec4f276de2b`
  - `media/cuts.py` `c73666ca2d09`
  - `media/index.py` `d9becae9d2b3`
  - `media/rangedecode.py` `d0133ce5903c`
  - `media/safepath.py` `b8f47d754db0`
  - `media/sources/__init__.py` `30b16c9dec00`
  - `media/sources/base.py` `2a5e5f266091`
  - `media/sources/lerobot_v21.py` `20726442a33a`
  - `media/sources/lerobot_v3.py` `659c5fea74cf`
  - `media/sources/mp4_dir.py` `b4b7f9813a1a`
  - `media/types.py` `29f1608ac67f`
  - `modal_app/factory.py` `52e5826554fd`
  - `modal_app/rewarder_app.py` `11338ffc0a49`
  - `modal_app/warm.py` `f3768d107f7e`
  - `modal_app/workers_app.py` `30b288a63562`
  - `nv/__init__.py` `52bb0565cbb7`
  - `nv/linker.py` `ee6c24e1469f`
  - `records.py` `1fa4bc7179ac`
  - `stages/__init__.py` `e3b0c44298fc`
  - `stages/reward.py` `a8ae59644323`
  - `workers_impl/__init__.py` `e3b0c44298fc`
  - `workers_impl/certstamp.py` `56ff03102855`
  - `workers_impl/reward.py` `1ed7c890d26d`
- covered by: **no certificate** (uncertified; nothing frozen yet)

## balanced / reward.rubric / H100! (default pool)

- key (`engine_spec_hash`, default modes, pinned derived digests): `5b48d5d080c4e80a`
- engine: `nvidia-vllm-offline` {'vllm': '0.24.0'}, image `img_vllm` fingerprint `4df6c7826a52`, patches [], engine_args `44136fa355b3`, sampling `44136fa355b3`, mm_kwargs `44136fa355b3`
- vllm_engine: none (not a balanced vLLM video kind)
- nvidia_pins: `rewarder`@7de68d22c4a9 (`0c6136d2830a`)
- decode: av 16.0.1, pillow 12.3.0, numpy 2.3.5, arch x86_64
- derived_weights: none
- modes: {"rubric": null}
- logic: `0.0.1`, cert logic digest `612531b1948abf36` over 32 modules:
  - `config/schema.py` `be1e93e2d01e`
  - `jobs/indexing.py` `82fc1ced0547`
  - `jobs/manifest.py` `1b2cd8b89ad6`
  - `jobs/planning.py` `92a62110f8ac`
  - `jobs/settings.py` `caaadcd39505`
  - `jobs/types.py` `9891533d7e60`
  - `media/__init__.py` `e3b0c44298fc`
  - `media/avsafe.py` `bec4f276de2b`
  - `media/cuts.py` `c73666ca2d09`
  - `media/index.py` `d9becae9d2b3`
  - `media/rangedecode.py` `d0133ce5903c`
  - `media/safepath.py` `b8f47d754db0`
  - `media/sources/__init__.py` `30b16c9dec00`
  - `media/sources/base.py` `2a5e5f266091`
  - `media/sources/lerobot_v21.py` `20726442a33a`
  - `media/sources/lerobot_v3.py` `659c5fea74cf`
  - `media/sources/mp4_dir.py` `b4b7f9813a1a`
  - `media/types.py` `29f1608ac67f`
  - `modal_app/factory.py` `52e5826554fd`
  - `modal_app/rewarder_app.py` `11338ffc0a49`
  - `modal_app/warm.py` `f3768d107f7e`
  - `modal_app/workers_app.py` `30b288a63562`
  - `nv/__init__.py` `52bb0565cbb7`
  - `nv/errors.py` `efc4f2053481`
  - `nv/linker.py` `ee6c24e1469f`
  - `records.py` `1fa4bc7179ac`
  - `stages/__init__.py` `e3b0c44298fc`
  - `stages/reward.py` `a8ae59644323`
  - `stages/rubric.py` `ff6853682925`
  - `workers_impl/__init__.py` `e3b0c44298fc`
  - `workers_impl/certstamp.py` `56ff03102855`
  - `workers_impl/reward.py` `1ed7c890d26d`
- covered by: **no certificate** (uncertified; nothing frozen yet)
```

</details>
