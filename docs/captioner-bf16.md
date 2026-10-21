# OPT-2: BF16 Captioner, loaded bit-identically

Status: measured 2026-10-21 on Modal (app `flora-convert`, ephemeral). The proof covers the
Captioner pin `nvidia/Flora-Captioner@fef0bbfe` with NVIDIA's SEGCAP code at `78a0e4ac`, using
torch 2.11.0+cu130 and transformers 5.8.1 on an H200.

## Why

NVIDIA ships the Captioner as FP32: 109.4 GB in 11 shards. Its loader builds the model in BF16
anyway. Every reference-worker start used to pay for all of this
(`docs/reference-smoke.md`):

- a 109 GB copy from the Volume to local disk, which `snapshot_download` needs because it takes
  a filelock inside `local_dir`;
- NVIDIA's `validate_manifest`, which re-hashes all 109 GB;
- a load that reads 109 GB and casts it.

## What NVIDIA's loader does (read from `batch_utils/models.py`, not changed)

`Session.load('captioner')` runs these steps:

1. It calls `production_gdn_fix._keep_native_gdn_fp32(Qwen3_5PreTrainedModel)`, which adds `A_log`
   and `dt_bias` to `_keep_in_fp32_modules_strict`.
2. It calls `path = snapshot(kind)`, then `validate_manifest(path)`.
3. It calls `AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, sdpa,
   device_map='balanced', ...)` and checks the loading info.
4. It runs `verify_cuda_residency` and `verify_gdn.verify(model, path)`. The second checks that
   `A_log`/`dt_bias` are FP32, on CUDA, and equal to the checkpoint.

In step 3, transformers 5.8.1 (`core_model_loading`) materialises each checkpoint tensor as
`tensor.to(device, dtype=target)`. The target is chosen in this order:

1. The `dtype_plan` glob that matches, which is FP32 for `A_log`/`dt_bias`.
2. Otherwise, the model's own parameter dtype.
3. Otherwise, BF16.

The 256 `*._extra_state` tensors (U8) are unexpected keys and are never loaded. Qwen3.5 has no
other keep-FP32 modules, and its buffers (`inv_freq`) are non-persistent, so they are computed
rather than loaded.

## Conversion (`src/flora_deploy/prep/convert_captioner.py`)

The conversion does not hard-code which tensors stay FP32. It applies NVIDIA's own
`_keep_native_gdn_fp32` (taken from the linked family), builds the model on the meta device, and
reads `_get_dtype_plan(bf16)` and the meta dtypes. That gives each key its target dtype. Each
source shard is read once into a buffer and sha256-hashed while it is read, so the bytes hashed
are the bytes converted. The hash must equal NVIDIA's `SOURCE_MANIFEST.sha256` entry, and that
manifest's own digest must equal the one pinned in `pins.py`. Each tensor is then cast with
`Tensor.to` and streamed into a new safetensors file. Peak memory is one source shard per
worker.

Output is written to `flora-weights:/derived/captioner-bf16@fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6/`:

- The BF16 shards, with the same names and the same key-to-shard map. `verify_gdn` reads that
  map.
- The index, with a new `total_size`.
- Every other snapshot file, copied byte for byte. `config.json` still says `float32`, but
  NVIDIA passes `dtype` explicitly, so that field is never read.
- `derived_receipt.json`, written last.

After writing, every file is re-read from the Volume and hashed.

| Result | Value |
|---|---|
| Output | 54.73 GB (54,733,799,263 B), 21 files |
| Casts | F32→BF16: 1,088; F32 kept: 96 (`A_log`, `dt_bias`); U8 `_extra_state` copied: 256 |
| Time | 267 s total: 240 s convert, 17 s re-read. CPU container: 8 cores, 56 GiB, 3 workers |
| Receipt sha256 | `0e91e467c7ffdb636573e779672d11b203571f7869ca94317f1005eb700d9c65` |
| Converter | v1, module sha256 `d4623073…`, git `399c49ec` (clean) |

The receipt records the following:

- for each file: its path, size, sha256 and kind;
- for each shard: its safetensors header sha256 and the sha256 of its FP32 source shard;
- the source pin record and NVIDIA's manifest digest;
- the dtype plan, with the count and digest of the keys that are not BF16;
- the cast counts;
- the versions of torch, transformers, safetensors and Python.

## Runtime patch (`src/flora_deploy/accel/captioner_load.py`)

`install(nv)` replaces two attributes of NVIDIA's `models` module. `Session.load` reads both as
module globals, and a unit test checks that against NVIDIA's bytecode.

- `snapshot('captioner')` returns the derived directory, read straight from the read-only Volume.
  No copy is made and no lock is taken. Every other kind goes to NVIDIA's function.
- `validate_manifest(derived_dir)` checks our receipt instead. Any other path goes to NVIDIA's
  function.

The checks (review 3, H1 and M1) run at install and again inside `Session.load`:

- **Pinned content.** The pin is a *content digest*, not the sha256 of the receipt file.
  `content_digest(receipt)` (`prep/convert_captioner.py`) is the sha256 of a canonical JSON of
  only the fields that determine the output:
  - the kind and the converter version;
  - the source pin and NVIDIA's manifest digest;
  - the dtype policy and the cast counts;
  - the sorted (name, sha256, size, header sha256) of every file.

  Timings, `created_utc`, tool versions and their build tags (`+cu130`), the platform, the
  commit and dirty flag, and the module sha256 are left out. Every user's own build of the same
  converter version therefore has the same content digest.
  - The receipt stores `content_sha256` when it is written. The runtime always recomputes the
    digest from the bytes it read, and refuses a receipt whose stored value disagrees.
  - The recomputed digest must equal `DERIVED_PINS["captioner-bf16"].content_sha256` in
    `pins.py`. A pin of `None` fails closed.
  - The schema, `ok`, converter version (code = pin = receipt), source revision and NVIDIA
    manifest digest must equal the pins.

  The sha256 of the receipt file still identifies one build. `flora prepare` records it under
  `weights.derived`, and the config hash includes it.
- **Full hash by default.** The directory must hold exactly the receipt's files, all regular
  files: symlinks, directories and devices are refused, and files are opened with `O_NOFOLLOW`.
  Every byte of every file is sha256-hashed on 8 threads, and a file that changes while it is
  hashed is refused. The old default (headers only for shards ≥ 64 MiB) accepted a byte flipped
  inside a shard (review p3). NVIDIA's `verify_gdn` cannot catch that either, because it compares
  against the derived shards.
- **Overlap with the TempSeg load.** `install(nv, background=True)` does the cheap checks, starts
  the hash on its own thread and returns. `Session.load('captioner')` (or `patch.wait()`) joins
  it and raises if it failed. A failed verification fails every later load too. The hash also
  warms the page cache for `from_pretrained`. Once the full hash has passed, later loads re-check
  the pinned receipt plus each file's (device, inode, size, mtime, ctime) stamp, the way NVIDIA's
  own `validate_manifest` reuses a verification.
- **Opt-out.** `full_verify=False` or `FLORA_CAPTIONER_FULL_VERIFY=0` falls back to the fast
  check: sizes, full hashes of non-shard and small files, header hashes of large shards. It is
  recorded as `full_verify_opt_out: true` in `patch.provenance()`.
- **Toolchain (M1).** The torch, transformers and safetensors versions of the process (local
  build tag such as `+cu128` ignored) must equal both the receipt's `tools` and
  `DerivedPin.proven_with` (torch 2.11.0, transformers 5.8.1, safetensors 0.7.0). The dtype plan
  is rebuilt with NVIDIA's own `_keep_native_gdn_fp32` (from the linked snapshot) and
  transformers' `_get_dtype_plan` on the meta device, and `DtypePolicy.record()` must equal the
  receipt's `dtype_policy`. Any difference fails closed, because the proof would no longer cover
  the load: for example, a transformers upgrade that keeps another module in FP32.
  - This runs synchronously in `install`, because building a meta model touches torch's
    process-wide default dtype. Call `install` before any model load and before episode threads
    start.
  - `check_toolchain=False` opts out, and is recorded in provenance.
- `patch.provenance()` records what was checked: the pinned digest, the opt-outs, the toolchain,
  the dtype plan and every verification report (`full` / `fast` / `reuse`, bytes hashed, timing).

`_keep_native_gdn_fp32`, the `from_pretrained` call, the loading-info check, residency, and
`verify_gdn` all run unchanged. `uninstall()` restores NVIDIA's functions. No NVIDIA file is
modified.

## Identity proof (`convert_app.py::identity`, one H200, network blocked)

On one GPU, two `Session` objects are loaded through NVIDIA's `Session.load('captioner')`:

- **A** runs NVIDIA's path unmodified. It loads the FP32 snapshot, copied to local disk exactly
  as the reference worker does.
- **B** runs the same call with the patch installed.

Both models then stay resident (101.9 GiB allocated in total).

| Check | Result |
|---|---|
| Parameters | 1,184 names equal: 1,088 BF16 + 96 FP32, all on `cuda:0`. 54.71 GB compared as raw bytes; dtype, shape and device equal; **0 mismatches** |
| Buffers | 3 (`inv_freq` ×2, `original_inv_freq`); **0 mismatches** |
| Forward pass (seeded 32 random tokens, text-only) | logits `[1, 32, 248320]` BF16 are **bit-identical**, sha256 `e2054819…` for both. A repeated A pass is bit-identical too, which rules out non-determinism |
| NVIDIA audit | `verify_gdn` returns `PASS_NATIVE_GDN_FP32_CHECKPOINT_VALUES` for both; the 96 FP32 value hashes are equal; `loading_info` is equal (256 unexpected `_extra_state` keys in both) |

Verdict: **identical**.

## Load time (Captioner only, H200)

| Step | Before (NVIDIA path) | After (patched) |
|---|---|---|
| Copy Volume → local disk | 172 s (109.4 GB) | none |
| Integrity check | 101 s (`manifest_full_verification`) | 0.1 s (+3.6 s fast check at install; see below) |
| `from_pretrained` + `verify_gdn` + processor | 27 s (from local disk) | 51 s (from the Volume) |
| **Total** | **≈305 s** | **≈56 s** |

These timings used the old fast check. The full hash that is now the default reads 54.7 GB.
At the Volume's measured read rate, that is roughly 20–40 s on 8 threads. Run it with
`background=True` so it overlaps the TempSeg load. Not yet measured on Modal.

The copy in this run took 172 s; the earlier smoke run measured 123 s. The patched load also needs
no 109 GB of local disk. Weights are now read from the Volume rather than from local NVMe; if
that 51 s matters, the next step would be a parallel prefetch of the 55 GB.

## Spend

The CPU conversion ran for about 5 min (≈$0.1). The H200 identity check ran for about 6.6 min
(≈$0.5 GPU, plus ≈$0.1 for CPU and memory). The total is about $0.7, against a cap of $3.

## Re-running

```
uv run modal run src/flora_deploy/modal_app/convert_app.py --step convert [--force]
uv run modal run src/flora_deploy/modal_app/convert_app.py --step identity
```

`convert` reuses a passing receipt of the same converter version unless `--force` is given.
Changing the converter, torch, transformers or the pins means re-running both steps. The runtime
enforces this: after a re-run, update `DERIVED_PINS["captioner-bf16"]` in `pins.py`, setting
`content_sha256` to the build's content digest and `proven_with` to the identity run's versions.
Until then, every load fails closed.

To read the digest, run the following. It starts no container and builds nothing: it is a
Modal client read of the Volume.

```
uv run flora prepare --show-receipt          # --json for machine-readable output
```

The content digest of the 2026-10-21 build is not pinned yet. This doc does not record the
per-file sha256s the digest covers, so it cannot be derived from here. `content_sha256` stays
`None` until it is read with `--show-receipt`. Until then, every load fails closed.

## Is the output byte-deterministic across machines?

Yes, for a fixed converter version, torch version and source snapshot. Each tensor is handled
in one of three ways:
- copied byte for byte;
- upcast exactly, from BF16 to FP32;
- cast from FP32 to BF16 with `Tensor.to`.

torch implements that cast as integer round-to-nearest-even on the FP32 bits. It does so in
the scalar path (`c10::detail::round_to_nearest_even`) and in the AVX2/AVX-512 paths
(`cvtfp32_bf16`) alike. The cast uses no floating-point arithmetic and no BF16 instructions, so
there is no flush-to-zero. Every element is independent, so thread count, SIMD width and CPU
features cannot change a byte.

The one exception is NaN. The scalar path encodes it as `0x7FC0` and the SIMD path as `0xFFFF`,
and which elements fall on the scalar tail depends on the SIMD width. The converter therefore
refuses a NaN in any tensor it casts.

Workers convert whole shards into separate files, and the safetensors header is encoded
deterministically. A different torch version could in principle change the cast. That is why
the content digest covers a single converter version and `proven_with` pins torch.

## Not done here

- Wiring `install` into `ReferenceWorker`. `reference_app.py` belongs to another owner. Call
  `install(nv, background=True)` once per worker, before `Session.load('tempseg')`, and record
  `patch.provenance()`.
- Measuring the default full hash on Modal.
