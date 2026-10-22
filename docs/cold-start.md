# W3-B cold-start lab: where startup time goes, and how to cut it

Run 2026-10-21 with the ephemeral app `flora-coldstart-lab`
(`src/flora_deploy/modal_app/coldstart_lab.py`). The helpers are in
`src/flora_deploy/modal_app/warm.py`. Raw results are in `reports/w3b/*.json`, which is
git-ignored.

```bash
uv run modal run src/flora_deploy/modal_app/coldstart_lab.py::run_throughput   # CPU
uv run modal run src/flora_deploy/modal_app/coldstart_lab.py::run_manifest     # CPU, no network
uv run modal run src/flora_deploy/modal_app/coldstart_lab.py::run_engine --container 1 --tag T
uv run modal run src/flora_deploy/modal_app/coldstart_lab.py::run_engine --container 2 --tag T
```

## Summary

| Question | Answer |
|---|---|
| Volume read throughput | **About 1–2 GB/s for one sequential stream and 3–5 GB/s with 4–16 parallel readers** (US regions, cold). One container in `ap-northeast` got 0.55 GB/s single and 1.0–1.3 GB/s parallel. Beyond 4–8 readers there is no further gain. |
| Second read | A re-read **in the same container** comes from the page cache (7–28 GB/s), as long as the data fits in container memory. A **new container** gets no speed-up from earlier reads (no cross-container cache was seen). |
| The 8-min Rewarder load | **Sequential throughput variance does not explain it.** The worst sequential rate measured, 0.55 GB/s, would load 52 GiB in about 100 s. This lab's cold lazy (mmap) load took 35 s. 8 min means about 0.11 GB/s, which looks like slow mmap page-fault reads on a bad host or region. A parallel prefetch into the page cache removes the dependence on the mmap access pattern (weights load in 10 s once prefetched). |
| Rewarder engine startup on H100 | Cold everything: **341 s** (weights 35 s, compile 66 s, warm-up/profiling 81 + 81 s, CUDA graphs 3 s). With the `flora-cache` compile cache, same container: **95 s**. With the compile cache, a **fresh** container and prefetch: **224 s**. About 95 s of that is a cache that lives on container-local disk, not on `flora-cache` (see below). |
| Load-format options in vLLM 0.24.0 | `safetensors_load_strategy="prefetch"` exists. It reads the files into the page cache with 8 threads and 16 MiB blocks, starting when weight loading starts (about 45 s into startup). The auto mode never enables it on Modal, because the Volume is `9P`, not NFS or Lustre. With it on, output tokens were identical. `fastsafetensors` / `instanttensor` / `runai_streamer` need extra packages that `img_vllm` doesn't have, so they were not tested. |
| Avoiding the 109 GB Captioner copy | **Yes.** A writable "link farm" passes `huggingface_hub`'s offline resolution and NVIDIA's unmodified `snapshot()` + `validate_manifest` (full hash, 109.4 GB, 92–145 s). No copy is made and no local disk is used. |
| NVIDIA's manifest re-hash | It is paid **once per process**. The second `validate_manifest` returns `manifest_verified_reuse` in 0.04–0.07 s with 0 bytes read. The hash is single-threaded SHA-256, so it is CPU-bound at about 1.1–1.2 GB/s (92–103 s with prefetch, 145 s without). |

## 1. Volume read throughput (`flora-weights`, v2, CPU containers)

Setup: `warm.read_files` on 8 cores with 32 GiB. Each first read uses shards that this
container had not read before, so it is cold. Files are split into 256 MiB chunks so that N readers
can share 2–3 shards. GB/s are decimal. There were six fresh containers: one in `ap-northeast`,
four in `us-east`, one in `us-west`. Three ran concurrently.

| Read (≈10 GiB unless noted) | ap-northeast | us-east / us-west: median (range), n |
|---|---|---|
| Rewarder, 1 stream, sequential | 0.55 | 1.7 (1.0–1.8), 5 |
| Rewarder, 4 readers | 0.54 | 3.0 (2.9–4.3), 5 |
| Rewarder, 8 readers | 1.04 | 3.1 (1.8–4.1), 5 |
| Rewarder, 16 readers | 0.71–0.92 | 3.0 (1.1–3.9), 10 |
| Rewarder, 32 readers (64 MiB chunks) | — | 2.5, 1 |
| Captioner shard, 1 stream (9.8 GiB) | 0.61 | 1.9 (1.2–2.4), 5 |
| Captioner 92–98 GiB, 16 readers | 1.27 (78 s) | 3.7 (2.9–5.0), 5 (20–34 s) |
| Re-read in the same container, 1 stream | 7.5 | 1.2–11.4 |
| Re-read in the same container, 16 readers | 21.9 | 3.3–28.5 |

- Parallel readers give about 2× (US) and 2× (ap-northeast) over one stream. Use 8–16. More
  readers and smaller chunks are not better.
- Same-container re-reads are only fast while the data stays resident. In one container
  (us-east "D") the re-read ran at 1.2 GB/s after 50 GiB had been read into 32 GiB of memory, so the
  pages had been evicted. **Prefetch only helps if container memory exceeds the snapshot.**
- The first Modal run accidentally reused one warm container for three calls, at 22–34 GB/s.
  The cause was the page cache, not the Volume. The lab now uses `single_use_containers=True`.
- Region dominates. The `ap-northeast` container was 3× slower on every read pattern. GPU
  workers aren't region-pinned today, so a worker can land far from the Volume.

## 2. Rewarder: vLLM 0.24.0 engine startup on `H100!`

Each variant runs as a fresh Python process (NVIDIA's `engine.load_llm` with
`stages.reward.engine_spec`, which are NVIDIA's terminal-script arguments) in an `H100!`
container with 8 cores and 80 GiB. `VLLM_CACHE_ROOT=/cache/coldstart-lab/w3b-1` is on
`flora-cache`. Phase times come from vLLM's own log lines. "Wall" runs from process start to the engine being ready and one
greedy generation done. **All four variants produced identical token ids** (32 greedy tokens).

| Phase (s) | C1 cold: no cache, cold pages | C1 warm: same container, cache warm | C2 fresh container: cache warm + `warm.start_prefetch` (16 readers) at t=0 | C2 again + `safetensors_load_strategy="prefetch"` |
|---|---|---|---|---|
| Region | us-central | us-central | us-west | us-west |
| Process start → weight load starts (imports, HF config/processor, EngineCore spawn) | 55 | 42 | 54 | 45 |
| Weight load (`Loading weights took`) | **35.5** (lazy mmap from the Volume, 1.5 GB/s) | 14.0 (page cache) | **9.9** (prefetch done after 14.8 s at 3.8 GB/s) | 11.0 |
| torch.compile: Dynamo + Inductor, or cache load | **66.0** (Dynamo 10.9, graph compile 45.8) | 16.5 (cache load 13.0) | 30.0 (cache load 25.1, from the Volume, cold) | 8.4 |
| Initial profiling/warm-up run (Triton JIT/autotune) | **81.5** | 6.2 | 22.8 | 6.3 |
| Rest of profiling (silent) → CUDA graph memory | **80.9** | ≈3 | **77.4** | ≈3 |
| CUDA graph capture | 3 | 2 | 5 | 2 |
| `init engine (profile, KV cache, warm-up)` | 239.2 | 30.6 | 144.9 | 22.5 |
| **Wall** | **340.6** | **94.9** | **224.3** | **85.5** |

Findings:

- **CUDA graphs are cheap** (2–5 s, 9 graphs). They are not worth optimizing.
- **The compile cache on `flora-cache` works across containers.** The AOT-compiled model and
  Inductor graphs load in 13–25 s instead of 66 s of compile.
- **About 95 s is cached only on container-local disk.** In a fresh container (C2), "profiling/warm-up" took
  23 s and the silent stretch after it 77 s, against 6 s + 3 s in a second process in the same container.
  The likely cause is in vLLM 0.24.0 itself:
  - on the compile path (`compiler_interface.py:470-480`) vLLM points `TRITON_CACHE_DIR` into the compile cache;
  - on the **AOT-load** path (`decorators.py:550-557`) it only sets `TORCHINDUCTOR_CACHE_DIR`;
  - so on a cache hit, the Triton kernels launched outside the compiled graph (FLA/GDN prefill and conv kernels, plus their autotune results) are JIT-compiled into the default `~/.triton/cache` on local disk.
  
  **Fix (not yet measured):** set `TRITON_CACHE_DIR` (and
  `CUDA_CACHE_PATH`, plus FlashInfer's `XDG_CACHE_HOME`) to Volume paths keyed like the
  compile cache. Expected saving: about 90 s per fresh container.
- **Weight loading is not the bottleneck when the Volume is healthy** (35 s lazy, 10 s with
  prefetch). Prefetch is the insurance against the slow-mmap case (WS-H's 8 min).
- vLLM's own `"prefetch"` strategy is equivalent to `warm.start_prefetch`, but it starts later (at weight
  load, not at t=0) and with 8 threads. Our t=0 prefetch overlaps the 45–55 s of imports and
  config before loading, so the weights are already resident when loading starts.

## 3. Reference worker: Captioner with no local copy

**Layout** (`warm.build_linked_layout`, env `warm.linked_offline_env()`):

```
/snapshots/captioner/                 real, writable dir  (FLORA_MODEL_CACHE=/snapshots)
  model-0000N-of-00011.safetensors -> /weights/snapshots/nvidia--Flora-Captioner@fef0…/…   (one symlink per payload file)
  .cache/huggingface/…                real copies of the prep metadata (24 small files; locks go here)
/tmp/flora-offline/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> -> Volume  (as in WS-D)
```

**Why it passes unmodified:**

- `huggingface_hub` offline resolution with a commit-sha revision only stats the files. It
  follows the symlinks, so sizes and mtimes match the prep metadata. It takes its `filelock` in
  the writable `.cache/huggingface`.
- NVIDIA's `validate_manifest` path guard uses `allowed_root = path.parent.parent` when the parent
  dir is named `snapshots`. That is NVIDIA's provision for the hub cache layout, where snapshot
  files are symlinks into `blobs/`.
- **On Modal a Volume mount is a symlink to `/__modal/volumes/<vo-id>`** (measured). So the only
  root that contains both the Volume and a directory we may create is `/`, which means
  `FLORA_MODEL_CACHE=/snapshots`. With that root NVIDIA's guard accepts any path, so
  `warm.check_links` re-imposes it per snapshot: every link must resolve to a file inside the
  resolved Volume snapshot it came from.
- Integrity is unchanged: NVIDIA's manifest still hashes all 109.4 GB.

Tests: `tests/unit/test_warm_layout.py` runs real `huggingface_hub` and, with the code pin
fetched, NVIDIA's `snapshot`/`validate_manifest` on a read-only fake Volume. Those tests also check
that NVIDIA's guard rejects links leaving the root, and that tampered bytes fail the hash.

**Measured on Modal** (CPU, 8 cores, 96 GiB, network blocked, `img_hfref`, NVIDIA's code linked
from the Volume):

| Step | No prefetch (us-west) | 16-reader prefetch at hash start (us-east / us-west) |
|---|---|---|
| Link farm (TempSeg + Captioner) | 1.4 s | 1.2 / 1.4 s |
| `snapshot("tempseg")` + `validate_manifest` (0.37 GB) | 0.5 + 0.7 s | 0.5 + 0.5 s |
| TempSeg base `snapshot_download` (`models.py:130`) | 0.2 s | 0.2 s |
| `snapshot("captioner")` | 0.06 s | 0.06 s |
| `validate_manifest(captioner)` #1, `manifest_full_verification`, 109,447,256,940 bytes | **145.2 s** (0.75 GB/s) | **92.1 / 103.0 s** (prefetch done after 28 / 39 s, so the hasher is CPU-bound) |
| `validate_manifest(captioner)` #2 and `snapshot()` + validate #3 | 0.06 s, `manifest_verified_reuse` | 0.04–0.07 s, `manifest_verified_reuse` |
| Touch every page of every tensor via safetensors mmap (stand-in for the GPU load's reads) | — | 18.3 s (6 GB/s; mostly page cache) |

**`_VERIFIED_MANIFESTS` is per process.** It is keyed on the resolved snapshot path and
fingerprinted by path, inode, size, mtime and ctime. The re-hash is paid once per container as long as
the worker loads in one process; `CachedSession` already ensures that. It cannot be paid zero times
without changing NVIDIA's code.

## Recommended startup recipes

Prices: H100 $3.95/h, H200 $4.54/h, CPU $0.0472/core-h, memory $0.008/GiB-h.

**Reference / `exact` (H200, 8 cores, 96 GiB), for `reference_app.py`'s owner:**

1. Use `warm.linked_offline_env()` as the container env instead of `offline_env()`, and
   `warm.build_linked_layout()` instead of `build_offline_layout` (the weights stay at `/weights`).
   This drops the 123 s copy and 109 GB of local disk.
2. At t=0, `warm.start_prefetch(<captioner snapshot>, threads=16)`.
3. Run the Captioner hash **while TempSeg loads**:
   - start `Session.load("tempseg")`;
   - once TempSeg's own `validate_manifest` has returned (it holds NVIDIA's `_MANIFEST_LOCK`), call
     `nv.models.validate_manifest(nv.models.snapshot("captioner"))` in a background thread. Do not start
     earlier, or TempSeg's validate blocks behind the 100 s hash;
   - `Session.load("captioner")` then hits `manifest_verified_reuse`.
4. Expected `@modal.enter`: max(TempSeg 55 s, hash ≈ 95–105 s) + Captioner GPU load ≈ 22–30 s +
   ≈ 3 s ≈ **125–140 s**, against 251 s measured. That is about $0.20 per cold start instead of $0.40.
   Steps 1–2 alone (sequential, no overlap) give ≈ 180 s.
   - Not measured on GPU. The GPU load reads from the page cache, and with 96 GiB of memory against 102 GiB of
     data part of it may come from the Volume again (about 3.7 GB/s).

**Rewarder (H100, 8 cores), for `rewarder_app.py` / `images.py` owners:**

1. Mount `flora-cache` at `/cache`. `img_vllm` already sets `VLLM_CACHE_ROOT=/cache`; key the
   subdirectory by image + model + engine args. Populate it with one warm-up job that mounts
   the Volume read-write; workers mount it read-only. That read-only mount is untested, so check it
   once, because vLLM may write on a hit.
2. Set `TRITON_CACHE_DIR=/cache/<key>/triton` (plus `CUDA_CACHE_PATH`, and `XDG_CACHE_HOME` for
   FlashInfer) so the ≈ 95 s of container-local JIT/autotune is cached too.
3. `warm.start_prefetch(<rewarder snapshot>, threads=16)` as the first line of `@modal.enter`, and
   raise memory from 48 GiB to **≥ 72 GiB** so the 52 GiB stays resident (+$0.19/h). Equivalently,
   pass `safetensors_load_strategy="prefetch"`, which is I/O only, as an extra `load_llm` kwarg.
4. Expected: 224 s measured (fresh container, warm compile cache, prefetch, about $0.31 at $4.97/h) →
   ≈ **130 s** with step 2 (≈ $0.18). Fully cold (no cache) is 341 s (≈ $0.47). WS-H's worst case,
   929 s, was $1.0+.
5. Keep containers alive across shards (`scaledown_window`). A second engine in the same container
   costs 85–95 s.

**Future vLLM TempSeg / Captioner (`balanced`):** use the Rewarder recipe:
- compile + Triton caches on `flora-cache`, keyed per model and engine args;
- t=0 prefetch, with memory ≥ snapshot + 16 GiB (the Captioner BF16 derived weights ≈ 19 GiB; FP32 would need ≥ 120 GiB);
- no NVIDIA manifest, since vLLM doesn't run NVIDIA's loader. Our prep receipts are the integrity anchor.

Expect ≈ 100–130 s per fresh container. Also consider pinning GPU workers to the Volume's region
(US): the one `ap-northeast` container read 3× slower. Regional pinning has a price multiplier
(≥ 1.15×), so weigh it against that.

## Ending idle

`scaledown_window` is flora's own `WORKER_SCALEDOWN_S` (300 s); Modal's default is 60 s. On the
first real balanced job (6 episodes) the segment L40S sat idle for the full 300 s after its last
shard ($0.23 of its $0.50), and caption did the same at job end. `jobs/pools.py` removes both
tails.

**What modal 1.6.2 offers** (read from the installed client):
- `Cls.from_name(app, cls)().update_autoscaler(scaledown_window=, min_containers=,
  max_containers=, buffer_containers=)` overrides one class until the next deploy.
  `scaledown_window` must be in [2, 3600]. `Function.autoscaler.update()` is the newer spelling
  (`update_autoscaler` warns as deprecated since 2026-10-16), but a `Cls` instance doesn't
  expose it yet.
- `max_containers` can't be set to 0.
- There's no per-class "stop idle containers" call. `modal container stop` takes one container
  id, and `modal app stop` stops every class.
- `modal.experimental.stop_fetching_inputs()` exists, but it runs inside a worker. A worker
  can't tell its pool is finished, so a respawned shard would then need a cold container, and
  those files belong to the worker owners.

**What flora does.** When a class has no shard left that is queued, retrying, live or waiting on
a dependency, the orchestrator drops that class's window to 2 s. It does this on the tick where
it sees the last receipt, so at most one 15 s poll after the container went idle. It puts the
window back to 300 s before that class gets work again. Jobs on one app agree through the app's
pool Dict (`<watch dict>-pools`): a class is drained only when no other live job needs it.
Caption is never drained while segment still produces.

`calls_cost` charges a container that goes idle only up to *drain + 2 s*. Each drain is saved in
`<job>/<cfg>/pools/drains.json`. The planner charges each container a 17 s tail (one poll plus
2 s) in place of nothing.

**Not verified live:** whether Modal applies a lowered window to a container that is *already*
idle, or only to containers that go idle afterwards. The H0 rehearsal settles it
(`scripts/h0_rehearsal.py`, default `--preset balanced`): it polls Modal's running containers
during the jobs and checks that each one is gone at most 30 s after its class's drain, printing
the measured gap per container. If the window only
applies to containers going idle later, the fallback is to drain one tick *before* the last
shard ends. For that, the orchestrator would need the shard's ETA, which it already has from the
pre-warm estimates.

**Keep 300 s between shards.** With job-aware drain, the window only matters for gaps *inside* a
busy pool. Gaps come in two kinds:
- A pool with queued shards respawns within one poll (≤ 15 s), so any window above about 20 s
  covers it.
- The long gaps are a dependent pool waiting on its upstream mid-job, for example caption
  waiting for the next segment output.

For a gap of *g* seconds, keeping the container costs *g · r*, and letting it go costs one
startup *S · r* plus *S* of wall-clock. This is the ski-rental problem, so the best fixed window
is about *S*: segment 300 s, caption 325 s (`planner/rates.json`). A 60 s window would turn every
60–325 s caption gap into a 325 s cold start, which costs more and is slower. So the default
stays at 300 s. Changing it would also change every config hash, because `jobs/settings.py` is
hashed.

**Caption pre-warm.** Caption shards spawn only once their segment dependency is met
(`orchestrator.deps_ready`, tested in `tests/unit/test_pools.py`). On a small job, that puts
caption's 325 s cold start after segment's, one after the other. Here is how the orchestrator
handles it:
1. It estimates when the first caption shard becomes runnable: segment's spawn time, plus its
   startup, plus its work, from the budget gate's rate-table estimator.
2. At that time minus caption's startup, it sets caption's `min_containers` to 1.
3. Once the first caption call is live, it sets `min_containers` back.
4. It records the decision as a `pool_prewarm_plan` event, either `prewarm` with `saves_s`, or
   `skip` with the reason.

This saves up to about 5 min of wall-clock, and the cost is the estimate's error in idle H100
time (about $0.09 per minute early). The budget charges the warm container from its start. At
job end, `min_containers` is reset on every class a pre-warm may have raised.

**Orchestrator.**
- It now runs preemptible, at 0.125 cores, with 256 MiB requested (1 GiB limit) and a 2 s idle
  window. Before, it ran non-preemptible at 0.25 cores and 1 GiB, with Modal's default 60 s
  idle.
- Its own cost is small, about $0.015 per small job before this change. Most of the "$0.38"
  attributed to it was caption's idle tail.
- Preemption is survivable. The lease belongs to the function call id, which a restarted input
  keeps, and ledgers re-attach in-flight calls.
- Reconciler cron: it stays at every 10 min. Each run is now a few CPU-seconds with a 2 s idle
  window. A slower cron would make a dead orchestrator's gap longer, and a gap of caption idle
  plus restart cost $2.65 in the scale-test model. A faster cron would buy nothing that
  `modal.Retries` (30 s initial delay) doesn't already cover.

## Spend

- CPU throughput: six fresh containers plus one reused container, ≈ 8 container-minutes. ≈ $0.10.
- Manifest: three containers at 96 GiB, ≈ 6.5 minutes. ≈ $0.13.
- Mount probe: < $0.01.
- H100: two containers, 436 s + 310 s of function time, plus image and mount startup. ≈ $1.2.
- **Total ≈ $1.4** of the $4 cap. These are my own minute counts.
- `scripts/spend.py --app flora-coldstart-lab` still showed $0.00 because billing lags by about an hour.

## Warm start: keyed compile caches for the balanced workers (2026-10-22)

**Why the first balanced run compiled cold.** The deployed segment worker started at
02:11:55Z (300 s on an L40S) and the caption worker at 02:18:31Z (325 s on an H100). The lab's
caches on `flora-cache:/balanced/{tempseg,captioner}` were written from 02:23Z (07:53 IST, read
with `modal volume ls --json`). So the workers had nothing to copy. The lab's settings were not
the cause. Nothing wrote a cache back, so every later container would also have compiled cold.

### Design (`src/flora_deploy/engines/compile_cache.py`)

| Step | What happens |
|---|---|
| Key | `<engine kind>[-dtbias]-<GPU>-<24 hex>`, the sha256 of: `img_vllm` fingerprint; declared vLLM, torch, Triton and CUDA base versions (pins and the `img_vllm` lock); the engine's compile-relevant kwargs (model path, dtype, `max_model_len`, scheduler limits, `gdn_prefill_backend`, `enforce_eager`, video kwargs, ...); `dtbias_fp32`; the derived weights' content digest; the sha256 of `engines/patches/dt_bias_fp32.py` + `engines/worker_ext.py` (they run inside vLLM's workers); the normalised GPU type. A laptop can compute it (`worker_key`), so `flora plan --check-cache` can ask whether one exists. |
| Load (every container) | `seed`: read `balanced-cc/<key>/MANIFEST.json` from the **read-only** mount. Check the schema, key digest, all key factors and the container's own runtime versions (vLLM, torch, Triton, `torch.version.cuda`). Then copy every listed file to `/tmp/flora-cache/balanced/<key>` while hashing it, into a staging dir that is renamed into place only if every size and sha256 matched. On any failure the local dir is emptied, the reason is logged, and the engine compiles cold. |
| Legacy | With no keyed cache, the lab's unkeyed `balanced/<kind>[-dtbias]` tree is copied, but only on the GPU it was compiled on (`LEGACY_LAB_GPUS`: TempSeg L40S, Captioner H100). It is never counted as verified, so that container then publishes a keyed cache. |
| Cold retry | `build_engine`: if the engine build fails on a loaded (`warm` / `legacy`) cache for a reason other than the spec itself, the cache is emptied and the engine is built once more, cold. `EngineArgsError`, `EngineVerifyError` and `DtBiasActivationError` are not retried, because a cold build fails the same way. |
| Publish (one container per key) | After a shard whose engine ran requests and is alive (`after_shard` → `WorkerCache.maybe_publish`), a container that did not load a verified cache takes a lease: a `modal.Dict` `put(skip_if_exists=True)` on `<key>/<generation>`, TTL 30 min. An expired lease lets exactly one successor take generation + 1. The container then uploads its tree to a fresh `balanced-cc/<key>/<publish id>/` through `Volume.batch_upload`, so the mount stays read-only (M7). It re-checks the lease and only then writes `MANIFEST.json`, one file and atomic on a v2 Volume, as the commit. A partial or orphaned tree is never read, because no manifest names it. Lock and temp files and symlinks are not published. At most one attempt per container. `FLORA_COMPILE_CACHE_PUBLISH=0` turns publishing off. |

A marker file on the Volume would not work as the lease. Each container sees its own view of a
v2 Volume until it commits, so `O_EXCL` there is not exclusive across containers. `FileLeases`
is for tests and local runs only.

The key and the load result are in the worker's startup facts
(`balanced <kind> worker ready: {... "compile_cache": {...}}`), so a run report can tell a warm
start from a cold one.

### Do the lab's caches hit for the workers? Yes, on the same GPU

The lab's engines (`balanced_lab.child`) differ from the workers' engines in two engine
arguments, `logprobs_mode="raw_logits"` and `max_logprobs=20`, and in `keep_prompt_ids`, which
is ours and not passed to vLLM. vLLM 0.24.0 source:

- `vllm/config/model.py` `ModelConfig.compute_hash`: `logprobs_mode` is in `ignored_factors`.
  `max_logprobs` is hashed by value, and its default is 20 (`max_logprobs: int = 20`), so the
  lab's explicit 20 and the workers' default 20 hash identically.
- `vllm/compilation/backends.py:1026-1065`: the compile-cache dir hashes the env factors, the
  `VllmConfig.compute_hash()`, the traced code and the compiler (Inductor system + torch)
  factors. `vllm/envs.py:compile_factors` ignores `VLLM_CACHE_ROOT`, so the lab's
  `/cache/balanced/...` root and the workers' `/tmp/flora-cache/...` copy key the same.
- `vllm/compilation/decorators.py:528-560`: the AOT artifact
  (`torch_compile_cache/torch_aot_compile/<hash>/rank_0_0/model`, the path torch ≥ 2.10 takes)
  is keyed by `aot_compile_hash_factors`, which is env + config (+ Inductor factors only with
  `VLLM_USE_MEGA_AOT_ARTIFACT`, off below torch 2.12) + the forward's qualname. It is a
  self-contained serialized file, so relocating the root is safe. **It has no GPU factor.** That
  is why flora's key includes the GPU.
- Same model path (`/weights/derived/<name>@<rev>` in both), same `worker_extension_cls`, same
  scheduler limits from `GPU_LIMITS`, and the same `VLLM_*` env (the plugin's `VLLM_PLUGINS`
  only in the `-dtbias` caches).

`tests/unit/test_compile_cache.py::test_lab_gate_a_engine_has_the_worker_key` pins this: the
lab's spec and the worker's spec derive the same flora key. Today's lab caches (tempseg L40S,
captioner H100, tempseg-dtbias; captioner-dtbias is partial) are therefore used by the next
segment and caption containers through the legacy path. The first of each then publishes the
keyed cache.

### Pricing warm starts (`planner/rates.json`)

Startup rows carry `compile_cache: cold | warm`, which extends Lab-fix's flag. Two new `balanced`
rows, `segment` 150 s and `caption` 180 s (±60%), are `warm` **placeholders** until a warm
start is measured. `RateTable.startup_for(..., warm=False)`, the default and what `jobs.budget`
charges, never returns a warm row. `warm=True` prefers one:

- `flora plan --check-cache` passes the balanced kinds whose key has a published manifest
  (`PlanRequest.warm_kinds`);
- `--assume-warm` prices every pool warm, as a projection.

Options priced warm carry a `startup priced warm (...)` flag naming the row's status.
`scripts/scale_test.py:kind_rates` and `scripts/wave_h.py:balanced_job_seconds` take
`warm_kinds` too, and default to cold.

### Determinism

vLLM does not promise bitwise-identical outputs across compiles, and it does not promise that a
cache hit is bitwise identical to a fresh compile. What loading a cache does is replay the
artifacts one compile produced. The question is whether a cold compile would choose differently:

- Inductor's `max_autotune` (timing-based kernel choice) is enabled only for single-size compile
  ranges (`compiler_interface.py:set_inductor_config`), that is `compile_sizes`. Flora does not
  set it, and the default is none (`config/compilation.py:556`). The dynamic-shape graph uses
  Inductor's deterministic heuristics.
- The FLA/GDN Triton kernels use `@triton.autotune` (`model_executor/layers/fla/ops/*`), which
  times configs in every process. vLLM 0.24 sets no `cache_results`, so no choice is read from
  disk. A warm start autotunes exactly as a cold one does.
- vLLM recompiles when a traced source changed (`_verify_source_unchanged`) or an AOT load
  fails (`_try_load_aot_compiled_fn` logs and compiles again).

So a warm start runs the kernels a cold start would have compiled. The cache key is in the
startup facts but **not** in record provenance. `engines/compile_cache.py` is classified
non-output in `config/logic.py`. `modal_app/balanced_app.py`, which applies it, stays in the
`vllm-video` group.

### GPU memory snapshots (investigated, not built)

Modal can snapshot a container after `@modal.enter(snap=True)`, with `enable_memory_snapshot=True`
and `experimental_options={"enable_gpu_snapshot": True}` (both accepted by the pinned client
1.6.2), and restore it with GPU memory instead of re-running startup.

**Feasibility for vLLM 0.24: plausible but unproven.**

- The V1 engine runs EngineCore in a child process (`VLLM_ENABLE_V1_MULTIPROCESSING=1`). The
  front end talks to it over ZMQ IPC sockets in `/tmp`.
- A container snapshot covers the whole process tree and its file system, so both processes
  and the sockets come back together.
- Modal's own vLLM examples pair GPU snapshots with vLLM sleep mode
  (`enable_sleep_mode=True`, `llm.sleep()` before the snapshot, `wake_up()` after). That shrinks
  the snapshot, because the KV cache and, at level 1, the weights are offloaded.
- I have not checked that this works with our `OfflineEngine` and worker extension.

**Risks.**

1. *What goes into the snapshot.* Everything in `@modal.enter` before the snapshot is frozen:
   the job store, `FLORA_*` per-deployment env, preflight facts and the derived-weight
   verification. The start would have to be split. The `snap=True` phase would be the image
   and engine only (weights from the pinned path, the engine build). The `snap=False` phase
   would be the store, preflight, receipts, cert stamp and runner.
2. *Re-verification after restore.* The weights in GPU memory were verified when the snapshot
   was taken, not when it was restored. After restore, re-run `verify_gdn_dtypes` (worker
   extension RPC) and a canary greedy generate compared token for token with a recorded
   sequence. Keep the receipt and content-digest checks on the Volume in the `snap=False`
   phase.
3. *Snapshot keying.* Modal keys snapshots by app version and GPU type. Flora's
   image/spec/content key must still gate it: a redeploy with a different engine spec must
   never restore an old engine.
4. *NCCL / torch.distributed.* At TP=1 vLLM still initialises a world-size-1 process group. A
   restored NCCL communicator would be unsafe, but TP=1 should not create one. Any TP>1 pool,
   such as the rubric, is out of scope.
5. *CUDA graphs and allocator state* are restored at the same virtual addresses. That is fine
   in principle, but it is exactly what the canary has to prove.
6. *Determinism.* A restored engine is the snapshotting process's engine, so outputs match it.
   Gate C certifies the engine spec, so a canary mismatch must fail the container closed.

**A $0.50 experiment.** Use an ephemeral app with TempSeg on an L40S
(≈ $2.5/h with 8 cores and 48 GiB):

1. One snapshotting container: about 5 min cold build, a sleep, then the snapshot, ≈ $0.21.
2. Three restores, each with wake-up, the GDN dtype RPC and 2 greedy golden requests, about
   1 min each, ≈ $0.13.
3. Compare the token ids with a non-snapshot engine's tokens from the same lab call (reuse the
   balanced lab's recorded Gate A requests), and measure restore → first token.

Stop at the first restore that fails or mismatches. Total ≈ $0.35–0.50.
