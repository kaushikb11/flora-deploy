# CI gates

`.github/workflows/ci.yml` runs on every push to `main` and every pull request. Each gate is its
own named step (`Gate: ...`), so a red run says which guarantee broke, and steps run in order with
`bash -eo pipefail`, so the first failing gate stops the job. Nothing in CI talks to Modal or
needs a GPU. Tests marked `modal` never run in CI. Locally, a plain `pytest` also skips `nvlink` and `torch`
tests (`addopts = "-m 'not modal and not nvlink and not torch'"`); run them with
`pytest -m "(nvlink or torch) and not modal"`, as the nvlink job does. An explicit `-m` replaces
the default.

The gates exist because reviews kept finding the same kinds of bug. Each one below names the
review or commit that motivated it, so a gate is never dropped without knowing what it protects.

## Job `test (3.11)` / `test (3.12)`

Installs `--group dev` only, so torch is **not** available. A test that needs torch must inject
a stand-in, or be marked `torch` (needs only the CPU torch stack) or `nvlink` (also needs
NVIDIA's code); both run in the `nvlink` job. The full CPU suite runs in parallel with
pytest-xdist (`-n auto --dist loadfile`: each file stays on one worker, so module-level state and
module fixtures behave as in a serial run).

| Step | Command | Protects against | Motivated by |
| --- | --- | --- | --- |
| `Gate: uv.lock in sync with pyproject.toml` | `uv lock --check` (and `UV_LOCKED=1` for every later `uv` call) | A pin bumped in `pyproject.toml` without re-locking, so CI resolves something nobody reviewed | [Images: hashed transitive locks (review M3)](https://github.com/kaushikb11/flora-deploy/commit/7cc7648470769a68b003a1770f04f9bf981b6cd0) |
| `Gate: ruff` | `uv run ruff check .` | Lint/import-order regressions (one `main` run failed only on this) | CI run 39623409791 |
| `Gate: license guard` | `pytest tests/unit/test_license_guard.py` | NVIDIA code, prompts, weights or `LICENSE.pdf` (OneWay Noncommercial, see `NOTICE.md`), or local-only `research_notes/` / `reports/` (at any depth), being committed to this public Apache-2.0 repo; also NVIDIA's chat templates and HF tokenizer/processor/generation configs under any path, weights of any size (`.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.onnx`, `.gguf`), and any file over 1 MB. Paths only: the content half runs in the `nvlink` job | [WS0 shared contracts](https://github.com/kaushikb11/flora-deploy/commit/4fdb5835a478ef4feab01c4ec64e587090c475cb), `NOTICE.md` |
| `Gate: logic-digest classification coverage` | `pytest tests/unit/test_config_logic.py` | A new module that changes outputs but is left out of the logic digest, so old results get reused after a behaviour change. Every module found on disk must be classified as output-affecting or not | Review 3 ([logic digest](https://github.com/kaushikb11/flora-deploy/commit/a3f7e961708e4e10916ba18505a9d41814ae4063)) and review 4 M2 ([every module classified](https://github.com/kaushikb11/flora-deploy/commit/e5431eb2a0b4c7d38ceabb29099c6f7ba3d61ff6)) |
| `Gate: pin / lock consistency` | `pytest tests/unit/test_pins.py tests/unit/test_modal_app_images.py` | NVIDIA pins that are not full commit SHAs, image locks that drift from the runtime pins or are not fully hashed, unpinned bases, or a `modal` client that differs from the pin the image fingerprints were made with | [review M3](https://github.com/kaushikb11/flora-deploy/commit/7cc7648470769a68b003a1770f04f9bf981b6cd0), [review 3 modal pin](https://github.com/kaushikb11/flora-deploy/commit/a3f7e961708e4e10916ba18505a9d41814ae4063) |
| `Gate: conformance suites` | `pytest tests/conformance` | A `ShardWorker` or `Store` implementation that breaks the shared contract (completion, retries, idempotent re-runs, refresh). Every check is also proven to fail on a deliberately broken worker | [C1 conformance suites](https://github.com/kaushikb11/flora-deploy/commit/09fcdce33e7184bc655562a525f7ebc79a1aa397) (they found 3 bugs in [the merge](https://github.com/kaushikb11/flora-deploy/commit/ec182263e43a814d5c7d61325af832b872eea845)) |
| `Gate: full CPU suite` | `pytest -n auto --dist loadfile -m "not nvlink and not modal and not torch"` | Everything else that runs on CPU | — |
| `Gate: local end-to-end job` | `flora run --local --fake-models` twice: once clean, once with injected preemption, OOM, failure, infra, spawn and NVIDIA errors (must exit 3 = `completed_with_errors`), then `flora results` | Problems that only show up when the real orchestrator, stores, workers, compose and CLI run together | [G2 local runner](https://github.com/kaushikb11/flora-deploy/commit/208ab2d25441185de0b30c788b7833c6a3f6dbac) |

The targeted gates run again inside the full suite. That costs a few seconds, and in return a
failure names the guarantee that broke.

## Job `order (3.11)` / `order (3.12)`

| Step | Command | Protects against | Motivated by |
| --- | --- | --- | --- |
| `Gate: test-order independence` | the full CPU suite with `FLORA_TEST_ORDER=reverse` (hook in `tests/conftest.py`), **serially** (`-p no:xdist`) | Tests that pass only because an earlier test left state behind (`sys.modules`, `sys.path`, env vars, module globals), which hides bugs and breaks when tests are added or filtered | [WS-B review H1: order-safe tests](https://github.com/kaushikb11/flora-deploy/commit/6bd4e496b4f9e8749cb8a809ba93f09e78a3a4e6) |

It is serial so the order is the same on every run: under xdist the split of files across
workers is dynamic, so a cross-file leak would show up only sometimes. It is its own job so it
runs alongside `test` instead of after it. The hook also works under xdist (every worker reverses
the same collection, so xdist accepts it), e.g. for a quick local check.

## Job `nvlink`

Installs `--group dev --group nvlink` (the CPU torch/transformers stack that matches NVIDIA's
`batch_utils/requirements.txt`) and runs `pytest -m "(nvlink or torch) and not modal"` against
NVIDIA's real code at the pinned revisions. This protects the glue between our code and NVIDIA's:
linking, capture, budget, pool and spawn safety, media/reward parity, and the reference entry
point. `torch` tests are the ones that need real CPU torch but not NVIDIA's code (TempSeg weight
merge, BF16 Captioner conversion, the DT-bias plugin, real-torch determinism flags); before
review 7 they ran in no CI job. The `-m` expression repeats `not modal` because a command-line
`-m` replaces the one in `addopts` (review 7 G4-3).

The suite runs on all cores with `-n auto --dist load` (per test: three files hold about 70% of
the CPU time, so per-file scheduling would leave workers idle; module fixtures are built once per
worker that needs them).

| Step | Protects against |
| --- | --- |
| `Gate: NVIDIA code and processor files present` | The nvlink tests skipping (the `nv_code` fixture skips when a pin is not fetched; about nine modules skip without the `--with-processor` files) and the job going green without testing anything |
| `Gate: nvlink + torch` | Glue regressions against NVIDIA's real code. It includes `tests/nvlink/test_nv_link_order.py`, which runs the unit and nvlink linker suites in both orders in fresh interpreters (the unit fakes and the real SEGCAP links must not leak into each other; WS-B review H1/M5), and `tests/nvlink/test_license_content.py`, the content half of the license guard: no tracked file may contain a 6-line block of any NVIDIA text file or any long prose string (prompts, docstrings, JSON values) from NVIDIA's `.py`/`.json` files. The fingerprints are computed in the job from the fetched files and compared in memory; no NVIDIA text is stored in the repo. With `FLORA_FAIL_ON_SKIP=1`, a test skipped because a module (`could not import`, `No module named`) or an NVIDIA file (`not fetched`) is missing **fails** the run (`tests/conftest.py`; review 7 G4-1/G4-2). `-rs` lists the remaining (intended) skips and `--durations=15` shows the slowest tests |

## Caching

- **uv cache**: `astral-sh/setup-uv` with `enable-cache`, keyed on `uv.lock`, with separate
  suffixes for `dev` and `nvlink`. The nvlink cache is pruned before saving, because the torch
  wheels are faster to download again than to restore and would use up most of the repo's cache
  quota.
- **NVIDIA code** (`~/.cache/flora-deploy/nv`): `actions/cache`, keyed on the three pinned
  code revisions read from `flora_deploy.config.pins` plus a hash of
  `scripts/fetch_nv_code.py` (which decides the file patterns), with `proc` in the key because the
  cache holds the `--with-processor` file set. A pin bump, a pattern change or a change of fetch
  flags gives a new key, and the fetch runs only on a cache miss. Only code and processor/tokenizer
  configs are cached, never weights.
- **Where NVIDIA files go**: only the runner's disk and this repository's Actions cache.
  Actions caches are scoped to the repository and are not public downloads. The workflow uploads
  **no artifacts**, so NVIDIA files never become downloadable from a run page. Do not add an
  `upload-artifact` step that could pick up `~/.cache/flora-deploy`.
- **Fork PRs** can read the base branch's Actions cache and could print the NVIDIA files to the
  (public) run log. The NVIDIA Hugging Face repos are public anyway, so this exposes nothing new,
  but treat a PR that prints `~/.cache/flora-deploy` as a license problem.

## Run hygiene

- `concurrency: ci-<ref>` with `cancel-in-progress`: a newer push to the same branch or PR
  cancels the run it supersedes.
- Timeouts: `test` and `order` 10 min (the local e2e step 2 min), `nvlink` 30 min (about 21 min
  serially before xdist; see "Test speed" below).
- `permissions: contents: read`. Third-party actions are pinned: `setup-uv` by commit SHA, and
  the `actions/*` actions by major version.
- `fail-fast: false` on the Python matrix, so a failure on 3.11 does not hide the 3.12 result.

## Failures fixed when the gates were added (G4)

- `tests/unit/test_segcap_worker.py::test_determinism_label_and_drift` failed on `main` in every
  run from P1a-5s onward (`No module named 'torch'`). It relied on real torch being importable,
  but the `test` job has no torch, and dev machines with torch installed passed it. The test now
  injects a stand-in with torch's CPU defaults, so the drift check (and its assertions) runs the
  same way with or without torch.

- Review 7 G4-5: `test_determinism_label_and_drift` (stand-in) only reaches the "mode no longer
  in effect" branch. `test_determinism_flag_drift_is_refused` now puts the stand-in in
  deterministic mode and flips one flag, so the runner's "flags changed" branch is covered in the
  `test` job, and `test_accel_determinism.py::test_real_torch_deterministic_apply_then_drift_is_caught`
  does the same with real torch in the `nvlink` job.

## Running the gates locally

```bash
uv lock --check && uv sync --group dev
uv run ruff check .
uv run pytest -q tests/unit/test_license_guard.py tests/unit/test_config_logic.py \
  tests/unit/test_pins.py tests/unit/test_modal_app_images.py tests/conformance
uv run pytest -q -n auto --dist loadfile -m "not nvlink and not modal"    # fast, parallel
FLORA_TEST_ORDER=reverse uv run pytest -q -p no:xdist -m "not nvlink and not modal"
# nvlink + torch
uv sync --group dev --group nvlink
uv run python scripts/fetch_nv_code.py --with-processor
FLORA_FAIL_ON_SKIP=1 uv run pytest -q -n auto --dist load -m "(nvlink or torch) and not modal"
```

A plain `uv run pytest` (no `-n`) still runs serially; `-n` is opt-in so a single targeted test
does not pay for starting workers.

## Test speed

Profiled in October 2026 (`--durations=60`, 8-core laptop, load average 10-25 from other jobs):

- **CPU suite** (1735 tests with torch installed, 1699 without): about 90-115 s serially, with no
  single cause. The slowest tests are real-time concurrency smoke tests (SIGINT/preempt with
  episode sleeps, 1-3 s each), whole local jobs through the real orchestrator (0.5-2 s each),
  `uv build` / fresh-interpreter subprocess tests, and the 200+60-seed progress property tests
  (about 10 s in total). There is no hypothesis, no polling against real long timeouts (the local
  runner already polls every 20 ms) and no repeated expensive fixture. With torch installed,
  `test_convert_captioner.py` adds about 13 s of fixture setup. With xdist (`-n auto`) the suite
  takes well under a minute.
- **nvlink + torch** (222 tests): about 23 min serially. Nearly all of it is NVIDIA's own CPU
  work (synthetic videos through the real Flora processors, decode and parity comparisons); the
  heaviest files are `test_balanced_nvlink.py`, `test_accel_parity.py`, `test_accel_bench.py`
  and `test_reference_concurrency_nvlink.py`. Module fixtures already load each NVIDIA family and
  processor once per module, and fixture setup is a few seconds in total, so session-caching them
  would gain little and would share NVIDIA module state across files (which
  `test_nv_link_order.py` exists to keep separate). The tests are mostly single-threaded, so
  xdist scales close to linearly with cores.
- **Not done, on purpose**: no test was shortened, deleted or given fewer seeds. The seed loops
  cost about 10 s in total, and the parametrized ones cannot be capped without dropping collected
  tests. Real-time sleeps remain only in tests whose subject is real-time behaviour (signals,
  preemption while episodes are in flight, timeouts).
- If the `nvlink` job needs to be faster still, split it across two runners (for example the
  three heaviest files on one, everything else via `--ignore` on the other); each runner then
  pays the about 2 min torch install and NVIDIA fetch.
