# CI gates

`.github/workflows/ci.yml` runs on every push to `main` and every pull request. Each gate is its
own named step (`Gate: ...`), so a red run says which guarantee broke, and steps run in order with
`bash -eo pipefail`, so the first failing gate stops the job. Nothing in CI talks to Modal or
needs a GPU. Tests marked `modal` never run in CI (`addopts = "-m 'not modal'"`).

The gates exist because reviews kept finding the same kinds of bug. Each one below names the
review or commit that motivated it, so a gate is never dropped without knowing what it protects.

## Job `test (3.11)` / `test (3.12)`

Installs `--group dev` only, so torch is **not** available. A test that needs torch must inject
a stand-in or be marked `nvlink`.

| Step | Command | Protects against | Motivated by |
| --- | --- | --- | --- |
| `Gate: uv.lock in sync with pyproject.toml` | `uv lock --check` (and `UV_LOCKED=1` for every later `uv` call) | A pin bumped in `pyproject.toml` without re-locking, so CI resolves something nobody reviewed | [Images: hashed transitive locks (review M3)](https://github.com/kaushikb11/flora-deploy/commit/7cc7648470769a68b003a1770f04f9bf981b6cd0) |
| `Gate: ruff` | `uv run ruff check .` | Lint/import-order regressions (one `main` run failed only on this) | CI run 39623409791 |
| `Gate: license guard` | `pytest tests/unit/test_license_guard.py` | NVIDIA code, prompts, weights or `LICENSE.pdf` (OneWay Noncommercial, see `NOTICE.md`), or local-only `research_notes/` / `reports/`, being committed to this public Apache-2.0 repo; also any file over 1 MB | [WS0 shared contracts](https://github.com/kaushikb11/flora-deploy/commit/4fdb5835a478ef4feab01c4ec64e587090c475cb), `NOTICE.md` |
| `Gate: logic-digest classification coverage` | `pytest tests/unit/test_config_logic.py` | A new module that changes outputs but is left out of the logic digest, so old results get reused after a behaviour change. Every module found on disk must be classified as output-affecting or not | Review 3 ([logic digest](https://github.com/kaushikb11/flora-deploy/commit/a3f7e961708e4e10916ba18505a9d41814ae4063)) and review 4 M2 ([every module classified](https://github.com/kaushikb11/flora-deploy/commit/e5431eb2a0b4c7d38ceabb29099c6f7ba3d61ff6)) |
| `Gate: pin / lock consistency` | `pytest tests/unit/test_pins.py tests/unit/test_modal_app_images.py` | NVIDIA pins that are not full commit SHAs, image locks that drift from the runtime pins or are not fully hashed, unpinned bases, or a `modal` client that differs from the pin the image fingerprints were made with | [review M3](https://github.com/kaushikb11/flora-deploy/commit/7cc7648470769a68b003a1770f04f9bf981b6cd0), [review 3 modal pin](https://github.com/kaushikb11/flora-deploy/commit/a3f7e961708e4e10916ba18505a9d41814ae4063) |
| `Gate: conformance suites` | `pytest tests/conformance` | A `ShardWorker` or `Store` implementation that breaks the shared contract (completion, retries, idempotent re-runs, refresh). Every check is also proven to fail on a deliberately broken worker | [C1 conformance suites](https://github.com/kaushikb11/flora-deploy/commit/09fcdce33e7184bc655562a525f7ebc79a1aa397) (they found 3 bugs in [the merge](https://github.com/kaushikb11/flora-deploy/commit/ec182263e43a814d5c7d61325af832b872eea845)) |
| `Gate: full CPU suite` | `pytest -m "not nvlink and not modal"` | Everything else that runs on CPU | — |
| `Gate: test-order independence` | the same suite with `FLORA_TEST_ORDER=reverse` (hook in `tests/conftest.py`) | Tests that pass only because an earlier test left state behind (`sys.modules`, `sys.path`, env vars, module globals), which hides bugs and breaks when tests are added or filtered | [WS-B review H1: order-safe tests](https://github.com/kaushikb11/flora-deploy/commit/6bd4e496b4f9e8749cb8a809ba93f09e78a3a4e6) |
| `Gate: local end-to-end job` | `flora run --local --fake-models` twice: once clean, once with injected preemption, OOM, failure, infra, spawn and NVIDIA errors (must exit 3 = `completed_with_errors`), then `flora results` | Problems that only show up when the real orchestrator, stores, workers, compose and CLI run together | [G2 local runner](https://github.com/kaushikb11/flora-deploy/commit/208ab2d25441185de0b30c788b7833c6a3f6dbac) |

The targeted gates run again inside the full suite. That costs a few seconds, and in return a
failure names the guarantee that broke.

## Job `nvlink`

Installs `--group dev --group nvlink` (the CPU torch/transformers stack that matches NVIDIA's
`batch_utils/requirements.txt`) and runs `pytest -m nvlink` against NVIDIA's real code at the
pinned revisions. This protects the glue between our code and NVIDIA's: linking, capture, budget,
pool and spawn safety, media/reward parity, and the reference entry point.

| Step | Protects against |
| --- | --- |
| `Gate: NVIDIA code present` | The nvlink tests skipping (the `nv_code` fixture skips when a pin is not fetched) and the job going green without testing anything |
| `Gate: nvlink` | Glue regressions against NVIDIA's real code. It includes `tests/nvlink/test_nv_link_order.py`, which runs the unit and nvlink linker suites in both orders in fresh interpreters (the unit fakes and the real SEGCAP links must not leak into each other; WS-B review H1/M5). `-rs` lists any skips and `--durations=15` shows the slowest tests |

## Caching

- **uv cache**: `astral-sh/setup-uv` with `enable-cache`, keyed on `uv.lock`, with separate
  suffixes for `dev` and `nvlink`. The nvlink cache is pruned before saving, because the torch
  wheels are faster to download again than to restore and would use up most of the repo's cache
  quota.
- **NVIDIA code** (`~/.cache/flora-deploy/nv`): `actions/cache`, keyed on the three pinned
  code revisions read from `flora_deploy.config.pins` plus a hash of
  `scripts/fetch_nv_code.py` (which decides the file patterns). A pin bump or a pattern change
  gives a new key, and the fetch runs only on a cache miss. Only code and processor/tokenizer
  configs are cached, never weights.
- **Where NVIDIA files go**: only the runner's disk and this repository's Actions cache.
  Actions caches are scoped to the repository and are not public downloads. The workflow uploads
  **no artifacts**, so NVIDIA files never become downloadable from a run page. Do not add an
  `upload-artifact` step that could pick up `~/.cache/flora-deploy`.

## Run hygiene

- `concurrency: ci-<ref>` with `cancel-in-progress`: a newer push to the same branch or PR
  cancels the run it supersedes.
- Timeouts: `test` 15 min (the local e2e step 2 min), `nvlink` 45 min (it takes about 21 min today).
- `permissions: contents: read`. Third-party actions are pinned: `setup-uv` by commit SHA, and
  the `actions/*` actions by major version.
- `fail-fast: false` on the Python matrix, so a failure on 3.11 does not hide the 3.12 result.

## Failures fixed when the gates were added (G4)

- `tests/unit/test_segcap_worker.py::test_determinism_label_and_drift` failed on `main` in every
  run from P1a-5s onward (`No module named 'torch'`). It relied on real torch being importable,
  but the `test` job has no torch, and dev machines with torch installed passed it. The test now
  injects a stand-in with torch's CPU defaults, so the drift check (and its assertions) runs the
  same way with or without torch.

## Running the gates locally

```bash
uv lock --check && uv sync --group dev
uv run ruff check .
uv run pytest -q tests/unit/test_license_guard.py tests/unit/test_config_logic.py \
  tests/unit/test_pins.py tests/unit/test_modal_app_images.py tests/conformance
uv run pytest -q -m "not nvlink and not modal"
FLORA_TEST_ORDER=reverse uv run pytest -q -m "not nvlink and not modal"
# nvlink
uv sync --group dev --group nvlink
uv run python scripts/fetch_nv_code.py --with-processor
uv run pytest -q -m nvlink
```
