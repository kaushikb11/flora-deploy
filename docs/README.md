# Docs index

Start with the repository [`README.md`](../README.md).

## Using flora-deploy

| Doc | What it covers |
|---|---|
| [`modal.md`](modal.md) | What runs where on Modal, private or local data, S3 output, what a job creates and how to clean up |
| [`python.md`](python.md) | The Python API: submit, wait, results and export from a script or notebook |
| [`cli.md`](cli.md) | Every `flora` command, its options and defaults, and exit codes (generated from the CLI) |
| [`costs.md`](costs.md) | What a job costs by size, where the money goes and how to spend less (tables generated from the rates) |
| [`limitations.md`](limitations.md) | What flora-deploy doesn't do yet: which stages are certified, measured vs estimated costs, unsupported inputs |
| [`export.md`](export.md) | `flora export --to lerobot`: which labels go where in the dataset copy, and alignment rules |
| [`output-format.md`](output-format.md) | Every field a finished job writes (JSONL, Parquet, `summary.json`, exclusions) |
| [`certification.md`](certification.md) | What `certified` in a record means, why `exact` needs no certificate, and what `allow_uncertified` does |

## Contributing

| Doc | What it covers |
|---|---|
| [`architecture.md`](architecture.md) | The pipeline from dataset to labels, the design decisions, determinism and licensing |
| [`ci.md`](ci.md) | What each CI gate protects against, and how to run the gates locally |
| [`certificate-freeze.md`](certificate-freeze.md) | What a committed certificate freezes and how the CI certificate-drift check works |

## Design notes

| Doc | What it covers |
|---|---|
| [`balanced-engines.md`](balanced-engines.md) | The `balanced` stages: vLLM engine settings, the micro-batcher, the NVIDIA generate tap, deployment |
| [`accel.md`](accel.md) | Runtime patches for `exact` (decode-once, in-memory PNG, pools) and their identity proofs |
| [`media-notes.md`](media-notes.md) | Rules behind `flora_deploy.media`: per-episode `average_rate`, PTS, cuts, decode branches, envelope |
| [`determinism.md`](determinism.md) | Why NVIDIA's reference is not run-to-run deterministic on GPU, and the opt-in deterministic mode |
| [`captioner-bf16.md`](captioner-bf16.md) | The BF16 Captioner conversion and its bit-identity proof |
| [`tempseg-merge.md`](tempseg-merge.md) | The FP32-merged TempSeg checkpoint and its equivalence check |
| [`cold-start.md`](cold-start.md) | Where container startup time goes, idle tails, and warm starts from compile caches |
| [`exact-cost.md`](exact-cost.md) | Input-identical levers on the `exact` preset's cost |
