# Docs index

Who reads what:
- **user**: runs `flora` on their own data;
- **contributor**: changes the code and needs the design and the rules;
- **maintainer**: decides plans, spends GPU money, certifies stacks and reviews evidence.

Start with the repository [`README.md`](../README.md) (users) or [`PLAN.md`](PLAN.md)
(contributors and maintainers).

## Plan and reference

| Doc | What it is for | Reader |
|---|---|---|
| [`PLAN.md`](PLAN.md) | Build plan v4: architecture, the numbered design decisions (D1-D18), cost model, determinism, licensing, roadmap | contributor, maintainer |
| [`cli.md`](cli.md) | Every `flora` command, its options and defaults, and exit codes; generated from the CLI | user |
| [`costs.md`](costs.md) | What a `balanced` job costs by size, where the money goes and how to spend less, including what warm starts and batch fill would save (estimates); its tables are generated from the rates | user |
| [`limitations.md`](limitations.md) | What flora-deploy doesn't do yet: certification status per embodiment, measured vs estimated costs, unsupported inputs, known approximations; its status tables are generated | user |
| [`export.md`](export.md) | `flora export --to lerobot`: which labels go where in the dataset copy, and alignment rules | user |
| [`output-format.md`](output-format.md) | Every field a finished job writes (JSONL, Parquet, `summary.json`, exclusions); checked against the code by a test | user, contributor |
| [`certification.md`](certification.md) | How a `balanced` certificate is computed, what it binds to, and when it unlocks a preset | user, maintainer |
| [`certificate-freeze.md`](certificate-freeze.md) | What a committed certificate freezes, how to tell whether a change affects it (the CI certificate-drift check), and the re-certification workflow; with the current freeze listing | contributor, maintainer |
| [`ci.md`](ci.md) | What each CI gate protects against, and how to run the gates locally | contributor |
| [`wave-h-runbook.md`](wave-h-runbook.md) | The GPU measurement-session runbook ("Wave H"): steps, caps, stop rules, spend accounting | maintainer |

## Design notes (how a component works and why)

| Doc | What it is for | Reader |
|---|---|---|
| [`balanced-engines.md`](balanced-engines.md) | The `balanced` stages: vLLM engine settings, the micro-batcher, the NVIDIA generate tap, deployment, validation commands | contributor |
| [`accel.md`](accel.md) | Runtime patches for `exact` (decode-once, in-memory PNG, pools) and their identity proofs (OPT-1) | contributor |
| [`media-notes.md`](media-notes.md) | Rules behind `flora_deploy.media`: per-episode `average_rate`, PTS, cuts, decode branches, envelope | contributor |
| [`determinism.md`](determinism.md) | Why NVIDIA's reference is not run-to-run deterministic on GPU (L40S, H200), and the opt-in deterministic mode | contributor, maintainer |
| [`captioner-bf16.md`](captioner-bf16.md) | The BF16 Captioner conversion, its bit-identity proof and load time (OPT-2) | contributor, maintainer |
| [`tempseg-merge.md`](tempseg-merge.md) | The FP32-merged TempSeg checkpoint, its determinism and equivalence check | contributor, maintainer |
| [`cold-start.md`](cold-start.md) | Where container startup time goes, the startup recipe per worker, ending idle tails at job end, and warm starts from compile caches | contributor |
| [`exact-cost.md`](exact-cost.md) | Input-identical levers on the `exact` preset's cost, and which are implemented | maintainer |

## Proposals

| Doc | What it is for | Reader |
|---|---|---|
| [`interactive-design.md`](interactive-design.md) | Proposal (not implemented): an online/interactive API | contributor, maintainer |

## Evidence (measurements and gate results)

| Doc | What it is for | Reader |
|---|---|---|
| [`gate-c.md`](gate-c.md) | vLLM model inputs vs NVIDIA's HF processor: the video-token fix and its proof (the "Gate C" capture check) | maintainer |
| [`exact-worker.md`](exact-worker.md) | The fast `exact` worker, its identity proof and measured throughput | maintainer |
| [`exact-concurrency.md`](exact-concurrency.md) | Episode concurrency under a GPU lock, identity result, where time goes, `exact` $/video-h | maintainer |
| [`perf-1.md`](perf-1.md) | Why `generate` is slower on the cu128 image, and the persistent GPU thread | maintainer |
| [`reference-smoke.md`](reference-smoke.md) | The first run of NVIDIA's pipeline on an H200 (partly superseded by `exact-worker.md`) | maintainer |
| [`rewarder-smoke.md`](rewarder-smoke.md) | Batched Rewarder vs NVIDIA's scripts, engine settings and rates | maintainer |
| [`golden-candidates.md`](golden-candidates.md) | Public datasets that stand in for each embodiment in flora's test set | maintainer |
| [`golden-manifest-v1.md`](golden-manifest-v1.md) | What is staged in flora's test set ("golden set") v1: counts, media, licenses, coverage | maintainer |
| [`golden-manifest-v2.md`](golden-manifest-v2.md) | The 12-episode `human` certification set for the certified run (staged, verified) | maintainer |
| [`wave-h-report.md`](wave-h-report.md) | The first measured `balanced` run: $/video-hour, startup, the rates in `planner/rates.json` | maintainer |
| [`scale-test.md`](scale-test.md) | A one-video-hour `balanced` scale test: prepared, not yet run on Modal | maintainer |

## Upstream

| Doc | What it is for | Reader |
|---|---|---|
| [`upstream-suggestions.md`](upstream-suggestions.md) | Draft suggestions for NVIDIA's Flora HF repos (not posted) | maintainer |

## Archive

Historical documents, kept for reference and not updated. Each one starts with a note saying so.

| Doc | What it was |
|---|---|
| [`archive/PLAN-v3.md`](archive/PLAN-v3.md) | Build plan v3 (`exact` first). Code comments cite its section numbers; `PLAN.md` §11 maps them |
| [`archive/P0_EXECUTION.md`](archive/P0_EXECUTION.md) | Phase 0 execution log: workstreams, waves 1-3, results |
| [`archive/NEXT_EXECUTION.md`](archive/NEXT_EXECUTION.md) | Execution log for the `exact` and `balanced` build phases and the first GPU measurement session, with dated updates |
