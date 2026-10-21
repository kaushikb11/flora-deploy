# Certification: `balanced` vs `exact`, within NVIDIA's own noise

Status: 2026-10-22 (review 7 fixes). The generator (`validate/certify.py`, `flora certify`) and
its consumption (`validate/certkey.py`, `config/presets.find_certificates`, config validation,
`plan_app`, `flora plan`) are built and tested on CPU. No certificate exists yet. **No record
written today can be certified**: the workers do not yet stamp the binding provenance below.

Background:
- `exact` runs NVIDIA's own HF code with model inputs proven identical. It is the reference.
- NVIDIA's own default H200 path is **not** run-to-run deterministic (`docs/determinism.md`: cuDNN
  SDPA). Deterministic mode is a different reference, not the same bits.
- So "faithful" can only mean: `balanced` diverges from `exact` no more than `exact` diverges from
  itself (archive/PLAN-v3.md §9, "Self-noise").

Principle: a certificate is **derived from the records themselves** and **fails closed**.
Anything that cannot be shown from the records refuses or downgrades the certificate.

## Inputs

| Input | What | Required |
|---|---|---|
| `--exact` | `exact` records of the oracle run (JSONL, `flora_labels_v1`) | yes |
| `--balanced` | the candidate stack's records on the same episodes | yes |
| `--noise` | the oracle run **repeated** on a subset of its episodes: same stage hash, key, GPU and determinism mode. `exact` vs `noise` is the noise floor | no (without it: `provisional`) |
| `--l0` | `modal_app/balanced_lab.py` result JSON (L0 rows and Gate A), one per model kind | no (without it, video stages are `provisional`) |
| `--config` | the candidate run's `flora.yaml`: preset, pool GPUs, output-affecting modes, embodiment | yes |
| `--derived-content NAME=SHA256` | content digest of each derived weight the kind needs (`tempseg_merged`, `captioner_bf16`): W1-Prepare's `content_digest(receipt)`, the value `DerivedPin.content_sha256` pins | yes for `segment`/`caption` |

There is no `--pools`: the embodiment pool comes from the records (`provenance.embodiment`).

### Provenance a record must carry

Every record of every set (oracle, noise, candidate) must carry, in `provenance`:

| Field | Content | Today |
|---|---|---|
| `config_hash` | the D11 stage hash of its shard kind (64 hex) | set (segcap: stamped by the sink) |
| `gpu` | the GPU it ran on (`nvidia-smi` name; compared via `engines.offline_llm.normalize_gpu`) | stamped by every worker (`workers_impl.certstamp`; a record's own value wins) |
| `embodiment` | the job's `pipeline.embodiment` | stamped by every worker (`FLORA_EMBODIMENT`, `workers_impl.certstamp`) |
| `cert_key` | `validate.certkey.key_document(preset, kind, pool_gpu, modes=kind_modes(cfg, kind), derived_content=..., dtbias_fp32=...)` of the stack that produced it | stamped by every real worker: exact segcap (`reference_app`), the balanced `SegmentWorker` / `CaptionWorker` (`balanced_app.build_stage`) and the Rewarder (`workers_app`); fake workers stamp none (refused as `fake_models`) |
| `engine` | for `segment`/`caption`: the vLLM `EngineSpec.label` (must equal `cert_key.vllm_engine.label`) | set |
| `models` | pin key -> revision; each must agree with `cert_key.nvidia_pins` | set |

`flora certify` refuses records without them (`certified: false`, `missing_provenance`, naming
the fields). Records made by fake engines (`provenance.certificate` in `{fake-models,
local-fake-models}`, or an engine/model name starting `fake` / `flora-local-fake`) are refused
outright (`certified: false`, `fake_models`).

### Binding checks

1. **Candidate (H1).** Uniform in key, stage hash, GPU, embodiment and preset. Its key (sha256
   of `cert_key`) must equal the key of the stack this checkout would run, computed from
   `--config`, the pool GPU and `--derived-content`. The error names the differing key fields.
   The GPU must be the certified GPU.
2. **Oracle (H2).** Uniform in the same fields: one oracle run. Same embodiment as the candidate.
3. **Noise (H2).** Same stage hash, key, GPU and determinism mode as the oracle; its episodes a
   subset of the oracle's; no record with the oracle's own `audit_ref` (a copy). The oracle's
   stage hash, key, GPU and determinism mode are recorded in the certificate (`oracle`).

Note for Wave H-lite: an oracle run in `deterministic` mode needs its noise run in
`deterministic` mode too (they would otherwise differ in key and determinism).

## Metrics

Every metric is a **divergence** (0 = identical, larger = further apart). It is computed per
episode between a reference record `A` (`exact`) and another record `B` (`balanced`, or the
noise run's), then averaged over episodes. A per-episode value that is undefined (`None`) is
left out of the mean.

### Segment (`segment` records)

Intervals are matched by a **monotone alignment** of `A`'s and `B`'s ordered subtask intervals
that maximizes total IoU. Only pairs with IoU > 0 can match. Ties prefer a match, then skipping
an `A` interval. `IoU([s1,e1],[s2,e2]) = overlap / (len1 + len2 - overlap)`. Let `P` be the
matched pairs and `slots = max(|A|, |B|)`.

| Metric | Per episode |
|---|---|
| `iou_divergence` | `1 - sum(IoU over P) / slots`. An unmatched interval counts as IoU 0 |
| `boundary_error_s` | mean over `P` of `(|start_A - start_B| + |end_A - end_B|) / 2`, in seconds. `None` if `P` is empty |
| `count_mismatch` | `1` if `|A| != |B|`, else `0` (so the mean is the rate of unequal subtask counts) |
| `label_disagreement` | share of `P` whose label `(status, segmentation_provenance)` differs. `None` if `P` is empty |

Two empty episodes score 0 on every metric.

### Caption (`caption` records)

The intervals of the two caption records are aligned as above. Captions are compared after
`strip()`, and a `None` caption is `""`.

| Metric | Per episode |
|---|---|
| `caption_mismatch` | `1 - (exact matches over P) / slots`: 1 minus the exact-match rate |
| `caption_edit_distance` | `(sum over P of lev(a, b) / max(len a, len b) + unmatched) / slots`. Normalized Levenshtein in characters; an unmatched slot counts 1 |
| `caption_similarity_divergence` | only with `--similarity module:function`: `(sum over P of 1 - sim(a, b) + unmatched) / slots`, with `sim` clipped to [0, 1]. **Off by default.** The hook is the caller's own code; nothing here touches the network |

Caption records carry their own (balanced) segmentation, so caption metrics measure the
end-to-end segment + caption result.

### Rewards

| Stage | Metric | Per episode |
|---|---|---|
| `reward.terminal` | `expected_score_abs_delta` | `|E[s]_A - E[s]_B|` (renormalized top-k expectation) |
| `reward.terminal` | `score_mismatch` | `1` if the decoded score differs |
| `reward.dense` | `per_step_abs_delta` | mean `|a_i - b_i|` over the common prefix of `per_step` |
| `reward.dense` | `rank_divergence` | `1 - rho`, Spearman rank correlation (average ranks for ties) over the common prefix. When rho is undefined (either side constant, or fewer than 2 steps): the mean `|a_i - b_i|`. So two equal constant series score 0, and one `2` in ten `3`s scores 0.1, not the maximum (M6) |
| `reward.dense` | `length_mismatch` | `1` if the `per_step` lengths differ |

Rubric has no certificate yet (P2, distributional).

## Pass rule

The noise floor is computed per stage: `noise_m = mean over noise episodes of m(exact, noise)`.
For the stage's (single) embodiment pool and each metric:

```
mean over episodes of m(exact, balanced)  <=  min(cap_m, tolerance x max(noise_m, floor_m) + slack_m)
```

(`certkey.metric_threshold`).

### Policy (`certkey.POLICY`)

The knobs are bounded. `flora certify` refuses values outside the bounds, and matching refuses a
certificate that records them (H3).

| Knob | Default | Bound |
|---|---|---|
| `tolerance` | 1.5 (PLAN.md §4) | [1, 2] |
| `min_samples` (candidate episodes per pool) | 10 | >= 10 |
| `min_noise_samples` | 5 | >= 5 |
| `min_coverage` | 1.0 | >= 1.0 |
| `slack_m` (`--slack`) | 0 | [0, `floor_m`] |

**Absolute noise floors** `floor_m` (M4). A zero measured noise (two greedy runs that agree,
likely on short episodes) would otherwise demand a bit-identical engine on another GPU. Each
floor is below what a real regression costs:

| Metric | Floor | Why |
|---|---|---|
| `iou_divergence` | 0.02 | 1.5 x 0.02 = 3% of total IoU, less than one interval shifted by a tick on a 3-interval episode |
| `boundary_error_s` | 0.1 s | under half a TempSeg frame tick (4 fps = 0.25 s): a one-tick shift on every boundary still fails |
| `count_mismatch` | 0.05 | 1.5 x 0.05 = 7.5%: on 10 episodes, no unequal count is allowed; on 20, one |
| `label_disagreement` | 0.02 | as IoU |
| `caption_mismatch` | 0.05 | as `count_mismatch` for whole captions |
| `caption_edit_distance` | 0.02 | about one character in 30-50 per caption |
| `caption_similarity_divergence` | 0.02 | as edit distance |
| `expected_score_abs_delta` | 0.02 | 1.5 x 0.02 = 0.03, under the PLAN cap 0.05 |
| `score_mismatch` | 0.05 | as `count_mismatch` |
| `per_step_abs_delta` | 0.02 | one step in 50 off by one level |
| `rank_divergence` | 0.05 | rho >= 0.925 |
| `length_mismatch` | 0 | the step count is fixed by the video length |

**Absolute caps** from archive/PLAN-v3.md §2: terminal `E[s]` MAE <= 0.05 (`expected_score_abs_delta`).
PLAN's other bar, "boundary F1 at most 1 point below", is not enforced: no boundary-F1 metric
is computed yet.

### Verdict

In this order:
1. **`fail`** if any check fails (`divergence_above_noise_floor`), or a present L0 fails
   (`l0_failed`: any request whose `prompt_token_ids` or `video_grid_thw` differ from NVIDIA's HF
   processor).
2. **`provisional`** if any of these holds:
   - no noise run (`no_noise_run`): checks have no threshold and cannot fail or pass;
   - noise episodes < `min_noise_samples` (`too_few_noise_samples`);
   - candidate episodes < `min_samples` (`too_few_samples`);
   - coverage < `min_coverage` (`incomplete_coverage`): the candidate did not label every
     `exact` episode;
   - no L0 for a video stage (`no_l0`). `segment` and `caption` need L0; rewards run the same
     NVIDIA vLLM engine as `exact`, so their L0 is `not_applicable`;
   - a metric could not be evaluated (`metric_not_evaluable`).
3. **`certified`** otherwise.

**What `certified` needs** (H4), per stage and GPU: binding provenance on every record; at least
10 candidate episodes of the pool, all oracle episodes labelled; a noise run of at least 5 of the
oracle's episodes; L0 passing for `segment`/`caption`; every check within threshold. **Wave
H-lite (6 episodes, 2 noise episodes) is `provisional` by construction**, with or without
anything else.

Gate A (FP32 `dt_bias`) numbers from the L0 JSON are recorded, but they do not affect the verdict.
The plugin flag is part of the key (`vllm_engine.dtbias_fp32`): plugin-on and plugin-off stacks
are different stacks. No config field selects the plugin yet, so the checkout's key is plugin-off
and plugin-on records are refused.

**`provisional` never unlocks a preset.**

## Certificate

`schema: flora-certificate/v2`, one per (preset, stage/shard kind, GPU, embodiment pool). It is
written to `src/flora_deploy/config/certificates/<id>.json`, or to `--out-dir`. A failed
certificate is written only with an explicit `--out-dir`.

| Field | Content |
|---|---|
| `id` | `cert_` + the first 32 hex characters of the sha256 of the canonical body without `id` (content address; a hand-edited file is ignored) |
| `generator` | `{name: flora-certify, version: "2", framework}` |
| `generated_at` | UTC `YYYY-MM-DDTHH:MM:SSZ`; orders certificates for one key (below) |
| `verdict`, `reasons`, `unlocks` | `certified` / `provisional` / `fail`; machine reasons; whether it unlocks the preset |
| `preset`, `stage`, `kind`, `gpu` | what it certifies; `gpu` is the Modal GPU type (`H100!` is not `H100`) |
| `covered_pools` | the embodiment pool(s) measured (D16) |
| `stage_hash` | the candidate records' stage hash |
| `engine_spec_hash` | **the match key**, computed from the candidate records' `cert_key` |
| `engine`, `modes`, `logic`, `pins`, `derived_weights` | from the key: engine name, versions, image and fingerprint, patches and vLLM spec; output-affecting modes; logic version and certificate logic digest; NVIDIA pin revisions; derived-weight content digests |
| `oracle` | the oracle's `stage_hash`, `engine_spec_hash`, normalized `gpu` and `determinism` mode |
| `episodes` | `{count, digest, uid_sha256: [...]}`: sha256 of each episode uid, never the uid |
| `metrics` | per pool `{n, means}`; noise `{n, means, episodes: {count, digest}}`; `coverage`; the similarity hook's name |
| `thresholds` | the knobs, and per metric `{slack, floor, cap, threshold}` |
| `checks` | `[{class, metric, value, threshold, pass}]` (`class` is the pool) |
| `l0`, `gate_a` | L0 status and counts; Gate A summary numbers |

**Metrics only.** No caption, prompt, rationale, metadata, episode id or dataset name ever enters
a certificate. A property test feeds random strings into every free-text field of the input
records and checks that none appears in the certificate. The record values carried are our own
and validated: stage hashes (64 hex), the key document (equal to this checkout's), the GPU
(normalized), the embodiment and determinism mode (enums).

## The key (`validate/certkey.py`)

`key_document(preset, kind, gpu, *, modes, derived_content, dtbias_fp32=False)`:

- `engine`: the preset's `EngineIdentity` with its image fingerprint;
- `vllm_engine` (`segment`, `caption`): `engines.offline_llm.EngineSpec.for_gpu(...).hashed()`,
  the same object the worker hashes: every engine argument, the `dt_bias` plugin flag and the
  per-GPU scheduler limits (`GPU_LIMITS`);
- `nvidia_pins`, `decode` (av, Pillow, numpy, arch);
- `derived_weights`: name -> content digest, for every derived weight the kind needs (a missing
  one is an error: no key);
- `modes`: `config.hashing.kind_modes` (`determinism` for segcap, `terminal_mode` for terminal);
- `logic_version` and `logic_digest`: the **certificate logic digest** (M1).

The certificate logic digest (`cert_logic_digest`) is not the D11 config-hash digest (which is
unchanged). It covers the kind's D11 modules (`config.logic.modules_for`) minus
`config/logic.py` (the registry), `config/hashing.py` and `validate/certkey.py` (versioned by
`KEY_SCHEMA`), each **normalized**: comments, docstrings, trailing whitespace and blank lines
removed (from the source text, so it is the same under every Python version). A comment or
docstring edit, a registry edit or a new module elsewhere does not invalidate certificates; a
code edit on the kind's path, or adding a module to its group, does.

## Matching

A certificate applies to a pool when **all** of these hold:
- `preset`, `kind` and `gpu` are equal;
- the config's `pipeline.embodiment` is in `covered_pools` (H5; no embodiment: no match);
- `engine_spec_hash` equals the key of the stack that would run now. The caller passes the
  derived content digests; **without them a kind with derived weights matches nothing** (H6,
  fail closed);
- a `certified`/`provisional` certificate passes `policy_problems`: known generator version,
  knobs within policy and, for `certified`, every check recomputed from its own noise means and
  knobs, its sample counts and coverage (so editing a body and recomputing its id does not
  help);
- the file's `id` is its content address and `generated_at` is set.

**Order and revocation (M2).** Among the matching certificates for one key and pool, the newest
`generated_at` decides; on a tie `fail` beats `provisional` beats `certified`. So a newer `fail`
revokes an older `certified`, and a later `certified` (e.g. on more data) supersedes an older
`fail`. Only a deciding `certified` unlocks.

The stage hash is recorded but not matched: it also covers the input dataset, and a certificate
must apply to the same pool on other data (archive/PLAN-v3.md §9).

One lookup for every caller (H7): `certkey.lookup_context(cfg, kinds, derived_content=...)`
gives embodiment, modes and content digests to:
- config validation (`parse_config(..., derived_content=...)`): refuses a non-`exact` preset
  unless every pool has a deciding `certified` certificate, or `allow_uncertified` is set;
- `plan_app(..., derived_content=...)` (`flora deploy`): the same check;
- `flora plan`: each non-`exact` option's per-pool status (`certified` / `provisional` /
  `fail` / `none` / `unverified_weights`), and `recommended()` treats an all-certified option
  like the reference, so a cheaper certified `balanced` wins over `exact`.

Not wired yet: the CLI and SDK do not pass content digests (W1-Prepare's `content_digest`), so
`segment`/`caption` certificates match only through the API until integration.

## Usage

```bash
flora certify -c flora.yaml --exact oracle/segment.jsonl --balanced bal/segment.jsonl \
  --noise noise/segment.jsonl --l0 reports/balanced-lab/tempseg-r1.json \
  --derived-content tempseg_merged=<sha256>
```

It prints the verdict per stage, the failing checks and where each certificate was written. It
exits 1 if any stage fails or the records are refused (`certified: false`), 2 on bad input.
