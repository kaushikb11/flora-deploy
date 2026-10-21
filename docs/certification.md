# Certification: `balanced` vs `exact`, within NVIDIA's own noise

Status: 2026-10-22. The generator (`validate/certify.py`, `flora certify`) and its consumption
(`validate/certkey.py`, `config/presets.find_certificates`, `flora plan`) are built and tested on
CPU. No certificate exists yet: the Wave H-lite GPU runs produce the first inputs.

Background:
- `exact` runs NVIDIA's own HF code with model inputs proven identical. It is the reference.
- NVIDIA's own default H200 path is **not** run-to-run deterministic (`docs/determinism.md`: cuDNN
  SDPA). Deterministic mode is a different reference, not the same bits.
- So "faithful" can only mean: `balanced` diverges from `exact` no more than `exact` diverges from
  itself (PLAN.md §9, "Self-noise").

## Inputs

| Input | What | Required |
|---|---|---|
| `--exact` | `exact` records of the oracle run (JSONL, `flora_labels_v1`) | yes |
| `--balanced` | the candidate stack's records on the same episodes | yes |
| `--noise` | a second `exact` run of a subset, in NVIDIA's **default** mode. `exact` vs `noise` is the noise floor | no (without it: `provisional`) |
| `--l0` | `modal_app/balanced_lab.py` result JSON (L0 rows and Gate A), one per model kind | no (without it, video stages are `provisional`) |
| `--config` | the candidate run's `flora.yaml`: preset, pool GPUs, output-affecting modes | yes |
| `--pools` | `{uid: embodiment class}`, e.g. from the job manifest | no (one class, `all`) |

Records are joined on `(uid, stage)`. A stage's records on each side must be of the right preset
(`exact`, or the candidate preset), and unique per episode. The candidate's records must share
one stage hash (`provenance.config_hash`), and their `provenance.gpu`, where set, must name the
certified GPU. Their `logic_version` must be the checkout's, so certify from the commit that ran
them: the match key is computed from the code as it stands.

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
`strip()`, and a `None` caption is `""`. For unmatched slots, see below.

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
| `reward.dense` | `rank_divergence` | `1 - rho`, Spearman rank correlation (average ranks for ties) over the common prefix. If either side is constant, or fewer than 2 steps: 0 if equal, else 1 |
| `reward.dense` | `length_mismatch` | `1` if the `per_step` lengths differ |

Rubric has no certificate yet (P2, distributional).

## Pass rule

The noise floor is computed per stage, pooled over embodiment classes (noise runs are small):
`noise_m = mean over noise episodes of metric m(exact, noise)`. Then, for each (stage,
embodiment class) and metric:

```
mean over class episodes of m(exact, balanced)  <=  tolerance x noise_m + slack_m
```

Defaults: `tolerance = 1.5` (PLAN.md §9), `slack_m = 0` (`--slack metric=value` adds an
absolute allowance, recorded in the certificate). With zero noise (for example on L40S, which is
deterministic), the candidate must be identical unless a slack is given.

Verdict, in this order:
1. **`fail`** if any check fails (`divergence_above_noise_floor`), or a present L0 fails
   (`l0_failed`: any request whose `prompt_token_ids` or `video_grid_thw` differ from NVIDIA's HF
   processor).
2. **`provisional`** if any of these holds:
   - no noise run (`no_noise_run`): checks have no threshold and cannot fail or pass;
   - noise episodes < `min_noise_samples` (default 2);
   - candidate episodes < `min_samples` (default 5) in **any** embodiment class
     (`too_few_samples`);
   - coverage < `min_coverage` (default 1.0): the candidate did not label every `exact` episode;
   - no L0 for a video stage (`no_l0`). `segment` and `caption` need L0; rewards run the same
     NVIDIA vLLM engine as `exact`, so their L0 is `not_applicable`;
   - a metric could not be evaluated (`metric_not_evaluable`).
3. **`certified`** otherwise.

Gate A (FP32 `dt_bias`) numbers from the L0 JSON are recorded, but they do not affect the verdict.
Gate A decides which stack is certified: with the plugin on, the engine label and spec differ.

**`provisional` never unlocks a preset.** `find_certificates` returns only `certified`
certificates, so a provisional stack still needs `allow_uncertified`. The certificate says so in
`verdict`, `reasons` and `unlocks: false`.

## Certificate

`schema: flora-certificate/v1`, one per (preset, stage/shard kind, GPU). It is written to
`src/flora_deploy/config/certificates/<id>.json`, or to `--out-dir`. A failed certificate is
written only with an explicit `--out-dir`.

| Field | Content |
|---|---|
| `id` | `cert_` + the first 32 hex characters of the sha256 of the canonical body without `id` (content address; a hand-edited file is ignored) |
| `generator` | `{name: flora-certify, version, framework}` |
| `verdict`, `reasons`, `unlocks` | `certified` / `provisional` / `fail`; machine reasons; whether it unlocks the preset |
| `preset`, `stage`, `kind`, `gpu` | what it certifies; `gpu` is the Modal GPU type (`H100!` is not `H100`) |
| `stage_hash` | the candidate records' stage hash (`provenance.config_hash`) |
| `engine_spec_hash` | **the match key** (below) |
| `engine`, `modes`, `logic`, `pins`, `derived_weights` | engine name, versions, image and fingerprint, patches; output-affecting modes; logic version and digest; NVIDIA pin revisions; derived-weight receipts (`--receipt`) |
| `episodes` | `{count, digest, uid_sha256: [...]}`: sha256 of each episode uid, never the uid |
| `metrics` | per class `{n, means}`; noise `{n, means, episodes: {count, digest}}`; `coverage`; the similarity hook's name |
| `thresholds` | tolerance, sample minimums, coverage, slack, and each metric's resulting threshold |
| `checks` | `[{class, metric, value, threshold, pass}]` |
| `l0`, `gate_a` | L0 status and counts; Gate A summary numbers |

**Metrics only.** No caption, prompt, rationale, metadata, episode id or dataset name ever enters
a certificate. A property test feeds random strings into every string field of the input records
and checks that none appears in the certificate. The one record value carried is the stage hash,
our own digest. Embodiment class names come from `--pools`.

## Matching (`validate/certkey.py`)

A certificate applies to a pool when **all** of these hold:
- `preset`, `kind` and `gpu` are equal;
- `engine_spec_hash` equals the hash of the stack that would run now: the kind's stage-hash
  ingredients without the per-job part (input data, embodiment, instruction, envelope). That
  is the engine identity with its image fingerprint, the NVIDIA pins, the kind's output-affecting
  modes (`config.hashing.kind_modes`), the logic version and the kind's logic digest. Any engine,
  image, pin, mode or framework-logic change invalidates every certificate it touches, which
  forces re-certification (PLAN.md §12: "any engine/image/pin/logic bump");
- when the caller passes derived-weight receipts, any receipts the certificate recorded equal
  them;
- the file's `id` is its content address.

The stage hash itself is recorded but not matched: it also covers the input dataset, and a
certificate must apply to the same pool on other data (PLAN.md §9).

`certkey` is output-affecting (`config/logic.py`, `common`). It decides a record's `certified`
flag and `provenance.certificate`. The certificates and the generator are not.

Where it shows:
- `flora.yaml` validation refuses a non-`exact` preset unless every pool has a `certified`
  certificate, or `allow_uncertified` is set;
- `flora plan` prints each non-`exact` option's per-pool status (`certified` / `provisional` /
  `fail` / `none`, with the certificate id), and includes it in `--json`.

## Usage

```bash
flora certify -c flora.yaml --exact oracle/segment.jsonl --balanced bal/segment.jsonl \
  --noise noise/segment.jsonl --l0 reports/balanced-lab/tempseg-r1.json --pools pools.json
```

It prints the verdict per stage, the failing checks and where each certificate was written. It
exits 1 if any stage fails.
