# Certification: `balanced` vs `exact`, within NVIDIA's own noise

Status: 2026-10-23. The generator (`validate/certify.py`, `flora certify`) and its consumption
(`validate/certkey.py`, `config/presets.find_certificates`, config validation, `plan_app`,
`flora plan`) are built and tested on CPU. **Generator version 4** is the current rule
(`certkey.GENERATOR_VERSIONS = ("4",)`): caption certificates gate on **stage-isolated**
metrics (the balanced Captioner captions the oracle's own segments, with byte-identical model
inputs checked per call), and the end-to-end caption metrics are published as reported-only.
Every version-3 certificate is retired: it matches nothing under v4.

What this means for the committed files:
- One certificate is committed, the version-3 **`segment`** certificate for `human` on the L40S
  (`config/certificates/cert_f5d6433f….json`, with its freeze lock in `certificates/freeze/`),
  from run `10221548f1b8`. It is **retired**: v4 does not accept its generator version, and its
  key drifted anyway when `engines/offline_llm.py` changed (the Captioner limits below). The
  drift guard reports it as `retired` (information, not an error;
  [`certificate-freeze.md`](certificate-freeze.md)). The file stays until the re-certified v4
  segment certificate exists; the commit that adds the new one deletes it and its lock.
  **Balanced segmentation is therefore uncertified** between landing the v4 code and that
  re-certification.
- **`caption` is not certified.** On run `10221548f1b8` it failed its end-to-end noise-floor
  check (`caption_mismatch`, `caption_edit_distance`), and a failed certificate is never
  committed. Both stages are re-certified under v4 by the `recertify` plan, with rules fixed
  before that run ("Pre-registration and re-runs" below; [`wave-h-runbook.md`](wave-h-runbook.md),
  "The re-certification run"; [`wave-h-report.md`](wave-h-report.md)).

Background:
- `exact` runs NVIDIA's own HF code with model inputs proven identical. It is the reference.
- NVIDIA's own default H200 path is **not** run-to-run deterministic (`docs/determinism.md`: cuDNN
  SDPA). Deterministic mode is a different reference, not the same bits.
- So "faithful" can only mean: `balanced` diverges from `exact` no more than `exact` diverges from
  itself (archive/PLAN-v3.md §9, "Self-noise").

Contributors: once a certificate is committed, the code its key covers is frozen. CI fails a
change that makes a `certified` certificate stop matching. What is frozen, the drift report and
the re-certification workflow: [`certificate-freeze.md`](certificate-freeze.md).

Principle: a certificate is **derived from the records themselves** and **fails closed**.
Anything that cannot be shown from the records refuses or downgrades the certificate.

## What you see

`balanced`'s reward stages (`reward.terminal`, `reward.dense`, `reward.rubric` on `H100!`) run
exactly what `exact` runs, so they are the **reference** and need no certificate ("Reference by
identity" below). `segment` and `caption` need one for your embodiment and GPU; which exist is
in [`limitations.md`](limitations.md#accuracy-certificates). So:

- `flora plan` labels each option `reference`, `certified`, `partly certified` (some stages are
  certified or the reference, others are not) or `uncertified`. A `balanced` option is
  **certified** when its `segment` and `caption` both have one, and **partly certified** when
  only its rewards (or one of the two) are covered. Each option also gets a `certificates:` line with one label per stage:
  `certified (human, L40S)`, `reference (NVIDIA's own engine, no certificate needed)`, `not
  certified`, `certified for human, not for yam`, or `not certified (...)` with the reason: `only a
  provisional certificate`, `its certificate failed`, `certified for another build (engine,
  image, models, run modes, code or weights changed)` or `certificate not checked without the
  prepared weights (run flora prepare)`.
- A `balanced` config with an uncovered stage is refused unless you opt in with
  `allow_uncertified: true` in `flora.yaml`, or `--allow-uncertified` on `flora deploy` / `flora
  run`. The refusal says either "has no accuracy certificate for ..." (nothing certified) or "is
  not certified for every stage of this config (certified: ...; reference: ...): ..." (partial
  coverage). With the opt-in, `flora plan` keeps the option's `uncertified` label, ranks it on
  cost and flags it "balanced: uncertified, allowed by `allow_uncertified: true`"; without it the
  option is `unavailable`.
- The uncovered stages' records say `certified: false`. `flora export` refuses uncertified
  records unless you pass `--allow-uncertified` (the `flora.yaml` key does not carry over); the
  choice is recorded in the export's metadata.
- Once a certificate is committed, CI fails a change that makes a `certified` one drift; a
  drifted `provisional` one, or one committed without its freeze lock, only warns
  ([`certificate-freeze.md`](certificate-freeze.md)).

The committed (now retired) certificate came from the certified run (`scripts/wave_h.py --plan
certified`: 12 human episodes, a noise run on 6 of them, a 6-episode cold witness). Its v4
replacement and the first caption certificate come from the re-certification run
(`scripts/wave_h.py --plan recertify`: 12 human episodes, a noise run on **all 12**, 6-episode
cold witnesses; [`wave-h-runbook.md`](wave-h-runbook.md)).

## Method (contributors)

Terms used below:

- **oracle**: the `exact` reference run that a candidate is compared with.
- **noise run**: the oracle run repeated on its episodes (all 12 under v4); `exact` vs noise is
  the noise floor.
- **stage-isolated caption run**: a balanced caption-only job whose Captioner captions a pinned
  copy of the oracle's own `segment` records instead of balanced TempSeg's
  (`pipeline.caption_segments`). It measures the Captioner alone: segmentation differences
  cannot reach its captions.
- **input identity**: the per-call check that a Captioner request's model inputs are
  byte-identical on both sides. Each side records NVIDIA's `input_tensor_digest` per caption
  row (the oracle and noise runs in their audit's `model_calls`, the balanced workers through
  the same digest rule restated in the tap); input identity holds when the digests are equal.
- **input-identical pair**: an oracle caption row and the other side's row whose Captioner
  digests are equal. The isolated metrics compare captions over these pairs only.
- **reported-only**: a metric computed from the records and published in the certificate
  (`metrics.reported`) that is never a check, never part of the verdict and never read by
  matching. Under v4 the end-to-end caption metrics are reported-only.
- **L0**: the in-engine token/grid parity check (each request's `prompt_token_ids` and
  `video_grid_thw` against NVIDIA's HF processor), from `modal_app/balanced_lab.py`.
- **Gate A**: the FP32 `dt_bias` logit drift check, recorded alongside L0.
- **stage hash**: the hash of a shard kind's config and code (`config/hashing.py`).

## Inputs

| Input | What | Required |
|---|---|---|
| `--exact` | `exact` records of the oracle run (JSONL, `flora_labels_v1`) | yes |
| `--balanced` | the candidate stack's records on the same episodes; for a v4 `caption`, the stage-isolated caption run's records | yes |
| `--noise` | the oracle run **repeated** on some of its episodes: same stage hash, key, GPU and determinism mode. `exact` vs `noise` is the noise floor. Noise episodes the oracle did not label are dropped (`metrics.noise.unpaired`), so a spare fills in | no (without it: `provisional`) |
| `--witness` | the **cold witness**: candidate records from one container forced cold (`FLORA_COMPILE_CACHE=off`), the same stack, on some of the oracle's episodes. Its means are held to the same thresholds as the candidate's on their own; for `segment`, its subtask-count flips are counted together with the candidate's ("Pass rule", "Warm and cold compiles"). For a v4 `caption` it is a stage-isolated caption-only job, held to input identity like the candidate | no (without it, or with no record forced cold, `segment`/`caption` are `provisional`: `no_witness`) |
| `--oracle-segments` | `caption`: the pinned oracle segment JSONL the stage-isolated caption run captioned (the oracle's `segment` records, `exact` preset, one record per line), byte-identical to the file pinned in the run's `pipeline.caption_segments` (its sha256 is checked against every candidate record's `provenance.segments_from`) | yes for `caption` (without it: refused) |
| `--digests SIDE=PATH` | `caption`: one digest JSON per side, `{uid: [hex64 or null, one per caption row in record order]}`, hex only, no text (`scripts/wave_h_finalize.py --fetch` writes them). Sides: `exact` (the oracle, from its audit's Captioner `model_calls`), `noise`, `balanced` (the isolated candidate), `witness` (the isolated cold witness), and optionally `repeat` (reported only). Repeatable | `exact` and `balanced` yes for `caption` (without them: refused); `noise` and `witness` as their records |
| `--reported-e2e` | `caption`: the end-to-end balanced job's (J1) caption records, on its own segmentation, for the reported-only block (`metrics.reported.e2e`) | no (without it the block is absent) |
| `--reported-repeat`, `--families` | reported only: the old 6-episode noise run's caption records (needs `--digests repeat=`); a JSON `{uid: label}` (label `[a-z0-9_.-]{1,32}`, e.g. `n-egodex`) for the per-family split | no |
| `--l0` | `modal_app/balanced_lab.py` result JSON (L0 rows, Gate A and `engine_hashes`, the `certkey.engine_settings_hash` of each engine it built), one per model kind. A lab whose baseline engine hash is not the candidate's engine settings is stale (`stale_l0`) | no (without it, or stale, video stages are `provisional`) |
| `--config` / `-c` | the candidate run's `flora.yaml`: preset, pool GPUs, output-affecting modes, embodiment | default `flora.yaml` |
| `--derived-content NAME=SHA256` | content digest of each derived weight the kind needs (`tempseg_merged`, `captioner_bf16`): `flora prepare`'s `content_digest(receipt)`, the value `DerivedPin.content_sha256` pins | yes for `segment`/`caption` |
| `--stage` | stage(s) to certify (repeatable) | default: every certifiable stage in the candidate records |
| `--gpu` | GPU type certified | default: the config's pool GPU for the stage |
| `--tolerance`, `--min-samples`, `--min-noise-samples`, `--slack` | the policy knobs (below) | defaults 1.5, 10, 5, 0 |
| `--generated-at` | certificate time `YYYY-MM-DDTHH:MM:SSZ`; pass the run's own time so re-certifying the same records reproduces the same certificate id | default: now |
| `--json`, `--out-dir`, `--similarity` | print JSON; where to write; the optional caption-similarity hook | no |

`flora certify` refuses an `exact` config ("exact is the reference by construction"), a records
file with no certifiable stage (e.g. `reward.rubric` only), and records whose embodiment differs
from the config's. After a `provisional` verdict it prints "Provisional certificates do not
unlock a preset: runs still need allow_uncertified."

There is no `--pools`: the embodiment pool comes from the records (`provenance.embodiment`).

### Provenance a record must carry

Every record of every set (oracle, noise, candidate) must carry, in `provenance`:

| Field | Content | Today |
|---|---|---|
| `config_hash` | the stage hash of its shard kind (64 hex) | set (segcap: stamped by the sink) |
| `gpu` | the GPU it ran on (`nvidia-smi` name; compared via `engines.offline_llm.normalize_gpu`) | stamped by every worker (`workers_impl.certstamp`; a record's own value wins) |
| `embodiment` | the job's `pipeline.embodiment` | stamped by every worker (`FLORA_EMBODIMENT`, `workers_impl.certstamp`) |
| `cert_key` | `validate.certkey.key_document(preset, kind, pool_gpu, modes=kind_modes(cfg, kind), derived_content=..., dtbias_fp32=...)` of the stack that produced it | stamped by every real worker: exact segcap (`reference_app`), the balanced `SegmentWorker` / `CaptionWorker` (`balanced_app.build_stage`) and the Rewarder (`workers_app`); fake workers stamp none (refused as `fake_models`) |
| `engine` | for `segment`/`caption`: the vLLM `EngineSpec.label` (must equal `cert_key.vllm_engine.label`) | set |
| `models` | pin key -> revision; each must agree with `cert_key.nvidia_pins` | set |
| `compile_cache` | `segment`/`caption`: the compile cache its container ran (`warm:<key>:<manifest sha256[:16]>`, `legacy:<key>`, `cold`, `cold:forced`); counted, not required | stamped by the balanced workers |
| `engine_preemptions` | `segment`/`caption`: vLLM preemptions while the episode's requests were in flight (`vllm:num_preemptions`); `None` is unknown | stamped by the balanced workers; > 0 or `None` makes the certificate `provisional` |
| `segments_from` | stage-isolated `caption` records only: `{sha256, oracle_stage_hash}` of the pinned oracle segment file they captioned. Absent from every other record (and from its JSON, so older records and their bytes are unchanged) | stamped by the balanced `CaptionWorker` when `pipeline.caption_segments` is set |

The per-row Captioner input digests are not provenance: they are in the caption audits (the
balanced episode audit rows' `input_digests`; the oracle's and noise run's
`result.flora_reference.model_calls`), and certify reads them from the digest JSONs.

`flora certify` refuses records without them (`certified: false`, `missing_provenance`, naming
the fields). Records made by fake engines (`provenance.certificate` in `{fake-models,
local-fake-models}`, or an engine/model name starting `fake` / `flora-local-fake`) are refused
outright (`certified: false`, `fake_models`).

### Binding checks

1. **Candidate.** Uniform in key, stage hash, GPU, embodiment and preset. Its key (sha256
   of `cert_key`) must equal the key of the stack this checkout would run, computed from
   `--config`, the pool GPU and `--derived-content`. The error names the differing key fields.
   The GPU must be the certified GPU.
2. **Oracle.** Uniform in the same fields: one oracle run. Same embodiment as the candidate.
3. **Noise.** Same stage hash, key, GPU and determinism mode as the oracle; no record with the
   oracle's own `audit_ref` (a copy). Only its episodes the oracle labelled are paired; the rest
   are counted (`unpaired`), not an error. The oracle's stage hash, key, GPU and determinism
   mode are recorded in the certificate (`oracle`).
4. **Witness.** Same key, stage hash, GPU and embodiment as the candidate; paired with the
   oracle like the noise run.
5. **Isolated caption inputs** (`caption`, v4). Each of these **refuses**: no certificate is
   written (`CertifyRefused`), so it can neither pass nor revoke anything.
   - `segments_not_oracle`: a candidate (or witness) caption record's
     `provenance.segments_from.sha256` is not the sha256 of the `--oracle-segments` file, or its
     `oracle_stage_hash` is not the oracle records' stage hash, or an episode's caption rows are
     not the oracle's segments: a different row count, or a start or end more than 1e-6 s from
     the oracle row's.
   - `oracle_digests_missing`: an oracle caption row with a Captioner call (status `complete`)
     has no digest. The k-th Captioner call of the oracle's audit (`output_limit` 512, in order)
     pairs with its k-th `complete` caption row; other rows have no call and a `null` digest.
   - `inputs_not_isolated`: for some row the candidate's (or witness's) digest is not the
     oracle's, including one side `null` and the other not. Byte-identical model inputs are the
     premise of isolation, so a mismatch means the harness (the row reconstruction from the
     oracle's records) is wrong, not the engine. The fix is the harness, never a threshold.

   The isolated run's caption stage hash differs from an end-to-end job's (the pinned file is in
   its caption shard entry, so its shards never mix with production ones), but its certificate
   key is the production caption key (`caption_segments` is not a mode and not in
   `key_document`): the certificate is for the stack, not for the input. A v4 `caption`
   certification without `--oracle-segments`, `--digests exact=` and `--digests balanced=`
   refuses too. `segment` and the reward stages are unchanged.

Note: an oracle run in `deterministic` mode needs its noise run in `deterministic` mode too (they
would otherwise differ in key and determinism). The certified run uses NVIDIA's default mode
(`determinism: nvidia`) for all of its jobs.

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

### Caption (`caption` records, stage-isolated)

Under v4 the caption certificate measures the **Captioner alone**. The candidate is a
stage-isolated caption run: it captions the oracle's own segments, and every compared caption had
byte-identical model inputs on both sides (input identity, verified per call). Segmentation is
gated by the `segment` certificate, not charged a second time here.

Notation, per episode in oracle ∩ candidate: `o_1..o_n` are the oracle's caption rows and
`c_1..c_m` the candidate's, in order; `cap(x)` is the caption after `strip()`, with `None` as
`""`; `ned(a, b) = lev(a, b) / max(len a, len b)` (normalized Levenshtein in characters, 0 when
both are empty); `dO(i)` and `dC(i)` are the Captioner input digests of row `i` (`null` for a
row with no call, e.g. `unsupported_target_frames`).

**Candidate (the gate).** After the refusals ("Binding checks", 5), `m == n` and the bounds are
the oracle's. The slots are `S = {i : dO(i) is not null}`, the rows the oracle actually
captioned, and the pairs are `(o_i, c_i)` for `i` in `S`: no IoU alignment is needed.

| Metric | Per episode |
|---|---|
| `caption_mismatch` | `#{i in S : cap(o_i) != cap(c_i)} / |S|` |
| `caption_edit_distance` | mean over `S` of `ned(cap(o_i), cap(c_i))`. No unmatched term: it is structurally 0 |

Every oracle episode with a candidate record and `|S| >= 1` counts; there is no count-matched
restriction, because the counts match by construction. Aggregation is macro (the plain mean
over episodes), as in v3. `count_mismatch` is **not** a caption metric under v4: it is
structurally 0 (`segments_not_oracle` guarantees it), so the caption certificate runs no
count-flip check. Subtask counts stay gated by the `segment` certificate. The cold witness (an
isolated caption-only job forced cold) is scored the same way and must pass input identity too.

**Input identity** (required evidence). The certificate records `input_identity = {candidate,
witness}`, each `{pairs, equal, episodes, status, stage_hash, oracle_stage_hash}`. It is
`pass` only when `equal == pairs` over every slot of every episode; anything else has already
refused (`inputs_not_isolated`). It covers **every** Captioner request (98 for the human
candidate), not the 6 that L0 samples. It complements L0: input identity shows the candidate's
HF-side inputs equal the oracle's, and L0 shows vLLM's own rendering equals HF's.

**Noise (stage-isolated): `exact` vs `exact` on input-identical pairs.** The noise run is the
oracle job repeated on all its episodes, with digests `dN(j)` from its audit. Per episode in
oracle ∩ noise:
- `A` is the monotone IoU alignment of the oracle's and the noise run's rows (as for segments);
- `P = {(i, j) in A : dO(i) is not null and dO(i) == dN(j)}`, the pairs whose Captioner inputs
  were byte-identical in the two `exact` runs;
- `caption_mismatch_noise = #{(i, j) in P : cap(o_i) != cap(n_j)} / |P|`, and
  `caption_edit_distance_noise` the mean `ned` over `P`;
- an episode with `|P| = 0` is left out of the isolated noise means. A count-flip episode is
  **not** left out: its input-identical pairs count.

The noise `n` is the number of episodes with `|P| >= 1`; means are macro. The candidate is
conditioned on identical inputs, so the noise is conditioned the same way (equal bounds alone
are not enough). The selection keeps the segments whose TempSeg output was stable across the
two `exact` runs; they are plausibly easier, so this estimate leans **low**, which makes the
threshold stricter, not looser. If the oracle's or the noise run's audits lack digests, certify
stops; no bounds-plus-neighbours proxy is substituted without a new pre-registration.

`caption_similarity_divergence` (only with `--similarity module:function`: `1 - sim(a, b)`
with `sim` clipped to [0, 1], over the same slots) is **off by default** and not used by the
certified run. The hook is the caller's own code; nothing here touches the network.

### Reported-only (`metrics.reported`)

These are computed from the records and published, but they are never in `checks`, never part
of the verdict and never read by `policy_problems` or matching:

- **`e2e`**: the v3 end-to-end definitions, unchanged, comparing the end-to-end balanced job's
  (J1) caption records, on balanced TempSeg's own segmentation, against the oracle and against
  the noise run. The intervals are aligned as for segments, and per episode:
  `caption_mismatch = 1 - (exact matches over P) / slots`; `caption_edit_distance = (sum over P
  of ned + unmatched) / slots` (an unmatched slot counts 1); `count_mismatch = 1` if the two
  records have different numbers of intervals. The block records the means (over the
  count-matched episodes, `count_mismatch` over all), `n`, the count flips and the thresholds
  these would have had, each labelled `"reported_only": true`.
- **Isolated micro rate**: the pooled differing captions over all slots, for the candidate and
  the noise run.
- **Split by family**: `n-` vs `x-` episodes, and EgoDex vs WGO, for the isolated candidate
  (`--families`).
- **Second repeat**: the old 6-episode noise run vs the oracle under the isolated definition
  (run-to-run stability of the noise estimate; `--reported-repeat`).
- **Set-valued agreement**: per input-identical slot, the candidate equal to the oracle **or** to
  the noise run. Information only.

### Rewards

| Stage | Metric | Per episode |
|---|---|---|
| `reward.terminal` | `expected_score_abs_delta` | `|E[s]_A - E[s]_B|` (renormalized top-k expectation) |
| `reward.terminal` | `score_mismatch` | `1` if the decoded score differs |
| `reward.dense` | `per_step_abs_delta` | mean `|a_i - b_i|` over the common prefix of `per_step` |
| `reward.dense` | `rank_divergence` | `1 - rho`, Spearman rank correlation (average ranks for ties) over the common prefix. When rho is undefined (either side constant, or fewer than 2 steps): the mean `|a_i - b_i|`. So two equal constant series score 0, and one `2` in ten `3`s scores 0.1, not the maximum |
| `reward.dense` | `length_mismatch` | `1` if the `per_step` lengths differ |

Rubric has no certificate metric yet (it would need a distributional comparison).

## Pass rule

The noise floor is computed per stage: `noise_m = mean over noise episodes of m(exact, noise)`.
For the stage's (single) embodiment pool and each metric:

```
mean of m(exact, balanced)  <=  min(cap_m, max(absolute_floor_m, tolerance x max(noise_m, floor_m) + slack_m))
```

(`certkey.metric_threshold`). Version 4 leaves this formula, every knob and every floor
**unchanged**; it changes only what the caption metrics are computed on (the stage-isolated
definitions above). For `caption` that gives `1.5 x max(noise, 0.05) = 0.075` for
`caption_mismatch` unless the measured isolated noise exceeds 5%, and `1.5 x max(noise, 0.02)`
for `caption_edit_distance`.

**Which mean** (`per_metric.over`): `count_mismatch` is averaged over every paired episode; for
`segment` every other metric is averaged over the episodes whose subtask counts agree
(`count_matched`). For `caption` (v4) both metrics are averaged (`over: input_identical`) over every episode with at least
one slot (candidate) or one input-identical pair (noise). The same rule gives `noise_m`.

**Count flips are counted over the union.** For `segment` (and, before v4, `caption`), the `count_mismatch`
check is not made per group but once (`class: union`), over the candidate's episodes and the
cold witness's together: each witness episode is a sample of its own (the same episode, run in
another container). It passes while

```
flips  <=  round_half_up(threshold_count_mismatch x n)        (certkey.count_flips_allowed)
```

with `n` the candidate's plus the witness's paired episodes. **Rounding:** half up, so "about
one episode in ten" means the nearest whole episode: at 0.10, 1 flip of 10-14, 2 of 15-24.
At the floor, **2 flips in 18** (12 candidate + 6 witness, 11.1%) **pass**, 3 in 18 fail, and 2
in 14 fail. Without a witness the union is the candidate alone (1 in 12 passes, 2 in 12 fail).
The threshold is the floor only when the measured noise is low; see "The committed
certificate" below for what applied on the certified run. Judged per
group, one flip among the witness's 6 episodes (17%) would fail a certificate that one flip in
18 does not. The witness's **other** means (boundary error, IoU divergence, label disagreement,
caption mismatch and edit distance) are still checked **on their own** against the same
thresholds (`class: cold-witness`), so a systematically broken cold path fails even with no
flip. `metrics.count_flips` records the per-group figures (`groups: {<pool>, cold-witness}:
{n, flips}`) and the union's (`{n, flips, rate, threshold, allowed}`); matching recomputes the
union from them (`certkey.policy_problems`).

**Why.** With zero measured noise (an earlier H200 repeat gave one distinct text) the floors
decide, and `1.5 x 0.05 = 7.5%` made a single count flip in 12
episodes (`1/12 = 8.3%`) a `fail` for a correct but different engine (vLLM BF16 on L40S/H100 vs
HF on H200). A count flip also drags every interval and caption metric of that episode (an
extra interval leaves slots unmatched, and its captions follow), so those were fatal in the same
way. The rule now allows **up to about one episode in ten** to differ in subtask count
(`absolute_floor` 0.10: 1 in 12 passes, 2 in 12 fail), bounds those episodes by that check
alone, and keeps the **tight floors** below for the means over the episodes that agree. It stays
fail-closed for real regressions: at the floor, three count flips in 18 episodes (two in 12
without a witness), or a boundary error mean over 0.15 s with every count equal, in the
candidate or in the witness, fail.

**The committed certificate** (`segment`, `human`, L40S; version 3, now retired, kept here as
the worked example: the v4 segment rule is the same). The 10% is a floor, not the
threshold. The threshold is `max(0.10, 1.5 x max(noise, 0.05) + slack)`, and the noise run
measured `count_mismatch` noise of **1 flip in 6** episodes (0.1667). So the applied threshold
is `1.5 x 0.1667 = 0.25` (recorded as 0.250001), which allows **5 of the 18** union episodes to
flip. The run observed **1 of 18** (5.6%): 1 of the 12 candidate episodes, 0 of the 6 witness
episodes. Three flips would have passed this certificate too. Read the figures from
`metrics.count_flips.union` (`threshold`, `allowed`, `flips`, `n`), not from the floor.

### Policy (`certkey.POLICY`)

The knobs are bounded. `flora certify` refuses values outside the bounds, and matching refuses a
certificate that records them.

| Knob | Default | Bound |
|---|---|---|
| `tolerance` | 1.5 (PLAN.md §4) | [1, 2] |
| `min_samples` (candidate episodes per pool) | 10 | >= 10 |
| `min_noise_samples` | 5 | >= 5 |
| `max_missing` (oracle episodes without a candidate record: the spares) | 2 | [0, 2] |
| `slack_m` (`--slack`) | 0 | [0, `floor_m`] |

**Absolute noise floors** `floor_m`. A zero measured noise (two greedy runs that agree,
likely on short episodes) would otherwise demand a bit-identical engine on another GPU. Each
floor is below what a real regression costs:

| Metric | Floor | Why |
|---|---|---|
| `iou_divergence` | 0.02 | 1.5 x 0.02 = 3% of total IoU, less than one interval shifted by a tick on a 3-interval episode |
| `boundary_error_s` | 0.1 s | under half a TempSeg frame tick (4 fps = 0.25 s): a one-tick shift on every boundary still fails |
| `count_mismatch` | 0.05 | 1.5 x 0.05 = 7.5%, below the 0.10 absolute floor, so with low noise the floor decides (1 of 10-14, 2 of 15-24); measured noise above 0.0667 raises it |
| `label_disagreement` | 0.02 | as IoU |
| `caption_mismatch` | 0.05 | as `count_mismatch` for whole captions |
| `caption_edit_distance` | 0.02 | about one character in 30-50 per caption |
| `caption_similarity_divergence` | 0.02 | as edit distance |
| `expected_score_abs_delta` | 0.02 | 1.5 x 0.02 = 0.03, under the PLAN cap 0.05 |
| `score_mismatch` | 0.05 | as `count_mismatch` |
| `per_step_abs_delta` | 0.02 | one step in 50 off by one level |
| `rank_divergence` | 0.05 | rho >= 0.925 |
| `length_mismatch` | 0 | the step count is fixed by the video length |

Version 4 changed none of these knobs or floors. A rule change of any kind needs a new generator
version and a new pre-registration before any run ("Pre-registration and re-runs").

**Absolute floors** `absolute_floor_m` on the threshold itself: `count_mismatch` 0.10 (above).
The certificate records `noise_floor` and `absolute_floor` (and each check's `floor`,
`absolute_floor` and `over` in `per_metric`); matching refuses a certificate whose recorded
floors are not the policy's (`certkey.floor_problems`).

**Spares.** The certificate evaluates **every** episode both sides labelled,
never a chosen subset: the candidate paired with the oracle (`n`), the noise run paired with the
oracle, the witness paired with the oracle. So a failed episode is replaced by whichever spare
is present, deterministically. The golden/v2 plans run 12 oracle episodes for the 10
`min_samples` (2 spares, `max_missing`). Under v4 the noise run covers **all 12** oracle
episodes, and the cold witnesses are the 6 `n-` episodes (the selection's 5 plus one spare, for
the 5 `min_noise_samples`, which is still 5). For `caption` the isolated noise `n` counts the
episodes with at least one input-identical pair, so an episode where no Captioner input
repeated is not a noise sample. One missing balanced, noise or oracle episode still certifies
while >= 10 candidate and >= 5 noise (and witness) episodes pair. (The v3 certified plan ran
the noise on the 6 `n-` episodes only.)

**Absolute caps** from archive/PLAN-v3.md §2: terminal `E[s]` MAE <= 0.05 (`expected_score_abs_delta`).
PLAN's other bar, "boundary F1 at most 1 point below", is not enforced: no boundary-F1 metric
is computed yet.

### Verdict

In this order:
0. **Refused** (no certificate is written, `CertifyRefused`): missing or fake provenance and the
   binding checks above, and for `caption` the three isolation refusals,
   `segments_not_oracle`, `oracle_digests_missing` and `inputs_not_isolated` ("Binding checks",
   5). A refusal is not a verdict, so it cannot revoke a certificate.
1. **`fail`** if any check fails (`divergence_above_noise_floor`), or a present L0 fails
   (`l0_failed`: any request whose `prompt_token_ids` or `video_grid_thw` differ from NVIDIA's HF
   processor).
2. **`provisional`** if any of these holds:
   - no noise run (`no_noise_run`): checks have no threshold and cannot fail or pass;
   - noise episodes < `min_noise_samples` (`too_few_noise_samples`);
   - candidate episodes < `min_samples` (`too_few_samples`);
   - more than `max_missing` oracle episodes without a candidate record (`incomplete_coverage`);
   - no L0 for a video stage (`no_l0`). `segment` and `caption` need L0; rewards run the same
     NVIDIA vLLM engine as `exact`, so their L0 is `not_applicable`;
   - an L0 measured under other engine settings (`stale_l0`: the lab's baseline
     `engine_hashes` is not `certkey.engine_settings_hash` of the candidate's `vllm_engine`, or
     the lab predates the hash);
   - a candidate or witness record whose engine preempted while it ran (`engine_preempted`) or
     that does not say (`preemptions_unknown`);
   - for `segment`/`caption`, no cold witness (`no_witness`): no witness record paired with the
     oracle, or none of them with `compile_cache: cold:forced`. The evidence must cover a
     forced-cold container, so a certificate without one is never `certified` (the independent
     check gives the same verdict);
   - a witness with fewer than `min_noise_samples` episodes (`too_few_witness_samples`) or with
     some, not all, records not forced cold (`witness_not_forced_cold`);
   - a metric could not be evaluated (`metric_not_evaluable`).
3. **`certified`** otherwise.

The reported-only metrics (`metrics.reported`) never affect the verdict, in any direction.

**What `certified` needs**, per stage and GPU: binding provenance on every record; at least
10 candidate episodes of the pool, at most 2 oracle episodes unlabelled; a noise run pairing at
least 5 of the oracle's episodes (for `caption`, 5 with an input-identical pair); L0 passing for
`segment`/`caption` under the candidate's engine settings; for `segment`/`caption`, a cold
witness of at least 5 episodes, every record forced cold; for `caption`, a stage-isolated
candidate and witness that pass input identity on every Captioner request; no preempted or
unknown-preemption engine record; every check within threshold. **The six-episode `--plan full`
session (6 candidate episodes, a 6-episode noise run) is `provisional` by construction**: 6 is
below `min_samples` (10). The re-certification run (`scripts/wave_h.py --plan recertify`: 12
episodes, a noise run on all 12, a segment-only and a caption-only isolated cold witness on the
6 `n-` episodes) is sized to meet these requirements.

The engine limits behind "no preempted record" (`engines/offline_llm.py`,
[`balanced-engines.md`](balanced-engines.md)): the Captioner on the H100 runs `max_num_seqs`
**11**, fixed before the run for margin. Its worst-case request is **17,920 tokens** (the
16,811-token 3-view prompt, about 600 tokens of instruction and the 512-token budget), which
takes **26 blocks of 784 tokens** (`ceil(17920 / 784) + 3`). Eleven sequences need 286 blocks of
the estimated ~313, and the build check (`check_mamba_blocks`) fails closed below 286. TempSeg is
unchanged. yam 3-view prompts (~16.8k tokens) are longer than `max_num_batched_tokens` 16384,
so they are **always** prefilled in two chunks; that chunked path is what the yam certificate
measures.

**Preemptions.** A preempted GDN sequence recomputes its generated tokens through the
chunked-prefill kernel instead of the recurrent decode kernel, a numerics path the oracle and
noise runs never used, and timing-dependent. Records whose engine preempted are therefore **not
certified** rather than covered: covering them would vouch for a path that was measured only by
accident, and excluding them would drop episodes silently. The engine limits fit a full batch of
worst-case requests (`engines/offline_llm.py`: TempSeg `max_num_seqs` 48 on L40S, room for 55;
the Captioner 11 on H100 at 17,920 worst-case tokens, room for 12 in the estimated 313-block
pool), and the build fails when vLLM's blocks do not (`check_mamba_blocks`), so a certified run
should see none. (Before v4 the Captioner ran 16 against a 10,752-token worst case, which a
full yam batch would not have fit.) A preempted run is `provisional`; the pre-registered remedy
is a lower `max_num_seqs`, which re-keys every embodiment (the key has none).

### Warm and cold compiles

A warm container replays the publisher's Inductor autotune picks (`.best_config`); a cold one
benchmarks afresh and may pick other reduction kernels (`engines/compile_cache.py`,
"Determinism"). The cache is **not** in the key (a key per publish would make every cold
fallback an uncertified stack). Instead every `segment`/`caption` record carries
`provenance.compile_cache`, and the certificate's `compile_cache` block counts the candidate's
and the witness's caches and lists what its evidence `covers`. The certified plan
(`scripts/wave_h.py --plan certified`) makes that cover both: (d)'s job runs with one container
per kind, which compiles (cold, or from the lab's legacy cache) and publishes the cache every
later warm container loads, so its records are exactly the kernels warm production replays;
then the **cold witness** re-runs the same config on the 6 noise episodes with
`FLORA_COMPILE_CACHE=off`, one container per kind forced cold. The `recertify` plan splits the
witness by stage: a segment-only forced-cold job (J1w) for `segment`, and a caption-only
stage-isolated forced-cold job (J2w, the same config as the isolated candidate) for `caption`.
A certificate with a witness
therefore covers warm containers loading that publish and independent cold compiles; one
without covers only the caches it lists.

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
| `generator` | `{name: flora-certify, version: "4", framework}` (`certkey.GENERATOR_VERSIONS = ("4",)`: version 4 is the current rule; version 3 and earlier no longer match, and the drift guard reports them as `retired`) |
| `scoring` | `caption`: `{"caption": "stage_isolated"}`. `policy_problems` refuses a v4 caption certificate without it |
| `segments_from` | `caption`: `{sha256, oracle_stage_hash}` of the pinned oracle segment file the candidate captioned |
| `input_identity` | `caption`: for the candidate and the witness, `{candidate, witness}`, each `{pairs, equal, episodes, status, stage_hash, oracle_stage_hash}`; `policy_problems` requires `status: pass` for both |
| `generated_at` | UTC `YYYY-MM-DDTHH:MM:SSZ`; orders certificates for one key (below) |
| `verdict`, `reasons`, `unlocks` | `certified` / `provisional` / `fail`; machine reasons; whether it unlocks the preset |
| `preset`, `stage`, `kind`, `gpu` | what it certifies; `gpu` is the Modal GPU type (`H100!` is not `H100`) |
| `covered_pools` | the embodiment pool(s) measured |
| `stage_hash` | the candidate records' stage hash |
| `engine_spec_hash` | **the match key**, computed from the candidate records' `cert_key` |
| `engine`, `modes`, `logic`, `pins`, `derived_weights` | from the key: engine name, versions, image and fingerprint, patches and vLLM spec; output-affecting modes; logic version and certificate logic digest; NVIDIA pin revisions; derived-weight content digests |
| `oracle` | the oracle's `stage_hash`, `engine_spec_hash`, normalized `gpu` and `determinism` mode |
| `episodes` | `{count, digest, uid_sha256: [...]}`: sha256 of each episode uid, never the uid |
| `metrics` | per pool `{n, count_matched, means}`; noise and witness `{n, means, unpaired, episodes: {count, digest}}`; `count_flips` (`segment`; `caption` before v4): per group `{n, flips}` and over the union `{n, flips, rate, threshold, allowed}`; `coverage`; `missing`; the similarity hook's name; `reported` (`caption`): the reported-only block `{reported_only, micro, e2e, families, repeat, set_valued}`; the noise block also records its `pairs` and the candidate block its `slots` |
| `thresholds` | the knobs, `noise_floor`, `absolute_floor`, and per metric `{slack, floor, absolute_floor, cap, over, threshold}` |
| `preemptions`, `compile_cache` | `segment`/`caption`: candidate and witness records whose engine preempted / could not say; the compile caches the candidate and the witness ran, and what the evidence `covers` |
| `checks` | `[{class, metric, value, threshold, pass}]` (`class` is the pool, `cold-witness`, or `union` for the one `count_mismatch` check, which also carries `flips`, `n` and `allowed`). `policy_problems` refuses a v4 caption certificate whose checks name a reported metric or include `count_mismatch` |
| `l0`, `gate_a` | L0 status and counts; Gate A summary numbers |

**Metrics only.** No caption, prompt, rationale, metadata, episode id or dataset name ever enters
a certificate. A property test feeds random strings into every free-text field of the input
records and checks that none appears in the certificate. The record values carried are our own
and validated: stage hashes (64 hex), the key document (equal to this checkout's), the GPU
(normalized), the embodiment and determinism mode (enums). The digest JSONs carry hex digests
only, and only their pair counts enter the certificate.

## Pre-registration and re-runs

Fixed on **2026-10-23**, before the re-certification run, by the user's decisions on the
pre-registration of the stage-isolated caption certificate. These rules hold for the
`recertify` plan (human) and for the yam `certified` plan on the same v4 code and keys.

**Fixed before the run:**
- **Order A** for human: the noise run on all 12 episodes runs from the unchanged checkout
  **before** the v4 code lands, so the existing `exact` oracle (run `10221548f1b8`, step (b)) is
  reused and the noise run reproduces its stage hash and key.
- **Engine limits:** Captioner H100 `max_num_seqs` **11** (fixed for margin: 11 x 26 = 286
  blocks of the estimated ~313; `check_mamba_blocks` fails the build closed below 286),
  `WORST_CASE_TOKENS["captioner"]` **17,920** (26 blocks of 784 tokens); TempSeg unchanged; yam
  3-view prompts always prefill in two chunks of 16384, which the yam certificate measures.
- **Policy unchanged:** tolerance 1.5, `NOISE_FLOOR`, `ABSOLUTE_FLOOR`, `min_samples` 10,
  `min_noise_samples` 5, `max_missing` 2, slack 0, no similarity hook. The caption threshold is
  `1.5 x max(noise, 0.05)` = **0.075** unless the measured isolated noise exceeds 5%.
- **Generator v4**, `GENERATOR_VERSIONS = ("4",)`: every v3 certificate is retired, including
  the committed v3 segment certificate (see the status at the top).

**Outcome, committed in advance:**
- **`certified`**: publish after `scripts/cert_check.py` passes
  ([`certificate-check.md`](certificate-check.md)).
- **`fail`**: no re-roll. Either ship the hybrid (balanced `segment`, certified, with `exact`'s
  Captioner), or fund diagnostics toward a **new** stack (for example FP32 `dt_bias` or
  batch-invariant decoding) that gets its own pre-registration.
- **`provisional`** for an infrastructure reason or a preemption: re-run only the affected job.

**Re-runs.** One attempt per job. A job is re-run only for an infrastructure outcome: no records
or an infrastructure `EpisodeError`; more than `max_missing` missing episodes;
`engine_preempted` or `preemptions_unknown`; a killed app; or `inputs_not_isolated` fixed in the
harness, which re-runs the isolated caption jobs only and must not touch scoring code. Never
because of a metric value. A completed `flora certify` on complete evidence is final for its key.
Changing any rule needs a new generator version and a new pre-registration **before** any new
run.

**Power.** Twelve episodes and about 98 captions implement the pre-registered rule; they do not
give statistical confidence in either direction. The gate is a point comparison (a macro mean
against 0.075), not a hypothesis test: simulated at the human episode sizes, the probability of
passing is about 0.92 for a true isolated mismatch rate of 4%, 0.72 at 6%, 0.60 at 7% and 0.48 at
8%, so the verdict is close to a coin flip for true rates of 6-9%, and one caption in a
3-caption episode moves the mean by 0.028. Telling a true 5% from 7.5% at 80% power would take
about 600 captions (roughly 75 episodes). State this next to the verdict, whatever it is.

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
- `logic_version` and `logic_digest`: the **certificate logic digest**.

The certificate logic digest (`cert_logic_digest`) is not the config-hash logic digest
(`config/logic.py`, which is unchanged). It covers the kind's output-affecting modules (`config.logic.modules_for`) minus
`config/logic.py` (the registry), `config/hashing.py` and `validate/certkey.py` (versioned by
`KEY_SCHEMA`), each **normalized**: comments, docstrings, trailing whitespace and blank lines
removed (from the source text, so it is the same under every Python version). A comment or
docstring edit, a registry edit or a new module elsewhere does not invalidate certificates; a
code edit on the kind's path, or adding a module to its group, does.

## Matching

A certificate applies to a pool when **all** of these hold:
- `preset`, `kind` and `gpu` are equal;
- the config's `pipeline.embodiment` is in `covered_pools` (no embodiment: no match);
- `engine_spec_hash` equals the key of the stack that would run now. The caller passes the
  derived content digests; **without them a kind with derived weights matches nothing** (fail
  closed);
- a `certified`/`provisional` certificate passes `policy_problems`: known generator version,
  knobs within policy and, for `certified`, every check recomputed from its own noise means and
  knobs, its sample counts and coverage (so editing a body and recomputing its id does not
  help);
- the file's `id` is its content address and `generated_at` is set.

**Order and revocation.** Among the matching certificates for one key and pool, the newest
`generated_at` decides; on a tie `fail` beats `provisional` beats `certified`. So a newer `fail`
revokes an older `certified`, and a later `certified` (e.g. on more data) supersedes an older
`fail`. Only a deciding `certified` unlocks.

The stage hash is recorded but not matched: it also covers the input dataset, and a certificate
must apply to the same pool on other data (archive/PLAN-v3.md §9).

One lookup for every caller: `certkey.lookup_context(cfg, kinds, derived_content=...)`
gives embodiment, modes and content digests to:
- config validation (`parse_config(..., derived_content=...)`): refuses a non-`exact` preset
  unless every pool has a deciding `certified` certificate, or `allow_uncertified` is set. The
  opt-in is one boolean: a config whose every stage is covered needs none, and the refusal
  names each stage that is not covered and why, in plain words
  (`config.presets.refusal_message`), e.g. `segment on L40S: no certificate; embodiment yam:
  segment/caption certified for human only` (a reward stage appears only when its stack diverges
  from `exact`'s; see "Reference by identity"). Records say per stage whether
  they are certified, so a per-stage knob would add config without adding information;
- `plan_app(..., derived_content=...)` (`flora deploy`): the same check;
- `flora plan`: each non-`exact` option's per-pool status (`certified` / `reference` /
  `provisional` / `fail` / `none` / `unverified_weights`), and `recommended()` treats an all-certified option
  like the reference, so a cheaper certified `balanced` wins over `exact`.

`flora plan`, `flora doctor` and `flora deploy` print each stage's status in plain words
(`config.presets.coverage_label`): `certified (human, L40S)`, `not certified`, `certified for
human, not for yam`. `flora doctor` also says when `allow_uncertified: true` is no longer
needed. `flora results` shows each record's `certified` flag in plain words and the
certificate id (`provenance.certificate`). `scripts/wave_h_finalize.py` rewrites the README's
`wave-h:certified` and `wave-h:allow-uncertified` blocks from the certificates it writes.

### Reference by identity

A (preset, kind) runs exactly what `exact` runs when its live key document
(`key_document(preset, kind, gpu, ...)`, with the config's modes and derived content) equals
`exact`'s for the same kind once the `preset` field alone is removed, and the pool GPU is
`exact`'s pool GPU for that kind. Then it is the **reference**, like `exact` itself: no
certificate is needed, **unless** the deciding certificate that matches its live key (same
preset, kind, GPU, embodiment pool and key) is a `fail`: a failure measured
under the very key overrides the identity argument, and the kind then needs a `certified`
certificate like any other. The rule is `certkey.reference_identical` / `reference_kinds`; the config gate
and the deploy factory get it through `config.presets.find_certificates` (a
`reference:exact-identical` entry, no file) and `certkey.status` reports `reference` (or `fail`
when a matching fail overrides it), so plan, doctor and deploy say `reference (NVIDIA's own
engine, no certificate needed)`.

Today this holds for `balanced`'s `reward.terminal`, `reward.dense` and `reward.rubric` on
`H100!`: same engine (`nvidia-vllm-offline`), image, pins, modes and logic. It is computed
from the key documents, never listed, so it fails closed: a different engine argument, image,
pin, run mode, logic digest or GPU (or a key that cannot be computed) makes the kind need a
certificate again. `segment` and `caption` are never reference by identity: `exact` has no such
kinds (it runs them as `segcap` on NVIDIA's HF code).

Records: the reward stage stamps `certified` from NVIDIA's envelope alone in both presets
(`stages/reward.py`, `certification`), with `provenance.certificate` null, which is what this
rule says while it holds. Applying the rule at stamping time as well (so a divergent reward
stack's records say `uncertified_stack`) belongs in `workers_impl/certstamp.py`
(`CertStamp.apply`, which has the record's `cert_key`), calling `certkey.reference_identical`.

The CLI and SDK pass the content digests: `flora deploy`, `flora run`, `flora plan`, `flora
doctor` and `sdk.submit` load the config with `config.receipts.load_certified`. That takes the
digest from the `pins.DERIVED_PINS` entry of each derived weight recorded under `weights.derived`
(`flora prepare`). A weight with no recorded receipt, or no pinned digest, is left out, and its
kind shows `unverified_weights` (fail closed).

## Usage

```bash
flora certify -c flora.yaml --exact oracle/segment.jsonl --balanced bal/segment.jsonl \
  --noise noise/segment.jsonl --witness witness/segment.jsonl \
  --l0 reports/wave-h/balanced-tempseg.json \
  --derived-content tempseg_merged=<sha256> --generated-at <run time, YYYY-MM-DDTHH:MM:SSZ>
```

A v4 `caption` certificate takes the stage-isolated records, the pinned oracle segment file and
the digest JSONs; the end-to-end records go in only as the reported-only block:

```bash
flora certify -c flora.yaml --stage caption \
  --exact oracle/caption.jsonl --balanced iso/caption.jsonl \
  --noise noise/caption.jsonl --witness iso-witness/caption.jsonl \
  --oracle-segments oracle-segments/<oracle stage hash>.jsonl \
  --digests exact=digests/exact.json --digests noise=digests/noise.json \
  --digests balanced=digests/balanced.json --digests witness=digests/witness.json \
  --reported-e2e e2e/caption.jsonl \
  --l0 reports/wave-h/balanced-captioner.json \
  --derived-content captioner_bf16=<sha256> --generated-at <run time, YYYY-MM-DDTHH:MM:SSZ>
```

Optional, reported only: `--reported-repeat old-noise/caption.jsonl --digests
repeat=digests/repeat.json`, `--families families.json`. Without
`--oracle-segments`, `--digests exact=` and `--digests balanced=`, a caption certification
refuses.

Without `--witness` (records from a container forced cold) `segment` and `caption` are at best
`provisional` (`no_witness`); `--generated-at` with the run's own time makes a re-run reproduce
the same certificate id. `scripts/wave_h_finalize.py` passes both. (From a standalone lab run
the L0 file is `reports/balanced-lab/<run>-tempseg.json`.) It prints
the verdict per stage, the failing checks and where each certificate was written. Next
to each certificate it writes a freeze lock (`freeze/<id>.json`: the full key document and
per-module hashes) for the drift report ([`certificate-freeze.md`](certificate-freeze.md)). It
exits 1 if any stage fails or the records are refused (`certified: false`), 2 on bad input.
