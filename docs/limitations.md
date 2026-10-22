# What flora-deploy doesn't do yet

The gaps we know about, so you can judge whether flora-deploy fits your data. The two tables are
generated from the repository's certificates and rates, so they change when either does.

## Accuracy certificates

`exact` runs NVIDIA's own pipeline and is the reference. `balanced` runs faster engines, so each
of its segmentation and captioning stages needs a certificate for your embodiment and GPU. Rewards
run NVIDIA's own engine in both presets, so they count as the reference and need none.

<!-- limitations:certification -->
| Embodiment | `segment` (L40S) | `caption` (H100!) | `reward.terminal` (H100!) | `reward.dense` (H100!) | `reward.rubric` (H100!) |
|---|---|---|---|---|---|
| `human` | not yet | not yet | reference | reference | reference |
| `yam` | not yet | not yet | reference | reference | reference |
| `yam32` | not yet | not yet | reference | reference | reference |
| `dex` | not yet | not yet | reference | reference | reference |
| `tianji` | not yet | not yet | reference | reference | reference |

Nothing is certified yet, so `balanced` segmentation and captioning need `allow_uncertified: true`
and their records say `certified: false`.
<!-- /limitations:certification -->

Certification sets are staged for `human` ([v2](golden-manifest-v2.md)) and `yam`
([v3-yam](golden-manifest-v3-yam.md)); the `human` run comes first.

## What "certified" promises, and what it doesn't

A certificate is a statistical claim about the episodes it was measured on: `balanced` differs
from `exact` no more than 1.5 times as much as `exact` differs from itself between two runs
(NVIDIA's pipeline is not bit-for-bit repeatable on a GPU). It is not a promise of identical output.
For example, `balanced` may split up to 10.0% of episodes into a different number of subtasks (more
when NVIDIA's own two runs disagree more often), counted over every episode measured, warm and
cold containers together, to the nearest episode: 2 of 18 pass. Boundaries and captions must
still agree closely on average, in the cold container on its own too. Each certificate records
its exact thresholds.
It covers only the embodiment, GPU and software it was measured on. Details:
[`certification.md`](certification.md).

## Costs: measured or estimated

<!-- limitations:costs -->
| Figure `flora plan` uses | Status |
|---|---|
| GPU work, `segment` | measured on one job, ±50% |
| GPU work, `caption` | measured on one job, ±50% |
| GPU work, `reward.terminal` | measured, ±100% |
| GPU work, `reward.dense` | measured, ±100% |
| GPU work, `reward.rubric` | estimate (not measured), ±100% |
| Startup, `segment`, cold | measured on one container, ±50% |
| Startup, `caption`, cold | measured on one container, ±50% |
| Startup, rewards | measured, ±50% |
| Startup, `segment`, warm | estimate (not measured), ±60% |
| Startup, `caption`, warm | estimate (not measured), ±60% |
<!-- /limitations:costs -->

A figure from one job carries a ±50% band until the [scale test](scale-test.md) measures more; a
±100% band means a rough estimate. `flora plan` labels each option with what its price rests on.
Where the money goes: [`costs.md`](costs.md).

## Never run on a GPU

`reward.rubric` has only run in CPU tests with stand-in models. Its first GPU run is prepared
([`scripts/rubric_smoke.py`](../scripts/rubric_smoke.py)) but has not happened yet.

## Not supported

- **`gs://` and `r2://` outputs** are refused. Use `s3://<bucket>` with your provider's S3 endpoint
  in `AWS_ENDPOINT_URL`, set in the `flora-bucket` Modal Secret and in your shell.
- **`.mkv` and `.webm` videos** are refused: only MP4 and MOV files are read. Convert them first.
- **`dex`, `tianji` and `yam32`** have no certification path yet: there are too few suitable public
  episodes to build a certification set. `balanced` segmentation and captioning run on them only
  with `allow_uncertified: true`; `exact` and rewards are unaffected.
- **No interactive or online API** yet: jobs run in batches. There is a design
  [proposal](interactive-design.md).
- **No FP8 or quantized models**, by design: the engines refuse quantization and other settings
  that change the numbers, because no certificate could cover them.

## Known approximations

- **Three-camera costs.** `balanced` rates were measured on single-camera episodes, and `flora
  plan` applies them to three-camera data too. Those prompts are longer, so expect more; scaling
  the rate by prompt length, as our own three-camera estimates do, is an assumption.
- **Warm starts.** `flora plan` prices a stage's startup warm only when a compile cache is saved
  for it, or with `--assume-warm`; the table above says whether those times are measured yet.
- **No H200 upgrades.** Captioning and rewards ask Modal for `H100!`, which stops Modal from
  moving them to an H200. That keeps every container on the GPU a certificate names.

## Licensing

The Flora models are for non-commercial research only: see [`NOTICE.md`](../NOTICE.md).
