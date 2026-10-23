# Certification: what `certified` means

Every record flora-deploy writes says whether it is `certified`. This page explains what that
field means, why `balanced` records say `certified: false` today, and what
`allow_uncertified` changes.

## `exact` is the reference

`exact` runs NVIDIA's own Flora code, unchanged, with model inputs proven identical to NVIDIA's
pipeline. It is the reference: it needs no certificate, and its records are `certified: true`
whenever the episode is inside NVIDIA's input limits.

`balanced` serves the same models with vLLM on cheaper GPUs. That can change the numbers a
little, so each `balanced` segment and caption stage needs an **accuracy certificate**: a
measurement showing that `balanced` differs from `exact` no more than `exact` differs from
itself between two runs. A certificate covers one stage, GPU and embodiment.

## Reference by identity

`balanced`'s reward stages (`reward.terminal`, `reward.dense`, `reward.rubric`) run NVIDIA's own
engine, exactly as `exact` does. They count as the reference and need no certificate. If any
part of that stack changes, they need a certificate again.

## The `certified` field

A record is `certified: true` only when both hold:

- the episode is inside NVIDIA's input limits (length, size, frame rate, cameras), and
- its stage is the reference or has a matching certificate.

Otherwise it is `certified: false`, and `uncertified_reason` says why, for example
`uncertified_stack` (no certificate for this stage) or `outside_reference_envelope` (the episode
is outside NVIDIA's limits). Records with `certified: false` are still in your outputs; filter on
the field if you want only certified labels. [`output-format.md`](output-format.md) lists every
reason.

Until a certificate covers them, `balanced` segment and caption records say `certified: false`.
Which stages are covered today, per embodiment, is in
[`limitations.md`](limitations.md#accuracy-certificates).

## `allow_uncertified`

A `balanced` config with a stage that has no certificate is refused unless you opt in:

- `allow_uncertified: true` in `flora.yaml`, or `--allow-uncertified` on `flora deploy` and
  `flora run`, lets the job run. Its uncovered stages write `certified: false`.
- `flora export` refuses uncertified records unless you pass `--allow-uncertified` (the
  `flora.yaml` key doesn't carry over). The choice is recorded in the export's metadata.

`flora plan` labels each option `reference`, `certified`, `partly certified` or `uncertified`,
and prints one `certificates:` line per option with each stage's status. `flora doctor` shows
the same status for your config.

Contributors: what a committed certificate freezes, and the CI check that guards it, are in
[`certificate-freeze.md`](certificate-freeze.md).
