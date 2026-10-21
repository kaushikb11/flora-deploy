# Gate certificates

Each `cert_<hash>.json` here certifies one shard kind of a non-`exact` preset on one GPU type.
`flora certify` writes them (`validate/certify.py`). `find_certificates`
(`config/presets.py`, match rule in `validate/certkey.py`) reads them. See
`docs/certification.md`.

- **Metrics only.** A certificate holds divergence metrics, thresholds, hashes and pins. It never
  holds NVIDIA text: no captions, prompts, rationales, episode ids (they are hashed) or dataset
  names.
- **Content-addressed.** The file name is the `id`, and the `id` is the sha256 of the body. A
  hand-edited file is ignored.
- **Scoped.** A certificate applies only to the same preset, shard kind and GPU, and to the same
  `engine_spec_hash`: engine identity, image fingerprint, NVIDIA pins, output-affecting modes, the
  logic version and the kind's logic digest. Change any of these and it stops applying, so the
  stack must be re-certified.
- **Only `certified` unlocks a preset.** A `provisional` certificate (no noise run, too few
  samples, incomplete coverage or no L0) is shown by `flora plan`, but its stack still needs
  `allow_uncertified`. A `fail` certificate is never written here by default.
