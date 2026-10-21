# Gate certificates

Each `cert_<hash>.json` here certifies one shard kind of a non-`exact` preset on one GPU type,
for the embodiment pools it measured. `flora certify` writes them (`validate/certify.py`).
`find_certificates` (`config/presets.py`, match rule in `validate/certkey.py`) reads them. See
`docs/certification.md`.

- **Metrics only.** A certificate holds divergence metrics, thresholds, hashes and pins. It never
  holds NVIDIA text: no captions, prompts, rationales, episode ids (they are hashed) or dataset
  names.
- **Content-addressed.** The file name is the `id`, and the `id` is the sha256 of the body. A
  hand-edited file is ignored. Matching also recomputes every check from the body's own metrics
  and refuses knobs outside `certkey.POLICY`.
- **Scoped.** A certificate applies only to the same preset, shard kind, GPU and covered
  embodiment pool, and to the same `engine_spec_hash`, which was computed from the records'
  own provenance: engine identity and image fingerprint, the vLLM engine spec (incl. `dt_bias`
  and per-GPU limits), NVIDIA and decode pins, derived-weight content digests, output-affecting
  modes, the logic version and the kind's certificate logic digest. Change any of these and it
  stops applying, so the stack must be re-certified.
- **Only `certified` unlocks a preset.** A `provisional` certificate (no noise run, too few
  samples, incomplete coverage or no L0) is shown by `flora plan`, but its stack still needs
  `allow_uncertified`. A `fail` certificate is never written here by default; if one is, and it
  is newer than a `certified` one for the same key, it revokes it.
