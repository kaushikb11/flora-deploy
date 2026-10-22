# lerobot_small

The `balanced` preset on [`lerobot/aloha_static_cups_open`](https://huggingface.co/datasets/lerobot/aloha_static_cups_open)
(MIT; 50 episodes of about 8 s, 4 cameras, about 0.5 GB), pinned to a full commit. Three cameras
are used: `cam_high` as the main view and the two wrist cameras. `cam_low` is not, because the
`yam` embodiment takes exactly three views and `cam_low`, a second low overview, has no YAM role
(flora's own test set drops it the same way). ALOHA is not YAM hardware; it stands in for it here.

**Embodiment: `yam`** (bimanual ALOHA arms standing in for NVIDIA's `yam` embodiment; camera
keys from `meta/info.json` at the pinned revision).

Set `output.sink` to your bucket or Volume, then run `flora doctor -c flora.yaml` and follow the
"Label your own data" steps in the [top-level README](../../README.md#label-your-own-data).
`flora plan` calls this config "partly certified": the rewards are NVIDIA's own engine (records
say `certified: true`), while segment and caption have no certificate for `yam` (`certified:
false`), so `flora export` needs `--allow-uncertified`.

**Estimated cost: about $2 ($1.40-$3.30)** for the labelling job (segment, caption,
terminal and dense rewards), from `flora plan --video-hours 0.11 --episodes 50`. That is the
all-in estimate: GPU work plus each container's startup, from rates measured on one small run
(±50%). It does not include the one-time `flora prepare` and weight conversions.
`budget_usd: 15` caps the job either way.
