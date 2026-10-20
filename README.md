# flora-deploy

Deploy NVIDIA's [Flora](https://research.nvidia.com/labs/gear/flora/) robotics labeling models
(temporal segmentation, contextual captioning, outcome, dense and rubric rewards) at scale on
[Modal](https://modal.com), inside your own workspace.

> **Status: pre-alpha (Phase 0).** Nothing here is usable for labeling yet. See
> [`docs/PLAN.md`](docs/PLAN.md) for the design and roadmap.

## What it will do

- **Deploy with one command** into your own Modal account. There is no cluster to manage, and it
  scales to zero.
- **Run resumable batch jobs** over LeRobot datasets or MP4 directories, with a cost estimate up
  front and a hard budget cap.
- **Offer presets** that trade cost against fidelity:
  - `exact`: NVIDIA's reference code, unchanged.
  - `balanced`: faster engines, certified against that reference.
- **Serve an online API** for segmentation, captioning and rewards (later phase).

## License

The framework code is licensed under Apache-2.0.

**The Flora models are research-only** (NVIDIA OneWay Noncommercial License). This repository
contains no NVIDIA weights, prompts or noncommercially licensed code. Read [`NOTICE.md`](NOTICE.md)
before use.

## Development

```bash
uv sync
uv run pytest
uv run flora prepare --dry-run   # list the pinned Hugging Face files and their sizes (no downloads)
```
