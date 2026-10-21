"""``flora-prep``: download and verify every pinned snapshot into the ``flora-weights`` Volume.

Run through ``flora prepare`` (``app.run()`` + ``prepare_pin.map(keys)``). Importing this module
creates lazy Modal objects only; nothing contacts Modal until the app runs.

Sizing for the largest pin (Captioner, ~102 GiB in 11 shards):
- ``cpu=4.0`` physical cores (8 vCPU). They feed hf_xet's parallel high-performance download and
  hash up to 8 files at once. It is billed whether used or not, so it stays modest; the job is
  network- and Volume-bound, not CPU-bound.
- ``memory=16 GiB``: hf_xet buffers in high-performance mode, plus 8 x 8 MiB hash buffers.
- No ``ephemeral_disk`` override. Files stream straight onto the Volume
  (``local_dir`` with its ``.cache`` resume metadata). Only hf_xet's chunk cache uses local disk
  (``HF_HOME=/tmp/hf``), well inside the default 512 GiB quota. An explicit request would be
  billed as memory at 20:1.
- ``timeout=6 h`` per pin with 2 retries. A retry resumes the partial download. A pin that still
  fails verification after its in-call repair raises, so the retries apply to it too.
"""

from __future__ import annotations

import modal

from flora_deploy.layout import WEIGHTS_MOUNT, WEIGHTS_VOLUME
from flora_deploy.modal_app.images import cpu_image

app = modal.App("flora-prep")
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True, version=2)

CPU_CORES = 4.0
MEMORY_MIB = 16 * 1024
TIMEOUT_S = 6 * 3600


@app.function(
    image=cpu_image(),
    volumes={str(WEIGHTS_MOUNT): weights},
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
    env={"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/tmp/hf"},
)
def prepare_pin(key: str, reverify: bool = False, provenance: dict | None = None) -> dict:
    """Download, verify and receipt one pin; skips if a reusable receipt exists (unless
    ``reverify``). ``provenance`` is the CLI's git commit info, recorded in the receipt."""
    from flora_deploy.prep.run import prepare_snapshot

    return prepare_snapshot(key, root=WEIGHTS_MOUNT, commit=weights.commit, reverify=reverify,
                            provenance=provenance)
