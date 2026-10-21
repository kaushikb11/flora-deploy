"""Gate C local entrypoint: ``uv run modal run scripts/gatec.py [--max-requests 2] [--quota 6]``.

Stages the public sample video into the ``flora-golden`` Volume (``samples/wse/``) if missing,
then (ephemeral app ``flora-gatec``):

1. ``build_part`` on CPU containers, in parallel: the builder cases on synthetic frames and on
   the sample (``--max-requests`` calls per case and source), and the staged golden episodes
   (``golden/v1/``, up to ``--quota`` calls per pool and kind);
2. ``run_gate_c`` on one L40S per processor kind (TempSeg, Captioner).

Writes the JSON results outside the repo and prints the Markdown tables for ``docs/gate-c.md``.
The results hold token counts, positions, grids and digests only (no prompt or task text).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

from flora_deploy.layout import GOLDEN_VOLUME
from flora_deploy.modal_app.gatec_app import SAMPLE_DIR, app, build_part, run_gate_c
from flora_deploy.validate.gatec import case_verdicts, render_report
from flora_deploy.validate.gatec_driver import GOLDEN_POOLS

# Big Buck Bunny, 10 s, 640x360, H.264, 30 fps CFR (c) Blender Foundation, CC BY 3.0.
SAMPLE_NAME = "bbb_360_10s.mp4"
SAMPLE_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
SAMPLE_SHA256 = "77145c94c11f3754207499158df22406e1fe7635553c1c86dc5e881dfeb32016"
L40S_USD_PER_H = 1.95
KINDS = ("tempseg", "captioner")


def stage_sample() -> None:
    import modal

    vol = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)
    try:
        present = {Path(e.path).name for e in vol.listdir(SAMPLE_DIR)}
    except Exception:  # directory missing
        present = set()
    if SAMPLE_NAME in present:
        print(f"sample present: {SAMPLE_DIR}/{SAMPLE_NAME}")
        return
    req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent": "curl/8.7.1"})
    data = urllib.request.urlopen(req, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SAMPLE_SHA256:
        raise RuntimeError(f"sample sha256 {digest} != expected {SAMPLE_SHA256}")
    with vol.batch_upload() as batch:
        batch.put_file(io.BytesIO(data), f"{SAMPLE_DIR}/{SAMPLE_NAME}")
    print(f"staged {SAMPLE_DIR}/{SAMPLE_NAME} ({len(data)} bytes, sha256 {digest})")


@app.local_entrypoint()
def main(max_requests: int = 2, quota: int = 6, golden: bool = True,
         out: str = "~/.cache/flora-deploy/gatec/results.json"):
    stage_sample()
    t0 = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    parts = ["cases:synthetic", "cases:sample"]
    if golden:
        parts += [f"golden:{pool}" for pool in GOLDEN_POOLS]
    calls = [build_part.spawn(kind, part, run_id, SAMPLE_NAME, max_requests, quota)
             for kind in KINDS for part in parts]
    builds = [c.get() for c in calls]
    build_wall = time.time() - t0
    for b in builds:
        print(f"built {b['kind']:9s} {b['part']:26s} {b['n']:3d} requests in {b['seconds']}s"
              + (f"  errors {b['errors']}" if b["errors"] else ""))
    gpu_calls = {kind: run_gate_c.spawn(kind, [b["path"] for b in builds
                                              if b["kind"] == kind and b["n"]])
                 for kind in KINDS}
    results = {kind: call.get() for kind, call in gpu_calls.items()}
    wall = time.time() - t0
    gpu_s = sum(r["timings"]["total_s"] for r in results.values())
    path = Path(out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(results=results, builds=builds, wall_s=wall,
                                    build_wall_s=build_wall), indent=1, default=str))
    rows = [row for r in results.values() for row in r["rows"]]
    for kind, r in results.items():
        print(f"\n== {kind}: {r['gpu']} {r['versions']} timings {r['timings']} "
              f"vLLM pixel dtypes {r['vllm_pixel_dtypes']}")
        print("video processor attr diff vs NVIDIA:", r["video_processor_attr_diff"] or "none")
    print("\n" + render_report(rows))
    verdict = case_verdicts(rows, "wrapped")
    print("wrapped verdict:", "PASS" if verdict and all(verdict.values()) else "FAIL", verdict)
    print(f"wall {wall:.0f}s (build {build_wall:.0f}s); in-function GPU time {gpu_s:.0f}s "
          f"(≈${gpu_s / 3600 * L40S_USD_PER_H:.2f} at ${L40S_USD_PER_H}/h, excl. container start)")
    print(f"results: {path}")
