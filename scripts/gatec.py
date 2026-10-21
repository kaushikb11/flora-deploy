"""Gate C local entrypoint: ``uv run modal run scripts/gatec.py [--max-requests 2] [--quota 6]``.

Stages the public sample video into the ``flora-golden`` Volume (``samples/wse/``) if missing,
then (ephemeral app ``flora-gatec``):

1. ``build_part`` on CPU containers, in parallel: the builder cases on synthetic frames and on
   the sample (``--max-requests`` calls per case and source), and the staged golden episodes
   (``golden/v1/``, up to ``--quota`` calls per pool and kind);
2. ``run_gate_c`` on one L40S per processor kind (TempSeg, Captioner).

Writes the JSON results outside the repo and prints the Markdown tables for ``docs/gate-c.md``.
The results hold token counts, positions, grids and digests only (no prompt or task text).

The verdict (review 5 M-A) applies every criterion: the golden ``--quota`` per pool and kind
(checked against the golden selection before anything is spent: a pool that cannot reach it is
reported up front and FAILS), and the HF reference image (build_part runs it in ``img_vllm``,
so C2 needs an ``img_hfref`` cross-check, which does not exist yet: the gate FAILS on C2
until it does). ``--rescore PATH`` re-scores a recorded results file under the same rules
without running anything on a GPU: every row's pass/fail is recomputed from its raw fields
(``validate.gatec.rescore_row``; a stored ``passed`` is never trusted), and a row that lacks a
criterion's data (e.g. rows recorded before ``placeholders_ok`` existed) FAILS it.
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
from flora_deploy.validate.gatec import render_report, results_provenance
from flora_deploy.validate.gatec_driver import golden_capacity, run_parts, run_verdict
from flora_deploy.validate.golden import GOLDEN_PREFIX, MANIFEST_NAME

# Big Buck Bunny, 10 s, 640x360, H.264, 30 fps CFR (c) Blender Foundation, CC BY 3.0.
SAMPLE_NAME = "bbb_360_10s.mp4"
SAMPLE_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
SAMPLE_SHA256 = "77145c94c11f3754207499158df22406e1fe7635553c1c86dc5e881dfeb32016"
L40S_USD_PER_H = 1.95
KINDS = ("tempseg", "captioner")


def golden_manifest() -> dict:
    import modal

    vol = modal.Volume.from_name(GOLDEN_VOLUME, version=2)
    return json.loads(b"".join(vol.read_file(f"{GOLDEN_PREFIX}/{MANIFEST_NAME}")))


def preflight(capacity: dict[str, dict], quota: int) -> None:
    for pool, c in capacity.items():
        if not c["satisfiable"]:
            print(f"WARNING: golden:{pool} has {c['episodes']} episode(s), at most "
                  f"{c['max_requests']} requests < quota {quota}: the gate will FAIL for this "
                  "pool until the golden selection gains episodes")


def print_verdict(verdict: dict) -> None:
    print("coverage:", verdict["coverage"])
    for problem in verdict["problems"]:
        print("problem:", problem)
    for warning in verdict["warnings"]:
        print("warning:", warning)
    print("wrapped verdict:", "PASS" if verdict["passed"] else "FAIL",
          f"failed cases {verdict['failed_cases']}" if verdict["failed_cases"] else "")


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
         out: str = "~/.cache/flora-deploy/gatec/results.json", rescore: str = ""):
    capacity = golden_capacity(golden_manifest(), quota) if golden else None
    if capacity:
        preflight(capacity, quota)
    if rescore:  # re-score a recorded run under the current criteria; nothing runs
        doc = json.loads(Path(rescore).expanduser().read_text())
        has_golden = any(b["part"].startswith("golden:") for b in doc["builds"])
        print_verdict(run_verdict(doc["builds"], doc["results"], golden=has_golden,
                                  quota=quota, capacity=capacity))
        return
    stage_sample()
    t0 = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    provenance = {"run_id": run_id, **results_provenance(cwd=str(Path(__file__).parent))}
    parts = run_parts(golden)
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
    verdict = run_verdict(builds, results, golden=golden, quota=quota, capacity=capacity)
    path.write_text(json.dumps(dict(provenance=provenance, verdict=verdict, results=results,
                                    builds=builds, quota=quota, capacity=capacity,
                                    wall_s=wall, build_wall_s=build_wall),
                               indent=1, default=str))
    rows = [row for r in results.values() for row in r["rows"]]
    for kind, r in results.items():
        print(f"\n== {kind}: {r['gpu']} {r['versions']} timings {r['timings']} "
              f"vLLM pixel dtypes {r['vllm_pixel_dtypes']}")
        print("video processor attr diff vs NVIDIA:", r["video_processor_attr_diff"] or "none")
    print("\n" + render_report(rows))
    print_verdict(verdict)
    print(f"git {provenance['git_sha']}{' (dirty)' if provenance['git_dirty'] else ''}, "
          f"images {provenance['image_fingerprints']}")
    print(f"wall {wall:.0f}s (build {build_wall:.0f}s); in-function GPU time {gpu_s:.0f}s "
          f"(≈${gpu_s / 3600 * L40S_USD_PER_H:.2f} at ${L40S_USD_PER_H}/h, excl. container start)")
    print(f"results: {path}")
