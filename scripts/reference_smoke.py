"""WS-D smoke: stage the public samples, then label them with NVIDIA's pipeline on one H200.

    uv run python scripts/reference_smoke.py [--only EPISODE ...] [--summary PATH] [--force-stage]

Runs the ephemeral ``flora-reference`` app (``app.run()``; nothing is deployed). Both models load
once in ``ReferenceWorker``; every episode's full result stays on the ``flora-golden`` Volume.
Locally this prints, and optionally writes, numbers only (no captions or prompt hashes).
Costs money: one H200 container for model loading plus the episodes (see docs/reference-smoke.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import modal

from flora_deploy.modal_app.reference_app import ReferenceWorker, app, stage_samples
from flora_deploy.stages.reference import summary_row

COLUMNS = (
    "episode", "views", "profile", "duration_s", "windows", "segments", "captions",
    "probe_s", "segment_s", "caption_s", "seg_out_tok", "cap_out_tok", "peak_gib",
)
DEADLINE_S = 40 * 60  # stop submitting episodes after this much wall-clock


def table(rows: list[dict]) -> str:
    cells = [[str(r.get(c, "")) for c in COLUMNS] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(COLUMNS)]
    line = lambda vals: "  ".join(v.rjust(w) for v, w in zip(vals, widths, strict=True))  # noqa: E731
    return "\n".join([line(COLUMNS), line(["-" * w for w in widths]), *map(line, cells)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", action="append", default=[], help="episode name(s) to run")
    p.add_argument("--summary", type=Path, help="write the numbers-only summary JSON here")
    p.add_argument("--force-stage", action="store_true", help="re-stage the samples")
    p.add_argument("--stage-only", action="store_true", help="stage samples, skip the GPU")
    args = p.parse_args(argv)

    t0 = time.monotonic()
    rows, errors = [], []
    with modal.enable_output(), app.run():
        manifest = stage_samples.remote(force=args.force_stage)
        episodes = [e for e in manifest["episodes"] if not args.only or e["episode"] in args.only]
        print(f"staged {len(manifest['episodes'])} episodes; running {len(episodes)}")
        if args.stage_only:
            print(json.dumps(manifest, indent=2))
            return 0
        worker = ReferenceWorker()
        load = worker.report.remote()
        print("load report:", json.dumps(load, indent=2))
        for ep in episodes:
            if time.monotonic() - t0 > DEADLINE_S:
                errors.append({"episode": ep["episode"], "error": "deadline: not submitted"})
                continue
            try:
                result = worker.annotate.remote(ep["videos"], ep["profile"], episode=ep["episode"])
            except Exception as e:  # noqa: BLE001 - report and continue with the next episode
                errors.append({"episode": ep["episode"], "error": f"{type(e).__name__}: {e}"})
                continue
            row = summary_row(ep["episode"], result)
            row["stored_at"] = result["flora_reference"].get("stored_at")
            rows.append(row)
            print(table([row]))
        final = worker.report.remote()

    print()
    print(table(rows))
    for e in errors:
        print("FAILED", e["episode"], e["error"])
    summary = {
        "load": load,
        "final_gpu": final["gpu_now"],
        "container_age_s": final["container_age_s"],
        "episodes": rows,
        "errors": errors,
        "local_wall_s": round(time.monotonic() - t0, 1),
    }
    print(f"container age {final['container_age_s']}s; local wall {summary['local_wall_s']}s")
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
