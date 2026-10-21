"""WS-D smoke: stage the public samples, then label them with NVIDIA's pipeline on one H200.

    uv run python scripts/reference_smoke.py [--only EPISODE ...] [--summary PATH] [--force-stage]

W3-A concurrency experiment (docs/exact-concurrency.md), one container, one model load:

    uv run python scripts/reference_smoke.py --concurrency 4 --sequential --measure \\
        [--cpu 8] [--torch-threads N] [--repeat K] [--compare-to SUMMARY.json] --summary PATH

``--sequential`` first runs every episode with concurrency 1, then all of them with
``--concurrency`` threads, and compares NVIDIA's full result dicts (minus wall-clock fields).

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

from flora_deploy.modal_app.reference_app import (
    HARD_LIFETIME_S,
    ReferenceWorker,
    app,
    stage_samples,
)
from flora_deploy.stages.reference import compare_runs, summary_row

COLUMNS = (
    "episode", "views", "profile", "duration_s", "windows", "segments", "captions",
    "probe_s", "segment_s", "caption_s", "seg_out_tok", "cap_out_tok", "peak_gib",
)
DEADLINE_S = 40 * 60  # stop submitting episodes after this much wall-clock
# Skip the concurrent batch if the container is already this old after the sequential one
# (the container watchdog kills it at reference_app.HARD_LIFETIME_S).
MAX_AGE_FOR_CONCURRENT_S = 28 * 60


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
    p.add_argument("--concurrency", type=int, help="W3-A: run one batch with N episode threads")
    p.add_argument("--sequential", action="store_true",
                   help="W3-A: first run the same episodes with concurrency 1 and compare")
    p.add_argument("--measure", action="store_true", help="W3-A: per-call timers, RSS, /dev/shm")
    p.add_argument("--cpu", type=float, help="W3-A: physical cores for the worker")
    p.add_argument("--torch-threads", type=int, help="W3-A: torch intra-op threads per batch")
    p.add_argument("--repeat", type=int, default=1, help="W3-A: run each episode K times")
    p.add_argument("--compare-to", type=Path, help="W3-A: earlier summary to compare digests")
    args = p.parse_args(argv)
    if args.concurrency:
        return concurrency_experiment(args)

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


# --- W3-A ------------------------------------------------------------------------------------


def batch_summary(out: dict) -> dict:
    """Numbers only: no captions, no prompts, no prompt hashes."""
    rows = []
    for name, result in out["results"].items():
        fr = result["flora_reference"]
        row = summary_row(name, result)
        row.update(total_s=fr["timings_s"]["total"], end_s=fr.get("batch", {}).get("end_s"),
                   digest=out["digests"][name], stored_at=fr.get("stored_at"),
                   tokens=fr.get("tokens"), generation=fr.get("generation"))
        if "observe" in fr:
            row["observe"] = fr["observe"]
        rows.append(row)
    return {"batch": out["batch"], "episodes": rows, "errors": out["errors"]}


def base_name(episode: str) -> str:
    return episode.rsplit("-r", 1)[0]


def concurrency_experiment(args) -> int:
    t0 = time.monotonic()
    summary: dict = {"args": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()}}
    runs: dict[str, dict] = {}
    with modal.enable_output(), app.run():
        manifest = stage_samples.remote(force=False)
        base = [e for e in manifest["episodes"] if not args.only or e["episode"] in args.only]
        episodes = [
            {"episode": e["episode"] if args.repeat == 1 else f"{e['episode']}-r{k}",
             "videos": e["videos"], "profile": e["profile"]}
            for k in range(args.repeat) for e in base
        ]
        # longest first, so the concurrent batch does not end on one long straggler
        dur = {e["episode"]: e["probe"]["main"]["duration_s"] for e in manifest["episodes"]}
        episodes.sort(key=lambda e: -dur[base_name(e["episode"])])
        worker_cls = ReferenceWorker.with_options(cpu=args.cpu) if args.cpu else ReferenceWorker
        worker = worker_cls()
        summary["load"] = worker.report.remote()
        print("load report:", json.dumps(summary["load"], indent=2))
        if args.sequential:
            runs["c1"] = worker.annotate_batch.remote(
                episodes, 1, measure=args.measure, torch_threads=args.torch_threads, tag="c1")
            summary["c1"] = batch_summary(runs["c1"])
            print(table(summary["c1"]["episodes"]))
        age = worker.report.remote()["container_age_s"]
        tag = f"c{args.concurrency}"
        if age > min(MAX_AGE_FOR_CONCURRENT_S, HARD_LIFETIME_S - 10 * 60):
            summary["skipped"] = f"container age {age}s: concurrent batch not submitted"
        elif tag not in runs:
            runs[tag] = worker.annotate_batch.remote(
                episodes, args.concurrency, measure=args.measure,
                torch_threads=args.torch_threads, tag=tag)
            summary[tag] = batch_summary(runs[tag])
            print(table(summary[tag]["episodes"]))
            if "c1" in runs:
                summary["identity"] = compare_runs(runs["c1"]["results"], runs[tag]["results"])
        final = worker.report.remote()
    summary["container_age_s"] = final["container_age_s"]
    summary["final_gpu"] = final["gpu_now"]
    summary["local_wall_s"] = round(time.monotonic() - t0, 1)
    if args.compare_to:
        earlier = json.loads(args.compare_to.read_text())
        old = {base_name(r["episode"]): r["digest"]
               for k, v in earlier.items() if k.startswith("c") and isinstance(v, dict)
               for r in v.get("episodes", [])}
        summary["cross_run"] = {
            f"{k}/{r['episode']}": old.get(base_name(r["episode"])) == r["digest"]
            for k, v in summary.items() if k in runs for r in v["episodes"]
        }
    for k in runs:
        print(k, "batch:", json.dumps({x: y for x, y in summary[k]["batch"].items()
                                       if x != "calls"}, indent=1))
    for key in ("identity", "cross_run", "skipped"):
        if key in summary:
            print(key, json.dumps(summary[key], indent=1))
    print(f"container age {summary['container_age_s']}s; local wall {summary['local_wall_s']}s")
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    errors = any(summary[k]["errors"] for k in runs)
    identical = summary.get("identity", {}).get("identical", True)
    return 0 if identical and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
