"""The fast ``exact`` worker on one H200: S1 identity proof, WS-D smoke, W3-A concurrency.

S1 (docs/exact-worker.md): CPU pre-check, then the identity proof in ONE H200 container
(fast composite vs NVIDIA unmodified on golden episodes, model-input digests compared):

    uv run python scripts/reference_smoke.py --offline-check
    uv run python scripts/reference_smoke.py --identity [--golden ID ...] [--memory-gib 160] \
        [--deadline-min 16] [--disable FEATURES] --summary PATH

WS-D smoke (docs/reference-smoke.md):

    uv run python scripts/reference_smoke.py [--only EPISODE ...] [--summary PATH] [--force-stage]

W3-A concurrency experiment (docs/exact-concurrency.md), one container, one model load:

    uv run python scripts/reference_smoke.py --concurrency 4 --sequential --measure \\
        [--cpu 8] [--repeat K] [--compare-to SUMMARY.json] --summary PATH

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
    STARTUP_TIMEOUT_S,
    TIMEOUT_S,
    ReferenceWorker,
    app,
    identity_from_volume,
    offline_check,
    stage_samples,
)
from flora_deploy.stages.reference import (
    IDENTITY_EPISODES,
    compare_runs,
    new_run_id,
    summary_row,
)

COLUMNS = (
    "episode", "views", "profile", "duration_s", "windows", "segments", "captions",
    "probe_s", "segment_s", "caption_s", "seg_out_tok", "cap_out_tok", "peak_gib",
)
DEADLINE_S = 25 * 60  # stop submitting episodes after this much wall-clock
# Skip the concurrent batch if the container is already this old after the sequential one.
MAX_AGE_FOR_CONCURRENT_S = 15 * 60
# Modal list prices (docs/exact-concurrency.md): H200, per physical core, per GiB of memory.
H200_PER_S, CORE_PER_S, GIB_PER_S = 0.001261, 0.0000131, 0.00000222


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
    p.add_argument("--offline-check", action="store_true", help="S1: CPU receipt/offline check")
    p.add_argument("--identity", action="store_true", help="S1: the identity proof (one H200)")
    p.add_argument("--golden", action="append", default=[], help="S1: golden manifest id(s)")
    p.add_argument("--memory-gib", type=int, help="worker memory request (GiB)")
    p.add_argument("--deadline-min", type=float, default=16.0,
                   help="S1: no NVIDIA-unmodified episode starts after this (call minutes)")
    p.add_argument("--disable", default="", help="ExactProfile toggles to switch off")
    p.add_argument("--determinism", default="nvidia", choices=("nvidia", "deterministic"),
                   help="D-1 mode (docs/determinism.md)")
    p.add_argument("--no-instruction", action="store_true",
                   help="S1: blank instruction instead of the dataset task text")
    p.add_argument("--repeat", type=int, default=1, help="W3-A: run each episode K times")
    p.add_argument("--compare-to", type=Path, help="W3-A: earlier summary to compare digests")
    args = p.parse_args(argv)
    if args.offline_check:
        with modal.enable_output(), app.run():
            out = offline_check.remote()
        print(json.dumps(out, indent=2))
        _write(args.summary, out)
        return 0 if all("error" not in r for r in out["receipts"].values()) else 1
    if args.identity:
        return identity(args)
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
        worker = _worker(args)
        load = worker.report.remote()
        print("load report:", json.dumps(load, indent=2))
        run_id = new_run_id("wsd")
        container_error = False
        for ep in episodes:
            if time.monotonic() - t0 > DEADLINE_S:
                errors.append({"episode": ep["episode"], "error": "deadline: not submitted"})
                continue
            try:
                result = worker.annotate.remote(ep["videos"], ep["profile"],
                                                episode=ep["episode"], run_id=run_id)
            except Exception as e:  # noqa: BLE001 - report and continue with the next episode
                errors.append({"episode": ep["episode"], "error": f"{type(e).__name__}: {e}"})
                # NVIDIA errors arrive as NvidiaError; anything else is container-level
                container_error = type(e).__name__ not in ("NvidiaError", "NvidiaInputError")
                continue
            row = summary_row(ep["episode"], result)
            row["stored_at"] = result["flora_reference"].get("stored_at")
            rows.append(row)
            print(table([row]))
        # review2-wsd L7: never start a fresh container (a full model load) just to report
        final = {} if container_error else worker.report.remote()

    print()
    print(table(rows))
    for e in errors:
        print("FAILED", e["episode"], e["error"])
    summary = {
        "load": load,
        "final_gpu": final.get("gpu_now"),
        "container_age_s": final.get("container_age_s"),
        "episodes": rows,
        "errors": errors,
        "local_wall_s": round(time.monotonic() - t0, 1),
    }
    print(f"container age {final.get('container_age_s')}s; local wall {summary['local_wall_s']}s")
    _write(args.summary, summary)
    return 1 if errors else 0


def _write(path: Path | None, obj: dict) -> None:
    if path:
        path.write_text(json.dumps(obj, indent=2) + "\n")


def _worker(args):
    options = {}
    if getattr(args, "cpu", None):
        options["cpu"] = args.cpu
    if getattr(args, "memory_gib", None):
        options["memory"] = args.memory_gib * 1024
    cls = ReferenceWorker.with_options(**options) if options else ReferenceWorker
    return cls(disable=getattr(args, "disable", ""), concurrency=args.concurrency or 4,
               determinism=getattr(args, "determinism", "nvidia"))


def container_rate(cpu: float, memory_gib: float) -> float:
    """$/s of one H200 container at Modal list prices."""
    return H200_PER_S + cpu * CORE_PER_S + memory_gib * GIB_PER_S


# --- S1 ----------------------------------------------------------------------------------------


def identity(args) -> int:
    """Fast composite vs NVIDIA unmodified in one container (``ReferenceWorker.identity_proof``)."""
    from flora_deploy.modal_app.reference_app import CPU_CORES, MEMORY_MIB

    t0 = time.monotonic()
    ids = args.golden or list(IDENTITY_EPISODES)
    run_id = new_run_id("s1")
    memory_gib = args.memory_gib or MEMORY_MIB / 1024
    cpu = args.cpu or CPU_CORES
    rate = container_rate(cpu, memory_gib)
    bound_s = STARTUP_TIMEOUT_S + TIMEOUT_S
    print(f"run {run_id}: {len(ids)} episodes; worst case {bound_s / 60:.0f} min "
          f"= ${rate * bound_s:.2f} at ${rate * 3600:.2f}/h")
    summary: dict = {"run_id": run_id, "ids": ids, "rate_per_h": round(rate * 3600, 3)}
    with modal.enable_output(), app.run():
        worker = _worker(args)
        try:
            summary["load"] = worker.report.remote()
            print("load:", json.dumps({k: summary["load"][k] for k in
                                       ("enter_s", "marks_s", "container_import_to_enter_s")}))
            summary["proof"] = worker.identity_proof.remote(
                ids, run_id, deadline_s=args.deadline_min * 60,
                instruction=not args.no_instruction)
        except Exception as e:  # noqa: BLE001 - keep what we have
            summary["error"] = f"{type(e).__name__}: {e}"
            summary["identity_from_volume"] = identity_from_volume.remote(run_id)
            print("identity (from the Volume):", json.dumps(
                {k: v for k, v in summary["identity_from_volume"].items() if k != "episodes"}))
    summary["local_wall_s"] = round(time.monotonic() - t0, 1)
    proof = summary.get("proof")
    if proof:
        summary["throughput"] = throughput(summary["load"], proof, rate)
        ident = proof["identity"]
        print(json.dumps({k: v for k, v in ident.items() if k != "episodes"}, indent=1))
        for name, row in ident["episodes"].items():
            print(name, json.dumps({k: row.get(k) for k in (
                "calls_a", "calls_b", "strict", "multiset", "gate_ok", "gated_calls",
                "first_output_difference_call", "same_input_pairs",
                "same_input_different_output", "output_identical")}))
        print("throughput:", json.dumps(summary["throughput"], indent=1))
    else:
        print("FAILED:", summary.get("error"))
    _write(args.summary, summary)
    return 0 if proof and proof["identity"]["gate_ok"] else 1


def throughput(load: dict, proof: dict, rate_per_s: float) -> dict:
    """Video-hours per GPU-hour and $/video-hour of both arms (numbers only)."""
    fast = proof["fast"]["batch"]
    nv_eps = proof["nvidia"]["episodes"]
    nv_video = sum(r["duration_s"] for r in nv_eps.values())
    nv_wall = sum(w["wall_s"] for w in proof["nvidia"]["walls"].values())
    out = {"fast": {"video_s": fast["video_s"], "wall_s": fast["wall_s"],
                    "concurrency": fast["concurrency"],
                    "gpu_busy_fraction": fast.get("gpu_busy_fraction")},
           "nvidia_unmodified": {"video_s": round(nv_video, 2), "wall_s": round(nv_wall, 1),
                                 "episodes": len(nv_eps)},
           "startup_s": load.get("enter_s")}
    for arm in out.values():
        if isinstance(arm, dict) and arm.get("wall_s"):
            vh_per_h = arm["video_s"] / arm["wall_s"]
            arm["video_h_per_gpu_h"] = round(vh_per_h, 4)
            arm["usd_per_video_h"] = round(rate_per_s * 3600 / vh_per_h, 2) if vh_per_h else None
    return out


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
        worker = _worker(args)
        run_id = new_run_id("w3a")
        summary["load"] = worker.report.remote()
        print("load report:", json.dumps(summary["load"], indent=2))
        if args.sequential:
            runs["c1"] = worker.annotate_batch.remote(
                episodes, run_id, concurrency=1, measure=args.measure, tag="c1")
            summary["c1"] = batch_summary(runs["c1"])
            print(table(summary["c1"]["episodes"]))
        age = worker.report.remote()["container_age_s"]
        tag = f"c{args.concurrency}"
        if age > MAX_AGE_FOR_CONCURRENT_S:
            summary["skipped"] = f"container age {age}s: concurrent batch not submitted"
        elif tag not in runs:
            runs[tag] = worker.annotate_batch.remote(
                episodes, run_id, concurrency=args.concurrency, measure=args.measure, tag=tag)
            summary[tag] = batch_summary(runs[tag])
            print(table(summary[tag]["episodes"]))
            if "c1" in runs:
                summary["identity"] = compare_runs(runs["c1"]["results"], runs[tag]["results"])
        final = worker.report.remote()
    if args.repeat > 1:  # the same episode run K times in one batch: r0 vs r1..rK-1
        summary["repeat_identity"] = {}
        for k, out in runs.items():
            res = out["results"]
            first = {base_name(n): r for n, r in res.items() if n.endswith("-r0")}
            for j in range(1, args.repeat):
                other = {base_name(n): r for n, r in res.items() if n.endswith(f"-r{j}")}
                summary["repeat_identity"][f"{k}:r0-vs-r{j}"] = compare_runs(first, other)
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
    for key in ("identity", "repeat_identity", "cross_run", "skipped"):
        if key in summary:
            print(key, json.dumps(summary[key], indent=1))
    print(f"container age {summary['container_age_s']}s; local wall {summary['local_wall_s']}s")
    _write(args.summary, summary)
    errors = any(summary[k]["errors"] for k in runs)
    identical = summary.get("identity", {}).get("identical", True)
    return 0 if identical and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
