"""Real-Modal CPU canary of the job system (``flora_deploy.modal_app.canary_app``).

Usage: uv run python scripts/canary.py [--scenarios happy,preempt,poison,restart,budget]
                                       [--keep] [--logs]

Runs the ``flora-canary`` app ephemerally (``app.run()``; nothing is deployed), one orchestrator
per scenario in parallel, then reads the final state back from the ``flora-canary`` Volume,
prints a summary, deletes this run's job directories from that Volume (only that Volume; skip
with ``--keep``) and exits non-zero if any scenario failed. CPU-only, cents of cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any


def run(names: list[str], *, keep: bool = False, logs: bool = False) -> dict[str, Any]:
    import modal

    from flora_deploy.modal_app import canary_app as C

    all_sc = C.scenarios()
    unknown = sorted(set(names) - set(all_sc))
    if unknown:
        raise SystemExit(f"unknown scenarios {unknown}; known: {sorted(all_sc)}")
    run_id = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:4]
    jobs = {name: f"canary-{run_id}-{name}" for name in names}
    results: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    t0 = time.time()
    ctx = modal.enable_output() if logs else _null()
    try:
        with ctx, C.app.run():
            calls = {n: C.orchestrate.spawn(jobs[n], all_sc[n].to_dict()) for n in names}
            for n, fc in calls.items():
                try:
                    results[n] = fc.get(timeout=C.TIMEOUT_S + 60)
                except Exception as exc:  # report it; keep going with the others
                    results[n] = {"error": repr(exc)}
                print(f"[{time.time() - t0:6.1f}s] {n}: "
                      f"{results[n].get('outcome', results[n].get('error'))}", flush=True)
            summaries = C.verify.remote({jobs[n]: all_sc[n].to_dict() for n in names})
    finally:
        if not keep:
            _cleanup(C, list(jobs.values()))
    report: dict[str, Any] = {"run_id": run_id, "wall_s": round(time.time() - t0, 1),
                              "scenarios": {}}
    for n in names:
        s = summaries.get(jobs[n], {"error": "not verified"})
        problems = C.evaluate(all_sc[n], results.get(n, {"error": "no result"}), s)
        report["scenarios"][n] = {"job_id": jobs[n], "pass": not problems, "problems": problems,
                                  "run": results.get(n), "files": s}
    return report


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _cleanup(C: Any, job_ids: list[str]) -> None:  # noqa: N803 - module alias
    """Remove this run's job trees from the canary Volume (never any other Volume) and its
    lease keys from the canary Dict."""
    import modal

    assert C.VOLUME_NAME == "flora-canary"
    vol = modal.Volume.from_name(C.VOLUME_NAME, version=2)
    for job in job_ids:
        try:
            vol.remove_file(f"flora/{job}", recursive=True)
        except Exception as exc:  # e.g. never created
            print(f"cleanup {job}: {exc!r}", file=sys.stderr)
    leases = modal.Dict.from_name(C.LEASE_DICT)
    for key in list(leases.keys()):
        if any(str(key).startswith(f"orchestrator/{job}/") for job in job_ids):
            leases.pop(key)


def print_report(report: dict[str, Any]) -> None:
    print(f"\ncanary run {report['run_id']} ({report['wall_s']} s)")
    for n, r in report["scenarios"].items():
        f = r["files"]
        run_ = r["run"] or {}
        out = (run_.get("outcome") or {}).get("result", run_.get("error"))
        line = (f"  {'PASS' if r['pass'] else 'FAIL'} {n:8s} result={out} "
                f"records={f.get('records')}/{f.get('expected')}"
                f" missing={len(f.get('missing', []))} excused={len(f.get('excused', []))}"
                f" dup={len(f.get('raw_duplicates', []))}"
                f" controls={f.get('controls')} wall={run_.get('wall_s')}s")
        print(line)
        for p in r["problems"]:
            print(f"       - {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="happy,preempt,poison,restart,budget")
    ap.add_argument("--keep", action="store_true", help="leave the job files on the Volume")
    ap.add_argument("--logs", action="store_true", help="stream container logs")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)
    report = run([s for s in args.scenarios.split(",") if s], keep=args.keep, logs=args.logs)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
    print_report(report)
    return 0 if all(r["pass"] for r in report["scenarios"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
