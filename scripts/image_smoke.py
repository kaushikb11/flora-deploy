"""CPU smoke of every image the GPU session uses (``modal_app/smoke_app.py``), for cents.

Usage:
  uv run python scripts/image_smoke.py --dry-run        # plans and env, contacts nothing
  uv run python scripts/image_smoke.py                  # all five CPU containers
  uv run python scripts/image_smoke.py --only caption,segcap --out reports/image-smoke.json
  uv run python scripts/image_smoke.py --receipt tempseg_merged=<sha256>

Runs one ephemeral app (``app.run()``: stopped when this script exits, also on Ctrl-C), spawns
the five functions concurrently (each ``max_containers=1``, CPU only), prints a pass/fail table,
each container's seconds and a list-price cost estimate (billing lags about an hour:
``uv run python scripts/spend.py --app flora-image-smoke``). Exit status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_receipts(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        name, _, sha = item.partition("=")
        if not sha:
            raise SystemExit(f"--receipt {item!r}: use NAME=<sha256>")
        out[name] = sha
    return out


def _table(results: dict[str, dict]) -> tuple[str, bool, float]:
    lines, all_ok, total = [], True, 0.0
    lines.append(f"{'target':13s} {'image':10s} {'check':38s} {'ok':4s} {'s':>6s}")
    for name, r in results.items():
        if "error" in r:
            lines.append(f"{name:13s} {'-':10s} {'(call failed)':38s} {'FAIL':4s}  {r['error']}")
            all_ok = False
            continue
        for c in r["checks"]:
            mark = "ok" if c["ok"] else "FAIL"
            lines.append(f"{name:13s} {r['image']:10s} {c['check'][:38]:38s} {mark:4s} "
                         f"{c['s']:6.1f}")
            if not c["ok"]:
                lines.append(f"{'':13s} -> {c['detail']['error'][:600]}")
        all_ok &= r["ok"]
        total += r["usd_estimate"]
    lines.append("")
    lines.append(f"{'target':13s} {'function_s':>10s} {'since_import_s':>14s} {'wall_s':>8s} "
                 f"{'usd_est':>8s}")
    for name, r in results.items():
        if "error" not in r:
            lines.append(f"{name:13s} {r['function_s']:10.1f} {r['since_import_s']:14.1f} "
                         f"{r.get('wall_s', 0):8.1f} {r['usd_estimate']:8.4f}")
    return "\n".join(lines), all_ok, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the plan; contact nothing")
    ap.add_argument("--only", default="", help="comma-separated subset of the targets")
    ap.add_argument("--receipt", action="append", default=[], help="NAME=<sha256> override")
    ap.add_argument("--out", default="", help="write the full JSON results here")
    args = ap.parse_args()
    if args.receipt:
        os.environ["FLORA_SMOKE_RECEIPTS"] = json.dumps(_parse_receipts(args.receipt))

    from flora_deploy.modal_app import smoke_app as S

    if S.PLAN_ERROR:
        print(f"factory.plan_app failed: {S.PLAN_ERROR}", file=sys.stderr)
        return 2
    names = [n for n in (args.only.split(",") if args.only else S.TARGETS) if n]
    bad = sorted(set(names) - set(S.TARGETS))
    if bad:
        ap.error(f"unknown target(s) {bad}; choose from {', '.join(S.TARGETS)}")

    print(f"app {S.APP_NAME}: {len(names)} CPU function(s), cpu={S.CPU:g} "
          f"memory={S.MEMORY_MIB // 1024} GiB, timeout={S.TIMEOUT_S}s, no GPU")
    for n in names:
        e, t = S.EXPECTED[n], S.TARGETS[n]
        vols = ", ".join(f"{p}={m['name']}(ro)" for p, m in e["volumes"].items()
                         if m["read_only"] and ((m["name"] == "flora-weights" and t.weights)
                                                or (m["name"] == "flora-cache" and t.cache)))
        print(f"  {n:13s} {t.image:10s} fp={e['fingerprint'][:12]} as {t.obj} "
              f"(plan gpu={e['gpu_in_plan']}), env {len(e['env'])} keys, volumes: {vols or '-'}")
    worst = len(names) * S.TIMEOUT_S * S.USD_PER_S
    print(f"cost ceiling (every container to its timeout): ${worst:.3f}")
    if args.dry_run:
        for n in names:
            print(f"\n[{n}] env:")
            for k, v in sorted(S.EXPECTED[n]["env"].items()):
                print(f"  {k}={v[:110]}")
        return 0

    results: dict[str, dict] = {}
    t0 = time.time()
    with S.app.run():
        calls = {n: (S.FUNCTIONS[n].spawn(S.EXPECTED[n]), time.time()) for n in names}
        for n, (call, started) in calls.items():
            try:
                results[n] = call.get(timeout=S.TIMEOUT_S + 300)
                results[n]["wall_s"] = round(time.time() - started, 1)
            except Exception as e:  # noqa: BLE001 - reported in the table
                results[n] = {"error": f"{type(e).__name__}: {e}"}
    print(f"\napp {S.APP_NAME} stopped ({time.time() - t0:.0f} s)\n")
    table, ok, total = _table(results)
    print(table)
    print(f"\nestimated cost: ${total:.4f} (list price, CPU + memory since module import; "
          "billing: scripts/spend.py --app flora-image-smoke)")
    print("RESULT:", "PASS" if ok else "FAIL")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
