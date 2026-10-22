"""Actual Modal spend per app from the workspace billing API, against a build budget.

Usage: uv run python scripts/spend.py [--since 2026-10-20] [--budget 50] [--app flora-gatec]
Billing data covers complete hours only, so the last hour or so of usage is not yet included.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime


def report(since: str) -> list[dict]:
    out = subprocess.run(
        ["modal", "billing", "report", "--start", since, "--resolution", "h", "--json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def interval_start(value: object) -> float | None:
    """A row's ``interval_start`` (ISO 8601, UTC when naive) as a POSIX time; None if absent."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def rows(since: str) -> list[dict]:
    """The hourly report as ``{object_id, description, interval_start, cost}`` rows
    (``interval_start``: POSIX seconds of the billed hour's start, or None)."""
    out = []
    for row in report(since):
        out.append({"object_id": row.get("object_id") or row.get("Object ID") or "",
                    "description": row.get("description") or row.get("Description") or "?",
                    "interval_start": interval_start(row.get("interval_start")
                                                     or row.get("Interval Start")),
                    "cost": float(row.get("cost") or row.get("Cost") or 0)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-20")
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--app", help="only this app (description)")
    args = ap.parse_args()

    totals: dict[str, float] = defaultdict(float)
    for row in report(args.since):
        name = row.get("description") or row.get("Description") or "?"
        totals[name] += float(row.get("cost") or row.get("Cost") or 0)
    if args.app:
        print(f"{args.app}: ${totals.get(args.app, 0.0):.2f}")
        return
    for name, cost in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{name:24s} ${cost:7.2f}")
    total = sum(totals.values())
    print(f"{'TOTAL':24s} ${total:7.2f}  of ${args.budget:.2f}  (left ${args.budget - total:.2f})")


if __name__ == "__main__":
    main()
