"""Aggregate recorder events into per-episode tables."""

from __future__ import annotations

import math
from collections.abc import Iterable


def distribution(values: Iterable[float]) -> dict:
    """count/sum/mean/p50/p90/max (nearest-rank percentiles)."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"count": 0, "sum": 0, "mean": None, "p50": None, "p90": None, "max": None}

    def pct(p):
        return vals[min(len(vals) - 1, max(0, math.ceil(p * len(vals)) - 1))]

    total = sum(vals)
    return {"count": len(vals), "sum": round(total, 3), "mean": round(total / len(vals), 3),
            "p50": pct(0.5), "p90": pct(0.9), "max": vals[-1]}


def call_table(events: Iterable[dict]) -> dict[str, dict]:
    """``{label: {calls, wall_s, self_s, cpu_s, max_wall_s, errors}}`` over ``call`` events,
    plus ``lock:<name>`` rows (``calls``, ``wait_s``, ``hold_s``, ``max_wait_s``)."""
    table: dict[str, dict] = {}
    for e in events:
        if e.get("kind") == "call":
            row = table.setdefault(e["label"], {"calls": 0, "wall_s": 0.0, "self_s": 0.0,
                                                "cpu_s": 0.0, "max_wall_s": 0.0, "errors": 0})
            row["calls"] += 1
            row["wall_s"] += e["wall_s"]
            row["self_s"] += e["self_s"]
            row["cpu_s"] += e["cpu_s"]
            row["max_wall_s"] = max(row["max_wall_s"], e["wall_s"])
            row["errors"] += bool(e.get("error"))
        elif e.get("kind") == "lock":
            row = table.setdefault(f"lock:{e['lock']}", {"calls": 0, "wait_s": 0.0,
                                                         "hold_s": 0.0, "max_wait_s": 0.0,
                                                         "contended": 0})
            row["calls"] += 1
            row["wait_s"] += e["wait_s"]
            row["hold_s"] += e["hold_s"]
            row["max_wait_s"] = max(row["max_wait_s"], e["wait_s"])
            row["contended"] += bool(e.get("contended"))
    for row in table.values():
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 4)
    return table


def merge_tables(tables: Iterable[dict[str, dict]]) -> dict[str, dict]:
    """Sum several ``call_table`` results (max columns take the max)."""
    out: dict[str, dict] = {}
    for table in tables:
        for label, row in table.items():
            acc = out.setdefault(label, {k: 0 for k in row})
            for k, v in row.items():
                acc[k] = max(acc.get(k, 0), v) if k.startswith("max_") else acc.get(k, 0) + v
    for row in out.values():
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 4)
    return out


def token_stats(runtime_events: Iterable[dict], end_event: str = "generation_end") -> dict:
    """Prompt/output token distributions from NVIDIA-style ``generation_end`` events."""
    gens = [e for e in runtime_events if e.get("event") == end_event]
    return {
        "prompt_tokens": distribution(int(e.get("input_tokens", 0)) for e in gens),
        "output_tokens": distribution(int(e.get("generated_tokens", 0)) for e in gens),
        "output_limit": sorted({int(e["output_limit"]) for e in gens if "output_limit" in e}),
        "generation_s": distribution(float(e.get("elapsed_s", 0.0)) for e in gens),
    }
