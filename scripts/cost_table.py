"""Generate the tables of ``docs/costs.md`` from ``planner/rates.json`` and the planner.

    uv run python scripts/cost_table.py            # print the regenerated doc
    uv run python scripts/cost_table.py --write    # rewrite docs/costs.md

Every number is priced by the repository's own code: ``planner.options.build_option`` (the model
``flora plan`` uses: GPU work + each container's startup + its idle tail at job end + the
orchestrator) at the pinned Modal list prices of ``jobs.budget``, from the shipped rate table.
The prose of ``docs/costs.md`` is edited by hand; only the blocks between
``<!-- cost-table:<name> -->`` and ``<!-- /cost-table:<name> -->`` are generated.
``tests/unit/test_cost_docs.py`` fails when the doc differs from :func:`render`, and
``scripts/wave_h_finalize.py`` regenerates it whenever it rewrites ``rates.json``.

Workloads come from ``WorkloadSummary.from_hours`` (1-view episodes of one minute, except the
``cameras`` block's 3-view rows, priced with ``rates.json`` ``camera_scaling``), on the
``balanced`` preset's default pools.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "costs.md"

PRESET = "balanced"
SIZES = (0.05, 0.5, 1.0, 10.0, 100.0)  # video-hours
SEGCAP = ("segment", "caption")
REWARDS = ("reward.terminal", "reward.dense")  # the rewards the templates turn on
BREAKDOWN_SIZES = (0.05, 1.0, 100.0)
LEVER_SIZES = (0.05, 1.0, 100.0)
GPU_CAPS = (4, 10)  # `modal.max_gpus` values for the "more GPUs" table (templates: 4)
GPU_SIZES = (0.5, 10.0, 100.0)
SMALL_JOB_H, SMALL_JOBS = 0.05, 10  # the batching example: ten small jobs vs one
CAMERA_SIZES = (1.0, 100.0)  # video-hours of the 1-view vs 3-view table
CAMERA_LAYOUTS = (1, 3)

# Batch fill (episodes prepared outside the micro-batcher, larger batches) is deployed and was
# on when the balanced rows were measured, but that job was too small to fill a batch. These
# busy rates are the cost model's estimate of a filled batch at steady state (GPU-bound, >= 16
# episodes per shard); a job too small to fill a batch gains nothing. Not a measurement: shown
# as an estimate, and only while it is below the measured row (once a job large enough to fill
# batches is measured, the row drops out by itself).
BATCH_FILL_ESTIMATE = {"segment": 0.20, "caption": 0.55}
BATCH_FILL_UNCERTAINTY = 0.5
BATCH_FILL_MIN_EPISODES = 16

_BLOCK = re.compile(r"(<!-- cost-table:(?P<name>[\w-]+) -->\n)(?P<body>.*?)"
                    r"(<!-- /cost-table:(?P=name) -->)", re.DOTALL)


# ----------------------------------------------------------------------------------------------
# Pricing (the planner's)


@dataclass(frozen=True)
class Priced:
    video_h: float
    episodes: int
    cost: float
    low: float
    high: float
    wall_s: float
    containers: Mapping[str, int]
    busy: float
    startup: float
    idle: float
    orchestrator: float
    label: str

    def per_h(self, x: float | None = None) -> float:
        return (self.cost if x is None else x) / self.video_h


def price(video_h: float, stages: tuple[str, ...], *, table=None, max_gpus: int | None = None,
          assume_warm: bool = False, cameras: int = 1) -> Priced:
    """One ``balanced`` job of ``video_h`` hours, as ``flora plan`` prices it. ``max_gpus``
    ``None``: one container per pool (the templates' ``max_containers: 1``)."""
    from flora_deploy.config.presets import get_preset, shard_kinds_for
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.planner.options import (
        POOL_END_TAIL_S,
        PlanRequest,
        WorkloadSummary,
        build_option,
        rates_label,
    )
    from flora_deploy.planner.rates import RateTable

    table = table or RateTable.load()
    rates = pinned_rates()
    spec = get_preset(PRESET)
    kinds = shard_kinds_for(spec, stages)  # type: ignore[arg-type]
    base = sum(spec.pools[k].gpu_count for k in kinds)
    req = PlanRequest(workload=WorkloadSummary.from_hours(video_h, cameras=cameras),
                      stages=stages,  # type: ignore[arg-type]
                      presets=(PRESET,), max_gpus=max_gpus or base, assume_warm=assume_warm)
    o = build_option(req, table, rates, PRESET, 1)
    busy = startup = idle = 0.0
    for k in o.kinds:
        per_s = rates.pool_per_second(spec.pools[k.kind])
        busy += per_s * k.work_s
        startup += per_s * k.containers * k.startup_s
        idle += per_s * k.containers * POOL_END_TAIL_S
    return Priced(video_h=video_h, episodes=req.workload.episodes, cost=o.cost_usd,
                  low=o.cost_low_usd, high=o.cost_high_usd, wall_s=o.wall_s,
                  containers={k.kind: k.containers for k in o.kinds}, busy=busy,
                  startup=startup, idle=idle, orchestrator=max(0.0, o.cost_usd - busy - startup
                                                               - idle),
                  label=rates_label(o))


def batch_fill_table(table):
    """``table`` with the balanced busy rates replaced by :data:`BATCH_FILL_ESTIMATE` (as
    ``projected``), or ``None`` when no estimate is below the measured rate any more."""
    rows, changed = [], False
    for e in table.rates:
        est = BATCH_FILL_ESTIMATE.get(e.kind) if e.preset == PRESET else None
        if est is not None and e.status != "stale" and est < e.container_s_per_video_s:
            e = replace(e, container_s_per_video_s=est, uncertainty=BATCH_FILL_UNCERTAINTY,
                        status="projected", n=None)
            changed = True
        rows.append(e)
    return replace(table, rates=tuple(rows)) if changed else None


def busy_per_video_h(kinds: tuple[str, ...], table=None) -> float:
    """$ per video-hour of GPU work alone (no startup, no idle): Σ rate × pool $/h."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.planner.rates import RateTable

    table = table or RateTable.load()
    rates = pinned_rates()
    spec = get_preset(PRESET)
    return sum(3600 * table.rate(PRESET, k, 1, 1).container_s_per_video_s
               * rates.pool_per_second(spec.pools[k]) for k in kinds)


# ----------------------------------------------------------------------------------------------
# Formatting


def usd(x: float) -> str:
    if x >= 1000:
        return f"${x:,.0f}"
    if x >= 100:
        return f"${x:.0f}"
    return f"${x:.2f}"


def per_h(p: Priced) -> str:
    return f"**{usd(p.per_h())}** ({usd(p.per_h(p.low))}-{usd(p.per_h(p.high))})"


def hours(s: float) -> str:
    if s < 3600:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def size(v: float) -> str:
    return f"{v:g}"


def band(u: float) -> str:
    return f"±{u:.0%}"


def _status(e) -> str:
    if e.status == "measured":
        n = f"n={e.n}, " if e.n is not None else ""
        return f"measured ({n}{band(e.uncertainty)})"
    return f"estimate ({e.status}, {band(e.uncertainty)})"


def basis(table, kinds: tuple[str, ...]) -> str:
    """What the numbers rest on, kind by kind, from the rate table."""
    parts = []
    for k in kinds:
        e = table.rate(PRESET, k, 1, 1)
        st = table.startup_for(PRESET, k)
        parts.append(f"`{k}`: GPU work {e.container_s_per_video_s:g} container-s per video-s, "
                     f"{_status(e)}; startup {st.startup_s:.0f} s, {_status(st)}")
    return "\n".join(parts)


def single_job(table, kinds: tuple[str, ...] = SEGCAP) -> bool:
    """Whether any of ``kinds``' busy rates rests on one measured job (n=1): a figure for a
    large job is then an extrapolation, not a measurement at that scale."""
    return any(table.rate(PRESET, k, 1, 1).status == "measured"
               and table.rate(PRESET, k, 1, 1).n == 1 for k in kinds)


SCALE_CAVEAT = ("projected from rates measured on one small job; not yet confirmed at scale "
                "([scale test](scale-test.md))")


def table_md(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------
# Blocks


def block_short(table) -> str:
    rows = []
    for v in SIZES:
        a = price(v, SEGCAP, table=table)
        b = price(v, SEGCAP + REWARDS, table=table)
        rows.append([size(v), f"{a.episodes:,}", per_h(a), usd(a.cost), hours(a.wall_s),
                     per_h(b), usd(b.cost)])
    t = table_md(["Video-hours", "Episodes", "Segment + caption, per video-hour", "Job",
                  "Wall-clock", "+ rewards, per video-hour", "Job"], rows)
    lines = [f"- {b}" for b in basis(table, SEGCAP + REWARDS).splitlines()]
    big = [size(v) for v in SIZES if v >= 10]
    scale = (f" The {' and '.join(big)} video-hour rows are {SCALE_CAVEAT}."
             if single_job(table) and big else "")
    return (t + "\n\nOne container per stage, cold starts, Modal list prices. Bold: the "
            "estimate; in brackets: its range from the error bands below. Rewards are "
            f"terminal + dense (`reward.rubric` is not measured yet).{scale} What each rate "
            "rests on:\n\n" + "\n".join(lines) + "\n")


def block_busy(table) -> str:
    return (f"**{usd(busy_per_video_h(SEGCAP, table))} per video-hour** "
            f"(segment + caption; {usd(busy_per_video_h(SEGCAP + REWARDS, table))} with "
            "rewards)" + (f", {SCALE_CAVEAT}" if single_job(table) else "") + "\n")


def block_breakdown(table) -> str:
    rows = []
    for v in BREAKDOWN_SIZES:
        p = price(v, SEGCAP, table=table)

        def cell(x: float, p: Priced = p) -> str:
            return f"{usd(p.per_h(x))} ({x / p.cost:.0%})"

        rows.append([size(v), cell(p.startup), cell(p.busy), cell(p.idle), cell(p.orchestrator),
                     f"**{usd(p.per_h())}**"])
    from flora_deploy.planner.options import POOL_END_TAIL_S

    return (table_md(["Video-hours", "Startup", "GPU work", "Idle at the end", "Orchestrator",
                      "Total per video-hour"], rows)
            + f"\n\nIdle at the end is an estimate ({POOL_END_TAIL_S:.0f} s per container), "
            "not yet confirmed on a live job. Everything else comes from the measured rates "
            "above" + ("; for segment and caption that is one small job, so the large rows "
                       "are an extrapolation" if single_job(table) else "") + ".\n")


def block_levers(table) -> str:
    fill = batch_fill_table(table)
    head = ["Video-hours", "Measured rates (cold)", "Warm start"]
    if fill is not None:
        head += ["Full batches", "Both"]
    rows = []
    for v in LEVER_SIZES:
        today = price(v, SEGCAP, table=table)
        row = [size(v), per_h(today), per_h(price(v, SEGCAP, table=table, assume_warm=True))]
        if fill is not None:
            if today.episodes < BATCH_FILL_MIN_EPISODES:
                row += ["no gain (too few episodes to fill a batch)"] * 2
            else:
                row += [per_h(price(v, SEGCAP, table=fill)),
                        per_h(price(v, SEGCAP, table=fill, assume_warm=True))]
        rows.append(row)
    notes = ["Measured rates: the rates and cold startups above. Batch fill is on by default "
             "and was on when they were measured, but the measuring job was too small to fill "
             "a batch.",
             "Warm start: " + "; ".join(
                 f"{k} startup {st.startup_s:.0f} s, {_status(st)}"
                 for k in SEGCAP if (st := table.startup_for(PRESET, k, warm=True)))
             + (". Not measured yet." if any(
                 table.startup_for(PRESET, k, warm=True).status != "measured" for k in SEGCAP)
                else ".")]
    if fill is not None:
        rates = ", ".join(f"{k} {BATCH_FILL_ESTIMATE[k]} container-s per video-s"
                          for k in SEGCAP)
        notes.append(f"Full batches: what batch fill should reach on a job big enough to fill "
                     f"its batches; an estimate, not measured ({rates}, "
                     f"{band(BATCH_FILL_UNCERTAINTY)}, for jobs of at least "
                     f"{BATCH_FILL_MIN_EPISODES} episodes; the gain can be smaller when "
                     "preparing frames is the bottleneck rather than the GPU).")
    return table_md(head, rows) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"


def block_gpus(table) -> str:
    head = ["Video-hours", "1 container per stage"] + [f"`max_gpus: {g}`" for g in GPU_CAPS]
    rows = []
    for v in GPU_SIZES:
        cells = [size(v)]
        for g in (None, *GPU_CAPS):
            p = price(v, SEGCAP, table=table, max_gpus=g)
            n = "+".join(str(p.containers[k]) for k in SEGCAP)
            cells.append(f"{usd(p.per_h())}, {hours(p.wall_s)} ({n})")
        rows.append(cells)
    return (table_md(head, rows) + "\n\nEach cell: cost per video-hour, wall-clock and the "
            "containers `flora plan` picks (segment + caption).\n")


def block_batching(table) -> str:
    small = price(SMALL_JOB_H, SEGCAP, table=table)
    one = price(SMALL_JOB_H * SMALL_JOBS, SEGCAP, table=table)
    return (f"{SMALL_JOBS} separate jobs of {size(SMALL_JOB_H)} video-hours cost about "
            f"**{usd(small.cost * SMALL_JOBS)}**; the same "
            f"{size(SMALL_JOB_H * SMALL_JOBS)} video-hours as one job cost about "
            f"**{usd(one.cost)}** (segment + caption, one container per stage).\n")


def block_cameras(table) -> str:
    """1-view vs 3-view (yam-style) episodes: the 3-view rows use ``camera_scaling``."""
    head = ["Cameras per episode"] + [f"{size(v)} video-hour{'s' * (v != 1)}, per video-hour"
                                      for v in CAMERA_SIZES]
    rows = []
    for c in CAMERA_LAYOUTS:
        rows.append([f"{c}"] + [per_h(price(v, SEGCAP, table=table, cameras=c))
                                for v in CAMERA_SIZES])
    cs = table.camera_scaling
    if cs is None:
        note = "No `camera_scaling` table: 3-view episodes are priced at the 1-view rates."
    else:
        f = cs.factors(3)
        what = ", ".join(f"`{k}` x{f[k]:.2f}" for k in SEGCAP if k in f)
        state = ("an estimate, not measured on 3-view episodes" if cs.status == "estimate"
                 else "measured")
        note = (f"3-view GPU work is the 1-view rate times {what}: {state} ({cs.source}, "
                f"{cs.date}). Only the share of the work spent generating text grows with the "
                "longer three-view prompt; decoding and frame preparation are kept at 1x. "
                "`flora plan` applies the same multipliers and says so.")
    return (table_md(head, rows) + "\n\nSegment + caption, one container per stage, cold "
            "starts. " + note + "\n")


def block_prices(table) -> str:
    from flora_deploy.config.presets import get_preset
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.jobs.pools import (
        ORCHESTRATOR_CORES,
        ORCHESTRATOR_MEMORY_GIB,
        ORCHESTRATOR_PREEMPTIBLE,
    )

    rates = pinned_rates()
    spec = get_preset(PRESET)
    rows = []
    for k in SEGCAP + REWARDS:
        p = spec.pools[k]
        rows.append([f"`{k}`", p.gpu.rstrip("!") if p.gpu_count == 1 else f"{p.gpu_count}x{p.gpu}",
                     f"{p.cpu_cores:g} cores, {p.memory_gib:g} GiB",
                     f"${3600 * rates.pool_per_second(p):.2f}"])
    orch = 3600 * rates.per_second(None, 0, ORCHESTRATOR_CORES, ORCHESTRATOR_MEMORY_GIB,
                                   preemptible=ORCHESTRATOR_PREEMPTIBLE)
    rows.append(["orchestrator", "none (CPU)",
                 f"{ORCHESTRATOR_CORES:g} cores, {ORCHESTRATOR_MEMORY_GIB:g} GiB", f"${orch:.3f}"])
    return (table_md(["Stage", "GPU", "CPU and memory", "Per container-hour"], rows)
            + f"\n\nPrices: {rates.source}. Rates: {table.source}, updated {table.updated}.\n")


BLOCKS: dict[str, Callable] = {
    "short": block_short,
    "busy": block_busy,
    "breakdown": block_breakdown,
    "levers": block_levers,
    "gpus": block_gpus,
    "batching": block_batching,
    "cameras": block_cameras,
    "prices": block_prices,
}


def render(text: str | None = None, *, table=None) -> str:
    """``text`` (default: ``docs/costs.md``) with every generated block regenerated. Raises when
    a block is missing or unknown."""
    from flora_deploy.planner.rates import RateTable

    text = DOC.read_text(encoding="utf-8") if text is None else text
    table = table or RateTable.load()
    seen: set[str] = set()

    def sub(m: re.Match) -> str:
        name = m.group("name")
        if name not in BLOCKS:
            raise ValueError(f"unknown cost-table block {name!r}")
        seen.add(name)
        return m.group(1) + BLOCKS[name](table) + m.group(4)

    out = _BLOCK.sub(sub, text)
    missing = sorted(set(BLOCKS) - seen)
    if missing:
        raise ValueError(f"docs/costs.md misses cost-table blocks: {', '.join(missing)}")
    return out


def main(argv: list[str]) -> int:
    text = render()
    if argv[:1] == ["--write"]:
        DOC.write_text(text, encoding="utf-8")
        print(f"wrote {DOC}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
