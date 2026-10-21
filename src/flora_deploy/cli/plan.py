"""`flora plan` (options table, optionally written back into flora.yaml) and `flora estimate`."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console

_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)
_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "": 3600}


def parse_duration(text: str) -> float:
    """``48h`` / ``90m`` / ``2d`` / ``3600s``; a bare number is hours."""
    m = _DUR.match(text)
    if not m or float(m.group(1)) <= 0:
        raise typer.BadParameter(f"{text!r}: use e.g. 48h, 90m, 2d or 3600s")
    return float(m.group(1)) * _UNIT[m.group(2).lower()]


def _money(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def _fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]")
    return typer.Exit(code=code)


def certificate_status(cfg, option) -> dict[str, dict]:
    """Per shard kind of ``option``: the Gate certificate status of its (kind, GPU) pool
    (``validate.certkey.status``: reference / certified / provisional / fail / none)."""
    from types import SimpleNamespace

    from flora_deploy.config.hashing import kind_modes
    from flora_deploy.validate.certkey import status

    pools = {k.kind: SimpleNamespace(gpu=k.gpu) for k in option.kinds}
    return status(option.preset, pools, modes={k: kind_modes(cfg, k) for k in pools})


def _cameras_from_config(cfg) -> int:
    cams = cfg.input.cameras
    return len(cams) if cams else 1


@app.command()
def plan(
    config: Annotated[Path, typer.Option("--config", "-c", help="flora.yaml")] = Path("flora.yaml"),
    budget: Annotated[float | None, typer.Option(
        help="USD cap (default: modal.budget_usd)")] = None,
    deadline: Annotated[str | None, typer.Option(help="e.g. 48h, 90m, 2d")] = None,
    preset: Annotated[list[str] | None, typer.Option(
        help="Preset(s) to consider (repeatable; default: every available preset).")] = None,
    video_hours: Annotated[float | None, typer.Option(
        help="Total video hours (until `flora index` writes a summary).")] = None,
    episodes: Annotated[int | None, typer.Option(help="Episode count (with --video-hours).")
                        ] = None,
    cameras: Annotated[int | None, typer.Option(help="Views per episode (default: config).")
                       ] = None,
    index: Annotated[Path | None, typer.Option(
        help='Workload summary JSON: {"groups": [{pool, cameras, video_s, episodes}]}.')] = None,
    max_gpus: Annotated[int | None, typer.Option(help="GPU cap (default: modal.max_gpus)")
                        ] = None,
    rates_file: Annotated[Path | None, typer.Option(
        "--rates", help="rates.json to use instead of the shipped one.")] = None,
    live_prices: Annotated[bool, typer.Option(
        help="Read prices from Workspace.billing.rates() (needs Modal credentials).")] = False,
    write: Annotated[int | None, typer.Option(
        help="Write option N (1-based) into the config.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print options as JSON.")] = False,
) -> None:
    """Show 2-4 run options (cost, wall-clock, GPUs, certification) and optionally pick one."""
    import yaml
    from rich.markup import escape
    from rich.table import Table

    from flora_deploy.config.presets import PresetUnavailableError
    from flora_deploy.config.schema import ConfigError, load_config, parse_config
    from flora_deploy.jobs.budget import load_rates, modal_rates_provider, region_multiplier_for
    from flora_deploy.planner.options import (
        PlanRequest,
        WorkloadSummary,
        apply_option,
        default_presets,
        plan_options,
        rates_label,
        recommended,
        with_pool_overrides,
    )
    from flora_deploy.planner.rates import RatesFileError, RateTable

    try:
        cfg = load_config(config)
    except FileNotFoundError:
        raise _fail(f"{config} not found; run `flora init` first") from None
    except ConfigError as e:
        raise _fail(str(e)) from None

    try:
        if index is not None:
            workload = WorkloadSummary.from_dict(json.loads(index.read_text(encoding="utf-8")))
        elif video_hours is not None:
            workload = WorkloadSummary.from_hours(
                video_hours, cameras=cameras or _cameras_from_config(cfg),
                pool=cfg.pipeline.embodiment, episodes=episodes)
        else:
            raise _fail("give the workload: --video-hours H [--episodes N] or --index "
                        "summary.json (`flora index` will write it once wired)")
    except (ValueError, KeyError, OSError) as e:
        raise _fail(f"bad workload: {e}") from None

    try:
        table = RateTable.load(rates_file)
    except (RatesFileError, OSError) as e:
        raise _fail(str(e)) from None
    region = region_multiplier_for(cfg.modal.region)
    rates = load_rates(modal_rates_provider if live_prices else None, region_multiplier=region)

    presets = tuple(preset or default_presets())  # balanced first
    deadline_s = parse_duration(deadline) if deadline else None
    req = PlanRequest(workload=workload, stages=cfg.pipeline.stages, presets=presets,
                      deadline_s=deadline_s,
                      budget_usd=budget if budget is not None else cfg.modal.budget_usd,
                      max_gpus=max_gpus or cfg.modal.max_gpus)
    req = with_pool_overrides(req, cfg.preset, cfg.pools())
    try:
        options = plan_options(req, table=table, rates=rates)
    except (PresetUnavailableError, KeyError) as e:
        raise _fail(str(e.args[0] if e.args else e)) from None
    best = recommended(options)

    if as_json:
        typer.echo(json.dumps({
            "workload": {"video_hours": round(workload.video_s / 3600, 3),
                         "episodes": workload.episodes},
            "budget_usd": req.budget_usd, "deadline_s": deadline_s, "max_gpus": req.max_gpus,
            "prices": rates.source, "rates": table.source,
            "recommended": options.index(best) + 1 if best else None,
            "options": [{**o.as_dict(), "rates": rates_label(o),
                         "certificates": certificate_status(cfg, o)}
                        for o in options]}, indent=2))
    else:
        dl = f", deadline {deadline_s / 3600:.0f} h" if deadline_s else ""
        console.print(f"[bold]{escape(cfg.project)}[/bold]: {workload.video_s / 3600:,.1f} "
                      f"video-hours, {workload.episodes:,} episodes; budget "
                      f"{_money(req.budget_usd or 0)}{dl}, max {req.max_gpus} GPUs")
        t = Table(show_lines=False)
        for col in ("#", "option", "GPUs (containers x inputs)", "cost (range)", "wall-clock",
                    "peak GPUs", "fidelity", "rates", "status"):
            t.add_column(col)
        for i, o in enumerate(options, 1):
            gpus = ", ".join(
                f"{k.kind} {k.containers}x{k.gpu}" + (f":{k.gpu_count}" if k.gpu_count > 1
                                                       else "") + f" x{k.max_inputs}"
                for k in o.kinds)
            mark = " *" if o is best else ""
            colour = {"ok": "green", "at risk": "yellow", "infeasible": "red"}[o.status]
            status = f"[{colour}]{o.status}[/{colour}]"
            if o.unavailable:
                status += " [red](unavailable)[/red]"
            t.add_row(str(i) + mark, o.name, gpus,
                      f"{_money(o.cost_usd)} ({_money(o.cost_low_usd)}-{_money(o.cost_high_usd)})",
                      f"{o.wall_s / 3600:.1f} h (<= {o.wall_high_s / 3600:.1f} h)",
                      str(o.peak_gpus), o.certification, rates_label(o), status)
        console.print(t)
        for i, o in enumerate(options, 1):
            for f in o.flags:
                console.print(f"  {i}. {escape(f)}")
            if o.preset != "exact":
                certs = certificate_status(cfg, o)
                console.print(f"  {i}. certificates: " + ", ".join(
                    f"{k} on {c['gpu']}: {c['status']}" + (f" ({c['certificate']})"
                                                         if c["certificate"] else "")
                    for k, c in certs.items()))
        console.print(f"Prices: {escape(rates.source)}" +
                      (f" (x{rates.region_multiplier} region)" if rates.region_multiplier != 1
                       else "") + f". Rates: {escape(table.source)} ({table.updated}).")
        if best is None:
            console.print("[red]No option fits (of those available): raise --budget / --deadline / "
                          "max_gpus, or label fewer episodes.[/red]")
        else:
            console.print(f"Recommended: {options.index(best) + 1} ({escape(best.name)}). "
                          f"Write it with `flora plan ... --write {options.index(best) + 1}`.")

    if write is not None:
        if not 1 <= write <= len(options):
            raise _fail(f"--write {write}: choose 1..{len(options)}")
        chosen = options[write - 1]
        if chosen.status == "infeasible":
            raise _fail(f"option {write} ({chosen.name}) is infeasible; not writing it", 1)
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        new = apply_option(data, chosen, budget_usd=budget)
        try:
            parse_config(new, source=str(config))
        except ConfigError as e:
            raise _fail(str(e), 1) from None
        backup = config.with_suffix(config.suffix + ".bak")
        backup.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
        config.write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
        console.print(f"Wrote option {write} ({escape(chosen.name)}) to {escape(str(config))} "
                      f"(previous file, with its comments, saved as {escape(str(backup))}).")


@app.command()
def estimate(
    config: Annotated[Path, typer.Option("--config", "-c", help="flora.yaml")] = Path("flora.yaml"),
    sample: Annotated[int, typer.Option(help="Episodes to run for calibration.")] = 20,
    rates_file: Annotated[Path | None, typer.Option(
        "--rates", help="A rates.json (e.g. from an earlier estimate) to show instead.")] = None,
) -> None:
    """Measure per-stage rates on a sample of the dataset (needs a deployed app)."""
    from rich.markup import escape
    from rich.table import Table

    from flora_deploy.config.presets import get_preset, shard_kinds_for
    from flora_deploy.config.schema import ConfigError, load_config
    from flora_deploy.planner.rates import RatesFileError, RateTable

    if sample < 1:
        raise _fail("--sample must be >= 1")
    try:
        cfg = load_config(config)
    except FileNotFoundError:
        raise _fail(f"{config} not found; run `flora init` first") from None
    except ConfigError as e:
        raise _fail(str(e)) from None
    if rates_file is None:
        console.print(
            f"`flora estimate --sample {sample}` runs {sample} episodes of your dataset on the "
            "deployed workers and writes measured rates. It needs a deployed app (`flora "
            "deploy`), which is not wired yet. Until then `flora plan` uses the shipped rates "
            "(planner/rates.json); pass --rates FILE to inspect a rates file.")
        raise typer.Exit(code=2)
    try:
        table = RateTable.load(rates_file)
    except (RatesFileError, OSError) as e:
        raise _fail(str(e)) from None
    t = Table(title=f"{cfg.preset} rates from {escape(table.source)}")
    for col in ("kind", "cameras", "inputs", "container-s / video-s", "+/-", "startup s",
                "status", "date"):
        t.add_column(col)
    seen = set()
    for kind in shard_kinds_for(get_preset(cfg.preset), cfg.pipeline.stages):
        st = table.startup_for(cfg.preset, kind)
        for c in (1, 3):
            for conc in table.concurrencies(cfg.preset, kind, c) or [1]:
                try:
                    e = table.rate(cfg.preset, kind, c, conc)
                except KeyError:
                    continue
                if e in seen:
                    continue
                seen.add(e)
                t.add_row(kind, str(e.cameras), str(e.concurrency),
                          f"{e.container_s_per_video_s:g}",
                          f"{e.uncertainty:.0%}", f"{st.startup_s:g}", e.status, e.date)
    console.print(t)
