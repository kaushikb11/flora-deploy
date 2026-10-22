"""`flora plan` (options table, optionally written back into flora.yaml) and `flora estimate`."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console, term_width

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


def certificate_status(cfg, option, derived_content=None) -> dict[str, dict]:
    """Per shard kind of ``option``: the certificate status of its (kind, GPU) pool
    (``cli._certs``: reference / certified / provisional / fail / none /
    unverified_weights, and why), for the config's embodiment, modes and derived content
    digests."""
    from flora_deploy.cli._certs import certificate_report

    pools = {k.kind: k.gpu for k in option.kinds}
    return certificate_report(cfg, pools, preset=option.preset,
                              derived_content=derived_content)


def certified_options(cfg, options, derived_content=None) -> set[str]:
    """Names of the options whose every pool is ``reference`` or ``certified``."""
    return {o.name for o in options
            if all(c["status"] in ("reference", "certified")
                   for c in certificate_status(cfg, o, derived_content).values())}


# Below this terminal width the options are stacked (one block each) instead of a table.
WIDE_COLUMNS = 120


def fidelity_label(o) -> str:
    """``reference`` / ``certified`` / ``uncertified``: what the option's records will say."""
    return {"needs certificates": "uncertified"}.get(o.certification, o.certification)


def status_label(o) -> str:
    """``unavailable`` when something blocks the option, ``projection`` when it needs a feature
    that is not deployed, else ``ok`` / ``at risk`` / ``infeasible`` (budget, deadline or GPU
    cap)."""
    if o.unavailable and o.status != "infeasible":
        return "unavailable"
    return "projection" if o.requires else o.status


def certificate_note(cfg, o, content) -> str | None:
    """One line on the option's certificates (``None`` for the reference preset): the pools that
    have one, or that none do yet."""
    if o.preset == "exact":
        return None
    certs = certificate_status(cfg, o, content)
    ok = [k for k, c in certs.items() if c["status"] in ("certified", "reference")]
    if not ok:
        return (f"no accuracy certificates for {o.preset} yet (its records say certified: "
                "false; `exact` is the reference)")
    if len(ok) == len(certs):
        return "certified on every pool"
    return "certified: " + ", ".join(ok) + "; not yet: " + ", ".join(
        k for k in certs if k not in ok)


def _gpus_label(o) -> str:
    from flora_deploy.planner.options import episodes_label

    return ", ".join(f"{k.kind} {k.containers}x{k.gpu}" + (f":{k.gpu_count}" if k.gpu_count > 1
                                                          else "") + episodes_label(k)
                     for k in o.kinds)


def _print_plan(cfg, workload, req, deadline_s, options, best, rates, table, content) -> None:
    from rich.markup import escape
    from rich.table import Table

    from flora_deploy.planner.options import rates_label

    dl = f", deadline {deadline_s / 3600:.0f} h" if deadline_s else ""
    console.print(f"[bold]{escape(cfg.project)}[/bold]: {workload.video_s / 3600:,.1f} "
                  f"video-hours, {workload.episodes:,} episodes; budget "
                  f"{_money(req.budget_usd or 0)}{dl}, max {req.max_gpus} GPUs", soft_wrap=True)
    colour = {"ok": "green", "at risk": "yellow", "infeasible": "red", "unavailable": "red",
              "projection": "yellow"}

    def cost(o) -> str:
        return f"{_money(o.cost_usd)} ({_money(o.cost_low_usd)}-{_money(o.cost_high_usd)})"

    def wall(o) -> str:
        return f"{o.wall_s / 3600:.1f} h (<= {o.wall_high_s / 3600:.1f} h)"

    def status(o) -> str:
        s = status_label(o)
        return f"[{colour[s]}]{s}[/{colour[s]}]"

    wide = term_width() >= WIDE_COLUMNS
    if wide:
        t = Table(show_lines=False)
        for col in ("#", "option", "cost (range)", "wall-clock", "GPUs", "fidelity", "rates",
                    "status"):
            t.add_column(col)
        for i, o in enumerate(options, 1):
            t.add_row(str(i) + (" *" if o is best else ""), escape(o.name), cost(o), wall(o),
                      str(o.peak_gpus), fidelity_label(o), escape(rates_label(o)), status(o))
        console.print(t)
    for i, o in enumerate(options, 1):
        if not wide:
            mark = " [bold](recommended)[/bold]" if o is best else ""
            console.print(f"[bold]{i}. {escape(o.name)}[/bold]{mark}: {status(o)}, "
                          f"{fidelity_label(o)}", soft_wrap=True)
            console.print(f"   cost {cost(o)}", soft_wrap=True)
            console.print(f"   wall-clock {wall(o)}, {o.peak_gpus} GPUs at peak", soft_wrap=True)
            console.print(f"   rates {escape(rates_label(o))}", soft_wrap=True)
        console.print(f"  {i}. pools: {escape(_gpus_label(o))}", soft_wrap=True)
        for f in o.flags:
            console.print(f"  {i}. {escape(f)}", soft_wrap=True)
        note = certificate_note(cfg, o, content)
        if note:
            console.print(f"  {i}. certificates: {escape(note)}", soft_wrap=True)
    if any("!" in k.gpu for o in options for k in o.kinds):
        console.print("H100! means exactly an H100 (Modal never upgrades it to a pricier GPU).",
                      soft_wrap=True)
    console.print("Cost: the all-in estimate for this workload at Modal list prices: GPU work, "
                  "plus each container's startup and the orchestrator (idle time after the "
                  "work is not included). `flora report` shows what a job actually spent.",
                  soft_wrap=True)
    console.print(f"Prices: {escape(rates.source)}" +
                  (f" (x{rates.region_multiplier} region)" if rates.region_multiplier != 1
                   else "") + f". Rates: {escape(table.source)} ({table.updated}).",
                  soft_wrap=True)
    if best is not None:
        n = options.index(best) + 1
        console.print(f"Recommended: {n} ({escape(best.name)}), the cheapest option your "
                      f"config can run. Write it with `flora plan ... --write {n}`.",
                      soft_wrap=True)
        return
    fits = [o for o in options if o.status != "infeasible" and not o.requires]
    if fits:  # something fits the budget and deadline but cannot run: say why, not "raise it"
        why = "; ".join(f"{options.index(o) + 1} ({o.name}): {', '.join(o.unavailable)}"
                        for o in fits)
        console.print(f"[red]No runnable option: {escape(why)}.[/red]", soft_wrap=True)
    else:
        console.print("[red]No option fits the budget, deadline and GPU cap: raise --budget / "
                      "--deadline / max_gpus, or label fewer episodes.[/red]", soft_wrap=True)


def published_cache_kinds(cfg, exists=None) -> tuple[str, ...]:
    """The balanced shard kinds whose compile cache is published for the key of the pool they
    would run on (``engines.compile_cache.published_kinds``); ``exists``: a path probe of the
    ``flora-cache`` Volume (default: through the Modal client). Any error: none (cold)."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.engines.compile_cache import published_kinds

    pools = dict(get_preset("balanced").pools)
    if cfg.preset == "balanced":
        pools.update(cfg.pools())
    gpus = {k: p.gpu for k, p in pools.items()}
    if exists is None:
        try:
            import modal

            from flora_deploy.jobs.stores.client_volume import ClientVolumeStore
            from flora_deploy.layout import CACHE_VOLUME

            exists = ClientVolumeStore(modal.Volume.from_name(CACHE_VOLUME, version=2)).exists
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]--check-cache: cannot read flora-cache ({e}); pricing "
                          "startups cold[/yellow]")
            return ()
    return published_kinds(gpus, exists)


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
        help="Total video hours, instead of the `flora index` summary.")] = None,
    episodes: Annotated[int | None, typer.Option(help="Episode count (with --video-hours).")
                        ] = None,
    cameras: Annotated[int | None, typer.Option(help="Views per episode (default: config).")
                       ] = None,
    index: Annotated[Path | None, typer.Option(
        help="Workload summary JSON (default: the summary.json `flora index` wrote next to "
             "the config).", show_default=False)] = None,
    max_gpus: Annotated[int | None, typer.Option(help="GPU cap (default: modal.max_gpus)")
                        ] = None,
    rates_file: Annotated[Path | None, typer.Option(
        "--rates", help="rates.json to use instead of the shipped one.")] = None,
    live_prices: Annotated[bool, typer.Option(
        help="Read prices from Workspace.billing.rates() (needs Modal credentials).")] = False,
    write: Annotated[int | None, typer.Option(
        help="Write option N (1-based) into the config.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print options as JSON.")] = False,
    check_cache: Annotated[bool, typer.Option(
        "--check-cache", help="Price balanced pools whose compile cache is published on "
        "flora-cache (needs Modal credentials) with the warm startup row.")] = False,
    assume_warm: Annotated[bool, typer.Option(
        "--assume-warm", help="Projection: price every pool's startup warm, as if a compile "
        "cache were published.")] = False,
) -> None:
    """Show 2-4 run options (cost, wall-clock, GPUs, certification) and optionally pick one."""
    import yaml
    from rich.markup import escape

    from flora_deploy.config.presets import PresetUnavailableError
    from flora_deploy.config.receipts import load_certified
    from flora_deploy.config.schema import ConfigError, parse_config
    from flora_deploy.jobs.budget import load_rates, modal_rates_provider, region_multiplier_for
    from flora_deploy.planner.options import (
        PlanRequest,
        WorkloadSummary,
        apply_option,
        default_presets,
        plan_options,
        rates_label,
        recommended,
        with_certificates,
        with_pool_overrides,
    )
    from flora_deploy.planner.rates import RatesFileError, RateTable

    try:
        cfg, content = load_certified(config)
    except FileNotFoundError:
        raise _fail(f"{config} not found; run `flora init` first") from None
    except ConfigError as e:
        raise _fail(str(e)) from None

    try:
        if index is None and video_hours is None:
            from flora_deploy.cli.index import default_index_summary

            index = default_index_summary(config)
            if not index.is_file():
                raise _fail(f"no workload: {index} not found. Run `flora index` first (it "
                            "writes it), or give --index FILE or --video-hours H")
        if index is not None:
            workload = WorkloadSummary.from_dict(json.loads(index.read_text(encoding="utf-8")))
        elif video_hours is not None:
            workload = WorkloadSummary.from_hours(
                video_hours, cameras=cameras or _cameras_from_config(cfg),
                pool=cfg.pipeline.embodiment, episodes=episodes)
        else:
            raise AssertionError("unreachable: the index default is set above")
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
    warm_kinds = published_cache_kinds(cfg) if check_cache else ()
    req = PlanRequest(workload=workload, stages=cfg.pipeline.stages, presets=presets,
                      deadline_s=deadline_s,
                      budget_usd=budget if budget is not None else cfg.modal.budget_usd,
                      max_gpus=max_gpus or cfg.modal.max_gpus,
                      allow_uncertified=cfg.allow_uncertified, warm_kinds=warm_kinds,
                      assume_warm=assume_warm)
    req = with_pool_overrides(req, cfg.preset, cfg.pools())
    try:
        options = plan_options(req, table=table, rates=rates)
    except (PresetUnavailableError, KeyError) as e:
        raise _fail(str(e.args[0] if e.args else e)) from None
    certified = certified_options(cfg, options, content)
    options = with_certificates(options, certified)
    best = recommended(options, certified=certified)

    if as_json:
        typer.echo(json.dumps({
            "workload": {"video_hours": round(workload.video_s / 3600, 3),
                         "episodes": workload.episodes},
            "budget_usd": req.budget_usd, "deadline_s": deadline_s, "max_gpus": req.max_gpus,
            "prices": rates.source, "rates": table.source,
            "recommended": options.index(best) + 1 if best else None,
            "options": [{**o.as_dict(), "rates": rates_label(o),
                         "certificates": certificate_status(cfg, o, content)}
                        for o in options]}, indent=2))
    else:
        _print_plan(cfg, workload, req, deadline_s, options, best, rates, table, content)

    if write is not None:
        if not 1 <= write <= len(options):
            raise _fail(f"--write {write}: choose 1..{len(options)}")
        chosen = options[write - 1]
        if chosen.status == "infeasible":
            raise _fail(f"option {write} ({chosen.name}) is infeasible; not writing it", 1)
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        new = apply_option(data, chosen, budget_usd=budget)
        try:
            parse_config(new, source=str(config), derived_content=content)
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
