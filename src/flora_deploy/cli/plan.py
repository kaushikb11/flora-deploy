"""`flora plan` (options table, optionally written back into flora.yaml) and `flora estimate`."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console, hours_text, term_width

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


def fidelity_label(o, certs: dict[str, dict] | None = None) -> str:
    """``reference`` / ``certified`` / ``partly certified`` / ``uncertified``: what the option's
    records will say (``certs``: its :func:`certificate_status`, to tell partly certified)."""
    if o.certification == "needs certificates" and certs and any(
            c.get("state") in ("certified", "reference") for c in certs.values()):
        return "partly certified"
    return {"needs certificates": "uncertified"}.get(o.certification, o.certification)


def fidelity_detail(o, certs: dict[str, dict] | None = None) -> str:
    """:func:`fidelity_label`, with the stages behind ``partly certified`` named, e.g.
    ``partly certified (reward.terminal, reward.dense: reference; segment, caption: not
    certified)``."""
    label = fidelity_label(o, certs)
    if label != "partly certified":
        return label
    groups: dict[str, list[str]] = {}
    for kind, c in (certs or {}).items():
        st = c.get("state")
        groups.setdefault(st if st in ("reference", "certified") else "not certified",
                          []).append(kind)
    return label + " (" + "; ".join(f"{', '.join(ks)}: {st}" for st, ks in groups.items()) + ")"


def display_flags(o, certs: dict[str, dict] | None = None) -> tuple[str, ...]:
    """``o.flags`` with the planner's ``<preset>: uncertified`` wording replaced by the same
    fidelity words the table shows (:func:`fidelity_detail`), so one option has one label."""
    from flora_deploy.planner.options import UNCERTIFIED

    head = f"{o.preset}: uncertified"
    blocked = f"{head} until every stage is certified; set "
    detail = fidelity_detail(o, certs)
    out = []
    for f in o.flags:
        if f == "unavailable: " + UNCERTIFIED.format(preset=o.preset):
            continue  # the uncertified flag below already says it and how to opt in
        if f.startswith(blocked):
            f = f"{o.preset}: {detail}; set {f[len(blocked):]}"
        elif f.startswith(head):
            f = f"{o.preset}: {detail}{f[len(head):]}"
        out.append(f)
    return tuple(out)


NEEDS_OPT_IN = "needs allow_uncertified: true"


def status_label(o) -> str:
    """``needs allow_uncertified: true`` when the only thing blocking the option is that a stage
    is not certified, ``unavailable`` when something else blocks it, ``projection`` when it
    needs a feature that is not deployed, else ``ok`` / ``at risk`` / ``infeasible`` (budget,
    deadline or GPU cap)."""
    from flora_deploy.planner.options import UNCERTIFIED

    if o.unavailable and o.status != "infeasible":
        if tuple(o.unavailable) == (UNCERTIFIED.format(preset=o.preset),):
            return NEEDS_OPT_IN
        return "unavailable"
    return "projection" if o.requires else o.status


_PINNED_GPU = re.compile(r"\b([A-Z][A-Za-z0-9-]*)!")


def gpu_text(text: str) -> str:
    """User-facing GPU names: flora.yaml's ``H100!`` (exactly an H100, never upgraded) reads
    ``H100 only``."""
    return _PINNED_GPU.sub(r"\1 only", text)


def write_command(n: int, config: Path, args: list[str]) -> str:
    """The ``flora plan`` command that writes option ``n``: the options the user passed (they
    change the options and their numbers), then ``--write n``."""
    import shlex

    parts = ["flora", "plan"]
    if config != Path("flora.yaml"):
        parts += ["-c", str(config)]
    return shlex.join([*parts, *args, "--write", str(n)])


def certificate_note(cfg, o, content, certs: dict[str, dict] | None = None) -> str | None:
    """One line on the option's certificates (``None`` for the reference preset): each stage's
    status in plain words, or that none has one yet."""
    if o.preset == "exact":
        return None
    certs = certs if certs is not None else certificate_status(cfg, o, content)
    if all(c.get("state") == "none" for c in certs.values()):
        return (f"no accuracy certificates for {o.preset} yet (its records say certified: "
                "false; `exact` is the reference)")
    return "; ".join(f"{k} {c['label']}" for k, c in certs.items())


def _gpus_label(o) -> str:
    from flora_deploy.planner.options import episodes_label

    def one(k) -> str:
        boxes = f"{k.containers} container{'s' if k.containers != 1 else ''}"
        return f"{k.kind} {boxes} × {k.gpu_count} {gpu_text(k.gpu)}, {episodes_label(k)}"

    return "; ".join(one(k) for k in o.kinds)


def _print_plan(cfg, workload, req, deadline_s, options, best, rates, table, content,
                write_cmd=None) -> None:
    from rich.markup import escape
    from rich.table import Table

    from flora_deploy.planner.options import rates_label

    dl = f", deadline {deadline_s / 3600:.0f} h" if deadline_s else ""
    layouts: dict[int, float] = {}
    for g in workload.groups:
        layouts[g.cameras] = layouts.get(g.cameras, 0.0) + g.video_s
    views = ("" if set(layouts) <= {1} else " (" + ", ".join(
        f"{c}-view {hours_text(s / 3600)} h" for c, s in sorted(layouts.items())) + ")")
    console.print(f"[bold]{escape(cfg.project)}[/bold]: {hours_text(workload.video_s / 3600)} "
                  f"video-hours{views}, {workload.episodes:,} episodes; budget "
                  f"{_money(req.budget_usd or 0)}{dl}, max {req.max_gpus} GPUs", soft_wrap=True)
    colour = {"ok": "green", "at risk": "yellow", "infeasible": "red", "unavailable": "red",
              "projection": "yellow", NEEDS_OPT_IN: "yellow"}

    def cost(o) -> str:
        return f"{_money(o.cost_usd)} ({_money(o.cost_low_usd)}-{_money(o.cost_high_usd)})"

    def wall(o) -> str:
        return f"{o.wall_s / 3600:.1f} h (<= {o.wall_high_s / 3600:.1f} h)"

    def status(o) -> str:
        s = status_label(o)
        return f"[{colour[s]}]{s}[/{colour[s]}]"

    cs = {o.name: certificate_status(cfg, o, content) for o in options if o.preset != "exact"}
    wide = term_width() >= WIDE_COLUMNS
    if wide:
        t = Table(show_lines=False)
        for col in ("#", "option", "cost (range)", "wall-clock", "GPUs", "fidelity", "rates",
                    "status"):
            t.add_column(col)
        for i, o in enumerate(options, 1):
            t.add_row(str(i) + (" *" if o is best else ""), escape(o.name), cost(o), wall(o),
                      str(o.peak_gpus), fidelity_label(o, cs.get(o.name)),
                      escape(rates_label(o)), status(o))
        console.print(t)
    for i, o in enumerate(options, 1):
        if not wide:
            mark = " [bold](recommended)[/bold]" if o is best else ""
            console.print(f"[bold]{i}. {escape(o.name)}[/bold]{mark}: {status(o)}, "
                          f"{fidelity_label(o, cs.get(o.name))}", soft_wrap=True)
            console.print(f"   cost {cost(o)}", soft_wrap=True)
            console.print(f"   wall-clock {wall(o)}, {o.peak_gpus} GPUs at peak", soft_wrap=True)
            console.print(f"   rates {escape(rates_label(o))}", soft_wrap=True)
        if wide:  # the table names the option; the stacked layout has its heading above
            console.print(f"[bold]{i}. {escape(o.name)}[/bold]", soft_wrap=True)
        console.print(f"   pools: {escape(_gpus_label(o))}", soft_wrap=True)
        for f in display_flags(o, cs.get(o.name)):
            console.print(f"   {escape(gpu_text(f))}", soft_wrap=True)
        note = certificate_note(cfg, o, content, cs.get(o.name))
        if note:
            console.print(f"   certificates: {escape(gpu_text(note))}", soft_wrap=True)
    if any("!" in k.gpu for o in options for k in o.kinds):
        console.print("H100 only: exactly an H100; Modal never upgrades it to a pricier GPU "
                      "(flora.yaml writes it `H100!`).", soft_wrap=True)
    from flora_deploy.planner.options import POOL_END_TAIL_S

    console.print("Cost: the all-in estimate for this workload at Modal list prices: GPU work, "
                  "plus each container's startup, a short idle tail once its stage has no "
                  f"work left (about {POOL_END_TAIL_S:g} s, then it scales to zero) and the "
                  "orchestrator. `flora report` shows what a job actually spent.",
                  soft_wrap=True)
    console.print(f"Prices: {escape(rates.source)}" +
                  (f" (x{rates.region_multiplier} region)" if rates.region_multiplier != 1
                   else "") + f". Rates: {escape(table.source)} ({table.updated}).",
                  soft_wrap=True)
    if best is not None:
        n = options.index(best) + 1
        cmd = write_cmd(n) if write_cmd else f"flora plan --write {n}"
        console.print(f"Recommended: {n} ({escape(best.name)}), the cheapest option your "
                      f"config can run. Write it into flora.yaml with `{escape(cmd)}`.",
                      soft_wrap=True)
        return
    fits = [o for o in options if o.status != "infeasible" and not o.requires]
    if fits:  # something fits the budget and deadline but cannot run: say why, not "raise it"
        from flora_deploy.planner.options import UNCERTIFIED

        def reasons(o) -> str:
            opt_in = (f"{UNCERTIFIED.format(preset=o.preset)} (set `allow_uncertified: true` "
                      "in flora.yaml to run it anyway)")
            return ", ".join(opt_in if u == UNCERTIFIED.format(preset=o.preset) else u
                             for u in o.unavailable)

        why = "; ".join(f"{options.index(o) + 1} ({o.name}): {reasons(o)}" for o in fits)
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


def local_index_summary(config: Path, cfg) -> Path | None:
    """The index summary ``flora run --local`` left in the default local store
    (``<config dir>/flora-local-store``) for the config's job or its fake-models twin
    (``<project>.fake``), or ``None``."""
    from flora_deploy.cli._jobref import local_store_dir
    from flora_deploy.jobs.indexing import summary_path
    from flora_deploy.jobs.store import fake_job_id

    root = local_store_dir(config)
    for jid in dict.fromkeys((cfg.project, fake_job_id(cfg.project))):
        p = root / summary_path(jid)
        if p.is_file():
            return p
    return None


def no_workload_message(config: Path, cfg, index: Path) -> str:
    """Why ``flora plan`` has no workload, and the free command that writes one. A config whose
    input is a local directory gets ``flora run --local --fake-models``: ``flora index --local``
    writes its receipts to ``output.sink``, which for a ``volume://`` sink creates a Modal
    Volume."""
    alt = "or give --index FILE or --video-hours H [--episodes N]"
    if cfg.input.repo is None:
        p = Path(str(cfg.input.path)).expanduser()
        missing = not (p if p.is_absolute() else config.parent / p).exists()
        cmd = "flora run --local --fake-models" + (
            " --sample-data" if missing and cfg.input.format in ("mp4_dir", "lerobot_v3")
            else "") + ("" if config == Path("flora.yaml") else f" --config {config}")
        return (f"no workload: {index} not found, and no `flora run --local` index in "
                f"{config.parent / 'flora-local-store'}. Run `{cmd}` first (free: it indexes "
                f"the dataset on this machine), {alt}")
    return (f"no workload: {index} not found. Run `flora index` first (it writes it; "
            f"`flora index --local` indexes on this machine instead of Modal containers), {alt}")


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
        help="Read live prices from your Modal workspace (needs Modal credentials).")] = False,
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

    passed: list[str] = []  # the options that shape the plan, for the `--write N` hint
    for flag, value in (("--budget", budget), ("--deadline", deadline),
                        ("--video-hours", video_hours), ("--episodes", episodes),
                        ("--cameras", cameras), ("--index", index), ("--max-gpus", max_gpus),
                        ("--rates", rates_file)):
        if value is not None:
            passed += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
    for p in preset or ():
        passed += ["--preset", p]
    passed += [f for f, on in (("--live-prices", live_prices), ("--check-cache", check_cache),
                               ("--assume-warm", assume_warm)) if on]

    try:
        if index is None and video_hours is None:
            from flora_deploy.cli.index import default_index_summary

            index = default_index_summary(config)
            if not index.is_file():
                found = local_index_summary(config, cfg)
                if found is None:
                    raise _fail(no_workload_message(config, cfg, index))
                index = found
                if not as_json:
                    console.print(f"workload: {escape(str(index))} (the index `flora run "
                                  "--local` wrote)", soft_wrap=True, highlight=False)
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
        def option_json(o) -> dict:
            # the table's fidelity words (fidelity_label), not the planner's internal state
            certs = certificate_status(cfg, o, content)
            return {**o.as_dict(), "certification": fidelity_label(o, certs),
                    "flags": list(display_flags(o, certs)), "rates": rates_label(o),
                    "certificates": certs}

        typer.echo(json.dumps({
            "workload": {"video_hours": round(workload.video_s / 3600, 3),
                         "episodes": workload.episodes},
            "budget_usd": req.budget_usd, "deadline_s": deadline_s, "max_gpus": req.max_gpus,
            "prices": rates.source, "rates": table.source,
            "recommended": options.index(best) + 1 if best else None,
            "options": [option_json(o) for o in options]}, indent=2))
    else:
        _print_plan(cfg, workload, req, deadline_s, options, best, rates, table, content,
                    write_cmd=lambda n: write_command(n, config, passed))

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


@app.command(hidden=True)  # measuring rates on a sample is not wired yet
def estimate(
    config: Annotated[Path, typer.Option("--config", "-c", help="flora.yaml")] = Path("flora.yaml"),
    sample: Annotated[int, typer.Option(
        help="Episodes to run for calibration (not wired yet: without --rates it exits 2).")] = 20,
    rates_file: Annotated[Path | None, typer.Option(
        "--rates", help="A rates.json to show, per stage of this config.")] = None,
) -> None:
    """Show the per-stage rates of a rates.json (--rates). Measuring them on a sample of your
    dataset is not wired yet: without --rates it says so and exits 2."""
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
