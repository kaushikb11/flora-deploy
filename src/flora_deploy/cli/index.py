"""`flora index`: dataset -> per-episode index receipts + ``index/summary.json`` in the job store.

Default: runs on Modal CPU containers (``modal_app/index_app.py``). ``--local`` runs in-process
(small datasets, tests). The summary is also written locally (``--out``, default
``summary.json`` next to the config: :func:`default_index_summary`), where ``flora plan`` reads
it by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console


def _fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]")
    return typer.Exit(code=code)


INDEX_SUMMARY_NAME = "summary.json"


def default_index_summary(config: Path) -> Path:
    """Where ``flora index`` writes its summary and ``flora plan`` reads it by default: next to
    the config (``./summary.json`` for ``./flora.yaml``)."""
    return config.parent / INDEX_SUMMARY_NAME


def store_for_sink(sink: str) -> str:
    """``output.sink`` -> a job-store URI (``volume://`` -> ``modal-volume://``)."""
    from flora_deploy.config.schema import unsupported_sink_message

    why = unsupported_sink_message(sink)
    if why:
        raise ValueError(f"sink {sink!r}: {why}")
    if sink.startswith("volume://"):
        return "modal-volume://" + sink[len("volume://"):]
    if sink.startswith(("file://", "s3://", "modal-volume://")):
        return sink
    raise ValueError(f"sink {sink!r}: pass --store (file://, s3:// or modal-volume://)")


def _require_volume(st, store_uri: str, environment: str | None = None) -> None:
    """``--local`` with a Modal Volume store: refuse before downloading anything when the Volume
    does not exist yet (the index would otherwise fail every episode's receipt write, then crash
    with a traceback). One metadata lookup, as ``flora doctor`` makes; nothing is created."""
    vol = getattr(st, "volume", None)
    if vol is None or not hasattr(vol, "hydrate"):
        return
    try:
        vol.hydrate()
    except Exception as e:  # noqa: BLE001 - modal.exception.NotFoundError, classified by name
        if type(e).__name__ != "NotFoundError":
            return  # anything else surfaces as before, from the index itself
        from urllib.parse import urlparse

        name = urlparse(store_uri).netloc
        env = f" --env {environment}" if environment else ""
        where = f" in Modal environment {environment!r}" if environment else ""
        raise _fail(
            f"Volume {name!r} (from {store_uri}) does not exist yet{where}, and `flora index "
            "--local` writes its receipts there. Create it: `modal volume create "
            f"{name} --version 2{env}` (free), or run `flora index` without --local once (it "
            "creates the Volume).") from None


def _print_summary(s: dict) -> None:
    from rich.markup import escape

    c = s["counts"]
    ds = s["dataset"]
    console.print(f"[bold]{escape(ds['name'])}[/bold] @ {escape(str(ds['revision']))}: "
                  f"{c['episodes']} episodes, {c['indexed']} indexed, {c['source_error']} "
                  f"source errors, {c['failed']} failed ({c['resumed']} resumed)")
    console.print(f"envelope ({s['envelope_policy']}): {c['envelope_ok']} ok, "
                  f"{c['out_of_envelope']} out; oracle-eligible {c['oracle_eligible']}; "
                  f"planned {c['planned']}; video {s['video_s'] / 3600:.3f} h")
    for k, d in s["durations"].items():
        console.print(f"  {escape(k)}: {d['episodes']} episodes, {d['video_s']:.1f} s "
                      f"(min {d['min_s']:.1f}, max {d['max_s']:.1f})")
    for title, key in (("envelope reasons", "envelope_reasons"),
                       ("ineligibility", "ineligibility"), ("source errors", "source_errors"),
                       ("codecs", "codecs")):
        if s[key]:
            console.print(f"  {title}: " + ", ".join(f"{escape(k)}={v}"
                                                     for k, v in s[key].items()))
    for f in s["failed"][:10]:
        console.print(f"  [yellow]failed {escape(f['uid'])}: {escape(str(f['error']))}[/yellow]")


@app.command()
def index(
    config: Annotated[Path, typer.Option("--config", "-c", help="flora.yaml")] = Path("flora.yaml"),
    local: Annotated[bool, typer.Option(
        "--local", help="Run in this process instead of on Modal (free; downloads the "
                         "dataset here).")] = False,
    job: Annotated[str | None, typer.Option("--job", help="Job id (default: project).")] = None,
    store: Annotated[str | None, typer.Option(
        "--store", envvar="FLORA_STORE",
        help="Job store: file:///dir, s3://bucket\\[/prefix], modal-volume://name\\[/prefix] "
             "(default: from output.sink).")] = None,
    limit: Annotated[int | None, typer.Option(help="Index only the first N episodes.")] = None,
    decode: Annotated[bool | None, typer.Option(
        "--decode/--no-decode",
        help="Decode-check the cameras that can be certified against NVIDIA's reference "
             "(default: on for preset exact).")
    ] = None,
    max_episodes: Annotated[int, typer.Option(
        "--max-episodes-per-unit", help="Episodes per work unit (a source file is never "
                                        "split).")] = 64,
    work: Annotated[Path | None, typer.Option(
        help="--local: where Hub files are downloaded (default: input.stage_to or "
             "~/.cache/flora-deploy/datasets).")] = None,
    force: Annotated[bool, typer.Option(help="Re-index episodes that have receipts.")] = False,
    out: Annotated[Path | None, typer.Option(
        help="Also write the summary here (default: summary.json next to the config, where "
             "`flora plan` reads it).", show_default=False)] = None,
) -> None:
    """Index a dataset: per-episode receipts (video probe, envelope, eligibility).

    Runs on Modal CPU containers (no GPU; 1 core and 2 GiB each, about $0.06 per
    container-hour at list prices), so it costs a few cents for most datasets. --local runs
    it in this process for free.
    """
    from flora_deploy.config.schema import ConfigError, load_config
    from flora_deploy.jobs.indexing import IndexJob, open_dataset, run_index

    try:
        cfg = load_config(config)
    except FileNotFoundError:
        raise _fail(f"{config} not found; run `flora init` first") from None
    except ConfigError as e:
        raise _fail(str(e)) from None
    job_id = job or cfg.project
    try:
        store_uri = store or store_for_sink(cfg.output.sink)
        ijob = IndexJob.from_config(cfg, job_id, decode=decode)
    except ValueError as e:
        raise _fail(str(e)) from None

    if local:
        from flora_deploy.export.results import store_from_uri

        try:
            # the config's Modal Environment, not Modal's default one: a modal-volume:// store
            # is looked up (and written) where `flora deploy` and `flora doctor` look
            st = store_from_uri(store_uri, environment=cfg.modal.environment)
        except (ValueError, ImportError) as e:
            raise _fail(str(e)) from None
        _require_volume(st, store_uri, cfg.modal.environment)
        local_root = None
        if cfg.input.repo is None:
            p = Path(str(cfg.input.path)).expanduser()
            local_root = p if p.is_absolute() else (config.parent / p)
            if not local_root.is_dir():
                raise _fail(f"input.path {local_root} is not a directory")
        stage = cfg.input.stage_to
        work_dir = work or (Path(stage).expanduser() if stage and "://" not in stage else
                            Path("~/.cache/flora-deploy/datasets").expanduser())
        files = open_dataset(ijob.dataset, work=work_dir, local_root=local_root, staged=True)
        try:
            summary = run_index(ijob, st, files, limit=limit, max_episodes=max_episodes,
                                force=force, log=lambda m: console.print(m, highlight=False,
                                                                         markup=False))
        except ValueError as e:  # SourceError (bad layout / metadata) is a ValueError
            raise _fail(str(e), 1) from None
    else:
        if cfg.input.repo is None:
            raise _fail("a local input.path cannot be read on Modal; use --local (or a Hub "
                        "dataset)")
        from flora_deploy.modal_app.index_app import volume_target

        try:
            target = volume_target(store_uri)
        except ValueError as e:
            raise _fail(str(e)) from None
        if target is not None:
            os.environ["FLORA_INDEX_VOLUME"] = target[0]
        if target is None and not store_uri.startswith("s3://"):
            raise _fail(f"store {store_uri!r}: on Modal use modal-volume:// or s3://")
        import importlib

        import modal

        from flora_deploy.modal_app import index_app

        index_app = importlib.reload(index_app)  # pick up FLORA_INDEX_VOLUME
        with modal.enable_output(), index_app.app.run(environment_name=cfg.modal.environment):
            summary = index_app.run_index_remote.remote(
                ijob.to_dict(), store_uri, cfg.input.stage_to, limit, max_episodes, force)

    _print_summary(summary)
    default_out = default_index_summary(config)
    out = out or default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(f"summary: {out} (store: {store_uri}); next: flora plan"
                  + ("" if out == default_out else f" --index {out}"))
