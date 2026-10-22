"""`flora export`: a composed job's labels written into a copy of its dataset (docs/export.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console
from flora_deploy.cli.results import JobStoreOpt
from flora_deploy.cli.run import ConfigOpt, EnvOpt


def _fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]")
    return typer.Exit(code=code)


@app.command("export")
def export_cmd(
    out: Annotated[Path, typer.Option("--out", help="New directory for the labeled copy.")],
    job: Annotated[str | None, typer.Option(
        "--job", help="Job id (default: the config's project; its fake-models run when there is no "
                "real one).", show_default=False)] = None,
    store: JobStoreOpt = None,
    config: ConfigOpt = Path("flora.yaml"),
    env: EnvOpt = None,
    to: Annotated[str, typer.Option("--to", help="Target format (lerobot).")] = "lerobot",
    stages: Annotated[list[str] | None, typer.Option(
        "--stages", help="Stages to export (repeat or comma-separate): segment, caption, "
                         "reward.terminal, reward.dense. Default: every composed one.")] = None,
    source: Annotated[Path | None, typer.Option(
        "--source", help="Local copy of the source dataset (default: the job's dataset; a Hub "
                         "dataset is fetched at its pinned revision, meta/ and data/ only).")
    ] = None,
    include_videos: Annotated[bool, typer.Option(
        "--include-videos", help="Copy the videos too (default: reference the source).")] = False,
    allow_uncertified: Annotated[bool, typer.Option(
        "--allow-uncertified", help="Export uncertified records (recorded in the metadata).")
    ] = False,
    allow_fake: Annotated[bool, typer.Option(
        "--allow-fake", help="Export fake-model records (recorded in the metadata).")] = False,
    push_to_hub: Annotated[str | None, typer.Option(
        "--push-to-hub", help="Upload the export to this dataset repo (user/repo).")] = None,
    public: Annotated[bool, typer.Option(
        "--public", help="With --push-to-hub: make the repo public (default private).")] = False,
    config_hash: Annotated[str | None, typer.Option(
        "--config-hash", help="Config hash (default: the job manifest's).")] = None,
    keep_local_paths: Annotated[bool, typer.Option(
        "--keep-local-paths", help="Record local source / output paths in the export (local "
                                   "use only; not with --push-to-hub).")] = False,
) -> None:
    """Write a job's labels into a copy of its LeRobot dataset (the source is never modified)."""
    from flora_deploy import sdk
    from flora_deploy.export.lerobot import ExportError
    from flora_deploy.export.results import ResultsNotFound

    if to != "lerobot":
        raise _fail(f"unknown --to {to!r}; only lerobot is supported")
    chosen = None if not stages else [s.strip() for x in stages for s in x.split(",")
                                      if s.strip()]
    from flora_deploy.cli._jobref import ResolveError, print_notes, resolve
    from flora_deploy.config.schema import ConfigError

    try:
        ref = resolve(job, store, config, env)
    except (ResolveError, ConfigError) as e:
        raise _fail(str(e)) from None
    print_notes(ref)
    st, job = ref.store, ref.job_id
    try:
        sdk.check_results_namespace(st, job, config_hash)
        report = sdk.export_job(st, job, out, config_hash=config_hash, source=source,
                                stages=chosen, allow_uncertified=allow_uncertified,
                                allow_fake=allow_fake, include_videos=include_videos,
                                push_to_hub=push_to_hub, private=not public,
                                keep_local_paths=keep_local_paths)
    except (ExportError, ResultsNotFound, ValueError, sdk.FloraError) as e:
        raise _fail(str(e)) from None
    except FileNotFoundError as e:  # a bare store path: the job is not in this store
        raise _fail(f"no job {job!r} in {ref.store_uri} (missing {e}); check --job / --store, "
                    "or run it first") from None
    console.print(f"Wrote {out} (LeRobot {report.codebase_version}): "
                  f"{report.labeled}/{report.episodes} episodes labeled, stages "
                  f"{', '.join(report.stages)}")
    r = report.records
    console.print(f"records: {r['certified']} certified, {r['uncertified']} uncertified, "
                  f"{r['fake']} fake")
    if report.unlabeled:
        console.print(f"[yellow]{len(report.unlabeled)} episodes without a label (null "
                      "fields); see flora_export_report.json[/yellow]")
    if not report.videos_included:
        console.print("videos not copied: see meta/info.json flora.videos.note")
    if report.pushed:
        console.print(f"pushed to {report.pushed} ({'public' if public else 'private'})")
