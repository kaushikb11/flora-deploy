"""`flora init`: write a commented ``flora.yaml`` from a template shipped in the package.

The templates live in ``flora_deploy/config/templates/`` and are read with
``importlib.resources``, so ``flora init`` works from an installed wheel."""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console

TEMPLATES = {
    "exact_lerobot": "exact_lerobot.yaml",
    "exact_mp4_dir": "exact_mp4_dir.yaml",
    "rewards_only": "rewards_only.yaml",
    "local_demo": "local_demo.yaml",
}


def templates_dir() -> Traversable:
    """The packaged templates (``flora_deploy/config/templates``)."""
    return resources.files("flora_deploy.config").joinpath("templates")


def template_text(name: str) -> str:
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; choose one of {', '.join(TEMPLATES)}")
    path = templates_dir().joinpath(TEMPLATES[name])
    if not path.is_file():
        raise FileNotFoundError(f"template {TEMPLATES[name]} is missing from the installed "
                                "package (reinstall flora-deploy)")
    return path.read_text(encoding="utf-8")


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the config.")] = Path("flora.yaml"),
    template: str = typer.Option(
        "exact_lerobot", "--template", "-t", help=f"One of: {', '.join(TEMPLATES)}."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Write a commented flora.yaml to start from."""
    from rich.markup import escape

    from flora_deploy.config.schema import ConfigError, loads_config

    try:
        text = template_text(template)
    except (KeyError, FileNotFoundError) as e:
        console.print(f"[red]{escape(str(e.args[0]))}[/red]")
        raise typer.Exit(code=2) from None
    if path.exists() and not force:
        console.print(f"[red]{escape(str(path))} exists; pass --force to overwrite it[/red]")
        raise typer.Exit(code=1)
    try:
        loads_config(text, source=f"template {template}")
    except ConfigError as e:  # a broken template is our bug; never write it
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(code=1) from None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    console.print(f"Wrote {escape(str(path))} (template {template}). Edit it, then run "
                  "`flora prepare`.")
