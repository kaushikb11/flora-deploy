"""`flora init`: write a commented ``flora.yaml`` from a template in ``examples/``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console

TEMPLATES = {
    "exact_lerobot": "exact_lerobot.yaml",
    "exact_mp4_dir": "exact_mp4_dir.yaml",
    "rewards_only": "rewards_only.yaml",
}


def examples_dir() -> Path:
    """``examples/`` of the source checkout (``src/flora_deploy/cli/init.py`` -> repo root)."""
    return Path(__file__).resolve().parents[3] / "examples"


def template_text(name: str) -> str:
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; choose one of {', '.join(TEMPLATES)}")
    path = examples_dir() / TEMPLATES[name]
    if not path.is_file():
        raise FileNotFoundError(f"template {path} not found (is this a source checkout?)")
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
