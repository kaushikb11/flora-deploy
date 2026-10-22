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

DEFAULT_TEMPLATE = "balanced_lerobot"

# name -> (file, one-line description). `balanced_*` are the user-facing defaults; `exact_*` are
# the reference / certification templates.
TEMPLATE_INFO: dict[str, tuple[str, str]] = {
    "balanced_lerobot": ("balanced_lerobot.yaml",
                         "LeRobot dataset, balanced preset (vLLM; L40S + H100!); the default"),
    "balanced_mp4_dir": ("balanced_mp4_dir.yaml",
                         "folder of MP4s, balanced preset (vLLM; L40S + H100!)"),
    "exact_lerobot": ("exact_lerobot.yaml",
                      "LeRobot dataset, exact preset (NVIDIA's HF code on H200); reference / "
                      "certification"),
    "exact_mp4_dir": ("exact_mp4_dir.yaml",
                      "folder of MP4s, exact preset; reference / certification"),
    "rewards_only": ("rewards_only.yaml",
                     "score policy rollouts with the Rewarder only (terminal + dense)"),
    "local_demo": ("local_demo.yaml", "local demo: fake models, no GPU, no Modal account"),
    "local_demo_lerobot": ("local_demo_lerobot.yaml",
                           "local demo on a sample LeRobot dataset (try `flora export`)"),
}
TEMPLATES: dict[str, str] = {name: f for name, (f, _) in TEMPLATE_INFO.items()}

# name -> the keys to edit after `flora init` (README: "set your dataset and output location")
# and the command to run next.
_LEROBOT = ("input.repo", "input.revision", "pipeline.embodiment", "output.sink")
_MP4 = ("input.path", "input.stage_to", "pipeline.embodiment", "output.sink")
NEXT_STEPS: dict[str, tuple[tuple[str, ...], str]] = {
    "balanced_lerobot": (_LEROBOT, "flora doctor"),
    "exact_lerobot": (_LEROBOT, "flora doctor"),
    "balanced_mp4_dir": (_MP4, "flora doctor"),
    "exact_mp4_dir": (_MP4, "flora doctor"),
    "rewards_only": (_MP4, "flora doctor"),
    "local_demo": ((), "flora run --local --fake-models --sample-data"),
    "local_demo_lerobot": ((), "flora run --local --fake-models --sample-data"),
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
        DEFAULT_TEMPLATE, "--template", "-t", help=f"One of: {', '.join(TEMPLATES)}."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
    list_: bool = typer.Option(False, "--list", help="List the templates and exit."),
) -> None:
    """Write a commented flora.yaml to start from."""
    from rich.markup import escape

    from flora_deploy.config.schema import ConfigError, loads_config

    if list_:
        width = max(map(len, TEMPLATE_INFO))
        for name, (_, desc) in TEMPLATE_INFO.items():
            mark = "*" if name == DEFAULT_TEMPLATE else " "
            console.print(f"{mark} {name:<{width}}  {escape(desc)}", soft_wrap=True)
        return
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
    keys, nxt = NEXT_STEPS.get(template, ((), "flora doctor"))
    console.print(f"Wrote {escape(str(path))} (template {template}).", soft_wrap=True)
    if keys:
        console.print("Edit next: " + ", ".join(keys) + " (and project, the job's name).",
                      soft_wrap=True, highlight=False)
    console.print(f"Then run: {nxt}", soft_wrap=True, highlight=False)
