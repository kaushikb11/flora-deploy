"""`flora certify`: Gate certificates for a non-``exact`` stack (``validate/certify.py``).

Compares the candidate run's records with the ``exact`` oracle's, against the ``exact`` noise
run, and writes one content-addressed certificate per stage into ``config/certificates/`` (or
``--out-dir``). The certificate is bound to the records' own provenance; the candidate's key
must be the stack this checkout would run. Metrics only: no captions, prompts or other record
text are written. Runs on CPU; contacts nothing.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console


def _fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]")
    return typer.Exit(code=code)


def _pairs(items: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or ():
        k, sep, v = item.partition("=")
        if not sep or not k or not v:
            raise _fail(f"{what} {item!r}: use NAME=VALUE")
        out[k] = v
    return out


def _similarity(spec: str | None):
    if not spec:
        return None
    mod, sep, fn = spec.partition(":")
    if not sep:
        raise _fail(f"--similarity {spec!r}: use module:function")
    try:
        return getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        raise _fail(f"--similarity {spec!r}: {e}") from None


@app.command()
def certify(
    exact: Annotated[Path, typer.Option(help="exact (oracle) records, JSONL.")],
    balanced: Annotated[Path, typer.Option(help="Candidate records on the same episodes, JSONL.")],
    config: Annotated[Path, typer.Option(
        "--config", "-c", help="The candidate run's flora.yaml (preset, pools, modes).")
    ] = Path("flora.yaml"),
    noise: Annotated[Path | None, typer.Option(
        help="The oracle run repeated on a subset of its episodes, JSONL: the noise floor.")
    ] = None,
    l0: Annotated[list[Path] | None, typer.Option(
        help="balanced_lab result JSON (L0 / Gate A), repeatable (one per model kind).")
    ] = None,
    stage: Annotated[list[str] | None, typer.Option(
        help="Stage(s) to certify (default: every certifiable stage in the candidate records).")
    ] = None,
    gpu: Annotated[str | None, typer.Option(
        help="GPU type certified (default: the config's pool GPU for the stage).")] = None,
    tolerance: Annotated[float, typer.Option(
        help="Allowed multiple of the noise floor, in [1, 2].")] = 1.5,
    min_samples: Annotated[int, typer.Option(
        help="Episodes needed per (stage, embodiment pool); at least 10.")] = 10,
    min_noise_samples: Annotated[int, typer.Option(
        help="Noise-run episodes per stage; at least 5.")] = 5,
    slack: Annotated[list[str] | None, typer.Option(
        help="Absolute slack METRIC=VALUE added to the threshold (repeatable; default 0; at "
             "most the metric's noise floor).")] = None,
    derived_content: Annotated[list[str] | None, typer.Option(
        help="Derived-weight content digest NAME=SHA256 (DerivedPin.content_sha256) of the "
             "weights the candidate ran with (repeatable; required for segment/caption).")
    ] = None,
    similarity: Annotated[str | None, typer.Option(
        help="Optional caption text-similarity hook module:function -> [0,1] (off by default).")
    ] = None,
    out_dir: Annotated[Path | None, typer.Option(
        help="Where to write certificates (default: the package registry; a failed one is "
             "written only with --out-dir).")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the certificates as JSON.")
                       ] = False,
) -> None:
    """Certify a balanced stack against exact, within NVIDIA's own run-to-run noise."""
    from flora_deploy.config.hashing import kind_modes
    from flora_deploy.config.presets import get_preset
    from flora_deploy.config.schema import ConfigError, load_config
    from flora_deploy.validate import certify as C
    from flora_deploy.validate.certkey import CertKeyError, key_document

    try:
        cfg = load_config(config, allow_uncertified=True)
    except FileNotFoundError:
        raise _fail(f"{config} not found") from None
    except ConfigError as e:
        raise _fail(str(e)) from None
    if cfg.preset == "exact":
        raise _fail("exact is the reference by construction; certify a different preset")
    spec = get_preset(cfg.preset)
    try:
        ex, bal = C.read_records(exact), C.read_records(balanced)
        nz = C.read_records(noise) if noise else None
        labs = [json.loads(p.read_text(encoding="utf-8")) for p in l0 or ()]
        slacks = {k: float(v) for k, v in _pairs(slack, "--slack").items()}
    except (OSError, ValueError) as e:
        raise _fail(f"bad input: {e}") from None
    lab_by_kind = {lab.get("kind"): lab for lab in labs}
    stages = stage or [s for s in C.STAGE_METRICS if any(r.stage == s for r in bal)]
    sim = _similarity(similarity)
    try:
        th = C.Thresholds(tolerance=tolerance, min_samples=min_samples,
                          min_noise_samples=min_noise_samples, slack=slacks)
    except C.CertifyError as e:
        raise _fail(str(e)) from None
    contents = _pairs(derived_content, "--derived-content")
    pool_map = cfg.pools()
    certs, worst = [], 0
    for st in stages:
        kind = spec.stage_kinds.get(st)
        if kind is None or kind not in pool_map:
            raise _fail(f"stage {st!r} is not run by this config")
        g = gpu or pool_map[kind].gpu
        try:
            doc = key_document(cfg.preset, kind, g, modes=kind_modes(cfg, kind),
                               derived_content=contents)
        except CertKeyError as e:
            raise _fail(f"{st}: {e} (pass --derived-content NAME=SHA256)") from None
        try:
            cert = C.certify_stage(
                st, exact=ex, balanced=bal, noise=nz, engine_spec=doc, gpu=g,
                preset=cfg.preset, kind=kind, l0=lab_by_kind.get(C.L0_STAGES.get(st)),
                thresholds=th, similarity=sim, similarity_name=similarity)
        except C.CertifyRefused as e:
            raise _fail(f"{st}: certified: false ({e.reason}): {e}", 1) from None
        except C.CertifyError as e:
            raise _fail(f"{st}: {e}") from None
        if cert["covered_pools"] != [cfg.pipeline.embodiment]:
            raise _fail(f"{st}: the records are for embodiment {cert['covered_pools']}, the "
                        f"config for {cfg.pipeline.embodiment!r}")
        path = None
        if cert["verdict"] != "fail" or out_dir is not None:
            path = C.write_certificate(cert, out_dir)
        worst = max(worst, ("certified", "provisional", "fail").index(cert["verdict"]))
        certs.append((cert, path))
    if as_json:
        typer.echo(json.dumps([c for c, _ in certs], indent=1, sort_keys=True))
    else:
        colour = {"certified": "green", "provisional": "yellow", "fail": "red"}
        for cert, path in certs:
            v = cert["verdict"]
            console.print(f"{cert['stage']} on {cert['gpu']} for {cert['covered_pools']}: "
                          f"[{colour[v]}]{v}[/{colour[v]}] {cert['id']}"
                          + (f" ({', '.join(cert['reasons'])})" if cert["reasons"] else ""))
            for ch in cert["checks"]:
                if ch["pass"] is False:
                    console.print(f"  {ch['class']} {ch['metric']}: {ch['value']} > "
                                  f"{ch['threshold']}")
            console.print(f"  {'written to ' + str(path) if path else 'not written'}")
        if any(c["verdict"] == "provisional" for c, _ in certs):
            console.print("Provisional certificates do not unlock a preset: runs still need "
                          "allow_uncertified.")
    raise typer.Exit(code=1 if worst == 2 else 0)
