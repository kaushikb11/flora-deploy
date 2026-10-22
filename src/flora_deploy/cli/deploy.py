"""`flora deploy`: build the config's Modal app (``modal_app.factory``) and deploy it.

``--dry-run`` prints the plan without contacting Modal; every run also prints each stage's
certificate status on its GPU in plain words ("certified (human, L40S)", "not certified",
"certified for human, not for yam", or reference for ``exact``) and why (``cli._certs``);
``--emit out.py`` writes the equivalent
standalone script for review instead of deploying. A real deploy records the deployment (app
name, stage hashes, image fingerprints, worker classes) in the job store at
``flora/deployments/<app>.json``.

``--fake-models`` (H0 rehearsal): the same app with G2's fake engines in CPU worker containers
(``modal_app.factory`` "Fake models"): a distinct config hash and app name, no weights, records
marked ``certificate: fake-models``. Run jobs on it with ``flora run --fake-models``.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from flora_deploy.cli._app import app, console


def _fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]")
    return typer.Exit(code=code)


def _receipts(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or ():
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise _fail(f"--receipt {item!r}: use NAME=HASH (e.g. captioner_bf16=<sha256>)")
        out[name] = value
    return out


def _deploy(modal_app: Any, plan: Any) -> str | None:
    """``App.deploy()``; returns the app id. (A seam for tests.)"""
    import modal

    with modal.enable_output():
        modal_app.deploy(name=plan.app_name, environment_name=plan.environment)
    return getattr(modal_app, "app_id", None)


def _record(plan: Any, record: dict[str, Any]) -> str:
    """Write the deployment record whole into the job store; returns where."""
    from flora_deploy.modal_app.factory import deployment_path

    rel = deployment_path(plan.app_name)
    data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    st = plan.store
    if st.kind == "volume":
        import modal

        vol = modal.Volume.from_name(st.volume, version=2, create_if_missing=True,
                                     environment_name=plan.environment)
        remote = "/" + "/".join(p for p in (st.root, rel) if p)
        with vol.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), remote)
        return f"modal-volume://{st.volume}{remote}"
    from flora_deploy.jobs.stores import BucketStore

    BucketStore(st.bucket, prefix=st.root,
                endpoint_url=os.environ.get("AWS_ENDPOINT_URL")).put_bytes(rel, data)
    return f"{st.uri.rstrip('/')}/{rel}"


def _certificates(cfg: Any, plan: Any, derived_content: dict[str, str] | None
                  ) -> dict[str, dict[str, Any]]:
    """Gate certificate status of each deployed shard kind on its GPU (``cli._certs``)."""
    pools: dict[str, str | None] = {}
    for o in plan.objects:
        for k in o.shard_kinds:
            pools.setdefault(k, o.gpu)
    if getattr(plan, "fake_models", False):
        return {k: {"gpu": g, "status": "none", "certificate": None,
                    "why": "fake models (CPU rehearsal): records are marked "
                           "certificate: fake-models"} for k, g in pools.items()}
    from flora_deploy.cli._certs import certificate_report

    try:
        return certificate_report(cfg, pools, derived_content=derived_content)
    except Exception as e:  # noqa: BLE001 - a report; never block the deploy on it
        return {k: {"gpu": g, "status": "unknown", "certificate": None,
                    "why": f"could not check: {e!r}"} for k, g in pools.items()}


_CERT_STYLE = {"reference": "green", "certified": "green", "provisional": "yellow",
               "none": "yellow", "other_embodiment": "yellow", "other_build": "yellow",
               "fail": "red", "unknown": "red"}


def _print_certificates(certs: dict[str, dict[str, Any]]) -> None:
    from rich.markup import escape

    for kind, c in certs.items():
        style = _CERT_STYLE.get(c.get("state", c["status"]), "yellow")
        label = c.get("label") or c["status"]
        console.print(f"  certificate {kind} on {c['gpu'] or 'cpu'}: "
                      f"[{style}]{escape(label)}[/{style}] - {escape(c['why'])}",
                      soft_wrap=True)


def _print_plan(plan: Any) -> None:
    fake = ("  [yellow]FAKE MODELS (CPU rehearsal)[/yellow]"
            if getattr(plan, "fake_models", False) else "")
    console.print(f"[bold]{plan.app_name}[/bold]  (environment {plan.environment}, "
                  f"preset {plan.preset}, store {plan.store.uri}){fake}")
    t = Table("object", "kind", "gpu", "cpu", "memory MiB", "containers", "inputs", "image")
    for o in plan.objects:
        mem = "/".join(str(m) for m in o.memory) if isinstance(o.memory, tuple) else str(o.memory)
        containers = ("-" if o.max_containers is None
                      else f"{o.min_containers or 0}..{o.max_containers}")
        t.add_row(o.name, o.kind if not o.schedule else f"cron {o.schedule}", o.gpu or "-",
                  f"{o.cpu:g}", mem, containers, str(o.max_inputs or 1), o.image)
    console.print(t)
    for kind, h in plan.stage_hashes.items():
        console.print(f"  stage hash {kind}: {h}")
    for what, why in plan.skipped.items():
        console.print(f"  [yellow]not deployed[/yellow] {what}: {why}")


@app.command()
def deploy(
    config: Annotated[Path, typer.Option("--config", "-c", help="flora.yaml")] = Path("flora.yaml"),
    store: Annotated[str | None, typer.Option(
        "--store", envvar="FLORA_STORE",
        help="Job store seen by containers: modal-volume://name\\[/path] or s3://bucket"
             "\\[/prefix] (default: output.sink).")] = None,
    bucket_secret: Annotated[str, typer.Option(
        help="Modal Secret holding the bucket credentials.")] = "flora-bucket",
    receipt: Annotated[list[str] | None, typer.Option(
        help="Derived-weight receipt NAME=HASH (repeatable): overrides weights.derived, which "
             "`flora prepare` records in flora.yaml.")
    ] = None,
    allow_uncertified: Annotated[bool, typer.Option(
        "--allow-uncertified", help="Deploy even if a stage has no accuracy certificate for "
                                    "this config (its records say certified: false).")] = False,
    min_containers: Annotated[int, typer.Option(help="Warm containers per worker class.")] = 0,
    reconcile_cron: Annotated[str | None, typer.Option(
        help="Schedule the reconciler: 'on' (every 10 min) or a cron string.")] = None,
    determinism: Annotated[str | None, typer.Option(
        help="Override pipeline.determinism (exact segcap worker): nvidia | deterministic. "
             "Hashed: it changes the config hash and the app.")] = None,
    emit: Annotated[Path | None, typer.Option(
        "--emit", help="Write the equivalent standalone script here instead of deploying.")
    ] = None,
    dry_run: Annotated[bool, typer.Option(
        "--dry-run", help="Print the plan; do not contact Modal.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the plan as JSON.")] = False,
    fake_models: Annotated[bool, typer.Option(
        "--fake-models", help="Rehearsal app: fake engines on CPU workers (no GPU, no weights; "
                              "hashed apart from real outputs).")] = False,
) -> None:
    """Deploy the config's Modal app (one app per config; workers scale from zero)."""
    from flora_deploy.config.hashing import MissingReceiptError
    from flora_deploy.config.receipts import load_certified
    from flora_deploy.config.receipts import resolve as resolve_receipts
    from flora_deploy.config.schema import ConfigError, unsupported_sink_message
    from flora_deploy.modal_app.factory import DeployError, deployment_record, materialize, plan_app

    if store is not None and (why := unsupported_sink_message(store)):
        raise _fail(f"store {store!r}: {why}")
    overrides = _receipts(receipt)
    try:
        cfg, content = load_certified(config, allow_uncertified=allow_uncertified,
                                      overrides=overrides)
    except (ConfigError, OSError) as e:
        raise _fail(str(e)) from None
    try:
        derived = resolve_receipts(cfg, overrides, fake_models=fake_models)
        plan = plan_app(cfg, store=store, bucket_secret=bucket_secret,
                        derived_receipts=derived, derived_content=content,
                        allow_uncertified=allow_uncertified,
                        min_containers=min_containers, reconcile_cron=reconcile_cron,
                        determinism=determinism, fake_models=fake_models)
    except (DeployError, MissingReceiptError, ValueError) as e:
        raise _fail(str(e)) from None

    cert_cfg = cfg
    if determinism is not None and determinism != cfg.pipeline.determinism:
        from flora_deploy.config.schema import with_pipeline

        cert_cfg = with_pipeline(cfg, determinism=determinism)  # plan_app accepted it
    certs = _certificates(cert_cfg, plan, content)
    if as_json:
        print(json.dumps({**plan.to_json(), "certificates": certs}, indent=2, sort_keys=True))
    else:
        _print_plan(plan)
        _print_certificates(certs)
    if emit is not None:
        from flora_deploy.modal_app.emit import write_script

        write_script(plan, emit)
        console.print(f"wrote {emit} (review it; `modal deploy -e {plan.environment} {emit}`)")
        return
    if dry_run:
        console.print("dry run: nothing deployed")
        return

    try:
        app_id = _deploy(materialize(plan), plan)
    except DeployError as e:
        raise _fail(str(e)) from None
    record = deployment_record(plan, app_id=app_id,
                               deployed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        where = _record(plan, record)
    except Exception as e:  # the app is deployed; say so, but fail loudly
        console.print(f"[green]deployed {plan.app_name}[/green]")
        raise _fail(f"could not record the deployment in {plan.store.uri}: {e!r}", 1) from None
    console.print(f"[green]deployed {plan.app_name}[/green] (recorded at {where})")
