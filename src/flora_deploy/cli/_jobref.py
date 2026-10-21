"""Which job, in which store: the defaults every job-scoped command shares.

``flora status | logs | cancel | resume | report | results`` take the job id as an optional
argument and the store as an optional ``--store`` (or ``FLORA_STORE``). Without them:

* **job id**: the config's ``project`` (``--config``, default ``./flora.yaml``).
* **store**: the config's ``output.sink`` (``volume://`` means ``modal-volume://``), and the local
  store ``flora run --local`` writes (``<config dir>/flora-local-store``) when that directory
  exists. With ``--store`` only that store is searched.

**Real vs fake-models runs** (review 8b M5: they never share a job id; a fake one is
``<job>.fake``). With the job id *defaulted*, the real run is preferred; when there is none but a
fake-models run exists, that one is used and one line on stderr says the results come from
stand-in models. When both exist the real one is used and the line says the fake one exists. An
*explicit* job id is used exactly as given, without a note (a real id with only a fake run is
refused with a pointer to ``<job>.fake``, as before).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCAL_STORE_DIRNAME = "flora-local-store"  # local.runner's default store, next to flora.yaml


def local_store_dir(config: Path) -> Path:
    """The store ``flora run --local`` uses by default for ``config``."""
    return config.parent / LOCAL_STORE_DIRNAME


def is_local_uri(uri: str) -> bool:
    return uri.startswith("file://") or "://" not in uri


@dataclass
class JobRef:
    """A resolved (job id, store). ``notes``: lines to show the user (stderr)."""

    job_id: str
    store_uri: str
    store: Any
    cfg: Any = None
    environment: str | None = None
    fake: bool = False
    notes: list[str] = field(default_factory=list)


class ResolveError(Exception):
    """Not enough to find the job (exit 2)."""


def _open(uri: str, environment: str | None):
    """A local path / ``file://`` opens directly (it is on this machine); anything else through
    the SDK backend (the Modal client, or the tests' fake)."""
    from flora_deploy import sdk
    from flora_deploy.export.results import store_from_uri

    if is_local_uri(uri):
        return store_from_uri(uri)
    return sdk.get_backend().store(uri, environment=environment)


def load_cfg(config: Path):
    """The config at ``config``, or ``None`` when the file does not exist."""
    if not config.exists():
        return None
    from flora_deploy.config.schema import load_config

    return load_config(config, allow_uncertified=True)


def resolve(job: str | None, store: str | None, config: Path, env: str | None = None,
            *, cfg: Any = None) -> JobRef:
    """The job and store a job-scoped command acts on (module docstring). Raises
    :class:`ResolveError` (and ``ConfigError`` for a broken config)."""
    from flora_deploy import sdk
    from flora_deploy.jobs.store import fake_job_id, is_fake_job_id

    if cfg is None:
        try:
            cfg = load_cfg(config)
        except Exception:  # noqa: BLE001 - ConfigError; irrelevant when both are explicit
            if job is None or store is None:
                raise
    if job is None and cfg is None:
        raise ResolveError(f"no job id: {config} not found. Pass the job id, or run from the "
                           "directory with your flora.yaml (or pass --config)")
    if store is None and cfg is None:
        raise ResolveError(f"{config} not found: pass --store (or FLORA_STORE) or --config")
    environment = env or (cfg.modal.environment if cfg is not None else None)
    job_id = job or cfg.project

    # Candidate stores, in order. Real runs only ever live in the sink (or --store); a local
    # store only holds `flora run --local` (fake-models) runs.
    cands: list[str] = []
    if store is not None:
        cands.append(store)
    else:
        cands.append(sdk.store_uri_for(cfg))
        local = local_store_dir(config)
        if local.is_dir() and str(local) not in cands:
            cands.append(str(local))

    opened: dict[str, Any] = {}
    unreachable: dict[str, str] = {}

    def has_run(uri: str, jid: str) -> bool:
        try:
            if uri not in opened:
                opened[uri] = _open(uri, environment)
            return opened[uri].exists(sdk.run_pointer_path(jid))
        except (ValueError, ImportError):
            raise
        except Exception as e:  # noqa: BLE001 - e.g. no Modal credentials: skip that store
            unreachable.setdefault(uri, f"{uri} ({type(e).__name__}: {e})")
            return False

    def ref(uri: str, jid: str, notes: list[str]) -> JobRef:
        if uri not in opened:
            try:
                opened[uri] = _open(uri, environment)
            except (ValueError, ImportError) as e:
                raise ResolveError(str(e)) from None
        return JobRef(jid, uri, opened[uri], cfg, environment, is_fake_job_id(jid), notes)

    try:
        if job is not None or is_fake_job_id(job_id):
            # explicit id: exactly that job, in the first store that has it
            found = next((u for u in cands if has_run(u, job_id)), cands[0])
            return ref(found, job_id, [])  # asked for by name: no note
        fake_id = fake_job_id(job_id)
        real = next((u for u in cands if has_run(u, job_id)), None)
        fake = next((u for u in cands if has_run(u, fake_id)), None)
    except (ValueError, ImportError) as e:
        raise ResolveError(str(e)) from None
    if real is not None:
        notes = []
        if fake is not None:
            notes.append(f"note: a fake-models run of this job also exists ({fake_id}"
                         + ("" if fake == real else f" in {fake}") + f"); pass {fake_id} to "
                         "see it")
        return ref(real, job_id, notes)
    if fake is not None:
        return ref(fake, fake_id, [stand_in_line(fake_id, real_id=job_id)])
    r = ref(cands[0], job_id, [])
    if unreachable:
        r.notes.append("could not read " + "; ".join(unreachable.values()))
    return r


def stand_in_line(fake_id: str, *, real_id: str | None = None) -> str:
    lead = f"no real run of {real_id}; " if real_id else ""
    return (f"{lead}showing {fake_id}, a fake-models run: these labels come from stand-in "
            "models, not Flora")


def print_notes(r: JobRef) -> None:
    """The resolution's notes, on stderr (stdout stays clean for --json / --format jsonl)."""
    import typer

    for n in r.notes:
        typer.secho(n, err=True, fg="yellow")
