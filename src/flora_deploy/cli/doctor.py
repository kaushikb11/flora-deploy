"""`flora doctor`: fast preflight checks before `flora prepare` / `flora deploy` / `flora run`.

Every check reports ``pass``, ``warn`` or ``fail`` with a one-line fix:

* **config**: ``flora.yaml`` validates and carries ``license_ack``;
* **modal**: client installed at the pinned version, a token configured and accepted, the
  config's Environment exists;
* **hf**: every pinned ``nvidia/Flora-*`` repo is readable at its pinned revision, with the
  token if there is one and anonymously otherwise (the repos are not gated; a token only raises
  rate limits). Access is tested with ``auth_check``, since ``model_info`` succeeds on a gated
  repo without access (review 8b M1);
* **store**: the output sink is reachable and writable (a tiny probe object, deleted again); the
  ``flora-bucket`` Secret exists for ``s3://`` / ``gs://`` / ``r2://`` sinks, the Volume for
  ``volume://`` sinks;
* **cache**: the ``flora-cache`` Volume the config's workers mount read-only exists (``flora
  deploy`` creates it when missing; a ``--emit`` script deployed by hand does not);
* **weights**: the pinned snapshots (and their prep receipts) are on the ``flora-weights``
  Volume, and the derived-weight receipts are in ``flora.yaml`` (``weights.derived``);
* **certificates**: each stage's certificate status in plain words ("certified (human, L40S)",
  "not certified", "certified for human, not for yam"), and whether ``allow_uncertified`` is
  still needed;
* **budget**: ``modal.budget_usd`` and the Environment-budget backstop to set by hand;
* **input**: the dataset revision resolves, or the local files exist.

**Zero Modal spend.** Only control-plane and metadata calls are made (object lookups, a Volume
listing, Hub ``*_info`` calls, a probe write); never ``modal run`` and never a container.

Everything that touches the outside world goes through :class:`Probes`, so tests swap in a fake.
Exit code 0 when nothing failed, else 2 (a refusal, as in the rest of the CLI).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import urlparse

import typer

from flora_deploy.cli._app import app, console

Status = Literal["pass", "warn", "fail"]
BUCKET_SECRET = "flora-bucket"
PROBE_PREFIX = "flora/.doctor"

# Probe outcomes. ``hf_repo``: ok | no_token | unauthorized | gated | revision_missing |
# repo_missing | error:<detail>. ``modal_*``: ok | missing | unauthenticated | error:<detail>.
# ``store_write``: ok | missing | skip:<why> | error:<detail>.


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    message: str
    fix: str = ""


class Probes(Protocol):
    """Everything ``flora doctor`` asks the outside world. All calls are metadata-only."""

    def modal_version(self) -> str | None: ...
    def modal_token(self) -> bool: ...
    def modal_environment(self, name: str) -> str: ...
    def modal_secret(self, name: str, environment: str) -> str: ...
    def modal_volume(self, name: str, environment: str) -> str: ...
    def volume_listdir(self, name: str, environment: str, path: str) -> list[str] | None: ...
    def hf_token(self) -> str | None: ...
    def hf_repo(self, repo: str, revision: str, token: str | None, repo_type: str) -> str: ...
    def store_write(self, sink: str, environment: str) -> str: ...
    def local_files(self, path: str, pattern: str) -> int | None: ...
    def certificate_status(self, cfg: Any) -> dict[str, str] | None: ...


def _err(e: BaseException) -> str:
    return f"error:{type(e).__name__}: {e}".splitlines()[0][:300]


class RealProbes:
    """The real probes: Modal control plane, Hugging Face metadata, the sink itself."""

    def modal_version(self) -> str | None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("modal")
        except PackageNotFoundError:
            return None

    def modal_token(self) -> bool:
        from modal.config import config

        return bool(config.get("token_id") and config.get("token_secret"))

    def _lookup(self, obj: Any) -> str:
        try:
            obj.hydrate()
        except Exception as e:  # modal.exception.* (no import needed to classify them)
            name = type(e).__name__
            if name == "NotFoundError":
                return "missing"
            if name in ("AuthError", "PermissionDeniedError"):
                return "unauthenticated"
            return _err(e)
        return "ok"

    def modal_environment(self, name: str) -> str:
        import modal

        return self._lookup(modal.Environment.from_name(name))

    def modal_secret(self, name: str, environment: str) -> str:
        import modal

        return self._lookup(modal.Secret.from_name(name, environment_name=environment))

    def modal_volume(self, name: str, environment: str) -> str:
        import modal

        return self._lookup(modal.Volume.from_name(name, environment_name=environment))

    def volume_listdir(self, name: str, environment: str, path: str) -> list[str] | None:
        import modal

        vol = modal.Volume.from_name(name, environment_name=environment)
        try:
            entries = vol.listdir(path)
        except Exception as e:
            if isinstance(e, FileNotFoundError) or type(e).__name__ == "NotFoundError":
                return None
            raise
        return sorted(e.path.rstrip("/").rsplit("/", 1)[-1] for e in entries)

    def hf_token(self) -> str | None:
        from huggingface_hub import get_token

        return get_token()

    def hf_repo(self, repo: str, revision: str, token: str | None, repo_type: str) -> str:
        from huggingface_hub import HfApi
        from huggingface_hub import errors as hfe

        api = HfApi()
        info = api.dataset_info if repo_type == "dataset" else api.model_info
        try:
            # ``*_info`` checks the revision exists (its metadata is public even on a gated
            # repo); ``auth_check`` checks this token (or anonymous) can read the files.
            info(repo, revision=revision, token=token or False)
            api.auth_check(repo, repo_type=repo_type, token=token or False)
        except hfe.GatedRepoError:
            return "gated" if token else "no_token"
        except hfe.RevisionNotFoundError:
            return "revision_missing"
        except hfe.RepositoryNotFoundError:
            return "repo_missing" if token else "no_token"
        except hfe.HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403):
                return "unauthorized"
            return _err(e)
        except Exception as e:  # network
            return _err(e)
        return "ok"

    def store_write(self, sink: str, environment: str) -> str:
        u = urlparse(sink)
        key = f"{PROBE_PREFIX}/probe-{uuid.uuid4().hex}.txt"
        data = b"flora doctor probe; safe to delete\n"
        try:
            if u.scheme == "file":
                return _probe_local(Path(u.path), key, data)
            if u.scheme == "volume":
                return self._probe_volume(u.netloc, u.path, environment, key, data)
            if u.scheme in ("s3", "r2", "gs"):
                return _probe_bucket(u.scheme, u.netloc, u.path.strip("/"), key, data)
        except Exception as e:
            return _err(e)
        return f"error:unknown sink scheme {u.scheme!r}"

    def _probe_volume(self, name: str, sub: str, environment: str, key: str, data: bytes) -> str:
        import modal

        from flora_deploy.sdk import RemoteVolumeStore

        vol = modal.Volume.from_name(name, environment_name=environment)
        status = self._lookup(vol)
        if status != "ok":
            return status
        st = RemoteVolumeStore(vol, sub)
        st.put_bytes(key, data)
        try:
            if st.get_bytes(key) != data:
                return "error:probe read back different bytes"
        finally:
            vol.remove_file(st._abs(key))
        return "ok"

    def local_files(self, path: str, pattern: str) -> int | None:
        p = Path(path).expanduser()
        if not p.is_dir():
            return None
        return sum(1 for _ in p.rglob(pattern))

    def certificate_status(self, cfg: Any) -> dict[str, str] | None:
        """Kind -> its status in plain words (``cli._certs`` ``label``): ``certified (human,
        L40S)``, ``not certified``, ``certified for human, not for yam``, ..."""
        from flora_deploy.cli._certs import certificate_report
        from flora_deploy.config.receipts import derived_content

        rep = certificate_report(cfg, {k: p.gpu for k, p in cfg.pools().items()},
                                 derived_content=derived_content(cfg))
        return {k: r["label"] for k, r in rep.items()}


def _probe_local(root: Path, key: str, data: bytes) -> str:
    if not root.is_dir():
        return "missing"
    from flora_deploy.jobs.stores import LocalStore

    st = LocalStore(root)
    st.put_bytes(key, data)
    try:
        if st.get_bytes(key) != data:
            return "error:probe read back different bytes"
    finally:
        (root / key).unlink(missing_ok=True)
    return "ok"


def _probe_bucket(scheme: str, bucket: str, prefix: str, key: str, data: bytes) -> str:
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        return "skip:boto3 is not installed here (pip install boto3)"
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if scheme == "gs" and not endpoint:
        endpoint = "https://storage.googleapis.com"
    if scheme == "r2" and not endpoint:
        return "skip:set AWS_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com to probe R2"
    client = boto3.client("s3", endpoint_url=endpoint)
    full = f"{prefix}/{key}" if prefix else key
    try:
        client.put_object(Bucket=bucket, Key=full, Body=data)
    except NoCredentialsError:
        return "skip:no bucket credentials on this machine"
    try:
        body = client.get_object(Bucket=bucket, Key=full)["Body"].read()
    finally:
        client.delete_object(Bucket=bucket, Key=full)
    return "ok" if body == data else "error:probe read back different bytes"


# ---------------------------------------------------------------------------------------------
# Checks


def check_config(path: Path) -> tuple[Check, Any]:
    from flora_deploy.config.receipts import load_certified
    from flora_deploy.config.schema import ConfigError

    if not path.is_file():
        return Check("config", "fail", f"{path} not found",
                     "write one with `flora init` (or pass --config)"), None
    try:
        cfg = load_certified(path)[0]
    except ConfigError as e:
        text = str(e)
        first = next((ln.strip(" -") for ln in text.splitlines()[1:]), text)
        if "license_ack" in text:
            fix = "add `license_ack: research-only` (NVIDIA's models are research-only)"
        elif "accuracy certificate" in text or "not certified for every stage" in text:
            fix = ("set `allow_uncertified: true` to run the stages named above without a "
                   "certificate (their records say certified: false), or use preset exact")
        else:
            fix = f"edit {path}; see src/flora_deploy/config/templates/ for every key"
        return Check("config", "fail", f"{path} is invalid: {first}", fix), None
    return Check("config", "pass",
                 f"{path} valid (preset {cfg.preset}, license_ack {cfg.license_ack})"), cfg


def check_modal(p: Probes, environment: str | None) -> list[Check]:
    from flora_deploy.config.pins import RUNTIME

    pin = RUNTIME["modal_client"]
    out: list[Check] = []
    v = p.modal_version()
    if v is None:
        return [Check("modal.client", "fail", "the modal client is not installed",
                      f"pip install modal=={pin}")]
    if v != pin:
        out.append(Check("modal.client", "fail", f"modal {v} installed; flora needs {pin} (the "
                         "image builders refuse another client)", f"pip install modal=={pin}"))
    else:
        out.append(Check("modal.client", "pass", f"modal {v}"))
    if not p.modal_token():
        out.append(Check("modal.auth", "fail", "no Modal token configured",
                         "run `modal setup` (or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)"))
        return out
    if environment is None:
        out.append(Check("modal.auth", "pass", "a Modal token is configured"))
        return out
    r = p.modal_environment(environment)
    if r == "unauthenticated":
        out.append(Check("modal.auth", "fail", "Modal rejected the configured token",
                         "run `modal token new`"))
    elif r == "missing":
        out.append(Check("modal.auth", "pass", "the Modal token is valid"))
        out.append(Check("modal.environment", "fail", f"Modal environment {environment!r} does "
                         "not exist", f"modal environment create {environment} (or set "
                         "modal.environment)"))
    elif r == "ok":
        out.append(Check("modal.auth", "pass", "the Modal token is valid"))
        out.append(Check("modal.environment", "pass", f"environment {environment!r} exists"))
    else:
        out.append(Check("modal.auth", "warn", f"could not reach Modal ({r[6:]})",
                         "check your network and retry"))
    return out


_HF_FIX = {
    "no_token": ("no Hugging Face token", "run `hf auth login` or set HF_TOKEN"),
    "unauthorized": ("the Hugging Face token was rejected", "create a new read token at "
                     "huggingface.co/settings/tokens"),
    "gated": ("gated access not granted", "request access on the repo's Hub page and wait for "
              "approval"),
    "revision_missing": ("pinned revision not found", "the pin is gone upstream; update "
                         "flora-deploy"),
    "repo_missing": ("repo not found (or not visible to this token)", "check the token's account "
                     "has access"),
}


def _hf_result(name: str, what: str, r: str) -> Check:
    if r == "ok":
        return Check(name, "pass", f"{what} readable")
    if r in _HF_FIX:
        msg, fix = _HF_FIX[r]
        return Check(name, "fail", f"{what}: {msg}", fix)
    return Check(name, "warn", f"{what}: could not check ({r[6:]})", "check your network")


def check_hf(p: Probes) -> tuple[list[Check], str | None]:
    from flora_deploy.config.pins import PINS

    token = p.hf_token()
    if token:
        out = [Check("hf.token", "pass", "a Hugging Face token is present")]
    else:
        out = [Check("hf.token", "warn", "no Hugging Face token: anonymous access works "
                     "(nvidia/Flora-* is not gated); a token raises rate limits",
                     "run `hf auth login` or set HF_TOKEN")]
    for pin in PINS:
        if not pin.repo.startswith("nvidia/Flora-"):
            continue
        r = p.hf_repo(pin.repo, pin.revision, token, "model")
        out.append(_hf_result(f"hf.{pin.key}", f"{pin.repo}@{pin.revision[:8]}", r))
    return out, token


def check_store(p: Probes, cfg: Any) -> list[Check]:
    sink = cfg.output.sink
    env = cfg.modal.environment
    scheme = urlparse(sink).scheme
    out: list[Check] = []
    if scheme in ("s3", "r2", "gs"):
        r = p.modal_secret(BUCKET_SECRET, env)
        if r == "ok":
            out.append(Check("store.secret", "pass", f"Modal Secret {BUCKET_SECRET!r} exists"))
        elif r == "missing":
            out.append(Check("store.secret", "fail", f"no Modal Secret {BUCKET_SECRET!r} in "
                             f"environment {env!r} (workers write the bucket with it)",
                             f"modal secret create {BUCKET_SECRET} AWS_ACCESS_KEY_ID=... "
                             f"AWS_SECRET_ACCESS_KEY=... --env {env}"))
        else:
            out.append(Check("store.secret", "warn", f"could not look up the Secret ({r})",
                             "check `modal secret list`"))
    elif scheme == "volume":
        name = urlparse(sink).netloc
        r = p.modal_volume(name, env)
        if r == "ok":
            out.append(Check("store.volume", "pass", f"Volume {name!r} exists"))
        elif r == "missing":  # nothing to set up: deploy makes it (create_if_missing)
            return [Check("store.volume", "pass", f"Volume {name!r} does not exist yet in "
                          f"environment {env!r}; `flora deploy` creates it",
                          f"or create it now: modal volume create {name} --version 2 "
                          f"--env {env}")]
        else:
            out.append(Check("store.volume", "warn", f"could not look up the Volume ({r})",
                             "check `modal volume list`"))
    w = p.store_write(sink, env)
    if w == "ok":
        out.append(Check("store.write", "pass", f"{sink} is writable (probe written, deleted)"))
    elif w == "missing":
        out.append(Check("store.write", "fail", f"{sink} does not exist",
                         "create it, or fix output.sink"))
    elif w.startswith("skip:"):
        out.append(Check("store.write", "warn", f"write probe skipped: {w[5:]}",
                         "workers write through the Modal Secret; set local credentials to "
                         "probe it from here"))
    else:
        out.append(Check("store.write", "fail", f"cannot write {sink}: {w[6:]}",
                         "check the bucket name, credentials and permissions"))
    return out


def _mounts_cache(cfg: Any) -> bool:
    """Whether a worker of the config's shard kinds mounts ``flora-cache`` (read-only)."""
    from flora_deploy.modal_app.factory import WORKERS

    return any(ref.cache for k in cfg.pools() if (ref := WORKERS.get(k)) is not None)


def check_cache(p: Probes, cfg: Any) -> list[Check]:
    from flora_deploy.layout import CACHE_VOLUME

    if not _mounts_cache(cfg):
        return []
    env = cfg.modal.environment
    fix = f"modal volume create {CACHE_VOLUME} --version 2 --env {env}"
    r = p.modal_volume(CACHE_VOLUME, env)
    if r == "ok":
        return [Check("cache.volume", "pass", f"Volume {CACHE_VOLUME!r} exists")]
    if r == "missing":
        return [Check("cache.volume", "warn", f"Volume {CACHE_VOLUME!r} does not exist in "
                      f"environment {env!r}; the workers mount it read-only, which cannot "
                      "create it. `flora deploy` creates it; a `flora deploy --emit` script "
                      "deployed with `modal deploy` does not", f"or create it now: {fix}")]
    return [Check("cache.volume", "warn", f"could not look up the Volume {CACHE_VOLUME!r} ({r})",
                  f"check `modal volume list --env {env}`; if it is missing: {fix}")]


def _needed(cfg: Any) -> tuple[list[str], list[str]]:
    """Pin keys and derived-weight names the config's shard kinds need."""
    from flora_deploy.config.presets import get_preset, shard_kinds_for

    preset = get_preset(cfg.preset)
    kinds = shard_kinds_for(preset, cfg.pipeline.stages)
    pins: list[str] = []
    derived: list[str] = []
    for k in kinds:
        pins += [x for x in preset.pins.get(k, ()) if x not in pins]
        derived += [x for x in preset.derived_weights.get(k, ()) if x not in derived]
    return pins, derived


def check_weights(p: Probes, cfg: Any) -> list[Check]:
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import RECEIPT_NAME, WEIGHTS_VOLUME, snapshot_dirname

    env = cfg.modal.environment
    pins, derived = _needed(cfg)
    out: list[Check] = []
    listing = p.volume_listdir(WEIGHTS_VOLUME, env, "/snapshots")
    if listing is None:
        out.append(Check("weights.snapshots", "fail", f"no snapshots on Volume {WEIGHTS_VOLUME!r}",
                         "run `flora prepare`"))
    else:
        missing = []
        for key in pins:
            d = snapshot_dirname(PINS_BY_KEY[key])
            if d not in listing or RECEIPT_NAME not in (
                    p.volume_listdir(WEIGHTS_VOLUME, env, f"/snapshots/{d}") or ()):
                missing.append(key)
        if missing:
            out.append(Check("weights.snapshots", "fail", f"snapshots without a prep receipt: "
                             f"{', '.join(missing)}", "run `flora prepare`"))
        else:
            out.append(Check("weights.snapshots", "pass", f"{len(pins)} pinned snapshots "
                             "prepared"))
    if not derived:
        return out
    from flora_deploy.config.receipts import needed_by_kind, unpinned, unpinned_message

    bad = unpinned(derived)
    if bad:  # the workers refuse these at start (review 10 M1); plan / deploy refuse them too
        kinds = needed_by_kind(cfg)
        out.append(Check("weights.derived.pins", "fail",
                         f"derived weight(s) with no pinned content digest: "
                         f"{', '.join(n.replace('_', '-') for n in bad)}",
                         " ".join(unpinned_message(n, kinds=kinds.get(n, ()),
                                                   preset=cfg.preset) for n in bad)))
    weights = getattr(cfg, "weights", None)
    have = getattr(weights, "derived", None) if weights is not None else None
    if not have:  # nothing recorded yet: the usual state before `flora prepare`
        out.append(Check("weights.derived", "warn", "this config records no `weights.derived`; "
                         f"{', '.join(derived)} receipts are not checked",
                         "run `flora prepare` (it builds them and records them in flora.yaml), "
                         "or pass `--receipt name=<sha>` to deploy and run"))
        return out
    have = dict(have)
    missing = [d for d in derived if not have.get(d)]
    if len(missing) == len(derived):
        out.append(Check("weights.derived", "warn", "no derived-weight receipts recorded yet "
                         f"({', '.join(missing)})", "run `flora prepare` (it builds them and "
                         "records them under weights.derived), or pass --receipt name=<sha>"))
    elif missing:
        out.append(Check("weights.derived", "fail", f"missing derived-weight receipts: "
                         f"{', '.join(missing)}", "run `flora prepare` (it builds them and "
                         "records them under weights.derived)"))
    else:
        out.append(Check("weights.derived", "pass", f"receipts for {', '.join(derived)}"))
    return out


_CERT_OK = {"certified", "reference", "ok", "valid", "pass", "true"}


def _cert_ok(s: str) -> bool:
    """A probe's per-stage status that needs no opt-in: ``certified (human, L40S)``,
    ``reference (...)`` or a bare ``certified`` / ``reference``."""
    return s.lower() in _CERT_OK or s.startswith(("certified (", "reference ("))


def check_certificates(p: Probes, cfg: Any) -> list[Check]:
    """One line per certified stage, one per distinct reason a stage is not certified, in
    plain words. With ``allow_uncertified: true`` the uncertified stages are expected (pass);
    once every stage is certified, the opt-in is said to be no longer needed."""
    from flora_deploy.config.presets import find_certificates, needs_certificate

    if not needs_certificate(cfg.preset):
        return [Check("certificates", "pass", f"preset {cfg.preset!r} is the reference by "
                      "construction")]
    status = p.certificate_status(cfg)
    if status is None:
        certs = find_certificates(cfg.preset, cfg.pools())
        status = {k: ("not certified" if k not in certs else
                      "reference (NVIDIA's own engine, no certificate needed)"
                      if certs[k].verdict == "reference" else "certified")
                  for k in cfg.pools()}
    gpus = {k: pool.gpu for k, pool in cfg.pools().items()} if hasattr(cfg, "pools") else {}
    out = []
    lacking: dict[str, list[str]] = {}  # status -> kinds: one line per distinct problem
    for kind, s in sorted(status.items()):
        if _cert_ok(s):
            where = f" on {gpus[kind]}" if kind in gpus and gpus[kind] not in s else ""
            out.append(Check(f"certificates.{kind}", "pass", f"{kind}{where}: {s}"))
        else:
            lacking.setdefault(s, []).append(kind)
    allowed = bool(getattr(cfg, "allow_uncertified", False))
    for s, kinds in lacking.items():
        name = "certificates." + (kinds[0] if len(kinds) == 1 else ",".join(kinds))
        what = f"{', '.join(kinds)}: {_CERT_WHY.get(s, s)}"
        if allowed:  # expected for the README's config: say it once, not as a problem
            out.append(Check(name, "pass", what + "; runs uncertified (allow_uncertified: "
                             "true), its records say certified: false",
                             "use preset exact for reference outputs"))
        else:
            out.append(Check(name, "warn", what, "set allow_uncertified: true to run it "
                             "uncertified, or use preset exact"))
    if allowed and status and not lacking:
        emb = getattr(getattr(cfg, "pipeline", None), "embodiment", None)
        out.append(Check("allow_uncertified", "warn",
                         f"every stage is certified{f' for {emb}' if emb else ''}: "
                         "`allow_uncertified: true` is no longer needed",
                         "remove it, so a later change that is not certified is refused "
                         "instead of running uncertified"))
    return out


# raw statuses (probes that report them) in plain words; labels pass through unchanged
_CERT_WHY = {"none": "not certified", "missing": "not certified",
             "unverified_weights": "not checked: run `flora prepare` first",
             "provisional": "not certified (only a provisional certificate)",
             "fail": "not certified (its certificate failed)"}


def check_budget(cfg: Any) -> Check:
    from flora_deploy.jobs.budget import environment_budget_instruction, environment_budget_usd

    amount = environment_budget_usd(SimpleNamespace(
        budget_usd=cfg.modal.budget_usd,
        environment_budget_factor=cfg.modal.environment_budget_factor))
    return Check("budget", "pass", f"budget_usd ${cfg.modal.budget_usd:,.2f} (hard cap)",
                 environment_budget_instruction(cfg.modal.environment, amount))


def check_input(p: Probes, cfg: Any, token: str | None) -> Check:
    inp = cfg.input
    if inp.repo is not None:
        r = p.hf_repo(inp.repo, inp.revision, token, "dataset")
        c = _hf_result("input", f"dataset {inp.repo}@{inp.revision[:8]}", r)
        if c.status == "fail" and r == "revision_missing":
            return Check("input", "fail", c.message, "pin a commit that exists (the repo's "
                         f"History tab on huggingface.co/datasets/{inp.repo})")
        return c
    pattern = "*.mp4" if inp.format == "mp4_dir" else "*"
    root = cfg.local_input_root() if hasattr(cfg, "local_input_root") else inp.path
    n = p.local_files(str(root), pattern)
    if n is None:
        return Check("input", "fail", f"{inp.path} is not a directory", "fix input.path")
    if n == 0:
        return Check("input", "fail", f"no {pattern} files under {inp.path}", "fix input.path")
    return Check("input", "pass", f"{n} {pattern} file(s) under {inp.path}")


def run_checks(config: Path, probes: Probes) -> list[Check]:
    """Every check, in order. A config that does not load skips the checks that need it."""
    cfg_check, cfg = check_config(config)
    out = [cfg_check]
    out += check_modal(probes, cfg.modal.environment if cfg is not None else None)
    hf, token = check_hf(probes)
    out += hf
    if cfg is None:
        out.append(Check("rest", "warn", "store, cache, weights, certificate, budget and input checks "
                         "skipped (the config did not load)", "fix the config first"))
        return out
    steps: Iterable[Callable[[], list[Check] | Check]] = (
        lambda: check_store(probes, cfg),
        lambda: check_cache(probes, cfg),
        lambda: check_weights(probes, cfg),
        lambda: check_certificates(probes, cfg),
        lambda: check_budget(cfg),
        lambda: check_input(probes, cfg, token),
    )
    for step in steps:
        try:
            r = step()
        except Exception as e:  # a probe blew up: report it, keep going
            r = Check("doctor", "warn", f"a check crashed: {_err(e)[6:]}", "rerun; report a bug")
        out += r if isinstance(r, list) else [r]
    return out


ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="flora.yaml to check.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON.")]
_COLOR = {"pass": "green", "warn": "yellow", "fail": "red"}


@app.command()
def doctor(config: ConfigOpt = Path("flora.yaml"), as_json: JsonOpt = False) -> None:
    """Preflight checks (config, Modal, Hugging Face, store, weights, budget, input). No spend."""
    from rich.markup import escape

    checks = run_checks(config, _probes_factory())
    failed = any(c.status == "fail" for c in checks)
    if as_json:
        typer.echo(json.dumps({"ok": not failed, "checks": [asdict(c) for c in checks]},
                              indent=2))
    else:
        for c in checks:
            col = _COLOR[c.status]
            console.print(f"[{col}]{c.status.upper():4}[/{col}] {escape(c.name)}: "
                          f"{escape(c.message)}", soft_wrap=True)
            if c.fix:
                label = "note" if c.status == "pass" else "fix"
                console.print(f"     {label}: {escape(c.fix)}", soft_wrap=True)
        n = {s: sum(c.status == s for c in checks) for s in ("pass", "warn", "fail")}
        console.print(f"{n['pass']} passed, {n['warn']} warnings, {n['fail']} failed")
    raise typer.Exit(code=2 if failed else 0)


_probes_factory: Callable[[], Probes] = RealProbes
