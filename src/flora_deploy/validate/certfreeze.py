"""Certificate freeze: do the committed certificates still match the code?

See ``docs/certificate-freeze.md``. A certificate (``config/certificates/<id>.json``) matches a
pool only while the key of the stack that would run now (``certkey.engine_spec_hash``) equals
the key it was issued for. Any change to what the key covers (the logic digest of the kind's
modules, the engine identity and image fingerprint, the vLLM engine spec, the NVIDIA and
decode pins, the pinned derived-weight content digests, the kind's modes, ``LOGIC_VERSION``)
silently stops it matching, and users then see "uncertified". This module finds that before
it merges:

- :func:`check` recomputes the current key of every committed certificate and reports drift,
  with a diff of the key documents (per module, engine kwarg, pin and derived digest);
  ``tests/unit/test_certificate_drift.py`` fails on it, ``scripts/cert_freeze.py --check``
  prints it;
- :func:`freeze_listing` lists, per (preset, kind, GPU), what is frozen and which certificate
  covers it (``scripts/cert_freeze.py``);
- a **freeze lock** (``config/certificates/freeze/<id>.json``: the full key document and the
  sha256 of each module's normalized source) makes the diff exact. ``flora certify`` writes it
  next to every certificate it writes; ``scripts/cert_freeze.py --lock`` backfills it. It is
  diagnostic only: matching never reads it.

The current key is computed the way a worker would stamp it: the certificate's own modes (a
config choice) and ``dtbias_fp32`` flag, and each derived weight's **pinned** content digest
(``pins.DERIVED_PINS``; the certificate's value when none is pinned). Nothing here needs NVIDIA
code, a GPU or the network: NVIDIA pins enter the key as revisions and digests only.

Severity follows what a certificate unlocks. Only the deciding certificate of each (preset,
kind, GPU, key, pool) is checked (newest ``generated_at``; on a tie ``fail`` beats
``provisional`` beats ``certified``, as in ``certkey.matching``). A drifted ``certified`` one is
an **error**; a drifted ``provisional`` one is a **warning** (it unlocks nothing); a deciding
``fail`` (a revocation) and superseded certificates are **info**.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

from flora_deploy.validate import certkey as K

FREEZE_SCHEMA = "flora-cert-freeze/v1"
FREEZE_DIRNAME = "freeze"
SEVERITIES = ("error", "warning", "info")
RECERTIFY_DOC = "docs/certificate-freeze.md"


# ---------------------------------------------------------------------------------------------
# Freeze locks


def module_hashes(kind: str, root: Path | None = None) -> dict[str, str]:
    """``relpath -> sha256(normalized source)`` of the modules ``kind``'s certificate logic
    digest covers (``certkey.cert_modules``): the per-module view of ``logic_digest``."""
    from flora_deploy.config.logic import package_root

    base = root or package_root()
    out = {}
    for rel in sorted(K.cert_modules(kind)):
        text = K.normalized_source((base / rel).read_text(encoding="utf-8"))
        out[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def freeze_document(cert: Mapping[str, Any], doc: Mapping[str, Any],
                    root: Path | None = None) -> dict[str, Any]:
    """The freeze lock of ``cert``, whose key document is ``doc``. ``root``: the package
    source that produced ``doc`` (default: this checkout)."""
    if K.sha256_hex(dict(doc)) != cert.get("engine_spec_hash"):
        raise ValueError(f"{cert.get('id')}: the key document is not the certificate's key")
    return {"schema": FREEZE_SCHEMA, "certificate": cert["id"],
            "engine_spec_hash": cert["engine_spec_hash"], "key_document": dict(doc),
            "modules": module_hashes(doc["kind"], root)}


def freeze_path(cert_dir: Path, cert_id: str) -> Path:
    return Path(cert_dir) / FREEZE_DIRNAME / f"{cert_id}.json"


def write_freeze(cert: Mapping[str, Any], doc: Mapping[str, Any], cert_dir: Path,
                 root: Path | None = None) -> Path:
    """Write the freeze lock of ``cert`` under ``cert_dir/freeze/``."""
    p = freeze_path(cert_dir, cert["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(freeze_document(cert, doc, root), indent=1, sort_keys=True) + "\n",
                 encoding="utf-8")
    return p


def read_freeze(cert_dir: Path, cert: Mapping[str, Any]) -> dict[str, Any] | None:
    """The freeze lock of ``cert`` when present and consistent with it (else None)."""
    p = freeze_path(cert_dir, cert["id"])
    try:
        lock = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(lock, dict) or lock.get("schema") != FREEZE_SCHEMA
            or lock.get("certificate") != cert["id"]
            or not isinstance(lock.get("key_document"), dict)
            or K.sha256_hex(lock["key_document"]) != cert.get("engine_spec_hash")):
        return None
    return lock


# ---------------------------------------------------------------------------------------------
# The current key of a certificate


def embodiments() -> tuple[str, ...]:
    from flora_deploy.config.schema import Embodiment

    return tuple(get_args(Embodiment))


def current_derived(preset: str, kind: str, recorded: Mapping[str, str]) -> dict[str, str]:
    """The derived-weight content digests the workers would load now: the pinned ones
    (``workers_impl.certstamp.pinned_content``), else the certificate's own."""
    from flora_deploy.workers_impl.certstamp import pinned_content

    return {**{k: v for k, v in recorded.items()}, **pinned_content(preset, kind)}


def current_key_document(cert: Mapping[str, Any], **kw: Any) -> dict[str, Any]:
    """``certkey.key_document`` of ``cert``'s (preset, kind, GPU) as the code stands, under
    the certificate's modes and ``dtbias_fp32``. Raises ``CertKeyError`` / ``KeyError`` /
    ``PresetUnavailableError`` when the key cannot be computed."""
    vllm = (cert.get("engine") or {}).get("vllm") or {}
    return K.key_document(
        cert["preset"], cert["kind"], cert["gpu"], modes=cert.get("modes") or {},
        derived_content=current_derived(cert["preset"], cert["kind"],
                                        cert.get("derived_weights") or {}),
        dtbias_fp32=bool(vllm.get("dtbias_fp32")), **kw)


def lock_document(cert: Mapping[str, Any], source_dir: Path | None = None) -> dict[str, Any]:
    """The key document to freeze ``cert`` with: the one in ``source_dir``'s freeze lock (what
    ``flora certify`` keyed it on) when that lock is consistent, else the key this tree
    computes. Raises ``CertKeyError`` when neither hashes to the certificate's
    ``engine_spec_hash`` (the tree no longer computes its key: drift), or when the key cannot
    be computed."""
    if source_dir is not None:
        lock = read_freeze(source_dir, cert)
        if lock is not None:
            return dict(lock["key_document"])
    doc = current_key_document(cert)
    if K.sha256_hex(doc) != cert.get("engine_spec_hash"):
        raise K.CertKeyError(f"{cert.get('id')}: this tree no longer computes its key (drift; "
                             "run scripts/cert_freeze.py --check)")
    return doc


def recorded_view(cert: Mapping[str, Any]) -> dict[str, Any]:
    """What a certificate body records of its key document (no freeze lock)."""
    eng = cert.get("engine") or {}
    logic = cert.get("logic") or {}
    return {"engine": {k: eng.get(k) for k in ("name", "version", "image", "image_fingerprint",
                                               "patches")},
            "vllm_engine": eng.get("vllm"),
            "nvidia_pins": {k: {"revision": v} for k, v in (cert.get("pins") or {}).items()},
            "derived_weights": dict(cert.get("derived_weights") or {}),
            "modes": dict(cert.get("modes") or {}),
            "logic_version": logic.get("version"), "logic_digest": logic.get("digest")}


def project(doc: Mapping[str, Any]) -> dict[str, Any]:
    """``doc`` reduced to :func:`recorded_view`'s shape."""
    eng = doc.get("engine") or {}
    return {"engine": {k: eng.get(k) for k in ("name", "version", "image", "image_fingerprint",
                                               "patches")},
            "vllm_engine": doc.get("vllm_engine"),
            "nvidia_pins": {k: {"revision": (v or {}).get("revision")}
                            for k, v in (doc.get("nvidia_pins") or {}).items()},
            "derived_weights": dict(doc.get("derived_weights") or {}),
            "modes": dict(doc.get("modes") or {}),
            "logic_version": doc.get("logic_version"), "logic_digest": doc.get("logic_digest")}


_MISSING = object()


def _short(v: Any) -> str:
    if v is _MISSING:
        return "(absent)"
    s = json.dumps(v, sort_keys=True)
    return s if len(s) <= 72 else s[:69] + "..."


def diff_docs(old: Any, new: Any, path: str = "") -> list[str]:
    """``path: old -> new`` per differing leaf (dicts recursed, everything else compared)."""
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        out = []
        for k in sorted(set(old) | set(new), key=str):
            p = f"{path}.{k}" if path else str(k)
            out += diff_docs(old.get(k, _MISSING), new.get(k, _MISSING), p)
        return out
    if old == new:
        return []
    return [f"{path}: {_short(old)} -> {_short(new)}"]


def module_diff(old: Mapping[str, str], new: Mapping[str, str]) -> list[str]:
    out = []
    for rel in sorted(set(old) | set(new)):
        if rel not in new:
            out.append(f"{rel}: removed from the kind's modules")
        elif rel not in old:
            out.append(f"{rel}: added to the kind's modules")
        elif old[rel] != new[rel]:
            out.append(f"{rel}: code changed ({old[rel][:12]} -> {new[rel][:12]})")
    return out


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30,
                           check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def modules_touched_since(cert_path: Path, kind: str) -> list[str] | None:
    """Without a freeze lock: the kind's modules that commits after the one adding
    ``cert_path`` touched (git history; None when it is not available, e.g. a shallow clone)."""
    from flora_deploy.config.logic import package_root

    base = package_root()
    added = _git(["log", "--diff-filter=A", "--format=%H", "-n", "1", "--",
                  str(cert_path)], cert_path.parent)
    if not added or not added.strip():
        return None
    top = _git(["rev-parse", "--show-toplevel"], cert_path.parent)
    if not top:
        return None
    paths = [str(base / rel) for rel in K.cert_modules(kind)]
    out = _git(["log", "--format=", "--name-only", f"{added.strip()}..HEAD", "--", *paths],
               Path(top.strip()))
    if out is None:
        return None
    prefix = Path(base).resolve().relative_to(Path(top.strip()).resolve()).as_posix() + "/"
    return sorted({ln[len(prefix):] if ln.startswith(prefix) else ln
                   for ln in out.splitlines() if ln.strip()})


# ---------------------------------------------------------------------------------------------
# Drift check


@dataclass
class Finding:
    severity: str  # error | warning | info
    status: str  # ok | drift | orphan | invalid | policy | superseded | revoked | no_lock | ...
    cert: str  # certificate id or file name
    what: str  # (preset, kind, gpu, pools, verdict) summary
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class Report:
    findings: list[Finding]
    certificates: int

    def by(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def ok(self) -> bool:
        return not self.by("error")


def _what(c: Mapping[str, Any]) -> str:
    return (f"{c.get('preset')} {c.get('kind')} on {c.get('gpu')} for "
            f"{c.get('covered_pools')} ({c.get('verdict')}, {c.get('generated_at')})")


def raw_certificates(cert_dir: Path) -> list[tuple[Path, Any]]:
    """Every ``*.json`` directly in ``cert_dir`` (freeze locks live in a subdirectory)."""
    out = []
    for p in sorted(Path(cert_dir).glob("*.json")):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as e:
            out.append((p, e))
    return out


def deciding(certs: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Certificate id -> the pools it decides (``certkey.matching``'s order per (preset, kind,
    GPU, key, pool)). A certificate absent here decides nothing (superseded)."""
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for c in certs:
        for pool in c.get("covered_pools") or ():
            groups.setdefault((c["preset"], c["kind"], c["gpu"], c["engine_spec_hash"], pool),
                              []).append(c)
    out: dict[str, list[str]] = {}
    for key, cs in groups.items():
        best = max(cs, key=lambda c: (c["generated_at"], K.VERDICTS.index(c["verdict"]),
                                      c["id"]))
        out.setdefault(best["id"], []).append(key[-1])
    return out


def orphan_problems(c: Mapping[str, Any]) -> list[str]:
    """Why ``c`` names a preset, kind, GPU or embodiment the code no longer has."""
    from flora_deploy.config.presets import GPU_MEMORY_GIB, PRESETS, gpu_problem

    spec = PRESETS.get(str(c.get("preset")))
    if spec is None or not spec.available:
        return [f"preset {c.get('preset')!r} is not an available preset"]
    if c.get("kind") not in spec.engines:
        return [f"shard kind {c.get('kind')!r} is not in preset {c['preset']!r}"]
    out = []
    pool = spec.pools[c["kind"]]
    if c.get("gpu") not in GPU_MEMORY_GIB:
        out.append(f"GPU {c.get('gpu')!r} is not a known Modal GPU (presets.GPU_MEMORY_GIB)")
    else:
        bad = gpu_problem(c["preset"], c["kind"], c["gpu"], pool.gpu_count)
        if bad:
            out.append(f"GPU no longer usable for this kind: {bad}")
    gone = [p for p in c.get("covered_pools") or () if p not in embodiments()]
    if gone:
        out.append(f"embodiment pool(s) {gone} no longer exist (config.schema.Embodiment)")
    return out


def _modes_shape(c: Mapping[str, Any]) -> list[str]:
    """The kind's output-affecting mode *names* now (``config.hashing.kind_modes`` on the
    template config) vs the certificate's: a new or removed mode changes every worker's key."""
    try:
        now = default_modes(c["preset"]).get(c["kind"])
    except Exception:  # noqa: BLE001 - diagnostic only
        return []
    if now is None:
        return []
    had, has = sorted(c.get("modes") or {}), sorted(now)
    if had == has:
        return []
    return [f"modes: the kind's mode names changed (config.hashing.kind_modes): {had} -> {has}; "
            "workers now stamp the new modes"]


def check(cert_dir: Path | None = None, *, logic_root: Path | None = None,
          image_fingerprint: Callable[[str], str] | None = None,
          git_history: bool = True) -> Report:
    """Every committed certificate in ``cert_dir`` against the current code (module doc)."""
    d = Path(cert_dir) if cert_dir is not None else K.CERT_DIR
    findings: list[Finding] = []
    valid: list[dict[str, Any]] = []
    raws = raw_certificates(d)
    for p, body in raws:
        if isinstance(body, Exception):
            findings.append(Finding("error", "invalid", p.name, "?", f"not JSON: {body}"))
            continue
        why = _invalid(body)
        if why:
            findings.append(Finding(
                "error", "invalid", p.name, _what(body) if isinstance(body, dict) else "?",
                f"the registry ignores this file ({why}): it matches nothing. Delete it, or "
                "re-certify (a hand edit breaks the content address)"))
            continue
        if p.stem != body["id"]:
            findings.append(Finding("warning", "misnamed", p.name, _what(body),
                                    f"file name is not its id {body['id']}.json"))
        valid.append(body)
    decides = deciding(valid)
    kw: dict[str, Any] = {"logic_root": logic_root, "image_fingerprint": image_fingerprint}
    for c in valid:
        findings += _check_one(c, d, decides.get(c["id"]), kw, git_history)
    live_ids = {c["id"] for c in valid}
    for lock in sorted((d / FREEZE_DIRNAME).glob("*.json")) if (d / FREEZE_DIRNAME).is_dir() \
            else ():
        if lock.stem not in live_ids:
            findings.append(Finding("warning", "orphan_lock", lock.name, "-",
                                    "freeze lock without its certificate: delete it "
                                    "(scripts/cert_freeze.py --lock prunes it)"))
    return Report(findings, len(valid))


def _invalid(body: Any) -> str | None:
    if not isinstance(body, dict):
        return "not an object"
    if body.get("schema") != K.CERT_SCHEMA:
        return f"schema {body.get('schema')!r} != {K.CERT_SCHEMA!r}"
    if body.get("verdict") not in K.VERDICTS:
        return f"verdict {body.get('verdict')!r}"
    if not K._STAMP.match(str(body.get("generated_at") or "")):
        return "generated_at missing or malformed"
    if not isinstance(body.get("covered_pools"), list):
        return "covered_pools is not a list"
    for f in ("preset", "kind", "gpu", "engine_spec_hash"):
        if not isinstance(body.get(f), str):
            return f"{f} missing"
    if body.get("id") != K.certificate_id(body):
        return "id is not its content address"
    return None


def _check_one(c: dict[str, Any], d: Path, pools: list[str] | None, kw: dict[str, Any],
               git_history: bool) -> list[Finding]:
    what = _what(c)
    if pools is None:
        return [Finding("info", "superseded", c["id"], what,
                        "superseded by a newer certificate for the same key and pools; it "
                        "decides nothing (delete it when convenient)")]
    if c["verdict"] == "fail":
        return [Finding("info", "revoked", c["id"], what,
                        f"a fail certificate decides {pools}: it revokes any older certificate "
                        "for this key; not checked for drift")]
    sev = "error" if c["verdict"] == "certified" else "warning"
    out: list[Finding] = []
    orphan = orphan_problems(c)
    if orphan:
        return [Finding(sev, "orphan", c["id"], what,
                        "the certificate names something the code no longer has, so it can "
                        "never match", orphan + _choice(c))]
    pol = K.policy_problems(c)
    if pol:
        out.append(Finding(sev, "policy", c["id"], what,
                           "no longer meets certkey.POLICY (or GENERATOR_VERSIONS), so "
                           "matching() refuses it", pol + _choice(c)))
    lock = read_freeze(d, c)
    try:
        doc = current_key_document(c, **kw)
    except Exception as e:  # noqa: BLE001 - CertKeyError, KeyError, PresetUnavailableError...
        return [*out, Finding(sev, "drift", c["id"], what,
                              f"the current key cannot be computed: {e}", _choice(c))]
    key = K.sha256_hex(doc)
    if key == c["engine_spec_hash"]:
        if not out:
            out.append(Finding("info", "ok", c["id"], what, f"key {key[:16]} matches"))
            out += _confirm_match(c, doc, pools, kw, sev, d)
        if lock is None:
            out.append(Finding("warning", "no_lock", c["id"], what,
                               "no freeze lock (config/certificates/freeze/"
                               f"{c['id']}.json): a later drift report cannot name the changed "
                               "modules. Run `uv run python scripts/cert_freeze.py --lock` and "
                               "commit it"))
        return out
    details = []
    if lock is not None:
        details += ["key document (certified -> now):",
                    *(f"  {x}" for x in diff_docs(lock["key_document"], doc))]
        root = kw.get("logic_root")
        mods = module_diff(lock.get("modules") or {}, module_hashes(c["kind"], root))
        if mods:
            details += ["modules (normalized source; comments and docstrings ignored):",
                        *(f"  {x}" for x in mods)]
    else:
        visible = diff_docs(recorded_view(c), project(doc))
        details += ["key fields the certificate records (certified -> now):",
                    *(f"  {x}" for x in visible)]
        if not visible:
            details.append("  (none: the change is in a field the certificate does not record "
                           "in full: decode pins, NVIDIA pin digests or engine args)")
        if any(x.startswith("logic_digest") for x in visible):
            touched = (modules_touched_since(d / f"{c['id']}.json", c["kind"])
                       if git_history else None)
            if touched:
                details += ["modules of the kind touched since the certificate was committed "
                            "(git; no freeze lock):", *(f"  {x}" for x in touched)]
            else:
                details.append("  (no freeze lock and no git history: per-module breakdown "
                               "unavailable; the kind's modules are listed by "
                               "`scripts/cert_freeze.py`)")
    details += _modes_shape(c)
    details += _choice(c)
    out.append(Finding(sev, "drift", c["id"], what,
                       f"key drift: the certificate is for {c['engine_spec_hash'][:16]}, the code "
                       f"now computes {key[:16]}; it no longer matches", details))
    return out


def _confirm_match(c: Mapping[str, Any], doc: Mapping[str, Any], pools: list[str],
                   kw: dict[str, Any], sev: str, cert_dir: Path) -> list[Finding]:
    """End to end: ``certkey.matching`` itself must select ``c`` for every pool it decides."""
    out = []
    vllm = (c.get("engine") or {}).get("vllm") or {}
    for pool in pools:
        found = K.matching(
            c["preset"], {c["kind"]: SimpleNamespace(gpu=c["gpu"])}, embodiment=pool,
            modes={c["kind"]: c.get("modes") or {}},
            derived_content=dict(doc["derived_weights"]),
            dtbias_fp32=bool(vllm.get("dtbias_fp32")), cert_dir=cert_dir,
            **{k: v for k, v in kw.items() if v is not None})
        best = (found.get(c["kind"]) or [None])[0]
        if best is None or best["id"] != c["id"]:
            out.append(Finding(sev, "no_match", c["id"], _what(c),
                               "the key matches but certkey.matching() does not select it for "
                               f"pool {pool!r} (selected: {best and best['id']})", _choice(c)))
    return out


def _choice(c: Mapping[str, Any]) -> list[str]:
    what = f"{c['preset']} {c['kind']} on {c['gpu']}"
    return ["Choose one:",
            f"  (a) revert the change, if it was not meant to change {what} outputs; or",
            f"  (b) re-certify {what}: `uv run python scripts/wave_h.py --plan certified`, then "
            "`flora certify` (or `scripts/wave_h_finalize.py --write`), then `uv run python "
            "scripts/cert_freeze.py --lock`; delete the superseded certificate and commit "
            f"both files ({RECERTIFY_DOC})"]


def format_report(report: Report, *, markdown: bool = False, verbose: bool = False) -> str:
    """The report as text: errors first (the drift diffs), then warnings, then info."""
    lines = []
    n_err, n_warn = len(report.by("error")), len(report.by("warning"))
    head = (f"certificate drift: {n_err} error(s), {n_warn} warning(s) over "
            f"{report.certificates} committed certificate(s)")
    lines.append(f"### {head}" if markdown else head)
    if not report.certificates and not report.findings:
        lines.append("no certificates committed (src/flora_deploy/config/certificates/): "
                     "nothing is frozen")
    for sev in SEVERITIES:
        if sev == "info" and not verbose:
            continue
        for f in report.by(sev):
            tag = sev.upper()
            lines.append("")
            lines.append(f"{'**' if markdown else ''}{tag} [{f.status}] {f.cert}"
                         f"{'**' if markdown else ''}: {f.what}")
            lines.append(f"  {f.message}")
            if f.details:
                if markdown:
                    lines.append("```")
                lines += [f"  {x}" for x in f.details]
                if markdown:
                    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Freeze listing


def default_modes(preset: str) -> dict[str, dict[str, Any]]:
    """``config.hashing.kind_modes`` of each kind of ``preset`` under its template config
    (``config/templates/<preset>_lerobot.yaml``, every stage): the modes a default job stamps."""
    from importlib import resources

    from flora_deploy.config.hashing import kind_modes
    from flora_deploy.config.presets import get_preset
    from flora_deploy.config.schema import loads_yaml, parse_config

    text = (resources.files("flora_deploy.config") / "templates"
            / f"{preset}_lerobot.yaml").read_text(encoding="utf-8")
    cfg = parse_config(loads_yaml(text), allow_uncertified=True)
    out = {}
    for kind in get_preset(preset).pools:
        try:
            out[kind] = kind_modes(cfg, kind)
        except Exception:  # noqa: BLE001 - e.g. a rubric spec the template lacks
            continue
    return out


def _digest(v: Any) -> str:
    return K.sha256_hex(v)[:12]


def freeze_listing(cert_dir: Path | None = None, *,
                   presets: Iterable[str] | None = None) -> str:
    """Per (preset, kind, GPU): the key's modules and settings with their hashes, and the
    certificates that cover them. GPUs: the preset's default pool, plus any GPU a committed
    certificate names."""
    from flora_deploy.config.presets import AVAILABLE_PRESETS, get_preset

    d = Path(cert_dir) if cert_dir is not None else K.CERT_DIR
    certs = K.load_certificates(d)
    decides = deciding(certs)
    names = list(presets) if presets is not None else [p for p in AVAILABLE_PRESETS
                                                        if p != "exact"]
    lines = ["# Certificate freeze listing (scripts/cert_freeze.py)", "",
             "`exact` is the reference by construction: nothing of it is frozen."]
    for preset in names:
        spec = get_preset(preset)
        modes = default_modes(preset)
        for kind, pool in spec.pools.items():
            gpus = [pool.gpu] + sorted({c["gpu"] for c in certs if c["preset"] == preset
                                        and c["kind"] == kind and c["gpu"] != pool.gpu})
            for gpu in gpus:
                lines += ["", *_listing_one(preset, kind, gpu, gpu == pool.gpu,
                                            modes.get(kind, {}), certs, decides)]
    return "\n".join(lines) + "\n"


def _listing_one(preset: str, kind: str, gpu: str, default: bool, modes: Mapping[str, Any],
                 certs: list[dict[str, Any]], decides: Mapping[str, list[str]]) -> list[str]:
    from flora_deploy.workers_impl.certstamp import pinned_content

    head = f"## {preset} / {kind} / {gpu}" + (" (default pool)" if default else "")
    derived = pinned_content(preset, kind)
    needed = K.needed_derived(preset, kind)
    placeholder = {n: derived.get(n) or "0" * 64 for n in needed}
    try:
        doc = K.key_document(preset, kind, gpu, modes=modes, derived_content=placeholder)
    except Exception as e:  # noqa: BLE001
        return [head, f"key: cannot be computed ({e})"]
    key = K.sha256_hex(doc)
    unpinned = [n for n in needed if n not in derived]
    out = [head, "",
           f"- key (`engine_spec_hash`, default modes, pinned derived digests): `{key[:16]}`"
           + (f" (derived {unpinned} unpinned: placeholder digest)" if unpinned else "")]
    e = doc["engine"]
    out.append(f"- engine: `{e['name']}` {e['version']}, image `{e['image']}` fingerprint "
               f"`{e['image_fingerprint'][:12]}`, patches {list(e['patches'])}, engine_args "
               f"`{_digest(e['engine_args'])}`, sampling `{_digest(e['sampling'])}`, mm_kwargs "
               f"`{_digest(e['mm_kwargs'])}`")
    v = doc["vllm_engine"]
    if v:
        kw = v["engine_kwargs"]
        out.append(f"- vllm_engine: `{v['label']}` (dtbias_fp32={v['dtbias_fp32']}), "
                   f"max_num_seqs={kw.get('max_num_seqs')}, max_num_batched_tokens="
                   f"{kw.get('max_num_batched_tokens')}, gpu_memory_utilization="
                   f"{kw.get('gpu_memory_utilization')}, engine_kwargs `{_digest(kw)}`")
    else:
        out.append("- vllm_engine: none (not a balanced vLLM video kind)")
    pins = ", ".join(f"`{k}`@{p['revision'][:12]} (`{_digest(p)}`)"
                     for k, p in doc["nvidia_pins"].items())
    out.append(f"- nvidia_pins: {pins}")
    dec = doc["decode"]
    out.append(f"- decode: av {dec['av']}, pillow {dec['pillow']}, numpy {dec['numpy']}, "
               f"arch {dec['arch']}")
    dw = ", ".join(f"`{n}` {'`' + d[:12] + '`' if n in derived else 'UNPINNED'}"
                   for n, d in doc["derived_weights"].items()) or "none"
    out.append(f"- derived_weights: {dw}")
    out.append(f"- modes: {json.dumps(doc['modes'], sort_keys=True)}")
    out.append(f"- logic: `{doc['logic_version']}`, cert logic digest "
               f"`{doc['logic_digest'][:16]}` over {len(K.cert_modules(kind))} modules:")
    for rel, h in module_hashes(kind).items():
        out.append(f"  - `{rel}` `{h[:12]}`")
    cov = [c for c in certs if c["preset"] == preset and c["kind"] == kind and c["gpu"] == gpu]
    if not cov:
        out.append("- covered by: **no certificate** (uncertified; nothing frozen yet)")
    for c in sorted(cov, key=lambda c: c["generated_at"], reverse=True):
        state = ("matches" if c["engine_spec_hash"] == key else
                 "DOES NOT match the default key (other modes / derived digest, or drift)")
        role = (f"decides {decides[c['id']]}" if c["id"] in decides else "superseded")
        out.append(f"- covered by: `{c['id']}` {c['verdict']} {c['covered_pools']} "
                   f"{c['generated_at']}, {role}; {state}")
    return out
