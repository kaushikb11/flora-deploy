"""Go / no-go before the certified run (``wave_h.py --plan certified --yes-spend``).

Usage::

    uv run python scripts/cert_preflight.py                     # human (default), text
    uv run python scripts/cert_preflight.py --json              # the same, as JSON
    uv run python scripts/cert_preflight.py --embodiment yam    # golden/v3-yam episodes
    uv run python scripts/cert_preflight.py --repo ../flora-deploy   # check another checkout

Every check is ``pass``, ``warn`` or ``fail`` and carries a one-line fix. Any ``fail`` is
NO-GO (exit 1); warnings alone are GO (exit 0); a usage error exits 2. It only reads: git and
``gh`` metadata, ``modal app list``, ``modal billing report`` (``scripts/spend.py``), Volume
listings and small files read through the Modal client (no container), the runbook's own
``--status`` / ``--dry-run`` (which never write the state), ``cert_freeze.py --check``,
``lock_images.py --check`` (re-resolves the image locks against PyPI in a scratch directory)
and one unit test file. It never starts, stops or deploys anything (``docs/wave-h-runbook.md``,
"Before a paid run").

Checks: ``git`` (clean, pushed, CI green for this exact SHA), ``plan`` (the next step of the
certified plan would run, caps, holds), ``episodes`` (the golden episodes are staged, sizes
match), ``weights`` (``DERIVED_PINS`` match the receipts), ``images`` (the image locks are
current), ``apps`` (no ``flora*`` app running), ``lab`` (the lab runs, or its saved evidence was
taken with the current engine settings), ``freeze`` (the certificate freeze guard works) and
``budget`` (billed + the plan's upper total fits the project budget with a margin).

The plan is read from ``wave_h.py --status --plan certified --json`` when the runbook has it
(the shared ``scripts/_runbook.py`` ``decide()``); otherwise from ``--status`` text plus the
``--dry-run`` listing (:func:`load_plan`), each for the preflight's ``--embodiment``. Only the
runbook's CLI is used, except for the lab check's current engine-settings hash, which comes from
``wave_h.lab_engine_hashes`` (what the runbook's lab reuse compares), in a child process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_BUDGET_USD = 50.0  # the user's total Modal budget for the whole build
MARGIN_USD = 2.0  # kept free on top of the upper projection (billing lags about an hour)
BILLING_SINCE = "2026-10-20"
PLAN = "certified"
GOLDEN_VOLUME = "flora-golden"
GOLDEN = {"human": ("golden/v2", "human"), "yam": ("golden/v3-yam", "yam")}
LAB_KINDS = ("tempseg", "captioner")
HASH_KEY = "engine_settings_hash"  # a lab hash per kind (status JSON; older lab files)
LAB_HASHES = "engine_hashes"  # balanced_lab results: {"baseline", "plugin"} engine hashes
LIVE_STATES = {"running", "deployed", "ephemeral", "initializing", "stopping",
               "deployed (detached)"}
PASS, WARN, FAIL = "pass", "warn", "fail"
TEST_ENV = {"OMP_NUM_THREADS": "1"}

Runner = Callable[..., "Result"]


@dataclass
class Result:
    rc: int
    out: str = ""
    err: str = ""


@dataclass
class Check:
    name: str
    status: str
    message: str
    fix: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def run(cmd: Sequence[str], *, cwd: Path, timeout: float = 300.0,
        env: Mapping[str, str] | None = None) -> Result:
    """One subprocess; a missing binary or a timeout is rc 127 / 124, never an exception."""
    try:
        p = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
    except FileNotFoundError as e:
        return Result(127, "", str(e))
    except subprocess.TimeoutExpired:
        return Result(124, "", f"timed out after {timeout:.0f} s")
    return Result(p.returncode, p.stdout, p.stderr)


def _tail(r: Result, n: int = 200) -> str:
    return " ".join((r.err or r.out).split())[-n:]


@dataclass
class Env:
    """Everything the checks read, injectable for tests (fake ``gh``, Modal, Volume)."""

    repo: Path = ROOT
    run: Runner = run
    volume: Callable[[str], Any] | None = None  # name -> object with read_file / listdir
    uv: tuple[str, ...] = ("uv", "run")

    def sh(self, *cmd: str, timeout: float = 300.0, env: Mapping[str, str] | None = None
           ) -> Result:
        return self.run(cmd, cwd=self.repo, timeout=timeout, env=env)

    def py(self, *args: str, timeout: float = 300.0, env: Mapping[str, str] | None = None
           ) -> Result:
        return self.sh(*self.uv, "python", *args, timeout=timeout, env=env)

    def vol(self, name: str) -> Any:
        if self.volume is not None:
            return self.volume(name)
        import modal

        return modal.Volume.from_name(name, version=2)


# ---------------------------------------------------------------------------------------------
# 1. Git


def check_git(env: Env) -> Check:
    name = "git"
    r = env.sh("git", "status", "--porcelain")
    if r.rc != 0:
        return Check(name, FAIL, f"git status failed: {_tail(r)}", "run from a git checkout")
    if r.out.strip():
        n = len(r.out.strip().splitlines())
        return Check(name, FAIL, f"working tree is dirty ({n} path(s))",
                     "commit or remove the changes, then push", {"dirty": r.out.splitlines()})
    sha = env.sh("git", "rev-parse", "HEAD").out.strip()
    remote = env.sh("git", "ls-remote", "origin", "refs/heads/main")
    tip = remote.out.split()[0] if remote.rc == 0 and remote.out.split() else ""
    d: dict[str, Any] = {"sha": sha, "origin_main": tip}
    if not tip:
        return Check(name, FAIL, f"cannot read origin/main: {_tail(remote)}",
                     "check the network and `git remote -v`", d)
    if tip != sha:
        anc = env.sh("git", "merge-base", "--is-ancestor", sha, tip)
        if anc.rc == 0:
            return Check(name, FAIL, f"HEAD {sha[:10]} is behind origin/main {tip[:10]}",
                         "git pull --ff-only, so the paid run uses the reviewed tip", d)
        return Check(name, FAIL, f"HEAD {sha[:10]} is not on origin/main ({tip[:10]})",
                     "push HEAD to main (or check out origin/main)", d)
    ci, why = ci_runs(env, sha)
    if ci is None:
        return Check(name, WARN, f"HEAD {sha[:10]} is origin/main; CI not checked ({why})",
                     "install and log in to `gh`, or check the Actions page for this SHA", d)
    d["ci"] = ci
    if not ci:
        return Check(name, FAIL, f"no CI run for {sha[:10]}",
                     "wait for the push's CI run to start", d)
    pending = [c for c in ci if c["status"] != "completed"]
    if pending:
        return Check(name, FAIL, f"CI for {sha[:10]} is still {pending[0]['status']} "
                     f"({', '.join(c['name'] for c in pending)})",
                     "wait for CI to finish, then re-run this preflight", d)
    bad = [f"{c['name']}: {c['conclusion']}" for c in ci
           if c["conclusion"] not in ("success", "skipped", "neutral")]
    if bad:
        return Check(name, FAIL, f"CI for {sha[:10]} did not pass: {'; '.join(bad)}",
                     "fix the failing job and push; a paid run needs green CI on its SHA", d)
    return Check(name, PASS, f"clean; HEAD {sha[:10]} is origin/main; CI green "
                 f"({len(ci)} job(s))", "", d)


def ci_runs(env: Env, sha: str) -> tuple[list[dict] | None, str]:
    """Every job of every CI run for ``sha`` (``name``, ``status``, ``conclusion``); None when
    ``gh`` cannot answer. Old ``gh`` has no ``--commit``: the recent runs are filtered by SHA."""
    fields = "databaseId,name,status,conclusion,headSha"
    r = env.sh("gh", "run", "list", "--commit", sha, "--json", fields, timeout=60)
    if r.rc != 0 and "unknown flag" in (r.err + r.out):
        r = env.sh("gh", "run", "list", "--limit", "100", "--json", fields, timeout=60)
    if r.rc != 0:
        return None, ("`gh` not installed" if r.rc == 127 else f"gh: {_tail(r, 120)}")
    try:
        runs = [x for x in json.loads(r.out or "[]") if x.get("headSha", sha) == sha]
    except ValueError:
        return None, "gh printed no JSON"
    jobs: list[dict] = []
    for x in runs:
        v = env.sh("gh", "run", "view", str(x["databaseId"]), "--json", "jobs", timeout=60)
        try:
            js = json.loads(v.out)["jobs"] if v.rc == 0 else None
        except (ValueError, KeyError):
            js = None
        if not js:  # the run itself stands for its jobs
            js = [{"name": x.get("name"), "status": x.get("status"),
                   "conclusion": x.get("conclusion")}]
        for j in js:
            jobs.append({"name": f"{x.get('name')}/{j.get('name')}",
                         "status": j.get("status") or "",
                         "conclusion": j.get("conclusion") or ""})
    return jobs, ""


# ---------------------------------------------------------------------------------------------
# 2. The plan decision (the runbook's CLI only)


@dataclass
class PlanInfo:
    """The certified plan as the runbook reports it. ``next_action``: run | wait | stop | done."""

    source: str
    embodiment: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)  # name, status, cap, upper, rule
    nominal: float | None = None
    upper: float | None = None
    global_cap: float | None = None
    spent: float | None = None
    next_step: str | None = None
    next_action: str | None = None
    next_message: str = ""
    reasons: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    lab_runs: bool | None = None  # None: the lab is not in the plan / not reported
    lab_why: str = ""
    lab_hashes: dict[str, str | None] = field(default_factory=dict)  # current, per kind


_MONEY = r"\$(-?[0-9.]+)"
_STEP = re.compile(r"^\s{2}(?P<title>(?P<letter>[a-z0-9][0-9]?)\.\s.*?)\s+\[(?P<status>[^\]]+)\]\s+"
                   rf"cap {_MONEY}\s+nominal {_MONEY}\s+upper {_MONEY}\s+rule: (?P<rule>.*)$")
_LETTERS = {"b": "exact_e2e", "f": "noise", "d": "balanced", "d2": "lab", "e": "certify",
            "a": "preflight", "0": "tempseg"}


def parse_dry_run(text: str) -> PlanInfo:
    """``wave_h.py --plan certified --dry-run``'s listing (the fallback until ``--status
    --json``). The next step is the first one not done; its ``rule: ok`` means it runs."""
    p = PlanInfo("dry-run")
    for line in text.splitlines():
        if m := re.match(r"^Wave H-lite run \S+ \((\w+) only\).*spent " + _MONEY, line):
            p.embodiment, p.spent = m.group(1), float(m.group(2))
        elif m := _STEP.match(line):
            letter = m.group("title").split(".")[0].strip()
            p.steps.append({"name": _LETTERS.get(letter, letter), "title": m.group("title"),
                            "status": m.group("status"), "cap": float(m.group(4)),
                            "nominal": float(m.group(5)), "upper": float(m.group(6)),
                            "rule": m.group("rule").strip()})
        elif m := re.match(rf"^\s+total: nominal {_MONEY}, upper {_MONEY}.*global cap "
                           rf"{_MONEY}", line):
            p.nominal, p.upper, p.global_cap = (float(m.group(i)) for i in (1, 2, 3))
        elif m := re.match(r"^\s+lab \(d2\): (.*?) \((.*)\)$", line):
            p.lab_runs = m.group(1).strip() == "runs"
            p.lab_why = m.group(2)
        elif "held" in line and "attempt" in line:
            p.held.append(line.strip())
    nxt = next((s for s in p.steps if s["status"] != "done"), None)
    if nxt is None:
        p.next_action = "done" if p.steps else None
    else:
        p.next_step = nxt["name"]
        rule = nxt["rule"]
        p.next_action = ("run" if rule == "ok" else
                         "wait" if "billing covers" in rule or "held" in rule else "stop")
        p.next_message = f"{nxt['name']}: rule: {rule}"
    return p


def parse_status_text(text: str, p: PlanInfo) -> PlanInfo:
    """``wave_h.py --status`` (no ``--json``): its ``next:`` line is ``decide()``'s answer."""
    for line in text.splitlines():
        s = line.strip()
        if m := re.match(r"^next: (\S+): (run|wait|stop|done): (.*)$", s):
            p.next_step, p.next_action, p.next_message = m.group(1), m.group(2), m.group(3)
            p.source = "status+dry-run"
        elif s.startswith("next: nothing"):
            p.next_step, p.next_action, p.source = None, "done", "status+dry-run"
        elif s.startswith("because "):
            p.reasons.append(s[len("because "):])
        elif "held" in s and s not in p.held:
            p.held.append(s)
    return p


def parse_status_json(doc: Mapping[str, Any]) -> PlanInfo:
    """``wave_h.py --status --plan certified --json`` (expected shape; unknown keys ignored)::

        {"plan": "certified", "embodiment": "human", "spent": 0.0, "global_cap": 10.6,
         "nominal": 6.75, "upper": 10.58,
         "steps": [{"name": "exact_e2e", "status": "pending", "cap": 4.0, "upper": 3.97}, ...],
         "held": [{"step": ..., "attempt": 1, "charged": ..., "clears_at": ...}, ...],
         "next": {"step": "exact_e2e", "action": "run", "message": "...", "reasons": [...]},
         "lab": {"runs": true, "why": "...", "engine_settings_hash": {"tempseg": "..."}}}
    """
    nxt = doc.get("next") or {}
    lab = doc.get("lab") or {}
    steps = [dict(s) for s in doc.get("steps") or ()]
    upper = doc.get("upper", doc.get("upper_total"))
    if upper is None and steps:
        upper = sum(float(s.get("upper") or 0) for s in steps)
    return PlanInfo(
        "status-json", embodiment=doc.get("embodiment"), steps=steps,
        nominal=doc.get("nominal"), upper=None if upper is None else float(upper),
        global_cap=doc.get("global_cap"), spent=doc.get("spent"),
        next_step=nxt.get("step"), next_action=nxt.get("action") or ("done" if not nxt else None),
        next_message=nxt.get("message") or "", reasons=list(nxt.get("reasons") or ()),
        held=[h if isinstance(h, str) else json.dumps(h) for h in doc.get("held") or ()],
        lab_runs=lab.get("runs"), lab_why=lab.get("why") or "",
        lab_hashes=dict(lab.get(HASH_KEY) or {}))


def load_plan(env: Env, embodiment: str = "human") -> tuple[PlanInfo | None, str]:
    """``--status --json`` when the runbook has it; else ``--status`` text over ``--dry-run``,
    all for ``embodiment``'s certified plan."""
    wave, emb = "scripts/wave_h.py", ("--embodiment", embodiment)
    r = env.py(wave, "--status", "--plan", PLAN, "--json", *emb)
    if r.rc == 0:
        try:
            return parse_status_json(json.loads(r.out)), ""
        except ValueError:
            pass  # an older --status without JSON output
    d = env.py(wave, "--plan", PLAN, "--dry-run", *emb)
    if d.rc != 0:
        return None, f"wave_h.py --dry-run exited {d.rc}: {_tail(d)}"
    p = parse_dry_run(d.out)
    s = env.py(wave, "--status", "--plan", PLAN, *emb)
    if s.rc == 0:
        parse_status_text(s.out, p)
    return p, ""


def check_plan(env: Env, plan: PlanInfo | None, why: str, embodiment: str) -> Check:
    name = "plan"
    if plan is None:
        return Check(name, FAIL, f"the runbook could not be read: {why}",
                     "run `uv run python scripts/wave_h.py --plan certified --dry-run` and fix "
                     "what it reports")
    d = asdict(plan)
    caps = ", ".join(f"{s['name']} ${float(s.get('cap') or 0):.2f}" for s in plan.steps)
    held = f"; holding: {' | '.join(plan.held)}" if plan.held else ""
    if plan.embodiment and plan.embodiment != embodiment:
        return Check(name, FAIL, f"the certified plan runs {plan.embodiment}, not {embodiment}",
                     f"run the preflight with --embodiment {plan.embodiment}, or add a "
                     f"{embodiment} certified plan", d)
    if not plan.steps and plan.source == "dry-run":
        return Check(name, FAIL, "the dry run listed no steps (output format changed?)",
                     "update parse_dry_run in scripts/cert_preflight.py, or land --status "
                     "--json", d)
    if plan.upper is not None and plan.global_cap is not None and \
            plan.upper > plan.global_cap + 1e-9:
        return Check(name, FAIL, f"upper ${plan.upper:.2f} exceeds the global cap "
                     f"${plan.global_cap:.2f}", "re-derive the caps (wave_h.py --dry-run shows "
                     "the honest ones) and get the user's approval", d)
    head = (f"next {plan.next_step}: {plan.next_action} [{plan.source}]; caps {caps}; upper "
            f"${plan.upper or 0:.2f} of global cap ${plan.global_cap or 0:.2f}{held}")
    if plan.next_action == "run":
        return Check(name, PASS, head, "", d)
    if plan.next_action == "done":
        return Check(name, FAIL, "every step of the certified plan is already done",
                     "nothing to run; read reports/wave-h/certified/ instead", d)
    if plan.next_action == "wait":
        return Check(name, FAIL, f"{head}: {plan.next_message}",
                     "wait until billing covers the held attempts (the time above), then "
                     "re-run this preflight", d)
    return Check(name, FAIL, f"{head}: {plan.next_message}",
                 "fix what the stop rule names (`wave_h.py --plan certified --status`); never "
                 "raise --global-cap without the user's approval", d)


# ---------------------------------------------------------------------------------------------
# 3. Episodes


def expected_episodes(env: Env, version: str, pool: str) -> tuple[list[str] | None, str]:
    """``pool/dataset/slug`` of the selection's ``pool`` episodes, read by the repo's code."""
    code = ("import json,sys; from flora_deploy.validate.golden import SELECTIONS, "
            "load_selection as L; v=sys.argv[1]; p=sys.argv[2]\n"
            "if v not in SELECTIONS: print(json.dumps(None)); raise SystemExit\n"
            "s=L(SELECTIONS[v]); print(json.dumps([f\"{e['pool']}/{e['dataset']}/{e['slug']}\""
            " for e in s['episodes'] if e['pool']==p]))")
    r = env.py("-c", code, version, pool, timeout=120)
    if r.rc != 0:
        return None, f"selection unreadable: {_tail(r)}"
    eps = json.loads(r.out.strip().splitlines()[-1])
    return (None, f"no {version} selection in validate/golden.py") if eps is None else (eps, "")


def check_episodes(env: Env, embodiment: str) -> Check:
    name = "episodes"
    version, pool = GOLDEN[embodiment]
    fix_stage = f"uv run python scripts/golden_stage.py --version {version}"
    want, why = expected_episodes(env, version, pool)
    if want is None:
        return Check(name, FAIL, why, f"build and commit the {version} selection, then "
                     f"`{fix_stage}`")
    if not want:
        return Check(name, FAIL, f"{version} selects no {pool} episodes",
                     f"add {pool} episodes to the {version} selection")
    try:
        vol = env.vol(GOLDEN_VOLUME)
        manifest = json.loads(b"".join(vol.read_file(f"{version}/manifest.json")))
        listing = {e.path: e.size for e in vol.listdir(f"{version}/{pool}", recursive=True)}
    except FileNotFoundError:
        return Check(name, FAIL, f"no {version} on the {GOLDEN_VOLUME} Volume", fix_stage)
    except Exception as e:  # noqa: BLE001
        return Check(name, FAIL, f"the {GOLDEN_VOLUME} Volume could not be read: {e!r}",
                     "check `modal token` / the network")
    by_id = {e.get("id"): e for e in manifest.get("episodes") or ()}
    problems: list[str] = []
    for gid in want:
        ep = by_id.get(gid)
        if ep is None:
            problems.append(f"{gid}: not in the manifest")
            continue
        if ep.get("status") != "staged":
            problems.append(f"{gid}: status {ep.get('status')}")
            continue
        for role, f in (ep.get("files") or {}).items():
            path = f"{ep['path']}/{f.get('file')}"
            size = listing.get(path)
            if size is None:
                problems.append(f"{gid}: {role} missing ({path})")
            elif size != f.get("bytes"):
                problems.append(f"{gid}: {role} is {size} bytes, manifest {f.get('bytes')}")
        if not ep.get("files"):
            problems.append(f"{gid}: no files")
    d = {"version": version, "expected": len(want), "problems": problems}
    if problems:
        return Check(name, FAIL, f"{len(problems)} problem(s) in {version} {pool}: "
                     + "; ".join(problems[:3]), fix_stage, d)
    return Check(name, PASS, f"all {len(want)} {version} {pool} episodes staged, sizes match",
                 "", d)


# ---------------------------------------------------------------------------------------------
# 4. Weights


def check_weights(env: Env) -> Check:
    name = "weights"
    fix = ("run `uv run flora prepare --show-receipt`; pin its content digest in config/pins.py "
           "DERIVED_PINS (or rebuild with `flora prepare`)")
    r = env.sh(*env.uv, "flora", "prepare", "--show-receipt", "--json", timeout=180)
    try:
        rows = json.loads(r.out) if r.rc == 0 else None
    except ValueError:
        rows = None
    if rows is None:
        return Check(name, FAIL, f"receipts unreadable: {_tail(r)}", "check `modal token` / "
                     "the network, then `uv run flora prepare --show-receipt`")
    bad, unpinned = [], []
    for row in rows:
        k = row.get("key")
        if row.get("error"):
            bad.append(f"{k}: {row['error']}")
        elif row.get("content_sha256_pinned") is None:
            unpinned.append(k)
        elif row.get("matches_pin") is not True:
            bad.append(f"{k}: receipt {str(row.get('content_sha256'))[:12]} != pin "
                       f"{str(row.get('content_sha256_pinned'))[:12]}")
        elif row.get("stale"):
            bad.append(f"{k}: stale ({row['stale']})")
    d = {"rows": rows}
    if bad or unpinned:
        msg = "; ".join(bad + [f"{k}: not pinned" for k in unpinned])
        return Check(name, FAIL, msg, fix, d)
    return Check(name, PASS, f"{len(rows)} derived weight(s) match DERIVED_PINS: "
                 + ", ".join(f"{r['key']} {str(r.get('content_sha256'))[:12]}" for r in rows),
                 "", d)


# ---------------------------------------------------------------------------------------------
# 5. Image locks

def check_images(env: Env) -> Check:
    """``lock_images.py --check``: re-resolves every lock (network, never Modal) and fails if a
    re-lock would change a committed ``.in`` / ``.overrides.txt`` / ``.txt``."""
    name = "images"
    fix = "uv run python scripts/lock_images.py, then commit the locks"
    r = env.py("scripts/lock_images.py", "--check", timeout=600)
    if r.rc == 0:
        return Check(name, PASS, "lock_images.py --check: the locks are current")
    what = "stale" if r.rc == 1 else f"could not re-resolve (exit {r.rc})"
    return Check(name, FAIL, f"lock_images.py --check: {what}: {_tail(r)}",
                 fix if r.rc == 1 else "check the network (PyPI, download.pytorch.org), then "
                 "uv run python scripts/lock_images.py --check")


# ---------------------------------------------------------------------------------------------
# 6. Modal apps


def check_apps(env: Env) -> Check:
    name = "apps"
    r = env.sh(*env.uv, "modal", "app", "list", "--json", timeout=120)
    try:
        apps = json.loads(r.out) if r.rc == 0 else None
    except ValueError:
        apps = None
    if apps is None:
        return Check(name, FAIL, f"modal app list failed: {_tail(r)}", "check `modal token`")
    live = [a for a in apps if str(a.get("description") or a.get("name") or "")
            .startswith("flora") and str(a.get("state", "")).lower() in LIVE_STATES]
    d = {"live": live}
    if live:
        names = ", ".join(f"{a.get('description')} ({a.get('app_id')}, {a.get('state')})"
                          for a in live)
        return Check(name, FAIL, f"flora app(s) still up: {names}",
                     "stop each one you started: `modal app stop <app_id>` (only flora* apps)",
                     d)
    return Check(name, PASS, "no flora* app running (other apps ignored)", "", d)


# ---------------------------------------------------------------------------------------------
# 7. Lab evidence

_CURRENT_HASH = (  # wave_h.lab_engine_hashes: what the runbook's lab reuse compares
    "import json, sys\n"
    "sys.path.insert(0, 'scripts')\n"
    "import wave_h as W\n"
    "print(json.dumps({k: W.lab_engine_hashes(k)['baseline'] for k in ('tempseg','captioner')}))\n"
)


def saved_hash(res: Mapping[str, Any]) -> str | None:
    """The baseline engine-settings hash a saved lab result was measured under."""
    return ((res.get(LAB_HASHES) or {}).get("baseline") or res.get(HASH_KEY)
            or (res.get("baseline") or {}).get(HASH_KEY))


def check_lab(env: Env, plan: PlanInfo | None) -> Check:
    name = "lab"
    if plan is None:
        return Check(name, FAIL, "plan unknown (see the plan check)", "fix the plan check first")
    if plan.lab_runs is None:
        return Check(name, PASS, "the plan does not report a lab step")
    if plan.lab_runs:
        return Check(name, PASS, f"the lab is scheduled to run ({plan.lab_why})")
    current = dict(plan.lab_hashes)
    if not current:
        r = env.py("-c", _CURRENT_HASH, timeout=120)
        try:
            current = json.loads(r.out.strip().splitlines()[-1]) if r.rc == 0 else {}
        except (ValueError, IndexError):
            current = {}
    rerun = f"move reports/wave-h/balanced-<kind>.json aside so the lab re-runs ({PLAN} plan)"
    fails, warns, oks, d = [], [], [], {}
    for kind in LAB_KINDS:
        p = env.repo / "reports" / "wave-h" / f"balanced-{kind}.json"
        try:
            res = json.loads(p.read_text())
        except (OSError, ValueError):
            fails.append(f"{kind}: the runbook skips the lab but {p.name} is unreadable")
            continue
        have, now = saved_hash(res), current.get(kind)
        d[kind] = {"saved": have, "current": now}
        if have is None:
            warns.append(f"{kind}: no engine-settings hash (an older lab file): treated as "
                         "will re-run")
        elif now is None:
            warns.append(f"{kind}: saved {have[:12]} but the current hash is unavailable")
        elif have != now:
            fails.append(f"{kind}: saved {have[:12]} != current {now[:12]} (stale evidence)")
        else:
            oks.append(f"{kind} {have[:12]}")
    if fails:
        return Check(name, FAIL, "; ".join(fails), rerun, d)
    if warns:
        return Check(name, WARN, "; ".join(warns), rerun, d)
    return Check(name, PASS, "saved lab evidence matches the current engine settings: "
                 + ", ".join(oks), "", d)


# ---------------------------------------------------------------------------------------------
# 8. The freeze guard


def check_freeze(env: Env) -> Check:
    name = "freeze"
    r = env.py("scripts/cert_freeze.py", "--check", timeout=300)
    if r.rc != 0:
        return Check(name, FAIL, f"cert_freeze.py --check exited {r.rc}: {_tail(r)}",
                     "fix the drift it reports (docs/certificate-freeze.md), or `cert_freeze.py "
                     "--lock` for missing locks")
    t = env.sh("nice", "-n", "10", *env.uv, "pytest", "-q", "-p", "no:cacheprovider",
               "tests/unit/test_certificate_drift.py", timeout=600, env=TEST_ENV)
    if t.rc != 0:
        return Check(name, FAIL, f"test_certificate_drift.py failed: {_tail(t)}",
                     "OMP_NUM_THREADS=1 uv run pytest tests/unit/test_certificate_drift.py and "
                     "fix it")
    return Check(name, PASS, "cert_freeze.py --check ok; test_certificate_drift.py passes")


# ---------------------------------------------------------------------------------------------
# 9. Budget


def billed_total(env: Env) -> tuple[float | None, str]:
    r = env.py("scripts/spend.py", "--since", BILLING_SINCE, "--budget",
               f"{PROJECT_BUDGET_USD:g}", timeout=120)
    m = re.search(r"^TOTAL\s+\$\s*([0-9.]+)", r.out, re.M) if r.rc == 0 else None
    return (float(m.group(1)), "") if m else (None, _tail(r) or "no TOTAL line")


def check_budget(env: Env, plan: PlanInfo | None, *, budget: float, margin: float) -> Check:
    name = "budget"
    billed, why = billed_total(env)
    if billed is None:
        return Check(name, FAIL, f"billing unreadable: {why}", "check `modal token`, then "
                     "`uv run python scripts/spend.py`")
    upper = None if plan is None else plan.upper
    if upper is None:
        return Check(name, FAIL, f"billed ${billed:.2f}; the plan's upper total is unknown",
                     "fix the plan check first", {"billed": billed})
    left = budget - billed - upper
    d = {"billed": billed, "upper": upper, "budget": budget, "margin": margin,
         "remaining_after_upper": round(left, 2)}
    msg = (f"billed ${billed:.2f} + upper ${upper:.2f} = ${billed + upper:.2f} of "
           f"${budget:.2f}; remaining after the upper case ${left:.2f} (margin ${margin:.2f})")
    if left < margin:
        return Check(name, FAIL, msg, "shrink the plan (fewer episodes / steps) or get the user "
                     "to raise the project budget", d)
    return Check(name, PASS, msg, "", d)


# ---------------------------------------------------------------------------------------------


def preflight(env: Env, *, embodiment: str = "human", budget: float = PROJECT_BUDGET_USD,
              margin: float = MARGIN_USD) -> list[Check]:
    plan, why = load_plan(env, embodiment)
    checks: list[Callable[[], Check]] = [
        lambda: check_git(env),
        lambda: check_plan(env, plan, why, embodiment),
        lambda: check_episodes(env, embodiment),
        lambda: check_weights(env),
        lambda: check_images(env),
        lambda: check_apps(env),
        lambda: check_lab(env, plan),
        lambda: check_freeze(env),
        lambda: check_budget(env, plan, budget=budget, margin=margin),
    ]
    out = []
    for i, c in enumerate(checks):
        try:
            out.append(c())
        except Exception as e:  # noqa: BLE001 - one broken check is a fail, not a crash
            out.append(Check(f"check{i + 1}", FAIL, f"crashed: {e!r}",
                             "fix scripts/cert_preflight.py"))
    return out


def verdict(checks: Sequence[Check]) -> str:
    return "NO-GO" if any(c.status == FAIL for c in checks) else "GO"


def render(checks: Sequence[Check]) -> str:
    lines = []
    for i, c in enumerate(checks, 1):
        lines.append(f"{i}. [{c.status.upper():4s}] {c.name:9s} {c.message}")
        if c.fix and c.status != PASS:
            lines.append(f"   {'':6s} {'':9s} fix: {c.fix}")
    v = verdict(checks)
    bad = [c.name for c in checks if c.status == FAIL]
    lines.append(f"{v}" + (f": failing {', '.join(bad)}" if bad else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None, env: Env | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--embodiment", choices=tuple(GOLDEN), default="human")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--repo", type=Path, default=None, help="checkout to check (default: this)")
    ap.add_argument("--budget", type=float, default=PROJECT_BUDGET_USD)
    ap.add_argument("--margin", type=float, default=MARGIN_USD)
    args = ap.parse_args(argv)  # a usage error exits 2
    if args.repo is not None and not (args.repo / "scripts" / "wave_h.py").is_file():
        ap.error(f"--repo {args.repo}: no scripts/wave_h.py there")
    env = env or Env()
    if args.repo is not None:
        env.repo = args.repo.resolve()
    checks = preflight(env, embodiment=args.embodiment, budget=args.budget, margin=args.margin)
    if args.json:
        print(json.dumps({"verdict": verdict(checks), "embodiment": args.embodiment,
                          "checks": [asdict(c) for c in checks]}, indent=2, default=str))
    else:
        print(render(checks))
    return 1 if verdict(checks) == "NO-GO" else 0


if __name__ == "__main__":
    sys.exit(main())
