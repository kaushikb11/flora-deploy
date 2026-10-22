"""After the certified run: finalize, check, regenerate and summarize, in one guarded pass.

Usage::

    uv run python scripts/after_certified.py                     # dry run (human)
    uv run python scripts/after_certified.py --embodiment yam    # dry run (yam)
    uv run python scripts/after_certified.py --write             # apply when every check passes

Run it after ``scripts/wave_h.py --plan certified [--embodiment yam] --yes-spend``. It never
starts a container, never commits and never pushes; the coordinator commits after review
(``docs/wave-h-runbook.md``, "After the certified run"). The steps, in order:

1. **Records.** When any of the run's records is missing, it exports them first
   (``wave_h_finalize.fetch_records``, the ``--fetch`` logic: ``flora results``, a Volume
   read). Then it runs ``wave_h_finalize.py --write`` against a **staging copy** of the files
   finalize owns, so the checkout is untouched until every check below has passed.
2. **Independent check.** ``scripts/cert_check.py`` on the staged certificates, report and
   locks, with the run's records and state. A failing row stops here: nothing is applied,
   and the failing rows are printed.
3. **Freeze lock.** ``scripts/cert_freeze.py --check`` on the staged certificates. With
   ``--write``, ``tests/unit/test_certificate_drift.py`` runs on the checkout after the files
   are applied.
4. **Generated docs.** ``docs/costs.md`` (``scripts/cost_table.py``), ``docs/limitations.md``
   (``scripts/limitations_status.py``), the README blocks ``wave-h:certified`` and
   ``wave-h:allow-uncertified`` (``wave_h_finalize.certification_blocks``; ``wave-h:cost`` is
   finalize's) and ``docs/cli.md`` (``scripts/cli_docs.py``) are regenerated, and each must be
   its generator's fixed point. With ``--write``, their drift tests (:data:`DRIFT_TESTS`) run
   on the checkout after the files are applied; a failure restores every file byte for byte
   (so does an error while writing them, and a timeout or interrupt of the drift tests).
5. **Leakage.** Every file it would change is scanned for NVIDIA text (the license shingles,
   against the fetched NVIDIA code) and for record strings (``cert_check``'s scan: captions,
   ids, dataset names, NVIDIA metadata).
6. **Summary.** One block to paste: the verdict and the metrics against their thresholds and
   floors, coverage, the warm/cold witness, preemptions, spend (tracked, billed), and per file
   what changed (a diff stat; the README blocks in full).

``--embodiment yam`` applies only what follows from its certificates: the certificates and
their locks, the README certification blocks and ``docs/limitations.md``. The rates, the
report and the cost docs stay the human run's (``wave_h_finalize.py`` measures 1-view rates;
``scripts/wave_h.py`` never runs it for yam), and the staged yam report is used only for
``cert_check``'s spend rows.

Exit codes: 0 every check passed (and, with ``--write``, the files are applied); 1 a check
failed, nothing applied; 2 usage (bad arguments, no run state).
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cert_check as cc  # noqa: E402
import cert_freeze  # noqa: E402
import wave_h_finalize as fin  # noqa: E402

wh = fin.wh

OK, FAILED, USAGE = 0, 1, 2
EMBODIMENTS = tuple(wh.CERT_SPECS)
DRIFT_TESTS = ("tests/unit/test_certificate_drift.py", "tests/unit/test_cost_docs.py",
               "tests/unit/test_limitations_doc.py", "tests/unit/test_cli_docs.py")
README_BLOCKS = ("certified", "allow-uncertified", "cost")
CLI_DOC = Path("docs") / "cli.md"


class Refused(Exception):
    """A check failed: nothing is applied. ``problems`` are printed as they are."""

    def __init__(self, step: str, problems: Sequence[str]):
        super().__init__(step)
        self.step, self.problems = step, list(problems)


@dataclass
class Change:
    path: Path  # in the checkout
    old: bytes | None
    new: bytes | None  # None: removed

    def rel(self, root: Path) -> str:
        return self.path.relative_to(root).as_posix()

    def stat(self) -> str:
        if self.old is None:
            return f"new, {len(_lines(self.new))} lines"
        if self.new is None:
            return "removed"
        diff = list(difflib.unified_diff(_lines(self.old), _lines(self.new), lineterm="", n=0))
        add = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        rem = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        return f"+{add} -{rem}"


@dataclass
class Outcome:
    st: Any
    certs: dict[str, dict[str, Any]] = field(default_factory=dict)  # stage -> certificate
    rows: list[cc.Row] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    readme: str | None = None
    notes: list[str] = field(default_factory=list)


def _lines(b: bytes | None) -> list[str]:
    return (b or b"").decode("utf-8", errors="replace").splitlines()


def captured(fn: Callable[[list[str]], int], argv: list[str]) -> tuple[int, str]:
    """``fn(argv)`` with its stdout captured; a ``SystemExit`` is its exit code."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = fn(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
            if not isinstance(e.code, int) and e.code:
                print(e.code)
    return int(rc or 0), buf.getvalue()


def run_pytest(root: Path, tests: Sequence[str]) -> tuple[int, str]:
    """The drift tests on the checkout (they read the installed package's files)."""
    if root.resolve() != ROOT.resolve():
        return 0, "skipped: --root is not this checkout"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        *tests], cwd=root, capture_output=True, text=True, timeout=1800)
    lines = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode, "\n".join(lines[-15:])


# ---------------------------------------------------------------------------------------------
# Files


def single_files(p: fin.Paths) -> list[Path]:
    """The files finalize owns, besides the certificate directory."""
    return [p.rates, p.report, p.engines_doc, p.readme, p.manifest, p.costs_doc,
            p.limitations_doc]


def tree(d: Path) -> dict[str, bytes]:
    return {q.relative_to(d).as_posix(): q.read_bytes() for q in sorted(d.rglob("*"))
            if q.is_file()} if d.is_dir() else {}


def stage_copy(real: fin.Paths, staged: fin.Paths) -> None:
    for src, dst in zip(single_files(real), single_files(staged), strict=True):
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    if real.certs.is_dir():
        shutil.copytree(real.certs, staged.certs, dirs_exist_ok=True)
    staged.certs.mkdir(parents=True, exist_ok=True)
    staged.lab.mkdir(parents=True, exist_ok=True)
    for q in sorted(real.lab.glob("balanced-*.json")) if real.lab.is_dir() else ():
        shutil.copyfile(q, staged.lab / q.name)
    cli = real.root / CLI_DOC
    if cli.exists():
        (staged.root / CLI_DOC).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cli, staged.root / CLI_DOC)


def missing_records(p: fin.Paths, st: Any) -> list[Path]:
    """The records ``fetch_records`` would export that are not on disk."""
    out = []
    for side, step, key in [(s, st_, "jobs") for s, st_ in fin.SETS.items()] + [fin.WITNESS]:
        for job, r in sorted(fin.jobs_of(st, step, key).items()):
            if "job_id" in r:
                out += [q for stage in fin.CERT_STAGES
                        if not (q := fin.record_path(p, side, job, stage)).exists()]
    return out


# ---------------------------------------------------------------------------------------------
# Steps


def finalize(staged: fin.Paths, state: Path, records: Path, emb: str) -> str:
    """Step 1: ``wave_h_finalize.py --write`` into the staging copy."""
    rc, out = captured(fin.main, ["--root", str(staged.root), "--state", str(state),
                                  "--records", str(records), "--cert-dir", str(staged.certs),
                                  "--write"])
    ok = rc == 0 or (rc == 3 and emb != wh.EMBODIMENT)  # yam: its rates are never applied
    if not ok:
        what = "refused the rates" if rc == 3 else f"exited {rc}"
        raise Refused("wave_h_finalize.py", [f"wave_h_finalize.py {what}:",
                                             *(f"  {x}" for x in out.strip().splitlines())])
    return out


def check(inp: cc.Inputs) -> list[cc.Row]:
    """Step 2: the independent check."""
    rows = cc.check_all(inp)
    bad = [r for r in rows if r.status != cc.PASS]
    if bad:
        raise Refused("cert_check.py", [cc.render(bad)])
    return rows


def freeze(cert_dir: Path) -> None:
    """Step 3 (staged half): ``cert_freeze.py --check``."""
    rc, out = captured(cert_freeze.main, ["--check", "--cert-dir", str(cert_dir)])
    if rc != 0:
        raise Refused("cert_freeze.py --check", out.strip().splitlines())


def regenerate(real: fin.Paths, staged: fin.Paths, emb: str) -> dict[Path, str]:
    """Step 4: every generated doc, regenerated from the staged certificates (and, for human,
    the staged rates), as text for the checkout path. Each must be its generator's fixed
    point. yam: the README and limitations of the checkout, with the new certificates."""
    import cli_docs
    import cost_table
    import limitations_status

    from flora_deploy.planner.rates import RateTable

    base = staged if emb == wh.EMBODIMENT else real
    table = RateTable.from_dict(json.loads(base.rates.read_text(encoding="utf-8")),
                                source=str(real.rates))
    cert_dirs = [staged.certs]
    gens: dict[Path, tuple[Path, Callable[[str], str]]] = {
        real.readme: (base.readme, lambda t: fin.certification_blocks(t, cert_dirs)),
        real.limitations_doc: (base.limitations_doc, lambda t: limitations_status.render(
            t, cert_dirs=cert_dirs, table=table)),
    }
    if emb == wh.EMBODIMENT:
        gens[real.costs_doc] = (base.costs_doc, lambda t: cost_table.render(
            t, table=replace(table, source="planner/rates.json")))
        gens[real.root / CLI_DOC] = (base.root / CLI_DOC, lambda _t: cli_docs.render())
    out: dict[Path, str] = {}
    drift = []
    for dst, (src, gen) in gens.items():
        if not src.exists():
            continue
        text = gen(src.read_text(encoding="utf-8"))
        if gen(text) != text:
            drift.append(f"{real.rel(dst)}: regenerating it again changes it (not a fixed point)")
        out[dst] = text
    if drift:
        raise Refused("generated docs", drift)
    return out


def planned_changes(real: fin.Paths, staged: fin.Paths, docs: Mapping[Path, str], emb: str
                    ) -> list[Change]:
    """What the checkout would get: (path, old, new) for each file that differs."""
    want: dict[Path, bytes | None] = {}
    if emb == wh.EMBODIMENT:
        for r, s in zip(single_files(real), single_files(staged), strict=True):
            if s.exists():
                want[r] = s.read_bytes()
    else:
        want[real.manifest] = staged.manifest.read_bytes() if staged.manifest.exists() else None
    old_certs, new_certs = tree(real.certs), tree(staged.certs)
    for name in sorted(set(old_certs) | set(new_certs)):
        want[real.certs / name] = new_certs.get(name)
    for p, text in docs.items():
        want[p] = text.encode("utf-8")
    out = []
    for p, new in want.items():
        old = p.read_bytes() if p.exists() else None
        if old != new:
            out.append(Change(p, old, new))
    return sorted(out, key=lambda c: str(c.path))


def leakage(changes: Sequence[Change], inp: cc.Inputs, root: Path) -> None:
    """Step 5: NVIDIA text and record strings in every file that changes."""
    run = cc.load_run(inp)
    forbidden = cc.record_strings(run)
    allowed = set().union(*(cc.our_strings(run, s, c) for s, (_, c) in run.certs.items())) \
        if run.certs else set()
    ref = cc.nv_reference(inp.nv_root)
    hits = []
    if ref is None and changes:
        hits.append("NVIDIA code not fetched (scripts/fetch_nv_code.py): the shingle check "
                    "cannot run")
    for c in changes:
        if c.new is None:
            continue
        text = c.new.decode("utf-8", errors="replace")
        name = c.rel(root)
        if ref is not None and (hit := ref.check(name, text)) is not None:
            hits.append(f"{name}: NVIDIA text (shingle ratio {hit.ratio:.2f})")
        if n := cc._leaks(text, forbidden, allowed):
            hits.append(f"{name}: {n} record string(s) (captions, ids, metadata)")
    if hits:
        raise Refused("leakage", hits)


def _missing_dirs(p: Path) -> list[Path]:
    """``p``'s ancestors that do not exist yet, deepest first."""
    out = []
    while not p.exists():
        out.append(p)
        p = p.parent
    return out


def apply(changes: Sequence[Change]) -> list[Path]:
    """Write ``changes``; returns the directories it created (deepest first), for
    :func:`restore`. A failure part way restores what was written, then re-raises."""
    created: list[Path] = []
    try:
        for c in changes:
            if c.new is None:
                c.path.unlink(missing_ok=True)
            else:
                new_dirs = _missing_dirs(c.path.parent)
                c.path.parent.mkdir(parents=True, exist_ok=True)
                created = new_dirs + created
                c.path.write_bytes(c.new)
    except BaseException:
        restore(changes, created)
        raise
    return created


def restore(changes: Sequence[Change], created: Sequence[Path] = ()) -> None:
    """Put back every file of ``changes`` byte for byte, and remove the directories
    :func:`apply` created when they are empty again."""
    for c in changes:
        if c.old is None:
            c.path.unlink(missing_ok=True)
        else:
            c.path.parent.mkdir(parents=True, exist_ok=True)
            c.path.write_bytes(c.old)
    for d in created:
        with contextlib.suppress(OSError):
            d.rmdir()


# ---------------------------------------------------------------------------------------------
# Summary


def block(text: str, name: str) -> str | None:
    start, end = f"<!-- wave-h:{name} -->", fin.close_marker(name)
    lines = text.splitlines()
    try:
        i = lines.index(start)
        j = lines.index(end, i)
    except ValueError:
        return None
    return "\n".join(lines[i + 1:j])


def _num(x: Any) -> str:
    return "-" if x is None else f"{x:g}" if isinstance(x, float) else str(x)


def cert_lines(stage: str, c: Mapping[str, Any], emb: str) -> list[str]:
    m = c.get("metrics") or {}
    th = (c.get("thresholds") or {}).get("per_metric") or {}
    out = [f"### {stage} on {c.get('gpu')}: {c.get('verdict')} (`{c.get('id')}`)", "",
           "| metric | class | value | threshold | noise floor | pass |",
           "|---|---|---|---|---|---|"]
    for k in c.get("checks") or ():
        t = th.get(k.get("metric")) or {}
        extra = (f" ({k['flips']} of {k['n']} flips, {k['allowed']} allowed)"
                 if "flips" in k else "")
        out.append(f"| {k.get('metric')} | {k.get('class')} | {_num(k.get('value'))}{extra} | "
                   f"{_num(k.get('threshold'))} | {_num(t.get('floor'))} | "
                   f"{'yes' if k.get('pass') else 'NO'} |")
    cand = (m.get("candidate") or {}).get(emb) or {}
    nz, wit = m.get("noise") or {}, m.get("witness") or {}
    cache = c.get("compile_cache") or {}
    pre = c.get("preemptions") or {}
    caches = lambda d: ", ".join(f"{k} x{v}" for k, v in sorted((d or {}).items())) or "none"  # noqa: E731
    out += ["",
            f"- coverage: {_num(m.get('coverage'))}, {cand.get('n', '-')} episodes "
            f"({cand.get('count_matched', '-')} count-matched), {m.get('missing', '-')} "
            f"missing; noise run {nz.get('n', '-')} episodes ({nz.get('unpaired', '-')} "
            "unpaired)",
            f"- witness: {wit.get('n', '-')} episodes ({wit.get('unpaired', '-')} unpaired), "
            f"caches {caches(cache.get('witness'))}; candidate caches "
            f"{caches(cache.get('candidate'))}",
            f"- preemptions: {pre.get('preempted', '-')} preempted, {pre.get('unknown', '-')} "
            "unknown"]
    if c.get("reasons"):
        out.append(f"- reasons: {', '.join(map(str, c['reasons']))}")
    return out


def summary(o: Outcome, *, emb: str, root: Path, write: bool, error: Refused | None = None
            ) -> str:
    st = o.st
    tracked = round(sum(s.tracked_usd for s in st.steps.values()), 2)
    head = ("APPLIED (not committed)" if write else "DRY RUN (nothing written)") if error is None \
        else "FAILED: nothing applied"
    out = [f"## After the certified run: `{emb}`, run `{st.run_id}`: {head}", ""]
    if error is not None:
        out += [f"Failed at **{error.step}**:", "", "```", *error.problems, "```", ""]
    if o.certs:
        verdicts = ", ".join(f"{s} {c.get('verdict')}" for s, c in sorted(o.certs.items()))
        out += [f"Verdict: {verdicts}.", ""]
        for stage, c in sorted(o.certs.items()):
            out += cert_lines(stage, c, emb) + [""]
    if o.rows and error is None:
        out += [f"cert_check: PASS ({len(o.rows)} rows); freeze lock and generated docs "
                "consistent; leakage scan clean.", ""]
    out += [f"Spend: tracked {fin.usd(tracked)}, billed {fin.usd(st.billed_total)} "
            f"(the runbook's figure {fin.usd(st.spent)} of the {fin.usd(fin.global_cap(st))} "
            "global cap).", ""]
    if error is None:
        out += ["Files:" if o.changes else "Files: nothing to change.", ""]
        out += [f"- `{c.rel(root)}`: {c.stat()}" for c in o.changes]
        if o.readme is not None:
            out += ["", "README blocks:"]
            for name in README_BLOCKS:
                out += ["", f"`wave-h:{name}`:", "", "```",
                        block(o.readme, name) or "(missing)", "```"]
    if o.notes:
        out += [""] + [f"- {n}" for n in o.notes]
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------------------------
# Driver


def decided_certs(inp: cc.Inputs) -> dict[str, dict[str, Any]]:
    """stage -> the certificate ``cert_check`` decides on (the newest for the embodiment)."""
    return {s: c for s, (_, c) in cc.load_run(inp).certs.items()}


def main(argv: list[str] | None = None, *, fetch_run: Callable[..., Any] = subprocess.run,
         pytest_runner: Callable[[Path, Sequence[str]], tuple[int, str]] = run_pytest,
         drift_runner: Callable[[Path], tuple[int, str]] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--embodiment", choices=EMBODIMENTS, default=wh.EMBODIMENT)
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    ap.add_argument("--root", type=Path, default=ROOT, help="repository root")
    ap.add_argument("--run-root", type=Path,
                    help="the certified run's directory (default: the embodiment's)")
    ap.add_argument("--nv-root", type=Path, help="fetched NVIDIA code (default: the cache)")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return USAGE if e.code else OK
    emb = a.embodiment
    spec = wh.spec_for(wh.CERTIFIED, emb)
    run_root = a.run_root or spec.state.parent
    state, records = run_root / "state.json", run_root / "records"
    if not state.exists():
        print(f"usage: no run state at {state} (run scripts/wave_h.py --plan certified "
              f"--embodiment {emb} first)", file=sys.stderr)
        return USAGE
    st = wh.State.from_json(json.loads(state.read_text(encoding="utf-8")))
    if fin.run_plan(st) != wh.CERTIFIED:
        print(f"usage: {state} was run under plan {fin.run_plan(st)!r}, not certified",
              file=sys.stderr)
        return USAGE
    real = fin.Paths(root=a.root, state=state, records=records)
    o = Outcome(st=st)
    # 1. the records (a Volume read only when some are missing), then finalize into staging
    if missing := missing_records(real, st):
        print(f"exporting the records ({len(missing)} missing, e.g. {real.rel(missing[0])})")
        fin.fetch_records(real, st, run=fetch_run)
    with tempfile.TemporaryDirectory(prefix="after-certified-") as tmp:
        staged = fin.Paths(root=Path(tmp), state=state, records=records)
        stage_copy(real, staged)
        inp = cc.Inputs(run_root=run_root, records=records, state=state,
                        cert_dir=staged.certs, lab_dir=staged.lab, report=staged.report,
                        job=wh.JOB, embodiment=emb, nv_root=a.nv_root, drift_runner=drift_runner)
        try:
            finalize(staged, state, records, emb)
            o.certs = decided_certs(inp)
            o.rows = check(inp)  # 2.
            freeze(staged.certs)  # 3. (staged)
            docs = regenerate(real, staged, emb)  # 4.
            o.changes = planned_changes(real, staged, docs, emb)
            leakage(o.changes, inp, real.root)  # 5.
        except Refused as e:
            print(summary(o, emb=emb, root=real.root, write=a.write, error=e))
            return FAILED
        except Exception as e:  # noqa: BLE001 - staged only: the checkout is untouched
            err = Refused("an unexpected error (staging only)", [f"{type(e).__name__}: {e}"])
            print(summary(o, emb=emb, root=real.root, write=a.write, error=err))
            return FAILED
        readme = next((c.new for c in o.changes if c.path == real.readme), None)
        o.readme = (readme.decode("utf-8") if readme is not None
                    else real.readme.read_text(encoding="utf-8") if real.readme.exists()
                    else None)
        if not a.write:
            o.notes.append("the drift tests run on the checkout with --write")
            print(summary(o, emb=emb, root=real.root, write=False))
            return OK
        try:
            created = apply(o.changes)
        except Exception as e:  # apply restored what it wrote
            err = Refused("apply (restored every file)", [f"{type(e).__name__}: {e}"])
            print(summary(o, emb=emb, root=real.root, write=True, error=err))
            return FAILED
        try:
            rc, tail = pytest_runner(real.root, DRIFT_TESTS)  # 3. and 4. (the checkout)
        except BaseException as e:  # a timeout, an interrupt: the checkout goes back too
            restore(o.changes, created)
            if not isinstance(e, Exception):
                raise
            rc, tail = 1, f"{type(e).__name__}: {e}"
        if rc != 0:
            restore(o.changes, created)
            err = Refused("drift tests (restored every file)", [*DRIFT_TESTS, tail])
            print(summary(o, emb=emb, root=real.root, write=True, error=err))
            return FAILED
        o.notes.append(f"drift tests: {tail.splitlines()[-1] if tail else 'passed'}")
        print(summary(o, emb=emb, root=real.root, write=True))
        return OK


if __name__ == "__main__":
    sys.exit(main())
