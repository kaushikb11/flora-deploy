"""A rehearsal of ``scripts/after_certified.py`` on a synthetic certified run
(``tests/fakes/after_run.py``): what a developer reads after the flip (README blocks and
``docs/limitations.md``), the rollback when any stage fails, the leak scan on planted NVIDIA text
and a private episode id, and that nothing is written outside the fake repository and the
system temp directory. No Modal; the drift tests are replaced by a runner the test controls."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.fakes import after_run as A
from tests.fakes import certcheck_run as R

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import after_certified as ac  # noqa: E402
import limitations_status  # noqa: E402


@pytest.fixture(autouse=True)
def no_volume(monkeypatch):
    """No test here reaches Modal: ``flora results`` (``fetch_records``' default runner) and
    the audit reads are refused unless a test passes its own runner."""
    real = ac.fin.fetch_records

    def guarded(paths, st, run=None, **kw):
        if run is None or run is ac.subprocess.run:
            pytest.fail("fetch_records would read the Volume (flora results)")
        return real(paths, st, run=run, **kw)  # a stand-in runner: no audits are read

    monkeypatch.setattr(ac.fin, "fetch_records", guarded)

EMBODIMENTS = ("human", "yam", "yam32", "dex", "tianji")


def drift_ok(root):
    return 0, "17 passed"


def passing(root, tests):
    return 0, "4 passed"


def run(f: A.AfterFake, *extra: str, pytest_runner=passing, root: Path | None = None,
        run_root: Path | None = None) -> int:
    argv = ["--root", str(root or f.root), "--run-root", str(run_root or f.run_root),
            "--nv-root", str(f.nv_root), *extra]
    return ac.main(argv, drift_runner=drift_ok, pytest_runner=pytest_runner)


def snapshot(f: A.AfterFake) -> tuple[dict[str, bytes], set[str]]:
    """Every file (bytes) and every directory of the fake repository but the run's inputs."""
    dirs = {str(p.relative_to(f.root)) for p in f.root.rglob("*") if p.is_dir()
            and not {"certified", "certified-yam", "work"} & set(p.relative_to(f.root).parts)}
    return f.tree(), dirs


def limitations_rows(text: str) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        if m := re.match(r"\| `(\w+)` \| (.*) \|$", line):
            rows[m[1]] = [c.strip() for c in m[2].split("|")]
    return rows


@pytest.fixture
def human(tmp_path):
    return A.build(tmp_path)


# ---------------------------------------------------------------------------------------------
# What a developer reads after the flip


def test_the_human_flip_reads_cleanly(human, capsys):
    before = human.tree()
    assert run(human, "--write") == ac.OK
    capsys.readouterr()
    after = human.tree()
    readme, lim = after["README.md"].decode(), after["docs/limitations.md"].decode()

    assert ac.block(readme, "certified") == (
        "Accuracy certificates: segmentation and captioning for `human` data (`L40S` / `H100!`).\n"
        "Rewards run NVIDIA's own engine, as `exact` does, and need none. Everything else runs "
        "uncertified.")
    allow = ac.block(readme, "allow-uncertified")
    assert allow.startswith("`allow_uncertified: true` runs the stages that have no accuracy "
                            "certificate for your embodiment and GPU;")
    assert "Certified today: segmentation and captioning for `human` data" in allow
    assert "does not need it; `flora doctor` tells you." in allow
    assert ac.block(readme, "cost") != ac.block(before["README.md"].decode(), "cost")
    assert "per video-hour" in ac.block(readme, "cost")

    rows = limitations_rows(lim)
    assert list(rows) == list(EMBODIMENTS)
    assert rows["human"] == ["**certified**", "**certified**", "reference", "reference",
                             "reference"]
    for emb in EMBODIMENTS[1:]:  # yam is still to come; the others have no certification set
        assert rows[emb][:2] == ["not yet", "not yet"], emb
    assert ("Certified today: `segment` and `caption` for `human`.\nEvery other `balanced` "
            "segmentation or captioning stage needs `allow_uncertified: true`,\nand its records "
            "say `certified: false`.") in lim

    # nothing stale, no ids, no local paths, short lines in what changed
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    for name in ("README.md", "docs/limitations.md"):
        text = after[name].decode()
        for stale in ("Accuracy certificates: none yet", "None exist yet", "Nothing is certified",
                      "comes first", "shows as \"partly certified\"", "needed while"):
            assert stale not in text, (name, stale)
        new = "\n".join(line for line in text.splitlines()
                        if line not in before[name].decode().splitlines())
        assert "cert_" not in new and str(human.root.parent) not in new and "/Users/" not in new
        assert max(map(len, new.splitlines())) <= 130, name
    assert {k for k in changed if not k.startswith(str(A.CERT_REL))} == {
        "README.md", "docs/balanced-engines.md", "docs/costs.md", "docs/limitations.md",
        "docs/wave-h-report.md", "reports/wave-h/finalize.json",
        "src/flora_deploy/planner/rates.json"}
    # the regenerated docs are their generators' fixed points
    assert limitations_status.render(lim, cert_dirs=[human.cert_dir]) == lim
    assert ac.fin.certification_blocks(readme, [human.cert_dir]) == readme


def test_human_then_yam_lists_both(human, tmp_path, capsys):
    assert run(human, "--write") == ac.OK
    readme_human = (human.root / "README.md").read_text()
    yam = A.build(tmp_path / "yam", embodiment="yam")
    yrun = human.root / "reports" / "wave-h" / "certified-yam"
    shutil.copytree(yam.run_root, yrun)
    assert run(human, "--embodiment", "yam", "--write", run_root=yrun) == ac.OK
    capsys.readouterr()
    readme = (human.root / "README.md").read_text()
    lim = (human.root / "docs" / "limitations.md").read_text()
    assert "segmentation and captioning for `human` and `yam` data (`L40S` / `H100!`)" in \
        ac.block(readme, "certified")
    assert ac.block(readme, "cost") == ac.block(readme_human, "cost")  # yam never prices
    rows = limitations_rows(lim)
    assert rows["human"][:2] == rows["yam"][:2] == ["**certified**", "**certified**"]
    assert rows["dex"][:2] == ["not yet", "not yet"]
    assert "Certified today: `segment` and `caption` for `human` and `yam`." in lim
    assert len(list(human.cert_dir.glob("cert_*.json"))) == 4


# ---------------------------------------------------------------------------------------------
# A partly certified run: segment certified, caption fail (the real human run, 2026-10-22)


def _partial(tmp_path, *, record_fail: bool = True, plan: str = "certified") -> A.AfterFake:
    """The caption witness's (J2w) captions all differ, so caption fails and only segment
    certifies. The runbook records each stage's verdict in the certify step
    (``wave_h.run_flora_certify``). ``plan``: the certified or the recertify layout."""
    f = A.build(tmp_path, R.Options(iso_witness_drift=True), plan=plan)
    if record_fail:
        st = json.loads(f.state.read_text(encoding="utf-8"))
        st["steps"]["certify"]["status"] = "failed"
        st["steps"]["certify"]["result"] = {"ok": False, "stages": {
            "segment": {"verdict": "certified", "reasons": [], "refused": None},
            "caption": {"verdict": "fail", "reasons": ["divergence_above_noise_floor"],
                        "refused": None}}}
        f.state.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return f


@pytest.mark.parametrize("plan", ["certified", "recertify"])
def test_a_partial_run_flips_only_segment(tmp_path, capsys, plan):
    """Outcome (b) of the v4 pre-registration: segment certified, caption fail. Only the
    segment certificate (and its lock) lands; caption stays uncertified."""
    f = _partial(tmp_path, plan=plan)
    extra = ("--plan", plan)
    assert run(f, *extra) == ac.OK  # the dry run
    out = capsys.readouterr().out
    assert ("Verdict: segment certified, caption fail (divergence_above_noise_floor; no "
            "certificate, it stays uncertified).") in out
    assert run(f, *extra, "--write") == ac.OK
    capsys.readouterr()
    readme = (f.root / "README.md").read_text()
    lim = (f.root / "docs" / "limitations.md").read_text()
    certified = ac.block(readme, "certified")
    assert certified.startswith("Accuracy certificates: segmentation for `human` data (`L40S`).")
    assert "captioning for" not in certified
    assert "Certified today: segmentation for `human` data (`L40S`)." in \
        ac.block(readme, "allow-uncertified")
    rows = limitations_rows(lim)
    assert rows["human"][:2] == ["**certified**", "not yet"]
    assert "Certified today: `segment` for `human`.\n" in lim
    certs = [json.loads(p.read_text()) for p in f.cert_dir.glob("cert_*.json")]
    assert [(c["stage"], c["verdict"]) for c in certs] == [("segment", "certified")]
    locks = list((f.cert_dir / "freeze").glob("*.json"))
    assert [q.stem for q in locks] == [certs[0]["id"]]


def test_a_recertify_run_certifies_both_stages_from_its_own_layout(tmp_path, capsys):
    """Outcome (a), the recertify layout: its state on the certified run's id, f2 as the noise
    run, (b)'s oracle and old noise run from the certified state (``wave_h.certify_view``),
    the lab JSON beside its state; no record is missing, so nothing is exported."""
    f = A.build(tmp_path, plan="recertify")
    assert f.run_root.name == "recertified"
    assert run(f, "--plan", "recertify") == ac.OK
    out = capsys.readouterr().out
    assert "Verdict: caption certified, segment certified." in out
    assert "exporting the records" not in out and "cert_check: PASS" in out
    assert run(f, "--plan", "recertify", "--write") == ac.OK
    capsys.readouterr()
    certs = sorted((json.loads(p.read_text()) for p in f.cert_dir.glob("cert_*.json")),
                   key=lambda c: c["stage"])
    assert [(c["stage"], c["verdict"]) for c in certs] == [("caption", "certified"),
                                                           ("segment", "certified")]
    assert certs[0]["scoring"] == {"caption": "stage_isolated"}
    assert ac.block((f.root / "README.md").read_text(), "certified").startswith(
        "Accuracy certificates: segmentation and captioning for `human` data")
    # a recertify state is not read as a certified run, nor the other way round
    assert run(f) == ac.USAGE
    assert "run under plan 'recertify', not certified" in capsys.readouterr().err


def test_a_missing_certificate_without_the_runs_fail_is_still_refused(tmp_path, capsys):
    """Only the run's own recorded ``fail`` excuses a stage's missing certificate."""
    f = _partial(tmp_path, record_fail=False)
    before = f.tree()
    assert run(f, "--write") == ac.FAILED
    out = capsys.readouterr().out
    assert "no balanced caption certificate for human" in out
    assert f.tree() == before


# ---------------------------------------------------------------------------------------------
# Rollback: a forced failure at each stage leaves the checkout as it was


def _fail_after(monkeypatch, name: str, exc: BaseException):
    real = getattr(ac, name)

    def step(*a, **kw):
        real(*a, **kw)
        raise exc

    monkeypatch.setattr(ac, name, step)


@pytest.mark.parametrize("stage", ["finalize", "check", "freeze", "regenerate",
                                   "planned_changes", "leakage"])
@pytest.mark.parametrize("exc", [ac.Refused("forced", ["forced failure"]),
                                 RuntimeError("forced crash")], ids=["refused", "crash"])
def test_a_failure_before_apply_changes_nothing(human, capsys, monkeypatch, stage, exc):
    before = snapshot(human)
    calls = []
    _fail_after(monkeypatch, stage, exc)
    assert run(human, "--write", pytest_runner=lambda r, t: calls.append(t) or (0, "")) == \
        ac.FAILED
    out = capsys.readouterr().out
    assert snapshot(human) == before and calls == []
    assert "FAILED: nothing applied" in out and "forced" in out


def test_a_write_error_part_way_restores_every_file(human, capsys, monkeypatch):
    before = snapshot(human)
    real, written, failed = Path.write_bytes, [], []

    def flaky(self, data):  # the last file to apply fails once, after the freeze locks
        if self.is_relative_to(human.root) and self.name == "rates.json" and not failed:
            failed.append(self)
            raise OSError(28, "No space left on device")
        if self.is_relative_to(human.root):
            written.append(self.relative_to(human.root).as_posix())
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)
    assert run(human, "--write") == ac.FAILED
    out = capsys.readouterr().out
    assert failed and "README.md" in written  # some files were written before the error
    assert any("/freeze/" in w for w in written)  # in a directory apply created
    assert snapshot(human) == before  # files and directories
    assert "apply (restored every file)" in out and "No space left" in out


@pytest.mark.parametrize("outcome", ["rc", "timeout", "oserror"])
def test_a_drift_failure_restores_every_file(human, capsys, outcome):
    before = snapshot(human)

    def drift(root, tests):
        assert (human.root / "README.md").read_bytes() != before[0]["README.md"]  # applied
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("pytest", 1800)
        if outcome == "oserror":
            raise OSError("pytest went away")
        return 1, "1 failed, 3 passed"

    assert run(human, "--write", pytest_runner=drift) == ac.FAILED
    out = capsys.readouterr().out
    assert snapshot(human) == before
    assert "drift tests (restored every file)" in out


def test_an_interrupt_during_the_drift_tests_restores_and_propagates(human, capsys):
    before = snapshot(human)

    def drift(root, tests):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(human, "--write", pytest_runner=drift)
    assert snapshot(human) == before


# ---------------------------------------------------------------------------------------------
# Leak scan: planted NVIDIA text and a private episode id are refused


@pytest.mark.parametrize("leak, why", [
    (R.FAKE_NV, "NVIDIA text"),  # the stand-in NVIDIA prompt
    (R.REST_IDS[2], "record string"),  # a private episode id
    (R.uid(R.NOISE_IDS[0]), "record string"),  # its uid
])
@pytest.mark.parametrize("where", ["limitations", "certificate"])
def test_a_planted_leak_is_refused(human, capsys, monkeypatch, leak, why, where):
    if where == "limitations":
        real = limitations_status.block_certification
        monkeypatch.setattr(limitations_status, "block_certification",
                            lambda d: real(d) + f"\nSee {leak} for an example.\n")
        name = "docs/limitations.md"
    else:  # one more file the run would add to the certificate directory
        real_leakage = ac.leakage
        name = f"{A.CERT_REL.as_posix()}/notes.json"
        extra = ac.Change(human.root / name, None, f'{{"note": "{leak}"}}\n'.encode())
        monkeypatch.setattr(ac, "leakage", lambda changes, inp, root: real_leakage(
            [*changes, extra], inp, root))
    before = snapshot(human)
    assert run(human, "--write") == ac.FAILED
    out = capsys.readouterr().out
    assert snapshot(human) == before
    assert "Failed at **leakage**" in out and f"{name}: " in out and why in out
    assert leak not in out  # the report names the file, never the leaked text


# Episode ids made only of hex digits (one only digits): the hash allowance must not let them
# through, and the run's own hashes (sha256 of the uids, cert keys) must not trip the scan.
HEX_IDS = (*(f"{i:x}" * 16 for i in range(10, 16)), *(f"{i}{i}9f3c0a7e4b2d1" for i in range(5)),
           "202610229876")
HEX_RUN = R.Options(episodes=HEX_IDS, noise=HEX_IDS[:6])


def test_a_hex_id_run_flips_cleanly(tmp_path, capsys):
    f = A.build(tmp_path, HEX_RUN)
    assert all(ac.cc._SAFE.match(e) for e in HEX_IDS)
    assert run(f, "--write") == ac.OK, capsys.readouterr().out
    assert "leakage scan clean" in capsys.readouterr().out


@pytest.mark.parametrize("leak", [HEX_IDS[7], HEX_IDS[-1], R.uid(HEX_IDS[0])])
@pytest.mark.parametrize("where", ["limitations", "certificate"])
def test_a_planted_hex_id_is_refused(tmp_path, capsys, monkeypatch, leak, where):
    f = A.build(tmp_path, HEX_RUN)
    if where == "limitations":
        real = limitations_status.block_certification
        monkeypatch.setattr(limitations_status, "block_certification",
                            lambda d: real(d) + f"\nSlowest episode: {leak}.\n")
        name = "docs/limitations.md"
    else:
        real_leakage = ac.leakage
        name = f"{A.CERT_REL.as_posix()}/notes.json"
        extra = ac.Change(f.root / name, None, f'{{"slowest": "{leak}"}}\n'.encode())
        monkeypatch.setattr(ac, "leakage", lambda changes, inp, root: real_leakage(
            [*changes, extra], inp, root))
    before = snapshot(f)
    assert run(f, "--write") == ac.FAILED
    out = capsys.readouterr().out
    assert snapshot(f) == before
    assert "Failed at **leakage**" in out and f"{name}: " in out and "record string" in out
    assert leak not in out


# ---------------------------------------------------------------------------------------------
# Nothing is written outside the fake repository and the temp directory


class Writes:
    """Paths written, created, renamed or removed while ``on`` (a process-wide audit hook;
    hooks cannot be removed, so it stays installed and idle)."""

    on = False
    seen: list[tuple[str, str]] = []
    installed = False
    EVENTS = {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.replace", "os.truncate",
              "os.symlink", "os.link", "shutil.copyfile", "shutil.copytree", "shutil.move",
              "shutil.rmtree", "os.chmod"}

    @classmethod
    def hook(cls, event: str, args: tuple) -> None:
        if not cls.on:
            return
        if event == "open":
            path, mode, flags = args
            write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            paths = [path] if write and isinstance(path, (str, bytes, os.PathLike)) else []
        elif event in cls.EVENTS:
            if any(isinstance(a, int) and a >= 0 for a in args[1:]):  # dir_fd: rmtree internals
                return
            paths = [a for a in args if isinstance(a, (str, bytes, os.PathLike))]
            if event in ("shutil.copyfile", "shutil.copytree", "shutil.move"):
                paths = paths[1:2]  # the destination
        else:
            return
        cls.on = False
        try:
            cls.seen += [(event, str(Path(os.fsdecode(p)).resolve())) for p in paths]
        finally:
            cls.on = True

    @classmethod
    def watch(cls) -> None:
        if not cls.installed:
            sys.addaudithook(cls.hook)
            cls.installed = True
        cls.seen = []
        cls.on = True


@pytest.mark.parametrize("write", [False, True], ids=["dry-run", "write"])
def test_nothing_is_written_outside_the_repository_or_temp(tmp_path, capsys, write):
    f = A.build(tmp_path)
    inputs = {str(p.relative_to(f.run_root)): p.read_bytes()
              for p in sorted(f.run_root.rglob("*")) if p.is_file()}
    allowed = (tmp_path.resolve(), Path(tempfile.gettempdir()).resolve())
    Writes.watch()
    try:
        assert run(f, *(["--write"] if write else [])) == ac.OK
    finally:
        Writes.on = False
    capsys.readouterr()
    outside = sorted({p for _, p in Writes.seen
                      if not any(Path(p).is_relative_to(a) for a in allowed)})
    assert outside == []
    assert Writes.seen  # the hook saw the staging writes
    run_root = str(f.run_root.resolve())
    assert [p for _, p in Writes.seen if p.startswith(run_root + os.sep)] == []
    assert {str(p.relative_to(f.run_root)): p.read_bytes() for p in sorted(
        f.run_root.rglob("*")) if p.is_file()} == inputs  # the run's inputs are read only
