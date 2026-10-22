"""``scripts/after_certified.py`` end to end on a fake certified run (``tests/fakes/after_run.py``):
the real ``wave_h_finalize.py`` (and through it the real ``flora certify``), the independent
``cert_check.py``, ``cert_freeze.py --check`` and the doc generators. Nothing here contacts
Modal; the drift tests that read the checkout are replaced by a runner the test controls."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from tests.fakes import after_run as A
from tests.fakes import certcheck_run as R

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import after_certified as ac  # noqa: E402
import cert_check as cc  # noqa: E402

pytestmark = pytest.mark.skipif(R.VERSION != "3", reason="the generator predates rule 3")


def drift_ok(root):
    return 0, "17 passed"


class Pytest:
    """The drift-test runner: records its calls, passes unless told otherwise."""

    def __init__(self, rc: int = 0):
        self.rc, self.calls = rc, []

    def __call__(self, root, tests):
        self.calls.append((root, tuple(tests)))
        return self.rc, "4 passed" if self.rc == 0 else "1 failed, 3 passed"


def run(f: A.AfterFake, *extra: str, pytest_runner=None, **kw) -> int:
    return ac.main(f.args(*extra), drift_runner=drift_ok,
                   pytest_runner=pytest_runner or Pytest(), **kw)


@pytest.fixture
def fake(tmp_path):
    return A.build(tmp_path)


def test_dry_run_writes_nothing_and_summarizes(fake, capsys):
    before = fake.tree()
    py = Pytest()
    assert run(fake, pytest_runner=py) == ac.OK
    out = capsys.readouterr().out
    assert fake.tree() == before and py.calls == []
    assert "DRY RUN (nothing written)" in out
    assert "Verdict: caption certified, segment certified." in out
    assert "| count_mismatch | union | 0.055556 (1 of 18 flips, 2 allowed) | 0.1 | 0.05 | yes |" \
        in out
    assert "- coverage: 1, 12 episodes (11 count-matched), 0 missing; noise run 6" in out
    assert "- witness: 6 episodes (0 unpaired), caches cold:forced x6" in out
    assert "- preemptions: 0 preempted, 0 unknown" in out
    assert "Spend: tracked $4.57, billed $3.10" in out
    assert "cert_check: PASS" in out
    for name in ("README.md", "docs/costs.md", "docs/limitations.md", "docs/wave-h-report.md",
                 "src/flora_deploy/planner/rates.json"):
        assert f"- `{name}`: +" in out
    assert out.count("src/flora_deploy/config/certificates/cert_") == 2
    assert out.count("src/flora_deploy/config/certificates/freeze/cert_") == 2
    for block in ac.README_BLOCKS:
        assert f"`wave-h:{block}`:" in out
    assert "Accuracy certificates: segmentation and captioning for `human` data" in out
    assert "per video-hour" in out  # the cost block in full


def test_write_applies_what_cert_check_passes_and_is_idempotent(fake, capsys):
    py = Pytest()
    assert run(fake, "--write", pytest_runner=py) == ac.OK
    out = capsys.readouterr().out
    assert "APPLIED (not committed)" in out and "drift tests: 4 passed" in out
    assert py.calls == [(fake.root, ac.DRIFT_TESTS)]
    certs = sorted(fake.cert_dir.glob("cert_*.json"))
    assert len(certs) == 2 and len(list((fake.cert_dir / "freeze").glob("*.json"))) == 2
    # the applied tree passes the independent check as it stands
    rows = cc.check_all(cc.Inputs(run_root=fake.run_root, cert_dir=fake.cert_dir,
                                  lab_dir=fake.root / "reports" / "wave-h",
                                  report=fake.root / "docs" / "wave-h-report.md",
                                  nv_root=fake.nv_root, drift_runner=drift_ok))
    assert [r for r in rows if r.status != cc.PASS] == []
    readme = (fake.root / "README.md").read_text()
    assert "Accuracy certificates: segmentation and captioning for `human`" in readme
    applied = fake.tree()
    assert run(fake, "--write") == ac.OK
    again = capsys.readouterr().out
    assert fake.tree() == applied
    assert "Files: nothing to change." in again
    assert run(fake) == ac.OK and fake.tree() == applied


def test_cert_check_failure_writes_nothing(tmp_path, capsys):
    fake = A.build(tmp_path, R.Options(preemptions=1))  # provisional: may not be published
    before = fake.tree()
    py = Pytest()
    assert run(fake, "--write", pytest_runner=py) == ac.FAILED
    out = capsys.readouterr().out
    assert fake.tree() == before and py.calls == []
    assert "FAILED: nothing applied" in out and "Failed at **cert_check.py**" in out
    assert "| FAIL |" in out and "| PASS |" not in out  # only the failing rows
    assert "Verdict: caption provisional, segment provisional." in out
    assert "Files:" not in out


def test_drift_after_apply_restores_every_file(fake, capsys):
    before = fake.tree()
    assert run(fake, "--write", pytest_runner=Pytest(rc=1)) == ac.FAILED
    out = capsys.readouterr().out
    assert fake.tree() == before
    assert "drift tests (restored every file)" in out and "1 failed" in out


def test_a_doc_that_is_not_its_generators_fixed_point_is_refused(fake, capsys, monkeypatch):
    import limitations_status

    real = limitations_status.render
    monkeypatch.setattr(limitations_status, "render",
                        lambda text=None, **kw: real(text, **kw) + "drift\n")
    before = fake.tree()
    assert run(fake, "--write") == ac.FAILED
    out = capsys.readouterr().out
    assert fake.tree() == before
    assert "docs/limitations.md: regenerating it again changes it" in out


def test_freeze_check_failure_is_refused(fake, capsys, monkeypatch):
    monkeypatch.setattr(ac.cert_freeze, "main", lambda argv: print("cert_x: drifted") or 1)
    before = fake.tree()
    assert run(fake, "--write") == ac.FAILED
    out = capsys.readouterr().out
    assert fake.tree() == before and "cert_freeze.py --check" in out and "drifted" in out


@pytest.mark.parametrize("leak, why", [
    (R.CAPTIONS[1], "record string"),  # a caption from the records
    (R.FAKE_NV, "NVIDIA text"),  # the stand-in NVIDIA prompt
])
def test_a_leak_in_a_changed_file_is_refused(fake, capsys, monkeypatch, leak, why):
    real = ac.fin.cost_block
    monkeypatch.setattr(ac.fin, "cost_block", lambda f: real(f) + f"\n{leak}")
    before = fake.tree()
    assert run(fake, "--write") == ac.FAILED
    out = capsys.readouterr().out
    assert fake.tree() == before
    assert "Failed at **leakage**" in out and "README.md: " in out and why in out


def test_missing_records_are_exported_first(fake, capsys):
    gone = fake.records / "witness" / "human" / "caption.jsonl"
    keep = fake.root.parent / "keep.jsonl"
    shutil.copyfile(gone, keep)
    gone.unlink()
    calls = []

    def results(args, check):
        calls.append(args)
        out = Path(args[args.index("--out") + 1])
        src = fake.root.parent / "cc" / "records" / out.relative_to(fake.records)
        shutil.copyfile(keep if out == gone else src, out)

    assert run(fake, fetch_run=results) == ac.OK
    assert "exporting the records (1 missing" in capsys.readouterr().out
    assert calls and all(a[1] == "results" and "--store" in a for a in calls)
    assert gone.read_bytes() == keep.read_bytes()


def test_yam_applies_only_what_its_certificates_change(tmp_path, capsys):
    fake = A.build(tmp_path, embodiment="yam")
    before = fake.tree()
    assert run(fake, "--embodiment", "yam", "--write") == ac.OK
    out = capsys.readouterr().out
    after = fake.tree()
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert {k for k in changed if not k.startswith("src/flora_deploy/config/certificates/")} == {
        "README.md", "docs/limitations.md", "reports/wave-h/finalize.json"}
    assert "for `yam` data" in out
    assert ac.block(after["README.md"].decode(), "cost") == ac.block(
        before["README.md"].decode(), "cost")


def test_usage(tmp_path, capsys):
    assert ac.main(["--embodiment", "dex"]) == ac.USAGE
    assert ac.main(["--run-root", str(tmp_path)]) == ac.USAGE
    assert "no run state" in capsys.readouterr().err


def test_it_never_runs_git():
    text = (ROOT / "scripts" / "after_certified.py").read_text()
    assert '"git"' not in text and "'git'" not in text
