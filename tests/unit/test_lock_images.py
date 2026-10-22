"""scripts/lock_images.py: ``--check`` re-resolves in a scratch copy and never writes the repo.

The compile (``uvx uv pip compile``, the only network step) is replaced by a fake that acts on
the files in the directory it runs in, like uv does.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flora_deploy.modal_app import images

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lock_images.py"
spec = importlib.util.spec_from_file_location("lock_images", SCRIPT)
L = importlib.util.module_from_spec(spec)
sys.modules["lock_images"] = L
spec.loader.exec_module(L)

REL = Path("src/flora_deploy/modal_app/locks")


class FakeUv:
    """Stands in for ``uv pip compile``: records each call and, by default, leaves the output
    file as it found it (a re-resolve that keeps every committed version)."""

    def __init__(self, edit=None, fail=False):
        self.edit, self.fail = edit, fail
        self.calls: list[tuple[list[str], Path, str | None]] = []

    def __call__(self, cmd, cwd):
        out = Path(cwd) / cmd[cmd.index("--output-file") + 1]
        self.calls.append((cmd, Path(cwd), out.read_text() if out.exists() else None))
        if self.fail:
            raise subprocess.CalledProcessError(2, cmd)
        assert (Path(cwd) / cmd[4]).exists(), "the .in is written before the compile"
        if self.edit:
            out.write_text(self.edit(out.read_text() if out.exists() else ""))


def _digest(root: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / REL).iterdir())}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo root holding a copy of the committed lock directory; the script points at it."""
    shutil.copytree(images.LOCK_DIR, tmp_path / REL)
    (tmp_path / ".python-version").write_text("3.12\n")
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(L, "LOCK_DIR", tmp_path / REL)
    return tmp_path


def _fake(monkeypatch, **kw) -> FakeUv:
    fake = FakeUv(**kw)
    monkeypatch.setattr(L, "_run", fake)
    return fake


def test_check_current_passes_and_writes_nothing(repo, monkeypatch, capsys):
    before = _digest(repo)
    fake = _fake(monkeypatch)
    assert L.main(["--check"]) == 0
    assert "locks current: img_cpu, img_hfref, img_vllm" in capsys.readouterr().out
    assert _digest(repo) == before
    assert len(fake.calls) == 3
    for (cmd, cwd, seeded), name in zip(fake.calls, L.SHORT.values(), strict=True):
        assert cwd != repo and not cwd.exists()  # a scratch root, removed afterwards
        assert cmd == L.compile_cmd(name, upgrade=False) and "--upgrade" not in cmd
        # seeded with the committed lock, so uv keeps its versions as the default mode does
        assert seeded == (repo / REL / images.LOCK_FILES[name]).read_text()


def test_committed_inputs_are_what_the_script_writes():
    """The real lock directory's inputs match what the check compares against."""
    for name in L.SHORT.values():
        src, ovr, _ = L._files(name)
        assert (images.LOCK_DIR / src).read_text() == L.input_text(name)
        want = L.overrides_text(name)
        assert L._read(images.LOCK_DIR / ovr) == want


def test_check_uses_the_default_mode_command_and_relative_paths(repo, monkeypatch):
    fake = _fake(monkeypatch)
    L.compile_lock("img_hfref", upgrade=False)
    L.check_lock("img_hfref")
    (write_cmd, write_cwd, _), (check_cmd, _, _) = fake.calls
    assert write_cwd == repo and write_cmd == check_cmd
    assert check_cmd[4] == str(REL / "hfref.in")
    assert check_cmd[check_cmd.index("--overrides") + 1] == str(REL / "hfref.overrides.txt")
    assert not any(a.startswith("/") for a in check_cmd)  # stable ``# via`` annotations


def test_check_fails_when_resolution_changes_the_lock(repo, monkeypatch, capsys):
    before = _digest(repo)
    numpy = f"numpy=={images.RUNTIME['numpy']}"
    _fake(monkeypatch, edit=lambda t: t.replace(numpy, "numpy==9.9.9"))
    assert L.main(["--check", "cpu"]) == 1
    cap = capsys.readouterr()
    assert f"{REL}/cpu.txt: would change" in cap.out
    assert f"-{numpy}" in cap.out and "+numpy==9.9.9" in cap.out
    assert "stale lock(s): img_cpu" in cap.err
    assert _digest(repo) == before


def test_check_is_byte_exact(repo, monkeypatch, capsys):
    """Byte-exact: a re-lock that only rewrites a comment would still change the file."""
    _fake(monkeypatch, edit=lambda t: t + "    # via extra\n")
    assert L.main(["--check", "vllm"]) == 1
    assert "vllm.txt: would change" in capsys.readouterr().out


def test_check_fails_on_stale_input(repo, monkeypatch, capsys):
    before = _digest(repo)
    monkeypatch.setitem(L.PACKAGES, "img_cpu", (*L.PACKAGES["img_cpu"], "six==1.16.0"))
    fake = _fake(monkeypatch)
    assert L.main(["--check", "cpu"]) == 1
    out = capsys.readouterr().out
    assert f"{REL}/cpu.in: would change" in out and "+six==1.16.0" in out
    assert _digest(repo) == before
    cmd, cwd, _ = fake.calls[0]
    assert cmd == L.compile_cmd("img_cpu", upgrade=False)


def test_check_overrides_missing_and_obsolete(repo, monkeypatch, capsys):
    _fake(monkeypatch)
    (repo / REL / "hfref.overrides.txt").unlink()
    assert L.main(["--check", "hfref"]) == 1
    assert "hfref.overrides.txt: missing" in capsys.readouterr().out
    assert not (repo / REL / "hfref.overrides.txt").exists()

    shutil.copy2(images.LOCK_DIR / "hfref.overrides.txt", repo / REL / "hfref.overrides.txt")
    monkeypatch.setitem(L.OVERRIDES, "img_hfref", ())
    assert L.main(["--check", "hfref"]) == 1
    assert "hfref.overrides.txt: would be removed" in capsys.readouterr().out
    assert (repo / REL / "hfref.overrides.txt").exists()


def test_check_missing_lock(repo, monkeypatch, capsys):
    (repo / REL / "cpu.txt").unlink()
    fake = _fake(monkeypatch, edit=lambda t: "numpy==1\n")
    assert L.main(["--check", "cpu"]) == 1
    assert fake.calls[0][2] is None  # nothing to seed from
    assert "cpu.txt: missing" in capsys.readouterr().out
    assert not (repo / REL / "cpu.txt").exists()


def test_check_compile_failure_exits_2(repo, monkeypatch, capsys):
    before = _digest(repo)
    _fake(monkeypatch, fail=True)
    assert L.main(["--check"]) == 2
    assert "img_cpu: compile failed" in capsys.readouterr().err
    assert _digest(repo) == before


def test_check_diff_is_truncated(repo, monkeypatch, capsys):
    _fake(monkeypatch, edit=lambda t: "".join(f"pkg{i}==1\n" for i in range(100)))
    assert L.main(["--check", "cpu"]) == 1
    assert "more" in capsys.readouterr().out.splitlines()[-1]


def test_check_and_upgrade_are_exclusive(repo, monkeypatch):
    _fake(monkeypatch)
    with pytest.raises(SystemExit) as e:
        L.main(["--check", "--upgrade"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        L.main(["--check", "gpu"])


def test_check_leaves_image_fingerprints_alone(repo, monkeypatch):
    """--check reads the locks only: the image fingerprints (and so every config / stage hash
    built on them) are the same before and after a check, pass or fail."""
    before = images.image_fingerprints()
    _fake(monkeypatch, edit=lambda t: t.replace("==", "==0"))
    monkeypatch.setattr(L, "LOCK_DIR", images.LOCK_DIR)  # check the real committed locks
    monkeypatch.setattr(L, "REPO_ROOT", SCRIPT.parents[1])
    assert L.main(["--check"]) == 1
    assert images.image_fingerprints() == before


def test_default_mode_writes_inputs_and_compiles_in_repo(repo, monkeypatch):
    (repo / REL / "cpu.in").write_text("stale\n")
    fake = _fake(monkeypatch)
    assert L.main(["cpu"]) == 0
    assert (repo / REL / "cpu.in").read_text() == L.input_text("img_cpu")
    cmd, cwd, _ = fake.calls[0]
    assert cwd == repo and cmd == L.compile_cmd("img_cpu", upgrade=False)
    assert L.main(["--upgrade", "cpu"]) == 0
    assert fake.calls[1][0][-1] == "--upgrade"
