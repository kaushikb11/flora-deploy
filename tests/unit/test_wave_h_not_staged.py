"""``wave_h.py --plan certified --embodiment yam --dry-run`` (online) when golden/v3-yam was
never staged: a one-line refusal naming ``golden_stage.py``, exit 2, not a traceback. The golden
Volume is faked (no Modal)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
if "wave_h" in sys.modules:
    wh = sys.modules["wave_h"]
else:
    spec = importlib.util.spec_from_file_location("wave_h", ROOT / "scripts" / "wave_h.py")
    wh = importlib.util.module_from_spec(spec)
    sys.modules["wave_h"] = wh  # dataclasses look their module up while it loads
    spec.loader.exec_module(wh)


class NotFoundError(Exception):
    """Named like ``modal.exception.NotFoundError`` (a Volume that does not exist)."""


def fake_modal(monkeypatch, exc: BaseException):
    reads = []

    class Vol:
        def read_file(self, path):
            reads.append(path)
            raise exc

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(
        Volume=types.SimpleNamespace(from_name=lambda name, **kw: Vol())))
    return reads


@pytest.mark.parametrize("exc", [FileNotFoundError("golden/v3-yam/manifest.json"),
                                 NotFoundError("Volume 'flora-golden' not found")])
def test_yam_dry_run_not_staged_is_a_clear_refusal(monkeypatch, tmp_path, capsys, exc):
    reads = fake_modal(monkeypatch, exc)
    rc = wh.main(["--plan", "certified", "--embodiment", "yam", "--dry-run",
                  "--state", str(tmp_path / "state.json")])
    out = capsys.readouterr()
    assert rc == 2
    assert reads == ["golden/v3-yam/manifest.json"]
    assert "golden/v3-yam is not staged" in out.out
    assert "run scripts/golden_stage.py --version golden/v3-yam" in out.out
    assert "Traceback" not in out.out + out.err
    assert not (tmp_path / "state.json").exists()  # a dry run never writes the state


def test_other_volume_errors_still_raise(monkeypatch):
    fake_modal(monkeypatch, PermissionError("token rejected"))
    with pytest.raises(PermissionError):
        wh.load_manifest(False, "golden/v3-yam")


def test_v1_not_staged_names_its_version(monkeypatch):
    def missing():
        raise FileNotFoundError("golden/v1/manifest.json")

    monkeypatch.setattr(wh.h0, "golden_manifest", missing)
    with pytest.raises(wh.NotStaged, match=r"golden_stage\.py --version golden/v1"):
        wh.load_manifest(False, "golden/v1")


def test_offline_never_reads_the_volume(monkeypatch):
    fake_modal(monkeypatch, AssertionError("read the Volume"))
    assert wh.load_manifest(True, "golden/v3-yam")["offline"] is True
