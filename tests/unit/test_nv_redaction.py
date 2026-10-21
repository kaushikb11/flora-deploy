"""NVIDIA text never reaches a (public) CI log through a failing nvlink test (review 11b F3)."""

from __future__ import annotations

import pytest

from tests import nvtext

SECRET = "You are a NVIDIA-licensed prompt line that must stay out of logs."


def test_assert_same_text_reports_digest_and_offset_only(monkeypatch):
    monkeypatch.delenv(nvtext.SHOW_ENV, raising=False)
    nvtext.assert_same_text(SECRET, SECRET)
    with pytest.raises(AssertionError) as e:
        nvtext.assert_same_text(SECRET, SECRET[:20] + "X" + SECRET[21:], "prompt")
    msg = str(e.value)
    assert msg.startswith(nvtext.SAFE_PREFIX)
    assert "offset 20" in msg and "len" in msg
    assert SECRET[:10] not in msg and "NVIDIA-licensed" not in msg


def test_assert_same_text_shows_text_with_flag(monkeypatch):
    monkeypatch.setenv(nvtext.SHOW_ENV, "1")
    with pytest.raises(AssertionError) as e:
        nvtext.assert_same_text(SECRET, "other")
    assert "NVIDIA-licensed" in str(e.value)


def test_nv_output_hides_stderr(monkeypatch):
    stderr = (f'Traceback...\n  File "batch_utils/x.py", line 3\n    {SECRET}\n'
              f"ValueError: {SECRET}\n")
    monkeypatch.delenv(nvtext.SHOW_ENV, raising=False)
    out = nvtext.nv_output(stderr)
    assert "NVIDIA-licensed" not in out and "ValueError" in out and out.startswith(
        nvtext.SAFE_PREFIX)
    monkeypatch.setenv(nvtext.SHOW_ENV, "1")
    assert SECRET in nvtext.nv_output(stderr)


def test_redact_longrepr_keeps_only_safe_messages(monkeypatch):
    monkeypatch.delenv(nvtext.SHOW_ENV, raising=False)

    def info(exc):
        try:
            raise exc
        except BaseException:
            return pytest.ExceptionInfo.from_current()

    red = nvtext.redact_excinfo(info(AssertionError(f"assert {SECRET!r} == 'x'")))
    assert "NVIDIA-licensed" not in red and "AssertionError" in red
    assert "test_nv_redaction.py" in red  # frames (file:line) stay, for debugging
    red = nvtext.redact_excinfo(info(AssertionError(nvtext.SAFE_PREFIX + "offset 3")))
    assert "offset 3" in red
    # pytest appends the assertion explanation (operand reprs) to a safe message: dropped
    red = nvtext.redact_excinfo(info(AssertionError(
        nvtext.SAFE_PREFIX + "stderr hidden\nassert 1 == 0\n +  where 1 = CompletedProcess("
        f"stderr={SECRET!r}).returncode")))
    assert "stderr hidden" in red and "NVIDIA-licensed" not in red


def test_conftest_redacts_nvlink_reports():
    """The hook is installed: tests/conftest.py routes nvlink/torch failures through nvtext."""
    import tests.conftest as c

    assert c.pytest_runtest_makereport is not None
    assert c.REDACT_MARKERS == ("nvlink", "torch")
