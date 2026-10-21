"""Keep NVIDIA text (prompts, source lines, outputs) out of test failure output (review 11b F3).

CI logs of this public repository are public. A failing nvlink test must not print NVIDIA's
noncommercially licensed prompt text or source lines, so:

* :func:`assert_same_text` compares two texts by sha256 and length and reports only the first
  differing offset;
* :func:`nv_output` reduces a subprocess's stdout/stderr to the last line's exception type;
* :func:`redact_excinfo` (used by ``tests/conftest.py`` for every ``nvlink``/``torch`` test)
  replaces a failure report with frame locations (file:line, no source) and the exception type;
  a message is kept only when it starts with :data:`SAFE_PREFIX` (our own digest/offset text).

``FLORA_SHOW_NV_OUTPUT=1`` (never set in CI) turns all of this off, for local debugging.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

SHOW_ENV = "FLORA_SHOW_NV_OUTPUT"
SAFE_PREFIX = "[nv-redacted] "


def show() -> bool:
    return os.environ.get(SHOW_ENV) == "1"


def _digest(s: str | bytes) -> str:
    b = s.encode("utf-8", "surrogatepass") if isinstance(s, str) else s
    return hashlib.sha256(b).hexdigest()[:16]


def first_diff(a: str | bytes, b: str | bytes) -> int:
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def assert_same_text(got: str | bytes, want: str | bytes, what: str = "text") -> None:
    if got == want:
        return
    if show():
        raise AssertionError(f"{what} differs:\n--- got\n{got!r}\n--- want\n{want!r}")
    raise AssertionError(
        f"{SAFE_PREFIX}{what} differs: sha256 {_digest(got)} len {len(got)} vs sha256 "
        f"{_digest(want)} len {len(want)}; first difference at offset {first_diff(got, want)} "
        f"(set {SHOW_ENV}=1 locally to see the text)")


_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))\b")


def nv_output(text: str | None, what: str = "output") -> str:
    """Subprocess output safe for a public log: the exception type of the last matching line."""
    text = text or ""
    if show():
        return text[-4000:]
    kinds = [m.group(1) for line in text.splitlines() if (m := _EXC_LINE.match(line.strip()))]
    kind = kinds[-1] if kinds else "no exception line"
    return (f"{SAFE_PREFIX}{what} hidden ({len(text)} chars, sha256 {_digest(text)}, last "
            f"exception type: {kind}; set {SHOW_ENV}=1 locally to see it)")


def redact_excinfo(excinfo) -> str:
    """A failure report without source lines or (unsafe) messages."""
    exc = excinfo.value
    lines = ["NVIDIA-adjacent failure, redacted (tests/nvtext.py; set "
             f"{SHOW_ENV}=1 locally for the full report). Frames:"]
    tb = excinfo.traceback
    try:
        tb = tb.filter(excinfo)  # drop pytest/pluggy internals, as pytest's own report does
    except Exception:  # noqa: BLE001 - an older pytest: keep every frame
        pass
    for entry in tb:
        if any(part in ("_pytest", "pluggy") for part in Path(str(entry.path)).parts):
            continue
        lines.append(f"  {entry.path}:{entry.lineno + 1} in {entry.name}")
    msg = str(exc)
    if msg.startswith(SAFE_PREFIX):  # our one-line text only: pytest appends its assertion
        lines.append(f"{type(exc).__name__}: {msg.splitlines()[0]}")  # explanation (operands)
    else:
        lines.append(f"{type(exc).__name__} (message hidden, {len(msg)} chars)")
    return "\n".join(lines)
