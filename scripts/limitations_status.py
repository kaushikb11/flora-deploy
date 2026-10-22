"""Generate the status blocks of ``docs/limitations.md`` from the certificates and the rates.

    uv run python scripts/limitations_status.py            # print the regenerated doc
    uv run python scripts/limitations_status.py --write    # rewrite docs/limitations.md

Two blocks, between ``<!-- limitations:<name> -->`` and ``<!-- /limitations:<name> -->``:

- ``certification``: per embodiment and stage, what ``balanced`` runs as today, from the
  committed certificates through ``validate.certkey.status`` (the same lookup ``flora plan`` and
  config validation use; ``reference`` is ``certkey.reference_identical``);
- ``costs``: whether each rate ``flora plan`` prices with is measured or an estimate, from the
  ``status``, ``n`` and ``uncertainty`` fields of ``planner/rates.json``.

The prose around them is edited by hand. ``tests/unit/test_limitations_doc.py`` fails when the
doc differs from :func:`render`, and ``scripts/wave_h_finalize.py`` regenerates it whenever it
writes rates or certificates.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "limitations.md"

PRESET = "balanced"
STAGES = ("segment", "caption", "reward.terminal", "reward.dense", "reward.rubric")
WORDS = {"reference": "reference", "certified": "**certified**", "provisional": "provisional",
         "fail": "failed", "none": "not yet", "unverified_weights": "not yet"}

_BLOCK = re.compile(r"(<!-- limitations:(?P<name>[\w-]+) -->\n)(?P<body>.*?)"
                    r"(<!-- /limitations:(?P=name) -->)", re.DOTALL)


def derived_content() -> dict[str, str]:
    """The pinned derived-weight content digests (what a fresh ``flora prepare`` verifies)."""
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
    from flora_deploy.config.receipts import pin_key

    out = {}
    for name in ("tempseg_merged", "captioner_bf16"):
        pin = DERIVED_PINS_BY_KEY.get(pin_key(name))
        if pin is not None and pin.content_sha256:
            out[name] = pin.content_sha256
    return out


def statuses(cert_dir: Path | None = None) -> dict[str, dict[str, str]]:
    """Embodiment -> stage -> ``certkey.status`` of ``balanced`` on its default pools."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.validate import certkey
    from flora_deploy.validate.certify import EMBODIMENTS

    spec = get_preset(PRESET)
    out = {}
    for emb in EMBODIMENTS:
        st = certkey.status(PRESET, spec.pools, embodiment=emb, derived_content=derived_content(),
                            cert_dir=cert_dir)
        out[emb] = {s: st[spec.stage_kinds[s]]["status"] for s in STAGES}
    return out


def block_certification(cert_dir: Path | None) -> str:
    from flora_deploy.config.presets import get_preset

    pools = get_preset(PRESET).pools
    st = statuses(cert_dir)
    head = " | ".join(f"`{s}` ({pools[s].gpu})" for s in STAGES)
    lines = [f"| Embodiment | {head} |", "|---|" + "---|" * len(STAGES)]
    for emb, row in st.items():
        lines.append(f"| `{emb}` | " + " | ".join(WORDS[row[s]] for s in STAGES) + " |")
    certified = sorted({f"`{s}` for `{e}`" for e, row in st.items() for s in STAGES
                        if row[s] == "certified"})
    lines.append("")
    if certified:
        lines.append(f"Certified today: {', '.join(certified)}. Every other `balanced` "
                     "segmentation or captioning stage needs `allow_uncertified: true`.")
    else:
        lines.append("Nothing is certified yet, so `balanced` segmentation and captioning need "
                     "`allow_uncertified: true`\nand their records say `certified: false`.")
    return "\n".join(lines) + "\n"


def describe(status: str, n: int | None, uncertainty: float, unit: str = "job") -> str:
    """``n``: jobs behind a rate, containers behind a startup (``planner.rates``)."""
    band = f"±{uncertainty:.0%}"
    if status == "measured":
        if n == 1:
            return f"measured on one {unit}, {band}"
        return f"measured on {n} {unit}s, {band}" if n else f"measured, {band}"
    if status == "projected":
        return f"estimate (projected, not measured), {band}"
    return f"estimate (not measured), {band}"


def block_costs(table) -> str:
    rows = [(f"GPU work, `{k}`", table.rate(PRESET, k, 1)) for k in STAGES]
    rows += [(f"Startup, `{k}`, cold", table.startup_for(PRESET, k))
             for k in ("segment", "caption")]
    rows.append(("Startup, rewards", table.startup_for(PRESET, "reward.terminal")))
    rows += [(f"Startup, `{k}`, warm", table.startup_for(PRESET, k, warm=True))
             for k in ("segment", "caption")]
    lines = ["| Figure `flora plan` uses | Status |", "|---|---|"]
    for label, e in rows:
        unit = "container" if label.startswith("Startup") else "job"
        lines.append(f"| {label} | {describe(e.status, e.n, e.uncertainty, unit)} |")
    return "\n".join(lines) + "\n"


def render(text: str | None = None, *, cert_dirs: Sequence[Path] | None = None,
           table=None) -> str:
    """``text`` (default: ``docs/limitations.md``) with its generated blocks regenerated from
    the certificates in ``cert_dirs`` (default: the committed ones) and the rate ``table``
    (default: the shipped ``rates.json``). Raises when a block is missing or unknown."""
    from flora_deploy.planner.rates import RateTable

    text = DOC.read_text(encoding="utf-8") if text is None else text
    table = table or RateTable.load()
    with tempfile.TemporaryDirectory(prefix="limitations-") as tmp:
        cert_dir = None
        if cert_dirs is not None:  # one directory for certkey: later ones win on a name clash
            cert_dir = Path(tmp)
            for d in cert_dirs:
                for p in sorted(Path(d).glob("*.json")) if Path(d).is_dir() else ():
                    shutil.copyfile(p, cert_dir / p.name)
        blocks = {"certification": lambda: block_certification(cert_dir),
                  "costs": lambda: block_costs(table)}
        seen: set[str] = set()

        def sub(m: re.Match) -> str:
            name = m.group("name")
            if name not in blocks:
                raise ValueError(f"unknown limitations block {name!r}")
            seen.add(name)
            return m.group(1) + blocks[name]() + m.group(4)

        out = _BLOCK.sub(sub, text)
    missing = sorted(set(blocks) - seen)
    if missing:
        raise ValueError(f"docs/limitations.md misses limitations blocks: {', '.join(missing)}")
    return out


def main(argv: list[str]) -> int:
    text = render()
    if argv[:1] == ["--write"]:
        DOC.write_text(text, encoding="utf-8")
        print(f"wrote {DOC}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
