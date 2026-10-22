"""Stage a golden selection on the ``flora-golden`` Volume (CPU only), then verify every file.

Usage::

    uv run python scripts/golden_stage.py --version golden/v2            # stage + verify
    uv run python scripts/golden_stage.py --version golden/v2 --verify-only
    uv run python scripts/golden_stage.py --version golden/v2 --summary  # numbers for the doc

Staging is ``modal_app/golden_app.py``'s ``stage`` function (one ``img_cpu`` container, 2 cores,
8 GiB, ephemeral app ``flora-golden-stage``), called with the chosen selection; ``stage_all``
writes under that selection's own prefix (``golden/v2/``), so v1's files and manifest are never
touched. It is resumable: an episode whose ``record.json`` carries the current selection digest
is skipped.

**Verify** (no container; Volume reads from this machine), per episode of the manifest:
- ``status == staged``, ``oracle_eligible``, ``verify.ok`` and every ``cut_check`` ``ok``;
- every staged file is read back from the Volume: its size and sha256 equal the manifest's;
- the manifest on the Volume equals the one the staging call returned (when there is one);
- an episode the earlier version also staged (same selection entry): its files have the same
  sha256 as there (``--compare golden/v1``, default for v2).

The manifest names private episodes and task text, so it is saved under the git-ignored
``reports/golden/`` only. The committed doc (``docs/golden-manifest-v2.md``) takes the numbers
``--summary`` prints. The cost line is the container's wall time x the pinned list $/h; billing
(``scripts/spend.py``, app ``flora-golden-stage``) lags about an hour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "golden"
CPU_CORES, MEMORY_GIB = 2.0, 8.0  # golden_app.CPU_CORES / MEMORY_MIB


def stage_usd(wall_s: float) -> float:
    from flora_deploy.jobs.budget import pinned_rates

    r = pinned_rates()
    return wall_s * (CPU_CORES * r.core_per_s + MEMORY_GIB * r.gib_per_s)


def read_volume(path: str) -> bytes:
    import modal

    from flora_deploy.layout import GOLDEN_VOLUME

    vol = modal.Volume.from_name(GOLDEN_VOLUME, version=2)
    return b"".join(vol.read_file(path))


def verify(manifest: Mapping[str, Any], *, read: Callable[[str], bytes] = read_volume,
           earlier: Mapping[str, Any] | None = None) -> list[str]:
    """Every problem with the staged files (empty: all verified). See the module docstring."""
    problems: list[str] = []
    if not manifest.get("complete"):
        problems.append(f"manifest incomplete: failed {manifest.get('failed')}, missing "
                        f"{manifest.get('missing')}")
    before = {e["id"]: e for e in (earlier or {}).get("episodes", [])}
    for ep in manifest["episodes"]:
        gid = ep["id"]
        if ep.get("status") != "staged":
            problems.append(f"{gid}: status {ep.get('status')} ({ep.get('error')})")
            continue
        if not ep.get("oracle_eligible"):
            problems.append(f"{gid}: not oracle-eligible ({ep.get('ineligibility')})")
        if not (ep.get("verify") or {}).get("ok"):
            problems.append(f"{gid}: verify_episode failed")
        bad = {k: v for k, v in (ep.get("cut_check") or {}).items() if v != "ok"}
        if bad:
            problems.append(f"{gid}: cut_check {bad}")
        files = ep.get("files") or {}
        if not files:
            problems.append(f"{gid}: no files")
        for role, f in files.items():
            if not f.get("file"):
                problems.append(f"{gid}: {role} not staged")
                continue
            data = read(f"{ep['path']}/{f['file']}")
            if len(data) != f.get("bytes"):
                problems.append(f"{gid}: {role} is {len(data)} bytes, manifest {f.get('bytes')}")
            if hashlib.sha256(data).hexdigest() != f.get("sha256"):
                problems.append(f"{gid}: {role} sha256 differs from the manifest")
            prev = _earlier_file(before, ep, role)
            if prev is not None and prev != f.get("sha256"):
                problems.append(f"{gid}: {role} differs from the earlier version's file")
    return problems


def _earlier_file(before: Mapping[str, Any], ep: Mapping[str, Any], role: str) -> str | None:
    """The earlier version's sha256 of the same file, when it staged the same entry."""
    prev = before.get(ep["id"])
    if not prev or prev.get("selection_digest") != ep.get("selection_digest"):
        return None
    return ((prev.get("files") or {}).get(role) or {}).get("sha256")


def summary(manifest: Mapping[str, Any]) -> str:
    """Numbers only, for docs/golden-manifest-v2.md."""
    from flora_deploy.validate.golden import profile_rows, summary_markdown

    lines = [summary_markdown(manifest), "",
             "| Episode | Codec | Size | fps | Frames | Duration (s) | CFR | Kind | Eligible |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in profile_rows(manifest):
        lines.append(f"| `{r['id']}` | {r['codec']} | {r['size']} | {r['fps']} | {r['frames']} "
                     f"| {r['duration_s']:.2f} | {r['cfr']} | {', '.join(r['kind'])} | "
                     f"{r['oracle_eligible']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from flora_deploy.validate.golden import (
        MANIFEST_NAME,
        SELECTIONS,
        load_selection,
        pool_counts,
        prefix_for,
        validate_selection,
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", default="golden/v2", choices=tuple(SELECTIONS))
    ap.add_argument("--verify-only", action="store_true", help="read and check; stage nothing")
    ap.add_argument("--summary", action="store_true", help="print the numbers-only tables")
    ap.add_argument("--compare", default=None,
                    help="an earlier version whose shared episodes must be byte-identical "
                         "(default for golden/v2: golden/v1)")
    ap.add_argument("--only", default="", help="episode ids or dataset keys, comma-separated")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    selection = validate_selection(load_selection(SELECTIONS[args.version]))
    prefix = prefix_for(args.version)
    OUT.mkdir(parents=True, exist_ok=True)
    local = OUT / f"manifest-{prefix.name}.json"
    returned = None
    if not (args.verify_only or args.summary):
        import modal

        from flora_deploy.modal_app import golden_app

        print("selection:", {k: v for k, v in pool_counts(selection).items() if v})
        t0 = time.time()
        with modal.enable_output(), golden_app.app.run():
            returned = golden_app.stage.remote(selection, [s for s in args.only.split(",") if s],
                                               args.force)
        wall = time.time() - t0
        run = returned.get("run") or {}
        print(json.dumps({"counts": returned["counts"], "complete": returned["complete"],
                          "failed": returned["failed"], "missing": returned["missing"],
                          "run": run}, indent=1))
        print(f"staging: app wall {wall:.0f} s (container {run.get('wall_s')} s); tracked cost "
              f"<= ${stage_usd(wall):.4f} at list prices ({CPU_CORES:g} cores, {MEMORY_GIB:g} "
              "GiB); billing: scripts/spend.py --app flora-golden-stage (lags ~1 h)")
    on_volume = json.loads(read_volume(f"{prefix}/{MANIFEST_NAME}"))
    local.write_text(json.dumps(on_volume, indent=1, sort_keys=True) + "\n")
    if args.summary:
        print(summary(on_volume))
        return 0
    problems = []
    if returned is not None and {k: v for k, v in returned.items() if k not in ("run",)} != {
            k: v for k, v in on_volume.items() if k != "run"}:
        problems.append("the manifest on the Volume differs from the one staging returned")
    compare = args.compare or ("golden/v1" if args.version == "golden/v2" else None)
    earlier = (json.loads(read_volume(f"{prefix_for(compare)}/{MANIFEST_NAME}"))
               if compare else None)
    problems += verify(on_volume, earlier=earlier)
    n = sum(len(e.get("files") or {}) for e in on_volume["episodes"])
    shared = sum(1 for e in on_volume["episodes"]
                 if earlier and _earlier_file({x["id"]: x for x in earlier["episodes"]}, e,
                                              "main"))
    if problems:
        print("VERIFY FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"verified: {len(on_volume['episodes'])} episodes, {n} files (size + sha256 read back "
          f"from the Volume), all oracle-eligible, cut_check ok; {shared} byte-identical to "
          f"{compare}; manifest saved to {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
