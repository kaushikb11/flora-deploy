"""What a Gate certificate freezes, and whether the committed ones still match the code.

    uv run python scripts/cert_freeze.py            # freeze listing per (preset, kind, GPU)
    uv run python scripts/cert_freeze.py --check    # drift report; exit 1 on an error
    uv run python scripts/cert_freeze.py --lock     # write missing freeze locks, prune orphans

``--check`` is the report ``tests/unit/test_certificate_drift.py`` fails with. ``--markdown``
formats it for a GitHub step summary; ``--github`` also prints ``::error`` annotations.
``--lock`` writes ``config/certificates/freeze/<id>.json`` for every committed certificate whose
key still matches this tree (``flora certify`` writes it too; ``wave_h_finalize.py`` copies only
the certificates, so run this after it). See ``docs/certificate-freeze.md``. CPU only: no Modal,
no GPU, no NVIDIA code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flora_deploy.validate import certfreeze as F
from flora_deploy.validate import certkey as K


def lock(cert_dir: Path) -> int:
    rc = 0
    certs = K.load_certificates(cert_dir)
    for c in certs:
        if F.read_freeze(cert_dir, c) is not None:
            continue
        try:
            doc = F.current_key_document(c)
        except Exception as e:  # noqa: BLE001
            print(f"{c['id']}: no lock written: the current key cannot be computed ({e})")
            rc = 1
            continue
        if K.sha256_hex(doc) != c["engine_spec_hash"]:
            print(f"{c['id']}: no lock written: this tree no longer computes its key (drift; "
                  "run --check)")
            rc = 1
            continue
        print(f"wrote {F.write_freeze(c, doc, cert_dir)}")
    live = {c["id"] for c in certs}
    fdir = cert_dir / F.FREEZE_DIRNAME
    for p in sorted(fdir.glob("*.json")) if fdir.is_dir() else ():
        if p.stem not in live:
            p.unlink()
            print(f"removed orphan lock {p}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cert-dir", type=Path, default=K.CERT_DIR)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="drift report; exit 1 on an error")
    mode.add_argument("--lock", action="store_true", help="write missing freeze locks")
    ap.add_argument("--markdown", action="store_true", help="--check: Markdown output")
    ap.add_argument("--github", action="store_true", help="--check: ::error annotations")
    ap.add_argument("--verbose", action="store_true", help="--check: include info findings")
    args = ap.parse_args(argv)
    if args.lock:
        return lock(args.cert_dir)
    if args.check:
        rep = F.check(args.cert_dir)
        print(F.format_report(rep, markdown=args.markdown, verbose=args.verbose))
        if args.github:
            for f in rep.by("error"):
                msg = f"{f.cert} {f.what}: {f.message}".replace("\n", " ")
                print(f"::error title=Certificate drift ({f.status})::{msg}")
        return 0 if rep.ok else 1
    sys.stdout.write(F.freeze_listing(args.cert_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
