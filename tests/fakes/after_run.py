"""A fake repository after a certified run, for ``scripts/after_certified.py``
(``tests/unit/test_after_certified.py``).

:func:`build` lays out, under ``tmp``:

- ``repo/``: the files ``scripts/wave_h_finalize.py`` owns, copied from this checkout
  (``planner/rates.json``, ``README.md``, the generated docs, the certificate directory without
  its certificates), and the lab JSON under ``reports/wave-h/``;
- ``repo/reports/wave-h/certified/``: the run's ``state.json`` and ``records/``, from
  ``tests/fakes/certcheck_run.py`` (the records the real workers would write, with the current
  key), plus what finalize needs on top of the cert-check state: each job's config, its
  per-container receipts and video seconds;
- ``nv/``: stand-in "NVIDIA" text for the shingle check (our own words).

No certificate is written here: finalize writes them with the real ``flora certify``, so the
happy path is the real generators end to end.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tests.fakes import certcheck_run as R

REPO = Path(__file__).resolve().parents[2]
CERT_REL = Path("src/flora_deploy/config/certificates")
COPIED = ("src/flora_deploy/planner/rates.json", "README.md", "docs/balanced-engines.md",
          "docs/costs.md", "docs/limitations.md", "docs/cli.md", "docs/wave-h-report.md")
VIDEO_S = {"exact_e2e": 250.0, "noise": 102.0, "balanced": 250.0, "witness": 102.0}


@dataclass
class AfterFake:
    root: Path  # the fake repository
    run_root: Path  # reports/wave-h/certified
    nv_root: Path

    @property
    def cert_dir(self) -> Path:
        return self.root / CERT_REL

    @property
    def state(self) -> Path:
        return self.run_root / "state.json"

    @property
    def records(self) -> Path:
        return self.run_root / "records"

    def args(self, *extra: str) -> list[str]:
        """``scripts/after_certified.py`` arguments pointing at this fake."""
        return ["--root", str(self.root), "--run-root", str(self.run_root), "--nv-root",
                str(self.nv_root), *extra]

    def tree(self) -> dict[str, bytes]:
        """Every file of the fake repository except the run's own inputs."""
        skip = {"certified", "certified-yam", "work"}
        return {str(p.relative_to(self.root)): p.read_bytes() for p in sorted(self.root.rglob("*"))
                if p.is_file() and not skip & set(p.relative_to(self.root).parts)}


def _containers(kinds: tuple[str, ...], gpus: dict[str, str]) -> list[dict[str, Any]]:
    return [{"container_id": f"ta-{k}", "kind": k, "gpu": gpus[k], "gpu_count": 1,
             "usd": 0.5, "stopped": True, "startup_s": 300.0, "busy_s": 90.0, "seconds": 420.0}
            for k in kinds]


def _job(job_id: str, step: str, cfg: Path | None, preset: str) -> dict[str, Any]:
    kinds = ("segcap",) if preset == "exact" else ("segment", "caption")
    gpus = {"segcap": "H200", "segment": "L40S", "caption": "H100!"}
    return {"app": f"flora-{job_id}", "job_id": job_id, "state": "done", "spend": 1.0,
            "containers": _containers(kinds, gpus), "video_s": VIDEO_S[step],
            "cfg": str(cfg) if cfg else None}


def config(path: Path, embodiment: str = "human") -> Path:
    d = {"project": f"wave-h-test-balanced-{embodiment}", "license_ack": "research-only",
         "modal": {"max_gpus": 4, "budget_usd": 4}, "preset": "balanced",
         "allow_uncertified": True,
         "input": {"format": "lerobot_v3", "repo": "org/ds",
                   "revision": "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"},
         "pipeline": {"stages": ["segment", "caption"], "embodiment": embodiment},
         "output": {"sink": "s3://b/p"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    return path


def build(tmp: Path, opts: R.Options | None = None, *, embodiment: str = "human") -> AfterFake:
    """``embodiment``: whose run it is (its records' ``provenance.embodiment`` and its config;
    the state's job key stays ``human``, as ``scripts/wave_h.py`` names it for every
    embodiment)."""
    cc_run = R.build(tmp / "cc", opts or R.Options())
    root = tmp / "repo"
    for rel in COPIED:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    certs = root / CERT_REL
    certs.mkdir(parents=True, exist_ok=True)
    for p in (REPO / CERT_REL).glob("*.md"):  # the directory's README, no certificate
        shutil.copyfile(p, certs / p.name)
    wave = root / "reports" / "wave-h"
    wave.mkdir(parents=True, exist_ok=True)
    for p in (cc_run.root / "lab").glob("*.json"):
        shutil.copyfile(p, wave / p.name)
    run_root = wave / ("certified" if embodiment == "human" else f"certified-{embodiment}")
    shutil.copytree(cc_run.root / "records", run_root / "records")
    if embodiment != "human":
        for p in sorted((run_root / "records").rglob("*.jsonl")):
            rows = R.read_jsonl(p)
            for r in rows:
                r["provenance"]["embodiment"] = embodiment
            R.write_jsonl(p, rows)
    cfg = config(run_root / "work" / "flora-balanced-human.yaml", embodiment)
    st = json.loads((cc_run.root / "state.json").read_text(encoding="utf-8"))
    steps = st["steps"]
    steps["exact_e2e"]["result"] = {"jobs": {"human": _job("wave-h-test-b", "exact_e2e", None,
                                                           "exact")}}
    steps["noise"]["result"] = {"jobs": {"human": _job("wave-h-test-f", "noise", None, "exact")}}
    bal = steps["balanced"]["result"]
    bal["jobs"]["human"] = {**bal["jobs"]["human"], **_job("wave-h-test-h", "balanced", cfg,
                                                          "balanced")}
    for j, w in bal.get("witness_jobs", {}).items():
        bal["witness_jobs"][j] = {**w, **_job(w["job_id"], "witness", cfg, "balanced")}
    (run_root / "state.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return AfterFake(root, run_root, cc_run.root / "nv")
