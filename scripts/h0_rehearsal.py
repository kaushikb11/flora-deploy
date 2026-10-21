"""H0: CPU rehearsal of Wave H on real Modal (``docs/NEXT_EXECUTION.md``, "de-risking the one
GPU session").

Usage::

    uv run python scripts/h0_rehearsal.py [--dry-run] [--keep] [--work DIR] [--max-wall-s 1500]

What it runs, per job (``human``: 2 EgoDex episodes; ``yam``: 2 three-view + 2 single-view robot
episodes + 1 injected poison episode), exactly the commands Wave H runs, with fake models:

1. fetch the episodes from the golden Volume (``flora-golden``, ``golden/v1/``) into a local
   ``mp4_dir`` dataset (task text next to each video) and upload it to a dedicated media Volume
   (``flora-h0-media``), which the config names as ``input.stage_to``;
2. ``flora index --local`` (receipts into a local store, mirrored into the job store on the
   dedicated Volume ``flora-h0``: ``flora index --local`` cannot write a ``modal-volume://``
   store from a laptop, and ``flora index`` on Modal refuses a local ``input.path``);
3. ``flora plan --index``, ``flora deploy --fake-models --determinism deterministic``;
4. fault spec into the yam job (``workers_app.fake_faults_path``): ``simulate_preemption()``
   before the 2nd episode of ``segcap-00000`` and a poison episode (``os._exit(137)`` every
   time), then ``flora run --fake-models --determinism deterministic`` for both jobs in parallel;
5. ``flora status``, ``flora results`` and the SDK's ``Job.results`` / store listing;
6. assertions (printed as PASS/FAIL), then ``modal app stop`` for the apps it deployed and
   deletion of the two dedicated Volumes and this run's lease/watch entries (``--keep`` skips
   the Volume deletion).

``--dry-run`` plans everything offline (configs, fake app plans, the real-vs-fake hashes) and
contacts Modal only to read the golden manifest (nothing is deployed or written).
CPU only; well under $0.30.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOLDEN_VOLUME = "flora-golden"
GOLDEN_ROOT = "golden/v1"
STORE_VOLUME = "flora-h0"
MEDIA_VOLUME = "flora-h0-media"
DETERMINISM = "deterministic"
POISON_ID = "zz_h0_poison"  # sorts last: the preemption lands before it
POISON_FROM = "single_view_robot/wgo/robointer_droid_000010"  # tiny clip, copied as poison
PREEMPT_SHARD = "segcap"  # a kind: shard ids are content-addressed
PREEMPT_AT = 2  # before the 2nd episode of the shard's first start

# The 6 Wave H-lite episodes (short; across human / yam 3-view / single-view robot). The yam
# job holds the 3-view and the single-view robot episodes (golden profile ``yam``); human
# episodes need their own job (one embodiment per config).
EPISODES: dict[str, tuple[str, ...]] = {
    "human": ("human/egodex_add_remove_lid/episode_000014",
              "human/egodex_braid_unbraid/episode_000001"),
    "yam": ("yam/aloha_static_cups_open/episode_000000",
            "yam/aloha_static_towel/episode_000000",  # also a dense-reward target
            "single_view_robot/wgo/robointer_droid_000010",
            "single_view_robot/wgo/galaxea_045"),
}  # 76.5 s of video in all (scripts/wave_h.py prices exactly this set)
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")


def flora_bin() -> str:
    exe = Path(sys.executable).with_name("flora")
    return str(exe) if exe.exists() else "flora"


def episode_id(golden_id: str) -> str:
    """``<pool>/<dataset>/<episode>`` -> ``<dataset>-<episode>``."""
    parts = golden_id.split("/")
    return f"{parts[-2]}-{parts[-1]}"


# ---------------------------------------------------------------------------------------------
# Dataset


def golden_manifest() -> dict[str, Any]:
    import modal

    vol = modal.Volume.from_name(GOLDEN_VOLUME, version=2)
    return json.loads(b"".join(vol.read_file(f"{GOLDEN_ROOT}/manifest.json")))


def fetch_dataset(manifest: dict[str, Any], work: Path, *, download: bool = True,
                  episodes: dict[str, tuple[str, ...]] | None = None,
                  poison_job: str | None = "yam") -> dict[str, dict[str, Any]]:
    """Lay out ``work/<job>/`` as ``mp4_dir`` (single view: ``<id>.mp4`` + ``<id>.txt``; three
    views: ``<id>/{main,left_wrist,right_wrist}.mp4`` + ``task.txt``) for each job of
    ``episodes`` (default :data:`EPISODES`; ``poison_job`` also gets the poison copy). Returns
    per job ``{episode_id: {"golden": id, "task": str, "views": n, ...}}``."""
    import modal

    by_id = {e["id"]: e for e in manifest["episodes"]}
    vol = modal.Volume.from_name(GOLDEN_VOLUME, version=2) if download else None
    out: dict[str, dict[str, Any]] = {}
    for job, ids in (EPISODES if episodes is None else episodes).items():
        root = work / job
        root.mkdir(parents=True, exist_ok=True)
        eps: dict[str, Any] = {}
        todo = [(i, episode_id(i)) for i in ids]
        if job == poison_job:
            todo.append((POISON_FROM, POISON_ID))
        for gid, eid in todo:
            e = by_id[gid]
            task = (e.get("task") or "").strip()
            files = e["files"]
            single = len(files) == 1
            eps[eid] = {"golden": gid, "task": task, "views": len(files),
                        "duration_s": e["duration_s"], "poison": eid == POISON_ID}
            if not download:
                continue
            for role, f in files.items():
                dest = (root / f"{eid}.mp4") if single else (root / eid / f"{role}.mp4")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    data = b"".join(vol.read_file(f"{e['path']}/{f['file']}"))
                    dest.write_bytes(data)
            (root / (f"{eid}.txt" if single else f"{eid}/task.txt")).write_text(
                task + "\n", encoding="utf-8")
        out[job] = eps
    return out


def upload_dir(volume: str, local: Path, remote: str) -> int:
    import modal

    vol = modal.Volume.from_name(volume, version=2, create_if_missing=True)
    n = 0
    with vol.batch_upload(force=True) as batch:
        for p in sorted(local.rglob("*")):
            if p.is_file():
                batch.put_file(str(p), f"/{remote.strip('/')}/{p.relative_to(local).as_posix()}")
                n += 1
    return n


# ---------------------------------------------------------------------------------------------
# Config


def config_text(*, project: str, data_dir: str, embodiment: str, stages: tuple[str, ...],
                store_volume: str, media_volume: str, run_id: str, budget_usd: float = 20.0,
                resources: dict[str, str] | None = None, tag: str = "flora-h0",
                header: str = "H0 rehearsal; generated by scripts/h0_rehearsal.py") -> str:
    """One ``flora.yaml`` for an ``mp4_dir`` job staged on ``media_volume``. Shared with
    ``scripts/wave_h.py``, so H0 rehearses exactly Wave H's configs."""
    from flora_deploy.config.presets import get_preset, shard_kinds_for

    kinds = shard_kinds_for(get_preset("exact"), list(stages))
    res = {k: "{max_containers: 1}" for k in kinds}
    res.update(resources or {})
    res_lines = "\n".join(f"  {k}: {v}" for k, v in res.items())
    return f"""# {header} ({data_dir})
project: {project}
license_ack: research-only
modal:
  environment: main
  max_gpus: 4
  budget_usd: {budget_usd:g}
  tags: {{{tag}: "{run_id}"}}
preset: exact
input:
  format: mp4_dir
  path: ./{data_dir}
  stage_to: volume://{media_volume}/{run_id}/{data_dir}
pipeline:
  stages: [{", ".join(stages)}]
  embodiment: {embodiment}
  episode_instruction:
    source: dataset_task
  envelope: warn
  shard_target_gpu_minutes: 10
resources:
{res_lines}
output:
  sink: volume://{store_volume}/{run_id}
  formats: [jsonl]
retries:
  max_starts_before_quarantine: 2
"""


def write_config(work: Path, job: str, run_id: str) -> Path:
    text = config_text(project=f"h0-{run_id}-{job}", data_dir=job,
                       embodiment="human" if job == "human" else "yam", stages=STAGES,
                       store_volume=STORE_VOLUME, media_volume=MEDIA_VOLUME, run_id=run_id)
    path = work / f"flora-{job}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def plans(cfg_path: Path) -> dict[str, Any]:
    """The fake app plan and, for comparison, the real ones (nvidia / deterministic)."""
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import FAKE_RECEIPT, plan_app

    cfg = load_config(cfg_path)
    receipts = {"captioner_bf16": FAKE_RECEIPT}
    fake = plan_app(cfg, determinism=DETERMINISM, fake_models=True)
    real = plan_app(cfg, determinism=DETERMINISM, derived_receipts=receipts)
    real_nv = plan_app(cfg, determinism="nvidia", derived_receipts=receipts)
    fake_nv = plan_app(cfg, determinism="nvidia", fake_models=True)
    return {"cfg": cfg, "fake": fake, "real": real, "real_nvidia": real_nv,
            "fake_nvidia": fake_nv}


# ---------------------------------------------------------------------------------------------
# Running


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    gap: bool = False  # a known product gap outside H0's scope: reported, not counted


@dataclass
class JobRun:
    job: str
    cfg_path: Path
    job_id: str
    episodes: dict[str, Any]
    plan: Any = None
    run_rc: int | None = None
    run_out: str = ""
    status: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "", *, gap: bool = False) -> None:
        self.checks.append(Check(name, bool(ok), detail, gap))


def sh(args: list[str], *, cwd: Path, timeout: float = 900, check: bool = True,
       log: Path | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(args)}", flush=True)
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    if log is not None:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(args)}\n{out}\n[rc={p.returncode}]\n")
    tail = "\n".join(out.strip().splitlines()[-12:])
    if tail:
        print("  " + tail.replace("\n", "\n  "), flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"{args[:3]} failed ({p.returncode})")
    return p


def mirror_index(local_store: Path, job_id: str, run_id: str, volume: str = STORE_VOLUME) -> int:
    """``flora index --local`` wrote the receipts to a local store; put them into the job store
    on the Volume, at the same relative paths (``flora/<job>/...``)."""
    src = local_store / "flora" / job_id
    return upload_dir(volume, src, f"{run_id}/flora/{job_id}")


def remote_store(run_id: str):
    import modal

    from flora_deploy.sdk import RemoteVolumeStore

    return RemoteVolumeStore(modal.Volume.from_name(STORE_VOLUME, version=2), run_id)


def poison_uid(local_store: Path, job_id: str) -> str:
    for p in sorted((local_store / "flora" / job_id).rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if d.get("episode_id") == POISON_ID and d.get("uid"):
            return str(d["uid"])
    raise RuntimeError("poison episode not found in the index receipts")


# ---------------------------------------------------------------------------------------------
# Assertions


def assert_job(r: JobRun, run_id: str, poison: str | None) -> None:
    from flora_deploy import sdk
    from flora_deploy.jobs.store import JobPaths

    p = r.plan
    want_state = "completed_with_errors" if poison else "done"
    r.check("end state", r.status.get("state") == want_state,
            f"state={r.status.get('state')} (want {want_state}), flora run rc={r.run_rc}")
    job = sdk.open_job(r.job_id, store=p.store.uri, environment=p.environment)
    summary = job.summary() or {}
    store = job.store
    manifest = job._deps().manifests.load()
    manifest_uids = {e.uid for e in manifest.episodes}
    want_uids = manifest_uids - ({poison} if poison else set())
    tasks = {}
    for eid, meta in r.episodes.items():
        for u in manifest_uids:
            if u.endswith(f"--{eid}"):
                tasks[u] = meta["task"]
    r.check("planned episodes", len(manifest_uids) == len(r.episodes),
            f"{len(manifest_uids)} planned, {len(r.episodes)} staged")

    records: dict[str, list[Any]] = {}
    for stage in STAGES:
        try:
            records[stage] = job.results(stage)
        except Exception as e:  # noqa: BLE001 - reported as a failed check
            records[stage] = []
            r.check(f"results {stage}", False, repr(e))
    for stage, recs in records.items():
        uids = Counter(x.uid for x in recs)
        dup = sorted(u for u, n in uids.items() if n > 1)
        r.check(f"{stage}: no lost/duplicate records",
                set(uids) == want_uids and not dup,
                f"{len(uids)} uids (want {len(want_uids)}), missing "
                f"{sorted(want_uids - set(uids))}, extra {sorted(set(uids) - want_uids)}, "
                f"dup {dup}")
        certs = {x.provenance.certificate for x in recs}
        models = {v for x in recs for v in x.provenance.models.values()}
        r.check(f"{stage}: records marked fake", certs <= {"fake-models"} and models <= {"fake"}
                and bool(recs), f"certificates {certs}, models {models}")
    stages_sum = summary.get("stages", {})
    excused = {s: c.get("excused", 0) for s, c in stages_sum.items()}
    if poison:
        pills = summary.get("poison_pills", [])
        r.check("poison excluded (exactly)",
                all(n == 1 for n in excused.values()) and any(poison in json.dumps(x)
                                                             for x in pills),
                f"excused {excused}, poison pills {pills}")
    else:
        r.check("nothing excused", not any(excused.values()) and not summary.get("poison_pills"),
                f"excused {excused}")

    # instruction text: the fake captioner writes "(task: <instruction>)" into every caption
    bad = []
    for x in records.get("caption", []):
        t = tasks.get(x.uid, "")
        if not t or not all(f"(task: {t})" in (s.caption or "") for s in x.segments):
            bad.append(x.uid)
    r.check("instruction text flows (index receipt -> worker -> captions)",
            not bad and bool(records.get("caption")), f"wrong/missing task in {bad}")

    # stage-hash keyed paths: every part file lives under its kind's stage hash
    files = list(store.list(f"{JobPaths(r.job_id).root}/"))
    parts = [f for f in files if "/parts/" in f]
    hashes = dict(manifest.stage_hashes)
    off = [f for f in parts if not any(f"/{h}/parts/{k}/" in f for k, h in hashes.items())]
    kinds_seen = {k for k, h in hashes.items() if any(f"/{h}/parts/{k}/" in f for f in parts)}
    r.check("stage-hash keyed part paths (manifest = deployed plan)",
            bool(parts) and not off and kinds_seen == set(hashes) and hashes == p.stage_hashes,
            f"{len(parts)} part files, {len(off)} off-hash, kinds {sorted(kinds_seen)}, "
            f"manifest==plan {hashes == p.stage_hashes}")

    # determinism forwarded: env, hash and the segcap shard receipts' runtime record
    env = p.object(p.worker_classes["segcap"]).env
    modes: Counter = Counter()
    reward_tasks_ok, reward_audits = True, 0
    for f in files:
        if "/audit/" not in f:
            continue
        for line in store.get_bytes(f).decode().splitlines():
            with contextlib.suppress(ValueError):
                a = json.loads(line)
                if "/segcap/" in f:
                    modes[str(((a.get("runtime") or {}).get("determinism") or {}).get("mode"))] += 1
                elif a.get("stage") == "reward.terminal" and a.get("uid") in tasks:
                    reward_audits += 1  # NVIDIA's terminal line carries the task it scored
                    reward_tasks_ok &= (a.get("nv") or {}).get("task") == tasks[a["uid"]]
    r.check("determinism forwarded (env, hash, worker engine)",
            env.get("FLORA_DETERMINISM") == DETERMINISM and
            json.loads(env["FLORA_OUTPUT_SETTINGS"])["determinism"] == DETERMINISM and
            set(modes) == {DETERMINISM}, f"env {env.get('FLORA_DETERMINISM')}, segcap audit "
            f"modes {dict(modes)}")
    r.check("instruction text flows into reward requests (terminal audits)",
            reward_audits >= len(want_uids) and reward_tasks_ok,
            f"{reward_audits} terminal audit rows")
    r.check("media root resolved", env.get("FLORA_MEDIA_ROOT") ==
            f"/media/{MEDIA_VOLUME}/{run_id}/{r.job}" and bool(records.get("segment")),
            f"FLORA_MEDIA_ROOT={env.get('FLORA_MEDIA_ROOT')}")
    if poison:
        markers = [f for f in files if "/attempts/" in f and f"/{PREEMPT_SHARD}/" in f]
        preempted = 0
        for f in markers:
            with contextlib.suppress(ValueError):
                preempted += bool(json.loads(store.get_bytes(f)).get("preempted"))
        fired = any("fake_faults/fired/" in f for f in files)
        r.check("simulate_preemption on a worker (marked preempted, shard still done)",
                fired and preempted >= 1, f"fired={fired}, {len(markers)} markers on "
                f"{PREEMPT_SHARD}, {preempted} preempted")
    r.summary = summary  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------------------------
# Cleanup


def cleanup(app_names: list[str], run_id: str, job_ids: list[str], watch_dicts: list[str], *,
            keep: bool, env: str = "main") -> list[str]:
    import modal

    notes = []
    names = set(app_names)
    with contextlib.suppress(Exception):  # every app of this run, whatever the client planned
        p = subprocess.run(["modal", "app", "list", "--json", "--env", env],
                           capture_output=True, text=True, timeout=120)
        for a in json.loads(p.stdout):
            desc = a.get("Description") or a.get("description") or ""
            state = (a.get("State") or a.get("state") or "").lower()
            if desc.startswith(f"flora-h0-{run_id}-") and "stopped" not in state:
                names.add(desc)
    for name in sorted(names):
        p = subprocess.run(["modal", "app", "stop", "--yes", name, "--env", env],
                           capture_output=True, text=True)
        notes.append(f"modal app stop {name}: rc={p.returncode}")
    try:
        leases = modal.Dict.from_name("flora-orchestrator-leases", environment_name=env)
        for key in list(leases.keys()):
            if any(str(key).startswith(f"orchestrator/{j}") for j in job_ids):
                leases.pop(key)
                notes.append(f"lease {key} removed")
    except Exception as e:  # noqa: BLE001
        notes.append(f"lease cleanup: {e!r}")
    for d in watch_dicts:
        p = subprocess.run(["modal", "dict", "delete", "--yes", d, "--env", env],
                           capture_output=True, text=True)
        notes.append(f"modal dict delete {d}: rc={p.returncode}")
    if not keep:
        for vol in (STORE_VOLUME, MEDIA_VOLUME):
            p = subprocess.run(["modal", "volume", "delete", vol, "--yes", "--env", env],
                               capture_output=True, text=True)
            notes.append(f"modal volume delete {vol}: rc={p.returncode} "
                         f"{(p.stdout + p.stderr).strip()[-160:]}")
    return notes


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="plan only; deploy nothing")
    ap.add_argument("--keep", action="store_true", help="keep the two H0 Volumes")
    ap.add_argument("--work", type=Path, help="work directory (default: a temp dir)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-wall-s", type=float, default=1500.0,
                    help="cancel the jobs if `flora run` has not returned by then")
    ap.add_argument("--json", type=Path, help="write the report here")
    args = ap.parse_args(argv)

    run_id = args.run_id or (time.strftime("%m%d%H%M") + uuid.uuid4().hex[:4])
    work = (args.work or Path(tempfile.mkdtemp(prefix="flora-h0-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # A relative input.path is hashed relative to the cwd (config.hashing.local_input_digest),
    # so plan in-process from the same directory the CLI commands run in.
    os.chdir(work)
    log = work / "h0.log"
    t0 = time.time()
    print(f"H0 run {run_id} in {work}", flush=True)

    manifest = golden_manifest()
    eps = fetch_dataset(manifest, work, download=not args.dry_run)
    runs: list[JobRun] = []
    for job in EPISODES:
        cfg_path = write_config(work, job, run_id)
        runs.append(JobRun(job, cfg_path, f"h0-{run_id}-{job}", eps[job]))
    if args.dry_run:
        for r in runs:
            pl = plans(r.cfg_path) if (work / r.job).is_dir() else None
            if pl is None:
                continue
            f, real = pl["fake"], pl["real"]
            print(f"\n[{r.job}] fake app {f.app_name} vs real {real.app_name}")
            for k in f.stage_hashes:
                print(f"  {k}: fake {f.stage_hashes[k][:12]} real {real.stage_hashes[k][:12]}")
            for o in f.objects:
                print(f"  {o.name:20s} gpu={o.gpu} cpu={o.cpu} mem={o.memory} image={o.image}")
        print(json.dumps({j: list(e) for j, e in eps.items()}, indent=1))
        return 0

    app_names: list[str] = []
    watch: list[str] = []
    notes: list[str] = []
    poison = None
    try:
        # media to its own Volume (input.stage_to), as Wave H stages it
        for r in runs:
            n = upload_dir(MEDIA_VOLUME, work / r.job, f"{run_id}/{r.job}")
            print(f"[{r.job}] uploaded {n} files to {MEDIA_VOLUME}:/{run_id}/{r.job}", flush=True)
        fl = flora_bin()
        for r in runs:
            local_store = work / f"store-{r.job}"
            sh([fl, "index", "--local", "--config", str(r.cfg_path), "--job", r.job_id,
                "--store", f"file://{local_store}", "--out", str(work / f"summary-{r.job}.json")],
               cwd=work, log=log)
            n = mirror_index(local_store, r.job_id, run_id)
            print(f"[{r.job}] mirrored {n} index files into {STORE_VOLUME}:/{run_id}", flush=True)
            sh([fl, "plan", "--config", str(r.cfg_path), "--index",
                str(work / f"summary-{r.job}.json")], cwd=work, log=log)
            pl = plans(r.cfg_path)
            r.plan = pl["fake"]
            r.check("fake hashes differ from real (every stage, both modes)",
                    all(pl["fake"].stage_hashes[k] != pl["real"].stage_hashes[k]
                        for k in pl["fake"].stage_hashes)
                    and pl["fake"].app_name != pl["real"].app_name
                    and pl["fake"].config_hash != pl["fake_nvidia"].config_hash,
                    f"fake {pl['fake'].app_name}, real {pl['real'].app_name}")
            r.check("fake workers on CPU", all(o.gpu is None for o in pl["fake"].objects),
                    ", ".join(f"{o.name}:{o.gpu}" for o in pl["fake"].objects))
            dep = sh([fl, "deploy", "--config", str(r.cfg_path), "--fake-models",
                      "--determinism", DETERMINISM], cwd=work, log=log, timeout=1200)
            app_names.append(r.plan.app_name)
            watch.append(r.plan.watch_dict)
            r.check("deployed app == the client's plan", f"deployed {r.plan.app_name}" in
                    " ".join((dep.stdout + dep.stderr).split()), r.plan.app_name)
            if r.job == "yam":
                from flora_deploy.modal_app.workers_app import fake_faults_path

                poison = poison_uid(local_store, r.job_id)
                remote_store(run_id).put_bytes(fake_faults_path(r.job_id), json.dumps({
                    "preempt": {PREEMPT_SHARD: PREEMPT_AT}, "poison": [poison]}).encode())
                print(f"[yam] faults: preempt {PREEMPT_SHARD}@{PREEMPT_AT}, poison {poison}")

        procs = {}
        for r in runs:
            args_ = [fl, "run", "--config", str(r.cfg_path), "--fake-models", "--determinism",
                     DETERMINISM, "--job", r.job_id, "--poll", "10"]
            print(f"$ {' '.join(args_)}", flush=True)
            procs[r.job] = subprocess.Popen(args_, cwd=work, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
        deadline = t0 + args.max_wall_s
        for r in runs:
            pr = procs[r.job]
            try:
                out, _ = pr.communicate(timeout=max(deadline - time.time(), 1))
            except subprocess.TimeoutExpired:
                pr.kill()
                out, _ = pr.communicate()
                sh([fl, "cancel", r.job_id, "--config", str(r.cfg_path)], cwd=work, log=log,
                   check=False)
                out += "\n[h0: wall-clock limit reached; job cancelled]"
            r.run_rc, r.run_out = pr.returncode, out
            with log.open("a", encoding="utf-8") as f:
                f.write(f"[{r.job} flora run rc={pr.returncode}]\n{out}\n")
            print(f"[{r.job}] flora run rc={pr.returncode}\n  " +
                  "\n  ".join(out.strip().splitlines()[-8:]), flush=True)
        for r in runs:
            st = sh([fl, "status", r.job_id, "--config", str(r.cfg_path), "--json"], cwd=work,
                    log=log, check=False)
            with contextlib.suppress(ValueError):
                r.status = json.loads(st.stdout)
            res = sh([fl, "results", "--job", r.job_id, "--stage", "caption", "--store",
                      r.plan.store.uri], cwd=work, log=log, check=False)
            r.check("`flora results` CLI reads a modal-volume store from the laptop",
                    res.returncode == 0, " ".join((res.stdout + res.stderr).split())[-160:],
                    gap=True)
            assert_job(r, run_id, poison if r.job == "yam" else None)
    except Exception as e:  # noqa: BLE001 - always clean up, then report
        import traceback

        traceback.print_exc()
        notes.append(f"aborted: {e!r}")
    finally:
        notes += cleanup(app_names, run_id, [r.job_id for r in runs], watch, keep=args.keep)
        if not args.work:
            shutil.rmtree(work / "human", ignore_errors=True)
            shutil.rmtree(work / "yam", ignore_errors=True)

    report = {"run_id": run_id, "wall_s": round(time.time() - t0, 1), "apps": app_names,
              "notes": notes, "jobs": {r.job: {
                  "job_id": r.job_id, "state": r.status.get("state"), "rc": r.run_rc,
                  "spend_usd": r.status.get("spend_usd"),
                  "checks": [c.__dict__ for c in r.checks]} for r in runs}}
    print(f"\nH0 {run_id}: {report['wall_s']} s, apps {app_names}")
    ok = not any(n.startswith("aborted") for n in notes)
    for r in runs:
        print(f"  [{r.job}] {r.job_id}: state={r.status.get('state')} rc={r.run_rc} "
              f"spend(est)=${r.status.get('spend_usd', 0) or 0:.4f}")
        for c in r.checks:
            tag = "PASS" if c.ok else ("GAP " if c.gap else "FAIL")
            print(f"    {tag} {c.name}" + (f": {c.detail}" if c.detail else ""))
            ok &= c.ok or c.gap
    for n in notes:
        print(f"  note: {n}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str))
    print(f"log: {log}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
