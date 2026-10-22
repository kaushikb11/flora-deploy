"""H0: CPU rehearsal of Wave H on real Modal (``docs/archive/NEXT_EXECUTION.md``, "de-risking
the one GPU session").

Usage::

    uv run python scripts/h0_rehearsal.py [--preset balanced|exact|both] [--dry-run] [--keep]
        [--work DIR] [--max-wall-s 1500] [--json REPORT]

What it runs, per job, exactly the commands Wave H runs, with fake models. ``exact`` jobs:
``human`` (2 EgoDex episodes) and ``yam`` (2 three-view + 2 single-view robot episodes + 1
injected poison episode). ``balanced`` jobs (``preset: balanced``, ``allow_uncertified: true``;
the factory's ``FakeSegmentWorker`` / ``FakeCaptionWorker``) on the same 6 episodes:
``bal-human`` (clean), ``bal-yam-seg`` (one preemption of a ``segment`` shard, the poison episode
poisoned at ``segment``: excluded at caption too, counted once) and ``bal-yam-cap`` (one
preemption of a ``caption`` shard, poisoned at ``caption`` only). ``--preset`` picks the jobs:
``balanced`` (the default, the fast path: only the episodes and apps those 3 jobs need),
``exact`` or ``both``. Before anything is deployed it prints the projected cost
(:func:`projected_cost`, about $0.4 for the default):

1. fetch the episodes from the golden Volume (``flora-golden``, ``golden/v1/``) into a local
   ``mp4_dir`` dataset (task text next to each video) and upload it to this run's media Volume
   (``flora-h0-media-<run_id>``), which the config names as ``input.stage_to``;
2. ``flora index --local`` (receipts into a local store, mirrored into the job store on this
   run's Volume ``flora-h0-<run_id>``: ``flora index --local`` cannot write a ``modal-volume://``
   store from a laptop, and ``flora index`` on Modal refuses a local ``input.path``);
3. ``flora plan --index``, ``flora deploy --fake-models --determinism deterministic``;
4. fault specs (``workers_app.fake_faults_path``): ``simulate_preemption()`` before the 2nd
   episode of the first start of a ``segcap`` (``segment`` / ``caption``) shard and a poison
   episode (``os._exit(137)`` every time; for the balanced jobs only in workers of the named
   kind), then ``flora run --fake-models --determinism deterministic`` for every job in parallel;
   while the jobs run, :class:`ContainerWatch` polls Modal's running containers (``TaskList``,
   metadata only) every 5 s for this run's apps; after the last ``flora run`` returns it waits
   up to 45 s for them to go before anything is stopped;
5. ``flora status``, ``flora results`` and the SDK's ``Job.results`` / store listing;
6. assertions (printed as PASS/FAIL). Besides the original H0 checks, the behaviours that had
   never run on Modal (:func:`assert_new`):

   - *job-end scale-to-zero* (``jobs.pools``): every container of the job's app is gone at most
     ``DRAIN_BOUND_S`` (30 s) after its class was drained (the drain start in the job's pool
     file; Modal's lifetime from the watch, matched to the job report's per-container rows by
     task id), instead of the 300 s window. The measured gap per container is printed and
     written to the JSON report: this answers whether a lowered ``scaledown_window`` reaches a
     container that is already idle;
   - *caption pre-warm* (balanced): the ``pool_prewarm_plan`` decision is recorded (pool file
     ``prewarm_plans``) as ``prewarm``, it started, and the first caption container started
     within 120 s of the planned start;
   - *batch-fill fields* (balanced): two-phase gates, batch policy and resource samples (peak
     RSS, cores, GPU memory when there is a GPU) in the segment / caption shard audits and in
     each container's startup facts (``jobs.containers.read_startup_facts``, written by the
     fake balanced workers on Modal);
   - *compile-cache provenance per shard* (Fix-13): checked when any shard records it,
     otherwise reported as a skipped GAP;

   then ``modal app stop`` for the apps it deployed and
   deletion of this run's two Volumes and its lease/watch entries (``--keep`` skips the Volume
   deletion). The Volumes are named per run, so a cleanup never touches another H0 run's
   (review 8b L3).

Job ids are fake-models job ids (``h0-<run_id>-<job>.fake``, ``jobs.store.fake_job_id``), the
ones ``flora run --fake-models`` submits under.

``--dry-run`` plans everything offline (configs, fake app plans, the real-vs-fake hashes) and
contacts Modal only to read the golden manifest (nothing is deployed or written).
CPU only.
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
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOLDEN_VOLUME = "flora-golden"
GOLDEN_ROOT = "golden/v1"
STORE_VOLUME = "flora-h0"  # name prefixes: each run has its own Volumes (store_volume(run_id))
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


@dataclass(frozen=True)
class JobSpec:
    """One H0 job: its data directory (``EPISODES`` key), preset and injected faults."""

    name: str
    data: str
    preset: str = "exact"
    preempt: str | None = None  # shard kind preempted once (before its 2nd episode)
    poison_at: str | None = None  # "any" (exact: every worker) or a balanced shard kind

    @property
    def embodiment(self) -> str:
        return "human" if self.data == "human" else "yam"

    @property
    def determinism(self) -> str:
        """``deterministic`` applies to exact segcap only (the config refuses it otherwise)."""
        return DETERMINISM if self.preset == "exact" else "nvidia"

    def lost(self, stage: str) -> bool:
        """Whether the poison episode has no record of ``stage``."""
        if self.poison_at is None:
            return False
        if self.poison_at == "any":
            return True
        return stage == "caption" if self.poison_at == "caption" else stage in ("segment",
                                                                                 "caption")

    def faults(self, poison: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.preempt:
            out["preempt"] = {self.preempt: PREEMPT_AT}
        if self.poison_at == "any":
            out["poison"] = [poison]
        elif self.poison_at:
            out["poison"] = {self.poison_at: [poison]}
        return out


JOBS: tuple[JobSpec, ...] = (
    JobSpec("human", "human"),
    JobSpec("yam", "yam", preempt=PREEMPT_SHARD, poison_at="any"),
    JobSpec("bal-human", "human", "balanced"),
    JobSpec("bal-yam-seg", "yam", "balanced", preempt="segment", poison_at="segment"),
    JobSpec("bal-yam-cap", "yam", "balanced", preempt="caption", poison_at="caption"),
)
JOB_SPECS = {j.name: j for j in JOBS}


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


def store_volume(run_id: str) -> str:
    """This run's job-store Volume (deleted at cleanup unless --keep)."""
    return f"{STORE_VOLUME}-{run_id}"


def media_volume(run_id: str) -> str:
    """This run's media Volume (``input.stage_to``; deleted at cleanup unless --keep)."""
    return f"{MEDIA_VOLUME}-{run_id}"


def job_id_for(run_id: str, job: str) -> str:
    """The job id ``flora run --fake-models`` submits under (``<project>.fake``)."""
    from flora_deploy.jobs.store import fake_job_id

    return fake_job_id(f"h0-{run_id}-{job}")


def config_text(*, project: str, data_dir: str, embodiment: str, stages: tuple[str, ...],
                store_volume: str, media_volume: str, run_id: str, budget_usd: float = 20.0,
                resources: dict[str, str] | None = None, tag: str = "flora-h0",
                header: str = "H0 rehearsal; generated by scripts/h0_rehearsal.py",
                preset: str = "exact", derived: dict[str, str] | None = None) -> str:
    """One ``flora.yaml`` for an ``mp4_dir`` job staged on ``media_volume``. Shared with
    ``scripts/wave_h.py``, so H0 rehearses exactly Wave H's configs. ``preset: balanced`` sets
    ``allow_uncertified: true`` (no certificate exists yet). ``derived``: ``weights.derived``
    receipts, as ``flora prepare`` records them (``config.receipts``)."""
    from flora_deploy.config.presets import get_preset, shard_kinds_for

    kinds = shard_kinds_for(get_preset(preset), list(stages))
    res = {k: "{max_containers: 1}" for k in kinds}
    res.update(resources or {})
    res_lines = "\n".join(f"  {k}: {v}" for k, v in res.items())
    extra = "allow_uncertified: true\n" if preset != "exact" else ""
    weights = ""
    if derived:
        weights = "weights:\n  derived:\n" + "".join(
            f"    {k}: {v}\n" for k, v in sorted(derived.items()))
    return f"""# {header} ({data_dir})
project: {project}
license_ack: research-only
modal:
  environment: main
  max_gpus: 4
  budget_usd: {budget_usd:g}
  tags: {{{tag}: "{run_id}"}}
preset: {preset}
{extra}{weights}input:
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
    spec = JOB_SPECS[job]
    text = config_text(project=f"h0-{run_id}-{job}", data_dir=spec.data,
                       embodiment=spec.embodiment, stages=STAGES,
                       store_volume=store_volume(run_id), media_volume=media_volume(run_id),
                       run_id=run_id, preset=spec.preset)
    path = work / f"flora-{job}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def plans(cfg_path: Path, mode: str | None = None) -> dict[str, Any]:
    """The fake app plan and, for comparison, the real ones (nvidia / deterministic). ``mode``
    (default: the job's; ``deterministic`` for exact) is the deployed determinism mode; the
    other-mode plans exist for exact only (balanced has no determinism mode)."""
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import FAKE_RECEIPT, plan_app

    cfg = load_config(cfg_path)
    mode = mode or (DETERMINISM if cfg.preset == "exact" else "nvidia")
    receipts = {"captioner_bf16": FAKE_RECEIPT, "tempseg_merged": FAKE_RECEIPT}
    fake = plan_app(cfg, determinism=mode, fake_models=True)
    with standin_pins():  # comparison plans only, never deployed
        real = plan_app(cfg, determinism=mode, derived_receipts=receipts)
        out = {"cfg": cfg, "fake": fake, "real": real, "real_nvidia": None, "fake_nvidia": None}
        if cfg.preset == "exact":
            out["real_nvidia"] = plan_app(cfg, determinism="nvidia", derived_receipts=receipts)
            out["fake_nvidia"] = plan_app(cfg, determinism="nvidia", fake_models=True)
    return out


@contextlib.contextmanager
def standin_pins() -> Iterator[None]:
    """A stand-in content digest for every unpinned derived weight, for the *real* comparison
    plans only (``plan_app`` refuses an unpinned one, review 10 M1; these plans are compared
    with the fake app, never deployed)."""
    from dataclasses import replace

    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

    saved = dict(DERIVED_PINS_BY_KEY)
    for k, p in saved.items():
        if p.content_sha256 is None:
            DERIVED_PINS_BY_KEY[k] = replace(p, content_sha256="0" * 63 + "1")
    try:
        yield
    finally:
        DERIVED_PINS_BY_KEY.clear()
        DERIVED_PINS_BY_KEY.update(saved)


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

    @property
    def spec(self) -> JobSpec:
        return JOB_SPECS[self.job]

    def check(self, name: str, ok: bool, detail: str = "", *, gap: bool = False) -> None:
        self.checks.append(Check(name, bool(ok), detail, gap))


_LOCK = threading.Lock()  # setup runs jobs in threads: one writer at a time


def sh(args: list[str], *, cwd: Path, timeout: float = 900, check: bool = True,
       log: Path | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(args)}", flush=True)
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-12:])
    with _LOCK:
        if log is not None:
            with log.open("a", encoding="utf-8") as f:
                f.write(f"$ {' '.join(args)}\n{out}\n[rc={p.returncode}]\n")
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

    return RemoteVolumeStore(modal.Volume.from_name(store_volume(run_id), version=2), run_id)


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
    spec = r.spec
    if spec.poison_at is None:
        poison = None
    want_state = "completed_with_errors" if poison else "done"
    r.check("end state", r.status.get("state") == want_state,
            f"state={r.status.get('state')} (want {want_state}), flora run rc={r.run_rc}")
    job = sdk.open_job(r.job_id, store=p.store.uri, environment=p.environment)
    summary = job.summary() or {}
    store = job.store
    manifest = job._deps().manifests.load()
    manifest_uids = {e.uid for e in manifest.episodes}
    def want(stage: str) -> set[str]:
        return manifest_uids - ({poison} if poison and spec.lost(stage) else set())

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
        w = want(stage)
        r.check(f"{stage}: no lost/duplicate records",
                set(uids) == w and not dup,
                f"{len(uids)} uids (want {len(w)}), missing "
                f"{sorted(w - set(uids))}, extra {sorted(set(uids) - w)}, dup {dup}")
        certs = {x.provenance.certificate for x in recs}
        models = {v for x in recs for v in x.provenance.models.values()}
        r.check(f"{stage}: records marked fake", certs <= {"fake-models"} and models <= {"fake"}
                and bool(recs), f"certificates {certs}, models {models}")
    stages_sum = summary.get("stages", {})
    excused = {s: c.get("excused", 0) for s, c in stages_sum.items()}
    if poison:
        pills = summary.get("poison_pills", [])
        r.check("poison excluded (exactly)",
                all(n == int(spec.lost(s)) for s, n in excused.items())
                and any(poison in json.dumps(x) for x in pills),
                f"excused {excused}, poison pills {pills}")
        if spec.preset == "balanced":  # excluded downstream with a reason, counted once
            up = {s: c.get("excused_upstream", 0) for s, c in stages_sum.items()}
            heads = sorted({str(x).split(":")[0] for x in summary.get("problems", [])})
            r.check("upstream exclusion counted once",
                    up.get("caption", 0) == int(spec.poison_at == "segment")
                    and heads == [spec.poison_at],
                    f"excused_upstream {up}, problems {summary.get('problems')}")
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
    first = "segcap" if "segcap" in p.worker_classes else "segment"
    env = p.object(p.worker_classes[first]).env
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
    mode = spec.determinism
    r.check("determinism forwarded (env, hash, worker engine)",
            env.get("FLORA_DETERMINISM") == mode and
            json.loads(env["FLORA_OUTPUT_SETTINGS"])["determinism"] == mode and
            (set(modes) == {mode} or (first == "segment" and not modes)),
            f"env {env.get('FLORA_DETERMINISM')}, segcap audit modes {dict(modes)}")
    r.check("instruction text flows into reward requests (terminal audits)",
            reward_audits >= len(want("reward.terminal")) and reward_tasks_ok,
            f"{reward_audits} terminal audit rows")
    r.check("media root resolved", env.get("FLORA_MEDIA_ROOT") ==
            f"/media/{media_volume(run_id)}/{run_id}/{spec.data}" and bool(records.get("segment")),
            f"FLORA_MEDIA_ROOT={env.get('FLORA_MEDIA_ROOT')}")
    if spec.preempt:
        markers = [f for f in files if "/attempts/" in f and f"/{spec.preempt}/" in f]
        preempted = 0
        for f in markers:
            with contextlib.suppress(ValueError):
                preempted += bool(json.loads(store.get_bytes(f)).get("preempted"))
        fired = any("fake_faults/fired/" in f for f in files)
        r.check("simulate_preemption on a worker (marked preempted, shard still done)",
                fired and preempted >= 1, f"fired={fired}, {len(markers)} markers on "
                f"{spec.preempt}, {preempted} preempted")
    r.summary = summary  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------------------------
# Projected cost (printed before anything is deployed)

# Projection model (CPU fake workers; nothing here is measured): each container of a worker
# class lives a cold start plus the class's active span, then the drained tail (one orchestrator
# poll + the 2 s window). Every class is assumed active for PROJ_ACTIVE_S (conservative: segment
# is active for a fraction of the job). One container per shard kind of a class, up to
# max_containers, plus one for the preempted kind.
PROJ_STARTUP_S = 60.0
PROJ_ACTIVE_S = 480.0
PROJ_JOB_WALL_S = 900.0  # the orchestrator's (0.125-core) lifetime per job
PROJ_UNDRAINED_TAIL_S = 300.0  # jobs.settings.WORKER_SCALEDOWN_S: what the drain removes


def projected_cost(plans_by_job: dict[str, Any], specs: dict[str, JobSpec] | None = None
                   ) -> dict[str, Any]:
    """``{"usd", "usd_undrained", "jobs": {job: usd}, "model"}`` from each job's fake app plan
    (its containers' CPU / memory requests at ``jobs.budget`` list prices)."""
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.jobs.orchestrator import POLL_INTERVAL_S
    from flora_deploy.jobs.pools import (
        DRAINED_SCALEDOWN_S,
        ORCHESTRATOR_CORES,
        ORCHESTRATOR_MEMORY_GIB,
        ORCHESTRATOR_PREEMPTIBLE,
    )

    specs = specs or JOB_SPECS
    rates = pinned_rates()
    tail = POLL_INTERVAL_S + DRAINED_SCALEDOWN_S
    jobs: dict[str, float] = {}
    undrained = 0.0
    for job, plan in plans_by_job.items():
        spec = specs[job]
        usd = PROJ_JOB_WALL_S * rates.per_second(None, 0, ORCHESTRATOR_CORES,
                                                 ORCHESTRATOR_MEMORY_GIB,
                                                 preemptible=ORCHESTRATOR_PREEMPTIBLE)
        by_cls: dict[str, list[str]] = {}
        for kind, cls in plan.worker_classes.items():
            by_cls.setdefault(cls, []).append(kind)
        for cls, kinds in by_cls.items():
            o = plan.object(cls)
            mem = o.memory[0] if isinstance(o.memory, (tuple, list)) else o.memory
            per_s = rates.per_second(None, 0, float(o.cpu or 0.0), float(mem or 0) / 1024.0,
                                     preemptible=not o.nonpreemptible)
            n = min(int(o.max_containers or len(kinds)), len(kinds)) + int(spec.preempt in kinds)
            usd += n * (PROJ_STARTUP_S + PROJ_ACTIVE_S + tail) * per_s
            undrained += n * (PROJ_UNDRAINED_TAIL_S - tail) * per_s
        jobs[job] = round(usd, 4)
    total = sum(jobs.values())
    return {"usd": round(total, 3), "usd_undrained": round(total + undrained, 3), "jobs": jobs,
            "model": {"startup_s": PROJ_STARTUP_S, "active_s": PROJ_ACTIVE_S,
                      "drained_tail_s": tail, "orchestrator_s": PROJ_JOB_WALL_S}}


# ---------------------------------------------------------------------------------------------
# New-behaviour checks: job-end drain, caption pre-warm, batch-fill fields, cache provenance

DRAIN_BOUND_S = 30.0  # a drained container must be gone this long after its class's drain
WATCH_INTERVAL_S = 5.0  # Modal TaskList poll period (the resolution of every measured end)
WATCH_TAIL_S = DRAIN_BOUND_S + 15.0  # keep watching this long after the last job returned
PREWARM_TOL_S = 120.0  # pre-warmed caption container start vs the planned start


def modal_task_list(env: str = "main") -> list[dict[str, Any]]:
    """Modal's running containers in ``env`` (``modal container list``'s ``TaskList``):
    ``{"task_id", "app_id", "app", "started_at"}``. The task id is the ``MODAL_TASK_ID`` a
    worker records as its container id (``jobs.containers``). Metadata only: no compute."""
    from modal._utils.async_utils import synchronizer
    from modal.client import _Client
    from modal_proto import api_pb2

    @synchronizer.create_blocking
    async def _list() -> list[dict[str, Any]]:
        client = await _Client.from_env()
        res = await client._stub.TaskList(api_pb2.TaskListRequest(environment_name=env))
        return [{"task_id": t.task_id, "app_id": t.app_id, "app": t.app_description,
                 "started_at": float(t.started_at) or None} for t in res.tasks]

    return _list()


class ContainerWatch:
    """Polls Modal's running containers every ``interval_s`` and keeps, per container of an app
    ``match`` accepts: its app, Modal's start time, first / last time seen and ``ended_at`` (the
    first poll it was missing from: an upper bound at ``interval_s`` resolution). This is the
    observed lifetime. ``flora report``'s container ``seconds`` are derived from the recorded
    drains, not observed, so they cannot tell whether a lowered window reached a container that
    was already idle."""

    def __init__(self, match: Callable[[str], bool], *, interval_s: float = WATCH_INTERVAL_S,
                 list_fn: Callable[[], list[dict[str, Any]]] | None = None,
                 clock: Callable[[], float] = time.time):
        self.match, self.interval_s, self.clock = match, interval_s, clock
        self.list_fn = list_fn or modal_task_list
        self.seen: dict[str, dict[str, Any]] = {}
        self.errors = 0
        self.last_poll: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll(self) -> None:
        try:
            tasks = self.list_fn()
        except Exception:  # noqa: BLE001 - a missed poll only widens the resolution
            self.errors += 1
            return
        now = self.clock()
        live = set()
        for t in tasks:
            if not self.match(str(t.get("app") or "")):
                continue
            tid = str(t["task_id"])
            live.add(tid)
            rec = self.seen.setdefault(tid, {"task_id": tid, "app": t.get("app"),
                                             "started_at": t.get("started_at"),
                                             "first_seen": now, "ended_at": None})
            rec["last_seen"] = now
            rec["started_at"] = rec.get("started_at") or t.get("started_at")
            rec["ended_at"] = None
        for tid, rec in self.seen.items():
            if tid not in live and rec["ended_at"] is None:
                rec["ended_at"] = now
        self.last_poll = now

    def alive(self) -> list[str]:
        return sorted(t for t, r in self.seen.items() if r["ended_at"] is None)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.interval_s)

    def start(self) -> ContainerWatch:
        self._thread = threading.Thread(target=self._run, name="h0-watch", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
        self.poll()

    def wait_drained(self, until: float, *, sleep: Callable[[float], Any] = time.sleep) -> None:
        """Return once no watched container is alive, or at ``until`` (wall clock)."""
        while self.clock() < until:
            if self.seen and not self.alive():
                return
            sleep(min(self.interval_s, max(until - self.clock(), 0.0)))


def drain_rows(app: str, report_rows: list[dict[str, Any]], seen: dict[str, dict[str, Any]],
               drains: dict[str, tuple[tuple[float, float | None], ...]],
               classes: dict[str, str], *, observed_until: float | None,
               bound_s: float = DRAIN_BOUND_S) -> list[dict[str, Any]]:
    """Per container of ``app`` that Modal ran: when its class had no work left (``drain_at``)
    and how long after that it was gone (``gap_s``).

    A worker container (in ``report_rows``, ``flora report``'s per-container rows: container id
    = Modal task id) is measured from the latest drain start of its class at or before its end
    (``drains``: kind -> ``(start, end | None)``, the orchestrator's record). One that ended
    before any drain (preempted, replaced) has no gap and passes. Any other container (the
    orchestrator, a worker that ran no shard) is measured from the job's last drain start (the
    job-end drain). A container still alive at ``observed_until`` gets the lower bound
    ``observed_until - drain_at`` (``alive``)."""
    kind_of = {str(r["container_id"]): r.get("kind") for r in report_rows}
    starts_by_cls: dict[str, list[float]] = {}
    for kind, ivs in drains.items():
        starts_by_cls.setdefault(classes.get(kind, kind), []).extend(float(a) for a, _ in ivs)
    every = sorted(a for v in starts_by_cls.values() for a in v)
    out = []
    for tid, rec in sorted(seen.items()):
        if rec.get("app") != app:
            continue
        kind = kind_of.get(tid)
        end = rec.get("ended_at")
        alive = end is None
        t_end = observed_until if alive else float(end)
        if kind is not None:
            role = classes.get(kind, kind)
            cands = [a for a in starts_by_cls.get(role, ()) if t_end is None or a <= t_end]
        else:
            role = "other"
            cands = every[-1:]
        drain = max(cands) if cands else None
        gap = round(t_end - drain, 1) if drain is not None and t_end is not None else None
        if gap is not None:
            ok = gap <= bound_s
        else:  # a worker that ended before its class drained; anything else is unmeasurable
            ok = kind is not None and not alive
        out.append({"container_id": tid, "kind": kind, "class": role, "drain_at": drain,
                    "ended_at": end, "alive": alive, "gap_s": gap, "ok": bool(ok)})
    return out


def drain_check(rows: list[dict[str, Any]], bound_s: float = DRAIN_BOUND_S
                ) -> tuple[bool, str]:
    """``(ok, detail)`` over :func:`drain_rows`: every container gone within ``bound_s`` of its
    class's drain, and at least one worker container measured against one."""
    measured = [r for r in rows if r["kind"] is not None and r["gap_s"] is not None]
    parts = []
    for r in rows:
        g = "ended before any drain" if r["gap_s"] is None else (
            f"{'>=' if r['alive'] else ''}{r['gap_s']:.0f} s")
        parts.append(f"{r['container_id'][-8:]} {r['kind'] or r['class']}: {g}")
    ok = bool(measured) and all(r["ok"] for r in rows)
    head = (f"bound {bound_s:.0f} s, {len(measured)} worker container(s) measured" if measured
            else "no worker container measured against a drain")
    return ok, head + ("; " + ", ".join(parts) if parts else "")


def prewarm_check(plans: dict[str, dict[str, Any]], prewarms: dict[str, Any],
                  report_rows: list[dict[str, Any]], seen: dict[str, dict[str, Any]], *,
                  kind: str = "caption", tol_s: float = PREWARM_TOL_S) -> tuple[bool, str]:
    """``kind``'s ``pool_prewarm_plan`` is recorded and says ``prewarm``, the pre-warm started,
    and ``kind``'s first container started within ``tol_s`` of the planned start
    (``max(start_at, decision time)``: a planned start in the past means "now"; the container's
    start is Modal's, else its first attempt marker)."""
    from flora_deploy.jobs.containers import parse_utc

    plan = plans.get(kind)
    if not plan:
        return False, f"no pool_prewarm_plan recorded for {kind} (plans: {sorted(plans)})"
    if plan.get("decision") != "prewarm":
        return False, f"{kind} pre-warm skipped: {plan.get('reason')}"
    planned = max(float(plan["start_at"]), float(plan.get("t", plan["start_at"])))
    pw = prewarms.get(kind)
    if not pw:
        return False, f"{kind} pre-warm planned at {planned:.0f} but never started"
    starts = []
    for r in report_rows:
        if r.get("kind") != kind:
            continue
        cid = str(r["container_id"])
        t = (seen.get(cid) or {}).get("started_at") or parse_utc(r.get("started_utc"))
        if t is not None:
            starts.append((float(t), cid))
    if not starts:
        return False, f"no {kind} container found (pre-warm started at {float(pw[0]):.0f})"
    t0, cid = min(starts)
    delta = t0 - planned
    return abs(delta) <= tol_s, (
        f"plan saves {plan.get('saves_s')} s; pre-warm started {float(pw[0]) - planned:+.0f} s "
        f"after the planned start, first {kind} container {cid[-8:]} {delta:+.0f} s "
        f"(tolerance {tol_s:.0f} s)")


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def resources_ok(res: Any) -> tuple[bool, str]:
    """Plausible container resources (``observe.sampler.ContainerSampler``): cores >= 1, a peak
    host RSS in (0, 64] GiB, the process tree's high-water mark at least the process's RSS; GPU
    memory, when sampled, used <= total (CPU fake workers have none: reported, not required)."""
    if not isinstance(res, dict) or "error" in res:
        return False, f"resources {res!r}"
    bad = []
    if not (_num(res.get("cores")) and res["cores"] >= 1):
        bad.append(f"cores={res.get('cores')}")
    peak = res.get("rss_peak_gib")
    if not (_num(peak) and 0 < peak <= 64):
        bad.append(f"rss_peak_gib={peak}")
    tree = res.get("tree_rss_hwm_gib")
    if tree is not None and not (_num(tree) and tree >= (res.get("rss_max_gib") or 0) * 0.99):
        bad.append(f"tree_rss_hwm_gib={tree}")
    gpu = "GPU memory n/a (CPU)"
    if "gpu_mem_used_max_mib" in res:
        used, total = res.get("gpu_mem_used_max_mib"), res.get("gpu_mem_total_mib")
        if not (_num(used) and _num(total) and 0 <= used <= total):
            bad.append(f"gpu_mem {used}/{total}")
        else:
            gpu = f"GPU memory {used:.0f}/{total:.0f} MiB"
    return not bad, (", ".join(bad) if bad else
                     f"cores {res.get('cores')}, peak RSS {peak} GiB, {gpu}")


def policy_ok(pol: Any, kind: str) -> bool:
    from flora_deploy.engines.batch_policy import POLICY_SCHEMA

    return (isinstance(pol, dict) and pol.get("schema") == POLICY_SCHEMA
            and pol.get("kind") == kind and isinstance(pol.get("digest"), str)
            and len(pol["digest"]) == 16 and int(pol.get("infer_episodes") or 0) >= 1
            and int(pol.get("prefetch_episodes") or 0) >= 1)


def batch_fill_check(audits: list[dict[str, Any]], startup: dict[str, dict[str, Any]],
                     kinds: tuple[str, ...] = ("segment", "caption")) -> tuple[bool, str]:
    """The batch-fill fields (``workers_impl.segment``). Per shard audit row of ``kinds``: a
    valid batch policy, ``phased`` (two-phase episodes), gate peaks inside their bounds with at
    least one episode through each phase, batcher rounds and plausible resources; every episode
    audit row of the kind names that policy's digest. Per container of those kinds: startup
    facts (``jobs.containers.read_startup_facts``) with the policy, ``phased``, an executor
    size and plausible resources."""
    bad: list[str] = []
    rows: dict[str, int] = {}
    digests: dict[str, set[str]] = {}
    note = ""
    for a in audits:
        if not a.get("shard_audit") or a.get("stage") not in kinds:
            continue
        k = a["stage"]
        rows[k] = rows.get(k, 0) + 1
        sid = str(a.get("shard_id"))[-14:]
        pol = a.get("batch_policy")
        if not policy_ok(pol, k):
            bad.append(f"{sid}: batch_policy {pol!r}")
            continue
        digests.setdefault(k, set()).add(pol["digest"])
        if a.get("phased") is not True:
            bad.append(f"{sid}: phased={a.get('phased')}")
        g = a.get("gates") or {}
        if int(a.get("episodes_run") or 0) > 0:
            if not (1 <= int(g.get("peak_preparing") or 0) <= int(g.get("prefetch") or 0)
                    and 1 <= int(g.get("peak_inferring") or 0) <= int(g.get("infer") or 0)):
                bad.append(f"{sid}: gates {g}")
            if not (a.get("batcher") or {}).get("rounds"):
                bad.append(f"{sid}: batcher {a.get('batcher')}")
        ok, why = resources_ok(a.get("resources"))
        if not ok:
            bad.append(f"{sid}: {why}")
        note = note or why
    bad += [f"no {k} shard audit row" for k in kinds if not rows.get(k)]
    for a in audits:  # episode rows name their shard's policy digest
        k = a.get("stage")
        if a.get("shard_audit") or k not in kinds or "uid" not in a:
            continue
        if a.get("batch_policy") not in digests.get(k, set()):
            bad.append(f"{a['uid']}: episode batch_policy {a.get('batch_policy')!r}")
    facts = {cid: f for cid, f in startup.items() if f.get("kind") in kinds}
    bad += [f"no {k} startup facts" for k in kinds
            if not any(f.get("kind") == k for f in facts.values())]
    for cid, f in sorted(facts.items()):
        st = f.get("startup") or {}
        if not policy_ok(st.get("batch_policy"), f["kind"]) or st.get("phased") is not True \
                or not (_num(st.get("executor_workers")) and st["executor_workers"] >= 1):
            bad.append(f"{cid[-8:]} startup facts {sorted(st)}")
            continue
        ok, why = resources_ok(st.get("resources"))
        if not ok:
            bad.append(f"{cid[-8:]} startup: {why}")
    detail = (f"shard audits {rows}, startup facts from {len(facts)} container(s)"
              + (f"; {note}" if note else ""))
    return not bad, detail + ("; " + "; ".join(bad[:6]) if bad else "")


CACHE_KEYS = ("compile_cache", "cache_provenance")


def cache_provenance_check(audits: list[dict[str, Any]], receipts: list[dict[str, Any]],
                           kinds: tuple[str, ...] = ("segment", "caption")
                           ) -> tuple[bool | None, str]:
    """Compile-cache provenance per shard (Fix-13). ``None`` (skipped) when no shard audit row
    or receipt of ``kinds`` carries one of :data:`CACHE_KEYS`; otherwise every such shard must,
    naming a ``state`` (or ``source``)."""
    def prov(d: dict[str, Any]) -> Any:
        return next((d[k] for k in CACHE_KEYS if d.get(k) is not None), None)

    shards: dict[str, Any] = {}
    for d in [a for a in audits if a.get("shard_audit") and a.get("stage") in kinds] + \
            [x for x in receipts if x.get("kind") in kinds]:
        sid = str(d.get("shard_id"))
        if shards.get(sid) is None:
            shards[sid] = prov(d)
    if not any(v is not None for v in shards.values()):
        return None, "no shard records it (Fix-13 compile-cache provenance not merged): skipped"
    missing = sorted(s for s, v in shards.items() if v is None)

    def named(v: Any) -> str | None:
        if isinstance(v, dict):
            return v.get("state") or v.get("source")
        return v if isinstance(v, str) and v else None

    unnamed = sorted(s for s, v in shards.items() if v is not None and not named(v))
    states = Counter(str(named(v)) for v in shards.values() if v is not None)
    return (not missing and not unnamed,
            f"{len(shards) - len(missing)}/{len(shards)} shards, states {dict(states)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; no state {unnamed}" if unnamed else ""))


def job_audits(store: Any, root: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in store.list(f"{root}/"):
        if "/audit/" not in f:
            continue
        for line in store.get_bytes(f).decode().splitlines():
            with contextlib.suppress(ValueError):
                a = json.loads(line)
                if isinstance(a, dict):
                    out.append(a)
    return out


def assert_new(r: JobRun, watch: ContainerWatch | None) -> None:
    """The checks of the behaviours H0 had never run on Modal (module docstring, step 6)."""
    from dataclasses import asdict

    from flora_deploy import sdk
    from flora_deploy.jobs import attempts as A
    from flora_deploy.jobs import progress as P
    from flora_deploy.jobs.containers import read_startup_facts
    from flora_deploy.jobs.pools import read_drains, read_prewarm_plans
    from flora_deploy.jobs.store import JobPaths

    p = r.plan
    job = sdk.open_job(r.job_id, store=p.store.uri, environment=p.environment)
    store = job.store
    root = JobPaths(r.job_id).root
    drains, prewarms = read_drains(store, root, p.config_hash)
    plans = read_prewarm_plans(store, root, p.config_hash)
    try:
        rows = list(job.report()["spend"]["containers"])
    except Exception as e:  # noqa: BLE001 - reported as a failed check
        rows = []
        r.check("job report: per-container rows", False, repr(e))
    seen = watch.seen if watch is not None else {}
    if watch is not None:
        dr = drain_rows(p.app_name, rows, seen, drains, dict(p.worker_classes),
                        observed_until=watch.last_poll)
        r.drain = dr  # type: ignore[attr-defined]
        ok, detail = drain_check(dr)
        r.check(f"job-end scale-to-zero: containers gone <= {DRAIN_BOUND_S:.0f} s after drain",
                ok, detail + (f" ({watch.errors} missed polls)" if watch.errors else ""))
    else:
        r.check("job-end scale-to-zero", False, "no container watch (Modal TaskList)")
    if r.spec.preset != "balanced":
        return
    ok, detail = prewarm_check(plans, prewarms, rows, seen)
    r.check("caption pre-warm: planned, started, container on time", ok, detail)
    audits = job_audits(store, root)
    ok, detail = batch_fill_check(audits, read_startup_facts(store, r.job_id))
    r.check("batch-fill fields (two-phase, policy, resources) in audits + startup facts",
            ok, detail)
    receipts: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        manifest = job._deps().manifests.load()
        jf = A.JobFiles(store, r.job_id, p.config_hash, stage_hashes=dict(manifest.stage_hashes))
        for s in manifest.shards:
            rec = P.read_receipt(jf, s.kind, s.shard_id)
            if rec is not None:
                receipts.append(asdict(rec))
    okc, detail = cache_provenance_check(audits, receipts)
    r.check("compile-cache provenance per shard", bool(okc), detail, gap=okc is None)


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
            if str(key) in {f"orchestrator/{j}" for j in job_ids}:  # this run's only
                leases.pop(key)
                notes.append(f"lease {key} removed")
    except Exception as e:  # noqa: BLE001
        notes.append(f"lease cleanup: {e!r}")
    for d in watch_dicts:  # this run's fake apps' own Dicts (named per app: review 9a M2)
        if not d.startswith("flora-watch-fake-"):  # never a real app's watch Dict
            notes.append(f"watch Dict {d}: not a fake app's; kept")
            continue
        p = subprocess.run(["modal", "dict", "delete", "--yes", d, "--env", env],
                           capture_output=True, text=True)
        notes.append(f"modal dict delete {d}: rc={p.returncode}")
    if not keep:
        for vol in (store_volume(run_id), media_volume(run_id)):
            p = subprocess.run(["modal", "volume", "delete", vol, "--yes", "--env", env],
                               capture_output=True, text=True)
            notes.append(f"modal volume delete {vol}: rc={p.returncode} "
                         f"{(p.stdout + p.stderr).strip()[-160:]}")
    return notes


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="plan only; deploy nothing")
    ap.add_argument("--keep", action="store_true", help="keep this run's two Volumes")
    ap.add_argument("--work", type=Path, help="work directory (default: a temp dir)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-wall-s", type=float, default=1500.0,
                    help="cancel the jobs if `flora run` has not returned by then")
    ap.add_argument("--json", type=Path, help="write the report here")
    ap.add_argument("--preset", choices=("exact", "balanced", "both"), default="balanced",
                    help="which jobs to run (default balanced: the 3 balanced jobs; both: "
                         "also the 2 exact ones)")
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

    runs: list[JobRun] = []
    selected = [s for s in JOBS if args.preset == "both" or s.preset == args.preset]
    needed = {s.data for s in selected}
    manifest = golden_manifest()
    eps = fetch_dataset(manifest, work, download=not args.dry_run,
                        episodes={k: v for k, v in EPISODES.items() if k in needed},
                        poison_job="yam" if "yam" in needed else None)
    for spec in selected:
        cfg_path = write_config(work, spec.name, run_id)
        runs.append(JobRun(spec.name, cfg_path, job_id_for(run_id, spec.name),
                           eps[spec.data]))
    projection = projected_cost({r.job: plans(r.cfg_path)["fake"] for r in runs})
    print(f"projected cost: ${projection['usd']:.2f} (${projection['usd_undrained']:.2f} if "
          f"idle pools kept their {PROJ_UNDRAINED_TAIL_S:.0f} s window); per job "
          f"{projection['jobs']}", flush=True)
    if args.dry_run:
        for r in runs:
            pl = plans(r.cfg_path) if (work / r.spec.data).is_dir() else None
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
    watch: ContainerWatch | None = None
    try:
        # media to its own Volume (input.stage_to), as Wave H stages it
        for data in sorted({r.spec.data for r in runs}):
            n = upload_dir(media_volume(run_id), work / data, f"{run_id}/{data}")
            print(f"[{data}] uploaded {n} files to {media_volume(run_id)}:/{run_id}/{data}",
                  flush=True)
        fl = flora_bin()

        def setup(r: JobRun) -> None:
            # index -> plan -> deploy for one job. Jobs touch only their own config, store
            # prefix and app, so they set up side by side (deploys dominate the wall clock).
            local_store = work / f"store-{r.job}"
            sh([fl, "index", "--local", "--config", str(r.cfg_path), "--job", r.job_id,
                "--store", f"file://{local_store}", "--out", str(work / f"summary-{r.job}.json")],
               cwd=work, log=log)
            n = mirror_index(local_store, r.job_id, run_id, store_volume(run_id))
            print(f"[{r.job}] mirrored {n} index files into {store_volume(run_id)}:/{run_id}",
                  flush=True)
            sh([fl, "plan", "--config", str(r.cfg_path), "--index",
                str(work / f"summary-{r.job}.json")], cwd=work, log=log)
            pl = plans(r.cfg_path)
            r.plan = pl["fake"]
            r.check("fake hashes differ from real (every stage, both modes)",
                    all(pl["fake"].stage_hashes[k] != pl["real"].stage_hashes[k]
                        for k in pl["fake"].stage_hashes)
                    and pl["fake"].app_name != pl["real"].app_name
                    and (pl["fake_nvidia"] is None
                         or pl["fake"].config_hash != pl["fake_nvidia"].config_hash),
                    f"fake {pl['fake'].app_name}, real {pl['real'].app_name}")
            r.check("fake workers on CPU", all(o.gpu is None for o in pl["fake"].objects),
                    ", ".join(f"{o.name}:{o.gpu}" for o in pl["fake"].objects))
            dep = sh([fl, "deploy", "--config", str(r.cfg_path), "--fake-models",
                      "--determinism", r.spec.determinism], cwd=work, log=log, timeout=1200)
            with _LOCK:
                app_names.append(r.plan.app_name)
                watch.append(r.plan.watch_dict)
                from flora_deploy.jobs.pools import pool_dict_name

                watch.append(pool_dict_name(r.plan.watch_dict))  # jobs.pools: same prefix
            r.check("deployed app == the client's plan", f"deployed {r.plan.app_name}" in
                    " ".join((dep.stdout + dep.stderr).split()), r.plan.app_name)
            if r.spec.preempt or r.spec.poison_at:
                from flora_deploy.modal_app.workers_app import fake_faults_path

                poison = poison_uid(local_store, r.job_id) if r.spec.poison_at else ""
                r.poison = poison or None  # type: ignore[attr-defined]
                faults = r.spec.faults(poison)
                remote_store(run_id).put_bytes(fake_faults_path(r.job_id),
                                               json.dumps(faults).encode())
                print(f"[{r.job}] faults: {json.dumps(faults)}")

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(runs)) as pool:
            for fut in [pool.submit(setup, r) for r in runs]:
                fut.result()  # re-raise the first setup failure

        # every container of this run's apps, as Modal sees it (the drain / pre-warm checks)
        watch = ContainerWatch(lambda app: app in app_names).start()
        procs = {}
        for r in runs:
            args_ = [fl, "run", "--config", str(r.cfg_path), "--fake-models", "--determinism",
                     r.spec.determinism, "--job", r.job_id, "--poll", "10"]
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
        # the job-end drains happen as the jobs end: give them the bound (and one poll) to act
        # before anything is stopped (`modal app stop` would end the containers itself)
        watch.wait_drained(time.time() + WATCH_TAIL_S)
        watch.stop()
        print(f"container watch: {len(watch.seen)} containers, still alive {watch.alive()}",
              flush=True)
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
            assert_job(r, run_id, getattr(r, "poison", None))
            try:
                assert_new(r, watch)
            except Exception as e:  # noqa: BLE001 - reported as a failed check
                r.check("new-behaviour checks ran", False, repr(e))
    except Exception as e:  # noqa: BLE001 - always clean up, then report
        import traceback

        traceback.print_exc()
        notes.append(f"aborted: {e!r}")
    finally:
        if watch is not None:
            watch.stop()
        notes += cleanup(app_names, run_id, [r.job_id for r in runs], watch, keep=args.keep)
        if not args.work:
            for data in EPISODES:
                shutil.rmtree(work / data, ignore_errors=True)

    report = {"run_id": run_id, "wall_s": round(time.time() - t0, 1), "apps": app_names,
              "projected": projection, "notes": notes, "jobs": {r.job: {
                  "job_id": r.job_id, "state": r.status.get("state"), "rc": r.run_rc,
                  "spend_usd": r.status.get("spend_usd"),
                  "drain": getattr(r, "drain", None),
                  "checks": [c.__dict__ for c in r.checks]} for r in runs}}
    print(f"\nH0 {run_id}: {report['wall_s']} s, apps {app_names}, projected "
          f"${projection['usd']:.2f}")
    ok = not any(n.startswith("aborted") for n in notes)
    for r in runs:
        print(f"  [{r.job}] {r.job_id}: state={r.status.get('state')} rc={r.run_rc} "
              f"spend(est)=${r.status.get('spend_usd', 0) or 0:.4f}")
        for c in r.checks:
            tag = "PASS" if c.ok else ("GAP " if c.gap else "FAIL")
            print(f"    {tag} {c.name}" + (f": {c.detail}" if c.detail else ""))
            ok &= c.ok or c.gap
        for d in getattr(r, "drain", None) or ():
            gap = "-" if d["gap_s"] is None else f"{'>=' if d['alive'] else ''}{d['gap_s']} s"
            print(f"      container {d['container_id']} {d['kind'] or d['class']}: "
                  f"gone {gap} after its drain")
    for n in notes:
        print(f"  note: {n}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str))
    print(f"log: {log}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
