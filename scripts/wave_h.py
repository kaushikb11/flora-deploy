"""Wave H-lite runbook, one embodiment (``human``) plus the lab checks (``docs/wave-h-runbook.md``).

Usage::

    uv run python scripts/wave_h.py --dry-run              # plan + estimates; spends nothing
    uv run python scripts/wave_h.py --dry-run --offline    # ... without reading the Volume
    uv run python scripts/wave_h.py --yes-spend            # run (resumes: skips done steps)
    uv run python scripts/wave_h.py --yes-spend --only balanced
    uv run python scripts/wave_h.py --report               # rewrite reports/wave-h/runbook.md
    uv run python scripts/wave_h.py --plan full --dry-run  # every v1 step (cannot certify)
    uv run python scripts/wave_h.py --plan certified --dry-run  # the certified run (golden/v2)
    uv run python scripts/wave_h.py --plan certified --embodiment yam --dry-run  # golden/v3-yam
    uv run python scripts/wave_h.py --plan recertify --dry-run  # f2: (b) repeated on all 12

**Plans** (``--plan``, :data:`PLANS`). ``balanced-only`` (the default; the user's decision of
2026-10-22) runs steps 0 -> a -> d and nothing else; (b), (e) and (f) are listed as "deferred to
the certified run" and never run. Its global cap is the sum of those three steps' upper
projections rounded up to $0.10 (:func:`plan_global_cap`). ``full`` runs every step under
:data:`GLOBAL_CAP_USD`. Both share the state file, so a later ``--plan full`` skips what
``balanced-only`` finished. ``full`` cannot certify: it stages one data dir per step, so its
oracle's and balanced's episode uids (``<data dir>--<episode id>``) never pair.

``certified`` (:class:`PlanSpec`, ``docs/wave-h-runbook.md`` "The certified run") makes
``balanced`` certified for ``human``: b (exact oracle on the 12 golden/v2 episodes, default
mode, H200) -> f (noise: 6 of them, the selection's 5 plus one spare, the oracle's config with
its own project, so its own job and app, and its data dir re-indexed with ``flora index --limit
6``, so the same stage hash: :func:`run_cert_noise`) -> d (balanced
on the 12, the real ``flora deploy`` / ``flora run``, then the **cold witness**: the same config
re-run on the noise subset with ``FLORA_COMPILE_CACHE=off``, one container per kind forced
cold) -> d2 (the lab, skipped only when its evidence exists **for the current engine
settings**: :func:`lab_already_done`) -> e (``flora certify`` per stage, with the witness,
then ``wave_h_finalize.py --write`` when both are ``certified``: :func:`run_flora_certify`).
Every job stages one data dir (:data:`CERT_DATA`). Its caps are each step's upper projection
rounded up to $0.05 (:func:`plan_caps`), its global cap their sum rounded up to $0.10; its
state, work dir and records live under ``reports/wave-h/certified/``.

Under the v4 (stage-isolated) caption certificate the certified plan also runs **(g)**
``caption_iso`` after (b) (:func:`run_caption_iso`): J2, a ``preset: balanced``, ``stages:
[caption]`` job on every episode whose caption shards read the oracle's own segment records
(``pipeline.caption_segments``: the JSONL staged byte-exact into the job store,
:func:`stage_oracle_segments`), then J2w, the same config forced cold on the witness episodes.
:attr:`PlanSpec.witness` (the cold witnesses' episodes and the ``n-`` local ids) is separate
from :attr:`PlanSpec.noise` (the noise run's): yam's noise run covers all 12 and its (d) cold
witness (J1w) runs the segment stage alone (:attr:`PlanSpec.segment_only_witness`).

**Parallel steps** (``--max-parallel N``, default 3; certified only: the other plans are one
chain). b, f and d are independent and run at once; d2 waits for d, e for all four
(:data:`CERT_DEPS`, which says why). Each step runs in its own thread with its own heartbeat
and ``[<step>]`` log prefix, under the same stop rule as a sequential run, which also counts
the running steps: a starting step **reserves** its budget against the global cap (``spent +
every running step's reservation + its upper + the reserve for the steps still to start <=
global cap``), and its in-step deadlines use that reservation alone; on finish the
reservation becomes its tracked, then billed, spend. A step refused only while others run
waits for one to end. ``--max-parallel 1`` is the sequential run, unchanged. (e) certifies the
finished steps' jobs only (:func:`finished_view`). The dry run prints the expected wall-clock
both ways; ``--status`` the running steps and their reservations. ``--max-parallel`` counts
steps, not GPUs: b, f and d hold **4 GPUs** at once, 2 of them H200 (:data:`CERT_GPUS`);
``--max-gpus`` caps the GPUs held across the running steps' apps (default 4, the peak; e.g.
``--max-gpus H200=1`` runs f after b).

``recertify`` (human; the stage-isolated caption re-certification, order A) is its own state
(``reports/wave-h/recertified/``) on the certified run's ``run_id``, one step after the other.
Its step **f2** ``noise_all`` repeats that run's (b) on **all 12** of its episodes: (b)'s own
config file (``project`` renamed per attempt: its own job and app) and data dir, re-indexed with
no ``--limit``, nvidia mode, H200; the config hash and stage hashes are checked equal to (b)'s
recorded ones, in this checkout, before any app (:func:`noise_all_check`), so it runs before the
v4 code lands (once it landed the check refuses: f2 is then already done in this state). (b) is
read from ``reports/wave-h/certified/state.json`` (``--source-state``), never written, so the
certified run's finished steps keep their records. Then, on the v4 code (``"4" in
certkey.GENERATOR_VERSIONS``, :func:`require_code_landed`): **d'** J1 + the segment-only J1w,
**d2'** the Captioner lab only (TempSeg's saved result reused, :func:`run_recert_lab`; after
d', the certified run's order: the lab still reads the certified run's (d) segments), **g** J2
+ J2w, **e'** ``flora certify`` v4 (segment: J1 + J1w + noise-12; caption: J2 + J2w +
noise-12 with the digests, the staged oracle segments, J1's captions reported only, the
families and (b)'s old noise run as the repeat: :func:`certify_view`). Its jobs, configs and
apps take the ``rc-`` prefix (:attr:`PlanSpec.job_prefix`): it shares (b)'s run id, work dir
and store, so none may be (b)'s. Same stop rules, billing watermark and caps rule as
``certified``; its global cap is the uppers' sum rounded up to $0.10, never above the
user-approved :data:`RECERT_GLOBAL_CAP_USD` (f2 included) unless ``--global-cap`` raises it: a
done f2 counts at its actual cost (spent), the later steps at their uppers.

``--embodiment`` (``certified`` only; default ``human``) picks the certified plan's embodiment
(:data:`CERT_SPECS`): ``yam`` runs the same steps on golden/v3-yam (12 three-view episodes,
``docs/golden-manifest-v3-yam.md``) with its own data dir (``cert-yam``), state, work dir and
records (``reports/wave-h/certified-yam/``), so the two certified runs never share a state and
resume independently; its caps derive from its own projections, priced on the measured rates
scaled for three-view prompts (:func:`three_view_scale`). ``dex``, ``tianji`` and ``yam32`` have
no golden certification set yet and are refused.

Six ``human`` golden episodes (:data:`EPISODES`, ``docs/golden-manifest-v1.md``), one job and one
app per stage. Steps, in order (each with a projected-spend stop rule before it and a spend
check after it):

0. ``tempseg`` (CPU merge + one L40S check): build TempSeg with merger v2
   (``convert_app.py::tempseg --step merge``), run its equivalence check (``--step
   equivalence``), print the receipt's content digest (``flora prepare --show-receipt``) and
   **stop** (exit 4): the operator pins ``pins.DERIVED_PINS["tempseg-merged"].content_sha256``,
   commits, and re-runs. The step is done once the pin matches the receipt on the Volume; it
   never rebuilds a receipt it already reported. Every step that needs TempSeg refuses until
   then (:func:`require_tempseg_pin`);
a. ``preflight``: balanced preflight (CPU): derived weight dirs, receipts, EOS ids
   (``balanced_lab.preflight``);
b. ``exact_e2e``: the ``exact`` **oracle** on the 6 episodes through the real ``flora index ->
   plan -> deploy -> run`` path, in NVIDIA's **default mode** (``determinism: nvidia``), one
   H200 app, stopped as soon as its job is composed;
d. ``balanced``: ``preset: balanced`` (``allow_uncertified: true``) through the same real
   ``flora deploy`` / ``flora run`` on the same 6 episodes, taken from :data:`EPISODES` (the
   golden selection) directly, never from (b)'s state (one app: ``SegmentWorker`` on L40S,
   ``CaptionWorker`` on H100). Its startup is priced cold (:func:`cold_startup_s`: the
   measured cold startups as they are; a projection x :data:`COLD_COMPILE_FACTOR`). The deployed
   workers copy ``flora-cache:/balanced/<engine>`` to local disk and never write it back;
d2. ``lab``: **only against (d)'s finished job** (``--only lab`` retries it alone), the
   ``balanced_lab`` L0 input parity + Gate A on 6 captured requests (``run_tempseg``, then
   ``run_captioner`` on (d)'s segment records), on (d)'s episodes. Preconditions are checked
   before any container (:func:`lab_preconditions`: episodes, camera files at the lab's paths,
   segments), and a run that gets no request in is an error, not evidence. TempSeg's one call
   compiles two engines (baseline + plugin); the Captioner's lab is two calls, the baseline
   (L0, stage, Gate A requests) and then the plugin (Gate A on the staged requests), each its
   own container and deadline (:data:`LAB_CALLS`). The lab writes its own unkeyed
   ``flora-cache:/balanced/<engine>[-dtbias]`` (committed after each engine), which only later
   lab calls read: the deployed workers' caches are keyed (``balanced-cc/``), never the lab's;
e. ``certify``: offline provisional certification of ``balanced`` from (b) + (d) ($0);
f. ``noise``: (b)'s job **repeated** (same config file, staged data, stage hash, GPU and
   default mode; a new job id): the noise floor; (e) is re-evaluated. Its episodes are a subset
   of the oracle's (Fix-Cert H2) by being all of them: the input content is hashed, so a strict
   subset would be another stage hash, which ``flora certify`` refuses. (The certified plan's
   (f) has its own project, job and app, beside (b): :func:`run_cert_noise`.)

(The step letters keep their earlier names; the reward step was dropped with the yam job.)

**Derived weights** are resolved the way ``flora deploy`` resolves them
(:func:`resolve_receipts`): each needed receipt is read from the ``flora-weights`` Volume,
``config.receipts.check_pinned`` checks its content digest against ``pins.DERIVED_PINS`` and its
file sha256 is written into the generated config's ``weights.derived``; ``flora deploy`` /
``flora run`` then read it from there (``config.receipts.resolve``). An unpinned one is refused
with ``config.receipts.unpinned_message`` (``plan_app`` refuses it too, review 10 M1).

**Spend.** Every attempt of a step is kept (:class:`Attempt`); a step's spend is the sum over its
attempts, so a retry adds instead of overwriting. Tracked cost is container-seconds x list $/h
(``flora report`` for the jobs, the lab's own wall time), and while a step runs a heartbeat stores
``elapsed x the step's peak $/s`` so a killed process leaves its spend behind. Each attempt runs
its own app (:func:`app_name`), so billing is attributed per attempt. An attempt that was
killed, or whose ``flora report`` failed, is held until billing covers it (the watermark of the
billing rows' ``interval_start``, read again before every step for **every** app seen so far).
Until then it is charged its peak $/s over the time it ran plus its stop slack when its stop
was confirmed, else its budget (``_runbook.Attempt.bounded_usd``). An attempt that failed
**before it noted any app** started nothing and is charged $0 (review 10 L1). The global figure
is ``max(sum of charged attempts, billed delta of all apps)``; the global cap is checked on it.

**Stop rules** (``scripts/_runbook.py``: one ``budget_for`` and one ``decide`` for the stop rule,
the attempt's budget, its in-step deadlines, the dry run and ``--status``). An attempt may spend
``min(step cap, global cap - spent - the upper projections of the plan's later steps)``: the step
cap is a per-attempt ceiling. Before a step its **upper** projection (upper rate quantiles,
every cold start, the scale-down tail and :data:`CLIENT_SLACK_S` per app) must fit that. During
a step GPU wall time is bounded by ``(budget - slack x $/s) / $/s`` (``flora run --budget`` plus
a client-side deadline that cancels the job and stops the app; lab and conversion calls get the
same deadline as their timeout). After a step: the attempt's spend over its cap, or the total
over the global cap, stops the runbook. The dry run prints nominal and upper costs per step and
the decision a run takes; ``--status`` adds the holds, who approved the cap and the next command.

**State** is ``reports/wave-h/state.json`` (git-ignored: it names private episodes). Per job it
records what ``scripts/wave_h_finalize.py`` reads: ``state``, ``spend``, ``video_s``, ``cfg``,
``job_id``, the record JSONL paths (``records``: where ``wave_h_finalize.py --fetch`` exports
them) and per container ``kind``, ``gpu``, ``startup_s``, ``busy_s``, ``seconds`` (``flora
report``'s, ``jobs.containers``: derived from the attempt markers, shard receipts and call
ledgers; :func:`job_record` cuts them at the app stop);
the balanced step also records the derived content digests its workers verified
(``derived_content``). This script's own summary goes to ``reports/wave-h/runbook.md``
(git-ignored); ``docs/wave-h-report.md`` is ``wave_h_finalize.py``'s. A step that finished is
never re-run; a stopped, failed or killed step is retried on the next ``--yes-spend`` (a killed
step's apps are stopped first).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _runbook as rb  # noqa: E402 - state, billing, heartbeat, stop rules (shared)
import h0_rehearsal as h0  # noqa: E402 - the same staging and configs as H0
import spend as _spend  # noqa: E402 - the project budget (FLORA_PROJECT_BUDGET_USD)

STATE = ROOT / "reports" / "wave-h" / "state.json"
REPORT = ROOT / "reports" / "wave-h" / "runbook.md"  # docs/wave-h-report.md: wave_h_finalize.py
RECORDS = ROOT / "reports" / "wave-h" / "records"  # wave_h_finalize.py --fetch writes here
LAB_OUT = ROOT / "reports" / "wave-h"  # balanced-<kind>.json (wave_h_finalize.py reads them)
STORE_VOLUME = "flora-wave-h"
MEDIA_VOLUME = "flora-wave-h-media"
EMBODIMENT = "human"
JOB = "human"  # the one job (key in each step's ``jobs``, shared with wave_h_finalize.py)
# Six of the seven staged ``human`` golden episodes (docs/golden-manifest-v1.md: 7 selected, 7
# staged, all oracle-eligible): the EgoDex three, and three HomER rows (VFR, one without
# instruction text). The longest, ``homer_40`` (59.6 s), is left out to keep the cost down.
EPISODES: tuple[str, ...] = (
    "human/egodex_add_remove_lid/episode_000014",
    "human/egodex_braid_unbraid/episode_000001",
    "human/egodex_make_sandwich/episode_000002",
    "human/wgo/homer_52",
    "human/wgo/homer_53",
    "human/wgo/homer_50",
)
NOISE_EPISODES = EPISODES  # (f) repeats the oracle job: all of its episodes
# Plans (``--plan``). balanced-only: the user's decision of 2026-10-22 (run the GPU session
# balanced only, now; the oracle, certification and noise move into one later certified run).
PLANS: dict[str, tuple[str, ...]] = {
    "balanced-only": ("tempseg", "preflight", "balanced", "lab"),
    "full": ("tempseg", "preflight", "exact_e2e", "balanced", "lab", "certify", "noise"),
    # The certified run (module docstring): b, f, d -> (d2 after d, skipped when done) -> e
    # (in this order with --max-parallel 1; b, f and d at once otherwise: CERT_DEPS).
    "certified": ("exact_e2e", "noise", "balanced", "lab", "caption_iso", "certify"),
    # The stage-isolated re-certification of human (module docstring, "recertify"): f2 -> d'
    # -> d2' -> g -> e', one after the other, on (b)'s oracle (read from the certified state).
    # d' before d2' as (d) before (d2) in the certified run. It does not warm the lab: the lab
    # builds its engines on its own unkeyed caches (balanced_lab.cache_env), J1 publishes to
    # the keyed balanced-cc/ ones, and the 2026-10-22 certified lab (after (d)) was cold too.
    "recertify": ("noise_all", "balanced", "lab", "caption_iso", "certify"),
}
# What each step needs done first (``--max-parallel``; ``_runbook.run_parallel``). The
# certified plan: (b), (f) and (d) are independent (one data dir, staged once:
# :func:`stage_data`). (f) used to share (b)'s app: the same config file, so the same deployed
# app name (``factory.app_name_for``: project + stage hashes), and beside (b) one ``modal app
# stop`` of it would stop the other's job. It now has its own app: (b)'s config with its own
# project (per attempt, as the cold witness has; the project is not hashed, so the stage hash
# stays (b)'s, checked before any app: :func:`run_cert_noise`). Each job was its own cold
# start anyway ((b)'s app is stopped when its job is composed), so the noise floor is measured
# as before. (d2) needs (d): the Captioner half reads (d)'s segment records, both halves run on
# (d)'s episodes, and the lab writes the ``flora-cache`` engines (d)'s workers seed from (a lab
# beside (d) would warm (d)'s cold compile). (e) needs them all. The other plans are one chain
# (each step needs the one before): they run as they always did.
CERT_DEPS: dict[str, tuple[str, ...]] = {
    "exact_e2e": (), "noise": (), "balanced": (), "lab": ("balanced",),
    # (g) captions the oracle's own segment records: it needs (b) only
    "caption_iso": ("exact_e2e",),
    "certify": ("exact_e2e", "noise", "balanced", "lab", "caption_iso")}
# The GPUs each certified step holds at once (``--max-gpus``; ``_runbook.gpu_wait``): (b) and
# (f) one H200 container each (``max_containers: 1``, one app each), (d) its two pools at once
# (L40S segment + H100 caption, one container each; the cold witness runs after), (d2) one GPU
# at a time (L40S, then H100), (e) none. With b, f and d at once the peak is 4 GPUs, 2 of them
# H200. A Modal app's ``max_gpus`` caps that app only, and ``--max-parallel`` counts steps.
CERT_GPUS: dict[str, dict[str, int]] = {
    "exact_e2e": {"H200": 1}, "noise": {"H200": 1}, "balanced": {"L40S": 1, "H100": 1},
    "lab": {"L40S": 1}, "caption_iso": {"H100": 1}, "noise_all": {"H200": 1}, "certify": {}}
DEFAULT_MAX_GPUS = 4  # the certified plan's peak: the default never holds a step back


def parse_gpu_caps(specs: Sequence[str] | None) -> dict[str, int]:
    """``--max-gpus`` values (``N`` for the total, ``TYPE=N`` per GPU type) -> caps."""
    caps = {"total": DEFAULT_MAX_GPUS}
    for spec in specs or ():
        kind, _, n = spec.rpartition("=")
        if not n.isdigit() or int(n) < 1:
            raise ValueError(f"--max-gpus {spec!r}: want N or TYPE=N with N >= 1")
        caps[kind.strip() or "total"] = int(n)
    return caps


def plan_deps(plan: str) -> dict[str, tuple[str, ...]]:
    """Per step of ``plan``, the steps it needs done first."""
    return dict(CERT_DEPS) if plan == "certified" else rb.chain(PLANS[plan])


def is_cert(plan: str | None) -> bool:
    """The plans that certify: their own state, golden/v2 or v3-yam, honest caps."""
    return plan in CERT_PLANS


DEFAULT_PLAN = "balanced-only"
DEFERRED_NOTE = "deferred to the certified run"
CERTIFIED = "certified"
RECERTIFY = "recertify"
CERT_PLANS = (CERTIFIED, RECERTIFY)  # the plans with their own state (never shared)
# --- The certified plan (golden/v2, docs/golden-manifest-v2.md) -------------------------------
CERT_ROOT = ROOT / "reports" / "wave-h" / "certified"  # its own state, work dir and records
RECERT_ROOT = ROOT / "reports" / "wave-h" / "recertified"  # recertify: state, records, lab
CERT_GOLDEN = "golden/v2"
CERT_DATA = "cert-human"  # ONE data dir for the oracle, noise and balanced jobs (see CertSpec)
NOISE_PREFIX, REST_PREFIX = "n-", "x-"  # local id prefixes: the noise episodes sort first
MIN_CERT_EPISODES = 10  # certkey.POLICY min_samples (per pool)
MIN_NOISE_EPISODES = 5  # certkey.POLICY min_noise_samples
# One spare noise episode (review 13 S3-3): a noise episode the oracle or the noise run fails
# is replaced by it, as the 2 spare oracle episodes replace a failed balanced one.
NOISE_SPARES = 1
COLD_JOB = "cold"  # (d)'s cold witness job (review 13 W1): the noise subset, forced cold
COMPILE_CACHE_ENV = "FLORA_COMPILE_CACHE"  # modal_app.balanced_app.COMPILE_CACHE_ENV


def golden_selection(version: str) -> dict[str, Any]:
    """``validate/golden_selection_<v>.json`` (package data)."""
    from flora_deploy.validate.golden import SELECTIONS, load_selection

    return load_selection(SELECTIONS[version])


def _cert_sets(version: str = CERT_GOLDEN, embodiment: str = EMBODIMENT
               ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A certified plan's episodes (``embodiment``'s in ``version``) and noise subset: the
    selection's ``noise_subset`` plus :data:`NOISE_SPARES` spare(s), the first other episodes
    in selection order. All of them are oracle episodes (the noise run repeats the oracle's
    job), so every one can pair."""
    sel = golden_selection(version)
    eps = tuple(f"{e['pool']}/{e['dataset']}/{e['slug']}" for e in sel["episodes"]
                if e["pool"] == embodiment)
    noise = tuple(g for g in sel.get("noise_subset") or () if g in eps)
    spares = tuple(g for g in eps if g not in noise)[:NOISE_SPARES]
    return eps, noise + spares


# All 12 golden/v2 human episodes (10 for the certificate + 2 spares, all run), and the 6 the
# noise job repeats (the selection's ``noise_subset``, one VFR HomER row and four EgoDex clips,
# plus one spare).
CERT_EPISODES, CERT_NOISE = _cert_sets()
# --- The yam certified plan (golden/v3-yam, docs/golden-manifest-v3-yam.md) ----------------------
CERT_YAM_GOLDEN = "golden/v3-yam"
# Three-view pricing (:func:`three_view_scale`). The balanced rates in planner/rates.json were
# measured on six 1-view human episodes (run 102207184317, 178.25 video s). Per request, a
# three-view prompt is longer; the requests per video second are assumed equal (the ALOHA
# 3-view runs had fewer: 2 TempSeg / 6 Captioner calls in 22 s, docs/exact-concurrency.md).
# Measured per-request prompt tokens (1-view: the balanced lab on (d)'s job,
# reports/wave-h/balanced-<kind>.json; 3-view: docs/exact-concurrency.md, ALOHA, NVIDIA's own
# requests):
PROMPT_TOKENS_1VIEW = {"segment": 49_099 / 27, "caption": 407_769 / 72}  # 1,818 / 5,663
PROMPT_TOKENS_3VIEW = {"segment": 5_493.0, "caption": 11_145.0}  # caption max 13,650
# Generation's share of the deployed job's busy time: the lab's generate_s (50.4 s / 70.5 s
# for the same six episodes) over (d)'s busy seconds (0.367 / 0.906 container-s/video-s x
# 178.25 video s = 65.4 s / 161.5 s). Only this share is scaled by the prompt ratio; the rest
# (decode, frame preparation, waits) is kept at 1x: three 640-680x480 views (11 of the 12 yam
# episodes) hold ~0.44-0.47x the pixels of one 1080p human view, so 1x overstates them (the
# RoboCOIN episode's three 720p views are ~1.3x).
GENERATE_S = {"segment": 50.374, "caption": 70.454}
BUSY_S = {"segment": 0.367 * 178.25, "caption": 0.906 * 178.25}
# exact segcap (6.4 container-s/video-s) was measured on a 1-/3-view mix, half its episodes
# 3-view (docs/perf-1.md): only the 3-view half's difference is added (EXACT_3VIEW_SHARE). Its
# generation share is 5.04 of the 6.4 (docs/perf-1.md); its token mix is NVIDIA's call mix per
# video second (0.137 TempSeg, 0.286 Captioner calls, docs/perf-1.md).
EXACT_GENERATE_SHARE = 5.04 / 6.4
EXACT_CALLS_PER_S = {"segment": 0.137, "caption": 0.286}
EXACT_3VIEW_SHARE = 0.5


def three_view_scale() -> dict[str, float]:
    """Container-s per video-s multipliers for three-view prompts, per shard kind: balanced
    ``segment`` and ``caption`` (generation share x prompt-token ratio + the rest at 1x), and
    exact ``segcap`` (the same rule on NVIDIA's token mix, applied to the measured rate's 1-view
    half only)."""
    out = {}
    for kind in ("segment", "caption"):
        g = GENERATE_S[kind] / BUSY_S[kind]
        r = PROMPT_TOKENS_3VIEW[kind] / PROMPT_TOKENS_1VIEW[kind]
        out[kind] = round((1 - g) + g * r, 3)
    tok1 = sum(EXACT_CALLS_PER_S[k] * PROMPT_TOKENS_1VIEW[k] for k in EXACT_CALLS_PER_S)
    tok3 = sum(EXACT_CALLS_PER_S[k] * PROMPT_TOKENS_3VIEW[k] for k in EXACT_CALLS_PER_S)
    f = (1 - EXACT_GENERATE_SHARE) + EXACT_GENERATE_SHARE * tok3 / tok1
    # the measured rate = (1 - share) x r1 + share x r1 x f; a 3-view-only rate is r1 x f
    out["segcap"] = round(f / ((1 - EXACT_3VIEW_SHARE) + EXACT_3VIEW_SHARE * f), 3)
    return out


CERT_YAM_EPISODES, CERT_YAM_NOISE = _cert_sets(CERT_YAM_GOLDEN, "yam")


@dataclass(frozen=True)
class PlanSpec:
    """What a plan runs on and where it keeps its state. ``balanced-only`` / ``full``: the six
    golden/v1 episodes, ``reports/wave-h/``. ``certified``: golden/v2 and
    ``reports/wave-h/certified/`` (its own state, so it never inherits the v1 jobs; the lab
    evidence in ``reports/wave-h/balanced-<kind>.json`` is shared)."""

    name: str
    golden: str
    episodes: tuple[str, ...]
    noise: tuple[str, ...]
    state: Path
    work: Path
    records: Path | None  # None: the module's RECORDS / REPORT (read when used)
    report: Path | None
    # certified: every job stages ONE data dir (CERT_DATA), so the oracle's, the noise run's and
    # balanced's episode uids (``<data dir>--<episode id>``) are the same and `flora certify`
    # can pair them. (``full`` stages one dir per step: its uids never pair; see the runbook.)
    shared_data: bool = False
    # certified: the embodiment, its one data dir, the three-view rate multipliers per shard
    # kind (empty: the measured rates as they are) and the lab's episodes (None: EPISODES)
    embodiment: str = EMBODIMENT
    data: str = CERT_DATA
    rate_scale: tuple[tuple[str, float], ...] = ()
    lab_eps: tuple[str, ...] | None = None
    # recertify: the certified run's state whose (b) it repeats (read-only; ``--source-state``)
    # and whose (b) and old (f) the later steps read
    source: Path | None = None
    # The cold witnesses' episodes (() : the noise subset, as the golden/v2 human run had it).
    # They keep the :data:`NOISE_PREFIX` local ids, so ``flora index --limit`` selects them
    # and the uids stay those of (b)'s run when the noise run grows to every episode.
    witness: tuple[str, ...] = ()
    # recertify: where its lab JSON lives (None: :data:`LAB_OUT`)
    lab_dir: Path | None = None
    # the balanced step's cold witness (J1w) runs the segment stage only: the caption
    # certificate's witness is the isolated caption job's (J2w)
    segment_only_witness: bool = False
    # recertify runs on (b)'s run id, in (b)'s work dir and store: its own jobs, configs and
    # apps take this prefix, so none is (b)'s (``balanced-human`` is (b)'s (d))
    job_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.witness:
            object.__setattr__(self, "witness", self.noise)

    @property
    def scale(self) -> dict[str, float]:
        return dict(self.rate_scale)

    def local_id(self, gid: str) -> str:
        """The episode id a job sees (the ``mp4_dir`` file stem). Certified: the witness
        episodes (the old noise subset) get :data:`NOISE_PREFIX`, the rest
        :data:`REST_PREFIX`, so ``flora index --limit 6`` on the job's own config and data dir
        indexes exactly them. Keyed on the witness set, never on the noise run's episodes, so
        a noise run on every episode leaves every uid as (b) had it."""
        if not self.shared_data:
            return h0.episode_id(gid)
        return (NOISE_PREFIX if gid in self.witness else REST_PREFIX) + h0.episode_id(gid)

    def data_dir(self, step: str) -> str:
        return self.data if self.shared_data else f"{step}-{JOB}"

    @property
    def lab_episodes(self) -> tuple[str, ...]:
        """The episodes the lab (d2) captures its requests on. Certified: the golden/v1 six
        (all in golden/v2), so :data:`LAB_WORK_S` and the lab deadlines, measured on them,
        still hold; L0 is per model kind, not per episode. yam: its first six."""
        if not self.shared_data:
            return self.episodes
        return EPISODES if self.lab_eps is None else self.lab_eps

    @property
    def noise_limit(self) -> int | None:
        """``flora index --limit`` for the noise job (None: all of the oracle's episodes)."""
        return len(self.noise) if self.shared_data and self.noise != self.episodes else None

    @property
    def witness_limit(self) -> int:
        """``flora index --limit`` for a cold witness job: the witness episodes sort first."""
        return len(self.witness)

    @property
    def labs(self) -> Path:
        return self.lab_dir if self.lab_dir is not None else LAB_OUT


_V1 = {"golden": "golden/v1", "episodes": EPISODES, "noise": NOISE_EPISODES, "state": STATE,
       "work": ROOT / "reports" / "wave-h" / "work", "records": None, "report": None}
SPECS: dict[str, PlanSpec] = {
    "balanced-only": PlanSpec("balanced-only", **_V1),
    "full": PlanSpec("full", **_V1),
    CERTIFIED: PlanSpec(CERTIFIED, CERT_GOLDEN, CERT_EPISODES, CERT_NOISE,
                        CERT_ROOT / "state.json", CERT_ROOT / "work", CERT_ROOT / "records",
                        CERT_ROOT / "runbook.md", shared_data=True),
}
CERT_YAM_ROOT = ROOT / "reports" / "wave-h" / "certified-yam"
from flora_deploy.planner.rates import RateTable  # noqa: E402 - camera_scaling, as flora plan

# 3-view work multipliers; a rates.json without camera_scaling prices them at 1.0 (and the
# plan output warns: NO_CAMERA_SCALING).
_CAMERA_SCALING = RateTable.load().camera_scaling
NO_CAMERA_SCALING = _CAMERA_SCALING is None
_YAM_SCALE = () if NO_CAMERA_SCALING else tuple(sorted(_CAMERA_SCALING.factors(3).items()))

# The certified plan per embodiment (``--embodiment``): each its own golden set, data dir,
# state, work dir and records, so certified runs never collide and resume independently.
CERT_SPECS: dict[str, PlanSpec] = {
    "human": SPECS[CERTIFIED],
    # noise on all 12 (R2 of the v4 pre-registration), the witnesses on the noise subset
    "yam": PlanSpec(CERTIFIED, CERT_YAM_GOLDEN, CERT_YAM_EPISODES, CERT_YAM_EPISODES,
                    CERT_YAM_ROOT / "state.json", CERT_YAM_ROOT / "work",
                    CERT_YAM_ROOT / "records", CERT_YAM_ROOT / "runbook.md", shared_data=True,
                    embodiment="yam", data="cert-yam",
                    rate_scale=_YAM_SCALE,
                    lab_eps=CERT_YAM_EPISODES[:6], witness=CERT_YAM_NOISE,
                    segment_only_witness=True),
}
# config.schema.Embodiment; the ones without a golden certification set are refused.
ALL_EMBODIMENTS = ("human", "yam", "yam32", "dex", "tianji")


def spec_for(plan: str, embodiment: str = EMBODIMENT) -> PlanSpec:
    """The plan's spec; ``certified`` per embodiment (:data:`CERT_SPECS`). Raises KeyError for
    an embodiment that has no certified set, or a non-human one on another plan."""
    if embodiment == EMBODIMENT:
        return SPECS[plan]
    if plan == CERTIFIED:
        return CERT_SPECS[embodiment]
    raise KeyError(embodiment)


# --- The recertify plan (human; scratchpad caption-prereg.md, order A) -----------------------
# Step f2 ``noise_all``: the certified run's (b) repeated on ALL of its episodes (12), exact vs
# exact, from a checkout that still reproduces (b)'s stage hash: (b)'s own config file (its
# ``project`` renamed, so this attempt has its own job and app) and data dir, re-indexed with no
# ``--limit``, so the same stage hash, GPU (H200) and mode (nvidia). (b) is reused, not re-run.
# Its own state (``reports/wave-h/recertified/state.json``), so the certified run's state, and
# with it every finished step's record (b, f, d, d2, e), is never written; it reads (b) from the
# certified state, read-only (:func:`oracle_source`), and takes that run's ``run_id``: (b)'s
# config writes into that run's store and reads that run's staged media. The later steps
# (PLANS[RECERTIFY]: d' balanced, d2' lab, g caption_iso, e' certify) run on the v4 code: the
# noise run is all 12 (noise == episodes), the witnesses (b)'s old noise subset (its n- local
# ids, so (b)'s uids), the J1 witness segment-only, the lab JSON in RECERT_ROOT, and every job,
# config and app of theirs prefixed ``rc-`` (it shares (b)'s run id and work dir).
RECERT_SOURCE = CERT_ROOT / "state.json"  # the certified run whose (b) is repeated (read-only)
RECERT_STEPS = PLANS[RECERTIFY]
NOISE_ALL_JOB = "noise12"  # job / app / index tag: wave-h-<run>-noise12-<embodiment>-a<n>
SPECS[RECERTIFY] = replace(SPECS[CERTIFIED], name=RECERTIFY, state=RECERT_ROOT / "state.json",
                           # (b)'s work dir: its config's relative ``input.path`` (the data dir,
                           # hashed by content) resolves there, and flora runs from it
                           work=CERT_ROOT / "work", records=RECERT_ROOT / "records",
                           report=RECERT_ROOT / "runbook.md", source=RECERT_SOURCE,
                           noise=CERT_EPISODES, witness=CERT_NOISE, lab_dir=RECERT_ROOT,
                           segment_only_witness=True, job_prefix="rc-")
# Per-step caps (USD): each step's upper projection, priced on the golden selection's expected
# durations x (1 + DURATION_HEADROOM) (the staged human files run ~5% longer in all), rounded up
# to 5 cents (``honest_caps``), on the shipped ``planner/rates.json`` (the balanced rows measured
# 2026-10-22). (d) is the deployed job alone (its cold-compile startup measured, so not scaled
# again: :func:`cold_startup_s`); (d2) the lab, priced with two cold compiles per call
# (:func:`lab_seconds`).
# Global cap: ``--plan balanced-only`` (default) uses the sum of its steps' upper projections
# rounded up to $0.10 (:func:`plan_global_cap`); ``--plan full`` the earlier $6.50, which the
# full plan's upper total does not fit (the dry run says so, and the stop rule refuses the first
# step whose upper projection no longer fits the cap left). Raising a cap is the user's decision.
DURATION_HEADROOM = 0.10
CAPS_USD = {"tempseg": 0.75, "preflight": 0.05, "exact_e2e": 3.30, "balanced": 2.35,
            "lab": 2.35, "certify": 0.0, "noise": 3.30,
            # the certifying plans' own steps (their caps are honest_caps; these never gate)
            "noise_all": 4.20, "caption_iso": 2.85}
GLOBAL_CAP_USD = 6.50  # --plan full; balanced-only derives its own (plan_global_cap)
# --plan recertify: the user-approved global cap (2026-10-23, pre-registration D5: the human
# recertify upper total, f2's noise-12 included). Its derived cap (the uppers' sum rounded up
# to $0.10) is used when lower; above it only a --global-cap the user approves raises it.
RECERT_GLOBAL_CAP_USD = 11.70
# The most --global-cap may approve: the project's whole GPU budget (review 12 F7: ``inf`` once
# turned every global check into a no-op): ``$FLORA_PROJECT_BUDGET_USD``, default $50, the one
# setting ``spend.py`` and ``cert_preflight.py`` read too.
MAX_GLOBAL_CAP_USD = _spend.project_budget_usd()
# Cold compile (image smoke, 2026-10-22): ``flora-cache`` holds no compile cache for
# balanced/tempseg or balanced/captioner at the current image key, so (d)'s first L40S and H100
# engine starts compile from scratch (torch.compile, Triton, CUDA graphs). The balanced startup
# row of planner/rates.json (130 s, projected with a warm cache) is scaled by the Rewarder's
# measured cold/warm @enter ratio, 318 s / 229 s (docs/rewarder-smoke.md, docs/cold-start.md).
COLD_COMPILE_FACTOR = 318.0 / 229.0
# (d2) lab call timing. Every engine the lab builds compiles cold: the lab's compile caches
# are its own unkeyed ``flora-cache:/balanced/<engine>[-dtbias]`` dirs
# (``balanced_lab.cache_env``), which no deployed worker writes (they publish keyed caches under
# ``balanced-cc/``, ``engines/compile_cache.py``); the 2026-10-22 certified lab, run after (d)'s
# job, recorded both of its caches cold (``compile_cache.*.warm: false``). The calls per kind
# (:data:`LAB_CALLS`): TempSeg one call, both engines (``"both"``); the Captioner two calls, each
# its own container under the 20-minute timeout: ``"baseline"`` (L0, the stage, the Gate A
# requests) and ``"plugin"`` (the FP32 ``dt_bias`` engine's Gate A on the staged requests).
# Measured 2026-10-22 (reports/wave-h/balanced-<kind>.json, the certified run's lab): TempSeg
# 683 s for the one call (builds 272 s + 201 s); the Captioner one call 1,096 s wall, 1,075 s in
# the container: the baseline child 810 s (engine build 465 s with the 51 GiB of weights read
# cold, then the stage 330 s and L0 15 s) and the plugin child 264 s (build 232 s with the
# weights already in the page cache, Gate A 32 s). The v4 caption stage is slower
# (``max_num_seqs`` 16 -> 11 on H100, and one HF processor replay per call for its input
# digest): the recertify lab of 2026-10-23, still one call, passed its 1,131 s deadline, so
# both engines in one container no longer fit the timeout. :data:`LAB_V4_CAPTION_S` estimates
# the extra stage time (more than the 38 s that attempt shows, at least). The plugin call starts
# a fresh container, so it reads the weights cold again: its build is priced as the baseline's.
# The split is the certifying plans' (:func:`lab_calls`); the v1 plans (``balanced-only``,
# ``full``, from before v4) keep their one call per kind, priced as they always were (the
# Captioner's two builds at the deployed worker's 325 s cold startup, 225 s of work).
LAB_V4_CAPTION_S = 120.0
LAB_CALLS: dict[str, tuple[str, ...]] = {"tempseg": ("both",),
                                         "captioner": ("baseline", "plugin")}
LAB_ONE_CALL = ("both",)
LAB_CALL_PARTS_S: dict[tuple[str, str], tuple[float, float]] = {  # (engine builds, work)
    ("tempseg", "both"): (270.0 + 190.0, 125.0),
    ("captioner", "both"): (325.0 + 325.0, 225.0),  # the v1 plans' one call
    ("captioner", "baseline"): (465.0, 345.0 + LAB_V4_CAPTION_S),
    ("captioner", "plugin"): (465.0, 35.0),
}
LAB_OVERHEAD_S = 30.0  # container start, weight prefetch, child spawn
LAB_UNCERTAINTY = 0.25
LAB_SLACK_S = 60  # lab: client timeout -> cancel -> container gone
LAB_TIMEOUT_S = 20 * 60  # modal_app/balanced_lab.py TIMEOUT_S: the container's hard limit
LAB_CALL_DESC = {"both": "L0 + Gate A, one call: 2 cold compiles",
                 "baseline": "baseline call: L0 + stage + Gate A requests, 1 cold compile",
                 "plugin": "plugin call: dt_bias Gate A on the staged requests, 1 cold compile"}


def lab_call_seconds(kind: str, arm: str) -> tuple[float, float]:
    """(expected, client deadline) seconds of one lab call (``arm`` of :data:`LAB_CALLS`),
    every compile cold; the deadline is expected x (1 + :data:`LAB_UNCERTAINTY`), at most the
    container timeout."""
    build, work = LAB_CALL_PARTS_S[(kind, arm)]
    expected = LAB_OVERHEAD_S + build + work
    return expected, min(float(LAB_TIMEOUT_S), expected * (1 + LAB_UNCERTAINTY))


def lab_calls(kind: str, spec: PlanSpec | None = None) -> tuple[str, ...]:
    """The lab calls ``kind`` runs as under ``spec``'s plan: the certifying plans
    :data:`LAB_CALLS` (the Captioner split), the v1 plans one call (:data:`LAB_ONE_CALL`)."""
    return LAB_CALLS[kind] if spec is not None and spec.shared_data else LAB_ONE_CALL


def lab_seconds(kind: str, spec: PlanSpec | None = None) -> tuple[float, float]:
    """(expected, deadline) seconds of ``kind``'s whole lab under ``spec``'s plan (default a
    v1 plan's one call): the sums over its calls."""
    calls = [lab_call_seconds(kind, a) for a in lab_calls(kind, spec)]
    return sum(c[0] for c in calls), sum(c[1] for c in calls)


LAB_CALL_EXPECTED_S = {(k, a): lab_call_seconds(k, a)[0] for k, a in LAB_CALL_PARTS_S}
LAB_CALL_DEADLINE_S = {(k, a): lab_call_seconds(k, a)[1] for k, a in LAB_CALL_PARTS_S}
# one call per kind (the v1 plans)
LAB_EXPECTED_S = {k: lab_seconds(k)[0] for k in LAB_CALLS}
LAB_DEADLINE_S = {k: lab_seconds(k)[1] for k in LAB_CALLS}
TEMPSEG_APP = "flora-convert"  # modal_app/convert_app.py
MERGE_EXPECTED_S, MERGE_MAX_S = 300, 1200  # v2: FP64 merge, "a few minutes" (tempseg-merge.md)
EQUIV_EXPECTED_S, EQUIV_MAX_S = 240, 600  # v1: 142 s + image start on one L40S
IDLE_AFTER_JOB_S = 30  # last receipt -> compose -> `modal app stop`
CLIENT_SLACK_S = rb.CLIENT_SLACK_S  # per app: `flora run` timeout, then cancel + app stop
BILLING_LAG_S = rb.BILLING_LAG_S  # a state without a billing watermark: the old lag rule
HEARTBEAT_S = rb.HEARTBEAT_S
ORACLE_DETERMINISM = "nvidia"  # Fix-Cert: oracle and noise run in NVIDIA's default mode
NOISE_JOB = JOB  # the oracle job (f) repeats
NEEDS_PIN = 4  # exit code: step 0 built TempSeg; pin its content digest, then re-run
# Provisional thresholds (until the S2b noise floor exists): balanced vs exact must be within
# NOISE_FACTOR x the measured exact-vs-exact noise, or these absolute bounds without one.
NOISE_FACTOR = 2.0
ABS_BOUNDS = {"boundary_mae_s": 1.0, "segment_iou": 0.6, "caption_jaccard": 0.3}
GATE_A_MAX_DLOGIT = 0.5


# ---------------------------------------------------------------------------------------------
# Prices and projections (pure)


@dataclass(frozen=True)
class Container:
    """One container shape: GPU type (None for CPU), cores, memory GiB."""

    gpu: str | None
    cpu: float
    memory_gib: float

    def usd_per_s(self, rates: Any = None) -> float:
        from flora_deploy.jobs.budget import pinned_rates

        r = rates or pinned_rates()
        gpu = r.gpu_per_s.get((self.gpu or "").rstrip("!"), 0.0) if self.gpu else 0.0
        return gpu + self.cpu * r.core_per_s + self.memory_gib * r.gib_per_s

    def usd_per_h(self, rates: Any = None) -> float:
        return self.usd_per_s(rates) * 3600.0


def pool_container(kind: str, preset: str = "exact") -> Container:
    from flora_deploy.config.presets import get_preset

    p = get_preset(preset).pools[kind]
    return Container(p.gpu, float(p.cpu_cores), float(p.memory_gib))


LAB = {"tempseg": Container("L40S", 8.0, 48.0), "captioner": Container("H100", 8.0, 96.0)}
PREFLIGHT = Container(None, 2.0, 4.0)
MERGE = Container(None, 8.0, 32.0)  # convert_app.MERGE_CPU / MERGE_MEMORY_MIB
EQUIV = Container("L40S", 8.0, 48.0)  # convert_app.EQUIV_GPU / EQUIV_MEMORY_MIB


@dataclass
class Projection:
    step: str
    items: list[dict[str, Any]] = field(default_factory=list)
    peak_usd_per_s: float = 0.0  # the most the step can burn at once (heartbeat estimate)
    # The upper the plan's global cap is derived from, when it differs from ``usd_upper`` (a
    # lab step that reuses saved results: the cap stays what the whole step was planned at).
    cap_usd_upper: float | None = None
    # wall-clock: items of one group run at the same time (``what`` -> group); ``wall``
    # overrides the step's (nominal, upper) wall seconds (scale_test's run: its schedule)
    groups: dict[str, str] = field(default_factory=dict)
    wall: tuple[float, float] | None = None

    def add(self, what: str, c: Container, seconds: float, *, upper_s: float | None = None,
            group: str | None = None) -> None:
        self.items.append({"what": what, "gpu": c.gpu or "cpu", "seconds": round(seconds, 1),
                           "upper_seconds": round(upper_s or seconds, 1),
                           "usd_per_h": round(c.usd_per_h(), 3),
                           "usd": round(c.usd_per_s() * seconds, 4),
                           "usd_upper": round(c.usd_per_s() * (upper_s or seconds), 4)})
        if group is not None:
            self.groups[what] = group

    def wall_s(self, upper: bool = False) -> float:
        """The step's wall seconds: its items one after the other, but the items of one group
        at the same time (the longest of them)."""
        if self.wall is not None:
            return float(self.wall[1 if upper else 0])
        key = "upper_seconds" if upper else "seconds"
        total, spans = 0.0, {}
        for i in self.items:
            g = self.groups.get(i["what"])
            if g is None:
                total += i[key]
            else:
                spans[g] = max(spans.get(g, 0.0), i[key])
        return round(total + sum(spans.values()), 1)

    @property
    def usd(self) -> float:
        return round(sum(i["usd"] for i in self.items), 4)

    @property
    def usd_upper(self) -> float:
        return round(sum(i["usd_upper"] for i in self.items), 4)


def durations(manifest: Mapping[str, Any]) -> dict[str, float]:
    return {e["id"]: float(e["duration_s"]) for e in manifest["episodes"]}


def episodes_in(manifest: Mapping[str, Any], episodes: Sequence[str] = EPISODES
                ) -> tuple[str, ...]:
    """The ``episodes`` (default :data:`EPISODES`) the manifest has staged."""
    have = durations(manifest)
    return tuple(e for e in episodes if e in have)


def missing_episodes(manifest: Mapping[str, Any], episodes: Sequence[str] = EPISODES
                     ) -> tuple[str, ...]:
    have = durations(manifest)
    return tuple(e for e in episodes if e not in have)


def offline_manifest(headroom: float = 0.0, version: str = "golden/v1") -> dict[str, Any]:
    """The golden selection's expected durations (``validate/golden_selection_<v>.json``) as a
    manifest, for a dry run that reads no Volume; ``headroom`` scales them up."""
    sel = golden_selection(version)
    eps = []
    for e in sel["episodes"]:
        gid = f"{e['pool']}/{e['dataset']}/{e['slug']}"
        if e.get("expected_s") is not None:
            eps.append({"id": gid, "duration_s": round(float(e["expected_s"]) * (1 + headroom),
                                                       2)})
    return {"episodes": eps, "offline": True}


def exact_job_seconds(video_s: float, table: Any = None) -> tuple[float, float]:
    """Nominal and upper container-seconds of one exact segcap job in one container (startup,
    work, idle until ``modal app stop``; upper: upper quantiles plus :data:`CLIENT_SLACK_S`)."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    e = t.rate("exact", "segcap", 1, 1)
    st = t.startup_for("exact", "segcap")
    nominal = st.startup_s + video_s * e.container_s_per_video_s + IDLE_AFTER_JOB_S
    upper = (st.startup_s * (1 + st.uncertainty)
             + video_s * e.container_s_per_video_s * (1 + e.uncertainty) + IDLE_AFTER_JOB_S
             + CLIENT_SLACK_S)
    return nominal, upper


def cold_startup_s(st: Any, compile_factor: float = COLD_COMPILE_FACTOR) -> tuple[float, str]:
    """A cold-compile startup from a ``planner.rates.StartupEntry``, and what it rests on. A
    startup **measured** with a cold compile cache (``compile_cache: cold``) already includes
    the compile: taken as is. A projection, a placeholder or a warm measurement is scaled by
    ``compile_factor`` (:data:`COLD_COMPILE_FACTOR`), once."""
    cache = getattr(st, "compile_cache", None)
    if st.status == "measured" and cache == "cold":
        return float(st.startup_s), "measured cold"
    basis = f"measured {cache}" if st.status == "measured" and cache else st.status
    return float(st.startup_s) * compile_factor, f"{basis} x {compile_factor:.2f}"


def startup_basis(table: Any = None) -> dict[str, str]:
    """Per balanced pool: what (d)'s startup is priced on (for the dry run)."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    out = {}
    for kind in ("segment", "caption"):
        st = t.startup_for("balanced", kind)
        sec, basis = cold_startup_s(st)
        out[kind] = f"{sec:.0f} s ({basis}, rates.json {st.startup_s:g} s)"
    return out


def balanced_engine_limits() -> dict[str, dict[str, Any]]:
    """Per balanced pool: the engine limits (``engines.offline_llm.GPU_LIMITS``) and batch
    policy (``engines.batch_policy``) its deployed worker builds on the preset's GPU, from the
    code (what (d)'s ``flora deploy`` runs and its certificate key hashes)."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.engines.batch_policy import policy_for
    from flora_deploy.engines.offline_llm import spec_for

    out: dict[str, dict[str, Any]] = {}
    for kind in ("segment", "caption"):
        gpu = get_preset("balanced").pools[kind].gpu
        lim = spec_for(kind, "<pinned>", gpu).limits
        out[kind] = {"gpu": gpu, "max_num_seqs": lim.max_num_seqs,
                     "max_num_batched_tokens": lim.max_num_batched_tokens,
                     "infer_episodes": policy_for(kind).infer_episodes}
    return out


def warm_startup_s(st: Any) -> tuple[float, str] | None:
    """A startup row timed (or placeheld) with a loaded compile cache, as is; ``None`` when
    ``st`` is not a ``compile_cache: warm`` row (price it cold instead)."""
    if getattr(st, "compile_cache", None) != "warm":
        return None
    return float(st.startup_s), f"{st.status} warm"


def balanced_job_seconds(video_s: float, table: Any = None, *,
                         compile_factor: float = COLD_COMPILE_FACTOR,
                         warm_kinds: tuple[str, ...] = (),
                         scale: Mapping[str, float] | None = None
                         ) -> dict[str, tuple[float, float]]:
    """One balanced job (one app): kind -> (nominal, upper) container-seconds. Each pool has
    its own cold start, its startup scaled by ``compile_factor`` (a cold compile cache:
    :data:`COLD_COMPILE_FACTOR`), except the pools in ``warm_kinds`` (a compile cache is
    published for their key), priced with the warm startup row when there is one; the segment
    container then idles through the scale-down window while captions run (the caption shards
    wait for it); both stay up through the client slack. ``scale``: per kind, a multiplier on
    the work (three-view prompts, :func:`three_view_scale`)."""
    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    out: dict[str, tuple[float, float]] = {}
    work: dict[str, tuple[float, float, float]] = {}
    for kind in ("segment", "caption"):
        e = t.rate("balanced", kind, 1, 1)
        st = t.startup_for("balanced", kind, warm=kind in warm_kinds)
        start, _basis = warm_startup_s(st) or cold_startup_s(st, compile_factor)
        w = video_s * e.container_s_per_video_s
        if scale and kind in scale:
            w *= scale[kind]
        work[kind] = (start, w, max(e.uncertainty, st.uncertainty))
    cap_nominal = work["caption"][0] + work["caption"][1]
    for kind, (start, w, unc) in work.items():
        idle = min(WORKER_SCALEDOWN_S, cap_nominal) if kind == "segment" else 0.0
        idle_up = WORKER_SCALEDOWN_S if kind == "segment" else 0.0
        out[kind] = (start + w + idle + IDLE_AFTER_JOB_S,
                     (start + w) * (1 + unc) + idle_up + IDLE_AFTER_JOB_S + CLIENT_SLACK_S)
    return out


def single_pool_seconds(kind: str, video_s: float, table: Any = None, *,
                        compile_factor: float = COLD_COMPILE_FACTOR,
                        scale: Mapping[str, float] | None = None) -> tuple[float, float]:
    """Nominal and upper container-seconds of a balanced job with **one** pool (a
    segment-only cold witness, a caption-only isolated job): its cold start, the work and the
    idle until ``modal app stop`` (upper: ``(start + work) x (1 + uncertainty)`` plus the
    client slack). No other pool waits on it, so there is no scale-down tail."""
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    e = t.rate("balanced", kind, 1, 1)
    st = t.startup_for("balanced", kind)
    start, _basis = cold_startup_s(st, compile_factor)
    w = video_s * e.container_s_per_video_s * ((scale or {}).get(kind, 1.0))
    unc = max(e.uncertainty, st.uncertainty)
    return (start + w + IDLE_AFTER_JOB_S,
            (start + w) * (1 + unc) + IDLE_AFTER_JOB_S + CLIENT_SLACK_S)


def project(step: str, manifest: Mapping[str, Any], spec: PlanSpec | None = None, *,
            reuse: Iterable[str] = (), lab_done: bool = False,
            witness_only: bool = False) -> Projection:
    """What ``step`` is expected to cost (nominal rates; ``usd_upper``: upper quantiles plus
    every cold start, scale-down tail and client slack the step can incur). ``spec``: the
    plan's episodes (default: the golden/v1 six). ``reuse``: lab kinds whose saved result is
    reused (:func:`lab_reuse`), priced at $0. ``lab_done`` (certified plan): the lab's L0 +
    Gate A evidence already exists (:func:`lab_already_done`), so the step starts nothing: $0
    (and the global cap leaves it out: the step never runs). ``witness_only`` (certified (d),
    :func:`witness_pending`): only the cold witness re-runs, so (d) is priced at its items; the
    global cap keeps the whole step's upper (``cap_usd_upper``)."""
    spec = spec or SPECS[DEFAULT_PLAN]
    dur = durations(manifest)
    eps = episodes_in(manifest, spec.episodes)
    v = sum(dur[i] for i in eps)
    job, sc = spec.embodiment, spec.scale

    def work(kind: str) -> str:  # the label's note on a scaled rate
        return f" x{sc[kind]:.2f} 3-view work" if kind in sc else ""

    p = Projection(step)
    if step == "tempseg":
        p.add("convert_app tempseg --step merge (merger v2)", MERGE, MERGE_EXPECTED_S,
              upper_s=MERGE_MAX_S + LAB_SLACK_S)
        p.add("convert_app tempseg --step equivalence", EQUIV, EQUIV_EXPECTED_S,
              upper_s=EQUIV_MAX_S + LAB_SLACK_S)
        p.peak_usd_per_s = max(MERGE.usd_per_s(), EQUIV.usd_per_s())  # one after the other
    elif step == "preflight":
        p.add("balanced_lab.preflight", PREFLIGHT, 90.0, upper_s=90.0 + LAB_SLACK_S)
        p.peak_usd_per_s = PREFLIGHT.usd_per_s()
    elif step == "exact_e2e":
        nominal, upper = exact_job_seconds(v * sc["segcap"] if "segcap" in sc else v)
        p.add(f"segcap {job} ({len(eps)} ep, {v:.1f} s{work('segcap')}, cold start)",
              pool_container("segcap"), nominal, upper_s=upper)
        p.peak_usd_per_s = pool_container("segcap").usd_per_s()
    elif step == "balanced":
        seg, cap = pool_container("segment", "balanced"), pool_container("caption", "balanced")
        secs = balanced_job_seconds(v, scale=sc or None)
        p.add(f"balanced segment {job} ({len(eps)} ep, {v:.1f} s{work('segment')}, cold "
              "compile)", seg, secs["segment"][0], upper_s=secs["segment"][1], group="job")
        p.add(f"balanced caption {job} ({len(eps)} ep, {v:.1f} s{work('caption')}, cold "
              "compile)", cap, secs["caption"][0], upper_s=secs["caption"][1], group="job")
        if spec.shared_data:  # certified: the forced-cold job on the witness episodes
            neps = [i for i in spec.witness if i in dur]
            nv = sum(dur[i] for i in neps)
            if spec.segment_only_witness:  # J1w: the segment stage alone, one pool
                n, u = single_pool_seconds("segment", nv, scale=sc or None)
                p.add(f"cold witness segment-only {job} ({len(neps)} ep, {nv:.1f} s"
                      f"{work('segment')}, forced cold)", seg, n, upper_s=u, group="witness")
            else:
                ws = balanced_job_seconds(nv, scale=sc or None)
                for kind, c in (("segment", seg), ("caption", cap)):
                    p.add(f"cold witness {kind} {job} ({len(neps)} ep, {nv:.1f} s{work(kind)}, "
                          "forced cold)", c, ws[kind][0], upper_s=ws[kind][1], group="witness")
            if witness_only:  # (d)'s job is done: the witness alone re-runs
                full = p.usd_upper
                p.items = [i for i in p.items if i["what"].startswith("cold witness")]
                p.cap_usd_upper = full
        p.peak_usd_per_s = seg.usd_per_s() + cap.usd_per_s()
    elif step == "lab" and lab_done:
        p.add("balanced lab: L0 + Gate A already done (evidence present): skipped",
              LAB["tempseg"], 0.0)
    elif step == "lab":
        reuse = set(reuse)
        full = 0.0
        for kind in ("tempseg", "captioner"):
            full += LAB[kind].usd_per_s() * sum(LAB_CALL_DEADLINE_S[(kind, a)] + LAB_SLACK_S
                                                for a in lab_calls(kind, spec))
            if kind in reuse:
                p.add(f"balanced lab {kind}: saved result reused (has evidence), not run",
                      LAB[kind], 0.0, upper_s=0.0)
                continue
            for arm in lab_calls(kind, spec):
                if f"{kind}:{arm}" in reuse:  # a saved baseline call: only the plugin runs
                    p.add(f"balanced lab {kind} {arm} call: saved result reused, not run",
                          LAB[kind], 0.0, upper_s=0.0)
                    continue
                p.add(f"balanced lab {kind} ({LAB_CALL_DESC[arm]})", LAB[kind],
                      LAB_CALL_EXPECTED_S[(kind, arm)],
                      upper_s=LAB_CALL_DEADLINE_S[(kind, arm)] + LAB_SLACK_S)
        p.peak_usd_per_s = max((LAB[k].usd_per_s() for k in LAB if k not in reuse),
                               default=0.0)  # one after the other
        # recertify reuses TempSeg's saved result by design (its engine did not change): its
        # cap is the Captioner call's upper alone
        if reuse and spec.name != RECERTIFY:
            p.cap_usd_upper = round(full, 4)
    elif step == "noise":
        neps = [i for i in spec.noise if i in dur]
        nv = sum(dur[i] for i in neps)
        nominal, upper = exact_job_seconds(nv * sc["segcap"] if "segcap" in sc else nv)
        p.add(f"segcap {job} repeat ({len(neps)} ep, {nv:.1f} s{work('segcap')}, cold start)",
              pool_container("segcap"), nominal, upper_s=upper)
        p.peak_usd_per_s = pool_container("segcap").usd_per_s()
    elif step == "noise_all":  # recertify f2: (b) repeated on every one of its episodes
        nominal, upper = exact_job_seconds(v * sc["segcap"] if "segcap" in sc else v)
        p.add(f"segcap {job} repeat of (b), all {len(eps)} ep ({v:.1f} s{work('segcap')}, "
              "cold start)", pool_container("segcap"), nominal, upper_s=upper)
        p.peak_usd_per_s = pool_container("segcap").usd_per_s()
    elif step == "caption_iso":  # J2 (every episode), then J2w (the witnesses, forced cold)
        cap = pool_container("caption", "balanced")
        weps = [i for i in spec.witness if i in dur]
        wv = sum(dur[i] for i in weps)
        for what, n_ep, vs, extra in (("isolated caption", len(eps), v, "cold compile"),
                                      ("isolated caption cold witness", len(weps), wv,
                                       "forced cold")):
            n, u = single_pool_seconds("caption", vs, scale=sc or None)
            p.add(f"{what} {job} ({n_ep} ep, {vs:.1f} s{work('caption')}, oracle segments, "
                  f"{extra})", cap, n, upper_s=u)
        if witness_only:  # J2 is done: the witness alone re-runs
            full = p.usd_upper
            p.items = [i for i in p.items if "cold witness" in i["what"]]
            p.cap_usd_upper = full
        p.peak_usd_per_s = cap.usd_per_s()
    elif step == "certify":
        pass
    else:
        raise KeyError(step)
    return p


def deferred_steps(plan: str) -> tuple[str, ...]:
    """The steps a plan never runs ("deferred to the certified run")."""
    return tuple(s for s in STEPS if s not in PLANS[plan])


def _cap_upper(p: Any) -> float:
    cap = getattr(p, "cap_usd_upper", None)
    return float(p.usd_upper if cap is None else cap)


def plan_global_cap(plan: str, projections: Mapping[str, Projection],
                    override: float | None = None) -> float:
    """``full``: :data:`GLOBAL_CAP_USD`. Otherwise the sum of the plan's upper projections,
    rounded up to $0.10. ``override``: a cap the user approved (``--global-cap``, kept in the
    state); it can raise the cap, never lower it below the derived one."""
    if plan == "full":
        derived = GLOBAL_CAP_USD
    else:
        up = round(sum(_cap_upper(projections[s]) for s in PLANS[plan]), 6)
        derived = round(math.ceil(up * 10 - 1e-9) / 10, 2)
        if plan == RECERTIFY:  # never above what the user approved (f2 included)
            derived = min(derived, RECERT_GLOBAL_CAP_USD)
    return max(derived, override) if override is not None else derived


def check_global_cap(value: float, *, spent: float, derived: float) -> str | None:
    """Why ``--global-cap value`` is refused (None: it is accepted). A finite amount, at most
    :data:`MAX_GLOBAL_CAP_USD`, not below what was spent, and above the plan's derived cap (an
    override only raises the cap; a lower one would have no effect)."""
    if not math.isfinite(value) or value <= 0:
        return f"--global-cap {value!r} is not a positive finite amount"
    if value > MAX_GLOBAL_CAP_USD + 1e-9:
        return (f"--global-cap ${value:.2f} is over the ${MAX_GLOBAL_CAP_USD:.2f} ceiling (the "
                "project budget)")
    if value < spent - 1e-9:
        return f"--global-cap ${value:.2f} is below the ${spent:.2f} already spent"
    if value < derived - 1e-9:
        return (f"--global-cap ${value:.2f} is below the plan's derived cap ${derived:.2f}; an "
                "override can only raise it")
    return None


def global_cap_label(st: State, plan: str, projections: Mapping[str, Projection]) -> str:
    """How the plan's global cap in force (:func:`plan_global_cap`) came about."""
    derived = plan_global_cap(plan, projections)
    ov = st.override_for(plan)
    if ov is not None and ov > derived + 1e-9:
        return f"user-approved override for plan {plan}; derived ${derived:.2f}"
    if plan == RECERTIFY and derived >= RECERT_GLOBAL_CAP_USD - 1e-9:
        return (f"the user-approved ${RECERT_GLOBAL_CAP_USD:.2f} (f2 included): the sum of the "
                "plan's upper projections, rounded up to $0.10, is not below it")
    return ("the sum of the plan's upper projections, rounded up to $0.10" if plan != "full"
            else "GLOBAL_CAP_USD")


def honest_caps(projections: Mapping[str, Projection]) -> dict[str, float]:
    """Per step, the smallest cap (rounded up to 5 cents) its upper projection fits, and the
    global total of those (``"global"``)."""
    out = {s: math.ceil(p.usd_upper * 20 - 1e-9) / 20 for s, p in projections.items()}
    out["global"] = round(sum(out.values()), 2)
    return out


def plan_caps(plan: str, projections: Mapping[str, Projection]) -> dict[str, float]:
    """Per-step caps (USD). ``certified``: each step's own upper projection (measured rates on
    the golden/v2 durations) rounded up to 5 cents (:func:`honest_caps`), so the stop rule
    before a step is its upper price and nothing more (``recertify`` too). Other plans:
    :data:`CAPS_USD`."""
    if not is_cert(plan):
        return dict(CAPS_USD)
    honest = honest_caps({s: projections[s] for s in PLANS[plan]})
    return {**CAPS_USD, **{s: honest[s] for s in PLANS[plan]}}


def plan_projections(plan: str, manifest: Mapping[str, Any], *, lab_done: bool = False,
                     reuse: Iterable[str] = (), spec: PlanSpec | None = None,
                     witness_only: bool = False,
                     iso_witness_only: bool = False) -> dict[str, Projection]:
    """Every step's projection on the plan's episodes (``lab_done`` only applies to the
    certifying plans: the other plans run their lab; ``reuse``: :func:`lab_reuse`; recertify
    always reuses TempSeg's; ``spec``: default the plan's, :func:`spec_for` for another
    embodiment; ``witness_only``: :func:`witness_pending`, (d) priced at its cold witness
    alone; ``iso_witness_only``: :func:`iso_witness_pending`, (g) at J2w alone). ``reuse``
    may also name a saved call (``"captioner:baseline"``: :func:`saved_lab_calls`)."""
    spec = spec or SPECS[plan]
    if plan == RECERTIFY:
        reuse = tuple(sorted(set(reuse) | {"tempseg"}))
    cert = is_cert(plan)
    return {s: project(s, manifest, spec, lab_done=lab_done and cert,
                       reuse=reuse if s == "lab" else (),
                       witness_only=cert and ((witness_only and s == "balanced")
                                              or (iso_witness_only and s == "caption_iso")))
            for s in ALL_STEPS + tuple(x for x in PLANS[plan] if x not in ALL_STEPS)}


def lab_engine_hashes(kind: str) -> dict[str, str]:
    """``certkey.engine_settings_hash`` of the engines the lab builds for ``kind`` now (the
    baseline and the ``dt_bias`` plugin one), on the lab's GPU: what a saved lab result must
    have recorded to count (review 13 S3-1)."""
    from flora_deploy.engines.offline_llm import EngineSpec
    from flora_deploy.validate.certkey import engine_settings_hash

    gpu = LAB[kind].gpu or ""
    return {"baseline": engine_settings_hash(EngineSpec.for_gpu(kind, "", gpu).hashed()),
            "plugin": engine_settings_hash(EngineSpec.for_gpu(kind, "", gpu,
                                                              dtbias_fp32=True).hashed())}


def lab_hash_problem(kind: str, res: Mapping[str, Any]) -> str | None:
    """Why a saved lab result was not measured under the current engine settings (None: it
    was): its baseline engine hash, and its plugin engine's when it built one."""
    got = res.get("engine_hashes") or {}
    want = lab_engine_hashes(kind)
    if got.get("baseline") != want["baseline"]:
        return f"{kind}: measured under other engine settings (lab engine hash " \
            f"{str(got.get('baseline'))[:12]}, now {want['baseline'][:12]})"
    if res.get("dtbias") is not None and got.get("plugin") != want["plugin"]:
        return f"{kind}: plugin engine measured under other engine settings"
    return None


def lab_already_done(lab_dir: Path | None = None,
                     kinds: Sequence[str] = ("tempseg", "captioner")) -> tuple[bool, str]:
    """Whether the lab's L0 + Gate A evidence exists (``balanced-<kind>.json``, as step d2 or
    ``--only lab`` writes it): per kind non-empty L0 rows, every one with equal ids and grids,
    Gate A over at least one request, measured on the balanced pool's GPU **under the current
    engine settings** (:func:`lab_hash_problem`; a lab from before batch fill, or from before
    the hash was recorded, is not done). The certificate takes L0 per model kind, not per
    episode (``validate.certify.l0_summary``), so the certified run reuses it instead of
    paying two cold compiles per kind again."""
    from flora_deploy.engines.offline_llm import EngineArgsError, normalize_gpu

    lab_dir = lab_dir or LAB_OUT
    pool = {k: v for k, v in {"tempseg": "segment", "captioner": "caption"}.items()
            if k in kinds}
    why = []
    for kind, stage in pool.items():
        p = lab_dir / f"balanced-{kind}.json"
        try:
            res = json.loads(p.read_text())
        except (OSError, ValueError):
            why.append(f"{kind}: no {p.name}")
            continue
        if not isinstance(res, dict):
            why.append(f"{kind}: unreadable {p.name}")
            continue
        base = res.get("baseline") or {}
        rows = base.get("l0") or []
        gate = res.get("gate_a") or {}
        err = (res.get("error") or base.get("error") or (res.get("dtbias") or {}).get("error")
               or ("no_evidence" if res.get("no_evidence") else None))
        if err:
            why.append(f"{kind}: error {str(err)[:80]}")
        if not rows:
            why.append(f"{kind}: no L0 rows")
        elif not all(r.get("ids_equal") is True and r.get("grid_equal") is True for r in rows):
            why.append(f"{kind}: L0 not passing")
        if not gate.get("requests"):
            why.append(f"{kind}: no Gate A requests")
        if stale := lab_hash_problem(kind, res):
            why.append(stale)
        gpu = base.get("gpu") or res.get("gpu")
        want = pool_container(stage, "balanced").gpu or ""
        try:
            if gpu and normalize_gpu(str(gpu)) != normalize_gpu(want):
                why.append(f"{kind}: measured on {gpu}, not {want}")
        except EngineArgsError:
            why.append(f"{kind}: unknown GPU {gpu!r}")
    if why:
        return False, "; ".join(why)
    return True, (f"L0 + Gate A evidence for the current engine settings in "
                  f"{lab_dir}/balanced-{{{','.join(pool)}}}.json")


# ---------------------------------------------------------------------------------------------
# State and stop rules


STEPS = ("tempseg", "preflight", "exact_e2e", "balanced", "lab", "certify", "noise")
TITLES = {"tempseg": "0. TempSeg merger v2 build + equivalence (CPU + L40S), then pin",
          "preflight": "a. balanced preflight (CPU)",
          "exact_e2e": "b. exact oracle, 6 human episodes, default mode, H200",
          "balanced": "d. balanced via flora deploy/run (L40S + H100)",
          "lab": "d2. balanced lab L0 / Gate A on (d)'s job (L40S, then H100)",
          "certify": "e. offline provisional certification",
          "noise": "f. noise: oracle job repeated, default mode, H200",
          "noise_all": "f2. noise: (b)'s oracle job repeated on all its episodes, H200",
          "caption_iso": "g. caption-only isolated job on the oracle's segments + cold witness"}
# The certifying plans' own steps (not in the v1 plans: never listed as deferred there)
EXTRA_STEPS = ("noise_all", "caption_iso")
ALL_STEPS = STEPS + EXTRA_STEPS


def cert_titles(spec: PlanSpec) -> dict[str, str]:
    seg_w = " + segment-only cold witness" if spec.segment_only_witness else ""
    out = {**TITLES,
           "exact_e2e": f"b. exact oracle, {len(spec.episodes)} {spec.embodiment} episodes "
                        f"({spec.golden}), nvidia mode, H200",
           "noise": f"f. noise: {len(spec.noise)} of (b)'s episodes, (b)'s stage hash/GPU/mode",
           "balanced": f"d. balanced via flora deploy/run, {len(spec.episodes)} episodes{seg_w}",
           "lab": "d2. lab L0 / Gate A (skipped when already done)",
           "caption_iso": f"g. caption-only isolated J2 ({len(spec.episodes)} ep) + cold witness "
                          f"J2w ({len(spec.witness)} ep) on the oracle's segments",
           "certify": ("e. flora certify, then wave_h_finalize.py" if spec.embodiment == "human"
                       else "e. flora certify (wave_h_finalize.py not run: human only)")}
    if spec.name == RECERTIFY:
        out.update({
            "noise_all": "f2. noise-12: (b) repeated on all its episodes, (b)'s stage "
                         "hash/GPU/mode",
            "lab": "d2'. Captioner lab L0 / Gate A, new engine (TempSeg's reused)",
            "balanced": f"d'. balanced J1 ({len(spec.episodes)} ep) + segment-only cold "
                        f"witness J1w ({len(spec.witness)} ep)",
            "caption_iso": f"g. caption-only isolated J2 ({len(spec.episodes)} ep) + cold "
                           f"witness J2w ({len(spec.witness)} ep), oracle segments",
            "certify": "e'. flora certify (v4), then wave_h_finalize.py"})
    return out


CERT_TITLES = cert_titles(SPECS[CERTIFIED])
RECERT_TITLES = cert_titles(SPECS[RECERTIFY])


def titles(plan: str, spec: PlanSpec | None = None) -> dict[str, str]:
    if not is_cert(plan):
        return TITLES
    if plan == RECERTIFY:
        return RECERT_TITLES if spec is None or spec is SPECS[RECERTIFY] else cert_titles(spec)
    return CERT_TITLES if spec is None or spec is SPECS[CERTIFIED] else cert_titles(spec)


# The state, billing reads, heartbeat and stop rules live in scripts/_runbook.py (shared with
# scale_test.py and rubric_smoke.py); re-exported here, with this runbook's defaults.
Attempt, StepState, State, StopRule, Heartbeat = (rb.Attempt, rb.StepState, rb.State,
                                                   rb.StopRule, rb.Heartbeat)
_SAVE_LOCK = rb._SAVE_LOCK
billed_delta = rb.billed_delta


def load_state(path: Path = STATE) -> State:
    return rb.load_state(path)


def save_state(st: State, path: Path = STATE) -> None:
    rb.save_state(st, path)


Budget, Decision = rb.Budget, rb.Decision
budget_for, reserve_for = rb.budget_for, rb.reserve_for


def check_before(st: State, step: str, proj: Projection, *,
                 caps: Mapping[str, float] = CAPS_USD, global_cap: float = GLOBAL_CAP_USD,
                 reserve: Mapping[str, float] | None = None) -> Budget:
    """The **upper** projection must fit the step's budget (``_runbook.budget_for``; H2)."""
    return rb.check_before(st, step, proj, caps=caps, global_cap=global_cap, reserve=reserve)


def check_after(st: State, step: str, *, caps: Mapping[str, float] = CAPS_USD,
                global_cap: float = GLOBAL_CAP_USD) -> None:
    rb.check_after(st, step, caps=caps, global_cap=global_cap)


def step_budget(st: State, step: str, *, spent_in_step: float = 0.0,
                caps: Mapping[str, float] = CAPS_USD, global_cap: float = GLOBAL_CAP_USD,
                reserve: Mapping[str, float] | None = None) -> float:
    """What the step may still spend (``_runbook.step_budget``)."""
    return rb.step_budget(st, step, spent_in_step=spent_in_step, caps=caps,
                          global_cap=global_cap, reserve=reserve)


def gpu_deadline_s(st: State, step: str, c: Container | float, *, spent_in_step: float = 0.0,
                   slack_s: float = CLIENT_SLACK_S, caps: Mapping[str, float] = CAPS_USD,
                   global_cap: float = GLOBAL_CAP_USD,
                   reserve: Mapping[str, float] | None = None) -> float:
    """Wall seconds still affordable at ``c``'s $/s in this step (``_runbook.gpu_deadline_s``)."""
    return rb.gpu_deadline_s(st, step, c, spent_in_step=spent_in_step, slack_s=slack_s,
                             caps=caps, global_cap=global_cap, reserve=reserve)


def decide(st: State, plan: str, step: str, projections: Mapping[str, Projection], *,
           only: str | None = None, embodiment: str = EMBODIMENT) -> Decision:
    """The decision the dry run, ``--status`` and a real run take for ``step``
    (``_runbook.decide``, on the plan's caps and global cap), with the command that acts on it
    (``--embodiment`` named when it is not the default)."""
    gcap = plan_global_cap(plan, projections, st.override_for(plan))
    d = rb.decide(st, PLANS[plan], step, projections, caps=plan_caps(plan, projections),
                  global_cap=gcap)
    run = f"uv run python scripts/wave_h.py --plan {plan} --yes-spend" + (
        f" --embodiment {embodiment}" if embodiment != EMBODIMENT else "") + (
        f" --only {only}" if only else "")
    if d.action == "run":
        d.command = run
    elif d.action == "wait":
        d.command = f"after {rb.utc(d.clears_at)}: {run}"
    elif d.action == "stop" and d.budget is not None and d.budget.binding != "attempt cap":
        short = projections[step].usd_upper - d.budget.usd
        need = math.ceil((gcap + short) * 20 - 1e-9) / 20
        d.command = f"{run} --global-cap {need:.2f}  (only with the user's approval)"
    return d


# ---------------------------------------------------------------------------------------------
# Billing (scripts/spend.py, through _runbook)


def billed_by_app(since: str) -> dict[str, float]:
    return rb.billed_by_app(since)


def refresh_billing(st: State, *, now: Mapping[str, float] | None = None,
                    clock: Callable[[], float] = time.time) -> bool:
    """Re-read billing for every app seen so far (H3; ``_runbook.refresh_billing``). False when
    billing cannot be read (tracking still applies)."""
    if now is None:
        try:
            now = billed_by_app(st.billing_since)
        except Exception:  # noqa: BLE001 - billing is a cross-check; tracking still applies
            return False
    return rb.refresh_billing(st, now=now, clock=clock)


# ---------------------------------------------------------------------------------------------
# Offline certification metrics (pure; unit-tested)


def _spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float, str]]:
    out = []
    for r in rows:
        a = r.get("start_s", r.get("start"))
        b = r.get("end_s", r.get("end"))
        if a is None or b is None:
            continue
        text = r.get("caption") or r.get("text") or r.get("description") or ""
        out.append((float(a), float(b), str(text)))
    return sorted(out)


def _iou(a: tuple[float, float, str], b: tuple[float, float, str]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _jaccard(x: str, y: str) -> float:
    a, b = set(x.lower().split()), set(y.lower().split())
    return len(a & b) / len(a | b) if a | b else 1.0


def compare_segments(ref: Sequence[Mapping[str, Any]], got: Sequence[Mapping[str, Any]]
                     ) -> dict[str, float]:
    """One episode: segment count delta, boundary MAE (each reference boundary to the nearest
    candidate boundary, seconds), mean best-match IoU of reference segments, and mean caption
    token Jaccard over those best matches."""
    r, g = _spans(ref), _spans(got)
    if not r or not g:
        return {"count_delta": float(len(g) - len(r)), "boundary_mae_s": math.inf,
                "segment_iou": 0.0, "caption_jaccard": 0.0}
    rb = sorted({x for s in r for x in s[:2]})
    gb = sorted({x for s in g for x in s[:2]})
    mae = sum(min(abs(b - c) for c in gb) for b in rb) / len(rb)
    ious, jac = [], []
    for s in r:
        best = max(g, key=lambda t: _iou(s, t))
        ious.append(_iou(s, best))
        jac.append(_jaccard(s[2], best[2]))
    return {"count_delta": float(len(g) - len(r)), "boundary_mae_s": round(mae, 4),
            "segment_iou": round(sum(ious) / len(ious), 4),
            "caption_jaccard": round(sum(jac) / len(jac), 4)}


def aggregate(per_episode: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_episode:
        return {}
    keys = ("boundary_mae_s", "segment_iou", "caption_jaccard")
    out = {k: round(sum(m[k] for m in per_episode.values()) / len(per_episode), 4)
           for k in keys}
    out["abs_count_delta"] = round(sum(abs(m["count_delta"]) for m in per_episode.values())
                                   / len(per_episode), 4)
    out["episodes"] = len(per_episode)
    return out


def certify(balanced: Mapping[str, float], *, l0: Sequence[Mapping[str, Any]] = (),
            gate_a: Mapping[str, Any] | None = None,
            noise: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Provisional verdict: L0 parity on every captured request, Gate A within bounds, and the
    balanced-vs-exact metrics within ``NOISE_FACTOR`` x the exact noise floor (or
    :data:`ABS_BOUNDS` before one exists)."""
    checks: dict[str, Any] = {}
    checks["l0"] = bool(l0) and all(r.get("ids_equal") and r.get("grid_equal") for r in l0)
    if gate_a:
        mx = gate_a.get("max_abs_dlogit")
        checks["gate_a"] = mx is not None and float(mx) <= GATE_A_MAX_DLOGIT
    else:
        checks["gate_a"] = False
    for k, bound in ABS_BOUNDS.items():
        v = balanced.get(k)
        if v is None:
            checks[k] = False
            continue
        if noise and noise.get(k) is not None:
            n = float(noise[k])
            if k == "boundary_mae_s":
                bound = max(NOISE_FACTOR * n, 0.1)
            else:  # similarity: allowed drop below 1 is NOISE_FACTOR x the noise drop
                bound = 1.0 - NOISE_FACTOR * (1.0 - n)
        checks[k] = (v <= bound) if k == "boundary_mae_s" else (v >= bound)
    return {"provisional": all(checks.values()), "checks": checks,
            "against": "noise floor" if noise else "absolute bounds", "metrics": dict(balanced),
            "noise": dict(noise or {})}




# ---------------------------------------------------------------------------------------------
# Job records for wave_h_finalize.py (pure; contract-tested against Job.report())


def clip_containers(rows: Iterable[Mapping[str, Any]], stopped_at: float | None
                    ) -> list[dict[str, Any]]:
    """``flora report``'s containers (``jobs.containers``: ``kind``, ``gpu``, ``startup_s``,
    ``busy_s``, ``seconds``) with ``seconds`` / ``usd`` cut at the app stop (``modal app stop``
    ends every container; the report only knows the scale-down window)."""
    from flora_deploy.jobs.containers import parse_utc

    out = []
    for c in rows:
        c = dict(c)
        born = parse_utc(c.get("started_utc"))
        if stopped_at is not None and born is not None and c.get("seconds") is not None:
            if c.get("startup_s") is not None:
                born -= float(c["startup_s"])
            cut = max(0.0, stopped_at - born)
            if cut < float(c["seconds"]):
                if c.get("usd"):
                    c["usd"] = round(float(c["usd"]) * cut / float(c["seconds"]), 4)
                c["seconds"] = round(cut, 1)
        out.append(c)
    return out


def job_record(report: Mapping[str, Any], *, stopped_at: float | None = None
               ) -> dict[str, Any]:
    """The fields of a job's state entry that come from ``flora report --json``
    (``sdk.Job.report``): ``state``, ``spend``, ``video_hours`` / ``video_s``, $/video-h and the
    per-container rows ``wave_h_finalize.measure`` reads."""
    vh = report.get("video_hours") or {}
    sp = report.get("spend") or {}
    return {"state": report.get("state"), "spend": sp.get("total_usd"),
            "usd_per_video_hour": report.get("usd_per_video_hour"), "video_hours": vh,
            "video_s": round(float(vh.get("done") or 0.0) * 3600.0, 2),
            "containers": clip_containers(sp.get("containers") or [], stopped_at)}


def record_paths(side: str, job: str = JOB, root: Path | None = None) -> dict[str, str]:
    """Where ``wave_h_finalize.py --fetch`` exports a job's records (its ``record_path``);
    ``root``: the plan's records directory (default :data:`RECORDS`)."""
    return {stage: str((root or RECORDS) / side / job / f"{stage}.jsonl")
            for stage in ("segment", "caption")}


# ---------------------------------------------------------------------------------------------
# Step runners (GPU steps only with --yes-spend)


@dataclass
class Ctx:
    st: State
    manifest: dict[str, Any]
    work: Path
    log: Callable[[str], None] = print
    state_path: Path = STATE
    step: str = ""
    global_cap: float = GLOBAL_CAP_USD  # the plan's (plan_global_cap)
    spec: PlanSpec = field(default_factory=lambda: SPECS[DEFAULT_PLAN])
    caps: dict[str, float] = field(default_factory=lambda: dict(CAPS_USD))  # plan_caps
    # the plan's steps and projections: what the global cap keeps for its later steps
    plan_steps: tuple[str, ...] = ()
    projections: Mapping[str, Projection] | None = None
    # added to the environment of this step's commands only (the cold witness's
    # FLORA_COMPILE_CACHE=off): never os.environ, which a step beside it would inherit
    env: dict[str, str] = field(default_factory=dict)

    @property
    def episodes(self) -> tuple[str, ...]:
        return episodes_in(self.manifest, self.spec.episodes)

    def reserve(self, step: str) -> dict[str, float]:
        return reserve_for(self.st, self.plan_steps, step, self.projections, self.caps)

    def budget_of(self, step: str, **kw: Any) -> Budget:
        """The running attempt's budget: the same rule as the stop rule before it."""
        return budget_for(self.st, step, caps=self.caps, global_cap=self.global_cap,
                          reserve=self.reserve(step), **kw)

    def deadline_s(self, step: str, c: Container | float, **kw: Any) -> float:
        return gpu_deadline_s(self.st, step, c, caps=self.caps, global_cap=self.global_cap,
                              reserve=self.reserve(step), **kw)

    def budget(self, step: str, **kw: Any) -> float:
        return self.budget_of(step, **kw).usd

    def sh(self, args: list[str], *, timeout: float = 1800, check: bool = True):
        return h0.sh(args, cwd=self.work, timeout=timeout, check=check,
                     log=self.work / "wave-h.log", **({"env": self.env} if self.env else {}))

    def note_app(self, name: str) -> None:
        """Record an app in the running attempt *before* it can spend (H3). A new app means
        the attempt is no longer known to be stopped."""
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None:
            if name not in cur.apps:
                cur.apps.append(name)
            cur.stopped_at = None
            save_state(self.st, self.state_path)

    def app_name(self, part: str = "") -> str:
        """This attempt's own app name (:func:`app_name`), ``part`` appended for a step that
        starts more than one app (step 0: merge, then equivalence)."""
        n = max(1, len(self.st.step(self.step).attempts))
        name = app_name(self.st.run_id, self.spec.job_prefix + self.step, n)
        return name + (f"-{part}" if part else "")

    def note_app_id(self, app_id: str | None) -> None:
        """Record a started app's id (``ap-...``): it is stopped and billed by id."""
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None and app_id and app_id not in cur.app_ids:
            cur.app_ids.append(app_id)
            save_state(self.st, self.state_path)

    def confirm_stop(self) -> None:
        """Every app the running attempt started is confirmed stopped (the ``with app.run()``
        block exited, or a stop returned 0): until billing covers it, a failure is charged its
        tracked time plus the slack, not its budget (``_runbook.Attempt.bounded_usd``)."""
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None:
            cur.stopped_at = time.time()
            save_state(self.st, self.state_path)

    def receipt_rows(self) -> list[dict[str, Any]]:
        """``flora prepare --show-receipt`` (a Volume read; no container)."""
        from flora_deploy.cli.prepare import show_receipt_rows

        return show_receipt_rows()


def resolve_receipts(names: Iterable[str], *, read: Callable[[str], bytes | None] | None = None,
                     require_pin: bool = False) -> dict[str, str]:
    """``weights.derived`` for ``names``, as ``flora prepare`` records it (H1): each receipt read
    from the ``flora-weights`` Volume, its content digest checked against ``pins.DERIVED_PINS``
    (``config.receipts.check_pinned``), its file sha256 returned. ``require_pin``: also refuse a
    derived weight without a pinned content digest, with ``config.receipts.unpinned_message``
    (``plan_app`` and the balanced workers refuse it too)."""
    from flora_deploy.config.receipts import check_pinned, unpinned, unpinned_message
    from flora_deploy.prep.derived import DERIVED_BUILDS, receipt_sha256

    if read is None:
        from flora_deploy.cli.prepare import _read_weights_file as read
    if require_pin:
        bad = unpinned(names)
        if bad:  # before any Volume read: nothing has started
            raise RuntimeError(" ".join(unpinned_message(n, preset="balanced") for n in bad))
    out: dict[str, str] = {}
    for name in sorted(set(names)):
        path = DERIVED_BUILDS[name].receipt_path()
        raw = read(path)
        if raw is None:
            raise RuntimeError(f"{name}: no receipt at flora-weights:{path} (run `flora "
                               "prepare`)")
        check_pinned(name, json.loads(raw), where=f"flora-weights:{path}")
        out[name] = receipt_sha256(raw)
    return out


def tempseg_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(next((r for r in rows if r.get("key") == "tempseg_merged"), {}))


def require_tempseg_pin(ctx: Ctx) -> dict[str, Any]:
    """Refuse (before any app) until step 0's TempSeg receipt matches the pin."""
    from flora_deploy.config.receipts import unpinned_message

    row = tempseg_row(ctx.receipt_rows())
    if row.get("content_sha256_pinned") is None:
        raise RuntimeError("step 0 has not been pinned: " + unpinned_message(
            "tempseg_merged", kinds=("segment",), preset="balanced"))
    if row.get("matches_pin") is not True:
        raise RuntimeError(f"tempseg_merged on flora-weights has content digest "
                           f"{row.get('content_sha256')}, not the pinned "
                           f"{row.get('content_sha256_pinned')} ({row.get('error') or 'rebuilt?'}"
                           "); re-run step 0 (`--only tempseg`)")
    return row


def _gdn_pass(status: Any) -> bool:
    """NVIDIA's ``verify_gdn`` reports ``PASS`` or a qualified pass such as
    ``PASS_NATIVE_GDN_FP32_CHECKPOINT_VALUES``; anything else is a failure."""
    v = str(status).upper()
    return v == "PASS" or v.startswith("PASS_")


def equivalence_ok(res: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """``convert_app.tempseg_equivalence``: every window has identical inputs and greedy ids,
    ``verify_gdn`` passes on B, and the merged files verify (docs/tempseg-merge.md)."""
    wins = list(res.get("windows") or [])
    gdn = res.get("b_gdn") or {}
    summary = {
        "windows": len(wins),
        "inputs_identical": all(w.get("inputs_identical") for w in wins),
        "ids_identical": all(w.get("ids_identical") for w in wins),
        "json_equal": all(w.get("json_equal") for w in wins),
        "gdn": {k: v for k, v in gdn.items()},
        "verify_merged": bool(res.get("verify_merged")),
        "max_abs_dlogit_tf": max((float(((w.get("teacher_forced_on_a") or {}).get("a_vs_b")
                                         or {}).get("max_abs", 0.0)) for w in wins),
                                 default=None),
        "total_s": res.get("total_s"),
    }
    ok = (bool(wins) and summary["inputs_identical"] and summary["ids_identical"]
          and bool(gdn) and all(_gdn_pass(v) for v in gdn.values())
          and summary["verify_merged"])
    return ok, summary


def _modal_run(ctx: Ctx, target: str, args: list[str], *, limit: float, out: Path,
               name: str) -> tuple[dict[str, Any] | None, float, str | None]:
    """``modal run -n <name> <target> <args> --out <out>`` with a client deadline: the run's
    own app name (billing reports it as the app's description). The app is then stopped by
    id (``modal app stop <name>`` finds no ephemeral app) either way, and its id recorded.
    Returns (result JSON, wall s, error)."""
    t0 = time.time()
    err = None
    ctx.note_app(name)
    try:
        ctx.sh(["modal", "run", "-n", name, target, *args, "--out", str(out)], timeout=limit,
               check=True)
    except subprocess.TimeoutExpired:
        err = f"client deadline {limit:.0f} s"
    except RuntimeError as e:
        err = str(e)
    finally:
        run = _sh_runner(ctx)
        try:
            ids = rb.app_ids_named(name, rb.list_apps(run) or [])
        except rb.AppListError as e:  # stop_apps below says so too and stops by name
            print(f"warning: {e}", file=sys.stderr)
            ids = []
        for i in ids:
            ctx.note_app_id(i)
        if rb.stop_apps([name], ids, run=run):
            ctx.confirm_stop()
    wall = time.time() - t0
    try:
        res = json.loads(out.read_text()) if out.exists() and err is None else None
    except ValueError as e:
        res, err = None, f"unreadable {out.name}: {e}"
    return res, wall, err


def _sh_runner(ctx: Ctx) -> Callable[..., Any]:
    """``ctx.sh`` in the shape ``_runbook.stop_apps`` calls ``subprocess.run``."""
    def run(args: list[str], **kw: Any) -> Any:
        return ctx.sh(args, check=False, timeout=kw.get("timeout", 120))
    return run


def run_tempseg(ctx: Ctx) -> dict[str, Any]:
    """Step 0 (module docstring): build, check, report the digest, stop for the pin."""
    rows = ctx.receipt_rows()
    row = tempseg_row(rows)
    pinned = row.get("content_sha256_pinned")
    if pinned and row.get("matches_pin") is True:
        return {"ok": True, "pinned": pinned, "receipt_sha256": row.get("receipt_sha256"),
                "tracked_usd": 0.0, "apps": [], "built": False}
    prev = ctx.st.step("tempseg").result
    if (prev.get("content_sha256") and row.get("content_sha256") == prev["content_sha256"]
            and prev.get("equivalence_ok")):  # already built and checked: never rebuild
        return {**prev, "ok": False, "needs_pin": True, "tracked_usd": 0.0, "apps": [],
                "built": False, "pin_mismatch": pinned}
    saved = ctx.work / "tempseg-equivalence.json"
    if (prev.get("content_sha256") and row.get("content_sha256") == prev["content_sha256"]
            and saved.is_file()):  # same build already checked on the GPU: re-score, no re-run
        ok, summary = equivalence_ok(json.loads(saved.read_text()))
        if ok:
            return {**prev, "equivalence": {**summary, "rescored": True}, "equivalence_ok": True,
                    "ok": False, "needs_pin": True, "error": None, "tracked_usd": 0.0,
                    "apps": [], "built": False}
    target = str(ROOT / "src" / "flora_deploy" / "modal_app" / "convert_app.py") + "::tempseg"
    spent = 0.0
    names = {"merge": ctx.app_name("merge"), "equivalence": ctx.app_name("equivalence")}
    ctx.note_app(names["merge"])
    out: dict[str, Any] = {"apps": [names["merge"]], "built": True}
    rdir = ctx.work
    limit = min(MERGE_MAX_S, ctx.deadline_s("tempseg", MERGE, slack_s=LAB_SLACK_S))
    if limit < 120:
        return {**out, "ok": False, "error": f"budget left allows {limit:.0f} s",
                "tracked_usd": 0.0}
    merge, wall, err = _modal_run(ctx, target, ["--step", "merge", "--force"], limit=limit,
                                  out=rdir / "tempseg-merge.json", name=names["merge"])
    spent += MERGE.usd_per_s() * wall
    out["merge"] = {"wall_s": round(wall, 1), "error": err,
                    "receipt": (merge or {}).get("receipt_sha256")}
    if err:
        return {**out, "ok": False, "error": f"merge: {err}", "tracked_usd": spent}
    limit = min(EQUIV_MAX_S, ctx.deadline_s("tempseg", EQUIV, spent_in_step=spent,
                                            slack_s=LAB_SLACK_S))
    if limit < 120:
        return {**out, "ok": False, "error": f"budget left allows {limit:.0f} s for the "
                "equivalence check", "tracked_usd": spent}
    out["apps"].append(names["equivalence"])
    eq, wall, err = _modal_run(ctx, target, ["--step", "equivalence"], limit=limit,
                               out=rdir / "tempseg-equivalence.json", name=names["equivalence"])
    spent += EQUIV.usd_per_s() * wall
    ok, summary = equivalence_ok(eq or {}) if not err else (False, {"error": err})
    out.update(equivalence={**summary, "wall_s": round(wall, 1)}, equivalence_ok=ok)
    row = tempseg_row(ctx.receipt_rows())
    out.update(content_sha256=row.get("content_sha256"), receipt_sha256=row.get(
        "receipt_sha256"), version=row.get("version"), tracked_usd=spent)
    if not ok or not row.get("content_sha256"):
        return {**out, "ok": False, "error": err or "equivalence check failed: do not pin"}
    if row.get("version") != 2:
        return {**out, "ok": False, "error": f"receipt merger version {row.get('version')}, "
                "not 2: do not pin"}
    return {**out, "ok": False, "needs_pin": True}


PINS_FILE = "src/flora_deploy/config/pins.py"
UNPINNED_LINE = "        content_sha256=None,"  # in DerivedPin(key="tempseg-merged", ...)


def pin_edit(digest: str | None) -> tuple[str, str]:
    """The one line of :data:`PINS_FILE` to change: (old, new)."""
    return UNPINNED_LINE, f'        content_sha256="{digest}",'


def pin_instructions(res: Mapping[str, Any]) -> str:
    old, new = pin_edit(res.get("content_sha256"))
    return (f"STOP: pin TempSeg before steps (a) and (d). The one-line edit, in {PINS_FILE}, "
            'inside DerivedPin(key="tempseg-merged", ...) (DERIVED_PINS["tempseg-merged"]'
            ".content_sha256):\n"
            f"  -{old}\n"
            f"  +{new}\n"
            f"(receipt file sha256 {res.get('receipt_sha256')}; `flora prepare --show-receipt` "
            "prints both), record the equivalence result in docs/tempseg-merge.md, commit, then "
            "re-run `scripts/wave_h.py --yes-spend` from that commit. Steps that need TempSeg "
            "refuse until the pin matches.")


def run_preflight(ctx: Ctx) -> dict[str, Any]:
    import modal

    from flora_deploy.modal_app import balanced_lab as lab

    require_tempseg_pin(ctx)
    t0 = time.time()
    name = ctx.app_name()
    with _lab_app(ctx, lab, modal, name):
        out = lab.preflight.remote()
    wall = time.time() - t0
    ok = all(out.get(k, {}).get("receipt") for k in ("tempseg", "captioner"))
    return {"ok": ok, "wall_s": round(wall, 1), "tracked_usd": PREFLIGHT.usd_per_s() * wall,
            "apps": [name],
            "facts": {k: {kk: v.get(kk) for kk in ("receipt", "eos_token",
                                                    "extra_state_prefixes")}
                      for k, v in out.items() if isinstance(v, dict)}}


def stage_local(ctx: Ctx, data: str, ids: Sequence[str]) -> Path:
    """``work/<data>/`` as an ``mp4_dir`` of ``ids`` (``h0.fetch_dataset``), each episode named
    by the plan's :meth:`PlanSpec.local_id` (certified: noise episodes sort first). Idempotent:
    an episode already laid out under its local id is not fetched again."""
    root = ctx.work / data
    todo = [g for g in ids if not ((root / f"{ctx.spec.local_id(g)}.mp4").is_file()
                                   or (root / ctx.spec.local_id(g) / "main.mp4").is_file())]
    if todo:
        h0.fetch_dataset(ctx.manifest, ctx.work, episodes={data: tuple(todo)}, poison_job=None)
    for g in todo:
        base, local = h0.episode_id(g), ctx.spec.local_id(g)
        if base == local:
            continue
        if (root / base).is_dir():  # three views: <id>/{main,left_wrist,right_wrist}.mp4
            os.replace(root / base, root / local)
            continue
        for ext in (".mp4", ".txt"):
            os.replace(root / f"{base}{ext}", root / f"{local}{ext}")
    return root


# The certified plan's jobs share one data dir: (b) and (d), which may run at once, stage it
# one at a time, and it is uploaded once per run (a second upload would rewrite the files a
# running job reads).
_STAGE_LOCK = threading.Lock()
_UPLOADED: set[tuple[str, str]] = set()


def _stage_job(ctx: Ctx, job: str, ids: Sequence[str], *, embodiment: str,
               stages: tuple[str, ...], budget: float, preset: str = "exact",
               resources: dict[str, str] | None = None,
               derived: dict[str, str] | None = None, data: str | None = None
               ) -> tuple[Path, str]:
    """Fetch, upload, write the config, ``flora index --local`` + mirror, ``flora plan``.
    ``data``: the data directory (default ``job``); the certified plan stages every job from
    one (:data:`CERT_DATA`), so their episode uids pair."""
    data = data or job
    stage_data(ctx, data, ids)
    text, job_id = config_text(ctx, job, embodiment=embodiment, stages=stages, budget=budget,
                               preset=preset, resources=resources, derived=derived, data=data)
    cfg = ctx.work / f"flora-{job}.yaml"
    cfg.write_text(text)
    _index(ctx, cfg, job_id, job)
    return cfg, job_id


def stage_data(ctx: Ctx, data: str, ids: Sequence[str]) -> None:
    """``work/<data>/`` staged (:func:`stage_local`) and uploaded to the media Volume, one step
    at a time and uploaded once per process (:data:`_STAGE_LOCK`)."""
    run = ctx.st.run_id
    with _STAGE_LOCK:
        stage_local(ctx, data, ids)
        key = (str(ctx.work / data), f"{run}/{data}")
        # recertify stages nothing new: the data dir is (b)'s, uploaded by (b) (its content is
        # in every stage hash, checked equal to (b)'s before f2's app)
        if key not in _UPLOADED and ctx.spec.name != RECERTIFY:
            h0.upload_dir(MEDIA_VOLUME, ctx.work / data, f"{run}/{data}")
            _UPLOADED.add(key)


def config_text(ctx: Ctx, job: str, *, embodiment: str, stages: tuple[str, ...], budget: float,
                preset: str = "exact", resources: dict[str, str] | None = None,
                derived: dict[str, str] | None = None, data: str | None = None
                ) -> tuple[str, str]:
    """A job's config text and job id (``wave-h-<run>-<job>``, also its ``project``)."""
    run = ctx.st.run_id
    job_id = f"wave-h-{run}-{job}"
    return h0.config_text(project=job_id, data_dir=data or job, embodiment=embodiment,
                          stages=stages, store_volume=STORE_VOLUME, media_volume=MEDIA_VOLUME,
                          run_id=run, budget_usd=budget, resources=resources,
                          tag="flora-wave-h", preset=preset, derived=derived,
                          header="Wave H-lite; generated by scripts/wave_h.py"), job_id


def _index(ctx: Ctx, cfg: Path, job_id: str, tag: str, *, limit: int | None = None) -> None:
    """``flora index --local`` into a local store, mirrored into the job store, then ``flora
    plan``. ``limit``: index only the first N episodes of the data dir (the certified noise
    job: the same config and data dir as the oracle, so the same stage hash)."""
    fl = h0.flora_bin()
    local = ctx.work / f"store-{tag}"
    ctx.sh([fl, "index", "--local", "--config", str(cfg), "--job", job_id, "--store",
            f"file://{local}", "--out", str(ctx.work / f"summary-{tag}.json")]
           + (["--limit", str(limit)] if limit is not None else []))
    h0.mirror_index(local, job_id, ctx.st.run_id, STORE_VOLUME)
    ctx.sh([fl, "plan", "--config", str(cfg), "--index", str(ctx.work / f"summary-{tag}.json")])


def store_uri(st: State) -> str:
    return f"modal-volume://{STORE_VOLUME}/{st.run_id}"


def _run_job(ctx: Ctx, step: str, cfg: Path, job_id: str, *, determinism: str,
             usd_per_s: float, spent_in_step: float, side: str) -> dict[str, Any]:
    """``flora deploy`` + ``flora run`` with the in-step stop rule, then ``modal app stop``.
    Receipts come from the config's ``weights.derived`` (``config.receipts.resolve``), as for
    ``flora deploy``. Records the fields ``wave_h_finalize.py`` reads (module docstring)."""
    from flora_deploy.config.receipts import resolve
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import plan_app

    fl = h0.flora_bin()
    c = load_config(cfg)
    plan = plan_app(c, determinism=determinism, derived_receipts=resolve(c))  # refuses first
    deadline = ctx.deadline_s(step, usd_per_s, spent_in_step=spent_in_step)
    budget = max(0.01, ctx.budget(step, spent_in_step=spent_in_step))
    out: dict[str, Any] = {"app": plan.app_name, "job_id": job_id, "deadline_s": round(deadline),
                           "budget_usd": round(budget, 4), "cfg": str(cfg),
                           # what its records' provenance.config_hash must be (review 12 F5)
                           "config_hash": plan.config_hash,
                           "stage_hashes": dict(plan.stage_hashes),
                           "preset": c.preset, "records": record_paths(side,
                                                                       root=ctx.spec.records)}
    if deadline < 60:
        out["state"] = "skipped"
        out["error"] = f"budget left allows {deadline:.0f} s"
        return out
    ctx.note_app(plan.app_name)
    ctx.sh([fl, "deploy", "--config", str(cfg), "--determinism", determinism], timeout=1800)
    try:
        args = [fl, "run", "--config", str(cfg), "--determinism", determinism, "--job", job_id,
                "--poll", "15", "--budget", f"{budget:.2f}"]
        try:
            p = ctx.sh(args, timeout=deadline, check=False)
            out["rc"] = p.returncode
        except subprocess.TimeoutExpired:
            out["rc"] = "deadline"
            ctx.sh([fl, "cancel", job_id, "--yes", "--config", str(cfg)], check=False,
                   timeout=120)
    finally:
        p = ctx.sh(["modal", "app", "stop", "--yes", plan.app_name], check=False, timeout=120)
        out["stopped_at"] = time.time()
        if getattr(p, "returncode", 1) == 0:  # a deployed app: stopped by its name
            ctx.confirm_stop()
    rep = ctx.sh([fl, "report", job_id, "--config", str(cfg), "--json"], check=False,
                 timeout=900)
    try:
        out.update(job_record(json.loads(rep.stdout), stopped_at=out["stopped_at"]))
    except ValueError:
        out["report_error"] = (rep.stdout + rep.stderr)[-400:]
    if out.get("spend") is None:  # unknown: charge what it was allowed (H3)
        out["report_error"] = out.get("report_error") or "no spend in flora report"
        out["spend"] = budget
    return out


def _jobs_result(jobs: Mapping[str, Mapping[str, Any]], n: int) -> dict[str, Any]:
    spent = sum(float(r.get("spend") or 0.0) for r in jobs.values())
    return {"ok": len(jobs) == n and all(r.get("state") == "done" for r in jobs.values()),
            "jobs": dict(jobs), "tracked_usd": spent,
            "report_error": any(r.get("report_error") for r in jobs.values()),
            "apps": [r["app"] for r in jobs.values() if "app" in r]}


def oracle_job(ctx: Ctx, step: str = "exact_e2e") -> dict[str, Any]:
    """(b)'s job: its name, episodes and config inputs. (b) runs it; the certified (f) builds
    the same config from them (:func:`run_cert_noise`), so the same stage hash."""
    derived = resolve_receipts(["captioner_bf16"])
    eps = ctx.episodes
    if ctx.spec.shared_data:  # certified: every episode, at least the certificate's minimum
        eps = balanced_episodes(ctx.manifest, ctx.spec, side="(b)")
    return {"job": f"{step}-{ctx.spec.embodiment}", "ids": eps,
            "embodiment": ctx.spec.embodiment, "stages": ("segment", "caption"),
            "budget": ctx.caps[step], "derived": derived, "data": ctx.spec.data_dir(step)}


def run_exact(ctx: Ctx, step: str = "exact_e2e") -> dict[str, Any]:
    """(b): the oracle, default mode, one app."""
    require_tempseg_pin(ctx)  # every GPU step runs from the pinned commit
    o = oracle_job(ctx, step)
    eps = o["ids"]
    cfg, job_id = _stage_job(ctx, o["job"], eps, embodiment=o["embodiment"],
                             stages=o["stages"], budget=o["budget"], derived=o["derived"],
                             data=o["data"])
    r = _run_job(ctx, step, cfg, job_id, determinism=ORACLE_DETERMINISM,
                 usd_per_s=pool_container("segcap").usd_per_s(), spent_in_step=0.0,
                 side="exact")
    r["data"] = ctx.spec.data_dir(step)
    return {**_jobs_result({JOB: r}, 1), "episodes": len(eps)}


def balanced_episodes(manifest: Mapping[str, Any], spec: PlanSpec | None = None, *,
                      side: str = "(d)") -> tuple[str, ...]:
    """(d)'s episodes: the plan's golden selection (default :data:`EPISODES`), in that order,
    read from nothing (b) wrote. All must be staged: refused (before any app) otherwise. The
    certified plan also needs at least :data:`MIN_CERT_EPISODES` of them, and its noise subset
    of at least :data:`MIN_NOISE_EPISODES` inside them."""
    spec = spec or SPECS[DEFAULT_PLAN]
    miss = missing_episodes(manifest, spec.episodes)
    if miss:
        raise RuntimeError(f"{side} runs all {len(spec.episodes)} golden {spec.embodiment} "
                           f"episodes; not staged: {', '.join(miss)}")
    if spec.shared_data:
        if len(spec.episodes) < MIN_CERT_EPISODES:
            raise RuntimeError(f"{side}: {len(spec.episodes)} episodes; a certificate needs at "
                               f"least {MIN_CERT_EPISODES}")
        for what, eps in (("noise subset", spec.noise), ("witness subset", spec.witness)):
            if len(eps) < MIN_NOISE_EPISODES or not set(eps) <= set(spec.episodes):
                raise RuntimeError(f"{side}: the {what} must be at least "
                                   f"{MIN_NOISE_EPISODES} of the oracle's episodes")
    return spec.episodes


def run_balanced(ctx: Ctx) -> dict[str, Any]:
    """(d): balanced through the real ``flora deploy`` / ``flora run`` on the 6 golden episodes
    (:func:`balanced_episodes`; independent of (b)). The lab's L0 + Gate A on that job is its
    own step (:func:`run_lab`, ``--only lab``), so a lab retry never redoes this job."""
    from flora_deploy.config.receipts import derived_content as content_of
    from flora_deploy.config.schema import load_config

    if ctx.spec.name == RECERTIFY:
        require_code_landed("balanced")
    require_tempseg_pin(ctx)
    rate = (pool_container("segment", "balanced").usd_per_s()
            + pool_container("caption", "balanced").usd_per_s())
    if ctx.spec.shared_data and witness_pending(ctx.st):
        return rerun_cold_witness(ctx, rate)
    episodes = balanced_episodes(ctx.manifest, ctx.spec)
    derived = resolve_receipts(["captioner_bf16", "tempseg_merged"], require_pin=True)
    cfg, job_id = _stage_job(ctx, f"{ctx.spec.job_prefix}balanced-{ctx.spec.embodiment}", episodes,
                             embodiment=ctx.spec.embodiment,
                             stages=("segment", "caption"), budget=ctx.caps["balanced"],
                             preset="balanced", derived=derived,
                             data=ctx.spec.data_dir("balanced"))
    r = _run_job(ctx, "balanced", cfg, job_id, determinism=ORACLE_DETERMINISM, usd_per_s=rate,
                 spent_in_step=0.0, side="balanced")
    jobs = {JOB: r}
    e2e = _jobs_result(jobs, 1)
    out: dict[str, Any] = {"episodes": list(episodes),
                           "derived_content": content_of(load_config(cfg))}
    if not e2e["ok"]:  # L2: the lab step refuses a job that did not finish
        out["lab"] = (f"not run: deployed balanced job ended {r.get('state')!r}"
                      + (f" ({r['error']})" if r.get("error") else ""))
    elif ctx.spec.shared_data:  # certified: the cold witness (review 13 W1)
        w = run_cold_witness(ctx, cfg, rate, spent=float(e2e["tracked_usd"]))
        out["witness_jobs"] = {JOB: w}
        e2e["tracked_usd"] = float(e2e["tracked_usd"]) + float(w.get("spend") or 0.0)
        e2e["apps"] = sorted(set(e2e["apps"]) | ({w["app"]} if "app" in w else set()))
        if w.get("state") != "done":
            e2e["ok"] = False
            out["error"] = (f"cold witness job ended {w.get('state')!r}"
                            + (f" ({w['error']})" if w.get("error") else ""))
    return {**e2e, **out}


def witness_config_text(text: str, project: str,
                        stages: Sequence[str] | None = None) -> str:
    """(d)'s config text with its ``project`` replaced: the cold witness's own job id and app
    name. The project name is not hashed (``config.hashing``), so the stage hash stays (d)'s.
    ``stages``: also keep only these pipeline stages (J1w: ``("segment",)``) and drop the
    ``resources`` lines of the shard kinds they no longer run. The segment stage hash does not
    depend on whether ``caption`` is in the job (``config.hashing._stage_hashes``)."""
    lines = text.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.startswith("project:")]
    if len(hits) != 1:
        raise RuntimeError(f"cold witness: {len(hits)} top-level project lines in (d)'s config")
    lines[hits[0]] = f"project: {project}\n"
    if stages is not None:
        st = [i for i, ln in enumerate(lines) if ln.startswith("  stages: [")]
        if len(st) != 1:
            raise RuntimeError(f"cold witness: {len(st)} pipeline stages lines in (d)'s config")
        lines[st[0]] = f"  stages: [{', '.join(stages)}]\n"
        res = next((i for i, ln in enumerate(lines) if ln.startswith("resources:")), None)
        if res is not None:
            end = next((i for i in range(res + 1, len(lines))
                        if not lines[i].startswith("  ")), len(lines))
            keep = [ln for ln in lines[res + 1:end]
                    if ln.split(":", 1)[0].strip() in stages]
            lines[res + 1:end] = keep
    return "".join(lines)


def with_caption_segments(text: str, *, records: str, sha256: str,
                          oracle_stage_hash: str) -> str:
    """A ``stages: [caption]`` config text with ``pipeline.caption_segments`` (the pinned
    oracle segment records the caption shards read instead of upstream segment shards)."""
    lines = text.splitlines(keepends=True)
    st = [i for i, ln in enumerate(lines) if ln.startswith("  stages: [caption]")]
    if len(st) != 1:
        raise RuntimeError("caption_segments: the config is not one `stages: [caption]` job")
    if any(ln.startswith("  caption_segments:") for ln in lines):
        raise RuntimeError("caption_segments: already set")
    lines[st[0] + 1:st[0] + 1] = [
        "  caption_segments:\n", f"    records: {records}\n", f"    sha256: {sha256}\n",
        f"    oracle_stage_hash: {oracle_stage_hash}\n"]
    return "".join(lines)


def run_cold_witness(ctx: Ctx, cfg: Path, rate: float, *, spent: float) -> dict[str, Any]:
    """(d)'s **cold witness**: (d)'s own config (its ``project`` renamed, so this attempt has its
    own job and app: ``wave-h-<run>-cold-<embodiment>-a<attempt>``) and data dir, re-indexed
    with ``flora index --limit`` to the noise subset (the same stage hash, as the noise job does
    for the oracle), then deployed and run with ``FLORA_COMPILE_CACHE=off`` in the deploying
    process's env (``modal_app.factory`` forwards it), so its one container per kind
    (``max_containers: 1``) compiles cold and publishes nothing. A warm start replays the
    publisher's Inductor autotune picks; the witness is the independent cold compile the
    certificate's evidence must also cover (``docs/certification.md``). Its budget and deadline
    are (d)'s attempt's, less what (d)'s job spent (``_runbook.budget_for``)."""
    emb = ctx.spec.embodiment
    n = max(1, len(ctx.st.step("balanced").attempts))
    job_id = f"wave-h-{ctx.st.run_id}-{ctx.spec.job_prefix}{COLD_JOB}-{emb}-a{n}"
    tag = f"{ctx.spec.job_prefix}{COLD_JOB}-{emb}"
    wcfg = ctx.work / f"flora-{tag}.yaml"
    seg_only = ctx.spec.segment_only_witness
    stages = ("segment",) if seg_only else None
    wcfg.write_text(witness_config_text(cfg.read_text(), job_id, stages))
    _index(ctx, wcfg, job_id, tag, limit=ctx.spec.witness_limit)
    got = noise_index_problems(ctx.work / f"summary-{tag}.json", ctx.manifest, ctx.spec,
                               ctx.spec.witness)
    if got:
        return {"job_id": job_id, "state": "skipped", "error": f"witness index: {got}",
                "spend": 0.0, "compile_cache": "off"}
    if seg_only:  # one pool: the segment container's $/s alone
        rate = pool_container("segment", "balanced").usd_per_s()
    # in this step's commands' environment only (Ctx.env): a step running beside it keeps
    # its compile cache
    wctx = replace(ctx, env={**ctx.env, COMPILE_CACHE_ENV: "off"})
    r = _run_job(wctx, "balanced", wcfg, job_id, determinism=ORACLE_DETERMINISM,
                 usd_per_s=rate, spent_in_step=spent, side="witness")
    out = {**r, "compile_cache": "off", "episodes": len(ctx.spec.witness)}
    if seg_only:
        out["stages"] = list(stages)
    return out


def rerun_cold_witness(ctx: Ctx, rate: float) -> dict[str, Any]:
    """(d) re-entered with its deployed job done and its cold witness not
    (:func:`witness_pending`): run the witness alone (its own job and app, this attempt's
    number) on the job's saved config; the job's record (and so its records) is reused. This
    attempt's spend is the witness's."""
    prev = ctx.st.step("balanced").result
    main = dict(prev["jobs"][JOB])
    cfg = Path(str(main.get("cfg") or ""))
    if not main.get("cfg") or not cfg.is_file():
        raise RuntimeError(f"cold witness re-run: (d)'s job config {main.get('cfg')!r} is gone; "
                           "re-run (d) in full")
    w = run_cold_witness(ctx, cfg, rate, spent=0.0)
    out: dict[str, Any] = {k: prev[k] for k in ("episodes", "derived_content") if k in prev}
    out.update({"ok": w.get("state") == "done", "jobs": {JOB: main}, "witness_jobs": {JOB: w},
                "tracked_usd": float(w.get("spend") or 0.0),
                "report_error": bool(w.get("report_error")),
                "apps": [w["app"]] if "app" in w else [], "witness_only": True})
    if not out["ok"]:
        out["error"] = (f"cold witness job ended {w.get('state')!r}"
                        + (f" ({w['error']})" if w.get("error") else ""))
    return out


def witness_pending(st: State) -> bool:
    """(d)'s deployed job is done but its cold witness is missing or not done, on a state that
    needs one (the certified plan, or a result that ran one): only the witness re-runs."""
    res = (st.steps.get("balanced") or StepState()).result
    if not balanced_job_done(st) or (not is_cert(st.plan) and "witness_jobs" not in res):
        return False
    return ((res.get("witness_jobs") or {}).get(JOB) or {}).get("state") != "done"


def iso_witness_pending(st: State) -> bool:
    """(g)'s isolated caption job (J2) is done but its cold witness (J2w) is not: only J2w
    re-runs."""
    res = (st.steps.get("caption_iso") or StepState()).result
    job = (res.get("jobs") or {}).get(JOB) or {}
    if job.get("state") != "done" or not job.get("job_id"):
        return False
    return ((res.get("witness_jobs") or {}).get(JOB) or {}).get("state") != "done"


def balanced_job_done(st: State) -> bool:
    """(d)'s deployed job finished (``done``), whatever its step's status says: a state from
    before the lab became its own step marks (d) ``failed`` when only the lab failed."""
    job = ((st.steps.get("balanced") or StepState()).result.get("jobs") or {}).get(JOB) or {}
    return job.get("state") == "done" and bool(job.get("job_id"))


def adopt_balanced_job(st: State) -> bool:
    """Mark (d) ``done`` when its deployed job finished but the step failed on the lab it
    used to run inline (the 2026-10-22 state). Never while its cold witness is pending
    (:func:`witness_pending`): (d) is done only when both are. Returns whether it changed
    anything."""
    s = st.steps.get("balanced")
    if s is None or s.status == "done" or not balanced_job_done(st) or witness_pending(st):
        return False
    s.status = "done"
    s.notes.append("deployed job done; its lab part moved to step lab (d2), re-run there")
    return True


def records_source_path(jsonl: Path | str) -> Path:
    """The sidecar ``wave_h_finalize.py --fetch`` writes beside an exported record JSONL: which
    job (and store) the records were exported from."""
    return Path(f"{jsonl}.source.json")


def segment_binding(ctx: Ctx, job: Mapping[str, Any]) -> dict[str, str]:
    """``{"job_id", "segment_hash"}`` of (d)'s deployed job: the segment stage hash its segment
    records carry as ``provenance.config_hash``. From the state (``_run_job`` records the plan's
    ``stage_hashes``), else from the job's run spec in the store (a Volume read; no container).
    Raises when neither has it: the lab never captions segments it cannot bind to (d)."""
    sh = (job.get("stage_hashes") or {}).get("segment")
    if not sh:
        from flora_deploy import sdk

        j = sdk.open_job(job["job_id"], store=store_uri(ctx.st))
        sh = (j.settings.stage_hashes or {}).get("segment")
    if not sh:
        raise RuntimeError(f"(d)'s job {job['job_id']} has no segment stage hash")
    return {"job_id": str(job["job_id"]), "segment_hash": str(sh)}


def _segment_records(ctx: Ctx, job: Mapping[str, Any]) -> list[Any]:
    """(d)'s segment records: the exported JSONL (``wave_h_finalize.py --fetch``) when its
    sidecar (:func:`records_source_path`) names (d)'s job id, else read from the job store (a
    Volume read; no container). The JSONL's fixed path names no job, so without the sidecar
    match it may hold another job's records (review 12 F5)."""
    from flora_deploy.records import StageRecord

    path = Path((job.get("records") or {}).get("segment")
                or record_paths("balanced", root=ctx.spec.records)["segment"])
    try:
        src = json.loads(records_source_path(path).read_text())
    except (OSError, ValueError):
        src = {}
    if path.is_file() and isinstance(src, dict) and src.get("job_id") == job["job_id"]:
        return [StageRecord.from_json(line) for line in path.read_text().splitlines()
                if line.strip()]
    from flora_deploy import sdk

    return list(sdk.open_job(job["job_id"], store=store_uri(ctx.st)).results("segment"))


def segment_provenance_problems(records: Iterable[Any], episodes: Sequence[str],
                                binding: Mapping[str, str],
                                local_id: Callable[[str], str] = h0.episode_id) -> list[str]:
    """Why (d)'s segment records for ``episodes`` are not (d)'s: any segment record of those
    episodes whose ``provenance.config_hash`` is not the job's segment stage hash (empty: all
    of them are). ``local_id``: the plan's episode id of a golden id."""
    want = binding["segment_hash"]
    eids = {local_id(g) for g in episodes}
    bad = sorted({str(getattr(getattr(r, "provenance", None), "config_hash", None))
                  for r in records if r.episode_id in eids and r.stage == "segment"
                  and getattr(getattr(r, "provenance", None), "config_hash", None) != want})
    if bad:
        return [f"segment records are not from (d)'s job {binding['job_id']}: config_hash "
                f"{[b[:12] for b in bad]} != its segment stage hash {want[:12]}"]
    return []


def deployed_segments(records: Iterable[Any], episodes: Sequence[str],
                      local_id: Callable[[str], str] = h0.episode_id) -> dict[str, list[dict]]:
    """Golden id -> NVIDIA segment rows (``stages.segment.rows_from_record``, what the deployed
    caption worker feeds ``caption_*``) from (d)'s segment records, for ``episodes``
    (``local_id``: the plan's episode id of a golden id)."""
    from flora_deploy.stages.segment import rows_from_record

    by_eid = {local_id(g): g for g in episodes}
    out: dict[str, list[dict]] = {}
    for rec in records:
        g = by_eid.get(rec.episode_id)
        if g is not None and rec.stage == "segment":
            out[g] = rows_from_record(rec)
    return out


def golden_missing(paths: Iterable[str]) -> list[str]:
    """The ``flora-golden`` paths that do not exist (directory listings; no container)."""
    import modal

    vol = modal.Volume.from_name(h0.GOLDEN_VOLUME, version=2)
    listed: dict[str, set[str]] = {}
    out = []
    for p in paths:
        parent = p.rsplit("/", 1)[0]
        if parent not in listed:
            try:
                listed[parent] = {e.path.rstrip("/").rsplit("/", 1)[-1]
                                  for e in vol.listdir(parent)}
            except Exception:  # noqa: BLE001 - a missing directory: everything under it
                listed[parent] = set()
        if p.rstrip("/").rsplit("/", 1)[-1] not in listed[parent]:
            out.append(p)
    return out


def lab_preconditions(manifest: Mapping[str, Any], episodes: Sequence[str],
                      segments: Mapping[str, Sequence[Mapping[str, Any]]], *,
                      missing: Callable[[Iterable[str]], list[str]] = golden_missing,
                      expected: Sequence[str] = EPISODES) -> list[str]:
    """What must hold before a lab container starts (empty: all of it). The episodes are (d)'s
    and staged; every camera file the lab child will open, at the path it will open it
    (``balanced_lab.episode_inputs``), exists on ``flora-golden``; (d)'s segments cover every
    episode (the Captioner lab captions them)."""
    from flora_deploy.modal_app import balanced_lab as lab

    out = []
    if not episodes:
        return ["no episodes: (d) recorded none"]
    if tuple(episodes) != tuple(expected):
        out.append(f"episodes differ from (d)'s {len(expected)}: {list(episodes)}")
    try:
        inputs = lab.episode_inputs(dict(manifest), list(episodes))
        paths = lab.volume_paths(inputs)
    except (KeyError, ValueError) as e:
        return [*out, f"episode inputs: {e}"]
    if not paths:
        out.append("the episodes name no camera files")
    if miss := missing(paths):
        out.append(f"{len(miss)} of {len(paths)} camera files not on {h0.GOLDEN_VOLUME}: "
                   f"{miss[:3]}")
    if empty := [g for g in episodes if not segments.get(g)]:
        out.append(f"(d)'s segment records cover {len(episodes) - len(empty)} of "
                   f"{len(episodes)} episodes (missing: {empty})")
    return out


NO_EVIDENCE = "produced no L0 / Gate A evidence"


def lab_result_path(kind: str) -> Path:
    return LAB_OUT / f"balanced-{kind}.json"


def reusable_lab_result(kind: str, run_id: str, episodes: Sequence[str],
                        path: Path | None = None, *,
                        segments_from: Mapping[str, str] | None = None
                        ) -> dict[str, Any] | None:
    """A saved lab result for ``kind`` a re-run can keep instead of paying for it again: same
    run id and kind, (d)'s episodes, no error anywhere, L0 rows and Gate A requests, measured
    under the current engine settings (:func:`lab_hash_problem`). A Captioner result must also
    have captioned ``segments_from`` ((d)'s job id and segment stage hash,
    :func:`segment_binding`); without a known binding it is never reused."""
    p = path or lab_result_path(kind)
    try:
        res = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(res, dict):
        return None
    base = res.get("baseline") or {}
    gate = res.get("gate_a") or {}
    ok = (res.get("run") == f"wave-h-{run_id}" and res.get("kind") == kind
          and not res.get("error") and not base.get("error") and not res.get("no_evidence")
          and bool(base.get("l0")) and int(gate.get("requests") or 0) > 0
          and list(base.get("episodes_requested") or ()) == list(episodes)
          and not (res.get("dtbias") or {}).get("error")
          and lab_hash_problem(kind, res) is None)
    if kind == "captioner":
        ok = ok and segments_from is not None and res.get("segments_from") == dict(segments_from)
    return res if ok else None


def baseline_call_path(kind: str, lab_dir: Path | None = None) -> Path:
    """Where a split lab's baseline call result is kept (beside ``balanced-<kind>.json``), so
    a retry whose plugin call failed pays for the plugin call alone."""
    return (lab_dir or LAB_OUT) / f"balanced-{kind}-baseline.json"


def reusable_baseline_call(kind: str, run_id: str, episodes: Sequence[str] | None = None,
                           path: Path | None = None, *,
                           segments_from: Mapping[str, str] | None = None,
                           check_binding: bool = True) -> dict[str, Any] | None:
    """A saved baseline call (``arm: baseline``) a retry can keep: same run and kind, no error,
    L0 rows, the Gate A inputs it staged (``gate_rel``, ``gate_sha256``), measured under the
    current baseline engine settings; with ``check_binding`` also on ``episodes`` and, for the
    Captioner, the same ``segments_from`` (as :func:`reusable_lab_result`)."""
    p = path or baseline_call_path(kind)
    try:
        res = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(res, dict):
        return None
    base = res.get("baseline") or {}
    ok = (res.get("arm") == "baseline" and res.get("run") == f"wave-h-{run_id}"
          and res.get("kind") == kind and not res.get("error") and not base.get("error")
          and not res.get("no_evidence") and bool(base.get("l0"))
          and bool(res.get("gate_rel")) and bool(res.get("gate_sha256"))
          and (res.get("engine_hashes") or {}).get("baseline")
          == lab_engine_hashes(kind)["baseline"])
    if check_binding:
        ok = ok and list(base.get("episodes_requested") or ()) == list(episodes or ())
        if kind == "captioner":
            ok = ok and segments_from is not None and \
                res.get("segments_from") == dict(segments_from)
    return res if ok else None


def saved_lab_calls(run_id: str, spec: PlanSpec) -> tuple[str, ...]:
    """The split lab calls a dry run prices at $0 (``"<kind>:baseline"``): a saved baseline
    call of this run under the current engine settings, in the plan's lab dir (its episode
    and segment binding is checked when the lab runs: :func:`reusable_baseline_call`)."""
    return tuple(f"{k}:baseline" for k in LAB_CALLS if "baseline" in lab_calls(k, spec)
                 and reusable_baseline_call(k, run_id, path=baseline_call_path(k, spec.labs),
                                            check_binding=False) is not None)


def merge_lab_calls(base: Mapping[str, Any], plug: Mapping[str, Any]) -> dict[str, Any]:
    """One ``balanced-<kind>.json`` from a split lab's baseline and plugin calls, in the shape
    of a one-call result (``baseline``, ``dtbias``, ``gate_a``, ``engine_hashes``,
    ``compile_cache``, ``container_s``, ``usd_estimate``), with each call's own figures under
    ``calls``. The plugin call must have read the requests the baseline call staged
    (``gate_sha256``) for the baseline engine it ran (its engine hash); else an error."""
    out = {k: v for k, v in base.items() if k not in ("arm", "error", "no_evidence")}
    out["engine_hashes"] = {**(base.get("engine_hashes") or {}),
                            **{k: v for k, v in (plug.get("engine_hashes") or {}).items()
                               if k == "plugin"}}
    out["compile_cache"] = {**(base.get("compile_cache") or {}),
                            **{k: v for k, v in (plug.get("compile_cache") or {}).items()
                               if k == "plugin"}}
    if plug.get("dtbias") is not None:
        out["dtbias"] = plug["dtbias"]
    if plug.get("gate_a") is not None:
        out["gate_a"] = plug["gate_a"]
    out["calls"] = {arm: {k: r.get(k) for k in ("container_s", "usd_estimate", "prefetch",
                                                  "compile_cache", "error")}
                    for arm, r in (("baseline", base), ("plugin", plug))}
    out["split"] = True
    out["container_s"] = round(sum(float(r.get("container_s") or 0.0) for r in (base, plug)), 1)
    out["usd_estimate"] = round(sum(float(r.get("usd_estimate") or 0.0) for r in (base, plug)),
                                3)
    err = plug.get("error")
    if not err and plug.get("gate_sha256") != base.get("gate_sha256"):
        err = "the Gate A requests it read are not the baseline call's (gate_sha256)"
    elif not err and plug.get("baseline_engine_hash") != (base.get("engine_hashes")
                                                          or {}).get("baseline"):
        err = "the staged Gate A baseline is another baseline engine's"
    if err:
        out["error"] = err if str(err).startswith("plugin:") else f"plugin: {err}"
        out.pop("gate_a", None)
    return out


def lab_reuse(st: State, spec: PlanSpec | None = None) -> tuple[str, ...]:
    """The lab kinds whose saved result (d2) reuses (:func:`reusable_lab_result`), on the
    plan's lab episodes (:attr:`PlanSpec.lab_episodes`) of (d)'s."""
    spec = spec or SPECS[DEFAULT_PLAN]
    bal = (st.steps.get("balanced") or StepState()).result
    eps = [e for e in bal.get("episodes") or () if e in spec.lab_episodes]
    if not eps:
        return ()
    job = (bal.get("jobs") or {}).get(JOB) or {}
    sh = (job.get("stage_hashes") or {}).get("segment")
    sf = {"job_id": str(job["job_id"]), "segment_hash": sh} if sh and job.get("job_id") else None
    return tuple(k for k in LAB
                 if reusable_lab_result(k, st.run_id, eps, segments_from=sf) is not None)


@contextlib.contextmanager
def _lab_app(ctx: Ctx, lab: Any, modal: Any, name: str):
    """``lab.app.run(name=name)`` (Modal's output on): the attempt's own app name, which
    billing reports as its description, noted before it can spend, and its app id once it
    runs. The app stops when the block exits, which is recorded as a confirmed stop."""
    ctx.note_app(name)
    raised: BaseException | None = None
    with modal.enable_output(), lab.app.run(name=name):
        ctx.note_app_id(getattr(lab.app, "app_id", None))
        try:
            yield
        except BaseException as e:  # noqa: BLE001 - re-raised once the app has stopped
            raised = e
    ctx.confirm_stop()  # reached only when app.run()'s exit (the app's stop) did not raise
    if raised is not None:
        raise raised


def _lab_kind_summary(res: Mapping[str, Any], wall: float, usd: float) -> dict[str, Any]:
    base = res.get("baseline", {}) if isinstance(res, Mapping) else {}
    err = res.get("error") or base.get("error")
    l0, gate = base.get("l0") or [], res.get("gate_a")
    evidence = bool(l0) and bool(gate and gate.get("requests"))
    return {"wall_s": round(wall, 1), "usd": round(usd, 4),
            "lab_usd_estimate": res.get("usd_estimate"), "throughput": base.get("throughput"),
            "l0": l0, "gate_a": gate, "evidence": evidence,
            "episode_errors": len(base.get("episode_errors") or {}),
            "compile_cache": res.get("compile_cache"),
            "engine_hashes": res.get("engine_hashes"),
            "error": err or (None if evidence else NO_EVIDENCE)}


def run_lab(ctx: Ctx, *, missing: Callable[[Iterable[str]], list[str]] | None = None,
            records: Callable[[Ctx, Mapping[str, Any]], list[Any]] | None = None,
            binding: Callable[[Ctx, Mapping[str, Any]], dict[str, str]] | None = None,
            lab_module: Any = None, source: State | None = None,
            kinds: Sequence[str] = tuple(LAB), lab_dir: Path | None = None) -> dict[str, Any]:
    """(d2): the lab's L0 + Gate A against (d)'s finished deployed job: (d)'s episodes, and
    (d)'s segment records for the Captioner (``--only lab`` retries it alone). Every
    precondition is checked before any container (:func:`lab_preconditions`); a failed one
    charges $0. TempSeg runs first; when it gets no request in, the Captioner is not started
    (it reads the same inputs). A kind runs as its :data:`LAB_CALLS` (the Captioner: a
    baseline call, then a plugin call on the Gate A inputs it staged, merged by
    :func:`merge_lab_calls`; a saved baseline call is reused, :func:`reusable_baseline_call`).
    Each call's deadline is :data:`LAB_CALL_DEADLINE_S` (its cold compiles), shortened by the
    budget left; a call the budget cannot carry to its expected length is not started. Spend
    is each started call's wall time x list $/s. ``source``: the
    state whose (d) the lab reads (default this run's; recertify: (b)'s certified state,
    :func:`run_recert_lab`); ``kinds``: the lab kinds to run; ``lab_dir``: where the lab JSON
    is written (default :data:`LAB_OUT`)."""
    if ctx.spec.name == CERTIFIED:
        done, why_done = lab_already_done()
        if done:  # the certified run's "lab already done" skip: nothing starts, $0
            return {"ok": True, "skipped": f"lab already done: {why_done}", "kinds": {},
                    "tracked_usd": 0.0, "apps": []}
    if lab_module is None:
        from flora_deploy.modal_app import balanced_lab as lab_module
    lab = lab_module
    src = ctx.st if source is None else source
    sctx = ctx if source is None else replace(ctx, st=source)  # its store, its (d)
    kinds = tuple(k for k in LAB if k in kinds)
    bal = src.step("balanced").result
    if not balanced_job_done(src):
        raise RuntimeError("the lab runs against (d)'s finished deployed job; (d) has none "
                           "(run --only balanced first)")
    if witness_pending(src):
        raise RuntimeError("(d)'s cold witness is not done; the lab waits for it (run --only "
                           "balanced: it re-runs the witness alone)")
    job = bal["jobs"][JOB]
    episodes = [e for e in bal.get("episodes") or () if e in ctx.spec.lab_episodes]
    bound: dict[str, str] | None = None
    try:
        bound = (binding or segment_binding)(sctx, job)
        recs = list((records or _segment_records)(sctx, job))
        why = segment_provenance_problems(recs, episodes, bound, ctx.spec.local_id)
        segments = {} if why else deployed_segments(recs, episodes, ctx.spec.local_id)
    except Exception as e:  # noqa: BLE001 - before any container: a precondition
        segments = {}
        why = [f"reading (d)'s segment records: {e!r}"]
    why += lab_preconditions(ctx.manifest, episodes, segments,
                             missing=missing or golden_missing,
                             expected=ctx.spec.lab_episodes)
    out: dict[str, Any] = {"episodes": episodes, "job_id": job["job_id"], "kinds": {},
                           "segments_from": bound,
                           "segments": {"source": "deployed job records",
                                        "episodes": len(segments),
                                        "rows": sum(len(v) for v in segments.values())}}
    if why:
        out["error"] = "preconditions: " + "; ".join(why)
        out["note"] = f"no lab container started; charged $0; {NO_EVIDENCE}"
        return {"ok": False, **out, "tracked_usd": 0.0, "apps": []}
    spent = 0.0
    name = ctx.app_name()

    def persist() -> None:  # each kind's summary lands in the state as soon as it returns
        ctx.st.step("lab").result = {"ok": False, "partial": True, **out, "tracked_usd": spent,
                                     "apps": [name] if todo else []}
        save_state(ctx.st, ctx.state_path)

    rdir = lab_dir or LAB_OUT
    for kind in kinds:
        saved = reusable_lab_result(kind, ctx.st.run_id, episodes,
                                    path=rdir / f"balanced-{kind}.json", segments_from=bound)
        if saved is not None:
            out["kinds"][kind] = {**_lab_kind_summary(saved, 0.0, 0.0), "reused": True,
                                  "requests": ((saved.get("baseline") or {}).get("batcher")
                                               or {}).get("requests")}
    todo = [k for k in kinds if k not in out["kinds"]]
    if not todo:
        out["note"] = "every kind reused a saved result; nothing started"
        return {"ok": True, **out, "tracked_usd": 0.0, "apps": []}
    persist()
    import modal

    rdir.mkdir(parents=True, exist_ok=True)
    fns = {"tempseg": lab.run_tempseg, "captioner": lab.run_captioner}
    saved_base = {k: reusable_baseline_call(k, ctx.st.run_id, episodes,
                                            path=baseline_call_path(k, rdir),
                                            segments_from=bound)
                  for k in todo if "baseline" in lab_calls(k, ctx.spec)}
    arms = {k: tuple(a for a in lab_calls(k, ctx.spec)
                     if not (a == "baseline" and saved_base.get(k))) for k in todo}
    first = (todo[0], arms[todo[0]][0])  # a reused TempSeg result always had requests in
    limit = min(LAB_CALL_DEADLINE_S[first],
                ctx.deadline_s("lab", LAB[first[0]], slack_s=LAB_SLACK_S))
    if limit < LAB_CALL_EXPECTED_S[first]:  # the stop rule, before the app starts: charged $0
        b = ctx.budget_of("lab")
        raise StopRule(f"lab: budget left ${b.usd:.2f} allows {limit:.0f} s of the {first[0]} "
                       f"{first[1]} call, under the {LAB_CALL_EXPECTED_S[first]:.0f} s it "
                       f"needs (binding: {b.binding}); no lab app started")
    # the app is noted only now: a precondition failure above charged $0
    with _lab_app(ctx, lab, modal, name):
        for kind in todo:
            fn, c = fns[kind], LAB[kind]
            prev = out["kinds"].get("tempseg") or {}
            if kind == "captioner" and prev.get("requests") == 0:
                out["kinds"][kind] = {"skipped": "TempSeg sent zero requests on the same "
                                      "inputs; not started", "usd": 0.0, "evidence": False,
                                      "error": NO_EVIDENCE}
                continue
            cap = kind == "captioner"
            res: dict[str, Any] | None = saved_base.get(kind)
            calls: dict[str, dict[str, Any]] = (
                {"baseline": {"reused": str(baseline_call_path(kind, rdir)), "usd": 0.0}}
                if res is not None else {})
            wall = usd = 0.0
            for arm in arms[kind]:
                key = (kind, arm)
                limit = min(LAB_CALL_DEADLINE_S[key],
                            ctx.deadline_s("lab", c, spent_in_step=spent, slack_s=LAB_SLACK_S))
                if limit < LAB_CALL_EXPECTED_S[key]:
                    why = (f"budget left {limit:.0f} s < the {LAB_CALL_EXPECTED_S[key]:.0f} s "
                           "it needs")
                    calls[arm] = {"skipped": why, "usd": 0.0}
                    res = None if res is None else {**res, "error": f"{arm} call not started: "
                                                    f"{why}"}
                    break
                args: tuple[Any, ...] = (f"wave-h-{ctx.st.run_id}", "", episodes, True,
                                         segments if cap else None, bound if cap else None)
                if arm != "both":
                    args += (arm, str((res or {}).get("gate_rel") or "") if arm == "plugin"
                             else "")
                t0 = time.time()
                call = fn.spawn(*args)
                try:
                    got = call.get(timeout=limit)
                except (TimeoutError, modal.exception.TimeoutError):
                    call.cancel()
                    got = {"error": f"client deadline {limit:.0f} s"}
                except Exception as e:  # noqa: BLE001 - the remote call raised: it failed
                    got = {"error": f"lab call raised {e!r}"[:2000]}
                w = time.time() - t0
                u = c.usd_per_s() * w
                spent, wall, usd = spent + u, wall + w, usd + u
                calls[arm] = {"wall_s": round(w, 1), "usd": round(u, 4),
                              "deadline_s": round(limit), "error": got.get("error")}
                if arm == "plugin":
                    res = merge_lab_calls(res or {}, got)
                else:
                    res = got
                if arm == "baseline":
                    baseline_call_path(kind, rdir).write_text(json.dumps(got, default=str))
                    if got.get("error") or not got.get("gate_rel"):
                        if not got.get("error"):
                            res = {**got, "error": "baseline call staged no Gate A inputs"}
                        break  # no plugin call without the baseline's Gate A inputs
                if arm != arms[kind][-1]:  # a split kind's progress lands in the state too
                    out["kinds"][kind] = {**_lab_kind_summary(res, wall, usd), "calls": calls,
                                          "partial": True}
                    persist()
            if res is None:  # its first call was not started
                out["kinds"][kind] = {"skipped": calls[arms[kind][0]]["skipped"], "usd": 0.0,
                                      "evidence": False, "error": NO_EVIDENCE, "calls": calls}
                continue
            (rdir / f"balanced-{kind}.json").write_text(json.dumps(res, default=str))
            out["kinds"][kind] = _lab_kind_summary(res, wall, usd)
            if len(lab_calls(kind, ctx.spec)) > 1:
                out["kinds"][kind]["calls"] = calls
            stats = (res.get("baseline") or {}).get("batcher") or {}
            out["kinds"][kind]["requests"] = stats.get("requests")
            persist()
    bad = {k: v["error"] for k, v in out["kinds"].items() if v.get("error")}
    bad.update({k: f"{v['episode_errors']} episode(s) failed" for k, v in out["kinds"].items()
                if v.get("episode_errors") and k not in bad})
    if bad:
        out["error"] = "; ".join(f"{k}: {v}" for k, v in bad.items())
    return {"ok": not bad and len(out["kinds"]) == len(kinds), **out, "tracked_usd": spent,
            "apps": [name]}


def pre_v4_state(st: State, spec: PlanSpec) -> str | None:
    """Why ``st`` is a certified state from before generator v4 (None: it is not), so the v4
    certified plan never runs its new steps (``caption_iso``) on it. Two markers, from what such
    a state holds: (1) its certify step (e) has attempts but (g) ``caption_iso`` never ran (under
    v4, e needs g first: :data:`CERT_DEPS`); (2) a certificate e recorded (its
    ``stages.<stage>.id``, written under ``<work>/certify/``) whose generator version is not
    ``"4"``. A fresh or partly run v4 state has neither."""
    from flora_deploy.validate.certkey import GENERATOR_VERSIONS

    cert = st.steps.get("certify") or StepState()
    iso = st.steps.get("caption_iso") or StepState()
    if cert.attempts and not iso.attempts:
        return "its certify step ran with no caption_iso step"
    for stage, v in sorted(((cert.result or {}).get("stages") or {}).items()):
        cid = (v or {}).get("id")
        p = spec.work / "certify" / f"{cid}.json"
        try:
            doc = json.loads(p.read_text()) if cid else None
        except (OSError, ValueError):
            doc = None
        ver = ((doc or {}).get("generator") or {}).get("version")
        if doc is not None and ver not in GENERATOR_VERSIONS:
            return f"its {stage} certificate {cid} is generator version {ver!r}"
    return None


def require_code_landed(step: str) -> None:
    """recertify's d2', d' and g run the v4 code (the stage-isolated caption certificate):
    refused, before any app, on a checkout without it (``"4" in
    certkey.GENERATOR_VERSIONS``)."""
    from flora_deploy.validate.certkey import GENERATOR_VERSIONS

    if "4" not in GENERATOR_VERSIONS:
        raise RuntimeError(f"{step}: needs the v4 certification code (certkey."
                           f"GENERATOR_VERSIONS {GENERATOR_VERSIONS} has no '4'); land it first "
                           "(the noise run f2 runs before it, on the unchanged checkout)")


def source_state(spec: PlanSpec) -> State:
    """The certified state recertify reads (b)'s oracle and the old noise run from (read-only;
    never saved). Raises when there is none."""
    if spec.source is None or not spec.source.is_file():
        raise RuntimeError(f"{spec.name}: no certified state at {spec.source} (run --plan "
                           "certified first: its (b) is the oracle)")
    return rb.load_state(spec.source)


def oracle_of(st: State) -> dict[str, Any]:
    """(b)'s finished oracle job (its state record), with the store it wrote to."""
    s = st.steps.get("exact_e2e") or StepState()
    job = dict((s.result.get("jobs") or {}).get(JOB) or {})
    if s.status != "done" or job.get("state") != "done" or not job.get("job_id"):
        raise RuntimeError("the oracle (b) has not finished (no done exact_e2e job)")
    sh = (job.get("stage_hashes") or {}).get("segcap")
    if not sh:
        raise RuntimeError(f"(b)'s job {job['job_id']} has no recorded segcap stage hash")
    job.setdefault("store", store_uri(st))
    return job


def run_recert_lab(ctx: Ctx, **kw: Any) -> dict[str, Any]:
    """recertify's d2': the Captioner lab only (L0 + Gate A) on the new engine settings, on
    (b)'s (d) job's segment records (L0 is per engine, not per episode). TempSeg's saved result
    (:data:`LAB_OUT`) is reused: it must still be evidence for the current TempSeg engine
    (:func:`lab_already_done`), and is copied beside the new Captioner one in the plan's lab
    dir. Skipped at $0 when both are there for the current engines."""
    lab_dir = ctx.spec.labs
    done, why = lab_already_done(lab_dir)
    if done:
        return {"ok": True, "skipped": f"lab already done: {why}", "kinds": {},
                "tracked_usd": 0.0, "apps": []}
    require_code_landed("lab")
    ok, why = lab_already_done(LAB_OUT, ("tempseg",))
    if not ok:
        raise RuntimeError(f"lab: TempSeg's saved lab result is not reusable ({why}); "
                           "recertify re-runs the Captioner lab only")
    lab_dir.mkdir(parents=True, exist_ok=True)
    src, dst = LAB_OUT / "balanced-tempseg.json", lab_dir / "balanced-tempseg.json"
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    res = run_lab(ctx, source=source_state(ctx.spec), kinds=("captioner",), lab_dir=lab_dir,
                  **kw)
    res["tempseg_reused"] = str(dst)
    return res


def _job_rows(ctx: Ctx, step: str) -> dict[str, dict[str, list[dict]]]:
    """``{episode_id: {"segment": rows, "caption": rows}}`` from a finished job step."""
    from flora_deploy import sdk

    rows: dict[str, dict[str, list[dict]]] = {}
    for r in ctx.st.step(step).result.get("jobs", {}).values():
        if "job_id" not in r:
            continue
        job = sdk.open_job(r["job_id"], store=store_uri(ctx.st))
        for stage in ("segment", "caption"):
            for rec in job.results(stage):
                rows.setdefault(rec.episode_id, {})[stage] = [
                    {"start_s": s.start_s, "end_s": s.end_s, "caption": s.caption or ""}
                    for s in rec.segments]
    return rows


_exact_rows = _job_rows


def _lab_rows(kind: str) -> dict[str, list[dict]]:
    p = ROOT / "reports" / "wave-h" / f"balanced-{kind}.json"
    if not p.exists():
        return {}
    res = json.loads(p.read_text())
    return {h0.episode_id(e["id"]): e.get("rows", []) for e in
            res.get("baseline", {}).get("episodes", []) if e.get("ok")}


def run_certify(ctx: Ctx, noise: Mapping[str, float] | None = None) -> dict[str, Any]:
    exact = _job_rows(ctx, "exact_e2e")
    bal_rows = _job_rows(ctx, "balanced")  # the deployed balanced workers' records
    seg, cap = _lab_rows("tempseg"), _lab_rows("captioner")
    per = {}
    for eid, ex in exact.items():
        b = bal_rows.get(eid) or {}
        got = b.get("caption") or cap.get(eid) or b.get("segment") or seg.get(eid)
        if got is None:
            continue
        m = compare_segments(ex.get("caption") or ex.get("segment", []), got)
        if not (b.get("caption") or eid in cap):
            m["caption_jaccard"] = 0.0
        per[eid] = m
    bal = (ctx.st.step("lab").result.get("kinds")
           or ctx.st.step("balanced").result.get("kinds", {}))  # pre-(d2) states: inline lab
    l0 = [r for k in ("tempseg", "captioner") for r in (bal.get(k, {}).get("l0") or [])]
    gate = max((bal.get(k, {}).get("gate_a") or {} for k in ("tempseg", "captioner")),
               key=lambda g: float(g.get("max_abs_dlogit") or 0.0), default={})
    verdict = certify(aggregate(per), l0=l0, gate_a=gate or None, noise=noise)
    cert = {"schema": "flora_provisional_certificate_v0", "preset": "balanced",
            "pools": {"segment": "L40S", "caption": "H100"},
            "reference": f"exact ({ORACLE_DETERMINISM} mode)",
            "verdict": verdict, "per_episode": per, "run_id": ctx.st.run_id}
    (ROOT / "reports" / "wave-h" / "certificate-provisional.json").write_text(
        json.dumps(cert, indent=1, default=str))
    return {"ok": True, "verdict": verdict, "episodes": len(per), "tracked_usd": 0.0,
            "apps": []}


def stage_hashes(cfg: Path) -> dict[str, str]:
    """The stage hashes ``flora deploy`` plans ``cfg`` with (``factory.plan_app``; local)."""
    from flora_deploy.config.receipts import resolve
    from flora_deploy.config.schema import load_config
    from flora_deploy.modal_app.factory import plan_app

    c = load_config(cfg)
    return dict(plan_app(c, determinism=ORACLE_DETERMINISM,
                         derived_receipts=resolve(c)).stage_hashes)


def run_cert_noise(ctx: Ctx) -> dict[str, Any]:
    """The certified (f): (b)'s job repeated on the noise subset, **beside** (b). Its config is
    (b)'s (:func:`oracle_job`: the same inputs, data dir and staged data) with its own
    ``project`` (``wave-h-<run>-noise-<embodiment>-a<attempt>``), so its own job and app, as
    the cold witness has: the project is not hashed, so the stage hash is (b)'s, which is
    checked before any app (against (b)'s recorded hashes too, once (b) ran). Then ``flora
    index --limit`` on it indexes exactly the noise subset, and it runs in default mode."""
    require_tempseg_pin(ctx)
    o = oracle_job(ctx)
    emb = ctx.spec.embodiment
    n = max(1, len(ctx.st.step("noise").attempts))
    job_id = f"wave-h-{ctx.st.run_id}-noise-{emb}-a{n}"
    tag = f"noise-{emb}"
    stage_data(ctx, o["data"], o["ids"])
    text, _ = config_text(ctx, o["job"], embodiment=emb, stages=o["stages"],
                          budget=o["budget"], derived=o["derived"], data=o["data"])
    ocfg, cfg = ctx.work / f"flora-{tag}-oracle.yaml", ctx.work / f"flora-{tag}.yaml"
    ocfg.write_text(text)
    cfg.write_text(witness_config_text(text, job_id))
    want = stage_hashes(ocfg)
    ran = (((ctx.st.steps.get("exact_e2e") or StepState()).result.get("jobs") or {})
           .get(NOISE_JOB) or {}).get("stage_hashes")
    got = stage_hashes(cfg)
    if got != want or (ran and dict(ran) != got):  # before any app: charged $0
        raise RuntimeError(f"noise: stage hashes {got} are not (b)'s ({ran or want})")
    limit = ctx.spec.noise_limit
    _index(ctx, cfg, job_id, tag, limit=limit)
    if limit is not None:  # before any app: the index must hold exactly the noise subset
        why = noise_index_problems(ctx.work / f"summary-{tag}.json", ctx.manifest, ctx.spec)
        if why:
            raise RuntimeError(f"noise index: {why}")
    r = _run_job(ctx, "noise", cfg, job_id, determinism=ORACLE_DETERMINISM,
                 usd_per_s=pool_container("segcap").usd_per_s(), spent_in_step=0.0,
                 side="noise")
    res = _jobs_result({NOISE_JOB: r}, 1)
    res["episodes"] = len(ctx.spec.noise) if limit is not None else len(o["ids"])
    res["stage_hashes_of_oracle"] = True  # checked above
    return res


ORACLE_SEGMENTS_DIR = "oracle-segments"  # under the job store's sink (pipeline.caption_segments)


def oracle_segment_bytes(records: Iterable[Any], oracle_stage_hash: str,
                         uids: Iterable[str]) -> bytes:
    """The pinned oracle segment JSONL: the oracle's ``segment`` records of ``uids`` (exact
    preset, ``provenance.config_hash`` the oracle's segcap stage hash), one
    ``StageRecord.to_json`` line each, by uid. Raises when a uid has no such record, or has
    more than one."""
    want = set(uids)
    by: dict[str, list[Any]] = {}
    for r in records:
        prov = getattr(r, "provenance", None)
        if (r.stage == "segment" and r.uid in want and getattr(prov, "preset", None) == "exact"
                and getattr(prov, "config_hash", None) == oracle_stage_hash):
            by.setdefault(r.uid, []).append(r)
    miss = sorted(want - set(by))
    if miss:
        raise RuntimeError(f"oracle segments: no exact segment record of stage hash "
                           f"{oracle_stage_hash[:12]} for {len(miss)} episode(s)")
    dup = sorted(u for u, rs in by.items() if len({x.to_json() for x in rs}) > 1)
    if dup:
        raise RuntimeError(f"oracle segments: {len(dup)} episode(s) with differing records")
    return "".join(by[u][0].to_json() + "\n" for u in sorted(by)).encode()


def oracle_segment_records(ctx: Ctx, oracle: Mapping[str, Any]) -> list[Any]:
    """The oracle's segment records: the exported JSONL when its sidecar names the oracle job
    (``wave_h_finalize.py --fetch``), else read from the oracle's store (a Volume read)."""
    from flora_deploy.records import StageRecord

    path = Path(str((oracle.get("records") or {}).get("segment") or ""))
    try:
        src = json.loads(records_source_path(path).read_text())
    except (OSError, ValueError):
        src = {}
    if path.is_file() and isinstance(src, dict) and src.get("job_id") == oracle["job_id"]:
        return [StageRecord.from_json(line) for line in path.read_text().splitlines()
                if line.strip()]
    from flora_deploy import sdk

    return list(sdk.open_job(oracle["job_id"], store=oracle["store"]).results("segment"))


def stage_oracle_segments(ctx: Ctx, oracle: Mapping[str, Any], *,
                          records: Callable[[Ctx, Mapping[str, Any]], list[Any]] | None = None,
                          upload: Callable[[str, Path, str], Any] | None = None
                          ) -> dict[str, Any]:
    """Stage (b)'s segment records for the isolated caption jobs: written locally
    (``work/oracle-segments/<oracle stage hash>.jsonl``, byte-exact, its sha256 computed here)
    and uploaded into this run's job store at ``<run>/oracle-segments/<hash>.jsonl``, where
    the ``caption_segments.records`` URI (under ``output.sink``) points. Returns the
    ``caption_segments`` fields and the local path ``flora certify --oracle-segments`` reads."""
    import hashlib

    from flora_deploy.jobs.types import episode_uid

    sh = str(oracle["stage_hashes"]["segcap"])
    uids = [episode_uid(ctx.spec.data, ctx.spec.local_id(g)) for g in ctx.spec.episodes]
    data = oracle_segment_bytes((records or oracle_segment_records)(ctx, oracle), sh, uids)
    d = ctx.work / ORACLE_SEGMENTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    local = d / f"{sh}.jsonl"
    if local.exists() and local.read_bytes() != data:
        raise RuntimeError(f"oracle segments: {local} exists with other bytes")
    local.write_bytes(data)
    (upload or h0.upload_dir)(STORE_VOLUME, d, f"{ctx.st.run_id}/{ORACLE_SEGMENTS_DIR}")
    return {"records": f"volume://{STORE_VOLUME}/{ctx.st.run_id}/{ORACLE_SEGMENTS_DIR}/"
                       f"{sh}.jsonl",
            "sha256": hashlib.sha256(data).hexdigest(), "oracle_stage_hash": sh,
            "path": str(local), "episodes": len(uids), "oracle_job": oracle["job_id"]}


def run_caption_iso(ctx: Ctx) -> dict[str, Any]:
    """(g): the stage-isolated caption evidence. J2: a ``preset: balanced``, ``stages:
    [caption]`` job on every episode whose caption shards read the oracle's own segment
    records (``pipeline.caption_segments``: :func:`stage_oracle_segments`), so every pair has
    one Captioner input; then J2w: the same config (its ``project`` renamed) on the witness
    episodes with ``FLORA_COMPILE_CACHE=off``. A re-run with J2 done re-runs J2w alone
    (:func:`iso_witness_pending`). Needs the v4 code (:func:`require_code_landed`)."""
    require_code_landed("caption_iso")
    require_tempseg_pin(ctx)
    src = source_state(ctx.spec) if ctx.spec.source is not None else ctx.st
    oracle = oracle_of(src)
    rate = pool_container("caption", "balanced").usd_per_s()
    emb, pre = ctx.spec.embodiment, ctx.spec.job_prefix
    prev = ctx.st.step("caption_iso").result
    out: dict[str, Any] = {}
    if iso_witness_pending(ctx.st):
        main = dict(prev["jobs"][JOB])
        cfg = Path(str(main.get("cfg") or ""))
        if not main.get("cfg") or not cfg.is_file():
            raise RuntimeError(f"isolated witness re-run: J2's config {main.get('cfg')!r} is "
                               "gone; re-run (g) in full")
        segs = dict(prev["oracle_segments"])
        jobs, spent = {JOB: main}, 0.0
        out["witness_only"] = True
    else:
        episodes = balanced_episodes(ctx.manifest, ctx.spec, side="(g)")
        segs = stage_oracle_segments(ctx, oracle)
        derived = resolve_receipts(["captioner_bf16"], require_pin=True)
        stage_data(ctx, ctx.spec.data, episodes)
        text, job_id = config_text(ctx, f"{pre}iso-{emb}", embodiment=emb, stages=("caption",),
                                   budget=ctx.caps["caption_iso"], preset="balanced",
                                   derived=derived, data=ctx.spec.data)
        cfg = ctx.work / f"flora-{pre}iso-{emb}.yaml"
        cfg.write_text(with_caption_segments(text, records=segs["records"],
                                             sha256=segs["sha256"],
                                             oracle_stage_hash=segs["oracle_stage_hash"]))
        _index(ctx, cfg, job_id, f"{pre}iso-{emb}")
        r = _run_job(ctx, "caption_iso", cfg, job_id, determinism=ORACLE_DETERMINISM,
                     usd_per_s=rate, spent_in_step=0.0, side="isolated")
        jobs = {JOB: {**r, "stages": ["caption"], "oracle_segments": segs}}
        spent = float(r.get("spend") or 0.0)
        if r.get("state") != "done":
            res = _jobs_result(jobs, 1)
            res.update(oracle_segments=segs, error=f"J2 ended {r.get('state')!r}"
                       + (f" ({r['error']})" if r.get("error") else ""))
            return res
    n = max(1, len(ctx.st.step("caption_iso").attempts))
    wjob = f"wave-h-{ctx.st.run_id}-{pre}isocold-{emb}-a{n}"
    wtag = f"{pre}isocold-{emb}"
    wcfg = ctx.work / f"flora-{wtag}.yaml"
    wcfg.write_text(witness_config_text(cfg.read_text(), wjob))
    _index(ctx, wcfg, wjob, wtag, limit=ctx.spec.witness_limit)
    got = noise_index_problems(ctx.work / f"summary-{wtag}.json", ctx.manifest, ctx.spec,
                               ctx.spec.witness)
    if got:
        w: dict[str, Any] = {"job_id": wjob, "state": "skipped", "spend": 0.0,
                             "error": f"witness index: {got}", "compile_cache": "off"}
    else:
        wctx = replace(ctx, env={**ctx.env, COMPILE_CACHE_ENV: "off"})
        w = _run_job(wctx, "caption_iso", wcfg, wjob, determinism=ORACLE_DETERMINISM,
                     usd_per_s=rate, spent_in_step=spent, side="isolated_witness")
        w.update(compile_cache="off", episodes=len(ctx.spec.witness), stages=["caption"])
    res = _jobs_result(jobs, 1)
    res.update(witness_jobs={JOB: w}, oracle_segments=segs,
               tracked_usd=(0.0 if out.get("witness_only") else spent)
               + float(w.get("spend") or 0.0),
               apps=sorted({*(res["apps"] if not out.get("witness_only") else []),
                            *([w["app"]] if "app" in w else [])}), **out)
    if w.get("state") != "done":
        res["ok"] = False
        res["error"] = (f"isolated cold witness ended {w.get('state')!r}"
                        + (f" ({w['error']})" if w.get("error") else ""))
    return res


def run_noise(ctx: Ctx) -> dict[str, Any]:
    """(f): (b)'s job repeated: its own config file and staged data (same stage hash, so the
    same app), a new job id, default mode; all of its episodes (module docstring). The
    certified plan's runs beside (b) instead (:func:`run_cert_noise`)."""
    if ctx.spec.name == CERTIFIED:
        return run_cert_noise(ctx)
    oracle = ctx.st.step("exact_e2e").result.get("jobs", {}).get(NOISE_JOB) or {}
    if not oracle.get("cfg"):
        raise RuntimeError(f"noise repeats (b)'s {NOISE_JOB} job, which has not run")
    require_tempseg_pin(ctx)
    cfg = Path(oracle["cfg"])
    job_id = f"wave-h-{ctx.st.run_id}-noise-{ctx.spec.embodiment}"
    tag = f"noise-{ctx.spec.embodiment}"
    limit = ctx.spec.noise_limit
    _index(ctx, cfg, job_id, tag, limit=limit)
    if limit is not None:  # before any app: the index must hold exactly the noise subset
        got = noise_index_problems(ctx.work / f"summary-{tag}.json", ctx.manifest, ctx.spec)
        if got:
            raise RuntimeError(f"noise index: {got}")
    r = _run_job(ctx, "noise", cfg, job_id, determinism=ORACLE_DETERMINISM,
                 usd_per_s=pool_container("segcap").usd_per_s(), spent_in_step=0.0,
                 side="noise")
    res = _jobs_result({NOISE_JOB: r}, 1)
    res["episodes"] = len(ctx.spec.noise) if limit is not None else len(ctx.episodes)
    res["same_app_as_oracle"] = r.get("app") == oracle.get("app")
    ctx.st.step("noise").result = res  # so _job_rows can read this step's job
    exact = _job_rows(ctx, "exact_e2e")
    noise_rows = _job_rows(ctx, "noise")
    per = {eid: compare_segments(exact[eid].get("caption", []), rows.get("caption", []))
           for eid, rows in noise_rows.items() if eid in exact}
    floor = aggregate(per)
    res["noise_floor"] = floor
    if ctx.st.step("certify").status == "done":
        res["certify_with_noise"] = run_certify(ctx, noise=floor)["verdict"]
    return res


def noise_index_problems(summary: Path, manifest: Mapping[str, Any], spec: PlanSpec,
                         episodes: Sequence[str] | None = None) -> str:
    """Empty when the noise job's index (``flora index --limit``) holds exactly the plan's
    noise subset (``episodes``: another set, e.g. the cold witnesses'): its episode count and
    its video seconds (within 1%) equal theirs."""
    eps = spec.noise if episodes is None else tuple(episodes)
    try:
        d = json.loads(summary.read_text())
    except (OSError, ValueError) as e:
        return f"unreadable {summary.name}: {e}"
    n = (d.get("counts") or {}).get("episodes")
    dur = durations(manifest)
    want = sum(dur.get(g, 0.0) for g in eps)
    got = float(d.get("video_s") or 0.0)
    if n != len(eps) or abs(got - want) > 0.01 * max(want, 1.0):
        return (f"{n} episodes / {got:.1f} video s indexed, not the {len(eps)} noise "
                f"episodes / {want:.1f} s")
    return ""


def planned_hashes(text: str, base_dir: Path) -> tuple[str, dict[str, str]]:
    """``(config hash, stage hashes)`` ``flora deploy`` plans config ``text`` with, as if it
    were a file in ``base_dir`` (its relative ``input.path`` resolves there; local, nothing
    written: ``factory.plan_app``)."""
    from flora_deploy.config.receipts import resolve
    from flora_deploy.config.schema import loads_config
    from flora_deploy.modal_app.factory import plan_app

    c = loads_config(text, source=str(base_dir / "flora.yaml"), base_dir=base_dir)
    plan = plan_app(c, determinism=ORACLE_DETERMINISM, derived_receipts=resolve(c))
    return plan.config_hash, dict(plan.stage_hashes)


def oracle_source(path: Path | None) -> dict[str, Any]:
    """(b) of a finished certified run, from its state file (read, never written): its run id,
    (b)'s job record (config file, data dir, hashes, GPU, video seconds) and episode count, and
    every job id that state ran. Raises when (b) cannot be repeated as it ran."""
    if path is None:
        raise RuntimeError("recertify: no certified state to repeat (b) from")
    try:
        src = rb.State.from_json(json.loads(Path(path).read_text()))
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(f"recertify: no readable certified state at {path}: {e}") from None
    b = src.steps.get("exact_e2e") or StepState()
    job = dict((b.result.get("jobs") or {}).get(JOB) or {})
    why = []
    if src.plan != CERTIFIED:
        why.append(f"its plan is {src.plan!r}, not {CERTIFIED!r}")
    if b.status != "done" or job.get("state") != "done":
        why.append(f"(b) is {b.status!r}, its job {job.get('state')!r}, not done")
    why += [f"(b)'s job has no {k}" for k in ("cfg", "job_id", "config_hash", "stage_hashes",
                                              "data", "video_s") if not job.get(k)]
    if job.get("preset") != "exact":
        why.append(f"(b)'s job preset is {job.get('preset')!r}, not exact")
    cfg = Path(str(job.get("cfg") or ""))
    if job.get("cfg") and not cfg.is_file():
        why.append(f"(b)'s config {cfg} is gone")
    elif job.get("data") and not (cfg.parent / str(job["data"])).is_dir():
        why.append(f"(b)'s data dir {cfg.parent / str(job['data'])} is gone")
    want_gpu = (pool_container("segcap").gpu or "").rstrip("!")
    gpus = {str(c.get("gpu") or "").rstrip("!") for c in job.get("containers") or ()}
    if gpus != {want_gpu}:
        why.append(f"(b) ran on {sorted(gpus) or 'no recorded GPU'}, not {want_gpu}")
    episodes = int(b.result.get("episodes") or 0)
    if why:
        raise RuntimeError(f"recertify: (b) of run {src.run_id} ({path}) cannot be repeated: "
                           + "; ".join(why))
    job_ids = sorted({str(r.get("job_id")) for s in src.steps.values()
                      for key in ("jobs", "witness_jobs")
                      for r in (s.result.get(key) or {}).values() if r.get("job_id")})
    return {"state": str(path), "run_id": src.run_id, "job": job, "episodes": episodes,
            "job_ids": job_ids}


def noise_all_job(st: State, emb: str, *, preview: bool = False) -> tuple[str, str]:
    """This attempt's (job id, index tag): ``wave-h-<run>-noise12-<emb>-a<n>`` (its own job
    and app, :func:`witness_config_text`) and ``noise12-<emb>-a<n>`` (its own local store and
    index summary beside (b)'s and (f)'s, never theirs). ``preview``: the attempt a run would
    start next (the running attempt is already counted in a run)."""
    s = st.steps.get("noise_all") or StepState()
    n = len(s.attempts) + 1 if preview else max(1, len(s.attempts))
    return f"wave-h-{st.run_id}-{NOISE_ALL_JOB}-{emb}-a{n}", f"{NOISE_ALL_JOB}-{emb}-a{n}"


def noise_all_check(st: State, spec: PlanSpec, *, preview: bool = False) -> dict[str, Any]:
    """Everything f2 checks before any app, from local reads only (the dry run and the run):
    (b) from the certified state, this state's run id is (b)'s, the job id is new, (b)'s config
    still names (b)'s project, and the renamed config plans to (b)'s config hash and stage
    hashes in this checkout. Returns the source, the config text, job id, tag and hashes;
    raises otherwise."""
    src = oracle_source(spec.source)
    job = src["job"]
    if st.run_id != src["run_id"]:
        raise RuntimeError(f"recertify: this state's run {st.run_id} is not (b)'s run "
                           f"{src['run_id']} (its store and staged media)")
    if src["episodes"] != len(spec.episodes):
        raise RuntimeError(f"recertify: (b) ran {src['episodes']} episodes, not the "
                           f"{len(spec.episodes)} of {spec.golden}")
    job_id, tag = noise_all_job(st, spec.embodiment, preview=preview)
    if job_id in src["job_ids"] or job_id == job["job_id"]:
        raise RuntimeError(f"recertify: job id {job_id} was already used by run {st.run_id}")
    ocfg = Path(job["cfg"])
    text = ocfg.read_text()
    if f"project: {job['job_id']}\n" not in text:
        raise RuntimeError(f"recertify: {ocfg} no longer names (b)'s project {job['job_id']}")
    new = witness_config_text(text, job_id)
    want = (str(job["config_hash"]), dict(job["stage_hashes"]))
    got_b, got = planned_hashes(text, ocfg.parent), planned_hashes(new, ocfg.parent)
    if got != want or got_b != want:
        raise RuntimeError(f"recertify: this checkout plans (b)'s config to {got_b[1]} "
                           f"(config {got_b[0][:12]}) and the noise config to {got[1]} (config "
                           f"{got[0][:12]}), not (b)'s {want[1]} (config {want[0][:12]}): run "
                           "f2 from a checkout that reproduces (b) (order A), or re-run (b)")
    return {"source": src, "cfg": ocfg.parent / f"flora-{tag}.yaml", "text": new,
            "job_id": job_id, "tag": tag, "config_hash": got[0], "stage_hashes": got[1]}


def noise_all_index_problems(summary: Path, oracle_summary: Path, src: Mapping[str, Any]
                             ) -> str:
    """Empty when f2's index holds what (b)'s did: every one of its episodes, the same dataset
    revision and video seconds as (b)'s own index summary (else (b)'s recorded video seconds,
    within 1%)."""
    try:
        d = json.loads(summary.read_text())
    except (OSError, ValueError) as e:
        return f"unreadable {summary.name}: {e}"
    try:
        o = json.loads(oracle_summary.read_text())
    except (OSError, ValueError):
        o = None
    n, got = (d.get("counts") or {}).get("episodes"), float(d.get("video_s") or 0.0)
    want_n = src["episodes"] if o is None else (o.get("counts") or {}).get("episodes")
    want = float(src["job"]["video_s"] if o is None else o.get("video_s") or 0.0)
    tol = 0.01 * max(want, 1.0) if o is None else 1e-6 * max(want, 1.0)
    if n != want_n or abs(got - want) > tol:
        return (f"{n} episodes / {got:.1f} video s indexed, not (b)'s {want_n} / {want:.1f} s")
    rev = ((d.get("dataset") or {}).get("revision"), ((o or {}).get("dataset") or {})
           .get("revision"))
    if o is not None and rev[0] != rev[1]:
        return f"dataset revision {rev[0]}, not (b)'s {rev[1]}"
    return ""


def run_noise_all(ctx: Ctx) -> dict[str, Any]:
    """Recertify f2: (b) of the certified run repeated on **all** of its episodes, exact vs
    exact (``noise_all``). (b)'s own config file with its ``project`` renamed (this attempt's
    job and app; the project is not hashed) and its own data dir, re-indexed with no
    ``--limit``; the config hash and stage hashes are checked equal to (b)'s recorded ones
    before any app (:func:`noise_all_check`), and the index to hold (b)'s episodes. Then
    ``flora deploy`` / ``flora run`` in nvidia mode on (b)'s pool (H200), under this step's
    budget and deadline. Nothing of the certified state or its records is written."""
    require_tempseg_pin(ctx)  # every GPU step runs from the pinned commit
    chk = noise_all_check(ctx.st, ctx.spec)
    src, job, cfg, tag = chk["source"], chk["source"]["job"], chk["cfg"], chk["tag"]
    cfg.write_text(chk["text"])
    if planned_hashes(cfg.read_text(), cfg.parent) != (chk["config_hash"],
                                                        chk["stage_hashes"]):
        raise RuntimeError(f"recertify: {cfg} as written does not plan to (b)'s hashes")
    _index(ctx, cfg, chk["job_id"], tag)  # no --limit: every episode of (b)'s data dir
    ocfg = Path(job["cfg"])
    why = noise_all_index_problems(
        ctx.work / f"summary-{tag}.json",
        ocfg.parent / f"summary-{ocfg.stem.removeprefix('flora-')}.json", src)
    if why:  # before any app: charged $0
        raise RuntimeError(f"noise_all index: {why}")
    r = _run_job(ctx, "noise_all", cfg, chk["job_id"], determinism=ORACLE_DETERMINISM,
                 usd_per_s=pool_container("segcap").usd_per_s(), spent_in_step=0.0,
                 side="noise_all")
    r["data"] = job["data"]
    res = _jobs_result({JOB: r}, 1)
    res.update(episodes=src["episodes"], stage_hashes_of_oracle=True,
               oracle={"state": src["state"], "run_id": src["run_id"],
                       "job_id": job["job_id"], "config_hash": job["config_hash"],
                       "stage_hashes": dict(job["stage_hashes"]),
                       "records": job.get("records")})
    return res


def new_recert_state(spec: PlanSpec) -> State:
    """A fresh recertify state: its own billing baseline and attempts, but (b)'s ``run_id``
    (:func:`oracle_source`), since (b)'s config writes into that run's store and reads that
    run's staged media; its apps are its own (per attempt), so billing never mixes them up."""
    src = oracle_source(spec.source)
    st = rb.new_state()
    st.run_id, st.plan = src["run_id"], RECERTIFY
    return st


def noise_all_lines(st: State, spec: PlanSpec) -> list[str]:
    """The dry run's / ``--status``'s f2 lines: what it repeats and whether this checkout
    plans it to (b)'s hashes (local reads only; nothing written)."""
    try:
        chk = noise_all_check(st, spec, preview=True)
    except Exception as e:  # noqa: BLE001 - shown, not raised: a run refuses on it
        return [f"  f2 check: REFUSED before any app: {e}"]
    job = chk["source"]["job"]
    sh = ", ".join(f"{k} {v}" for k, v in sorted(chk["stage_hashes"].items()))
    return [f"  f2 repeats (b) of run {chk['source']['run_id']} (job {job['job_id']}, "
            f"{chk['source']['episodes']} episodes, {job['video_s']} video s, "
            f"{', '.join(sorted({c.get('gpu') for c in job.get('containers') or ()}))}, "
            f"{ORACLE_DETERMINISM} mode) from {chk['source']['state']} (read-only)",
            f"  f2 config: {chk['cfg']} (project {chk['job_id']}), data dir "
            f"{Path(job['cfg']).parent / job['data']}, flora index with no --limit",
            f"  f2 stage hash: {sh} == (b)'s; config hash {chk['config_hash']} == (b)'s "
            "(this checkout)"]


CERT_STAGES = ("segment", "caption")
# the steps whose jobs' records (e) reads
RECORD_STEPS = ("exact_e2e", "noise", "balanced", "noise_all", "caption_iso", "repeat")


def certify_view(st: State, spec: PlanSpec | None = None) -> State:
    """What (e) exports and certifies, as the step names ``wave_h_finalize.py`` reads:
    ``exact_e2e`` (the oracle), ``noise``, ``balanced`` (J1 + J1w), ``caption_iso`` (J2 +
    J2w) and, when there is one, ``repeat`` (an older noise run: reported only). recertify
    takes (b)'s oracle from the certified state, its own f2 as the noise run and (b)'s old
    noise run as the repeat, each job with the store it wrote to. The certified state is the
    one f2 recorded it repeated (``noise_all.result.oracle.state``, :func:`run_noise_all`) when
    that file is there, else ``spec.source``. Other plans: ``st`` as is (a copy). Never saved."""
    view = copy.deepcopy(st)
    spec = spec or SPECS.get(st.plan or "")
    if spec is None or spec.name != RECERTIFY:
        return view
    rec = ((st.steps.get("noise_all") or StepState()).result.get("oracle") or {}).get("state")
    if rec and Path(rec).is_file():
        spec = replace(spec, source=Path(rec))
    src = source_state(spec)
    for name, frm in (("exact_e2e", "exact_e2e"), ("repeat", "noise")):
        s = copy.deepcopy(src.steps.get(frm) or StepState())
        for r in (s.result.get("jobs") or {}).values():
            r.setdefault("store", store_uri(src))
        view.steps[name] = s
    view.steps["noise"] = copy.deepcopy(st.steps.get("noise_all") or StepState())
    return view


def finished_view(st: State) -> State:
    """A copy of ``st`` in which a record-producing step (:data:`RECORD_STEPS`) that is not
    ``done`` has no result: what (e) exports and certifies. A failed or killed attempt's jobs
    (a partial result, or a step another process still runs) never reach a certificate."""
    view = copy.deepcopy(st)
    for name in RECORD_STEPS:
        s = view.steps.get(name)
        if s is not None and s.status != "done":
            s.result = {}
    return view


def run_flora_certify(ctx: Ctx, *, finalize: Any = None, sh: Callable[..., Any] | None = None,
                      lab_dir: Path | None = None) -> dict[str, Any]:
    """(e) of the certified plan, on CPU ($0): export the oracle's, the noise run's and
    balanced's records (``wave_h_finalize.fetch_records``, a Volume read), then ``flora
    certify`` per stage (``wave_h_finalize.certify_all``: the oracle, the noise run, the L0 JSON
    and the derived content digests (d)'s workers verified, stamped with the run's own time),
    of the **finished** steps only (:func:`finished_view`).
    Done only when **every** stage is ``certified``; then ``scripts/wave_h_finalize.py --write``
    (rates, certificates, report, docs) for this state. Otherwise nothing is written and the
    step fails with what is missing (``missing_for_certified``)."""
    if finalize is None:
        import wave_h_finalize as finalize
    paths = finalize.Paths(root=ROOT, state=ctx.state_path, records=ctx.spec.records,
                           lab=lab_dir or ctx.spec.labs)
    save_state(ctx.st, ctx.state_path)
    # only the finished steps' jobs (a parallel run: see there), recertify's with (b)'s oracle
    done = finished_view(certify_view(ctx.st, ctx.spec))
    finalize.fetch_records(paths, done)  # every job's records and digests (no container)
    out_dir = ctx.work / "certify"
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = finalize.certify_all(paths, done, out_dir,
                                    derived=finalize.derived_content_used(done))
    stages: dict[str, Any] = {}
    for o in outcomes:
        c = o.get("cert") or {}
        metrics = c.get("metrics") or {}
        stages[o["stage"]] = {
            "verdict": c.get("verdict"), "id": c.get("id"), "reasons": c.get("reasons"),
            "episodes": sum(b.get("n", 0) for b in (metrics.get("candidate") or {}).values()),
            "noise_episodes": (metrics.get("noise") or {}).get("n"),
            "witness_episodes": (metrics.get("witness") or {}).get("n"),
            # review 13 B1 / W1: what the engines did, as the certificate states it
            "preemptions": c.get("preemptions"),
            "compile_cache": (c.get("compile_cache") or {}).get("covers"),
            "missing": finalize.missing_for_certified(c) if c else None,
            "refused": o.get("refused") or o.get("deferred")}
    # certified, and the evidence covers a forced-cold container (the cold witness, W1)
    no_witness = sorted(k for k, v in stages.items() if not v.get("witness_episodes"))
    complete = set(stages) == set(CERT_STAGES) and not no_witness
    ok = complete and all(v["verdict"] == "certified" for v in stages.values())
    # Outcome (b) of the pre-registration: each stage final (certified or fail, nothing
    # refused or provisional) and at least one certified. A fail is final too (no re-roll):
    # finalize still lands the certified stages' certificates and locks; it never writes a
    # fail's (wave_h_finalize.analyze), so that stage stays uncertified.
    partial = (complete and not ok
               and all(v["verdict"] in ("certified", "fail") and not v.get("refused")
                       for v in stages.values())
               and any(v["verdict"] == "certified" for v in stages.values()))
    order = sorted(stages, key=lambda k: k != "segment")  # "segment certified, caption fail"
    res: dict[str, Any] = {"ok": ok or partial, "stages": stages, "tracked_usd": 0.0,
                           "apps": [],
                           "outcome": ", ".join(f"{k} {stages[k]['verdict'] or 'refused'}"
                                                for k in order)}
    if partial:
        res["partial"] = True
        res["note"] = (f"{res['outcome']}: the fail is final (one attempt, no re-roll); only "
                       "the certified stage's certificate is written")
    if not (ok or partial):
        res["error"] = "not certified: " + "; ".join(
            f"{k}: {v['verdict'] or v['refused']}" + (f" ({', '.join(v['missing'])})"
                                                      if v.get("missing") else "")
            + (" (no cold witness: (d)'s forced-cold job on the noise subset)"
               if k in no_witness else "")
            for k, v in sorted(stages.items()))
        return res
    if ctx.spec.embodiment != EMBODIMENT:  # finalize writes the human rates, report and docs
        res["finalize"] = (f"not run for {ctx.spec.embodiment}: wave_h_finalize.py --write "
                           "is human only; the certificates are in " + str(out_dir))
        return res
    args = [sys.executable, str(ROOT / "scripts" / "wave_h_finalize.py"), "--state",
            str(ctx.state_path), "--records", str(ctx.spec.records), "--write"]
    if ctx.spec.lab_dir is not None:  # recertify: its own lab JSON
        args[-1:-1] = ["--lab", str(lab_dir or ctx.spec.labs)]
    p = (sh or ctx.sh)(args, timeout=1800, check=False)
    res["finalize_rc"] = getattr(p, "returncode", None)
    res["finalize"] = " ".join(args[1:])
    return res


RUNNERS: dict[str, Callable[[Ctx], dict[str, Any]]] = {
    "tempseg": run_tempseg, "preflight": run_preflight, "exact_e2e": run_exact,
    "balanced": run_balanced, "lab": run_lab, "certify": run_certify, "noise": run_noise,
    "noise_all": run_noise_all, "caption_iso": run_caption_iso}


def runner(plan: str, step: str) -> Callable[[Ctx], dict[str, Any]]:
    """The certifying plans' (e) is ``flora certify`` + finalize, recertify's lab the
    Captioner-only one; the rest are shared."""
    if is_cert(plan) and step == "certify":
        return run_flora_certify
    if plan == RECERTIFY and step == "lab":
        return run_recert_lab
    return RUNNERS[step]


# ---------------------------------------------------------------------------------------------
# Report


def render_report(st: State, projections: Mapping[str, Projection], *,
                  plan: str = DEFAULT_PLAN, spec: PlanSpec | None = None) -> str:
    gcap = plan_global_cap(plan, projections, st.override_for(plan))
    caps, title = plan_caps(plan, projections), titles(plan, spec)
    lines = ["# Wave H-lite runbook summary", "",
             "Generated by `scripts/wave_h.py` (runbook: `docs/wave-h-runbook.md`). Numbers only;",
             "`docs/wave-h-report.md` is written by `scripts/wave_h_finalize.py`.", "",
             f"Run `{st.run_id}`, plan `{plan}`. Spent so far **${st.spent:.2f}** of the "
             f"${gcap:.2f} "
             "global cap (every attempt's tracked container-seconds x list $/h, a killed or",
             "unreported attempt at its budget, or the billed delta of every app from",
             "`scripts/spend.py` when larger; billing lags about an hour).", "",
             "| step | status | cap | projected (upper) | attempts | tracked | billed |",
             "|---|---|---|---|---|---|---|"]
    shown = (PLANS[plan] if plan == RECERTIFY  # recertify: its own steps only
             else STEPS + tuple(s for s in EXTRA_STEPS if s in PLANS[plan] or s in st.steps))
    for name in shown:
        s = st.steps.get(name, StepState())
        if name not in PLANS[plan] and s.status == "pending":
            if not is_cert(plan):
                lines.append(f"| {title[name]} | {DEFERRED_NOTE} | - | - | 0 | - | - |")
            continue
        p = projections.get(name)
        proj = f"${p.usd:.2f} (${p.usd_upper:.2f})" if p else "-"
        billed = f"${s.billed_usd:.2f}" if s.billed_usd is not None else "-"
        lines.append(f"| {title[name]} | {s.status} | ${caps[name]:.2f} | {proj} | "
                     f"{len(s.attempts)} | ${s.tracked_usd:.2f} | {billed} |")
    for name in shown:
        s = st.steps.get(name)
        if not s or s.status == "pending":
            continue
        lines += ["", f"## {title[name]}", "", f"Status: {s.status}."]
        lines += [f"- {n}" for n in s.notes]
        if s.attempts:  # only the latest attempt's notes: older errors are history
            lines += [f"- attempt {len(s.attempts)}: {n}" for n in s.attempts[-1].notes]
        res = {k: v for k, v in s.result.items() if k not in ("apps",)}
        if res:
            lines += ["", "```json", json.dumps(_numbers_only(res), indent=1, default=str)[:6000],
                      "```"]
    return "\n".join(lines) + "\n"


_PRIVATE_KEYS = {"rows", "caption", "captions", "text", "task", "facts", "episodes", "cfg",
                 "records"}


def _numbers_only(d: Any) -> Any:
    """Drop text that could carry private episode content (captions, tasks, rows)."""
    if isinstance(d, Mapping):
        return {k: _numbers_only(v) for k, v in d.items() if k not in _PRIVATE_KEYS}
    if isinstance(d, list):
        return [_numbers_only(x) for x in d]
    return d


# ---------------------------------------------------------------------------------------------


def letter(title: str) -> str:
    """A step title's letter (``"d2. lab ..."`` -> ``"d2"``)."""
    return title.split(".")[0]


def max_parallel_for(plan: str, only: str | None, requested: int) -> int:
    """The steps a run starts at once: ``--max-parallel`` for the certified plan's steps; 1
    for ``--only`` and for the other plans (one chain: nothing to run beside)."""
    return max(1, requested) if plan == CERTIFIED and not only else 1  # recertify: a chain


def wall_lines(st: State, projections: Mapping[str, Projection], plan: str,
               steps: Sequence[str], title: Mapping[str, str],
               max_parallel: int = rb.DEFAULT_MAX_PARALLEL) -> list[str]:
    """The dry run's wall-clock lines: the steps' order (what each needs) and the expected
    wall-clock one after the other vs ``--max-parallel`` (a done step takes none)."""
    deps = plan_deps(plan)
    wall = {s: (0.0, 0.0) if (st.steps.get(s) or StepState()).status == "done"
            else (projections[s].wall_s(), projections[s].wall_s(upper=True)) for s in steps}
    order = "; ".join(f"{letter(title[s])} after " + ", ".join(letter(title[d]) for d in need)
                      for s in steps if (need := rb.deps_within(s, deps, steps)))
    first = [letter(title[s]) for s in steps if not rb.deps_within(s, deps, steps)]
    out = [f"  order: {', '.join(first)} first" + (f"; {order}" if order else ""),
           rb.wall_line(steps, deps, wall, max_parallel)]
    if plan == CERTIFIED and wall.get("lab", (0.0, 0.0))[0] > 0:  # the lab may be skipped
        out.append(rb.wall_line(steps, deps, {**wall, "lab": (0.0, 0.0)}, max_parallel)
                   .replace("wall-clock", "with d2 skipped (lab already done)", 1))
    return out


def print_plan(st: State, projections: Mapping[str, Projection], out=print, *,
               manifest: Mapping[str, Any] | None = None, plan: str = DEFAULT_PLAN,
               only: str | None = None, spec: PlanSpec | None = None,
               max_parallel: int = rb.DEFAULT_MAX_PARALLEL) -> None:
    steps = (only,) if only else PLANS[plan]
    spec = spec or SPECS[plan]
    gcap = plan_global_cap(plan, projections, st.override_for(plan))
    caps, title = plan_caps(plan, projections), titles(plan, spec)
    out(f"Wave H-lite run {st.run_id} ({spec.embodiment} only), plan {plan} "
        f"({' -> '.join(title[s].split('.')[0] for s in steps)}): spent ${st.spent:.2f} of "
        f"${gcap:.2f} (billing read "
        f"{'never' if st.billed_at is None else time.ctime(st.billed_at)})")
    if NO_CAMERA_SCALING:
        out("  warning: planner/rates.json has no camera_scaling: 3-view episodes are priced "
            "at factor 1.0 (the 1-view rates)")
    for line in held_lines(st):
        out(line)
    if manifest is not None:
        eps, miss = (episodes_in(manifest, spec.episodes),
                     missing_episodes(manifest, spec.episodes))
        src = (f"expected durations from the {spec.golden} selection (offline, +"
               f"{DURATION_HEADROOM:.0%})" if manifest.get("offline")
               else f"{spec.golden} manifest")
        dur = durations(manifest)
        out(f"  episodes: {len(eps)} {spec.embodiment}, {sum(dur[e] for e in eps):.1f} video s "
            f"({src}"
            + (f"; NOT STAGED, left out: {', '.join(miss)}" if miss else "") + ")")
        if is_cert(plan):
            nv = sum(dur.get(e, 0.0) for e in spec.noise)
            lim = (f"`flora index --limit {spec.noise_limit}`" if spec.noise_limit is not None
                   else "`flora index` with no --limit")
            out(f"  noise run: {len(spec.noise)} of them, {nv:.1f} video s ({lim} on the "
                "oracle's config and data dir: the same stage hash)")
            if spec.witness != spec.noise:
                wv = sum(dur.get(e, 0.0) for e in spec.witness)
                out(f"  cold witnesses: {len(spec.witness)} of them ({NOISE_PREFIX} local ids, "
                    f"`--limit {spec.witness_limit}`), {wv:.1f} video s")
    for name in steps:
        p = projections[name]
        s = st.steps.get(name, StepState())
        d = decide(st, plan, name, projections, only=only, embodiment=spec.embodiment)
        rule = {"run": "ok", "done": "done"}.get(d.action) or f"{d.action.upper()}: {d.message}"
        out(f"  {title[name]:66s} [{s.status}] cap ${caps[name]:.2f}  nominal "
            f"${p.usd:.2f}  upper ${p.usd_upper:.2f}  rule: {rule}")
        for i in p.items:
            out(f"      {i['what']:60s} {i['gpu']:>5s} {i['seconds']:6.0f} s / "
                f"{i['upper_seconds']:6.0f} s x ${i['usd_per_h']:.2f}/h = ${i['usd']:.3f} "
                f"(upper ${i['usd_upper']:.3f})")
    if (plan == RECERTIFY and "noise_all" in steps
            and (st.steps.get("noise_all") or StepState()).status != "done"):
        for line in noise_all_lines(st, spec):
            out(line)
    for name in () if only or is_cert(plan) else deferred_steps(plan):
        out(f"  {title[name]:66s} [{DEFERRED_NOTE}; never runs in plan {plan}]")
    if plan == RECERTIFY:
        out("  assumption: f2 (noise_all) runs from a checkout whose hashed code is (b)'s (before "
            "the v4 code lands): its segcap stage hash must equal (b)'s recorded one, else it "
            "refuses before any app; d2', d' and g need the v4 code ('4' in "
            "certkey.GENERATOR_VERSIONS). The single-pool jobs (J1w, J2, J2w) are priced "
            "start + work + 30 s (upper (start + work) x (1 + band) + 30 + "
            f"{CLIENT_SLACK_S} s), no scale-down tail; caps are each step's upper rounded up to "
            "$0.05, the global cap the uppers' sum rounded up to $0.10")
        f2 = projections["noise_all"]
        rest = [s for s in steps if s != "noise_all"]
        out(f"  split: f2 noise_all nominal ${f2.usd:.2f}, upper ${f2.usd_upper:.2f}; the rest "
            f"(phase 2) nominal ${sum(projections[s].usd for s in rest):.2f}, upper "
            f"${sum(projections[s].usd_upper for s in rest):.2f}")
    if is_cert(plan) and ("exact_e2e" in steps or "noise_all" in steps):
        from flora_deploy.planner.rates import RateTable

        t = RateTable.load()
        e, su = t.rate("exact", "segcap", 1, 1), t.startup_for("exact", "segcap")
        out(f"  assumption: (b) and (f) are priced on the measured exact segcap rate "
            f"{e.container_s_per_video_s:g} container-s/video-s +/-{e.uncertainty:.0%} (H200, "
            f"default mode, c={e.concurrency}, {e.date}) and the measured @enter "
            f"{su.startup_s:g} s +/-{su.uncertainty:.0%} per job (each job is its own cold "
            "start: (f) deploys its own app, and the oracle's is stopped when its job is "
            "composed). Image pull and "
            "scheduling are not in the measured startup; the upper bound's +100% and the "
            f"{CLIENT_SLACK_S} s client slack cover them")
        out("  assumption: certified plan caps are each step's own upper projection rounded "
            "up to $0.05 (plan_caps); the global cap is their sum rounded up to $0.10")
    if plan == RECERTIFY and "noise_all" in steps:
        from flora_deploy.planner.rates import RateTable

        t = RateTable.load()
        e, su = t.rate("exact", "segcap", 1, 1), t.startup_for("exact", "segcap")
        out(f"  assumption: f2 is priced as (b) was: the measured exact segcap rate "
            f"{e.container_s_per_video_s:g} container-s/video-s +/-{e.uncertainty:.0%} (H200, "
            f"default mode) and the measured @enter {su.startup_s:g} s +/-{su.uncertainty:.0%} "
            f"(its own app: a cold start), {CLIENT_SLACK_S} s client slack; its cap is its "
            "upper projection rounded up to $0.05 (plan_caps), the global cap the plan's "
            "uppers rounded up to $0.10")
    if spec.scale:
        sc = spec.scale
        out(f"  assumption: {spec.embodiment} is three-view; the measured 1-view rates are scaled "
            "per kind by (1 - g) + g x r: g = generation's share of the deployed busy time (lab "
            "generate_s / (d)'s busy s: "
            + ", ".join(f"{k} {GENERATE_S[k]:.1f}/{BUSY_S[k]:.1f} s" for k in GENERATE_S)
            + "), r = per-request prompt tokens 3-view / 1-view ("
            + ", ".join(f"{k} {PROMPT_TOKENS_3VIEW[k]:,.0f}/{PROMPT_TOKENS_1VIEW[k]:,.0f}"
                        for k in PROMPT_TOKENS_3VIEW)
            + "; 3-view: ALOHA in docs/exact-concurrency.md, 1-view: the balanced lab): "
            f"segment x{sc['segment']:.2f}, caption x{sc['caption']:.2f}; exact segcap "
            f"x{sc['segcap']:.2f} (NVIDIA's call mix, only the 1-view half of its measured "
            "1-/3-view mix). Same requests per video s as human; decode and frame preparation "
            "kept at 1x (11 of 12 are three 640-680x480 views, ~0.45x the pixels of 1080p). The "
            "caption max "
            "(13,650 tokens) is within the measured +/-50%")
    if "balanced" in steps:
        basis = startup_basis()
        out("  assumption: (d) is priced with a cold compile: startup segment "
            f"{basis['segment']}, caption {basis['caption']} (a startup measured cold is taken "
            f"as is; any other is scaled by {COLD_COMPILE_FACTOR:.2f}, the Rewarder's cold/warm "
            "@enter 318 s / 229 s, docs/cold-start.md). The warm compile-cache rows "
            "(rates.json `compile_cache: warm`, placeholders) are never used here: a cache is "
            "keyed by the engine spec, so none is published for this code's yet; a worker that "
            "compiled publishes one after its first shard (engines/compile_cache.py)")
        if is_cert(plan):
            what = ("the segment stage alone (J1w; the caption witness is (g)'s J2w)"
                    if spec.segment_only_witness else "the same config")
            out(f"  assumption: (d) then runs the cold witness, {what} on the "
                f"{len(spec.witness)} witness episodes with {COMPILE_CACHE_ENV}=off (one "
                "container per kind forced cold, priced cold): the certificate's evidence "
                "covers warm (the cache (d)'s first container publishes) and cold compiles")
        lim = balanced_engine_limits()
        out("  assumption: (d) runs the batch-fill engine limits and batch policy: "
            + ", ".join(f"{k} on {v['gpu']} max_num_seqs {v['max_num_seqs']}, "
                        f"max_num_batched_tokens {v['max_num_batched_tokens']}, "
                        f"{v['infer_episodes']} episodes per batcher" for k, v in lim.items())
            + "; its rates were measured before batch fill, so they can only overstate it")
    if "lab" in steps:
        out("  assumption: every lab engine compiles cold (the lab's own unkeyed "
            "flora-cache:/balanced/<engine>[-dtbias]; no deployed job writes it, and the "
            "2026-10-22 lab after (d) was cold too); calls and deadlines: "
            + "; ".join(f"{k} " + " + ".join(f"{a} {LAB_CALL_DEADLINE_S[(k, a)]:.0f} s"
                                              for a in lab_calls(k, spec)) for k in LAB_CALLS)
            + f" (expected x {1 + LAB_UNCERTAINTY:.2f}, each <= the {LAB_TIMEOUT_S} s "
            "container timeout)"
            + ("; the Captioner is split (baseline, then plugin: one cold engine per "
               "container; its v4 stage is priced at the measured 2026-10-22 lab + "
               f"{LAB_V4_CAPTION_S:.0f} s); a saved baseline call is reused by a retry"
               if len(lab_calls("captioner", spec)) > 1 else ""))
    if is_cert(plan) and "lab" in steps:
        done, why = lab_already_done(spec.labs)
        out(f"  lab (d2): {'already done, skipped at $0' if done else 'runs'} ({why})")
    nominal = sum(projections[s].usd for s in steps)
    upper = sum(projections[s].usd_upper for s in steps)
    how = global_cap_label(st, plan, projections)
    out(f"  total: nominal ${nominal:.2f}, upper ${upper:.2f} (incl. {CLIENT_SLACK_S} s client "
        f"slack per app); global cap ${gcap:.2f} ({how})")
    for line in wall_lines(st, projections, plan, steps, title,
                           max_parallel_for(plan, only, max_parallel)):
        out(line)
    honest = honest_caps({s: projections[s] for s in steps})
    left = [s for s in steps if (st.steps.get(s) or StepState()).status != "done"]
    upper_left = st.spent + sum(projections[s].usd_upper for s in left)
    if left != list(steps) and upper_left <= gcap + 1e-9:  # a done step's actual cost counts
        out(f"  fits: spent ${st.spent:.2f} + the upper of the steps not done "
            f"${upper_left - st.spent:.2f} = ${upper_left:.2f} <= the ${gcap:.2f} global cap")
    elif left != list(steps):
        out(f"  DOES NOT FIT: spent ${st.spent:.2f} + the upper of the steps not done "
            f"${upper_left - st.spent:.2f} = ${upper_left:.2f} > the ${gcap:.2f} global cap "
            f"(short ${upper_left - gcap:.2f}); the runbook stops at the first step whose upper "
            "projection, with the later steps' kept for them, no longer fits the cap left")
    elif upper > gcap + 1e-9:
        out(f"  NOT HONEST AT ${gcap:.2f}: the upper total needs a global cap of at "
            f"least ${honest['global']:.2f} (per step: "
            + ", ".join(f"{s} ${honest[s]:.2f}" for s in steps) + "); the runbook stops at the "
            "first step whose upper projection, with the later steps' kept for them, no longer "
            "fits the cap left")


def app_name(run_id: str, step: str, attempt: int) -> str:
    """The app name of one attempt of one step: ``flora-wh-<run>-<step>-a<n>`` (Modal's
    billing reports it per app, so each attempt's spend is its own)."""
    return f"flora-wh-{run_id}-{step.replace('_', '-')}-a{attempt}"


# Per step: stop -> container gone (the lab and conversion calls end with their client timeout;
# a deployed job with ``flora run``'s), and the hard limit its containers cannot outlive.
STEP_SLACK_S = {"tempseg": LAB_SLACK_S, "preflight": LAB_SLACK_S, "lab": LAB_SLACK_S}
# every lab call, one after the other (TempSeg one, the Captioner two)
STEP_HARD_TIMEOUT_S = {"lab": sum(len(a) for a in LAB_CALLS.values()) * LAB_TIMEOUT_S}


stop_killed = rb.stop_killed


def held_lines(st: State) -> list[str]:
    """One line per attempt charged over its tracked cost until billing covers it."""
    return [f"  held: {name} attempt {i} charged ${a.charged(st.watermark):.2f} (tracked "
            f"${a.tracked_usd:.2f}) until billing covers it, about {rb.utc(a.hold_clears_at())}"
            for name, i, a in st.held()]


DEAD_HEARTBEAT_S = 3 * HEARTBEAT_S  # a running attempt silent this long: its process is gone


def approver() -> str:
    """Who approved a ``--global-cap`` (the login running ``--yes-spend``)."""
    import getpass

    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name: still recorded as approved
        return "unknown"


note_stop = rb.note_stop


def simulate_run_start(st: State, *, out: Callable[[str], None] = print) -> None:
    """On a copy of the state (the dry run, ``--status``): what ``--yes-spend`` does before its
    first step, without writing or stopping anything. It reads the billing baseline when there
    is none and billing (read-only), and takes the apps of killed attempts as stopped now (a
    run stops them by id first; if that stop fails, the run charges them their budget)."""
    if not st.billing_baseline:
        try:
            st.billing_baseline = dict(billed_by_app(st.billing_since))
        except Exception as e:  # noqa: BLE001
            out(f"note: billing unavailable ({e!r}); tracking only")
    now = time.time()
    for a in st.killed_unstopped():
        a.stopped_at = now
        out(f"note: a run first stops {', '.join(a.app_ids or a.apps)} (left by a killed run) "
            "by id; this preview assumes that stop succeeds")
    refresh_billing(st)


def next_decision(st: State, plan: str, projections: Mapping[str, Projection],
                  only: str | None = None, embodiment: str = EMBODIMENT) -> Decision | None:
    """The decision a run takes first: the first step (of ``--only`` or the plan) not done,
    not running in a parallel run (one that reserved: it is shown as running), and not
    waiting for a running step it needs (``lab`` while ``balanced`` runs cannot start)."""
    steps = (only,) if only else PLANS[plan]
    deps = plan_deps(plan)

    def running(n: str) -> bool:
        cur = (st.steps.get(n) or StepState()).current
        return cur is not None and cur.reserved_usd is not None

    def waits(n: str) -> bool:  # ``n`` or a step it needs (directly or not) is running
        return running(n) or any(waits(d) for d in rb.deps_within(n, deps, steps))

    for name in steps:
        if waits(name):
            continue
        d = decide(st, plan, name, projections, only=only, embodiment=embodiment)
        if d.action != "done":
            return d
    return None


def print_status(st: State, projections: Mapping[str, Projection], *, plan: str,
                 only: str | None = None, state_path: Path | None = None,
                 out: Callable[[str], None] = print,
                 spec: PlanSpec | None = None) -> Decision | None:
    """``--status``: spend, holds, the cap in force and the next decision (on a copy), for the
    plan's embodiment (``spec``: its episodes, step titles and next command)."""
    spec = spec or SPECS[plan]
    gcap = plan_global_cap(plan, projections, st.override_for(plan))
    caps, title = plan_caps(plan, projections), titles(plan, spec)
    out(f"Wave H-lite run {st.run_id} ({spec.embodiment}), plan {plan}"
        + (f", state {state_path}" if state_path else ""))
    out(f"  episodes: {len(spec.episodes)} {spec.embodiment} ({spec.golden}), noise subset "
        f"{len(spec.noise)}")
    basis = ("from the billing rows" if st.billed_through is not None else
             f"by the {BILLING_LAG_S} s lag rule: no rows kept yet")
    out(f"  billing: read {rb.utc(st.billed_at)}, complete through {rb.utc(st.watermark)} "
        f"({basis})")
    ov = st.override_for(plan)
    approved = ""
    if ov is not None:
        approved = (f"; --global-cap ${ov:.2f} approved by {st.cap_override_by} at "
                    f"{rb.utc(st.cap_override_at)}" if st.cap_override_by else
                    f"; --global-cap ${ov:.2f} approved (who and when not recorded)")
    out(f"  global cap ${gcap:.2f} ({global_cap_label(st, plan, projections)}{approved}), "
        f"derived ${plan_global_cap(plan, projections):.2f}")
    wm = st.watermark
    tracked = sum(s.tracked_usd for s in st.steps.values())
    charged = sum(s.charged(wm) for s in st.steps.values())
    out(f"  spent ${st.spent:.2f}: tracked ${tracked:.2f}, charged ${charged:.2f}, billed "
        + ("-" if st.billed_total is None else f"${st.billed_total:.2f}"))
    for name in (only,) if only else PLANS[plan]:
        s = st.steps.get(name) or StepState()
        last = s.attempts[-1] if s.attempts else None
        err = next((n for n in reversed(last.notes) if n.startswith(("error", "stop rule"))),
                   None) if last else None
        stop = next((n for n in s.notes if n.startswith("stop rule")), None)
        why = f"  [{err or stop}]" if (err or stop) and s.status != "done" else ""
        if name == "balanced" and witness_pending(st):
            why += (f"  [witness pending: will re-run the {len(spec.witness)}-episode cold job "
                    "only]")
        out(f"  {title[name]:66s} {s.status:9s} cap ${caps[name]:.2f}  attempts "
            f"{len(s.attempts)}  charged ${s.charged(wm):.2f}  tracked ${s.tracked_usd:.2f}"
            + ("" if s.billed_usd is None else f"  billed ${s.billed_usd:.2f}") + why)
    for line in held_lines(st):
        out(line)
    if plan == RECERTIFY and (st.steps.get("noise_all") or StepState()).status != "done":
        for line in noise_all_lines(st, spec):
            out(line)
    for line in running_lines(st, plan, title, only):
        out(line)
    d = next_decision(st, plan, projections, only, spec.embodiment)
    if d is None:
        busy = st.running()
        out(f"  next: none until {', '.join(busy)} ends (every other step waits for it)"
            if busy else "  next: nothing (every step is done)")
        return None
    out(f"  next: {d.step}: {d.action}: {d.message}")
    for r in d.reasons:
        out(f"    because {r}")
    if d.command:
        out(f"    command: {d.command}")
    return d


def running_lines(st: State, plan: str, title: Mapping[str, str], only: str | None = None
                  ) -> list[str]:
    """``--status``: the running steps (a parallel run has several) with what each reserved
    against the global cap, what it charged so far and what is still kept for it; then the
    steps a run could start beside them (what they need is done)."""
    wm, out = st.watermark, []
    running = st.running()
    for name, a in running.items():
        res = a.budget_usd if a.reserved_usd is None else a.reserved_usd
        out.append(f"  running: {letter(title.get(name, name))} {name}: reserved ${res:.2f}, "
                   f"charged ${a.charged(wm):.2f} so far, ${a.outstanding(wm):.2f} still kept "
                   f"(heartbeat {rb.utc(a.last_activity)})")
    if running:
        out.append(f"  reserved by running steps: ${sum(st.reservations().values()):.2f} "
                   f"(spent ${st.spent:.2f})")
    steps = (only,) if only else PLANS[plan]
    deps = plan_deps(plan)
    ready = [s for s in steps if s not in running
             and (st.steps.get(s) or StepState()).status != "done"
             and all((st.steps.get(d) or StepState()).status == "done"
                     for d in rb.deps_within(s, deps, steps))]
    if running or len(ready) > 1:
        out.append("  ready to start: " + (", ".join(ready) or "none")
                   + " (what each needs is done)")
    return out


class NotStaged(Exception):
    """The golden set's manifest is not on the golden Volume (or the Volume does not exist)."""


def _not_found(e: BaseException) -> bool:
    """A missing Volume file (``FileNotFoundError``) or Volume (``modal`` ``NotFoundError``)."""
    return isinstance(e, FileNotFoundError) or type(e).__name__ == "NotFoundError"


def load_manifest(offline: bool, version: str = "golden/v1") -> dict[str, Any]:
    """The ``version`` manifest from the golden Volume (``offline``: the selection's expected
    durations). Raises :class:`NotStaged` when the set was never staged there."""
    if offline:
        return offline_manifest(DURATION_HEADROOM, version)
    try:
        if version == "golden/v1":
            return h0.golden_manifest()
        import modal

        vol = modal.Volume.from_name(h0.GOLDEN_VOLUME, version=2)
        return json.loads(b"".join(vol.read_file(f"{version}/manifest.json")))
    except Exception as e:
        if not _not_found(e):
            raise
        raise NotStaged(f"{version} is not staged: no {version}/manifest.json on Volume "
                        f"{h0.GOLDEN_VOLUME} ({type(e).__name__}); run "
                        f"scripts/golden_stage.py --version {version} (or pass --offline to "
                        "price the selection's expected durations)") from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print the plan; spend nothing")
    g.add_argument("--yes-spend", action="store_true", help="run (GPU steps spend money)")
    g.add_argument("--report", action="store_true",
                   help="rewrite reports/wave-h/runbook.md from the state (never "
                        "docs/wave-h-report.md: wave_h_finalize.py owns it)")
    g.add_argument("--status", action="store_true",
                   help="where the run stands: spend, held attempts and when they clear, the "
                        "cap and who approved it, and the next decision (reads; never writes)")
    ap.add_argument("--plan", choices=tuple(PLANS), default=DEFAULT_PLAN,
                    help="balanced-only (default): steps 0, a, d; full: every step; "
                         "certified: b, f, d, (d2), g, e on golden/v2 (its own state); "
                         "recertify: f2 (the certified run's (b) repeated on all 12 episodes), "
                         "d2', d', g, e' (stage-isolated caption on (b)'s oracle; its own "
                         "state; reads the certified state, never writes it)")
    ap.add_argument("--embodiment", choices=ALL_EMBODIMENTS, default=EMBODIMENT,
                    help="certified only: the embodiment to certify (human: golden/v2; yam: "
                         "golden/v3-yam; each its own state). dex, tianji, yam32: no golden "
                         "certification set yet")
    ap.add_argument("--only", choices=ALL_STEPS, help="run just this step (if not done)")
    ap.add_argument("--source-state", type=Path, default=None,
                    help="recertify only: the certified state whose (b) f2 repeats (default "
                         "reports/wave-h/certified/state.json; read, never written)")
    ap.add_argument("--global-cap", type=float, default=None, metavar="USD",
                    help="raise this plan's global cap to this (a cap the user approved; "
                         "saved in the state only with --yes-spend, for this --plan only; "
                         f"finite, at most ${MAX_GLOBAL_CAP_USD:.0f}, never below the plan's "
                         "derived cap or what was spent)")
    ap.add_argument("--offline", action="store_true",
                    help="dry run / report: price the golden selection's expected durations "
                         "instead of reading the manifest from the Volume")
    ap.add_argument("--state", type=Path, default=None,
                    help="default: the plan's (reports/wave-h/state.json; certified: "
                         "reports/wave-h/certified/state.json)")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--max-parallel", type=int, default=rb.DEFAULT_MAX_PARALLEL, metavar="N",
                    help="certified: run up to N independent steps at once (default "
                         f"{rb.DEFAULT_MAX_PARALLEL}: b, f and d at once, d2 after d, e last); "
                         "each reserves its budget against the global cap while it runs. It "
                         "counts steps, not GPUs: b, f and d hold 4 GPUs at once (2x H200, "
                         "L40S, H100); see --max-gpus. 1: one after the other, as before")
    ap.add_argument("--max-gpus", action="append", metavar="N|TYPE=N", default=None,
                    help="certified, with --max-parallel > 1: the most GPUs held at once across "
                         f"every running step's apps (default {DEFAULT_MAX_GPUS}, the peak); "
                         "TYPE=N caps one GPU type, e.g. --max-gpus H200=1 runs f after b (for "
                         "a workspace that would queue a second H200, which eats into f's "
                         "in-step deadline). A step that does not fit waits for one to end. "
                         "Repeatable")
    args = ap.parse_args(argv)
    if args.max_parallel < 1:
        ap.error("--max-parallel must be at least 1")
    try:
        args.gpu_caps = parse_gpu_caps(args.max_gpus)
    except ValueError as e:
        ap.error(str(e))
    try:
        spec = spec_for(args.plan, args.embodiment)
    except KeyError:
        ap.error(f"--embodiment {args.embodiment}: " + (
            f"no golden certification set (certified: {', '.join(CERT_SPECS)})"
            if args.plan == CERTIFIED else "only --plan certified takes another embodiment"))
    other = {s.state.resolve() for e, s in CERT_SPECS.items() if e != args.embodiment}
    if args.state is not None and args.plan == CERTIFIED and args.state.resolve() in other:
        ap.error(f"--state {args.state} is another embodiment's certified state")
    if args.source_state is not None:
        if args.plan != RECERTIFY:
            ap.error("--source-state is for --plan recertify")
        spec = replace(spec, source=args.source_state)
    if args.plan == RECERTIFY:
        certified = {s.state.resolve() for s in CERT_SPECS.values()} | {spec.source.resolve()}
        if args.state is not None and args.state.resolve() in certified:
            ap.error(f"--state {args.state} is a certified run's state: recertify keeps its "
                     "own and never writes that one")
    args.state = args.state or spec.state
    args.work = args.work or spec.work
    if args.offline and args.yes_spend:
        ap.error("--offline is for --dry-run / --report (a run stages the real files)")
    if args.only and args.only not in PLANS[args.plan]:
        ap.error(f"--only {args.only}: {DEFERRED_NOTE} (not in plan {args.plan}; use --plan "
                 "full)")

    if args.plan == RECERTIFY and not args.state.exists():
        try:  # a new recertify state takes (b)'s run id: (b)'s store and staged media
            st = new_recert_state(spec)
        except RuntimeError as e:
            print(f"refused: {e}")
            return 2
    else:
        st = load_state(args.state)
    if (st.plan is not None and st.plan != args.plan
            and (is_cert(st.plan) or is_cert(args.plan))):
        print(f"refused: {args.state} was run under plan {st.plan}, not {args.plan}")
        return 2
    preview = args.dry_run or args.status or not (args.yes_spend or args.report)
    if preview:  # the dry run and --status decide on a copy: the state file is never written
        st = copy.deepcopy(st)
    if is_cert(args.plan):
        st.plan = args.plan
    if args.plan == CERTIFIED and (why := pre_v4_state(st, spec)):
        if args.yes_spend:  # before anything starts: no app, no billing baseline written
            print(f"refused: {args.state} is a pre-v4 certified run ({why}); its steps ran "
                  "without the stage-isolated caption job (g), so none of its v4 steps run "
                  "on it. Human re-certifies with `--plan recertify` (its own state, reading "
                  "this one read-only)")
            return 2
        print(f"note: {args.state} is a pre-v4 certified run ({why}): --yes-spend refuses it "
              "(use --plan recertify)")
    if adopt_balanced_job(st):
        print("note: (d)'s deployed job is done; (d) counts as done and its lab is step lab")
    if args.yes_spend:
        st.mark_killed()
    elif preview:  # what a run would find: a heartbeat this old is a dead process
        st.mark_killed(stale_before=time.time() - DEAD_HEARTBEAT_S)
    try:
        manifest = load_manifest(args.offline, spec.golden)
    except NotStaged as e:
        print(f"refused: {e}")
        return 2
    miss = missing_episodes(manifest, spec.episodes)
    if miss:
        print(f"note: only {len(spec.episodes) - len(miss)} of the {len(spec.episodes)} "
              f"{spec.embodiment} episodes are staged; using those (missing: "
              f"{', '.join(miss)})")
    lab_done = is_cert(args.plan) and lab_already_done(spec.labs)[0]
    reuse = () if lab_done or args.plan == RECERTIFY else lab_reuse(st, spec)
    if not lab_done:  # a saved split baseline call: a retry pays for the plugin call alone
        reuse = tuple(reuse) + saved_lab_calls(st.run_id, spec)
    pending = is_cert(args.plan) and witness_pending(st)
    iso_pending = is_cert(args.plan) and iso_witness_pending(st)
    projections = plan_projections(args.plan, manifest, lab_done=lab_done, reuse=reuse,
                                   spec=spec, witness_only=pending,
                                   iso_witness_only=iso_pending)
    caps = plan_caps(args.plan, projections)
    if pending:
        print(f"note: (d)'s deployed job is done; witness pending: will re-run the "
              f"{len(spec.witness)}-episode cold job only")
    if iso_pending:
        print(f"note: (g)'s J2 is done; its cold witness J2w ({len(spec.witness)} episodes) "
              "re-runs alone")
    if args.plan == RECERTIFY and not lab_done:
        print("note: recertify's lab reuses TempSeg's saved result; the Captioner lab runs")
    if reuse:
        print(f"note: lab reuses the saved {', '.join(reuse)} result(s) (evidence, run "
              f"{st.run_id}, (d)'s episodes); only the rest runs")
    if args.global_cap is not None:
        why = check_global_cap(args.global_cap, spent=st.spent,
                               derived=plan_global_cap(args.plan, projections))
        if why:
            print(f"refused: {why}")
            return 2
        if args.report:
            print("note: --global-cap ignored by --report (the report shows the cap in force)")
        else:
            st.cap_override, st.cap_override_plan = round(args.global_cap, 2), args.plan
            if args.yes_spend:  # only a run approves it; a dry run only shows its effect
                st.cap_override_by, st.cap_override_at = approver(), time.time()
                save_state(st, args.state)
                print(f"global cap for plan {args.plan} raised to ${st.cap_override:.2f} "
                      "(user-approved; saved in the state)")
            else:
                print(f"dry run: would raise the global cap for plan {args.plan} to "
                      f"${st.cap_override:.2f}; not saved (pass --yes-spend to approve it)")
    gcap = plan_global_cap(args.plan, projections, st.override_for(args.plan))
    if preview and not args.offline:  # what a run reads first (billing is read-only)
        simulate_run_start(st)
    if args.status:
        print_status(st, projections, plan=args.plan, only=args.only, state_path=args.state,
                     spec=spec)
        return 0
    print_plan(st, projections, manifest=manifest, plan=args.plan, only=args.only, spec=spec,
               max_parallel=args.max_parallel)
    report = spec.report or REPORT
    if args.report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(st, projections, plan=args.plan, spec=spec))
        print(f"wrote {report}")
        return 0
    if not args.yes_spend:
        print("dry run: nothing started (run with --yes-spend after the user approves)")
        return 0

    args.work = args.work.resolve()
    args.state = args.state.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    if not st.billing_baseline:
        try:
            st.billing_baseline = billed_by_app(st.billing_since)
        except Exception as e:  # noqa: BLE001
            print(f"billing baseline unavailable ({e!r}); tracking only")
    stop_killed(st)
    st.gate_cap, st.gate_cap_plan = gcap, args.plan  # the cap this run gates on
    save_state(st, args.state)
    os.chdir(args.work)  # relative input.path is hashed against the cwd: plan where flora runs
    ctx = Ctx(st, manifest, args.work, state_path=args.state, global_cap=gcap, spec=spec,
              caps=caps, plan_steps=PLANS[args.plan], projections=projections)
    def after(name: str, s: StepState) -> int | None:
        if s.status == "needs_pin":
            print(pin_instructions(s.result))
            return NEEDS_PIN
        return None

    rc = rb.run_steps(
        st, [n for n in PLANS[args.plan] if not args.only or n == args.only],
        state_path=args.state, ctx=ctx, run=lambda name, c: runner(args.plan, name)(c),
        projection=projections.__getitem__,
        decide=lambda name, p: decide(st, args.plan, name, projections, only=args.only,
                                      embodiment=spec.embodiment),
        check_after=lambda name: check_after(st, name, caps=caps, global_cap=gcap),
        refresh=lambda s: refresh_billing(s),  # looked up per call (tests patch it)
        attempt_kw=lambda name: {"slack_s": STEP_SLACK_S.get(name, CLIENT_SLACK_S),
                                 "hard_timeout_s": STEP_HARD_TIMEOUT_S.get(name)},
        status_of=lambda res: ("done" if res.get("ok") else
                               "needs_pin" if res.get("needs_pin") else "failed"),
        after=after, deps=plan_deps(args.plan),
        max_parallel=max_parallel_for(args.plan, args.only, args.max_parallel),
        gpus=CERT_GPUS if args.plan == CERTIFIED else None, gpu_caps=args.gpu_caps)
    save_state(st, args.state)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(st, projections, plan=args.plan, spec=spec))
    print(f"wrote {report}; state {args.state}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
