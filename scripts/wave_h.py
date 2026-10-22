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

**Plans** (``--plan``, :data:`PLANS`). ``balanced-only`` (the default; the user's decision of
2026-10-22) runs steps 0 -> a -> d and nothing else; (b), (e) and (f) are listed as "deferred to
the certified run" and never run. Its global cap is the sum of those three steps' upper
projections rounded up to $0.10 (:func:`plan_global_cap`). ``full`` runs every step under
:data:`GLOBAL_CAP_USD`. Both share the state file, so a later ``--plan full`` skips what
``balanced-only`` finished. ``full`` cannot certify: it stages one data dir per step, so its
oracle's and balanced's episode uids (``<data dir>--<episode id>``) never pair.

``certified`` (:class:`PlanSpec`, ``docs/wave-h-runbook.md`` "The certified run") makes
``balanced`` certified for ``human``: b (exact oracle on the 12 golden/v2 episodes, default
mode, H200) -> f (noise: 6 of them, the selection's 5 plus one spare, the oracle's own config
and data dir re-indexed with ``flora index --limit 6``, so the same stage hash) -> d (balanced
on the 12, the real ``flora deploy`` / ``flora run``, then the **cold witness**: the same config
re-run on the noise subset with ``FLORA_COMPILE_CACHE=off``, one container per kind forced
cold) -> d2 (the lab, skipped only when its evidence exists **for the current engine
settings**: :func:`lab_already_done`) -> e (``flora certify`` per stage, with the witness,
then ``wave_h_finalize.py --write`` when both are ``certified``: :func:`run_flora_certify`).
Every job stages one data dir (:data:`CERT_DATA`). Its caps are each step's upper projection
rounded up to $0.05 (:func:`plan_caps`), its global cap their sum rounded up to $0.10; its
state, work dir and records live under ``reports/wave-h/certified/``.

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
   segments), and a run that gets no request in is an error, not evidence. Each call compiles
   two engines (baseline + plugin) and writes ``flora-cache:/balanced/<engine>[-dtbias]``,
   committed after each engine, which later runs (lab and deployed workers) start from;
e. ``certify``: offline provisional certification of ``balanced`` from (b) + (d) ($0);
f. ``noise``: (b)'s job **repeated** (same config file, staged data, stage hash, GPU and
   default mode; a new job id): the noise floor; (e) is re-evaluated. Its episodes are a subset
   of the oracle's (Fix-Cert H2) by being all of them: the input content is hashed, so a strict
   subset would be another stage hash, which ``flora certify`` refuses.

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
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
    # The certified run (module docstring): b -> f -> d -> (d2, skipped when done) -> e.
    "certified": ("exact_e2e", "noise", "balanced", "lab", "certify"),
}
DEFAULT_PLAN = "balanced-only"
DEFERRED_NOTE = "deferred to the certified run"
CERTIFIED = "certified"
# --- The certified plan (golden/v2, docs/golden-manifest-v2.md) -------------------------------
CERT_ROOT = ROOT / "reports" / "wave-h" / "certified"  # its own state, work dir and records
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

    @property
    def scale(self) -> dict[str, float]:
        return dict(self.rate_scale)

    def local_id(self, gid: str) -> str:
        """The episode id a job sees (the ``mp4_dir`` file stem). Certified: the noise episodes
        get :data:`NOISE_PREFIX`, the rest :data:`REST_PREFIX`, so ``flora index --limit 5``
        on the oracle's own config and data dir indexes exactly the noise subset."""
        if not self.shared_data:
            return h0.episode_id(gid)
        return (NOISE_PREFIX if gid in self.noise else REST_PREFIX) + h0.episode_id(gid)

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
    "yam": PlanSpec(CERTIFIED, CERT_YAM_GOLDEN, CERT_YAM_EPISODES, CERT_YAM_NOISE,
                    CERT_YAM_ROOT / "state.json", CERT_YAM_ROOT / "work",
                    CERT_YAM_ROOT / "records", CERT_YAM_ROOT / "runbook.md", shared_data=True,
                    embodiment="yam", data="cert-yam",
                    rate_scale=_YAM_SCALE,
                    lab_eps=CERT_YAM_EPISODES[:6]),
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
CAPS_USD = {"tempseg": 0.75, "preflight": 0.05, "exact_e2e": 3.30, "balanced": 2.30,
            "lab": 2.35, "certify": 0.0, "noise": 3.30}
GLOBAL_CAP_USD = 6.50  # --plan full; balanced-only derives its own (plan_global_cap)
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
# (d2) lab call timing, per kind. Each call builds two engines one after the other (baseline,
# then the FP32 ``dt_bias`` plugin, each with its own compile cache), so it pays two compiles.
# Cold builds (2026-10-22 run): TempSeg 270 s baseline + 187 s plugin (lab, L40S); Captioner
# 325 s (the deployed caption worker's cold startup on H100; the lab's own build was cut off by
# the old 600 s deadline), assumed for the plugin build too. Work: the stage over the 6
# episodes (the deployed job's busy 65 s / 161 s), L0 replay and two Gate A generations.
LAB_BUILD_S = {"tempseg": (270.0, 190.0), "captioner": (325.0, 325.0)}  # (baseline, plugin)
LAB_WORK_S = {"tempseg": 125.0, "captioner": 225.0}
LAB_OVERHEAD_S = 30.0  # container start, weight prefetch, child spawn
LAB_UNCERTAINTY = 0.25
LAB_SLACK_S = 60  # lab: client timeout -> cancel -> container gone
LAB_TIMEOUT_S = 20 * 60  # modal_app/balanced_lab.py TIMEOUT_S: the container's hard limit


def lab_seconds(kind: str) -> tuple[float, float]:
    """(expected, client deadline) seconds of one lab call with both compiles cold."""
    expected = LAB_OVERHEAD_S + sum(LAB_BUILD_S[kind]) + LAB_WORK_S[kind]
    return expected, min(float(LAB_TIMEOUT_S), expected * (1 + LAB_UNCERTAINTY))


LAB_EXPECTED_S = {k: lab_seconds(k)[0] for k in LAB_BUILD_S}
LAB_DEADLINE_S = {k: lab_seconds(k)[1] for k in LAB_BUILD_S}
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

    def add(self, what: str, c: Container, seconds: float, *, upper_s: float | None = None
            ) -> None:
        self.items.append({"what": what, "gpu": c.gpu or "cpu", "seconds": round(seconds, 1),
                           "upper_seconds": round(upper_s or seconds, 1),
                           "usd_per_h": round(c.usd_per_h(), 3),
                           "usd": round(c.usd_per_s() * seconds, 4),
                           "usd_upper": round(c.usd_per_s() * (upper_s or seconds), 4)})

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
              "compile)", seg, secs["segment"][0], upper_s=secs["segment"][1])
        p.add(f"balanced caption {job} ({len(eps)} ep, {v:.1f} s{work('caption')}, cold "
              "compile)", cap, secs["caption"][0], upper_s=secs["caption"][1])
        if spec.shared_data:  # certified: the forced-cold job on the noise subset
            neps = [i for i in spec.noise if i in dur]
            nv = sum(dur[i] for i in neps)
            ws = balanced_job_seconds(nv, scale=sc or None)
            for kind, c in (("segment", seg), ("caption", cap)):
                p.add(f"cold witness {kind} {job} ({len(neps)} ep, {nv:.1f} s{work(kind)}, "
                      "forced cold)", c, ws[kind][0], upper_s=ws[kind][1])
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
            full += LAB[kind].usd_per_s() * (LAB_DEADLINE_S[kind] + LAB_SLACK_S)
            if kind in reuse:
                p.add(f"balanced lab {kind}: saved result reused (has evidence), not run",
                      LAB[kind], 0.0, upper_s=0.0)
                continue
            p.add(f"balanced lab {kind} (L0 + Gate A, 2 cold compiles; after (d)'s job)",
                  LAB[kind], LAB_EXPECTED_S[kind], upper_s=LAB_DEADLINE_S[kind] + LAB_SLACK_S)
        p.peak_usd_per_s = max((LAB[k].usd_per_s() for k in LAB if k not in reuse),
                               default=0.0)  # one after the other
        if reuse:
            p.cap_usd_upper = round(full, 4)
    elif step == "noise":
        neps = [i for i in spec.noise if i in dur]
        nv = sum(dur[i] for i in neps)
        nominal, upper = exact_job_seconds(nv * sc["segcap"] if "segcap" in sc else nv)
        p.add(f"segcap {job} repeat ({len(neps)} ep, {nv:.1f} s{work('segcap')}, cold start)",
              pool_container("segcap"), nominal, upper_s=upper)
        p.peak_usd_per_s = pool_container("segcap").usd_per_s()
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
    before a step is its upper price and nothing more. Other plans: :data:`CAPS_USD`."""
    if plan != CERTIFIED:
        return dict(CAPS_USD)
    honest = honest_caps({s: projections[s] for s in PLANS[plan]})
    return {**CAPS_USD, **{s: honest[s] for s in PLANS[plan]}}


def plan_projections(plan: str, manifest: Mapping[str, Any], *, lab_done: bool = False,
                     reuse: Iterable[str] = (), spec: PlanSpec | None = None,
                     witness_only: bool = False) -> dict[str, Projection]:
    """Every step's projection on the plan's episodes (``lab_done`` only applies to
    ``certified``: the other plans run their lab; ``reuse``: :func:`lab_reuse`; ``spec``:
    default the plan's, :func:`spec_for` for another embodiment; ``witness_only``:
    :func:`witness_pending`, (d) priced at its cold witness alone)."""
    spec = spec or SPECS[plan]
    return {s: project(s, manifest, spec, lab_done=lab_done and plan == CERTIFIED,
                       reuse=reuse if s == "lab" else (),
                       witness_only=witness_only and plan == CERTIFIED and s == "balanced")
            for s in STEPS}


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


def lab_already_done(lab_dir: Path | None = None) -> tuple[bool, str]:
    """Whether the lab's L0 + Gate A evidence exists (``balanced-<kind>.json``, as step d2 or
    ``--only lab`` writes it): per kind non-empty L0 rows, every one with equal ids and grids,
    Gate A over at least one request, measured on the balanced pool's GPU **under the current
    engine settings** (:func:`lab_hash_problem`; a lab from before batch fill, or from before
    the hash was recorded, is not done). The certificate takes L0 per model kind, not per
    episode (``validate.certify.l0_summary``), so the certified run reuses it instead of
    paying two cold compiles per kind again."""
    from flora_deploy.engines.offline_llm import EngineArgsError, normalize_gpu

    lab_dir = lab_dir or LAB_OUT
    pool = {"tempseg": "segment", "captioner": "caption"}
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
                  f"{lab_dir}/balanced-{{tempseg,captioner}}.json")


# ---------------------------------------------------------------------------------------------
# State and stop rules


STEPS = ("tempseg", "preflight", "exact_e2e", "balanced", "lab", "certify", "noise")
TITLES = {"tempseg": "0. TempSeg merger v2 build + equivalence (CPU + L40S), then pin",
          "preflight": "a. balanced preflight (CPU)",
          "exact_e2e": "b. exact oracle, 6 human episodes, default mode, H200",
          "balanced": "d. balanced via flora deploy/run (L40S + H100)",
          "lab": "d2. balanced lab L0 / Gate A on (d)'s job (L40S, then H100)",
          "certify": "e. offline provisional certification",
          "noise": "f. noise: oracle job repeated, default mode, H200"}
def cert_titles(spec: PlanSpec) -> dict[str, str]:
    return {**TITLES,
            "exact_e2e": f"b. exact oracle, {len(spec.episodes)} {spec.embodiment} episodes "
                         f"({spec.golden}), nvidia mode, H200",
            "noise": f"f. noise: {len(spec.noise)} of (b)'s episodes, same config/GPU/mode",
            "balanced": f"d. balanced via flora deploy/run, {len(spec.episodes)} episodes",
            "lab": "d2. lab L0 / Gate A (skipped when already done)",
            "certify": ("e. flora certify, then wave_h_finalize.py" if spec.embodiment == "human"
                        else "e. flora certify (wave_h_finalize.py not run: human only)")}


CERT_TITLES = cert_titles(SPECS[CERTIFIED])


def titles(plan: str, spec: PlanSpec | None = None) -> dict[str, str]:
    if plan != CERTIFIED:
        return TITLES
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
                     log=self.work / "wave-h.log")

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
        return app_name(self.st.run_id, self.step, n) + (f"-{part}" if part else "")

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
        ids = rb.app_ids_named(name, rb.list_apps(run) or [])
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


def _stage_job(ctx: Ctx, job: str, ids: Sequence[str], *, embodiment: str,
               stages: tuple[str, ...], budget: float, preset: str = "exact",
               resources: dict[str, str] | None = None,
               derived: dict[str, str] | None = None, data: str | None = None
               ) -> tuple[Path, str]:
    """Fetch, upload, write the config, ``flora index --local`` + mirror, ``flora plan``.
    ``data``: the data directory (default ``job``); the certified plan stages every job from
    one (:data:`CERT_DATA`), so their episode uids pair."""
    run = ctx.st.run_id
    data = data or job
    stage_local(ctx, data, ids)
    h0.upload_dir(MEDIA_VOLUME, ctx.work / data, f"{run}/{data}")
    job_id = f"wave-h-{run}-{job}"
    cfg = ctx.work / f"flora-{job}.yaml"
    cfg.write_text(h0.config_text(project=job_id, data_dir=data, embodiment=embodiment,
                                  stages=stages, store_volume=STORE_VOLUME,
                                  media_volume=MEDIA_VOLUME, run_id=run, budget_usd=budget,
                                  resources=resources, tag="flora-wave-h", preset=preset,
                                  derived=derived,
                                  header="Wave H-lite; generated by scripts/wave_h.py"))
    _index(ctx, cfg, job_id, job)
    return cfg, job_id


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
            ctx.sh([fl, "cancel", job_id, "--config", str(cfg)], check=False, timeout=120)
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


def run_exact(ctx: Ctx, step: str = "exact_e2e") -> dict[str, Any]:
    """(b): the oracle, default mode, one app."""
    require_tempseg_pin(ctx)  # every GPU step runs from the pinned commit
    derived = resolve_receipts(["captioner_bf16"])
    eps = ctx.episodes
    if ctx.spec.shared_data:  # certified: every episode, at least the certificate's minimum
        eps = balanced_episodes(ctx.manifest, ctx.spec, side="(b)")
    cfg, job_id = _stage_job(ctx, f"{step}-{ctx.spec.embodiment}", eps,
                             embodiment=ctx.spec.embodiment,
                             stages=("segment", "caption"), budget=ctx.caps[step],
                             derived=derived, data=ctx.spec.data_dir(step))
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
        if len(spec.noise) < MIN_NOISE_EPISODES or not set(spec.noise) <= set(spec.episodes):
            raise RuntimeError(f"{side}: the noise subset must be at least "
                               f"{MIN_NOISE_EPISODES} of the oracle's episodes")
    return spec.episodes


def run_balanced(ctx: Ctx) -> dict[str, Any]:
    """(d): balanced through the real ``flora deploy`` / ``flora run`` on the 6 golden episodes
    (:func:`balanced_episodes`; independent of (b)). The lab's L0 + Gate A on that job is its
    own step (:func:`run_lab`, ``--only lab``), so a lab retry never redoes this job."""
    from flora_deploy.config.receipts import derived_content as content_of
    from flora_deploy.config.schema import load_config

    require_tempseg_pin(ctx)
    rate = (pool_container("segment", "balanced").usd_per_s()
            + pool_container("caption", "balanced").usd_per_s())
    if ctx.spec.shared_data and witness_pending(ctx.st):
        return rerun_cold_witness(ctx, rate)
    episodes = balanced_episodes(ctx.manifest, ctx.spec)
    derived = resolve_receipts(["captioner_bf16", "tempseg_merged"], require_pin=True)
    cfg, job_id = _stage_job(ctx, f"balanced-{ctx.spec.embodiment}", episodes,
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


def witness_config_text(text: str, project: str) -> str:
    """(d)'s config text with its ``project`` replaced: the cold witness's own job id and app
    name. The project name is not hashed (``config.hashing``), so the stage hash stays (d)'s."""
    lines = text.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.startswith("project:")]
    if len(hits) != 1:
        raise RuntimeError(f"cold witness: {len(hits)} top-level project lines in (d)'s config")
    lines[hits[0]] = f"project: {project}\n"
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
    job_id = f"wave-h-{ctx.st.run_id}-{COLD_JOB}-{emb}-a{n}"
    tag = f"{COLD_JOB}-{emb}"
    wcfg = ctx.work / f"flora-{tag}.yaml"
    wcfg.write_text(witness_config_text(cfg.read_text(), job_id))
    _index(ctx, wcfg, job_id, tag, limit=len(ctx.spec.noise))
    got = noise_index_problems(ctx.work / f"summary-{tag}.json", ctx.manifest, ctx.spec)
    if got:
        return {"job_id": job_id, "state": "skipped", "error": f"witness index: {got}",
                "spend": 0.0, "compile_cache": "off"}
    prev = os.environ.get(COMPILE_CACHE_ENV)
    os.environ[COMPILE_CACHE_ENV] = "off"
    try:
        r = _run_job(ctx, "balanced", wcfg, job_id, determinism=ORACLE_DETERMINISM,
                     usd_per_s=rate, spent_in_step=spent, side="witness")
    finally:
        if prev is None:
            os.environ.pop(COMPILE_CACHE_ENV, None)
        else:
            os.environ[COMPILE_CACHE_ENV] = prev
    return {**r, "compile_cache": "off", "episodes": len(ctx.spec.noise)}


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
    if not balanced_job_done(st) or (st.plan != CERTIFIED and "witness_jobs" not in res):
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
            lab_module: Any = None) -> dict[str, Any]:
    """(d2): the lab's L0 + Gate A against (d)'s finished deployed job: (d)'s episodes, and
    (d)'s segment records for the Captioner (``--only lab`` retries it alone). Every
    precondition is checked before any container (:func:`lab_preconditions`); a failed one
    charges $0. TempSeg runs first; when it gets no request in, the Captioner is not started
    (it reads the same inputs). Each call's deadline is :data:`LAB_DEADLINE_S` (two cold
    compiles), shortened by the budget left; a call the budget cannot carry to its expected
    length is not started. Spend is each started call's wall time x list $/s."""
    if ctx.spec.name == CERTIFIED:
        done, why_done = lab_already_done()
        if done:  # the certified run's "lab already done" skip: nothing starts, $0
            return {"ok": True, "skipped": f"lab already done: {why_done}", "kinds": {},
                    "tracked_usd": 0.0, "apps": []}
    if lab_module is None:
        from flora_deploy.modal_app import balanced_lab as lab_module
    lab = lab_module
    bal = ctx.st.step("balanced").result
    if not balanced_job_done(ctx.st):
        raise RuntimeError("the lab runs against (d)'s finished deployed job; (d) has none "
                           "(run --only balanced first)")
    if witness_pending(ctx.st):
        raise RuntimeError("(d)'s cold witness is not done; the lab waits for it (run --only "
                           "balanced: it re-runs the witness alone)")
    job = bal["jobs"][JOB]
    episodes = [e for e in bal.get("episodes") or () if e in ctx.spec.lab_episodes]
    bound: dict[str, str] | None = None
    try:
        bound = (binding or segment_binding)(ctx, job)
        recs = list((records or _segment_records)(ctx, job))
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

    for kind in LAB:
        saved = reusable_lab_result(kind, ctx.st.run_id, episodes, segments_from=bound)
        if saved is not None:
            out["kinds"][kind] = {**_lab_kind_summary(saved, 0.0, 0.0), "reused": True,
                                  "requests": ((saved.get("baseline") or {}).get("batcher")
                                               or {}).get("requests")}
    todo = [k for k in LAB if k not in out["kinds"]]
    if not todo:
        out["note"] = "every kind reused a saved result; nothing started"
        return {"ok": True, **out, "tracked_usd": 0.0, "apps": []}
    persist()
    import modal

    rdir = LAB_OUT
    rdir.mkdir(parents=True, exist_ok=True)
    fns = {"tempseg": lab.run_tempseg, "captioner": lab.run_captioner}
    first = todo[0]  # a reused TempSeg result always had requests in
    limit = min(LAB_DEADLINE_S[first],
                ctx.deadline_s("lab", LAB[first], slack_s=LAB_SLACK_S))
    if limit < LAB_EXPECTED_S[first]:  # the stop rule, before the app starts: charged $0
        b = ctx.budget_of("lab")
        raise StopRule(f"lab: budget left ${b.usd:.2f} allows {limit:.0f} s of the {first} "
                       f"call, under the {LAB_EXPECTED_S[first]:.0f} s it needs (binding: "
                       f"{b.binding}); no lab app started")
    # the app is noted only now: a precondition failure above charged $0
    with _lab_app(ctx, lab, modal, name):
        for kind in todo:
            fn, c = fns[kind], LAB[kind]
            limit = min(LAB_DEADLINE_S[kind],
                        ctx.deadline_s("lab", c, spent_in_step=spent, slack_s=LAB_SLACK_S))
            prev = out["kinds"].get("tempseg") or {}
            if kind == "captioner" and prev.get("requests") == 0:
                out["kinds"][kind] = {"skipped": "TempSeg sent zero requests on the same "
                                      "inputs; not started", "usd": 0.0, "evidence": False,
                                      "error": NO_EVIDENCE}
                continue
            if limit < LAB_EXPECTED_S[kind]:
                out["kinds"][kind] = {"skipped": f"budget left {limit:.0f} s < the "
                                      f"{LAB_EXPECTED_S[kind]:.0f} s it needs", "usd": 0.0,
                                      "evidence": False, "error": NO_EVIDENCE}
                continue
            t0 = time.time()
            cap = kind == "captioner"
            call = fn.spawn(f"wave-h-{ctx.st.run_id}", "", episodes, True,
                            segments if cap else None, bound if cap else None)
            try:
                res = call.get(timeout=limit)
            except (TimeoutError, modal.exception.TimeoutError):
                call.cancel()
                res = {"error": f"client deadline {limit:.0f} s"}
            except Exception as e:  # noqa: BLE001 - the remote call raised: this kind failed
                res = {"error": f"lab call raised {e!r}"[:2000]}
            wall = time.time() - t0
            usd = c.usd_per_s() * wall
            spent += usd
            (rdir / f"balanced-{kind}.json").write_text(json.dumps(res, default=str))
            out["kinds"][kind] = _lab_kind_summary(res, wall, usd)
            stats = (res.get("baseline") or {}).get("batcher") or {}
            out["kinds"][kind]["requests"] = stats.get("requests")
            persist()
    bad = {k: v["error"] for k, v in out["kinds"].items() if v.get("error")}
    bad.update({k: f"{v['episode_errors']} episode(s) failed" for k, v in out["kinds"].items()
                if v.get("episode_errors") and k not in bad})
    if bad:
        out["error"] = "; ".join(f"{k}: {v}" for k, v in bad.items())
    return {"ok": not bad and len(out["kinds"]) == 2, **out, "tracked_usd": spent,
            "apps": [name]}


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


def run_noise(ctx: Ctx) -> dict[str, Any]:
    """(f): (b)'s job repeated: its own config file and staged data (same stage hash, so the
    same app), a new job id, default mode; all of its episodes (module docstring)."""
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
    if ctx.spec.name == CERTIFIED:  # (e) runs flora certify on the exported records
        return res
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


def noise_index_problems(summary: Path, manifest: Mapping[str, Any], spec: PlanSpec) -> str:
    """Empty when the noise job's index (``flora index --limit``) holds exactly the plan's
    noise subset: its episode count and its video seconds (within 1%) equal theirs."""
    try:
        d = json.loads(summary.read_text())
    except (OSError, ValueError) as e:
        return f"unreadable {summary.name}: {e}"
    n = (d.get("counts") or {}).get("episodes")
    dur = durations(manifest)
    want = sum(dur.get(g, 0.0) for g in spec.noise)
    got = float(d.get("video_s") or 0.0)
    if n != len(spec.noise) or abs(got - want) > 0.01 * max(want, 1.0):
        return (f"{n} episodes / {got:.1f} video s indexed, not the {len(spec.noise)} noise "
                f"episodes / {want:.1f} s")
    return ""


CERT_STAGES = ("segment", "caption")


def run_flora_certify(ctx: Ctx, *, finalize: Any = None, sh: Callable[..., Any] | None = None,
                      lab_dir: Path | None = None) -> dict[str, Any]:
    """(e) of the certified plan, on CPU ($0): export the oracle's, the noise run's and
    balanced's records (``wave_h_finalize.fetch_records``, a Volume read), then ``flora
    certify`` per stage (``wave_h_finalize.certify_all``: the oracle, the noise run, the L0 JSON
    and the derived content digests (d)'s workers verified, stamped with the run's own time).
    Done only when **every** stage is ``certified``; then ``scripts/wave_h_finalize.py --write``
    (rates, certificates, report, docs) for this state. Otherwise nothing is written and the
    step fails with what is missing (``missing_for_certified``)."""
    if finalize is None:
        import wave_h_finalize as finalize
    paths = finalize.Paths(root=ROOT, state=ctx.state_path, records=ctx.spec.records,
                           lab=lab_dir or LAB_OUT)
    save_state(ctx.st, ctx.state_path)
    finalize.fetch_records(paths, ctx.st)  # every job's records, freshly (no container)
    out_dir = ctx.work / "certify"
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = finalize.certify_all(paths, ctx.st, out_dir,
                                    derived=finalize.derived_content_used(ctx.st))
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
    ok = (set(stages) == set(CERT_STAGES) and not no_witness
          and all(v["verdict"] == "certified" for v in stages.values()))
    res: dict[str, Any] = {"ok": ok, "stages": stages, "tracked_usd": 0.0, "apps": []}
    if not ok:
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
    p = (sh or ctx.sh)(args, timeout=1800, check=False)
    res["finalize_rc"] = getattr(p, "returncode", None)
    res["finalize"] = " ".join(args[1:])
    return res


RUNNERS: dict[str, Callable[[Ctx], dict[str, Any]]] = {
    "tempseg": run_tempseg, "preflight": run_preflight, "exact_e2e": run_exact,
    "balanced": run_balanced, "lab": run_lab, "certify": run_certify, "noise": run_noise}


def runner(plan: str, step: str) -> Callable[[Ctx], dict[str, Any]]:
    """The certified plan's (e) is ``flora certify`` + finalize; the rest are shared."""
    if plan == CERTIFIED and step == "certify":
        return run_flora_certify
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
    for name in STEPS:
        s = st.steps.get(name, StepState())
        if name not in PLANS[plan] and s.status == "pending":
            if plan != CERTIFIED:
                lines.append(f"| {title[name]} | {DEFERRED_NOTE} | - | - | 0 | - | - |")
            continue
        p = projections.get(name)
        proj = f"${p.usd:.2f} (${p.usd_upper:.2f})" if p else "-"
        billed = f"${s.billed_usd:.2f}" if s.billed_usd is not None else "-"
        lines.append(f"| {title[name]} | {s.status} | ${caps[name]:.2f} | {proj} | "
                     f"{len(s.attempts)} | ${s.tracked_usd:.2f} | {billed} |")
    for name in STEPS:
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


def print_plan(st: State, projections: Mapping[str, Projection], out=print, *,
               manifest: Mapping[str, Any] | None = None, plan: str = DEFAULT_PLAN,
               only: str | None = None, spec: PlanSpec | None = None) -> None:
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
        if plan == CERTIFIED:
            nv = sum(dur.get(e, 0.0) for e in spec.noise)
            out(f"  noise subset: {len(spec.noise)} of them, {nv:.1f} video s (`flora index "
                f"--limit {spec.noise_limit}` on the oracle's config and data dir: the same "
                "stage hash)")
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
    for name in () if only or plan == CERTIFIED else deferred_steps(plan):
        out(f"  {title[name]:66s} [{DEFERRED_NOTE}; never runs in plan {plan}]")
    if plan == CERTIFIED and "exact_e2e" in steps:
        from flora_deploy.planner.rates import RateTable

        t = RateTable.load()
        e, su = t.rate("exact", "segcap", 1, 1), t.startup_for("exact", "segcap")
        out(f"  assumption: (b) and (f) are priced on the measured exact segcap rate "
            f"{e.container_s_per_video_s:g} container-s/video-s +/-{e.uncertainty:.0%} (H200, "
            f"default mode, c={e.concurrency}, {e.date}) and the measured @enter "
            f"{su.startup_s:g} s +/-{su.uncertainty:.0%} per job (each job is its own cold "
            "start: the oracle's app is stopped before (f) deploys it again). Image pull and "
            "scheduling are not in the measured startup; the upper bound's +100% and the "
            f"{CLIENT_SLACK_S} s client slack cover them")
        out("  assumption: certified plan caps are each step's own upper projection rounded "
            "up to $0.05 (plan_caps); the global cap is their sum rounded up to $0.10")
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
        if plan == CERTIFIED:
            out("  assumption: (d) then runs the cold witness, the same config on the "
                f"{len(spec.noise)} noise episodes with {COMPILE_CACHE_ENV}=off (one container "
                "per kind forced cold, priced cold): the certificate's evidence covers warm "
                "(the cache (d)'s first container publishes) and cold compiles")
        lim = balanced_engine_limits()
        out("  assumption: (d) runs the batch-fill engine limits and batch policy: "
            + ", ".join(f"{k} on {v['gpu']} max_num_seqs {v['max_num_seqs']}, "
                        f"max_num_batched_tokens {v['max_num_batched_tokens']}, "
                        f"{v['infer_episodes']} episodes per batcher" for k, v in lim.items())
            + "; its rates were measured before batch fill, so they can only overstate it")
    if "lab" in steps:
        out("  assumption: each lab call compiles two engines cold (baseline + dt_bias "
            "plugin): deadline " + ", ".join(f"{k} {LAB_DEADLINE_S[k]:.0f} s"
                                             for k in LAB_DEADLINE_S)
            + f" (expected x {1 + LAB_UNCERTAINTY:.2f}, <= the {LAB_TIMEOUT_S} s container "
            "timeout); the lab commits flora-cache:/balanced/<engine>[-dtbias] after each "
            "engine, so a warm cache only makes it cheaper")
    if plan == CERTIFIED and "lab" in steps:
        done, why = lab_already_done()
        out(f"  lab (d2): {'already done, skipped at $0' if done else 'runs'} ({why})")
    nominal = sum(projections[s].usd for s in steps)
    upper = sum(projections[s].usd_upper for s in steps)
    how = global_cap_label(st, plan, projections)
    out(f"  total: nominal ${nominal:.2f}, upper ${upper:.2f} (incl. {CLIENT_SLACK_S} s client "
        f"slack per app); global cap ${gcap:.2f} ({how})")
    honest = honest_caps({s: projections[s] for s in steps})
    if upper > gcap + 1e-9:
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
STEP_HARD_TIMEOUT_S = {"lab": 2 * LAB_TIMEOUT_S}  # two calls, one after the other


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
    """The decision a run takes first: the first step (of ``--only`` or the plan) not done."""
    for name in (only,) if only else PLANS[plan]:
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
            why += (f"  [witness pending: will re-run the {len(spec.noise)}-episode cold job "
                    "only]")
        out(f"  {title[name]:66s} {s.status:9s} cap ${caps[name]:.2f}  attempts "
            f"{len(s.attempts)}  charged ${s.charged(wm):.2f}  tracked ${s.tracked_usd:.2f}"
            + ("" if s.billed_usd is None else f"  billed ${s.billed_usd:.2f}") + why)
    for line in held_lines(st):
        out(line)
    d = next_decision(st, plan, projections, only, spec.embodiment)
    if d is None:
        out("  next: nothing (every step is done)")
        return None
    out(f"  next: {d.step}: {d.action}: {d.message}")
    for r in d.reasons:
        out(f"    because {r}")
    if d.command:
        out(f"    command: {d.command}")
    return d


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
                         "certified: b, f, d, (d2), e on golden/v2 (its own state)")
    ap.add_argument("--embodiment", choices=ALL_EMBODIMENTS, default=EMBODIMENT,
                    help="certified only: the embodiment to certify (human: golden/v2; yam: "
                         "golden/v3-yam; each its own state). dex, tianji, yam32: no golden "
                         "certification set yet")
    ap.add_argument("--only", choices=STEPS, help="run just this step (if not done)")
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
    args = ap.parse_args(argv)
    try:
        spec = spec_for(args.plan, args.embodiment)
    except KeyError:
        ap.error(f"--embodiment {args.embodiment}: " + (
            f"no golden certification set (certified: {', '.join(CERT_SPECS)})"
            if args.plan == CERTIFIED else "only --plan certified takes another embodiment"))
    other = {s.state.resolve() for e, s in CERT_SPECS.items() if e != args.embodiment}
    if args.state is not None and args.plan == CERTIFIED and args.state.resolve() in other:
        ap.error(f"--state {args.state} is another embodiment's certified state")
    args.state = args.state or spec.state
    args.work = args.work or spec.work
    if args.offline and args.yes_spend:
        ap.error("--offline is for --dry-run / --report (a run stages the real files)")
    if args.only and args.only not in PLANS[args.plan]:
        ap.error(f"--only {args.only}: {DEFERRED_NOTE} (not in plan {args.plan}; use --plan "
                 "full)")

    st = load_state(args.state)
    if st.plan is not None and st.plan != args.plan and CERTIFIED in (st.plan, args.plan):
        print(f"refused: {args.state} was run under plan {st.plan}, not {args.plan}")
        return 2
    preview = args.dry_run or args.status or not (args.yes_spend or args.report)
    if preview:  # the dry run and --status decide on a copy: the state file is never written
        st = copy.deepcopy(st)
    if args.plan == CERTIFIED:
        st.plan = CERTIFIED
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
    lab_done = args.plan == CERTIFIED and lab_already_done()[0]
    reuse = () if lab_done else lab_reuse(st, spec)
    pending = args.plan == CERTIFIED and witness_pending(st)
    projections = plan_projections(args.plan, manifest, lab_done=lab_done, reuse=reuse,
                                   spec=spec, witness_only=pending)
    caps = plan_caps(args.plan, projections)
    if pending:
        print(f"note: (d)'s deployed job is done; witness pending: will re-run the "
              f"{len(spec.noise)}-episode cold job only")
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
    print_plan(st, projections, manifest=manifest, plan=args.plan, only=args.only, spec=spec)
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
        after=after)
    save_state(st, args.state)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(st, projections, plan=args.plan, spec=spec))
    print(f"wrote {report}; state {args.state}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
