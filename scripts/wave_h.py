"""Wave H-lite runbook, one embodiment (``human``) plus the lab checks (``docs/wave-h-runbook.md``).

Usage::

    uv run python scripts/wave_h.py --dry-run              # plan + estimates; spends nothing
    uv run python scripts/wave_h.py --dry-run --offline    # ... without reading the Volume
    uv run python scripts/wave_h.py --yes-spend            # run (resumes: skips done steps)
    uv run python scripts/wave_h.py --yes-spend --only balanced
    uv run python scripts/wave_h.py --report               # rewrite reports/wave-h/runbook.md
    uv run python scripts/wave_h.py --plan full --dry-run  # every v1 step (cannot certify)
    uv run python scripts/wave_h.py --plan certified --dry-run  # the certified run (golden/v2)

**Plans** (``--plan``, :data:`PLANS`). ``balanced-only`` (the default; the user's decision of
2026-10-22) runs steps 0 -> a -> d and nothing else; (b), (e) and (f) are listed as "deferred to
the certified run" and never run. Its global cap is the sum of those three steps' upper
projections rounded up to $0.10 (:func:`plan_global_cap`). ``full`` runs every step under
:data:`GLOBAL_CAP_USD`. Both share the state file, so a later ``--plan full`` skips what
``balanced-only`` finished. ``full`` cannot certify: it stages one data dir per step, so its
oracle's and balanced's episode uids (``<data dir>--<episode id>``) never pair.

``certified`` (:class:`PlanSpec`, ``docs/wave-h-runbook.md`` "The certified run") makes
``balanced`` certified for ``human``: b (exact oracle on the 12 golden/v2 episodes, default
mode, H200) -> f (noise: 5 of them, the oracle's own config and data dir re-indexed with
``flora index --limit 5``, so the same stage hash) -> d (balanced on the 12, the real ``flora
deploy`` / ``flora run``) -> d2 (the lab, skipped when its evidence already exists:
:func:`lab_already_done`) -> e (``flora certify`` per stage, then ``wave_h_finalize.py
--write`` when both are ``certified``: :func:`run_flora_certify`). Every job stages one data
dir (:data:`CERT_DATA`). Its caps are each step's upper projection rounded up to $0.05
(:func:`plan_caps`), its global cap their sum rounded up to $0.10; its state, work dir and
records live under ``reports/wave-h/certified/``.

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
``elapsed x the step's peak $/s`` so a killed process leaves its spend behind. An attempt that was
killed, or whose ``flora report`` failed, is charged the most it was allowed to spend (its
budget) until billing, read again before every step for **every** app seen so far, is at least
:data:`BILLING_LAG_S` past it. An attempt that failed **before it noted any app** started
nothing and is charged $0 (review 10 L1). The global figure is ``max(sum of charged attempts,
billed delta of all apps)``; the global cap is checked on it.

**Stop rules.** Before a step: its **upper** projection (upper rate quantiles, every cold start,
the scale-down tail and :data:`CLIENT_SLACK_S` per app) must fit its cap and the global cap minus
the cumulative spend. During a step: GPU wall time is bounded by ``(remaining step cap - slack x
$/s) / $/s`` (``flora run --budget`` plus a client-side deadline that cancels the job and stops
the app; lab and conversion calls get the same deadline as their timeout). After a step:
tracked/billed spend over the step cap or the global cap stops the runbook. The dry run prints
nominal and upper costs per step.

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
import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h0_rehearsal as h0  # noqa: E402 - the same staging and configs as H0

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


def golden_selection(version: str) -> dict[str, Any]:
    """``validate/golden_selection_<v>.json`` (package data)."""
    from flora_deploy.validate.golden import SELECTIONS, load_selection

    return load_selection(SELECTIONS[version])


def _cert_sets() -> tuple[tuple[str, ...], tuple[str, ...]]:
    sel = golden_selection(CERT_GOLDEN)
    eps = tuple(f"{e['pool']}/{e['dataset']}/{e['slug']}" for e in sel["episodes"]
                if e["pool"] == EMBODIMENT)
    return eps, tuple(sel.get("noise_subset") or ())


# All 12 golden/v2 human episodes (10 for the certificate + 2 spares, all run), and the 5 the
# noise job repeats (the selection's ``noise_subset``: one VFR HomER row and four EgoDex clips).
CERT_EPISODES, CERT_NOISE = _cert_sets()


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

    def local_id(self, gid: str) -> str:
        """The episode id a job sees (the ``mp4_dir`` file stem). Certified: the noise episodes
        get :data:`NOISE_PREFIX`, the rest :data:`REST_PREFIX`, so ``flora index --limit 5``
        on the oracle's own config and data dir indexes exactly the noise subset."""
        if not self.shared_data:
            return h0.episode_id(gid)
        return (NOISE_PREFIX if gid in self.noise else REST_PREFIX) + h0.episode_id(gid)

    def data_dir(self, step: str) -> str:
        return CERT_DATA if self.shared_data else f"{step}-{JOB}"

    @property
    def lab_episodes(self) -> tuple[str, ...]:
        """The episodes the lab (d2) captures its requests on. Certified: the golden/v1 six
        (all in golden/v2), so :data:`LAB_WORK_S` and the lab deadlines, measured on them,
        still hold; L0 is per model kind, not per episode."""
        return EPISODES if self.shared_data else self.episodes

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
# turned every global check into a no-op).
MAX_GLOBAL_CAP_USD = 50.0
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
CLIENT_SLACK_S = 240  # per app: `flora run` timeout margin, then cancel + `modal app stop`
BILLING_LAG_S = 3600  # Modal billing covers complete hours only
HEARTBEAT_S = 30.0
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


def warm_startup_s(st: Any) -> tuple[float, str] | None:
    """A startup row timed (or placeheld) with a loaded compile cache, as is; ``None`` when
    ``st`` is not a ``compile_cache: warm`` row (price it cold instead)."""
    if getattr(st, "compile_cache", None) != "warm":
        return None
    return float(st.startup_s), f"{st.status} warm"


def balanced_job_seconds(video_s: float, table: Any = None, *,
                         compile_factor: float = COLD_COMPILE_FACTOR,
                         warm_kinds: tuple[str, ...] = ()
                         ) -> dict[str, tuple[float, float]]:
    """One balanced job (one app): kind -> (nominal, upper) container-seconds. Each pool has
    its own cold start, its startup scaled by ``compile_factor`` (a cold compile cache:
    :data:`COLD_COMPILE_FACTOR`), except the pools in ``warm_kinds`` (a compile cache is
    published for their key), priced with the warm startup row when there is one; the segment
    container then idles through the scale-down window while captions run (the caption shards
    wait for it); both stay up through the client slack."""
    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S
    from flora_deploy.planner.rates import RateTable

    t = table or RateTable.load()
    out: dict[str, tuple[float, float]] = {}
    work: dict[str, tuple[float, float, float]] = {}
    for kind in ("segment", "caption"):
        e = t.rate("balanced", kind, 1, 1)
        st = t.startup_for("balanced", kind, warm=kind in warm_kinds)
        start, _basis = warm_startup_s(st) or cold_startup_s(st, compile_factor)
        work[kind] = (start, video_s * e.container_s_per_video_s,
                      max(e.uncertainty, st.uncertainty))
    cap_nominal = work["caption"][0] + work["caption"][1]
    for kind, (start, w, unc) in work.items():
        idle = min(WORKER_SCALEDOWN_S, cap_nominal) if kind == "segment" else 0.0
        idle_up = WORKER_SCALEDOWN_S if kind == "segment" else 0.0
        out[kind] = (start + w + idle + IDLE_AFTER_JOB_S,
                     (start + w) * (1 + unc) + idle_up + IDLE_AFTER_JOB_S + CLIENT_SLACK_S)
    return out


def project(step: str, manifest: Mapping[str, Any], spec: PlanSpec | None = None, *,
            reuse: Iterable[str] = (), lab_done: bool = False) -> Projection:
    """What ``step`` is expected to cost (nominal rates; ``usd_upper``: upper quantiles plus
    every cold start, scale-down tail and client slack the step can incur). ``spec``: the
    plan's episodes (default: the golden/v1 six). ``reuse``: lab kinds whose saved result is
    reused (:func:`lab_reuse`), priced at $0. ``lab_done`` (certified plan): the lab's L0 +
    Gate A evidence already exists (:func:`lab_already_done`), so the step starts nothing: $0
    (and the global cap leaves it out: the step never runs)."""
    spec = spec or SPECS[DEFAULT_PLAN]
    dur = durations(manifest)
    eps = episodes_in(manifest, spec.episodes)
    v = sum(dur[i] for i in eps)
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
        nominal, upper = exact_job_seconds(v)
        p.add(f"segcap {JOB} ({len(eps)} ep, {v:.1f} s, cold start)", pool_container("segcap"),
              nominal, upper_s=upper)
        p.peak_usd_per_s = pool_container("segcap").usd_per_s()
    elif step == "balanced":
        seg, cap = pool_container("segment", "balanced"), pool_container("caption", "balanced")
        secs = balanced_job_seconds(v)
        p.add(f"balanced segment {JOB} ({len(eps)} ep, {v:.1f} s, cold compile)", seg,
              secs["segment"][0], upper_s=secs["segment"][1])
        p.add(f"balanced caption {JOB} ({len(eps)} ep, {v:.1f} s, cold compile)", cap,
              secs["caption"][0], upper_s=secs["caption"][1])
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
        nominal, upper = exact_job_seconds(nv)
        p.add(f"segcap {NOISE_JOB} repeat ({len(neps)} ep, {nv:.1f} s, cold start)",
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
                     reuse: Iterable[str] = ()) -> dict[str, Projection]:
    """Every step's projection on the plan's episodes (``lab_done`` only applies to
    ``certified``: the other plans run their lab; ``reuse``: :func:`lab_reuse`)."""
    spec = SPECS[plan]
    return {s: project(s, manifest, spec, lab_done=lab_done and plan == CERTIFIED,
                       reuse=reuse if s == "lab" else ()) for s in STEPS}


def lab_already_done(lab_dir: Path | None = None) -> tuple[bool, str]:
    """Whether the lab's L0 + Gate A evidence exists (``balanced-<kind>.json``, as step d2 or
    ``--only lab`` writes it): per kind non-empty L0 rows, every one with equal ids and grids,
    Gate A over at least one request, measured on the balanced pool's GPU. The certificate
    takes L0 per model kind, not per episode (``validate.certify.l0_summary``), so the
    certified run reuses it instead of paying two cold compiles per kind again."""
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
        gpu = base.get("gpu") or res.get("gpu")
        want = pool_container(stage, "balanced").gpu or ""
        try:
            if gpu and normalize_gpu(str(gpu)) != normalize_gpu(want):
                why.append(f"{kind}: measured on {gpu}, not {want}")
        except EngineArgsError:
            why.append(f"{kind}: unknown GPU {gpu!r}")
    if why:
        return False, "; ".join(why)
    return True, f"L0 + Gate A evidence in {lab_dir}/balanced-{{tempseg,captioner}}.json"


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
CERT_TITLES = {**TITLES,
               "exact_e2e": f"b. exact oracle, {len(CERT_EPISODES)} human episodes (golden/v2), "
                            "nvidia mode, H200",
               "noise": f"f. noise: {len(CERT_NOISE)} of (b)'s episodes, same config/GPU/mode",
               "balanced": f"d. balanced via flora deploy/run, {len(CERT_EPISODES)} episodes",
               "lab": "d2. lab L0 / Gate A (skipped when already done)",
               "certify": "e. flora certify, then wave_h_finalize.py"}


def titles(plan: str) -> dict[str, str]:
    return CERT_TITLES if plan == CERTIFIED else TITLES


@dataclass
class Attempt:
    """One run of a step (module docstring, "Spend")."""

    started: float
    status: str = "running"  # running | done | failed | stopped | killed | needs_pin
    finished: float | None = None
    heartbeat: float | None = None
    rate_usd_per_s: float = 0.0  # the step's peak burn (heartbeat estimate)
    budget_usd: float = 0.0  # the most this attempt was allowed to spend
    tracked_usd: float = 0.0  # final: report / wall time; while running: the heartbeat estimate
    apps: list[str] = field(default_factory=list)
    report_error: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def last_activity(self) -> float:
        return self.finished or self.heartbeat or self.started

    def charged(self, billed_at: float | None) -> float:
        """Tracked cost, or the attempt's budget while it cannot be known (killed, no report)
        and billing has not caught up with it."""
        if self.status == "killed" or self.report_error:
            reconciled = billed_at is not None and billed_at - self.last_activity >= BILLING_LAG_S
            if not reconciled:
                return max(self.tracked_usd, self.budget_usd)
        return self.tracked_usd


@dataclass
class StepState:
    status: str = "pending"  # pending | running | done | failed | stopped | needs_pin
    projected_usd: float | None = None
    projected_upper_usd: float | None = None
    billed_usd: float | None = None  # billed delta of the apps only this step used
    attempts: list[Attempt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def tracked_usd(self) -> float:
        return round(sum(a.tracked_usd for a in self.attempts), 4)

    def charged(self, billed_at: float | None = None) -> float:
        return round(sum(a.charged(billed_at) for a in self.attempts), 4)

    def spent_at(self, billed_at: float | None = None) -> float:
        return max(self.charged(billed_at), self.billed_usd or 0.0)

    @property
    def spent(self) -> float:
        return self.spent_at(None)

    @property
    def current(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts and self.attempts[-1].status == "running" \
            else None

    @property
    def apps(self) -> list[str]:
        return sorted({a for x in self.attempts for a in x.apps})


@dataclass
class State:
    run_id: str
    created: float
    billing_since: str
    billing_baseline: dict[str, float] = field(default_factory=dict)
    steps: dict[str, StepState] = field(default_factory=dict)
    billed_total: float | None = None  # billed delta over every app seen so far
    billed_at: float | None = None  # when billing was last read
    cap_override: float | None = None  # a global cap the user approved (--global-cap)
    # the plan that cap was approved for: it never carries over to another plan. ``None`` in a
    # state from before the scope was saved: the default plan, the only one it was set on
    cap_override_plan: str | None = None
    # the global cap the last --yes-spend run gated its steps on, and its plan (what
    # wave_h_finalize.py reports; review 12 F6)
    gate_cap: float | None = None
    gate_cap_plan: str | None = None
    plan: str | None = None  # the plan this state was run under (wave_h_finalize.py reads it)

    def step(self, name: str) -> StepState:
        return self.steps.setdefault(name, StepState())

    @property
    def apps_seen(self) -> list[str]:
        return sorted({a for s in self.steps.values() for a in s.apps})

    @property
    def spent(self) -> float:
        """The cumulative figure the global cap is checked on (module docstring)."""
        tracked = sum(s.spent_at(self.billed_at) for s in self.steps.values())
        return round(max(tracked, self.billed_total or 0.0), 4)

    def override_for(self, plan: str) -> float | None:
        """The approved global cap override, if it was approved for ``plan``."""
        if self.cap_override is None or (self.cap_override_plan or DEFAULT_PLAN) != plan:
            return None
        return self.cap_override

    def spent_outside(self, step: str) -> float:
        """Cumulative spend minus ``step``'s running attempt (in-step deadlines add their own)."""
        cur = self.step(step).current
        own = cur.charged(self.billed_at) if cur is not None else 0.0
        return round(max(0.0, self.spent - own), 4)

    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "created": self.created,
                "billing_since": self.billing_since, "billing_baseline": self.billing_baseline,
                "billed_total": self.billed_total, "billed_at": self.billed_at,
                "cap_override": self.cap_override, "cap_override_plan": self.cap_override_plan,
                "gate_cap": self.gate_cap, "gate_cap_plan": self.gate_cap_plan,
                "plan": self.plan,
                "steps": {k: asdict(v) for k, v in self.steps.items()}}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> State:
        steps = {}
        for k, v in d.get("steps", {}).items():
            v = dict(v)
            atts = [Attempt(**a) for a in v.pop("attempts", [])]
            legacy = float(v.pop("tracked_usd", 0.0) or 0.0)  # pre-attempts state files
            started, finished = v.pop("started", None), v.pop("finished", None)
            if legacy and not atts:
                atts = [Attempt(started=started or 0.0, finished=finished, tracked_usd=legacy,
                                status=v.get("status", "failed"))]
            steps[k] = StepState(**{**v, "attempts": atts})
        return cls(d["run_id"], d["created"], d["billing_since"],
                   dict(d.get("billing_baseline", {})), steps, d.get("billed_total"),
                   d.get("billed_at"), d.get("cap_override"), d.get("cap_override_plan"),
                   d.get("gate_cap"), d.get("gate_cap_plan"), d.get("plan"))

    def mark_killed(self) -> list[str]:
        """Attempts left ``running`` by a killed process: ``killed`` (charged their budget
        until billing reconciles them), their step ``failed`` (retried). Returns their apps."""
        apps: list[str] = []
        for name, s in self.steps.items():
            for a in s.attempts:
                if a.status == "running":
                    a.status = "killed"
                    a.notes.append("process killed while running; charged its budget until "
                                   "billing reconciles it")
                    apps += a.apps
            if s.status == "running":
                s.status = "failed"
                s.notes.append(f"{name}: the previous run was killed mid-step")
        return sorted(set(apps))


def load_state(path: Path = STATE) -> State:
    if path.exists():
        return State.from_json(json.loads(path.read_text()))
    return State(run_id=time.strftime("%m%d%H%M") + uuid.uuid4().hex[:4], created=time.time(),
                 billing_since=time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400)))


_SAVE_LOCK = threading.RLock()


def save_state(st: State, path: Path = STATE) -> None:
    with _SAVE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.to_json(), indent=1, sort_keys=True, default=str))
        os.replace(tmp, path)


class StopRule(RuntimeError):
    """The runbook stops here (projected or actual spend over a cap)."""


def check_before(st: State, step: str, proj: Projection, *,
                 caps: Mapping[str, float] = CAPS_USD, global_cap: float = GLOBAL_CAP_USD
                 ) -> None:
    """The **upper** projection must fit the step cap and the global cap left (H2)."""
    cap = caps[step]
    up = proj.usd_upper
    if up > cap + 1e-9:
        raise StopRule(f"{step}: upper projection ${up:.2f} (nominal ${proj.usd:.2f}) is over "
                       f"its cap ${cap:.2f}")
    left = global_cap - st.spent
    if up > left + 1e-9:
        raise StopRule(f"{step}: upper projection ${up:.2f} is over the global budget left "
                       f"(${left:.2f} of ${global_cap:.2f}, ${st.spent:.2f} spent)")


def check_after(st: State, step: str, *, caps: Mapping[str, float] = CAPS_USD,
                global_cap: float = GLOBAL_CAP_USD) -> None:
    s = st.step(step)
    spent = s.spent_at(st.billed_at)
    if spent > caps[step] + 1e-9:
        raise StopRule(f"{step}: spent ${spent:.2f} (tracked ${s.tracked_usd:.2f}, billed "
                       f"${s.billed_usd or 0:.2f}) is over its cap ${caps[step]:.2f}")
    if st.spent > global_cap + 1e-9:
        raise StopRule(f"spent ${st.spent:.2f} in all is over the global cap ${global_cap:.2f}")


def step_budget(st: State, step: str, *, spent_in_step: float = 0.0,
                caps: Mapping[str, float] = CAPS_USD, global_cap: float = GLOBAL_CAP_USD
                ) -> float:
    """What the step may still spend: its cap minus its earlier attempts, and the global cap
    left, minus what this attempt already spent."""
    s = st.step(step)
    cur = s.current
    earlier = s.spent_at(st.billed_at) - (cur.charged(st.billed_at) if cur else 0.0)
    return max(0.0, min(caps[step] - earlier, global_cap - st.spent_outside(step))
               - spent_in_step)


def gpu_deadline_s(st: State, step: str, c: Container | float, *, spent_in_step: float = 0.0,
                   slack_s: float = CLIENT_SLACK_S, caps: Mapping[str, float] = CAPS_USD,
                   global_cap: float = GLOBAL_CAP_USD) -> float:
    """Wall seconds still affordable at ``c``'s $/s (a container or a $/s figure) in this step,
    keeping ``slack_s`` of it for the cancel / stop that follows the deadline (H2)."""
    rate = c if isinstance(c, float | int) else c.usd_per_s()
    left = step_budget(st, step, spent_in_step=spent_in_step, caps=caps, global_cap=global_cap)
    return max(0.0, left - slack_s * rate) / rate


# ---------------------------------------------------------------------------------------------
# Billing (scripts/spend.py)


def billed_by_app(since: str) -> dict[str, float]:
    import spend  # scripts/spend.py

    out: dict[str, float] = {}
    for row in spend.report(since):
        name = row.get("description") or row.get("Description") or "?"
        out[name] = out.get(name, 0.0) + float(row.get("cost") or row.get("Cost") or 0)
    return out


def billed_delta(st: State, apps: Iterable[str], now: Mapping[str, float] | None = None
                 ) -> float | None:
    if now is None:
        try:
            now = billed_by_app(st.billing_since)
        except Exception:  # noqa: BLE001 - billing is a cross-check; tracking still applies
            return None
    return round(sum(now.get(a, 0.0) - st.billing_baseline.get(a, 0.0) for a in set(apps)), 4)


def refresh_billing(st: State, *, now: Mapping[str, float] | None = None,
                    clock: Callable[[], float] = time.time) -> bool:
    """Re-read billing for every app seen so far (H3): the global billed total, and per step
    the billed delta of the apps only that step used (shared apps count globally only)."""
    if now is None:
        try:
            now = billed_by_app(st.billing_since)
        except Exception:  # noqa: BLE001
            return False
    st.billed_total = billed_delta(st, st.apps_seen, now)
    owners: dict[str, set[str]] = {}
    for name, s in st.steps.items():
        for a in s.apps:
            owners.setdefault(a, set()).add(name)
    for name, s in st.steps.items():
        mine = [a for a in s.apps if owners.get(a) == {name}]
        s.billed_usd = billed_delta(st, mine, now) if mine else s.billed_usd
    st.billed_at = clock()
    return True


class Heartbeat:
    """While a step runs: every ``every`` seconds, store ``elapsed x peak $/s`` as the running
    attempt's tracked spend and save the state, so a killed process leaves it behind (H3)."""

    def __init__(self, st: State, attempt: Attempt, path: Path, *, every: float = HEARTBEAT_S,
                 clock: Callable[[], float] = time.time):
        self.st, self.attempt, self.path, self.every, self.clock = st, attempt, path, every, clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wave-h-heartbeat")

    def beat(self) -> None:
        now = self.clock()
        a = self.attempt
        a.heartbeat = now
        a.tracked_usd = round(max(a.tracked_usd, (now - a.started) * a.rate_usd_per_s), 4)
        save_state(self.st, self.path)

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            self.beat()

    def __enter__(self) -> Heartbeat:
        self.beat()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


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

    @property
    def episodes(self) -> tuple[str, ...]:
        return episodes_in(self.manifest, self.spec.episodes)

    def deadline_s(self, step: str, c: Container | float, **kw: Any) -> float:
        return gpu_deadline_s(self.st, step, c, caps=self.caps, global_cap=self.global_cap,
                              **kw)

    def budget(self, step: str, **kw: Any) -> float:
        return step_budget(self.st, step, caps=self.caps, global_cap=self.global_cap, **kw)

    def sh(self, args: list[str], *, timeout: float = 1800, check: bool = True):
        return h0.sh(args, cwd=self.work, timeout=timeout, check=check,
                     log=self.work / "wave-h.log")

    def note_app(self, name: str) -> None:
        """Record an app in the running attempt *before* it can spend (H3)."""
        cur = self.st.step(self.step).current if self.step else None
        if cur is not None and name not in cur.apps:
            cur.apps.append(name)
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


def _modal_run(ctx: Ctx, target: str, args: list[str], *, limit: float, out: Path
               ) -> tuple[dict[str, Any] | None, float, str | None]:
    """``modal run <target> <args> --out <out>`` with a client deadline; the app is stopped
    after it either way. Returns (result JSON, wall s, error)."""
    t0 = time.time()
    err = None
    try:
        ctx.sh(["modal", "run", target, *args, "--out", str(out)], timeout=limit, check=True)
    except subprocess.TimeoutExpired:
        err = f"client deadline {limit:.0f} s"
    except RuntimeError as e:
        err = str(e)
    finally:
        ctx.sh(["modal", "app", "stop", "--yes", TEMPSEG_APP], check=False, timeout=120)
    wall = time.time() - t0
    try:
        res = json.loads(out.read_text()) if out.exists() and err is None else None
    except ValueError as e:
        res, err = None, f"unreadable {out.name}: {e}"
    return res, wall, err


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
    ctx.note_app(TEMPSEG_APP)
    out: dict[str, Any] = {"apps": [TEMPSEG_APP], "built": True}
    rdir = ctx.work
    limit = min(MERGE_MAX_S, ctx.deadline_s("tempseg", MERGE, slack_s=LAB_SLACK_S))
    if limit < 120:
        return {**out, "ok": False, "error": f"budget left allows {limit:.0f} s",
                "tracked_usd": 0.0}
    merge, wall, err = _modal_run(ctx, target, ["--step", "merge", "--force"], limit=limit,
                                  out=rdir / "tempseg-merge.json")
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
    eq, wall, err = _modal_run(ctx, target, ["--step", "equivalence"], limit=limit,
                               out=rdir / "tempseg-equivalence.json")
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
    ctx.note_app("flora-balanced-lab")
    with modal.enable_output(), lab.app.run():
        out = lab.preflight.remote()
    wall = time.time() - t0
    ok = all(out.get(k, {}).get("receipt") for k in ("tempseg", "captioner"))
    return {"ok": ok, "wall_s": round(wall, 1), "tracked_usd": PREFLIGHT.usd_per_s() * wall,
            "apps": ["flora-balanced-lab"],
            "facts": {k: {kk: v.get(kk) for kk in ("receipt", "eos_token",
                                                    "extra_state_prefixes")}
                      for k, v in out.items() if isinstance(v, dict)}}


def stage_local(ctx: Ctx, data: str, ids: Sequence[str]) -> Path:
    """``work/<data>/`` as an ``mp4_dir`` of ``ids`` (``h0.fetch_dataset``), each episode named
    by the plan's :meth:`PlanSpec.local_id` (certified: noise episodes sort first). Idempotent:
    an episode already laid out under its local id is not fetched again."""
    root = ctx.work / data
    todo = [g for g in ids if not (root / f"{ctx.spec.local_id(g)}.mp4").is_file()]
    if todo:
        h0.fetch_dataset(ctx.manifest, ctx.work, episodes={data: tuple(todo)}, poison_job=None)
    for g in todo:
        base, local = h0.episode_id(g), ctx.spec.local_id(g)
        if base != local:
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
        ctx.sh(["modal", "app", "stop", "--yes", plan.app_name], check=False, timeout=120)
        out["stopped_at"] = time.time()
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
    cfg, job_id = _stage_job(ctx, f"{step}-{JOB}", eps, embodiment=EMBODIMENT,
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
        raise RuntimeError(f"{side} runs all {len(spec.episodes)} golden {EMBODIMENT} episodes; "
                           f"not staged: {', '.join(miss)}")
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
    episodes = balanced_episodes(ctx.manifest, ctx.spec)
    derived = resolve_receipts(["captioner_bf16", "tempseg_merged"], require_pin=True)
    rate = (pool_container("segment", "balanced").usd_per_s()
            + pool_container("caption", "balanced").usd_per_s())
    cfg, job_id = _stage_job(ctx, f"balanced-{JOB}", episodes, embodiment=EMBODIMENT,
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
    return {**e2e, **out}


def balanced_job_done(st: State) -> bool:
    """(d)'s deployed job finished (``done``), whatever its step's status says: a state from
    before the lab became its own step marks (d) ``failed`` when only the lab failed."""
    job = ((st.steps.get("balanced") or StepState()).result.get("jobs") or {}).get(JOB) or {}
    return job.get("state") == "done" and bool(job.get("job_id"))


def adopt_balanced_job(st: State) -> bool:
    """Mark (d) ``done`` when its deployed job finished but the step failed on the lab it
    used to run inline (the 2026-10-22 state). Returns whether it changed anything."""
    s = st.steps.get("balanced")
    if s is None or s.status == "done" or not balanced_job_done(st):
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
    run id and kind, (d)'s episodes, no error anywhere, L0 rows and Gate A requests. A
    Captioner result must also have captioned ``segments_from`` ((d)'s job id and segment stage
    hash, :func:`segment_binding`); without a known binding it is never reused."""
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
          and not (res.get("dtbias") or {}).get("error"))
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

    def persist() -> None:  # each kind's summary lands in the state as soon as it returns
        ctx.st.step("lab").result = {"ok": False, "partial": True, **out, "tracked_usd": spent,
                                     "apps": ["flora-balanced-lab"] if todo else []}
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
    ctx.note_app("flora-balanced-lab")  # only now: a precondition failure above charged $0
    fns = {"tempseg": lab.run_tempseg, "captioner": lab.run_captioner}
    with modal.enable_output(), lab.app.run():
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
            "apps": ["flora-balanced-lab"]}


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
    job_id = f"wave-h-{ctx.st.run_id}-noise-{NOISE_JOB}"
    tag = f"noise-{NOISE_JOB}"
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
        stages[o["stage"]] = {
            "verdict": c.get("verdict"), "id": c.get("id"), "reasons": c.get("reasons"),
            "episodes": sum(b.get("n", 0) for b in
                            ((c.get("metrics") or {}).get("candidate") or {}).values()),
            "noise_episodes": ((c.get("metrics") or {}).get("noise") or {}).get("n"),
            "missing": finalize.missing_for_certified(c) if c else None,
            "refused": o.get("refused") or o.get("deferred")}
    ok = (set(stages) == set(CERT_STAGES)
          and all(v["verdict"] == "certified" for v in stages.values()))
    res: dict[str, Any] = {"ok": ok, "stages": stages, "tracked_usd": 0.0, "apps": []}
    if not ok:
        res["error"] = "not certified: " + "; ".join(
            f"{k}: {v['verdict'] or v['refused']}" + (f" ({', '.join(v['missing'])})"
                                                      if v.get("missing") else "")
            for k, v in sorted(stages.items()))
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
                  plan: str = DEFAULT_PLAN) -> str:
    gcap = plan_global_cap(plan, projections, st.override_for(plan))
    caps, title = plan_caps(plan, projections), titles(plan)
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
               only: str | None = None) -> None:
    steps = (only,) if only else PLANS[plan]
    spec = SPECS[plan]
    gcap = plan_global_cap(plan, projections, st.override_for(plan))
    caps, title = plan_caps(plan, projections), titles(plan)
    out(f"Wave H-lite run {st.run_id} ({EMBODIMENT} only), plan {plan} "
        f"({' -> '.join(title[s].split('.')[0] for s in steps)}): spent ${st.spent:.2f} of "
        f"${gcap:.2f} (billing read "
        f"{'never' if st.billed_at is None else time.ctime(st.billed_at)})")
    if manifest is not None:
        eps, miss = (episodes_in(manifest, spec.episodes),
                     missing_episodes(manifest, spec.episodes))
        src = (f"expected durations from the {spec.golden} selection (offline, +"
               f"{DURATION_HEADROOM:.0%})" if manifest.get("offline")
               else f"{spec.golden} manifest")
        dur = durations(manifest)
        out(f"  episodes: {len(eps)} {EMBODIMENT}, {sum(dur[e] for e in eps):.1f} video s ({src}"
            + (f"; NOT STAGED, left out: {', '.join(miss)}" if miss else "") + ")")
        if plan == CERTIFIED:
            nv = sum(dur.get(e, 0.0) for e in spec.noise)
            out(f"  noise subset: {len(spec.noise)} of them, {nv:.1f} video s (`flora index "
                f"--limit {spec.noise_limit}` on the oracle's config and data dir: the same "
                "stage hash)")
    for name in steps:
        p = projections[name]
        s = st.steps.get(name, StepState())
        rule = "ok"
        try:
            check_before(st, name, p, caps=caps, global_cap=gcap)
        except StopRule as e:
            rule = f"STOP: {e}"
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
    if "balanced" in steps:
        basis = startup_basis()
        out("  assumption: (d) is priced with a cold compile: startup segment "
            f"{basis['segment']}, caption {basis['caption']} (a startup measured cold is taken "
            f"as is; any other is scaled by {COLD_COMPILE_FACTOR:.2f}, the Rewarder's cold/warm "
            "@enter 318 s / 229 s, docs/cold-start.md). The deployed workers copy "
            "flora-cache:/balanced/<engine> to local disk and never write it back")
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
            "first step whose upper projection no longer fits the cap left")


def load_manifest(offline: bool, version: str = "golden/v1") -> dict[str, Any]:
    if offline:
        return offline_manifest(DURATION_HEADROOM, version)
    if version == "golden/v1":
        return h0.golden_manifest()
    import modal

    vol = modal.Volume.from_name(h0.GOLDEN_VOLUME, version=2)
    return json.loads(b"".join(vol.read_file(f"{version}/manifest.json")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print the plan; spend nothing")
    g.add_argument("--yes-spend", action="store_true", help="run (GPU steps spend money)")
    g.add_argument("--report", action="store_true",
                   help="rewrite reports/wave-h/runbook.md from the state (never "
                        "docs/wave-h-report.md: wave_h_finalize.py owns it)")
    ap.add_argument("--plan", choices=tuple(PLANS), default=DEFAULT_PLAN,
                    help="balanced-only (default): steps 0, a, d; full: every step; "
                         "certified: b, f, d, (d2), e on golden/v2 (its own state)")
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
    spec = SPECS[args.plan]
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
    if args.plan == CERTIFIED:
        st.plan = CERTIFIED
    if adopt_balanced_job(st):
        print("note: (d)'s deployed job is done; (d) counts as done and its lab is step lab")
    killed_apps = st.mark_killed() if args.yes_spend else []
    manifest = load_manifest(args.offline, spec.golden)
    miss = missing_episodes(manifest, spec.episodes)
    if miss:
        print(f"note: only {len(spec.episodes) - len(miss)} of the {len(spec.episodes)} "
              f"{EMBODIMENT} episodes are staged; using those (missing: {', '.join(miss)})")
    lab_done = args.plan == CERTIFIED and lab_already_done()[0]
    reuse = () if lab_done else lab_reuse(st, spec)
    projections = plan_projections(args.plan, manifest, lab_done=lab_done, reuse=reuse)
    caps = plan_caps(args.plan, projections)
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
                save_state(st, args.state)
                print(f"global cap for plan {args.plan} raised to ${st.cap_override:.2f} "
                      "(user-approved; saved in the state)")
            else:
                print(f"dry run: would raise the global cap for plan {args.plan} to "
                      f"${st.cap_override:.2f}; not saved (pass --yes-spend to approve it)")
    gcap = plan_global_cap(args.plan, projections, st.override_for(args.plan))
    print_plan(st, projections, manifest=manifest, plan=args.plan, only=args.only)
    report = spec.report or REPORT
    if args.report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(st, projections, plan=args.plan))
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
    for app in killed_apps:  # a killed run may have left them up
        subprocess.run(["modal", "app", "stop", "--yes", app], capture_output=True, text=True)
        print(f"stopped {app} (left by a killed run)")
    st.gate_cap, st.gate_cap_plan = gcap, args.plan  # the cap this run gates on
    save_state(st, args.state)
    os.chdir(args.work)  # relative input.path is hashed against the cwd: plan where flora runs
    ctx = Ctx(st, manifest, args.work, state_path=args.state, global_cap=gcap, spec=spec,
              caps=caps)
    rc = 0
    for name in PLANS[args.plan]:
        if args.only and name != args.only:
            continue
        s = st.step(name)
        if s.status == "done":
            print(f"skip {name}: done")
            continue
        refresh_billing(st)  # every app seen so far, before the step (H3)
        p = projections[name]
        s.projected_usd, s.projected_upper_usd = p.usd, p.usd_upper
        try:
            check_before(st, name, p, caps=caps, global_cap=gcap)
        except StopRule as e:
            s.status, rc = "stopped", 3
            s.notes.append(f"stop rule before: {e}")
            break
        att = Attempt(started=time.time(), rate_usd_per_s=p.peak_usd_per_s,
                      budget_usd=round(step_budget(st, name, caps=caps, global_cap=gcap), 4))
        s.attempts.append(att)
        s.status = "running"
        ctx.step = name
        save_state(st, args.state)
        try:
            with Heartbeat(st, att, args.state):
                res = runner(args.plan, name)(ctx)
            s.result = res
            att.tracked_usd = round(float(res.get("tracked_usd") or 0.0), 4)
            att.report_error = bool(res.get("report_error"))
            att.status = s.status = ("done" if res.get("ok") else
                                     "needs_pin" if res.get("needs_pin") else "failed")
        except Exception as e:  # noqa: BLE001 - recorded; the runbook stops
            att.status = s.status = "failed"
            if att.apps:
                att.report_error = True  # what it spent is unknown: charged its budget
            else:  # L1: failed before it started any app: nothing to charge
                att.tracked_usd = 0.0
                att.notes.append("failed before any app started: charged $0")
            s.notes.append(f"error: {e!r}")
        att.finished = time.time()
        for a in s.result.get("apps", ()):
            if a not in att.apps:
                att.apps.append(a)
        refresh_billing(st)
        save_state(st, args.state)
        try:
            check_after(st, name, caps=caps, global_cap=gcap)
        except StopRule as e:
            s.notes.append(f"stop rule after: {e}")
            rc = 3
            break
        if s.status == "needs_pin":
            print(pin_instructions(s.result))
            rc = NEEDS_PIN
            break
        if s.status != "done":
            rc = 1
            break
    save_state(st, args.state)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(st, projections, plan=args.plan))
    print(f"wrote {report}; state {args.state}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
