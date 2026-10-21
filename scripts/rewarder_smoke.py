"""WS-H smoke: NVIDIA's Rewarder scripts (reference) vs ``stages.reward`` (batched) on Modal.

    uv run modal run scripts/rewarder_smoke.py                       # stage + all modes
    uv run modal run scripts/rewarder_smoke.py --skip-stage --variants 3
    uv run modal run scripts/rewarder_smoke.py --skip-stage --modes terminal --packed-requests 2
    uv run modal run scripts/rewarder_smoke.py --stage-only

Per mode, on the same engine: NVIDIA's script ("reference"), then ``stages.reward.run`` with
default rounds (``align="nvidia"``: NVIDIA's own ``generate`` grouping, the ``exact`` path;
"aligned"), then one unbounded packed round ("unbounded", the previous single-``generate``
behaviour), and optionally small packed rounds (``--packed-requests N``). ``--variants K``
repeats every staged episode under K task phrasings (3 x 3 = 9 episodes > NVIDIA's
``max_num_seqs`` of 8, so NVIDIA's terminal script makes two calls).

Ephemeral (``modal run``) only. One ``H100!`` container (``max_containers=1``) serves every call,
so the 52 GiB engine loads once; the container's startup breakdown (prefetch, compile/Triton cache
state, engine build) is printed. Raw outputs (model outputs, never committed) go to ``--out``
(default ``reports/wsh/``, git-ignored); the printed summary has numbers only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from flora_deploy.modal_app.rewarder_app import USD_PER_H, Rewarder, app, stage_samples
from flora_deploy.stages.reward import MODES, compare, same_outputs

HUGE = 1 << 30


def brief(stats: dict) -> dict:
    keys = ("generate_calls", "num_requests", "prompt_tokens", "output_tokens",
            "generate_s", "decode_s", "decode_wait_s", "wall_s")
    return {**{k: stats.get(k) for k in keys}, **stats["throughput"]}


def max_prob_diff(parity: dict) -> float | None:
    diffs = [e["max_abs_prob_diff"] for e in parity["episodes"] if "max_abs_prob_diff" in e]
    return max(diffs) if diffs else None


@app.local_entrypoint()
def main(modes: str = ",".join(MODES), out: str = "reports/wsh", skip_stage: bool = False,
         stage_only: bool = False, variants: int = 1, packed_requests: int = 0) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not skip_stage:
        staged = stage_samples.remote()
        (out_dir / "staged.json").write_text(json.dumps(staged, indent=2))
        for s in staged["sources"]:
            print(f"staged {s['id']}: {len(s['cameras'])} cam(s), {s['duration_s']:.2f} s, "
                  f"{s['cameras'][0]['fps']:g} fps, {s['license']}")
        if stage_only:
            return

    rewarder = Rewarder()
    t0 = time.time()
    summaries, engine, first_call_s = [], None, None
    for mode in [m for m in modes.split(",") if m]:
        try:
            t_call = time.time()
            ref = rewarder.reference.remote(mode, variants=variants)
            if first_call_s is None:
                first_call_s = time.time() - t_call
            runs = {
                "aligned": rewarder.batched.remote(mode, variants=variants, tag="aligned"),
                "unbounded": rewarder.batched.remote(mode, variants=variants, align="packed",
                                                     round_requests=HUGE, round_mib=HUGE,
                                                     decoder="inline", tag="unbounded"),
            }
            if packed_requests:
                runs["packed"] = rewarder.batched.remote(mode, variants=variants, align="packed",
                                                         round_requests=packed_requests,
                                                         tag="packed")
        except Exception as exc:  # keep the warm container working on the other modes
            print(f"{mode}: FAILED {exc!r}")
            summaries.append({"mode": mode, "error": repr(exc)})
            continue
        probs = ref.get("probabilities")
        parity = {name: compare(mode, ref["records"], run, probs) for name, run in runs.items()}
        same = {name: same_outputs(runs["aligned"], run)
                for name, run in runs.items() if name != "aligned"}
        (out_dir / f"{mode}.json").write_text(json.dumps(
            {"reference": ref, **runs, "parity": parity, "vs_aligned": same}, indent=2))
        summary = {
            "mode": mode,
            "episodes": len(ref["records"]),
            "reference_calls": ref["stats"]["call_sizes"],
            **{f"{name}_vs_reference": {
                "identical": p["identical"],
                "text_equal": all(e.get("text_equal", not e.get("clip_text_mismatches"))
                                  for e in p["episodes"]),
                "max_abs_prob_diff": max_prob_diff(p)} for name, p in parity.items()},
            **{f"{name}_vs_aligned": s["identical"] for name, s in same.items()},
            "rounds": {name: [r["requests"] for r in run["stats"]["rounds"]]
                       for name, run in runs.items()},
            "errors": [e["message"] for run in runs.values() for e in run["errors"]],
            "flagged": runs["aligned"]["stats"]["flagged"],
            "budget_splits": runs["aligned"]["stats"]["budget_splits"],
            "uncertified_reasons": sorted({r["uncertified_reason"] or "certified"
                                           for r in runs["aligned"]["records"]}),
            "parts": runs["aligned"]["part_files"],
            "reference": brief(ref["stats"]),
            **{name: brief(run["stats"]) for name, run in runs.items()},
        }
        summaries.append(summary)
        engine = runs["aligned"]["engine"]
        print(json.dumps(summary, indent=2))
    uptime = engine["container_uptime_s"] if engine else 0.0
    report = {
        "summaries": summaries,
        "engine": engine,
        "local_elapsed_s": time.time() - t0,
        "local_first_call_s": first_call_s,
        "gpu_container_uptime_s_at_last_call": uptime,
        "usd_estimate_at_last_call": uptime / 3600 * USD_PER_H,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "summaries"}, indent=2))
