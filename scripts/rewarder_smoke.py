"""WS-H smoke: NVIDIA's Rewarder scripts (reference) vs ``stages.reward`` (batched) on Modal.

    uv run modal run scripts/rewarder_smoke.py                       # stage + all modes
    uv run modal run scripts/rewarder_smoke.py --skip-stage --modes terminal,dense
    uv run modal run scripts/rewarder_smoke.py --stage-only

Ephemeral (``modal run``) only. One ``H100!`` container (``max_containers=1``) serves every
call, so the 52 GiB engine loads once. Raw outputs (model outputs, never committed) go to
``--out`` (default ``reports/wsh/``, git-ignored); the printed summary has numbers only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from flora_deploy.modal_app.rewarder_app import H100_USD_PER_H, Rewarder, app, stage_samples
from flora_deploy.stages.reward import MODES, compare


def summarize(mode: str, ref: dict, bat: dict, parity: dict) -> dict:
    def brief(stats: dict) -> dict:
        keys = ("generate_calls", "num_requests", "prompt_tokens", "output_tokens",
                "generate_s", "wall_s")
        return {**{k: stats.get(k) for k in keys}, **stats["throughput"]}

    return {
        "mode": mode,
        "identical": parity["identical"],
        "episodes": parity["episodes"],
        "reference": brief(ref["stats"]),
        "batched": brief(bat["stats"]),
        "per_request_prompt_tokens": {
            "reference": [r["prompt_tokens"] for r in ref["stats"]["requests"]],
            "batched": [r["prompt_tokens"] for r in bat["stats"]["requests"]],
        },
    }


@app.local_entrypoint()
def main(modes: str = ",".join(MODES), out: str = "reports/wsh", skip_stage: bool = False,
         stage_only: bool = False) -> None:
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
    summaries, engine = [], None
    for mode in [m for m in modes.split(",") if m]:
        ref = rewarder.reference.remote(mode)
        bat = rewarder.batched.remote(mode)
        parity = compare(mode, ref["records"], bat["records"], ref.get("probabilities"))
        (out_dir / f"{mode}.json").write_text(
            json.dumps({"reference": ref, "batched": bat, "parity": parity}, indent=2)
        )
        summary = summarize(mode, ref, bat, parity)
        summaries.append(summary)
        engine = bat["engine"]
        print(json.dumps(summary, indent=2))
    uptime = engine["container_uptime_s"] if engine else 0.0
    report = {
        "summaries": summaries,
        "engine": engine,
        "local_elapsed_s": time.time() - t0,
        "gpu_container_uptime_s_at_last_call": uptime,
        "gpu_usd_estimate_at_last_call": uptime / 3600 * H100_USD_PER_H,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "summaries"}, indent=2))
