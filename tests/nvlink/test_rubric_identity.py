"""``stages.rubric`` builds exactly the model calls NVIDIA's unmodified ``rubric_reward.main``
builds, and picks NVIDIA's medoid.

NVIDIA's real ``rubric_reward.main`` runs on two jobs files (the ``robot`` + ``cameras`` form and
the ``header`` + ``task_name`` form) with the real Rewarder processor (tokenizer + chat template,
no weights). Only the model call and the media I/O are replaced, identically for both runs:
``load_llm`` returns a fake ``LLM`` through ``stages.rubric.shared_engine`` (which also checks
NVIDIA's ``load_llm`` arguments against ``stages.rubric.nvidia_recipe``), ``vllm`` is a stub
module, and ``rubric_reward.video_inputs`` (``qwen_vl_utils`` + decord, not in the CPU stack)
is a deterministic stand-in that records the ``messages`` it is given. Then:

1. every request (rendered prompt, per-camera frames and metadata, keys), every ``messages``
   list handed to the video reader, the ``SamplingParams`` and the ``load_llm`` call are equal;
2. NVIDIA's output lines equal our audit lines (``nv``), including the per-sample parse, the
   medoid choice on a **tie** (two clusters at equal summed L1 distance: the lowest parsed
   index), clamped / wrong-length / unclosed-thinking answers, and the no-parsed-sample case
   (NVIDIA's ``scores: null``; ours an ``nv_failure`` carrying that line);
3. ``medoid_sample`` with NVIDIA's real ``core.medoid`` matches a brute-force argmin on
   synthetic score vectors with ties, all-equal vectors and unparsed samples.

Fresh interpreter, as ``test_reward_parity``: the REWARDER family must not share a process with
SEGCAP.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.nvlink

RUBRIC = [{"action": "Grasp the cup",
           "grading": [{"label": "Not attempted", "points": 0.0},
                       {"label": "Grasped after re-tries", "points": 0.1},
                       {"label": "Clean grasp", "points": 0.2}]},
          {"action": "  Set the cup\n on the tray ", "max_score": 0.8,
           "grading": [{"label": "Not attempted", "points": 0.0},
                       {"label": "Placed but tipped over", "points": 0.4},
                       {"label": "Placed upright", "points": 0.8}]}]
ROBOT_FORM = {"robot": "bimanual robot",
              "cameras": ["a top (scene) camera", "a left-wrist camera", "a right-wrist camera"]}
HEADER_FORM = {"header": "You are an expert evaluator. Video 1 is the scene, Video 2 the left "
                         "wrist, and Video 3 the right wrist.\n\n",
               "task_name": "cup to tray"}
CASES = ("spread", "tie", "none", "messy")


def write_jobs(root: Path, form: dict, name: str) -> Path:
    lines = []
    for n, case in enumerate(CASES):
        videos = [str(root / f"{name}-{case}-cam{v}.mp4") for v in range(3)]
        lines.append({"id": f"{name}_{case}", "task": f"  Put the cup on the tray (case={case}). ",
                      "videos": videos, "rubric": RUBRIC, **form})
        assert n >= 0
    path = root / f"{name}.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


@pytest.fixture
def rewarder_dir(nv_code) -> Path:
    path = nv_code("rewarder")
    if not (path / "processor_config.json").is_file():
        pytest.skip("Rewarder processor files not fetched")
    pytest.importorskip("transformers")
    return path


def test_rubric_requests_lines_and_medoid_match_nvidia(rewarder_dir, tmp_path):
    jobs = [write_jobs(tmp_path, ROBOT_FORM, "robot"), write_jobs(tmp_path, HEADER_FORM, "header")]
    out = subprocess.run([sys.executable, __file__, str(rewarder_dir), *map(str, jobs)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    for form in result["forms"]:
        assert form["load_llm_ok"] is True, form
        assert form["load_llm_calls"] == 1
        assert form["recipe"]["engine"]["tensor_parallel_size"] == 2
        assert form["recipe"]["sampling"] == {"n": 9, "temperature": 1.0, "top_p": 1.0,
                                              "top_k": 0, "max_tokens": 20480}
        assert form["recipe"]["engine"]["max_model_len"] == 98304
        assert form["generate_calls"] == [len(CASES), len(CASES)]  # one per episode, both
        assert form["request_mismatches"] == []
        assert form["messages_equal"] is True
        assert form["params_equal"] is True
        assert form["line_mismatches"] == [], form["line_mismatches"]
        assert form["medoids"]["tie"] == 1  # first parsed sample of the tied clusters
        assert form["medoids"]["none"] is None and form["none_error"] == [
            "nv_failure", "no parsed rubric answers"]
        assert form["records"] == len(CASES) - 1
    assert result["medoid_cases"] and all(result["medoid_cases"])


# ---------------------------------------------------------------------------
# Worker (fresh interpreter): python test_rubric_identity.py <code_dir> <jobs>...
# ---------------------------------------------------------------------------


def _install_fake_vllm():
    import types

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n = kwargs.get("n", 1)

        def __eq__(self, other):
            return type(self) is type(other) and self.kwargs == other.kwargs

        def __repr__(self):
            return f"SamplingParams({self.kwargs!r})"

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = SamplingParams
    sys.modules["vllm"] = vllm


def fake_video_inputs(seen: list):
    def video_inputs(messages):
        seen.append(json.loads(json.dumps(messages)))
        out = []
        for item in messages[-1]["content"]:
            if item.get("type") != "video":
                continue
            seed = sum(map(ord, item["video"])) % 9973
            frames = np.random.default_rng(seed).integers(0, 255, (4, 32, 48, 3), np.uint8)
            out.append((frames, {"fps": item["fps"], "frames_indices": [0, 5, 10, 15],
                                 "total_num_frames": 40, "duration": 20.0,
                                 "do_sample_frames": False}))
        return out
    return video_inputs


A, B = "[0.0, 0.8]", "[0.2, 0.0]"
ANSWERS = {
    "spread": ["[0.2, 0.8]", "[0.1, 0.8]", "[0.2, 0.4]", "[0.0, 0.8]", "[0.2, 0.8]",
               "[0.1, 0.4]", "[0.2, 0.8]", "[0.0, 0.0]", "[0.2, 0.8]"],
    # unparsed, then two clusters at equal summed L1 distance: NVIDIA ties to the lowest index
    "tie": [None, A, B, None, B, A, None, None, None],
    "none": [None] * 9,
    # clamped (5 -> 0.2), wrong length, thinking never closed, prose numbers after ANSWER:
    "messy": ["[5, 0.8]", "[0.1]", "<cut>", "0.1 and then 0.4", "[0.2, 0.8, 0.4]",
              "[0.1, -3]", "[0.2, 0.8]", "<cut>", "[0.0, 0.4]"],
}


class _Completion:
    def __init__(self, text, n_tokens):
        self.text, self.token_ids, self.finish_reason = text, [0] * n_tokens, "stop"


class _Output:
    def __init__(self, prompt_ids, outputs):
        self.prompt_token_ids, self.outputs = prompt_ids, outputs


class FakeLLM:
    """Answers depend only on the request's ``case=`` tag (content-keyed)."""

    def __init__(self):
        self.calls = []

    def reset_prefix_cache(self):
        return True

    def generate(self, requests, params, use_tqdm=True):
        self.calls.append((list(requests), params))
        outs = []
        for r in requests:
            case = r["prompt"].split("case=", 1)[1].split(")", 1)[0]
            completions = []
            for i, a in enumerate(ANSWERS[case][: params.n]):
                if a is None:
                    text = f"<think>sample {i} thinks and fails to answer</think>\nNo idea."
                elif a == "<cut>":
                    text = f"<think>sample {i} runs out of tokens"
                else:
                    text = f"<think>sample {i} weighs the evidence</think>\nANSWER: {a}"
                completions.append(_Completion(text, 10 + i))
            outs.append(_Output(list(range(len(r["prompt"]) // 3)), completions))
        return outs


def _same_request(a, b) -> list[str]:
    problems = []
    if set(a) != set(b) or set(a["multi_modal_data"]) != set(b["multi_modal_data"]):
        problems.append("keys")
    if a["prompt"] != b["prompt"]:
        problems.append("prompt")
    va, vb = a["multi_modal_data"]["video"], b["multi_modal_data"]["video"]
    if len(va) != len(vb):
        return [*problems, "num_videos"]
    for i, ((fa, ma), (fb, mb)) in enumerate(zip(va, vb, strict=True)):
        if fa.dtype != fb.dtype or fa.shape != fb.shape or not np.array_equal(fa, fb):
            problems.append(f"frames{i}")
        if ma != mb:
            problems.append(f"metadata{i}")
    return problems


def _brute_medoid(samples):
    parsed = [i for i, s in enumerate(samples) if s["parsed"]]
    if not parsed:
        return None
    cost = {i: sum(sum(abs(x - y) for x, y in zip(samples[i]["scores"], samples[j]["scores"],
                                                   strict=True)) for j in parsed)
            for i in parsed}
    best = min(cost.values())
    return min(i for i in parsed if cost[i] == best)


def _medoid_cases(core, rb) -> list[bool]:
    rng = np.random.default_rng(7)
    grid = [0.0, 0.1, 0.2, 0.4, 0.8]
    cases = [
        [{"parsed": True, "scores": [0.0, 0.8]}] * 9,  # all equal: index 0
        [{"parsed": False, "scores": [0.0, 0.0]}, {"parsed": True, "scores": [0.2, 0.0]},
         {"parsed": True, "scores": [0.0, 0.8]}],  # two-way tie behind an unparsed sample
        [{"parsed": True, "scores": [0.1]}, {"parsed": True, "scores": [0.3]},
         {"parsed": True, "scores": [0.3]}, {"parsed": True, "scores": [0.1]}],  # 2+2 tie
    ]
    for _ in range(300):  # small grids make ties common
        n, k = int(rng.integers(1, 10)), int(rng.integers(1, 4))
        cases.append([{"parsed": bool(rng.random() > 0.3),
                       "scores": [float(rng.choice(grid)) for _ in range(k)]}
                      for _ in range(n)])
    ok = []
    for samples in cases:
        want = _brute_medoid(samples)
        try:
            got = rb.medoid_sample(core, samples)
        except ValueError:
            got = None
        ok.append(got == want)
    return ok


def _worker(code_dir: str, *jobs_files: str) -> dict:
    import tempfile

    _install_fake_vllm()
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family
    from flora_deploy.stages import reward as R
    from flora_deploy.stages import rubric as RB

    nv = load_family(Family.REWARDER, code_dir)
    processor = nv.engine.load_processor(code_dir)
    module = nv.rubric_reward
    forms = []
    for jobs_file in jobs_files:
        lines = [json.loads(x) for x in Path(jobs_file).read_text().splitlines()]
        first = lines[0]
        spec = RB.RubricSpec.from_json({
            "steps": first["rubric"],
            "cameras": [{"role": r, **({"description": d} if "cameras" in first else {})}
                        for r, d in zip(("main", "left_wrist", "right_wrist"),
                                        first.get("cameras", [None] * 3), strict=True)],
            **{k: first[k] for k in ("robot", "header", "task_name") if k in first}})
        recipe = RB.nvidia_recipe(nv, code_dir, 3)

        ref_seen, our_seen = [], []
        original = module.video_inputs
        ref_llm, our_llm = FakeLLM(), FakeLLM()
        output = Path(tempfile.mkdtemp()) / "rubric.jsonl"
        load_llm_ok = True
        module.video_inputs = fake_video_inputs(ref_seen)
        try:
            with RB.shared_engine(module, ref_llm, recipe) as calls:
                try:
                    assert module.main(["--jobs", jobs_file, "--output", str(output),
                                        "--model", code_dir, "--overwrite"]) == 0
                except R.EngineMismatchError:
                    load_llm_ok = False
                    raise
            module.video_inputs = fake_video_inputs(our_seen)
            # absolute, resolved paths, as NVIDIA's read_jobs makes them (and our workers pass)
            episodes = [R.Episode(id=x["id"], task=x["task"],
                                  videos=tuple(str(Path(v).resolve()) for v in x["videos"]))
                        for x in lines]
            run = RB.run(our_llm, nv, processor, episodes, spec, recipe=recipe, prefetch=False)
        finally:
            module.video_inputs = original
        reference = [json.loads(x) for x in output.read_text().splitlines()]
        ours = {a["episode_id"]: a["nv"] for a in run.audits}
        ours.update({e.diagnostics["episode_id"]: e.diagnostics["nv"] for e in run.errors})
        ours = json.loads(json.dumps(ours))
        line_mismatches = [ref["id"] for ref in reference if ours.get(ref["id"]) != ref]
        ref_requests = [r for reqs, _ in ref_llm.calls for r in reqs]
        our_requests = [r for reqs, _ in our_llm.calls for r in reqs]
        mismatches = [] if len(ref_requests) == len(our_requests) else ["count"]
        for i, (a, b) in enumerate(zip(ref_requests, our_requests, strict=False)):
            mismatches += [f"{i}:{p}" for p in _same_request(a, b)]
        none_err = [e for e in run.errors if e.diagnostics["episode_id"].endswith("_none")]
        forms.append({
            "load_llm_ok": load_llm_ok and calls == [calls[0]] and len(calls) == 1,
            "load_llm_calls": len(calls),
            "recipe": recipe.as_json(),
            "generate_calls": [len(ref_llm.calls), len(our_llm.calls)],
            "request_mismatches": mismatches,
            "messages_equal": ref_seen == our_seen and len(ref_seen) == len(CASES),
            "params_equal": all(p == our_llm.calls[0][1] for _, p in ref_llm.calls)
            and all(p == our_llm.calls[0][1] for _, p in our_llm.calls),
            "line_mismatches": line_mismatches,
            "medoids": {ref["id"].split("_", 1)[1]: ref["medoid_sample"] for ref in reference},
            "none_error": [none_err[0].kind, none_err[0].message] if none_err else None,
            "records": len(run.records),
        })
    return {"forms": forms, "medoid_cases": _medoid_cases(nv.core, RB)}


if __name__ == "__main__":
    print(json.dumps(_worker(*sys.argv[1:])))
