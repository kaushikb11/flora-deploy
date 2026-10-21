"""``stages.reward`` builds exactly the requests NVIDIA's Rewarder scripts build.

NVIDIA's real ``terminal_reward.main`` / ``dense_reward.main`` run on synthetic multi-camera
videos with the real Rewarder processor (tokenizer + chat template, no weights). Only the engine
is faked: ``vllm`` is a stub module and ``load_llm`` returns a fake ``LLM`` (via
``stages.reward.shared_engine``, which also checks NVIDIA's ``load_llm`` arguments against
``stages.reward.engine_spec``). The fake answers depend only on request content, so the two
runs get the same answers whatever the batching. We then check that

1. the batched run makes one ``generate`` call carrying every request NVIDIA made over its
   several calls, with identical prompts, frames, metadata, per-request kwargs and
   ``SamplingParams``;
2. the batched records' ``nv`` field equals the lines NVIDIA wrote, and the unrounded terminal
   probabilities are equal.

Everything runs in a fresh interpreter: the REWARDER family must not share a process with
SEGCAP, which other nvlink tests load into the pytest process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.nvlink

# id, views, (width, height), fps, seconds
EPISODES = (
    ("three_view", 3, (160, 120), 10, 21.0),
    ("two_view", 2, (128, 128), 15, 12.0),
    ("one_view", 1, (176, 96), 30, 7.5),
)


def write_video(path: Path, size: tuple[int, int], fps: int, frames: int, seed: int) -> None:
    import av

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(frames):
            image = np.roll(base, shift=3 * i, axis=1)
            image[: size[1] // 4, : size[0] // 4] = (7 * i) % 256
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def jobs(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("rewarder_jobs")
    lines = []
    for n, (name, views, size, fps, seconds) in enumerate(EPISODES):
        (root / name).mkdir()
        videos = []
        for v in range(views):
            rel = f"{name}/cam{v}.mp4"
            write_video(root / rel, size, fps, round(seconds * fps), seed=10 * n + v)
            videos.append(rel)
        task = f"  Put the {name} cup\n on the tray. "
        lines.append({"id": name, "task": task, "videos": videos})
    path = root / "jobs.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def run_worker(code_dir: Path, jobs: Path, mode: str) -> dict:
    out = subprocess.run(
        [sys.executable, __file__, str(code_dir), str(jobs), mode],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def rewarder_dir(nv_code) -> Path:
    path = nv_code("rewarder")
    if not (path / "processor_config.json").is_file():
        pytest.skip("Rewarder processor files not fetched")
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    return path


@pytest.mark.parametrize("mode", ["terminal", "terminal_rationale", "dense"])
def test_batched_requests_and_records_match_nvidia_scripts(rewarder_dir, jobs, mode):
    result = run_worker(rewarder_dir, jobs, mode)
    expected_requests = {"terminal": 3, "terminal_rationale": 3}.get(mode)
    assert result["batched_generate_calls"] == 1
    if expected_requests:
        assert result["num_requests"] == expected_requests
    else:  # dense: every clip of every episode, while NVIDIA calls once per episode
        assert result["num_requests"] > len(EPISODES)
        assert result["reference_generate_calls"] == len(EPISODES)
    assert result["request_mismatches"] == []
    assert result["params_equal"] is True
    assert result["load_llm_ok"] is True
    assert result["parity"]["identical"] is True, result["parity"]
    if mode != "dense":
        assert all(e["max_abs_prob_diff"] == 0.0 for e in result["parity"]["episodes"])
        assert all(e["text_equal"] and e["score_equal"] for e in result["parity"]["episodes"])
    else:
        assert all(e["frame_scores_equal"] for e in result["parity"]["episodes"])
        assert result["per_step_lengths"] == result["num_frames"]


# ---------------------------------------------------------------------------
# Worker (fresh interpreter): python test_reward_parity.py <code_dir> <jobs> <mode>
# ---------------------------------------------------------------------------


def _install_fake_vllm():
    import types

    class _Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __eq__(self, other):
            return type(self) is type(other) and self.kwargs == other.kwargs

        def __repr__(self):
            return f"{type(self).__name__}({self.kwargs!r})"

    class SamplingParams(_Params):
        pass

    class StructuredOutputsParams(_Params):
        pass

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = SamplingParams
    sp = types.ModuleType("vllm.sampling_params")
    sp.StructuredOutputsParams = StructuredOutputsParams
    vllm.sampling_params = sp
    sys.modules["vllm"], sys.modules["vllm.sampling_params"] = vllm, sp


class _Logprob:
    def __init__(self, logprob):
        self.logprob = logprob


class _Completion:
    def __init__(self, text, token_ids, logprobs):
        self.text, self.token_ids, self.logprobs = text, token_ids, logprobs


class _Output:
    def __init__(self, prompt_token_ids, completion):
        self.prompt_token_ids, self.outputs = prompt_token_ids, [completion]


class FakeLLM:
    """Answers depend only on the request (prompt + frames), never on batch position."""

    def __init__(self, tokenizer, mode):
        self.tokenizer, self.mode = tokenizer, mode
        self.calls = []

    def get_tokenizer(self):
        return self.tokenizer

    def reset_prefix_cache(self):
        return True

    def _key(self, request) -> int:
        frames, meta = request["multi_modal_data"]["video"]
        return int(frames[:, ::7, ::5].sum()) + 31 * sum(meta["frames_indices"])

    def _answer(self, request):
        key = self._key(request)
        if self.mode == "dense":
            run = 1 + key % 20
            scores = "3" * run + "21" + "3" * (30 - run)
            return (f"Worst: 1\nQuality:\n{scores}\nRationales:\n"
                    f"{run + 1}-{run + 1} slow approach\n{run + 2}-{run + 2} drops the cup")
        score = 1 + key % 5
        if self.mode == "terminal_rationale":
            return f"Rationale: the cup ends {key % 97} mm from the tray.\n\nScore: {score}"
        return f"Score: {score}"

    def generate(self, requests, params, use_tqdm=True):
        self.calls.append((list(requests), params))
        outputs = []
        digit_ids = [self.tokenizer.convert_tokens_to_ids(str(d)) for d in range(1, 6)]
        for request in requests:
            text = self._answer(request)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            logprobs = None
            if self.mode != "dense":
                key = self._key(request)
                logprobs = []
                for position, token in enumerate(ids):
                    cands = {d: _Logprob(-1.0 - ((key + position + d) % 7) / 3.0)
                             for d in digit_ids}
                    cands[token] = _Logprob(-0.01 * (1 + key % 9))
                    logprobs.append(cands)
            prompt_ids = self.tokenizer.encode(request["prompt"], add_special_tokens=False)
            outputs.append(_Output(prompt_ids, _Completion(text, ids, logprobs)))
        return outputs


def _same_request(a, b) -> list[str]:
    problems = []
    for key in ("prompt", "mm_processor_kwargs"):
        if a[key] != b[key]:
            problems.append(key)
    (fa, ma), (fb, mb) = a["multi_modal_data"]["video"], b["multi_modal_data"]["video"]
    if fa.dtype != fb.dtype or fa.shape != fb.shape or not np.array_equal(fa, fb):
        problems.append("frames")
    if ma != mb:
        problems.append("metadata")
    if set(a) != set(b) or set(a["multi_modal_data"]) != set(b["multi_modal_data"]):
        problems.append("keys")
    return problems


def _worker(code_dir: str, jobs: str, mode: str) -> dict:
    import tempfile

    _install_fake_vllm()
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family
    from flora_deploy.stages import reward

    nv = load_family(Family.REWARDER, code_dir)
    processor = nv.engine.load_processor(code_dir)
    spec = reward.engine_spec(nv, code_dir)
    script = nv.dense_reward if mode == "dense" else nv.terminal_reward

    ref_llm = FakeLLM(processor.tokenizer, mode)
    output = Path(tempfile.mkdtemp()) / "out.jsonl"
    argv = reward.reference_argv(mode, jobs, output, code_dir)
    load_llm_ok = True
    with reward.shared_engine(script, ref_llm, spec, {"max_logprobs": 20}) as calls, \
            reward.tap_score_probabilities(nv.core) as probabilities:
        try:
            assert script.main(argv) == 0
        except reward.EngineMismatchError:
            load_llm_ok = False
            raise
    load_llm_ok = load_llm_ok and len(calls) == 1
    reference = [json.loads(line) for line in output.read_text().splitlines()]

    bat_llm = FakeLLM(processor.tokenizer, mode)
    episodes = reward.read_jobs(nv, jobs)
    result = reward.run(bat_llm, nv, processor, episodes, mode)

    ref_requests = [r for reqs, _ in ref_llm.calls for r in reqs]
    (bat_requests, bat_params), = bat_llm.calls
    mismatches = []
    if len(ref_requests) != len(bat_requests):
        mismatches.append(f"count {len(ref_requests)} != {len(bat_requests)}")
    for i, (a, b) in enumerate(zip(ref_requests, bat_requests, strict=False)):
        mismatches += [f"{i}:{p}" for p in _same_request(a, b)]
    params_equal = all(params == bat_params for _, params in ref_llm.calls)
    parity = reward.compare(mode, reference, result.records,
                            probabilities if mode != "dense" else None)
    out = {
        "batched_generate_calls": len(bat_llm.calls),
        "reference_generate_calls": len(ref_llm.calls),
        "num_requests": len(bat_requests),
        "request_mismatches": mismatches,
        "params_equal": params_equal,
        "load_llm_ok": load_llm_ok,
        "parity": parity,
    }
    if mode == "dense":
        out["per_step_lengths"] = [len(r["dense"]["per_step"]) for r in result.records]
        out["num_frames"] = [r["dense"]["num_frames"] for r in result.records]
    return out


if __name__ == "__main__":
    print(json.dumps(_worker(*sys.argv[1:4])))
