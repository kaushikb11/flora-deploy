import pytest

from flora_deploy.validate import gatec as G

SP = G.SpecialIds(vision_start=1, vision_end=2, video_pad=3)


def encode(s: str) -> list[int]:
    # fake tokenizer: one id per character, offset past the special ids
    return [100 + ord(c) for c in s]


def expanded(grids, stamps, prefix=(10, 11), suffix=(12,)):
    out = list(prefix)
    for (_t, h, w), ts in zip(grids, stamps, strict=True):
        for s in ts:
            out += encode(f"<{s:.1f} seconds>") + [1] + [3] * (h * w // 4) + [2]
        out.append(50)
    return out + list(suffix)


def test_frame_runs_and_summary():
    ids = [9, 1, 3, 3, 2, 8, 1, 3, 3, 2, 3]
    runs = G.frame_runs(ids, SP)
    assert runs == [G.FrameRun(1, 2), G.FrameRun(6, 2)]
    s = G.summarize_ids(ids, SP)
    assert s["frame_groups"] == 2 and s["pads_per_group"] == [2]
    assert s["vision_start"] == 2 and s["video_pad"] == 5
    assert s["bare_video_pad_positions"] == [10]


def test_frame_runs_malformed():
    with pytest.raises(ValueError):
        G.frame_runs([1, 3, 3], SP)
    with pytest.raises(ValueError):
        G.frame_runs([3, 2], SP)
    assert "structure_error" in G.summarize_ids([1, 3], SP)


def test_vllm_timestamps_pad_and_average():
    assert G.vllm_timestamps([0, 1, 2, 3], 2.0, 2) == [0.25, 1.25]
    # odd length: last index repeated
    assert G.vllm_timestamps([0, 2, 4], 4, 2) == [0.25, 1.0]


def test_simulate_expansion_matches_layout():
    grids = [[2, 4, 4], [1, 2, 2]]
    stamps = [[0.25, 1.25], [3.0]]
    prompt = [10, 11, 1, 3, 2, 50, 1, 3, 2, 50, 12]
    out = G.simulate_vllm_expansion(prompt, grids, stamps, encode, SP, 2)
    assert out == expanded(grids, stamps)
    assert G.grids_match_runs(out, grids, SP, 2)
    assert not G.grids_match_runs(out, [[2, 4, 4], [1, 4, 4]], SP, 2)


def test_simulate_expansion_bare_pad_is_not_a_target():
    with pytest.raises(RuntimeError, match="Found 0 video placeholders but expected 1"):
        G.simulate_vllm_expansion([10, 3, 12], [[1, 2, 2]], [[0.0]], encode, SP, 2)
    with pytest.raises(RuntimeError, match="timestamps"):
        G.simulate_vllm_expansion([1, 3, 2], [[2, 2, 2]], [[0.0]], encode, SP, 2)


def test_compare_pass_and_fail_details():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    ok = G.compare(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf,
                   hf_grids=grids, vllm_ids=list(hf), vllm_grids=grids, sp=SP, merge_size=2)
    assert ok.passed and ok.first_diff is None
    bad_ids = list(hf)
    bad_ids[5] = 999
    bad = G.compare(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf,
                    hf_grids=grids, vllm_ids=bad_ids, vllm_grids=grids, sp=SP, merge_size=2)
    assert not bad.passed and bad.first_diff == 5 and bad.grids_equal
    short = G.compare(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf,
                      hf_grids=grids, vllm_ids=hf[:-1], vllm_grids=[[2, 2, 4]], sp=SP,
                      merge_size=2)
    assert short.first_diff == len(hf) - 1 and not short.grids_equal
    err = G.compare(case_id="c", index=0, source="s", variant="bare", hf_ids=hf,
                    hf_grids=grids, vllm_ids=None, vllm_grids=None, sp=SP, merge_size=2,
                    error="RuntimeError: x")
    d = err.to_dict()
    assert d["passed"] is False and d["error"] == "RuntimeError: x" and d["vllm"] == {}


def test_sanitize_error_elides_payload():
    exc = ValueError("Failed to apply Qwen3VLProcessor on data={'text': 'SECRET'} with kwargs")
    msg = G.sanitize_error(exc)
    assert "SECRET" not in msg and msg.startswith("ValueError: Failed to apply")
    assert G.sanitize_error(RuntimeError("a\nb")) == "RuntimeError: a"


def test_report_and_verdicts():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    rows = []
    for case, ids in (("a", hf), ("b", hf[:-1])):
        for idx in (0, 1):
            rows.append(G.compare(case_id=case, index=idx, source="syn", variant="wrapped",
                                  hf_ids=hf, hf_grids=grids,
                                  vllm_ids=ids if idx else hf, vllm_grids=grids, sp=SP,
                                  merge_size=2).to_dict())
        rows.append(G.compare(case_id=case, index=0, source="syn", variant="bare", hf_ids=hf,
                              hf_grids=grids, vllm_ids=None, vllm_grids=None, sp=SP,
                              merge_size=2, error="RuntimeError: 0 placeholders").to_dict())
    assert G.case_verdicts(rows, "wrapped") == {"a": True, "b": False}
    assert G.case_verdicts(rows, "bare") == {"a": False, "b": False}
    md = G.render_report(rows)
    head = md.splitlines()[0]
    assert head == "| case | wrapped | bare |"
    assert "| a | PASS (2/2) | FAIL (0/1) |" in md
    assert "| b | FAIL (1/2) | FAIL (0/1) |" in md
    assert "RuntimeError: 0 placeholders" in md


def test_detail_table_is_well_formed():
    grids = [[1, 2, 2]]
    hf = expanded(grids, [[0.0]])
    row = G.compare(case_id="a", index=0, source="syn", variant="wrapped", hf_ids=hf,
                    hf_grids=grids, vllm_ids=hf, vllm_grids=grids, sp=SP,
                    merge_size=2).to_dict()
    detail = G.render_report([row]).split("\n\n", 1)[1].splitlines()
    assert detail[1] == "|" + "---|" * 12
    assert {line.count("|") for line in detail} == {13}
