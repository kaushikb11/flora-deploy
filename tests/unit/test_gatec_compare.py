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
                   hf_grids=grids, vllm_ids=list(hf), vllm_grids=grids, sp=SP, merge_size=2,
                   pixels_equal=[True], generate_ids_equal=True)
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


def test_sanitize_error_is_an_allowlist():
    exc = ValueError("Failed to apply Qwen3VLProcessor on data={'text': 'SECRET'} with kwargs")
    assert G.sanitize_error(exc) == "ValueError: Failed to apply Qwen3VLProcessor [rest elided]"
    bare = AssertionError("Failed to apply prompt replacement for mm_items['video'][0]")
    assert G.sanitize_error(bare) == ("AssertionError: Failed to apply prompt replacement for "
                                      "mm_items['video'][0]")
    # unknown formats lose the whole message, whatever payload shape they use
    for msg in ("prompt text SECRET", "Bad input: {'prompt': 'SECRET'}", "a\nSECRET",
                "SECRET Failed to apply x"):
        out = G.sanitize_error(RuntimeError(msg))
        assert out == "RuntimeError: [message elided]" and "SECRET" not in out
    assert G.sanitize_error(KeyError()) == "KeyError"


def test_pass_requires_pixels_generate_and_no_bare_pads():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    kw = dict(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf, hf_grids=grids,
              vllm_ids=list(hf), vllm_grids=grids, sp=SP, merge_size=2)
    assert G.compare(**kw, pixels_equal=[True]).passed
    assert G.compare(**kw, pixels_equal=[True], generate_ids_equal=None).passed
    # ids and grids equal, pixels not: FAIL (e.g. frame order or resample changed)
    bad = G.compare(**kw, pixels_equal=[False])
    assert bad.ids_equal and bad.grids_equal and not bad.passed
    assert bad.to_dict()["pixels_ok"] is False and bad.to_dict()["passed"] is False
    assert not G.compare(**kw).passed  # pixels not compared
    assert not G.compare(**kw, pixels_equal=[]).passed
    assert not G.compare(**kw, pixels_equal=[True, True]).passed  # count != videos
    assert not G.compare(**kw, pixels_equal=[True], generate_ids_equal=False).passed
    stray = hf[:3] + [3] + hf[3:]  # a bare video_pad outside any frame run, on both sides
    s = G.compare(**{**kw, "hf_ids": stray, "vllm_ids": stray}, pixels_equal=[True])
    assert s.ids_equal and s.runs_match_grid and not s.passed


def test_pixel_digests_catch_frame_order_and_resample():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    # two videos: (T=2, H=2, W=2) and (T=1, H=2, W=2) -> 8 + 4 rows of patch vectors
    grids = [[2, 2, 2], [1, 2, 2]]
    hf = torch.rand(12, 6, generator=g) * 2 - 1
    digests = G.video_pixel_digests(hf, grids)
    assert [d["rows"] for d in digests] == [8, 4]
    v1, v2 = hf[:8].clone(), hf[8:].clone()
    assert G.pixels_equal_per_video(digests, [v1, v2]) == [True, True]
    assert G.pixels_equal_per_video(digests, [v1.bfloat16(), v2.bfloat16()]) == [True, True]
    # frames in the wrong order (temporal groups swapped): same shape, same grid, FAIL
    swapped = torch.cat([v1[4:], v1[:4]])
    assert G.pixels_equal_per_video(digests, [swapped, v2]) == [False, True]
    # videos swapped across items
    assert G.pixels_equal_per_video(digests, [v2, v1]) == [False, False]
    # a different resample filter: values move by more than a bf16 step
    assert G.pixels_equal_per_video(digests, [v1, (v2 + 0.02).bfloat16()]) == [True, False]
    assert G.pixels_equal_per_video(digests, [v1.half(), v2]) == [False, True]  # other dtype
    assert G.pixels_equal_per_video(digests, [v1]) == [False, False]
    with pytest.raises(ValueError, match="pixel rows"):
        G.video_pixel_digests(hf[:10], grids)


def test_report_and_verdicts():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    rows = []
    for case, ids in (("a", hf), ("b", hf[:-1])):
        for idx in (0, 1):
            rows.append(G.compare(case_id=case, index=idx, source="syn", variant="wrapped",
                                  hf_ids=hf, hf_grids=grids,
                                  vllm_ids=ids if idx else hf, vllm_grids=grids, sp=SP,
                                  merge_size=2, pixels_equal=[True]).to_dict())
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
    assert detail[1] == "|" + "---|" * 14
    assert {line.count("|") for line in detail} == {15}
    assert "| = | = | = | - | - |" in detail[2]  # pixels/gen not given
    row = G.compare(case_id="a", index=0, source="syn", variant="wrapped", hf_ids=hf,
                    hf_grids=grids, vllm_ids=hf, vllm_grids=grids, sp=SP, merge_size=2,
                    pixels_equal=[False], generate_ids_equal=True).to_dict()
    assert "| = | = | = | ≠ | = |" in G.render_report([row])
