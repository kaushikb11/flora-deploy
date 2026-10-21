from fractions import Fraction
from pathlib import Path

import pytest

from flora_deploy.media import synth
from flora_deploy.media.cuts import stream_copy_cut
from flora_deploy.media.index import ScanCache, index_episode, index_stream
from flora_deploy.media.sources import (
    LeRobotV3Source,
    LeRobotV21Source,
    Mp4DirSource,
    SourceError,
    open_source,
    resolve_camera_map,
    verify_episode,
)
from flora_deploy.media.synth import Spec

KEYS = ("observation.images.top", "observation.images.left", "observation.images.right")
CMAP = dict(zip(KEYS, ("main", "left_wrist", "right_wrist"), strict=True))


def _episodes(lengths):
    return [
        {k: Spec(n, seed=e * 10 + c) for c, k in enumerate(KEYS)} for e, n in enumerate(lengths)
    ]


def test_lerobot_v21_tree(tmp_path):
    root = synth.lerobot_v21_tree(tmp_path / "ds", _episodes([12, 9]), ["pick", "place"])
    src = open_source(root, camera_map=CMAP)
    assert isinstance(src, LeRobotV21Source)
    eps = src.list_episodes()
    assert [e.episode_id for e in eps] == ["episode_000000", "episode_000001"]
    assert eps[1].task == "place"
    ref = eps[1].cameras["left_wrist"]
    assert ref.uri.endswith("videos/chunk-000/observation.images.left/episode_000001.mp4")
    assert ref.start_s is None and ref.end_s is None
    ix = index_episode(eps[1].episode_id, src.name, eps[1].cameras, task=eps[1].task)
    assert len(ix.streams["main"].pts_us) == 9
    assert ix.envelope.ok and ix.oracle_eligible


def test_lerobot_v3_tree_ranges_and_cuts(tmp_path):
    root = synth.lerobot_v3_tree(tmp_path / "ds", _episodes([10, 14, 8]), ["stack cups"])
    src = open_source(root, camera_map=CMAP)
    assert isinstance(src, LeRobotV3Source)
    eps = src.list_episodes()
    assert len(eps) == 3 and eps[2].task == "stack cups"
    scans: dict = {}
    for ep, n in zip(eps, (10, 14, 8), strict=True):
        ref = ep.cameras["main"]
        assert ref.uri.endswith("videos/observation.images.top/chunk-000/file-000.mp4")
        ix = index_episode(ep.episode_id, src.name, ep.cameras, scans=scans, profile="yam")
        assert ix.envelope.ok, ix.envelope.reasons
        assert all(len(s.pts_us) == n for s in ix.streams.values())
        assert ix.streams["main"].rate == 30 and ix.streams["main"].cfr
        cut = stream_copy_cut(ref, tmp_path / f"{ep.episode_id}.mp4")
        assert index_stream(type(ref)(str(cut))).pts_us == ix.streams["main"].pts_us
    assert len(scans) == 3  # one demux pass per concatenated file


def test_mp4_dir_layouts(tmp_path):
    synth.encode(tmp_path / "a.mp4", Spec(5))
    (tmp_path / "a.txt").write_text("wipe the table\n")
    for role in ("main", "left_wrist", "right_wrist"):
        synth.encode(tmp_path / "b" / f"{role}.mp4", Spec(5))
    (tmp_path / "b" / "task.txt").write_text("fold")
    (tmp_path / ".hidden.mp4").write_bytes(b"")
    eps = {e.episode_id: e for e in Mp4DirSource(tmp_path).list_episodes()}
    assert set(eps) == {"a", "b"}
    assert eps["a"].task == "wipe the table" and set(eps["a"].cameras) == {"main"}
    assert set(eps["b"].cameras) == {"main", "left_wrist", "right_wrist"}
    assert eps["b"].task == "fold"


def test_mp4_dir_custom_keys_need_map(tmp_path):
    for key in ("top", "l", "r"):
        synth.encode(tmp_path / "ep" / f"{key}.mp4", Spec(4))
    with pytest.raises(SourceError):
        Mp4DirSource(tmp_path).list_episodes()
    eps = Mp4DirSource(
        tmp_path, camera_map={"top": "main", "l": "left_wrist", "r": "right_wrist"}
    ).list_episodes()
    assert eps[0].cameras["left_wrist"].uri.endswith("l.mp4")


def test_camera_map_validation():
    assert resolve_camera_map(["cam"], None) == {"cam": "main"}
    with pytest.raises(SourceError):
        resolve_camera_map(["a", "b"], None)
    with pytest.raises(SourceError):
        resolve_camera_map(["a", "b"], {"a": "main", "b": "main"})
    with pytest.raises(SourceError):
        resolve_camera_map(["a"], {"a": "top"})
    with pytest.raises(SourceError):
        resolve_camera_map(["a"], {"x": "main"})
    with pytest.raises(SourceError):
        resolve_camera_map(["a"], {"a": "left_wrist"})
    # unmapped keys (e.g. depth) are ignored
    assert resolve_camera_map(["a", "depth"], {"a": "main"}) == {"a": "main"}


def test_version_mismatch(tmp_path):
    root = synth.lerobot_v21_tree(tmp_path / "ds", _episodes([4]), ["t"])
    with pytest.raises(SourceError):
        LeRobotV3Source(root, camera_map=CMAP)


def test_vfr_episode_index_is_not_cfr(tmp_path):
    p = synth.encode(
        tmp_path / "v.mp4",
        Spec(12, time_base=Fraction(1, 90000), pts=[i * 3000 + (i % 2) * 500 for i in range(12)]),
    )
    ix = index_stream(type(Mp4DirSource(tmp_path).list_episodes()[0].cameras["main"])(str(p)))
    assert not ix.cfr and ix.rate != 30


# ---------------------------------------------------------------------------------------------
# Metadata is verified against the video, not trusted (golden-candidates §3.3)


def _rewrite_v3(root, edit):
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = root / "meta/episodes/chunk-000/file-000.parquet"
    cols = pq.read_table(f).to_pydict()
    edit(cols)
    pq.write_table(pa.Table.from_pydict(cols), f)


def test_v3_verify_episode_ok_and_wrong_file(tmp_path):
    root = synth.lerobot_v3_tree(tmp_path / "ds", _episodes([10, 14]), ["t"])
    src = open_source(root, camera_map=CMAP)
    cache = ScanCache()
    for ep in src.list_episodes():
        verify_episode(ep, scans=cache)
    assert cache.scans_done == 3

    key = "videos/observation.images.top"
    end = src.list_episodes()[1].cameras["main"].end_s

    def past_end(cols):  # aloha_mobile_wipe_wine ep 43: the range starts where the file ends
        cols[f"{key}/from_timestamp"][1] = end
        cols[f"{key}/to_timestamp"][1] = end + 14 / 30

    _rewrite_v3(root, past_end)
    ep = open_source(root, camera_map=CMAP).list_episodes()[1]
    with pytest.raises(SourceError, match="wrong file"):
        verify_episode(ep)


def test_v3_verify_episode_length_mismatch(tmp_path):
    root = synth.lerobot_v3_tree(tmp_path / "ds", _episodes([10, 14]), ["t"])
    _rewrite_v3(root, lambda cols: cols["length"].__setitem__(0, 11))
    ep = open_source(root, camera_map=CMAP).list_episodes()[0]
    with pytest.raises(SourceError, match="holds 10 frames"):
        verify_episode(ep)
    verify_episode(ep, check_length=False)


def test_v3_null_column_is_source_error(tmp_path):
    root = synth.lerobot_v3_tree(tmp_path / "ds", _episodes([6, 7]), ["t"])
    key = "videos/observation.images.left/file_index"
    _rewrite_v3(root, lambda cols: cols[key].__setitem__(1, None))
    with pytest.raises(SourceError, match="null"):
        open_source(root, camera_map=CMAP).list_episodes()


def test_v3_reads_only_needed_columns(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    root = synth.lerobot_v3_tree(tmp_path / "ds", _episodes([6]), ["t"])
    _rewrite_v3(root, lambda cols: cols.__setitem__("stats/action/mean", [[0.0] * 64]))
    seen = []
    real = pq.read_table
    monkeypatch.setattr(pq, "read_table", lambda f, **kw: seen.append(kw) or real(f, **kw))
    src = open_source(root, camera_map={"observation.images.top": "main"})
    assert len(src.list_episodes()) == 1
    cols = seen[0]["columns"]
    assert "stats/action/mean" not in cols
    assert "videos/observation.images.left/file_index" not in cols
    assert "videos/observation.images.top/from_timestamp" in cols


def test_v21_missing_video_is_source_error(tmp_path):
    root = synth.lerobot_v21_tree(tmp_path / "ds", _episodes([5, 6]), ["t"])
    ep = open_source(root, camera_map=CMAP).list_episodes()[1]
    verify_episode(ep)
    Path(ep.cameras["right_wrist"].uri).unlink()
    with pytest.raises(SourceError, match="missing"):
        verify_episode(ep)


def test_mp4_dir_camera_stem_collision(tmp_path):
    synth.encode(tmp_path / "ep" / "main.mp4", Spec(4))
    synth.encode(tmp_path / "ep" / "main.mov", Spec(4))
    with pytest.raises(SourceError, match="two files"):
        Mp4DirSource(tmp_path).list_episodes()
