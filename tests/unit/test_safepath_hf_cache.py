"""``media.safepath``: a dataset inside a Hugging Face cache snapshot (files are symlinks into the
repo's ``blobs/``) resolves; nothing else outside the root does (review 11b F1)."""

from __future__ import annotations

import pytest

from flora_deploy.media.safepath import UnsafePath, _allowed_roots, contain, is_within


def _cache(tmp_path, repo="datasets--org--data"):
    base = tmp_path / "hub" / repo
    blob = base / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"video")
    snap = base / "snapshots" / "rev1"
    (snap / "videos").mkdir(parents=True)
    (snap / "videos" / "ep0.mp4").symlink_to(blob)
    return snap, blob


def test_snapshot_symlink_into_its_own_blobs_resolves(tmp_path):
    snap, blob = _cache(tmp_path)
    assert contain(snap, "videos/ep0.mp4").resolve() == blob.resolve()
    assert is_within(snap, snap / "videos" / "ep0.mp4")


def test_symlink_into_another_repos_blobs_is_refused(tmp_path):
    snap, _ = _cache(tmp_path)
    _other_snap, other_blob = _cache(tmp_path, "datasets--evil--other")
    (snap / "videos" / "ep1.mp4").symlink_to(other_blob)
    with pytest.raises(UnsafePath):
        contain(snap, "videos/ep1.mp4")


def test_symlink_to_an_arbitrary_file_is_refused(tmp_path):
    snap, _ = _cache(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    (snap / "videos" / "ep2.mp4").symlink_to(secret)
    with pytest.raises(UnsafePath):
        contain(snap, "videos/ep2.mp4")


def test_a_plain_directory_gets_no_extra_root(tmp_path):
    root = tmp_path / "data"
    (root / "videos").mkdir(parents=True)
    outside = tmp_path / "blobs" / "x"
    outside.parent.mkdir()
    outside.write_bytes(b"v")
    (root / "videos" / "ep.mp4").symlink_to(outside)
    with pytest.raises(UnsafePath):
        contain(root, "videos/ep.mp4")


# --- review 12 F3: a shipped ``blobs`` symlink must not widen the allowed roots ---------------


def _fake_repo(tmp_path, blobs_target):
    secret = tmp_path / "secret" / "creds.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("TOP SECRET")
    repo = tmp_path / "dl" / "datasets--evil"
    snap = repo / "snapshots" / "r"
    (snap / "videos").mkdir(parents=True)
    (repo / "blobs").symlink_to(blobs_target(secret))
    return snap, secret


@pytest.mark.parametrize("target", [lambda s: "/", lambda s: s.parent],
                         ids=["blobs->/", "blobs->secret-dir"])
def test_symlinked_blobs_dir_is_not_an_allowed_root(tmp_path, target):
    snap, secret = _fake_repo(tmp_path, target)
    (snap / "videos" / "a.mp4").symlink_to(secret)
    assert _allowed_roots(snap) == [snap.resolve()]
    with pytest.raises(UnsafePath):
        contain(snap, "videos/a.mp4")
    assert not is_within(snap, secret)


def test_blob_symlink_pointing_outside_is_refused(tmp_path):
    snap, blob = _cache(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    link = blob.parent / "deadbeef"
    link.symlink_to(secret)
    (snap / "videos" / "ep3.mp4").symlink_to(link)
    with pytest.raises(UnsafePath):
        contain(snap, "videos/ep3.mp4")


def test_only_flat_blob_files_count(tmp_path):
    snap, blob = _cache(tmp_path)
    nested = blob.parent / "sub" / "x"
    nested.parent.mkdir()
    nested.write_bytes(b"v")
    (snap / "videos" / "ep4.mp4").symlink_to(nested)
    with pytest.raises(UnsafePath):
        contain(snap, "videos/ep4.mp4")
