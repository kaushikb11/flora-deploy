"""Untrusted inputs (security review 11b): dataset metadata, decoder input, Dict entries, keys.

Every case is a hostile value that used to be accepted. No secrets, no network, no Modal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import safe_key
from flora_deploy.media.safepath import UnsafePath, contain, format_template
from flora_deploy.media.sources import LeRobotV3Source, LeRobotV21Source, Mp4DirSource
from flora_deploy.media.sources.base import SourceError
from flora_deploy.media.types import SourceRef
from flora_deploy.workers_impl.reward import VideoResolver

V21_DEFAULT = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CAM = "observation.images.top"


def v21_tree(root: Path, video_path: str, keys=(CAM,)) -> Path:
    (root / "meta").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "chunks_size": 1000, "video_path": video_path,
            "features": {k: {"dtype": "video"} for k in keys}}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick"], "length": 10}) + "\n")
    return root


def v3_tree(root: Path, video_path: str) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "video_path": video_path,
            "features": {CAM: {"dtype": "video"}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    cols = {"episode_index": [0], "tasks": [["pick"]], "length": [10],
            f"videos/{CAM}/chunk_index": [0], f"videos/{CAM}/file_index": [0],
            f"videos/{CAM}/from_timestamp": [0.0], f"videos/{CAM}/to_timestamp": [1.0]}
    pq.write_table(pa.Table.from_pydict(cols), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


# --- F1: video_path templates ---------------------------------------------------------------

HOSTILE_TEMPLATES = [
    "../../../../outside/{video_key}/{episode_index.__class__.__name__}.mp4",  # review probe
    "{episode_index.__class__}.mp4",  # attribute access
    "{video_key[0]}.mp4",  # index access
    "{video_key!r}.mp4",  # conversion
    "{}.mp4",  # positional
    "{unknown_field}.mp4",  # unknown field
    "{episode_index:{episode_chunk}}.mp4",  # nested spec
    "/etc/{video_key}.mp4",  # absolute
    "videos/../../{video_key}.mp4",  # escape
    "videos//{video_key}.mp4",  # empty component
    "videos\\{video_key}.mp4",  # backslash
    "videos/{video_key}.mp4\0x",  # NUL
    "videos/{episode_index",  # malformed
]


@pytest.mark.parametrize("template", HOSTILE_TEMPLATES)
def test_v21_refuses_hostile_video_path(tmp_path, template):
    root = v21_tree(tmp_path / "ds", template)
    with pytest.raises(SourceError):
        LeRobotV21Source(root).list_episodes()


@pytest.mark.parametrize("template", [*HOSTILE_TEMPLATES,
                                      "videos/{video_key}/{episode_index:06d}.mp4"])
def test_v3_refuses_hostile_video_path(tmp_path, template):
    """v3 rows have no episode_index field in the template (LeRobot v3 DEFAULT_VIDEO_PATH)."""
    root = v3_tree(tmp_path / "ds", template)
    with pytest.raises(SourceError):
        LeRobotV3Source(root).list_episodes()


def test_default_templates_still_work(tmp_path):
    from flora_deploy.media.sources.lerobot_v3 import DEFAULT_VIDEO_PATH as V3

    eps = LeRobotV21Source(v21_tree(tmp_path / "a", V21_DEFAULT)).list_episodes()
    assert eps[0].cameras["main"].uri == str(
        tmp_path / "a" / f"videos/chunk-000/{CAM}/episode_000000.mp4")
    eps = LeRobotV3Source(v3_tree(tmp_path / "b", V3)).list_episodes()
    assert eps[0].cameras["main"].uri == str(
        tmp_path / "b" / f"videos/{CAM}/chunk-000/file-000.mp4")


@pytest.mark.parametrize("key", ["../x", "a/b", "..", ".hidden", "a\nb"])
def test_v21_refuses_hostile_camera_key(tmp_path, key):
    root = v21_tree(tmp_path / "ds", V21_DEFAULT, keys=(key,))
    with pytest.raises(SourceError):
        LeRobotV21Source(root, camera_map={key: "main"}).list_episodes()


def test_template_symlink_escape_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = v21_tree(tmp_path / "ds", "videos/{video_key}.mp4")
    (root / "videos").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceError):
        LeRobotV21Source(root).list_episodes()


def test_format_template_is_strict():
    assert format_template("a/{n:03d}/{k}.mp4", n=7, k="cam") == "a/007/cam.mp4"
    with pytest.raises(UnsafePath):
        format_template("{k:>300}.mp4", k="cam")  # a spec only ints get, and short
    with pytest.raises(UnsafePath):
        format_template("{k:03d}.mp4", k="cam")


def test_contain_resolves_symlinks(tmp_path):
    root = tmp_path / "root"
    (root / "d").mkdir(parents=True)
    (root / "d" / "f.mp4").write_bytes(b"")
    assert contain(root, "d/f.mp4") == root / "d" / "f.mp4"
    (root / "link.mp4").symlink_to("/etc/hosts")
    with pytest.raises(UnsafePath):
        contain(root, "link.mp4")


def test_mp4_dir_refuses_symlinked_files_outside_root(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("not a task")
    root = tmp_path / "ds"
    root.mkdir()
    (root / "ep1.mp4").write_bytes(b"")
    (root / "ep1.txt").symlink_to(secret)
    with pytest.raises(SourceError):
        Mp4DirSource(root).list_episodes()
    (root / "ep1.txt").unlink()
    (root / "ep2.mp4").symlink_to(secret)
    with pytest.raises(SourceError):
        Mp4DirSource(root).list_episodes()


# --- F1: VideoResolver ----------------------------------------------------------------------

@pytest.mark.parametrize("uri", ["/etc/passwd", "../../etc/passwd", "x/../../../etc/passwd",
                                 "", "x//y.mp4", "x\\y.mp4"])
def test_video_resolver_refuses_escapes(tmp_path, uri):
    r = VideoResolver(media_root=str(tmp_path / "media"))
    with pytest.raises(UnsafePath):
        r.local(uri)
    with pytest.raises(ValueError):  # the worker treats it as a per-episode data error
        r.resolve("u", "main", SourceRef(uri))


def test_video_resolver_absolute_inside_media_root_ok(tmp_path):
    media = tmp_path / "media"
    (media / "x").mkdir(parents=True)
    (media / "x" / "a.mp4").write_bytes(b"")
    r = VideoResolver(media_root=str(media))
    assert r.local(str(media / "x" / "a.mp4")) == str(media / "x" / "a.mp4")
    assert r.local("x/a.mp4") == str(media / "x" / "a.mp4")
    (media / "evil.mp4").symlink_to("/etc/hosts")
    with pytest.raises(UnsafePath):
        r.local("evil.mp4")
    with pytest.raises(UnsafePath):  # an absolute uri with no media root to contain it
        VideoResolver().local(str(media / "x" / "a.mp4"))


def test_video_resolver_hub_branch_refuses_escapes(tmp_path):
    fetched = []

    def fetch(repo, rev, rel, dest):
        fetched.append(rel)
        return str(Path(dest) / rel)

    r = VideoResolver(media_root="hf://datasets/o/n@" + "a" * 40, hub_dir=tmp_path / "hub",
                      fetch=fetch)
    for uri in ("../../../etc/passwd", "/etc/passwd"):
        with pytest.raises(UnsafePath):
            r.local(uri)
    assert fetched == []


# --- F5: decoder input whitelists -----------------------------------------------------------

def test_open_input_refuses_playlists_and_protocols(tmp_path):
    from flora_deploy.media.avsafe import open_input
    from flora_deploy.media.synth import Spec, encode

    real = encode(tmp_path / "real.mp4", Spec(frames=3))
    with open_input(real) as c:
        assert len(c.streams.video) == 1
    (tmp_path / "cc.mp4").write_text(f"ffconcat version 1.0\nfile '{real}'\n")
    (tmp_path / "hls.mp4").write_text(f"#EXTM3U\n#EXTINF:1,\n{real}\n#EXT-X-ENDLIST\n")
    for bad in (tmp_path / "cc.mp4", tmp_path / "hls.mp4", f"concat:{real}",
                "http://127.0.0.1:9/x.mp4"):
        with pytest.raises((ValueError, OSError)):
            with open_input(bad):
                pass


def test_no_bare_av_open_on_inputs():
    """Every read of a dataset file goes through ``media.avsafe.open_input``."""
    import flora_deploy

    base = Path(flora_deploy.__file__).parent
    reads = ("media/index.py", "media/cuts.py", "accel/framecache.py", "local/engines.py")
    for rel in reads:
        text = (base / rel).read_text()
        bare = [line for line in text.splitlines()
                if "av.open(" in line and '"w"' not in line]
        assert not bare, (rel, bare)


# --- F4: load_factory -----------------------------------------------------------------------

def test_load_factory_allowlist():
    from flora_deploy.modal_app.orchestrator_app import FACTORIES, load_factory
    from flora_deploy.sdk import FACTORY, deployed_deps

    assert FACTORY in FACTORIES
    assert load_factory(FACTORY) is deployed_deps
    for bad in ("builtins:len", "os:system", "flora_deploy.sdk:get_backend",
                "flora_deploy.sdk:deployed_deps\n"):
        with pytest.raises(ValueError, match="allowed"):
            load_factory(bad)


def test_reconcile_all_refuses_unsafe_dict_keys():
    from flora_deploy.modal_app.orchestrator_app import reconcile_all

    called = []
    out = reconcile_all([("../x", "flora_deploy.sdk:deployed_deps")],
                        lambda *a: called.append(a) or {}, lambda k: None)
    assert called == [] and "error" in out["../x"]


# --- F6, F7: keys and listing ---------------------------------------------------------------

@pytest.mark.parametrize("key", ["abc\n", "abc\r\n", "a\nb"])
def test_safe_key_refuses_newlines(key):
    with pytest.raises(ValueError):
        safe_key(key)


def test_local_store_list_refuses_escapes(tmp_path):
    (tmp_path / "outside_secret").mkdir()
    (tmp_path / "outside_secret" / "x.txt").write_text("x")
    s = LocalStore(tmp_path / "store")
    s.put_bytes("jobs/a.json", b"{}")
    for bad in ("../", "../outside_secret/", "/", "jobs/../../"):
        with pytest.raises(ValueError):
            list(s.list(bad))
    assert list(s.list("")) == ["jobs/a.json"]
    assert list(s.list("jobs/")) == ["jobs/a.json"]
    assert list(s.list("jobs/a")) == ["jobs/a.json"]


# --- F13: .gitignore mirrors the license guard ----------------------------------------------

def test_gitignore_covers_license_guard_examples():
    import subprocess

    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    samples = ["vendor/x/chat_template.jinja", "tests/fakes/tokenizer_config.json",
               "a/tokenizer.json", "a/video_preprocessor_config.json", "b/generation_config.json",
               "c/chat_template.json", "lora/adapter_model.bin", "w.pt", "w.pth", "m.ckpt",
               "m.onnx", "m.gguf", "m.safetensors", "docs/reports/r.md", "reports/x.md",
               "x/research_notes/n.md", "tests/fakes/batch_utils/models.py",
               "x/TEMPSEG_PROMPTS.json", "x/LICENSE.pdf", ".cache/flora-deploy/nv/x.py"]
    out = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=root,
                         input="\n".join(samples), capture_output=True, text=True)
    ignored = set(out.stdout.split())
    assert sorted(set(samples) - ignored) == []
