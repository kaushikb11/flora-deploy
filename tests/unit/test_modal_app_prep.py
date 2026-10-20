"""Import the Modal app and images and drive ``flora prepare`` with Modal faked out."""

import contextlib
import json

import modal
from typer.testing import CliRunner

from flora_deploy import cli
from flora_deploy.config.pins import PINS, RUNTIME
from flora_deploy.layout import CACHE_MOUNT


def test_images_build_lazily_with_pins():
    from flora_deploy.modal_app import images

    for name in ("img_cpu", "img_hfref", "img_vllm"):
        assert isinstance(getattr(images, name), modal.Image)
    assert images.img_cpu is images.cpu_image()
    assert f"torch=={RUNTIME['torch_reference']}" in images.HFREF_PACKAGES
    assert "huggingface_hub==1.33.0" in images.HFREF_PACKAGES
    assert f"vllm=={RUNTIME['vllm']}" in images.VLLM_PACKAGES
    assert f"Pillow=={RUNTIME['pillow']}" in images.CPU_PACKAGES
    assert str(CACHE_MOUNT) == "/cache"


def test_prep_app_imports_without_network():
    from flora_deploy.modal_app import prep_app

    assert prep_app.app.name == "flora-prep"
    assert isinstance(prep_app.weights, modal.Volume)
    assert prep_app.CPU_CORES <= 8


class FakeFn:
    def __init__(self):
        self.mapped = None

    def map(self, keys, **kw):
        self.mapped = list(keys)
        for k in self.mapped:
            if k == "tempseg_base":
                yield RuntimeError("x")
            else:
                yield {"key": k, "status": "prepared", "ok": k != "captioner", "revision": "r",
                       "files": 3, "bytes": 2**30, "receipt": f"/weights/snapshots/{k}/r.json",
                       "timings_s": {"download": 1.0, "hash": 2.0},
                       "failures": [] if k != "captioner" else
                       [{"name": "lfs_sha256", "ok": False, "detail": "bad"}]}


class FakeApp:
    def __init__(self):
        self.ran = 0

    @contextlib.contextmanager
    def run(self):
        self.ran += 1
        yield self


def fake_modal(monkeypatch):
    from flora_deploy.modal_app import prep_app

    fn, fapp = FakeFn(), FakeApp()
    monkeypatch.setattr(prep_app, "prepare_pin", fn)
    monkeypatch.setattr(prep_app, "app", fapp)
    monkeypatch.setattr(modal, "enable_output", contextlib.nullcontext)
    return fn, fapp


def test_prepare_maps_all_pins(monkeypatch):
    fn, fapp = fake_modal(monkeypatch)
    result = CliRunner().invoke(cli.app, ["prepare", "--only", "tempseg", "--only", "rewarder"])
    assert result.exit_code == 0, result.output
    assert fn.mapped == ["tempseg", "rewarder"] and fapp.ran == 1
    assert "prepared" in result.output

    result = CliRunner().invoke(cli.app, ["prepare", "--json"])
    assert fn.mapped == [p.key for p in PINS]
    assert result.exit_code == 1  # the fake captioner fails
    rows = json.loads(result.output[result.output.index("["):])
    assert [r["key"] for r in rows] == fn.mapped
    assert "RuntimeError" in next(r["error"] for r in rows if r["key"] == "tempseg_base")


def test_prepare_reports_failures(monkeypatch):
    fake_modal(monkeypatch)
    result = CliRunner().invoke(cli.app, ["prepare", "--only", "captioner"])
    assert result.exit_code == 1
    assert "lfs_sha256 failed" in result.output


def test_prepare_rejects_unknown_key(monkeypatch):
    fn, _ = fake_modal(monkeypatch)
    result = CliRunner().invoke(cli.app, ["prepare", "--only", "nope"])
    assert result.exit_code == 2 and fn.mapped is None


def test_dry_run_respects_only(monkeypatch):
    class Api:
        def list_repo_tree(self, repo_id, *, revision, recursive):
            return []

    monkeypatch.setattr(cli, "_hf_api", Api)
    result = CliRunner().invoke(cli.app, ["prepare", "--dry-run", "--json", "--only", "rewarder"])
    assert result.exit_code == 0, result.output
    assert [r["key"] for r in json.loads(result.output)] == ["rewarder"]
