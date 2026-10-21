"""Import the Modal app and images and drive ``flora prepare`` with Modal faked out."""

import contextlib
import json

import modal
from typer.testing import CliRunner

from flora_deploy.cli import prepare as cli
from flora_deploy.config.pins import PINS, RUNTIME
from flora_deploy.layout import CACHE_MOUNT
from flora_deploy.prep.run import SnapshotVerificationError


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
    def __init__(self, noisy=False, detail="bad"):
        self.mapped = None
        self.kwargs = None
        self.noisy = noisy
        self.detail = detail

    def map(self, keys, *, kwargs=None, **kw):
        assert "wrap_returned_exceptions" not in kw  # deprecated no-op in modal 1.6
        assert kw == {"return_exceptions": True}
        self.mapped, self.kwargs = list(keys), kwargs
        if self.noisy:
            print("Downloading model.safetensors: 100%|#####| [hf progress on stdout]")
        for k in self.mapped:
            if k == "tempseg_base":
                yield RuntimeError("x")
            elif k == "rewarder":
                yield SnapshotVerificationError(
                    "rewarder: verification failed after deleting and re-downloading the "
                    "failing files: sizes: 1/2 failed: [/weights/x] (receipt: /w/r.json)")
            else:
                yield {"key": k, "status": "prepared", "ok": k != "captioner", "revision": "r",
                       "files": 3, "bytes": 2**30, "receipt": f"/weights/snapshots/{k}/r.json",
                       "timings_s": {"download": 1.0, "hash": 2.0}, "note": "",
                       "failures": [] if k != "captioner" else
                       [{"name": "lfs_sha256", "ok": False, "detail": self.detail}]}


class FakeApp:
    def __init__(self):
        self.ran = 0

    @contextlib.contextmanager
    def run(self):
        self.ran += 1
        yield self


def fake_modal(monkeypatch, **fn_kw):
    from flora_deploy.modal_app import prep_app

    fn, fapp = FakeFn(**fn_kw), FakeApp()
    calls = []

    @contextlib.contextmanager
    def enable_output():
        calls.append(1)
        print("modal: streaming logs to stdout")
        yield

    monkeypatch.setattr(prep_app, "prepare_pin", fn)
    monkeypatch.setattr(prep_app, "app", fapp)
    monkeypatch.setattr(modal, "enable_output", enable_output)
    monkeypatch.setattr(cli, "_git_provenance", lambda: {"git_commit": "c" * 40,
                                                         "git_dirty": False})
    fn.enable_output_calls = calls
    return fn, fapp


def test_prepare_maps_all_pins(monkeypatch):
    fn, fapp = fake_modal(monkeypatch)
    result = CliRunner().invoke(cli.app, ["prepare", "--only", "tempseg",
                                          "--only", "captioner_code"])
    assert result.exit_code == 0, result.output
    assert fn.mapped == ["tempseg", "captioner_code"] and fapp.ran == 1
    assert fn.kwargs == {"reverify": False,
                         "provenance": {"git_commit": "c" * 40, "git_dirty": False}}
    assert "prepared" in result.output and fn.enable_output_calls == [1]

    CliRunner().invoke(cli.app, ["prepare", "--reverify", "--only", "tempseg"])
    assert fn.kwargs["reverify"] is True


def test_json_mode_stdout_is_pure_json(monkeypatch):
    fn, _ = fake_modal(monkeypatch, noisy=True)
    result = CliRunner().invoke(cli.app, ["prepare", "--json"])
    assert fn.mapped == [p.key for p in PINS]
    assert result.exit_code == 1  # the fake captioner fails
    rows = json.loads(result.stdout)  # the whole of stdout, no slicing
    assert [r["key"] for r in rows] == fn.mapped
    assert fn.enable_output_calls == []
    assert "hf progress" in result.stderr
    err = {r["key"]: r for r in rows if "error" in r}
    assert err["tempseg_base"]["error"] == "RuntimeError: x"
    assert err["rewarder"]["error"].startswith("SnapshotVerificationError: rewarder:")
    assert err["rewarder"]["status"] == "error" and err["rewarder"]["ok"] is False


def test_prepare_reports_failures(monkeypatch):
    fake_modal(monkeypatch)
    result = CliRunner().invoke(cli.app, ["prepare", "--only", "captioner"])
    assert result.exit_code == 1
    assert "lfs_sha256 failed" in result.output


def test_error_text_with_markup_is_escaped(monkeypatch):
    from rich.console import Console

    fake_modal(monkeypatch, detail="bad [/weights/x] path [red]")
    monkeypatch.setattr(cli, "console", Console(width=400))
    result = CliRunner().invoke(cli.app, ["prepare", "--only", "captioner", "--only", "rewarder"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert "bad [/weights/x] path [red]" in result.output
    # The raised per-pin error is shown in full, literally.
    assert "rewarder: SnapshotVerificationError: rewarder: verification failed" in result.output
    assert "[/weights/x] (receipt: /w/r.json)" in result.output


def test_prepare_rejects_unknown_key(monkeypatch):
    fn, _ = fake_modal(monkeypatch)
    result = CliRunner().invoke(cli.app, ["prepare", "--only", "[/nope]"])
    assert result.exit_code == 2 and fn.mapped is None
    assert "[/nope]" in result.output


def test_dry_run_respects_only(monkeypatch):
    class Api:
        def list_repo_tree(self, repo_id, *, revision, recursive):
            return []

    monkeypatch.setattr(cli, "_hf_api", Api)
    result = CliRunner().invoke(cli.app, ["prepare", "--dry-run", "--json", "--only", "rewarder"])
    assert result.exit_code == 0, result.output
    assert [r["key"] for r in json.loads(result.output)] == ["rewarder"]


def test_git_provenance_reads_local_checkout():
    prov = cli._git_provenance()
    assert set(prov) == {"git_commit", "git_dirty"}
    if prov["git_commit"] is not None:
        assert len(prov["git_commit"]) == 40 and isinstance(prov["git_dirty"], bool)
