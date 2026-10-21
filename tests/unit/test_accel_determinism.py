"""``accel.determinism``: the opt-in deterministic mode (D-1, ``docs/determinism.md``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flora_deploy.accel import determinism as d


class FakeTorch:
    """Just the torch surface ``accel.determinism`` touches."""

    __version__ = "fake"

    def __init__(self, cuda_initialized: bool = False):
        self._det = False
        self._warn = False
        self.calls: list[tuple] = []
        self.sdp = {"flash": True, "mem_efficient": True, "math": True, "cudnn": True}
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=False, allow_tf32=True),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=False),
                flash_sdp_enabled=lambda: self.sdp["flash"],
                mem_efficient_sdp_enabled=lambda: self.sdp["mem_efficient"],
                math_sdp_enabled=lambda: self.sdp["math"],
                cudnn_sdp_enabled=lambda: self.sdp["cudnn"],
                enable_cudnn_sdp=lambda on: self.sdp.__setitem__("cudnn", bool(on)),
            ),
        )
        self.cuda = SimpleNamespace(is_initialized=lambda: cuda_initialized)
        self.utils = SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=True))

    def use_deterministic_algorithms(self, mode, *, warn_only=False):
        self.calls.append(("use_deterministic_algorithms", mode, warn_only))
        self._det, self._warn = mode, warn_only

    def are_deterministic_algorithms_enabled(self):
        return self._det

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self._warn

    def get_float32_matmul_precision(self):
        return "highest"


def test_container_env():
    assert d.container_env("nvidia") == {}
    assert d.container_env("deterministic") == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    with pytest.raises(d.DeterminismError):
        d.container_env("fast")


def test_nvidia_mode_changes_nothing():
    torch, env = FakeTorch(), {}
    before = d.flags(torch, env)
    rec = d.apply("nvidia", torch=torch, environ=env)
    assert torch.calls == [] and env == {}
    assert rec["flags"] == before
    assert rec["mode"] == "nvidia" and rec["in_effect"] and rec["schema"] == d.DETERMINISM_SCHEMA


def test_deterministic_sets_flags_and_only_disables_cudnn_sdpa():
    torch, env = FakeTorch(), {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    rec = d.apply("deterministic", torch=torch, environ=env)
    assert torch.calls == [("use_deterministic_algorithms", True, False)]
    f = rec["flags"]
    assert f["deterministic_algorithms"] and not f["deterministic_warn_only"]
    assert f["cudnn_deterministic"] and not f["cudnn_benchmark"]
    # NVIDIA's numerics settings are untouched.
    assert f["matmul_allow_tf32"] is False and f["cudnn_allow_tf32"] is True
    # The measured root cause: SDPA's cuDNN backend. The other backends stay allowed.
    assert f["sdpa"] == {"flash": True, "mem_efficient": True, "math": True, "cudnn": False}
    assert f["fill_uninitialized_memory"] is True
    assert rec["in_effect"] and not rec["cublas_env_set_by_apply"]


def test_deterministic_sets_env_before_cuda_init():
    torch, env = FakeTorch(cuda_initialized=False), {}
    rec = d.apply("deterministic", torch=torch, environ=env)
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert rec["cublas_env_set_by_apply"] and rec["in_effect"]


def test_deterministic_refuses_late_or_wrong_env():
    with pytest.raises(d.DeterminismError, match="before CUDA"):
        d.apply("deterministic", torch=FakeTorch(cuda_initialized=True), environ={})
    with pytest.raises(d.DeterminismError, match="needs one of"):
        d.apply("deterministic", torch=FakeTorch(),
                environ={"CUBLAS_WORKSPACE_CONFIG": ":0:0"})
    rec = d.apply("deterministic", torch=FakeTorch(cuda_initialized=True),
                  environ={"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
    assert rec["in_effect"]


def test_verify_detects_drift():
    torch, env = FakeTorch(), {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    rec = d.apply("deterministic", torch=torch, environ=env)
    d.verify(rec, torch, env)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(d.DeterminismError, match="no longer in effect"):
        d.verify(rec, torch, env)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    with pytest.raises(d.DeterminismError, match="changed"):
        d.verify(rec, torch, env)
    torch.backends.cudnn.allow_tf32 = True
    d.verify(rec, torch, env)
    torch.sdp["cudnn"] = True  # cuDNN SDPA back on: no longer deterministic
    with pytest.raises(d.DeterminismError, match="no longer in effect"):
        d.verify(rec, torch, env)
    torch.sdp.update(cudnn=False, math=False)
    with pytest.raises(d.DeterminismError, match="sdpa"):
        d.verify(rec, torch, env)


def test_nvidia_mode_leaves_cudnn_sdpa_on():
    rec = d.apply("nvidia", torch=FakeTorch(), environ={})
    assert rec["flags"]["sdpa"]["cudnn"] is True


def test_provenance_not_in_effect_without_flags():
    rec = d.provenance("deterministic", torch=FakeTorch(), environ={})
    assert rec["in_effect"] is False


def test_real_torch_flags_shape():
    torch = pytest.importorskip("torch")
    f = d.flags(torch, {})
    assert set(f) >= {"deterministic_algorithms", "cudnn_deterministic", "sdpa",
                      "cublas_workspace_config", "optional_kernels"}
