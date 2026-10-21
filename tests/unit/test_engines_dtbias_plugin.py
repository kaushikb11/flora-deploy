"""The FP32 dt_bias plugin is off by default, idempotent, and keeps vLLM's weight loader."""

from __future__ import annotations

import importlib.metadata
import sys

import pytest

from flora_deploy.engines.offline_llm import EngineSpec
from flora_deploy.engines.patches import dt_bias_fp32 as D

pytestmark = pytest.mark.torch  # CI runs these in the nvlink job (docs/ci.md)
torch = pytest.importorskip("torch")


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(D.ENV_FLAG, raising=False)
    assert not D.enabled()
    assert EngineSpec.for_gpu("tempseg", "/m", "L40S").dtbias_fp32 is False
    D.check_activation(False)  # default: nothing to do
    with pytest.raises(D.DtBiasActivationError):
        D.check_activation(True)


def test_register_is_a_noop_without_the_flag(monkeypatch):
    monkeypatch.delenv(D.ENV_FLAG, raising=False)
    monkeypatch.setitem(sys.modules, "vllm", None)  # importing vllm would raise
    D.register()


def test_flag_without_spec_is_refused(monkeypatch):
    monkeypatch.setenv(D.ENV_FLAG, "1")
    with pytest.raises(D.DtBiasActivationError):
        D.check_activation(False)
    monkeypatch.setenv("VLLM_PLUGINS", "something_else")
    with pytest.raises(D.DtBiasActivationError):
        D.check_activation(True)


class FakeGDN(torch.nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.dt_bias = torch.nn.Parameter(torch.ones(heads, dtype=torch.bfloat16),
                                          requires_grad=False)
        self.A_log = torch.nn.Parameter(torch.zeros(heads, dtype=torch.float32))
        self.dt_bias.weight_loader = self.loader
        self.dt_bias.other_attr = "kept"

    @staticmethod
    def loader(param, loaded):
        param.data.copy_(loaded)


def test_patch_makes_dt_bias_fp32_and_keeps_the_loader():
    cls = type("GDN", (FakeGDN,), {})
    assert D.patch_class(cls) is True
    assert D.patch_class(cls) is False  # idempotent
    m = cls()
    assert m.dt_bias.dtype == torch.float32 and m.A_log.dtype == torch.float32
    assert m.dt_bias.weight_loader is FakeGDN.loader and m.dt_bias.other_attr == "kept"
    assert isinstance(m.dt_bias, torch.nn.Parameter)
    value = torch.tensor([0.1, 1e-3, 3.14159, -2.5], dtype=torch.float32)
    m.dt_bias.weight_loader(m.dt_bias, value)
    assert torch.equal(m.dt_bias.data, value)  # exact, no bf16 rounding
    plain = FakeGDN()
    plain.dt_bias.weight_loader(plain.dt_bias, value)
    assert not torch.equal(plain.dt_bias.data.float(), value)  # what unpatched vLLM does
    assert dict(m.named_parameters())["dt_bias"] is m.dt_bias


def test_entry_point_is_discoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    env = D.install_entry_point(tmp_path)
    assert env[D.ENV_FLAG] == "1" and env["VLLM_PLUGINS"] == D.ENTRY_POINT_NAME
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path)
    eps = [ep for dist in importlib.metadata.distributions(path=[str(tmp_path)])
           for ep in dist.entry_points if ep.group == D.ENTRY_POINT_GROUP]
    assert [(e.name, e.value) for e in eps] == [(D.ENTRY_POINT_NAME, D.ENTRY_POINT_VALUE)]
    assert eps[0].load() is D.register
