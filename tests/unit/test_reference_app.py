"""``modal_app/reference_app.py`` wiring that needs no Modal account or GPU."""

import inspect

from flora_deploy.modal_app import reference_app as app_mod
from flora_deploy.modal_app.warm import linked_offline_env


def test_runtime_is_bounded_by_modal_timeouts_not_a_kill_switch():
    src = inspect.getsource(app_mod)
    assert "os._exit" not in src and "_watchdog" not in src  # review2-wsd M3
    # one container's worst case: startup + one call, inside S1's 35-minute bound
    assert app_mod.STARTUP_TIMEOUT_S + app_mod.TIMEOUT_S <= 35 * 60


def test_worker_uses_the_no_copy_layout_env():
    env = linked_offline_env()
    assert app_mod.MODEL_CACHE.as_posix() == env["FLORA_MODEL_CACHE"]
    assert (app_mod.HUB_ROOT / "hf" / "hub").as_posix() == env["HF_HUB_CACHE"]
    assert "ephemeral_disk=" not in inspect.getsource(app_mod)


def test_worker_parameters_and_methods():
    cls = app_mod.ReferenceWorker
    for name in ("annotate", "annotate_batch", "identity_proof", "report", "run_shard"):
        assert hasattr(cls, name), name
