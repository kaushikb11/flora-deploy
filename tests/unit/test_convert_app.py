"""``flora-convert`` imports lazily and stays inside its spend and write limits."""

from flora_deploy.config.pins import PINS_BY_KEY


def test_convert_app_imports_and_targets_derived_dir():
    from flora_deploy.modal_app import convert_app

    assert convert_app.app.name == "flora-convert"
    assert convert_app.IDENTITY_TIMEOUT_S <= 30 * 60
    rev = PINS_BY_KEY["captioner"].revision
    assert str(convert_app._derived()) == f"/weights/derived/captioner-bf16@{rev}"
    assert str(convert_app._source()).startswith("/weights/snapshots/")
