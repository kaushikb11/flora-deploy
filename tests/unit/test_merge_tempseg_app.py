"""``flora-convert`` TempSeg merge functions stay inside their write and spend limits."""

from flora_deploy.config.pins import PINS_BY_KEY


def test_merge_targets_derived_dir_and_caps():
    from flora_deploy.modal_app import convert_app as c

    rev = PINS_BY_KEY["tempseg"].revision
    assert str(c._tempseg_merged()) == f"/weights/derived/tempseg-merged@{rev}"
    assert str(c._tempseg_base()).endswith(PINS_BY_KEY["tempseg_base"].revision)
    assert str(c._tempseg_flora()).endswith(rev)
    assert c.EQUIV_TIMEOUT_S <= 25 * 60 and c.EQUIV_GPU == "L40S"
    assert c.MERGE_TIMEOUT_S <= 30 * 60
    assert len(c.EQUIV_EPISODES) == 2


def test_one_model_session():
    from flora_deploy.modal_app.convert_app import _Enough, _OneModel

    s = _OneModel("m", "p")
    assert s.load("tempseg") == ("m", "p")
    s.close()
    assert not issubclass(_Enough, Exception)
