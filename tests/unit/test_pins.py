import re

from flora_deploy.config.pins import PINS, PINS_BY_KEY

SHA = re.compile(r"^[0-9a-f]{40}$")


def test_every_pin_is_a_full_commit_sha():
    for pin in PINS:
        assert SHA.match(pin.revision), pin.key


def test_keys_are_unique():
    assert len(PINS_BY_KEY) == len(PINS)


def test_each_nvidia_family_has_weights_and_code():
    roles = {p.key: p.role for p in PINS}
    assert roles["tempseg"] == "weights" and roles["tempseg_code"] == "code"
    assert roles["captioner"] == "weights" and roles["captioner_code"] == "code"
    assert roles["rewarder"] == "weights+code"
