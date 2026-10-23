"""``certified`` on balanced records (``records.certification``, ``CertStamp.certify``): a
non-reference record is ``certified: true`` only when a committed certificate matching its
stack's key (preset, kind, pool GPU, embodiment, engine spec, derived content and logic digest:
the whole ``cert_key``) decides ``certified`` **and** the episode is in-envelope."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from flora_deploy.records import Provenance, StageRecord, certification
from flora_deploy.workers_impl.balanced import StampedJob
from flora_deploy.workers_impl.certstamp import CertStamp
from flora_deploy.workers_impl.segment import EpisodeOutcome
from tests.unit.test_certificate_drift import LATER, commit, current_doc, make_cert


@pytest.mark.parametrize("envelope_ok, stack, want", [
    (True, True, (True, None)),
    (True, False, (False, "uncertified_stack")),
    (False, True, (False, "outside_reference_envelope")),
    (False, False, (False, "outside_reference_envelope")),  # the envelope wins
    (None, True, (False, "oracle_ineligible_source")),
])
def test_one_rule_for_every_kind(envelope_ok, stack, want):
    assert certification(envelope_ok, stack_certified=stack) == want


def stamp(doc, tmp_path, *, embodiment="human", gpu="L40S"):
    return CertStamp(embodiment, doc, "NVIDIA L40S", preset="balanced", kind="segment",
                     pool_gpu=gpu, cert_dir=tmp_path)


def record(reason="uncertified_stack"):
    p = Provenance(models={}, engine="vllm", preset="balanced", config_hash="b" * 64,
                   logic_version="l", framework="f")
    return StageRecord(uid="u", dataset="d", episode_id="u", stage="segment", provenance=p,
                       certified=False, uncertified_reason=reason)


def test_matching_certificate_certifies_in_envelope_records(tmp_path):
    doc = current_doc("segment", "L40S")
    cert = commit(tmp_path, make_cert(doc, pools=("human",)))
    out = stamp(doc, tmp_path).certify(record(), True)
    assert out.certified and out.uncertified_reason is None
    assert out.provenance.certificate == cert["id"]
    off = stamp(doc, tmp_path).certify(record(), False)
    assert not off.certified and off.uncertified_reason == "outside_reference_envelope"


def test_key_mismatch_wrong_embodiment_or_gpu_is_not_certified(tmp_path):
    doc = current_doc("segment", "L40S")
    commit(tmp_path, make_cert(doc, pools=("human",)))
    other = {**doc, "logic_digest": "0" * 64}  # another build
    for s in (stamp(other, tmp_path), stamp(doc, tmp_path, embodiment="yam"),
              stamp(doc, tmp_path, gpu="H100")):
        out = s.certify(record(), True)
        assert not out.certified and out.uncertified_reason == "uncertified_stack"
        assert out.provenance.certificate is None


def test_newer_fail_revokes_and_no_registry_certifies_nothing(tmp_path):
    doc = current_doc("segment", "L40S")
    commit(tmp_path, make_cert(doc, pools=("human",)))
    commit(tmp_path, make_cert(doc, pools=("human",), verdict="fail", stamp=LATER))
    assert not stamp(doc, tmp_path).certify(record(), True).certified
    assert not stamp(doc, tmp_path / "empty").certify(record(), True).certified


def test_fake_records_are_never_certified(tmp_path):
    doc = current_doc("segment", "L40S")
    commit(tmp_path, make_cert(doc, pools=("human",)))
    rec = record("fake_models")
    assert stamp(doc, tmp_path).certify(rec, True) == rec


def test_stamped_job_applies_the_rule_per_episode(tmp_path):
    doc = current_doc("segment", "L40S")
    commit(tmp_path, make_cert(doc, pools=("human",)))

    class Job:
        stage = "segment"
        nv = None

        async def run(self, ref, batcher, ctx):
            return EpisodeOutcome([record()], [])

    job = StampedJob(Job(), stamp(doc, tmp_path))
    inside = asyncio.run(job.run(SimpleNamespace(envelope_ok=True), None, None))
    outside = asyncio.run(job.run(SimpleNamespace(envelope_ok=False), None, None))
    assert inside.records[0].certified
    assert outside.records[0].uncertified_reason == "outside_reference_envelope"
