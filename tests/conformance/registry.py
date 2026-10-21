"""Workers and stores that must pass the conformance suites.

A new ``ShardWorker`` (RewardWorker, SegcapWorker, ...) adds a ``WorkerSpec`` here whose factory
builds it over the suite's store with its engine faked to follow ``WorkerEnv.script``; see
``worker_suite.py``. A new ``Store`` adds a factory to ``STORES`` in ``store_suite.py``.
"""

from __future__ import annotations

from tests.conformance.reference_worker import fake_spec
from tests.conformance.worker_suite import WorkerSpec


def _canary_spec() -> WorkerSpec:
    from flora_deploy.modal_app.canary_app import conformance_spec

    return conformance_spec()


WORKERS: list[WorkerSpec] = [
    fake_spec("segcap"),
    fake_spec("reward.terminal"),
    _canary_spec(),
]
