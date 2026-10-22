"""The cold-witness count-flip cases, shared by the generator's tests
(``tests/unit/test_certify_witness_union.py``) and the independent checker's
(``tests/unit/test_cert_check.py``), so both implementations are held to the same table.

The rule (``docs/certification.md``, "Pass rule"): the count-flip check is over the union of the
candidate's and the witness's episodes, ``flips <= 0.10 x n`` rounded half up (2 of 18 pass,
2 of 14 fail); the witness's other means keep their tight floors on their own.

Each case: ``main`` candidate episodes (``main_flips`` of them, outside the witness subset,
split one subtask in two), a 6-episode witness (``witness_flips`` of them split), and an
optional witness ``drift``: ``"boundary"`` (every witness boundary 0.2 s late: segment fails),
``"caption"`` (every witness caption differs: caption fails), ``"absent"`` (no witness records)
or ``"warm"`` (witness records, none forced cold). The last two are ``provisional``
(``no_witness``): the evidence covers no forced-cold container.
"""

from __future__ import annotations

from dataclasses import dataclass

WITNESS_N = 6


@dataclass(frozen=True)
class Case:
    id: str
    main: int
    main_flips: int
    witness_flips: int
    drift: str | None
    verdict: str  # of the stage the case is about
    stage: str = "segment"
    failing: tuple[tuple[str, str], ...] = ()  # (class, metric) checks that fail
    reasons: tuple[str, ...] = ()  # the provisional reasons, when the verdict is provisional


CASES = (
    Case("one_witness_flip_none_in_main", 12, 0, 1, None, "certified"),
    Case("two_flips_in_eighteen", 12, 1, 1, None, "certified"),
    Case("three_flips_in_eighteen", 12, 2, 1, None, "fail",
         failing=(("union", "count_mismatch"),)),
    Case("two_flips_in_fourteen", 8, 1, 1, None, "fail",
         failing=(("union", "count_mismatch"),)),
    Case("witness_boundaries_drift", 12, 0, 0, "boundary", "fail",
         failing=(("cold-witness", "iou_divergence"), ("cold-witness", "boundary_error_s"))),
    Case("witness_captions_drift", 12, 0, 0, "caption", "fail", stage="caption",
         failing=(("cold-witness", "caption_mismatch"),
                  ("cold-witness", "caption_edit_distance"))),
    Case("no_witness", 12, 1, 0, "absent", "provisional", reasons=("no_witness",)),
    Case("witness_never_forced_cold", 12, 0, 0, "warm", "provisional",
         reasons=("no_witness",)),
)
CASE_IDS = [c.id for c in CASES]
UNION = {"one_witness_flip_none_in_main": (1, 18, 2), "two_flips_in_eighteen": (2, 18, 2),
         "three_flips_in_eighteen": (3, 18, 2), "two_flips_in_fourteen": (2, 14, 1),
         "witness_boundaries_drift": (0, 18, 2), "witness_captions_drift": (0, 18, 2),
         "no_witness": (1, 12, 1), "witness_never_forced_cold": (0, 18, 2)}
"""case id -> (union flips, union n, flips allowed)"""
