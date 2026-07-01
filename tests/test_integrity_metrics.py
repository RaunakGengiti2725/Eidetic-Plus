"""Offline test for the scoreboard integrity rollup (verified recall vs fabrication).

Feeds synthetic log rows to aggregate() and asserts the per-system integrity counters, which are
read straight off the logged correct/abstained/extra.verified flags (no model calls).
"""
from __future__ import annotations

from bench.scoreboard import aggregate


def _row(system, correct, verified, abstained, error=False, sample="toy0_q0", has_verify_key=True):
    return {
        "system": system, "dataset": "locomo", "category": "multi-hop",
        "sample_id": sample, "run_idx": 0, "correct": correct, "abstained": abstained,
        "error": error, "query_tokens": 100, "search_ms": 1.0, "e2e_ms": 2.0,
        "write_tokens": 0, "extra": ({"verified": verified} if has_verify_key else {}),
    }


def test_integrity_counts_verified_unverified_abstained():
    rows = [
        _row("eidetic-plus-full", correct=True, verified=True, abstained=False, sample="toy0_q0"),
        _row("eidetic-plus-full", correct=True, verified=False, abstained=False, sample="toy0_q1"),
        _row("eidetic-plus-full", correct=False, verified=False, abstained=True, sample="toy0_q2"),
        _row("eidetic-plus-full", correct=False, verified=False, abstained=False, sample="toy0_q3"),
    ]
    ig = aggregate(rows)["integrity"]["eidetic-plus-full"]
    assert ig["n"] == 4
    assert ig["verified_correct"] == 1          # only q0 is correct AND proven
    assert ig["abstained"] == 1                 # q2
    assert ig["answered"] == 3                  # q0,q1,q3
    assert ig["unverified_answered"] == 2       # q1,q3 emitted without a proof
    assert ig["has_verify"] is True             # q0 verified + q2 abstained -> has a verify step


def test_integrity_excludes_error_rows():
    rows = [
        _row("rag-full", correct=True, verified=False, abstained=False,
             sample="toy0_q0", has_verify_key=False),
        _row("rag-full", correct=True, verified=False, abstained=False,
             error=True, sample="toy0_q1", has_verify_key=False),
    ]
    ig = aggregate(rows)["integrity"]["rag-full"]
    assert ig["n"] == 1                          # the error row is excluded
    assert ig["unverified_answered"] == 1        # baseline has no verify step -> all unproven


def test_integrity_verified_false_key_means_verify_step_present():
    rows = [_row("eidetic-plus-full", correct=True, verified=False, abstained=False)]
    ig = aggregate(rows)["integrity"]["eidetic-plus-full"]
    assert ig["has_verify"] is True
    assert ig["unverified_answered"] == 1


def test_integrity_baseline_zero_verified():
    rows = [_row("rag-vector", correct=True, verified=False, abstained=False,
                 sample=f"c0_q{i}", has_verify_key=False)
            for i in range(5)]
    ig = aggregate(rows)["integrity"]["rag-vector"]
    assert ig["verified_correct"] == 0
    assert ig["unverified_answered"] == 5        # all unproven -> shows N/A, never "fabrication"
    assert ig["has_verify"] is False             # no verify step -> N/A in the rendered table
