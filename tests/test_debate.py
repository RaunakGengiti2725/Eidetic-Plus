"""Offline tests for the bounded-debate aggregation guard (no key)."""
from __future__ import annotations

import types

from eidetic.debate import aggregate_verdicts, run_conflict_debate


def test_majority_consensus():
    out = aggregate_verdicts([("Paris", 0.6), ("Paris", 0.7), ("Lyon", 0.9)], min_agreement=2)
    assert out["consensus"] is True and out["answer"] == "Paris" and out["votes"] == 2


def test_no_majority_abstains():
    out = aggregate_verdicts([("A", 0.9), ("B", 0.8), ("C", 0.7)], min_agreement=2)
    assert out["consensus"] is False and out["answer"] is None


def test_lone_confident_voter_cannot_override_majority():
    # the communication-hallucination guard: 1 very-confident "A" loses to 2 "B" votes.
    out = aggregate_verdicts([("A", 0.99), ("B", 0.5), ("B", 0.51)], min_agreement=2)
    assert out["answer"] == "B" and out["votes"] == 2


def test_normalizes_answer_text():
    out = aggregate_verdicts([("the Eiffel Tower", 0.5), ("The  Eiffel   Tower", 0.5)])
    assert out["consensus"] is True and out["votes"] == 2


def test_empty_is_abstain():
    out = aggregate_verdicts([])
    assert out["consensus"] is False and out["answer"] is None


def test_debate_disabled_noop():
    fake = types.SimpleNamespace(settings=types.SimpleNamespace(debate_enabled=False))
    assert run_conflict_debate(fake, "q") == {"skipped": "disabled"}
