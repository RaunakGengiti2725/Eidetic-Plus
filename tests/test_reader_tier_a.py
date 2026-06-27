"""Offline tests for the Tier A reader layer (no model calls).

Covers the deterministic question classifier and the scaffold assembly against the ACTUAL n=40
failing questions, plus the flag-off byte-identical invariant.
"""
from __future__ import annotations

import importlib

from bench.reader import build_reader_prompt, classify_question

# The real failing questions from artifacts/bench_allon_n40 (FAILURES_allon_n40.md).
Q_TEMPORAL = [
    "When did Melanie run a charity race?",                       # q5
    "When did Melanie sign up for a pottery class?",             # q16
    "When did Melanie go camping in June?",                      # q31
    "How long has Caroline had her current group of friends for?",  # q10
]
Q_INFERENCE = [
    "Would Caroline likely have Dr. Seuss books on her bookshelf?",   # q22
    "Would Melanie be considered a member of the LGBTQ community?",   # q30
    "Would Caroline pursue writing as a career option?",             # q27
]
Q_LIST = [
    "What books has Melanie read?",                  # q23
    "What does Melanie do to destress?",            # q24
    "What events has Caroline participated in to help children?",  # q34
]
Q_RECENCY = ["What did Melanie paint recently?"]    # q37


def test_classifier_flags_temporal_questions():
    for q in Q_TEMPORAL:
        assert classify_question(q)["temporal"] is True, q


def test_classifier_flags_inference_questions():
    for q in Q_INFERENCE:
        assert classify_question(q)["inference"] is True, q


def test_classifier_flags_list_questions():
    for q in Q_LIST:
        assert classify_question(q)["list"] is True, q


def test_classifier_flags_recency_question():
    assert classify_question(Q_RECENCY[0])["recency"] is True


def test_classifier_does_not_overfire_inference_on_factual():
    # A plain factual question must NOT be tagged inference (would license hallucination).
    for q in ["What books has Melanie read?", "When did Melanie run a charity race?",
              "What is Caroline's identity?"]:
        assert classify_question(q)["inference"] is False, q


def test_recency_question_is_not_a_list():
    # "What did Melanie paint recently?" -> recency, NOT list (single latest item).
    c = classify_question("What did Melanie paint recently?")
    assert c["recency"] is True and c["list"] is False


# ---- scaffold assembly ------------------------------------------------------------------

BASE = "BASE_PROMPT"


def test_all_flags_off_is_byte_identical():
    for q in Q_TEMPORAL + Q_INFERENCE + Q_LIST + Q_RECENCY:
        out = build_reader_prompt(q, BASE, temporal=False, inference=False, list_=False, recency=False)
        assert out == BASE, q


def test_temporal_scaffold_appended_only_when_flag_and_match():
    out = build_reader_prompt(Q_TEMPORAL[0], BASE, temporal=True, inference=False, list_=False, recency=False)
    assert out.startswith(BASE) and "Temporal question" in out
    # A non-temporal question with the temporal flag on gets no scaffold.
    out2 = build_reader_prompt(Q_LIST[0], BASE, temporal=True, inference=False, list_=False, recency=False)
    assert out2 == BASE


def test_inference_scaffold_appended_last_to_override():
    q = Q_INFERENCE[0]
    out = build_reader_prompt(q, BASE, temporal=True, inference=True, list_=True, recency=True)
    # Inference block must be the final section so its "you MAY infer" overrides earlier rules.
    assert out.rstrip().endswith("absent from memory as certain.")
    assert "Inference question" in out


def test_list_and_recency_scaffolds():
    out_list = build_reader_prompt(Q_LIST[1], BASE, temporal=False, inference=False, list_=True, recency=False)
    assert "List question" in out_list and "Step 1" in out_list
    out_rec = build_reader_prompt(Q_RECENCY[0], BASE, temporal=False, inference=False, list_=False, recency=True)
    assert "Most-recent" in out_rec


def test_reader_tier_a_flags_default_off(monkeypatch):
    for v in ("READER_TIER_A", "READER_TEMPORAL_SCAFFOLD", "READER_GATED_INFERENCE",
              "READER_LIST_TWOPASS", "READER_RECENCY_NUDGE", "READER_JSON_RESILIENT"):
        monkeypatch.delenv(v, raising=False)
    import bench.reader as reader
    importlib.reload(reader)
    assert reader.READER_TIER_A is False
    assert reader.READER_TEMPORAL_SCAFFOLD is False
    assert reader.READER_GATED_INFERENCE is False
    assert reader.READER_LIST_TWOPASS is False
    assert reader.READER_RECENCY_NUDGE is False
    assert reader.READER_JSON_RESILIENT is False
    # Default-flag build returns base unchanged for every failing question.
    for q in Q_TEMPORAL + Q_INFERENCE + Q_LIST + Q_RECENCY:
        assert reader.build_reader_prompt(q, BASE) == BASE


def test_master_switch_enables_all(monkeypatch):
    monkeypatch.setenv("READER_TIER_A", "1")
    import bench.reader as reader
    importlib.reload(reader)
    try:
        assert reader.READER_TEMPORAL_SCAFFOLD and reader.READER_GATED_INFERENCE
        assert reader.READER_LIST_TWOPASS and reader.READER_RECENCY_NUDGE
        assert reader.READER_JSON_RESILIENT
        # Now a temporal question DOES get the scaffold via module-level flags.
        assert "Temporal question" in reader.build_reader_prompt(Q_TEMPORAL[0], BASE)
    finally:
        monkeypatch.delenv("READER_TIER_A", raising=False)
        importlib.reload(reader)
