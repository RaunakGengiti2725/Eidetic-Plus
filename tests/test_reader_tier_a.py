"""Offline tests for the Tier A reader layer (no model calls).

Covers the deterministic question classifier and the scaffold assembly against the ACTUAL n=40
failing questions, plus the flag-off byte-identical invariant.
"""
from __future__ import annotations

import importlib

from bench.reader import build_reader_prompt, classify_question, normalize_date_weekdays

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
    "What pets wouldn't cause any discomfort to Joanna?",
    "Would Tim enjoy reading books by C. S. Lewis or John Greene?",
    "Do you think Caroline should pursue writing as a career?",
    "Is Melanie a good fit for the pottery workshop?",
]
Q_LIST = [
    "What books has Melanie read?",                  # q23
    "What does Melanie do to destress?",            # q24
    "What events has Caroline participated in to help children?",  # q34
    "What is an indoor activity that Andrew would enjoy doing while making his dog happy?",
]
Q_RECENCY = ["What did Melanie paint recently?"]    # q37
Q_PREFERENCE = [
    "What does the user prefer for long flights?",
    "What is the user's favorite type of music?",
    "What food is the user allergic to?",
    "What does the user dislike?",
]
Q_AGGREGATION = [
    "How many plants did I acquire in the last month?",
    "How much total money have I spent on bike-related expenses since the start of the year?",
    "How many items of clothing do I need to pick up or return from a store?",
]


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


def test_classifier_flags_preference_questions():
    for q in Q_PREFERENCE:
        assert classify_question(q)["preference"] is True, q


def test_classifier_does_not_overfire_inference_on_factual():
    # A plain factual question must NOT be tagged inference (would license hallucination).
    for q in ["What books has Melanie read?", "When did Melanie run a charity race?",
              "What is Caroline's identity?", "Can you tell me when Melanie ran the race?"]:
        assert classify_question(q)["inference"] is False, q


def test_recency_question_is_not_a_list():
    # "What did Melanie paint recently?" -> recency, NOT list (single latest item).
    c = classify_question("What did Melanie paint recently?")
    assert c["recency"] is True and c["list"] is False


def test_aggregation_questions_are_not_single_latest_recency():
    for q in Q_AGGREGATION:
        c = classify_question(q)
        assert c["aggregation"] is True, q
        assert c["recency"] is False, q


# ---- scaffold assembly ------------------------------------------------------------------

BASE = "BASE_PROMPT"


def test_all_flags_off_is_byte_identical():
    for q in Q_TEMPORAL + Q_INFERENCE + Q_LIST + Q_RECENCY + Q_PREFERENCE:
        out = build_reader_prompt(
            q, BASE, temporal=False, inference=False, list_=False, recency=False, preference=False
        )
        assert out == BASE, q


def test_temporal_scaffold_appended_only_when_flag_and_match():
    out = build_reader_prompt(Q_TEMPORAL[0], BASE, temporal=True, inference=False, list_=False, recency=False)
    assert out.startswith(BASE) and "Temporal question" in out
    assert "preserve that relative wording" in out
    assert "absolute date in parentheses" in out
    # A non-temporal question with the temporal flag on gets no scaffold.
    out2 = build_reader_prompt(Q_LIST[0], BASE, temporal=True, inference=False, list_=False, recency=False)
    assert out2 == BASE


def test_inference_scaffold_appended_last_to_override():
    q = Q_INFERENCE[0]
    out = build_reader_prompt(q, BASE, temporal=True, inference=True, list_=True, recency=True)
    # Inference block must be the final section so its "you MAY infer" overrides earlier rules.
    assert out.rstrip().endswith("absent from memory as certain.")
    assert "Inference question" in out


def test_inference_scaffold_handles_category_and_absence_inference():
    category = build_reader_prompt(
        "Would Caroline likely have Dr. Seuss books on her bookshelf?",
        BASE,
        temporal=False,
        inference=True,
        list_=False,
        recency=False,
    )
    assert "classic children's books" in category
    assert "ordinary world knowledge" in category
    assert 'Begin your answer with "Likely yes" or "Likely no"' in category

    membership = build_reader_prompt(
        "Would Melanie be considered a member of the LGBTQ community?",
        BASE,
        temporal=False,
        inference=True,
        list_=False,
        recency=False,
    )
    assert "absence of self-identification/membership evidence" in membership
    assert "support, allyship, or interest" in membership


def test_list_and_recency_scaffolds():
    out_list = build_reader_prompt(Q_LIST[1], BASE, temporal=False, inference=False, list_=True, recency=False)
    assert "List question" in out_list and "Step 1" in out_list
    assert "generic hobby/event is not enough" in out_list
    assert "destress/relax/unwind questions" in out_list
    out_rec = build_reader_prompt(Q_RECENCY[0], BASE, temporal=False, inference=False, list_=False, recency=True)
    assert "Most-recent" in out_rec
    children = build_reader_prompt(Q_LIST[2], BASE, temporal=False, inference=False, list_=True, recency=False)
    assert "children/kids/students" in children
    assert "exclude unrelated events" in children
    out_count = build_reader_prompt(Q_AGGREGATION[0], BASE, temporal=False, inference=False,
                                    list_=True, recency=True)
    assert "Count / total question" in out_count
    assert "Most-recent" not in out_count


def test_preference_scaffold_preserves_polarity():
    out = build_reader_prompt(
        Q_PREFERENCE[0],
        BASE,
        temporal=False,
        inference=False,
        list_=False,
        recency=False,
        preference=True,
    )
    assert "Preference question" in out
    assert "first-person statements" in out
    assert "Preserve polarity" in out
    assert "allergic to" in out


def test_normalize_date_weekdays_corrects_iso_pairs():
    assert normalize_date_weekdays("2023-05-07 (Monday) [S20]") == "2023-05-07 (Sunday) [S20]"
    assert normalize_date_weekdays("The date was 2023-08-11, Tuesday.") == (
        "The date was 2023-08-11, Friday."
    )


def test_reader_tier_a_flags_default_off(monkeypatch):
    for v in ("READER_TIER_A", "READER_TEMPORAL_SCAFFOLD", "READER_GATED_INFERENCE",
              "READER_LIST_TWOPASS", "READER_RECENCY_NUDGE", "READER_PREFERENCE_RUBRIC",
              "READER_JSON_RESILIENT"):
        monkeypatch.delenv(v, raising=False)
    import bench.reader as reader
    importlib.reload(reader)
    assert reader.READER_TIER_A is False
    assert reader.READER_TEMPORAL_SCAFFOLD is False
    assert reader.READER_GATED_INFERENCE is False
    assert reader.READER_LIST_TWOPASS is False
    assert reader.READER_RECENCY_NUDGE is False
    assert reader.READER_PREFERENCE_RUBRIC is False
    assert reader.READER_JSON_RESILIENT is False
    # Default-flag build returns base unchanged for every failing question.
    for q in Q_TEMPORAL + Q_INFERENCE + Q_LIST + Q_RECENCY + Q_PREFERENCE:
        assert reader.build_reader_prompt(q, BASE) == BASE


def test_master_switch_enables_all(monkeypatch):
    monkeypatch.setenv("READER_TIER_A", "1")
    import bench.reader as reader
    importlib.reload(reader)
    try:
        assert reader.READER_TEMPORAL_SCAFFOLD and reader.READER_GATED_INFERENCE
        assert reader.READER_LIST_TWOPASS and reader.READER_RECENCY_NUDGE
        assert reader.READER_PREFERENCE_RUBRIC
        assert reader.READER_JSON_RESILIENT
        # Now a temporal question DOES get the scaffold via module-level flags.
        assert "Temporal question" in reader.build_reader_prompt(Q_TEMPORAL[0], BASE)
    finally:
        monkeypatch.delenv("READER_TIER_A", raising=False)
        importlib.reload(reader)
