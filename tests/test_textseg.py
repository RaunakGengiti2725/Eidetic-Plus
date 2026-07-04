"""Abbreviation-aware sentence segmentation + value capture (live UX2 catch)."""
from __future__ import annotations

from eidetic.smqe.record_ops import _answer_value
from eidetic.textseg import split_sentences


def test_split_survives_honorifics_and_initials():
    """The naive split recalled the user's dentist as literally 'Dr' -- honorifics and
    single-letter initials must never end a sentence."""
    assert split_sentences(
        "I switched dentists: now I see Dr. Okafor at River Smiles. She is great.") == [
        "I switched dentists: now I see Dr. Okafor at River Smiles.", "She is great."]
    assert split_sentences("We met Prof. Chen at 3 pm. Then we left.") == [
        "We met Prof. Chen at 3 pm.", "Then we left."]
    assert split_sentences("J. R. R. Tolkien wrote it. I loved it!") == [
        "J. R. R. Tolkien wrote it.", "I loved it!"]
    assert split_sentences("It cost $5. Then we ate.") == ["It cost $5.", "Then we ate."]


def test_value_capture_carries_through_honorifics():
    """Second blindness, same shape: the copular value capture stopped at the period in
    'Dr.' and shipped 'Dr' as a verified answer on the live as_of path."""
    assert _answer_value("Who is my dentist?",
                         "My dentist is Dr. Alvarez at Maple Dental.") == \
        "Dr. Alvarez at Maple Dental"
    assert _answer_value("What is the plan?", "The plan is simple. We leave at dawn.") == \
        "simple"
