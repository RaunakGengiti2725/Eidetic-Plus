"""Shared sentence segmentation: one split pattern, abbreviation-aware.

The naive (?<=[.!?])\\s+ split cut answers at honorifics -- a live MCP exercise recalled
the user's dentist as literally 'Dr' because 'Dr. Okafor' split mid-name. Fixed-width
lookbehinds veto the split after common abbreviations and single-letter initials.
"""
from __future__ import annotations

import re

_GUARDS = "".join((
    r"(?<!\bDr\.)", r"(?<!\bMr\.)", r"(?<!\bMs\.)", r"(?<!\bSt\.)", r"(?<!\bJr\.)",
    r"(?<!\bSr\.)", r"(?<!\bMrs\.)", r"(?<!\bProf\.)", r"(?<!\bMt\.)", r"(?<!\bvs\.)",
    r"(?<!\bno\.)", r"(?<!\bNo\.)", r"(?<!\b[A-Z]\.)",
))
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])" + _GUARDS + r"\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split((text or "").strip()) if s.strip()]
