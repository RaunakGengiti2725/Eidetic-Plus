"""Unified dataset schema shared by the LongMemEval and LoCoMo loaders.

A Sample is one question plus the conversation history (sessions of turns) it must be
answered from. The harness ingests the sessions into a system, then asks the question and
grades against `gold` with the fixed judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    role: str
    content: str
    timestamp: Optional[str] = None


@dataclass
class Session:
    session_id: str
    turns: list[Turn]
    session_time: Optional[float] = None   # epoch seconds, if the dataset provides a date


@dataclass
class Sample:
    sample_id: str
    sessions: list[Session]
    question: str
    gold: str
    category: str
    dataset: str
    question_time: Optional[float] = None
    meta: dict = field(default_factory=dict)


def category_counts(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        out[s.category] = out.get(s.category, 0) + 1
    return dict(sorted(out.items()))
