"""Unified dataset schema shared by the LongMemEval and LoCoMo loaders.

A Sample is one question plus the conversation history (sessions of turns) it must be
answered from. The harness ingests the sessions into a system, then asks the question and
grades against `gold` with the fixed judge.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional


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


# ---- The integrity wall primitive ------------------------------------------
#
# Every continuous optimizer (the offline sweep, the abstention calibrator, the
# online fusion-weight / bandit learners) is allowed to read EXACTLY ONE thing:
# the private "dev" split. Reported/official benchmark numbers come from the
# disjoint "test" split. This is the non-negotiable wall: no optimizer may read,
# fit to, or cache a benchmark test item. The partition is a stable hash of the
# sample_id, so it is deterministic, dataset-agnostic, and requires no extra file
# -- every dataset is split the same way on every machine.
DEV_SPLIT_PCT = 20  # ~20% of every dataset reserved as the private optimization split.
_VALID_SPLITS = ("dev", "test", "all")


def split_of(sample_id: str, dev_pct: int = DEV_SPLIT_PCT) -> str:
    """Deterministically assign a sample to the 'dev' (optimization) or 'test'
    (reported) split. Stable across machines and runs; the same sample_id always
    lands on the same side. dev_pct controls the dev fraction."""
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % 100
    return "dev" if bucket < dev_pct else "test"


def filter_split(samples: Iterable[Sample], split: Optional[str],
                 dev_pct: int = DEV_SPLIT_PCT) -> list[Sample]:
    """Keep only the samples on the requested split. split in {None, 'all'} is a
    no-op; 'dev'/'test' apply the wall partition. Raises on an unknown split so a
    typo can never silently leak the test set into an optimizer."""
    if split is None or split == "all":
        return list(samples)
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    return [s for s in samples if split_of(s.sample_id, dev_pct) == split]
