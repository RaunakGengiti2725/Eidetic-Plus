"""AUTORESEARCH: the scientific-method organ of the epistemic organism.

The mutable surface is the MIND (retrieval/read knobs, operator pipelines, laws);
the immutable physics is PROOF (NLI floors, abstention thresholds, the witness) plus
the DEV-split guard. One hypothesis per trial; a trial is a paired champion-vs-
challenger evaluation over identical dev rows; only a McNemar-significant win
promotes. Every trial lands in an append-only ledger (trials.jsonl) a judge can cat.

The ratchet is subordinate to the epistemic map: the agenda drains the FRONTIER
(contested first, then unknown by information gain), with live ask-failures, MemMA
repair proposals, surprise ingest, and knob imbalance behind it.
"""
from .agenda import ResearchAgenda
from .registry import ChampionRegistry
from .types import (FailureClass, ResearchHypothesis, ResearchTask, ResearchTrial,
                    classify_failure)

__all__ = [
    "FailureClass", "ResearchAgenda", "ResearchHypothesis", "ResearchTask",
    "ResearchTrial", "ChampionRegistry", "classify_failure",
]
