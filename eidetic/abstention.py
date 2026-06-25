"""Calibrated abstention (Phase 2): a tuned CONTROL gate, not a claim of metacognition.

Confidence blends four signals; two of them (channel agreement, proof completeness) are
STRUCTURAL -- they do not depend on the model's self-report, which is largely a shared
difficulty heuristic. The threshold tau is fit on the dev split (pick_tau), never hardcoded.

All pure + offline-testable; the retriever supplies the signal values.
"""
from __future__ import annotations

from typing import Iterable


def channel_agreement(top) -> float:
    """Fraction of the independent CONTENT channels (dense / bm25 / graph) that surfaced the
    strongest candidate. A structural agreement signal: more channels agreeing -> more trustworthy,
    and it does not rely on the model judging its own confidence."""
    vals = (getattr(top, "dense_score", 0.0), getattr(top, "bm25_score", 0.0),
            getattr(top, "graph_score", 0.0))
    present = sum(1 for v in vals if v and v > 0.0)
    return present / 3.0


def proof_completeness(citations) -> float:
    """Fraction of citations backed by an immutable content hash (a complete provenance chain)."""
    cits = list(citations)
    if not cits:
        return 0.0
    return sum(1 for c in cits if getattr(c, "content_hash", "")) / len(cits)


def combine_confidence(entail: float, coverage: float, agreement: float, proof: float, *,
                       w_entail: float, w_coverage: float, w_agreement: float,
                       w_proof: float) -> float:
    """Weighted confidence in [0, sum(weights)] from the four signals (coverage clamped to [0,1])."""
    cov = min(1.0, max(0.0, coverage))
    return (w_entail * entail + w_coverage * cov
            + w_agreement * agreement + w_proof * proof)


def pick_tau(samples: Iterable[tuple], precision_target: float = 0.95) -> float:
    """Choose the LOWEST tau (max coverage) such that, among answered items (confidence >= tau),
    precision >= precision_target. `samples` is [(confidence, is_correct)] from the DEV split only.
    If no tau reaches the target, return just above the max confidence (abstain everything -- safe).
    """
    pts = [(float(c), bool(ok)) for c, ok in samples]
    if not pts:
        return 0.0
    feasible = []
    for tau in sorted({c for c, _ in pts}):
        answered = [ok for c, ok in pts if c >= tau]
        if answered and (sum(answered) / len(answered)) >= precision_target:
            feasible.append(tau)
    return float(min(feasible)) if feasible else float(max(c for c, _ in pts) + 1e-9)
