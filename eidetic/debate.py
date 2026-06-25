"""Bounded cross-check debate for conflict resolution (PDF Theme 2a/2b).

Multi-agent debate improves factuality, but has a documented failure mode: "communication
hallucination" / misleading consensus, where debate amplifies a confident error. So Eidetic-Plus
uses debate ONLY on detected conflicts (not every query) and aggregates with a guard:

  * A claim wins only with >= min_agreement INDEPENDENT votes (a lone, even highly-confident,
    voter can never override a genuine majority).
  * No majority -> ABSTAIN rather than emit a confident-but-unsupported answer.

`aggregate_verdicts` is the pure, deterministic, offline-tested guard. The debate ROUNDS
themselves (asking 2-3 role-specialized retrievers) are LLM-gated and run only on conflicts.
"""
from __future__ import annotations

from collections import defaultdict


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def aggregate_verdicts(verdicts: list[tuple[str, float]], min_agreement: int = 2) -> dict:
    """Aggregate (answer, confidence) verdicts with the communication-hallucination guard.

    Returns {answer, consensus, votes, total, support}. consensus is True only when the winning
    answer has >= min_agreement votes; otherwise answer is None and consensus is False (abstain).
    The winner is by VOTE COUNT first (confidence only breaks ties), so one overconfident voter
    cannot overrule a majority."""
    if not verdicts:
        return {"answer": None, "consensus": False, "votes": 0, "total": 0, "support": 0.0}
    groups: dict[str, dict] = defaultdict(lambda: {"votes": 0, "conf": 0.0, "rep": ""})
    for ans, conf in verdicts:
        key = _norm(ans)
        g = groups[key]
        g["votes"] += 1
        g["conf"] += float(conf)
        if not g["rep"]:
            g["rep"] = ans
    # winner: most votes, then highest summed confidence.
    key = max(groups, key=lambda k: (groups[k]["votes"], groups[k]["conf"]))
    win = groups[key]
    total = len(verdicts)
    consensus = win["votes"] >= min_agreement
    return {
        "answer": win["rep"] if consensus else None,
        "consensus": consensus,
        "votes": win["votes"],
        "total": total,
        "support": win["votes"] / total if total else 0.0,
    }


def run_conflict_debate(engine, query: str, role_retrievers=None) -> dict:
    """Gated bounded debate over a DETECTED conflict. Off -> no-op. Enabled -> ask 2-3
    role-specialized retrievers and aggregate with the guard (real LLM calls; fail-loud; not run
    under the current quota block). Advisory only -- it never overrides a verified answer."""
    if not getattr(engine.settings, "debate_enabled", False):
        return {"skipped": "disabled"}
    from .errors import FeatureNotImplementedError
    raise FeatureNotImplementedError(
        "DEBATE is experimental and not implemented yet; default off. aggregate_verdicts IS the "
        "offline-validated consensus guard; the enabled debate rounds (2-3 retriever calls, real) "
        "are a documented integration point to build + measure on the dev split (only on detected "
        "conflicts) when quota is restored."
    )
