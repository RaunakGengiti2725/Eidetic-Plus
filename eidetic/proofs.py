"""Proof traces: make provenance a first-class, machine-readable output rather than metadata.

`prove_answer` turns an Answer into a proof tree (claim, the immutable source span/hash/timestamp
behind each cited memory, its NLI grounding label, and any contradictions). This is the
"citable photographic recall" differentiator: every answer can show its work against immutable
evidence. Pure formatting over data the engine already produced; no model call, no fabrication.
"""
from __future__ import annotations

from .models import Answer, NLILabel


def prove_answer(answer: Answer) -> dict:
    """A machine-readable proof tree for an answer. Read-only, deterministic."""
    cites = answer.citations or []
    evidence = []
    for c in cites:
        evidence.append({
            "memory_id": c.memory_id,
            "content_hash": c.content_hash,        # the immutable raw record this rests on
            "raw_uri": c.raw_uri,
            "source": c.source,
            "valid_at": c.valid_at,
            "snippet": c.snippet,
            "nli_label": c.nli_label.value if isinstance(c.nli_label, NLILabel) else str(c.nli_label),
            "nli_score": c.nli_score,
            "grounded": c.nli_label == NLILabel.ENTAILMENT,
        })
    grounded = [e for e in evidence if e["grounded"]]
    contradictions = [e for e in evidence if e["nli_label"] == "contradiction"]
    return {
        "claim": answer.answer,
        "verified": answer.verified,
        "confidence": answer.confidence,
        "grounded_count": len(grounded),
        "evidence_count": len(evidence),
        "evidence": evidence,
        "contradictions": contradictions,
        "unverified_claims": list(answer.unverified_claims or []),
        # every cited memory carries a content hash, so the claim is fully provenance-backed
        "provenance_complete": bool(evidence) and all(e["content_hash"] for e in evidence),
        "note": answer.note,
    }
