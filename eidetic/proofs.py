"""Proof traces: make provenance a first-class, machine-readable output rather than metadata.

`prove_answer` turns an Answer into a proof tree (claim, the immutable source span/hash/timestamp
behind each cited memory, its NLI grounding label, and any contradictions). This is the
"citable photographic recall" differentiator: every answer can show its work against immutable
evidence. Pure formatting over data the engine already produced; no model call, no fabrication.
"""
from __future__ import annotations

from typing import Optional

from .models import Answer, NLILabel, RecallTrace


def prove_answer(answer: Answer, trace: Optional[RecallTrace] = None) -> dict:
    """A machine-readable proof tree for an answer. Read-only, deterministic.

    When a matching `trace` is supplied (Connected Brain Loop), each evidence item gains
    ADDITIVE recall-path metadata -- which channels surfaced the memory and, if it came in
    through a dream gist, the gist's cid -- so the proof can answer 'why this memory?'. With
    `trace=None` the output is byte-identical to the legacy proof tree (no new keys)."""
    cites = answer.citations or []
    # Paths are only trustworthy when the trace belongs to THIS answer (guards cache hits /
    # a stale last_trace from a later retrieve).
    aligned = trace is not None and trace.query == answer.question
    evidence = []
    for c in cites:
        item = {
            "memory_id": c.memory_id,
            "content_hash": c.content_hash,        # the immutable raw record this rests on
            "raw_uri": c.raw_uri,
            "source": c.source,
            "valid_at": c.valid_at,
            "snippet": c.snippet,
            "nli_label": c.nli_label.value if isinstance(c.nli_label, NLILabel) else str(c.nli_label),
            "nli_score": c.nli_score,
            "grounded": c.nli_label == NLILabel.ENTAILMENT,
        }
        if aligned:
            item["recall_paths"] = trace.paths_for(c.memory_id)
            via_gist = trace.gist_provenance.get(c.memory_id)
            if via_gist:
                item["via_gist"] = via_gist
        evidence.append(item)
    grounded = [e for e in evidence if e["grounded"]]
    contradictions = [e for e in evidence if e["nli_label"] == "contradiction"]
    out = {
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
    if aligned:
        out["recall_trace"] = {
            "enabled_channels": list(trace.enabled_channels),
            "selected": list(trace.selected_candidates),
            "region_hints": list(trace.region_hints),
            "latency_by_stage": dict(trace.latency_by_stage),
        }
    return out
