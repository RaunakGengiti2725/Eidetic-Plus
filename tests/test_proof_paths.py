"""Offline tests for prove_answer recall-path metadata (Connected Brain Loop, Phase 6).

The contract: with trace=None the proof tree is BYTE-IDENTICAL to the legacy output (no new
keys); with a matching trace it gains additive recall_paths / via_gist / recall_trace keys.
"""
from __future__ import annotations

from eidetic.models import Answer, Citation, NLILabel, RecallTrace
from eidetic.proofs import prove_answer


def _answer():
    return Answer(
        question="who is the cto?", answer="Dana",
        verified=True, confidence=0.8,
        citations=[Citation(memory_id="m4", content_hash="h4", raw_uri="uri4", source="user",
                            valid_at=1.0, snippet="Dana is the CTO",
                            nli_label=NLILabel.ENTAILMENT, nli_score=0.95)],
        generated_by="x", retrieved_count=1, note="",
    )


def test_no_trace_proof_is_byte_identical_to_legacy():
    out = prove_answer(_answer())
    # exact legacy shape -- no spine keys leak in when no trace is passed.
    assert out == {
        "claim": "Dana",
        "verified": True,
        "confidence": 0.8,
        "grounded_count": 1,
        "evidence_count": 1,
        "evidence": [{
            "memory_id": "m4", "content_hash": "h4", "raw_uri": "uri4", "source": "user",
            "valid_at": 1.0, "snippet": "Dana is the CTO", "nli_label": "entailment",
            "nli_score": 0.95, "grounded": True,
        }],
        "contradictions": [],
        "unverified_claims": [],
        "provenance_complete": True,
        "note": "",
    }
    assert "recall_trace" not in out
    assert all("recall_paths" not in e for e in out["evidence"])


def test_matching_trace_adds_paths_and_gist_provenance():
    trace = RecallTrace(query="who is the cto?",
                        enabled_channels=["dense", "gist"],
                        channel_results={"dense": ["m0", "m1"], "gist": ["m4"]},
                        gist_provenance={"m4": "g7"},
                        selected_candidates=["m4"],
                        latency_by_stage={"total_ms": 1.2})
    out = prove_answer(_answer(), trace)
    e = out["evidence"][0]
    assert e["recall_paths"] == ["gist"] and e["via_gist"] == "g7"
    assert out["recall_trace"]["enabled_channels"] == ["dense", "gist"]
    assert out["recall_trace"]["selected"] == ["m4"]


def test_mismatched_trace_is_treated_as_no_trace():
    trace = RecallTrace(query="a different question", channel_results={"gist": ["m4"]},
                        gist_provenance={"m4": "g7"})
    out = prove_answer(_answer(), trace)
    assert "recall_trace" not in out
    assert all("recall_paths" not in e for e in out["evidence"])   # no misattribution
