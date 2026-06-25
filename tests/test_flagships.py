"""Offline tests for the read-only flagship functions: prove_answer + memory_health_report."""
from __future__ import annotations

import asyncio

from eidetic.models import Answer, Citation, Edge, MemoryRecord, NLILabel, Scope
from eidetic.proofs import prove_answer


def test_prove_answer_builds_a_grounded_proof_tree():
    ans = Answer(
        question="where does alice work", answer="Globex", verified=True, confidence=0.9,
        citations=[
            Citation(memory_id="m1", content_hash="h1", raw_uri="u1", source="user",
                     valid_at=10.0, snippet="Alice works at Globex",
                     nli_label=NLILabel.ENTAILMENT, nli_score=0.95),
            Citation(memory_id="m0", content_hash="h0", raw_uri="u0", source="user",
                     valid_at=5.0, snippet="Alice works at Acme",
                     nli_label=NLILabel.CONTRADICTION, nli_score=0.8),
        ],
    )
    proof = prove_answer(ans)
    assert proof["claim"] == "Globex" and proof["verified"] is True
    assert proof["grounded_count"] == 1                 # only the entailed citation grounds it
    assert proof["evidence_count"] == 2
    assert len(proof["contradictions"]) == 1 and proof["contradictions"][0]["memory_id"] == "m0"
    assert proof["provenance_complete"] is True          # every cited memory has a content hash
    assert proof["evidence"][0]["content_hash"] == "h1"


def test_prove_answer_flags_incomplete_provenance():
    ans = Answer(question="q", answer="x", verified=False,
                 citations=[Citation(memory_id="m", content_hash="", raw_uri="", source="s",
                                     valid_at=0.0, nli_label=NLILabel.NEUTRAL)])
    assert prove_answer(ans)["provenance_complete"] is False


def test_memory_health_report_counts_from_store(engine):
    scope = Scope(namespace="health")
    for i in range(4):
        engine.store.upsert_record(MemoryRecord(
            memory_id=f"m{i}", content_hash=f"h{i}", text=f"fact {i}",
            entities=(["alice"] if i < 3 else []), scope=scope, valid_at=float(i)))
    # one active edge + one bi-temporally closed (superseded) edge
    engine.store.add_edge(Edge(src="Alice", dst="Acme", relation="works_at",
                               scope=scope, valid_at=1.0))
    engine.store.add_edge(Edge(src="Alice", dst="Globex", relation="works_at", scope=scope,
                               valid_at=2.0, invalid_at=None, expired_at=3.0))
    rep = engine.memory_health_report(scope)
    assert rep["memories"] == 4
    assert rep["orphan_records"] == 1                    # m3 has no entities
    assert rep["distinct_entities"] == 1                 # 'alice'
    assert rep["contradiction_load"] == 1                # the expired_at edge
    assert rep["edges"] == 2
    assert rep["has_api_key"] in (True, False)           # real, never fabricated


def test_mcp_health_report_tool_works_without_key(mcp_engine_for_health):
    import eidetic.mcp_server as m
    out = m.health_report(namespace="empty")
    assert out["memories"] == 0 and isinstance(out, dict)
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert "health_report" in names


import pytest


@pytest.fixture()
def mcp_engine_for_health(fresh_settings, monkeypatch):
    import eidetic.mcp_server as m
    from eidetic.engine import Engine
    eng = Engine(fresh_settings, client=object())
    monkeypatch.setattr(m, "_engine", eng)
    yield eng
    monkeypatch.setattr(m, "_engine", None)
