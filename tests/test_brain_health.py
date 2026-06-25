"""Offline tests for BrainHealthScore + the updated-fact supersession invariant (Phase 8)."""
from __future__ import annotations

from dataclasses import replace

from eidetic.engine import Engine
from eidetic.graph import KnowledgeGraph
from eidetic.models import BrainEventType, MemoryRecord, Scope, now
from eidetic.store import RecordStore


def test_brain_health_score_is_bounded_and_decomposed(engine):
    out = engine.brain_health_score(scope=Scope(namespace="empty"))
    assert 0.0 <= out["brain_health_score"] <= 1.0
    for key in ("recall_connectivity", "proof_coverage", "temporal_coverage",
                "channel_diversity", "orphan_rate", "contradiction_rate",
                "stale_gist_rate", "unsupported_answer_rate"):
        assert key in out["components"]


def test_brain_health_reflects_events_and_orphans(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True))
    scope = Scope(namespace="ns")
    # one orphan record (no entities) + a mix of verified/missed answers.
    e.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1", text="fact",
                                       scope=scope, valid_at=now(), entities=[]))
    e._brain(BrainEventType.ANSWER_VERIFIED, namespace="ns")
    e._brain(BrainEventType.RETRIEVAL_MISSED, namespace="ns")
    out = e.brain_health_score(scope=scope)
    assert out["components"]["proof_coverage"] == 0.5         # 1 verified of 2 answered
    assert out["components"]["unsupported_answer_rate"] == 0.5
    assert out["components"]["orphan_rate"] == 1.0           # the only record is an orphan


# ---- INVARIANT: an updated fact supersedes the old one WITHOUT deleting history -------------
def test_updated_fact_supersedes_without_deleting(fresh_settings, tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)
    scope = Scope(namespace="ns")
    g.add_fact("Alice", "works_at", "Acme", valid_at=100.0, scope=scope)
    g.add_fact("Alice", "works_at", "Globex", valid_at=200.0, scope=scope)   # the update

    # As of the later time, only the new value is active...
    active_dst = {e.dst for e in store.active_edges_at(250.0, scope)
                  if e.src == "Alice" and e.relation == "works_at"}
    assert active_dst == {"Globex"}
    # ...but the old fact is CLOSED (invalidated), never deleted -- full history is retained.
    all_dst = {e.dst for e in store.all_edges(scope)
               if e.src == "Alice" and e.relation == "works_at"}
    assert all_dst == {"Acme", "Globex"}
    acme = next(e for e in store.all_edges(scope) if e.dst == "Acme")
    assert acme.invalid_at == 200.0                          # closed at the update time
