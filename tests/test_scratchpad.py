"""Offline tests for the working scratchpad + salience explanation (Phase 6 + 7)."""
from __future__ import annotations

import types

from eidetic.models import MemoryRecord, Scope, now
from eidetic.scratchpad import select_scratchpad


def test_select_scratchpad_orders_by_salience_then_helpful_and_filters():
    recs = [
        types.SimpleNamespace(memory_id="a", content_hash="ha", text="A", summary=None,
                              salience=0.9, verified_helpful_count=1),
        types.SimpleNamespace(memory_id="b", content_hash="hb", text="B", summary=None,
                              salience=0.9, verified_helpful_count=3),
        types.SimpleNamespace(memory_id="c", content_hash="hc", text="C", summary=None,
                              salience=0.4, verified_helpful_count=9),
    ]
    sp = select_scratchpad(recs, top_k=5, min_salience=0.6)
    assert [e["memory_id"] for e in sp] == ["b", "a"]      # low-salience c filtered; b>a on usage
    assert all(e["content_hash"] for e in sp)              # every entry links to a raw source hash


def test_build_scratchpad_expires_superseded_facts(engine):
    scope = Scope(namespace="sp")
    engine.store.upsert_record(MemoryRecord(memory_id="active", content_hash="ha",
                                            text="current fact", scope=scope,
                                            valid_at=now() - 100, salience=0.9))
    stale = MemoryRecord(memory_id="stale", content_hash="hs", text="old fact", scope=scope,
                         valid_at=now() - 1000, salience=0.95)
    stale.invalid_at = now() - 10                          # superseded / closed
    engine.store.upsert_record(stale)
    ids = {e["memory_id"] for e in engine.build_scratchpad(scope, min_salience=0.5)}
    assert "active" in ids and "stale" not in ids          # superseded fact expired from scratchpad


def test_salience_explanation_exposes_components_and_provenance(engine):
    scope = Scope(namespace="sp")
    engine.store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="h1", text="the trophy day", scope=scope, valid_at=now(),
        salience=0.8, importance=0.6, surprise=0.5, verified_helpful_count=2,
        metadata={"arousal": 0.7, "valence": 0.3, "emphasis": 0.5}))
    ex = engine.salience_explanation("m1")
    assert ex["salience"] == 0.8
    assert ex["components"]["arousal"] == 0.7 and ex["components"]["verified_helpful_count"] == 2
    assert ex["provenance"]["content_hash"] == "h1"
    assert engine.salience_explanation("not_a_memory") is None


def test_scratchpad_off_by_default(fresh_settings):
    assert fresh_settings.scratchpad_enabled is False      # baseline context unchanged
