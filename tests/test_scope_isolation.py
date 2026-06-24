"""Required test: scope prevents cross-tool / cross-namespace bleed.

The offline tests assert the storage+graph isolation guarantee that the universal
plugin depends on (write in namespace A, invisible from namespace B). The engine-level
end-to-end test makes real embedding calls and skips without a key."""
from __future__ import annotations

import time

import pytest

from eidetic.config import get_settings
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.store import RecordStore


def test_store_scope_isolation(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    A, B = Scope(namespace="alpha"), Scope(namespace="beta")
    t = time.time()
    ra = MemoryRecord(content_hash="ha", text="alpha secret", scope=A, valid_at=t)
    rb = MemoryRecord(content_hash="hb", text="beta note", scope=B, valid_at=t)
    future = MemoryRecord(content_hash="hf", text="future", scope=A, valid_at=t + 100)
    expired = MemoryRecord(content_hash="he", text="expired", scope=A, valid_at=t - 100, expired_at=t - 1)
    store.upsert_record(ra)
    store.upsert_record(rb)
    store.upsert_record(future)
    store.upsert_record(expired)

    assert ra.memory_id in store.ids_in_scope(A)
    assert ra.memory_id not in store.ids_in_scope(B)       # no bleed A -> B
    assert rb.memory_id not in store.ids_in_scope(A)       # no bleed B -> A
    assert ra.memory_id in store.active_ids_at(t, scope=A)
    assert ra.memory_id not in store.active_ids_at(t, scope=B)
    assert [r.memory_id for r in store.active_records_at(t, scope=A)] == [ra.memory_id]
    assert [r.memory_id for r in store.active_records_at(t, scope=B)] == [rb.memory_id]
    assert store.count(A) == 3 and store.count(B) == 1


def test_dedup_is_per_scope(tmp_path):
    """Identical raw content in a different namespace gets a DISTINCT index record
    (raw bytes are shared globally by the substrate; the index entry is scoped)."""
    store = RecordStore(tmp_path / "db.sqlite")
    A, B = Scope(namespace="alpha"), Scope(namespace="beta")
    store.upsert_record(MemoryRecord(content_hash="shared", text="same text", scope=A))
    assert store.get_by_hash("shared", A) is not None
    assert store.get_by_hash("shared", B) is None          # not visible cross-namespace


def test_graph_contradiction_does_not_cross_namespace(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)
    t0 = time.time()
    g.add_fact("Alice", "role", "engineer", valid_at=t0, scope=Scope(namespace="alpha"))
    # Same subject+relation, different value, but a DIFFERENT namespace -> no contradiction.
    _, inv = g.add_fact("Alice", "role", "manager", valid_at=t0 + 10, scope=Scope(namespace="beta"))
    assert inv == []
    # Same namespace -> contradiction closes the old edge.
    _, inv2 = g.add_fact("Alice", "role", "director", valid_at=t0 + 20, scope=Scope(namespace="alpha"))
    assert len(inv2) == 1


def test_engine_end_to_end_scope_isolation(engine):
    """Write in namespace A, confirm a recall in namespace B cannot see it."""
    if not get_settings().has_api_key:
        pytest.skip("No DASHSCOPE_API_KEY: end-to-end scope test needs real embeddings.")
    A, B = Scope(namespace="teamA"), Scope(namespace="teamB")
    engine.ingest_text("The launch code for project Nimbus is 7741.", source="A",
                       extract_graph=False, scope=A)
    ans_a = engine.ask("What is the launch code for project Nimbus?", scope=A)
    ans_b = engine.ask("What is the launch code for project Nimbus?", scope=B)
    assert ans_a.retrieved_count >= 1
    assert ans_b.retrieved_count == 0          # namespace B sees nothing from A
