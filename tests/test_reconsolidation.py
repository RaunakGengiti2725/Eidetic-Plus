"""Required test: reconsolidation makes the system sharper with use, without ever
deleting a raw record.

- Confirmed recall -> FSRS up-weight (affinity maturation): stability grows,
  retrievability resets. (Re-embedding is verified separately in the vector-index test.)
- Contradicted recall -> suppress (down-weight), never delete.
- Memory linking by co-activation -> a strengthened edge between co-confirmed memories,
  kept out of the entity PPR graph and scoped."""
from __future__ import annotations

import time

from eidetic import fsrs
from eidetic.graph import CO_ACTIVATED, KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.store import RecordStore


def test_confirmed_recall_upweights():
    rec = MemoryRecord(text="confirmed fact", fsrs=fsrs.init_state(0.5, 0.5))
    s0, r_decayed_at = rec.fsrs.stability, rec.fsrs.last_review + 30 * 86400
    fsrs.decay(rec.fsrs, at=r_decayed_at)
    assert fsrs.current_retrievability(rec.fsrs, r_decayed_at) < 1.0
    fsrs.reinforce(rec.fsrs, importance=0.8, at=r_decayed_at)
    assert rec.fsrs.stability > s0                  # affinity maturation strengthens it
    assert abs(rec.fsrs.retrievability - 1.0) < 1e-6


def test_contradicted_recall_suppresses_but_keeps_record(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    rec = MemoryRecord(text="stale fact", fsrs=fsrs.init_state(0.9, 0.9))
    store.upsert_record(rec)
    s0 = rec.fsrs.stability
    fsrs.lapse(rec.fsrs)                            # contradicted -> suppress
    store.upsert_record(rec)
    assert rec.fsrs.stability < s0                  # down-weighted
    assert store.get_record(rec.memory_id) is not None  # but never deleted


def test_memory_linking_by_coactivation(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)
    A = Scope(namespace="alpha")
    m1, m2, m3 = "mem_a", "mem_b", "mem_c"
    added = g.link_memories([m1, m2, m3], scope=A, valid_at=time.time())
    assert added == 3                               # 3 pairwise co-activation edges
    assert set(g.linked_memories(m1, scope=A)) == {m2, m3}
    # co-activated links are scoped and excluded from the entity PPR graph
    assert g.linked_memories(m1, scope=Scope(namespace="beta")) == []
    nxg = g.build_nx(scope=A)
    assert m1 not in nxg.nodes()
    # the links are real edges in the store (nothing is faked)
    assert any(e.relation == CO_ACTIVATED for e in store.all_edges(A))
