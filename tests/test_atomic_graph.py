"""F2: the contradiction closure must be atomic -- concurrent conflicting facts leave exactly one
active edge, with full history retained (offline)."""
from __future__ import annotations

import threading

from eidetic.graph import KnowledgeGraph
from eidetic.models import Scope
from eidetic.store import RecordStore


def test_concurrent_conflicting_facts_leave_exactly_one_active(fresh_settings):
    store = RecordStore(fresh_settings.sqlite_path)
    g = KnowledgeGraph(store, deterministic_conflicts=True)
    scope = Scope(namespace="race")
    errors: list = []

    def add(company, t):
        try:
            g.add_fact("Alice", "works_at", company, valid_at=t, scope=scope)
        except Exception as e:
            errors.append(repr(e))

    # 20 concurrent conflicting "Alice works_at X" writes at distinct valid_at times.
    threads = [threading.Thread(target=add, args=(f"Co{i}", 100.0 + i)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    active = [e for e in store.active_edges_at(10_000.0, scope)
              if e.src == "Alice" and e.relation == "works_at"]
    assert len(active) == 1                      # exactly one active value, despite the race
    assert active[0].dst == "Co19"               # deterministic: the latest valid_at wins
    # history retained: every conflicting fact still exists (closed, never deleted).
    allw = [e for e in store.all_edges(scope) if e.src == "Alice" and e.relation == "works_at"]
    assert len(allw) == 20
