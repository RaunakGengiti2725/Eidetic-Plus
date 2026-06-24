"""Required test: a contradicting fact invalidates (closes) the old edge bi-temporally,
but never deletes it -- the full history stays queryable at any point in time."""
from __future__ import annotations

import time

from eidetic.graph import KnowledgeGraph
from eidetic.models import Scope
from eidetic.store import RecordStore


def test_contradiction_closes_old_edge_keeps_history(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)

    t0 = time.time()
    old, inv = g.add_fact("Alice", "works_at", "Acme", valid_at=t0)
    assert inv == []

    t1 = t0 + 365 * 86400  # one year later she changes jobs
    new, invalidated = g.add_fact("Alice", "works_at", "Globex", valid_at=t1)
    assert len(invalidated) == 1
    assert invalidated[0].edge_id == old.edge_id

    edges = store.all_edges()
    # Nothing deleted: both edges persist.
    assert len(edges) == 2
    old_db = next(e for e in edges if e.edge_id == old.edge_id)
    new_db = next(e for e in edges if e.edge_id == new.edge_id)

    # Time-travel queries: the old fact was true before the switch, false after.
    assert old_db.is_active_at(t0 + 30 * 86400) is True
    assert old_db.is_active_at(t1 + 30 * 86400) is False
    assert new_db.is_active_at(t1 + 30 * 86400) is True
    assert new_db.is_active_at(t0 + 30 * 86400) is False  # not yet valid back then


def test_non_contradicting_facts_coexist(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)
    t0 = time.time()
    g.add_fact("Alice", "works_at", "Acme", valid_at=t0)
    _, inv = g.add_fact("Alice", "lives_in", "Paris", valid_at=t0)  # different relation
    assert inv == []  # no contradiction
    assert len(store.all_edges()) == 2
    assert all(e.is_active_at(t0 + 10) for e in store.all_edges())


def test_seed_neighborhood_ppr_excludes_unrelated_edges(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)
    t0 = time.time()
    g.add_fact("Alice", "works_at", "Acme", valid_at=t0)
    g.add_fact("Acme", "located_in", "Paris", valid_at=t0)
    for i in range(50):
        g.add_fact(f"Unrelated{i}", "knows", f"Other{i}", valid_at=t0)

    local = g.build_seed_neighborhood_nx(["Alice"], at=t0 + 1)
    assert "alice" in local
    assert "acme" in local
    assert "paris" in local
    assert "unrelated0" not in local

    scores = g.ppr_entities(["Alice"], at=t0 + 1)
    assert "acme" in scores
    assert "unrelated0" not in scores


def test_active_edges_touching_many_filters_in_sql_semantics(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)
    alpha = Scope(namespace="alpha")
    beta = Scope(namespace="beta")
    t0 = time.time()
    g.add_fact("Alice", "works_at", "Acme", valid_at=t0, scope=alpha)
    g.add_fact("Acme", "located_in", "Paris", valid_at=t0, scope=alpha)
    g.add_fact("Alice", "works_at", "BetaCorp", valid_at=t0, scope=beta)
    g.add_fact("FutureAlice", "works_at", "Mars", valid_at=t0 + 100, scope=alpha)

    rows = store.active_edges_touching_many({"alice"}, t0 + 1, scope=alpha)
    assert [(e.src, e.dst) for e in rows] == [("Alice", "Acme")]
    acme_rows = store.active_edges_touching_many({"ACME"}, t0 + 1, scope=alpha)
    assert {(e.src, e.dst) for e in acme_rows} == {("Alice", "Acme"), ("Acme", "Paris")}
    assert store.active_edges_touching_many({"futurealice"}, t0 + 1, scope=alpha) == []
