"""Offline tests for the activation burst's scoring and its non-negotiable invariants:
age-independence, query-time (not recency) temporal scoring, scope isolation, bi-temporal
correctness, and co-activation multi-hop expansion."""
from __future__ import annotations

from datetime import datetime

from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.reflex_activation import build_memory_packet
from eidetic.reflex_index import ReflexIndex
from eidetic.store import RecordStore


def _rec(mid, text, *, namespace="default", valid_at=1.0, invalid_at=None, entities=None):
    return MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}",
                        text=text, scope=Scope(namespace=namespace), valid_at=valid_at,
                        invalid_at=invalid_at, entities=list(entities or []))


def _fixture(fresh_settings, records, *, links=None):
    store = RecordStore(fresh_settings.sqlite_path)
    for rec in records:
        store.upsert_record(rec)
    graph = KnowledgeGraph(store)
    if links:
        graph.link_memories(list(links), scope=Scope(), valid_at=1.0)
    index = ReflexIndex()
    index.rebuild_from_store(store)
    return store, graph, index


def _build(fresh_settings, query, store, graph, index, **kw):
    return build_memory_packet(query, Scope(), store=store, graph=graph, index=index,
                               settings=fresh_settings, **kw)


def test_score_axes_decomposed(fresh_settings):
    store, graph, index = _fixture(fresh_settings, [_rec("m1", "alpha beta gamma project")])
    p = _build(fresh_settings, "alpha project", store, graph, index)
    s = p.scores["m1"]
    assert s.lexical > 0.0
    assert s.aggregate > 0.0
    assert 0.0 <= s.match_strength <= 1.0


def test_age_independence_no_time_constraint(fresh_settings):
    """Identical content, very different ages, NO query time constraint -> identical scores.
    The activation burst must never reward recency."""
    young = _rec("young", "project alpha milestone details", valid_at=1_700_000_000.0)
    old = _rec("old", "project alpha milestone details", valid_at=1_000_000.0)
    store, graph, index = _fixture(fresh_settings, [young, old])
    p = _build(fresh_settings, "tell me about project alpha milestone", store, graph, index)
    assert {"young", "old"} <= set(p.candidate_ids())
    sy, so = p.scores["young"], p.scores["old"]
    assert abs(sy.aggregate - so.aggregate) < 1e-9
    assert abs(sy.match_strength - so.match_strength) < 1e-9
    assert sy.temporal == 0.0 and so.temporal == 0.0


def test_temporal_scores_query_overlap_not_recency(fresh_settings):
    in_range = datetime(2023, 6, 1).timestamp()
    out_range = datetime(2019, 6, 1).timestamp()
    a = _rec("inrange", "project milestone happened", valid_at=in_range)
    b = _rec("outrange", "project milestone happened", valid_at=out_range)
    store, graph, index = _fixture(fresh_settings, [a, b])
    # as_of in 2024 so both records are bi-temporally active; the query constrains 2023.
    as_of = datetime(2024, 1, 1).timestamp()
    p = _build(fresh_settings, "what project milestone happened in 2023", store, graph, index,
               as_of=as_of)
    assert p.scores["inrange"].temporal == 1.0
    assert p.scores["outrange"].temporal == 0.0
    # the in-range record outranks the out-range one purely on the query's time constraint.
    assert p.scores["inrange"].aggregate > p.scores["outrange"].aggregate


def test_scope_isolation_namespace_hard_boundary(fresh_settings):
    store, graph, index = _fixture(fresh_settings, [
        _rec("a1", "secret revenue figure", namespace="alpha"),
        _rec("b1", "secret revenue figure", namespace="beta"),
    ])
    p = build_memory_packet("secret revenue", Scope(namespace="alpha"), store=store, graph=graph,
                            index=index, settings=fresh_settings)
    assert p.candidate_ids() == ["a1"]


def test_bitemporal_invalidated_record_excluded(fresh_settings):
    # invalid_at strictly before as_of -> the store filters it out; reflex must not surface it.
    rec = _rec("gone", "alpha keyword", valid_at=1.0, invalid_at=1000.0)
    store, graph, index = _fixture(fresh_settings, [rec])
    p = _build(fresh_settings, "alpha keyword", store, graph, index, as_of=5000.0)
    assert "gone" not in p.candidate_ids()


def test_coactivation_pulls_in_linked_memory(fresh_settings):
    seed = _rec("seed", "alpha keyword unique target")
    linked = _rec("linked", "completely different beta wording")
    store, graph, index = _fixture(fresh_settings, [seed, linked], links=["seed", "linked"])
    p = _build(fresh_settings, "alpha keyword unique target", store, graph, index)
    assert "linked" in p.candidate_ids()           # pulled in via co-activation, not lexical match
    assert p.scores["linked"].coactivation > 0.0
    assert p.scores["linked"].lexical == 0.0
    assert "linked" in p.coactivation_paths.get("seed", [])


def test_hotset_boosts_but_does_not_invent_candidates(fresh_settings):
    store, graph, index = _fixture(fresh_settings, [
        _rec("m1", "alpha project keyword"),
        _rec("m2", "alpha project keyword"),
    ])
    p = _build(fresh_settings, "alpha project keyword", store, graph, index, hot_ids={"m1"})
    assert p.scores["m1"].hotset == 1.0
    assert p.scores["m2"].hotset == 0.0
    assert p.scores["m1"].aggregate > p.scores["m2"].aggregate
    # a hot id that is NOT a content/coactivation match is never conjured into the candidate set.
    p2 = _build(fresh_settings, "alpha project keyword", store, graph, index, hot_ids={"ghost"})
    assert "ghost" not in p2.candidate_ids()
