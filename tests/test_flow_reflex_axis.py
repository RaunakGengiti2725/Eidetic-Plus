"""Track 9 Task 3: the reflex hotset axis accepts continuous activation. Activation feeds RANKING
(aggregate) only -- never match_strength/coverage -- so instinct breaks ties and surfaces memories
but can never turn a content-less candidate into a confident hit. Binary hot_ids stays the flag-off
path (byte-identical)."""
from __future__ import annotations

from dataclasses import replace

from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.reflex_activation import build_memory_packet
from eidetic.reflex_index import ReflexIndex
from eidetic.store import RecordStore


def _rec(mid, text, *, ns="default", valid_at=1.0, invalid_at=None):
    return MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}", text=text,
                        scope=Scope(namespace=ns), valid_at=valid_at, invalid_at=invalid_at)


def _fixture(settings, records):
    store = RecordStore(settings.sqlite_path)
    for r in records:
        store.upsert_record(r)
    idx = ReflexIndex()
    idx.rebuild_from_store(store)
    return store, KnowledgeGraph(store), idx


def _packet(settings, query, store, graph, idx, **kw):
    return build_memory_packet(query, Scope(), store=store, graph=graph, index=idx,
                               settings=settings, **kw)


def test_activation_breaks_ties_among_equal_content(fresh_settings):
    s = replace(fresh_settings, reflex_w_activation=0.4)
    store, graph, idx = _fixture(s, [_rec("a", "project alpha keyword"),
                                     _rec("b", "project alpha keyword")])
    p = _packet(s, "project alpha keyword", store, graph, idx, activation={"a": 1.0})
    assert p.candidate_ids()[0] == "a"
    assert p.scores["a"].activation == 1.0 and p.scores["b"].activation == 0.0


def test_activation_never_changes_match_strength(fresh_settings):
    s = replace(fresh_settings, reflex_w_activation=0.4)
    store, graph, idx = _fixture(s, [_rec("a", "project alpha keyword")])
    base = _packet(s, "project alpha keyword", store, graph, idx)
    act = _packet(s, "project alpha keyword", store, graph, idx, activation={"a": 1.0})
    assert base.scores["a"].match_strength == act.scores["a"].match_strength
    assert base.coverage == act.coverage           # coverage gate untouched by activation


def test_binary_hotset_path_is_unchanged_without_activation(fresh_settings):
    s = replace(fresh_settings)
    store, graph, idx = _fixture(s, [_rec("a", "alpha"), _rec("b", "alpha")])
    p = _packet(s, "alpha", store, graph, idx, hot_ids={"a"})
    assert p.scores["a"].hotset == 1.0 and p.scores["b"].hotset == 0.0
    assert p.scores["a"].activation == 0.0         # no activation map -> activation axis is 0
