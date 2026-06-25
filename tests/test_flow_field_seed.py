"""Track 9 Task 3.5: field-seeded candidates -- the instinct upgrade. A hot memory the query never
named gets unioned into the seed set and pulled into candidates, but the existing store-load gates
it (scope + bi-temporal validity), and match_strength stays content-only so it can never raise
coverage or bypass abstention. Instinct surfaces; it never fabricates."""
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


def test_activation_seeds_unnamed_memory(fresh_settings):
    s = replace(fresh_settings, flow_field_seed=True, flow_seed_topk=8, reflex_w_activation=0.4)
    store, graph, idx = _fixture(s, [_rec("m_quiet", "completely different beta wording"),
                                     _rec("m_named", "alpha keyword target")])
    p = build_memory_packet("alpha keyword target", Scope(), store=store, graph=graph, index=idx,
                            settings=s, activation={"m_quiet": 1.0})
    assert "m_quiet" in p.candidate_ids()                 # instinct surfaced an unnamed memory
    assert p.scores["m_quiet"].match_strength == 0.0      # but it cannot inflate coverage


def test_field_seed_respects_store_scope_and_validity(fresh_settings):
    s = replace(fresh_settings, flow_field_seed=True)
    rec_gone = _rec("gone", "beta", valid_at=1.0, invalid_at=100.0)
    store, graph, idx = _fixture(s, [_rec("other", "beta", ns="other"), rec_gone,
                                     _rec("here", "alpha keyword")])
    p = build_memory_packet("alpha keyword", Scope(namespace="default"), store=store, graph=graph,
                            index=idx, settings=s, as_of=5000.0,
                            activation={"other": 1.0, "gone": 1.0})
    assert "other" not in p.candidate_ids()               # cross-namespace activated id dropped
    assert "gone" not in p.candidate_ids()                # invalid-at-as_of activated id dropped


def test_field_seed_off_does_not_seed(fresh_settings):
    s = replace(fresh_settings, flow_field_seed=False)
    store, graph, idx = _fixture(s, [_rec("m_quiet", "beta wording"),
                                     _rec("m_named", "alpha keyword")])
    p = build_memory_packet("alpha keyword", Scope(), store=store, graph=graph, index=idx,
                            settings=s, activation={"m_quiet": 1.0})
    assert "m_quiet" not in p.candidate_ids()             # flag off -> no field seed
