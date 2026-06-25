"""Offline tests for the reflex MemoryPacket: a sub-second, LOCAL-ONLY candidate path.

The builder takes no model client at all -- so "no network call" is a structural guarantee, not
something a test has to police. These tests pin the packet contract the API/MCP and the reader
path both consume, and the score contract that lets a reflex hit feed answer() without
spuriously abstaining (dense_score/rerank_score must be populated)."""
from __future__ import annotations

import time

from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, RetrievalCandidate, Scope
from eidetic.reflex import MemoryPacket, ReflexScore
from eidetic.reflex_activation import build_memory_packet
from eidetic.reflex_index import ReflexIndex
from eidetic.store import RecordStore


def _fixture(fresh_settings, records):
    store = RecordStore(fresh_settings.sqlite_path)
    for rec in records:
        store.upsert_record(rec)
    graph = KnowledgeGraph(store)
    index = ReflexIndex()
    index.rebuild_from_store(store)
    return store, graph, index


def _rec(mid, text, *, namespace="default", valid_at=1.0, entities=None):
    return MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}",
                        text=text, scope=Scope(namespace=namespace), valid_at=valid_at,
                        entities=list(entities or []))


def _packet(fresh_settings, query, records, **kw):
    store, graph, index = _fixture(fresh_settings, records)
    return build_memory_packet(query, Scope(), store=store, graph=graph, index=index,
                               settings=fresh_settings, **kw)


def test_packet_returns_matching_candidate(fresh_settings):
    p = _packet(fresh_settings, "What was the Helios project revenue?",
                [_rec("m1", "The Helios project quarterly revenue was 4.2 million dollars"),
                 _rec("m2", "Bob went hiking in the mountains last weekend")])
    assert isinstance(p, MemoryPacket)
    assert "m1" in p.candidate_ids()
    assert "m2" not in p.candidate_ids()
    assert p.coverage > 0.0


def test_packet_score_contract_is_populated(fresh_settings):
    """answer() reads dense_score (coverage gate) and rerank_score (confidence) off candidates.
    A reflex candidate MUST carry both, or a confident hit would spuriously abstain."""
    p = _packet(fresh_settings, "Helios project revenue",
                [_rec("m1", "The Helios project quarterly revenue was 4.2 million dollars")])
    cands = p.to_candidates()
    assert cands and all(isinstance(c, RetrievalCandidate) for c in cands)
    top = cands[0]
    assert top.dense_score > 0.0
    assert top.rerank_score > 0.0
    assert top.fused_score > 0.0
    # coverage == top candidate's dense_score: the gate and the downstream read agree.
    assert abs(p.coverage - max(c.dense_score for c in cands)) < 1e-9


def test_packet_has_per_candidate_score_breakdown(fresh_settings):
    p = _packet(fresh_settings, "Helios revenue",
                [_rec("m1", "Helios revenue grew")])
    assert "m1" in p.scores
    s = p.scores["m1"]
    assert isinstance(s, ReflexScore)
    for axis in ("entity", "lexical", "temporal", "coactivation", "hotset", "aggregate",
                 "match_strength"):
        assert hasattr(s, axis)


def test_packet_records_latency_breakdown(fresh_settings):
    p = _packet(fresh_settings, "Helios revenue", [_rec("m1", "Helios revenue grew")])
    assert "total" in p.latency_ms
    assert p.latency_ms["total"] >= 0.0


def test_packet_under_local_budget_on_1k_records(fresh_settings):
    records = [_rec(f"m{i}", f"record number {i} about projects and revenue and metrics")
               for i in range(1000)]
    records.append(_rec("hit", "the special Helios alpha keyword target memory"))
    store, graph, index = _fixture(fresh_settings, records)
    t0 = time.perf_counter()
    p = build_memory_packet("special Helios alpha keyword target", Scope(),
                            store=store, graph=graph, index=index, settings=fresh_settings)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert "hit" in p.candidate_ids()
    assert elapsed_ms < 100.0, f"reflex packet took {elapsed_ms:.1f}ms (budget 100ms)"


def test_packet_candidate_ids_are_deterministic(fresh_settings):
    records = [_rec(f"m{i}", "shared project revenue keyword") for i in range(15)]
    store, graph, index = _fixture(fresh_settings, records)
    a = build_memory_packet("project revenue keyword", Scope(), store=store, graph=graph,
                            index=index, settings=fresh_settings).candidate_ids()
    b = build_memory_packet("project revenue keyword", Scope(), store=store, graph=graph,
                            index=index, settings=fresh_settings).candidate_ids()
    assert a == b


def test_strong_lexical_only_query_is_a_confident_hit(fresh_settings):
    """A lowercase query with no extractable entity and no time window (entity/temporal/hot all 0)
    but a strong lexical match MUST still clear reflex_min_coverage -- otherwise reflex never fires
    on ordinary queries and only capitalized-entity / temporal queries activate the fast path."""
    p = _packet(fresh_settings, "quarterly revenue figures summary",
                [_rec("m1", "the quarterly revenue figures summary was published")])
    assert p.scores["m1"].entity == 0.0
    assert p.scores["m1"].temporal == 0.0
    assert p.coverage >= fresh_settings.reflex_min_coverage


def test_weak_lexical_overlap_is_a_miss(fresh_settings):
    """Precision guard: only a small fraction of query terms present -> below the bar -> the reflex
    coverage must stay under reflex_min_coverage so a recalibrated gate never feeds junk candidates."""
    p = _packet(fresh_settings, "alpha beta gamma delta epsilon omega",
                [_rec("m1", "alpha completely unrelated content here today")])
    assert p.coverage < fresh_settings.reflex_min_coverage


def test_packet_public_dict_is_serializable_without_record_bodies(fresh_settings):
    p = _packet(fresh_settings, "Helios revenue", [_rec("m1", "Helios revenue grew")])
    pub = p.public_dict()
    assert pub["query"] == "Helios revenue"
    assert "m1" in pub["candidate_ids"]
    assert "coverage" in pub and "latency_ms" in pub
    # snippet + content hash are present for proof-readiness; full record json is not.
    assert pub["snippets"]["m1"]
    assert pub["content_hashes"]["m1"] == "h_m1"
