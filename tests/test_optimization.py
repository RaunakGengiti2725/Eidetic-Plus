"""Offline unit tests for the optimization playbook (no key, no fabricated scores):
event-calendar date normalization, typed preference extraction, query-adaptive RRF,
conformal-threshold plumbing, edge-placement, and the sweep dry-run plan."""
from __future__ import annotations

from datetime import datetime

import pytest

from bench.calibrate import conformal_threshold
from bench.index_timing import run_timing
from bench.sweep import STAGES, plan, stage_assignment
from eidetic.bm25 import BM25, PersistentBM25
from eidetic.events import EventRecord, normalize_dates, parse_query, select_for_query
from eidetic.preferences import extract_preference, is_preference
from eidetic.retrieval import (
    _hippo2_seed_entities, _reader_model, _rrf, _temporal_context_order,
    compress_chunk, edge_place,
)

REF = datetime(2026, 6, 23, 12, 0, 0).timestamp()   # Tue 23 Jun 2026


# ---- 1. event-calendar date normalization (ranges, reference-relative) ----
def test_date_normalization_ranges():
    got = {d["expr"]: (d["start"], d["end"]) for d in
           normalize_dates("we met yesterday, in May 2023, and 3 days ago", REF)}
    assert got["yesterday"] == ("2026-06-22T00:00:00", "2026-06-22T23:59:59")
    assert got["May 2023"] == ("2023-05-01T00:00:00", "2023-05-31T23:59:59")  # month RANGE
    assert got["3 days ago"][0].startswith("2026-06-20")


def test_structured_selection_interval_overlap():
    p = parse_query("How many times did Caroline visit Paris in May 2023?", REF)
    assert p["operation"] == "count" and "Caroline" in p["entities"] and p["ranges"]
    evs = [EventRecord(subject="Caroline", verb="visit", object="Paris", fact="Caroline visited Paris",
                       start=datetime(2023, 5, 10).timestamp(), end=datetime(2023, 5, 10).timestamp()),
           EventRecord(subject="Caroline", verb="visit", object="Lyon", fact="Caroline visited Lyon",
                       start=datetime(2024, 1, 1).timestamp(), end=datetime(2024, 1, 1).timestamp())]
    sel = select_for_query(evs, p, REF)
    assert [e.object for e in sel] == ["Paris"]   # overlap + entity; reader does the counting


# ---- 2. typed preference extraction ----
def test_typed_preference_extraction():
    assert is_preference("user: I prefer window seats on long flights.")
    assert is_preference("I'm allergic to peanuts.")
    assert not is_preference("The meeting is at 3pm on Tuesday.")
    assert "window seats" in extract_preference("I prefer window seats.")


# ---- 3. query-adaptive / weighted RRF ----
def test_weighted_rrf_changes_ranking():
    a, b = ["x", "y"], ["y", "z"]
    fair = _rrf([a, b], 60)
    boosted = _rrf([a, b], 60, [0.1, 3.0])           # trust list b more
    assert boosted["z"] > fair["z"]
    assert _rrf([a], 60)["x"] > _rrf([a], 60)["y"]   # rank-1 beats rank-2


def test_query_adaptive_flags():
    namey = parse_query("What is order EX-7741 status?", REF)
    assert namey["is_namey"]
    multi = parse_query("How are Alice and Bob and Carol connected?", REF)
    assert multi["is_multihop"]


# ---- 4. conformal-threshold plumbing ----
def test_conformal_threshold_hits_target_precision():
    # Low-signal items are wrong; high-signal items are correct -> threshold should land
    # where answered-set precision >= 0.95.
    samples = ([{"signal": 0.9, "correct": True}] * 18
               + [{"signal": 0.8, "correct": True}] * 1
               + [{"signal": 0.2, "correct": False}] * 10)
    res = conformal_threshold(samples, target_precision=0.95)
    assert res["ok"] and res["precision"] >= 0.95
    assert res["threshold"] >= 0.8          # excludes the low-signal wrong ones
    assert conformal_threshold([], 0.95)["ok"] is False


# ---- 5. edge-placement (lost-in-the-middle) ----
def test_edge_placement_puts_top_at_edges():
    assert edge_place(["1", "2", "3", "4", "5"]) == ["1", "3", "5", "4", "2"]
    # highest (1) and 2nd-highest (2) end up at the two ends
    placed = edge_place(["A", "B", "C", "D"])
    assert placed[0] == "A" and placed[-1] == "B"


def test_compression_keeps_relevant_sentences():
    out = compress_chunk("The cat sat. Revenue rose to 9M in Q2. Birds migrate south.",
                         "revenue Q2", 0.34)
    assert "Revenue" in out and "cat" not in out
    # ratio >= 1.0 is a no-op
    assert compress_chunk("a. b. c.", "q", 1.0) == "a. b. c."


# ---- integration: events + preferences reach the assembled context (no LLM) ----
def test_assemble_context_surfaces_events_and_preferences(engine):
    from eidetic.events import EventRecord
    from eidetic.models import Scope

    ns = "teamX"
    t = datetime(2023, 5, 10).timestamp()
    engine.store.add_event(EventRecord(subject="Caroline", verb="visited", object="Paris",
                                       fact="Caroline visited Paris", start=t, end=t,
                                       namespace=ns, valid_at=t))
    engine.store.add_profile_line(ns, "prefers window seats", salience=0.8)
    # This is exactly what the neutral benchmark adapter calls (candidates empty here).
    blocks = engine.retriever.assemble_context(
        "Where did Caroline go in May 2023?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)
    assert "Caroline visited Paris" in joined          # event calendar reached the context
    assert "User preference: prefers window seats" in joined  # typed preference surfaced


# ---- sweep plan (coordinate descent, offline) ----
def test_sweep_plan_is_coordinate_descent():
    trials = plan(subset=50, runs=1)
    assert len(trials) == sum(len(vals) for _, vals in STAGES)
    stages = [stage for stage, _ in STAGES]
    assert trials[0]["stage"] == "READER_COT"   # benchmark-visible feature flags first
    assert "CONFLICT_RESOLVER" in stages
    assert "ABSTENTION_THRESHOLD" not in stages
    assert "CASCADE_CONFIDENCE" not in stages
    assert stage_assignment("COMPRESSION_RATIO", "0.75") == {
        "CONTEXT_COMPRESS": "1",
        "COMPRESSION_RATIO": "0.75",
    }


def test_persistent_bm25_matches_legacy_and_filters_scope(tmp_path):
    scope_items = [
        ("a", "Alice works at Acme"),
        ("b", "Alice works at Globex"),
    ]
    items = [
        *scope_items,
        ("c", "Bob likes tea"),
        ("noise", "Alice Alice Alice works nowhere"),
    ]
    legacy = BM25().index(scope_items).search("Alice works", 10)
    idx = PersistentBM25(tmp_path / "bm25.json")
    idx.index(items)
    idx.save()
    loaded = PersistentBM25(tmp_path / "bm25.json")
    assert loaded.search("Alice works", 10, allowed_ids={"a", "b"}) == legacy
    filtered = loaded.search("Alice works", 10, allowed_ids={"b"})
    assert [mid for mid, _ in filtered] == ["b"]
    assert filtered[0][1] > 0.0


def test_index_timing_harness_reports_p95(tmp_path):
    res = run_timing(doc_count=40, query_count=5, index_path=tmp_path / "bm25.json")
    assert res["doc_count"] == 40
    assert res["legacy_p95_ms"] >= 0.0
    assert res["persistent_p95_ms"] >= 0.0


def test_numpy_vector_search_allowed_ids_prevents_underfill(tmp_path):
    import numpy as np

    from eidetic.vector_index import NumpyVectorIndex

    idx = NumpyVectorIndex(tmp_path, 4, 2)
    idx.add("outside", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    idx.add("inside", np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32))
    hits = idx.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 1, allowed_ids={"inside"})
    assert hits[0][0] == "inside"


def test_retrieve_does_not_pad_zero_score_records(fresh_settings):
    from dataclasses import replace
    import numpy as np

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    settings = replace(fresh_settings, rerank_enabled=False)

    class EmptyIndex:
        def __len__(self):
            return 10

        def search(self, _qvec, _k, allowed_ids=None):
            return []

    class FakeClient:
        pass

    scope = Scope(namespace="pad")
    store = RecordStore(settings.sqlite_path)
    for i in range(5):
        store.upsert_record(MemoryRecord(
            memory_id=f"m{i}", content_hash=f"h{i}", text=f"unmatched {i}",
            scope=scope, valid_at=float(i),
        ))
    retriever = Retriever(store, EmptyIndex(), KnowledgeGraph(store), object(), FakeClient(), settings)
    assert retriever.retrieve("no lexical overlap", scope=scope,
                              qvec=np.ones(4, dtype=np.float32), use_recency=False) == []
    with_recency = retriever.retrieve("no lexical overlap", scope=scope,
                                      qvec=np.ones(4, dtype=np.float32), use_recency=True)
    assert 0 < len(with_recency) <= settings.final_topk


def test_context_compress_master_flag_controls_assembly(fresh_settings):
    from dataclasses import replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, RetrievalCandidate
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    text = "The cat sat. Revenue rose to 9M in Q2. Birds migrate south."
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text=text, valid_at=1.0)
    cand = RetrievalCandidate(record=rec, fused_score=1.0)
    store = RecordStore(fresh_settings.sqlite_path)

    off = replace(fresh_settings, context_compress_enabled=False, compression_ratio=0.34)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), off)
    assert "The cat sat" in " ".join(retriever.assemble_context("revenue Q2", [cand]))

    on = replace(fresh_settings, context_compress_enabled=True, compression_ratio=0.34)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), on)
    joined = " ".join(retriever.assemble_context("revenue Q2", [cand]))
    assert "Revenue rose" in joined and "The cat sat" not in joined


def test_temporal_rerank_orders_latest_context_by_timestamp(fresh_settings):
    from dataclasses import replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, RetrievalCandidate
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    old = RetrievalCandidate(
        record=MemoryRecord(memory_id="old", content_hash="old", text="Alice works at Acme.",
                            valid_at=10.0),
        fused_score=10.0,
    )
    new = RetrievalCandidate(
        record=MemoryRecord(memory_id="new", content_hash="new", text="Alice works at Globex.",
                            valid_at=20.0),
        fused_score=1.0,
    )
    store = RecordStore(fresh_settings.sqlite_path)

    off = replace(fresh_settings, temporal_rerank_enabled=False)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), off)
    default_joined = "\n".join(retriever.assemble_context("What is Alice's latest job?", [old, new]))
    assert default_joined.index("Acme") < default_joined.index("Globex")

    on = replace(fresh_settings, temporal_rerank_enabled=True)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), on)
    temporal_joined = "\n".join(retriever.assemble_context("What is Alice's latest job?", [old, new]))
    assert temporal_joined.index("Globex") < temporal_joined.index("Acme")


def test_temporal_rerank_does_not_treat_last_week_as_latest(fresh_settings):
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate

    newer_low_score = RetrievalCandidate(
        record=MemoryRecord(memory_id="new", content_hash="new", text="new", valid_at=20.0),
        fused_score=1.0,
    )
    older_high_score = RetrievalCandidate(
        record=MemoryRecord(memory_id="old", content_hash="old", text="old", valid_at=10.0),
        fused_score=10.0,
    )
    parsed = parse_query("What did Alice do last week?", reference_time=30.0)
    ordered = _temporal_context_order(
        "What did Alice do last week?", parsed, [newer_low_score, older_high_score]
    )
    assert [c.record.memory_id for c in ordered] == ["old", "new"]


def test_hippo2_query_to_triple_seeding_is_scoped_and_relation_aware(tmp_path):
    from eidetic.events import parse_query
    from eidetic.graph import KnowledgeGraph
    from eidetic.models import Scope
    from eidetic.store import RecordStore

    store = RecordStore(tmp_path / "db.sqlite")
    graph = KnowledgeGraph(store)
    alpha = Scope(namespace="alpha")
    beta = Scope(namespace="beta")
    graph.add_fact("Alice", "works_at", "Acme", valid_at=10.0, scope=alpha)
    graph.add_fact("Alice", "lives_in", "Paris", valid_at=10.0, scope=alpha)
    graph.add_fact("Alice", "works_at", "BetaCorp", valid_at=10.0, scope=beta)

    parsed = parse_query("Where does Alice work now?", reference_time=20.0)
    seeds = _hippo2_seed_entities("Where does Alice work now?", parsed, store, 20.0, alpha)
    assert "Alice" in seeds and "Acme" in seeds
    assert "Paris" not in seeds
    assert "BetaCorp" not in seeds


def test_reader_router_flag_can_pin_product_reader(fresh_settings):
    from dataclasses import replace

    routed = replace(fresh_settings, reader_router_enabled=True,
                     salience_model="qwen-flash", gen_model="qwen3-max")
    pinned = replace(fresh_settings, reader_router_enabled=False,
                     salience_model="qwen-flash", gen_model="qwen3-max")
    assert _reader_model("What is Alice's favorite tea?", routed) == "qwen-flash"
    assert _reader_model("What is Alice's favorite tea?", pinned) == "qwen3-max"


def test_extract_light_uses_salience_model(fresh_settings):
    from dataclasses import replace

    from eidetic.dashscope_client import DashScopeClient

    settings = replace(fresh_settings, extract_light_enabled=True,
                       salience_model="qwen-flash", extract_model="qwen-plus")
    client = DashScopeClient.__new__(DashScopeClient)
    client.settings = settings
    seen = {}

    def fake_chat(model, *_args, **_kw):
        # extraction now calls chat() (raw string) + a truncation-resilient parser, not chat_json.
        seen["model"] = model
        return '{"triples": [{"src": "Alice", "relation": "likes", "dst": "tea"}]}'

    client.chat = fake_chat
    triples = client.extract_edges("Alice likes tea.")
    assert seen["model"] == "qwen-flash"
    assert triples[0]["src"] == "Alice"


def test_rerank_failure_is_loud_unless_fail_open(fresh_settings):
    from dataclasses import replace
    import numpy as np

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    scope = Scope(namespace="rerank")
    store = RecordStore(fresh_settings.sqlite_path)
    store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="h1", text="Alice likes green tea.",
        scope=scope, valid_at=1.0,
    ))

    class OneIndex:
        def __len__(self):
            return 1

        def search(self, _qvec, _k, allowed_ids=None):
            return [("m1", 0.9)]

    class BadReranker:
        def rerank(self, *_args, **_kw):
            raise RuntimeError("rerank quota exhausted")

    strict = replace(fresh_settings, rerank_enabled=True, rerank_fail_open=False,
                     rerank_depth=1, final_topk=1)
    retriever = Retriever(store, OneIndex(), KnowledgeGraph(store), object(), BadReranker(), strict)
    with pytest.raises(RuntimeError, match="rerank quota exhausted"):
        retriever.retrieve("Alice tea", at=2.0, scope=scope,
                           qvec=np.ones(4, dtype=np.float32))

    fail_open = replace(strict, rerank_fail_open=True)
    retriever = Retriever(store, OneIndex(), KnowledgeGraph(store), object(), BadReranker(), fail_open)
    hits = retriever.retrieve("Alice tea", at=2.0, scope=scope,
                              qvec=np.ones(4, dtype=np.float32))
    assert [h.record.memory_id for h in hits] == ["m1"]
