"""Offline tests for the Connected Brain Loop spine (no key).

The spine is OBSERVATION-ONLY: enabling RECALL_TRACE/BRAIN_EVENTS must not change the
candidate list, candidate order, or answer text. These tests pin that invariant (the
discriminating check is `retrieve()` returns identical ids+order trace-on vs trace-off),
plus the EvidencePacket builder, the QualityGate predicates, and the BrainEventLog.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from eidetic.brain import BrainEventLog, QualityGate, build_evidence_packets
from eidetic.graph import KnowledgeGraph
from eidetic.models import (Answer, BrainEvent, BrainEventType, Citation, MemoryRecord,
                            NLILabel, RecallTrace, Scope)
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _FakeIndex:
    """Dense returns what the test programs; honors allowed_ids; exposes vectors for MMR/gist."""
    def __init__(self, dense, vecs=None, struct=None):
        self.dense, self.vecs, self.struct = dense, vecs or {}, struct or []

    def __len__(self):
        return max(len(self.dense), len(self.vecs), 1)

    def search(self, q, k, allowed_ids=None, ef=None):
        items = [(m, s) for m, s in self.dense if allowed_ids is None or m in allowed_ids]
        return sorted(items, key=lambda x: -x[1])[:k]

    def search_struct(self, qs, k):
        return self.struct[:k]

    def get_vectors(self, ids):
        return {m: self.vecs[m] for m in ids if m in self.vecs}


def _retriever(tmp_path, settings, index, ns="proj", n=5):
    store = RecordStore(tmp_path / "db.sqlite")
    scope = Scope(namespace=ns)
    for i in range(n):
        store.upsert_record(MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
                                         text=f"memory {i}", scope=scope, valid_at=1.0))
    return Retriever(store, index, KnowledgeGraph(store), object(), object(), settings), store, scope


# ---- RecallTrace: present only when on, absent when off ------------------------------------
def test_trace_absent_when_flag_off(fresh_settings, tmp_path):
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)])
    r, _, scope = _retriever(tmp_path, replace(fresh_settings, rerank_enabled=False), idx)
    r.retrieve("q", scope=scope, qvec=np.array([1.0, 0.0], np.float32), use_recency=False)
    assert r.last_trace is None                       # no trace built when RECALL_TRACE is off


def test_trace_built_and_well_formed_when_on(fresh_settings, tmp_path):
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)])
    s = replace(fresh_settings, rerank_enabled=False, recall_trace_enabled=True)
    r, _, scope = _retriever(tmp_path, s, idx)
    out = r.retrieve("q", scope=scope, qvec=np.array([1.0, 0.0], np.float32), use_recency=False)
    t = r.last_trace
    assert isinstance(t, RecallTrace) and t.query == "q"
    assert "dense" in t.enabled_channels and "bm25" in t.channel_results
    sel = {c.record.memory_id for c in out}
    assert set(t.selected_candidates) == sel
    assert set(t.selected_candidates) <= set(t.fused_scores)        # selected drawn from fused
    assert not (set(t.selected_candidates) & set(t.dropped_candidates))  # disjoint
    assert "total_ms" in t.latency_by_stage


def test_trace_is_a_true_side_channel_identical_results(fresh_settings, tmp_path):
    # THE discriminating check: identical candidate ids AND order with the trace on vs off.
    idx_off = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)])
    idx_on = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)])
    qvec = np.array([1.0, 0.0], np.float32)
    r_off, _, sc = _retriever(tmp_path / "off", replace(fresh_settings, rerank_enabled=False), idx_off)
    r_on, _, sc2 = _retriever(tmp_path / "on",
                              replace(fresh_settings, rerank_enabled=False, recall_trace_enabled=True),
                              idx_on)
    ids_off = [c.record.memory_id for c in r_off.retrieve("q", scope=sc, qvec=qvec, use_recency=False)]
    ids_on = [c.record.memory_id for c in r_on.retrieve("q", scope=sc2, qvec=qvec, use_recency=False)]
    assert ids_off == ids_on                          # trace never perturbs ranking/order
    assert QualityGate.flag_off_preserves_baseline(ids_off, ids_on)


# ---- EvidencePacket builder ----------------------------------------------------------------
def test_evidence_packets_carry_paths_and_gist_provenance():
    ans = Answer(question="q", answer="a",
                 citations=[Citation(memory_id="m4", content_hash="h4", raw_uri="u", source="user",
                                     valid_at=1.0, snippet="snip", nli_label=NLILabel.ENTAILMENT,
                                     nli_score=0.9)])
    trace = RecallTrace(query="q", channel_results={"dense": ["m0"], "gist": ["m4", "m0"]},
                        gist_provenance={"m4": "g1"})
    pkts = build_evidence_packets(ans, trace)
    assert len(pkts) == 1
    p = pkts[0]
    assert p.retrieval_paths == ["gist"] and p.derived_from == "g1"
    assert p.channel_scores["gist"] == 2.0            # rank 0 of a 2-long list -> score 2
    assert p.nli_label == NLILabel.ENTAILMENT and p.raw_span == "snip"


def test_evidence_packets_drop_paths_on_trace_mismatch():
    # A stale trace (from a different query / a cache hit) must not misattribute paths.
    ans = Answer(question="other-question", answer="a",
                 citations=[Citation(memory_id="m4", content_hash="h4", raw_uri="", source="user",
                                     valid_at=1.0)])
    trace = RecallTrace(query="q", channel_results={"gist": ["m4"]}, gist_provenance={"m4": "g1"})
    p = build_evidence_packets(ans, trace)[0]
    assert p.retrieval_paths == [] and p.derived_from == ""   # no paths when trace != answer


# ---- QualityGate predicates ----------------------------------------------------------------
def test_quality_gate_predicates():
    assert QualityGate.flag_off_preserves_baseline(["a", "b"], ["a", "b"]) is True
    assert QualityGate.flag_off_preserves_baseline(["a", "b"], ["b", "a"]) is False   # order matters
    assert QualityGate.no_raw_mutation(["h1", "h2"], ["h2", "h1"]) is True            # set equality
    assert QualityGate.no_raw_mutation(["h1"], ["h1", "h2"]) is False
    assert QualityGate.no_scope_leak([Scope(namespace="a")], Scope(namespace="a")) is True
    assert QualityGate.no_scope_leak([Scope(namespace="b")], Scope(namespace="a")) is False
    assert QualityGate.no_age_bias(0.01, 0.2) is True
    assert QualityGate.no_age_bias(0.2, 0.0) is False        # recall slopes with age -> fail
    assert QualityGate.proof_coverage_not_dropped(2, 3) is True
    assert QualityGate.proof_coverage_not_dropped(3, 2) is False
    assert QualityGate.latency_within_budget(50.0, 100.0) is True
    assert QualityGate.latency_within_budget(150.0, 100.0) is False


def test_quality_gate_evaluate_aggregates_and_fails_closed():
    ok = QualityGate.evaluate("feat", checks={"baseline": True, "raw": True})
    assert ok.passed is True and ok.checks == {"baseline": True, "raw": True}
    bad = QualityGate.evaluate("feat", checks={"baseline": True, "raw": False})
    assert bad.passed is False
    empty = QualityGate.evaluate("feat", checks={})
    assert empty.passed is False                            # proving nothing fails closed


# ---- BrainEventLog -------------------------------------------------------------------------
def test_brain_event_log_is_bounded_and_typed():
    log = BrainEventLog(capacity=3)
    for i in range(5):
        log.emit(BrainEvent(type=BrainEventType.MEMORY_INGESTED, namespace="x",
                            memory_ids=[f"m{i}"]))
    log.emit(BrainEvent(type=BrainEventType.ANSWER_VERIFIED, namespace="x"))
    assert len(log) == 3                                    # oldest evicted past capacity
    assert log.by_type(BrainEventType.ANSWER_VERIFIED)
    assert log.counts().get("answer_verified") == 1
    assert log.recent(1)[0].type == BrainEventType.ANSWER_VERIFIED   # newest first


# ---- engine emission is gated --------------------------------------------------------------
def test_engine_brain_emission_is_flag_gated(fresh_settings):
    from eidetic.engine import Engine
    on = Engine(replace(fresh_settings, brain_events_enabled=True))
    on._brain(BrainEventType.MEMORY_INGESTED, namespace="x", memory_ids=["m1"])
    assert len(on.brain_log) == 1
    off = Engine(fresh_settings)                            # BRAIN_EVENTS default off
    off._brain(BrainEventType.MEMORY_INGESTED, namespace="x", memory_ids=["m1"])
    assert len(off.brain_log) == 0                          # no emission when off


# ---- explain_candidate ('why this memory?') ------------------------------------------------
def test_explain_candidate_from_trace(engine):
    engine.retriever.last_trace = RecallTrace(
        query="q", channel_results={"dense": ["m1", "m2"], "gist": ["m1"]},
        channel_weights={"dense": 1.0, "gist": 0.4},
        fused_scores={"m1": 0.05, "m2": 0.03}, gist_provenance={"m1": "g1"},
        selected_candidates=["m1"])
    ex = engine.explain_candidate("m1")
    assert ex["retrieval_paths"] == ["dense", "gist"]
    assert ex["channel_ranks"]["dense"] == 2               # rank 0 of a 2-long list
    assert ex["channel_weights"]["gist"] == 0.4
    assert ex["via_gist"] == "g1" and ex["selected"] is True
    assert engine.explain_candidate("not_in_trace") is None


def test_recall_trace_visible_across_threads_after_ask(fresh_settings, tmp_path):
    """The retriever trace is thread-local (concurrent asks must not clobber each other), but
    the ENGINE accessor must still serve 'the most recent traced recall' to introspection
    surfaces (MCP recall_trace, /api/recall_trace) that run on a different worker thread."""
    import hashlib
    import re as _re
    import threading

    from eidetic.engine import Engine

    class _Client:
        def __init__(self, dim):
            self.dim = dim

        def _e(self, t):
            v = np.zeros(self.dim, np.float32)
            for tok in _re.findall(r"[a-z0-9]+", (t or "").lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        def embed_text(self, t):
            return self._e(t)

        def embed_texts(self, ts):
            return (np.stack([self._e(t) for t in ts])
                    if ts else np.zeros((0, self.dim), np.float32))

        def extract_edges(self, text):
            return []

        def generate_answer(self, q, blocks, model=None):
            return blocks[0][:200] if blocks else "I do not have that in memory."

        def nli(self, premise, hypothesis):
            return ("entailment", 0.9)

    s = replace(fresh_settings, rerank_enabled=False, recall_trace_enabled=True)
    e = Engine(s, client=_Client(s.embed_dim))
    scope = Scope(namespace="xthread")
    e.ingest_text("The berry harvest starts in June.", scope=scope, consolidate_now=False)

    def _ask():
        e.ask("when does the berry harvest start", scope=scope)

    t = threading.Thread(target=_ask)
    t.start()
    t.join()

    trace = e.recall_trace(scope=scope)    # main thread: worker's thread-local is invisible
    assert trace is not None
    assert "berry" in trace.query

    # scope isolation: another namespace must never see this trace (query text + memory ids
    # would otherwise leak across the namespace boundary)
    assert e.recall_trace(scope=Scope(namespace="other")) is None
    # no-arg call resolves the DEFAULT scope, not "whatever ask ran last"
    assert e.recall_trace() is None
