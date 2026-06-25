"""Offline synthetic improvement gates (Connected Brain Loop, Phase 8).

Each test encodes the plan's improvement contract: a connection must both RECOVER a dense
miss (measurable lift) AND be EXPLAINABLE (the RecallTrace / proof shows which channel won).
A connection that surfaces a memory invisibly to the brain should not ship; these gates make
that visibility a test assertion. All offline (FakeIndex, no key).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np

from eidetic.brain import QualityGate
from eidetic.events import EventRecord
from eidetic.graph import KnowledgeGraph
from eidetic.models import Answer, Citation, DerivedRecord, MemoryRecord, Scope
from eidetic.proofs import prove_answer
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _FakeIndex:
    def __init__(self, dense, vecs=None):
        self.dense, self.vecs = dense, vecs or {}

    def __len__(self):
        return max(len(self.dense), len(self.vecs), 1)

    def search(self, q, k, allowed_ids=None, ef=None):
        items = [(m, s) for m, s in self.dense if allowed_ids is None or m in allowed_ids]
        return sorted(items, key=lambda x: -x[1])[:k]

    def get_vectors(self, ids):
        return {m: self.vecs[m] for m in ids if m in self.vecs}


def _store(tmp_path, ns="proj", n=5):
    store = RecordStore(tmp_path / "db.sqlite")
    scope = Scope(namespace=ns)
    for i in range(n):
        store.upsert_record(MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
                                         text=f"memory {i}", scope=scope, valid_at=1.0))
    return store, scope


def _retriever(store, settings, index):
    return Retriever(store, index, KnowledgeGraph(store), object(), object(), settings)


# ---- event channel recovers a dense miss AND the trace explains it -------------------------
def test_event_channel_recovers_miss_and_is_traced(fresh_settings, tmp_path):
    store, scope = _store(tmp_path)
    may = datetime(2023, 5, 15).timestamp()
    store.add_event(EventRecord(subject="exercise", verb="did", object="run", start=may, end=may,
                                source_memory_id="m3", namespace=scope.namespace, valid_at=may))
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85)])           # m3 is NOT a dense hit
    s = replace(fresh_settings, rerank_enabled=False, event_ranking_enabled=True,
                rrf_w_event=2.0, recall_trace_enabled=True)
    r = _retriever(store, s, idx)
    q = "How many times did Exercise happen in May 2023?"
    out = r.retrieve(q, scope=scope, qvec=np.array([1.0, 0.0], np.float32), use_recency=False)
    ids = {c.record.memory_id for c in out}
    assert "m3" in ids                                            # recovered the dense miss
    assert "event" in r.last_trace.paths_for("m3")               # and the brain knows why


# ---- gist channel recovers a dense miss AND proof shows the gist provenance -----------------
def test_gist_channel_recovers_miss_and_proof_shows_provenance(fresh_settings, tmp_path):
    store, scope = _store(tmp_path)
    qvec = np.array([1.0, 0.0], np.float32)
    store.add_derived(DerivedRecord(cid="g", kind="gist", namespace=scope.namespace,
                                    member_ids=["m4"], vector=[1.0, 0.0]))
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)],
                     vecs={f"m{i}": qvec for i in range(5)})
    s = replace(fresh_settings, rerank_enabled=False, gist_channel_enabled=True,
                rrf_w_gist=2.0, recall_trace_enabled=True)
    r = _retriever(store, s, idx)
    out = r.retrieve("q", scope=scope, qvec=qvec, use_recency=False)
    assert "m4" in {c.record.memory_id for c in out}             # gist surfaced the dense miss
    # the proof tree, given the matching trace, attributes m4 to the dream gist 'g'.
    ans = Answer(question="q", answer="a",
                 citations=[Citation(memory_id="m4", content_hash="h4", raw_uri="", source="user",
                                     valid_at=1.0, snippet="memory 4")])
    proof = prove_answer(ans, r.last_trace)
    m4 = next(e for e in proof["evidence"] if e["memory_id"] == "m4")
    assert m4["via_gist"] == "g" and "gist" in m4["recall_paths"]


# ---- neutral-path improvement gate: off preserves baseline, on lifts recall ----------------
def test_channel_off_preserves_baseline_on_lifts_recall(fresh_settings, tmp_path):
    qvec = np.array([1.0, 0.0], np.float32)

    def run(gist_on, sub):
        store, scope = _store(tmp_path / sub)
        store.add_derived(DerivedRecord(cid="g", kind="gist", namespace=scope.namespace,
                                        member_ids=["m4"], vector=[1.0, 0.0]))
        idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)],
                         vecs={f"m{i}": qvec for i in range(5)})
        s = replace(fresh_settings, rerank_enabled=False, gist_channel_enabled=gist_on,
                    rrf_w_gist=2.0)
        out = _retriever(store, s, idx).retrieve("q", scope=scope, qvec=qvec, use_recency=False)
        return [c.record.memory_id for c in out]

    off_a, off_b = run(False, "a"), run(False, "b")
    on = run(True, "c")
    assert QualityGate.flag_off_preserves_baseline(off_a, off_b)  # deterministic baseline
    assert "m4" not in off_a                                      # dense miss when channel off
    assert "m4" in on                                            # connection lifts recall


# ---- co-activation channel recovers a multi-hop miss AND the trace explains it -------------
def test_coactivation_channel_recovers_multihop_miss(fresh_settings, tmp_path):
    store, scope = _store(tmp_path)
    # m0 and m4 were co-confirmed together in a past recall -> a CO_ACTIVATED link exists.
    KnowledgeGraph(store).link_memories(["m0", "m4"], scope=scope, valid_at=1.0)
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85)])          # m4 is NOT a dense hit
    s = replace(fresh_settings, rerank_enabled=False, coactivation_channel_enabled=True,
                rrf_w_coact=2.0, recall_trace_enabled=True)
    r = _retriever(store, s, idx)
    out = r.retrieve("q", scope=scope, qvec=np.array([1.0, 0.0], np.float32), use_recency=False)
    ids = {c.record.memory_id for c in out}
    assert "m4" in ids                                           # co-activation surfaced m4
    assert "coactivation" in r.last_trace.paths_for("m4")        # and the brain knows why
