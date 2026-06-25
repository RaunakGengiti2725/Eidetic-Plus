"""Connectivity tests (Connected Brain Loop, Phase 7): prove each subsystem actually FEEDS
another, with the non-negotiable invariants held (raw immutable, scope isolation, dev/test wall).
All offline.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from eidetic.brain import QualityGate
from eidetic.engine import Engine
from eidetic.graph import KnowledgeGraph
from eidetic.models import BrainEventType, DerivedRecord, MemoryRecord, Scope, now
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


# ---- feedback (dev) -> learned weights -> retrieval, with the integrity wall held ----------
def test_feedback_dev_buffer_feeds_learned_retrieval_weights(fresh_settings):
    e = Engine(fresh_settings)
    # The discriminating setup: only a FEW dev rows (favor dense) but MANY benchmark rows that
    # strongly favor graph. If the integrity wall leaked, 30 graph rows would swamp 5 dense rows
    # and the learner would pick graph. The wall (is_dev=0 for benchmark namespaces) must keep
    # every benchmark row out of sample(), so dense still wins.
    for _ in range(5):
        e.feedback.append("user-ns", "q", {"contrib_dense": 1.0, "contrib_bm25": 0.1,
                                            "contrib_graph": 0.0}, reward=1.0)
    for _ in range(30):
        e.feedback.append("eidetic-plus-locomo-g0-r0", "bq",
                          {"contrib_dense": 0.0, "contrib_bm25": 0.0, "contrib_graph": 1.0},
                          reward=1.0)
    assert e.feedback.count(dev_only=True) == 5           # benchmark rows excluded from learnable set
    w = e.learn_fusion_weights()
    assert w and w["dense"] == max(w.values())            # graph rows never reached the learner
    # a retriever with the learner ON reads exactly those learned weights (same DATA_DIR).
    e2 = Engine(replace(fresh_settings, fusion_learner_enabled=True))
    assert e2.retriever._content_weights()[0] == w["dense"]


# ---- dream gist (a dream output) -> gist channel -> recall ----------------------------------
def test_dream_gist_feeds_recall(fresh_settings, tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    scope = Scope(namespace="ns")
    for i in range(5):
        store.upsert_record(MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
                                         text=f"memory {i}", scope=scope, valid_at=1.0))
    store.add_derived(DerivedRecord(cid="g", kind="gist", namespace="ns",
                                    member_ids=["m4"], vector=[1.0, 0.0]))
    qvec = np.array([1.0, 0.0], np.float32)
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85)], vecs={f"m{i}": qvec for i in range(5)})
    s = replace(fresh_settings, rerank_enabled=False, gist_channel_enabled=True, rrf_w_gist=2.0)
    r = Retriever(store, idx, KnowledgeGraph(store), object(), object(), s)
    ids = {c.record.memory_id for c in r.retrieve("q", scope=scope, qvec=qvec, use_recency=False)}
    assert "m4" in ids


# ---- recall co-activation -> graph link -> later recall -------------------------------------
def test_recall_coactivation_feeds_later_recall(fresh_settings, tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    scope = Scope(namespace="ns")
    for i in range(5):
        store.upsert_record(MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
                                         text=f"memory {i}", scope=scope, valid_at=1.0))
    KnowledgeGraph(store).link_memories(["m0", "m4"], scope=scope, valid_at=1.0)
    idx = _FakeIndex(dense=[("m0", 0.9)])
    s = replace(fresh_settings, rerank_enabled=False, coactivation_channel_enabled=True,
                rrf_w_coact=2.0)
    r = Retriever(store, idx, KnowledgeGraph(store), object(), object(), s)
    ids = {c.record.memory_id for c in r.retrieve("q", scope=scope,
                                                  qvec=np.array([1.0, 0.0], np.float32),
                                                  use_recency=False)}
    assert "m4" in ids                                    # the past co-activation surfaced m4


# ---- brain events -> connection effectiveness report ---------------------------------------
def test_brain_events_feed_connection_effectiveness(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True))
    e._brain(BrainEventType.MEMORY_RECALLED, namespace="x", memory_ids=["m1"])
    e._brain(BrainEventType.ANSWER_VERIFIED, namespace="x")
    eff = e.connection_effectiveness()
    assert eff["events"].get("memory_recalled") == 1
    assert eff["total_events"] == 2


# ---- INVARIANT: a dream pass never mutates or deletes the raw store -------------------------
def test_dream_does_not_mutate_raw_store(engine):
    # Seed REAL vectors + edges so the dream actually does work (multires clusters, replay/infer
    # run). Otherwise dream exits early on an empty scope and the no-mutation check is vacuous.
    scope = Scope(namespace="ns")
    rng = np.random.default_rng(0)
    dim = engine.settings.embed_dim
    for i in range(6):
        rec = MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}", text=f"alice fact {i}",
                           scope=scope, valid_at=now(), entities=["alice", f"thing{i}"])
        engine.store.upsert_record(rec)
        engine.index.add(rec.memory_id, rng.standard_normal(dim).astype(np.float32))
    engine.graph.add_fact("alice", "likes", "tea", valid_at=now(), scope=scope)
    engine.graph.add_fact("alice", "visited", "paris", valid_at=now(), scope=scope)

    before = {r.memory_id: (r.text, r.content_hash) for r in engine.store.all_records(scope)}
    out = engine.dream(scope=scope)                        # additive derived layer only
    assert out["multires"]["members"] == 6                 # dream really processed all 6 records
    after = {r.memory_id: (r.text, r.content_hash) for r in engine.store.all_records(scope)}
    assert before == after                                # raw records byte-identical after real work
    assert QualityGate.no_raw_mutation([h for _, h in before.values()],
                                       [h for _, h in after.values()])


# ---- INVARIANT: scope isolation holds through the gist channel ------------------------------
def test_gist_channel_does_not_leak_across_scopes(fresh_settings, tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    a, b = Scope(namespace="A"), Scope(namespace="B")
    store.upsert_record(MemoryRecord(memory_id="a1", content_hash="ha", text="secret",
                                     scope=a, valid_at=1.0))
    store.upsert_record(MemoryRecord(memory_id="b1", content_hash="hb", text="public",
                                     scope=b, valid_at=1.0))
    store.add_derived(DerivedRecord(cid="g", kind="gist", namespace="A",
                                    member_ids=["a1"], vector=[1.0, 0.0]))
    qvec = np.array([1.0, 0.0], np.float32)
    idx = _FakeIndex(dense=[("b1", 0.9)], vecs={"a1": qvec, "b1": qvec})
    s = replace(fresh_settings, rerank_enabled=False, gist_channel_enabled=True, rrf_w_gist=2.0)
    r = Retriever(store, idx, KnowledgeGraph(store), object(), object(), s)
    ids = {c.record.memory_id for c in r.retrieve("q", scope=b, qvec=qvec, use_recency=False)}
    assert ids == {"b1"}                                  # A's gist never surfaces in B's scope


# ---- anti-island: the fastest-best-agent waves are wired to the ONE spine (not islands) -----
def test_anti_island_features_use_the_shared_spine(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True))
    # (d) writes go through the lock; (c) model calls go through the governor.
    assert hasattr(e, "_write_lock")
    assert getattr(e.client, "_governor", None) is not None
    # the spine carries an event type for each new operation.
    for name in ("REEMBED_DEFERRED", "SUPERSEDED", "RATE_LIMITED", "CACHE_HIT", "INTEGRITY_CHECKED"):
        assert hasattr(BrainEventType, name)
    # (a) emit + (b) schedule: deferred re-embed emits REEMBED_DEFERRED and is drained by the kernel.
    e._enqueue_reembed(["m1"])
    assert e.brain_log.by_type(BrainEventType.REEMBED_DEFERRED)
    assert "reembed_drain" in e.lifecycle.idle_tick()          # reachable from a LifecycleController loop
    # the integrity scan emits its own event (consumes the stream, not a private log).
    e.integrity_report()
    assert e.brain_log.by_type(BrainEventType.INTEGRITY_CHECKED)
