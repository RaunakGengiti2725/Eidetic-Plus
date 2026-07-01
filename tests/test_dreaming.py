"""Offline tests for the Dreaming Engine (token-free; no key). Enforces the cardinal rules:
additive-only (lossless store never mutated), inferred-namespace separation, near-linear
(no O(N^2)) replay, the inferred-edge gate, multi-resolution retrieval, and prefetch hits."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from eidetic import fsrs
from eidetic.dreaming import gate as gatemod
from eidetic.dreaming import multires, prefetch
from eidetic.models import Edge, MemoryRecord, Modality, Scope


def _seed(engine, n=24, namespace="dream"):
    """Populate store + index + observed graph synthetically (NO model calls)."""
    ns = Scope(namespace=namespace)
    rng = np.random.default_rng(0)
    D = engine.settings.embed_dim
    topics = {t: rng.normal(size=D).astype("float32") for t in ("c", "j", "f")}
    relations = [("lives_in", "c"), ("capital_of", "c"), ("works_at", "j"), ("likes", "f")]
    raw_hash, _ = engine.substrate.put(b"sacred lossless record")
    for i in range(n):
        rel, topic = relations[i % len(relations)]
        s_, o_ = f"S{i}", f"O{i % 7}"
        rec = MemoryRecord(content_hash=raw_hash, text=f"{s_} {rel} {o_}", entities=[s_, o_],
                           modality=Modality.TEXT, scope=ns, fsrs=fsrs.init_state(0.6, 0.6))
        engine.store.upsert_record(rec)
        engine.index.add(rec.memory_id, topics[topic] + 0.05 * rng.normal(size=D).astype("float32"))
        engine.graph.add_fact(s_, rel, o_, fact=rec.text, source_memory_id=rec.memory_id, scope=ns)
    engine.index.save()
    return ns, raw_hash


# ---- CARDINAL 1: additive-only; lossless store never mutated ----
def test_dream_never_mutates_lossless_store(engine):
    ns, raw_hash = _seed(engine)
    before = hashlib.sha256(engine.substrate.get(raw_hash)).hexdigest()
    texts_before = {r.memory_id: (r.content_hash, r.text) for r in engine.store.all_records(ns)}

    engine.dream(scope=ns)

    assert hashlib.sha256(engine.substrate.get(raw_hash)).hexdigest() == before  # bytes identical
    # observed records' immutable identity (hash + raw text) unchanged; no centroid written in.
    after = {r.memory_id: (r.content_hash, r.text) for r in engine.store.all_records(ns)}
    assert after == texts_before


# ---- CARDINAL 2: inferred items live in a SEPARATE layer, flagged + provenance-tagged ----
def test_inferred_namespace_separation(engine):
    ns, _ = _seed(engine)
    engine.store.add_edge(Edge(src="X", dst="Y", relation="related", scope=ns,
                               inferred=True, confidence=0.9, provenance="transe"))
    observed = engine.store.all_edges(ns)                      # default EXCLUDES inferred
    with_inferred = engine.store.all_edges(ns, include_inferred=True)
    assert all(not e.inferred for e in observed)
    inferred = [e for e in with_inferred if e.inferred]
    assert len(inferred) >= 1
    assert inferred[0].provenance and inferred[0].confidence < 1.0 + 1e-9
    # The observed knowledge graph must not contain inferred edges by default.
    g = engine.graph.build_nx(scope=ns)
    assert not g.has_edge("x", "y")


# ---- CARDINAL 3: near-linear; replay does NOT rebuild the graph per record ----
def test_replay_is_near_linear_not_quadratic(engine, monkeypatch):
    ns, _ = _seed(engine, n=40)
    calls = {"n": 0}
    real = engine.graph.node_features

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(engine.graph, "node_features", counting)
    engine.dream_replay(scope=ns)
    # Graph features computed a CONSTANT number of times, not once-per-record (the O(N^2) hang).
    assert calls["n"] <= 2, f"node_features called {calls['n']}x -> quadratic risk"


def test_replay_tolerates_entities_missing_from_feature_map(engine, monkeypatch):
    import warnings

    ns, _ = _seed(engine, n=6)
    monkeypatch.setattr(engine.graph, "node_features", lambda *a, **k: {"other": {"ppr": 1.0}})
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        res = engine.dream_replay(scope=ns)
    assert res["memories"] == 6
    assert seen == []


# ---- the token-free gate ----
def test_inferred_gate_threshold():
    assert gatemod.gate(0.9, 0.9, 0.7).passed
    assert not gatemod.gate(0.2, 0.2, 0.7).passed
    # optional LLM check can veto a high-confidence item
    assert not gatemod.gate(0.95, 0.95, 0.7, llm_nli=lambda: False).passed
    assert gatemod.gate(0.95, 0.95, 0.7, llm_nli=lambda: True).passed
    assert 0.0 <= gatemod.confidence_from(0.8, 0.6) <= 1.0


# ---- multi-resolution retrieval ----
def test_multiresolution_tree_and_search():
    rng = np.random.default_rng(1)
    # three tight clusters in embedding space
    items = []
    for c in range(3):
        center = rng.normal(size=32).astype("float32")
        for i in range(12):
            items.append((f"m{c}_{i}", center + 0.02 * rng.normal(size=32).astype("float32")))
    gists = multires.build_tree(items, namespace="t", levels=3, min_cluster=4, max_k=8)
    assert gists, "no gist nodes built"
    assert all(g.vector and g.member_ids for g in gists)        # centroids + members
    assert max(g.level for g in gists) >= 1
    # query near cluster 0's members retrieves a gist that covers them
    q = np.array([v for i, v in items if i.startswith("m0_")]).mean(axis=0)
    hits = multires.search(q, gists, k=3)
    assert hits and hits[0][1] > 0.5


# ---- predictive pre-fetch cache hit ----
def test_prefetch_cache_hit_and_miss():
    pc = prefetch.PrefetchCache(threshold=0.9)
    v = np.array([1.0, 0.0, 0.0], dtype="float32")
    pc.add(v, ["pre-assembled context"])
    assert pc.get(v) == ["pre-assembled context"]               # exact -> hit
    assert pc.get(np.array([0.0, 1.0, 0.0], dtype="float32")) is None  # orthogonal -> miss
    assert 0.0 <= pc.hit_rate() <= 1.0
    labels, centroids = prefetch.PrefetchCache.cluster_queries(
        np.random.default_rng(2).normal(size=(20, 8)).astype("float32"), max_clusters=4)
    assert centroids.shape[0] <= 4 and labels.shape[0] == 20
