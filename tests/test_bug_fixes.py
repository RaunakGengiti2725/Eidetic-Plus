"""Regression tests for bugs found by the codebase bug scan and patched. Each test fails on the
pre-fix code and passes after the fix. All offline (no key)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np


# ---- Bug 1: case-insensitive contradiction detection ---------------------------------------
def test_contradiction_detects_differently_cased_subject(tmp_path):
    from eidetic.graph import KnowledgeGraph
    from eidetic.store import RecordStore

    store = RecordStore(tmp_path / "db.sqlite")
    g = KnowledgeGraph(store)
    _, inv0 = g.add_fact("Alice", "lives_in", "Paris", valid_at=100.0)
    assert inv0 == []
    # the same subject in a different case must still close the old single-valued fact
    _, invalidated = g.add_fact("alice", "lives_in", "London", valid_at=200.0)
    assert len(invalidated) == 1                                   # was 0 before the fix
    active = store.active_edges_touching_many({"alice"}, 300.0)
    assert {e.dst for e in active} == {"London"}                  # Paris no longer active


# ---- Bug 3: k-means++ does not crash when k exceeds distinct directions ---------------------
def test_kmeans_survives_more_clusters_than_distinct_points():
    from eidetic.dreaming.multires import _kmeans

    # two tight groups of bit-identical unit vectors; ask for more clusters than directions
    X = np.zeros((12, 4), dtype=np.float32)
    X[:6] = np.array([1.0, 0, 0, 0])
    X[6:] = np.array([0, 1.0, 0, 0])
    labels, C = _kmeans(X, k=5)                                    # raised ValueError before fix
    assert C.shape[0] <= 5 and labels.shape == (12,)              # k collapsed to real centers
    assert set(labels.tolist()) <= set(range(C.shape[0]))


# ---- Bug 4: anomaly scoring works with lowercased entity-centroid keys + raw-case edges -----
def test_edge_anomaly_case_insensitive_entity_lookup():
    from eidetic.dreaming.anomaly import edge_anomaly_scores
    from eidetic.models import Edge

    # centroids keyed lowercase (as _entity_centroids produces); edges in raw case
    cents = {"alice": np.array([1.0, 0.0]), "acme": np.array([0.99, 0.1]),
             "paris": np.array([0.98, 0.2]), "far": np.array([-1.0, 0.0])}
    edges = [Edge(src="Alice", dst="Acme", relation="r"),
             Edge(src="Alice", dst="Paris", relation="r"),
             Edge(src="Alice", dst="Far", relation="r")]            # incoherent -> most anomalous
    scores = edge_anomaly_scores(edges, cents, k=2)
    assert len(set(np.round(scores, 6))) > 1            # NOT all identical (was uniform before fix)
    # the coherent Alice->Acme edge is the LEAST anomalous; the incoherent Alice->Far is more so
    assert int(np.argmin(scores)) == 0
    assert scores[2] > scores[0]


# ---- Bug 5: boolean config flags parse capitalized word forms --------------------------------
def test_capitalized_boolean_env_flags(monkeypatch):
    from eidetic.config import get_settings
    monkeypatch.setenv("RERANK_ENABLED", "False")
    monkeypatch.setenv("MMR_ENABLED", "True")
    monkeypatch.setenv("SEMANTIC_CACHE", "No")
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.rerank_enabled is False           # was True (inverted) before the fix
        assert s.mmr_enabled is True               # was False before the fix
        assert s.semantic_cache_enabled is False   # was True before the fix
    finally:
        get_settings.cache_clear()


# ---- Bug 6: the exact-hash cache is bounded -------------------------------------------------
def test_semantic_cache_exact_is_bounded():
    from eidetic.semantic_cache import SemanticCache

    cache = SemanticCache(max_entries=10)
    for i in range(50):
        cache.put("scope", f"query number {i}", None, f"answer-{i}")
    assert len(cache._exact) <= 10                 # grew unbounded before the fix
    # most-recent entries survive; oldest were evicted
    assert cache.get("scope", "query number 49", None) == "answer-49"
    assert cache.get("scope", "query number 0", None) is None


# ---- Bug 8: is_benchmark_namespace does not over-match the generic 'beam' token --------------
def test_benchmark_namespace_does_not_over_match():
    from eidetic.feedback import is_benchmark_namespace

    # real harness namespaces still detected
    assert is_benchmark_namespace("eidetic-plus-locomo-g0-r0")
    assert is_benchmark_namespace("beam-g1-r0")
    assert is_benchmark_namespace("beam")                          # exact dataset name
    # ordinary user namespaces containing 'beam' are NOT benchmark (were misflagged before)
    assert not is_benchmark_namespace("team-beam-knowledge")
    assert not is_benchmark_namespace("beam-search-notes")


# ---- Bug 2: adaptive-k cut keeps the high-score set even with MMR reordering -----------------
def test_adaptive_k_after_mmr_keeps_high_score_items(fresh_settings):
    from tests.test_layer2_optim import _wired_retriever

    # _wired_retriever has dense scores with a sharp cliff after the alice cluster (m0,m1,m2).
    # With MMR + adaptive-k both on, depth-select must run on the score order BEFORE MMR
    # diversifies, so the kept set is the high-score cluster, not a low-score diverse item that
    # MMR reordered into an early position.
    s = replace(fresh_settings, rerank_enabled=False, mmr_enabled=True, mmr_lambda=0.7,
                adaptive_k_enabled=True, adaptive_k_min=2)
    r, scope = _wired_retriever(s)
    out = r.retrieve("alice", scope=scope, qvec=np.ones(4, dtype=np.float32), use_recency=False)
    ids = {c.record.memory_id for c in out}
    assert 2 <= len(out) <= 4
    assert ids <= {"m0", "m1", "m2"}        # only the high-score cluster survived the cut
