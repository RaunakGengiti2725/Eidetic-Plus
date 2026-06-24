"""Offline unit tests for the Layer-2 hot-path optimizers (no key)."""
from __future__ import annotations

import numpy as np

from eidetic.optim.adaptive_k import adaptive_k_cut, largest_gap_k
from eidetic.optim.conformal import (calibrate_qhat_from_pairs, coverage_cutoff,
                                     select_by_conformal, split_conformal_qhat)
from eidetic.optim.fusion import combine_borda, combine_scores
from eidetic.optim.gating import score_entropy, score_margin, should_skip_rerank
from eidetic.optim.mmr import mmr_order


# ---- adaptive-k --------------------------------------------------------------
def test_largest_gap_cuts_at_the_cliff():
    # sharp cliff after the top 3 (0.84 -> 0.30 is the biggest gap)
    scores = [0.90, 0.85, 0.84, 0.30, 0.28]
    assert largest_gap_k(scores, min_k=1, max_k=5) == 3


def test_largest_gap_respects_bounds_and_edge_cases():
    assert largest_gap_k([], min_k=1) == 0
    assert largest_gap_k([0.5], min_k=1) == 1
    # min_k floors the cut even if the biggest gap is at the very top.
    assert largest_gap_k([0.9, 0.1, 0.08, 0.07], min_k=2, max_k=4) >= 2
    # max_k caps it.
    assert largest_gap_k([0.9, 0.89, 0.88, 0.2], min_k=1, max_k=2) <= 2


def test_largest_gap_can_reach_max_k_via_a_gap():
    # regression for an off-by-one: the largest gap (0.9->0.1) keeps k=2, which max_k=2
    # permits. The buggy bound never examined the cut at i=hi-1 and returned 1.
    assert largest_gap_k([1.0, 0.9, 0.1], min_k=1, max_k=2) == 2
    assert [c[0] for c in adaptive_k_cut(
        [("a", 1.0), ("b", 0.9), ("c", 0.1)], score_fn=lambda c: c[1],
        min_k=1, max_k=2)] == ["a", "b"]


def test_adaptive_k_cut_preserves_order_and_drops_tail():
    cands = [("a", 0.9), ("b", 0.88), ("c", 0.2), ("d", 0.18)]
    kept = adaptive_k_cut(cands, score_fn=lambda c: c[1], min_k=1, max_k=4)
    assert [c[0] for c in kept] == ["a", "b"]


# ---- split-conformal ---------------------------------------------------------
def test_split_conformal_qhat_quantile():
    nonconf = [0.1, 0.2, 0.3, 0.4, 0.5]            # n=5
    # alpha=0.2 -> rank=ceil(6*0.8)=5 -> qhat=0.5
    assert split_conformal_qhat(nonconf, alpha=0.2) == 0.5
    # alpha=0.5 -> rank=ceil(6*0.5)=3 -> qhat=0.3
    assert split_conformal_qhat(nonconf, alpha=0.5) == 0.3
    # alpha too small for n -> +inf (cannot certify -> include all)
    assert split_conformal_qhat(nonconf, alpha=0.01) == float("inf")
    assert split_conformal_qhat([], alpha=0.1) == float("inf")


def test_conformal_selection_and_min_keep():
    cands = [("a", 0.95), ("b", 0.72), ("c", 0.40)]
    # qhat=0.3 -> cutoff sim 0.7 -> keep a,b
    kept = select_by_conformal(cands, sim_fn=lambda c: c[1], qhat=0.3, min_keep=1)
    assert [c[0] for c in kept] == ["a", "b"]
    # an extreme qhat would drop everything; min_keep guarantees at least one.
    kept2 = select_by_conformal(cands, sim_fn=lambda c: c[1], qhat=0.9, min_keep=1)
    assert len(kept2) >= 1


def test_calibrate_qhat_from_pairs():
    pairs = [{"answer_sim": s} for s in [0.9, 0.8, 0.7, 0.6, 0.5]]
    res = calibrate_qhat_from_pairs(pairs, alpha=0.5)
    assert res["ok"] and abs(res["qhat"] - 0.3) < 1e-9
    assert abs(res["sim_cutoff"] - coverage_cutoff(0.3)) < 1e-9


# ---- fusion variants ---------------------------------------------------------
def test_zscore_single_channel_preserves_order():
    fused = combine_scores([{"a": 0.9, "b": 0.5, "c": 0.1}], weights=[1.0], method="zscore")
    order = [mid for mid, _ in sorted(fused.items(), key=lambda x: -x[1])]
    assert order == ["a", "b", "c"]


def test_fusion_handles_scale_mismatch():
    # dense in [-1,1], BM25 unbounded; both should still contribute, not be dominated.
    dense = {"a": 0.9, "b": 0.4, "c": 0.1}
    bm25 = {"a": 2.0, "b": 80.0, "c": 5.0}
    for method in ("zscore", "minmax", "dbsf"):
        fused = combine_scores([dense, bm25], weights=[1.0, 1.0], method=method)
        # b is BM25-dominant; normalization lets it rise above c despite raw-scale gaps.
        assert fused["b"] > fused["c"]


def test_zero_variance_channel_contributes_nothing():
    flat = {"a": 0.5, "b": 0.5, "c": 0.5}          # no discriminative signal
    fused = combine_scores([flat], weights=[1.0], method="minmax")
    assert all(abs(v) < 1e-12 for v in fused.values())


def test_borda_counts_rank_positions():
    fused = combine_borda([["a", "b", "c"], ["b", "a", "c"]], weights=[1.0, 1.0])
    assert fused["a"] == 5 and fused["b"] == 5     # symmetric top two
    assert fused["c"] == 2                          # last in both


def test_normalized_fusion_beats_raw_sum_on_scale_mismatch():
    """Existence proof (not a benchmark guarantee): when one channel's score scale dwarfs the
    other, a naive raw-score sum lets the big-scale channel dominate and buries the doc both
    channels agree on. z-score/DBSF give each channel equal voice and surface the consensus."""
    dense = {"d_spike": 0.9, "gold": 0.7, "b_spike": 0.2, "x": 0.1}   # cosine ~[0,1]
    bm25 = {"b_spike": 100.0, "gold": 60.0, "d_spike": 5.0, "x": 1.0}  # unbounded
    raw = {m: dense.get(m, 0.0) + bm25.get(m, 0.0) for m in set(dense) | set(bm25)}
    assert max(raw, key=raw.get) == "b_spike"       # raw sum: BM25 scale dominates
    for method in ("zscore", "dbsf"):
        fused = combine_scores([dense, bm25], [1.0, 1.0], method)
        assert max(fused, key=fused.get) == "gold", method  # normalized: consensus wins


# ---- MMR ---------------------------------------------------------------------
def test_mmr_demotes_a_near_duplicate():
    rel = [1.0, 0.9, 0.8]
    vecs = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    order = mmr_order(rel, vecs, lam=0.6)
    # 0 first (most relevant); then the orthogonal 2 beats the duplicate-of-0 (index 1).
    assert order[0] == 0
    assert order.index(2) < order.index(1)


def test_mmr_lambda_zero_is_pure_relevance():
    rel = [0.3, 0.9, 0.6]
    vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    assert mmr_order(rel, vecs, lam=0.0) == [1, 2, 0]


# ---- gating ------------------------------------------------------------------
def test_margin_and_entropy_signals():
    assert score_margin([0.9, 0.2, 0.1]) > score_margin([0.5, 0.49, 0.48])
    peaked = score_entropy([10.0, 0.0, 0.0])
    uniform = score_entropy([1.0, 1.0, 1.0])
    assert peaked < 0.2 and uniform > 0.99


def test_skip_rerank_gate():
    assert should_skip_rerank([0.9, 0.1], margin_threshold=0.5) is True
    assert should_skip_rerank([0.5, 0.49], margin_threshold=0.5) is False
    assert should_skip_rerank([0.9, 0.1], margin_threshold=0.0) is False   # disabled


# ---- end-to-end wiring through retrieve() (no API; rerank off) ---------------
def _wired_retriever(settings):
    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    scope = Scope(namespace="wire")
    store = RecordStore(settings.sqlite_path)
    texts = {
        "m0": "alice lives in paris", "m1": "alice works at globex",
        "m2": "alice enjoys green tea", "m3": "bob plays the guitar",
        "m4": "carol studies law", "m5": "dave likes hiking", "m6": "erin paints",
    }
    for mid, t in texts.items():
        store.upsert_record(MemoryRecord(memory_id=mid, content_hash=mid, text=t,
                                         scope=scope, valid_at=1.0))
    dense_scores = {"m0": 0.95, "m1": 0.93, "m2": 0.90, "m3": 0.20,
                    "m4": 0.18, "m5": 0.17, "m6": 0.15}
    vecs = {mid: np.eye(7, dtype=np.float32)[i] for i, mid in enumerate(texts)}

    class FakeIndex:
        def __len__(self):
            return len(texts)

        def search(self, _q, k, allowed_ids=None):
            items = [(m, sc) for m, sc in dense_scores.items()
                     if allowed_ids is None or m in allowed_ids]
            return sorted(items, key=lambda x: -x[1])[:k]

        def get_vectors(self, ids):
            return {m: vecs[m] for m in ids if m in vecs}

    return Retriever(store, FakeIndex(), KnowledgeGraph(store), object(), object(), settings), scope


def test_adaptive_k_fires_through_retrieve(fresh_settings):
    from dataclasses import replace
    qvec = np.ones(4, dtype=np.float32)

    base = replace(fresh_settings, rerank_enabled=False, adaptive_k_enabled=False)
    r0, scope = _wired_retriever(base)
    n_off = len(r0.retrieve("where does alice live", scope=scope, qvec=qvec, use_recency=False))

    on = replace(fresh_settings, rerank_enabled=False, adaptive_k_enabled=True, adaptive_k_min=2)
    r1, scope = _wired_retriever(on)
    out = r1.retrieve("where does alice live", scope=scope, qvec=qvec, use_recency=False)
    ids = [c.record.memory_id for c in out]
    assert 2 <= len(out) <= n_off          # adaptive-k can only trim, never pad
    assert len(ids) == len(set(ids))       # deduped


def test_all_fusion_methods_wire_through_retrieve(fresh_settings):
    from dataclasses import replace
    qvec = np.ones(4, dtype=np.float32)
    for method in ("rrf", "zscore", "minmax", "dbsf", "borda"):
        s = replace(fresh_settings, rerank_enabled=False, fusion_method=method)
        r, scope = _wired_retriever(s)
        out = r.retrieve("alice", scope=scope, qvec=qvec, use_recency=False)
        assert out, f"fusion method {method} returned nothing"
        top_ids = {c.record.memory_id for c in out[:3]}
        assert top_ids & {"m0", "m1", "m2"}, f"{method} lost the alice records"


def test_mmr_wiring_is_safe_and_changes_order(fresh_settings):
    from dataclasses import replace
    qvec = np.ones(4, dtype=np.float32)
    s = replace(fresh_settings, rerank_enabled=False, mmr_enabled=True, mmr_lambda=0.7)
    r, scope = _wired_retriever(s)
    out = r.retrieve("alice", scope=scope, qvec=qvec, use_recency=False)
    assert out and len({c.record.memory_id for c in out}) == len(out)
