"""Offline tests for the Layer-3 dreaming gaps, the optimizer daemon, and engine feedback."""
from __future__ import annotations

import numpy as np

from eidetic.dreaming.fademem import half_life, lambda_from_importance, reinforce, retention
from eidetic.optim.cache_policies import ARCCache, LFUCache, LRUCache
from eidetic.optim.markov import MarkovPrefetcher


# ---- FadeMem -----------------------------------------------------------------
def test_fademem_decay_and_importance():
    # strength decays with time
    assert retention(1.0, lam=0.2, beta=1.0, t=5.0) < retention(1.0, lam=0.2, beta=1.0, t=1.0)
    # higher importance -> smaller lambda -> slower decay
    lam_lo = lambda_from_importance(0.1)
    lam_hi = lambda_from_importance(0.9)
    assert lam_hi < lam_lo
    # half-life closed form: retention at t_half is ~0.5
    lam, beta = 0.2, 1.0
    th = half_life(lam, beta)
    assert abs(retention(1.0, lam, beta, th) - 0.5) < 1e-9


def test_fademem_reinforce_saturates():
    v = 0.2
    v1 = reinforce(v, dv=0.5, n=0, N=10)
    v2 = reinforce(v1, dv=0.5, n=5, N=10)
    assert v < v1 < v2 <= 1.0
    # diminishing returns: the same dv adds less when the access count n is higher
    a = reinforce(0.2, dv=0.5, n=0, N=10) - 0.2
    b = reinforce(0.2, dv=0.5, n=20, N=10) - 0.2
    assert a > b


# ---- cache policies ----------------------------------------------------------
def test_lru_evicts_oldest():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") is None and c.get("b") == 2 and c.get("c") == 3


def test_lfu_evicts_least_frequent():
    c = LFUCache(2)
    c.put("a", 1); c.get("a"); c.get("a")     # a is frequent
    c.put("b", 2)                              # b seen once
    c.put("c", 3)                              # evicts the least frequent (b)
    assert c.get("a") == 1 and c.get("b") is None and c.get("c") == 3


def test_arc_is_scan_resistant():
    c = ARCCache(2)
    c.put("a", 1); c.get("a")                  # a becomes frequent (T2)
    for k in ("b", "c", "d"):                  # a scan of one-shot keys
        c.put(k, 0)
    assert c.get("a") == 1                      # the frequent item survives the scan
    assert c.get("b") is None                   # one-shots are evicted


# ---- Markov prefetcher -------------------------------------------------------
def test_markov_predicts_next():
    m = MarkovPrefetcher()
    for _ in range(3):
        m.observe_sequence("A", "B")
    m.observe_sequence("A", "C")
    assert abs(m.transition_prob("A", "B") - 0.75) < 1e-9
    assert m.predict("A")[0][0] == "B"


def test_markov_stream_observe():
    m = MarkovPrefetcher()
    for c in ["A", "B", "A", "B", "A", "C"]:
        m.observe(c)
    assert m.predict("A")[0][0] == "B"          # A->B seen twice, A->C once


# ---- daemon ------------------------------------------------------------------
def test_daemon_swap_config_refuses_rebuild_knobs(tmp_path):
    import json
    import os

    from eidetic.config import get_settings
    from eidetic.optim.daemon import OptimizerDaemon

    cfg = tmp_path / "best_config.json"
    cfg.write_text(json.dumps({"best_env": {"FUSION_METHOD": "dbsf", "HNSW_M": "64"}}))
    os.environ.pop("FUSION_METHOD", None)
    try:
        res = OptimizerDaemon.swap_config(cfg, apply=True)
        assert res["applied"].get("FUSION_METHOD") == "dbsf"
        assert "HNSW_M" in res["refused_rebuild_knobs"]   # rebuild knob never applied live
        assert os.environ.get("FUSION_METHOD") == "dbsf"
        assert get_settings().fusion_method == "dbsf"
    finally:
        os.environ.pop("FUSION_METHOD", None)
        get_settings.cache_clear()


def test_daemon_offline_command_is_dev_locked():
    from eidetic.optim.daemon import OptimizerDaemon
    cmd = OptimizerDaemon.offline_sweep_command(subset=40, trials=12)
    assert "--split dev" in cmd and "--sampler tpe" in cmd


# ---- engine feedback + idle learning (no API) --------------------------------
def test_engine_emit_feedback_and_learn(engine):
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    def cand(mid, dense, bm25, graph):
        rec = MemoryRecord(memory_id=mid, content_hash=mid, text=mid, valid_at=1.0)
        return RetrievalCandidate(record=rec, dense_score=dense, bm25_score=bm25,
                                  graph_score=graph, fused_score=dense)

    scope = Scope(namespace="user-9")
    cands = [cand("m0", 0.9, 0.2, 0.0), cand("m1", 0.4, 0.1, 0.0)]
    for _ in range(20):
        engine._emit_feedback(scope, "where is alice", np.array([1.0, 0.0], np.float32),
                              cands, confirmed=["m0"])
    # a benchmark namespace must NOT feed the learner
    engine._emit_feedback(Scope(namespace="eidetic-plus-locomo-g0-r0"), "bq",
                          np.array([1.0, 0.0], np.float32),
                          [cand("g", 0.0, 0.0, 0.9)], confirmed=["g"])

    weights = engine.learn_fusion_weights()
    assert weights and weights["dense"] == max(weights.values())   # dev signal favored dense
    assert (engine.settings.index_dir / "fusion_weights.json").exists()
