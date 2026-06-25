"""Offline tests for affect-modulated salience (Phase 3).

The make-or-break audit is age-invariance: the salience retrieval boost must lift a memory by its
STATIC salience, never by its age. Two memories with equal salience get an identical contribution
regardless of valid_at; salience (not age) is what flips ranking.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from eidetic import fsrs
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import Retriever
from eidetic.salience import affect_salience, emphasis_score
from eidetic.store import RecordStore


class _FakeIndex:
    def __init__(self, dense):
        self.dense = dense

    def __len__(self):
        return max(len(self.dense), 1)

    def search(self, q, k, allowed_ids=None, ef=None):
        items = [(m, s) for m, s in self.dense if allowed_ids is None or m in allowed_ids]
        return sorted(items, key=lambda x: -x[1])[:k]

    def get_vectors(self, ids):
        return {}


def _store_recs(path, recs):
    store = RecordStore(path)
    scope = Scope(namespace="t")
    for mid, sal, valid_at in recs:
        store.upsert_record(MemoryRecord(memory_id=mid, content_hash=mid, text=f"memory {mid}",
                                         scope=scope, valid_at=valid_at, salience=sal))
    return store, scope


# ---- pure affect functions -----------------------------------------------------------------
def test_emphasis_score_detects_user_cues():
    assert emphasis_score("Please REMEMBER THIS, it is important!") > 0.5
    assert emphasis_score("the weather was mild today") == 0.0


def test_affect_salience_is_monotonic_and_age_free_and_centered():
    lo = affect_salience(0.1, 0.1, 0.1, 0.0, 0.0)
    hi = affect_salience(0.9, 0.9, 0.9, 1.0, 0.0)
    assert hi > lo                                          # rises with arousal/importance/etc.
    assert abs(affect_salience(0.5, 0.5, 0.5, 0.5, 0.0) - 0.5) < 1e-9   # neutral centered at 0.5


# ---- FSRS S0 coupling (replay/scheduling only; default unchanged) ---------------------------
def test_fsrs_s0_coupling_only_when_salience_given():
    base = fsrs.init_state(0.5, 0.5)
    coupled = fsrs.init_state(0.5, 0.5, salience=1.0, gamma=0.5)
    assert coupled.stability > base.stability               # salient -> starts more stable
    assert fsrs.init_state(0.5, 0.5, salience=None).stability == base.stability   # baseline intact


# ---- the age-invariance audit (make-or-break) ----------------------------------------------
def test_equal_salience_gives_equal_boost_regardless_of_age(fresh_settings, tmp_path):
    # Isolate the salience BOOST term: fused(on) - fused(off). For two memories with EQUAL salience
    # the boost delta must be identical despite a ~billion-second age gap -> the boost is age-free.
    old_t, new_t = 1.0, 1_000_000_000.0
    store, scope = _store_recs(tmp_path / "db.sqlite",
                               [("m_old", 0.9, old_t), ("m_new", 0.9, new_t)])
    idx = _FakeIndex(dense=[("m_old", 0.8), ("m_new", 0.8)])

    def fused(affect_on):
        s = replace(fresh_settings, rerank_enabled=False, persistent_bm25_enabled=False,
                    affect_salience_enabled=affect_on, lambda_salience=0.3)
        r = Retriever(store, idx, KnowledgeGraph(store), object(), object(), s)
        return {c.record.memory_id: c.fused_score for c in
                r.retrieve("memory", scope=scope, qvec=np.array([1.0, 0.0], np.float32),
                           use_recency=False)}

    off, on = fused(False), fused(True)
    boost_old = on["m_old"] - off["m_old"]
    boost_new = on["m_new"] - off["m_new"]
    assert boost_old > 0.0                                  # the boost is actually applied
    assert abs(boost_old - boost_new) < 1e-12              # identical boost; the age gap is irrelevant


def test_salience_flips_ranking_and_it_is_salience_not_age(fresh_settings, tmp_path):
    old_t, new_t = 1.0, 1_000_000_000.0

    def run(affect_on, sub):
        store, scope = _store_recs(tmp_path / sub, [("m_old", 0.95, old_t), ("m_new", 0.05, new_t)])
        idx = _FakeIndex(dense=[("m_new", 0.85), ("m_old", 0.80)])   # dense favors the NEW one
        s = replace(fresh_settings, rerank_enabled=False, persistent_bm25_enabled=False,
                    affect_salience_enabled=affect_on, lambda_salience=0.5)
        r = Retriever(store, idx, KnowledgeGraph(store), object(), object(), s)
        return [c.record.memory_id for c in r.retrieve("memory", scope=scope,
                                                       qvec=np.array([1.0, 0.0], np.float32),
                                                       use_recency=False)]

    assert run(False, "off")[0] == "m_new"     # baseline: stronger dense content wins
    assert run(True, "on")[0] == "m_old"       # salience surfaces the salient (and OLDER) memory
