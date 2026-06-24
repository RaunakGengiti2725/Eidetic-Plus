"""Offline tests for Layer-3a/3b online learners + Layer-1c bandits (no key)."""
from __future__ import annotations

import numpy as np

from eidetic.optim.bandits import DiscountedUCB, LinUCB, ThompsonBeta, UCB1
from eidetic.optim.online_weights import (FTRL, FusionWeightLearner, eg_update,
                                          fusion_gradient, nonneg_normalize)
from eidetic.optim.rocchio import rocchio_expand, should_expand


# ---- Exponentiated Gradient --------------------------------------------------
def test_eg_stays_on_simplex_and_moves_toward_rewarded_channel():
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    # channel 0 keeps surfacing the correct memory (reciprocal rank 1.0), others don't.
    for _ in range(20):
        w = eg_update(w, fusion_gradient([1.0, 0.0, 0.0], reward=1.0), eta=0.2)
    assert abs(w.sum() - 1.0) < 1e-9          # simplex preserved
    assert w[0] > w[1] and w[0] > w[2]        # rewarded channel up-weighted


def test_eg_no_push_on_zero_reward():
    w0 = np.array([0.5, 0.5])
    w1 = eg_update(w0, fusion_gradient([1.0, 1.0], reward=0.0), eta=0.3)
    assert np.allclose(w0, w1)


# ---- FTRL --------------------------------------------------------------------
def test_ftrl_l1_zeros_a_useless_channel():
    f = FTRL(dim=3, alpha=0.3, l1=0.5, l2=0.1)
    # channels 0,1 get consistent (rewarded) gradient; channel 2 never does.
    for _ in range(30):
        f.update(np.array([-1.0, -1.0, 0.0]))
    w = f.weights()
    assert w[2] == 0.0                         # L1 prunes the never-active channel
    assert w[0] > 0 and w[1] > 0


def test_fusion_weight_learner_eg_and_ftrl():
    for method in ("eg", "ftrl"):
        learner = FusionWeightLearner(["dense", "bm25", "graph"], method=method, eta=0.2)
        for _ in range(25):
            learner.observe([1.0, 0.2, 0.0], reward=1.0)   # dense surfaces the answer
        d = learner.as_dict()
        assert abs(sum(d.values()) - 1.0) < 1e-6
        assert d["dense"] >= d["graph"]


def test_nonneg_normalize():
    w = nonneg_normalize(np.array([-1.0, 2.0, 0.0]))
    assert (w >= 0).all() and abs(w.sum() - 1.0) < 1e-9


# ---- Rocchio PRF -------------------------------------------------------------
def test_rocchio_moves_query_toward_relevant_centroid():
    q = np.array([1.0, 0.0], dtype=np.float32)
    new = rocchio_expand(q, relevant_vecs=[[0.0, 1.0]], beta=0.6)
    assert new[1] > 0 and new[0] > 0           # pulled toward [0,1] but keeps q direction
    # positive-only is the default (gamma=0): negative set ignored.
    same = rocchio_expand(q, relevant_vecs=[[0.0, 1.0]], nonrelevant_vecs=[[1.0, 0.0]], beta=0.6)
    assert np.allclose(new, same)


def test_rocchio_negative_feedback_pushes_away():
    q = np.array([1.0, 1.0], dtype=np.float32)
    pushed = rocchio_expand(q, relevant_vecs=[[1.0, 0.0]], nonrelevant_vecs=[[0.0, 1.0]],
                            beta=0.6, gamma=0.6)
    assert pushed[0] > pushed[1]               # toward [1,0], away from [0,1]


def test_should_expand_confidence_gate():
    assert should_expand(0.8, 0.5) is True
    assert should_expand(0.3, 0.5) is False


# ---- bandits -----------------------------------------------------------------
def test_ucb1_plays_each_arm_then_exploits():
    b = UCB1(3)
    assert [b.select() for _ in range(0)] == []
    # first three selects must be the unplayed arms
    first = []
    for _ in range(3):
        a = b.select()
        first.append(a)
        b.update(a, reward=1.0 if a == 0 else 0.0)
    assert set(first) == {0, 1, 2}
    # arm 0 is best; after more pulls it should dominate selections
    picks = []
    for _ in range(50):
        a = b.select()
        picks.append(a)
        b.update(a, reward=1.0 if a == 0 else 0.0)
    assert picks.count(0) > picks.count(1) + picks.count(2)


def test_thompson_converges_to_best_arm():
    b = ThompsonBeta(3, seed=1)
    rng = np.random.default_rng(2)
    for _ in range(300):
        a = b.select()
        # arm 1 has the highest true reward probability
        p = {0: 0.2, 1: 0.8, 2: 0.5}[a]
        b.update(a, reward=float(rng.random() < p))
    assert b.best_arm() == 1


def test_linucb_uses_context():
    b = LinUCB(n_arms=2, dim=2, alpha=0.5, lam=1.0)
    x0, x1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    # arm 0 is rewarded under context x0; arm 1 under context x1.
    for _ in range(40):
        b.update(0, x0, reward=1.0)
        b.update(0, x1, reward=0.0)
        b.update(1, x1, reward=1.0)
        b.update(1, x0, reward=0.0)
    assert b.select(x0) == 0
    assert b.select(x1) == 1


def test_discounted_ucb_tracks_drift():
    b = DiscountedUCB(2, gamma=0.85)
    # early: arm 0 is good; later: arm 1 becomes good. Discounting should follow the switch.
    for _ in range(40):
        b.update(0, 1.0)
        b.update(1, 0.0)
    for _ in range(40):
        b.update(0, 0.0)
        b.update(1, 1.0)
    assert b.means()[1] > b.means()[0]


# ---- buffer -> learner -> persistence (dev-only) -----------------------------
def test_learn_fusion_weights_from_dev_buffer(tmp_path):
    from eidetic.feedback import FeedbackBuffer
    from eidetic.optim.online_weights import (learn_fusion_weights, load_weights,
                                              save_weights)

    fb = FeedbackBuffer(tmp_path / "fb.sqlite")
    chans = ["dense", "bm25", "graph"]
    for _ in range(30):
        fb.append("user-1", "q", {"contrib_dense": 1.0, "contrib_bm25": 0.2,
                                   "contrib_graph": 0.0}, reward=1.0)
    # a BENCHMARK feedback row that points at 'graph' -- must be ignored by the learner.
    fb.append("eidetic-plus-locomo-g0-r0", "bq",
              {"contrib_dense": 0.0, "contrib_bm25": 0.0, "contrib_graph": 1.0}, reward=1.0)

    rows = fb.sample(limit=100)                 # dev rows only (is_dev=1)
    w = learn_fusion_weights(rows, chans, method="eg", eta=0.2)
    assert w["dense"] == max(w.values())        # dev signal favored dense, not graph
    p = tmp_path / "fusion_weights.json"
    save_weights(p, w)
    assert load_weights(p)["dense"] == w["dense"]


def test_retrieve_uses_learned_weights(fresh_settings):
    from dataclasses import replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, Scope
    from eidetic.optim.online_weights import save_weights
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    save_weights(fresh_settings.index_dir / "fusion_weights.json",
                 {"dense": 0.7, "bm25": 0.2, "graph": 0.1})
    settings = replace(fresh_settings, fusion_learner_enabled=True)
    store = RecordStore(settings.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), object(), object(), settings)
    assert r._content_weights() == (0.7, 0.2, 0.1)

    # with the flag off the static config weights are used.
    r2 = Retriever(store, object(), KnowledgeGraph(store), object(), object(), fresh_settings)
    assert r2._content_weights() == (fresh_settings.rrf_w_dense, fresh_settings.rrf_w_bm25,
                                     fresh_settings.rrf_w_graph)


def test_rocchio_runs_through_retrieve(fresh_settings):
    from dataclasses import replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    scope = Scope(namespace="prf")
    store = RecordStore(fresh_settings.sqlite_path)
    vecs = {f"m{i}": np.eye(4, dtype=np.float32)[i % 4] for i in range(4)}
    for mid in vecs:
        store.upsert_record(MemoryRecord(memory_id=mid, content_hash=mid,
                                         text=f"alice fact {mid}", scope=scope, valid_at=1.0))

    class FakeIndex:
        def __len__(self):
            return len(vecs)

        def search(self, q, k, allowed_ids=None, ef=None):
            q = q / (np.linalg.norm(q) or 1.0)
            sims = {m: float(v @ q) for m, v in vecs.items()
                    if allowed_ids is None or m in allowed_ids}
            return sorted(sims.items(), key=lambda x: -x[1])[:k]

        def get_vectors(self, ids):
            return {m: vecs[m] for m in ids if m in vecs}

    settings = replace(fresh_settings, rerank_enabled=False, rocchio_enabled=True,
                       rocchio_conf_gate=0.0)            # gate open so PRF always fires
    r = Retriever(store, FakeIndex(), KnowledgeGraph(store), object(), object(), settings)
    out = r.retrieve("alice", scope=scope, qvec=np.array([1.0, 0, 0, 0], np.float32),
                     use_recency=False)
    assert out and len({c.record.memory_id for c in out}) == len(out)
