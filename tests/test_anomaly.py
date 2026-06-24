"""Offline tests for per-triple anomaly scoring (no key)."""
from __future__ import annotations

import numpy as np

from eidetic.dreaming.anomaly import (edge_anomaly_scores, edge_coherence, flag_anomalous,
                                      local_outlier_factor)
from eidetic.models import Edge


def test_lof_flags_the_planted_outlier():
    rng = np.random.default_rng(0)
    cluster = rng.normal(0, 0.05, (30, 4)) + np.array([1.0, 0, 0, 0])
    outlier = np.array([[5.0, 5.0, 5.0, 5.0]])
    X = np.vstack([cluster, outlier])
    lof = local_outlier_factor(X, k=5)
    assert int(np.argmax(lof)) == len(X) - 1      # the outlier is the most anomalous
    assert lof[-1] > 1.5 and lof[:-1].max() < lof[-1]


def _edge(src, dst):
    return Edge(src=src, dst=dst, relation="rel", fact=f"{src} rel {dst}")


def test_incoherent_edge_is_most_anomalous():
    # a tight cluster of entities near [1,0]; one "far" entity at [-1,0].
    rng = np.random.default_rng(1)
    vecs = {f"e{i}": (np.array([1.0, 0.0]) + rng.normal(0, 0.02, 2)) for i in range(12)}
    vecs["far"] = np.array([-1.0, 0.0])
    edges = [_edge(f"e{i}", f"e{i+1}") for i in range(11)]   # coherent neighbours
    edges.append(_edge("e0", "far"))                          # the suspect edge
    scores = edge_anomaly_scores(edges, vecs, k=4)
    assert int(np.argmax(scores)) == len(edges) - 1           # the incoherent edge wins
    flagged = flag_anomalous(edges, scores, threshold=0.4)
    assert edges[-1] in flagged


def test_edge_coherence_neutral_when_vector_missing():
    coh = edge_coherence([_edge("a", "b")], entity_vectors={})
    assert coh[0] == 0.5


def test_transe_signal_blends_in():
    class FakeTransE:
        def score(self, h, r, t):
            return 0.1 if t == "far" else 0.9     # 'far' triples are implausible
    rng = np.random.default_rng(2)
    vecs = {f"e{i}": (np.array([1.0, 0.0]) + rng.normal(0, 0.02, 2)) for i in range(6)}
    vecs["far"] = np.array([1.0, 0.0])            # coherent vectors, so only TransE flags it
    edges = [_edge(f"e{i}", f"e{i+1}") for i in range(5)] + [_edge("e0", "far")]
    scores = edge_anomaly_scores(edges, vecs, transe=FakeTransE(), k=3,
                                 w_lof=0.0, w_coh=0.0, w_transe=1.0)
    assert int(np.argmax(scores)) == len(edges) - 1   # TransE alone surfaces the bad triple
