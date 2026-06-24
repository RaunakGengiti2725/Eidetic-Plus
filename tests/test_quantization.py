"""Offline tests for Layer-3c quantization: recall vs exact cosine + estimator math."""
from __future__ import annotations

import numpy as np

from eidetic.optim.quantize import (QuantizedANN, make_rotation, rabitq_cosine_estimate,
                                    rabitq_encode, rabitq_hamming, sq8_encode, sq8_scores)


def _unit_rows(n, d, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _exact_topk(M, q, k):
    sims = M @ (q / np.linalg.norm(q))
    return set(np.argsort(-sims)[:k].tolist())


def _recall(M, k, kind, refine, seed=1):
    rng = np.random.default_rng(seed)
    ann = QuantizedANN(kind=kind, refine_enabled=refine, refine_topn=100).fit(M)
    hits = total = 0
    for _ in range(20):
        q = rng.standard_normal(M.shape[1]).astype(np.float32)
        exact = _exact_topk(M, q, k)
        got = {i for i, _ in ann.search(q, k)}
        hits += len(exact & got)
        total += k
    return hits / total


# ---- SQ8 --------------------------------------------------------------------
def test_sq8_codes_are_int8_ranged():
    M = _unit_rows(50, 64, seed=0)
    codes = sq8_encode(M)
    assert codes.dtype == np.int8
    assert codes.min() >= -127 and codes.max() <= 127


def test_sq8_scores_track_exact_cosine():
    M = _unit_rows(40, 64, seed=2)
    q = M[0]                                   # query equals a stored vector
    approx = sq8_scores(sq8_encode(M), q)
    assert int(np.argmax(approx)) == 0         # its own row ranks first
    assert approx[0] > 0.98


def test_sq8_two_stage_recall_is_high():
    M = _unit_rows(300, 128, seed=3)
    assert _recall(M, k=10, kind="sq8", refine=True) >= 0.95


# ---- RaBitQ / binary 1-bit --------------------------------------------------
def test_rabitq_estimator_endpoints():
    v = _unit_rows(1, 256, seed=4)[0]
    packed, dim = rabitq_encode(v)
    # identical -> hamming 0 -> cos 1
    h_same = rabitq_hamming(packed, packed[0])
    assert h_same[0] == 0
    assert abs(rabitq_cosine_estimate(h_same, dim)[0] - 1.0) < 1e-9
    # exact opposite sign pattern -> hamming D -> cos(pi) = -1
    opp_packed, _ = rabitq_encode(-v)
    h_opp = rabitq_hamming(opp_packed, packed[0])
    assert h_opp[0] == dim
    assert abs(rabitq_cosine_estimate(h_opp, dim)[0] + 1.0) < 1e-9


def test_rabitq_estimate_approximates_true_cosine():
    M = _unit_rows(200, 512, seed=5)
    q = M[0]
    packed, dim = rabitq_encode(M)
    qpacked, _ = rabitq_encode(q)
    est = rabitq_cosine_estimate(rabitq_hamming(packed, qpacked[0]), dim)
    true = M @ q
    # 1-bit sign codes track the true cosine strongly (the angle estimator), though coarse:
    # this correlation is precisely why a second exact-refine stage is needed for top-k.
    assert np.corrcoef(est, true)[0, 1] > 0.8


def test_rabitq_rotation_restores_estimator_on_anisotropic_data():
    # Anisotropic embedding: variance concentrated in a few dims. WITHOUT the random rotation,
    # raw-axis sign codes correlate poorly with true cosine (the SimHash guarantee needs random
    # hyperplanes). The fixed orthonormal rotation restores the random-hyperplane property.
    rng = np.random.default_rng(7)
    scale = np.ones(128, dtype=np.float32)
    scale[:8] = 10.0
    M = (rng.standard_normal((200, 128)) * scale).astype(np.float32)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    q = M[0]
    true = M @ q

    p0, d = rabitq_encode(M)                        # no rotation (broken on anisotropic data)
    qp0, _ = rabitq_encode(q)
    est0 = rabitq_cosine_estimate(rabitq_hamming(p0, qp0[0]), d)

    R = make_rotation(128, seed=1)                  # rotated (RaBitQ-correct)
    p1, _ = rabitq_encode(M, R)
    qp1, _ = rabitq_encode(q, R)
    est1 = rabitq_cosine_estimate(rabitq_hamming(p1, qp1[0]), d)

    corr0 = np.corrcoef(est0, true)[0, 1]
    corr1 = np.corrcoef(est1, true)[0, 1]
    assert corr1 > 0.8                              # rotated estimator tracks true cosine
    assert corr1 > corr0 + 0.3                      # and is far better than raw-axis sign codes


def test_rabitq_two_stage_recovers_recall():
    M = _unit_rows(300, 256, seed=6)
    # The cheap 1-bit codes carry real signal (far above the 10/300≈0.03 random baseline)
    # but are lossy at top-10; this is the documented tradeoff.
    raw_recall = _recall(M, k=10, kind="rabitq", refine=False)
    assert raw_recall >= 0.15
    # The exact float32 refine pass over the code-shortlist recovers high recall@10 -- the
    # whole point of keeping the raw vectors for stage 2.
    refined_recall = _recall(M, k=10, kind="rabitq", refine=True)
    assert refined_recall >= 0.8
    assert refined_recall > raw_recall


# ---- QuantizedVectorIndex backend (matches the exact index with refine on) --
def test_quantized_index_backend_matches_exact(tmp_path):
    from eidetic.vector_index import NumpyVectorIndex, QuantizedVectorIndex

    M = _unit_rows(120, 64, seed=7)
    ids = [f"m{i}" for i in range(len(M))]
    exact = NumpyVectorIndex(tmp_path / "exact", dim=64, struct_dim=4)
    quant = QuantizedVectorIndex(tmp_path / "quant", dim=64, struct_dim=4,
                                 kind="rabitq", refine=True, refine_topn=64)
    for mid, v in zip(ids, M):
        exact.add(mid, v)
        quant.add(mid, v)

    rng = np.random.default_rng(11)
    hits = total = 0
    for _ in range(15):
        q = rng.standard_normal(64).astype(np.float32)
        e = {mid for mid, _ in exact.search(q, 10)}
        g = {mid for mid, _ in quant.search(q, 10)}
        hits += len(e & g)
        total += 10
    assert hits / total >= 0.85          # refine keeps it close to exact

    # allowed_ids restricts results to the in-scope subset.
    allowed = set(ids[:20])
    res = quant.search(rng.standard_normal(64).astype(np.float32), 10, allowed_ids=allowed)
    assert res and all(mid in allowed for mid, _ in res)


def test_make_vector_index_selects_quantized(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("VECTOR_QUANT", "sq8")
    from eidetic.config import get_settings
    from eidetic.vector_index import QuantizedVectorIndex

    get_settings.cache_clear()
    idx = None
    try:
        from eidetic.vector_index import make_vector_index
        idx = make_vector_index(get_settings())
        assert isinstance(idx, QuantizedVectorIndex) and idx.kind == "sq8"
    finally:
        get_settings.cache_clear()
