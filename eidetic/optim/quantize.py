"""Layer 3c -- vector quantization (the PDF's single highest-ROI memory/latency win).

Two lightweight, numpy-only quantizers plus a two-stage search that recovers recall:

  * Scalar quantization (SQ8): float32 -> int8, 4x compression. Vectors are already
    L2-normalized (components in [-1,1]), so a global scale of 127 is exact-ranged.
    Approximate cosine = int8 dot / 127^2.

  * Binary 1-bit (RaBitQ-style / SimHash sign codes): rotate by a fixed random orthonormal
    matrix (make_rotation), then store the sign bit per dimension -- 1-to-32 compression.
    Distance is a bitwise XOR + popcount (Hamming). The rotation is what makes the sign bits
    behave as random hyperplanes, so the angle estimator cos_est = cos(pi * hamming / D) is
    asymptotically unbiased with error O(1/sqrt(D)). (Without the rotation this holds only for
    already-isotropic data and fails on anisotropic real embeddings.) popcount uses a uint8
    lookup table (no popcount intrinsic needed).

Two-stage search (the recall recovery): stage 1 ranks ALL vectors cheaply over the compact
codes to get a shortlist of N >> k; stage 2 re-scores that shortlist with EXACT float32
cosine to produce the final top-k. Keeping the raw float32 for the refine pass trades some of
the compression for recall -- a documented tension: drop the raw vectors only once a dev-split
recall check confirms the estimator holds within the >1%-drop rule.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# popcount lookup table over a byte: bits set in 0..255.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

# RaBitQ random-rotation: sign codes estimate the angle (cos(pi*h/D)) ONLY when the
# projection directions are random. Real embeddings are anisotropic, so we rotate every
# vector (and the query) by a fixed random orthonormal matrix before sign-coding -- this is
# what makes the standard-basis sign bits behave as random hyperplanes. The rotation is
# deterministic from (dim, seed), so it is reproducible across index rebuilds without
# persisting the matrix, and the DB and query always share it.
RABITQ_SEED = 1234567
_ROTATION_CACHE: dict[tuple[int, int], np.ndarray] = {}


def make_rotation(dim: int, seed: int = RABITQ_SEED) -> np.ndarray:
    """A deterministic DxD orthonormal rotation (QR of a seeded Gaussian), cached per (dim, seed)."""
    key = (int(dim), int(seed))
    R = _ROTATION_CACHE.get(key)
    if R is None:
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        R = Q.astype(np.float32)
        _ROTATION_CACHE[key] = R
    return R


def _l2(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# ---- scalar quantization (SQ8) ---------------------------------------------
def sq8_encode(matrix: np.ndarray, scale: float = 127.0) -> np.ndarray:
    """L2-normalize then map each component in [-1,1] to int8 [-127,127]."""
    m = _l2(matrix)
    return np.clip(np.round(m * scale), -127, 127).astype(np.int8)


def sq8_scores(codes: np.ndarray, qvec: np.ndarray, scale: float = 127.0) -> np.ndarray:
    """Approximate cosine of every row code against the query (encoded the same way)."""
    qcode = sq8_encode(qvec, scale).astype(np.int32).reshape(-1)
    return (codes.astype(np.int32) @ qcode) / float(scale * scale)


# ---- binary 1-bit quantization (RaBitQ-style sign codes) -------------------
def rabitq_encode(matrix: np.ndarray, rotation: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
    """Pack the per-dimension sign bits of the (optionally rotated) L2-normalized rows. The
    rotation makes the sign bits behave as random hyperplanes (see make_rotation). Returns
    (packed uint8 [N, ceil(D/8)], D)."""
    m = _l2(matrix)
    if rotation is not None:
        m = m @ rotation
    bits = (m >= 0).astype(np.uint8)
    return np.packbits(bits, axis=1), m.shape[1]


def rabitq_hamming(packed: np.ndarray, qpacked: np.ndarray) -> np.ndarray:
    """Hamming distance (number of differing sign bits) of every row against the query,
    via XOR + a byte popcount table."""
    xor = np.bitwise_xor(packed, qpacked.reshape(1, -1).astype(np.uint8))
    return _POPCOUNT[xor].sum(axis=1).astype(np.int64)


def rabitq_cosine_estimate(hamming: np.ndarray, dim: int) -> np.ndarray:
    """Unbiased angle estimator for sign codes: cos(pi * hamming / D). The fraction of
    differing sign bits estimates angle/pi (random-hyperplane / SimHash)."""
    return np.cos(np.pi * np.asarray(hamming, dtype=np.float64) / max(1, dim))


def rabitq_scores(packed: np.ndarray, dim: int, qvec: np.ndarray) -> np.ndarray:
    qpacked, _ = rabitq_encode(qvec)
    return rabitq_cosine_estimate(rabitq_hamming(packed, qpacked[0]), dim)


# ---- two-stage (approximate shortlist -> exact refine) ----------------------
def refine(shortlist_idx: np.ndarray, raw: np.ndarray, qvec: np.ndarray,
           k: int) -> list[tuple[int, float]]:
    """Re-score a shortlist with EXACT float32 cosine and return the top-k (idx, sim)."""
    if len(shortlist_idx) == 0:
        return []
    q = _l2(qvec)[0]
    sub = _l2(raw[shortlist_idx])
    sims = sub @ q
    k = min(k, sims.shape[0])
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(shortlist_idx[i]), float(sims[i])) for i in top]


class QuantizedANN:
    """Brute-force quantized index: cheap code ranking + optional exact refine. Holds the
    raw float32 vectors so refine is exact; testable standalone (no DashScope/SQLite)."""

    def __init__(self, kind: str = "rabitq", refine_enabled: bool = True,
                 refine_topn: int = 100, scale: float = 127.0):
        if kind not in ("sq8", "rabitq"):
            raise ValueError("kind must be 'sq8' or 'rabitq'")
        self.kind = kind
        self.refine_enabled = refine_enabled
        self.refine_topn = int(refine_topn)
        self.scale = scale
        self.raw = np.zeros((0, 0), dtype=np.float32)
        self._codes = None
        self._dim = 0
        self._rotation = None

    def fit(self, matrix: np.ndarray) -> "QuantizedANN":
        self.raw = _l2(matrix)
        self._dim = self.raw.shape[1]
        if self.kind == "sq8":
            self._codes = sq8_encode(self.raw, self.scale)
        else:
            self._rotation = make_rotation(self._dim)
            self._codes, self._dim = rabitq_encode(self.raw, self._rotation)
        return self

    def _approx_scores(self, qvec: np.ndarray) -> np.ndarray:
        if self.kind == "sq8":
            return sq8_scores(self._codes, qvec, self.scale)
        qpacked, _ = rabitq_encode(qvec, self._rotation)
        return rabitq_cosine_estimate(rabitq_hamming(self._codes, qpacked[0]), self._dim)

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Stage 1: rank all rows by the cheap code score -> shortlist of N. Stage 2 (if
        enabled): exact float32 cosine on the shortlist -> top-k. Returns [(idx, sim)]."""
        n = self.raw.shape[0]
        if n == 0:
            return []
        approx = self._approx_scores(qvec)
        if not self.refine_enabled:
            kk = min(k, n)
            top = np.argpartition(-approx, kk - 1)[:kk]
            top = top[np.argsort(-approx[top])]
            return [(int(i), float(approx[i])) for i in top]
        n_short = min(max(self.refine_topn, k), n)
        shortlist = np.argpartition(-approx, n_short - 1)[:n_short]
        return refine(shortlist, self.raw, qvec, k)
