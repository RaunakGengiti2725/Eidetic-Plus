"""Layer 2d -- Score-fusion variants beyond weighted RRF.

Weighted RRF (RRF(d)=sum_c w_c/(k+rank_c(d))) lives in retrieval.py and stays the robust,
scale-free default. These are the score-based alternatives the playbook adds for channels
whose magnitudes vary greatly (BM25 is unbounded; cosine is in [-1,1]):

  * z-score (ZMUV): per channel (x-mean)/std, then weighted sum.
  * min-max / Relative Score Fusion: per channel (x-min)/(max-min), then weighted sum.
  * DBSF (Distribution-Based Score Fusion): per channel normalize with mean +- 3*sigma as
    the limits (clip to [0,1]), then weighted sum -- robust to outliers and heavy tails.
  * Borda: pure rank-position votes (no scores needed).

All pure numpy. A channel that carries no discriminative signal (zero variance / single
item) contributes nothing rather than a spurious constant.
"""
from __future__ import annotations

import numpy as np

SCORE_METHODS = ("zscore", "minmax", "dbsf")
ALL_METHODS = ("rrf", "borda") + SCORE_METHODS


def _normalize(values: np.ndarray, method: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    if method == "zscore":
        std = v.std()
        return (v - v.mean()) / std if std > 0 else np.zeros_like(v)
    if method == "minmax":
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    if method == "dbsf":
        mean, std = v.mean(), v.std()
        lo, hi = mean - 3.0 * std, mean + 3.0 * std
        if hi <= lo:
            return np.zeros_like(v)
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    raise ValueError(f"unknown score-fusion method {method!r}")


def combine_scores(channel_maps: list[dict], weights: list[float] | None,
                   method: str) -> dict:
    """Weighted sum of per-channel normalized scores. channel_maps[c] = {id: raw_score}.
    Each channel is normalized independently (so scale mismatch is handled), then summed
    with weights. An id absent from a channel contributes 0 from that channel."""
    if method not in SCORE_METHODS:
        raise ValueError(f"{method!r} is not a score-fusion method; use one of {SCORE_METHODS}")
    fused: dict[str, float] = {}
    for c, cmap in enumerate(channel_maps):
        if not cmap:
            continue
        w = weights[c] if weights and c < len(weights) else 1.0
        ids = list(cmap.keys())
        norm = _normalize(np.array([cmap[i] for i in ids], dtype=float), method)
        for i, mid in enumerate(ids):
            fused[mid] = fused.get(mid, 0.0) + w * float(norm[i])
    return fused


def combine_borda(orderings: list[list[str]], weights: list[float] | None = None) -> dict:
    """Borda count: in a channel ranking of length L, the item at rank r (0-indexed) scores
    (L - r). Weighted sum across channels. Rank-only -- no scores required."""
    fused: dict[str, float] = {}
    for c, ranking in enumerate(orderings):
        w = weights[c] if weights and c < len(weights) else 1.0
        L = len(ranking)
        for r, mid in enumerate(ranking):
            fused[mid] = fused.get(mid, 0.0) + w * (L - r)
    return fused
