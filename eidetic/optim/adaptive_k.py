"""Layer 2a -- Adaptive-k context selection by the largest-gap cut.

"No Tuning, No Iteration, Just Adaptive-k" (EMNLP 2025): instead of a fixed top-k, sort
the relevance scores descending and cut at the single largest gap in the distribution.
Factoid queries have a sharp cliff after the few real hits, so this keeps just those few
(big token savings); diffuse queries have no cliff, so it keeps more. Pure numpy on
already-computed scores -> latency-neutral, single pass, scale-free (works for
cosine/dot/RRF scores alike).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def largest_gap_k(scores, min_k: int = 1, max_k: Optional[int] = None) -> int:
    """Return the cut size k: the rank just before the largest consecutive gap in the
    descending score distribution, clamped to [min_k, max_k]. The input need not be
    pre-sorted. Empty -> 0; a single score -> min_k."""
    s = np.sort(np.asarray(list(scores), dtype=float))[::-1]
    n = s.size
    if n == 0:
        return 0
    hi = n if max_k is None else max(1, min(int(max_k), n))
    lo = max(1, min(int(min_k), hi))
    if n == 1 or hi == lo:
        return lo
    # gap after rank i (0-indexed) is s[i]-s[i+1]; cutting there keeps k=i+1 items.
    # Valid cut indices i give k=i+1 in [lo, hi] AND need s[i+1] to exist (i<=n-2),
    # i.e. i in [lo-1, min(hi-1, n-2)]. (min(hi,n)-2 was one too small for max_k<n.)
    i_hi = min(hi - 1, n - 2)
    if i_hi < lo - 1:
        return lo
    gaps = s[lo - 1:i_hi + 1] - s[lo:i_hi + 2]
    return lo + int(np.argmax(gaps))


def adaptive_k_cut(candidates: list, score_fn, min_k: int = 1,
                   max_k: Optional[int] = None) -> list:
    """Truncate an already-ranked candidate list at the largest score gap. `score_fn`
    extracts the relevance score from a candidate. Order is preserved; only the tail past
    the cliff is dropped."""
    if not candidates:
        return candidates
    scores = [float(score_fn(c)) for c in candidates]
    cap = len(candidates) if max_k is None else min(int(max_k), len(candidates))
    k = largest_gap_k(scores, min_k=min_k, max_k=cap)
    return candidates[:k]
