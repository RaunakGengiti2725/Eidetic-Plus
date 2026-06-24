"""Layer 2a/2c -- training-free margin/entropy gates (TARG-style).

Two cheap, model-free signals computed from a score distribution you already have:

  * margin  = s[0] - s[1]   (how dominant the top hit is)
  * entropy = Shannon entropy of softmax(scores), normalized to [0,1]
              (how spread-out / uncertain the distribution is)

Used for two levers from the playbook:
  * skip the cross-encoder rerank when the first-stage margin is already large (2c):
    a confident shortlist does not need the extra latency.
  * TARG retrieve/skip gate (2a): when the top-1 evidence is overwhelmingly strong, the
    answer is effectively settled; the gate reports high confidence so the caller may skip
    heavier retrieval. (We gate on the strong, conservative side -- only skip when very
    confident -- so recall is never traded away on a borderline query.)
"""
from __future__ import annotations

import numpy as np


def score_margin(scores) -> float:
    s = np.sort(np.asarray(list(scores), dtype=float))[::-1]
    if s.size == 0:
        return 0.0
    if s.size == 1:
        return float(s[0])
    return float(s[0] - s[1])


def score_entropy(scores) -> float:
    """Normalized Shannon entropy of softmax(scores) in [0,1]. 0 = one item dominates,
    1 = uniform/uncertain."""
    s = np.asarray(list(scores), dtype=float)
    n = s.size
    if n <= 1:
        return 0.0
    s = s - s.max()
    p = np.exp(s)
    p = p / p.sum()
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return float(h / np.log(n))


def should_skip_rerank(fused_scores, margin_threshold: float) -> bool:
    """True when the first-stage margin is at/above the threshold, so the expensive
    cross-encoder rerank can be skipped. margin_threshold <= 0 -> never skip."""
    if margin_threshold <= 0.0:
        return False
    return score_margin(fused_scores) >= margin_threshold


def retrieval_confidence(top1_sim: float, margin: float) -> float:
    """A simple confidence blend for the TARG gate: both a strong top-1 match and a clear
    margin over the runner-up. Bounded [0,1]."""
    return float(max(0.0, min(1.0, 0.5 * top1_sim + 0.5 * min(1.0, margin))))
