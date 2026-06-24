"""Layer 2b -- Split-conformal prediction for calibrated retrieval depth & abstention.

Split-conformal procedure (distribution-free, pure numpy: a sort + a quantile):
  1. On a held-out DEV calibration set, compute nonconformity scores s_i = 1 - sim, where
     sim is the similarity of the chunk that actually contains the answer.
  2. q_hat = the ceil((n+1)(1-alpha)) / n empirical quantile of {s_i}.
  3. At inference, return every chunk with nonconformity <= q_hat, i.e. sim >= 1 - q_hat.
     This includes the true evidence with probability >= 1 - alpha.

CAVEAT carried from the literature (and surfaced honestly): exchangeability does NOT hold
a priori for retrieval scores at finite depth, so the 1-alpha coverage is a calibrated
TARGET, not a hard proof. Calibrate per retriever and recalibrate under drift.

This is the genuine split-conformal q_hat, distinct from the precision-target grid search
in bench/calibrate.py (which is a useful operating-point selector but offers no
distribution-free coverage statement).
"""
from __future__ import annotations

import math

import numpy as np

INF = float("inf")


def nonconformity_from_sims(sims) -> np.ndarray:
    """s_i = 1 - sim. Higher = less conforming (worse evidence match)."""
    return 1.0 - np.asarray(list(sims), dtype=float)


def split_conformal_qhat(nonconformity, alpha: float = 0.1) -> float:
    """The split-conformal threshold q_hat = ceil((n+1)(1-alpha))/n empirical quantile of
    the nonconformity scores. If the required rank exceeds n (alpha too small for the
    calibration size to certify), returns +inf -> include everything (cannot guarantee, so
    do not prune). Empty calibration -> +inf for the same reason."""
    s = np.sort(np.asarray(list(nonconformity), dtype=float))
    n = s.size
    if n == 0:
        return INF
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:                      # no finite threshold certifies this coverage
        return INF
    rank = max(1, min(rank, n))
    return float(s[rank - 1])


def coverage_cutoff(qhat: float) -> float:
    """The similarity cutoff implied by q_hat: keep chunks with sim >= 1 - q_hat."""
    if qhat == INF:
        return -INF                   # include everything
    return 1.0 - qhat


def select_by_conformal(candidates: list, sim_fn, qhat: float, *, min_keep: int = 1) -> list:
    """Keep candidates whose similarity meets the conformal cutoff (sim >= 1 - q_hat),
    preserving order. Always keep at least `min_keep` so a strict q_hat never empties the
    context. q_hat = +inf keeps everything."""
    if not candidates or qhat == INF:
        return candidates
    cutoff = coverage_cutoff(qhat)
    kept = [c for c in candidates if float(sim_fn(c)) >= cutoff]
    if len(kept) < min_keep:
        return candidates[:min_keep]
    return kept


def calibrate_qhat_from_pairs(pairs: list[dict], alpha: float = 0.1,
                              sim_key: str = "answer_sim") -> dict:
    """Compute q_hat from DEV calibration rows [{answer_sim: float}, ...], where answer_sim
    is the similarity of the evidence chunk that contained the gold answer. Returns the
    threshold + the implied similarity cutoff + n. Pure math, no model call, no fabrication.
    """
    sims = [float(p[sim_key]) for p in pairs if sim_key in p]
    if not sims:
        return {"ok": False, "note": f"no calibration rows with a '{sim_key}' field"}
    qhat = split_conformal_qhat(nonconformity_from_sims(sims), alpha)
    return {
        "ok": True, "alpha": alpha, "qhat": qhat,
        "sim_cutoff": coverage_cutoff(qhat), "n": len(sims),
        "note": ("coverage is a calibrated target, not a proof: retrieval scores are not "
                 "exchangeable at finite depth -- recalibrate under drift"),
    }
