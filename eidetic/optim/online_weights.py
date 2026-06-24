"""Layer 3a -- online learning of per-channel fusion weights (FTRL + Exponentiated Gradient).

Both consume a streaming (per-channel contribution, reward) signal from the DEV-split
feedback buffer and adapt the fusion weights so the channel that actually surfaces the
correct memory is up-weighted over time.

  * Exponentiated Gradient (multiplicative weights, a special case of FTRL with an entropic
    regularizer): w_i <- w_i * exp(-eta * grad_i), then renormalize. Keeps weights on the
    simplex -- the natural domain for convex fusion weights.
  * FTRL-Proximal (the Google-Ads CTR workhorse): per-coordinate z/n state with an L1+L2
    closed form. The L1 term drives useless channels to exactly zero (sparse, stable
    weights -> mitigates catastrophic forgetting); regret is O(sqrt T).

AGE-INDEPENDENCE INVARIANT: these learners adjust the CONTENT channels (dense/BM25/graph)
only. The recency channel weight is never learned here, so a reward signal can never inflate
recency and slope the flat recall-vs-age curve.
"""
from __future__ import annotations

import numpy as np


# ---- Exponentiated Gradient (simplex) --------------------------------------
def eg_update(weights, gradient, eta: float = 0.1) -> np.ndarray:
    """Multiplicative-weights step on the simplex: w_i <- w_i*exp(-eta*g_i), renormalized."""
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    w = w * np.exp(-eta * g)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def fusion_gradient(channel_contribs, reward: float) -> np.ndarray:
    """Gradient for a single feedback event. channel_contribs[i] = how strongly channel i
    surfaced the correct memory (e.g. its reciprocal rank). A positive reward makes the
    gradient NEGATIVE for high-contributing channels, so EG raises their weight; a zero
    reward produces no push."""
    c = np.asarray(channel_contribs, dtype=np.float64)
    return -float(reward) * c


# ---- FTRL-Proximal ----------------------------------------------------------
class FTRL:
    """Per-coordinate FTRL-Proximal (McMahan 2013). Learns a linear weight per channel from
    streaming gradients with L1 (sparsity) + L2 regularization."""

    def __init__(self, dim: int, alpha: float = 0.1, beta: float = 1.0,
                 l1: float = 1.0, l2: float = 1.0):
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        self.z = np.zeros(dim, dtype=np.float64)
        self.n = np.zeros(dim, dtype=np.float64)

    def weights(self) -> np.ndarray:
        sign = np.sign(self.z)
        denom = (self.beta + np.sqrt(self.n)) / self.alpha + self.l2
        w = -(self.z - sign * self.l1) / denom
        return np.where(np.abs(self.z) <= self.l1, 0.0, w)   # L1 sparsity gate

    def update(self, gradient) -> None:
        g = np.asarray(gradient, dtype=np.float64)
        w = self.weights()
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.alpha
        self.z += g - sigma * w
        self.n += g * g


# ---- a fusion-weight policy combining the above -----------------------------
def nonneg_normalize(w: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Clip to non-negative and renormalize to a usable weight vector (fusion weights are
    non-negative). A small floor keeps every channel minimally alive."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    if w.sum() <= 0:
        return np.full_like(w, 1.0 / len(w))
    w = w + floor
    return w / w.sum()


class FusionWeightLearner:
    """Maintains content-channel fusion weights via EG (default) or FTRL from feedback.
    `channels` names the content channels it governs (recency is intentionally excluded)."""

    def __init__(self, channels: list[str], method: str = "eg", eta: float = 0.1,
                 init: list[float] | None = None):
        self.channels = list(channels)
        self.method = method
        self.eta = eta
        d = len(channels)
        self._w = (np.asarray(init, dtype=np.float64) if init is not None
                   else np.full(d, 1.0 / d))
        self._w = self._w / self._w.sum()
        # A small L1 so the sparsity gate engages at the scale of reciprocal-rank gradients
        # (the class default l1=1.0 would keep the learner uniform for far too long here).
        self._ftrl = FTRL(d, l1=0.02) if method == "ftrl" else None

    def weights(self) -> np.ndarray:
        if self.method == "ftrl":
            return nonneg_normalize(self._ftrl.weights())
        return self._w.copy()

    def observe(self, channel_contribs, reward: float) -> None:
        g = fusion_gradient(channel_contribs, reward)
        if self.method == "ftrl":
            self._ftrl.update(g)
        else:
            self._w = eg_update(self._w, g, self.eta)

    def as_dict(self) -> dict[str, float]:
        w = self.weights()
        return {ch: float(w[i]) for i, ch in enumerate(self.channels)}


# ---- buffer -> learner loop + persistence (idle cadence) --------------------
def learn_fusion_weights(rows, channels: list[str], method: str = "eg",
                         eta: float = 0.1) -> dict[str, float]:
    """Replay DEV feedback rows through the learner. Each row exposes per-channel reciprocal
    -rank contributions under features['contrib_<channel>'] and a scalar reward. Returns the
    learned content-channel weight dict. Pure replay -- no model call, dev-split only by
    construction (the FeedbackBuffer.sample() that produced `rows` returns is_dev=1 only)."""
    learner = FusionWeightLearner(channels, method=method, eta=eta)
    for r in rows:
        feats = getattr(r, "features", None) or (r.get("features") if isinstance(r, dict) else {})
        reward = getattr(r, "reward", None)
        if reward is None and isinstance(r, dict):
            reward = r.get("reward", 0.0)
        contribs = [float(feats.get(f"contrib_{ch}", 0.0)) for ch in channels]
        learner.observe(contribs, float(reward or 0.0))
    return learner.as_dict()


def save_weights(path, weights: dict[str, float]) -> None:
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(weights, indent=2))


def load_weights(path) -> dict[str, float] | None:
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    try:
        return {str(k): float(v) for k, v in json.loads(p.read_text()).items()}
    except (ValueError, OSError):
        return None
