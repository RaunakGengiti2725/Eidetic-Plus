"""Component 4: write-time salience gating (neuromodulatory analog) + EM-LLM-style
surprise-based event segmentation.

salience = f(novelty/surprise, importance). Surprise is Bayesian-surprise-style: the
embedding distance to the nearest already-stored memory WITHIN THE SAME SCOPE (a novel
event is far from everything seen). Importance is judged by qwen-flash (a real call).
The result sets each memory's initial FSRS state and replay priority.

Segmentation: long inputs are chunked at Bayesian-surprise boundaries (spikes in
consecutive-sentence embedding distance) rather than fixed windows, so stored episodes
align to natural event boundaries (EM-LLM, dossier 4.2/13.5).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dashscope_client import DashScopeClient
from .models import Scope
from .store import RecordStore
from .vector_index import VectorIndex


@dataclass
class Salience:
    surprise: float
    importance: float
    salience: float


def compute_surprise(
    content_vec: np.ndarray,
    index: VectorIndex,
    store: Optional[RecordStore] = None,
    scope: Optional[Scope] = None,
) -> float:
    """1 - cosine similarity to the nearest stored memory in scope. 1.0 if none yet."""
    if len(index) == 0:
        return 1.0
    cand = index.search(content_vec, k=min(50, len(index)))
    if store is not None and scope is not None:
        in_scope = store.ids_in_scope(scope)
        cand = [(mid, s) for mid, s in cand if mid in in_scope]
    if not cand:
        return 1.0
    sim = cand[0][1]
    return float(max(0.0, min(1.0, 1.0 - sim)))


def score(
    text: str,
    content_vec: np.ndarray,
    index: VectorIndex,
    client: DashScopeClient,
    store: Optional[RecordStore] = None,
    scope: Optional[Scope] = None,
) -> Salience:
    surprise = compute_surprise(content_vec, index, store, scope)
    importance = client.score_importance(text)  # real qwen-flash call
    salience = float(max(0.0, min(1.0, 0.45 * surprise + 0.55 * importance)))
    return Salience(surprise=surprise, importance=importance, salience=salience)


# ---- affect-modulated salience (Phase 3; pure, age-free) ----------------------------------
_EMPHASIS_RE = re.compile(
    r"\b(remember (this|that)|important|never forget|don'?t forget|do not forget|"
    r"keep in mind|make sure|note that|crucial|critical|by the way remember)\b", re.I)


def emphasis_score(text: str) -> float:
    """User-emphasis cue strength in [0,1] from deterministic signals: explicit 'remember this /
    important / never forget', exclamation, and shouted ALL-CAPS words. No model call, no age."""
    t = text or ""
    score = 0.0
    if _EMPHASIS_RE.search(t):
        score += 0.5
    if "!" in t:
        score += 0.2 * min(t.count("!"), 3) / 3.0
    if re.findall(r"\b[A-Z]{3,}\b", t):
        score += 0.3
    return float(max(0.0, min(1.0, score)))


def affect_salience(arousal: float, importance: float, surprise: float, emphasis: float,
                    verified_helpful: float, *, w_arousal: float = 1.0, w_importance: float = 1.0,
                    w_surprise: float = 1.0, w_emphasis: float = 1.0, w_helpful: float = 0.0) -> float:
    """Static salience s = sigmoid(z - midpoint), z = weighted sum of the affect/usage signals.

    CRITICAL: there is NO timestamp / age term here (audited by inspection and the age-stratified
    test). The midpoint centers neutral (all-0.5) inputs at s=0.5 so the score spreads either way.
    `verified_helpful` enters with w_helpful (default 0 until Phase 4 wires the count)."""
    z = (w_arousal * arousal + w_importance * importance + w_surprise * surprise
         + w_emphasis * emphasis + w_helpful * verified_helpful)
    midpoint = 0.5 * (w_arousal + w_importance + w_surprise + w_emphasis + w_helpful)
    return float(1.0 / (1.0 + np.exp(-(z - midpoint))))


_SENT = re.compile(r"(?<=[.!?])\s+")


def segment_by_surprise(text: str, client: DashScopeClient, *, max_sentences: int = 4) -> list[str]:
    """Split text into episodes at Bayesian-surprise boundaries.

    Surprise = consecutive-sentence embedding distance; a spike (> mean + std) starts a
    new episode. Short inputs are returned as a single episode. Real embedding calls."""
    sentences = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return [text.strip()] if text.strip() else []

    embs = client.embed_texts(sentences)  # real call, batched
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    dists = np.array([1.0 - float(embs[i] @ embs[i - 1]) for i in range(1, len(sentences))])
    thresh = float(dists.mean() + dists.std()) if len(dists) > 1 else 1.0

    episodes: list[str] = []
    cur: list[str] = [sentences[0]]
    for i, d in enumerate(dists, start=1):
        # New episode on a surprise spike, or when the current episode gets long.
        if d > thresh or len(cur) >= max_sentences * 2:
            episodes.append(" ".join(cur))
            cur = [sentences[i]]
        else:
            cur.append(sentences[i])
    if cur:
        episodes.append(" ".join(cur))
    return episodes
