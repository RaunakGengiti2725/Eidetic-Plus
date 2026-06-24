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
