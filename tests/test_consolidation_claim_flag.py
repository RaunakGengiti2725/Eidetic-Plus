"""CLAIM_EXTRACTION gates consolidation-written claims; affect salience scores during sleep."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import Scope


class _FakeClient:
    """Deterministic no-API client with an affect scorer."""

    def __init__(self, dim: int):
        self.dim = dim
        self.affect_calls = 0

    def _e(self, text):
        v = np.zeros(self.dim, np.float32)
        v[hash(text) % self.dim] = 1.0
        return v

    def embed_text(self, text):
        return self._e(text)

    def embed_texts(self, texts):
        return np.stack([self._e(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def extract_edges(self, text):
        return []

    def extract_edges_bounded(self, text, *, max_windows=0):
        return []

    def extract_claims(self, text):
        return []

    def extract_claims_bounded(self, text, *, max_windows=0):
        return []

    def score_affect(self, text):
        self.affect_calls += 1
        return {"arousal": 0.9, "importance": 0.8, "valence": 0.7}

    def nli(self, premise, hypothesis):
        return ("neutral", 0.0)


def _engine(settings, **overrides):
    s = replace(settings, vector_backend="numpy", rerank_enabled=False,
                semantic_cache_enabled=False, **overrides)
    return Engine(s, client=_FakeClient(s.embed_dim))


def test_claim_extraction_flag_off_writes_no_claims(fresh_settings):
    eng = _engine(fresh_settings, claim_extraction_enabled=False)
    scope = Scope(namespace="claims-off")
    eng.ingest_text("user: I adopted a corgi named Biscuit last spring.",
                    source="s0", scope=scope, consolidate_now=False)
    report = eng.consolidate_pending(scope=scope, score_importance=False)
    assert report["claims_extracted"] == 0
    assert eng.store.active_claims_at(2_000_000_000, scope) == []


def test_claim_extraction_default_on_writes_claims(fresh_settings):
    eng = _engine(fresh_settings)
    scope = Scope(namespace="claims-on")
    eng.ingest_text("user: I adopted a corgi named Biscuit last spring.",
                    source="s0", scope=scope, consolidate_now=False)
    report = eng.consolidate_pending(scope=scope, score_importance=False)
    assert report["claims_extracted"] > 0
    assert eng.store.active_claims_at(2_000_000_000, scope)


def test_affect_salience_scores_during_consolidation(fresh_settings):
    eng = _engine(fresh_settings, affect_salience_enabled=True)
    scope = Scope(namespace="affect-sleep")
    eng.ingest_text("user: I finally finished the marathon and I cried at the finish line!",
                    source="s0", scope=scope, consolidate_now=False)
    before = eng.store.active_records_at(2_000_000_000, scope)[0].salience
    eng.consolidate_pending(scope=scope, score_importance=False)
    rec = eng.store.active_records_at(2_000_000_000, scope)[0]
    assert eng.client.affect_calls == 1
    assert "arousal" in rec.metadata
    assert rec.salience != before  # affect reshaped the static salience
    assert 0.0 <= rec.salience <= 1.0


def test_affect_salience_off_skips_affect_call(fresh_settings):
    eng = _engine(fresh_settings, affect_salience_enabled=False)
    scope = Scope(namespace="affect-off-sleep")
    eng.ingest_text("user: I finally finished the marathon and I cried at the finish line!",
                    source="s0", scope=scope, consolidate_now=False)
    eng.consolidate_pending(scope=scope, score_importance=False)
    rec = eng.store.active_records_at(2_000_000_000, scope)[0]
    assert eng.client.affect_calls == 0
    assert "arousal" not in rec.metadata
