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


def test_claim_extraction_flag_off_makes_no_claim_calls(fresh_settings):
    """CLAIM_EXTRACTION=0 must not PAY for claim extraction: the calls fired anyway and the
    results were discarded at the write - half the extraction spend bought nothing, and
    ablation cost accounting lied about what the flag saves."""
    eng = _engine(fresh_settings, claim_extraction_enabled=False)
    calls = {"claims": 0, "edges": 0}
    client = eng.client

    def counting_claims(text):
        calls["claims"] += 1
        return []

    def counting_claims_bounded(text, *, max_windows=0):
        calls["claims"] += 1
        return []

    def counting_edges(text):
        calls["edges"] += 1
        return []

    def counting_edges_bounded(text, *, max_windows=0):
        calls["edges"] += 1
        return []

    client.extract_claims = counting_claims
    client.extract_claims_bounded = counting_claims_bounded
    client.extract_edges = counting_edges
    client.extract_edges_bounded = counting_edges_bounded

    scope = Scope(namespace="claims-cost-off")
    eng.ingest_text("user: I adopted a corgi named Biscuit last spring.",
                    source="s0", scope=scope, consolidate_now=False)
    eng.consolidate_pending(scope=scope, score_importance=False)
    assert calls["claims"] == 0            # flag off -> zero claim-extraction spend
    assert calls["edges"] >= 1             # edge extraction unaffected

    eng2 = _engine(fresh_settings, claim_extraction_enabled=True)
    calls2 = {"claims": 0}
    eng2.client.extract_claims = lambda text: (calls2.__setitem__("claims", calls2["claims"] + 1) or [])
    eng2.client.extract_claims_bounded = (
        lambda text, *, max_windows=0: (calls2.__setitem__("claims", calls2["claims"] + 1) or []))
    scope2 = Scope(namespace="claims-cost-on")
    eng2.ingest_text("user: I adopted a corgi named Biscuit last spring.",
                     source="s0", scope=scope2, consolidate_now=False)
    eng2.consolidate_pending(scope=scope2, score_importance=False)
    assert calls2["claims"] >= 1           # flag on -> claim extraction still runs
