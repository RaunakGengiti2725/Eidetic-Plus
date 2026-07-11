from __future__ import annotations

import numpy as np

from eidetic.semantic_cache import SemanticCache


def test_semantic_cache_exact_and_scope_isolation():
    cache = SemanticCache(cosine_threshold=0.9)
    qvec = np.array([1.0, 0.0], dtype=np.float32)
    cache.put("scope-a", "Where is Alice?", qvec, "Paris")

    assert cache.get("scope-a", "where is alice?", None) == "Paris"
    assert cache.get("scope-b", "Where is Alice?", qvec) is None


def test_semantic_cache_adaptive_thresholds_are_query_sensitive():
    cache = SemanticCache(cosine_threshold=0.9, adaptive=True)
    short = cache.threshold_for("Alice?")
    temporal = cache.threshold_for("Where does Alice work now?")
    long = cache.threshold_for(
        "What city did Alice say she would move to after the finance conference in May?"
    )

    assert short > cache.threshold
    assert temporal > cache.threshold
    assert long < temporal
    assert 0.80 <= long <= 0.98


def test_semantic_cache_near_hit_uses_adaptive_threshold():
    cache = SemanticCache(cosine_threshold=0.9, adaptive=True)
    cache.put("s", "Alice moved to Paris after the finance conference in May.", np.array([1.0, 0.0]), "Paris")

    # Cosine 0.91 is enough for a long specific query after adaptive relaxation.
    assert cache.get(
        "s",
        "What city did Alice say she would move to after the finance conference in May?",
        np.array([0.91, np.sqrt(1 - 0.91 ** 2)], dtype=np.float32),
    ) == "Paris"
    # The same similarity is not enough for a short ambiguous query.
    assert cache.get("s", "Alice?", np.array([0.91, np.sqrt(1 - 0.91 ** 2)], dtype=np.float32)) is None


def test_engine_respects_semantic_cache_flag(fresh_settings):
    from dataclasses import replace

    from eidetic.engine import Engine
    from eidetic.models import Answer, Scope

    class FakeClient:
        def __init__(self):
            self.embed_calls = 0

        def embed_text(self, _text):
            self.embed_calls += 1
            return np.array([1.0, 0.0], dtype=np.float32)

    class FakeRetriever:
        def answer(self, query, at=None, verify=True, scope=None, qvec=None, reader_model=None):
            return Answer(question=query, answer="ok", generated_by="fake", verified=False)

    settings = replace(fresh_settings, semantic_cache_enabled=False)
    client = FakeClient()
    engine = Engine(settings=settings, client=client)
    engine.retriever = FakeRetriever()
    answer = engine.ask("cached?", scope=Scope(namespace="cache-test"))
    assert answer.status.value == "ABSTAINED"
    assert answer.verified is False
    assert answer.citations == []
    assert answer.answer != "ok"
    assert client.embed_calls == 0
