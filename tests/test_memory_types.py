"""Offline tests for MIRIX role typing + Markov prefetch wiring (no key)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.memory_types import MemoryType, classify_memory_type, type_priority
from eidetic.models import BrainEventType, MemoryRecord, Scope


class _VectorClient:
    def __init__(self, dim: int):
        self.dim = dim

    def _e(self, text: str):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, text: str):
        return self._e(text)

    def embed_texts(self, texts: list[str]):
        return np.stack([self._e(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def rerank(self, *_args, **_kwargs):
        raise AssertionError("Markov warm-up should not call the cross-encoder reranker")


def _warmup_engine(fresh_settings, *, rerank_enabled: bool = False, **overrides):
    from eidetic.engine import Engine

    settings = replace(
        fresh_settings,
        markov_prefetch_enabled=True,
        flow_warmup_enabled=True,
        flow_warmup_topk=1,
        brain_events_enabled=True,
        rerank_enabled=rerank_enabled,
        **overrides,
    )
    engine = Engine(settings, client=_VectorClient(fresh_settings.embed_dim))
    scope = Scope(namespace="markov-warmup")
    rec = MemoryRecord(
        memory_id="bob", content_hash="bob", text="Bob works at Acme",
        scope=scope, valid_at=1.0, entities=["Bob", "Acme"])
    engine.store.upsert_record(rec)
    engine.index.add(rec.memory_id, engine.client.embed_text(rec.text))
    return engine, scope


def test_resource_by_modality():
    assert classify_memory_type("a scanned invoice", modality="pdf") == MemoryType.RESOURCE
    assert classify_memory_type("photo of a whiteboard", modality="image") == MemoryType.RESOURCE


def test_knowledge_vault_for_sensitive():
    assert classify_memory_type("my api key is sk-abc123") == MemoryType.KNOWLEDGE_VAULT
    assert classify_memory_type("account number 12345678 routing 9999") == MemoryType.KNOWLEDGE_VAULT


def test_core_for_preferences():
    assert classify_memory_type("I prefer window seats") == MemoryType.CORE
    assert classify_memory_type("My favorite color is teal") == MemoryType.CORE


def test_procedural_for_howto():
    assert classify_memory_type("How to deploy the service to prod") == MemoryType.PROCEDURAL
    assert classify_memory_type("Step 1 install deps; step 2 build") == MemoryType.PROCEDURAL


def test_semantic_vs_episodic():
    assert classify_memory_type("The capital of France is Paris", consolidated=True) == MemoryType.SEMANTIC
    assert classify_memory_type("We met for coffee yesterday") == MemoryType.EPISODIC


def test_type_priority_ordering():
    assert type_priority("preference")[0] == MemoryType.CORE
    assert type_priority("procedural")[0] == MemoryType.PROCEDURAL
    assert type_priority("unknown-class") == type_priority("default")


# ---- Markov prefetch wiring (deterministic; no API) --------------------------
def test_query_signature_is_deterministic(engine):
    # same input -> same signature (a stable Markov state), and always non-empty.
    a = engine._query_signature("where does bob work")
    b = engine._query_signature("where does bob work")
    assert a == b and a


def test_engine_observes_query_transitions(engine):
    # feed an A,B,A,B,A,C signature sequence via repeated queries; the model learns A->B.
    seq = ["zzaa one", "zzbb two", "zzaa one", "zzbb two", "zzaa one", "zzcc three"]
    for q in seq:
        engine._observe_query(q)
    sig_a = engine._query_signature("zzaa one")
    pred = engine.markov.predict(sig_a, top_k=2)
    assert pred and pred[0][0] == engine._query_signature("zzbb two")   # A -> B is most likely


def test_markov_warmup_prefetches_predicted_signature(fresh_settings):
    engine, scope = _warmup_engine(fresh_settings)
    for q in ["alice question", "bob question", "alice question"]:
        engine._observe_query(q)

    report = engine.warmup_predicted_prefetch("alice question", scope=scope, at=2.0)

    assert report["warmed"] == 1
    assert report["signatures"] == ["bob"]
    blocks = engine.prefetch_context(engine.client.embed_text("bob"))
    assert blocks and "Bob works at Acme" in "\n".join(blocks)
    assert engine.brain_log.by_type(BrainEventType.FLOW_WARMED)


def test_idle_tick_runs_markov_prefetch_warmup(fresh_settings):
    engine, scope = _warmup_engine(fresh_settings)
    for q in ["alice question", "bob question", "alice question"]:
        engine._observe_query(q)

    report = engine.idle_tick(scope=scope)

    assert report["prefetch_warmup"]["warmed"] == 1
    assert "Bob works at Acme" in "\n".join(engine.prefetch_context(engine.client.embed_text("bob")))


def test_markov_warmup_skips_cross_encoder_rerank(fresh_settings):
    engine, scope = _warmup_engine(fresh_settings, rerank_enabled=True)
    for q in ["alice question", "bob question", "alice question"]:
        engine._observe_query(q)

    report = engine.warmup_predicted_prefetch("alice question", scope=scope, at=2.0)

    assert report["warmed"] == 1


def test_markov_warmup_uses_flow_activation_when_enabled(fresh_settings):
    engine, scope = _warmup_engine(
        fresh_settings,
        flow_activation_enabled=True,
        flow_hybrid_channel_enabled=True,
        flow_hybrid_weight=0.5,
    )
    quiet = MemoryRecord(
        memory_id="quiet", content_hash="quiet", text="Quiet field-warm memory",
        scope=scope, valid_at=1.0, entities=["Quiet"])
    engine.store.upsert_record(quiet)
    engine.index.add(quiet.memory_id, engine.client.embed_text(quiet.text))
    engine.activation.inject(scope.namespace, ["quiet"], 1.0)
    for q in ["alice question", "bob question", "alice question"]:
        engine._observe_query(q)

    report = engine.warmup_predicted_prefetch("alice question", scope=scope, at=2.0)

    assert report["warmed"] == 1
    blocks = engine.prefetch_context(engine.client.embed_text("bob"))
    joined = "\n".join(blocks or [])
    assert "Bob works at Acme" in joined
    assert "Quiet field-warm memory" in joined
