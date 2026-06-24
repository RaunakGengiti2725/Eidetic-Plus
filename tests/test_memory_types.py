"""Offline tests for MIRIX role typing + Markov prefetch wiring (no key)."""
from __future__ import annotations

from eidetic.memory_types import MemoryType, classify_memory_type, type_priority


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
