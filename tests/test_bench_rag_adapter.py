"""Offline tests for the RAG baseline adapters (Track 5.1). They conform to the same
MemorySystem contract as Eidetic/Mem0/Graphiti so the harness drives them identically.
The fixed reader is stubbed (echoes its prompt) so we can assert WHAT context each baseline
fed it -- no key needed. rag-vector's embeddings use an injected deterministic fake client."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from bench import reader as bench_reader
from bench.adapters.rag_adapter import RagFullSystem, RagVectorSystem


class _FakeEmbed:
    def __init__(self, dim=64):
        self.dim = dim

    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._e(t)

    def embed_texts(self, ts):
        return np.stack([self._e(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)


def _stub_reader(monkeypatch):
    """Reader that echoes the prompt (which embeds the context) so the test can read it back."""
    class _Echo:
        def chat(self, model, system, user, **kw):
            return user
    monkeypatch.setattr(bench_reader, "get_client", lambda: _Echo())


def test_rag_full_stuffs_all_sessions_into_context(monkeypatch):
    _stub_reader(monkeypatch)
    s = RagFullSystem()
    s.reset("ns")
    s.ingest_session("ns", "s0", [{"role": "user", "content": "Alice lives in Paris"}])
    s.ingest_session("ns", "s1", [{"role": "user", "content": "Bob lives in Rome"}])
    ar = s.answer("ns", "where does Alice live")
    assert "Paris" in ar.answer and "Rome" in ar.answer     # whole history stuffed in
    assert ar.context_tokens > 0


def test_rag_full_respects_as_of_availability(monkeypatch):
    _stub_reader(monkeypatch)
    s = RagFullSystem()
    s.reset("ns")
    s.ingest_session("ns", "s0", [{"role": "user", "content": "early fact about Paris"}], session_time=100.0)
    s.ingest_session("ns", "s1", [{"role": "user", "content": "later fact about Rome"}], session_time=500.0)
    ar = s.answer("ns", "what", as_of=200.0)                 # only sessions <= 200 are available
    assert "Paris" in ar.answer and "Rome" not in ar.answer


def test_rag_vector_retrieves_topk_only(monkeypatch):
    _stub_reader(monkeypatch)
    s = RagVectorSystem(client=_FakeEmbed(), top_k=1)
    s.reset("ns")
    s.ingest_session("ns", "s0", [{"role": "user", "content": "Alice lives in Paris"}])
    s.ingest_session("ns", "s1", [{"role": "user", "content": "quantum chromodynamics lattice gauge"}])
    ar = s.answer("ns", "where does Alice live in Paris")
    assert "Paris" in ar.answer                              # the relevant chunk was retrieved
    assert "quantum" not in ar.answer                        # top-1 excluded the irrelevant chunk
    assert ar.search_ms >= 0.0
    assert ar.extra.get("hits") == 1


def test_rag_systems_are_scope_isolated(monkeypatch):
    _stub_reader(monkeypatch)
    s = RagVectorSystem(client=_FakeEmbed())
    s.reset("A")
    s.reset("B")
    s.ingest_session("A", "s", [{"role": "user", "content": "the launch value is alpha-seven"}])
    ar = s.answer("B", "what is the launch value")
    # the unique stored token must not leak into another namespace's context (the question itself
    # never contains it, so its presence would prove a cross-scope leak).
    assert "alpha-seven" not in ar.answer


def test_rag_adapters_wired_into_make_system():
    from bench.run import make_system
    assert make_system("rag-full").name == "rag-full"
    assert make_system("rag-vector").name == "rag-vector"
