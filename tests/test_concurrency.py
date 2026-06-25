"""F0 concurrency safety: concurrent ingest + search + save on one Engine must not corrupt the
index, and the per-request RecallTrace must be thread-local. Runs fully offline with a fake embed
client (never skips on a missing key, so F0 is actually exercised)."""
from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

from eidetic.engine import Engine
from eidetic.models import RecallTrace, Scope


class _FakeEmbedClient:
    """Deterministic, no-network embeddings -- enough to exercise the ingest/search/save paths."""
    def __init__(self, dim: int):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, text):
        return self._embed(text)

    def embed_texts(self, texts):
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


def test_concurrent_ingest_search_save_no_corruption(fresh_settings):
    eng = Engine(fresh_settings, client=_FakeEmbedClient(fresh_settings.embed_dim))
    scope = Scope(namespace="stress")
    errors: list = []
    n_writers, per_writer = 5, 10

    def ingest_worker(i):
        try:
            for j in range(per_writer):
                eng.ingest_text(f"memory {i}-{j} about alice and bob and carol",
                                scope=scope, consolidate_now=False)
        except Exception as e:                      # any race -> corruption/exception -> fail
            errors.append(repr(e))

    def search_worker():
        try:
            rng = np.random.default_rng(0)
            for _ in range(40):
                eng.index.search(rng.standard_normal(eng.settings.embed_dim).astype(np.float32), 5)
        except Exception as e:
            errors.append(repr(e))

    threads = ([threading.Thread(target=ingest_worker, args=(i,)) for i in range(n_writers)]
               + [threading.Thread(target=search_worker) for _ in range(5)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    total = n_writers * per_writer
    assert len(eng.index) == total                  # every concurrent add landed
    assert eng.store.count(scope) == total
    # the persisted index reloads cleanly from disk (atomic saves -> no truncated/half-written file).
    from eidetic.vector_index import make_vector_index
    assert len(make_vector_index(fresh_settings)) == total


def test_recall_trace_is_thread_local(engine):
    results: dict = {}
    barrier = threading.Barrier(5)

    def worker(name):
        engine.retriever.last_trace = RecallTrace(query=name)
        barrier.wait()                              # force interleaving across threads
        results[name] = engine.retriever.last_trace.query

    threads = [threading.Thread(target=worker, args=(f"q{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # each thread saw ITS OWN trace, never another thread's (was last-writer-wins shared state).
    assert all(results[f"q{i}"] == f"q{i}" for i in range(5))
