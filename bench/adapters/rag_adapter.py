"""RAG baselines (Track 5.1) -- the "what RAG actually is" controls Eidetic must beat.

Two flavors, both conforming to the neutral MemorySystem contract (same ingest loop, same ONE
fixed reader, same judge), so the only thing that varies vs Eidetic is the MEMORY model:

  * rag-full   -- no retrieval at all: stuff the whole (time-available) conversation history
                  into the fixed reader's context. The "just give the LLM everything" baseline.
  * rag-vector -- classic chunk -> embed -> top-k cosine retrieval. NO knowledge graph, NO
                  bi-temporal supersession, NO proof ledger, NO abstention. Plain vector RAG.

Both are REAL (rag-vector embeds with the same DashScope text-embedding-v4 every system uses);
neither fabricates. The embedding client is injectable so the adapter is unit-testable offline.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from eidetic.models import now

from ..reader import answer_with_fixed_reader
from .base import AnswerResult, MemorySystem, WriteResult, approx_tokens


def _join_turns(turns: list[dict]) -> str:
    return "\n".join(
        f"{(t.get('role') or 'user')}: {t.get('content', '')}".strip()
        for t in turns if (t.get("content") or "").strip()
    ).strip()


def _available(session_time: Optional[float], as_of: Optional[float]) -> bool:
    """A memory is available at query time if it has no timestamp or it precedes `as_of`.
    Faithful to "what the system could have known when the question was asked"."""
    return as_of is None or session_time is None or session_time <= as_of


class RagFullSystem(MemorySystem):
    """Full-context baseline: no retrieval, stuff all time-available history into the reader."""

    name = "rag-full"

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[Optional[float], str]]] = {}

    def reset(self, namespace: str) -> None:
        self._store[namespace] = []

    def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                       session_time: Optional[float] = None) -> WriteResult:
        text = _join_turns(turns)
        if not text:
            return WriteResult(tokens=0, ms=0.0)
        self._store.setdefault(namespace, []).append((session_time, text))
        return WriteResult(tokens=approx_tokens(text), ms=0.0)   # cost is paid at QUERY time

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        t0 = time.perf_counter()
        blocks = [text for (st, text) in self._store.get(namespace, []) if _available(st, as_of)]
        search_ms = (time.perf_counter() - t0) * 1000.0      # ~0: there is no retrieval step
        text = answer_with_fixed_reader(question, blocks)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        return AnswerResult(answer=text, context_tokens=sum(approx_tokens(b) for b in blocks),
                            search_ms=search_ms, e2e_ms=e2e_ms, abstained=False,
                            extra={"sessions_in_context": len(blocks), "rag": "full-context"})


class RagVectorSystem(MemorySystem):
    """Classic vector RAG: chunk -> embed -> top-k cosine. No graph / supersession / proof."""

    name = "rag-vector"

    def __init__(self, client=None, top_k: int = 10, chunk_chars: int = 800,
                 overlap: int = 100) -> None:
        self._client = client
        self.top_k = max(1, int(top_k))
        self.chunk_chars = max(1, int(chunk_chars))
        self.overlap = max(0, int(overlap))
        self._chunks: dict[str, list[dict]] = {}

    def _embed(self):
        if self._client is None:
            from eidetic.dashscope_client import get_client
            self._client = get_client()
        return self._client

    def _chunk(self, text: str) -> list[str]:
        step = max(1, self.chunk_chars - self.overlap)
        return [text[i:i + self.chunk_chars] for i in range(0, len(text), step)] or [text]

    def reset(self, namespace: str) -> None:
        self._chunks[namespace] = []

    def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                       session_time: Optional[float] = None) -> WriteResult:
        text = _join_turns(turns)
        if not text:
            return WriteResult(tokens=0, ms=0.0)
        t0 = time.perf_counter()
        chunks = self._chunk(text)
        vecs = self._embed().embed_texts(chunks)             # real embeddings (write cost)
        bucket = self._chunks.setdefault(namespace, [])
        st = session_time if session_time is not None else now()
        for ch, v in zip(chunks, vecs):
            bucket.append({"text": ch, "vec": np.asarray(v, dtype=np.float32), "st": st})
        return WriteResult(tokens=approx_tokens(text), ms=(time.perf_counter() - t0) * 1000.0)

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        t0 = time.perf_counter()
        qv = np.asarray(self._embed().embed_text(question), dtype=np.float32)
        cands = [c for c in self._chunks.get(namespace, []) if _available(c["st"], as_of)]
        scored = sorted(cands, key=lambda c: -float(qv @ c["vec"]))[: self.top_k]
        search_ms = (time.perf_counter() - t0) * 1000.0
        blocks = [c["text"] for c in scored]
        text = answer_with_fixed_reader(question, blocks)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        return AnswerResult(answer=text, context_tokens=sum(approx_tokens(b) for b in blocks),
                            search_ms=search_ms, e2e_ms=e2e_ms, abstained=False,
                            extra={"hits": len(blocks), "rag": "vector-only (no graph/supersession/proof)"})
