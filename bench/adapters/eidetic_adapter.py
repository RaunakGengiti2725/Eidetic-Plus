"""Eidetic-Plus adapter: wraps the real Engine. No new logic -- the same engine the API
and MCP server use. Write path is the LLM-free fast path (consolidate_now=False); the
graph/facts are built by consolidate_pending() between ingest and query, off the hot path.
"""
from __future__ import annotations

import time
from typing import Optional

from eidetic.engine import Engine
from eidetic.models import Scope, now

from ..reader import answer_with_fixed_reader
from .base import AnswerResult, MemorySystem, WriteResult, approx_tokens


class EideticSystem(MemorySystem):
    name = "eidetic-plus"

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or Engine()

    def reset(self, namespace: str) -> None:
        # Scope isolates conversations, so no global clear is needed; just drop the cache.
        self.engine.cache.clear()

    def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                       session_time: Optional[float] = None) -> WriteResult:
        # One memory PER SESSION (joined turns) -- session granularity, not per-turn. This
        # cuts ingest+consolidation cost ~15x on a ~300-turn LoCoMo conversation while keeping
        # assistant turns first-class. The LLM-free fast write only embeds (no LLM here).
        scope = Scope(namespace=namespace)
        valid_at = session_time if session_time is not None else now()
        text = "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}".strip()
            for t in turns if (t.get("content") or "").strip()
        ).strip()
        if not text:
            return WriteResult(tokens=0, ms=0.0)
        t0 = time.perf_counter()
        self.engine.ingest_text(text, source=session_id, valid_at=valid_at,
                                scope=scope, consolidate_now=False)  # LLM-free write
        return WriteResult(tokens=approx_tokens(text), ms=(time.perf_counter() - t0) * 1000.0)

    def consolidate(self, namespace: str) -> None:
        # Async build: parallel fact extraction, bi-temporal graph, events, date normalization,
        # typed preferences. score_importance=False (importance isn't in the ranking path).
        self.engine.consolidate_pending(scope=Scope(namespace=namespace), score_importance=False)
        # Dreaming-engine A/B hook (token-free): set DREAM_AB=1 to run one idle consolidation
        # pass (replay + inferred link prediction + multi-resolution gist) so the dreaming
        # layer can be measured dream-on vs dream-off against the scoreboard. Off by default.
        import os
        if os.environ.get("DREAM_AB", "0") in ("1", "true", "yes"):
            self.engine.dream(scope=Scope(namespace=namespace))

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        scope = Scope(namespace=namespace)
        # Time retrieval (search) separately from the full answer (e2e).
        t0 = time.perf_counter()
        cands = self.engine.retriever.retrieve(question, at=as_of, scope=scope)
        search_ms = (time.perf_counter() - t0) * 1000.0
        # NEUTRALITY: answer with the SAME fixed reader (model + prompt) the baselines use.
        # But assemble context the Eidetic way (event calendar + surfaced preferences +
        # edge-placement + compression) -- this is RETRIEVAL/CONTEXT-ASSEMBLY, the legitimate
        # "memory quality" edge, and is the SAME method answer() uses, so upgrades #1 (event
        # calendar) and #2 (typed preferences) actually reach the scoreboard. The cascade /
        # NLI verification / abstention stay OUT of this neutral path (product features).
        blocks = self.engine.retriever.assemble_context(question, cands, at=as_of, scope=scope)
        text = answer_with_fixed_reader(question, blocks)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        ctx_tokens = sum(approx_tokens(b) for b in blocks)
        coverage = max((c.dense_score for c in cands), default=0.0)  # abstention calibration signal
        return AnswerResult(
            answer=text, context_tokens=ctx_tokens,
            search_ms=search_ms, e2e_ms=e2e_ms, abstained=False,
            extra={"citations": len(cands), "coverage": coverage},
        )
