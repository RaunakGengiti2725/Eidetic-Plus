"""Eidetic-Plus adapter: wraps the real Engine. No new logic -- the same engine the API
and MCP server use. Write path is the LLM-free fast path (consolidate_now=False); the
graph/facts are built by consolidate_pending() between ingest and query, off the hot path.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from eidetic.engine import Engine
from eidetic.models import Scope, now

from ..reader import answer_with_fixed_reader
from .base import AnswerResult, MemorySystem, WriteResult, approx_tokens


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


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
        lines = [f"{t.get('role', 'user')}: {t.get('content', '')}".strip()
                 for t in turns if (t.get("content") or "").strip()]
        text = "\n".join(lines).strip()
        if not text:
            return WriteResult(tokens=0, ms=0.0)
        # INGEST_GRANULARITY: session (default, byte-identical to the historical write) | turn
        # (one record per non-empty turn, ~15x cost) | hybrid (session record PLUS short turn-window
        # records for sharper embeddings, so a buried fact is also retrievable at finer grain). All
        # records carry the SAME session valid_at, so the bi-temporal anchor is unchanged.
        granularity = os.environ.get("INGEST_GRANULARITY", "session").strip().lower()
        t0 = time.perf_counter()
        if granularity == "turn":
            for i, line in enumerate(lines):
                self.engine.ingest_text(line, source=f"{session_id}#t{i}", valid_at=valid_at,
                                        scope=scope, consolidate_now=False)
        elif granularity == "hybrid":
            self.engine.ingest_text(text, source=session_id, valid_at=valid_at,
                                    scope=scope, consolidate_now=False)
            win = max(1, int(os.environ.get("INGEST_WINDOW_TURNS", "5")))
            for j in range(0, len(lines), win):
                chunk = "\n".join(lines[j:j + win]).strip()
                if chunk and chunk != text:
                    self.engine.ingest_text(chunk, source=f"{session_id}#w{j // win}",
                                            valid_at=valid_at, scope=scope, consolidate_now=False)
        else:  # "session" (default) and any unknown value -> the original single-record write
            self.engine.ingest_text(text, source=session_id, valid_at=valid_at,
                                    scope=scope, consolidate_now=False)  # LLM-free write
        return WriteResult(tokens=approx_tokens(text), ms=(time.perf_counter() - t0) * 1000.0)

    def consolidate(self, namespace: str) -> None:
        # Async build: parallel fact extraction, bi-temporal graph, events, date normalization,
        # typed preferences. score_importance=False (importance isn't in the ranking path).
        scope = Scope(namespace=namespace)
        # FULL_SLEEP=1: run the unified lifecycle sleep (consolidate_pending -> dream replay+infer
        # +multi-resolution gist), so the dream/gist channels have output to read. Token-free
        # (llm_summaries=False). Default path stays consolidate_pending-only + the DREAM_AB hook,
        # byte-identical to the historical bench build.
        if _truthy("FULL_SLEEP"):
            self.engine.sleep(scope=scope, llm_summaries=False)
            return
        self.engine.consolidate_pending(scope=scope, score_importance=False)
        # Dreaming-engine A/B hook (token-free): set DREAM_AB=1 to run one idle consolidation
        # pass (replay + inferred link prediction + multi-resolution gist) so the dreaming
        # layer can be measured dream-on vs dream-off against the scoreboard. Off by default.
        if _truthy("DREAM_AB"):
            self.engine.dream(scope=scope)

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


class EideticFullSystem(EideticSystem):
    """The PRODUCT row. Same write/consolidate path as the neutral eidetic-plus row, but the
    answer applies the product policy: NLI verification + abstention + proof (engine.retriever's
    full answer()), so the scoreboard sees the honesty differentiators -- verified recall with a
    citable immutable source, and an explicit abstention when evidence is insufficient -- that no
    baseline has. Reports verified/abstained/confidence in `extra` for the report's product metrics.

    (Retrieval is the same retrieve() the neutral row uses, so search_ms/context_tokens stay
    cleanly comparable; the cache/reflex/reconsolidation WRAPPERS in engine.ask() are latency
    features that do not change this accuracy/honesty comparison.)
    """

    name = "eidetic-plus-full"

    _ABSTAIN_TEXT = "I don't have enough verified evidence in memory to answer that confidently."

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        scope = Scope(namespace=namespace)
        r = self.engine.retriever
        s = self.engine.settings
        t0 = time.perf_counter()
        cands = r.retrieve(question, at=as_of, scope=scope)
        search_ms = (time.perf_counter() - t0) * 1000.0
        blocks = r.assemble_context(question, cands, at=as_of, scope=scope)
        # NEUTRALITY: generate through the SAME fixed reader (model + prompt) as every baseline, so
        # the scoreboard measures memory, not answerer. The product policy -- NLI verification +
        # abstention + proof -- is then layered on THAT answer (the honesty differentiator), not on
        # a stronger private reader. (engine.ask()'s own reader/cascade is a separate latency/quality
        # feature, deliberately kept OUT of the neutral accuracy comparison.)
        text = answer_with_fixed_reader(question, blocks)
        citations, entailed = r._verify_candidates(cands, text, True)
        verified = entailed > 0
        coverage = max((c.dense_score for c in cands), default=0.0)
        if s.abstention_v2_enabled:
            conf, _sig = r._abstention_confidence(cands, citations)
            abstained = conf < s.abstention_v2_tau
        else:
            abstained = (not verified) and coverage < s.abstention_threshold
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        return AnswerResult(
            answer=(self._ABSTAIN_TEXT if abstained else text),
            context_tokens=sum(approx_tokens(b) for b in blocks),
            search_ms=search_ms, e2e_ms=e2e_ms, abstained=abstained,
            extra={"verified": bool(verified and not abstained), "coverage": coverage,
                   "citations": len(citations), "policy": "fixed-reader + verify+abstain+proof"},
        )


class EideticProductSystem(EideticSystem):
    """The PRODUCT-CEILING row: the full engine.ask() path users actually run -- semantic cache,
    reflex local recall, flow activation, the difficulty cascade (qwen-flash -> qwen3-max on a
    grounding miss), NLI verify + abstention + proof, and post-answer coactivation/reconsolidation.
    Unlike the two neutral rows it does NOT pin the shared fixed reader: the cascade picks the
    answerer, so this measures the deployed product's ceiling, not the memory-only comparison.
    The delta (product minus eidetic-plus-full) is the cascade+reflex+flow value.

    Bundle to exercise it: REFLEX_RECALL=1 FLOW_ACTIVATION=1 SPECULATIVE_CASCADE=1
    SEMANTIC_CACHE=1 COVE=1 ACTIVE_RETRIEVAL=1 (plus the promoted graph/coact defaults).
    """

    name = "eidetic-product"

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        scope = Scope(namespace=namespace)
        t0 = time.perf_counter()
        ans = self.engine.ask(question, at=as_of, scope=scope, verify=True)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        note = ans.note or ""
        abstained = note.startswith("abstained")
        # engine.ask() bundles retrieval into the answer, so search_ms isn't separately timed here;
        # report e2e and let the cost/latency tables carry the cascade/cache effect. context_tokens
        # come from the cited sources (proof surface), approximated from citation snippets.
        ctx_tokens = sum(approx_tokens(getattr(c, "snippet", "") or "") for c in ans.citations)
        return AnswerResult(
            answer=ans.answer, context_tokens=ctx_tokens,
            search_ms=0.0, e2e_ms=e2e_ms, abstained=abstained,
            extra={"verified": bool(ans.verified and not abstained),
                   "confidence": ans.confidence, "citations": len(ans.citations),
                   "note": note, "policy": "engine.ask: cache+reflex+flow+cascade+verify+abstain"},
        )
