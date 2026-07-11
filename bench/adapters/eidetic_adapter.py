"""Eidetic-Plus adapter: wraps the real Engine. No new logic -- the same engine the API
and MCP server use. Write path is the LLM-free fast path (consolidate_now=False); the
graph/facts are built by consolidate_pending() between ingest and query, off the hot path.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from eidetic.engine import Engine
from eidetic.models import ABSTENTION_TEXT, AnswerStatus, NLILabel, Scope, now

from ..reader import answer_with_fixed_reader
from .base import AnswerResult, MemorySystem, WriteResult, approx_tokens


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


def _is_entailment(citation) -> bool:
    label = getattr(citation, "nli_label", "")
    return getattr(label, "value", label) == "entailment"


def _entailed_memory_ids(citations) -> list[str]:
    return [
        str(c.memory_id)
        for c in citations
        if _is_entailment(c) and getattr(c, "memory_id", "")
    ]


def _entailed_content_hashes(citations) -> list[str]:
    return list(dict.fromkeys(
        str(c.content_hash)
        for c in citations
        if _is_entailment(c) and getattr(c, "content_hash", "")
    ))


def _entailed_raw_uris(citations) -> list[str]:
    return list(dict.fromkeys(
        str(c.raw_uri)
        for c in citations
        if _is_entailment(c) and getattr(c, "raw_uri", "")
    ))


def _proof_surface_tokens(citations) -> int:
    return sum(
        approx_tokens(getattr(c, "snippet", "") or "")
        for c in citations
        if _is_entailment(c)
    )


def _smqe_extra(note: str) -> dict:
    parts = (note or "").split(":")
    if len(parts) >= 3 and parts[0] == "smqe":
        return {
            "structured_recall": True,
            "smqe_operator": parts[1],
            "smqe_backend": parts[2],
            "smqe_policy": note,
        }
    return {"structured_recall": False}


def _region_context_extra(retriever, question: str) -> dict:
    telemetry = getattr(retriever, "last_context_telemetry", {}) or {}
    if not isinstance(telemetry, dict) or telemetry.get("query") != question:
        return {"region_hint_count": 0, "region_ids": [], "region_member_ids": []}
    hints = telemetry.get("region_hints")
    if not isinstance(hints, list):
        hints = []
    member_ids: list[str] = []
    for hint in hints:
        if isinstance(hint, dict):
            member_ids.extend(str(mid) for mid in hint.get("members", []) if mid)
    return {
        "region_hint_count": int(telemetry.get("region_hint_count", 0) or 0),
        "region_ids": list(telemetry.get("region_ids", []) or []),
        "region_member_ids": list(dict.fromkeys(member_ids)),
    }


class EideticSystem(MemorySystem):
    name = "eidetic-plus"

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or Engine()

    def reset(self, namespace: str) -> None:
        clear = getattr(self.engine, "clear_namespace", None)
        if callable(clear):
            clear(namespace)
            return
        # Lightweight fake engines in tests may expose only the answer cache.
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
        # FAIRNESS GUARDRAIL: hybrid/turn change ONLY the eidetic write path (the baseline adapters
        # ignore this env), and hybrid intentionally double-ingests (session + windows), which adds
        # duplicate graph facts and enlarges eidetic's candidate pool. Use these for eidetic-vs-
        # eidetic ablations ONLY; do NOT enable them in a cross-system reported run, or the
        # comparison is no longer apples-to-apples.
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

    def consolidate(self, namespace: str) -> dict:
        # Async build: parallel fact extraction, bi-temporal graph, events, date normalization,
        # typed preferences. score_importance=False (importance isn't in the ranking path).
        scope = Scope(namespace=namespace)
        # Real model-spend accounting: the write_tokens column counts ingested CONTENT volume,
        # not model calls - extraction savings were invisible to it. The API's own usage
        # numbers, deltaed around consolidation, land in the report (stored per row under
        # extra['consolidate']) so write-cost claims measure dollars-shaped tokens.
        _usage_before = None
        _snap = getattr(self.engine.client, "usage_snapshot", None)
        if callable(_snap):
            _usage_before = _snap()

        def _with_usage(report: dict) -> dict:
            if _usage_before is not None:
                delta_fn = getattr(self.engine.client, "usage_delta", None)
                if callable(delta_fn):
                    report["model_usage"] = delta_fn(_usage_before, _snap())
            return report
        # FULL_SLEEP=1: consolidate_pending + dream (replay + inferred links + multi-resolution
        # gist), so the dream/gist channels have output to read. This is exactly the default build
        # PLUS the DREAM_AB hook -- a true superset, and token-free (score_importance=False keeps
        # the per-record qwen-flash importance call off, matching the default path). Default path
        # stays consolidate_pending-only + the optional DREAM_AB hook, byte-identical to before.
        if _truthy("FULL_SLEEP"):
            pending = self.engine.consolidate_pending(scope=scope, score_importance=False)
            dream = self.engine.dream(scope=scope)
            return _with_usage({"consolidate_pending": pending, "dream": dream})
        pending = self.engine.consolidate_pending(scope=scope, score_importance=False)
        out = {"consolidate_pending": pending}
        # Dreaming-engine A/B hook (token-free): set DREAM_AB=1 to run one idle consolidation
        # pass (replay + inferred link prediction + multi-resolution gist) so the dreaming
        # layer can be measured dream-on vs dream-off against the scoreboard. Off by default.
        if _truthy("DREAM_AB"):
            out["dream"] = self.engine.dream(scope=scope)
        return _with_usage(out)

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
            extra={"output_type": "UNTRUSTED_DRAFT", "status": "DRAFT",
                   "verified": False, "policy": "retrieval-diagnostic-fixed-reader",
                   "citations": len(cands), "coverage": coverage,
                   "candidate_memory_ids": [c.record.memory_id for c in cands],
                   **_region_context_extra(self.engine.retriever, question)},
        )

    def after_answer(self, namespace: str, question: str, result: AnswerResult, *,
                     correct: Optional[bool] = None,
                     as_of: Optional[float] = None) -> dict:
        # BENCH_COACTIVATION gives the neutral benchmark the same pure graph write that ask()
        # gets through verified reconsolidation. It never reads the judge label; `correct` is
        # accepted only because the generic harness passes it to every system hook.
        if not self.engine.settings.bench_coactivation_enabled or result.abstained:
            return {}
        ids = (result.extra or {}).get("candidate_memory_ids") or []
        return self.engine.link_coactivated(ids[:5], scope=Scope(namespace=namespace), valid_at=as_of)


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

    _ABSTAIN_TEXT = ABSTENTION_TEXT

    def after_answer(self, namespace: str, question: str, result: AnswerResult, *,
                     correct: Optional[bool] = None,
                     as_of: Optional[float] = None) -> dict:
        return {}

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        scope = Scope(namespace=namespace)
        r = self.engine.retriever
        t0 = time.perf_counter()
        ans = self.engine.ask(
            question,
            at=as_of,
            scope=scope,
            verify=True,
            use_cache=False,
            reader_model="bench-fixed-reader",
            reader=answer_with_fixed_reader,
        )
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        note = ans.note or ""
        abstained = ans.status == AnswerStatus.ABSTAINED
        structured_recall = ans.generated_by == "smqe" or note.startswith("smqe:")
        if structured_recall:
            search_ms = e2e_ms
            context_tokens = sum(approx_tokens(c.snippet) for c in ans.citations)
            candidate_memory_ids = [c.memory_id for c in ans.citations]
            region_extra = {"region_hint_count": 0, "region_ids": [], "region_member_ids": []}
        else:
            telemetry = r.last_answer_telemetry
            if (telemetry.get("query") != question
                    or telemetry.get("scope_key") != scope.key()):
                telemetry = {}
            search_ms = float(telemetry.get("search_ms", 0.0) or 0.0)
            blocks = telemetry.get("blocks") or []
            context_tokens = sum(approx_tokens(block) for block in blocks)
            candidate_memory_ids = list(telemetry.get("candidate_memory_ids") or [])
            region_extra = _region_context_extra(r, question)
        return AnswerResult(
            answer=ans.answer,
            context_tokens=context_tokens,
            search_ms=search_ms,
            e2e_ms=e2e_ms,
            abstained=abstained,
            extra={
                "status": ans.status.value,
                "verified": ans.status == AnswerStatus.VERIFIED,
                "confidence": ans.confidence,
                "citations": len(ans.citations),
                "candidate_memory_ids": candidate_memory_ids,
                "entailed_memory_ids": _entailed_memory_ids(ans.citations),
                "entailed_content_hashes": _entailed_content_hashes(ans.citations),
                "entailed_raw_uris": _entailed_raw_uris(ans.citations),
                "proof_surface_tokens": _proof_surface_tokens(ans.citations),
                "note": note,
                "policy": note if structured_recall else "fixed-reader + verify+abstain+proof",
                "orchestration": "engine.ask:fixed-reader+canonical-proof",
                **region_extra,
                **_smqe_extra(note),
            },
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

    def after_answer(self, namespace: str, question: str, result: AnswerResult, *,
                     correct: Optional[bool] = None,
                     as_of: Optional[float] = None) -> dict:
        # engine.ask() already performs verified reconsolidation and co-activation. The benchmark
        # hook is only for rows that bypass ask().
        return {}

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        scope = Scope(namespace=namespace)
        r = self.engine.retriever
        t0 = time.perf_counter()
        ans = self.engine.ask(question, at=as_of, scope=scope, verify=True)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        note = ans.note or ""
        abstained = ans.status == AnswerStatus.ABSTAINED
        structured_recall = ans.generated_by == "smqe" or note.startswith("smqe:")
        if structured_recall:
            cands = []
            search_ms = e2e_ms
            ctx_tokens = sum(
                approx_tokens(getattr(c, "snippet", "") or "")
                for c in ans.citations
                if c.nli_label == NLILabel.ENTAILMENT
            )
            candidate_memory_ids = [c.memory_id for c in ans.citations]
        else:
            telemetry = r.last_answer_telemetry
            if (telemetry.get("query") != question
                    or telemetry.get("scope_key") != scope.key()):
                telemetry = {}
            cands = []
            search_ms = float(telemetry.get("search_ms", 0.0) or 0.0)
            blocks = telemetry.get("blocks") or []
            ctx_tokens = sum(approx_tokens(block) for block in blocks)
            candidate_memory_ids = list(telemetry.get("candidate_memory_ids") or [])
        return AnswerResult(
            answer=ans.answer, context_tokens=ctx_tokens,
            search_ms=search_ms, e2e_ms=e2e_ms, abstained=abstained,
            extra={"status": ans.status.value,
                   "verified": ans.status == AnswerStatus.VERIFIED,
                   "confidence": ans.confidence, "citations": len(ans.citations),
                   "candidate_memory_ids": candidate_memory_ids,
                   "entailed_memory_ids": _entailed_memory_ids(ans.citations),
                   "entailed_content_hashes": _entailed_content_hashes(ans.citations),
                   "entailed_raw_uris": _entailed_raw_uris(ans.citations),
                   "proof_surface_tokens": sum(
                       approx_tokens(getattr(c, "snippet", "") or "") for c in ans.citations),
                   "note": note,
                   "policy": "engine.ask: smqe+cache+reflex+flow+cascade+verify+abstain",
                   **(
                       {"region_hint_count": 0, "region_ids": [], "region_member_ids": []}
                       if structured_recall else _region_context_extra(r, question)
                   ),
                   **_smqe_extra(note)},
        )
