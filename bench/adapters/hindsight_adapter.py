"""Hindsight baseline adapter -- REAL, never mocked.

Targets ``hindsight-all`` (Vectorize.io, MIT) via ``hindsight.HindsightEmbedded`` -- the
embedded pg0/pgvector backend, so NO external server is required. Hindsight is NOT a core
Eidetic-Plus dependency; it lives in ``requirements-bench.txt`` and is imported lazily
INSIDE ``__init__``. A missing package or an empty ``DASHSCOPE_API_KEY`` FAILS LOUD -- this
adapter never mocks or fabricates.

Neutrality (the whole point of the harness): Hindsight drives its OWN retrieval via
``recall`` (pure parallel retrieval + rerank, NOT ``reflect`` -- reflect is its own LLM
answerer and would break the single-fixed-reader guarantee). The recalled texts are handed
to the ONE shared fixed reader (``answer_with_fixed_reader``, qwen-plus), exactly as every
other baseline. So the scoreboard measures MEMORY quality, not answerer quality.

Write cost: ``retain`` runs Hindsight's own extraction LLM (pointed at DashScope's
OpenAI-compatible endpoint via litellm, same Qwen family as every other system). That is
genuine work; we time ingest end-to-end and account the ingested text with the
harness-uniform ``approx_tokens`` so writes compare across systems.

Bi-temporal: the benchmark's per-question ``as_of`` maps directly to Hindsight's
``recall(query_timestamp=...)``; each turn is retained with its own event timestamp, so
"answer as of T" replays the store's state at T.
"""
from __future__ import annotations

import importlib.util
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from eidetic.config import get_settings

from ..reader import answer_with_fixed_reader
from .base import AnswerResult, MemorySystem, WriteResult, approx_tokens

_LLM_MODEL = os.environ.get("BENCH_BASELINE_LLM", "qwen-flash")
_RECALL_MAX_TOKENS = int(os.environ.get("HINDSIGHT_RECALL_MAX_TOKENS", "4096"))
_RECALL_BUDGET = os.environ.get("HINDSIGHT_RECALL_BUDGET", "mid")


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


class HindsightSystem(MemorySystem):
    name = "hindsight"

    def __init__(self) -> None:
        if importlib.util.find_spec("hindsight") is None:
            raise RuntimeError(
                "hindsight is not installed. `.venv/bin/pip install hindsight-all` "
                "(MIT; in requirements-bench.txt). This adapter never mocks.")
        settings = get_settings()
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is empty -- Hindsight's retain() extraction LLM needs it "
                "(pointed at DashScope's OpenAI-compatible endpoint). No key => fail loud.")
        import hindsight as h  # lazy

        # litellm 'openai/<model>' + DashScope OpenAI-compatible base = same Qwen family as
        # every other baseline. Embedded pg0 backend: no external Postgres to provision.
        self._h = h.HindsightEmbedded(
            profile="eidetic-bench",
            llm_provider="openai",
            llm_api_key=key,
            llm_model=f"openai/{_LLM_MODEL}",
            llm_base_url=settings.compatible_base_url,
        )
        self._seen_banks: set[str] = set()

    def reset(self, namespace: str) -> None:
        # Banks are logical + isolated by bank_id; the harness already hands a unique
        # namespace per conversation (sys-dataset-gN-rM). Best-effort clear if the client
        # exposes one, else the fresh bank_id is the isolation boundary.
        for m in ("delete_bank", "clear_bank", "forget_bank"):
            fn = getattr(self._h, m, None)
            if callable(fn):
                try:
                    fn(namespace)
                except Exception:
                    pass
                break
        self._seen_banks.discard(namespace)

    def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                       session_time: Optional[float] = None) -> WriteResult:
        t0 = time.perf_counter()
        total_chars = 0
        for turn in turns:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            role = turn.get("role") or ""
            body = f"{role}: {content}" if role else content
            ts = turn.get("timestamp") or session_time
            self._h.retain(
                bank_id=namespace,
                content=body,
                timestamp=datetime.fromtimestamp(float(ts), tz=timezone.utc) if ts else None,
            )
            total_chars += len(body)
        self._seen_banks.add(namespace)
        return WriteResult(tokens=approx_tokens("x" * total_chars),
                           ms=(time.perf_counter() - t0) * 1000.0)

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        t0 = time.perf_counter()
        resp = self._h.recall(
            bank_id=namespace,
            query=question,
            max_tokens=_RECALL_MAX_TOKENS,
            budget=_RECALL_BUDGET,
            query_timestamp=_iso(as_of),
        )
        search_ms = (time.perf_counter() - t0) * 1000.0
        blocks = self._blocks(resp)
        text = answer_with_fixed_reader(question, blocks)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        return AnswerResult(
            answer=text,
            context_tokens=approx_tokens("\n\n".join(blocks)),
            search_ms=search_ms,
            e2e_ms=e2e_ms,
            abstained=False,
            extra={"hits": len(blocks), "recall_budget": _RECALL_BUDGET},
        )

    @staticmethod
    def _blocks(resp: Any) -> list[str]:
        """Extract recalled text defensively across client shapes (RecallResponse.results
        [].text, or a bare list, or dicts)."""
        results = getattr(resp, "results", None)
        if results is None and isinstance(resp, dict):
            results = resp.get("results")
        if results is None and isinstance(resp, (list, tuple)):
            results = resp
        blocks: list[str] = []
        for r in (results or []):
            txt = getattr(r, "text", None)
            if txt is None and isinstance(r, dict):
                txt = r.get("text") or r.get("content")
            if isinstance(txt, str) and txt.strip():
                blocks.append(txt.strip())
        return blocks
