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

LIVE-VERIFIED (2026-07-08) and the two integration bugs the live smoke caught:
  1. Model name: pass the BARE model (``qwen-plus``); Hindsight prefixes the provider, so
     ``openai/qwen-plus`` becomes ``openai/openai/qwen-plus`` -> 404. Confirmed fixed:
     daemon logs ``Connection verified: openai/qwen-plus``.
  2. Drain signal: ``list_memories`` returns ``.total``/``.items`` (NOT ``memory_units``);
     ``consolidate`` polls ``.total`` until the async worker settles.
KNOWN NEXT-SESSION INFRA ITEM (not adapter logic): ``HindsightEmbedded`` owns a persistent
daemon and its startup has a race -- on a cold ``pg0`` + LLM connection-verify the client can
raise "Failed to start daemon" even though the daemon comes up moments later; and a stale
daemon must be stopped before a config change is picked up (``teardown`` closes it). Give the
daemon a warm start (or retry construction) before the r13/r14/r15 head-to-head.
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

        # Hindsight's retain fact-extraction defaults max_completion_tokens=64000, but
        # qwen-plus caps OUTPUT at 32768 -> HTTP 400. The daemon (a subprocess) reads this
        # from the environment, so set it BEFORE launch. Must exceed RETAIN_CHUNK_SIZE(3000).
        os.environ.setdefault("HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS",
                              os.environ.get("HINDSIGHT_RETAIN_MAX_COMPLETION_TOKENS", "32000"))

        # litellm 'openai/<model>' + DashScope OpenAI-compatible base = same Qwen family as
        # every other baseline. Embedded pg0 backend: no external Postgres to provision.
        # llm_provider already selects the openai-compatible path; pass the BARE model name
        # (Hindsight prefixes the provider itself -- passing "openai/qwen-plus" yields the
        # 404'ing "openai/openai/qwen-plus").
        profile = os.environ.get("HINDSIGHT_PROFILE", "eidetic-bench")
        self._h = h.HindsightEmbedded(
            profile=profile,
            llm_provider="openai",
            llm_api_key=key,
            llm_model=_LLM_MODEL,
            llm_base_url=settings.compatible_base_url,
        )
        self._seen_banks: set[str] = set()
        self._warm_start()

    def _warm_start(self) -> None:
        """Force the embedded daemon up, tolerating the cold-start race. HindsightEmbedded's
        internal start timeout (~30s) can be exceeded by a first-ever ``pg0`` init + LLM
        connection-verify, raising "Failed to start daemon" even though the daemon comes up
        moments later. ``ensure_running`` is idempotent, so we retry the first daemon-touch
        with backoff (total ~4 min) -- a later attempt attaches to the now-healthy daemon.
        Fails loud only if every attempt fails."""
        attempts = int(os.environ.get("HINDSIGHT_START_ATTEMPTS", "5"))
        last = None
        for i in range(attempts):
            try:
                _ = self._h.client            # triggers _ensure_started
                return
            except Exception as e:             # noqa: BLE001 - retry the documented race
                last = e
                time.sleep(min(20.0 * (i + 1), 60.0))
        raise RuntimeError(
            f"Hindsight daemon failed to start after {attempts} attempts "
            f"(cold pg0 start race). Last error: {last}")

    @property
    def _c(self):
        # bank management + list_memories live on the underlying client; retain/recall are
        # proxied on the embedded object, but be explicit for the management calls.
        return getattr(self._h, "client", None) or self._h

    def reset(self, namespace: str) -> None:
        # Real isolation: drop + recreate the bank so a re-run never sees stale memories.
        c = self._c
        try:
            c.delete_bank(namespace)
        except Exception:
            pass  # first run: bank does not exist yet (only not-found is tolerable)
        try:
            c.create_bank(namespace)
        except Exception:
            pass  # 'create or update'; retain also lazily creates
        # VERIFY isolation instead of trusting a swallowed delete: a silently-failed delete
        # leaves stale memories that would corrupt this namespace's row. Fail loud if the
        # bank is not empty after reset.
        try:
            resp = c.list_memories(namespace, limit=1)
            total = getattr(resp, "total", None)
            if isinstance(resp, dict):
                total = resp.get("total")
            if isinstance(total, int) and total > 0:
                raise RuntimeError(
                    f"Hindsight reset did not clear bank '{namespace}' (total={total} after "
                    f"delete+create). Refusing to ingest into a dirty bank -- would corrupt "
                    f"the row. Use a fresh HINDSIGHT_PROFILE.")
        except RuntimeError:
            raise
        except Exception:
            pass  # list_memories unavailable on this client shape -> best effort
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

    def consolidate(self, namespace: str) -> dict:
        """Hindsight ingests ASYNC (a background worker extracts + indexes retained turns).
        recall() before that drains returns 0 hits -- so we WAIT here (the harness calls
        consolidate after all ingest, before any answer) by polling list_memories until the
        indexed count stops growing, bounded by HINDSIGHT_CONSOLIDATE_TIMEOUT_SEC (default
        180). This is a fair, real settle -- not a fixed sleep -- and fails loud if the
        worker never indexes anything."""
        timeout = float(os.environ.get("HINDSIGHT_CONSOLIDATE_TIMEOUT_SEC", "180"))
        deadline = time.perf_counter() + timeout
        last, stable, polls = -1, 0, 0
        while time.perf_counter() < deadline:
            try:
                resp = self._c.list_memories(namespace, limit=1000)
            except Exception:
                resp = None
            # ListMemoryUnitsResponse exposes .total (int) + .items (list).
            count = 0
            if resp is not None:
                total = getattr(resp, "total", None)
                items = getattr(resp, "items", None)
                if isinstance(resp, dict):
                    total = resp.get("total"); items = resp.get("items") or resp.get("results")
                count = int(total) if isinstance(total, int) else (len(items) if items is not None else 0)
            polls += 1
            if count > 0 and count == last:
                stable += 1
                if stable >= 2:            # two consecutive equal non-zero reads => drained
                    return {"indexed": count, "polls": polls, "settled": True}
            else:
                stable = 0
            last = count
            time.sleep(3.0)
        # FAIL LOUD if the worker indexed NOTHING after ingest (e.g. the extraction LLM
        # 400'd on every chunk): a silent 0-index would answer every question from an empty
        # store and those wrong rows would count in accuracy, silently corrupting the row.
        if namespace in self._seen_banks and max(last, 0) == 0:
            raise RuntimeError(
                f"Hindsight consolidation indexed 0 memories for '{namespace}' after ingest "
                f"({polls} polls / {timeout}s). The extraction worker likely failed (check "
                f"the daemon log). Refusing to score an empty store as answers.")
        return {"indexed": max(last, 0), "polls": polls, "settled": False,
                "note": "consolidation did not settle within timeout (partial index)"}

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

    def teardown(self) -> None:
        # HindsightEmbedded starts a PERSISTENT daemon (survives the process) whose LLM
        # config is fixed at launch. A bare close() defaults to stop_daemon=False, leaving the
        # daemon alive with STALE config (observed live: a stale daemon 404'd on the old model)
        # -> a later run silently reuses the wrong model. Force an ACTUAL stop.
        h = self._h
        stopped = False
        fn = getattr(h, "close", None)
        if callable(fn):
            try:
                fn(stop_daemon=True)   # the real fix: actually stop the daemon
                stopped = True
            except TypeError:
                try:
                    fn()               # older signature without the kwarg
                except Exception:
                    pass
            except Exception:
                pass
        if not stopped:
            for m in ("stop", "shutdown"):
                g = getattr(h, m, None)
                if callable(g):
                    try:
                        g()
                    except Exception:
                        pass
                    break

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
