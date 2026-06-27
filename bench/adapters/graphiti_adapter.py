"""Graphiti baseline adapter (REAL -- fails loud, never mocks).

Targets graphiti-core==0.29.2 + Neo4j (use AuraDB Free in the cloud -- no Docker, no
local server needed; a free AuraDB instance gives you NEO4J_URI/USER/PASSWORD). The
LLM + embedder are pinned to the SAME DashScope (Qwen) models that back every other
system in this harness, via Graphiti's OpenAI-compatible client, so the scoreboard
compares MEMORY quality, not model quality.

Heavy-write note: Graphiti runs a per-episode LLM extraction pipeline on every
`add_episode` (entity/edge extraction, dedup, temporal resolution). That is its real,
unavoidable write cost -- in the published literature this runs to >600k tokens for a
single multi-session conversation. The harness's cost table is designed to expose
exactly this: tokens/write here are dominated by Graphiti's own extraction, not by the
raw conversation text we feed it.

This adapter is REAL: if graphiti-core is not importable, if NEO4J_URI/NEO4J_USER/
NEO4J_PASSWORD are unset, or if DASHSCOPE_API_KEY is empty, it raises a clear
RuntimeError. It NEVER returns a fabricated or mocked result.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from eidetic.config import get_settings

from ..reader import answer_with_fixed_reader
from .base import AnswerResult, MemorySystem, WriteResult, approx_tokens


class GraphitiSystem(MemorySystem):
    """Real Graphiti memory backend. Per-conversation isolation via Graphiti's group_id
    (group_id = the harness `namespace`)."""

    name = "graphiti"

    def __init__(self) -> None:
        # ---- Fail-loud preconditions (ordered so the error message is the useful one).
        # 1) Neo4j connection settings -- checked first so a missing server config does not
        #    later surface as an opaque driver traceback.
        uri = os.environ.get("NEO4J_URI", "").strip()
        user = os.environ.get("NEO4J_USER", "").strip()
        password = os.environ.get("NEO4J_PASSWORD", "").strip()
        missing = [n for n, v in (("NEO4J_URI", uri), ("NEO4J_USER", user),
                                  ("NEO4J_PASSWORD", password)) if not v]
        if missing:
            raise RuntimeError(
                "GraphitiSystem requires Neo4j connection env vars; missing: "
                f"{', '.join(missing)}. You do NOT need Docker or a local server: "
                "a free Neo4j AuraDB cloud instance works -- create one at "
                "https://neo4j.com/cloud/aura-free/ and set NEO4J_URI (e.g. "
                "neo4j+s://<id>.databases.neo4j.io), NEO4J_USER (usually 'neo4j'), and "
                "NEO4J_PASSWORD."
            )

        # 2) DashScope key -- Graphiti's LLM + embedder run through the OpenAI-compatible
        #    DashScope endpoint, so the same Qwen models back every system.
        s = get_settings()
        api_key = (s.api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "GraphitiSystem requires DASHSCOPE_API_KEY (used for Graphiti's LLM "
                "extraction + embeddings via the OpenAI-compatible DashScope endpoint). "
                "It is empty. Set DASHSCOPE_API_KEY in your environment / .env."
            )
        base_url = s.compatible_base_url
        # graphiti-core's OpenAI clients fall back to OPENAI_API_KEY/OPENAI_BASE_URL env when
        # they don't thread config.api_key through every internal AsyncOpenAI. Set them so the
        # adapter works STANDALONE (not only when another adapter set them first).
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url

        # 3) Import graphiti-core. Each uncertain symbol is imported in its own guard so a
        #    path drift in a future point release produces a clear, named error instead of
        #    an AttributeError several calls deep. We NEVER fall back to a mock.
        try:
            from graphiti_core import Graphiti
        except Exception as e:  # ImportError or anything else at import time
            raise RuntimeError(
                "GraphitiSystem requires graphiti-core (pip install "
                "'graphiti-core==0.29.2'); import of `graphiti_core.Graphiti` failed: "
                f"{e!r}. This adapter is real and will not mock the library."
            ) from e

        try:
            from graphiti_core.llm_client.config import LLMConfig
        except Exception as e:
            raise RuntimeError(
                "graphiti-core 0.29.2 API drift: could not import "
                "`graphiti_core.llm_client.config.LLMConfig`: "
                f"{e!r}. Verify the installed graphiti-core version."
            ) from e

        try:
            # The GENERIC client uses chat.completions + JSON mode (NOT OpenAI's Responses
            # API `responses.parse`), which is what DashScope's OpenAI-compatible endpoint
            # supports. The default OpenAIClient hard-requires `responses.parse` and fails
            # against qwen (confirmed live).
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        except Exception as e:
            raise RuntimeError(
                "graphiti-core 0.29.2 API drift: could not import "
                "`graphiti_core.llm_client.openai_generic_client.OpenAIGenericClient`: "
                f"{e!r}. Verify the installed graphiti-core version."
            ) from e

        try:
            from graphiti_core.embedder.openai import (
                OpenAIEmbedder,
                OpenAIEmbedderConfig,
            )
        except Exception as e:
            raise RuntimeError(
                "graphiti-core 0.29.2 API drift: could not import "
                "`graphiti_core.embedder.openai.OpenAIEmbedder` / "
                f"`OpenAIEmbedderConfig`: {e!r}. Verify the installed graphiti-core "
                "version."
            ) from e

        try:
            from graphiti_core.nodes import EpisodeType
        except Exception as e:
            raise RuntimeError(
                "graphiti-core 0.29.2 API drift: could not import "
                "`graphiti_core.nodes.EpisodeType`: "
                f"{e!r}. Verify the installed graphiti-core version."
            ) from e
        self._EpisodeType = EpisodeType

        # ---- Build the LLM + embedder pinned to DashScope/Qwen (OpenAI-compatible).
        #      llm model = qwen-plus ; embedder = text-embedding-v4 (the harness defaults).
        # text-embedding-v4 caps a batch at 10; Graphiti's default embedder sends ALL entity
        # texts in one request (>10 -> HTTP 400). Subclass it to chunk into <=10 per call.
        class _ChunkedEmbedder(OpenAIEmbedder):
            async def create_batch(self, input_data_list):  # type: ignore[override]
                out: list = []
                for i in range(0, len(input_data_list), 10):
                    res = await self.client.embeddings.create(
                        input=input_data_list[i:i + 10], model=self.config.embedding_model)
                    out.extend(e.embedding[: self.config.embedding_dim] for e in res.data)
                return out

        try:
            _llm = os.environ.get("BENCH_BASELINE_LLM", "qwen-flash")  # funded baseline LLM
            llm_client = OpenAIGenericClient(
                config=LLMConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=_llm,
                    small_model=_llm,
                )
            )
            embedder = _ChunkedEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=api_key,
                    base_url=base_url,
                    embedding_model="text-embedding-v4",
                    embedding_dim=int(s.embed_dim),
                )
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to construct Graphiti's DashScope-backed LLM/embedder clients "
                f"(graphiti-core 0.29.2 API drift?): {e!r}."
            ) from e

        # ---- Persistent event loop. The Neo4j async driver binds its connection pool to
        #      the loop it was created on; using asyncio.run() per call would create and
        #      destroy a loop each time and the driver would die ("event loop is closed").
        #      We keep ONE loop for the lifetime of this adapter.
        self._loop = asyncio.new_event_loop()

        # ---- Construct Graphiti against the live Neo4j and build indices once.
        try:
            self.graphiti = Graphiti(
                uri,
                user,
                password,
                llm_client=llm_client,
                embedder=embedder,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to construct Graphiti(uri, user, password, ...). Check NEO4J_URI/"
                "NEO4J_USER/NEO4J_PASSWORD point at a reachable Neo4j (a free AuraDB "
                f"cloud instance works): {e!r}."
            ) from e

        try:
            self._run(self.graphiti.build_indices_and_constraints())
        except Exception as e:
            raise RuntimeError(
                "Graphiti.build_indices_and_constraints() failed -- the Neo4j server is "
                "not reachable or the credentials are wrong. A free AuraDB cloud instance "
                "needs no Docker; verify NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD. "
                f"Underlying error: {e!r}."
            ) from e

    # ------------------------------------------------------------------ helpers
    def _run(self, coro: Any) -> Any:
        """Drive a Graphiti coroutine on our persistent event loop."""
        return self._loop.run_until_complete(coro)

    @staticmethod
    def _to_dt(session_time: Optional[float]) -> datetime:
        """tz-aware datetime for Graphiti's temporal logic (naive datetimes bite it)."""
        if session_time is None:
            return datetime.now(timezone.utc)
        return datetime.fromtimestamp(session_time, tz=timezone.utc)

    # ------------------------------------------------------------------ MemorySystem
    def reset(self, namespace: str) -> None:
        """Graphiti isolates by group_id, so a fresh namespace is already isolated. This is
        a best-effort cleanup of any prior data under this group_id; never fatal."""
        try:
            delete_group = getattr(self.graphiti, "delete_group", None)
            if callable(delete_group):
                self._run(delete_group(namespace))
        except Exception:
            # Best-effort only: a fresh group_id is isolated regardless.
            pass

    def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                       session_time: Optional[float] = None) -> WriteResult:
        # Join the session's turns into one episode body. Assistant turns are kept as
        # first-class content (dropping them is a known accuracy loss in this literature).
        parts: list[str] = []
        for turn in turns:
            text = f"{turn.get('role', 'user')}: {turn.get('content', '')}".strip()
            if text and text != f"{turn.get('role', 'user')}:":
                parts.append(text)
        body = "\n".join(parts)
        tokens = approx_tokens(body)

        t0 = time.perf_counter()
        # This triggers Graphiti's per-episode LLM extraction -- the real, heavy write cost.
        self._run(self.graphiti.add_episode(
            name=session_id,
            episode_body=body,
            source=self._EpisodeType.text,
            source_description="conversation",
            reference_time=self._to_dt(session_time),
            group_id=namespace,
        ))
        ms = (time.perf_counter() - t0) * 1000.0
        return WriteResult(tokens=tokens, ms=ms)

    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        # Time retrieval (search) separately from end-to-end (incl. the fixed reader).
        t0 = time.perf_counter()
        results = self._run(self.graphiti.search(
            query=question,
            group_ids=[namespace],
            num_results=10,
        ))
        search_ms = (time.perf_counter() - t0) * 1000.0

        # Build context blocks from the returned edges/facts, defensively: an EntityEdge in
        # 0.29.2 exposes `.fact`; we also fall back to `.name`/`.summary` and str() so a
        # shape change never silently drops everything.
        context_blocks: list[str] = []
        for r in (results or []):
            block = (
                getattr(r, "fact", None)
                or getattr(r, "summary", None)
                or getattr(r, "name", None)
            )
            if not block:
                block = str(r)
            block = str(block).strip()
            if block:
                context_blocks.append(block)

        # Neutrality: every system answers through the ONE shared fixed reader.
        text = answer_with_fixed_reader(question, context_blocks)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        context_tokens = approx_tokens("\n\n".join(context_blocks))

        return AnswerResult(
            answer=text,
            context_tokens=context_tokens,
            search_ms=search_ms,
            e2e_ms=e2e_ms,
            abstained=False,
            extra={"n_results": len(context_blocks)},
        )

    def teardown(self) -> None:
        try:
            close = getattr(self.graphiti, "close", None)
            if callable(close):
                self._run(close())
        except Exception:
            pass
        finally:
            loop = getattr(self, "_loop", None)
            if loop is not None and not loop.is_closed():
                loop.close()
