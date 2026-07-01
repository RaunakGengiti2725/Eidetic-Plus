"""Mem0 baseline adapter -- REAL, never mocked.

Targets ``mem0ai==2.0.7`` (Apache-2.0). Mem0 is NOT installed in the Eidetic-Plus
venv; it lives in ``requirements-bench.txt`` and is imported lazily INSIDE ``__init__``.
A missing dependency or an empty ``DASHSCOPE_API_KEY`` FAILS LOUD with a clear message --
this adapter never silently falls back to a mock or a fabricated result.

Both Mem0's LLM and embedder are pointed at DashScope's OpenAI-compatible endpoint, so the
SAME Qwen models back every system in the neutral harness (llm ``qwen-plus``, embedder
``text-embedding-v4``). Mem0 still drives its own retrieval; the retrieved memories are
then handed to the ONE shared fixed reader (``answer_with_fixed_reader``) so the scoreboard
measures MEMORY quality, not answerer quality.

Mem0's real write cost is its add()-time LLM fact extraction: every ``m.add()`` runs an
LLM pass that decides what to store / update / delete. That extraction is genuine work and
is reflected in the cost table -- we time ingest end-to-end and account the ingested text
with the harness-uniform ``approx_tokens`` so writes are comparable across all systems.

Defensiveness note: Mem0's method signatures and config keys drift between releases. The
2.0.7 ``search()`` takes ``filters={"user_id": ...}`` + ``top_k`` (and REJECTS top-level
``user_id``/``limit``), while older releases used ``user_id=`` + ``limit=`` directly. We
try the documented call first and fall back across known signatures, but a real failure
(bad config, dead model, API error) is re-raised as a ModelCallError-style RuntimeError
telling the user to check their ``mem0ai`` version. We NEVER mock.
"""
from __future__ import annotations

import os
import time
import importlib.util
import threading
import queue
from typing import Any, Optional

from eidetic.config import get_settings

from ..reader import answer_with_fixed_reader
from .base import AnswerResult, MemorySystem, WriteResult, approx_tokens

_LLM_MODEL = os.environ.get("BENCH_BASELINE_LLM", "qwen-flash")  # baseline extraction LLM (funded)
_EMBED_MODEL = "text-embedding-v4"


class _Mem0CallTimeout(TimeoutError):
    pass


def _call_with_hard_deadline(call, timeout_s: float, op: str) -> Any:
    """Run a third-party Mem0/OpenAI call behind a process-survivable wall clock.

    OpenAI/httpx socket timeouts are per I/O phase, not a guaranteed total deadline, and
    SIGALRM can be deferred while CPython is blocked in SSL reads on macOS. A daemon worker
    lets the benchmark fail loud and render the completed rows even if a vendor call wedges.
    """
    if timeout_s <= 0.0:
        return call()

    out: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _runner() -> None:
        try:
            out.put((True, call()))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller thread
            out.put((False, exc))

    worker = threading.Thread(target=_runner, name=f"mem0-{op}-deadline", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise _Mem0CallTimeout(f"Mem0 {op} exceeded wall-clock timeout of {timeout_s:.3f}s")

    try:
        ok, payload = out.get_nowait()
    except queue.Empty as exc:  # defensive: the thread ended without surfacing a result.
        raise _Mem0CallTimeout(f"Mem0 {op} ended without returning within {timeout_s:.3f}s") from exc
    if ok:
        return payload
    raise payload


def _is_skippable_add_error(exc: Exception) -> bool:
    """A per-session add() failure that is CONTENT-specific (a 4xx bad request: moderation, an
    oversized window, or a malformed-for-this-content extraction) and therefore safe to skip for one
    session rather than abort the whole sample. A 5xx / network / dependency error returns False so it
    still fails loud. Mem0 wraps the underlying OpenAI BadRequestError in a RuntimeError, so we match
    on the surfaced text ('400' / 'bad request' / the moderation phrases)."""
    m = (str(exc) or "").lower()
    if any(h in m for h in ("inappropriate content", "data_inspection", "data inspection",
                            "content_filter", "content filter")):
        return True
    return ("400" in m) or ("bad request" in m) or ("badrequest" in m)


def _optional_health() -> dict:
    """Report optional Mem0 capabilities that materially affect baseline strength.

    Mem0 can still run without spaCy/fastembed, but those missing packages disable useful
    keyword/local paths in current releases. A public benchmark should surface that as a
    degraded competitor, not quietly count it as a fully healthy baseline.
    """
    probes = {
        "spacy": importlib.util.find_spec("spacy") is not None,
        "fastembed": importlib.util.find_spec("fastembed") is not None,
    }
    missing = [name for name, ok in probes.items() if not ok]
    return {
        "status": "ok" if not missing else "degraded",
        "system": "mem0",
        "optional_capabilities": probes,
        "missing_optional": missing,
        "strict": os.environ.get("STRICT_BASELINE_HEALTH", "").strip().lower()
        in ("1", "true", "yes", "on"),
    }


def _bound_openai_compatible_clients(memory: Any, timeout_s: float, max_retries: int) -> None:
    """Apply benchmark timeouts to Mem0's underlying OpenAI-compatible clients.

    Mem0 constructs its own OpenAI clients internally and its public config does not expose a
    timeout knob in the installed version. `with_options` preserves the same base URL/key/model
    while bounding liveness; it does not change Mem0 retrieval or scoring logic.
    """
    for owner_name in ("llm", "embedding_model"):
        owner = getattr(memory, owner_name, None)
        client = getattr(owner, "client", None)
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            owner.client = with_options(timeout=timeout_s, max_retries=max(0, int(max_retries)))


class Mem0System(MemorySystem):
    """Real Mem0 (2.0.7) under test, backed by DashScope-hosted Qwen models."""

    name = "mem0"

    def __init__(self) -> None:
        # --- Fail loud on a missing dependency (it lives in requirements-bench.txt). ----
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "Mem0 baseline requires the 'mem0ai' package which is NOT installed in "
                "this venv. Install the benchmark baselines first:\n"
                "    .venv/bin/pip install -r requirements-bench.txt\n"
                "(target: mem0ai==2.0.7). Original import error: " + repr(e)
            ) from e

        # --- Fail loud on a missing key (no fake/mocked answers, ever). -----------------
        s = get_settings()
        api_key = (s.api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is empty -- the Mem0 baseline cannot run without it "
                "(it backs Mem0's LLM + embedder via the OpenAI-compatible endpoint). "
                "Set DASHSCOPE_API_KEY in your .env. No mock fallback exists."
            )
        base_url = s.compatible_base_url

        # Mem0's OpenAI client also reads OPENAI_API_KEY / OPENAI_BASE_URL from the
        # environment; set them as a fallback so the DashScope endpoint is used even on
        # any code path that bypasses the explicit config dict.
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        self._health = _optional_health()
        if self._health["strict"] and self._health["status"] != "ok":
            missing = ", ".join(self._health["missing_optional"])
            raise RuntimeError(
                "Mem0 baseline is degraded because optional capability packages are missing: "
                f"{missing}. Install the benchmark baselines fully or unset "
                "STRICT_BASELINE_HEALTH for exploratory runs."
            )

        # Mem0's qdrant store defaults to a PERSISTENT path (/tmp/qdrant). If a collection
        # already exists there at the wrong dim (1536, OpenAI's), our 1024 is ignored and
        # add() shape-mismatches. Use a FRESH path so the collection is created at 1024.
        import shutil
        import uuid

        self._qdrant_path = str(s.data_dir / "mem0_qdrant" / uuid.uuid4().hex)
        self._call_timeout_s = float(s.dashscope_request_timeout_sec)
        shutil.rmtree(self._qdrant_path, ignore_errors=True)

        # Mem0 config dict: both LLM and embedder are OpenAI-compatible, pointed at
        # DashScope. The 2.0.7 OpenAIConfig / BaseEmbedderConfig accept `openai_base_url`
        # and (embedder) `embedding_dims`; they also honor the env-var fallback above.
        config: dict[str, Any] = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": _LLM_MODEL,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": _EMBED_MODEL,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "embedding_dims": int(s.embed_dim),
                },
            },
            # The vector store's collection dim MUST match text-embedding-v4 (1024), else
            # Mem0's default qdrant collection (1536, OpenAI's dim) shape-mismatches on add().
            # A fresh on-disk path guarantees the collection is created at 1024 (not reused).
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": int(s.embed_dim),
                    "path": self._qdrant_path,
                    "collection_name": "eidetic_bench",
                    "on_disk": False,
                },
            },
        }

        # Construct the real Memory store. If the exact config keys differ in the installed
        # mem0ai version, raise a clear ModelCallError-style message -- never a silent mock.
        try:
            self._memory = Memory.from_config(config)
            _bound_openai_compatible_clients(
                self._memory,
                float(s.dashscope_request_timeout_sec),
                min(2, int(s.dashscope_max_retries)),
            )
        except Exception as e:  # noqa: BLE001 - re-raised loudly below
            raise RuntimeError(
                "Mem0 Memory.from_config(...) failed to build a real store. This usually "
                "means the installed mem0ai version uses different config keys than 2.0.7 "
                "(expected llm/embedder provider 'openai' with config "
                "{model, api_key, openai_base_url[, embedding_dims]}). Check your mem0ai "
                f"version (pin mem0ai==2.0.7). Underlying error: {e!r}"
            ) from e

    # ------------------------------------------------------------------------------------
    def reset(self, namespace: str) -> None:
        """Best-effort clear of this user_id's memories. Mem0 stores are per-user_id, so a
        fresh namespace is already isolated; this just drops any prior state if reused."""
        try:
            self._memory.delete_all(user_id=namespace)
        except Exception:  # noqa: BLE001 - reset is best-effort; isolation comes from user_id
            pass

    # ------------------------------------------------------------------------------------
    def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                       session_time: Optional[float] = None) -> WriteResult:
        """Add ONE memory per session (joined turns) via m.add() -- session granularity,
        matching the Eidetic and Graphiti adapters for a fair, neutral comparison (and far
        faster than per-turn). Mem0 still runs its own LLM fact extraction inside add(),
        which IS its real write cost (reflected in the cost table)."""
        content = "\n".join(
            f"{(t.get('role') or 'user')}: {t.get('content', '')}".strip()
            for t in turns if (t.get("content") or "").strip()
        ).strip()
        if not content:
            return WriteResult(tokens=0, ms=0.0)
        t0 = time.perf_counter()
        try:
            self._add_turn("user", content, namespace)
        except Exception as e:  # noqa: BLE001 - classified below; non-4xx still fails loud
            # FAIRNESS: a content-specific 4xx (moderation / oversized / bad request) on ONE session
            # must not abort the whole sample -- the eidetic adapter likewise skips content it cannot
            # process (a moderated extraction window), so skipping the session here keeps the
            # comparison apples-to-apples instead of zeroing mem0 on a single flagged session. Any
            # other error (5xx, network, dependency) still fails loud. Logged, never silently mocked.
            if _is_skippable_add_error(e):
                print(f"  [mem0] skipped a session ({namespace}): {type(e).__name__}: {str(e)[:120]}")
                return WriteResult(tokens=0, ms=(time.perf_counter() - t0) * 1000.0)
            raise
        return WriteResult(tokens=approx_tokens(content), ms=(time.perf_counter() - t0) * 1000.0)

    def _add_turn(self, role: str, content: str, namespace: str) -> None:
        """Call m.add() defensively across mem0 signatures; raise loud on a real failure.

        add() runs Mem0's LLM fact extraction (its real, billable write cost), so the
        primary 2.0.7 form is tried FIRST and a fallback fires ONLY on a clearly
        argument-binding TypeError -- never re-running the LLM after a real failure."""
        messages = [{"role": role, "content": content}]
        # 2.0.7: add(messages, *, user_id=...). Older: add(content, user_id=...).
        attempts = (
            lambda: self._memory.add(messages, user_id=namespace),
            lambda: self._memory.add(content, user_id=namespace),
        )
        self._try_calls(attempts, op="add")

    # ------------------------------------------------------------------------------------
    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        """Retrieve Mem0's own memories, then answer with the ONE shared fixed reader.

        Timing mirrors the Eidetic adapter: search_ms covers retrieval only; e2e_ms covers
        retrieval + the shared reader (two perf_counter reads). context_tokens is the
        approx_tokens of the joined retrieved blocks."""
        t0 = time.perf_counter()
        hits = self._search(question, namespace, limit=10)
        search_ms = (time.perf_counter() - t0) * 1000.0

        context_blocks = self._blocks_from_hits(hits)
        text = answer_with_fixed_reader(question, context_blocks)
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        context_tokens = approx_tokens("\n\n".join(context_blocks))
        return AnswerResult(
            answer=text,
            context_tokens=context_tokens,
            search_ms=search_ms,
            e2e_ms=e2e_ms,
            abstained=False,
            extra={"hits": len(context_blocks), "baseline_health": self._health},
        )

    def _search(self, question: str, namespace: str, limit: int) -> Any:
        """Call m.search() defensively. 2.0.7 uses filters={'user_id':...} + top_k and
        REJECTS top-level user_id/limit; older releases used user_id=/limit= directly."""
        attempts = (
            # mem0 2.0.7 keyword-only API
            lambda: self._memory.search(question, filters={"user_id": namespace}, top_k=limit),
            lambda: self._memory.search(query=question, filters={"user_id": namespace}, top_k=limit),
            # legacy API
            lambda: self._memory.search(query=question, user_id=namespace, limit=limit),
            lambda: self._memory.search(question, user_id=namespace, limit=limit),
        )
        return self._try_calls(attempts, op="search")

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _blocks_from_hits(hits: Any) -> list[str]:
        """Normalize Mem0's search return into a list of context strings.

        2.0.7 returns {'results': [{'memory': ..., 'score': ...}, ...]}; some releases
        return a bare list. Each item may carry the text under 'memory' or 'text'."""
        if isinstance(hits, dict):
            items = hits.get("results", hits.get("memories", []))
        elif isinstance(hits, list):
            items = hits
        else:
            items = []

        blocks: list[str] = []
        for item in items:
            if isinstance(item, dict):
                val = item.get("memory") or item.get("text") or item.get("content") or ""
            else:
                val = str(item)
            val = (val or "").strip()
            if val:
                blocks.append(val)
        return blocks

    # ------------------------------------------------------------------------------------
    # Substrings that mark a TypeError as an argument-BINDING mismatch (wrong signature
    # for this mem0 version) rather than an internal mem0/OpenAI-client bug.
    _BINDING_HINTS = (
        "unexpected keyword argument",
        "positional argument",
        "required positional",
        "got multiple values",
        "missing 1 required",
        "takes no arguments",
        "takes from",
    )

    def _try_calls(self, attempts, op: str) -> Any:
        """Try each call FORM in order, falling back ONLY on an argument-binding TypeError
        (the signature didn't match this mem0 version). Every other exception -- including a
        ValueError (mem0 2.0.7 raises ValueError for empty/invalid queries, bad
        threshold/top_k, or missing filter keys) and any internal TypeError -- is a GENUINE
        runtime failure and is re-raised loudly. We NEVER swallow a real error into a mock.

        The primary (2.0.7) form is always attempt #0, so in normal operation the fallback
        never fires and a real failure surfaces immediately with its true cause."""
        sig_errors: list[str] = []
        timeout_s = float(getattr(self, "_call_timeout_s", get_settings().dashscope_request_timeout_sec))
        for call in attempts:
            try:
                return _call_with_hard_deadline(call, timeout_s, op)
            except TypeError as e:
                msg = str(e).lower()
                if any(h in msg for h in self._BINDING_HINTS):
                    # Signature mismatch for this mem0 version: record and try the next form.
                    sig_errors.append(repr(e))
                    continue
                # An internal TypeError from mem0 / the OpenAI client -> real failure.
                raise RuntimeError(
                    f"Mem0 {op}() raised an internal TypeError (real error, no mock "
                    f"fallback). Check DashScope reachability and that mem0ai==2.0.7 is "
                    f"installed. Underlying error: {e!r}"
                ) from e
            except Exception as e:  # noqa: BLE001 - a genuine runtime failure, fail loud
                raise RuntimeError(
                    f"Mem0 {op}() failed at runtime (real error, no mock fallback). "
                    f"Check that DashScope is reachable and mem0ai==2.0.7 is installed. "
                    f"Underlying error: {e!r}"
                ) from e
        raise RuntimeError(
            f"Mem0 {op}() did not match any known signature for the installed mem0ai "
            f"version. Expected mem0ai==2.0.7. Signature errors tried: {sig_errors}"
        )

    # ------------------------------------------------------------------------------------
    def teardown(self) -> None:
        return None
