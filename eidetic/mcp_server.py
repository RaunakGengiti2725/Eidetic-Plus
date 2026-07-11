"""Eidetic-Plus as a UNIVERSAL MEMORY PLUGIN: an MCP server over the same Engine.

Any MCP host (Claude Code, Claude Desktop, Cursor, Cline, Windsurf, Zed, and others) can mount
Eidetic-Plus as its memory backend by adding one config line. This module is a THIN adapter: each
tool validates inputs, calls an existing engine method, and formats the result. No retrieval,
storage, or consolidation logic lives here. The immutable lossless store stays byte-identical
because the MCP layer only calls existing engine methods, it never bypasses them.

SCOPING is enforced on every tool. Scope resolves as: explicit tool argument, then the env
defaults EIDETIC_NAMESPACE / EIDETIC_AGENT_ID / EIDETIC_PROJECT_ID, then the safe global default
"default". A read in namespace A never returns a memory written in namespace B.

STARTUP WITHOUT A KEY: the server starts and lists every tool with no DASHSCOPE_API_KEY. The
read-only tools (list_memories, get_raw, stats) work without a key. A tool that needs the model
fails loud with an actionable message only when it is actually called. It never fabricates a
result and never silently no-ops.

Run:
    uvx eidetic-plus                          # one-command install + run (stdio)
    python -m eidetic.mcp_server              # stdio (default; Claude Code / Cursor / Cline)
    python -m eidetic.mcp_server --http --http-port 8765             # remote / shared
    python -m eidetic.mcp_server --transport http --http-port 8765   # same, explicit form
"""
from __future__ import annotations

import argparse
import base64
import functools
import os
import threading
from typing import Optional

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from .dashscope_client import ModelCallError
from .engine import Engine
from .models import Scope

_HOST = os.environ.get("EIDETIC_MCP_HOST", "127.0.0.1")
_PORT = int(os.environ.get("EIDETIC_MCP_PORT", "8765"))

mcp = FastMCP(
    "eidetic-plus",
    host=_HOST,
    port=_PORT,
    instructions=(
        "Persistent, lossless, VERIFIABLE long-term memory for AI agents. Answers are "
        "verify-or-abstain: every recall is NLI-checked against immutable stored sources and "
        "returns citations (content hash, validity window, entailment label) or an explicit "
        "abstention - never a confabulation. The store is bi-temporal: `remember` accepts "
        "`valid_at` to backdate a fact's EVENT time, and `recall`/`truth_ledger` accept "
        "`as_of` to answer as of any past moment (superseded facts answer for their era; "
        "later facts are invisible). Retractions are first-class: a negated assertion "
        "answers 'No - <premise>' with its source, and the latest assertion wins. "
        "Tool guide: `remember` durable facts (backdate imports with valid_at); "
        "`remember_file` PDFs/images/documents losslessly; `recall` the verified cited "
        "answer (prove=True adds a machine-checkable proof tree); `truth_ledger` the full "
        "raw-bytes-to-current-truth chain with supersession history; `structured_recall` "
        "the deterministic typed-claim path (no generation); `reflex_recall` sub-second "
        "candidate recall with no model call; `get_raw` byte-identical source bytes; "
        "`value_as_of`/`fact_history` deterministic entity-relation time travel; `forget`/"
        "`reawaken` reversible priority decay (never deletion); `remember_many` bulk import "
        "with in-batch dedup; `repair` rebuilds derived indexes from the source of truth. "
        "WAR ROOM (shared problem memory): `remember_problem` opens a goal with status/"
        "blockers; `add_hypothesis`/`resolve_hypothesis` track theories with evidence refs; "
        "`update_problem` records decisions ({choice, rationale}) and handoffs; `add_witness` "
        "attaches hash-verified files; `recall_problem` returns the folded state (as_of "
        "replays any past moment); `ask_problem` answers natural-language questions against "
        "the history with revision-backed citations. Every tool takes an optional "
        "`namespace` (+ `agent_id`/`project_id`); omitted values use EIDETIC_NAMESPACE / "
        "EIDETIC_AGENT_ID / EIDETIC_PROJECT_ID, then the global default. Reads never cross "
        "namespaces - use a stable namespace per project or agent."
    ),
)

_DEFAULT_PAGE_LIMIT = 50
_MAX_PAGE_LIMIT = 500
_MAX_CITATION_LIMIT = 50
_DEFAULT_RAW_MAX_BYTES = 200_000
_MAX_RAW_BYTES = 2_000_000
_MAX_QUERY_CHARS = 12_000
_MAX_CONTENT_CHARS = int(os.environ.get("EIDETIC_MCP_MAX_CONTENT_CHARS", "1000000"))
_MAX_PROBE_K = 50

# One long-lived Engine, built lazily on first model-or-store use, shared by all tool calls.
# Construction does NOT call the API, so the server starts and lists tools without a key.
_engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:                 # double-checked: never construct two Engines concurrently
            if _engine is None:
                _engine = Engine()
    return _engine


def _scope(namespace: Optional[str] = None, agent_id: Optional[str] = None,
           project_id: Optional[str] = None) -> Scope:
    """Resolve scope: explicit argument, then env default, then the safe global default."""
    ns = namespace or os.environ.get("EIDETIC_NAMESPACE") or "default"
    aid = agent_id or os.environ.get("EIDETIC_AGENT_ID") or None
    pid = project_id or os.environ.get("EIDETIC_PROJECT_ID") or None
    return Scope(namespace=ns, agent_id=aid, project_id=pid)


def _bounded_int(value: int | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        n = int(default if value is None else value)
    except (TypeError, ValueError):
        n = default
    return max(minimum, min(maximum, n))


def _text_arg(value: str, name: str, *, max_chars: int) -> str:
    text = (value or "").strip()
    if not text:
        raise RuntimeError(f"{name} must not be empty")
    if len(text) > max_chars:
        raise RuntimeError(f"{name} is too large ({len(text)} chars; max {max_chars})")
    return text


def _bounded_raw(raw: bytes, *, offset: int, max_bytes: int) -> tuple[bytes, dict]:
    start = _bounded_int(offset, default=0, minimum=0, maximum=max(0, len(raw)))
    cap = _bounded_int(max_bytes, default=_DEFAULT_RAW_MAX_BYTES,
                       minimum=1, maximum=_MAX_RAW_BYTES)
    end = min(len(raw), start + cap)
    # A byte slice may split a multibyte UTF-8 character at either edge; for text content that
    # would flip the WHOLE page to base64. Trim up to 3 continuation bytes (0b10xxxxxx) from the
    # leading edge and an incomplete sequence from the trailing edge, and report the ADJUSTED
    # offsets so byte-accurate reassembly still holds. Genuinely binary content is unaffected
    # (it fails decoding regardless and falls to base64 as before).
    for _ in range(3):
        if start < end and (raw[start] & 0b1100_0000) == 0b1000_0000:
            start += 1
        else:
            break
    if end < len(raw):
        back = end - 1
        while back > start and end - back <= 3 and (raw[back] & 0b1100_0000) == 0b1000_0000:
            back -= 1
        if back >= start and (raw[back] & 0b1100_0000) == 0b1100_0000:
            lead = raw[back]
            need = 2 if lead >= 0b1100_0000 else 0
            if lead >= 0b1110_0000:
                need = 3
            if lead >= 0b1111_0000:
                need = 4
            if back + need > end:            # sequence incomplete inside this page: trim it
                end = back
    chunk = raw[start:end]
    return chunk, {
        "raw_total_bytes": len(raw),
        "raw_offset": start,
        "raw_returned_bytes": len(chunk),
        "raw_truncated": start + len(chunk) < len(raw),
        "raw_max_bytes": cap,
    }


def _threaded_tool(fn):
    """Register `fn` as an MCP tool that runs on a WORKER THREAD, not the event loop.

    FastMCP executes sync tools inline on the asyncio loop thread, so in the shared HTTP mode
    one multi-second model-backed call (recall, remember, truth_ledger, preflight) freezes
    every session: no other client's request, no key-free read-only tool, no protocol ping is
    serviced until it returns. The wrapper offloads via anyio.to_thread (the Engine is already
    exercised multithreaded by the HTTP API). Each tool body runs whole on ONE worker thread,
    so same-call thread-local flows (ask -> prove trace splicing) keep exact semantics; the
    cross-call recall_trace surface reads the engine's published completed-trace snapshot.
    Returns the ORIGINAL sync function so direct in-process calls keep their signature."""
    @functools.wraps(fn)
    async def _async_tool(*args, **kwargs):
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
    mcp.tool()(_async_tool)
    return fn


def _brief(rec) -> dict:
    return {
        "memory_id": rec.memory_id,
        "content_hash": rec.content_hash,
        "modality": rec.modality.value,
        "source": rec.source,
        "scope": rec.scope.model_dump(),
        "valid_at": rec.valid_at,
        "salience": round(rec.salience, 3),
        "retrievability": round(rec.fsrs.priority(), 3),
        "snippet": (rec.text or rec.summary or "")[:200],
    }


@_threaded_tool
def remember(content: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
             project_id: Optional[str] = None, metadata: Optional[dict] = None,
             consolidate_now: bool = False, valid_at: Optional[float] = None,
             source: Optional[str] = None) -> dict:
    """Store a durable memory in the given scope. Use this for facts worth keeping across
    conversations. The text is stored losslessly in the immutable record store; returns the
    stored memory id plus provenance. `valid_at` (unix seconds) backdates the EVENT time the
    fact became true - set it when importing history so bi-temporal reads (recall as_of,
    value_as_of, fact_history, truth_ledger) report correct validity windows; omitted means
    now. `source` labels provenance (default "user"). Needs DASHSCOPE_API_KEY (it embeds the
    text)."""
    content = _text_arg(content, "content", max_chars=_MAX_CONTENT_CHARS)
    scope = _scope(namespace, agent_id, project_id)
    try:
        eng = engine()
        rec = eng.ingest_text(content, scope=scope, consolidate_now=consolidate_now,
                              valid_at=valid_at, source=source or "user")
        if metadata:
            eng.set_metadata(rec.memory_id, metadata, scope=scope)
        return {
            "ok": True,
            "memory_id": rec.memory_id,
            "pending_consolidation": bool(rec.metadata.get("pending_consolidation")),
            "auto_sleep": eng.auto_sleep_status(scope),
            **_brief(rec),
        }
    except ModelCallError as e:
        raise RuntimeError(
            f"remember needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable remember/recall."
        )


@_threaded_tool
def remember_file(content_base64: str, filename: str, namespace: Optional[str] = None,
                  agent_id: Optional[str] = None, project_id: Optional[str] = None,
                  source: Optional[str] = None, valid_at: Optional[float] = None,
                  consolidate_now: bool = False) -> dict:
    """Store a FILE (PDF, image, screenshot, document, code) as a lossless memory: base64 bytes
    + filename in, modality-aware ingestion (OCR/description for images, text extraction for
    documents), immutable raw bytes retrievable byte-identical via get_raw. `valid_at` backdates
    the event time for bi-temporal reads. Write-path twin of get_raw. Max
    2,000,000 bytes decoded. Needs DASHSCOPE_API_KEY."""
    filename = _text_arg(filename, "filename", max_chars=512)
    encoded = _text_arg(content_base64, "content_base64", max_chars=(_MAX_RAW_BYTES * 4) // 3 + 8)
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as e:
        raise RuntimeError(f"content_base64 is not valid base64: {e}")
    if not data:
        raise RuntimeError("decoded file is empty")
    if len(data) > _MAX_RAW_BYTES:
        raise RuntimeError(f"decoded file is too large ({len(data)} bytes; max {_MAX_RAW_BYTES})")
    scope = _scope(namespace, agent_id, project_id)
    try:
        eng = engine()
        rec = eng.ingest_bytes(data, filename, source=source or filename,
                               valid_at=valid_at, scope=scope,
                               consolidate_now=consolidate_now)
        return {
            "ok": True,
            "memory_id": rec.memory_id,
            "pending_consolidation": bool(rec.metadata.get("pending_consolidation")),
            **_brief(rec),
        }
    except ModelCallError as e:
        raise RuntimeError(
            f"remember_file needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable it."
        )


@_threaded_tool
def recall(query: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
           project_id: Optional[str] = None, limit: int = 10, verify: bool = True,
           prove: bool = False, as_of: Optional[float] = None) -> dict:
    """Retrieve relevant prior memories for a query within a scope. Returns a verified answer
    plus the cited immutable sources (hash, timestamp, NLI label, score) so the calling app can
    cite them, or an explicit abstention. Set prove=True to also return a machine-readable proof
    tree. `as_of` (unix seconds) answers AS OF that past moment using the bi-temporal store:
    facts not yet valid then are invisible, superseded facts current then answer - verified
    time travel, not an LLM guess. Never confabulates. Needs DASHSCOPE_API_KEY."""
    try:
        query = _text_arg(query, "query", max_chars=_MAX_QUERY_CHARS)
        limit = _bounded_int(limit, default=10, minimum=1, maximum=_MAX_CITATION_LIMIT)
        ans = engine().ask(query, verify=True, as_of=as_of,
                           scope=_scope(namespace, agent_id, project_id))
        out = ans.model_dump()
        if isinstance(out.get("citations"), list):
            out["citations"] = out["citations"][:limit]
        if prove:
            # include recall-path metadata in the proof when RECALL_TRACE is on (the trace from
            # this ask is the freshest one); otherwise the legacy pathless proof.
            out["proof"] = engine().prove(ans, with_paths=engine().settings.recall_trace_enabled,
                                          check_refs=True)
        return out
    except ModelCallError as e:
        raise RuntimeError(
            f"recall needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable remember/recall."
        )


@_threaded_tool
def reflex_recall(query: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
                  project_id: Optional[str] = None, as_of: Optional[float] = None) -> dict:
    """LOCAL recall: the candidate memories a query activates, with their provenance (content hash,
    validity, score breakdown, co-activation paths, supersession chains), built from a derived index
    + live graph reads with NO model call -- no embedding, no NLI, no reader. Works WITHOUT a key.
    Scope-filtered. Returns RECALL (candidates), not a verified answer; use `recall` for the
    NLI-gated, cited answer. Sub-second when REFLEX_RECALL=1 (the index is maintained
    incrementally); with the flag off the index is rebuilt from the store per call (O(records),
    correct but not sub-second on a large store). Useful as a fast pre-check or a debugging/control
    view of what the engine would activate."""
    query = _text_arg(query, "query", max_chars=_MAX_QUERY_CHARS)
    packet = engine().reflex_recall(query, scope=_scope(namespace, agent_id, project_id),
                                    as_of=as_of)
    return packet.public_dict()


@_threaded_tool
def region_hints(query: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
                 project_id: Optional[str] = None, as_of: Optional[float] = None,
                 limit: int = 3, member_limit: int = 6) -> dict:
    """LOCAL memory-region/cocoon routing hints for a query, with active raw member ids, short
    content hashes, and raw URIs. No model call. Scope-filtered. These are route hints only; use
    `recall` or `get_raw` to verify source-backed answers."""
    query = _text_arg(query, "query", max_chars=_MAX_QUERY_CHARS)
    limit = _bounded_int(limit, default=3, minimum=0, maximum=20)
    member_limit = _bounded_int(member_limit, default=6, minimum=0, maximum=50)
    return engine().region_hints(
        query,
        scope=_scope(namespace, agent_id, project_id),
        as_of=as_of,
        limit=limit,
        member_limit=member_limit,
    )


@_threaded_tool
def structured_recall(query: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
                      project_id: Optional[str] = None, as_of: Optional[float] = None,
                      verify: bool = True) -> dict:
    """Run the SMQE typed memory path directly: plan -> claim backend first -> record backend
    second -> verify supports or abstain. Returns plan/backend/supports/citations. No generation;
    only verification may call the model if exact source proof is insufficient."""
    query = _text_arg(query, "query", max_chars=_MAX_QUERY_CHARS)
    return engine().structured_recall(
        query,
        scope=_scope(namespace, agent_id, project_id),
        as_of=as_of,
        verify=True,
    )


@_threaded_tool
def notebooklm_answer(question: str, notebook_id: str,
                      namespace: Optional[str] = None) -> dict:
    """UNTRUSTED NotebookLM/Gemini research draft over previously exported sources.
    Gemini generation costs 0 tokens on the caller's metered model. Returns the text only as
    `draft`, with `output_type=UNTRUSTED_DRAFT`, provenance mapping, and deterministic lexical
    grounding diagnostics. It is never a verified factual answer. Use `notebooklm_recall` for
    canonical VERIFIED-or-ABSTAINED output. Requires `nlm login`."""
    from eidetic.integrations.notebooklm import CliBackend, NotebookLMBridge, NotebookLMError
    question = _text_arg(question, "question", max_chars=_MAX_QUERY_CHARS)
    notebook_id = _text_arg(notebook_id, "notebook_id", max_chars=128)
    ns = namespace or _scope(namespace, None, None).namespace
    try:
        out = NotebookLMBridge(engine(), CliBackend()).answer(ns, question, notebook_id)
        out["draft"] = out.pop("answer", out.get("draft", ""))
        out["output_type"] = "UNTRUSTED_DRAFT"
        out["status"] = "DRAFT"
        out["verified"] = False
        return out
    except NotebookLMError as e:
        raise RuntimeError(
            f"NotebookLM read failed (no answer fabricated): {e}. Install/login the free "
            "CLI: `.venv/bin/pip install notebooklm-mcp-cli` then `nlm login`."
        )


@_threaded_tool
def notebooklm_recall(question: str, notebook_id: str, namespace: Optional[str] = None,
                      top_k: int = 6, iterative: bool = False) -> dict:
    """Governed retrieval-guided NotebookLM recall. Eidetic selects and exports the exact
    evidence set; Gemini produces an untrusted draft at 0 caller-generation tokens; the draft
    then passes through `Engine.prove_external_draft`. Returns only VERIFIED with immutable
    citations/proof or citation-free ABSTAINED. Proof-model usage is reported separately.
    `iterative=True` may widen the exported set before the same canonical proof step. Requires
    `nlm login` and an existing notebook_id."""
    from eidetic.integrations.notebooklm import CliBackend, NotebookLMBridge, NotebookLMError
    question = _text_arg(question, "question", max_chars=_MAX_QUERY_CHARS)
    notebook_id = _text_arg(notebook_id, "notebook_id", max_chars=128)
    top_k = max(1, min(int(top_k), 24))
    ns = namespace or _scope(namespace, None, None).namespace
    try:
        bridge = NotebookLMBridge(engine(), CliBackend())
        return bridge.governed_recall(
            ns,
            question,
            notebook_id,
            top_k=top_k,
            iterative=iterative,
        )
    except NotebookLMError as e:
        raise RuntimeError(
            f"NotebookLM retrieval-guided read failed (no answer fabricated): {e}. "
            "Install/login the free CLI: `.venv/bin/pip install notebooklm-mcp-cli` "
            "then `nlm login`."
        )


@_threaded_tool
def recall_routed(question: str, namespace: Optional[str] = None,
                  notebook_id: Optional[str] = None,
                  require_gate_verification: bool = False) -> dict:
    """ONE routed recall across the cost tiers, cheapest-verified first: Tier 0 reflex
    pre-filter (free) -> Tier 1 structured verify-or-abstain (cheap, gate-verified) -> then
    EITHER the free NotebookLM read (0 caller tokens, provenance-mapped, NOT gate-verified;
    needs notebook_id) OR, when require_gate_verification=True, the metered verified reader
    (the only path that runs the proof gate on a generated answer). Every result labels its
    tier, cost, and provenance_verb honestly; with no notebook_id the free tier abstains
    rather than fail or silently charge you."""
    from eidetic.integrations.notebooklm import CliBackend, NotebookLMBridge, NotebookLMError
    question = _text_arg(question, "question", max_chars=_MAX_QUERY_CHARS)
    if notebook_id is not None:
        notebook_id = _text_arg(notebook_id, "notebook_id", max_chars=128)
    ns = namespace or _scope(namespace, None, None).namespace
    try:
        return NotebookLMBridge(engine(), CliBackend()).routed_answer(
            ns, question, notebook_id,
            require_gate_verification=bool(require_gate_verification))
    except NotebookLMError as e:
        raise RuntimeError(
            f"NotebookLM tier failed (no answer fabricated): {e}. Structured/metered tiers "
            "are unaffected -- retry with require_gate_verification=True, or fix the free "
            "tier: `.venv/bin/pip install notebooklm-mcp-cli` then `nlm login`."
        )


@_threaded_tool
def truth_ledger(query: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
                 project_id: Optional[str] = None, verify: bool = True,
                 as_of: Optional[float] = None) -> dict:
    """Answer `query` and return its full TRUTH LEDGER: the complete chain from raw bytes to current
    truth. Each cited source carries its immutable hash/snippet, NLI grounding, bi-temporal validity
    window, whether it is still current, and the supersession chain of any fact it sourced (oldest
    first, closed facts retained); plus a final claim_status (verified / contradicted / abstained /
    unverified). The proof-grade 'show your work' surface. Needs DASHSCOPE_API_KEY (it answers)."""
    query = _text_arg(query, "query", max_chars=_MAX_QUERY_CHARS)
    scope = _scope(namespace, agent_id, project_id)
    try:
        ans = engine().ask(query, verify=True, as_of=as_of, scope=scope)
        return engine().truth_ledger(ans, scope=scope)
    except ModelCallError as e:
        raise RuntimeError(
            f"truth_ledger needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable it.")


@_threaded_tool
def sync_health(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """Track 2 synchronization report for a scope: whether the rebuildable surfaces (vector index,
    BM25) are consistent with the source-of-truth store, the namespace memory version, and reflex
    index status. Reports a `repair` hint (rebuild_index_from_store) when a surface is behind.
    Read-only, no key, no fabricated numbers (all counted from the live surfaces)."""
    return engine().sync_health(scope=_scope(namespace, agent_id, project_id))


@_threaded_tool
def remember_many(contents: list[str], namespace: Optional[str] = None,
                  agent_id: Optional[str] = None, project_id: Optional[str] = None,
                  valid_at: Optional[float] = None, source: Optional[str] = None) -> dict:
    """Bulk-store many durable memories in ONE call (batched embedding: N/10 round trips
    instead of N; one lock acquisition). Duplicates -- against the store AND within the
    batch -- resolve to the existing record instead of writing twice. Records are marked
    pending consolidation; run `consolidate` afterwards for graph/claim extraction.
    `valid_at` backdates every item's event time (bulk history imports). Max 500 items."""
    from .ingestion import from_text
    if not isinstance(contents, list) or not contents:
        raise RuntimeError("contents must be a non-empty list of strings")
    if len(contents) > 500:
        raise RuntimeError(f"too many items ({len(contents)}; max 500 per call)")
    items = [from_text(_text_arg(c, f"contents[{i}]", max_chars=_MAX_CONTENT_CHARS),
                       source or "user")
             for i, c in enumerate(contents)]
    scope = _scope(namespace, agent_id, project_id)
    try:
        recs = engine().ingest_many(items, scope=scope, valid_at=valid_at)
    except ModelCallError as e:
        raise RuntimeError(
            f"remember_many needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable it.")
    ids = [r.memory_id for r in recs]
    return {
        "ok": True,
        "count": len(recs),
        "unique": len(set(ids)),
        "deduped": len(recs) - len(set(ids)),
        "memory_ids": ids,
        "pending_consolidation": any(r.metadata.get("pending_consolidation") for r in recs),
    }


@_threaded_tool
def remember_problem(goal: str, namespace: Optional[str] = None,
                     agent_id: Optional[str] = None, project_id: Optional[str] = None,
                     status: str = "open", blockers: Optional[list[str]] = None,
                     valid_at: Optional[float] = None) -> dict:
    """Open a shared PROBLEM record (war-room memory): a goal with status, blockers,
    hypotheses, handoffs, and decisions that any agent in the scope can read and extend.
    Every change is an immutable bitemporal revision -- the full investigation history is
    replayable with as_of. Returns the problem_id used by the other problem tools."""
    from . import problems
    goal = _text_arg(goal, "goal", max_chars=_MAX_QUERY_CHARS)
    return problems.remember_problem(engine(), goal, scope=_scope(namespace, agent_id, project_id),
                                     status=status, blockers=blockers, valid_at=valid_at)


@_threaded_tool
def update_problem(problem_id: str, namespace: Optional[str] = None,
                   agent_id: Optional[str] = None, project_id: Optional[str] = None,
                   status: Optional[str] = None, blockers: Optional[list[str]] = None,
                   handoffs: Optional[list[str]] = None,
                   decisions: Optional[list[dict]] = None,
                   valid_at: Optional[float] = None) -> dict:
    """Append a revision to a problem: status change, new blockers, a handoff note, or a
    decision ({choice, rationale, witnesses}). Nothing is mutated -- the update is a new
    immutable record and the folded current state comes back."""
    from . import problems
    return problems.update_problem(engine(), problem_id,
                                   scope=_scope(namespace, agent_id, project_id),
                                   status=status, blockers=blockers, handoffs=handoffs,
                                   decisions=decisions, valid_at=valid_at)


@_threaded_tool
def add_hypothesis(problem_id: str, claim: str, namespace: Optional[str] = None,
                   agent_id: Optional[str] = None, project_id: Optional[str] = None,
                   evidence: Optional[list[str]] = None,
                   valid_at: Optional[float] = None) -> dict:
    """Attach a hypothesis to a problem. `evidence` is a list of memory_ids already in
    this scope (validated -- a ref to a foreign or missing memory fails loud), so the
    hypothesis is provable through the same citation machinery as every answer."""
    from . import problems
    return problems.add_hypothesis(engine(), problem_id,
                                   _text_arg(claim, "claim", max_chars=_MAX_QUERY_CHARS),
                                   scope=_scope(namespace, agent_id, project_id),
                                   evidence=evidence, valid_at=valid_at)


@_threaded_tool
def resolve_hypothesis(problem_id: str, hypothesis_id: str, status: str,
                       namespace: Optional[str] = None, agent_id: Optional[str] = None,
                       project_id: Optional[str] = None, rationale: str = "",
                       evidence: Optional[list[str]] = None,
                       valid_at: Optional[float] = None) -> dict:
    """Mark a hypothesis supported/refuted/confirmed with a rationale and optional new
    evidence refs. The old status stays in history (bitemporal); latest resolution wins
    in the folded state."""
    from . import problems
    return problems.resolve_hypothesis(engine(), problem_id, hypothesis_id, status,
                                       scope=_scope(namespace, agent_id, project_id),
                                       rationale=rationale, evidence=evidence,
                                       valid_at=valid_at)


@_threaded_tool
def ask_problem(problem_id: str, question: str, namespace: Optional[str] = None,
                agent_id: Optional[str] = None, project_id: Optional[str] = None,
                as_of: Optional[float] = None) -> dict:
    """Ask a natural-language question against a problem's war-room history ('what did we
    decide about the pool size and why?') through the same verify-or-abstain path as
    every recall: the answer arrives with citations, each marked revision-backed when it
    points into this problem's own revision records, plus the folded state. `as_of`
    replays both the answer and the state at a past moment. Needs DASHSCOPE_API_KEY."""
    from . import problems
    try:
        return problems.ask_problem(engine(), problem_id,
                                    _text_arg(question, "question", max_chars=_MAX_QUERY_CHARS),
                                    scope=_scope(namespace, agent_id, project_id),
                                    as_of=as_of)
    except ModelCallError as e:
        raise RuntimeError(
            f"ask_problem needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable it.")


@_threaded_tool
def add_witness(problem_id: str, path: str, namespace: Optional[str] = None,
                agent_id: Optional[str] = None, project_id: Optional[str] = None,
                note: str = "", valid_at: Optional[float] = None) -> dict:
    """Attach an operational-truth WITNESS file (screenshot, log, dump) to a problem.
    The bytes land losslessly in the content-addressed substrate (get_raw returns them
    byte-identical, hash-checked); the problem's folded state lists the witness with its
    content hash so hypotheses and answers citing it are checkable down to raw bytes."""
    from . import problems
    return problems.add_witness(engine(), problem_id, path,
                                scope=_scope(namespace, agent_id, project_id),
                                note=note, valid_at=valid_at)


@_threaded_tool
def recall_problem(problem_id: Optional[str] = None, query: str = "",
                   namespace: Optional[str] = None, agent_id: Optional[str] = None,
                   project_id: Optional[str] = None,
                   as_of: Optional[float] = None) -> dict:
    """Current folded state of a problem (goal, status, blockers, hypotheses with
    evidence refs, handoffs, decisions), by id or by query match on the goal. `as_of`
    replays the state as it stood at a past moment. Read-only, no model call."""
    from . import problems
    state = problems.recall_problem(engine(), problem_id=problem_id, query=query,
                                    scope=_scope(namespace, agent_id, project_id),
                                    as_of=as_of)
    if state is None:
        raise RuntimeError("no matching problem in this scope")
    return state


@_threaded_tool
def repair() -> dict:
    """Rebuild the derived retrieval surfaces (vector index, reflex index) from the source of
    truth (raw substrate + SQLite records) -- the fix sync_health names when a surface is
    behind or corrupt. No data loss: raw records are never touched. COSTS embedding calls
    (re-embeds every record's text), so run it on sync_health's advice, not routinely."""
    return engine().rebuild_index_from_store()


@_threaded_tool
def consolidate(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """Run the unified sleep cycle for a scope: consolidate_pending (so pending fast writes flow
    into graph/events/gists) THEN the token-free dream pass (replay, link inference, multi-res
    gist). Returns per-phase counts only, no fabricated metrics. Token-free + key-free when nothing
    is pending; never deletes a raw record. (Alias of `sleep`; both call the one lifecycle path.)"""
    return engine().sleep(scope=_scope(namespace, agent_id, project_id))


@_threaded_tool
def list_memories(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                  project_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> dict:
    """List stored memories within a scope (newest first), paginated. Read-only; works without a
    key. Returns ids, short previews, timestamps, salience, and FSRS retrievability."""
    recs = engine().list_memories(_scope(namespace, agent_id, project_id))
    total = len(recs)
    offset = _bounded_int(offset, default=0, minimum=0, maximum=max(0, total))
    limit = _bounded_int(limit, default=_DEFAULT_PAGE_LIMIT, minimum=1, maximum=_MAX_PAGE_LIMIT)
    page = recs[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit,
            "memories": [_brief(r) for r in page]}


@_threaded_tool
def get_raw(memory_id: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
            project_id: Optional[str] = None, max_bytes: int = _DEFAULT_RAW_MAX_BYTES,
            offset: int = 0) -> dict:
    """Return the IMMUTABLE raw record for a memory id verbatim (the show-your-work tool that
    proves no confabulation). Scope-filtered: an id from another namespace is invisible. Read-only;
    works without a key. Raw bytes are returned exactly as stored, never paraphrased; very large
    records are returned in bounded byte ranges with truncation metadata."""
    scope = _scope(namespace, agent_id, project_id)
    rec = engine().get_record_in_scope(memory_id, scope)
    if rec is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    raw, raw_meta = _bounded_raw(engine().get_raw(rec.content_hash),
                                 offset=offset, max_bytes=max_bytes)
    try:
        raw_repr, encoding = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        raw_repr, encoding = base64.b64encode(raw).decode("ascii"), "base64"
    return {
        "memory_id": rec.memory_id,
        "content_hash": rec.content_hash,
        "raw_uri": rec.raw_uri,
        "raw_encoding": encoding,
        "raw": raw_repr,
        **raw_meta,
        "provenance": {
            "source": rec.source, "modality": rec.modality.value,
            "scope": rec.scope.model_dump(), "valid_at": rec.valid_at,
            "created_at": rec.created_at, "invalid_at": rec.invalid_at,
            "expired_at": rec.expired_at, "is_described": rec.is_described,
        },
        "verified_against_self": engine().substrate.verify(rec.content_hash),
    }


@_threaded_tool
def forget(memory_id: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
           project_id: Optional[str] = None) -> dict:
    """Lower a memory's retrieval PRIORITY via the FSRS forgetting path. This is priority decay,
    NOT deletion: the immutable raw record is never deleted and can be brought back with reawaken.
    Scope-filtered. Works without a key (no model call)."""
    scope = _scope(namespace, agent_id, project_id)
    if engine().get_record_in_scope(memory_id, scope) is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    rec = engine().forget(memory_id)
    return {"ok": True, "note": "priority decayed; raw record NOT deleted", **_brief(rec)}


@_threaded_tool
def reawaken(memory_id: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
             project_id: Optional[str] = None) -> dict:
    """Re-promote a down-weighted memory (reset retrievability, boost stability). The inverse of
    forget. Scope-filtered. Works without a key."""
    scope = _scope(namespace, agent_id, project_id)
    if engine().get_record_in_scope(memory_id, scope) is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    rec = engine().reawaken(memory_id)
    return {"ok": True, **_brief(rec)}


@_threaded_tool
def stats(namespace: Optional[str] = None, agent_id: Optional[str] = None,
          project_id: Optional[str] = None) -> dict:
    """Scope-level counts: number of memories, edges, indexed vectors, backend, key presence.
    Read-only, no fabricated numbers, works without a key."""
    return engine().stats(scope=_scope(namespace, agent_id, project_id))


@_threaded_tool
def health_report(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                  project_id: Optional[str] = None) -> dict:
    """Read-only self-diagnosis of a scope: coverage, contradiction load, low-confidence and
    inferred facts, derived/replay debt, orphan records, and age spread. Every figure is counted
    from the store, never fabricated. Works without a key."""
    return engine().memory_health_report(scope=_scope(namespace, agent_id, project_id))


@_threaded_tool
def value_as_of(entity: str, relation: str, as_of: Optional[float] = None,
                namespace: Optional[str] = None, agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """C2 time-travel: the DETERMINISTIC value of (entity, relation) valid at unix time `as_of`
    (now if omitted), chosen from the bi-temporal graph -- not an LLM guess. Read-only, no key.
    Answers 'where did Alice work on date X', which vector-only systems cannot."""
    out = engine().value_as_of(entity, relation, as_of=as_of,
                               scope=_scope(namespace, agent_id, project_id))
    return out if out is not None else {"value": None, "note": "no fact valid as of that time"}


@_threaded_tool
def fact_history(entity: str, relation: str, namespace: Optional[str] = None,
                 agent_id: Optional[str] = None, project_id: Optional[str] = None) -> dict:
    """C2 current-vs-historical: the full superseded chain for (entity, relation), oldest first,
    each with its validity window (closed facts retained, never deleted). Read-only, no key."""
    return {"history": engine().fact_history(entity, relation,
                                             scope=_scope(namespace, agent_id, project_id))}


@_threaded_tool
def integrity_report(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                     project_id: Optional[str] = None) -> dict:
    """C1 operation-level integrity: fabrication / abstention / verified rates + conflict load,
    counted from the BrainEvent stream + the store (never fabricated). Read-only, no key."""
    return engine().integrity_report(scope=_scope(namespace, agent_id, project_id))


@_threaded_tool
def scratchpad(namespace: Optional[str] = None, agent_id: Optional[str] = None,
               project_id: Optional[str] = None) -> dict:
    """The working scratchpad for a scope: high-salience, verified, ACTIVE facts, each linked to its
    immutable source hash. A quick-recall context channel, NOT a source of truth (superseded facts
    expire automatically). Read-only, no key."""
    return {"scratchpad": engine().build_scratchpad(scope=_scope(namespace, agent_id, project_id))}


@_threaded_tool
def preference_profile(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                       project_id: Optional[str] = None, include_inactive: bool = False,
                       limit: int = 50) -> dict:
    """Read the current preference profile for a scope, with source memory ids, content hashes,
    and raw URIs. Superseded profile history is hidden unless include_inactive=True. Read-only,
    no key."""
    limit = _bounded_int(limit, default=50, minimum=0, maximum=500)
    return engine().preference_profile(
        scope=_scope(namespace, agent_id, project_id),
        include_inactive=include_inactive,
        limit=limit,
    )


@_threaded_tool
def why_remembered(memory_id: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
                   project_id: Optional[str] = None) -> dict:
    """'Why I remember this strongly': the affect/usage components behind a memory's salience
    (importance, arousal, valence, surprise, emphasis, verified-helpful count) plus immutable
    source provenance. The explanation is hedged and non-clinical. Scope-filtered, read-only,
    no key."""
    out = engine().salience_explanation(memory_id, scope=_scope(namespace, agent_id, project_id))
    if out is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    return out


@_threaded_tool
def preflight() -> dict:
    """Run the preflight doctor: one real call per capability (embed / chat / rerank / multimodal /
    document) against the configured model IDs, reporting pass/fail + latency and telling a quota
    block apart from a dead key/model. Needs DASHSCOPE_API_KEY (makes real calls); without one it
    reports every capability as skipped:no_key -- never a fake pass."""
    from .doctor import preflight as _preflight
    return _preflight(engine())


@_threaded_tool
def brain_health_score(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                       project_id: Optional[str] = None) -> dict:
    """A local BrainHealthScore in [0,1] for a scope plus its components (recall connectivity,
    proof coverage, temporal coverage, channel diversity, orphan/contradiction/stale-gist debt).
    A diagnostic rollup of counted store + event figures, NOT a benchmark. Read-only, no key."""
    return engine().brain_health_score(scope=_scope(namespace, agent_id, project_id))


@_threaded_tool
def sleep(namespace: Optional[str] = None, agent_id: Optional[str] = None,
          project_id: Optional[str] = None, llm_summaries: bool = False) -> dict:
    """Run the unified sleep cycle for a scope: consolidate_pending -> dream -> optional LLM
    summaries (the same path the HTTP API uses, so both transports behave identically). Token-free
    and key-free when nothing is pending and llm_summaries is False; never deletes a raw record."""
    return engine().sleep(scope=_scope(namespace, agent_id, project_id),
                          llm_summaries=llm_summaries)


@_threaded_tool
def memory_autopsy(question: str, namespace: Optional[str] = None, agent_id: Optional[str] = None,
                   project_id: Optional[str] = None) -> dict:
    """Read-only diagnosis of WHY a question would miss in a scope (missing write, pending
    consolidation, entity-extraction failure, event-normalization failure, vector underfill, or a
    retrieval/reader failure), with a suggested repair. Counted from the store; no key needed."""
    question = _text_arg(question, "question", max_chars=_MAX_QUERY_CHARS)
    return engine().memory_autopsy(question, scope=_scope(namespace, agent_id, project_id))


@_threaded_tool
def recall_trace(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                 project_id: Optional[str] = None) -> dict:
    """The RecallTrace from the most recent traced recall IN THIS SCOPE: which channels fired,
    their weights, fused scores, and stage latency. Returns {} unless RECALL_TRACE is enabled or
    no traced recall has run in the scope. Scope-filtered like every tool - another namespace's
    trace (its query text and memory ids) is never visible. Read-only, no key. This is the 'why
    did recall find/miss that?' introspection surface."""
    t = engine().recall_trace(scope=_scope(namespace, agent_id, project_id))
    return t.model_dump() if t is not None else {}


@_threaded_tool
def prove_age_independence(namespace: Optional[str] = None, agent_id: Optional[str] = None,
                           project_id: Optional[str] = None, k: int = 5) -> dict:
    """Compute recall@k and p95 latency vs memory AGE on the current scope and report the slopes
    (flat = age-independent recall, the signature property). Needs DASHSCOPE_API_KEY (it embeds
    partial cues for the probe). Mirrors the HTTP /api/prove_age_independence route."""
    try:
        k = _bounded_int(k, default=5, minimum=1, maximum=_MAX_PROBE_K)
        return engine().prove_age_independence(
            scope=_scope(namespace, agent_id, project_id), k=k)
    except ModelCallError as e:
        raise RuntimeError(
            f"prove_age_independence needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eidetic-plus", description="Eidetic-Plus MCP memory server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                        help="stdio (default; local hosts) or http (remote/shared, streamable-http)")
    parser.add_argument("--http-port", type=int, default=None,
                        help="port for --transport http (default from EIDETIC_MCP_PORT or 8765)")
    parser.add_argument("--http", action="store_true",
                        help="deprecated alias for --transport http")
    args = parser.parse_args()
    transport = "http" if args.http else args.transport
    if args.http_port is not None:
        try:
            mcp.settings.port = args.http_port
        except Exception:
            pass
    mcp.run(transport="streamable-http" if transport == "http" else "stdio")


if __name__ == "__main__":
    main()
