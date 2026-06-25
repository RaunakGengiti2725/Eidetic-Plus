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
    python -m eidetic.mcp_server --transport http --http-port 8765   # remote / shared
"""
from __future__ import annotations

import argparse
import base64
import os
import threading
from typing import Optional

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
        "Persistent, lossless, verifiable long-term memory for AI agents. Every tool takes a "
        "`namespace` (default 'default') plus optional `agent_id` / `project_id`; reads never "
        "cross namespaces, so use a stable namespace per project or agent. Use `remember` for "
        "durable facts worth keeping, `recall` to retrieve prior context with cited sources, "
        "and `get_raw` to verify a source against the immutable record."
    ),
)

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


@mcp.tool()
def remember(content: str, namespace: str = "default", agent_id: Optional[str] = None,
             project_id: Optional[str] = None, metadata: Optional[dict] = None,
             consolidate_now: bool = False) -> dict:
    """Store a durable memory in the given scope. Use this for facts worth keeping across
    conversations. The text is stored losslessly in the immutable record store; returns the
    stored memory id plus provenance. Needs DASHSCOPE_API_KEY (it embeds the text)."""
    scope = _scope(namespace, agent_id, project_id)
    try:
        rec = engine().ingest_text(content, scope=scope, consolidate_now=consolidate_now)
        if metadata:
            engine().set_metadata(rec.memory_id, metadata, scope=scope)
        return {"ok": True, "memory_id": rec.memory_id, **_brief(rec)}
    except ModelCallError as e:
        raise RuntimeError(
            f"remember needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable remember/recall."
        )


@mcp.tool()
def recall(query: str, namespace: str = "default", agent_id: Optional[str] = None,
           project_id: Optional[str] = None, limit: int = 10, verify: bool = True,
           prove: bool = False) -> dict:
    """Retrieve relevant prior memories for a query within a scope. Returns a verified answer
    plus the cited immutable sources (hash, timestamp, NLI label, score) so the calling app can
    cite them, or an explicit abstention. Set prove=True to also return a machine-readable proof
    tree. Never confabulates. Needs DASHSCOPE_API_KEY."""
    try:
        ans = engine().ask(query, verify=verify, scope=_scope(namespace, agent_id, project_id))
        out = ans.model_dump()
        if isinstance(out.get("citations"), list) and limit and limit > 0:
            out["citations"] = out["citations"][:limit]
        if prove:
            # include recall-path metadata in the proof when RECALL_TRACE is on (the trace from
            # this ask is the freshest one); otherwise the legacy pathless proof.
            out["proof"] = engine().prove(ans, with_paths=engine().settings.recall_trace_enabled)
        return out
    except ModelCallError as e:
        raise RuntimeError(
            f"recall needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable remember/recall."
        )


@mcp.tool()
def reflex_recall(query: str, namespace: str = "default", agent_id: Optional[str] = None,
                  project_id: Optional[str] = None, as_of: Optional[float] = None) -> dict:
    """LOCAL recall: the candidate memories a query activates, with their provenance (content hash,
    validity, score breakdown, co-activation paths, supersession chains), built from a derived index
    + live graph reads with NO model call -- no embedding, no NLI, no reader. Works WITHOUT a key.
    Scope-filtered. Returns RECALL (candidates), not a verified answer; use `recall` for the
    NLI-gated, cited answer. Sub-second when REFLEX_RECALL=1 (the index is maintained
    incrementally); with the flag off the index is rebuilt from the store per call (O(records),
    correct but not sub-second on a large store). Useful as a fast pre-check or a debugging/control
    view of what the engine would activate."""
    packet = engine().reflex_recall(query, scope=_scope(namespace, agent_id, project_id),
                                    as_of=as_of)
    return packet.public_dict()


@mcp.tool()
def sync_health(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """Track 2 synchronization report for a scope: whether the rebuildable surfaces (vector index,
    BM25) are consistent with the source-of-truth store, the namespace memory version, and reflex
    index status. Reports a `repair` hint (rebuild_index_from_store) when a surface is behind.
    Read-only, no key, no fabricated numbers (all counted from the live surfaces)."""
    return engine().sync_health(scope=_scope(namespace, agent_id, project_id))


@mcp.tool()
def consolidate(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """Run the unified sleep cycle for a scope: consolidate_pending (so pending fast writes flow
    into graph/events/gists) THEN the token-free dream pass (replay, link inference, multi-res
    gist). Returns per-phase counts only, no fabricated metrics. Token-free + key-free when nothing
    is pending; never deletes a raw record. (Alias of `sleep`; both call the one lifecycle path.)"""
    return engine().sleep(scope=_scope(namespace, agent_id, project_id))


@mcp.tool()
def list_memories(namespace: str = "default", agent_id: Optional[str] = None,
                  project_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> dict:
    """List stored memories within a scope (newest first), paginated. Read-only; works without a
    key. Returns ids, short previews, timestamps, salience, and FSRS retrievability."""
    recs = engine().list_memories(_scope(namespace, agent_id, project_id))
    total = len(recs)
    offset = max(0, int(offset))
    page = recs[offset: offset + max(1, int(limit))]
    return {"total": total, "offset": offset, "limit": limit,
            "memories": [_brief(r) for r in page]}


@mcp.tool()
def get_raw(memory_id: str, namespace: str = "default", agent_id: Optional[str] = None,
            project_id: Optional[str] = None) -> dict:
    """Return the IMMUTABLE raw record for a memory id verbatim (the show-your-work tool that
    proves no confabulation). Scope-filtered: an id from another namespace is invisible. Read-only;
    works without a key. The raw bytes are returned exactly as stored, never paraphrased."""
    scope = _scope(namespace, agent_id, project_id)
    rec = engine().get_record_in_scope(memory_id, scope)
    if rec is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    raw = engine().get_raw(rec.content_hash)
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
        "provenance": {
            "source": rec.source, "modality": rec.modality.value,
            "scope": rec.scope.model_dump(), "valid_at": rec.valid_at,
            "created_at": rec.created_at, "invalid_at": rec.invalid_at,
            "expired_at": rec.expired_at, "is_described": rec.is_described,
        },
        "verified_against_self": engine().substrate.verify(rec.content_hash),
    }


@mcp.tool()
def forget(memory_id: str, namespace: str = "default", agent_id: Optional[str] = None,
           project_id: Optional[str] = None) -> dict:
    """Lower a memory's retrieval PRIORITY via the FSRS forgetting path. This is priority decay,
    NOT deletion: the immutable raw record is never deleted and can be brought back with reawaken.
    Scope-filtered. Works without a key (no model call)."""
    scope = _scope(namespace, agent_id, project_id)
    if engine().get_record_in_scope(memory_id, scope) is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    rec = engine().forget(memory_id)
    return {"ok": True, "note": "priority decayed; raw record NOT deleted", **_brief(rec)}


@mcp.tool()
def reawaken(memory_id: str, namespace: str = "default", agent_id: Optional[str] = None,
             project_id: Optional[str] = None) -> dict:
    """Re-promote a down-weighted memory (reset retrievability, boost stability). The inverse of
    forget. Scope-filtered. Works without a key."""
    scope = _scope(namespace, agent_id, project_id)
    if engine().get_record_in_scope(memory_id, scope) is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    rec = engine().reawaken(memory_id)
    return {"ok": True, **_brief(rec)}


@mcp.tool()
def stats(namespace: str = "default", agent_id: Optional[str] = None,
          project_id: Optional[str] = None) -> dict:
    """Scope-level counts: number of memories, edges, indexed vectors, backend, key presence.
    Read-only, no fabricated numbers, works without a key."""
    return engine().stats(scope=_scope(namespace, agent_id, project_id))


@mcp.tool()
def health_report(namespace: str = "default", agent_id: Optional[str] = None,
                  project_id: Optional[str] = None) -> dict:
    """Read-only self-diagnosis of a scope: coverage, contradiction load, low-confidence and
    inferred facts, derived/replay debt, orphan records, and age spread. Every figure is counted
    from the store, never fabricated. Works without a key."""
    return engine().memory_health_report(scope=_scope(namespace, agent_id, project_id))


@mcp.tool()
def value_as_of(entity: str, relation: str, as_of: Optional[float] = None,
                namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """C2 time-travel: the DETERMINISTIC value of (entity, relation) valid at unix time `as_of`
    (now if omitted), chosen from the bi-temporal graph -- not an LLM guess. Read-only, no key.
    Answers 'where did Alice work on date X', which vector-only systems cannot."""
    out = engine().value_as_of(entity, relation, as_of=as_of,
                               scope=_scope(namespace, agent_id, project_id))
    return out if out is not None else {"value": None, "note": "no fact valid as of that time"}


@mcp.tool()
def fact_history(entity: str, relation: str, namespace: str = "default",
                 agent_id: Optional[str] = None, project_id: Optional[str] = None) -> dict:
    """C2 current-vs-historical: the full superseded chain for (entity, relation), oldest first,
    each with its validity window (closed facts retained, never deleted). Read-only, no key."""
    return {"history": engine().fact_history(entity, relation,
                                             scope=_scope(namespace, agent_id, project_id))}


@mcp.tool()
def integrity_report(namespace: str = "default", agent_id: Optional[str] = None,
                     project_id: Optional[str] = None) -> dict:
    """C1 operation-level integrity: fabrication / abstention / verified rates + conflict load,
    counted from the BrainEvent stream + the store (never fabricated). Read-only, no key."""
    return engine().integrity_report(scope=_scope(namespace, agent_id, project_id))


@mcp.tool()
def scratchpad(namespace: str = "default", agent_id: Optional[str] = None,
               project_id: Optional[str] = None) -> dict:
    """The working scratchpad for a scope: high-salience, verified, ACTIVE facts, each linked to its
    immutable source hash. A quick-recall context channel, NOT a source of truth (superseded facts
    expire automatically). Read-only, no key."""
    return {"scratchpad": engine().build_scratchpad(scope=_scope(namespace, agent_id, project_id))}


@mcp.tool()
def why_remembered(memory_id: str, namespace: str = "default", agent_id: Optional[str] = None,
                   project_id: Optional[str] = None) -> dict:
    """'Why I remember this strongly': the affect/usage components behind a memory's salience
    (importance, arousal, valence, surprise, emphasis, verified-helpful count) plus its provenance.
    Scope-filtered, read-only, no key."""
    out = engine().salience_explanation(memory_id, scope=_scope(namespace, agent_id, project_id))
    if out is None:
        raise RuntimeError(f"No such memory in scope: {memory_id}")
    return out


@mcp.tool()
def preflight() -> dict:
    """Run the preflight doctor: one real call per capability (embed / chat / rerank / multimodal /
    document) against the configured model IDs, reporting pass/fail + latency and telling a quota
    block apart from a dead key/model. Needs DASHSCOPE_API_KEY (makes real calls); without one it
    reports every capability as skipped:no_key -- never a fake pass."""
    from .doctor import preflight as _preflight
    return _preflight(engine())


@mcp.tool()
def brain_health_score(namespace: str = "default", agent_id: Optional[str] = None,
                       project_id: Optional[str] = None) -> dict:
    """A local BrainHealthScore in [0,1] for a scope plus its components (recall connectivity,
    proof coverage, temporal coverage, channel diversity, orphan/contradiction/stale-gist debt).
    A diagnostic rollup of counted store + event figures, NOT a benchmark. Read-only, no key."""
    return engine().brain_health_score(scope=_scope(namespace, agent_id, project_id))


@mcp.tool()
def sleep(namespace: str = "default", agent_id: Optional[str] = None,
          project_id: Optional[str] = None, llm_summaries: bool = False) -> dict:
    """Run the unified sleep cycle for a scope: consolidate_pending -> dream -> optional LLM
    summaries (the same path the HTTP API uses, so both transports behave identically). Token-free
    and key-free when nothing is pending and llm_summaries is False; never deletes a raw record."""
    return engine().sleep(scope=_scope(namespace, agent_id, project_id),
                          llm_summaries=llm_summaries)


@mcp.tool()
def memory_autopsy(question: str, namespace: str = "default", agent_id: Optional[str] = None,
                   project_id: Optional[str] = None) -> dict:
    """Read-only diagnosis of WHY a question would miss in a scope (missing write, pending
    consolidation, entity-extraction failure, event-normalization failure, vector underfill, or a
    retrieval/reader failure), with a suggested repair. Counted from the store; no key needed."""
    return engine().memory_autopsy(question, scope=_scope(namespace, agent_id, project_id))


@mcp.tool()
def recall_trace() -> dict:
    """The RecallTrace from the most recent traced recall: which channels fired, their weights,
    fused scores, and stage latency. Returns {} unless RECALL_TRACE is enabled. Read-only, no key.
    This is the 'why did recall find/miss that?' introspection surface."""
    t = engine().recall_trace()
    return t.model_dump() if t is not None else {}


@mcp.tool()
def prove_age_independence(namespace: str = "default", agent_id: Optional[str] = None,
                           project_id: Optional[str] = None, k: int = 5) -> dict:
    """Compute recall@k and p95 latency vs memory AGE on the current scope and report the slopes
    (flat = age-independent recall, the signature property). Needs DASHSCOPE_API_KEY (it embeds
    partial cues for the probe). Mirrors the HTTP /api/prove_age_independence route."""
    try:
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
