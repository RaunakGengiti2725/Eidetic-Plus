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


def engine() -> Engine:
    global _engine
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
            out["proof"] = engine().prove(ans)
        return out
    except ModelCallError as e:
        raise RuntimeError(
            f"recall needs the model and no result was fabricated: {e}. "
            "Set DASHSCOPE_API_KEY to enable remember/recall."
        )


@mcp.tool()
def consolidate(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """Run the token-free dreaming/consolidation pass for a scope (replay, link inference,
    multi-resolution gist). Returns per-phase counts only, no fabricated metrics. This is the
    free, no-key consolidation path; it never deletes a raw record."""
    return engine().dream(scope=_scope(namespace, agent_id, project_id))


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
