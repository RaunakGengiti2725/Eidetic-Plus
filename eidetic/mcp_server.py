"""Eidetic-Plus as a UNIVERSAL MEMORY PLUGIN — an MCP server over the same Engine.

This is the headline: any MCP host (Claude, Claude Code, Cursor, Cline, Zed, ...) can
mount Eidetic-Plus as its memory backend with zero per-tool integration. The MCP server
is an ADDITIONAL transport over the exact same `engine.py` that FastAPI uses — no logic
is duplicated. Supports stdio (local hosts) and streamable-http (remote/shared).

SCOPING is enforced on every tool: a required `namespace` plus optional `agent_id` /
`project_id`. Reads filter by scope; writes tag with scope. A memory written by Claude
Code in namespace A is invisible to Cursor reading namespace B — no cross-tool bleed.
The default namespace is the explicit string "default", never a global wildcard.

Fail-loud: a missing DASHSCOPE_API_KEY raises a clear error (surfaced as an MCP tool
error), never a fabricated result.

Run:
    python -m eidetic.mcp_server            # stdio (default; for Claude Code / Cursor)
    python -m eidetic.mcp_server --http     # streamable-http on EIDETIC_MCP_HOST:PORT
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
        "Lossless, verifiable, recency-independent memory for AI agents. Every tool "
        "takes a `namespace` (default 'default') plus optional `agent_id`/`project_id`; "
        "reads NEVER cross namespaces, so use a stable namespace per project/agent. "
        "`recall` returns answers VERIFIED against an immutable record (it abstains "
        "rather than confabulate). `get_raw` returns that immutable source so you can "
        "prove provenance. `prove_age_independence` shows recall/latency are flat vs age."
    ),
)

_engine: Optional[Engine] = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def _scope(namespace: str, agent_id: Optional[str], project_id: Optional[str]) -> Scope:
    return Scope(namespace=namespace or "default",
                 agent_id=agent_id or None, project_id=project_id or None)


def _brief(rec) -> dict:
    return {
        "memory_id": rec.memory_id,
        "content_hash": rec.content_hash,
        "raw_uri": rec.raw_uri,
        "modality": rec.modality.value,
        "source": rec.source,
        "scope": rec.scope.model_dump(),
        "valid_at": rec.valid_at,
        "salience": round(rec.salience, 3),
        "retrievability": round(rec.fsrs.priority(), 3),
        "snippet": (rec.text or rec.summary or "")[:200],
    }


@mcp.tool()
def remember(text: str, namespace: str = "default", agent_id: Optional[str] = None,
             project_id: Optional[str] = None, source: str = "agent",
             valid_at: Optional[float] = None, extract_graph: bool = True,
             segment: bool = False) -> dict:
    """Store a memory (text) losslessly and immutably in the given scope. Returns the
    content hash + provenance. Set segment=True to split long input at surprise boundaries."""
    try:
        rec = engine().ingest_text(text, source=source, valid_at=valid_at,
                                   extract_graph=extract_graph,
                                   scope=_scope(namespace, agent_id, project_id), segment=segment)
        return _brief(rec)
    except ModelCallError as e:
        raise RuntimeError(f"Model call failed (no result fabricated): {e}")


@mcp.tool()
def recall(query: str, namespace: str = "default", agent_id: Optional[str] = None,
           project_id: Optional[str] = None, verify: bool = True) -> dict:
    """Verified retrieval within a scope. Returns the answer, cited immutable sources
    (hash + timestamp + NLI label + score), and confidence — or an explicit abstention.
    Never confabulates: unentailed answers are flagged unverified."""
    try:
        ans = engine().ask(query, verify=verify, scope=_scope(namespace, agent_id, project_id))
        return ans.model_dump()
    except ModelCallError as e:
        raise RuntimeError(f"Model call failed (no result fabricated): {e}")


@mcp.tool()
def consolidate(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None) -> dict:
    """Trigger the sleep loop for a scope: dedup, verified semantic summaries, FSRS
    index-priority decay. Never deletes a raw record."""
    try:
        return engine().consolidate(scope=_scope(namespace, agent_id, project_id))
    except ModelCallError as e:
        raise RuntimeError(f"Model call failed (no result fabricated): {e}")


@mcp.tool()
def reawaken(memory_id: str) -> dict:
    """Re-promote a down-weighted memory (the O(1) revert path: reset retrievability,
    boost stability). Forgetting only down-weighted it; the raw record was never deleted."""
    rec = engine().reawaken(memory_id)
    if rec is None:
        raise RuntimeError(f"No such memory: {memory_id}")
    return _brief(rec)


@mcp.tool()
def list_memories(namespace: str = "default", agent_id: Optional[str] = None,
                  project_id: Optional[str] = None) -> list[dict]:
    """List memories within a scope (newest first) with salience and FSRS retrievability."""
    recs = engine().list_memories(_scope(namespace, agent_id, project_id))
    return [_brief(r) for r in recs]


@mcp.tool()
def get_raw(memory_id: str) -> dict:
    """Return the IMMUTABLE raw record + hash + full provenance for a memory — the
    'show your work' tool that proves no confabulation. The raw bytes are ground truth."""
    rec = engine().get_record(memory_id)
    if rec is None:
        raise RuntimeError(f"No such memory: {memory_id}")
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
def prove_age_independence(namespace: str = "default", agent_id: Optional[str] = None,
                           project_id: Optional[str] = None, k: int = 5) -> dict:
    """Compute recall@k and p95 retrieval latency vs memory AGE on the current store,
    on demand. The headline claim, callable live: both come back flat (slopes ~0)."""
    try:
        return engine().prove_age_independence(
            scope=_scope(namespace, agent_id, project_id), k=k)
    except ModelCallError as e:
        raise RuntimeError(f"Model call failed (no result fabricated): {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eidetic-Plus MCP server")
    parser.add_argument("--http", action="store_true",
                        help="serve streamable-http instead of stdio")
    args = parser.parse_args()
    mcp.run(transport="streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
