"""FastAPI app: the agent API (HTTP transport) + a static web UI.

The MCP server (eidetic/mcp_server.py) is an ADDITIONAL transport over the same
engine. Both talk to one Engine; no logic is duplicated.

Every memory-touching route accepts an explicit SCOPE (namespace + optional agent_id /
project_id). Reads filter by scope, writes tag with scope -- a write in namespace A is
invisible from namespace B. The app starts WITHOUT a key; any model call fails loudly
with a clear 503 if DASHSCOPE_API_KEY is missing -- never a fabricated result.
"""
from __future__ import annotations

import mimetypes
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .dashscope_client import ModelCallError
from .engine import Engine
from .graph import CO_ACTIVATED
from .models import Scope

app = FastAPI(title="Eidetic-Plus", version="1.1.0",
              description="Universal, lossless, verifiable, recency-independent memory for AI agents.")

_engine: Optional[Engine] = None
_engine_lock = threading.Lock()
_WEB_DIR = Path(__file__).parent / "web"


def engine() -> Engine:
    global _engine
    if _engine is None:
        # Double-checked locking: concurrent first requests must not construct two Engines that
        # both write the same index files.
        with _engine_lock:
            if _engine is None:
                _engine = Engine()
    return _engine


def _scope(namespace: str, agent_id: Optional[str], project_id: Optional[str]) -> Scope:
    return Scope(namespace=namespace or "default",
                 agent_id=agent_id or None, project_id=project_id or None)


# ---- request models -------------------------------------------------------
class ScopeIn(BaseModel):
    namespace: str = "default"
    agent_id: Optional[str] = None
    project_id: Optional[str] = None

    def to_scope(self) -> Scope:
        return Scope(namespace=self.namespace or "default",
                     agent_id=self.agent_id or None, project_id=self.project_id or None)


class TextMemoryIn(ScopeIn):
    text: str
    source: str = "user"
    valid_at: Optional[float] = None
    extract_graph: bool = True
    segment: bool = False
    consolidate_now: bool = True


class AskIn(ScopeIn):
    query: str
    verify: bool = True
    prove: bool = False
    as_of: Optional[float] = None


class StructuredRecallIn(ScopeIn):
    query: str
    verify: bool = True
    as_of: Optional[float] = None


class ReflexRecallIn(ScopeIn):
    query: str
    as_of: Optional[float] = None
    limit: int = 3
    member_limit: int = 6


# ---- helpers --------------------------------------------------------------
def _record_brief(rec) -> dict:
    return {
        "memory_id": rec.memory_id,
        "modality": rec.modality.value,
        "source": rec.source,
        "scope": rec.scope.model_dump(),
        "valid_at": rec.valid_at,
        "created_at": rec.created_at,
        "snippet": (rec.text or rec.summary or "")[:200],
        "salience": round(rec.salience, 3),
        "retrievability": round(rec.fsrs.priority(), 3),
        "entities": rec.entities,
        "invalid_at": rec.invalid_at,
        "expired_at": rec.expired_at,
        "content_hash": rec.content_hash,
    }


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except ModelCallError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---- routes ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    f = _WEB_DIR / "index.html"
    return HTMLResponse(f.read_text()) if f.exists() else HTMLResponse("<h1>Eidetic-Plus</h1>")


@app.get("/map", response_class=HTMLResponse)
async def memory_map():
    f = _WEB_DIR / "map.html"
    return HTMLResponse(f.read_text()) if f.exists() else HTMLResponse(
        "<h1>3D memory map not built yet</h1>")


@app.get("/api/stats")
async def stats(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None):
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(engine().stats, scope)


@app.get("/api/preflight")
async def preflight():
    """Preflight doctor: one real call per capability against the configured model IDs (embed /
    chat / rerank / multimodal / document), distinguishing a quota block from a dead key/model.
    Makes real calls when a key is set; reports skipped:no_key otherwise. Never a fabricated pass."""
    from .doctor import preflight as _preflight
    return await run_in_threadpool(_preflight, engine())


@app.post("/api/memories/text")
async def add_text(body: TextMemoryIn):
    eng = engine()
    rec = await run_in_threadpool(
        lambda: _guard(eng.ingest_text, body.text, source=body.source,
                       valid_at=body.valid_at, extract_graph=body.extract_graph,
                       scope=body.to_scope(), segment=body.segment,
                       consolidate_now=body.consolidate_now)
    )
    return {
        **_record_brief(rec),
        "pending_consolidation": bool(rec.metadata.get("pending_consolidation")),
        "auto_sleep": await run_in_threadpool(eng.auto_sleep_status, body.to_scope()),
    }


@app.post("/api/memories/file")
async def add_file(file: UploadFile = File(...), source: str = Form(""),
                   extract_graph: bool = Form(True), namespace: str = Form("default"),
                   agent_id: str = Form(""), project_id: str = Form(""),
                   consolidate_now: bool = Form(True)):
    data = await file.read()
    scope = _scope(namespace, agent_id, project_id)
    eng = engine()
    rec = await run_in_threadpool(
        lambda: _guard(eng.ingest_bytes, data, file.filename,
                       source=source or file.filename, extract_graph=extract_graph, scope=scope,
                       consolidate_now=consolidate_now)
    )
    return {
        **_record_brief(rec),
        "pending_consolidation": bool(rec.metadata.get("pending_consolidation")),
        "auto_sleep": await run_in_threadpool(eng.auto_sleep_status, scope),
    }


@app.post("/api/ask")
async def ask(body: AskIn):
    def _ask_and_prove():
        # ask + prove must run on the SAME threadpool thread: the retriever's last_trace is
        # thread-local (concurrency safety), so a second dispatch can land on another worker
        # and silently see a foreign/None trace, dropping the recall paths from the proof.
        # Same rule the truth_ledger route already follows.
        ans = _guard(engine().ask, body.query, verify=body.verify, as_of=body.as_of,
                     scope=body.to_scope())
        out = ans.model_dump()
        if body.prove:
            # API proof parity with MCP recall(prove=True): include recall paths when
            # RECALL_TRACE is on.
            out["proof"] = engine().prove(
                ans, with_paths=engine().settings.recall_trace_enabled)
        return out
    return await run_in_threadpool(_ask_and_prove)


@app.post("/api/reflex_recall")
async def reflex_recall(body: ReflexRecallIn):
    """LOCAL recall: the MemoryPacket of candidate memories for a query, built from the derived
    index + live graph/store with NO model call (no embed, no NLI, no reader). Works WITHOUT a key.
    Mirrors the MCP `reflex_recall` tool. This is recall (candidates + provenance), not a verified
    answer -- use /api/ask for the NLI-gated answer. Sub-second when REFLEX_RECALL=1 (the index is
    maintained incrementally); with the flag off the index is rebuilt from the store per call
    (O(records), correct but not sub-second on a large store)."""
    packet = await run_in_threadpool(
        lambda: engine().reflex_recall(body.query, scope=body.to_scope(), as_of=body.as_of))
    return packet.public_dict()


@app.post("/api/region_hints")
async def region_hints(body: ReflexRecallIn):
    """LOCAL memory-region/cocoon routing hints for a query, proof-linked to active raw members.
    No model call. Scope-filtered. Mirrors the MCP `region_hints` tool."""
    limit = max(0, min(20, int(body.limit)))
    member_limit = max(0, min(50, int(body.member_limit)))
    return await run_in_threadpool(
        lambda: engine().region_hints(
            body.query,
            scope=body.to_scope(),
            as_of=body.as_of,
            limit=limit,
            member_limit=member_limit,
        )
    )


@app.post("/api/structured_recall")
async def structured_recall(body: StructuredRecallIn):
    """Run the typed SMQE memory path directly: plan -> execute -> verify-or-abstain.
    Returns plan/backend/supports/citations with no generation step."""
    return await run_in_threadpool(
        lambda: engine().structured_recall(
            body.query,
            scope=body.to_scope(),
            as_of=body.as_of,
            verify=body.verify,
        )
    )


@app.get("/api/truth_ledger")
async def truth_ledger(query: str, namespace: str = "default", agent_id: Optional[str] = None,
                       project_id: Optional[str] = None, verify: bool = True,
                       as_of: Optional[float] = None):
    """Answer `query` and return its full TRUTH LEDGER: the proof tree enriched with each
    citation's validity window, current-ness, supersession chains, and the final claim_status
    (verified / contradicted / abstained / unverified). `as_of` answers as of a past unix time
    (is_current in the ledger stays relative to the present). Needs DASHSCOPE_API_KEY (it
    answers). Mirrors the MCP `truth_ledger` tool."""
    scope = _scope(namespace, agent_id, project_id)

    def _answer_and_prove():
        # ONE threadpool thread: truth_ledger(with_paths) reads the retriever's THREAD-LOCAL
        # last_trace, so ask + truth_ledger must run on the same thread or the recall-paths splice
        # silently sees a foreign/None trace and drops the paths.
        ans = _guard(engine().ask, query, verify=verify, as_of=as_of, scope=scope)
        return engine().truth_ledger(ans, scope=scope)

    return await run_in_threadpool(_answer_and_prove)


@app.get("/api/sync_health")
async def sync_health(namespace: str = "default", agent_id: Optional[str] = None,
                      project_id: Optional[str] = None):
    """Track 2 synchronization report: are the rebuildable surfaces (vector index, BM25) consistent
    with the source-of-truth store, plus the namespace memory version + reflex status. Read-only,
    no key. Mirrors the MCP `sync_health` tool."""
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(engine().sync_health, scope)


@app.get("/api/memories")
async def list_memories(namespace: str = "default", agent_id: Optional[str] = None,
                        project_id: Optional[str] = None,
                        limit: int = 100, offset: int = 0):
    """Paged listing (newest first), enveloped like the MCP list_memories tool: a 100k-record
    store must not serialize whole into one response, and `total` lets callers page without
    a second count call."""
    scope = _scope(namespace, agent_id, project_id)
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    recs = await run_in_threadpool(engine().list_memories, scope)
    return {
        "total": len(recs),
        "offset": offset,
        "limit": limit,
        "memories": [_record_brief(r) for r in recs[offset:offset + limit]],
    }


@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: str, namespace: str = "default",
                     agent_id: Optional[str] = None, project_id: Optional[str] = None):
    # Scope-safe single read: an id from another namespace is invisible (no cross-scope leak).
    scope = _scope(namespace, agent_id, project_id)
    rec = await run_in_threadpool(engine().get_record_in_scope, memory_id, scope)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such memory")
    return rec.model_dump()


@app.get("/api/raw/{content_hash}")
async def get_raw(content_hash: str, namespace: str = "default",
                  agent_id: Optional[str] = None, project_id: Optional[str] = None):
    """Return immutable bytes only when the hash is attached to a record visible in scope.

    The substrate is globally content-addressed internally, but the HTTP proof surface is scoped
    like memory reads: knowing a hash from namespace A must not open raw bytes from namespace B.
    """
    scope = _scope(namespace, agent_id, project_id)
    rec = await run_in_threadpool(lambda: engine().store.get_by_hash(content_hash, scope))
    if rec is None:
        raise HTTPException(status_code=404, detail="No such immutable object in scope")
    try:
        data = await run_in_threadpool(engine().get_raw, content_hash)
    except KeyError:
        raise HTTPException(status_code=404, detail="No such immutable object")
    ctype = mimetypes.guess_type(content_hash)[0] or "application/octet-stream"
    return Response(content=data, media_type=ctype)


@app.post("/api/consolidate")
async def consolidate(namespace: str = "default", agent_id: Optional[str] = None,
                      project_id: Optional[str] = None):
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(lambda: _guard(engine().consolidate, scope=scope))


@app.post("/api/reawaken/{memory_id}")
async def reawaken(memory_id: str, namespace: str = "default",
                   agent_id: Optional[str] = None, project_id: Optional[str] = None):
    scope = _scope(namespace, agent_id, project_id)
    if await run_in_threadpool(engine().get_record_in_scope, memory_id, scope) is None:
        raise HTTPException(status_code=404, detail="No such memory")
    rec = await run_in_threadpool(engine().reawaken, memory_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such memory")
    return _record_brief(rec)


@app.post("/api/forget/{memory_id}")
async def forget(memory_id: str, namespace: str = "default",
                 agent_id: Optional[str] = None, project_id: Optional[str] = None):
    """Decay retrieval priority only; immutable raw bytes remain available in scope."""
    scope = _scope(namespace, agent_id, project_id)
    if await run_in_threadpool(engine().get_record_in_scope, memory_id, scope) is None:
        raise HTTPException(status_code=404, detail="No such memory")
    rec = await run_in_threadpool(engine().forget, memory_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="No such memory")
    return {"ok": True, "note": "priority decayed; raw record NOT deleted", **_record_brief(rec)}


@app.post("/api/sleep")
async def sleep(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None, llm_summaries: bool = Query(False)):
    """The unified sleep cycle (consolidate_pending -> dream -> optional LLM summaries), identical
    to the MCP `sleep` tool. Token-free + key-free when nothing is pending and llm_summaries off."""
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(
        lambda: _guard(engine().sleep, scope=scope, llm_summaries=llm_summaries))


@app.post("/api/dream")
async def dream(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None):
    """Token-free dreaming pass (replay / inference / multi-resolution gists) for a scope. Mirrors
    the MCP `consolidate` tool; never deletes a raw record. Works without a key."""
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(engine().dream, scope=scope)


@app.get("/api/health_report")
async def health_report(namespace: str = "default", agent_id: Optional[str] = None,
                        project_id: Optional[str] = None):
    """Read-only scope self-diagnosis (coverage, contradiction load, debt, orphans). Mirrors the
    MCP `health_report` tool. Every figure is counted from the store; works without a key."""
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(engine().memory_health_report, scope)


@app.get("/api/value_as_of")
async def value_as_of(entity: str, relation: str, as_of: Optional[float] = None,
                      namespace: str = "default", agent_id: Optional[str] = None,
                      project_id: Optional[str] = None):
    """C2 time-travel: deterministic value of (entity, relation) valid at `as_of` (mirrors MCP)."""
    scope = _scope(namespace, agent_id, project_id)
    out = await run_in_threadpool(lambda: engine().value_as_of(entity, relation, as_of=as_of, scope=scope))
    return out if out is not None else {"value": None, "note": "no fact valid as of that time"}


@app.get("/api/fact_history")
async def fact_history(entity: str, relation: str, namespace: str = "default",
                       agent_id: Optional[str] = None, project_id: Optional[str] = None):
    """C2 current-vs-historical supersession chain for (entity, relation) (mirrors MCP)."""
    scope = _scope(namespace, agent_id, project_id)
    hist = await run_in_threadpool(lambda: engine().fact_history(entity, relation, scope=scope))
    return {"history": hist}


@app.get("/api/integrity_report")
async def integrity_report(namespace: str = "default", agent_id: Optional[str] = None,
                           project_id: Optional[str] = None):
    """C1 operation-level integrity report (mirrors MCP). Read-only, no key."""
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(engine().integrity_report, scope)


@app.get("/api/scratchpad")
async def scratchpad(namespace: str = "default", agent_id: Optional[str] = None,
                     project_id: Optional[str] = None):
    """Working scratchpad of high-salience verified ACTIVE facts for a scope (each linked to its
    raw source hash). Mirrors the MCP `scratchpad` tool. Read-only, no key."""
    scope = _scope(namespace, agent_id, project_id)
    entries = await run_in_threadpool(engine().build_scratchpad, scope)
    return {"scratchpad": entries}


@app.get("/api/preference_profile")
async def preference_profile(namespace: str = "default", agent_id: Optional[str] = None,
                             project_id: Optional[str] = None,
                             include_inactive: bool = Query(False),
                             limit: int = Query(50)):
    """Current preference profile with source memory provenance. Mirrors MCP
    `preference_profile`. Read-only, no key."""
    scope = _scope(namespace, agent_id, project_id)
    limit = max(0, min(500, int(limit)))
    return await run_in_threadpool(
        lambda: engine().preference_profile(
            scope=scope,
            include_inactive=include_inactive,
            limit=limit,
        )
    )


@app.get("/api/why_remembered/{memory_id}")
async def why_remembered(memory_id: str, namespace: str = "default",
                         agent_id: Optional[str] = None, project_id: Optional[str] = None):
    """'Why I remember this strongly': proof-linked salience signals, hedged and non-clinical."""
    scope = _scope(namespace, agent_id, project_id)
    out = await run_in_threadpool(engine().salience_explanation, memory_id, scope)
    if out is None:
        raise HTTPException(status_code=404, detail="No such memory in scope")
    return out


@app.get("/api/brain_health_score")
async def brain_health_score(namespace: str = "default", agent_id: Optional[str] = None,
                             project_id: Optional[str] = None):
    """Local BrainHealthScore + components for a scope (diagnostic rollup, not a benchmark).
    Mirrors the MCP `brain_health_score` tool. Read-only, no key."""
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(engine().brain_health_score, scope)


@app.get("/api/memory_autopsy")
async def memory_autopsy(question: str, namespace: str = "default",
                         agent_id: Optional[str] = None, project_id: Optional[str] = None):
    """Read-only failure diagnosis for a would-be-missed question (mirrors the MCP tool). No key."""
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(engine().memory_autopsy, question, scope=scope)


@app.get("/api/recall_trace")
async def recall_trace(namespace: str = "default", agent_id: Optional[str] = None,
                       project_id: Optional[str] = None):
    """The most recent RecallTrace IN THIS SCOPE (why the last recall found/missed what it
    did). {} unless RECALL_TRACE is enabled. Scope-filtered - another namespace's trace is
    never visible. Mirrors the MCP `recall_trace` tool. Read-only, no key."""
    scope = _scope(namespace, agent_id, project_id)
    t = await run_in_threadpool(lambda: engine().recall_trace(scope=scope))
    return t.model_dump() if t is not None else {}


@app.get("/api/prove_age_independence")
async def prove_age_independence(namespace: str = "default", agent_id: Optional[str] = None,
                                 project_id: Optional[str] = None, k: int = Query(5)):
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(
        lambda: _guard(engine().prove_age_independence, scope=scope, k=k))


# ---- 3D memory map data (a PROJECTION of high-dim embeddings, not the storage) ----
def _project_3d(vectors: dict) -> dict:
    """PCA(1024-2048D -> 3D). The map is a human-navigation projection; memory is stored
    in high dimensions. Storing in 3D would collapse the separating structure."""
    ids = list(vectors)
    if len(ids) < 3:
        return {mid: [float(i) * 30.0, 0.0, 0.0] for i, mid in enumerate(ids)}
    M = np.array([vectors[i] for i in ids], dtype=np.float64)
    Mc = M - M.mean(axis=0)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    coords = Mc @ Vt[: min(3, Vt.shape[0])].T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    coords = coords / (np.abs(coords).max() + 1e-9) * 100.0
    return {ids[i]: coords[i].tolist() for i in range(len(ids))}


def _graph_data(eng: Engine, scope: Scope, max_edges: int = 2500) -> dict:
    recs = eng.list_memories(scope)
    ids = [r.memory_id for r in recs]
    vecs = eng.index.get_vectors(ids)
    pos = _project_3d(vecs) if vecs else {mid: [0.0, 0.0, 0.0] for mid in ids}
    nodes = []
    for r in recs:
        x, y, z = pos.get(r.memory_id, [0.0, 0.0, 0.0])
        nodes.append({
            "id": r.memory_id, "label": (r.text or r.summary or "")[:60],
            "salience": round(r.salience, 3), "retrievability": round(r.fsrs.priority(), 3),
            "modality": r.modality.value, "valid_at": r.valid_at,
            "content_hash": r.content_hash, "source": r.source,
            "invalidated": r.invalid_at is not None or r.expired_at is not None,
            "x": x, "y": y, "z": z,
        })
    edges = []
    # Association edges: memories sharing an entity.
    ent_map: dict[str, list[str]] = {}
    for r in recs:
        for e in r.entities:
            ent_map.setdefault(e.lower(), []).append(r.memory_id)
    seen = set()
    for e, mids in ent_map.items():
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                key = tuple(sorted((mids[i], mids[j])))
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"source": key[0], "target": key[1], "kind": "entity",
                              "relation": e, "active": True})
                if len(edges) >= max_edges:
                    break
    # Co-activated memory links (bi-temporal: active vs invalidated drawn differently).
    for edge in eng.store.all_edges(scope):
        if edge.relation == CO_ACTIVATED and len(edges) < max_edges:
            edges.append({"source": edge.src, "target": edge.dst, "kind": "co_activated",
                          "active": edge.is_active_at()})
    return {
        "nodes": nodes, "edges": edges,
        "projection": "PCA(high-D embeddings -> 3D)",
        "note": "3D is a PROJECTION for navigation. Memory is stored in "
                f"{eng.settings.embed_dim}-D; the engine never stores in 3D.",
    }


@app.get("/api/graph")
async def graph(namespace: str = "default", agent_id: Optional[str] = None,
                project_id: Optional[str] = None):
    scope = _scope(namespace, agent_id, project_id)
    return await run_in_threadpool(_graph_data, engine(), scope)


# Static assets (vendored 3D lib, etc.).
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")
