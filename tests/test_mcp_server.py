"""Offline tests for the MCP server (no key). All run through the REAL engine via an injected
deterministic fake client, consistent with how the rest of the offline suite stubs model calls."""
from __future__ import annotations

import asyncio
import hashlib
import re

import numpy as np
import pytest

import eidetic.mcp_server as mcp_server


# ---- a deterministic, no-network client (test fake, not production fabrication) -------------
class FakeClient:
    def __init__(self, dim: int):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            b = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            v[b] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, text):
        return self._embed(text)

    def embed_texts(self, texts):
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def score_importance(self, text):
        return 0.5

    def extract_edges(self, text):
        return []

    def find_contradictions(self, new_fact, candidates):
        return []

    def extract_current_value_matches(self, query, candidates):
        return []

    def generate_probes(self, memory_text, n=3):
        return []

    def rerank(self, query, documents, top_n):
        return [(i, 1.0 - 0.001 * i) for i in range(min(top_n, len(documents)))]

    def generate_answer(self, question, context_blocks, model=None):
        return context_blocks[0][:300] if context_blocks else "I do not have that in memory."

    def nli(self, premise, hypothesis):
        pt = set(re.findall(r"[a-z0-9]+", premise.lower()))
        ht = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))
        overlap = len(pt & ht) / (len(ht) or 1)
        return ("entailment", 0.9) if overlap >= 0.5 else ("neutral", 0.4)


@pytest.fixture()
def mcp_engine(tmp_path, monkeypatch):
    """A real Engine on the numpy backend with the fake client, injected into the MCP server."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    from eidetic.config import get_settings
    from eidetic.engine import Engine

    get_settings.cache_clear()
    settings = get_settings()
    eng = Engine(settings, client=FakeClient(settings.embed_dim))
    monkeypatch.setattr(mcp_server, "_engine", eng)
    yield eng
    monkeypatch.setattr(mcp_server, "_engine", None)
    get_settings.cache_clear()


# ---- server + schema (no key needed) -------------------------------------------------------
def test_server_lists_all_expected_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"remember", "recall", "consolidate", "list_memories", "get_raw",
            "forget", "reawaken", "stats"} <= names


def test_tool_schemas_mark_required_fields():
    tools = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert "content" in tools["remember"].inputSchema.get("required", [])
    assert "query" in tools["recall"].inputSchema.get("required", [])
    assert "memory_id" in tools["get_raw"].inputSchema.get("required", [])


# ---- round-trip through the real engine ----------------------------------------------------
def test_remember_then_recall_round_trip(mcp_engine):
    out = mcp_server.remember("Alice works at Acme Corporation", namespace="proj")
    assert out["ok"] and out["memory_id"]
    ans = mcp_server.recall("where does Alice work", namespace="proj")
    assert "acme" in ans["answer"].lower()
    assert ans["citations"], "recall must return cited sources"


# ---- scope isolation (the no-leak guarantee) -----------------------------------------------
def test_scope_isolation_no_cross_namespace_leak(mcp_engine):
    mcp_server.remember("The launch code is alpha-seven", namespace="A")
    # a recall in a different namespace must not surface namespace A's memory
    ans_b = mcp_server.recall("what is the launch code", namespace="B")
    assert "alpha-seven" not in ans_b["answer"].lower()
    assert mcp_server.list_memories(namespace="B")["total"] == 0
    assert mcp_server.list_memories(namespace="A")["total"] == 1


# ---- get_raw byte-identical + scope-filtered -----------------------------------------------
def test_get_raw_is_byte_identical_and_scope_filtered(mcp_engine):
    content = "Carol lives in Paris since 2019"
    out = mcp_server.remember(content, namespace="A")
    mid = out["memory_id"]
    raw = mcp_server.get_raw(mid, namespace="A")
    assert raw["raw_encoding"] == "utf-8" and raw["raw"] == content   # verbatim, not paraphrased
    assert raw["verified_against_self"] is True
    # the same id is invisible from another namespace (no cross-scope read)
    with pytest.raises(RuntimeError, match="No such memory in scope"):
        mcp_server.get_raw(mid, namespace="B")


# ---- forget lowers priority without deleting the raw record --------------------------------
def test_forget_decays_priority_without_deleting_raw(mcp_engine):
    out = mcp_server.remember("Dave enjoys hiking on weekends", namespace="A")
    mid = out["memory_id"]
    before = mcp_engine.get_record(mid).fsrs
    lapses0, diff0 = before.lapses, before.difficulty
    res = mcp_server.forget(mid, namespace="A")
    assert res["ok"]
    after = mcp_engine.get_record(mid).fsrs
    # forget = an FSRS lapse: it records a lapse and raises difficulty (faster future decay),
    # which lowers index priority over time. It never deletes the raw record.
    assert after.lapses == lapses0 + 1 and after.difficulty > diff0
    assert mcp_server.get_raw(mid, namespace="A")["raw"] == "Dave enjoys hiking on weekends"


# ---- read-only tools + consolidate work without a key --------------------------------------
def test_readonly_and_consolidate_need_no_key(mcp_engine):
    assert mcp_server.stats(namespace="empty")["memories"] == 0
    assert mcp_server.list_memories(namespace="empty")["memories"] == []
    assert isinstance(mcp_server.consolidate(namespace="empty"), dict)   # token-free dream pass


# ---- entry point ---------------------------------------------------------------------------
def test_main_entry_point_runs_stdio(monkeypatch):
    called = {}
    monkeypatch.setattr(mcp_server.mcp, "run", lambda transport: called.setdefault("t", transport))
    monkeypatch.setattr("sys.argv", ["eidetic-plus"])
    mcp_server.main()
    assert called["t"] == "stdio"
