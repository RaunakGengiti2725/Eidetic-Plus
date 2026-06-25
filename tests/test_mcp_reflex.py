"""Offline tests for the MCP reflex_recall tool: the sub-second local recall surface exposed to
agents. No key, no model call. Scope-safe, and correct against the current store even with the
ask()-path flag off (it rebuilds the derived index on demand when not maintained)."""
from __future__ import annotations

import asyncio

import pytest

import eidetic.mcp_server as mcp_server


@pytest.fixture()
def mcp_eng(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    from eidetic.config import get_settings
    from eidetic.engine import Engine

    get_settings.cache_clear()
    eng = Engine(get_settings())
    monkeypatch.setattr(mcp_server, "_engine", eng)
    yield eng
    monkeypatch.setattr(mcp_server, "_engine", None)
    get_settings.cache_clear()


def _rec(mid, text, ns):
    from eidetic.models import MemoryRecord, Scope, now
    return MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}",
                        text=text, scope=Scope(namespace=ns), valid_at=now())


def test_reflex_recall_tool_returns_packet_without_a_key(mcp_eng):
    mcp_eng.store.upsert_record(_rec("m1", "The Helios project quarterly revenue was 4.2 million", "proj"))
    out = mcp_server.reflex_recall("Helios project revenue", namespace="proj")
    assert "m1" in out["candidate_ids"]
    assert out["coverage"] > 0.0


def test_reflex_recall_tool_is_scope_isolated(mcp_eng):
    mcp_eng.store.upsert_record(_rec("a1", "secret revenue figure", "alpha"))
    mcp_eng.store.upsert_record(_rec("b1", "secret revenue figure", "beta"))
    assert mcp_server.reflex_recall("secret revenue", namespace="alpha")["candidate_ids"] == ["a1"]
    assert mcp_server.reflex_recall("secret revenue", namespace="beta")["candidate_ids"] == ["b1"]


def test_reflex_recall_tool_reflects_new_writes_when_flag_off(mcp_eng):
    # REFLEX_RECALL is off by default -> the index is not incrementally maintained. The on-demand
    # tool must still reflect the CURRENT store on every call (rebuild when not maintained).
    mcp_eng.store.upsert_record(_rec("m1", "alpha keyword one", "proj"))
    assert "m1" in mcp_server.reflex_recall("alpha keyword one", namespace="proj")["candidate_ids"]
    mcp_eng.store.upsert_record(_rec("m2", "beta keyword two", "proj"))
    assert "m2" in mcp_server.reflex_recall("beta keyword two", namespace="proj")["candidate_ids"]


def test_reflex_recall_tool_is_listed():
    tools = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert "reflex_recall" in tools
