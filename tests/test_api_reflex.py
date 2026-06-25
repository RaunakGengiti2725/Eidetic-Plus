"""Offline test for the HTTP reflex_recall route: a key-free, model-free sub-second recall surface
that mirrors the MCP tool. Scope-safe; reflects the current store."""
from __future__ import annotations

import pytest


def _client(tmp_path, monkeypatch):
    pytest.importorskip("starlette")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    from starlette.testclient import TestClient

    import eidetic.api as api_mod
    from eidetic.config import get_settings
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope, now

    get_settings.cache_clear()
    eng = Engine(get_settings())
    eng.store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="h1", raw_uri="cas://h1",
        text="The Helios project quarterly revenue was 4.2 million dollars",
        scope=Scope(namespace="proj"), valid_at=now()))
    monkeypatch.setattr(api_mod, "_engine", eng)
    return api_mod, TestClient(api_mod.app)


def test_reflex_recall_route_returns_packet(tmp_path, monkeypatch):
    from eidetic.config import get_settings
    api_mod, client = _client(tmp_path, monkeypatch)
    try:
        r = client.post("/api/reflex_recall",
                        json={"query": "Helios project revenue", "namespace": "proj"})
        assert r.status_code == 200
        body = r.json()
        assert "m1" in body["candidate_ids"]
        assert body["coverage"] > 0.0
        assert body["snippets"]["m1"]
        assert body["content_hashes"]["m1"] == "h1"
        assert "total" in body["latency_ms"]
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()


def test_reflex_recall_route_is_scope_isolated(tmp_path, monkeypatch):
    from eidetic.config import get_settings
    api_mod, client = _client(tmp_path, monkeypatch)
    try:
        r = client.post("/api/reflex_recall",
                        json={"query": "Helios project revenue", "namespace": "other"})
        assert r.status_code == 200
        assert r.json()["candidate_ids"] == []
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()
