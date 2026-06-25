"""Offline test for the Phase 0 scope-safe HTTP single-memory read."""
from __future__ import annotations

import pytest


def test_http_single_read_is_scope_safe(tmp_path, monkeypatch):
    pytest.importorskip("starlette")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    from starlette.testclient import TestClient

    import eidetic.api as api_mod
    from eidetic.config import get_settings
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope, now

    get_settings.cache_clear()
    try:
        eng = Engine(get_settings())
        eng.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1", text="secret",
                                             scope=Scope(namespace="A"), valid_at=now()))
        monkeypatch.setattr(api_mod, "_engine", eng)
        client = TestClient(api_mod.app)

        assert client.get("/api/memories/m1?namespace=A").status_code == 200   # in-scope read OK
        assert client.get("/api/memories/m1?namespace=B").status_code == 404   # cross-scope hidden
        assert client.get("/api/memories/m1").status_code == 200               # legacy unscoped OK
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()
