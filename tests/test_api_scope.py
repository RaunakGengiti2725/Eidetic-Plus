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
        assert client.get("/api/memories/m1").status_code == 404               # default != wildcard
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()


def test_http_raw_read_is_scope_safe(tmp_path, monkeypatch):
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
        raw = b"namespace A immutable source bytes"
        h, uri = eng.substrate.put(raw)
        eng.store.upsert_record(MemoryRecord(memory_id="m1", content_hash=h, raw_uri=uri,
                                             raw_bytes_len=len(raw), text=raw.decode("utf-8"),
                                             scope=Scope(namespace="A"), valid_at=now()))
        monkeypatch.setattr(api_mod, "_engine", eng)
        client = TestClient(api_mod.app)

        ok = client.get(f"/api/raw/{h}?namespace=A")
        assert ok.status_code == 200 and ok.content == raw
        assert client.get(f"/api/raw/{h}?namespace=B").status_code == 404
        assert client.get(f"/api/raw/{h}").status_code == 404
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()


def test_http_collection_reads_default_scope_not_all_scopes(tmp_path, monkeypatch):
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
        eng.store.upsert_record(MemoryRecord(memory_id="default_mem", content_hash="hd",
                                             text="default visible", scope=Scope(),
                                             valid_at=now()))
        eng.store.upsert_record(MemoryRecord(memory_id="tenant_mem", content_hash="ht",
                                             text="tenant secret", scope=Scope(namespace="tenant"),
                                             valid_at=now()))
        monkeypatch.setattr(api_mod, "_engine", eng)
        client = TestClient(api_mod.app)

        default_rows = client.get("/api/memories").json()
        tenant_rows = client.get("/api/memories?namespace=tenant").json()
        assert [r["memory_id"] for r in default_rows] == ["default_mem"]
        assert [r["memory_id"] for r in tenant_rows] == ["tenant_mem"]
        assert client.get("/api/stats").json()["memories"] == 1
        assert client.get("/api/stats?namespace=tenant").json()["memories"] == 1
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()


def test_http_reawaken_is_scope_safe(tmp_path, monkeypatch):
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

        assert client.post("/api/reawaken/m1?namespace=B").status_code == 404
        assert client.post("/api/reawaken/m1?namespace=A").status_code == 200
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()


def test_http_forget_is_scope_safe_and_preserves_raw(tmp_path, monkeypatch):
    pytest.importorskip("starlette")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    from starlette.testclient import TestClient

    import eidetic.api as api_mod
    from eidetic import fsrs
    from eidetic.config import get_settings
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope, now

    get_settings.cache_clear()
    try:
        eng = Engine(get_settings())
        raw = b"namespace A source bytes that must survive HTTP forget"
        h, uri = eng.substrate.put(raw)
        rec = MemoryRecord(memory_id="m1", content_hash=h, raw_uri=uri,
                           raw_bytes_len=len(raw), text=raw.decode("utf-8"),
                           fsrs=fsrs.init_state(importance=0.8, surprise=0.4),
                           scope=Scope(namespace="A"), valid_at=now())
        eng.store.upsert_record(rec)
        monkeypatch.setattr(api_mod, "_engine", eng)
        client = TestClient(api_mod.app)

        before = eng.store.get_record("m1")
        assert before is not None
        before_lapses = before.fsrs.lapses
        before_difficulty = before.fsrs.difficulty

        assert client.post("/api/forget/m1?namespace=B").status_code == 404
        unchanged = eng.store.get_record("m1")
        assert unchanged is not None
        assert unchanged.fsrs.lapses == before_lapses

        res = client.post("/api/forget/m1?namespace=A")
        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        assert body["note"] == "priority decayed; raw record NOT deleted"
        assert body["content_hash"] == h

        forgotten = eng.store.get_record("m1")
        assert forgotten is not None
        assert forgotten.fsrs.lapses == before_lapses + 1
        assert forgotten.fsrs.difficulty > before_difficulty

        ok = client.get(f"/api/raw/{h}?namespace=A")
        assert ok.status_code == 200
        assert ok.content == raw
        assert client.get(f"/api/raw/{h}?namespace=B").status_code == 404
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()
