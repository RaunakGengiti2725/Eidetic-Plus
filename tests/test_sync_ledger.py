"""Track 2: Engine.sync_health(scope) -- a derived synchronization report. Are the rebuildable
surfaces (vector index, BM25) consistent with the source-of-truth store? Counts compared are
like-for-like GLOBAL (the index/BM25 are not per-scope). Emits SYNC_DEBT_DETECTED when behind."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import BrainEventType, MemoryRecord, Scope, now


class _FakeEmbed:
    def __init__(self, dim):
        self.dim = dim

    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._e(t)

    def embed_texts(self, ts):
        return np.stack([self._e(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)


def _engine(fresh_settings, **kw):
    s = replace(fresh_settings, **kw)
    return Engine(s, client=_FakeEmbed(s.embed_dim))


def test_sync_health_clean_after_ingest(fresh_settings):
    e = _engine(fresh_settings)
    e.ingest_text("alpha fact one", consolidate_now=False)
    e.ingest_text("beta fact two", consolidate_now=False)
    h = e.sync_health()
    assert h["in_sync"] is True
    assert h["debt"] == []
    assert h["surfaces"]["store_records_global"] == 2
    assert h["surfaces"]["vector_index_global"] == 2


def test_sync_health_detects_index_debt_and_emits_event(fresh_settings):
    e = _engine(fresh_settings, brain_events_enabled=True)
    e.ingest_text("alpha fact one", consolidate_now=False)
    # induce debt: a record present in the source-of-truth store but absent from the vector index.
    e.store.upsert_record(MemoryRecord(memory_id="ghost", content_hash="hg", text="ghost record",
                                       scope=Scope(), valid_at=now()))
    h = e.sync_health()
    assert h["in_sync"] is False
    assert any(d["surface"] == "vector_index" for d in h["debt"])
    assert h["repair"] == "rebuild_index_from_store"
    assert e.brain_log.by_type(BrainEventType.SYNC_DEBT_DETECTED)


def test_sync_health_reports_namespace_memory_version(fresh_settings):
    e = _engine(fresh_settings)
    e.ingest_text("alpha fact one", consolidate_now=False)
    assert e.sync_health()["surfaces"]["memory_version"] == 1


def test_sync_debt_event_type_exists():
    assert hasattr(BrainEventType, "SYNC_DEBT_DETECTED")


def test_sync_health_http_route(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("starlette")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    from starlette.testclient import TestClient

    import eidetic.api as api_mod
    from eidetic.config import get_settings

    get_settings.cache_clear()
    try:
        eng = Engine(get_settings(), client=_FakeEmbed(get_settings().embed_dim))
        eng.ingest_text("alpha fact one", consolidate_now=False)
        monkeypatch.setattr(api_mod, "_engine", eng)
        r = TestClient(api_mod.app).get("/api/sync_health?namespace=default")
        assert r.status_code == 200
        assert r.json()["in_sync"] is True
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()


def test_sync_health_mcp_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    import eidetic.mcp_server as mcp_server
    from eidetic.config import get_settings

    get_settings.cache_clear()
    try:
        eng = Engine(get_settings(), client=_FakeEmbed(get_settings().embed_dim))
        eng.ingest_text("alpha fact one", consolidate_now=False)
        monkeypatch.setattr(mcp_server, "_engine", eng)
        out = mcp_server.sync_health(namespace="default")
        assert out["in_sync"] is True
        assert out["surfaces"]["store_records_global"] == 1
    finally:
        monkeypatch.setattr(mcp_server, "_engine", None)
        get_settings.cache_clear()
