"""Offline tests for the S4 persistent embedding cache."""
from __future__ import annotations

import numpy as np

from eidetic.embed_cache import PersistentEmbedCache


def test_cache_keys_on_model_and_dim(tmp_path):
    c = PersistentEmbedCache(tmp_path / "e.sqlite")
    v = np.arange(4, dtype=np.float32)
    c.put("m1", 4, "hello", v)
    assert np.array_equal(c.get("m1", 4, "hello"), v)
    assert c.get("m2", 4, "hello") is None         # model rename -> miss (never a stale vector)
    assert c.get("m1", 8, "hello") is None         # dim change -> miss
    assert c.get("m1", 4, "other") is None         # different text -> miss
    assert c.count() == 1


def test_client_embed_texts_serves_from_cache_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    from eidetic.config import get_settings
    from eidetic.dashscope_client import DashScopeClient

    get_settings.cache_clear()
    try:
        c = DashScopeClient(get_settings())
        dim = c.settings.embed_dim
        calls = {"n": 0}

        def fake_raw(texts):
            calls["n"] += 1
            return np.stack([np.full(dim, float(len(t)), np.float32) for t in texts])

        monkeypatch.setattr(c, "_embed_raw", fake_raw)
        monkeypatch.setattr(c, "_require_key", lambda: None)
        a = c.embed_texts(["alpha", "beta"])
        assert calls["n"] == 1 and a.shape == (2, dim)

        # second call: both texts cached -> no _embed_raw call, and NO key required (warm path).
        def boom():
            raise AssertionError("a full cache hit must not require a key")

        monkeypatch.setattr(c, "_require_key", boom)
        b = c.embed_texts(["alpha", "beta"])
        assert calls["n"] == 1 and np.array_equal(a, b)

        # a partial miss embeds ONLY the new text.
        monkeypatch.setattr(c, "_require_key", lambda: None)
        c.embed_texts(["alpha", "gamma"])
        assert calls["n"] == 2
    finally:
        get_settings.cache_clear()
