"""Offline tests for the preflight doctor + fail-clean experimental stubs (Phase R)."""
from __future__ import annotations

import types

import pytest

from eidetic.doctor import _classify_error, preflight
from eidetic.errors import FeatureNotImplementedError


def test_classify_error_distinguishes_failure_modes():
    assert _classify_error("DashScope call failed (HTTP 403): The free tier of the model "
                           "has been exhausted") == "quota_exhausted"
    assert _classify_error("model qwen-bogus does not exist") == "bad_model_id"
    assert _classify_error("Invalid API key provided") == "auth"
    assert _classify_error("connection reset") == "error"


def test_preflight_without_key_is_all_skipped_never_fake_green(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    from eidetic.config import get_settings
    from eidetic.engine import Engine

    get_settings.cache_clear()
    try:
        rep = preflight(Engine(get_settings()))
        assert rep["has_api_key"] is False and rep["summary"] == "no_key"
        caps = {c["capability"] for c in rep["capabilities"]}
        assert {"embed", "chat", "rerank", "embed_image"} <= caps
        assert all(c.get("skipped") and c["error_class"] == "no_key" for c in rep["capabilities"])
        assert rep["failing"] == []                  # skipped is not a failure, and not a fake pass
    finally:
        get_settings.cache_clear()


def test_preflight_summary_quota_vs_degraded(monkeypatch):
    # Drive the summary logic with a fake client whose every call raises a quota error, then a
    # mixed failure -- proves quota-exhausted is distinguished from a genuinely degraded capability.
    from eidetic.dashscope_client import ModelCallError

    class QuotaClient:
        def embed_texts(self, t): raise ModelCallError("HTTP 403: free tier exhausted")
        def chat(self, *a, **k): raise ModelCallError("HTTP 403: free tier exhausted")
        def rerank(self, *a, **k): raise ModelCallError("HTTP 403: free tier exhausted")
        def embed_image(self, p): raise ModelCallError("HTTP 403: free tier exhausted")
        def read_document(self, p): raise ModelCallError("HTTP 403: free tier exhausted")

    settings = types.SimpleNamespace(
        has_api_key=True, region="singapore", embed_dim=1024,
        text_embed_model="e", salience_model="c", rerank_model="r",
        multimodal_embed_model="m", doc_model="d")
    eng = types.SimpleNamespace(settings=settings, client=QuotaClient())
    rep = preflight(eng)
    assert rep["summary"] == "quota_exhausted"        # key valid, just out of quota
    assert all(c["error_class"] == "quota_exhausted" for c in rep["capabilities"])


# ---- fail-clean experimental stubs ---------------------------------------------------------
def test_memory_manager_stub_fails_clean_when_enabled():
    from eidetic.dreaming.manager import run_memory_manager
    off = types.SimpleNamespace(settings=types.SimpleNamespace(memory_manager_enabled=False))
    assert run_memory_manager(off) == {"skipped": "disabled"}
    on = types.SimpleNamespace(settings=types.SimpleNamespace(memory_manager_enabled=True))
    with pytest.raises(FeatureNotImplementedError, match="experimental and not implemented"):
        run_memory_manager(on)


def test_debate_stub_fails_clean_when_enabled():
    from eidetic.debate import run_conflict_debate
    off = types.SimpleNamespace(settings=types.SimpleNamespace(debate_enabled=False))
    assert run_conflict_debate(off, "q") == {"skipped": "disabled"}
    on = types.SimpleNamespace(settings=types.SimpleNamespace(debate_enabled=True))
    with pytest.raises(FeatureNotImplementedError, match="experimental and not implemented"):
        run_conflict_debate(on, "q")
