"""Regression: the visual model methods must FAIL LOUD on an unparseable response, never fabricate
a verdict/extraction (Phase R no-fakes; closes a confirmed adversarial-review finding)."""
from __future__ import annotations

import pytest

from eidetic.config import get_settings
from eidetic.dashscope_client import DashScopeClient, ModelCallError


def _client(monkeypatch):
    c = DashScopeClient(get_settings())
    monkeypatch.setattr(c, "_require_key", lambda: None)             # bypass the key gate
    monkeypatch.setattr(c, "_file_uri", lambda p: "uri://x")        # no file IO
    monkeypatch.setattr(c, "_mm_text", lambda resp: "this is not json at all")

    class _Stub:
        class MultiModalConversation:
            @staticmethod
            def call(**kw):
                return {}

    monkeypatch.setattr(c, "_ds", _Stub)
    return c


def test_verify_visual_fails_loud_not_fabricated(monkeypatch):
    c = _client(monkeypatch)
    with pytest.raises(ModelCallError, match="not fabricated"):
        c.verify_visual("/x.png", "the chart shows growth")


def test_extract_visual_graph_fails_loud_not_fabricated(monkeypatch):
    c = _client(monkeypatch)
    with pytest.raises(ModelCallError, match="not fabricated"):
        c.extract_visual_graph("/x.png")
