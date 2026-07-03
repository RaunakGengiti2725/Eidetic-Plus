"""EXTRACT_RESULT_CACHE: re-ingesting identical content stops re-paying temp-0 extraction."""
from __future__ import annotations

from dataclasses import replace

import pytest

from eidetic.dashscope_client import DashScopeClient, ModelCallError


def _client(fresh_settings, tmp_path, monkeypatch, *, enabled=True, responses=None):
    s = replace(fresh_settings, data_dir=tmp_path / "data",
                extract_result_cache_enabled=enabled)
    c = DashScopeClient(s)
    calls = {"n": 0}

    def fake_chat(model, system, user, **kw):
        calls["n"] += 1
        if isinstance(responses, Exception):
            raise responses
        return responses or '{"triples": [{"src":"Ari","relation":"waters","dst":"the fern","fact":"Ari waters the fern"}]}'

    monkeypatch.setattr(c, "chat", fake_chat)
    return c, calls


def test_repeat_window_costs_one_call_and_parses_identically(fresh_settings, tmp_path, monkeypatch):
    c, calls = _client(fresh_settings, tmp_path, monkeypatch)
    first = c._extract_edges_window("Ari waters the fern every Sunday.")
    second = c._extract_edges_window("Ari waters the fern every Sunday.")
    assert calls["n"] == 1                       # second window served from the cache
    assert first == second and first[0]["src"] == "Ari"
    # a different window is a miss
    c._extract_edges_window("Ari repotted the fern in spring.")
    assert calls["n"] == 2


def test_prompt_identity_is_part_of_the_key(fresh_settings, tmp_path, monkeypatch):
    """Edges and claims share window text but not prompts: one must never serve the other."""
    c, calls = _client(
        fresh_settings, tmp_path, monkeypatch,
        responses='{"triples": [], "claims": []}')
    c._extract_edges_window("Ari waters the fern every Sunday.")
    c._extract_claims_window("Ari waters the fern every Sunday.")
    assert calls["n"] == 2                       # different prompt -> different key


def test_errors_are_never_cached(fresh_settings, tmp_path, monkeypatch):
    c, calls = _client(fresh_settings, tmp_path, monkeypatch,
                       responses=ModelCallError("transient 500"))
    with pytest.raises(ModelCallError):
        c._extract_edges_window("Ari waters the fern every Sunday.")
    assert c._extract_cache.count() == 0         # the failure left no cache entry

    def ok_chat(model, system, user, **kw):
        calls["n"] += 1
        return '{"triples": []}'

    monkeypatch.setattr(c, "chat", ok_chat)
    c._extract_edges_window("Ari waters the fern every Sunday.")
    assert c._extract_cache.count() == 1         # retry after recovery caches normally


def test_flag_off_never_touches_the_cache(fresh_settings, tmp_path, monkeypatch):
    c, calls = _client(fresh_settings, tmp_path, monkeypatch, enabled=False)
    c._extract_edges_window("Ari waters the fern every Sunday.")
    c._extract_edges_window("Ari waters the fern every Sunday.")
    assert calls["n"] == 2                       # no cache: both calls paid
    assert c._extract_cache is None
    assert not (tmp_path / "data" / "extract_cache.sqlite").exists()


def test_extract_combined_halves_calls_and_feeds_both_channels(fresh_settings, tmp_path, monkeypatch):
    """EXTRACT_COMBINED: one call per window yields identical triples AND claims; flag off
    keeps the two-call path byte-identical."""
    from dataclasses import replace as _replace

    combined_raw = (
        '{"triples": [{"src":"Ari","relation":"waters","dst":"the fern",'
        '"fact":"Ari waters the fern"}],'
        ' "claims": [{"claim_type":"state","subject":"Ari","predicate":"waters",'
        '"object":"the fern","value":"Ari waters the fern weekly",'
        '"proof_atom":"Ari waters the fern every Sunday."}]}'
    )
    s = _replace(fresh_settings, data_dir=tmp_path / "data", extract_combined_enabled=True)
    c = DashScopeClient(s)
    calls = {"n": 0}

    def fake_chat(model, system, user, **kw):
        calls["n"] += 1
        return combined_raw

    monkeypatch.setattr(c, "chat", fake_chat)
    triples, claims = c.extract_edges_and_claims_bounded("Ari waters the fern every Sunday.")
    assert calls["n"] == 1
    assert triples and triples[0]["src"] == "Ari"
    assert claims and claims[0]["proof_atom"].startswith("Ari waters")

    # truncated combined payload: salvage keeps the complete objects per channel
    truncated = combined_raw[:combined_raw.rfind('"claims"') + 200]
    monkeypatch.setattr(c, "chat", lambda *a, **k: truncated)
    t2, c2 = c.extract_edges_and_claims_bounded("different window text entirely")
    assert t2 and t2[0]["src"] == "Ari"          # the complete triple survives truncation
