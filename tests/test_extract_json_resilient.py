"""Truncation-resilient triple-extraction parsing (forgetting-machine bench robustness).

A dense chunk can make the extractor emit more triples than max_tokens allows, cutting the JSON
array mid-object. Strict json.loads then raised and aborted the whole consolidation/run.
_parse_triples recovers every COMPLETE triple the model actually emitted and drops only the cut-off
trailing one -- it never fabricates. Pure functions, fully offline."""
from __future__ import annotations

from eidetic.dashscope_client import (_is_content_moderation, _parse_triples,
                                      _salvage_json_objects)


def test_parse_triples_strict_valid():
    raw = '{"triples": [{"src":"A","relation":"r","dst":"B","fact":"A r B"}]}'
    assert _parse_triples(raw) == [{"src": "A", "relation": "r", "dst": "B", "fact": "A r B"}]


def test_parse_triples_handles_code_fence():
    raw = '```json\n{"triples": [{"src":"A","relation":"r","dst":"B"}]}\n```'
    assert len(_parse_triples(raw)) == 1


def test_parse_triples_salvages_truncated_array():
    # Two complete triples, then a third cut off by max_tokens (no closing brace/bracket).
    raw = ('{"triples": [{"src":"A","relation":"likes","dst":"tea","fact":"A likes tea"},'
           '{"src":"B","relation":"lives in","dst":"Paris","fact":"B lives in Paris"},'
           '{"src":"C","relation":"works at","dst":"Acme corp with a very long trailing')
    out = _parse_triples(raw)
    assert [t["src"] for t in out] == ["A", "B"]   # C dropped (incomplete), A+B salvaged


def test_parse_triples_drops_triples_missing_required_keys():
    raw = '{"triples": [{"src":"A","relation":"r"}, {"src":"X","relation":"r","dst":"Y"}]}'
    assert [t["src"] for t in _parse_triples(raw)] == ["X"]


def test_parse_triples_empty_on_garbage_or_empty():
    assert _parse_triples("totally not json") == []
    assert _parse_triples("") == []
    assert _parse_triples('{"triples": []}') == []


def test_salvage_finds_complete_objects_in_truncated_outer():
    # Outer object never closes (truncated); two inner objects do.
    raw = '{"triples":[{"a":1},{"b":2},{"c":'
    objs = _salvage_json_objects(raw)
    assert {"a": 1} in objs and {"b": 2} in objs
    assert all(isinstance(o, dict) for o in objs)


def test_salvage_respects_braces_inside_strings():
    # A literal "}" inside a string value must not prematurely close the object.
    raw = '{"triples":[{"src":"A","relation":"says","dst":"use {curly} braces","fact":"x"}'
    objs = _salvage_json_objects(raw)
    assert any(o.get("dst") == "use {curly} braces" for o in objs)


def test_is_content_moderation_detects_only_true_moderation():
    assert _is_content_moderation(
        "DashScope call failed (HTTP 400): Input data may contain inappropriate content.") is True
    assert _is_content_moderation(
        "DashScope call failed (HTTP 400): data_inspection_failed: blocked") is True
    # NOT a moderation error -> must not be skipped (these are different failure classes).
    assert _is_content_moderation(
        "DashScope call failed (HTTP 400): InvalidParameter: Range of input length") is False
    assert _is_content_moderation("DashScope call failed (HTTP 500): InternalError.Algo") is False
    # REGRESSION (review finding 1): a REAL 5xx whose body happens to mention 'data inspection'
    # must NOT be mis-classified as moderation and swallowed -- it is a server error.
    assert _is_content_moderation(
        "DashScope call failed (HTTP 500): internal, data inspection subsystem crashed") is False


def test_extract_edges_skips_moderated_window_gracefully(fresh_settings):
    from dataclasses import replace

    from eidetic.dashscope_client import DashScopeClient, ModelCallError

    settings = replace(fresh_settings, extract_chunking_enabled=False)
    client = DashScopeClient.__new__(DashScopeClient)
    client.settings = settings

    def moderated(*_a, **_k):
        raise ModelCallError(
            "DashScope call failed (HTTP 400): Input data may contain inappropriate content.")

    client.chat = moderated
    # Content the filter rejects yields NO triples but never aborts -- the raw stays in the
    # substrate; we just cannot extract a graph from moderated content.
    assert client.extract_edges("some flagged passage") == []


def test_extract_edges_still_raises_on_a_real_error(fresh_settings):
    from dataclasses import replace

    from eidetic.dashscope_client import DashScopeClient, ModelCallError
    import pytest

    settings = replace(fresh_settings, extract_chunking_enabled=False)
    client = DashScopeClient.__new__(DashScopeClient)
    client.settings = settings

    def boom(*_a, **_k):
        raise ModelCallError("DashScope call failed (HTTP 400): InvalidParameter: bad request")

    client.chat = boom
    # A non-moderation error is NOT silently swallowed -> it still fails loud.
    with pytest.raises(ModelCallError):
        client.extract_edges("normal text")
