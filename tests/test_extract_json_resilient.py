"""Truncation-resilient triple-extraction parsing (forgetting-machine bench robustness).

A dense chunk can make the extractor emit more triples than max_tokens allows, cutting the JSON
array mid-object. Strict json.loads then raised and aborted the whole consolidation/run.
_parse_triples recovers every COMPLETE triple the model actually emitted and drops only the cut-off
trailing one -- it never fabricates. Pure functions, fully offline."""
from __future__ import annotations

from eidetic.dashscope_client import _parse_triples, _salvage_json_objects


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
