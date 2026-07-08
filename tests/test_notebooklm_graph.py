"""eidetic -> NotebookLM GRAPH-NATIVE serializer: offline unit tests. No network, no
Google auth -- the store/engine are synthetic fakes. Proves the verified-claim-graph
serializer emits a compact, provenance-carrying source with four labeled regions, honest
boundary strings, correct supersession pointers, and hub-first ordering.

Ground truth these tests pin (all verified against the code):
- store.all_edges(scope, include_inferred=False) returns CLOSED edges too (store.py:466-473,
  no invalid_at/expired_at filter) -> powers HISTORY.
- Edge.supersedes is the edge_id of the CLOSED predecessor and lives on the SUCCESSOR
  (graph.py:83). So HISTORY renders "superseded by <successor memory ref>" via a reverse
  index, never supersedes[:16] (which is an edge_id -> dangling token).
- graph.node_features keys are _norm(src)/_norm(dst) (graph.py:159), so grouping by
  _norm(e.src) aligns with the degree/ppr lookup.
- The short ref eidetic:<memory_id[:16]> must match _EIDETIC_REF_RE = eidetic:[..]{4,32}.
"""
from __future__ import annotations

from eidetic.integrations.notebooklm import (
    NotebookLMBridge,
    _EIDETIC_REF_RE,
    format_graph_source,
)
from eidetic.graph import CO_ACTIVATED, _norm
from eidetic.models import Edge, MemoryRecord, Scope

_SCOPE = Scope(namespace="nb-graph")


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------
def _rec(mid, ch, *, text="", source="user", valid_at=1_700_000_000.0):
    return MemoryRecord(text=text, source=source, scope=_SCOPE, valid_at=valid_at,
                        memory_id=mid, content_hash=ch, raw_uri=f"cas://{ch}")


def _edge(src, dst, relation, mid, *, edge_id=None, supersedes=None, valid_at=1_700_000_000.0,
          invalid_at=None, inferred=False, pruned=False):
    kw = dict(src=src, dst=dst, relation=relation, source_memory_id=mid,
              valid_at=valid_at, invalid_at=invalid_at, inferred=inferred, pruned=pruned,
              scope=_SCOPE)
    if edge_id is not None:
        kw["edge_id"] = edge_id
    if supersedes is not None:
        kw["supersedes"] = supersedes
    if invalid_at is not None:
        kw["expired_at"] = invalid_at
    return Edge(**kw)


def _by_id(records):
    return {r.memory_id: r for r in records}


# --------------------------------------------------------------------------
# (a) superseded fact appears in HISTORY, not ACTIVE FACTS
# --------------------------------------------------------------------------
def test_superseded_edge_goes_to_history_not_active():
    at = 1_700_500_000.0
    closed = _edge("Priya", "Boston", "lives_in", "mem_old0000000001",
                   edge_id="edge_old", valid_at=1_700_000_000.0, invalid_at=1_700_200_000.0)
    active = _edge("Priya", "Berlin", "lives_in", "mem_new0000000002",
                   edge_id="edge_new", supersedes="edge_old", valid_at=1_700_200_000.0)
    recs = [_rec("mem_old0000000001", "a" * 64, text="Priya lived in Boston."),
            _rec("mem_new0000000002", "b" * 64, text="Priya moved to Berlin.")]
    out = format_graph_source([closed, active], _by_id(recs), scope_label="nb-graph", at=at)
    text = out["text_content"]
    active_region = text.split("--- ACTIVE FACTS")[1].split("--- END ACTIVE FACTS")[0]
    history_region = text.split("--- HISTORY")[1].split("--- END HISTORY")[0]
    assert "Berlin" in active_region and "Boston" not in active_region
    assert "Boston" in history_region and "Berlin" not in history_region.split("superseded by")[0]


# --------------------------------------------------------------------------
# (b) every inline eidetic:<id> token is in the LEGEND and matches the regex
# --------------------------------------------------------------------------
def test_every_inline_token_is_in_legend_and_matches_regex():
    at = 1_700_500_000.0
    closed = _edge("Priya", "Boston", "lives_in", "mem_old0000000001",
                   edge_id="edge_old", valid_at=1_700_000_000.0, invalid_at=1_700_200_000.0)
    active = _edge("Priya", "Berlin", "lives_in", "mem_new0000000002",
                   edge_id="edge_new", supersedes="edge_old", valid_at=1_700_200_000.0)
    other = _edge("Priya", "Acme", "works_at", "mem_wrk0000000003", edge_id="edge_wrk")
    recs = [_rec("mem_old0000000001", "a" * 64), _rec("mem_new0000000002", "b" * 64),
            _rec("mem_wrk0000000003", "c" * 64)]
    out = format_graph_source([closed, active, other], _by_id(recs), scope_label="nb-graph", at=at)
    text = out["text_content"]
    legend = text.split("--- PROVENANCE LEGEND")[1].split("--- END LEGEND")[0]
    tokens = set(_EIDETIC_REF_RE.findall(text))
    assert tokens, "expected inline eidetic tokens"
    for tok in tokens:
        assert 4 <= len(tok) <= 32
        assert f"eidetic:{tok}" in legend, f"dangling token {tok} not in legend"


# --------------------------------------------------------------------------
# (c) round-trip on a SUPERSEDED token: history token resolves to its hash
# --------------------------------------------------------------------------
def test_superseded_token_round_trips_through_resolver():
    """Guards the Part-A3 decision: _resolve_provenance widened to all_records, so a
    history-section token (superseded record) still maps to its immutable content hash."""
    at = 1_700_500_000.0
    closed = _edge("Priya", "Boston", "lives_in", "mem_old0000000001",
                   edge_id="edge_old", valid_at=1_700_000_000.0, invalid_at=1_700_200_000.0)
    active = _edge("Priya", "Berlin", "lives_in", "mem_new0000000002",
                   edge_id="edge_new", supersedes="edge_old", valid_at=1_700_200_000.0)
    old_hash, new_hash = "a" * 64, "b" * 64
    all_recs = [_rec("mem_old0000000001", old_hash), _rec("mem_new0000000002", new_hash)]
    active_recs = [_rec("mem_new0000000002", new_hash)]  # only the CURRENT record is "active"

    class _Store:
        def all_edges(self, scope, include_inferred=False):
            return [closed, active]

        def all_records(self, scope):
            return list(all_recs)

        def active_records_at(self, t, scope):
            return list(active_recs)

    class _Graph:
        def node_features(self, at, scope):
            return {}

    class _Eng:
        def __init__(self):
            self.store = _Store()
            self.graph = _Graph()

    bridge = NotebookLMBridge(_Eng(), backend=None)
    src = bridge.build_graph_source("nb-graph", at=at)
    # the superseded record's short token
    superseded_tok = "eidetic:" + "mem_old0000000001"[:16]
    fabricated_gemini_answer = f"She used to live in Boston (source {superseded_tok})."
    prov = bridge._resolve_provenance("nb-graph", fabricated_gemini_answer)
    hashes = {p["content_sha256"] for p in prov}
    assert old_hash in hashes, "superseded token failed to resolve (resolver not widened to all_records)"


# --------------------------------------------------------------------------
# (d) compression_ratio > 1 on a multi-turn fixture (long records, few triples)
# --------------------------------------------------------------------------
def test_compression_ratio_gt_one_on_verbose_fixture():
    at = 1_700_500_000.0
    long_body = ("Over a long conversation Priya described in exhausting detail how she "
                 "relocated across several cities, changed jobs twice, and adopted a dog. " * 8)
    edges = [_edge("Priya", "Berlin", "lives_in", "mem_aaa0000000001", edge_id="e1"),
             _edge("Priya", "Acme", "works_at", "mem_bbb0000000002", edge_id="e2")]
    recs = [_rec("mem_aaa0000000001", "a" * 64, text=long_body),
            _rec("mem_bbb0000000002", "b" * 64, text=long_body)]
    out = format_graph_source(edges, _by_id(recs), scope_label="nb-graph", at=at)
    assert out["stats"]["compression_ratio"] > 1.0
    assert out["stats"]["raw_record_chars"] > out["stats"]["serialized_chars"]


# --------------------------------------------------------------------------
# (e) CO_ACTIVATED, pruned, and (default) inferred edges are excluded
# --------------------------------------------------------------------------
def test_coactivated_pruned_inferred_excluded():
    at = 1_700_500_000.0
    keep = _edge("Priya", "Berlin", "lives_in", "mem_keep000000001", edge_id="ek")
    coact = _edge("mem_x", "mem_y", CO_ACTIVATED, "mem_co00000000001", edge_id="ec")
    pruned = _edge("Priya", "Ghost", "knows", "mem_pru000000001", edge_id="ep", pruned=True)
    inferred = _edge("Priya", "Inferred", "maybe", "mem_inf000000001", edge_id="ei", inferred=True)
    recs = [_rec("mem_keep000000001", "a" * 64), _rec("mem_co00000000001", "b" * 64),
            _rec("mem_pru000000001", "c" * 64), _rec("mem_inf000000001", "d" * 64)]
    out = format_graph_source([keep, coact, pruned, inferred], _by_id(recs),
                              scope_label="nb-graph", at=at, include_inferred=False)
    text = out["text_content"]
    assert "Berlin" in text
    assert "co_activated" not in text and CO_ACTIVATED not in text.split("HONESTY")[0].split("scope:")[-1] or "Ghost" not in text
    assert "Ghost" not in text  # pruned excluded
    assert "Inferred" not in text  # inferred excluded by default
    assert out["stats"]["n_relations"] == 1


# --------------------------------------------------------------------------
# (f) honesty-header substrings present
# --------------------------------------------------------------------------
def test_honesty_header_substrings_present():
    edges = [_edge("Priya", "Berlin", "lives_in", "mem_aaa0000000001", edge_id="e1")]
    recs = [_rec("mem_aaa0000000001", "a" * 64)]
    out = format_graph_source(edges, _by_id(recs), scope_label="nb-graph", at=1_700_500_000.0)
    text = out["text_content"]
    for needle in (
        "~0 tokens on YOUR metered model",
        "NOT free globally",
        "NOT",  # verify-or-abstain boundary present
        "eidetic-verify-or-abstain",
        "NOT a row in the fixed-qwen-reader benchmark table",
    ):
        assert needle in text, f"missing honesty substring: {needle!r}"
    # SOTA/best appear ONLY inside the honest negation ("No SOTA/\"best\" claim"),
    # never as an asserted claim. Confirm the negation is present.
    low = text.lower()
    assert "no sota" in low
    assert "best in the world" not in low
    assert "strongest" not in low


# --------------------------------------------------------------------------
# (g) supersession pointer correctness (successor ref, not supersedes[:16])
# --------------------------------------------------------------------------
def test_supersession_pointer_uses_successor_memory_ref():
    at = 1_700_500_000.0
    closed = _edge("Priya", "Boston", "lives_in", "mem_old0000000001",
                   edge_id="edge_old", valid_at=1_700_000_000.0, invalid_at=1_700_200_000.0)
    successor = _edge("Priya", "Berlin", "lives_in", "mem_new0000000002",
                      edge_id="edge_new", supersedes="edge_old", valid_at=1_700_200_000.0)
    # a second closed edge with NO successor
    orphan = _edge("Priya", "Cairo", "born_in", "mem_orp0000000003",
                   edge_id="edge_orp", valid_at=1_600_000_000.0, invalid_at=1_650_000_000.0)
    recs = [_rec("mem_old0000000001", "a" * 64), _rec("mem_new0000000002", "b" * 64),
            _rec("mem_orp0000000003", "c" * 64)]
    out = format_graph_source([closed, successor, orphan], _by_id(recs),
                              scope_label="nb-graph", at=at)
    history = out["text_content"].split("--- HISTORY")[1].split("--- END HISTORY")[0]
    succ_ref = "eidetic:" + "mem_new0000000002"[:16]
    assert f"superseded by {succ_ref}" in history
    # NEVER render the edge_id as a memory ref
    assert "edge_old" not in history
    # orphan closed edge renders bare "(superseded)"
    orphan_line = [ln for ln in history.splitlines() if "Cairo" in ln][0]
    assert "(superseded)" in orphan_line and "superseded by" not in orphan_line


# --------------------------------------------------------------------------
# (h) hub ordering + max_entities truncation keeps the hub
# --------------------------------------------------------------------------
def test_hub_ordering_and_max_entities_keeps_hub():
    at = 1_700_500_000.0
    # X is a hub (degree 3), Y is a leaf (degree 1)
    edges = [
        _edge("X", "a", "r", "mem_x1000000000001", edge_id="ex1"),
        _edge("X", "b", "r", "mem_x2000000000002", edge_id="ex2"),
        _edge("X", "c", "r", "mem_x3000000000003", edge_id="ex3"),
        _edge("Y", "z", "r", "mem_y1000000000004", edge_id="ey1"),
    ]
    recs = [_rec("mem_x1000000000001", "1" * 64), _rec("mem_x2000000000002", "2" * 64),
            _rec("mem_x3000000000003", "3" * 64), _rec("mem_y1000000000004", "4" * 64)]
    nf = {"x": {"ppr": 0.4, "degree": 3.0}, "y": {"ppr": 0.1, "degree": 1.0},
          "a": {"ppr": 0.1, "degree": 1.0}, "b": {"ppr": 0.1, "degree": 1.0},
          "c": {"ppr": 0.1, "degree": 1.0}, "z": {"ppr": 0.1, "degree": 1.0}}
    out = format_graph_source(edges, _by_id(recs), scope_label="nb-graph", at=at,
                              node_features=nf)
    active = out["text_content"].split("--- ACTIVE FACTS")[1].split("--- END ACTIVE FACTS")[0]
    assert active.index("X") < active.index("Y")  # hub first
    # truncation keeps the hub, drops the leaf
    out1 = format_graph_source(edges, _by_id(recs), scope_label="nb-graph", at=at,
                               node_features=nf, max_entities=1)
    active1 = out1["text_content"].split("--- ACTIVE FACTS")[1].split("--- END ACTIVE FACTS")[0]
    assert "X" in active1 and "\nY" not in active1
    assert out1["stats"]["n_entities"] == 1


# --------------------------------------------------------------------------
# alphabetical fallback when node_features is empty
# --------------------------------------------------------------------------
def test_alpha_fallback_when_no_node_features():
    at = 1_700_500_000.0
    edges = [_edge("Zeta", "a", "r", "mem_z1000000000001", edge_id="ez"),
             _edge("Alpha", "b", "r", "mem_a1000000000002", edge_id="ea")]
    recs = [_rec("mem_z1000000000001", "1" * 64), _rec("mem_a1000000000002", "2" * 64)]
    out = format_graph_source(edges, _by_id(recs), scope_label="nb-graph", at=at,
                              node_features={})
    active = out["text_content"].split("--- ACTIVE FACTS")[1].split("--- END ACTIVE FACTS")[0]
    assert active.index("Alpha") < active.index("Zeta")
