"""eidetic -> NotebookLM bridge: offline unit tests. No network, no Google auth --
the HTTP session and CLI runner are injected fakes. Proves: (1) exported sources carry
eidetic's provenance header, (2) export batches every record through the backend,
(3) query pass-through is labeled unverified, (4) the enterprise backend refuses to run
without a token (fails loud, never silently 'succeeds')."""
from __future__ import annotations

import json

import pytest

from eidetic.integrations.notebooklm import (
    CliBackend,
    EnterpriseBackend,
    IncrementalSync,
    NotebookLMBridge,
    NotebookLMError,
    format_source,
)
from eidetic.models import ClaimRecord, Edge, MemoryRecord, Scope

_SCOPE = Scope(namespace="nb-test")


def _rec(text, mid="mem_abc123def456", ch="deadbeef", valid_at=1_700_000_000.0):
    return MemoryRecord(text=text, source="user", scope=_SCOPE, valid_at=valid_at,
                        memory_id=mid, content_hash=ch, raw_uri="mem://x")


def test_format_source_carries_provenance_header():
    rec = _rec("Priya adopted a beagle named Biscuit.")
    claim = ClaimRecord(claim_type="event", scope=_SCOPE, subject="Priya",
                        predicate="adopted", object="a beagle",
                        source_memory_id=rec.memory_id, proof_atom=rec.text, valid_at=10.0)
    src = format_source(rec, [claim])
    text = src["text_content"]
    assert "EIDETIC VERIFIED MEMORY" in text
    assert "content_sha256: deadbeef" in text
    assert rec.memory_id in text
    assert "Priya adopted a beagle" in text          # the claim is listed
    assert "Priya adopted a beagle named Biscuit." in text  # the body survives
    assert src["display_name"].startswith("eidetic:")


def test_format_source_marks_superseded_records():
    rec = _rec("old fact")
    rec.invalid_at = 1_700_100_000.0
    assert "superseded" in format_source(rec, [])["text_content"]


class _FakeStore:
    def __init__(self, records, claims=None):
        self._records = records
        self._claims = claims or {}

    def active_records_at(self, t, scope):
        return list(self._records)

    def all_records(self, scope):
        return list(self._records)

    def claims_by_source(self, memory_id):
        return self._claims.get(memory_id, [])


class _FakeEngine:
    def __init__(self, store):
        self.store = store


class _RecordingBackend:
    def __init__(self):
        self.batches = []

    def batch_create_sources(self, notebook_id, sources):
        self.batches.append((notebook_id, list(sources)))
        return {"created": len(sources)}

    def query(self, notebook_id, question):
        return {"answer": "gemini says hi"}


def test_export_namespace_batches_all_records():
    recs = [_rec(f"fact {i}", mid=f"mem_{i:03d}") for i in range(45)]
    eng = _FakeEngine(_FakeStore(recs))
    backend = _RecordingBackend()
    bridge = NotebookLMBridge(eng, backend)
    res = bridge.export_namespace("nb-test", "nbk_1", batch_size=20)
    assert res["exported"] == 45
    # 45 records / 20 per batch = 3 batches
    assert len(backend.batches) == 3
    assert sum(len(b[1]) for b in backend.batches) == 45
    # every pushed source carries the provenance header
    for _, sources in backend.batches:
        for s in sources:
            assert "EIDETIC VERIFIED MEMORY" in s["text_content"]


def test_export_skips_empty_and_reports_note():
    eng = _FakeEngine(_FakeStore([]))
    res = NotebookLMBridge(eng, _RecordingBackend()).export_namespace("nb-test", "nbk_1")
    assert res["exported"] == 0


def test_query_is_labeled_unverified():
    eng = _FakeEngine(_FakeStore([]))
    out = NotebookLMBridge(eng, _RecordingBackend()).query("nbk_1", "who?")
    assert "NOT verified" in out["caveat"]


def test_enterprise_backend_refuses_without_token():
    be = EnterpriseBackend(project_number="123", access_token=None,
                           session=object())  # session present so __post_init__ skips requests
    with pytest.raises(NotebookLMError):
        be.batch_create_sources("nbk_1", [{"text_content": "x", "display_name": "y"}])


def test_enterprise_backend_posts_provenance_payload():
    class _FakeResp:
        status_code = 200
        text = "{}"
        def json(self):
            return {"ok": True}

    class _FakeSession:
        def __init__(self):
            self.calls = []
        def post(self, url, headers=None, data=None):
            self.calls.append({"url": url, "headers": headers, "data": data})
            return _FakeResp()

    sess = _FakeSession()
    be = EnterpriseBackend(project_number="42", location="global",
                           access_token="tok", session=sess, endpoint_location="us")
    be.batch_create_sources("nbk_9", [{"text_content": "hello", "display_name": "d"}])
    call = sess.calls[0]
    assert "notebooks/nbk_9/sources:batchCreate" in call["url"]
    assert call["headers"]["Authorization"] == "Bearer tok"
    body = json.loads(call["data"])
    assert body["userContents"][0]["content"] == "hello"


def test_cli_backend_uses_real_nlm_command_syntax():
    """Pinned against the notebooklm-mcp-cli docs: `nlm source add <notebook> --text ...`
    (notebook POSITIONAL, no --notebook/--name flags) and `nlm notebook query <notebook> "q"`.
    A regression here means the live CLI call would fail on first use."""
    ran = []
    be = CliBackend(runner=lambda args: ran.append(args) or "")
    be.batch_create_sources("nbk_1", [{"text_content": "a", "display_name": "n1"},
                                      {"text_content": "b", "display_name": "n2"}])
    assert len(ran) == 2
    assert ran[0] == ["source", "add", "nbk_1", "--text", "a"]  # positional notebook, no flags
    assert "--notebook" not in ran[0] and "--name" not in ran[0]
    captured = []
    q = CliBackend(runner=lambda args: captured.append(args) or "the answer")
    out = q.query("nbk_1", "q?")
    assert captured[0] == ["notebook", "query", "nbk_1", "q?"]
    assert "UNVERIFIED" in out["backend"]


def test_cli_doctor_reports_status_and_commands_without_raising():
    """`doctor` is a preflight: it must NEVER raise, and it prints the exact commands +
    login state so the user sees the plan before a live export."""
    be = CliBackend(runner=lambda args: "logged in as user@example.com")
    doc = be.doctor()
    assert doc["backend"] == "cli"
    assert doc["reachable"] is True
    assert "logged in" in doc["logged_in"]
    assert "nlm source add" in doc["commands"]["add_source"]
    assert "nlm notebook query" in doc["commands"]["query"]


def test_cli_doctor_flags_not_logged_in():
    be = CliBackend(runner=lambda args: "You are NOT logged in")
    assert "NOT logged in" in be.doctor()["logged_in"]


def test_enterprise_doctor_reports_token_and_endpoints():
    be = EnterpriseBackend(project_number="42", access_token="tok", session=object())
    doc = be.doctor()
    assert doc["backend"] == "enterprise" and doc["token_present"] is True
    assert "notebooks/<id>/sources:batchCreate" in doc["commands"]["add_source"]
    d2 = EnterpriseBackend(project_number="42", access_token=None, session=object()).doctor()
    assert d2["token_present"] is False and "gcloud auth" in d2["hint"]


def test_reader_mode_zero_user_tokens_and_provenance():
    """NotebookLM reader mode: 0 tokens on the caller's model, and the free Gemini answer
    still resolves back to eidetic content hashes via the stamped eidetic:<id> refs."""
    rec = _rec("Priya moved to Berlin in 2021.", mid="mem_prov9999", ch="cafebabe")
    eng = _FakeEngine(_FakeStore([rec]))

    class _AnswerBackend:
        def query(self, notebook_id, question):
            # NotebookLM cites the source by the display-name token eidetic stamped.
            return {"answer": "Berlin (source eidetic:mem_prov9999).",
                    "backend": "nlm-cli (gemini free tier)"}

    out = NotebookLMBridge(eng, _AnswerBackend()).answer("nb-test", "Where did Priya move?", "nbk_1")
    assert out["user_llm_tokens"] == 0
    assert out["answer"].startswith("Berlin")
    assert out["provenance"] and out["provenance"][0]["content_sha256"] == "cafebabe"
    assert "NOT eidetic-verify" in out["caveat"]


# ==========================================================================
# Router-aware answer path (routed_answer) + incremental sync tests.
# Shared richer fakes exposing structured_recall / reflex_recall / all_edges /
# all_records / graph.node_features, all synthetic (no network, no model call).
# ==========================================================================
class _FakePacket:
    def __init__(self, cids, coverage=1.0):
        self._cids = list(cids)
        self.coverage = coverage

    def candidate_ids(self):
        return list(self._cids)


class _RouterStore:
    def __init__(self, records):
        self._records = list(records)

    def all_records(self, scope):
        return list(self._records)

    def active_records_at(self, t, scope):
        return list(self._records)

    def all_edges(self, scope, include_inferred=False):
        return []


class _RouterGraph:
    def node_features(self, at, scope):
        return {}


class _FakeAnswer:
    """Stands in for the real pydantic `Answer`: exposes .model_dump() (NOT a bare dict), so
    production code that does `engine.ask(...).model_dump()` exercises the real shape."""

    def __init__(self, answer, citations):
        self._answer = answer
        self._citations = citations

    def model_dump(self):
        return {"answer": self._answer, "citations": list(self._citations)}


class _RouterEngine:
    """A fake engine whose structured/reflex outputs are canned per test."""

    def __init__(self, records, structured, reflex_cids):
        self.store = _RouterStore(records)
        self.graph = _RouterGraph()
        self._structured = structured
        self._reflex_cids = reflex_cids
        self.recall_calls = []

    def structured_recall(self, query, *, scope=None, as_of=None):
        return dict(self._structured)

    def reflex_recall(self, query, *, scope=None, as_of=None, emit=True, begin_turn=True):
        return _FakePacket(self._reflex_cids)

    def ask(self, query, *, verify=True, scope=None, as_of=None, at=None):
        # Mirror the REAL Engine.ask contract: no `prove=` kwarg, returns an object whose
        # .model_dump() yields a dict with a `citations` list of dicts (as Citation.model_dump()
        # would). Records the verify flag so the test can assert the gate actually ran.
        self.recall_calls.append((query, verify))
        return _FakeAnswer(
            answer="gate-verified answer",
            citations=[{"memory_id": "mem_gate00000001", "content_hash": "f" * 64,
                        "raw_uri": "cas://" + "f" * 64, "valid_at": 42.0}],
        )


def _prov_answer_backend(token):
    class _B:
        def query(self, notebook_id, question):
            return {"answer": f"free gemini answer citing eidetic:{token}.",
                    "backend": "nlm free tier"}
    return _B()


def test_router_tier1_when_structured_verified():
    rec = _rec("Priya moved to Berlin.", mid="mem_t1000000000001", ch="a" * 64)
    struct = {"answered": True, "abstained": False, "verified": True,
              "immutable_proof": True, "confidence": 0.9, "answer": "Berlin",
              "citations": [{"memory_id": rec.memory_id, "content_hash": "a" * 64,
                             "raw_uri": "cas://" + "a" * 64}]}
    eng = _RouterEngine([rec], struct, reflex_cids=[rec.memory_id])
    bridge = NotebookLMBridge(eng, _prov_answer_backend("mem_t1000000000001"))
    out = bridge.routed_answer("nb-test", "Where?", "nbk_1", struct_tau=0.5)
    assert out["tier"] == 1
    assert out["gate_verified"] is True
    assert out["provenance_verb"] == "gate-verified"
    assert out["caller_llm_tokens"] > 0  # structured path is cheap but metered
    assert eng.recall_calls == []  # never escalated


def test_router_tier2_free_read_when_abstained_no_gate():
    rec = _rec("Priya moved to Berlin.", mid="mem_t2000000000001", ch="b" * 64)
    struct = {"answered": False, "abstained": True, "verified": False,
              "immutable_proof": False, "confidence": 0.0}
    eng = _RouterEngine([rec], struct, reflex_cids=[rec.memory_id])
    bridge = NotebookLMBridge(eng, _prov_answer_backend("mem_t2000000000001"))
    out = bridge.routed_answer("nb-test", "Where?", "nbk_1", require_gate_verification=False)
    assert out["tier"] == 2
    assert out["caller_llm_tokens"] == 0
    assert out["gate_verified"] is False
    assert out["provenance_verb"] == "provenance-mapped"
    # reflex cross-check populated and intersecting the resolved provenance
    assert rec.memory_id in out["reflex_cross_check"]["candidate_ids"]
    assert rec.memory_id in out["reflex_cross_check"]["intersection"]
    assert eng.recall_calls == []


def test_router_tier3_metered_only_when_abstained_and_gate_required():
    rec = _rec("Priya moved to Berlin.", mid="mem_t3000000000001", ch="c" * 64)
    struct = {"answered": False, "abstained": True, "verified": False,
              "immutable_proof": False, "confidence": 0.0}
    eng = _RouterEngine([rec], struct, reflex_cids=[rec.memory_id])
    bridge = NotebookLMBridge(eng, _prov_answer_backend("mem_t3000000000001"))
    out = bridge.routed_answer("nb-test", "Where?", "nbk_1", require_gate_verification=True)
    assert out["tier"] == 3
    assert out["gate_verified"] is True
    assert out["provenance_verb"] == "gate-verified"
    assert out["caller_llm_tokens"] > 0
    assert eng.recall_calls and eng.recall_calls[0][1] is True  # prove=True


def test_router_no_fallthrough_low_conf_verified_with_gate():
    """Advisor blocking-fix #1: answered+verified+immutable_proof but confidence<tau AND
    require_gate_verification=True must NOT fall through -- it escalates to Tier 3."""
    rec = _rec("Priya moved to Berlin.", mid="mem_t4000000000001", ch="d" * 64)
    struct = {"answered": True, "abstained": False, "verified": True,
              "immutable_proof": True, "confidence": 0.2,  # below tau
              "citations": [{"memory_id": rec.memory_id, "content_hash": "d" * 64,
                             "raw_uri": "cas://" + "d" * 64}]}
    eng = _RouterEngine([rec], struct, reflex_cids=[rec.memory_id])
    bridge = NotebookLMBridge(eng, _prov_answer_backend("mem_t4000000000001"))
    out = bridge.routed_answer("nb-test", "Where?", "nbk_1",
                               require_gate_verification=True, struct_tau=0.8)
    assert out["tier"] == 3  # escalated, not dropped
    assert eng.recall_calls  # gate ran


def test_router_return_carries_honesty_boundaries():
    rec = _rec("Priya moved to Berlin.", mid="mem_t5000000000001", ch="e" * 64)
    struct = {"answered": False, "abstained": True, "verified": False,
              "immutable_proof": False, "confidence": 0.0}
    eng = _RouterEngine([rec], struct, reflex_cids=[rec.memory_id])
    bridge = NotebookLMBridge(eng, _prov_answer_backend("mem_t5000000000001"))
    out = bridge.routed_answer("nb-test", "Where?", "nbk_1")
    blob = " ".join(str(v) for v in out["honesty"].values())
    low = blob.lower()
    # the boundary strings NEGATE these claims, so the words appear only in a "No SOTA" /
    # "NOT ... best" negation -- assert the honest negations are present, not absent.
    assert "not free globally" in low
    assert "not" in low and "eidetic-verify-or-abstain" in low
    assert "no sota" in low  # explicitly disclaims SOTA
    assert "provenance" in low


# ---- graph export wiring ----
class _GraphStore:
    def __init__(self, edges, records):
        self._edges = list(edges)
        self._records = list(records)

    def all_edges(self, scope, include_inferred=False):
        return [e for e in self._edges if include_inferred or not e.inferred]

    def all_records(self, scope):
        return list(self._records)

    def active_records_at(self, t, scope):
        return list(self._records)


class _GraphEngine:
    def __init__(self, edges, records):
        self.store = _GraphStore(edges, records)
        self.graph = _RouterGraph()


def test_build_graph_source_offline_and_export_wires_backend():
    edges = [Edge(src="Priya", dst="Berlin", relation="lives_in",
                  source_memory_id="mem_g1000000000001", scope=_SCOPE, valid_at=10.0)]
    recs = [_rec("Priya lives in Berlin.", mid="mem_g1000000000001", ch="a" * 64)]
    eng = _GraphEngine(edges, recs)
    # offline (backend=None) still measures compression
    bridge = NotebookLMBridge(eng, backend=None)
    src = bridge.build_graph_source("nb-test", at=100.0)
    assert src["stats"]["n_relations"] == 1
    assert "EIDETIC VERIFIED CLAIM GRAPH" in src["text_content"]
    # export pushes exactly one graph source through the backend
    backend = _RecordingBackend()
    bridge2 = NotebookLMBridge(eng, backend)
    res = bridge2.export_graph("nb-test", "nbk_1")
    assert res["exported"] == 1
    assert len(backend.batches) == 1 and len(backend.batches[0][1]) == 1


def test_export_namespace_include_graph_appends_source():
    """MUST 'additional source': export_namespace(include_graph=True) appends the graph
    as one extra source without changing default behaviour."""
    edges = [Edge(src="Priya", dst="Berlin", relation="lives_in",
                  source_memory_id="mem_h1000000000001", scope=_SCOPE, valid_at=10.0)]
    recs = [_rec("Priya lives in Berlin.", mid="mem_h1000000000001", ch="a" * 64)]
    eng = _GraphEngine(edges, recs)
    backend = _RecordingBackend()
    bridge = NotebookLMBridge(eng, backend)
    base = bridge.export_namespace("nb-test", "nbk_1")
    n_base = base["exported"]
    backend2 = _RecordingBackend()
    bridge2 = NotebookLMBridge(_GraphEngine(edges, recs), backend2)
    withg = bridge2.export_namespace("nb-test", "nbk_1", include_graph=True)
    assert withg["exported"] == n_base + 1
    pushed = [s for _, ss in backend2.batches for s in ss]
    assert any("EIDETIC VERIFIED CLAIM GRAPH" in s["text_content"] for s in pushed)


# ---- incremental content-hash sync ----
def test_sync_idempotent_by_content_hash(tmp_path):
    recs = [_rec("fact one", mid="mem_s1000000000001", ch="h1"),
            _rec("fact two", mid="mem_s2000000000002", ch="h2")]
    eng = _GraphEngine([], recs)
    backend = _RecordingBackend()
    bridge = NotebookLMBridge(eng, backend)
    manifest = str(tmp_path / "sync_manifest.json")
    r1 = IncrementalSync(bridge, manifest).sync("nb-test", "nbk_1")
    assert r1["pushed"] == 2 and r1["skipped"] == 0
    r2 = IncrementalSync(bridge, manifest).sync("nb-test", "nbk_1")
    assert r2["pushed"] == 0 and r2["skipped"] == 2  # idempotent


def test_sync_pushes_only_new_content_hash(tmp_path):
    recs = [_rec("fact one", mid="mem_s1000000000001", ch="h1")]
    eng = _GraphEngine([], recs)
    backend = _RecordingBackend()
    bridge = NotebookLMBridge(eng, backend)
    manifest = str(tmp_path / "sync_manifest.json")
    IncrementalSync(bridge, manifest).sync("nb-test", "nbk_1")
    # add a new record with a NEW content hash; the old one is unchanged
    eng.store._records.append(_rec("fact three", mid="mem_s3000000000003", ch="h3"))
    r = IncrementalSync(bridge, manifest).sync("nb-test", "nbk_1")
    assert r["pushed"] == 1 and r["skipped"] == 1


# ---- regression: fake-masks-reality guard for the Tier 3 verified-reader path ----
def test_real_engine_has_ask_not_recall():
    """The flagship gate-verified path (routed_answer Tier 3) MUST call an API the real
    Engine actually exposes. The prior bug called `engine.recall(...)`, which does not exist
    on the real class -- it only passed tests because a fake supplied it. Assert against the
    REAL class so the fakes can never drift from reality again."""
    from eidetic.engine import Engine
    assert hasattr(Engine, "ask"), "Engine.ask is the verify-or-abstain entry point"
    assert not hasattr(Engine, "recall"), (
        "Engine has NO `recall` method; routed_answer Tier 3 must use ask()/prove() "
        "(see mcp_server.recall). A `recall` attribute here would mean the fix regressed.")


def test_router_tier3_calls_ask_with_verify_and_maps_provenance():
    """Tier 3 uses ask(verify=True) (NOT the non-existent recall) and maps the returned
    Citation dicts back to content-hash provenance."""
    rec = _rec("Priya moved to Berlin.", mid="mem_t3000000000001", ch="c" * 64)
    struct = {"answered": False, "abstained": True, "verified": False,
              "immutable_proof": False, "confidence": 0.0}
    eng = _RouterEngine([rec], struct, reflex_cids=[rec.memory_id])
    bridge = NotebookLMBridge(eng, _prov_answer_backend("mem_t3000000000001"))
    out = bridge.routed_answer("nb-test", "Where?", "nbk_1", require_gate_verification=True)
    assert out["tier"] == 3
    assert out["gate_verified"] is True
    # ask() was called with verify=True (the gate actually ran)
    assert eng.recall_calls and eng.recall_calls[0][1] is True
    # the Citation dicts from ask().model_dump() mapped into provenance with content hashes
    assert out["provenance"] and out["provenance"][0]["content_sha256"] == "f" * 64
    assert out["provenance"][0]["memory_id"] == "mem_gate00000001"


# ---- regression: _resolve_provenance must not misattribute a short/truncated token ----
def test_resolve_provenance_rejects_short_prefix_misattribution():
    """A truncated/hallucinated citation token (e.g. `eidetic:mem_`) must NOT be mapped to
    every `mem_`-prefixed record. Only exact memory_id or memory_id[:16] matches count."""
    recs = [
        _rec("fact A", mid="mem_aaaa000000000001", ch="ch_a"),
        _rec("fact B", mid="mem_bbbb000000000002", ch="ch_b"),
        _rec("fact C", mid="mem_cccc000000000003", ch="ch_c"),
    ]
    eng = _FakeEngine(_FakeStore(recs))
    bridge = NotebookLMBridge(eng, _RecordingBackend())
    # A lazy 4-char token that is a prefix of all three memory_ids.
    prov = bridge._resolve_provenance("nb-test", "see eidetic:mem_ for details")
    assert prov == [], "short prefix must resolve to nothing, not to every mem_ record"
    # An EXACT 16-char short id still resolves.
    prov2 = bridge._resolve_provenance("nb-test", "see eidetic:mem_bbbb000000000002")
    assert len(prov2) == 1 and prov2[0]["content_sha256"] == "ch_b"


# ---- regression: compression_ratio numerator counts only rendered records ----
def test_compression_ratio_ignores_unrendered_records():
    """A big record that is truncated out by max_entities (never serialized) must NOT inflate
    raw_record_chars -- the surfaced ratio measures only what was actually compacted."""
    edges = [
        Edge(src="Priya", dst="Berlin", relation="lives_in",
             source_memory_id="mem_ref00000000001", scope=_SCOPE, valid_at=10.0),
        Edge(src="Zzz", dst="Moon", relation="visits",
             source_memory_id="mem_drop00000000002", scope=_SCOPE, valid_at=10.0),
    ]
    records_by_id = {
        "mem_ref00000000001": _rec("Priya lives in Berlin.", mid="mem_ref00000000001", ch="a" * 64),
        # dropped entity's record is HUGE; it must not enter raw_record_chars when truncated out.
        "mem_drop00000000002": _rec("X" * 5000, mid="mem_drop00000000002", ch="b" * 64),
    }
    from eidetic.integrations.notebooklm import format_graph_source
    # node_features ranks "priya" as the hub so max_entities=1 keeps it and drops "zzz".
    nf = {"priya": {"degree": 5.0, "ppr": 1.0}, "zzz": {"degree": 0.0, "ppr": 0.0}}
    src = format_graph_source(edges, records_by_id, scope_label="nb-test", at=100.0,
                              node_features=nf, max_entities=1)
    assert src["stats"]["n_entities"] == 1
    # only the rendered record's ~22 chars count -- not the 5000-char dropped one.
    assert src["stats"]["raw_record_chars"] < 100
    # and the ratio is not inflated by the phantom 5000 chars.
    assert src["stats"]["compression_ratio"] < 1.0


def test_cost_report_measures_and_labels_by_construction(tmp_path):
    """The billable-caller-token report: rag-vector/mem0/eidetic are MEASURED from logs;
    the NotebookLM free-read row is 0 BY CONSTRUCTION and labeled as such (not measured)."""
    import json as _json
    from bench.notebooklm_cost import build_report

    d = tmp_path / "win"
    d.mkdir()
    (d / "rag-vector__run0.jsonl").write_text(
        "\n".join(_json.dumps({"query_tokens": t}) for t in (1800, 2000, 1900)))
    (d / "mem0__run0.jsonl").write_text(
        "\n".join(_json.dumps({"query_tokens": t}) for t in (380, 400)))
    rep = build_report([d])
    sys = rep["systems"]
    assert sys["rag-vector"]["caller_tokens_per_query"] == 1900  # measured median
    assert "MEASURED" in sys["rag-vector"]["basis"]
    free = sys["eidetic+notebooklm (routed, free-read tier)"]
    assert free["caller_tokens_per_query"] == 0
    assert "BY CONSTRUCTION" in free["basis"]
    assert "NOT gate-verified" in free["verified"]
    # honesty boundaries present in the claim
    assert "NOT free globally" in rep["honest_claim"] or "not free globally" in rep["honest_claim"].lower()
    assert "benchmark" in rep["honest_claim"]


def test_find_notebook_id_handles_every_nlm_json_shape():
    """The notebook-id resolver must survive every shape `nlm notebook list/create --json`
    might emit (a live-run bug: my awk grabbed '"title":'). Recursively finds the id and
    prefers a title match."""
    from eidetic.integrations.notebooklm import find_notebook_id
    # bare array
    assert find_notebook_id('[{"id":"nb_ABCDEFGH12","title":"My Memory"}]', "My Memory") == "nb_ABCDEFGH12"
    # {notebooks:[...]} with a distractor -> title match wins
    assert find_notebook_id(
        '{"notebooks":[{"id":"nb_OTHER111","title":"Other"},{"id":"nb_TARGET4567","title":"My Memory"}]}',
        "My Memory") == "nb_TARGET4567"
    # create-shape {notebook_id:...}
    assert find_notebook_id('{"notebook_id":"nb_CREATED890","title":"My Memory"}', "My Memory") == "nb_CREATED890"
    # nested {data:{items:[{notebookId, name}]}}
    assert find_notebook_id('{"data":{"items":[{"notebookId":"nb_NESTED777","name":"My Memory"}]}}', "My Memory") == "nb_NESTED777"
    # no title match -> first id; junk -> None
    assert find_notebook_id('[{"id":"nb_FIRST0001"},{"id":"nb_SECOND002"}]', "absent") == "nb_FIRST0001"
    assert find_notebook_id("not json at all", "x") is None
    assert find_notebook_id("{}", "x") is None
