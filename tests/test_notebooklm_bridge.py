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
    NotebookLMBridge,
    NotebookLMError,
    format_source,
)
from eidetic.models import ClaimRecord, MemoryRecord, Scope

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


def test_cli_backend_runs_per_source():
    ran = []
    be = CliBackend(runner=lambda args: ran.append(args) or "")
    be.batch_create_sources("nbk_1", [{"text_content": "a", "display_name": "n1"},
                                      {"text_content": "b", "display_name": "n2"}])
    assert len(ran) == 2
    q = CliBackend(runner=lambda args: "the answer")
    assert "UNVERIFIED" in q.query("nbk_1", "q?")["backend"]


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
