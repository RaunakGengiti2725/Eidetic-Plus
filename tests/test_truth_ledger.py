"""Track 3.1 truth ledger: the full chain from raw bytes to current truth. Extends the proof tree
with each citation's validity window + supersession chain (via fact_history) and a final
claim_status. Deterministic, no model call. claim_status derives ONLY from verified/note/NLI --
never downgraded by an empty supersession chain."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import Answer, Citation, MemoryRecord, NLILabel, Scope


class _Embed:
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


class _Reader(_Embed):
    def generate_answer(self, q, blocks, model=None):
        return "Alice works at Acme Corporation"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if "acme" in (premise or "").lower() else ("neutral", 0.2)


def _engine(fresh_settings, reader_cls=_Embed, **kw):
    s = replace(fresh_settings, **kw)
    return Engine(s, client=reader_cls(s.embed_dim))


def _cite(mid, label=NLILabel.ENTAILMENT, valid_at=100.0):
    return Citation(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}", source="user",
                    valid_at=valid_at, snippet="snip", nli_label=label, nli_score=0.9)


def test_truth_ledger_enriches_validity_window_and_status(fresh_settings):
    e = _engine(fresh_settings)
    e.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h_m1", text="Alice works at Acme",
                                       scope=Scope(), valid_at=100.0))
    ans = Answer(question="where does Alice work", answer="Acme", verified=True, confidence=0.9,
                 citations=[_cite("m1")])
    led = e.truth_ledger(ans)
    assert led["claim_status"] == "verified"
    item = led["evidence"][0]
    assert item["validity_window"]["valid_at"] == 100.0
    assert item["validity_window"]["invalid_at"] is None
    assert item["is_current"] is True


def test_truth_ledger_includes_supersession_chain(fresh_settings):
    e = _engine(fresh_settings)
    e.store.upsert_record(MemoryRecord(memory_id="m2", content_hash="h_m2",
                                       text="Alice now at Globex", scope=Scope(), valid_at=200.0))
    e.graph.add_fact("Alice", "works_at", "Acme", fact="Alice works at Acme",
                     source_memory_id="m0", valid_at=100.0, scope=Scope())
    e.graph.add_fact("Alice", "works_at", "Globex", fact="Alice works at Globex",
                     source_memory_id="m2", valid_at=200.0, scope=Scope())
    ans = Answer(question="where does Alice work", answer="Globex", verified=True, confidence=0.9,
                 citations=[_cite("m2", valid_at=200.0)])
    item = e.truth_ledger(ans)["evidence"][0]
    chains = item.get("supersession_chains", [])
    assert chains and chains[0]["relation"] == "works_at"
    assert [h["value"] for h in chains[0]["history"]] == ["Acme", "Globex"]   # oldest first


def test_truth_ledger_status_abstained(fresh_settings):
    e = _engine(fresh_settings)
    ans = Answer(question="q", answer="...", verified=False,
                 note="abstained: insufficient evidence (coverage 0.10)")
    assert e.truth_ledger(ans)["claim_status"] == "abstained"


def test_truth_ledger_status_contradicted(fresh_settings):
    e = _engine(fresh_settings)
    e.store.upsert_record(MemoryRecord(memory_id="m3", content_hash="h_m3", text="x",
                                       scope=Scope(), valid_at=1.0))
    ans = Answer(question="q", answer="...", verified=False, note="",
                 citations=[_cite("m3", label=NLILabel.CONTRADICTION)])
    assert e.truth_ledger(ans)["claim_status"] == "contradicted"


def test_empty_supersession_chain_never_downgrades_status(fresh_settings):
    e = _engine(fresh_settings)
    e.store.upsert_record(MemoryRecord(memory_id="m4", content_hash="h_m4", text="standalone fact",
                                       scope=Scope(), valid_at=1.0))   # sources no edges
    ans = Answer(question="q", answer="ok", verified=True, citations=[_cite("m4")])
    item = e.truth_ledger(ans)["evidence"][0]
    assert e.truth_ledger(ans)["claim_status"] == "verified"
    assert "supersession_chains" not in item          # empty -> omitted, not a downgrade


def test_truth_ledger_http_and_mcp(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import eidetic.api as api_mod
    import eidetic.mcp_server as mcp_server
    from eidetic.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    get_settings.cache_clear()
    try:
        eng = Engine(replace(get_settings(), rerank_enabled=False, semantic_cache_enabled=False),
                     client=_Reader(get_settings().embed_dim))
        eng.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
        monkeypatch.setattr(api_mod, "_engine", eng)
        monkeypatch.setattr(mcp_server, "_engine", eng)
        r = TestClient(api_mod.app).get("/api/truth_ledger?query=where does Alice work&namespace=default")
        assert r.status_code == 200 and "claim_status" in r.json()
        out = mcp_server.truth_ledger("where does Alice work", namespace="default")
        assert "claim_status" in out and "evidence" in out
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        monkeypatch.setattr(mcp_server, "_engine", None)
        get_settings.cache_clear()


def test_truth_ledger_http_preserves_recall_paths(tmp_path, monkeypatch):
    # ask + truth_ledger must run on ONE threadpool thread, else the thread-local last_trace is lost
    # and recall_paths silently vanish over HTTP.
    import pytest
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import eidetic.api as api_mod
    from eidetic.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    get_settings.cache_clear()
    try:
        eng = Engine(replace(get_settings(), rerank_enabled=False, semantic_cache_enabled=False,
                             recall_trace_enabled=True), client=_Reader(get_settings().embed_dim))
        eng.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
        monkeypatch.setattr(api_mod, "_engine", eng)
        led = TestClient(api_mod.app).get(
            "/api/truth_ledger?query=where does Alice work&namespace=default").json()
        assert led["evidence"] and "recall_paths" in led["evidence"][0]
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
        get_settings.cache_clear()
