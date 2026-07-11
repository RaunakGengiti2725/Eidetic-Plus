"""Host-facing structured recall surface.

The normal ask path already uses SMQE before retrieval. These tests prove host agents can call the
same typed path directly and inspect plan/backend/supports/citations without invoking a generator.
"""
from __future__ import annotations

import pytest

from eidetic.models import ClaimRecord, MemoryRecord, Scope


def _seed_claim_backed_memory(eng, scope: Scope) -> MemoryRecord:
    text = "User: I keep my climbing pass at Blue Arch Gym."
    content_hash, raw_uri = eng.substrate.put(text.encode("utf-8"))
    rec = MemoryRecord(
        memory_id="pass-memory",
        text=text,
        source="user",
        scope=scope,
        valid_at=1_700_000_000.0,
        content_hash=content_hash,
        raw_uri=raw_uri,
        raw_bytes_len=len(text.encode("utf-8")),
    )
    eng.store.upsert_record(rec)
    eng.store.add_claim(ClaimRecord(
        claim_type="state",
        scope=scope,
        subject="user",
        predicate="keep climbing pass",
        object="Blue Arch Gym",
        source_memory_id=rec.memory_id,
        proof_atom="User: I keep my climbing pass at Blue Arch Gym.",
        valid_at=rec.valid_at,
    ))
    return rec


def test_engine_structured_recall_exposes_plan_supports_and_proof(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope = Scope(namespace="structured-host")
    rec = _seed_claim_backed_memory(eng, scope)

    out = eng.structured_recall(
        "Where do I keep my climbing pass?",
        scope=scope,
        as_of=rec.valid_at + 1,
    )

    assert out["answered"] is True
    assert out["abstained"] is False
    assert out["verified"] is True
    assert out["status"] == "VERIFIED"
    assert out["draft"] == ""
    assert out["immutable_proof"] is True
    assert out["proof_link_checks"] == len(out["citations"]) == 1
    assert out["generated_by"] == "smqe"
    assert out["plan"]["op"] == "latest_value"
    assert out["backend"] == "claim"
    assert out["supports"][0]["memory_id"] == rec.memory_id
    assert out["supports"][0]["claim_id"]
    assert out["citations"][0]["content_hash"] == rec.content_hash
    assert out["citations"][0]["raw_uri"] == rec.raw_uri
    assert eng.substrate.verify(out["citations"][0]["content_hash"]) is True
    assert "Blue Arch Gym" in out["answer"]


def test_engine_structured_recall_abstains_without_active_memory(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    out = eng.structured_recall(
        "Where do I keep my climbing pass?",
        scope=Scope(namespace="empty-structured"),
    )

    assert out["answered"] is False
    assert out["abstained"] is True
    assert out["status"] == "ABSTAINED"
    assert out["answer"] == ""
    assert out["failure_reason"] == "no_active_records"
    assert out["immutable_proof"] is False
    assert out["proof_link_checks"] == 0
    assert out["citations"] == []


def test_engine_structured_recall_abstains_without_immutable_source_bytes(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope = Scope(namespace="structured-fake-proof")
    rec = MemoryRecord(
        memory_id="fake-memory",
        text="User: I keep my climbing pass at Blue Arch Gym.",
        source="user",
        scope=scope,
        valid_at=1_700_000_000.0,
        content_hash="not-a-real-sha",
        raw_uri="cas://not-a-real-sha",
    )
    eng.store.upsert_record(rec)
    eng.store.add_claim(ClaimRecord(
        claim_type="state",
        scope=scope,
        subject="user",
        predicate="keep climbing pass",
        object="Blue Arch Gym",
        source_memory_id=rec.memory_id,
        proof_atom="User: I keep my climbing pass at Blue Arch Gym.",
        valid_at=rec.valid_at,
    ))

    out = eng.structured_recall(
        "Where do I keep my climbing pass?",
        scope=scope,
        as_of=rec.valid_at + 1,
    )

    assert out["answered"] is False
    assert out["abstained"] is True
    assert out["verified"] is False
    assert out["status"] == "ABSTAINED"
    assert out["answer"] == ""
    assert out["draft"] == "Blue Arch Gym"
    assert out["immutable_proof"] is False
    assert out["proof_link_checks"] == 0
    assert out["failure_reason"] == "missing_immutable_proof"


def test_mcp_structured_recall_tool_exposes_verified_claim_backend(fresh_settings, monkeypatch):
    import eidetic.mcp_server as mcp_server
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope = Scope(namespace="structured-mcp")
    rec = _seed_claim_backed_memory(eng, scope)
    monkeypatch.setattr(mcp_server, "_engine", eng)
    try:
        out = mcp_server.structured_recall(
            "Where do I keep my climbing pass?",
            namespace=scope.namespace,
            as_of=rec.valid_at + 1,
        )
        assert out["answered"] is True
        assert out["immutable_proof"] is True
        assert out["proof_link_checks"] == 1
        assert out["backend"] == "claim"
        assert out["citations"][0]["memory_id"] == rec.memory_id
        assert out["citations"][0]["content_hash"] == rec.content_hash
    finally:
        monkeypatch.setattr(mcp_server, "_engine", None)


def test_api_structured_recall_route_exposes_verified_claim_backend(fresh_settings, monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import eidetic.api as api_mod
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope = Scope(namespace="structured-api")
    rec = _seed_claim_backed_memory(eng, scope)
    monkeypatch.setattr(api_mod, "_engine", eng)
    try:
        r = TestClient(api_mod.app).post("/api/structured_recall", json={
            "query": "Where do I keep my climbing pass?",
            "namespace": scope.namespace,
            "as_of": rec.valid_at + 1,
        })
        assert r.status_code == 200
        out = r.json()
        assert out["answered"] is True
        assert out["immutable_proof"] is True
        assert out["proof_link_checks"] == 1
        assert out["backend"] == "claim"
        assert out["citations"][0]["raw_uri"] == rec.raw_uri
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
