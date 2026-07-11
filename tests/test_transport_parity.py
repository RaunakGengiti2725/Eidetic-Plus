from __future__ import annotations

import pytest

from eidetic.engine import Engine
from eidetic.models import ClaimRecord, MemoryRecord, Scope


def _seed(engine: Engine, scope: Scope) -> MemoryRecord:
    text = "User: I keep my climbing pass at Blue Arch Gym."
    content_hash, raw_uri = engine.substrate.put(text.encode("utf-8"))
    record = MemoryRecord(
        memory_id="transport-pass-memory",
        text=text,
        source="user",
        scope=scope,
        valid_at=1_700_000_000.0,
        content_hash=content_hash,
        raw_uri=raw_uri,
        raw_bytes_len=len(text.encode("utf-8")),
    )
    engine.store.upsert_record(record)
    engine.store.add_claim(ClaimRecord(
        claim_type="state",
        scope=scope,
        subject="user",
        predicate="keep climbing pass",
        object="Blue Arch Gym",
        source_memory_id=record.memory_id,
        proof_atom=text,
        valid_at=record.valid_at,
    ))
    return record


def _status(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _citation_contract(payload: dict) -> list[tuple]:
    return [
        (
            citation["memory_id"],
            citation["content_hash"],
            citation["raw_uri"],
            citation["nli_label"].value if hasattr(citation["nli_label"], "value")
            else citation["nli_label"],
        )
        for citation in payload.get("citations") or []
    ]


def test_verified_answer_and_proof_are_transport_invariant(fresh_settings, monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import eidetic.api as api
    import eidetic.mcp_server as mcp

    engine = Engine(fresh_settings)
    scope = Scope(namespace="transport-parity")
    record = _seed(engine, scope)
    monkeypatch.setattr(api, "_engine", engine)
    monkeypatch.setattr(mcp, "_engine", engine)

    direct_answer = engine.ask("Where do I keep my climbing pass?", scope=scope)
    direct = direct_answer.model_dump()
    direct["proof"] = engine.prove(direct_answer, check_refs=True)
    api_result = TestClient(api.app).post("/api/ask", json={
        "query": "Where do I keep my climbing pass?",
        "namespace": scope.namespace,
        "prove": True,
    }).json()
    mcp_result = mcp.recall(
        "Where do I keep my climbing pass?",
        namespace=scope.namespace,
        prove=True,
    )

    assert {_status(direct["status"]), _status(api_result["status"]),
            _status(mcp_result["status"])} == {"VERIFIED"}
    assert direct["answer"] == api_result["answer"] == mcp_result["answer"]
    assert _citation_contract(direct) == _citation_contract(api_result) == _citation_contract(mcp_result)
    assert _citation_contract(direct)[0][0] == record.memory_id
    assert direct["proof"]["refs_verified"] is True
    assert api_result["proof"]["refs_verified"] is True
    assert mcp_result["proof"]["refs_verified"] is True


def test_abstention_is_transport_invariant(fresh_settings, monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import eidetic.api as api
    import eidetic.mcp_server as mcp

    engine = Engine(fresh_settings)
    scope = Scope(namespace="transport-empty")
    monkeypatch.setattr(api, "_engine", engine)
    monkeypatch.setattr(mcp, "_engine", engine)

    direct = engine.ask("What is my launch code?", scope=scope).model_dump()
    api_result = TestClient(api.app).post("/api/ask", json={
        "query": "What is my launch code?",
        "namespace": scope.namespace,
        "verify": False,
    }).json()
    mcp_result = mcp.recall(
        "What is my launch code?", namespace=scope.namespace, verify=False)

    assert {_status(direct["status"]), _status(api_result["status"]),
            _status(mcp_result["status"])} == {"ABSTAINED"}
    assert direct["answer"] == api_result["answer"] == mcp_result["answer"]
    assert direct["verified"] is api_result["verified"] is mcp_result["verified"] is False
    assert direct["citations"] == api_result["citations"] == mcp_result["citations"] == []
