"""Host-facing memory-region hint surfaces.

These tests keep the cocoon/router contract outside the benchmark harness too: API/MCP callers get
cheap route hints, but every hint resolves back to active, scoped raw memories with proof pointers.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from eidetic.models import DerivedRecord, MemoryRecord, Scope


def _record(mid: str, text: str, *, scope: Scope, valid_at: float,
            invalid_at: float | None = None, expired_at: float | None = None) -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid,
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        content_hash=f"{mid}_content_hash_1234567890",
        raw_uri=f"cas://{mid}",
    )


def _engine_with_regions(fresh_settings):
    from eidetic.engine import Engine

    settings = replace(
        fresh_settings,
        gist_channel_enabled=True,
        persistent_bm25_enabled=False,
        reflex_recall_enabled=False,
    )
    eng = Engine(settings)
    scope = Scope(namespace="regions", agent_id="agent-a", project_id="project")
    other = Scope(namespace="regions", agent_id="agent-b", project_id="project")
    records = [
        _record("active", "active ginger tea route source", scope=scope, valid_at=900.0),
        _record("nested", "nested afternoon tea route source", scope=scope, valid_at=910.0),
        _record("invalid", "invalid stale route source", scope=scope, valid_at=800.0,
                invalid_at=950.0),
        _record("expired", "expired stale route source", scope=scope, valid_at=810.0,
                expired_at=960.0),
        _record("future", "future route source", scope=scope, valid_at=1100.0),
        _record("other", "other agent route source", scope=other, valid_at=900.0),
    ]
    for rec in records:
        eng.store.upsert_record(rec)
    eng.store.add_derived(DerivedRecord(
        cid="direct-region",
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text="ginger tea direct route",
        member_ids=["active", "invalid"],
    ))
    eng.store.add_derived(DerivedRecord(
        cid="child-cocoon",
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text="afternoon tea nested route",
        member_ids=["nested", "expired", "future", "other"],
    ))
    eng.store.add_derived(DerivedRecord(
        cid="parent-cocoon",
        kind="gist",
        namespace=scope.namespace,
        level=2,
        text="ginger tea parent cocoon route",
        member_ids=["child-cocoon"],
    ))
    return eng, scope


def _all_members(out: dict) -> set[str]:
    return {
        str(mid)
        for hint in out.get("hints", [])
        for mid in (hint.get("members", []) or [])
    }


def test_engine_region_hints_are_active_scoped_and_proof_linked(fresh_settings):
    eng, scope = _engine_with_regions(fresh_settings)

    out = eng.region_hints("ginger tea route", scope=scope, as_of=1000.0)

    assert out["enabled"] is True
    assert out["hint_count"] >= 2
    assert {h["region_id"] for h in out["hints"]} >= {"direct-region", "parent-cocoon"}
    members = _all_members(out)
    assert {"active", "nested"} <= members
    assert not {"invalid", "expired", "future", "other"}.intersection(members)
    hashes = {ch for h in out["hints"] for ch in h["content_hashes"]}
    raw_uris = {uri for h in out["hints"] for uri in h["raw_uris"]}
    assert any(ch.startswith("active_content") for ch in hashes)
    assert "cas://active" in raw_uris and "cas://nested" in raw_uris
    assert "routing hints only" in out["note"]


def test_region_hints_do_not_score_or_show_hidden_mixed_gist_text(fresh_settings):
    eng, scope = _engine_with_regions(fresh_settings)
    eng.store.add_derived(DerivedRecord(
        cid="mixed-secret-region",
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text="private espresso allergy from other agent",
        member_ids=["active", "other"],
    ))

    secret_query = eng.region_hints(
        "private espresso allergy",
        scope=scope,
        as_of=1000.0,
        use_reflex=False,
    )

    assert "mixed-secret-region" not in {h["region_id"] for h in secret_query["hints"]}
    assert "private espresso allergy" not in " ".join(
        h.get("text", "") for h in secret_query["hints"]
    )

    visible_query = eng.region_hints(
        "active ginger route",
        scope=scope,
        as_of=1000.0,
        use_reflex=False,
    )
    mixed = next(h for h in visible_query["hints"] if h["region_id"] == "mixed-secret-region")

    assert mixed["members"] == ["active"]
    assert mixed["member_count"] == 1
    assert mixed["text"] == "memory region level 1"
    assert "private espresso allergy" not in mixed["text"]


def test_engine_region_hints_report_disabled_when_regions_off(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(replace(fresh_settings, gist_channel_enabled=False))
    out = eng.region_hints("ginger tea route", scope=Scope(namespace="regions"))

    assert out["enabled"] is False
    assert out["hints"] == []
    assert "GIST_CHANNEL=0" in out["note"]


def test_mcp_region_hints_tool_is_model_free_and_scope_safe(fresh_settings, monkeypatch):
    import eidetic.mcp_server as mcp_server

    eng, scope = _engine_with_regions(fresh_settings)
    monkeypatch.setattr(mcp_server, "_engine", eng)
    try:
        out = mcp_server.region_hints(
            "ginger tea route",
            namespace=scope.namespace,
            agent_id=scope.agent_id,
            project_id=scope.project_id,
        )
        assert {"active", "nested"} <= _all_members(out)
        assert "other" not in _all_members(out)
    finally:
        monkeypatch.setattr(mcp_server, "_engine", None)


def test_api_region_hints_route_is_model_free_and_scope_safe(fresh_settings, monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import eidetic.api as api_mod

    eng, scope = _engine_with_regions(fresh_settings)
    monkeypatch.setattr(api_mod, "_engine", eng)
    try:
        r = TestClient(api_mod.app).post("/api/region_hints", json={
            "query": "ginger tea route",
            "namespace": scope.namespace,
            "agent_id": scope.agent_id,
            "project_id": scope.project_id,
            "as_of": 1000.0,
        })
        assert r.status_code == 200
        out = r.json()
        assert {"active", "nested"} <= _all_members(out)
        assert "other" not in _all_members(out)
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
