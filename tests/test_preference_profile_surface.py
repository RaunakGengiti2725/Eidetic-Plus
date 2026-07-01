"""Host-facing preference profile surface.

Preference memory is useful only if agents can read the current profile cheaply and verify every
line against source bytes. These tests also guard sub-scope filtering because profile rows are stored
namespace-wide while source memories carry the full Scope.
"""
from __future__ import annotations

import pytest

from eidetic.models import MemoryRecord, Scope
from eidetic.preferences import preference_dedup_key


def _seed_profile(eng):
    scope_a = Scope(namespace="prefs", agent_id="agent-a", project_id="project")
    scope_b = Scope(namespace="prefs", agent_id="agent-b", project_id="project")
    records = [
        MemoryRecord(memory_id="a_old", text="user: My favorite music is jazz.",
                     scope=scope_a, valid_at=100.0, content_hash="h_old",
                     raw_uri="cas://h_old"),
        MemoryRecord(memory_id="a_new", text="user: My favorite music is techno.",
                     scope=scope_a, valid_at=200.0, content_hash="h_new",
                     raw_uri="cas://h_new"),
        MemoryRecord(memory_id="b_pref", text="user: I love espresso.",
                     scope=scope_b, valid_at=150.0, content_hash="h_b",
                     raw_uri="cas://h_b"),
    ]
    for rec in records:
        eng.store.upsert_record(rec)
    eng.store.add_profile_line(
        "prefs",
        "User's favorite music is jazz.",
        source_memory_id="a_old",
        content_hash="h_old",
        raw_uri="cas://h_old",
        valid_at=100.0,
        dedup_key="favorite:music:jazz",
    )
    eng.store.add_profile_line(
        "prefs",
        "User's favorite music is techno.",
        source_memory_id="a_new",
        content_hash="h_new",
        raw_uri="cas://h_new",
        valid_at=200.0,
        dedup_key="favorite:music:techno",
    )
    eng.store.add_profile_line(
        "prefs",
        "User likes espresso.",
        source_memory_id="b_pref",
        content_hash="h_b",
        raw_uri="cas://h_b",
        valid_at=150.0,
        dedup_key="likes:espresso",
    )
    return scope_a, scope_b


def test_engine_preference_profile_is_current_scoped_and_proof_linked(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope_a, _scope_b = _seed_profile(eng)

    out = eng.preference_profile(scope=scope_a)
    lines = [entry["line"] for entry in out["profile"]]

    assert "User's favorite music is techno." in lines
    assert "User's favorite music is jazz." not in lines
    assert "User likes espresso." not in lines
    assert out["provenance_complete"] is True
    entry = next(e for e in out["profile"] if "techno" in e["line"])
    assert entry["source_memory_id"] == "a_new"
    assert entry["content_hash"] == "h_new"
    assert entry["raw_uri"] == "cas://h_new"
    assert entry["source_scope"] == scope_a.model_dump()


def test_engine_preference_profile_can_show_superseded_history(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope_a, _scope_b = _seed_profile(eng)

    out = eng.preference_profile(scope=scope_a, include_inactive=True)
    by_line = {entry["line"]: entry for entry in out["profile"]}

    assert by_line["User's favorite music is jazz."]["status"] == "inactive"
    assert by_line["User's favorite music is techno."]["status"] == "active"
    assert "User likes espresso." not in by_line


def test_preference_supersession_does_not_cross_agent_subscope(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope_a = Scope(namespace="prefs-cross", agent_id="agent-a", project_id="project")
    scope_b = Scope(namespace="prefs-cross", agent_id="agent-b", project_id="project")
    records = [
        MemoryRecord(memory_id="a_jazz", text="user: My favorite music is jazz.",
                     scope=scope_a, valid_at=100.0, content_hash="h_a_jazz",
                     raw_uri="cas://h_a_jazz"),
        MemoryRecord(memory_id="b_jazz", text="user: My favorite music is jazz.",
                     scope=scope_b, valid_at=110.0, content_hash="h_b_jazz",
                     raw_uri="cas://h_b_jazz"),
        MemoryRecord(memory_id="a_techno", text="user: My favorite music is techno.",
                     scope=scope_a, valid_at=200.0, content_hash="h_a_techno",
                     raw_uri="cas://h_a_techno"),
    ]
    for rec in records:
        eng.store.upsert_record(rec)
        profile_line = (
            "User's favorite music is techno."
            if "techno" in rec.text else "User's favorite music is jazz."
        )
        eng.store.add_profile_line(
            rec.scope.namespace,
            profile_line,
            source_memory_id=rec.memory_id,
            content_hash=rec.content_hash,
            raw_uri=rec.raw_uri,
            valid_at=rec.valid_at,
            dedup_key=preference_dedup_key(profile_line),
        )

    a_profile = eng.preference_profile(scope=scope_a, include_inactive=True)["profile"]
    b_profile = eng.preference_profile(scope=scope_b, include_inactive=True)["profile"]
    a_by_source = {entry["source_memory_id"]: entry for entry in a_profile}
    b_by_source = {entry["source_memory_id"]: entry for entry in b_profile}

    assert a_by_source["a_jazz"]["status"] == "inactive"
    assert a_by_source["a_jazz"]["invalid_at"] == 200.0
    assert a_by_source["a_techno"]["status"] == "active"
    assert b_by_source["b_jazz"]["status"] == "active"
    assert b_by_source["b_jazz"]["invalid_at"] is None


def test_engine_preference_profile_hides_legacy_unattributed_rows_from_subscope(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope_a, _scope_b = _seed_profile(eng)
    eng.store.add_profile_line("prefs", "Legacy namespace-wide preference.")

    narrowed = eng.preference_profile(scope=scope_a, include_inactive=True)
    wide = eng.preference_profile(scope=Scope(namespace="prefs"), include_inactive=True)

    assert all(entry["line"] != "Legacy namespace-wide preference."
               for entry in narrowed["profile"])
    assert any(entry["line"] == "Legacy namespace-wide preference."
               for entry in wide["profile"])
    assert narrowed["skipped_unattributed"] == 1


def test_engine_preference_profile_filters_missing_source_rows(fresh_settings):
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    ns = "prefs-missing-source"
    h = "e" * 64
    eng.store.add_profile_line(
        ns,
        "User likes phantom tea.",
        source_memory_id="missing-pref",
        content_hash=h,
        raw_uri=f"cas://{h}",
        valid_at=100.0,
    )
    eng.store.add_profile_line(ns, "Legacy namespace-wide preference.")

    out = eng.preference_profile(
        scope=Scope(namespace=ns),
        include_inactive=True,
    )
    lines = [entry["line"] for entry in out["profile"]]

    assert "User likes phantom tea." not in lines
    assert "Legacy namespace-wide preference." in lines
    assert out["skipped_missing_source"] == 1
    assert out["provenance_complete"] is False


def test_mcp_preference_profile_tool_is_scope_safe(fresh_settings, monkeypatch):
    import eidetic.mcp_server as mcp_server
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope_a, _scope_b = _seed_profile(eng)
    monkeypatch.setattr(mcp_server, "_engine", eng)
    try:
        out = mcp_server.preference_profile(
            namespace=scope_a.namespace,
            agent_id=scope_a.agent_id,
            project_id=scope_a.project_id,
        )
        lines = [entry["line"] for entry in out["profile"]]
        assert "User's favorite music is techno." in lines
        assert "User likes espresso." not in lines
    finally:
        monkeypatch.setattr(mcp_server, "_engine", None)


def test_api_preference_profile_route_is_scope_safe(fresh_settings, monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import eidetic.api as api_mod
    from eidetic.engine import Engine

    eng = Engine(fresh_settings)
    scope_a, _scope_b = _seed_profile(eng)
    monkeypatch.setattr(api_mod, "_engine", eng)
    try:
        r = TestClient(api_mod.app).get(
            "/api/preference_profile",
            params={
                "namespace": scope_a.namespace,
                "agent_id": scope_a.agent_id,
                "project_id": scope_a.project_id,
            },
        )
        assert r.status_code == 200
        lines = [entry["line"] for entry in r.json()["profile"]]
        assert "User's favorite music is techno." in lines
        assert "User likes espresso." not in lines
    finally:
        monkeypatch.setattr(api_mod, "_engine", None)
