"""Offline tests for the MCP server (no key). All run through the REAL engine via an injected
deterministic fake client, consistent with how the rest of the offline suite stubs model calls."""
from __future__ import annotations

import asyncio
import hashlib
import re

import numpy as np
import pytest

import eidetic.mcp_server as mcp_server


# ---- a deterministic, no-network client (test fake, not production fabrication) -------------
class FakeClient:
    def __init__(self, dim: int):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            b = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            v[b] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, text):
        return self._embed(text)

    def embed_texts(self, texts):
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def score_importance(self, text):
        return 0.5

    def extract_edges(self, text):
        return []

    def find_contradictions(self, new_fact, candidates):
        return []

    def extract_current_value_matches(self, query, candidates):
        return []

    def generate_probes(self, memory_text, n=3):
        return []

    def rerank(self, query, documents, top_n):
        return [(i, 1.0 - 0.001 * i) for i in range(min(top_n, len(documents)))]

    def generate_answer(self, question, context_blocks, model=None):
        return context_blocks[0][:300] if context_blocks else "I do not have that in memory."

    def nli(self, premise, hypothesis):
        pt = set(re.findall(r"[a-z0-9]+", premise.lower()))
        ht = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))
        overlap = len(pt & ht) / (len(ht) or 1)
        return ("entailment", 0.9) if overlap >= 0.5 else ("neutral", 0.4)


@pytest.fixture()
def mcp_engine(tmp_path, monkeypatch):
    """A real Engine on the numpy backend with the fake client, injected into the MCP server."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    from eidetic.config import get_settings
    from eidetic.engine import Engine

    get_settings.cache_clear()
    settings = get_settings()
    eng = Engine(settings, client=FakeClient(settings.embed_dim))
    monkeypatch.setattr(mcp_server, "_engine", eng)
    yield eng
    monkeypatch.setattr(mcp_server, "_engine", None)
    get_settings.cache_clear()


# ---- server + schema (no key needed) -------------------------------------------------------
def test_server_lists_all_expected_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"remember", "recall", "consolidate", "list_memories", "get_raw",
            "forget", "reawaken", "stats"} <= names


def test_tool_schemas_mark_required_fields():
    tools = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert "content" in tools["remember"].inputSchema.get("required", [])
    assert "query" in tools["recall"].inputSchema.get("required", [])
    assert "memory_id" in tools["get_raw"].inputSchema.get("required", [])


# ---- round-trip through the real engine ----------------------------------------------------
def test_remember_then_recall_round_trip(mcp_engine):
    out = mcp_server.remember("Alice works at Acme Corporation", namespace="proj")
    assert out["ok"] and out["memory_id"]
    assert out["pending_consolidation"] is True
    assert out["auto_sleep"]["enabled"] is False
    ans = mcp_server.recall("where does Alice work", namespace="proj")
    assert "acme" in ans["answer"].lower()
    assert ans["citations"], "recall must return cited sources"


# ---- scope isolation (the no-leak guarantee) -----------------------------------------------
def test_scope_isolation_no_cross_namespace_leak(mcp_engine):
    mcp_server.remember("The launch code is alpha-seven", namespace="A")
    # a recall in a different namespace must not surface namespace A's memory
    ans_b = mcp_server.recall("what is the launch code", namespace="B")
    assert "alpha-seven" not in ans_b["answer"].lower()
    assert mcp_server.list_memories(namespace="B")["total"] == 0
    assert mcp_server.list_memories(namespace="A")["total"] == 1


def test_omitted_namespace_uses_env_default(mcp_engine, monkeypatch):
    monkeypatch.setenv("EIDETIC_NAMESPACE", "envproj")
    out = mcp_server.remember("The env scoped memory is violet")
    assert out["scope"]["namespace"] == "envproj"
    assert mcp_server.list_memories()["total"] == 1
    assert mcp_server.list_memories(namespace="default")["total"] == 0
    assert mcp_server.list_memories(namespace="envproj")["total"] == 1

    explicit = mcp_server.remember("The explicit scoped memory is amber", namespace="explicit")
    assert explicit["scope"]["namespace"] == "explicit"
    assert mcp_server.list_memories(namespace="explicit")["total"] == 1


# ---- get_raw byte-identical + scope-filtered -----------------------------------------------
def test_get_raw_is_byte_identical_and_scope_filtered(mcp_engine):
    content = "Carol lives in Paris since 2019"
    out = mcp_server.remember(content, namespace="A")
    mid = out["memory_id"]
    raw = mcp_server.get_raw(mid, namespace="A")
    assert raw["raw_encoding"] == "utf-8" and raw["raw"] == content   # verbatim, not paraphrased
    assert raw["verified_against_self"] is True
    # the same id is invisible from another namespace (no cross-scope read)
    with pytest.raises(RuntimeError, match="No such memory in scope"):
        mcp_server.get_raw(mid, namespace="B")


def test_get_raw_is_bounded_with_truncation_metadata(mcp_engine):
    content = "x" * 120
    mid = mcp_server.remember(content, namespace="A")["memory_id"]
    raw = mcp_server.get_raw(mid, namespace="A", max_bytes=25, offset=10)
    assert raw["raw_encoding"] == "utf-8"
    assert raw["raw"] == "x" * 25
    assert raw["raw_total_bytes"] == 120
    assert raw["raw_offset"] == 10
    assert raw["raw_returned_bytes"] == 25
    assert raw["raw_truncated"] is True


def test_mcp_list_memories_bounds_pagination(mcp_engine):
    for i in range(3):
        mcp_server.remember(f"bounded list memory {i}", namespace="A")
    page = mcp_server.list_memories(namespace="A", limit=10_000, offset=-10)
    assert page["limit"] == 500
    assert page["offset"] == 0
    assert page["total"] == 3
    assert len(page["memories"]) == 3


def test_mcp_rejects_empty_text_arguments(mcp_engine):
    with pytest.raises(RuntimeError, match="content must not be empty"):
        mcp_server.remember("   ", namespace="A")
    with pytest.raises(RuntimeError, match="query must not be empty"):
        mcp_server.recall("", namespace="A")


# ---- forget lowers priority without deleting the raw record --------------------------------
def test_forget_decays_priority_without_deleting_raw(mcp_engine):
    out = mcp_server.remember("Dave enjoys hiking on weekends", namespace="A")
    mid = out["memory_id"]
    before = mcp_engine.get_record(mid).fsrs
    lapses0, diff0 = before.lapses, before.difficulty
    res = mcp_server.forget(mid, namespace="A")
    assert res["ok"]
    after = mcp_engine.get_record(mid).fsrs
    # forget = an FSRS lapse: it records a lapse and raises difficulty (faster future decay),
    # which lowers index priority over time. It never deletes the raw record.
    assert after.lapses == lapses0 + 1 and after.difficulty > diff0
    assert mcp_server.get_raw(mid, namespace="A")["raw"] == "Dave enjoys hiking on weekends"


# ---- read-only tools + consolidate work without a key --------------------------------------
def test_readonly_and_consolidate_need_no_key(mcp_engine):
    assert mcp_server.stats(namespace="empty")["memories"] == 0
    assert mcp_server.list_memories(namespace="empty")["memories"] == []
    assert isinstance(mcp_server.consolidate(namespace="empty"), dict)   # token-free dream pass


# ---- Connected Brain Loop parity tools (Phase 6) -------------------------------------------
def test_brain_parity_tools_are_listed():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"sleep", "memory_autopsy", "recall_trace", "prove_age_independence"} <= names


def test_mcp_sleep_is_offline_on_quiet_scope(mcp_engine):
    out = mcp_server.sleep(namespace="quiet")
    assert out["consolidate_pending"]["pending_processed"] == 0 and "dream" in out


def test_mcp_memory_autopsy_diagnoses_missing_write(mcp_engine):
    out = mcp_server.memory_autopsy("what is the zebra protocol", namespace="empty")
    assert out["diagnosis"] == "missing_write"


def test_mcp_recall_trace_empty_without_flag(mcp_engine):
    assert mcp_server.recall_trace() == {}        # RECALL_TRACE off -> no trace surfaced


def test_mcp_consolidate_runs_unified_sleep(mcp_engine):
    out = mcp_server.consolidate(namespace="empty")
    # consolidate is now an alias of the unified sleep: pending writes flow before the dream pass.
    assert "consolidate_pending" in out and "dream" in out


def test_mcp_scratchpad_and_why_remembered(mcp_engine):
    from eidetic.models import MemoryRecord, Scope, now
    scope = Scope(namespace="sp")
    mcp_engine.store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="h1", text="the trophy day", scope=scope, valid_at=now(),
        salience=0.95, metadata={"arousal": 0.9}))
    sp = mcp_server.scratchpad(namespace="sp")
    assert any(e["memory_id"] == "m1" for e in sp["scratchpad"])
    why = mcp_server.why_remembered("m1", namespace="sp")
    assert why["salience"] == 0.95 and why["components"]["arousal"] == 0.9
    assert "not a diagnosis" in why["why"]
    assert why["provenance"]["source_preview"] == "the trophy day"


# ---- entry point ---------------------------------------------------------------------------
def test_main_entry_point_runs_stdio(monkeypatch):
    called = {}
    monkeypatch.setattr(mcp_server.mcp, "run", lambda transport: called.setdefault("t", transport))
    monkeypatch.setattr("sys.argv", ["eidetic-plus"])
    mcp_server.main()
    assert called["t"] == "stdio"


def test_main_entry_point_runs_http_alias(monkeypatch):
    called = {}
    old_port = mcp_server.mcp.settings.port
    monkeypatch.setattr(mcp_server.mcp.settings, "port", old_port)
    monkeypatch.setattr(mcp_server.mcp, "run", lambda transport: called.setdefault("t", transport))
    monkeypatch.setattr("sys.argv", ["eidetic-plus", "--http", "--http-port", "9876"])
    mcp_server.main()
    assert called["t"] == "streamable-http"
    assert mcp_server.mcp.settings.port == 9876


def test_main_entry_point_runs_http_transport(monkeypatch):
    called = {}
    monkeypatch.setattr(mcp_server.mcp, "run", lambda transport: called.setdefault("t", transport))
    monkeypatch.setattr("sys.argv", ["eidetic-plus", "--transport", "http"])
    mcp_server.main()
    assert called["t"] == "streamable-http"


# ---- bitemporal write/read parity over MCP --------------------------------------------------
def test_remember_backdates_valid_at_and_source(mcp_engine):
    """A backfilled fact must carry its EVENT time, not ingest time: bitemporal time travel
    (value_as_of / fact_history / truth_ledger windows) is wrong otherwise."""
    event_t = 1_600_000_000.0  # 2020-09-13, long before any test wall clock
    out = mcp_server.remember(
        "Historical note: the archive migration finished successfully.",
        namespace="hist", valid_at=event_t, source="import",
    )
    assert out["ok"]
    rec = mcp_engine.store.get_record(out["memory_id"])
    assert rec.valid_at == event_t
    assert rec.source == "import"
    assert out["valid_at"] == event_t


def test_recall_accepts_as_of_time_travel(mcp_engine):
    """as_of on the flagship recall tool: a memory valid only AFTER as_of must not answer."""
    mcp_server.remember("The rotation lead is Priya.", namespace="tt",
                        valid_at=1_700_000_000.0)
    # ask BEFORE the fact became valid -> no verified answer from it
    before = mcp_server.recall("who is the rotation lead", namespace="tt",
                               as_of=1_600_000_000.0)
    assert "priya" not in (before["answer"] or "").lower()
    after = mcp_server.recall("who is the rotation lead", namespace="tt",
                              as_of=1_800_000_000.0)
    assert "priya" in (after["answer"] or "").lower()


def test_get_raw_paged_reads_stay_utf8_across_multibyte_boundaries(mcp_engine):
    """A byte slice that splits a multibyte UTF-8 character must trim to character
    boundaries and stay readable text (with adjusted offsets), not flip the whole page
    to base64."""
    text = ("météo " * 40) + ("日本語テキスト" * 30) + (" fin")
    out = mcp_server.remember(text, namespace="raw8")
    mid = out["memory_id"]
    # pick an offset that lands inside a multibyte sequence
    raw = text.encode("utf-8")
    offset = 0
    for i in range(10, len(raw)):
        if (raw[i] & 0b1100_0000) == 0b1000_0000:  # continuation byte
            offset = i
            break
    assert offset > 0
    page = mcp_server.get_raw(mid, namespace="raw8", offset=offset, max_bytes=101)
    assert page["raw_encoding"] == "utf-8"
    # returned text must reassemble against the source at the ADJUSTED offset
    start = page["raw_offset"]
    assert raw[start:start + page["raw_returned_bytes"]].decode("utf-8") == page["raw"]


def test_remember_file_round_trips_bytes_with_provenance(mcp_engine):
    """Write-path parity with get_raw: an MCP host can persist a non-chat artifact and read
    back the identical bytes with provenance."""
    import base64 as b64
    payload = "# Design notes\nThe cache keys rotate per namespace version.\n".encode("utf-8")
    out = mcp_server.remember_file(
        content_base64=b64.b64encode(payload).decode("ascii"),
        filename="design-notes.md",
        namespace="files",
        valid_at=1_650_000_000.0,
    )
    assert out["ok"] and out["memory_id"]
    rec = mcp_engine.store.get_record(out["memory_id"])
    assert rec.valid_at == 1_650_000_000.0
    raw = mcp_server.get_raw(out["memory_id"], namespace="files")
    assert raw["raw_encoding"] == "utf-8"
    assert "cache keys rotate" in raw["raw"]
    assert raw["verified_against_self"] is True


def test_remember_file_rejects_oversize_and_bad_base64(mcp_engine):
    import base64 as b64
    with pytest.raises(RuntimeError):
        mcp_server.remember_file(content_base64="!!!not-base64!!!", filename="x.txt",
                                 namespace="files")
    big = b64.b64encode(b"x" * (mcp_server._MAX_RAW_BYTES + 1)).decode("ascii")
    with pytest.raises(RuntimeError):
        mcp_server.remember_file(content_base64=big, filename="big.txt", namespace="files")
