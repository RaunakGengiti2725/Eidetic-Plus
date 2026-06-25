from __future__ import annotations

from eidetic.conflicts import is_current_value_query, resolve_current_value_question
from eidetic.config import get_settings
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, RetrievalCandidate
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


def _cand(memory_id: str, text: str, valid_at: float) -> RetrievalCandidate:
    rec = MemoryRecord(memory_id=memory_id, content_hash=memory_id, text=text, valid_at=valid_at)
    return RetrievalCandidate(record=rec, fused_score=1.0)


def test_current_value_resolver_uses_timestamp_argmax():
    candidates = [
        _cand("old", "Alice works at Acme.", 10.0),
        _cand("new", "Alice works at Globex.", 20.0),
    ]

    def extractor(_query: str, payload: list[dict]) -> list[dict]:
        assert {p["memory_id"] for p in payload} == {"old", "new"}
        return [
            {"memory_id": "old", "timestamp": 10.0, "answer": "Alice works at Acme."},
            {"memory_id": "new", "timestamp": 20.0, "answer": "Alice works at Globex."},
        ]

    res = resolve_current_value_question("Where does Alice work now?", candidates, extractor)
    assert res is not None
    assert res.answer == "Alice works at Globex."
    assert [r.memory_id for r in res.records] == ["new"]


def test_current_value_resolver_ignores_model_timestamp_for_argmax():
    candidates = [
        _cand("old", "Alice works at Acme.", 10.0),
        _cand("new", "Alice works at Globex.", 20.0),
    ]

    def extractor(_query: str, _payload: list[dict]) -> list[dict]:
        return [
            {"memory_id": "old", "timestamp": 9999.0, "answer": "Alice works at Acme."},
            {"memory_id": "new", "timestamp": 20.0, "answer": "Alice works at Globex."},
        ]

    res = resolve_current_value_question("Where does Alice work now?", candidates, extractor)
    assert res is not None
    assert res.answer == "Alice works at Globex."


def test_current_value_router_rejects_historical_queries():
    assert is_current_value_query("Where does Alice work now?")
    assert not is_current_value_query("Where did Alice work previously?")
    assert not is_current_value_query("What books has Alice read?")
    assert not is_current_value_query("Has Alice worked at Acme?")


def test_retriever_conflict_resolver_flag_returns_direct_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("CONFLICT_RESOLVER", "1")
    get_settings.cache_clear()
    settings = get_settings()

    class FakeClient:
        def extract_current_value_matches(self, _query: str, _candidates: list[dict]) -> list[dict]:
            return [
                {"memory_id": "old", "timestamp": 10.0, "answer": "Alice works at Acme."},
                {"memory_id": "new", "timestamp": 20.0, "answer": "Alice works at Globex."},
            ]

    store = RecordStore(tmp_path / "db.sqlite")
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), FakeClient(), settings)
    ans = retriever.answer(
        "Where does Alice work now?",
        precomputed=[
            _cand("old", "Alice works at Acme.", 10.0),
            _cand("new", "Alice works at Globex.", 20.0),
        ],
        verify=False,
    )
    assert ans.answer == "Alice works at Globex."
    assert ans.generated_by == "conflict-resolver"
    # the note now surfaces the supersession chain (the older Acme value, closed not deleted).
    assert ans.note.startswith("conflict-resolver")
    assert "superseded 1 older value(s)" in ans.note
    get_settings.cache_clear()


def test_conflict_resolver_reaches_fixed_reader_context(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("CONFLICT_RESOLVER", "1")
    get_settings.cache_clear()
    settings = get_settings()

    class FakeClient:
        def extract_current_value_matches(self, _query: str, _candidates: list[dict]) -> list[dict]:
            return [
                {"memory_id": "old", "timestamp": 10.0, "answer": "Alice works at Acme."},
                {"memory_id": "new", "timestamp": 20.0, "answer": "Alice works at Globex."},
            ]

    store = RecordStore(tmp_path / "db.sqlite")
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), FakeClient(), settings)
    blocks = retriever.assemble_context(
        "Where does Alice work now?",
        [
            _cand("old", "Alice works at Acme.", 10.0),
            _cand("new", "Alice works at Globex.", 20.0),
        ],
    )
    joined = "\n".join(blocks)
    assert "Current-value resolver selected latest matching evidence" in joined
    assert "Alice works at Globex." in joined
    get_settings.cache_clear()


def test_deterministic_graph_conflicts_survive_out_of_order_backfill(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    graph = KnowledgeGraph(store, deterministic_conflicts=True)

    graph.add_fact("Alice", "works_at", "Globex", valid_at=20.0)
    graph.add_fact("Alice", "works_at", "Acme", valid_at=10.0)

    edges = store.all_edges()
    old = next(e for e in edges if e.dst == "Acme")
    new = next(e for e in edges if e.dst == "Globex")
    assert old.is_active_at(15.0)
    assert not old.is_active_at(25.0)
    assert new.is_active_at(25.0)
