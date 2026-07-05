from __future__ import annotations

from eidetic.conflicts import is_current_value_query, resolve_current_value_question
from eidetic.config import get_settings
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, RetrievalCandidate
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


def _cand(memory_id: str, text: str, valid_at: float,
          entities: list[str] | None = None) -> RetrievalCandidate:
    rec = MemoryRecord(
        memory_id=memory_id,
        content_hash=memory_id,
        text=text,
        valid_at=valid_at,
        entities=list(entities or []),
    )
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


def test_retriever_current_value_routes_through_smqe(tmp_path, monkeypatch):
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
        verify=True,
    )
    assert ans.answer == "Globex"
    assert ans.generated_by == "smqe"
    assert ans.note == "smqe:latest_value:record"
    assert ans.verified is True
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
    assert "SMQE latest-value operator selected matching evidence" in joined
    assert "Globex" in joined
    get_settings.cache_clear()


def test_conflict_resolver_skips_hypothetical_activity_questions():
    assert is_current_value_query("Where does Alice work now?") is True
    assert is_current_value_query(
        "What is an outdoor activity that Priya would enjoy doing while keeping her parrot entertained?"
    ) is False
    assert is_current_value_query("What activity would Priya enjoy?") is False


def test_conflict_resolver_fail_open_on_extractor_error(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("CONFLICT_RESOLVER", "1")
    get_settings.cache_clear()
    settings = get_settings()

    class FakeClient:
        def extract_current_value_matches(self, _query: str, _candidates: list[dict]) -> list[dict]:
            raise ValueError("malformed model JSON")

    store = RecordStore(tmp_path / "db.sqlite")
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), FakeClient(), settings)
    candidates = [_cand("new", "Caroline is a transgender woman.", 20.0)]

    assert retriever._try_conflict_resolver("What is Caroline's identity?", candidates) is None
    blocks = retriever.assemble_context("What is Caroline's identity?", candidates)
    assert "Caroline is a transgender woman." in "\n".join(blocks)
    get_settings.cache_clear()


def test_conflict_resolver_uses_graph_closure_for_stale_source(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("CONFLICT_RESOLVER", "1")
    get_settings.cache_clear()
    settings = get_settings()

    class FakeClient:
        def extract_current_value_matches(self, _query: str, candidates: list[dict]) -> list[dict]:
            return [{"memory_id": candidates[0]["memory_id"], "answer": candidates[0]["text"]}]

    store = RecordStore(tmp_path / "db.sqlite")
    graph = KnowledgeGraph(store)
    graph.add_fact(
        "Alice", "works_at", "Acme", fact="Alice works at Acme.",
        source_memory_id="old", valid_at=10.0)
    graph.add_fact(
        "Alice", "works_at", "Globex", fact="Alice works at Globex.",
        source_memory_id="new", valid_at=20.0)
    retriever = Retriever(store, object(), graph, object(), FakeClient(), settings)
    old = _cand("old", "Alice works at Acme.", 10.0, ["Alice", "Acme"])

    current = retriever._try_conflict_resolver("Where does Alice work now?", [old], as_of=30.0)
    assert current is not None and current.abstained
    assert current.records == []

    historical = retriever._try_conflict_resolver("Where does Alice work now?", [old], as_of=15.0)
    assert historical is not None and not historical.abstained
    assert [r.memory_id for r in historical.records] == ["old"]
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
