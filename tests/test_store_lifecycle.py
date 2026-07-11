from __future__ import annotations

import gc
import threading
import warnings

from eidetic.embed_cache import PersistentEmbedCache
from eidetic.extract_cache import PersistentExtractCache
from eidetic.feedback import FeedbackBuffer
from eidetic.models import Scope
from eidetic.store import RecordStore


def test_record_store_close_closes_connections_from_all_threads(tmp_path):
    store = RecordStore(tmp_path / "threaded-store.sqlite")
    connections = [store._conn()]

    def use_store():
        store.count(Scope(namespace="threaded-store"))
        connections.append(store._conn())

    thread = threading.Thread(target=use_store)
    thread.start()
    thread.join()

    store.close()
    store.close()

    for connection in connections:
        try:
            connection.execute("SELECT 1")
        except Exception as exc:
            assert "closed" in str(exc).lower()
        else:
            raise AssertionError("RecordStore.close left a SQLite connection open")


def test_persistent_sqlite_helpers_close_idempotently(tmp_path):
    helpers = [
        PersistentEmbedCache(tmp_path / "embed.sqlite"),
        PersistentExtractCache(tmp_path / "extract.sqlite"),
        FeedbackBuffer(tmp_path / "feedback.sqlite"),
    ]
    connections = [helper._conn() for helper in helpers]

    for helper in helpers:
        helper.close()
        helper.close()

    for connection in connections:
        try:
            connection.execute("SELECT 1")
        except Exception as exc:
            assert "closed" in str(exc).lower()
        else:
            raise AssertionError("persistent SQLite helper left a connection open")


def test_record_store_finalizer_emits_no_resource_warning(tmp_path):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ResourceWarning)
        for index in range(20):
            store = RecordStore(tmp_path / f"store-{index}.sqlite")
            store.count(Scope(namespace="lifecycle"))
            del store
        gc.collect()

    resource_warnings = [warning for warning in seen
                         if issubclass(warning.category, ResourceWarning)]
    assert resource_warnings == []


def test_persistent_sqlite_helper_finalizers_emit_no_resource_warning(tmp_path):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ResourceWarning)
        for index in range(10):
            helpers = [
                PersistentEmbedCache(tmp_path / f"embed-{index}.sqlite"),
                PersistentExtractCache(tmp_path / f"extract-{index}.sqlite"),
                FeedbackBuffer(tmp_path / f"feedback-{index}.sqlite"),
            ]
            del helpers
        gc.collect()

    resource_warnings = [warning for warning in seen
                         if issubclass(warning.category, ResourceWarning)]
    assert resource_warnings == []
