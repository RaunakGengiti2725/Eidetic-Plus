"""Mem0 adapter per-session resilience (forgetting-machine fair-comparison fix).

A content-specific 4xx (moderation / oversized / bad request) on ONE session must skip that session
(like the eidetic adapter skips a moderated extraction window) instead of aborting the whole sample,
so mem0 still produces a comparable row. A 5xx / network error still fails loud. Fully offline."""
from __future__ import annotations

import pytest

from bench.adapters.mem0_adapter import Mem0System, _is_skippable_add_error


def test_is_skippable_add_error_classifies_4xx_not_5xx():
    assert _is_skippable_add_error(RuntimeError("Underlying error: Error code: 400 - bad request")) is True
    assert _is_skippable_add_error(RuntimeError("inappropriate content detected")) is True
    assert _is_skippable_add_error(RuntimeError("data_inspection failed")) is True
    # genuine failures must NOT be skipped
    assert _is_skippable_add_error(RuntimeError("Error code: 500 - internal server error")) is False
    assert _is_skippable_add_error(ConnectionError("connection reset")) is False


class _FakeMem:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def add(self, *_a, **_k):
        self.calls += 1
        raise self.exc


def test_ingest_session_skips_a_4xx_session():
    sys = Mem0System.__new__(Mem0System)
    sys._memory = _FakeMem(Exception("Error code: 400 - {'message': 'inappropriate content'}"))
    wr = sys.ingest_session("ns", "s0", [{"role": "user", "content": "hello world"}])
    assert wr.tokens == 0          # skipped, no raise
    assert sys._memory.calls >= 1  # it really tried


def test_ingest_session_fails_loud_on_5xx():
    sys = Mem0System.__new__(Mem0System)
    sys._memory = _FakeMem(Exception("Error code: 500 - internal server error"))
    with pytest.raises(RuntimeError):
        sys.ingest_session("ns", "s0", [{"role": "user", "content": "hello world"}])
