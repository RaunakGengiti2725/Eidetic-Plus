"""Mem0 adapter per-session resilience (forgetting-machine fair-comparison fix).

A content-specific 4xx (moderation / oversized / bad request) on ONE session must skip that session
(like the eidetic adapter skips a moderated extraction window) instead of aborting the whole sample,
so mem0 still produces a comparable row. A 5xx / network error still fails loud. Fully offline."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from bench.adapters.mem0_adapter import (Mem0System, _bound_openai_compatible_clients,
                                         _call_with_hard_deadline,
                                         _is_skippable_add_error)


def test_requirements_include_mem0_strict_health_capabilities():
    req = (Path(__file__).resolve().parents[1] / "requirements-bench.txt").read_text()
    assert "mem0ai==2.0.7" in req
    assert "spacy==3.8.13" in req
    assert "fastembed==0.8.0" in req


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


class _FakeClient:
    def __init__(self):
        self.options = None

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


class _ClientOwner:
    def __init__(self):
        self.client = _FakeClient()


def test_bound_openai_compatible_clients_applies_timeout_and_retries():
    class _Memory:
        llm = _ClientOwner()
        embedding_model = _ClientOwner()

    mem = _Memory()
    _bound_openai_compatible_clients(mem, timeout_s=12.5, max_retries=2)

    assert mem.llm.client.options == {"timeout": 12.5, "max_retries": 2}
    assert mem.embedding_model.client.options == {"timeout": 12.5, "max_retries": 2}


def test_try_calls_wall_clock_timeout_interrupts_blocking_mem0_call():
    sys = Mem0System.__new__(Mem0System)
    sys._call_timeout_s = 0.05

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="wall-clock timeout"):
        sys._try_calls((lambda: time.sleep(1.0),), op="add")
    assert time.monotonic() - t0 < 0.5


def test_hard_deadline_returns_successful_vendor_call():
    assert _call_with_hard_deadline(lambda: "ok", 0.1, "search") == "ok"


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
