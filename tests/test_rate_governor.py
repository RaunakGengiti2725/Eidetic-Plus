"""Offline tests for the F1 DashScope rate governor (no network)."""
from __future__ import annotations

from dataclasses import replace
import time

import pytest

from eidetic.dashscope_client import (DashScopeClient, ModelCallError, ModelCallTimeout,
                                      RateGovernor,
                                      _is_rate_limit, _is_server_error,
                                      _retry_after_seconds)


def _fast(**kw):
    return RateGovernor(rpm=6000, max_concurrency=8, backoff_base=0.001, backoff_max=0.01, **kw)


def test_is_rate_limit_distinguishes_quota_exhaustion():
    assert _is_rate_limit("DashScope call failed (HTTP 429): Throttling") is True
    assert _is_rate_limit("requests rate increased too quickly") is True
    # free-tier exhaustion is NOT retryable -- retrying can never succeed -> fail loud.
    assert _is_rate_limit("HTTP 403: The free tier of the model has been exhausted") is False
    assert _is_rate_limit("HTTP 500: internal error") is False


def test_retry_after_is_parsed():
    assert _retry_after_seconds("rate limited; Retry-After: 5") == 5.0
    assert _retry_after_seconds("no header here") is None


def test_governor_retries_rate_limit_then_succeeds():
    g = _fast(max_retries=5)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        if n["c"] < 3:
            raise ModelCallError("DashScope call failed (HTTP 429): Throttling")
        return "ok"

    assert g.run(fn) == "ok"
    assert n["c"] == 3                     # two retries, then success


def test_is_server_error_retries_5xx_not_4xx():
    # 5xx == server-side failure -> the same request can succeed on retry (retryable).
    assert _is_server_error("DashScope call failed (HTTP 500): "
                            "<5000203> InternalError.Algo.Embedding_pipeline_Error") is True
    assert _is_server_error("DashScope call failed (HTTP 502): Bad Gateway") is True
    assert _is_server_error("DashScope call failed (HTTP 503): unavailable") is True
    # 4xx == client error (bad request) -> deterministic, NEVER retried.
    assert _is_server_error("DashScope call failed (HTTP 400): "
                            "InternalError.Algo.InvalidParameter: Range of input length") is False
    assert _is_server_error("DashScope call failed (HTTP 429): Throttling") is False
    # quota exhaustion stays non-retryable even if surfaced as a 5xx.
    assert _is_server_error("(HTTP 500): the free tier has been exhausted") is False
    # REGRESSION (review finding 1): classify on the parenthesized status code, NOT a substring scan.
    # A real 400 whose BODY mentions 'http 500' must NOT be retried as a server error.
    assert _is_server_error("DashScope call failed (HTTP 400): error http 500 referenced in docs") is False
    # No parseable (HTTP <code>) token (e.g. a JSON-parse error) -> not a server error.
    assert _is_server_error("Model returned non-JSON for a JSON request") is False


def test_governor_fails_loud_on_4xx_no_retry():
    # A 4xx (bad request: input too long / invalid param) is deterministic; retrying it can never
    # succeed, so it must fail loud immediately (no wasted quota, no fabricated result).
    g = _fast(max_retries=5)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise ModelCallError("DashScope call failed (HTTP 400): "
                             "InternalError.Algo.InvalidParameter: Range of input length")

    with pytest.raises(ModelCallError, match="400"):
        g.run(fn)
    assert n["c"] == 1                     # not retried


def test_governor_retries_transient_5xx_then_succeeds():
    # The embedding service intermittently returns a 500 InternalError.Algo.Embedding_pipeline_Error
    # independent of content (proven empirically). It MUST be retried, or one hiccup aborts a whole
    # run of tens of thousands of embed calls.
    g = _fast(max_retries=5)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        if n["c"] < 3:
            raise ModelCallError("DashScope call failed (HTTP 500): "
                                 "<5000203> InternalError.Algo.Embedding_pipeline_Error")
        return "ok"

    assert g.run(fn) == "ok"
    assert n["c"] == 3                     # two retries, then success


def test_governor_retries_wall_clock_timeout_then_succeeds():
    g = _fast(max_retries=3)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        if n["c"] < 2:
            raise ModelCallTimeout("DashScope call exceeded wall-clock timeout of 0.050s")
        return "ok"

    assert g.run(fn) == "ok"
    assert n["c"] == 2


def test_governor_5xx_exhausts_retries_then_fails_loud_never_fabricates():
    g = _fast(max_retries=3)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise ModelCallError("DashScope call failed (HTTP 500): InternalError.Algo")

    with pytest.raises(ModelCallError, match="500"):
        g.run(fn)
    assert n["c"] == 4                     # initial + 3 retries, then loud (never a fake result)


def test_governor_gives_up_after_max_retries_never_fabricates():
    g = _fast(max_retries=3)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise ModelCallError("HTTP 429 Throttling")

    with pytest.raises(ModelCallError):
        g.run(fn)
    assert n["c"] == 4                     # initial + 3 retries; never returns a fake result


def test_governor_slot_acquire_timeout_fails_loud_instead_of_parking():
    g = RateGovernor(
        rpm=6000,
        max_concurrency=1,
        max_retries=0,
        backoff_base=0.0,
        backoff_max=0.0,
        slot_acquire_timeout=0.01,
    )
    assert g._sem.acquire(blocking=False) is True
    try:
        t0 = time.monotonic()
        with pytest.raises(ModelCallError, match="concurrency slot unavailable"):
            g.run(lambda: "should not run")
        assert time.monotonic() - t0 < 0.25
    finally:
        g._sem.release()


def test_generation_call_passes_request_timeout_to_dashscope_sdk(fresh_settings):
    calls = {}

    class _Resp:
        status_code = 200
        output = {"choices": [{"message": {"content": "ok"}}]}

    class _Generation:
        @staticmethod
        def call(**kwargs):
            calls.update(kwargs)
            return _Resp()

    class _DS:
        Generation = _Generation

    client = DashScopeClient.__new__(DashScopeClient)
    client.settings = replace(
        fresh_settings,
        api_key="test-key",
        dashscope_request_timeout_sec=12.5,
    )
    client._governor = None
    client._embed_cache = None
    client._ds = _DS()

    assert client.chat("qwen-plus", "system", "user") == "ok"
    assert calls["request_timeout"] == 12.5
    assert calls["timeout"] == 12.5


def test_dashscope_client_wall_clock_deadline_interrupts_blocking_sdk_call(fresh_settings):
    client = DashScopeClient.__new__(DashScopeClient)
    client.settings = replace(fresh_settings, dashscope_request_timeout_sec=0.05)
    client._governor = None

    t0 = time.monotonic()
    with pytest.raises(ModelCallTimeout):
        client._governed(lambda: time.sleep(1.0))
    assert time.monotonic() - t0 < 0.5


def test_token_bucket_rate_limits_after_initial_burst():
    g = RateGovernor(rpm=60, max_concurrency=8)   # 1 token/sec, capacity 60
    t0 = time.monotonic()
    for _ in range(60):
        g._take_token()                    # drain the full initial bucket -> instant
    assert time.monotonic() - t0 < 0.5
    t1 = time.monotonic()
    g._take_token()                        # the next token must wait for a ~1s refill
    assert time.monotonic() - t1 >= 0.5
