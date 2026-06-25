"""Offline tests for the F1 DashScope rate governor (no network)."""
from __future__ import annotations

import time

import pytest

from eidetic.dashscope_client import (ModelCallError, RateGovernor, _is_rate_limit,
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


def test_governor_fails_loud_on_non_rate_error_no_retry():
    g = _fast(max_retries=5)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise ModelCallError("DashScope call failed (HTTP 500): internal")

    with pytest.raises(ModelCallError, match="500"):
        g.run(fn)
    assert n["c"] == 1                     # not retried


def test_governor_gives_up_after_max_retries_never_fabricates():
    g = _fast(max_retries=3)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise ModelCallError("HTTP 429 Throttling")

    with pytest.raises(ModelCallError):
        g.run(fn)
    assert n["c"] == 4                     # initial + 3 retries; never returns a fake result


def test_token_bucket_rate_limits_after_initial_burst():
    g = RateGovernor(rpm=60, max_concurrency=8)   # 1 token/sec, capacity 60
    t0 = time.monotonic()
    for _ in range(60):
        g._take_token()                    # drain the full initial bucket -> instant
    assert time.monotonic() - t0 < 0.5
    t1 = time.monotonic()
    g._take_token()                        # the next token must wait for a ~1s refill
    assert time.monotonic() - t1 >= 0.5
