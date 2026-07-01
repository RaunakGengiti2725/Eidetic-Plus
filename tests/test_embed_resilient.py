"""Embedding input-too-long resilience (robustness audit finding 1).

text-embedding-v4 rejects an over-length input with a deterministic 'input too long' 400. _embed_raw
re-embeds that batch once with inputs truncated to embed_truncate_chars (guaranteed under the limit),
so an oversized memory/session does not abort an otherwise-fine run. Normal inputs never hit this.
The full raw text always stays in the substrate; only the over-length input's vector is from a prefix.
Fully offline (fake SDK)."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from eidetic.dashscope_client import (DashScopeClient, ModelCallError,
                                      _is_input_too_long)


def test_is_input_too_long_classifies_only_length_4xx():
    assert _is_input_too_long(
        "DashScope call failed (HTTP 400): Range of input length should be [1, 33000]") is True
    assert _is_input_too_long("DashScope call failed (HTTP 400): input too long") is True
    # other 400s and any 5xx are NOT this error
    assert _is_input_too_long("DashScope call failed (HTTP 400): inappropriate content") is False
    assert _is_input_too_long("DashScope call failed (HTTP 500): input length internal") is False


class _Resp:
    def __init__(self, status_code, output=None, message=""):
        self.status_code = status_code
        self.output = output
        self.message = message


class _FakeTE:
    """Rejects any input longer than 100 chars with a length-400; embeds the rest."""
    def __init__(self):
        self.calls = []

    def call(self, model, input, dimension, **_kwargs):
        self.calls.append(list(input))
        if any(len(t) > 100 for t in input):
            return _Resp(400, message="InvalidParameter: Range of input length should be [1, 33000]")
        return _Resp(200, output={"embeddings": [
            {"text_index": j, "embedding": [0.1] * dimension} for j, _ in enumerate(input)]})


class _FakeBadRequestTE(_FakeTE):
    """A non-length 400 (must NOT be truncate-retried)."""
    def call(self, model, input, dimension, **_kwargs):
        self.calls.append(list(input))
        return _Resp(400, message="InvalidParameter: some other bad request")


def _client(fresh_settings, te):
    c = DashScopeClient.__new__(DashScopeClient)
    c.settings = replace(fresh_settings, embed_truncate_chars=50)
    c._governor = None
    c._embed_cache = None

    class _DS:
        TextEmbedding = te
    c._ds = _DS()
    return c


def test_embed_raw_truncates_and_retries_on_length_400(fresh_settings):
    te = _FakeTE()
    c = _client(fresh_settings, te)
    vecs = c._embed_raw(["short text", "x" * 500])   # second input is over-length
    assert vecs.shape == (2, fresh_settings.embed_dim)
    assert len(te.calls) == 2                          # first batch 400'd, retried truncated
    assert all(len(t) <= 50 for t in te.calls[1])      # retry inputs were truncated to the cap


def test_embed_raw_propagates_non_length_400(fresh_settings):
    c = _client(fresh_settings, _FakeBadRequestTE())
    with pytest.raises(ModelCallError):
        c._embed_raw(["x" * 500])                      # a non-length 400 still fails loud


def test_embed_raw_parallel_batches_preserve_input_order(fresh_settings):
    c = _client(fresh_settings, _FakeTE())
    c.settings = replace(c.settings, embed_batch_parallelism=3)
    seen_batches = []

    def fake_batch(batch):
        seen_batches.append(list(batch))
        # Complete later batches first; _embed_raw must still return vectors in input order.
        delay = 0.03 if batch[0] == "0" else 0.005
        time.sleep(delay)
        return [[float(text)] * fresh_settings.embed_dim for text in batch]

    c._embed_batch_call = fake_batch
    vecs = c._embed_raw([str(i) for i in range(25)])
    assert vecs.shape == (25, fresh_settings.embed_dim)
    assert np.array_equal(vecs[:, 0], np.arange(25, dtype=np.float32))
    assert sorted(len(batch) for batch in seen_batches) == [5, 10, 10]
