"""Offline tests for the eidetic-plus-full adapter (Track 5.2): the PRODUCT row. Unlike the
neutral eidetic-plus row (retrieval-context only), -full applies the product policy --
NLI verification + abstention + proof -- and reports verified/abstained/confidence so the
report can score the honesty differentiators no baseline has. Offline via a fake reader+NLI."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from bench.adapters.eidetic_adapter import EideticFullSystem
from eidetic.config import get_settings
from eidetic.engine import Engine


class _FakeReader:
    def __init__(self, dim):
        self.dim = dim
        self.reader_models = []

    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._e(t)

    def embed_texts(self, ts):
        return np.stack([self._e(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)

    def extract_edges(self, text):
        return []

    def chat(self, model, system, user, **kw):
        # the ONE fixed reader path (answer_with_fixed_reader) calls client.chat(READER_MODEL, ...).
        self.reader_models.append(model)
        return "Alice works at Acme Corporation"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if "acme" in (premise or "").lower() else ("neutral", 0.2)


def _engine(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    get_settings.cache_clear()
    s = replace(get_settings(), rerank_enabled=False, **kw)
    e = Engine(s, client=_FakeReader(s.embed_dim))
    # The fixed reader (answer_with_fixed_reader) uses the MODULE-level get_client; point it at the
    # same fake the engine uses so the offline test exercises the real parity path with no key.
    from bench import reader as bench_reader
    monkeypatch.setattr(bench_reader, "get_client", lambda: e.client)
    return e


def test_eidetic_full_applies_verification_and_reports(tmp_path, monkeypatch):
    e = _engine(tmp_path, monkeypatch)
    sys = EideticFullSystem(engine=e)
    assert sys.name == "eidetic-plus-full"
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [{"role": "user", "content": "Alice works at Acme Corporation"}])
    sys.consolidate("ns")
    ar = sys.answer("ns", "where does Alice work")
    assert "Acme" in ar.answer
    assert ar.extra["verified"] is True
    assert ar.abstained is False
    assert ar.context_tokens > 0
    # NEUTRALITY: the product row must answer through the SAME fixed reader as every baseline
    # (answer_with_fixed_reader -> client.chat(READER_MODEL, ...)), not the product's own qwen3-max
    # reader -- otherwise an accuracy edge is confounded by answerer strength.
    from bench import reader as bench_reader
    assert e.client.reader_models == [bench_reader.READER_MODEL]
    get_settings.cache_clear()


def test_eidetic_full_abstains_on_no_evidence(tmp_path, monkeypatch):
    e = _engine(tmp_path, monkeypatch, abstention_threshold=0.4)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [{"role": "user", "content": "completely unrelated content about gardening"}])
    sys.consolidate("ns")
    ar = sys.answer("ns", "what is the secret launch code")
    assert ar.abstained is True                     # product honesty policy: no evidence -> abstain
    assert ar.extra["verified"] is False
    get_settings.cache_clear()


def test_eidetic_full_wired_into_make_system():
    from bench.run import make_system
    assert make_system("eidetic-full").name == "eidetic-plus-full"
    assert make_system("eidetic-plus-full").name == "eidetic-plus-full"
