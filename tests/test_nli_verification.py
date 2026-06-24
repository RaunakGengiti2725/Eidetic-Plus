"""Required test: NLI verification rejects an answer the immutable record does not support.

This is the anti-confabulation gate. It makes a REAL qwen-plus NLI call, so it skips
automatically when no DASHSCOPE_API_KEY is present (we never mock a model output)."""
from __future__ import annotations

import pytest

from eidetic.config import get_settings
from eidetic.models import NLILabel


def _need_key():
    if not get_settings().has_api_key:
        pytest.skip("No DASHSCOPE_API_KEY: NLI verification needs a real model call.")


def test_unsupported_hypothesis_is_not_entailed(engine):
    _need_key()
    premise = "The quarterly revenue for the Helios project was 4.2 million dollars."
    # A claim the premise does NOT support.
    hypothesis = "The Helios project lost 9 million dollars and was cancelled."
    label, conf = engine.retriever.verify(premise, hypothesis)
    assert label != NLILabel.ENTAILMENT  # must be neutral or contradiction


def test_supported_hypothesis_is_entailed(engine):
    _need_key()
    premise = "The quarterly revenue for the Helios project was 4.2 million dollars."
    hypothesis = "Helios made 4.2 million dollars in quarterly revenue."
    label, conf = engine.retriever.verify(premise, hypothesis)
    assert label == NLILabel.ENTAILMENT


def test_end_to_end_answer_is_verified_and_cited(engine):
    _need_key()
    engine.ingest_text(
        "The Helios project quarterly revenue was 4.2 million dollars in Q2 2026.",
        source="finance-memo", extract_graph=False,
    )
    ans = engine.ask("What was the Helios project quarterly revenue?")
    assert ans.retrieved_count >= 1
    assert ans.verified is True
    assert any(c.nli_label == NLILabel.ENTAILMENT for c in ans.citations)
    # Every cited source carries immutable provenance.
    for c in ans.citations:
        assert c.content_hash and c.raw_uri
