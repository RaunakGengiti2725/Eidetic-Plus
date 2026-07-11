from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.integrations.notebooklm import NotebookLMBridge
from eidetic.models import ABSTENTION_TEXT, AnswerStatus, Scope


class _Client:
    def __init__(self, dim: int):
        self.dim = dim
        self.nli_calls = 0

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
            vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(text) for text in texts])

    def extract_edges(self, text: str) -> list:
        return []

    def nli(self, premise: str, hypothesis: str):
        self.nli_calls += 1
        lowered = hypothesis.lower()
        if lowered.count(".") >= 2:
            return "entailment", 0.95
        if "berlin" in premise.lower() and "berlin" in lowered and "nobel" not in lowered:
            return "entailment", 0.95
        return "neutral", 0.1

    def nli_batch(self, pairs):
        return [self.nli(premise, hypothesis) for premise, hypothesis in pairs]

    def usage_snapshot(self):
        return {"nli_calls": self.nli_calls}

    def usage_delta(self, before, after):
        return {"nli_calls": after["nli_calls"] - before["nli_calls"]}


class _Backend:
    def __init__(self, answer: str):
        self.answer = answer
        self.sources = []

    def batch_create_sources(self, notebook_id: str, sources: list[dict]):
        self.sources.extend(sources)
        return {"created": len(sources)}

    def query(self, notebook_id: str, question: str):
        return {
            "answer": self.answer,
            "references": [{"cited_text": "Priya moved to Berlin in 2021."}],
            "backend": "notebooklm-test-reader",
        }


def _engine(fresh_settings):
    settings = replace(
        fresh_settings,
        rerank_enabled=False,
        cascade_enabled=False,
        span_nli_enabled=False,
        abstention_v2_enabled=False,
        abstention_threshold=0.0,
    )
    return Engine(settings, client=_Client(settings.embed_dim))


def _seed(engine: Engine, scope: Scope):
    return engine.ingest_text(
        "Priya moved to Berlin in 2021.",
        source="user",
        scope=scope,
        consolidate_now=False,
    )


def test_notebooklm_governed_recall_verifies_draft_against_immutable_source(fresh_settings):
    engine = _engine(fresh_settings)
    scope = Scope(namespace="notebooklm-governed")
    record = _seed(engine, scope)
    backend = _Backend("Priya moved to Berlin in 2021.")

    out = NotebookLMBridge(engine, backend).governed_recall(
        scope.namespace,
        "Where did Priya move?",
        "nb-test",
    )

    assert out["status"] == AnswerStatus.VERIFIED.value
    assert out["verified"] is True
    assert out["abstained"] is False
    assert out["answer"] == "Priya moved to Berlin in 2021."
    assert out["citations"][0]["memory_id"] == record.memory_id
    assert out["proof"]["refs_verified"] is True
    assert out["reader"]["raw_output_type"] == "UNTRUSTED_DRAFT"
    assert out["reader"]["draft_sha256"] == hashlib.sha256(
        backend.answer.encode("utf-8")
    ).hexdigest()
    assert out["reader"]["proof_model_usage"]["nli_calls"] >= 0


def test_notebooklm_governed_recall_abstains_on_unsupported_draft(fresh_settings):
    engine = _engine(fresh_settings)
    scope = Scope(namespace="notebooklm-unsupported")
    _seed(engine, scope)

    out = NotebookLMBridge(engine, _Backend("Priya moved to the Moon.")).governed_recall(
        scope.namespace,
        "Where did Priya move?",
        "nb-test",
    )

    assert out["status"] == AnswerStatus.ABSTAINED.value
    assert out["verified"] is False
    assert out["abstained"] is True
    assert out["answer"] == ABSTENTION_TEXT
    assert out["citations"] == []
    assert "draft" not in out


def test_notebooklm_governed_recall_abstains_when_one_claim_is_unsupported(fresh_settings):
    engine = _engine(fresh_settings)
    scope = Scope(namespace="notebooklm-partial")
    _seed(engine, scope)
    draft = "Priya moved to Berlin in 2021. She won a Nobel prize."

    out = NotebookLMBridge(engine, _Backend(draft)).governed_recall(
        scope.namespace,
        "What happened to Priya?",
        "nb-test",
    )

    assert out["status"] == AnswerStatus.ABSTAINED.value
    assert out["answer"] == ABSTENTION_TEXT
    assert out["citations"] == []


def test_external_draft_proof_rejects_cross_scope_evidence(fresh_settings):
    engine = _engine(fresh_settings)
    source_scope = Scope(namespace="notebooklm-source")
    record = _seed(engine, source_scope)

    answer = engine.prove_external_draft(
        "Where did Priya move?",
        "Priya moved to Berlin in 2021.",
        [record.memory_id],
        scope=Scope(namespace="notebooklm-other"),
        generated_by="notebooklm",
    )

    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.citations == []
