"""MCP surfaces of the epistemic organism: knowledge_map / explain_gap /
research_status / improve(dry_run). Real Engine + fake client, no key, no network."""
from __future__ import annotations

import asyncio
import hashlib
import re

import numpy as np
import pytest

import eidetic.mcp_server as mcp_server


class FakeClient:
    def __init__(self, dim: int):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, text):
        return self._embed(text)

    def embed_texts(self, texts):
        return (np.stack([self._embed(t) for t in texts])
                if texts else np.zeros((0, self.dim), np.float32))

    def score_importance(self, text):
        return 0.5

    def extract_edges(self, text):
        return []

    def find_contradictions(self, new_fact, candidates):
        return []

    def extract_current_value_matches(self, query, candidates):
        return []

    def generate_probes(self, memory_text, n=3):
        return []

    def rerank(self, query, documents, top_n):
        return [(i, 1.0 - 0.001 * i) for i in range(min(top_n, len(documents)))]

    def generate_answer(self, question, context_blocks, model=None):
        return context_blocks[0][:300] if context_blocks else "I do not have that in memory."

    def nli(self, premise, hypothesis):
        pt = set(re.findall(r"[a-z0-9]+", premise.lower()))
        ht = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))
        return (("entailment", 0.9) if len(pt & ht) / (len(ht) or 1) >= 0.5
                else ("neutral", 0.4))


@pytest.fixture()
def mcp_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("EPISTEMIC_MAP", "1")
    monkeypatch.setenv("AUTORESEARCH", "1")
    monkeypatch.setenv("AUTORESEARCH_DIR", str(tmp_path / "ar"))
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    from eidetic.config import get_settings
    from eidetic.engine import Engine
    get_settings.cache_clear()
    settings = get_settings()
    eng = Engine(settings, client=FakeClient(settings.embed_dim))
    monkeypatch.setattr(mcp_server, "_engine", eng)
    yield eng
    monkeypatch.setattr(mcp_server, "_engine", None)
    get_settings.cache_clear()


def test_new_tools_listed():
    tools = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"knowledge_map", "explain_gap", "research_status", "improve"} <= tools


def test_knowledge_map_counts_and_gaps(mcp_engine):
    from eidetic.models import Answer, Scope
    scope = Scope(namespace="km")
    ab = Answer.abstain("Where does Ada work now?",
                        note="abstained: no source entails the answer", retrieved_count=2)
    mcp_engine._epistemic_after_ask_sync("Where does Ada work now?", ab, scope, [])
    out = mcp_server.knowledge_map(namespace="km")
    assert out["enabled"] and out["unknown_n"] == 1
    gap = out["top_gaps"][0]
    assert gap["state"] == "UNKNOWN" and gap["cell_id"].startswith("cell_")
    # rebuild path stays consistent (no enumerable store content in this ns)
    out2 = mcp_server.knowledge_map(namespace="km", rebuild=True)
    assert out2["unknown_n"] == 1


def test_explain_gap_ships_probe_and_why(mcp_engine):
    from eidetic.models import Answer, Scope
    scope = Scope(namespace="eg")
    ab = Answer.abstain("What is the launch date?",
                        note="abstained: insufficient evidence (coverage 0.10)")
    mcp_engine._epistemic_after_ask_sync("What is the launch date?", ab, scope, [])
    gap = mcp_server.knowledge_map(namespace="eg")["top_gaps"][0]
    exp = mcp_server.explain_gap(gap["cell_id"])
    assert exp["suggested_probe"] == "what is the launch date?"
    assert exp["reason"].startswith("abstained")
    with pytest.raises(RuntimeError, match="no such cell"):
        mcp_server.explain_gap("cell_doesnotexist00000")


def test_research_status_no_content_leak(mcp_engine):
    from eidetic.models import Answer, Scope
    ab = Answer.abstain("Secret question about Ada?",
                        note="abstained: no source entails the answer")
    mcp_engine._epistemic_after_ask_sync("Secret question about Ada?", ab,
                                    Scope(namespace="rs"), [])
    out = mcp_server.research_status()
    assert out["agenda"]["queued"] >= 1
    assert out["champion"]["champion_id"] == "baseline"
    flat = str(out)
    assert "Secret question" not in flat        # method metadata only


def test_improve_dry_run_zero_model_calls(mcp_engine):
    out = mcp_server.improve(namespace="im", dry_run=True)
    assert out["dry_run"] is True
    assert "map_delta" in out and "curiosity" not in out
