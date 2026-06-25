"""Offline tests for S5 intelligence upgrades: difficulty-adaptive depth + speculative cascade."""
from __future__ import annotations

from dataclasses import replace

from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, RetrievalCandidate, Scope
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _FakeSub:
    def get(self, h):
        raise KeyError(h)


def _retriever(settings, client=object(), store=None):
    store = store or RecordStore(settings.sqlite_path)
    return Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), client, settings)


# ---- difficulty-adaptive retrieval depth ---------------------------------------------------
def test_query_difficulty_and_adaptive_topk(fresh_settings):
    s = replace(fresh_settings, difficulty_adaptive_depth_enabled=True, final_topk=10,
                adaptive_k_min=3)
    r = _retriever(s)
    easy = r._query_difficulty("what is tea")
    hard = r._query_difficulty("how did Alice and Bob and Carol both change after the merger then")
    assert easy < hard
    assert r._adaptive_final_topk("what is tea") < r._adaptive_final_topk(
        "how did Alice and Bob and Carol both change after the merger then before")
    # easy query never returns more than the configured cap, never fewer than the floor.
    assert 3 <= r._adaptive_final_topk("what is tea") <= 10


# ---- speculative cascade -------------------------------------------------------------------
class _CascadeClient:
    def __init__(self):
        self.models_used = []

    def generate_answer(self, q, blocks, model=None):
        self.models_used.append(model)
        return "the right answer" if model == "qwen3-max" else "a wrong answer"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if "right" in hypothesis else ("neutral", 0.3)


def test_cascade_escalates_to_strong_tier_on_grounding_miss(fresh_settings):
    s = replace(fresh_settings, cascade_enabled=True, rerank_enabled=False,
                conflict_resolver_enabled=False, gen_model="qwen3-max", salience_model="qwen-flash",
                abstention_threshold=0.4)
    client = _CascadeClient()
    r = _retriever(s, client=client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="the right answer is here",
                       scope=Scope(), valid_at=1.0)
    cand = RetrievalCandidate(record=rec, dense_score=0.8, fused_score=1.0)

    ans = r.answer("q", precomputed=[cand], verify=True)
    assert client.models_used == ["qwen-flash", "qwen3-max"]   # cheap first, then escalate
    assert ans.answer == "the right answer" and ans.verified    # strong tier grounded


def test_cascade_off_uses_single_routed_call(fresh_settings):
    s = replace(fresh_settings, cascade_enabled=False, rerank_enabled=False,
                conflict_resolver_enabled=False, gen_model="qwen3-max", salience_model="qwen-flash")
    client = _CascadeClient()
    r = _retriever(s, client=client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="x", scope=Scope(), valid_at=1.0)
    r.answer("q", precomputed=[RetrievalCandidate(record=rec, dense_score=0.8, fused_score=1.0)],
             verify=True)
    assert len(client.models_used) == 1                        # no escalation when cascade off
