"""qwen_memory_check: post-hoc per-claim faithfulness AUDITOR of the Gemini free answer.
Synthetic fixtures + a stubbed NLI (no live qwen). Verifies the honest contract:
- decision states (consistent / unsupported / diverges) from per-claim tiers,
- default answer() path unchanged + user_llm_tokens stays 0 (NLI cost never folded in),
- the label is an AUDIT, never a "verified"/gate claim.
"""
from eidetic.integrations.notebooklm import NotebookLMBridge, _HONESTY_BOUNDARIES
from eidetic.models import MemoryRecord, NLILabel, Scope


def _rec(mid, text):
    return MemoryRecord(memory_id=mid, content_hash="c" * 64, raw_uri=f"raw://{mid}",
                        source="test", text=text, summary=text[:40],
                        valid_at=1_700_000_000.0, scope=Scope(namespace="qc"))


class _Store:
    def __init__(self, recs):
        self._by = {r.memory_id: r for r in recs}
        self._recs = recs

    def get_record(self, mid):
        return self._by.get(mid)

    def active_records_at(self, t, scope):
        return list(self._recs)

    def all_records(self, scope):
        return list(self._recs)

    def claims_by_source(self, mid):
        return []


class _Retriever:
    """Stubbed qwen NLI: entails if the claim's first content word is in the premise;
    contradicts if the claim contains 'NOT-IN-MEMORY'."""
    def __init__(self, store):
        self.store = store
        self.calls = 0

    def verify(self, premise, hypothesis):
        self.calls += 1
        if "contradict-me" in hypothesis.lower():
            return NLILabel.CONTRADICTION, 0.9
        key = hypothesis.lower().split()[0] if hypothesis.split() else ""
        if key and key in premise.lower():
            return NLILabel.ENTAILMENT, 0.95
        return NLILabel.NEUTRAL, 0.2


class _Settings:
    span_nli_min_chars = 12
    verify_model = "qwen-plus"


class _Engine:
    def __init__(self, recs):
        self.store = _Store(recs)
        self.retriever = _Retriever(self.store)
        self.settings = _Settings()


REC = _rec("mem_qc_00000001", "Priya relocated to Lisbon and joined the harbor observatory team.")


def _bridge():
    return NotebookLMBridge(_Engine([REC]), backend=None)


def test_consistent_when_every_claim_entailed_lexically():
    b = _bridge()
    out = b.qwen_memory_check("qc", "Where did Priya move?",
                              "Priya relocated to Lisbon.", [REC.memory_id])
    assert out["decision"] == "consistent_with_cited_memory"
    assert out["contradicted"] == 0
    assert out["premise_scope"] == "cited_only"


def test_unsupported_when_a_substantive_claim_is_neutral():
    b = _bridge()
    # second claim shares no content with the cited memory -> neutral -> unsupported
    out = b.qwen_memory_check("qc", "Tell me about Priya.",
                              "Priya relocated to Lisbon. Zorblatt quantum kettles erupted underwater.",
                              [REC.memory_id])
    assert out["decision"] == "unsupported_by_cited_memory"
    assert out["neutral"] >= 1


def test_diverges_on_contradiction():
    b = _bridge()
    out = b.qwen_memory_check("qc", "Where did Priya move?",
                              "Priya definitely did contradict-me about the whole thing.",
                              [REC.memory_id])
    assert out["decision"] == "diverges_from_cited_memory"
    assert out["contradicted"] >= 1


def test_cost_is_reported_and_labeled_metered_not_zero():
    b = _bridge()
    out = b.qwen_memory_check("qc", "q?", "Xyzzy floop never appears in memory at all here.",
                              [REC.memory_id])
    assert "qwen_nli_calls" in out["cost"]
    assert "NOT 0 tokens" in out["cost"]["note"]


def test_label_is_audit_not_verified():
    note = _HONESTY_BOUNDARIES["qwen_memory_check"]
    assert "AUDIT" in note
    assert "NOT a correctness guarantee" in note
    assert "verify-or-abstain generation gate" in note  # explicitly disclaims the gate


class _QBackend:
    def query(self, notebook_id, question):
        return {"answer": "Priya relocated to Lisbon.",
                "references": [{"cited_text": f"Priya relocated to Lisbon [eidetic:{REC.memory_id[:16]}]"}],
                "cited_text": f"Priya relocated to Lisbon [eidetic:{REC.memory_id[:16]}]",
                "backend": "nlm-cli"}


def test_answer_default_has_no_qwen_check_and_zero_tokens():
    b = NotebookLMBridge(_Engine([REC]), _QBackend())
    out = b.answer("qc", "Where did Priya move?", "nbk")
    assert "qwen_memory_check" not in out          # opt-in only
    assert out["user_llm_tokens"] == 0


def test_answer_optin_adds_check_and_keeps_user_tokens_zero():
    eng = _Engine([REC])
    b = NotebookLMBridge(eng, _QBackend())
    out = b.answer("qc", "Where did Priya move?", "nbk", verify_with_qwen=True)
    assert "qwen_memory_check" in out
    assert out["user_llm_tokens"] == 0            # NLI cost NEVER folded into caller tokens
    assert out["qwen_memory_check"]["cost"]["qwen_nli_calls"] >= 0
