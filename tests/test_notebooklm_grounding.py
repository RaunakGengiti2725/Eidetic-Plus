"""Deterministic grounding check for NotebookLM answers: quote-faithfulness against the
exported source bytes (rebuilt from the immutable store) + answer token coverage. All
fixtures are SYNTHETIC. No model calls anywhere -- the whole point is that this layer is
free and offline."""
from eidetic.integrations.notebooklm import NotebookLMBridge, format_source
from eidetic.models import MemoryRecord, Scope


def _rec(text: str, mid: str = "mem_ground_000001") -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid,
        content_hash="c" * 64,
        raw_uri=f"raw://{mid}",
        source="test",
        text=text,
        summary=text[:40],
        valid_at=1_700_000_000.0,
        scope=Scope(namespace="ground-ns"),
    )


class _Store:
    def __init__(self, records):
        self._records = records

    def active_records_at(self, t, scope):
        return list(self._records)

    def all_records(self, scope):
        return list(self._records)

    def claims_by_source(self, memory_id):
        return []


class _Engine:
    def __init__(self, store):
        self.store = store


def _bridge(records):
    return NotebookLMBridge(_Engine(_Store(records)), backend=None)


REC = _rec("Priya moved to Lisbon in March 2024 and joined the harbor observatory team.")


def test_verbatim_quote_from_exported_source_is_verbatim():
    bridge = _bridge([REC])
    # quote a real span of the EXPORTED source (the record body survives format_source)
    exported = format_source(REC, [])["text_content"]
    assert "Priya moved to Lisbon" in exported
    out = bridge.verify_grounding(
        "ground-ns", "Priya moved to Lisbon.",
        [{"cited_text": "Priya moved to Lisbon in March 2024 [eidetic:mem_ground_0000]"}])
    assert out["quotes_verbatim"] == 1
    assert out["quotes_unmatched"] == 0


def test_fabricated_quote_is_unmatched():
    bridge = _bridge([REC])
    out = bridge.verify_grounding(
        "ground-ns", "answer",
        [{"cited_text": "Priya moved to Barcelona in 2019 to become a chess referee"}])
    assert out["quotes_unmatched"] == 1
    assert out["quotes_verbatim"] == 0


def test_lightly_reformatted_quote_lands_high_overlap():
    bridge = _bridge([REC])
    # same content tokens, different whitespace/punctuation and one dropped stopword
    out = bridge.verify_grounding(
        "ground-ns", "answer",
        [{"cited_text": "Priya moved to Lisbon -- March 2024 -- joined harbor observatory team"}])
    assert out["quotes_high_overlap"] + out["quotes_verbatim"] == 1
    assert out["quotes_unmatched"] == 0


def test_too_short_quote_is_not_counted_as_grounded():
    bridge = _bridge([REC])
    out = bridge.verify_grounding("ground-ns", "answer", [{"cited_text": "Priya"}])
    assert out["quotes_too_short"] == 1
    assert out["quotes_verbatim"] == 0


def test_answer_token_coverage_full_vs_alien():
    bridge = _bridge([REC])
    grounded = bridge.verify_grounding("ground-ns", "Priya moved to Lisbon", [])
    alien = bridge.verify_grounding(
        "ground-ns", "Zorblatt quantum kettle discovered underwater volcano", [])
    assert grounded["answer_token_coverage"] == 1.0
    assert alien["answer_token_coverage"] < 0.5


def test_method_label_is_honest():
    out = _bridge([REC]).verify_grounding("ground-ns", "x", [])
    assert "NOT NLI" in out["method"]
    assert "proof gate" in out["method"]


class _QuotingBackend:
    """Backend whose query() returns the live nlm parse shape: clean answer + references
    carrying cited_text with the eidetic:<id> tokens."""

    def __init__(self, references):
        self._refs = references

    def query(self, notebook_id, question):
        return {"answer": "Priya moved to Lisbon.",
                "references": self._refs,
                "cited_text": " ".join(str(r.get("cited_text", "")) for r in self._refs),
                "backend": "notebooklm-cli"}


def test_answer_carries_grounding_and_cited_sources():
    refs = [{"cited_text":
             "Priya moved to Lisbon in March 2024 [eidetic:" + REC.memory_id[:16] + "]"}]
    bridge = NotebookLMBridge(_Engine(_Store([REC])), _QuotingBackend(refs))
    out = bridge.answer("ground-ns", "Where did Priya move?", "nbk_1")
    assert out["grounding"]["references_checked"] == 1
    assert out["grounding"]["quotes_verbatim"] == 1
    assert out["grounding"]["answer_token_coverage"] == 1.0
    assert out["cited_sources"]["confirmed_in_eidetic"] == 1
    assert out["user_llm_tokens"] == 0


def test_answer_flags_fabricated_reference_but_still_returns():
    refs = [{"cited_text": "Priya opened a bakery on the moon [eidetic:mem_fake_9999]"}]
    bridge = NotebookLMBridge(_Engine(_Store([REC])), _QuotingBackend(refs))
    out = bridge.answer("ground-ns", "Where did Priya move?", "nbk_1")
    assert out["grounding"]["quotes_unmatched"] == 1
    assert out["cited_sources"]["confirmed_in_eidetic"] == 0
