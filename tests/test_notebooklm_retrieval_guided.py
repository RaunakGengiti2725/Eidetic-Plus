"""Retrieval-guided free-read source selection: eidetic's retriever picks the top_k records
relevant to the question (focused notebook), instead of the whole conversation that buries
facts on long sessions. Synthetic retriever stub -- no live backend."""
from eidetic.integrations.notebooklm import NotebookLMBridge
from eidetic.models import MemoryRecord, Scope


def _rec(mid, text):
    return MemoryRecord(memory_id=mid, content_hash="c" * 64, raw_uri=f"raw://{mid}",
                        source="test", text=text, summary=text[:40],
                        valid_at=1_700_000_000.0, scope=Scope(namespace="rg"))


ALL = [_rec(f"mem_{i:03d}", t) for i, t in enumerate([
    "Filler about game night plans.",
    "Jessica Poole (@jessica_poole_jewellery) is a UK-based jewelry designer I recommended.",
    "Filler about photography courses.",
    "More filler about gitlab issues.",
])]


class _Cand:
    def __init__(self, rec, score):
        self.record = rec
        self.dense_score = score


class _Retriever:
    store = None
    def retrieve(self, question, at=None, scope=None):
        # rank the jewelry record first for a jewelry question
        ranked = sorted(ALL, key=lambda r: (0 if "jewelry" in r.text.lower()
                                            and "jewelry" in question.lower() else 1))
        return [_Cand(r, 0.7 - i * 0.1) for i, r in enumerate(ranked)]


class _Store:
    def claims_by_source(self, mid):
        return []


class _Engine:
    def __init__(self):
        self.retriever = _Retriever()
        self.store = _Store()
        self.retriever.store = self.store


def test_top_k_limits_the_exported_sources():
    b = NotebookLMBridge(_Engine(), backend=None)
    srcs = b.retrieval_guided_sources("rg", "who is the jewelry designer?", top_k=2)
    assert len(srcs) == 2                       # only the focused top-2, not all 4 records


def test_relevant_record_is_selected_first():
    b = NotebookLMBridge(_Engine(), backend=None)
    srcs = b.retrieval_guided_sources("rg", "who is the jewelry designer?", top_k=2)
    joined = " ".join(s["text_content"] for s in srcs)
    assert "@jessica_poole_jewellery" in joined  # the buried fact is now front-and-centre


def test_sources_carry_provenance():
    b = NotebookLMBridge(_Engine(), backend=None)
    srcs = b.retrieval_guided_sources("rg", "jewelry designer?", top_k=1)
    assert srcs and "content_sha256" in srcs[0]["text_content"]


def test_inject_computed_prepends_advisory_source_when_present():
    b = NotebookLMBridge(_Engine(), backend=None)
    # stub the deterministic computation to a known advisory
    b._computed_advisory_source = lambda ns, q: {
        "display_name": "eidetic-computed (deterministic, advisory)",
        "text_content": "--- EIDETIC STRUCTURED RECALL (deterministic computation, ADVISORY) ---\n"
                        "computed_answer: 29 days\n--- END EIDETIC COMPUTED ---"}
    srcs = b.retrieval_guided_sources("rg", "how many days between A and B?", top_k=2,
                                      inject_computed=True)
    assert "EIDETIC STRUCTURED RECALL" in srcs[0]["text_content"]     # advisory is FIRST
    assert "29 days" in srcs[0]["text_content"]
    assert len(srcs) == 3                                             # advisory + top-2


def test_inject_computed_off_by_default_leaves_sources_unchanged():
    b = NotebookLMBridge(_Engine(), backend=None)
    b._computed_advisory_source = lambda ns, q: {"display_name": "x", "text_content": "y"}
    srcs = b.retrieval_guided_sources("rg", "jewelry designer?", top_k=2)  # inject_computed default False
    assert all("EIDETIC STRUCTURED RECALL" not in s["text_content"] for s in srcs)


def test_advisory_source_is_labeled_non_authoritative():
    # the advisory text must tell the reader to trust the records if they disagree
    b = NotebookLMBridge(_Engine(), backend=None)
    b._computed_advisory_source = lambda ns, q: {
        "display_name": "eidetic-computed (deterministic, advisory)",
        "text_content": "ADVISORY ... VERIFY it against the record sources below; if they "
                        "disagree, trust the records."}
    srcs = b.retrieval_guided_sources("rg", "q?", top_k=1, inject_computed=True)
    assert "trust the records" in srcs[0]["text_content"]
