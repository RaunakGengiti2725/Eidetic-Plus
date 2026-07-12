"""P1 organs: law induction lifecycle (verify-or-discard), the contested-resolution
program (bi-temporal first, probe second, hold honestly), and the READ_CLAIM_SELECT
read stage (r19 read-recovery). Offline; model paths use deterministic fakes."""
import re

import pytest

from eidetic.epistemic.cells import CellKind, CellState, EpistemicCell
from eidetic.epistemic.contested import contested_wave, resolve_contested_cell
from eidetic.epistemic.laws import (CANDIDATE, FALSIFIED, LAW, VERIFYING, LawBook,
                                    law_verification_wave)
from eidetic.epistemic.map import KnowledgeMap
from eidetic.models import (Answer, AnswerStatus, Citation, Edge, NLILabel, Scope, now)
from eidetic.store import RecordStore

SCOPE = Scope(namespace="t")


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path / "s.sqlite")


@pytest.fixture
def kmap(tmp_path):
    return KnowledgeMap(tmp_path / "map.sqlite")


def _edge(store, src, dst, relation, valid_at=None, invalid_at=None,
          source_memory_id="", inferred=False):
    e = Edge(src=src, dst=dst, relation=relation, scope=SCOPE,
             valid_at=now() - 1000 if valid_at is None else valid_at,
             invalid_at=invalid_at, source_memory_id=source_memory_id,
             inferred=inferred)
    store.add_edge(e)
    return e


def _verified(question="q", answer="a", memory_ids=("mem_0",)) -> Answer:
    cits = [Citation(memory_id=m, content_hash=f"h_{m}", raw_uri="u", source="s",
                     valid_at=1.0, snippet="x", nli_label=NLILabel.ENTAILMENT,
                     nli_score=0.9) for m in memory_ids]
    return Answer(question=question, answer=answer, status=AnswerStatus.VERIFIED,
                  verified=True, confidence=0.9, citations=cits)


def _seed_rule_graph(store):
    """born_in & capital_of => citizen_of, witnessed twice, predicted once more."""
    for person, city, country in (("ada", "london", "uk"), ("bob", "paris", "france")):
        _edge(store, person, city, "born_in")
        _edge(store, city, country, "capital_of")
        _edge(store, person, country, "citizen_of")
    _edge(store, "eve", "rome", "born_in")            # eve's citizenship unwitnessed
    _edge(store, "rome", "italy", "capital_of")


# ---- law lifecycle -----------------------------------------------------------------

def test_mine_candidates_persists_rules(store, kmap):
    _seed_rule_graph(store)
    book = LawBook(kmap)
    out = book.mine_candidates(store, SCOPE, min_confidence=0.5, min_support=2)
    assert out["new_candidates"] >= 1
    laws = book.laws(SCOPE, CANDIDATE)
    assert any(l["r3"] == "citizen_of" for l in laws)
    # idempotent re-mine: no duplicates
    again = book.mine_candidates(store, SCOPE, min_confidence=0.5, min_support=2)
    assert again["new_candidates"] == 0


def test_law_prediction_enumerated_as_unknown_cell(store, kmap):
    _seed_rule_graph(store)
    kmap.rebuild(store, SCOPE)
    cells = kmap.cells_in_state(SCOPE, CellState.UNKNOWN.value, limit=100)
    law_cells = [c for c in cells if c.kind == CellKind.LAW_PREDICTION.value]
    assert any(c.subject == "eve" and "citizen_of" in c.relation for c in law_cells)


def test_record_check_promotion_needs_min_checks(store, kmap):
    _seed_rule_graph(store)
    book = LawBook(kmap)
    book.mine_candidates(store, SCOPE, min_confidence=0.5, min_support=2)
    law = next(l for l in book.laws(SCOPE, CANDIDATE) if l["r3"] == "citizen_of")
    book.begin_verification(law["law_id"])
    assert book.record_check(law["law_id"], passed=True) == VERIFYING
    assert book.record_check(law["law_id"], passed=True) == VERIFYING
    assert book.record_check(law["law_id"], passed=True) == LAW
    assert book.counts(SCOPE)["laws_promoted"] == 1


def test_falsification_is_immediate_and_mints_contested(store, kmap):
    _seed_rule_graph(store)
    book = LawBook(kmap)
    book.mine_candidates(store, SCOPE, min_confidence=0.5, min_support=2)
    law = next(l for l in book.laws(SCOPE, CANDIDATE) if l["r3"] == "citizen_of")
    status = book.record_check(law["law_id"], passed=False,
                               counterexample="eve holds vatican citizenship")
    assert status == FALSIFIED
    contested = kmap.cells_in_state(SCOPE, CellState.CONTESTED.value, limit=10)
    assert any(c.relation == "falsified_law" and "vatican" in c.reason
               for c in contested)
    # a falsified law never resurrects
    assert book.record_check(law["law_id"], passed=True) == FALSIFIED


def test_apply_promoted_writes_only_inferred_edges(store, kmap):
    _seed_rule_graph(store)
    book = LawBook(kmap)
    book.mine_candidates(store, SCOPE, min_confidence=0.5, min_support=2)
    law = next(l for l in book.laws(SCOPE, CANDIDATE) if l["r3"] == "citizen_of")
    for _ in range(3):
        book.record_check(law["law_id"], passed=True)

    class _Eng:
        pass
    eng = _Eng()
    eng.store = store
    out = book.apply_promoted(eng, SCOPE)
    assert out["applied_inferred_edges"] >= 1
    inferred = [e for e in store.all_edges(SCOPE, include_inferred=True) if e.inferred]
    assert any(e.src == "eve" and e.dst == "italy" and e.relation == "citizen_of"
               and e.provenance.startswith("law:") for e in inferred)
    observed = store.all_edges(SCOPE, include_inferred=False)
    assert not any(e.src == "eve" and e.relation == "citizen_of" for e in observed)


class _LawProbeEngine:
    """ask() verifies eve's citizenship (the law's prediction holds)."""
    def __init__(self, store, kmap):
        self.store = store
        self.knowledge_map_store = kmap

        class _S:
            abstention_threshold = 0.4
        self.settings = _S()

    def ask(self, probe, **kw):
        if "eve" in probe and "citizen" in probe:
            return _verified(question=probe, answer="eve is a citizen of italy")
        return Answer.abstain(probe, note="no source entails the answer")


def test_law_verification_wave_records_checks(store, kmap):
    _seed_rule_graph(store)
    kmap.rebuild(store, SCOPE)                       # enumerates the prediction cell
    book = LawBook(kmap)
    book.mine_candidates(store, SCOPE, min_confidence=0.5, min_support=2)
    eng = _LawProbeEngine(store, kmap)
    report = law_verification_wave(eng, SCOPE, max_probes=4, promote_min_checks=1)
    assert report["probed"] >= 1 and report["passed"] >= 1
    assert report["promoted"] >= 1
    assert book.counts(SCOPE)["laws_promoted"] >= 1


# ---- contested resolution program ---------------------------------------------------

class _ContestedEngine:
    def __init__(self, store, kmap, verdict=None):
        self.store = store
        self.knowledge_map_store = kmap
        self._verdict = verdict

    def ask(self, probe, **kw):
        if self._verdict is not None:
            return self._verdict
        return Answer.abstain(probe, note="no source entails the answer")


def test_bitemporal_supersession_resolves_without_probe(store, kmap):
    t0 = now() - 5000
    old = _edge(store, "ada", "acme", "employer", valid_at=t0)
    new = _edge(store, "ada", "globex", "employer", valid_at=t0 + 3000)
    kmap.rebuild(store, SCOPE)                       # -> CONTESTED conflict cell
    cell = kmap.cells_in_state(SCOPE, CellState.CONTESTED.value, limit=5)[0]
    eng = _ContestedEngine(store, kmap)
    out = resolve_contested_cell(eng, cell.cell_id, scope=SCOPE, allow_probe=False)
    assert out["outcome"] == "resolved_supersession"
    active = store.active_edges_at(None, SCOPE)
    assert [e.dst for e in active if e.src == "ada" and e.relation == "employer"] \
        == ["globex"]
    closed = [e for e in store.all_edges(SCOPE) if e.edge_id == old.edge_id]
    assert closed[0].invalid_at == pytest.approx(new.valid_at)   # closed, not deleted
    assert kmap.get_cell(cell.cell_id) is None                   # cell closed w/ ledger


def test_identical_times_hold_contested_without_verified_probe(store, kmap):
    t0 = now() - 5000
    _edge(store, "ada", "acme", "employer", valid_at=t0)
    _edge(store, "ada", "globex", "employer", valid_at=t0)
    kmap.rebuild(store, SCOPE)
    cell = kmap.cells_in_state(SCOPE, CellState.CONTESTED.value, limit=5)[0]
    eng = _ContestedEngine(store, kmap)              # probe abstains
    out = resolve_contested_cell(eng, cell.cell_id, scope=SCOPE, allow_probe=True)
    assert out["outcome"] == "held_contested"
    held = kmap.get_cell(cell.cell_id)
    assert held.state == CellState.CONTESTED.value
    assert "resolution program held" in held.reason
    assert len(store.active_edges_at(None, SCOPE)) == 2          # nothing closed


def test_verified_probe_resolves_and_closes_losers(store, kmap):
    t0 = now() - 5000
    _edge(store, "ada", "acme", "employer", valid_at=t0, source_memory_id="mem_old")
    _edge(store, "ada", "globex", "employer", valid_at=t0, source_memory_id="mem_new")
    kmap.rebuild(store, SCOPE)
    cell = kmap.cells_in_state(SCOPE, CellState.CONTESTED.value, limit=5)[0]
    eng = _ContestedEngine(store, kmap,
                           verdict=_verified(answer="ada works at globex",
                                             memory_ids=("mem_new",)))
    out = resolve_contested_cell(eng, cell.cell_id, scope=SCOPE, allow_probe=True)
    assert out["outcome"] == "resolved_verified"
    active = [e for e in store.active_edges_at(None, SCOPE)
              if e.src == "ada" and e.relation == "employer"]
    assert [e.dst for e in active] == ["globex"]
    resolved = kmap.get_cell(cell.cell_id)
    assert resolved.state == CellState.KNOWN.value
    assert resolved.proof["citations"][0]["memory_id"] == "mem_new"


def test_cross_layer_repair_closes_loser_claims(store, kmap):
    from eidetic.epistemic.gaps import cross_layer_conflict
    from eidetic.models import ClaimRecord, MemoryRecord
    t0 = now() - 5000
    _edge(store, "dana", "+351 912 111 222", "has phone number",
          valid_at=t0, invalid_at=t0 + 10)
    _edge(store, "dana", "+351 933 444 555", "has phone number", valid_at=t0 + 10)
    mem = MemoryRecord(text="Dana's phone number is +351 912 111 222.",
                       scope=SCOPE, valid_at=t0)
    store.upsert_record(mem)
    loser_claim = ClaimRecord(claim_type="state", scope=SCOPE, subject="Dana",
                              value="Dana's phone number is +351 912 111 222.",
                              valid_at=t0, source_memory_id=mem.memory_id)
    store.add_claim(loser_claim)
    for cell in cross_layer_conflict(store, SCOPE):
        kmap.upsert_cell(cell)
    cell = kmap.cells_in_state(SCOPE, CellState.CONTESTED.value, limit=5)[0]
    eng = _ContestedEngine(store, kmap)
    out = resolve_contested_cell(eng, cell.cell_id, scope=SCOPE, allow_probe=False)
    assert out["outcome"] == "resolved_cross_layer_repair"
    assert out["steps"][0]["claims_closed"][0]["claim_id"] == loser_claim.claim_id
    # the claim layer no longer serves the loser as current
    active_vals = [str(c.value) for c in store.active_claims_at(None, SCOPE)]
    assert not any("912" in v for v in active_vals)
    # history preserved: the claim row still exists, bi-temporally closed
    all_claims = store.claims_in_scope(SCOPE)
    assert any(c.claim_id == loser_claim.claim_id and c.invalid_at is not None
               for c in all_claims)
    assert kmap.get_cell(cell.cell_id) is None


def test_contested_wave_bounded(store, kmap):
    t0 = now() - 5000
    for i in range(6):
        _edge(store, f"s{i}", "a", "employer", valid_at=t0)
        _edge(store, f"s{i}", "b", "employer", valid_at=t0)
    kmap.rebuild(store, SCOPE)
    eng = _ContestedEngine(store, kmap)
    out = contested_wave(eng, SCOPE, max_programs=3, allow_probe=False)
    assert out["programs"] == 3
    assert out["outcomes"].get("held_contested", 0) == 3


# ---- READ_CLAIM_SELECT ----------------------------------------------------------------

def test_claim_select_draft_extracts_verbatim():
    from eidetic.retrieval import _claim_select_draft

    class _Rec:
        def __init__(self, text):
            self.text = text
            self.summary = ""

    class _Cand:
        def __init__(self, text):
            self.record = _Rec(text)

    cands = [_Cand("evan: hey sam! long time. you could try flavored seltzer water "
                   "instead of soda. also dark chocolate with high cocoa content is "
                   "a nice candy alternative.\nsam: thanks!")]
    out = _claim_select_draft("What did Evan suggest about soda and candy?", cands)
    assert "flavored seltzer water" in out
    assert not out.startswith("evan:")               # speaker prefix stripped
    assert _claim_select_draft("completely unrelated query about quantum physics",
                               cands) == ""          # weak extraction refuses


def test_claim_select_flag_recovers_provable_answer(tmp_path, monkeypatch):
    """A/B through the REAL engine: paraphrase draft fails NLI -> abstain; with
    READ_CLAIM_SELECT=1 the verbatim sentence verifies. Same store, same question."""
    import numpy as np
    import hashlib as _hashlib

    class _Client:
        dim = 64

        def _embed(self, text):
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
                v[int(_hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        def embed_text(self, t):
            return self._embed(t)

        def embed_texts(self, ts):
            return (np.stack([self._embed(t) for t in ts])
                    if ts else np.zeros((0, self.dim), np.float32))

        def score_importance(self, t):
            return 0.5

        def extract_edges(self, t):
            return []

        def find_contradictions(self, a, b):
            return []

        def extract_current_value_matches(self, q, c):
            return []

        def generate_probes(self, m, n=3):
            return []

        def rerank(self, q, docs, top_n):
            return [(i, 1.0 - 0.001 * i) for i in range(min(top_n, len(docs)))]

        def generate_answer(self, q, blocks, model=None):
            return "He thinks meticulousness produces exceptional work."   # paraphrase

        def nli(self, premise, hypothesis):
            pt = set(re.findall(r"[a-z0-9]+", premise.lower()))
            ht = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))
            overlap = len(pt & ht) / (len(ht) or 1)
            return ("entailment", 0.9) if overlap >= 0.6 else ("neutral", 0.4)

    def _fresh_engine():
        from eidetic.config import get_settings
        from eidetic.engine import Engine
        get_settings.cache_clear()
        s = get_settings()
        return Engine(s, client=_Client())

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    fact = ("Calvin believes paying attention to small details makes an artist "
            "create something extraordinary")
    q = "What does Calvin believe makes an artist create something extraordinary?"

    monkeypatch.setenv("READ_CLAIM_SELECT", "0")
    eng = _fresh_engine()
    eng.ingest_text(fact, source="conv1")
    base = eng.ask(q, verify=True, use_cache=False)
    eng.close()
    assert base.status == AnswerStatus.ABSTAINED     # paraphrase grounded nowhere

    monkeypatch.setenv("READ_CLAIM_SELECT", "1")
    eng2 = _fresh_engine()
    flagged = eng2.ask(q, verify=True, use_cache=False)
    eng2.close()
    assert flagged.status == AnswerStatus.VERIFIED   # verbatim sentence proves
    assert "small details" in flagged.answer.lower()
    assert flagged.citations
    from eidetic.config import get_settings
    get_settings.cache_clear()
