"""The epistemic map: deterministic enumerators, cell lifecycle, proof-gated KNOWN,
snapshot/delta auditability, curiosity probe templates + diagnosis routing.
All offline -- zero model calls."""
import json
import time

import pytest

from eidetic.epistemic.cells import CellKind, CellState, EpistemicCell, cell_id_for
from eidetic.epistemic.curiosity import _diagnose_answer, probe_for_cell, run_curiosity
from eidetic.epistemic.gaps import (enumerate_cells, event_missing_date,
                                    multi_active_conflict, superseded_no_current,
                                    temporal_hole)
from eidetic.epistemic.map import KnowledgeMap
from eidetic.events import EventRecord
from eidetic.models import Answer, AnswerStatus, Citation, NLILabel, Scope, now
from eidetic.store import RecordStore


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path / "s.sqlite")


@pytest.fixture
def kmap(tmp_path):
    return KnowledgeMap(tmp_path / "map.sqlite")


SCOPE = Scope(namespace="t")


def _edge(store, src, dst, relation, valid_at, invalid_at=None, inferred=False):
    from eidetic.models import Edge
    e = Edge(src=src, dst=dst, relation=relation, scope=SCOPE,
             valid_at=valid_at, invalid_at=invalid_at, inferred=inferred)
    store.add_edge(e)
    return e


def _verified(question="q", answer="a", n_citations=1) -> Answer:
    cits = [Citation(memory_id=f"mem_{i}", content_hash=f"h{i}", raw_uri=f"u{i}",
                     source="s", valid_at=1.0, snippet="x",
                     nli_label=NLILabel.ENTAILMENT, nli_score=0.9)
            for i in range(n_citations)]
    return Answer(question=question, answer=answer, status=AnswerStatus.VERIFIED,
                  verified=True, confidence=0.9, citations=cits)


# ---- cells ---------------------------------------------------------------------

def test_cell_id_deterministic_and_normalized():
    a = cell_id_for("t", None, None, "fact", "Ada  Lovelace", "employer")
    b = cell_id_for("t", None, None, "fact", "ada lovelace", "EMPLOYER")
    assert a == b
    assert cell_id_for("t2", None, None, "fact", "ada lovelace", "employer") != a


def test_cell_brief_truncates_and_never_ships_proof():
    cell = EpistemicCell(namespace="t", kind="query", subject="s" * 500,
                         reason="r" * 500, proof={"answer": "SECRET"})
    brief = cell.brief()
    assert len(brief["subject"]) <= 160 and len(brief["reason"]) <= 240
    assert "proof" not in brief and "SECRET" not in json.dumps(brief)


# ---- gap enumerators -------------------------------------------------------------

def test_superseded_no_current_names_the_gap(store):
    t0 = now() - 1000
    _edge(store, "ada", "acme", "employer", t0, invalid_at=t0 + 10)
    cells = superseded_no_current(store, SCOPE)
    assert len(cells) == 1
    c = cells[0]
    assert (c.kind, c.state) == (CellKind.FACT.value, CellState.UNKNOWN.value)
    assert c.subject == "ada" and c.relation == "employer"
    assert "acme" in c.reason and c.evidence_ids


def test_superseded_ignores_active_and_multivalued(store):
    t0 = now() - 1000
    _edge(store, "ada", "acme", "employer", t0)                      # still active
    _edge(store, "bob", "rome", "visited", t0, invalid_at=t0 + 1)    # multi-valued
    assert superseded_no_current(store, SCOPE) == []


def test_multi_active_conflict_contested(store):
    t0 = now() - 1000
    _edge(store, "ada", "acme", "employer", t0)
    _edge(store, "ada", "globex", "employer", t0 + 5)
    cells = multi_active_conflict(store, SCOPE)
    assert len(cells) == 1
    assert cells[0].state == CellState.CONTESTED.value
    assert cells[0].kind == CellKind.CONFLICT.value


def test_temporal_hole_needs_wide_gap(store):
    t0 = now() - 90 * 86400
    _edge(store, "ada", "acme", "employer", t0, invalid_at=t0 + 86400)
    _edge(store, "ada", "globex", "employer", t0 + 60 * 86400)       # 59-day hole
    holes = temporal_hole(store, SCOPE)
    assert len(holes) == 1 and holes[0].kind == CellKind.TEMPORAL_HOLE.value
    # narrow gap -> no cell
    store2_edges = temporal_hole(store, SCOPE, min_gap_seconds=365 * 86400.0)
    assert store2_edges == []


def test_event_missing_date(store):
    store.add_event(EventRecord(subject="ada", verb="visited", object="rome",
                                fact="ada visited rome", namespace="t", valid_at=now()))
    cells = event_missing_date(store, SCOPE)
    assert len(cells) == 1 and cells[0].kind == CellKind.EVENT_DATE.value
    store.add_event(EventRecord(subject="b", verb="met", object="c", fact="b met c",
                                namespace="t", start=now(), valid_at=now()))
    assert len(event_missing_date(store, SCOPE)) == 1     # dated event not a gap


def test_end_without_newer_begin_names_the_gap(store):
    from eidetic.epistemic.gaps import end_without_newer_begin
    t0 = now() - 1000
    _edge(store, "maya", "alvalade gym", "joined", t0)
    _edge(store, "maya", "alvalade gym membership", "cancelled membership", t0 + 100)
    cells = end_without_newer_begin(store, SCOPE)
    assert len(cells) == 1
    assert cells[0].relation.startswith("current_state_of")
    # a NEWER begin-verb fact in the same object family closes the gap
    _edge(store, "maya", "campo gym membership", "signed up for", t0 + 200)
    assert end_without_newer_begin(store, SCOPE) == []


def test_cross_layer_conflict_flags_disagreeing_surfaces(store):
    from eidetic.epistemic.gaps import cross_layer_conflict
    from eidetic.models import ClaimRecord, MemoryRecord
    t0 = now() - 1000
    _edge(store, "dana", "+351 912 111 222", "has phone number", t0, invalid_at=t0 + 10)
    _edge(store, "dana", "+351 933 444 555", "has phone number", t0 + 10)
    mem = MemoryRecord(text="Dana's phone number is +351 912 111 222.",
                       scope=SCOPE, valid_at=t0)
    store.upsert_record(mem)
    store.add_claim(ClaimRecord(claim_type="state", scope=SCOPE, subject="Dana",
                                value="Dana's phone number is +351 912 111 222.",
                                valid_at=t0, source_memory_id=mem.memory_id))
    cells = cross_layer_conflict(store, SCOPE)
    assert len(cells) == 1
    c = cells[0]
    assert c.state == CellState.CONTESTED.value and "[cross-layer]" in c.relation
    assert "912" in c.reason and len(c.evidence_ids) >= 3


def test_enumerate_cells_contested_outranks_unknown(store):
    t0 = now() - 1000
    _edge(store, "ada", "acme", "employer", t0)
    _edge(store, "ada", "globex", "employer", t0 + 5)
    cells = enumerate_cells(store, SCOPE)
    by_key = {(c.subject, c.relation.split("@")[0]): c for c in cells}
    assert by_key[("ada", "employer")].state == CellState.CONTESTED.value


# ---- map lifecycle ---------------------------------------------------------------

def test_mark_known_requires_verified_with_citations(kmap):
    cell = EpistemicCell(namespace="t", kind="query", subject="where is ada?")
    kmap.upsert_cell(cell)
    ab = Answer.abstain("where is ada?", note="no source entails the answer")
    assert kmap.mark_known(cell.cell_id, ab, cause="x") is False
    no_cit = Answer(question="q", answer="a", status=AnswerStatus.VERIFIED,
                    verified=True, citations=[])
    assert kmap.mark_known(cell.cell_id, no_cit, cause="x") is False
    assert kmap.get_cell(cell.cell_id).state == CellState.UNKNOWN.value
    ok = kmap.mark_known(cell.cell_id, _verified(), cause="probe verified")
    assert ok is True
    got = kmap.get_cell(cell.cell_id)
    assert got.state == CellState.KNOWN.value
    assert got.proof["citations"][0]["memory_id"] == "mem_0"


def test_enumerator_refresh_never_demotes_proven_known(kmap):
    cell = EpistemicCell(namespace="t", kind="fact", subject="ada", relation="employer")
    kmap.upsert_cell(cell)
    kmap.mark_known(cell.cell_id, _verified(), cause="probe")
    again = EpistemicCell(namespace="t", kind="fact", subject="ada", relation="employer",
                          state=CellState.UNKNOWN.value, origin="enumerator")
    kmap.upsert_cell(again, cause="re-enumeration")
    assert kmap.get_cell(cell.cell_id).state == CellState.KNOWN.value


def test_rebuild_closes_stale_enumerated_keeps_minted(store, kmap):
    t0 = now() - 1000
    e = _edge(store, "ada", "acme", "employer", t0, invalid_at=t0 + 10)
    kmap.rebuild(store, SCOPE)
    assert kmap.counts(SCOPE)["unknown_n"] == 1
    minted = EpistemicCell(namespace="t", kind="query", subject="who is bob?", origin="ask")
    kmap.upsert_cell(minted)
    # the gap resolves: a new active edge arrives
    _edge(store, "ada", "globex", "employer", now())
    out = kmap.rebuild(store, SCOPE)
    assert out["closed"] == 1
    assert kmap.get_cell(minted.cell_id) is not None          # minted cell survives
    ledger = kmap.transitions_since(SCOPE, 0)
    assert any(t["to_state"] == "CLOSED" for t in ledger)


def test_on_answer_mints_unknown_then_marks_known(kmap):
    q = "Where does Ada work now?"
    ab = Answer.abstain(q, note="insufficient evidence (coverage 0.10)")
    cid = kmap.on_answer(q, ab, SCOPE)
    assert cid and kmap.get_cell(cid).state == CellState.UNKNOWN.value
    # recurrence bumps info gain
    gain1 = kmap.get_cell(cid).info_gain
    kmap.on_answer(q, ab, SCOPE)
    assert kmap.get_cell(cid).info_gain > gain1
    kmap.on_answer(q, _verified(question=q), SCOPE)
    assert kmap.get_cell(cid).state == CellState.KNOWN.value


def test_on_contradiction_mints_contested(kmap):
    cid = kmap.on_contradiction("is ada at acme?", ["mem_1", "mem_2"], SCOPE,
                                note="active memory contradicts the answer")
    cell = kmap.get_cell(cid)
    assert cell.state == CellState.CONTESTED.value
    assert cell.evidence_ids == ["mem_1", "mem_2"]


def test_snapshot_delta_and_digest(kmap, tmp_path):
    a = EpistemicCell(namespace="t", kind="query", subject="q1?", origin="ask")
    kmap.upsert_cell(a)
    before = kmap.snapshot(SCOPE, tmp_path / "before.json", label="day0")
    kmap.mark_known(a.cell_id, _verified(), cause="probe")
    after = kmap.snapshot(SCOPE, tmp_path / "after.json", label="day1")
    d = KnowledgeMap.delta(before, after)
    assert d["unknown_delta"] == -1 and d["known_delta"] == 1
    assert d["unknown_closed"] == [a.cell_id]
    assert before["digest"] != after["digest"]


# ---- curiosity -------------------------------------------------------------------

def test_probe_templates_deterministic():
    fact = EpistemicCell(namespace="t", kind="fact", subject="ada", relation="employer")
    assert probe_for_cell(fact) == "What is ada's current employer?"
    q = EpistemicCell(namespace="t", kind="query", subject="where does ada work?")
    assert probe_for_cell(q) == "where does ada work?"
    law = EpistemicCell(namespace="t", kind="law_prediction", subject="ada",
                        relation="works_in?london")
    assert "ada works in london" in probe_for_cell(law)
    assert probe_for_cell(fact) == probe_for_cell(fact)     # stable


def test_diagnose_answer_mapping():
    from eidetic.dreaming.repair import Diagnosis
    assert _diagnose_answer(_verified(), 0.4) == Diagnosis.PASSED
    ab = Answer.abstain("q", note="active memory contradicts the answer")
    assert _diagnose_answer(ab, 0.4) == Diagnosis.CONTRADICTED
    missing = Answer.abstain("q", note="insufficient evidence (coverage 0.05)")
    assert _diagnose_answer(missing, 0.4) == Diagnosis.MISSING
    hard = Answer.abstain("q", note="no source entails the answer", retrieved_count=5)
    assert _diagnose_answer(hard, 0.4) == Diagnosis.HARD_TO_RETRIEVE


class _FakeSettings:
    abstention_threshold = 0.4


class _FakeEngine:
    """Probe path double: canned answers keyed by substring."""
    def __init__(self, kmap, canned):
        self.knowledge_map_store = kmap
        self.settings = _FakeSettings()
        self._canned = canned
        self.asked = []

    def ask(self, probe, **kw):
        self.asked.append(probe)
        for key, ans in self._canned.items():
            if key in probe:
                return ans
        return Answer.abstain(probe, note="no source entails the answer")


class _ListAgenda:
    def __init__(self):
        self.tasks = []

    def enqueue(self, task):
        self.tasks.append(task)
        return task.dedup_key


def test_run_curiosity_routes_outcomes(kmap, tmp_path):
    known_q = EpistemicCell(namespace="t", kind="query", subject="what is x?",
                            origin="ask", info_gain=2.0)
    missing_q = EpistemicCell(namespace="t", kind="query", subject="what is y?",
                              origin="ask", info_gain=1.0)
    kmap.upsert_cell(known_q)
    kmap.upsert_cell(missing_q)
    eng = _FakeEngine(kmap, {"what is x?": _verified(question="what is x?")})
    agenda = _ListAgenda()
    log = tmp_path / "probes.jsonl"
    report = run_curiosity(eng, SCOPE, max_probes=8, agenda=agenda, probes_log=log)
    assert report["probed"] == 2 and report["passed"] == 1
    assert kmap.get_cell(known_q.cell_id).state == CellState.KNOWN.value
    assert kmap.get_cell(missing_q.cell_id).state == CellState.UNKNOWN.value
    assert kmap.get_cell(missing_q.cell_id).probe_count == 1
    assert len(agenda.tasks) == 1 and agenda.tasks[0].origin == "unknown_cell"
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 2 and all("diagnosis" in r for r in rows)


def test_curiosity_cooldown_skips_recently_probed(kmap):
    c = EpistemicCell(namespace="t", kind="query", subject="q?", origin="ask")
    c.last_probed = now()
    kmap.upsert_cell(c)
    assert kmap.sample_frontier(SCOPE, 5, probe_cooldown_sec=3600.0) == []
    assert len(kmap.sample_frontier(SCOPE, 5, probe_cooldown_sec=0.0)) == 1
