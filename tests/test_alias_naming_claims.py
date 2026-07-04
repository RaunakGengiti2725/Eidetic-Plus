"""Alias/naming claims: vocative nicknames, quoted titles, and named-category artifacts
become typed claims at write time; the read side answers naming questions from them
without the reader. All dialogues are FABRICATED shapes; no benchmark rows.
"""
from __future__ import annotations

from datetime import datetime

from eidetic.models import MemoryRecord, Scope
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.smqe.executor import execute_plan
from eidetic.smqe.planner import plan_query
from eidetic.store import RecordStore


def _rec(text, dt, scope, i):
    return MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}", text=text,
                        scope=scope, valid_at=dt.timestamp())


def _store_with(tmp_path, rows):
    store = RecordStore(tmp_path / "al.sqlite")
    scope = Scope(namespace="al")
    for i, (text, dt) in enumerate(rows):
        rec = _rec(text, dt, scope, i)
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))
    return store, scope


def _ask(store, scope, q):
    return execute_plan(plan_query(q), q,
                        records=store.active_records_at(scope=scope),
                        claims=store.claims_in_scope(scope))


def _naming(rec):
    return [c for c in claims_for_record(rec) if c.filters.get("naming")]


def test_vocative_prefix_nickname_becomes_claim():
    scope = Scope(namespace="nick")
    rec = _rec("Torvald: Hey Mel! Great hearing from you.\n"
               "Melinda: Thanks! The move went smoothly.",
               datetime(2024, 2, 1, 12, 0), scope, 0)
    claims = [c for c in _naming(rec) if c.filters["naming"] == "nickname"]
    assert len(claims) == 1
    c = claims[0]
    assert c.subject == "Torvald" and c.object == "Mel"
    assert c.filters["target"] == "Melinda"


def test_full_name_address_is_not_a_nickname():
    scope = Scope(namespace="full")
    rec = _rec("Torvald: Hey Melinda! Great hearing from you.\n"
               "Melinda: Thanks!", datetime(2024, 2, 1, 12, 0), scope, 0)
    assert [c for c in _naming(rec) if c.filters["naming"] == "nickname"] == []


def test_quoted_title_after_called_becomes_claim():
    scope = Scope(namespace="title")
    rec = _rec('Gwen: We just did a contemporary piece called "Chasing Dawn." '
               "It was really emotional.", datetime(2024, 3, 1, 12, 0), scope, 0)
    claims = [c for c in _naming(rec) if c.filters["naming"] == "title"]
    assert any(c.object == "Chasing Dawn" and c.filters["named_head"] == "piece"
               for c in claims)


def test_named_category_technique_becomes_claim():
    scope = Scope(namespace="cat")
    rec = _rec("Tomas: I'm using the Cadence Technique - 25 minutes work, 5-minute break.",
               datetime(2024, 3, 5, 12, 0), scope, 0)
    claims = [c for c in _naming(rec) if c.filters["naming"] == "title"]
    assert any(c.object == "Cadence Technique" and c.filters["named_head"] == "technique"
               for c in claims)


def test_nickname_question_answers_from_claim(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Torvald: Hey Mel! Great hearing from you.\nMelinda: Thanks! The move went smoothly.",
         datetime(2024, 2, 1, 12, 0)),
        ("Melinda: I finally unpacked the last box.", datetime(2024, 2, 3, 12, 0)),
    ])
    res = _ask(store, scope, "What nickname does Torvald use for Melinda?")
    assert res is not None and res.answer == "Mel"
    assert res.backend == "claim"


def test_technique_question_answers_from_claim(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Tomas: I'm using the Cadence Technique - 25 minutes work, 5-minute break.",
         datetime(2024, 3, 5, 12, 0)),
        ("Tomas: Exams are close, feeling okay about it.", datetime(2024, 3, 6, 12, 0)),
    ])
    res = _ask(store, scope, "Which popular time management technique does Tomas use?")
    assert res is not None and res.answer == "Cadence Technique"


def test_piece_title_question_answers_from_claim(tmp_path):
    store, scope = _store_with(tmp_path, [
        ('Gwen: We just did a contemporary piece called "Chasing Dawn." It was emotional.',
         datetime(2024, 3, 1, 12, 0)),
    ])
    res = _ask(store, scope, "What kind of dance piece did Gwen's team perform?")
    assert res is not None and "Chasing Dawn" in res.answer


def test_wrong_person_fails_closed(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Tomas: I'm using the Cadence Technique - 25 minutes work, 5-minute break.",
         datetime(2024, 3, 5, 12, 0)),
    ])
    res = _ask(store, scope, "Which time management technique does Priya use?")
    assert res is None or "Cadence" not in (res.answer or "")


def test_when_question_never_answered_with_a_title(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Tomas: I started the Couch Challenge yesterday.", datetime(2024, 3, 5, 12, 0)),
    ])
    res = _ask(store, scope, "When did Tomas start the challenge?")
    assert res is None or "Couch Challenge" != (res.answer or "")
    res2 = _ask(store, scope, "How often does Tomas do his challenge?")
    assert res2 is None or "Couch Challenge" != (res2.answer or "")


def test_interjection_vocative_is_not_a_nickname():
    scope = Scope(namespace="oh")
    rec = _rec("Sam: Oh God, I can't believe Godzilla won the costume contest.\n"
               "Godzilla: It was a good costume!", datetime(2024, 2, 1, 12, 0), scope, 0)
    assert [c for c in _naming(rec) if c.filters["naming"] == "nickname"] == []


def test_nonspeaker_prefix_target_is_not_a_nickname():
    scope = Scope(namespace="nsp")
    rec = _rec("Rob: Hey Mel, the Melbourne trip is on!", datetime(2024, 2, 1, 12, 0),
               scope, 0)
    assert [c for c in _naming(rec) if c.filters["naming"] == "nickname"] == []


def test_third_party_owner_blocks_speaker_person_tie():
    from eidetic.smqe.record_ops import _named_alias_answer
    scope = Scope(namespace="owner")
    rec = _rec("Melanie: My brother started a diet called Carnivore Reset.",
               datetime(2024, 3, 5, 12, 0), scope, 0)
    claims = [c for c in claims_for_record(rec) if c.filters.get("naming")]
    assert claims and claims[0].subject == "brother"
    atoms = [(1.0, c, c.proof_atom) for c in claims]
    answer, _sel = _named_alias_answer("Which diet does Melanie follow?", atoms)
    assert answer == ""


def test_nickname_direction_giver_and_target(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Torvald: Hey Mel! Great hearing from you.\nMelinda: Thanks! All good here.",
         datetime(2024, 2, 1, 12, 0)),
    ])
    res = _ask(store, scope, "What nickname does Melinda use for Torvald?")
    assert res is None or (res.answer or "") != "Mel"
