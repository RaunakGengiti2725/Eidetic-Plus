"""Event-date claim family: a claim is emitted ONLY when the source sentence itself
carries a date expression tied to the event verb/noun -- the structural fix for
mention-time-vs-event-time and ambient-year selection, with no read-time regex rescue.

All dialogues/fixtures are FABRICATED shapes; no benchmark rows, holdout text, or
holdout entity combos.
"""
from __future__ import annotations

from datetime import datetime

from eidetic.models import ClaimRecord, MemoryRecord, Scope
from eidetic.smqe import event_identity as ei
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.smqe.executor import execute_plan
from eidetic.smqe.planner import plan_query
from eidetic.smqe.record_ops import _event_date_claim_answer, _support
from eidetic.store import RecordStore


def _rec(text, dt, scope=None, i=0):
    return MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}", text=text,
                        scope=scope or Scope(namespace="ed"), valid_at=dt.timestamp())


def _dated(rec):
    return [c for c in claims_for_record(rec) if c.filters.get("event") == "dated"]


def _store_with(tmp_path, rows):
    store = RecordStore(tmp_path / "ed.sqlite")
    scope = Scope(namespace="ed")
    for i, (text, dt) in enumerate(rows):
        rec = _rec(text, dt, scope, i)
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))
    return store, scope


def _ask(store, scope, q):
    return execute_plan(plan_query(q), q,
                        records=store.active_records_at(scope=scope),
                        claims=store.claims_in_scope(scope))


# ---------------------------------------------------------------- write side


def test_explicit_full_date_emits_dated_claim():
    rec = _rec("Dana: I adopted Biscuit on March 3, 2024.", datetime(2024, 3, 10, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    f = claims[0].filters
    assert f["lemma"] == "adopt"
    assert f["obj_head"] == "biscuit"
    assert f["event_date"] == "2024-03-03"
    assert f["date_precision"] == ei.PRECISION_EXPLICIT
    assert claims[0].claim_type == "event"


def test_relative_day_resolves_against_valid_at():
    rec = _rec("Dana: We moved into the new apartment yesterday.",
               datetime(2024, 5, 10, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    f = claims[0].filters
    assert f["event_date"] == "2024-05-09"
    assert f["date_precision"] == ei.PRECISION_RELATIVE_DAY
    assert f["lemma"] == "move"
    assert f["obj_head"] == "apartment"


def test_window_phrase_month_anchor_and_date_phrase():
    rec = _rec("Mina: My concert in Osaka was during the last week of May.",
               datetime(2023, 6, 2, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    f = claims[0].filters
    assert f["date_precision"] == ei.PRECISION_WINDOW
    assert f["event_date"].startswith("2023-05")
    assert "last week" in f.get("date_phrase", "").lower()
    assert f["lemma"] == "concert"
    assert f.get("place") == "Osaka"
    assert claims[0].subject == "Mina"


def test_event_noun_third_party_owner_subject():
    rec = _rec("Dana: Rosa's wedding was on June 10, 2023.", datetime(2023, 6, 20, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    assert claims[0].subject == "Rosa"
    f = claims[0].filters
    assert f["lemma"] == "wedding"
    assert f["event_date"] == "2023-06-10"
    assert f["date_precision"] == ei.PRECISION_EXPLICIT


def test_no_in_atom_date_no_claim():
    rec = _rec("Dana: I performed in Osaka.", datetime(2023, 6, 2, 12, 0))
    assert _dated(rec) == []


def test_question_atom_emits_nothing():
    rec = _rec("Dana: When did you adopt Biscuit?", datetime(2024, 3, 10, 12, 0))
    assert _dated(rec) == []


def test_negated_event_emits_nothing():
    rec = _rec("Dana: We never made it to the show on May 5, 2024.",
               datetime(2024, 5, 8, 12, 0))
    assert _dated(rec) == []


def test_future_plan_polarity_emits_nothing():
    rec = _rec("Dana: I'm hoping to start the new job next month.",
               datetime(2024, 5, 8, 12, 0))
    assert _dated(rec) == []


def test_ambient_year_in_relative_clause_never_selected():
    rec = _rec("Dana: I adopted Biscuit, who was born in 2019, on March 3, 2024.",
               datetime(2024, 3, 10, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    assert claims[0].filters["event_date"] == "2024-03-03"
    assert not any("2019" in str(c.filters.get("event_date")) for c in claims)


def test_two_dated_events_one_sentence_both_survive_dedup():
    rec = _rec("Dana: I adopted Biscuit on March 3, 2024, and I married Sam "
               "on June 10, 2023.", datetime(2024, 3, 10, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 2
    got = {(c.filters["lemma"], c.filters["event_date"]) for c in claims}
    assert got == {("adopt", "2024-03-03"), ("marry", "2023-06-10")}


def test_p2_composition_keeps_family_filters_intact():
    """Once-ish verb with explicit date: exactly one family claim after full
    claims_for_record dedup, and _tag_event_identity (setdefault) never overwrites
    the family-set filter values."""
    rec = _rec("Dana: I married Sam on June 10, 2023.", datetime(2023, 6, 20, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    f = claims[0].filters
    assert f["lemma"] == "marry"
    assert f["obj_head"] == "sam"
    assert f["event_date"] == "2023-06-10"
    assert f["date_precision"] == ei.PRECISION_EXPLICIT


# ----------------------------------------------------------------- read side


def _sup(item, atom, score):
    return _support(item.source_memory_id, atom, claim_id=item.claim_id, score=score)


def _dated_claim(lemma, obj, head, iso, prec, place="", subject="Dana",
                 atom="", mem="m1", valid_at=1_700_000_000.0):
    filters = {"event": "dated", "lemma": lemma, "obj_head": head,
               "event_date": iso, "date_precision": prec}
    if place:
        filters["place"] = place
    return ClaimRecord(claim_type="event", subject=subject, predicate=lemma,
                       object=obj, value=iso, filters=filters, valid_at=valid_at,
                       source_memory_id=mem, proof_atom=atom or f"{subject}: {obj} event.")


def _call(q, claims):
    atoms = [(1.0, c, c.proof_atom) for c in claims]
    return _event_date_claim_answer(plan_query(q), q, atoms, "claim", _sup)


def test_read_answers_iso_with_event_date_note():
    # NOTE: uses a non-once-ish dated verb (join); once-ish lemmas (start/open family)
    # are owned by :event_instance upstream by design.
    c = _dated_claim("join", "bakery", "bakery", "2024-04-01", ei.PRECISION_EXPLICIT,
                     atom="Dana: I joined the bakery on April 1, 2024.")
    res = _call("When did I join the bakery?", [c])
    assert res is not None and res.answer == "2024-04-01"
    assert res.note.endswith(":event_date")
    assert res.supports[0].claim_id == c.claim_id


def test_multi_instance_span_guard_declines():
    a = _dated_claim("concert", "concert", "concert", "2023-05-25", ei.PRECISION_WINDOW)
    b = _dated_claim("concert", "concert", "concert", "2023-11-02", ei.PRECISION_EXPLICIT)
    assert _call("When was my concert?", [a, b]) is None


def test_place_disambiguates_between_instances():
    a = _dated_claim("concert", "concert", "concert", "2023-05-25",
                     ei.PRECISION_WINDOW, place="Osaka")
    b = _dated_claim("concert", "concert", "concert", "2023-11-02",
                     ei.PRECISION_EXPLICIT, place="Berlin")
    res = _call("When was my concert in Berlin?", [a, b])
    assert res is not None and res.answer == "2023-11-02"


def test_no_object_or_place_tie_declines():
    c = _dated_claim("attend", "gala", "gala", "2024-02-10", ei.PRECISION_EXPLICIT)
    assert _call("When did I attend the ceremony?", [c]) is None


def test_window_claim_never_renders_a_day():
    c = _dated_claim("concert", "concert", "concert", "2023-05-25", ei.PRECISION_WINDOW)
    res = _call("When was my concert?", [c])
    assert res is not None
    assert res.answer == "May 2023"
    assert "2023-05-25" not in res.answer


def test_onceish_question_owned_by_event_instance(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Dana: I adopted Biscuit on March 3, 2024.", datetime(2024, 3, 10, 12, 0)),
    ])
    res = _ask(store, scope, "When did I adopt Biscuit?")
    assert res is not None and res.answer == "2024-03-03"
    assert ":event_instance" in res.note
    assert ":event_date" not in res.note


def test_inverse_direction_wh_shape_never_enters():
    c = _dated_claim("visit", "Kyoto", "kyoto", "2023-05-05", ei.PRECISION_EXPLICIT)
    assert _call("Which country was Milo visiting in May 2023?", [c]) is None


def test_end_to_end_verify_anchor_short_circuit(tmp_path):
    from eidetic.smqe.verify import answer_from_result

    class _NoNLIRetriever:
        def __init__(self, s):
            self.store = s

        def verify(self, *a, **k):  # presence enables the strict claim path
            raise AssertionError("retriever.verify must not be called")

        def verify_citation(self, rec, atom):
            raise AssertionError("NLI must not be called: verbatim anchor expected")

    store, scope = _store_with(tmp_path, [
        ("Dana: I joined the bakery on April 1, 2024.", datetime(2024, 4, 3, 12, 0)),
        ("Dana: The sourdough class was fun.", datetime(2024, 4, 5, 12, 0)),
    ])
    q = "When did I join the bakery?"
    res = _ask(store, scope, q)
    assert res is not None and res.answer == "2024-04-01"
    assert res.note.endswith(":event_date") and res.backend == "claim"
    ans = answer_from_result(_NoNLIRetriever(store), q, res, verify=True)
    assert ans is not None and ans.verified


# ------------------------------------------------- adversarial-review regressions


def test_segment_date_never_bleeds_to_other_verb():
    """One clause-splittable sentence, one relative date, two dated verbs: the adoption
    must NOT inherit the vet visit's 'yesterday'."""
    rec = _rec("Dana: I adopted a puppy and yesterday we visited the vet.",
               datetime(2024, 5, 10, 12, 0))
    claims = _dated(rec)
    assert not any(c.filters["lemma"] == "adopt" for c in claims)
    visits = [c for c in claims if c.filters["lemma"] == "visit"]
    assert visits and visits[0].filters["event_date"] == "2024-05-09"


def test_unsplittable_clause_dates_nearest_verb_only():
    rec = _rec("Dana: I adopted a puppy and visited the vet yesterday.",
               datetime(2024, 5, 10, 12, 0))
    claims = _dated(rec)
    assert not any(c.filters["lemma"] == "adopt" for c in claims)


def test_supposed_to_be_date_never_crystallizes():
    """'was supposed to be in June 2020' is a plan the next clause cancels: no claim
    from either segment (the postponement is negation-guarded; fail closed)."""
    rec = _rec("Dana: My wedding was supposed to be in June 2020 but we postponed "
               "it to May 2021.", datetime(2021, 6, 1, 12, 0))
    assert _dated(rec) == []


def test_restrictive_relative_clause_date_never_dates_main_verb():
    rec = _rec("Dana: I adopted a kitten that was born in March 2023.",
               datetime(2023, 8, 1, 12, 0))
    assert _dated(rec) == []


def test_contact_clause_and_bare_who_clause_dates_excised():
    a = _rec("Dana: I met Sara at the bakery she opened in March 2021.",
             datetime(2023, 8, 1, 12, 0))
    b = _rec("Dana: I visited my cousin who moved to Denver in March 2021.",
             datetime(2023, 8, 1, 12, 0), i=1)
    assert _dated(a) == []
    assert _dated(b) == []


def test_duration_in_n_weeks_is_not_a_future_event_date():
    """'moved the whole house in two weeks' states DURATION; a past-surface verb with a
    strictly-future resolved date is self-contradictory and must emit nothing."""
    rec = _rec("Dana: I moved the whole house in two weeks.",
               datetime(2024, 5, 10, 12, 0))
    assert _dated(rec) == []


def test_present_tense_week_of_month_fails_closed():
    """'is the first week of March' reads as upcoming; the past-only year inference
    would date it a year early -- decline instead."""
    rec = _rec("Dana: My graduation ceremony is the first week of March.",
               datetime(2024, 2, 5, 12, 0))
    assert _dated(rec) == []


def test_quoted_reported_speech_never_becomes_first_person_claim():
    rec = _rec('Dana: Rosa told me "I adopted a kitten on March 3, 2024" at lunch.',
               datetime(2024, 3, 5, 12, 0))
    assert _dated(rec) == []


def test_last_week_straddling_month_boundary_emits_nothing():
    """valid_at July 3: 'last week' covers Jun 26-Jul 2 -- no honest single-month anchor,
    so the claim path declines and the legacy path keeps the verbatim range."""
    rec = _rec("Dana: I adopted Biscuit last week.", datetime(2026, 7, 3, 12, 0))
    assert _dated(rec) == []


def test_last_week_within_one_month_keeps_window_anchor():
    rec = _rec("Dana: I adopted Biscuit last week.", datetime(2026, 7, 10, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    f = claims[0].filters
    assert f["event_date"] == "2026-07-03"
    assert f["date_precision"] == ei.PRECISION_WINDOW


def test_year_granularity_relative_emits_nothing():
    rec = _rec("Dana: We adopted Biscuit a year ago.", datetime(2026, 7, 4, 12, 0))
    assert _dated(rec) == []


def test_week_and_month_granularity_relatives_are_window_precision():
    """'six months ago' / 'three weeks ago' must never carry PRECISION_RELATIVE_DAY
    (a fabricated exact day); they anchor at WINDOW so only the month renders."""
    a = _rec("Dana: We adopted Biscuit six months ago.", datetime(2026, 7, 4, 12, 0))
    b = _rec("Dana: We adopted Biscuit three weeks ago.", datetime(2026, 7, 4, 12, 0), i=1)
    ca, cb = _dated(a), _dated(b)
    assert len(ca) == 1 and ca[0].filters["date_precision"] == ei.PRECISION_WINDOW
    assert ca[0].filters["event_date"] == "2026-01-04"
    assert ca[0].filters["obj_head"] == "biscuit"
    assert len(cb) == 1 and cb[0].filters["date_precision"] == ei.PRECISION_WINDOW
    assert cb[0].filters["event_date"] == "2026-06-13"


def test_may_june_july_participate_in_week_of_month_branch():
    """The month alternation must include ALL full month names: 'last week of May'
    previously fell through to the generic 'last week' branch and anchored to the
    session month (July)."""
    rec = _rec("Dana: My concert was during the last week of May.",
               datetime(2023, 7, 20, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    f = claims[0].filters
    assert f["event_date"] == "2023-05-25"
    assert "last week of may" in f.get("date_phrase", "").lower()


def test_month_name_never_captured_as_place():
    rec = _rec("Dana: I visited Kyoto in June 2024.", datetime(2024, 7, 1, 12, 0))
    claims = _dated(rec)
    assert len(claims) == 1
    assert claims[0].filters.get("place", "") != "June"


def test_empty_object_head_emits_nothing():
    """'We opened IT on June 1' is unanchorable: empty obj_head auto-passes object ties
    downstream and hijacked unrelated instances."""
    rec = _rec("Rosa: We opened it on June 1, 2024.", datetime(2024, 6, 14, 12, 0))
    assert _dated(rec) == []


def test_dated_claim_never_hijacks_event_instance_pool(tmp_path):
    """Pre-change behavior restored: the honest 'last Saturday' bakery claim answers,
    not the empty-head 'We opened it on June 1' contaminant."""
    store, scope = _store_with(tmp_path, [
        ("Rosa: Our bakery opened last Saturday.", datetime(2024, 6, 13, 12, 0)),
        ("Rosa: We opened it on June 1, 2024.", datetime(2024, 6, 14, 12, 0)),
    ])
    res = _ask(store, scope, "When did the bakery open?")
    assert res is not None and res.answer == "2024-06-08"


def test_owner_named_question_never_serves_speakers_own_event(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Dana: My wedding was on June 5, 2021.", datetime(2021, 7, 1, 12, 0)),
        ("Dana: Mina's wedding was on June 19, 2021.", datetime(2021, 7, 2, 12, 0)),
    ])
    res = _ask(store, scope, "When was Mina's wedding?")
    assert res is not None and res.answer == "2021-06-19"
    my = _ask(store, scope, "When was my wedding?")
    assert my is not None and my.answer == "2021-06-05"


def test_later_correction_beats_earlier_date(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Dana: I joined the bakery on April 1, 2024.", datetime(2024, 4, 2, 12, 0)),
        ("Dana: Correction: I joined the bakery on April 9, 2024.",
         datetime(2024, 4, 10, 12, 0)),
    ])
    res = _ask(store, scope, "When did I join the bakery?")
    assert res is not None and res.answer == "2024-04-09"


def test_not_old_date_correction_still_writes_a_claim(tmp_path):
    """'..., not April 1' corrects the DATE; it must not trip the negation guard, and
    the corrected claim must win."""
    store, scope = _store_with(tmp_path, [
        ("Dana: I joined the bakery on April 1, 2024.", datetime(2024, 4, 2, 12, 0)),
        ("Dana: Actually, I joined the bakery on April 9, 2024, not April 1.",
         datetime(2024, 4, 10, 12, 0)),
    ])
    res = _ask(store, scope, "When did I join the bakery?")
    assert res is not None and res.answer == "2024-04-09"


def test_ordinal_question_bails_out_of_event_date_op():
    """'my FIRST concert' has counting semantics: the op must decline so the ordinal
    machinery owns the shape, instead of serving a later instance at 0.9."""
    c = _dated_claim("concert", "concert", "concert", "2024-05-05",
                     ei.PRECISION_EXPLICIT)
    assert _call("When was my first concert?", [c]) is None


def test_tier_count_never_counts_dated_claims():
    from eidetic.smqe.record_ops import _claim_tier_count_answer
    a = _dated_claim("visit", "Modern Art Museum", "museum", "2024-03-03",
                     ei.PRECISION_EXPLICIT,
                     atom="Dana: I visited the Modern Art Museum on March 3, 2024.")
    b = _dated_claim("visit", "Science Museum", "museum", "2024-04-02",
                     ei.PRECISION_EXPLICIT,
                     atom="Dana: I visited the Science Museum on April 2, 2024.")
    a.predicate = b.predicate = "visited"
    q = "How many museums did I visit?"
    atoms = [(1.0, a, a.proof_atom), (1.0, b, b.proof_atom)]
    assert _claim_tier_count_answer(plan_query(q), q, atoms, "claim", _sup) is None


def test_count_question_not_inflated_by_dated_family(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Dana: I visited the Modern Art Museum with Rosa on March 3, 2024.",
         datetime(2024, 3, 4, 12, 0)),
        ("Dana: I visited the Science Museum on April 2, 2024.",
         datetime(2024, 4, 3, 12, 0)),
    ])
    res = _ask(store, scope, "How many museums did I visit?")
    assert res is not None and res.answer == "2"


def test_enumeration_never_selects_event_dated_claims():
    from eidetic.smqe.qa_ops import _claim_enumeration_answer
    a = _dated_claim("visit", "Kyoto", "kyoto", "2023-05-05", ei.PRECISION_EXPLICIT,
                     atom="Dana: I visited Kyoto on May 5, 2023.")
    b = _dated_claim("visit", "Oslo", "oslo", "2023-07-09", ei.PRECISION_EXPLICIT,
                     atom="Dana: I visited Oslo on July 9, 2023.")
    a.predicate = b.predicate = "visited"
    atoms = [(1.0, a, a.proof_atom), (1.0, b, b.proof_atom)]
    answer, selected = _claim_enumeration_answer("What cities has Dana visited?", atoms)
    assert answer == "" and selected == []
