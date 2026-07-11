"""Runtime accuracy guards mined from the burned-window replay (r1-r10 forensics).

Covers, with SYNTHETIC fixtures only (no benchmark strings):
  1. clean-fact shapes 3-5: dangling separator tails, degenerate conjunction repetition,
     and junk-stripped question echoes;
  2. `structured_answer_form_floor` -- the pure floor gauntlet with named rejections and
     its op/note carve-outs;
  3. the preference category-object anchor (a 'genre/kind/type of X' question cannot be
     answered from evidence about a different object class);
  4. the relative_temporal ambiguity guard: explicitly STATED dates outrank deictic
     session-derived resolutions on evidence ties, and irreconcilably conflicting deictic
     ties fail closed;
  5. `_answer_period_key` / `_periods_conflict` primitives.

Non-negotiable: every floor must PASS real answers -- each guard has keep-cases proving
correct shapes never trip.
"""
from __future__ import annotations

from datetime import datetime

from eidetic.models import (MemoryRecord, NLILabel, Scope, StructuredAnswerResult,
                            StructuredSupport)
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.smqe.record_ops import _answer_period_key, _periods_conflict
from eidetic.smqe.verify import (_category_object_anchored, _clean_fact_form_credible,
                                 answer_from_result, structured_answer_form_floor)
from eidetic.store import RecordStore


class _Retriever:
    def __init__(self, store):
        self.store = store

    def verify_citation(self, rec, atom):
        return (
            (NLILabel.ENTAILMENT, 1.0)
            if " ".join(atom.lower().split()) in " ".join((rec.text or "").lower().split())
            else (NLILabel.NEUTRAL, 0.0)
        )


def _record(text: str, *, scope: Scope, valid_at: float = 1_700_000_000.0) -> MemoryRecord:
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash="h-" + str(abs(hash(text))),
        raw_uri="mem://synthetic",
    )


def _r(answer: str, op: str = "latest_value", note: str = "") -> StructuredAnswerResult:
    return StructuredAnswerResult(answer=answer, op=op, backend="claim", note=note,
                                  supports=[])


# --- clean-fact shape 3: dangling separator tail ------------------------------------------
def test_rejects_dangling_comma_tail():
    assert _clean_fact_form_credible(
        "What pastimes were shared with the club?",
        _r("last month, garden, painting,")) is False


def test_rejects_dangling_colon_tail():
    assert _clean_fact_form_credible("What was announced?", _r("The plan was:")) is False


def test_keeps_complete_list_without_dangling_tail():
    assert _clean_fact_form_credible(
        "What pastimes were shared with the club?",
        _r("gardening, painting, and a picnic")) is True


# --- clean-fact shape 4: degenerate conjunction repetition --------------------------------
def test_rejects_degenerate_conjunction_repetition():
    assert _clean_fact_form_credible(
        "What projects interest her?",
        _r("involved and involved with organizations")) is False


def test_keeps_legitimate_repeated_common_noun_across_items():
    assert _clean_fact_form_credible(
        "Which shops?", _r("Cedar corner shop and Harbor corner shop")) is True


# --- clean-fact shape 5: junk-stripped question echo ---------------------------------------
def test_rejects_junk_plus_name_echo():
    # 'Yeah, <Name>' where the name is the question's own subject answers nothing.
    assert _clean_fact_form_credible(
        "What did Priya take part in at the fair?", _r("Yeah, Priya")) is False


def test_rejects_greeting_plus_name_echo():
    assert _clean_fact_form_credible(
        "Which hobby did Ravi pick back up in winter?", _r("Hey Ravi")) is False


def test_rejects_all_junk_answer():
    assert _clean_fact_form_credible("What plans were made?", _r("Check")) is False


def test_keeps_junk_headed_answer_with_real_new_content():
    # A junk head with genuinely novel content after stripping is a real answer.
    assert _clean_fact_form_credible(
        "Where did they move?", _r("Well, a lakeside cabin in Vermont")) is True


def test_keeps_bare_number_answers_unevaluable():
    # No content tokens at all: stays fail-open exactly like the raw echo floor.
    assert _clean_fact_form_credible("What time works?", _r("11 pm")) is True


# --- structured_answer_form_floor: named floors and carve-outs -----------------------------
def test_floor_names_dangling_tail_as_clean_fact():
    assert structured_answer_form_floor(
        "What pastimes were shared?", _r("last month, garden, painting,")) == "clean_fact"


def test_floor_passes_computed_op_bare_value():
    assert structured_answer_form_floor("How many?", _r("3", op="count_aggregate")) is None


def test_floor_keeps_suggestion_synth_carve_out():
    # Provenance-gated suggestion fragments keep their deliberate carve-out.
    assert structured_answer_form_floor(
        "What did Priya recommend watching?",
        _r("maybe watch one", op="preference_synth",
           note="smqe:preference_synth:claim:suggestion_synth")) is None


def test_floor_rejects_when_question_without_temporal_token():
    assert structured_answer_form_floor(
        "When did the workshop take place?", _r("a pottery workshop")) == "when_type"


def test_floor_passes_when_question_with_temporal_token():
    assert structured_answer_form_floor(
        "When did the workshop take place?", _r("March 2021")) is None


# --- preference category-object anchor ------------------------------------------------------
def test_category_object_rejects_unrelated_object_class():
    supports = [StructuredSupport(memory_id="m1",
                                  proof_atom="That plum galette was one of my favorites.")]
    assert _category_object_anchored(
        "What is Nadia's favorite genre of films?", "plum galette", supports) is False


def test_category_object_keeps_answer_anchored_in_atom():
    supports = [StructuredSupport(memory_id="m1",
                                  proof_atom="I mostly watch sci-fi films, my favorites.")]
    assert _category_object_anchored(
        "What is Nadia's favorite genre of films?", "sci-fi", supports) is True


def test_category_object_keeps_answer_anchored_in_answer_text():
    assert _category_object_anchored(
        "What is her favorite kind of music?", "folk music", []) is True


def test_non_category_questions_are_untouched():
    assert _category_object_anchored(
        "What is Nadia's favorite dessert?", "plum galette", []) is True


def test_answer_from_result_enforces_category_anchor(tmp_path):
    store = RecordStore(tmp_path / "pref-anchor.sqlite")
    scope = Scope(namespace="pref-anchor")
    rec = _record("Nadia: That plum galette was one of my favorites.", scope=scope)
    store.upsert_record(rec)
    result = StructuredAnswerResult(
        answer="plum galette",
        op="preference_synth",
        backend="claim",
        supports=[StructuredSupport(
            memory_id=rec.memory_id,
            proof_atom="That plum galette was one of my favorites.")],
        note="smqe:preference_synth:claim",
    )
    ans = answer_from_result(
        _Retriever(store), "What is Nadia's favorite genre of films?", result, verify=True)
    assert ans is None


# --- relative_temporal ambiguity guard ------------------------------------------------------
def test_stated_year_outranks_deictic_tie(tmp_path):
    """Two atoms tie on the event terms; one states the year explicitly, the other is a
    deictic mention resolved against a much later session. The stated year must win."""
    store = RecordStore(tmp_path / "stated-vs-deictic.sqlite")
    scope = Scope(namespace="stated-vs-deictic")
    rows = [
        ("Deb: I visited Osaka back in 2019, unforgettable trip.",
         datetime(2023, 9, 10, 12, 0)),
        ("Deb: Talking about my visit to Osaka today brings it all back.",
         datetime(2023, 9, 12, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store), "When did Deb visit Osaka?",
        at=datetime(2023, 12, 1, 12, 0).timestamp(), scope=scope)

    assert ans is not None
    assert "2019" in ans.answer
    assert "2023" not in ans.answer


def test_distinct_deictic_mentions_fail_closed_as_mention_selected(tmp_path):
    """Two DIFFERENT dated mentions of the same activity months apart: nothing proves which
    occurrence the question asks about, so the selection ships tagged :mention_selected and
    the note-keyed floor fails it closed (the measured 57%-VW legacy class)."""
    store = RecordStore(tmp_path / "deictic-distinct.sqlite")
    scope = Scope(namespace="deictic-distinct")
    rows = [
        ("Rui: Went kayaking on the fjord today, what a rush!",
         datetime(2023, 3, 4, 12, 0)),
        ("Rui: Went kayaking on the fjord today, once more!",
         datetime(2023, 10, 21, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store), "When did Rui go kayaking on the fjord?",
        at=datetime(2023, 12, 1, 12, 0).timestamp(), scope=scope)

    assert ans is None or ans.verified is False


def test_identical_reassertion_keeps_latest_instance_convention(tmp_path):
    """The SAME sentence re-asserted at a later session is a knowledge-update refresh, not a
    contested selection: the latest resolution answers, verified, tagged atom_derived (the
    bench time-invariant locks this exact shape)."""
    store = RecordStore(tmp_path / "identical-refresh.sqlite")
    scope = Scope(namespace="identical-refresh")
    rows = [
        ("User: Yesterday I picked up the garden permit 7.", datetime(2024, 2, 10, 12, 0)),
        ("User: Yesterday I picked up the garden permit 7.", datetime(2024, 3, 15, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store), "When did I pick up the garden permit 7?",
        at=datetime(2024, 3, 16, 12, 0).timestamp(), scope=scope)

    assert ans is not None
    assert ans.answer == "2024-03-14"
    assert ans.note.endswith(":atom_derived")


def test_week_phrase_statement_never_overrides_deictic_day(tmp_path):
    """Only an explicit bare-YEAR statement may reassign a tied contest; week/month PHRASES
    must not (reassigning to a phrase shipped a wrong week window over the gold exact day
    on the 2026-07-11 selection replay). Whatever ships here, it is never the phrase's
    period presented as the event date over the deictic day."""
    store = RecordStore(tmp_path / "phrase-tie.sqlite")
    scope = Scope(namespace="phrase-tie")
    rows = [
        ("Lena: Went to a pottery class with a group of neighbors yesterday, loved it!",
         datetime(2023, 7, 22, 12, 0)),
        ("Lena: The first week of July I joined a pottery class with a group of neighbors.",
         datetime(2023, 7, 30, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store), "When did Lena go to a pottery class with neighbors?",
        at=datetime(2023, 8, 15, 12, 0).timestamp(), scope=scope)

    if ans is not None and ans.verified:
        assert "2023-07-21" in ans.answer
        assert "week of" not in ans.answer


def test_temporal_selection_floor_is_note_keyed():
    assert structured_answer_form_floor(
        "When did she go hiking?",
        _r("2023-05-06", op="relative_temporal",
           note="smqe:relative_temporal:claim:mention_selected")) == "temporal_selection"
    # atom_derived and untagged legacy notes are untouched.
    assert structured_answer_form_floor(
        "When did she go hiking?",
        _r("2023-05-06", op="relative_temporal",
           note="smqe:relative_temporal:claim:atom_derived")) is None
    assert structured_answer_form_floor(
        "When did she go hiking?",
        _r("2023-05-06", op="relative_temporal",
           note="smqe:relative_temporal:claim")) is None


def test_conflicting_stated_years_fail_closed(tmp_path):
    """Two explicit date STATEMENTS that name materially different periods for the same event
    are a genuine evidence contradiction -- no verified answer may ship from this operator."""
    store = RecordStore(tmp_path / "stated-conflict.sqlite")
    scope = Scope(namespace="stated-conflict")
    rows = [
        ("Deb: I visited Osaka back in 2019, unforgettable trip.",
         datetime(2023, 9, 10, 12, 0)),
        ("Deb: My visit to Osaka was back in 2016, I am fairly sure.",
         datetime(2023, 9, 12, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store), "When did Deb visit Osaka?",
        at=datetime(2023, 12, 1, 12, 0).timestamp(), scope=scope)

    assert ans is None or ans.verified is False


def test_single_unambiguous_deictic_still_answers(tmp_path):
    """One clean deictic mention: the guard must not disturb the ordinary resolution."""
    store = RecordStore(tmp_path / "single-deictic.sqlite")
    scope = Scope(namespace="single-deictic")
    rec = _record("Ines: I repotted the fig tree today, finally.",
                  scope=scope, valid_at=datetime(2023, 5, 6, 12, 0).timestamp())
    store.upsert_record(rec)
    store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store), "When did Ines repot the fig tree?",
        at=datetime(2023, 8, 1, 12, 0).timestamp(), scope=scope)

    assert ans is not None
    assert "2023-05-06" in ans.answer


# --- period primitives ----------------------------------------------------------------------
def test_answer_period_key_parses_iso_month_and_year():
    assert _answer_period_key("2023-09-10") == (2023, 9, 10)
    assert _answer_period_key("September 2023") == (2023, 9, None)
    assert _answer_period_key("2020") == (2020, None, None)
    assert _answer_period_key("the weekend of 2023-09-09 to 2023-09-10") == (2023, 9, 9)
    assert _answer_period_key("no date here") is None


def test_periods_conflict_semantics():
    assert _periods_conflict((2020, None, None), (2023, 9, 10)) is True
    assert _periods_conflict((2023, 9, None), (2023, 10, None)) is True
    assert _periods_conflict((2023, 9, 1), (2023, 9, 24)) is True
    assert _periods_conflict((2023, 9, 9), (2023, 9, 10)) is False
    assert _periods_conflict((2020, None, None), (2020, 3, 5)) is False
