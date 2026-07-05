"""P1b itemized-list claims: one claim per listed item under a shared list_id, and the
existence-count path over them ('how many blockers' from 'blockers: A, B, C' -- no action
verb, the exact shape that NO-GO'd deletion tranche 2 twice).

All dialogues are FABRICATED shapes; no benchmark rows.
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
    store = RecordStore(tmp_path / "il.sqlite")
    scope = Scope(namespace="il")
    for i, (text, dt) in enumerate(rows):
        rec = _rec(text, dt, scope, i)
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))
    return store, scope


def _ask(store, scope, q):
    return execute_plan(plan_query(q), q,
                        records=store.active_records_at(scope=scope),
                        claims=store.claims_in_scope(scope))


def _list_claims(rec):
    return [c for c in claims_for_record(rec) if c.filters.get("list") == "item"]


def test_copular_list_emits_one_claim_per_item_with_shared_list_id():
    scope = Scope(namespace="cop")
    rec = _rec("Dara: My release blockers are the API migration, the load test, and the sign-off.",
               datetime(2024, 3, 2, 12, 0), scope, 0)
    claims = _list_claims(rec)
    objs = {c.object for c in claims}
    assert objs == {"API migration", "load test", "sign-off"}
    assert len({c.filters["list_id"] for c in claims}) == 1
    assert all(c.filters["list_label"] == "release blockers" for c in claims)
    assert all(c.filters["list_size"] == 3 for c in claims)
    assert all(c.subject == "Dara" for c in claims)


def test_like_list_labels_items_with_head_noun():
    scope = Scope(namespace="like")
    rec = _rec("Priya: They can do tricks like flip, spin, twirl, and bow.",
               datetime(2024, 4, 1, 12, 0), scope, 0)
    claims = _list_claims(rec)
    assert {c.object for c in claims} == {"flip", "spin", "twirl", "bow"}
    assert all(c.filters["list_label"] == "tricks" for c in claims)


def test_enjoy_verb_list_keeps_verb_predicate():
    scope = Scope(namespace="verb")
    rec = _rec("Sena: I really enjoy trail running, cold brew, and jigsaw puzzles.",
               datetime(2024, 5, 6, 12, 0), scope, 0)
    claims = _list_claims(rec)
    assert {c.object for c in claims} == {"trail running", "cold brew", "jigsaw puzzles"}
    assert all(c.predicate == "enjoy" for c in claims)


def test_gerund_pair_without_comma_is_captured():
    scope = Scope(namespace="pair")
    rec = _rec("Wyn: We both enjoy watching documentaries and baking bread.",
               datetime(2024, 5, 7, 12, 0), scope, 0)
    claims = _list_claims(rec)
    assert {c.object for c in claims} == {"watching documentaries", "baking bread"}


def test_pronoun_chatter_and_questions_emit_no_list_claims():
    scope = Scope(namespace="junk")
    rec = _rec("Ivo: I like you and the weather. Kel: What hobbies do you have, Ivo?",
               datetime(2024, 5, 8, 12, 0), scope, 0)
    assert _list_claims(rec) == []


def test_dedupe_keeps_every_item_from_one_proof_sentence():
    scope = Scope(namespace="dd")
    rec = _rec("Dara: My release blockers are the API migration, the load test, and the sign-off.",
               datetime(2024, 3, 2, 12, 0), scope, 0)
    claims = claims_for_record(rec)
    assert len([c for c in claims if c.filters.get("list") == "item"]) == 3


def test_existence_count_answers_from_list_claims(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Dara: My release blockers are the API migration, the load test, and the sign-off.",
         datetime(2024, 3, 2, 12, 0)),
        ("Dara: The weather is lovely today.", datetime(2024, 3, 3, 12, 0)),
    ])
    res = _ask(store, scope, "How many release blockers are there?")
    assert res is not None
    assert res.answer.startswith("3 release blockers")
    assert ":claim_list_count" in res.note
    assert len(res.supports) == 3


def test_latest_list_wins_the_existence_count(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Dara: My release blockers are the API migration, the load test, and the sign-off.",
         datetime(2024, 3, 2, 12, 0)),
        ("Dara: My release blockers are the load test and the sign-off.",
         datetime(2024, 4, 2, 12, 0)),
    ])
    res = _ask(store, scope, "How many release blockers are there?")
    assert res is not None
    assert res.answer.startswith("2 release blockers")


def test_enumeration_composes_from_list_label_claims(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Priya: They can do tricks like flip, spin, twirl, and bow.",
         datetime(2024, 4, 1, 12, 0)),
        ("Priya: The recital went fine.", datetime(2024, 4, 2, 12, 0)),
    ])
    res = _ask(store, scope, "What tricks do Priya's parrots know?")
    assert res is not None
    for item in ("flip", "spin", "twirl", "bow"):
        assert item in res.answer


def test_single_item_never_becomes_a_list():
    scope = Scope(namespace="one")
    rec = _rec("Dara: My release blockers are the API migration.",
               datetime(2024, 3, 2, 12, 0), scope, 0)
    assert _list_claims(rec) == []


def test_third_person_lists_are_not_speaker_claims():
    scope = Scope(namespace="3p")
    rec = _rec("Anna: He enjoys darts, snooker and poker.",
               datetime(2024, 5, 8, 12, 0), scope, 0)
    assert _list_claims(rec) == []


def test_predicate_adjective_soup_is_not_a_list():
    scope = Scope(namespace="soup")
    rec = _rec("Marco: The activities are canceled, refunded and rescheduled. "
               "Nina: My symptoms are gone, thankfully, and finally.",
               datetime(2024, 5, 8, 12, 0), scope, 0)
    assert _list_claims(rec) == []


def test_clause_boundary_stops_verb_list_capture():
    scope = Scope(namespace="clause")
    rec = _rec("Noor: I love hiking and swimming, and I tried judo, karate, and aikido.",
               datetime(2024, 5, 8, 12, 0), scope, 0)
    claims = _list_claims(rec)
    love = {c.object for c in claims if c.predicate == "love"}
    tried = {c.object for c in claims if c.predicate == "tried"}
    assert love == {"hiking", "swimming"}
    assert tried == {"judo", "karate", "aikido"}
    assert not any("judo" in o for o in love)


def test_social_media_likes_count_not_answered_with_foods(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Marco: I like pizza, sushi and ramen.", datetime(2024, 5, 8, 12, 0)),
    ])
    res = _ask(store, scope, "How many likes did my post get?")
    assert res is None or "pizza" not in (res.answer or "")


def test_count_requires_person_match(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Caleb: My allergies are peanuts, shellfish, and dust.",
         datetime(2024, 5, 8, 12, 0)),
    ])
    res = _ask(store, scope, "How many allergies does Noor have?")
    assert res is None or ":claim_list_count" not in (res.note or "")


def test_qualified_count_prefers_matching_list(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Dara: The Apollo project release blockers are the auth bug, the load test, "
         "and the sign-off.", datetime(2024, 3, 2, 12, 0)),
        ("Dara: The Zephyr project release blockers are the API freeze and the rollout.",
         datetime(2024, 4, 2, 12, 0)),
    ])
    res = _ask(store, scope, "How many release blockers does the Apollo project have?")
    assert res is not None and res.answer.startswith("3 release blockers")


def test_short_item_list_enum_survives_verify_form_floor(tmp_path):
    from eidetic.models import NLILabel
    from eidetic.smqe import structured_answer

    class _R:
        def __init__(self, s):
            self.store = s

        def verify_citation(self, rec, atom):
            ok = " ".join(atom.lower().split()) in " ".join((rec.text or "").lower().split())
            return (NLILabel.ENTAILMENT, 1.0) if ok else (NLILabel.NEUTRAL, 0.0)

    store, scope = _store_with(tmp_path, [
        ("Priya: They can do tricks like flip, spin, twirl, and bow.",
         datetime(2024, 4, 1, 12, 0)),
    ])
    ans = structured_answer(_R(store), "What tricks do Priya's parrots know?",
                            at=1_800_000_000, verify=True, scope=scope)
    assert ans is not None and ans.verified
    for item in ("flip", "spin", "twirl", "bow"):
        assert item in ans.answer


def test_fragment_soup_without_claim_backing_still_refused(tmp_path):
    from eidetic.models import ExecutionPlan, StructuredAnswerResult, StructuredSupport
    from eidetic.smqe.verify import answer_from_result

    class _R:
        def __init__(self, s):
            self.store = s

        def verify_citation(self, rec, atom):
            from eidetic.models import NLILabel
            return (NLILabel.ENTAILMENT, 1.0)

    store, scope = _store_with(tmp_path, [
        ("Kip: Good, ok, you get the idea.", datetime(2024, 4, 1, 12, 0)),
    ])
    rec = next(iter(store.active_records_at(scope=scope)))
    result = StructuredAnswerResult(
        answer="Good, Ok, You Get", op="open_inference", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id, proof_atom=rec.text,
                                    answer_atom=rec.text)],
        note="smqe:open_inference:claim",
    )
    assert answer_from_result(_R(store), "What did Kip say?", result, verify=True) is None


def test_who_gave_answers_from_support_relation_claim(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Priya: When I was in school, we hit some cash troubles and had to lean on "
         "outside help from out cousin.", datetime(2024, 4, 1, 12, 0)),
        ("Priya: The balcony plants are doing great this month.", datetime(2024, 4, 2, 12, 0)),
    ])
    res = _ask(store, scope,
               "Who gave Priya's family cash when she was going through a rough patch?")
    assert res is not None and res.answer == "cousin"
    assert res.backend == "claim"


def test_visited_question_never_composes_from_favorite_list_claims():
    from eidetic.smqe.qa_ops import _claim_enumeration_answer
    scope = Scope(namespace="fav")
    rec = _rec("Farid: My favorite cities are Rome and Lisbon.",
               datetime(2024, 5, 8, 12, 0), scope, 0)
    atoms = [(1.0, c, c.proof_atom) for c in claims_for_record(rec)
             if c.filters.get("list") == "item"]
    assert atoms
    answer, _sel = _claim_enumeration_answer("Which cities has Farid visited?", atoms)
    assert answer == ""
    answer2, _sel2 = _claim_enumeration_answer("What cities does Farid like?", atoms)
    assert "Rome" in answer2 and "Lisbon" in answer2
