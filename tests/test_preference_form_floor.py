"""Preference-fragment verified-wrong killers: the ':suggestion_synth' carve-out tag, the
untagged preference_synth form floor, and the mid-sentence provenance refusal.

All dialogues are FABRICATED shapes; no benchmark rows.
"""
from __future__ import annotations

from eidetic.models import (
    ClaimRecord,
    MemoryRecord,
    NLILabel,
    Scope,
    StructuredAnswerResult,
    StructuredSupport,
)
from eidetic.smqe.executor import execute_plan
from eidetic.smqe.planner import plan_query
from eidetic.smqe.record_ops import execute_claim_op
from eidetic.smqe.verify import (
    _atom_anchor_allowed,
    answer_from_result,
    preference_answer_form_credible,
)
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


def _store_rec(tmp_path, name, text):
    store = RecordStore(tmp_path / f"{name}.sqlite")
    rec = MemoryRecord(text=text, source="user", scope=Scope(namespace=name),
                       valid_at=1_700_000_000.0, content_hash="h-" + name,
                       raw_uri="mem://synthetic")
    store.upsert_record(rec)
    return store, rec


def _pref_result(rec, answer, atom=None, *, note="smqe:preference_synth:claim"):
    return StructuredAnswerResult(
        answer=answer, op="preference_synth", backend="claim", confidence=0.9,
        supports=[StructuredSupport(memory_id=rec.memory_id, proof_atom=atom or answer)],
        note=note,
    )


# ------------------------------------------------------------------ the floor itself

def test_floor_rejects_first_person_clause_fragment():
    assert not preference_answer_form_credible(
        "What has Lena been working on for her craft stall?",
        "I have put together a slide deck to show folks how to fold paper cranes, btw",
    )


def test_floor_rejects_discourse_opener_heads():
    q = "What does Marco prefer for breakfast?"
    for answer in ("btw, the tart was great", "it's the tart", "and the tart",
                   "well, whatever is around", "that's the tart"):
        assert not preference_answer_form_credible(q, answer), answer


def test_floor_rejects_dangling_tails_and_afterthoughts():
    q = "What treats does Noor like?"
    assert not preference_answer_form_credible(q, "cakes and")
    assert not preference_answer_form_credible(q, "the tart with")
    assert not preference_answer_form_credible(q, "lemon tart, btw")
    assert not preference_answer_form_credible(q, "lemon tart, though")


def test_floor_rejects_unbounded_or_clausal_phrases():
    q = "Which dance style does Ada like most?"
    assert not preference_answer_form_credible(
        q, "all styles, but tango is my top pick")
    assert not preference_answer_form_credible(
        q, "the long winding program of dances she keeps rehearsing every single week")


def test_floor_survivors():
    assert preference_answer_form_credible("What snack does Noor love?", "oat biscuits")
    assert preference_answer_form_credible(
        "Which car would Raj pick, a Falcon Roadster or a Comet Wagon?", "Falcon Roadster")
    assert preference_answer_form_credible(
        "Does Dana prefer mornings?", "Yes - she prefers mornings")
    assert preference_answer_form_credible(
        "Would Dana likely prefer mornings?", "Yes - she prefers mornings")
    assert preference_answer_form_credible("What show does Ira rewatch?", '"That"')
    assert preference_answer_form_credible("What bedtime does Kim prefer?", "11 pm")


def test_floor_accepts_unquoted_titlecase_titles():
    q = "What is Ira's favorite movie?"
    for title in ("I Am Legend", "You Belong With Me",
                  "The Lord of the Rings: The Return of the King"):
        assert preference_answer_form_credible(q, title), title


def test_floor_rejects_embedded_quote_fragments():
    q = "What snack does Mel like?"
    assert not preference_answer_form_credible(q, 'I told him "no way", btw')
    assert not preference_answer_form_credible(q, 'and she said "wow" so I did, btw')
    # the WHOLE-answer quote (a bare title) still fails open
    assert preference_answer_form_credible("What show does Ira rewatch?", '"That"')


def test_floor_rejects_short_token_junk_but_keeps_times():
    q = "What snack does Mel like?"
    assert not preference_answer_form_credible(q, "oh no")
    assert not preference_answer_form_credible(q, "so so so")
    assert preference_answer_form_credible("What bedtime does Kim prefer?", "11 pm")
    assert preference_answer_form_credible("When does Kim get up?", "7:30")


def test_position_floor_accepts_prepositional_preference_frames():
    from eidetic.smqe.verify import _pref_premise_position_ok
    survivors = [
        ("pottery", "Mira: I'm really into pottery these days."),
        ("jazz", "I've been obsessed with jazz lately."),
        ("sushi", "On Fridays we usually go out for sushi."),
        ("11 pm", "I usually head to bed at 11 pm."),
        ("hiking", "My weekends revolve around hiking."),
    ]
    for answer, premise in survivors:
        assert _pref_premise_position_ok(answer, premise), (answer, premise)
    # an object mid-way through an UNRELATED clause is still a shard
    assert not _pref_premise_position_ok(
        "woodworking kits", "I spent the weekend browsing woodworking kits at the fair.")


def test_floor_rejects_yes_no_on_non_polarity_question():
    assert not preference_answer_form_credible(
        "What does Dana prefer for commuting?", "Yes - glad you have people to lean on")


def test_floor_rejects_raw_atom_join_composite():
    assert not preference_answer_form_credible(
        "What projects does Lena prefer?",
        "I've been working on the mural; We should catch up soon")


# ------------------------------------------------------------------ verify integration

def test_untagged_preference_fragment_is_refused(tmp_path):
    store, rec = _store_rec(
        tmp_path, "fragment",
        "Lena: I have built a slideshow to teach how to style my scarves, btw. It was fun.")
    result = _pref_result(
        rec, "I have built a slideshow to teach how to style my scarves, btw")
    ans = answer_from_result(
        _Retriever(store), "What fashion project does Lena prefer to work on?",
        result, verify=True)
    assert ans is None


def test_tagged_suggestion_synth_keeps_current_behavior(tmp_path):
    store, rec = _store_rec(
        tmp_path, "tagged",
        "Lena: I have built a slideshow to teach how to style my scarves, btw. It was fun.")
    result = _pref_result(
        rec, "I have built a slideshow to teach how to style my scarves, btw",
        note="smqe:preference_synth:claim:suggestion_synth")
    ans = answer_from_result(
        _Retriever(store), "What fashion project does Lena prefer to work on?",
        result, verify=True)
    assert ans is not None
    assert ans.verified


def test_untagged_preference_junk_enumeration_is_refused(tmp_path):
    store, rec = _store_rec(
        tmp_path, "junkenum", "Pat: Good, Ok, You Get the idea for the picnic plan.")
    result = _pref_result(rec, "Good, Ok, You Get",
                          atom="Good, Ok, You Get the idea for the picnic plan.")
    ans = answer_from_result(
        _Retriever(store), "What picnic snacks does Pat prefer?", result, verify=True)
    assert ans is None


def test_mid_clause_provenance_is_refused(tmp_path):
    store, rec = _store_rec(
        tmp_path, "midclause",
        "Marco: I spent the weekend browsing woodworking kits at the fair.")
    result = _pref_result(rec, "woodworking kits",
                          atom="I spent the weekend browsing woodworking kits at the fair.")
    ans = answer_from_result(
        _Retriever(store), "What hobby supplies does Marco prefer?", result, verify=True)
    assert ans is None


def test_clean_boundary_provenance_survives(tmp_path):
    store, rec = _store_rec(
        tmp_path, "boundary", "Marco: My favorite gift: Woodworking kits. They rock.")
    result = _pref_result(rec, "Woodworking kits",
                          atom="My favorite gift: Woodworking kits.")
    ans = answer_from_result(
        _Retriever(store), "What hobby supplies does Marco prefer?", result, verify=True)
    assert ans is not None
    assert ans.verified


# ------------------------------------------------------------------ producer quarantine

def test_untyped_junk_never_feeds_untagged_preference_answers():
    """A non-suggestion preference query must read the untyped-QUARANTINED pool: the
    suggestion pass-through must not resurrect junk claims as answer text, and its
    output must not wear the ':suggestion_synth' exemption tag."""
    scope = Scope(namespace="junkpool")
    junk = ClaimRecord(
        claim_type="state", scope=scope, subject="it", predicate="",
        object="btw my favorite movies list keeps growing",
        source_memory_id="mj0",
        proof_atom="btw my favorite movies list keeps growing, lol",
        valid_at=100.0, filters={"untyped": "1"},
    )
    q = "What are Pat's favorite movies?"
    res = execute_claim_op(plan_query(q), q, [junk])
    if res is not None:
        assert ":suggestion_synth" not in (res.note or "")
        assert "btw" not in (res.answer or "")


def test_non_suggestion_answer_is_not_tagged_suggestion_synth():
    """An option-choice answer for an ordinary preference question is NOT suggestion
    synthesis: it must not carry the tag that exempts it from every floor."""
    from datetime import datetime
    q = "Would Pat prefer a cedar desk or a walnut desk?"
    rec = MemoryRecord(
        memory_id="mo0", content_hash="ho0", scope=Scope(namespace="untag"),
        text="Pat: I'd take a cedar desk any day - I love the cedar grain.",
        valid_at=datetime(2024, 3, 1, 12, 0).timestamp(),
    )
    res = execute_plan(plan_query(q), q, records=[rec], claims=[])
    assert res is not None and "cedar" in res.answer
    assert ":suggestion_synth" not in (res.note or "")


# ------------------------------------------------------------------ anchor narrowing

def test_same_record_multi_support_preference_loses_anchor_exemption():
    result = StructuredAnswerResult(
        answer="creating a scrapbook", op="preference_synth", backend="claim",
        confidence=0.9,
        supports=[
            StructuredSupport(memory_id="m1", proof_atom="totally unrelated words here"),
            StructuredSupport(memory_id="m1", proof_atom="other unrelated words there"),
        ],
        note="smqe:preference_synth:claim",
    )
    assert _atom_anchor_allowed("What hobby does Lena prefer?", result) is False


def test_tagged_suggestion_synth_keeps_anchor_exemption():
    result = StructuredAnswerResult(
        answer="creating a scrapbook", op="preference_synth", backend="claim",
        confidence=0.9,
        supports=[
            StructuredSupport(memory_id="m1", proof_atom="totally unrelated words here"),
            StructuredSupport(memory_id="m1", proof_atom="other unrelated words there"),
        ],
        note="smqe:preference_synth:claim:suggestion_synth",
    )
    assert _atom_anchor_allowed("What hobby does Lena prefer?", result) is True
