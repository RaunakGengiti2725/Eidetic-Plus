"""Partial-list verified-wrong killers: claim hygiene (deterministic ids + untyped
quarantine), write-side activity/acquisition coverage, and the read-side sibling-union
completion sweep with its completeness refusal gate.

All dialogues are FABRICATED shapes; no benchmark rows.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from eidetic.models import ClaimRecord, MemoryRecord, Scope
from eidetic.smqe.claim_extraction import (
    claims_for_record,
    deterministic_claim_id,
)
from eidetic.smqe.executor import execute_plan
from eidetic.smqe.planner import plan_query
from eidetic.smqe.qa_ops import _claim_enumeration_answer
from eidetic.smqe.record_ops import execute_claim_op
from eidetic.store import RecordStore


def _rec(text, dt, scope, i):
    return MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}", text=text,
                        scope=scope, valid_at=dt.timestamp())


def _store_with(tmp_path, rows, ns):
    store = RecordStore(tmp_path / f"{ns}.sqlite")
    scope = Scope(namespace=ns)
    for i, (text, dt) in enumerate(rows):
        rec = _rec(text, dt, scope, i)
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))
    return store, scope


def _claim_rows(claims):
    return [(1.0 + i * 0.01, c, c.proof_atom or str(c.value or ""))
            for i, c in enumerate(claims)]


# ------------------------------------------------------------------ wave 0.1: ids

def test_deterministic_claim_id_is_stable_across_reextraction():
    scope = Scope(namespace="detid")
    rec = _rec("Dara: My release blockers are the API migration, the load test, and the sign-off.",
               datetime(2024, 3, 2, 12, 0), scope, 0)
    first = sorted(c.claim_id for c in claims_for_record(rec))
    second = sorted(c.claim_id for c in claims_for_record(rec))
    assert first == second
    assert all(cid.startswith("claim_") for cid in first)


def test_reingest_does_not_duplicate_claim_rows(tmp_path):
    store = RecordStore(tmp_path / "reingest.sqlite")
    scope = Scope(namespace="reingest")
    rec = _rec("Sena: I really enjoy trail running, cold brew, and jigsaw puzzles.",
               datetime(2024, 5, 6, 12, 0), scope, 0)
    store.upsert_record(rec)
    store.add_claims(claims_for_record(rec))
    n_first = len(store.claims_in_scope(scope))
    store.add_claims(claims_for_record(rec))       # re-ingest of the same record
    n_second = len(store.claims_in_scope(scope))
    assert n_first == n_second


def test_same_object_in_two_lists_keeps_distinct_ids():
    scope = Scope(namespace="twolists")
    rec = _rec("Kai: My hobbies are hiking, painting, and chess. "
               "My weekend plans are hiking, baking, and a nap.",
               datetime(2024, 6, 1, 12, 0), scope, 0)
    claims = [c for c in claims_for_record(rec) if c.filters.get("list") == "item"]
    hiking = [c for c in claims if c.object.lower() == "hiking"]
    assert len(hiking) == 2
    assert hiking[0].claim_id != hiking[1].claim_id


def test_store_dedupe_claims_collapses_hand_built_duplicates(tmp_path):
    store = RecordStore(tmp_path / "dedupe.sqlite")
    scope = Scope(namespace="dedupe")
    rec = _rec("Noor: I love oat biscuits.", datetime(2024, 2, 2, 12, 0), scope, 0)
    store.upsert_record(rec)
    base = claims_for_record(rec)
    store.add_claims(base)
    # Simulate the historic random-id duplication: same claims, fresh random ids.
    dupes = []
    for i, c in enumerate(base):
        d = c.model_copy(deep=True)
        d.claim_id = f"claim_random_{i}"
        dupes.append(d)
    store.add_claims(dupes)
    assert len(store.claims_in_scope(scope)) == 2 * len(base)
    out = store.dedupe_claims(scope)
    assert out["before"] == 2 * len(base)
    assert out["after"] == len(base)
    assert len(store.claims_in_scope(scope)) == len(base)


# ------------------------------------------------------------------ wave 0.2: untyped

def test_fallback_claims_are_tagged_untyped():
    scope = Scope(namespace="untyped")
    rec = _rec("Rio: Keep your chin up and rock the recital prep grind everyone.",
               datetime(2024, 2, 3, 12, 0), scope, 0)
    claims = claims_for_record(rec)
    generic = [c for c in claims if not c.filters.get("list")
               and not c.filters.get("event") == "dated"]
    assert generic, "expected at least the catch-all claim"
    assert any(c.filters.get("untyped") == "1" for c in generic)


def test_typed_preference_claim_is_not_tagged_untyped():
    scope = Scope(namespace="typed")
    rec = _rec("Noor: I love oat biscuits.", datetime(2024, 2, 4, 12, 0), scope, 0)
    claims = claims_for_record(rec)
    love = [c for c in claims if c.predicate == "love"]
    assert love
    assert all(c.filters.get("untyped") != "1" for c in love)


def test_session_id_shaped_subject_is_tagged_untyped():
    scope = Scope(namespace="sess")
    rec = MemoryRecord(memory_id="m0", content_hash="h0", scope=scope,
                       valid_at=datetime(2024, 2, 5, 12, 0).timestamp(),
                       source="c1_session_9",
                       text="Wandering around the plaza after sunset with everyone nearby.")
    claims = claims_for_record(rec)
    assert claims
    assert all(c.filters.get("untyped") == "1" for c in claims
               if c.subject == "c1_session_9")


# ------------------------------------------------------------------ 1a: write coverage

def test_go_gerund_activity_claim_minted():
    scope = Scope(namespace="kayak")
    rec = _rec("Mia: I'm off to go kayaking with my nieces.",
               datetime(2024, 4, 2, 12, 0), scope, 0)
    claims = [c for c in claims_for_record(rec) if c.filters.get("enum_fact") == "1"]
    assert claims, "expected an activity claim from the go-gerund shape"
    claim = claims[0]
    assert claim.subject == "Mia"
    assert claim.predicate == "kayak"
    assert claim.object.startswith("kayaking")
    assert claim.filters.get("action") == "activity"
    assert claim.filters.get("untyped") != "1"


def test_bare_acquisition_claim_minted():
    scope = Scope(namespace="binoculars")
    rec = _rec("Rhea: Just got some new binoculars for the trip!",
               datetime(2024, 4, 3, 12, 0), scope, 0)
    claims = [c for c in claims_for_record(rec)
              if c.filters.get("enum_fact") == "1" and c.filters.get("action") == "acquire"]
    assert claims, "expected an acquisition claim from the bare got-shape"
    claim = claims[0]
    assert claim.subject == "Rhea"
    # OUTSIDE the offered/received/given gift family on purpose: a conversational
    # acquisition must never enumerate as a gift.
    assert claim.predicate == "acquired"
    assert "binoculars" in claim.object.lower()


def test_junk_go_shape_mints_nothing():
    scope = Scope(namespace="junkgo")
    rec = _rec("Mia: I'm off to a good start.", datetime(2024, 4, 4, 12, 0), scope, 0)
    claims = [c for c in claims_for_record(rec) if c.filters.get("enum_fact") == "1"]
    assert claims == []


# ------------------------------------------------------------------ 1b: sibling union

def _seven_trick_record(scope):
    return _rec("Bram: The dogs know tricks like sit, stay, paw, rollover, spin, "
                "fetch, and speak.", datetime(2024, 5, 1, 12, 0), scope, 0)


def test_sweep_unions_same_list_siblings_missing_from_atoms():
    scope = Scope(namespace="sweep")
    rec = _seven_trick_record(scope)
    claims = [c for c in claims_for_record(rec) if c.filters.get("list") == "item"]
    assert len(claims) == 7
    subset = [c for c in claims if int(c.filters["list_index"]) < 4]
    q = "What tricks does the dog know?"
    answer, selected = _claim_enumeration_answer(q, _claim_rows(subset),
                                                 claim_pool=claims)
    assert answer
    for obj in ("sit", "stay", "paw", "rollover", "spin", "fetch", "speak"):
        assert obj in answer
    assert len(selected) == 7
    # every item carries its own typed claim support
    assert all(isinstance(item, ClaimRecord) for _s, item, _a in selected)


def test_sweep_declines_when_a_sibling_is_unrecoverable():
    scope = Scope(namespace="gate")
    rec = _seven_trick_record(scope)
    claims = [c for c in claims_for_record(rec) if c.filters.get("list") == "item"]
    pool = [c for c in claims if int(c.filters["list_index"]) != 3]  # one sibling gone
    subset = [c for c in pool if int(c.filters["list_index"]) < 3]
    q = "What tricks does the dog know?"
    answer, selected = _claim_enumeration_answer(q, _claim_rows(subset),
                                                 claim_pool=pool)
    assert (answer, selected) == ("", [])


def test_invalidated_sibling_blocks_partial_verified_enum(tmp_path):
    store, scope = _store_with(
        tmp_path,
        [("Bram: The dogs know tricks like sit, stay, paw, rollover, spin, fetch, "
          "and speak.", datetime(2024, 5, 1, 12, 0))],
        "inval",
    )
    victim = next(c for c in store.claims_in_scope(scope)
                  if c.filters.get("list") == "item" and c.object == "spin")
    victim.invalid_at = datetime(2024, 5, 2, 12, 0).timestamp()
    store.add_claim(victim)
    q = "What tricks does the dog know?"
    res = execute_plan(plan_query(q), q,
                       records=store.active_records_at(scope=scope),
                       claims=store.active_claims_at(scope=scope))
    if res is not None:
        assert ":claim_list_enum" not in (res.note or "")
        assert "spin" not in (res.answer or "") or "fetch" not in (res.answer or "")


def test_cross_family_union_adds_activity_claim(tmp_path):
    store, scope = _store_with(
        tmp_path,
        [("Rhea: I really enjoy trail running and jigsaw puzzles.",
          datetime(2024, 5, 3, 12, 0)),
         ("Rhea: I'm off to go kayaking with my nieces.",
          datetime(2024, 5, 4, 12, 0))],
        "crossfam",
    )
    q = "What activities does Rhea enjoy doing?"
    res = execute_claim_op(plan_query(q), q, store.active_claims_at(scope=scope))
    assert res is not None
    for obj in ("trail running", "jigsaw puzzles", "kayaking"):
        assert obj in res.answer
    # mixed provenance must NOT take the claim_list_enum verification carve-out
    assert ":claim_list_enum" not in (res.note or "")


def test_owner_pet_bridge_unions_across_subjects():
    scope = Scope(namespace="bridge")
    bridge = ClaimRecord(
        claim_type="state", scope=scope, subject="Omar", predicate="has",
        object="Biscuit", source_memory_id="mb0",
        proof_atom="Omar has a puppy named Biscuit.", valid_at=100.0,
    )
    tricks = []
    atom = "Biscuit can do tricks like sit, stay, and paw."
    for idx, obj in enumerate(["sit", "stay", "paw"]):
        tricks.append(ClaimRecord(
            claim_type="state", scope=scope, subject="Biscuit", predicate="",
            object=obj, source_memory_id="mb1", proof_atom=atom, valid_at=101.0,
            filters={"list": "item", "list_id": "lidb", "list_label": "tricks",
                     "list_size": 3, "list_index": idx},
        ))
    q = "What tricks do Omar's pets know?"
    res = execute_claim_op(plan_query(q), q, [bridge] + tricks)
    assert res is not None
    for obj in ("sit", "stay", "paw"):
        assert obj in res.answer


# ------------------------------------------------------------------ family-gating regressions

def test_enum_fact_claims_do_not_invent_hobbies(tmp_path):
    """A bare acquisition and a hated one-off attempt must never enumerate as enjoyed
    hobbies: enum_fact claims are gated on the query's RESOLVED verb family plus the
    claim's action type, never on do-support."""
    store, scope = _store_with(
        tmp_path,
        [("Sena: I really enjoy trail running, watercolor painting, and chess.",
          datetime(2024, 6, 2, 12, 0)),
         ("Sena: Just got some new binoculars for the trip!",
          datetime(2024, 6, 3, 12, 0)),
         ("Sena: I tried skiing once and hated every minute of it.",
          datetime(2024, 6, 4, 12, 0))],
        "hobbygate",
    )
    q = "What hobbies does Sena enjoy?"
    res = execute_claim_op(plan_query(q), q, store.active_claims_at(scope=scope))
    assert res is not None
    for good in ("trail running", "watercolor painting", "chess"):
        assert good in res.answer
    assert "binoculars" not in res.answer
    assert "skiing" not in res.answer


def test_activity_claims_do_not_leak_into_other_families(tmp_path):
    """'What books does Sam read?' resolves the read family; activity/acquire enum_fact
    claims in the pool must not union into the answer -- and with ZERO book claims the
    sweep must not fabricate an answer from them."""
    store, scope = _store_with(
        tmp_path,
        [("Sam: I read Dune and Neuromancer last month.", datetime(2024, 6, 5, 12, 0)),
         ("Sam: I'm off to go kayaking with my nieces.", datetime(2024, 6, 6, 12, 0)),
         ("Sam: Just got some new binoculars for the trip!", datetime(2024, 6, 7, 12, 0))],
        "bookgate",
    )
    q = "What books does Sam read?"
    res = execute_claim_op(plan_query(q), q, store.active_claims_at(scope=scope))
    if res is not None:
        assert "kayaking" not in (res.answer or "")
        assert "binoculars" not in (res.answer or "")

    store2, scope2 = _store_with(
        tmp_path,
        [("Sam: I'm off to go kayaking with my nieces.", datetime(2024, 6, 6, 12, 0)),
         ("Sam: Just got some new binoculars for the trip!", datetime(2024, 6, 7, 12, 0))],
        "bookgate2",
    )
    res2 = execute_claim_op(plan_query(q), q, store2.active_claims_at(scope=scope2))
    assert res2 is None or ("kayaking" not in (res2.answer or "")
                            and "binoculars" not in (res2.answer or ""))


def test_bare_acquisition_never_enumerates_as_gift():
    """(speaker, acquired, 'nasty cough') is lemma-adjacent to the received/gift family
    but is not an answer to a gifts question -- on the direct path or the sweep."""
    scope = Scope(namespace="giftgate")
    atom = "Rob received a watch, a scarf, and cufflinks for his birthday."
    gifts = [
        ClaimRecord(
            claim_type="state", scope=scope, subject="Rob", predicate="received",
            object=obj, source_memory_id="mg0", proof_atom=atom, valid_at=100.0,
            filters={"list": "item", "list_id": "lg", "list_label": "gifts",
                     "list_size": 3, "list_index": idx},
        )
        for idx, obj in enumerate(["watch", "scarf", "cufflinks"])
    ]
    cough_rec = _rec("Rob: Just got a nasty cough, so staying in this weekend.",
                     datetime(2024, 6, 8, 12, 0), scope, 77)
    cough_claims = [c for c in claims_for_record(cough_rec)
                    if c.filters.get("action") == "acquire"]
    assert cough_claims, "acquisition claim should still be minted (for typed consumers)"
    pool = gifts + cough_claims
    q = "What gifts has Rob received?"
    res = execute_claim_op(plan_query(q), q, pool)
    assert res is not None
    for obj in ("watch", "scarf", "cufflinks"):
        assert obj in res.answer
    assert "cough" not in res.answer


def test_bridge_does_not_fire_on_the_enumeration_head_noun():
    """'Priya's hobbies' names the question's own head, not a possession entity: the
    owner->possession bridge must not union a different subject's claims."""
    scope = Scope(namespace="nobridge")
    claims = [
        ClaimRecord(claim_type="state", scope=scope, subject="Priya",
                    predicate="enjoys", object="gardening", source_memory_id="mp0",
                    proof_atom="Priya enjoys gardening.", valid_at=100.0),
        ClaimRecord(claim_type="state", scope=scope, subject="Priya",
                    predicate="enjoys", object="reading", source_memory_id="mp1",
                    proof_atom="Priya enjoys reading.", valid_at=100.0),
        ClaimRecord(claim_type="state", scope=scope, subject="Priya",
                    predicate="has", object="Mango", source_memory_id="mp2",
                    proof_atom="Priya has a parrot named Mango.", valid_at=100.0),
        ClaimRecord(claim_type="state", scope=scope, subject="Mango",
                    predicate="loves", object="sunflower seeds", source_memory_id="mp3",
                    proof_atom="Mango loves sunflower seeds.", valid_at=100.0),
        ClaimRecord(claim_type="state", scope=scope, subject="Mango",
                    predicate="loves", object="shiny bells", source_memory_id="mp4",
                    proof_atom="Mango loves shiny bells.", valid_at=100.0),
    ]
    q = "What do Priya's hobbies include?"
    res = execute_claim_op(plan_query(q), q, claims)
    if res is not None:
        assert "sunflower" not in res.answer
        assert "bells" not in res.answer


def test_no_person_query_does_not_union_other_speakers():
    """With no TitleCase person in the query, the sweep's family union must NOT merge
    every speaker's family-matching claims from the namespace-wide pool."""
    scope = Scope(namespace="noperson")

    def claim(i, subject, obj):
        return ClaimRecord(claim_type="state", scope=scope, subject=subject,
                           predicate="enjoys", object=obj, source_memory_id=f"mn{i}",
                           proof_atom=f"{subject} enjoys {obj}.", valid_at=100.0)

    angela = [claim(0, "Angela", "hiking"), claim(1, "Angela", "pottery")]
    bob = [claim(2, "Bob", "chess"), claim(3, "Bob", "karate")]
    q = "What hobbies does she enjoy?"
    answer, _sel = _claim_enumeration_answer(q, _claim_rows(angela),
                                             claim_pool=angela + bob)
    assert "chess" not in answer and "karate" not in answer


def test_aggregate_list_claim_does_not_double_count(tmp_path):
    """The extractor's whole-list aggregate claim must never re-enter the enumeration
    as an extra member ('gardening, baking, reading, and gardening, reading, and
    baking' was a shipped verified answer shape)."""
    store, scope = _store_with(
        tmp_path,
        [("Priya: I enjoy gardening, reading, and baking.",
          datetime(2024, 6, 9, 12, 0))],
        "agg",
    )
    for q in ("What hobbies does Priya enjoy?", "What do Priya's hobbies include?"):
        res = execute_claim_op(plan_query(q), q, store.active_claims_at(scope=scope))
        if res is None:
            continue
        items = [i.strip() for i in
                 res.answer.replace(", and ", ", ").replace(" and ", ", ").split(", ")]
        assert sorted(items) == sorted(set(items)), res.answer
        assert set(items) <= {"gardening", "reading", "baking"}, res.answer


def _city_list_claims(scope, objs, label="cities"):
    atom = "Angela visited " + ", ".join(objs) + "."
    return [
        ClaimRecord(
            claim_type="state", scope=scope, subject="Angela", predicate="visited",
            object=obj, source_memory_id="mc0", proof_atom=atom, valid_at=100.0,
            filters={"list": "item", "list_id": "lc", "list_label": label,
                     "list_size": len(objs), "list_index": idx},
        )
        for idx, obj in enumerate(objs)
    ]


def test_junk_member_does_not_forfeit_partial_enum():
    """A member that the read-side per-item floor deliberately filters ('home' is not a
    proper-noun city) is OUR selection, not a missing sibling: the gate must not turn a
    previously-correct partial answer into a decline."""
    scope = Scope(namespace="junkmember")
    claims = _city_list_claims(scope, ["Rome", "Paris", "home"])
    q = "Which cities has Angela visited?"
    answer, _sel = _claim_enumeration_answer(q, _claim_rows(claims), claim_pool=claims)
    assert "Rome" in answer and "Paris" in answer
    assert "home" not in answer


def test_nine_member_list_caps_without_declining():
    """The 8-value cap truncates; it must not convert a fully-live list into a decline."""
    scope = Scope(namespace="ninecap")
    objs = ["Rome", "Paris", "Lisbon", "Oslo", "Prague", "Vienna", "Dublin",
            "Athens", "Warsaw"]
    claims = _city_list_claims(scope, objs)
    q = "Which cities has Angela visited?"
    answer, _sel = _claim_enumeration_answer(q, _claim_rows(claims), claim_pool=claims)
    assert answer, "cap truncation must not decline"
    assert sum(1 for o in objs if o in answer) == 8


def test_sweep_refuses_invalidated_pool_member():
    """Defense in depth: an invalidated sibling handed to the sweep in the pool is
    neither resurrected nor counted as collected -- the enumeration declines."""
    scope = Scope(namespace="invalpool")
    claims = _city_list_claims(scope, ["Rome", "Paris", "Lisbon"])
    claims[2].invalid_at = 50.0            # long past
    q = "Which cities has Angela visited?"
    answer, sel = _claim_enumeration_answer(
        q, _claim_rows(claims[:2]), claim_pool=claims)
    assert (answer, sel) == ("", [])


def test_dedupe_claims_is_atomic_on_failure(tmp_path, monkeypatch):
    """A failure mid-dedupe must never leave the claims table emptied: the delete and
    the re-insert are one transaction, and serialization happens before it opens."""
    store = RecordStore(tmp_path / "atomic.sqlite")
    scope = Scope(namespace="atomic")
    rec = _rec("Noor: I love oat biscuits.", datetime(2024, 2, 2, 12, 0), scope, 0)
    store.upsert_record(rec)
    base = claims_for_record(rec)
    store.add_claims(base)
    n_before = len(store.claims_in_scope(scope))
    assert n_before

    def boom(self, *a, **k):
        raise RuntimeError("serialization failure")

    monkeypatch.setattr(ClaimRecord, "model_dump_json", boom)
    with pytest.raises(RuntimeError):
        store.dedupe_claims(scope)
    monkeypatch.undo()
    assert len(store.claims_in_scope(scope)) == n_before


def test_complete_list_answer_is_unchanged_and_keeps_note(tmp_path):
    store, scope = _store_with(
        tmp_path,
        [("Sena: I really enjoy trail running, cold brew, and jigsaw puzzles.",
          datetime(2024, 4, 1, 12, 0))],
        "stable",
    )
    q = "What hobbies does Sena enjoy?"
    res = execute_claim_op(plan_query(q), q, store.active_claims_at(scope=scope))
    assert res is not None
    items = {i.strip() for i in res.answer.replace(", and ", ", ").split(", ")}
    assert items == {"trail running", "cold brew", "jigsaw puzzles"}
    assert ":claim_list_enum" in (res.note or "")
