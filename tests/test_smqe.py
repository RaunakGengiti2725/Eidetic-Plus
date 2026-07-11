from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from eidetic.models import ClaimRecord, MemoryRecord, Modality, NLILabel, Scope, StructuredAnswerResult, StructuredSupport
from eidetic.smqe import structured_answer
from eidetic.smqe.engine import structured_answer as engine_structured_answer
from eidetic.smqe.engine import structured_recall as engine_structured_recall
from eidetic.smqe.claim_extraction import claims_for_record, validate_extracted_claims
from eidetic.smqe.executor import execute_plan
from eidetic.smqe.planner import plan_query
from eidetic.smqe.verify import answer_from_result
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


def _assert_aggregate_fails_closed(store, query, *, at, scope):
    """P0 fail-closed contract (2026-07-09; eidetic/smqe/verify.py aggregate citation floor).

    A count / cross-session sum DERIVED by enumerating across atoms is no longer marked
    verified. NLI proves each cited atom is verbatim-present, never that the atom SET is the
    right one, so this path shipped 5/6 verified-WRONG on real holdout data (place-names in a
    generic suggestion counted as weddings; body-weight formulas summed into a feed total). Only
    a SINGLE source that STATES the value verifies (the negroni-"10 times" live case) or a fixed
    two-anchor DIFFERENCE (recompute-exact). Everything else abstains -- correct-or-silent.

    The derivation itself is unchanged. This returns the structured_recall trace so the caller
    can still assert the computation produced the right value and selected the right atoms; only
    the verified badge is withheld. `structured_answer` (the verify-or-abstain surface) now
    returns None for these queries."""
    out = engine_structured_recall(_Retriever(store), query, at=at, scope=scope)
    assert out["answered"] is False, f"expected fail-closed abstention, got {out['answer']!r}"
    assert out["verified"] is False
    assert structured_answer(_Retriever(store), query, at=at, scope=scope) is None
    return out


def _proof_of(out) -> str:
    """Support-atom text of an abstained aggregate trace (the derivation's selected atoms),
    the fail-closed analogue of joining citation snippets on a verified answer."""
    return " ".join(s.get("proof_atom") or s.get("answer_atom") or ""
                    for s in (out.get("supports") or []))


def test_claim_store_is_scope_and_time_filtered(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    alpha = Scope(namespace="alpha")
    beta = Scope(namespace="beta")
    alpha_rec = _record("Mira keeps the spare badge in drawer seven.", scope=alpha, valid_at=1.0)
    beta_rec = _record("Mira keeps the red folder in the archive.", scope=beta, valid_at=1.0)
    store.upsert_record(alpha_rec)
    store.upsert_record(beta_rec)
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=alpha,
        subject="Mira",
        predicate="keeps",
        object="spare badge",
        source_memory_id=alpha_rec.memory_id,
        proof_atom="Mira keeps the spare badge in drawer seven.",
        valid_at=10.0,
    ))
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=beta,
        subject="Mira",
        predicate="keeps",
        object="red folder",
        source_memory_id=beta_rec.memory_id,
        proof_atom="Mira keeps the red folder in the archive.",
        valid_at=10.0,
    ))

    assert len(store.active_claims_at(11.0, alpha)) == 1
    assert store.active_claims_at(9.0, alpha) == []
    assert store.active_claims_at(11.0, beta)[0].object == "red folder"


def test_active_claims_require_active_source_memory(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="claim-active-source")
    live = _record("User: I keep the live notebook at Maple Archive.", scope=scope, valid_at=10.0)
    expired = _record("User: I keep the stale notebook at River Gate.", scope=scope, valid_at=10.0)
    expired.expired_at = 20.0
    future = _record("User: I keep the future notebook at Orchid Room.", scope=scope, valid_at=30.0)
    for rec in (live, expired, future):
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=scope,
        subject="User",
        predicate="keep orphan notebook",
        object="Juniper Shelf",
        source_memory_id="missing-source",
        proof_atom="User: I keep the orphan notebook at Juniper Shelf.",
        valid_at=10.0,
    ))

    active = store.active_claims_at(25.0, scope)
    proof = " ".join(claim.proof_atom for claim in active)

    assert "live notebook" in proof
    assert "stale notebook" not in proof
    assert "future notebook" not in proof
    assert "orphan notebook" not in proof


def test_active_claims_scope_is_governed_by_source_memory(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    alpha = Scope(namespace="claim-source-alpha")
    beta = Scope(namespace="claim-source-beta")
    source = _record(
        "User: Mira keeps the blue folder at Cedar Desk.",
        scope=alpha,
        valid_at=10.0,
    )
    store.upsert_record(source)
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=beta,
        subject="Mira",
        predicate="keeps",
        object="blue folder",
        source_memory_id=source.memory_id,
        proof_atom="User: Mira keeps the blue folder at Cedar Desk.",
        valid_at=10.0,
    ))

    beta_claims = store.active_claims_at(11.0, beta)
    alpha_claims = store.active_claims_at(11.0, alpha)

    assert beta_claims == []
    assert len(alpha_claims) == 1
    assert alpha_claims[0].source_memory_id == source.memory_id


def test_structured_answer_rejects_claim_with_cross_scope_source(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    alpha = Scope(namespace="smqe-claim-source-alpha")
    beta = Scope(namespace="smqe-claim-source-beta")
    source = _record(
        "User: Mira keeps the blue folder at Cedar Desk.",
        scope=alpha,
        valid_at=10.0,
    )
    beta_noise = _record(
        "User: Beta namespace has no folder location.",
        scope=beta,
        valid_at=10.0,
    )
    store.upsert_record(source)
    store.upsert_record(beta_noise)
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=beta,
        subject="Mira",
        predicate="keeps",
        object="blue folder",
        source_memory_id=source.memory_id,
        proof_atom="User: Mira keeps the blue folder at Cedar Desk.",
        valid_at=10.0,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Where does Mira keep the blue folder?",
        at=11.0,
        scope=beta,
    )

    assert ans is None


def test_structured_answer_explicit_records_respect_bitemporal_window(tmp_path):
    store = RecordStore(tmp_path / "explicit-record-time.sqlite")
    scope = Scope(namespace="explicit-record-time")
    old = _record("User: Ari keeps the studio key at Cedar Annex.", scope=scope, valid_at=10.0)
    old.invalid_at = 20.0
    new = _record("User: Ari keeps the studio key at Harbor Desk.", scope=scope, valid_at=20.0)
    store.upsert_record(old)
    store.upsert_record(new)

    before = structured_answer(
        _Retriever(store),
        "Where does Ari keep the studio key now?",
        records=[old, new],
        at=15.0,
        scope=scope,
    )
    after = structured_answer(
        _Retriever(store),
        "Where does Ari keep the studio key now?",
        records=[old, new],
        at=25.0,
        scope=scope,
    )

    assert before is not None
    assert before.answer == "Cedar Annex"
    assert before.note == "smqe:latest_value:record"
    assert after is not None
    assert after.answer == "Harbor Desk"
    assert after.note == "smqe:latest_value:record"
    proof = " ".join(c.snippet for c in after.citations)
    assert "Cedar Annex" not in proof


def test_claims_expire_when_source_record_expires(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="claim-source-expiry")
    rec = _record("User: I keep the slate notebook at Cedar Annex.", scope=scope, valid_at=10.0)
    store.upsert_record(rec)
    store.add_claims(claims_for_record(rec))

    assert store.active_records_at(19.0, scope)
    assert store.active_claims_at(19.0, scope)

    store.invalidate_record(rec.memory_id, at=20.0)

    assert store.active_records_at(21.0, scope) == []
    assert store.active_claims_at(21.0, scope) == []
    claims = store.claims_by_source(rec.memory_id)
    assert claims
    assert all(claim.invalid_at == 20.0 for claim in claims)


def test_relation_object_claim_answers_concisely(tmp_path):
    store = RecordStore(tmp_path / "relation-object.sqlite")
    scope = Scope(namespace="relation-object")
    rec = _record(
        "Riley: I wore my old jacket today. This silver compass is special to me - a gift from my aunt.",
        scope=scope,
        valid_at=10.0,
    )
    store.upsert_record(rec)
    store.add_claims(claims_for_record(rec))
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=scope,
        subject="Riley",
        predicate="gift from my aunt",
        object="silver compass",
        source_memory_id=rec.memory_id,
        proof_atom="This silver compass is special to me - a gift from my aunt.",
        valid_at=10.0,
    ))

    ans = structured_answer(
        _Retriever(store),
        "What was the gift from Riley's aunt?",
        at=11.0,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "silver compass"
    assert ans.note == "smqe:latest_value:claim"


def test_temporal_claim_uses_speaker_acquisition_metadata(tmp_path):
    scope = Scope(namespace="temporal-acquisition")
    rec = _record(
        "Priya: These walnut bookends I bought yesterday remind me of cozy evenings.",
        scope=scope,
        valid_at=datetime(2023, 10, 22, 12, 0).timestamp(),
    )
    claims = claims_for_record(rec)
    acquisition = [c for c in claims if c.predicate == "buy" and c.object == "walnut bookends"]

    assert acquisition
    assert acquisition[0].subject == "Priya"

    plan = plan_query("When did Priya buy the walnut bookends?")
    result = execute_plan(plan, "When did Priya buy the walnut bookends?", records=[rec], claims=claims)

    assert result is not None
    assert result.answer == "2023-10-21"
    assert result.backend == "claim"


def test_temporal_claim_prefers_query_month_and_rejects_non_dates(tmp_path):
    scope = Scope(namespace="temporal-month")
    july = _record(
        "Noor: Last weekend our town held a lantern festival!",
        scope=scope,
        valid_at=datetime(2023, 7, 21, 12, 0).timestamp(),
    )
    august = _record(
        "Noor: I went to a lantern festival last Friday and it was magical.",
        scope=scope,
        valid_at=datetime(2023, 8, 14, 12, 0).timestamp(),
    )
    filler = _record("Noor: Absolutely!", scope=scope, valid_at=datetime(2023, 8, 14, 12, 0).timestamp())
    claims = claims_for_record(july) + claims_for_record(august) + claims_for_record(filler)

    plan = plan_query("When did Noor attend a lantern festival in August?")
    result = execute_plan(plan, "When did Noor attend a lantern festival in August?", records=[july, august, filler], claims=claims)

    assert result is not None
    assert result.answer == "2023-08-11"
    assert "Absolutely" not in result.answer


def test_duration_claim_uses_speaker_metadata_for_first_person_sentence(tmp_path):
    scope = Scope(namespace="duration-speaker")
    rec = _record(
        "Wei: They've stood by me through everything, I've known these teammates for 4 years.",
        scope=scope,
        valid_at=10.0,
    )
    claims = claims_for_record(rec)
    duration = [c for c in claims if "known these teammates for 4 years" in c.proof_atom]

    assert duration
    assert duration[0].subject == "Wei"

    plan = plan_query("How long has Wei had her current group of teammates for?")
    result = execute_plan(plan, "How long has Wei had her current group of teammates for?", records=[rec], claims=claims)

    assert result is not None
    assert result.answer == "4 years"
    assert result.backend == "claim"


def test_conversation_duration_answer_carries_prior_question_metadata(tmp_path):
    scope = Scope(namespace="duration-qa")
    rec = _record(
        "Ari: How long have you been married?\nBlair: 5 years already!",
        scope=scope,
        valid_at=10.0,
    )
    claims = claims_for_record(rec)
    duration = [c for c in claims if c.predicate == "duration answer"]

    assert duration
    assert duration[0].subject == "Blair"
    assert duration[0].object == "5 years"
    assert "married" in duration[0].filters["question"]

    plan = plan_query("How long have Blair and her husband been married?")
    result = execute_plan(plan, "How long have Blair and her husband been married?", records=[rec], claims=claims)

    assert plan.op == "latest_value"
    assert result is not None
    assert result.answer == "5 years"
    assert result.backend == "claim"


def test_smqe_fails_closed_for_priority_synthesis_query(tmp_path):
    scope = Scope(namespace="synthesis-priority")
    rec = _record(
        "Blair: I'm starting to realize self-care matters. Blair: Great.",
        scope=scope,
        valid_at=10.0,
    )
    claims = claims_for_record(rec)
    plan = plan_query("How does Blair prioritize self-care?")
    result = execute_plan(plan, "How does Blair prioritize self-care?", records=[rec], claims=claims)

    assert result is None


def test_smqe_fails_closed_for_plan_query_without_direct_structured_answer(tmp_path):
    scope = Scope(namespace="synthesis-plans")
    rec = _record(
        "Ari: We could do a family outing, or wanna plan something special for the weekend.",
        scope=scope,
        valid_at=10.0,
    )
    claims = claims_for_record(rec)
    plan = plan_query("What are Ari's plans for the summer?")
    result = execute_plan(plan, "What are Ari's plans for the summer?", records=[rec], claims=claims)

    assert result is None


def test_smqe_fails_closed_for_open_inference_list_query(tmp_path):
    scope = Scope(namespace="open-list")
    rec = _record(
        "Blair: I painted it yesterday. Blair: I've done a landscape mural too, come see.",
        scope=scope,
        valid_at=10.0,
    )
    claims = claims_for_record(rec)
    plan = plan_query("What has Blair painted?")
    result = execute_plan(plan, "What has Blair painted?", records=[rec], claims=claims)

    assert plan.op == "open_inference"
    assert result is None


def test_action_object_list_claims_answer_from_multiple_sources(tmp_path):
    store = RecordStore(tmp_path / "action-list.sqlite")
    scope = Scope(namespace="action-list")
    first = _record("Blair: I painted a horse recently.", scope=scope, valid_at=10.0)
    second = _record("Blair: I painted a lake sunrise last year.", scope=scope, valid_at=11.0)
    for rec in (first, second):
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(_Retriever(store), "What has Blair painted?", at=12.0, scope=scope)

    assert ans is not None
    assert ans.note == "smqe:open_inference:claim"
    assert ans.verified is True
    assert "horse" in ans.answer
    assert "lake sunrise" in ans.answer


def test_action_location_list_claims_answer_from_multiple_sources(tmp_path):
    store = RecordStore(tmp_path / "action-location-list.sqlite")
    scope = Scope(namespace="action-location-list")
    first = _record("Blair: I camped at the beach last week.", scope=scope, valid_at=10.0)
    second = _record("Blair: I went camping in the forest last month.", scope=scope, valid_at=11.0)
    for rec in (first, second):
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(_Retriever(store), "Where has Blair camped?", at=12.0, scope=scope)

    assert ans is not None
    assert ans.note == "smqe:latest_value:claim"
    assert ans.verified is True
    assert "beach" in ans.answer
    assert "forest" in ans.answer


def test_action_location_claims_scan_late_session_sentences():
    scope = Scope(namespace="action-location-late")
    filler = " ".join(f"Filler sentence {idx}." for idx in range(100))
    rec = _record(
        f"Blair: {filler} Blair: We even went on one more camping trip in the forest.",
        scope=scope,
    )

    claims = claims_for_record(rec)

    assert any(
        claim.filters.get("action") == "location"
        and claim.predicate == "camping"
        and claim.object == "forest"
        for claim in claims
    )


def test_action_location_claims_reject_infinitive_purpose_phrases():
    scope = Scope(namespace="action-location-purpose")
    rec = _record(
        "Blair: I want to help people who need it. Blair: I am looking forward to reading.",
        scope=scope,
    )

    claims = claims_for_record(rec)

    assert not [claim for claim in claims if claim.filters.get("action") == "location"]


def test_smqe_package_reexports_engine_entrypoint():
    assert structured_answer is engine_structured_answer


def test_structured_answer_uses_claims_before_raw_records(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="synthetic-smqe")
    rec = _record("User: I keep my climbing pass at Blue Arch Gym.", scope=scope)
    store.upsert_record(rec)
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=scope,
        subject="user",
        predicate="keep climbing pass",
        object="Blue Arch Gym",
        source_memory_id=rec.memory_id,
        proof_atom="User: I keep my climbing pass at Blue Arch Gym.",
        valid_at=rec.valid_at,
    ))

    ans = structured_answer(_Retriever(store), "Where do I keep my climbing pass?", at=rec.valid_at + 1, scope=scope)

    assert ans is not None
    assert ans.generated_by == "smqe"
    assert ans.note == "smqe:latest_value:claim"
    assert ans.verified is True
    assert "Blue Arch Gym" in ans.answer


def test_structured_answer_infers_scope_from_explicit_records_for_claims(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="explicit-record-scope")
    rec = _record("User: I keep my climbing pass at Blue Arch Gym.", scope=scope)
    store.upsert_record(rec)
    store.add_claim(ClaimRecord(
        claim_type="state",
        scope=scope,
        subject="user",
        predicate="keep climbing pass",
        object="Blue Arch Gym",
        source_memory_id=rec.memory_id,
        proof_atom="User: I keep my climbing pass at Blue Arch Gym.",
        valid_at=rec.valid_at,
    ))

    ans = engine_structured_answer(
        _Retriever(store),
        "Where do I keep my climbing pass?",
        records=[rec],
        at=rec.valid_at + 1,
    )

    assert ans is not None
    assert ans.note == "smqe:latest_value:claim"
    assert ans.verified is True


def test_structured_answer_record_backend_handles_relative_dates(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="synthetic-smqe")
    rec = _record(
        "User: Yesterday I picked up the ceramic kit from the studio.",
        scope=scope,
        valid_at=1_704_196_800.0,  # 2024-01-02 noon UTC, stable as 2024-01-02 in Pacific time
    )
    store.upsert_record(rec)

    ans = structured_answer(_Retriever(store), "When did I pick up the ceramic kit?", at=rec.valid_at + 10, scope=scope)

    assert ans is not None
    assert ans.note == "smqe:relative_temporal:record:atom_derived"
    assert ans.answer == "2024-01-01"


def test_structured_answer_suggestion_uses_query_shape_not_domain_keywords(tmp_path):
    store = RecordStore(tmp_path / "suggestion-shape.sqlite")
    scope = Scope(namespace="suggestion-shape")
    rows = [
        "Assistant: For the aurora sketch kit, good options include graphite vellum sheets and prism clips.",
        "Assistant: For the cedar snack tray, good options include pepper crackers.",
        "User: I only mentioned the aurora sketch kit as a label, not as a snack tray.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_100 + idx))

    ans = structured_answer(
        _Retriever(store),
        "What should I use for the aurora sketch kit?",
        at=1_700_000_200,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:preference_synth:record:suggestion_synth"
    assert "graphite vellum" in ans.answer.lower()
    assert "prism clips" in ans.answer.lower()
    proof = " ".join(c.snippet for c in ans.citations)
    assert "cedar snack" not in proof.lower()


def test_structured_answer_compatibility_suggestion_is_domain_neutral(tmp_path):
    store = RecordStore(tmp_path / "compatibility-suggestion.sqlite")
    scope = Scope(namespace="compatibility-suggestion")
    rec = _record(
        "User: I use a Novum S3 field recorder for my audio setup.\n"
        "Assistant: Consider a Novum S3 shock mount or ArcSound cable kit.\n"
        "User: I want accessories that are compatible with Novum recorders.\n"
        "Assistant: FieldKit makes high-quality weather sleeves that are compatible with Novum recorders.",
        scope=scope,
        valid_at=1_700_000_180,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "Can you suggest accessories that would complement my current audio setup?",
        at=1_700_000_220,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:preference_synth:record:suggestion_synth"
    assert "Novum S3" in ans.answer
    assert "shock mount" in ans.answer
    assert "compatible" in ans.answer.lower()


def test_structured_answer_resource_suggestion_uses_available_items_generically(tmp_path):
    store = RecordStore(tmp_path / "resource-suggestion.sqlite")
    scope = Scope(namespace="resource-suggestion")
    rec = _record(
        "User: I am trying to find project ideas that use copper wire and indigo paper.\n"
        "Assistant: Good options include a shadow lantern and a folded signal card.\n"
        "User: I've been using copper wire and indigo paper lately. I also collected amber beads from the craft shelf.\n"
        "User: I stored a velvet ribbon for a different label.",
        scope=scope,
        valid_at=1_700_000_240,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "What should I make with my craft materials?",
        at=1_700_000_300,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:preference_synth:record:suggestion_synth"
    assert "copper wire" in ans.answer
    assert "indigo paper" in ans.answer
    assert "amber beads" in ans.answer
    assert "velvet ribbon" not in ans.answer
    proof = " ".join(c.snippet for c in ans.citations)
    assert "velvet ribbon" not in proof


def test_structured_answer_organization_suggestion_uses_context_generically(tmp_path):
    store = RecordStore(tmp_path / "organization-suggestion.sqlite")
    scope = Scope(namespace="organization-suggestion")
    rec = _record(
        "User: I need help organizing my drafting desk. I recently bought a brass tray to keep sketch pencils clutter-free.\n"
        "User: I noticed some wax marks on the oak desktop near the lamp.\n"
        "User: I keep a blue scarf by the hallway hooks.",
        scope=scope,
        valid_at=1_700_000_260,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "My workspace is becoming messy again. Any tips for keeping it tidy?",
        at=1_700_000_320,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:preference_synth:record:suggestion_synth"
    assert "brass tray" in ans.answer
    assert "sketch pencils clutter-free" in ans.answer
    assert "oak desktop near the lamp" in ans.answer
    assert "blue scarf" not in ans.answer
    proof = " ".join(c.snippet for c in ans.citations)
    assert "blue scarf" not in proof


def test_structured_answer_support_suggestion_uses_generic_tools(tmp_path):
    store = RecordStore(tmp_path / "support-suggestion.sqlite")
    scope = Scope(namespace="support-suggestion")
    rec = _record(
        "User: I am looking for advice on organizing printer maintenance accessories, like my silicone roller and paper guide kit.\n"
        "Assistant: Keep frequently used tools, like the printer and silicone roller, in the side drawer.\n"
        "User: I keep a picnic blanket in the hallway bin.",
        scope=scope,
        valid_at=1_700_000_280,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "I've been having trouble with the feed on my printer lately. Any tips?",
        at=1_700_000_340,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:preference_synth:record:suggestion_synth"
    assert "silicone roller" in ans.answer
    assert "paper guide kit" in ans.answer
    assert "picnic blanket" not in ans.answer
    proof = " ".join(c.snippet for c in ans.citations)
    assert "picnic blanket" not in proof


def test_structured_answer_inspiration_suggestion_uses_generic_sources(tmp_path):
    store = RecordStore(tmp_path / "inspiration-suggestion.sqlite")
    scope = Scope(namespace="inspiration-suggestion")
    rec = _record(
        "User: I've been looking at modular synth demos on Signal Garden for inspiration.\n"
        "User: I've been looking at some patching tutorials, but I'm not sure where to start.\n"
        "User: I have been getting inspiration from live looping forums and recently started a 10-day sound sketch challenge.\n"
        "User: I filed unrelated notes about a tax binder.",
        scope=scope,
        valid_at=1_700_000_300,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "I've been feeling stuck with my sound sketches lately. Any ideas for how I could find fresh inspiration?",
        at=1_700_000_360,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:preference_synth:record:suggestion_synth"
    assert "modular synth demos on Signal Garden" in ans.answer
    assert "patching tutorials" in ans.answer
    assert "live looping forums" in ans.answer
    assert "10-day sound sketch challenge" in ans.answer
    assert "tax binder" not in ans.answer
    proof = " ".join(c.snippet for c in ans.citations)
    assert "tax binder" not in proof


def test_structured_answer_social_connection_suggestions_are_group_neutral(tmp_path):
    store = RecordStore(tmp_path / "social-suggestion.sqlite")
    scope = Scope(namespace="social-suggestion")
    rec = _record(
        "User: I'm looking for suggestions on how to socialize with my remote poetry circle.\n"
        "Assistant: Suggestions to stay connected with the remote poetry circle: "
        "1. Voice Check-ins. 2. Shared Draft Nights. 3. Prompt Swaps. 4. Reading Rooms.\n"
        "User: I filed unrelated notes about a cable pouch.",
        scope=scope,
        valid_at=1_700_000_320,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "I want to stay connected with my remote poetry circle. Any suggestions?",
        at=1_700_000_380,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:preference_synth:record:suggestion_synth"
    assert "Voice Check-ins" in ans.answer
    assert "Shared Draft Nights" in ans.answer
    assert "Prompt Swaps" in ans.answer
    assert "Reading Rooms" in ans.answer
    assert "cable pouch" not in ans.answer
    proof = " ".join(c.snippet for c in ans.citations)
    assert "cable pouch" not in proof


def test_structured_answer_hobbies_are_extracted_from_generic_interest_phrases(tmp_path):
    store = RecordStore(tmp_path / "generic-hobbies.sqlite")
    scope = Scope(namespace="generic-hobbies")
    rows = [
        "Nila: My interests include copper sketching, quiet rowing, and prism baking.",
        "Theo: My interests include tax ledgers and alarm drills.",
        "Nila: I also enjoy moon gardening.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_340 + idx))

    ans = structured_answer(
        _Retriever(store),
        "What are Nila's interests?",
        at=1_700_000_400,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.answer == "Copper sketching, quiet rowing, prism baking, moon gardening"
    assert "tax ledgers" not in ans.answer
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Theo" not in proof


def test_structured_answer_awareness_event_uses_query_event_terms(tmp_path):
    store = RecordStore(tmp_path / "awareness-event.sqlite")
    scope = Scope(namespace="awareness-event")
    rows = [
        "User: I joined the Lantern Walk event for river safety last weekend.",
        "User: The Cedar Run event was for tax filing practice.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_360 + idx))

    ans = structured_answer(
        _Retriever(store),
        "What did the Lantern Walk event raise awareness for?",
        at=1_700_000_420,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.answer == "river safety"
    proof = " ".join(c.snippet for c in ans.citations)
    assert "tax filing" not in proof


def test_structured_answer_media_example_is_streaming_service_neutral(tmp_path):
    store = RecordStore(tmp_path / "media-example.sqlite")
    scope = Scope(namespace="media-example")
    rec = _record(
        'User: I wanted access to every season of archive shows. The example was "silver orchard" show, '
        "which later only had the final season available.",
        scope=scope,
        valid_at=1_700_000_380,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "Which streaming show did I use as the example with only the final season available?",
        at=1_700_000_440,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.answer == "Silver Orchard"


def test_structured_answer_affiliation_followup_is_not_team_specific(tmp_path):
    store = RecordStore(tmp_path / "affiliation-followup.sqlite")
    scope = Scope(namespace="affiliation-followup")
    rows = [
        (
            "Nila: I just joined a new research cohort - excited for the spring rotation!\n"
            "Nila: The Silver Orchard Lab! I start next week."
        ),
        (
            "Theo: I just joined a new research cohort too.\n"
            "Theo: The Brass Ledger Group! They start later."
        ),
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_390 + idx))

    ans = structured_answer(
        _Retriever(store),
        "Which research cohort did Nila join?",
        at=1_700_000_460,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.answer == "The Silver Orchard Lab"
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Brass Ledger" not in proof


def test_structured_answer_affiliation_requires_wh_target_or_action(tmp_path):
    """A what-question that merely CONTAINS an affiliation noun ('...did Vera's team perform...')
    is not an affiliation lookup: the cheer 'Go Dara!' must never surface as a verified answer."""
    store = RecordStore(tmp_path / "affiliation-wh-target.sqlite")
    scope = Scope(namespace="affiliation-wh-target")
    rows = [
        "Vera: My favorite moment was when my team took first place at sectionals.\n"
        "Vera: We just did a lyrical routine called Quiet Thunder.",
        "Vera: Go Dara!\nVera: I just got accepted for a textile internship!",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_500 + idx))

    ans = structured_answer(
        _Retriever(store),
        "What kind of routine did Vera's team perform to win first place?",
        at=1_700_000_900,
        scope=scope,
    )

    if ans is not None:
        assert "Go Dara" not in ans.answer
        assert "internship" not in ans.answer


def test_structured_answer_beverage_event_suggestion_uses_generic_context(tmp_path):
    store = RecordStore(tmp_path / "beverage-suggestion.sqlite")
    scope = Scope(namespace="beverage-suggestion")
    rec = _record(
        "User: I was thinking of making a mocktail for a gallery gathering.\n"
        "Assistant: Orchid Spark: A bright citrus beverage gets a thyme syrup finish.\n"
        "User: I liked the Orchid Spark, but I already made plain citrus drinks after a fermentation class.\n"
        "User: I think I'll try Lumen Citrus for the simple syrup.\n"
        "User: I think I'll try serving the Orchid Spark in a prism cup.\n"
        "User: I filed unrelated notes about a cedar binder.",
        scope=scope,
        valid_at=1_700_000_390,
    )
    store.upsert_record(rec)

    ans = structured_answer(
        _Retriever(store),
        "I'm choosing a beverage for a gallery gathering. Any suggestions?",
        at=1_700_000_450,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert "Orchid Spark" in ans.answer
    assert "fermentation class" in ans.answer
    assert "Lumen Citrus" in ans.answer
    proof = " ".join(c.snippet for c in ans.citations)
    assert "cedar binder" not in proof


def test_structured_answer_process_list_uses_source_list_not_fixed_domain_terms(tmp_path):
    store = RecordStore(tmp_path / "process-list.sqlite")
    scope = Scope(namespace="process-list")
    rows = [
        "Assistant: The obsidian workshop processes include ribbon curing, foil mapping, and lens sorting.",
        "Assistant: The cedar studio processes include wax batching and tray polishing.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_300 + idx))

    ans = structured_answer(
        _Retriever(store),
        "What processes does the obsidian workshop use?",
        at=1_700_000_400,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.answer == "ribbon curing, foil mapping, lens sorting"
    assert ans.note == "smqe:open_inference:record"
    proof = " ".join(c.snippet for c in ans.citations)
    assert "cedar studio" not in proof.lower()


def test_structured_answer_done_activity_list_is_not_fixed_vocabulary(tmp_path):
    store = RecordStore(tmp_path / "done-activity-list.sqlite")
    scope = Scope(namespace="done-activity-list")
    rows = [
        "Nila: I'm doing cyanotype printing and it is giving me ideas.",
        "Nila: I'm off to do some ceramic glazing!",
        "Rowan: I'm doing copper etching this week.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_420 + idx))

    ans = structured_answer(
        _Retriever(store),
        "What creative workshops has Nila done?",
        at=1_700_000_500,
        scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    assert ans.note == "smqe:open_inference:record"
    assert ans.answer == "Ceramic Glazing, Cyanotype Printing"
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Rowan" not in proof


def test_structured_answer_handles_randomized_unseen_record_questions(tmp_path):
    rng = random.Random(9731)
    names = ["Ari", "Nila", "Rowan", "Mika", "Tessa", "Owen"]
    objects = ["backup badge", "ceramic pass", "travel charger", "field notebook", "vault key", "garden permit"]
    locations = ["Quartz Loft", "North Pier Studio", "Cedar Annex", "Blue Finch Lab", "Orchid Room", "Silver Gate Gym"]

    for idx, (name, obj, location) in enumerate(zip(names, objects, locations)):
        store = RecordStore(tmp_path / f"slot-{idx}.sqlite")
        scope = Scope(namespace=f"random-slot-{idx}")
        rec = _record(f"{name}: I keep the {obj} at {location}.", scope=scope, valid_at=1_700_000_000 + idx)
        store.upsert_record(rec)

        ans = structured_answer(_Retriever(store), f"Where does {name} keep the {obj}?", at=rec.valid_at + 1, scope=scope)

        assert ans is not None
        assert ans.answer == location
        assert ans.verified is True
        assert ans.note == "smqe:latest_value:record"

    for idx, topic in enumerate(["kayak", "violin", "printer"]):
        store = RecordStore(tmp_path / f"sum-{idx}.sqlite")
        scope = Scope(namespace=f"random-sum-{idx}")
        amounts = [rng.randint(10, 90), rng.randint(10, 90), rng.randint(10, 90)]
        other_topic = {"kayak": "violin", "violin": "printer", "printer": "kayak"}[topic]
        texts = [
            f"User: I spent ${amounts[0]} on a {topic} repair.",
            f"User: I spent ${amounts[1]} on {topic} straps.",
            f"User: The {topic} case cost me ${amounts[2]}.",
            "User: I spent $999 on groceries this week.",
            f"User: I wrote a {topic} note that mentioned $300 as a fantasy budget.",
            f"User: The {other_topic} case cost me $77.",
        ]
        for offset, text in enumerate(texts):
            store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_100 + offset))

        out = _assert_aggregate_fails_closed(
            store, f"How much money in total have I spent on {topic}-related expenses?",
            at=1_700_000_200, scope=scope)
        assert out["answer"] == f"${sum(amounts):,}"
        assert out["note"] == "smqe:multi_session_sum:record"
        proof = _proof_of(out)
        assert "$999" not in proof
        assert "fantasy budget" not in proof
        assert f"{other_topic} case" not in proof

    store = RecordStore(tmp_path / "sum-numeric-suffix.sqlite")
    scope = Scope(namespace="sum-numeric-suffix")
    for text, valid_at in [
        ("User: I spent 4 hours on the harbor map 15.", 1_700_000_300),
        ("User: I spent 2 hours on the harbor map 15.", 1_700_000_310),
        ("User: I spent 9 hours on the mural ledger 15.", 1_700_000_320),
    ]:
        store.upsert_record(_record(text, scope=scope, valid_at=valid_at))

    out = _assert_aggregate_fails_closed(
        store, "How many total hours did I spend on the harbor map 15?",
        at=1_700_000_400, scope=scope)
    assert out["answer"] == "6 hours"
    assert "mural ledger" not in _proof_of(out)

    store = RecordStore(tmp_path / "sum-unseen-days.sqlite")
    scope = Scope(namespace="sum-unseen-days")
    for text, valid_at in [
        ("User: I spent a 3-day field survey on the quartz meadow index.", 1_700_000_500),
        ("User: I spent 2 days on the quartz meadow index follow-up.", 1_700_000_510),
        ("User: I spent 9 days on the cedar archive index.", 1_700_000_520),
        ("User: I never spent 7 days on the quartz meadow index backup.", 1_700_000_530),
    ]:
        store.upsert_record(_record(text, scope=scope, valid_at=valid_at))

    out = _assert_aggregate_fails_closed(
        store, "How many days did I spend on the quartz meadow index?",
        at=1_700_000_600, scope=scope)
    assert out["answer"] == "5 days"
    assert out["note"] == "smqe:count_aggregate:record"
    proof = _proof_of(out)
    assert "cedar archive" not in proof
    assert "never spent" not in proof

    for idx, item in enumerate(["linen order", "camera strap", "kiln shelf"]):
        store = RecordStore(tmp_path / f"relative-{idx}.sqlite")
        scope = Scope(namespace=f"random-relative-{idx}")
        valid_at = datetime(2024, 3, 10 + idx, 12, 0).timestamp()
        rec = _record(f"User: Yesterday I picked up the {item}.", scope=scope, valid_at=valid_at)
        store.upsert_record(rec)

        ans = structured_answer(_Retriever(store), f"When did I pick up the {item}?", at=valid_at + 1, scope=scope)

        assert ans is not None
        assert ans.answer == f"2024-03-{9 + idx:02d}"
        assert ans.verified is True
        assert ans.note == "smqe:relative_temporal:record:atom_derived"

    store = RecordStore(tmp_path / "count.sqlite")
    scope = Scope(namespace="random-count")
    studios = ["Cedar Wheel", "North Clay", "Kiln House", "River Glaze"]
    for idx, studio in enumerate(studios):
        store.upsert_record(_record(
            f"User: I visited {studio} pottery studio this month.",
            scope=scope,
            valid_at=1_700_000_300 + idx,
        ))
    for idx, text in enumerate([
        "User: I bookmarked a pottery studio directory this month.",
        "User: I visited the river museum this month.",
        "User: Studio 7 has a pottery mural.",
    ]):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_350 + idx))

    out = _assert_aggregate_fails_closed(
        store, "How many pottery studios did I visit this month?", at=1_700_000_400, scope=scope)
    assert out["answer"] == str(len(studios))
    assert out["note"] == "smqe:count_aggregate:record"
    proof = _proof_of(out)
    assert "bookmarked" not in proof
    assert "river museum" not in proof
    assert "Studio 7" not in proof


def test_structured_answer_count_matches_es_plural_to_singular_route(tmp_path):
    store = RecordStore(tmp_path / "count-route-plural.sqlite")
    scope = Scope(namespace="count-route-plural")
    routes = ["Cedar Loop", "North Pier", "Harbor Desk"]
    for idx, route in enumerate(routes):
        store.upsert_record(_record(
            f"User: I visited the {route} bike route this month.",
            scope=scope,
            valid_at=1_700_000_300 + idx,
        ))
    store.upsert_record(_record(
        "User: I visited 8 new museum exhibits this month.",
        scope=scope,
        valid_at=1_700_000_350,
    ))

    out = _assert_aggregate_fails_closed(
        store, "How many bike routes did I visit this month?", at=1_700_000_400, scope=scope)
    assert out["answer"] == str(len(routes))
    assert out["note"].startswith("smqe:count_aggregate:record")
    assert "museum exhibits" not in _proof_of(out)


def test_structured_answer_count_computes_subject_item_lists(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"count-subject-list-{backend}.sqlite")
        scope = Scope(namespace=f"count-subject-list-{backend}")
        rec = _record(
            "User: The release blockers are auth, billing, and search.",
            scope=scope,
            valid_at=1_700_000_300,
        )
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))

        out = _assert_aggregate_fails_closed(
            store, "How many release blockers are there?", at=1_700_000_400, scope=scope)
        assert out["answer"] == "3 release blockers: auth; billing; search"
        assert out["note"].startswith(f"smqe:count_aggregate:{backend}")


def test_structured_answer_count_computes_generic_action_item_lists(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"count-action-list-{backend}.sqlite")
        scope = Scope(namespace=f"count-action-list-{backend}")
        rec = _record(
            "User: I bought apples, oranges, and pears at the market.",
            scope=scope,
            valid_at=1_700_000_300,
        )
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))

        out = _assert_aggregate_fails_closed(
            store, "How many items did I buy?", at=1_700_000_400, scope=scope)
        assert out["answer"] == "3 items: apples; oranges; pears"
        assert out["note"].startswith(f"smqe:count_aggregate:{backend}")


def test_structured_answer_count_extracts_unseen_action_from_query(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"count-dynamic-action-{backend}.sqlite")
        scope = Scope(namespace=f"count-dynamic-action-{backend}")
        rows = [
            "User: I read the book Dune this month.",
            "User: I read the book Neuromancer this month.",
            "User: I did not read the book Foundation this month.",
            "User: I watched the film Arrival this month.",
        ]
        for idx, text in enumerate(rows):
            rec = _record(text, scope=scope, valid_at=1_700_000_300 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        out = _assert_aggregate_fails_closed(
            store, "How many books did I read this month?", at=1_700_000_400, scope=scope)
        assert out["answer"] == "2"
        assert out["note"].startswith(f"smqe:count_aggregate:{backend}")
        proof = _proof_of(out)
        assert "Foundation" not in proof
        assert "Arrival" not in proof


def test_structured_answer_count_rejects_instead_of_action_mentions(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"count-instead-action-{backend}.sqlite")
        scope = Scope(namespace=f"count-instead-action-{backend}")
        rows = [
            "User: I called client Northstar about renewal.",
            "User: I called client Bluecap about onboarding.",
            "User: I emailed client Cedarline instead of calling.",
        ]
        for idx, text in enumerate(rows):
            rec = _record(text, scope=scope, valid_at=1_700_000_300 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        out = _assert_aggregate_fails_closed(
            store, "How many clients did I call?", at=1_700_000_400, scope=scope)
        assert out["answer"] == "2"
        assert out["note"].startswith(f"smqe:count_aggregate:{backend}")
        assert "Cedarline" not in _proof_of(out)


def test_structured_answer_dynamic_action_count_requires_target_evidence(tmp_path):
    store = RecordStore(tmp_path / "count-dynamic-target-guard.sqlite")
    scope = Scope(namespace="count-dynamic-target-guard")
    for idx, text in enumerate([
        "User: I read Dune this month.",
        "User: I read Neuromancer this month.",
    ]):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_300 + idx))

    ans = structured_answer(
        _Retriever(store),
        "How many books did I read this month?",
        at=1_700_000_400,
        scope=scope,
    )

    assert ans is None


def test_structured_answer_count_handles_checked_out_action_without_directory_decoy(tmp_path):
    store = RecordStore(tmp_path / "count-checked-out.sqlite")
    scope = Scope(namespace="count-checked-out")
    studios = ["Cedar Wheel", "North Clay", "River Glaze"]
    for idx, studio in enumerate(studios):
        store.upsert_record(_record(
            f"User: I checked out the {studio} ceramic studio this month.",
            scope=scope,
            valid_at=1_700_000_300 + idx,
        ))
    store.upsert_record(_record(
        "User: I bookmarked a directory of ceramic studios.",
        scope=scope,
        valid_at=1_700_000_350,
    ))

    out = _assert_aggregate_fails_closed(
        store, "What is the number of ceramic studios I checked out this month?",
        at=1_700_000_400, scope=scope)
    assert out["answer"] == str(len(studios))
    assert "directory" not in _proof_of(out)


def test_structured_answer_count_uses_counted_entity_not_just_action_or_plural(tmp_path):
    store = RecordStore(tmp_path / "explicit-count-distractor.sqlite")
    scope = Scope(namespace="count-distractor")
    store.upsert_record(_record(
        "User: Have you tried any good blue loom studios in your city lately? "
        "I've tried four different ones so far.",
        scope=scope,
        valid_at=1_700_000_100,
    ))
    store.upsert_record(_record(
        "User: Have you tried any good copper garden studios lately? "
        "I've tried nine different ones recently.",
        scope=scope,
        valid_at=1_700_000_200,
    ))

    out = _assert_aggregate_fails_closed(
        store, "How many blue loom studios have I tried in my city?",
        at=1_700_000_300, scope=scope)
    assert out["answer"] == "four"
    proof = _proof_of(out)
    assert "blue loom studios" in proof
    assert "copper garden" not in proof
    assert "nine" not in proof


def test_structured_answer_count_rejects_same_record_action_target_mismatch(tmp_path):
    store = RecordStore(tmp_path / "count-action-target-mismatch.sqlite")
    scope = Scope(namespace="count-action-target-mismatch")
    store.upsert_record(_record(
        "User: I bookmarked a list of blue loom studios. I tried seven new hiking trails this month.",
        scope=scope,
        valid_at=datetime(2024, 6, 4, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "How many blue loom studios have I tried?",
        at=datetime(2024, 6, 4, 12, 1).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_count_rejects_negated_action_quantity(tmp_path):
    store = RecordStore(tmp_path / "count-negated-action.sqlite")
    scope = Scope(namespace="count-negated-action")
    store.upsert_record(_record(
        "User: The directory lists 6 different tea shops, but I have not visited them.",
        scope=scope,
        valid_at=datetime(2024, 6, 4, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "How many tea shops did I visit this month?",
        at=datetime(2024, 6, 4, 12, 1).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_count_rejects_recently_as_bare_anaphora(tmp_path):
    store = RecordStore(tmp_path / "count-recently-not-anaphora.sqlite")
    scope = Scope(namespace="count-recently-not-anaphora")
    store.upsert_record(_record(
        "User: Blue loom studios are popular in my city. I tried nine copper garden studios recently.",
        scope=scope,
        valid_at=datetime(2024, 6, 4, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "How many blue loom studios have I tried in my city?",
        at=datetime(2024, 6, 4, 12, 1).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_count_respects_rolling_temporal_windows(tmp_path):
    ref = datetime(2024, 6, 30, 12, 0)
    rows = [
        ("User: I visited the Cedar tea shop.", ref - timedelta(days=2)),
        ("User: I visited the Harbor tea shop.", ref - timedelta(days=5)),
        ("User: I visited the Maple tea shop.", ref - timedelta(days=20)),
    ]
    questions = [
        "How many tea shops did I visit recently?",
        "How many tea shops did I visit in the past week?",
    ]
    for question in questions:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"count-window-{backend}-{abs(hash(question))}.sqlite")
            scope = Scope(namespace=f"count-window-{backend}-{abs(hash(question))}")
            for text, valid_at in rows:
                rec = _record(text, scope=scope, valid_at=valid_at.timestamp())
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            out = _assert_aggregate_fails_closed(store, question, at=ref.timestamp(), scope=scope)
            assert out["note"].startswith(f"smqe:count_aggregate:{backend}")
            assert out["answer"] == "2"
            assert "Maple" not in _proof_of(out)


def test_structured_answer_sum_respects_rolling_temporal_window(tmp_path):
    ref = datetime(2024, 6, 30, 12, 0)
    rows = [
        ("User: I spent 2 hours on the cedar audit.", ref - timedelta(days=2)),
        ("User: I spent 3 hours on the harbor audit.", ref - timedelta(days=4)),
        ("User: I spent 9 hours on the maple audit.", ref - timedelta(days=40)),
    ]
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"sum-window-{backend}.sqlite")
        scope = Scope(namespace=f"sum-window-{backend}")
        for text, valid_at in rows:
            rec = _record(text, scope=scope, valid_at=valid_at.timestamp())
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        out = _assert_aggregate_fails_closed(
            store, "What is the total number of hours I spent on audits recently?",
            at=ref.timestamp(), scope=scope)
        assert out["note"] == f"smqe:multi_session_sum:{backend}"
        assert out["answer"] == "5 hours"
        assert "maple audit" not in _proof_of(out)


def test_structured_answer_plural_list_respects_rolling_temporal_window(tmp_path):
    ref = datetime(2024, 6, 30, 12, 0)
    rows = [
        ("User: I visited the Cedar tea shop.", ref - timedelta(days=2)),
        ("User: I visited the Harbor tea shop.", ref - timedelta(days=5)),
        ("User: I visited the Maple tea shop.", ref - timedelta(days=20)),
    ]
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"list-window-{backend}.sqlite")
        scope = Scope(namespace=f"list-window-{backend}")
        for text, valid_at in rows:
            rec = _record(text, scope=scope, valid_at=valid_at.timestamp())
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        ans = structured_answer(
            _Retriever(store),
            "Which tea shops did I visit in the past week?",
            at=ref.timestamp(),
            scope=scope,
        )

        assert ans is not None
        assert ans.note == f"smqe:latest_value:{backend}"
        assert ans.verified is True
        assert ans.answer == "Cedar tea shop and Harbor tea shop"
        proof = " ".join(c.snippet for c in ans.citations)
        assert "Maple" not in proof


def test_structured_answer_most_recently_returns_latest_object_only(tmp_path):
    ref = datetime(2024, 6, 30, 12, 0)
    rows = [
        ("User: I visited the Cedar tea shop.", ref - timedelta(days=2)),
        ("User: I visited the Harbor tea shop.", ref - timedelta(days=5)),
        ("User: I visited the Maple tea shop.", ref - timedelta(days=20)),
    ]
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"most-recent-object-{backend}.sqlite")
        scope = Scope(namespace=f"most-recent-object-{backend}")
        for text, valid_at in rows:
            rec = _record(text, scope=scope, valid_at=valid_at.timestamp())
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        ans = structured_answer(
            _Retriever(store),
            "What tea shop did I visit most recently?",
            at=ref.timestamp(),
            scope=scope,
        )

        assert ans is not None
        assert ans.note == f"smqe:latest_value:{backend}"
        assert ans.verified is True
        assert ans.answer == "Cedar tea shop"
        proof = " ".join(c.snippet for c in ans.citations)
        assert "Harbor" not in proof
        assert "Maple" not in proof


def test_structured_answer_source_location_respects_temporal_window_with_pronoun(tmp_path):
    ref = datetime(2024, 6, 30, 12, 0)
    rows = [
        (
            "User: I bought a copper notebook for travel notes. I got it from Cedar Market.",
            ref - timedelta(days=2),
        ),
        (
            "User: I bought a copper notebook for travel notes. "
            "I got it from Maple Archive after asking where to get the copper notebook.",
            ref - timedelta(days=20),
        ),
    ]
    for question in (
        "Where did I get the copper notebook recently?",
        "Where did I get the copper notebook in the past week?",
    ):
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"source-window-{backend}-{abs(hash(question))}.sqlite")
            scope = Scope(namespace=f"source-window-{backend}-{abs(hash(question))}")
            for text, valid_at in rows:
                rec = _record(text, scope=scope, valid_at=valid_at.timestamp())
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            ans = structured_answer(_Retriever(store), question, at=ref.timestamp(), scope=scope)

            assert ans is not None
            # the op may be latest_value or speaker_fact depending on the verb ("who told me"
            # routes to speaker-attributed recall); the BACKEND pin is what this test owns
            assert ans.note.startswith("smqe:") and ans.note.endswith(f":{backend}")
            assert ans.verified is True
            assert ans.answer == "Cedar Market"
            proof = " ".join(c.snippet for c in ans.citations)
            assert "Maple Archive" not in proof


def test_structured_answer_pick_up_location_is_recall_not_preference(tmp_path):
    ref = datetime(2024, 6, 30, 12, 0)
    rows = [
        (
            "User: I ordered a brass field compass for hiking. "
            "I picked it up at Cedar Outfitters.",
            ref - timedelta(days=2),
        ),
        (
            "User: I ordered a brass field compass for hiking. "
            "I picked it up at Maple Archive after asking where to get the brass field compass.",
            ref - timedelta(days=20),
        ),
    ]
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"pickup-location-{backend}.sqlite")
        scope = Scope(namespace=f"pickup-location-{backend}")
        for text, valid_at in rows:
            rec = _record(text, scope=scope, valid_at=valid_at.timestamp())
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        ans = structured_answer(
            _Retriever(store),
            "Where did I pick up the brass field compass recently?",
            at=ref.timestamp(),
            scope=scope,
        )

        assert ans is not None
        assert ans.note == f"smqe:latest_value:{backend}"
        assert ans.verified is True
        assert ans.answer == "Cedar Outfitters"
        proof = " ".join(c.snippet for c in ans.citations)
        assert "Maple Archive" not in proof


def test_structured_answer_who_attribution_returns_actor_and_skips_negated_distractor(tmp_path):
    cases = [
        (
            "Who recommended Cedar Cafe?",
            "Mira: I recommend Cedar Cafe for brunch.",
            "Nolan: Cedar Cafe is near the station but I did not recommend it.",
            "Mira",
            "Nolan",
        ),
        (
            "Who gave me the brass compass?",
            "Ari: I gave you the brass compass after the hike.",
            "Nila: I borrowed the brass compass but did not give it to you.",
            "Ari",
            "Nila",
        ),
        (
            "Who told me about Harbor Books?",
            "Tessa: I told you about Harbor Books yesterday.",
            "Omar: Harbor Books has a sale, but I never told you about it.",
            "Tessa",
            "Omar",
        ),
    ]
    for question, positive, negative, expected, forbidden in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"who-attribution-{backend}-{abs(hash(question))}.sqlite")
            scope = Scope(namespace=f"who-attribution-{backend}-{abs(hash(question))}")
            for idx, text in enumerate((positive, negative)):
                rec = _record(text, scope=scope, valid_at=1_700_000_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            ans = structured_answer(_Retriever(store), question, at=1_800_000_000, scope=scope)

            assert ans is not None
            # the op may be latest_value or speaker_fact depending on the verb ("who told me"
            # routes to speaker-attributed recall); the BACKEND pin is what this test owns
            assert ans.note.startswith("smqe:") and ans.note.endswith(f":{backend}")
            assert ans.verified is True
            assert ans.answer == expected
            proof = " ".join(c.snippet for c in ans.citations)
            assert forbidden not in proof


def test_structured_answer_count_allows_anaphoric_same_record_target_bridge(tmp_path):
    store = RecordStore(tmp_path / "count-anaphoric-target.sqlite")
    scope = Scope(namespace="count-anaphoric-target")
    store.upsert_record(_record(
        "User: Have you tried any good blue loom studios in your city lately? I have tried four different ones so far.",
        scope=scope,
        valid_at=datetime(2024, 6, 4, 12, 0).timestamp(),
    ))

    out = _assert_aggregate_fails_closed(
        store, "How many blue loom studios have I tried?",
        at=datetime(2024, 6, 4, 12, 1).timestamp(), scope=scope)
    assert out["answer"] == "four"
    proof = _proof_of(out)
    assert "blue loom studios" in proof
    assert "four different ones" in proof


def test_structured_answer_randomized_count_rejects_same_record_target_decoys(tmp_path):
    cases = [
        ("ceramic studios", "hiking trails", "seven"),
        ("blue loom studios", "garden plots", "six"),
        ("library workshops", "bike routes", "five"),
        ("coffee shops", "museum exhibits", "eight"),
    ]
    for idx, (target, decoy, number) in enumerate(cases):
        store = RecordStore(tmp_path / f"count-context-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"count-context-fuzz-{idx}")
        store.upsert_record(_record(
            f"User: I saved a list of {target}. I tried {number} new {decoy} this month.",
            scope=scope,
            valid_at=datetime(2024, 6, 4, 12, idx).timestamp(),
        ))

        ans = structured_answer(
            _Retriever(store),
            f"How many {target} have I tried?",
            at=datetime(2024, 6, 4, 13, 0).timestamp(),
            scope=scope,
        )

        assert ans is None


def test_structured_answer_model_kit_count_splits_generic_scale_models(tmp_path):
    store = RecordStore(tmp_path / "model-kit-generic.sqlite")
    scope = Scope(namespace="model-kit-generic")
    rows = [
        "User: I recently finished a simple Orion Falcon glider kit from the hobby store.",
        "User: I recently finished a 1/48 scale Harbor tug boat and had to learn new techniques.",
        "User: I started working on a diorama featuring a 1/16 scale Alpine tram vehicle.",
        "User: I just got this 1/72 scale lunar rover model kit and a 1/24 scale metro bus at a model show.",
        "User: My laptop model number is printed on the receipt.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_420 + idx))

    out = _assert_aggregate_fails_closed(
        store, "How many model kits have I bought or worked on?",
        at=1_700_000_500, scope=scope)
    assert out["answer"].startswith("5 model kits:")
    assert "Orion Falcon glider kit" in out["answer"]
    assert "1/48 scale Harbor tug boat" in out["answer"]
    assert "1/16 scale Alpine tram vehicle" in out["answer"]
    assert "1/72 scale lunar rover" in out["answer"]
    assert "1/24 scale metro bus" in out["answer"]
    assert "had to learn" not in out["answer"]
    assert "laptop" not in _proof_of(out)


def test_structured_answer_itemized_count_splits_arbitrary_target_lists(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"itemized-count-{backend}.sqlite")
        scope = Scope(namespace=f"itemized-count-{backend}")
        rows = [
            "User: I bought a crimson linen fabric swatch and a blue wool fabric swatch at the market.",
            "User: I bookmarked a directory of fabric swatches, but I did not buy from it.",
            "User: I bought a cedar kiln token for a different project.",
        ]
        for idx, text in enumerate(rows):
            rec = _record(text, scope=scope, valid_at=1_700_000_500 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        out = _assert_aggregate_fails_closed(
            store, "How many fabric swatches did I buy?", at=1_700_000_600, scope=scope)
        assert out["answer"].startswith("2 fabric swatches:")
        assert "crimson linen fabric swatch" in out["answer"]
        assert "blue wool fabric swatch" in out["answer"]
        assert "directory" not in out["answer"]
        assert "kiln token" not in out["answer"]
        assert out["note"].startswith(f"smqe:count_aggregate:{backend}")
        proof = _proof_of(out)
        assert "directory" not in proof
        assert "kiln token" not in proof


def test_structured_answer_acquired_item_count_is_domain_neutral(tmp_path):
    store = RecordStore(tmp_path / "acquired-item-count.sqlite")
    scope = Scope(namespace="acquired-item-count")
    rows = [
        "User: My blue awl, which I got from the tool library last month along with a copper clamp.",
        "User: My brass workshop supply bin, which I got from the studio last month, is now under the bench.",
        "User: My coffee mug, which I got last month, is beside the bench.",
        "User: I polished a steel ruler but did not acquire it.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_620 + idx))

    out = _assert_aggregate_fails_closed(
        store, "How many workshop supplies did I acquire last month?",
        at=1_700_000_700, scope=scope)
    assert out["answer"].startswith("3 workshop supplies:")
    assert "blue awl" in out["answer"]
    assert "copper clamp" in out["answer"]
    assert "brass workshop supply bin" in out["answer"]
    assert "coffee mug" not in out["answer"]
    assert "steel ruler" not in out["answer"]
    proof = _proof_of(out)
    assert "coffee mug" not in proof
    assert "steel ruler" not in proof


def test_structured_answer_scalar_amount_uses_governing_phrase_and_recency(tmp_path):
    store = RecordStore(tmp_path / "scalar-governed-money.sqlite")
    scope = Scope(namespace="scalar-governed-money")
    rows = [
        (
            "User: I'm buying a $325,000 cottage, and I got pre-approved for $350,000 "
            "from Blue Harbor Credit Union.",
            datetime(2023, 8, 11, 7, 1).timestamp(),
        ),
        (
            "User: I later got pre-approved for $400,000 from Blue Harbor Credit Union.",
            datetime(2023, 11, 30, 8, 36).timestamp(),
        ),
    ]
    for text, valid_at in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=valid_at))

    ans = structured_answer(
        _Retriever(store),
        "What amount was I pre-approved for when I took out my mortgage from Blue Harbor Credit Union?",
        at=datetime(2023, 12, 18, 12, 17).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "$400,000"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "$325,000 cottage" not in proof


def test_structured_answer_money_sum_parses_text_number_amounts(tmp_path):
    questions = [
        "How much did I spend on coffee altogether?",
        "How much have I paid for coffee overall?",
        "How much did coffee cost me overall?",
    ]
    for question in questions:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"text-money-sum-{backend}-{abs(hash(question))}.sqlite")
            scope = Scope(namespace=f"text-money-sum-{backend}-{abs(hash(question))}")
            rows = [
                "User: I paid five dollars for coffee beans.",
                "User: I paid seven bucks for coffee filters.",
                "User: I paid three dollars for tea tins.",
            ]
            for idx, text in enumerate(rows):
                rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            out = _assert_aggregate_fails_closed(store, question, at=1_700_001_100, scope=scope)
            assert out["answer"] == "$12"
            assert out["note"] == f"smqe:multi_session_sum:{backend}"
            assert "tea tins" not in _proof_of(out)


def test_structured_answer_duration_sum_ignores_negated_work_duration(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"duration-sum-negated-work-{backend}.sqlite")
        scope = Scope(namespace=f"duration-sum-negated-work-{backend}")
        rows = [
            "User: I worked 2 hours on the atlas migration.",
            "User: I worked 3 hours on the atlas migration follow-up.",
            "User: I did not work 7 hours on the atlas migration backup.",
        ]
        for idx, text in enumerate(rows):
            rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        out = _assert_aggregate_fails_closed(
            store, "How many hours did I work on the atlas migration total?",
            at=1_700_001_100, scope=scope)
        assert out["answer"] == "5 hours"
        assert out["note"] == f"smqe:multi_session_sum:{backend}"
        assert "did not work" not in _proof_of(out)


def test_structured_answer_generic_quantity_sum_handles_unseen_units(tmp_path):
    cases = [
        (
            "miles",
            [
                "User: I ran 3 miles on the river loop.",
                "User: I ran 4 miles on the cedar trail.",
                "User: I did not run 9 miles on the treadmill.",
                "User: I biked 10 miles on Sunday.",
            ],
            "How many miles did I run total?",
            "7 miles",
            ("treadmill", "biked"),
        ),
        (
            "pages",
            [
                "User: I revised 12 pages of the launch memo.",
                "User: I revised 8 pages of the launch memo appendix.",
                "User: I skimmed 40 pages of a novel.",
            ],
            "How many pages did I revise in total?",
            "20 pages",
            ("novel",),
        ),
        (
            "reps",
            [
                "User: I completed 15 reps of squats.",
                "User: I completed 20 reps of lunges.",
                "User: I planned 50 reps but did not complete them.",
            ],
            "How many reps did I complete total?",
            "35 reps",
            ("planned 50",),
        ),
    ]
    for name, rows, question, expected, forbidden in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"generic-unit-sum-{name}-{backend}.sqlite")
            scope = Scope(namespace=f"generic-unit-sum-{name}-{backend}")
            for idx, text in enumerate(rows):
                rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            out = _assert_aggregate_fails_closed(store, question, at=1_700_001_100, scope=scope)
            assert out["answer"] == expected
            assert out["note"] == f"smqe:multi_session_sum:{backend}"
            proof = _proof_of(out)
            for bad in forbidden:
                assert bad not in proof


def test_structured_answer_sum_abstains_when_generic_unit_values_missing(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"generic-unit-sum-missing-{backend}.sqlite")
        scope = Scope(namespace=f"generic-unit-sum-missing-{backend}")
        for idx, text in enumerate([
            "User: I ran the river loop.",
            "User: I ran the cedar trail.",
            "User: I planned 9 miles but never ran them.",
        ]):
            rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        ans = structured_answer(
            _Retriever(store),
            "How many miles did I run total?",
            at=1_700_001_100,
            scope=scope,
        )

        assert ans is None


def test_structured_answer_numeric_extreme_compares_same_unit_values(tmp_path):
    cases = [
        (
            "longest-run",
            [
                "User: I ran the river loop for 3 miles.",
                "User: I ran the cedar trail for 5 miles.",
                "User: I did not run the mountain loop for 10 miles.",
                "User: I biked the quarry path for 8 miles.",
            ],
            "Which run was longest?",
            "cedar trail (5 miles)",
            ("mountain loop", "quarry path"),
        ),
        (
            "lowest-score",
            [
                "User: My quiz score in algebra was 88 points.",
                "User: My quiz score in biology was 72 points.",
                "User: My quiz score in chemistry was 95 points.",
            ],
            "What was my lowest quiz score?",
            "72 points",
            (),
        ),
        (
            "highest-cost",
            [
                "User: The oak desk cost $120.",
                "User: The lamp cost $45.",
                "User: The chair cost $200.",
            ],
            "Which purchase cost the most?",
            "chair ($200)",
            (),
        ),
        (
            "shortest-commute",
            [
                "User: The blue commute took 18 minutes.",
                "User: The green commute took 12 minutes.",
                "User: The red commute took 30 minutes.",
            ],
            "Which commute was shortest?",
            "green commute (12 minutes)",
            (),
        ),
    ]
    for name, rows, question, expected, forbidden in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"numeric-extreme-{name}-{backend}.sqlite")
            scope = Scope(namespace=f"numeric-extreme-{name}-{backend}")
            for idx, text in enumerate(rows):
                rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            ans = structured_answer(_Retriever(store), question, at=1_700_001_100, scope=scope)

            assert ans is not None
            assert ans.answer == expected
            assert ans.verified is True
            assert ans.note == f"smqe:latest_value:{backend}"
            proof = " ".join(c.snippet for c in ans.citations)
            for bad in forbidden:
                assert bad not in proof


def test_structured_answer_numeric_extreme_abstains_on_ambiguous_units(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"numeric-extreme-ambiguous-{backend}.sqlite")
        scope = Scope(namespace=f"numeric-extreme-ambiguous-{backend}")
        rows = [
            "User: The alpha drill used 5 clamps and 3 bolts.",
            "User: The beta drill used 4 clamps and 9 bolts.",
        ]
        for idx, text in enumerate(rows):
            rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        ans = structured_answer(_Retriever(store), "Which drill had the most?", at=1_700_001_100, scope=scope)

        assert ans is None


def test_structured_answer_numeric_difference_computes_two_anchored_values(tmp_path):
    cases = [
        (
            "cost-more",
            [
                "User: The oak desk cost $120.",
                "User: The lamp cost $45.",
                "User: The chair cost $200.",
            ],
            "How much more did the chair cost than the lamp?",
            "$155",
            ("oak desk",),
        ),
        (
            "cost-less",
            [
                "User: The oak desk cost $120.",
                "User: The lamp cost $45.",
                "User: The chair cost $200.",
            ],
            "How much more did the lamp cost than the chair?",
            "$155 less",
            ("oak desk",),
        ),
        (
            "score-difference",
            [
                "User: My quiz score in algebra was 88 points.",
                "User: My quiz score in biology was 72 points.",
                "User: My quiz score in chemistry was 95 points.",
            ],
            "What was the difference between my algebra and biology quiz scores?",
            "16 points",
            ("chemistry",),
        ),
        (
            "miles-more",
            [
                "User: I ran the river loop for 3 miles.",
                "User: I ran the cedar trail for 5 miles.",
                "User: I biked the quarry path for 8 miles.",
            ],
            "How many more miles was the cedar trail than the river loop?",
            "2 miles",
            ("quarry path",),
        ),
    ]
    for name, rows, question, expected, forbidden in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"numeric-difference-{name}-{backend}.sqlite")
            scope = Scope(namespace=f"numeric-difference-{name}-{backend}")
            for idx, text in enumerate(rows):
                rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            ans = structured_answer(_Retriever(store), question, at=1_700_001_100, scope=scope)

            assert ans is not None
            assert ans.answer == expected
            assert ans.verified is True
            assert ans.note.endswith(f":{backend}")
            assert len(ans.citations) == 2
            proof = " ".join(c.snippet for c in ans.citations)
            for bad in forbidden:
                assert bad not in proof


def test_structured_answer_numeric_difference_abstains_on_ambiguous_units_or_missing_anchor(tmp_path):
    cases = [
        (
            "ambiguous-units",
            [
                "User: The alpha drill used 5 clamps and 3 bolts.",
                "User: The beta drill used 4 clamps and 9 bolts.",
            ],
            "What was the difference between alpha drill and beta drill?",
        ),
        (
            "missing-anchor",
            [
                "User: The alpha drill used 5 clamps.",
                "User: The beta drill used 4 clamps.",
            ],
            "What was the difference between alpha drill and gamma drill?",
        ),
    ]
    for name, rows, question in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"numeric-difference-abstain-{name}-{backend}.sqlite")
            scope = Scope(namespace=f"numeric-difference-abstain-{name}-{backend}")
            for idx, text in enumerate(rows):
                rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            ans = structured_answer(_Retriever(store), question, at=1_700_001_100, scope=scope)

            assert ans is None


def test_structured_answer_numeric_average_computes_same_unit_values(tmp_path):
    cases = [
        (
            "quiz-score",
            [
                "User: My quiz score in algebra was 88 points.",
                "User: My quiz score in biology was 72 points.",
                "User: My quiz score in chemistry was 95 points.",
            ],
            "What was my average quiz score?",
            "85 points",
            3,
            (),
        ),
        (
            "run-distance",
            [
                "User: I ran the river loop for 3 miles.",
                "User: I ran the cedar trail for 5 miles.",
                "User: I biked the quarry path for 8 miles.",
            ],
            "What was my average run distance?",
            "4 miles",
            2,
            ("quarry path",),
        ),
        (
            "purchase-cost",
            [
                "User: The oak desk cost $120.",
                "User: The lamp cost $45.",
                "User: The chair cost $200.",
            ],
            "What was the average purchase cost?",
            "$121.67",
            3,
            (),
        ),
    ]
    for name, rows, question, expected, citation_count, forbidden in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"numeric-average-{name}-{backend}.sqlite")
            scope = Scope(namespace=f"numeric-average-{name}-{backend}")
            for idx, text in enumerate(rows):
                rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            # P0 fail-closed extension (adversarial review 2026-07-09): a derived multi-atom
            # AVERAGE is a value NO source states -- the same verified-wrong class as derived
            # counts/sums (a mistagged latest_value average shipped '$60' verified from atoms
            # stating only $80 and $40). The derivation still computes (trace below); the
            # verify-or-abstain surface withholds the badge.
            out = _assert_aggregate_fails_closed(store, question, at=1_700_001_100, scope=scope)
            assert out["answer"] == expected
            assert out["note"] == f"smqe:latest_value:{backend}"
            proof = _proof_of(out)
            for bad in forbidden:
                assert bad not in proof


def test_structured_answer_numeric_average_abstains_without_clear_same_unit_set(tmp_path):
    cases = [
        (
            "mixed-units",
            [
                "User: The alpha drill used 5 clamps and 3 bolts.",
                "User: The beta drill used 4 clamps and 9 bolts.",
            ],
            "What was the average drill amount?",
        ),
        (
            "single-value",
            [
                "User: My quiz score in algebra was 88 points.",
                "User: My quiz notes covered two chapters.",
            ],
            "What was my average quiz score?",
        ),
    ]
    for name, rows, question in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"numeric-average-abstain-{name}-{backend}.sqlite")
            scope = Scope(namespace=f"numeric-average-abstain-{name}-{backend}")
            for idx, text in enumerate(rows):
                rec = _record(text, scope=scope, valid_at=1_700_001_000 + idx)
                store.upsert_record(rec)
                if add_claims:
                    store.add_claims(claims_for_record(rec))

            ans = structured_answer(_Retriever(store), question, at=1_700_001_100, scope=scope)

            assert ans is None


def test_structured_answer_duration_sum_ignores_non_distance_durations(tmp_path):
    store = RecordStore(tmp_path / "duration-sum-distractor.sqlite")
    scope = Scope(namespace="duration-sum-distractor")
    rows = [
        "User: My recent trip to Harbor Point took four hours to drive there from my place.",
        "Assistant: Marigold Bay is a beach town with an old lighthouse. (~5-6 hours from home)",
        "User: I drove for six hours to Capital City recently.",
        "User: I spent two hours packing snacks for the trip.",
        "User: The museum visit lasted seven hours.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_000_500 + idx))

    out = _assert_aggregate_fails_closed(
        store, "How many total hours did I spend driving to reach my three road trip destinations?",
        at=1_700_000_600, scope=scope)
    assert out["answer"] == "15 hours for getting to the three destinations (or 30 hours for the round trip)"
    proof = _proof_of(out)
    assert "packing snacks" not in proof
    assert "museum visit" not in proof


def test_structured_answer_temporal_delta_matches_both_query_anchors(tmp_path):
    store = RecordStore(tmp_path / "temporal-anchor-distractors.sqlite")
    scope = Scope(namespace="temporal-anchor-distractors")
    rows = [
        ("User: I started calibrating the greenhouse sensor today.", datetime(2024, 4, 1, 12, 0).timestamp()),
        ("User: I started calibrating the hallway sensor today.", datetime(2024, 4, 3, 12, 0).timestamp()),
        ("User: I finished installing the bookshelf controller today.", datetime(2024, 4, 6, 12, 0).timestamp()),
        ("User: I finished installing the irrigation controller today.", datetime(2024, 4, 8, 12, 0).timestamp()),
    ]
    for text, valid_at in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=valid_at))

    ans = structured_answer(
        _Retriever(store),
        "How many days passed between when I started calibrating the greenhouse sensor "
        "and when I finished installing the irrigation controller?",
        at=datetime(2024, 4, 10, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "7 days"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "greenhouse sensor" in proof
    assert "irrigation controller" in proof
    assert "hallway sensor" not in proof
    assert "bookshelf controller" not in proof


def test_structured_answer_temporal_delta_abstains_when_between_anchor_missing(tmp_path):
    store = RecordStore(tmp_path / "temporal-between-missing-anchor.sqlite")
    scope = Scope(namespace="temporal-between-missing-anchor")
    store.upsert_record(_record(
        "User: I opened the field notebook today.",
        scope=scope,
        valid_at=datetime(2024, 4, 1, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "How many days passed between opening the field notebook and closing the backup badge?",
        at=datetime(2024, 4, 10, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_temporal_delta_single_anchor_uses_question_time(tmp_path):
    store = RecordStore(tmp_path / "temporal-single-anchor.sqlite")
    scope = Scope(namespace="temporal-single-anchor")
    store.upsert_record(_record(
        "User: I picked up the ceramic kit today.",
        scope=scope,
        valid_at=datetime(2024, 1, 1, 12, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "User: I cleaned the ceramic kit today.",
        scope=scope,
        valid_at=datetime(2024, 1, 4, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "How many days ago did I pick up the ceramic kit?",
        at=datetime(2024, 1, 10, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "9"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "picked up the ceramic kit" in proof
    assert "cleaned the ceramic kit" not in proof


def test_structured_answer_temporal_delta_single_anchor_verb_family(tmp_path):
    """'How many days ago did I BUY X?' must anchor on 'I just GOT X today' (acquisition
    synonym family), even when higher-scoring atoms match only operator words like 'days'."""
    store = RecordStore(tmp_path / "temporal-verb-family.sqlite")
    scope = Scope(namespace="temporal-verb-family")
    event_day = datetime(2024, 3, 5, 12, 0)
    question_day = datetime(2024, 3, 15, 12, 0)
    store.upsert_record(_record(
        "User: I just got a kiln today and I'm excited to try slow firing.",
        scope=scope,
        valid_at=event_day.timestamp(),
    ))
    distractors = [
        "User: We rented a cabin and spent two days hiking in the hills.",
        "User: I've been into photography lately, especially since I got my new tripod last month.",
        "User: I want to buy a good quality duffel bag that will last a long time.",
        "User: I'll adjust my bedtime by fifteen minutes every few days.",
        "User: I was buying groceries when the rain started, days of drizzle ahead.",
    ]
    for idx, text in enumerate(distractors):
        store.upsert_record(_record(text, scope=scope, valid_at=event_day.timestamp() + (idx + 1) * 86_400))

    ans = structured_answer(
        _Retriever(store),
        "How many days ago did I buy the kiln?",
        at=question_day.timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "10"
    proof = " ".join(c.snippet for c in ans.citations)
    assert "kiln" in proof
    assert "tripod" not in proof
    assert "hiking" not in proof


def test_structured_answer_temporal_delta_abstains_without_topical_anchor(tmp_path):
    """No memory mentions the queried action/object -> abstain. Never compute a delta between
    two arbitrary dated atoms and ship it with citations."""
    store = RecordStore(tmp_path / "temporal-no-anchor.sqlite")
    scope = Scope(namespace="temporal-no-anchor")
    store.upsert_record(_record(
        "User: We spent two days repainting the porch.",
        scope=scope,
        valid_at=datetime(2024, 5, 1, 9, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "User: I'll stretch every few days after the morning run.",
        scope=scope,
        valid_at=datetime(2024, 5, 3, 15, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "How many days ago did I submit the transfer form?",
        at=datetime(2024, 5, 9, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_randomized_temporal_delta_single_anchor_distractors(tmp_path):
    cases = [
        ("picked up", "cleaned", "ceramic kit", 9),
        ("filed", "reviewed", "garden permit", 12),
        ("installed", "repaired", "garage charger", 5),
        ("booked", "packed", "museum ticket", 17),
    ]
    for idx, (target_action, decoy_action, obj, days_ago) in enumerate(cases):
        store = RecordStore(tmp_path / f"temporal-single-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"temporal-single-fuzz-{idx}")
        as_of = datetime(2024, 2, 20, 12, 0)
        target_at = as_of - timedelta(days=days_ago)
        decoy_at = as_of - timedelta(days=max(1, days_ago - 3))
        store.upsert_record(_record(
            f"User: I {target_action} the {obj} today.",
            scope=scope,
            valid_at=target_at.timestamp(),
        ))
        store.upsert_record(_record(
            f"User: I {decoy_action} the {obj} today.",
            scope=scope,
            valid_at=decoy_at.timestamp(),
        ))

        ans = structured_answer(
            _Retriever(store),
            f"How many days ago did I {target_action} the {obj}?",
            at=as_of.timestamp(),
            scope=scope,
        )

        assert ans is not None
        assert ans.answer == str(days_ago)
        proof = " ".join(c.snippet for c in ans.citations)
        assert target_action in proof
        assert decoy_action not in proof


def test_numeric_extreme_who_which_answers_name_the_subject():
    """A who/which-<entity> superlative must name the SUBJECT of the winning measurement.
    Labeling the adjacent time adverbial ('Wednesday (9 miles)' as the friend) is a
    verified-wrong machine; a bare measurement for a who-question is shape-wrong."""
    from eidetic.smqe.record_ops import _numeric_extreme_answer

    def rec(i):
        return type("R", (), {"valid_at": float(i), "memory_id": f"m{i}"})()

    atoms = [
        (1.0, rec(1), "My friend Jonas ran 9 miles on Wednesday."),
        (0.9, rec(2), "My friend Priya ran 6 miles on Saturday."),
    ]
    ans, sel = _numeric_extreme_answer("Which friend ran the longest distance?", atoms)
    assert "Jonas" in ans and "Wednesday" not in ans

    ans2, _ = _numeric_extreme_answer("Who ran the longest distance among my friends?", atoms)
    assert "Jonas" in ans2

    # leading time adverbial must not shadow the subject
    atoms3 = [
        (1.0, rec(1), "On Wednesday, Jonas ran 9 miles."),
        (0.9, rec(2), "Priya ran 6 miles."),
    ]
    ans3, _ = _numeric_extreme_answer("Which friend ran the longest distance?", atoms3)
    assert ans3 == "" or ("Jonas" in ans3 and "Wednesday" not in ans3)

    # first-person atoms cannot name a third-party subject -> fail closed, never a bare guess
    atoms4 = [
        (1.0, rec(1), "I ran 9 miles on Wednesday."),
        (0.9, rec(2), "I ran 6 miles on Saturday."),
    ]
    ans4, sel4 = _numeric_extreme_answer("Which friend ran the longest distance?", atoms4)
    assert (ans4, sel4) == ("", [])

    # time-word wh-heads keep the temporal label
    ans5, _ = _numeric_extreme_answer("Which day did I run the longest distance?", [
        (1.0, rec(1), "I ran the longest distance, 9 miles, on Wednesday."),
        (0.9, rec(2), "I ran a distance of 6 miles on Saturday."),
    ])
    assert ans5 == "" or "Wednesday" in ans5


def test_count_answer_never_reads_calendar_clock_or_money_tokens():
    """A count extractor that returns a year, clock time, or price as a cardinality is a
    verified-wrong machine: the atom is quotable so the wrong number verifies."""
    from eidetic.smqe.record_ops import _count_answer

    assert _count_answer("I visited the dentist twice in 2023.") == "twice"
    assert _count_answer("In 2023 I visited the museum with my cousin.") == ""
    assert _count_answer("The race finished at 10:45 after two laps.") == "two"
    assert _count_answer("I paid $30 for three tickets.") == "three"
    assert _count_answer("On March 3, 2024 we planted trees.") == ""
    assert _count_answer("I own 12 paintbrushes.") == "12"


def test_structured_answer_yes_no_proposition_confirmation(tmp_path):
    """A yes/no question whose proposition is literally stated in memory answers
    'Yes - <premise>' anchored on the stating atom, instead of failing to the reader."""
    store = RecordStore(tmp_path / "yesno-confirmation.sqlite")
    scope = Scope(namespace="yesno-confirmation")
    store.upsert_record(_record(
        "User: By the way, my aunt is actually using the same meal planning app as me now, "
        "so we can share templates.",
        scope=scope,
        valid_at=1_700_000_100,
    ))
    store.upsert_record(_record(
        "User: I reorganized the pantry shelves over the weekend.",
        scope=scope,
        valid_at=1_700_000_200,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Is my aunt using the same meal planning method as me?",
        at=1_700_000_900,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer.lower().startswith("yes")
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "meal planning app" in proof


def test_structured_answer_yes_no_confirmation_needs_stated_proposition(tmp_path):
    """No stored assertion covers the proposition -> no structured yes (falls to the reader);
    a negated assertion must never surface as 'Yes'."""
    store = RecordStore(tmp_path / "yesno-no-proposition.sqlite")
    scope = Scope(namespace="yesno-no-proposition")
    store.upsert_record(_record(
        "User: My aunt is not using the shared meal planning app anymore.",
        scope=scope,
        valid_at=1_700_000_100,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Is my aunt using the same meal planning method as me?",
        at=1_700_000_900,
        scope=scope,
    )

    if ans is not None:
        assert not ans.answer.lower().startswith("yes")


def test_structured_answer_yes_no_negative_assertion_answers_no(tmp_path):
    """Antimemory: a stored NEGATED assertion of the proposition answers 'No - <premise>'
    anchored on the negating atom, instead of falling to the reader."""
    store = RecordStore(tmp_path / "yesno-negative.sqlite")
    scope = Scope(namespace="yesno-negative")
    store.upsert_record(_record(
        "User: I've never been to a jazz festival, honestly.",
        scope=scope,
        valid_at=1_700_000_100,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Have I ever been to a jazz festival?",
        at=1_700_000_900,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer.lower().startswith("no")
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "jazz festival" in proof


def test_structured_answer_yes_no_retraction_latest_assertion_wins(tmp_path):
    """Retraction: when a proposition is asserted and later negated, the LATEST assertion wins;
    a later re-assertion flips it back."""
    store = RecordStore(tmp_path / "yesno-retraction.sqlite")
    scope = Scope(namespace="yesno-retraction")
    store.upsert_record(_record(
        "User: I signed up for weekly pottery lessons at the studio.",
        scope=scope,
        valid_at=1_700_000_100,
    ))
    store.upsert_record(_record(
        "User: I dropped the pottery lessons, not doing them anymore.",
        scope=scope,
        valid_at=1_700_050_000,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Am I taking pottery lessons?",
        at=1_700_099_000,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer.lower().startswith("no")
    proof = " ".join(c.snippet for c in ans.citations)
    assert "dropped the pottery lessons" in proof

    store2 = RecordStore(tmp_path / "yesno-retraction2.sqlite")
    scope2 = Scope(namespace="yesno-retraction2")
    store2.upsert_record(_record(
        "User: I dropped the pottery lessons, not doing them anymore.",
        scope=scope2,
        valid_at=1_700_000_100,
    ))
    store2.upsert_record(_record(
        "User: Good news - I restarted my weekly pottery lessons at the studio.",
        scope=scope2,
        valid_at=1_700_050_000,
    ))

    ans2 = structured_answer(
        _Retriever(store2),
        "Am I taking pottery lessons?",
        at=1_700_099_000,
        scope=scope2,
    )

    assert ans2 is not None
    assert ans2.answer.lower().startswith("yes")
    proof2 = " ".join(c.snippet for c in ans2.citations)
    assert "restarted" in proof2


def test_structured_answer_latest_value_explicit_date_uses_last_night(tmp_path):
    """A question naming an explicit calendar day must anchor on the atom datable to that day:
    'last night' said in a May 4 session resolves to May 3, while undatable or other-day
    chatter is filtered out instead of feeding the slot extractor."""
    store = RecordStore(tmp_path / "explicit-date-last-night.sqlite")
    scope = Scope(namespace="explicit-date-last-night")
    store.upsert_record(_record(
        "Rhea: My mom and I cooked some pasta for dinner together last night!",
        scope=scope,
        valid_at=datetime(2023, 5, 4, 22, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "Rhea: Back in my school days, my family drove out to Halden on a long road trip.",
        scope=scope,
        valid_at=datetime(2023, 6, 12, 9, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "Rhea: I'm heading out for dinner with some friends from the gym.",
        scope=scope,
        valid_at=datetime(2023, 7, 8, 18, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "Who did Rhea have dinner with on May 3, 2023?",
        at=datetime(2023, 8, 16, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert "mom" in ans.answer.lower()
    proof = " ".join(c.snippet for c in ans.citations)
    assert "dinner together last night" in proof
    assert "road trip" not in proof
    assert "friends from the gym" not in proof


def test_structured_answer_latest_value_explicit_date_no_match_abstains(tmp_path):
    """An explicit-date question with no atom datable to that day must abstain, not answer
    from an atom on a different day."""
    store = RecordStore(tmp_path / "explicit-date-no-match.sqlite")
    scope = Scope(namespace="explicit-date-no-match")
    store.upsert_record(_record(
        "Rhea: I'm heading out for dinner with some friends from the gym.",
        scope=scope,
        valid_at=datetime(2023, 7, 8, 18, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "Who did Rhea have dinner with on May 3, 2023?",
        at=datetime(2023, 8, 16, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_latest_value_matches_named_subject(tmp_path):
    store = RecordStore(tmp_path / "latest-subject-distractor.sqlite")
    scope = Scope(namespace="latest-subject-distractor")
    store.upsert_record(_record(
        "Ari: I keep the backup badge at Quartz Loft.",
        scope=scope,
        valid_at=1_700_000_100,
    ))
    store.upsert_record(_record(
        "Nila: I keep the backup badge at North Pier Studio.",
        scope=scope,
        valid_at=1_700_000_200,
    ))

    ans = structured_answer(_Retriever(store), "Where does Ari keep the backup badge?", at=1_700_000_300, scope=scope)

    assert ans is not None
    assert ans.answer == "Quartz Loft"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Ari" in proof
    assert "Nila" not in proof


def test_structured_answer_latest_value_ignores_future_intent_for_current_question(tmp_path):
    store = RecordStore(tmp_path / "latest-future-intent.sqlite")
    scope = Scope(namespace="latest-future-intent")
    store.upsert_record(_record(
        "Ari: I keep the backup badge at Cedar Annex.",
        scope=scope,
        valid_at=1_700_000_100,
    ))
    store.upsert_record(_record(
        "Ari: I moved the backup badge to Quartz Loft.",
        scope=scope,
        valid_at=1_700_000_200,
    ))
    store.upsert_record(_record(
        "Ari: I will move the backup badge to Orchid Room.",
        scope=scope,
        valid_at=1_700_000_300,
    ))

    ans = structured_answer(_Retriever(store), "Where does Ari keep the backup badge now?", at=1_700_000_400, scope=scope)

    assert ans is not None
    assert ans.answer == "Quartz Loft"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Quartz Loft" in proof
    assert "Cedar Annex" not in proof
    assert "Orchid Room" not in proof


def test_structured_answer_latest_value_abstains_without_named_subject_support(tmp_path):
    store = RecordStore(tmp_path / "latest-missing-subject.sqlite")
    scope = Scope(namespace="latest-missing-subject")
    store.upsert_record(_record(
        "Nila: I keep the backup badge at North Pier Studio.",
        scope=scope,
        valid_at=1_700_000_200,
    ))

    ans = structured_answer(_Retriever(store), "Where does Ari keep the backup badge?", at=1_700_000_300, scope=scope)

    assert ans is None


def test_structured_answer_latest_value_requires_answer_atom_target_support(tmp_path):
    store = RecordStore(tmp_path / "latest-context-distractor.sqlite")
    scope = Scope(namespace="latest-context-distractor")
    store.upsert_record(_record(
        "User: I asked about the pottery wheel. My gym pass is at Quartz Loft.",
        scope=scope,
        valid_at=datetime(2024, 6, 3, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "Where is my pottery wheel?",
        at=datetime(2024, 6, 3, 12, 1).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_latest_value_allows_anaphoric_same_record_bridge(tmp_path):
    store = RecordStore(tmp_path / "latest-anaphoric-bridge.sqlite")
    scope = Scope(namespace="latest-anaphoric-bridge")
    store.upsert_record(_record(
        "User: I checked the pottery wheel after class. It is at Quartz Loft.",
        scope=scope,
        valid_at=datetime(2024, 6, 3, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "Where is my pottery wheel?",
        at=datetime(2024, 6, 3, 12, 1).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "Quartz Loft"
    assert ans.verified is True


def test_structured_answer_randomized_latest_value_rejects_same_record_decoys(tmp_path):
    cases = [
        ("ceramic wheel", "library card", "North Pier Studio"),
        ("backup charger", "gym pass", "Quartz Loft"),
        ("field notebook", "museum ticket", "Cedar Annex"),
        ("garden permit", "studio badge", "Blue Finch Lab"),
    ]
    for idx, (target, decoy, location) in enumerate(cases):
        store = RecordStore(tmp_path / f"latest-context-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"latest-context-fuzz-{idx}")
        store.upsert_record(_record(
            f"User: I was thinking about the {target}. My {decoy} is at {location}.",
            scope=scope,
            valid_at=datetime(2024, 6, 3, 12, idx).timestamp(),
        ))

        ans = structured_answer(
            _Retriever(store),
            f"Where is my {target}?",
            at=datetime(2024, 6, 3, 13, 0).timestamp(),
            scope=scope,
        )

        assert ans is None


def test_structured_answer_relative_temporal_matches_named_subject(tmp_path):
    store = RecordStore(tmp_path / "relative-subject-distractor.sqlite")
    scope = Scope(namespace="relative-subject-distractor")
    store.upsert_record(_record(
        "Ari: Yesterday I picked up the ceramic kit.",
        scope=scope,
        valid_at=datetime(2024, 1, 10, 12, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "Nila: Yesterday I picked up the ceramic kit from the downtown studio.",
        scope=scope,
        valid_at=datetime(2024, 1, 15, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "When did Ari pick up the ceramic kit?",
        at=datetime(2024, 1, 16, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "2024-01-09"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Ari" in proof
    assert "Nila" not in proof


def test_structured_answer_relative_temporal_abstains_without_named_subject_support(tmp_path):
    store = RecordStore(tmp_path / "relative-missing-subject.sqlite")
    scope = Scope(namespace="relative-missing-subject")
    store.upsert_record(_record(
        "Nila: Yesterday I picked up the ceramic kit from the downtown studio.",
        scope=scope,
        valid_at=datetime(2024, 1, 15, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "When did Ari pick up the ceramic kit?",
        at=datetime(2024, 1, 16, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_table_lookup_uses_row_and_column_keys(tmp_path):
    store = RecordStore(tmp_path / "table-row-column.sqlite")
    scope = Scope(namespace="table-row-column")
    store.upsert_record(_record(
        "| Name | Sunday | Monday |\n"
        "| Mira | 9 AM | off |\n"
        "| Nila | off | 2 PM |",
        scope=scope,
        valid_at=1_700_000_500,
    ))

    ans = structured_answer(
        _Retriever(store),
        "What shift does Mira have on Sunday in the schedule?",
        at=1_700_000_600,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "9 AM"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Mira" in proof
    assert "Sunday" in proof


def test_structured_answer_table_lookup_abstains_when_named_row_missing(tmp_path):
    store = RecordStore(tmp_path / "table-missing-row.sqlite")
    scope = Scope(namespace="table-missing-row")
    store.upsert_record(_record(
        "| Name | Sunday |\n"
        "| Rowan | 7 AM |",
        scope=scope,
        valid_at=1_700_000_500,
    ))

    ans = structured_answer(
        _Retriever(store),
        "What shift does Sana have on Sunday in the schedule?",
        at=1_700_000_600,
        scope=scope,
    )

    assert ans is None


def test_structured_answer_table_lookup_can_return_row_label_for_column_query(tmp_path):
    store = RecordStore(tmp_path / "table-column-person.sqlite")
    scope = Scope(namespace="table-column-person")
    store.upsert_record(_record(
        "| Name | Sunday | Monday |\n"
        "| Mira | 9 AM | off |\n"
        "| Nila | off | 2 PM |",
        scope=scope,
        valid_at=1_700_000_500,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Who works Sunday in the schedule?",
        at=1_700_000_600,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "Mira"
    assert ans.verified is True
    assert "Nila" not in " ".join(c.snippet for c in ans.citations)


def test_structured_answer_randomized_table_lookup_row_column_decoys(tmp_path):
    cases = [
        ("Ari", "Tuesday", "7 AM", "Lina", "11 AM", "Thursday", "3 PM"),
        ("Mika", "Friday", "late", "Owen", "early", "Monday", "midday"),
        ("Sana", "Wednesday", "north desk", "Theo", "south desk", "Sunday", "west desk"),
        ("Nia", "Saturday", "2 PM", "Rowan", "4 PM", "Tuesday", "8 AM"),
    ]
    for idx, (person, day, expected, other, same_day_decoy, other_day, same_row_decoy) in enumerate(cases):
        store = RecordStore(tmp_path / f"table-row-col-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"table-row-col-fuzz-{idx}")
        store.upsert_record(_record(
            f"| Name | {day} | {other_day} |\n"
            f"| {other} | {same_day_decoy} | off |\n"
            f"| {person} | {expected} | {same_row_decoy} |",
            scope=scope,
            valid_at=1_700_002_000 + idx,
        ))

        ans = structured_answer(
            _Retriever(store),
            f"What shift does {person} have on {day} in the schedule?",
            at=1_700_002_100,
            scope=scope,
        )

        assert ans is not None
        assert ans.answer == expected
        assert ans.verified is True
        assert same_day_decoy not in ans.answer
        assert same_row_decoy not in ans.answer
        proof = " ".join(c.snippet for c in ans.citations)
        assert person in proof
        assert day in proof
        assert other not in proof


def test_structured_answer_randomized_table_lookup_column_query_ignores_off_rows(tmp_path):
    cases = [
        ("Tuesday", "Ari", "9 AM", "Nila"),
        ("Friday", "Mika", "late", "Owen"),
        ("Sunday", "Sana", "north desk", "Theo"),
        ("Thursday", "Nia", "2 PM", "Rowan"),
    ]
    for idx, (day, person, value, off_person) in enumerate(cases):
        store = RecordStore(tmp_path / f"table-column-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"table-column-fuzz-{idx}")
        store.upsert_record(_record(
            f"| Name | {day} | Backup |\n"
            f"| {off_person} | off | standby |\n"
            f"| {person} | {value} | n/a |",
            scope=scope,
            valid_at=1_700_002_500 + idx,
        ))

        ans = structured_answer(
            _Retriever(store),
            f"Who works {day} in the schedule?",
            at=1_700_002_600,
            scope=scope,
        )

        assert ans is not None
        assert ans.answer == person
        assert ans.verified is True
        proof = " ".join(c.snippet for c in ans.citations)
        assert person in proof
        assert off_person not in proof


def test_structured_answer_preference_choice_rejects_negative_option_evidence(tmp_path):
    store = RecordStore(tmp_path / "preference-negative-option.sqlite")
    scope = Scope(namespace="preference-negative-option")
    rows = [
        "User: I dislike coffee because it makes me jittery.",
        "User: I enjoy tea in the afternoon.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_003_000 + idx))

    ans = structured_answer(
        _Retriever(store),
        "Would I prefer coffee or tea?",
        at=1_700_003_100,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "tea"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "enjoy tea" in proof
    assert "dislike coffee" not in proof


def test_structured_answer_preference_choice_abstains_when_only_negative_option_matches(tmp_path):
    store = RecordStore(tmp_path / "preference-only-negative.sqlite")
    scope = Scope(namespace="preference-only-negative")
    store.upsert_record(_record(
        "User: I dislike coffee because it makes me jittery.",
        scope=scope,
        valid_at=1_700_003_100,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Would I prefer coffee or tea?",
        at=1_700_003_200,
        scope=scope,
    )

    assert ans is None


def test_structured_answer_preference_choice_uses_latest_polarity(tmp_path):
    store = RecordStore(tmp_path / "preference-latest-polarity.sqlite")
    scope = Scope(namespace="preference-latest-polarity")
    rows = [
        ("User: I enjoy mint tea after work.", 1_700_003_000),
        ("User: I avoid graphite pens before meetings.", 1_700_003_001),
        ("User: I avoid mint tea before meetings now.", 1_700_003_500),
        ("User: I enjoy graphite pens after work now.", 1_700_003_501),
    ]
    for text, valid_at in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=valid_at))

    ans = structured_answer(
        _Retriever(store),
        "Would I prefer mint tea or graphite pens?",
        at=1_700_003_600,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "graphite pens"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "enjoy graphite pens" in proof
    assert "enjoy mint tea" not in proof


def test_structured_answer_preference_choice_ignores_shared_option_terms(tmp_path):
    store = RecordStore(tmp_path / "preference-shared-option-terms.sqlite")
    scope = Scope(namespace="preference-shared-option-terms")
    rows = [
        "User: I enjoy mint tea 20 after work.",
        "User: I avoid cedar tea 20 before meetings.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_003_700 + idx))

    ans = structured_answer(
        _Retriever(store),
        "Would I prefer mint tea 20 or cedar tea 20?",
        at=1_700_003_800,
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "mint tea 20"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "enjoy mint tea 20" in proof
    assert "avoid cedar tea 20" not in proof


def test_structured_answer_randomized_preference_choice_uses_positive_over_negative(tmp_path):
    cases = [
        ("kale salad", "berry salad", "hate", "love"),
        ("cedar tea", "mint tea", "avoid", "enjoy"),
        ("brass pen", "graphite pen", "dislike", "prefer"),
        ("tax manuals", "fantasy novels", "hate", "like"),
    ]
    for idx, (bad_option, good_option, negative, positive) in enumerate(cases):
        store = RecordStore(tmp_path / f"preference-choice-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"preference-choice-fuzz-{idx}")
        store.upsert_record(_record(
            f"User: I {negative} {bad_option} during long work sessions.",
            scope=scope,
            valid_at=1_700_003_300 + idx * 10,
        ))
        store.upsert_record(_record(
            f"User: I {positive} {good_option} during long work sessions.",
            scope=scope,
            valid_at=1_700_003_301 + idx * 10,
        ))

        ans = structured_answer(
            _Retriever(store),
            f"Would I prefer {bad_option} or {good_option}?",
            at=1_700_003_400 + idx * 10,
            scope=scope,
        )

        assert ans is not None
        assert ans.answer == good_option
        proof = " ".join(c.snippet for c in ans.citations)
        assert good_option in proof
        assert bad_option not in proof


def test_structured_answer_preference_choice_does_not_infer_from_neutral_mentions(tmp_path):
    store = RecordStore(tmp_path / "preference-neutral-mention.sqlite")
    scope = Scope(namespace="preference-neutral-mention")
    rows = [
        "User: The pantry inventory lists coffee, filters, and mugs.",
        "User: I dislike tea because it upsets my stomach.",
    ]
    for idx, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=1_700_003_600 + idx))

    ans = structured_answer(
        _Retriever(store),
        "Would I prefer coffee or tea?",
        at=1_700_003_700,
        scope=scope,
    )

    assert ans is None


def test_structured_answer_preference_choice_not_like_is_negative_not_positive(tmp_path):
    store = RecordStore(tmp_path / "preference-not-like.sqlite")
    scope = Scope(namespace="preference-not-like")
    store.upsert_record(_record(
        "User: I do not like tea during work sessions.",
        scope=scope,
        valid_at=1_700_003_710,
    ))
    store.upsert_record(_record(
        "User: Coffee was mentioned in the office supply order.",
        scope=scope,
        valid_at=1_700_003_711,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Would I prefer coffee or tea?",
        at=1_700_003_800,
        scope=scope,
    )

    assert ans is None


def test_structured_answer_randomized_preference_choice_rejects_neutral_option_mentions(tmp_path):
    cases = [
        ("puzzle book", "garden book", "catalog"),
        ("mint tea", "cedar tea", "inventory"),
        ("blue notebook", "red notebook", "receipt"),
        ("quiet playlist", "brass playlist", "archive"),
    ]
    for idx, (neutral_option, negative_option, neutral_context) in enumerate(cases):
        store = RecordStore(tmp_path / f"preference-neutral-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"preference-neutral-fuzz-{idx}")
        store.upsert_record(_record(
            f"User: The {neutral_context} mentions {neutral_option} as a label.",
            scope=scope,
            valid_at=1_700_003_900 + idx * 10,
        ))
        store.upsert_record(_record(
            f"User: I avoid {negative_option} before meetings.",
            scope=scope,
            valid_at=1_700_003_901 + idx * 10,
        ))

        ans = structured_answer(
            _Retriever(store),
            f"Would I prefer {neutral_option} or {negative_option}?",
            at=1_700_004_000 + idx * 10,
            scope=scope,
        )

        assert ans is None


def test_structured_answer_event_order_uses_both_query_anchors(tmp_path):
    store = RecordStore(tmp_path / "event-order-distractors.sqlite")
    scope = Scope(namespace="event-order-distractors")
    rows = [
        ("User: I repaired the greenhouse sensor today.", datetime(2024, 4, 1, 12, 0).timestamp()),
        ("User: I started calibrating the greenhouse sensor today.", datetime(2024, 4, 5, 12, 0).timestamp()),
        ("User: I filed the garden permit today.", datetime(2024, 4, 3, 12, 0).timestamp()),
    ]
    for text, valid_at in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=valid_at))

    ans = structured_answer(
        _Retriever(store),
        "Which event happened first, starting to calibrate the greenhouse sensor or filing the garden permit?",
        at=datetime(2024, 4, 10, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "I filed the garden permit today"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "garden permit" in proof
    assert "repaired the greenhouse sensor" not in proof


def test_structured_answer_event_order_abstains_when_anchor_missing(tmp_path):
    store = RecordStore(tmp_path / "event-order-missing-anchor.sqlite")
    scope = Scope(namespace="event-order-missing-anchor")
    store.upsert_record(_record(
        "User: I repaired the greenhouse sensor today.",
        scope=scope,
        valid_at=datetime(2024, 4, 1, 12, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "User: I filed the garden permit today.",
        scope=scope,
        valid_at=datetime(2024, 4, 3, 12, 0).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "Which event happened first, starting to calibrate the greenhouse sensor or filing the garden permit?",
        at=datetime(2024, 4, 10, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_speaker_fact_requires_speaker_and_topic_same_support(tmp_path):
    store = RecordStore(tmp_path / "speaker-topic.sqlite")
    scope = Scope(namespace="speaker-topic")
    store.upsert_record(_record(
        "Mira: I asked about the kiln pickup window.",
        scope=scope,
        valid_at=datetime(2024, 6, 1, 12, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "Nila: I said the blue envelope stays in drawer four.",
        scope=scope,
        valid_at=datetime(2024, 6, 1, 12, 1).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "What did Mira say about the blue envelope?",
        at=datetime(2024, 6, 1, 12, 2).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_speaker_fact_rejects_partial_topic_overlap(tmp_path):
    store = RecordStore(tmp_path / "speaker-partial-topic.sqlite")
    scope = Scope(namespace="speaker-partial-topic")
    store.upsert_record(_record(
        "Nila: I said the backup badge 23 stays in the north pier studio.",
        scope=scope,
        valid_at=datetime(2024, 6, 1, 12, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "Mika: I said the studio key 23 stays in the cedar annex.",
        scope=scope,
        valid_at=datetime(2024, 6, 1, 12, 1).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "What did Nila say about the studio key 23?",
        at=datetime(2024, 6, 1, 12, 2).timestamp(),
        scope=scope,
    )

    assert ans is None


def test_structured_answer_speaker_fact_answers_matching_speaker_topic(tmp_path):
    store = RecordStore(tmp_path / "speaker-topic-positive.sqlite")
    scope = Scope(namespace="speaker-topic-positive")
    store.upsert_record(_record(
        "Nila: I said the blue envelope stays in drawer four.",
        scope=scope,
        valid_at=datetime(2024, 6, 1, 12, 0).timestamp(),
    ))
    store.upsert_record(_record(
        "Mira: I said the blue envelope stays in drawer nine.",
        scope=scope,
        valid_at=datetime(2024, 6, 1, 12, 1).timestamp(),
    ))

    ans = structured_answer(
        _Retriever(store),
        "What did Mira say about the blue envelope?",
        at=datetime(2024, 6, 1, 12, 2).timestamp(),
        scope=scope,
    )

    assert ans is not None
    assert ans.answer == "the blue envelope stays in drawer nine"
    assert ans.verified is True
    proof = " ".join(c.snippet for c in ans.citations)
    assert "Mira" in proof
    assert "drawer nine" in proof
    assert "drawer four" not in proof


def test_structured_answer_randomized_speaker_fact_distractors(tmp_path):
    cases = [
        ("Ari", "Mika", "silver badge", "locker two", "locker seven", "studio pass"),
        ("Rowan", "Nila", "travel receipt", "green folder", "red folder", "backup code"),
        ("Sana", "Omar", "porch key", "ceramic bowl", "brass hook", "museum ticket"),
        ("Theo", "Lina", "garden permit", "north tray", "south tray", "library card"),
    ]
    for idx, (speaker, other, topic, wanted, distractor, other_topic) in enumerate(cases):
        store = RecordStore(tmp_path / f"speaker-fuzz-{idx}.sqlite")
        scope = Scope(namespace=f"speaker-fuzz-{idx}")
        store.upsert_record(_record(
            f"{speaker}: I said the {other_topic} stays in the {wanted}.",
            scope=scope,
            valid_at=datetime(2024, 6, 2, 12, 0).timestamp(),
        ))
        store.upsert_record(_record(
            f"{other}: I said the {topic} stays in the {distractor}.",
            scope=scope,
            valid_at=datetime(2024, 6, 2, 12, 1).timestamp(),
        ))
        store.upsert_record(_record(
            f"{speaker}: I said the {topic} stays in the {wanted}.",
            scope=scope,
            valid_at=datetime(2024, 6, 2, 12, 2).timestamp(),
        ))

        ans = structured_answer(
            _Retriever(store),
            f"What did {speaker} say about the {topic}?",
            at=datetime(2024, 6, 2, 12, 3).timestamp(),
            scope=scope,
        )

        assert ans is not None
        assert wanted in ans.answer
        assert distractor not in ans.answer
        proof = " ".join(c.snippet for c in ans.citations)
        assert speaker in proof
        assert topic in proof
        assert other not in proof


def test_structured_answer_randomized_unsupported_inferences_abstain(tmp_path):
    for idx, name in enumerate(["Ari", "Nila", "Rowan", "Mika"]):
        store = RecordStore(tmp_path / f"abstain-{idx}.sqlite")
        scope = Scope(namespace=f"random-abstain-{idx}")
        rec = _record(
            f"{name}: My kids have so much already, so we donated extra toys this year.",
            scope=scope,
            valid_at=1_700_001_000 + idx,
        )
        store.upsert_record(rec)

        ans = structured_answer(_Retriever(store), f"What might {name}'s financial status be?", at=rec.valid_at + 1, scope=scope)

        assert ans is None


def test_smqe_verification_requires_every_declared_support(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="verify-all-supports")
    first = _record("User: The north sensor is calibrated.", scope=scope)
    second = _record("User: The east sensor is still offline.", scope=scope)
    store.upsert_record(first)
    store.upsert_record(second)
    result = StructuredAnswerResult(
        answer="Two sensors are ready.",
        op="count_aggregate",
        backend="record",
        confidence=0.9,
        supports=[
            StructuredSupport(memory_id=first.memory_id, proof_atom="User: The north sensor is calibrated."),
            StructuredSupport(memory_id=second.memory_id, proof_atom="User: The east sensor is calibrated."),
        ],
        note="smqe:count_aggregate:record",
    )

    ans = answer_from_result(_Retriever(store), "How many sensors are calibrated?", result, verify=True)

    assert ans is None


def test_smqe_verification_requires_support_records_to_exist(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="verify-missing-support")
    rec = _record("User: The kiln pickup was yesterday.", scope=scope)
    store.upsert_record(rec)
    result = StructuredAnswerResult(
        answer="2024-01-01",
        op="relative_temporal",
        backend="record",
        supports=[
            StructuredSupport(memory_id=rec.memory_id, proof_atom="User: The kiln pickup was yesterday."),
            StructuredSupport(memory_id="missing-memory", proof_atom="User: The glaze order was ready."),
        ],
        note="smqe:relative_temporal:record",
    )

    ans = answer_from_result(_Retriever(store), "When was the kiln pickup?", result, verify=True)

    assert ans is None


def test_structured_image_support_requires_pixel_verification(tmp_path):
    store = RecordStore(tmp_path / "structured-image-proof.sqlite")
    scope = Scope(namespace="structured-image-proof")
    record = MemoryRecord(
        text="The chart shows revenue increasing every quarter.",
        source="image",
        modality=Modality.IMAGE,
        scope=scope,
        valid_at=1_700_000_000.0,
        content_hash="image-hash",
        raw_uri="cas://image-hash",
    )
    store.upsert_record(record)
    result = StructuredAnswerResult(
        answer="Revenue increased every quarter.",
        op="latest_value",
        backend="record",
        supports=[StructuredSupport(
            memory_id=record.memory_id,
            proof_atom="The chart shows revenue increasing every quarter.",
        )],
        note="smqe:latest_value:record",
    )

    class _PixelReject(_Retriever):
        def __init__(self, source_store):
            super().__init__(source_store)
            self.calls = 0

        def verify_citation(self, rec, atom):
            self.calls += 1
            return NLILabel.NEUTRAL, 0.0

    retriever = _PixelReject(store)
    answer = answer_from_result(
        retriever,
        "What trend does the chart show?",
        result,
        verify=True,
    )

    assert answer is None
    assert retriever.calls == 1


class _StrictRetriever(_Retriever):
    """Claim-backend strict-hypothesis retriever: exposes .verify so answer_from_result takes
    the query-aware path; entailment is substring-only, like the base test retriever."""

    def verify(self, premise, hypothesis):
        return self.verify_citation(
            type("R", (), {"text": premise, "summary": ""})(), hypothesis)


def test_smqe_multi_support_anchor_exemption_needs_independent_witnesses(tmp_path):
    """Witness rule: the multi-support anchor-level verification exemption requires two
    INDEPENDENT source records. Two quotable atoms from the SAME record must not smuggle a
    derived answer past the strict query-aware hypothesis."""
    store = RecordStore(tmp_path / "witness-same-record.sqlite")
    scope = Scope(namespace="witness-same-record")
    rec = _record(
        "Vera: I just got accepted for a textile internship!\nVera: Go Dara!",
        scope=scope,
    )
    store.upsert_record(rec)
    result = StructuredAnswerResult(
        answer="Go Dara",
        op="open_inference",
        backend="claim",
        confidence=0.9,
        supports=[
            StructuredSupport(memory_id=rec.memory_id,
                              proof_atom="I just got accepted for a textile internship!"),
            StructuredSupport(memory_id=rec.memory_id, proof_atom="Go Dara!"),
        ],
        note="smqe:open_inference:claim",
    )

    ans = answer_from_result(
        _StrictRetriever(store),
        "What kind of routine did Vera's team perform to win first place?",
        result,
        verify=True,
    )

    assert ans is None


def test_smqe_multi_support_anchor_exemption_holds_for_independent_witnesses(tmp_path):
    """Two distinct source records each carrying a verbatim anchor keep the composed answer
    verifiable at anchor level (the honest standard for derived multi-support answers)."""
    store = RecordStore(tmp_path / "witness-independent.sqlite")
    scope = Scope(namespace="witness-independent")
    first = _record("User: I filed the garden permit on Monday.", scope=scope, valid_at=1_700_000_100)
    second = _record("User: I mailed the fee cheque on Thursday.", scope=scope, valid_at=1_700_050_000)
    store.upsert_record(first)
    store.upsert_record(second)
    result = StructuredAnswerResult(
        answer="the permit was filed before the cheque was mailed",
        op="open_inference",
        backend="claim",
        confidence=0.9,
        supports=[
            StructuredSupport(memory_id=first.memory_id,
                              proof_atom="I filed the garden permit on Monday."),
            StructuredSupport(memory_id=second.memory_id,
                              proof_atom="I mailed the fee cheque on Thursday."),
        ],
        note="smqe:open_inference:claim",
    )

    ans = answer_from_result(
        _StrictRetriever(store),
        "Did I file the garden permit before mailing the fee cheque?",
        result,
        verify=True,
    )

    assert ans is not None
    assert ans.verified is True


def test_smqe_verify_false_returns_unverified_compatibility_answer(tmp_path):
    store = RecordStore(tmp_path / "mem.sqlite")
    scope = Scope(namespace="verify-false-compat")
    rec = _record("User: The spare charger is in the teal pouch.", scope=scope)
    store.upsert_record(rec)
    result = StructuredAnswerResult(
        answer="teal pouch",
        op="latest_value",
        backend="record",
        confidence=0.8,
        supports=[StructuredSupport(memory_id=rec.memory_id, proof_atom="unsupported atom")],
        note="smqe:latest_value:record",
    )

    ans = answer_from_result(_Retriever(store), "Where is the spare charger?", result, verify=False)

    assert ans is not None
    assert ans.verified is False
    assert ans.confidence == 0.8


def test_claim_extraction_requires_verbatim_source_proof():
    rec = _record("User: Nila prefers matte black notebooks.", scope=Scope(namespace="claims"))
    valid = validate_extracted_claims(rec, [{
        "claim_type": "preference",
        "subject": "Nila",
        "predicate": "prefers",
        "object": "matte black notebooks",
        "proof_atom": "User: Nila prefers matte black notebooks.",
    }])
    invalid = validate_extracted_claims(rec, [{
        "claim_type": "preference",
        "subject": "Nila",
        "predicate": "prefers",
        "object": "orange notebooks",
        "proof_atom": "User: Nila prefers orange notebooks.",
    }])

    assert len(valid) == 1
    assert invalid == []


def test_claims_for_record_adds_sentence_claims_without_triples():
    rec = _record("User: Rowan's backup code is in the green envelope.", scope=Scope(namespace="claims"))
    claims = claims_for_record(rec, triples=[])

    assert claims
    assert claims[0].source_memory_id == rec.memory_id
    assert "green envelope" in claims[0].proof_atom


def test_claims_for_record_classifies_currency_word_amounts_as_quantity():
    rec = _record(
        "User: My repair budget is 425 dollars. User: The backup fund is 30 usd. User: I spent five bucks on tape.",
        scope=Scope(namespace="claims-money-words"),
    )
    claims = claims_for_record(rec, triples=[])

    quantity_claims = [claim for claim in claims if claim.claim_type == "quantity"]
    assert len(quantity_claims) == 3
    assert {claim.proof_atom for claim in quantity_claims} == {
        "User: My repair budget is 425 dollars.",
        "User: The backup fund is 30 usd.",
        "User: I spent five bucks on tape.",
    }


def test_planner_maps_generic_query_shapes():
    assert plan_query("How many weeks passed since the kiln pickup?").op == "temporal_delta"
    assert plan_query("What is my current permit status?").op == "latest_value"
    assert plan_query("Which row in the schedule has the late shift?").op == "table_lookup"
    assert plan_query("What shift does Ari have on Tuesday in the schedule?").op == "table_lookup"
    assert plan_query("When will I schedule the kiln checklist?").op == "relative_temporal"
    assert plan_query("Which happened first, filing the permit or calibrating the sensor?").op == "event_order"
    assert plan_query("What did Tessa say about the field notebook?").op == "speaker_fact"


def test_structured_answer_normalizes_source_relative_date_phrases(tmp_path):
    cases = [
        (
            "two-weeks",
            "User: Two weeks ago I picked up the cedar permit.",
            "When did I pick up the cedar permit?",
            "2024-05-06",
        ),
        (
            "fortnight",
            "User: A fortnight ago I mailed the orchid catalog.",
            "When did I mail the orchid catalog?",
            "2024-05-06",
        ),
        (
            "next-week",
            "User: Next week I will review the kiln checklist.",
            "When will I review the kiln checklist?",
            "the week of 2024-05-27 to 2024-06-02",
        ),
        (
            "next-month",
            "User: Next month I will schedule the harbor map.",
            "When will I schedule the harbor map?",
            "June 2024",
        ),
    ]
    for name, text, question, expected in cases:
        for add_claims, backend in ((False, "record"), (True, "claim")):
            store = RecordStore(tmp_path / f"relative-phrase-{name}-{backend}.sqlite")
            scope = Scope(namespace=f"relative-phrase-{name}-{backend}")
            rec = _record(text, scope=scope, valid_at=datetime(2024, 5, 20, 12, 0).timestamp())
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

            ans = structured_answer(_Retriever(store), question, at=1_900_000_000, scope=scope)

            assert ans is not None
            assert ans.note == f"smqe:relative_temporal:{backend}:atom_derived"
            assert ans.verified is True
            assert ans.answer == expected


def test_structured_answer_commonality_requires_both_named_people_and_excludes_distractor(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"commonality-{backend}.sqlite")
        scope = Scope(namespace=f"commonality-{backend}")
        rows = [
            "User: Ari unwinds by sketching maps after work.",
            "User: Nila unwinds by sketching maps after work.",
            "User: Theo unwinds by baking bread after work.",
        ]
        for idx, text in enumerate(rows):
            rec = _record(text, scope=scope, valid_at=1_700_000_000 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        ans = structured_answer(
            _Retriever(store),
            "What activity do Ari and Nila both use to unwind?",
            at=1_800_000_000,
            scope=scope,
        )

        assert ans is not None
        assert ans.note == f"smqe:open_inference:{backend}"
        assert ans.verified is True
        assert ans.answer == "sketching maps"
        proof = " ".join(c.snippet for c in ans.citations)
        assert "Theo" not in proof
        assert "baking bread" not in proof


def test_structured_answer_before_event_time_uses_clock_not_iso_date_fragment(tmp_path):
    for add_claims, backend in ((False, "record"), (True, "claim")):
        store = RecordStore(tmp_path / f"before-event-time-{backend}.sqlite")
        scope = Scope(namespace=f"before-event-time-{backend}")
        rows = [
            "User: On 2024-04-09 I went to bed at 10:15 PM.",
            "User: On 2024-04-10 I had a studio review at 6:30 AM.",
            "User: On 2024-04-11 I woke up at 6:00 AM.",
        ]
        for idx, text in enumerate(rows):
            rec = _record(text, scope=scope, valid_at=1_700_000_000 + idx)
            store.upsert_record(rec)
            if add_claims:
                store.add_claims(claims_for_record(rec))

        ans = structured_answer(
            _Retriever(store),
            "What time did I go to bed the day before my studio review?",
            at=1_800_000_000,
            scope=scope,
        )

        assert ans is not None
        assert ans.note == f"smqe:open_inference:{backend}"
        assert ans.verified is True
        assert ans.answer == "10:15 PM"
        proof = " ".join(c.snippet for c in ans.citations)
        assert "6:00 AM" not in proof
        assert "2024-04" in proof


def test_samples_file_filter_preserves_requested_order():
    from types import SimpleNamespace

    from bench.run import _filter_samples_file

    samples = [
        SimpleNamespace(dataset="alpha", sample_id="a1"),
        SimpleNamespace(dataset="alpha", sample_id="a2"),
        SimpleNamespace(dataset="beta", sample_id="b1"),
    ]
    rows = [{"dataset": "beta", "sample_id": "b1"}, {"dataset": "alpha", "sample_id": "a1"}]

    assert [s.sample_id for s in _filter_samples_file(samples, rows)] == ["b1", "a1"]


def test_holdout_audit_finds_banned_strings(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text('["secret_case_001"]')
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    (root / "x.py").write_text('NOTE = "secret_case_001"')

    result = audit(holdout, [root], include_legacy_policy=False)

    assert result["pass"] is False
    assert result["findings"][0]["needle"] == "secret_case_001"


def test_holdout_audit_exemption_registry_is_pair_exact_and_reported(tmp_path):
    """The exemption registry sanctions EXACT (needle, path) pairs -- forensic references
    and symmetric scoring policy (judge-v2 quarantine) with a reason + evidence pointer.
    Every used exemption is REPORTED; the SAME needle in any unregistered file still fails;
    the registry file itself is the one structurally skipped path."""
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    # a REAL registered pair: the quarantine id inside the registered report tool path
    (holdout / "leaked_sample_ids.json").write_text('["c9_q137"]')
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    import shutil
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parent.parent
    root = tmp_path / "bench"
    root.mkdir()
    shutil.copy(repo / "bench" / "notebooklm_freetier_report.py",
                root / "notebooklm_freetier_report.py")
    # same needle in an UNREGISTERED file -> still a finding
    (root / "rogue.py").write_text('SAMPLE = "c9_q137"\n')

    result = audit(holdout, [root], include_legacy_policy=False)

    # rogue.py fails; the registered pair does not, and its use is visible
    assert result["pass"] is False
    finding_paths = {f["path"] for f in result["findings"]}
    assert any(p.endswith("rogue.py") for p in finding_paths)
    assert not any(p.endswith("notebooklm_freetier_report.py") for p in finding_paths)
    assert any(e["needle"] == "c9_q137"
               and e["path"].endswith("notebooklm_freetier_report.py")
               for e in result["exemptions_used"])


def test_holdout_audit_digit_ending_id_does_not_match_longer_distinct_id(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text('["s9_q4"]')
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    (root / "ledger.md").write_text("dev row s9_q42 flipped at 24 tokens\n")

    result = audit(holdout, [root], include_legacy_policy=False)

    assert result["pass"] is True, result["findings"]


def test_holdout_audit_digit_ending_id_still_matches_exact_occurrences(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text('["s9_q4"]')
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.md").write_text("the s9_q42 row and then s9_q4, the real leak\n")
    (root / "b.py").write_text('SAMPLE = "s9_q4"\n')

    result = audit(holdout, [root], include_legacy_policy=False)

    assert result["pass"] is False
    assert {f["path"] for f in result["findings"]} == {
        str(root / "a.md"),
        str(root / "b.py"),
    }


def test_holdout_audit_flags_speaker_name_literal_in_runtime_code(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text('["s9_q4"]')
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    datasets = tmp_path / "datasets" / "locomo"
    datasets.mkdir(parents=True)
    (datasets / "conv.json").write_text(json.dumps(
        [{"conversation": {"speaker_a": "Zorblatt", "speaker_b": "Quixana"}}]
    ))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "verify.py").write_text(
        'def check(hyp, prem):\n    if "zorblatt" in hyp:\n        return True\n'
    )

    result = audit(
        holdout, [runtime], include_legacy_policy=False,
        dataset_dir=tmp_path / "datasets", runtime_roots=[runtime],
    )

    assert result["pass"] is False
    assert any(
        f["needle"] == "zorblatt" and f.get("kind") == "entity-name"
        for f in result["findings"]
    )
    assert result["entity_names_checked"] == 2


def test_holdout_audit_entity_scan_ignores_embedded_and_docs(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text('["s9_q4"]')
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    datasets = tmp_path / "datasets" / "locomo"
    datasets.mkdir(parents=True)
    (datasets / "conv.json").write_text(json.dumps(
        [{"conversation": {"speaker": "Maria"}}]
    ))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    # Embedded occurrence (word-boundary miss) and a non-runtime doc mention are fine.
    (runtime / "db.py").write_text('BACKEND = "mariadb"\n')
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("Maria is a speaker in the corpus.\n")

    result = audit(
        holdout, [runtime, docs], include_legacy_policy=False,
        dataset_dir=tmp_path / "datasets", runtime_roots=[runtime],
    )

    assert result["pass"] is True, result["findings"]


def test_holdout_audit_rejects_empty_registry_without_explicit_override(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text("[]")
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    (root / "x.py").write_text("SAFE = True\n")

    result = audit(holdout, [root], include_legacy_policy=False)
    allowed = audit(
        holdout,
        [root],
        include_legacy_policy=False,
        require_holdout_needles=False,
    )

    assert result["pass"] is False
    assert result["registry_error"] == "holdout registry is empty"
    assert result["holdout_needles_checked"] == 0
    assert allowed["pass"] is True


def test_holdout_audit_finds_removed_dataset_scan_symbols(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text("[]")
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    symbol = "_long" + "memeval_source_scan"
    (root / "x.py").write_text(f"def {symbol}():\\n    return None\\n")

    result = audit(holdout, [root])

    assert result["pass"] is False
    assert result["findings"][0]["needle"] == symbol


def test_holdout_audit_finds_removed_adapter_rescue_symbols(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text("[]")
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    symbol = "_extract_" + "direct_fact_match"
    (root / "adapter.py").write_text(f"def {symbol}():\\n    return None\\n")

    result = audit(holdout, [root])

    assert result["pass"] is False
    assert result["findings"][0]["needle"] == symbol


def test_holdout_audit_finds_removed_temporal_adapter_postprocessor(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text("[]")
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    symbol = "_compact_" + "temporal_slot_answer"
    (root / "adapter.py").write_text(f"def {symbol}():\\n    return None\\n")

    result = audit(holdout, [root])

    assert result["pass"] is False
    assert result["findings"][0]["needle"] == symbol


def test_holdout_audit_finds_fixed_answer_literals(tmp_path):
    from bench.audit_no_holdout_leakage import audit

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "leaked_sample_ids.json").write_text("[]")
    (holdout / "longmemeval_test_holdout.json").write_text("[]")
    (holdout / "locomo_test_holdout.json").write_text("[]")
    (holdout / "manifest.json").write_text("{}")
    root = tmp_path / "src"
    root.mkdir()
    canned_answer = "Middle-class" + " or wealthy"
    (root / "answer.py").write_text(f"ANSWER = {canned_answer!r}\\n")

    result = audit(holdout, [root])

    assert result["pass"] is False
    assert result["findings"][0]["needle"] == canned_answer


def test_smqe_source_has_no_fixed_slice_entity_literals():
    source = "\n".join(
        path.read_text(errors="ignore").lower()
        for path in Path("eidetic/smqe").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    forbidden = [
        "so" + "ny",
        "go" + "dox",
        "mo" + "ma",
        "metro" + "politan museum",
        "c. s. " + "lewis",
        "harry " + "potter",
        "eternal " + "sunshine",
        "wells " + "fargo",
        "spit" + "fire",
        "ca" + "maro",
    ]

    for needle in forbidden:
        assert needle not in source


def test_smqe_source_has_no_fixed_slice_answer_literals():
    source = "\n".join(
        path.read_text(errors="ignore").lower()
        for path in Path("eidetic/smqe").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    forbidden = [
        "middle-class" + " or wealthy",
        "hairless pets" + ", such as hairless cats or pigs",
        "cook " + "dog treats",
        "utensil " + "holder",
        "pesto " + "pasta",
        "minty " + "fresh salad",
        "insta" + "gram",
        "pimm" + "'s cup with a twist",
        "portable " + "power bank",
        "virtual " + "coffee breaks",
        "battery-saving " + "mode",
        "fully " + "charged",
        "atmospheric distillation, fluid catalytic cracking" + " (fcc), alkylation, and hydrotreating",
    ]

    for needle in forbidden:
        assert needle not in source


def test_speaker_fact_skips_dative_addressee():
    """'I told Maya that X' answers X, never the addressee 'Maya'; complement-clause subjects
    after non-ditransitive verbs are untouched ('said Tom's party was fun')."""
    from eidetic.smqe.record_ops import _speaker_fact_value

    assert "deadline" in _speaker_fact_value(
        "User: I told Maya that the project deadline moved to Friday.")
    assert "Maya" not in _speaker_fact_value(
        "User: I told Maya that the project deadline moved to Friday.")
    assert _speaker_fact_value("She said Tom's party was fun.").startswith("Tom")
    assert "review" in _speaker_fact_value("User: I asked Priya to review the draft contract.")


def test_who_told_me_answers_the_speaker(tmp_path):
    """'Who told me X?' answers the SPEAKER from the role prefix, not the content."""
    store = RecordStore(tmp_path / "who-told.sqlite")
    scope = Scope(namespace="who-told")
    store.upsert_record(_record(
        "Maya: The venue books up three months in advance.",
        scope=scope, valid_at=1_700_000_100,
    ))
    store.upsert_record(_record(
        "User: I should start planning the reception soon.",
        scope=scope, valid_at=1_700_000_200,
    ))

    ans = structured_answer(
        _Retriever(store),
        "Who told me the venue books up three months in advance?",
        at=1_800_000_000, scope=scope,
    )

    assert ans is not None
    assert "maya" in ans.answer.lower()
    assert "books up" not in ans.answer.lower()


def test_plural_enumeration_lists_distinct_values_across_records(tmp_path, fresh_settings, monkeypatch):
    """'Which countries have I visited?' with three visits in memory must enumerate all three
    (one support per record), not ship a 1-of-3 single-record atom as the verified answer.
    Flag-gated: off keeps today's behavior."""
    from dataclasses import replace as _replace

    from eidetic.config import get_settings as _gs

    store = RecordStore(tmp_path / "plural-enum.sqlite")
    scope = Scope(namespace="plural-enum")
    rows = [
        ("User: I visited Japan in March.", 1_700_000_100),
        ("User: I visited Peru during the fall harvest.", 1_700_050_000),
        ("User: Last week I visited Kenya for a safari.", 1_700_090_000),
    ]
    for text, t in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=t))

    # flag OFF: today's behavior (whatever single-support answer the tail produces)
    off = structured_answer(_Retriever(store), "Which countries have I visited?",
                            at=1_800_000_000, scope=scope)
    if off is not None:
        assert len({c.memory_id for c in off.citations}) <= 1

    # flag ON: enumerate distinct values across records
    monkeypatch.setenv("PLURAL_ENUMERATION", "1")
    _gs.cache_clear()
    try:
        on = structured_answer(_Retriever(store), "Which countries have I visited?",
                               at=1_800_000_000, scope=scope)
        assert on is not None
        low = on.answer.lower()
        assert "japan" in low and "peru" in low and "kenya" in low
        assert on.verified is True
        assert len({c.memory_id for c in on.citations}) == 3
    finally:
        monkeypatch.delenv("PLURAL_ENUMERATION", raising=False)
        _gs.cache_clear()


def test_plural_enumeration_stays_out_of_singular_and_counted_shapes(tmp_path, monkeypatch):
    from eidetic.config import get_settings as _gs

    store = RecordStore(tmp_path / "plural-enum-neg.sqlite")
    scope = Scope(namespace="plural-enum-neg")
    store.upsert_record(_record("User: I visited Japan in March.", scope=scope,
                                valid_at=1_700_000_100))
    monkeypatch.setenv("PLURAL_ENUMERATION", "1")
    _gs.cache_clear()
    try:
        # singular s-final noun ("class") must not trigger the enumerator
        ans = structured_answer(_Retriever(store), "Which class did I enjoy most?",
                                at=1_800_000_000, scope=scope)
        if ans is not None:
            assert "," not in ans.answer
        # a single distinct value cannot enumerate -> falls through to today's paths
        one = structured_answer(_Retriever(store), "Which countries have I visited?",
                                at=1_800_000_000, scope=scope)
        if one is not None:
            assert "japan" in one.answer.lower()
    finally:
        monkeypatch.delenv("PLURAL_ENUMERATION", raising=False)
        _gs.cache_clear()


def test_open_inference_extracts_titlecase_copular_value(tmp_path):
    """'What play did I attend?' with memory literally restating 'The play I attended was
    actually a production of <Title>' must answer the title, not fail closed to the reader."""
    store = RecordStore(tmp_path / "copular-title.sqlite")
    scope = Scope(namespace="copular-title")
    store.upsert_record(_record(
        "User: The play I attended was actually a production of The Glass Lantern, "
        "have you heard of it?",
        scope=scope, valid_at=1_700_000_100,
    ))
    store.upsert_record(_record(
        "User: The theater lobby smelled of fresh popcorn.",
        scope=scope, valid_at=1_700_000_200,
    ))

    ans = structured_answer(
        _Retriever(store),
        "What play did I attend at the neighborhood theater downtown?",
        at=1_800_000_000, scope=scope,
    )

    assert ans is not None
    assert "Glass Lantern" in ans.answer
    assert "popcorn" not in ans.answer
    assert ans.verified is True


def test_open_inference_copular_needs_titlecase_and_target(tmp_path):
    """No TitleCase value or no target overlap -> still fails closed (no free association)."""
    store = RecordStore(tmp_path / "copular-neg.sqlite")
    scope = Scope(namespace="copular-neg")
    store.upsert_record(_record(
        "User: The evening was actually quite chilly for spring.",
        scope=scope, valid_at=1_700_000_100,
    ))

    ans = structured_answer(
        _Retriever(store),
        "What play did I attend at the neighborhood theater downtown?",
        at=1_800_000_000, scope=scope,
    )

    assert ans is None or "chilly" not in ans.answer


def test_event_order_composes_dated_timeline_for_three_events(tmp_path):
    """'Which three events took place in the order from earliest to latest: A, B, and C?'
    composes a DATED timeline from anchored records - deterministic and judge-checkable,
    instead of failing closed to a reader that echoes the question's order undated."""
    store = RecordStore(tmp_path / "event-order-3.sqlite")
    scope = Scope(namespace="event-order-3")
    rows = [
        ("User: I just helped my friend repaint a studio today.", datetime(2023, 2, 5, 12, 0)),
        ("User: I just helped my cousin choose decorations for her housewarming party.",
         datetime(2023, 2, 10, 12, 0)),
        ("User: I just ordered an engraved travel mug for my friend's birthday today.",
         datetime(2023, 2, 20, 12, 0)),
        ("User: The weather was lovely all week.", datetime(2023, 2, 12, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store),
        "Which three events took place in the order from earliest to latest: the day I helped my "
        "friend repaint the studio, the day I helped my cousin choose decorations for her "
        "housewarming, and the day I ordered an engraved travel mug for my friend's birthday?",
        at=datetime(2023, 3, 1, 12, 0).timestamp(), scope=scope,
    )

    assert ans is not None
    assert ans.verified is True
    low = ans.answer.lower()
    assert "2023-02-05" in ans.answer and "2023-02-10" in ans.answer and "2023-02-20" in ans.answer
    assert low.index("studio") < low.index("housewarming") < low.index("travel mug")


def test_event_order_three_events_fails_closed_when_one_unanchored(tmp_path):
    store = RecordStore(tmp_path / "event-order-miss.sqlite")
    scope = Scope(namespace="event-order-miss")
    store.upsert_record(_record(
        "User: I just helped my friend repaint a studio today.",
        scope=scope, valid_at=datetime(2023, 2, 5, 12, 0).timestamp()))

    ans = structured_answer(
        _Retriever(store),
        "Which three events took place in the order from earliest to latest: the day I helped my "
        "friend repaint the studio, the day I adopted the parrot, and the day I sold my "
        "kayak?",
        at=datetime(2023, 3, 1, 12, 0).timestamp(), scope=scope,
    )

    assert ans is None or "parrot" not in ans.answer


def test_relative_temporal_duration_held_and_first_prefers_earliest(tmp_path):
    """'When did X get his FIRST two geckos?' must resolve a session-dated 3-year tenure
    statement to session-minus-3-years and prefer the EARLIEST resolved date over a
    later higher-scoring acquisition mention."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "first-earliest.sqlite")
    scope = Scope(namespace="first-earliest")
    rows = [
        ("Ravi: My two geckos are doing great. I've had my geckos for 3 years now and they "
         "make me smile every day!", datetime(2022, 1, 23, 12, 0)),
        ("Ravi: I saw another gecko at a pet store and just had to get him - a third gecko "
         "this year!", datetime(2022, 11, 9, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store),
        "When did Ravi get his first two geckos?",
        at=datetime(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )

    assert ans is not None
    assert "2019" in ans.answer
    assert "2022-11" not in ans.answer


def test_fact_shaped_like_questions_do_not_route_to_advice_synthesis(tmp_path):
    """'What sports does John like besides basketball?' is a slot/list FACT question - routing
    it to preference/advice synthesis returned brand chatter as a verified answer."""
    assert plan_query("What sports does John like besides basketball?").op != "preference_synth"
    assert plan_query("Which desserts does Mira enjoy most?").op != "preference_synth"
    # genuine advice requests keep the synthesis route
    assert plan_query("Can you suggest a dessert I should bake this weekend?").op == "preference_synth"
    assert plan_query("What should I serve for dinner with my garden ingredients?").op == "preference_synth"


def test_date_anchored_latest_value_verifies_on_the_anchor(tmp_path):
    """An explicit-date lookup whose winning atom was PROVEN in-window by the date filter must
    verify on the verbatim anchor - asking NLI to re-derive the 'last night' -> May 3 link is
    what flapped identical runs between verified and unverified."""
    from eidetic.models import NLILabel

    class _NeverEntailRetriever(_Retriever):
        """Strict-hypothesis NLI that never entails: only the anchor rule can verify."""

        def verify(self, premise, hypothesis):
            return (NLILabel.NEUTRAL, 0.2)

        def verify_citation(self, rec, hypothesis):
            return (NLILabel.NEUTRAL, 0.2)

    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "date-anchor-verify.sqlite")
    scope = Scope(namespace="date-anchor-verify")
    rec = _record(
        "Rhea: My mom and I cooked some pasta for dinner together last night!",
        scope=scope, valid_at=datetime(2023, 5, 4, 22, 0).timestamp(),
    )
    store.upsert_record(rec)
    store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _NeverEntailRetriever(store),
        "Who did Rhea have dinner with on May 3, 2023?",
        at=datetime(2023, 8, 16, 12, 0).timestamp(), scope=scope,
    )

    assert ans is not None
    assert "mom" in ans.answer.lower()
    assert ans.verified is True
    assert ":date_anchored" in ans.note


def test_assembled_enumerations_never_verify_on_anchors_alone(tmp_path):
    """A comma-assembled list from a non-computed op is the one derived shape whose anchors
    prove nothing: every fragment is quotable, so junk lists ('Good, Ok, You Get') shipped
    verified at n=40. Such answers must survive the strict query-aware hypothesis."""
    from eidetic.models import NLILabel

    class _NeverEntail(_Retriever):
        def verify(self, premise, hypothesis):
            return (NLILabel.NEUTRAL, 0.2)

        def verify_citation(self, rec, hypothesis):
            return (NLILabel.NEUTRAL, 0.2)

    store = RecordStore(tmp_path / "enum-verify.sqlite")
    scope = Scope(namespace="enum-verify")
    a = _record("User: You're doing great, keep it up!", scope=scope, valid_at=1.0)
    b = _record("User: Ok, sounds good to me.", scope=scope, valid_at=2.0)
    store.upsert_record(a)
    store.upsert_record(b)
    result = StructuredAnswerResult(
        answer="Good, Great Job, Ok, You Get",
        op="open_inference",
        backend="claim",
        supports=[
            StructuredSupport(memory_id=a.memory_id, proof_atom="doing great"),
            StructuredSupport(memory_id=b.memory_id, proof_atom="Ok"),
        ],
        note="smqe:open_inference:claim",
    )

    ans = answer_from_result(
        _NeverEntail(store),
        "What outdoor activities has the user done other than hiking?",
        result, verify=True,
    )
    assert ans is None                    # strict hypothesis failed -> no verified junk list

    # computed enumerations (event_order timelines) keep their anchor exemption
    result2 = StructuredAnswerResult(
        answer="[2023-02-05] prepared the nursery; [2023-02-10] baby shower, gifts, and cake",
        op="event_order",
        backend="claim",
        supports=[StructuredSupport(memory_id=a.memory_id, proof_atom="doing great")],
        note="smqe:event_order:claim",
    )
    # anchor path: proof atom is verbatim in the record
    a2 = answer_from_result(_NeverEntail(store), "Which happened first?", result2, verify=True)
    assert a2 is not None and a2.verified is True


def test_relative_temporal_extracts_bare_year_from_atom(tmp_path):
    """'she gave it to me in 2010' answers a when-question with the stated bare year - the
    date extractor only knew month+year/ISO/relative forms, so the explicit-year atom never
    became a candidate and a weaker 'last year' atom shipped verified-wrong."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "bare-year.sqlite")
    scope = Scope(namespace="bare-year")
    rows = [
        ("Suki: This bracelet was a keepsake from my mother, she gave me the bracelet in 2010 in Lisbon.",
         datetime(2023, 1, 23, 12, 0)),
        ("Suki: Here's me and my partner at a retreat last year - had an awesome time!",
         datetime(2023, 1, 25, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store),
        "When did Suki's mom gift her the bracelet?",
        at=datetime(2023, 8, 1, 12, 0).timestamp(), scope=scope,
    )

    assert ans is not None
    assert "2010" in ans.answer
    assert "2022" not in ans.answer


def test_ordinal_anchor_slot_answers_from_the_labeled_occurrence(tmp_path):
    """'What game was the SECOND tournament based on?' - the source self-labels ordinals
    ('I won my second tournament!') and states the slot value in the same record; the answer
    is the TitleCase phrase modifying the anchor noun there, never a value from a different
    occurrence's record."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "ordinal-slot.sqlite")
    scope = Scope(namespace="ordinal-slot")
    rows = [
        ("Marco: I won my first board game tournament last week - so exciting! It was a "
         "Pebble Rush tournament downtown.", datetime(2022, 1, 21, 12, 0)),
        ("Marco: Last week I won my second tournament!\n"
         "Priya: Wow, congrats! What game were you playing?\n"
         "Marco: I usually play checkers online, but this time I entered the local Cinder Duel "
         "tournament and turns out I'm really good!", datetime(2022, 5, 2, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store),
        "What game was the second tournament that Marco won based on?",
        at=datetime(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )

    assert ans is not None
    assert "Cinder Duel" in ans.answer
    assert "Pebble Rush" not in ans.answer
    assert ans.verified is True


def test_dialogue_crystals_respect_wh_class_and_reject_pleasantries(tmp_path):
    """A recorded 'How did X go?' crystal must not answer a 'What game was X?' question
    (wh-class mismatch), and greeting-only crystal answers ('Hey Priya, thanks!') are never
    served as answers to anything."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "crystal-wh.sqlite")
    scope = Scope(namespace="crystal-wh")
    rec = _record(
        "Priya: How did the last game tournament go?\n"
        "Marco: Hey Priya, thanks! It went really well overall.",
        scope=scope, valid_at=1_700_000_100,
    )
    store.upsert_record(rec)
    store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store),
        "What game was the second tournament that Marco won based on?",
        at=1_800_000_000, scope=scope,
    )

    assert ans is None or "Hey Priya" not in ans.answer


def test_past_when_questions_never_answer_from_future_intent_atoms(tmp_path):
    """'When WAS the concert?' must not answer with the date of a future PLAN ('going to Tokyo
    next month') - a future-intent atom can never date a past event."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "past-future.sqlite")
    scope = Scope(namespace="past-future")
    rows = [
        ("Calvin: My concert in Osaka was electric back in March 2023, the crowd sang along!",
         datetime(2023, 4, 2, 12, 0)),
        ("Calvin: I'm actually going to Osaka next month after the tour ends - my concert there "
         "will be huge.", datetime(2023, 10, 19, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store),
        "When was Calvin's concert in Osaka?",
        at=datetime(2023, 12, 1, 12, 0).timestamp(), scope=scope,
    )

    assert ans is not None
    assert "March 2023" in ans.answer or "2023-03" in ans.answer
    assert "November" not in ans.answer


def test_junk_claim_enumeration_does_not_shadow_record_backend(tmp_path):
    """The executor takes the CLAIM backend's result even when it is a junk enumeration that
    verification will kill - shadowing a legit RECORD-backend answer behind it. Non-credible
    enumerations must decline at dispatch so downstream backends get their chance."""
    from eidetic.models import ClaimRecord

    store = RecordStore(tmp_path / "junk-shadow.sqlite")
    scope = Scope(namespace="junk-shadow")
    rec = _record(
        "Andrew: Besides hiking, my hobbies are rock climbing and fishing these days.",
        scope=scope, valid_at=1_700_000_100,
    )
    store.upsert_record(rec)
    # hand-crafted junk claims that trip the hobbies collector into a fragment list
    for frag in ("doing great", "Ok", "you get", "in the park"):
        store.add_claim(ClaimRecord(
            claim_type="state", scope=scope, subject="Andrew",
            predicate="enjoy", object=frag,
            value=f"Andrew: I also enjoy {frag}!",
            proof_atom=f"Andrew: I also enjoy {frag}!",
            source_memory_id=rec.memory_id, valid_at=1_700_000_100,
        ))

    ans = structured_answer(
        _Retriever(store),
        "What are Andrew's hobbies and interests these days?",
        at=1_800_000_000, scope=scope,
    )

    assert ans is not None                      # the record backend must get its chance
    low = ans.answer.lower()
    assert "you get" not in low and "doing great" not in low
    assert "rock climbing" in low or "fishing" in low


def test_claim_enumeration_answers_from_tier1_claims(tmp_path):
    """Collector-rewrite step 1: enumerations come from typed CLAIMS (subject + predicate
    family + object with its own proof atom), not per-query regex re-parsing of raw text.
    Junk-fragment claims never qualify; fewer than two credible objects falls through."""
    from eidetic.models import ClaimRecord

    store = RecordStore(tmp_path / "claim-enum.sqlite")
    scope = Scope(namespace="claim-enum")
    rows = [
        ("preference", "Dave", "enjoys", "hiking in the hills",
         "Dave: I really enjoy hiking in the hills."),
        ("preference", "Dave", "loves", "film photography",
         "Dave: I love film photography on weekends."),
        ("preference", "Dave", "likes", "live concerts",
         "Dave: I like going to live concerts."),
        ("preference", "Dave", "enjoys", "you get",          # junk fragment - must not qualify
         "Dave: you get the idea!"),
        ("preference", "Mira", "enjoys", "rock climbing",    # wrong person - must not qualify
         "Mira: I enjoy rock climbing."),
    ]
    for i, (ctype, subj, pred, obj, atom) in enumerate(rows):
        rec = _record(atom, scope=scope, valid_at=1_700_000_100 + i)
        store.upsert_record(rec)
        store.add_claim(ClaimRecord(
            claim_type=ctype, scope=scope, subject=subj, predicate=pred, object=obj,
            value=atom, proof_atom=atom, source_memory_id=rec.memory_id,
            valid_at=1_700_000_100 + i,
        ))

    ans = structured_answer(
        _Retriever(store),
        "What hobbies does Dave enjoy these days?",
        at=1_800_000_000, scope=scope,
    )

    assert ans is not None
    low = ans.answer.lower()
    assert "hiking" in low and "photography" in low and "concerts" in low
    assert "you get" not in low and "rock climbing" not in low
    assert ans.verified is True


def test_deterministic_claims_destructure_enjoy_statements(tmp_path):
    """Claim quality gates the enumerator: 'I really enjoy hiking in the hills' must yield a
    claim with the SPEAKER subject, an enjoy-family predicate, and the activity as the object -
    and sentence-initial adverbs/imperatives ('Remember', 'Simply') are never subjects."""
    from eidetic.smqe.claim_extraction import claims_for_record

    scope = Scope(namespace="claimq")
    rec = _record("Nate: I really enjoy hiking in the hills.", scope=scope,
                  valid_at=1_700_000_100)
    claims = claims_for_record(rec)
    pref = [c for c in claims if "enjoy" in (c.predicate or "").lower()]
    assert pref, f"no enjoy-predicate claim in {[c.predicate for c in claims]}"
    assert pref[0].subject == "Nate"
    assert "hiking" in (pref[0].object or "").lower()

    rec2 = _record("Nate: Remember to water the fern before the trip.", scope=scope,
                   valid_at=1_700_000_200)
    for c in claims_for_record(rec2):
        assert c.subject.lower() not in {"remember", "simply", "anyway"}


def test_malformed_enumerations_are_refused_even_when_nli_entails(tmp_path):
    """Live probe shipped 'Nike and Under Armour, these make me, basketball, a blast...' as a
    VERIFIED answer - the NLI entailed fragment soup against a long premise. A non-credible
    enumeration from a non-computed op is malformed regardless of entailment: verification
    refuses it outright (fail closed to the reader), on every producer path."""
    from eidetic.models import NLILabel

    class _AlwaysEntail(_Retriever):
        def verify(self, premise, hypothesis):
            return (NLILabel.ENTAILMENT, 0.99)

        def verify_citation(self, rec, hypothesis):
            return (NLILabel.ENTAILMENT, 0.99)

    store = RecordStore(tmp_path / "malformed-enum.sqlite")
    scope = Scope(namespace="malformed-enum")
    a = _record("User: Nike and Under Armour make great gear.", scope=scope, valid_at=1.0)
    b = _record("User: these make me happy, what a blast.", scope=scope, valid_at=2.0)
    store.upsert_record(a)
    store.upsert_record(b)
    result = StructuredAnswerResult(
        answer="Nike and Under Armour, these make me, basketball, a blast, big games",
        op="open_inference",
        backend="claim",
        supports=[
            StructuredSupport(memory_id=a.memory_id, proof_atom="Nike and Under Armour"),
            StructuredSupport(memory_id=b.memory_id, proof_atom="a blast"),
        ],
        note="smqe:open_inference:claim",
    )

    ans = answer_from_result(
        _AlwaysEntail(store),
        "What sports does John like besides basketball?",
        result, verify=True,
    )
    assert ans is None


def test_compound_wh_questions_fail_closed_to_the_reader(tmp_path):
    """'When AND where is X?' answered only the venue, verified - a half answer. Compound
    interrogatives need both facets composed; single-slot structured ops decline them."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "compound-wh.sqlite")
    scope = Scope(namespace="compound-wh")
    rec = _record(
        "User: Lily's first violin recital is on August 9th at the Aurora Hall.",
        scope=scope, valid_at=1_700_000_100,
    )
    store.upsert_record(rec)
    store.add_claims(claims_for_record(rec))

    ans = structured_answer(
        _Retriever(store),
        "When and where is Lily's recital?",
        at=1_800_000_000, scope=scope,
    )

    # either both facets present, or fail closed (reader composes them) - never a verified half
    if ans is not None:
        low = ans.answer.lower()
        assert ("august" in low or "9" in low) and "aurora" in low


def test_option_choice_answer_must_name_an_option(tmp_path):
    """'Would Farid prefer a Falcon Roadster GT or a Meadow Cruiser SE?' is answered by
    NAMING one. The option-choice anchor exemption let a verbatim-but-irrelevant fragment
    ("nine, I've been obsessed with how engines fit together") ship VERIFIED at n=40 because
    the atom quoted the source while answering neither option. Deterministic form floor:
    exact-token overlap with an option segment, every op, preference_synth included."""
    store = RecordStore(tmp_path / "option-form.sqlite")
    scope = Scope(namespace="option-form")
    rec = _record("Farid: Ever since I was nine, I've been obsessed with how engines fit together. "
                  "Vintage roadsters are my passion - I'd take a Falcon Roadster GT any day.",
                  scope=scope, valid_at=1.0)
    store.upsert_record(rec)
    query = "Would Farid prefer working on a Falcon Roadster GT or a Meadow Cruiser SE?"

    junk = StructuredAnswerResult(
        answer="nine, I've been obsessed with how engines fit together",
        op="preference_synth", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id,
                                    proof_atom="nine, I've been obsessed with how engines fit together")],
        note="smqe:preference_synth:claim",
    )
    assert answer_from_result(_Retriever(store), query, junk, verify=True) is None

    named = StructuredAnswerResult(
        answer="Falcon Roadster GT",
        op="preference_synth", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id,
                                    proof_atom="I'd take a Falcon Roadster GT any day")],
        note="smqe:preference_synth:claim",
    )
    ans = answer_from_result(_Retriever(store), query, named, verify=True)
    assert ans is not None and ans.answer == "Falcon Roadster GT"

    # non-option-choice queries are untouched by the rule
    plain = StructuredAnswerResult(
        answer="restoring roadsters",
        op="latest_value", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id,
                                    proof_atom="Vintage roadsters are my passion")],
        note="smqe:latest_value:claim",
    )
    assert answer_from_result(_Retriever(store), "What is Farid's passion?", plain,
                              verify=True) is not None


def test_future_polarity_atom_floors_earlier_derived_dates(tmp_path):
    """'I snapped that photo in Oslo last night' (05-16) dated the concert 05-15 while 'my
    upcoming performance in Oslo this month', spoken the same day, proves the event had
    not happened yet. A same-event future-polarity statement floors derived dates: earlier
    candidates are contradicted, and losing all of them fails closed (reader takes over).
    Same-event needs EVERY non-entity target term matched (event-synonym families count);
    a mere trip mention must never floor a concert date."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "future-floor.sqlite")
    scope = Scope(namespace="future-floor")
    rows = [
        ("Wei: I snapped that photo in Oslo last night.", _dt(2023, 5, 16, 12, 0)),
        ("Wei: Super stoked for my upcoming performance in Oslo this month.",
         _dt(2023, 5, 16, 12, 0)),
        ("Wei: I'm actually heading to Oslo next month once the tour wraps up.",
         _dt(2023, 10, 19, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "When was Wei's concert in Oslo?",
        at=_dt(2023, 11, 1, 12, 0).timestamp(), scope=scope,
    )
    # the 05-15 photo date is floored by the 05-16 future statement -> fail closed
    assert ans is None or "05-15" not in (ans.answer or "")

    # a LATER dated same-event atom survives the floor
    store.upsert_record(_record(
        "Wei: The concert in Oslo on May 28, 2023 was unreal -- the crowd was insane.",
        scope=scope, valid_at=_dt(2023, 6, 1, 12, 0).timestamp()))
    ans2 = structured_answer(
        _Retriever(store), "When was Wei's concert in Oslo?",
        at=_dt(2023, 11, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans2 is not None and ("2023-05-28" in ans2.answer or "May 28" in ans2.answer)


def test_date_anchored_activity_lookup_reads_verb_form_atoms(tmp_path):
    """'Which recreational activity was Noor pursuing on March 16, 2022?' abstained live:
    the doing-atom ('yesterday I went curling') never echoes the abstract wh-noun
    'activity', so every lexical target gate starves. Once the explicit-day window has
    proven membership deterministically, a tight verb-form extractor (went+gerund /
    played+object) may answer; preference statements ('I love curling') stay out."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "activity-day.sqlite")
    scope = Scope(namespace="activity-day")
    rows = [
        ("Noor: By the way, yesterday I went curling and scored 2 points. Farid: Nice!",
         _dt(2022, 3, 17, 12, 0)),
        ("Noor: I love curling! But honestly I don't play often.",
         _dt(2022, 4, 2, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "Which recreational activity was Noor pursuing on March 16, 2022?",
        at=_dt(2022, 5, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans is not None and ans.answer == "curling"
    assert ":date_anchored" in ans.note
    assert "went curling" in ans.citations[0].snippet

    # without the explicit day there is no window proof -> the extractor must not fire
    ans2 = structured_answer(
        _Retriever(store), "Which recreational activity does Noor pursue?",
        at=_dt(2022, 5, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans2 is None or ":date_anchored" not in (ans2.note or "")


def test_duration_question_requires_target_named_in_the_duration_atom(tmp_path):
    """'How long did it take Noor to finish writing her book?' pulled a pets tenure ('I've
    had them for 3 years') through the pronoun-group anaphora bridge: durations are
    ubiquitous, so a plural pronoun tied to session-level 'book' terms shipped an unrelated
    tenure as a writing duration. Duration questions demand the target named in the atom
    itself, in BOTH the specific and generic lookup loops; with no tied duration atom the
    op fails closed to the reader."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "duration-tie.sqlite")
    scope = Scope(namespace="duration-tie")
    store.upsert_record(_record(
        "Priya: I've had them for 3 years already and they fill my days with laughter! "
        "Noor: That's lovely. By the way, my book is coming along.",
        scope=scope, valid_at=_dt(2022, 5, 1, 12, 0).timestamp()))

    ans = structured_answer(
        _Retriever(store), "How long did it take for Noor to finish writing her book?",
        at=_dt(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans is None or "3 years" not in (ans.answer or "")

    # a duration atom that NAMES the target answers normally
    store.upsert_record(_record(
        "Noor: Writing the book took me four months in the end.",
        scope=scope, valid_at=_dt(2022, 10, 6, 12, 0).timestamp()))
    ans2 = structured_answer(
        _Retriever(store), "How long did it take for Noor to finish writing her book?",
        at=_dt(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans2 is not None and "four months" in ans2.answer


def test_duration_questions_skip_hypotheticals_and_extract_stated_durations(tmp_path):
    """Fresh-holdout conv7-row33 shape: 'How long have Suki and her partner been together?'
    shipped 'one day' VERIFIED from a hypothetical sunrise-watching wish -- future-intent,
    but duration-shaped. Two rules: (1) elapsed-time questions
    skip future-intent atoms; (2) the stated duration ('been together FOR THREE YEARS')
    outranks generic slot extraction, which returned the nearby noun 'Married'."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "duration-hypo.sqlite")
    scope = Scope(namespace="duration-hypo")
    rows = [
        ("Suki: Maybe one day the two of us will get to watch the sunrise from the ridge!",
         _dt(2023, 3, 1, 12, 0)),
        ("Suki: We aren't married yet, but we have been together for three years.",
         _dt(2023, 1, 23, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "How long have Suki and her partner been together?",
        at=_dt(2023, 8, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans is not None and ans.answer == "three years"
    assert "for three years" in ans.citations[0].snippet


def test_zero_information_answers_are_refused(tmp_path):
    """Fresh-holdout conv5-row3 shape: 'My girlfriend' shipped VERIFIED for 'what kind of spots
    have Ravi and his girlfriend been checking out?' -- every content token already sits in the
    question, so the answer restates it. Deterministic form floor; clock-time answers
    ('11 pm') tokenize to nothing and must fail OPEN."""
    from eidetic.smqe.verify import _answer_adds_information

    q = "What kind of spots have Ravi and his girlfriend been checking out around town?"
    assert not _answer_adds_information(q, "My girlfriend")
    assert not _answer_adds_information(q, "checking out around town")
    assert _answer_adds_information(q, "cafes, hikes, a pet shelter")
    assert _answer_adds_information("What time did I go to bed?", "11 pm")

    store = RecordStore(tmp_path / "zero-info.sqlite")
    scope = Scope(namespace="zero-info")
    rec = _record("Ravi: My girlfriend and I have been checking out spots around town.", scope=scope, valid_at=1.0)
    store.upsert_record(rec)
    junk = StructuredAnswerResult(
        answer="My girlfriend", op="latest_value", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id, proof_atom="My girlfriend")],
        note="smqe:latest_value:claim",
    )
    assert answer_from_result(_Retriever(store), q, junk, verify=True) is None


def test_bare_day_of_month_resolves_against_session_date(tmp_path):
    """Fresh-holdout conv7-row57 shape: 'I bought a handheld for my partner as a birthday
    gift ON THE 17TH' (spoken Aug 19) names Aug 17, but no extractor knew the bare
    day-of-month form, so a January 'last week' handheld atom outscored the exact statement
    and shipped a wrong week window verified. The session month anchors the day; a day
    after the session date in non-future speech is the previous month's instance."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "bare-day.sqlite")
    scope = Scope(namespace="bare-day")
    rows = [
        ("Suki: We played the game Skylane on the handheld last week.",
         _dt(2023, 1, 27, 12, 0)),
        ("Suki: I bought a handheld for my partner as a birthday gift on the 17th and it's such a blast!",
         _dt(2023, 8, 19, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "When did Suki gift her partner a new handheld?",
        at=_dt(2023, 9, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans is not None and ans.answer == "2023-08-17"
    assert "on the 17th" in ans.citations[0].snippet


def test_ordinal_kth_event_interpolates_between_numbered_anchors(tmp_path):
    """Fresh-holdout conv3-row26 shape: 'when did Marco win his THIRD tourney?' shipped a late
    unrelated mention verified -- the generic loop has no counting semantics. The kth
    instance is the earliest unnumbered same-event atom strictly between the (k-1)th and
    (k+1)th anchors ('my second' 05-02, 'my fourth' 07-10 bound the 06-04 'another regional
    win last week'); with no k-1 anchor the op fails CLOSED."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "ordinal-k.sqlite")
    scope = Scope(namespace="ordinal-k")
    rows = [
        ("Marco: I won my first board game tournament last week - so exciting!",
         _dt(2022, 1, 22, 12, 0)),
        ("Marco: Last week I won my second tournament!", _dt(2022, 5, 2, 12, 0)),
        ("Marco: Things are going great - I just won another regional board game tournament "
         "last week!", _dt(2022, 6, 4, 12, 0)),
        ("Marco: I won my fourth board game tournament on Friday!", _dt(2022, 7, 10, 12, 0)),
        ("Marco: My game tournament got postponed, so I tried out some baking.",
         _dt(2022, 11, 10, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "When did Marco win his third tourney?",
        at=_dt(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans is not None
    assert ans.answer == "the week of 2022-05-28 to 2022-06-03"
    assert "another regional" in ans.citations[0].snippet

    # an explicit ordinal atom anchors directly
    ans2 = structured_answer(
        _Retriever(store), "When did Marco win his second tourney?",
        at=_dt(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans2 is not None and "2022-04" in ans2.answer  # 'last week' of 05-02

    # no (k-1) anchor -> fail closed, never the generic junk
    ans3 = structured_answer(
        _Retriever(store), "When did Marco win his seventh tourney?",
        at=_dt(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans3 is None


def test_favorite_category_noun_gates_preference_atoms(tmp_path):
    """Category-noun gating shape (FABRICATED dialogue): 'What is X's favorite FOOD?'
    once shipped a wrong-domain favorites atom verified -- it matched on 'favorite'
    alone. The category noun gates the atom pool through a general domain-family table,
    and the stated preference OBJECT ('even though I love oat biscuits' -> 'oat
    biscuits') beats an atom echo. Unknown category nouns stay ungated; an emptied pool
    fails closed."""
    store = RecordStore(tmp_path / "fav-category.sqlite")
    scope = Scope(namespace="fav-category")
    rows = [
        "Arlo: Harbor walks at dusk are one of my favorites - great for unwinding after work.",
        "Arlo: I'm cutting back on fried food and salty snacks, even though I love oat biscuits.",
    ]
    for i, text in enumerate(rows):
        store.upsert_record(_record(text, scope=scope, valid_at=float(i + 1)))

    ans = structured_answer(_Retriever(store), "What is Arlo's favorite food?",
                            at=100.0, scope=scope)
    assert ans is not None and ans.answer == "oat biscuits"
    assert "oat biscuits" in ans.citations[0].snippet

    # wrong-domain-only store: fail closed rather than ship harbor walks as a food
    store2 = RecordStore(tmp_path / "fav-category2.sqlite")
    scope2 = Scope(namespace="fav-category2")
    store2.upsert_record(_record(rows[0], scope=scope2, valid_at=1.0))
    ans2 = structured_answer(_Retriever(store2), "What is Arlo's favorite food?",
                             at=100.0, scope=scope2)
    assert ans2 is None or "harbor" not in (ans2.answer or "").lower()


def test_visited_cities_enumerate_from_claims_end_to_end(tmp_path):
    """Fresh-holdout conv1-row29 shape: 'Which cities has Jon visited?' had ZERO travel claims
    ('I've been to Paris' -- the clitic 've and the verb 'been' were both outside the
    extraction patterns) and the enumerator's head/verb gates only knew hobby nouns. Write
    path now extracts been/visited-to-X events with clean objects (terminators strip
    'together'/'yesterday'); the enumerator selects by the QUERY VERB'S family, so a
    visited-question reads visit-family claims and never like-family ones."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "cities.sqlite")
    scope = Scope(namespace="cities")
    texts = [
        "Jon: Oh, I've been to Paris yesterday. The croissants were divine.",
        "Jon: Last spring we visited Rome together. The Colosseum was breathtaking.",
        "Jon: My sister loves gardening and painting.",
    ]
    for i, t in enumerate(texts):
        rec = _record(t, scope=scope, valid_at=float(i + 1))
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(_Retriever(store), "Which cities has Jon visited?",
                            at=100.0, scope=scope)
    assert ans is not None
    assert set(ans.answer.replace(" and ", ", ").split(", ")) == {"Paris", "Rome"}
    # the like-family claim (gardening/painting) never leaks into a visited-question
    assert "gardening" not in ans.answer and "painting" not in ans.answer


def test_pronoun_contractions_never_count_as_information():
    """Slice-2 live catch: \"I'm reading\" shipped verified for a what-books question --
    the contraction token defeated the zero-information rule. Pronoun contractions are
    speaker scaffolding; an answer whose only non-query content is scaffolding is refused,
    while unevaluable empty-token answers (clock times) stay fail-open."""
    from eidetic.smqe.verify import _answer_adds_information

    q = "What books is John reading these days?"
    assert not _answer_adds_information(q, "I'm reading")
    assert _answer_adds_information(q, "I'm reading Winter Crossing")
    assert _answer_adds_information("What time did I go to bed?", "11 pm")


def test_why_questions_refuse_enumeration_shaped_answers(tmp_path):
    """Slice-2 live catch (conv0-row87 shape): 'Why did Caroline choose the adoption agency?'
    shipped 'Friday, adoption agency interviews, adoption agencies, LGBTQ, Research'
    VERIFIED -- every item is a quotable content noun, so the credibility rule and NLI
    anchors both pass, but a comma list answers nothing causal. Why-questions refuse
    enumeration-shaped answers unless they carry a reason clause."""
    store = RecordStore(tmp_path / "why-form.sqlite")
    scope = Scope(namespace="why-form")
    rec = _record("Caroline: On Friday I did adoption agency interviews. I liked their "
                  "support for LGBTQ+ individuals.", scope=scope, valid_at=1.0)
    store.upsert_record(rec)

    listy = StructuredAnswerResult(
        answer="Friday, adoption agency interviews, adoption agencies, LGBTQ, Research",
        op="open_inference", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id, proof_atom="adoption agency interviews")],
        note="smqe:open_inference:claim",
    )
    q = "Why did Caroline choose the adoption agency?"
    assert answer_from_result(_Retriever(store), q, listy, verify=True) is None

    reasoned = StructuredAnswerResult(
        answer="Because of their inclusivity, support for LGBTQ+ individuals, and warmth",
        op="open_inference", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id,
                                    proof_atom="I liked their support for LGBTQ+ individuals")],
        note="smqe:open_inference:claim",
    )
    assert answer_from_result(_Retriever(store), q, reasoned, verify=True) is not None


def test_books_read_enumerate_from_irregular_past_claims(tmp_path):
    """Slice-2 conv4-row4 shape: 'What books has Tim read?' shipped \"I'm reading\" -- 'read' is
    an irregular past invisible to the ed|t suffix rule, so no claims existed to enumerate.
    Irregular-past pattern + books head noun + read family: the enumerator now composes
    the titles, each with its own proof atom."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "books.sqlite")
    scope = Scope(namespace="books")
    texts = [
        "Tim: I read The Name of the Wind this month, it was fantastic.",
        "Tim: I've read The Alchemist recently. Highly recommend.",
    ]
    for i, t in enumerate(texts):
        rec = _record(t, scope=scope, valid_at=float(i + 1))
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(_Retriever(store), "What books has Tim read?",
                            at=100.0, scope=scope)
    assert ans is not None
    assert "Name of the Wind" in ans.answer and "Alchemist" in ans.answer


def test_last_monthname_resolves_against_statement_date(tmp_path):
    """Slice-2 conv4-row58 shape: 'Last August I told you about ... the trivia contest'
    (spoken in December) dates the event to August of the same year -- the strongest
    relative month form, previously unresolvable, so a 'last week' restaurant atom
    outranked it and shipped a wrong week window verified."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "last-month.sqlite")
    scope = Scope(namespace="last-month")
    rows = [
        ("Farid: I attended a neighborhood bistro with some new teammates last week.",
         _dt(2023, 9, 22, 12, 0)),
        ("Farid: Last August I told you about my great evening at a fundraiser event with "
         "a big movie trivia contest.", _dt(2023, 12, 9, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "When did Farid attend the movie trivia contest?",
        at=_dt(2023, 12, 20, 12, 0).timestamp(), scope=scope,
    )
    assert ans is not None and ans.answer == "August 2023"
    assert "Last August" in ans.citations[0].snippet

    # 'last May' spoken in March reaches back to the PREVIOUS year
    store.upsert_record(_record(
        "Farid: Last May we hosted the bake sale.", scope=scope,
        valid_at=_dt(2024, 3, 10, 12, 0).timestamp()))
    ans2 = structured_answer(
        _Retriever(store), "When did Farid host the bake sale?",
        at=_dt(2024, 4, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans2 is not None and ans2.answer == "May 2023"


def test_how_old_answers_stated_age_and_defers_inference(tmp_path):
    """Fresh-holdout shape: 'How old is Bruno?' ABSTAINED while 'Bruno is getting old, he is
    8 years old' sat in the store -- no extractor knew age statements. Entity-tied stated
    ages answer directly (latest statement wins); with no age atom the op falls through so
    the reader can still INFER a range ('likely under 30, she's in school')."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "age.sqlite")
    scope = Scope(namespace="age")
    rows = [
        ("Suki: Honestly, Bruno is getting old, he is 8 years old.", _dt(2023, 8, 27, 12, 0)),
        ("Suki: When Bruno was a puppy he was 1 year old obviously!", _dt(2020, 1, 1, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(_Retriever(store), "How old is Bruno?",
                            at=_dt(2023, 12, 1, 12, 0).timestamp(), scope=scope)
    assert ans is not None and ans.answer == "8 years old"   # latest statement wins
    assert "8 years old" in ans.citations[0].snippet

    ans2 = structured_answer(_Retriever(store), "How old is Tamsin?",
                             at=_dt(2023, 12, 1, 12, 0).timestamp(), scope=scope)
    assert ans2 is None                                       # no stated age -> reader owns it


def test_dialogue_crystal_requires_full_query_term_coverage(tmp_path):
    """Fresh-holdout wrong-instance class: 'What is Ravi working on OPENING?' matched a
    broader working-on crystal whose reply never mentions opening -- the slot-defining
    term was covered by nothing and the wrong instance shipped verified. Every query
    content term must appear in the recorded question or its answer; the correctly-tied
    record atom ('Still working on opening a dance studio') then answers. Also locks the
    two enablers: entity checks consult the record's turn prefix (first-person atoms do
    not name their speaker), and progressive-verb questions get the specific-loop target
    guard."""
    from datetime import datetime as _dt

    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "crystal-cover.sqlite")
    scope = Scope(namespace="crystal-cover")
    rows = [
        ("Priya: What are you working on these days, Ravi? "
         "Ravi: I'm refining the business plan and lining up investors.",
         _dt(2023, 6, 25, 12, 0)),
        ("Ravi: Still working on opening a dance studio. It takes a while!",
         _dt(2023, 6, 19, 12, 0)),
    ]
    for text, dt in rows:
        rec = _record(text, scope=scope, valid_at=dt.timestamp())
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(_Retriever(store), "What is Ravi working on opening?",
                            at=_dt(2023, 8, 1, 12, 0).timestamp(), scope=scope)
    assert ans is not None
    assert "dance studio" in ans.answer
    assert "business plan" not in ans.answer


def test_reader_form_floor_wh_temporal_type_agreement():
    """Slice-3 catches: 'When did Noor make the torte?' shipped the recipe INGREDIENTS
    verified (no temporal token); a what-question shipped a bare date; a what-question
    shipped junk wearing a 'Yes'. Type agreement both directions, yes/no exempt only for
    polarity questions; granular date answers keep passing when-questions."""
    from eidetic.smqe.verify import reader_answer_form_credible as f

    assert not f("When did Noor make a hazelnut torte with cherries?",
                 "I make it with rye flour, sesame oil, hazelnut and cherries")
    assert not f("When did Priya go camping in July?", "We're thinking about going camping")
    assert f("When did Noor make a hazelnut torte?", "on 5 October, 2022")
    assert f("When did Farid attend the contest?", "August 2023")

    assert not f("What did he and his uncle do in the workshop?", "2023-10-05")
    assert f("What did he and his uncle do in the workshop?", "tinkering with motorbike engines")

    assert not f("What would Priya's political leaning likely be?",
                 "Yes - Glad you've got folks to count on, Priya")
    assert f("Did Farid fix the roadster?", "Yes, he finished the restoration last week")

    # junk-head enumeration item ('up with developer forums') disqualifies the list
    assert not f("Where does Marco get his ideas from?",
                 "books, movies, various sources, up with developer forums for i")


def test_first_instance_uses_explicit_ordinal_anchor_and_skips_future_placements(tmp_path):
    """Slice-3 catch: 'when did Marco win his FIRST tournament?' shipped May 2022 verified
    from 'I've got a board game tournament NEXT MONTH' -- a forward placement with no
    will/going-to, invisible to the intent detector, and the explicit 'my first' anchor
    was ignored. Future-polarity now covers next-week/month/year placements, and k=1
    ordinal questions answer directly from an explicit first-anchor atom."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "first-anchor.sqlite")
    scope = Scope(namespace="first-anchor")
    rows = [
        ("Marco: I won my first board game tournament last week - so exciting!",
         _dt(2022, 1, 22, 12, 0)),
        ("Marco: I've got a board game tournament next month and I'm feeling ready for it.",
         _dt(2022, 4, 22, 12, 0)),
        ("Marco: Last week I won my second tournament!", _dt(2022, 5, 2, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "When did Marco win his first board game tournament?",
        at=_dt(2022, 12, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans is not None
    assert "2022-01" in ans.answer            # the January first-win week, never May
    assert "my first" in ans.citations[0].snippet


def test_city_visits_extract_across_phrasings_and_enumerate_proper_nouns(tmp_path):
    """Slice-3 multi-hop shape: 'Which cities has John been to?' -- city visits phrase as
    'I WAS IN Chicago' / 'my TRIP TO Seattle' / 'we FLEW TO New York', none of which the
    went/been-to patterns saw; and the visit family also carries 'charity thing' objects
    that must never appear in a which-cities answer (place-name heads enumerate proper
    nouns only)."""
    from eidetic.smqe.claim_extraction import claims_for_record

    store = RecordStore(tmp_path / "city-phrasings.sqlite")
    scope = Scope(namespace="city-phrasings")
    texts = [
        "John: I was in Chicago, it was awesome! The pizza was great.",
        "John: My trip to Seattle was rainy but fun.",
        "John: Oh, I've been to Paris yesterday.",
        "John: We flew to New York for the finals.",
        "John: I went to this charity thing and it was intense.",
    ]
    for i, t in enumerate(texts):
        rec = _record(t, scope=scope, valid_at=float(i + 1))
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))

    ans = structured_answer(_Retriever(store), "Which cities has John been to?",
                            at=100.0, scope=scope)
    assert ans is not None
    for city in ("Chicago", "Seattle", "Paris", "New York"):
        assert city in ans.answer, city
    assert "charity" not in ans.answer


def test_form_floor_quoted_names_and_them_head(tmp_path):
    """Slice-4 pair: a show literally titled \"That\" is stopwords to the tokenizer but
    real information -- the quoted-span check must run on RAW text before token stripping
    (the four-slice matrix caught the flip pre-commit); and 'them looking good' finally
    hits the enumeration head-stop ('them' was missing from the set)."""
    from eidetic.smqe.verify import reader_answer_form_credible as f

    q = "What is one of Marco's favorite mystery TV shows, as mentioned on November 14, 2023?"
    a = '"That" is one of Marco\'s favorite mystery TV shows, as mentioned on November 14, 2023'
    assert f(q, a)                                # quoted name = information

    junk = ("difference regarding, them looking good, Regular, Regular grooming, "
            "dog grooming course, Biscuit, groom Biscuit, and learn dog grooming")
    assert not f("What advice did Priya give to Ravi regarding grooming Biscuit?", junk)


def test_when_question_verb_selects_the_event_instance(tmp_path):
    """Third attempt at the wrong-instance temporal class, and the one that ships: the
    question's ACTION VERB identifies the instance. 'When was the album RELEASED?' must
    answer from the 'dropped on the 11th' atom, not the later launch party that shares
    the noun 'album' and wins on recency. Lemma families bridge released<->dropped;
    questions naming no family verb keep the existing ranking untouched."""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "lemma-instance.sqlite")
    scope = Scope(namespace="lemma-instance")
    rows = [
        ("Wei: My album officially dropped on the 11th and it felt surreal.",
         _dt(2023, 9, 13, 12, 0)),
        ("Wei: Last week I hosted a small gathering at my lakeside house for my new album.",
         _dt(2023, 11, 3, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(_Retriever(store), "When was Wei's album released?",
                            at=_dt(2023, 12, 1, 12, 0).timestamp(), scope=scope)
    assert ans is not None and ans.answer == "2023-09-11"
    assert "dropped on the 11th" in ans.citations[0].snippet


def test_likelihood_anchor_exemption_requires_the_inference_marker(tmp_path):
    """Dev-arm catch: 'Would Caroline likely have classic picture books?' shipped a bare
    fragment verified -- the likely-inference anchor exemption exists BECAUSE the yes/no
    marker is the executor's labeled inference over the cited premise, so an answer
    without that marker cannot use it and must face the strict hypothesis."""
    store = RecordStore(tmp_path / "likely-marker.sqlite")
    scope = Scope(namespace="likely-marker")
    rec = _record("Caroline: Building the children's library is a long process, and I "
                  "want friends to check them out.", scope=scope, valid_at=1.0)
    store.upsert_record(rec)
    q = "Would Caroline likely have classic picture books on her shelf?"

    from eidetic.models import NLILabel as _L

    class _NoEntail(_Retriever):
        def verify(self, premise, hypothesis):
            return (_L.NEUTRAL, 0.2)

        def verify_citation(self, rec, hypothesis):
            return (_L.NEUTRAL, 0.2)

    frag = StructuredAnswerResult(
        answer="a long process, and to check them out",
        op="open_inference", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id,
                                    proof_atom="a long process, and I want friends to check them out")],
        note="smqe:open_inference:claim",
    )
    assert answer_from_result(_NoEntail(store), q, frag, verify=True) is None

    labeled = StructuredAnswerResult(
        answer="Likely yes - she is building a children's library",
        op="open_inference", backend="claim",
        supports=[StructuredSupport(memory_id=rec.memory_id,
                                    proof_atom="Building the children's library is a long process")],
        note="smqe:open_inference:claim",
    )
    ans = answer_from_result(_Retriever(store), q, labeled, verify=True)
    assert ans is not None and ans.verified is True


def test_month_of_superlative_ties_the_metric_unit(tmp_path):
    """'In which MONTH did X achieve a career-high in POINTS?' composes the month of the
    superlative event, and the metric unit must tie -- a December career-high in ASSISTS
    answered a points question verified-wrong on a holdout window. (The gold on that row
    contradicts its own evidence sentence -- annotation anchored on a 'last month' opener
    while the sentence says 'last week'; this fix optimizes truthfulness, not the judge.)"""
    from datetime import datetime as _dt

    store = RecordStore(tmp_path / "month-sup.sqlite")
    scope = Scope(namespace="month-sup")
    rows = [
        ("Farid: Last week I dropped 40 points, my highest ever, in our league game.",
         _dt(2023, 7, 16, 12, 0)),
        ("Farid: By the way, I posted a career-high in assists this past Friday in our derby match.",
         _dt(2023, 12, 12, 12, 0)),
    ]
    for text, dt in rows:
        store.upsert_record(_record(text, scope=scope, valid_at=dt.timestamp()))

    ans = structured_answer(
        _Retriever(store), "In which month's game did Farid achieve a career-high score in points?",
        at=_dt(2024, 1, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans is not None and ans.answer == "July 2023"
    assert "40 points" in ans.citations[0].snippet

    ans2 = structured_answer(
        _Retriever(store), "In which month did Farid achieve a career-high in assists?",
        at=_dt(2024, 1, 1, 12, 0).timestamp(), scope=scope,
    )
    assert ans2 is not None and ans2.answer == "December 2023"


def test_iso_date_answers_are_never_query_echoes():
    """Six-window matrix catch: '2023-08-15' was refused as an 'echo' of the query's bare
    'August 2023' -- the prefix tolerance consumed the full ISO date against the year
    token. A full ISO date is always information unless the query quotes it exactly."""
    from eidetic.smqe.verify import reader_answer_form_credible as f

    assert f("When did Farid link up with his crew again after his trip in August 2023?",
             "2023-08-15")
