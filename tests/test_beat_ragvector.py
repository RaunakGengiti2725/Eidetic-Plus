"""Beat rag-vector on accuracy while keeping verify-or-abstain intact.

Levers, each behind a default-OFF flag (product_cost.json turns them on):

  (Lever 1 structured_attribute_gate was REMOVED: a lexical query-vs-claim tie cannot
   tell a correct paraphrase from a wrong-slot steal, so it over-deferred correct rows.)
  Lever 2  abstention_reader_coverage      -- eidetic_adapter coverage-backed override
           ship the reader text (verified=False, honest) when dense coverage is strong,
           instead of abstaining -- abstention scores 0 exactly like wrong.
  Lever 3  raw_dense_floor                 -- retrieval order raw dense above audit/pref
           so the top-N dense passages reach the reader without cropping structured.
  Lever 4  dense_topk_fallback             -- eidetic_adapter ENSEMBLE: when the primary
           answer is unverified, answer over ONLY the top-k dense slice rag-vector feeds,
           and prefer it (verified=False). Absorbs the baseline's retrieval; the verified
           answers (which never reach the fallback) keep the provenance edge.

INTEGRITY: every entity/attribute here is SYNTHETIC, invented in this file. No holdout
question text, gold, or benchmark speaker names. The gates are GENERAL mechanisms; these
fixtures mirror the MECHANISM, never a seen row. The leakage audit
(python -m bench.audit_no_holdout_leakage) fails closed on speaker names + 8-gram shingles.

Flag-default asymmetry (advisor): config defaults every flag OFF; only product_cost.json
turns them on. So each PASS-direction / ON-behavior fixture EXPLICITLY enables its flag on
the settings object -- otherwise the gate never runs and the assert greens for the wrong
reason. The DEFER fixtures fail loudly if the flag is forgotten (they self-protect); the
PASS ones do not, so they set the flag by hand.
"""
from __future__ import annotations

import dataclasses

from eidetic.config import get_settings
from eidetic.models import (
    Citation,
    ClaimRecord,
    MemoryRecord,
    NLILabel,
    Scope,
    StructuredAnswerResult,
    StructuredSupport,
)
from eidetic.store import RecordStore


# --------------------------------------------------------------------------- #
# Shared synthetic scaffolding
# --------------------------------------------------------------------------- #
_SCOPE = Scope(namespace="synthetic-beat-ragvector")


def _record(text: str, *, valid_at: float = 1_700_000_000.0) -> MemoryRecord:
    return MemoryRecord(
        text=text,
        source="user",
        scope=_SCOPE,
        valid_at=valid_at,
        content_hash="h-" + str(abs(hash(text))),
        raw_uri="mem://synthetic",
    )


class _StubRetriever:
    """Minimal retriever exposing what answer_from_result touches: store, settings,
    verify_citation, _ground_truth. verify_citation ENTAILS iff the atom text is a
    verbatim substring of the record -- so we can force the NLI path deterministically."""

    def __init__(self, store, *, settings=None):
        self.store = store
        self.settings = settings if settings is not None else get_settings()
        # presence of a callable `verify` flips answer_from_result onto the strict-hypothesis
        # (claim-backend) path, matching production.
        self.verify = object()

    def _ground_truth(self, rec):
        return rec.text or rec.summary or ""

    def verify_citation(self, rec, atom):
        hay = " ".join((rec.text or "").lower().split())
        needle = " ".join((atom or "").lower().split())
        if needle and needle in hay:
            return (NLILabel.ENTAILMENT, 1.0)
        return (NLILabel.NEUTRAL, 0.0)


def _claim_backed_result(store, retriever, *, subject, predicate, obj, answer, query,
                         proof_atom) -> StructuredAnswerResult:
    """Build a claim-backed StructuredAnswerResult whose single support resolves to a real
    ClaimRecord in the store (so the gate can read subject / predicate / object)."""
    rec = _record(proof_atom)
    store.upsert_record(rec)
    claim = ClaimRecord(
        claim_type="state",
        scope=_SCOPE,
        subject=subject,
        predicate=predicate,
        object=obj,
        source_memory_id=rec.memory_id,
        proof_atom=proof_atom,
        valid_at=10.0,
    )
    store.add_claim(claim)
    return StructuredAnswerResult(
        answer=answer,
        op="lookup",
        backend="claim",
        supports=[StructuredSupport(
            memory_id=rec.memory_id, claim_id=claim.claim_id,
            proof_atom=proof_atom, answer_atom=answer, score=1.0)],
        confidence=1.0,
        note="smqe:lookup:claim",
    )


def _settings_with(**overrides):
    # Settings is a frozen dataclass -> replace() to build an overridden copy.
    return dataclasses.replace(get_settings(), **overrides)




def test_unverified_benchmark_escape_hatches_are_removed():
    import bench.adapters.eidetic_adapter as adapter

    assert not hasattr(adapter, "_coverage_backed_abstain")
    assert not hasattr(adapter, "_dense_topk_fallback")
    assert not hasattr(adapter, "_dense_topk_blocks")


# --------------------------------------------------------------------------- #
# Lever 3 -- raw dense floor
# --------------------------------------------------------------------------- #
def test_l3_ordering_promotes_raw_above_audit_without_cropping_structured(tmp_path):
    """Lever 3 is applied by ORDERING, not head-reservation: the top-N raw dense passages are
    placed just ABOVE the low-value audit/pref channels. So under a tight budget the raw dense
    passage displaces a verbose AUDIT block (good) while every higher-priority STRUCTURED block
    that fits survives (finding #4). _budget_blocks itself is priority-order truncation."""
    from eidetic.retrieval import _budget_blocks

    structured = ["STRUCTURED-" + ("s" * 350)]          # ~360 chars, highest priority
    floor_raw = ["RAW-DENSE-TOP " + ("y" * 350)]         # ~364 chars, promoted above audit
    audit = ["AUDIT-" + ("x" * 350) for _ in range(10)]  # verbose, lowest priority
    token_budget = 250  # char_budget = 1000: fits structured + raw, crops into audit
    # New call site order: structured channels ... + floor_raw + audit + pref + rest_raw
    out = _budget_blocks(structured + floor_raw + audit, token_budget)
    joined = " ".join(out)
    assert "STRUCTURED-" in joined       # higher-priority structured never cropped out
    assert "RAW-DENSE-TOP" in joined     # raw dense promoted ahead of verbose audit


def test_l3_high_priority_structured_never_evicted_by_floor(tmp_path):
    """The exact finding-#4 scenario: with the floor placed after the structured channels,
    a fitting higher-priority structured block is NEVER evicted to make room for raw dense --
    because _budget_blocks fills strictly in the given priority order and the structured
    blocks come first."""
    from eidetic.retrieval import _budget_blocks

    structA = ["STRUCT-A " + ("a" * 300)]
    structB = ["STRUCT-B " + ("b" * 300)]
    floor_raw = ["RAW-DENSE " + ("y" * 4000)]  # huge, would eat everything if head-reserved
    token_budget = 200  # char_budget = 800: A+B fit (~616), raw only fills the residual
    out = _budget_blocks(structA + structB + floor_raw, token_budget)
    joined = " ".join(out)
    assert "STRUCT-A" in joined and "STRUCT-B" in joined  # both survive, neither evicted


def test_l3_budget_blocks_is_priority_truncation(tmp_path):
    """_budget_blocks reverts to plain single-arg priority truncation (no floor kwargs): the
    lowest-priority tail is cropped first, higher-priority blocks kept."""
    from eidetic.retrieval import _budget_blocks

    blocks = ["HEAD-" + ("h" * 500), "TAIL-" + ("t" * 500)]
    out = _budget_blocks(blocks, 100)  # char_budget = 400: HEAD alone overflows it
    joined = " ".join(out)
    assert "HEAD-" in joined and "TAIL-" not in joined
