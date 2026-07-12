"""Rule-based hypothesis proposer: failure class -> ordered mind-layer changes.

Priors are MINED FROM FORENSICS, not vibes -- each entry names the evidence that put
it there (r19 read-recovery classification, r16/r17 sweep history, dominance-ledger
slices). One hypothesis = one knob (Tier A) or one pipeline (Tier B) or one law
(Tier C). ResearchMemory blocks exact repeats, so the loop advances through a
class's priors instead of rediscovering a dead end.

The proposer CANNOT name proof DNA: Tier A draws only from ALLOWED_KNOBS, Tier B
compiles through the operator whitelist, and the trial runner re-validates the final
env. Three walls, one property test.
"""
from __future__ import annotations

from typing import Iterator, Optional

from .registry import ResearchMemory
from .space import ALLOWED_KNOBS
from .types import FailureClass, ResearchHypothesis, ResearchTask

# Tier B pipeline priors (referenced below). claim_select is the r19-verdict stage:
# 10/11 entail-failures had the gold IN candidates -- the draft, not the evidence,
# failed proof (artifacts/forensics/read_recovery_r19.json).
_P_CLAIM_SELECT = {
    "retrieve": {"channels": ["dense", "bm25", "graph"], "fusion": "rrf"},
    "read": ["rerank", "claim_select"],
}
_P_DEEP_RETRIEVE = {
    "retrieve": {"channels": ["dense", "bm25", "graph"],
                 "weights": {"bm25": 1.0, "graph": 1.2},
                 "ann_topk": 200, "final_topk": 15, "fusion": "rrf"},
    "read": ["rerank"],
}
_P_TEMPORAL_READ = {
    "retrieve": {"channels": ["dense", "bm25", "graph"], "fusion": "rrf"},
    "read": ["rerank", "temporal_rerank", "claim_select"],
}
_P_WIDE_CONTEXT = {
    "retrieve": {"channels": ["dense", "bm25", "graph"], "final_topk": 15, "fusion": "rrf"},
    "read": ["rerank", "mmr"],
}

# (tier, knob-or-pipeline, value, rationale) priors per failure class, best-first.
_PRIORS: dict[FailureClass, list[tuple]] = {
    FailureClass.ENTAIL_FAILURE: [
        ("B", _P_CLAIM_SELECT, "", "r19: 10/11 entail-failures had gold IN candidates; "
                                   "verbatim claim selection makes the draft provable"),
        ("A", "READER_COT", "1", "draft wording drifts from sources; CoT reader states "
                                 "the fact sentence before answering"),
        ("A", "FINAL_TOPK", "15", "more sources -> more chances one entails the draft"),
        ("A", "RERANK_DEPTH", "100", "deeper rerank surfaces the entailing record"),
    ],
    FailureClass.HARD_TO_RETRIEVE: [
        ("B", _P_DEEP_RETRIEVE, "", "present-but-unproven: widen every retrieve stage"),
        ("A", "ANN_TOPK", "200", "wider candidate pool"),
        ("A", "RRF_W_GRAPH", "1.2", "associative hop reaches the fragmented evidence"),
        ("A", "MMR_ENABLED", "1", "diversity keeps the minority-topic record in topk"),
    ],
    FailureClass.MISSING_KNOWLEDGE: [
        ("A", "RRF_W_BM25", "1.0", "lexical channel finds what dense missed"),
        ("A", "ANN_TOPK", "200", "the gap may be reachable, just below the cut"),
        ("B", _P_DEEP_RETRIEVE, "", "full-width retrieve before declaring it missing"),
    ],
    FailureClass.LOW_COVERAGE: [
        ("A", "RRF_W_BM25", "1.0", "coverage is dense-similarity; lexical rescues it"),
        ("A", "ANN_TOPK", "200", "coverage floor needs a nearer neighbor"),
        ("B", _P_DEEP_RETRIEVE, "", "joint widen"),
    ],
    FailureClass.TEMPORAL_SELECTION: [
        ("B", _P_TEMPORAL_READ, "", "temporal rerank + claim_select: date evidence "
                                    "selected then quoted verbatim"),
        ("A", "TEMPORAL_RERANK", "1", "order candidates by query-time overlap"),
    ],
    FailureClass.LATEST_VALUE_SELECTION: [
        ("A", "CONFLICT_RESOLVER", "1", "freshest-timestamp discipline on current-value asks"),
        ("A", "TEMPORAL_RERANK", "1", "recency-of-VALIDITY ordering (not memory age)"),
    ],
    FailureClass.SPAN_UNGROUNDED: [
        ("B", _P_CLAIM_SELECT, "", "quote the grounding sentence instead of restating"),
        ("A", "CONTEXT_TOKEN_BUDGET", "8000", "the ungrounded sentence's evidence may "
                                              "have been compressed out"),
    ],
    FailureClass.CONTESTED_CONFLICT: [
        ("A", "CONFLICT_RESOLVER", "1", "deterministic current-value resolution"),
        ("B", _P_WIDE_CONTEXT, "", "both sides of the conflict in context"),
    ],
    FailureClass.SUGGESTION_SYNTH: [
        ("A", "READER_COT", "1", "fragment answers: reasoned reader completes them"),
    ],
    FailureClass.VERIFIED_WRONG: [
        ("A", "RERANK_DEPTH", "100", "the right record exists deeper in the pool"),
        ("B", _P_WIDE_CONTEXT, "", "wrong-record wins shrink when context widens"),
    ],
    FailureClass.REPAIR_PROPOSAL: [
        ("A", "RRF_W_GRAPH", "1.2", "MERGE-shaped repairs: strengthen the link channel"),
    ],
    FailureClass.SURPRISE_INGEST: [
        ("A", "MMR_ENABLED", "1", "novel-cluster memories survive dedup diversity"),
    ],
    FailureClass.KNOB_IMBALANCE: [],   # filled by generic ladder below
}

# Generic ladder when a class has no prior left: walk ALLOWED_KNOBS off-default.
_GENERIC_LADDER: list[tuple[str, str, str]] = [
    ("FUSION_METHOD", "dbsf", "distribution-based fusion (generic ladder)"),
    ("ADAPTIVE_K", "1", "query-adaptive depth (generic ladder)"),
    ("RERANK_SKIP_MARGIN", "0.05", "skip rerank on clear wins (generic ladder)"),
    ("HNSW_EF_SEARCH", "500", "wider graph search (generic ladder)"),
    ("PERSISTENT_BM25", "0", "in-memory bm25 variant (generic ladder)"),
]


def propose(task: ResearchTask, memory: ResearchMemory,
            *, laws: Optional[list] = None) -> Iterator[ResearchHypothesis]:
    """Yield untried hypotheses for this task, best prior first, then the generic
    ladder, then (Tier C) any pending candidate law whose class matches."""
    seen_keys = {l.get("hypothesis_key") for l in memory.lessons()}

    def _mk(tier: str, knob_or_pipe, value: str, rationale: str) -> ResearchHypothesis:
        if tier == "A":
            return ResearchHypothesis(
                tier="A", failure_class=task.failure_class, rationale=rationale,
                knob=knob_or_pipe, value=value,
                origin_cell=task.cell_id, origin_task_query=task.query[:200])
        return ResearchHypothesis(
            tier="B", failure_class=task.failure_class, rationale=rationale,
            pipeline=knob_or_pipe,
            origin_cell=task.cell_id, origin_task_query=task.query[:200])

    for tier, knob_or_pipe, value, rationale in _PRIORS.get(task.failure_class, []):
        if tier == "A" and knob_or_pipe not in ALLOWED_KNOBS:
            continue                                   # priors can never leave the space
        hyp = _mk(tier, knob_or_pipe, value, rationale)
        if hyp.key not in seen_keys:
            yield hyp
    for knob, value, rationale in _GENERIC_LADDER:
        if knob not in ALLOWED_KNOBS:
            continue
        hyp = _mk("A", knob, value, rationale)
        if hyp.key not in seen_keys:
            yield hyp
    for law in laws or []:
        hyp = ResearchHypothesis(
            tier="C", failure_class=task.failure_class,
            rationale=f"candidate law check: {getattr(law, 'text', lambda: law)()}",
            law_id=str(getattr(law, "law_id", "") or getattr(law, "text", lambda: "")()),
            origin_cell=task.cell_id, origin_task_query=task.query[:200])
        if hyp.key not in seen_keys:
            yield hyp


def propose_one(task: ResearchTask, memory: ResearchMemory,
                *, laws: Optional[list] = None) -> Optional[ResearchHypothesis]:
    return next(propose(task, memory, laws=laws), None)
