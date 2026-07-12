"""The search space and its hard wall.

ALLOWED_KNOBS is the Tier A mutable mind: answer-path knobs that do not rebuild the
index and do not touch ingest (the lab store is frozen, so an ingest-side knob would
silently measure nothing).

PROOF_DNA is the constitution: no proposer, pipeline, or promotion may ever name one
of these. It covers every verification floor and threshold, the abstention gates,
the contradiction proof gate, and RRF_W_RECENCY (age-neutral ranking is a published
claim -- evolution re-enabling recency would falsify it silently).

Both sets are asserted disjoint at import time and property-tested.
"""
from __future__ import annotations

# Tier A: one-knob hypotheses. (env var -> candidate values, mined from bench/sweep
# STAGES minus ingest-side and rebuild knobs.)
ALLOWED_KNOBS: dict[str, tuple[str, ...]] = {
    "READER_COT": ("0", "1"),
    "CONFLICT_RESOLVER": ("0", "1"),
    "COMPRESSION_RATIO": ("1.0", "0.75", "0.5"),
    "TEMPORAL_RERANK": ("0", "1"),
    "PERSISTENT_BM25": ("1", "0"),
    "RERANK_ENABLED": ("1", "0"),
    "RERANK_DEPTH": ("50", "100"),
    "CONTEXT_TOKEN_BUDGET": ("8000", "6000", "3000"),
    "ANN_TOPK": ("100", "200"),
    "FINAL_TOPK": ("10", "15"),
    "RRF_W_BM25": ("0.6", "1.0"),
    "RRF_W_GRAPH": ("0.8", "1.2"),
    "HNSW_EF_SEARCH": ("256", "500"),
    "FUSION_METHOD": ("rrf", "dbsf"),
    "ADAPTIVE_K": ("0", "1"),
    "MMR_ENABLED": ("0", "1"),
    "RERANK_SKIP_MARGIN": ("0.0", "0.05"),
    "ADAPTIVE_EF": ("0", "1"),
    # Track B read stage (verbatim claim selection; ships with the READ-recovery fix).
    "READ_CLAIM_SELECT": ("0", "1"),
}

# The constitution. Evolution can NEVER name these (proposer + pipeline compiler +
# swap path all refuse). Includes prefix-matched families.
PROOF_DNA: frozenset[str] = frozenset({
    # verification + abstention physics
    "ABSTENTION_THRESHOLD", "ABSTENTION_V2", "ABSTENTION_V2_TAU",
    "ABSTENTION_W_ENTAIL", "ABSTENTION_W_COVERAGE", "ABSTENTION_W_AGREEMENT",
    "ABSTENTION_W_PROOF", "FAST_ABSTAIN", "FAST_ABSTAIN_FLOOR",
    "VERIFY_MODEL", "VERIFY_NLI_CACHE", "VERIFY_CITATION_CAP", "FAST_VERIFY",
    "BATCH_NLI", "SPAN_NLI", "SPAN_NLI_MIN_CHARS", "COVE", "COVE_QUESTIONS",
    "READER_FORM_FLOOR", "READER_NUMERIC_FLOOR", "CONTRADICTION_PROOF_GATE",
    "CLAIM_GROUNDING_EARLY_STOP", "RAW_SPAN_MIN_CHARS",
    # published age-neutrality claim
    "RRF_W_RECENCY",
    # index rebuild (OtterTune blacklist)
    "HNSW_M", "HNSW_EF_CONSTRUCTION",
    # ingest-side (frozen lab store would measure nothing)
    "EXTRACT_LIGHT", "HIPPO2_SEEDING", "DREAM_AB", "FULL_SLEEP",
    "INGEST_GRANULARITY", "INGEST_WINDOW_TURNS",
    # the wall itself
    "AUTORESEARCH", "AUTORESEARCH_AUTO_DRAIN", "EPISTEMIC_MAP",
})

_PROOF_DNA_PREFIXES: tuple[str, ...] = ("ABSTENTION_", "VERIFY_", "NLI_")


def is_proof_dna(env_var: str) -> bool:
    name = (env_var or "").upper()
    return name in PROOF_DNA or name.startswith(_PROOF_DNA_PREFIXES)


def assert_hypothesis_env_legal(env: dict) -> None:
    """Raise if a hypothesis overlay names proof DNA or leaves the allowed space.
    OPERATOR_PIPELINE is legal (the compiler enforces its own whitelist)."""
    for key in env:
        k = key.upper()
        if is_proof_dna(k):
            raise ValueError(f"hypothesis names proof DNA: {key}")
        if k != "OPERATOR_PIPELINE" and k not in ALLOWED_KNOBS \
                and k != "CONTEXT_COMPRESS":     # stage_assignment companion of COMPRESSION_RATIO
            raise ValueError(f"hypothesis names a knob outside the allowed space: {key}")


_ILLEGAL = {k for k in ALLOWED_KNOBS if is_proof_dna(k)}
assert not _ILLEGAL, f"ALLOWED_KNOBS intersects PROOF_DNA: {_ILLEGAL}"
