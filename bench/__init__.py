"""Neutral benchmark harness: runs Eidetic-Plus, RAG baselines, Mem0, and Graphiti through ONE
answerer-and-judge by construction, on LongMemEval and LoCoMo, with multi-run variance.

The harness is the deliverable that turns "we beat them" from a claim into a scoreboard.
Discipline (from the spec): one fixed judge + one fixed reader prompt across all
systems; restrict LoCoMo to the four validated categories; report mean +/- variance over
multiple runs; publish raw per-question logs + a one-line reproduce command. No mocks; a
missing key fails loud. A number that does not reproduce does not exist -- so the
scoreboard/curves render ONLY from real run logs, never from invented numbers.
"""

__all__ = [
    "adapters", "datasets", "judge", "harness", "scoreboard", "curves",
    "release_gate", "claim_scope", "calibration_handoff", "merge_artifacts",
    "build_holdout_registry",
    "affect_salience_invariant", "scratchpad_invariant", "region_routing_invariant",
    "reflex_recall_invariant",
    "smqe_planner_invariant", "smqe_synthetic_invariant", "smqe_claim_coverage",
    "smqe_fullpath_invariant", "smqe_paraphrase_invariant",
    "smqe_conflict_invariant", "smqe_composition_invariant", "smqe_relative_phrase_invariant",
    "smqe_temporal_window_invariant", "smqe_attribution_invariant",
    "smqe_abstention_invariant", "smqe_scope_invariant", "smqe_subscope_invariant",
    "smqe_time_invariant", "smqe_invalidation_invariant",
]
