"""Neutral benchmark harness: runs Eidetic-Plus, Mem0, and Graphiti through ONE
answerer-and-judge by construction, on LongMemEval and LoCoMo, with multi-run variance.

The harness is the deliverable that turns "we beat them" from a claim into a scoreboard.
Discipline (from the spec): one fixed judge + one fixed reader prompt across all three
systems; restrict LoCoMo to the four validated categories; report mean +/- variance over
multiple runs; publish raw per-question logs + a one-line reproduce command. No mocks; a
missing key fails loud. A number that does not reproduce does not exist -- so the
scoreboard/curves render ONLY from real run logs, never from invented numbers.
"""

__all__ = ["adapters", "datasets", "judge", "harness", "scoreboard", "curves"]
