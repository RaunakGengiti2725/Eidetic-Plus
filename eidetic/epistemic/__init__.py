"""The EPISTEMIC MAP: a live, derived, recomputable model of what this memory
Knows / does not know / Contests — over the immutable witness, never instead of it.

Three states for every claim-shaped cell (entity-relation, event date, query):
  KNOWN     — the system can produce a VERIFIED answer with citations (proof stored)
  UNKNOWN   — a gap the system can NAME (deterministically enumerated, or minted
              from an abstained ask/probe)
  CONTESTED — contradictory witnesses / unresolved supersession / an open NLI
              conflict (resolution is a research program, not a delete)

Design laws:
  * The map is a DERIVED layer in its own SQLite file. Dropping it loses nothing;
    `rebuild()` re-derives every enumerated cell from the store. It never holds the
    only copy of anything and never mutates witness bytes.
  * Unknown/Contested enumeration is DETERMINISTIC (zero LLM) so the map cannot be
    inflated: a judge can recompute it from the witness (gaps.py).
  * `mark_known` demands a VERIFIED answer with citations — a proof object is stored
    on the cell; there is no bare "trust me" transition to KNOWN.
  * Curiosity (curiosity.py) closes the frontier: template probes through the REAL
    prove path; contested cells route to the ContestedResolutionProgram
    (contested.py); law induction (laws.py) both predicts new gaps and, when
    falsified, mints CONTESTED cells.
"""
from .cells import CellKind, CellState, EpistemicCell
from .map import KnowledgeMap

__all__ = ["CellKind", "CellState", "EpistemicCell", "KnowledgeMap"]
