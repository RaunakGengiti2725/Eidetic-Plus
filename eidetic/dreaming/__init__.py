"""The Dreaming Engine: a TOKEN-FREE, offline, continuous consolidation layer.

While idle, Eidetic-Plus replays/reinforces (replay.py), derives implied facts via
KG-embedding link prediction (kg_embed.py) + Horn-rule mining (rules.py) gated into a
separate INFERRED layer (gate.py, infer.py), forms multi-resolution gist/schema centroids
(multires.py), and pre-assembles likely answers (prefetch.py).

Cardinal rules (enforced by tests in tests/test_dreaming.py):
  * The lossless store is SACRED -- consolidation NEVER merges/averages/mutates it. Every
    output is an ADDITIVE, reversible, content-addressed, provenance-tagged DERIVED record
    (models.DerivedRecord) or a flagged inferred Edge -- never written into observed memory.
  * Every machine-inferred edge/fact is confidence-gated (token-free) into a SEPARATE
    inferred layer, never presented as observed fact. Real-NLI is optional enrichment.
  * Every operation is near-linear or incremental -- NEVER naive all-pairs O(N^2).
  * ZERO LLM calls in the default path (no tokens): only local math over stored
    embeddings + the graph. A single O(N^2) graph rebuild already hung a live run.
"""

__all__ = ["gate", "kg_embed", "rules", "multires", "prefetch", "replay", "infer"]
