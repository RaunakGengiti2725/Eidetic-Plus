"""Eidetic-Plus: a lossless, verifiable, recency-independent memory engine for AI agents.

Seven components (see docs/architecture.md):
  1. Immutable lossless content-addressed substrate      -> substrate.py
  2. Hippocampal index = vector ANN + bi-temporal graph   -> vector_index.py, graph.py
  3. Cognitive-coordinate map (metadata structure-code)   -> structure_code.py
  4. Write-time salience gate                             -> salience.py
  5. Offline consolidation/replay + FSRS forgetting       -> consolidation.py, fsrs.py
  6. Reconstructive, verifiable retrieval                 -> retrieval.py
  7. Provenance + contradiction engine                    -> graph.py, retrieval.py

All model calls are REAL Qwen/DashScope calls (dashscope_client.py). Nothing is mocked.
"""

__version__ = "1.0.0"
