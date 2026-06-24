"""Component 3: the cognitive-coordinate map (metadata structure-code).

Tolman-Eichenbaum inspiration, shipped as the honest metadata fallback (dossier 13.4):
each memory gets a second vector built from explicit STRUCTURE features -- entity
types/roles, modality, graph position (PPR / degree), and CYCLIC temporal coordinates
(hour-of-day, day-of-week). It is stored beside the content embedding and composed
at retrieval time for cross-context generalization.

Deliberate omission: absolute age / valid_at is NOT encoded. Putting recency into a
vector that influences ranking would slope the recall-vs-age curve -- the very thing
the project disproves. Temporal structure is cyclic only.

No model call here: structure is computed from metadata, deterministically.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np

from .models import MemoryRecord


def _bucket(name: str, dim: int) -> int:
    return int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % dim


def _sign(name: str) -> float:
    return 1.0 if int(hashlib.sha1(("s:" + name).encode()).hexdigest(), 16) % 2 == 0 else -1.0


def _add_feature(vec: np.ndarray, name: str, weight: float = 1.0) -> None:
    vec[_bucket(name, vec.shape[0])] += weight * _sign(name)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def build_structure_code(
    record: MemoryRecord,
    dim: int,
    graph_features: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Feature-hashed structure vector for a memory."""
    import time as _t

    vec = np.zeros(dim, dtype=np.float32)
    _add_feature(vec, f"modality:{record.modality.value}", 1.0)
    _add_feature(vec, f"source:{record.source}", 0.7)
    for ent in record.entities[:32]:
        _add_feature(vec, f"entity:{ent.lower()}", 1.0)

    # Scope as a coordinate axis (keeps namespaces structurally separated).
    _add_feature(vec, f"ns:{record.scope.namespace}", 0.8)
    if record.scope.agent_id:
        _add_feature(vec, f"agent:{record.scope.agent_id}", 0.5)
    if record.scope.project_id:
        _add_feature(vec, f"project:{record.scope.project_id}", 0.5)

    # Relational ROLE features (Frontier 7.1): which relations this memory's entities
    # participate in. Two memories with the same relational structure but different
    # content land near each other in STRUCTURE space and far apart in CONTENT space.
    if graph_features:
        for rel in graph_features.get("relations", [])[:16]:
            _add_feature(vec, f"role:{str(rel).lower()}", 0.9)

    # Cyclic temporal coordinate (time-of-day, day-of-week) -- NOT absolute age.
    lt = _t.localtime(record.valid_at)
    hour_angle = 2 * math.pi * lt.tm_hour / 24.0
    dow_angle = 2 * math.pi * lt.tm_wday / 7.0
    vec[_bucket("tcoord:hour_sin", dim)] += 0.5 * math.sin(hour_angle)
    vec[_bucket("tcoord:hour_cos", dim)] += 0.5 * math.cos(hour_angle)
    vec[_bucket("tcoord:dow_sin", dim)] += 0.5 * math.sin(dow_angle)
    vec[_bucket("tcoord:dow_cos", dim)] += 0.5 * math.cos(dow_angle)

    # Graph-position features (PPR centrality, degree) bucketed into coarse bins.
    if graph_features:
        ppr = float(graph_features.get("ppr", 0.0))
        deg = float(graph_features.get("degree", 0.0))
        vec[_bucket("graph:ppr_bin", dim)] += min(1.0, ppr * 10.0)
        vec[_bucket("graph:deg_bin", dim)] += min(1.0, deg / 10.0)

    return _normalize(vec)


def build_query_structure_code(entities: list[str], dim: int, modality: str = "text") -> np.ndarray:
    """Structure code for a query, from its extracted entities + modality."""
    vec = np.zeros(dim, dtype=np.float32)
    _add_feature(vec, f"modality:{modality}", 1.0)
    for ent in entities[:32]:
        _add_feature(vec, f"entity:{ent.lower()}", 1.0)
    return _normalize(vec)
