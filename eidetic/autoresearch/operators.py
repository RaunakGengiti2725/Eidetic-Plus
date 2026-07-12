"""Tier B: the memory-operator DSL. Programs over HOW to know, never over proof.

An OperatorPipeline is a small JSON program:

    {
      "retrieve": {"channels": ["dense", "bm25", "graph"],
                    "weights": {"bm25": 0.6, "graph": 0.8},
                    "fusion": "rrf", "ann_topk": 100, "final_topk": 10},
      "read":     ["rerank", "mmr", "temporal_rerank", "adaptive_k",
                    "compress:0.75", "reader_cot", "claim_select"]
    }

`compile_pipeline` lowers it ONTO EXISTING EXECUTION PATHS ONLY -- every op maps to
a real, already-tested env knob; there is no interpreter in the hot path, so a
promoted pipeline is exactly as debuggable as any config. The whitelist is closed:
`prove` is not an op, no abstention/NLI surface is reachable, and the compiled env
is re-validated against the PROOF_DNA wall before it can run or be promoted.

Executed-stage honesty: the compiled env carries EXPECT_STAGES (a declarative list);
the lab asserts after each eval that the recall trace/telemetry shows the declared
channels actually participated. A pipeline that silently no-ops cannot win a trial.
"""
from __future__ import annotations

import json

from .space import assert_hypothesis_env_legal

_RETRIEVE_CHANNELS = ("dense", "bm25", "graph")
_READ_OPS = ("rerank", "mmr", "temporal_rerank", "adaptive_k", "reader_cot",
             "claim_select")          # compress:<ratio> handled structurally
_FUSIONS = ("rrf", "dbsf")
_WEIGHT_RANGE = (0.0, 2.0)
_TOPK_RANGE = (5, 400)


class PipelineError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise PipelineError(msg)


def validate_pipeline(pipeline: dict) -> None:
    _check(isinstance(pipeline, dict) and pipeline, "pipeline must be a non-empty dict")
    unknown_keys = set(pipeline) - {"retrieve", "read"}
    _check(not unknown_keys, f"unknown pipeline sections: {sorted(unknown_keys)}")
    retrieve = pipeline.get("retrieve", {})
    _check(isinstance(retrieve, dict), "retrieve must be a dict")
    channels = retrieve.get("channels", list(_RETRIEVE_CHANNELS))
    _check(isinstance(channels, list) and channels, "retrieve.channels must be non-empty")
    bad = [c for c in channels if c not in _RETRIEVE_CHANNELS]
    _check(not bad, f"unknown retrieve channels: {bad}")
    _check("dense" in channels, "the dense channel is the substrate access path and "
                                "cannot be compiled out")
    weights = retrieve.get("weights", {})
    _check(isinstance(weights, dict), "retrieve.weights must be a dict")
    for name, w in weights.items():
        _check(name in ("bm25", "graph"), f"weight for unknown channel: {name}")
        _check(_WEIGHT_RANGE[0] <= float(w) <= _WEIGHT_RANGE[1],
               f"weight {name}={w} outside {_WEIGHT_RANGE}")
    fusion = retrieve.get("fusion", "rrf")
    _check(fusion in _FUSIONS, f"unknown fusion: {fusion}")
    for key in ("ann_topk", "final_topk"):
        if key in retrieve:
            _check(_TOPK_RANGE[0] <= int(retrieve[key]) <= _TOPK_RANGE[1],
                   f"{key}={retrieve[key]} outside {_TOPK_RANGE}")
    read = pipeline.get("read", [])
    _check(isinstance(read, list), "read must be a list of ops")
    for op in read:
        if isinstance(op, str) and op.startswith("compress:"):
            ratio = float(op.split(":", 1)[1])
            _check(0.25 <= ratio <= 1.0, f"compress ratio {ratio} outside [0.25, 1.0]")
            continue
        _check(op in _READ_OPS, f"unknown read op: {op!r} (prove is not an op; "
                                "verification is DNA, not search space)")


def compile_pipeline(pipeline: dict) -> dict[str, str]:
    """Lower a validated pipeline to its env overlay (existing knobs only)."""
    validate_pipeline(pipeline)
    env: dict[str, str] = {}
    retrieve = pipeline.get("retrieve", {})
    channels = retrieve.get("channels", list(_RETRIEVE_CHANNELS))
    weights = retrieve.get("weights", {})
    env["RRF_W_BM25"] = str(float(weights.get("bm25", 0.6))) if "bm25" in channels else "0.0"
    env["RRF_W_GRAPH"] = str(float(weights.get("graph", 0.8))) if "graph" in channels else "0.0"
    env["FUSION_METHOD"] = str(retrieve.get("fusion", "rrf"))
    if "ann_topk" in retrieve:
        env["ANN_TOPK"] = str(int(retrieve["ann_topk"]))
    if "final_topk" in retrieve:
        env["FINAL_TOPK"] = str(int(retrieve["final_topk"]))
    read = pipeline.get("read", [])
    env["RERANK_ENABLED"] = "1" if "rerank" in read else "0"
    env["MMR_ENABLED"] = "1" if "mmr" in read else "0"
    env["TEMPORAL_RERANK"] = "1" if "temporal_rerank" in read else "0"
    env["ADAPTIVE_K"] = "1" if "adaptive_k" in read else "0"
    env["READER_COT"] = "1" if "reader_cot" in read else "0"
    if "claim_select" in read:
        env["READ_CLAIM_SELECT"] = "1"
    for op in read:
        if isinstance(op, str) and op.startswith("compress:"):
            env["CONTEXT_COMPRESS"] = "1"
            env["COMPRESSION_RATIO"] = str(float(op.split(":", 1)[1]))
    # Executed-stage declaration for the lab's honesty assert (not a Settings knob).
    env["EXPECT_STAGES"] = json.dumps(sorted(channels))
    assert_hypothesis_env_legal({k: v for k, v in env.items() if k != "EXPECT_STAGES"})
    return env


def expected_stages(env: dict) -> list[str]:
    try:
        return json.loads(env.get("EXPECT_STAGES", "[]"))
    except (TypeError, ValueError):
        return []
