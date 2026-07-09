"""One-line entrypoint for the neutral harness.

    python -m bench.run --systems eidetic,rag-full,rag-vector,mem0,graphiti --dataset both --subset 20 --runs 1

Subset-first by default (a handful of questions) so it is demonstrably real cheaply; the
full >=10-run is `bash bench/reproduce.sh`. Renders the scoreboard + curves from the real
logs it just produced. Fail-loud: missing key / baseline lib / Neo4j raises clearly.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path

from . import curves, scoreboard
from .datasets import (beam, category_counts, filter_split, locomo, longmemeval,
                       memoryagentbench)
from .harness import run_system
from .judge import Judge
from eidetic.config import METABOLISM_PROFILE, apply_bench_full_consolidation_overlay

# Every score-affecting env flag is recorded so a claim reproduces from its manifest alone.
# The list is deliberately exhaustive; an unrecorded flag is a reproducibility hole the
# attribution program cannot tolerate (see plan: "Unrecorded env flags in any claim").
_MANIFEST_ENV = sorted(set([
    # Master switch + the full metabolism profile (unioned from the single source of truth, so
    # adding a profile flag automatically extends the manifest).
    "METABOLISM_MODE", *METABOLISM_PROFILE.keys(),
    # Storage path for release audits (snap-back must inspect the same store the run used).
    "DATA_DIR",
    # Shared reader / judge (the fairness pins).
    "READER_MODEL", "JUDGE_MODEL", "JUDGE_BASE_URL", "READER_MODE", "READER_BLOCK_CHARS",
    "READER_COT", "READER_ROUTER", "READER_TIER_A", "READER_TEMPORAL_SCAFFOLD",
    "READER_GATED_INFERENCE", "READER_LIST_TWOPASS", "READER_RECENCY_NUDGE",
    "READER_JSON_RESILIENT", "BENCH_BASELINE_LLM",
    # Capture / write path.
    "EXTRACT_CHUNKING", "EXTRACT_CHUNK_CHARS", "EXTRACT_CHUNK_OVERLAP", "EXTRACT_LIGHT",
    "EXTRACT_RESULT_CACHE", "EXTRACT_COMBINED",
    "CONSOLIDATION_EXTRACT_DEADLINE_SEC", "CONSOLIDATION_TIMEOUT_POLICY",
    "CONSOLIDATION_EXTRACT_CALL_BUDGET", "CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY",
    "CONSOLIDATION_RAW_ONLY_WINDOW_THRESHOLD",
    "BENCH_FULL_CONSOLIDATION",
    "MEMORY_TYPING", "PREF_SENTENCE_SCAN", "MEMORY_MANAGER", "INGEST_GRANULARITY",
    # Retrieval channels + fusion.
    "CONFLICT_RESOLVER", "CONTEXT_COMPRESS", "TEMPORAL_RERANK", "HIPPO2_SEEDING",
    "PERSISTENT_BM25", "GIST_CHANNEL", "STRUCT_CHANNEL", "EVENT_RANKING", "GRAPH_VOCAB_SEEDING",
    "COACTIVATION_CHANNEL", "EVENT_CHAIN_CONTEXT", "SCRATCHPAD", "ACTIVE_RETRIEVAL",
    "MARKOV_PREFETCH", "AFFECT_SALIENCE", "REFLEX_RECALL", "FLOW_ACTIVATION",
    "FLOW_HYBRID_CHANNEL", "FALSE_PREMISE", "CACHE_VERSIONING",
    "SEMANTIC_CACHE", "SEMANTIC_CACHE_ADAPTIVE",
    "RRF_W_GIST", "RRF_W_STRUCT", "RRF_W_EVENT", "RRF_W_COACT",
    # Caches + reranking + context budget.
    "DREAM_AB", "RERANK_ENABLED", "RERANK_FAIL_OPEN", "RERANK_DEPTH", "CONTEXT_TOKEN_BUDGET",
    "RAW_SPAN_AUDIT_TOPK", "RAW_SPAN_MIN_CHARS", "RAW_SPAN_PER_RECORD",
    "COMPRESSION_RATIO", "ANN_TOPK", "FINAL_TOPK", "RRF_K", "RRF_W_DENSE", "RRF_W_BM25",
    "RRF_W_GRAPH", "RRF_W_RECENCY", "HNSW_M", "HNSW_EF_SEARCH", "HNSW_EF_CONSTRUCTION",
    "SALIENCE_PRUNE_THRESHOLD",
    "CLAIM_EXTRACTION", "CRYSTAL_SPAN_DEMOTION", "CRYSTAL_SPAN_CHARS", "VIVID_FRACTION",
    # Proof / verification gate.
    "ABSTENTION_V2", "ABSTENTION_THRESHOLD", "ABSTENTION_V2_TAU", "BATCH_NLI", "FAST_VERIFY",
    "VERIFY_CITATION_CAP", "COVE", "SPAN_NLI", "DEFER_REEMBED",
    "EIDETIC_ENABLE_DATASET_SOURCE_SCANS",
    # Consolidation / dreaming.
    "FULL_SLEEP", "DREAM_REPAIR", "DREAM_REPAIR_APPLY", "DREAM_USE_LLM_NLI",
    "DREAM_INFER_CONFIDENCE", "DREAM_PRUNE_PERCENTILE",
    # Rate governor (affects concurrency, not scores, but reproducibility-relevant).
    "DASHSCOPE_MAX_CONCURRENCY", "DASHSCOPE_RPM", "DASHSCOPE_REQUEST_TIMEOUT_SEC",
    "DASHSCOPE_SLOT_TIMEOUT_SEC", "EMBED_BATCH_PARALLELISM",
    "STRICT_BASELINE_HEALTH",
    # Layer-2/3 optimizer flags.
    "FUSION_METHOD", "ADAPTIVE_K", "ADAPTIVE_K_MIN", "ADAPTIVE_EF", "HNSW_EF_SEARCH_HARD",
    "MMR_ENABLED", "MMR_LAMBDA", "RERANK_SKIP_MARGIN", "CONFORMAL_DEPTH", "CONFORMAL_ALPHA",
    "CONFORMAL_QHAT", "PARALLEL_CHANNELS", "VECTOR_QUANT", "QUANT_REFINE", "ROCCHIO",
    "FUSION_LEARNER", "FUSION_LEARNER_METHOD",
]))


def make_system(name: str):
    name = name.strip().lower()
    if name in ("eidetic", "eidetic-plus"):
        from .adapters.eidetic_adapter import EideticSystem
        return EideticSystem()
    if name == "mem0":
        from .adapters.mem0_adapter import Mem0System
        return Mem0System()
    if name == "graphiti":
        from .adapters.graphiti_adapter import GraphitiSystem
        return GraphitiSystem()
    if name == "hindsight":
        from .adapters.hindsight_adapter import HindsightSystem
        return HindsightSystem()
    if name in ("rag-full", "ragfull"):
        from .adapters.rag_adapter import RagFullSystem
        return RagFullSystem()
    if name in ("rag-vector", "ragvector", "rag"):
        from .adapters.rag_adapter import RagVectorSystem
        return RagVectorSystem()
    if name in ("eidetic-full", "eidetic-plus-full"):
        from .adapters.eidetic_adapter import EideticFullSystem
        return EideticFullSystem()
    if name in ("eidetic-product", "eidetic-plus-product"):
        from .adapters.eidetic_adapter import EideticProductSystem
        return EideticProductSystem()
    raise SystemExit(
        f"Unknown system '{name}' (choose from eidetic, eidetic-full, eidetic-product, mem0, "
        "graphiti, hindsight, rag-full, rag-vector).")


def _slice(samples: list, subset: int, offset: int) -> list:
    offset = max(0, offset)
    if subset and subset > 0:
        return samples[offset: offset + subset]
    return samples[offset:]


def _load_samples_file(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("--samples-file must be a JSON list of {dataset, sample_id} objects")
    rows: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"--samples-file row {idx} is not an object")
        dataset = str(item.get("dataset", "")).strip()
        sample_id = str(item.get("sample_id", "")).strip()
        if not dataset or not sample_id:
            raise ValueError(f"--samples-file row {idx} must include dataset and sample_id")
        rows.append({"dataset": dataset, "sample_id": sample_id})
    return rows


def _filter_samples_file(samples: list, rows: list[dict[str, str]]) -> list:
    by_key = {(s.dataset, s.sample_id): s for s in samples}
    out = []
    missing = []
    for row in rows:
        key = (row["dataset"], row["sample_id"])
        sample = by_key.get(key)
        if sample is None:
            missing.append({"dataset": key[0], "sample_id": key[1]})
        else:
            out.append(sample)
    if missing:
        raise ValueError(f"--samples-file references {len(missing)} unknown sample(s): {missing[:5]}")
    return out


def _conversation_prefix(sample_id: str) -> str:
    return str(sample_id).split("_q", 1)[0]


def _stratified_order(samples: list) -> list:
    """Deterministic round-robin by dataset/category, then conversation prefix.

    Contiguous benchmark files are often grouped: LoCoMo by conversation and LongMemEval by
    category. A small "stratified" slice must therefore balance categories first, then rotate
    conversation/sample prefixes inside each category, while preserving source order within each
    prefix. Otherwise a LongMemEval subset of 8 can still be eight single-session-user rows.
    """
    buckets: dict[tuple[str, str, str], list] = defaultdict(list)
    key_order: list[tuple[str, str, str]] = []
    for sample in samples:
        key = (sample.dataset, sample.category, _conversation_prefix(sample.sample_id))
        if key not in buckets:
            key_order.append(key)
        buckets[key].append(sample)

    strata: dict[tuple[str, str], deque[tuple[str, str, str]]] = defaultdict(deque)
    stratum_order: list[tuple[str, str]] = []
    for key in key_order:
        stratum = (key[0], key[1])
        if stratum not in strata:
            stratum_order.append(stratum)
        strata[stratum].append(key)

    out = []
    while True:
        progressed = False
        for stratum in stratum_order:
            keys = strata[stratum]
            for _ in range(len(keys)):
                key = keys.popleft()
                if buckets[key]:
                    out.append(buckets[key].pop(0))
                    progressed = True
                    if buckets[key]:
                        keys.append(key)
                    break
        if not progressed:
            return out


def _order_samples(samples: list, strategy: str) -> list:
    strategy = (strategy or "contiguous").strip().lower()
    if strategy == "contiguous":
        return samples
    if strategy == "stratified":
        return _stratified_order(samples)
    raise ValueError("sample_strategy must be one of: contiguous, stratified")


def load_samples(dataset: str, subset: int, variant: str, offset: int = 0,
                 split: str | None = None, sample_strategy: str = "contiguous"):
    """Load benchmark samples. `split` enforces the integrity wall: 'test' for
    reported runs, 'dev' for optimizers, None/'all' for ad-hoc full runs. The split
    filter is applied to the FULL dataset BEFORE the subset/offset slice, so a subset
    of the dev split never overlaps a subset of the test split."""
    def _take(loaded):
        return _slice(_order_samples(filter_split(loaded, split), sample_strategy), subset, offset)

    out = []
    if dataset in ("longmemeval", "both", "all"):
        out += _take(longmemeval.load(variant=variant, limit=None))
    if dataset in ("locomo", "both", "all"):
        out += _take(locomo.load(limit=None))
    if dataset in ("memoryagentbench", "all"):
        out += _take(memoryagentbench.load(limit=None))
    if dataset in ("beam", "all"):
        out += _take(beam.load(limit=None))
    return out


def _truthy_env(env: dict[str, str], key: str) -> bool:
    return str(env.get(key, "")).strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(str(env.get(key, default)).strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(str(env.get(key, default)).strip() or default)
    except (TypeError, ValueError):
        return default


def _eidetic_systems_requested(systems: str) -> bool:
    for raw in (systems or "").split(","):
        name = raw.strip().lower()
        if name in {
            "eidetic", "eidetic-plus", "eidetic-full", "eidetic-plus-full",
            "eidetic-product", "eidetic-plus-product",
        }:
            return True
    return False


def _session_text_len(session) -> int:
    return sum(len(t.content or "") + len(t.role or "") + 3 for t in session.turns)


def longmemeval_liveness_errors(samples: list, systems: str,
                                env: dict[str, str] | None = None) -> list[str]:
    """Fail-fast guard for LongMemEval-scale Eidetic runs.

    The public benchmark path relies on long-haystack raw-only fallback plus raw-span audit. Without
    those knobs, a huge LongMemEval session can spend the run budget on extraction timeouts before
    any useful score appears. This guard keeps accidental unsafe reruns from producing misleading
    "LongMemEval didn't complete" artifacts.
    """
    env = os.environ if env is None else env
    if not _eidetic_systems_requested(systems):
        return []
    lme_samples = [s for s in samples if getattr(s, "dataset", "") == "longmemeval"]
    if not lme_samples:
        return []

    seen: set[int] = set()
    long_sessions = []
    for sample in lme_samples:
        for session in sample.sessions:
            marker = id(session)
            if marker in seen:
                continue
            seen.add(marker)
            n = _session_text_len(session)
            if n > _env_int(env, "RAW_SPAN_MIN_CHARS", 12000):
                long_sessions.append((n, session))
    if not long_sessions:
        return []

    chunk_chars = max(1, _env_int(env, "EXTRACT_CHUNK_CHARS", 4000))
    max_chars = max(n for n, _ in long_sessions)
    total_windows = sum(max(1, (n + chunk_chars - 1) // chunk_chars) for n, _ in long_sessions)
    max_windows = max(max(1, (n + chunk_chars - 1) // chunk_chars) for n, _ in long_sessions)
    raw_only_threshold = _env_int(env, "CONSOLIDATION_RAW_ONLY_WINDOW_THRESHOLD", 0)
    call_budget = _env_int(env, "CONSOLIDATION_EXTRACT_CALL_BUDGET", 0)
    deadline = _env_float(env, "CONSOLIDATION_EXTRACT_DEADLINE_SEC", 0.0)
    policy = str(env.get("CONSOLIDATION_TIMEOUT_POLICY", "degrade")).strip().lower()
    raw_only_by_record = raw_only_threshold > 0 and max_windows > raw_only_threshold
    raw_only_by_batch = (
        call_budget > 0
        and _truthy_env(env, "CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY")
        and total_windows > max(1, int(call_budget * 0.8))
    )

    errors: list[str] = []
    if _truthy_env(env, "BENCH_FULL_CONSOLIDATION"):
        if not _truthy_env(env, "EXTRACT_CHUNKING"):
            errors.append("set EXTRACT_CHUNKING=1")
        if raw_only_threshold != 0:
            errors.append("set CONSOLIDATION_RAW_ONLY_WINDOW_THRESHOLD=0")
        if call_budget != 0:
            errors.append("set CONSOLIDATION_EXTRACT_CALL_BUDGET=0")
        if _truthy_env(env, "CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY"):
            errors.append("set CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY=0")
        if policy == "defer":
            errors.append("use CONSOLIDATION_TIMEOUT_POLICY=degrade, not defer")
        if errors:
            prefix = (
                "Unsafe LongMemEval full-consolidation profile: "
                f"{len(long_sessions)} long session(s), max_chars={max_chars}, "
                f"planned_windows={total_windows}. "
            )
            return [prefix + "; ".join(errors)]
        return []
    if deadline <= 0.0:
        errors.append("set CONSOLIDATION_EXTRACT_DEADLINE_SEC>0")
    if policy == "defer":
        errors.append("use CONSOLIDATION_TIMEOUT_POLICY=degrade, not defer")
    if not _truthy_env(env, "RAW_SPAN_AUDIT"):
        errors.append("set RAW_SPAN_AUDIT=1")
    if not (raw_only_by_record or raw_only_by_batch):
        errors.append(
            "enable long-haystack bounding via CONSOLIDATION_RAW_ONLY_WINDOW_THRESHOLD below "
            f"max planned windows ({max_windows}) or CONSOLIDATION_EXTRACT_CALL_BUDGET with "
            "CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY=1"
        )
    if errors:
        prefix = (
            "Unsafe LongMemEval Eidetic liveness profile: "
            f"{len(long_sessions)} long session(s), max_chars={max_chars}, "
            f"planned_windows={total_windows}. "
        )
        return [prefix + "; ".join(errors)]
    return []


def _manifest_log_facts(out: Path) -> dict:
    rows: dict[tuple[str, str, str], dict] = {}
    systems: set[str] = set()
    datasets: set[str] = set()
    run_indices: set[int] = set()
    for p in sorted(Path(out).glob("*__run*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            dataset = str(row.get("dataset", "")).strip()
            sample_id = str(row.get("sample_id", "")).strip()
            category = str(row.get("category", "")).strip()
            system = str(row.get("system", "")).strip()
            if system:
                systems.add(system)
            if dataset:
                datasets.add(dataset)
            try:
                run_indices.add(int(row["run_idx"]))
            except (KeyError, TypeError, ValueError):
                pass
            if not dataset or not sample_id:
                continue
            key = (dataset, sample_id, category)
            rows.setdefault(key, {
                "dataset": dataset,
                "sample_id": sample_id,
                "category": category,
            })
    return {
        "sample_rows": [rows[key] for key in sorted(rows)],
        "systems": sorted(systems),
        "datasets": sorted(datasets),
        "run_indices": sorted(run_indices),
    }


def _manifest_sample_rows_from_logs(out: Path) -> list[dict]:
    return _manifest_log_facts(out)["sample_rows"]


def _category_counts_from_sample_rows(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("category", ""))] += 1
    return dict(counts)


def _read_existing_manifest(out: Path) -> dict:
    path = Path(out) / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _dataset_from_log_datasets(datasets: list[str], fallback: str) -> str:
    if not datasets:
        return fallback
    if datasets == ["locomo", "longmemeval"]:
        return "both"
    if len(datasets) == 1:
        return datasets[0]
    return "all"


def write_manifest(out: Path, args, judge_desc: dict, samples: list | None = None,
                   system_failures: list[dict] | None = None) -> Path:
    previous_manifest = _read_existing_manifest(out) if samples is None else {}
    preserve_previous = bool(previous_manifest) and not bool(previous_manifest.get("render_only"))
    if samples is not None:
        sample_rows = [
            {"dataset": s.dataset, "sample_id": s.sample_id, "category": s.category}
            for s in samples
        ]
        sample_count = len(samples)
        sample_category_counts = category_counts(samples)
        log_facts: dict = {}
    else:
        log_facts = _manifest_log_facts(out)
        previous_rows = previous_manifest.get("sample_rows", []) if preserve_previous else []
        sample_rows = log_facts["sample_rows"] or (previous_rows if isinstance(previous_rows, list) else [])
        sample_count = len(sample_rows)
        sample_category_counts = _category_counts_from_sample_rows(sample_rows)
    log_systems = log_facts.get("systems", [])
    log_datasets = log_facts.get("datasets", [])
    log_runs = log_facts.get("run_indices", [])
    previous_env = (
        previous_manifest.get("env", {})
        if preserve_previous and isinstance(previous_manifest.get("env"), dict)
        else {}
    )
    manifest_env = previous_env or {k: os.environ.get(k, "") for k in _MANIFEST_ENV}
    manifest_system_failures = (
        system_failures
        if system_failures is not None
        else (previous_manifest.get("system_failures", []) if preserve_previous else [])
    )
    manifest = {
        "systems": (
            ",".join(log_systems)
            if log_systems
            else (previous_manifest.get("systems", args.systems) if preserve_previous else args.systems)
        ),
        "dataset": _dataset_from_log_datasets(
            log_datasets,
            previous_manifest.get("dataset", args.dataset) if preserve_previous else args.dataset,
        ),
        "split": (
            previous_manifest.get("split", getattr(args, "split", "all"))
            if preserve_previous else getattr(args, "split", "all")
        ),
        "subset": previous_manifest.get("subset", args.subset) if preserve_previous else args.subset,
        "sample_offset": (
            previous_manifest.get("sample_offset", args.sample_offset)
            if preserve_previous else args.sample_offset
        ),
        "sample_strategy": (
            previous_manifest.get("sample_strategy", getattr(args, "sample_strategy", "contiguous"))
            if preserve_previous else getattr(args, "sample_strategy", "contiguous")
        ),
        "samples_file": (
            previous_manifest.get("samples_file", getattr(args, "samples_file", ""))
            if preserve_previous else getattr(args, "samples_file", "")
        ),
        "holdout_profile": (
            previous_manifest.get("holdout_profile", getattr(args, "holdout_profile", "dev"))
            if preserve_previous else getattr(args, "holdout_profile", "dev")
        ),
        "runs": (
            len(log_runs)
            if log_runs
            else (previous_manifest.get("runs", args.runs) if preserve_previous else args.runs)
        ),
        "run_offset": (
            min(log_runs)
            if log_runs
            else (previous_manifest.get("run_offset", args.run_offset) if preserve_previous else args.run_offset)
        ),
        "variant": previous_manifest.get("variant", args.variant) if preserve_previous else args.variant,
        "render_only": bool(args.render_only),
        "judge": judge_desc,
        "sample_count": sample_count,
        "category_counts": sample_category_counts,
        "sample_rows": sample_rows,
        "system_failures": manifest_system_failures or [],
        "metabolism_mode": (
            previous_manifest.get("metabolism_mode")
            if preserve_previous and "metabolism_mode" in previous_manifest
            else os.environ.get("METABOLISM_MODE", "0").strip().lower()
            in ("1", "true", "yes", "on")
        ),
        "env": manifest_env,
    }
    path = out / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Eidetic-Plus neutral benchmark harness")
    ap.add_argument("--systems", default="eidetic",
                    help="comma list: eidetic, eidetic-full, eidetic-product, mem0, graphiti, "
                         "rag-full, rag-vector")
    ap.add_argument("--dataset", default="locomo",
                    choices=["longmemeval", "locomo", "memoryagentbench", "beam", "both", "all"])
    ap.add_argument("--subset", type=int, default=10, help="limit samples per dataset (<=0 = full)")
    ap.add_argument("--sample-offset", type=int, default=0, help="start offset for fresh-slice reruns")
    ap.add_argument("--sample-strategy", default="contiguous",
                    choices=["contiguous", "stratified"],
                    help="contiguous preserves loader order; stratified round-robins by category "
                         "and conversation prefix before applying subset/offset")
    ap.add_argument("--samples-file", default="",
                    help="JSON list of {dataset, sample_id} rows. When set, the exact listed samples "
                         "are used in file order and subset/offset are ignored.")
    ap.add_argument("--holdout-profile", default="dev", choices=["dev", "holdout", "legacy"],
                    help="dev allows tuning; holdout is release-grade and forbids legacy rescue flags; "
                         "legacy is for historical reports only.")
    ap.add_argument("--runs", type=int, default=1, help="independent runs (>=10 for variance)")
    ap.add_argument("--run-offset", type=int, default=0, help="first run index, to append without clobbering")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing run log files")
    ap.add_argument("--variant", default="longmemeval_s")
    ap.add_argument("--out", default="artifacts/bench")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"],
                    help="integrity wall: 'test' = reported runs, 'dev' = optimizer split, "
                         "'all' = full (ad-hoc). Reported numbers MUST use 'test'.")
    ap.add_argument("--render-only", action="store_true", help="re-render from existing logs")
    ap.add_argument("--allow-unsafe-longmemeval", action="store_true",
                    help="bypass the LongMemEval liveness preflight for exploratory debugging")
    args = ap.parse_args()
    if args.sample_offset < 0:
        raise SystemExit("--sample-offset must be >= 0")
    if args.run_offset < 0:
        raise SystemExit("--run-offset must be >= 0")
    if not args.render_only and args.runs <= 0:
        raise SystemExit("--runs must be > 0 unless --render-only is used")
    apply_bench_full_consolidation_overlay(os.environ)
    if args.holdout_profile == "holdout":
        if not args.samples_file:
            raise SystemExit("--holdout-profile holdout requires --samples-file")
        if _truthy_env(os.environ, "EIDETIC_ENABLE_DATASET_SOURCE_SCANS"):
            raise SystemExit("holdout profile forbids EIDETIC_ENABLE_DATASET_SOURCE_SCANS")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = Judge()
    judge_desc = judge.describe()
    print(f"Judge: {judge_desc}")
    loaded_samples = None

    if not args.render_only:
        if args.samples_file:
            rows = _load_samples_file(args.samples_file)
            all_samples = load_samples(args.dataset, 0, args.variant, 0,
                                       split=args.split, sample_strategy="contiguous")
            samples = _filter_samples_file(all_samples, rows)
        else:
            samples = load_samples(args.dataset, args.subset, args.variant, args.sample_offset,
                                   split=args.split, sample_strategy=args.sample_strategy)
        loaded_samples = samples
        print(f"Loaded {len(samples)} samples ({args.dataset}, split={args.split}); "
              f"categories: {category_counts(samples)}")
        if not samples:
            raise SystemExit("No samples loaded. Check --dataset, --subset, and --sample-offset.")
        if not args.allow_unsafe_longmemeval:
            errors = longmemeval_liveness_errors(samples, args.systems)
            if errors:
                raise SystemExit(errors[0])
        failures: list[dict] = []
        for raw in args.systems.split(","):
            # Per-system isolation: one system failing (e.g. a baseline lib/API issue) must
            # NOT lose the results of the others. Record it and still render the scoreboard.
            try:
                system = make_system(raw)
                print(f"Running {system.name}: {len(samples)} samples x {args.runs} run(s)...")
                run_system(system, samples, judge, runs=args.runs, out_dir=out,
                           run_offset=args.run_offset, overwrite=args.overwrite)
            except Exception as e:
                if isinstance(e, FileExistsError):
                    raise
                failures.append({
                    "system": raw.strip(),
                    "error_type": type(e).__name__,
                    "error": str(e)[:1000],
                })
                print(f"  !! {raw.strip()} FAILED (continuing): {type(e).__name__}: {str(e)[:200]}")
        if failures:
            print("\nSystems that failed (scoreboard still renders for the rest):")
            for failure in failures:
                print(f"  - {failure['system']}: {failure['error_type']}: "
                      f"{failure['error'][:160]}")

    manifest = write_manifest(out, args, judge_desc, loaded_samples,
                              system_failures=failures if not args.render_only else None)
    md = scoreboard.render(out, judge_desc)
    cv = curves.render(out)
    print(f"\nScoreboard -> {md}")
    print(f"Curves     -> {cv}")
    print(f"Manifest   -> {manifest}")
    print("Reproduce (full, >=10 runs): bash bench/reproduce.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
