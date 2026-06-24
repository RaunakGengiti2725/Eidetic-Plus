"""One-line entrypoint for the neutral harness.

    python -m bench.run --systems eidetic,mem0,graphiti --dataset both --subset 20 --runs 1

Subset-first by default (a handful of questions) so it is demonstrably real cheaply; the
full >=10-run is `bash bench/reproduce.sh`. Renders the scoreboard + curves from the real
logs it just produced. Fail-loud: missing key / baseline lib / Neo4j raises clearly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import curves, scoreboard
from .datasets import beam, category_counts, locomo, longmemeval, memoryagentbench
from .harness import run_system
from .judge import Judge

_MANIFEST_ENV = [
    "READER_MODEL", "JUDGE_MODEL", "READER_COT", "READER_ROUTER", "CONFLICT_RESOLVER",
    "CONTEXT_COMPRESS", "EXTRACT_LIGHT", "TEMPORAL_RERANK", "HIPPO2_SEEDING", "PERSISTENT_BM25",
    "SEMANTIC_CACHE", "SEMANTIC_CACHE_ADAPTIVE",
    "DREAM_AB", "RERANK_ENABLED", "RERANK_FAIL_OPEN", "RERANK_DEPTH", "CONTEXT_TOKEN_BUDGET",
    "COMPRESSION_RATIO", "ANN_TOPK", "FINAL_TOPK", "RRF_K", "RRF_W_DENSE", "RRF_W_BM25",
    "RRF_W_GRAPH", "RRF_W_RECENCY", "HNSW_M", "HNSW_EF_SEARCH", "HNSW_EF_CONSTRUCTION",
    "SALIENCE_PRUNE_THRESHOLD",
]


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
    raise SystemExit(f"Unknown system '{name}' (choose from eidetic, mem0, graphiti).")


def _slice(samples: list, subset: int, offset: int) -> list:
    offset = max(0, offset)
    if subset and subset > 0:
        return samples[offset: offset + subset]
    return samples[offset:]


def load_samples(dataset: str, subset: int, variant: str, offset: int = 0):
    out = []
    if dataset in ("longmemeval", "both", "all"):
        out += _slice(longmemeval.load(variant=variant, limit=None), subset, offset)
    if dataset in ("locomo", "both", "all"):
        out += _slice(locomo.load(limit=None), subset, offset)
    if dataset in ("memoryagentbench", "all"):
        out += _slice(memoryagentbench.load(limit=None), subset, offset)
    if dataset in ("beam", "all"):
        out += _slice(beam.load(limit=None), subset, offset)
    return out


def write_manifest(out: Path, args, judge_desc: dict, samples: list | None = None) -> Path:
    sample_rows = (
        [{"dataset": s.dataset, "sample_id": s.sample_id, "category": s.category} for s in samples]
        if samples is not None else None
    )
    manifest = {
        "systems": args.systems,
        "dataset": args.dataset,
        "subset": args.subset,
        "sample_offset": args.sample_offset,
        "runs": args.runs,
        "run_offset": args.run_offset,
        "variant": args.variant,
        "render_only": bool(args.render_only),
        "judge": judge_desc,
        "sample_count": len(samples) if samples is not None else None,
        "category_counts": category_counts(samples) if samples is not None else None,
        "sample_rows": sample_rows,
        "env": {k: os.environ.get(k, "") for k in _MANIFEST_ENV},
    }
    path = out / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Eidetic-Plus neutral benchmark harness")
    ap.add_argument("--systems", default="eidetic", help="comma list: eidetic,mem0,graphiti")
    ap.add_argument("--dataset", default="locomo",
                    choices=["longmemeval", "locomo", "memoryagentbench", "beam", "both", "all"])
    ap.add_argument("--subset", type=int, default=10, help="limit samples per dataset (<=0 = full)")
    ap.add_argument("--sample-offset", type=int, default=0, help="start offset for fresh-slice reruns")
    ap.add_argument("--runs", type=int, default=1, help="independent runs (>=10 for variance)")
    ap.add_argument("--run-offset", type=int, default=0, help="first run index, to append without clobbering")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing run log files")
    ap.add_argument("--variant", default="longmemeval_s")
    ap.add_argument("--out", default="artifacts/bench")
    ap.add_argument("--render-only", action="store_true", help="re-render from existing logs")
    args = ap.parse_args()
    if args.sample_offset < 0:
        raise SystemExit("--sample-offset must be >= 0")
    if args.run_offset < 0:
        raise SystemExit("--run-offset must be >= 0")
    if not args.render_only and args.runs <= 0:
        raise SystemExit("--runs must be > 0 unless --render-only is used")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = Judge()
    judge_desc = judge.describe()
    print(f"Judge: {judge_desc}")
    loaded_samples = None

    if not args.render_only:
        samples = load_samples(args.dataset, args.subset, args.variant, args.sample_offset)
        loaded_samples = samples
        print(f"Loaded {len(samples)} samples ({args.dataset}); categories: {category_counts(samples)}")
        if not samples:
            raise SystemExit("No samples loaded. Check --dataset, --subset, and --sample-offset.")
        failures = []
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
                failures.append((raw.strip(), repr(e)))
                print(f"  !! {raw.strip()} FAILED (continuing): {type(e).__name__}: {str(e)[:200]}")
        if failures:
            print("\nSystems that failed (scoreboard still renders for the rest):")
            for name, err in failures:
                print(f"  - {name}: {err[:160]}")

    manifest = write_manifest(out, args, judge_desc, loaded_samples)
    md = scoreboard.render(out, judge_desc)
    cv = curves.render(out)
    print(f"\nScoreboard -> {md}")
    print(f"Curves     -> {cv}")
    print(f"Manifest   -> {manifest}")
    print("Reproduce (full, >=10 runs): bash bench/reproduce.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
