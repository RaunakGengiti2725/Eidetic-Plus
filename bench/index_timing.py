"""Offline timing harness for the lexical retrieval channel.

This does not call any model API. It compares the legacy per-query BM25 build with
the persistent tokenized BM25 index on a synthetic corpus.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from eidetic.bm25 import BM25, PersistentBM25


def _synthetic_items(n: int) -> list[tuple[str, str]]:
    topics = ["travel", "health", "finance", "family", "project", "music", "food"]
    items: list[tuple[str, str]] = []
    for i in range(n):
        topic = topics[i % len(topics)]
        text = (
            f"memory {i} topic {topic} user fact value-{i % 97} "
            f"session-{i // 10} code-{i % 31} repeated {topic}"
        )
        items.append((f"mem_{i}", text))
    return items


def _queries(n: int) -> list[str]:
    topics = ["travel", "health", "finance", "family", "project", "music", "food"]
    return [f"{topics[i % len(topics)]} value-{i % 97} code-{i % 31}" for i in range(n)]


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.quantiles(xs, n=20)[18]) if len(xs) >= 20 else max(xs)


def run_timing(doc_count: int = 1000, query_count: int = 50,
               index_path: Path | None = None, topk: int = 20) -> dict:
    items = _synthetic_items(doc_count)
    queries = _queries(query_count)

    legacy_ms: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        BM25().index(items).search(q, topk)
        legacy_ms.append((time.perf_counter() - t0) * 1000.0)

    with tempfile.TemporaryDirectory() as td:
        path = index_path or (Path(td) / "bm25_index.json")
        idx = PersistentBM25(path)
        idx.index(items)
        idx.save()
        idx = PersistentBM25(path)
        persistent_ms: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            idx.search(q, topk)
            persistent_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "doc_count": doc_count,
        "query_count": query_count,
        "legacy_p50_ms": float(statistics.median(legacy_ms)),
        "legacy_p95_ms": _p95(legacy_ms),
        "persistent_p50_ms": float(statistics.median(persistent_ms)),
        "persistent_p95_ms": _p95(persistent_ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    print(json.dumps(run_timing(args.docs, args.queries, topk=args.topk), indent=2))


if __name__ == "__main__":
    main()
