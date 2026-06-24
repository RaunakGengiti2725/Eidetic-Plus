"""MemoryAgentBench loader for cached normalized files.

This loader never fabricates examples. Put real FactConsolidation/EventQA exports in
data/bench/memoryagentbench as JSONL or JSON using the normalized harness schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import Sample, category_counts
from .normalized import normalize_records, read_json_records

_DEFAULT_DIR = Path("data/bench/memoryagentbench")
_TASKS = ("factconsolidation", "eventqa")
_EXPECTED = {"factconsolidation", "eventqa", "fact-consolidation", "event-qa",
             "event_qa", "factcon-sh", "factcon-mh"}


def _files(data_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for name in ("memoryagentbench.jsonl", "memoryagentbench.json"):
        p = data_dir / name
        if p.exists():
            paths.append(p)
    for task in _TASKS:
        for suffix in (".jsonl", ".json"):
            p = data_dir / f"{task}{suffix}"
            if p.exists():
                paths.append(p)
    return paths


def load(data_dir: Path = _DEFAULT_DIR, limit: Optional[int] = None) -> list[Sample]:
    data_dir = Path(data_dir)
    paths = _files(data_dir)
    if not paths:
        raise FileNotFoundError(
            f"MemoryAgentBench cached files not found in {data_dir}. Export real "
            "FactConsolidation/EventQA records to memoryagentbench.jsonl, "
            "factconsolidation.jsonl, or eventqa.jsonl using the normalized harness schema."
        )
    samples = normalize_records(read_json_records(paths), dataset="memoryagentbench", limit=limit)
    return [s for s in samples if s.category.lower() in _EXPECTED or s.category != "unknown"]


def verify(samples: list[Sample]) -> dict:
    counts = category_counts(samples)
    return {"counts": counts, "has_target_task": bool(set(k.lower() for k in counts) & _EXPECTED)}
