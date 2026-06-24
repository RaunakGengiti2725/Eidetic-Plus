"""BEAM loader for cached normalized files.

BEAM is a scale benchmark. This loader only reads real cached exports and reports the
ability/category labels present in those files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import Sample, category_counts
from .normalized import normalize_records, read_json_records

_DEFAULT_DIR = Path("data/bench/beam")


def _files(data_dir: Path) -> list[Path]:
    names = ("beam.jsonl", "beam.json", "beam_1m.jsonl", "beam_1m.json",
             "beam_10m.jsonl", "beam_10m.json")
    return [data_dir / name for name in names if (data_dir / name).exists()]


def load(data_dir: Path = _DEFAULT_DIR, limit: Optional[int] = None) -> list[Sample]:
    data_dir = Path(data_dir)
    paths = _files(data_dir)
    if not paths:
        raise FileNotFoundError(
            f"BEAM cached files not found in {data_dir}. Export real BEAM records to "
            "beam.jsonl, beam_1m.jsonl, or beam_10m.jsonl using the normalized harness schema."
        )
    return normalize_records(read_json_records(paths), dataset="beam", limit=limit)


def verify(samples: list[Sample]) -> dict:
    counts = category_counts(samples)
    return {
        "counts": counts,
        "has_contradiction_resolution": "contradiction_resolution" in {
            k.lower().replace("-", "_") for k in counts
        },
    }
