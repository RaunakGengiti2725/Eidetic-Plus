"""HaluMem operation-level grading (PDF: measure self-repair on HaluMem FIRST).

HaluMem (arXiv 2511.03506) decomposes a memory system into extraction, updating, and QA and
grades each separately -- exposing that current systems hallucinate badly (no system >55% QA,
extraction recall ~43%, update accuracy <26%). The PDF makes this the Stage-1 measurement target
because that is where the headroom is.

The GRADING here is deterministic set-overlap / aggregation over LABELED memory points and is
fully offline-unit-testable. Producing the predicted points + QA judgments is LLM-gated (a real
benchmark run, no mock). The op-level eval routes every sample through split_of so it respects
the same dev/test integrity wall as the rest of the harness.
"""
from __future__ import annotations

from pathlib import Path

from .datasets import split_of

_DEFAULT_DIR = Path("data/bench/halumem")


def normalize_point(s: str) -> str:
    return " ".join((s or "").lower().split())


def _pset(points) -> set:
    return {normalize_point(p) for p in points if normalize_point(p)}


# ---- extraction ------------------------------------------------------------
def extraction_recall(gold_points, predicted_points) -> float:
    gold, pred = _pset(gold_points), _pset(predicted_points)
    return len(gold & pred) / len(gold) if gold else 0.0


def extraction_precision(gold_points, predicted_points) -> float:
    gold, pred = _pset(gold_points), _pset(predicted_points)
    return len(gold & pred) / len(pred) if pred else 0.0


def extraction_f1(gold_points, predicted_points) -> float:
    r = extraction_recall(gold_points, predicted_points)
    p = extraction_precision(gold_points, predicted_points)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ---- updating --------------------------------------------------------------
def update_accuracy(update_rows: list[dict]) -> float:
    """update_rows = [{'applied_correct': bool}, ...] -- fraction of gold updates applied right."""
    if not update_rows:
        return 0.0
    return sum(1 for r in update_rows if r.get("applied_correct")) / len(update_rows)


# ---- QA (hallucination / omission) -----------------------------------------
def qa_rates(qa_rows: list[dict]) -> dict:
    """qa_rows = [{'answerable': bool, 'answered': bool, 'correct': bool}, ...] (correct from the
    fixed judge). Returns QA accuracy + the two HaluMem failure rates:
      * hallucination_rate = answered-but-wrong / answered (a confident wrong answer)
      * omission_rate = answerable-but-not-answered / answerable (a missed answerable question)."""
    n = len(qa_rows)
    if not n:
        return {"qa_accuracy": 0.0, "hallucination_rate": 0.0, "omission_rate": 0.0, "n": 0}
    answered = [r for r in qa_rows if r.get("answered")]
    answerable = [r for r in qa_rows if r.get("answerable")]
    hallucinated = sum(1 for r in answered if not r.get("correct"))
    omitted = sum(1 for r in answerable if not r.get("answered"))
    correct = sum(1 for r in qa_rows if r.get("correct"))
    return {
        "qa_accuracy": correct / n,
        "hallucination_rate": hallucinated / len(answered) if answered else 0.0,
        "omission_rate": omitted / len(answerable) if answerable else 0.0,
        "n": n,
    }


def filter_points_by_split(points: list[dict], split: str, id_key: str = "sample_id") -> list[dict]:
    """Route op-level memory points through the integrity wall by sample_id (dev/test/all)."""
    if split in (None, "all"):
        return points
    return [p for p in points if split_of(str(p.get(id_key, ""))) == split]


def load(data_dir: Path = _DEFAULT_DIR) -> list[dict]:
    """Local-first, FAIL-LOUD loader (no mock): real HaluMem op-level exports must be placed in
    data/bench/halumem/ as halumem.jsonl. Raises if absent rather than fabricating examples."""
    data_dir = Path(data_dir)
    path = data_dir / "halumem.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No HaluMem export at {path}. Place the real operation-level dataset there "
            "(labeled memory points + updates + QA). Eidetic-Plus never fabricates benchmark data."
        )
    import json
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows
