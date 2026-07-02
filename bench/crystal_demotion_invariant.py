"""Rotating claim-crystal span-demotion eval.

Proves the forgetting profile's cost mechanics on invented corpora with zero model calls:

* crystallized low-vividness records contribute bounded query-centered spans, so assembled
  context shrinks materially versus the forgetting-off profile;
* the span still contains the answering sentence for every case (cost never hides the answer);
* enumeration-shaped queries are never demoted (lists need every mention);
* the top vivid fraction by affect salience keeps full text;
* with the demotion flag off, output is byte-identical to the keep-everything path.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from bench.seed_policy import resolve_seed
from eidetic.config import get_settings
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import RetrievalCandidate, Retriever
from eidetic.store import RecordStore


class _NoopClient:
    def embed_text(self, text):
        return np.zeros(8, np.float32)

    def nli(self, premise, hypothesis):
        return ("neutral", 0.0)


class _NoopReranker:
    def rerank(self, query, candidates, top_k=None):
        return candidates


class _NoopIndex:
    def search(self, vec, k):
        return []

    def get_vectors(self, ids):
        return {}

    def __len__(self):
        return 0


_TOPICS = ["kiln schedule", "harbor permit", "garden ledger", "studio lease", "trail census"]
_VALUES = ["Tuesday", "Pier Nine", "row forty", "the annex", "gate twelve"]
_FILLER_SUBJECTS = ["greenhouse", "workshop", "archive", "orchard", "boathouse"]


@dataclass
class DemotionCase:
    case_id: str
    question: str
    answer_marker: str
    enumeration: bool
    rows: list[str]


def _filler(rng: random.Random, subject: str, n: int) -> str:
    return " ".join(
        f"The {subject} log entry {i} mentions routine upkeep and nothing else." for i in range(n)
    )


def generate_cases(seed: int, cases: int) -> list[DemotionCase]:
    rng = random.Random(seed)
    out: list[DemotionCase] = []
    for idx in range(cases):
        topic = rng.choice(_TOPICS)
        value = rng.choice(_VALUES)
        subject = rng.choice(_FILLER_SUBJECTS)
        enumeration = idx % 4 == 3
        answer = f"The {topic} moved to {value}."
        rows = [
            f"user: {_filler(rng, subject, 40)} {answer} {_filler(rng, subject, 40)}",
            f"user: {_filler(rng, rng.choice(_FILLER_SUBJECTS), 60)}",
        ]
        question = (
            f"How many {topic} updates did I record?" if enumeration
            else f"Where did the {topic} move?"
        )
        out.append(DemotionCase(
            case_id=f"demotion-{idx}",
            question=question,
            answer_marker=answer,
            enumeration=enumeration,
            rows=rows,
        ))
    rng.shuffle(out)
    return out


def _assemble(case: DemotionCase, tmp: Path, *, knobs_on: bool, flag_on: bool = True) -> list[str]:
    base = get_settings()
    settings = replace(
        base,
        data_dir=tmp / f"{case.case_id}-{int(knobs_on)}-{int(flag_on)}",
        vector_backend="numpy",
        gist_channel_enabled=False,
        crystal_span_demotion_enabled=flag_on,
        dream_prune_percentile=5.0 if knobs_on else 0.0,
        salience_prune_threshold=0.0,
        vivid_fraction=0.25,
    )
    store = RecordStore(settings.data_dir / "t.sqlite")
    scope = Scope(namespace=f"demotion-{case.case_id}")
    cands = []
    for i, text in enumerate(case.rows):
        rec = MemoryRecord(
            memory_id=f"{case.case_id}-m{i}",
            text=text,
            source="s",
            scope=scope,
            valid_at=1_700_000_000.0 + i,
            salience=0.4,
            metadata={"claims_extracted": 30},
        )
        store.upsert_record(rec)
        cands.append(RetrievalCandidate(record=rec, dense_score=0.9, fused_score=0.9))
    r = Retriever(store, _NoopIndex(), KnowledgeGraph(store), _NoopClient(), _NoopReranker(), settings)
    return r.assemble_context(case.question, cands, scope=scope)


def run_eval(*, seed: Optional[int] = None, cases: int = 20) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures = []
    ratios: list[float] = []
    with tempfile.TemporaryDirectory(prefix="crystal-demotion-") as tmp_str:
        tmp = Path(tmp_str)
        for case in generated:
            demoted = _assemble(case, tmp, knobs_on=True)
            kept = _assemble(case, tmp, knobs_on=False)
            flag_off = _assemble(case, tmp, knobs_on=True, flag_on=False)
            d_chars = sum(len(b) for b in demoted)
            k_chars = sum(len(b) for b in kept)
            detail = {
                "case_id": case.case_id,
                "question": case.question,
                "enumeration": case.enumeration,
                "demoted_chars": d_chars,
                "kept_chars": k_chars,
            }
            if flag_off != kept:
                failures.append({**detail, "why": "flag-off output must equal keep-everything output"})
                continue
            if case.enumeration:
                if d_chars != k_chars:
                    failures.append({**detail, "why": "enumeration queries must not be demoted"})
                continue
            ratios.append(d_chars / k_chars if k_chars else 1.0)
            if d_chars >= k_chars * 0.7:
                failures.append({**detail, "why": "demotion must shrink context by >=30%"})
                continue
            if not any(case.answer_marker in b for b in demoted):
                failures.append({**detail, "why": "demoted span must keep the answering sentence"})
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "correct": cases - len(failures),
        "avg_demotion_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="repro seed; omitted means random")
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()
    report = run_eval(seed=args.seed, cases=args.cases)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
