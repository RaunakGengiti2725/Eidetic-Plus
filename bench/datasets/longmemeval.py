"""LongMemEval loader (official schema from github.com/xiaowu0162/LongMemEval;
data from HF xiaowu0162/longmemeval-cleaned).

Local-first: looks for data/bench/longmemeval/<file>.json; if absent, attempts a download
from the HF resolve URL; if that fails, raises with instructions (no mock, no fabricated
data). Maps question_type -> category and haystack_sessions -> our Session/Turn schema.

Expected category counts for longmemeval_s (spec Section 11), used by verify():
  single-session-user 70, single-session-assistant 56, single-session-preference 30,
  multi-session 133, knowledge-update 78, temporal-reasoning 133 (+ abstention variants).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import Sample, Session, Turn

_DEFAULT_DIR = Path("data/bench/longmemeval")
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

EXPECTED_COUNTS = {
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "multi-session": 133,
    "knowledge-update": 78,
    "temporal-reasoning": 133,
}


def _parse_time(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            # UTC-AWARE on purpose -- see bench/datasets/locomo.py:_parse_time. A naive
            # .timestamp() parses the source wall-clock in the run machine's local zone
            # while renders emit UTC, shifting evening sessions a full calendar day.
            return datetime.strptime(s.strip(), fmt).replace(
                tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def _local_or_download(variant: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{variant}.json"
    local = data_dir / fname
    if local.exists():
        return local
    # Attempt a real download (network only, no key). Fail loud with instructions.
    import httpx

    url = f"{_HF_BASE}/{fname}"
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as h:
            r = h.get(url)
            r.raise_for_status()
            local.write_bytes(r.content)
            return local
    except Exception as e:
        raise FileNotFoundError(
            f"LongMemEval '{variant}' not found at {local} and download from {url} failed "
            f"({e}). Download it manually from HF dataset 'xiaowu0162/longmemeval-cleaned' "
            f"(or the official repo) into {data_dir}/."
        )


def load(variant: str = "longmemeval_s", data_dir: Path = _DEFAULT_DIR,
         limit: Optional[int] = None) -> list[Sample]:
    path = _local_or_download(variant, Path(data_dir))
    raw = json.loads(path.read_text())
    samples: list[Sample] = []
    for item in raw:
        qid = str(item.get("question_id") or item.get("id") or len(samples))
        category = str(item.get("question_type") or item.get("category") or "unknown")
        sessions: list[Session] = []
        haystack = item.get("haystack_sessions") or item.get("sessions") or []
        session_ids = item.get("haystack_session_ids") or []
        dates = item.get("haystack_dates") or []
        for i, sess in enumerate(haystack):
            turns = [Turn(role=t.get("role", "user"), content=t.get("content", t.get("text", "")))
                     for t in sess]
            st = _parse_time(dates[i]) if i < len(dates) else None
            sid = str(session_ids[i]) if i < len(session_ids) and session_ids[i] else f"{qid}_s{i}"
            sessions.append(Session(session_id=sid, turns=turns, session_time=st))
        samples.append(Sample(
            sample_id=qid, sessions=sessions,
            question=item.get("question", ""), gold=str(item.get("answer", "")),
            category=category, dataset="longmemeval",
            question_time=_parse_time(item.get("question_date")),
            meta={"answer_session_ids": item.get("answer_session_ids", [])},
        ))
        if limit and len(samples) >= limit:
            break
    return samples


def verify(samples: list[Sample]) -> dict:
    from . import category_counts
    counts = category_counts(samples)
    matches = {k: counts.get(k, 0) for k in EXPECTED_COUNTS}
    ok = all(counts.get(k, 0) == v for k, v in EXPECTED_COUNTS.items())
    return {"counts": counts, "expected": EXPECTED_COUNTS, "matches_full_set": ok}
