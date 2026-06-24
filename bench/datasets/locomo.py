"""LoCoMo loader (official schema from github.com/snap-research/locomo, locomo10.json).

Local-first: looks for data/bench/locomo/locomo10.json; if absent, attempts a download from
the official raw GitHub URL; else raises with instructions (no mock).

Restricts to the FOUR validated categories (single-hop, multi-hop, temporal, open-domain)
and EXCLUDES adversarial (category 5), which lacks reliable ground truth and is excluded by
both Mem0 and Zep. Category integer mapping per the LoCoMo paper:
  1 -> multi-hop, 2 -> temporal, 3 -> open-domain, 4 -> single-hop, 5 -> adversarial (excluded).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import Sample, Session, Turn

_DEFAULT_DIR = Path("data/bench/locomo")
_RAW_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_MAP = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
VALIDATED = {"single-hop", "multi-hop", "temporal", "open-domain"}


def _parse_time(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B, %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def _local_or_download(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    local = data_dir / "locomo10.json"
    if local.exists():
        return local
    import httpx

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as h:
            r = h.get(_RAW_URL)
            r.raise_for_status()
            local.write_bytes(r.content)
            return local
    except Exception as e:
        raise FileNotFoundError(
            f"LoCoMo not found at {local} and download from {_RAW_URL} failed ({e}). "
            f"Clone github.com/snap-research/locomo and copy data/locomo10.json into {data_dir}/."
        )


def load(data_dir: Path = _DEFAULT_DIR, limit: Optional[int] = None,
         include_adversarial: bool = False) -> list[Sample]:
    path = _local_or_download(Path(data_dir))
    raw = json.loads(path.read_text())
    samples: list[Sample] = []
    for ci, conv in enumerate(raw):
        convo = conv.get("conversation", {})
        sessions: list[Session] = []
        # Sessions are keyed session_1, session_2, ... with session_N_date_time siblings.
        sess_keys = sorted([k for k in convo if k.startswith("session_") and "date_time" not in k],
                           key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0)
        for sk in sess_keys:
            turns = []
            for t in convo[sk]:
                spk = t.get("speaker", t.get("role", "user"))
                txt = t.get("text", t.get("content", ""))
                turns.append(Turn(role=spk, content=txt))
            st = _parse_time(convo.get(f"{sk}_date_time"))
            sessions.append(Session(session_id=f"c{ci}_{sk}", turns=turns, session_time=st))

        for qi, qa in enumerate(conv.get("qa", [])):
            cat = CATEGORY_MAP.get(qa.get("category"), "unknown")
            if cat == "adversarial" and not include_adversarial:
                continue
            if cat not in VALIDATED and not include_adversarial:
                continue
            gold = qa.get("answer", qa.get("adversarial_answer", ""))
            samples.append(Sample(
                sample_id=f"c{ci}_q{qi}", sessions=sessions,
                question=qa.get("question", ""), gold=str(gold),
                category=cat, dataset="locomo",
                meta={"evidence": qa.get("evidence", [])},
            ))
            if limit and len(samples) >= limit:
                return samples
    return samples


def verify(samples: list[Sample]) -> dict:
    from . import category_counts
    counts = category_counts(samples)
    return {"counts": counts, "validated_only": set(counts) <= VALIDATED,
            "adversarial_excluded": "adversarial" not in counts}
