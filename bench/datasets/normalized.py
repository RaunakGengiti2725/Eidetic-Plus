"""Helpers for cached benchmark files normalized to the harness schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from . import Sample, Session, Turn


def read_json_records(paths: Iterable[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
        else:
            raw = json.loads(path.read_text())
            if isinstance(raw, list):
                records.extend(x for x in raw if isinstance(x, dict))
            elif isinstance(raw, dict):
                data = raw.get("data") or raw.get("samples") or raw.get("questions") or []
                records.extend(x for x in data if isinstance(x, dict))
    return records


def normalize_records(records: list[dict], *, dataset: str,
                      limit: Optional[int] = None) -> list[Sample]:
    samples: list[Sample] = []
    for i, item in enumerate(records):
        sessions: list[Session] = []
        raw_sessions = item.get("sessions") or item.get("haystack_sessions") or []
        if raw_sessions:
            for si, sess in enumerate(raw_sessions):
                if isinstance(sess, dict):
                    turns_raw = sess.get("turns") or sess.get("messages") or sess.get("conversation") or []
                    sid = str(sess.get("session_id") or sess.get("id") or f"{i}_s{si}")
                    stime = sess.get("session_time")
                else:
                    turns_raw = sess
                    sid = f"{i}_s{si}"
                    stime = None
                turns = [
                    Turn(
                        role=str(t.get("role") or t.get("speaker") or "user"),
                        content=str(t.get("content") or t.get("text") or ""),
                        timestamp=t.get("timestamp"),
                    )
                    for t in turns_raw if isinstance(t, dict)
                ]
                sessions.append(Session(session_id=sid, turns=turns, session_time=stime))
        else:
            turns_raw = item.get("turns") or item.get("messages") or item.get("conversation") or []
            turns = [
                Turn(
                    role=str(t.get("role") or t.get("speaker") or "user"),
                    content=str(t.get("content") or t.get("text") or ""),
                    timestamp=t.get("timestamp"),
                )
                for t in turns_raw if isinstance(t, dict)
            ]
            sessions.append(Session(session_id=str(item.get("session_id") or f"{i}_s0"), turns=turns))

        sample_id = str(item.get("sample_id") or item.get("question_id") or item.get("id") or i)
        category = str(item.get("category") or item.get("task") or item.get("ability") or "unknown")
        question = str(item.get("question") or item.get("query") or "")
        raw_gold = item.get("gold")
        if raw_gold is None:
            raw_gold = item.get("answer")
        if raw_gold is None:
            raw_gold = item.get("target")
        aliases = item.get("gold_aliases") or item.get("aliases") or item.get("answers") or []
        if isinstance(raw_gold, list):
            aliases = raw_gold if not aliases else aliases
            gold = str(raw_gold[0]) if raw_gold else ""
        else:
            gold = str(raw_gold or "")
        meta = {k: v for k, v in item.items() if k not in {"sessions", "turns", "messages", "conversation"}}
        if aliases:
            meta["gold_aliases"] = aliases
        samples.append(Sample(
            sample_id=sample_id, sessions=sessions, question=question, gold=gold,
            category=category, dataset=dataset,
            question_time=item.get("question_time"),
            meta=meta,
        ))
        if limit and len(samples) >= limit:
            break
    return samples
