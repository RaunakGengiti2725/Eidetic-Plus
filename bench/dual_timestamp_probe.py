"""Dual-timestamp probe (offline, no API): quantifies the fleet's contested 53/60 finding
that CLAIMS inherit session time while the text states a different EVENT date. For every
claim whose proof_atom contains an explicit resolvable date, resolve it with the engine's
own normalizer against the record's session anchor and compare to claim.valid_at. Also
measures the EVENT channel (EventRecord.start) on the same records -- the skeptic split was
exactly "claims are session-stamped BUT events sometimes backdate"; this reports both rates
so the design decision rests on numbers, not anecdotes.

  DATA_DIR=<window>/data .venv/bin/python -m bench.dual_timestamp_probe <window> [--out X]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path


def _day(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime("%Y-%m-%d")


# Explicit, self-contained date phrases only -- relative phrases ("last week") are a
# different (harder) class and excluded so the measurement is exact.
_EXPLICIT_DATE_RE = re.compile(
    r"\b(?:on\s+)?(?P<month>january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(?P<dom>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>20\d{2}))?"
    r"|\b(?P<iso>20\d{2}-\d{2}-\d{2})\b", re.I)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}


def _resolve(m: re.Match, anchor_ts: float) -> str | None:
    if m.group("iso"):
        return m.group("iso")
    anchor = dt.datetime.fromtimestamp(anchor_ts, dt.timezone.utc)
    year = int(m.group("year")) if m.group("year") else anchor.year
    try:
        return dt.date(year, _MONTHS[m.group("month").lower()], int(m.group("dom"))).isoformat()
    except ValueError:
        return None


def run(window: Path) -> dict:
    from eidetic.engine import Engine
    from eidetic.models import Scope

    eng = Engine()
    con = sqlite3.connect(str(Path(eng.settings.data_dir) / "eidetic.sqlite"))
    namespaces = [r[0] for r in con.execute(
        "SELECT DISTINCT namespace FROM memories WHERE namespace LIKE 'eidetic-plus-full%'")]
    con.close()

    claim_dated = claim_mismatch = 0
    event_dated = event_match = 0
    examples: list[dict] = []
    for ns in namespaces:
        scope = Scope(namespace=ns)
        try:
            claims = eng.store.active_claims_at(None, scope)
        except Exception:
            claims = []
        recs = {r.memory_id: r for r in eng.store.all_records(scope)}
        events = list(eng.store.events_in_scope(ns, scope=scope))
        ev_by_rec: dict[str, list] = {}
        for ev in events:
            ev_by_rec.setdefault(ev.source_memory_id or "", []).append(ev)
        for c in claims:
            atom = c.proof_atom or ""
            m = _EXPLICIT_DATE_RE.search(atom)
            if not m:
                continue
            rec = recs.get(c.source_memory_id or "")
            anchor = rec.valid_at if rec is not None else c.valid_at
            stated = _resolve(m, anchor or 0.0)
            if not stated:
                continue
            claim_dated += 1
            if _day(c.valid_at) != stated:
                claim_mismatch += 1
                if len(examples) < 12:
                    examples.append({"namespace": ns[-8:], "channel": "claim",
                                     "stated": stated, "stamped": _day(c.valid_at),
                                     "atom": atom[:110]})
            # EVENT channel on the same record: does ANY event carry the stated day?
            evs = ev_by_rec.get(c.source_memory_id or "", [])
            if evs:
                event_dated += 1
                if any(_day(getattr(ev, "start", None)) == stated for ev in evs):
                    event_match += 1
    return {
        "window": str(window),
        "claims_with_explicit_date": claim_dated,
        "claims_stamped_differently": claim_mismatch,
        "claim_mismatch_rate": round(claim_mismatch / claim_dated, 3) if claim_dated else None,
        "records_with_events_and_dated_claims": event_dated,
        "events_carrying_stated_day": event_match,
        "event_backdate_rate": round(event_match / event_dated, 3) if event_dated else None,
        "examples": examples,
        "method": ("explicit in-atom date phrases only (month-day[-year] or ISO), resolved "
                   "against the source record's session anchor; compared by UTC calendar day "
                   "to claim.valid_at and to EventRecord.start on the same record. Relative "
                   "phrases excluded -- this measures the EXACT class the fleet flagged and "
                   "the skeptics split on."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("window")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = run(Path(args.window))
    text = json.dumps(rep, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
