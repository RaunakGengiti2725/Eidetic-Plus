"""Calendar-day round-trip invariant for dataset session times (miss-forensics fleet
2026-07-09, temporal cluster, skeptic-confirmed). The old naive `.timestamp()` parsed the
source wall-clock in the RUN MACHINE's zone while every render emits UTC, so an evening
session ('5:13 pm on 9 July, 2022') shifted a full calendar day between ingest and export
and the reader answered date questions off by one. The invariant: parse -> render must
reproduce the source's calendar day, on any machine, in any TZ."""
from __future__ import annotations

import datetime as dt

from bench.datasets.locomo import _parse_time as locomo_parse
from bench.datasets.longmemeval import _parse_time as lme_parse


def _utc_day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def test_locomo_evening_session_keeps_its_calendar_day():
    # the live c6_q29 shape: an evening PT wall-clock previously rendered as the NEXT day UTC
    ts = locomo_parse("5:13 pm on 9 July, 2022")
    assert ts is not None
    assert _utc_day(ts) == "2022-07-09"


def test_locomo_date_only_and_iso_formats_roundtrip():
    assert _utc_day(locomo_parse("14 August, 2023")) == "2023-08-14"
    assert _utc_day(locomo_parse("2023-08-14 21:30:00")) == "2023-08-14"


def test_longmemeval_formats_roundtrip():
    assert _utc_day(lme_parse("2023/05/30 (Tue) 23:40")) == "2023-05-30"
    assert _utc_day(lme_parse("2023-05-30")) == "2023-05-30"


def test_render_is_machine_timezone_independent():
    """The decisive property: the SAME source string yields the SAME UTC calendar day
    regardless of the machine's local zone (a tz-aware parse cannot consult local time)."""
    import os
    import time
    ts = locomo_parse("11:59 pm on 31 December, 2022")
    old = os.environ.get("TZ")
    try:
        for tz in ("America/Los_Angeles", "Asia/Tokyo", "UTC"):
            os.environ["TZ"] = tz
            time.tzset()
            assert _utc_day(locomo_parse("11:59 pm on 31 December, 2022")) == _utc_day(ts) == "2022-12-31"
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()
