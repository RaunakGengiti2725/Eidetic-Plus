"""Extended event-calendar date patterns from the dominance plan."""
from __future__ import annotations

from datetime import datetime

from eidetic.events import (
    EventRecord,
    effective_date_ranges,
    event_aliases_from_text,
    event_chain,
    normalize_dates,
    parse_query,
    select_for_query,
)


REF = datetime(2026, 6, 23, 12, 0, 0).timestamp()


def _by_expr(text: str, events=None):
    return {d["expr"]: (d["start"], d["end"]) for d in normalize_dates(text, REF, events)}


def test_recently_and_fortnight_windows():
    got = _by_expr("I painted recently and trained over the past fortnight.")
    assert got["recently"] == ("2026-06-16T12:00:00", "2026-06-23T12:00:00")
    assert got["fortnight"] == ("2026-06-09T12:00:00", "2026-06-23T12:00:00")


def test_past_week_bare_unit_window():
    got = _by_expr("Which tea shops did I visit in the past week?")
    assert got["past week"] == ("2026-06-16T12:00:00", "2026-06-23T12:00:00")


def test_recent_query_selects_latest_matching_event_first():
    old = EventRecord(
        subject="Noor", verb="painted", object="horse",
        fact="Noor painted a horse",
        start=datetime(2026, 6, 18).timestamp(),
        end=datetime(2026, 6, 18).timestamp(),
    )
    new = EventRecord(
        subject="Noor", verb="painted", object="sunset",
        fact="Noor painted a sunset",
        start=datetime(2026, 6, 22).timestamp(),
        end=datetime(2026, 6, 22).timestamp(),
    )
    parsed = parse_query("What did Noor paint recently?", REF, [old, new])
    assert parsed["temporal_order"] == "desc"
    assert [e.object for e in select_for_query([old, new], parsed, REF)] == ["sunset", "horse"]


def test_event_aliases_preserve_specific_source_object_phrase():
    aliases = event_aliases_from_text(
        "Alice attended the annual robotics conference on May 4, 2026.",
        {
            "src": "Alice",
            "relation": "attended",
            "dst": "conference",
            "fact": "Alice attended conference",
        },
    )
    assert "annual robotics conference" in aliases
    assert "attended the annual robotics conference" in aliases


def test_last_week_query_keeps_chronological_event_order():
    old = EventRecord(
        subject="Noor", verb="painted", object="horse",
        fact="Noor painted a horse",
        start=datetime(2026, 6, 17).timestamp(),
        end=datetime(2026, 6, 17).timestamp(),
    )
    new = EventRecord(
        subject="Noor", verb="painted", object="sunset",
        fact="Noor painted a sunset",
        start=datetime(2026, 6, 19).timestamp(),
        end=datetime(2026, 6, 19).timestamp(),
    )
    parsed = parse_query("What did Noor paint last week?", REF, [old, new])
    assert parsed["temporal_order"] is None
    assert [e.object for e in select_for_query([new, old], parsed, REF)] == ["horse", "sunset"]


def test_bare_last_query_selects_latest_matching_event_first():
    old = EventRecord(
        subject="Noor", verb="painted", object="horse",
        fact="Noor painted a horse",
        start=datetime(2026, 6, 17).timestamp(),
        end=datetime(2026, 6, 17).timestamp(),
    )
    new = EventRecord(
        subject="Noor", verb="painted", object="sunset",
        fact="Noor painted a sunset",
        start=datetime(2026, 6, 19).timestamp(),
        end=datetime(2026, 6, 19).timestamp(),
    )
    parsed = parse_query("What did Noor paint last?", REF, [old, new])
    assert parsed["temporal_order"] == "desc"
    assert [e.object for e in select_for_query([old, new], parsed, REF)] == ["sunset", "horse"]


def test_past_n_unit_window():
    got = _by_expr("during the past 3 weeks")
    assert got["past 3 weeks"] == ("2026-06-02T12:00:00", "2026-06-23T12:00:00")


def test_past_few_months_window():
    got = _by_expr("in the past few months")
    assert got["past few months"] == ("2026-03-25T12:00:00", "2026-06-23T12:00:00")


def test_early_mid_late_month_ranges():
    got = _by_expr("early May 2023, mid June 2026, late June 2026")
    assert got["early May 2023"] == ("2023-05-01T00:00:00", "2023-05-10T23:59:59")
    assert got["mid June 2026"] == ("2026-06-11T00:00:00", "2026-06-20T23:59:59")
    assert got["late June 2026"] == ("2026-06-21T00:00:00", "2026-06-30T23:59:59")


def test_month_day_year_and_day_month_year_ranges():
    got = _by_expr("May 4, 2026 and 5 June 2026")
    assert got["May 4, 2026"] == ("2026-05-04T00:00:00", "2026-05-04T23:59:59")
    assert got["5 June 2026"] == ("2026-06-05T00:00:00", "2026-06-05T23:59:59")


def test_event_relative_week_after_anchor():
    anchor = EventRecord(
        subject="Alice", verb="attended", object="the conference",
        fact="Alice attended the conference",
        aliases=["work summit"],
        start=datetime(2026, 5, 4).timestamp(),
        end=datetime(2026, 5, 4).timestamp(),
    )
    got = _by_expr("the week after the conference", [anchor])
    assert got["the week after the conference"] == (
        "2026-05-11T00:00:00", "2026-05-17T23:59:59"
    )
    assert normalize_dates("the week after the conference", REF, [anchor])[0]["anchored"] is True


def test_event_relative_counted_and_following_day_anchor():
    anchor = EventRecord(
        subject="Alice", verb="attended", object="the conference",
        fact="Alice attended the conference",
        aliases=["work summit"],
        start=datetime(2026, 5, 4).timestamp(),
        end=datetime(2026, 5, 4).timestamp(),
    )
    counted = _by_expr("two days after the conference", [anchor])
    assert counted["two days after the conference"] == (
        "2026-05-06T00:00:00", "2026-05-06T23:59:59"
    )
    following = _by_expr("the following day after the conference", [anchor])
    assert following["the following day after the conference"] == (
        "2026-05-05T00:00:00", "2026-05-05T23:59:59"
    )
    previous = _by_expr("the previous week before the conference", [anchor])
    assert previous["the previous week before the conference"] == (
        "2026-04-27T00:00:00", "2026-05-03T23:59:59"
    )


def test_absolute_anchor_counted_relative_dates():
    got = _by_expr("Noor signed up two days before July 4, 2023.")
    assert got["two days before july 4, 2023"] == (
        "2023-07-02T00:00:00", "2023-07-02T23:59:59"
    )
    after = _by_expr("Noor went back two days after July 4, 2023.")
    assert after["two days after july 4, 2023"] == (
        "2023-07-06T00:00:00", "2023-07-06T23:59:59"
    )


def test_absolute_anchor_previous_week_range():
    got = _by_expr("the week before 9 June 2023")
    assert got["the week before 9 june 2023"] == (
        "2023-05-29T00:00:00", "2023-06-04T23:59:59"
    )
    following = _by_expr("the following day after 9 June 2023")
    assert following["the following day after 9 june 2023"] == (
        "2023-06-10T00:00:00", "2023-06-10T23:59:59"
    )


def test_between_date_interval_inherits_endpoint_year():
    got = normalize_dates("Where was Priya between March 9 and March 13 2024?", REF)
    interval = next(r for r in got if r.get("interval"))
    assert interval["expr"] == "between march 9 and march 13 2024"
    assert interval["start"] == "2024-03-09T00:00:00"
    assert interval["end"] == "2024-03-13T23:59:59"


def test_effective_date_ranges_prefer_intended_relative_or_interval_range():
    week = normalize_dates("What did Marco do the week before October 6, 2024?", REF)
    assert [r["expr"] for r in effective_date_ranges(week)] == ["the week before october 6, 2024"]

    interval = normalize_dates("Where was Priya between March 9 and March 13 2024?", REF)
    assert [r["expr"] for r in effective_date_ranges(interval)] == [
        "between march 9 and march 13 2024"
    ]


def test_select_for_query_uses_interval_not_broad_year_range():
    before = EventRecord(
        subject="Priya", verb="visited", object="Lisbon",
        fact="Priya visited Lisbon",
        start=datetime(2024, 3, 8).timestamp(),
        end=datetime(2024, 3, 8).timestamp(),
    )
    inside = EventRecord(
        subject="Priya", verb="visited", object="Osaka",
        fact="Priya visited Osaka",
        start=datetime(2024, 3, 11).timestamp(),
        end=datetime(2024, 3, 11).timestamp(),
    )
    after = EventRecord(
        subject="Priya", verb="visited", object="Nairobi",
        fact="Priya visited Nairobi",
        start=datetime(2024, 4, 2).timestamp(),
        end=datetime(2024, 4, 2).timestamp(),
    )
    parsed = parse_query("Where was Priya between March 9 and March 13 2024?", REF)
    assert [e.object for e in select_for_query([before, inside, after], parsed, REF)] == ["Osaka"]


def test_event_relative_anchor_ignores_stopword_only_matches():
    conference = EventRecord(
        subject="Alice", verb="attended", object="the conference",
        fact="Alice attended the conference",
        start=datetime(2026, 5, 4).timestamp(),
        end=datetime(2026, 5, 4).timestamp(),
    )
    got = normalize_dates("the week after the trip", REF, [conference])
    assert [r for r in got if r.get("anchored")] == []


def test_event_relative_anchor_matches_alias_tokens():
    conference = EventRecord(
        subject="Alice", verb="attended", object="the conference",
        fact="Alice attended the conference",
        aliases=["work summit"],
        start=datetime(2026, 5, 4).timestamp(),
        end=datetime(2026, 5, 4).timestamp(),
    )
    got = _by_expr("the day after the work summit", [conference])
    assert got["the day after the work summit"] == (
        "2026-05-05T00:00:00", "2026-05-05T23:59:59"
    )


def test_parse_query_resolves_event_relative_range_with_anchor_events():
    anchor = EventRecord(
        subject="Alice", verb="attended", object="the conference",
        fact="Alice attended the conference",
        aliases=["work summit"],
        start=datetime(2026, 5, 4).timestamp(),
        end=datetime(2026, 5, 4).timestamp(),
    )
    followup = EventRecord(
        subject="Alice", verb="visited", object="Paris",
        fact="Alice visited Paris",
        start=datetime(2026, 5, 12).timestamp(),
        end=datetime(2026, 5, 12).timestamp(),
    )
    parsed = parse_query("What did Alice do the week after the conference?", REF, [anchor, followup])
    assert parsed["ranges"][0]["expr"] == "the week after the conference"
    assert [e.object for e in select_for_query([anchor, followup], parsed, REF)] == ["Paris"]
    assert [e.object for e in event_chain([anchor, followup], parsed, REF)] == ["Paris"]


def test_parse_query_selects_counted_event_relative_range():
    anchor = EventRecord(
        subject="Alice", verb="attended", object="the conference",
        fact="Alice attended the conference",
        start=datetime(2026, 5, 4).timestamp(),
        end=datetime(2026, 5, 4).timestamp(),
    )
    followup = EventRecord(
        subject="Alice", verb="visited", object="Paris",
        fact="Alice visited Paris",
        start=datetime(2026, 5, 6).timestamp(),
        end=datetime(2026, 5, 6).timestamp(),
    )
    parsed = parse_query("What did Alice do two days after the conference?", REF, [anchor, followup])
    assert parsed["ranges"][0]["expr"] == "two days after the conference"
    assert [e.object for e in select_for_query([anchor, followup], parsed, REF)] == ["Paris"]
