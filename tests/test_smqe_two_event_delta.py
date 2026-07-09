"""Two-event day-gap delta ('how many days between doing X and doing Y') -- the shape the
single-event 'days ago did I X' path returned 0 for. Pure-function test on synthetic atoms
with explicit dates. Fires ONLY on 'between X and Y'; leaves other temporal questions alone."""
from datetime import datetime, timezone
from eidetic.smqe.record_ops import _two_event_delta_answer


class _Item:
    def __init__(self, valid_at):
        self.valid_at = valid_at


def _ts(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def _atoms(*rows):
    # (score, item(valid_at), atom_text)
    return [(1.0 - i * 0.05, _Item(_ts(d)), t) for i, (d, t) in enumerate(rows)]


def test_two_event_gap_uses_each_events_date():
    atoms = _atoms(
        ("2023-02-15", "I replaced my spark plugs to optimize performance."),
        ("2023-03-16", "I joined the auto racing tuesdays event."),
        ("2023-02-20", "unrelated chatter about books"))
    ans, sel = _two_event_delta_answer(
        "How many days between when I replaced my spark plugs and when I joined the racing event?",
        atoms)
    assert ans == "29 days"           # Feb15 -> Mar16 = 29 days
    assert len(sel) == 2


def test_reversed_order_gives_absolute_gap():
    atoms = _atoms(("2023-03-16", "joined the racing event"),
                   ("2023-02-15", "replaced the spark plugs"))
    ans, _ = _two_event_delta_answer(
        "how many days between replacing the spark plugs and joining the racing event?", atoms)
    assert ans == "29 days"           # absolute value, order-independent


def test_does_not_fire_on_single_event_question():
    atoms = _atoms(("2023-02-15", "I replaced my spark plugs"))
    assert _two_event_delta_answer("how many days ago did I replace my spark plugs?", atoms) == ("", [])


def test_returns_empty_when_an_event_is_unmatched():
    atoms = _atoms(("2023-02-15", "I replaced my spark plugs"))
    # second event ('moon landing') matches nothing -> no confabulated gap
    ans, _ = _two_event_delta_answer(
        "how many days between replacing the spark plugs and the moon landing?", atoms)
    assert ans == ""


def test_same_date_events_return_empty_not_zero():
    atoms = _atoms(("2023-02-15", "replaced the spark plugs and joined the racing event same day"))
    # both phrases match the one atom -> same date -> empty (not a misleading "0 days")
    ans, _ = _two_event_delta_answer(
        "how many days between replacing the spark plugs and joining the racing event?", atoms)
    assert ans == ""
