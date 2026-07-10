"""Decimal integrity for heuristic quantity claims (extraction-audit fleet 2026-07-09).
The claim OBJECT capture stopped at any '.', so '$11.99' stored as 'on sale for $11' and
'7.5 miles' as 'covering 7' -- corrupted-but-plausible values that string-match-verified as
wrong answers (13/107 quantity claims mid-decimal in one live namespace). A '.' must end the
capture only when it is NOT a decimal point."""
from __future__ import annotations

from eidetic.models import MemoryRecord, Scope
from eidetic.smqe.claim_extraction import claims_for_record

_SCOPE = Scope(namespace="qty-dec")


def _rec(text: str, ch: str) -> MemoryRecord:
    return MemoryRecord(text=text, source="user", scope=_SCOPE, valid_at=1.0,
                        content_hash=ch, raw_uri="mem://x")


def _quantity_objects(text: str, ch: str) -> list[str]:
    return [str(c.object) for c in claims_for_record(_rec(text, ch))
            if c.claim_type == "quantity"]


def test_currency_decimals_survive_in_claim_objects():
    objs = _quantity_objects(
        "user: Tide Detergent (100 oz) is on sale for $11.99 (reg. $14.99).", "h1")
    assert any("$11.99" in o for o in objs), objs   # was 'on sale for $11'


def test_measurement_decimals_survive_in_claim_objects():
    objs = _quantity_objects(
        "user: I recently did a 3-hour hike at the state park, covering 7.5 miles.", "h2")
    assert any("7.5 miles" in o for o in objs), objs   # was 'covering 7'


def test_decimal_with_magnitude_word_survives():
    objs = _quantity_objects("user: The unit cost for a tank is around $8.5 million.", "h3")
    assert any("$8.5 million" in o for o in objs), objs   # was 'around $8'


def test_sentence_final_period_still_ends_the_object():
    # the fix must not swallow sentence boundaries: a '.' NOT followed by a digit still ends
    # the capture, so the second sentence never leaks into the first claim's object
    objs = _quantity_objects("user: The rug cost $40 in total. It arrived on Monday.", "h4")
    assert objs and all("Monday" not in o for o in objs), objs
