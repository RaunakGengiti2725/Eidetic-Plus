"""P2 write-time event identity: lemma+head tags at extraction, instance answers at read.

All dialogues are FABRICATED shapes (album-release, shop-opening, team-up, repeated
pickup); no benchmark rows. The class history: three read-time date-clustering designs
failed because proximity cannot distinguish retellings from distinct events -- identity
is decided at write time where the phrasing exists.
"""
from __future__ import annotations

from datetime import datetime

from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import event_identity as ei
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.smqe.executor import execute_plan
from eidetic.smqe.planner import plan_query
from eidetic.store import RecordStore


def _rec(text, dt, scope, i):
    return MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}", text=text,
                        scope=scope, valid_at=dt.timestamp())


def _store_with(tmp_path, rows):
    store = RecordStore(tmp_path / "ei.sqlite")
    scope = Scope(namespace="ei")
    for i, (text, dt) in enumerate(rows):
        rec = _rec(text, dt, scope, i)
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))
    return store, scope


def _ask(store, scope, q):
    return execute_plan(plan_query(q), q,
                        records=store.active_records_at(scope=scope),
                        claims=store.claims_in_scope(scope))


def test_write_time_tags_carry_lemma_head_date_precision():
    scope = Scope(namespace="tags")
    rec = _rec("Vale: My record finally dropped on the 11th, wild feeling.",
               datetime(2023, 9, 13, 12, 0), scope, 0)
    tagged = [c for c in claims_for_record(rec) if c.filters.get("lemma")]
    assert tagged
    f = tagged[0].filters
    assert f["lemma"] == "release" and f["obj_head"] == "record"
    assert f["event_date"] == "2023-09-11"
    assert f["date_precision"] == ei.PRECISION_RELATIVE_DAY


def test_release_instance_beats_the_later_party(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Vale: My record finally dropped on the 11th and it was a wild feeling.",
         datetime(2023, 9, 13, 12, 0)),
        ("Vale: Last week I threw a small party at my place for my new record.",
         datetime(2023, 11, 3, 12, 0)),
    ])
    res = _ask(store, scope, "When was Vale's record released?")
    assert res is not None and res.answer == "2023-09-11"
    assert ":event_instance" in res.note
    assert "dropped on the 11th" in res.supports[0].proof_atom


def test_statement_report_beats_later_retelling_window(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Rex: I finally opened my own bike repair shop!", datetime(2023, 5, 2, 12, 0)),
        ("Rex: I'm so excited, I opened my bike shop last week!", datetime(2023, 5, 16, 12, 0)),
    ])
    res = _ask(store, scope, "When did Rex start his bike repair shop?")
    assert res is not None and res.answer == "May 2023"
    assert ":event_instance" in res.note


def test_team_up_particle_verb_answers_month(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Mira: I teamed up with a local artist for some cool designs!",
         datetime(2023, 2, 1, 12, 0)),
        ("Mira: The local artist and I are planning a summer collection together.",
         datetime(2023, 5, 10, 12, 0)),
    ])
    res = _ask(store, scope, "When did Mira team up with a local artist?")
    assert res is not None and res.answer == "February 2023"


def test_repeatable_actions_and_multi_instance_stay_with_legacy(tmp_path):
    store, scope = _store_with(tmp_path, [
        ("Kai: Yesterday I picked up the backup badge.", datetime(2024, 2, 12, 12, 0)),
        ("Kai: Yesterday I picked up the backup badge.", datetime(2024, 3, 18, 12, 0)),
    ])
    res = _ask(store, scope, "When did Kai pick up the backup badge?")
    assert res is None or ":event_instance" not in (res.note or "")

    store2, scope2 = _store_with(tmp_path / "span", [
        ("Rex: I opened my first shop downtown!", datetime(2022, 1, 10, 12, 0)),
        ("Rex: I opened my second shop uptown, the shop expansion is real!",
         datetime(2023, 6, 10, 12, 0)),
    ])
    res2 = _ask(store2, scope2, "When did Rex open his shop?")
    assert res2 is None or ":event_instance" not in (res2.note or "")


def test_claim_tier_count_selects_by_head_verb_sense(tmp_path):
    """P1 claim-tier counting: 'how many BOOKS did X read' is COUNT(DISTINCT object) over
    read-family claims -- the head noun selects the verb sense, so the film seen at the
    cinema never counts as a book and each counted item carries its own proof atom.
    Thinner-than-two evidence stays with the legacy collectors."""
    store, scope = _store_with(tmp_path, [
        ("Tim: I read Winter Crossing this month, it was fantastic.",
         datetime(2023, 3, 4, 12, 0)),
        ("Tim: I've read The Long Field recently. Highly recommend.",
         datetime(2023, 3, 10, 12, 0)),
        ("Tim: I read Salt Roads last night, could not put it down.",
         datetime(2023, 3, 20, 12, 0)),
        ("Tim: We saw Arrival at the cinema, great film.", datetime(2023, 3, 22, 12, 0)),
        ("Tim: I was in Chicago for the finals.", datetime(2023, 4, 2, 12, 0)),
        ("Tim: Oh, I've been to Paris yesterday.", datetime(2023, 4, 9, 12, 0)),
    ])
    res = _ask(store, scope, "How many books did Tim read?")
    assert res is not None and res.answer == "3"
    assert ":claim_count" in res.note
    assert len(res.supports) == 3

    res2 = _ask(store, scope, "How many cities has Tim visited?")
    assert res2 is not None and res2.answer == "2"

    lone, lscope = _store_with(tmp_path / "lone", [
        ("Tim: I read Winter Crossing this month.", datetime(2023, 3, 4, 12, 0)),
    ])
    res3 = _ask(lone, lscope, "How many books did Tim read?")
    assert res3 is None or ":claim_count" not in (res3.note or "")


def test_week_only_evidence_answers_month_granularity(tmp_path):
    """When every retelling carries only a week window, the instance answers at MONTH
    granularity -- honest about what the evidence bounds -- rather than fabricating a day
    or shipping the window string."""
    store, scope = _store_with(tmp_path, [
        ("Ana: I launched my pottery studio last week, still buzzing!",
         datetime(2023, 7, 12, 12, 0)),
    ])
    res = _ask(store, scope, "When did Ana open her pottery studio?")
    assert res is not None and res.answer == "July 2023"
    assert ":event_instance" in res.note


def test_adverb_led_first_report_tags_and_wins(tmp_path):
    """'I JUST officially released my zine' -- adverb-led first report tags at statement
    precision and beats a later vague retelling."""
    scope = Scope(namespace="adv")
    rec = _rec("Noa: I just officially released my zine!", datetime(2023, 4, 6, 12, 0), scope, 0)
    tagged = [c for c in claims_for_record(rec) if c.filters.get("lemma") == "release"]
    assert tagged and tagged[0].filters["obj_head"] == "zine"
    assert tagged[0].filters["date_precision"] == ei.PRECISION_STATEMENT

    store, scope2 = _store_with(tmp_path, [
        ("Noa: I just officially released my zine!", datetime(2023, 4, 6, 12, 0)),
        ("Noa: People keep asking about the zine I released a while back.",
         datetime(2023, 9, 1, 12, 0)),
    ])
    res = _ask(store, scope2, "When was Noa's zine released?")
    assert res is not None and res.answer == "April 2023"


def test_started_working_on_shape_tags_open_lemma():
    scope = Scope(namespace="work")
    rec = _rec("Ira: I started working on my novel this winter.",
               datetime(2024, 1, 15, 12, 0), scope, 0)
    tagged = [c for c in claims_for_record(rec) if c.filters.get("lemma") == "open"]
    assert tagged
    assert tagged[0].filters["obj_head"] == "novel"
