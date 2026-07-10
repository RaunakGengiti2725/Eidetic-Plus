"""Offline unit tests for the optimization playbook (no key, no fabricated scores):
event-calendar date normalization, typed preference extraction, query-adaptive RRF,
conformal-threshold plumbing, edge-placement, and the sweep dry-run plan."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from bench.calibrate import conformal_threshold
from bench.index_timing import run_timing
from bench.sweep import STAGES, plan, stage_assignment
from eidetic.bm25 import BM25, PersistentBM25
from eidetic.events import EventRecord, normalize_dates, parse_query, select_for_query
from eidetic.preferences import extract_preference, is_preference
from eidetic.retrieval import (
    _hippo2_seed_entities, _reader_model, _rrf, _temporal_context_order,
    _user_query_terms, compress_chunk, edge_place,
)

REF = datetime(2026, 6, 23, 12, 0, 0).timestamp()   # Tue 23 Jun 2026


# ---- 1. event-calendar date normalization (ranges, reference-relative) ----
def test_date_normalization_ranges():
    got = {d["expr"]: (d["start"], d["end"]) for d in
           normalize_dates("we met yesterday, in May 2023, and 3 days ago", REF)}
    assert got["yesterday"] == ("2026-06-22T00:00:00", "2026-06-22T23:59:59")
    assert got["May 2023"] == ("2023-05-01T00:00:00", "2023-05-31T23:59:59")  # month RANGE
    assert got["3 days ago"][0].startswith("2026-06-20")


def test_structured_selection_interval_overlap():
    p = parse_query("How many times did Caroline visit Paris in May 2023?", REF)
    assert p["operation"] == "count" and "Caroline" in p["entities"] and p["ranges"]
    evs = [EventRecord(subject="Caroline", verb="visit", object="Paris", fact="Caroline visited Paris",
                       start=datetime(2023, 5, 10).timestamp(), end=datetime(2023, 5, 10).timestamp()),
           EventRecord(subject="Caroline", verb="visit", object="Lyon", fact="Caroline visited Lyon",
                       start=datetime(2024, 1, 1).timestamp(), end=datetime(2024, 1, 1).timestamp())]
    sel = select_for_query(evs, p, REF)
    assert [e.object for e in sel] == ["Paris"]   # overlap + entity; reader does the counting


# ---- 2. typed preference extraction ----
def test_typed_preference_extraction():
    assert is_preference("user: I prefer window seats on long flights.")
    assert is_preference("I'm allergic to peanuts.")
    assert not is_preference("The meeting is at 3pm on Tuesday.")
    assert "window seats" in extract_preference("I prefer window seats.")


# ---- 3. query-adaptive / weighted RRF ----
def test_weighted_rrf_changes_ranking():
    a, b = ["x", "y"], ["y", "z"]
    fair = _rrf([a, b], 60)
    boosted = _rrf([a, b], 60, [0.1, 3.0])           # trust list b more
    assert boosted["z"] > fair["z"]
    assert _rrf([a], 60)["x"] > _rrf([a], 60)["y"]   # rank-1 beats rank-2


def test_query_adaptive_flags():
    namey = parse_query("What is order EX-7741 status?", REF)
    assert namey["is_namey"]
    multi = parse_query("How are Alice and Bob and Carol connected?", REF)
    assert multi["is_multihop"]


# ---- 4. conformal-threshold plumbing ----
def test_conformal_threshold_hits_target_precision():
    # Low-signal items are wrong; high-signal items are correct -> threshold should land
    # where answered-set precision >= 0.95.
    samples = ([{"signal": 0.9, "correct": True}] * 18
               + [{"signal": 0.8, "correct": True}] * 1
               + [{"signal": 0.2, "correct": False}] * 10)
    res = conformal_threshold(samples, target_precision=0.95)
    assert res["ok"] and res["precision"] >= 0.95
    assert res["threshold"] >= 0.8          # excludes the low-signal wrong ones
    assert conformal_threshold([], 0.95)["ok"] is False


# ---- 5. edge-placement (lost-in-the-middle) ----
def test_edge_placement_puts_top_at_edges():
    assert edge_place(["1", "2", "3", "4", "5"]) == ["1", "3", "5", "4", "2"]
    # highest (1) and 2nd-highest (2) end up at the two ends
    placed = edge_place(["A", "B", "C", "D"])
    assert placed[0] == "A" and placed[-1] == "B"


def test_compression_keeps_relevant_sentences():
    out = compress_chunk("The cat sat. Revenue rose to 9M in Q2. Birds migrate south.",
                         "revenue Q2", 0.34)
    assert "Revenue" in out and "cat" not in out
    # ratio >= 1.0 is a no-op
    assert compress_chunk("a. b. c.", "q", 1.0) == "a. b. c."


# ---- integration: events + preferences reach the assembled context (no LLM) ----
def test_assemble_context_surfaces_events_and_preferences(engine):
    from eidetic.events import EventRecord
    from eidetic.models import Scope

    ns = "teamX"
    t = datetime(2023, 5, 10).timestamp()
    engine.store.add_event(EventRecord(subject="Caroline", verb="visited", object="Paris",
                                       fact="Caroline visited Paris", start=t, end=t,
                                       namespace=ns, valid_at=t))
    engine.store.add_profile_line(ns, "prefers window seats", salience=0.8)
    # This is exactly what the neutral benchmark adapter calls (candidates empty here).
    blocks = engine.retriever.assemble_context(
        "Where did Caroline go in May 2023?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)
    assert "Caroline visited Paris" in joined          # event calendar reached the context
    assert "User preference: prefers window seats" in joined  # typed preference surfaced


def test_assemble_context_filters_events_by_source_scope(engine):
    from eidetic.events import EventRecord
    from eidetic.models import MemoryRecord, Scope

    ns = "event-source-scope"
    scope_a = Scope(namespace=ns, agent_id="agent-a", project_id="project")
    scope_b = Scope(namespace=ns, agent_id="agent-b", project_id="project")
    t = datetime(2023, 5, 10).timestamp()
    rec_a = MemoryRecord(memory_id="event-a", text="Caroline visited Paris.",
                         scope=scope_a, valid_at=t,
                         content_hash="a" * 64, raw_uri="cas://" + "a" * 64)
    rec_b = MemoryRecord(memory_id="event-b", text="Caroline visited Rome.",
                         scope=scope_b, valid_at=t,
                         content_hash="b" * 64, raw_uri="cas://" + "b" * 64)
    engine.store.upsert_record(rec_a)
    engine.store.upsert_record(rec_b)
    engine.store.add_event(EventRecord(
        subject="Caroline", verb="visited", object="Paris", fact="Caroline visited Paris",
        start=t, end=t, namespace=ns, valid_at=t, source_memory_id=rec_a.memory_id))
    engine.store.add_event(EventRecord(
        subject="Caroline", verb="visited", object="Rome", fact="Caroline visited Rome",
        start=t, end=t, namespace=ns, valid_at=t, source_memory_id=rec_b.memory_id))

    blocks = engine.retriever.assemble_context(
        "Where did Caroline go in May 2023?", [], at=t, scope=scope_a)
    joined = " ".join(blocks)

    assert "Caroline visited Paris" in joined
    assert "Caroline visited Rome" not in joined


def test_assemble_context_filters_events_by_source_validity(engine):
    from eidetic.events import EventRecord
    from eidetic.models import MemoryRecord, Scope

    ns = "event-source-validity"
    scope = Scope(namespace=ns)
    old_t = datetime(2023, 5, 10).timestamp()
    new_t = datetime(2023, 5, 11).timestamp()
    old = MemoryRecord(memory_id="event-old", text="Caroline visited Paris.",
                       scope=scope, valid_at=old_t, invalid_at=new_t,
                       content_hash="c" * 64, raw_uri="cas://" + "c" * 64)
    new = MemoryRecord(memory_id="event-new", text="Caroline visited Lyon.",
                       scope=scope, valid_at=new_t,
                       content_hash="d" * 64, raw_uri="cas://" + "d" * 64)
    engine.store.upsert_record(old)
    engine.store.upsert_record(new)
    engine.store.add_event(EventRecord(
        subject="Caroline", verb="visited", object="Paris", fact="Caroline visited Paris",
        start=old_t, end=old_t, namespace=ns, valid_at=old_t, source_memory_id=old.memory_id))
    engine.store.add_event(EventRecord(
        subject="Caroline", verb="visited", object="Lyon", fact="Caroline visited Lyon",
        start=new_t, end=new_t, namespace=ns, valid_at=new_t, source_memory_id=new.memory_id))

    blocks = engine.retriever.assemble_context(
        "Where did Caroline go in May 2023?", [], at=new_t + 10, scope=scope)
    joined = " ".join(blocks)

    assert "Caroline visited Lyon" in joined
    assert "Caroline visited Paris" not in joined


def test_assemble_context_profile_preferences_carry_compact_source_hints(engine):
    from eidetic.models import MemoryRecord, Scope

    ns = "pref-prov"
    h = "a" * 64
    rec = MemoryRecord(memory_id="mem-tea", text="user: I love ginger tea.",
                       scope=Scope(namespace=ns), valid_at=123.0,
                       content_hash=h, raw_uri=f"cas://{h}")
    engine.store.upsert_record(rec)
    engine.store.add_profile_line(
        ns,
        "User likes ginger tea.",
        salience=0.8,
        source_memory_id=rec.memory_id,
        content_hash=h,
        raw_uri=f"cas://{h}",
        valid_at=123.0,
    )

    blocks = engine.retriever.assemble_context(
        "What tea does the user like?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)

    assert "User preference: User likes ginger tea." in joined
    assert "source_memory_id=mem-tea" in joined
    assert f"content_hash={h[:16]}" in joined


def test_assemble_context_filters_profile_preferences_by_source_scope(engine):
    from eidetic.models import MemoryRecord, Scope

    ns = "pref-source-scope"
    scope_a = Scope(namespace=ns, agent_id="agent-a", project_id="project")
    scope_b = Scope(namespace=ns, agent_id="agent-b", project_id="project")
    rec_a = MemoryRecord(memory_id="pref-a", text="user: I love jasmine tea.",
                         scope=scope_a, valid_at=100.0,
                         content_hash="a" * 64, raw_uri="cas://" + "a" * 64)
    rec_b = MemoryRecord(memory_id="pref-b", text="user: I love espresso.",
                         scope=scope_b, valid_at=100.0,
                         content_hash="b" * 64, raw_uri="cas://" + "b" * 64)
    for rec, line, salience in (
        (rec_a, "User likes jasmine tea.", 0.1),
        (rec_b, "User likes espresso.", 0.9),
    ):
        engine.store.upsert_record(rec)
        engine.store.add_profile_line(
            ns,
            line,
            salience=salience,
            source_memory_id=rec.memory_id,
            content_hash=rec.content_hash,
            raw_uri=rec.raw_uri,
            valid_at=rec.valid_at,
            scope=rec.scope,
        )

    blocks = engine.retriever.assemble_context(
        "What tea does the user like?", [], at=150.0, scope=scope_a)
    joined = " ".join(blocks)

    assert "User preference: User likes jasmine tea." in joined
    assert "User preference: User likes espresso." not in joined


def test_assemble_context_filters_profile_preferences_by_source_validity(engine):
    from eidetic.models import MemoryRecord, Scope

    ns = "pref-source-validity"
    scope = Scope(namespace=ns)
    old = MemoryRecord(memory_id="old-pref", text="user: I love coffee.",
                       scope=scope, valid_at=100.0, invalid_at=200.0,
                       content_hash="c" * 64, raw_uri="cas://" + "c" * 64)
    new = MemoryRecord(memory_id="new-pref", text="user: I love mint tea.",
                       scope=scope, valid_at=250.0,
                       content_hash="d" * 64, raw_uri="cas://" + "d" * 64)
    for rec, line, salience in (
        (old, "User likes coffee.", 0.9),
        (new, "User likes mint tea.", 0.1),
    ):
        engine.store.upsert_record(rec)
        engine.store.add_profile_line(
            ns,
            line,
            salience=salience,
            source_memory_id=rec.memory_id,
            content_hash=rec.content_hash,
            raw_uri=rec.raw_uri,
            valid_at=rec.valid_at,
            scope=rec.scope,
        )

    blocks = engine.retriever.assemble_context(
        "What does the user like?", [], at=300.0, scope=scope)
    joined = " ".join(blocks)

    assert "User preference: User likes mint tea." in joined
    assert "User preference: User likes coffee." not in joined


def test_assemble_context_filters_profile_preferences_with_missing_source(engine):
    from eidetic.models import Scope

    ns = "pref-source-missing"
    scope = Scope(namespace=ns)
    h = "e" * 64
    engine.store.add_profile_line(
        ns,
        "User likes phantom tea.",
        salience=0.9,
        source_memory_id="missing-pref",
        content_hash=h,
        raw_uri=f"cas://{h}",
        valid_at=100.0,
    )
    engine.store.add_profile_line(
        ns,
        "Legacy namespace-wide preference.",
        salience=0.1,
    )

    blocks = engine.retriever.assemble_context(
        "What does the user like?", [], at=150.0, scope=scope)
    joined = " ".join(blocks)

    assert "User preference: User likes phantom tea." not in joined
    assert "User preference: Legacy namespace-wide preference." in joined


def test_assemble_context_caps_count_events_at_twelve(engine):
    from eidetic.events import EventRecord
    from eidetic.models import Scope

    ns = "countcap"
    base = datetime(2023, 5, 1).timestamp()
    for i in range(15):
        t = base + i * 86400
        engine.store.add_event(EventRecord(subject="Caroline", verb="visited", object=f"place{i}",
                                           fact=f"Caroline visited place{i}", start=t, end=t,
                                           namespace=ns, valid_at=t))
    blocks = engine.retriever.assemble_context(
        "How many times did Caroline visit places?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)
    for i in range(12):
        assert f"place{i}" in joined
    assert "place12" not in joined


def test_assemble_context_surfaces_top_eight_preferences(engine):
    from eidetic.models import Scope

    ns = "prefcap"
    for i in range(10):
        engine.store.add_profile_line(ns, f"prefers item{i}", salience=10 - i)
    blocks = engine.retriever.assemble_context(
        "What does the user prefer?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)
    for i in range(8):
        assert f"prefers item{i}" in joined
    assert "prefers item8" not in joined


def test_assemble_context_promotes_query_relevant_preference(engine):
    from eidetic.models import Scope

    ns = "pref-relevant"
    for i in range(10):
        engine.store.add_profile_line(ns, f"User prefers item{i}.", salience=20 - i)
    engine.store.add_profile_line(ns, "User's favorite music is jazz.", salience=0.1)
    blocks = engine.retriever.assemble_context(
        "What is the user's favorite music?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)
    assert "User's favorite music is jazz" in joined
    # The profile is still capped at 8 lines; the relevant low-salience line displaces a generic one.
    assert "User prefers item7" not in joined


def test_assemble_context_promotes_allergy_preference_even_with_generic_food_word(engine):
    from eidetic.models import Scope

    ns = "pref-allergy"
    for i in range(10):
        engine.store.add_profile_line(ns, f"User likes snack{i}.", salience=20 - i)
    engine.store.add_profile_line(ns, "User is allergic to peanuts.", salience=0.1)
    blocks = engine.retriever.assemble_context(
        "What food is the user allergic to?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)
    assert "User is allergic to peanuts" in joined
    assert "User likes snack7" not in joined


def test_assemble_context_uses_active_preferences_not_stale_profile_history(engine):
    from eidetic.models import Scope

    ns = "pref-stale"
    engine.store.add_profile_line(ns, "User likes coffee.", salience=0.9)
    engine.store.add_profile_line(ns, "User dislikes coffee.", salience=0.4)

    blocks = engine.retriever.assemble_context(
        "Does the user like coffee?", [], at=None, scope=Scope(namespace=ns))
    joined = " ".join(blocks)

    assert "User preference: User dislikes coffee." in joined
    assert "User preference: User likes coffee." not in joined
    assert "User likes coffee." in engine.store.get_profile(ns, include_inactive=True)


def test_assemble_context_surfaces_active_graph_facts_only(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefacts")
    old = datetime(2024, 1, 1, 12).timestamp()
    new = datetime(2024, 2, 1, 12).timestamp()
    read_at = datetime(2024, 3, 1, 12).timestamp()

    engine.graph.add_fact(
        "Alice", "works_at", "Acme", fact="Alice works at Acme", valid_at=old, scope=scope)
    engine.graph.add_fact(
        "Alice", "works_at", "Beta", fact="Alice works at Beta", valid_at=new, scope=scope)

    blocks = engine.retriever.assemble_context(
        "Where does Alice work now?", [], at=read_at, scope=scope)
    joined = "\n".join(blocks)
    assert "Current active facts" in joined
    assert "Alice works at Beta" in joined
    assert "Alice works at Acme" not in joined


def test_active_graph_fact_context_filters_same_entity_unrelated_facts(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefacts-filter")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Alice", "works_at", "Beta", fact="Alice works at Beta", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Alice", "likes", "tea", fact="Alice likes tea", valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "Where does Alice work now?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Alice works at Beta" in joined
    assert "Alice likes tea" not in joined


def test_active_fact_source_channel_promotes_current_raw_source(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefact-source")
    old = datetime(2024, 1, 1, 12).timestamp()
    new = datetime(2024, 2, 1, 12).timestamp()
    read_at = datetime(2024, 3, 1, 12).timestamp()

    records = {
        "old": MemoryRecord(
            memory_id="old", content_hash="old", text="Alice works at Acme",
            scope=scope, valid_at=old, entities=["Alice", "Acme"]),
        "new": MemoryRecord(
            memory_id="new", content_hash="new", text="Alice works at Beta",
            scope=scope, valid_at=new, entities=["Alice", "Beta"]),
        "tea": MemoryRecord(
            memory_id="tea", content_hash="tea", text="Alice likes tea",
            scope=scope, valid_at=new, entities=["Alice", "tea"]),
    }
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_at", "Acme", fact="Alice works at Acme",
        source_memory_id="old", valid_at=old, scope=scope)
    engine.graph.add_fact(
        "Alice", "works_at", "Beta", fact="Alice works at Beta",
        source_memory_id="new", valid_at=new, scope=scope)
    engine.graph.add_fact(
        "Alice", "likes", "tea", fact="Alice likes tea",
        source_memory_id="tea", valid_at=new, scope=scope)

    order, scores = engine.retriever._run_active_fact_sources(
        "Where does Alice work now?", parse_query("Where does Alice work now?", read_at),
        records, [], read_at, scope, set(records))
    assert order == ["new"]
    assert scores["new"] > 0


def test_active_fact_channel_filters_edges_with_stale_source_memory(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefact-source-stale")
    t = datetime(2024, 1, 1, 12).timestamp()
    read_at = t + 10
    stale = MemoryRecord(
        memory_id="stale-work",
        content_hash="stale-work",
        text="Alice works at Acme.",
        scope=scope,
        valid_at=t,
        invalid_at=t + 5,
        entities=["Alice", "Acme"],
    )
    engine.store.upsert_record(stale)
    engine.graph.add_fact(
        "Alice", "works_at", "Acme", fact="Alice works at Acme",
        source_memory_id=stale.memory_id, valid_at=t, scope=scope)

    query = "Where does Alice work now?"
    order, scores = engine.retriever._run_active_fact_sources(
        query, parse_query(query, read_at), {stale.memory_id: stale}, [], read_at,
        scope, {stale.memory_id})
    blocks = engine.retriever.assemble_context(query, [], at=read_at, scope=scope)

    assert order == []
    assert scores == {}
    assert "Alice works at Acme" not in "\n".join(blocks)


def test_active_fact_channel_filters_edges_with_missing_source_memory(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefact-source-missing")
    t = datetime(2024, 1, 1, 12).timestamp()
    query = "Where does Alice work now?"
    engine.graph.add_fact(
        "Alice", "works_at", "Acme", fact="Alice works at Acme",
        source_memory_id="missing-work", valid_at=t, scope=scope)

    order, scores = engine.retriever._run_active_fact_sources(
        query, parse_query(query, t + 10), {}, [], t + 10, scope, {"missing-work"})
    blocks = engine.retriever.assemble_context(query, [], at=t + 10, scope=scope)

    assert order == []
    assert scores == {}
    assert "Alice works at Acme" not in "\n".join(blocks)


def test_active_fact_context_matches_employer_wording_without_topic_distractor(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefacts-employer")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Alice", "works_at", "Beta", fact="Alice works at Beta",
        valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Alice", "studied", "employer branding",
        fact="Alice studied employer branding",
        valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "Who is Alice's employer?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Alice works at Beta" in joined
    assert "Alice studied employer branding" not in joined


def test_active_fact_source_channel_matches_employed_by_wording(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefact-source-employer")
    t = datetime(2024, 1, 1, 12).timestamp()
    records = {
        "work": MemoryRecord(
            memory_id="work", content_hash="work", text="Alice works at Beta.",
            scope=scope, valid_at=t, entities=["Alice", "Beta"]),
        "branding": MemoryRecord(
            memory_id="branding", content_hash="branding",
            text="Alice studied employer branding.",
            scope=scope, valid_at=t, entities=["Alice", "employer branding"]),
    }
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_at", "Beta", fact="Alice works at Beta",
        source_memory_id="work", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Alice", "studied", "employer branding",
        fact="Alice studied employer branding",
        source_memory_id="branding", valid_at=t, scope=scope)

    q = "Who is Alice employed by?"
    order, scores = engine.retriever._run_active_fact_sources(
        q, parse_query(q, t + 10), records, [], t + 10, scope, set(records))
    assert order == ["work"]
    assert scores["work"] > 0


def test_active_fact_context_keeps_work_on_project_queries_distinct(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefacts-work-on")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios",
        fact="Alice works on Project Helios",
        valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Alice", "studied", "employer branding",
        fact="Alice studied employer branding",
        valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "What does Alice work on?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Alice works on Project Helios" in joined
    assert "Alice studied employer branding" not in joined


def test_active_fact_context_matches_based_city_wording_without_visit_distractor(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefacts-location")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Alice", "lives_in", "Seattle", fact="Alice lives in Seattle",
        valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Alice", "visited", "City Hall", fact="Alice visited City Hall",
        valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "What city is Alice based in?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Alice lives in Seattle" in joined
    assert "Alice visited City Hall" not in joined


def test_active_fact_source_channel_matches_based_location_wording(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefact-source-location")
    t = datetime(2024, 1, 1, 12).timestamp()
    records = {
        "home": MemoryRecord(
            memory_id="home", content_hash="home", text="Alice lives in Seattle.",
            scope=scope, valid_at=t, entities=["Alice", "Seattle"]),
        "visit": MemoryRecord(
            memory_id="visit", content_hash="visit", text="Alice visited City Hall.",
            scope=scope, valid_at=t, entities=["Alice", "City Hall"]),
    }
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "lives_in", "Seattle", fact="Alice lives in Seattle",
        source_memory_id="home", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Alice", "visited", "City Hall", fact="Alice visited City Hall",
        source_memory_id="visit", valid_at=t, scope=scope)

    q = "Where is Alice based now?"
    order, scores = engine.retriever._run_active_fact_sources(
        q, parse_query(q, t + 10), records, [], t + 10, scope, set(records))
    assert order == ["home"]
    assert scores["home"] > 0


def test_active_fact_context_keeps_visit_city_queries_distinct(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="activefacts-city-visit")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Alice", "lives_in", "Seattle", fact="Alice lives in Seattle",
        valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Alice", "visited", "Paris", fact="Alice visited Paris",
        valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "What city did Alice visit?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Alice visited Paris" in joined
    assert "Alice lives in Seattle" not in joined


def test_active_fact_context_expands_relationship_status_terms(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="relationship-status")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Caroline", "is", "single", fact="Caroline is single",
        valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Caroline", "joined", "mentorship", fact="Caroline joined a mentorship program",
        valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "What is Caroline's relationship status?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Caroline is single" in joined
    assert "Caroline joined a mentorship program" not in joined


def test_relationship_status_terms_do_not_pollute_date_questions(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="relationship-status-date-guard")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Caroline", "is", "single", fact="Caroline is single",
        valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Caroline", "joined", "mentorship", fact="Caroline joined a mentorship program",
        valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "What date did Caroline join mentorship?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Caroline joined a mentorship program" in joined
    assert "Caroline is single" not in joined


def test_active_fact_source_channel_promotes_relationship_status_source(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="relationship-status-source")
    t = datetime(2024, 1, 1, 12).timestamp()
    records = {
        "single": MemoryRecord(
            memory_id="single", content_hash="single", text="Caroline is single.",
            scope=scope, valid_at=t, entities=["Caroline"]),
        "mentor": MemoryRecord(
            memory_id="mentor", content_hash="mentor", text="Caroline joined a mentorship program.",
            scope=scope, valid_at=t, entities=["Caroline"]),
    }
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Caroline", "is", "single", fact="Caroline is single",
        source_memory_id="single", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Caroline", "joined", "mentorship", fact="Caroline joined a mentorship program",
        source_memory_id="mentor", valid_at=t, scope=scope)

    q = "What is Caroline's relationship status?"
    order, scores = engine.retriever._run_active_fact_sources(
        q, parse_query(q, t + 10), records, [], t + 10, scope, set(records))
    assert order == ["single"]
    assert scores["single"] > 0


def test_active_fact_context_expands_action_verb_inflections(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="action-verb-inflection")
    t = datetime(2024, 1, 1, 12).timestamp()

    engine.graph.add_fact(
        "Caroline", "researched", "adoption agencies",
        fact="Caroline researched adoption agencies",
        valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Caroline", "likes", "counseling",
        fact="Caroline is interested in counseling as a career option",
        valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "What did Caroline research?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Caroline researched adoption agencies" in joined
    assert "interested in counseling" not in joined


def test_active_fact_source_channel_promotes_action_verb_source(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, active_fact_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="action-verb-source")
    t = datetime(2024, 1, 1, 12).timestamp()
    records = {
        "adoption": MemoryRecord(
            memory_id="adoption", content_hash="adoption",
            text="Caroline researched adoption agencies.",
            scope=scope, valid_at=t, entities=["Caroline", "adoption agencies"]),
        "counseling": MemoryRecord(
            memory_id="counseling", content_hash="counseling",
            text="Caroline is interested in counseling as a career option.",
            scope=scope, valid_at=t, entities=["Caroline", "counseling"]),
    }
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Caroline", "researched", "adoption agencies",
        fact="Caroline researched adoption agencies",
        source_memory_id="adoption", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Caroline", "likes", "counseling",
        fact="Caroline is interested in counseling as a career option",
        source_memory_id="counseling", valid_at=t, scope=scope)

    q = "What did Caroline research?"
    order, scores = engine.retriever._run_active_fact_sources(
        q, parse_query(q, t + 10), records, [], t + 10, scope, set(records))
    assert order == ["adoption"]
    assert scores["adoption"] > 0


def test_graph_bridge_context_surfaces_entity_connection_edges(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, graph_bridge_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="bridge-context")
    t = datetime(2024, 1, 1, 12).timestamp()
    for rec in [
        MemoryRecord(memory_id="alice", content_hash="alice",
                     text="Alice works on Project Helios",
                     scope=scope, valid_at=t, entities=["Alice", "Project Helios"]),
        MemoryRecord(memory_id="bob", content_hash="bob",
                     text="Bob leads Project Helios",
                     scope=scope, valid_at=t + 10, entities=["Bob", "Project Helios"]),
        MemoryRecord(memory_id="carol", content_hash="carol",
                     text="Carol likes coffee",
                     scope=scope, valid_at=t, entities=["Carol", "coffee"]),
    ]:
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios", fact="Alice works on Project Helios",
        source_memory_id="alice", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Bob", "leads", "Project Helios", fact="Bob leads Project Helios",
        source_memory_id="bob", valid_at=t + 10, scope=scope)
    engine.graph.add_fact(
        "Carol", "likes", "coffee", fact="Carol likes coffee",
        source_memory_id="carol", valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "How are Alice and Bob connected?", [], at=t + 20, scope=scope)
    joined = "\n".join(blocks)
    assert "Graph bridge evidence" in joined
    assert "Alice works on Project Helios" in joined
    assert "Bob leads Project Helios" in joined
    assert "Carol likes coffee" not in joined


def test_graph_bridge_context_discovers_lowercase_query_entities(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, graph_bridge_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="bridge-lowercase")
    t = datetime(2024, 1, 1, 12).timestamp()
    for rec in [
        MemoryRecord(memory_id="alice", content_hash="alice",
                     text="Alice works on Project Helios",
                     scope=scope, valid_at=t, entities=["Alice", "Project Helios"]),
        MemoryRecord(memory_id="bob", content_hash="bob",
                     text="Bob leads Project Helios",
                     scope=scope, valid_at=t + 10, entities=["Bob", "Project Helios"]),
    ]:
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios", fact="Alice works on Project Helios",
        source_memory_id="alice", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Bob", "leads", "Project Helios", fact="Bob leads Project Helios",
        source_memory_id="bob", valid_at=t + 10, scope=scope)

    blocks = engine.retriever.assemble_context(
        "how are alice and bob connected?", [], at=t + 20, scope=scope)
    joined = "\n".join(blocks)
    assert "Graph bridge evidence" in joined
    assert "Alice works on Project Helios" in joined
    assert "Bob leads Project Helios" in joined


def test_graph_bridge_context_discovers_unquoted_multiword_entity(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, graph_bridge_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="bridge-multiword")
    t = datetime(2024, 1, 1, 12).timestamp()
    for rec in [
        MemoryRecord(memory_id="alice", content_hash="alice",
                     text="Alice works on Project Helios",
                     scope=scope, valid_at=t, entities=["Alice", "Project Helios"]),
        MemoryRecord(memory_id="bob", content_hash="bob",
                     text="Bob leads Project Helios",
                     scope=scope, valid_at=t + 10, entities=["Bob", "Project Helios"]),
    ]:
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios", fact="Alice works on Project Helios",
        source_memory_id="alice", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Bob", "leads", "Project Helios", fact="Bob leads Project Helios",
        source_memory_id="bob", valid_at=t + 10, scope=scope)

    blocks = engine.retriever.assemble_context(
        "How is Bob connected to project helios?", [], at=t + 20, scope=scope)
    joined = "\n".join(blocks)
    assert "Graph bridge evidence" in joined
    assert "Alice works on Project Helios" in joined
    assert "Bob leads Project Helios" in joined


def test_graph_bridge_vocab_avoids_generic_multiword_distractors(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, graph_bridge_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="bridge-vocab-precision")
    t = datetime(2024, 1, 1, 12).timestamp()
    alice = MemoryRecord(
        memory_id="alice", content_hash="alice", text="Alice works on Project Helios",
        scope=scope, valid_at=t, entities=["Alice", "Project Helios"])
    bob = MemoryRecord(
        memory_id="bob", content_hash="bob", text="Bob leads Project Helios",
        scope=scope, valid_at=t + 10, entities=["Bob", "Project Helios"])
    dana = MemoryRecord(
        memory_id="dana", content_hash="dana", text="Dana manages Project Atlas",
        scope=scope, valid_at=t + 20, entities=["Dana", "Project Atlas"])
    records = {r.memory_id: r for r in (alice, bob, dana)}
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios", fact=alice.text,
        source_memory_id="alice", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Bob", "leads", "Project Helios", fact=bob.text,
        source_memory_id="bob", valid_at=t + 10, scope=scope)
    engine.graph.add_fact(
        "Dana", "manages", "Project Atlas", fact=dana.text,
        source_memory_id="dana", valid_at=t + 20, scope=scope)

    query = "How is Bob connected to project helios?"
    blocks = engine.retriever.assemble_context(query, [], at=t + 30, scope=scope)
    joined = "\n".join(blocks)
    assert "Alice works on Project Helios" in joined
    assert "Bob leads Project Helios" in joined
    assert "Project Atlas" not in joined

    ensured = engine.retriever._ensure_graph_bridge_candidates(
        query,
        parse_query(query, t + 30),
        [RetrievalCandidate(record=bob, fused_score=1.0)],
        records,
        t + 30,
        scope,
    )
    assert [c.record.memory_id for c in ensured] == ["bob", "alice"]


def test_graph_bridge_completion_adds_raw_sources_for_bridge_edges(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, graph_bridge_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="bridge-candidate")
    t = datetime(2024, 1, 1, 12).timestamp()
    seed = MemoryRecord(
        memory_id="seed", content_hash="seed", text="unrelated seed",
        scope=scope, valid_at=t, entities=["seed"])
    alice = MemoryRecord(
        memory_id="alice", content_hash="alice", text="Alice works on Project Helios",
        scope=scope, valid_at=t, entities=["Alice", "Project Helios"])
    bob = MemoryRecord(
        memory_id="bob", content_hash="bob", text="Bob leads Project Helios",
        scope=scope, valid_at=t + 10, entities=["Bob", "Project Helios"])
    records = {r.memory_id: r for r in (seed, alice, bob)}
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios", fact=alice.text,
        source_memory_id="alice", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Bob", "leads", "Project Helios", fact=bob.text,
        source_memory_id="bob", valid_at=t + 10, scope=scope)

    parsed = parse_query("How are Alice and Bob connected?", t + 20)
    ensured = engine.retriever._ensure_graph_bridge_candidates(
        "How are Alice and Bob connected?",
        parsed,
        [RetrievalCandidate(record=seed, fused_score=1.0)],
        records,
        t + 20,
        scope,
    )
    assert [c.record.memory_id for c in ensured] == ["seed", "bob", "alice"]


def test_graph_bridge_context_filters_edges_with_hidden_source_memory(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, graph_bridge_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="bridge-source-scope", agent_id="agent-a", project_id="project")
    other = Scope(namespace="bridge-source-scope", agent_id="agent-b", project_id="project")
    t = datetime(2024, 1, 1, 12).timestamp()
    alice = MemoryRecord(
        memory_id="alice-source", content_hash="alice-source",
        text="Alice works on Project Helios.",
        scope=scope, valid_at=t, entities=["Alice", "Project Helios"])
    hidden_bob = MemoryRecord(
        memory_id="hidden-bob-source", content_hash="hidden-bob-source",
        text="Bob leads Project Secret.",
        scope=other, valid_at=t, entities=["Bob", "Project Secret"])
    engine.store.upsert_record(alice)
    engine.store.upsert_record(hidden_bob)
    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios", fact="Alice works on Project Helios",
        source_memory_id=alice.memory_id, valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Bob", "leads", "Project Secret", fact="Bob leads Project Secret",
        source_memory_id=hidden_bob.memory_id, valid_at=t, scope=scope)

    blocks = engine.retriever.assemble_context(
        "How are Alice and Bob connected?", [], at=t + 20, scope=scope)
    joined = "\n".join(blocks)

    assert "Alice works on Project Helios" in joined
    assert "Bob leads Project Secret" not in joined


def test_graph_bridge_completion_adds_sources_for_lowercase_bridge_query(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, graph_bridge_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="bridge-candidate-lowercase")
    t = datetime(2024, 1, 1, 12).timestamp()
    seed = MemoryRecord(
        memory_id="seed", content_hash="seed", text="unrelated seed",
        scope=scope, valid_at=t, entities=["seed"])
    alice = MemoryRecord(
        memory_id="alice", content_hash="alice", text="Alice works on Project Helios",
        scope=scope, valid_at=t, entities=["Alice", "Project Helios"])
    bob = MemoryRecord(
        memory_id="bob", content_hash="bob", text="Bob leads Project Helios",
        scope=scope, valid_at=t + 10, entities=["Bob", "Project Helios"])
    records = {r.memory_id: r for r in (seed, alice, bob)}
    for rec in records.values():
        engine.store.upsert_record(rec)
    engine.graph.add_fact(
        "Alice", "works_on", "Project Helios", fact=alice.text,
        source_memory_id="alice", valid_at=t, scope=scope)
    engine.graph.add_fact(
        "Bob", "leads", "Project Helios", fact=bob.text,
        source_memory_id="bob", valid_at=t + 10, scope=scope)

    query = "how are alice and bob connected?"
    parsed = parse_query(query, t + 20)
    ensured = engine.retriever._ensure_graph_bridge_candidates(
        query,
        parsed,
        [RetrievalCandidate(record=seed, fused_score=1.0)],
        records,
        t + 20,
        scope,
    )
    assert [c.record.memory_id for c in ensured] == ["seed", "bob", "alice"]


def test_user_evidence_context_surfaces_matching_user_turn(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, user_evidence_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="user-context")
    t = datetime(2024, 1, 1, 12).timestamp()
    engine.store.upsert_record(MemoryRecord(
        memory_id="book", content_hash="book",
        text=("assistant: Any books lately?\n"
              "user: I finished reading The Glass Menagerie last night.\n"
              "assistant: Nice choice."),
        scope=scope, valid_at=t, entities=["The Glass Menagerie"],
    ))
    engine.store.upsert_record(MemoryRecord(
        memory_id="unrelated-user", content_hash="unrelated-user",
        text="assistant: Any errands?\nuser: I bought oat milk.",
        scope=scope, valid_at=t, entities=["oat milk"],
    ))

    blocks = engine.retriever.assemble_context(
        "What book did I read?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "User evidence audit" in joined
    assert "The Glass Menagerie" in joined
    assert "oat milk" not in joined


def test_user_evidence_context_keeps_nearby_user_store_context(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, user_evidence_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="user-context-nearby-store")
    t = datetime(2024, 1, 1, 12).timestamp()
    engine.store.upsert_record(MemoryRecord(
        memory_id="coupon", content_hash="coupon",
        text=("assistant: Any coupon organization tools?\n"
              "user: I've been using the SnipSave app from GreenGrocer for pantry staples.\n"
              "assistant: Great, keep those offers organized.\n"
              "user: I actually redeemed a $4 coupon on almond butter last Tuesday."),
        scope=scope, valid_at=t, entities=["GreenGrocer", "almond butter"],
    ))

    blocks = engine.retriever.assemble_context(
        "Where did I redeem a $4 coupon on almond butter?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "User evidence audit" in joined
    assert "SnipSave app from GreenGrocer" in joined
    assert "redeemed a $4 coupon on almond butter" in joined


def test_user_evidence_context_keeps_current_match_before_long_prior_context(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, user_evidence_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="user-context-current-first")
    t = datetime(2024, 1, 1, 12).timestamp()
    long_prior = " ".join(["paperwork"] * 220)
    engine.store.upsert_record(MemoryRecord(
        memory_id="degree", content_hash="degree",
        text=(f"user: {long_prior}\n"
              "assistant: Any education background?\n"
              "user: I graduated with a degree in Business Administration."),
        scope=scope, valid_at=t,
    ))

    blocks = engine.retriever.assemble_context(
        "What degree did I graduate with?", [], at=t + 10, scope=scope)
    audit = next(b for b in blocks if b.startswith("User evidence audit"))

    assert "Business Administration" in audit
    assert audit.index("Business Administration") < 220


def test_user_query_terms_expand_e_ending_past_tense():
    terms = _user_query_terms("What degree did I graduate with?")
    assert "graduated" in terms
    assert "graduateing" not in terms


def test_user_evidence_completion_adds_missed_source_candidate(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, user_evidence_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="user-candidate")
    t = datetime(2024, 1, 1, 12).timestamp()
    other = MemoryRecord(
        memory_id="other", content_hash="other", text="assistant: unrelated project note",
        scope=scope, valid_at=t, entities=["project"])
    hit = MemoryRecord(
        memory_id="hit", content_hash="hit",
        text=("assistant: Any books lately?\n"
              "user: I finished reading The Glass Menagerie last night."),
        scope=scope, valid_at=t, entities=["The Glass Menagerie"])
    records = {other.memory_id: other, hit.memory_id: hit}

    ensured = engine.retriever._ensure_user_candidates(
        "What book did I read?",
        [RetrievalCandidate(record=other, fused_score=1.0)],
        records,
        t + 10,
    )
    assert [c.record.memory_id for c in ensured] == ["other", "hit"]


def test_assistant_evidence_context_surfaces_matching_assistant_turn(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, assistant_evidence_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="assistant-context")
    t = datetime(2024, 1, 1, 12).timestamp()
    engine.store.upsert_record(MemoryRecord(
        memory_id="playlist", content_hash="playlist",
        text=("user: Can you make me some music for the trip?\n"
              "assistant: I created a Spotify playlist called Summer Vibes for the trip."),
        scope=scope, valid_at=t, entities=["Spotify", "Summer Vibes"],
    ))
    engine.store.upsert_record(MemoryRecord(
        memory_id="unrelated", content_hash="unrelated",
        text="user: Any warmup tips?\nassistant: I suggested calf stretches.",
        scope=scope, valid_at=t, entities=["stretches"],
    ))

    blocks = engine.retriever.assemble_context(
        "What playlist did the assistant create?", [], at=t + 10, scope=scope)
    joined = "\n".join(blocks)
    assert "Assistant evidence audit" in joined
    assert "Summer Vibes" in joined
    assert "calf stretches" not in joined


def test_assistant_evidence_completion_adds_missed_source_candidate(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, assistant_evidence_context_enabled=True)
    engine = Engine(settings)
    scope = Scope(namespace="assistant-candidate")
    t = datetime(2024, 1, 1, 12).timestamp()
    other = MemoryRecord(
        memory_id="other", content_hash="other", text="user: unrelated project note",
        scope=scope, valid_at=t, entities=["project"])
    hit = MemoryRecord(
        memory_id="hit", content_hash="hit",
        text=("user: Can you make me some music for the trip?\n"
              "assistant: I created a Spotify playlist called Summer Vibes for the trip."),
        scope=scope, valid_at=t, entities=["Spotify", "Summer Vibes"])
    records = {other.memory_id: other, hit.memory_id: hit}

    ensured = engine.retriever._ensure_assistant_candidates(
        "What playlist did the assistant create?",
        [RetrievalCandidate(record=other, fused_score=1.0)],
        records,
        t + 10,
    )
    assert [c.record.memory_id for c in ensured] == ["other", "hit"]


def test_aggregation_audit_completes_amount_sources(monkeypatch, tmp_path):
    from eidetic.config import get_settings
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, NLILabel, RetrievalCandidate, Scope

    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("AGGREGATION_AUDIT", "1")
    get_settings.cache_clear()
    engine = Engine(get_settings())
    ref = datetime(2023, 6, 6, 12, 0, 0).timestamp()
    scope = Scope(namespace="agg-audit")

    def rec(mid: str, text: str, day: int) -> MemoryRecord:
        return MemoryRecord(
            memory_id=mid, content_hash=mid, text=text, source=mid, scope=scope,
            valid_at=datetime(2023, 5, day, 12, 0, 0).timestamp(),
        )

    espresso = rec(
        "espresso",
        "user: I recently bought a premium espresso machine for my kitchen. It was a big purchase, $800. "
        "assistant: Example budget: Entertainment: $800; Food: $500.",
        24,
    )
    bike = rec(
        "bike",
        "user: I splurge on premium gear, like that carbon road bike I just got from Velora for $1,200.",
        25,
    )
    shoes = rec(
        "shoes",
        "user: I recently bought a pack of plain socks from Uniqlo for $20, which is a steal. "
        "But I've also made some premium purchases, like a pair of trail shoes from a "
        "high-end Japanese maker that I got for $500.",
        29,
    )
    old_scope = MemoryRecord(
        memory_id="old-telescope", content_hash="old-telescope",
        text="user: I bought a premium telescope for $900 years ago.", source="old-telescope",
        scope=scope, valid_at=datetime(2022, 1, 1, 12, 0, 0).timestamp(),
    )
    q = "What is the combined total I spent on premium gear in the past few months?"
    parsed = parse_query(q, ref)
    current = [RetrievalCandidate(record=shoes)]
    records = {r.memory_id: r for r in [espresso, bike, shoes, old_scope]}

    ensured = engine.retriever._ensure_aggregation_candidates(q, parsed, current, records, ref)
    assert {c.record.memory_id for c in ensured} == {"espresso", "bike", "shoes"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Aggregation evidence audit"))
    assert "premium espresso machine" in audit and "$800" in audit
    assert "carbon road bike" in audit and "$1,200" in audit
    assert "trail shoes" in audit and "$500" in audit
    assert "premium telescope" not in audit
    assert "Entertainment: $800" not in audit

    engine.retriever.verify_citation = lambda _rec, _text: (NLILabel.NEUTRAL, 0.0)
    citations, entailed = engine.retriever._verify_candidates(
        ensured, "$1,200 + $500 + $800 = $2,500", True, query=q, at=ref)
    assert entailed == 3
    assert {c.memory_id for c in citations if c.nli_label == NLILabel.ENTAILMENT} == {
        "espresso", "bike", "shoes"
    }
    bad_citations, bad_entailed = engine.retriever._verify_candidates(
        ensured, "$500 + $1,200 = $1,700", True, query=q, at=ref)
    assert bad_entailed == 0
    assert all(c.nli_label == NLILabel.NEUTRAL for c in bad_citations)
    get_settings.cache_clear()


def test_list_audit_completes_exact_scope_activity_sources(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, list_audit_enabled=True, list_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="list-audit-destress")
    ref = datetime(2023, 6, 6, 12, 0, 0).timestamp()

    def rec(mid: str, text: str, day: int) -> MemoryRecord:
        item = MemoryRecord(
            memory_id=mid, content_hash=mid, text=text, source=mid, scope=scope,
            valid_at=datetime(2023, 5, day, 12, 0, 0).timestamp(),
        )
        engine.store.upsert_record(item)
        return item

    running = rec("running", "Melanie said running helps her destress after work.", 10)
    pottery = rec("pottery", "Melanie uses pottery class to unwind when she is stressed.", 12)
    violin = rec("violin", "Melanie plays violin as a hobby and enjoys reading novels.", 14)
    q = "What does Melanie do to destress?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_list_candidates(
        q, parsed, [RetrievalCandidate(record=violin, fused_score=1.0)],
        {r.memory_id: r for r in [running, pottery, violin]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"violin", "running", "pottery"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("List evidence audit"))
    assert "running helps her destress" in audit
    assert "pottery class to unwind" in audit
    assert "plays violin" not in audit


def test_list_audit_surfaces_book_titles_not_generic_reading(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, list_audit_enabled=True, list_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="list-audit-books")
    ref = datetime(2023, 6, 6, 12, 0, 0).timestamp()

    def rec(mid: str, text: str, day: int) -> MemoryRecord:
        item = MemoryRecord(
            memory_id=mid, content_hash=mid, text=text, source=mid, scope=scope,
            valid_at=datetime(2023, 5, day, 12, 0, 0).timestamp(),
        )
        engine.store.upsert_record(item)
        return item

    nothing = rec("nothing", "Melanie read Nothing is Impossible in 2022.", 10)
    charlotte = rec("charlotte", "Melanie finished Charlotte's Web last spring.", 12)
    generic = rec("generic", "Melanie enjoys reading novels to relax.", 14)
    q = "What books has Melanie read?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_list_candidates(
        q, parsed, [RetrievalCandidate(record=generic, fused_score=1.0)],
        {r.memory_id: r for r in [nothing, charlotte, generic]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"generic", "nothing", "charlotte"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("List evidence audit"))
    assert "Nothing is Impossible" in audit
    assert "Charlotte's Web" in audit
    assert "enjoys reading novels" not in audit


def test_list_audit_excludes_related_but_off_scope_events(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, list_audit_enabled=True, list_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="list-audit-children")
    ref = datetime(2023, 6, 20, 12, 0, 0).timestamp()

    def rec(mid: str, text: str, day: int) -> MemoryRecord:
        item = MemoryRecord(
            memory_id=mid, content_hash=mid, text=text, source=mid, scope=scope,
            valid_at=datetime(2023, 6, day, 12, 0, 0).timestamp(),
        )
        engine.store.upsert_record(item)
        return item

    mentor = rec("mentor", "Priya joined a tutoring program to help refugees with language skills.", 4)
    speech = rec("speech", "Priya gave a lecture at a community center to encourage refugees.", 8)
    school = rec("school", "Priya attended a community center board meeting with volunteers.", 10)
    pride = rec("pride", "Priya participated in a game night for local adults.", 12)
    q = "What events has Priya participated in to help refugees?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_list_candidates(
        q, parsed, [RetrievalCandidate(record=pride, fused_score=1.0)],
        {r.memory_id: r for r in [mentor, speech, school, pride]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"pride", "mentor", "speech"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("List evidence audit"))
    assert "tutoring program to help refugees" in audit
    assert "lecture at a community center" in audit
    assert "board meeting" not in audit
    assert "game night" not in audit


def test_temporal_evidence_audit_preserves_relative_date_source(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, temporal_evidence_audit_enabled=True, temporal_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="temporal-audit-relative")
    ref = datetime(2023, 6, 1, 12, 0, 0).timestamp()

    def rec(mid: str, text: str, day: int) -> MemoryRecord:
        item = MemoryRecord(
            memory_id=mid, content_hash=mid, text=text, source=mid, scope=scope,
            valid_at=datetime(2023, 5, day, 12, 0, 0).timestamp(),
        )
        engine.store.upsert_record(item)
        return item

    charity = rec(
        "charity",
        "Melanie ran a charity race for mental health the Sunday before 25 May 2023.",
        25,
    )
    saturday = rec(
        "saturday",
        "Melanie ran practice laps every Saturday before breakfast.",
        27,
    )
    q = "When did Melanie run a charity race?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_temporal_evidence_candidates(
        q, parsed, [RetrievalCandidate(record=saturday, fused_score=1.0)],
        {r.memory_id: r for r in [charity, saturday]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"saturday", "charity"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Temporal evidence audit"))
    assert "Sunday before 25 May 2023" in audit
    assert "practice laps every Saturday" not in audit


def test_temporal_evidence_audit_matches_speech_to_talk(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, temporal_evidence_audit_enabled=True, temporal_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="temporal-audit-speech-talk")
    ref = datetime(2023, 6, 9, 12, 0, 0).timestamp()
    hit = MemoryRecord(
        memory_id="school-talk", content_hash="school-talk",
        text="Noor gave a talk about her documentary project at a library event last week.",
        source="school-talk", scope=scope, valid_at=ref,
    )
    distractor = MemoryRecord(
        memory_id="school-board", content_hash="school-board",
        text="Noor attended a library board meeting today.",
        source="school-board", scope=scope, valid_at=ref,
    )
    for item in [hit, distractor]:
        engine.store.upsert_record(item)
    q = "When did Noor give a speech at a library?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_temporal_evidence_candidates(
        q, parsed, [RetrievalCandidate(record=distractor, fused_score=1.0)],
        {r.memory_id: r for r in [hit, distractor]},
        ref,
    )

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Temporal evidence audit"))
    assert "gave a talk" in audit
    assert "library board meeting" not in audit


def test_temporal_evidence_audit_surfaces_year_only_event_source(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, Scope

    settings = replace(fresh_settings, temporal_evidence_audit_enabled=True, temporal_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="temporal-audit-year")
    ref = datetime(2023, 6, 1, 12, 0, 0).timestamp()
    hit = MemoryRecord(
        memory_id="sunrise", content_hash="sunrise",
        text="Melanie painted a sunrise landscape in 2022 before switching to horses.",
        source="sunrise", scope=scope, valid_at=ref,
    )
    distractor = MemoryRecord(
        memory_id="sunset", content_hash="sunset",
        text="Caroline painted sunsets in 2022.",
        source="sunset", scope=scope, valid_at=ref,
    )
    engine.store.upsert_record(hit)
    engine.store.upsert_record(distractor)

    blocks = engine.retriever.assemble_context(
        "When did Melanie paint a sunrise?", [], at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Temporal evidence audit"))
    assert "painted a sunrise landscape in 2022" in audit
    assert "Caroline painted sunsets" not in audit


def test_temporal_evidence_audit_surfaces_dated_book_title_source(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, temporal_evidence_audit_enabled=True, temporal_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="temporal-audit-book-title")
    ref = datetime(2023, 6, 1, 12, 0, 0).timestamp()
    hit = MemoryRecord(
        memory_id="nothing", content_hash="nothing",
        text='Melanie read Nothing is Impossible in 2022 and found it inspiring.',
        source="nothing", scope=scope, valid_at=ref,
    )
    distractor = MemoryRecord(
        memory_id="charlotte", content_hash="charlotte",
        text="Melanie read Charlotte's Web in 2022.",
        source="charlotte", scope=scope, valid_at=ref,
    )
    engine.store.upsert_record(hit)
    engine.store.upsert_record(distractor)
    q = "When did Melanie read Nothing is Impossible?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_temporal_evidence_candidates(
        q, parsed, [RetrievalCandidate(record=distractor, fused_score=1.0)],
        {r.memory_id: r for r in [hit, distractor]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"charlotte", "nothing"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Temporal evidence audit"))
    assert "Nothing is Impossible in 2022" in audit
    assert "Charlotte's Web" not in audit


def test_temporal_evidence_audit_surfaces_duration_without_calendar_date(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, temporal_evidence_audit_enabled=True, temporal_evidence_topk=5)
    engine = Engine(settings)
    scope = Scope(namespace="temporal-audit-duration")
    ref = datetime(2023, 6, 1, 12, 0, 0).timestamp()

    duration = MemoryRecord(
        memory_id="duration", content_hash="duration",
        text="Wei has had her current circle of teammates for four years.",
        source="duration", scope=scope, valid_at=ref,
    )
    distractor = MemoryRecord(
        memory_id="picnic", content_hash="picnic",
        text="Wei met her teammates for a picnic and brought snacks.",
        source="picnic", scope=scope, valid_at=ref,
    )
    engine.store.upsert_record(duration)
    engine.store.upsert_record(distractor)
    q = "How long has Wei had her current circle of teammates for?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_temporal_evidence_candidates(
        q, parsed, [RetrievalCandidate(record=distractor, fused_score=1.0)],
        {r.memory_id: r for r in [duration, distractor]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"picnic", "duration"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Temporal evidence audit"))
    assert "current circle of teammates for four years" in audit
    assert "teammates for a picnic" not in audit


def test_temporal_anchor_audit_uses_session_dates_for_between_question(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, temporal_evidence_audit_enabled=True, temporal_evidence_topk=6)
    engine = Engine(settings)
    scope = Scope(namespace="temporal-anchor-between")
    ref = datetime(2023, 2, 20, 12, 0, 0).timestamp()

    science = MemoryRecord(
        memory_id="science", content_hash="science",
        text="I visited the Harborview Science Center and loved the planetarium wing.",
        source="science", scope=scope, valid_at=datetime(2023, 2, 3, 12, 0, 0).timestamp(),
    )
    botany = MemoryRecord(
        memory_id="botany", content_hash="botany",
        text="I went to the Desert Botany exhibit at the Riverside Natural History Museum.",
        source="botany", scope=scope, valid_at=datetime(2023, 2, 10, 12, 0, 0).timestamp(),
    )
    unrelated = MemoryRecord(
        memory_id="unrelated", content_hash="unrelated",
        text="I walked through a natural foods store and bought granola.",
        source="unrelated", scope=scope, valid_at=datetime(2023, 2, 4, 12, 0, 0).timestamp(),
    )
    for item in [science, botany, unrelated]:
        engine.store.upsert_record(item)
    q = (
        "How many days passed between my trip to the Harborview Science Center and the "
        "'Desert Botany' exhibit at the Riverside Natural History Museum?"
    )
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_temporal_evidence_candidates(
        q, parsed, [RetrievalCandidate(record=unrelated, fused_score=1.0)],
        {r.memory_id: r for r in [science, botany, unrelated]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"unrelated", "science", "botany"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Temporal anchor audit"))
    assert "[2023-02-03]" in audit
    assert "[2023-02-10]" in audit
    assert "Harborview Science Center" in audit
    assert "Desert Botany exhibit" in audit
    assert "natural foods store" not in audit


def test_temporal_anchor_audit_filters_single_word_pair_distractors(fresh_settings):
    from eidetic.engine import Engine
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    settings = replace(fresh_settings, temporal_evidence_audit_enabled=True, temporal_evidence_topk=6)
    engine = Engine(settings)
    scope = Scope(namespace="temporal-anchor-first")
    ref = datetime(2023, 7, 20, 12, 0, 0).timestamp()

    gala = MemoryRecord(
        memory_id="gala", content_hash="gala",
        text="I participated in the winter book fair and helped with the vendor table.",
        source="gala", scope=scope, valid_at=datetime(2023, 7, 5, 12, 0, 0).timestamp(),
    )
    bake = MemoryRecord(
        memory_id="bake", content_hash="bake",
        text="I joined the winter book swap and boxed paperbacks for guests.",
        source="bake", scope=scope, valid_at=datetime(2023, 7, 12, 12, 0, 0).timestamp(),
    )
    run = MemoryRecord(
        memory_id="run", content_hash="run",
        text="I ran a winter fun run for road safety.",
        source="run", scope=scope, valid_at=datetime(2023, 7, 1, 12, 0, 0).timestamp(),
    )
    for item in [gala, bake, run]:
        engine.store.upsert_record(item)
    q = "Which event happened first, the winter book fair or the winter book swap?"
    parsed = parse_query(q, ref)

    ensured = engine.retriever._ensure_temporal_evidence_candidates(
        q, parsed, [RetrievalCandidate(record=run, fused_score=1.0)],
        {r.memory_id: r for r in [gala, bake, run]},
        ref,
    )
    assert {c.record.memory_id for c in ensured} == {"run", "gala", "bake"}

    blocks = engine.retriever.assemble_context(q, ensured, at=ref, scope=scope)
    audit = next(b for b in blocks if b.startswith("Temporal anchor audit"))
    assert audit.index("winter book fair") < audit.index("winter book swap")
    assert "fun run" not in audit


def test_temporal_anchor_audit_does_not_fire_for_non_temporal_between_query():
    from eidetic.events import parse_query
    from eidetic.retrieval import _is_temporal_evidence_query, _temporal_anchor_groups

    q = "What is the gap in cost between my premium sneakers and the cheaper backup pair?"
    parsed = parse_query(q)

    assert _temporal_anchor_groups(q) == []
    assert _is_temporal_evidence_query(q, parsed) is False


# ---- sweep plan (coordinate descent, offline) ----
def test_sweep_plan_is_coordinate_descent():
    trials = plan(subset=50, runs=1)
    assert len(trials) == sum(len(vals) for _, vals in STAGES)
    stages = [stage for stage, _ in STAGES]
    assert trials[0]["stage"] == "READER_COT"   # benchmark-visible feature flags first
    assert "CONFLICT_RESOLVER" in stages
    assert "ABSTENTION_THRESHOLD" not in stages
    assert "CASCADE_CONFIDENCE" not in stages
    assert stage_assignment("COMPRESSION_RATIO", "0.75") == {
        "CONTEXT_COMPRESS": "1",
        "COMPRESSION_RATIO": "0.75",
    }


def test_persistent_bm25_matches_legacy_and_filters_scope(tmp_path):
    scope_items = [
        ("a", "Alice works at Acme"),
        ("b", "Alice works at Globex"),
    ]
    items = [
        *scope_items,
        ("c", "Bob likes tea"),
        ("noise", "Alice Alice Alice works nowhere"),
    ]
    legacy = BM25().index(scope_items).search("Alice works", 10)
    idx = PersistentBM25(tmp_path / "bm25.json")
    idx.index(items)
    idx.save()
    loaded = PersistentBM25(tmp_path / "bm25.json")
    assert loaded.search("Alice works", 10, allowed_ids={"a", "b"}) == legacy
    filtered = loaded.search("Alice works", 10, allowed_ids={"b"})
    assert [mid for mid, _ in filtered] == ["b"]
    assert filtered[0][1] > 0.0


def test_index_timing_harness_reports_p95(tmp_path):
    res = run_timing(doc_count=40, query_count=5, index_path=tmp_path / "bm25.json")
    assert res["doc_count"] == 40
    assert res["legacy_p95_ms"] >= 0.0
    assert res["persistent_p95_ms"] >= 0.0


def test_numpy_vector_search_allowed_ids_prevents_underfill(tmp_path):
    import numpy as np

    from eidetic.vector_index import NumpyVectorIndex

    idx = NumpyVectorIndex(tmp_path, 4, 2)
    idx.add("outside", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    idx.add("inside", np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32))
    hits = idx.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 1, allowed_ids={"inside"})
    assert hits[0][0] == "inside"


def test_retrieve_does_not_pad_zero_score_records(fresh_settings):
    from dataclasses import replace
    import numpy as np

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    settings = replace(fresh_settings, rerank_enabled=False)

    class EmptyIndex:
        def __len__(self):
            return 10

        def search(self, _qvec, _k, allowed_ids=None):
            return []

    class FakeClient:
        pass

    scope = Scope(namespace="pad")
    store = RecordStore(settings.sqlite_path)
    for i in range(5):
        store.upsert_record(MemoryRecord(
            memory_id=f"m{i}", content_hash=f"h{i}", text=f"unmatched {i}",
            scope=scope, valid_at=float(i),
        ))
    retriever = Retriever(store, EmptyIndex(), KnowledgeGraph(store), object(), FakeClient(), settings)
    assert retriever.retrieve("no lexical overlap", scope=scope,
                              qvec=np.ones(4, dtype=np.float32), use_recency=False) == []
    with_recency = retriever.retrieve("no lexical overlap", scope=scope,
                                      qvec=np.ones(4, dtype=np.float32), use_recency=True)
    assert 0 < len(with_recency) <= settings.final_topk


def test_context_compress_master_flag_controls_assembly(fresh_settings):
    from dataclasses import replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, RetrievalCandidate
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    text = "The cat sat. Revenue rose to 9M in Q2. Birds migrate south."
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text=text, valid_at=1.0)
    cand = RetrievalCandidate(record=rec, fused_score=1.0)
    store = RecordStore(fresh_settings.sqlite_path)

    off = replace(fresh_settings, context_compress_enabled=False, compression_ratio=0.34)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), off)
    assert "The cat sat" in " ".join(retriever.assemble_context("revenue Q2", [cand]))

    on = replace(fresh_settings, context_compress_enabled=True, compression_ratio=0.34)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), on)
    joined = " ".join(retriever.assemble_context("revenue Q2", [cand]))
    assert "Revenue rose" in joined and "The cat sat" not in joined


def test_temporal_rerank_orders_latest_context_by_timestamp(fresh_settings):
    from dataclasses import replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, RetrievalCandidate
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    old = RetrievalCandidate(
        record=MemoryRecord(memory_id="old", content_hash="old", text="Alice works at Acme.",
                            valid_at=10.0),
        fused_score=10.0,
    )
    new = RetrievalCandidate(
        record=MemoryRecord(memory_id="new", content_hash="new", text="Alice works at Globex.",
                            valid_at=20.0),
        fused_score=1.0,
    )
    store = RecordStore(fresh_settings.sqlite_path)

    off = replace(fresh_settings, temporal_rerank_enabled=False)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), off)
    default_joined = "\n".join(retriever.assemble_context("What is Alice's latest job?", [old, new]))
    assert default_joined.index("Acme") < default_joined.index("Globex")

    on = replace(fresh_settings, temporal_rerank_enabled=True)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), on)
    temporal_joined = "\n".join(retriever.assemble_context("What is Alice's latest job?", [old, new]))
    assert temporal_joined.index("Globex") < temporal_joined.index("Acme")


def test_temporal_rerank_does_not_treat_last_week_as_latest(fresh_settings):
    from eidetic.events import parse_query
    from eidetic.models import MemoryRecord, RetrievalCandidate

    newer_low_score = RetrievalCandidate(
        record=MemoryRecord(memory_id="new", content_hash="new", text="new", valid_at=20.0),
        fused_score=1.0,
    )
    older_high_score = RetrievalCandidate(
        record=MemoryRecord(memory_id="old", content_hash="old", text="old", valid_at=10.0),
        fused_score=10.0,
    )
    parsed = parse_query("What did Alice do last week?", reference_time=30.0)
    ordered = _temporal_context_order(
        "What did Alice do last week?", parsed, [newer_low_score, older_high_score]
    )
    assert [c.record.memory_id for c in ordered] == ["old", "new"]


def test_hippo2_query_to_triple_seeding_is_scoped_and_relation_aware(tmp_path):
    from eidetic.events import parse_query
    from eidetic.graph import KnowledgeGraph
    from eidetic.models import Scope
    from eidetic.store import RecordStore

    store = RecordStore(tmp_path / "db.sqlite")
    graph = KnowledgeGraph(store)
    alpha = Scope(namespace="alpha")
    beta = Scope(namespace="beta")
    graph.add_fact("Alice", "works_at", "Acme", valid_at=10.0, scope=alpha)
    graph.add_fact("Alice", "lives_in", "Paris", valid_at=10.0, scope=alpha)
    graph.add_fact("Alice", "works_at", "BetaCorp", valid_at=10.0, scope=beta)

    parsed = parse_query("Where does Alice work now?", reference_time=20.0)
    seeds = _hippo2_seed_entities("Where does Alice work now?", parsed, store, 20.0, alpha)
    assert "Alice" in seeds and "Acme" in seeds
    assert "Paris" not in seeds
    assert "BetaCorp" not in seeds


def test_reader_router_flag_can_pin_product_reader(fresh_settings):
    from dataclasses import replace

    routed = replace(fresh_settings, reader_router_enabled=True,
                     salience_model="qwen-flash", gen_model="qwen3-max")
    pinned = replace(fresh_settings, reader_router_enabled=False,
                     salience_model="qwen-flash", gen_model="qwen3-max")
    assert _reader_model("What is Alice's favorite tea?", routed) == "qwen-flash"
    assert _reader_model("What is Alice's favorite tea?", pinned) == "qwen3-max"


def test_extract_light_uses_salience_model(fresh_settings):
    from dataclasses import replace

    from eidetic.dashscope_client import DashScopeClient

    settings = replace(fresh_settings, extract_light_enabled=True,
                       salience_model="qwen-flash", extract_model="qwen-plus")
    client = DashScopeClient.__new__(DashScopeClient)
    client.settings = settings
    seen = {}

    def fake_chat(model, *_args, **_kw):
        # extraction now calls chat() (raw string) + a truncation-resilient parser, not chat_json.
        seen["model"] = model
        return '{"triples": [{"src": "Alice", "relation": "likes", "dst": "tea"}]}'

    client.chat = fake_chat
    triples = client.extract_edges("Alice likes tea.")
    assert seen["model"] == "qwen-flash"
    assert triples[0]["src"] == "Alice"


def test_consolidation_caps_extraction_windows_for_huge_batches(fresh_settings):
    from dataclasses import replace

    import numpy as np

    from eidetic.engine import Engine
    from eidetic.models import Scope

    class BoundedClient:
        def __init__(self, dim):
            self.dim = dim
            self.calls: list[int] = []

        def embed_text(self, _text):
            return np.ones(self.dim, np.float32)

        def embed_texts(self, texts):
            return np.ones((len(texts), self.dim), np.float32)

        def extract_edges_bounded(self, text, *, max_windows=0):
            self.calls.append(max_windows)
            return [{"src": "Alice", "relation": "mentioned", "dst": "project", "fact": "Alice mentioned project"}]

    settings = replace(
        fresh_settings,
        extract_chunking_enabled=True,
        extract_chunk_chars=1000,
        extract_chunk_overlap=0,
        consolidation_extract_call_budget=3,
        consolidation_extract_deadline_sec=0,
        memory_typing_enabled=False,
        pref_sentence_scan_enabled=False,
    )
    client = BoundedClient(settings.embed_dim)
    engine = Engine(settings, client=client)
    scope = Scope(namespace="bounded-consolidation")
    for i in range(5):
        long_text = f"user: Alice mentioned project details {i}.\n" + ("x " * 4000)
        engine.ingest_text(long_text, source=f"s{i}", scope=scope, consolidate_now=False)

    report = engine.consolidate_pending(scope=scope, score_importance=False)

    assert report["extraction_windows_planned"] > report["extraction_windows_submitted"]
    assert report["extraction_window_cap_per_record"] == 1
    assert report["extraction_call_budget"] == 3
    assert report["extraction_raw_only_bounded"] == 2
    assert report["extraction_partial_bounded"] == 3
    assert report["extraction_timed_out"] == 0
    assert client.calls == [1, 1, 1]


def test_long_haystack_raw_only_submits_no_extraction_windows(fresh_settings):
    from dataclasses import replace

    import numpy as np

    from eidetic.engine import Engine
    from eidetic.models import Scope

    class BoundedClient:
        def __init__(self, dim):
            self.dim = dim
            self.calls: list[int] = []

        def embed_text(self, _text):
            return np.ones(self.dim, np.float32)

        def embed_texts(self, texts):
            return np.ones((len(texts), self.dim), np.float32)

        def extract_edges_bounded(self, text, *, max_windows=0):
            self.calls.append(max_windows)
            return [{"src": "Alice", "relation": "mentioned", "dst": "project", "fact": "Alice mentioned project"}]

        def extract_edges(self, text):
            self.calls.append(-1)
            return []

    settings = replace(
        fresh_settings,
        extract_chunking_enabled=True,
        extract_chunk_chars=1000,
        extract_chunk_overlap=0,
        consolidation_extract_call_budget=3,
        consolidation_long_haystack_raw_only=True,
        consolidation_extract_deadline_sec=1,
        memory_typing_enabled=False,
        pref_sentence_scan_enabled=False,
    )
    client = BoundedClient(settings.embed_dim)
    engine = Engine(settings, client=client)
    scope = Scope(namespace="raw-only-bounded")
    for i in range(5):
        engine.ingest_text(f"user: raw searchable {i}.\n" + ("x " * 4000),
                           source=f"s{i}", scope=scope, consolidate_now=False)

    report = engine.consolidate_pending(scope=scope, score_importance=False)

    # DRAIN contract (extraction-audit fleet 2026-07-09): the old raw-only mode submitted ZERO
    # windows for the whole batch and cleared pending unconditionally -- the graph channel died
    # PERMANENTLY for the namespace (proven on 7 live LME-S namespaces). Now each sleep spends
    # up to the budget at one window per record and DEFERS the rest, which stay pending.
    assert report["long_haystack_bounded"] is True
    assert report["long_haystack_raw_only"] is True
    assert report["extraction_windows_submitted"] == 3          # budget-bounded slice
    assert report["extraction_raw_only_bounded"] == 2           # deferred, still pending
    assert report["extraction_timed_out"] == 0
    assert report["pending_processed"] == 5
    assert client.calls == [1, 1, 1]                            # one window each, never more

    # the deferred records drain on the NEXT sleep -- starvation is no longer permanent
    report2 = engine.consolidate_pending(scope=scope, score_importance=False)
    assert report2["pending_processed"] == 2
    assert len(client.calls) == 5
    report3 = engine.consolidate_pending(scope=scope, score_importance=False)
    assert report3["pending_processed"] == 0                    # fully drained, no re-work


def test_near_budget_long_haystack_raw_only_prevents_saturation(fresh_settings):
    from dataclasses import replace

    import numpy as np

    from eidetic.engine import Engine
    from eidetic.models import Scope

    class BoundedClient:
        def __init__(self, dim):
            self.dim = dim
            self.calls: list[str] = []

        def embed_text(self, _text):
            return np.ones(self.dim, np.float32)

        def embed_texts(self, texts):
            return np.ones((len(texts), self.dim), np.float32)

        def extract_edges(self, text):
            self.calls.append(text)
            return []

    settings = replace(
        fresh_settings,
        extract_chunking_enabled=False,
        consolidation_extract_call_budget=10,
        consolidation_long_haystack_raw_only=True,
        consolidation_extract_deadline_sec=1,
        memory_typing_enabled=False,
        pref_sentence_scan_enabled=False,
    )
    client = BoundedClient(settings.embed_dim)
    engine = Engine(settings, client=client)
    scope = Scope(namespace="near-budget-raw-only")
    for i in range(9):
        engine.ingest_text(f"user: raw searchable near budget {i}.", source=f"s{i}",
                           scope=scope, consolidate_now=False)

    report = engine.consolidate_pending(scope=scope, score_importance=False)

    assert report["extraction_windows_planned"] == 9
    assert report["extraction_call_budget"] == 10
    assert report["long_haystack_bounded"] is True
    assert report["long_haystack_raw_only"] is True
    # drain contract: near-budget batches now extract within the declared budget ceiling
    # (one window per record) instead of skipping everything -- the budget IS the spend
    # contract; saturation protection = the per-record cap plus deferral, not zero work.
    assert report["extraction_windows_submitted"] == 9
    assert report["extraction_raw_only_bounded"] == 0
    assert report["extraction_timed_out"] == 0
    assert len(client.calls) == 9


def test_single_giant_record_raw_only_does_not_starve_normal_records(fresh_settings):
    import numpy as np

    from eidetic.engine import Engine
    from eidetic.models import Scope

    class BoundedClient:
        def __init__(self, dim):
            self.dim = dim
            self.calls: list[str] = []

        def embed_text(self, _text):
            return np.ones(self.dim, np.float32)

        def embed_texts(self, texts):
            return np.ones((len(texts), self.dim), np.float32)

        def extract_edges(self, text):
            self.calls.append(text)
            if "normal" in text:
                return [{
                    "src": "Nora",
                    "relation": "likes",
                    "dst": "tea",
                    "fact": "Nora likes tea",
                }]
            raise AssertionError("giant record should not enter extraction")

    settings = replace(
        fresh_settings,
        extract_chunking_enabled=True,
        extract_chunk_chars=1000,
        extract_chunk_overlap=0,
        consolidation_extract_call_budget=0,
        consolidation_raw_only_window_threshold=3,
        consolidation_extract_deadline_sec=1,
        memory_typing_enabled=False,
        pref_sentence_scan_enabled=False,
    )
    client = BoundedClient(settings.embed_dim)
    engine = Engine(settings, client=client)
    scope = Scope(namespace="single-giant-raw-only")
    engine.ingest_text("user: normal record Nora likes tea.", source="normal",
                       scope=scope, consolidate_now=False)
    engine.ingest_text("user: giant searchable transcript.\n" + ("haystack " * 5000),
                       source="giant", scope=scope, consolidate_now=False)

    report = engine.consolidate_pending(scope=scope, score_importance=False)

    assert report["long_haystack_bounded"] is True
    assert report["long_haystack_raw_only"] is False
    assert report["record_raw_only_bounded"] == 1
    assert report["extraction_raw_only_bounded"] == 1
    assert report["extraction_windows_submitted"] == 1
    assert report["extraction_timed_out"] == 0
    assert len(client.calls) == 1 and "normal record" in client.calls[0]
    assert any(edge.src == "Nora" for edge in engine.store.all_edges(scope))
    giant = [rec for rec in engine.store.all_records(scope) if rec.source == "giant"][0]
    assert giant.metadata["pending_consolidation"] is False
    assert giant.metadata["consolidation_raw_only"] == "record_window_threshold"
    assert giant.metadata["consolidation_raw_only_window_threshold"] == 3


def test_rerank_failure_is_loud_unless_fail_open(fresh_settings):
    from dataclasses import replace
    import numpy as np

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    scope = Scope(namespace="rerank")
    store = RecordStore(fresh_settings.sqlite_path)
    store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="h1", text="Alice likes green tea.",
        scope=scope, valid_at=1.0,
    ))

    class OneIndex:
        def __len__(self):
            return 1

        def search(self, _qvec, _k, allowed_ids=None):
            return [("m1", 0.9)]

    class BadReranker:
        def rerank(self, *_args, **_kw):
            raise RuntimeError("rerank quota exhausted")

    strict = replace(fresh_settings, rerank_enabled=True, rerank_fail_open=False,
                     rerank_depth=1, final_topk=1)
    retriever = Retriever(store, OneIndex(), KnowledgeGraph(store), object(), BadReranker(), strict)
    with pytest.raises(RuntimeError, match="rerank quota exhausted"):
        retriever.retrieve("Alice tea", at=2.0, scope=scope,
                           qvec=np.ones(4, dtype=np.float32))

    fail_open = replace(strict, rerank_fail_open=True)
    retriever = Retriever(store, OneIndex(), KnowledgeGraph(store), object(), BadReranker(), fail_open)
    hits = retriever.retrieve("Alice tea", at=2.0, scope=scope,
                              qvec=np.ones(4, dtype=np.float32))
    assert [h.record.memory_id for h in hits] == ["m1"]
