"""Offline unit tests for the capture-fidelity helpers (no model calls).

Covers the pure functions behind chunked extraction (EXTRACT_CHUNKING) and the sentence-level
preference scan (PREF_SENTENCE_SCAN). The flag-off invariant is asserted at the config layer:
every new capture-fidelity flag defaults OFF so the historical write path is byte-identical.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import replace

import numpy as np

from eidetic.dashscope_client import chunk_text, dedupe_triples
from eidetic.engine import Engine
from eidetic.models import Scope
from eidetic.preferences import (
    canonicalize_preference,
    extract_all_preferences,
    preference_dedup_key,
    preference_polarity,
    preference_update_key,
)
from eidetic.store import RecordStore


class _FakeClient:
    """Offline client: deterministic embeddings, no graph extraction (isolates the scan)."""

    def __init__(self, dim):
        self.dim = dim

    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._e(t)

    def embed_texts(self, ts):
        return np.stack([self._e(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)

    def extract_edges(self, text):
        return []


# ---- consolidation resilience (robustness audit finding 2) ------------------------------

def test_consolidate_survives_one_record_whose_extraction_raises(fresh_settings):
    """One record's extraction blowing up (e.g. a transient past its retry budget) must NOT abort
    the whole sample's consolidation -- it degrades that record to no facts and continues."""
    from eidetic.dashscope_client import ModelCallError

    class _C(_FakeClient):
        def extract_edges(self, text):
            if "BOOM" in text:
                raise ModelCallError("DashScope call failed (HTTP 500): transient past retries")
            return [{"src": "Alice", "relation": "likes", "dst": "tea", "fact": "Alice likes tea"}]

    eng = Engine(fresh_settings, client=_C(fresh_settings.embed_dim))
    sc = Scope(namespace="t")
    eng.ingest_text("Alice likes tea", scope=sc, consolidate_now=False)
    eng.ingest_text("BOOM this record's extraction fails", scope=sc, consolidate_now=False)
    res = eng.consolidate_pending(scope=sc, score_importance=False)   # must NOT raise
    assert isinstance(res, dict)
    # the GOOD record's fact still made it into the graph despite the bad record failing
    assert len(eng.store.all_edges(sc)) >= 1


def test_consolidate_deadline_degrades_slow_record_to_raw_only(fresh_settings):
    class _C(_FakeClient):
        def extract_edges(self, text):
            if "SLOW" in text:
                time.sleep(0.5)
                return [{"src": "Slow", "relation": "likes", "dst": "tea", "fact": "Slow likes tea"}]
            return [{"src": "Fast", "relation": "likes", "dst": "coffee", "fact": "Fast likes coffee"}]

    settings = replace(
        fresh_settings,
        consolidation_extract_deadline_sec=0.05,
        consolidation_timeout_policy="degrade",
    )
    eng = Engine(settings, client=_C(settings.embed_dim))
    sc = Scope(namespace="deadline")
    eng.ingest_text("FAST record", scope=sc, consolidate_now=False)
    eng.ingest_text("SLOW record", scope=sc, consolidate_now=False)

    t0 = time.perf_counter()
    res = eng.consolidate_pending(scope=sc, score_importance=False, max_workers=2)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.35
    assert res["pending_processed"] == 2
    assert res["extraction_timed_out"] == 1
    assert res["extraction_deferred"] == 0
    assert any(e.src == "Fast" for e in eng.store.all_edges(sc))
    recs = eng.store.all_records(sc)
    assert all(not r.metadata.get("pending_consolidation") for r in recs)
    assert any(r.metadata.get("consolidation_timeout") for r in recs if "SLOW" in r.text)


def test_consolidate_deadline_can_defer_slow_record(fresh_settings):
    class _C(_FakeClient):
        def extract_edges(self, text):
            if "SLOW" in text:
                time.sleep(0.5)
                return [{"src": "Slow", "relation": "likes", "dst": "tea", "fact": "Slow likes tea"}]
            return [{"src": "Fast", "relation": "likes", "dst": "coffee", "fact": "Fast likes coffee"}]

    settings = replace(
        fresh_settings,
        consolidation_extract_deadline_sec=0.05,
        consolidation_timeout_policy="defer",
    )
    eng = Engine(settings, client=_C(settings.embed_dim))
    sc = Scope(namespace="defer")
    eng.ingest_text("FAST record", scope=sc, consolidate_now=False)
    eng.ingest_text("SLOW record", scope=sc, consolidate_now=False)

    res = eng.consolidate_pending(scope=sc, score_importance=False, max_workers=2)

    assert res["pending_processed"] == 1
    assert res["extraction_timed_out"] == 1
    assert res["extraction_deferred"] == 1
    recs = eng.store.all_records(sc)
    assert any(r.metadata.get("pending_consolidation") for r in recs if "SLOW" in r.text)
    assert any(e.src == "Fast" for e in eng.store.all_edges(sc))


# ---- chunk_text -------------------------------------------------------------------------

def test_chunk_text_short_returns_single_window():
    assert chunk_text("hello world", 4000, 400) == ["hello world"]


def test_chunk_text_at_boundary_is_single_window():
    text = "x" * 4000
    assert chunk_text(text, 4000, 400) == [text]


def test_chunk_text_covers_all_bytes_with_overlap():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(10000))
    windows = chunk_text(text, 4000, 400)
    # Every window is within the cap.
    assert all(len(w) <= 4000 for w in windows)
    # Step = chunk - overlap = 3600 -> windows at 0, 3600, 7200 -> reach the tail (char >6000).
    assert len(windows) >= 3
    # The last ~hundred chars (well beyond char 6000) appear in some window.
    assert any(text[-100:] in w for w in windows)
    # Consecutive windows overlap by `overlap` chars (capture continuity across the cut).
    assert windows[0][-400:] == windows[1][:400]


def test_chunk_text_degenerate_overlap_falls_back_to_full_coverage():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(9000))
    # overlap >= chunk would be a zero/negative step -> fall back to non-overlapping windows that
    # still cover every byte and never send one untruncated window to the LLM.
    windows = chunk_text(text, 100, 100)
    assert all(len(w) <= 100 for w in windows)
    assert "".join(windows) == text              # full coverage, no bytes dropped
    assert len(windows) == 90


def test_chunk_text_nonpositive_chunk_returns_single_window():
    text = "y" * 9000
    assert chunk_text(text, 0, 0) == [text]      # chunk<=0 guarded before any slicing


# ---- dedupe_triples ---------------------------------------------------------------------

def _t(src, rel, dst, fact=None):
    return {"src": src, "relation": rel, "dst": dst, "fact": fact or f"{src} {rel} {dst}"}


def test_dedupe_triples_case_insensitive_keeps_first():
    triples = [_t("Mel", "read", "Dune"), _t("mel", "READ", "dune"), _t("Mel", "read", "Hyperion")]
    out = dedupe_triples(triples)
    assert len(out) == 2
    assert out[0]["dst"] == "Dune"          # first-seen kept, original casing preserved
    assert out[1]["dst"] == "Hyperion"


def test_dedupe_triples_drops_incomplete():
    out = dedupe_triples([_t("", "read", "Dune"), _t("Mel", "", "Dune"), _t("Mel", "read", "")])
    assert out == []


def test_dedupe_triples_preserves_order():
    triples = [_t("A", "r", "1"), _t("B", "r", "2"), _t("C", "r", "3")]
    assert [t["src"] for t in dedupe_triples(triples)] == ["A", "B", "C"]


# ---- extract_all_preferences ------------------------------------------------------------

def test_extract_all_preferences_multiple_across_turns():
    text = (
        "user: I love hiking on weekends.\n"
        "assistant: Noted.\n"
        "user: I prefer tea over coffee. The weather is nice today.\n"
        "user: I always wake up early."
    )
    prefs = extract_all_preferences(text)
    assert any("hiking" in p for p in prefs)
    assert any("tea over coffee" in p for p in prefs)
    assert any("wake up early" in p for p in prefs)
    # The non-preference sentence ("The weather is nice") is not captured.
    assert not any("weather" in p for p in prefs)


def test_extract_all_preferences_strips_role_prefix():
    prefs = extract_all_preferences("user: I prefer window seats.")
    assert prefs == ["I prefer window seats."]


def test_extract_all_preferences_dedupes():
    text = "user: I love jazz.\nuser: I love jazz."
    assert extract_all_preferences(text) == ["I love jazz."]


def test_extract_all_preferences_splits_same_turn_clauses():
    prefs = extract_all_preferences("user: I love tea; I hate coffee and I always wake up early.")
    assert any("love tea" in p for p in prefs)
    assert any("hate coffee" in p for p in prefs)
    assert any("wake up early" in p for p in prefs)


def test_canonicalize_preference_collapses_paraphrases_but_keeps_negation():
    assert canonicalize_preference("user: I LOVE jazz.") == "User likes jazz."
    assert canonicalize_preference("I enjoy jazz.") == "User likes jazz."
    assert canonicalize_preference("I do not like jazz.") == "User dislikes jazz."
    assert preference_dedup_key("I love jazz.") == preference_dedup_key("I like jazz.")
    assert preference_dedup_key("I love jazz.") != preference_dedup_key("I dislike jazz.")


def test_preference_update_key_groups_outdated_profile_entries():
    assert preference_update_key("I like coffee.") == preference_update_key("I dislike coffee.")
    assert preference_polarity("I like coffee.") == "positive"
    assert preference_polarity("I dislike coffee.") == "negative"
    assert preference_update_key("My favorite music is jazz.") == "favorite:music"
    assert preference_update_key("My favorite music is techno.") == "favorite:music"


def test_extract_all_preferences_empty():
    assert extract_all_preferences("") == []
    assert extract_all_preferences("The sky is blue. Paris is in France.") == []


# ---- flag-off invariant -----------------------------------------------------------------

def test_capture_fidelity_flags_default_off(fresh_settings):
    s = fresh_settings
    assert s.extract_chunking_enabled is False
    assert s.pref_sentence_scan_enabled is False
    assert s.memory_typing_enabled is False


# ---- integration: PREF_SENTENCE_SCAN through the real consolidate path ------------------

_MULTI_PREF = ("user: I love hiking on weekends.\n"
               "assistant: Noted.\n"
               "user: I prefer tea over coffee.")


def test_pref_sentence_scan_on_surfaces_every_preference(fresh_settings):
    s = replace(fresh_settings, pref_sentence_scan_enabled=True)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="cap")
    engine.ingest_text(_MULTI_PREF, source="sess", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    profile = engine.store.get_profile("cap")
    assert any("hiking" in p for p in profile)
    assert any("tea over coffee" in p for p in profile)   # mid-conversation pref no longer lost


def test_pref_sentence_scan_profile_entries_keep_source_provenance(fresh_settings):
    s = replace(fresh_settings, pref_sentence_scan_enabled=True)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="capprov")
    rec = engine.ingest_text("user: I love ginger tea.", source="s1", scope=scope,
                             consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)

    entries = engine.store.get_profile_entries("capprov")

    assert len(entries) == 1
    assert entries[0]["line"] == "User likes ginger tea."
    assert entries[0]["source_memory_id"] == rec.memory_id
    assert entries[0]["content_hash"] == rec.content_hash
    assert entries[0]["raw_uri"] == rec.raw_uri
    assert entries[0]["valid_at"] == rec.valid_at


def test_pref_sentence_scan_dedupes_casing_variants_across_sessions(fresh_settings):
    # The normalized dedup_key collapses casing/whitespace variants so the same stated preference
    # across two sessions yields ONE profile line (not two).
    s = replace(fresh_settings, pref_sentence_scan_enabled=True)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="capdup")
    engine.ingest_text("user: I love jazz.", source="s1", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    engine.ingest_text("user: I   LOVE   jazz.", source="s2", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    jazz = [p for p in engine.store.get_profile("capdup") if "jazz" in p.lower()]
    assert len(jazz) == 1


def test_pref_sentence_scan_canonical_dedupes_paraphrases_across_sessions(fresh_settings):
    s = replace(fresh_settings, pref_sentence_scan_enabled=True)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="capcanon")
    engine.ingest_text("user: I love jazz.", source="s1", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    engine.ingest_text("user: I like jazz.", source="s2", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    assert engine.store.get_profile("capcanon") == ["User likes jazz."]


def test_pref_sentence_scan_invalidates_outdated_opposite_polarity(fresh_settings):
    s = replace(fresh_settings, pref_sentence_scan_enabled=True)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="cappolarity")
    engine.ingest_text("user: I like coffee.", source="s1", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    engine.ingest_text("user: I dislike coffee.", source="s2", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    profile = engine.store.get_profile("cappolarity")
    history = engine.store.get_profile("cappolarity", include_inactive=True)
    assert "User likes coffee." not in profile
    assert "User dislikes coffee." in profile
    assert "User likes coffee." in history and "User dislikes coffee." in history


def test_pref_sentence_scan_invalidates_outdated_favorite_slot(fresh_settings):
    s = replace(fresh_settings, pref_sentence_scan_enabled=True)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="capfavorite")
    engine.ingest_text("user: My favorite music is jazz.", source="s1", scope=scope,
                       consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    engine.ingest_text("user: My favorite music is techno.", source="s2", scope=scope,
                       consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)

    profile = engine.store.get_profile("capfavorite")
    history = engine.store.get_profile("capfavorite", include_inactive=True)

    assert "User's favorite music is techno." in profile
    assert "User's favorite music is jazz." not in profile
    assert "User's favorite music is jazz." in history


def test_profile_supersession_closes_old_preference_at_source_valid_time(tmp_path):
    store = RecordStore(tmp_path / "profiles.sqlite")

    store.add_profile_line(
        "prefs-time",
        "User's favorite music is jazz.",
        valid_at=100.0,
        dedup_key=preference_dedup_key("User's favorite music is jazz."),
    )
    store.add_profile_line(
        "prefs-time",
        "User's favorite music is techno.",
        valid_at=200.0,
        dedup_key=preference_dedup_key("User's favorite music is techno."),
    )

    active = store.get_profile("prefs-time")
    history = {entry["line"]: entry for entry in store.get_profile_entries(
        "prefs-time", include_inactive=True)}

    assert active == ["User's favorite music is techno."]
    assert history["User's favorite music is jazz."]["invalid_at"] == 200.0
    assert history["User's favorite music is techno."]["invalid_at"] is None


def test_profile_supersession_keeps_newer_preference_when_older_is_processed_late(tmp_path):
    store = RecordStore(tmp_path / "profiles-late.sqlite")

    store.add_profile_line(
        "prefs-late",
        "User's favorite music is techno.",
        valid_at=200.0,
        dedup_key=preference_dedup_key("User's favorite music is techno."),
    )
    store.add_profile_line(
        "prefs-late",
        "User's favorite music is jazz.",
        valid_at=100.0,
        dedup_key=preference_dedup_key("User's favorite music is jazz."),
    )

    active = store.get_profile("prefs-late")
    history = {entry["line"]: entry for entry in store.get_profile_entries(
        "prefs-late", include_inactive=True)}

    assert active == ["User's favorite music is techno."]
    assert history["User's favorite music is jazz."]["valid_at"] == 100.0
    assert history["User's favorite music is jazz."]["invalid_at"] == 200.0
    assert history["User's favorite music is techno."]["invalid_at"] is None


def test_pref_sentence_scan_off_keeps_first_only(fresh_settings):
    # Default behavior: one profile line per session (the first preference). Byte-identical path.
    s = replace(fresh_settings, pref_sentence_scan_enabled=False)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="cap2")
    engine.ingest_text(_MULTI_PREF, source="sess", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    profile = engine.store.get_profile("cap2")
    assert len(profile) == 1                               # only the first preference is captured
