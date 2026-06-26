"""Offline unit tests for the capture-fidelity helpers (no model calls).

Covers the pure functions behind chunked extraction (EXTRACT_CHUNKING) and the sentence-level
preference scan (PREF_SENTENCE_SCAN). The flag-off invariant is asserted at the config layer:
every new capture-fidelity flag defaults OFF so the historical write path is byte-identical.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.dashscope_client import chunk_text, dedupe_triples
from eidetic.engine import Engine
from eidetic.models import Scope
from eidetic.preferences import extract_all_preferences


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


def test_pref_sentence_scan_off_keeps_first_only(fresh_settings):
    # Default behavior: one profile line per session (the first preference). Byte-identical path.
    s = replace(fresh_settings, pref_sentence_scan_enabled=False)
    engine = Engine(s, client=_FakeClient(s.embed_dim))
    scope = Scope(namespace="cap2")
    engine.ingest_text(_MULTI_PREF, source="sess", scope=scope, consolidate_now=False)
    engine.consolidate_pending(scope=scope, score_importance=False)
    profile = engine.store.get_profile("cap2")
    assert len(profile) == 1                               # only the first preference is captured
