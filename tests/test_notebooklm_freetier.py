"""Unit floors for the free-tier collection harness (bench/notebooklm_freetier_run.py).
Synthetic fixtures only. The live run is a separate, labeled artifact -- these tests pin
the deterministic pieces: the prelim heuristic's tolerance and the source packing caps."""
from bench.notebooklm_freetier_run import pack_record_sources, prelim_contains


class _Bridge:
    def __init__(self, sources):
        self._sources = sources

    def build_sources(self, namespace):
        return list(self._sources)


def test_prelim_contains_requires_every_gold_content_token():
    assert prelim_contains("Lisbon in March", "She moved to Lisbon in March 2024") is True
    assert prelim_contains("Lisbon and Porto", "She moved to Lisbon") is False


def test_prelim_contains_is_prefix_tolerant_for_inflection():
    assert prelim_contains("electric and intense",
                           "the arena was electric with extra intensity") is True


def test_prelim_contains_rejects_empty_gold():
    assert prelim_contains("", "anything") is False


def test_pack_respects_source_and_char_caps():
    sources = [{"display_name": f"s{i}", "text_content": "x" * 5_000} for i in range(30)]
    packed = pack_record_sources(_Bridge(sources), "ns-abcdef",
                                 max_sources=10, max_chars=12_000)
    assert len(packed) <= 10
    # nothing dropped silently until the cap: all text present across packed sources
    total = sum(len(p["text_content"]) for p in packed)
    assert total >= 5_000 * 18  # 2 records per 12k-char pack x 9 packs + final pack


def test_find_notebook_id_strict_never_falls_back_to_first_id():
    """The contamination bug: non-strict lookup resolved a MISSING title to the first
    notebook in the listing, so ten namespaces' sources and questions silently landed in
    one notebook. strict=True must return None for a missing title -- forcing a create."""
    from eidetic.integrations.notebooklm import find_notebook_id
    listing = '[{"id":"nb_DEMO_12345","title":"My Memory"}]'
    assert find_notebook_id(listing, "eidetic-win-g3-r0") == "nb_DEMO_12345"  # legacy fallback
    assert find_notebook_id(listing, "eidetic-win-g3-r0", strict=True) is None  # the fix
    assert find_notebook_id(listing, "My Memory", strict=True) == "nb_DEMO_12345"


def test_pack_preserves_record_text_verbatim():
    sources = [{"display_name": "s0",
                "text_content": "--- EIDETIC VERIFIED MEMORY (provenance) ---\nbody text"}]
    packed = pack_record_sources(_Bridge(sources), "ns-abcdef")
    assert len(packed) == 1
    assert "EIDETIC VERIFIED MEMORY" in packed[0]["text_content"]
    assert "body text" in packed[0]["text_content"]
