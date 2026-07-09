"""Long-record export chunking (LME-S 'no information' fix): a fact buried in a long record
is present but the reader misses it over one big block. format_source_chunks splits long
records into per-chunk sources, each carrying the provenance header, so buried facts surface.
Default off => byte-identical to format_source (LoCoMo + shipped numbers untouched)."""
from eidetic.integrations.notebooklm import format_source, format_source_chunks
from eidetic.models import MemoryRecord, Scope


def _rec(text, mid="mem_chunk_00000001"):
    return MemoryRecord(memory_id=mid, content_hash="c" * 64, raw_uri=f"raw://{mid}",
                        source="test", text=text, summary=text[:40],
                        valid_at=1_700_000_000.0, scope=Scope(namespace="ns"))


def test_default_off_is_identical_to_format_source():
    r = _rec("Priya moved to Lisbon.")
    assert format_source_chunks(r, [], chunk_chars=0) == [format_source(r, [])]


def test_short_record_under_threshold_stays_single():
    r = _rec("Priya moved to Lisbon in March 2024.")
    assert format_source_chunks(r, [], chunk_chars=500) == [format_source(r, [])]


def _long_with_buried_fact():
    return _rec("Write a blog post about engagement rings. " * 30
                + "\n\nThe best designer is @jessica_poole_jewellery in the UK.\n\n"
                + "More filler content follows here. " * 30)


def test_long_record_splits_and_surfaces_the_buried_fact():
    chunks = format_source_chunks(_long_with_buried_fact(), [], chunk_chars=400)
    assert len(chunks) > 1
    assert any("@jessica_poole_jewellery" in c["text_content"] for c in chunks)


def test_every_chunk_carries_provenance_so_citations_round_trip():
    chunks = format_source_chunks(_long_with_buried_fact(), [], chunk_chars=400)
    for c in chunks:
        assert "mem_chunk_000000" in c["text_content"]      # the eidetic:<id> token
        assert "content_sha256" in c["text_content"]        # the immutable hash
        assert c["display_name"].startswith("eidetic:mem_chunk_000000")


def test_chunk_bodies_respect_the_char_budget_roughly():
    chunks = format_source_chunks(_long_with_buried_fact(), [], chunk_chars=400)
    # each chunk's BODY (after the header) should be within a small multiple of the budget
    for c in chunks:
        body = c["text_content"].split("\n\n", 1)[-1]
        assert len(body) <= 400 * 2   # sentence-boundary splitting is approximate, not hard
