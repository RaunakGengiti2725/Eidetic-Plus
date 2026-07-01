"""Claim-crystal span demotion: crystallized raw records stop paying full-text context cost
under the priority-forgetting profile, vivid (high-affect-salience) records stay whole, and the
default path is byte-identical with the flag off."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import RetrievalCandidate, Retriever
from eidetic.store import RecordStore
from eidetic.graph import KnowledgeGraph


class _NoopClient:
    def embed_text(self, text):
        return np.zeros(8, np.float32)

    def nli(self, premise, hypothesis):
        return ("neutral", 0.0)


class _NoopReranker:
    def rerank(self, query, candidates, top_k=None):
        return candidates


class _NoopIndex:
    def search(self, vec, k):
        return []

    def get_vectors(self, ids):
        return {}

    def __len__(self):
        return 0


_LONG_FILLER = " ".join(f"filler sentence number {i} about the garden project." for i in range(80))


def _rec(mid: str, *, salience: float, crystallized: bool, scope: Scope) -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid,
        text=f"user: The kiln schedule moved to Tuesday. {_LONG_FILLER}",
        source="s",
        scope=scope,
        valid_at=1_700_000_000.0,
        salience=salience,
        metadata={"claims_extracted": 40} if crystallized else {},
    )


def _retriever(settings, records) -> tuple[Retriever, list[RetrievalCandidate]]:
    store = RecordStore(settings.data_dir / "t.sqlite")
    for rec in records:
        store.upsert_record(rec)
    r = Retriever(store, _NoopIndex(), KnowledgeGraph(store), _NoopClient(), _NoopReranker(), settings)
    cands = [RetrievalCandidate(record=rec, dense_score=0.9, fused_score=0.9) for rec in records]
    return r, cands


def _raw_block_chars(blocks: list[str]) -> int:
    return sum(len(b) for b in blocks if "kiln schedule" in b)


def test_demotion_shrinks_crystallized_low_salience_context(fresh_settings):
    scope = Scope(namespace="crystal")
    settings = replace(
        fresh_settings,
        crystal_span_demotion_enabled=True,
        dream_prune_percentile=5.0,
        gist_channel_enabled=False,
    )
    records = [_rec("dull", salience=0.35, crystallized=True, scope=scope)]
    r, cands = _retriever(settings, records)
    demoted = r.assemble_context("When did the kiln schedule move?", cands, scope=scope)

    settings_off = replace(settings, dream_prune_percentile=0.0, salience_prune_threshold=0.0)
    r2, cands2 = _retriever(replace(settings_off, data_dir=settings.data_dir / "b"), records)
    kept = r2.assemble_context("When did the kiln schedule move?", cands2, scope=scope)

    assert _raw_block_chars(demoted) < _raw_block_chars(kept)
    assert _raw_block_chars(demoted) <= settings.crystal_span_chars + 120
    # the query-centered span keeps the answering sentence
    assert any("kiln schedule moved" in b for b in demoted)


def test_vivid_top_fraction_keeps_full_text_under_demotion(fresh_settings):
    scope = Scope(namespace="crystal-vivid")
    settings = replace(
        fresh_settings,
        crystal_span_demotion_enabled=True,
        dream_prune_percentile=5.0,
        gist_channel_enabled=False,
        vivid_fraction=0.25,
    )
    records = [
        _rec("vivid", salience=0.9, crystallized=True, scope=scope),
        _rec("dull-1", salience=0.35, crystallized=True, scope=scope),
        _rec("dull-2", salience=0.36, crystallized=True, scope=scope),
        _rec("dull-3", salience=0.37, crystallized=True, scope=scope),
    ]
    r, cands = _retriever(settings, records)
    blocks = r.assemble_context("When did the kiln schedule move?", cands, scope=scope)
    # 4 candidates x 0.25 -> exactly one vivid record keeps its full text; the other three are
    # crystallized and demote to bounded spans.
    full_len = len(records[0].text)
    kept_full = sum(1 for b in blocks if len(b) >= full_len)
    assert kept_full == 1


def test_uncrystallized_records_keep_full_text_under_demotion(fresh_settings):
    scope = Scope(namespace="crystal-raw")
    settings = replace(
        fresh_settings,
        crystal_span_demotion_enabled=True,
        dream_prune_percentile=5.0,
        gist_channel_enabled=False,
    )
    raw_only = [_rec("raw", salience=0.35, crystallized=False, scope=scope)]
    r, cands = _retriever(settings, raw_only)
    blocks = r.assemble_context("When did the kiln schedule move?", cands, scope=scope)
    assert _raw_block_chars(blocks) > settings.crystal_span_chars + 120


def test_flag_off_is_byte_identical(fresh_settings):
    scope = Scope(namespace="crystal-off")
    base = replace(fresh_settings, gist_channel_enabled=False, dream_prune_percentile=5.0)
    records = [_rec("dull", salience=0.35, crystallized=True, scope=scope)]

    r_off, cands_off = _retriever(base, records)
    blocks_off = r_off.assemble_context("When did the kiln schedule move?", cands_off, scope=scope)

    flagged = replace(base, crystal_span_demotion_enabled=False,
                      data_dir=base.data_dir / "c")
    r_ident, cands_ident = _retriever(flagged, records)
    blocks_ident = r_ident.assemble_context("When did the kiln schedule move?", cands_ident, scope=scope)

    assert blocks_off == blocks_ident
