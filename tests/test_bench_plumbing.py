"""Offline tests for the bench harness plumbing added by the dominance plan (no model calls).

Asserts the new rows/knobs are wired and that the harness defaults are byte-identical to the
historical run (READER_BLOCK_CHARS=3000, INGEST_GRANULARITY=session, FULL_SLEEP off).
"""
from __future__ import annotations

import importlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace

from bench.adapters.eidetic_adapter import (
    EideticFullSystem,
    EideticProductSystem,
    EideticSystem,
    _truthy,
)
from bench.datasets import Sample, Session, Turn


def test_product_row_is_eidetic_subclass():
    assert EideticProductSystem.name == "eidetic-product"
    assert issubclass(EideticProductSystem, EideticSystem)
    # The product row overrides answer() (engine.ask path), not write/consolidate.
    assert EideticProductSystem.answer is not EideticSystem.answer
    assert EideticProductSystem.ingest_session is EideticSystem.ingest_session
    assert EideticProductSystem.after_answer is not EideticSystem.after_answer
    assert EideticFullSystem.name == "eidetic-plus-full"


def test_truthy_helper():
    import os

    for v in ("1", "true", "yes", "TRUE", "Yes"):
        os.environ["XX_TRUTHY_TEST"] = v
        assert _truthy("XX_TRUTHY_TEST") is True
    for v in ("0", "no", "", "off"):
        os.environ["XX_TRUTHY_TEST"] = v
        assert _truthy("XX_TRUTHY_TEST") is False
    del os.environ["XX_TRUTHY_TEST"]
    assert _truthy("XX_DEFINITELY_UNSET_VAR") is False


def test_reproduce_ablation_bundle_matches_release_gate_contract():
    text = Path("bench/reproduce.sh").read_text()

    for env_name in (
        "ABLATION_FULL_DIR",
        "ABLATION_METABOLISM_OFF_DIR",
        "ABLATION_REGIONS_OFF_DIR",
        "ABLATION_FORGETTING_OFF_DIR",
        "ABLATION_AFFECT_OFF_DIR",
    ):
        assert env_name in text
    for flag in ("--full", "--metabolism-off", "--regions-off", "--forgetting-off", "--affect-off"):
        assert flag in text


def test_reproduce_invariant_sidecars_fail_through_release_gate():
    text = Path("bench/reproduce.sh").read_text()

    assert "run_sidecar()" in text
    assert "one or more invariant sidecars failed; release_gate will fail closed" in text
    for report_name in (
        "affect_salience_invariant.json",
        "scratchpad_invariant.json",
        "region_routing_invariant.json",
        "reflex_recall_invariant.json",
        "smqe_planner_invariant.json",
        "smqe_synthetic_invariant.json",
        "smqe_claim_coverage.json",
        "smqe_fullpath_invariant.json",
        "smqe_paraphrase_invariant.json",
        "smqe_conflict_invariant.json",
        "smqe_composition_invariant.json",
        "smqe_relative_phrase_invariant.json",
        "smqe_temporal_window_invariant.json",
        "smqe_attribution_invariant.json",
        "smqe_abstention_invariant.json",
        "smqe_scope_invariant.json",
        "smqe_subscope_invariant.json",
        "smqe_time_invariant.json",
        "smqe_invalidation_invariant.json",
    ):
        assert report_name in text
    assert text.count("run_sidecar ") >= 19


def test_reader_block_chars_default_is_3000(monkeypatch):
    monkeypatch.delenv("READER_BLOCK_CHARS", raising=False)
    import bench.reader as reader

    importlib.reload(reader)
    assert reader.READER_BLOCK_CHARS == 3000


def test_reader_block_chars_env_override(monkeypatch):
    monkeypatch.setenv("READER_BLOCK_CHARS", "8000")
    import bench.reader as reader

    importlib.reload(reader)
    assert reader.READER_BLOCK_CHARS == 8000
    monkeypatch.delenv("READER_BLOCK_CHARS", raising=False)
    importlib.reload(reader)  # restore default for other tests


def test_reader_mode_default_is_byte_identical(monkeypatch):
    monkeypatch.delenv("READER_MODE", raising=False)
    import bench.judge as judge
    import bench.reader as reader

    importlib.reload(reader)
    assert reader._READER_PROMPT is judge.FIXED_READER_PROMPT
    assert reader._READER_PROMPT == judge.FIXED_READER_PROMPT


def test_reader_mode_photographic_selects_extractive_prompt(monkeypatch):
    import bench.judge as judge

    for mode in ("photographic", "extractive"):
        monkeypatch.setenv("READER_MODE", mode)
        import bench.reader as reader

        importlib.reload(reader)
        assert reader._READER_PROMPT == judge.FIXED_READER_PHOTOGRAPHIC_PROMPT
        assert "photographic recall" in reader._READER_PROMPT
    monkeypatch.delenv("READER_MODE", raising=False)
    importlib.reload(reader)  # restore default for other tests


def test_make_system_knows_product_name(monkeypatch):
    # Exercise the ROUTING BEHAVIOR (not just that a string literal exists): both product aliases
    # must construct an EideticProductSystem, and an unknown name must still raise. Stub the Engine
    # constructor so no real Engine/index is built offline.
    import pytest

    import bench.adapters.eidetic_adapter as ea
    import bench.run as run

    monkeypatch.setattr(ea.EideticProductSystem, "__init__", lambda self: None)
    assert isinstance(run.make_system("eidetic-product"), ea.EideticProductSystem)
    assert isinstance(run.make_system("eidetic-plus-product"), ea.EideticProductSystem)
    with pytest.raises(SystemExit):
        run.make_system("totally-bogus-system")


def test_longmemeval_liveness_preflight_fails_unsafe_eidetic_profile():
    import bench.run as run

    sample = Sample(
        sample_id="lme-unsafe",
        dataset="longmemeval",
        category="multi-session",
        question="What did Alice buy?",
        gold="a compass",
        sessions=[Session(
            session_id="s0",
            turns=[Turn(role="user", content="Alice purchased a compass.\n" + ("x " * 7000))],
        )],
    )
    errors = run.longmemeval_liveness_errors(
        [sample],
        "eidetic-full",
        env={
            "RAW_SPAN_MIN_CHARS": "500",
            "EXTRACT_CHUNK_CHARS": "1000",
            "CONSOLIDATION_EXTRACT_DEADLINE_SEC": "0",
            "CONSOLIDATION_TIMEOUT_POLICY": "degrade",
            "RAW_SPAN_AUDIT": "0",
            "CONSOLIDATION_RAW_ONLY_WINDOW_THRESHOLD": "0",
            "CONSOLIDATION_EXTRACT_CALL_BUDGET": "0",
            "CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY": "0",
        },
    )

    assert errors
    assert "Unsafe LongMemEval Eidetic liveness profile" in errors[0]
    assert "CONSOLIDATION_EXTRACT_DEADLINE_SEC" in errors[0]
    assert "RAW_SPAN_AUDIT=1" in errors[0]


def test_longmemeval_liveness_preflight_allows_safe_raw_span_profile():
    import bench.run as run

    sample = Sample(
        sample_id="lme-safe",
        dataset="longmemeval",
        category="multi-session",
        question="What did Alice buy?",
        gold="a compass",
        sessions=[Session(
            session_id="s0",
            turns=[Turn(role="user", content="Alice purchased a compass.\n" + ("x " * 7000))],
        )],
    )
    errors = run.longmemeval_liveness_errors(
        [sample],
        "eidetic-full,rag-full",
        env={
            "RAW_SPAN_MIN_CHARS": "500",
            "EXTRACT_CHUNK_CHARS": "1000",
            "CONSOLIDATION_EXTRACT_DEADLINE_SEC": "30",
            "CONSOLIDATION_TIMEOUT_POLICY": "degrade",
            "RAW_SPAN_AUDIT": "1",
            "CONSOLIDATION_RAW_ONLY_WINDOW_THRESHOLD": "3",
            "CONSOLIDATION_EXTRACT_CALL_BUDGET": "0",
            "CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY": "0",
        },
    )

    assert errors == []


def test_longmemeval_liveness_preflight_allows_full_consolidation_profile():
    import bench.run as run

    sample = Sample(
        sample_id="lme-full-consolidation",
        dataset="longmemeval",
        category="multi-session",
        question="What did Alice buy?",
        gold="a compass",
        sessions=[Session(
            session_id="s0",
            turns=[Turn(role="user", content="Alice purchased a compass.\n" + ("x " * 7000))],
        )],
    )
    errors = run.longmemeval_liveness_errors(
        [sample],
        "eidetic-full,rag-full",
        env={
            "BENCH_FULL_CONSOLIDATION": "1",
            "RAW_SPAN_MIN_CHARS": "500",
            "EXTRACT_CHUNKING": "1",
            "EXTRACT_CHUNK_CHARS": "1000",
            "CONSOLIDATION_EXTRACT_DEADLINE_SEC": "0",
            "CONSOLIDATION_TIMEOUT_POLICY": "degrade",
            "RAW_SPAN_AUDIT": "0",
            "CONSOLIDATION_RAW_ONLY_WINDOW_THRESHOLD": "0",
            "CONSOLIDATION_EXTRACT_CALL_BUDGET": "0",
            "CONSOLIDATION_LONG_HAYSTACK_RAW_ONLY": "0",
        },
    )

    assert errors == []


def test_render_only_manifest_recovers_sample_contract_from_logs(tmp_path):
    import bench.run as run

    args = SimpleNamespace(
        systems="eidetic-plus-full,rag-full",
        dataset="longmemeval",
        split="test",
        subset=0,
        sample_offset=0,
        sample_strategy="stratified",
        runs=0,
        run_offset=0,
        variant="longmemeval_s",
        render_only=True,
    )
    row_a = {
        "system": "eidetic-plus-full",
        "dataset": "longmemeval",
        "category": "single-session-user",
        "sample_id": "lme_q0",
        "run_idx": 0,
        "correct": True,
    }
    row_b = {
        "system": "rag-full",
        "dataset": "longmemeval",
        "category": "temporal-reasoning",
        "sample_id": "lme_q1",
        "run_idx": 0,
        "correct": False,
    }
    duplicate_a = dict(row_a, system="rag-full")
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(
        json.dumps(row_a) + "\n" + json.dumps(row_b) + "\n"
    )
    (tmp_path / "rag-full__run0.jsonl").write_text(json.dumps(duplicate_a) + "\n")

    manifest_path = run.write_manifest(tmp_path, args, {"mode": "unit"}, samples=None)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["render_only"] is True
    assert manifest["systems"] == "eidetic-plus-full,rag-full"
    assert manifest["dataset"] == "longmemeval"
    assert manifest["runs"] == 1
    assert manifest["run_offset"] == 0
    assert manifest["sample_count"] == 2
    assert manifest["category_counts"] == {
        "single-session-user": 1,
        "temporal-reasoning": 1,
    }
    assert manifest["sample_rows"] == [
        {
            "dataset": "longmemeval",
            "sample_id": "lme_q0",
            "category": "single-session-user",
        },
        {
            "dataset": "longmemeval",
            "sample_id": "lme_q1",
            "category": "temporal-reasoning",
        },
    ]


def test_render_only_manifest_does_not_preserve_prior_render_only_env(monkeypatch, tmp_path):
    import bench.run as run

    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "systems": "stale-system",
        "dataset": "locomo",
        "split": "all",
        "runs": 99,
        "render_only": True,
        "env": {"DATA_DIR": "./stale-data"},
        "sample_rows": None,
    }))
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(json.dumps({
        "system": "eidetic-plus-full",
        "dataset": "longmemeval",
        "category": "multi-session",
        "sample_id": "lme_q7",
        "run_idx": 0,
        "correct": True,
    }) + "\n")
    monkeypatch.setenv("DATA_DIR", "/fresh-data")
    args = SimpleNamespace(
        systems="eidetic-plus-full",
        dataset="longmemeval",
        split="test",
        subset=1,
        sample_offset=0,
        sample_strategy="stratified",
        runs=0,
        run_offset=0,
        variant="longmemeval_s",
        render_only=True,
    )

    manifest_path = run.write_manifest(tmp_path, args, {"mode": "unit"}, samples=None)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["systems"] == "eidetic-plus-full"
    assert manifest["dataset"] == "longmemeval"
    assert manifest["split"] == "test"
    assert manifest["runs"] == 1
    assert manifest["env"]["DATA_DIR"] == "/fresh-data"


def test_graphiti_system_fails_fast_on_unresolvable_neo4j_dns(monkeypatch, tmp_path):
    import pytest

    from eidetic.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://missing.example.invalid")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    get_settings.cache_clear()

    def fail_dns(*_args, **_kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fail_dns)
    from bench.adapters.graphiti_adapter import GraphitiSystem

    with pytest.raises(RuntimeError, match="cannot resolve the Neo4j host"):
        GraphitiSystem()

    get_settings.cache_clear()
