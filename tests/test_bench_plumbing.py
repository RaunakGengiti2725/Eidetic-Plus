"""Offline tests for the bench harness plumbing added by the dominance plan (no model calls).

Asserts the new rows/knobs are wired and that the harness defaults are byte-identical to the
historical run (READER_BLOCK_CHARS=3000, INGEST_GRANULARITY=session, FULL_SLEEP off).
"""
from __future__ import annotations

import importlib

from bench.adapters.eidetic_adapter import (
    EideticFullSystem,
    EideticProductSystem,
    EideticSystem,
    _truthy,
)


def test_product_row_is_eidetic_subclass():
    assert EideticProductSystem.name == "eidetic-product"
    assert issubclass(EideticProductSystem, EideticSystem)
    # The product row overrides answer() (engine.ask path), not write/consolidate.
    assert EideticProductSystem.answer is not EideticSystem.answer
    assert EideticProductSystem.ingest_session is EideticSystem.ingest_session
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
