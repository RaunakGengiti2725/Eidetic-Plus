"""Shared pytest fixtures. Tests that need no model call run fully offline; the
NLI test skips automatically unless a real DASHSCOPE_API_KEY is present."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def fresh_settings(tmp_path, monkeypatch):
    """A Settings object rooted at an isolated temp data dir (numpy backend for determinism)."""
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    from eidetic.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    yield s
    get_settings.cache_clear()


@pytest.fixture()
def engine(fresh_settings):
    from eidetic.engine import Engine

    return Engine(fresh_settings)
