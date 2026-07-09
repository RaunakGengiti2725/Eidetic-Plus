"""Hindsight adapter: contract + registration + fail-loud. No live calls (no pg0 spin-up,
no LLM). The block-extraction is unit-tested against synthetic recall-response shapes so the
neutral recall->shared-reader mapping is verified without the heavy backend."""
import os

import pytest

from bench.adapters.hindsight_adapter import HindsightSystem
from bench.adapters.base import MemorySystem


def test_conforms_to_memory_system_contract():
    assert issubclass(HindsightSystem, MemorySystem)
    for m in ("reset", "ingest_session", "answer"):
        assert callable(getattr(HindsightSystem, m))
    assert HindsightSystem.name == "hindsight"


def test_registered_in_harness_factory():
    import bench.run as run
    src = __import__("inspect").getsource(run)
    assert '"hindsight"' in src and "HindsightSystem" in src


def test_fails_loud_without_key(monkeypatch):
    # Even with the package installed, an empty key must raise -- never mock/fabricate.
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with pytest.raises(RuntimeError):
        HindsightSystem()


class _Res:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, texts):
        self.results = [_Res(t) for t in texts]


def test_block_extraction_from_recall_response():
    blocks = HindsightSystem._blocks(_Resp(["Priya moved to Lisbon.", "  ", "Joined the team."]))
    assert blocks == ["Priya moved to Lisbon.", "Joined the team."]   # blanks dropped


def test_block_extraction_handles_dict_and_list_shapes():
    assert HindsightSystem._blocks({"results": [{"text": "a"}, {"content": "b"}]}) == ["a", "b"]
    assert HindsightSystem._blocks([{"text": "x"}]) == ["x"]
    assert HindsightSystem._blocks(None) == []
