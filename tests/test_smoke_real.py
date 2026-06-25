"""Key-gated real end-to-end smoke test (Phase R).

Runs ingest -> ask -> prove on a tiny fixture using REAL model calls. Skips CLEANLY (never fakes)
when no funded key is present, and skips with a clear reason if the key is valid but quota is
exhausted -- so CI stays green without a key while the real path is exercised whenever one works.
"""
from __future__ import annotations

import pytest

from eidetic.config import get_settings


def _has_key() -> bool:
    try:
        return bool(get_settings().has_api_key)
    except Exception:
        return False


@pytest.mark.skipif(not _has_key(), reason="needs a funded DASHSCOPE_API_KEY")
def test_real_ingest_ask_prove(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    get_settings.cache_clear()
    from eidetic.dashscope_client import ModelCallError
    from eidetic.engine import Engine
    from eidetic.models import NLILabel, Scope

    try:
        eng = Engine(get_settings())
        scope = Scope(namespace="smoke")
        eng.ingest_text("Alice works at Acme Corporation.", scope=scope)
        ans = eng.ask("where does Alice work", scope=scope)

        assert ans.answer.strip()                                   # real, non-empty answer
        assert ans.citations, "a verified answer must cite sources"
        for c in ans.citations:
            assert eng.get_record(c.memory_id) is not None          # every citation is a real source
            assert isinstance(c.nli_label, NLILabel)                # a real NLI label, not a stub
        proof = eng.prove(ans)
        assert proof["evidence"] and proof["provenance_complete"]   # complete provenance chain
    except ModelCallError as e:
        msg = str(e).lower()
        # Skip cleanly (never fake) when the real path can't run: quota exhausted, or the key is
        # not visible in this process (env/test-isolation). A genuinely broken capability re-raises.
        if any(s in msg for s in ("quota", "free tier", "not set", "api_key", "api key")):
            pytest.skip(f"real path not exercised (no working key/quota): {e}")
        raise
    finally:
        get_settings.cache_clear()
