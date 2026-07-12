"""Store-isolation guard (incident 2026-07-11): three windows launched without an explicit
DATA_DIR shared the default store with colliding namespaces; each window's reset destroyed
the previous window's records. bench.run now refuses shared non-empty stores, redirects
empty defaults to a dedicated per-window store, and honors the explicit override. Pure
unit tests on the extracted guard -- no subprocess, no pipeline."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bench.run import enforce_store_isolation


def _store_with_ns(path: Path, ns: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path / "eidetic.sqlite")
    con.execute("CREATE TABLE memories (memory_id TEXT, namespace TEXT)")
    con.execute("INSERT INTO memories VALUES ('m1', ?)", (ns,))
    con.commit()
    con.close()


def test_refuses_shared_store_with_existing_namespaces(tmp_path):
    shared = tmp_path / "shared"
    _store_with_ns(shared, "eidetic-plus-full-locomo-g0-r0")
    env = {"DATA_DIR": str(shared)}
    with pytest.raises(SystemExit) as e:
        enforce_store_isolation(tmp_path / "win", env)
    assert "REFUSING to run against a shared store" in str(e.value)


def test_redirects_empty_default_to_dedicated_store(tmp_path, capsys):
    env = {"DATA_DIR": str(tmp_path / "nonexistent_default")}
    out = tmp_path / "win2"
    enforce_store_isolation(out, env)
    assert env["DATA_DIR"] == str(out / "data")
    assert "redirected to dedicated store" in capsys.readouterr().out


def test_dedicated_store_under_out_passes_untouched(tmp_path):
    out = tmp_path / "win3"
    dedicated = out / "data"
    _store_with_ns(dedicated, "eidetic-plus-full-locomo-g0-r0")  # own prior run = fine
    env = {"DATA_DIR": str(dedicated)}
    enforce_store_isolation(out, env)
    assert env["DATA_DIR"] == str(dedicated)  # untouched, no exit


def test_override_allows_shared_store(tmp_path):
    shared = tmp_path / "shared2"
    _store_with_ns(shared, "x-ns")
    env = {"DATA_DIR": str(shared), "BENCH_ALLOW_SHARED_STORE": "1"}
    enforce_store_isolation(tmp_path / "win4", env)  # no exit
    assert env["DATA_DIR"] == str(shared)
