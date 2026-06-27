"""METABOLISM_MODE master switch (forgetting-machine plan).

Guards the one-switch design and its two hard invariants:
  1. OFF (or unset) => the overlay mutates NOTHING (baseline byte-identical).
  2. ON => every profile flag defaults on, BUT an explicit env var still wins, so a single
     component can be ABLATED (the attribution-under-ablation proof program depends on this).
"""
from __future__ import annotations

from eidetic.config import (METABOLISM_PROFILE, apply_metabolism_overlay,
                            metabolism_enabled)


def test_profile_is_nonempty_and_string_valued():
    assert METABOLISM_PROFILE, "profile must list the flags the switch wires"
    for k, v in METABOLISM_PROFILE.items():
        assert isinstance(k, str) and isinstance(v, str) and v != ""


def test_overlay_off_is_a_noop():
    env = {"SOMETHING_ELSE": "x"}
    set_keys = apply_metabolism_overlay(env)
    assert set_keys == []
    assert env == {"SOMETHING_ELSE": "x"}  # untouched


def test_overlay_off_when_explicitly_false():
    env = {"METABOLISM_MODE": "0"}
    assert apply_metabolism_overlay(env) == []
    assert env == {"METABOLISM_MODE": "0"}


def test_metabolism_enabled_parsing():
    for on in ("1", "true", "TRUE", "yes", "on", "On"):
        assert metabolism_enabled({"METABOLISM_MODE": on})
    for off in ("0", "false", "no", "off", "", "garbage"):
        assert not metabolism_enabled({"METABOLISM_MODE": off})
    assert not metabolism_enabled({})  # unset


def test_overlay_on_fills_every_profile_flag():
    env = {"METABOLISM_MODE": "1"}
    set_keys = apply_metabolism_overlay(env)
    assert set(set_keys) == set(METABOLISM_PROFILE)
    for k, v in METABOLISM_PROFILE.items():
        assert env[k] == v


def test_explicit_override_is_preserved_for_ablation():
    # METABOLISM on, but the operator ablates FULL_SLEEP and pins a different reader mode.
    env = {"METABOLISM_MODE": "1", "FULL_SLEEP": "0", "READER_MODE": "default"}
    set_keys = apply_metabolism_overlay(env)
    assert env["FULL_SLEEP"] == "0"          # ablation respected
    assert env["READER_MODE"] == "default"   # explicit value respected
    assert "FULL_SLEEP" not in set_keys
    assert "READER_MODE" not in set_keys
    # everything else still got the profile default
    assert env["GIST_CHANNEL"] == METABOLISM_PROFILE["GIST_CHANNEL"]


def test_overlay_is_idempotent():
    env = {"METABOLISM_MODE": "1"}
    first = apply_metabolism_overlay(env)
    snapshot = dict(env)
    second = apply_metabolism_overlay(env)
    assert second == []          # nothing left to set
    assert env == snapshot       # no further mutation
    assert set(first) == set(METABOLISM_PROFILE)
