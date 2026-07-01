"""Seed handling for rotating benchmark-neutral sidecars."""
from __future__ import annotations

import random


def resolve_seed(seed: object | None) -> tuple[int, str]:
    raw = "" if seed is None else str(seed).strip()
    if raw.lower() in {"", "auto", "random"}:
        return random.SystemRandom().randint(1, 2**31 - 1), "random"
    return int(raw), "fixed"
