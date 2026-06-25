"""Typed errors so callers can distinguish a real failure from an experimental-not-built path."""
from __future__ import annotations


class FeatureNotImplementedError(RuntimeError):
    """An explicitly experimental, flag-gated feature whose enabled path is not built yet.

    Distinct from a bug: it is a clear, documented 'this is off by default and not implemented'
    signal (never a silent no-op, never a raw NotImplementedError traceback)."""
