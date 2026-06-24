"""Required test: the immutable substrate is write-once and never overwrites."""
from __future__ import annotations

import pytest

from eidetic.substrate import ImmutableViolation, LocalCASSubstrate, sha256_hex


def test_content_addressing_and_dedup(tmp_path):
    sub = LocalCASSubstrate(tmp_path / "cas")
    data = b"the original immutable record"
    h1, uri1 = sub.put(data)
    assert h1 == sha256_hex(data)
    assert uri1 == f"cas://{h1}"

    # Writing identical content again dedupes to the same key, no error, no change.
    h2, uri2 = sub.put(data)
    assert (h1, uri1) == (h2, uri2)
    assert sub.get(h1) == data
    assert sub.verify(h1)


def test_object_is_read_only_on_disk(tmp_path):
    sub = LocalCASSubstrate(tmp_path / "cas")
    h, _ = sub.put(b"locked forever")
    path = sub.path_for(h)
    # The OS itself must refuse an overwrite (objects are chmod 0o444).
    with pytest.raises((PermissionError, OSError)):
        with open(path, "wb") as f:
            f.write(b"tampered")
    # And the bytes are unchanged.
    assert sub.get(h) == b"locked forever"


def test_there_is_no_overwrite_api(tmp_path):
    sub = LocalCASSubstrate(tmp_path / "cas")
    h, _ = sub.put(b"v1")
    # Different content -> different hash -> a NEW object, the old one survives untouched.
    h2, _ = sub.put(b"v2")
    assert h != h2
    assert sub.get(h) == b"v1"
    assert sub.get(h2) == b"v2"
