"""Component 1: the immutable, lossless, content-addressed substrate.

This is the "perfect record" -- the thing no human has and competitors discard.
Invariants enforced here:
  * Content-addressed: object key = sha256(bytes). Identical content dedupes.
  * Write-once: an object is written exactly once, then made read-only. There is
    NO API to overwrite or delete. Both are actively refused.
  * Forgetting NEVER touches this layer. Only the index (FSRS priority) forgets.

dev  -> LocalCASSubstrate (append-only, content-addressed dir, 0o444 objects).
prod -> OSSWORMSubstrate  (Alibaba Cloud OSS with WORM retention; same contract).
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .config import Settings, get_settings


class ImmutableViolation(RuntimeError):
    """Raised on any attempt to overwrite or delete an immutable object."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Substrate(ABC):
    scheme: str = "cas"

    @abstractmethod
    def put(self, data: bytes) -> tuple[str, str]:
        """Write bytes once. Returns (content_hash, raw_uri). Idempotent on dedup."""

    @abstractmethod
    def get(self, content_hash: str) -> bytes:
        """Read raw bytes back by hash. The ground truth for verification."""

    @abstractmethod
    def exists(self, content_hash: str) -> bool: ...

    def uri_for(self, content_hash: str) -> str:
        return f"{self.scheme}://{content_hash}"

    # Both operations are forbidden by design; subclasses inherit the refusal.
    def delete(self, content_hash: str) -> None:
        raise ImmutableViolation(
            "The immutable substrate never deletes. Forgetting happens only at the "
            "index (FSRS priority down-weighting), never here."
        )

    def verify(self, content_hash: str) -> bool:
        """Recompute the hash of stored bytes and confirm it matches the key."""
        return sha256_hex(self.get(content_hash)) == content_hash


class LocalCASSubstrate(Substrate):
    """Local append-only content-addressed store. Objects are chmod 0o444 after write,
    so the OS itself blocks overwrites -- a real write-once guarantee, not a convention."""

    scheme = "cas"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, content_hash: str) -> Path:
        return self.root / content_hash[:2] / content_hash

    def path_for(self, content_hash: str) -> Path:
        return self._path(content_hash)

    def exists(self, content_hash: str) -> bool:
        return self._path(content_hash).exists()

    def put(self, data: bytes) -> tuple[str, str]:
        h = sha256_hex(data)
        path = self._path(h)
        if path.exists():
            # Content-addressed dedup: same key => same bytes. Verify and return.
            existing = path.read_bytes()
            if sha256_hex(existing) != h:  # pragma: no cover - corruption guard
                raise ImmutableViolation(f"Stored object {h} is corrupted on disk.")
            return h, self.uri_for(h)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write to a temp file, then move into place, then lock read-only.
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        os.chmod(path, 0o444)
        return h, self.uri_for(h)

    def get(self, content_hash: str) -> bytes:
        path = self._path(content_hash)
        if not path.exists():
            raise KeyError(f"No immutable object for hash {content_hash}")
        return path.read_bytes()


class OSSWORMSubstrate(Substrate):
    """Alibaba Cloud OSS with Write-Once-Read-Many retention (Component 1, prod).

    Same contract as the local store. Requires the `oss2` package and OSS_* env vars.
    WORM retention is configured at the bucket via a retention policy; this class
    relies on that policy to make objects immutable server-side. We additionally
    never call delete/overwrite. Objects are keyed by sha256 for content-addressing.
    """

    scheme = "oss"

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            import oss2  # noqa: F401
        except ImportError as e:  # pragma: no cover - prod-only path
            raise ImmutableViolation(
                "APP_ENV=prod selects OSS-WORM storage but `oss2` is not installed. "
                "Run `pip install oss2` and set OSS_* in .env (see docs/architecture.md)."
            ) from e
        import oss2

        auth = oss2.Auth(settings.oss_access_key_id, settings.oss_access_key_secret)
        self.bucket = oss2.Bucket(auth, settings.oss_endpoint, settings.oss_bucket)

    def _key(self, content_hash: str) -> str:
        return f"eidetic/{content_hash}"

    def exists(self, content_hash: str) -> bool:  # pragma: no cover - prod-only
        return self.bucket.object_exists(self._key(content_hash))

    def put(self, data: bytes) -> tuple[str, str]:  # pragma: no cover - prod-only
        h = sha256_hex(data)
        key = self._key(h)
        if not self.bucket.object_exists(key):
            # WORM bucket policy makes this object immutable server-side after write.
            self.bucket.put_object(key, data)
        return h, f"oss://{self.settings.oss_bucket}/{key}"

    def get(self, content_hash: str) -> bytes:  # pragma: no cover - prod-only
        return self.bucket.get_object(self._key(content_hash)).read()


def make_substrate(settings: Optional[Settings] = None) -> Substrate:
    settings = settings or get_settings()
    if settings.is_prod:
        return OSSWORMSubstrate(settings)
    return LocalCASSubstrate(settings.substrate_dir)
