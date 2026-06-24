"""Component 2 (vector half) + Component 3 storage: the cognitive-coordinate index.

Stores, per memory, TWO vectors:
  * content embedding  -- qwen3-vl-embedding / text-embedding-v4 (semantic content)
  * structure code     -- metadata coordinate vector (entity type, role, temporal
                          coordinate, graph-position/PPR features) from structure_code.py

Retrieval ranks by CONTENT similarity (and optionally structure), where a memory's
age never enters the distance -- this is what makes recall@k age-independent. The
FSRS priority weight is deliberately NOT used here.

dev backends:
  * numpy   -- exact cosine (always available; age-independent latency by construction)
  * hnswlib -- real HNSW approximate-nearest-neighbour (matches the prod AnalyticDB-PG
               HNSW story; latency depends on N and recall target, never on age)
prod -> AnalyticDB for PostgreSQL with HNSW (see docs/architecture.md).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Settings, get_settings


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class VectorIndex(ABC):
    @abstractmethod
    def add(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None: ...

    @abstractmethod
    def update(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        """Replace a memory's content (re-embed on confirmed recall = affinity maturation).
        Re-embedding changes the CONTENT vector only; age never enters retrieval."""

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int,
               allowed_ids: Optional[set[str]] = None) -> list[tuple[str, float]]:
        """Top-k by content cosine similarity. Returns [(memory_id, score)] desc."""

    @abstractmethod
    def search_struct(self, query_struct: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Return stored content vectors for the given memory_ids (for the 3D projection)."""

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...


class NumpyVectorIndex(VectorIndex):
    """Exact cosine index. Brute force => latency is O(N), independent of memory age."""

    def __init__(self, index_dir: Path, dim: int, struct_dim: int):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.struct_dim = struct_dim
        self.ids: list[str] = []
        self.content = np.zeros((0, dim), dtype=np.float32)
        self.struct = np.zeros((0, struct_dim), dtype=np.float32)
        self._load()

    def _path(self) -> Path:
        return self.dir / "numpy_index.npz"

    def _load(self) -> None:
        p = self._path()
        if p.exists():
            data = np.load(p, allow_pickle=True)
            self.ids = list(data["ids"])
            self.content = data["content"].astype(np.float32)
            self.struct = data["struct"].astype(np.float32)
            if self.content.shape[0]:
                self.dim = self.content.shape[1]
                self.struct_dim = self.struct.shape[1]

    def add(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        cv = _normalize(content_vec).reshape(1, -1)
        sv = _normalize(struct_vec if struct_vec is not None else np.zeros(self.struct_dim)).reshape(1, -1)
        if self.content.shape[0] == 0:
            self.content = np.zeros((0, cv.shape[1]), dtype=np.float32)
            self.dim = cv.shape[1]
        self.ids.append(memory_id)
        self.content = np.vstack([self.content, cv])
        self.struct = np.vstack([self.struct, sv])

    def update(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        if memory_id not in self.ids:
            return self.add(memory_id, content_vec, struct_vec)
        i = self.ids.index(memory_id)
        self.content[i] = _normalize(content_vec)
        if struct_vec is not None:
            self.struct[i] = _normalize(struct_vec)

    def _topk(self, matrix: np.ndarray, query: np.ndarray, k: int,
              allowed_ids: Optional[set[str]] = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = _normalize(query)
        if allowed_ids is not None:
            pos = [i for i, mid in enumerate(self.ids) if mid in allowed_ids]
            if not pos:
                return []
            sub = matrix[pos]
            sims = sub @ q
            k = min(k, len(pos))
            idx_local = np.argpartition(-sims, k - 1)[:k]
            idx_local = idx_local[np.argsort(-sims[idx_local])]
            return [(self.ids[pos[i]], float(sims[i])) for i in idx_local]
        sims = matrix @ q
        k = min(k, len(self.ids))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def search(self, query_vec: np.ndarray, k: int,
               allowed_ids: Optional[set[str]] = None) -> list[tuple[str, float]]:
        return self._topk(self.content, query_vec, k, allowed_ids=allowed_ids)

    def search_struct(self, query_struct: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self._topk(self.struct, query_struct, k)

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        pos = {mid: i for i, mid in enumerate(self.ids)}
        return {mid: self.content[pos[mid]] for mid in ids if mid in pos}

    def save(self) -> None:
        np.savez(self._path(), ids=np.array(self.ids, dtype=object),
                 content=self.content, struct=self.struct)

    def __len__(self) -> int:
        return len(self.ids)


class HnswVectorIndex(VectorIndex):
    """Real HNSW approximate-nearest-neighbour over content vectors (hnswlib).

    Structure-code search uses the small numpy matrix (cheap, low-dim)."""

    def __init__(self, index_dir: Path, dim: int, struct_dim: int, ef: int = 128, M: int = 32,
                 ef_construction: int = 200):
        import hnswlib

        self._hnswlib = hnswlib
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.struct_dim = struct_dim
        self.ef = ef
        self.M = M
        self.ids: list[str] = []
        self.struct = np.zeros((0, struct_dim), dtype=np.float32)
        self.ef_construction = ef_construction
        self._capacity = 1024
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=self._capacity, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef)
        self._load()

    def _paths(self):
        return (self.dir / "hnsw.bin", self.dir / "hnsw_meta.json", self.dir / "hnsw_struct.npy")

    def _load(self) -> None:
        bin_p, meta_p, struct_p = self._paths()
        if bin_p.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text())
            self.ids = meta["ids"]
            self.dim = meta["dim"]
            self.struct_dim = meta["struct_dim"]
            self._capacity = max(self._capacity, len(self.ids) + 1024)
            self.index = self._hnswlib.Index(space="cosine", dim=self.dim)
            self.index.load_index(str(bin_p), max_elements=self._capacity)
            self.index.set_ef(self.ef)
            if struct_p.exists():
                self.struct = np.load(struct_p).astype(np.float32)

    def _ensure_capacity(self) -> None:
        if len(self.ids) >= self._capacity:
            self._capacity *= 2
            self.index.resize_index(self._capacity)

    def add(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        self._ensure_capacity()
        label = len(self.ids)
        self.index.add_items(_normalize(content_vec).reshape(1, -1), np.array([label]))
        self.ids.append(memory_id)
        sv = _normalize(struct_vec if struct_vec is not None else np.zeros(self.struct_dim)).reshape(1, -1)
        self.struct = np.vstack([self.struct, sv]) if self.struct.size else sv

    def update(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        if memory_id not in self.ids:
            return self.add(memory_id, content_vec, struct_vec)
        label = self.ids.index(memory_id)
        # hnswlib updates an existing element in place when re-added with the same label.
        self.index.add_items(_normalize(content_vec).reshape(1, -1), np.array([label]))
        if struct_vec is not None:
            self.struct[label] = _normalize(struct_vec)

    def search(self, query_vec: np.ndarray, k: int,
               allowed_ids: Optional[set[str]] = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        k = min(k, len(self.ids))
        q = _normalize(query_vec)
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            if not allowed:
                return []
            fetch = min(len(self.ids), max(k, min(len(self.ids), len(allowed) * 4)))
        else:
            fetch = k
        labels, distances = self.index.knn_query(q.reshape(1, -1), k=fetch)
        out = []
        for lab, dist in zip(labels[0], distances[0]):
            mid = self.ids[int(lab)]
            if allowed_ids is None or mid in allowed:
                out.append((mid, float(1.0 - dist)))  # cosine sim = 1 - dist
                if len(out) >= k:
                    return out
        if allowed_ids is not None and len(out) < min(k, len(allowed)):
            pos = {mid: i for i, mid in enumerate(self.ids)}
            labels = [pos[mid] for mid in allowed if mid in pos]
            if labels:
                items = self.index.get_items(labels)
                sims = np.asarray(items, dtype=np.float32) @ q
                order = np.argsort(-sims)[: min(k, len(labels))]
                out = [(self.ids[labels[i]], float(sims[i])) for i in order]
        return out[:k]

    def search_struct(self, query_struct: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = _normalize(query_struct)
        sims = self.struct @ q
        k = min(k, len(self.ids))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        pos = {mid: i for i, mid in enumerate(self.ids)}
        labels = [pos[mid] for mid in ids if mid in pos]
        if not labels:
            return {}
        items = self.index.get_items(labels)
        return {self.ids[lab]: np.asarray(v, dtype=np.float32) for lab, v in zip(labels, items)}

    def save(self) -> None:
        bin_p, meta_p, struct_p = self._paths()
        self.index.save_index(str(bin_p))
        meta_p.write_text(json.dumps({"ids": self.ids, "dim": self.dim, "struct_dim": self.struct_dim}))
        np.save(struct_p, self.struct)

    def __len__(self) -> int:
        return len(self.ids)


def make_vector_index(settings: Optional[Settings] = None) -> VectorIndex:
    settings = settings or get_settings()
    backend = settings.vector_backend

    def _hnsw() -> "HnswVectorIndex":
        return HnswVectorIndex(settings.index_dir, settings.embed_dim, settings.struct_dim,
                               ef=settings.hnsw_ef_search, M=settings.hnsw_m,
                               ef_construction=settings.hnsw_ef_construction)

    if backend in ("hnswlib", "hnsw"):
        return _hnsw()
    if backend == "numpy":
        return NumpyVectorIndex(settings.index_dir, settings.embed_dim, settings.struct_dim)
    # auto: prefer real HNSW if the wheel is present, else exact numpy.
    try:
        import hnswlib  # noqa: F401
        return _hnsw()
    except Exception:
        return NumpyVectorIndex(settings.index_dir, settings.embed_dim, settings.struct_dim)
