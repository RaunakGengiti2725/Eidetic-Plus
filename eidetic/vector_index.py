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
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Settings, get_settings


def _atomic_write_bytes(path: Path, write_fn) -> None:
    """Write via a temp file in the same dir, then os.replace (atomic on POSIX). A reader/crash
    never sees a half-written file. `write_fn(tmp_path)` does the actual write."""
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


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
               allowed_ids: Optional[set[str]] = None,
               ef: Optional[int] = None) -> list[tuple[str, float]]:
        """Top-k by content cosine similarity. Returns [(memory_id, score)] desc.
        `ef` optionally overrides the HNSW search breadth for this query (adaptive
        efSearch: raise it for hard queries). Exact backends ignore it."""

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
        # Publish the matrices (new arrays) BEFORE the id list, and publish ids as a NEW list rather
        # than an in-place append, so a concurrent lock-free search sees a consistent (ids, content)
        # snapshot. _topk also clamps to the common prefix as a second guard.
        self.content = np.vstack([self.content, cv])
        self.struct = np.vstack([self.struct, sv])
        self.ids = self.ids + [memory_id]

    def update(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        if memory_id not in self.ids:
            return self.add(memory_id, content_vec, struct_vec)
        i = self.ids.index(memory_id)
        self.content[i] = _normalize(content_vec)
        if struct_vec is not None:
            self.struct[i] = _normalize(struct_vec)

    def _topk(self, matrix: np.ndarray, query: np.ndarray, k: int,
              allowed_ids: Optional[set[str]] = None) -> list[tuple[str, float]]:
        # Capture a CONSISTENT snapshot: a concurrent lock-free reader can catch add() mid-publish
        # (ids and content grow in separate statements), so clamp to the common prefix n. `matrix`
        # is the caller's captured self.content reference; `ids` is captured once here.
        ids = self.ids
        n = min(len(ids), matrix.shape[0])
        if n == 0:
            return []
        matrix = matrix[:n]
        q = _normalize(query)
        if allowed_ids is not None:
            pos = [i for i in range(n) if ids[i] in allowed_ids]
            if not pos:
                return []
            sub = matrix[pos]
            sims = sub @ q
            k = min(k, len(pos))
            idx_local = np.argpartition(-sims, k - 1)[:k]
            idx_local = idx_local[np.argsort(-sims[idx_local])]
            return [(ids[pos[i]], float(sims[i])) for i in idx_local]
        sims = matrix @ q
        k = min(k, n)
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(ids[i], float(sims[i])) for i in idx]

    def search(self, query_vec: np.ndarray, k: int,
               allowed_ids: Optional[set[str]] = None,
               ef: Optional[int] = None) -> list[tuple[str, float]]:
        # Exact brute-force backend: ef has no meaning (always exact); accepted for parity.
        return self._topk(self.content, query_vec, k, allowed_ids=allowed_ids)

    def search_struct(self, query_struct: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self._topk(self.struct, query_struct, k)

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        pos = {mid: i for i, mid in enumerate(self.ids)}
        return {mid: self.content[pos[mid]] for mid in ids if mid in pos}

    def save(self) -> None:
        # Atomic: write to a temp file then os.replace, so a concurrent reader / a crash mid-save
        # never sees a truncated index file. Write to a HANDLE (np.savez would append .npz to a path,
        # breaking the temp name).
        def _w(tmp: Path) -> None:
            with open(tmp, "wb") as f:
                np.savez(f, ids=np.array(self.ids, dtype=object),
                         content=self.content, struct=self.struct)
        _atomic_write_bytes(self._path(), _w)

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
        # Serializes ONLY the adaptive-ef-override searches (which mutate the shared set_ef); the
        # default-ef search path is lock-free. hnswlib supports concurrent add + query natively.
        self._ef_lock = threading.Lock()
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=self._capacity, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef)
        # Deterministic, contention-free internal behaviour under OUR threading (the engine write
        # lock serializes writers; hnswlib's own thread pool would otherwise fight for cores).
        try:
            self.index.set_num_threads(1)
        except Exception:
            pass
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
            try:
                self.index.set_num_threads(1)
            except Exception:
                pass
            if struct_p.exists():
                self.struct = np.load(struct_p).astype(np.float32)

    def _ensure_capacity(self) -> None:
        if len(self.ids) >= self._capacity:
            self._capacity *= 2
            self.index.resize_index(self._capacity)

    def add(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        self._ensure_capacity()
        label = len(self.ids)
        sv = _normalize(struct_vec if struct_vec is not None else np.zeros(self.struct_dim)).reshape(1, -1)
        # Publish struct, then ids, then insert into the ANN. struct-before-ids keeps
        # search_struct's len(ids) <= struct rows; ids-before-add_items keeps _search's labels
        # resolvable (knn_query only returns a label after add_items, by when self.ids[label] exists).
        self.struct = np.vstack([self.struct, sv]) if self.struct.size else sv
        self.ids = self.ids + [memory_id]
        self.index.add_items(_normalize(content_vec).reshape(1, -1), np.array([label]))

    def update(self, memory_id: str, content_vec: np.ndarray, struct_vec: Optional[np.ndarray] = None) -> None:
        if memory_id not in self.ids:
            return self.add(memory_id, content_vec, struct_vec)
        label = self.ids.index(memory_id)
        # hnswlib updates an existing element in place when re-added with the same label.
        self.index.add_items(_normalize(content_vec).reshape(1, -1), np.array([label]))
        if struct_vec is not None:
            self.struct[label] = _normalize(struct_vec)

    def search(self, query_vec: np.ndarray, k: int,
               allowed_ids: Optional[set[str]] = None,
               ef: Optional[int] = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        # Default-ef path: lock-free (hnswlib query is thread-safe). Adaptive ef-override mutates
        # the shared set_ef, so serialize ONLY that branch to keep concurrent searches correct.
        if ef is None or ef == self.ef:
            return self._search(query_vec, k, allowed_ids)
        with self._ef_lock:
            self.index.set_ef(max(int(ef), k))
            try:
                return self._search(query_vec, k, allowed_ids)
            finally:
                self.index.set_ef(self.ef)

    def _search(self, query_vec: np.ndarray, k: int,
                allowed_ids: Optional[set[str]] = None) -> list[tuple[str, float]]:
        ids = self.ids                      # snapshot (a concurrent add publishes a NEW list)
        if not ids:
            return []
        n = len(ids)
        k = min(k, n)
        q = _normalize(query_vec)
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            if not allowed:
                return []
            fetch = min(n, max(k, min(n, len(allowed) * 4)))
        else:
            fetch = k
        labels, distances = self.index.knn_query(q.reshape(1, -1), k=fetch)
        out = []
        for lab, dist in zip(labels[0], distances[0]):
            lab = int(lab)
            if lab >= n:                    # label not yet in the id snapshot (benign under add)
                continue
            mid = ids[lab]
            if allowed_ids is None or mid in allowed:
                out.append((mid, float(1.0 - dist)))  # cosine sim = 1 - dist
                if len(out) >= k:
                    return out
        if allowed_ids is not None and len(out) < min(k, len(allowed)):
            pos = {mid: i for i, mid in enumerate(ids)}
            labels = [pos[mid] for mid in allowed if mid in pos]
            if labels:
                items = self.index.get_items(labels)
                sims = np.asarray(items, dtype=np.float32) @ q
                order = np.argsort(-sims)[: min(k, len(labels))]
                out = [(ids[labels[i]], float(sims[i])) for i in order]
        return out[:k]

    def search_struct(self, query_struct: np.ndarray, k: int) -> list[tuple[str, float]]:
        ids, struct = self.ids, self.struct        # snapshot both refs; clamp to the common prefix
        n = min(len(ids), struct.shape[0])
        if n == 0:
            return []
        q = _normalize(query_struct)
        sims = struct[:n] @ q
        k = min(k, n)
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(ids[i], float(sims[i])) for i in idx]

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        pos = {mid: i for i, mid in enumerate(self.ids)}
        labels = [pos[mid] for mid in ids if mid in pos]
        if not labels:
            return {}
        items = self.index.get_items(labels)
        return {self.ids[lab]: np.asarray(v, dtype=np.float32) for lab, v in zip(labels, items)}

    def save(self) -> None:
        # Each file atomically replaced. True cross-file atomicity is impossible; the real
        # crash-recovery backstop is rebuild_index_from_store() (the index is a derived cache of
        # the substrate + SQLite source of truth). The engine write lock prevents add() during save.
        bin_p, meta_p, struct_p = self._paths()

        def _w_struct(tmp: Path) -> None:
            with open(tmp, "wb") as f:        # handle, not path (np.save appends .npy to paths)
                np.save(f, self.struct)

        _atomic_write_bytes(bin_p, lambda tmp: self.index.save_index(str(tmp)))
        _atomic_write_bytes(meta_p, lambda tmp: tmp.write_text(json.dumps(
            {"ids": self.ids, "dim": self.dim, "struct_dim": self.struct_dim})))
        _atomic_write_bytes(struct_p, _w_struct)

    def __len__(self) -> int:
        return len(self.ids)


class QuantizedVectorIndex(NumpyVectorIndex):
    """Exact-cosine numpy index with a quantized first stage (Layer 3c). Keeps the raw
    float32 vectors (inherited) for an exact refine pass, and a compact code array (int8
    SQ8 or 1-bit RaBitQ) for the cheap shortlist ranking. Age never enters the distance, so
    the flat recall-vs-age property is preserved exactly as in the parent."""

    def __init__(self, index_dir: Path, dim: int, struct_dim: int, *, kind: str = "rabitq",
                 refine: bool = True, refine_topn: int = 100):
        self.kind = kind
        self.refine = refine
        self.refine_topn = int(refine_topn)
        self._codes = None
        self._code_dim = 0
        self._codes_dirty = True
        self._rotation = None
        super().__init__(index_dir, dim, struct_dim)

    def add(self, memory_id, content_vec, struct_vec=None):
        super().add(memory_id, content_vec, struct_vec)
        self._codes_dirty = True

    def update(self, memory_id, content_vec, struct_vec=None):
        super().update(memory_id, content_vec, struct_vec)
        self._codes_dirty = True

    def _ensure_codes(self) -> None:
        from .optim import quantize as _q
        if not self._codes_dirty and self._codes is not None:
            return
        if self.content.shape[0] == 0:
            self._codes = None
        elif self.kind == "sq8":
            self._codes = _q.sq8_encode(self.content)
        else:
            self._code_dim = self.content.shape[1]
            self._rotation = _q.make_rotation(self._code_dim)
            self._codes, _ = _q.rabitq_encode(self.content, self._rotation)
        self._codes_dirty = False

    def _approx(self, qvec: np.ndarray) -> np.ndarray:
        from .optim import quantize as _q
        if self.kind == "sq8":
            return _q.sq8_scores(self._codes, qvec)
        qpacked, _ = _q.rabitq_encode(qvec, self._rotation)
        return _q.rabitq_cosine_estimate(_q.rabitq_hamming(self._codes, qpacked[0]),
                                         self._code_dim)

    def search(self, query_vec, k, allowed_ids=None, ef=None):
        if not self.ids:
            return []
        self._ensure_codes()
        if self._codes is None:
            return []
        if allowed_ids is not None:
            pos = np.array([i for i, mid in enumerate(self.ids) if mid in allowed_ids])
            if pos.size == 0:
                return []
        else:
            pos = np.arange(len(self.ids))
        approx = self._approx(query_vec)[pos]
        if self.refine:
            n_short = min(max(self.refine_topn, k), pos.size)
            local = np.argpartition(-approx, n_short - 1)[:n_short]
            sl = pos[local]
            q = _normalize(query_vec)
            sims = self.content[sl] @ q
            kk = min(k, sims.shape[0])
            top = np.argpartition(-sims, kk - 1)[:kk]
            top = top[np.argsort(-sims[top])]
            return [(self.ids[int(sl[i])], float(sims[i])) for i in top]
        kk = min(k, pos.size)
        top = np.argpartition(-approx, kk - 1)[:kk]
        top = top[np.argsort(-approx[top])]
        return [(self.ids[int(pos[i])], float(approx[i])) for i in top]


def make_vector_index(settings: Optional[Settings] = None) -> VectorIndex:
    settings = settings or get_settings()
    backend = settings.vector_backend

    quant = getattr(settings, "vector_quant", "none")
    if quant in ("sq8", "rabitq"):
        return QuantizedVectorIndex(
            settings.index_dir, settings.embed_dim, settings.struct_dim,
            kind=quant, refine=settings.quant_refine, refine_topn=settings.quant_refine_topn)

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
