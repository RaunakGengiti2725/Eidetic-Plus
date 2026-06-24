"""Tiny BM25 lexical retrieval channel (Okapi BM25), pure-Python, no new dependency.

The hybrid read path fuses dense (vector) + BM25 (lexical) + PPR (graph) + recency via
Reciprocal Rank Fusion. BM25 recovers exact-term matches that dense recall can miss
(names, codes, numbers), which is why hybrid reaches ~95% on single-hop-class retrieval.
The base class can build from a corpus in memory. PersistentBM25 stores the tokenized
corpus on disk and updates it on ingest, so query time does not rebuild the lexical
index from every in-scope record.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.docs: list[list[str]] = []
        self.tfs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._pos: dict[str, int] = {}

    def index(self, items: list[tuple[str, str]]) -> "BM25":
        """items = [(memory_id, text)]."""
        self.ids, self.docs, self.tfs, self.doc_lens, self.df = [], [], [], [], Counter()
        self._pos = {}
        for mid, text in items:
            toks = _tokenize(text)
            self._pos[mid] = len(self.ids)
            self.ids.append(mid)
            self.docs.append(toks)
            self.tfs.append(Counter(toks))
            self.doc_lens.append(len(toks))
            for t in set(toks):
                self.df[t] += 1
        self._refresh_stats()
        return self

    def _refresh_stats(self) -> None:
        n = len(self.docs)
        self.avgdl = (sum(self.doc_lens) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def add_or_update(self, memory_id: str, text: str) -> bool:
        """Add or replace one document. Returns True when the index changed."""
        toks = _tokenize(text)
        pos = self._pos.get(memory_id)
        if pos is not None and self.docs[pos] == toks:
            return False
        if pos is None:
            self._pos[memory_id] = len(self.ids)
            self.ids.append(memory_id)
            self.docs.append(toks)
            self.tfs.append(Counter(toks))
            self.doc_lens.append(len(toks))
            for t in set(toks):
                self.df[t] += 1
        else:
            old_terms = set(self.docs[pos])
            for t in old_terms:
                self.df[t] -= 1
                if self.df[t] <= 0:
                    del self.df[t]
            self.docs[pos] = toks
            self.tfs[pos] = Counter(toks)
            self.doc_lens[pos] = len(toks)
            for t in set(toks):
                self.df[t] += 1
        self._refresh_stats()
        return True

    def has(self, memory_id: str) -> bool:
        return memory_id in self._pos

    def ensure_indexed(self, items: Iterable[tuple[str, str]], *, update_existing: bool = False) -> bool:
        """Backfill missing documents. Existing docs update only when requested."""
        changed = False
        for mid, text in items:
            if update_existing or not self.has(mid):
                changed = self.add_or_update(mid, text) or changed
        return changed

    def _search_positions(
        self,
        query_terms: list[str],
        positions: Iterable[int],
        k: int,
        *,
        scoped_stats: bool = False,
    ) -> list[tuple[str, float]]:
        pos_list = list(positions)
        if scoped_stats:
            n = len(pos_list)
            avgdl = (sum(self.doc_lens[pos] for pos in pos_list) / n) if n else 0.0
            df = {t: 0 for t in query_terms}
            for pos in pos_list:
                terms = set(self.docs[pos])
                for t in query_terms:
                    if t in terms:
                        df[t] += 1
            idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in query_terms}
        else:
            avgdl = self.avgdl
            idf = self.idf
        scores: list[tuple[str, float]] = []
        for pos in pos_list:
            if pos < 0 or pos >= len(self.ids):
                continue
            mid, tf, dl = self.ids[pos], self.tfs[pos], self.doc_lens[pos]
            if dl == 0:
                continue
            s = 0.0
            for t in query_terms:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (avgdl or 1))
                s += idf[t] * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((mid, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def search(self, query: str, k: int, allowed_ids: Optional[set[str]] = None) -> list[tuple[str, float]]:
        if not self.docs:
            return []
        q = [t for t in _tokenize(query) if t in self.idf]
        if not q:
            return []
        if allowed_ids is not None:
            positions = sorted(self._pos[mid] for mid in allowed_ids if mid in self._pos)
            return self._search_positions(q, positions, k, scoped_stats=True)
        return self._search_positions(q, range(len(self.ids)), k)


class PersistentBM25(BM25):
    """A disk-backed BM25 corpus that is updated on ingest and reused at query time."""

    def __init__(self, path: Path, k1: float = 1.5, b: float = 0.75):
        super().__init__(k1=k1, b=b)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.ids = [str(x) for x in data.get("ids", [])]
        self.docs = [[str(t) for t in doc] for doc in data.get("docs", [])]
        if len(self.docs) != len(self.ids):
            self.ids, self.docs = [], []
        self._pos = {mid: i for i, mid in enumerate(self.ids)}
        self.tfs = [Counter(doc) for doc in self.docs]
        self.doc_lens = [len(doc) for doc in self.docs]
        self.df = Counter()
        for doc in self.docs:
            for t in set(doc):
                self.df[t] += 1
        self._refresh_stats()

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"ids": self.ids, "docs": self.docs}))
        tmp.replace(self.path)
