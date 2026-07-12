"""The engine: wires all seven components into the wake and sleep paths.

wake (per input):   ingest -> immutable store -> embed -> salience -> graph extract
                    -> structure code -> index. And ask -> retrieve -> verify ->
                    cite -> reconsolidate (strengthen what was used).
sleep (scheduled):  consolidate -> dedup/pattern-separation -> verified semantic
                    summaries -> FSRS index-priority decay. Never deletes raw.
"""
from __future__ import annotations

import logging
import re
import tempfile
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from datetime import datetime

from . import fsrs, preferences, salience as salience_mod, structure_code as sc
from .config import Settings, get_settings
from .dashscope_client import DashScopeClient, chunk_text, get_client
from .events import EventRecord, event_aliases_from_text, normalize_dates
from .graph import KnowledgeGraph
from .ingestion import IngestInput, from_bytes, from_file, from_text
from .memory_types import classify_record
from .brain import BrainEventLog, build_evidence_packets
from .models import (Answer, AnswerStatus, BrainEvent, BrainEventType, EvidencePacket,
                     MemoryRecord, Modality, NLILabel, RecallTrace, RetrievalCandidate,
                     Scope, now)
from .activation import ActivationField
from .reflex import MemoryPacket
from .reflex_activation import build_memory_packet
from .reflex_index import ReflexIndex
from .retrieval import Retriever
from .semantic_cache import SemanticCache
from .store import RecordStore
from .substrate import make_substrate, sha256_hex
from .vector_index import make_vector_index

# Synaptic-tagging-and-capture window: a salient event up-weights temporally adjacent
# memories within this window (dossier 6.7 -> ~1 hour for the biological analog).
TAG_CAPTURE_WINDOW_SEC = 3600.0
TAG_CAPTURE_SALIENCE = 0.7  # only events at/above this salience tag their neighbors
_FALSE_PREMISE_STOP_TERMS = {
    "about", "after", "before", "between", "did", "does", "doing", "for", "from",
    "had", "has", "have", "how", "into", "is", "leave", "leaving", "left", "over",
    "quit", "the", "then", "there", "this", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "work", "worked", "working",
}
_FALSE_PREMISE_GENERIC_TERMS = {
    "company", "home", "job", "office", "place", "project", "school", "team",
    "thing", "work",
}

_log = logging.getLogger("eidetic.engine")


class Engine:
    def __init__(self, settings: Optional[Settings] = None, client: Optional[DashScopeClient] = None):
        self.settings = settings or get_settings()
        self.client = client or get_client()
        # F0 concurrency safety: a SINGLE reentrant write lock serializes every mutation of the
        # shared in-memory index / BM25 / graph-write tail / caches. Reads (search) stay lock-free.
        # INVARIANT: never hold this lock across a model call (that would deadlock the rate governor).
        self._write_lock = threading.RLock()
        # S1 deferred re-embed queue: confirmed-citation re-embeds pushed off the answer path and
        # drained on the idle/sleep cadence (the embed runs OFF the write lock).
        self._reembed_queue: set[str] = set()
        self._reembed_lock = threading.Lock()
        # Last COMPLETED traced ask PER SCOPE, published for cross-thread introspection
        # surfaces; the retriever's thread-local remains the in-flight isolation mechanism.
        # Scope-keyed and bounded: a global "last trace" would leak query text + memory ids
        # across the namespace boundary the whole product guarantees.
        self._trace_snapshots: "OrderedDict[str, RecallTrace]" = OrderedDict()
        self._trace_snapshot_lock = threading.Lock()
        self._ingest_since_save = 0          # S2 debounced-save counter (guarded by the write lock)
        self.substrate = make_substrate(self.settings)
        self.store = RecordStore(self.settings.sqlite_path)
        self.index = make_vector_index(self.settings)
        self.graph = KnowledgeGraph(
            self.store, deterministic_conflicts=self.settings.conflict_resolver_enabled
        )
        self.retriever = Retriever(
            self.store, self.index, self.graph, self.substrate, self.client, self.settings
        )
        self.cache = SemanticCache(
            self.settings.cache_cosine, adaptive=self.settings.semantic_cache_adaptive
        )
        # Dreaming engine: predictive pre-fetch cache + a bounded query log (token-free).
        from .dreaming.prefetch import PrefetchCache
        self.prefetch = PrefetchCache(self.settings.dream_prefetch_threshold)
        self._query_log: list[tuple[str, "np.ndarray"]] = []
        # Dev-only feedback replay buffer (the spine of the idle learning cadence).
        from .feedback import FeedbackBuffer
        self.feedback = FeedbackBuffer(self.settings.data_dir / "feedback.sqlite")
        # EPISTEMIC ORGANISM: the knowledge map (KNOWN/UNKNOWN/CONTESTED, derived layer in
        # its own sqlite) + the research agenda (frontier queue). Both passive on the wake
        # paths -- hooks only append to these side stores; retrieval/proof never read them.
        self.knowledge_map_store = None
        self.research_agenda = None
        if self.settings.epistemic_map_enabled:
            from .epistemic.map import KnowledgeMap
            self.knowledge_map_store = KnowledgeMap(
                self.settings.data_dir / "epistemic_map.sqlite")
        if self.settings.autoresearch_enabled:
            from .autoresearch.agenda import ResearchAgenda
            self.research_agenda = ResearchAgenda(
                self.settings.data_dir / "research_agenda.sqlite")
        # In-flight passive-hook threads (see _epistemic_after_ask): tracked so close()
        # can JOIN them before a caller tears down the data directory (tempdir evals).
        self._epistemic_threads: set = set()
        self._epistemic_threads_lock = threading.Lock()
        # Prospective memory: a first-order Markov model of query-signature transitions.
        from .optim.markov import MarkovPrefetcher
        self.markov = MarkovPrefetcher()
        self._last_query_text = ""
        # Connected Brain Loop: the single in-memory improvement stream. Constructing an empty
        # ring is free; emission is gated on BRAIN_EVENTS so baseline behavior is unchanged.
        self.brain_log = BrainEventLog()
        # Channel-win ledger (Phase 3): which channels surfaced the confirmed source per verified
        # answer, keyed BY NAMESPACE so brain_health_score(scope=...) never mixes activity across
        # scopes. In-memory counter, only written under BRAIN_EVENTS; never feeds a learner.
        self._channel_wins: dict[str, dict[str, int]] = {}
        # Track 1 Reflex Recall: a derived inverted index (entity/term -> memory_ids) + a small
        # per-namespace hot working set. Maintained ONLY when REFLEX_RECALL is on, so the flag-off
        # write path is byte-identical. Built once from the store at construction when enabled, then
        # kept current incrementally under the same write lock that guards the vector index --
        # text terms at ingest/ingest_many, and extracted entities at consolidate_pending (where
        # the fast-path record's entities are first populated). rebuild_from_store recovers it.
        self.reflex_index = ReflexIndex()
        self._hotset: dict[str, "deque"] = {}
        self._hotset_lock = threading.Lock()
        # Track 9 Flow: one per-namespace ActivationField (the shared working-memory substrate) +
        # per-namespace turn locks so a turn's decay+inject+spread sequence is atomic (never held
        # across the reader/model call). Built ONLY when flow is on -> flag-off byte-identical.
        self.activation = (ActivationField(decay=self.settings.flow_decay,
                                           floor=self.settings.flow_floor, cap=self.settings.flow_cap)
                           if self.settings.flow_activation_enabled else None)
        self._flow_turn_locks: dict[str, threading.Lock] = {}
        self._flow_locks_guard = threading.Lock()
        # Track 2 perfect sync: a monotonic per-NAMESPACE memory version. Bumped under the write
        # lock on every content write; the answer cache tags entries with it so a write makes prior
        # entries in that namespace unreachable (no stale-truth hits). Namespace-grained because a
        # read sees all records visible_to it in the namespace, so a sub-scope write must invalidate
        # a namespace-wide query's entry too.
        self._ns_versions: dict[str, int] = {}
        if self.settings.reflex_recall_enabled:
            self.reflex_index.rebuild_from_store(self.store)
        # Track 2.3: surface governor 429/backoff as a RATE_LIMITED BrainEvent. The hook runs on the
        # worker thread off-lock; it only appends an event (no re-entry into the governor).
        _gov = getattr(self.client, "_governor", None)
        if _gov is not None:
            _gov.on_rate_limit = self._on_rate_limit
        # One shared wake/sleep/idle/repair coordinator (Phase 1) -- API and MCP route through it.
        from .lifecycle import LifecycleController
        self.lifecycle = LifecycleController(self)

    def _brain(self, etype: BrainEventType, *, namespace: str = "default",
               memory_ids: Optional[list] = None, **payload) -> None:
        """Emit one BrainEvent -- a no-op unless BRAIN_EVENTS is on. Best-effort: a logging
        failure never breaks the wake/sleep path."""
        if not self.settings.brain_events_enabled:
            return
        try:
            self.brain_log.emit(BrainEvent(type=etype, namespace=namespace,
                                           memory_ids=list(memory_ids or []), payload=payload))
        except Exception as e:        # event emission is non-critical, but log rather than swallow
            _log.debug("brain-event emit failed: %s", e)

    def _ns_version(self, namespace: str) -> int:
        """Current memory version for a namespace (0 until the first content write)."""
        return self._ns_versions.get(namespace, 0)

    def _bump_ns_version(self, namespace: str) -> None:
        """Advance a namespace's memory version. MUST be called under self._write_lock, on content
        writes only (new records, extracted facts) -- not on FSRS/re-embed reconsolidation, which
        does not change what is true, so the answer cache should survive across reads."""
        self._ns_versions[namespace] = self._ns_versions.get(namespace, 0) + 1

    def _on_rate_limit(self, info: dict) -> None:
        """Governor hook: a real model call hit a retryable 429 and backed off. Emit RATE_LIMITED
        (no-op unless BRAIN_EVENTS is on). Best-effort, runs on the calling worker thread."""
        self._brain(BrainEventType.RATE_LIMITED, **{k: info.get(k) for k in
                                                    ("attempt", "sleep_s", "retry_after")})

    def _degraded(self, where: str, exc: Exception) -> None:
        """Record a best-effort hot-path failure WITHOUT silently swallowing it. A ModelCallError
        (key / quota / model) is logged at WARNING so a degraded run is never mistaken for a healthy
        one; other narrow errors log at debug. Best-effort behavior continues either way."""
        from .dashscope_client import ModelCallError
        if isinstance(exc, ModelCallError):
            _log.warning("degraded[%s]: a model call failed and was downgraded (best-effort): %s",
                         where, exc)
        else:
            _log.debug("degraded[%s]: %s", where, exc)

    # ---- wake: write path -------------------------------------------------
    def ingest(
        self,
        item: IngestInput,
        *,
        valid_at: Optional[float] = None,
        extract_graph: bool = True,
        scope: Optional[Scope] = None,
        consolidate_now: bool = True,
    ) -> MemoryRecord:
        """Ingest one item. With consolidate_now=False this is the benchmark WRITE PATH:
        LLM-free (append + embed only, <50ms target) -- fact extraction, importance, graph,
        and visual extraction are deferred to consolidate_pending() (async, off the hot path)."""
        valid_at = now() if valid_at is None else valid_at
        scope = scope or Scope()

        # SHA-256 dedup BEFORE embedding (cost control + provenance). Dedup is PER-SCOPE and
        # per valid_at: identical text repeated in a later session is still a distinct memory event
        # for count/sum recall, while the immutable raw bytes remain shared by content_hash.
        h = sha256_hex(item.raw_bytes)
        for existing in self.store.records_by_hash(h, scope):
            if abs(float(existing.valid_at or 0.0) - float(valid_at or 0.0)) <= 1e-6:
                return existing

        # Embed (real; allowed on the write path) before committing immutable bytes. If a
        # missing key/quota/model failure happens here, public MCP callers get a loud error and
        # no orphan raw blob is left in the write-once substrate.
        content_vec = self.client.embed_text(item.text)

        triples: list[dict[str, str]] = []
        entities: list[str] = []
        affect_meta: dict = {}
        if consolidate_now:
            sal = salience_mod.score(item.text, content_vec, self.index, self.client,
                                     self.store, scope)
            # Graph extraction (real qwen-plus); optional for cheap bulk ingest.
            if extract_graph and item.text.strip():
                triples.extend(self.client.extract_edges(item.text))
            # Vision FEEDS the graph: images/diagrams/tables become entities+edges.
            if extract_graph and item.modality in (Modality.IMAGE, Modality.VIDEO):
                triples.extend(self._visual_triples(item.raw_bytes, item.modality))
            # Affect-modulated salience (Phase 3, gated): one real qwen-flash call for
            # {importance, arousal, valence} + deterministic emphasis cues -> static (age-free)
            # salience. Replaces the salience field used for the bounded retrieval boost / S0.
            if self.settings.affect_salience_enabled:
                s = self.settings
                aff = self.client.score_affect(item.text)
                emph = salience_mod.emphasis_score(item.text)
                s_aff = salience_mod.affect_salience(
                    aff["arousal"], aff["importance"], sal.surprise, emph, 0.0,
                    w_arousal=s.affect_w_arousal, w_importance=s.affect_w_importance,
                    w_surprise=s.affect_w_surprise, w_emphasis=s.affect_w_emphasis,
                    w_helpful=s.affect_w_helpful)
                sal = salience_mod.Salience(surprise=sal.surprise, importance=aff["importance"],
                                            salience=s_aff)
                affect_meta = {"arousal": aff["arousal"], "valence": aff["valence"],
                               "emphasis": emph}
        else:
            # LLM-FREE write path: surprise from embedding distance only (no LLM call),
            # importance deferred; facts/visual extracted later by consolidate_pending().
            surprise = salience_mod.compute_surprise(content_vec, self.index, self.store, scope)
            sal = salience_mod.Salience(surprise=surprise, importance=0.5,
                                        salience=max(0.0, min(1.0, 0.45 * surprise + 0.275)))

        seen: set[str] = set()
        for t in triples:
            for e in (t["src"], t["dst"]):
                if e.lower() not in seen:
                    seen.add(e.lower())
                    entities.append(e)

        content_hash = h
        raw_uri = self.substrate.uri_for(h)
        record = MemoryRecord(
            content_hash=content_hash, modality=item.modality, raw_uri=raw_uri,
            raw_bytes_len=len(item.raw_bytes), text=item.text, is_described=item.is_described,
            source=item.source, scope=scope, valid_at=valid_at, entities=entities,
            surprise=sal.surprise, importance=sal.importance, salience=sal.salience,
            fsrs=fsrs.init_state(
                sal.importance, sal.surprise, valid_at,
                salience=(sal.salience if self.settings.affect_salience_enabled else None),
                gamma=self.settings.salience_gamma),
            metadata={"pending_consolidation": not consolidate_now, **affect_meta},
        )

        # Add extracted facts with bi-temporal contradiction handling (scoped).
        for t in triples:
            self.graph.add_fact(
                t["src"], t["relation"], t["dst"], fact=t["fact"],
                source_memory_id=record.memory_id, valid_at=valid_at, scope=scope,
            )

        # Structure code (Component 3): metadata + graph position + relational roles.
        gfeat: Optional[dict] = None
        if entities:
            feats = self.graph.node_features(valid_at, scope)
            agg = [feats[e.lower()] for e in entities if e.lower() in feats] if feats else []
            gfeat = {"relations": [t["relation"] for t in triples]}
            if agg:
                gfeat["ppr"] = float(np.mean([a["ppr"] for a in agg]))
                gfeat["degree"] = float(np.mean([a["degree"] for a in agg]))
        struct_vec = sc.build_structure_code(record, self.settings.struct_dim, gfeat)

        # MIRIX memory typing (Phase 4): tag the record's role from cheap deterministic signals
        # (no model call) so the retrieval coordinator can route by type. Gated; metadata-only.
        if self.settings.memory_typing_enabled:
            record.metadata["type"] = classify_record(record).value

        content_hash, raw_uri = self.substrate.put(item.raw_bytes)
        record.content_hash = content_hash
        record.raw_uri = raw_uri

        # Index-write tail under the write lock (content_vec / struct_vec were computed above, OFF
        # the lock -> no model call is held here).
        with self._write_lock:
            self.index.add(record.memory_id, content_vec, struct_vec)
            self._maybe_save_index()
            self.store.upsert_record(record)
            self.retriever.index_lexical(record)
            if self.settings.reflex_recall_enabled:
                self.reflex_index.add_record(record)
            self._bump_ns_version(scope.namespace)
        self._flow_prime_ingest(scope.namespace, record.memory_id)   # no-op unless FLOW_PRIME_INGEST>0

        # Synaptic tagging and capture: a salient event up-weights temporally adjacent
        # in-scope memories (FSRS priority only -- never the ranking score). Skipped on the
        # LLM-free fast path (its O(N) scan runs during consolidation instead).
        if consolidate_now and sal.salience >= TAG_CAPTURE_SALIENCE:
            self._tag_and_capture(record, scope)
        self._brain(BrainEventType.MEMORY_INGESTED, namespace=scope.namespace,
                    memory_ids=[record.memory_id], modality=record.modality.value,
                    pending=not consolidate_now)
        return self.lifecycle.after_ingest(record, scope)

    def _visual_triples(self, raw_bytes: bytes, modality: Modality) -> list[dict[str, str]]:
        """Run real visual graph extraction by writing the raw bytes to a temp file."""
        suffix = ".png" if modality == Modality.IMAGE else ".mp4"
        tmp_dir = self.settings.data_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=suffix, delete=False) as f:
            f.write(raw_bytes)
            path = f.name
        try:
            if modality == Modality.IMAGE:
                return self.client.extract_visual_graph(path)
            return []  # video keyframe extraction handled by description path for now
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass

    def _reinforce_verified_helpful(self, rec: MemoryRecord) -> None:
        """Phase 4: a confirmed (NLI-entailed) citation increments this memory's verified-helpful
        count and, when affect salience is on, folds a BOUNDED usage signal into its static
        salience (via affect_w_helpful, default 0). The count is plain state; it only touches the
        ranking score through the bounded, age-free affect salience -- audited age-stratified."""
        rec.verified_helpful_count = int(getattr(rec, "verified_helpful_count", 0)) + 1
        if not self.settings.affect_salience_enabled:
            return
        s = self.settings
        vh = salience_mod.verified_helpful_signal(rec.verified_helpful_count, s.verified_helpful_cap)
        rec.salience = salience_mod.affect_salience(
            float(rec.metadata.get("arousal", 0.3)), rec.importance, rec.surprise,
            float(rec.metadata.get("emphasis", 0.0)), vh,
            w_arousal=s.affect_w_arousal, w_importance=s.affect_w_importance,
            w_surprise=s.affect_w_surprise, w_emphasis=s.affect_w_emphasis,
            w_helpful=s.affect_w_helpful)

    def _maybe_save_index(self) -> None:
        """Debounced index save (S2). INDEX_SAVE_DEBOUNCE=1 (default) saves every time (baseline);
        higher amortizes saves. Must be called under the write lock. A lost index is recoverable
        via rebuild_index_from_store (the index is a derived cache of the source of truth)."""
        self._ingest_since_save += 1
        if self._ingest_since_save >= max(1, self.settings.index_save_debounce):
            self.index.save()
            self._ingest_since_save = 0

    def flush_index(self) -> None:
        """Force-persist the index (S2): call on shutdown / after a debounced batch."""
        with self._write_lock:
            self.index.save()
            self.retriever.save_lexical()
            self._ingest_since_save = 0

    def rebuild_index_from_store(self) -> dict:
        """Rebuild the vector index from the SOURCE OF TRUTH (substrate + SQLite records). The index
        is a derived cache; this recovers a debounced/lost/corrupt index with no data loss. Re-embeds
        each record's text (governed, OFF the lock); rebuilds UNDER the lock. Raw store untouched."""
        recs = list(self.store.all_records(None))
        texts = [r.text or r.summary or "" for r in recs]
        vecs = (self.client.embed_texts(texts) if texts
                else np.zeros((0, self.settings.embed_dim), np.float32))
        with self._write_lock:
            for f in list(self.settings.index_dir.glob("*")):
                if f.name.startswith(("numpy_index", "hnsw", "quant")):
                    try:
                        f.unlink()
                    except OSError:
                        pass
            self.index = make_vector_index(self.settings)     # fresh empty index (files removed)
            self.retriever.index = self.index
            for r, v in zip(recs, vecs):
                self.index.add(r.memory_id, v, sc.build_structure_code(r, self.settings.struct_dim))
            self.index.save()
            self._ingest_since_save = 0
            if self.settings.reflex_recall_enabled:
                self.reflex_index.rebuild_from_store(self.store)
            # Claims hygiene pass (wave 0.1 migration): collapse duplicate claim rows
            # minted under historic random ids onto their deterministic identity keys.
            # SQL-only, no embedding cost, a no-op on already-clean stores. UNDER the
            # write lock: a concurrent ingest's claim INSERT must not interleave with
            # the delete+reinsert scan (the scanned snapshot would clobber newer rows).
            claims_dedupe = self.store.dedupe_claims(None)
        return {"rebuilt": len(recs), "claims_dedupe": claims_dedupe}

    def clear_namespace(self, namespace: str) -> dict:
        """Administrative reset of one namespace's mutable memory state.

        Deletes only the forgettable SQLite index/state rows for the namespace; immutable substrate
        blobs remain content-addressed and untouched. Rebuildable in-memory surfaces are invalidated
        so repeated benchmark runs start from the same empty scoped state.
        """
        ns = namespace or "default"
        with self._write_lock:
            removed = self.store.clear_namespace(ns)
            self.cache.clear()
            self._bump_ns_version(ns)
            self.feedback.clear(ns)
            if self.reflex_index.built or self.settings.reflex_recall_enabled:
                self.reflex_index.rebuild_from_store(self.store)
            with self._hotset_lock:
                self._hotset.pop(ns, None)
            if self.activation is not None:
                self.activation.clear_namespace(ns)

            surfaces_reset = False
            if self.store.count(None) == 0:
                for f in list(self.settings.index_dir.glob("*")):
                    if f.name.startswith(("numpy_index", "hnsw", "quant", "bm25_index")):
                        try:
                            f.unlink()
                        except OSError:
                            pass
                self.index = make_vector_index(self.settings)
                self.retriever.index = self.index
                self.index.save()
                if self.settings.persistent_bm25_enabled:
                    self.retriever.bm25.index([])
                    self.retriever.bm25.save()
                self._ingest_since_save = 0
                surfaces_reset = True
        return {**removed, "surfaces_reset": surfaces_reset}

    def ingest_many(self, items, *, valid_at: Optional[float] = None,
                    scope: Optional[Scope] = None) -> list:
        """Batched bulk ingest (S2, LLM-free fast path): embed ALL texts in batches (N/10 round
        trips instead of N) and write every record under ONE lock acquisition. Per-scope dedup +
        substrate put. Records are marked pending_consolidation -> run consolidate_pending for full
        extraction. Returns the records (existing ones for duplicates)."""
        items = list(items)
        if not items:
            return []
        scope = scope or Scope()
        va = now() if valid_at is None else valid_at
        prepared = []                       # (item, content_hash, raw_uri, existing_or_None)
        # In-batch dedup: the store check below can't see records this call hasn't written
        # yet, so two identical items in ONE batch used to become two records. The first
        # occurrence wins; later ones resolve to its record after the write loop.
        first_occurrence: dict[str, int] = {}
        batch_dup_of: dict[int, int] = {}
        for idx, item in enumerate(items):
            h = sha256_hex(item.raw_bytes)
            existing = self.store.get_by_hash(h, scope)
            if existing is not None:
                prepared.append((item, None, None, existing))
            elif h in first_occurrence:
                batch_dup_of[idx] = first_occurrence[h]
                prepared.append((item, None, None, None))
            else:
                first_occurrence[h] = idx
                ch, uri = self.substrate.put(item.raw_bytes)
                prepared.append((item, ch, uri, None))
        new_idx = [i for i, p in enumerate(prepared) if p[3] is None and i not in batch_dup_of]
        texts = [prepared[i][0].text for i in new_idx]
        vecs = (self.client.embed_texts(texts) if texts          # batched embed, OFF the lock
                else np.zeros((0, self.settings.embed_dim), np.float32))
        vec_by_i = {new_idx[k]: vecs[k] for k in range(len(new_idx))}
        out: list = []
        with self._write_lock:
            for i, (item, ch, uri, existing) in enumerate(prepared):
                if existing is not None:
                    out.append(existing)
                    continue
                if i in batch_dup_of:
                    # duplicate WITHIN this batch: resolve to the first occurrence's record
                    # (already appended -- first_occurrence index precedes i)
                    out.append(out[batch_dup_of[i]])
                    continue
                content_vec = vec_by_i[i]
                surprise = salience_mod.compute_surprise(content_vec, self.index, self.store, scope)
                sal = salience_mod.Salience(surprise=surprise, importance=0.5,
                                            salience=max(0.0, min(1.0, 0.45 * surprise + 0.275)))
                rec = MemoryRecord(
                    content_hash=ch, modality=item.modality, raw_uri=uri,
                    raw_bytes_len=len(item.raw_bytes), text=item.text,
                    is_described=item.is_described, source=item.source, scope=scope, valid_at=va,
                    surprise=sal.surprise, importance=sal.importance, salience=sal.salience,
                    fsrs=fsrs.init_state(sal.importance, sal.surprise, va),
                    metadata={"pending_consolidation": True})
                self.index.add(rec.memory_id, content_vec,
                               sc.build_structure_code(rec, self.settings.struct_dim))
                self.store.upsert_record(rec)
                self.retriever.index_lexical(rec, save=False)
                if self.settings.reflex_recall_enabled:
                    self.reflex_index.add_record(rec)
                out.append(rec)
            if new_idx:                       # only a genuine new write changes what is true
                self._bump_ns_version(scope.namespace)
            self.index.save()
            self.retriever.save_lexical()
        if new_idx:
            for rec in out:
                if rec.metadata.get("pending_consolidation"):
                    self.lifecycle.after_ingest(rec, scope)
                    break
        return out

    def _enqueue_reembed(self, memory_ids) -> None:
        """S1: queue confirmed-citation memory_ids for a deferred re-embed (drained on idle/sleep)."""
        ids = list(memory_ids)
        with self._reembed_lock:
            self._reembed_queue.update(ids)
        self._brain(BrainEventType.REEMBED_DEFERRED, memory_ids=ids, queued=len(ids))

    def drain_reembed_queue(self, *, max_items: int = 256) -> dict:
        """Drain the deferred re-embed queue (S1): embed OFF the write lock, then apply the index
        updates UNDER the lock. Records superseded/forgotten before the drain are skipped. Idempotent
        and safe to call from the idle/sleep cadence."""
        with self._reembed_lock:
            ids = list(self._reembed_queue)[:max_items]
            self._reembed_queue.difference_update(ids)
        if not ids:
            return {"reembedded": 0}
        embedded: dict = {}
        for mid in ids:
            rec = self.store.get_record(mid)
            if rec is None:                          # superseded / forgotten -> skip
                continue
            try:
                embedded[mid] = self.client.embed_text(rec.text)   # model call, OFF the lock
            except Exception as e:
                self._degraded("drain-reembed", e)
        with self._write_lock:
            for mid, vec in embedded.items():
                self.index.update(mid, vec)
            if embedded:
                self.index.save()
        return {"reembedded": len(embedded)}

    def _tag_and_capture(self, record: MemoryRecord, scope: Scope) -> int:
        """Up-weight retention of memories temporally adjacent to a salient event. Store mutations
        (FSRS reinforce + upsert) run under the write lock; no model call is involved."""
        tagged = 0
        # S2: windowed query (O(window)) instead of an O(store) full scan.
        lo, hi = record.valid_at - TAG_CAPTURE_WINDOW_SEC, record.valid_at + TAG_CAPTURE_WINDOW_SEC
        with self._write_lock:
            for r in self.store.records_in_time_range(lo, hi, scope):
                if r.memory_id == record.memory_id:
                    continue
                if abs(r.valid_at - record.valid_at) <= TAG_CAPTURE_WINDOW_SEC:
                    fsrs.reinforce(r.fsrs, importance=0.4 * record.salience, at=record.valid_at)
                    self.store.upsert_record(r)
                    tagged += 1
        return tagged

    # convenience wrappers
    def ingest_text(self, text: str, *, source: str = "user", valid_at: Optional[float] = None,
                    extract_graph: bool = True, scope: Optional[Scope] = None,
                    segment: bool = False, consolidate_now: bool = True) -> MemoryRecord:
        """Ingest text. With segment=True, long inputs are split at Bayesian-surprise
        boundaries into separate episodes (EM-LLM); returns the FIRST episode's record.
        consolidate_now=False uses the LLM-free fast write path (benchmark mode)."""
        if segment:
            episodes = salience_mod.segment_by_surprise(text, self.client)
            if len(episodes) > 1:
                recs = [self.ingest(from_text(ep, source), valid_at=valid_at,
                                    extract_graph=extract_graph, scope=scope,
                                    consolidate_now=consolidate_now) for ep in episodes]
                return recs[0]
        rec = self.ingest(from_text(text, source), valid_at=valid_at,
                          extract_graph=extract_graph, scope=scope,
                          consolidate_now=consolidate_now)
        if (getattr(self.settings, "problem_extract_enabled", False)
                and source != "problem"):
            from . import problems as _problems
            _problems.apply_extracted_signals(self, rec, scope=scope)
        return rec

    def ingest_file(self, path: str, *, source: Optional[str] = None,
                    valid_at: Optional[float] = None, extract_graph: bool = True,
                    scope: Optional[Scope] = None, consolidate_now: bool = True) -> MemoryRecord:
        return self.ingest(from_file(path, self.client, source), valid_at=valid_at,
                           extract_graph=extract_graph, scope=scope, consolidate_now=consolidate_now)

    def ingest_bytes(self, data: bytes, filename: str, *, source: Optional[str] = None,
                     valid_at: Optional[float] = None, extract_graph: bool = True,
                     scope: Optional[Scope] = None, consolidate_now: bool = True) -> MemoryRecord:
        return self.ingest(from_bytes(data, filename, self.client, source), valid_at=valid_at,
                           extract_graph=extract_graph, scope=scope, consolidate_now=consolidate_now)

    # ---- Track 9 Flow hub: one writer (commit), many readers (snapshot) ----
    def _flow_turn_lock(self, namespace: str) -> "threading.Lock":
        with self._flow_locks_guard:
            lk = self._flow_turn_locks.get(namespace)
            if lk is None:
                lk = threading.Lock()
                self._flow_turn_locks[namespace] = lk
            return lk

    def _flow_snapshot(self, namespace: str) -> Optional[dict]:
        """The current activation map for a namespace (None when flow is off). Every recall surface
        reads this; it never mutates the field."""
        if self.activation is None:
            return None
        return self.activation.snapshot(namespace)

    def _salience_of(self, memory_id: str) -> float:
        """Access-time salience in [0,1] for salience-modulated decay: static importance + bounded
        verified-helpful usage. NEVER reads FSRS retrievability/priority (those decay with
        time-since-review and would smuggle a memory-age signal into the decay rate)."""
        rec = self.store.get_record(memory_id)
        if rec is None:
            return 0.0
        return max(0.0, min(1.0, 0.6 * rec.importance + 0.1 * rec.verified_helpful_count))

    def _flow_begin_turn(self, namespace: str, query: str, *, scope: Scope,
                         as_of: Optional[float] = None) -> None:
        """Start a turn: one salience-modulated decay, then an optional weak query-entity prime.
        The decay+prime sequence is atomic under the per-namespace turn lock (never held across a
        model call). No-op when flow is off."""
        if self.activation is None:
            return
        # Salience-modulated decay reads each active id's static salience. Precompute the map HERE
        # (off the ActivationField lock) and pass an in-memory dict.get to decay -- never a callable
        # that does a per-id store.get_record while the field's global lock is held (that would
        # serialize every namespace on N SQLite reads). NOTE: FLOW_SALIENCE_DECAY still costs
        # O(active-field) reads per turn, so it must NOT default on without an inject-time salience
        # snapshot; default 0 keeps this path cold.
        sal = None
        if self.settings.flow_salience_decay:
            sal_map = {mid: self._salience_of(mid) for mid in self.activation.snapshot(namespace)}
            sal = sal_map.get
        with self._flow_turn_lock(namespace):
            self.activation.decay(namespace, factor=self.settings.flow_decay, salience=sal)
            amt = self.settings.flow_prime_query
            if amt > 0.0:
                self.reflex_index.ensure_built(self.store)
                from .events import parse_query
                parsed = parse_query(query, as_of)
                from .reflex_index import tokenize
                seeds = self.reflex_index.seeds(namespace, entities=parsed.get("entities", []),
                                                terms=tokenize(query))
                if seeds:
                    self.activation.inject(namespace, list(seeds)[:self.settings.flow_seed_topk], amount=amt)
                    self._brain(BrainEventType.FLOW_PRIMED, namespace=namespace, kind="query")

    def _flow_commit_recall(self, namespace: str, confirmed: list, scope: Scope,
                            read_at: Optional[float]) -> None:
        """Write the turn's confirmed recall into the field: inject confirmed ids, then a one-hop
        CO_ACTIVATED spread to their neighbors (weaker). Atomic under the turn lock. The field is
        ephemeral and one-directional -- it NEVER writes back to the store or the CO_ACTIVATED graph
        (no self-reinforcing loop). No-op when flow is off."""
        if self.activation is None:
            return
        ids = [i for i in (confirmed or []) if i]
        if not ids:
            return
        amt = self.settings.flow_inject_confirmed
        spread_amt = amt * self.settings.flow_spread_factor
        with self._flow_turn_lock(namespace):
            self.activation.inject(namespace, ids, amount=amt)
            if spread_amt > 0.0:
                neighbors: set = set()
                for cid in ids:
                    for nid in self.graph.linked_memories(cid, scope, read_at):
                        if nid not in ids:
                            neighbors.add(nid)
                if neighbors:
                    self.activation.inject(namespace, list(neighbors), amount=spread_amt)

    def _flow_prime_ingest(self, namespace: str, memory_id: str) -> None:
        """Cold-start warm a freshly ingested memory (off by default: flow_prime_ingest=0.0)."""
        if self.activation is None or self.settings.flow_prime_ingest <= 0.0 or not memory_id:
            return
        self.activation.inject(namespace, [memory_id], amount=self.settings.flow_prime_ingest)
        self._brain(BrainEventType.FLOW_PRIMED, namespace=namespace, kind="ingest")

    # ---- reflex recall: the sub-second LOCAL candidate path ----------------
    def _hotset_ids(self, namespace: str) -> set:
        # Single warm-state: when flow is on, the activation field IS the working set (ids above the
        # floor). When off, the legacy binary hotset deque. No dual warm-state.
        if self.activation is not None:
            return {mid for mid, v in self.activation.snapshot(namespace).items()
                    if v >= self.settings.flow_floor}
        with self._hotset_lock:
            dq = self._hotset.get(namespace)
            return set(dq) if dq else set()

    def _touch_hotset(self, namespace: str, ids) -> None:
        """Record recently-recalled memory_ids per namespace (a small bounded working set). This is
        an ACCESS-recency signal feeding only the reflex hot-set axis -- never a memory-AGE term, so
        age-independence is preserved. Retired when flow is on (activation is the only warm-state)."""
        if self.activation is not None:
            return
        ids = [i for i in (ids or []) if i]
        if not ids:
            return
        with self._hotset_lock:
            dq = self._hotset.get(namespace)
            if dq is None:
                dq = deque(maxlen=max(1, self.settings.reflex_hotset_size))
                self._hotset[namespace] = dq
            for i in ids:
                dq.append(i)

    def reflex_recall(self, query: str, *, scope: Optional[Scope] = None,
                      as_of: Optional[float] = None, emit: bool = True,
                      begin_turn: bool = True) -> MemoryPacket:
        """Build a local MemoryPacket for `query` with NO model call (no embed, no NLI, no reader).
        This is the anti-RAG recall surface API/MCP expose and the ask() fast path consumes. The
        index is built lazily from the store on first use if it was not built at construction.

        Flow: a standalone call begins a turn (one decay + optional prime) so API/MCP reflex gets
        instinct too; ask() passes begin_turn=False because it already began the turn (avoids a
        double-decay when ask calls reflex_recall internally)."""
        scope = scope or Scope()
        # When REFLEX_RECALL is on the index is built + maintained incrementally, so ensure_built is
        # a no-op. When off it is NOT maintained, so this explicit on-demand call rebuilds -- but
        # only when the store's record count changed since the last build (a cheap COUNT probe that
        # catches even direct store writes). Repeated calls on an unchanged store reuse the index.
        if self.settings.reflex_recall_enabled:
            self.reflex_index.ensure_built(self.store)
        elif self.reflex_index.built_count != self.store.count(None):
            self.reflex_index.rebuild_from_store(self.store)
        if begin_turn and self.activation is not None:
            self._flow_begin_turn(scope.namespace, query, scope=scope, as_of=as_of)
        # Single warm-state: flow on -> activation snapshot (and no binary hot_ids); flow off ->
        # the legacy binary hotset and no activation map.
        if self.activation is not None:
            activation, hot = self._flow_snapshot(scope.namespace), set()
        else:
            activation, hot = None, self._hotset_ids(scope.namespace)
        packet = build_memory_packet(query, scope, store=self.store, graph=self.graph,
                                     index=self.reflex_index, settings=self.settings,
                                     as_of=as_of, hot_ids=hot, activation=activation)
        if emit and self.settings.brain_events_enabled:
            hit = packet.coverage >= self.settings.reflex_min_coverage and bool(packet.items)
            self._brain(BrainEventType.REFLEX_HIT if hit else BrainEventType.REFLEX_MISS,
                        namespace=scope.namespace, memory_ids=packet.candidate_ids(),
                        coverage=packet.coverage, latency_ms=packet.latency_ms.get("total"))
        return packet

    def region_hints(self, query: str, *, scope: Optional[Scope] = None,
                     as_of: Optional[float] = None, limit: int = 3,
                     member_limit: int = 6, use_reflex: bool = True) -> dict:
        """Model-free memory-neighborhood hints for host agents.

        Regions/cocoons route the caller toward raw memories; they are never treated as answer
        evidence. Every returned hint includes raw member ids plus content hashes/raw URIs so the
        caller can follow the path down to immutable substrate bytes.
        """
        t0 = time.perf_counter()
        scope = scope or Scope()
        limit = max(0, min(20, int(limit)))
        member_limit = max(0, min(50, int(member_limit)))
        if not self.settings.gist_channel_enabled:
            return {
                "query": query,
                "scope": scope.model_dump(),
                "as_of": as_of,
                "enabled": False,
                "source": "region_hints",
                "hint_count": 0,
                "hints": [],
                "reflex_candidate_ids": [],
                "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
                "note": "region routing disabled (GIST_CHANNEL=0)",
            }
        packet: Optional[MemoryPacket] = None
        candidates = []
        if use_reflex:
            packet = self.reflex_recall(
                query,
                scope=scope,
                as_of=as_of,
                emit=False,
                begin_turn=False,
            )
            candidates = packet.to_candidates()
        hints = self.retriever.memory_region_hints(
            query,
            scope=scope,
            at=as_of,
            candidates=candidates,
            limit=limit,
            member_limit=member_limit,
        )
        return {
            "query": query,
            "scope": scope.model_dump(),
            "as_of": as_of,
            "enabled": True,
            "source": "region_hints",
            "hint_count": len(hints),
            "hints": hints,
            "reflex_candidate_ids": packet.candidate_ids() if packet is not None else [],
            "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
            "note": "routing hints only; verify answers against raw source memories",
        }

    def structured_recall(self, query: str, *, scope: Optional[Scope] = None,
                          as_of: Optional[float] = None, verify: bool = True) -> dict:
        """Run the typed SMQE path directly and expose its plan/support/proof trace."""
        if not verify:
            raise ValueError("Engine.structured_recall requires verification")
        from .smqe import structured_recall
        selected_scope = scope or Scope()
        out = structured_recall(
            self.retriever,
            query,
            at=as_of,
            verify=verify,
            scope=selected_scope,
        )
        answer_model = out.pop("_answer_model", None)
        if answer_model is not None:
            governed = self._govern_answer(query, answer_model, selected_scope, as_of)
            if governed.status.value == "VERIFIED":
                out["answer"] = governed.answer
                out["citations"] = [citation.model_dump(mode="json")
                                    for citation in governed.citations]
                out["verified"] = True
                out["answered"] = True
                out["abstained"] = False
                proof = self.prove(governed, check_refs=True)
                out["proof_link_checks"] = len(governed.citations)
                out["immutable_proof"] = bool(proof.get("refs_verified"))
            else:
                out.update({
                    "answered": False,
                    "abstained": True,
                    "verified": False,
                    "confidence": 0.0,
                    "citations": [],
                    "proof_link_checks": 0,
                    "immutable_proof": False,
                    "failure_reason": "missing_immutable_proof",
                })
        else:
            out["proof_link_checks"] = 0
            out["immutable_proof"] = False
        if out.get("answered") and out.get("verified"):
            out["status"] = "VERIFIED"
            out["draft"] = ""
        else:
            out["status"] = "ABSTAINED"
            out["draft"] = out.get("answer", "")
            out["answer"] = ""
        return out

    def preference_profile(self, *, scope: Optional[Scope] = None,
                           include_inactive: bool = False, limit: int = 50) -> dict:
        """Read the current preference profile with source-memory provenance.

        The profile table is namespace-level, so this method enforces optional agent/project
        sub-scope through each line's source memory. Unattributed legacy lines are only visible to
        namespace-wide reads, never to a narrowed sub-scope where they cannot be proven visible.
        Source-linked rows whose source memory is missing are hidden from every scope.
        """
        scope = scope or Scope()
        limit = max(0, min(500, int(limit)))
        read_at = now()
        entries = self.store.get_profile_entries(
            scope.namespace,
            include_inactive=include_inactive,
        )
        profile: list[dict] = []
        skipped_unattributed = 0
        skipped_missing_source = 0
        skipped_scope = 0
        for entry in entries:
            source_id = str(entry.get("source_memory_id", "") or "")
            rec = self.store.get_record(source_id) if source_id else None
            if rec is None:
                if source_id:
                    skipped_missing_source += 1
                    continue
                if scope.agent_id is not None or scope.project_id is not None:
                    skipped_unattributed += 1
                    continue
            else:
                if not rec.scope.visible_to(scope):
                    skipped_scope += 1
                    continue
                if not include_inactive and not rec.is_active_at(read_at):
                    skipped_scope += 1
                    continue
            out = dict(entry)
            if rec is not None:
                out["content_hash"] = out.get("content_hash") or rec.content_hash
                out["raw_uri"] = out.get("raw_uri") or rec.raw_uri
                out["source_scope"] = rec.scope.model_dump()
                out["source_active"] = rec.is_active_at(read_at)
            else:
                out["source_scope"] = {"namespace": scope.namespace,
                                       "agent_id": None, "project_id": None}
                out["source_active"] = None
            out["provenance_complete"] = bool(
                out.get("source_memory_id") and out.get("content_hash") and out.get("raw_uri")
            )
            out["status"] = "inactive" if out.get("invalid_at") is not None else "active"
            profile.append(out)
            if len(profile) >= limit:
                break
        return {
            "scope": scope.model_dump(),
            "include_inactive": include_inactive,
            "profile_count": len(profile),
            "profile": profile,
            "provenance_complete": all(e["provenance_complete"] for e in profile) if profile else True,
            "skipped_unattributed": skipped_unattributed,
            "skipped_missing_source": skipped_missing_source,
            "skipped_out_of_scope": skipped_scope,
            "note": "preference profile lines are routing context; verify decisions with raw sources",
        }

    def sync_health(self, scope: Optional[Scope] = None) -> dict:
        """Track 2 derived synchronization report: are the rebuildable surfaces (vector index, BM25)
        consistent with the source-of-truth store, plus the namespace memory version and reflex
        status. The vector index and BM25 are GLOBAL (not per-scope), so only like-for-like global
        counts are compared -- comparing a global surface to a scoped store count would invent debt.
        Emits SYNC_DEBT_DETECTED when a surface is behind. Read-only; no model call; no key."""
        scope = scope or Scope()
        store_global = self.store.count(None)
        vector_global = len(self.index)
        bm25_global = (len(self.retriever.bm25.docs)
                       if self.settings.persistent_bm25_enabled else None)
        surfaces = {
            "store_records_global": store_global,
            "store_records_scope": self.store.count(scope),
            "vector_index_global": vector_global,
            "bm25_docs_global": bm25_global,
            "memory_version": self._ns_version(scope.namespace),
            "reflex_index": {"enabled": self.settings.reflex_recall_enabled,
                             "built": self.reflex_index.built,
                             "built_count": self.reflex_index.built_count},
        }
        debt: list[dict] = []
        if vector_global != store_global:
            debt.append({"surface": "vector_index", "expected": store_global, "actual": vector_global})
        if bm25_global is not None and bm25_global != store_global:
            debt.append({"surface": "bm25", "expected": store_global, "actual": bm25_global})
        if debt:
            self._brain(BrainEventType.SYNC_DEBT_DETECTED, namespace=scope.namespace, debt=debt)
        return {"in_sync": not debt, "surfaces": surfaces, "debt": debt,
                "repair": "rebuild_index_from_store" if debt else None}

    # ---- wake: read path --------------------------------------------------
    def _govern_answer(self, query: str, answer: Answer, scope: Scope,
                       at: Optional[float] = None) -> Answer:
        proof_at = now() if at is None else at
        entailed = [citation for citation in answer.citations
                    if citation.nli_label == NLILabel.ENTAILMENT]
        contradicted = any(citation.nli_label == NLILabel.CONTRADICTION
                           for citation in answer.citations)
        references_resolve = bool(entailed)
        for citation in entailed:
            record = self.store.get_record(citation.memory_id)
            normalized_snippet = re.sub(r"\s+", " ", citation.snippet or "").strip().lower()
            normalized_record = re.sub(
                r"\s+", " ", (record.text or record.summary or "") if record else ""
            ).strip().lower()
            if (record is None
                    or record.scope.key() != scope.key()
                    or record.content_hash != citation.content_hash
                    or record.raw_uri != citation.raw_uri
                    or record.valid_at != citation.valid_at
                    or not record.is_active_at(proof_at)
                    or not normalized_snippet
                    or normalized_snippet[:300] not in normalized_record):
                references_resolve = False
                break
            try:
                raw = self.substrate.get(citation.content_hash)
                if sha256_hex(raw) != citation.content_hash:
                    references_resolve = False
                    break
                if record.modality == Modality.TEXT:
                    normalized_raw = re.sub(
                        r"\s+", " ", raw.decode("utf-8")
                    ).strip().lower()
                    if normalized_snippet[:300] not in normalized_raw:
                        references_resolve = False
                        break
            except Exception:
                references_resolve = False
                break
        if answer.verified and entailed and references_resolve and not contradicted:
            return answer.model_copy(update={"citations": entailed})
        if contradicted:
            reason = "active memory contradicts the answer"
        elif answer.verified and not references_resolve:
            reason = "immutable proof resolution failed"
        else:
            reason = (answer.note or "no source entails the answer").removeprefix("unverified: ")
        return Answer.abstain(
            query,
            note=reason,
            generated_by=answer.generated_by,
            retrieved_count=answer.retrieved_count,
        )

    def prove_external_draft(self, query: str, draft: str, evidence_memory_ids: list[str], *,
                             scope: Optional[Scope] = None,
                             as_of: Optional[float] = None,
                             generated_by: str = "external-reader") -> Answer:
        scope = scope or Scope()
        at = now() if as_of is None else as_of
        candidates: list[RetrievalCandidate] = []
        for memory_id in dict.fromkeys(evidence_memory_ids or []):
            record = self.store.get_record(memory_id)
            if (record is None
                    or record.scope.key() != scope.key()
                    or not record.is_active_at(at)):
                continue
            candidates.append(RetrievalCandidate(
                record=record,
                dense_score=1.0,
                fused_score=1.0,
                rerank_score=1.0,
            ))
        if not draft.strip() or not candidates:
            return Answer.abstain(
                query,
                note="external reader returned no provable draft or exact-scope evidence",
                generated_by=generated_by,
                retrieved_count=len(candidates),
            )
        answer = self.retriever.answer(
            query,
            at=at,
            verify=True,
            scope=scope,
            precomputed=candidates,
            reader_model=generated_by,
            reader=lambda _query, _blocks: draft,
            allow_structured=False,
            require_all_claims=True,
            allow_grounding_rescue=False,
        )
        return self._govern_answer(query, answer, scope, at)

    def ask(self, query: str, *, at: Optional[float] = None, verify: bool = True,
            scope: Optional[Scope] = None, as_of: Optional[float] = None,
            use_cache: bool = True, reader_model: Optional[str] = None,
            reader: Optional[Callable[[str, list[str]], str]] = None) -> Answer:
        if not verify:
            raise ValueError("Engine.ask requires verification; use recall diagnostics for evidence-only access")
        scope = scope or Scope()
        read_at = as_of if as_of is not None else at
        sk = scope.key()
        # Track 3.3 false-premise gate (flag-off -> skipped, byte-identical). A presuppositional
        # question whose entities are totally disconnected in memory abstains HERE, before any
        # retrieval or model call -- the reader can never confabulate a relationship memory denies.
        if self.settings.false_premise_enabled:
            fp = self.check_false_premise(query, scope=scope, as_of=read_at)
            if fp is not None:
                note = f"abstained: false-premise ({fp['category']})"
                self._brain(BrainEventType.ANSWER_ABSTAINED, namespace=scope.namespace,
                            note=note, reason=fp["category"])
                return Answer.abstain(question=query, answer=fp["message"],
                                      retrieved_count=0, note=note)
        # Prospective memory: learn P(next query-signature | current) for predictive prefetch.
        if self.settings.markov_prefetch_enabled:
            self._observe_query(query)
        use_cache = use_cache and reader is None and self.settings.semantic_cache_enabled
        # Cache bypass when observing. RECALL_TRACE always bypasses: a cache hit carries no fresh
        # trace, and trace inspection wants the real retrieval. BRAIN_EVENTS bypasses ONLY when
        # versioning is off -- a non-versioned hit could be a stale truth with no fresh event. With
        # versioning on, a hit is truth-fresh, so we keep the cache and emit CACHE_HIT instead (a hit
        # IS the absence of retrieval; that is exactly what CACHE_HIT represents).
        if self.settings.recall_trace_enabled or (
                self.settings.brain_events_enabled and not self.settings.cache_versioning_enabled):
            use_cache = False
        # Time-travel (as_of) queries are not cached (the answer depends on the as-of time).
        if as_of is not None:
            use_cache = False

        # Versioned cache invalidation: tag every lookup/store with the namespace's current memory
        # version, so a content write makes prior entries unreachable (no stale-truth hits). version
        # 0 (flag off) == the legacy never-invalidated cache, byte-identical.
        cache_ver = self._ns_version(scope.namespace) if self.settings.cache_versioning_enabled else 0
        qvec = None
        if use_cache:
            hit = self.cache.get(sk, query, None, version=cache_ver)   # exact-hash (no embedding)
            if hit is not None:
                self._brain(BrainEventType.CACHE_HIT, namespace=scope.namespace, mode="exact")
                return self._govern_answer(query, hit, scope, read_at)

        # SMQE answers narrow extractive/compositional memory questions from typed claims first and
        # raw records second. It is the general structured recall path; retrieval remains the fallback.
        from .smqe import structured_answer
        structured_answered = structured_answer(
            self.retriever, query, at=read_at, verify=verify, scope=scope
        )
        if structured_answered is not None and (
                self.settings.recall_trace_enabled or self.settings.brain_events_enabled):
            cited_ids = [c.memory_id for c in structured_answered.citations]
            self.retriever.last_trace = RecallTrace(
                query=query,
                scope=scope,
                enabled_channels=["smqe"],
                channel_results={"smqe": cited_ids},
                fused_scores={mid: 1.0 for mid in cited_ids},
                selected_candidates=cited_ids,
                latency_by_stage={},
            )
        if structured_answered is not None and self.settings.flow_activation_enabled:
            self._flow_begin_turn(scope.namespace, query, scope=scope, as_of=read_at)

        if structured_answered is None and use_cache:
            qvec = self.client.embed_text(query)         # embed once, reuse in retrieval
            if len(self._query_log) < 5000:              # bounded query log for pre-fetch
                self._query_log.append((sk, qvec))
            hit = self.cache.get(sk, query, qvec, version=cache_ver)   # cosine >= threshold
            if hit is not None:
                self._brain(BrainEventType.CACHE_HIT, namespace=scope.namespace, mode="semantic")
                return self._govern_answer(query, hit, scope, read_at)

        # Track 9 Flow: begin the turn (one decay + optional query prime) ONLY now that this is a
        # real recall -- a cache hit returned above without mutating the field. Done once here, so
        # the nested reflex_recall call below passes begin_turn=False (no double-decay).
        if structured_answered is None and self.settings.flow_activation_enabled:
            self._flow_begin_turn(scope.namespace, query, scope=scope, as_of=read_at)

        # Track 1 Reflex Recall: try the LOCAL fast path first. On a confident hit, feed the reflex
        # candidates to the reader as `precomputed` (skipping ANN/rerank, with no embed/NLI in the
        # recall itself); NLI/abstention/proof still gate the FINAL answer. On a low-coverage miss,
        # fall back to full retrieval. Flag-off -> this block never runs (baseline byte-identical).
        reflex_candidates = None
        if structured_answered is None and self.settings.reflex_recall_enabled:
            packet = self.reflex_recall(query, scope=scope, as_of=read_at, emit=False, begin_turn=False)
            if packet.coverage >= self.settings.reflex_min_coverage and packet.items:
                reflex_candidates = packet.to_candidates()
                self._brain(BrainEventType.REFLEX_HIT, namespace=scope.namespace,
                            memory_ids=packet.candidate_ids(), coverage=packet.coverage,
                            latency_ms=packet.latency_ms.get("total"))
                # A reflex hit skips retrieve(), where last_trace is built. Set an honest reflex
                # trace so proof recall-paths and channel-win telemetry attribute THIS query (not a
                # stale one) to the reflex channel. Only meaningful when trace/brain is observing.
                if self.settings.recall_trace_enabled or self.settings.brain_events_enabled:
                    self.retriever.last_trace = RecallTrace(
                        query=query, scope=scope, enabled_channels=["reflex"],
                        channel_results={"reflex": packet.candidate_ids()},
                        fused_scores={c.memory_id: c.score.aggregate for c in packet.items},
                        selected_candidates=packet.candidate_ids(),
                        latency_by_stage=dict(packet.latency_ms))
            else:
                self._brain(BrainEventType.REFLEX_MISS, namespace=scope.namespace,
                            coverage=packet.coverage)
                self._brain(BrainEventType.REFLEX_FALLBACK, namespace=scope.namespace)

        # When the idle learner is fed, retrieve candidates explicitly so per-channel
        # contributions are available for feedback; otherwise answer() retrieves internally
        # exactly as before (the default call signature is unchanged). Track 9: on the hybrid path
        # (reflex miss / REFLEX_RECALL=0) pass the field snapshot so instinct reaches hybrid too
        # (inert when the activation channels are off; None when flow is off -> byte-identical).
        flow_act = self._flow_snapshot(scope.namespace) if self.settings.flow_activation_enabled else None
        precomputed = None
        reader_kwargs = {"reader": reader} if reader is not None else {}
        if structured_answered is not None:
            ans = structured_answered
        elif reflex_candidates is not None:
            ans = self.retriever.answer(query, at=read_at, verify=verify, scope=scope, qvec=qvec,
                                        precomputed=reflex_candidates, reader_model=reader_model,
                                        activation=flow_act, **reader_kwargs)
        elif self.settings.feedback_enabled or flow_act:
            precomputed = self.retriever.retrieve(query, at=read_at, scope=scope, qvec=qvec,
                                                  activation=flow_act)
            ans = self.retriever.answer(query, at=read_at, verify=verify, scope=scope, qvec=qvec,
                                        precomputed=precomputed, reader_model=reader_model,
                                        activation=flow_act, **reader_kwargs)
        else:
            ans = self.retriever.answer(query, at=read_at, verify=verify, scope=scope, qvec=qvec,
                                        reader_model=reader_model, **reader_kwargs)

        pre_govern_contradicted = [c.memory_id for c in ans.citations
                                   if c.nli_label == NLILabel.CONTRADICTION]
        ans = self._govern_answer(query, ans, scope, read_at)
        self._epistemic_after_ask(query, ans, scope, pre_govern_contradicted)

        # Reconsolidation as a write path (retrieval is no longer read-only).
        # PHASE 1 (NO lock): the confirmed-citation re-embed is the only model call here; it must
        # run OFF the write lock (holding the lock across a governed model call would deadlock).
        confirmed: list[str] = []
        contradicted: list[str] = []
        reembed: dict[str, "np.ndarray"] = {}
        defer = self.settings.defer_reembed_enabled
        for cit in ans.citations:
            if cit.nli_label == NLILabel.ENTAILMENT:
                confirmed.append(cit.memory_id)
                if not defer:                       # inline re-embed (off the lock); else deferred
                    rec = self.store.get_record(cit.memory_id)
                    if rec is not None:
                        try:
                            reembed[cit.memory_id] = self.client.embed_text(rec.text)
                        except Exception as e:      # best-effort; never silently swallow it
                            self._degraded("reinforce-reembed", e)
            elif cit.nli_label == NLILabel.CONTRADICTION:
                contradicted.append(cit.memory_id)
        if defer and confirmed:                     # push the re-embed to the idle/sleep drain
            self._enqueue_reembed(confirmed)
        # PHASE 2 (write lock): apply all index/store/graph mutations atomically. Records are
        # re-read under the lock so the read-modify-write is not lost under concurrent recall.
        with self._write_lock:
            for mid in confirmed:
                rec = self.store.get_record(mid)
                if rec is None:
                    continue
                if mid in reembed:
                    self.index.update(mid, reembed[mid])   # refreshes CONTENT vector only (age-free)
                fsrs.reinforce(rec.fsrs, importance=rec.importance)
                self._reinforce_verified_helpful(rec)
                self.store.upsert_record(rec)
            for mid in contradicted:
                rec = self.store.get_record(mid)
                if rec is None:
                    continue
                fsrs.lapse(rec.fsrs)                        # down-weight, never delete
                self.store.upsert_record(rec)
            # Memory linking by co-activation: co-confirmed memories gain a strengthened edge.
            if len(confirmed) >= 2:
                self.graph.link_memories(confirmed, scope=scope, valid_at=read_at)
            # Persist the index only when THIS ask mutated it (inline re-embeds applied above).
            # The save is an O(corpus) disk write under the write lock; with DEFER_REEMBED the
            # refresh is queued for the idle drain and the index is unchanged here, so saving
            # would serialize concurrent asks behind IO that grows with corpus size. FSRS/graph
            # durability is unaffected (sqlite writes above); a re-embed refresh is recomputable.
            if reembed:
                self.index.save()
        if self.settings.feedback_enabled and precomputed is not None:
            self._emit_feedback(scope, query, qvec, precomputed, confirmed)
        # Working set write. Track 9 Flow (single warm-state): when flow is on, commit the confirmed
        # recall into the ActivationField (inject + one-hop spread) -- this is the ONE writer, read
        # by every recall surface. When flow is off, the legacy reflex hotset. Access-recency only,
        # never a memory-age term. The flow commit runs even with REFLEX_RECALL off (hybrid reads it).
        warm_ids = confirmed or [c.memory_id for c in ans.citations]
        if self.settings.flow_activation_enabled:
            self._flow_commit_recall(scope.namespace, warm_ids, scope, read_at)
        elif self.settings.reflex_recall_enabled:
            self._touch_hotset(scope.namespace, warm_ids)
        # Connected Brain Loop: project the answer onto the improvement stream (gated).
        if self.settings.brain_events_enabled:
            cited = [c.memory_id for c in ans.citations]
            self._brain(BrainEventType.MEMORY_RECALLED, namespace=scope.namespace,
                        memory_ids=cited, retrieved=ans.retrieved_count, verified=ans.verified)
            if any(c.nli_label == NLILabel.CONTRADICTION for c in ans.citations):
                self._brain(BrainEventType.CONTRADICTION_DETECTED, namespace=scope.namespace,
                            memory_ids=cited)
            if ans.verified:
                self._brain(BrainEventType.ANSWER_VERIFIED, namespace=scope.namespace,
                            memory_ids=confirmed, confidence=ans.confidence)
            elif ans.note.startswith("abstained"):
                self._brain(BrainEventType.ANSWER_ABSTAINED, namespace=scope.namespace,
                            note=ans.note)
            else:
                self._brain(BrainEventType.RETRIEVAL_MISSED, namespace=scope.namespace,
                            memory_ids=cited, note=ans.note)
        # Route the recall through the shared lifecycle hook so channel-win telemetry updates on the
        # product path (no-op unless BRAIN_EVENTS is on; the answer is never altered).
        self.lifecycle.after_recall(ans, scope)
        if use_cache:
            self.cache.put(sk, query, qvec, ans, version=cache_ver)
        # Publish the completed trace for cross-thread introspection (MCP recall_trace,
        # /api/recall_trace run on their own worker threads and cannot see this thread's
        # thread-local). Scope-keyed + bounded; written once per completed ask under a lock.
        # In-flight concurrent asks still isolate via the retriever's thread-local.
        if self.settings.recall_trace_enabled:
            completed = self.retriever.last_trace
            if completed is not None:
                with self._trace_snapshot_lock:
                    self._trace_snapshots[sk] = completed
                    self._trace_snapshots.move_to_end(sk)
                    while len(self._trace_snapshots) > 128:
                        self._trace_snapshots.popitem(last=False)
        return ans

    def link_coactivated(
        self,
        memory_ids: list[str],
        *,
        scope: Optional[Scope] = None,
        valid_at: Optional[float] = None,
        cap: int = 5,
    ) -> dict:
        """Link top co-surfaced memories with CO_ACTIVATED edges.

        The neutral benchmark rows do not call ask(), so they would otherwise never write the
        co-activation graph that the product path writes after verified recalls. This hook is a
        pure graph mutation, gated by the caller, and never calls a model or changes raw memory.
        """
        scope = scope or Scope()
        ids = [mid for mid in dict.fromkeys(memory_ids or []) if mid][: max(0, cap)]
        if len(ids) < 2:
            return {"linked": 0, "memory_ids": ids}
        with self._write_lock:
            linked = self.graph.link_memories(ids, scope=scope, valid_at=valid_at)
        return {"linked": linked, "memory_ids": ids}

    def _epistemic_after_ask(self, query: str, ans: Answer, scope: Scope,
                             contradicted_ids: list[str]) -> None:
        """Dispatch the passive epistemic hooks OFF the answer path. The map/agenda
        writes are sqlite commits; inline they measurably cost the ask's latency
        budget (caught by the smqe fullpath p95 invariant), and nothing downstream
        of an ask reads them synchronously -- so a daemon thread absorbs the IO.
        The body is `_epistemic_after_ask_sync` (tests exercise it directly)."""
        if self.knowledge_map_store is None and self.research_agenda is None:
            return
        t = threading.Thread(
            target=self._epistemic_hook_worker,
            args=(query, ans, scope, contradicted_ids),
            name="epistemic-hook", daemon=True,
        )
        with self._epistemic_threads_lock:
            self._epistemic_threads.add(t)
        t.start()

    def _epistemic_hook_worker(self, query: str, ans: Answer, scope: Scope,
                               contradicted_ids: list[str]) -> None:
        try:
            self._epistemic_after_ask_sync(query, ans, scope, contradicted_ids)
        finally:
            with self._epistemic_threads_lock:
                self._epistemic_threads.discard(threading.current_thread())

    def close(self) -> None:
        """Idempotent shutdown: JOIN in-flight passive-hook threads, then close every
        sqlite owner. Callers that build an Engine inside a TemporaryDirectory MUST
        call this before the directory is removed -- a background hook creating a WAL
        file mid-rmtree is a teardown race (caught live by the smqe fullpath eval)."""
        with self._epistemic_threads_lock:
            pending = list(self._epistemic_threads)
        for t in pending:
            t.join(timeout=5.0)
        for owner in (self.knowledge_map_store, self.research_agenda,
                      self.feedback, self.store):
            try:
                if owner is not None:
                    owner.close()
            except Exception:
                pass

    def _epistemic_after_ask_sync(self, query: str, ans: Answer, scope: Scope,
                                  contradicted_ids: list[str]) -> None:
        """Passive epistemic hooks on the governed read path (ZERO model calls, best
        effort): an abstain mints/refreshes an UNKNOWN query cell and enqueues a
        research task; a contradiction veto mints a CONTESTED cell; a verified answer
        closes its matching query cell with the proof. Failure here degrades, never
        breaks an ask."""
        if self.knowledge_map_store is None and self.research_agenda is None:
            return
        try:
            contradiction_veto = bool(contradicted_ids) and "contradict" in (ans.note or "")
            if self.knowledge_map_store is not None:
                if contradiction_veto:
                    self.knowledge_map_store.on_contradiction(
                        query, contradicted_ids, scope, note=ans.note)
                self.knowledge_map_store.on_answer(query, ans, scope)
            if (self.research_agenda is not None
                    and ans.status == AnswerStatus.ABSTAINED):
                from .autoresearch.types import ResearchTask, classify_failure
                self.research_agenda.enqueue(ResearchTask(
                    query=query, namespace=scope.namespace, agent_id=scope.agent_id,
                    project_id=scope.project_id,
                    failure_class=classify_failure(ans.note),
                    origin="contested_cell" if contradiction_veto else "ask_fail",
                    trace_id=""))
        except Exception as e:
            self._degraded("epistemic-hook", e)

    def improve(self, *, scope: Optional[Scope] = None, max_trials: int = 0,
                max_probes: Optional[int] = None, dry_run: bool = False,
                lab=None, judge=None) -> dict:
        """The IMPROVE verb: one metabolic research cycle -- token-free map rebuild,
        curiosity wave over the frontier (real prove-path probes), then up to
        `max_trials` guarded dev experiments (requires `lab`). Model spend happens
        ONLY here, never on ingest/sleep."""
        from .autoresearch.loop import drain
        return drain(self, scope=scope, max_trials=max_trials, max_probes=max_probes,
                     dry_run=dry_run, judge=judge, lab=lab)

    def research_status(self, *, last_n: int = 5) -> dict:
        """Method metadata only (agenda depth, champion, last trials, map counts) --
        no answer text, no drafts."""
        from .autoresearch.loop import research_status as _status
        return _status(self, last_n=last_n)

    def _emit_feedback(self, scope: Scope, query: str, qvec, candidates: list,
                       confirmed: list[str]) -> None:
        """Append one dev-feedback tuple from the PRODUCT read path. The reward is whether any
        retrieved memory was NLI-confirmed; the per-channel contributions are the channel
        scores of the confirmed (or top) candidate. The FeedbackBuffer forces is_dev=0 for any
        benchmark namespace, so this can never feed a learner a benchmark item."""
        if not candidates:
            return
        by_id = {c.record.memory_id: c for c in candidates}
        ref = by_id.get(confirmed[0]) if confirmed else candidates[0]
        feats = {
            "coverage": float(max((c.dense_score for c in candidates), default=0.0)),
            "n_cands": len(candidates),
            "contrib_dense": float(getattr(ref, "dense_score", 0.0)),
            "contrib_bm25": float(getattr(ref, "bm25_score", 0.0)),
            "contrib_graph": float(getattr(ref, "graph_score", 0.0)),
        }
        try:
            self.feedback.append(scope.namespace or "default", query, feats,
                                 arm=self.settings.fusion_method,
                                 reward=1.0 if confirmed else 0.0, qvec=qvec)
        except Exception as e:        # feedback is best-effort; log rather than break the read path
            self._degraded("feedback-log", e)

    def learn_fusion_weights(self) -> dict:
        """Idle cadence: replay the dev feedback buffer through the EG/FTRL learner and persist
        the content-channel weights to index_dir/fusion_weights.json (read by the retriever when
        FUSION_LEARNER=1). Dev-only by construction; no model call."""
        from .optim.online_weights import learn_fusion_weights as _learn
        from .optim.online_weights import save_weights
        rows = self.feedback.sample(limit=2000)
        if not rows:
            return {}
        weights = _learn(rows, ["dense", "bm25", "graph"],
                         method=self.settings.fusion_learner_method)
        save_weights(self.settings.index_dir / "fusion_weights.json", weights)
        return weights

    # ---- prospective memory (Markov query-transition model) --------------
    def _query_signature(self, query: str) -> str:
        """A deterministic, model-free signature for a query (its first entity, else first
        salient token). Used as the Markov state so P(next|current) is learnable token-free."""
        from .events import parse_query
        ents = [str(e).lower() for e in (parse_query(query).get("entities") or []) if str(e).strip()]
        if ents:
            return ents[0]
        toks = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2]
        return toks[0] if toks else "_"

    def _observe_query(self, query: str) -> None:
        self._last_query_text = query or ""
        self.markov.observe(self._query_signature(query))

    def predict_next_signatures(self, query: str, top_k: int = 3) -> list:
        """The Markov model's most-likely NEXT query signatures given the current query."""
        return self.markov.predict(self._query_signature(query), top_k=top_k)

    def warmup_predicted_prefetch(self, query: Optional[str] = None, *,
                                  scope: Optional[Scope] = None,
                                  at: Optional[float] = None,
                                  top_k: Optional[int] = None) -> dict:
        """Pre-stage contexts for Markov-predicted next query signatures.

        This connects the prospective Markov model to the existing PrefetchCache during idle time.
        It is default-off, embedding-only, and still builds context through the normal retriever so
        future citation-preserving answer paths can consume the same evidence shape.
        """
        enabled = self.settings.markov_prefetch_enabled and self.settings.flow_warmup_enabled
        if not enabled:
            return {"enabled": False, "predictions": 0, "warmed": 0}
        scope = scope or Scope()
        current = query if query is not None else self._last_query_text
        if not current:
            return {"enabled": True, "predictions": 0, "warmed": 0, "note": "no-query"}
        k = self.settings.flow_warmup_topk if top_k is None else top_k
        if k <= 0:
            return {"enabled": True, "predictions": 0, "warmed": 0, "note": "topk<=0"}
        predictions = self.predict_next_signatures(current, top_k=k)
        signatures = [str(sig) for sig, _prob in predictions if str(sig).strip() and str(sig) != "_"]
        if not signatures:
            return {"enabled": True, "predictions": len(predictions), "warmed": 0,
                    "signatures": []}
        qvecs = np.asarray(self.client.embed_texts(signatures), dtype=np.float32)
        activation = self._flow_snapshot(scope.namespace) if self.settings.flow_activation_enabled else None
        warmed = 0
        for sig, qvec in zip(signatures, qvecs):
            cands = self.retriever.retrieve(
                sig, at=at, scope=scope, qvec=qvec, skip_rerank=True, activation=activation)
            blocks = self.retriever.assemble_context(
                sig, cands, at=at, scope=scope, activation=activation)
            if not blocks:
                continue
            self.prefetch.add(qvec, blocks)
            warmed += 1
        if warmed:
            self._brain(BrainEventType.FLOW_WARMED, namespace=scope.namespace,
                        signatures=signatures[:warmed], warmed=warmed)
        return {"enabled": True, "predictions": len(predictions), "warmed": warmed,
                "signatures": signatures}

    def reawaken(self, memory_id: str) -> Optional[MemoryRecord]:
        """Strong-cue reawakening: reset retrievability + boost stability (O(1))."""
        with self._write_lock:
            rec = self.store.get_record(memory_id)
            if rec is None:
                return None
            fsrs.reinforce(rec.fsrs, importance=max(0.6, rec.importance))
            self.store.upsert_record(rec)
            return rec

    def forget(self, memory_id: str) -> Optional[MemoryRecord]:
        """Lower a memory's retrieval PRIORITY via the FSRS forgetting path. This is the inverse
        of reawaken and NEVER deletes the raw record: the immutable substrate is untouched, only
        the mutable index-priority weight drops. Returns None if the memory does not exist."""
        with self._write_lock:
            rec = self.store.get_record(memory_id)
            if rec is None:
                return None
            fsrs.lapse(rec.fsrs)
            self.store.upsert_record(rec)
            return rec

    def get_record_in_scope(self, memory_id: str,
                            scope: Optional[Scope] = None) -> Optional[MemoryRecord]:
        """get_record but enforce scope visibility: a record outside the requested scope is
        invisible (returns None), so an id from another namespace can never be read cross-scope.
        Reuses the engine's existing Scope.visible_to model, not a parallel filter."""
        rec = self.store.get_record(memory_id)
        if rec is None:
            return None
        if scope is not None and not rec.scope.visible_to(scope):
            return None
        return rec

    def set_metadata(self, memory_id: str, metadata: dict,
                     scope: Optional[Scope] = None) -> Optional[MemoryRecord]:
        """Attach/merge metadata onto a memory's mutable state (the raw substrate is NOT touched).
        Scope-checked so it cannot write across a namespace boundary."""
        with self._write_lock:
            rec = self.get_record_in_scope(memory_id, scope)
            if rec is None:
                return None
            rec.metadata.update(dict(metadata or {}))
            self.store.upsert_record(rec)
            return rec

    # ---- sleep: consolidation/replay -------------------------------------
    def consolidate(self, *, verify: bool = True, scope: Optional[Scope] = None) -> dict:
        """Offline replay loop. Clusters by shared entity, writes verified semantic
        summaries, and decays FSRS priority. Never deletes.

        Schema-accelerated (Tse et al.): schema-CONSISTENT clusters (entities already in
        the graph schema) fast-track at a lower salience threshold; NOVEL clusters stay
        episodic longer (higher threshold) before abstraction."""
        records = self.store.all_records(scope)
        schema_entities = {e.lower() for e in self._graph_entities(scope)}

        clusters: dict[str, list[MemoryRecord]] = {}
        for r in records:
            key = r.entities[0].lower() if r.entities else r.modality.value
            clusters.setdefault(key, []).append(r)

        summaries_written = 0
        fast_tracked = 0
        for key, group in clusters.items():
            # Schema-consistent if the cluster key (lead entity) already exists in schema
            # via OTHER memories -- i.e. it fits an established structure.
            others = sum(1 for r in records if key in {e.lower() for e in r.entities})
            schema_consistent = key in schema_entities and others >= 2
            threshold = 0.5 if schema_consistent else 0.7
            high = [g for g in group if g.salience >= threshold and not g.consolidated]
            if len(high) < 2:
                continue
            summary = self.client.consolidate_summary([g.text for g in high])  # real call
            ok = True
            if verify:
                for g in high:
                    label, _ = self.client.nli(self.retriever._ground_truth(g), summary)
                    if label == "contradiction":
                        ok = False
                        break
            if ok:
                for g in high:
                    g.summary = summary
                    g.consolidated = True
                    self.store.upsert_record(g)
                summaries_written += 1
                if schema_consistent:
                    fast_tracked += 1

        # FSRS index-priority decay (down-weight only; raw untouched).
        for r in records:
            fsrs.decay(r.fsrs)
            self.store.upsert_record(r)

        return {
            "records": len(records),
            "clusters": len(clusters),
            "summaries_written": summaries_written,
            "schema_fast_tracked": fast_tracked,
        }

    def _graph_entities(self, scope: Optional[Scope]) -> set[str]:
        ents: set[str] = set()
        for e in self.store.all_edges(scope):
            ents.add(e.src)
            ents.add(e.dst)
        return ents

    # ---- async consolidation of LLM-free fast-written records ------------
    def consolidate_pending(self, *, scope: Optional[Scope] = None,
                            score_importance: bool = True, max_workers: int = 8) -> dict:
        """Process records written by the LLM-free fast path (consolidate_now=False):
        extract (s,p,o) facts (+visual), build the bi-temporal graph with active conflict
        resolution (invalidate-not-delete + supersedes), normalize dates, index events,
        type preferences. Never deletes raw.

        Performance: extraction LLM calls run CONCURRENTLY (I/O-bound); graph node-features
        are computed ONCE (not per record). `score_importance=False` skips the per-record
        qwen-flash call -- importance only feeds FSRS priority / salience pruning, neither of
        which is in the ranking path, so it has no effect on retrieval accuracy."""
        from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

        # Idle/sleep cadence also drains the deferred re-embed queue (S1).
        reembed = self.drain_reembed_queue()
        pending = [r for r in self.store.all_records(scope)
                   if r.metadata.get("pending_consolidation")]
        if not pending:
            return {"pending_processed": 0, "facts_extracted": 0, "events_indexed": 0,
                    "reembedded": reembed.get("reembedded", 0)}

        def _planned_extract_windows(rec: MemoryRecord) -> int:
            text = rec.text or ""
            if not text.strip():
                return 0
            if self.settings.extract_chunking_enabled and len(text) > 6000:
                return len(chunk_text(
                    text,
                    self.settings.extract_chunk_chars,
                    self.settings.extract_chunk_overlap,
                ))
            return 1

        window_plan = {rec.memory_id: _planned_extract_windows(rec) for rec in pending}
        planned_windows = sum(window_plan.values())
        call_budget = max(0, int(getattr(self.settings, "consolidation_extract_call_budget", 0) or 0))
        raw_only_window_threshold = max(
            0,
            int(getattr(self.settings, "consolidation_raw_only_window_threshold", 0) or 0),
        )
        record_raw_only_ids = {
            rec.memory_id
            for rec in pending
            if raw_only_window_threshold
            and window_plan.get(rec.memory_id, 0) > raw_only_window_threshold
        }
        extracting_records = sum(1 for n in window_plan.values() if n > 0)
        budget_trigger = max(1, int(call_budget * 0.8)) if call_budget else 0
        aggregate_long_haystack_bounded = bool(call_budget and planned_windows > budget_trigger)
        long_haystack_bounded = bool(aggregate_long_haystack_bounded or record_raw_only_ids)
        raw_only_long_haystack = (
            aggregate_long_haystack_bounded
            and bool(getattr(self.settings, "consolidation_long_haystack_raw_only", False))
        )
        window_cap_per_record = 0
        if aggregate_long_haystack_bounded and extracting_records:
            if raw_only_long_haystack:
                # DRAIN mode (extraction-audit fleet 2026-07-09, 7 zero-edge namespaces, one
                # proven cause): the old behavior zeroed the WHOLE batch's allowances when the
                # raw-only flag met an oversized batch, so extract_edges never ran for any
                # record and -- because pending_consolidation was cleared unconditionally --
                # never ran on ANY later sleep either: the graph/event channel died silently
                # for the namespace. Now each sleep processes a budget-bounded slice at one
                # window per record (per-record minimum) and DEFERS the rest (they stay
                # pending, below), so successive sleeps drain the backlog. Same per-sleep cost
                # ceiling as the budget intends; starvation is no longer permanent.
                window_cap_per_record = 1
            else:
                # Every record gets at least one extraction window; raw memory is still fully
                # indexed. This bounds auxiliary graph/event enrichment without making
                # long-haystack sleep fail.
                window_cap_per_record = max(1, call_budget // extracting_records)
        remaining_budget = call_budget if aggregate_long_haystack_bounded else None
        window_allowance: dict[str, int] = {}
        for rec in pending:
            n = window_plan.get(rec.memory_id, 0)
            if n <= 0:
                window_allowance[rec.memory_id] = 0
                continue
            if rec.memory_id in record_raw_only_ids:
                allow = 0
            elif remaining_budget is None:
                allow = n
            elif remaining_budget > 0:
                allow = min(n, window_cap_per_record, remaining_budget)
                remaining_budget -= allow
            else:
                allow = 0
            window_allowance[rec.memory_id] = allow
        submitted_windows = sum(window_allowance.values())
        raw_only_bounded = sum(
            1 for rec in pending
            if window_plan.get(rec.memory_id, 0) > 0 and window_allowance.get(rec.memory_id, 0) == 0
        )
        record_raw_only_bounded = sum(1 for rec in pending if rec.memory_id in record_raw_only_ids)
        partial_bounded = sum(
            1 for rec in pending
            if 0 < window_allowance.get(rec.memory_id, 0) < window_plan.get(rec.memory_id, 0)
        )

        def _extract(rec: MemoryRecord) -> tuple[MemoryRecord, list[dict], list[dict]]:
            triples: list[dict[str, str]] = []
            extracted_claims: list[dict] = []
            if rec.text.strip():
                try:
                    bounded = getattr(self.client, "extract_edges_bounded", None)
                    # CLAIM_EXTRACTION gates the CALLS, not just the write: with the flag off,
                    # claims_for_record below never runs, so paying the claim-extraction model
                    # calls here would buy nothing (roughly half of extraction spend discarded).
                    claims_on = self.settings.claim_extraction_enabled
                    claim_bounded = getattr(self.client, "extract_claims_bounded", None) if claims_on else None
                    claim_extract = getattr(self.client, "extract_claims", None) if claims_on else None
                    combined = (getattr(self.client, "extract_edges_and_claims_bounded", None)
                                if claims_on and self.settings.extract_combined_enabled else None)
                    allow = window_allowance.get(rec.memory_id, 0)
                    if allow <= 0:
                        triples = []
                        extracted_claims = []
                    elif callable(combined):
                        # EXTRACT_COMBINED: one call per window feeds BOTH channels.
                        t2, c2 = combined(rec.text, max_windows=allow if window_cap_per_record else 0)
                        triples.extend(t2)
                        extracted_claims.extend(c2)
                    elif window_cap_per_record and callable(bounded):
                        triples.extend(bounded(rec.text, max_windows=allow))
                        if callable(claim_bounded):
                            extracted_claims.extend(claim_bounded(rec.text, max_windows=allow))
                    else:
                        triples.extend(self.client.extract_edges(rec.text))   # real, concurrent
                        if callable(claim_extract):
                            extracted_claims.extend(claim_extract(rec.text))
                except Exception as e:   # one record's extraction must not abort the whole sweep
                    # extract_edges already degrades moderation/truncation internally; this catches
                    # the residual (e.g. a transient past its retry budget). The raw record stays in
                    # the substrate -- it just contributes no graph facts this pass. Log, continue.
                    self._degraded("extract-edges", e)
            if rec.modality in (Modality.IMAGE, Modality.VIDEO):
                try:
                    triples.extend(self._visual_triples(
                        self.substrate.get(rec.content_hash), rec.modality))
                except Exception as e:        # visual extraction is best-effort; log, don't swallow
                    self._degraded("visual-extract", e)
            return rec, triples, extracted_claims

        timed_out: list[MemoryRecord] = []
        deferred: list[MemoryRecord] = []
        deadline = max(0.0, float(getattr(self.settings, "consolidation_extract_deadline_sec", 0.0)))
        policy = str(getattr(self.settings, "consolidation_timeout_policy", "degrade")).lower()
        if deadline <= 0.0:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                extracted = list(ex.map(_extract, pending))
        else:
            ex = ThreadPoolExecutor(max_workers=max_workers)
            future_by_id = {ex.submit(_extract, rec): rec for rec in pending}
            done, not_done = wait(future_by_id, timeout=deadline, return_when=ALL_COMPLETED)
            result_by_id: dict[str, tuple[MemoryRecord, list[dict], list[dict]]] = {}
            for fut in done:
                rec = future_by_id[fut]
                try:
                    result_by_id[rec.memory_id] = fut.result()
                except Exception as e:  # _extract is best-effort, but keep the batch alive anyway.
                    self._degraded("extract-future", e)
                    result_by_id[rec.memory_id] = (rec, [], [])
            for fut in not_done:
                rec = future_by_id[fut]
                fut.cancel()
                timed_out.append(rec)
                rec.metadata["consolidation_timeout"] = True
                rec.metadata["consolidation_timeout_sec"] = deadline
                if policy == "defer":
                    deferred.append(rec)
                    self.store.upsert_record(rec)
                    continue
                result_by_id[rec.memory_id] = (rec, [], [])
            # Do not hold the whole sleep/query path waiting for slow network calls. Running calls may
            # finish in the background; their result is ignored because raw memory is already stored.
            ex.shutdown(wait=False, cancel_futures=True)
            extracted = [result_by_id[r.memory_id] for r in pending if r.memory_id in result_by_id]

        facts = events_total = claims_total = 0
        rel_by_id: dict[str, list[str]] = {}
        anchor_events_by_scope: dict[str, list[EventRecord]] = {}
        from .smqe.claim_extraction import claims_for_record
        # Pass 1: add facts/events (SEQUENTIAL -> contradiction ordering preserved), prefs.
        for rec, triples, extracted_claims in extracted:
            # BUDGET-DEFERRED record (drain mode above): it got zero extraction windows this
            # sleep purely because the batch budget ran out -- NOT because of the per-record
            # window threshold. Defer ALL derived work (claims/prefs/salience/typing) to the
            # retry sleep: claim_ids are random, so doing half the work now would duplicate
            # claims on retry. pending_consolidation stays True; the next sleep's drain slice
            # picks it up. This is the fix for the silent, PERMANENT graph-channel starvation
            # the extraction-audit fleet proved on 7 live namespaces.
            if (window_plan.get(rec.memory_id, 0) > 0
                    and window_allowance.get(rec.memory_id, 0) == 0
                    and rec.memory_id not in record_raw_only_ids):
                rec.metadata["consolidation_deferred"] = True
                self.store.upsert_record(rec)
                continue
            rec.metadata.pop("consolidation_deferred", None)
            anchor_events = anchor_events_by_scope.setdefault(
                rec.scope.key(),
                list(self.store.events_in_scope(rec.scope.namespace, scope=rec.scope)),
            )
            entities, seen = [], set()
            for t in triples:
                for e in (t["src"], t["dst"]):
                    if e.lower() not in seen:
                        seen.add(e.lower())
                        entities.append(e)
            date_ranges = normalize_dates(rec.text, rec.valid_at, anchor_events)
            for t in triples:
                event_text = self._event_source_window(rec.text, t)
                event_ranges = (
                    normalize_dates(event_text, rec.valid_at, anchor_events)
                    if event_text and event_text != rec.text else date_ranges
                )
                # Anchor events to the most local source sentence when possible, falling back to
                # whole-record dates and then the session date. Long benchmark sessions often pack
                # many dated facts into one memory; per-event windows keep their calendar entries
                # from all inheriting the first explicit date in the session.
                ev_range = self._event_epochs(event_ranges or date_ranges, rec.valid_at)
                _edge, invalidated = self.graph.add_fact(
                    t["src"], t["relation"], t["dst"], fact=t["fact"],
                    source_memory_id=rec.memory_id, valid_at=rec.valid_at, scope=rec.scope)
                if invalidated:                     # C2: an update closed an older edge
                    self._brain(BrainEventType.SUPERSEDED, namespace=rec.scope.namespace,
                                memory_ids=[rec.memory_id], closed=len(invalidated))
                facts += 1
                ev = EventRecord(
                    subject=t["src"], verb=t["relation"], object=t["dst"], fact=t["fact"],
                    aliases=(
                        event_aliases_from_text(rec.text, t)
                        if self.settings.event_alias_expansion_enabled
                        else [t["fact"], f"{t['src']} {t['dst']}", f"{t['relation']} {t['dst']}"]
                    ),
                    start=ev_range[0], end=ev_range[1],
                    source_memory_id=rec.memory_id, namespace=rec.scope.namespace,
                    valid_at=rec.valid_at,
                )
                self.store.add_event(ev)
                anchor_events.append(ev)
                events_total += 1
            if self.settings.pref_sentence_scan_enabled:
                # Scan every sentence/turn so mid-conversation preferences become profile lines.
                # Store a canonical profile line and key so casing/whitespace/simple paraphrase
                # variants across sessions don't bloat the profile.
                lines = preferences.extract_all_preferences(rec.text)
                for line in lines:
                    profile_line = preferences.canonicalize_preference(line) or line
                    self.store.add_profile_line(
                        rec.scope.namespace,
                        profile_line,
                        salience=rec.salience,
                        dedup_key=preferences.preference_dedup_key(profile_line),
                        source_memory_id=rec.memory_id,
                        content_hash=rec.content_hash,
                        raw_uri=rec.raw_uri,
                        valid_at=rec.valid_at,
                        scope=rec.scope,
                    )
                if lines:
                    rec.metadata["type"] = "preference"
            elif preferences.is_preference(rec.text):
                pref = preferences.extract_preference(rec.text)
                if pref:
                    self.store.add_profile_line(
                        rec.scope.namespace,
                        pref,
                        salience=rec.salience,
                        source_memory_id=rec.memory_id,
                        content_hash=rec.content_hash,
                        raw_uri=rec.raw_uri,
                        valid_at=rec.valid_at,
                        scope=rec.scope,
                    )
                    rec.metadata["type"] = "preference"
            # The affect scorer below returns importance too, fully overwriting this value and
            # salience: paying the separate importance call for a record the affect branch will
            # score is one wasted flash call per record. Skip it here; the affect block restores
            # baseline scoring when the scorer fails or omits the importance key.
            affect_will_score = (
                self.settings.affect_salience_enabled
                and "arousal" not in rec.metadata
                and callable(getattr(self.client, "score_affect", None))
            )
            if score_importance and not affect_will_score:
                rec.importance = self.client.score_importance(rec.text)   # real qwen-flash
                rec.salience = max(0.0, min(1.0, 0.45 * rec.surprise + 0.55 * rec.importance))
            rec.entities = entities
            if window_plan.get(rec.memory_id, 0) > 0 and window_allowance.get(rec.memory_id, 0) == 0:
                reason = (
                    "record_window_threshold"
                    if rec.memory_id in record_raw_only_ids
                    else "batch_window_budget"
                )
                rec.metadata["consolidation_raw_only"] = reason
                if rec.memory_id in record_raw_only_ids:
                    rec.metadata["consolidation_raw_only_window_threshold"] = raw_only_window_threshold
                    rec.metadata["consolidation_windows_planned"] = window_plan.get(rec.memory_id, 0)
            # MIRIX role typing on the async consolidate path via the token-free deterministic
            # classifier (reads text/modality only). This activates the MEMORY_TYPING retrieval
            # prior on the write path the bench uses -- the fast ingest path never sets
            # metadata.type. A record already typed "preference" above keeps that stronger label.
            # Gated (default OFF).
            if self.settings.memory_typing_enabled and rec.metadata.get("type") != "preference":
                rec.metadata["type"] = classify_record(rec).value
            rec.metadata["pending_consolidation"] = False
            rec.metadata["dates"] = [r["start"] for r in date_ranges]
            rel_by_id[rec.memory_id] = [t["relation"] for t in triples]
            if self.settings.claim_extraction_enabled:
                claims = claims_for_record(rec, triples=triples, extracted_claims=extracted_claims)
                if claims:
                    claims_total += self.store.add_claims(claims)
                    rec.metadata["claims_extracted"] = len(claims)
            # Affect-modulated salience for the async write path (the fast LLM-free ingest defers
            # it): one bounded qwen-flash affect call per record during sleep, so replay/retrieve
            # policy sees what mattered emotionally even for bulk-ingested sessions. Age-free.
            if self.settings.affect_salience_enabled and "arousal" not in rec.metadata:
                scorer = getattr(self.client, "score_affect", None)
                if callable(scorer):
                    try:
                        aff = scorer(rec.text)
                        emph = salience_mod.emphasis_score(rec.text)
                        s = self.settings
                        if "importance" in aff:
                            rec.importance = float(aff["importance"])
                        elif score_importance:
                            # affect result carries no importance: the baseline scoring that
                            # was skipped above still owes this record its importance value.
                            rec.importance = self.client.score_importance(rec.text)
                        rec.salience = salience_mod.affect_salience(
                            float(aff.get("arousal", 0.3)), rec.importance, rec.surprise, emph, 0.0,
                            w_arousal=s.affect_w_arousal, w_importance=s.affect_w_importance,
                            w_surprise=s.affect_w_surprise, w_emphasis=s.affect_w_emphasis,
                            w_helpful=s.affect_w_helpful)
                        rec.metadata.update({
                            "arousal": float(aff.get("arousal", 0.3)),
                            "valence": float(aff.get("valence", 0.0)),
                            "emphasis": emph,
                        })
                    except Exception:
                        # Consolidation must not die on one affect-scoring failure. Restore the
                        # baseline importance scoring that was skipped in favor of this branch,
                        # so a failed affect call never silently drops importance scoring.
                        if score_importance:
                            rec.importance = self.client.score_importance(rec.text)
                            rec.salience = max(
                                0.0, min(1.0, 0.45 * rec.surprise + 0.55 * rec.importance))
            self.store.upsert_record(rec)
            self.retriever.index_lexical(rec, save=False)

        # Graph features computed ONCE (was O(N^2) per-record), then a single structure pass.
        feats = self.graph.node_features(scope=scope)
        # Index-write tail under the write lock so a concurrent ingest cannot race the index here
        # (no model call inside: build_structure_code is pure, extraction already ran above).
        with self._write_lock:
            for rec, _, _ in extracted:
                gfeat: dict = {"relations": rel_by_id.get(rec.memory_id, [])}
                agg = [feats[e.lower()] for e in rec.entities if e.lower() in feats] if feats else []
                if agg:
                    gfeat["ppr"] = float(np.mean([a["ppr"] for a in agg]))
                    gfeat["degree"] = float(np.mean([a["degree"] for a in agg]))
                cur = self.index.get_vectors([rec.memory_id]).get(rec.memory_id)
                if cur is not None:
                    self.index.update(rec.memory_id, cur,
                                      sc.build_structure_code(rec, self.settings.struct_dim, gfeat))
                if self.settings.reflex_recall_enabled:
                    self.reflex_index.add_record(rec)   # rec.entities now populated -> index them
            for ns in {rec.scope.namespace for rec, _, _ in extracted}:
                self._bump_ns_version(ns)               # facts/entities changed -> invalidate cache
            self.index.save()
            self.retriever.save_lexical()
        return {"pending_processed": len(extracted), "facts_extracted": facts,
                "events_indexed": events_total,
                "claims_extracted": claims_total,
                "claim_coverage": (claims_total / len(extracted)) if extracted else 0.0,
                "extraction_timed_out": len(timed_out),
                "extraction_deferred": len(deferred),
                "extraction_windows_planned": planned_windows,
                "extraction_windows_submitted": submitted_windows,
                "extraction_window_cap_per_record": window_cap_per_record,
                "extraction_call_budget": call_budget,
                "extraction_raw_only_bounded": raw_only_bounded,
                "record_raw_only_bounded": record_raw_only_bounded,
                "extraction_partial_bounded": partial_bounded,
                "long_haystack_bounded": long_haystack_bounded,
                "long_haystack_raw_only": raw_only_long_haystack}

    @staticmethod
    def _event_epochs(date_ranges: list[dict],
                      default_epoch: Optional[float] = None) -> tuple[Optional[float], Optional[float]]:
        """Resolve an event's date range.

        Prefer explicit absolute dates and event-relative ranges anchored to a known calendar event;
        otherwise fall back to `default_epoch` (the session date) as a day range.
        """
        import re as _re
        from datetime import timedelta

        def _parse(r: dict) -> Optional[tuple[float, float]]:
            try:
                return (datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S").timestamp(),
                        datetime.strptime(r["end"], "%Y-%m-%dT%H:%M:%S").timestamp())
            except (ValueError, KeyError):
                return None

        for r in date_ranges:                       # prefer an explicit absolute/anchored date
            if r.get("anchored") or _re.search(r"\d{4}", r.get("expr", "")):
                got = _parse(r)
                if got:
                    return got
        if default_epoch:                            # anchor to the session date (day range)
            d = datetime.fromtimestamp(default_epoch).replace(hour=0, minute=0, second=0, microsecond=0)
            return (d.timestamp(), (d + timedelta(days=1) - timedelta(seconds=1)).timestamp())
        return (None, None)

    @staticmethod
    def _event_source_window(text: str, triple: dict[str, str]) -> str:
        """Return the sentence/span most likely to support an extracted event triple.

        The extractor emits triples, not character offsets. For event dating we need a local text
        window, otherwise every event in a multi-date session inherits whichever absolute date
        appeared first. This scorer is deliberately simple: choose the sentence with the largest
        overlap against the triple's subject/relation/object/fact tokens.
        """
        if not text:
            return text
        terms: set[str] = set()
        for field in ("src", "relation", "dst", "fact"):
            terms.update(t for t in re.findall(r"[a-z0-9]+", str(triple.get(field, "")).lower())
                         if len(t) > 2)
        if not terms:
            return text
        pieces = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        if not pieces:
            return text
        fact = str(triple.get("fact", "") or "").lower().strip()
        dst = str(triple.get("dst", "") or "").lower().strip()
        scored: list[tuple[int, int, str]] = []
        for idx, sentence in enumerate(pieces):
            low = sentence.lower()
            sent_terms = set(re.findall(r"[a-z0-9]+", low))
            score = len(terms & sent_terms)
            if fact and fact in low:
                score += 3
            if dst and dst in low:
                score += 2
            if score:
                scored.append((score, idx, sentence))
        if not scored:
            return text
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    @staticmethod
    def _legacy_normalize_dates(text: str) -> list[str]:
        """(Deprecated) explicit-date extractor kept for reference; events.normalize_dates
        is the structured replacement."""
        import re

        out: set[str] = set()
        for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
            out.add(m.group(0))
        for m in re.finditer(r"\b(?:19|20)\d{2}\b", text):
            out.add(m.group(0))
        return sorted(out)

    # ---- DREAMING ENGINE: token-free continuous consolidation ------------
    def dream(self, *, scope: Optional[Scope] = None, replay: bool = True,
              infer: bool = True, multires: bool = True) -> dict:
        """Run one idle consolidation pass (token-free by default; additive derived layer
        only -- never mutates the lossless store). Returns per-phase stats."""
        scope = scope or Scope()
        out: dict = {}
        if replay:
            out["replay"] = self.dream_replay(scope=scope)
        if infer:
            out["infer"] = self.dream_infer(scope=scope)
        if multires:
            out["multires"] = self.dream_multires(scope=scope)
        # MemMA self-repair sweep: only when DREAM_REPAIR is on (dream() unchanged when off).
        if self.settings.dream_repair_enabled:
            out["repair"] = self.dream_repair(scope=scope)
        return out

    def dream_repair(self, *, scope: Optional[Scope] = None) -> dict:
        """MemMA evidence-grounded self-repair sweep (proposal-only; LLM-gated). Returns
        {skipped:'disabled'} unless DREAM_REPAIR is set."""
        from .dreaming import repair as _repair
        return _repair.run_sweep(self, scope or Scope())

    def dream_replay(self, *, scope: Optional[Scope] = None) -> dict:
        from .dreaming import replay as _replay
        return _replay.cycle(self, scope or Scope(), self.settings)

    def dream_infer(self, *, scope: Optional[Scope] = None) -> dict:
        from .dreaming import infer as _infer
        return _infer.derive(self, scope or Scope(), self.settings)

    def dream_multires(self, *, scope: Optional[Scope] = None) -> dict:
        from .dreaming import multires as _multires
        scope = scope or Scope()
        recs = self.store.all_records(scope)
        vmap = self.index.get_vectors([r.memory_id for r in recs])
        items = [(r.memory_id, vmap[r.memory_id]) for r in recs if r.memory_id in vmap]
        gists = _multires.build_tree(
            items, namespace=scope.namespace, levels=self.settings.dream_multires_levels,
            min_cluster=self.settings.dream_cluster_min)
        for g in gists:
            self.store.add_derived(g)
        if gists:
            self._brain(BrainEventType.DREAM_GIST_CREATED, namespace=scope.namespace,
                        gists=len(gists), members=len(items))
        return {"gists_built": len(gists), "members": len(items)}

    def build_prefetch(self, *, scope: Optional[Scope] = None,
                       query_texts: Optional[list[str]] = None) -> dict:
        """Cluster the query log (or provided queries), pre-assemble each cluster's CONTEXT
        (token-free: retrieval + assembly, no reader), and cache it keyed by centroid."""
        scope = scope or Scope()
        sk = scope.key()
        if query_texts:
            qvecs = self.client.embed_texts(query_texts)
        else:
            qvecs = np.array([v for k, v in self._query_log if k == sk], dtype=np.float32)
        if qvecs.shape[0] < 2:
            return {"clusters": 0, "note": "need >=2 logged queries"}
        labels, centroids = self.prefetch.cluster_queries(qvecs, self.settings.dream_prefetch_clusters)
        built = 0
        for c in range(centroids.shape[0]):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            rep_vec = qvecs[members[0]]                 # representative query of the cluster
            cands = self.retriever.retrieve("", at=None, scope=scope, qvec=rep_vec)
            blocks = self.retriever.assemble_context("", cands, at=None, scope=scope)
            self.prefetch.add(centroids[c], blocks)
            built += 1
        return {"clusters": built, "queries": int(qvecs.shape[0])}

    def prefetch_context(self, query_vec) -> Optional[list]:
        """Query-time match to a pre-assembled context (zero tokens). None on a miss."""
        return self.prefetch.get(query_vec)

    # ---- live age-independence proof (Section 5.5) -----------------------
    def prove_age_independence(self, *, scope: Optional[Scope] = None, k: int = 5,
                               max_n: int = 300) -> dict:
        """Compute recall@k and p95 retrieval latency vs memory AGE on the CURRENT store.

        Partial-cue probe (pattern completion): query with a prefix of each memory and
        check it returns among the top-k against the rest as distractors. TWO channels
        are probed so the claim covers what actually ships (WP5 closed the gap where
        only the raw index was proven while the ranker fused a recency channel):
          - index: the pure content index (index.search), the original claim;
          - full_path: retriever.retrieve() -- the SHIPPED hybrid fusion (dense + BM25 +
            graph, recency off by default). Rerank is skipped for cost: qwen3-rerank
            scores only (query, text) pairs and receives no timestamp, so it cannot
            reintroduce age; the fusion ordering is where age could enter, and it is
            probed directly.
        `flat` requires BOTH recall curves flat and index latency flat."""
        scope = scope or Scope()
        recs = [r for r in self.store.all_records(scope) if (r.text or "").strip()]
        if len(recs) < 5:
            return {"ok": False, "note": "need >=5 memories in scope to prove flatness",
                    "n": len(recs)}

        active = self.store.active_ids_at(scope=scope)
        recs = [r for r in recs if r.memory_id in active][:max_n]
        now_t = now()
        ages, hits, lat_ms, full_hits = [], [], [], []
        for r in recs:
            cue = r.text[: max(24, int(len(r.text) * 0.6))]  # partial cue
            qvec = self.client.embed_text(cue)  # real
            t0 = time.perf_counter()
            res = self.index.search(qvec, min(max(k * 4, k), len(self.index)))
            dt = (time.perf_counter() - t0) * 1000.0
            res = [(mid, s) for mid, s in res if mid in active][:k]
            hits.append(1 if any(mid == r.memory_id for mid, _ in res) else 0)
            full = self.retriever.retrieve(cue, scope=scope, qvec=qvec, skip_rerank=True)
            full_hits.append(1 if any(c.record.memory_id == r.memory_id
                                      for c in full[:k]) else 0)
            lat_ms.append(dt)
            ages.append((now_t - r.valid_at) / 86400.0)

        ages_a, hits_a, lat_a = np.array(ages), np.array(hits), np.array(lat_ms)
        full_a = np.array(full_hits)
        nbins = min(8, max(2, len(recs) // 4))
        edges = np.linspace(ages_a.min(), ages_a.max() + 1e-9, nbins + 1)
        centers, recall_bin, p95_bin, full_bin = [], [], [], []
        for b in range(nbins):
            mask = (ages_a >= edges[b]) & (ages_a < edges[b + 1])
            if mask.sum() == 0:
                continue
            centers.append(float((edges[b] + edges[b + 1]) / 2 / 365.25))
            recall_bin.append(float(hits_a[mask].mean()))
            p95_bin.append(float(np.percentile(lat_a[mask], 95)))
            full_bin.append(float(full_a[mask].mean()))
        rec_slope = float(np.polyfit(centers, recall_bin, 1)[0]) if len(centers) > 1 else 0.0
        lat_slope = float(np.polyfit(centers, p95_bin, 1)[0]) if len(centers) > 1 else 0.0
        full_slope = float(np.polyfit(centers, full_bin, 1)[0]) if len(centers) > 1 else 0.0
        return {
            "ok": True, "n": len(recs), "k": k,
            "overall_recall": float(hits_a.mean()),
            "overall_p95_ms": float(np.percentile(lat_a, 95)),
            "recall_slope_per_year": rec_slope,
            "latency_slope_ms_per_year": lat_slope,
            "age_centers_years": centers, "recall_per_bin": recall_bin, "p95_ms_per_bin": p95_bin,
            "full_path_overall_recall": float(full_a.mean()),
            "full_path_recall_slope_per_year": full_slope,
            "full_path_recall_per_bin": full_bin,
            "full_path_rerank_skipped": True,
            "flat": (abs(rec_slope) < 0.05 and abs(lat_slope) < 1.0
                     and abs(full_slope) < 0.05),
        }

    # ---- introspection ----------------------------------------------------
    def list_memories(self, scope: Optional[Scope] = None) -> list[MemoryRecord]:
        return sorted(self.store.all_records(scope), key=lambda r: -r.created_at)

    def get_record(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.store.get_record(memory_id)

    def get_raw(self, content_hash: str) -> bytes:
        return self.substrate.get(content_hash)

    def snap_back_audit(self, scope: Optional[Scope] = None) -> dict:
        """Snap-back fidelity as a falsifiable NUMBER (not a UI demo): for EVERY memory in scope
        that was content-addressed, confirm the immutable substrate still returns the byte-identical
        original (sha256(get_raw(content_hash)) == content_hash, via substrate.verify). Forgetting
        and fading lower ONLY the FSRS index-priority weight; the raw record is never mutated or
        deleted (the substrate refuses delete by design), so a faded memory snaps back losslessly.
        A missing blob or a hash mismatch is corruption -> surfaced in ``failures`` and counted
        against the rate, never hidden. ``rate == 1.0`` over the corpus is the guarantee the
        content-addressed store is meant to make; the bench reports this number directly."""
        records = self.store.all_records(scope)
        total = 0
        lossless = 0
        audited_hashes: set[str] = set()
        failures: list[dict] = []
        for rec in records:
            h = (rec.content_hash or "").strip()
            if not h:
                continue  # derived/profile records carry no raw blob -- nothing to snap back
            total += 1
            try:
                ok = self.substrate.verify(h)
            except Exception as e:  # missing object / read error == a real fidelity failure
                failures.append({"memory_id": rec.memory_id, "content_hash": h,
                                 "error": type(e).__name__})
                continue
            if ok:
                lossless += 1
                audited_hashes.add(h)
            else:
                failures.append({"memory_id": rec.memory_id, "content_hash": h,
                                 "error": "hash_mismatch"})
        return {"total": total, "lossless": lossless,
                "rate": (lossless / total) if total else 1.0,
                "audited_content_hashes": sorted(audited_hashes),
                "failures": failures}

    def stats(self, scope: Optional[Scope] = None) -> dict:
        return {
            "app_env": self.settings.app_env,
            "region": self.settings.region,
            "memories": self.store.count(scope),
            "edges": len(self.store.all_edges(scope)),
            "vectors": len(self.index),
            "vector_backend": type(self.index).__name__,
            "has_api_key": self.settings.has_api_key,
            "scope": scope.model_dump() if scope else None,
        }

    def memory_health_report(self, scope: Optional[Scope] = None) -> dict:
        """Read-only self-diagnosis of a scope: coverage, contradiction load, low-confidence and
        inferred facts, derived/replay debt, orphan records, and age spread. No model call, no
        fabricated numbers; every figure is counted from the store."""
        scope = scope or Scope()
        ns = scope.namespace
        recs = self.store.all_records(scope)
        edges = self.store.all_edges(scope)
        now_t = now()
        closed = sum(1 for e in edges if e.expired_at is not None or e.invalid_at is not None)
        inferred = sum(1 for e in edges if getattr(e, "inferred", False))
        low_conf = sum(1 for e in edges if getattr(e, "confidence", 1.0) < 0.5)
        pruned = sum(1 for e in edges if getattr(e, "pruned", False))
        orphans = sum(1 for r in recs if not r.entities)
        distinct_entities = {e.lower() for r in recs for e in r.entities}
        ages = [r.age_days(now_t) for r in recs]
        gists = self.store.derived_count(ns)
        return {
            "scope": scope.model_dump(),
            "memories": len(recs),
            "edges": len(edges),
            "derived_gists": gists,
            "events": len(self.store.events_in_scope(ns, scope=scope)),
            "distinct_entities": len(distinct_entities),
            "contradiction_load": closed,        # bi-temporally closed (superseded) edges
            "inferred_edges": inferred,
            "low_confidence_edges": low_conf,
            "pruned_edges": pruned,
            "orphan_records": orphans,           # records with no extracted entities
            "replay_debt": max(0, len(recs) - gists),   # raw memories not yet consolidated to a gist
            "age_days_min": min(ages) if ages else 0.0,
            "age_days_max": max(ages) if ages else 0.0,
            "has_api_key": self.settings.has_api_key,
        }

    def brain_health_score(self, scope: Optional[Scope] = None) -> dict:
        """A LOCAL diagnostic composite (Phase 8) -- NOT a benchmark. Rolls the health report, the
        BrainEvent stream, and the channel-win ledger into one BrainHealthScore in [0,1] with its
        components, so connectivity debt is visible at a glance. Every input is counted from the
        store / in-memory stream; nothing is fabricated or measured against held-out data."""
        scope = scope or Scope()
        h = self.memory_health_report(scope)
        counts = self.brain_log.counts(scope.namespace)        # scope-isolated, no cross-ns mixing
        wins = self.channel_win_stats(scope)
        mem = max(1, h["memories"])
        verified = counts.get("answer_verified", 0)
        abstained = counts.get("answer_abstained", 0)
        missed = counts.get("retrieval_missed", 0)
        answered = verified + abstained + missed

        recall_connectivity = 1.0 - min(1.0, h["orphan_records"] / mem)
        proof_coverage = (verified / answered) if answered else 0.0
        temporal_coverage = min(1.0, h["events"] / mem)
        channel_diversity = min(1.0, len(wins) / 4.0)            # >=4 distinct winners = full
        repair_readiness = 1.0 if (counts.get("repair_proposed", 0)
                                   or counts.get("repair_applied", 0)) else 0.5
        orphan_rate = min(1.0, h["orphan_records"] / mem)
        contradiction_rate = min(1.0, h["contradiction_load"] / max(1, h["edges"]))
        stale_gist_rate = min(1.0, h["replay_debt"] / mem)
        unsupported_rate = ((abstained + missed) / answered) if answered else 0.0

        good = (recall_connectivity + proof_coverage + temporal_coverage
                + repair_readiness + channel_diversity) / 5.0
        bad = (orphan_rate + contradiction_rate + stale_gist_rate + unsupported_rate) / 4.0
        score = max(0.0, min(1.0, good - 0.5 * bad))
        components = {
            "recall_connectivity": recall_connectivity, "proof_coverage": proof_coverage,
            "temporal_coverage": temporal_coverage, "repair_readiness": repair_readiness,
            "channel_diversity": channel_diversity, "orphan_rate": orphan_rate,
            "contradiction_rate": contradiction_rate, "stale_gist_rate": stale_gist_rate,
            "unsupported_answer_rate": unsupported_rate,
        }
        return {
            "scope": scope.model_dump(),
            "brain_health_score": round(score, 4),
            "components": {k: round(v, 4) for k, v in components.items()},
            "events": counts, "channel_wins": dict(wins),
        }

    def value_as_of(self, entity: str, relation: str, *, as_of: Optional[float] = None,
                    scope: Optional[Scope] = None) -> Optional[dict]:
        """C2 time-travel: the value of (entity, relation) VALID at `as_of` (now() if None), chosen
        DETERMINISTICALLY from bi-temporal edges -- not an LLM guess. None if nothing was valid then.
        This is the 'where did Alice work on date X' primitive Mem0 cannot answer."""
        from .graph import _norm
        scope = scope or Scope()
        at = now() if as_of is None else as_of
        cands = [e for e in self.store.all_edges(scope)
                 if _norm(e.src) == _norm(entity) and _norm(e.relation) == _norm(relation)
                 and not getattr(e, "inferred", False) and e.is_active_at(at)]
        if not cands:
            return None
        best = max(cands, key=lambda e: (e.valid_at, e.created_at))
        return {"entity": entity, "relation": relation, "value": best.dst,
                "valid_at": best.valid_at, "as_of": at, "source_memory_id": best.source_memory_id}

    def fact_history(self, entity: str, relation: str, *,
                     scope: Optional[Scope] = None) -> list[dict]:
        """C2 current-vs-historical: the full superseded chain for (entity, relation), oldest first,
        each with its validity window. Closed facts are RETAINED (never deleted) -- the visible
        supersession primitive the Mem0 issue says is missing."""
        from .graph import _norm
        scope = scope or Scope()
        now_t = now()
        edges = [e for e in self.store.all_edges(scope)
                 if _norm(e.src) == _norm(entity) and _norm(e.relation) == _norm(relation)
                 and not getattr(e, "inferred", False)]
        edges.sort(key=lambda e: (e.valid_at, e.created_at))
        return [{"value": e.dst, "valid_at": e.valid_at, "invalid_at": e.invalid_at,
                 "current": e.is_active_at(now_t), "source_memory_id": e.source_memory_id}
                for e in edges]

    def check_false_premise(self, question: str, *, scope: Optional[Scope] = None,
                            as_of: Optional[float] = None) -> Optional[dict]:
        """Track 3.3: detect a presuppositional question whose entities are TOTALLY disconnected in
        memory. Returns a structured abstention dict when the premise is unsupported, else None
        (proceed to the normal answer+NLI path). Deterministic, no model call.

        Rule (biased toward answering -- a false abstain is a recall regression): only when the
        question names >=2 entities AND no in-scope active memory co-mentions a pair AND no active
        graph edge directly connects a pair. Matching is case-insensitive; lowercase questions are
        backed by known memory/graph entity vocabulary plus conservative high-signal tokens.
        Coreference and aliases are not resolved (documented limits)."""
        from .events import parse_query
        from .graph import _norm
        scope = scope or Scope()
        at = now() if as_of is None else as_of
        active_records = self.store.active_records_at(at, scope)
        active_edges = self.store.active_edges_at(at, scope)
        ents = self._false_premise_entities(
            question, parse_query(question, at).get("entities", []), active_records, active_edges)
        ent_norms = list(dict.fromkeys(e.strip().lower() for e in ents if e.strip()))
        if len(ent_norms) < 2:
            return None
        for r in active_records:
            hay = (r.text or r.summary or "").lower()
            if sum(1 for e in ent_norms if e in hay) >= 2:
                return None                       # a memory co-mentions >=2 query entities
        for e in active_edges:
            ends = {_norm(e.src), _norm(e.dst)}
            if sum(1 for x in ent_norms if x in ends) >= 2:
                return None                       # an edge directly connects two query entities
        return {
            "abstain": True, "category": "missing_premise", "entities": ents,
            "message": ("I don't have evidence in memory connecting " + " and ".join(ents)
                        + ", so I can't answer a question that presupposes a relationship "
                          "between them."),
        }

    @staticmethod
    def _false_premise_entities(question: str, parsed_entities: list,
                                records: list[MemoryRecord], edges: list) -> list[str]:
        """Entity candidates for false-premise checks, including lowercase user questions."""
        tokens = re.findall(r"[a-z0-9]+", (question or "").lower())
        qterms = set(tokens)
        qphrase = " ".join(tokens)
        out: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            clean = str(name or "").strip()
            key = clean.lower()
            if len(key) < 2 or key in seen:
                return
            seen.add(key)
            out.append(clean)

        for ent in parsed_entities:
            add(str(ent))

        known_names: list[str] = []
        for rec in records:
            known_names.extend(str(e) for e in rec.entities if str(e).strip())
        for edge in edges:
            known_names.extend([str(getattr(edge, "src", "")), str(getattr(edge, "dst", ""))])
        for name in known_names:
            name_tokens = re.findall(r"[a-z0-9]+", name.lower().replace("_", " "))
            if not name_tokens:
                continue
            phrase = " ".join(name_tokens)
            terms = set(name_tokens)
            if phrase in qphrase or (len(terms) == 1 and terms & qterms) or len(terms & qterms) >= 2:
                add(name)

        if len(out) < 2:
            for tok in tokens:
                if (tok in seen or tok in _FALSE_PREMISE_STOP_TERMS
                        or tok in _FALSE_PREMISE_GENERIC_TERMS or len(tok) < 3):
                    continue
                add(tok.title() if tok.isalpha() else tok)
                if len(out) >= 4:
                    break
        return out

    def integrity_report(self, scope: Optional[Scope] = None) -> dict:
        """C1 operation-level integrity (HaluMem-style), provable from the BrainEvent stream + the
        store. Every rate is counted, never fabricated. Emits INTEGRITY_CHECKED. Needs BRAIN_EVENTS
        on (and traced recalls) for the answer-rate figures; the conflict load is always available."""
        scope = scope or Scope()
        counts = self.brain_log.counts(scope.namespace)
        verified = counts.get("answer_verified", 0)
        abstained = counts.get("answer_abstained", 0)
        missed = counts.get("retrieval_missed", 0)
        answered = verified + abstained + missed
        h = self.memory_health_report(scope)
        self._brain(BrainEventType.INTEGRITY_CHECKED, namespace=scope.namespace)
        return {
            "scope": scope.model_dump(),
            "answered": answered,
            # an answered-but-ungrounded recall is the fabrication-risk surface; we abstain instead.
            "fabrication_rate": round(missed / answered, 4) if answered else 0.0,
            "abstention_rate": round(abstained / answered, 4) if answered else 0.0,
            "verified_rate": round(verified / answered, 4) if answered else 0.0,
            "conflict_load": h["contradiction_load"],          # bi-temporally closed (superseded) edges
            "superseded_events": counts.get("superseded", 0),
            "memory_recalled": counts.get("memory_recalled", 0),
        }

    def memory_autopsy(self, failed_question: str, *, scope: Optional[Scope] = None,
                       at: Optional[float] = None) -> dict:
        """Read-only failure autopsy (Phase 5): diagnose WHY a question would miss, from
        deterministic store/index state -- no model call, no fabricated numbers. Turns a miss into
        a targeted repair class so dream/repair can aim at the real failure, not guess."""
        from .events import parse_query
        scope = scope or Scope()
        at = now() if at is None else at
        parsed = parse_query(failed_question, at)
        recs = self.store.active_records_at(at, scope)

        def _terms(t: str) -> set:
            return set(re.findall(r"[a-z0-9]+", (t or "").lower()))

        qterms = _terms(failed_question)
        matching = [r for r in recs if qterms & _terms(r.text or r.summary or "")]
        pending = [r for r in recs if r.metadata.get("pending_consolidation")]
        events = self.store.events_in_scope(scope.namespace, scope=scope, at=at)

        if not recs or not matching:
            diagnosis = "missing_write"
            action = "no in-scope memory mentions the query terms; ingest the source"
        elif pending:
            diagnosis = "pending_consolidation_not_run"
            action = "run sleep()/consolidate_pending to extract facts, events, and types"
        elif all(not r.entities for r in matching):
            diagnosis = "entity_extraction_failure"
            action = "consolidate to extract entities, or enable graph vocab seeding"
        elif parsed.get("ranges") and not events:
            diagnosis = "event_normalization_failure"
            action = "consolidate_pending to index temporal events for the date constraint"
        elif not self.index.get_vectors([r.memory_id for r in matching]):
            diagnosis = "vector_underfill"
            action = "the matching memories are not indexed; repair/rebuild the vector index"
        else:
            diagnosis = "retrieval_or_reader"
            action = ("info present and indexed; likely a rerank / reader / proof failure -- "
                      "needs a live probe to disambiguate")
        return {
            "question": failed_question, "scope": scope.model_dump(),
            "diagnosis": diagnosis, "suggested_action": action,
            "in_scope_memories": len(recs), "matching_memories": len(matching),
            "pending_consolidation": len(pending), "events_in_scope": len(events),
            "has_api_key": self.settings.has_api_key,
        }

    def apply_repair_proposals(self, proposals, *, scope: Optional[Scope] = None,
                               apply: bool = False) -> dict:
        """Guarded repair apply (Phase 5). Dry-run unless apply=True AND DREAM_REPAIR_APPLY is on;
        applied repairs are additive immutable ingests (INSERT/MERGE), never raw deletions."""
        from .dreaming.repair import apply_proposals
        return apply_proposals(self, proposals, scope or Scope(), apply=apply)

    def prove(self, answer, *, with_paths: bool = False, check_refs: bool = False) -> dict:
        """Proof tree for an Answer (provenance as a first-class output). Read-only.

        with_paths=True splices in the last RecallTrace's recall-path metadata (which channels
        surfaced each cited memory, gist provenance). The trace is matched to the answer by
        question text, so a cache hit or a stale trace simply yields the legacy (pathless) proof
        rather than misattributed paths. Default with_paths=False is byte-identical to before.

        check_refs=True RESOLVES every citation reference instead of asserting it: the raw
        bytes behind content_hash are fetched and re-hashed (tamper check) and the snippet is
        located in the cited record's text. Local reads only, no model call; additive keys."""
        from .proofs import prove_answer
        trace = self.retriever.last_trace if with_paths else None
        out = prove_answer(answer, trace)
        if check_refs:
            for item in out["evidence"]:
                checked = {"raw_resolves": False, "hash_matches": False,
                           "snippet_in_record": False}
                ch = item.get("content_hash") or ""
                if ch:
                    try:
                        checked["raw_resolves"] = True if self.substrate.get(ch) is not None else False
                        checked["hash_matches"] = bool(self.substrate.verify(ch))
                    except KeyError:
                        checked["raw_resolves"] = False
                rec = self.store.get_record(item.get("memory_id") or "")
                snippet = item.get("snippet") or ""
                if rec is not None and snippet:
                    norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
                    checked["snippet_in_record"] = norm(snippet)[:300] in norm(
                        rec.text or rec.summary or "")
                item["refs_checked"] = checked
            out["refs_verified"] = bool(out["evidence"]) and all(
                i["refs_checked"]["raw_resolves"] and i["refs_checked"]["hash_matches"]
                for i in out["evidence"])
        return out

    def recall_trace(self, *, scope: Optional[Scope] = None) -> Optional[RecallTrace]:
        """The RecallTrace from the most recent traced recall IN THE GIVEN SCOPE (None unless
        RECALL_TRACE is on; default scope when omitted). Explains why the last read
        found/missed what it did. Prefers the calling thread's own trace when it belongs to
        the requested scope (same-thread flows keep exact semantics); otherwise serves the
        scope's last COMPLETED published snapshot. Never returns another scope's trace -
        query text and memory ids stay inside the namespace boundary."""
        scope = scope or Scope()
        sk = scope.key()
        own = self.retriever.last_trace
        if own is not None and own.scope.key() == sk:
            return own
        with self._trace_snapshot_lock:
            return self._trace_snapshots.get(sk)

    @staticmethod
    def _claim_status(answer) -> str:
        """The final status of an answer's claim, from verified/note/NLI ONLY -- never influenced
        by whether a citation happened to source a supersession chain."""
        note = answer.note or ""
        if note.startswith("abstained"):
            return "abstained"
        if answer.verified:
            return "verified"
        if any(c.nli_label == NLILabel.CONTRADICTION for c in (answer.citations or [])):
            return "contradicted"
        return "unverified"

    def truth_ledger(self, answer, *, with_paths: bool = True,
                     scope: Optional[Scope] = None) -> dict:
        """Track 3.1: the complete chain from raw bytes to current truth. The proof tree
        (prove_answer: raw hash/span, NLI label, recall paths) enriched per citation with its
        bi-temporal validity window, whether it is still current, and the supersession chain of any
        fact it sourced (oldest first, closed facts retained). Adds claim_status (verified /
        contradicted / abstained / unverified). Read-only, deterministic, no model call."""
        from collections import defaultdict
        scope = scope or Scope()
        base = self.prove(answer, with_paths=with_paths)
        edges_by_src: dict[str, list] = defaultdict(list)
        for e in self.store.all_edges(scope):
            if e.source_memory_id and not getattr(e, "inferred", False):
                edges_by_src[e.source_memory_id].append(e)
        for item in base.get("evidence", []):
            rec = self.store.get_record(item["memory_id"])
            if rec is None:
                continue
            item["validity_window"] = {"valid_at": rec.valid_at, "invalid_at": rec.invalid_at,
                                       "expired_at": rec.expired_at}
            item["is_current"] = rec.is_active_at()
            chains, seen = [], set()
            for e in edges_by_src.get(rec.memory_id, []):
                key = (e.src.lower(), e.relation.lower())
                if key in seen:
                    continue
                seen.add(key)
                hist = self.fact_history(e.src, e.relation, scope=scope)
                if len(hist) > 1:                       # only when there IS a supersession chain
                    chains.append({"src": e.src, "relation": e.relation, "history": hist})
            if chains:
                item["supersession_chains"] = chains
        base["claim_status"] = self._claim_status(answer)
        return base

    def build_scratchpad(self, scope: Optional[Scope] = None, *, top_k: Optional[int] = None,
                         min_salience: Optional[float] = None, at: Optional[float] = None) -> list:
        """A small derived scratchpad of high-salience, verified, ACTIVE facts (Phase 6). Every
        entry links to the immutable source hash; it is a context channel, not a source of truth.
        Built fresh from ACTIVE records, so a superseded/invalidated fact expires automatically.
        Read-only, no model call."""
        from .scratchpad import select_scratchpad
        scope = scope or Scope()
        recs = self.store.active_records_at(now() if at is None else at, scope)
        return select_scratchpad(
            recs, top_k=top_k if top_k is not None else self.settings.scratchpad_topk,
            min_salience=(min_salience if min_salience is not None
                          else self.settings.scratchpad_min_salience),
            activation=self._flow_snapshot(scope.namespace),     # Track 9: field-warm facts surface
            weight=self.settings.flow_context_weight)

    def salience_explanation(self, memory_id: str,
                             scope: Optional[Scope] = None) -> Optional[dict]:
        """'Why I remember this strongly' (Phase 7): the affect/usage components behind a memory's
        salience plus its provenance. Read-only, no model call. None if the id is not in scope.
        The explanation is about ranking signals only, never an emotional diagnosis."""
        rec = (self.get_record_in_scope(memory_id, scope) if scope is not None
               else self.store.get_record(memory_id))
        if rec is None:
            return None
        md = rec.metadata or {}
        components = {
            "importance": round(float(rec.importance), 3),
            "surprise": round(float(rec.surprise), 3),
            "arousal": md.get("arousal"),
            "valence": md.get("valence"),
            "emphasis": md.get("emphasis"),
            "verified_helpful_count": int(getattr(rec, "verified_helpful_count", 0)),
        }
        preview, preview_source = self._salience_source_preview(rec)
        return {
            "memory_id": rec.memory_id,
            "salience": round(float(rec.salience), 3),
            "why": self._salience_why_text(components),
            "components": components,
            "component_sources": {
                "importance": "write-time scorer or consolidation scorer",
                "surprise": "write-time novelty against the current memory index",
                "arousal": "write-time affect scorer when affect salience is enabled",
                "valence": "write-time affect scorer when affect salience is enabled",
                "emphasis": "deterministic cues in the source text, such as explicit remember/important phrasing",
                "verified_helpful_count": "later verified answers that cited this exact memory",
            },
            "provenance": {"content_hash": rec.content_hash, "source": rec.source,
                           "raw_uri": rec.raw_uri, "raw_bytes_len": rec.raw_bytes_len,
                           "valid_at": rec.valid_at, "source_preview": preview,
                           "source_preview_from": preview_source},
            "limits": [
                "This explains memory priority from recorded salience signals, not hidden intent.",
                "Arousal and valence are model-scored signals, not a clinical or emotional diagnosis.",
                "The source preview is for audit; factual answers still require normal proof verification.",
            ],
        }

    def _salience_source_preview(self, rec: MemoryRecord, *, max_chars: int = 220) -> tuple[str, str]:
        if rec.content_hash:
            try:
                raw = self.substrate.get(rec.content_hash)
                text = raw.decode("utf-8", errors="replace")
                preview = " ".join(text.split())
                if preview:
                    return preview[:max_chars], "immutable_substrate"
            except Exception:
                pass
        preview = " ".join((rec.text or rec.summary or "").split())
        return preview[:max_chars], "record_text" if preview else "unavailable"

    @staticmethod
    def _salience_why_text(components: dict) -> str:
        labels = {
            "importance": "importance",
            "surprise": "novelty",
            "arousal": "arousal signal",
            "emphasis": "explicit emphasis",
            "verified_helpful_count": "verified reuse",
        }
        scored: list[tuple[str, float]] = []
        for key in ("importance", "surprise", "arousal", "emphasis"):
            value = components.get(key)
            if isinstance(value, (int, float)) and float(value) > 0.0:
                scored.append((key, float(value)))
        helpful = components.get("verified_helpful_count")
        if isinstance(helpful, int) and helpful > 0:
            scored.append(("verified_helpful_count", min(1.0, helpful / 5.0)))
        scored.sort(key=lambda item: item[1], reverse=True)
        top = [labels[key] for key, _value in scored[:3]]
        if not top:
            top = ["baseline salience"]
        return (
            "This memory is prioritized because recorded ranking signals include "
            + ", ".join(top)
            + ". This is a hedged memory-priority explanation, not a diagnosis of what the user felt."
        )

    def explain_candidate(self, memory_id: str) -> Optional[dict]:
        """'Why this memory?' (Phase 6): from the last RecallTrace, the channels that surfaced
        `memory_id`, its per-channel rank score and weight, fused score, and gist provenance.
        None when no trace exists or the id never appeared in the last retrieval."""
        trace = self.retriever.last_trace
        if trace is None:
            return None
        paths = trace.paths_for(memory_id)
        if not paths and memory_id not in trace.fused_scores:
            return None
        return {
            "memory_id": memory_id,
            "retrieval_paths": paths,
            "channel_ranks": {ch: len(trace.channel_results[ch]) - trace.channel_results[ch].index(memory_id)
                              for ch in paths},
            "channel_weights": {ch: float(trace.channel_weights.get(ch, 0.0)) for ch in paths},
            "fused_score": float(trace.fused_scores.get(memory_id, 0.0)),
            "via_gist": trace.gist_provenance.get(memory_id, ""),
            "selected": memory_id in trace.selected_candidates,
        }

    def evidence_packets(self, answer) -> list[EvidencePacket]:
        """The Answer's citations as portable EvidencePackets, enriched with recall paths from
        the last (matching) RecallTrace. The one evidence shape proof/repair/health all consume."""
        return build_evidence_packets(answer, self.retriever.last_trace)

    # ---- unified lifecycle (Phase 1) -------------------------------------
    def sleep(self, *, scope: Optional[Scope] = None, llm_summaries: bool = False) -> dict:
        """The one sleep path (consolidate_pending -> dream -> optional LLM summaries) via the
        shared LifecycleController, so API and MCP get identical sleep semantics. Free + offline
        when nothing is pending and llm_summaries is False."""
        return self.lifecycle.sleep(scope, llm_summaries=llm_summaries)

    def idle_tick(self, *, run_dream: bool = False, scope: Optional[Scope] = None) -> dict:
        """One idle optimization cadence (learn fusion weights from the dev buffer, optional dream)
        plus a connection-effectiveness snapshot. `FLOW_WARMUP` may run embeddings; no reader call."""
        return self.lifecycle.idle_tick(run_dream=run_dream, scope=scope)

    def auto_sleep_status(self, scope: Optional[Scope] = None) -> dict:
        """Read-only status for the host-agent background write drain."""
        return self.lifecycle.auto_sleep_status(scope)

    # ---- channel-win ledger + connection effectiveness (Phase 3) ---------
    def record_channel_wins(self, answer) -> dict:
        """Tally which retrieval channels surfaced the ENTAILMENT-confirmed sources of `answer`,
        read off the last matching RecallTrace and bucketed by the trace's namespace. In-memory
        only; never feeds a learner (the integrity wall lives in FeedbackBuffer). Returns the
        affected namespace's win bucket; empty dict when there is no matching trace."""
        trace = self.retriever.last_trace
        if trace is None or trace.query != answer.question:
            return {}
        bucket = self._channel_wins.setdefault(trace.scope.namespace, {})
        for c in answer.citations:
            if c.nli_label == NLILabel.ENTAILMENT:
                for ch in trace.paths_for(c.memory_id):
                    bucket[ch] = bucket.get(ch, 0) + 1
        return dict(bucket)

    def channel_win_stats(self, scope: Optional[Scope] = None) -> dict:
        """Channel-win counts (which channels actually win on confirmed answers). Scoped to one
        namespace, or merged across all when scope is None. A channel that never wins and never
        feeds proof/repair is debt -- wire it or keep it off."""
        if scope is not None:
            return dict(self._channel_wins.get(scope.namespace, {}))
        merged: dict[str, int] = {}
        for bucket in self._channel_wins.values():
            for ch, n in bucket.items():
                merged[ch] = merged.get(ch, 0) + n
        return merged

    def connection_effectiveness(self, scope: Optional[Scope] = None) -> dict:
        """A local report of how the enabled brain paths are firing: BrainEvent counts + the
        channel-win ledger, scoped to one namespace (or merged when scope is None). Read-only,
        no model call, no fabricated numbers."""
        counts = self.brain_log.counts(scope.namespace if scope else None)
        return {
            "events": counts,
            "channel_wins": self.channel_win_stats(scope),
            "total_events": sum(counts.values()),
        }
