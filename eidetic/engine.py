"""The engine: wires all seven components into the wake and sleep paths.

wake (per input):   ingest -> immutable store -> embed -> salience -> graph extract
                    -> structure code -> index. And ask -> retrieve -> verify ->
                    cite -> reconsolidate (strengthen what was used).
sleep (scheduled):  consolidate -> dedup/pattern-separation -> verified semantic
                    summaries -> FSRS index-priority decay. Never deletes raw.
"""
from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from datetime import datetime

from . import fsrs, preferences, salience as salience_mod, structure_code as sc
from .config import Settings, get_settings
from .dashscope_client import DashScopeClient, get_client
from .events import EventRecord, normalize_dates
from .graph import KnowledgeGraph
from .ingestion import IngestInput, from_bytes, from_file, from_text
from .models import Answer, MemoryRecord, Modality, NLILabel, Scope, now
from .retrieval import Retriever
from .semantic_cache import SemanticCache
from .store import RecordStore
from .substrate import make_substrate, sha256_hex
from .vector_index import make_vector_index

# Synaptic-tagging-and-capture window: a salient event up-weights temporally adjacent
# memories within this window (dossier 6.7 -> ~1 hour for the biological analog).
TAG_CAPTURE_WINDOW_SEC = 3600.0
TAG_CAPTURE_SALIENCE = 0.7  # only events at/above this salience tag their neighbors


class Engine:
    def __init__(self, settings: Optional[Settings] = None, client: Optional[DashScopeClient] = None):
        self.settings = settings or get_settings()
        self.client = client or get_client()
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
        # Prospective memory: a first-order Markov model of query-signature transitions.
        from .optim.markov import MarkovPrefetcher
        self.markov = MarkovPrefetcher()

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

        # SHA-256 dedup BEFORE embedding (cost control + provenance). Dedup is PER-SCOPE:
        # raw bytes are shared globally by the substrate, but the index record is distinct
        # per namespace so the same text in scope B never inherits scope A's record.
        h = sha256_hex(item.raw_bytes)
        existing = self.store.get_by_hash(h, scope)
        if existing is not None:
            return existing

        content_hash, raw_uri = self.substrate.put(item.raw_bytes)

        # Embed (real; allowed on the write path). Salience uses the in-scope index state
        # BEFORE this memory is added.
        content_vec = self.client.embed_text(item.text)

        triples: list[dict[str, str]] = []
        entities: list[str] = []
        if consolidate_now:
            sal = salience_mod.score(item.text, content_vec, self.index, self.client,
                                     self.store, scope)
            # Graph extraction (real qwen-plus); optional for cheap bulk ingest.
            if extract_graph and item.text.strip():
                triples.extend(self.client.extract_edges(item.text))
            # Vision FEEDS the graph: images/diagrams/tables become entities+edges.
            if extract_graph and item.modality in (Modality.IMAGE, Modality.VIDEO):
                triples.extend(self._visual_triples(item.raw_bytes, item.modality))
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

        record = MemoryRecord(
            content_hash=content_hash, modality=item.modality, raw_uri=raw_uri,
            raw_bytes_len=len(item.raw_bytes), text=item.text, is_described=item.is_described,
            source=item.source, scope=scope, valid_at=valid_at, entities=entities,
            surprise=sal.surprise, importance=sal.importance, salience=sal.salience,
            fsrs=fsrs.init_state(sal.importance, sal.surprise, valid_at),
            metadata={"pending_consolidation": not consolidate_now},
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

        self.index.add(record.memory_id, content_vec, struct_vec)
        self.index.save()
        self.store.upsert_record(record)
        self.retriever.index_lexical(record)

        # Synaptic tagging and capture: a salient event up-weights temporally adjacent
        # in-scope memories (FSRS priority only -- never the ranking score). Skipped on the
        # LLM-free fast path (its O(N) scan runs during consolidation instead).
        if consolidate_now and sal.salience >= TAG_CAPTURE_SALIENCE:
            self._tag_and_capture(record, scope)
        return record

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

    def _tag_and_capture(self, record: MemoryRecord, scope: Scope) -> int:
        """Up-weight retention of memories temporally adjacent to a salient event."""
        tagged = 0
        for r in self.store.all_records(scope):
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
        return self.ingest(from_text(text, source), valid_at=valid_at,
                           extract_graph=extract_graph, scope=scope,
                           consolidate_now=consolidate_now)

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

    # ---- wake: read path --------------------------------------------------
    def ask(self, query: str, *, at: Optional[float] = None, verify: bool = True,
            scope: Optional[Scope] = None, as_of: Optional[float] = None,
            use_cache: bool = True, reader_model: Optional[str] = None) -> Answer:
        scope = scope or Scope()
        read_at = as_of if as_of is not None else at
        sk = scope.key()
        # Prospective memory: learn P(next query-signature | current) for predictive prefetch.
        if self.settings.markov_prefetch_enabled:
            self._observe_query(query)
        use_cache = use_cache and self.settings.semantic_cache_enabled
        # Time-travel (as_of) queries are not cached (the answer depends on the as-of time).
        if as_of is not None:
            use_cache = False

        qvec = None
        if use_cache:
            hit = self.cache.get(sk, query, None)        # exact-hash (no embedding)
            if hit is not None:
                return hit
            qvec = self.client.embed_text(query)         # embed once, reuse in retrieval
            if len(self._query_log) < 5000:              # bounded query log for pre-fetch
                self._query_log.append((sk, qvec))
            hit = self.cache.get(sk, query, qvec)        # cosine >= threshold
            if hit is not None:
                return hit

        # When the idle learner is fed, retrieve candidates explicitly so per-channel
        # contributions are available for feedback; otherwise answer() retrieves internally
        # exactly as before (the default call signature is unchanged).
        precomputed = None
        if self.settings.feedback_enabled:
            precomputed = self.retriever.retrieve(query, at=read_at, scope=scope, qvec=qvec)
            ans = self.retriever.answer(query, at=read_at, verify=verify, scope=scope, qvec=qvec,
                                        precomputed=precomputed, reader_model=reader_model)
        else:
            ans = self.retriever.answer(query, at=read_at, verify=verify, scope=scope, qvec=qvec,
                                        reader_model=reader_model)

        # Reconsolidation as a write path (retrieval is no longer read-only).
        confirmed: list[str] = []
        for cit in ans.citations:
            rec = self.store.get_record(cit.memory_id)
            if rec is None:
                continue
            if cit.nli_label == NLILabel.ENTAILMENT:
                # CONFIRMED recall = immune affinity maturation: re-embed + up-weight.
                # Re-embedding refreshes the CONTENT vector; the FSRS boost is priority
                # only -- neither enters the ranking score, so recall stays age-independent.
                try:
                    self.index.update(rec.memory_id, self.client.embed_text(rec.text))
                except Exception:
                    pass
                fsrs.reinforce(rec.fsrs, importance=rec.importance)
                self.store.upsert_record(rec)
                confirmed.append(rec.memory_id)
            elif cit.nli_label == NLILabel.CONTRADICTION:
                # CONTRADICTED recall: suppress (down-weight), never delete.
                fsrs.lapse(rec.fsrs)
                self.store.upsert_record(rec)

        # Memory linking by co-activation: co-confirmed memories gain a strengthened edge.
        if len(confirmed) >= 2:
            self.graph.link_memories(confirmed, scope=scope, valid_at=read_at)
        if confirmed:
            self.index.save()
        if self.settings.feedback_enabled and precomputed is not None:
            self._emit_feedback(scope, query, qvec, precomputed, confirmed)
        if use_cache:
            self.cache.put(sk, query, qvec, ans)
        return ans

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
        except Exception:
            pass        # feedback is best-effort; never break the read path on a logging error

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
        self.markov.observe(self._query_signature(query))

    def predict_next_signatures(self, query: str, top_k: int = 3) -> list:
        """The Markov model's most-likely NEXT query signatures given the current query."""
        return self.markov.predict(self._query_signature(query), top_k=top_k)

    def reawaken(self, memory_id: str) -> Optional[MemoryRecord]:
        """Strong-cue reawakening: reset retrievability + boost stability (O(1))."""
        rec = self.store.get_record(memory_id)
        if rec is None:
            return None
        fsrs.reinforce(rec.fsrs, importance=max(0.6, rec.importance))
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
        from concurrent.futures import ThreadPoolExecutor

        pending = [r for r in self.store.all_records(scope)
                   if r.metadata.get("pending_consolidation")]
        if not pending:
            return {"pending_processed": 0, "facts_extracted": 0, "events_indexed": 0}

        def _extract(rec: MemoryRecord) -> tuple[MemoryRecord, list[dict]]:
            triples: list[dict[str, str]] = []
            if rec.text.strip():
                triples.extend(self.client.extract_edges(rec.text))   # real, concurrent
            if rec.modality in (Modality.IMAGE, Modality.VIDEO):
                try:
                    triples.extend(self._visual_triples(
                        self.substrate.get(rec.content_hash), rec.modality))
                except Exception:
                    pass
            return rec, triples

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            extracted = list(ex.map(_extract, pending))

        facts = events_total = 0
        rel_by_id: dict[str, list[str]] = {}
        # Pass 1: add facts/events (SEQUENTIAL -> contradiction ordering preserved), prefs.
        for rec, triples in extracted:
            entities, seen = [], set()
            for t in triples:
                for e in (t["src"], t["dst"]):
                    if e.lower() not in seen:
                        seen.add(e.lower())
                        entities.append(e)
            date_ranges = normalize_dates(rec.text, rec.valid_at)
            # Anchor events to the SESSION date (rec.valid_at) by default -- that is the
            # temporal ground truth in conversational memory -- preferring an explicit
            # absolute (year-bearing) date if the text states one. (Off-by-one date phrasing
            # is tolerated by the LongMemEval temporal judge.)
            ev_range = self._event_epochs(date_ranges, rec.valid_at)
            for t in triples:
                self.graph.add_fact(t["src"], t["relation"], t["dst"], fact=t["fact"],
                                    source_memory_id=rec.memory_id, valid_at=rec.valid_at,
                                    scope=rec.scope)
                facts += 1
                self.store.add_event(EventRecord(
                    subject=t["src"], verb=t["relation"], object=t["dst"], fact=t["fact"],
                    aliases=[t["fact"], f"{t['src']} {t['dst']}", f"{t['relation']} {t['dst']}"],
                    start=ev_range[0], end=ev_range[1],
                    source_memory_id=rec.memory_id, namespace=rec.scope.namespace,
                    valid_at=rec.valid_at,
                ))
                events_total += 1
            if preferences.is_preference(rec.text):
                pref = preferences.extract_preference(rec.text)
                if pref:
                    self.store.add_profile_line(rec.scope.namespace, pref, salience=rec.salience)
                    rec.metadata["type"] = "preference"
            if score_importance:
                rec.importance = self.client.score_importance(rec.text)   # real qwen-flash
                rec.salience = max(0.0, min(1.0, 0.45 * rec.surprise + 0.55 * rec.importance))
            rec.entities = entities
            rec.metadata["pending_consolidation"] = False
            rec.metadata["dates"] = [r["start"] for r in date_ranges]
            rel_by_id[rec.memory_id] = [t["relation"] for t in triples]
            self.store.upsert_record(rec)
            self.retriever.index_lexical(rec, save=False)

        # Graph features computed ONCE (was O(N^2) per-record), then a single structure pass.
        feats = self.graph.node_features(scope=scope)
        for rec, _ in extracted:
            gfeat: dict = {"relations": rel_by_id.get(rec.memory_id, [])}
            agg = [feats[e.lower()] for e in rec.entities if e.lower() in feats] if feats else []
            if agg:
                gfeat["ppr"] = float(np.mean([a["ppr"] for a in agg]))
                gfeat["degree"] = float(np.mean([a["degree"] for a in agg]))
            cur = self.index.get_vectors([rec.memory_id]).get(rec.memory_id)
            if cur is not None:
                self.index.update(rec.memory_id, cur,
                                  sc.build_structure_code(rec, self.settings.struct_dim, gfeat))

        self.index.save()
        self.retriever.save_lexical()
        return {"pending_processed": len(pending), "facts_extracted": facts,
                "events_indexed": events_total}

    @staticmethod
    def _event_epochs(date_ranges: list[dict],
                      default_epoch: Optional[float] = None) -> tuple[Optional[float], Optional[float]]:
        """Resolve an event's date range. Prefer an explicit ABSOLUTE (year-bearing) date
        from the text; otherwise anchor to `default_epoch` (the session date) as a day range;
        else (None, None)."""
        import re as _re
        from datetime import timedelta

        def _parse(r: dict) -> Optional[tuple[float, float]]:
            try:
                return (datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S").timestamp(),
                        datetime.strptime(r["end"], "%Y-%m-%dT%H:%M:%S").timestamp())
            except (ValueError, KeyError):
                return None

        for r in date_ranges:                       # prefer an explicit absolute date
            if _re.search(r"\d{4}", r.get("expr", "")):
                got = _parse(r)
                if got:
                    return got
        if default_epoch:                            # anchor to the session date (day range)
            d = datetime.fromtimestamp(default_epoch).replace(hour=0, minute=0, second=0, microsecond=0)
            return (d.timestamp(), (d + timedelta(days=1) - timedelta(seconds=1)).timestamp())
        return (None, None)

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
        check it returns among the top-k against the rest as distractors. Ranking uses
        content similarity only, so both curves come out flat regardless of age."""
        scope = scope or Scope()
        recs = [r for r in self.store.all_records(scope) if (r.text or "").strip()]
        if len(recs) < 5:
            return {"ok": False, "note": "need >=5 memories in scope to prove flatness",
                    "n": len(recs)}

        active = self.store.active_ids_at(scope=scope)
        recs = [r for r in recs if r.memory_id in active][:max_n]
        now_t = now()
        ages, hits, lat_ms = [], [], []
        for r in recs:
            cue = r.text[: max(24, int(len(r.text) * 0.6))]  # partial cue
            qvec = self.client.embed_text(cue)  # real
            t0 = time.perf_counter()
            res = self.index.search(qvec, min(max(k * 4, k), len(self.index)))
            dt = (time.perf_counter() - t0) * 1000.0
            res = [(mid, s) for mid, s in res if mid in active][:k]
            hits.append(1 if any(mid == r.memory_id for mid, _ in res) else 0)
            lat_ms.append(dt)
            ages.append((now_t - r.valid_at) / 86400.0)

        ages_a, hits_a, lat_a = np.array(ages), np.array(hits), np.array(lat_ms)
        nbins = min(8, max(2, len(recs) // 4))
        edges = np.linspace(ages_a.min(), ages_a.max() + 1e-9, nbins + 1)
        centers, recall_bin, p95_bin = [], [], []
        for b in range(nbins):
            mask = (ages_a >= edges[b]) & (ages_a < edges[b + 1])
            if mask.sum() == 0:
                continue
            centers.append(float((edges[b] + edges[b + 1]) / 2 / 365.25))
            recall_bin.append(float(hits_a[mask].mean()))
            p95_bin.append(float(np.percentile(lat_a[mask], 95)))
        rec_slope = float(np.polyfit(centers, recall_bin, 1)[0]) if len(centers) > 1 else 0.0
        lat_slope = float(np.polyfit(centers, p95_bin, 1)[0]) if len(centers) > 1 else 0.0
        return {
            "ok": True, "n": len(recs), "k": k,
            "overall_recall": float(hits_a.mean()),
            "overall_p95_ms": float(np.percentile(lat_a, 95)),
            "recall_slope_per_year": rec_slope,
            "latency_slope_ms_per_year": lat_slope,
            "age_centers_years": centers, "recall_per_bin": recall_bin, "p95_ms_per_bin": p95_bin,
            "flat": abs(rec_slope) < 0.05 and abs(lat_slope) < 1.0,
        }

    # ---- introspection ----------------------------------------------------
    def list_memories(self, scope: Optional[Scope] = None) -> list[MemoryRecord]:
        return sorted(self.store.all_records(scope), key=lambda r: -r.created_at)

    def get_record(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.store.get_record(memory_id)

    def get_raw(self, content_hash: str) -> bytes:
        return self.substrate.get(content_hash)

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
