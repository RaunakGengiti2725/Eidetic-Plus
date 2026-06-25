"""Component 6: reconstructive, verifiable retrieval (+ Component 7 provenance).

Pipeline (dossier Section 12):
  1. embed query (text-embedding-v4) -> ANN top-k1 with a bi-temporal filter
  2. in-app Personalized PageRank over the graph (associative expansion)
  3. Reciprocal Rank Fusion of the dense + graph rankings
  4. qwen3-rerank -> final top-k2
  5. qwen3-max generation strictly over the retrieved sources
  6. NLI entailment check with the IMMUTABLE raw record as premise -> reject/flag
     anything unentailed; attach a cited, bi-temporal provenance to every answer.

Recency-independence guarantee: ranking uses content similarity + association + a
rerank model. The FSRS priority weight is NEVER read here, so recall@k does not
depend on a memory's age. That is what the signature flat curve proves.
"""
from __future__ import annotations

import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .bm25 import BM25, PersistentBM25
from .config import Settings, get_settings
from .conflicts import CurrentValueResolution, resolve_current_value_question
from .dashscope_client import DashScopeClient
from .events import event_chain, parse_query, select_for_query
from .graph import KnowledgeGraph
from .models import (Answer, Citation, MemoryRecord, Modality, NLILabel,
                     RecallTrace, RetrievalCandidate, Scope, now)
from .optim import adaptive_k as _adaptive_k
from .optim import conformal as _conformal
from .optim import fusion as _fusion
from .optim import gating as _gating
from .optim import mmr as _mmr
from .optim import online_weights as _online_weights
from .optim import rocchio as _rocchio
from .store import RecordStore
from .substrate import Substrate
from .vector_index import VectorIndex

# Difficulty routing keywords for the answer cascade (flash -> plus -> max).
_HARD_KW = ("contradict", "no longer", "instead", "actually", "used to", "earlier said",
            "which is true", "correct", "still")
_TEMPORAL_MULTI_KW = ("before", "after", "when", "first", "last", "then", "during", "how long",
                      "how many", "both", "and also", "since", "until", "by the time")
_LATEST_KW = ("latest", "last", "newest", "current", "currently", "now", "today", "still", "recent")
_EARLIEST_KW = ("first", "earliest", "initial", "original")
_CHRONO_KW = ("before", "after", "then", "during", "when", "timeline", "order", "sequence")
_RELATIVE_DATE_RE = re.compile(
    r"\b(last|this|next)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
)
_TERM_RE = re.compile(r"[a-z0-9]+")


def _route_model(query: str, settings: Settings) -> str:
    """Route to a cascade tier by difficulty. Ambiguous defaults to the conservative
    middle tier (a wrong cheap answer costs more than an unnecessary escalation)."""
    q = query.lower()
    if any(w in q for w in _HARD_KW):
        return settings.gen_model            # qwen3-max: contradiction / hard
    if any(w in q for w in _TEMPORAL_MULTI_KW):
        return settings.extract_model        # qwen-plus: multi-hop / temporal
    if len(q.split()) <= 12:
        return settings.salience_model       # qwen-flash: short single-hop / preference
    return settings.extract_model            # conservative default


def _reader_model(query: str, settings: Settings) -> str:
    if not settings.reader_router_enabled:
        return settings.gen_model
    return _route_model(query, settings)


def _temporal_direction(query: str, parsed: dict) -> Optional[str]:
    q = query.lower()
    relative_date = bool(_RELATIVE_DATE_RE.search(q))
    latest_words = [w for w in _LATEST_KW if w != "last" or not relative_date]
    if any(w in q for w in latest_words):
        return "desc"
    if any(w in q for w in _EARLIEST_KW):
        return "asc"
    if parsed.get("ranges") or parsed.get("operation") == "order" or any(w in q for w in _CHRONO_KW):
        return "asc"
    return None


def _temporal_context_order(
    query: str,
    parsed: dict,
    candidates: list["RetrievalCandidate"],
) -> list["RetrievalCandidate"]:
    direction = _temporal_direction(query, parsed)
    if direction is None:
        return candidates
    reverse_time = direction == "desc"

    def key(item: tuple[int, "RetrievalCandidate"]):
        rank, cand = item
        ts = cand.record.valid_at
        missing = ts is None
        value = 0.0 if ts is None else float(ts)
        return (missing, -value if reverse_time else value, rank)

    return [cand for _, cand in sorted(enumerate(candidates), key=key)]


def _simple_terms(text: str) -> set[str]:
    terms = set(_TERM_RE.findall(text.lower().replace("_", " ")))
    for term in list(terms):
        if len(term) > 3 and term.endswith("s"):
            terms.add(term[:-1])
    return terms


def _hippo2_seed_entities(query: str, parsed: dict, store: RecordStore,
                          at: float, scope: Scope) -> list[str]:
    names = {str(e) for e in parsed.get("entities", []) if str(e).strip()}
    if not names:
        return []
    qterms = _simple_terms(query)
    out: list[str] = []
    for edge in store.active_edges_touching_many(names, at, scope):
        endpoint_terms = _simple_terms(f"{edge.src} {edge.dst}")
        edge_terms = _simple_terms(f"{edge.relation} {edge.fact}") - endpoint_terms
        if qterms & edge_terms:
            out.extend([edge.src, edge.dst])
    return list(dict.fromkeys(out))[:16]


def _vocab_seed_entities(query: str, corpus: list) -> list[str]:
    """Graph-seed discovery from in-scope STORE vocabulary: match query tokens against the entity
    names that actually occur in the scoped corpus (not only capitalized spans the parser caught).
    This finds graph seeds for lowercase / multi-word entities a NER-style parse would miss."""
    qterms = set(_TERM_RE.findall(query.lower()))
    if not qterms:
        return []
    out: list[str] = []
    for r in corpus:
        for e in getattr(r, "entities", []):
            el = str(e).lower()
            if el in qterms or (qterms & set(el.split())):
                out.append(e)
    return list(dict.fromkeys(out))[:16]


def _budget_blocks(blocks: list[str], token_budget: int) -> list[str]:
    """Token-budget the hybrid context (~4 chars/token) so the slice stays lean
    (lean-beats-full: a precise slice beats stuffing the whole noisy history)."""
    char_budget = token_budget * 4
    out, used = [], 0
    for b in blocks:
        if used >= char_budget:
            break
        take = b[: max(0, char_budget - used)]
        if take:
            out.append(take)
            used += len(take)
    return out


def _softmax(xs: list[float], temp: float = 0.15) -> list[float]:
    if not xs:
        return []
    a = np.array(xs, dtype=np.float64) / max(temp, 1e-6)
    a -= a.max()
    e = np.exp(a)
    s = e.sum()
    return (e / s).tolist() if s > 0 else [1.0 / len(xs)] * len(xs)


def _rrf(rankings: list[list[str]], k: int, weights: Optional[list[float]] = None) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion over several ordered id lists. Weights default to
    1.0 per channel (vanilla RRF); query-adaptive weights are passed by the caller. k=60."""
    scores: dict[str, float] = {}
    for i, ranking in enumerate(rankings):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, mid in enumerate(ranking):
            scores[mid] = scores.get(mid, 0.0) + w / (k + rank + 1)
    return scores


def edge_place(blocks: list[str]) -> list[str]:
    """Lost-in-the-middle mitigation: place the highest-scored evidence at the EDGES of the
    context (models attend best to the beginning and end). blocks are highest-first."""
    head, tail = [], []
    for i, b in enumerate(blocks):
        (head if i % 2 == 0 else tail).append(b)
    return head + tail[::-1]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def compress_chunk(text: str, query: str, ratio: float) -> str:
    """LLMLingua-2-style EXTRACTIVE compression for RAW chunks only (never structured facts).
    Keeps the top `ratio` fraction of sentences by query-term overlap. ratio>=1.0 -> no-op.
    This is an extractive approximation, not the LLMLingua-2 model (no torch dependency)."""
    if ratio >= 1.0 or not text:
        return text
    sents = _sentences(text)
    if len(sents) <= 2:
        return text
    qterms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = sorted(range(len(sents)),
                    key=lambda i: -len(set(re.findall(r"[a-z0-9]+", sents[i].lower())) & qterms))
    keep = max(1, int(len(sents) * ratio))
    keep_idx = sorted(scored[:keep])           # preserve original order
    return " ".join(sents[i] for i in keep_idx)


def _dedup(cands: list["RetrievalCandidate"]) -> list["RetrievalCandidate"]:
    """Drop exact and near-duplicate candidates (same content hash or identical text)."""
    seen_hash, seen_text, out = set(), set(), []
    for c in cands:
        h = c.record.content_hash
        t = (c.record.text or c.record.summary or "").strip().lower()
        if h in seen_hash or (t and t in seen_text):
            continue
        seen_hash.add(h)
        if t:
            seen_text.add(t)
        out.append(c)
    return out


class Retriever:
    def __init__(
        self,
        store: RecordStore,
        index: VectorIndex,
        graph: KnowledgeGraph,
        substrate: Substrate,
        client: DashScopeClient,
        settings: Optional[Settings] = None,
    ):
        self.store = store
        self.index = index
        self.graph = graph
        self.substrate = substrate
        self.client = client
        self.settings = settings or get_settings()
        self.bm25 = PersistentBM25(self.settings.index_dir / "bm25_index.json")
        # Connected Brain Loop: the last RecallTrace, populated only when RECALL_TRACE is on.
        # Observation-only side channel -- never read by ranking. THREAD-LOCAL so concurrent asks
        # never read each other's trace (last_trace was last-writer-wins shared state). Direct
        # assignment (retriever.last_trace = ...) stays valid same-thread via the property setter.
        self._trace_tl = threading.local()

    @property
    def last_trace(self) -> Optional[RecallTrace]:
        """The current THREAD's most recent RecallTrace (None until a traced retrieve on it)."""
        return getattr(self._trace_tl, "trace", None)

    @last_trace.setter
    def last_trace(self, value: Optional[RecallTrace]) -> None:
        self._trace_tl.trace = value

    def index_lexical(self, rec: MemoryRecord, *, save: bool = True) -> bool:
        """Update the persistent lexical channel on ingest. No-op when the flag is off."""
        if not self.settings.persistent_bm25_enabled:
            return False
        changed = self.bm25.add_or_update(rec.memory_id, rec.text or rec.summary or "")
        if changed and save:
            self.bm25.save()
        return changed

    def save_lexical(self) -> None:
        if self.settings.persistent_bm25_enabled:
            self.bm25.save()

    # ---- ground truth for verification -----------------------------------
    def _ground_truth(self, rec: MemoryRecord) -> str:
        """The premise for NLI: the immutable raw record where it is text, else the
        stored transcription/description (whose raw bytes remain ground truth)."""
        try:
            raw = self.substrate.get(rec.content_hash)
            text = raw.decode("utf-8")
            if text.strip():
                return text
        except (KeyError, UnicodeDecodeError):
            pass
        return rec.text

    # ---- retrieval --------------------------------------------------------
    def retrieve(self, query: str, at: Optional[float] = None, scope: Optional[Scope] = None,
                 qvec: Optional[np.ndarray] = None, use_recency: bool = True) -> list[RetrievalCandidate]:
        """Hybrid read path: dense + BM25 + single-step PPR + recency -> RRF -> rerank.

        Scope + bi-temporal as-of filter applied first. `qvec` may be passed to avoid a
        duplicate embedding (the semantic cache embeds once). Recency is a MINOR RRF
        channel for benchmark accuracy; the age-independence proofs use the pure content
        index (index.search), so the flat recall-vs-age claim is unaffected."""
        at = now() if at is None else at
        scope = scope or Scope()
        if len(self.index) == 0:
            return []
        if qvec is None:
            qvec = self.client.embed_text(query)  # real call

        # Scope + bi-temporal as-of filter -> the in-scope, currently-valid corpus.
        corpus = self.store.active_records_at(at, scope)
        if not corpus:
            return []
        # Index pruning by STATIC salience (surprise+importance; NO time term, so
        # age-independent). Default 0.0 = off. Never touches the immutable WORM store.
        if self.settings.salience_prune_threshold > 0.0:
            corpus = [r for r in corpus if r.salience >= self.settings.salience_prune_threshold]
        if not corpus:
            return []
        records = {r.memory_id: r for r in corpus}

        parsed = parse_query(query, at)  # operation / entities / is_namey / is_multihop
        s = self.settings

        # Connected Brain Loop: RecallTrace instrumentation is fully gated -- when off, not a
        # single extra call runs and the candidate list is byte-identical to the baseline path.
        record_trace = s.recall_trace_enabled
        _lat: dict[str, float] = {}
        _t0 = time.perf_counter() if record_trace else 0.0
        _mark = _t0

        allowed = set(records)
        # 3b Rocchio PRF: confidence-gated query expansion toward the top evidence centroid.
        if s.rocchio_enabled:
            qvec = self._maybe_rocchio(qvec, allowed)
        # 2a Adaptive efSearch: widen the HNSW beam only for hard (multi-hop / long) queries.
        ef_override = None
        if s.adaptive_ef_enabled and (parsed["is_multihop"] or len(query.split()) > 16):
            ef_override = s.hnsw_ef_search_hard

        # Channels 1, 2, 4 are independent given (corpus, qvec); 3 (PPR) needs dense seeds.
        def _run_dense():
            if ef_override is None:           # default path: unchanged call signature
                return self.index.search(qvec, s.ann_topk, allowed_ids=allowed)
            return self.index.search(qvec, s.ann_topk, allowed_ids=allowed, ef=ef_override)

        # S3 PARALLEL_CHANNELS fix: do the persistent-BM25 backfill (a WRITE + save) BEFORE the
        # parallel fan-out, so every channel callback is READ-ONLY -- no save() inside _run_bm25 can
        # race a concurrent _run_dense. The backfill is serial here; bm25.save is atomic (temp+replace).
        if s.persistent_bm25_enabled:
            changed = self.bm25.ensure_indexed(
                (r.memory_id, r.text or r.summary or "") for r in corpus)
            if changed:
                self.bm25.save()

        def _run_bm25():
            if s.persistent_bm25_enabled:
                return self.bm25.search(query, s.ann_topk, allowed_ids=allowed)   # read-only
            bm25 = BM25().index([(r.memory_id, r.text or r.summary or "") for r in corpus])
            return bm25.search(query, s.ann_topk)

        def _run_recency():
            if not use_recency:
                return []
            return [r.memory_id for r in sorted(corpus, key=lambda r: -r.valid_at)][: s.ann_topk]

        # 2e Parallel channel fan-out: dense + BM25 + recency concurrently (latency ~= slowest).
        if s.parallel_channels_enabled:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as ex:
                fd, fb, fr = ex.submit(_run_dense), ex.submit(_run_bm25), ex.submit(_run_recency)
                dense, bm25_hits, recency_order = fd.result(), fb.result(), fr.result()
        else:
            dense, bm25_hits, recency_order = _run_dense(), _run_bm25(), _run_recency()

        dense_order = [mid for mid, _ in dense]
        dense_map = dict(dense)
        bm25_order = [mid for mid, _ in bm25_hits]
        bm25_map = dict(bm25_hits)

        # Channel 3: single-pass PPR seeded from QUERY-linked entities (+ best dense hits'
        # entities), letting activation reach passages sharing no query words. No LLM loop.
        seed_entities: list[str] = list(parsed["entities"])
        if s.hippo2_seeding_enabled:
            seed_entities.extend(_hippo2_seed_entities(query, parsed, self.store, at, scope))
        if s.graph_vocab_seeding:
            seed_entities.extend(_vocab_seed_entities(query, corpus))
        for mid, _ in dense[:10]:
            seed_entities.extend(records[mid].entities)
        graph_scores = self.graph.score_memories(
            seed_entities, corpus, at, scope) if seed_entities else {}
        graph_order = [mid for mid, _ in sorted(graph_scores.items(), key=lambda x: -x[1])]

        # Query-adaptive weighted fusion: BM25 up for name/date/ID queries, graph up for
        # multi-hop. `rankings` feed rank-based fusion (RRF/Borda); `score_maps` feed the
        # score-based variants (z-score/min-max/DBSF). Both stay aligned with `weights`.
        wd, wb, wg = self._content_weights()
        rankings = [dense_order, bm25_order]
        weights = [wd, wb * (1.6 if parsed["is_namey"] else 1.0)]
        score_maps: list[dict] = [dense_map, bm25_map]
        channel_names = ["dense", "bm25"] if record_trace else None
        if graph_order:
            rankings.append(graph_order)
            weights.append(wg * (1.6 if parsed["is_multihop"] else 1.0))
            score_maps.append(graph_scores)
            if record_trace:
                channel_names.append("graph")
        # Phase-1 multi-view channels (each appends only when its flag is on; neutral path
        # unchanged when off). Provenance for gist boosts is recorded for prove_answer.
        self._gist_provenance: dict = {}
        if s.struct_channel_enabled:
            so, sm = self._run_struct(parsed, allowed)
            if so:
                rankings.append(so); weights.append(s.rrf_w_struct); score_maps.append(sm)
                if record_trace:
                    channel_names.append("struct")
        if s.event_ranking_enabled:
            eo, em = self._run_event(parsed, records, at, scope)
            if eo:
                rankings.append(eo); weights.append(s.rrf_w_event); score_maps.append(em)
                if record_trace:
                    channel_names.append("event")
        if s.gist_channel_enabled:
            go, gm, self._gist_provenance = self._run_gist(qvec, scope, allowed)
            if go:
                rankings.append(go); weights.append(s.rrf_w_gist); score_maps.append(gm)
                if record_trace:
                    channel_names.append("gist")
        if s.coactivation_channel_enabled:
            co, cm = self._run_coactivation(dense, records, at, scope, allowed)
            if co:
                rankings.append(co); weights.append(s.rrf_w_coact); score_maps.append(cm)
                if record_trace:
                    channel_names.append("coactivation")
        if recency_order:
            rankings.append(recency_order)
            weights.append(s.rrf_w_recency)
            n_rec = len(recency_order)
            score_maps.append({mid: float(n_rec - i) for i, mid in enumerate(recency_order)})
            if record_trace:
                channel_names.append("recency")
        if record_trace:
            _lat["channels_ms"] = (time.perf_counter() - _mark) * 1000.0
            _mark = time.perf_counter()
        fused = self._fuse(rankings, score_maps, weights)
        # Memory typing coordinator (Phase 4): a soft, flag-gated prior that nudges the candidates
        # whose MIRIX type matches the query class. Bounded to a fraction of the top fused score,
        # so it re-orders ties without overriding strong content matches. Off -> fused untouched.
        if s.memory_typing_enabled:
            self._apply_type_prior(fused, records, parsed, query)
        # Affect salience boost (Phase 3): a small, bounded, AGE-FREE nudge by static salience.
        if s.affect_salience_enabled and s.lambda_salience != 0.0:
            self._apply_salience_boost(fused, records)
        if record_trace:
            _lat["fuse_ms"] = (time.perf_counter() - _mark) * 1000.0
            _mark = time.perf_counter()
        if len(fused) < s.final_topk and use_recency:
            for mid in recency_order:
                fused.setdefault(mid, 0.0)
                if len(fused) >= s.final_topk:
                    break

        cands = {mid: RetrievalCandidate(
            record=records[mid], dense_score=dense_map.get(mid, 0.0),
            bm25_score=bm25_map.get(mid, 0.0), graph_score=graph_scores.get(mid, 0.0),
            fused_score=fused[mid]) for mid in fused}
        ranked = _dedup(sorted(cands.values(), key=lambda c: -c.fused_score))
        final = self._finalize(query, ranked)
        if record_trace:
            _lat["finalize_ms"] = (time.perf_counter() - _mark) * 1000.0
            _lat["total_ms"] = (time.perf_counter() - _t0) * 1000.0
            sel = [c.record.memory_id for c in final]
            sel_set = set(sel)
            self.last_trace = RecallTrace(
                query=query, scope=scope, parsed_query=parsed,
                enabled_channels=list(channel_names),
                channel_results={n: list(r) for n, r in zip(channel_names, rankings)},
                channel_weights={n: float(w) for n, w in zip(channel_names, weights)},
                fused_scores={k: float(v) for k, v in fused.items()},
                gist_provenance=dict(self._gist_provenance),
                selected_candidates=sel,
                dropped_candidates=[mid for mid in fused if mid not in sel_set],
                latency_by_stage=_lat, token_budget=s.context_token_budget,
            )
        return final

    # ---- online weight learning + PRF ------------------------------------
    def _content_weights(self) -> tuple[float, float, float]:
        """Base (dense, bm25, graph) fusion weights. When the online learner is enabled and a
        learned vector has been written to index_dir/fusion_weights.json, use it; otherwise
        the static config floats. RECENCY is never learned here (age-independence)."""
        s = self.settings
        if s.fusion_learner_enabled:
            learned = _online_weights.load_weights(self.settings.index_dir / "fusion_weights.json")
            if learned:
                return (float(learned.get("dense", s.rrf_w_dense)),
                        float(learned.get("bm25", s.rrf_w_bm25)),
                        float(learned.get("graph", s.rrf_w_graph)))
        return s.rrf_w_dense, s.rrf_w_bm25, s.rrf_w_graph

    def _maybe_rocchio(self, qvec: np.ndarray, allowed: set) -> np.ndarray:
        """A single confidence-gated PRF expansion: cheap dense probe -> if the top match is
        strong, push the query toward the top-R evidence centroid. No model call."""
        s = self.settings
        try:
            probe = self.index.search(qvec, max(s.rocchio_topr, 1), allowed_ids=allowed)
        except TypeError:
            return qvec
        if not probe or not _rocchio.should_expand(probe[0][1], s.rocchio_conf_gate):
            return qvec
        ids = [mid for mid, _ in probe[: s.rocchio_topr]]
        vmap = self.index.get_vectors(ids) if hasattr(self.index, "get_vectors") else {}
        rel = [vmap[mid] for mid in ids if mid in vmap]
        if not rel:
            return qvec
        return _rocchio.rocchio_expand(qvec, rel, alpha=s.rocchio_alpha, beta=s.rocchio_beta)

    # ---- Phase-1 multi-view retrieval channels (dormant signals, flag-gated) ----------
    def _run_struct(self, parsed: dict, allowed: set) -> tuple[list[str], dict]:
        """Structure-code channel: rank by entity/role/modality similarity in STRUCTURE space.
        Age-safe: the query structure code carries no temporal dimension, and the stored codes
        encode only cyclic (not absolute-age) time, so this never slopes recall-vs-age."""
        from . import structure_code as _sc
        qstruct = _sc.build_query_structure_code(list(parsed.get("entities", [])),
                                                 self.settings.struct_dim)
        try:
            hits = self.index.search_struct(qstruct, self.settings.ann_topk)
        except Exception:
            return [], {}
        hits = [(mid, sc) for mid, sc in hits if mid in allowed]
        return [mid for mid, _ in hits], dict(hits)

    def _run_event(self, parsed: dict, records: dict, at, scope: Scope) -> tuple[list[str], dict]:
        """Event-overlap channel: promote memories whose normalized event interval matches the
        QUERY's temporal constraint (filter/count/order). Ranks by query-time match, NOT by the
        memory's age, so it does not affect the flat recall-vs-age curve."""
        events = self.store.events_in_scope(scope.namespace)
        if not events:
            return [], {}
        matched = select_for_query(events, parsed, at)
        order, m = [], {}
        n = len(matched)
        for rank, ev in enumerate(matched):
            mid = getattr(ev, "source_memory_id", "")
            if mid and mid in records and mid not in m:
                order.append(mid)
                m[mid] = float(n - rank)        # higher = earlier in the temporal match order
        return order, m

    def _run_gist(self, qvec, scope: Scope, allowed: set) -> tuple[list[str], dict, dict]:
        """Derived-gist channel: a gist that matches the query boosts its RAW member memories
        (gists help recall but never replace raw evidence). Returns (order, score_map, provenance:
        member_id -> gist cid) so prove_answer can show recall came via a gist."""
        gists = self.store.derived_in_scope(scope.namespace)
        if not gists or qvec is None:
            return [], {}, {}
        q = np.asarray(qvec, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-9
        scored = []
        for g in gists:
            if not getattr(g, "vector", None):
                continue
            gv = np.asarray(g.vector, dtype=np.float32)
            sim = float(gv @ q / ((np.linalg.norm(gv) + 1e-9) * qn))
            scored.append((g, sim))
        scored.sort(key=lambda x: -x[1])
        order, m, prov = [], {}, {}
        for g, sim in scored[:8]:
            for mid in getattr(g, "member_ids", []):
                if mid in allowed and mid not in m:
                    order.append(mid)
                    m[mid] = max(0.0, sim)
                    prov[mid] = g.cid
        return order, m, prov

    def _run_coactivation(self, dense: list, records: dict, at, scope: Scope,
                          allowed: set) -> tuple[list[str], dict]:
        """Co-activation channel: memories co-confirmed with the top dense hits in PAST recalls
        (graph CO_ACTIVATED links, Section 7.3) are pulled in as candidates. This is multi-hop
        recall -- a memory sharing no query words but repeatedly used together with a dense hit
        surfaces here. Ranks by co-activation frequency (how many seeds link to it), never by age."""
        seeds = [mid for mid, _ in dense[:10]]
        if not seeds:
            return [], {}
        seed_set = set(seeds)
        freq: dict[str, int] = {}
        for mid in seeds:
            for linked in self.graph.linked_memories(mid, scope, at):
                if linked in allowed and linked not in seed_set:
                    freq[linked] = freq.get(linked, 0) + 1
        if not freq:
            return [], {}
        order = [m for m, _ in sorted(freq.items(), key=lambda x: -x[1])]
        return order, {m: float(c) for m, c in freq.items()}

    # ---- memory typing coordinator (Phase 4, soft prior) ------------------
    @staticmethod
    def _query_class(parsed: dict, query: str) -> str:
        """Coarse query class for the type-priority coordinator (deterministic, no model)."""
        q = (query or "").lower()
        if parsed.get("ranges") or parsed.get("operation") in ("order", "count"):
            return "temporal"
        if any(k in q for k in ("how to", "how do i", "steps", "procedure", "instructions",
                                "install", "configure", "set up", "deploy", "recipe")):
            return "procedural"
        if any(k in q for k in ("prefer", "favorite", "favourite", "i like", "i love",
                                "allerg", "usually", "always", "my ")):
            return "preference"
        return "factual"

    def _apply_type_prior(self, fused: dict, records: dict, parsed: dict, query: str) -> None:
        """Mutate `fused` in place with a bounded type-match bonus. The bonus is at most
        type_prior_weight * max_fused, so it breaks ties toward the query class's preferred MIRIX
        types without swamping a strong content match. A no-op if no candidate carries a type."""
        from .memory_types import type_priority
        order = type_priority(self._query_class(parsed, query))
        rank = {t.value: (len(order) - i) for i, t in enumerate(order)}
        mx_rank = max(rank.values()) if rank else 1
        mx_fused = max(fused.values()) if fused else 0.0
        if mx_fused <= 0.0:
            return
        w = self.settings.type_prior_weight
        for mid in fused:
            rec = records.get(mid)
            t = (getattr(rec, "metadata", None) or {}).get("type") if rec else None
            if t and t in rank:
                fused[mid] += w * (rank[t] / mx_rank) * mx_fused

    def _apply_salience_boost(self, fused: dict, records: dict) -> None:
        """Phase 3 affect coupling: retrieval_score = fused + lambda_salience * s, bounded to a
        fraction of the top fused score so a salient memory ranks higher WITHOUT overriding a strong
        content match. `s` (record.salience) carries NO age/timestamp term, so two memories with
        equal salience get an identical boost regardless of their valid_at -> age-invariant."""
        lam = self.settings.lambda_salience
        mx_fused = max(fused.values()) if fused else 0.0
        if mx_fused <= 0.0:
            return
        for mid in fused:
            rec = records.get(mid)
            if rec is not None:
                fused[mid] += lam * float(getattr(rec, "salience", 0.0)) * mx_fused

    # ---- fusion + final selection ----------------------------------------
    def _fuse(self, rankings: list[list[str]], score_maps: list[dict],
              weights: list[float]) -> dict[str, float]:
        """Dispatch the configured fusion method. RRF (rank-based, scale-free) is the
        default and the unknown-method fallback; Borda is rank-based; z-score/min-max/DBSF
        use the per-channel raw scores."""
        method = self.settings.fusion_method
        if method == "borda":
            return _fusion.combine_borda(rankings, weights)
        if method in _fusion.SCORE_METHODS:
            return _fusion.combine_scores(score_maps, weights, method)
        return _rrf(rankings, self.settings.rrf_k, weights)   # rrf / unknown -> robust default

    def _finalize(self, query: str, ranked: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """Cross-encoder rerank (skippable on a large margin) -> MMR diversity -> adaptive-k /
        conformal depth -> top-k. With every Layer-2 flag off this is identical to the prior
        behaviour (rerank to final_topk, return final_topk)."""
        s = self.settings
        # Only a downstream depth consumer (MMR / adaptive-k / conformal) needs more than
        # final_topk reranked items. When none is active we request exactly final_topk, so the
        # reported (flags-off) path makes the identical client.rerank call it always did.
        need_depth = (s.mmr_enabled or s.adaptive_k_enabled
                      or (s.conformal_depth_enabled and s.conformal_qhat >= 0.0))
        rerank_topn = max(s.rerank_depth, s.final_topk) if need_depth else s.final_topk
        skip_rerank = _gating.should_skip_rerank([c.fused_score for c in ranked], s.rerank_skip_margin)
        if s.rerank_enabled and not skip_rerank and ranked:
            shortlist = ranked[: max(s.rerank_depth, s.final_topk)]
            docs = [c.record.text or c.record.summary or "" for c in shortlist]
            try:
                reranked = []
                for orig_idx, score in self.client.rerank(query, docs, rerank_topn):
                    shortlist[orig_idx].rerank_score = score
                    reranked.append(shortlist[orig_idx])
                ranked = reranked or shortlist
            except Exception:
                if not s.rerank_fail_open:
                    raise
                ranked = shortlist
        # Depth-select BEFORE MMR. adaptive_k_cut's largest-gap cut is a POSITIONAL slice that
        # assumes a score-descending list; MMR reorders by diversity, so cutting after MMR would
        # keep the first-k MMR positions, silently dropping high-score items past position k.
        # Cut on the relevance-descending order first, then diversify only the survivors.
        ranked = self._depth_select(ranked)
        ranked = self._mmr_pass(ranked)
        return ranked[: s.final_topk]

    def _mmr_pass(self, ranked: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """2c MMR diversity re-ordering over the candidate content vectors. No-op when off
        or when any vector is unavailable (fail safe, never drop a candidate silently)."""
        s = self.settings
        if not s.mmr_enabled or len(ranked) <= 2:
            return ranked
        ids = [c.record.memory_id for c in ranked]
        vmap = self.index.get_vectors(ids)
        vecs = [vmap.get(mid) for mid in ids]
        if any(v is None for v in vecs):
            return ranked
        rels = [c.rerank_score or c.fused_score for c in ranked]
        order = _mmr.mmr_order(rels, vecs, lam=s.mmr_lambda)
        return [ranked[i] for i in order]

    def _depth_select(self, ranked: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """2b/2a calibrated depth: split-conformal cutoff (if a dev-calibrated qhat is set),
        then the largest-gap adaptive-k cut. Both preserve order and keep >= adaptive_k_min."""
        s = self.settings
        if not ranked:
            return ranked
        if s.conformal_depth_enabled and s.conformal_qhat >= 0.0:
            ranked = _conformal.select_by_conformal(
                ranked, lambda c: c.dense_score, s.conformal_qhat,
                min_keep=min(s.adaptive_k_min, len(ranked)))
        if s.adaptive_k_enabled:
            ranked = _adaptive_k.adaptive_k_cut(
                ranked, score_fn=lambda c: (c.rerank_score or c.fused_score),
                min_k=min(s.adaptive_k_min, len(ranked)), max_k=s.final_topk)
        return ranked

    def assemble_context(self, query: str, candidates: list[RetrievalCandidate],
                         at: Optional[float] = None, scope: Optional[Scope] = None,
                         include_conflict_resolution: bool = True) -> list[str]:
        """Build the token-budgeted context blocks: structured event-calendar selection +
        surfaced typed preferences (uncompressed), then Hopfield-ordered raw chunks
        (optionally extractively compressed), with lost-in-the-middle edge placement.

        This is CONTEXT ASSEMBLY (retrieval), shared by `answer()` AND the neutral benchmark
        adapter, so the event calendar + preferences reach the scoreboard while the SHARED
        reader still produces the answer string. No answer is computed here."""
        scope = scope or Scope()
        parsed = parse_query(query, at)
        events = self.store.events_in_scope(scope.namespace)
        event_blocks = [e.as_text() for e in select_for_query(events, parsed, at)[:8]]
        # Phase 5: a chronological event chain for order/sequence/temporal queries (gated). Selection
        # + ordering only; the shared reader still computes the answer.
        chain_blocks: list[str] = []
        if (self.settings.event_chain_context_enabled
                and (parsed.get("operation") in ("order", "count") or parsed.get("ranges"))):
            chain = event_chain(events, parsed, at)
            if chain:
                chain_blocks = ["Event timeline (chronological): "
                                + " -> ".join(e.as_text() for e in chain)]
        pref_blocks = [f"User preference: {p}" for p in self.store.get_profile(scope.namespace)[:3]]
        # Phase 6: a working scratchpad of high-salience verified ACTIVE facts as a context channel
        # (gated; each entry links to a raw source hash, superseded facts expire via the active
        # filter). Off -> context is unchanged.
        scratchpad_blocks: list[str] = []
        if self.settings.scratchpad_enabled:
            from .scratchpad import select_scratchpad
            active = self.store.active_records_at(at if at is not None else now(), scope)
            entries = select_scratchpad(active, top_k=self.settings.scratchpad_topk,
                                        min_salience=self.settings.scratchpad_min_salience)
            if entries:
                scratchpad_blocks = ["Scratchpad (high-salience verified facts): "
                                     + " | ".join(e["text"] for e in entries)]
        resolver_blocks = (
            self._conflict_resolution_blocks(query, candidates, at)
            if include_conflict_resolution else []
        )

        # Raw chunks ordered by Hopfield attention weight, each PREFIXED with its session
        # date. This gives the reader the temporal anchor to resolve relative expressions
        # ("yesterday", "last week") that the LLM event extractor may miss -- the structured
        # session date is the temporal ground truth.
        import datetime as _dt

        raw_blocks = []
        ordered_candidates = self._hopfield_order(candidates)
        if self.settings.temporal_rerank_enabled:
            ordered_candidates = _temporal_context_order(query, parsed, ordered_candidates)
        for c in ordered_candidates:
            txt = c.record.text or c.record.summary or ""
            if self.settings.context_compress_enabled and self.settings.compression_ratio < 1.0:
                txt = compress_chunk(txt, query, self.settings.compression_ratio)
            if c.record.valid_at:
                txt = f"[Session date {_dt.date.fromtimestamp(c.record.valid_at).isoformat()}] {txt}"
            raw_blocks.append(txt)

        # Budget on the PRIORITY order first, then edge-place the survivors. Edge-placing first
        # then budgeting truncated from the tail, which is where edge_place puts the 2nd-highest
        # priority block, so a high-priority block was dropped before lower-priority raw chunks.
        budgeted = _budget_blocks(
            resolver_blocks + scratchpad_blocks + event_blocks + chain_blocks + pref_blocks + raw_blocks,
            self.settings.context_token_budget)
        return edge_place(budgeted)

    def _hopfield_order(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """Modern-Hopfield / attention readout (dossier 8.1-8.2): a single-step softmax over
        the retrieved set's scores yields attention weights; order candidates by that weight
        (pattern completion). Weights derive from content/rerank scores only -- no FSRS
        priority or age term participates."""
        scores = [c.rerank_score or c.fused_score for c in candidates]
        weights = _softmax(scores)
        order = sorted(range(len(candidates)), key=lambda i: -weights[i])
        return [candidates[i] for i in order]

    def _hopfield_readout(self, candidates: list[RetrievalCandidate]) -> list[str]:
        """Text-only variant of _hopfield_order (kept for compatibility)."""
        return [c.record.text or c.record.summary or "" for c in self._hopfield_order(candidates)]

    def _try_conflict_resolver(
        self, query: str, candidates: list[RetrievalCandidate], as_of: Optional[float] = None
    ) -> Optional[CurrentValueResolution]:
        if not self.settings.conflict_resolver_enabled:
            return None
        return resolve_current_value_question(
            query, candidates, self.client.extract_current_value_matches, as_of
        )

    def _answer_from_conflict_resolution(
        self, query: str, resolution: CurrentValueResolution, *, verify: bool
    ) -> Answer:
        # Deterministic abstention: candidates existed but none was valid as of the requested time.
        if resolution.abstained:
            return Answer(
                question=query,
                answer="I don't have a value valid as of the requested time.",
                verified=False, confidence=0.0, citations=[], unverified_claims=[],
                generated_by="conflict-resolver", retrieved_count=len(resolution.matches),
                note=resolution.note,
            )
        citations: list[Citation] = []
        entailed = 0
        for rec in resolution.records:
            label, conf = (NLILabel.NEUTRAL, 0.0)
            if verify:
                label, conf = self.verify_citation(rec, resolution.answer)
            citations.append(Citation(
                memory_id=rec.memory_id, content_hash=rec.content_hash,
                raw_uri=rec.raw_uri, source=rec.source, valid_at=rec.valid_at,
                snippet=(rec.text or rec.summary or "")[:240],
                nli_label=label, nli_score=conf,
            ))
            if label == NLILabel.ENTAILMENT:
                entailed += 1
        verified = (entailed > 0) if verify else False
        note = resolution.note
        if resolution.superseded:        # show the supersession chain (older values, not deleted)
            note = f"{note}; superseded {len(resolution.superseded)} older value(s)"
        if verify and not verified:
            note = f"{note}: unverified"
        return Answer(
            question=query, answer=resolution.answer, verified=verified,
            confidence=1.0 if verified else 0.0, citations=citations,
            unverified_claims=[] if verified or not verify else [resolution.answer],
            generated_by="conflict-resolver", retrieved_count=len(resolution.records),
            note=note,
        )

    def _conflict_resolution_blocks(self, query: str, candidates: list[RetrievalCandidate],
                                    as_of: Optional[float] = None) -> list[str]:
        resolved = self._try_conflict_resolver(query, candidates, as_of)
        if resolved is None or resolved.abstained:
            return []
        blocks = []
        for rec in resolved.records:
            timestamp = f"{rec.valid_at:.0f}" if rec.valid_at is not None else "unknown"
            evidence = (rec.text or rec.summary or "").strip()
            blocks.append(
                "Current-value resolver selected latest matching evidence.\n"
                f"Answer candidate: {resolved.answer}\n"
                f"Source timestamp: {timestamp}\n"
                f"Evidence: {evidence}"
            )
        return blocks

    # ---- verification -----------------------------------------------------
    def verify(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        label, conf = self.client.nli(premise, hypothesis)
        return NLILabel(label), conf

    def verify_citation(self, rec: MemoryRecord, hypothesis: str) -> tuple[NLILabel, float]:
        """Verify the answer against a source. For IMAGE memories the arbiter is the actual
        PIXELS (verified visual recall): a visual claim is judged against the raw image, so
        unsupported visual claims are rejected exactly like the text NLI path."""
        if rec.modality == Modality.IMAGE:
            try:
                raw = self.substrate.get(rec.content_hash)
                tmp_dir = self.settings.data_dir / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".png", delete=False) as f:
                    f.write(raw)
                    path = f.name
                try:
                    label, conf = self.client.verify_visual(path, hypothesis)
                finally:
                    try:
                        Path(path).unlink()
                    except OSError:
                        pass
                return NLILabel(label), conf
            except Exception:
                pass  # fall back to text verification below
        return self.verify(self._ground_truth(rec), hypothesis)

    def _verify_candidates(self, candidates: list[RetrievalCandidate], text: str,
                           verify: bool) -> tuple[list[Citation], int]:
        """Verify candidates against the answer and build citations. Strategy (S1, flag-gated):
        batched NLI (one request), short-circuit (stop after the citation cap), or the baseline
        per-candidate serial path. With both flags off this is byte-identical to the old loop."""
        s = self.settings
        labels: list[tuple] = [(NLILabel.NEUTRAL, 0.0)] * len(candidates)
        if verify:
            if s.batch_nli_enabled:
                # Text sources judged together in ONE call; image sources judged against pixels.
                text_idx = [i for i, c in enumerate(candidates)
                            if c.record.modality != Modality.IMAGE]
                pairs = [(self._ground_truth(candidates[i].record), text) for i in text_idx]
                batch = self.client.nli_batch(pairs) if pairs else []
                for j, i in enumerate(text_idx):
                    if j < len(batch):
                        lab, conf = batch[j]
                        labels[i] = (NLILabel(lab), conf)
                for i, c in enumerate(candidates):
                    if c.record.modality == Modality.IMAGE:
                        labels[i] = self.verify_citation(c.record, text)
            elif s.fast_verify_enabled:
                found = 0
                for i, c in enumerate(candidates):
                    if found >= s.verify_citation_cap:
                        break                       # short-circuit: the rest stay neutral
                    labels[i] = self.verify_citation(c.record, text)
                    if labels[i][0] == NLILabel.ENTAILMENT:
                        found += 1
            else:
                for i, c in enumerate(candidates):
                    labels[i] = self.verify_citation(c.record, text)
        citations: list[Citation] = []
        entailed = 0
        for i, c in enumerate(candidates):
            rec = c.record
            lab, conf = labels[i]
            citations.append(Citation(
                memory_id=rec.memory_id, content_hash=rec.content_hash,
                raw_uri=rec.raw_uri, source=rec.source, valid_at=rec.valid_at,
                snippet=(rec.text or rec.summary or "")[:240],
                nli_label=lab, nli_score=conf,
            ))
            if lab == NLILabel.ENTAILMENT:
                entailed += 1
        return citations, entailed

    def _abstention_confidence(self, candidates: list[RetrievalCandidate],
                               citations: list[Citation]) -> tuple[float, dict]:
        """Blend the four abstention signals into a confidence score (Phase 2). Two are structural
        (channel agreement, proof completeness) so the gate does not rest on the model's
        self-report. Returns (confidence, per-signal dict)."""
        from . import abstention as _ab
        s = self.settings
        entail = max((c.nli_score for c in citations if c.nli_label == NLILabel.ENTAILMENT),
                     default=0.0)
        coverage = max((c.dense_score for c in candidates), default=0.0)
        agreement = (_ab.channel_agreement(max(candidates, key=lambda c: c.fused_score))
                     if candidates else 0.0)
        proof = _ab.proof_completeness(citations)
        conf = _ab.combine_confidence(
            entail, coverage, agreement, proof,
            w_entail=s.abstention_w_entail, w_coverage=s.abstention_w_coverage,
            w_agreement=s.abstention_w_agreement, w_proof=s.abstention_w_proof)
        return conf, {"entail": float(entail), "coverage": min(1.0, max(0.0, coverage)),
                      "agreement": agreement, "proof": proof}

    # ---- end-to-end answer -----------------------------------------------
    def answer(self, query: str, at: Optional[float] = None, *, verify: bool = True,
               scope: Optional[Scope] = None, qvec: Optional[np.ndarray] = None,
               precomputed: Optional[list[RetrievalCandidate]] = None,
               reader_model: Optional[str] = None) -> Answer:
        at = now() if at is None else at
        scope = scope or Scope()
        # `precomputed` lets a caller time retrieval separately and avoid re-retrieving.
        candidates = precomputed if precomputed is not None else self.retrieve(query, at, scope, qvec=qvec)
        if not candidates:
            return Answer(
                question=query, answer="I do not have that in memory.",
                verified=True, confidence=1.0, generated_by=self.settings.gen_model,
                retrieved_count=0, note="empty-or-no-active-memory",
            )

        resolved = self._try_conflict_resolver(query, candidates, at)
        if resolved is not None:
            return self._answer_from_conflict_resolution(query, resolved, verify=verify)

        # Pre-generation coverage signal (strength of the best content match).
        coverage = max((c.dense_score for c in candidates), default=0.0)

        # Shared context assembly (event calendar + preferences + raw chunks, edge-placed).
        blocks = self.assemble_context(query, candidates, at, scope,
                                       include_conflict_resolution=False)
        # reader_model pins one fixed answerer (neutral harness); else the difficulty cascade.
        model = reader_model or _reader_model(query, self.settings)
        text = self.client.generate_answer(query, blocks, model=model)  # real call

        citations, entailed = self._verify_candidates(candidates, text, verify)

        verified = (entailed > 0) if verify else False
        unverified: list[str] = []
        abstained = False
        note = ""
        # Calibrated abstention (Phase 2). When ABSTENTION_V2 is on, gate on a multi-signal
        # confidence (entailment + coverage + structural channel-agreement + proof-completeness)
        # against the dev-calibrated tau. When off, the original coverage gate runs unchanged.
        if verify and self.settings.abstention_v2_enabled:
            conf, sig = self._abstention_confidence(candidates, citations)
            if conf < self.settings.abstention_v2_tau:
                abstained = True
                text = "I don't have enough verified evidence in memory to answer that confidently."
                note = (f"abstained: confidence {conf:.2f} < tau "
                        f"{self.settings.abstention_v2_tau:.2f} (entail={sig['entail']:.2f} "
                        f"coverage={sig['coverage']:.2f} agreement={sig['agreement']:.2f} "
                        f"proof={sig['proof']:.2f})")
            elif not verified:
                note = "unverified: no source entails the answer"
                unverified = [text]
        elif verify and not verified and coverage < self.settings.abstention_threshold:
            abstained = True
            text = "I don't have enough verified evidence in memory to answer that confidently."
            note = f"abstained: insufficient evidence (coverage {coverage:.2f})"
        elif verify and not verified:
            note = "unverified: no source entails the answer"
            unverified = [text]

        top_rerank = max((c.rerank_score for c in candidates), default=0.0)
        if abstained:
            confidence = 0.0
        elif verify:
            confidence = 0.5 * min(1.0, top_rerank) + 0.5 * (1.0 if verified else 0.0)
        else:
            confidence = min(1.0, top_rerank)

        if verified:
            citations = [c for c in citations if c.nli_label == NLILabel.ENTAILMENT] or citations

        return Answer(
            question=query, answer=text, verified=verified, confidence=confidence,
            citations=citations, unverified_claims=unverified,
            generated_by=model, retrieved_count=len(candidates), note=note,
        )
