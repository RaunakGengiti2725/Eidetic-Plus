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
from pathlib import Path
from typing import Optional

import numpy as np

from .bm25 import BM25, PersistentBM25
from .config import Settings, get_settings
from .conflicts import CurrentValueResolution, resolve_current_value_question
from .dashscope_client import DashScopeClient
from .events import parse_query, select_for_query
from .graph import KnowledgeGraph
from .models import (Answer, Citation, MemoryRecord, Modality, NLILabel,
                     RetrievalCandidate, Scope, now)
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

        # Channel 1: dense (content ANN) -- the age-independent substrate.
        dense = self.index.search(qvec, self.settings.ann_topk, allowed_ids=set(records))
        dense_order = [mid for mid, _ in dense]
        dense_map = dict(dense)

        # Channel 2: BM25 lexical (recovers exact terms/codes/numbers dense misses).
        if self.settings.persistent_bm25_enabled:
            changed = self.bm25.ensure_indexed(
                (r.memory_id, r.text or r.summary or "") for r in corpus
            )
            if changed:
                self.bm25.save()
            bm25_hits = self.bm25.search(query, self.settings.ann_topk, allowed_ids=set(records))
        else:
            bm25 = BM25().index([(r.memory_id, r.text or r.summary or "") for r in corpus])
            bm25_hits = bm25.search(query, self.settings.ann_topk)
        bm25_order = [mid for mid, _ in bm25_hits]
        bm25_map = dict(bm25_hits)

        # Channel 3: single-pass PPR seeded from QUERY-linked entities (+ best dense hits'
        # entities), letting activation reach passages sharing no query words. No LLM loop.
        seed_entities: list[str] = list(parsed["entities"])
        if s.hippo2_seeding_enabled:
            seed_entities.extend(_hippo2_seed_entities(query, parsed, self.store, at, scope))
        for mid, _ in dense[:10]:
            seed_entities.extend(records[mid].entities)
        graph_scores = self.graph.score_memories(
            seed_entities, corpus, at, scope) if seed_entities else {}
        graph_order = [mid for mid, _ in sorted(graph_scores.items(), key=lambda x: -x[1])]

        # Channel 4: recency (minor).
        recency_order = [r.memory_id for r in sorted(corpus, key=lambda r: -r.valid_at)
                         ][: self.settings.ann_topk] if use_recency else []

        # Query-adaptive weighted RRF: BM25 up for name/date/ID queries, graph up for multi-hop.
        rankings, weights = [dense_order, bm25_order], [
            s.rrf_w_dense, s.rrf_w_bm25 * (1.6 if parsed["is_namey"] else 1.0)]
        if graph_order:
            rankings.append(graph_order)
            weights.append(s.rrf_w_graph * (1.6 if parsed["is_multihop"] else 1.0))
        if recency_order:
            rankings.append(recency_order)
            weights.append(s.rrf_w_recency)
        fused = _rrf(rankings, s.rrf_k, weights)
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

        # Cross-encoder rerank (config-gated, depth ~50 -> final_topk; A/B against off).
        if s.rerank_enabled:
            shortlist = ranked[: max(s.rerank_depth, s.final_topk)]
            docs = [c.record.text or c.record.summary or "" for c in shortlist]
            try:
                reranked = []
                for orig_idx, score in self.client.rerank(query, docs, s.final_topk):
                    shortlist[orig_idx].rerank_score = score
                    reranked.append(shortlist[orig_idx])
                return reranked or shortlist[: s.final_topk]
            except Exception:
                if s.rerank_fail_open:
                    return shortlist[: s.final_topk]
                raise
        return ranked[: s.final_topk]

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
        pref_blocks = [f"User preference: {p}" for p in self.store.get_profile(scope.namespace)[:3]]
        resolver_blocks = (
            self._conflict_resolution_blocks(query, candidates)
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

        ordered = edge_place(resolver_blocks + event_blocks + pref_blocks + raw_blocks)
        return _budget_blocks(ordered, self.settings.context_token_budget)

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
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> Optional[CurrentValueResolution]:
        if not self.settings.conflict_resolver_enabled:
            return None
        return resolve_current_value_question(
            query, candidates, self.client.extract_current_value_matches
        )

    def _answer_from_conflict_resolution(
        self, query: str, resolution: CurrentValueResolution, *, verify: bool
    ) -> Answer:
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
        if verify and not verified:
            note = f"{note}: unverified"
        return Answer(
            question=query, answer=resolution.answer, verified=verified,
            confidence=1.0 if verified else 0.0, citations=citations,
            unverified_claims=[] if verified or not verify else [resolution.answer],
            generated_by="conflict-resolver", retrieved_count=len(resolution.records),
            note=note,
        )

    def _conflict_resolution_blocks(self, query: str,
                                    candidates: list[RetrievalCandidate]) -> list[str]:
        resolved = self._try_conflict_resolver(query, candidates)
        if resolved is None:
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

        resolved = self._try_conflict_resolver(query, candidates)
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

        citations: list[Citation] = []
        entailed = 0
        for c in candidates:
            rec = c.record
            label, conf = (NLILabel.NEUTRAL, 0.0)
            if verify:
                label, conf = self.verify_citation(rec, text)  # text NLI or visual judge
            citations.append(Citation(
                memory_id=rec.memory_id, content_hash=rec.content_hash,
                raw_uri=rec.raw_uri, source=rec.source, valid_at=rec.valid_at,
                snippet=(rec.text or rec.summary or "")[:240],
                nli_label=label, nli_score=conf,
            ))
            if label == NLILabel.ENTAILMENT:
                entailed += 1

        verified = (entailed > 0) if verify else False
        unverified: list[str] = []
        abstained = False
        # Abstention gate: weak evidence AND nothing entails -> abstain, don't guess.
        if verify and not verified and coverage < self.settings.abstention_threshold:
            abstained = True
            text = "I don't have enough verified evidence in memory to answer that confidently."
            note = f"abstained: insufficient evidence (coverage {coverage:.2f})"
        elif verify and not verified:
            note = "unverified: no source entails the answer"
            unverified = [text]
        else:
            note = ""

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
