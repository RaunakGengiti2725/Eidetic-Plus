"""Central configuration. Reads `.env` once and exposes a frozen Settings object.

The ONLY thing that changes between APP_ENV=dev and APP_ENV=prod is the storage/DB
backend (local content-addressed store + SQLite + hnswlib in dev; Alibaba Cloud
OSS-WORM + AnalyticDB-PG + GDB in prod). Every model call is identical in both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of this file's package dir) if present.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# DashScope endpoints per region. The dashscope SDK uses the /api/v1 base; the
# OpenAI-compatible base (used for the Files API / qwen-long document reading) differs.
_REGION_ENDPOINTS = {
    "singapore": "https://dashscope-intl.aliyuncs.com/api/v1",
    "beijing": "https://dashscope.aliyuncs.com/api/v1",
}
_REGION_COMPAT = {
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_bool(name: str, default: str = "0") -> bool:
    """Parse a boolean env flag, case-insensitively. Without this, a capitalized word form like
    RERANK_ENABLED=False or FEEDBACK=Yes was silently INVERTED (the raw 'False'/'Yes' did not
    match the lowercase-only tuples). Not folded into _get() because _get also reads
    case-sensitive strings (keys, model ids, paths)."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    # Environment
    app_env: str = field(default_factory=lambda: _get("APP_ENV", "dev").lower())

    # DashScope auth + region
    api_key: str = field(default_factory=lambda: _get("DASHSCOPE_API_KEY"))
    region: str = field(default_factory=lambda: _get("DASHSCOPE_REGION", "singapore").lower())

    # Model IDs
    text_embed_model: str = field(default_factory=lambda: _get("TEXT_EMBED_MODEL", "text-embedding-v4"))
    multimodal_embed_model: str = field(
        default_factory=lambda: _get("MULTIMODAL_EMBED_MODEL", "tongyi-embedding-vision-plus")
    )
    embed_dim: int = field(default_factory=lambda: _get_int("EMBED_DIM", 1024))
    # S4 persistent embedding cache keyed by (model, dim, sha256(text)) -> repeats/re-embeds are
    # free across restarts (a full cache hit needs no key and no model call). On by default; the
    # (model, dim) key guarantees a rename/dim change misses rather than returns a stale vector.
    embed_cache_enabled: bool = field(default_factory=lambda: _get_bool("EMBED_CACHE", "1"))

    # F1 DashScope rate governor: token-bucket RPM + concurrency semaphore + 429 backoff, so
    # batching/fan-out never trips per-account limits. Conservative defaults (a low-tier key never
    # self-throttles). Governs every model call; DASHSCOPE_GOVERN=0 disables (raw calls).
    dashscope_govern_enabled: bool = field(default_factory=lambda: _get_bool("DASHSCOPE_GOVERN", "1"))
    dashscope_rpm: int = field(default_factory=lambda: _get_int("DASHSCOPE_RPM", 60))
    dashscope_max_concurrency: int = field(default_factory=lambda: _get_int("DASHSCOPE_MAX_CONCURRENCY", 4))
    dashscope_max_retries: int = field(default_factory=lambda: _get_int("DASHSCOPE_MAX_RETRIES", 5))
    dashscope_backoff_base: float = field(default_factory=lambda: float(_get("DASHSCOPE_BACKOFF_BASE", "0.5")))
    dashscope_backoff_max: float = field(default_factory=lambda: float(_get("DASHSCOPE_BACKOFF_MAX", "30.0")))

    salience_model: str = field(default_factory=lambda: _get("SALIENCE_MODEL", "qwen-flash"))
    extract_model: str = field(default_factory=lambda: _get("EXTRACT_MODEL", "qwen-plus"))
    verify_model: str = field(default_factory=lambda: _get("VERIFY_MODEL", "qwen-plus"))
    rerank_model: str = field(default_factory=lambda: _get("RERANK_MODEL", "qwen3-rerank"))
    gen_model: str = field(default_factory=lambda: _get("GEN_MODEL", "qwen3-max"))
    consolidate_model: str = field(default_factory=lambda: _get("CONSOLIDATE_MODEL", "qwen-plus"))
    describe_model: str = field(default_factory=lambda: _get("DESCRIBE_MODEL", "qwen-vl-plus"))

    # Ingestion models
    ocr_model: str = field(default_factory=lambda: _get("OCR_MODEL", "qwen-vl-ocr"))
    asr_model: str = field(default_factory=lambda: _get("ASR_MODEL", "qwen3-asr-flash"))
    # Document reading goes through the OpenAI-compatible Files API (file-extract + fileid://),
    # which requires a long-context reader. qwen-long is the model that path needs; DOC_MODEL is
    # the single knob (the old default qwen-doc-turbo was never actually used -- read_document
    # hardcoded qwen-long, so DOC_MODEL was dead config). Needs account access to the chosen model.
    doc_model: str = field(default_factory=lambda: _get("DOC_MODEL", "qwen-long"))
    video_model: str = field(default_factory=lambda: _get("VIDEO_MODEL", "qwen-vl-plus"))

    # Local storage (dev)
    data_dir: Path = field(default_factory=lambda: Path(_get("DATA_DIR", "./data")).resolve())
    vector_backend: str = field(default_factory=lambda: _get("VECTOR_BACKEND", "auto").lower())
    struct_dim: int = field(default_factory=lambda: _get_int("STRUCT_DIM", 64))

    # Retrieval tuning
    ann_topk: int = field(default_factory=lambda: _get_int("ANN_TOPK", 100))
    final_topk: int = field(default_factory=lambda: _get_int("FINAL_TOPK", 10))
    rrf_k: int = field(default_factory=lambda: _get_int("RRF_K", 60))

    # Layer 3c vector quantization: none (raw float32) | sq8 (int8, 4x) | rabitq (1-bit, 32x).
    # A quantized backend always keeps the raw float32 for an exact refine pass (recall
    # recovery). Calibrate/promote only via the dev-split recall check (>1% drop -> keep refine).
    vector_quant: str = field(default_factory=lambda: _get("VECTOR_QUANT", "none").lower())
    quant_refine: bool = field(default_factory=lambda: _get_bool("QUANT_REFINE", "1"))
    quant_refine_topn: int = field(default_factory=lambda: _get_int("QUANT_REFINE_TOPN", 100))

    # HNSW index params (M=32; efSearch raised since retrieval is not the latency bottleneck).
    hnsw_m: int = field(default_factory=lambda: _get_int("HNSW_M", 32))
    hnsw_ef_search: int = field(default_factory=lambda: _get_int("HNSW_EF_SEARCH", 256))
    hnsw_ef_construction: int = field(default_factory=lambda: _get_int("HNSW_EF_CONSTRUCTION", 200))

    # Triple-win read path tuning.
    context_token_budget: int = field(default_factory=lambda: _get_int("CONTEXT_TOKEN_BUDGET", 8000))
    cache_cosine: float = field(default_factory=lambda: float(_get("CACHE_COSINE", "0.9")))
    reader_cot_enabled: bool = field(default_factory=lambda: _get_bool("READER_COT", "0"))
    reader_router_enabled: bool = field(
        default_factory=lambda: _get_bool("READER_ROUTER", "1")
    )
    conflict_resolver_enabled: bool = field(
        default_factory=lambda: _get_bool("CONFLICT_RESOLVER", "0")
    )
    context_compress_enabled: bool = field(
        default_factory=lambda: _get_bool("CONTEXT_COMPRESS", "0")
    )
    extract_light_enabled: bool = field(
        default_factory=lambda: _get_bool("EXTRACT_LIGHT", "0")
    )
    temporal_rerank_enabled: bool = field(
        default_factory=lambda: _get_bool("TEMPORAL_RERANK", "0")
    )
    hippo2_seeding_enabled: bool = field(
        default_factory=lambda: _get_bool("HIPPO2_SEEDING", "0")
    )
    persistent_bm25_enabled: bool = field(
        default_factory=lambda: _get_bool("PERSISTENT_BM25", "1")
    )
    semantic_cache_enabled: bool = field(
        default_factory=lambda: _get_bool("SEMANTIC_CACHE", "1")
    )
    semantic_cache_adaptive: bool = field(
        default_factory=lambda: _get_bool("SEMANTIC_CACHE_ADAPTIVE", "1")
    )

    # --- Sweepable optimization params (safe defaults; tune on a subset after the key) ---
    # Abstention gate: abstain when combined entailment+coverage is below this (calibrate it).
    abstention_threshold: float = field(default_factory=lambda: float(_get("ABSTENTION_THRESHOLD", "0.4")))
    # Phase 2 calibrated abstention (default OFF -> the coverage gate above is unchanged). When on,
    # confidence = w_entail*top_entailment + w_coverage*coverage + w_agreement*channel_agreement +
    # w_proof*proof_completeness; abstain if confidence < tau. tau is dev-calibrated via
    # abstention.pick_tau (precision target), never a magic literal. Agreement + proof are
    # STRUCTURAL signals (independent of the model's self-report).
    abstention_v2_enabled: bool = field(default_factory=lambda: _get_bool("ABSTENTION_V2", "0"))
    abstention_v2_tau: float = field(default_factory=lambda: float(_get("ABSTENTION_V2_TAU", "0.5")))
    abstention_w_entail: float = field(default_factory=lambda: float(_get("ABSTENTION_W_ENTAIL", "0.4")))
    abstention_w_coverage: float = field(default_factory=lambda: float(_get("ABSTENTION_W_COVERAGE", "0.2")))
    abstention_w_agreement: float = field(default_factory=lambda: float(_get("ABSTENTION_W_AGREEMENT", "0.2")))
    abstention_w_proof: float = field(default_factory=lambda: float(_get("ABSTENTION_W_PROOF", "0.2")))
    # Cross-encoder rerank (qwen3-rerank): on/off + candidate depth (~50 -> final_topk).
    rerank_enabled: bool = field(default_factory=lambda: _get_bool("RERANK_ENABLED", "1"))
    rerank_fail_open: bool = field(default_factory=lambda: _get_bool("RERANK_FAIL_OPEN", "0"))
    rerank_depth: int = field(default_factory=lambda: _get_int("RERANK_DEPTH", 50))
    # Reciprocal Rank Fusion: k fixed at 60; per-channel base weights (query-adaptive in code).
    rrf_w_dense: float = field(default_factory=lambda: float(_get("RRF_W_DENSE", "1.0")))
    rrf_w_bm25: float = field(default_factory=lambda: float(_get("RRF_W_BM25", "0.6")))
    rrf_w_graph: float = field(default_factory=lambda: float(_get("RRF_W_GRAPH", "0.8")))
    rrf_w_recency: float = field(default_factory=lambda: float(_get("RRF_W_RECENCY", "0.3")))
    # Difficulty cascade escalation confidence (0 = route by keywords only; product feature).
    cascade_confidence: float = field(default_factory=lambda: float(_get("CASCADE_CONFIDENCE", "0.0")))
    # LLMLingua-2-style EXTRACTIVE compression ratio for raw chunks only (1.0 = off; never facts).
    compression_ratio: float = field(default_factory=lambda: float(_get("COMPRESSION_RATIO", "1.0")))
    # Index pruning by STATIC salience (surprise+importance, age-independent); 0.0 = off. Never WORM.
    salience_prune_threshold: float = field(default_factory=lambda: float(_get("SALIENCE_PRUNE_THRESHOLD", "0.0")))

    # --- Layer 2: per-query hot-path optimizers (all default OFF / current behavior) -----
    # 2a Adaptive-k: cut the final candidate list at the largest score gap (token savings).
    adaptive_k_enabled: bool = field(default_factory=lambda: _get_bool("ADAPTIVE_K", "0"))
    adaptive_k_min: int = field(default_factory=lambda: _get_int("ADAPTIVE_K_MIN", 3))
    # 2a Adaptive efSearch: widen the HNSW beam for hard (multi-hop/long) queries only.
    adaptive_ef_enabled: bool = field(default_factory=lambda: _get_bool("ADAPTIVE_EF", "0"))
    hnsw_ef_search_hard: int = field(default_factory=lambda: _get_int("HNSW_EF_SEARCH_HARD", 512))
    # 2b Split-conformal retrieval depth: keep candidates with sim >= 1 - qhat. qhat<0 = off
    # (calibrate it on the DEV split via bench.calibrate; never on test items).
    conformal_depth_enabled: bool = field(default_factory=lambda: _get_bool("CONFORMAL_DEPTH", "0"))
    conformal_alpha: float = field(default_factory=lambda: float(_get("CONFORMAL_ALPHA", "0.1")))
    conformal_qhat: float = field(default_factory=lambda: float(_get("CONFORMAL_QHAT", "-1.0")))
    # 2c Skip the cross-encoder rerank when the first-stage margin is already large. 0 = never.
    rerank_skip_margin: float = field(default_factory=lambda: float(_get("RERANK_SKIP_MARGIN", "0.0")))
    # 2c MMR diversity post-pass (lambda in [0.3,0.7]; higher = more diverse).
    mmr_enabled: bool = field(default_factory=lambda: _get_bool("MMR_ENABLED", "0"))
    mmr_lambda: float = field(default_factory=lambda: float(_get("MMR_LAMBDA", "0.5")))
    # 2d Fusion method: rrf (default, scale-free) | zscore | minmax | dbsf | borda.
    fusion_method: str = field(default_factory=lambda: _get("FUSION_METHOD", "rrf").lower())
    # 2e Parallel channel fan-out (dense/BM25/recency concurrent; latency ~= slowest channel).
    parallel_channels_enabled: bool = field(default_factory=lambda: _get_bool("PARALLEL_CHANNELS", "0"))

    # --- Layer 3a/3b: online learning (all default OFF) ----------------------------------
    # 3b Rocchio pseudo-relevance feedback: expand the query toward the top-R evidence
    # centroid, confidence-gated to avoid topic drift. Positive-only (gamma=0) by default.
    rocchio_enabled: bool = field(default_factory=lambda: _get_bool("ROCCHIO", "0"))
    rocchio_alpha: float = field(default_factory=lambda: float(_get("ROCCHIO_ALPHA", "1.0")))
    rocchio_beta: float = field(default_factory=lambda: float(_get("ROCCHIO_BETA", "0.6")))
    rocchio_topr: int = field(default_factory=lambda: _get_int("ROCCHIO_TOPR", 5))
    rocchio_conf_gate: float = field(default_factory=lambda: float(_get("ROCCHIO_CONF_GATE", "0.35")))
    # 3a Online fusion-weight learner: when on, the content-channel (dense/bm25/graph) base
    # weights come from a dev-feedback-learned vector (index_dir/fusion_weights.json) instead
    # of the static config floats. Recency weight is NEVER learned (age-independence).
    fusion_learner_enabled: bool = field(default_factory=lambda: _get_bool("FUSION_LEARNER", "0"))
    fusion_learner_method: str = field(default_factory=lambda: _get("FUSION_LEARNER_METHOD", "eg").lower())
    # Producer side: emit a (features, reward) feedback tuple from the PRODUCT ask() path into
    # the dev-only FeedbackBuffer. Off by default; the neutral benchmark adapter never calls
    # ask(), and benchmark namespaces are recorded audit-only, so this can't touch the wall.
    feedback_enabled: bool = field(default_factory=lambda: _get_bool("FEEDBACK", "0"))

    # --- Revolutionary-architectures mechanisms (all default OFF) ------------------------
    # EvolveMem auto-revert guard: a tuned config is promoted only if it beats the champion on
    # the DEV split by >= min_delta_pp AND a paired McNemar test is significant. Dev-proxy only.
    guard_enabled: bool = field(default_factory=lambda: _get_bool("GUARD_ENABLED", "0"))
    guard_min_delta_pp: float = field(default_factory=lambda: float(_get("GUARD_MIN_DELTA_PP", "1.0")))
    guard_alpha: float = field(default_factory=lambda: float(_get("GUARD_ALPHA", "0.05")))
    # Heuristic write-time memory manager (Memory-R1 approximation): ADD/UPDATE/DELETE-tombstone
    # /NOOP. Off by default; ingest stays byte-identical when off. Never deletes a raw record.
    memory_manager_enabled: bool = field(default_factory=lambda: _get_bool("MEMORY_MANAGER", "0"))
    memory_manager_dup_cosine: float = field(default_factory=lambda: float(_get("MEMORY_MANAGER_DUP_COSINE", "0.97")))
    # MemMA evidence-grounded self-repair sweep inside the dreaming engine (LLM-gated, offline).
    dream_repair_enabled: bool = field(default_factory=lambda: _get_bool("DREAM_REPAIR", "0"))
    dream_repair_topk: int = field(default_factory=lambda: _get_int("DREAM_REPAIR_TOPK", 16))
    # Guarded repair APPLY (Phase 5): when on, apply_proposals may execute INSERT/MERGE repairs
    # via immutable ingest + bi-temporal supersession (NEVER raw deletion). Off -> apply is always
    # a dry-run that returns the plan without touching the store. Promote only behind the guard.
    dream_repair_apply_enabled: bool = field(default_factory=lambda: _get_bool("DREAM_REPAIR_APPLY", "0"))
    # Per-triple anomaly scoring threshold (flag low-confidence observed edges for repair).
    anomaly_threshold: float = field(default_factory=lambda: float(_get("ANOMALY_THRESHOLD", "0.35")))
    # MIRIX-style role typing of memories (deterministic classifier; LLM typing optional/gated).
    memory_typing_enabled: bool = field(default_factory=lambda: _get_bool("MEMORY_TYPING", "0"))
    # MIRIX Active Retrieval: generate an anticipated topic before retrieval (LLM-gated).
    active_retrieval_enabled: bool = field(default_factory=lambda: _get_bool("ACTIVE_RETRIEVAL", "0"))
    # Markov prospective prefetch: learn P(next-cluster|current) and pre-stage the predicted next.
    markov_prefetch_enabled: bool = field(default_factory=lambda: _get_bool("MARKOV_PREFETCH", "0"))
    # CoVe factored verification + bounded conflict-only debate (LLM-gated).
    cove_enabled: bool = field(default_factory=lambda: _get_bool("COVE", "0"))
    cove_questions: int = field(default_factory=lambda: int(_get("COVE_QUESTIONS", "3")))
    debate_enabled: bool = field(default_factory=lambda: _get_bool("DEBATE", "0"))
    # Span-level NLI: verify EACH sentence/claim of a multi-sentence answer against the cited
    # sources, not just the whole answer as one hypothesis. A single unentailed claim demotes the
    # answer to unverified -> abstain/strip, so a partly-grounded answer can't ride one good
    # sentence. Multiplies NLI on multi-sentence answers, hence gated (default OFF). Fires in
    # answer() -> the engine.ask product path only.
    span_nli_enabled: bool = field(default_factory=lambda: _get_bool("SPAN_NLI", "0"))
    span_nli_min_chars: int = field(default_factory=lambda: int(_get("SPAN_NLI_MIN_CHARS", "12")))

    # Chunked fact extraction (capture fidelity). When OFF (default) extract_edges sends a single
    # text[:6000] call -- byte-identical to the historical write path. When ON, a session longer
    # than the single-call cap is split into overlapping windows and the triples merged+deduped, so
    # facts beyond char ~6000 still enter the graph/events (long LoCoMo / LongMemEval sessions).
    # This multiplies the extraction LLM call 2-3x on long sessions, hence gated.
    extract_chunking_enabled: bool = field(default_factory=lambda: _get_bool("EXTRACT_CHUNKING", "0"))
    extract_chunk_chars: int = field(default_factory=lambda: int(_get("EXTRACT_CHUNK_CHARS", "4000")))
    extract_chunk_overlap: int = field(default_factory=lambda: int(_get("EXTRACT_CHUNK_OVERLAP", "400")))
    # Sentence-level preference scan. OFF (default): one profile line per session (first match) --
    # byte-identical to the historical write. ON: every preference-bearing sentence in the session
    # becomes a profile line (mid-conversation preferences are no longer lost). Token-free.
    pref_sentence_scan_enabled: bool = field(default_factory=lambda: _get_bool("PREF_SENTENCE_SCAN", "0"))

    # --- Phase-1 multi-view retrieval channels (Memory Agent Upgrade; all default OFF) ---
    # Each wires a dormant signal into the fused candidate ranking. Age-independence is preserved:
    # the structure code encodes only CYCLIC temporal coordinates (never absolute age, see
    # structure_code.py) and the query structure code has no temporal dims; the event channel ranks
    # by overlap with the QUERY's time constraint, not the memory's age. Re-prove with
    # engine.prove_age_independence after enabling.
    struct_channel_enabled: bool = field(default_factory=lambda: _get_bool("STRUCT_CHANNEL", "0"))
    rrf_w_struct: float = field(default_factory=lambda: float(_get("RRF_W_STRUCT", "0.5")))
    event_ranking_enabled: bool = field(default_factory=lambda: _get_bool("EVENT_RANKING", "0"))
    rrf_w_event: float = field(default_factory=lambda: float(_get("RRF_W_EVENT", "0.7")))
    gist_channel_enabled: bool = field(default_factory=lambda: _get_bool("GIST_CHANNEL", "0"))
    rrf_w_gist: float = field(default_factory=lambda: float(_get("RRF_W_GIST", "0.4")))
    graph_vocab_seeding: bool = field(default_factory=lambda: _get_bool("GRAPH_VOCAB_SEEDING", "0"))
    # Co-activation channel (Phase 2): pull in memories co-confirmed with the top dense hits in
    # past recalls (graph CO_ACTIVATED links) as candidates. Multi-hop recall; ranks by
    # co-activation frequency, never by age. Default OFF.
    coactivation_channel_enabled: bool = field(default_factory=lambda: _get_bool("COACTIVATION_CHANNEL", "0"))
    rrf_w_coact: float = field(default_factory=lambda: float(_get("RRF_W_COACT", "0.5")))
    # Memory typing coordinator (Phase 4): classify type on ingest + soft retrieval prior that
    # boosts the query class's preferred MIRIX types. MEMORY_TYPING gates BOTH (default OFF).
    type_prior_weight: float = field(default_factory=lambda: float(_get("TYPE_PRIOR_WEIGHT", "0.15")))

    # --- Affect-modulated salience (best-memory plan Phase 3; all default OFF) ----------------
    # Static salience s = sigmoid(wA*arousal + wL*importance + wP*surprise + wU*emphasis +
    # wR*verified_helpful), centered so neutral inputs -> ~0.5. NO age term (age-flatness audit).
    # Salience is used ONLY for a small bounded retrieval boost, FSRS initial stability, and replay
    # priority -- never to delete, prune, or encode absolute age.
    affect_salience_enabled: bool = field(default_factory=lambda: _get_bool("AFFECT_SALIENCE", "0"))
    affect_w_arousal: float = field(default_factory=lambda: float(_get("AFFECT_W_AROUSAL", "1.0")))
    affect_w_importance: float = field(default_factory=lambda: float(_get("AFFECT_W_IMPORTANCE", "1.0")))
    affect_w_surprise: float = field(default_factory=lambda: float(_get("AFFECT_W_SURPRISE", "1.0")))
    affect_w_emphasis: float = field(default_factory=lambda: float(_get("AFFECT_W_EMPHASIS", "1.0")))
    affect_w_helpful: float = field(default_factory=lambda: float(_get("AFFECT_W_HELPFUL", "0.0")))
    # Coupling strengths (dev-tuned): retrieval boost lambda, FSRS S0 gamma, replay priority alpha.
    lambda_salience: float = field(default_factory=lambda: float(_get("LAMBDA_SALIENCE", "0.1")))
    salience_gamma: float = field(default_factory=lambda: float(_get("SALIENCE_GAMMA", "0.5")))
    salience_alpha: float = field(default_factory=lambda: float(_get("SALIENCE_ALPHA", "1.0")))
    # Verified-helpful reinforcement (Phase 4): saturation cap for the bounded usage signal fed
    # into salience via affect_w_helpful. Bounding is the age-leakage guard.
    verified_helpful_cap: int = field(default_factory=lambda: _get_int("VERIFIED_HELPFUL_CAP", 5))

    # Phase 5 event-chain context: for order/sequence/temporal queries, add a chronologically
    # ordered event chain to the context ('what changed after X'). Off -> context is unchanged.
    event_chain_context_enabled: bool = field(default_factory=lambda: _get_bool("EVENT_CHAIN_CONTEXT", "0"))

    # Phase 6 working scratchpad: a small derived channel of high-salience verified ACTIVE facts
    # (each linked to its raw source hash). Context channel only; off -> context unchanged.
    scratchpad_enabled: bool = field(default_factory=lambda: _get_bool("SCRATCHPAD", "0"))
    scratchpad_topk: int = field(default_factory=lambda: _get_int("SCRATCHPAD_TOPK", 5))
    scratchpad_min_salience: float = field(default_factory=lambda: float(_get("SCRATCHPAD_MIN_SALIENCE", "0.6")))

    # --- S1 read-path latency (best-of-fastest plan; all default OFF, semantics preserved) -------
    # Batched NLI: judge all candidate sources in ONE request instead of N (faster + rate-friendly).
    batch_nli_enabled: bool = field(default_factory=lambda: _get_bool("BATCH_NLI", "0"))
    # Defer the confirmed-citation re-embed off the answer path to the idle/sleep drain (FSRS +
    # verified-helpful stay synchronous). Emits REEMBED_DEFERRED.
    defer_reembed_enabled: bool = field(default_factory=lambda: _get_bool("DEFER_REEMBED", "0"))
    # Short-circuit verification: verify in rerank order, stop once this many entailments are found.
    fast_verify_enabled: bool = field(default_factory=lambda: _get_bool("FAST_VERIFY", "0"))
    verify_citation_cap: int = field(default_factory=lambda: _get_int("VERIFY_CITATION_CAP", 3))
    # S2 write-path: debounce index saves (1 = save every ingest, baseline). Higher amortizes the
    # save; rebuild_index_from_store recovers a lost index from the substrate+SQLite source of truth.
    index_save_debounce: int = field(default_factory=lambda: _get_int("INDEX_SAVE_DEBOUNCE", 1))
    # S5 intelligence upgrades (flag-gated; guard-promoted on dev before default-on).
    # Speculative cascade: answer with the cheap tier first, escalate to the strong tier only when
    # the cheap answer fails to ground AND there is real coverage (so hard questions keep accuracy).
    cascade_enabled: bool = field(default_factory=lambda: _get_bool("SPECULATIVE_CASCADE", "0"))
    # Difficulty-adaptive retrieval depth: easy single-hop queries return fewer candidates (pay less).
    difficulty_adaptive_depth_enabled: bool = field(default_factory=lambda: _get_bool("DIFFICULTY_ADAPTIVE_DEPTH", "0"))

    # --- Track 1 Reflex Recall: the anti-RAG sub-second LOCAL candidate path (all default OFF) ---
    # A reflex packet is built from a derived inverted index + live graph/store reads -- NO model
    # call, NO embedding, NO NLI. When confident (top coverage >= reflex_min_coverage) the engine
    # feeds the reflex candidates to answer() as `precomputed`, skipping ANN/rerank; otherwise it
    # falls back to full retrieval. NLI/abstention/proof still gate every FINAL answer. The index
    # is maintained ONLY when enabled, so REFLEX_RECALL=0 is byte-identical to baseline.
    reflex_recall_enabled: bool = field(default_factory=lambda: _get_bool("REFLEX_RECALL", "0"))
    reflex_budget_ms: int = field(default_factory=lambda: _get_int("REFLEX_RECALL_BUDGET_MS", 100))
    # A reflex HIT requires the top candidate's content coverage to clear this. It is also the
    # `dense_score` the candidate carries into answer(), so a hit is always >= abstention_threshold
    # (no confident reflex hit can spuriously abstain on coverage).
    reflex_min_coverage: float = field(default_factory=lambda: float(_get("REFLEX_RECALL_MIN_COVERAGE", "0.65")))
    reflex_topk: int = field(default_factory=lambda: _get_int("REFLEX_RECALL_TOPK", 20))
    reflex_max_seeds: int = field(default_factory=lambda: _get_int("REFLEX_RECALL_MAX_SEEDS", 400))
    reflex_coact_seeds: int = field(default_factory=lambda: _get_int("REFLEX_RECALL_COACT_SEEDS", 8))
    # Activation-burst score weights. The temporal axis ranks by overlap with the QUERY's time
    # constraint, never by memory recency -- age-independence is preserved (re-prove after enabling).
    reflex_w_entity: float = field(default_factory=lambda: float(_get("REFLEX_W_ENTITY", "1.0")))
    reflex_w_lexical: float = field(default_factory=lambda: float(_get("REFLEX_W_LEXICAL", "1.0")))
    reflex_w_temporal: float = field(default_factory=lambda: float(_get("REFLEX_W_TEMPORAL", "0.7")))
    reflex_w_coactivation: float = field(default_factory=lambda: float(_get("REFLEX_W_COACTIVATION", "0.5")))
    reflex_w_hotset: float = field(default_factory=lambda: float(_get("REFLEX_W_HOTSET", "0.3")))
    reflex_hotset_size: int = field(default_factory=lambda: _get_int("REFLEX_HOTSET_SIZE", 64))

    # --- Track 2 perfect sync: versioned answer-cache invalidation (correctness fix, default ON) --
    # The answer cache is tagged with a per-NAMESPACE memory version; any content write in a
    # namespace bumps it, so every prior cached answer in that namespace becomes unreachable (no
    # WRITE-INDUCED stale-truth hits). Namespace (not scope_key) because a read sees every record
    # visible_to it in the namespace, so a sub-scope write must still invalidate a namespace-wide
    # query's entry. CACHE_VERSIONING=0 restores the legacy (never-invalidated) cache, byte-identical.
    # KNOWN LIMIT: invalidation is write-triggered, not time-triggered. A fact ingested with a
    # FUTURE invalid_at/expired_at can be served from a cache entry made before its expiry if no
    # write intervenes (the contradiction path sets invalid_at~=now, so this is only reachable via
    # explicit future-dated ingestion). A cache-entry TTL would bound this -- deferred follow-up.
    cache_versioning_enabled: bool = field(default_factory=lambda: _get_bool("CACHE_VERSIONING", "1"))

    # --- Track 3 false-premise abstention (default OFF; flag-off byte-identical) -----------------
    # A presuppositional question ("Why did Alice leave Google?") presupposes a relationship between
    # its entities. When ON, if the question names >=2 entities and NO in-scope memory co-mentions a
    # pair AND no graph edge connects a pair (TOTAL disconnection), abstain with a structured
    # missing-premise reason instead of letting the reader confabulate. Bias is toward answering: a
    # false abstain is a recall regression, a miss just falls back to the normal answer+NLI path.
    # Deterministic + no model call on the abstain path. KNOWN LIMITS (recall regressions, not
    # correctness): the entity signal is parse_query's capitalized/quoted/ID forms (a fully
    # lowercased question yields no entities -> no check); coreference/pronoun memories and aliases
    # (Meta vs stored Facebook) are not resolved. Matching IS case-insensitive (a lowercase stored
    # memory still prevents a false abstain).
    false_premise_enabled: bool = field(default_factory=lambda: _get_bool("FALSE_PREMISE", "0"))

    # --- Track 9 Flow / Instinct Recall (all default OFF / zero-amplitude; flag-off byte-identical) -
    # One per-namespace ActivationField is the shared working-memory substrate every recall path
    # reads and every confirmed recall writes. Activation is ACCESS-recency only (never memory age).
    # FLOW_ACTIVATION=0 => the field is never built, _hotset is unchanged, retrieve() activation is
    # None, no new BrainEvents -- byte-identical to today on every touched surface.
    flow_activation_enabled: bool = field(default_factory=lambda: _get_bool("FLOW_ACTIVATION", "0"))
    flow_decay: float = field(default_factory=lambda: float(_get("FLOW_DECAY", "0.85")))
    flow_floor: float = field(default_factory=lambda: float(_get("FLOW_FLOOR", "0.05")))
    flow_cap: int = field(default_factory=lambda: _get_int("FLOW_CAP", 512))
    flow_inject_confirmed: float = field(default_factory=lambda: float(_get("FLOW_INJECT_CONFIRMED", "1.0")))
    flow_spread_factor: float = field(default_factory=lambda: float(_get("FLOW_SPREAD_FACTOR", "0.4")))
    flow_session_decay: float = field(default_factory=lambda: float(_get("FLOW_SESSION_DECAY", "0.6")))
    # Continuous activation axis in the reflex packet. Feeds RANKING only, never match_strength /
    # coverage, so instinct surfaces a memory but can never fabricate a confident answer.
    reflex_w_activation: float = field(default_factory=lambda: float(_get("REFLEX_W_ACTIVATION", "0.4")))
    # Field-seeded candidates: union the top-activated ids into the reflex seed set (store-gated).
    flow_field_seed: bool = field(default_factory=lambda: _get_bool("FLOW_FIELD_SEED", "0"))
    flow_seed_topk: int = field(default_factory=lambda: _get_int("FLOW_SEED_TOPK", 8))
    # Cold-start priming amplitudes (0.0 = no prime, the default).
    flow_prime_query: float = field(default_factory=lambda: float(_get("FLOW_PRIME_QUERY", "0.0")))
    flow_prime_ingest: float = field(default_factory=lambda: float(_get("FLOW_PRIME_INGEST", "0.0")))
    # FOOTGUN: salience-modulated decay reads each active id's static salience per turn. The engine
    # precomputes the map off the field lock, but it still costs O(active-field) store reads per ask
    # -- do NOT default this on without an inject-time salience snapshot (in-memory decay). Default 0.
    flow_salience_decay: bool = field(default_factory=lambda: _get_bool("FLOW_SALIENCE_DECAY", "0"))
    # Predictive idle warmup of Markov-predicted next queries into the PrefetchCache.
    flow_warmup_enabled: bool = field(default_factory=lambda: _get_bool("FLOW_WARMUP", "0"))
    flow_warmup_topk: int = field(default_factory=lambda: _get_int("FLOW_WARMUP_TOPK", 3))
    # Hybrid retrieve activation channel (reflex-miss / REFLEX_RECALL=0 paths get instinct too).
    flow_hybrid_channel_enabled: bool = field(default_factory=lambda: _get_bool("FLOW_HYBRID_CHANNEL", "0"))
    flow_hybrid_weight: float = field(default_factory=lambda: float(_get("FLOW_HYBRID_WEIGHT", "0.35")))
    # Activation-informed scratchpad / context assembly blend.
    flow_context_weight: float = field(default_factory=lambda: float(_get("FLOW_CONTEXT_WEIGHT", "0.25")))

    # --- Connected Brain Loop: observation-only spine (all default OFF; baseline byte-identical) -
    # RecallTrace: the retriever records WHY it found/missed (enabled channels, per-channel rankings
    # and weights, fused scores, gist provenance, stage latency, dropped candidates). It is a pure
    # side channel -- when on, the returned candidate list is identical to the flags-off path.
    recall_trace_enabled: bool = field(default_factory=lambda: _get_bool("RECALL_TRACE", "0"))
    # BrainEvent log: the engine emits a typed improvement stream (ingest/recall/verify/abstain/
    # contradiction/...). In-memory and NON-LEARNING in this slice; if ever persisted or fed to a
    # learner it MUST honor the FeedbackBuffer integrity wall (is_benchmark_namespace / is_dev).
    brain_events_enabled: bool = field(default_factory=lambda: _get_bool("BRAIN_EVENTS", "0"))

    # --- Dreaming engine: token-free continuous consolidation (all sweepable) -----------
    # Replay priority = surprise^w_s * need^w_n * (1-retrievability)^w_r (exponents).
    dream_replay_topk: int = field(default_factory=lambda: _get_int("DREAM_REPLAY_TOPK", 32))
    dream_w_surprise: float = field(default_factory=lambda: float(_get("DREAM_W_SURPRISE", "1.0")))
    dream_w_need: float = field(default_factory=lambda: float(_get("DREAM_W_NEED", "1.0")))
    dream_w_retr: float = field(default_factory=lambda: float(_get("DREAM_W_RETR", "1.0")))
    # SHY downscaling: prune graph edges below this weight percentile from the INDEX (never
    # the store); 0 = off. Prune by edge WEIGHT only -- never by retrievability (flat curve).
    dream_prune_percentile: float = field(default_factory=lambda: float(_get("DREAM_PRUNE_PERCENTILE", "5.0")))
    dream_salience_cap: float = field(default_factory=lambda: float(_get("DREAM_SALIENCE_CAP", "20.0")))
    # KG embedding (TransE, numpy) -- slow-cadence batch.
    dream_kg_dim: int = field(default_factory=lambda: _get_int("DREAM_KG_DIM", 32))
    dream_kg_epochs: int = field(default_factory=lambda: _get_int("DREAM_KG_EPOCHS", 30))
    dream_kg_margin: float = field(default_factory=lambda: float(_get("DREAM_KG_MARGIN", "1.0")))
    # Inferred-edge gate: confidence threshold + max proposed per cycle. Token-free by default;
    # set DREAM_USE_LLM_NLI=1 for optional real-NLI enrichment (costs tokens).
    dream_infer_confidence: float = field(default_factory=lambda: float(_get("DREAM_INFER_CONFIDENCE", "0.7")))
    dream_infer_topk: int = field(default_factory=lambda: _get_int("DREAM_INFER_TOPK", 50))
    dream_use_llm_nli: bool = field(default_factory=lambda: _get_bool("DREAM_USE_LLM_NLI", "0"))
    # Horn rule mining.
    dream_rule_max_len: int = field(default_factory=lambda: _get_int("DREAM_RULE_MAX_LEN", 2))
    dream_rule_min_confidence: float = field(default_factory=lambda: float(_get("DREAM_RULE_MIN_CONFIDENCE", "0.5")))
    # Multi-resolution (RAPTOR-style) clustering levels + branching.
    dream_multires_levels: int = field(default_factory=lambda: _get_int("DREAM_MULTIRES_LEVELS", 3))
    dream_cluster_min: int = field(default_factory=lambda: _get_int("DREAM_CLUSTER_MIN", 4))
    # Predictive pre-fetch: cosine to reuse a pre-assembled context; number of query clusters.
    dream_prefetch_threshold: float = field(default_factory=lambda: float(_get("DREAM_PREFETCH_THRESHOLD", "0.9")))
    dream_prefetch_clusters: int = field(default_factory=lambda: _get_int("DREAM_PREFETCH_CLUSTERS", 16))

    # Consolidation
    consolidation_interval_sec: int = field(
        default_factory=lambda: _get_int("CONSOLIDATION_INTERVAL_SEC", 3600)
    )

    # Prod-only (Alibaba Cloud)
    oss_bucket: str = field(default_factory=lambda: _get("OSS_BUCKET"))
    oss_endpoint: str = field(default_factory=lambda: _get("OSS_ENDPOINT"))
    oss_access_key_id: str = field(default_factory=lambda: _get("OSS_ACCESS_KEY_ID"))
    oss_access_key_secret: str = field(default_factory=lambda: _get("OSS_ACCESS_KEY_SECRET"))
    oss_worm_retention_days: int = field(default_factory=lambda: _get_int("OSS_WORM_RETENTION_DAYS", 3650))
    adbpg_dsn: str = field(default_factory=lambda: _get("ADBPG_DSN"))
    gdb_endpoint: str = field(default_factory=lambda: _get("GDB_ENDPOINT"))

    # ---- Derived ----------------------------------------------------------
    @property
    def dashscope_base_url(self) -> str:
        return _REGION_ENDPOINTS.get(self.region, _REGION_ENDPOINTS["singapore"])

    @property
    def compatible_base_url(self) -> str:
        return _REGION_COMPAT.get(self.region, _REGION_COMPAT["singapore"])

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def substrate_dir(self) -> Path:
        return self.data_dir / "substrate"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "eidetic.sqlite"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.substrate_dir, self.index_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
