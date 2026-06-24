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
    doc_model: str = field(default_factory=lambda: _get("DOC_MODEL", "qwen-doc-turbo"))
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
    quant_refine: bool = field(default_factory=lambda: _get("QUANT_REFINE", "1") not in ("0", "false", "no"))
    quant_refine_topn: int = field(default_factory=lambda: _get_int("QUANT_REFINE_TOPN", 100))

    # HNSW index params (M=32; efSearch raised since retrieval is not the latency bottleneck).
    hnsw_m: int = field(default_factory=lambda: _get_int("HNSW_M", 32))
    hnsw_ef_search: int = field(default_factory=lambda: _get_int("HNSW_EF_SEARCH", 256))
    hnsw_ef_construction: int = field(default_factory=lambda: _get_int("HNSW_EF_CONSTRUCTION", 200))

    # Triple-win read path tuning.
    context_token_budget: int = field(default_factory=lambda: _get_int("CONTEXT_TOKEN_BUDGET", 8000))
    cache_cosine: float = field(default_factory=lambda: float(_get("CACHE_COSINE", "0.9")))
    reader_cot_enabled: bool = field(default_factory=lambda: _get("READER_COT", "0") in ("1", "true", "yes"))
    reader_router_enabled: bool = field(
        default_factory=lambda: _get("READER_ROUTER", "1") not in ("0", "false", "no")
    )
    conflict_resolver_enabled: bool = field(
        default_factory=lambda: _get("CONFLICT_RESOLVER", "0") in ("1", "true", "yes")
    )
    context_compress_enabled: bool = field(
        default_factory=lambda: _get("CONTEXT_COMPRESS", "0") in ("1", "true", "yes")
    )
    extract_light_enabled: bool = field(
        default_factory=lambda: _get("EXTRACT_LIGHT", "0") in ("1", "true", "yes")
    )
    temporal_rerank_enabled: bool = field(
        default_factory=lambda: _get("TEMPORAL_RERANK", "0") in ("1", "true", "yes")
    )
    hippo2_seeding_enabled: bool = field(
        default_factory=lambda: _get("HIPPO2_SEEDING", "0") in ("1", "true", "yes")
    )
    persistent_bm25_enabled: bool = field(
        default_factory=lambda: _get("PERSISTENT_BM25", "1") not in ("0", "false", "no")
    )
    semantic_cache_enabled: bool = field(
        default_factory=lambda: _get("SEMANTIC_CACHE", "1") not in ("0", "false", "no")
    )
    semantic_cache_adaptive: bool = field(
        default_factory=lambda: _get("SEMANTIC_CACHE_ADAPTIVE", "1") not in ("0", "false", "no")
    )

    # --- Sweepable optimization params (safe defaults; tune on a subset after the key) ---
    # Abstention gate: abstain when combined entailment+coverage is below this (calibrate it).
    abstention_threshold: float = field(default_factory=lambda: float(_get("ABSTENTION_THRESHOLD", "0.4")))
    # Cross-encoder rerank (qwen3-rerank): on/off + candidate depth (~50 -> final_topk).
    rerank_enabled: bool = field(default_factory=lambda: _get("RERANK_ENABLED", "1") not in ("0", "false", "no"))
    rerank_fail_open: bool = field(default_factory=lambda: _get("RERANK_FAIL_OPEN", "0") in ("1", "true", "yes"))
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
    adaptive_k_enabled: bool = field(default_factory=lambda: _get("ADAPTIVE_K", "0") in ("1", "true", "yes"))
    adaptive_k_min: int = field(default_factory=lambda: _get_int("ADAPTIVE_K_MIN", 3))
    # 2a Adaptive efSearch: widen the HNSW beam for hard (multi-hop/long) queries only.
    adaptive_ef_enabled: bool = field(default_factory=lambda: _get("ADAPTIVE_EF", "0") in ("1", "true", "yes"))
    hnsw_ef_search_hard: int = field(default_factory=lambda: _get_int("HNSW_EF_SEARCH_HARD", 512))
    # 2b Split-conformal retrieval depth: keep candidates with sim >= 1 - qhat. qhat<0 = off
    # (calibrate it on the DEV split via bench.calibrate; never on test items).
    conformal_depth_enabled: bool = field(default_factory=lambda: _get("CONFORMAL_DEPTH", "0") in ("1", "true", "yes"))
    conformal_alpha: float = field(default_factory=lambda: float(_get("CONFORMAL_ALPHA", "0.1")))
    conformal_qhat: float = field(default_factory=lambda: float(_get("CONFORMAL_QHAT", "-1.0")))
    # 2c Skip the cross-encoder rerank when the first-stage margin is already large. 0 = never.
    rerank_skip_margin: float = field(default_factory=lambda: float(_get("RERANK_SKIP_MARGIN", "0.0")))
    # 2c MMR diversity post-pass (lambda in [0.3,0.7]; higher = more diverse).
    mmr_enabled: bool = field(default_factory=lambda: _get("MMR_ENABLED", "0") in ("1", "true", "yes"))
    mmr_lambda: float = field(default_factory=lambda: float(_get("MMR_LAMBDA", "0.5")))
    # 2d Fusion method: rrf (default, scale-free) | zscore | minmax | dbsf | borda.
    fusion_method: str = field(default_factory=lambda: _get("FUSION_METHOD", "rrf").lower())
    # 2e Parallel channel fan-out (dense/BM25/recency concurrent; latency ~= slowest channel).
    parallel_channels_enabled: bool = field(default_factory=lambda: _get("PARALLEL_CHANNELS", "0") in ("1", "true", "yes"))

    # --- Layer 3a/3b: online learning (all default OFF) ----------------------------------
    # 3b Rocchio pseudo-relevance feedback: expand the query toward the top-R evidence
    # centroid, confidence-gated to avoid topic drift. Positive-only (gamma=0) by default.
    rocchio_enabled: bool = field(default_factory=lambda: _get("ROCCHIO", "0") in ("1", "true", "yes"))
    rocchio_alpha: float = field(default_factory=lambda: float(_get("ROCCHIO_ALPHA", "1.0")))
    rocchio_beta: float = field(default_factory=lambda: float(_get("ROCCHIO_BETA", "0.6")))
    rocchio_topr: int = field(default_factory=lambda: _get_int("ROCCHIO_TOPR", 5))
    rocchio_conf_gate: float = field(default_factory=lambda: float(_get("ROCCHIO_CONF_GATE", "0.35")))
    # 3a Online fusion-weight learner: when on, the content-channel (dense/bm25/graph) base
    # weights come from a dev-feedback-learned vector (index_dir/fusion_weights.json) instead
    # of the static config floats. Recency weight is NEVER learned (age-independence).
    fusion_learner_enabled: bool = field(default_factory=lambda: _get("FUSION_LEARNER", "0") in ("1", "true", "yes"))
    fusion_learner_method: str = field(default_factory=lambda: _get("FUSION_LEARNER_METHOD", "eg").lower())
    # Producer side: emit a (features, reward) feedback tuple from the PRODUCT ask() path into
    # the dev-only FeedbackBuffer. Off by default; the neutral benchmark adapter never calls
    # ask(), and benchmark namespaces are recorded audit-only, so this can't touch the wall.
    feedback_enabled: bool = field(default_factory=lambda: _get("FEEDBACK", "0") in ("1", "true", "yes"))

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
    dream_use_llm_nli: bool = field(default_factory=lambda: _get("DREAM_USE_LLM_NLI", "0") in ("1", "true", "yes"))
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
