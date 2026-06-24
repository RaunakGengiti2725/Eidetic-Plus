# Eidetic-Plus — Architecture

> A lossless, verifiable, recency-independent memory engine for AI agents.

Photographic memory is a human myth. Eidetic-Plus does not try to *be* a perfect
human memory; it builds the thing the brain only approximates — **lossless capture
decoupled from forgetting, recall that does not degrade with a memory's age, and
verified reconstruction with full provenance so it never confabulates.**

Three defensible, beyond-human properties fall out of the design:

1. **Decoupled lossless retention from forgetting** — raw content is captured once
   and never destroyed; only an *index* of priorities forgets.
2. **Uniform-latency, uniform-accuracy retrieval irrespective of memory age**,
   stated precisely as **age-independence at fixed store size N** (this is *not* a
   claim to defeat ANN's size-dependence).
3. **Full source + temporal provenance** with bi-temporal contradiction handling;
   every answer cites the immutable object it was verified against.

---

## 0. The organizing principle

There is exactly **one axis** that changes between the development and production
deployments: the **storage / database backend**. Every model call — embedding,
salience, extraction, rerank, generation, NLI — is **byte-identical** in dev and
prod (`config.py` docstring). This is why the per-component tables below show a
"dev backend / prod backend" split for the *storage* components (1, 2) and **no
backend swap at all** for the pure-compute components (3, 4, 6): those are the same
code regardless of environment.

```
APP_ENV=dev   -> LocalCASSubstrate + SQLite + hnswlib/numpy + networkx PPR
APP_ENV=prod  -> OSS-WORM         + AnalyticDB-PG (HNSW) + GDB(edges) + networkx PPR
                 ^ storage/DB only ^                       all model calls unchanged
```

A missing `DASHSCOPE_API_KEY` makes any model call **fail loudly**
(`ModelCallError` -> HTTP 503). Nothing is ever fabricated.

---

## 1. The two-layer spine: *the index is not the content*

Eidetic-Plus rests on a single architectural commitment borrowed from the
**hippocampal indexing theory** of memory: **the index is not the content.** The
hippocampus does not store experience; it stores a sparse *pointer* into the
neocortical pattern that *is* the experience. Recall is reconstruction through that
pointer, not playback of a stored copy.

We split the system the same way, into two layers that are physically and
semantically decoupled:

| Layer | What it is | Mutability | Components | Forgets? |
|-------|-----------|------------|-----------|----------|
| **Layer A — the substrate** | The immutable, lossless, content-addressed store of raw bytes. The "perfect record." | **Write-once. Never mutated, never deleted.** | Component 1 | **Never** |
| **Layer B — the index** | A mutable, forgettable set of pointers + structure + priorities *over* the substrate: vectors, graph, structure codes, salience, FSRS state, bi-temporal validity. | Fully mutable. | Components 2–7 | Down-weights priority; closes (never deletes) facts |

Concretely (`models.py` design note):

- **Raw bytes** live in the substrate (Component 1) keyed by `sha256`. They are the
  *only* ground truth and are never touched after the first write.
- A **`MemoryRecord`** is pure index/state. It *points* at the raw bytes via
  `content_hash` / `raw_uri`. **Forgetting** mutates only its FSRS priority. A
  **contradiction** mutates only its bi-temporal `invalid_at` / `expired_at`. The
  raw object it points at is untouched in both cases.

This decoupling is the load-bearing idea. Because forgetting only ever touches
Layer B, a "forgotten" memory is still losslessly present in Layer A and can be
re-promoted in O(1) (down-weighting an index entry, not re-creating a record). And
because the substrate is content-addressed and write-once, provenance is exact: a
citation is a hash, and the bytes behind that hash can never have silently changed.

---

## 2. The seven components, in depth

### Component 1 — Immutable, lossless, content-addressed substrate

| | |
|---|---|
| **Responsibility** | Store raw input bytes exactly once, content-addressed, write-once. Be the ground truth for every citation and every NLI verification. Refuse all overwrite/delete. |
| **File** | `eidetic/substrate.py` |
| **Dev backend** | `LocalCASSubstrate`: append-only directory, object key = `sha256(bytes)`, sharded `data/substrate/<hash[:2]>/<hash>`. After an atomic temp-write + `os.replace`, the object is `chmod 0o444` so the **OS itself** blocks overwrite — a real write-once guarantee, not a convention. |
| **Prod backend** | `OSSWORMSubstrate`: Alibaba Cloud **OSS with WORM** (Write-Once-Read-Many) retention. Same `put/get/exists` contract; immutability is enforced server-side by the bucket retention policy, and the class additionally never issues delete/overwrite. Keyed `eidetic/<sha256>`. Requires `oss2` + `OSS_*` env vars. |
| **Key data structures** | `Substrate` ABC; `put(data) -> (content_hash, raw_uri)` (idempotent on dedup); `get(content_hash) -> bytes`; `verify()` recomputes `sha256` and confirms it matches the key. `delete()` is **defined to raise** `ImmutableViolation` — both substrates inherit the refusal. |

Forgetting *never* reaches this layer. The only `delete` path is a hard error.

### Component 2 — Hippocampal index (vector ANN + bi-temporal graph)

This is the index *half* of the hippocampal analogy: two complementary structures
over the substrate.

**(2a) Vector ANN** — content similarity.

| | |
|---|---|
| **Responsibility** | Approximate-nearest-neighbour retrieval over content embeddings. Ranks by **content similarity only**; a memory's age never enters the distance. |
| **File** | `eidetic/vector_index.py` |
| **Dev backends** | `NumpyVectorIndex` — exact cosine, brute-force `matrix @ q` (O(N), always available). `HnswVectorIndex` — real **HNSW** (`hnswlib`, cosine space, `ef_construction=200`, `M=16`), latency sublinear in N. `VECTOR_BACKEND=auto` prefers HNSW if the wheel is present, else numpy. |
| **Prod backend** | **AnalyticDB for PostgreSQL** with an **HNSW** vector index — same ANN story as the dev `hnswlib` path. |
| **Key data structures** | Per memory, **two** vectors are stored: the **content embedding** (`text-embedding-v4`, `EMBED_DIM=1024`) and the **structure code** (Component 3, `STRUCT_DIM=64`). `search()` returns top-k `(memory_id, cosine_sim)`; `search_struct()` ranks by the small structure matrix. Persisted to `numpy_index.npz` (dev numpy) or `hnsw.bin` + meta (dev HNSW). |

**(2b) Bi-temporal knowledge graph + in-app Personalized PageRank** — association.

| | |
|---|---|
| **Responsibility** | Hold extracted `(src, relation, dst)` facts as bi-temporal edges; provide associative multi-hop retrieval via Personalized PageRank (PPR), exactly as HippoRAG does. |
| **File** | `eidetic/graph.py` (edges persisted by `eidetic/store.py`) |
| **Dev backend** | Edges in SQLite (`edges` table); the graph is materialized in-memory with **networkx** at query time. |
| **Prod backend** | Edge persistence in **Alibaba Cloud GDB** (graph database). **PPR still runs in-app via networkx** in both dev and prod — GDB stores edges; it is *not* a managed PPR service. |
| **Key data structures** | `Edge{src, dst, relation, fact, valid_at, invalid_at, created_at, expired_at}`. `add_fact()` applies the contradiction rule (below) and returns `(new_edge, invalidated_edges)`. `build_nx(at)` materializes only edges active at world-time `at`. `ppr_entities()` runs `nx.pagerank` (alpha 0.85) on the **undirected** graph with seed-personalization; `score_memories()` maps entity PPR onto candidate memories and normalizes to [0,1]; `node_features()` yields `{ppr, degree}` per entity for Component 3. |

**Contradiction rule (bi-temporal, also Component 7):** when a new fact shares
`(src, relation)` with an active edge but a *different* `dst`, the old edge is
**closed** by setting `invalid_at = expired_at = valid_at` of the new fact. It is
**never deleted** — the full history stays queryable "as of" any past time.

### Component 3 — Cognitive-coordinate map (metadata structure-code)

| | |
|---|---|
| **Responsibility** | Give each memory a *second* vector built from explicit **structure** (not content), so retrieval can generalize across context. |
| **File** | `eidetic/structure_code.py` |
| **Backend** | Pure compute — **identical in dev and prod**. Stored beside the content embedding in the vector index (2a). **No model call**; computed deterministically from metadata. |
| **Key data structures** | A feature-hashed vector (`STRUCT_DIM=64`). Features: `modality`, `source`, `entity:*`, **cyclic** temporal coordinates (sin/cos of hour-of-day and day-of-week), and bucketed graph-position features (`graph:ppr_bin`, `graph:deg_bin`) from Component 2b. `build_structure_code(record, dim, graph_features)` for memories; `build_query_structure_code(entities, dim)` for queries. |

**Honesty note (Tolman–Eichenbaum is *inspiration only*).** The Tolman–Eichenbaum
Machine motivates the *idea* of a reusable cognitive coordinate system. What ships
is the **honest metadata structure-code approximation**, not a learned TEM. A
deliberate consequence is baked in:

- **Absolute age / `valid_at` is NOT encoded.** Temporal structure is **cyclic
  only**. Putting recency into a vector that influences ranking would slope the
  recall-vs-age curve — the exact thing this project disproves.

### Component 4 — Write-time salience gate (neuromodulatory analog)

| | |
|---|---|
| **Responsibility** | At write time, score how strongly a memory should be retained, and seed its initial forgetting state. `salience = f(surprise, importance)`. |
| **File** | `eidetic/salience.py` |
| **Backend** | Pure compute + a real `qwen-flash` call — **identical in dev and prod**. |
| **Key data structures** | `Salience{surprise, importance, salience}`. **Surprise** is Bayesian-surprise-style: `1 − cosine_sim` to the nearest already-stored memory (a genuinely novel event is far from everything seen; `1.0` if nothing is stored yet). **Importance** is a real `qwen-flash` judgment in `[0,1]`. `salience = 0.45·surprise + 0.55·importance`. This seeds Component 5's initial FSRS state. |

### Component 5 — Offline consolidation/replay + FSRS forgetting

| | |
|---|---|
| **Responsibility** | (a) Model the forgetting curve (FSRS / DSR) that sets each memory's **index priority**. (b) Run the offline "sleep" replay loop that consolidates high-salience clusters into verified semantic summaries and decays priority. **Never deletes raw.** |
| **Files** | `eidetic/fsrs.py` (the DSR model) + `eidetic/engine.py` (`Engine.consolidate`, the replay loop). *(Note: `eidetic/__init__.py` mentions a `consolidation.py`; that file does not exist — consolidation lives in `engine.py`.)* |
| **Backend** | FSRS math is pure compute. The replay loop's summary + NLI are real `qwen-plus` calls. Identical in dev and prod; only the *scheduling* differs (see Section 3 / Section 6). |
| **Key data structures** | `FSRSState{stability, difficulty, retrievability, last_review, reps, lapses}`. Retrievability follows the **FSRS-6 power-law** curve `R = (1 + FACTOR·t/S)^DECAY` (`DECAY=-0.5`, `FACTOR=19/81`, so R=0.9 at t=S days), which fits human forgetting better than Ebbinghaus's exponential. `init_state()` seeds D/S/R from salience; `reinforce()` (reconsolidation) resets R→1 and grows S; `lapse()` lowers S; `decay()` refreshes cached R. |

**The critical invariant:** FSRS forgetting **down-weights index priority only** —
it lowers a memory's replay/surfacing priority. It **never deletes the raw record**,
and (see Section 4) it is **deliberately excluded from the cued-retrieval ranking
path**. `FSRSState.priority()` exists for replay scheduling and ambient surfacing,
and its docstring states outright that it is kept *out* of ranking. Reawakening (a
strong cue / confirmed recall) resets retrievability and boosts stability in **O(1)**
— it re-promotes a down-weighted index entry, never re-creates a record.

### Component 6 — Reconstructive, verifiable retrieval

| | |
|---|---|
| **Responsibility** | Turn a query into a grounded, cited answer through a multi-stage retrieve → fuse → rerank → generate → verify pipeline. |
| **File** | `eidetic/retrieval.py` (`Retriever`) |
| **Backend** | Pure orchestration + real Qwen calls — **identical in dev and prod**. |
| **Key data structures** | `RetrievalCandidate{record, dense_score, graph_score, fused_score, rerank_score}`; `Answer{...}`; `Citation{...}`. Tunables: `ANN_TOPK=100` (k1), `FINAL_TOPK=10` (k2), `RRF_K=60`. If `qwen3-rerank` is unavailable, the pipeline falls back to the fused order (still age-independent). |

Pipeline detailed in Section 3 (ASK path).

### Component 7 — Provenance + contradiction engine

| | |
|---|---|
| **Responsibility** | Make every fact and every answer fully traceable in **two time dimensions**, and resolve contradictions by closing — not deleting — superseded facts. |
| **Files** | `eidetic/graph.py` (bi-temporal edge invalidation), `eidetic/retrieval.py` (citations attached to answers), `eidetic/store.py` (bi-temporal SQL filters) |
| **Backend** | Pure logic over the index store — identical in dev and prod. |
| **Key data structures** | **Bi-temporal coordinates** on both `MemoryRecord` and `Edge`: `valid_at` / `invalid_at` = **world time** (when the fact was/ceased to be true) and `created_at` / `expired_at` = **system time** (when we learned / superseded it). `is_active_at(t)` gates visibility. `active_ids_at(t)` is an SQL-level bi-temporal filter. Every `Citation` carries `memory_id`, `content_hash`, `raw_uri`, `source`, `valid_at`, `snippet`, `nli_label`, `nli_score`. |

---

## 3. Data-flow paths

There are two paths, mirroring wake and sleep in biological memory.

### WAKE — per input (synchronous)

**Write path (`Engine.ingest`):**

```
input bytes
  -> ingest / multimodal view        (ingestion.py: OCR / ASR / doc / describe -> embeddable text)
  -> sha256 dedup                     (engine: get_by_hash; identical content returns the existing record)
  -> immutable substrate.put()        (Component 1; write-once, 0o444 / OSS-WORM)
  -> embed_text()                     (text-embedding-v4; real call)
  -> salience gate                    (Component 4; surprise vs index BEFORE add + qwen-flash importance)
  -> graph extraction                 (qwen-plus triples) with bi-temporal contradiction handling
                                        (Component 2b/7: same (src,relation), new dst -> close old edge)
  -> structure code                   (Component 3; metadata + graph PPR/degree; cyclic time only)
  -> vector index.add()               (Component 2a; content vec + structure vec) + save
  -> upsert MemoryRecord              (store.py; index/state row pointing at the raw bytes)
```

Key ordering details: **dedup happens before embedding** (cost control + provenance),
and **salience is computed against the index state *before* the new memory is
added** (so "surprise" measures true novelty).

**Read path (`Engine.ask` -> `Retriever.answer`):**

```
query
  -> embed_text()                     (text-embedding-v4)
  -> ANN top-k1 search                (Component 2a; content similarity, k1=ANN_TOPK)
  -> bi-temporal filter               (Component 7; keep only memories active at `at` via active_ids_at)
  -> Personalized PageRank            (Component 2b; seed PPR from entities of top dense hits -> associate)
  -> Reciprocal Rank Fusion (RRF)     (fuse dense ordering + graph ordering; RRF_K=60)
  -> qwen3-rerank                     (Component 6; rerank the fused shortlist -> final top-k2)
  -> qwen3-max generation             (answer strictly over retrieved sources, cite [S0],[S1]...)
  -> NLI entailment verify            (Component 6; premise = IMMUTABLE raw record, hypothesis = answer)
  -> cite                             (Component 7; attach content_hash + raw_uri + valid_at + nli label)
  -> reconsolidate                    (Component 5; reinforce() the memories that ENTAILED a verified answer)
```

Two precise notes on this path:

- **The FSRS priority weight is never read here.** Ranking is content similarity +
  graph association + the rerank model. This exclusion is what keeps recall@k
  age-independent (Section 4).
- **Verification flags; it does not hard-reject.** See Section 5.

### SLEEP — scheduled (offline replay; `Engine.consolidate`)

```
all records
  -> cluster                          (group by first-entity-or-modality key; pattern separation)
  -> select high-salience clusters    (keep records with salience >= 0.6, not yet consolidated, cluster size >= 2)
  -> consolidate_summary()            (qwen-plus: one faithful semantic summary over the replayed episodes)
  -> verify summary                   (NLI each raw trace vs summary; reject the summary only on a CONTRADICTION)
  -> write summary onto records        (sets record.summary + consolidated=True; raw is NEVER replaced)
  -> FSRS decay()                     (Component 5; refresh cached retrievability = lower index priority)
```

**Never deletes raw.** Consolidation produces a semantic *gist* (`record.summary`)
that sits *beside* the immutable trace; the raw object remains ground truth and the
citation path always resolves to it.

**Scheduling.** In **prod**, this loop is a scheduled cron job (Alibaba Cloud
**Function Compute**, period `CONSOLIDATION_INTERVAL_SEC`). In **dev**, there is no
background scheduler; the same loop is triggered manually via `POST /api/consolidate`.

---

## 4. The recency-independence argument (stated precisely)

The headline property is **age-independence at fixed store size N**. Stated exactly:

> At a fixed number of stored memories N, retrieval ranks candidates by **content
> similarity** (plus graph association plus the rerank model). The FSRS index-priority
> weight, and any absolute-age signal, are **intentionally excluded** from everything
> that influences rank. Therefore **recall@k and retrieval latency depend on the
> store size N, but never on a given memory's age.**

Why it holds, traced through the code:

1. **Ranking uses no age signal.** The ANN distance is cosine similarity over
   content embeddings (`vector_index.py`); the structure code (Component 3) encodes
   only *cyclic* time, never absolute age; PPR is associative; rerank is a
   content-relevance model. Nothing age-correlated touches the score.
2. **FSRS priority is deliberately kept out of ranking.** Forgetting exists (it
   schedules replay and ambient surfacing) but `FSRSState.priority()` is never read
   in `Retriever.retrieve`. The `models.py` docstring states this is intentional —
   *precisely so that recall@k stays age-independent.*
3. **The two exclusions share one intent.** Excluding FSRS priority (Component 5)
   and excluding absolute age from the structure code (Component 3) are the *same*
   design decision viewed twice: keep anything age-correlated out of anything that
   influences rank. Both feed the same observable — the flat curve.

**What this is NOT.** This is *not* a claim that latency is independent of how much
is stored. numpy is O(N); HNSW is sublinear in N. Both grow with N. The claim is
strictly about **age at fixed N**.

**The signature result.** `scripts/signature_demo.py` ingests N distinct memories
with `valid_at` spread uniformly across ~30 simulated years, queries each by a
strong content cue against the other N−1 as distractors, bins by age, and plots
**recall@k** and **p95 retrieval latency** versus age. Both curves come out **flat**
(near-zero linear slope), with real `text-embedding-v4` embeddings over real HNSW.
Output: `artifacts/signature_recall_latency_vs_age.png`. A 30-year-old memory is
recalled as well, and as fast, as a one-second-old one.

**Contrast with long-context "lost in the middle".** Stuffing history into a long
context window does the opposite: retrieval quality is *strongly* position-dependent
— LLMs reliably attend to the start and end of the window and lose information in
the middle, so an item's recall degrades with *where* it sits in the context. Worse,
context is bounded, so old memories eventually fall out entirely. Eidetic-Plus has
no "middle": every memory is an equal first-class entry in an external index, ranked
only by relevance to the cue, so position/age confer no advantage or penalty.

---

## 5. Verification / anti-confabulation design

The anti-confabulation backbone is **natural-language-inference (NLI) entailment
against the immutable record**:

1. `qwen3-max` generates an answer **strictly over the retrieved sources** (system
   prompt: answer only from sources, cite inline, never invent facts).
2. For each candidate, the **premise** for NLI is the **immutable raw record**
   (`Retriever._ground_truth` reads the bytes back from the substrate by
   `content_hash`; falls back to the stored transcription/description for non-text
   modalities, whose raw bytes still remain ground truth). The **hypothesis** is the
   generated answer. `qwen-plus` returns `entailment | neutral | contradiction` with
   a confidence.
3. **The answer is flagged, not silently dropped.** An answer is `verified=True`
   only if **≥1 source entails** it. If nothing entails it, the answer is still
   returned but with `verified=False`, `unverified_claims=[answer]`, an explanatory
   `note` ("unverified: no source entails the answer"), and its citations retained
   so a human/agent can inspect the gap. When verified, citations are filtered to
   the entailing sources. An empty store / no-active-memory returns the canned
   *"I do not have that in memory."* with `verified=True`.
4. **Every answer cites the immutable source.** Each `Citation` carries
   `content_hash` + `raw_uri` + `source` + `valid_at` + the NLI label/score, so the
   exact bytes the answer was checked against are addressable
   (`GET /api/raw/{content_hash}` returns those bytes).
5. **Reconsolidation closes the loop.** Only the sources that *entailed* a verified
   answer are reinforced (`fsrs.reinforce`) — confirmed recall strengthens the right
   memories.

The same NLI gate guards consolidation: a semantic summary is rejected if it
**contradicts** any raw trace in its cluster.

---

## 6. Alibaba Cloud service mapping

The only thing that changes from dev to prod is this column of backends. Every
model call stays the same.

| Capability | Component(s) | Dev (local) | Prod (Alibaba Cloud) |
|---|---|---|---|
| Immutable raw store | 1 | `LocalCASSubstrate` (0o444, sha256) | **OSS with WORM retention** (server-side immutability) |
| Vector ANN | 2a | `hnswlib` HNSW / numpy exact | **AnalyticDB for PostgreSQL** with **HNSW** |
| Bi-temporal graph (edges) | 2b / 7 | SQLite `edges` table | **GDB** (graph database) for edge persistence |
| Personalized PageRank | 2b | **networkx / scipy, in-app** | **networkx / scipy, in-app** (GDB stores edges; PPR is *not* a managed service) |
| Index/state store | 2 / 5 / 7 | SQLite (`MemoryRecord`, FSRS, structure codes, bi-temporal) | SQLite / managed RDBMS (same schema) |
| Scheduled "sleep" loop | 5 | manual `POST /api/consolidate` | **Function Compute** cron (`CONSOLIDATION_INTERVAL_SEC`) |
| API service | (host) | `uvicorn eidetic.api:app` | **ECS / ACK** running the same FastAPI app |
| Embeddings | 2a / 3-feed / 4 | `text-embedding-v4` (DashScope) | `text-embedding-v4` (DashScope) |
| Salience / importance | 4 | `qwen-flash` (DashScope) | `qwen-flash` (DashScope) |
| Extraction / NLI / contradiction | 2b / 6 / 7 | `qwen-plus` (DashScope) | `qwen-plus` (DashScope) |
| Rerank | 6 | `qwen3-rerank` (DashScope) | `qwen3-rerank` (DashScope) |
| Generation | 6 | `qwen3-max` (DashScope) | `qwen3-max` (DashScope) |
| Multimodal ingestion | (ingest) | `qwen-vl-ocr`, `qwen3-asr-flash`, `qwen-doc-turbo`, `qwen-vl-plus` | same DashScope models |

All model traffic goes through `eidetic/dashscope_client.py` (DashScope, region
default Singapore: `dashscope-intl.aliyuncs.com`). There are **no mocks** — a missing
`DASHSCOPE_API_KEY` raises `ModelCallError` (surfaced as HTTP 503), never a fabricated
result. Prod is selected by `APP_ENV=prod` with `OSS_*`, `ADBPG_DSN`, and
`GDB_ENDPOINT` in `.env`.

---

## 7. One-screen architecture diagram

```
============================================================================
 LAYER 0 — FRONTEND
   Browser UI (eidetic/web/index.html)  +  Agent / programmatic clients
============================================================================
            |  HTTP (FastAPI: GET / , /api/stats, /api/memories[/text|/file],
            v        /api/ask, /api/memories/{id}, /api/raw/{hash},
                     /api/consolidate, /api/reawaken/{id})
============================================================================
 LAYER 1 — BACKEND ORCHESTRATION   (eidetic/api.py -> eidetic/engine.py : Engine)
   WAKE-write: ingest -> sha256 dedup -> SUBSTRATE -> embed -> salience(4)
               -> extract+contradiction(2b/7) -> structure code(3) -> index(2a)
   WAKE-read : ASK -> ANN(2a) -> bitemporal filter(7) -> PPR(2b) -> RRF
               -> rerank(6) -> generate(6) -> NLI verify(6) -> cite(7)
               -> reconsolidate(5)
   SLEEP     : consolidate(5) -> verified semantic summaries -> FSRS decay
                                  (prod: Function Compute cron; never deletes raw)
============================================================================
 LAYER 2 — STORAGE
   [Component 1] IMMUTABLE SUBSTRATE   dev: LocalCAS(0o444) | prod: OSS-WORM
        ^   ground truth: sha256-addressed raw bytes (write-once)            |
        |                                                                    |
   [Component 2a] VECTOR ANN  dev: hnswlib/numpy | prod: AnalyticDB-PG HNSW  |
        + [Component 3] structure codes (cyclic time only; no absolute age)  |
   [Component 2b/7] BI-TEMPORAL GRAPH  dev: SQLite | prod: GDB ; PPR in-app  |
   [Component 5/7] INDEX/STATE (SQLite): MemoryRecord, FSRS, valid/invalid   |
        (Layer 2 forgets & contradicts; it NEVER mutates Layer 1)            |
============================================================================  |
 LAYER 3 — QWEN CLOUD (DashScope; identical dev/prod)                         |
   text-embedding-v4 | qwen-flash | qwen-plus | qwen3-rerank | qwen3-max      |
   qwen-vl-ocr | qwen3-asr-flash | qwen-doc-turbo | qwen-vl-plus              |
============================================================================  |
                                                                              |
  PROVENANCE ARROW (every answer):                                            |
     Answer.citations[*].content_hash / raw_uri  ----------------------------+
        verified by NLI against the IMMUTABLE object in Layer 2 / Component 1
        (GET /api/raw/{content_hash} returns those exact bytes)
============================================================================
```

The provenance arrow is the spine of the diagram: **every answer points back, by
`content_hash`, to the immutable object it was NLI-verified against.** Trust is not
asserted; it is addressable.

---

## 8. Residual risks (honest)

This section is deliberately candid about where the framing is narrow or where a
component is an approximation.

- **Beijing-only fusion embedding (with fallback).** The strongest *fused*
  multimodal embedding (`qwen3-vl-embedding` fusion mode) is currently **Beijing-region
  only** (`dashscope.aliyuncs.com`). The default deployment runs **Singapore**
  (`dashscope-intl`) for latency/governance and therefore uses the
  `tongyi-embedding-vision-plus` multimodal embedder plus `text-embedding-v4` for
  text — a **fallback**, not the fusion path. Switching `DASHSCOPE_REGION=beijing`
  (and `MULTIMODAL_EMBED_MODEL=qwen3-vl-embedding`) enables fusion at the cost of
  region lock-in. This is a real capability gap in the default config, not a bug; no
  region is hardcoded.

- **Component 3 ships as the metadata structure-code approximation.** The
  cognitive-coordinate map is motivated by the **Tolman–Eichenbaum Machine as
  inspiration only**. What is implemented is a **deterministic, feature-hashed
  metadata structure-code** — not a learned grid/place-cell representation. It
  delivers the useful, defensible part (structure-aware, age-free coordinates) and
  honestly omits the learned-map ambition.

- **Age-independence is "at fixed N," precisely.** The flat-curve result is a claim
  about **memory age at a fixed store size**, *not* a claim to beat ANN's
  size-dependence. Retrieval latency and recall@k still scale with N (numpy O(N),
  HNSW sublinear). Anyone reading the signature plot should read it as *"age does not
  matter,"* not *"size does not matter."*

- **Verification is model-based NLI.** The entailment judge is `qwen-plus`, not a
  formal prover; it can err at fine granularity. The mitigations are structural: the
  premise is always the *immutable* raw record, unentailed answers are flagged
  (`verified=False`, `unverified_claims`), and the cited bytes are independently
  retrievable so the check can be audited or re-run.

---

## Upgrade: universal plugin, deepened vision, reconsolidation, and the 3D map

Everything above stays exactly true; this section only **adds**. The upgrade makes
the engine *universal* (a second transport + a packaged plugin), gives vision real
**graph** standing (not just a vector), deepens the engine with four more
biologically-grounded write-time mechanisms, sharpens the frontier components, and
ships a navigable **3D projection** of the index. The single load-bearing invariant
from Section 4 is preserved verbatim: **nothing new touches the cued-retrieval
ranking score.** Every new strengthen-signal flows into FSRS priority, graph edges,
or re-embedding — never into rank — so the flat recall-vs-age curve is unchanged.

### Component 8 — MCP transport (the universal plugin)

| | |
|---|---|
| **Responsibility** | Expose the **same `engine.py`** over a second wire protocol so any MCP host (Claude Code, agents, IDEs) can use Eidetic-Plus as a memory backend, with no change to the engine or its guarantees. |
| **File** | `eidetic/mcp_server.py` (**FastMCP**) |
| **Backend** | Pure transport over `Engine` — **identical in dev and prod**. FastAPI (`eidetic/api.py`) and this FastMCP server are **two transports over one engine**, not two engines. |
| **Tools (7)** | `remember`, `recall`, `consolidate`, `list_memories`, `prove_age_independence` carry a `Scope` (namespace + optional agent_id/project_id). `reawaken` and `get_raw` address a record by its opaque `memory_id`, so they take no scope arguments. Each maps onto an existing engine call. |
| **Wire modes** | Runs **stdio** (for a local plugin host) and **streamable-http** (for networked hosts) from the same server object. |

**Plugin packaging.** Eidetic-Plus ships as a **Claude Code plugin**:

```
.claude-plugin/plugin.json        # plugin manifest (name, version, MCP entry)
.mcp.json                         # server spec; command path uses ${CLAUDE_PLUGIN_ROOT}
.claude-plugin/marketplace.json   # marketplace listing for discovery/install
```

`${CLAUDE_PLUGIN_ROOT}` keeps the server invocation path relocatable, so the plugin
resolves its own install directory regardless of where the host mounts it.

### The scoping model — namespace as a hard boundary

Every read and write carries a **`Scope`**: a **required `namespace`** (default
`"default"`, an explicit named scope — **never a global wildcard**) plus optional
`agent_id` and `project_id`. The namespace is a **hard isolation boundary**: a write
in namespace A is **invisible** from namespace B. Scope is enforced at three layers,
so isolation is not a single check that can be bypassed:

| Layer | Enforcement |
|---|---|
| **Store** (`store.py`) | Scope columns on each row; every query filters by `(namespace, agent_id?, project_id?)`. |
| **Graph** (`graph.py`) | Facts, PPR, and the contradiction rule all operate **only on scoped edges** — association never leaks across namespaces. |
| **Retrieval** (`retrieval.py`) | A **scope filter runs alongside the bi-temporal filter** (Component 7): candidates must be both *active at `at`* **and** *in scope*. |

**Dedup is per-scope.** The substrate (Component 1) is content-addressed, so identical
**raw bytes are shared globally** — there is one immutable object per `sha256`,
regardless of who wrote it. The **index record is distinct per `(scope, content_hash)`**:
the same bytes remembered in two namespaces produce two independent `MemoryRecord`
pointers at one shared object. Forgetting, FSRS state, and validity are per-record, so
the two scopes forget independently while never duplicating ground truth.

**Honesty note (post-filtering vs. per-namespace indexes).** Isolation is satisfied
by **post-filtering**: the vector ANN searches the shared index and results outside
the active scope are dropped before ranking. This is correct — no cross-namespace
result can ever surface. Under a single large store mixing many namespaces, post-filter
can thin a fixed top-k1 and cost recall for a small scope. **Per-namespace indexes are
a documented future optimization** for recall under large mixed stores; they change
performance, not the isolation guarantee.

### Deepened vision — images as first-class graph citizens

Section 2 stored an image as a vector plus a transcription. The upgrade makes vision a
**full participant in the bi-temporal graph and in verification.**

- **Images → entities + edges.** An ingested image is described/read by
  `qwen-vl-plus` / `qwen-vl-ocr` and the result is **extracted into `(src, relation, dst)`
  facts** — real **entities and edges in the bi-temporal graph** (Component 2b/7), subject
  to the same contradiction rule. An image is no longer just a point in vector space; it
  is structure.
- **Verified visual recall.** Visual verification is the **same NLI-against-the-immutable-record
  pattern** as Section 5, with a **vision judge**: a visual claim is checked by `qwen-vl-plus`
  against the **raw image pixels** — the **raw image is the arbiter**. An unsupported visual
  claim is rejected exactly as an unentailed text answer is flagged. The immutable object
  (here, the original image bytes) remains the ground truth the answer is checked against.

### Engine deepening — four more write-time mechanisms

Four mechanisms join the WAKE-write and SLEEP paths. **Each has a dossier basis, and
each carries the same invariant: it feeds FSRS priority / edges / re-embedding only —
none enters the ranking score.**

| Mechanism | What it does | Dossier basis | Invariant |
|---|---|---|---|
| **Reconsolidation as a write path** | A confirmed recall **re-embeds** the trace and **up-weights** its FSRS state (= immune **affinity maturation**); a contradicted recall **suppresses** it — **never deletes**. | **6.10** + immune **affinity maturation 7.2** | Feeds re-embedding + FSRS priority only; not rank. |
| **Synaptic tagging-and-capture** | A salient event **up-weights temporally adjacent memories** (a strong event "captures" its neighbors). | **6.7** | Feeds FSRS priority only; not rank. |
| **Surprise-based event segmentation** | Long input is **chunked at Bayesian-surprise boundaries** before storage (pattern separation at the event seam). | **EM-LLM 4.2 / 13.5** | Shapes *what* becomes a record; not rank. |
| **Schema-accelerated consolidation** | **Schema-consistent facts fast-track** through the SLEEP loop (familiar structure consolidates faster). | **Tse et al. 6.11** | Feeds consolidation scheduling; not rank. |

The `consolidate` result surfaces this as `schema_fast_tracked` in
`POST /api/consolidate`. The reconsolidation write path is the same `fsrs.reinforce` /
suppress hook described in Section 5, now generalized: **confirmed = re-embed + up-weight,
contradicted = suppress (never delete).**

### Frontier — sharpened structure/content factorization, attention readout, co-activation

These deepen Components 2–3 along the lines the dossier motivates, still as honest
approximations:

- **Sharpened structure-content factorization (TEM 6.4 / 9.3 / 13.4, *approximation only*).**
  The structure code (Component 3) is sharpened so the **structure code carries relational-role
  + scope features, independent of the content embedding** — a cleaner factorization of
  *where/how* from *what*. As Section 3 and Section 8 already commit: this is the **metadata
  structure-code approximation, not a learned TEM**, and it still encodes **cyclic time only,
  never absolute age**.
- **Hopfield / attention readout (8.1–8.2).** A **softmax pattern-completion over the retrieved
  set** (associative-memory readout) cleans up the shortlist. It operates *within* the
  already-retrieved candidates; it does not introduce an age signal into rank.
- **Memory linking by co-activation (6.7 / 6.10).** **Co-confirmed memories gain a
  strengthened edge** — memories that are repeatedly verified together become associatively
  linked. This adds a graph **edge** (and a `co_activated` edge kind, below); it is association,
  not a ranking-score term.

### prove_age_independence — a live capability

The signature result of Section 4 is now a **live capability**, available as both an API
route and an MCP tool:

```
GET /api/prove_age_independence?namespace=&agent_id=&project_id=&k=5
  -> {ok, n, k, overall_recall, overall_p95_ms,
      recall_slope_per_year, latency_slope_ms_per_year,
      age_centers_years, recall_per_bin, p95_ms_per_bin, flat}
```

It computes **recall@k and p95 latency versus memory age on the current store** and reports
the slopes; **both come out flat** (`flat: true`). The MCP tool `prove_age_independence`
returns the same result over the plugin transport — any host can demand the proof on demand,
in-scope, against live data.

### The 3D memory map — a *view*, never storage

`GET /map` (`eidetic/web/map.html`) renders a navigable **3D map** of the index using the
vendored `3d-force-graph` build (`/static/vendor/3d-force-graph.min.js`, UMD, no CDN). The
3D coordinates are a **PCA projection of the high-dimensional (1024–2048-D) embeddings** —
a **VIEW for navigation, not a storage format.** The data comes from:

```
GET /api/graph?namespace=&agent_id=&project_id=
  -> { nodes: [{id, label, salience, retrievability, modality,
                valid_at, content_hash, source, invalidated, x, y, z}],
       edges: [{source, target, kind:"entity"|"co_activated",
                active(bool), relation?}],
       projection, note }
```

Visual encoding: **node color = salience**, **node size = retrievability**, **edges carry
bi-temporal validity** (an `active` flag; closed/contradicted facts and `co_activated`
links are distinguishable). The `x, y, z` are the PCA coordinates; `projection` and `note`
record that they are a projection.

**Explicit invariant — memory is never stored in 3D.** The engine stores 1024–2048-D
embeddings; the 3D positions exist **only** as a projection for the human eye. Collapsing
memory into 3 dimensions would **destroy the separating structure** that makes
age-independent retrieval possible (Section 4, Section 8). The map *reads down* from high-D
to 3D for display; the index *never* round-trips through 3D.

### The invariant, restated in one line

> **The four new strengthen-signals — reconsolidation, synaptic tagging-and-capture,
> surprise segmentation, and schema-accelerated consolidation — feed FSRS priority,
> graph edges, and re-embedding ONLY; none of them ever enters the cued-retrieval
> ranking score, so recall@k stays age-independent at fixed store size N.**

---

## Upgrade: the Dreaming Engine (token-free continuous consolidation)

`eidetic/dreaming/` adds an offline consolidation layer that runs with **ZERO LLM calls**
(local math over stored embeddings + the graph only) and is **strictly additive** — every
output is a reversible, content-addressed, provenance-tagged `DerivedRecord` or a flagged
inferred `Edge`. The observed lossless store is **never** merged, averaged, or mutated.

**Components** (invoked via `Engine.dream(scope=...)`):
- `replay.py` — continuous replay scheduler. Priority = surprise^a·need^b·(1−R)^c. SHY-style
  global pass renormalizes edge weights and prunes the weakest **by weight only** from the
  index (never by FSRS retrievability — pruning by age would slope the flat curve; never the
  store). FSRS stability is capped to stop rich-get-richer.
- `kg_embed.py` (TransE, numpy) + `rules.py` (bounded 2-hop Horn-rule mining) + Louvain
  schemas — `infer.py` proposes edges/facts, gates each via `gate.py`, and writes admitted
  ones to the **inferred layer** (`Edge.inferred=True`) and schema centroids to `derived`.
- `multires.py` — RAPTOR-style recursive **bounded-k** k-means (near-linear) building centroid
  gist nodes per level (`DerivedRecord(kind="gist")`); `search()` ranks gists by cosine.
- `prefetch.py` — query-log clustering + a pre-assembled-context cache matched by cosine.

**Storage:** `models.DerivedRecord` + the `derived` table (kind/namespace/level/centroid);
`Edge` gains `inferred`/`confidence`/`provenance`/`weight`/`pruned`. `store.all_edges` and
`graph.build_nx` **exclude inferred edges and pruned edges by default**; `include_inferred=True`
surfaces them flagged.

**Cardinal-rule invariants** (tested in `tests/test_dreaming.py`): additive-only (substrate
byte-identical after a dream cycle), inferred-namespace separation, near-linear replay
(`node_features` computed a constant number of times — not the per-record O(N²) that hung an
earlier run), the token-free gate, multi-resolution retrieval, and the pre-fetch hit.

**Token-free gate, precisely:** confidence + embedding support by default; real-NLI is optional
enrichment (`DREAM_USE_LLM_NLI=1`), never required. **Honest note:** schema *naming* and
contradiction *resolution* may still want an optional LLM pass; the token-free layer delivers
value (replay, inferred links, gist, pre-fetch) without one. Measured dream-on vs dream-off
via the harness `DREAM_AB=1` hook against the neutral scoreboard.
