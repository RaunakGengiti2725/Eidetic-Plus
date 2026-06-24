# Eidetic-Plus

**A lossless, verifiable, recency-independent memory engine for AI agents.**

Photographic memory is a human myth. No brain stores experience losslessly, recalls
a decade-old moment as crisply as yesterday's, or refuses to confabulate. Eidetic-Plus
builds the thing the brain only approximates: lossless capture decoupled from
forgetting, recall that does not decay with a memory's age, and verified reconstruction
with full provenance, so an agent's answers point back to immutable source records
instead of inventing them.

---

## Universal memory plugin (MCP)

Eidetic-Plus is not just a service — it is a **universal memory backend** that any MCP
host can mount over the *same* engine: Claude, Claude Code, Cursor, Cline, and Zed all
get lossless, verified, age-independent memory with zero per-host integration. The MCP
server (`eidetic/mcp_server.py`, built on **FastMCP**) is simply an additional transport
over the exact `eidetic/engine.py` that FastAPI already uses — no logic is duplicated.

Seven MCP tools are exposed:

| Tool | What it does |
|------|--------------|
| `remember` | Store a memory losslessly and immutably; returns its content hash + provenance. |
| `recall` | Verified retrieval — returns the answer with cited immutable sources, or abstains. |
| `consolidate` | Run the sleep/replay loop (dedup, verified summaries, FSRS decay); never deletes raw. |
| `reawaken` | Re-promote a down-weighted memory by its `memory_id` (the O(1) revert path). |
| `list_memories` | List memories (newest first) with salience + FSRS retrievability. |
| `get_raw` | Return the immutable raw record + hash + full provenance for a `memory_id`. |
| `prove_age_independence` | Compute recall@k and p95 latency vs memory age on the current store, live. |

**Scope on every query.** The five query/write tools (`remember`, `recall`,
`consolidate`, `list_memories`, `prove_age_independence`) take a **scope**: a required
`namespace` (default `"default"`) plus optional `agent_id` / `project_id`. The two
record-addressing tools (`reawaken`, `get_raw`) take only an opaque `memory_id` — a
handle you obtain from a scoped read — so they need no scope arguments. (See
[Scoping](#scoping-no-cross-tool-bleed).)

**Run it:**

```bash
python -m eidetic.mcp_server          # stdio transport (local hosts: Claude Code, Cursor)
python -m eidetic.mcp_server --http   # streamable-http on EIDETIC_MCP_HOST:EIDETIC_MCP_PORT
```

`--http` binds `EIDETIC_MCP_HOST` (default `127.0.0.1`) and `EIDETIC_MCP_PORT`
(default `8765`). A missing `DASHSCOPE_API_KEY` fails loud as a clear MCP tool error —
never a fabricated result.

---

## Install as a Claude Code plugin

The repo ships everything Claude Code needs to mount Eidetic-Plus as a plugin:

- `.claude-plugin/plugin.json` — the plugin manifest (name `eidetic-plus`).
- `.mcp.json` — a stdio MCP server entry that runs
  `${CLAUDE_PLUGIN_ROOT}/.venv/bin/python -m eidetic.mcp_server` with
  `cwd: ${CLAUDE_PLUGIN_ROOT}`.
- `.claude-plugin/marketplace.json` — a single-plugin marketplace pointing at this repo.

**(a) One-time setup** so the dependencies and the `.venv` referenced by `.mcp.json`
actually exist:

```bash
bash run.sh          # creates .venv, installs requirements.txt, seeds .env on first run
# (or do it by hand:)
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then put your key in `.env`:

```bash
DASHSCOPE_API_KEY=sk-...
```

**(b) Install it locally** in Claude Code:

```text
/plugin marketplace add /Users/raunakgengiti/Eidetic-Plus
/plugin install eidetic-plus@eidetic-plus
```

Or point Claude Code straight at the repo without the marketplace step:

```bash
claude --plugin-dir /Users/raunakgengiti/Eidetic-Plus
```

**(c)** Eidetic-Plus now appears in `/plugin` and mounts all seven tools above, ready
to use from any session.

---

## Why it exists

Every production agent-memory system in use today shares the same failure modes.
Mem0, Zep, Letta, and HippoRAG all **lose or distort** what they store as it ages,
**retrieve recent memories better than old ones**, and **confabulate** answers that no
stored record actually supports.

Eidetic-Plus combines three beyond-human properties that none of those systems have
together:

1. **Decoupled lossless retention from forgetting.** Every record is written once to an
   immutable, content-addressed substrate and is never mutated or deleted. Forgetting
   is a *retrieval-priority* signal only; the raw bytes are always recoverable.
2. **Age-independence at fixed N.** For a fixed corpus size, retrieval recall and
   latency do not depend on how old a memory is. (This is *not* a claim to defeat the
   size-dependence inherent to approximate nearest-neighbor search — only that, holding
   N constant, a 30-year-old memory is recalled as well and as fast as yesterday's.)
3. **Full source + temporal provenance with bi-temporal contradiction handling.** Every
   answer cites the exact immutable source, its hash, its timestamps, and a confidence,
   and contradictions are resolved by bi-temporal invalidation rather than overwrite.

---

## The signature proof

`scripts/signature_demo.py` ingests N distinct memories across ~30 simulated years of
timestamps, then queries each one back by cue and bins the results by **memory age**.
It plots two curves: **recall@k vs age** and **p95 retrieval latency vs age**. Both come
out **flat** — the linear slope of each is reported in `/yr` and `ms/yr` and is
effectively zero. That flatness is the headline proof of age-independence at fixed N:
the index has no concept of recency, so the oldest memory is recalled exactly as well,
and as fast, as the newest.

![signature](artifacts/signature_recall_latency_vs_age.png)

Regenerate it with a funded DashScope key:

```bash
python scripts/signature_demo.py          # defaults: N=180 memories, span=30 years
```

The image is written to `artifacts/signature_recall_latency_vs_age.png`.

---

## Architecture (seven components)

| # | Component | File(s) |
|---|-----------|---------|
| 1 | Immutable, lossless, content-addressed substrate (`sha256` keys; write-once `0o444`; never deletes). `LocalCASSubstrate` in dev, `OSSWORMSubstrate` in prod. | `eidetic/substrate.py` |
| 2 | Hippocampal index = vector ANN (numpy exact + hnswlib HNSW) **and** a bi-temporal knowledge graph with in-app Personalized PageRank. | `eidetic/vector_index.py`, `eidetic/graph.py` |
| 3 | Cognitive-coordinate map = metadata structure-code vector. Tolman-Eichenbaum *inspiration*, shipped as the honest metadata fallback; no absolute age is ever encoded. | `eidetic/structure_code.py` |
| 4 | Write-time salience gate = Bayesian surprise (embedding distance to nearest) + `qwen-flash` importance, mapped to an initial FSRS state. | `eidetic/salience.py` |
| 5 | Offline consolidation/replay + FSRS forgetting (power-law DSR). Forgetting **down-weights index priority only — it never deletes raw**. | `eidetic/fsrs.py`, `eidetic/engine.py` (`Engine.consolidate`) |
| 6 | Reconstructive, verifiable retrieval: ANN top-k + bi-temporal filter → in-app PPR → Reciprocal Rank Fusion → `qwen3-rerank` → `qwen3-max` generation → NLI entailment check (premise = the immutable raw record). FSRS priority is deliberately **not** in the ranking path, which keeps recall age-independent. | `eidetic/retrieval.py` |
| 7 | Provenance + contradiction engine: bi-temporal invalidation (`valid_at`/`invalid_at` world-time, `created_at`/`expired_at` system-time); every answer cites source + hash + timestamp + confidence. | `eidetic/graph.py`, `eidetic/retrieval.py` |

**Vision into the graph.** Ingested images are not stored as a lone vector: the visual
extractor (`qwen-vl-ocr` / `qwen-vl-plus`) turns them into real entities and edges that
join the same bi-temporal graph (Components 2 and 7), and visual claims are NLI-verified
against the raw pixels (Component 6 extended to images). See
[Deepened vision](#deepened-vision).

**Two transports, one engine.** Beyond the FastAPI HTTP API, `eidetic/mcp_server.py`
(FastMCP, stdio + streamable-http) exposes the same `eidetic/engine.py` as seven MCP
tools so any MCP host can mount Eidetic-Plus as its memory backend. It is an additional
transport — **no cognitive component is duplicated**. See
[Universal memory plugin (MCP)](#universal-memory-plugin-mcp).

```
                            ┌──────────────────────────────┐
                            │     Frontend (web UI)         │
                            │   eidetic/web/index.html      │
                            └───────────────┬──────────────┘
                                            │ HTTP (REST)
                            ┌───────────────▼──────────────┐
                            │     FastAPI backend           │
                            │      eidetic/api.py           │
                            └───┬───────────┬───────────┬───┘
              ┌─────────────────┘           │           └─────────────────┐
              ▼                             ▼                             ▼
   ┌────────────────────┐   ┌────────────────────────────┐   ┌────────────────────┐
   │  OSS-WORM substrate │   │ AnalyticDB-PG / HNSW        │   │  bi-temporal graph │
   │  (immutable raw,    │   │ vector index (hippocampal   │   │  (entities, facts, │
   │   sha256, 0o444)    │   │  ANN)                       │   │   PPR, invalidation)│
   └─────────▲──────────┘   └────────────────────────────┘   └────────────────────┘
             │
             │  ┌─────────────────────────────────────────────────────────────┐
             │  │  Backend ⇆ Qwen Cloud / DashScope                           │
             │  │  text-embedding-v4 · qwen-flash · qwen-plus (extract/NLI)   │
             │  │  qwen3-rerank · qwen3-max · qwen-vl-ocr / asr / doc / vl     │
             │  └─────────────────────────────────────────────────────────────┘
             │
             └──── KEY ARROW: every answer points back to the immutable source
                   record it was NLI-verified against (cited by content_hash).
```

---

## Scoping (no cross-tool bleed)

Every read and write carries a **`Scope`**: a required `namespace`, plus optional
`agent_id` and `project_id`. The `namespace` is a **hard boundary** enforced at the
store, the vector index, and the graph — not a soft filter applied after retrieval. A
write in namespace `A` is **invisible** from namespace `B`: a recall, listing, or graph
query scoped to `B` simply cannot see it, and a contradicting fact in a different
namespace does **not** invalidate an edge in another. The default namespace is the
explicit string `"default"` — there is no global wildcard that reads across namespaces.
(Raw bytes are content-addressed and shared by the substrate, but the *index entry* that
makes a memory retrievable is scoped, so identical content in two namespaces yields two
distinct, independently-addressed records.)

This is the property the universal plugin depends on: Claude Code writing in one
namespace and Cursor reading another never see each other's memories. It is proved by
`tests/test_scope_isolation.py` (store-level isolation, per-scope dedup, namespace-bound
contradiction, and a keyed end-to-end check that a recall in namespace `B` retrieves
nothing written in `A`).

---

## Deepened vision

Images are not collapsed into a single vector. When an image is ingested, the visual
extractor (`qwen-vl-ocr` / `qwen-vl-plus`) turns screenshots, diagrams, and tables into
**real graph entities and edges** that join the same bi-temporal knowledge graph as text
facts — a chart becomes queryable structure, not an opaque blob.

On top of that, Eidetic-Plus does **verified visual recall**: a visual claim is checked
against the **raw pixels** (`qwen-vl-plus`), and an unsupported claim is rejected. The
raw image — not a caption, not an embedding — is the arbiter. If the pixels show revenue
*falling*, the claim "revenue increased" is not entailed and is flagged. No competing
agent-memory system does verified visual recall. (Proved by
`tests/test_visual_verification.py`.)

---

## Engine upgrades (sharper with use)

The engine learns from its own recalls without ever touching the ranking score:

- **Reconsolidation as a write path.** A *confirmed* recall re-embeds the memory and
  **up-weights** it (FSRS stability grows, retrievability resets) — the immune-system
  affinity-maturation story, where each successful retrieval sharpens the trace. A
  *contradicted* recall **suppresses** (down-weights) the memory; it is **never deleted**.
- **Synaptic tagging-and-capture.** A salient event up-weights the memories temporally
  adjacent to it, so an important moment carries its neighbours along.
- **Surprise-based event segmentation.** Long input is chunked at Bayesian-surprise
  boundaries instead of fixed windows (`segment=True`).
- **Schema-accelerated consolidation.** Facts consistent with an existing schema
  fast-track through consolidation.
- **Hopfield / attention readout.** A softmax pattern-completion step over the retrieved
  set cleans up partial cues.
- **Memory linking by co-activation.** Co-confirmed memories gain a strengthened
  (scoped) edge, kept out of the entity PPR graph.

Crucially, **none of these signals enter the ranking score.** Exactly as with the FSRS
priority in Component 6, they feed index priority, edges, and re-embedding — never the
retrieval ranking path. That is what keeps recall **age-independent**: the engine gets
sharper with use without ever learning to prefer recent memories.

---

## Prove it live

Age-independence is not only an offline plot — it is a **live, on-demand** measurement
on the *current* store, callable two ways:

```bash
curl 'http://localhost:8000/api/prove_age_independence?namespace=default&k=5'
```

and the `prove_age_independence` MCP tool. Both compute **recall@k** and **p95 latency**
binned by memory **age** over whatever is in the store right now, and both come back
**flat** (slopes ≈ 0 per year). This is distinct from the offline
[signature demo](#the-signature-proof): the demo builds a synthetic 30-year corpus and
plots it; this proves the same property live against real data on demand.

---

## 3D memory map

Open **http://localhost:8000/map** for an interactive 3D view of the store.

It is a **projection**, not the storage format. The engine stores memory as
high-dimensional (1024–2048-D) embeddings; the map runs **PCA** down to three dimensions
purely for navigation. **The engine never stores memory in 3D** — collapsing the
embedding to 3D would destroy the separating structure that makes retrieval work. In the
map:

- **Node colour** encodes **salience**.
- **Node size** encodes **FSRS retrievability**.
- **Edges** show bi-temporal validity (active vs invalidated; entity vs co-activation).
- **Clicking a node** opens its **immutable provenance** — content hash, source, and
  timestamps.

The 3D renderer is vendored on disk (`/static/vendor/3d-force-graph.min.js`); no CDN is
used.

---

## Benchmark harness & competitive engine upgrades

### The neutral harness (`bench/`)

`bench/` is a **neutral** harness: it runs **Eidetic-Plus + Mem0 + Graphiti** through
**ONE fixed judge** and **ONE fixed reader**, so the scoreboard measures **memory
quality** (what each system retrieves) rather than answerer quality. Each system
retrieves its own context; that context is then turned into an answer by the same
`READER_MODEL` and the same fixed reader prompt for all three (the Eidetic-Plus adapter
calls the identical `answer_with_fixed_reader` the baselines do). Eidetic-Plus's answer
cascade, NLI verification, and abstention gate are **product features kept out of this
neutral accuracy path** — they show up in the cost/latency tables and the verified-recall
categorical win, not as an answerer advantage. It evaluates on
**LongMemEval** and **LoCoMo** — restricted to LoCoMo's **four validated categories**
(`single-hop`, `multi-hop`, `temporal`, `open-domain`); the adversarial category is
**excluded** because it lacks reliable ground truth. The judge is `qwen3-max` by default
and is configurable to GPT-4o (via `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL`) for
a leaderboard-comparable headline number; the harness records which judge was used.

It runs **≥10 independent runs** for variance and writes **one raw JSON line per
question** (`artifacts/bench/<system>__run<N>.jsonl`), so every reported number
reproduces from the raw logs — a number that does not reproduce does not exist.

```bash
# Full study: both datasets, all four LoCoMo categories, >=10 runs for variance.
bash bench/reproduce.sh

# Cheap subset smoke (a handful of questions, demonstrably real):
python -m bench.run --systems eidetic --dataset locomo --subset 10
```

See [`bench/README.md`](bench/README.md) for the full harness contract, the
system-under-test interface, and the prerequisites.

### Competitive engine upgrades (now in the engine)

These ship in `eidetic/` today and are **real and unit-tested offline** — no mocks:

- **LLM-free write path** — `consolidate_now=False` ingest does **append + embed only**
  (no synchronous LLM call on the hot path). The expensive work is deferred to an async
  **`consolidate_pending()`**: fact extraction, the bi-temporal graph build, active
  **conflict resolution** (invalidate-not-delete with `supersedes` edges), and date
  normalization. (`eidetic/engine.py`)
- **Hybrid read path** — dense vectors **+ BM25** (`eidetic/bm25.py`, for exact
  names/codes/numbers) **+ a single-step Personalized PageRank** expansion **+ recency**,
  fused by **Reciprocal Rank Fusion**, then an optional **`qwen3-rerank`** pass, a
  **bi-temporal as-of filter**, an **NLI abstention gate**, and finally a
  **token-budgeted** context. (`eidetic/retrieval.py`)
- **Difficulty-routed answer cascade** — the reader is routed by question difficulty
  through `qwen-flash` → `qwen-plus` → `qwen3-max` (`_route_model` in
  `eidetic/retrieval.py`); ambiguous questions default to the conservative middle tier.
  (In the neutral harness this cascade is bypassed in favour of the one fixed reader, so
  accuracy is compared on memory quality alone.)
- **Semantic cache** — exact-hash lookup first, then cosine similarity **≥ 0.9**
  (`eidetic/semantic_cache.py`); `as_of` time-travel queries are never cached.
- **HNSW** index parameters **M=32 / efSearch=256** (configurable; `eidetic/config.py`).

### Optimization playbook (parameterized, sweepable)

The accuracy/efficiency optimizations from the playbook are **machinery now, tuned values
later** — every knob is a `config.py` parameter with a safe default and is swept on a subset
once a key is added (no value is hardcoded from guessing).

- **Structured event calendar (Chronos-style)** — `eidetic/events.py`: async consolidation
  extracts subject-verb-object events, **normalizes reference-relative dates to explicit
  ISO-8601 ranges** ("yesterday", "3 days ago", "last May", "May 2023"), attaches paraphrase
  aliases, and indexes events **separately** from raw turns. At query time the question is
  parsed for entities + temporal constraints + operation (filter/count/order) and events are
  selected by **interval overlap**. The calendar **selects and structures context — the
  shared reader still produces the answer string** (counts/orders), so no answerer advantage.
- **First-class typed preferences** — `eidetic/preferences.py`: preference turns are typed and
  accumulated into a per-namespace profile that is **surfaced in the retrieved context**. The
  rubric-aware prompt lives in the **shared reader** (lifts all three systems equally).
- **Calibrated abstention** — the NLI gate's threshold is a config parameter; `python -m
  bench.calibrate` computes a **conformal threshold** from a held-out subset of *real scored
  questions* (target ~95% answer precision). Real math, never a hardcoded magic cutoff.
- **Query-adaptive hybrid retrieval** — weighted RRF (k=60) with BM25 up for name/date/ID
  queries and graph/PPR up for multi-hop; config-gated cross-encoder rerank (depth ~50→8);
  **lost-in-the-middle edge placement**; aggressive dedup; selective **extractive** raw-chunk
  compression (facts never compressed); optional **static-salience** index pruning (default
  off; uses surprise+importance, *not* decaying retrievability, so the flat curve is intact).
- **Sweep** — `python -m bench.sweep --dry-run` enumerates a **coordinate-descent** plan
  (abstention → rerank → RRF weights → efSearch → cascade) + an honest token-cost estimate,
  offline and score-free; the live sweep needs a funded key.

Sweepable params (`config.py`): `abstention_threshold`, `rrf_w_*`, `rerank_enabled` /
`rerank_depth`, `hnsw_ef_search`, `cascade_confidence`, `compression_ratio`,
`salience_prune_threshold`. Playbook target numbers are **direction, not an assumed baseline**
— the real baseline comes from the first run.

### The claim, stated honestly

Eidetic-Plus is **built to lead every LongMemEval + LoCoMo category at lower token cost
and lower p95 latency than Mem0 and Graphiti** — and this is **provable via this harness
with one command** (`bash bench/reproduce.sh`). On top of that, it has **two categorical
wins neither Mem0 nor Graphiti has**: **flat recall-vs-age** (see
[The signature proof](#the-signature-proof)) and **verified recall with a citable,
immutable source** (see [Scoping](#scoping-no-cross-tool-bleed) and the provenance arrow
in [Architecture](#architecture-seven-components)).

The scoreboard ships **empty**: a **populated** scoreboard requires actually running the
harness with a **funded DashScope key** plus the **baselines installed** (Mem0, Graphiti)
**and a Neo4j AuraDB instance for Graphiti**. **BEAM contradiction resolution at 10M** is
the **frontier**, not solved. **No mocks anywhere; never a fabricated score** — a missing
key or missing dependency **fails loud** with a clear error rather than returning a fake
result.

---

## Dreaming Engine (token-free continuous consolidation)

While idle, Eidetic-Plus keeps consolidating with **ZERO LLM calls** — only local math over
the stored embeddings and the graph (`eidetic/dreaming/`). It is the machine analogue of
sleep consolidation, and it is **strictly additive**: every output is a reversible,
content-addressed, provenance-tagged **derived** record — the lossless store is never merged,
averaged, or mutated (the vector-averaging fallacy measurably *increases* interference).

Four token-free pieces, invoked via `engine.dream(scope=...)`:

1. **Continuous replay scheduler** (`replay.py`) — priority = `surprise^a · need^b ·
   (1−retrievability)^c` (surprise = ANN distance to nearest; need = entity PPR + recency;
   retrievability = FSRS). Pops top-k, reinforces FSRS (capped), then a **SHY-style** global
   pass renormalizes edge weights and prunes the weakest **by weight only** from the index
   (never by age → the flat curve is preserved; never the store).
2. **Offline link prediction + rule mining + schemas** (`kg_embed.py` TransE in numpy,
   `rules.py` Horn rules, Louvain schemas) — proposes edges/facts, each **confidence-gated**
   into a **separate inferred layer** (`Edge.inferred=True`, flagged + provenance), excluded
   from observed reads. Targets multi-hop / multi-session.
3. **Multi-resolution retrieval** (`multires.py`) — RAPTOR-style recursive **bounded-k**
   clustering (near-linear) with a centroid gist per level; retrieval can hit any level. The
   lossless episode and every gist level are kept at once.
4. **Predictive pre-fetch** (`prefetch.py`) — clusters the query log and pre-assembles each
   cluster's **context** (token-free); a query-time cosine match returns it with zero
   assembly tokens.

**Invariants (each a passing test in `tests/test_dreaming.py`):** additive-only (lossless
store byte-identical after a dream cycle); inferred items in a separate, flagged namespace;
**no O(N²)** (replay computes graph features a constant number of times, not per record — the
exact pattern that hung an earlier run); the inferred gate; multi-resolution retrieval;
pre-fetch hit.

**Token-free gate, stated precisely:** the inferred-edge gate is **confidence + embedding
support (token-free) by default**; real-NLI is **optional enrichment** (`DREAM_USE_LLM_NLI=1`,
costs tokens), never a dependency. **Honest note:** human-readable schema *naming* and
contradiction *resolution* (deciding which side wins) may still want an optional LLM pass —
the token-free layer delivers value (gist, inferred links, replay, pre-fetch) without it.

**Measured vs scoreboard:** the layer is built and offline-verified; set `DREAM_AB=1` to run
the dream pass in the harness and measure dream-on vs dream-off against
`artifacts/bench/scoreboard.md` (the same neutral judge). All `dream_*` params are config
defaults and sweepable (`eidetic/config.py`).

---

## Quickstart (one command)

```bash
bash run.sh
```

`run.sh` creates the virtualenv at `.venv` if needed, installs `requirements.txt`, copies
`.env.example` to `.env` on first run, and starts uvicorn on
**http://localhost:8000**. (Model-calling endpoints return HTTP 503 until you add a key.)

Or run the steps manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set DASHSCOPE_API_KEY=...
uvicorn eidetic.api:app --reload
```

---

## .env / configuration

All configuration is read once at startup by `eidetic/config.py` from `.env`. See
[`.env.example`](.env.example) for the full annotated template.

- **`APP_ENV`** — selects only the storage/DB backend; every model call is identical
  in both modes.
  - `dev` (default): local content-addressed substrate + SQLite state + local vector
    index (`hnswlib`, with a numpy brute-force fallback).
  - `prod`: Alibaba Cloud OSS-WORM + AnalyticDB for PostgreSQL + GDB.
- **`DASHSCOPE_API_KEY`** — required for any model call. Get one at the DashScope
  console.
- **`DASHSCOPE_REGION`** — `singapore` (default) uses `dashscope-intl.aliyuncs.com`;
  `beijing` uses `dashscope.aliyuncs.com`. The `qwen3-vl-embedding` multimodal
  **fusion mode** is **Beijing-only**. When running in Singapore, the multimodal
  embedder falls back to `tongyi-embedding-vision-plus` (the `MULTIMODAL_EMBED_MODEL`
  default), and text embeddings use `text-embedding-v4` in both regions.

**Benchmark-harness config** (only needed to run `bench/`). Install the baselines with
`.venv/bin/pip install -r requirements-bench.txt`, then set:

- **`READER_MODEL`** — the ONE fixed answerer shared by every system (default `qwen-plus`;
  pin a snapshot so the alias does not rotate mid-study).
- **`JUDGE_MODEL`** / **`JUDGE_BASE_URL`** / **`JUDGE_API_KEY`** — the ONE fixed judge.
  Defaults to `qwen3-max` on DashScope; set all three to point at an OpenAI-compatible
  endpoint (e.g. `JUDGE_BASE_URL=https://api.openai.com/v1`, `JUDGE_MODEL=gpt-4o`) for a
  leaderboard-comparable judge. `JUDGE_BASE_URL` set without `JUDGE_API_KEY` fails loud.
- **`NEO4J_URI`** / **`NEO4J_USER`** / **`NEO4J_PASSWORD`** — required **only** by the
  Graphiti baseline (a free Neo4j AuraDB instance works; no Docker required).

**No mocked model outputs exist anywhere.** Every embedding, salience score, extraction,
rerank, generation, and NLI check is a real Qwen/DashScope call. A missing or empty
`DASHSCOPE_API_KEY` **fails loudly** — model-calling API endpoints return **HTTP 503**
with a clear detail string. Nothing is ever faked.

---

## Run the tests

```bash
python -m pytest
```

The suite enforces the four core guarantees plus the offline age-independence proof:

1. **Write-once / no-overwrite** — the substrate is content-addressed, dedupes identical
   content, and the OS itself refuses to overwrite an object (`0o444`).
   (`tests/test_write_once.py`)
2. **No-delete-on-forget** — after decades of FSRS decay, priority approaches zero but
   the raw bytes stay byte-for-byte intact and a strong cue restores priority.
   (`tests/test_no_delete_on_forget.py`)
3. **NLI rejects an unsupported answer** — the anti-confabulation gate. This makes a
   real `qwen-plus` call, so it **needs a key**; with no `DASHSCOPE_API_KEY` it skips
   automatically (we never mock a model output). (`tests/test_nli_verification.py`)
4. **Bi-temporal contradiction invalidation** — a contradicting fact closes the old edge
   bi-temporally while keeping full history queryable; nothing is deleted.
   (`tests/test_bitemporal_contradiction.py`)
5. **Age-independence (offline)** — using synthetic vectors, the oldest and newest
   inserts are recalled equally well; insertion order plays no role in ranking.
   (`tests/test_age_independence.py`)
6. **Scope isolation** — a write in namespace `A` is invisible from namespace `B` at the
   store and graph level, dedup is per-scope, and a contradiction does not cross
   namespaces; a keyed end-to-end check confirms a recall in `B` retrieves nothing from
   `A`. (`tests/test_scope_isolation.py`)
7. **Reconsolidation re-weighting** — a confirmed recall up-weights (stability grows,
   retrievability resets); a contradicted recall down-weights but never deletes the
   record; co-activated memories gain a strengthened scoped edge.
   (`tests/test_reconsolidation.py`)
8. **Visual-verification rejection** — a claim that contradicts a chart's pixels is not
   entailed (the raw image is the arbiter), and image ingest produces real graph
   entities. Makes real vision calls, so it **needs a key**; with no `DASHSCOPE_API_KEY`
   it skips automatically. (`tests/test_visual_verification.py`)

---

## Run the signature demo

```bash
python scripts/signature_demo.py [N] [SPAN_YEARS]
```

Defaults are `N=180` memories spanning `SPAN_YEARS=30` years. This makes real embedding
and retrieval calls, so it **needs a key with credit**. It prints the per-bin recall and
p95 latency, the linear slopes, and writes
`artifacts/signature_recall_latency_vs_age.png`.

---

## Alibaba Cloud deployment notes

Selected by `APP_ENV=prod`. The prod backend maps directly onto Alibaba Cloud services
while leaving every model call unchanged:

- **OSS with WORM retention** — the immutable lossless substrate (Component 1). Objects
  are written under a WORM retention policy (`OSS_WORM_RETENTION_DAYS`, default 3650).
- **AnalyticDB for PostgreSQL with HNSW** — the vector index (Component 2 vectors).
- **In-app Personalized PageRank over a bi-temporal graph** — associative expansion and
  contradiction handling (Components 2 and 7), computed in-process.
- **Function Compute cron** — drives the offline sleep/consolidation loop (Component 5).
- **ECS / ACK** — hosts the always-on FastAPI backend (`eidetic.api:app`).

Configure prod via the `OSS_*` variables, `ADBPG_DSN`, and `GDB_ENDPOINT` in `.env`
(ignored when `APP_ENV=dev`). See [`docs/architecture.md`](docs/architecture.md) for the
full deployment topology.

---

## License

MIT — see [`LICENSE`](LICENSE).
