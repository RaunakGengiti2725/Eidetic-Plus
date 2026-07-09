# Benchmark-Comparator Integration Feasibility Report

Four candidate memory systems, assessed for drop-in integration behind the bench's neutral fixed-reader (qwen-plus) contract. Ranked by `integration_effort` (low first). All four have `claims_hold=true`, so no system is demoted for false claims — but **Chronos is demoted to last** under the second criterion: it does not clearly exist *as integrable software* (it's a published method with no code).

**Tiebreaker note:** three of the four tie at `integration_effort: medium`. I break the tie by (a) language match to the Python bench, (b) license openness, and (c) how cleanly retrieval separates from generation (the property that lets it route through the shared reader without breaking neutrality). That yields Hindsight > Mastra > ByteRover.

---

## 1. Hindsight (vectorize-io/hindsight) — effort: MEDIUM

- **What it is:** Purpose-built long-horizon agent memory system by Vectorize.io (retain / recall / reflect over 4 memory networks), backed by PostgreSQL/pgvector; ACL 2026 demo (arXiv 2512.12818).
- **Exists? / OSS / License / Key:** Exists as a real memory agent ✅ · OSS ✅ · **MIT** · key **optional** (keyless READ; WRITE needs an LLM provider, which can be local ollama/lmstudio → no *paid* key required).
- **Adapter fit:** Near-ideal — the cleanest of the four because `recall` is documented as pure parallel retrieval + rerank with no generation. `bank_id = namespace`; `retain` per turn → `WriteResult`; `recall(query, query_timestamp=as_of)` → concatenate `results[].text` → feed the **shared qwen-plus reader** (NOT `reflect()`, which is its own LLM answerer and would break neutrality). Bi-temporal `as_of` maps directly to `query_timestamp`. Python-native (`pip install hindsight-all`) — no cross-language bridge.
- **Blockers:** Server + embedded PostgreSQL/pgvector bring-up (heavier than a plain vector store; adapter code itself is light). WRITE requires a configured LLM provider (fails loud without one). **Keyless-recall is PENDING source verification** — it came from a docs summary, not code; it is the single fact carrying both `key=optional` and the recall→reader neutrality mapping, so confirm it by running `recall` with no provider key before trusting it. Intel Macs need `hindsight-all-slim`; confirm Python ≥3.11 for the embedded path.
- **Final confidence:** **High** (verdict upheld across repo + arXiv + ACL demo + docs; the keyless-read caveat is the one open item).

## 2. Mastra (@mastra/memory) — effort: MEDIUM

- **What it is:** A real OSS TypeScript/Node AI-agent framework (YC-backed); `@mastra/memory` is a first-class, separately-installable memory module with message history, semantic recall, and LLM-synthesized Observational/Working memory.
- **Exists? / OSS / License / Key:** Exists as a real memory component ✅ · OSS ✅ · **Apache-2.0** (core + memory module; only `ee/` auth dirs are separately licensed) · key **optional** for the RAG path, **required** for the flagship long-horizon path.
- **Adapter fit:** Two paths with opposite implications. **(A)** `recall({threadId, vectorSearchString})` with LibSQL (local file) + fastembed (local embeddings) is **truly zero-key on both read and write** and routes cleanly to the shared reader — but it is essentially vector RAG over raw turns, i.e. a near-duplicate of the bench's existing `rag-vector` row. **(B)** Observational/Working memory — its actual long-horizon differentiator — is LLM-synthesized *upstream* of retrieval: needs a paid LLM key **and** injects synthesized memory before any reader, breaking single-fixed-reader neutrality. Requires a Node sidecar (HTTP/stdio) + Python adapter; no Python package.
- **Blockers:** Language mismatch (Node-only; PyPI has no `mastra`/`mastra-memory` package — confirmed). Neutrality-vs-capability tension: the safe path duplicates `rag-vector`; the differentiated path can't be fairly benchmarked through the shared reader. `as_of` fidelity is conditional (temporal filter is on message *creation* date via `filter.dateRange`; bi-temporal answering works only if creation date can be set to event/session time at write). Standalone write-method name (`saveMessages` vs `saveMessageToMemory`) unconfirmed — minor.
- **Final confidence:** **High** (all central claims verified directly against npm registry + docs; no conflation).

## 3. ByteRover (Cipher / byterover-cli) — effort: MEDIUM

- **What it is:** A real, actively-developed "memory layer for AI coding agents"; open engine began as Cipher (`@byterover/cipher`, vector store + MCP) and evolved into `byterover-cli` (`brv`, a file-based markdown context tree). Vendor-reported LoCoMo 96.1% / LongMemEval-S 92.8%.
- **Exists? / OSS / License / Key:** Exists as a real memory agent ✅ · OSS **partial** — **Elastic License 2.0** (source-available, *not* OSI-approved; fine for internal benchmarking, but forbids offering as a managed service) · key **optional** (embedding provider required: paid OpenAI/Gemini/Qwen by default, or fully local via Ollama → no key).
- **Adapter fit:** Route via the **deprecated** `@byterover/cipher` 0.3.0 in **aggregator mode** (the default generative `ask_cipher` must NOT be used — it breaks reader neutrality). `cipher_extract_and_operate_memory` writes turns (runs an LLM extraction step → distilled, not verbatim; same shape as mem0). `cipher_memory_search` returns raw stored entries → feed the shared qwen-plus reader. Node/TS → run as MCP/HTTP subprocess driven from the Python adapter.
- **Blockers:** The clean retrieval-only path exists **only in the deprecated Cipher 0.3.0**; the current `brv query` is generation-oriented and **unconfirmed as retrieval-only** — so pin to 0.3.0 or reverse-engineer `brv`. No native namespace-isolation primitive: must provision a separate vector collection + DB per namespace and verify no cross-namespace leakage. Embedding provider required (fail loud if neither paid key nor local Ollama configured). Node/Python bridge. Elastic-2.0 redistribution caveat. Headline benchmark numbers are vendor-reported, not independently verified.
- **Final confidence:** **High** (finding was `medium`; verdict raised it — Elastic-2.0, the Cipher→byterover-cli evolution, tool names, and benchmark quotes all confirmed against LICENSE/README/registry).

## 4. Chronos (Temporal-Aware Conversational Agents, PwC US — arXiv 2603.16862) — effort: INFEASIBLE (demoted)

- **What it is:** A temporal-aware long-term conversational memory *method* from PwC US: SVO event tuples with ISO-8601 datetime ranges indexed in dual "event/turn calendars," answered via dense retrieval + Cohere rerank + dynamic prompting + a ReAct loop. Reports **SOTA 95.60% on LongMemEval-S**.
- **Exists? / OSS / License / Key:** **Method: yes / integrable software: no.** No repo, no pip/npm/docker package, no Papers-With-Code link — it exists only as a CC BY 4.0 paper (no software license because no code was released). Key **yes** — and not one key: a faithful reimplementation needs paid keys from **four** vendors (OpenAI + Anthropic + Google + Cohere).
- **Adapter fit:** Conceptually the write path maps to the ABC, but the answer path **is an agentic ReAct reader driven by a strong commercial LLM** — it cannot be routed through the shared qwen-plus reader without dismantling the method. At best it's a "different-reader product row" (like this repo's NotebookLM bridge), never an apples-to-apples fixed-reader comparator.
- **Blockers (dispositive):** No code → integration = full from-scratch reimplementation of a 4-stage pipeline (violates the spirit of a real drop-in comparator). Four paid providers. Category conflict: agentic reader can't be neutralized. Nondeterministic ReAct loop makes token/latency accounting noisy.
- **Final confidence:** **High** — high confidence that it's real, correctly disambiguated (distinct from Amazon Chronos time-series and the unrelated GitHub "Chronos" projects), and **high confidence that it is not integrable**. (Context: a newer project reportedly claims 96.2%, so Chronos may no longer even be leaderboard SOTA — but the finding only claims the paper *reports* 95.6%, which holds.)

---

## RECOMMENDATION

### Can be integrated key-free / cheaply this session
Honestly: **none is a trivial same-session drop-in**, and "key-free" is asymmetric across the three viable ones — don't treat them as equivalent.

- **Mastra local-RAG path** — the *only* genuinely zero-key option (LibSQL local file + fastembed local ONNX, **no LLM on read or write**). But it needs a Node sidecar + Python adapter, and the resulting row **duplicates the existing `rag-vector` baseline**. You'd add a named row that measures nothing new.
- **Hindsight** — zero-key on **read**, but **write (retain) needs an LLM** for fact extraction. Local ollama avoids a *paid* key, but that's real infra (local LLM + Hindsight server + Postgres/pgvector), not "zero-key." Python-native and the cleanest adapter — this is the best *differentiated-yet-neutral* candidate, contingent on confirming keyless recall against source.
- **ByteRover/Cipher** — key-free only via local Ollama, on a **deprecated package pin (0.3.0)**, with a Node bridge and hand-built namespace isolation. Most fragile of the three.

### Need paid keys or are infeasible
- **Chronos — infeasible.** No code to wrap; a reimplementation needs four paid providers and produces a non-neutral agentic reader. Skip it as a comparator.
- **The differentiated long-horizon paths** of Mastra (Observational/Working memory), ByteRover (LLM extraction on write), and Hindsight (`reflect`) all either require a paid LLM key or synthesize memory upstream of the reader — which **breaks the fixed-reader neutrality guarantee**. You cannot fairly measure what makes any of these "special" through the shared qwen-plus reader.

### Bottom line: is "named-comparator breadth" for a strongest-in-the-world claim achievable with these four?
**No — not in any honest fixed-reader sense.** You can *add rows*, but the fixed-reader (qwen-plus) contract admits only retrieval-only paths, and that collapses the value:

- Mastra's neutral row **duplicates `rag-vector`** — breadth in name, not in substance.
- ByteRover's neutral row exists only on a **deprecated pin** with unverified current-CLI behavior.
- Hindsight's multi-strategy fusion is the **one differentiated-but-neutral row you can actually get** — and even it rests on an unconfirmed keyless-read assumption.
- **Chronos — the only one of the four that actually reports SOTA (95.6%) — is the one you cannot integrate at all.** The most impressive name on the list is vaporware from an integration standpoint.

And the published headline numbers you'd be implicitly invoking by naming these systems (ByteRover 96.1/92.8, Hindsight ~91, Mastra 94.87%) all come from **their own agentic readers**, not a shared reader — so none is reproducible in-bench. Naming them as comparators while running them as retrieval-only rows would compare something *different* from the number that makes them notable.

**Practical path:** if breadth is the goal this session, wire **Hindsight** (Python-native, MIT, cleanest neutral fit — after confirming keyless recall against source) as the one genuinely additive neutral row; optionally add **Mastra local-RAG** only if a labeled "second RAG implementation" row is worth the Node sidecar. Treat **ByteRover** as a later, lower-priority add (Elastic-2.0 + deprecated pin). Drop **Chronos** entirely, or reserve it — like the NotebookLM bridge — as an explicitly labeled different-reader *product* row, never a fixed-reader comparator. A "strongest-in-the-world" claim is **not** supportable on this comparator set under the neutral contract.
