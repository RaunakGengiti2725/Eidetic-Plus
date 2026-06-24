# EIDETIC-PLUS: Upgrade Specification
### From a Memory App to the Universal Memory Substrate for Every AI Agent

> This is the build brief for taking Eidetic-Plus to its next level. It is written to be read by a coding agent (Claude Code) and by you. Read it alongside two companion files in this repo: `Eidetic-Plus_Master_Dossier.md` (the full research foundation and the source of every claim below) and `docs/architecture.md` (the current 7-component system as built). A condensed `/goal` prompt that triggers this entire spec, kept under 4000 characters, is at the very end in Section 12.
>
> Non-negotiable discipline carried from the dossier, repeated here because every upgrade must obey it: no mocks anywhere (every model call is a real DashScope call, fail loud on a missing key); never delete a raw record (forgetting only down-weights the index); FSRS priority stays OUT of ranking (recall must stay independent of memory age, and the flat curve is the proof); the immutable record is always the final arbiter of truth for verification, text and visual. Build on the existing code, do not rewrite what works, and ask before adding any dependency that is not standard.

---

## 0. THE ONE-SENTENCE GOAL

Turn Eidetic-Plus from a standalone application into a universal memory plugin that any AI tool can mount over a standard protocol, deepen its vision so images become structured graph knowledge that can be verified against pixels, implement the research-backed engine upgrades that make it get sharper with use the way a brain and an immune system do, and add a navigable 3D memory map as a presentation surface, all while keeping the engine high-dimensional, lossless, age-independent, and provably non-confabulating.

---

## 1. WHERE WE ARE NOW (CURRENT SYSTEM, VERIFIED)

The 7-component engine is built, real, and tested. Nothing here is a stub.

1. **Immutable substrate** (`substrate.py`): content-addressed by SHA-256, write-once (objects set to read-only, the OS blocks overwrite), dedup, no delete API (raises on attempt). Dev is a local content-addressed store; prod is OSS-WORM.
2. **Hippocampal index** (`vector_index.py` + `graph.py`): vector ANN (real HNSW via hnswlib, numpy exact fallback) storing a content embedding plus a structure-code vector, plus a bi-temporal knowledge graph in networkx with in-app Personalized PageRank and spreading activation.
3. **Cognitive-coordinate map** (`structure_code.py`): a metadata structure-code vector (entity type, modality, graph PPR and degree, cyclic time). Absolute age is deliberately NOT encoded, because encoding it would slope the flat recall-versus-age curve.
4. **Salience gate** (`salience.py`): Bayesian surprise (embedding distance to the nearest stored memory) plus qwen-flash importance, producing the initial FSRS state.
5. **FSRS forgetting plus consolidation** (`fsrs.py`, `engine.py`): power-law DSR decay sets index priority only, never deletes. Reawakening is an O(1) re-promote. The sleep loop does dedup, then verified semantic summaries, then decay.
6. **Verifiable retrieval** (`retrieval.py`): ANN top-k plus bi-temporal filter, then PPR, then Reciprocal Rank Fusion, then qwen3-rerank, then qwen3-max generation, then an NLI entailment check against the immutable raw record. FSRS priority is kept out of ranking.
7. **Provenance plus contradiction** (`graph.py`, `retrieval.py`): bi-temporal invalidation (a new fact closes the old edge, never deletes, history kept). Every answer cites source, hash, timestamp, and confidence.

Plumbing: a real DashScope client (`dashscope_client.py`, every model call real, fail-loud, no mocks: text-embedding-v4, qwen-flash, qwen-plus, qwen3-rerank, qwen3-max, qwen-vl-ocr, qwen3-asr, qwen-doc, qwen-vl-plus); multimodal ingestion (`ingestion.py`, text/image/pdf/audio/video/binary, SHA-256 dedup before embed, un-embeddable inputs are described then embedded); FastAPI (`api.py`, 11 routes); a self-contained web UI (`web/index.html`); config (`config.py`, `.env`-driven, dev/prod swap is storage-only, Singapore endpoint). Tests pass (write-once, no-delete-on-forget, bi-temporal contradiction, offline age-independence). Docs, README, `.env.example`, `run.sh`, pyproject, MIT license are present.

Known small bug to fix in passing: the web UI fails to load `/api/memories` when opened as a `file://` page because there is no origin to resolve the relative URL against. Serve `index.html` from FastAPI (or inject a configurable API base URL) so the path resolves. This is unrelated to the upgrades but should be fixed.

---

## 2. THE THESIS OF THIS UPGRADE

Two insights drive everything below, and they are independent.

**Distribution insight: a plugin, not an app.** "Works with every AI tool" has a concrete meaning in 2026, and it is the Model Context Protocol (MCP). An MCP server is the closest thing to a universal standard for giving an agent external capabilities. If Eidetic-Plus exposes its operations as MCP tools, then Claude, Claude Code, Cursor, Cline, Zed, and any other MCP host can mount it as their memory backend with zero per-tool integration. The existing FastAPI routes already define the operations, so MCP is a thin transport layer over the same engine, not a rewrite. This is the path to "applicable to every AI software ever." A Claude-Code-specific plugin is then a small manifest wrapping that same MCP server, so building the server first gives both universality and the Claude Code plugin at once.

**Depth insight: deepen the edge, do not widen the system.** The fastest way to make a clean, working system worse is to bolt on capability for its own sake. The way to make it better is to deepen the three things that are already its differentiators: the graph (associative, bi-temporal structure), the verification (no confabulation), and the structure-content factorization (the thing brains do that vector stores do not). Every upgrade in this spec feeds one of those three, or it is cut. A technical judge will ask "what does this feed," and the only acceptable answer is "a node, an edge, a vector, a verification, or the flat curve."

---

## 3. PART 1: THE UNIVERSAL PLUGIN (MCP SERVER)

This is the headline. It is what makes Eidetic-Plus mountable by every AI tool.

### 3.1 Build an MCP server over the existing engine
The MCP server is an additional transport over the same `engine.py`. Keep the FastAPI routes working. Both talk to the same engine. No logic is duplicated. Support both stdio transport (for local hosts like Claude Code and Cursor) and HTTP transport (for remote or shared deployments).

### 3.2 The MCP tools to expose
- `remember`: store a memory (text or a reference to a file already ingested), returns the content hash and provenance.
- `recall`: the full verified retrieval pipeline, returns the answer, the cited immutable sources, hashes, timestamps, and confidence, or an explicit abstention.
- `consolidate`: trigger the sleep loop (dedup, verified semantic summaries, FSRS decay).
- `reawaken`: re-promote a down-weighted memory (the O(1) revert path).
- `list_memories`: list within a scope, with salience and FSRS retrievability.
- `get_raw`: return the immutable raw record plus hash and full provenance chain for a given memory (this is the "show your work" tool that proves no confabulation).
- `prove_age_independence`: compute and return recall@k and p95 latency versus memory age on the current store, on demand. This turns the signature result from a one-off script into a live, callable capability. It must always come back flat.

### 3.3 THE SCOPING DECISION (build this in from the start, it is the one thing not to skip)
Memory crosses sessions and tools, so without explicit scoping you get cross-contamination the first time two tools share the server. When Claude Code writes a memory and Cursor recalls it, they must not bleed across contexts that should stay separate.

- Every tool takes an explicit `scope`, composed of a required `namespace` plus optional `agent_id` and `project_id`.
- Scope is enforced at the index and graph layer: every read filters by scope, every write tags the record with scope. Use the bi-temporal and provenance machinery already in the graph to carry scope.
- The default scope is explicit and named, never global-by-accident.
- Add a test that proves isolation: write a memory in namespace A, then confirm it is invisible from namespace B.

This is the single most important design decision in the upgrade. If it is skipped, the universal plugin leaks memory between unrelated agents.

### 3.4 Package it two ways
- A plain MCP server runnable by any host (stdio and HTTP), with clear install instructions.
- A Claude Code plugin manifest that wraps that MCP server so Eidetic-Plus appears in the `/plugin` menu. The plugin is built on top of the MCP server, not instead of it.

### 3.5 Auth and failure behavior
Key and auth handling stays fail-loud. A missing key returns a clear error or a 503, never a faked result. This honesty is itself a differentiator: the system never pretends to remember.

---

## 4. PART 2: DEEPENED VISION (FEED THE GRAPH AND THE VERIFICATION)

Today images become OCR text plus a flat vector. That is vision sitting beside the system. The upgrade makes vision feed the differentiators.

### 4.1 Structured visual extraction into the graph
An image, screenshot, diagram, or table becomes entities and edges in the bi-temporal graph, not just an embedding.
- Use `qwen-vl-ocr` (38,192-token context, tables rendered to HTML, formulas, bounding-box localization) to turn a table into structured rows and relations, which become graph edges.
- Use `qwen-vl-plus` to turn a scene or diagram into entities and relations. A photo of a whiteboard becomes nodes; a chart becomes structured facts with provenance pointing back to the image hash.

### 4.2 Video into a temporal graph
Video is not stored as one flat vector. Extract keyframes, turn each into entities and edges, and store the temporal sequence as ordered relations in the graph, so the agent can reason about what happened and in what order.

### 4.3 Visual verification (verified visual recall, a genuinely novel guarantee)
Extend the no-confabulation guarantee to images. No competitor does verified visual recall, so this is a real differentiator.
- When an answer makes a claim grounded in an image memory, verify the claim against the actual pixels using `qwen-vl-plus` as the judge (does the image support "the chart shows revenue up"?).
- Unsupported visual claims are rejected, or the system abstains, exactly like the NLI text path. The raw image in the immutable store is the arbiter.

### 4.4 Research backing
The dossier's ingestion section establishes the real, GA model capabilities used here (`qwen-vl-ocr`, `qwen-vl-plus`, the multimodal pipeline). The verifiable-retrieval component (Component 6) establishes the NLI-against-immutable-record pattern; visual verification is that exact pattern with a vision judge and the raw image as ground truth. Be skeptical of adding net-new computer vision models (object detection, segmentation, face recognition) whose outputs do not become a node, an edge, a vector, or a verification. If the output does not feed a differentiator, it is decoration and should be cut.

---

## 5. PART 3: RESEARCH-BACKED ENGINE UPGRADES

Each upgrade below sharpens an existing differentiator and is grounded in the dossier's neuroscience. Implement all of them.

### 5.1 Reconsolidation as a write path (the lead biological story)
Retrieval is no longer read-only. This is the upgrade that makes the system get sharper with use.
- On a CONFIRMED recall (the retrieved memory was used and verified), re-embed and up-weight that memory. This is immune affinity maturation: re-exposure selects and strengthens the highest-affinity clone, so confirmed-useful memories get promoted.
- On a CONTRADICTED recall, invalidate the stale edge (never delete) and suppress it, then write the corrected fact with full bi-temporal history.
- **Research backing.** Memory reconsolidation (dossier Section 6.10): reactivating a stored memory makes it transiently labile, and retrievals accompanied by reconsolidation result in strengthening (this is the testing effect, mechanistically). The 2026 survey "Memory for Autonomous LLM Agents" lists reconsolidation as an unimplemented frontier, so implementing it is a real novelty claim. Immune affinity maturation (Section 7.2) is graded REAL and is the least hand-wavy biological mapping in the whole dossier: re-exposure drives selection for higher-affinity clones, the secondary response is faster and higher quality, and memory is distributed across a population rather than a single point of failure. Lead the pitch with this mapping.

### 5.2 Synaptic tagging and capture (salience spreads to neighbors)
When a high-salience event lands, retroactively upgrade the retention and priority of temporally adjacent memories within a window. A surprising event makes its neighbors stick too.
- **Research backing.** Synaptic tagging and capture (Section 6.7): a weak memory can be made persistent if a salient or novel event occurs within roughly a one-hour window, neurons with higher excitability at encoding are preferentially allocated to an engram, and memories encoded close in time share overlapping engrams (memory linking). The design implication stated in the dossier is exactly this: a mechanism where salience and novelty signals retroactively upgrade the retention and priority of temporally adjacent memories.

### 5.3 Surprise-based event segmentation
Chunk long inputs at Bayesian-surprise boundaries instead of fixed-size windows, so stored episodes align to natural event boundaries. This improves both pattern separation and recall.
- Compute surprise as spikes in embedding distance or next-token negative log-likelihood, and place episode boundaries at the spikes.
- **Research backing.** EM-LLM (Sections 4.2 and 13.5): it segments token streams into events via Bayesian surprise plus graph-theoretic boundary refinement, and reports an overall relative improvement of 4.3 percent over InfLLM with a 33 percent improvement on PassageRetrieval, retrieving across 10M-token contexts. The dossier's salience guidance is to combine surprise (embedding distance to nearest), LLM-judged importance (qwen-flash), and explicit salience, all computable cheaply online.

### 5.4 Schema-accelerated consolidation
Consolidation is not uniform. In the sleep loop, schema-consistent facts fast-track into the semantic store, while novel or schema-inconsistent facts stay episodic longer before abstraction.
- **Research backing.** Tse et al., Science 2007 (Section 6.11): schema-consistent memories became hippocampus-independent in 48 hours, far faster than novel material. The dossier's design implication is an offline consolidation job that abstracts gist into a semantic store while preserving raw episodic traces, with schema-matching as an accelerator.

### 5.5 Age-independence as a live, provable capability
Do not leave the signature result as a one-off script. Expose `prove_age_independence` as both an MCP tool (Section 3.2) and an API route that computes recall@k and p95 latency versus memory age on the current store. The headline claim must be provable on demand, and the curves must always come out flat. Also RUN `signature_demo.py` and save the plot image as a committed artifact.
- **Research backing.** The three beyond-human claims (Section 3.3) and the gap analysis (Section 13.1): retrieval cost is a function of store size and recall target, never of a memory's age, which is the rigorous version of "perfect recall regardless of recency." The contrast is the "lost in the middle" degradation (Liu et al., TACL 2024), where mid-context accuracy can fall below the closed-book baseline. The flat curve refutes exactly that failure. State it precisely as age-independence at fixed store size N, never as defeating the size-dependence of nearest-neighbor search.

---

## 6. PART 4: THE 3D MEMORY MAP (A VIEW, NOT THE STORAGE)

Add a navigable 3D map of memory. Be explicit in code comments and UI copy about what it is and is not.

### 6.1 The honest framing (state it in the UI and to judges)
Memory is stored in high-dimensional space (1024 to 2048 dimension embeddings). 3D is a projection for human navigation, never the representation. Storing memory in 3D would collapse the separating structure and make retrieval worse, so the engine never does that. This sentence is itself a credibility signal, the same way the REAL-versus-METAPHOR table is.

### 6.2 The render spec
- Render the bi-temporal graph in 3D (react-force-graph-3d or three.js) over a projection of the embeddings (UMAP or a 3D force-directed layout).
- Nodes colored by salience, sized by FSRS retrievability, edges showing bi-temporal validity (active versus invalidated, drawn differently).
- Click a node to show the raw immutable record, hash, source, timestamp, and the full provenance chain. Let the user fly through the map. This is the demo surface that makes the system unforgettable, which matters for the 15 percent presentation score.

### 6.3 Research backing
The method of loci (Sections 5.4 and 5.5): Dresler et al. showed that imposing a navigable spatial structure on memory dramatically boosts retrievability, with trained subjects going from recalling 26 of 72 words to 62, and top athletes recalling 70.8 of 72 versus controls' 39.9. The cognitive-coordinate map and grid-cell and Tolman-Eichenbaum work (Sections 6.3, 6.4, 9.3) is the basis for "memory as a navigable place." The 3D map is that idea made literal for a human viewer, while the engine keeps the real high-dimensional structure underneath.

---

## 7. FRONTIER UPGRADES (PUSH TO THE NEXT LEVEL, GRADED REAL vs ASPIRATIONAL)

These go beyond the previous plan. They are the genuine frontier. Each is graded so the builder knows the risk. Implement the REAL ones; attempt the ASPIRATIONAL ones only if the core upgrades land first, and never let them turn into stubs.

### 7.1 Sharpened structure-content factorization (REAL, and this is THE frontier)
This is the single biggest lever toward "beyond human," bigger than any new modality. The thing brains do that a vector store does not is cleanly separate where a memory sits in relational space (its structure code) from what it contains (its content code), so that structure generalizes across different content.
- Make the existing `structure_code.py` factorization crisper: a stronger, more disentangled structure-code vector (relational position, role, graph topology features) that is genuinely independent of the content embedding, so two memories with the same relational structure but different content sit near each other in structure space and far apart in content space.
- Retrieve compositionally over both spaces (find things that fill this role, regardless of surface content; find things like this content, regardless of role).
- **Research backing.** The Tolman-Eichenbaum Machine (Sections 6.4, 9.3, 13.4): medial entorhinal cells form a structural basis, hippocampal cells bind it to sensory content, and separating the two allows generalization over environments sharing the same structure. Honest caveat carried from the dossier: there is no production retrieval system that does literal TEM factorization, so this stays the metadata-structure-code approximation, cited as inspiration, not as a neural implementation. Improving this is the real research contribution.

### 7.2 Associative Hopfield/attention readout (REAL)
Frame the final retrieval readout as a modern-Hopfield or attention step over the retrieved set, giving a near-instant single-step associative completion from a partial cue (pattern completion).
- **Research backing.** Modern Hopfield networks (Section 8.2): continuous Hopfield networks store exponentially many patterns, retrieve in one update step, and the update rule is exactly transformer attention. Sparse Distributed Memory (Section 8.1) is the graceful, content-addressable template. This reframes the readout as biologically and mathematically principled rather than ad hoc.

### 7.3 Memory linking by co-activation (REAL)
When two memories are retrieved and confirmed together, create or strengthen an edge between them in the graph. Over time the graph self-organizes around what is actually used together, which improves multi-hop recall.
- **Research backing.** Memory linking and engram overlap (Sections 6.7, 6.10): memories encoded or reactivated close together share overlapping engrams. This is the structural complement to reconsolidation (5.1).

### 7.4 Predict-then-verify retrieval (ASPIRATIONAL, attempt after the core lands)
Make retrieval constructive: predict the likely answer from the structure and compressed semantic store first, then verify and correct against the retrieved immutable records, closing a predict-calibrate loop. This mirrors how human memory is reconstructive rather than a literal playback, while the immutable record bounds the distortion.
- **Research backing.** Constructive and predictive memory (Section 6.12) and the generative consolidation model of Spens and Burgess (Section 8.6), framed by the dossier as compressive retrieval-augmented generation. Nemori's free-energy predict-calibrate loop is the nearest agent-system precedent. Keep raw traces as the arbiter so the generative step cannot confabulate.

### 7.5 Ordered working buffer (ASPIRATIONAL)
Give the agent a small ordered working-memory buffer with explicit position or slot tags for the active context, rather than a flat similarity set, so recent ordered sequences are preserved exactly while older memory stays in the age-independent store.
- **Research backing.** The theta-gamma phase code (Section 6.5): items are held in an ordered set by phase position, with order encoded explicitly. The dossier's design implication is an ordered, indexed buffer with explicit slot tags.

---

## 8. CONSTRAINTS (CARRY FORWARD, DO NOT VIOLATE)

- No mocked outputs anywhere. Fail loud on a missing key (503 or a clear error), never a faked result.
- Never delete a raw record. Keep the existing no-delete test, and add scope-isolation and reconsolidation tests.
- FSRS priority stays out of ranking. Recall must stay age-independent, and the flat curve is the proof.
- The immutable record is the final arbiter for all verification, text and visual.
- Scope must prevent cross-tool and cross-namespace bleed (test it: write in namespace A, confirm invisible in namespace B).
- Build on the existing files. Do not rewrite what works. Ask before adding any non-standard dependency.

---

## 9. DEFINITION OF DONE (THE OUTPUT)

- An MCP server runnable by any host (stdio and HTTP) AND a Claude Code plugin manifest that mounts it in `/plugin`, with install instructions in the README.
- Scoping implemented and tested: memories isolated per namespace, agent, and project.
- Vision feeds the graph: an image ingest produces real entities and edges, visible in the 3D map. Visual verification rejects an unsupported visual claim.
- All Part 3 upgrades implemented and wired: reconsolidation strengthening on confirmed recall, tagging-and-capture on salient events, surprise-based segmentation, schema-accelerated consolidation, and `prove_age_independence` as a live tool.
- The REAL frontier upgrades (7.1 sharpened factorization, 7.2 Hopfield readout, 7.3 memory linking) implemented. The ASPIRATIONAL ones (7.4, 7.5) implemented only if they do not become stubs; otherwise documented as designed-but-deferred.
- 3D memory map works, is navigable, node-click shows provenance, and is clearly labeled as a projection of high-dimensional memory.
- `signature_demo.py` executed, with the flat recall-versus-age and latency-versus-age plot saved as a committed image.
- Tests pass, including the new ones: scope isolation, reconsolidation re-weighting, visual-verification rejection, and no-delete-on-forget.
- The `file://` UI bug fixed (serve the UI from FastAPI or inject a configurable API base URL).
- README and `docs/architecture.md` updated with the universal-plugin framing, the MCP tool list, the scoping model, the vision-into-graph upgrade, the verified-visual-recall guarantee, the new engine upgrades, and the 3D-as-projection note. MIT license kept.

---

## 10. HONEST RISK NOTES AND BUILD ORDER

This is a large goal. Build in this order so that if effort runs short, what gets cut is presentation, not substance.

1. **MCP server plus scoping (Part 1).** This is the universality and the single most important piece. The scoping decision in 3.3 is the one thing not to skip; without it the plugin leaks memory between unrelated agents the first time two tools share it.
2. **Reconsolidation, tagging, segmentation, schema consolidation, live age-independence (Part 3).** These deepen the differentiators and are mostly modifications to existing files.
3. **Deepened vision (Part 2).** Structured extraction into the graph and visual verification. Real novelty, moderate effort.
4. **Sharpened factorization, Hopfield readout, memory linking (7.1 to 7.3).** The real frontier. 7.1 is the highest-value-highest-risk item in the whole spec.
5. **3D memory map (Part 4).** Pure presentation. Genuinely compelling for the demo, but it does not change what the system is, so it comes last and is the natural cut line if `/goal` stalls.
6. **Predict-then-verify and ordered buffer (7.4, 7.5).** Aspirational. Only if everything above lands cleanly.

A note on framing for the pitch, carried from the dossier: energy and "beyond human" ambition are good for momentum, but the concrete, checkable evidence is what wins the room. Lead with the flat-curve proof and the MCP universality (both demonstrable in one command or one tool call), lead the biology with immune affinity maturation, and preempt skepticism with the REAL-versus-METAPHOR discipline and the honest residual-risk register. Let the ambition ride on top of the concrete, never in place of it.

---

## 11. RESEARCH APPENDIX: THE EVIDENCE BEHIND EVERY UPGRADE

This is the condensed evidence map. Full detail, exact numbers, and sources are in `Eidetic-Plus_Master_Dossier.md` at the sections cited.

| Upgrade | Biological or computational basis | Dossier section | Grade |
|---|---|---|---|
| MCP universal plugin | Distribution layer over the memory substrate (not a memory mechanism) | Thesis, Section 3 | N/A (engineering) |
| Vision into the graph | Real GA multimodal models (qwen-vl-ocr, qwen-vl-plus) feeding the associative graph | 11 | REAL |
| Verified visual recall | NLI-against-immutable-record pattern, extended with a vision judge and raw image as arbiter | 9.6, 13.2 | REAL (novel) |
| Reconsolidation on confirmed recall | Reactivation makes memory labile then strengthens it (testing effect); listed as an open frontier | 6.10 | REAL (frontier) |
| Reinforcement-on-recall, affinity maturation | Immune clonal selection and somatic hypermutation select higher-affinity clones on re-exposure | 7.2 | REAL (lead story) |
| Tagging and capture (salience spreads) | Salient or novel events make temporally adjacent weak memories persist; engram overlap | 6.7 | REAL |
| Surprise-based segmentation | EM-LLM Bayesian-surprise event boundaries, +4.3% over InfLLM, +33% PassageRetrieval | 4.2, 13.5 | REAL |
| Schema-accelerated consolidation | Tse et al. 2007: schema-consistent memories consolidate in 48 hours | 6.11 | REAL |
| Age-independence as a live proof | Retrieval cost depends on N and recall target, never on age; contrast with lost-in-the-middle | 3.3, 13.1 | REAL (headline) |
| 3D map as projection | Method of loci boosts retrievability (Dresler et al.); cognitive-coordinate and grid-cell maps | 5.4, 5.5, 6.3, 6.4 | REAL (presentation) |
| Sharpened structure-content factorization | Tolman-Eichenbaum Machine: separate structural from sensory codes for generalization | 6.4, 9.3, 13.4 | REAL (the frontier; approximation only) |
| Hopfield/attention readout | Modern Hopfield equals attention; one-step pattern completion; Sparse Distributed Memory | 8.1, 8.2 | REAL |
| Memory linking by co-activation | Engram overlap for co-encoded or co-reactivated memories | 6.7, 6.10 | REAL |
| Predict-then-verify retrieval | Constructive/predictive memory; Spens-Burgess generative consolidation; free-energy calibration | 6.12, 8.6 | ASPIRATIONAL |
| Ordered working buffer | Theta-gamma phase code: ordered items by phase, explicit order | 6.5 | ASPIRATIONAL |

What to be skeptical of, carried from the dossier's REAL-versus-METAPHOR discipline: do not add computer vision models whose outputs do not become a node, edge, vector, or verification; do not claim literal Tolman-Eichenbaum factorization (ship the approximation); do not store memory in 3D (it is a projection); do not overclaim recency-independence (state it as age-independence at fixed N); and treat every benchmark number, including any this system produces, as an upper bound with methodology and variance reported.

---

## 12. THE CONDENSED `/goal` PROMPT (UNDER 4000 CHARACTERS)

Paste this into `/goal`. It triggers the entire spec above.

```text
GOAL: Read EIDETIC_PLUS_UPGRADE_SPEC.md and the files it references, then execute it in one shot on
the existing code (substrate.py, vector_index.py, graph.py, structure_code.py, salience.py, fsrs.py,
engine.py, retrieval.py, dashscope_client.py, ingestion.py, api.py, web/index.html). Don't rewrite
what works. Build, wire, test, run the demo, update docs; don't stop until all below works and tests
pass.

RULES: no mocks (real DashScope calls, fail loud on a missing key, never fake one); never delete a raw
record (forgetting only down-weights the index); FSRS priority stays OUT of ranking (recall must be
age-independent; the flat curve is the proof); the immutable record is the final arbiter for all
verification (text and visual). Ask before any non-standard dep.

BUILD (this order; presentation is the only thing cut if short):
1) UNIVERSAL PLUGIN: MCP server over the same engine.py (keep FastAPI; stdio + HTTP). Tools: remember,
recall, consolidate, reawaken, list_memories, get_raw, prove_age_independence. CRITICAL, don't skip:
SCOPING. Every tool takes an explicit scope (required namespace + optional agent_id, project_id),
enforced at the index and graph (filter on read, tag on write); default scope explicit, never global.
Test: write in namespace A, confirm invisible from B. Package twice: a plain MCP server any host can
mount, AND a Claude Code plugin manifest in /plugin.
2) ENGINE UPGRADES (edit existing files): reconsolidation as a write path (confirmed recall = re-embed
+ up-weight, the immune affinity-maturation idea; contradicted recall = invalidate old edge, never
delete, write the correction); tagging-and-capture (a salient event up-weights temporally adjacent
memories in a window); surprise-based segmentation (chunk long inputs at Bayesian-surprise boundaries,
not fixed windows); schema-accelerated consolidation (schema-consistent facts fast-track in the sleep
loop; novel facts stay episodic longer); prove_age_independence as a live tool + route, always flat;
RUN signature_demo.py and save the plot.
3) DEEPENED VISION (feed the graph): images/screenshots/diagrams/tables become real entities + edges
in the bi-temporal graph via qwen-vl-ocr and qwen-vl-plus; video = keyframe entities + ordered edges;
VISUAL VERIFICATION = extend no-confabulation to images, judge a visual claim against the pixels with
qwen-vl-plus, reject/abstain if unsupported, raw image is arbiter. No CV models whose output is not a
node/edge/vector/check.
4) FRONTIER (real): sharpen structure_code.py (disentangle structure code from content embedding,
retrieve compositionally over both, cite Tolman-Eichenbaum as inspiration, keep the approximation);
frame the final readout as a Hopfield/attention pattern-completion step; memory linking
(co-retrieved-and-confirmed memories gain a strengthened edge).
5) 3D MAP (presentation, a VIEW not storage; cut line if /goal stalls): render the bi-temporal graph
in 3D (react-force-graph-3d) over a UMAP/3D-force projection of embeddings; nodes colored by salience,
sized by FSRS retrievability; edges show bi-temporal validity; node-click shows raw record + hash +
provenance. UI and comments must say: stored in 1024-2048D, 3D is a projection, never the
representation.
6) ASPIRATIONAL (only if all above lands, never stubs): predict-then-verify retrieval; ordered working
buffer with explicit slot tags.

ALSO fix the file:// UI bug (serve UI from FastAPI or inject a configurable API base URL).

DONE = MCP mounts (stdio + HTTP) + as a Claude Code plugin; scoping isolates memories (tested); vision
yields real graph entities/edges and visual verification rejects an unsupported claim; step-2 upgrades
wired; frontier 7.1-7.3 done; 3D map navigable with provenance on click, labeled a projection;
signature_demo.py run and flat-curve plot saved; tests pass (scope isolation, reconsolidation
re-weighting, visual-verification rejection, no-delete-on-forget); README + docs/architecture.md
updated; MIT kept.
```

---

*One file, the whole upgrade. The engine stays lossless, age-independent, and non-confabulating. The plugin makes it universal. The vision makes it see and prove. The factorization makes it generalize. The map makes it unforgettable. Build the substance first, let the ambition ride on top of the proof, and never ship a stub.*
