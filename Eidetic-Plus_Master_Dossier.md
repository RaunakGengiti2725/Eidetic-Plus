# EIDETIC-PLUS: Complete Master Research Dossier
### A Beyond-Human, Lifelong, Multimodal, Provenanced Memory Agent for the Qwen Cloud Hackathon (Track 1: MemoryAgent)

> One consolidated file. Every piece of research from this project, organized end to end: competition analysis, strategic thesis, full competitive teardown, the complete neuroscience and cellular/tissue/immune foundation, the computational and mathematical memory models, the seven-component architecture, the verified Alibaba Cloud and Qwen service mapping, the multimodal ingestion and retrieval pipelines, every identified gap with its real fix, the evaluation plan, the build plan, and the honest risk register.
>
> Status note up front, because you asked directly: the major architectural gaps are identified and closed with real, GA-verified solutions. A small number of residual risks remain. They have documented mitigations rather than perfect closure. They are listed explicitly in Section 14 and Section 15. Nothing in this design depends on mock components.

---

## TABLE OF CONTENTS

1. Executive summary and the one-paragraph idea
2. The competition: tracks, judging, submission requirements, what actually wins
3. Strategic thesis and positioning
4. Competitive landscape: full teardown of every system and our counter
5. Human super-memory phenomenology (the conceptual core)
6. Brain anatomy and memory mechanisms
7. Cellular, tissue, immune, and embodied memory, graded REAL vs METAPHOR
8. Computational and mathematical memory models
9. The Eidetic-Plus architecture: seven components, fully specified
10. Real-service verification on Qwen and Alibaba Cloud
11. Multimodal ingestion pipeline (real model IDs)
12. Retrieval orchestration pipeline
13. Gap analysis: every gap and its fix
14. Evaluation plan: benchmarks, the disputed-scores story, the novel metric
15. Are all gaps filled? Honest assessment
16. Build plan (phased to the July 9 2026 deadline)
17. Risk register and preemptions
18. Key sources
19. Appendix A: Consolidated benchmark scoreboard (every number in one table)
20. Appendix B: Primary-source links (verified this research)
21. Appendix C: Exact-quote evidence locker (for the demo and Q&A)
22. Appendix D: One-screen architecture description (for the required diagram)

---

## 1. EXECUTIVE SUMMARY AND THE ONE-PARAGRAPH IDEA

**The idea.** Eidetic-Plus is a memory layer for AI agents that captures everything a user ever sends or does, across every modality, stores it losslessly and immutably, indexes it so that any memory can be retrieved at uniform speed regardless of how old it is, and reconstructs answers that are verified against the original record so the system never confabulates. It separates a permanent, write-once "perfect record" from a fast, mutable "index" that organizes, consolidates, forgets-by-deprioritizing, and resolves contradictions over time. The design is grounded in the actual neuroscience of human memory (hippocampal indexing, complementary learning systems, grid-cell coordinate maps, surprise-gated encoding, sleep consolidation) but is engineered to surpass the brain on the three axes the brain provably fails: perfect fidelity, no interference-based forgetting, and recall that is independent of recency.

**Why it wins the rubric.** The hackathon weights Innovation 30 percent and Technical Depth 30 percent, then Problem Value 25 percent and Presentation 15 percent. Eidetic-Plus targets all four: the architecture is novel against the entire 2026 field (verifiable reconstruction, immutable provenance, recency-independent retrieval are unclaimed combinations), it is deeply engineered (multi-model orchestration, bi-temporal graph, NLI verification, FSRS forgetting, ANN plus graph plus rerank fusion), it solves the track's exact asks (efficient storage and retrieval, timely forgetting, recall within limited context), and it produces a signature visual result (flat recall-versus-age and latency-versus-age curves) that no competitor reports.

**The honest framing that sets the tone.** Photographic memory does not exist in humans the way the word implies. We do not imitate a faculty humans lack. We build, in silicon, the thing the brain only approximates, and we are rigorous about which biological mechanisms transfer as real engineering and which are only metaphor.

---

## 2. THE COMPETITION

### 2.1 Event
- **Global AI Hackathon Series with Qwen Cloud**, hosted by Alibaba Cloud, managed by Devpost.
- **Deadline:** July 9, 2026, 2:00 pm PDT.
- **Prize per track:** 7,000 USD cash plus 3,000 USD cloud credits, blog feature, swag, ambassador opportunity. Five tracks, plus a Blog Post award (10 x 500 USD) and Top 10 Honorable Mentions (10 x 500 USD).

### 2.2 Track 1: MemoryAgent (our track)
Build an agent with persistent memory that autonomously accumulates experience, remembers user preferences, and makes increasingly accurate decisions across multi-turn, cross-session interactions. The three explicit focus areas:
1. Efficient memory storage and retrieval.
2. Timely forgetting of outdated information.
3. Recalling critical memories within limited context windows.

These three map one-to-one onto Eidetic-Plus components (the index plus ANN for retrieval, FSRS index-priority decay for forgetting, and reconstruction-into-limited-context for the third).

### 2.3 Judging criteria (the weights that drive every decision)
- **Technical Depth and Engineering, 30 percent.** Sophisticated use of Qwen Cloud APIs (custom skills, MCP integrations). Algorithmic or engineering innovation through novel solutions, custom components, performance optimization.
- **Innovation and AI Creativity, 30 percent.** High-quality architecture with modularity, scalability, error handling. Clean code and non-trivial logic. Advanced patterns and thoughtful adoption.
- **Problem Value and Impact, 25 percent.** Real-world relevance, authentic technical or business pain point. Scalability potential for productization or open-source adoption.
- **Presentation and Documentation, 15 percent.** Clear technical demo with key logic visualized. Clear documentation including architecture docs.

### 2.4 Submission requirements (hard gates)
- Public code repository with a detectable open-source license file visible in the About section.
- Proof the backend runs on Alibaba Cloud (a short recording plus a link to a code file demonstrating Alibaba Cloud services and APIs).
- An architecture diagram showing how Qwen Cloud connects to backend, database, and frontend.
- A roughly 3-minute public demo video (YouTube, Vimeo, or Facebook Video).
- A text description of features and functionality.
- Track identification.
- Optional: a published blog or social post about the build journey (eligible for the Blog Post prize).

### 2.5 What actually wins this
Sixty percent of the score is Innovation plus Technical Depth. So the project must be both genuinely novel against the current field and deeply engineered, not a wrapper. The remaining forty percent rewards solving a real pain point and presenting it clearly. Eidetic-Plus is designed so that the novelty (verifiable, immutable, recency-independent, provenanced lifelong multimodal memory) is also the engineering, and the demo produces a measurable result competitors cannot show.

---

## 3. STRATEGIC THESIS AND POSITIONING

### 3.1 The reframe that beats a crowded field
"Brain-inspired memory" is no longer differentiating. By early 2026 the field already has dual-store episodic/semantic systems, salience tagging, sleep-style consolidation, knowledge graphs, and event segmentation (Mem0, Zep, Letta, Cognee, HippoRAG, MIRIX, MemOS, SYNAPSE, TiMem, MAGMA, EverMemOS, A-MAC, Memory-R1, plus open-source "dream" engines). Submitting "hippocampus plus neocortex plus a sleep cron" reads as a remix.

The reframe: photographic memory is a human myth. We make it real in silicon, and the one capability computers have that brains structurally lack is lossless capture. The brain is the floor, not the ceiling.

### 3.2 The two ideas held strictly separate
1. **The hook:** photographic/eidetic memory. Memorable, on-theme, and (as Section 5 shows) genuinely unclaimed as an AI-agent framing.
2. **The substance:** lossless storage by itself is trivial and unimpressive (logging to object storage is something anyone can do, and the brain's celebrated capacity is dwarfed by cheap cloud storage). The novelty and the technical-depth points come from the indexing, the verified retrieval, the recency-independence, the provenance, and the cognitive-coordinate structure layered on top. Conflating the hook with the substance is the trap that loses technical judges. We never do that.

### 3.3 The three defensible, beyond-human claims
Every 2026 survey lists these as open problems. The brain cannot do them. Most current systems do not.
1. **Decoupled lossless retention from forgetting.** The brain cannot (interference is mandatory). Current systems delete to stay cheap. We keep everything immutably and forget only at the index.
2. **Uniform-latency retrieval irrespective of memory age.** Approximate nearest neighbor search gives retrieval cost that depends on store size and recall target, never on a memory's timestamp. A thirty-year-old memory and a one-second-old memory cost the same to fetch. This defeats both human recency bias and the long-context "lost in the middle" failure of large language models.
3. **Full source and temporal provenance with contradiction handling.** Bi-temporal records (event-time and ingestion-time) answer exactly when something was learned and when it was true, and resolve conflicts while retaining the full history.

### 3.4 The through-line of the whole design
Index is not content (split lossless store from mutable index). The index is a spatial coordinate system (factorize structure from content the way grid cells do). Recall must be age-blind (approximate nearest neighbor gives uniform latency). Forgetting lives in the index, never in the store (keep everything, still forget on time). The three beyond-human properties fall directly out of this spine.

---

## 4. COMPETITIVE LANDSCAPE: FULL TEARDOWN

### 4.1 The four original paradigms
- **Mem0:** vector store plus LLM fact extraction at write time. Optimizes stable user-preference recall. Roughly 48,000 GitHub stars. Reported about 49 percent on LongMemEval in independent framing; self-reports much higher on LoCoMo (see 4.3). Weakness: lossy extraction discards the raw signal.
- **Zep / Graphiti:** bi-temporal dynamic knowledge graph tracking fact-validity windows. Strong on chronological correctness. Reported about 63.8 percent on LongMemEval in one comparison. Weakness: heavy LLM cost per episode and post-ingestion retrieval lag.
- **Letta / MemGPT:** operating-system metaphor with tiered core (in-context), recall (session history), and archival (external) memory; the agent self-manages paging via tools. Weakness: heuristic, manual paging; no true consolidation; agent reasoning errors can corrupt memory.
- **Cognee:** Extract-Cognify-Load pipeline into a typed knowledge graph from heterogeneous documents. Weakness: slower, costlier graph construction; quality depends on the extraction model; fewer production/compliance guarantees.

### 4.2 The neurobiological and research systems
- **HippoRAG and HippoRAG 2** (Gutierrez et al., NeurIPS 2024; the 2025 follow-up "From RAG to Memory"): hippocampal-indexing-theory RAG. An LLM "neocortex" builds an open knowledge graph "hippocampal index," a parahippocampal encoder detects synonymy, and Personalized PageRank performs associative multi-hop retrieval (modeling pattern separation and completion). Original up to 20 percent better on multi-hop QA, 10 to 30 times cheaper and 6 to 13 times faster than iterative retrieval like IRCoT; HippoRAG 2 adds passage integration and recognition memory, about 7 percent over embedding models on associative tasks. Weakness: retrieval-only over a static corpus, entity-centric context loss, offline indexing overhead, text-only, no provenance, no forgetting, no temporality.
- **EM-LLM** (Fountas et al., ICLR 2025): segments token streams into events via Bayesian surprise plus graph-theoretic boundary refinement; two-stage retrieval (similarity k-NN plus temporal contiguity); handles 10M-token contexts. Reported overall relative improvement of 4.3 percent over InfLLM, with a 33 percent improvement on PassageRetrieval, and surpasses NV-Embed-v2 RAG by 30.5 percent on LongBench. Weakness: operates within a single model's context and KV cache; not a persistent lifelong store.
- **Larimar** (IBM, ICML 2024): complementary-learning-systems-inspired external episodic memory controller conditioning a frozen decoder; one-shot edits and selective forgetting; about 100 percent rewrite accuracy up to 512 slots, dropping to about 82 percent at 1024. Weakness: capacity-bounded, degrades beyond K slots.
- **MemOS / MemCube** (Li et al., 2025): a memory operating system unifying parametric, activation (KV-cache), and plaintext memory as composable MemCubes (content plus provenance plus versioning) with a scheduler; reports about 35 percent token savings. Weakness: largely an architectural framework; cross-model sharing unsolved.
- **MIRIX** (2025): multi-agent memory with six typed modules (Core, Episodic, Semantic, Procedural, Resource, Knowledge-Vault), state-of-the-art LoCoMo, and a real multimodal screenshot benchmark. A genuine frontier threat. Our differentiator against it is verifiability and immutability, not just memory typing.
- **AriGraph** (Anokhin et al., IJCAI 2024): semantic knowledge graph plus episodic vertices as a graph world model for exploration; tuned for text-game environments.
- **Generative Agents** (Park et al., Stanford 2023): a memory stream with importance-weighted retrieval (recency times importance times relevance) plus periodic reflection into higher-level insights. Weakness: hand-tuned scoring, no provenance or contradiction handling.
- **RAPTOR:** hierarchical recursive summarization tree for multi-resolution retrieval. **A-MEM:** Zettelkasten-style self-linking notes. **MemoryBank** (Zhong 2024): Ebbinghaus-curve forgetting. **MemoryOS.**
- **2025 to 2026 wave:** SYNAPSE (spreading activation plus lateral inhibition plus temporal decay; reported up to 23 percent multi-hop gain and 95 percent token reduction on LoCoMo), TiMem (five-level Temporal Memory Tree; reported 75.3 percent LoCoMo and 76.9 percent LongMemEval-S), MAGMA (four orthogonal graphs: semantic, temporal, causal, entity, with adaptive traversal), EverMemOS / EverMind (engram-lifecycle MemCells to MemScenes), A-MAC (admission control over future-utility, confidence, novelty, recency, type; reported LoCoMo F1 about 0.583, latency reduced about 31 percent), Memory-R1 (reinforcement-learned add/update/delete/retrieve, trained on as few as about 152 QA pairs), Nemori (event-segmentation-theory boundary alignment plus a free-energy predict-calibrate loop), HINDSIGHT (biomimetic, reported LoCoMo up to about 89.6 percent).

### 4.3 The benchmark-credibility situation (critical context)
The field's leaderboard numbers are contested and many are vendor self-reported.
- The Mem0-versus-Zep LoCoMo dispute is real and unresolved. Mem0's CTO reported that restricting to the first four validated LoCoMo categories and averaging over ten independent runs yields a Zep mean of 58.44 percent plus or minus 0.20. Zep's rebuttal claims about 75.14 percent plus or minus 0.17 (a "J score"), down from its original roughly 84 percent claim, still outperforming Mem0's best configuration by about 10 percent.
- LoCoMo conversations are only about 16,000 to 26,000 tokens, which fits inside modern context windows, so the field is moving to LongMemEval and BEAM (1M to 10M tokens).
- LLM-judge scores move by roughly 10 points just from swapping the judge model. Any benchmark claim must fix the judge model and prompt across all systems and report variance over multiple runs.

### 4.4 Per-competitor weakness mapped to our counter
- **Mem0:** lossy extraction; disputed scores. Counter: immutable lossless substrate plus NLI-verified reconstruction means we never lose the original and can prove faithfulness.
- **Zep / Graphiti:** best-in-class temporal graph but expensive (the Mem0 ECAI 2025 paper measured Zep's memory footprint exceeding 600,000 tokens per conversation versus 1,764 for Mem0) and post-ingestion retrieval lag (correct answers appear only after background processing). Counter: keep the bi-temporal model, but serve immediate approximate-nearest-neighbor recall while the graph catches up asynchronously.
- **Letta / MemGPT:** manual self-managed context tiers; roughly 74 to 83 percent LoCoMo. Counter: automated consolidation and replay.
- **HippoRAG 2:** strong associativity but text-only, no provenance, no forgetting, no temporality. Counter: multimodal plus provenance plus FSRS forgetting plus bi-temporal versioning.
- **MIRIX:** six memory types, multimodal, frontier-level. Counter: verifiability plus immutability plus provenance, which MIRIX does not provide.
- **Generative Agents, A-MEM, EM-LLM, Memory-R1, MemOS:** research systems; we adopt EM-LLM's surprise segmentation and note Memory-R1's RL management as a future enhancement.

### 4.5 The current frontier to not be blindsided by
Emergence AI (about 86 percent LongMemEval), MIRIX, MemOS, A-MEM, Memory-R1, and the BEAM benchmark (1M to 10M tokens, deliberately unsaturated; no system saturates it). Mem0's April 2026 token-efficient algorithm reports about 92.5 on LoCoMo and 94.4 on LongMemEval at roughly 6,900 tokens per query, with about plus 29.6 points on temporal reasoning and plus 23.1 on multi-hop, plus BEAM 64.1 (1M) and 48.6 (10M). Treat all of these as upper bounds.

---

## 5. HUMAN SUPER-MEMORY PHENOMENOLOGY (THE CONCEPTUAL CORE)

### 5.1 Eidetic memory is real only in a weak, transient form; photographic memory is unproven
Eidetic imagery (a vivid, externally projected afterimage of a scene) occurs in a minority of pre-pubescent children. Prevalence estimates range from about 2 to 10 percent; Giray et al. (1976), testing 280 children, identified just 5.6 percent as eidetic. The ability nearly vanishes in adults, the images fade within minutes, and they are subject to semantic distortion. They are not photographically accurate.

The single most famous adult case is "Elizabeth," studied by Charles Stromeyer III and Joseph Psotka (Nature, 1970), who reported she could mentally fuse two 10,000-dot random-dot stereograms shown to each eye a day apart to perceive a hidden three-dimensional figure. The case was never replicated: Stromeyer married her, she refused all further testing, and methodological concerns were raised. John Merritt's 1979 mass screening, titled "None in a million," is decisive: of an estimated one million people who took the published test, only 30 wrote in with the correct answer, and of the 15 he visited in person, none could replicate the feat under observation. The scientific consensus is that no one has demonstrated true photographic recall with perfect fidelity over time.

**Design implication.** Do not mimic a faculty humans do not have. Leverage what computers trivially do (lossless capture), which is precisely the capability humans lack.

### 5.2 HSAM (Highly Superior Autobiographical Memory) is real but mechanistically narrow
First documented as hyperthymesia in patient A.J. (Parker, Cahill, and McGaugh, 2006), then studied in a cohort of 11 by Aurora LePort, James McGaugh, and colleagues at UC Irvine. Whole-brain analysis identified 9 structures as morphologically different from controls (including the temporal pole and grey matter near the anterior putamen and caudate), yet HSAM participants performed comparably to controls on most standard laboratory memory tests, and retained no more lab detail than controls at one-day and one-week delays. The cohort showed elevated obsessive-compulsive tendencies.

**Interpretation.** HSAM is driven largely by compulsive, continuous rehearsal of autobiographical material rather than a superior storage substrate. A 2024 systematic review concluded that consistent structural differences do not characterize HSAM, though altered hippocampal resting-state connectivity is commonly observed.

**Design implication.** "Recall every day of your life" in an AI is achieved by logging plus indexing, not by a special encoder. The rehearsal mechanism maps to a consolidation and replay loop.

### 5.3 Savant memory implies a "store literally, do not abstract" strategy
Kim Peek (the megasavant behind Rain Man; born without a corpus callosum) could accurately recall the contents of at least 12,000 books, retaining an estimated 98 percent, reading the left page with his left eye and the right with his right in 8 to 10 seconds. Stephen Wiltshire draws accurate cityscapes after one helicopter flight. Daniel Tammet recited 22,514 digits of pi and experiences numbers synesthetically. The leading accounts are weak central coherence (autistic cognition favors local and literal detail over gestalt abstraction) and local hyperconnectivity. Allan Snyder's transcranial magnetic stimulation work induced savant-like skills in neurotypicals by inhibiting fronto-temporal regions, suggesting savant skill is partly release from top-down conceptual compression.

**Design implication.** Store raw, un-summarized detail. This is the opposite of most current agent systems, which summarize at write time and destroy the episodic signal. Summarization is a derived layer, never a replacement for the raw trace.

### 5.4 Mnemonists and the method of loci show spatial indexing is the master key
Solomon Shereshevsky (Luria, "The Mind of a Mnemonist," 1968) had fivefold synesthesia that gave every item multiple redundant sensory tags and used the method of loci, but suffered from an inability to forget and to abstract, which impaired comprehension. Modern memory athletes use the method of loci (memory palaces), Person-Action-Object systems, and spaced repetition.

**Design implication.** Synesthesia is multi-modal redundant binding. The inability to forget is a cautionary tale that argues for an index that forgets and reweights over an immutable store that does not.

### 5.5 The neuroscience of the method of loci, and the size of the effect
Dresler et al. (Neuron, 2017) studied 23 of the world's top memory athletes against matched controls, plus naive subjects split into method-of-loci, working-memory, and no-training groups. The athletes had no structural brain differences from controls but distinctive functional connectivity. After 40 days of daily 30-minute method-of-loci training, naive subjects went from recalling an average of 26 words out of 72 to remembering 62; the athletes recalled 70.8 plus or minus 0.6 of 72 versus controls' 39.9 plus or minus 3.6, and gains persisted at four months as trainees' connectivity shifted toward the athlete pattern. The technique recruits the hippocampus and visuospatial networks.

**Design implication.** Imposing a spatial or relational coordinate system on arbitrary information dramatically boosts retrievability. This directly motivates a cognitive-map coordinate index.

### 5.6 Information-theoretic capacity
The widely cited estimate is about 2.5 petabytes for the human brain (Reber, 2010). The mechanistic basis: Bartol, Sejnowski, and colleagues (eLife, 2016) reconstructed hippocampal synapses and found about 26 distinguishable size states, roughly 4.7 bits per synapse, about ten times prior estimates. Note these are upper-bound extrapolations; a simpler calculation yields only about 54 terabytes. Human recall fails not from running out of space but from interference (competing memories) and retrieval failure, not true decay.

**Design implication.** Capacity is not the human bottleneck and certainly not an artificial one. Interference and indexing are. Petabyte-scale lossless storage of a lifetime of multimodal experience is trivially feasible on cloud object storage.

### 5.7 Prior "photographic / eidetic memory for AI" attempts
A targeted search found no peer-reviewed paper or established product framing LLM or agent memory as "eidetic" or "photographic." The field uses "episodic," "long-term," "persistent," and "perfect recall" (in the game-theory sense). The closest prior uses are an informal product name ("Total Recall," closed beta) and assorted hobbyist "total-recall" repos, plus ubiquitous "never forgets" marketing. The eidetic/photographic framing for AI agents is therefore essentially unclaimed and defensible as novel, provided the architecture is differentiated from "episodic," "perfect-recall," and "never-forget" framings by the specific mechanisms in this dossier.

---

## 6. BRAIN ANATOMY AND MEMORY MECHANISMS

### 6.1 Hippocampal indexing theory (the master principle)
Teyler and DiScenna (1986): the hippocampus does not store content. It stores a sparse index or pointer to the distributed neocortical patterns that constitute a memory, and reactivates them on cue. Index is not content. This is the single most important biological principle for the architecture and justifies the two-layer split between an immutable lossless content store and a lightweight mutable index.

### 6.2 Complementary Learning Systems (the canonical theory)
McClelland, McNaughton, and O'Reilly (1995): the hippocampus is a sparse, pattern-separated system for rapidly learning episodic memories; the neocortex is a distributed, overlapping system for gradually integrating across episodes to extract latent semantic structure. During sleep, hippocampal replay teaches the neocortex through interleaved learning, incorporating new information gradually into existing knowledge. This framework already inspired machine learning directly: prioritized experience replay in Deep Q-Networks came from the observation that rewarded events replay more often.

### 6.3 Grid cells, place cells, time cells, and conceptual maps
Place cells (O'Keefe) fire at specific locations. Grid cells in the medial entorhinal cortex (Hafting, Fyhn, Moser, and Moser, Nature 2005) tile space in a hexagonal lattice providing a metric coordinate system. Time cells encode temporal position. Critically, Constantinescu, O'Reilly, and Behrens (Science, 2016) showed humans navigate abstract conceptual spaces using the same hexagonal grid-like code in entorhinal and ventromedial prefrontal cortex. The hippocampal formation is a general coordinate and indexing system, not merely spatial.

### 6.4 The Tolman-Eichenbaum Machine (the factorization)
Whittington, Muller, Mark, Chen, Barry, Burgess, and Behrens (Cell, 2020): medial entorhinal cells form a structural basis describing relational and positional knowledge, and hippocampal cells link this basis with sensory representations. Separating structural codes from sensory codes allows generalization over environments sharing the same structure. This is the architectural heart of the beyond-human design: factorize memory into a reusable relational coordinate "structure" code and a sensory "content" code, bound conjunctively.

### 6.5 Theta-gamma phase code
Lisman and Jensen (Neuron, 2013): multiple items (about 7 plus or minus 2) are held in working memory by nesting about 7 gamma subcycles (each roughly 20 to 30 milliseconds, one item per cell assembly) within one theta cycle (roughly 125 to 250 milliseconds); temporal order is encoded by phase position, and phase precession compresses behavioral sequences into theta sequences. The 20 to 30 milliseconds per item rate matches the Sternberg memory-scan response-time data.

**Design implication.** Ordered, indexed retrieval with explicit slot or position tags, a structured ordered buffer rather than a flat similarity search.

### 6.6 Dentate gyrus pattern separation and CA3 pattern completion
The dentate gyrus uses sparse expansion recoding (many granule cells, very sparse activity) to orthogonalize similar inputs (pattern separation), aided by adult neurogenesis. CA3's recurrent autoassociative network performs pattern completion, reconstructing a whole memory from a partial cue. The hippocampal memory indexing theory frames long-term memory around exactly these two objectives: pattern separation (distinct experiences stay unique) and pattern completion (retrieval of complete memories from partial stimuli).

**Design implication.** Separate near-duplicate episodes at write time (dedup plus distinct keys), but enable cue-based reconstruction at read time (associative and graph retrieval).

### 6.7 Molecular engram and plasticity
Long-term potentiation and depression, synaptic tagging and capture (Frey and Morris), CREB-based neuronal allocation, PKMzeta as a candidate memory-maintenance molecule, dendritic-spine dynamics, and intrinsic-excitability-based memory allocation. Synaptic tagging and capture explains how a weak memory can be made persistent if a salient or novel event occurs within roughly a one-hour window. Neurons with higher excitability at encoding are preferentially allocated to an engram, and memories encoded close in time share overlapping engrams (memory linking).

**Design implication.** A tagging mechanism where salience and novelty signals retroactively upgrade the retention and priority of temporally adjacent memories.

### 6.8 Neuromodulatory gating
Dopamine (the ventral tegmental area to hippocampus loop; Lisman and Grace, 2005) tags novel and rewarding events for persistence. Acetylcholine switches the hippocampus between encoding and retrieval modes and gates attention. Noradrenaline from the locus coeruleus signals arousal and surprise and, with co-released dopamine (Takeuchi, Duszkiewicz, and colleagues), boosts retention of surprising events, the basis of flashbulb memories. The amygdala tags emotionally arousing memories for durability. Moncada and Viola (PNAS, 2010) showed brief novelty exploration can convert a transient trace into a persistent one via ventral-tegmental-origin dopamine signaling. Novelty and surprise are the triggers for strong encoding.

**Design implication.** A write-time salience scorer (novelty plus surprise plus reward/importance plus emotional valence) controls retention priority and replay frequency.

### 6.9 Active forgetting (the principle most systems get wrong)
Ryan and Frankland (Nature Reviews Neuroscience, 2022): forgetting is a form of neuroplasticity that alters engram-cell accessibility in a manner sensitive to mismatches between expectations and the environment. Engrams that are subjectively less relevant for adaptive behavior are more likely to be forgotten. Crucially, in many cases forgetting reduces engram accessibility rather than engram loss; it produces retrieval failure rather than storage failure. Competing engrams coexist and compete, and decision-making is governed by a balance between them. Retrieval-induced forgetting (Anderson) shows that recalling some items actively suppresses related competitors.

**Design implication.** Never hard-delete. Forgetting is accessibility (index-priority) modulation, reversible on a strong cue. Contradictions are handled by competition, not overwriting.

### 6.10 Memory reconsolidation (retrieval is a write)
Reactivating a stored memory makes it transiently labile; during the window before it restabilizes (reconsolidates) it can be reduced, enhanced, or updated with new information. Retrievals of a young memory accompanied by reconsolidation result in strengthening and contribute to overall consolidation (the testing effect, mechanistically). The 2026 survey "Memory for Autonomous LLM Agents" lists reconsolidation as an unimplemented frontier: retrieval renders a memory labile and subject to revision, which could inform update mechanisms.

**Design implication.** Treat retrieval as a write path: confirmed memories are strengthened, contradicted memories are updated and the stale version suppressed, co-activated memories are linked.

### 6.11 Glia, the glymphatic system, and consolidation timescales
Astrocytes (the tripartite synapse) provide metabolic and modulatory support. The glymphatic system clears metabolic waste during sleep, enabling memory maintenance. Systems consolidation gradually transfers memories from hippocampus-dependence to neocortical independence. Multiple-trace and trace-transformation theory (Moscovitch, Nadel) argues vivid episodic detail always remains hippocampus-dependent while gist becomes neocortical. Schema-consistent information consolidates dramatically faster: Tse et al. (Science, 2007) showed schema-consistent memories became hippocampus-independent in 48 hours.

**Design implication.** An offline consolidation job that abstracts gist into a semantic store while preserving raw episodic traces, with schema-matching as an accelerator.

### 6.12 Memory is constructive and predictive
Episodic memory shares neural substrates with imagination and future thinking. The hippocampus supports simulation. Pattern completion is inference. This bridges to generative models (Section 8).

**Design implication.** Retrieval can be reconstructive (generate from compressed latents plus retrieved anchors) as long as a lossless ground-truth record exists to verify against, combining the brain's efficiency with a computer's fidelity.

---

## 7. CELLULAR, TISSUE, IMMUNE, AND EMBODIED MEMORY (GRADED REAL vs METAPHOR)

This section widens the biological foundation as requested, and grades every mechanism for whether it transfers as real engineering or is only evocative. The willingness to cut weak analogies is itself a credibility signal for technical judges.

### 7.1 Cellular and molecular memory beyond synapses
- **PKMzeta:** the molecule most associated with maintaining long-term potentiation; ZIP inhibition can erase established memories. Recent work (2025) suggests PKMzeta drives reconsolidation more than passive maintenance, and the field has controversy (PKMzeta-knockout animals still learn).
- **CPEB3:** prion-like aggregation maintains hippocampal long-term potentiation and spatial memory (Fioriti, Kandel, and colleagues, Neuron 2015).
- **DNA methylation:** Dnmt3a2 converts short- into long-lasting memory.
- **Grade: REAL principle.** Long-term persistence requires an active maintenance process distinct from encoding. This maps to the periodic consolidation and replay loop that refreshes index priority.
- **Grade: METAPHOR, cut as mechanism.** Prion self-templating as a literal storage mechanism, and RNA-based memory transfer (McConnell's planaria cannibalism work is discredited; Aplysia RNA-transfer is suggestive, not a storage mechanism). Do not claim these.

### 7.2 Immune memory (the strongest and most rigorous mapping)
Clonal selection plus affinity maturation: re-exposure to an antigen drives somatic hypermutation and selection for higher-affinity clones; the secondary response is faster and higher-quality; memory is maintained by antigen-specific restimulation that selects the highest-affinity clone. Memory is distributed across a population, not a single cell.
- **Grade: REAL, lead with this.** It maps cleanly to three architectural features: (1) reinforcement-on-recall, where memories retrieved and confirmed get "affinity-matured" (re-embedded and up-weighted), which is the reconsolidation strengthening mechanism; (2) distributed population coding, redundant traces rather than a single point of failure; (3) selective retention, where only confirmed-useful memories get promoted. This is the least hand-wavy biological story; use it as the headline.

### 7.3 Tissue and musculoskeletal memory
- **Muscle memory / myonuclear domain theory:** nuclei added during hypertrophy are retained during atrophy, enabling faster regrowth (debated but meta-analyzed). **Grade: REAL principle.** Retain latent index structure for down-weighted ("forgotten") memories so re-encoding or re-promotion is cheap. This directly justifies the FSRS down-weight-do-not-delete mechanism and the "user reverts a decision" reawakening case.
- **Wolff's law and osteocyte mechanosensing (bone remembers load); tendon and ligament mechanical memory.** These are adaptation and persistent state, not retrievable information. **Grade: METAPHOR, loose analogy only, label as such or cut.**
- **ACL injury and cortical reorganization:** mechanoreceptor loss after an anterior cruciate ligament rupture leads to bilateral sensorimotor and proprioceptive reorganization in the cortex. This is real neuroscience, but it is maladaptive plasticity, not an information-storage design principle. **Grade: ANALOGY ONLY.** Use only to illustrate that losing an input source forces distributed re-mapping, and flag it explicitly as analogy, or cut.

### 7.4 Other distributed and embodied systems
- **Enteric nervous system ("second brain") and peripheral sensitization / chronic pain as maladaptive memory.** Evocative, but they offer no genuine computational lesson beyond "distributed state persists." **Grade: METAPHOR, cut from the technical pitch.**

### 7.5 The REAL vs METAPHOR table (state this explicitly to judges)
- **REAL, maps to an architectural feature:**
  - Complementary Learning Systems -> dual store (immutable content plus mutable index)
  - Hippocampal indexing -> the index layer (pointers, not content)
  - Grid cells / Tolman-Eichenbaum Machine -> structure code vs content code (with the fallback caveat in 9.3 and 13.4)
  - Surprise and neuromodulation -> write-time salience gating
  - Synaptic tagging and capture -> retroactive upgrade of temporally adjacent memories
  - Reconsolidation -> retrieval as a write path (strengthen, update, suppress)
  - Active forgetting (accessibility, not loss) -> index-priority decay, reversible
  - Immune affinity maturation -> reranking plus reinforcement-on-recall
  - Myonuclear retention -> down-weight, do not delete
  - CPEB3 / PKMzeta maintenance -> consolidation refresh
  - FSRS forgetting curve -> per-memory retention state
- **METAPHOR, cut or clearly label as analogy:**
  - Prion self-templating as mechanism
  - RNA memory transfer
  - Fascia memory
  - Enteric nervous system
  - Bone and tendon mechanical memory
  - ACL cortical reorganization (analogy only)

---

## 8. COMPUTATIONAL AND MATHEMATICAL MEMORY MODELS

### 8.1 Sparse Distributed Memory (Kanerva, 1988)
A content-addressable memory over high-dimensional binary vectors (about 1,000 bits) with a sparse set of randomly distributed hard locations; writes distribute to all hard locations within a Hamming radius, reads pool nearby locations. It is auto-associative, noise-robust, degrades gracefully, supports one-shot learning, and exhibits human-like phenomena (interference, tip-of-the-tongue, knowing when it does not know). Kanerva mapped it to the cerebellum. The associative memory has exponential capacity.

**Design implication.** The theoretical template for graceful, content-addressable, high-dimensional recall, directly realizable with modern approximate-nearest-neighbor vector search.

### 8.2 Modern Hopfield networks (Ramsauer et al., "Hopfield Networks is All You Need," ICLR 2021)
Continuous-state Hopfield networks store exponentially many patterns (in dimension d), retrieve in one update step, with exponentially small retrieval error. The update rule is exactly the transformer attention mechanism. Three regimes: global averaging, metastable subset averaging, and single-pattern retrieval. (Krotov and Hopfield's dense associative memories and Demircigil et al. provide the exponential-capacity foundation.)

**Design implication.** Associative memory and attention are the same math. A retrieval layer can be framed as a Hopfield or attention readout over the stored set, giving near-instant single-step recall.

### 8.3 Vector Symbolic Architectures / Hyperdimensional Computing
Holographic Reduced Representations (Plate, 1995): binding via circular convolution, bundling via element-wise addition, similarity via dot product, with a cleanup memory to denoise. Variants include FHRR (complex phasors, binding by Hadamard product), MAP (bipolar, element-wise multiply), and Binary Spatter Codes (XOR). These store structured and compositional information (role-filler bindings) in fixed-width vectors with graceful capacity limits.

**Design implication.** Encode structured records (who, what, when, where, modality) as bound hypervectors for compositional, cue-based retrieval and one-shot binding, a substrate for the structure-times-content factorization.

### 8.4 Tensor Product Representations (Smolensky, 1990)
Bind roles to fillers via outer product, giving exact, decodable structured representations, but with dimensional explosion, which Holographic Reduced Representations and Vector Symbolic Architectures solve by compressing back to D dimensions.

**Design implication.** Tensor Product Representations are the exact-but-expensive end of the binding spectrum; use the reductions for scalability.

### 8.5 Memory-augmented neural networks (history and lessons)
Memory Networks (Weston, 2015), End-to-End Memory Networks (Sukhbaatar, 2015), Neural Turing Machines (Graves, 2014), and the Differentiable Neural Computer (Graves, 2016) added content- and location-based addressable external memory matrices. What they got right: differentiable read and write to external slots, separating computation from storage. What they got wrong: small fixed-size memory, hard to scale, training instability, which is why non-parametric retrieval-based external stores (RAG) won for lifelong memory.

**Design implication.** Build at the orchestration layer over a non-parametric store. Do not cram lifetime memory into weights.

### 8.6 Predictive coding, active inference, and generative memory
Friston's free-energy and active-inference framing treats memory as part of a generative world model. Spens and Burgess (Nature Human Behaviour, 2024, "A generative model of memory construction and consolidation") present a model where hippocampal replay (an autoassociative network) trains neocortical generative models (variational autoencoders) to recreate sensory experience from latent variables. It reproduces memory-age effects, hippocampal-lesion effects, semantic memory, imagination, future thinking, and schema-based distortions (for example boundary extension). Their 2025 follow-up frames hippocampo-neocortical interaction as compressive retrieval-augmented generation, a direct bridge to agent RAG.

**Design implication.** Consolidation is training or producing a generative summary on replayed episodes; retrieval is generative reconstruction anchored to retrieved indices. Keep raw traces to bound the distortion that pure generative memory introduces. Pure generative memory models human distortion, not fidelity, so adopt its consolidation mechanism and refuse its lossiness.

### 8.7 The mathematics of optimal forgetting and retention
Ebbinghaus's forgetting curve (1885) is exponential. SuperMemo's SM-2 (Wozniak, 1987) introduced ease factors. The modern FSRS (Free Spaced Repetition Scheduler) uses the DSR model (Difficulty, Stability, Retrievability) with a power-law forgetting curve that fits human memory better than exponential, cutting reviews roughly 20 to 30 percent for equal retention. FSRS-6 uses exactly 21 trainable parameters (the prior v4.5 used 17), trained on roughly 700 million reviews from about 10,000 Anki users. Settles and Meeder (ACL 2016, Half-Life Regression) showed machine-learned schedulers beat rule-based SM-2. Bjork's desirable difficulties and retrieval-induced strengthening show that effortful retrieval strengthens memory.

**Design implication.** Model each memory's retention with a DSR-style state; schedule replay and reinforcement of high-value memories and let the index priority (not the stored content) decay per a tunable curve, giving controllable, principled forgetting without data loss.

### 8.8 Catastrophic forgetting and continual learning
Elastic Weight Consolidation, generative replay, and rehearsal address forgetting in parametric learning. Because this architecture is non-parametric (no weight retraining), it sidesteps catastrophic forgetting entirely. Forgetting becomes a deliberate policy, not an accident.

---

## 9. THE EIDETIC-PLUS ARCHITECTURE: SEVEN COMPONENTS

A buildable, orchestration-layer design over Qwen (DashScope and Model Studio) plus Alibaba Cloud storage, vector, and graph services, with no weight retraining.

### 9.1 Component 1: Immutable Lossless Substrate ("neocortex / engram store")
Append-only object storage (raw multimodal blobs: text, image, audio, video, action traces), content-hashed (SHA-256 object keys for deduplication and provenance), never overwritten. This is the perfect-fidelity record, the thing no human has and current agent systems discard. Realized on Alibaba Cloud OSS with Write-Once-Read-Many (WORM) retention and versioning.

### 9.2 Component 2: Hippocampal Index Layer (index is not content)
Sparse pointers to the blobs, stored as two complementary structures:
- A vector index: qwen3-vl-embedding vectors in an HNSW or DiskANN approximate-nearest-neighbor store, giving uniform-latency, cross-modal recall, with fusion mode binding multimodal events into single vectors.
- A bi-temporal knowledge graph: entities and relations with both event-time (valid_at, invalid_at) and ingestion-time (created_at, expired_at), for associative, temporal, and causal multi-hop retrieval via Personalized PageRank and spreading activation.

### 9.3 Component 3: Cognitive-Coordinate Map (grid-cell / Tolman-Eichenbaum factorization)
Factorize each memory into a reusable structure code (graph position plus a coordinate embedding) and a content code (raw blob plus its embedding), bound conjunctively (Vector Symbolic Architecture or Holographic Reduced Representation binding of who, what, when, where, modality). This imports the method-of-loci advantage and the Tolman-Eichenbaum generalization, and is the highest-novelty element.
- **Honest status:** there is no production retrieval system that factorizes structure versus content codes; this is a research model. The shipped version is the metadata-structure-code fallback (Section 13.4): a structure code built from explicit metadata (entity type, role, temporal coordinate, graph-position or Personalized PageRank features) stored as a separate filterable vector alongside the content embedding. This delivers most of the cross-context generalization without claiming a literal neural Tolman-Eichenbaum Machine.

### 9.4 Component 4: Write-time Salience Gating (neuromodulatory analog)
A novelty/surprise scorer (Bayesian surprise via negative log-likelihood, EM-LLM style, plus embedding-distance novelty) combined with LLM-judged importance (qwen-flash, cheap) and explicit emotional-valence signals, setting each memory's initial DSR retention state and replay priority. Via synaptic-tagging-and-capture-style behavioral tagging, a salient event can retroactively upgrade temporally adjacent memories.

### 9.5 Component 5: Offline Consolidation and Replay (sleep analog)
A scheduled job (Function Compute cron trigger) that performs:
- Pattern separation: deduplication and distinct-keying of near-duplicates.
- Generative consolidation: Qwen (a thinking-tier model) summarizes episodes into a semantic store, Spens-Burgess style, verified against the raw traces.
- Schema-accelerated integration: fast-track schema-consistent facts (Tse et al.).
- Spaced-repetition reweighting: FSRS/DSR power-law update of index priority, never deleting content.

### 9.6 Component 6: Reconstructive, Verifiable Retrieval
Hybrid approximate-nearest-neighbor plus graph Personalized PageRank plus spreading-activation recall, then a Hopfield or attention-style readout, then generative reconstruction anchored to and checked against the immutable record. Verification uses NLI entailment (premise is the raw WORM record, hypothesis is the generated answer or summary); any unentailed content is rejected or flagged. This is what eliminates confabulation and earns the word "perfect."

### 9.7 Component 7: Provenance and Contradiction Engine
Every answer carries source, timestamp, and confidence. Conflicting facts are resolved temporally via bi-temporal versioning (newest-valid wins, full history retained), copying Graphiti's invalidation mechanism but with the immutable raw record beneath it.

### 9.8 The data flow in two paths
- **Wake (every turn):** ingest and encode the event losslessly into Component 1; compute salience (Component 4); write index entries (Component 2) with structure and content codes (Component 3); retrieve by pattern completion (Component 6); assemble context within the window; generate and verify the response (Component 6); reconsolidate what was used (strengthen, update, suppress; Components 6 and 7).
- **Sleep (scheduled or idle):** selective replay of high-salience episodes; abstract episodic to semantic with verification; link co-activated memories; decay index priority of the rest via FSRS (Component 5).

---

## 10. REAL-SERVICE VERIFICATION ON QWEN AND ALIBABA CLOUD

Everything below is a generally available product. The single most important deployment constraint is the Beijing-region requirement for qwen3-vl-embedding fusion mode.

### 10.1 Multimodal embeddings: qwen3-vl-embedding
Live on Model Studio. Two modes: independent vectors (one per input) and fused (enable_fusion true, combines text, image, and video into one vector). Supported dimensions: 2560 default, then 2048, 1536, 1024, 768, 512, 256. Text limit 32,000 tokens; max 1 image at or under 5MB; video at or under 50MB; up to 20 elements per request (at most 5 images). Pricing seen: image and video about 0.258, text about 0.1 per unit (verify on the live pricing page). Hard constraint: the qwen3-vl-embedding endpoint is documented as China (Beijing) region only; the whole vector pipeline must run in Beijing, or use the Singapore-available tongyi-embedding-vision-plus or flash (1152 or 768 dims, independent-only) as a regional fallback, or self-host the open-weights Qwen3-VL-Embedding 2B or 8B. Fusion mode is not supported via the OpenAI-compatible interface or the Java SDK; use the Python DashScope SDK or raw HTTP.

Reference point: Qwen3-VL-Embedding ranked first on MMEB-V2 at 77.8 as of January 2026.

### 10.2 Text embeddings: text-embedding-v4
Qwen3-Embedding, flexible dimensions from 64 to 2048, 100-plus languages, 8,192 tokens, batches of at most 10 texts per request. Use higher dimensions (for example 2048) for pattern-separated episodic keys and lower (768 to 1024) for semantic gist, mapping the sparse-hippocampus versus overlapping-neocortex distinction onto one Qwen model.

### 10.3 Reranking
gte-rerank is being retired (the official notice schedules removal on 2026-05-30, recommending qwen3-rerank as the replacement). Use qwen3-rerank and qwen3-vl-rerank (multimodal reranking with image and video queries). Limits: at most 4,000 tokens per document, 500 documents per request, 30,000 tokens per request.

### 10.4 Vector database options
- **Tablestore KNN:** DiskANN-based, under 10 percent of HNSW memory for comparable recall, serverless, pay-as-you-go, streaming (near-real-time after write), high-throughput insert, update, delete, scales to hundreds of billions of rows, inherits search-index filtering (good for bi-temporal).
- **AnalyticDB for PostgreSQL:** true HNSW (CREATE INDEX USING ann with dim, distancemeasure, hnsw_m, pq), L2, inner product, cosine, exact plus approximate. OpenAI-recommended pattern.
- **Recommendation:** AnalyticDB-PG for the hackathon (familiar pgvector-style SQL, easy bi-temporal WHERE clauses, hybrid BM25 plus dense), with Tablestore as the billion-scale story for the pitch. Alibaba also open-sourced Zvec (wraps the Proxima engine used in Taobao and Alipay) for a local demo. Milvus, OpenSearch Vector Engine, and Lindorm also exist on Alibaba Cloud.

### 10.5 Object storage: OSS WORM
BucketWorm (bucket-level) and ObjectWorm (object-level, mutually exclusive); retention 1 day to 70 years; a 24-hour unlocked window then lock makes it immutable; you can extend but never shorten; 409 FileImmutable on tamper attempts; Cohasset-certified (SEC 17a-4, FINRA); lifecycle transitions still allowed within retention; versioning can coexist. Content-addressing by naming objects with the SHA-256 of their content gives deduplication plus a provenance hash. This satisfies the "running on Alibaba Cloud" requirement directly.

### 10.6 Graph database
Alibaba GDB (Gremlin and TinkerPop 3.x, transactions, runs in a VPC alongside ECS) is generally available, but Personalized PageRank is not a native built-in. Implement Personalized PageRank and spreading activation in-application (scipy, networkx, or igraph) over edges pulled from GDB or a relational table. This is exactly what HippoRAG and HippoRAG 2 do, so it is well-precedented. Bi-temporal properties are stored as edge properties.

### 10.7 LLMs on DashScope
qwen3-max (262K context, up to 66K output), qwen-plus (about 1M context), qwen-flash (cheap tier), qwen-long (long documents), plus thinking variants. All OpenAI-compatible (the Singapore endpoint dashscope-intl.aliyuncs.com or the Beijing endpoint dashscope.aliyuncs.com). Tool and function calling, JSON and structured output, thinking modes, MCP, and the Qwen-Agent framework are all available. The free developer tier ended April 15, 2026 (now a one-time 1M-token-per-model trial), so budget for paid usage. Batch calling offers a roughly 50 percent discount.

Model-per-cognitive-function plan: qwen-flash for the constant write-path salience scoring; qwen-plus for response generation and the prediction-error or contradiction judge; a Qwen3 thinking-tier model for the offline consolidation reasoning. This itself is a "sophisticated use of the APIs" story for Technical Depth.

### 10.8 Compute and deployment
Function Compute: fully managed serverless, HTTP triggers, scheduled (cron-like) triggers, event triggers, millisecond scaling, pay-per-use. Ideal for the offline consolidation and replay loop and the scheduled FSRS recompute. ECS or Container Service (ACK) for the always-on FastAPI backend. The Serverless Framework has an official Aliyun Function Compute plugin.

---

## 11. MULTIMODAL INGESTION PIPELINE (REAL MODEL IDs)

Everything a user sends or does is read, understood, and stored. All model IDs below are real Qwen and Alibaba services.
- **Text:** chunk plus text-embedding-v4.
- **Images and screenshots:** qwen-vl-ocr (tables to HTML, formulas, bounding-box localization, 38,192-token context) for text-bearing images; qwen3-vl-embedding for semantic vectors.
- **PDFs and documents:** DocMind (doc, docx, pdf) or qwen-doc-turbo (up to 10 files via fileid).
- **Audio:** qwen3-asr-flash (at or under 10MB and 5 minutes, synchronous) or qwen3-asr-flash-filetrans (at or under 2GB and 12 hours, asynchronous, with word timestamps); paraformer-v2 for Mandarin plus 18 dialects (Beijing only).
- **Video:** qwen3-vl-plus directly (at or under 2GB, up to 20 minutes, fps and max_frames frame sampling) plus keyframe embedding via qwen3-vl-embedding (at or under 50MB for embedding).
- **Modalities Qwen cannot embed directly (arbitrary binary, sensor logs, code-action traces):** store the raw bytes in OSS, generate a text or JSON description via qwen3-max, embed the description; the OSS object remains the ground truth.

Deduplication via SHA-256 content-hash object keys before embedding controls cost.

---

## 12. RETRIEVAL ORCHESTRATION PIPELINE

One pipeline combining speed, association, and verification:
1. qwen3-vl-embedding query vector -> approximate-nearest-neighbor recall (top-k1, about 100) with a bi-temporal filter.
2. Seed nodes -> in-application Personalized PageRank and spreading activation over the GDB graph.
3. Reciprocal Rank Fusion of dense, BM25, and graph scores (Reciprocal Rank Fusion is Graphiti's documented default; Maximal Marginal Relevance and node-distance are options).
4. qwen3-rerank or qwen3-vl-rerank for the final top-k2 (about 10).

Latency budget: approximate-nearest-neighbor about 5 to 15 milliseconds, plus Personalized PageRank about 10 to 50 milliseconds in-application, plus rerank about 100 to 250 milliseconds, equals sub-second, comparable to Graphiti's sub-second target.

---

## 13. GAP ANALYSIS: EVERY GAP AND ITS FIX

### 13.1 Uniform-latency / recency-independence
Real numbers: HNSW reaches about 95 percent recall at 10 in 1 to 2 milliseconds on SIFT1M on CPU; going from 0.8 to 0.95 recall adds about 31 percent latency, but 0.95 to 0.99 is a 3 to 5 times cliff. HNSW RAM overhead is about 1.2 to 2.0 times raw vectors plus the graph; practical ceiling about 100 to 200 million vectors in RAM. DiskANN/Vamana (which Tablestore uses) reaches 1 billion vectors at 95 percent recall and about 5 milliseconds with about 90 percent less RAM (SSD I/O cost). Failure mode: an index tuned on the benchmark query distribution silently loses recall on production long-tail queries. **Fix and honest framing:** claim age-independence (retrieval cost is a function of store size and recall target, never of timestamp or age), demonstrated by a flat latency-versus-age curve. This is the rigorous version of "perfect recall regardless of recency," and it directly contrasts with the long-context "lost in the middle" degradation: Liu et al. ("Lost in the Middle," TACL 2024) found mid-context multi-document QA accuracy can fall below the closed-book baseline, with performance highest when relevant information is at the beginning or end and worst in the middle (the LongMemEval framing cites roughly a 30 percent accuracy drop).

### 13.2 Verified generative reconstruction (no confabulation)
NLI entailment is the validated backbone: treat the raw WORM record as the premise and the generated answer or summary as the hypothesis; entailment means grounded, contradiction or neutral means hallucination. Validated by the Attributable to Identified Sources framework, SummaC (sentence-level NLI aggregation), RAGTruth, FaithBench, and the HHEM leaderboard. Span-level support scoring (Luna, under 1 second on 16,000 tokens on an L4) and QA-based consistency add granularity. Reliability caveat to flag: NLI granularity mismatch can miss some inconsistencies, so the WORM raw record is always retained as ground truth and the system can always cite the immutable source. Optionally run verification asynchronously at consolidation time if per-query latency is too high.

### 13.3 Contradiction detection and bi-temporal resolution
Graphiti is the reference: EpisodicNodes (original input), EntityNodes, EntityEdges (fact edges with text and embedding), every fact edge bi-temporal (valid_at and invalid_at for world truth, created_at and expired_at for system knowledge). Incremental ingestion: LLM extraction, entity resolution (MinHash plus locality-sensitive hashing fast path, LLM fallback), edge dedup, then contradiction resolution that closes (invalidates) the old edge instead of deleting it. Known pitfalls: high LLM cost per episode and post-ingestion retrieval lag. Fix: serve approximate-nearest-neighbor recall immediately while the graph catches up asynchronously, with the immutable raw record beneath the graph.

### 13.4 Structure/content factorization (the highest-risk component)
The only real precedent is the Tolman-Eichenbaum Machine itself (a research model with PyTorch and TensorFlow implementations). There is no production retrieval system that factorizes structure versus content codes. **Fix (recommended for the hackathon):** approximate factorization by storing two vectors per memory, a content embedding (qwen3-vl-embedding) and a structure code built from explicit metadata (entity type, role, temporal coordinate, graph position or Personalized PageRank features), and filter or compose over both. Cite the Tolman-Eichenbaum Machine as inspiration, not as a literal implementation. Re-evaluate a deeper version only if the build finishes early.

### 13.5 Salience scoring
EM-LLM computes Bayesian surprise (negative log-likelihood of the next token), segments events at surprise spikes, refines boundaries with graph modularity and conductance, and retrieves via k-NN plus temporal contiguity. For write-time gating combine surprise/novelty (embedding distance to the nearest stored memory), LLM-judged importance (qwen-flash), and explicit user or emotional salience. These are computable cheaply online.

### 13.6 Forgetting without data loss
FSRS DSR model: Stability (days until retrievability falls to 90 percent), Difficulty (1 to 10), and Retrievability (power-law decay). FSRS-6 uses 21 trainable parameters (default weights trained on roughly 700 million reviews). Apply per memory to set the index-priority weight, never to delete; the raw record persists in WORM permanently. "User reverts a decision" equals reawakening equals resetting retrievability to high and boosting stability (like a review event); because the index entry was only down-weighted (the myonuclear-retention analogy), re-promotion is O(1).

### 13.7 Multimodal ingestion gaps
Covered in Section 11. The only true gap is modalities Qwen cannot embed; the fix is store-raw-plus-describe-plus-embed-the-description, with the raw object as ground truth.

### 13.8 Retrieval orchestration
Covered in Section 12. Reciprocal Rank Fusion plus qwen3-rerank, sub-second budget.

### 13.9 Scale and cost reality
text-embedding-v4 batches at most 10 texts per request. A lifetime of multimodal data is feasible in storage (OSS is cheap, with lifecycle to cold tiers), but embedding throughput and per-token LLM costs dominate. Dedup via content hash before embedding. Hackathon demo scope: ingest a bounded but multimodal corpus (a few thousand items across text, image, audio, video, PDF) to prove the pipeline end to end; present DiskANN and Tablestore billion-scale numbers as the production extrapolation. Use qwen-flash for salience and importance judging, and batch calling (about 50 percent discount).

### 13.10 Evaluation harness gaps
Covered in Section 14.

---

## 14. EVALUATION PLAN

### 14.1 The benchmarks
- **LoCoMo** (Maharana et al., ACL 2024): 10 conversations, about 300 turns across up to 35 sessions, about 1,540 QA (841 single-hop, 282 multi-hop, 321 temporal, 96 open-domain), scored by F1 plus LLM-judge. Documented flaws: conversations are only about 16,000 to 26,000 tokens (inside modern context windows), a missing-ground-truth category, multimodal captioning errors, and speaker-attribution errors.
- **LongMemEval** (Wu et al., ICLR 2025): 500 questions over five abilities (information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention); needle-in-haystack with about 115,000-token (S) to about 1.5M-token (M) histories; state-of-the-art long-context models drop 30 to 60 percent. The harder, more realistic test.
- Newer: LongMemEval-V2, BEAM (1 million to 10 million tokens, 100 conversations, 2,000 questions; deliberately unsaturated), MemoryAgentBench, MemoryArena, MemBench.

### 14.2 Fair baselines
Mem0 (Apache 2.0, self-hostable) and Graphiti (Apache 2.0, self-hostable; full Zep is SaaS). Run them through the identical harness.

### 14.3 The disputed-scores discipline
The Mem0-versus-Zep dispute (Section 4.3) is a cautionary tale. To be credible: fix the judge model and prompt across all systems, report variance over at least 5 runs (as Mem0's recalculation did), acknowledge the dispute openly, and pre-register the protocol. A benchmark claim that does not reproduce is a worse hole than no claim.

### 14.4 The novel signature metric (the headline result)
At fixed store size N, plot recall at k versus memory age and p95 latency versus memory age. A flat curve is the headline. This is rigorous, novel, and currently unmeasured by any competitor. Pre-register the protocol to avoid the benchmark-credibility trap. This is the metric that visually proves "acts as if the message was sent one second ago, no matter how old it is."

### 14.5 Targets
Beat Mem0 and Graphiti specifically on the temporal and multi-hop slices (where provenance and the bi-temporal graph win), and own the recency-independence curves outright (competitors report no number there). Aspirational reference points: Mem0 about 92.5 LoCoMo and 94.4 LongMemEval at about 6,900 tokens per query; Emergence about 86 percent LongMemEval; Letta about 83 percent LoCoMo (the highest verified open-source). BEAM is a serious aspirational target.

---

## 15. ARE ALL GAPS FILLED? HONEST ASSESSMENT

Short answer: every major architectural gap is identified and closed with a real, generally-available solution. A small set of residual risks remain. They have documented mitigations rather than perfect closure, and they are below in full so nothing is hidden. Overclaiming "zero risk, every gap perfectly closed" would be dishonest and would itself be a credibility risk with technical judges.

**Closed with real, verified solutions:**
- Lossless immutable storage: OSS WORM (Cohasset-certified). Closed.
- Multimodal capture and embedding across text, image, screenshot, PDF, audio, video, plus a fallback for un-embeddable modalities. Closed with real model IDs.
- Uniform-latency retrieval: HNSW and DiskANN with the correct age-independence framing. Closed.
- Verified reconstruction without confabulation: NLI entailment plus span-level support plus the immutable record as ground truth. Closed.
- Contradiction handling and provenance: bi-temporal invalidation (Graphiti pattern) plus the raw record. Closed.
- Forgetting without data loss: FSRS index-priority decay, reversible reawakening. Closed.
- Retrieval orchestration: approximate-nearest-neighbor plus in-app Personalized PageRank plus Reciprocal Rank Fusion plus qwen3-rerank, sub-second. Closed.
- Compute and scheduling: Function Compute plus ECS. Closed.
- Evaluation: LoCoMo plus LongMemEval with fixed judge, variance reporting, fair baselines, plus the novel recency-independence metric. Closed as a plan.
- Biological rigor: the REAL-versus-METAPHOR table, with weak analogies cut. Closed.

**Residual risks (mitigated, not perfectly closed):**
1. **Beijing-region constraint on qwen3-vl-embedding fusion mode.** This is the single most important deployment risk. Mitigation: run the vector pipeline in Beijing, or use the Singapore-available tongyi-embedding-vision (independent-only), or self-host Qwen3-VL-Embedding 8B. Verify your account's regional access before committing.
2. **Component 3 (Tolman-Eichenbaum factorization) has no production precedent.** Mitigation: ship the metadata-structure-code fallback as the real deliverable; cite the Tolman-Eichenbaum Machine as inspiration only.
3. **Recency-independence must be stated as age-independence at fixed N**, not as defeating the size-dependence of approximate-nearest-neighbor search. Overclaiming here is the fastest way to lose technical judges. Mitigation: the flat latency-versus-age curve at fixed N, stated precisely.
4. **All benchmark numbers in the field are contested.** Mitigation: pre-register, fix the judge model, report variance, run honest baselines, and treat every score (including ours) as an upper bound.
5. **Some pricing details (ASR, text-embedding-v4) were not numerically confirmed** from official pages during research. Mitigation: verify on the live Model Studio pricing page before budgeting. OCR pricing (qwen-vl-ocr, roughly 0.07 to 0.16 per million tokens internationally) is confirmed.
6. **The free developer tier ended April 15, 2026.** Mitigation: budget for paid usage (a one-time 1M-token-per-model trial only).
7. **NLI verification can miss some inconsistencies (granularity mismatch).** Mitigation: defense in depth (span-level plus QA-based checks) and the immutable record as the final arbiter.
8. **gte-rerank retirement (2026-05-30).** Mitigation: use qwen3-rerank from the start.
9. **Rapid Qwen model-ID churn in 2026.** Mitigation: confirm exact current model IDs on the live DashScope catalog at build time.

That is the complete picture. The design has no mock components, and the residual risks are operational and framing risks with concrete mitigations, not holes in the core architecture.

---

## 16. BUILD PLAN (PHASED TO JULY 9, 2026)

### Stage 1: Build the spine first (the differentiator, low-risk, all GA)
OSS WORM bucket with SHA-256 content addressing, the ingestion pipeline (qwen-vl-ocr, qwen3-asr-flash, DocMind, qwen3-vl-embedding), AnalyticDB-PG HNSW plus GDB graph, and the qwen3-max agent with tool calling. Demonstrate the two headline claims: uniform retrieval latency versus memory age, and lossless multimodal recall (100 percent exact retrieval of any stored raw item by ID or cue).

### Stage 2: Add the brain-inspired differentiators
NLI-verified reconstruction (premise is the WORM record), the bi-temporal contradiction engine (Graphiti-style invalidation), FSRS index-priority forgetting, Reciprocal Rank Fusion plus qwen3-rerank orchestration, and the EM-LLM surprise-based salience gate. Evaluate on LoCoMo and LongMemEval-S; target beating Mem0 and Graphiti on temporal and multi-hop categories.

### Stage 3: Consolidation, salience, and the signature metric
Implement the offline replay and consolidation job (Function Compute cron), the DSR reweighting, the Bayesian-surprise salience gating, and verified generative reconstruction. Define and report the recall-versus-age and latency-versus-age curves as the signature result. Ship Component 3 as the metadata-structure-code fallback.

### Deliverables for submission
Public repo with an open-source license, the Alibaba Cloud deployment proof (a code file plus a short recording), the architecture diagram (Qwen to backend to database to frontend), the roughly 3-minute demo video (lead with the recency-independence proof), the text description, the track ID, and optionally a build-journey blog post.

### Pitch framing
Lead with verifiability, immutability, and provenance (the clear wins). Use the immune-affinity-maturation mapping as the headline biological story. Present the REAL-versus-METAPHOR table to preempt skepticism. Open with the honest framing that photographic memory is a human myth made real in silicon.

### Thresholds that change the plan
- If qwen3-vl-embedding Beijing-only latency is unacceptable from your region, switch to self-hosted Qwen3-VL-Embedding 8B or tongyi-embedding-vision in Singapore.
- If in-app Personalized PageRank latency exceeds about 100 milliseconds at demo scale, cache results or fall back to pure approximate-nearest-neighbor plus rerank.
- If NLI verification adds too much latency, run it asynchronously at consolidation time.
- If LongMemEval scores trail Mem0 or MIRIX, emphasize the recency-independence and provenance metrics where competitors have no number at all.
- If graph construction cost dominates, serve approximate-nearest-neighbor immediately and build the graph asynchronously.

---

## 17. RISK REGISTER AND PREEMPTIONS (FOR THE JUDGES)

1. "Recency-independence is just nearest-neighbor search." Reframe precisely as age-independence at fixed N; show the flat latency-versus-age curve; contrast with lost-in-the-middle.
2. "Component 3 is vaporware." Present the metadata-structure-code fallback as the shipped version; cite the Tolman-Eichenbaum Machine as inspiration.
3. "The biology is hand-wavy." Show the REAL-versus-METAPHOR table and the cuts.
4. "Benchmarks are gamed in this field." Pre-register the protocol, fix the judge model, report variance, run Mem0 and Graphiti as honest baselines, and acknowledge the Mem0-Zep dispute openly.
5. "Beijing-only embedding limits deployment." Acknowledge it and show the regional fallback.
6. "How is this different from Mem0 or HippoRAG?" Lossless immutable substrate plus verified reconstruction plus bi-temporal provenance plus recency-independence, none of which they combine.

---

## 18. KEY SOURCES

**Neuroscience and cognitive science.**
- Teyler and DiScenna (1986), hippocampal memory indexing theory.
- McClelland, McNaughton, and O'Reilly (1995), Complementary Learning Systems, Psychological Review.
- Hafting, Fyhn, Molden, Moser, and Moser (2005), grid cells, Nature.
- Constantinescu, O'Reilly, and Behrens (2016), conceptual grid code, Science.
- Whittington, Muller, Mark, Chen, Barry, Burgess, and Behrens (2020), the Tolman-Eichenbaum Machine, Cell.
- Lisman and Jensen (2013), the theta-gamma neural code, Neuron.
- Ryan and Frankland (2022), forgetting as adaptive engram-cell plasticity, Nature Reviews Neuroscience.
- Lee, Nader, and Schiller (2017), an update on memory reconsolidation updating, Trends in Cognitive Sciences.
- Lisman and Grace (2005), the hippocampal-VTA loop; Moncada and Viola (2010), behavioral tagging, PNAS; Takeuchi et al. on noradrenaline and novelty.
- Bartol, Sejnowski, Harris, et al. (2016), synaptic capacity, eLife; Reber (2010), brain capacity estimate.
- Spens and Burgess (2024), a generative model of memory construction and consolidation, Nature Human Behaviour; 2025 follow-up framing it as compressive retrieval-augmented generation.
- Tse et al. (2007), schemas and accelerated consolidation, Science.
- Dresler et al. (2017), mnemonic training and memory athletes, Neuron.
- Parker, Cahill, and McGaugh (2006), hyperthymesia; LePort et al. (2012), HSAM neuroanatomy, Neurobiology of Learning and Memory.
- Stromeyer and Psotka (1970), the Elizabeth eidetic case, Nature; Merritt (1979), "None in a million," Behavioral and Brain Sciences.
- Treffert and Christensen (2005), inside the mind of a savant, Scientific American; Snyder, savant-skill induction via TMS.
- Fioriti, Kandel, et al. (2015), CPEB3 prion-like maintenance, Neuron; Sacktor, PKMzeta and LTP maintenance.

**Computational and mathematical.**
- Kanerva (1988), Sparse Distributed Memory.
- Ramsauer et al. (2021), "Hopfield Networks is All You Need," ICLR; Krotov and Hopfield, dense associative memories.
- Plate (1995), Holographic Reduced Representations; Smolensky (1990), Tensor Product Representations.
- Graves et al. (2014, 2016), Neural Turing Machines and the Differentiable Neural Computer.
- Behrouz, Zhong, and Mirrokni (2024), Titans: learning to memorize at test time (surprise metric and forget gate).
- FSRS / DSR model (the Free Spaced Repetition Scheduler, FSRS-6, 21 parameters); Settles and Meeder (2016), Half-Life Regression, ACL.
- Liu et al. (2024), "Lost in the Middle," TACL.

**Agentic memory systems.**
- Gutierrez et al. (2024), HippoRAG, NeurIPS; the 2025 follow-up "From RAG to Memory."
- Fountas et al. (2025), EM-LLM, ICLR.
- Das, Chaudhury, et al. (2024), Larimar, ICML.
- Packer et al. (2023), MemGPT / Letta.
- Park et al. (2023), Generative Agents, Stanford.
- Mem0 / Mem0g (2025), ECAI.
- Zep / Graphiti (2025).
- MIRIX (2025); MemOS / MemCube (2025); A-MEM; AriGraph (IJCAI 2024); RAPTOR; MemoryBank (2024).
- Surveys: "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers" (2026); "Memory in the Age of AI Agents" (2026); CoALA (2023).
- Benchmarks: LoCoMo (ACL 2024); LongMemEval (ICLR 2025); BEAM (ICLR 2026).

**Alibaba Cloud and Qwen (official Model Studio and product documentation).**
- qwen3-vl-embedding multimodal embedding API reference (modalities, dimensions, fusion mode, Beijing-region constraint).
- text-embedding-v4 (flexible dimensions).
- qwen3-rerank and qwen3-vl-rerank (gte-rerank retirement notice, 2026-05-30).
- Tablestore KNN (DiskANN); AnalyticDB for PostgreSQL (HNSW); Zvec.
- OSS WORM (BucketWorm, ObjectWorm, Cohasset certification).
- GDB (Gremlin and TinkerPop).
- DashScope LLMs (qwen3-max, qwen-plus, qwen-flash, qwen-long); Qwen-Agent; MCP support.
- Function Compute (scheduled and HTTP triggers); ECS and ACK.
- Ingestion: qwen-vl-ocr, qwen3-asr-flash, qwen3-asr-flash-filetrans, paraformer-v2, DocMind, qwen-doc-turbo, qwen3-vl-plus.

> Verify every model ID, regional availability, and price on the live DashScope and Model Studio catalog at build time, since the 2026 Qwen release cadence is rapid. Treat every benchmark number, including any this project produces, as an upper bound, and always report methodology and variance.

---

## 19. APPENDIX A: CONSOLIDATED BENCHMARK SCOREBOARD (EVERY NUMBER IN ONE TABLE)

Every benchmark figure surfaced across this research, gathered into one reference. Treat all of these as upper bounds: most are vendor self-reported, several are disputed (Section 4.3), and LLM-judge scores swing about 10 points on judge-model choice alone. "Indep." means an independent reproduction differs materially from the self-report.

| System | LoCoMo | LongMemEval | Cost / footprint | Other reported results |
|---|---|---|---|---|
| **Mem0** | 91.0-92.5% self-reported; ~58-66% indep. | 94.4% (Apr 2026 algo) | ~6,900 tokens/query; 1,764 tokens/conversation; p95 search 0.200s | LoCoMo 67.13% LLM-judge in the Mem0 ECAI paper; +29.6 temporal, +23.1 multi-hop; BEAM 64.1 (1M) / 48.6 (10M) |
| **Zep / Graphiti** | 58.44% ±0.20 (Mem0's recalc) vs 75.14% ±0.17 (Zep's J-score, down from ~84% original) | ~63.8% (one comparison) | >600,000 tokens/conversation; many LLM calls/episode; post-ingestion retrieval lag | Sub-second retrieval target |
| **Letta / MemGPT** | ~74-83% (highest *verified* open-source ~83%) | - | Manual tiered paging | - |
| **HippoRAG / HippoRAG 2** | - | - | 10-30x cheaper, 6-13x faster than IRCoT | Original +up to 20% multi-hop QA; HippoRAG 2 ~+7% over embedding models on associative tasks |
| **EM-LLM** | - | - | Operates in-context (to 10M tokens) | +4.3% overall vs InfLLM, +33% on PassageRetrieval; +30.5% vs NV-Embed-v2 RAG on LongBench |
| **Larimar** | - | - | Capacity-bounded | ~100% rewrite accuracy to 512 slots, ~82% at 1024 |
| **MemOS / MemCube** | - | - | ~35% token savings | Framework; cross-model sharing unsolved |
| **MIRIX** | SOTA LoCoMo | - | Multimodal | Real multimodal screenshot benchmark; frontier threat |
| **Emergence AI** | - | ~86% | - | - |
| **SYNAPSE** | +23% multi-hop, 95% token reduction | - | - | Spreading activation + lateral inhibition + temporal decay |
| **TiMem** | 75.3% | 76.9% (S) | - | Five-level Temporal Memory Tree |
| **A-MAC** | F1 ~0.583 | - | Latency -31% | Admission control over utility/confidence/novelty/recency/type |
| **Memory-R1** | - | - | Trained on as few as ~152 QA pairs | RL-learned add/update/delete/retrieve |
| **HINDSIGHT** | ~89.6% | - | - | Biomimetic |
| **Generative Agents** | - | - | Hand-tuned recency x importance x relevance | No provenance or contradiction handling |
| **Qwen3-VL-Embedding** (our embedder) | - | - | - | **#1 on MMEB-V2 at 77.8 as of Jan 2026** |

Benchmark suites referenced: **LoCoMo** (Maharana et al., ACL 2024; ~1,540 QA; ~16-26K-token conversations; documented flaws). **LongMemEval** (Wu et al., ICLR 2025; 500 QA; ~115K-token S to ~1.5M-token M; SOTA long-context models drop 30-60%). **BEAM** (ICLR 2026; 1M-10M tokens; 100 conversations; 2,000 QA; deliberately unsaturated). Plus LongMemEval-V2, MemoryAgentBench, MemoryArena, MemBench.

**Our two targets:** (1) beat Mem0 and Graphiti specifically on the temporal and multi-hop slices, where provenance and the bi-temporal graph win; (2) own the recall-versus-age and latency-versus-age curves outright, where no competitor reports any number.

---

## 20. APPENDIX B: PRIMARY-SOURCE LINKS (VERIFIED THIS RESEARCH)

The canonical citation list is Section 18. Below are the direct source URLs verified during this research pass, so claims can be checked at the source. Everything else in Section 18 is a standard, locatable academic citation.

- **MIRIX: Multi-Agent Memory System for LLM-Based Agents**. https://arxiv.org/pdf/2507.07957
- **The Tolman-Eichenbaum Machine** (Whittington et al.). bioRxiv: https://www.biorxiv.org/content/10.1101/770495v2.full ; published in *Cell* vol. 183, Nov 25 2020, DOI 10.1016/j.cell.2020.10.024
- **Governing Evolving Memory in LLM Agents (Stability and Safety Governed Memory, SSGM)**. https://arxiv.org/pdf/2603.11768
- **Mem0 / Mem0g** (the ECAI 2025 paper with the Zep 600,000-token and footprint comparisons). arXiv:2504.19413
- **Lost in the Middle: How Language Models Use Long Contexts** (Liu et al.). *Transactions of the ACL* (TACL) 2024, vol. 12, pp. 157-173

> Verify every model ID, regional availability, and price on the live DashScope and Model Studio catalog at build time. The Mem0-versus-Zep dispute (getzep/zep-papers issue #5 and Zep's rebuttal blog) is the field's clearest reminder to fix the judge model and report variance.

---

## 21. APPENDIX C: EXACT-QUOTE EVIDENCE LOCKER (FOR THE DEMO AND Q&A)

Verbatim, attributable evidence to deploy when a judge pushes on a specific claim. Use sparingly and accurately.

**On the long-context failure we beat (the contrast that justifies recency-independence).** Liu et al. (TACL 2024) found performance is "highest when relevant information occurs at the very beginning (primacy bias) or end (recency bias)... and significantly degrades when models must access information in the middle." GPT-3.5-Turbo's mid-context multi-document QA accuracy fell *below* its 56.1% closed-book baseline. This is the precise phenomenon a flat recall-versus-age curve refutes.

**On the structure/content factorization (Component 3 inspiration, stated honestly as inspiration).** Whittington et al. (*Cell* 2020): "medial entorhinal cells form a basis describing structural knowledge, and hippocampal cells link this basis with sensory representations... Separating structural codes... from sensory codes allows generalization over environments sharing the same structure." We ship the metadata-structure-code approximation and cite this as inspiration, never as a literal implementation.

**On competitor cost (the efficiency argument).** The Mem0 ECAI 2025 paper measured that "Zep's memory footprint exceeds 600,000 tokens per conversation (versus 1,764 for Mem0)." In the same study Mem0 reports 67.13% LLM-as-Judge on LoCoMo at p95 search latency 0.200s.

**On the benchmark-credibility discipline (why our numbers are trustworthy).** Mem0's recalculation restricted to "the first four validated LoCoMo categories" and averaged "over ten independent runs," yielding "Zep's mean accuracy is 58.44% ± 0.20." Zep's rebuttal reports "an 75.14% +/- 0.17 J score." The lesson we adopt: fix the judge model and prompt, report variance over at least five runs, and pre-register the protocol.

**On forgetting math (why our forgetting is principled, not ad hoc).** FSRS-6 uses exactly 21 trainable parameters (the prior v4.5 used 17), with default weights trained on roughly 700 million reviews from about 10,000 Anki users, using a power-law retrievability curve that fits human memory better than Ebbinghaus's exponential.

**On salience (why our write-gate is real).** EM-LLM (Fountas et al., ICLR 2025) "outperform[s] the state-of-the-art InfLLM model with an overall relative improvement of 4.3% across various tasks, including a 33% improvement on the PassageRetrieval task," using Bayesian surprise (next-token negative log-likelihood) for event boundaries.

---

## 22. APPENDIX D: ONE-SCREEN ARCHITECTURE DESCRIPTION (FOR THE REQUIRED DIAGRAM)

The submission requires an architecture diagram showing how Qwen Cloud connects to backend, database, and frontend. This is the text spec to draw from; it names only real, generally-available services.

**Layer 0, Clients / frontend.** A web UI (any framework) talks to the backend over HTTP. The 3-minute demo drives this UI.

**Layer 1, Backend / orchestration (FastAPI on ECS or ACK).** Two paths.
- *Wake path (per turn):* receive event -> salience gate (qwen-flash) -> write raw blob to OSS WORM (SHA-256 key) -> embed (qwen3-vl-embedding for multimodal, text-embedding-v4 for text) -> upsert vector into AnalyticDB-PG (HNSW) and Tablestore (DiskANN, the scale story) -> extract entities/edges (qwen-plus) and write bi-temporal edges into GDB -> retrieve (ANN top-100 with bi-temporal filter -> in-app Personalized PageRank over GDB -> Reciprocal Rank Fusion -> qwen3-rerank top-10) -> assemble context -> generate (qwen3-max) -> verify against the OSS record (NLI entailment) -> reconsolidate (strengthen/update/suppress).
- *Sleep path (scheduled, Function Compute cron):* selective replay of high-salience episodes -> generative consolidation into a semantic store (Qwen3 thinking-tier) verified against raw traces -> link co-activated memories -> FSRS/DSR index-priority decay (never deletes OSS content).

**Layer 2, Storage and data services (all Alibaba Cloud).**
- **OSS with WORM** = the immutable lossless substrate (Component 1). Ground truth and provenance.
- **AnalyticDB for PostgreSQL (HNSW)** = the primary vector index for the demo; **Tablestore (DiskANN)** = the billion-scale production index (Component 2 vectors).
- **GDB (Gremlin)** = the bi-temporal knowledge graph; Personalized PageRank runs in-app (Components 2 and 7).
- A small relational/state store holds per-memory FSRS/DSR state and structure-codes (Components 3, 4, 5).

**Layer 3, Qwen Cloud / Model Studio (the intelligence).** qwen-flash (salience), qwen3-vl-embedding and text-embedding-v4 (encoding), qwen-plus (extraction and the contradiction/verification judge), qwen3-rerank and qwen3-vl-rerank (final ranking), qwen3-max (generation), a Qwen3 thinking-tier model (consolidation), and the ingestion models (qwen-vl-ocr, qwen3-asr-flash and -filetrans, DocMind, qwen-doc-turbo, qwen3-vl-plus). Model-per-cognitive-function is itself the Technical-Depth story.

**The single arrow to emphasize in the diagram:** every generated answer points back to an immutable OSS object it was verified against. That arrow, the provenance/verification link, is the thing no competitor's diagram has.

---

*End of dossier. One file, all research. Build the spine first (Section 16, Stage 1), ship the differentiators (Stage 2), then the consolidation loop and the signature recall-versus-age metric (Stage 3). Lead the pitch with verifiability, immutability, and provenance; lead the biology with immune affinity maturation; and preempt skepticism with the REAL-versus-METAPHOR table and the honest residual-risk register in Section 15.*
