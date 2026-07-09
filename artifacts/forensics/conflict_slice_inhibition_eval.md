# Conflict-slice inhibition evaluation (polar/inhibitory edges) — 2026-07-09

Goal item: "Inhibitory/polar graph edges evaluated on conflict-heavy holdout slice."
Evaluated BEFORE building, per operating rules. Verdict below: **defer the build** — the
measured gap is not where the polar-edge proposal assumes.

## Slice

LME-S r1 window (`artifacts/lme_s_r1_codex`), the 4 knowledge-update questions — the
conflict-heavy class (a stated fact is later UPDATED; answering with the stale value is the
inhibition failure). Small n, stated plainly; the structural findings below do not depend on
score noise.

## Measured results

| path | knowledge-update |
|---|---|
| whole-export NotebookLM free read (`notebooklm_freetier.judged.json`) | **0/4** |
| retrieval-guided free read (`rgi_scored.json`, judge field `rg`) | **4/4** |

Probe (`conflict_inhibition_probe.json`, offline, cached embeddings):

1. **Retrieval ranking is NOT the failure.** On all 4 questions the record carrying the
   gold (current) answer ranks **#1** in eidetic's own retrieval. No superseded record
   outranks it (0/4). An inhibitory re-ranker would change nothing on this slice.
2. **Record-level supersession is 0 by design.** `record.invalid_at` marks explicit
   retraction only; benchmark ingest never sets it. The write-once record layer is not
   where conflict state lives.
3. **Edge-level supersession (the existing inhibitory signal) is PATCHY on this ingest:**
   g27: 161 edges / **43 closed** (bi-temporal contradiction handling fires);
   g15: 15 edges / 0 closed; g0 and g8: **0 edges extracted at all**. Where extraction
   produces triples, `graph.add_fact` already closes contradicted edges — the polar
   support/inhibit distinction EXISTS; its coverage is bounded by extraction coverage.
4. **The failure mode of whole-export is burying, not mis-ranking:** both stale and
   current statements are exported as equally-live record bodies; Gemini picks the wrong
   one when the update is buried (0/4). Focused retrieval leads with the current record —
   the shipped retrieval-guided path already solves the slice (4/4).

## Verdict

- The product path (retrieval-guided, now `notebooklm_recall` / `recall_routed` tier 2)
  **already passes the conflict slice**; whole-export is no longer the default path.
- Explicit inhibitory claim edges would (a) duplicate the closed-edge machinery that
  already exists, and (b) be capped by the real bottleneck: **extraction coverage**
  (2 of 4 conflict namespaces got zero triples). If conflict handling regresses on a
  bigger slice, the lever is extraction coverage + surfacing closed-edge history in
  per-record exports (a header line "fact X updated later — see record Y"), not a new
  edge type.
- Re-evaluate only with a bigger conflict slice (LME-S knowledge-update across multiple
  windows) if whole-export ever returns as a product surface.

Evidence: `artifacts/lme_s_r1_codex/conflict_inhibition_probe.json`, probe script
reproducible via the same logic (retrieval top-8 vs store supersession state).
