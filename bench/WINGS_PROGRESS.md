# Wings progress — problem memory + operational truth

Branch `feature/acceleration`. All flag-free additive surface; no core path changed.

## Wings 7: collective problem memory (SHIPPED — e076014a2)

A problem is an immutable record chain: every update is a NEW revision for the same
problem_id, folded deterministically into current state (later scalars override, lists
append, hypotheses merge by id latest-wins). Bitemporal by construction —
`recall_problem(as_of=...)` replays the war room as it stood at any moment.

MCP surface: `remember_problem`, `update_problem`, `add_hypothesis`,
`resolve_hypothesis`, `recall_problem`. Hypothesis evidence refs are memory_ids
validated against the scope (foreign/missing refs fail loud) so hypotheses prove through
the same citation machinery as answers.

Demo via MCP (any host):
1. `remember_problem(goal="Checkout latency spikes above 2s at peak", blockers=["no staging repro"])`
2. `add_hypothesis(problem_id, "Connection pool exhaustion under burst traffic")`
3. `update_problem(problem_id, status="investigating", handoffs=["night shift: check pool metrics"])`
4. `resolve_hypothesis(problem_id, hypothesis_id, "confirmed", rationale="pool at 100% during every spike")`
5. `update_problem(problem_id, status="resolved", decisions=[{"choice": "raise pool size to 64", ...}])`
6. `recall_problem(problem_id, as_of=<t3>)` → the state mid-investigation, hypothesis still proposed.

Tests: `tests/test_wings_problem_memory.py` — lifecycle + as_of replay, query recall +
scope isolation, evidence validation.

## Wings 8: witness scaffold (SHIPPED — e9452e7b5)

`add_witness(problem_id, path, note)` ingests the file losslessly into the
content-addressed substrate and appends a revision carrying
{memory_id, content_hash, raw_uri, note}. `get_raw(content_hash)` returns the bytes
byte-identical; `substrate.verify` re-hashes them (same tamper check as the proof
surface). Witness memory_ids serve as hypothesis evidence refs unchanged.

E2E test: log-file fixture → witness in folded state → byte-identical raw + hash
verification → hypothesis citing the witness.

## Deferred (honest)

- Write-time extraction of problem-shaped atoms from free conversation ("we decided X
  because Y" → decision revision) — rules-first sketch exists in the P6 proposal;
  not shipped tonight.
- recall() answering natural-language questions FROM problem state with citations into
  revisions (currently recall_problem returns structured state; the reader path sees
  revision records as ordinary memories).
- Witness summary atoms (image caption / log digest at ingest) — remember_file already
  describes files when the model key is set; not wired into the witness note.
- Cross-problem queries ("what did we decide about pools anywhere?").

## Wave 2 addition: ask_problem (22ce903e8)

NL questions against war-room history through the SAME verify-or-abstain path; citations
marked revision-backed vs general memory; folded state + as_of replay in the response.
Deferred list unchanged otherwise (write-time problem-shaped extraction remains the next
integration).
