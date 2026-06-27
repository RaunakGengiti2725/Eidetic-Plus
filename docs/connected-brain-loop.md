# Connected Brain Loop

Status: **all 8 phases of the [plan](/Users/raunakgengiti/.cursor/plans/connected_brain_loop_6448de96.plan.md) implemented**, in the order the plan prescribes (improvement gates + brain spine first, then connect outward). Everything is offline, deterministic, and default-OFF; with the flags off the read/write path is byte-identical to before and the full offline suite passes. No live benchmark lift is claimed (quota-blocked -- see the bottom).

The organizing discipline throughout: **a connection only ships if it (a) is gated so flags-off preserves the baseline, (b) recovers a real miss or adds real diagnostics, and (c) is explainable through a RecallTrace / EvidencePacket / BrainEvent.** A memory surfaced invisibly to the brain does not pass.

## The brain spine (Phase 1.5)

Observation-only contracts that make every subsystem speak one language. Nothing here ranks, selects, or generates -- it reads already-computed state.

| Contract | Module |
|---|---|
| `RecallTrace`, `EvidencePacket`, `BrainEvent`/`BrainEventType`, `QualityGateResult` | [`eidetic/models.py`](../eidetic/models.py) |
| `BrainEventLog`, `build_evidence_packets`, `QualityGate` | [`eidetic/brain.py`](../eidetic/brain.py) |

## Phases

| Phase | What landed | Key surface |
|---|---|---|
| 1 -- Unified lifecycle | `LifecycleController` (one wake/sleep/idle/repair path for API + MCP); `engine.sleep()` = consolidate_pending → dream → optional LLM summaries | [`eidetic/lifecycle.py`](../eidetic/lifecycle.py) |
| 1.5 -- Spine | RecallTrace / EvidencePacket / BrainEvent / QualityGate | `brain.py`, `models.py` |
| 2 -- Dream → recall | co-activation channel (`COACTIVATION_CHANNEL`); gist provenance into proof; inferred edges stay labeled hints | `retrieval.py` |
| 3 -- Idle optimization | channel-win ledger (`record_channel_wins`/`channel_win_stats`); `engine.idle_tick()`; `connection_effectiveness()` | `engine.py`, `lifecycle.py` |
| 4 -- Memory typing | classify type on ingest + soft type-priority retrieval prior (`MEMORY_TYPING`) | `engine.py`, `retrieval.py`, `memory_types.py` |
| 5 -- Guarded repair | `apply_proposals` (INSERT/MERGE via immutable ingest + supersession; dry-run default, `DREAM_REPAIR_APPLY`); `memory_autopsy()` offline failure classifier | `dreaming/repair.py`, `engine.py` |
| 6 -- Proof surface + parity | `explain_candidate` ("why this memory?"); MCP tools `sleep`/`recall_trace`/`memory_autopsy`/`prove_age_independence`/`brain_health_score`; matching `/api/*` routes | `engine.py`, `mcp_server.py`, `api.py` |
| 7 -- Connectivity | tests proving each subsystem feeds another with invariants held | [`tests/test_brain_connectivity.py`](../tests/test_brain_connectivity.py) |
| 8 -- Quality gates | `brain_health_score` (local composite, not a benchmark); synthetic improvement gates; updated-fact supersession invariant | `engine.py`, `tests/` |

## Flags (all default `0`)

`RECALL_TRACE`, `BRAIN_EVENTS`, `COACTIVATION_CHANNEL` (`RRF_W_COACT`), `MEMORY_TYPING` (`TYPE_PRIOR_WEIGHT`), `DREAM_REPAIR_APPLY`. See [`.env.example`](../.env.example).

## Invariants held (and tested)

- **Observation-only spine.** `retrieve()` returns identical candidate ids + order with `RECALL_TRACE` on vs off; `prove_answer(ans)` with no trace is byte-identical to the legacy dict (recall-path keys are strictly additive).
- **Raw store immutable.** A dream pass leaves every raw record byte-identical; guarded repair apply is additive-immutable ingest, never deletion (dry-run by default, and a no-op even with `apply=True` while `DREAM_REPAIR_APPLY` is off).
- **Updated facts supersede, never delete.** Bi-temporal: the old edge is closed (`invalid_at` set), the new value goes active, full history is retained.
- **Scope isolation.** No channel (gist, co-activation, type prior) leaks across namespaces.
- **Age-independence.** No new connection ranks on FSRS/age; the structure channel is cyclic-time-only; `QualityGate.no_age_bias` reads the `prove_age_independence` slopes.
- **Integrity wall.** `BrainEventLog` and the channel-win ledger are in-memory and NON-LEARNING. Learners read only `FeedbackBuffer.sample()` (dev-only). Before events are ever persisted or fed to a learner they must route namespaces through `feedback.is_benchmark_namespace` exactly like `FeedbackBuffer`.

## Tests

`test_brain_spine`, `test_proof_paths`, `test_brain_synthetic`, `test_lifecycle`, `test_typing_coordinator`, `test_repair_apply`, `test_brain_connectivity`, `test_brain_health` (+ MCP parity in `test_mcp_server`). 274 offline tests pass.

## Explicitly deferred (honest)

- **Multimodal dense path** (`embed_image` for image memories): needs a live model to embed/measure; not built in this pass.
- **Anti-regression weight promotion** through the EvolveMem guard for learned fusion weights: the guard exists (`bench/guard.py`); wiring learned-weight promotion through it is left to the live-measurement step.
- **Markov → prefetch warm-up bridge**: `predict_next_signatures` and `build_prefetch` both exist; binding predicted next-signatures to prefetched contexts is a latency optimization deferred until measurable.

## Not yet measured

No live recall/latency lift is claimed. The free-tier DashScope quota is exhausted (HTTP 403), so the five live-model tests (real NLI / answer / visual) cannot run and no end-to-end numbers were produced. Every connection here is built, invariant-checked, and offline-tested; measuring lift on the held-out dev split is the next step once quota is restored.
