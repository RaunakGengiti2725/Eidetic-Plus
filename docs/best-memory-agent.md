# Best Memory Agent

Status: **the Best Memory Agent plan is implemented** — Phase R hardening, the Phase 0 connected-brain correctness fixes, and the reliability/affect phases (conflict, abstention, affect salience, verified-helpful, temporal, scratchpad, demo surface). Built in the plan's own critical-path order (R → 0 → 1 → 2 first, fully verified offline; then the upside phases, lean and flag-off). Committed wave-by-wave on branch `connected-brain-loop` for bisectability.

The discipline throughout: **no mocks** (real DashScope, fail loud); **no fakes** (a number that doesn't reproduce doesn't exist); every new feature **default-OFF and baseline byte-identical**; **age-independence** preserved (audited offline); **nothing promotes to default-on** under the current quota block.

## Phase R — no-fakes hardening

| Piece | Where |
|---|---|
| `preflight()` doctor: one real call per capability; quota vs dead/auth/bad-model; document optional | [`eidetic/doctor.py`](../eidetic/doctor.py), `python -m eidetic.doctor`, `/api/preflight`, MCP `preflight` |
| Real bug caught + fixed: `read_document` hardcoded `qwen-long` (DOC_MODEL was dead config) | `dashscope_client.py`, `config.py` |
| `FeatureNotImplementedError` for the experimental stubs (memory-manager, debate) | [`eidetic/errors.py`](../eidetic/errors.py) |
| `engine._degraded`: hot-path `ModelCallError` logged at WARNING, never swallowed | `engine.py` |
| Key-gated `ingest → ask → prove` smoke test (skips cleanly without a working key) | [`tests/test_smoke_real.py`](../tests/test_smoke_real.py) |

Live doctor result on the configured key: **embed + multimodal real, chat/rerank quota-limited, doc-reading needs account access** — all surfaced honestly, never a fake green.

## Phase 0 — connected-brain correctness

`ask()` calls `lifecycle.after_recall` (channel-win telemetry on the product path); brain telemetry is **scope-isolated** (`_channel_wins`/`BrainEventLog.counts`/`brain_health_score` by namespace); semantic cache is **bypassed when `RECALL_TRACE`/`BRAIN_EVENTS` are on** (no stale proof/health); MCP `consolidate` is an alias of the unified `sleep`; HTTP single read is **scope-safe** (`get_record_in_scope`); `/api/ask` has **proof parity** with MCP.

## Reliability + affect phases (all flag-gated, default off)

| Phase | What | Flag |
|---|---|---|
| 1 Conflict | deterministic latest-valid resolution: validity window (`valid_at`/`invalid_at`/`as_of`), `created_at` serial tiebreak, abstain-when-none, missing-`valid_at` fails loud, supersession chain in the note | `CONFLICT_RESOLVER` |
| 2 Abstention | confidence = entailment + coverage + **structural** channel-agreement + proof-completeness; `pick_tau` dev-calibration | `ABSTENTION_V2` |
| 3 Affect salience | `score_affect` (importance/arousal/valence) + emphasis cues → age-free `affect_salience`; bounded retrieval boost + FSRS S0 + replay | `AFFECT_SALIENCE` |
| 4 Verified-helpful | per-memory `verified_helpful_count` on confirmed citations → bounded (capped) salience signal | `AFFECT_W_HELPFUL` > 0 |
| 5 Temporal | `event_chain` chronological context for order/sequence/'what changed after X' | `EVENT_CHAIN_CONTEXT` |
| 6 Scratchpad | derived high-salience verified ACTIVE facts as a context channel; links raw hashes; superseded facts expire | `SCRATCHPAD` |
| 7 Demo surface | `salience_explanation` ('why I remember this strongly') + scratchpad on MCP/API; proof tree, recall paths, age curve already live | — |

## Invariants held (and tested)

- **Baseline byte-identical** with every new flag off (each feature fully guarded).
- **Age-flatness (make-or-break, audited offline).** The salience retrieval boost is `lam * salience * max_fused`; `salience`/`affect_salience` carry **no** `valid_at`/age/FSRS term, so two equal-salience memories a billion seconds apart get an **identical boost delta**; salience (not age) flips ranking. Verified-helpful is **bounded** (saturates at cap). FSRS S0 coupling touches scheduling/replay only, never ranking. (`test_affect_salience.py`, `test_verified_helpful.py`.)
- **Deterministic conflict freshness** — Python decides; the model only extracts semantic candidates; missing `valid_at` fails loud.
- **Raw store immutable** — verified-helpful/salience mutate index state only; scratchpad is read-only and hash-linked; nothing deletes raw.
- **Dev/test integrity wall** — `pick_tau` is dev-split-only; channel-win ledger + brain events stay non-learning.
- **No fakes** — doctor makes real calls, stubs fail clean, no silent `ModelCallError`, smoke skips cleanly.

## Tests

+~38 offline tests across the phases (`test_doctor`, `test_phase0_scope_isolation`, `test_api_scope`, `test_conflict_resolution`, `test_abstention`, `test_affect_salience`, `test_verified_helpful`, `test_temporal_indexing`, `test_scratchpad`, plus MCP parity). **316 offline tests pass.** The only failures are the 5 pre-existing live-model tests (HTTP 403 quota); the key-gated smoke test skips cleanly.

## Not measured (honest)

No live recall/conflict/abstention/temporal **lift** is claimed. The free-tier DashScope quota is exhausted for the chat/rerank tiers (the doctor confirms it), so the EvolveMem-guard dev A/B that the plan requires before flipping any default cannot run. Every borrowed number (LUFY +17%, Memory-R1 +48% F1, deterministic CR ~100%, LiCoMemory +15.9pp) remains a **hypothesis to test**, never an achievement. The mechanisms are built, invariant-checked, and offline-tested; measuring lift on the held-out dev split is the next step once quota is restored.

## Explicitly not built (per the plan)

RL memory manager (GRPO/PPO), any LUFY-style salience-driven deletion (Eidetic-Plus is lossless), quantum features, and automatic repair-apply by default before guard approval.
