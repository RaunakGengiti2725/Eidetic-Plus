# Weakness queue — ranked by combined holdout leverage

Rows reference convN-rowM shapes only (leakage audit). Discipline per item: offline
on-store proof → dev subset → queue for holdout pin. Never skip a step.

| # | Class | Evidence (windows 1–4) | Fix class | Risk | Status |
|---|-------|------------------------|-----------|------|--------|
| 1 | Reader partial-lists / multi-hop subset | mem0 7/8 vs our 4/8 (w3); subsets judged wrong (conv8 rows) | Plural-list scaffold (shipped) + enumerator claims + city-visit write path | Low — additive claims | Slice 5 measures; do not claim before scoreboard |
| 2 | Single-hop parity on some windows | w2 10/21, w4 12/20 vs mem0 15/20 | Tier-1 claim coverage growth via collector deletion (P1 tranche); stated-age/latest-value paths shipped | Med — deletion needs matrix tests | Tranche 1 this branch |
| 3 | Temporal week-window vs exact date | conv9 rows w4; integrity OK (windows verify honestly), judge wants the day | WRITE-TIME event claims (P2). NO more read-time regex/cluster passes — 2 reverts + 1 measured no-op on record | High if read-time; low if write-time | Specced (P2), queued |
| 4 | Event instance: dateless first report / ambiguous second mention | conv9 shop, conv1 Gina shapes | Same P2 unit (lemma+obj_head instance tags at extraction) | Low | Queued behind P2 |
| 5 | Teaser echoes (generation-side) | 'gotten some cool deals' shape survivors | Form floor caught the filler variants; remaining need enumerator coverage, NOT NLI-as-form-arbiter | Low | Partially closed; watch slice 5 |
| 6 | Abstention latency (16-26s) | CLOSED AS MEASURED-INFEASIBLE pre-reader: feature study on dev-40 shows NO separator (abstained coverage 0.46-0.67 vs answered 0.54-0.74, full overlap; counts and agreement scattered both sides). The abstention decision requires reader+NLI evidence -- the latency is the price of verify-or-abstain under the cascade. Revisit ONLY with a fundamentally new signal (reflex-index features), never coverage re-thresholds | — | closed with data |
| 7 | Cross-sentence association (pic-vs-show residual, oblique witness recall) | conv7 amulet shape; Tokyo class | Graph/ActivationField retrieval spike behind flag; captions already widen evidence | Med — new retrieval channel | Design note only tonight |
| 8 | Person-name echo ("Wow, Caroline" shape) | conv0 row w4 | Left open deliberately — who-question flips risk; needs main-wh + person-role check | Med | Documented, not attempted |
| 9 | Gold-date-skew / gold-ambiguity | question date ≠ dataset session date; 3-vs-4-weeks rounding | NOT fixable honestly; date discipline stays strict | — | Recorded as dataset noise |

## r5 additions (eidetic-phase forensics, 2026-07-04)

| # | Class | Evidence | Fix class | Status |
|---|-------|----------|-----------|--------|
| 10 | Gerund-object instance heads | VERIFIED on 149 real tagged claims (fresh dev store): noise mostly inert via head-tie gate | head stops shipped for conjunctions/temporals/pronouns/numbers (b2cd95789) | closed |
| 11 | Month-of-achievement questions | 'in which MONTH's game did X achieve...' answered with a bare later date | superlative+month composition; needs month aggregation over scored-events, not another date pick | design note |
| 12 | Greeting/filler junk answers | 'Hey Sam', filler-item lists | CLOSED same-day (2a8e7a10f), matrix zero flips | closed |

Temporal 6/11 VW on r5 are {lemma-miss, week-window} tagged -- the SHIPPED P2 classes;
r5 ran the pre-P2 build. Slice 6 on the merged SHA is the measurement, per SLICE6_PLAN.

## Revert history (institutional memory — do not retry these)

- **Event-instance date clustering** (3 attempts on the same class):
  1. First-report ceiling — bare-year junk passes any ceiling; REVERTED uncommitted.
  2. Date-proximity clustering — every tie-break traded the Gina row for the
     time-invariant sidecar (repeated-event latest semantics vs distinct-event earliest);
     REVERTED uncommitted.
  3. Question-verb lemma families — SHIPPED (c796a0463); helps only when the atom
     carries the verb. Remainder is write-time work.
- **Non-entity event-term floor** — killed one wrong-instance, broke two legit tests
  (shared head nouns are what good AND bad atoms carry); REVERTED uncommitted.
- **Per-collector junk gating** (earlier wave) — whack-a-mole, 2 regressions; REVERTED,
  replaced by verify-layer form floors.

## Measurement invariants (all fixes pass ALL of these before commit)

1. Full suite green (1320 at branch point).
2. Wave-F replay byte-identical (18 agreement rows).
3. Four-slice form-floor matrix: kills may rise, correct-answer flips must stay ZERO.
4. Time-invariant + composition + lacuna sidecars green.
5. Leakage audit green (no sample IDs outside run artifacts).
