# Dominance Failure Scoreboard

No frozen holdout head-to-head scoreboard was produced because the dev ablation gate failed closed.
This file is an explicit failure artifact, not release evidence.

| evidence | result |
|---|---:|
| Holdout leakage audit | PASS |
| SMQE claim coverage synthetic sidecar | PASS |
| SMQE synthetic invariant sidecar | PASS |
| SMQE planner invariant sidecar | PASS |
| Dev ablation | FAIL |
| Slice-invariant eval | NOT RUN |
| Frozen holdout head-to-head | NOT RUN |
| Snap-back audit | NOT RUN |

Dev ablation failure:

| check | value |
|---|---:|
| full verified accuracy | 55.0% |
| metabolism delta | +0.0 pp |
| region delta | -10.0 pp |
| affect delta | +15.0 pp |
| forgetting cost ratio | 1.000063 |
| forgetting accuracy regression | +10.0 pp |

Diagnostic dev probe after strict SMQE verification:

| metric | value |
|---|---:|
| subset | 20 |
| raw correct | 15/20 |
| verified correct | 13/20 |
| verified wrong | 5/20 |
| abstained | 0/20 |

