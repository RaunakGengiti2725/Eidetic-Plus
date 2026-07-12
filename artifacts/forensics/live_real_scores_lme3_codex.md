# Benchmark Failure Forensics

Rows inspected: 9
Failures/abstentions/errors: 5

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| abstention | 3 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |
| retrieval_miss | 2 | raise lexical/graph/event recall depth; inspect candidate_memory_ids and query parsing |

## Categories

| category | buckets |
|---|---|
| single-session-user | abstention: 3, retrieval_miss: 2 |

## Consolidation Health

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred |
|---|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 3 | 148 | 68 | 68 | 137 | 0 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | single-session-user | 118b2229 | abstention | How long is my daily commute to work? | 45 minutes each way | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | single-session-user | 51a45a95 | abstention | Where did I redeem a $5 coupon on coffee creamer? | Target | I don't have enough verified evidence in memory to answer that confidently. |
| rag-full | single-session-user | e47becba | abstention | What degree did I graduate with? | Business Administration | I do not have that in memory. |
| rag-full | single-session-user | 51a45a95 | retrieval_miss | Where did I redeem a $5 coupon on coffee creamer? | Target | I redeemed a $5 coupon on coffee creamer last Sunday [S42]. |
| rag-vector | single-session-user | 51a45a95 | retrieval_miss | Where did I redeem a $5 coupon on coffee creamer? | Target | You redeemed the $5 coupon on coffee creamer from your email inbox [S0]. |
