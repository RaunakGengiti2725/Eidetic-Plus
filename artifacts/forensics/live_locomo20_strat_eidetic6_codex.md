# Benchmark Failure Forensics

Rows inspected: 20
Failures/abstentions/errors: 3

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| abstention | 2 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |
| reader_error | 1 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |

## Categories

| category | buckets |
|---|---|
| open-domain | abstention: 2, reader_error: 1 |

## Consolidation Health

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred |
|---|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 6 | 156 | 111 | 111 | 0 | 0 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | open-domain | c2_q8 | reader_error | What might John's financial status be? | Middle-class or wealthy | John's financial status is likely strained: he lost his job unexpectedly [S4], and his car broke down, causing 'a strain |
| eidetic-plus-full | open-domain | c3_q4 | abstention | What pets wouldn't cause any discomfort to Joanna? | Hairless cats or pigs,since they don't have fur, which is one of the main causes of Joanna's allergy. | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | open-domain | c4_q3 | abstention | Would Tim enjoy reading books by C. S. Lewis or John Greene? | C. S.Lewis | I don't have enough verified evidence in memory to answer that confidently. |
