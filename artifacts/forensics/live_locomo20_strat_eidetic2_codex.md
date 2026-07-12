# Benchmark Failure Forensics

Rows inspected: 20
Failures/abstentions/errors: 4

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| reader_error | 3 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |
| abstention | 1 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |

## Categories

| category | buckets |
|---|---|
| multi-hop | reader_error: 1 |
| open-domain | abstention: 1, reader_error: 2 |

## Consolidation Health

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred |
|---|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 6 | 156 | 1795 | 1795 | 2 | 0 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | open-domain | c2_q8 | reader_error | What might John's financial status be? | Middle-class or wealthy | John's financial status is likely strained: he lost his job [S4], and his car broke down, causing 'money problems' and s |
| eidetic-plus-full | open-domain | c3_q4 | reader_error | What pets wouldn't cause any discomfort to Joanna? | Hairless cats or pigs,since they don't have fur, which is one of the main causes of Joanna's allergy. | Turtles [S10] 'I wish I wasn't allergic! I would get two turtles today if I could!' — implying she is not allergic to th |
| eidetic-plus-full | multi-hop | c3_q5 | reader_error | What are Joanna's hobbies? | Writing, watchingmovies, exploringnature, hanging withfriends. | writing [S26], reading [S1], watching movies [S25], exploring nature [S2] |
| eidetic-plus-full | open-domain | c4_q3 | abstention | Would Tim enjoy reading books by C. S. Lewis or John Greene? | C. S.Lewis | I don't have enough verified evidence in memory to answer that confidently. |
