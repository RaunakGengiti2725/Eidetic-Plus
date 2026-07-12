# Benchmark Failure Forensics

Rows inspected: 20
Failures/abstentions/errors: 5

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| reader_error | 3 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |
| abstention | 2 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |

## Categories

| category | buckets |
|---|---|
| multi-hop | reader_error: 1 |
| open-domain | abstention: 2, reader_error: 1 |
| single-hop | reader_error: 1 |

## Consolidation Health

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred |
|---|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 6 | 156 | 117 | 117 | 0 | 0 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | multi-hop | c1_q3 | reader_error | What do Jon and Gina both have in common? | They lost their jobs and decided to start their own businesses. | Both Jon and Gina are passionate about dancing and use it as stress relief [S12] 'Dance is my stress fix too' / 'Dancing |
| eidetic-plus-full | open-domain | c2_q8 | reader_error | What might John's financial status be? | Middle-class or wealthy | John's financial status is likely strained: he recently lost his job [S4] and reported car trouble 'putting a strain on  |
| eidetic-plus-full | open-domain | c3_q4 | abstention | What pets wouldn't cause any discomfort to Joanna? | Hairless cats or pigs,since they don't have fur, which is one of the main causes of Joanna's allergy. | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | single-hop | c3_q88 | reader_error | What is one of Joanna's favorite movies? | "Eternal Sunshineof the Spotless Mind" | a romantic drama about memory and relationships [S5] |
| eidetic-plus-full | open-domain | c4_q3 | abstention | Would Tim enjoy reading books by C. S. Lewis or John Greene? | C. S.Lewis | I don't have enough verified evidence in memory to answer that confidently. |
