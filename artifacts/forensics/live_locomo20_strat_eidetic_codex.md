# Benchmark Failure Forensics

Rows inspected: 20
Failures/abstentions/errors: 6

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| reader_error | 4 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |
| abstention | 2 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |

## Categories

| category | buckets |
|---|---|
| multi-hop | abstention: 1, reader_error: 1 |
| open-domain | abstention: 1, reader_error: 2 |
| temporal | reader_error: 1 |

## Consolidation Health

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred |
|---|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 6 | 156 | 1824 | 1824 | 0 | 0 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | multi-hop | c2_q2 | abstention | What martial arts has John done? | Kickboxing, Taekwondo | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | open-domain | c2_q8 | reader_error | What might John's financial status be? | Middle-class or wealthy | John's financial status is likely strained: he recently lost his job [S4], reported his car breakdown was 'putting a str |
| eidetic-plus-full | open-domain | c3_q4 | reader_error | What pets wouldn't cause any discomfort to Joanna? | Hairless cats or pigs,since they don't have fur, which is one of the main causes of Joanna's allergy. | Turtles wouldn't cause any discomfort to Joanna [S10] 'I'm allergic to most reptiles and animals with fur' — but she exp |
| eidetic-plus-full | multi-hop | c3_q5 | reader_error | What are Joanna's hobbies? | Writing, watchingmovies, exploringnature, hanging withfriends. | writing [S26], reading [S1], exploring nature [S2], watching movies [S25], cooking and baking [S9] |
| eidetic-plus-full | open-domain | c4_q3 | abstention | Would Tim enjoy reading books by C. S. Lewis or John Greene? | C. S.Lewis | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | temporal | c4_q6 | reader_error | In which month's game did John achieve a career-high score in points? | June 2023 | July [S0] "John achieved his personal best score of 40 points." [S9] "Last week I scored 40 points, my highest ever". |
