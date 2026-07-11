# Burned-Window Guard Projection (policy v2)

Status: **PASS**

Applies the CURRENT deterministic runtime guards to frozen historically-VERIFIED rows.
No provider, NLI, judge, or generation calls.

| metric | value |
|---|---:|
| verified rows | 341 |
| verified wrong before | 129 |
| wrong converted to abstain | 21 |
| correct lost to abstain | 3 |
| verified wrong after | 108 |

## Wrong converted, by floor

- **clean_fact**: 9
- **enumeration_form**: 3
- **preference_form**: 4
- **when_type**: 2
- **zero_information**: 3

## Correct lost, by floor (every row enumerated)

- **clean_fact**: 3
  - `c0_q107` [clean_fact] Q: What is Melanie's reason for getting into running? | P: I've been running farther to de-stress
  - `c6_q114` [clean_fact] Q: What is James planning to do after receiving Samantha's phone number? | P: I think I'll call tomorrow
  - `c7_q100` [clean_fact] Q: What project did Jolene finish last week before 23 January, 2023? | P: I finished an electrical engineering project

## Residual verified-wrong, by family

- **count_aggregate**: 2
- **latest_value**: 14
- **open_inference**: 9
- **preference_synth**: 2
- **reader**: 47
- **relative_temporal**: 33
- **temporal_delta**: 1

## Limitations

- **Boundary**: Projects only the PURE deterministic form floors; support-dependent floors (aggregate citation, category anchoring, premise position) and the relative_temporal ambiguity guard need live candidate sets and are NOT projected here.
- **Boundary**: Reuses frozen historical correctness labels; it does not rerun retrieval, generation, NLI, or judges.
- **Boundary**: Correct-lost rows are enumerated in full; accepting them is a policy decision, not a mechanical one.
