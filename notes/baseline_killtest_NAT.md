# Prior-art baselines vs CAPRA — `outputs/killtest` (en vs targets, mode NAT)

Generated 2026-10-13T18:10:28. Every family is fitted on the calibration split and evaluated on test at **load = C** (interpolated when the family cannot control its own load). `D` = |risk gap| in pp; `K` = queue share of the second language; `K*` = 0.50. `load_test` is the ITEM share of traffic reviewed: for the cost-blind variant it must match C to within 0.5 pp, while in the cost-aware variant the budget is denominated in review TIME, so the item share is lower than C by construction (the budget-unit load is printed alongside). Dominance is assessed on CALIBRATION (where a deployment chooses) and requires a gap above 0.1 pp; `indistinguishable` means the rule's test disparity and queue share match CAPRA's to within 0.1 pp and 0.01. RANDOM is the null policy (a uniformly random review of each language, drawn on the budget's own unit: share C of the items when costs are equal, and the item share that spends C of review time when they are not) and ORACLE is the allocation that is optimal on test at the same load, matched in that same unit — an upper bound, not a deployable rule.


## vi · C = 0.05 · cost_blind

| family | design D (pp) | design K | load (design) | test D (pp) | test K | test load | |K−K*| | vs CAPRA on test | note |
|---|---|---|---|---|---|---|---|---|---|
| CAPRA | +1.99 | 1.000 | 0.050 | **+0.94** | 0.998 | 0.054 | 0.498 | indistinguishable | ours |
| RANDOM | +5.35 | 0.500 | 0.050 | **+4.76** | 0.572 | 0.050 | 0.072 | different | null |
| ORACLE(test-optimal) | — | — | — | **+0.89** | 1.000 | 0.055 | 0.500 | indistinguishable | oracle |
| GLOBAL | +4.50 | 0.631 | 0.050 | **+3.65** | 0.636 | 0.054 | 0.136 | different | prior |
| EQUAL-COV | +5.31 | 0.500 | 0.050 | **+4.54** | 0.507 | 0.053 | 0.007 | different | prior |
| ABS-RISK | +1.99 | 1.000 | 0.050 | **+0.94** | 0.998 | 0.054 | 0.498 | indistinguishable | prior |
| ABS-RISK-finite | +2.00 | 1.000 | 0.050 | **+0.94** | 0.998 | 0.054 | 0.498 | indistinguishable | prior |
| BONF-CRC | +2.00 | 1.000 | 0.050 | **+0.94** | 0.998 | 0.054 | 0.498 | indistinguishable | prior |
| MINMAX-CAP | +1.99 | 1.000 | 0.050 | **+0.94** | 0.998 | 0.054 | 0.498 | indistinguishable | prior |

## vi · C = 0.05 · cost_aware

| family | design D (pp) | design K | load (design) | test D (pp) | test K | test load | |K−K*| | vs CAPRA on test | note |
|---|---|---|---|---|---|---|---|---|---|
| CAPRA | +3.04 | 1.000 | 0.050 | **+1.98** | 0.997 | 0.037 | 0.497 | indistinguishable | ours |
| RANDOM | +5.35 | 0.500 | 0.050 | **+4.67** | 0.572 | 0.040 | 0.072 | different | null |
| ORACLE(test-optimal) | — | — | — | **+1.92** | 1.000 | 0.037 | 0.500 | indistinguishable | oracle |
| GLOBAL | +4.60 | 0.728 | 0.050 | **+3.87** | 0.641 | 0.040 | 0.141 | dominated (design) | prior |
| EQUAL-COV | +5.28 | 0.600 | 0.050 | **+4.55** | 0.515 | 0.043 | 0.015 | dominated (design) | prior |
| ABS-RISK | +3.03 | 1.000 | 0.050 | **+1.98** | 0.997 | 0.037 | 0.497 | indistinguishable | prior |
| ABS-RISK-finite | +3.06 | 1.000 | 0.050 | **+1.98** | 0.997 | 0.037 | 0.497 | indistinguishable | prior |
| BONF-CRC | +3.06 | 1.000 | 0.050 | **+1.98** | 0.997 | 0.037 | 0.497 | indistinguishable | prior |
| MINMAX-CAP | +3.04 | 1.000 | 0.050 | **+1.98** | 0.997 | 0.037 | 0.497 | indistinguishable | prior |
