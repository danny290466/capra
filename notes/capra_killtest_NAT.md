# CAPRA — capacity-aware parity vs queue composition (en vs vi, NAT)

Generated 2026-09-22T19:51:57 · detector `outputs/killtest` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 1.99 | **0.94** | 0.998 | 5.31 | **4.54** | 0.507 | 3.32 | no |
| 0.05 | cost_aware | 3.04 | **1.98** | 0.997 | 5.78 | **5.18** | 0.411 | 2.74 | no |
| 0.10 | cost_blind | 0.02 | **1.17** | 0.970 | 4.86 | **4.01** | 0.510 | 4.84 | no |
| 0.10 | cost_aware | 1.20 | **0.03** | 0.998 | 5.91 | **5.04** | 0.415 | 4.71 | no |
| 0.20 | cost_blind | 0.01 | **1.13** | 0.834 | 3.67 | **2.83** | 0.507 | 3.66 | no |
| 0.20 | cost_aware | 0.03 | **1.02** | 0.880 | 5.20 | **4.41** | 0.409 | 5.17 | no |
