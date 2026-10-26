# CAPRA — capacity-aware parity vs queue composition (en vs vi, MAT)

Generated 2026-09-22T19:51:58 · detector `outputs/killtest` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 0.01 | **0.73** | 0.859 | 2.41 | **1.79** | 0.509 | 2.41 | no |
| 0.05 | cost_aware | 0.09 | **0.75** | 0.998 | 2.90 | **2.37** | 0.418 | 2.80 | no |
| 0.10 | cost_blind | 0.00 | **0.92** | 0.736 | 2.03 | **1.26** | 0.511 | 2.02 | no |
| 0.10 | cost_aware | 0.01 | **0.80** | 0.782 | 2.99 | **2.34** | 0.411 | 2.99 | no |
| 0.20 | cost_blind | 0.00 | **0.50** | 0.616 | 1.16 | **0.56** | 0.503 | 1.16 | no |
| 0.20 | cost_aware | 0.01 | **0.79** | 0.665 | 2.47 | **1.79** | 0.409 | 2.46 | no |
