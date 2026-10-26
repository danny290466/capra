# CAPRA — capacity-aware parity vs queue composition (en vs vi, PAIR)

Generated 2026-09-22T19:51:58 · detector `outputs/killtest` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 1.12 | **1.00** | 0.998 | 4.23 | **4.53** | 0.471 | 3.11 | no |
| 0.05 | cost_aware | 1.99 | **2.14** | 0.996 | 4.81 | **4.98** | 0.378 | 2.82 | no |
| 0.10 | cost_blind | 0.00 | **0.30** | 0.941 | 3.99 | **4.06** | 0.490 | 3.99 | no |
| 0.10 | cost_aware | 0.44 | **0.24** | 0.998 | 4.86 | **5.20** | 0.378 | 4.41 | no |
| 0.20 | cost_blind | 0.03 | **0.21** | 0.856 | 3.39 | **3.37** | 0.502 | 3.36 | no |
| 0.20 | cost_aware | 0.01 | **0.23** | 0.880 | 4.44 | **4.68** | 0.394 | 4.43 | no |
