# CAPRA — capacity-aware parity vs queue composition (en vs vi, NAT)

Generated 2026-09-23T11:39:07 · detector `outputs/pool/xlmr_large` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 3.06 | **2.20** | 0.999 | 5.78 | **5.16** | 0.550 | 2.96 | no |
| 0.05 | cost_aware | 3.84 | **3.02** | 0.999 | 5.89 | **5.26** | 0.553 | 2.24 | no |
| 0.10 | cost_blind | 1.87 | **1.31** | 1.000 | 5.35 | **4.50** | 0.546 | 3.19 | no |
| 0.10 | cost_aware | 2.55 | **1.65** | 0.999 | 5.75 | **4.90** | 0.529 | 3.25 | no |
| 0.20 | cost_blind | 0.03 | **1.06** | 0.934 | 4.67 | **4.07** | 0.489 | 3.01 | no |
| 0.20 | cost_aware | 1.13 | **0.51** | 1.000 | 4.81 | **4.04** | 0.524 | 3.53 | no |
