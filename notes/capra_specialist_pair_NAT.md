# CAPRA — capacity-aware parity vs queue composition (en vs vi, NAT)

Generated 2026-09-23T11:39:08 · detector `outputs/pool/specialist_pair` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 1.19 | **1.29** | 1.000 | 4.62 | **4.70** | 0.490 | 3.41 | no |
| 0.05 | cost_aware | 2.30 | **2.35** | 0.999 | 4.54 | **4.65** | 0.492 | 2.30 | no |
| 0.10 | cost_blind | 0.02 | **0.16** | 0.911 | 4.10 | **4.19** | 0.505 | 4.03 | no |
| 0.10 | cost_aware | 0.27 | **0.37** | 1.000 | 4.42 | **4.44** | 0.496 | 4.07 | no |
| 0.20 | cost_blind | 0.02 | **0.73** | 0.809 | 3.08 | **2.91** | 0.506 | 2.18 | no |
| 0.20 | cost_aware | 0.01 | **0.28** | 0.839 | 3.42 | **3.40** | 0.506 | 3.12 | no |
