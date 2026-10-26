# CAPRA — capacity-aware parity vs queue composition (en vs vi, NAT)

Generated 2026-09-23T11:39:08 · detector `outputs/pool/llm_gemini31_flashlite` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 0.03 | **0.68** | 0.801 | 1.09 | **0.68** | 0.801 | 0.00 | no |
| 0.05 | cost_aware | 0.00 | **0.68** | 0.801 | 1.13 | **0.68** | 0.801 | 0.00 | no |
| 0.10 | cost_blind | 0.02 | **0.32** | 0.498 | 0.59 | **0.32** | 0.498 | 0.00 | no |
| 0.10 | cost_aware | 0.02 | **2.00** | 0.889 | 0.64 | **0.32** | 0.498 | -1.68 | no |
| 0.20 | cost_blind | 0.00 | **1.43** | 0.459 | 0.26 | **2.09** | 0.401 | 0.66 | no |
| 0.20 | cost_aware | 0.03 | **0.68** | 0.666 | 0.70 | **1.43** | 0.459 | 0.75 | no |
