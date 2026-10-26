# CAPRA — capacity-aware parity vs queue composition (en vs vi, NAT)

Generated 2026-09-23T11:39:08 · detector `outputs/pool/llm_deepseek_flash` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 0.01 | **0.25** | 0.428 | 0.54 | **0.25** | 0.428 | 0.00 | no |
| 0.05 | cost_aware | 0.02 | **0.63** | 0.462 | 0.20 | **0.63** | 0.462 | 0.00 | no |
| 0.10 | cost_blind | 0.01 | **0.40** | 0.576 | 0.36 | **0.40** | 0.576 | 0.00 | no |
| 0.10 | cost_aware | 0.02 | **0.32** | 0.631 | 0.87 | **1.07** | 0.324 | 0.75 | no |
| 0.20 | cost_blind | 0.04 | **0.54** | 0.404 | 0.90 | **0.54** | 0.404 | 0.00 | no |
| 0.20 | cost_aware | 0.01 | **0.54** | 0.404 | 0.55 | **0.54** | 0.404 | 0.00 | no |
