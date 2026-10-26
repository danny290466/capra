# CAPRA — capacity-aware parity vs queue composition (en vs vi, NAT)

Generated 2026-09-22T22:19:04 · detector `outputs/pool/llm_qwen38_flash` · composition target K* = 0.50 · queue cost of vi = 1.5× per item (cost-aware variant). Thresholds/curves fitted on the calibration split.


`K` = share of the review queue that comes from the second language; `D` = |r_en − r_vi| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are generally different points (proposition in the module docstring).


| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | cost_blind | 0.00 | **0.45** | 0.407 | 0.45 | **0.45** | 0.407 | 0.45 | no |
| 0.05 | cost_aware | 0.00 | **0.50** | 0.438 | 0.25 | **0.27** | 0.315 | 0.25 | no |
| 0.10 | cost_blind | 0.01 | **0.08** | 0.271 | 1.47 | **0.08** | 0.271 | 1.46 | no |
| 0.10 | cost_aware | 0.01 | **0.08** | 0.271 | 0.06 | **0.08** | 0.271 | 0.06 | YES |
| 0.20 | cost_blind | 0.08 | **0.08** | 0.271 | 1.79 | **4.54** | 0.639 | 1.71 | no |
| 0.20 | cost_aware | 0.10 | **0.08** | 0.271 | 0.47 | **0.08** | 0.271 | 0.37 | no |
