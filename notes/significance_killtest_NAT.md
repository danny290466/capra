# Significance — `outputs/killtest` (en vs targets, mode NAT)

Generated 2026-09-23T12:29:31. Item-level paired bootstrap (B = 1000, thresholds fixed at their calibration values) and seed-level paired t-tests over the training seeds. A dagger (†) marks a rule that is **not** significantly different from the best rule in that column at alpha = 0.05.


## target: vi · C = 0.05

| rule | disparity D (pp) | 95% bootstrap CI | p vs best (item-level) | p vs best (seed-level, n = 5) | seed mean ± SD (pp) |
|---|---|---|---|---|---|
| CONRA | **0.93** | [0.15, 1.75] | 1.000 | — | 0.97 ± 0.44 (n=5) |
| CAPRA-parity† | **0.94** | [0.17, 1.76] | 0.032 | 0.610 | 0.95 ± 0.47 (n=5) |
| ABS-RISK† | **0.94** | [0.17, 1.76] | 0.032 | 0.610 | 0.95 ± 0.47 (n=5) |
| GLOBAL | **3.55** | [2.82, 4.33] | 0.001 | 0.000 | 3.65 ± 0.32 (n=5) |
| EQUAL-COV | **4.54** | [3.78, 5.32] | 0.001 | 0.000 | 4.52 ± 0.26 (n=5) |
| CAPRA-composition | **4.54** | [3.78, 5.32] | 0.001 | 0.000 | 4.52 ± 0.26 (n=5) |
