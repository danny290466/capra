# Ablations — `outputs/killtest` (en vs targets, mode NAT, budget C = 0.05)

Generated 2026-09-23T12:44:40. Parity-first = min |risk gap| under the budget; composition-first = queue share K* = 0.50. All numbers on the TEST split after fitting on calibration.


## target: vi  (n_dev = 25000, seeds = 5)


**A. composition target**

| K* | parity-first D test (pp) | composition-first D test (pp) | its K test | price of composition (pp) |
|---|---|---|---|---|
| 0.4 | 0.94 | 5.08 | 0.423 | 4.14 |
| 0.5 | 0.94 | 4.54 | 0.507 | 3.59 |
| 0.6 | 0.94 | 3.86 | 0.607 | 2.92 |

**B. review cost of the second language**

| c_second | parity-first D test (pp) | composition-first D test (pp) | price of composition (pp) |
|---|---|---|---|
| 1.0× | 0.94 | 4.54 | 3.59 |
| 1.3× | 1.69 | 4.54 | 2.85 |
| 1.5× | 1.98 | 4.55 | 2.57 |
| 2.0× | 2.61 | 4.68 | 2.06 |

**C. traffic mix**

| w_second | parity-first D test (pp) | parity-first K test | min parity capacity |
|---|---|---|---|
| 0.3 | 0.84 | 0.999 | 9.5 % |
| 0.5 | 0.94 | 0.998 | 9.5 % |
| 0.7 | 1.85 | 0.997 | 9.5 % |

**D. calibration size** (mean ± SD over 3 draws, same n per language)

| n_dev | parity-first D test (pp) | min parity capacity |
|---|---|---|
| 250 | 1.42 ± 0.93 | 16.3 % |
| 500 | 0.38 ± 0.12 | 11.5 % |
| 1000 | 0.35 ± 0.19 | 8.7 % |
| 2000 | 0.33 ± 0.13 | 10.2 % |
| full | 0.94 | 9.5 % |

**E. seed spread** (real training seeds)

D test = 0.95 ± 0.47 pp, K test = 0.996 ± 0.004 over 5 seeds

| seed dir | parity-first D test (pp) | K test |
|---|---|---|
| seed42 | 1.15 | 0.993 |
| seed43 | 1.67 | 0.996 |
| seed44 | 0.58 | 0.991 |
| seed45 | 0.55 | 1.000 |
| seed46 | 0.80 | 0.998 |
