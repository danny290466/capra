# CAPRA with L = 7 groups (outputs/textdetox/xlmr_large)

Generated from the frozen artefacts · L = 7 groups · budget C = 5% · equal traffic weights w = 1/7 · cost-blind · thresholds on dev, everything measured on test · n per group = en 3375, de 3375, uk 3375, ar 3375, hi 3375, zh 3380, ru 3375 · seeds = 5 pooled

`max gap` = worst pairwise residual-risk gap (pp) between groups; `K dev` = max deviation of the queue composition from the traffic share; `load` = realised review share of traffic.

| rule | design max gap (pp) | **test max gap (pp)** | K dev | load | seed mean ± SD (pp) |
|---|---|---|---|---|---|
| pooled | 22.77 | **23.11** | 0.032 | 0.053 | 23.27 ± 0.90 |
| parity-first | 15.93 | **18.13** | 0.636 | 0.053 | 18.14 ± 1.21 |
| composition-first | 22.77 | **23.11** | 0.032 | 0.053 | 23.27 ± 0.90 |
| oracle (test-optimal) | 16.84 | **17.52** | 0.530 | 0.053 | 16.83 ± 1.02 |

## Per-group detail (test)

| group | risk pooled (pp) | risk parity-first (pp) | queue share pooled | queue share parity-first | traffic share |
|---|---|---|---|---|---|
| en | 1.84 | 3.50 | 0.175 | 0.006 | 0.143 |
| de | 7.35 | 9.02 | 0.111 | 0.004 | 0.143 |
| uk | 4.70 | 6.09 | 0.132 | 0.035 | 0.143 |
| ar | 19.97 | 20.03 | 0.153 | 0.147 | 0.143 |
| hi | 4.48 | 5.83 | 0.148 | 0.026 | 0.143 |
| zh | 23.67 | 18.27 | 0.133 | 0.779 | 0.143 |
| ru | 0.56 | 1.90 | 0.147 | 0.003 | 0.143 |

## Prices

- price of composition (test max gap at K = w minus at parity-first): **+4.97 pp**
- price of parity (L1 queue deviation of the parity-first point): **1.281**
- both objectives feasible: **no**
- pooled worst gap 23.11 pp -> parity-first 18.13 pp

## Trade-off frontier: the worst gap under a ceiling on the queue deviation

`delta` is the largest deviation of the queue composition from the traffic shares the policy is allowed; the row reports the smallest worst pairwise gap attainable under that ceiling.

| delta ceiling | design max gap (pp) | design K dev | test max gap (pp) | test K dev |
|---|---|---|---|---|
| 0 | 22.77 | 0.000 | 23.11 | 0.032 |
| 0.02 | 22.38 | 0.020 | 22.89 | 0.041 |
| 0.05 | 22.11 | 0.048 | 22.59 | 0.033 |
| 0.1 | 21.12 | 0.100 | 21.44 | 0.090 |
| 0.15 | 20.35 | 0.150 | 20.43 | 0.155 |
| 0.2 | 19.92 | 0.200 | 19.95 | 0.200 |
| 0.3 | 18.67 | 0.300 | 19.03 | 0.310 |
| 0.4 | 18.01 | 0.399 | 19.12 | 0.389 |
| 0.6 | 16.18 | 0.593 | 18.56 | 0.600 |
| 1 | 15.86 | 0.653 | 18.03 | 0.635 |

## Capacity at which the worst gap closes (parity-first allocation per budget)

| budget C | design max gap (pp) |
|---|---|
| 5% | 15.87 |
| 8% | 14.80 |
| 10% | 13.07 |
| 15% | 10.18 |
| 20% | 7.94 |
| 25% | 5.22 |
| 30% | 2.33 |
| 35% | 0.93 |
| 40% | 1.23 |
