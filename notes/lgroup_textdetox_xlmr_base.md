# CAPRA with L = 7 groups (outputs/textdetox/xlmr_base)

Generated from the frozen artefacts · L = 7 groups · budget C = 5% · equal traffic weights w = 1/7 · cost-blind · thresholds on dev, everything measured on test · n per group = en 3375, de 3375, uk 3375, ar 3375, hi 3375, zh 3380, ru 3375 · seeds = 5 pooled

`max gap` = worst pairwise residual-risk gap (pp) between groups; `K dev` = max deviation of the queue composition from the traffic share; `load` = realised review share of traffic.

| rule | design max gap (pp) | **test max gap (pp)** | K dev | load | seed mean ± SD (pp) |
|---|---|---|---|---|---|
| pooled | 23.29 | **22.74** | 0.052 | 0.052 | 22.95 ± 0.53 |
| parity-first | 17.71 | **17.52** | 0.467 | 0.053 | 17.48 ± 1.28 |
| composition-first | 23.29 | **22.74** | 0.052 | 0.052 | 22.95 ± 0.53 |
| oracle (test-optimal) | 18.59 | **17.27** | 0.383 | 0.054 | 17.09 ± 1.38 |

## Per-group detail (test)

| group | risk pooled (pp) | risk parity-first (pp) | queue share pooled | queue share parity-first | traffic share |
|---|---|---|---|---|---|
| en | 2.73 | 3.89 | 0.126 | 0.027 | 0.143 |
| de | 8.62 | 10.27 | 0.121 | 0.004 | 0.143 |
| uk | 7.76 | 9.59 | 0.147 | 0.006 | 0.143 |
| ar | 23.16 | 21.31 | 0.136 | 0.333 | 0.143 |
| hi | 4.99 | 6.44 | 0.135 | 0.018 | 0.143 |
| zh | 24.30 | 20.11 | 0.141 | 0.610 | 0.143 |
| ru | 1.56 | 3.79 | 0.195 | 0.002 | 0.143 |

## Prices

- price of composition (test max gap at K = w minus at parity-first): **+5.22 pp**
- price of parity (L1 queue deviation of the parity-first point): **1.314**
- both objectives feasible: **no**
- pooled worst gap 22.74 pp -> parity-first 17.52 pp

## Trade-off frontier: the worst gap under a ceiling on the queue deviation

`delta` is the largest deviation of the queue composition from the traffic shares the policy is allowed; the row reports the smallest worst pairwise gap attainable under that ceiling.

| delta ceiling | design max gap (pp) | design K dev | test max gap (pp) | test K dev |
|---|---|---|---|---|
| 0 | 23.29 | 0.000 | 22.74 | 0.052 |
| 0.02 | 23.00 | 0.020 | 22.35 | 0.027 |
| 0.05 | 22.57 | 0.050 | 21.80 | 0.065 |
| 0.1 | 21.41 | 0.100 | 20.50 | 0.113 |
| 0.15 | 20.52 | 0.148 | 19.66 | 0.165 |
| 0.2 | 20.12 | 0.200 | 20.25 | 0.217 |
| 0.3 | 19.35 | 0.285 | 18.78 | 0.297 |
| 0.4 | 18.39 | 0.399 | 18.06 | 0.426 |
| 0.6 | 17.74 | 0.450 | 17.68 | 0.469 |
| 1 | 17.74 | 0.450 | 17.68 | 0.469 |

## Capacity at which the worst gap closes (parity-first allocation per budget)

| budget C | design max gap (pp) |
|---|---|
| 5% | 17.74 |
| 8% | 15.87 |
| 10% | 13.81 |
| 15% | 9.64 |
| 20% | 7.33 |
| 25% | 4.20 |
| 30% | 2.32 |
| 35% | 0.43 |
| 40% | 0.46 |
