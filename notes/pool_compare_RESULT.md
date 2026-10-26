# Detector pool — does the EN/VI abstention finding generalise? (task 5)

Generated 2026-09-23T12:44:41 · reference budget **C = 5%** · balanced EN/VI traffic · thresholds fitted on dev, reported on test · seeds pooled per detector.


`min parity capacity` = first review capacity (0.5 %-steps) at which CONRA-Parity's risk gap reaches 0 *without either language being saturated at full coverage* — i.e. the capacity below which language parity is structurally unattainable.
`parity load @r=5%` = review load exact risk parity needs at a 5 % target residual risk.


## NAT

| detector | n/lang | acc EN | acc VI | AUC EN | AUC VI | ECE EN | ECE VI | GLOBAL Δ (pp) | EQUAL-COV Δ | CONRA Δ | queue VI (GLOBAL) | parity cap. (struct.) | r* @ min cap | Δ=0 crossing (meas.) | parity load @r=5% | refused (unscored) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xlmr_base | 25000 | 0.9387 | 0.8924 | 0.9142 | 0.9145 | 0.0319 | 0.0474 | **-3.55** | -4.54 | -0.93 | 0.637 | 9.5 % | 5.6 % | 7.0 % | 12.0 % | — |
| xlmr_large | 25000 | 0.9391 | 0.8855 | 0.8761 | 0.8800 | 0.0256 | 0.0553 | **-4.50** | -5.16 | -2.20 | 0.656 | 16.5 % | 5.8 % | 15.5 % | 19.0 % | — |
| specialist_pair | 25000 | 0.9402 | 0.8936 | 0.9190 | 0.9150 | 0.0375 | 0.0487 | **-3.16** | -4.70 | -1.29 | 0.703 | 7.5 % | 5.8 % | 7.5 % | 10.2 % | — |
| llm_deepseek_flash | 5000 | 0.8750 | 0.8850 | 0.8271 | 0.8880 | 0.0859 | 0.0506 | **-0.35** | +0.25 | +0.93 | 0.373 | 0.5 % | 12.3 % | 0.5 % | 10.8 % | — |
| llm_gemini31_flashlite | 5000 | 0.8928 | 0.8870 | 0.8458 | 0.9228 | 0.0821 | 0.0467 | **-0.80** | +0.68 | -0.32 | 0.144 | 2.0 % | 10.0 % | 2.0 % | 16.2 % | — |

## MAT

| detector | n/lang | acc EN | acc VI | AUC EN | AUC VI | ECE EN | ECE VI | GLOBAL Δ (pp) | EQUAL-COV Δ | CONRA Δ | queue VI (GLOBAL) | parity cap. (struct.) | r* @ min cap | Δ=0 crossing (meas.) | parity load @r=5% | refused (unscored) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xlmr_base | 23625 | 0.9374 | 0.9182 | 0.9144 | 0.9142 | 0.0319 | 0.0480 | **-1.10** | -1.79 | +0.75 | 0.597 | 3.5 % | 5.8 % | 2.5 % | 6.0 % | — |
| xlmr_large | 23625 | 0.9370 | 0.9226 | 0.8762 | 0.8795 | 0.0248 | 0.0396 | **-0.80** | -1.17 | +0.63 | 0.611 | 3.0 % | 5.9 % | 2.0 % | 6.6 % | — |
| specialist_pair | 23625 | 0.9385 | 0.9222 | 0.9192 | 0.9177 | 0.0385 | 0.0418 | **-0.34** | -1.62 | +0.06 | 0.661 | 2.5 % | 5.8 % | 3.0 % | 4.6 % | — |
| llm_deepseek_flash | 4725 | 0.8737 | 0.9065 | 0.8274 | 0.8862 | 0.0827 | 0.0642 | **+2.16** | +2.78 | +0.20 | 0.324 | 3.0 % | 9.9 % | 0.5 % | 10.4 % | — |
| llm_gemini31_flashlite | 4725 | 0.8908 | 0.9012 | 0.8456 | 0.9202 | 0.0790 | 0.0744 | **+0.79** | +2.26 | +1.54 | 0.089 | 0.5 % | 10.2 % | 0.5 % | 15.5 % | — |

## PAIR

| detector | n/lang | acc EN | acc VI | AUC EN | AUC VI | ECE EN | ECE VI | GLOBAL Δ (pp) | EQUAL-COV Δ | CONRA Δ | queue VI (GLOBAL) | parity cap. (struct.) | r* @ min cap | Δ=0 crossing (meas.) | parity load @r=5% | refused (unscored) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xlmr_base | 25000 | 0.9387 | 0.8965 | 0.9142 | 0.7346 | 0.0319 | 0.0671 | **-3.54** | -4.53 | -0.99 | 0.638 | 8.0 % | 5.6 % | 7.5 % | 11.1 % | — |
| xlmr_large | 25000 | 0.9391 | 0.9090 | 0.8761 | 0.7177 | 0.0256 | 0.0519 | **-2.42** | -3.06 | -0.32 | 0.630 | 6.5 % | 5.8 % | 6.5 % | 18.4 % | — |
| specialist_pair | 25000 | 0.9402 | 0.9062 | 0.9190 | 0.6730 | 0.0375 | 0.0626 | **-2.65** | -3.66 | -0.70 | 0.673 | 7.5 % | 5.7 % | 10.0 % | 12.3 % | — |
| llm_deepseek_flash | 5000 | 0.8750 | 0.8936 | 0.8271 | 0.7380 | 0.0859 | 0.0585 | **-0.22** | +1.10 | -1.11 | 0.251 | 2.0 % | 11.0 % | 0.5 % | 11.4 % | — |
| llm_gemini31_flashlite | 5000 | 0.8928 | 0.8924 | 0.8458 | 0.7516 | 0.0821 | 0.0797 | **-0.32** | +1.08 | -1.24 | 0.355 | 0.5 % | 10.0 % | 2.5 % | 18.3 % | — |

## Reading

- A finding that only holds for one architecture is an artefact; the table above is the test.
- `queue VI` under GLOBAL is the deployable-rule disparity; CONRA's Δ shows how much of it a risk-parity rule can remove at the same budget, and `min parity cap.` says at what capacity the removal becomes possible at all.
- The LLM moderator matters twice: as an external confidence source, and because its verbalised probabilities are typically poorly calibrated in the lower-resource language — the per-language ECE columns are the place to look for that.
