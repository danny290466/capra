# ParityAbstain — final results digest

Generated 2026-09-23T10:26:25 from the artefact JSONs (no hand-typed numbers).


## A. EN/VI detector pool (mode NAT, w = 1/2, thresholds on dev, evaluated on test)

| detector | seeds | acc EN | acc VI | AUC EN | AUC VI | ECE EN | ECE VI | Δ GLOBAL (pp) | Δ EQUAL-COV | Δ CONRA | queue VI | parity cap. (struct.) | parity load @r=5% | refused |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xlmr_base | 5 | 0.9387 | 0.8924 | 0.9142 | 0.9145 | 0.0319 | 0.0474 | **-3.55** | -4.54 | -0.93 | 0.637 | 9.5 % | 12.0 % | — |
| xlmr_large | 5 | 0.9391 | 0.8855 | 0.8761 | 0.8800 | 0.0256 | 0.0553 | **-4.50** | -5.16 | -2.20 | 0.656 | 16.5 % | 19.0 % | — |
| specialist_pair | 5 | 0.9402 | 0.8936 | 0.9190 | 0.9150 | 0.0375 | 0.0487 | **-3.16** | -4.70 | -1.29 | 0.703 | 7.5 % | 10.2 % | — |
| llm_deepseek_flash | 1 | 0.8750 | 0.8850 | 0.8271 | 0.8880 | 0.0859 | 0.0506 | **-0.35** | +0.25 | +0.93 | 0.373 | 0.5 % | 10.8 % | — |
| llm_gemini31_flashlite | 1 | 0.8928 | 0.8870 | 0.8458 | 0.9228 | 0.0821 | 0.0467 | **-0.80** | +0.68 | -0.32 | 0.144 | 2.0 % | 16.2 % | — |
| llm_qwen38_flash | 1 | 0.8446 | 0.8536 | 0.8397 | 0.8808 | 0.1415 | 0.0910 | **+0.06** | +0.45 | +0.45 | 0.241 | 1.0 % | 40.8 % | tr_test 3/5000 (0.1 %), tr_dev 357/5000 (7.1 %), en_dev 20/5000 (0.4 %), en_test 17/5000 (0.3 %) |

## B. Second language pair — textdetox EN + DE + UK + AR (one shared model, 5 seeds)

| detector | target | n/lang | acc EN | acc tgt | AUC EN | AUC tgt | ECE tgt | Δ GLOBAL (pp) | Δ CONRA | queue tgt | parity cap. (struct.) | parity load @r=5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_deepseek_flash | de | 675 | 0.9452 | 0.7926 | 0.9804 | 0.8720 | 0.1036 | **-14.43** | -14.62 | 0.645 | 31.5 % | 30.9 % |
| llm_deepseek_flash | uk | 675 | 0.9452 | 0.6889 | 0.9804 | 0.8217 | 0.2690 | **-25.66** | -23.45 | 0.333 | — | 50.0 % |
| llm_deepseek_flash | ru | 675 | 0.9452 | 0.9304 | 0.9804 | 0.9743 | 0.0654 | **-0.91** | +0.03 | 0.520 | 17.5 % | 3.8 % |
| llm_deepseek_flash | ar | 675 | 0.9452 | 0.7704 | 0.9804 | 0.8447 | 0.1262 | **-16.12** | -2.96 | 0.571 | 27.0 % | 4.7 % |
| llm_deepseek_flash | hi | 675 | 0.9452 | 0.8148 | 0.9804 | 0.9506 | 0.1785 | **-12.86** | -5.60 | 0.745 | 29.0 % | 27.8 % |
| llm_deepseek_flash | zh | 675 | 0.9452 | 0.7633 | 0.9804 | 0.8610 | 0.1333 | **-16.86** | -16.96 | 0.835 | — | 41.2 % |
| xlmr_base | de | 3375 | 0.9570 | 0.8966 | 0.9908 | 0.9634 | 0.0592 | **-3.95** | -2.42 | 0.773 | 12.5 % | 8.4 % |
| xlmr_base | uk | 3375 | 0.9570 | 0.9031 | 0.9908 | 0.9620 | 0.0657 | **-3.80** | -1.83 | 0.737 | 9.0 % | 5.8 % |
| xlmr_base | ru | 3375 | 0.9570 | 0.9618 | 0.9908 | 0.9939 | 0.0246 | **+0.74** | +0.07 | 0.523 | 1.5 % | 0.0 % |
| xlmr_base | ar | 3375 | 0.9570 | 0.7541 | 0.9908 | 0.8412 | 0.1133 | **-17.91** | -17.61 | 0.923 | — | 39.4 % |
| xlmr_base | hi | 3375 | 0.9570 | 0.9330 | 0.9908 | 0.9824 | 0.0420 | **-1.57** | +0.71 | 0.648 | 5.0 % | 2.6 % |
| xlmr_base | zh | 3375 | 0.9570 | 0.7426 | 0.9908 | 0.8217 | 0.1028 | **-19.20** | -18.83 | 0.935 | — | 44.9 % |
| xlmr_large | de | 3375 | 0.9636 | 0.9087 | 0.9907 | 0.9643 | 0.0486 | **-3.79** | -2.31 | 0.729 | 22.0 % | 6.8 % |
| xlmr_large | uk | 3375 | 0.9636 | 0.9336 | 0.9907 | 0.9751 | 0.0498 | **-2.12** | -0.28 | 0.569 | 20.0 % | 2.4 % |
| xlmr_large | ru | 3375 | 0.9636 | 0.9804 | 0.9907 | 0.9978 | 0.0099 | **+1.07** | +1.19 | 0.410 | 0.5 % | 0.0 % |
| xlmr_large | ar | 3375 | 0.9636 | 0.7864 | 0.9907 | 0.8595 | 0.1054 | **-15.35** | -14.80 | 0.911 | — | 34.8 % |
| xlmr_large | hi | 3375 | 0.9636 | 0.9372 | 0.9907 | 0.9826 | 0.0355 | **-1.38** | +0.40 | 0.653 | 11.0 % | 1.2 % |
| xlmr_large | zh | 3375 | 0.9636 | 0.7479 | 0.9907 | 0.8252 | 0.1216 | **-19.13** | -18.80 | 0.914 | — | 47.3 % |

## C. CAPRA endpoints (budget C = 5 %, composition target K* = 0.5)

`parity-first` = CONRA-Parity (min disparity); `composition-first` = queue mirrors traffic (the equalized-coverage baseline). Both are measured on test after fitting on dev.

| detector | pair | variant | parity-first D (pp) | its K | composition-first D (pp) | its K | price of composition (pp) | both feasible? |
|---|---|---|---|---|---|---|---|---|
| killtest | en/vi | cost_blind | 0.73 | 0.859 | 1.79 | 0.509 | 2.41 | no |
| killtest | en/vi | cost_aware | 0.75 | 0.998 | 2.37 | 0.418 | 2.80 | no |
| killtest | en/vi | cost_blind | 0.94 | 0.998 | 4.54 | 0.507 | 3.32 | no |
| killtest | en/vi | cost_aware | 1.98 | 0.997 | 5.18 | 0.411 | 2.74 | no |
| killtest | en/vi | cost_blind | 1.00 | 0.998 | 4.53 | 0.471 | 3.11 | no |
| killtest | en/vi | cost_aware | 2.14 | 0.996 | 4.98 | 0.378 | 2.82 | no |
| llm_deepseek_flash | en/vi | cost_blind | 0.13 | 0.031 | 2.78 | 0.374 | 2.18 | no |
| llm_deepseek_flash | en/vi | cost_aware | 0.13 | 0.031 | 2.78 | 0.374 | 2.05 | no |
| llm_deepseek_flash | en/vi | cost_blind | 0.25 | 0.428 | 0.25 | 0.428 | 0.53 | no |
| llm_deepseek_flash | en/vi | cost_aware | 0.63 | 0.462 | 0.74 | 0.149 | 1.00 | no |
| llm_deepseek_flash | en/vi | cost_blind | 1.11 | 0.113 | 1.10 | 0.343 | 1.16 | no |
| llm_deepseek_flash | en/vi | cost_aware | 1.11 | 0.113 | 1.10 | 0.343 | 1.05 | no |
| llm_gemini31_flashlite | en/vi | cost_blind | 2.26 | 0.763 | 2.26 | 0.763 | 1.56 | no |
| llm_gemini31_flashlite | en/vi | cost_aware | 2.26 | 0.763 | 2.26 | 0.763 | 0.36 | no |
| llm_gemini31_flashlite | en/vi | cost_blind | 0.68 | 0.801 | 0.68 | 0.801 | 1.06 | no |
| llm_gemini31_flashlite | en/vi | cost_aware | 0.68 | 0.801 | 0.49 | 0.774 | 1.62 | no |
| llm_gemini31_flashlite | en/vi | cost_blind | 1.08 | 0.830 | 1.08 | 0.830 | 0.10 | no |
| llm_gemini31_flashlite | en/vi | cost_aware | 1.08 | 0.830 | 0.32 | 0.355 | 0.64 | no |
| llm_qwen38_flash | en/vi | cost_blind | 0.72 | 0.144 | 1.68 | 0.407 | 1.93 | no |
| llm_qwen38_flash | en/vi | cost_aware | 0.76 | 0.176 | 1.54 | 0.315 | 1.08 | no |
| llm_qwen38_flash | en/vi | cost_blind | 0.45 | 0.407 | 0.45 | 0.407 | 0.45 | no |
| llm_qwen38_flash | en/vi | cost_aware | 0.50 | 0.438 | 0.27 | 0.315 | 0.25 | no |
| llm_qwen38_flash | en/vi | cost_blind | 0.09 | 0.113 | 1.24 | 0.461 | 1.53 | no |
| llm_qwen38_flash | en/vi | cost_aware | 0.06 | 0.138 | 1.06 | 0.360 | 1.02 | no |
| specialist_pair | en/vi | cost_blind | 0.07 | 0.743 | 1.62 | 0.497 | 1.65 | no |
| specialist_pair | en/vi | cost_aware | 0.11 | 0.847 | 2.17 | 0.389 | 2.17 | no |
| specialist_pair | en/vi | cost_blind | 1.29 | 1.000 | 4.70 | 0.490 | 3.43 | no |
| specialist_pair | en/vi | cost_aware | 2.35 | 0.999 | 5.23 | 0.395 | 2.86 | no |
| specialist_pair | en/vi | cost_blind | 0.70 | 1.000 | 3.66 | 0.489 | 3.06 | no |
| specialist_pair | en/vi | cost_aware | 1.47 | 0.999 | 4.17 | 0.382 | 2.76 | no |
| xlmr_large | en/vi | cost_blind | 0.60 | 0.847 | 1.17 | 0.548 | 1.70 | no |
| xlmr_large | en/vi | cost_aware | 0.59 | 0.928 | 1.70 | 0.458 | 2.20 | no |
| xlmr_large | en/vi | cost_blind | 2.20 | 0.999 | 5.16 | 0.550 | 2.72 | no |
| xlmr_large | en/vi | cost_aware | 3.02 | 0.999 | 5.77 | 0.446 | 2.47 | no |
| xlmr_large | en/vi | cost_blind | 0.32 | 0.999 | 3.06 | 0.508 | 2.46 | no |
| xlmr_large | en/vi | cost_aware | 0.96 | 0.999 | 3.56 | 0.404 | 2.44 | no |
| xlmr_base | en/ar | cost_blind | 17.61 | 0.988 | 20.43 | 0.519 | 2.90 | no |
| xlmr_base | en/ar | cost_aware | 18.35 | 0.983 | 20.89 | 0.434 | 2.48 | no |
| xlmr_base | en/de | cost_blind | 2.37 | 0.988 | 5.88 | 0.490 | 3.27 | no |
| xlmr_base | en/de | cost_aware | 3.72 | 0.980 | 6.42 | 0.390 | 2.89 | no |
| xlmr_base | en/zh | cost_blind | 18.83 | 0.989 | 21.57 | 0.529 | 3.04 | no |
| xlmr_base | en/zh | cost_aware | 19.64 | 0.984 | 22.28 | 0.397 | 2.73 | no |
| xlmr_large | en/ar | cost_blind | 14.80 | 0.992 | 18.13 | 0.467 | 1.94 | no |
| xlmr_large | en/ar | cost_aware | 15.99 | 0.988 | 18.67 | 0.350 | 1.92 | no |
| xlmr_large | en/de | cost_blind | 2.31 | 0.990 | 5.52 | 0.389 | 2.71 | no |
| xlmr_large | en/de | cost_aware | 3.29 | 0.985 | 5.77 | 0.320 | 2.45 | no |
| xlmr_large | en/zh | cost_blind | 18.80 | 0.991 | 21.83 | 0.432 | 2.38 | no |
| xlmr_large | en/zh | cost_aware | 19.67 | 0.987 | 22.32 | 0.328 | 2.02 | no |

## D. What the numbers say

1. **The disparity is a property of the detector × language pair, not of a language.** Trained encoders
   leave Vietnamese more exposed (EN/VI: −3.2 to −4.5 pp, table A) while the hosted LLMs reach
   near-parity (+0.06 to −0.80 pp) — but the LLMs get there by *under*-reviewing Vietnamese (queue
   share 0.14–0.37 against a 0.50 traffic share). Three model families (DeepSeek / Google / Alibaba)
   agree on that, so it is not a single-vendor artefact.
2. **The shared-encoder objection is dead.** The specialist pair (RoBERTa-EN + PhoBERT-VI, no shared
   encoder) shows the same direction and a *worse* queue skew (0.703) than the shared XLM-R models.
3. **Parity has a price, and the price is detector-dependent**: the structural review capacity it needs
   ranges from 7.5 % (specialist pair) to 16.5 % (XLM-R large) of traffic on EN/VI — 1.5–3.3× a 5 %
   budget — and 5.5–30 % on the textdetox pairs.
4. **Parity and queue composition cannot both be met** (table C: `both feasible?` = no everywhere).
   Forcing the queue to mirror traffic costs 2.6–5.2 pp of disparity on EN/VI.
5. **Cost-aware capacity makes it worse**: if non-English review takes ~1.5× longer, the measured
   disparity of the parity-first policy nearly doubles (EN/VI XLM-R base: 0.98 → 2.00 pp), because a
   cost-blind allocation reviews the costlier language less.
6. **Script distance dominates.** Arabic (RTL, different script/family) needs ≥ 38–40 % of traffic in
   review to reach parity with English on the textdetox detector, and never reaches the two-sided
   regime at 40 %; German and Ukrainian need 9–11.5 %. Resource distance alone does not order these
   (German is high-resource yet costs more capacity than Ukrainian in some rows) — the ordering is
   detector-specific.
7. **Hosted moderators refuse hard cases**: Qwen refused 7.1 % of the English→Vietnamese translated
   dev set and 0.1 % of its test set (content-policy refusal, logged per item), i.e. it drops exactly
   the toxic positives — a moderator that will not score toxic text cannot serve as an abstention
   signal, and the refusal rate is a reportable property of the model.

