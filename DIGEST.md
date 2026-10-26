# Digest — every table and figure, its artefact, and the command that produces it

Manuscript under review. All paths are relative to this repository. Per-detector artefacts live under three
roots: `outputs/killtest/` (the flagship English–Vietnamese detector), `outputs/pool/<detector>/` (the other
English–Vietnamese detectors) and `outputs/textdetox/<detector>/` (the seven-language benchmark). Map a row's
`<detector root>` to whichever root that manuscript item names, and read the commands from `src/drivers/` as
the authoritative line for each artefact.

| manuscript item | artefact | produced by |
|---|---|---|
| Table 2 benchmarks, splits, prevalence | `outputs/killtest/analysis.json`, `<detector root>/seed*/metrics.json` | `src/killtest_data.py`, `src/textdetox_prep.py` |
| Table 3 detectors and endpoints | `notes/pool_compare_RESULT.md`, `outputs/pool/<llm detector>/usage.json`, `outputs/textdetox/<llm detector>/usage.json` | `src/pool_compare.py`, `src/pool_llm.py` |
| Tables 4–6 detector quality, allocation per detector, per-language behaviour | `<detector root>/capra*.json`, `outputs/killtest/{capra,capra_NAT}.json`, `outputs/killtest/capra_weighted_*.json` | `src/capra.py` (per detector), `src/capra_weighted.py` |
| Table 7 the three operating modes (NAT/MAT/PAIR) | `outputs/killtest/{capra_MAT,capra_PAIR,modes_ENVI}.json`, `<detector root>/capra_{MAT,PAIR}.json` | `src/capra.py` + `src/modes_report.py` + `src/killtest_translate_pair.py` |
| Table 8 seven languages per detector | `outputs/textdetox/<detector>/pair_parity.json`, `outputs/textdetox/parity_capacity_ci_*.json` | `src/pair_parity.py`, `src/tools/parity_capacity_ci.py` |
| Table 9 the achievable set and its two prices | `outputs/killtest/capra.json` | `src/capra.py` |
| Table 10 parity under a risk ceiling | `outputs/killtest/capra.json` (constrained program), `outputs/r4_ceiling_transfer.json` | `src/capra.py`, `src/ceiling_transfer_check.py` |
| Table 11 ablations | `outputs/killtest/ablations_NAT.json` | `src/ablate.py` |
| Table 12 seed spread of the parity-first policy | `outputs/killtest/ablations_NAT.json`, `outputs/textdetox/seed_quality_spread.json` | `src/ablate.py`, `src/tools/seed_quality_spread.py` |
| Table 13 matched-budget audit (prior art, null, oracle) | `outputs/killtest/baseline_NAT.json` | `src/baseline_rules.py` |
| Table 14 uncertainty of the rule comparison | `outputs/killtest/significance_NAT.json` | `src/significance.py` |
| Section 5.2 held-out-language controls (LOLO/LOSO: the detector retrained without the scored language, and the same without the Cyrillic languages) | `notes/lolo_loso.json` | `src/tools/holdout_number_check.py` gates every value Section 5.2 quotes from it; the retraining run itself is not part of this release |
| Table 15 + Figure 8 live budget under a drifting mix | `outputs/sim/operational_sim_{A_strong,B_mild,D_static}.json` | `src/sim/sim_operational.py`, `src/sim/fig_operational.py` |
| Table B.1 sensitivity to the monotone repair | `outputs/killtest/envelope_convention_check.json` | `src/tools/envelope_convention_check.py` |
| Table B.2 uncertainty of the structural parity capacity | `outputs/killtest/parity_capacity_ci.json`, `outputs/textdetox/parity_capacity_ci_*.json` | `src/tools/parity_capacity_ci.py` |
| Table B.3 per-seed quality, seven languages | `outputs/textdetox/seed_quality_spread.json` | `src/tools/seed_quality_spread.py` |
| Table C.1 + Figure C.1 the L-group allocation | `outputs/textdetox/capra_lgroup_xlmr_{base,large}.json`, `outputs/killtest/capra_lgroup_envi.json` | `src/capra_lgroup.py`, `src/fig_lgroup.py` |
| Figures 2–4, 7 (decision outcomes, disparity map, capacity, ablations) | per-detector artefacts above | `src/figs_extra.py`, `src/fig_capacity_by_lang.py` |
| Figure 5 the achievable set | `outputs/killtest/capra.json`, `<detector root>/capra_NAT.json` | `src/fig_capra_front.py` |
| Figure 6 behaviour against the review budget | `outputs/killtest/conra_parity.json`, `outputs/killtest/conra_frontier.json` | `src/fig_conra_frontier.py`, `src/conra_parity.py` |
| Figure 1 architecture (schematic, no data) | — | diagram source, not data-derived |

Two checks verify the manuscript against these artefacts rather than against themselves. Both read the
`.docx`, which is not part of this release, so they are run with the manuscript in hand:

```bash
CAPRA_DOCX=<manuscript .docx> python src/tools/lgroup_number_check.py   # every number quoted from the L-group artefacts
python src/tools/audit_consistency.py --docx <manuscript .docx>         # terminology, notation, number agreement, provenance
```

Notes on conventions that a reader of the code should know, because the manuscript reports them:

* the review budget `C` is a share of traffic; in the cost-aware variant it is a share of review **time**,
  so the realised item share is lower than `C` by construction, and every row of the audit prints both;
* the null policy and the oracle bound are drawn and matched in the unit the budget is counted in;
* the risk curves are estimated on the calibration split, the reported envelope is the running minimum from
  the low-review end, and the raw curve's sensitivity to that repair is Table B.1;
* every allocation is selected on calibration and measured on test with the thresholds frozen.

## Round-3 additions (external review of the manuscript, 2026-10-13)

* **Figure 8 / Table 15 windowing.** `src/sim/fig_operational.py` draws panel A at 6 windows × 2 000 items,
  the windowing Table 15 aggregates (`--windows 6 --window 2000`); panel B plots the **paired** change in
  kept-set risk and in disparity between the frozen and the enforced regime, with the empirical 95 %
  interval as an error bar. Reproduce:
  `python src/sim/fig_operational.py --windows 6 --window 2000 --out <figure path>`.
* **Envelope convention.** `sim_operational.envelope_curve` takes the running minimum from the low-review
  end, matching `capra.py`. Under the previous (index-ascending) convention the envelope was identically ~0
  and every allocation tied at D = 0 — verified inert on these data: the four policies select identical
  thresholds on all five seeds, and `src/r3_sim_diff.py` reports 0 of 120 metric rows plus paired blocks
  changed after the re-run.
* **Envelope vs the raw empirical curve** (`src/probe_r3_envelope_gap.py`, `outputs/r3_envelope_gap.json`):
  the envelope sits at most 0.87 pp (English) and 0.30 pp (Vietnamese) below the raw curve at any review
  share, and at most 0.005 pp at C = 5 %, the shares the reported operating points review.
* **Multi-objective baselines** (`src/moo_baselines.py`, `outputs/r3_moo_baselines.json`): a weighted sum and
  an ε-constraint program on the same curves and the same five seeds. The set of weights that meets a ±0.05
  composition tolerance depends on how the two objectives are normalised (32–36 of 51 by their maxima, 28–33
  by their ranges, 5–7 in raw units); the ε-constraint prices each tolerance in one solve (D* = 4.40 ± 0.27
  pp at ε = 0.02, 4.21 ± 0.26 at 0.05, 3.92 ± 0.30 at 0.10, 3.23 ± 0.29 at 0.20, against 0.95 pp
  unconstrained), while the family answers all of them from one sweep.
* The manuscript-side quality gates (which read the `.docx`) and the packaging scripts are not part of this
  release.

## Round-4 additions (external review of the manuscript, 2026-10-13)

* **Held-out value of the risk ceiling** (`src/ceiling_transfer_check.py`, `outputs/r4_ceiling_transfer.json`):
  the calibration-selected constrained solution's traffic-weighted average risk on test is 6.53 / 6.88 / 4.96
  / 5.65 / 3.11 / 3.78 pp (cost-blind, cost-aware at C = 5, 10, 20 %), 0.07–0.27 pp under its ceiling in
  every cell, with no violation; the unconstrained parity-first point exceeds the ceiling in four cells of
  six.
* **Discretisation scale:** at 5 000 English and 2 672 Vietnamese calibration items the realised review share
  of a reported policy is its design share rounded to 1/n (0.02 pp and 0.037 pp), and the allocation sweep
  step is 0.005 — the three objects (continuous family, estimated curve, realisable threshold pairs) are
  separated in the paper.
