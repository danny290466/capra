# CAPRA — analysis code and frozen artefacts

Analysis code and the frozen artefacts behind **CAPRA (Capacity-Aware Parity Risk Allocation)**: the
allocation of a bounded human-review budget between two (or L) language populations, its achievable set,
the price of parity and the price of composition. Every number, table and figure of the manuscript is
reproducible from what is in this repository without retraining anything: the per-item detector scores and
the per-seed metrics are shipped, so the tables are a pure function of these files.

The method is detector-agnostic; the empirical study here uses English–Vietnamese toxic-content data and a
seven-language benchmark (see `data/README.md`).

## What is in here

| path | what |
|---|---|
| `src/` | the analysis code: the two-group allocation sweep and its prices (`capra.py`), the L-group solver (`capra_lgroup.py`), the prior-art families including the null and the oracle bound (`baseline_rules.py`), the item-level bootstrap and seed-level tests (`significance.py`), the ablations (`ablate.py`), the figure generators (`figs_extra.py`, `fig_capra_front.py`, `fig_lgroup.py`, …) |
| `src/sim/` | the live-budget simulation (drifting traffic mix, frozen vs enforced capacity) and its figure |
| `src/tools/` | the small artefact-side utilities: `lgroup_number_check.py`, `parity_capacity_ci.py`, `seed_quality_spread.py`, `envelope_convention_check.py`, `detector_cost_table.py`. `audit_consistency.py` and `holdout_number_check.py` are the manuscript-side gates and read the `.docx`, so they cannot run from this release alone. |
| `src/drivers/` | the shell drivers that re-derive every affected artefact in order, with the exact command line for each. `rerun_r2.sh` is the authoritative command line for the shipped artefacts and uses the same roots as this README. |
| `outputs/killtest/` | the flagship English–Vietnamese detector (XLM-R base): allocation fronts, prior-art audits, significance runs, ablations, the operating modes, the EN–VI L-group run, the envelope and ceiling probes, and `seed42…seed46/{probs.npz,metrics.json}` |
| `outputs/pool/<detector>/` | the other English–Vietnamese detectors (`roberta_en`, `phobert_vi`, `specialist_pair`, `xlmr_large`, `llm_deepseek_flash`, `llm_gemini31_flashlite`, `llm_qwen38_flash`): the same front, prior-art audit and significance artefacts, per detector |
| `outputs/textdetox/<detector>/` | the seven-language benchmark (`xlmr_base`, `xlmr_large`, `llm_deepseek_flash`): per-pair allocations, `pair_parity.json`, `baseline.json`, `ablations.json`, `significance.json` and the per-seed scores |
| `outputs/killtest/*.json`, `outputs/textdetox/*.json` | the detector-level analyses that sit beside the per-detector directories: the L-group runs, the parity-capacity intervals, the reference-swap probes, the per-seed quality spread, the detector costs |
| `outputs/sim/` | the live-budget simulation runs (`operational_sim_{A_strong,B_mild,D_static}.json`) |
| `outputs/r3_*.json`, `outputs/r4_ceiling_transfer.json` | the add-on probes: the envelope-versus-raw gap, the envelope convention check, the multi-objective baselines, and the held-out value of the risk ceiling |
| `notes/` | the generated report tables the manuscript was written from (`pool_compare_RESULT.md`, `capra_*_NAT.md`, `baseline_*.md`, `significance_*.md`, `ablations_*.md`, `lgroup_*.md`) and `lolo_loso.json`, the held-out-language control measurements Section 5.2 quotes |
| `DIGEST.md` | the mapping table: manuscript table/figure → artefact file → the command that produces it |

Per-detector artefacts share one layout: `<detector root>/seed*/probs.npz` carries the per-item scores for
every language of that benchmark (`<lang>_dev_probs`, `<lang>_test_probs` and their labels, plus the
translated `tr_*` sets on the English–Vietnamese benchmark), and `<detector root>/seed*/metrics.json`
carries accuracy, AUC and ECE per language and seed.

Not included: the manuscript itself, the raw datasets (both are public — see `data/README.md`), the
detector checkpoints, and the run logs the training-time figures were read from. Nothing in the paper
depends on the checkpoints: the shipped per-item scores are what the allocation consumes.

## Reproducing a table or a figure

1. Get the environment (below) and the per-item scores, which are already in `outputs/`.
2. Point the scripts at a detector root; the arguments are the ones in `src/drivers/rerun_r2.sh` — the
   drivers are the authoritative command line for each artefact, e.g.

```bash
python src/capra.py        --out-root outputs/killtest     --langs en,vi --mode NAT \
                           --out notes/capra_killtest_NAT.md --json outputs/killtest/capra.json
python src/baseline_rules.py --out-root outputs/killtest   --lang-a en --lang-b vi --mode NAT \
                           --out notes/baseline_killtest_NAT.md --json outputs/killtest/baseline_NAT.json
python src/significance.py --out-root outputs/killtest     --lang-a en --lang-b vi --mode NAT \
                           --json outputs/killtest/significance_NAT.json
python src/capra_lgroup.py --out-root outputs/textdetox/xlmr_base \
                           --langs en,de,uk,ar,hi,zh,ru --budget 0.05 \
                           --json outputs/textdetox/capra_lgroup_xlmr_base.json
python src/sim/sim_operational.py --dir outputs/killtest --sched 100 --amp 0.18 --burst 0.25 \
                           --window 2000 --windows 6 --out outputs/sim/operational_sim_A_strong.json
```

A detector root is `outputs/killtest`, `outputs/pool/<detector>` or `outputs/textdetox/<detector>`, as
above. The drivers were written on the machine that produced the paper and carry absolute paths for the raw
data; the `--out-root`/`--dir` arguments are relative to this repository. The scripts write their report
tables to `notes/` and their JSON to wherever `--json` points, so re-running overwrites nothing in place
unless you ask it to.

Determinism: every random step is seeded (the allocation sweep is a grid, the bootstrap and the simulation
draws carry explicit seeds in the source), so a re-run reproduces the shipped artefacts.

## Environment

The artefacts were produced with Python 3.10 and:

```
numpy 2.2.6 · scipy 1.15.3 · matplotlib 3.10.9 · scikit-learn 1.7.2
```

`requirements.txt` lists these; `torch`/`transformers` are needed only if you want to *regenerate* the
per-item scores from the raw data (the scripts for that are in `src/` too: `killtest_data.py`,
`killtest_train.py`, `textdetox_prep.py`), not to reproduce any table.

## Consistency checks

The manuscript quotes a large number of derived quantities; the artefact side of that bookkeeping is
included, and both checks below read the manuscript itself as well:

```bash
CAPRA_DOCX=<manuscript .docx> python src/tools/lgroup_number_check.py       # every number Appendix C quotes
python src/tools/audit_consistency.py --docx <manuscript .docx>             # terminology, notation, number agreement, provenance
```

`lgroup_number_check.py` reads the shipped L-group artefacts
(`outputs/textdetox/capra_lgroup_xlmr_{base,large}.json`, `outputs/killtest/capra_lgroup_envi.json`); both
scripts need the `.docx`, which is not part of this release, so they are run with the manuscript in hand
rather than from the release alone.

## Data

Both benchmarks are public and are not redistributed here: Civil Comments (Borkan et al., 2019), ViHSD
(Luu et al., 2021) and the multilingual toxicity benchmark released with the PAN 2024 TextDetox task
(Dementieva et al., 2024). `data/README.md` says which files the pipeline expects and how the splits are
built.

## Licence and citation

Code and artefacts: Creative Commons Attribution 4.0 (`LICENSE`), attributed to the authors of the CAPRA
manuscript. See `CITATION.cff`. The paper is under review; please cite the repository only (no DOI yet).
