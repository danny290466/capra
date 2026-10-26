# Repeat pass of the hosted moderator (evidence, not a second source of truth)

This directory holds one **repeat pass** of the DeepSeek Flash moderator on the English–Vietnamese
benchmark, run to bound how far a hosted model's numbers move when the same prompt, model identifier,
temperature and data are re-run. It does **not** replace the pass behind the paper's tables: the tabulated
moderator row stays the original run, and this directory exists so the claimed spread can be inspected.

## What was run

| Step | Command / input |
|---|---|
| Data | `src/killtest_data.py` over Civil Comments (60k/5k/5k) + ViHSD; splits reproduce the original counts exactly: 5 000 / 5 000 / 2 672 / 6 680 |
| Scoring | `src/pool_llm.py --model deepseek-flash --base-url https://opencode.ai/zen/go/v1 --thinking disabled --max-tokens 120 --sets en_dev,en_test,vi_dev,vi_test` |
| Coverage | 19 352 items; 16 empty completions (provider refused), 5 recovered with a longer budget or without JSON mode, **11 permanently refused** (0.057 %) |
| Analysis | `src/capra.py --out-root <this pass> --langs en,vi --mode NAT` |
| Tie probe | `src/tools/tie_order_bootstrap.py --B 200 --seed 20261026 --budget 0.05` |

Artifacts released here: `rebuild_capra.json` (the allocation output of the repeat pass),
`tie_order_bootstrap.json` (the tie probe: its 200 draws plus the census of distinct confidence values) and
`usage.json` (model identifier, temperature, token budget, per-split counts). The pass's per-item scores and
per-item records are **not** redistributed, following the Data availability statement: the hosted moderator's
judgements come from a third-party API, so what ships is the aggregate, the draw set and the census, and every
number the paper quotes from this pass is read from those three files.

## What it measures

Against the pass behind the tables, at the printed precision:

* **11 of 48** parity-first/composition-first quantities move a printed digit.
* Review-queue share moves by up to **0.31**.
* Composition-first disparity moves by up to **1.38 pp**.
* 11 of 19 352 items are refused outright on the repeat pass.

The qualitative picture does not move: the parity-first disparity stays near zero on calibration
(0.003 pp here against 0.012 pp), and the queue remains far from an even split.

## Tie probe (why the paper no longer quotes a 200-draw figure)

Two properties of this score matrix, from `tie_order_bootstrap.json`:

* 19 distinct confidence values over 4 995 English calibration items, 17 over 2 672 Vietnamese,
  with the largest single block covering **1 721** and **816** items. Integer elicitation makes blocks
  that large unavoidable.
* If a deployment split those blocks at the design share, the holding of the block is arbitrary: with the
  file order the parity-first test disparity is **0.990 pp**, and over 200 random orders it is
  **0.535 pp** on average (SD 0.077, 95 % [0.373, 0.712]). The threshold-only rule the paper reports gives
  **0.449 pp** on this pass.

Because the probe's input is a repeat pass, its numbers describe the mechanism rather than the tabulated
row, which is why Section 5.1 reports the threshold-only rule and Section 7 reports the spread instead of
quoting a tie-splitting figure.

## Reproducing

```bash
python src/killtest_data.py --force                     # needs data/cc and data/vihsd (public sources)
python src/pool_llm.py --model deepseek-flash \
    --base-url https://opencode.ai/zen/go/v1 --key-env OPENCODE_GO_API_KEY --opencode \
    --thinking disabled --max-tokens 120 --sets en_dev,en_test,vi_dev,vi_test \
    --data-dir data/killtest --out outputs/pool/llm_deepseek_flash_rebuild
python src/capra.py --out-root outputs/pool/llm_deepseek_flash_rebuild --langs en,vi --mode NAT \
    --json outputs/killtest/rebuild_llm/rebuild_capra.json
# re-scoring first is a prerequisite: the tie probe reads the pass's per-item scores, which are not part of
# this release (see the artifact note above)
python src/tools/tie_order_bootstrap.py --probs outputs/pool/llm_deepseek_flash_rebuild/probs.npz \
    --budget 0.05 --B 200 --seed 20261026 --out outputs/killtest/tie_order_bootstrap.json
```

The scoring step is the only non-deterministic one; every number downstream of the pass's per-item scores
re-derives exactly, which is the property `tools/` and the aggregate JSONs are checked against.
