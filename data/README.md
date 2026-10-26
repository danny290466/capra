# Data

The two benchmarks are public and are **not** redistributed here. This repository ships the per-item
detector scores (`artefacts/detectors/*/seed*/probs.npz`) and the per-seed metrics, which is what every
table in the paper is computed from; the data scripts below rebuild the splits from the original sources if
you want to retrain the detectors.

| benchmark | source | used for |
|---|---|---|
| Civil Comments (English) | Borkan et al. (2019), *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification* | English side of the English–Vietnamese setting; an item is positive when the fraction of annotators marking it toxic reaches 0.5 |
| ViHSD (Vietnamese) | Luu et al. (2021), *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts* | Vietnamese side; the positive class is the union of the OFFENSIVE and HATE labels |
| PAN 2024 TextDetox multilingual toxicity benchmark | Dementieva et al. (2024), *Overview of the multilingual text detoxification task at PAN 2024* | the seven-language setting: German, Ukrainian, Russian, Arabic, Hindi and Chinese, each against English; one shared annotation protocol across languages |
| OPUS-MT (translation) | Tiedemann & Thottingal (2020) | the machine-translated control (English test items rendered into Vietnamese) |

## Splits and what the pipeline expects

* `src/killtest_data.py` builds the English–Vietnamese splits (60 000 / 5 000 / 5 000 items for English
  after de-duplication, and the capped Vietnamese splits of Table 2) and writes the `*.jsonl` files the
  training scripts read.
* `src/textdetox_prep.py` builds the seven-language splits (675 items per language per split in the
  released version; Chinese carries 676).
* `src/killtest_train.py` and `src/pool_train.py` fine-tune the trained encoders and write
  `seed*/probs.npz` and `seed*/metrics.json` per training seed; `src/pool_llm.py` collects the hosted
  moderators' scores the same way. The hosted moderators are called through their public APIs, whose
  prompts and decoding settings are in Appendix D of the manuscript.
* The machine-translated control is produced by `src/killtest_translate_pair.py`.

## Licences of the source data

Each dataset keeps its own licence and terms; check them at the sources above before redistributing
anything derived from the raw text. This repository redistributes no raw text — only model scores and
aggregate metrics.
