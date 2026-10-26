#!/usr/bin/env python3
"""Per-seed quality spread for the trained detectors on the seven-language benchmark.

Table 8 reports one accuracy, AUC and ECE per detector and target language, computed over the five seeds
pooled. That is a descriptive aggregate over 5n predictions, not an uncertainty estimate: a pooled AUC
mixes scores produced by five different fitted models, so it is not the mean of the per-seed AUCs. This
script recomputes the three metrics per seed with the same definitions the pooled table uses
(`killtest_analyze.conf_err` / `auc` / `ece`) and reports the mean and the sample SD across seeds, so the
appendix can carry the spread and Table 8's note can say which of the two it prints.

    python tools/seed_quality_spread.py                     # writes outputs/textdetox/seed_quality_spread.json
    python tools/seed_quality_spread.py --print             # + a markdown table on stdout
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import killtest_analyze as K  # noqa: E402

DETECTORS = [("XLM-R Base", "outputs/textdetox/xlmr_base"),
             ("XLM-R Large", "outputs/textdetox/xlmr_large")]
LANGS = ["en", "de", "uk", "ru", "ar", "hi", "zh"]
METRICS = ("acc", "auc", "ece")


def seed_quality(seed_dir, lang):
    P, L = K.load(seed_dir)
    key = f"{lang}_test"
    p1, _conf, err = K.conf_err(P[key], L[key])
    return {"acc": float(1 - err.mean()), "auc": float(K.auc(p1, L[key])),
            "ece": float(K.ece(p1, L[key])), "n": int(len(L[key]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="outputs/textdetox/seed_quality_spread.json")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()

    out = {}
    for det, rel in DETECTORS:
        root = os.path.join(a.root, rel)
        seeds = sorted(d for d in glob.glob(os.path.join(root, "seed*")) if os.path.isdir(d))
        if not seeds:
            sys.exit(f"no seed directories under {root}")
        out[det] = {"seeds": [os.path.basename(s) for s in seeds], "langs": {}}
        for lang in LANGS:
            per = [seed_quality(s, lang) for s in seeds]
            entry = {"n": per[0]["n"], "per_seed": per}
            for m in METRICS:
                vals = np.array([p[m] for p in per], dtype=float)
                entry[m] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                            "min": float(vals.min()), "max": float(vals.max())}
            out[det]["langs"][lang] = entry
            print(f"  {det:12s} {lang:3s} n={entry['n']:5d}  "
                  + "  ".join(f"{m}={entry[m]['mean']:.4f}±{entry[m]['sd']:.4f}" for m in METRICS))
    path = os.path.join(a.root, a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")

    if a.print:
        print("\n| detector | language | accuracy | AUC | ECE |")
        print("|---|---|---|---|---|")
        for det in (d for d, _ in DETECTORS):
            for lang in LANGS:
                e = out[det]["langs"][lang]
                print(f"| {det} | {lang} | {e['acc']['mean']*100:.2f} ± {e['acc']['sd']*100:.2f} | "
                      f"{e['auc']['mean']:.4f} ± {e['auc']['sd']:.4f} | "
                      f"{e['ece']['mean']*100:.2f} ± {e['ece']['sd']*100:.2f} |")


if __name__ == "__main__":
    main()
