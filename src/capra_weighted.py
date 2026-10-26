#!/usr/bin/env python3
"""Weighted-loss sensitivity for CAPRA (external review round 8, Major 1).

The shipped analysis scores a kept item with the 0/1 classification error, so a missed toxic item and a
wrongly flagged clean item cost the same. For moderation they do not, and the two benchmarks have
different toxic prevalence (0.076 on English against 0.169 on Vietnamese), so equal residual risk is not
the same amount of protection. This script re-runs the whole allocation on the SAME frozen per-item
scores and with the SAME estimator, under an asymmetric loss

    err_w(item) = k * 1[false negative] + 1 * 1[false positive],      k in {1, 2, 5},

and reports what moves: the parity-first disparity measured on test, the queue composition it produces,
the price of composition, the minimum capacity at which two-sided parity becomes satisfiable, the
pooled-threshold disparity, and the class split of the kept-set risk. k = 1 reproduces the shipped
numbers exactly, which is the check that the machinery is the paper's own.

CPU-only, reads probs.npz artefacts:
  python capra_weighted.py --out-root outputs/killtest/xlmr_base --json outputs/killtest/capra_weighted.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capra as CA            # noqa: E402
import conra_parity as CP     # noqa: E402
import pair_parity as PP      # noqa: E402


def weighted_err(p1, labels, ratio):
    """0/1 error split into its two classes: k * FN + 1 * FP."""
    pred = (p1 >= 0.5).astype(int)
    fn = ((labels == 1) & (pred == 0)).astype(float)
    fp = ((labels == 0) & (pred == 1)).astype(float)
    return ratio * fn + fp, fn, fp


def seed_dirs_of(out_root):
    dirs = sorted(d for d in glob.glob(os.path.join(out_root, "seed*"))
                  if os.path.exists(os.path.join(d, "probs.npz")))
    return dirs or [out_root]


def run(out_root, langs, mode, budget, k_star, ratios):
    a, b = langs
    CP.LANGS = [a, b]
    d = CP.prep(seed_dirs_of(out_root), mode)

    test_w = {}
    res = {"generated": datetime.now().isoformat(timespec="seconds"), "detector_out_root": out_root,
           "pair": [a, b], "mode": mode, "budget": budget, "composition_target": k_star,
           "n": {l: int(len(d["test"][l]["labels"])) for l in langs},
           "prevalence": {l: float(d["test"][l]["labels"].mean()) for l in langs}, "ratios": {}}

    # weight-independent: how much of each class the detector gets wrong on test, and how often
    raw = {}
    for l in langs:
        _, fn, fp = weighted_err(d["test"][l]["p1"], d["test"][l]["labels"], 1.0)
        raw[l] = {"unweighted_err": float(d["test"][l]["err"].mean()),
                  "fn_rate": float(fn.mean()), "fp_rate": float(fp.mean())}

    for ratio in ratios:
        cal_w, test_w = {}, {}
        comp = {}
        for l in langs:
            for part, store in (("cal", cal_w), ("test", test_w)):
                e, fn, fp = weighted_err(d[part][l]["p1"], d[part][l]["labels"], ratio)
                store[l] = dict(d[part][l])
                store[l]["err"] = e
                store[l]["fn"] = fn
                store[l]["fp"] = fp
            comp[l] = raw[l]

        curves = {l: CP.build_curve(cal_w[l]["conf"], cal_w[l]["err"]) for l in langs}
        env = {l: CA.review_risk_curve(cal_w[l]["conf"], cal_w[l]["err"]) for l in langs}
        front = CA.sweep(env, budget, (a, b), {a: 1.0, b: 1.0})
        m = CA.summarise(front, k_star)
        m["price_of_composition_design"] = m["price_of_composition"]   # design side; never the printed price
        for key in ("parity_first", "composition_first"):
            pt = m[key]
            tau = {l: CA.tau_from_share(cal_w[l]["conf"], pt["s"][l]) for l in langs}
            pt["test"] = CA.measure_on_test(tau, test_w, (a, b))
        m["price_of_composition_test"] = (m["composition_first"]["test"]["D_abs"]
                                         - m["parity_first"]["test"]["D_abs"])
        m["price_of_parity_test"] = abs(m["parity_first"]["test"]["K"] - k_star)

        cap = PP.min_parity_capacity(curves)
        tau_g = CP.rule_thresholds("GLOBAL", budget, curves)
        glob_rule = CP.evaluate(tau_g, test_w)

        # class split of the kept set at the pooled rule: the two error classes are what the weighting
        # is about, so they are reported separately rather than only through the pooled rate
        keep_split = {}
        for l in langs:
            keep = test_w[l]["conf"] >= tau_g[l]
            toxic, clean = test_w[l]["labels"] == 1, test_w[l]["labels"] == 0
            keep_split[l] = {
                "kept_n": int(keep.sum()),
                "fn_per_kept_toxic": float(test_w[l]["fn"][keep & toxic].mean()) if (keep & toxic).any() else None,
                "fp_per_kept_clean": float(test_w[l]["fp"][keep & clean].mean()) if (keep & clean).any() else None,
                "weighted_risk": float(test_w[l]["err"][keep].mean()) if keep.any() else None,
            }

        res["ratios"][f"{ratio:g}"] = {
            "weight_fn_over_fp": ratio,
            "design": {k: m[k] for k in ("price_of_composition_design", "feasible_both")},
            "parity_first": {"test_D_abs_pp": 100 * m["parity_first"]["test"]["D_abs"],
                             "test_D_signed_pp": 100 * m["parity_first"]["test"]["D_signed"],
                             "test_K": m["parity_first"]["test"]["K"],
                             "test_load": m["parity_first"]["test"]["load"],
                             "s": m["parity_first"]["s"]},
            "composition_first": {"test_D_abs_pp": 100 * m["composition_first"]["test"]["D_abs"],
                                  "test_K": m["composition_first"]["test"]["K"],
                                  "test_load": m["composition_first"]["test"]["load"]},
            "price_of_composition_test_pp": 100 * m["price_of_composition_test"],
            "price_of_parity_test": m["price_of_parity_test"],
            "min_parity_capacity": {"capacity": cap[0], "r_natural": cap[1], "risk_at_capacity": cap[2]},
            "pooled_rule": {"risk_gap_pp": 100 * glob_rule["risk_gap"],
                            "queue_share_second": glob_rule["queue_share_second"],
                            "review_load": glob_rule["review_load"]},
            "kept_set_class_split": keep_split,
            "component_rates": comp,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--langs", default="en,vi")
    ap.add_argument("--mode", default="NAT")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--composition-target", type=float, default=0.5)
    ap.add_argument("--ratios", default="1,2,5")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    langs = tuple(s.strip() for s in args.langs.split(","))
    ratios = [float(x) for x in args.ratios.split(",")]
    res = run(args.out_root, langs, args.mode, args.budget, args.composition_target, ratios)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(res, open(args.json, "w", encoding="utf-8"), indent=2)

    lines = [f"# Weighted-loss sensitivity — {args.out_root}", "",
             f"Generated {res['generated']} · pair {res['pair']} · mode {res['mode']} · "
             f"C = {res['budget']} · n = {res['n']} · prevalence {res['prevalence']}", "",
             "`k` = weight of a false negative relative to a false positive. `D` is the signed",
             "test disparity of the parity-first allocation (first language minus second), `K` the share",
             "of the review queue read in the second language, `cap.` the smallest capacity at which",
             "two-sided parity is satisfiable, `pooled gap` the disparity of the single pooled threshold.", "",
             "| k | parity-first D (pp) | parity-first K | price of composition (pp) | min parity cap. | pooled gap (pp) |",
             "|---|---|---|---|---|---|"]
    for tag, e in res["ratios"].items():
        pf, cap, pr = e["parity_first"], e["min_parity_capacity"], e["pooled_rule"]
        lines.append(f"| {tag} | {pf['test_D_signed_pp']:+.2f} | {pf['test_K']:.3f} | "
                     f"{e['price_of_composition_test_pp']:.2f} | "
                     f"{('%.1f %%' % (100 * cap['capacity'])) if cap['capacity'] else 'not reached'} | "
                     f"{pr['risk_gap_pp']:+.2f} |")
    lines += ["", "Class split of the kept set at the pooled threshold:", "",
              "| k | language | fn per kept toxic | fp per kept clean | weighted risk |", "|---|---|---|---|---|"]
    for tag, e in res["ratios"].items():
        for l, s in e["kept_set_class_split"].items():
            lines.append(f"| {tag} | {l} | {s['fn_per_kept_toxic']:.4f} | {s['fp_per_kept_clean']:.4f} | "
                         f"{s['weighted_risk']:.4f} |")
    md = "\n".join(lines) + "\n"
    if args.md:
        open(args.md, "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
