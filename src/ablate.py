"""Ablations for the parity study (reviewer-defence package).

Five knobs a method-journal reviewer will pull on:

  A. composition target K* in {0.4, 0.5, 0.6}   -> how much disparity does hitting a stricter/looser
                                                   queue composition cost?
  B. review cost c_second in {1.0, 1.3, 1.5, 2.0} -> does the cost-aware budget change the verdict?
  C. traffic mix w_second in {0.3, 0.5, 0.7}    -> is the result an artefact of a balanced mix?
  D. calibration size n in {250, 500, 1000, 2000, full} -> how much do the fitted thresholds and the
                                                   minimum parity capacity move with less dev data?
  E. seed spread (the real model seeds)         -> is the measured disparity stable, or one lucky run?

Everything is CPU-only and reads the same probs.npz artefacts; the test split is never resampled.
Usage:
  python ablate.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
      --out notes/ablations_killtest.md --json outputs/killtest/ablations.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capra as CA          # noqa: E402
import conra_parity as CP   # noqa: E402

SIZES = [250, 500, 1000, 2000, 0]          # 0 = full dev split
DRAWS = 3


def load_pair(out_root, a, b, mode, seed_subset=None):
    """Pooled (all seeds) or single-seed dev/test arrays for a pair."""
    seed_dirs = sorted(d for d in glob.glob(os.path.join(out_root, "seed*"))
                       if os.path.exists(os.path.join(d, "probs.npz")))
    if not seed_dirs and os.path.exists(os.path.join(out_root, "probs.npz")):
        seed_dirs = [out_root]
    assert seed_dirs, f"no probs.npz under {out_root}"
    if seed_subset is not None:
        seed_dirs = [seed_dirs[seed_subset]]
    CP.LANGS = [a, b]
    if (a, b) == ("en", "vi"):
        return CP.prep(seed_dirs, mode), seed_dirs
    import pair_parity as PP
    return PP.build_pair(seed_dirs, a, b), seed_dirs


def curves_of(d, langs):
    return {l: CA.review_risk_curve(d["cal"][l]["conf"], d["cal"][l]["err"]) for l in langs}


def test_metrics(d, tau, langs):
    t = CA.measure_on_test(tau, d["test"], langs)
    return float(t["D_abs"]), float(t["K"]), float(t["load"]), float(t["D_signed"])


def parity_first(curves, d, langs, budget, costs, weights=(0.5, 0.5)):
    front = CA.sweep(curves, budget, langs, costs, weights)
    feas = [p for p in front if not p["clipped"]] or front
    p = min(feas, key=lambda x: x["D_abs"])
    tau = {l: CA.tau_from_share(d["cal"][l]["conf"], p["s"][l]) for l in langs}
    D, K, load, D_signed = test_metrics(d, tau, langs)
    return {"alpha": p["alpha"], "D_design": p["D_abs"], "K_design": p["K"], "load": p["load"],
            "D_test": D, "D_signed_test": D_signed, "K_test": K, "s": {l: p["s"][l] for l in langs}}


def comp_first(curves, d, langs, budget, costs, k_star, weights=(0.5, 0.5)):
    front = CA.sweep(curves, budget, langs, costs, weights)
    feas = [p for p in front if not p["clipped"]] or front
    p = min(feas, key=lambda x: abs(x["K"] - k_star))
    tau = {l: CA.tau_from_share(d["cal"][l]["conf"], p["s"][l]) for l in langs}
    D, K, load, _ = test_metrics(d, tau, langs)
    pf = parity_first(curves, d, langs, budget, costs, weights)   # same split for both endpoints
    # the price is a TEST-minus-TEST difference: both endpoints are selected on calibration and then
    # measured on test, so mixing a design value into the difference is not a price (review R2-02/R3-03)
    return {"alpha": p["alpha"], "D_design": p["D_abs"], "K_design": p["K"],
            "D_test": D, "K_test": K, "price_of_composition": D - pf["D_test"]}


def min_capacity(d, langs):
    """Minimum parity capacity, from the calibration split, through the SHARED curve loader.

    The previous version hand-built a {raw, risk} dict for pair_parity.min_parity_capacity, which
    reaches conra_parity.tau_at_risk and reads curve["conf"]: the KeyError was swallowed and every cell
    of the capacity column printed as NaN (review findings R2-18, R3-04). It also called float() on a
    None result for censored pairs. Both are fixed here; a censored pair now returns NaN on purpose and
    is counted in `censored`, never silently.
    """
    import pair_parity as PP
    CP.LANGS = list(langs)
    cp_curves = {l: CP.build_curve(d["cal"][l]["conf"], d["cal"][l]["err"]) for l in langs}
    cap, _r_nat, _r_par = PP.min_parity_capacity(cp_curves)
    return float("nan") if cap is None else float(cap)


def subsample_cal(d, langs, n, rng, n_seeds=1):
    """Same n per language (binding convention), seeded, and drawn from UNIQUE items.

    For a trained detector the calibration pool is the dev split concatenated once per training seed, so
    sampling from it directly can put the same item in a "n = 250" draw several times - the same
    pseudo-replication defect the test bootstrap was fixed for (review findings R2-13, R3-17). One
    seed's block is used instead.
    """
    cal = {}
    for l in langs:
        conf, err = d["cal"][l]["conf"], d["cal"][l]["err"]
        # the dedupe exists so a SUBSAMPLE cannot draw the same item twice; the full pool must stay the
        # full pool, otherwise the n = full row describes one seed's block and no longer reconciles with
        # the headline configuration (Round-3 finding R3A-14)
        if n and n_seeds > 1 and len(conf) % n_seeds == 0:
            k = len(conf) // n_seeds
            conf, err = conf[:k], err[:k]
        if n and n < len(conf):
            idx = np.sort(rng.choice(len(conf), size=n, replace=False))
            cal[l] = {"conf": conf[idx], "err": err[idx]}
        else:
            cal[l] = {"conf": conf, "err": err}
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lang-a", default="en")
    ap.add_argument("--lang-b", required=True)
    ap.add_argument("--mode", default="NAT")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--composition-target", type=float, default=0.5)
    ap.add_argument("--cost-second", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    a = args.lang_a
    C = args.budget
    K_STAR = args.composition_target
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "detector": args.out_root,
           "lang_a": a, "mode": args.mode, "budget": C, "pairs": {}}
    lines = [f"# Ablations — `{args.out_root}` ({a} vs targets, mode {args.mode}, budget C = {C:.2f})\n",
             f"Generated {out['generated']}. Parity-first = min |risk gap| under the budget; "
             f"composition-first = queue share K* = {K_STAR:.2f}. All numbers on the TEST split after "
             f"fitting on calibration.\n"]

    for b in [s.strip() for s in args.lang_b.split(",") if s.strip()]:
        langs = (a, b)
        d, seed_dirs = load_pair(args.out_root, a, b, args.mode)
        n_seeds = len(seed_dirs)
        cur = curves_of(d, langs)
        R = {}

        # ---- A. composition target sweep
        R["composition_target"] = {}
        for k in (0.4, 0.5, 0.6):
            pf = parity_first(cur, d, langs, C, {a: 1.0, b: 1.0})
            cf = comp_first(cur, d, langs, C, {a: 1.0, b: 1.0}, k)
            R["composition_target"][f"{k:.1f}"] = {"parity_first_D_test": pf["D_test"],
                                                  "composition_first_D_test": cf["D_test"],
                                                  "composition_first_K_test": cf["K_test"],
                                                  "price_of_composition": cf["price_of_composition"]}

        # ---- B. review-cost sweep
        R["review_cost"] = {}
        for c in (1.0, 1.3, 1.5, 2.0):
            pf = parity_first(cur, d, langs, C, {a: 1.0, b: c})
            cf = comp_first(cur, d, langs, C, {a: 1.0, b: c}, K_STAR)
            R["review_cost"][f"{c:.1f}"] = {"parity_first_D_test": pf["D_test"],
                                            "composition_first_D_test": cf["D_test"],
                                            "price_of_composition": cf["price_of_composition"],
                                            "parity_first_load": pf["load"]}

        # ---- C. traffic-mix sweep
        R["traffic_mix"] = {}
        for w in (0.3, 0.5, 0.7):
            pf = parity_first(cur, d, langs, C, {a: 1.0, b: 1.0}, (1.0 - w, w))
            R["traffic_mix"][f"{w:.1f}"] = {"parity_first_D_test": pf["D_test"],
                                            "parity_first_D_signed_test": pf["D_signed_test"],
                                            "parity_first_K_test": pf["K_test"],
                                            "min_capacity": min_capacity(d, langs)}

        # ---- D. calibration-size sweep
        R["calibration_size"] = {}
        for n in SIZES:
            Ds, Cs = [], []
            for draw in range(DRAWS):
                # seed by (n, draw): the old seed 1000 + draw + len(str(n)) gave the SAME three streams
                # to n = 250 and n = 500, so the draws were not independent (review finding R3-28)
                rng = np.random.default_rng(1000 + (n or 0) * 100 + draw)
                cal = subsample_cal(d, langs, n, rng, n_seeds=n_seeds)
                cur_n = {l: CA.review_risk_curve(cal[l]["conf"], cal[l]["err"]) for l in langs}
                dd = {"cal": cal, "test": d["test"]}
                pf = parity_first(cur_n, dd, langs, C, {a: 1.0, b: 1.0})
                Ds.append(pf["D_test"])
                Cs.append(min_capacity(dd, langs))
            n_unique = len(subsample_cal(d, langs, None, np.random.default_rng(0), n_seeds=n_seeds)[a]["conf"])
            R["calibration_size"][str(n) if n else "full"] = {
                "D_test_mean": float(np.mean(Ds)), "D_test_sd": float(np.std(Ds, ddof=1)) if len(Ds) > 1 else 0.0,
                "min_capacity_mean": float(np.nanmean(Cs)),
                "min_capacity_n_valid": int(np.sum(np.isfinite(Cs))),
                "n_dev": int(n_unique), "n_dev_pool": int(len(d["cal"][a]["conf"])),
                "draws": [float(x) for x in Ds]}

        # ---- E. seed spread
        R["seed_spread"] = []
        for i, sd in enumerate(seed_dirs):
            d_i, _ = load_pair(args.out_root, a, b, args.mode, seed_subset=i)
            cur_i = curves_of(d_i, langs)
            pf = parity_first(cur_i, d_i, langs, C, {a: 1.0, b: 1.0})
            R["seed_spread"].append({"seed_dir": os.path.basename(sd), "D_test": pf["D_test"],
                                     "K_test": pf["K_test"],
                                     "signed_gap": pf["D_signed_test"]})
        Ds = [x["D_test"] for x in R["seed_spread"]]
        Ks = [x["K_test"] for x in R["seed_spread"]]
        R["seed_summary"] = {"n_seeds": len(Ds), "D_mean": float(np.mean(Ds)),
                             "D_sd": float(np.std(Ds, ddof=1)) if len(Ds) > 1 else 0.0,
                             "K_mean": float(np.mean(Ks)),
                             "K_sd": float(np.std(Ks, ddof=1)) if len(Ks) > 1 else 0.0}

        out["pairs"][b] = R
        lines += [f"\n## target: {b}  (n_dev = {R['calibration_size']['full']['n_dev']}, "
                  f"seeds = {R['seed_summary']['n_seeds']})\n",
                  "\n**A. composition target**\n",
                  "| K* | parity-first D test (pp) | composition-first D test (pp) | its K test | price of composition (pp) |",
                  "|---|---|---|---|---|"]
        for k, m in R["composition_target"].items():
            lines.append(f"| {k} | {m['parity_first_D_test']*100:.2f} | {m['composition_first_D_test']*100:.2f} | "
                         f"{m['composition_first_K_test']:.3f} | {m['price_of_composition']*100:.2f} |")
        lines += ["\n**B. review cost of the second language**\n",
                  "| c_second | parity-first D test (pp) | composition-first D test (pp) | price of composition (pp) |",
                  "|---|---|---|---|"]
        for c, m in R["review_cost"].items():
            lines.append(f"| {c}× | {m['parity_first_D_test']*100:.2f} | "
                         f"{m['composition_first_D_test']*100:.2f} | {m['price_of_composition']*100:.2f} |")
        lines += ["\n**C. traffic mix**\n",
                  "| w_second | parity-first D test (pp) | parity-first K test | min parity capacity |",
                  "|---|---|---|---|"]
        for w, m in R["traffic_mix"].items():
            lines.append(f"| {w} | {m['parity_first_D_test']*100:.2f} | {m['parity_first_K_test']:.3f} | "
                         f"{m['min_capacity']*100:.1f} % |")
        lines += ["\n**D. calibration size** (mean ± SD over %d draws, same n per language)\n" % DRAWS,
                  "| n_dev | parity-first D test (pp) | min parity capacity |",
                  "|---|---|---|"]
        for n, m in R["calibration_size"].items():
            sd = f" ± {m['D_test_sd']*100:.2f}" if m["D_test_sd"] else ""
            lines.append(f"| {n} | {m['D_test_mean']*100:.2f}{sd} | {m['min_capacity_mean']*100:.1f} % |")
        lines += ["\n**E. seed spread** (real training seeds)\n",
                  f"D test = {R['seed_summary']['D_mean']*100:.2f} ± {R['seed_summary']['D_sd']*100:.2f} pp, "
                  f"K test = {R['seed_summary']['K_mean']:.3f} ± {R['seed_summary']['K_sd']:.3f} "
                  f"over {R['seed_summary']['n_seeds']} seeds\n",
                  "| seed dir | parity-first D test (pp) | K test |", "|---|---|---|"]
        for m in R["seed_spread"]:
            lines.append(f"| {m['seed_dir']} | {m['D_test']*100:.2f} | {m['K_test']:.3f} |")
        print(f"[{args.out_root}] {a}/{b} ablations done", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w", encoding="utf-8"), indent=2)
        print("json ->", args.json, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print("report ->", args.out, flush=True)


if __name__ == "__main__":
    main()
