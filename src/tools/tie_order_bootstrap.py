"""Tie-order bootstrap: what an arbitrary order inside a tied-confidence block costs.

The reported policies are threshold-only (Section 3.1): an item tied at the threshold stays in the kept
set, so the realised load can fall short of the design budget and the block cannot be split. This tool
measures the other side of that coin — the disparity a deployment would see if it *did* split blocks, by
reviewing exactly the lowest-confidence items up to the design share and breaking every tie uniformly at
random on each draw.

Input is the per-item score artefact of one detector (probs.npz with {lang}_{split}_probs/_labels), so the
probe is reproducible from the release: same file, same seed, same draws.

For every replicate the whole selection is repeated on the calibration split and evaluated on test, so the
spread covers both the tie order and the resulting re-selection. Output JSON carries the draws, their
summary, and the tie census that motivates the analysis (distinct confidence values, largest block) for
the sections that quote it.

Usage:
    python src/tools/tie_order_bootstrap.py --probs outputs/pool/llm_deepseek_flash/probs.npz \
        --budget 0.05 --B 200 --seed 20261026 --out outputs/killtest/tie_order_bootstrap.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import capra as CP          # noqa: E402  (the analysis stack this probe must agree with)

LANGS = ("en", "vi")
K_STAR = 0.5


def load_pair(path, langs=LANGS):
    z = np.load(path, allow_pickle=True)
    cal, test = {}, {}
    coverage = {}
    for l in langs:
        for split, box in (("dev", cal), ("test", test)):
            p = z[f"{l}_{split}_probs"]
            y = z[f"{l}_{split}_labels"]
            p1 = p[:, 1] if p.ndim == 2 else p
            ok = np.isfinite(p1)                      # items the provider refused stay NaN in pool_llm
            coverage[f"{l}_{split}"] = {"n_scored": int(ok.sum()), "n_total": int(len(p1)),
                                        "unscored": int((~ok).sum())}
            p1, y = p1[ok], y[ok]
            box[l] = {"conf": np.maximum(p1, 1.0 - p1), "err": ((p1 >= 0.5).astype(int) != y).astype(float)}
    return cal, test, coverage


def shuffle_within_ties(conf, rng):
    """Indices that sort by confidence and break every tie uniformly at random.

    review_risk_curve sorts with a stable mergesort, so passing the arrays through this permutation is
    exactly 'random tie order, same confidence order'.
    """
    return np.lexsort((rng.random(len(conf)), conf))


def curve_of(conf, err, order):
    return CP.review_risk_curve(conf[order], err[order])


def risk_at_share_ordered(conf, err, order, s):
    """Kept-set risk when exactly round(s*n) lowest-confidence items are reviewed, ties already ordered."""
    c, e = conf[order], err[order]
    n = len(c)
    k = int(np.rint(min(max(s, 0.0), 1.0) * n))
    k = min(max(k, 0), n - 1) if n > 1 else 0
    kept = e[k:]
    return float(kept.mean()) if kept.size else 0.0


def threshold_baseline(cal, test, budget):
    """The policy the paper actually reports: a frozen threshold, ties left in the kept set.

    This is capra.py's own pipeline (curves, sweep, endpoints, tau_from_share, measure_on_test) run on the
    same arrays, so its number is the one the tables quote and the tie-order draws are read against it.
    """
    order = {l: np.argsort(-cal[l]["conf"], kind="mergesort") for l in LANGS}
    curves = {l: curve_of(cal[l]["conf"], cal[l]["err"], order[l]) for l in LANGS}
    front = CP.sweep(curves, budget, LANGS, {l: 1.0 for l in LANGS})
    parity, _comp = CP.endpoints(front, K_STAR)
    tau = {l: CP.tau_from_share(cal[l]["conf"], parity["s"][l]) for l in LANGS}
    m = CP.measure_on_test(tau, test, LANGS)
    return {"alpha": parity["alpha"], "s": dict(parity["s"]), "D_cal": parity["D_abs"],
            "K_cal": parity["K"], "tau": {l: float(tau[l]) for l in LANGS},
            "D_test": float(m["D_abs"]), "r_test": {l: float(m[l]["risk"]) for l in LANGS},
            "load_test": float(m["load"])}


def one_replicate(cal, test, budget, rng, shuffle=True):
    order_cal, order_test = {}, {}
    for l in LANGS:
        order_cal[l] = shuffle_within_ties(cal[l]["conf"], rng) if shuffle \
            else np.argsort(-cal[l]["conf"], kind="mergesort")
        order_test[l] = shuffle_within_ties(test[l]["conf"], rng) if shuffle \
            else np.argsort(-test[l]["conf"], kind="mergesort")
    curves = {l: curve_of(cal[l]["conf"], cal[l]["err"], order_cal[l]) for l in LANGS}
    costs = {l: 1.0 for l in LANGS}
    front = CP.sweep(curves, budget, LANGS, costs)
    parity, comp = CP.endpoints(front, K_STAR)
    out = {"alpha": parity["alpha"], "s": dict(parity["s"]),
           "D_cal": parity["D_abs"], "K_cal": parity["K"]}
    r = {l: risk_at_share_ordered(test[l]["conf"], test[l]["err"], order_test[l], parity["s"][l])
         for l in LANGS}
    out["r_test"] = r
    out["D_test"] = abs(r[LANGS[0]] - r[LANGS[1]])
    return out


def census(conf):
    vals, counts = np.unique(conf, return_counts=True)
    return {"n": int(len(conf)), "distinct": int(len(vals)), "largest_tie": int(counts.max()),
            "median_tie": float(np.median(counts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--grid", type=int, default=0,
                    help="quantise confidences to `grid` integer levels first (100 = the moderator's "
                         "integer scale); 0 keeps the raw scores")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cal, test, coverage = load_pair(args.probs)
    if args.grid:
        for box in (cal, test):
            for l in LANGS:
                q = np.rint(box[l]["conf"] * args.grid) / args.grid
                box[l] = {"conf": q, "err": box[l]["err"]}

    base = threshold_baseline(cal, test, args.budget)
    share_base = one_replicate(cal, test, args.budget, np.random.default_rng(args.seed), shuffle=False)
    rng = np.random.default_rng(args.seed)
    draws = [one_replicate(cal, test, args.budget, rng) for _ in range(args.B)]
    d_test = np.array([d["D_test"] for d in draws])
    d_cal = np.array([d["D_cal"] for d in draws])

    def stab(k):
        return {"B": k, "mean": float(d_test[:k].mean()), "sd": float(d_test[:k].std(ddof=1)) if k > 1 else 0.0}

    payload = {
        "tool": "tie_order_bootstrap",
        "probs": os.path.abspath(args.probs),
        "budget": args.budget, "B": args.B, "seed": args.seed, "grid": args.grid,
        "langs": list(LANGS), "k_star": K_STAR,
        "coverage": coverage,
        "fixed_order": base,
        "fixed_order_share": share_base,
        "census": {"cal": {l: census(cal[l]["conf"]) for l in LANGS},
                   "test": {l: census(test[l]["conf"]) for l in LANGS}},
        "summary_test": {"mean": float(d_test.mean()), "sd": float(d_test.std(ddof=1)),
                         "min": float(d_test.min()), "max": float(d_test.max()),
                         "p2_5": float(np.percentile(d_test, 2.5)),
                         "p97_5": float(np.percentile(d_test, 97.5))},
        "summary_cal": {"mean": float(d_cal.mean()), "sd": float(d_cal.std(ddof=1))},
        "stability": [stab(max(1, args.B // 8)), stab(max(1, args.B // 4)),
                      stab(max(1, args.B // 2)), stab(args.B)],
        "draws": draws,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"threshold policy: D_test = {base['D_test']*100:.3f} pp (alpha {base['alpha']:.3f}, K {base['K_cal']:.3f})")
    print(f"fixed order, tie-splitting: D_test = {share_base['D_test']*100:.3f} pp")
    print(f"{args.B} draws:      mean {d_test.mean()*100:.3f} pp, sd {d_test.std(ddof=1)*100:.3f} pp, "
          f"[{np.percentile(d_test,2.5)*100:.3f}, {np.percentile(d_test,97.5)*100:.3f}] pp")
    print("cal draws:        mean %.3f pp, sd %.3f pp" % (d_cal.mean()*100, d_cal.std(ddof=1)*100))
    print("census cal:", {l: payload['census']['cal'][l] for l in LANGS})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
