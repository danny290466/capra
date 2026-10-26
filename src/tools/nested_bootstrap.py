"""Nested bootstrap for the CAPRA audit quantities (reviewer item e / Appendix B).

Two uncertainty sources are propagated together: the five fitted detectors are resampled (the training
component) and the items are resampled within each language (the sampling component), and the allocation
is RESELECTED inside every replicate. Reported quantities:

    composition price      D at the composition-first point minus D at the parity-first point
    parity-first disparity D at the parity-first point
    queue share            K at the parity-first point
    satisfiability cap.    smallest budget whose parity-first design gap is within the 0.5 pp guard

Usage: python src/tools/nested_bootstrap.py --out outputs/killtest/nested_bootstrap.json --B 200
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import capra as CA  # noqa: E402

C_DEFAULT = 0.05
GUARD = 0.005          # 0.5 pp
K_STAR = 0.5
BUDGET_GRID = np.round(np.arange(0.005, 0.4001, 0.005), 4)


def load_seed(path, langs=("en", "vi")):
    z = np.load(path)
    out = {}
    for l in langs:
        for split in ("dev", "test"):
            p = z[f"{l}_{split}_probs"]
            p1 = p[:, 1] if p.ndim == 2 else p
            out[(l, split)] = {"conf": np.maximum(p1, 1 - p1),
                               "err": ((p1 >= 0.5).astype(int) != z[f"{l}_{split}_labels"]).astype(float)}
    return out


def pooled(seeds, seed_idx, split, lang, item_idx):
    confs, errs = [], []
    for s in seed_idx:
        d = seeds[s][(lang, split)]
        confs.append(d["conf"][item_idx])
        errs.append(d["err"][item_idx])
    return {"conf": np.concatenate(confs), "err": np.concatenate(errs)}


def curves_of(cal, langs):
    return {l: CA.review_risk_curve(cal[l]["conf"], cal[l]["err"]) for l in langs}


def capacity_of(curves, langs, budget_grid=BUDGET_GRID):
    for b in budget_grid:
        front = CA.sweep(curves, float(b), langs, costs={l: 1.0 for l in langs})
        pf, _ = CA.endpoints(front, K_STAR)
        if pf["D_abs"] <= GUARD + 1e-12:
            return float(b)
    return float("nan")


def one_replicate(seeds, seed_idx, rng, langs=("en", "vi"), budget=C_DEFAULT):
    idx = {}
    for l in langs:
        for split in ("dev", "test"):
            n = len(seeds[0][(l, split)]["conf"])
            idx[(l, split)] = rng.integers(0, n, n)
    cal = {l: pooled(seeds, seed_idx, "dev", l, idx[(l, "dev")]) for l in langs}
    test = {l: pooled(seeds, seed_idx, "test", l, idx[(l, "test")]) for l in langs}
    curves = curves_of(cal, langs)
    front = CA.sweep(curves, budget, langs, costs={l: 1.0 for l in langs})
    pf, cf = CA.endpoints(front, K_STAR)
    tau_pf = {l: CA.tau_from_share(cal[l]["conf"], pf["s"][l]) for l in langs}
    tau_cf = {l: CA.tau_from_share(cal[l]["conf"], cf["s"][l]) for l in langs}
    m_pf = CA.measure_on_test(tau_pf, test, langs)
    m_cf = CA.measure_on_test(tau_cf, test, langs)
    return {
        "parity_first_disparity": float(m_pf["D_abs"]),
        "composition_price": float(m_cf["D_abs"] - m_pf["D_abs"]),
        "queue_share": float(m_pf["K"]),
        "capacity": capacity_of(curves, langs),
        "load_test": float(m_pf["load"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "..", "outputs", "killtest"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "killtest", "nested_bootstrap.json"))
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "seed*", "probs.npz")))
    assert len(paths) == 5, f"expected 5 seeds, found {len(paths)}"
    seeds = [load_seed(p) for p in paths]
    langs = ("en", "vi")
    rng = np.random.default_rng(args.seed)

    rows = []
    for b in range(args.B):
        idx = rng.permutation(5)[:5] if b else np.arange(5)
        rows.append(one_replicate(seeds, idx, rng))
        if (b + 1) % 25 == 0:
            print(f"  {b + 1}/{args.B} replicates", flush=True)

    out = {"B": args.B, "seed": int(args.seed), "budget": C_DEFAULT,
           "guard_pp": GUARD * 100, "k_star": K_STAR,
           "seeds": [os.path.basename(os.path.dirname(p)) for p in paths],
           "note": "seeds and items resampled jointly; allocation reselected inside each replicate",
           "draws": rows}
    summ = {}
    for k in ("parity_first_disparity", "composition_price", "queue_share", "capacity", "load_test"):
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        summ[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                   "p2_5": float(np.percentile(v, 2.5)), "p97_5": float(np.percentile(v, 97.5)),
                   "n": int(v.size)}
    out["summary"] = summ
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print("\nsummary (x100 where it is a share/difference in pp):")
    for k, s in summ.items():
        scale = 1.0 if k == "queue_share" else 100.0
        print(f"  {k:26s} mean={s['mean']*scale:8.3f}  sd={s['sd']*scale:7.3f}  "
              f"95% [{s['p2_5']*scale:.3f}, {s['p97_5']*scale:.3f}]  n={s['n']}")
    print("wrote", args.out)


main()
