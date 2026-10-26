#!/usr/bin/env python3
"""Does the monotone-envelope convention change any published number?

The manuscript's Appendix B printed `r_env(s) = min_{s' >= s} r(s')`, while `capra.py:review_risk_curve`
enforces monotonicity with `min_{s' <= s}` on the review-share axis. Three conventions are compared:

    code     min over s' <= s   (what capra.py does: takes the best risk seen at a SMALLER review share,
                                 i.e. in coverage terms the best risk at coverage >= c — the risk a
                                 deployment can always realise by reviewing slightly more)
    raw      no envelope        (the empirical step function itself)
    max_fwd  max over s' >= s   (the reviewer's proposed "non-increasing upper envelope": never
                                 optimistic, always above the raw curve)

Runs the full CAPRA pipeline (sweep -> endpoints -> thresholds -> test measurement) for EVERY
detector root of the English-Vietnamese benchmark, at all three budgets, under all three conventions,
and reports (a) how far the curves themselves differ, (b) whether any reported quantity moves. Also
prints the structural parity capacity, which is computed on CONRA (un-enveloped) curves and therefore
must not move at all.

Run on ubuntu-2 (the prediction artefacts live there):
    ~/miniconda3/envs/toxkgvi-gpu-ft/bin/python tools/envelope_convention_check.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
import conra_parity as CP          # noqa: E402
import capra                       # noqa: E402
import pair_parity as PP           # noqa: E402

ROOTS = ["outputs/killtest", "outputs/pool/xlmr_large", "outputs/pool/specialist_pair",
         "outputs/pool/roberta_en", "outputs/pool/phobert_vi", "outputs/pool/llm_deepseek_flash",
         "outputs/pool/llm_gemini31_flashlite", "outputs/pool/llm_qwen38_flash"]
KINDS = ("code", "raw", "max_fwd")
# Table 8 (flagship, XLM-R base, cost-blind) — the published values the pipeline must reproduce
PUBLISHED = {"0.05": (0.94, 0.998, 4.54, 0.507, 3.59), "0.10": (1.17, 0.970, 4.01, 0.510, 2.83),
             "0.20": (1.13, 0.834, 2.83, 0.507, 1.70)}


def curve_fn(kind):
    """A `review_risk_curve` for the requested envelope convention."""
    def f(conf, err):
        order = np.argsort(-conf, kind="mergesort")
        e = err[order]
        k = np.arange(1, len(e) + 1, dtype=float)
        risk = np.cumsum(e) / k
        s = 1.0 - k / len(e)
        if kind == "code":
            env = np.minimum.accumulate(risk[::-1])[::-1]
        elif kind == "raw":
            env = risk
        elif kind == "max_fwd":
            env = np.maximum.accumulate(risk)
        else:
            raise ValueError(kind)
        o2 = np.argsort(s, kind="mergesort")
        return s[o2], env[o2]
    return f


def load(root, a="en", b="vi"):
    CP.LANGS = [a, b]
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, root, "seed*"))
                  if os.path.exists(os.path.join(d, "probs.npz")))
    if not dirs and os.path.exists(os.path.join(ROOT, root, "probs.npz")):
        dirs = [os.path.join(ROOT, root)]
    d = CP.prep(dirs, "NAT") if (a, b) == ("en", "vi") else PP.build_pair(dirs, a, b)
    return d, dirs


def main():
    a, b = "en", "vi"
    out, curve_dev = {}, {}
    for root in ROOTS:
        try:
            d, dirs = load(root)
        except Exception as exc:                                     # noqa: BLE001
            print(f"skip {root}: {exc}", flush=True)
            continue
        row, ctrl = {}, {}
        for kind in KINDS:
            f = curve_fn(kind)
            ctrl[kind] = {l: f(d["cal"][l]["conf"], d["cal"][l]["err"])[1] for l in (a, b)}
            curves = {l: f(d["cal"][l]["conf"], d["cal"][l]["err"]) for l in (a, b)}
            for C in (0.05, 0.10, 0.20):
                front = capra.sweep(curves, C, (a, b), {a: 1.0, b: 1.0})
                m = capra.summarise(front, 0.5)
                for key in ("parity_first", "composition_first"):
                    pt = m[key]
                    tau = {l: capra.tau_from_share(d["cal"][l]["conf"], pt["s"][l]) for l in (a, b)}
                    pt["test"] = capra.measure_on_test(tau, d["test"], [a, b], costs={a: 1.0, b: 1.0})
                row.setdefault(f"C{C:.2f}", {})[kind] = {
                    "pf_D": m["parity_first"]["test"]["D_abs"] * 100,
                    "pf_K": m["parity_first"]["test"]["K"],
                    "cf_D": m["composition_first"]["test"]["D_abs"] * 100,
                    "cf_K": m["composition_first"]["test"]["K"],
                    "price": (m["composition_first"]["test"]["D_abs"]
                              - m["parity_first"]["test"]["D_abs"]) * 100}
            row.setdefault("cap", {})[kind] = PP.min_parity_capacity(CP.curves_of(d["cal"]))[0]
        curve_dev[root] = max(float(np.max(np.abs(ctrl["code"][l] - ctrl["max_fwd"][l]))) * 100
                              for l in (a, b))
        out[root] = row
        print(f"done {root} ({len(dirs)} seed dir(s))", flush=True)

    print("\n=== pointwise curve deviation |code - max_fwd| (pp), per detector root ===")
    for root, v in curve_dev.items():
        print(f"  {root:38s} {v:.4f}")

    print("\n=== did ANY reported quantity move? (spread max-min over the 3 conventions) ===")
    worst, worst_where, cap_spread = 0.0, None, 0.0
    for root, row in out.items():
        for C in ("C0.05", "C0.10", "C0.20"):
            for key in ("pf_D", "pf_K", "cf_D", "cf_K", "price"):
                vals = [row[C][k][key] for k in KINDS]
                spread = max(vals) - min(vals)
                if spread > worst:
                    worst, worst_where = spread, f"{root} {C} {key}"
        caps = [row["cap"][k] for k in KINDS if row["cap"][k]]
        if len(caps) > 1:
            cap_spread = max(cap_spread, (max(caps) - min(caps)) * 100)
    print(f"  largest spread over every detector x budget x quantity: {worst:.6f} pp  ({worst_where})")
    print(f"  structural parity capacity spread: {cap_spread:.6f} pp")

    flag = out["outputs/killtest"]
    print("\n=== flagship (XLM-R base) vs the published Table 8 row, all three conventions ===")
    for C in ("C0.05", "C0.10", "C0.20"):
        print(f"  -- {C}  published={PUBLISHED[C[1:]]}")
        for kind in KINDS:
            r = flag[C][kind]
            print(f"     {kind:8s} D={r['pf_D']:.3f}/{r['cf_D']:.3f}  K={r['pf_K']:.3f}/"
                  f"{r['cf_K']:.3f}  price={r['price']:.3f}")
    print("\njson:", json.dumps(out))


if __name__ == "__main__":
    main()
