"""The three operating modes in one table (review finding R2-15).

NAT   natural items of each language
MAT   prevalence-matched subsets (both languages forced to the same base rate)
PAIR  the machine-translated control: English test items translated into the target language, so the
      SAME content is scored in both languages and a distribution difference cannot explain the gap.

Reports, for each mode, the signed risk gap of the pooled rule and of the parity-first endpoint, plus
the queue share, at C = 5%. Usage: python modes_report.py --out-root outputs/killtest --out notes/modes_NAT.md
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capra as CA          # noqa: E402
import conra_parity as CP   # noqa: E402

C = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/killtest")
    ap.add_argument("--out", default="notes/modes_report.md")
    ap.add_argument("--langs", default="en,vi")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import glob
    CP.LANGS = [s.strip() for s in args.langs.split(",")]
    sd = sorted(d for d in glob.glob(os.path.join(args.out_root, "seed*"))
                if os.path.exists(os.path.join(d, "probs.npz")))
    if not sd and os.path.exists(os.path.join(args.out_root, "probs.npz")):
        sd = [args.out_root]
    rows, lines = [], ["# Operating modes (EN/VI, XLM-R base, C = 5 %)\n",
                       "| mode | n EN | n VI | Δ pooled (pp) | Δ parity-first (pp) | queue VI |",
                       "|---|---|---|---|---|---|"]
    for mode in ("NAT", "MAT", "PAIR"):
        try:
            d = CP.prep(sd, mode)
        except Exception as e:                                     # noqa: BLE001
            lines.append(f"| {mode} | — | — | — | — | mode unavailable: {str(e)[:40]} |")
            continue
        cur = {l: CA.review_risk_curve(d["cal"][l]["conf"], d["cal"][l]["err"]) for l in CP.LANGS}
        cpcur = CP.curves_of(d["cal"])
        out = {}
        for rule in ("GLOBAL", "CONRA"):
            tau = CP.rule_thresholds(rule, C, cpcur)
            t = CA.measure_on_test(tau, d["test"], CP.LANGS)
            out[rule] = t
        n_en, n_vi = len(d["test"]["en"]["conf"]), len(d["test"]["vi"]["conf"])
        lines.append(f"| {mode} | {n_en} | {n_vi} | {out['GLOBAL']['D_signed']*100:+.2f} | "
                     f"{out['CONRA']['D_signed']*100:+.2f} | {out['GLOBAL']['K']:.3f} |")
        rows.append({"mode": mode, "n": [n_en, n_vi],
                     "delta_pooled": float(out["GLOBAL"]["D_signed"]),
                     "delta_parity": float(out["CONRA"]["D_signed"]),
                     "queue_vi": float(out["GLOBAL"]["K"])})
        print(f"  {mode}: d_pooled {out['GLOBAL']['D_signed']*100:+.2f} pp, "
              f"d_parity {out['CONRA']['D_signed']*100:+.2f} pp", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if args.json:
        import json as _j
        open(args.json, "w", encoding="utf-8").write(_j.dumps({"budget": C, "modes": rows}, indent=2))
    print("report ->", args.out)


if __name__ == "__main__":
    main()
