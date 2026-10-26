"""Uncertainty for the structural parity capacity — the paper's headline range (0.5 % to 31.5 % of traffic).

`min_parity_capacity` (pair_parity.py) reads the calibration curves and returns the smallest review budget at
which the two-sided parity constraint becomes satisfiable. It is reported as a point estimate in Table 7 and
quoted in the abstract, but the curves it reads are estimated from a finite calibration split, so the
capacity inherits that sampling error — and the headline of the paper is a RANGE over capacities.

This script bootstraps the calibration split (items resampled within each language, with replacement),
rebuilds the curves, recomputes the capacity, and reports a percentile interval. Replicates whose capacity
is not reached inside the 0-40 % sweep are censored: they are counted, not imputed, and the interval is
reported together with the censored fraction.

Run on ubuntu-2 (the prediction artefacts live there):
  ~/miniconda3/envs/toxkgvi-gpu-ft/bin/python tools/parity_capacity_ci.py --mode envi  --json outputs/killtest/parity_capacity_ci.json
  ~/miniconda3/envs/toxkgvi-gpu-ft/bin/python tools/parity_capacity_ci.py --mode pairs --out-root outputs/textdetox/xlmr_base --targets de,uk,ar,hi,zh,ru --json outputs/textdetox/parity_capacity_ci_xlmr_base.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import conra_parity as CP          # noqa: E402
import pair_parity as PP           # noqa: E402

ENVI = ["outputs/killtest", "outputs/pool/xlmr_large", "outputs/pool/specialist_pair",
        "outputs/pool/llm_deepseek_flash", "outputs/pool/llm_gemini31_flashlite"]
B = 1000


def seed_dirs_of(out_root):
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, out_root, "seed*"))
                  if os.path.exists(os.path.join(d, "probs.npz")))
    if not dirs and os.path.exists(os.path.join(ROOT, out_root, "probs.npz")):
        dirs = [os.path.join(ROOT, out_root)]
    return dirs


def load(out_root, langs):
    dirs = seed_dirs_of(out_root)
    if langs == ("en", "vi"):
        return CP.prep(dirs, "NAT"), dirs
    return PP.build_pair(dirs, langs[0], langs[1]), dirs


def capacity(cal, langs):
    CP.LANGS = list(langs)
    cap, r_nat, r_cap = PP.min_parity_capacity(CP.curves_of(cal))
    return cap, r_nat, r_cap


def bootstrap(cal, langs, b=B, seed=20260924):
    rng = np.random.default_rng(seed)
    base = {}
    for l in langs:
        base[l] = {k: np.asarray(v) for k, v in cal[l].items()}
    caps, censored = [], 0
    for _ in range(b):
        res = {}
        for l in langs:
            n = len(base[l]["conf"])
            idx = rng.integers(0, n, n)
            res[l] = {k: v[idx] for k, v in base[l].items()}
        cap, _, _ = capacity(res, langs)
        if cap is None:
            censored += 1
        else:
            caps.append(float(cap))
    return caps, censored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("envi", "pairs"), default="envi")
    ap.add_argument("--out-root", default="outputs/textdetox/xlmr_base")
    ap.add_argument("--targets", default="de,uk,ar,hi,zh,ru")
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = {"generated": datetime.now().isoformat(timespec="seconds"), "boot": args.boot, "rows": []}
    if args.mode == "envi":
        jobs = [(root, ("en", "vi")) for root in ENVI]
    else:
        jobs = [(args.out_root, ("en", t)) for t in args.targets.split(",") if t.strip()]

    for root, langs in jobs:
        d, dirs = load(root, langs)
        cap, r_nat, r_cap = capacity(d["cal"], langs)
        caps, censored = bootstrap(d["cal"], langs, b=args.boot)
        lo, hi = (float(np.percentile(caps, 2.5)), float(np.percentile(caps, 97.5))) if caps else (None, None)
        row = {"root": root, "pair": list(langs), "n_seeds": len(dirs),
               "capacity": None if cap is None else float(cap),
               "ci": [lo, hi], "censored_replicates": censored,
               "n_ok": len(caps), "r_nat": float(r_nat),
               "median": float(np.median(caps)) if caps else None}
        out["rows"].append(row)
        print(f"{root:36s} {'/'.join(langs):8s} cap="
              f"{'—' if cap is None else f'{cap*100:.1f}%'}"
              f"  CI={'—' if lo is None else f'[{lo*100:.1f}, {hi*100:.1f}]'}"
              f"  censored={censored}/{args.boot}", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(os.path.join(ROOT, args.json)), exist_ok=True)
        with open(os.path.join(ROOT, args.json), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("json ->", args.json, flush=True)


if __name__ == "__main__":
    main()
