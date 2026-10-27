"""Structural parity capacity on a widened sweep — the answer to "capacities above 40 % are censored".

`min_parity_capacity` (pair_parity.py) scans `FINE`, which by construction ends at 40 % of traffic. Pairs whose
capacity lies above that ceiling are printed as a dash in Table 8 and counted as censored in Table 19: the ceiling
is a reporting convention, not a limit of the search. This script re-runs the same point estimate with the ceiling
lifted (0-80 % and 0-95 %), so the censored cells get a value instead of a dash.

Read-only: `pair_parity.FINE` is patched in memory, no module on disk is modified, no training, no new scores.

Run (ubuntu-2, the prediction artefacts live there):
  ~/miniconda3/envs/toxkgvi-gpu-ft/bin/python tools/parity_capacity_widened.py \
      --roots outputs/textdetox/xlmr_base,outputs/textdetox/xlmr_large,outputs/textdetox/llm_deepseek_flash \
      --targets de,uk,ar,hi,zh,ru --json outputs/textdetox/parity_capacity_widened.json

Validated against Table 19: every uncensored capacity is reproduced at the printed precision under both grids,
so the wider grid changes nothing except the cells that were censored.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

_RUN = os.path.abspath(os.getcwd())
_HERE = os.path.dirname(os.path.abspath(__file__))
# tools/ lives one level under the repo in capra-release, but this file is also run from the repo root in the
# older selective-parity tree; resolve against whichever directory actually holds outputs/.
ROOT = _RUN if os.path.isdir(os.path.join(_RUN, "outputs")) else os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
import conra_parity as CP          # noqa: E402
import pair_parity as PP           # noqa: E402

GRIDS = [(0.40, 80), (0.80, 160), (0.95, 190)]   # same 0.5 % step as the reported sweep


def seed_dirs_of(out_root):
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, out_root, "seed*"))
                  if os.path.exists(os.path.join(d, "probs.npz")))
    if not dirs and os.path.exists(os.path.join(ROOT, out_root, "probs.npz")):
        dirs = [os.path.join(ROOT, out_root)]
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="outputs/textdetox/xlmr_base,outputs/textdetox/xlmr_large,"
                                      "outputs/textdetox/llm_deepseek_flash")
    ap.add_argument("--targets", default="de,uk,ar,hi,zh,ru")
    ap.add_argument("--json", default="outputs/textdetox/parity_capacity_widened.json")
    args = ap.parse_args()

    out = {"generated": datetime.now().isoformat(timespec="seconds"), "grids": GRIDS, "cells": {}}
    for root in [r.strip() for r in args.roots.split(",") if r.strip()]:
        dirs = seed_dirs_of(root)
        if not dirs:
            print("skip %s (no probs.npz)" % root, flush=True)
            continue
        for tgt in [t.strip() for t in args.targets.split(",") if t.strip()]:
            cell = {"out_root": root, "pair": "en/%s" % tgt, "seeds": len(dirs), "capacity_pct": {}}
            for ceil, n in GRIDS:
                PP.FINE = np.linspace(0.005, ceil, n)
                CP.LANGS = ["en", tgt]
                d = PP.build_pair(dirs, "en", tgt)
                cap, r_nat, r_cap = PP.min_parity_capacity(CP.curves_of(d["cal"]))
                cell["capacity_pct"]["%.2f" % ceil] = None if cap is None else round(float(cap) * 100, 2)
                cell["r_nat_pct"] = round(float(r_nat) * 100, 3)
                cell["r_cap_pct"] = None if r_cap is None else round(float(r_cap) * 100, 3)
            out["cells"]["%s|%s" % (os.path.basename(root), tgt)] = cell
            print("%-20s en/%-3s  %s" % (os.path.basename(root), tgt,
                  "  ".join("%s: %s" % (k, v) for k, v in cell["capacity_pct"].items())), flush=True)

    if args.json:
        dst = os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(out, open(dst, "w"), indent=1)
        print("\nwrote", dst)


if __name__ == "__main__":
    main()
